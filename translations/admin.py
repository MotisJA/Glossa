import csv
import json
from typing import Dict, List, Tuple

from django.contrib import admin
from django import forms
from django.urls import reverse, path
from django.shortcuts import redirect, render
from django.contrib.auth.forms import (
    AdminPasswordChangeForm,
    AdminUserCreationForm,
    UserChangeForm,
)
from unfold.admin import ModelAdmin

from .models import GlossaryEntry, CorpusEntry, SystemConfiguration, Translation, CustomUser, EvalRow, EvalRecord


def _normalize_target_language(row: Dict[str, str], fallback_language: str) -> str:
    return (
        (row.get("target_language") or row.get("lang") or fallback_language or "")
        .strip()
        .lower()
    )


def _read_csv_rows(csv_file) -> List[Dict[str, str]]:
    decoded = csv_file.read().decode("utf-8-sig").splitlines()
    reader = csv.DictReader(decoded)
    return list(reader)


def _dedupe_parallel_rows(rows: List[Dict[str, str]], fallback_language: str) -> List[Tuple[str, str, str]]:
    deduped = {}
    for row in rows:
        source_text = (row.get("en") or "").strip()
        translated_text = (row.get("tgt") or "").strip()
        if not source_text or not translated_text:
            continue
        target_language = _normalize_target_language(row, fallback_language)
        deduped[(source_text, translated_text, target_language)] = (
            source_text,
            translated_text,
            target_language,
        )
    return list(deduped.values())


def _parse_jsonl_rows(uploaded_file) -> List[Dict]:
    rows = []
    for idx, raw_line in enumerate(uploaded_file.read().decode("utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise forms.ValidationError(f"第 {idx} 行 JSON 格式错误: {exc}") from exc
    return rows


def _extract_jsonl_payload(jsonl_rows: List[Dict], fallback_language: str):
    terms = []
    memories = []

    for idx, row in enumerate(jsonl_rows, start=1):
        source_text = (row.get("en") or "").strip()
        if not source_text:
            continue

        term_maps = []
        for key in ("proper_terms", "random_terms", "terms"):
            value = row.get(key)
            if isinstance(value, dict):
                term_maps.append(value)

        target_pairs = [
            (lang_key.strip().lower(), (lang_value or "").strip())
            for lang_key, lang_value in row.items()
            if lang_key not in {"en", "proper_terms", "random_terms", "terms", "proper", "random"} and isinstance(lang_value, str)
        ]
        if not target_pairs:
            target_pairs = [(fallback_language, "")]

        for target_language, translated_text in target_pairs:
            if translated_text:
                memories.append(
                    {
                        "en": source_text,
                        "tgt": translated_text,
                        "target_language": target_language or fallback_language,
                        "line_number": idx,
                    }
                )
            for term_map in term_maps:
                for src_term, tgt_term in term_map.items():
                    src = (src_term or "").strip()
                    tgt = (tgt_term or "").strip()
                    if not src or not tgt:
                        continue
                    terms.append(
                        {
                            "en": src,
                            "tgt": tgt,
                            "target_language": target_language or fallback_language,
                            "line_number": idx,
                        }
                    )

    term_dedup = {
        (item["en"], item["tgt"], item["target_language"]): item
        for item in terms
    }
    memory_dedup = {
        (item["en"], item["tgt"], item["target_language"]): item
        for item in memories
    }
    return list(term_dedup.values()), list(memory_dedup.values())


class ImportCSVForm(forms.Form):
    csv_file = forms.FileField(help_text='CSV 需包含列: "en", "tgt"，可选 "target_language"/"lang"')


class ImportJSONLUploadForm(forms.Form):
    jsonl_file = forms.FileField(help_text='JSONL 每行一个 JSON；支持 en + 目标语言键 + proper_terms/random_terms')


class ImportJSONLCommitForm(forms.Form):
    terms_json = forms.CharField(widget=forms.HiddenInput())
    memories_json = forms.CharField(widget=forms.HiddenInput())
    source_filename = forms.CharField(required=False, widget=forms.HiddenInput())

@admin.register(GlossaryEntry)
class GlossaryEntryAdmin(ModelAdmin):
    list_display = ('english_key', 'translated_entry', 'target_language', 'created_at')
    search_fields = ('english_key', 'translated_entry')
    list_filter = ('target_language', 'created_at')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    change_list_template = 'translations/glossary_changelist.html'
    readonly_fields = ('created_at',)
    fieldsets = (
        ("Entry", {
            "fields": ("english_key", "translated_entry", "target_language"),
        }),
        ("Metadata", {
            "fields": ("created_at",),
        }),
    )

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('import-glossary/', self.import_csv, name='glossary_import_csv'),
            path('import-jsonl/', self.import_jsonl, name='glossary_import_jsonl'),
        ]
        return custom_urls + urls

    def import_csv(self, request):
        if not request.user.is_staff:
            return redirect("..")
        form = ImportCSVForm(request.POST or None, request.FILES or None)
        sample_rows = []
        if request.method == "POST" and form.is_valid():
            config = SystemConfiguration.load()
            csv_file = form.cleaned_data["csv_file"]
            rows = _dedupe_parallel_rows(_read_csv_rows(csv_file), config.target_language_code)
            for source_text, translated_text, target_language in rows:
                GlossaryEntry.objects.get_or_create(
                    english_key=source_text,
                    translated_entry=translated_text,
                    target_language=target_language,
                )
            self.message_user(request, f"CSV 导入完成，共处理 {len(rows)} 条术语。")
            return redirect("..")
        if request.method == "POST" and "csv_file" in request.FILES:
            try:
                sample_rows = _read_csv_rows(request.FILES["csv_file"])[:5]
            except Exception:
                sample_rows = []
        payload = {"form": form, "sample_rows": sample_rows, "import_title": "术语 CSV 导入"}
        return render(
            request, "translations/corpusentry_import_csv.html", payload
        )

    def import_jsonl(self, request):
        if not request.user.is_staff:
            return redirect("..")
        config = SystemConfiguration.load()

        if request.method == "POST":
            action = request.POST.get("action")
            if action == "commit":
                commit_form = ImportJSONLCommitForm(request.POST)
                if commit_form.is_valid():
                    terms = json.loads(commit_form.cleaned_data["terms_json"] or "[]")
                    memories = json.loads(commit_form.cleaned_data["memories_json"] or "[]")
                    upserts = 0
                    memory_upserts = 0
                    filename = commit_form.cleaned_data.get("source_filename") or "manual-reviewed-jsonl"
                    for row in terms:
                        source_term = (row.get("en") or "").strip()
                        target_term = (row.get("tgt") or "").strip()
                        target_language = (row.get("target_language") or config.target_language_code).strip().lower()
                        if not source_term or not target_term:
                            continue
                        GlossaryEntry.objects.get_or_create(
                            english_key=source_term,
                            translated_entry=target_term,
                            target_language=target_language,
                        )
                        upserts += 1
                    for row in memories:
                        source_text = (row.get("en") or "").strip()
                        translated_text = (row.get("tgt") or "").strip()
                        target_language = (row.get("target_language") or config.target_language_code).strip().lower()
                        if not source_text or not translated_text:
                            continue
                        CorpusEntry.objects.get_or_create(
                            english_text=source_text,
                            translated_text=translated_text,
                            target_language=target_language,
                            defaults={"source": f"jsonl import from file {filename}"},
                        )
                        memory_upserts += 1
                    self.message_user(request, f"JSONL 导入完成：术语 {upserts} 条，翻译记忆 {memory_upserts} 条。")
                    return redirect("..")
            elif action == "preview":
                upload_form = ImportJSONLUploadForm(request.POST, request.FILES)
                if upload_form.is_valid():
                    rows = _parse_jsonl_rows(upload_form.cleaned_data["jsonl_file"])
                    terms, memories = _extract_jsonl_payload(rows, config.target_language_code)
                    payload = {
                        "upload_form": upload_form,
                        "terms": terms,
                        "memories": memories,
                        "source_filename": upload_form.cleaned_data["jsonl_file"].name,
                    }
                    return render(request, "translations/import_jsonl.html", payload)

        payload = {
            "upload_form": ImportJSONLUploadForm(),
            "terms": [],
            "memories": [],
            "source_filename": "",
        }
        return render(request, "translations/import_jsonl.html", payload)


@admin.register(CorpusEntry)
class CorpusEntryAdmin(ModelAdmin):
    list_display = ('english_text', 'translated_text', 'target_language', 'created_at')
    search_fields = ('english_text', 'translated_text')
    list_filter = ('target_language', 'created_at', 'source')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    change_list_template = 'translations/corpusentry_changelist.html'
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ("Entry", {
            "fields": ("english_text", "translated_text", "target_language", "source"),
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
        }),
    )

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('import-csv/', self.import_csv, name='corpus_entry_import_csv'),
            path('import-jsonl/', self.import_jsonl, name='corpus_entry_import_jsonl'),
        ]
        return custom_urls + urls

    def import_csv(self, request):
        if not request.user.is_staff:
            return redirect("..")
        form = ImportCSVForm(request.POST or None, request.FILES or None)
        sample_rows = []
        if request.method == "POST" and form.is_valid():
            config = SystemConfiguration.load()
            csv_file = form.cleaned_data["csv_file"]
            rows = _dedupe_parallel_rows(_read_csv_rows(csv_file), config.target_language_code)
            for source_text, translated_text, target_language in rows:
                CorpusEntry.objects.get_or_create(
                    english_text=source_text,
                    translated_text=translated_text,
                    target_language=target_language,
                    defaults={"source": f"csv import from file {csv_file.name}"},
                )
            self.message_user(request, f"CSV 导入完成，共处理 {len(rows)} 条翻译记忆。")
            return redirect("..")
        if request.method == "POST" and "csv_file" in request.FILES:
            try:
                sample_rows = _read_csv_rows(request.FILES["csv_file"])[:5]
            except Exception:
                sample_rows = []
        payload = {"form": form, "sample_rows": sample_rows, "import_title": "翻译记忆 CSV 导入"}
        return render(
            request, "translations/corpusentry_import_csv.html", payload
        )

    def import_jsonl(self, request):
        if not request.user.is_staff:
            return redirect("..")
        config = SystemConfiguration.load()

        if request.method == "POST":
            action = request.POST.get("action")
            if action == "commit":
                commit_form = ImportJSONLCommitForm(request.POST)
                if commit_form.is_valid():
                    terms = json.loads(commit_form.cleaned_data["terms_json"] or "[]")
                    memories = json.loads(commit_form.cleaned_data["memories_json"] or "[]")
                    term_upserts = 0
                    upserts = 0
                    filename = commit_form.cleaned_data.get("source_filename") or "manual-reviewed-jsonl"
                    for row in terms:
                        source_term = (row.get("en") or "").strip()
                        target_term = (row.get("tgt") or "").strip()
                        target_language = (row.get("target_language") or config.target_language_code).strip().lower()
                        if not source_term or not target_term:
                            continue
                        GlossaryEntry.objects.get_or_create(
                            english_key=source_term,
                            translated_entry=target_term,
                            target_language=target_language,
                        )
                        term_upserts += 1
                    for row in memories:
                        source_text = (row.get("en") or "").strip()
                        translated_text = (row.get("tgt") or "").strip()
                        target_language = (row.get("target_language") or config.target_language_code).strip().lower()
                        if not source_text or not translated_text:
                            continue
                        CorpusEntry.objects.get_or_create(
                            english_text=source_text,
                            translated_text=translated_text,
                            target_language=target_language,
                            defaults={"source": f"jsonl import from file {filename}"},
                        )
                        upserts += 1
                    self.message_user(request, f"JSONL 导入完成：术语 {term_upserts} 条，翻译记忆 {upserts} 条。")
                    return redirect("..")
            elif action == "preview":
                upload_form = ImportJSONLUploadForm(request.POST, request.FILES)
                if upload_form.is_valid():
                    rows = _parse_jsonl_rows(upload_form.cleaned_data["jsonl_file"])
                    terms, memories = _extract_jsonl_payload(rows, config.target_language_code)
                    payload = {
                        "upload_form": upload_form,
                        "terms": terms,
                        "memories": memories,
                        "source_filename": upload_form.cleaned_data["jsonl_file"].name,
                    }
                    return render(request, "translations/import_jsonl.html", payload)

        payload = {
            "upload_form": ImportJSONLUploadForm(),
            "terms": [],
            "memories": [],
            "source_filename": "",
        }
        return render(request, "translations/import_jsonl.html", payload)

@admin.register(Translation)
class Translation(ModelAdmin):
    list_display = ('source_text', 'final_translation', 'created_by', 'created_at', 'num_TM')
    search_fields = ('source_text', 'final_translation')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    readonly_fields = ['source_text', 'mt_translation', 'final_translation', 'glossary_entries', 'corpus_entries', 'created_by', 'created_at']
    fieldsets = (
        ("Source", {
            "fields": ("source_text", "mt_translation"),
        }),
        ("Final", {
            "fields": ("final_translation",),
        }),
        ("Context", {
            "fields": ("glossary_entries", "corpus_entries", "created_by", "created_at"),
        }),
    )

    def num_TM(self, obj):
        return obj.corpus_entries.count()


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(ModelAdmin):
    readonly_fields = ['created_at', 'updated_at']
    
    def has_add_permission(self, request):
        # Prevent adding if an instance already exists
        return not SystemConfiguration.objects.exists()

    def has_delete_permission(self, request, obj=None):
        # Prevent deletion of the only configuration
        return False

    def changelist_view(self, request, extra_context=None):
        # Redirect to the edit page of the first object
        try:
            config = SystemConfiguration.objects.first()
            if config:
                return redirect(reverse('admin:translations_systemconfiguration_change', args=[config.id]))
        except SystemConfiguration.DoesNotExist:
            pass
        return super().changelist_view(request, extra_context)

@admin.register(CustomUser)
class CustomUserAdmin(ModelAdmin):
    add_form_template = "admin/auth/user/add_form.html"
    change_user_password_template = None
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Permissions", {
            "fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions"),
        }),
        ("Important dates", {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "password1", "password2"),
        }),
    )
    form = UserChangeForm
    add_form = AdminUserCreationForm
    change_password_form = AdminPasswordChangeForm
    list_display = ("email", "is_staff", "is_active", "last_login")
    list_filter = ("is_staff", "is_superuser", "is_active", "groups") 
    search_fields = ("email",)
    ordering = ("-date_joined",)
    filter_horizontal = ("groups", "user_permissions",)
    readonly_fields = ("date_joined", "last_login")

    def get_fieldsets(self, request, obj=None):
        if not obj:
            return self.add_fieldsets
        return super().get_fieldsets(request, obj)

    def get_form(self, request, obj=None, **kwargs):
        defaults = {}
        if obj is None:
            defaults["form"] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)

@admin.register(EvalRow)
class EvalRowAdmin(ModelAdmin):
    list_display = ('en', 'tgt')
    search_fields = ('en', 'tgt')


@admin.register(EvalRecord)
class EvalRecordAdmin(ModelAdmin):
    list_display = (
        "source_text",
        "output_text",
        "target_language",
        "use_term_recognition",
        "use_tm_retrieval",
        "created_at",
    )
    search_fields = ("source_text", "output_text", "target_language", "input_filename", "output_filename")
    list_filter = ("target_language", "use_term_recognition", "use_tm_retrieval", "created_at")
