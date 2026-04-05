import threading
import json
import os
from datetime import datetime

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.http import JsonResponse
from django.http import FileResponse, Http404
from asgiref.sync import sync_to_async
import litellm

from translations.utils import TranslatorGoogle, TranslatorHuggingFace
from translations.alignment import suggest_term_pairs
from .models import GlossaryEntry, CorpusEntry, SystemConfiguration, Translation, EvalRow, EvalRecord

# Initialize these outside the view, using a lock for thread safety
translator = None
lock = threading.Lock()


def _build_domain_instruction(domain_hint: str) -> str:
    if domain_hint:
        return (
            f"Domain hint provided by user: {domain_hint}. "
            "Apply domain-appropriate terminology and style consistently."
        )
    return (
        "First infer the most likely domain from the source text internally, "
        "then translate using that domain's terminology. Do not output the inferred domain."
    )


def initialize_resources():
    global translator
    config = SystemConfiguration.load()
    with lock:
        if translator is None:
            if config.translation_model.lower() == 'google translate':
                translator = TranslatorGoogle()
            else:
                translator = TranslatorHuggingFace(config.translation_model)


def translate_view(request):
    config = SystemConfiguration.load()
    if config.login_required and not request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect('login')
    if request.headers.get('accept') == 'text/event-stream':
        # This is the SSE request
        source_text = request.GET.get('source_text')
        llm_post_edit = request.GET.get('llm_post_edit', request.GET.get('use_mt', '1')).lower() not in ('0', 'false', 'off', 'no')
        nmt_only = request.GET.get('nmt_only', '0').lower() in ('1', 'true', 'on', 'yes')
        domain_hint = (request.GET.get('domain_hint') or '').strip()
        if nmt_only:
            llm_post_edit = False
        user = request.user if request.user.is_authenticated else None
        needs_mt = nmt_only or llm_post_edit
        if needs_mt and translator is None:
            initialize_resources()
        
        async def event_stream():
            # Send MT translation if enabled; otherwise, follow pure LLM path.
            mt_translation = ''
            if needs_mt:
                mt_translation = await sync_to_async(
                    translator.translate,
                    thread_sensitive=True,
                )(source_text)
            yield f"data: {json.dumps({'type': 'mt_translation', 'data': mt_translation})}\n\n"

            glossary_entries = []
            similar_sentences = []
            if not nmt_only:
                # Send glossary entries
                glossary_entries = await sync_to_async(GlossaryEntry.get_entries)(
                    source_text,
                    target_language=config.target_language_code,
                )
                yield f"data: {json.dumps({'type': 'glossary_entries', 'data': [gloss_entry.as_dict() for gloss_entry in glossary_entries]})}\n\n"

                # Send similar sentences
                similar_sentences = await sync_to_async(CorpusEntry.get_top_similar_hybrid)(
                    source_text,
                    top_k=config.num_sentences_retrieved,
                    recall_n=20,
                    target_language=config.target_language_code,
                )
                serialized_similar_sentences = [line.as_dict() for line in similar_sentences]
                yield f"data: {json.dumps({'type': 'similar_sentences', 'data': serialized_similar_sentences})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'glossary_entries', 'data': []})}\n\n"
                yield f"data: {json.dumps({'type': 'similar_sentences', 'data': []})}\n\n"
            
            # Send final translation
            if nmt_only:
                final_translation = mt_translation
                corrections = []
            elif llm_post_edit:
                ai_response = await translator.get_post_edited_translation(
                    source_text,
                    similar_sentences,
                    glossary_entries,
                    domain_hint=domain_hint,
                )
                final_translation = ai_response['final_translation']
                corrections = [c.as_dict() for c in ai_response['corrections']]
            else:
                domain_instruction = _build_domain_instruction(domain_hint)
                user_message = ''
                if glossary_entries:
                    user_message += "<glossary entries>\n"
                    for i, entry in enumerate(glossary_entries):
                        user_message += f"no {i}: {entry.as_txt()}\n"
                    user_message += "</glossary entries>\n\n"

                user_message += "<past translations>\n"
                for sentence in similar_sentences:
                    user_message += (
                        f"English: {sentence.english_text}\n"
                        f"{config.target_language_name}: {sentence.translated_text}\n\n"
                    )
                user_message += "</past translations>\n\n"
                user_message += (
                    f"{domain_instruction}\n\n"
                    "Translate the following text directly. "
                    "Do not include explanations.\n\n"
                    "Text to translate:\n"
                    f"English: {source_text}\n"
                    f"{config.target_language_name}: "
                )

                response = litellm.completion(
                    model=config.post_editing_model,
                    messages=[
                        {'role': 'system', 'content': config.translation_prompt},
                        {'role': 'user', 'content': user_message},
                    ],
                    temperature=0.5,
                )
                final_translation = response.choices[0].message.content.strip()
                corrections = []

            yield f"data: {json.dumps({'type': 'final_translation', 'data': final_translation})}\n\n"

            yield f"data: {json.dumps({'type': 'corrections', 'data': corrections})}\n\n"

            translation = await sync_to_async(Translation.objects.create)(
                source_text=source_text,
                mt_translation=mt_translation,
                final_translation=final_translation,
                created_by=user
            )
            await sync_to_async(translation.glossary_entries.set)(glossary_entries)
            await sync_to_async(translation.corpus_entries.set)(similar_sentences)

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response["X-Accel-Buffering"] = "no"
        return response

    context = {'eval_rows': [row.as_dict() for row in EvalRow.objects.all()]}
    return render(request, 'translations/translate.html', context)


def _is_enabled(flag_value: str) -> bool:
    return str(flag_value).lower() in ("1", "true", "on", "yes")


def _parse_jsonl_file(uploaded_file):
    rows = []
    for idx, raw_line in enumerate(uploaded_file.read().decode("utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {idx} 行 JSON 解析失败: {exc}") from exc
    return rows


def _build_eval_prompt(config, source_text, output_language, glossary_entries, similar_sentences):
    resource_selection_instruction = (
        "Before applying glossary entries or past translations, evaluate each candidate for "
        "domain match and target-language match. Only use entries/examples that are clearly helpful."
    )
    user_message = ""
    if glossary_entries:
        user_message += "<glossary entries>\n"
        for i, entry in enumerate(glossary_entries):
            user_message += f"no {i}: {entry.as_txt()}\n"
        user_message += "</glossary entries>\n\n"
    if similar_sentences:
        user_message += "<past translations>\n"
        for sentence in similar_sentences:
            user_message += f"English: {sentence.english_text}\n{output_language}: {sentence.translated_text}\n\n"
        user_message += "</past translations>\n\n"

    user_message += (
        "Translate the following text directly. Do not include explanations.\n\n"
        f"English: {source_text}\n"
        f"{output_language}: "
    )
    return [
        {"role": "system", "content": f"{config.translation_prompt}\n\n{resource_selection_instruction}"},
        {"role": "user", "content": user_message},
    ]


@login_required
def eval_module_view(request):
    config = SystemConfiguration.load()
    context = {
        "records": EvalRecord.objects.order_by("-created_at")[:200],
        "latest_output_file": "",
        "error": "",
    }

    if request.method != "POST":
        return render(request, "translations/eval_module.html", context)

    uploaded_file = request.FILES.get("jsonl_file")
    output_language = (request.POST.get("output_language") or "").strip().lower()
    use_term_recognition = _is_enabled(request.POST.get("use_term_recognition"))
    use_tm_retrieval = _is_enabled(request.POST.get("use_tm_retrieval"))

    if not uploaded_file:
        context["error"] = "请上传 JSONL 文件。"
        return render(request, "translations/eval_module.html", context)
    if not output_language:
        context["error"] = "请选择输出语言标签（如 de）。"
        return render(request, "translations/eval_module.html", context)
    if output_language != (config.target_language_code or "").strip().lower():
        context["error"] = (
            f"当前系统输出语言为 `{config.target_language_code}`，"
            f"请先在系统配置中切换到 `{output_language}` 后再评估。"
        )
        return render(request, "translations/eval_module.html", context)

    try:
        rows = _parse_jsonl_file(uploaded_file)
    except ValueError as exc:
        context["error"] = str(exc)
        return render(request, "translations/eval_module.html", context)

    missing_output = [idx for idx, row in enumerate(rows, start=1) if output_language not in row]
    if missing_output:
        context["error"] = f"输出语言与文件不一致：第 {missing_output[0]} 行缺少 `{output_language}` 字段。"
        return render(request, "translations/eval_module.html", context)

    if translator is None:
        initialize_resources()

    output_dir = "datafiles/output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{output_language}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    output_lines = []
    for row in rows:
        source_text = (row.get("en") or "").strip()
        reference_text = (row.get(output_language) or "").strip()
        if not source_text:
            continue

        glossary_entries = []
        similar_sentences = []
        if use_term_recognition:
            glossary_entries = GlossaryEntry.get_entries(source_text, target_language=output_language)
        if use_tm_retrieval:
            similar_sentences = CorpusEntry.get_top_similar_hybrid(
                source_text,
                top_k=config.num_sentences_retrieved,
                recall_n=20,
                target_language=output_language,
            )

        mt_translation = translator.translate(source_text)
        output_text = mt_translation
        if glossary_entries or similar_sentences:
            try:
                messages = _build_eval_prompt(
                    config,
                    source_text=source_text,
                    output_language=output_language,
                    glossary_entries=glossary_entries,
                    similar_sentences=similar_sentences,
                )
                response = litellm.completion(
                    model=config.post_editing_model,
                    messages=messages,
                    temperature=0.5,
                )
                output_text = response.choices[0].message.content.strip()
            except Exception:
                output_text = mt_translation

        output_row = {"en": source_text, output_language: output_text}
        if isinstance(row.get("terms"), dict):
            output_row["terms"] = row["terms"]
        output_lines.append(json.dumps(output_row, ensure_ascii=False))

        EvalRecord.objects.create(
            source_text=source_text,
            reference_text=reference_text,
            output_text=output_text,
            target_language=output_language,
            use_term_recognition=use_term_recognition,
            use_tm_retrieval=use_tm_retrieval,
            input_filename=uploaded_file.name,
            output_filename=output_filename,
        )

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(output_lines) + ("\n" if output_lines else ""))

    context["records"] = EvalRecord.objects.order_by("-created_at")[:200]
    context["latest_output_file"] = output_filename
    return render(request, "translations/eval_module.html", context)


@login_required
def download_eval_output(request, filename):
    base_dir = os.path.abspath("datafiles/output")
    filepath = os.path.abspath(os.path.join(base_dir, filename))
    if not filepath.startswith(base_dir) or not os.path.exists(filepath):
        raise Http404("文件不存在")
    return FileResponse(open(filepath, "rb"), as_attachment=True, filename=filename)


@login_required
def create_corpus_entry(request):
    if not request.user.has_perm('translations.add_corpusentry'):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    config = SystemConfiguration.load()
    if request.method == 'POST':
        source_text = request.POST.get('src')
        translation_text = request.POST.get('translation')
        raw_term_pairs = request.POST.get('term_pairs')
        corpus_entry = CorpusEntry.objects.update_or_create(
            english_text=source_text,
            translated_text=translation_text,
            target_language=config.target_language_code,
            defaults={'source': 'user-created'},
        )[0]
        added_terms = 0
        skipped_terms = 0

        can_add_glossary = request.user.has_perm('translations.add_glossaryentry')
        if raw_term_pairs and can_add_glossary:
            try:
                term_pairs = json.loads(raw_term_pairs)
            except (TypeError, json.JSONDecodeError):
                term_pairs = []
            for pair in term_pairs:
                en_term = (pair.get('source_term') or pair.get('source') or '').strip()
                tgt_term = (pair.get('target_term') or pair.get('target') or '').strip()
                if not en_term or not tgt_term:
                    skipped_terms += 1
                    continue
                _, created = GlossaryEntry.objects.get_or_create(
                    english_key=en_term,
                    translated_entry=tgt_term,
                    target_language=config.target_language_code,
                )
                if created:
                    added_terms += 1
                else:
                    skipped_terms += 1
        elif raw_term_pairs:
            try:
                skipped_terms = len(json.loads(raw_term_pairs))
            except (TypeError, json.JSONDecodeError):
                skipped_terms = 0

        return JsonResponse(
            {
                'id': corpus_entry.id,
                'source_text': corpus_entry.english_text,
                'translation_text': corpus_entry.translated_text,
                'added_terms': added_terms,
                'skipped_terms': skipped_terms,
            }
        )
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def suggest_terms(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    source_text = (request.POST.get('src') or '').strip()
    translation_text = (request.POST.get('translation') or '').strip()
    if not source_text or not translation_text:
        return JsonResponse({'error': 'src and translation are required'}, status=400)

    result = suggest_term_pairs(source_text, translation_text, limit=16)
    return JsonResponse(result)
