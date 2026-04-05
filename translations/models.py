import re
import time
import threading
import logging
import sqlite3
import tantivy
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import spacy
from django.db.models import Case, Count, IntegerField, Max, When

from django.db import models
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.contrib.auth.models import AbstractUser
from django.contrib.auth import get_user_model
from django.utils.functional import LazyObject

from sentence_transformers import SentenceTransformer

from .trie import Trie

nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger(__name__)


class CustomUser(AbstractUser):
    title = models.TextField(max_length=100, blank=True)
    email = models.EmailField('Email Address', unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

class GlossaryEntry(models.Model):
    english_key = models.CharField(max_length=200)
    translated_entry = models.CharField(max_length=200)
    target_language = models.CharField(max_length=10, default='zh')
    created_at = models.DateTimeField(auto_now_add=True)
    _TRIE_VERSION_KEY = "translations:glossary_trie_version"
    _trie_lock = threading.Lock()
    _trie_local: Optional[Trie] = None
    _trie_local_version: Optional[int] = None

    class Meta:
        unique_together = ['english_key', 'translated_entry', 'target_language']
        verbose_name_plural = 'glossary entries'
    
    def __str__(self) -> str:
        return self.as_txt()

    def as_txt(self):
        translated_entry_no_markdown = self.translated_entry.replace('_', '').replace('*', '')
        return f"{self.english_key} -> {translated_entry_no_markdown}"

    def as_dict(self) -> dict:
        return {
            'en': self.english_key,
            'tgt': self.translated_entry,
            'target_language': self.target_language,
        }

    @classmethod
    def _get_trie_version(cls) -> int:
        version = cache.get(cls._TRIE_VERSION_KEY)
        if version is None:
            cache.add(cls._TRIE_VERSION_KEY, 1, timeout=None)
            version = cache.get(cls._TRIE_VERSION_KEY, 1)
        return int(version)

    @classmethod
    def _build_trie(cls) -> Trie:
        trie = Trie()
        entries = cls.objects.only('id', 'english_key')
        trie.build(entries)
        return trie

    @classmethod
    def _get_trie(cls) -> Trie:
        cache_version = cls._get_trie_version()
        if cls._trie_local is not None and cls._trie_local_version == cache_version:
            return cls._trie_local

        with cls._trie_lock:
            # Double-check to avoid duplicate rebuild under concurrency.
            cache_version = cls._get_trie_version()
            if cls._trie_local is not None and cls._trie_local_version == cache_version:
                return cls._trie_local

            cls._trie_local = cls._build_trie()
            cls._trie_local_version = cache_version
            return cls._trie_local

    @classmethod
    def invalidate_trie_cache(cls) -> None:
        cls._trie_local = None
        cls._trie_local_version = None
        if cache.add(cls._TRIE_VERSION_KEY, 1, timeout=None):
            return

        try:
            cache.incr(cls._TRIE_VERSION_KEY)
        except ValueError:
            # Some cache backends can lose keys under eviction; reset safely.
            cache.set(cls._TRIE_VERSION_KEY, 1, timeout=None)

    @classmethod
    def get_entries(
        cls,
        sentence: str,
        target_language: str = 'zh',
    ) -> List['GlossaryEntry']:
        if not sentence:
            return []

        trie = cls._get_trie()
        matched_ids = trie.extract_longest_match_ids(sentence)
        if not matched_ids:
            return []

        preserved_order = Case(
            *[When(pk=entry_id, then=pos) for pos, entry_id in enumerate(matched_ids)],
            output_field=IntegerField(),
        )
        return list(
            cls.objects.filter(pk__in=matched_ids, target_language=target_language).order_by(preserved_order)
        )


class CorpusEntry(models.Model):
    english_text = models.TextField()
    translated_text = models.TextField()
    target_language = models.CharField(max_length=10, default='zh')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    source = models.CharField(max_length=200)
    _st_model_name = "all-MiniLM-L6-v2"
    _st_model: Optional[SentenceTransformer] = None
    _st_lock = threading.Lock()
    _dense_lock = threading.Lock()
    _dense_embeddings: Dict[str, Optional[np.ndarray]] = {}
    _dense_entries: Dict[str, List["CorpusEntry"]] = {}
    _dense_signature: Dict[str, Tuple[int, Optional[int], Optional[float]]] = {}

    @property
    def tokens(self):
        text = re.sub(r'[^\w\s]', '', self.english_text)
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    
    def as_txt(self):
        config = SystemConfiguration.load()
        return f"English: {self.english_text}\n{config.target_language_name}: {self.translated_text}"

    def as_dict(self):
        return {
            'en': self.english_text,
            'tgt': self.translated_text,
            'target_language': self.target_language,
        }
    
    def __str__(self) -> str:
        return self.as_txt()

    @classmethod
    def _get_sentence_transformer(cls) -> SentenceTransformer:
        """Lazy-load and cache the dense encoder model."""
        if cls._st_model is not None:
            return cls._st_model

        with cls._st_lock:
            if cls._st_model is None:
                cls._st_model = SentenceTransformer(cls._st_model_name)
        return cls._st_model

    @classmethod
    def _get_dense_signature(cls, target_language: str) -> Tuple[int, Optional[int], Optional[float]]:
        """
        Build a lightweight corpus fingerprint.

        If this signature changes, the in-memory dense index must be rebuilt.
        """
        stats = cls.objects.filter(target_language=target_language).aggregate(
            count=Count("id"),
            max_id=Max("id"),
            max_updated=Max("updated_at"),
        )
        max_updated = stats["max_updated"]
        updated_ts = max_updated.timestamp() if max_updated else None
        return (
            int(stats["count"] or 0),
            int(stats["max_id"]) if stats["max_id"] is not None else None,
            updated_ts,
        )

    @classmethod
    def init_tantivy_index(cls, lines: List['CorpusEntry']) -> tantivy.Index:
        """ Initialize the Tantivy BM25 index for all corpus entries """
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("text", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("translated_text", stored=True)
        schema_builder.add_integer_field("id", stored=True)  # Add ID field
        schema = schema_builder.build()

        index = tantivy.Index(schema)

        writer = index.writer(num_threads=1)
        for i, line in enumerate(lines):
            writer.add_document(tantivy.Document(
                text=line.english_text,
                translated_text=line.translated_text,
                id=i  # Store the index
            ))
        writer.commit()
        writer.wait_merging_threads()

        return index

    @classmethod
    def init_dense_index(
        cls,
        target_language: str = 'zh',
        lines: Optional[Sequence["CorpusEntry"]] = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Initialize or refresh the in-memory dense vector index.

        This method is intentionally lightweight and zero-cost to deploy:
        vectors are kept in process memory and cosine search is done via NumPy.
        """
        signature = cls._get_dense_signature(target_language)
        cached_embeddings = cls._dense_embeddings.get(target_language)
        cached_signature = cls._dense_signature.get(target_language)
        if not force_rebuild and cached_embeddings is not None and cached_signature == signature:
            return

        with cls._dense_lock:
            signature = cls._get_dense_signature(target_language)
            cached_embeddings = cls._dense_embeddings.get(target_language)
            cached_signature = cls._dense_signature.get(target_language)
            if not force_rebuild and cached_embeddings is not None and cached_signature == signature:
                return

            if lines is None:
                lines = list(cls.objects.filter(target_language=target_language).order_by("id"))
            else:
                lines = list(lines)

            if not lines:
                cls._dense_embeddings[target_language] = None
                cls._dense_entries[target_language] = []
                cls._dense_signature[target_language] = signature
                return

            model = cls._get_sentence_transformer()
            embeddings = model.encode(
                [line.english_text for line in lines],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            cls._dense_embeddings[target_language] = embeddings.astype(np.float32, copy=False)
            cls._dense_entries[target_language] = list(lines)
            cls._dense_signature[target_language] = signature

    @classmethod
    def upsert_dense_index_sqlite(
        cls,
        db_path: str = "datafiles/tm_dense.sqlite3",
        extension: str = "sqlite-vec",
        target_language: str = 'zh',
    ) -> None:
        """
        Optional persistence demo using SQLite vector extensions.

        - For `sqlite-vec`, create a regular table with `FLOAT32` vector bytes.
        - For `sqlite-vss`, load extension then create a `vss0` virtual table.
        """
        lines = list(
            cls.objects.filter(target_language=target_language).only("id", "english_text").order_by("id")
        )
        if not lines:
            return

        cls.init_dense_index(target_language=target_language, lines=lines)
        language_embeddings = cls._dense_embeddings.get(target_language)
        if language_embeddings is None:
            return

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus_dense (
                    entry_id INTEGER PRIMARY KEY,
                    english_text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            cur.executemany(
                """
                INSERT INTO corpus_dense (entry_id, english_text, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    english_text=excluded.english_text,
                    embedding=excluded.embedding
                """,
                [
                    (
                        entry.id,
                        entry.english_text,
                        language_embeddings[i].astype(np.float32).tobytes(),
                    )
                    for i, entry in enumerate(lines)
                ],
            )
            conn.commit()
        finally:
            conn.close()

        if extension not in {"sqlite-vec", "sqlite-vss"}:
            logger.warning("Unknown SQLite vector extension: %s", extension)
            return

        logger.info(
            "Dense vectors persisted to %s. You can attach %s for ANN search.",
            db_path,
            extension,
        )

    @classmethod
    def get_top_similar_bm25(
        cls,
        sent: str,
        top_n: int = 10,
        target_language: str = 'zh',
    ) -> List['CorpusEntry']:
        lines = list(cls.objects.filter(target_language=target_language))
        if not lines or not sent:
            return []

        index = cls.init_tantivy_index(lines)
        searcher = index.searcher()
        query_tokens = cls(english_text=sent, translated_text=None).tokens
        if not query_tokens:
            return lines[:top_n]
        query = index.parse_query(' '.join(query_tokens), ["text"])
        search_results = searcher.search(query, top_n).hits
        if not search_results:
            # race condition: the index is not ready yet
            logger.info("Tantivy index not ready, retrying once")
            time.sleep(0.5)
            searcher = index.searcher()
            query_tokens = cls(english_text=sent, translated_text=None).tokens
            query = index.parse_query(' '.join(query_tokens), ["text"])
            search_results = searcher.search(query, top_n).hits

        results = []
        for score, doc_address in search_results:
            doc = searcher.doc(doc_address)
            # Use doc ID to get the relevant CorpusEntry
            doc_id = doc["id"][0]
            results.append(lines[doc_id])

        return results

    @staticmethod
    def _levenshtein_ratio_fallback(text_a: str, text_b: str) -> float:
        """Normalized edit-distance similarity in [0, 1]."""
        if text_a == text_b:
            return 1.0
        len_a = len(text_a)
        len_b = len(text_b)
        if len_a == 0 or len_b == 0:
            return 0.0

        prev = list(range(len_b + 1))
        for i, char_a in enumerate(text_a, start=1):
            current = [i]
            for j, char_b in enumerate(text_b, start=1):
                cost = 0 if char_a == char_b else 1
                current.append(
                    min(
                        prev[j] + 1,
                        current[j - 1] + 1,
                        prev[j - 1] + cost,
                    )
                )
            prev = current

        distance = prev[-1]
        max_len = max(len_a, len_b)
        return 1.0 - (distance / max_len)

    @classmethod
    def _led_similarity(cls, text_a: str, text_b: str) -> float:
        """Use python-Levenshtein when available, else fallback implementation."""
        try:
            import Levenshtein  # type: ignore

            return float(Levenshtein.ratio(text_a, text_b))
        except Exception:
            return cls._levenshtein_ratio_fallback(text_a, text_b)

    @classmethod
    def get_top_similar_dense(
        cls,
        sent: str,
        top_n: int = 20,
        target_language: str = 'zh',
    ) -> List["CorpusEntry"]:
        """Dense retrieval using `all-MiniLM-L6-v2` + cosine similarity."""
        if not sent:
            return []

        cls.init_dense_index(target_language=target_language)
        dense_embeddings = cls._dense_embeddings.get(target_language)
        dense_entries = cls._dense_entries.get(target_language, [])
        if dense_embeddings is None or not dense_entries:
            return []

        model = cls._get_sentence_transformer()
        query_emb = model.encode(
            [sent],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)

        similarities = dense_embeddings @ query_emb
        if top_n >= len(similarities):
            ranked_idx = np.argsort(-similarities)
        else:
            top_idx = np.argpartition(similarities, -top_n)[-top_n:]
            ranked_idx = top_idx[np.argsort(-similarities[top_idx])]

        return [dense_entries[i] for i in ranked_idx]

    @classmethod
    def get_top_similar_hybrid(
        cls,
        sent: str,
        top_k: int = 10,
        recall_n: int = 20,
        alpha: float = 0.1,
        target_language: str = 'zh',
    ) -> List["CorpusEntry"]:
        """
        Dual-recall + RRF + contrastive reranking.

        1) BM25 lexical recall (Top `recall_n`)
        2) Dense cosine recall (Top `recall_n`)
        3) RRF fusion: 1 / (60 + rank_bm25) + 1 / (60 + rank_dense)
        4) Greedy contrastive reranking with LED similarity penalty.
        """
        if not sent or top_k <= 0 or recall_n <= 0:
            return []

        bm25_hits = cls.get_top_similar_bm25(
            sent,
            top_n=recall_n,
            target_language=target_language,
        )
        dense_hits = cls.get_top_similar_dense(
            sent,
            top_n=recall_n,
            target_language=target_language,
        )

        if not bm25_hits and not dense_hits:
            return []

        bm25_rank: Dict[int, int] = {
            entry.id: rank for rank, entry in enumerate(bm25_hits, start=1)
        }
        dense_rank: Dict[int, int] = {
            entry.id: rank for rank, entry in enumerate(dense_hits, start=1)
        }

        candidates: Dict[int, CorpusEntry] = {}
        for entry in bm25_hits + dense_hits:
            candidates[entry.id] = entry

        rrf_scores: List[Tuple[CorpusEntry, float]] = []
        for entry_id, entry in candidates.items():
            score = 0.0
            if entry_id in bm25_rank:
                score += 1.0 / (60 + bm25_rank[entry_id])
            if entry_id in dense_rank:
                score += 1.0 / (60 + dense_rank[entry_id])
            rrf_scores.append((entry, score))

        rrf_scores.sort(key=lambda item: item[1], reverse=True)
        if top_k >= len(rrf_scores):
            return [entry for entry, _ in rrf_scores]

        selected: List[CorpusEntry] = []
        remaining = rrf_scores.copy()

        first_entry, _ = remaining.pop(0)
        selected.append(first_entry)

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for i, (entry, rrf_score) in enumerate(remaining):
                avg_led = mean(
                    cls._led_similarity(entry.english_text, chosen.english_text)
                    for chosen in selected
                )
                contrastive_score = rrf_score - (alpha * avg_led)

                if contrastive_score > best_score:
                    best_score = contrastive_score
                    best_idx = i

            next_entry, _ = remaining.pop(best_idx)
            selected.append(next_entry)

        return selected

    class Meta:
        verbose_name = 'translation memory'
        verbose_name_plural = 'translation memories'
        unique_together = ['english_text', 'translated_text', 'target_language']


class Translation(models.Model):
    source_text = models.TextField()
    mt_translation = models.TextField()
    final_translation = models.TextField()
    glossary_entries = models.ManyToManyField(GlossaryEntry, blank=True, help_text='Relevant glossary entries')
    corpus_entries = models.ManyToManyField(CorpusEntry, blank=True, help_text='Relevant corpus entries')
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)

    def __str__(self) -> str:
        return f"{self.pk}: Translation of '{self.source_text}'"


class SystemConfigurationLazy(LazyObject):
    def _setup(self):
        config = SystemConfiguration.objects.first()
        if config is None:
            config = SystemConfiguration.objects.create()
        self._wrapped = config

    @classmethod
    def load(cls):
        return cls()

class SystemConfiguration(models.Model):
    site_title = models.CharField(max_length=100, default="Glossa: 自适应、交互式的机器翻译系统", blank=False)
    target_language_name = models.CharField(max_length=100, default="Tetun", blank=False)
    target_language_code = models.CharField(max_length=5, default="tet", blank=False, help_text="ISO 639 language code, to be used in the Google Translate API")
    placeholder = models.CharField(max_length=100, default="Stop the wound from bleeding, then do wound dressing.", blank=False, help_text="Placeholder text for the translation input field")
    translation_prompt = models.TextField(
        default="You are a linguist helping to post-edit translations from English to Tetun. "
                "Candidate translations are provided by Google Translate, and you are asked to "
                "correct them, if necessary, using examples and glossary entries. Only use "
                "examples and glossary entries to correct the translations.",
        help_text="System prompt used for the translation post-editing process"
    )
    post_editing_model = models.CharField(
        max_length=100,
        default='gemini/gemini-2.0-flash',
        help_text="The model used for post-editing translations. Choose among https://docs.litellm.ai/docs/providers"
    )
    translation_model = models.CharField(
        max_length=100,
        default='Google Translate',
        help_text="The model used for generating translations. Can be 'Google Translate', or a model from HuggingFace, e.g. 'Helsinki-NLP/opus-mt-en-tdt'"
    )
    num_sentences_retrieved = models.IntegerField(
        default=5,
        help_text="The number of similar sentences retrieved for post-editing"
    )
    dspy_config = models.JSONField(
        default=None,
        blank=True,
        null=True,
        help_text="Optional: Configuration for the dspy model. If present, this takes precedence over the translation_prompt"
    )
    login_required = models.BooleanField(
        default=True,
        help_text="Require users to log in to access the translation interface."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Configuration"
        verbose_name_plural = "System Configuration"

    def save(self, *args, **kwargs):
        if not self.pk and SystemConfiguration.objects.exists():
            raise ValidationError('There can be only one SystemConfiguration instance')
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> 'SystemConfiguration':
        return SystemConfigurationLazy.load()


class EvalRow(models.Model):
    en = models.TextField()
    tgt = models.TextField()

    def as_dict(self):
        return {
            'en': self.en,
            'tgt': self.tgt,
        }


class EvalRecord(models.Model):
    source_text = models.TextField()
    reference_text = models.TextField(blank=True, default="")
    output_text = models.TextField()
    target_language = models.CharField(max_length=10, default="zh")
    use_term_recognition = models.BooleanField(default=True)
    use_tm_retrieval = models.BooleanField(default=True)
    input_filename = models.CharField(max_length=255, blank=True, default="")
    output_filename = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def as_dict(self):
        return {
            "source_text": self.source_text,
            "reference_text": self.reference_text,
            "output_text": self.output_text,
            "target_language": self.target_language,
            "use_term_recognition": self.use_term_recognition,
            "use_tm_retrieval": self.use_tm_retrieval,
            "input_filename": self.input_filename,
            "output_filename": self.output_filename,
            "created_at": self.created_at.isoformat(),
        }
