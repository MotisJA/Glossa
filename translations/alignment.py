import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

from .models import GlossaryEntry

try:
    import jieba.posseg as pseg  # type: ignore
except Exception:
    pseg = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

EN_NOUN_TAGS = {"NN", "NNP", "NNS", "NNPS"}
ZH_NOUN_TAG_PREFIXES = ("n",)
ZH_NOUN_TAGS = {"n", "nt", "nz", "nw", "nr", "ns"}
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
MAX_CANDIDATES_PER_SIDE = 24

try:
    import spacy
except Exception:
    spacy = None
ZH_MODIFIER_TAGS = {"a", "ad", "an", "b", "f", "m", "q"}
ZH_CORE_TAGS = ZH_NOUN_TAGS | {"vn", "eng"}
ZH_FUNCTION_CHARS = set("的了对并后向于和与及或而将把被等所其并且若可由因以从")

_NLP_EN = None
_NLP_EN_LOCK = threading.Lock()
_MODEL = None
_MODEL_LOCK = threading.Lock()
_EMBEDDING_IMPORT_ERR: Optional[str] = None

if SentenceTransformer is None:
    _EMBEDDING_IMPORT_ERR = "sentence_transformers_not_installed_or_failed_to_import"
elif torch is None or F is None:
    _EMBEDDING_IMPORT_ERR = "torch_not_installed_or_failed_to_import"

try:
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_EN_STOP_WORDS
except Exception:
    SPACY_EN_STOP_WORDS = set()


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _is_single_cjk_token(token: str) -> bool:
    return len(token) == 1 and _contains_chinese(token)


def _normalize_phrase(phrase: str) -> str:
    return " ".join((phrase or "").strip().split())


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _normalize_target_for_compare(text: str) -> str:
    cleaned = (text or "").replace("*", "").replace("_", "")
    return re.sub(r"\s+", "", cleaned).lower()


def _simple_en_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text or "")


def _is_en_stopword_term(term: str) -> bool:
    words = _simple_en_tokens(term.lower())
    if not words:
        return False
    if len(words) == 1:
        return words[0] in SPACY_EN_STOP_WORDS
    return all(word in SPACY_EN_STOP_WORDS for word in words)


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_phrase(item)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def _get_en_nlp() -> Optional[Any]:
    global _NLP_EN
    if spacy is None:
        return None
    if _NLP_EN is None:
        with _NLP_EN_LOCK:
            if _NLP_EN is None:
                try:
                    _NLP_EN = spacy.load("en_core_web_sm")
                except Exception:
                    _NLP_EN = None
    return _NLP_EN


def _extract_en_candidates(text: str) -> List[str]:
    nlp = _get_en_nlp()
    candidates: List[str] = []
    if nlp is not None:
        doc = nlp(text)

        for chunk in doc.noun_chunks:
            phrase = _normalize_phrase(chunk.text)
            if phrase:
                candidates.append(phrase)
            chunk_tokens = [tok for tok in chunk if tok.text.strip()]
            if len(chunk_tokens) >= 2:
                tail_phrase = _normalize_phrase(" ".join(tok.text for tok in chunk_tokens[1:]))
                if len(_simple_en_tokens(tail_phrase)) >= 2:
                    candidates.append(tail_phrase)

        for token in doc:
            if token.pos_ == "NOUN" or token.tag_ in EN_NOUN_TAGS:
                phrase = _normalize_phrase(token.text)
                if phrase and len(phrase) > 1:
                    candidates.append(phrase)
    else:
        for token in _simple_en_tokens(text):
            if len(token) > 2:
                candidates.append(token)

    candidates = _dedupe_keep_order(candidates)
    candidates = [c for c in candidates if not _is_en_stopword_term(c)]
    return candidates[:MAX_CANDIDATES_PER_SIDE]


def _extract_zh_candidates(text: str) -> List[str]:
    if pseg is None:
        def split_function_chars(segment: str) -> List[str]:
            separators = (
                "现在正在",
                "并且",
                "此外",
                "现在",
                "正在",
                "参与",
                "出现",
                "进行",
                "提交",
                "提供",
                "申请",
                "主张",
                "症状",
                "期间",
                "之后",
                "向",
                "在",
                "对",
                "并",
                "后",
                "若",
                "可",
                "将",
                "了",
                "的",
                "和",
                "与",
                "及",
                "而",
                "于",
            )
            pattern = "|".join(re.escape(sep) for sep in separators)
            parts = [p.strip() for p in re.split(pattern, segment) if p and p.strip()]
            return [p for p in parts if len(p) > 1]

        trim_prefixes = ("当前", "该", "两名", "出现", "参与", "正在", "现在", "进行", "采集", "提交", "提供", "申请", "主张", "向", "可")
        trim_suffixes = ("现在正在", "正在", "进行", "出现", "症状", "期间", "之后", "后", "提交", "提供", "申请", "主张", "采集")

        def trim_context(segment: str) -> str:
            cleaned = segment.strip()
            if not cleaned:
                return ""
            changed = True
            while changed and cleaned:
                changed = False
                for prefix in trim_prefixes:
                    if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 1:
                        cleaned = cleaned[len(prefix) :]
                        changed = True
                for suffix in trim_suffixes:
                    if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
                        cleaned = cleaned[: -len(suffix)]
                        changed = True
            return cleaned.strip()

        coarse_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*|[\u4e00-\u9fff]+|[^\s]", text or "")
        initial: List[str] = []

        for token in coarse_tokens:
            if _contains_chinese(token):
                for part in split_function_chars(token):
                    trimmed = trim_context(part)
                    if len(trimmed) > 1:
                        initial.append(trimmed)

        for i in range(len(coarse_tokens) - 1):
            left = coarse_tokens[i]
            right = coarse_tokens[i + 1]
            if not re.search(r"[A-Za-z0-9]", left):
                continue
            if not _contains_chinese(right):
                continue
            right_parts = split_function_chars(right)
            if not right_parts:
                continue
            right_trimmed = trim_context(right_parts[0])
            if len(right_trimmed) > 0:
                initial.append(f"{left}{right_trimmed}")

        expanded: List[str] = list(initial)
        for segment in initial:
            if not _contains_chinese(segment):
                continue
            seg_len = len(segment)
            if seg_len < 4:
                continue
            for window in range(min(8, seg_len), 1, -1):
                for i in range(seg_len - window + 1):
                    sub = segment[i : i + window]
                    if len(sub) > 1 and sub[0] not in ZH_FUNCTION_CHARS and sub[-1] not in ZH_FUNCTION_CHARS:
                        expanded.append(sub)

        candidates = _dedupe_keep_order([re.sub(r"\s+", "", c) for c in expanded if len(c) > 1])
        return candidates[:MAX_CANDIDATES_PER_SIDE]

    def is_noun_like(flag: str) -> bool:
        return (
            flag in ZH_CORE_TAGS
            or flag.startswith(ZH_NOUN_TAG_PREFIXES)
            or flag == "vn"
            or flag == "eng"
        )

    def is_meaningful_token(word: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", word or ""))

    tokens: List[Tuple[str, str]] = []
    for word, flag in pseg.cut(text):
        w = (word or "").strip()
        if not w:
            continue
        tokens.append((w, (flag or "").strip()))

    candidates: List[str] = []

    for word, flag in tokens:
        if is_noun_like(flag):
            if _contains_chinese(word) and len(word) <= 1:
                continue
            candidates.append(word)

    span: List[Tuple[str, str]] = []

    def flush_span() -> None:
        nonlocal span
        if not span:
            return

        words = [w for w, _ in span]
        flags = [f for _, f in span]
        noun_positions = [idx for idx, f in enumerate(flags) if is_noun_like(f)]
        if not noun_positions:
            span = []
            return

        full_phrase = "".join(words).strip()
        if len(full_phrase) > 1:
            candidates.append(full_phrase)

        for noun_idx in noun_positions:
            start = noun_idx
            while start > 0 and flags[start - 1] in ZH_MODIFIER_TAGS:
                start -= 1
            phrase = "".join(words[start : noun_idx + 1]).strip()
            if len(phrase) > 1:
                candidates.append(phrase)

        span = []

    for word, flag in tokens:
        if not is_meaningful_token(word):
            flush_span()
            continue
        if flag in ZH_MODIFIER_TAGS or is_noun_like(flag):
            span.append((word, flag))
            if len(span) > 8:
                span = span[-8:]
            continue
        flush_span()

    flush_span()

    candidates = [
        c
        for c in candidates
        if len(c) > 1 and not (_contains_chinese(c) and len(c) == 1)
    ]
    candidates = _dedupe_keep_order(candidates)
    return candidates[:MAX_CANDIDATES_PER_SIDE]


def _get_model() -> Any:
    global _MODEL
    if SentenceTransformer is None:
        return None
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _MODEL


def _encode_texts(texts: Sequence[str]) -> Any:
    model = _get_model()
    if model is None or torch is None or F is None:
        raise RuntimeError("sentence-transformers is unavailable")
    embeddings = model.encode(
        list(texts),
        batch_size=min(16, max(1, len(texts))),
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    return F.normalize(embeddings, p=2, dim=1)


def _greedy_bipartite_match(
    en_terms: Sequence[str],
    zh_terms: Sequence[str],
    similarity: Any,
    threshold: float,
) -> List[Dict[str, Any]]:
    if similarity is None or similarity.numel() == 0:
        return []

    num_zh = similarity.shape[1]
    ranked_flat_indices = torch.argsort(similarity.reshape(-1), descending=True).tolist()
    used_en = set()
    used_zh = set()
    matches: List[Dict[str, Any]] = []

    for flat_idx in ranked_flat_indices:
        en_idx = int(flat_idx // num_zh)
        zh_idx = int(flat_idx % num_zh)
        score = float(similarity[en_idx, zh_idx].item())
        # Since the matrix is globally sorted descending, once below threshold all remaining pairs are too weak.
        if score < threshold:
            break
        if en_idx in used_en or zh_idx in used_zh:
            continue
        used_en.add(int(en_idx))
        used_zh.add(int(zh_idx))
        source = _normalize_phrase(en_terms[int(en_idx)])
        target = _normalize_phrase(zh_terms[int(zh_idx)])
        if not source or not target:
            continue
        matches.append(
            {
                "source": source,
                "target": target,
                "source_term": source,
                "target_term": target,
                "confidence": round(score, 4),
                "method": "chunk_greedy_cosine",
            }
        )
    return matches


def _score_positional_fallback(
    en_idx: int,
    zh_idx: int,
    en_term: str,
    zh_term: str,
    num_en: int,
    num_zh: int,
) -> float:
    if num_en <= 1:
        en_pos = 0.0
    else:
        en_pos = en_idx / float(num_en - 1)
    if num_zh <= 1:
        zh_pos = 0.0
    else:
        zh_pos = zh_idx / float(num_zh - 1)

    # Keep term order roughly aligned when embeddings are unavailable.
    position_score = 1.0 - abs(en_pos - zh_pos)
    score = 0.45 + 0.35 * position_score

    src_norm = _normalize_for_compare(en_term)
    tgt_norm = _normalize_for_compare(zh_term)
    if src_norm and tgt_norm and (src_norm in tgt_norm or tgt_norm in src_norm):
        score += 0.15
    if len(en_term.split()) >= 2:
        score += 0.04
    if _contains_chinese(zh_term) and len(zh_term) >= 3:
        score += 0.03
    return max(0.0, min(0.89, score))


def _fallback_bipartite_match(en_terms: Sequence[str], zh_terms: Sequence[str]) -> List[Dict[str, Any]]:
    if not en_terms or not zh_terms:
        return []

    pairs: List[Tuple[float, int, int]] = []
    num_en = len(en_terms)
    num_zh = len(zh_terms)
    for en_idx, en_term in enumerate(en_terms):
        for zh_idx, zh_term in enumerate(zh_terms):
            score = _score_positional_fallback(
                en_idx=en_idx,
                zh_idx=zh_idx,
                en_term=en_term,
                zh_term=zh_term,
                num_en=num_en,
                num_zh=num_zh,
            )
            pairs.append((score, en_idx, zh_idx))

    pairs.sort(key=lambda x: x[0], reverse=True)
    used_en = set()
    used_zh = set()
    matches: List[Dict[str, Any]] = []
    for score, en_idx, zh_idx in pairs:
        if en_idx in used_en or zh_idx in used_zh:
            continue
        used_en.add(en_idx)
        used_zh.add(zh_idx)
        source = _normalize_phrase(en_terms[en_idx])
        target = _normalize_phrase(zh_terms[zh_idx])
        if not source or not target:
            continue
        matches.append(
            {
                "source": source,
                "target": target,
                "source_term": source,
                "target_term": target,
                "confidence": round(score, 4),
                "method": "chunk_positional_fallback",
            }
        )
    return matches


def _score_adjustment(source: str, target: str) -> float:
    src_norm = _normalize_for_compare(source)
    tgt_norm = _normalize_for_compare(target)
    bonus = 0.0

    # Bonus for transliterated/shared tokens such as product names and abbreviations.
    if src_norm and tgt_norm and (src_norm in tgt_norm or tgt_norm in src_norm):
        bonus += 0.06

    # Slight preference for multi-token noun phrases over isolated generic terms.
    if len(source.split()) >= 2:
        bonus += 0.04

    # CJK technical terms are often compact but meaningful.
    if len(target) >= 4 and _contains_chinese(target):
        bonus += 0.02

    # Penalize suspiciously short generic matches.
    if len(src_norm) <= 3:
        bonus -= 0.06
    if _is_en_stopword_term(source):
        bonus -= 0.08

    return bonus


def _rerank_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reranked: List[Dict[str, Any]] = []
    for item in candidates:
        source = str(item.get("source") or item.get("source_term") or "")
        target = str(item.get("target") or item.get("target_term") or "")
        base = float(item.get("confidence", 0.0))
        adjusted = max(0.0, min(1.0, base + _score_adjustment(source, target)))
        candidate = dict(item)
        candidate["confidence"] = round(adjusted, 4)
        reranked.append(candidate)
    reranked.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return reranked


def _get_glossary_pair_keys(source_text: str) -> set[Tuple[str, str]]:
    pair_keys: set[Tuple[str, str]] = set()
    try:
        entries = GlossaryEntry.get_entries(source_text)
    except Exception:
        return pair_keys

    for entry in entries:
        source = _normalize_for_compare((entry.english_key or "").strip())
        target = _normalize_target_for_compare((entry.translated_entry or "").strip())
        if not source or not target:
            continue
        pair_keys.add((source, target))
    return pair_keys


def _exclude_glossary_candidates(
    candidates: Sequence[Dict[str, Any]],
    glossary_pair_keys: set[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    if not glossary_pair_keys:
        return list(candidates), 0

    kept: List[Dict[str, Any]] = []
    removed = 0
    for item in candidates:
        source = _normalize_for_compare(str(item.get("source") or item.get("source_term") or ""))
        target = _normalize_target_for_compare(str(item.get("target") or item.get("target_term") or ""))
        if source and target and (source, target) in glossary_pair_keys:
            removed += 1
            continue
        kept.append(dict(item))
    return kept, removed


def _dedupe_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in candidates:
        source = _normalize_phrase(str(item.get("source") or item.get("source_term") or ""))
        target = _normalize_phrase(str(item.get("target") or item.get("target_term") or ""))
        if not source or not target:
            continue
        key = (source.lower(), target.lower())
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(item)
        candidate["source"] = source
        candidate["target"] = target
        candidate["source_term"] = source
        candidate["target_term"] = target
        deduped.append(candidate)
    deduped.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return deduped


def suggest_term_pairs(
    source_text: str,
    target_text: str,
    limit: int = 12,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    start = time.time()
    src = (source_text or "").strip()
    tgt = (target_text or "").strip()
    if not src or not tgt:
        return {
            "method": "empty_input",
            "latency_ms": int((time.time() - start) * 1000),
            "candidates": [],
        }

    # Expect EN->ZH, but auto-detect if caller swaps sentence order.
    if _contains_chinese(src) and not _contains_chinese(tgt):
        en_text, zh_text = tgt, src
    else:
        en_text, zh_text = src, tgt

    en_candidates = _extract_en_candidates(en_text)
    zh_candidates = _extract_zh_candidates(zh_text)
    glossary_pair_keys = _get_glossary_pair_keys(en_text)
    if not en_candidates or not zh_candidates:
        deduped: List[Dict[str, Any]] = []
        return {
            "method": "no_candidates",
            "latency_ms": int((time.time() - start) * 1000),
            "source_chunks": en_candidates,
            "target_chunks": zh_candidates,
            "similarity_matrix_shape": [0, 0],
            "similarity_matrix": [],
            "raw_candidates": [],
            "glossary_filtered": 0,
            "candidates": deduped[:limit],
        }

    similarity_matrix: List[List[float]] = []
    similarity_matrix_shape = [len(en_candidates), len(zh_candidates)]
    embedding_error: Optional[str] = _EMBEDDING_IMPORT_ERR
    try:
        all_terms = list(en_candidates) + list(zh_candidates)
        all_vectors = _encode_texts(all_terms)
        en_count = len(en_candidates)
        en_vectors = all_vectors[:en_count]
        zh_vectors = all_vectors[en_count:]
        similarity = torch.matmul(en_vectors, zh_vectors.T)
        similarity_matrix = [
            [round(float(value), 4) for value in row]
            for row in similarity.detach().cpu().tolist()
        ]
        candidates = _greedy_bipartite_match(
            en_terms=en_candidates,
            zh_terms=zh_candidates,
            similarity=similarity,
            threshold=similarity_threshold,
        )
    except Exception:
        candidates = _fallback_bipartite_match(en_candidates, zh_candidates)
        method = "chunk_fallback_no_embedding"
        if embedding_error is None:
            embedding_error = "embedding_runtime_error"
    else:
        method = "multilingual_chunk_greedy"

    candidates = _rerank_candidates(candidates)
    filtered_candidates, glossary_filtered = _exclude_glossary_candidates(candidates, glossary_pair_keys)
    deduped = _dedupe_candidates(filtered_candidates)
    response = {
        "method": method,
        "latency_ms": int((time.time() - start) * 1000),
        "source_chunks": en_candidates,
        "target_chunks": zh_candidates,
        "similarity_matrix_shape": similarity_matrix_shape,
        "similarity_matrix": similarity_matrix,
        "raw_candidates": filtered_candidates,
        "glossary_filtered": glossary_filtered,
        "candidates": deduped[:limit],
    }
    if embedding_error and method != "multilingual_chunk_greedy":
        response["embedding_error"] = embedding_error
    return response
