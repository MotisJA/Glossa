import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import stanza
from stanza.pipeline.core import UnsupportedProcessorError
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent



def _load_termbased_metric_class():
    module_path = SCRIPT_DIR / "term-consistency" / "termbasedmetric.py"
    spec = importlib.util.spec_from_file_location("termbasedmetric_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load termbasedmetric module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TermBasedMetric


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no} JSON parse failed: {exc}") from exc
    return rows


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _compute_bleu(hyps: list[str], refs: list[str]) -> float:
    metric = BLEU(max_ngram_order=4, tokenize="13a")
    return float(metric.corpus_score(hyps, [refs]).score)


def _compute_chrf(hyps: list[str], refs: list[str]) -> float:
    metric = CHRF(char_order=6, word_order=2)
    return float(metric.corpus_score(hyps, [refs]).score)


def _build_stanza_pipeline(lang: str):
    processors_with_mwt = "tokenize,mwt,pos,lemma"
    try:
        return stanza.Pipeline(lang=lang, processors=processors_with_mwt)
    except UnsupportedProcessorError:
        # Some languages (e.g. ru) do not provide MWT and must skip it.
        return stanza.Pipeline(lang=lang, processors="tokenize,pos,lemma")


def _lemmatize_texts(texts: list[str], pipeline, desc: str | None = None) -> list[str]:
    out = []
    iterator = tqdm(texts, desc=desc, unit="sent") if desc else texts
    for text in iterator:
        doc = pipeline(text)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        out.append("|||".join(lemmas).lower())
    return out


def _get_lemmatized_dict_lists(
    tgt_lang: str,
    dict_lists: list[list[tuple[str, str]]],
    nlp_pipelines: dict,
    show_progress: bool = False,
) -> list[list[tuple[str, str]]]:
    lemmatized = []
    iterator = tqdm(dict_lists, desc="Lemmatizing term dictionaries", unit="sent") if show_progress else dict_lists
    for dict_list in iterator:
        lemmatized_dict_list = []
        for src_term, tgt_term in dict_list:
            src_lemmas = "|||".join(
                [word.lemma for sent in nlp_pipelines["en"](src_term).sentences for word in sent.words]
            ).lower()
            tgt_lemmas = "|||".join(
                [word.lemma for sent in nlp_pipelines[tgt_lang](tgt_term).sentences for word in sent.words]
            ).lower()
            lemmatized_dict_list.append((src_lemmas, tgt_lemmas))
        lemmatized.append(lemmatized_dict_list)
    return lemmatized


def _compute_term_success_rate(
    src_texts: list[str],
    hyp_texts: list[str],
    dict_rows: list[dict[str, str]],
    nlp_pipelines: dict,
    tgt_lang: str,
    show_progress: bool = False,
) -> float:
    dict_lists = [list(d.items()) for d in dict_rows]
    lemmatized_dict_lists = _get_lemmatized_dict_lists(
        tgt_lang,
        dict_lists,
        nlp_pipelines,
        show_progress=show_progress,
    )
    lemmatized_src = _lemmatize_texts(
        src_texts,
        nlp_pipelines["en"],
        desc="Lemmatizing source" if show_progress else None,
    )
    lemmatized_hyp = _lemmatize_texts(
        hyp_texts,
        nlp_pipelines[tgt_lang],
        desc=f"Lemmatizing hypothesis ({tgt_lang})" if show_progress else None,
    )

    total_terms = 0
    matched_terms = 0
    rows = zip(
        lemmatized_src,
        lemmatized_hyp,
        lemmatized_dict_lists,
        src_texts,
        hyp_texts,
        dict_lists,
    )
    if show_progress:
        rows = tqdm(rows, total=len(src_texts), desc="Matching terms", unit="sent")

    for lem_src, lem_hyp, lem_dict_list, src, hyp, dict_list in rows:
        for (lem_k, lem_v), (k, v) in zip(lem_dict_list, dict_list):
            if lem_k in lem_src or k.lower() in src.lower():
                total_terms += 1
                if lem_v in lem_hyp or v.lower() in hyp.lower():
                    matched_terms += 1

    if total_terms == 0:
        raise ValueError("No source terms found when computing term success rate.")
    return matched_terms / total_terms


def _load_fewshot_prompt(lang_src: str, lang_tgt: str) -> str:
    fewshot_path = SCRIPT_DIR / "term-consistency" / "fewshot" / f"{lang_src}-{lang_tgt}-20.txt"
    if fewshot_path.exists():
        with fewshot_path.open("r", encoding="utf-8") as f:
            return f.read()
    return (
        "You are an alignment assistant. Given a source sentence, one source term, and "
        "a target sentence, output ONLY the exact target-language span that translates "
        "the source term. If absent, output an empty string."
    )


def _init_alignment_cache(
    cache_file: Path,
    expected_size: int,
    system_name: str,
    tgt_lang: str,
    mode: str,
    llm_model: str | None,
    resume: bool,
) -> dict:
    base = {
        "version": 1,
        "system_name": system_name,
        "target_lang": tgt_lang,
        "mode": mode,
        "llm_model": llm_model,
        "size": expected_size,
        "rows": [None] * expected_size,
        "error_count": 0,
        "completed": False,
    }
    if not resume or not cache_file.exists():
        return base

    with cache_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    same_shape = loaded.get("size") == expected_size
    same_meta = (
        loaded.get("system_name") == system_name
        and loaded.get("target_lang") == tgt_lang
        and loaded.get("mode") == mode
        and loaded.get("llm_model") == llm_model
    )
    if not (same_shape and same_meta):
        return base

    rows = loaded.get("rows", [])
    if not isinstance(rows, list) or len(rows) != expected_size:
        return base

    return {
        **base,
        "rows": rows,
        "error_count": int(loaded.get("error_count", 0)),
        "completed": bool(loaded.get("completed", False)),
    }


def _align_with_checkpoint(
    tbm,
    cache_file: Path,
    checkpoint_every: int,
    resume: bool,
    show_progress: bool,
) -> tuple[int, int]:
    fewshots_prompt = _load_fewshot_prompt(tbm.lang_src, tbm.lang_tgt)
    n_rows = len(tbm.bitext_df)
    cache = _init_alignment_cache(
        cache_file=cache_file,
        expected_size=n_rows,
        system_name=getattr(tbm, "system_name", "unknown"),
        tgt_lang=tbm.lang_tgt,
        mode=getattr(tbm, "mode", "proper"),
        llm_model=getattr(tbm, "llm_model", None),
        resume=resume,
    )

    reused = 0
    failed = 0

    iterator = range(n_rows)
    if show_progress:
        iterator = tqdm(iterator, total=n_rows, desc="LLM terminology alignment", unit="sent")

    dirty_count = 0
    for idx in iterator:
        cached_row = cache["rows"][idx]
        if isinstance(cached_row, dict) and "alg_terms" in cached_row and "over_aligned" in cached_row:
            reused += 1
            if cached_row.get("error"):
                failed += 1
            continue

        row = tbm.bitext_df.iloc[idx]
        try:
            alg_terms, over_aligned = tbm._llm_align_one_segment(
                row.src_raw,
                row.src_terms,
                row.mt_raw,
                fewshots_prompt,
                row.terms,
            )
            record = {
                "alg_terms": alg_terms,
                "over_aligned": bool(over_aligned),
                "error": None,
            }
        except Exception as exc:
            # Do not abort the full run when one sentence pair fails.
            record = {
                "alg_terms": {term: tbm.failed_dummy for term in row.src_terms},
                "over_aligned": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed += 1

        cache["rows"][idx] = record
        dirty_count += 1

        if dirty_count >= checkpoint_every:
            cache["error_count"] = failed
            _save_json_atomic(cache_file, cache)
            dirty_count = 0

    cache["error_count"] = failed
    cache["completed"] = True
    _save_json_atomic(cache_file, cache)

    tbm.bitext_df["alg_terms"] = [
        row["alg_terms"] if isinstance(row, dict) and "alg_terms" in row else {} for row in cache["rows"]
    ]
    tbm.bitext_df["over_aligned"] = [
        bool(row["over_aligned"]) if isinstance(row, dict) and "over_aligned" in row else False
        for row in cache["rows"]
    ]

    return reused, failed


def _compute_consistency(
    src_texts: list[str],
    hyp_texts: list[str],
    proper_terms: list[dict[str, str]],
    average: str,
    llm_model: str | None,
    tgt_lang: str,
    system_name: str,
    mode: str,
    alignment_checkpoint_file: Path,
    checkpoint_every: int,
    resume_alignment: bool,
    show_progress: bool,
) -> tuple[float, float, dict]:
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    if llm_model:
        os.environ["TERMINOLOGY_CONSISTENCY_MODEL"] = llm_model

    TermBasedMetric = _load_termbased_metric_class()
    tbm = TermBasedMetric("en", tgt_lang, "predefined", "llm")
    tbm.system_name = system_name
    tbm.mode = mode
    tbm.bitext_df = pd.DataFrame(
        {
            "src_raw": src_texts,
            "mt_raw": hyp_texts,
            "terms": proper_terms,
        }
    )
    tbm.bitext_df["src_terms"] = [list(d.keys()) for d in proper_terms]

    reused, failed = _align_with_checkpoint(
        tbm=tbm,
        cache_file=alignment_checkpoint_file,
        checkpoint_every=max(1, checkpoint_every),
        resume=resume_alignment,
        show_progress=show_progress,
    )

    tbm.assign_pseudoreferences("frequent")
    consistency_frequent, _ = tbm.compute_metric(average)

    tbm.assign_pseudoreferences("predefined")
    consistency_predefined, _ = tbm.compute_metric(average)

    return float(consistency_frequent), float(consistency_predefined), {
        "alignment_cache_file": str(alignment_checkpoint_file),
        "alignment_reused": reused,
        "alignment_failed_rows": failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one WMT25 Track1 system output JSONL and write a score JSON with "
            "official-style metrics."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="System output JSONL path. Each row must include fields: en, <target-lang>.",
    )
    parser.add_argument(
        "--target-lang",
        default="de",
        help="Target language code, e.g. de/es/ru. Default: de.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Reference JSONL path. Default: datafiles/wmt25/track1/reference/full_data.en<target-lang>.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["proper", "random", "noterm"],
        default="proper",
        help="Terminology mode of input system output.",
    )
    parser.add_argument(
        "--system-name",
        default=None,
        help="System name in output JSON. Default: derived from input filename stem.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--consistency-average",
        choices=["micro", "macro"],
        default="micro",
        help="Average mode for term consistency metrics.",
    )
    parser.add_argument(
        "--consistency-model",
        default=None,
        help="Google AI model for term-consistency alignment, e.g. gemini/gemini-2.0-flash.",
    )
    parser.add_argument(
        "--alignment-checkpoint-file",
        default=None,
        help=(
            "Path to alignment checkpoint JSON. "
            "Default: <output-file>.alignment_checkpoint.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush alignment checkpoint every N newly processed rows. Default: 10.",
    )
    parser.add_argument(
        "--no-resume-alignment",
        action="store_true",
        help="Do not resume from existing alignment checkpoint.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )

    args = parser.parse_args()
    show_progress = not args.no_progress

    tgt_lang = args.target_lang
    input_file = Path(args.input_file)
    default_reference_file = Path(f"datafiles/wmt25/track1/reference/full_data.en{tgt_lang}.jsonl")
    default_output_file = Path(f"datafiles/wmt25/track1/reference/track1_score_{tgt_lang}_dict.json")
    reference_file = Path(args.reference_file) if args.reference_file else default_reference_file
    output_file = Path(args.output_file) if args.output_file else default_output_file

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    system_name = args.system_name or input_file.stem

    reference_rows = _read_jsonl(reference_file)
    hyp_rows = _read_jsonl(input_file)

    if len(reference_rows) != len(hyp_rows):
        raise ValueError(
            f"Input/reference size mismatch: input={len(hyp_rows)} reference={len(reference_rows)}"
        )

    src_texts = [str(r["en"]).strip() for r in reference_rows]
    ref_texts = [str(r[tgt_lang]).strip() for r in reference_rows]
    hyp_texts = [str(r.get(tgt_lang, "")).strip() for r in hyp_rows]

    if any(not x for x in hyp_texts):
        raise ValueError(
            f"At least one line in input JSONL misses `{tgt_lang}` or has empty value."
        )

    nlp_pipelines = {
        "en": _build_stanza_pipeline("en"),
        tgt_lang: _build_stanza_pipeline(tgt_lang),
    }

    proper_dict_rows = [row["proper"] for row in reference_rows]
    random_dict_rows = [row["random"] for row in reference_rows]

    bleu4 = _compute_bleu(hyp_texts, ref_texts)
    chrf2pp = _compute_chrf(hyp_texts, ref_texts)
    proper_term_success_rate = _compute_term_success_rate(
        src_texts,
        hyp_texts,
        proper_dict_rows,
        nlp_pipelines,
        tgt_lang,
        show_progress=show_progress,
    )
    random_term_success_rate = _compute_term_success_rate(
        src_texts,
        hyp_texts,
        random_dict_rows,
        nlp_pipelines,
        tgt_lang,
        show_progress=show_progress,
    )

    if args.alignment_checkpoint_file:
        alignment_checkpoint_file = Path(args.alignment_checkpoint_file)
    else:
        alignment_checkpoint_file = output_file.with_suffix(output_file.suffix + ".alignment_checkpoint.json")

    consistency_frequent, consistency_predefined, consistency_debug = _compute_consistency(
        src_texts=src_texts,
        hyp_texts=hyp_texts,
        proper_terms=proper_dict_rows,
        average=args.consistency_average,
        llm_model=args.consistency_model,
        tgt_lang=tgt_lang,
        system_name=system_name,
        mode=args.mode,
        alignment_checkpoint_file=alignment_checkpoint_file,
        checkpoint_every=args.checkpoint_every,
        resume_alignment=not args.no_resume_alignment,
        show_progress=show_progress,
    )

    result = {
        tgt_lang: {
            args.mode: {
                system_name: {
                    "bleu4": bleu4,
                    "chrf2++": chrf2pp,
                    "proper_term_success_rate": proper_term_success_rate,
                    "random_term_success_rate": random_term_success_rate,
                    "consistency_frequent": consistency_frequent,
                    "consistency_predefined": consistency_predefined,
                }
            }
        }
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output": str(output_file),
                "system_name": system_name,
                "alignment_cache": consistency_debug["alignment_cache_file"],
                "alignment_reused": consistency_debug["alignment_reused"],
                "alignment_failed_rows": consistency_debug["alignment_failed_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
