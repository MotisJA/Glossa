import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import stanza
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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


def _compute_bleu(hyps: list[str], refs: list[str]) -> float:
    metric = BLEU(max_ngram_order=4, tokenize="13a")
    return float(metric.corpus_score(hyps, [refs]).score)


def _compute_chrf(hyps: list[str], refs: list[str]) -> float:
    metric = CHRF(char_order=6, word_order=2)
    return float(metric.corpus_score(hyps, [refs]).score)


def _lemmatize_texts(texts: list[str], pipeline) -> list[str]:
    out = []
    for text in texts:
        doc = pipeline(text)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        out.append("|||".join(lemmas).lower())
    return out


def _get_lemmatized_dict_lists(
    tgt_lang: str,
    dict_lists: list[list[tuple[str, str]]],
    nlp_pipelines: dict,
) -> list[list[tuple[str, str]]]:
    lemmatized = []
    for dict_list in dict_lists:
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
) -> float:
    dict_lists = [list(d.items()) for d in dict_rows]
    lemmatized_dict_lists = _get_lemmatized_dict_lists(tgt_lang, dict_lists, nlp_pipelines)
    lemmatized_src = _lemmatize_texts(src_texts, nlp_pipelines["en"])
    lemmatized_hyp = _lemmatize_texts(hyp_texts, nlp_pipelines[tgt_lang])

    total_terms = 0
    matched_terms = 0
    for lem_src, lem_hyp, lem_dict_list, src, hyp, dict_list in zip(
        lemmatized_src,
        lemmatized_hyp,
        lemmatized_dict_lists,
        src_texts,
        hyp_texts,
        dict_lists,
    ):
        for (lem_k, lem_v), (k, v) in zip(lem_dict_list, dict_list):
            if lem_k in lem_src or k.lower() in src.lower():
                total_terms += 1
                if lem_v in lem_hyp or v.lower() in hyp.lower():
                    matched_terms += 1

    if total_terms == 0:
        raise ValueError("No source terms found when computing term success rate.")
    return matched_terms / total_terms


def _compute_consistency(
    src_texts: list[str],
    hyp_texts: list[str],
    proper_terms: list[dict[str, str]],
    average: str,
    llm_model: str | None,
) -> tuple[float, float]:
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    if llm_model:
        os.environ["TERMINOLOGY_CONSISTENCY_MODEL"] = llm_model

    TermBasedMetric = _load_termbased_metric_class()
    tbm = TermBasedMetric("en", "de", "predefined", "llm")
    tbm.bitext_df = pd.DataFrame(
        {
            "src_raw": src_texts,
            "mt_raw": hyp_texts,
            "terms": proper_terms,
        }
    )
    tbm.bitext_df["src_terms"] = [list(d.keys()) for d in proper_terms]
    tbm.align(test=False)

    tbm.assign_pseudoreferences("frequent")
    consistency_frequent, _ = tbm.compute_metric(average)

    tbm.assign_pseudoreferences("predefined")
    consistency_predefined, _ = tbm.compute_metric(average)

    return float(consistency_frequent), float(consistency_predefined)


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
        help="System output JSONL path. Each row must include fields: en, de.",
    )
    parser.add_argument(
        "--reference-file",
        default="datafiles/wmt25/track1/reference/full_data.ende.jsonl",
        help="Reference JSONL path (full_data.ende.jsonl).",
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
        default="datafiles/wmt25/track1/reference/track1_score_de_dict.json",
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

    args = parser.parse_args()
    input_file = Path(args.input_file)
    reference_file = Path(args.reference_file)
    output_file = Path(args.output_file)

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
    ref_texts = [str(r["de"]).strip() for r in reference_rows]
    hyp_texts = [str(r.get("de", "")).strip() for r in hyp_rows]

    if any(not x for x in hyp_texts):
        raise ValueError("At least one line in input JSONL misses `de` or has empty value.")

    nlp_pipelines = {
        "en": stanza.Pipeline(lang="en", processors="tokenize,mwt,pos,lemma"),
        "de": stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma"),
    }

    proper_dict_rows = [row["proper"] for row in reference_rows]
    random_dict_rows = [row["random"] for row in reference_rows]

    bleu4 = _compute_bleu(hyp_texts, ref_texts)
    chrf2pp = _compute_chrf(hyp_texts, ref_texts)
    proper_term_success_rate = _compute_term_success_rate(
        src_texts, hyp_texts, proper_dict_rows, nlp_pipelines, "de"
    )
    random_term_success_rate = _compute_term_success_rate(
        src_texts, hyp_texts, random_dict_rows, nlp_pipelines, "de"
    )
    consistency_frequent, consistency_predefined = _compute_consistency(
        src_texts=src_texts,
        hyp_texts=hyp_texts,
        proper_terms=proper_dict_rows,
        average=args.consistency_average,
        llm_model=args.consistency_model,
    )

    result = {
        "de": {
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

    print(json.dumps({"output": str(output_file), "system_name": system_name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
