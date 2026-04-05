import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import django

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tulun.settings")
    django.setup()


def _parse_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {idx} JSON parse failed: {exc}") from exc


def _load_input_rows(input_file: Path) -> List[dict]:
    rows = []
    for row in _parse_jsonl(input_file):
        source = str(row.get("en", "")).strip()
        if not source:
            continue
        terms = row.get("terms")
        if not isinstance(terms, dict):
            terms = {}
        rows.append({"en": source, "terms": {str(k): str(v) for k, v in terms.items()}})
    return rows


def _load_ref_map(ref_file: Path, target_language: str) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for row in _parse_jsonl(ref_file):
        source = str(row.get("en", "")).strip()
        target = str(row.get(target_language, "")).strip()
        if source and target:
            refs[source] = target
    if not refs:
        raise ValueError(f"No refs found in {ref_file} for target={target_language}")
    return refs


@dataclass
class PreparedExample:
    source_text: str
    input_obj: object
    ref_text: str
    terms: Dict[str, str]


@dataclass
class CorpusScores:
    bleu: float
    chrf: float
    term_acc: float
    combined: float


def _sentence_term_acc(pred_text: str, terms: Dict[str, str]) -> float:
    if not terms:
        return 1.0
    pred_lower = pred_text.lower()
    hit = 0
    total = 0
    for _, tgt in terms.items():
        total += 1
        if tgt.lower() in pred_lower:
            hit += 1
    return hit / total if total else 1.0


def _build_metric(weight_chrf: float, weight_term: float):
    from sacrebleu.metrics.chrf import CHRF

    chrf_metric = CHRF(char_order=6, word_order=2)

    def metric(example, prediction, trace=None):
        try:
            pred_text = str(prediction.output.output_text).strip()
        except Exception:
            pred_text = ""

        gold_text = str(example.output.output_text).strip()
        chrf = chrf_metric.sentence_score(pred_text, [gold_text]).score

        terms = {}
        try:
            for item in getattr(example.input, "glossary_entries", []) or []:
                src = str(item.get("en", ""))
                tgt = str(item.get("tgt", ""))
                if src and tgt:
                    terms[src] = tgt
        except Exception:
            terms = {}

        term_acc = _sentence_term_acc(pred_text, terms)
        return (weight_chrf * chrf) + (weight_term * (term_acc * 100.0))

    return metric


def _compute_corpus_scores(
    hyps: List[str],
    refs: List[str],
    terms_per_sentence: List[Dict[str, str]],
    weight_bleu: float,
    weight_chrf: float,
    weight_term: float,
) -> CorpusScores:
    from sacrebleu.metrics.bleu import BLEU
    from sacrebleu.metrics.chrf import CHRF

    bleu = BLEU(max_ngram_order=4, tokenize="13a").corpus_score(hyps, [refs]).score
    chrf = CHRF(char_order=6, word_order=2).corpus_score(hyps, [refs]).score

    total_terms = 0
    hit_terms = 0
    for hyp, terms in zip(hyps, terms_per_sentence):
        hyp_lower = hyp.lower()
        for _, tgt in terms.items():
            total_terms += 1
            if tgt.lower() in hyp_lower:
                hit_terms += 1

    term_acc = hit_terms / total_terms if total_terms else 0.0
    combined = weight_bleu * bleu + weight_chrf * chrf + weight_term * (term_acc * 100.0)
    return CorpusScores(bleu=bleu, chrf=chrf, term_acc=term_acc, combined=combined)


def _prepare_examples(
    input_file: Path,
    ref_file: Path,
    target_language: str,
    top_k: int,
    max_samples: int,
) -> List[PreparedExample]:
    from translations.dspy_models import Input as DSPyInput
    from translations.models import CorpusEntry, GlossaryEntry, SystemConfiguration
    from translations.utils import TranslatorGoogle, TranslatorHuggingFace

    config = SystemConfiguration.load()

    if config.translation_model.lower() == "google translate":
        translator = TranslatorGoogle()
    else:
        translator = TranslatorHuggingFace(config.translation_model)

    refs_map = _load_ref_map(ref_file, target_language)
    input_rows = _load_input_rows(input_file)

    prepared: List[PreparedExample] = []
    for row in input_rows:
        source = row["en"]
        if source not in refs_map:
            continue

        terms = row["terms"]
        similar_sentences = CorpusEntry.get_top_similar_hybrid(
            source,
            top_k=top_k,
            recall_n=max(20, top_k),
            target_language=target_language,
        )
        mt_translation = translator.translate(source)

        glossary_entries = [
            GlossaryEntry(
                english_key=src,
                translated_entry=tgt,
                target_language=target_language,
            )
            for src, tgt in terms.items()
        ]

        input_obj = DSPyInput(
            input_text=source,
            machine_translated=mt_translation,
            glossary_entries=[ge.as_dict() for ge in glossary_entries],
            past_translations=[line.as_dict() for line in similar_sentences],
        )

        prepared.append(
            PreparedExample(
                source_text=source,
                input_obj=input_obj,
                ref_text=refs_map[source],
                terms=terms,
            )
        )

        if max_samples > 0 and len(prepared) >= max_samples:
            break

    if not prepared:
        raise ValueError("No usable examples prepared.")
    return prepared


def _to_dspy_examples(prepared: List[PreparedExample]):
    import dspy
    from translations.dspy_models import Output as DSPyOutput

    result = []
    for ex in prepared:
        example = dspy.Example(input=ex.input_obj, output=DSPyOutput(output_text=ex.ref_text)).with_inputs("input")
        result.append(example)
    return result


def _split_train_dev(examples: List, dev_ratio: float, seed: int) -> Tuple[List, List]:
    if len(examples) < 3:
        return examples, examples

    rng = random.Random(seed)
    data = list(examples)
    rng.shuffle(data)

    dev_size = max(1, int(len(data) * dev_ratio))
    if dev_size >= len(data):
        dev_size = len(data) - 1

    devset = data[:dev_size]
    trainset = data[dev_size:]
    return trainset, devset


def _build_student_module(base_prompt: str):
    import dspy
    from translations.dspy_models import PostEditSignature

    class PostEditor(dspy.Module):
        def __init__(self):
            super().__init__()
            signature = PostEditSignature.with_instructions(base_prompt)
            self.predictor = dspy.Predict(signature)

        def forward(self, input):
            return self.predictor(input=input)

    return PostEditor()


def _compile_with_optimizer(
    optimizer_name: str,
    student,
    trainset: List,
    devset: List,
    metric,
    num_threads: int,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
    num_candidate_programs: int,
    num_trials: int,
    seed: int,
):
    import dspy

    opt_name = optimizer_name.lower()

    if opt_name == "mipro":
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto="light",
            num_threads=num_threads,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_candidates=num_candidate_programs,
            seed=seed,
            verbose=False,
        )
        return optimizer.compile(
            student,
            trainset=trainset,
            valset=devset,
            num_trials=num_trials,
            requires_permission_to_run=False,
            minibatch=True,
        )

    if opt_name == "bootstrap-rs":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_candidate_programs=num_candidate_programs,
            num_threads=num_threads,
            max_errors=10,
        )
        return optimizer.compile(student, trainset=trainset, valset=devset)

    if opt_name == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=1,
            max_errors=10,
        )
        return optimizer.compile(student, trainset=trainset)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _run_program_eval(program, prepared: List[PreparedExample], target_language: str, output_file: Path) -> CorpusScores:
    hyps: List[str] = []
    refs: List[str] = []
    terms_per_sentence: List[Dict[str, str]] = []

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as fw:
        for ex in prepared:
            try:
                pred = program(input=ex.input_obj)
                pred_text = str(pred.output.output_text).strip()
                if not pred_text:
                    pred_text = ex.input_obj.machine_translated
            except Exception:
                pred_text = ex.input_obj.machine_translated

            fw.write(json.dumps({"en": ex.source_text, target_language: pred_text}, ensure_ascii=False) + "\n")
            hyps.append(pred_text)
            refs.append(ex.ref_text)
            terms_per_sentence.append(ex.terms)

    return _compute_corpus_scores(
        hyps=hyps,
        refs=refs,
        terms_per_sentence=terms_per_sentence,
        weight_bleu=0.2,
        weight_chrf=0.4,
        weight_term=0.4,
    )


def _extract_optimized_prompt(program) -> str:
    try:
        return str(program.predictor.signature.instructions)
    except Exception:
        try:
            return str(program.signature.instructions)
        except Exception:
            return ""


def _dump_predictor_state(program) -> Dict:
    try:
        return program.predictor.dump_state()
    except Exception:
        try:
            state = program.dump_state()
            if isinstance(state, dict) and "predictor" in state:
                return state["predictor"]
            return state
        except Exception:
            return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use DSPy optimizer to automatically search best prompt/program on WMT terminology data, "
            "and save best prompt + predictor state."
        )
    )

    parser.add_argument("--input-file", default="eval/wmt25-terminology/ende.test.input.jsonl")
    parser.add_argument("--ref-file", default="eval/wmt25-terminology/ende.ref.output.jsonl")
    parser.add_argument("--target-lang", default="de")
    parser.add_argument("--output-file", default="datafiles/wmt25/track1/output/ende.test.dspy.output.jsonl")

    parser.add_argument("--optimizer", default="mipro", choices=["mipro", "bootstrap-rs", "bootstrap"])
    parser.add_argument("--max-samples", type=int, default=120, help="0 means all")
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=16)
    parser.add_argument("--num-candidate-programs", type=int, default=12)
    parser.add_argument("--num-trials", type=int, default=18)

    parser.add_argument("--save-best-prompt", default="datafiles/wmt25/track1/best_prompt.txt")
    parser.add_argument("--save-best-result", default="datafiles/wmt25/track1/best_prompt_metrics.json")
    parser.add_argument("--save-best-dspy-state", default="datafiles/wmt25/track1/best_prompt_dspy_state.json")

    args = parser.parse_args()

    input_file = Path(args.input_file)
    ref_file = Path(args.ref_file)
    output_file = Path(args.output_file)
    target_language = args.target_lang.strip().lower()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}")

    _setup_django()

    from translations.models import SystemConfiguration

    config = SystemConfiguration.load()

    print(f"Input file: {input_file}")
    print(f"Reference file: {ref_file}")
    print(f"Target language: {target_language}")
    print(f"Optimizer: {args.optimizer}")

    prepared = _prepare_examples(
        input_file=input_file,
        ref_file=ref_file,
        target_language=target_language,
        top_k=args.top_k,
        max_samples=args.max_samples,
    )

    dspy_examples = _to_dspy_examples(prepared)
    trainset, devset = _split_train_dev(dspy_examples, dev_ratio=args.dev_ratio, seed=args.seed)

    print(f"Prepared samples: {len(prepared)}")
    print(f"Train/Dev: {len(trainset)}/{len(devset)}")

    student = _build_student_module(config.translation_prompt)
    metric = _build_metric(weight_chrf=0.6, weight_term=0.4)

    optimized_program = _compile_with_optimizer(
        optimizer_name=args.optimizer,
        student=student,
        trainset=trainset,
        devset=devset,
        metric=metric,
        num_threads=args.num_threads,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        num_candidate_programs=args.num_candidate_programs,
        num_trials=args.num_trials,
        seed=args.seed,
    )

    scores = _run_program_eval(
        program=optimized_program,
        prepared=prepared,
        target_language=target_language,
        output_file=output_file,
    )

    best_prompt = _extract_optimized_prompt(optimized_program).strip()
    predictor_state = _dump_predictor_state(optimized_program)

    best_prompt_path = Path(args.save_best_prompt)
    best_result_path = Path(args.save_best_result)
    best_state_path = Path(args.save_best_dspy_state)

    best_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    best_result_path.parent.mkdir(parents=True, exist_ok=True)
    best_state_path.parent.mkdir(parents=True, exist_ok=True)

    best_prompt_path.write_text(best_prompt, encoding="utf-8")
    best_state_path.write_text(json.dumps(predictor_state, ensure_ascii=False, indent=2), encoding="utf-8")

    result_payload = {
        "optimizer": args.optimizer,
        "input_file": str(input_file),
        "ref_file": str(ref_file),
        "target_lang": target_language,
        "output_file": str(output_file),
        "prepared_samples": len(prepared),
        "train_size": len(trainset),
        "dev_size": len(devset),
        "metrics": {
            "bleu4": scores.bleu,
            "chrf2pp": scores.chrf,
            "term_acc": scores.term_acc,
            "combined": scores.combined,
        },
        "best_prompt_file": str(best_prompt_path),
        "best_dspy_state_file": str(best_state_path),
    }
    best_result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDSPy optimization completed.")
    print(f"- BLEU4: {scores.bleu:.2f}")
    print(f"- chrF2++: {scores.chrf:.2f}")
    print(f"- term_acc: {scores.term_acc:.4f}")
    print(f"- combined: {scores.combined:.4f}")
    print(f"- best prompt: {best_prompt_path}")
    print(f"- dspy state: {best_state_path}")
    print(f"- output jsonl: {output_file}")


if __name__ == "__main__":
    main()
