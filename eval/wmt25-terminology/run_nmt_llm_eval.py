import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import django
import litellm


DEFAULT_OUTPUT_DIR = Path("datafiles/wmt25/track1/output")


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
                raise ValueError(f"{path} 第 {idx} 行 JSON 解析失败: {exc}") from exc


def _evaluate_file(
    input_file: Path,
    output_file: Path,
    target_language: str,
    top_k: int,
    llm_temperature: float,
    domain_hint: str,
) -> Tuple[int, int]:
    from translations.models import CorpusEntry, GlossaryEntry, SystemConfiguration
    from translations.utils import TranslatorGoogle, TranslatorHuggingFace

    config = SystemConfiguration.load()

    if config.translation_model.lower() == "google translate":
        translator = TranslatorGoogle()
    else:
        translator = TranslatorHuggingFace(config.translation_model)

    total = 0
    written = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as fw:
        for row in _parse_jsonl(input_file):
            total += 1
            source_text = str(row.get("en", "")).strip()
            if not source_text:
                continue

            terms = row.get("terms")
            if not isinstance(terms, dict):
                terms = {}

            similar_sentences = CorpusEntry.get_top_similar_hybrid(
                source_text,
                top_k=top_k,
                recall_n=max(20, top_k),
                target_language=target_language,
            )

            mt_translation = translator.translate(source_text)
            output_text = mt_translation

            try:
                # Reuse the production prompt constructor exactly.
                glossary_entries = [
                    GlossaryEntry(
                        english_key=str(src_term),
                        translated_entry=str(tgt_term),
                        target_language=target_language,
                    )
                    for src_term, tgt_term in terms.items()
                ]
                messages_obj = asyncio.run(
                    translator.construct_prompt_post_edit(
                        sent=source_text,
                        sent_mt=mt_translation,
                        top_similar_sentences=similar_sentences,
                        glossary_entries=glossary_entries,
                        domain_hint=domain_hint,
                    )
                )
                messages = [m.as_dict() for m in messages_obj]
                response = litellm.completion(
                    model=config.post_editing_model,
                    messages=messages,
                    temperature=llm_temperature,
                )
                output_text = response.choices[0].message.content.strip()
            except Exception:
                output_text = mt_translation

            out_row = {"en": source_text, target_language: output_text}
            fw.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    return total, written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluate JSONL files with NMT + LLM post-editing. "
            "Covers TM retrieval and prompting modules; does not run glossary recognition."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to an input JSONL file.",
    )
    parser.add_argument(
        "--target-lang",
        required=True,
        help="Target language code used as output field name, e.g. de.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for output JSONL files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many TM examples to retrieve per source sentence.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.5,
        help="Temperature for LLM post-editing.",
    )
    parser.add_argument(
        "--domain-hint",
        default="information technology",
        help="Domain hint for prompt construction. Default: information technology",
    )

    args = parser.parse_args()
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    target_language = args.target_lang.strip().lower()
    domain_hint = (args.domain_hint or "").strip()

    if not input_file.exists() or not input_file.is_file():
        raise FileNotFoundError(f"输入文件不存在或不可读: {input_file}")

    if input_file.suffix.lower() != ".jsonl":
        raise ValueError(f"输入文件必须是 .jsonl: {input_file}")

    _setup_django()

    print(f"Input file: {input_file}")
    print(f"Target language: {target_language}")
    print(f"Output directory: {output_dir}")
    print(f"Domain hint: {domain_hint}")

    output_file = output_dir / f"{input_file.stem}.{target_language}.output.jsonl"
    total, written = _evaluate_file(
        input_file=input_file,
        output_file=output_file,
        target_language=target_language,
        top_k=args.top_k,
        llm_temperature=args.llm_temperature,
        domain_hint=domain_hint,
    )
    print(f"[{input_file.name}] total={total}, written={written}, output={output_file}")


if __name__ == "__main__":
    main()
