from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
from unittest.mock import patch

from django.test import SimpleTestCase
import torch

from translations import alignment


@dataclass
class _FakeGlossaryEntry:
    english_key: str
    translated_entry: str


def _norm_en(text: str) -> str:
    return alignment._normalize_for_compare(text)


def _norm_zh(text: str) -> str:
    return alignment._normalize_target_for_compare(text)


def _make_encode_mock(expected_pairs: Sequence[Tuple[str, str]]) -> Callable[[Sequence[str]], torch.Tensor]:
    en_concepts = {_norm_en(src): idx for idx, (src, _) in enumerate(expected_pairs)}
    zh_concepts = {_norm_zh(tgt): idx for idx, (_, tgt) in enumerate(expected_pairs)}
    concept_dim = max(1, len(expected_pairs))

    def _encode(texts: Sequence[str]) -> torch.Tensor:
        dim = concept_dim + max(1, len(texts))
        emb = torch.zeros((len(texts), dim), dtype=torch.float32)
        for i, term in enumerate(texts):
            en_key = _norm_en(term)
            zh_key = _norm_zh(term)
            if en_key in en_concepts:
                emb[i, en_concepts[en_key]] = 1.0
            elif zh_key in zh_concepts:
                emb[i, zh_concepts[zh_key]] = 1.0
            else:
                emb[i, concept_dim + i] = 1.0
        return emb

    return _encode


class AlignmentSentencePairTests(SimpleTestCase):
    def test_sentence_pair_term_extraction_quality(self):
        cases: List[Dict[str, object]] = [
            {
                "name": "virology_surveillance_case",
                "source": (
                    "Additionally, practices participating in current virology surveillance are now taking "
                    "samples for COVID-19 surveillance from low-risk patients presenting with LRTIs."
                ),
                "target": "此外，参与当前病毒学监测的诊所现在正在对出现下呼吸道感染症状的低风险患者进行 COVID-19 监测并采集样本。",
                "expected": [
                    ("virology surveillance", "病毒学监测"),
                    ("samples", "样本"),
                    ("COVID-19 surveillance", "COVID-19 监测"),
                    ("low-risk patients", "低风险患者"),
                    ("LRTIs", "下呼吸道感染"),
                    ("practices", "诊所"),
                ],
                "glossary_existing": [],
            },
            {
                "name": "adverse_event_reporting_case",
                "source": (
                    "The hospital submitted an adverse event report to the national pharmacovigilance center "
                    "after two patients developed severe allergic reactions."
                ),
                "target": "医院在两名患者出现严重过敏反应后，向国家药物警戒中心提交了不良事件报告。",
                "expected": [
                    ("adverse event report", "不良事件报告"),
                    ("national pharmacovigilance center", "国家药物警戒中心"),
                    ("patients", "患者"),
                    ("severe allergic reactions", "严重过敏反应"),
                ],
                "glossary_existing": [],
            },
            {
                "name": "regulatory_inspection_case",
                "source": (
                    "During the regulatory inspection, the manufacturer provided batch records and quality "
                    "control data for the sterile injectable product."
                ),
                "target": "在监管检查期间，制造商提供了该无菌注射剂产品的批次记录和质量控制数据。",
                "expected": [
                    ("regulatory inspection", "监管检查"),
                    ("manufacturer", "制造商"),
                    ("batch records", "批次记录"),
                    ("quality control data", "质量控制数据"),
                    ("sterile injectable product", "无菌注射剂产品"),
                ],
                "glossary_existing": [("batch records", "批次记录")],
            },
            {
                "name": "contract_dispute_case",
                "source": (
                    "If either party materially breaches the confidentiality agreement, the non-breaching party "
                    "may seek injunctive relief and claim liquidated damages."
                ),
                "target": "若任一方实质性违反保密协议，守约方可申请禁令救济并主张约定违约金。",
                "expected": [
                    ("confidentiality agreement", "保密协议"),
                    ("non-breaching party", "守约方"),
                    ("injunctive relief", "禁令救济"),
                    ("liquidated damages", "约定违约金"),
                ],
                "glossary_existing": [],
            },
        ]

        total_expected = 0
        total_hits = 0

        for case in cases:
            name = str(case["name"])
            source = str(case["source"])
            target = str(case["target"])
            expected_pairs = list(case["expected"])  # type: ignore[assignment]
            glossary_existing = list(case["glossary_existing"])  # type: ignore[assignment]

            encode_mock = _make_encode_mock(expected_pairs)
            glossary_entries = [
                _FakeGlossaryEntry(english_key=src, translated_entry=tgt) for src, tgt in glossary_existing
            ]

            with self.subTest(case=name), patch("translations.alignment._encode_texts", side_effect=encode_mock), patch(
                "translations.alignment.GlossaryEntry.get_entries", return_value=glossary_entries
            ):
                result = alignment.suggest_term_pairs(source, target, limit=20, similarity_threshold=0.75)

            extracted_pairs = [
                (str(item["source"]), str(item["target"])) for item in result.get("candidates", [])
            ]
            extracted_keys = {(_norm_en(src), _norm_zh(tgt)) for src, tgt in extracted_pairs}
            expected_keys = {(_norm_en(src), _norm_zh(tgt)) for src, tgt in expected_pairs}
            glossary_keys = {(_norm_en(src), _norm_zh(tgt)) for src, tgt in glossary_existing}

            hit_keys = expected_keys.intersection(extracted_keys)
            missing_keys = expected_keys - extracted_keys
            leaked_glossary = glossary_keys.intersection(extracted_keys)

            total_expected += len(expected_keys)
            total_hits += len(hit_keys)

            print("")
            print(f"[CASE] {name}")
            print(f"method={result.get('method')} latency_ms={result.get('latency_ms')}")
            print(f"source_chunks={result.get('source_chunks')}")
            print(f"target_chunks={result.get('target_chunks')}")
            print(f"similarity_matrix_shape={result.get('similarity_matrix_shape')}")
            similarity_matrix = result.get("similarity_matrix") or []
            if similarity_matrix:
                zh_chunks = result.get("target_chunks") or []
                print("similarity_matrix_columns=")
                for col_idx, zh in enumerate(zh_chunks):
                    print(f"  [{col_idx}] {zh}")
                print("similarity_matrix_rows=")
                for row_idx, row in enumerate(similarity_matrix):
                    source_chunks = result.get("source_chunks") or []
                    en_chunk = source_chunks[row_idx] if row_idx < len(source_chunks) else f"<row_{row_idx}>"
                    print(f"  [{row_idx}] {en_chunk} => {row}")
            print("expected_pairs=")
            for src, tgt in expected_pairs:
                print(f"  - {src} -> {tgt}")
            print("extracted_pairs=")
            for src, tgt in extracted_pairs:
                print(f"  - {src} -> {tgt}")
            print(f"hits={len(hit_keys)}/{len(expected_keys)} missing={len(missing_keys)}")
            if missing_keys:
                print("missing_pairs=")
                for src_key, tgt_key in sorted(missing_keys):
                    print(f"  - {src_key} -> {tgt_key}")
            print(f"glossary_filtered={result.get('glossary_filtered', 0)} leaked_glossary={len(leaked_glossary)}")

            self.assertFalse(leaked_glossary, f"{name}: glossary terms should not reappear in candidates")
            self.assertGreaterEqual(
                len(hit_keys),
                max(1, int(len(expected_keys) * 0.8)),
                f"{name}: expected term recall too low",
            )

        overall_recall = (total_hits / total_expected) if total_expected else 0.0
        print("")
        print(f"[SUMMARY] overall_hits={total_hits}/{total_expected} overall_recall={overall_recall:.2%}")
        self.assertGreaterEqual(overall_recall, 0.85, "Overall term extraction recall is lower than expected")
