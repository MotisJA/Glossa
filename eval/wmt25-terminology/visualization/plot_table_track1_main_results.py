# %%
import json
import os
import statistics
from pathlib import Path

import utils

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR.parent / "result"
OUTPUT_DIR = SCRIPT_DIR / "generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGS = ["es", "de", "ru"]


def load_data_with_glossa() -> dict:
    with (RESULT_DIR / "track1_score_dict.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    for lang in LANGS:
        lang_block = data.setdefault(lang, {})
        lang_block.setdefault("proper", {})
        lang_block.setdefault("random", {})
        lang_block.setdefault("noterm", {})

        glossa_path = RESULT_DIR / f"glossa_track1_score_{lang}_dict.json"
        if glossa_path.exists():
            with glossa_path.open("r", encoding="utf-8") as gf:
                glossa_data = json.load(gf)
            if lang in glossa_data and "proper" in glossa_data[lang]:
                lang_block["proper"].update(glossa_data[lang]["proper"])

    return data


def color_cell_chrf(val):
    color = f"SeaGreen3!{max(0, min(95, (val-50)*4.5)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_acc(val):
    color = f"SeaGreen3!{max(0, min(95, (val-70)*3)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_cons(val):
    color = f"SeaGreen3!{max(0, min(95, (val-80)*10)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def nocolor_cell(val):
    return f"{val:.1f}"


def metric_values(data: dict, mode: str, sys: str, metric: str, scale: float = 1.0):
    vals = []
    for lang in LANGS:
        row = data.get(lang, {}).get(mode, {}).get(sys, {})
        val = row.get(metric)
        if isinstance(val, (int, float)) and val >= 0:
            vals.append(val * scale)
    return vals


def metric_cells(data: dict, mode: str, sys: str, metric: str, scale: float = 1.0):
    cells = []
    for lang in LANGS:
        row = data.get(lang, {}).get(mode, {}).get(sys, {})
        val = row.get(metric)
        if isinstance(val, (int, float)) and val >= 0:
            cells.append(nocolor_cell(val * scale))
        else:
            cells.append("")
    return cells


def mean_cell(data: dict, mode: str, sys: str, metric: str, formatter, scale: float = 1.0):
    vals = metric_values(data, mode, sys, metric, scale)
    return formatter(statistics.mean(vals)) if vals else ""


def main():
    data = load_data_with_glossa()

    systems = sorted(
        set().union(
            *[
                set(data.get(lang, {}).get("proper", {}).keys())
                | set(data.get(lang, {}).get("random", {}).keys())
                | set(data.get(lang, {}).get("noterm", {}).keys())
                for lang in LANGS
            ]
        )
    )

    systems.sort(
        key=lambda sys: (
            statistics.mean(
                metric_values(data, "proper", sys, "chrf2++")
            ) + statistics.mean(
                metric_values(data, "proper", sys, "proper_term_success_rate")
            )
        ) if metric_values(data, "proper", sys, "chrf2++") and metric_values(data, "proper", sys, "proper_term_success_rate") else -1000,
        reverse=True,
    )

    with (OUTPUT_DIR / "track1_main_results.tex").open("w", encoding="utf-8") as f:
        print(
            r"\begin{tabular}{l  cvvv cvvv cvvv|c cvvv cvvv|c cvvv}",
            r"\toprule",
            r"& \multicolumn{4}{c}{\bf Proper, ChrF} & \multicolumn{4}{c}{\bf Proper, Acc.} & \multicolumn{4}{c|}{\bf Proper, Cons.} &",
            r"& \multicolumn{4}{c}{\bf Random, ChrF} & \multicolumn{4}{c|}{\bf Random, Acc.} &",
            r"& \multicolumn{4}{c}{\bf NoTerm, ChrF} \\",
            r"\bf System  ",
            r"& \bf Avg & \bf Es & \bf De & \bf Ru   & \bf Avg & \bf Es & \bf De & \bf Ru  & \bf Avg & \bf Es & \bf De & \bf Ru  &",
            r"& \bf Avg & \bf Es & \bf De & \bf Ru   & \bf Avg & \bf Es & \bf De & \bf Ru  &",
            r"& \bf Avg & \bf Es & \bf De & \bf Ru   \\",
            r"\midrule",
            sep="\n",
            file=f,
        )

        for sys in systems:
            print(
                utils.SYS_TO_NAME.get(sys, sys),
                mean_cell(data, "proper", sys, "chrf2++", color_cell_chrf),
                *metric_cells(data, "proper", sys, "chrf2++"),
                mean_cell(data, "proper", sys, "proper_term_success_rate", color_cell_acc, 100),
                *metric_cells(data, "proper", sys, "proper_term_success_rate", 100),
                mean_cell(data, "proper", sys, "consistency_frequent", color_cell_cons, 100),
                *metric_cells(data, "proper", sys, "consistency_frequent", 100),
                "",
                mean_cell(data, "random", sys, "chrf2++", color_cell_chrf),
                *metric_cells(data, "random", sys, "chrf2++"),
                mean_cell(data, "random", sys, "proper_term_success_rate", color_cell_acc, 100),
                *metric_cells(data, "random", sys, "proper_term_success_rate", 100),
                "",
                mean_cell(data, "noterm", sys, "chrf2++", color_cell_chrf),
                *metric_cells(data, "noterm", sys, "chrf2++"),
                sep=" & ",
                end="\\\\\n",
                file=f,
            )

        print(
            r"\bottomrule",
            r"\end{tabular}",
            sep="\n",
            file=f,
        )


if __name__ == "__main__":
    main()

# %%
