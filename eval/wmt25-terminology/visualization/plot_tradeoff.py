# %%

import json
import os
import statistics
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils

LANGS = ["de", "ru", "es"]
EXCLUDED_SYSTEMS = {"TranssionMT", "ContexTerm"}
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR.parent / "result"
OUTPUT_DIR = SCRIPT_DIR / "generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "serif"


def load_data_with_glossa() -> dict:
    with (RESULT_DIR / "track1_score_dict.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    for lang in LANGS:
        glossa_path = RESULT_DIR / f"glossa_track1_score_{lang}_dict.json"
        if not glossa_path.exists():
            continue
        with glossa_path.open("r", encoding="utf-8") as f:
            glossa_data = json.load(f)
        if lang in glossa_data and "proper" in glossa_data[lang]:
            data.setdefault(lang, {}).setdefault("proper", {}).update(glossa_data[lang]["proper"])

    return data


def collect_system_points(data: dict, x_metric: str, y_metric: str) -> list[dict]:
    all_systems = sorted(
        set().union(*[set(data[lang]["proper"].keys()) for lang in LANGS if lang in data and "proper" in data[lang]])
    )
    all_systems = [sys for sys in all_systems if sys not in EXCLUDED_SYSTEMS]

    points = []
    for system in all_systems:
        x_values = []
        y_values = []
        for lang in LANGS:
            metrics = data.get(lang, {}).get("proper", {}).get(system, {})
            x_val = metrics.get(x_metric)
            y_val = metrics.get(y_metric)
            if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                if x_val >= 0 and y_val >= 0:
                    x_values.append(x_val)
                    y_values.append(y_val)

        if x_values and y_values:
            points.append(
                {
                    "name": system,
                    "x": statistics.mean(x_values),
                    "y": statistics.mean(y_values),
                }
            )

    return points


def collect_proper_table_rows(data: dict) -> list[dict]:
    all_systems = sorted(
        set().union(*[set(data[lang]["proper"].keys()) for lang in LANGS if lang in data and "proper" in data[lang]])
    )
    all_systems = [sys for sys in all_systems if sys not in EXCLUDED_SYSTEMS]
    rows = []
    for system in all_systems:
        acc_vals = []
        chrf_vals = []
        cons_vals = []
        for lang in LANGS:
            metrics = data.get(lang, {}).get("proper", {}).get(system, {})
            acc = metrics.get("proper_term_success_rate")
            chrf = metrics.get("chrf2++")
            cons = metrics.get("consistency_frequent")
            if isinstance(acc, (int, float)) and acc >= 0:
                acc_vals.append(acc)
            if isinstance(chrf, (int, float)) and chrf >= 0:
                chrf_vals.append(chrf)
            if isinstance(cons, (int, float)) and cons >= 0:
                cons_vals.append(cons)
        if acc_vals and chrf_vals and cons_vals:
            rows.append(
                {
                    "name": system,
                    "term_acc": statistics.mean(acc_vals),
                    "chrf": statistics.mean(chrf_vals),
                    "consistency": statistics.mean(cons_vals),
                }
            )
    rows.sort(key=lambda r: (r["term_acc"], r["chrf"]), reverse=True)
    return rows


def save_proper_table(rows: list[dict], data: dict):
    csv_path = OUTPUT_DIR / "proper_term_summary_with_glossa.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "term_acc", "chrf2++", "consistency_frequent"])
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    f"{row['term_acc']:.6f}",
                    f"{row['chrf']:.6f}",
                    f"{row['consistency']:.6f}",
                ]
            )

    def color_cell_chrf(val: float) -> str:
        color = f"SeaGreen3!{max(0, min(95, (val-50)*4.5)):.0f}!Firebrick3!50"
        return f"\\cellcolor{{{color}}} {val:.1f}"

    def color_cell_acc(val: float) -> str:
        color = f"SeaGreen3!{max(0, min(95, (val-70)*3)):.0f}!Firebrick3!50"
        return f"\\cellcolor{{{color}}} {val:.1f}"

    def color_cell_cons(val: float) -> str:
        color = f"SeaGreen3!{max(0, min(95, (val-80)*10)):.0f}!Firebrick3!50"
        return f"\\cellcolor{{{color}}} {val:.1f}"

    def nocolor_cell(val: float) -> str:
        return f"{val:.1f}"

    def metric_or_empty(lang: str, sys: str, metric: str, scale: float = 1.0) -> str:
        metrics = data.get(lang, {}).get("proper", {}).get(sys, {})
        val = metrics.get(metric)
        if isinstance(val, (int, float)) and val >= 0:
            return nocolor_cell(val * scale)
        return ""

    row_map = {row["name"]: row for row in rows}
    systems = [row["name"] for row in rows]
    tex_path = OUTPUT_DIR / "track1_proper_with_glossa.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        print(
            r"\begin{tabular}{l  cvvv cvvv cvvv}",
            r"\toprule",
            r"& \multicolumn{4}{c}{\bf Proper, ChrF} & \multicolumn{4}{c}{\bf Proper, Acc.} & \multicolumn{4}{c}{\bf Proper, Cons.} \\",
            r"\bf System  ",
            r"& \bf Avg & \bf Es & \bf De & \bf Ru   & \bf Avg & \bf Es & \bf De & \bf Ru  & \bf Avg & \bf Es & \bf De & \bf Ru  \\",
            r"\midrule",
            sep="\n",
            file=f,
        )
        for sys in systems:
            row = row_map[sys]
            sys_name = utils.SYS_TO_NAME.get(sys, sys)
            if sys == "Glossa":
                sys_name = r"\rowcolor{Apricot!25} \textbf{" + sys_name + "}"
            print(
                sys_name,
                color_cell_chrf(row["chrf"]),
                *[
                    metric_or_empty(lang, sys, "chrf2++")
                    for lang in ["es", "de", "ru"]
                ],
                color_cell_acc(row["term_acc"] * 100),
                *[
                    metric_or_empty(lang, sys, "proper_term_success_rate", 100)
                    for lang in ["es", "de", "ru"]
                ],
                color_cell_cons(row["consistency"] * 100),
                *[
                    metric_or_empty(lang, sys, "consistency_frequent", 100)
                    for lang in ["es", "de", "ru"]
                ],
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


def scatter_with_glossa_highlight(
    points: list[dict],
    x_label: str,
    y_label: str,
    output_name: str,
    y_min_floor: float | None = None,
    x_min_floor: float | None = None,
    x_dense_sparse: bool = False,
    fig_size: tuple[float, float] = (4.6, 3.2),
):
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    others = [p for p in points if p["name"] != "Glossa"]
    glossa = [p for p in points if p["name"] == "Glossa"]

    if others:
        plt.scatter(
            [p["x"] for p in others],
            [p["y"] for p in others],
            color="#999999",
            marker=".",
            s=110,
            zorder=1,
            label="Other Systems",
        )

    if glossa:
        plt.scatter(
            [p["x"] for p in glossa],
            [p["y"] for p in glossa],
            color="#d94801",
            marker="o",
            s=60,
            zorder=3,
            label="Glossa",
        )

    for p in points:
        display_name = utils.SYS_TO_NAME_2.get(p["name"], p["name"])
        color = "#d94801" if p["name"] == "Glossa" else "#444444"
        weight = "bold" if p["name"] == "Glossa" else "normal"
        plt.text(
            p["x"],
            p["y"],
            display_name,
            fontsize=6,
            ha="center",
            va="center",
            color=color,
            fontweight=weight,
            zorder=4,
        )

    # tighten y-axis to improve visual focus
    if points:
        y_vals = [p["y"] for p in points]
        y_min = min(y_vals)
        y_max = max(y_vals)
        span = max(y_max - y_min, 0.5)
        pad = max(0.15, span * 0.08)
        lower = y_min - pad
        upper = y_max + pad
        if y_min_floor is not None:
            lower = max(y_min_floor, lower)
        ax.set_ylim(lower, upper)

    if x_dense_sparse:
        # 70%-85% occupies very small width, 85%-90% occupies very large width.
        x0, x1, x2 = 0.70, 0.85, 0.90
        w1, w2 = 0.18, 0.82

        def forward(v):
            arr = np.asarray(v)
            out = np.where(
                arr <= x1,
                (arr - x0) / (x1 - x0) * w1,
                w1 + (arr - x1) / (x2 - x1) * w2,
            )
            if np.isscalar(v):
                return float(out)
            return out

        def inverse(u):
            arr = np.asarray(u)
            out = np.where(
                arr <= w1,
                x0 + arr / w1 * (x1 - x0),
                x1 + (arr - w1) / w2 * (x2 - x1),
            )
            if np.isscalar(u):
                return float(out)
            return out

        ax.set_xscale("function", functions=(forward, inverse))
        ax.set_xlim(x0, x2)
        ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90])
        ax.set_xticklabels(["70%", "75%", "80%", "85%", "90%"])
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
        if x_min_floor is not None:
            ax.set_xlim(left=x_min_floor)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if glossa:
        plt.legend(frameon=False, loc="best", fontsize=7)

    plt.tight_layout(pad=0.2)
    plt.savefig(OUTPUT_DIR / output_name, format="svg")
    plt.close()


def main():
    data = load_data_with_glossa()

    # Proper Term mode only: terminology accuracy vs chrF++
    points_acc = collect_system_points(
        data,
        x_metric="proper_term_success_rate",
        y_metric="chrf2++",
    )
    scatter_with_glossa_highlight(
        points=points_acc,
        x_label="Terminology Accuracy (Proper Term)",
        y_label="ChrF++",
        output_name="tradeoff_accuracy_proper_with_glossa.svg",
        y_min_floor=60,
        fig_size=(4.8, 3.2),
    )

    # Proper Term mode only: consistency vs chrF++
    points_cons = collect_system_points(
        data,
        x_metric="consistency_frequent",
        y_metric="chrf2++",
    )
    points_cons = [p for p in points_cons if p["x"] >= 0.70]
    scatter_with_glossa_highlight(
        points=points_cons,
        x_label="Terminology Consistency (Proper Term)",
        y_label="ChrF++",
        output_name="tradeoff_consistency_proper_with_glossa.svg",
        y_min_floor=60,
        x_dense_sparse=True,
        fig_size=(5.6, 3.8),
    )

    table_rows = collect_proper_table_rows(data)
    save_proper_table(table_rows, data)


if __name__ == "__main__":
    main()

# %%
