import argparse
import json
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import utils

LANGS = ["de", "ru", "es"]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR.parent / "result"
OUTPUT_DIR = SCRIPT_DIR / "generated"


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


def collect_points(data: dict) -> list[dict]:
    all_systems = sorted(
        set().union(*[set(data[lang]["proper"].keys()) for lang in LANGS if lang in data and "proper" in data[lang]])
    )
    points = []
    for system in all_systems:
        bleu_vals = []
        acc_vals = []
        cons_vals = []
        for lang in LANGS:
            metrics = data.get(lang, {}).get("proper", {}).get(system, {})
            bleu = metrics.get("bleu4")
            acc = metrics.get("proper_term_success_rate")
            cons = metrics.get("consistency_frequent")
            if isinstance(bleu, (int, float)) and isinstance(acc, (int, float)) and isinstance(cons, (int, float)):
                if bleu >= 0 and acc >= 0 and cons >= 0:
                    bleu_vals.append(bleu)
                    acc_vals.append(acc)
                    cons_vals.append(cons)
        if bleu_vals and acc_vals and cons_vals:
            points.append(
                {
                    "name": system,
                    "bleu": statistics.mean(bleu_vals),
                    "acc": statistics.mean(acc_vals) * 100,
                    "cons": statistics.mean(cons_vals) * 100,
                }
            )
    return points


def plot_3d(points: list[dict], output_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    others = [p for p in points if p["name"] != "Glossa"]
    glossa = [p for p in points if p["name"] == "Glossa"]

    if others:
        ax.scatter(
            [p["bleu"] for p in others],
            [p["acc"] for p in others],
            [p["cons"] for p in others],
            color="#888888",
            s=34,
            alpha=0.85,
            depthshade=True,
            label="Other Systems",
        )

    if glossa:
        ax.scatter(
            [p["bleu"] for p in glossa],
            [p["acc"] for p in glossa],
            [p["cons"] for p in glossa],
            color="#d94801",
            s=85,
            alpha=1.0,
            depthshade=True,
            label="Glossa",
        )

    for p in points:
        display_name = utils.SYS_TO_NAME_2.get(p["name"], p["name"])
        color = "#d94801" if p["name"] == "Glossa" else "#303030"
        ax.text(
            p["bleu"],
            p["acc"],
            p["cons"],
            display_name,
            fontsize=7,
            color=color,
        )

    ax.set_xlabel("BLEU-4", labelpad=8)
    ax.set_ylabel("Terminology Accuracy (%)", labelpad=8)
    ax.set_zlabel("Terminology Consistency (%)", labelpad=8)
    ax.view_init(elev=20, azim=-58)

    if points:
        bleu_vals = [p["bleu"] for p in points]
        acc_vals = [p["acc"] for p in points]
        cons_vals = [p["cons"] for p in points]
        ax.set_xlim(min(bleu_vals) - 1.2, max(bleu_vals) + 1.2)
        ax.set_ylim(min(acc_vals) - 1.5, max(acc_vals) + 1.5)
        ax.set_zlim(min(cons_vals) - 1.5, max(cons_vals) + 1.5)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, format="svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="tradeoff_3d_bleu_acc_cons_with_glossa.svg")
    args = parser.parse_args()

    data = load_data_with_glossa()
    points = collect_points(data)
    plot_3d(points, args.output_name)


if __name__ == "__main__":
    main()
