#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_gamma(config):
    m = re.search(r"gamma([0-9]+)p([0-9]+)", config)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def gamma_label(g):
    if g is None:
        return ""
    if abs(g - int(g)) < 1e-9:
        return str(int(g))
    return str(g)


def load_points(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["support_coverage"] = float(r["support_coverage"])
            r["camc_flip_coverage"] = float(r["camc_flip_coverage"])
            r["pairwise_tanimoto_mean"] = float(r["pairwise_tanimoto_mean"])
            r["is_pareto"] = str(r["is_pareto"]).lower() == "true"
            r["gamma"] = parse_gamma(r["config"])
            rows.append(r)
    return rows


def method_short(family):
    return "B" if family.startswith("Baseline") else "O"


def method_name(family):
    return "Baseline-MolCLR-GNN" if family.startswith("Baseline") else "Ours-MolCLR-GNN"


def get_key_label(row):
    prefix = method_short(row["family"])
    g = gamma_label(row["gamma"])
    return f"{prefix}-γ{g}"


def plot_tradeoff(points, y_key, y_label, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6.2))

    families = ["Baseline-MolCLR-GNN", "Ours-MolCLR-GNN"]

    style = {
        "Baseline-MolCLR-GNN": {"marker": "o"},
        "Ours-MolCLR-GNN": {"marker": "s"},
    }

    for fam in families:
        rows = [p for p in points if p["family"] == fam]
        rows = sorted(rows, key=lambda r: r["gamma"])

        xs = [p["pairwise_tanimoto_mean"] for p in rows]
        ys = [p[y_key] * 100 for p in rows]

        ax.plot(
            xs,
            ys,
            linewidth=2.2,
            alpha=0.75,
            label=method_name(fam),
        )

        ax.scatter(
            xs,
            ys,
            marker=style[fam]["marker"],
            s=90,
            alpha=0.85,
        )

    # Pareto points: black hollow circles
    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p[y_key] * 100 for p in pareto],
        marker="o",
        s=190,
        facecolors="none",
        edgecolors="black",
        linewidths=1.35,
        label="3-objective Pareto points",
    )

    # 只标关键点，避免满屏文字
    label_whitelist = {
        "B-γ5",
        "B-γ20",
        "B-γ100",
        "O-γ50",
        "O-γ100",
    }

    # Ours γ5/10/20 基本重合，合并标一次
    ours_low_gamma = [p for p in points if p["family"].startswith("Ours") and p["gamma"] in {5.0, 10.0, 20.0}]
    if ours_low_gamma:
        p = ours_low_gamma[-1]
        ax.annotate(
            "O-γ5/10/20",
            (p["pairwise_tanimoto_mean"], p[y_key] * 100),
            textcoords="offset points",
            xytext=(8, -18),
            fontsize=10,
        )

    for p in points:
        label = get_key_label(p)
        if label not in label_whitelist:
            continue

        x = p["pairwise_tanimoto_mean"]
        y = p[y_key] * 100

        if label == "B-γ100":
            offset = (8, -14)
        elif label == "B-γ20":
            offset = (8, 8)
        elif label == "B-γ5":
            offset = (8, 8)
        elif label == "O-γ50":
            offset = (8, -18)
        elif label == "O-γ100":
            offset = (8, -18)
        else:
            offset = (8, 8)

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
        )

    ax.set_xlabel("Pairwise Tanimoto Mean ↓", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)

    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)

    # 更简洁的方向说明
    ax.text(
        0.02,
        0.96,
        "Preferred direction: ← lower redundancy, ↑ higher score",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.12, linewidth=0),
    )

    ax.legend(
        fontsize=9,
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=400)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    points = load_points(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_tradeoff(
        points,
        "camc_flip_coverage",
        "CAMC Flip Coverage ↑ (%)",
        "Redundancy vs. CAMC Flip Coverage",
        out_dir / "final_redundancy_vs_camc_flip",
    )

    plot_tradeoff(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Redundancy vs. Support Coverage",
        out_dir / "final_redundancy_vs_support",
    )

    print("Saved final PPT figures to:", out_dir)


if __name__ == "__main__":
    main()
