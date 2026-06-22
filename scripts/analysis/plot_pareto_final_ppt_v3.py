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


def gamma_text(g):
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


def plot_tradeoff(points, y_key, y_label, title, out_path, legend_loc):
    fig, ax = plt.subplots(figsize=(10, 6.2))

    families = [
        ("Baseline-MolCLR-GNN", "B", "o", "#1f77b4"),
        ("Ours-MolCLR-GNN", "O", "s", "#ff7f0e"),
    ]

    for family, prefix, marker, color in families:
        rows = [p for p in points if p["family"] == family]
        rows = sorted(rows, key=lambda r: r["gamma"])

        xs = [p["pairwise_tanimoto_mean"] for p in rows]
        ys = [p[y_key] * 100 for p in rows]

        ax.plot(
            xs,
            ys,
            linewidth=2.3,
            alpha=0.8,
            color=color,
            label=family,
        )

        ax.scatter(
            xs,
            ys,
            marker=marker,
            s=95,
            alpha=0.9,
            color=color,
            edgecolors=color,
        )

    # 三目标 Pareto 点：黑色空心圈，缩小一点
    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p[y_key] * 100 for p in pareto],
        marker="o",
        s=170,
        facecolors="none",
        edgecolors="black",
        linewidths=1.25,
        label="3-objective Pareto points",
    )

    # 关键标签：不用 γ，改成 g，避免字体问题
    def add_label(row, text, offset):
        ax.annotate(
            text,
            (row["pairwise_tanimoto_mean"], row[y_key] * 100),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
        )

    # Baseline 关键点
    for g, label, offset in [
        (5.0, "B-g5", (8, 8)),
        (20.0, "B-g20", (8, 8)),
        (100.0, "B-g100", (8, -16)),
    ]:
        matched = [p for p in points if p["family"].startswith("Baseline") and p["gamma"] == g]
        if matched:
            add_label(matched[0], label, offset)

    # Ours 低 gamma 重合点，合并标注，并移到左侧避免出界
    ours_low = [p for p in points if p["family"].startswith("Ours") and p["gamma"] in {5.0, 10.0, 20.0}]
    if ours_low:
        # 取 gamma20 这个重合点作为标注位置
        row = sorted(ours_low, key=lambda r: r["gamma"])[-1]
        add_label(row, "O-g5/10/20", (-78, -18))

    # Ours 关键点
    for g, label, offset in [
        (50.0, "O-g50", (8, -18)),
        (100.0, "O-g100", (8, -18)),
    ]:
        matched = [p for p in points if p["family"].startswith("Ours") and p["gamma"] == g]
        if matched:
            add_label(matched[0], label, offset)

    # 坐标轴留白，避免标签贴边
    xs = [p["pairwise_tanimoto_mean"] for p in points]
    ys = [p[y_key] * 100 for p in points]
    x_pad = (max(xs) - min(xs)) * 0.08
    y_pad = (max(ys) - min(ys)) * 0.12
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad * 1.4)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ax.set_xlabel("Pairwise Tanimoto Mean ↓", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, pad=18)

    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)


    ax.legend(
        fontsize=9,
        loc=legend_loc,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
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
        out_dir / "final_v2_redundancy_vs_camc_flip",
        "lower right",
    )

    plot_tradeoff(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Redundancy vs. Support Coverage",
        out_dir / "final_v2_redundancy_vs_support",
        "upper right",
    )

    print("Saved final v3 PPT figures to:", out_dir)


if __name__ == "__main__":
    main()
