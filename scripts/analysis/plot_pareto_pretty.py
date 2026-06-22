#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def parse_gamma(config):
    m = re.search(r"gamma([0-9]+)p([0-9]+)", config)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def short_family(family):
    if family.startswith("Baseline"):
        return "B"
    if family.startswith("Ours"):
        return "O"
    return family[:1]


def short_label(row):
    g = parse_gamma(row["config"])
    g_txt = str(int(g)) if g is not None and abs(g - int(g)) < 1e-9 else str(g)
    return f"{short_family(row['family'])}-γ{g_txt}"


def load_points(path):
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["support_coverage"] = float(r["support_coverage"])
            r["camc_flip_coverage"] = float(r["camc_flip_coverage"])
            r["pairwise_tanimoto_mean"] = float(r["pairwise_tanimoto_mean"])
            r["camc_at_05"] = float(r["camc_at_05"])
            r["mean_cf_drop_covered"] = float(r["mean_cf_drop_covered"])
            r["is_pareto"] = str(r["is_pareto"]).lower() == "true"
            r["gamma"] = parse_gamma(r["config"])
            points.append(r)
    return points


def style_for_family(family):
    if family.startswith("Baseline"):
        return {
            "marker": "o",
            "label": "Baseline-MolCLR-GNN",
        }
    return {
        "marker": "s",
        "label": "Ours-MolCLR-GNN",
    }


def plot_projection(points, y_key, y_label, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6.6))

    families = ["Baseline-MolCLR-GNN", "Ours-MolCLR-GNN"]

    for family in families:
        rows = [p for p in points if p["family"] == family]
        rows = sorted(rows, key=lambda x: x["gamma"])

        style = style_for_family(family)

        xs = [p["pairwise_tanimoto_mean"] for p in rows]
        ys = [p[y_key] * 100 for p in rows]

        ax.plot(xs, ys, linewidth=1.4, alpha=0.55)

        dominated = [p for p in rows if not p["is_pareto"]]
        pareto = [p for p in rows if p["is_pareto"]]

        ax.scatter(
            [p["pairwise_tanimoto_mean"] for p in dominated],
            [p[y_key] * 100 for p in dominated],
            marker=style["marker"],
            s=90,
            alpha=0.35,
            label=style["label"] + " dominated",
        )

        ax.scatter(
            [p["pairwise_tanimoto_mean"] for p in pareto],
            [p[y_key] * 100 for p in pareto],
            marker=style["marker"],
            s=115,
            alpha=0.9,
            label=style["label"] + " Pareto",
        )

        # Pareto points: black hollow outline, not a huge star
        ax.scatter(
            [p["pairwise_tanimoto_mean"] for p in pareto],
            [p[y_key] * 100 for p in pareto],
            marker="o",
            s=250,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
        )

    # 合并几乎重合的标签，避免 O-γ5/10/20 挤成一团
    grouped = defaultdict(list)
    for p in points:
        key = (
            p["family"],
            round(p["pairwise_tanimoto_mean"], 4),
            round(p[y_key] * 100, 2),
        )
        grouped[key].append(p)

    for (family, x, y), rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["gamma"])
        prefix = short_family(family)
        gammas = []
        for r in rows:
            g = parse_gamma(r["config"])
            gammas.append(str(int(g)) if abs(g - int(g)) < 1e-9 else str(g))

        if len(gammas) == 1:
            label = f"{prefix}-γ{gammas[0]}"
        else:
            label = f"{prefix}-γ{'/'.join(gammas)}"

        if prefix == "B":
            dx, dy = 7, 7
        else:
            dx, dy = 7, -16

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
        )

    ax.set_xlabel("Pairwise Tanimoto Mean ↓", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    # 标注理想方向
    ax.annotate(
        "Better region\nlow redundancy + high score",
        xy=(0.038, ax.get_ylim()[1] - 0.12),
        xytext=(0.044, ax.get_ylim()[1] - 0.38),
        arrowprops=dict(arrowstyle="->", linewidth=1.1),
        fontsize=9,
    )

    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_3d_clean(points, out_path):
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    families = ["Baseline-MolCLR-GNN", "Ours-MolCLR-GNN"]

    for family in families:
        rows = [p for p in points if p["family"] == family]
        rows = sorted(rows, key=lambda x: x["gamma"])
        style = style_for_family(family)

        ax.plot(
            [p["pairwise_tanimoto_mean"] for p in rows],
            [p["support_coverage"] * 100 for p in rows],
            [p["camc_flip_coverage"] * 100 for p in rows],
            linewidth=1.2,
            alpha=0.5,
        )

        ax.scatter(
            [p["pairwise_tanimoto_mean"] for p in rows],
            [p["support_coverage"] * 100 for p in rows],
            [p["camc_flip_coverage"] * 100 for p in rows],
            marker=style["marker"],
            s=80,
            alpha=0.8,
            label=style["label"],
        )

    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p["support_coverage"] * 100 for p in pareto],
        [p["camc_flip_coverage"] * 100 for p in pareto],
        marker="o",
        s=220,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Pareto frontier",
    )

    ax.set_xlabel("Pairwise Tanimoto Mean ↓")
    ax.set_ylabel("Support Coverage ↑ (%)")
    ax.set_zlabel("CAMC Flip Coverage ↑ (%)")
    ax.set_title("3-objective Pareto frontier under legacy protocol")

    # 更适合看的角度
    ax.view_init(elev=24, azim=-55)

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
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

    plot_projection(
        points,
        "camc_flip_coverage",
        "CAMC Flip Coverage ↑ (%)",
        "Pareto trade-off: redundancy vs CAMC flip coverage",
        out_dir / "pretty_redundancy_vs_camc_flip",
    )

    plot_projection(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Pareto trade-off: redundancy vs support coverage",
        out_dir / "pretty_redundancy_vs_support",
    )

    plot_3d_clean(
        points,
        out_dir / "pretty_3obj_3d_no_labels",
    )

    print("Saved pretty figures to:", out_dir)


if __name__ == "__main__":
    main()
