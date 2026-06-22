#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_gamma(config):
    m = re.search(r"gamma([0-9]+)p([0-9]+)", config)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


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


def short_family(family):
    if family.startswith("Baseline"):
        return "B"
    if family.startswith("Ours"):
        return "O"
    return family[:1]


def gamma_label(g):
    if abs(g - int(g)) < 1e-9:
        return str(int(g))
    return str(g)


def family_style(family):
    if family.startswith("Baseline"):
        return {
            "marker": "o",
            "line": "-",
            "name": "Baseline-MolCLR-GNN",
        }
    return {
        "marker": "s",
        "line": "-",
        "name": "Ours-MolCLR-GNN",
    }


def annotate_grouped_labels(ax, points, y_key):
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
        gammas = [gamma_label(r["gamma"]) for r in rows]

        if len(gammas) == 1:
            label = f"{prefix}-γ{gammas[0]}"
        else:
            label = f"{prefix}-γ{'/'.join(gammas)}"

        if prefix == "B":
            dx, dy = 8, 8
        else:
            dx, dy = 8, -18

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
        )


def draw_tradeoff_plot(points, y_key, y_label, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    families = ["Baseline-MolCLR-GNN", "Ours-MolCLR-GNN"]

    for family in families:
        rows = [p for p in points if p["family"] == family]
        rows = sorted(rows, key=lambda r: r["gamma"])
        style = family_style(family)

        xs = [p["pairwise_tanimoto_mean"] for p in rows]
        ys = [p[y_key] * 100 for p in rows]

        # 参数扫描轨迹线
        ax.plot(
            xs,
            ys,
            style["line"],
            linewidth=2.0,
            alpha=0.65,
            label=style["name"] + " trajectory",
        )

        # 所有点
        ax.scatter(
            xs,
            ys,
            marker=style["marker"],
            s=95,
            alpha=0.85,
            label=style["name"] + " configs",
        )

        # 箭头表示 gamma 增大方向
        for i in range(len(xs) - 1):
            if abs(xs[i + 1] - xs[i]) < 1e-6 and abs(ys[i + 1] - ys[i]) < 1e-6:
                continue
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.45),
            )

    # Pareto 点用黑色空心圈，不遮挡原点
    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p[y_key] * 100 for p in pareto],
        marker="o",
        s=260,
        facecolors="none",
        edgecolors="black",
        linewidths=1.7,
        label="3-objective Pareto points",
    )

    annotate_grouped_labels(ax, points, y_key)

    ax.set_xlabel("Pairwise Tanimoto Mean ↓", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    # 说明理想区域
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    ax.annotate(
        "Better trade-off\nlower redundancy + higher score",
        xy=(x_min + 0.01 * (x_max - x_min), y_max - 0.05 * (y_max - y_min)),
        xytext=(x_min + 0.12 * (x_max - x_min), y_max - 0.18 * (y_max - y_min)),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=10,
    )

    ax.legend(fontsize=8, loc="best", frameon=True)
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

    draw_tradeoff_plot(
        points,
        "camc_flip_coverage",
        "CAMC Flip Coverage ↑ (%)",
        "Trade-off trajectory: redundancy vs CAMC flip coverage",
        out_dir / "trajectory_redundancy_vs_camc_flip",
    )

    draw_tradeoff_plot(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Trade-off trajectory: redundancy vs support coverage",
        out_dir / "trajectory_redundancy_vs_support",
    )

    print("Saved trajectory figures to:", out_dir)


if __name__ == "__main__":
    main()
