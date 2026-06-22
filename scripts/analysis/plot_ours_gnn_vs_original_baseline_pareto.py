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


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_point(item, k=20):
    path = Path(item["eval_dir"]) / "camc_comparison_table.csv"
    method = item["row_method"]

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["method"] == method and int(float(r["k"])) == k:
                return {
                    "family": item["family"],
                    "config": item["config"],
                    "support_coverage": float(r["support_coverage"]),
                    "camc_flip_coverage": float(r["camc_flip_coverage"]),
                    "pairwise_tanimoto_mean": float(r["pairwise_tanimoto_mean"]),
                    "camc_at_05": float(r["camc_delta_0.5"]),
                    "mean_cf_drop_covered": float(r["mean_cf_drop_covered"]),
                    "gamma": parse_gamma(item["config"]),
                    "eval_dir": item["eval_dir"],
                    "row_method": method,
                }

    raise RuntimeError(f"No matched row in {path}: method={method}, k={k}")


def dominates(a, b):
    better_or_equal = (
        a["support_coverage"] >= b["support_coverage"]
        and a["camc_flip_coverage"] >= b["camc_flip_coverage"]
        and a["pairwise_tanimoto_mean"] <= b["pairwise_tanimoto_mean"]
    )
    strictly_better = (
        a["support_coverage"] > b["support_coverage"]
        or a["camc_flip_coverage"] > b["camc_flip_coverage"]
        or a["pairwise_tanimoto_mean"] < b["pairwise_tanimoto_mean"]
    )
    return better_or_equal and strictly_better


def mark_pareto(points):
    for i, p in enumerate(points):
        p["is_pareto"] = True
        for j, q in enumerate(points):
            if i == j:
                continue
            if dominates(q, p):
                p["is_pareto"] = False
                break
    return points


def write_points(points, out_csv):
    fields = [
        "family",
        "config",
        "support_coverage",
        "camc_flip_coverage",
        "pairwise_tanimoto_mean",
        "camc_at_05",
        "mean_cf_drop_covered",
        "is_pareto",
        "eval_dir",
        "row_method",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in points:
            row = {k: p[k] for k in fields}
            writer.writerow(row)


def label_for_point(p):
    if p["family"] == "Ours-MolCLR-GNN":
        g = p["gamma"]
        if g is None:
            return "Ours"
        if abs(g - int(g)) < 1e-9:
            return f"O-g{int(g)}"
        return f"O-g{g}"

    if p["config"] == "seed13":
        return "B-s13"
    if p["config"] == "seed21":
        return "B-s21"
    if p["config"] == "seed42":
        return "B-s42"
    return "B"


def plot_tradeoff(points, y_key, y_label, title, out_path, legend_loc):
    fig, ax = plt.subplots(figsize=(10, 6.2))

    ours = [p for p in points if p["family"] == "Ours-MolCLR-GNN"]
    baseline = [p for p in points if p["family"] == "Baseline-original"]

    ours = sorted(ours, key=lambda p: p["gamma"] if p["gamma"] is not None else 1e9)

    # Ours-MolCLR-GNN trajectory
    ax.plot(
        [p["pairwise_tanimoto_mean"] for p in ours],
        [p[y_key] * 100 for p in ours],
        linewidth=2.4,
        alpha=0.85,
        label="Ours-MolCLR-GNN",
    )
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in ours],
        [p[y_key] * 100 for p in ours],
        marker="s",
        s=95,
        alpha=0.9,
    )

    # Original baseline no-GNN seed points
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in baseline],
        [p[y_key] * 100 for p in baseline],
        marker="D",
        s=105,
        alpha=0.9,
        label="Baseline-original no-GNN",
    )

    # baseline seed mean, 仅作为视觉参考
    if baseline:
        bx = sum(p["pairwise_tanimoto_mean"] for p in baseline) / len(baseline)
        by = sum(p[y_key] for p in baseline) / len(baseline) * 100
        ax.scatter(
            [bx],
            [by],
            marker="X",
            s=160,
            alpha=0.95,
            label="Baseline mean",
        )
        ax.annotate(
            "B-mean",
            (bx, by),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
        )

    # Pareto points
    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p[y_key] * 100 for p in pareto],
        marker="o",
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
        label="3-objective Pareto points",
    )

    # Labels: Ours 合并低 gamma，baseline 标 seed
    ours_low = [p for p in ours if p["gamma"] in {5.0, 10.0, 20.0}]
    if ours_low:
        p = ours_low[-1]
        ax.annotate(
            "O-g5/10/20",
            (p["pairwise_tanimoto_mean"], p[y_key] * 100),
            textcoords="offset points",
            xytext=(-80, -18),
            fontsize=10,
        )

    for p in ours:
        if p["gamma"] in {50.0, 100.0}:
            ax.annotate(
                label_for_point(p),
                (p["pairwise_tanimoto_mean"], p[y_key] * 100),
                textcoords="offset points",
                xytext=(8, -18),
                fontsize=10,
            )

    for p in baseline:
        ax.annotate(
            label_for_point(p),
            (p["pairwise_tanimoto_mean"], p[y_key] * 100),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
        )

    xs = [p["pairwise_tanimoto_mean"] for p in points]
    ys = [p[y_key] * 100 for p in points]
    x_pad = (max(xs) - min(xs)) * 0.10 if max(xs) > min(xs) else 0.01
    y_pad = (max(ys) - min(ys)) * 0.13 if max(ys) > min(ys) else 0.2
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ax.set_xlabel("Pairwise Tanimoto Mean ↓", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, pad=15)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)

    ax.legend(fontsize=9, loc=legend_loc, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest)
    points = [load_point(item, k=args.k) for item in manifest]
    points = mark_pareto(points)

    out_csv = out_dir / "pareto_points_ours_gnn_vs_original_baseline.csv"
    write_points(points, out_csv)

    plot_tradeoff(
        points,
        "camc_flip_coverage",
        "CAMC Flip Coverage ↑ (%)",
        "Ours-MolCLR-GNN vs. Original Baseline: Redundancy vs. CAMC Flip",
        out_dir / "ours_gnn_vs_original_baseline_redundancy_vs_camc_flip",
        "lower right",
    )

    plot_tradeoff(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Ours-MolCLR-GNN vs. Original Baseline: Redundancy vs. Support",
        out_dir / "ours_gnn_vs_original_baseline_redundancy_vs_support",
        "upper right",
    )

    print("Saved CSV:", out_csv)
    print("Saved figures to:", out_dir)

    print("\nPareto frontier points:")
    for p in points:
        if p["is_pareto"]:
            print(
                f"{p['family']:22s} {p['config']:16s} "
                f"support={p['support_coverage']:.4f} "
                f"flip={p['camc_flip_coverage']:.4f} "
                f"tanimoto={p['pairwise_tanimoto_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
