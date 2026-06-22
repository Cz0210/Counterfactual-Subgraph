#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_point(item, k=20):
    path = Path(item["eval_dir"]) / "camc_comparison_table.csv"
    method = item.get("row_method", "ours_selected_subgraph")

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
                    "eval_dir": item["eval_dir"],
                }

    raise RuntimeError(f"No row found: {path}, method={method}, k={k}")


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


def write_csv(points, out_csv):
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
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow(p)


def plot_3d(points, out_prefix):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    families = sorted(set(p["family"] for p in points))
    markers = ["o", "s", "^", "D", "P"]

    for idx, fam in enumerate(families):
        rows = [p for p in points if p["family"] == fam]
        ax.scatter(
            [p["pairwise_tanimoto_mean"] for p in rows],
            [p["support_coverage"] * 100 for p in rows],
            [p["camc_flip_coverage"] * 100 for p in rows],
            marker=markers[idx % len(markers)],
            s=75,
            alpha=0.75,
            label=fam,
        )

    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p["support_coverage"] * 100 for p in pareto],
        [p["camc_flip_coverage"] * 100 for p in pareto],
        marker="*",
        s=240,
        alpha=0.95,
        label="Pareto frontier",
    )

    for p in points:
        ax.text(
            p["pairwise_tanimoto_mean"],
            p["support_coverage"] * 100,
            p["camc_flip_coverage"] * 100,
            p["config"].replace("beta20_", ""),
            fontsize=7,
        )

    ax.set_xlabel("Pairwise Tanimoto Mean ↓")
    ax.set_ylabel("Support Coverage ↑ (%)")
    ax.set_zlabel("CAMC Flip Coverage ↑ (%)")
    ax.set_title("3-objective Pareto frontier under legacy protocol")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=300)
    fig.savefig(str(out_prefix) + ".pdf")
    plt.close(fig)


def plot_2d(points, y_key, y_label, title, out_prefix):
    fig, ax = plt.subplots(figsize=(9, 6))

    families = sorted(set(p["family"] for p in points))
    markers = ["o", "s", "^", "D", "P"]

    for idx, fam in enumerate(families):
        rows = [p for p in points if p["family"] == fam]
        ax.scatter(
            [p["pairwise_tanimoto_mean"] for p in rows],
            [p[y_key] * 100 for p in rows],
            marker=markers[idx % len(markers)],
            s=80,
            alpha=0.75,
            label=fam,
        )

    pareto = [p for p in points if p["is_pareto"]]
    ax.scatter(
        [p["pairwise_tanimoto_mean"] for p in pareto],
        [p[y_key] * 100 for p in pareto],
        marker="*",
        s=240,
        alpha=0.95,
        label="Pareto frontier",
    )

    for p in points:
        ax.annotate(
            p["config"].replace("beta20_", ""),
            (p["pairwise_tanimoto_mean"], p[y_key] * 100),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Pairwise Tanimoto Mean ↓")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=300)
    fig.savefig(str(out_prefix) + ".pdf")
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

    out_csv = out_dir / "pareto_points_baseline_vs_ours_gnn.csv"
    write_csv(points, out_csv)

    plot_3d(points, out_dir / "pareto_3obj_3d")

    plot_2d(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Pareto projection: redundancy vs support coverage",
        out_dir / "pareto_projection_redundancy_support",
    )

    plot_2d(
        points,
        "camc_flip_coverage",
        "CAMC Flip Coverage ↑ (%)",
        "Pareto projection: redundancy vs CAMC flip coverage",
        out_dir / "pareto_projection_redundancy_camc_flip",
    )

    print(f"Saved CSV: {out_csv}")
    print(f"Saved figures to: {out_dir}")

    print("\nPareto frontier points:")
    for p in points:
        if p["is_pareto"]:
            print(
                f"{p['family']:22s} {p['config']:18s} "
                f"support={p['support_coverage']:.4f} "
                f"flip={p['camc_flip_coverage']:.4f} "
                f"tanimoto={p['pairwise_tanimoto_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
