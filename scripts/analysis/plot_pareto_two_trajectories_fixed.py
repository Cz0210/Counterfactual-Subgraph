#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_MANIFEST_COLUMNS = ("family", "config", "eval_dir", "row_method")


def parse_gamma(config):
    m = re.search(r"gamma([0-9]+)p([0-9]+)", config)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def gamma_label(g):
    if g is None:
        return "NA"
    if abs(g - int(g)) < 1e-9:
        return str(int(g))
    return str(g)


def read_manifest(manifest_path):
    path = Path(manifest_path)
    with path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = "\t" if "\t" in first_line else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []

        missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}; "
                f"required={REQUIRED_MANIFEST_COLUMNS}; "
                f"fieldnames={fieldnames}; "
                f"detected_delimiter={repr(delimiter)}"
            )

        rows = list(reader)

    return rows


def load_camc_point(item, k):
    eval_dir = Path(item["eval_dir"])
    camc_path = eval_dir / "camc_comparison_table.csv"
    row_method = item["row_method"]

    if not camc_path.exists():
        raise FileNotFoundError(f"Missing {camc_path}")

    with camc_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("method") == row_method and int(float(r.get("k", -1))) == k:
                return {
                    "family": item["family"],
                    "config": item["config"],
                    "gamma": parse_gamma(item["config"]),
                    "eval_dir": str(eval_dir),
                    "row_method": row_method,
                    "support_coverage": float(r["support_coverage"]),
                    "camc_flip_coverage": float(r["camc_flip_coverage"]),
                    "pairwise_tanimoto_mean": float(r["pairwise_tanimoto_mean"]),
                    "camc_at_05": float(r["camc_delta_0.5"]),
                    "mean_cf_drop_covered": float(r["mean_cf_drop_covered"]),
                }

    raise RuntimeError(
        f"No matched k={k}, method={row_method} row in {camc_path}"
    )


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
            writer.writerow({c: p[c] for c in fields})


def short_method_label(family):
    if family.startswith("Ours"):
        return "O"
    if family.startswith("Baseline"):
        return "B"
    return family[:1]


def point_label(p):
    return f"{short_method_label(p['family'])}-g{gamma_label(p['gamma'])}"


def plot_tradeoff(points, y_key, y_label, title, out_prefix, legend_loc):
    fig, ax = plt.subplots(figsize=(10, 6.2))

    families = list(dict.fromkeys([p["family"] for p in points]))

    markers = ["s", "o", "D", "^", "P"]

    for idx, fam in enumerate(families):
        rows = [p for p in points if p["family"] == fam]
        rows = sorted(rows, key=lambda x: (x["gamma"] is None, x["gamma"] if x["gamma"] is not None else 1e9))

        xs = [p["pairwise_tanimoto_mean"] for p in rows]
        ys = [p[y_key] * 100 for p in rows]

        ax.plot(
            xs,
            ys,
            linewidth=2.2,
            alpha=0.78,
            label=fam,
        )

        ax.scatter(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            s=90,
            alpha=0.9,
        )

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

    # 标签：只标关键点，避免过密
    for p in points:
        g = p["gamma"]
        if g not in {0.0, 1.0, 2.0, 5.0, 20.0, 50.0, 100.0}:
            continue

        label = point_label(p)

        # Ours 的 5/10/20 经常重合，20 标一次即可
        if p["family"].startswith("Ours") and g in {5.0, 10.0}:
            continue
        if p["family"].startswith("Ours") and g == 20.0:
            label = "O-g5/10/20"

        if p["family"].startswith("Ours"):
            offset = (8, -18)
        else:
            offset = (8, 8)

        ax.annotate(
            label,
            (p["pairwise_tanimoto_mean"], p[y_key] * 100),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
        )

    xs = [p["pairwise_tanimoto_mean"] for p in points]
    ys = [p[y_key] * 100 for p in points]

    x_pad = (max(xs) - min(xs)) * 0.10 if max(xs) > min(xs) else 0.01
    y_pad = (max(ys) - min(ys)) * 0.12 if max(ys) > min(ys) else 0.2

    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ax.set_xlabel("Pairwise Tanimoto Mean ↓", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, pad=15)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax.legend(fontsize=9, loc=legend_loc, frameon=True)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(args.manifest)
    points = [load_camc_point(row, args.k) for row in manifest_rows]
    points = mark_pareto(points)

    out_csv = out_dir / "pareto_points.csv"
    write_points(points, out_csv)

    plot_tradeoff(
        points,
        "camc_flip_coverage",
        "CAMC Flip Coverage ↑ (%)",
        "Ours-MolCLR-GNN vs. Baseline-noGNN-Tanimoto: Redundancy vs. CAMC Flip",
        out_dir / "trajectory_redundancy_vs_camc_flip",
        "lower right",
    )

    plot_tradeoff(
        points,
        "support_coverage",
        "Support Coverage ↑ (%)",
        "Ours-MolCLR-GNN vs. Baseline-noGNN-Tanimoto: Redundancy vs. Support",
        out_dir / "trajectory_redundancy_vs_support",
        "upper right",
    )

    print("Saved CSV:", out_csv)
    print("Saved figures to:", out_dir)
    print("\nPareto frontier points:")
    for p in points:
        if p["is_pareto"]:
            print(
                f"{p['family']:28s} {p['config']:18s} "
                f"support={p['support_coverage']:.4f} "
                f"flip={p['camc_flip_coverage']:.4f} "
                f"tanimoto={p['pairwise_tanimoto_mean']:.4f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
