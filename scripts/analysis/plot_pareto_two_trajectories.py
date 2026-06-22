#!/usr/bin/env python3
"""Plot two CAMC Pareto trajectories from legacy evaluator outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_COLUMNS = ("family", "config", "eval_dir", "row_method")


@dataclass(frozen=True, slots=True)
class EvalPoint:
    """One resolved point from a legacy CAMC output directory."""

    family: str
    config: str
    eval_dir: Path
    row_method: str
    gamma: float | None
    label: str
    support_coverage: float
    camc_flip_coverage: float
    pairwise_tanimoto_mean: float
    raw_row: dict[str, str]
    pareto: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for Slurm wrapper parity.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for Slurm wrapper parity.",
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest: family,config,eval_dir,row_method")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--title-prefix", default="HIV label=1")
    return parser


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _parse_gamma(config: str, eval_dir: Path) -> float | None:
    text = f"{config} {'/'.join(eval_dir.parts[-4:])}"
    patterns = [
        r"(?:^|[_\-\s])g(?:amma)?[_\-]?(?P<value>\d+(?:p\d+)?(?:\.\d+)?)",
        r"gamma[_\-](?P<value>\d+(?:p\d+)?(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        return float(match.group("value").replace("p", "."))
    return None


def _family_prefix(family: str) -> str:
    lowered = family.lower()
    if "ours" in lowered or lowered.startswith("o"):
        return "O"
    if "baseline" in lowered or "gt" in lowered or lowered.startswith("b"):
        return "B"
    return family[:1].upper() or "?"


def _format_gamma(gamma: float | None, config: str) -> str:
    if gamma is None:
        return config
    if abs(gamma - round(gamma)) < 1e-9:
        return str(int(round(gamma)))
    return f"{gamma:g}"


def _load_camc_row(eval_dir: Path, row_method: str, k: int) -> dict[str, str]:
    table_path = eval_dir / "camc_comparison_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"missing camc_comparison_table.csv: {table_path}")
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        matches = [
            row
            for row in reader
            if str(row.get("method", "")).strip() == str(row_method)
            and int(float(row.get("k", "nan"))) == int(k)
        ]
    if not matches:
        raise ValueError(f"No row found in {table_path} for method={row_method!r}, k={k}")
    return matches[0]


def read_manifest(manifest: Path, k: int) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}; required={REQUIRED_MANIFEST_COLUMNS}")
        for row in reader:
            family = str(row["family"]).strip()
            config = str(row["config"]).strip()
            eval_dir = Path(row["eval_dir"]).expanduser()
            row_method = str(row["row_method"]).strip()
            camc_row = _load_camc_row(eval_dir, row_method, k)
            gamma = _parse_gamma(config, eval_dir)
            label = f"{_family_prefix(family)}-g{_format_gamma(gamma, config)}"
            support = _as_float(camc_row.get("support_coverage"))
            flip = _as_float(camc_row.get("camc_flip_coverage"))
            tanimoto = _as_float(camc_row.get("pairwise_tanimoto_mean"))
            if support is None or flip is None or tanimoto is None:
                raise ValueError(f"Missing CAMC metrics in {eval_dir}: {camc_row}")
            points.append(
                EvalPoint(
                    family=family,
                    config=config,
                    eval_dir=eval_dir,
                    row_method=row_method,
                    gamma=gamma,
                    label=label,
                    support_coverage=support,
                    camc_flip_coverage=flip,
                    pairwise_tanimoto_mean=tanimoto,
                    raw_row=camc_row,
                )
            )
    return points


def _dominates(left: EvalPoint, right: EvalPoint) -> bool:
    no_worse = (
        left.support_coverage >= right.support_coverage
        and left.camc_flip_coverage >= right.camc_flip_coverage
        and left.pairwise_tanimoto_mean <= right.pairwise_tanimoto_mean
    )
    strictly_better = (
        left.support_coverage > right.support_coverage
        or left.camc_flip_coverage > right.camc_flip_coverage
        or left.pairwise_tanimoto_mean < right.pairwise_tanimoto_mean
    )
    return bool(no_worse and strictly_better)


def mark_pareto(points: list[EvalPoint]) -> list[EvalPoint]:
    marked: list[EvalPoint] = []
    for point in points:
        dominated = any(_dominates(other, point) for other in points if other is not point)
        marked.append(
            EvalPoint(
                family=point.family,
                config=point.config,
                eval_dir=point.eval_dir,
                row_method=point.row_method,
                gamma=point.gamma,
                label=point.label,
                support_coverage=point.support_coverage,
                camc_flip_coverage=point.camc_flip_coverage,
                pairwise_tanimoto_mean=point.pairwise_tanimoto_mean,
                raw_row=point.raw_row,
                pareto=not dominated,
            )
        )
    return marked


def _sorted_family_points(points: list[EvalPoint], family: str) -> list[EvalPoint]:
    family_points = [point for point in points if point.family == family]
    return sorted(
        family_points,
        key=lambda point: (
            float("inf") if point.gamma is None else point.gamma,
            point.config,
            str(point.eval_dir),
        ),
    )


def _plot(points: list[EvalPoint], *, y_metric: str, y_label: str, output_stem: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    families = list(dict.fromkeys(point.family for point in points))
    colors = {
        families[0]: "#1f77b4" if families else "#1f77b4",
    }
    if len(families) > 1:
        colors[families[1]] = "#d62728"
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for family_index, family in enumerate(families):
        family_points = _sorted_family_points(points, family)
        x_values = [point.pairwise_tanimoto_mean for point in family_points]
        y_values = [float(getattr(point, y_metric)) for point in family_points]
        ax.plot(
            x_values,
            y_values,
            marker=markers[family_index % len(markers)],
            linewidth=1.8,
            markersize=5.5,
            color=colors.get(family, None),
            label=family,
        )
        for index, point in enumerate(family_points):
            offset_y = 7 if index % 2 == 0 else -11
            ax.annotate(
                point.label,
                (point.pairwise_tanimoto_mean, float(getattr(point, y_metric))),
                textcoords="offset points",
                xytext=(5, offset_y),
                fontsize=8,
                color=colors.get(family, "black"),
            )

    pareto_points = [point for point in points if point.pareto]
    if pareto_points:
        ax.scatter(
            [point.pairwise_tanimoto_mean for point in pareto_points],
            [float(getattr(point, y_metric)) for point in pareto_points],
            s=86,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="Pareto",
            zorder=5,
        )

    ax.set_xlabel("Pairwise Tanimoto Mean (lower is better)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_resolved_tables(points: list[EvalPoint], out_dir: Path) -> None:
    rows = [
        {
            "family": point.family,
            "config": point.config,
            "label": point.label,
            "gamma": "" if point.gamma is None else f"{point.gamma:g}",
            "eval_dir": str(point.eval_dir),
            "row_method": point.row_method,
            "support_coverage": point.support_coverage,
            "camc_flip_coverage": point.camc_flip_coverage,
            "pairwise_tanimoto_mean": point.pairwise_tanimoto_mean,
            "pareto": point.pareto,
        }
        for point in points
    ]
    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "resolved_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "pareto_points.json").write_text(
        json.dumps([row for row in rows if row["pareto"]], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    points = mark_pareto(read_manifest(manifest, int(args.k)))
    if not points:
        raise ValueError(f"No points resolved from manifest: {manifest}")
    write_resolved_tables(points, out_dir)
    _plot(
        points,
        y_metric="camc_flip_coverage",
        y_label="CAMC Flip Coverage (higher is better)",
        output_stem=out_dir / "pareto_tanimoto_vs_camc_flip",
        title=f"{args.title_prefix}: CAMC flip trajectory",
    )
    _plot(
        points,
        y_metric="support_coverage",
        y_label="Support Coverage (higher is better)",
        output_stem=out_dir / "pareto_tanimoto_vs_support",
        title=f"{args.title_prefix}: support trajectory",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "out_dir": str(out_dir),
                "num_points": len(points),
                "pareto_count": sum(1 for point in points if point.pareto),
                "outputs": [
                    str(out_dir / "pareto_tanimoto_vs_camc_flip.png"),
                    str(out_dir / "pareto_tanimoto_vs_camc_flip.pdf"),
                    str(out_dir / "pareto_tanimoto_vs_support.png"),
                    str(out_dir / "pareto_tanimoto_vs_support.pdf"),
                    str(out_dir / "resolved_points.csv"),
                    str(out_dir / "pareto_points.json"),
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
