#!/usr/bin/env python3
"""Select GT-fullgraph action motifs with Morgan/Tanimoto redundancy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - depends on runtime env
    from rdkit import DataStructs
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - depends on runtime env
    DataStructs = None
    Chem = None
    rdFingerprintGenerator = None


DEFAULT_GT_MOTIF_POOL = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "comparison"
    / "hiv_quick"
    / "label1_1594411"
    / "camc_gt_fullgraph_motif_pool.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "selectors"
    / "gt_fullgraph_tanimoto_baseline_label1"
    / "beta_20p0_gamma_5p0"
)
FRAGMENT_FIELDS = (
    "final_fragment",
    "motif_smiles",
    "selected_fragment",
    "motif",
    "fragment",
    "final_fragment_smiles",
    "canonical_motif_smiles",
    "deleted_motif_smiles",
    "fragment_smiles",
    "core_fragment",
    "subgraph",
    "subgraph_smiles",
    "smiles",
)
SUPPORT_FIELDS = ("support_coverage", "support_rate", "coverage")
SUPPORT_COUNT_FIELDS = ("support_count",)
CF_FIELDS = (
    "camc_flip_coverage",
    "flip_rate",
    "mean_cf_drop_covered",
    "cf_drop",
    "fullgraph_cf_flip",
    "cf_flip",
    "flip",
)
ATOM_RATIO_FIELDS = ("motif_atom_ratio", "atom_ratio", "final_fragment_atom_ratio")
ATOM_COUNT_FIELDS = ("motif_atom_count", "atom_count", "fragment_atom_count")


@dataclass(slots=True)
class MotifAggregate:
    """Aggregated motif-level candidate for greedy MMR."""

    fragment: str
    source_fragments: set[str] = field(default_factory=set)
    source_rows: list[dict[str, Any]] = field(default_factory=list)
    support_ids: set[str] = field(default_factory=set)
    parent_examples: list[str] = field(default_factory=list)
    cf_values: list[float] = field(default_factory=list)
    explicit_support_values: list[float] = field(default_factory=list)
    explicit_support_counts: list[float] = field(default_factory=list)
    atom_ratios: list[float] = field(default_factory=list)
    atom_counts: list[float] = field(default_factory=list)
    fingerprint: Any | None = None

    support_score: float = 0.0
    cf_score: float = 0.0
    size_penalty: float = 0.0
    motif_atom_count: float | None = None
    motif_atom_ratio: float | None = None


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
    parser.add_argument("--gt-motif-pool", default=str(DEFAULT_GT_MOTIF_POOL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha-cf", type=float, default=0.8)
    parser.add_argument("--beta-coverage", type=float, default=20.0)
    parser.add_argument("--gamma-redundancy", type=float, default=5.0)
    parser.add_argument("--eta-size", type=float, default=0.3)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "na"}:
        return None
    if text.lower() in {"true", "yes", "y"}:
        return 1.0
    if text.lower() in {"false", "no", "n"}:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _as_bool(value: Any) -> bool | None:
    numeric = _as_float(value)
    if numeric is not None:
        return bool(numeric)
    return None


def _first_text(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str | None, str | None]:
    for field_name in fields:
        value = row.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, field_name
    return None, None


def _first_float(row: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    for field_name in fields:
        value = _as_float(row.get(field_name))
        if value is not None:
            return value
    return None


def _canonicalize_smiles(smiles: str) -> str | None:
    if Chem is None:
        raise RuntimeError("RDKit is required for GT-fullgraph Tanimoto baseline selection.")
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() <= 0:
        return None
    try:
        return str(Chem.MolToSmiles(mol, canonical=True))
    except Exception:
        return str(smiles).strip()


def _get_morgan_generator() -> Any:
    if rdFingerprintGenerator is None:
        raise RuntimeError("RDKit rdFingerprintGenerator is required for Morgan/Tanimoto similarity.")
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fingerprint(smiles: str, generator: Any) -> Any | None:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return generator.GetFingerprint(mol)
    except Exception:
        return None


def _tanimoto(left: MotifAggregate, right: MotifAggregate) -> float:
    if left.fragment == right.fragment:
        return 1.0
    if DataStructs is None or left.fingerprint is None or right.fingerprint is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left.fingerprint, right.fingerprint))


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def load_gt_motif_pool(path: Path) -> tuple[list[MotifAggregate], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"GT motif pool not found: {path}")
    if DataStructs is None or Chem is None or rdFingerprintGenerator is None:
        raise RuntimeError("RDKit is required for Morgan/Tanimoto baseline selection.")

    generator = _get_morgan_generator()
    by_fragment: dict[str, MotifAggregate] = {}
    total_rows = 0
    skipped_rows = 0
    invalid_motif_rows = 0
    fragment_field_counts: dict[str, int] = {}
    support_universe: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row_index, row in enumerate(reader):
            total_rows += 1
            valid_motif = _as_bool(row.get("valid_motif"))
            if valid_motif is False:
                invalid_motif_rows += 1
                skipped_rows += 1
                continue
            fragment_raw, fragment_field = _first_text(row, FRAGMENT_FIELDS)
            if not fragment_raw:
                skipped_rows += 1
                continue
            canonical = _canonicalize_smiles(fragment_raw)
            if not canonical:
                invalid_motif_rows += 1
                skipped_rows += 1
                continue

            support_id = str(
                row.get("input_idx")
                or row.get("parent_smiles")
                or row.get("source_row_index")
                or row_index
            )
            support_universe.add(support_id)

            aggregate = by_fragment.get(canonical)
            if aggregate is None:
                aggregate = MotifAggregate(
                    fragment=canonical,
                    fingerprint=_fingerprint(canonical, generator),
                )
                by_fragment[canonical] = aggregate
            aggregate.source_fragments.add(fragment_raw)
            aggregate.source_rows.append(row)
            aggregate.support_ids.add(support_id)
            parent = str(row.get("parent_smiles") or "").strip()
            if parent and len(aggregate.parent_examples) < 3 and parent not in aggregate.parent_examples:
                aggregate.parent_examples.append(parent)

            support_value = _first_float(row, SUPPORT_FIELDS)
            if support_value is not None:
                aggregate.explicit_support_values.append(support_value)
            support_count = _first_float(row, SUPPORT_COUNT_FIELDS)
            if support_count is not None:
                aggregate.explicit_support_counts.append(support_count)
            cf_value = _first_float(row, CF_FIELDS)
            if cf_value is not None:
                aggregate.cf_values.append(cf_value)
            atom_ratio = _first_float(row, ATOM_RATIO_FIELDS)
            if atom_ratio is not None:
                aggregate.atom_ratios.append(atom_ratio)
            atom_count = _first_float(row, ATOM_COUNT_FIELDS)
            if atom_count is not None:
                aggregate.atom_counts.append(atom_count)
            if fragment_field:
                fragment_field_counts[fragment_field] = fragment_field_counts.get(fragment_field, 0) + 1

    denominator = max(1, len(support_universe))
    max_atom_count = max(
        [max(candidate.atom_counts) for candidate in by_fragment.values() if candidate.atom_counts],
        default=1.0,
    )

    candidates: list[MotifAggregate] = []
    for candidate in by_fragment.values():
        if candidate.explicit_support_values:
            candidate.support_score = float(max(candidate.explicit_support_values))
        elif candidate.explicit_support_counts:
            candidate.support_score = float(max(candidate.explicit_support_counts)) / float(denominator)
        else:
            candidate.support_score = float(len(candidate.support_ids)) / float(denominator)
        candidate.cf_score = float(_safe_mean(candidate.cf_values) or 0.0)
        candidate.motif_atom_count = _safe_mean(candidate.atom_counts)
        candidate.motif_atom_ratio = _safe_mean(candidate.atom_ratios)
        if candidate.motif_atom_ratio is not None:
            candidate.size_penalty = max(0.0, float(candidate.motif_atom_ratio))
        elif candidate.motif_atom_count is not None:
            candidate.size_penalty = max(0.0, float(candidate.motif_atom_count) / float(max_atom_count))
        else:
            candidate.size_penalty = 0.30
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.support_score, -item.cf_score, item.fragment))
    diagnostics = {
        "fieldnames": fieldnames,
        "total_rows": total_rows,
        "skipped_rows": skipped_rows,
        "invalid_motif_rows": invalid_motif_rows,
        "unique_motif_count": len(candidates),
        "support_denominator": denominator,
        "fragment_field_counts": fragment_field_counts,
    }
    return candidates, diagnostics


def greedy_select(
    candidates: list[MotifAggregate],
    *,
    top_k: int,
    alpha_cf: float,
    beta_coverage: float,
    gamma_redundancy: float,
    eta_size: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    remaining = list(candidates)
    selected: list[MotifAggregate] = []
    selected_rows: list[dict[str, Any]] = []
    covered_support: set[str] = set()
    denominator = max(1, len(set().union(*(candidate.support_ids for candidate in candidates))) if candidates else 1)

    while remaining and len(selected) < int(top_k):
        best_candidate: MotifAggregate | None = None
        best_payload: dict[str, Any] | None = None
        best_key: tuple[float, float, float, str] | None = None

        for candidate in remaining:
            coverage_gain = float(len(candidate.support_ids - covered_support)) / float(denominator)
            max_similarity = max((_tanimoto(candidate, item) for item in selected), default=0.0)
            score = (
                float(alpha_cf) * candidate.cf_score
                + float(beta_coverage) * coverage_gain
                - float(gamma_redundancy) * max_similarity
                - float(eta_size) * candidate.size_penalty
            )
            payload = {
                "rank": len(selected) + 1,
                "fragment": candidate.fragment,
                "final_fragment": candidate.fragment,
                "core_fragment": candidate.fragment,
                "selected_fragment": candidate.fragment,
                "score": score,
                "cf_score": candidate.cf_score,
                "support_count": len(candidate.support_ids),
                "support_rate": candidate.support_score,
                "support_score": candidate.support_score,
                "coverage_gain": coverage_gain,
                "cumulative_coverage": float(len(covered_support | candidate.support_ids)) / float(denominator),
                "size_penalty": candidate.size_penalty,
                "motif_atom_count": candidate.motif_atom_count,
                "motif_atom_ratio": candidate.motif_atom_ratio,
                "max_similarity_to_previous": max_similarity,
                "representative_parent_smiles": "|".join(candidate.parent_examples),
                "source_row_count": len(candidate.source_rows),
                "source_fragments": "|".join(sorted(candidate.source_fragments)[:8]),
            }
            key = (score, coverage_gain, candidate.cf_score, candidate.fragment)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate
                best_payload = payload

        assert best_candidate is not None
        assert best_payload is not None
        selected.append(best_candidate)
        selected_rows.append(best_payload)
        covered_support |= best_candidate.support_ids
        remaining = [candidate for candidate in remaining if candidate.fragment != best_candidate.fragment]

    pairwise_values: list[float] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            pairwise_values.append(_tanimoto(left, right))
    selection_stats = {
        "selected_count": len(selected),
        "final_cumulative_coverage": float(len(covered_support)) / float(denominator),
        "selected_pairwise_tanimoto_mean": _safe_mean(pairwise_values),
        "selected_pairwise_tanimoto_max": max(pairwise_values) if pairwise_values else None,
        "selected_mean_cf_score": _safe_mean([candidate.cf_score for candidate in selected]),
        "selected_mean_size_penalty": _safe_mean([candidate.size_penalty for candidate in selected]),
        "support_denominator": denominator,
    }
    return selected_rows, selection_stats


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "fragment",
        "final_fragment",
        "core_fragment",
        "selected_fragment",
        "score",
        "cf_score",
        "support_count",
        "support_rate",
        "support_score",
        "coverage_gain",
        "cumulative_coverage",
        "size_penalty",
        "motif_atom_count",
        "motif_atom_ratio",
        "max_similarity_to_previous",
        "representative_parent_smiles",
        "source_row_count",
        "source_fragments",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GT Fullgraph Tanimoto Baseline Selector",
        "",
        f"gt_motif_pool: {summary['run_config']['gt_motif_pool']}",
        f"target_label: {summary['run_config']['target_label']}",
        f"top_k: {summary['run_config']['top_k']}",
        f"alpha_cf: {summary['run_config']['alpha_cf']}",
        f"beta_coverage: {summary['run_config']['beta_coverage']}",
        f"gamma_redundancy: {summary['run_config']['gamma_redundancy']}",
        f"eta_size: {summary['run_config']['eta_size']}",
        f"selected_count: {summary['selected_count']}",
        f"final_cumulative_coverage: {summary['final_cumulative_coverage']}",
        f"selected_pairwise_tanimoto_mean: {summary['selected_pairwise_tanimoto_mean']}",
        "",
        "## Selected motifs",
    ]
    for row in rows:
        lines.append(
            "rank={rank} fragment={fragment} score={score:.6g} "
            "support={support_rate:.6g} cf={cf_score:.6g} "
            "max_tanimoto={max_similarity_to_previous:.6g}".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    gt_motif_pool = Path(args.gt_motif_pool).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, diagnostics = load_gt_motif_pool(gt_motif_pool)
    selected_rows, selection_stats = greedy_select(
        candidates,
        top_k=int(args.top_k),
        alpha_cf=float(args.alpha_cf),
        beta_coverage=float(args.beta_coverage),
        gamma_redundancy=float(args.gamma_redundancy),
        eta_size=float(args.eta_size),
    )

    run_config = {
        "gt_motif_pool": str(gt_motif_pool),
        "out_dir": str(out_dir),
        "top_k": int(args.top_k),
        "alpha_cf": float(args.alpha_cf),
        "beta_coverage": float(args.beta_coverage),
        "gamma_redundancy": float(args.gamma_redundancy),
        "eta_size": float(args.eta_size),
        "target_label": int(args.target_label),
        "seed": int(args.seed),
        "sim_metric": "morgan_tanimoto",
    }
    summary = {
        "method": "gt_fullgraph_tanimoto_baseline",
        "run_config": run_config,
        "metadata": {
            "alpha_cf": float(args.alpha_cf),
            "beta_coverage": float(args.beta_coverage),
            "gamma_redundancy": float(args.gamma_redundancy),
            "eta_size": float(args.eta_size),
            "target_label": int(args.target_label),
        },
        "selected_fragments": [row["fragment"] for row in selected_rows],
        **selection_stats,
        "input_diagnostics": diagnostics,
        "outputs": {
            "selected_subgraphs_csv": str(out_dir / "selected_subgraphs.csv"),
            "selected_subgraphs_json": str(out_dir / "selected_subgraphs.json"),
            "selector_summary_json": str(out_dir / "selector_summary.json"),
            "selector_report_txt": str(out_dir / "selector_report.txt"),
        },
    }

    _write_csv(out_dir / "selected_subgraphs.csv", selected_rows)
    (out_dir / "selected_subgraphs.json").write_text(
        json.dumps(selected_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selector_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir / "selector_report.txt", summary, selected_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
