"""Compare overlap between two selected fragment sets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.chem import parse_smiles
from src.data.hiv_dataset_utils import murcko_scaffold_smiles
from src.utils.io import ensure_directory

try:  # pragma: no cover - depends on runtime env
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - depends on runtime env
    DataStructs = None
    AllChem = None


@dataclass(frozen=True, slots=True)
class SelectedFragment:
    """One selected fragment normalized for overlap analysis."""

    fragment: str
    canonical_fragment: str
    scaffold: str
    raw_payload: dict[str, Any]


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonicalize_fragment(fragment: str) -> tuple[str, str]:
    normalized = str(fragment).strip()
    parsed = parse_smiles(normalized, sanitize=True, canonicalize=True)
    if parsed.parseable and parsed.sanitized and parsed.canonical_smiles:
        return str(parsed.canonical_smiles), murcko_scaffold_smiles(parsed.mol)
    return normalized, "ACYCLIC"


def _fingerprint(fragment: str) -> Any | None:
    if DataStructs is None or AllChem is None:
        return None
    parsed = parse_smiles(fragment, sanitize=True, canonicalize=True)
    if not parsed.parseable or not parsed.sanitized or parsed.mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(parsed.mol, 2, nBits=2048)


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_fp = _fingerprint(left)
    right_fp = _fingerprint(right)
    if left_fp is None or right_fp is None or DataStructs is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _load_selected_fragments(path: str | Path) -> list[SelectedFragment]:
    selected_path = Path(path).expanduser().resolve()
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"selected_subgraphs JSON must be a list: {selected_path}")
    fragments: list[SelectedFragment] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        fragment = _normalize_text(row.get("fragment"))
        if fragment is None:
            continue
        canonical_fragment, scaffold = _canonicalize_fragment(fragment)
        fragments.append(
            SelectedFragment(
                fragment=fragment,
                canonical_fragment=canonical_fragment,
                scaffold=scaffold,
                raw_payload=row,
            )
        )
    return fragments


def _pairwise_rows(
    label0_fragments: list[SelectedFragment],
    label1_fragments: list[SelectedFragment],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in label0_fragments:
        for right in label1_fragments:
            similarity = _similarity(left.canonical_fragment, right.canonical_fragment)
            rows.append(
                {
                    "label0_fragment": left.canonical_fragment,
                    "label1_fragment": right.canonical_fragment,
                    "similarity": similarity,
                    "exact_match": bool(left.canonical_fragment == right.canonical_fragment),
                    "label0_scaffold": left.scaffold,
                    "label1_scaffold": right.scaffold,
                    "same_scaffold": bool(left.scaffold == right.scaffold),
                }
            )
    rows.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return rows


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def render_overlap_report(summary: dict[str, Any]) -> str:
    """Render a concise overlap report."""

    exact = summary["exact_overlap"]
    soft = summary["soft_overlap"]
    scaffold = summary["scaffold_overlap"]
    lines = [
        "Selected Subgraph Overlap",
        f"label0_selected_json: {summary['metadata']['label0_selected_json']}",
        f"label1_selected_json: {summary['metadata']['label1_selected_json']}",
        "",
        "[exact_overlap]",
        f"label0_count={exact['label0_count']}",
        f"label1_count={exact['label1_count']}",
        f"exact_intersection_count={exact['exact_intersection_count']}",
        f"exact_jaccard={exact['exact_jaccard']:.4f}",
        f"label0_overlap_rate={exact['label0_overlap_rate']:.4f}",
        f"label1_overlap_rate={exact['label1_overlap_rate']:.4f}",
        f"exact_overlap_fragments={exact['exact_overlap_fragments']}",
        "",
        "[soft_overlap]",
        f"mean_max_sim_0_to_1={soft['mean_max_sim_0_to_1']}",
        f"mean_max_sim_1_to_0={soft['mean_max_sim_1_to_0']}",
        f"bidirectional_mean_max_sim={soft['bidirectional_mean_max_sim']}",
    ]
    for threshold_key, count in soft["count_sim_ge"].items():
        lines.append(f"count_sim_ge_{threshold_key}={count}")
    lines.extend(
        [
            "",
            "[scaffold_overlap]",
            f"label0_scaffold_count={scaffold['label0_scaffold_count']}",
            f"label1_scaffold_count={scaffold['label1_scaffold_count']}",
            f"scaffold_intersection_count={scaffold['scaffold_intersection_count']}",
            f"scaffold_jaccard={scaffold['scaffold_jaccard']:.4f}",
            f"overlapping_scaffolds={scaffold['overlapping_scaffolds']}",
            "",
        ]
    )
    return "\n".join(lines)


def compare_selected_subgraph_overlap(
    label0_selected_json: str | Path,
    label1_selected_json: str | Path,
    *,
    out_dir: str | Path,
    sim_thresholds: list[float],
) -> dict[str, Any]:
    """Compare exact, soft, and scaffold overlap between two selected sets."""

    label0 = _load_selected_fragments(label0_selected_json)
    label1 = _load_selected_fragments(label1_selected_json)
    label0_keys = sorted({item.canonical_fragment for item in label0})
    label1_keys = sorted({item.canonical_fragment for item in label1})
    exact_intersection = sorted(set(label0_keys).intersection(label1_keys))
    exact_union = sorted(set(label0_keys).union(label1_keys))

    pair_rows = _pairwise_rows(label0, label1)
    max_sim_0_to_1: list[float] = []
    for left in label0_keys:
        similarities = [row["similarity"] for row in pair_rows if row["label0_fragment"] == left]
        max_sim_0_to_1.append(max(similarities) if similarities else 0.0)
    max_sim_1_to_0: list[float] = []
    for right in label1_keys:
        similarities = [row["similarity"] for row in pair_rows if row["label1_fragment"] == right]
        max_sim_1_to_0.append(max(similarities) if similarities else 0.0)

    count_sim_ge = {
        str(threshold): int(sum(1 for row in pair_rows if float(row["similarity"]) >= float(threshold)))
        for threshold in sim_thresholds
    }
    pairs_above_threshold = {
        str(threshold): [
            row
            for row in pair_rows
            if float(row["similarity"]) >= float(threshold)
        ]
        for threshold in sim_thresholds
    }

    label0_scaffolds = sorted({item.scaffold for item in label0})
    label1_scaffolds = sorted({item.scaffold for item in label1})
    overlapping_scaffolds = sorted(set(label0_scaffolds).intersection(label1_scaffolds))

    summary = {
        "metadata": {
            "label0_selected_json": str(Path(label0_selected_json).expanduser().resolve()),
            "label1_selected_json": str(Path(label1_selected_json).expanduser().resolve()),
            "sim_thresholds": list(sim_thresholds),
            "rdkit_available": bool(DataStructs is not None and AllChem is not None),
        },
        "exact_overlap": {
            "label0_count": len(label0_keys),
            "label1_count": len(label1_keys),
            "exact_intersection_count": len(exact_intersection),
            "exact_jaccard": _rate(len(exact_intersection), len(exact_union)),
            "label0_overlap_rate": _rate(len(exact_intersection), len(label0_keys)),
            "label1_overlap_rate": _rate(len(exact_intersection), len(label1_keys)),
            "exact_overlap_fragments": exact_intersection,
        },
        "soft_overlap": {
            "mean_max_sim_0_to_1": _mean(max_sim_0_to_1),
            "mean_max_sim_1_to_0": _mean(max_sim_1_to_0),
            "bidirectional_mean_max_sim": _mean(max_sim_0_to_1 + max_sim_1_to_0),
            "count_sim_ge": count_sim_ge,
            "pairs_above_threshold": pairs_above_threshold,
        },
        "scaffold_overlap": {
            "label0_scaffold_count": len(label0_scaffolds),
            "label1_scaffold_count": len(label1_scaffolds),
            "scaffold_intersection_count": len(overlapping_scaffolds),
            "scaffold_jaccard": _rate(
                len(overlapping_scaffolds),
                len(set(label0_scaffolds).union(label1_scaffolds)),
            ),
            "overlapping_scaffolds": overlapping_scaffolds,
        },
    }

    output_dir = Path(out_dir).expanduser().resolve()
    ensure_directory(output_dir)
    json_path = output_dir / "overlap_summary.json"
    txt_path = output_dir / "overlap_report.txt"
    csv_path = output_dir / "overlap_pairs.csv"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt_path.write_text(render_overlap_report(summary), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label0_fragment",
                "label1_fragment",
                "similarity",
                "exact_match",
                "label0_scaffold",
                "label1_scaffold",
                "same_scaffold",
            ],
        )
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)
    return summary


__all__ = [
    "compare_selected_subgraph_overlap",
    "render_overlap_report",
]
