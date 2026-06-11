#!/usr/bin/env python3
"""Sanity-check selected fragment sets under one evaluation protocol.

This script compares old Morgan-selected and new embedding-selected top20 sets
using the same candidate-pool evidence evaluator. It is intended to explain
whether metric drops come from selection changes or from comparing different
evaluation protocols.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_embedding_selector_camc_table import (  # noqa: E402
    DEFAULT_OURS_POOL,
    FRAGMENT_KEYS,
    PARENT_KEYS,
    _as_bool,
    _as_float,
    _first_text,
    _fmt,
    _load_dataset_denominator,
    _pick_representative_rows,
    evaluate_selector,
    load_selected_fragments,
)
from src.eval.candidate_pool_audit import _canonical_fragment_key  # noqa: E402
from src.eval.subgraph_similarity import get_candidate_embedding  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402


DEFAULT_OUT_DIR = Path("outputs/hpc/comparison/hiv_quick/embedding_selector_sanity_check_label1")
LEGACY_CAMC_LOCATOR = {
    "script": "scripts/eval/compare_hiv_recourse_baselines.py",
    "slurm": "scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh",
    "doc": "docs/baselines/hiv_quick_recourse_comparison.md",
    "primary_outputs": [
        "camc_comparison_table.csv",
        "camc_summary.json",
        "camc_per_input.csv",
        "camc_ours_action_motifs.csv",
        "camc_gt_fullgraph_motif_pool.csv",
        "camc_gt_fullgraph_selected_motifs.csv",
    ],
    "main_inputs": [
        "--hiv-csv",
        "--teacher-path",
        "--ours-selected-dir",
        "--target-label",
        "--top-k-list",
        "--theta-list",
    ],
    "metric_function": "build_camc_table_and_per_input",
    "motif_evaluator": "evaluate_action_motifs",
    "coverage_source": "legacy_camc_evaluator_requires_teacher_and_full_dataset",
}
TABLE_COLUMNS = [
    "selected_set",
    "coverage_source",
    "selected_count",
    "candidate_rows_matched",
    "num_inputs",
    "support_coverage",
    "camc_flip_coverage",
    "camc_at_0p5",
    "mean_cf_drop_covered",
    "pairwise_embedding_cosine_mean",
    "pairwise_embedding_cosine_max",
    "pairwise_tanimoto_mean",
    "pairwise_tanimoto_max",
    "coverage_theta_0p20",
    "theta_coverage_source",
    "selector_dir",
]


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
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.20)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--selected-set",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Selected-set selector directory. Repeatable.",
    )
    parser.add_argument(
        "--candidate-pool-jsonl",
        default=str(DEFAULT_OURS_POOL),
        help="Candidate pool evidence used for all selected sets.",
    )
    parser.add_argument("--embedding-field", default="final_fragment_embedding")
    return parser


def _parse_selected_set_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--selected-set must have NAME=PATH format, got: {spec}")
        name, path_text = spec.split("=", 1)
        name = name.strip()
        path_text = path_text.strip()
        if not name or not path_text:
            raise ValueError(f"--selected-set must have non-empty NAME and PATH, got: {spec}")
        if name in seen:
            raise ValueError(f"Duplicate selected-set name: {name}")
        seen.add(name)
        parsed.append((name, Path(path_text).expanduser()))
    return parsed


def _load_selector_rows(selector_dir: Path) -> dict[str, dict[str, Any]]:
    selector_dir = selector_dir.expanduser()
    rows_by_key: dict[str, dict[str, Any]] = {}

    selected_json = selector_dir / "selected_subgraphs.json"
    if selected_json.exists():
        payload = json.loads(selected_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for index, row in enumerate(payload, start=1):
                if not isinstance(row, dict):
                    continue
                fragment = _first_text(row, FRAGMENT_KEYS)
                key = _canonical_fragment_key(fragment)
                if key is None:
                    continue
                enriched = dict(row)
                enriched.setdefault("rank", index)
                rows_by_key[key] = enriched

    summary_json = selector_dir / "selector_summary.json"
    if summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("selected_fragments"), list):
            for index, fragment in enumerate(payload["selected_fragments"], start=1):
                key = _canonical_fragment_key(str(fragment))
                if key is None or key in rows_by_key:
                    continue
                rows_by_key[key] = {
                    "rank": index,
                    "fragment": fragment,
                    "source": str(summary_json),
                }
    return rows_by_key


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fragment_evidence_dump(
    *,
    selected_fragments: list[str],
    selector_rows: dict[str, dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    selected_keys: set[str],
    num_inputs: int,
    embedding_field: str,
) -> list[dict[str, Any]]:
    rows_by_fragment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        fragment = _first_text(row, FRAGMENT_KEYS)
        key = _canonical_fragment_key(fragment)
        if key is not None and key in selected_keys:
            rows_by_fragment[key].append(row)

    dump: list[dict[str, Any]] = []
    previous_embeddings: list[Any] = []
    for index, fragment in enumerate(selected_fragments, start=1):
        key = _canonical_fragment_key(fragment)
        evidence_rows = rows_by_fragment.get(key or "", [])
        parents = {
            parent
            for row in evidence_rows
            if (parent := _first_text(row, PARENT_KEYS)) is not None
        }
        flip_parents = {
            parent
            for row in evidence_rows
            if (parent := _first_text(row, PARENT_KEYS)) is not None and bool(_as_bool(row.get("cf_flip")))
        }
        atom_counts = [
            value
            for row in evidence_rows
            if (value := _as_float(row.get("atom_count"))) is not None
        ]
        atom_ratios = [
            value
            for row in evidence_rows
            if (value := _as_float(row.get("atom_ratio"))) is not None
        ]
        cf_drops = [
            value
            for row in evidence_rows
            if (value := _as_float(row.get("cf_drop"))) is not None
        ]
        rep_embedding = None
        for row in evidence_rows:
            try:
                rep_embedding = get_candidate_embedding(row, embedding_field).vector
                break
            except Exception:
                continue
        pairwise_to_previous: float | None = None
        if rep_embedding is not None and previous_embeddings:
            from src.eval.subgraph_similarity import cosine_embedding_similarity

            similarities: list[float] = []
            for previous in previous_embeddings:
                try:
                    similarities.append(cosine_embedding_similarity(rep_embedding, previous))
                except Exception:
                    continue
            pairwise_to_previous = max(similarities) if similarities else None
        if rep_embedding is not None:
            previous_embeddings.append(rep_embedding)

        selector_row = selector_rows.get(key or "", {})
        support_rate = float(len(parents)) / float(num_inputs) if num_inputs else 0.0
        dump.append(
            {
                "rank": int(selector_row.get("rank") or index),
                "final_fragment": fragment,
                "canonical_fragment": key,
                "atom_count": _safe_mean(atom_counts),
                "atom_ratio": _safe_mean(atom_ratios),
                "candidate_evidence_row_count": len(evidence_rows),
                "candidate_evidence_support_count": len(parents),
                "candidate_evidence_support_rate": support_rate,
                "candidate_evidence_flip_parent_count": len(flip_parents),
                "candidate_evidence_cf_flip_rate": (
                    float(len(flip_parents)) / float(len(parents)) if parents else None
                ),
                "candidate_evidence_cf_drop_mean": _safe_mean(cf_drops),
                "candidate_evidence_cf_drop_max": max(cf_drops) if cf_drops else None,
                "embedding_redundancy_to_previous": pairwise_to_previous,
                "selector_score": _as_float(selector_row.get("score") or selector_row.get("mmr_score")),
                "selector_coverage_gain": _as_float(selector_row.get("coverage_gain")),
                "selector_support_count": _as_float(selector_row.get("support_count")),
                "selector_support_rate": _as_float(selector_row.get("support_rate")),
                "selector_mean_cf_drop": _as_float(selector_row.get("mean_cf_drop")),
                "selector_mean_atom_ratio": _as_float(selector_row.get("mean_atom_ratio")),
                "selector_max_similarity_to_previous": _as_float(
                    selector_row.get("max_similarity_to_previous")
                    or selector_row.get("max_redundancy_sim_at_selection")
                ),
                "selector_redundancy_sim_metric": selector_row.get("redundancy_sim_metric"),
            }
        )
    return sorted(dump, key=lambda row: int(row["rank"]))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in TABLE_COLUMNS})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "selected_set",
        "coverage_source",
        "support_coverage",
        "camc_flip_coverage",
        "camc_at_0p5",
        "mean_cf_drop_covered",
        "pairwise_embedding_cosine_mean",
        "pairwise_tanimoto_mean",
        "coverage_theta_0p20",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_interpretation(rows: list[dict[str, Any]]) -> list[str]:
    by_name = {str(row["selected_set"]): row for row in rows}
    old = by_name.get("old_morgan")
    embeddings = [row for row in rows if str(row["selected_set"]).startswith("embedding")]
    lines = [
        "How to read this sanity check:",
        "1. If old_morgan is also much lower here than in the legacy CAMC table, the old-vs-new drop is mostly protocol/evidence-source driven (B/C).",
        "2. If old_morgan stays near the old CAMC numbers but embedding sets drop, the embedding selector truly traded off coverage (A).",
        "3. If candidate evidence support is much lower than full CAMC support, this table is a lower bound tied to generated candidate-pool rows (C).",
        "4. GT cf_drop proxy issues are not tested unless a GT selected set is included; GT proxy rows should not be interpreted as teacher-recomputed CF strength (D).",
    ]
    if old and embeddings:
        old_support = _as_float(old.get("support_coverage"))
        old_flip = _as_float(old.get("camc_flip_coverage"))
        for row in embeddings:
            support = _as_float(row.get("support_coverage"))
            flip = _as_float(row.get("camc_flip_coverage"))
            lines.append(
                "- {name}: support_delta_vs_old={support_delta} flip_delta_vs_old={flip_delta}".format(
                    name=row.get("selected_set"),
                    support_delta=_fmt(support - old_support) if support is not None and old_support is not None else "",
                    flip_delta=_fmt(flip - old_flip) if flip is not None and old_flip is not None else "",
                )
            )
    return lines


def _render_report(
    *,
    rows: list[dict[str, Any]],
    dataset_metadata: dict[str, Any],
    candidate_pool: Path,
) -> str:
    lines = [
        "Embedding Selector Sanity Check",
        "",
        "Legacy CAMC table locator:",
        f"- script: {LEGACY_CAMC_LOCATOR['script']}",
        f"- slurm: {LEGACY_CAMC_LOCATOR['slurm']}",
        f"- metric function: {LEGACY_CAMC_LOCATOR['metric_function']}",
        f"- motif evaluator: {LEGACY_CAMC_LOCATOR['motif_evaluator']}",
        "- old CAMC protocol: full target dataset + RF teacher deletion evaluation",
        "",
        "This sanity-check protocol:",
        "- coverage_source: candidate_pool_evidence",
        f"- candidate_pool_jsonl: {candidate_pool}",
        f"- dataset_denominator: {dataset_metadata.get('denominator')}",
        f"- denominator_source: {dataset_metadata.get('denominator_source')}",
        "",
        "Rows:",
    ]
    for row in rows:
        lines.append(
            "- {name}: support={support} flip={flip} camc@0.5={delta} "
            "cf_drop={cf_drop} emb_mean={emb} tanimoto_mean={tan} theta_cov={theta}".format(
                name=row.get("selected_set"),
                support=_fmt(row.get("support_coverage")),
                flip=_fmt(row.get("camc_flip_coverage")),
                delta=_fmt(row.get("camc_at_0p5")),
                cf_drop=_fmt(row.get("mean_cf_drop_covered")),
                emb=_fmt(row.get("pairwise_embedding_cosine_mean")),
                tan=_fmt(row.get("pairwise_tanimoto_mean")),
                theta=_fmt(row.get("coverage_theta_0p20")),
            )
        )
    lines.extend(["", *_build_interpretation(rows)])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = _parse_selected_set_specs(args.selected_set)
    candidate_pool = Path(args.candidate_pool_jsonl).expanduser()
    if not candidate_pool.exists():
        raise FileNotFoundError(f"candidate pool JSONL not found: {candidate_pool}")
    pool_rows = read_jsonl(candidate_pool)
    num_inputs, dataset_metadata = _load_dataset_denominator(
        Path(args.dataset_path),
        int(args.label),
    )

    table_rows: list[dict[str, Any]] = []
    selected_dump: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}

    for name, selector_dir in selected_specs:
        selected_fragments, discovered = load_selected_fragments(selector_dir)
        selected_keys = {
            key for fragment in selected_fragments if (key := _canonical_fragment_key(fragment)) is not None
        }
        selector_rows = _load_selector_rows(selector_dir)
        _reps, matched_rows = _pick_representative_rows(
            selected_keys,
            pool_rows,
            label=int(args.label),
        )
        eval_row, eval_diag = evaluate_selector(
            method=name,
            seed="selected_set",
            selector_dir=selector_dir,
            pool_jsonl=candidate_pool,
            num_inputs=num_inputs,
            label=int(args.label),
            theta=float(args.theta),
            embedding_field=str(args.embedding_field),
            beta=None,
            gamma=None,
        )
        row = {
            "selected_set": name,
            "coverage_source": "candidate_pool_evidence",
            "selected_count": eval_row.get("selected_count"),
            "candidate_rows_matched": eval_row.get("candidate_rows_matched"),
            "num_inputs": eval_row.get("num_inputs"),
            "support_coverage": eval_row.get("support_coverage"),
            "camc_flip_coverage": eval_row.get("camc_flip_coverage"),
            "camc_at_0p5": eval_row.get("camc_at_0p5"),
            "mean_cf_drop_covered": eval_row.get("mean_cf_drop_covered"),
            "pairwise_embedding_cosine_mean": eval_row.get("pairwise_embedding_cosine_mean"),
            "pairwise_embedding_cosine_max": eval_row.get("pairwise_embedding_cosine_max"),
            "pairwise_tanimoto_mean": eval_row.get("pairwise_tanimoto_mean"),
            "pairwise_tanimoto_max": eval_row.get("pairwise_tanimoto_max"),
            "coverage_theta_0p20": eval_row.get("coverage_theta_0p20"),
            "theta_coverage_source": eval_row.get("theta_coverage_source"),
            "selector_dir": str(selector_dir),
        }
        table_rows.append(row)
        selected_dump[name] = {
            "selector_dir": str(selector_dir),
            "selector_files": discovered,
            "selected_fragments": _fragment_evidence_dump(
                selected_fragments=selected_fragments,
                selector_rows=selector_rows,
                pool_rows=pool_rows,
                selected_keys=selected_keys,
                num_inputs=num_inputs,
                embedding_field=str(args.embedding_field),
            ),
        }
        diagnostics[name] = {
            **eval_diag,
            "matched_rows_via_sanity": len(matched_rows),
            "selected_key_count": len(selected_keys),
        }

    output_paths = {
        "sanity_check_table_tsv": str(out_dir / "sanity_check_table.tsv"),
        "sanity_check_table_md": str(out_dir / "sanity_check_table.md"),
        "sanity_check_summary_json": str(out_dir / "sanity_check_summary.json"),
        "sanity_check_report_txt": str(out_dir / "sanity_check_report.txt"),
        "selected_fragments_dump_json": str(out_dir / "selected_fragments_dump.json"),
    }
    _write_tsv(out_dir / "sanity_check_table.tsv", table_rows)
    _write_markdown(out_dir / "sanity_check_table.md", table_rows)
    report = _render_report(
        rows=table_rows,
        dataset_metadata=dataset_metadata,
        candidate_pool=candidate_pool,
    )
    summary = {
        "run_config": {
            "dataset_path": str(Path(args.dataset_path).expanduser()),
            "label": int(args.label),
            "theta": float(args.theta),
            "out_dir": str(out_dir),
            "candidate_pool_jsonl": str(candidate_pool),
            "embedding_field": str(args.embedding_field),
            "selected_sets": {name: str(path) for name, path in selected_specs},
        },
        "coverage_source": "candidate_pool_evidence",
        "legacy_camc_locator": LEGACY_CAMC_LOCATOR,
        "legacy_camc_reused": False,
        "legacy_camc_reuse_reason": (
            "This sanity command does not receive --teacher-path/--hiv-csv, "
            "so it cannot run the legacy teacher-based CAMC evaluator. It uses "
            "candidate-pool evidence consistently across selected sets."
        ),
        "dataset_metadata": dataset_metadata,
        "rows": table_rows,
        "diagnostics": diagnostics,
        "interpretation": _build_interpretation(table_rows),
        "output_paths": output_paths,
    }
    (out_dir / "sanity_check_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "sanity_check_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "selected_fragments_dump.json").write_text(
        json.dumps(selected_dump, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(report)
    print(json.dumps(output_paths, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
