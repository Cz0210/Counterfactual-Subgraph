#!/usr/bin/env python3
"""Replay and audit Mutagenicity v1 SFT target candidate selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity import write_csv, write_json  # noqa: E402
from src.data.mutagenicity_sft_ppo import (  # noqa: E402
    DEFAULT_EXPECTED_COUNTS,
    MutagenicitySFTPPOConfig,
    load_teacher_consistent_parents,
)
from src.data.mutagenicity_sft_v2 import (  # noqa: E402
    build_teacher_scorer,
    canonicalize_fragment,
    completion_frequency,
    enumerate_split_candidates,
    is_positive_fallback_candidate,
    is_teacher_strict_candidate,
    select_strict_first_candidate,
)
from src.data.sft_v3_builder import candidate_ranking_key  # noqa: E402


DEFAULT_DATA_ROOT = Path("outputs/hpc/mutagenicity/final/sft_ppo_data_v1")
DEFAULT_PARENT_ROOT = Path(
    "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"
)
DEFAULT_TEACHER = Path(
    "outputs/hpc/oracle/final/mutagenicity_rf_v1/mutagenicity_rf_model.pkl"
)
DEFAULT_OUTPUT = Path(
    "outputs/hpc/mutagenicity/audits/sft_target_selection_v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hpc.yaml"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--teacher-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--min-atom-ratio", type=float, default=0.10)
    parser.add_argument("--max-atom-ratio", type=float, default=0.55)
    parser.add_argument("--min-frag-atoms", type=int, default=3)
    parser.add_argument("--max-frag-atoms", type=int, default=30)
    parser.add_argument("--max-candidates-per-parent", type=int, default=160)
    return parser


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _strategy_summary(
    candidates: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["candidate_strategy"])].append(row)
    selected_counts = Counter(
        str(row.get("candidate_strategy") or "") for row in selected_rows
    )
    output: list[dict[str, Any]] = []
    for strategy, rows in sorted(grouped.items()):
        drops = [
            value
            for value in (_finite(row.get("cf_drop")) for row in rows)
            if value is not None
        ]
        ratios = [
            value
            for value in (_finite(row.get("atom_ratio")) for row in rows)
            if value is not None
        ]
        fragments = [str(row.get("core_fragment") or "") for row in rows]
        strict_count = sum(bool(row.get("strict_flip")) for row in rows)
        positive_count = sum(
            bool(row.get("positive_cf_drop")) for row in rows
        )
        output.append(
            {
                "candidate_strategy": strategy,
                "candidate_count": len(rows),
                "selected_count": int(selected_counts.get(strategy, 0)),
                "strict_flip_count": strict_count,
                "strict_flip_rate": strict_count / len(rows),
                "positive_cf_drop_count": positive_count,
                "positive_cf_drop_rate": positive_count / len(rows),
                "cf_drop_mean": mean(drops) if drops else None,
                "cf_drop_median": median(drops) if drops else None,
                "atom_ratio_mean": mean(ratios) if ratios else None,
                "atom_ratio_median": median(ratios) if ratios else None,
                "unique_fragment_count": len(set(fragments)),
                "duplicate_rate": (
                    1.0 - len(set(fragments)) / len(fragments)
                    if fragments
                    else 0.0
                ),
            }
        )
    return output


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    data_root = _resolve(args.data_root)
    parent_root = _resolve(args.parent_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = _resolve(args.teacher_path)
    config_adapter = MutagenicitySFTPPOConfig(
        min_atom_ratio=float(args.min_atom_ratio),
        max_atom_ratio=float(args.max_atom_ratio),
        min_frag_atoms=int(args.min_frag_atoms),
        max_frag_atoms=int(args.max_frag_atoms),
        max_candidates_per_parent=int(args.max_candidates_per_parent),
        use_teacher_ranking=True,
    )
    sft_config = config_adapter.sft_v3_config(teacher_path)
    scorer = build_teacher_scorer(teacher_path)

    all_candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    non_strict_selected_strategies: Counter[str] = Counter()
    wrong_priority_strategies: Counter[str] = Counter()
    for split, selected_name in (
        ("train", "mutagenicity_sft_train.csv"),
        ("val", "mutagenicity_sft_val.csv"),
    ):
        parents = load_teacher_consistent_parents(
            parent_root / f"{split}_source_label1_teacher_correct.csv",
            expected_split=split,
            expected_count=DEFAULT_EXPECTED_COUNTS[split],
        )
        if int(args.max_parents) > 0:
            parents = parents[: int(args.max_parents)]
        selected = _read_csv(data_root / selected_name)
        if int(args.max_parents) > 0:
            allowed = {parent.molecule_id for parent in parents}
            selected = [
                row for row in selected if str(row.get("molecule_id")) in allowed
            ]
        selected_by_parent = {
            str(row["molecule_id"]): row for row in selected
        }
        candidate_sets = enumerate_split_candidates(
            parents,
            config=sft_config,
            scorer=scorer,
        )
        frequencies = completion_frequency(candidate_sets)
        split_candidate_rows: list[dict[str, Any]] = []
        parents_with_oracle = 0
        parents_with_positive = 0
        parents_with_strict = 0
        selected_strict_count = 0
        wrong_priority_count = 0
        for candidate_set in candidate_sets:
            selected_row = selected_by_parent.get(candidate_set.parent.molecule_id)
            selected_fragment = (
                canonicalize_fragment(str(selected_row.get("core_fragment") or ""))
                if selected_row
                else None
            )
            selected_strategy = str(
                (selected_row or {}).get("candidate_strategy")
                or (selected_row or {}).get("selected_strategy")
                or ""
            )
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidate_set.candidates
                    if canonicalize_fragment(candidate.core_fragment)
                    == selected_fragment
                    and (
                        not selected_strategy
                        or candidate.candidate_strategy == selected_strategy
                    )
                ),
                None,
            )
            strict_candidates = [
                candidate
                for candidate in candidate_set.candidates
                if is_teacher_strict_candidate(candidate)
            ]
            positive_candidates = [
                candidate
                for candidate in candidate_set.candidates
                if is_positive_fallback_candidate(candidate)
            ]
            oracle_candidates = [
                candidate
                for candidate in candidate_set.candidates
                if candidate.oracle_ok
            ]
            parents_with_oracle += bool(oracle_candidates)
            parents_with_positive += bool(positive_candidates)
            parents_with_strict += bool(strict_candidates)
            selected_is_strict = bool(
                selected_candidate
                and is_teacher_strict_candidate(selected_candidate)
            )
            selected_strict_count += selected_is_strict
            wrong_priority = bool(strict_candidates and not selected_is_strict)
            wrong_priority_count += wrong_priority
            if selected_candidate is not None and not selected_is_strict:
                non_strict_selected_strategies[
                    selected_candidate.candidate_strategy
                ] += 1
                if wrong_priority:
                    wrong_priority_strategies[
                        selected_candidate.candidate_strategy
                    ] += 1
            best_strict = select_strict_first_candidate(
                candidate_set.candidates,
                completion_frequency=frequencies,
                allow_soft_fallback=False,
            ).selected
            reason = (
                "strict_candidate_exists_but_selected_non_strict"
                if wrong_priority
                else (
                    "selected_strict_candidate"
                    if selected_is_strict
                    else (
                        "no_strict_candidate_in_enumerated_pool"
                        if not strict_candidates
                        else "selected_candidate_not_reconstructed"
                    )
                )
            )
            parent_rows.append(
                {
                    "split": split,
                    "parent_id": candidate_set.parent.molecule_id,
                    "parent_smiles": candidate_set.parent.parent_smiles,
                    "proposal_count": candidate_set.proposal_count,
                    "valid_candidate_count": len(candidate_set.candidates),
                    "oracle_ok_candidate_count": len(oracle_candidates),
                    "positive_cf_drop_candidate_count": len(positive_candidates),
                    "strict_candidate_count": len(strict_candidates),
                    "selected_is_strict": selected_is_strict,
                    "selection_regret_type": reason,
                }
            )
            if wrong_priority or not selected_is_strict:
                regret_rows.append(
                    {
                        "split": split,
                        "parent_id": candidate_set.parent.molecule_id,
                        "parent_smiles": candidate_set.parent.parent_smiles,
                        "selected_fragment": (
                            selected_candidate.core_fragment
                            if selected_candidate
                            else str((selected_row or {}).get("core_fragment") or "")
                        ),
                        "selected_strategy": (
                            selected_candidate.candidate_strategy
                            if selected_candidate
                            else str((selected_row or {}).get("candidate_strategy") or "")
                        ),
                        "selected_is_strict": selected_is_strict,
                        "selected_cf_drop": (
                            selected_candidate.cf_drop if selected_candidate else None
                        ),
                        "selected_atom_ratio": (
                            selected_candidate.atom_ratio if selected_candidate else None
                        ),
                        "selected_score": (
                            json.dumps(
                                list(
                                    candidate_ranking_key(
                                        selected_candidate,
                                        config=sft_config,
                                    )
                                )
                            )
                            if selected_candidate
                            else ""
                        ),
                        "best_strict_fragment": (
                            best_strict.core_fragment if best_strict else ""
                        ),
                        "best_strict_strategy": (
                            best_strict.candidate_strategy if best_strict else ""
                        ),
                        "best_strict_cf_drop": (
                            best_strict.cf_drop if best_strict else None
                        ),
                        "best_strict_atom_ratio": (
                            best_strict.atom_ratio if best_strict else None
                        ),
                        "best_strict_score": (
                            json.dumps(
                                list(
                                    candidate_ranking_key(
                                        best_strict,
                                        config=sft_config,
                                    )
                                )
                            )
                            if best_strict
                            else ""
                        ),
                        "strict_candidate_count": len(strict_candidates),
                        "selection_regret_type": reason,
                        "reason": reason,
                    }
                )
            for index, candidate in enumerate(candidate_set.candidates):
                split_candidate_rows.append(
                    {
                        "split": split,
                        "parent_id": candidate_set.parent.molecule_id,
                        "candidate_index": index,
                        "candidate_strategy": candidate.candidate_strategy,
                        "core_fragment": candidate.core_fragment,
                        "cf_drop": candidate.cf_drop,
                        "atom_ratio": candidate.atom_ratio,
                        "strict_flip": is_teacher_strict_candidate(candidate),
                        "positive_cf_drop": is_positive_fallback_candidate(candidate),
                    }
                )
        all_candidate_rows.extend(split_candidate_rows)
        selected_rows.extend(selected)
        split_summaries[split] = {
            "parent_total": len(parents),
            "parents_with_oracle_ok_candidate": parents_with_oracle,
            "parents_with_positive_cf_drop_candidate": parents_with_positive,
            "parents_with_strict_candidate": parents_with_strict,
            "parents_without_strict_candidate": len(parents) - parents_with_strict,
            "final_target_strict_count": selected_strict_count,
            "strict_exists_but_selected_non_strict_count": wrong_priority_count,
            "selected_target_rows": len(selected),
        }

    strategy_rows = _strategy_summary(all_candidate_rows, selected_rows)
    completion_counts = Counter(
        str(row.get("core_fragment") or "") for row in selected_rows
    )
    completion_rows = [
        {"completion": completion, "frequency": frequency}
        for completion, frequency in sorted(
            completion_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    write_csv(
        output_dir / "strategy_summary.csv",
        strategy_rows,
        tuple(strategy_rows[0]) if strategy_rows else ("candidate_strategy",),
    )
    write_csv(
        output_dir / "parent_candidate_counts.csv",
        parent_rows,
        tuple(parent_rows[0]) if parent_rows else ("split", "parent_id"),
    )
    regret_fields = (
        "split",
        "parent_id",
        "parent_smiles",
        "selected_fragment",
        "selected_strategy",
        "selected_is_strict",
        "selected_cf_drop",
        "selected_atom_ratio",
        "selected_score",
        "best_strict_fragment",
        "best_strict_strategy",
        "best_strict_cf_drop",
        "best_strict_atom_ratio",
        "best_strict_score",
        "strict_candidate_count",
        "selection_regret_type",
        "reason",
    )
    write_csv(output_dir / "selection_regret.csv", regret_rows, regret_fields)
    write_csv(
        output_dir / "completion_frequency.csv",
        completion_rows,
        ("completion", "frequency"),
    )
    priority_error_count = sum(
        summary["strict_exists_but_selected_non_strict_count"]
        for summary in split_summaries.values()
    )
    summary = {
        "audit_kind": "deterministic_candidate_replay",
        "raw_candidate_artifact_available_in_v1": False,
        "candidate_replay_implementation": (
            "src.data.sft_v3_builder.enumerate_reference_candidates_for_parent"
        ),
        "legacy_ranking_implementation": (
            "src.data.sft_v3_builder.candidate_ranking_key"
        ),
        "strict_definition": "oracle_ok and pred_before == 1 and pred_after == 0",
        "splits": split_summaries,
        "selection_priority_error_count": priority_error_count,
        "non_strict_selected_strategy_counts": dict(
            sorted(non_strict_selected_strategies.items())
        ),
        "wrong_priority_strategy_counts": dict(
            sorted(wrong_priority_strategies.items())
        ),
        "selector_priority_issue_confirmed": priority_error_count > 0,
        "bond_k2_is_not_banned": True,
        "audit_passed": True,
    }
    write_json(output_dir / "audit_summary.json", summary)
    report = [
        "# Mutagenicity SFT target selection audit",
        "",
        "The v1 build did not persist its complete candidate table. This audit",
        "replays the exact deterministic v1 proposal/filter path with the fixed RF teacher.",
        "",
        "## Selection semantics",
        "",
        "- Legacy key: oracle availability, exported cf_flip, CFDrop, then size/strategy heuristics.",
        "- Audit strict key: oracle_ok and pred_before=1 and pred_after=0.",
        f"- Strict-exists/non-strict-selected count: {priority_error_count}",
        f"- Non-strict selected strategies: {dict(sorted(non_strict_selected_strategies.items()))}",
        f"- Wrong-priority strategies: {dict(sorted(wrong_priority_strategies.items()))}",
        "- bond_k2 remains eligible when it is itself strict.",
        "",
        "No performance claim is made by this read-only audit.",
    ]
    (output_dir / "audit_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print("[MUTAGENICITY_SFT_TARGET_SELECTION_AUDIT_OK]", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
