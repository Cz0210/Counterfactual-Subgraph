#!/usr/bin/env python3
"""Generate deterministic BBBP random candidate-source controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidates.base_generator import CandidateRequest  # noqa: E402
from src.candidates.candidate_source_registry import build_random_generator  # noqa: E402
from src.eval.candidate_lineage_audit import audit_candidate_lineage  # noqa: E402


RANDOM_VARIANTS = ("random_connected_size_matched", "random_brics_size_matched")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--parent-csv", required=True)
    parser.add_argument("--variant", choices=RANDOM_VARIANTS, required=True)
    parser.add_argument("--candidates-per-parent", type=int, required=True)
    parser.add_argument("--size-targets", help="Comma-separated atom counts.")
    parser.add_argument(
        "--reference-candidate-jsonl",
        help="Train/val ChemLLM pool used only for per-parent size matching.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Random candidate parent CSV is empty: {path}")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank reference candidate row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Reference candidate row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _target_map(args: argparse.Namespace, parent_ids: set[str]) -> dict[str, tuple[int, ...]]:
    if bool(args.size_targets) == bool(args.reference_candidate_jsonl):
        raise ValueError(
            "Specify exactly one of --size-targets or --reference-candidate-jsonl."
        )
    if args.size_targets:
        values = tuple(int(value.strip()) for value in args.size_targets.split(",") if value.strip())
        if not values or any(value <= 0 for value in values):
            raise ValueError("Random candidate size targets must be positive integers.")
        return {parent_id: values for parent_id in parent_ids}
    rows = _read_jsonl(Path(args.reference_candidate_jsonl).expanduser().resolve())
    if any(str(row.get("parent_split")) == "test" for row in rows):
        raise ValueError("Test candidates cannot define random-control size distributions.")
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        parent_id = str(row.get("parent_id") or "")
        if parent_id not in parent_ids:
            continue
        try:
            rank = int(row["generation_rank"])
            size = int(row["num_fragment_atoms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Reference candidates require rank and num_fragment_atoms.") from exc
        grouped[parent_id].append((rank, size))
    missing = sorted(parent_ids - set(grouped))
    if missing:
        raise ValueError(f"Reference size distribution lacks parents: {missing[:10]}")
    return {
        parent_id: tuple(size for _rank, size in sorted(values))
        for parent_id, values in grouped.items()
    }


def _derived_seed(seed: int, parent_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{parent_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parent_path = Path(args.parent_csv).expanduser().resolve()
    rows = _read_csv(parent_path)
    required = {"molecule_id", "smiles", "label", "split"}
    missing_fields = sorted(required - set(rows[0]))
    if missing_fields:
        raise ValueError(f"Random candidate parents are missing fields {missing_fields}.")
    if any(str(row["split"]) not in {"train", "val"} for row in rows):
        raise ValueError("Random candidate discovery is restricted to train/val parents.")
    parent_ids = {str(row["molecule_id"]) for row in rows}
    if len(parent_ids) != len(rows):
        raise ValueError("Random candidate parent IDs must be unique.")
    targets = _target_map(args, parent_ids)
    if args.validate_only or args.dry_run:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_RUN",
                    "dataset": "BBBP",
                    "variant": args.variant,
                    "num_parents": len(rows),
                    "candidates_per_parent": args.candidates_per_parent,
                    "seed": args.seed,
                    "size_distribution_source": "train_val_reference" if args.reference_candidate_jsonl else "explicit",
                    "test_statistics_used": False,
                    "formal_output_written": False,
                },
                sort_keys=True,
            )
        )
        return 0
    generator = build_random_generator(args.variant)
    output_rows: list[dict[str, Any]] = []
    shortfalls: Counter[str] = Counter()
    for row in rows:
        parent_id = str(row["molecule_id"])
        batch = generator.generate(
            CandidateRequest(
                parent_id=parent_id,
                parent_smiles=str(row["smiles"]),
                parent_split=str(row["split"]),
                label=int(row["label"]),
                candidates_per_parent=args.candidates_per_parent,
                size_targets=targets[parent_id],
                seed=_derived_seed(args.seed, parent_id),
                max_attempts=args.max_attempts,
            )
        )
        output_rows.extend({**candidate, "dataset": "BBBP", "source_git_commit": _git_commit()} for candidate in batch.rows)
        shortfalls.update(batch.shortfall_reason_counts)
    lineage = audit_candidate_lineage(
        output_rows,
        allowed_candidate_splits=("train", "val"),
        selector_source_splits=("calibration",),
        threshold_source="calibration",
        expected_dataset="BBBP",
    )
    output_path = Path(args.output_jsonl).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Random candidate output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    summary = {
        "schema_version": "bbbp_random_candidate_control_v1",
        "dataset": "BBBP",
        "variant": args.variant,
        "seed": args.seed,
        "num_parents": len(rows),
        "requested_count": len(rows) * args.candidates_per_parent,
        "generated_count": len(output_rows),
        "shortfall_count": len(rows) * args.candidates_per_parent - len(output_rows),
        "shortfall_reason_counts": dict(sorted(shortfalls.items())),
        "size_distribution_source": "train_val_reference" if args.reference_candidate_jsonl else "explicit",
        "test_statistics_used": False,
        "lineage_audit": lineage,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print("[BBBP_RANDOM_CANDIDATE_CONTROL_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
