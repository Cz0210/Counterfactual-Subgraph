#!/usr/bin/env python3
"""Add MolCLR pretrained GNN embeddings to a candidate_pool JSONL file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.molclr_gnn_embedding import encode_smiles_list  # noqa: E402


DEFAULT_POOL = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "full_candidate_pools"
    / "stable300_label1_merged_base_temp07"
    / "candidate_pool.jsonl"
)
DEFAULT_OUT = DEFAULT_POOL.with_name("candidate_pool_with_molclr_gnn_embeddings.jsonl")
SMILES_FALLBACK_FIELDS = ("fragment", "raw_fragment", "final_fragment_smiles")


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One candidate row plus its optional fragment SMILES source."""

    row_index: int
    line_number: int
    row: dict[str, Any]
    smiles: str | None
    smiles_field: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for Slurm parity. This script uses explicit CLI paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept only for Slurm wrapper parity.",
    )
    parser.add_argument("--candidate-pool", default=str(DEFAULT_POOL), help="Input candidate pool JSONL.")
    parser.add_argument("--out-jsonl", default=str(DEFAULT_OUT), help="Output JSONL with GNN embeddings.")
    parser.add_argument("--summary-json", default=None, help="Summary JSON path.")
    parser.add_argument("--failed-jsonl", default=None, help="Failed rows JSONL path.")
    parser.add_argument("--molclr-root", required=True, help="Runtime MolCLR repository/root directory.")
    parser.add_argument("--molclr-ckpt", required=True, help="MolCLR checkpoint path.")
    parser.add_argument("--encoder-type", choices=("gin", "gcn"), default="gin")
    parser.add_argument("--smiles-field", default="final_fragment")
    parser.add_argument("--embedding-field", default="final_fragment_gnn_embedding")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--invalid-policy", choices=("error", "skip", "zero"), default="error")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional small dry-run limit. Reads/writes at most this many non-empty JSONL rows.",
    )
    return parser


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=False) + "\n"


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _smiles_fields(primary_field: str) -> list[str]:
    fields = [str(primary_field)]
    if primary_field != "final_fragment":
        fields.append("final_fragment")
    for fallback in SMILES_FALLBACK_FIELDS:
        if fallback not in fields:
            fields.append(fallback)
    return fields


def resolve_fragment_smiles(row: dict[str, Any], primary_field: str) -> tuple[str, str] | None:
    """Resolve the fragment SMILES used for MolCLR graph embedding."""

    for field_name in _smiles_fields(primary_field):
        value = _normalize_text(row.get(field_name))
        if value:
            return value, field_name
    return None


def _read_candidate_rows(path: Path, primary_field: str, max_rows: int | None) -> tuple[list[CandidateRow], list[dict[str, Any]], int]:
    rows: list[CandidateRow] = []
    failed_rows: list[dict[str, Any]] = []
    total_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if max_rows is not None and total_rows >= max_rows:
                break
            total_rows += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON at line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {line_number}: {path}")

            resolved = resolve_fragment_smiles(row, primary_field)
            if resolved is None:
                rows.append(
                    CandidateRow(
                        row_index=total_rows - 1,
                        line_number=line_number,
                        row=row,
                        smiles=None,
                        smiles_field=None,
                    )
                )
                failed_rows.append(
                    {
                        "row_index": total_rows - 1,
                        "line_number": line_number,
                        "failure_reason": "missing fragment SMILES source",
                        "tried_fields": _smiles_fields(primary_field),
                        "row": row,
                    }
                )
                continue
            smiles, field_name = resolved
            rows.append(
                CandidateRow(
                    row_index=total_rows - 1,
                    line_number=line_number,
                    row=row,
                    smiles=smiles,
                    smiles_field=field_name,
                )
            )
    return rows, failed_rows, total_rows


def _write_failed_rows(path: Path, failed_rows: list[dict[str, Any]]) -> None:
    if not failed_rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in failed_rows:
            handle.write(_json_dumps(row))


def main() -> int:
    args = build_parser().parse_args()

    candidate_pool = Path(args.candidate_pool).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    molclr_root = Path(args.molclr_root).expanduser().resolve()
    molclr_ckpt = Path(args.molclr_ckpt).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else out_jsonl.parent / "molclr_gnn_embedding_summary.json"
    )
    failed_jsonl = (
        Path(args.failed_jsonl).expanduser().resolve()
        if args.failed_jsonl
        else out_jsonl.parent / "molclr_gnn_embedding_failed_rows.jsonl"
    )

    if candidate_pool == out_jsonl:
        raise ValueError("--out-jsonl must differ from --candidate-pool to preserve the original pool.")
    if not candidate_pool.exists():
        raise FileNotFoundError(f"candidate pool not found: {candidate_pool}")
    if not molclr_root.exists():
        raise FileNotFoundError(f"MolCLR root not found: {molclr_root}")
    if not molclr_ckpt.exists():
        raise FileNotFoundError(f"MolCLR checkpoint not found: {molclr_ckpt}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_rows is not None and int(args.max_rows) <= 0:
        raise ValueError("--max-rows must be positive when provided.")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    candidate_rows, failed_rows, total_rows = _read_candidate_rows(
        candidate_pool,
        str(args.smiles_field),
        int(args.max_rows) if args.max_rows is not None else None,
    )
    rows_with_smiles = [row for row in candidate_rows if row.smiles is not None]
    unique_smiles = sorted({str(row.smiles) for row in rows_with_smiles})

    embeddings = encode_smiles_list(
        unique_smiles,
        molclr_root=molclr_root,
        molclr_ckpt=molclr_ckpt,
        encoder_type=str(args.encoder_type),
        batch_size=int(args.batch_size),
        device=str(args.device),
        invalid_policy=str(args.invalid_policy),
    )

    encoded_rows = 0
    skipped_embedding_rows = 0
    embedding_dim: int | None = None
    dim_counter: Counter[int] = Counter()
    source_counter: Counter[str] = Counter()

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for candidate in candidate_rows:
            row = dict(candidate.row)
            if candidate.smiles is None:
                handle.write(_json_dumps(row))
                continue
            embedding = embeddings.get(candidate.smiles)
            if embedding is None:
                skipped_embedding_rows += 1
                failed_rows.append(
                    {
                        "row_index": candidate.row_index,
                        "line_number": candidate.line_number,
                        "failure_reason": "MolCLR embedding missing for resolved SMILES",
                        "smiles": candidate.smiles,
                        "smiles_field": candidate.smiles_field,
                        "row": candidate.row,
                    }
                )
            elif not embedding or any(not math.isfinite(float(value)) for value in embedding):
                skipped_embedding_rows += 1
                failed_rows.append(
                    {
                        "row_index": candidate.row_index,
                        "line_number": candidate.line_number,
                        "failure_reason": "MolCLR embedding was empty or non-finite",
                        "smiles": candidate.smiles,
                        "smiles_field": candidate.smiles_field,
                        "row": candidate.row,
                    }
                )
            else:
                row[str(args.embedding_field)] = [float(value) for value in embedding]
                encoded_rows += 1
                embedding_dim = int(len(embedding))
                dim_counter[embedding_dim] += 1
                source_counter[str(candidate.smiles_field)] += 1
            handle.write(_json_dumps(row))

    _write_failed_rows(failed_jsonl, failed_rows)

    summary = {
        "candidate_pool": str(candidate_pool),
        "out_jsonl": str(out_jsonl),
        "total_rows": int(total_rows),
        "rows_with_resolved_smiles": int(len(rows_with_smiles)),
        "encoded_rows": int(encoded_rows),
        "failed_rows": int(len(failed_rows)),
        "skipped_embedding_rows": int(skipped_embedding_rows),
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
        "embedding_dimension_distribution": {
            str(dim): count for dim, count in sorted(dim_counter.items())
        },
        "embedding_field": str(args.embedding_field),
        "smiles_field": str(args.smiles_field),
        "smiles_field_fallbacks": _smiles_fields(str(args.smiles_field)),
        "smiles_field_counts": dict(sorted(source_counter.items())),
        "molclr_root": str(molclr_root),
        "molclr_ckpt": str(molclr_ckpt),
        "encoder_type": str(args.encoder_type),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "invalid_policy": str(args.invalid_policy),
        "max_rows": int(args.max_rows) if args.max_rows is not None else None,
        "summary_json": str(summary_json),
        "failed_rows_jsonl": str(failed_jsonl) if failed_rows else None,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if failed_rows and str(args.invalid_policy) == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
