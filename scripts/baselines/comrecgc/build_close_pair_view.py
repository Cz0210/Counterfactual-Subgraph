#!/usr/bin/env python3
"""Build and validate a hash-closed COMRECGC theta-close logical view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.close_pair_view import (  # noqa: E402
    FILTER_OPERATOR,
    NORMALIZED_DISTANCE_CONTRACT,
    PAIR_ORIENTATION,
    SCALE_CONTRACT,
    ThetaClosePairContract,
    materialize_theta_close_pair_view,
)
from src.baselines.comrecgc.contracts import sha256_file  # noqa: E402
from src.utils.autodl_aids_greed_full_scan_supervisor import (  # noqa: E402
    validate_receipt,
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _required(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    raise ValueError(f"pair-semantics contract lacks required field: {'/'.join(names)}")


def _write_pass_last(output_dir: Path) -> None:
    path = output_dir / "PASS"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".PASS.", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("PASS\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--pair-semantics-contract", required=True)
    parser.add_argument("--pair-semantics-receipt")
    parser.add_argument("--expected-pair-semantics-science-root")
    parser.add_argument("--expected-execution-commit")
    parser.add_argument("--physical-vectors", required=True)
    parser.add_argument("--normalized-distances", required=True)
    parser.add_argument("--all-pairs-close-certificate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=1_000_000)
    parser.add_argument(
        "--max-compact-gb",
        type=float,
        default=0.0,
        help=(
            "Explicit upper bound for selected vector/pair copies. Zero keeps "
            "partial-close views bitmap/index-only and fail-closed for path DBSCAN."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_path = Path(args.pair_semantics_contract).expanduser().resolve(strict=True)
    receipt_args = (
        args.pair_semantics_receipt,
        args.expected_pair_semantics_science_root,
        args.expected_execution_commit,
    )
    if any(receipt_args) and not all(receipt_args):
        raise ValueError(
            "receipt validation requires receipt, fixed science root, and execution commit"
        )
    if all(receipt_args):
        receipt = validate_receipt(
            receipt_path=args.pair_semantics_receipt,
            expected_science_root=args.expected_pair_semantics_science_root,
            expected_execution_commit=args.expected_execution_commit,
        )
        if receipt["close_pair_contract"] != str(source_path):
            raise ValueError(
                "receipt does not authorize the requested pair-semantics contract"
            )
        expected_distance = receipt["terminal_files"][
            "distance_scan/normalized_distances.greed.float32.npy"
        ]
        distance_path = Path(args.normalized_distances).expanduser().resolve(strict=True)
        if (
            expected_distance["path"] != str(distance_path)
            or expected_distance["sha256"]
            != sha256_file(distance_path)
        ):
            raise ValueError(
                "receipt does not authorize the requested normalized-distance array"
            )
    source = _load_object(source_path)
    pair_orientation = str(_required(source, "pair_orientation", "pair_axis"))
    if pair_orientation not in {PAIR_ORIENTATION, "col0=parent;col1=candidate"}:
        raise ValueError("pair-semantics contract has the wrong pair orientation")
    if str(_required(source, "filter_operator")) != FILTER_OPERATOR:
        raise ValueError("pair-semantics contract must use inclusive <= filtering")
    contract = ThetaClosePairContract(
        theta=float(_required(source, "theta")),
        parent_count=int(_required(source, "parent_count", "num_parents")),
        candidate_count=int(
            _required(source, "candidate_count", "num_candidates")
        ),
        distance_checkpoint_sha256=str(
            _required(
                source, "distance_checkpoint_sha256", "distance_checkpoint_hash"
            )
        ),
        embedding_checkpoint_sha256=str(
            _required(
                source,
                "embedding_checkpoint_sha256",
                "embedding_checkpoint_hash",
            )
        ),
        scale_contract=str(source.get("scale_contract") or SCALE_CONTRACT),
        normalized_distance_contract=str(
            source.get("normalized_distance_contract")
            or NORMALIZED_DISTANCE_CONTRACT
        ),
    )
    result = materialize_theta_close_pair_view(
        physical_vectors_path=args.physical_vectors,
        normalized_distances_path=args.normalized_distances,
        output_dir=args.output_dir,
        contract=contract,
        expected_physical_vectors_sha256=source.get("physical_vectors_sha256"),
        expected_normalized_distances_sha256=source.get(
            "normalized_distances_sha256"
        ),
        pair_semantics_contract_path=source_path,
        all_pairs_close_certificate_path=args.all_pairs_close_certificate,
        max_compact_bytes=int(float(args.max_compact_gb) * 1024**3),
        block_size=args.block_size,
        resume=args.resume,
    )
    payload = {
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "physical_pair_count": result.physical_store_rows,
        "logical_close_pair_count": result.logical_close_rows,
        "all_pairs_close": result.all_pairs_close,
        "view_storage": result.view_storage,
        "eligible_for_dbscan": result.eligible_for_dbscan,
        "blocking_reason": result.blocking_reason,
    }
    print(json.dumps(payload, sort_keys=True))
    if not result.eligible_for_dbscan:
        if (Path(args.output_dir) / "PASS").exists():
            raise RuntimeError("blocked close view unexpectedly has a PASS marker")
        print("[COMRECGC_CLOSE_PAIR_VIEW_BLOCKED_STORAGE]")
        return 75
    _write_pass_last(Path(args.output_dir).expanduser().resolve(strict=True))
    print("[COMRECGC_CLOSE_PAIR_VIEW_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
