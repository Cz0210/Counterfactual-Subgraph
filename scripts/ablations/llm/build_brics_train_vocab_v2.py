#!/usr/bin/env python3
"""Build the BACE full-train BRICS vocabulary and fixed proposal pool on CPU."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ablations.llm.brics import (  # noqa: E402
    build_train_only_brics_vocabulary,
    training_molecules_from_mappings,
)
from src.ablations.llm.contracts import (  # noqa: E402
    LLMAblationContractError,
    canonical_json_sha256,
    require_sha256,
)
from src.eval.bace_frozen_gnn_contracts import stable_sha256  # noqa: E402

try:  # noqa: E402
    from rdkit import Chem
except ImportError:  # pragma: no cover - runtime dependency
    Chem = None


SCHEMA = "bace_brics_full_train_cpu_v2"
FORBIDDEN_PATH_TOKENS = ("calibration", "validation", "valid", "test")
EXPECTED_COHORT_SCHEMA = "bace_frozen_parent_ids_v1"
MAIN_FILTER_PROVENANCE = {
    "source_commit": "0ad149420577c683baa2ef03f78f70ee6841f3a1",
    "bace_frozen_gnn_pool_sha256": (
        "ed0f6ce93219b09c3fcfb879a3d49cb2eb5383917ef59d98e517d78df32bc3d2"
    ),
    "hard_deletion_sha256": (
        "0a5485aeeff2d24dc01d68f70ee51ceb623f667dc299ce39495e4815dc71b2e5"
    ),
    "active_candidate_predicate": [
        "parse_ok",
        "valid",
        "connected",
        "direct_substructure",
        "oracle_ok",
    ],
    "numeric_size_filter": {
        "min_fragment_atoms": None,
        "max_fragment_atoms": None,
        "min_atom_ratio": None,
        "max_atom_ratio": None,
    },
    "substructure_use_chirality": True,
    "hard_deletion_downstream_required": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_file(path_like: str | Path, *, role: str) -> Path:
    lexical = Path(path_like).expanduser()
    if not lexical.is_absolute() or lexical.is_symlink():
        raise LLMAblationContractError(f"{role} must be an absolute physical file")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise LLMAblationContractError(f"{role} is not a regular file: {path}")
    return path


def _assert_train_path(path: Path, *, role: str) -> None:
    lowered = path.name.lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise LLMAblationContractError(f"{role} path looks like calibration/test data")
    if role == "train_csv" and path.name.lower() != "train.csv":
        raise LLMAblationContractError("full BACE input must be the physical train.csv")
    if role == "proposal_cohort" and not lowered.startswith("train_"):
        raise LLMAblationContractError("proposal cohort must be a train manifest")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    _atomic_bytes(path, content.encode("utf-8"))


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _read_full_train_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    actual_sha = sha256_file(path)
    if actual_sha != require_sha256(expected_sha256, field="train_csv_sha256"):
        raise LLMAblationContractError("full BACE train.csv SHA256 mismatch")
    required = {"molecule_id", "parent_id", "smiles", "label", "split"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise LLMAblationContractError("BACE train.csv lacks required identity columns")
        fieldnames = list(reader.fieldnames)
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(reader):
            # This is the complete read allowlist. Scores, predictions, rewards,
            # calibration metadata, and test metadata cannot enter BRICS.
            row = {key: str(raw.get(key) or "").strip() for key in required}
            if row["split"] != "train":
                raise LLMAblationContractError(
                    f"full train.csv row {index} has split={row['split']!r}"
                )
            if not row["parent_id"] or row["molecule_id"] != row["parent_id"]:
                raise LLMAblationContractError(
                    f"row {index} molecule_id/parent_id identity changed"
                )
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise LLMAblationContractError(f"row {index} label is invalid") from exc
            if label not in (0, 1):
                raise LLMAblationContractError(f"row {index} label must be binary")
            rows.append({**row, "label": label})
    if expected_rows <= 386:
        raise LLMAblationContractError(
            "expected full-train rows must exceed the 386-parent source cohort"
        )
    if len(rows) != expected_rows:
        raise LLMAblationContractError(
            f"full BACE train row count mismatch: {len(rows)} != {expected_rows}"
        )
    ids = [row["parent_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise LLMAblationContractError("full BACE train parent IDs must be unique")
    return rows, fieldnames


def _read_cohort(
    path: Path,
    *,
    expected_sha256: str,
    expected_parent_count: int,
    train_csv: Path,
    train_sha256: str,
    train_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if sha256_file(path) != require_sha256(
        expected_sha256, field="proposal_cohort_manifest_sha256"
    ):
        raise LLMAblationContractError("proposal cohort manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LLMAblationContractError("proposal cohort manifest must be an object")
    required = {
        "schema_version": EXPECTED_COHORT_SCHEMA,
        "status": "FROZEN",
        "dataset": "bace",
        "split": "train",
        "source_label": 1,
    }
    changed = [key for key, expected in required.items() if payload.get(key) != expected]
    if changed:
        raise LLMAblationContractError(
            "proposal cohort contract changed: " + ", ".join(changed)
        )
    raw_ids = payload.get("parent_ids")
    if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
        raise LLMAblationContractError("proposal cohort parent_ids must be list[str]")
    parent_ids = tuple(raw_ids)
    if parent_ids != tuple(sorted(parent_ids)) or len(parent_ids) != len(set(parent_ids)):
        raise LLMAblationContractError("proposal cohort IDs must be sorted and unique")
    if (
        len(parent_ids) != expected_parent_count
        or payload.get("parent_count") != expected_parent_count
    ):
        raise LLMAblationContractError("proposal cohort parent count mismatch")
    if payload.get("parent_ids_sha256") != stable_sha256(list(parent_ids)):
        raise LLMAblationContractError("proposal cohort parent_ids SHA changed")
    missing = [parent_id for parent_id in parent_ids if parent_id not in train_rows]
    wrong_label = [
        parent_id for parent_id in parent_ids if train_rows.get(parent_id, {}).get("label") != 1
    ]
    if missing or wrong_label:
        raise LLMAblationContractError(
            f"proposal cohort is not a source-label train subset: missing={missing[:3]}, "
            f"wrong_label={wrong_label[:3]}"
        )
    split_identity = payload.get("split_identity")
    if not isinstance(split_identity, Mapping):
        raise LLMAblationContractError("proposal cohort lacks train split identity")
    if (
        str(split_identity.get("sha256") or "") != train_sha256
        or Path(str(split_identity.get("path") or "")).resolve(strict=False) != train_csv
        or split_identity.get("size") != train_csv.stat().st_size
    ):
        raise LLMAblationContractError("proposal cohort is bound to another train.csv")
    return payload, parent_ids


def _read_reference_contract(
    path: Path,
    *,
    expected_sha256: str,
    train_csv: Path,
    train_sha256: str,
    cohort_path: Path,
    cohort_sha256: str,
    expected_parent_count: int,
    attempts_per_parent: int,
) -> dict[str, Any]:
    if sha256_file(path) != require_sha256(
        expected_sha256, field="reference_contract_file_sha256"
    ):
        raise LLMAblationContractError("BACE LLM reference contract file SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LLMAblationContractError("BACE LLM reference contract must be an object")
    required = {
        "schema_version": "bace_ours_llm_reference_v2",
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "source_label": 1,
        "scientific_values_inferred": False,
    }
    changed = [key for key, expected in required.items() if payload.get(key) != expected]
    if changed:
        raise LLMAblationContractError(
            "BACE LLM reference identity changed: " + ", ".join(changed)
        )
    claimed_self = require_sha256(
        payload.get("reference_contract_sha256"), field="reference_contract_sha256"
    )
    body = dict(payload)
    body.pop("reference_contract_sha256")
    if canonical_json_sha256(body) != claimed_self:
        raise LLMAblationContractError("BACE LLM reference self-hash changed")

    generation = payload.get("candidate_generation")
    downstream = payload.get("frozen_downstream")
    if not isinstance(generation, Mapping) or not isinstance(downstream, Mapping):
        raise LLMAblationContractError("BACE LLM reference lacks generation/downstream pins")
    if (
        generation.get("train_only") is not True
        or generation.get("test_loaded") is not False
        or generation.get("parent_count") != expected_parent_count
        or generation.get("attempts_per_parent") != attempts_per_parent
    ):
        raise LLMAblationContractError("BACE reference proposal budget/split changed")
    parent_manifest = generation.get("parent_manifest")
    if not isinstance(parent_manifest, Mapping) or (
        str(parent_manifest.get("sha256") or "") != cohort_sha256
        or Path(str(parent_manifest.get("path") or "")).resolve(strict=False) != cohort_path
        or parent_manifest.get("size") != cohort_path.stat().st_size
    ):
        raise LLMAblationContractError("BACE reference points to another proposal cohort")
    split_paths = downstream.get("dataset_split_paths")
    split_hashes = downstream.get("dataset_split_hashes")
    if not isinstance(split_paths, Mapping) or not isinstance(split_hashes, Mapping):
        raise LLMAblationContractError("BACE reference lacks frozen split pins")
    if (
        Path(str(split_paths.get("train") or "")).resolve(strict=False) != train_csv
        or split_hashes.get("train") != train_sha256
    ):
        raise LLMAblationContractError("BACE reference points to another full train.csv")
    # Calibration/test are identity pins only.  This CPU builder never opens
    # their paths, and the manifest reports that boundary explicitly.
    for split in ("calibration", "test"):
        if not str(split_paths.get(split) or ""):
            raise LLMAblationContractError(f"BACE reference lacks {split} split pin")
        require_sha256(split_hashes.get(split), field=f"{split}_split_sha256")

    expected_regimes = {
        "base_regime": {
            "batch_size": 1,
            "num_return_sequences": 4,
            "oracle_batch_size": 256,
            "seed": 7,
            "stage": "B8_POOL_BASE",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
        "high_temperature_regime": {
            "batch_size": 1,
            "num_return_sequences": 4,
            "oracle_batch_size": 256,
            "seed": 13,
            "stage": "B9_POOL_HIGHTEMP",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
    }
    allowed_regime_keys = {
        "num_return_sequences",
        "seed",
        "temperature",
        "top_p",
        "max_new_tokens",
        "batch_size",
        "oracle_batch_size",
        "stage",
    }
    for name, expected in expected_regimes.items():
        observed = generation.get(name)
        if not isinstance(observed, Mapping):
            raise LLMAblationContractError(f"BACE reference lacks {name}")
        unknown = set(observed) - allowed_regime_keys
        changed = [key for key, value in expected.items() if observed.get(key) != value]
        if unknown or changed:
            raise LLMAblationContractError(
                f"BACE reference {name} changed: unknown={sorted(unknown)}, "
                f"core={changed}"
            )
    variants = payload.get("stage_variants")
    entries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(variants, Mapping):
        entries = [
            (str(key), value)
            for key, value in variants.items()
            if isinstance(value, Mapping)
        ]
    elif isinstance(variants, list):
        entries = [
            ("", value) for value in variants if isinstance(value, Mapping)
        ]
    a0 = next(
        (
            item
            for key, item in entries
            if key in {"A0", "A0_BRICS_FIXED"}
            or item.get("id") == "A0"
            or item.get("variant") == "BRICS_FIXED"
            or item.get("name") == "BRICS_FIXED"
        ),
        None,
    )
    if a0 is None or a0.get("status") != "CPU_FRAMEWORK_AVAILABLE":
        raise LLMAblationContractError("BACE reference does not authorize A0 CPU framework")
    return payload


def _build_proposals(
    *,
    vocabulary: Any,
    parent_ids: Sequence[str],
    train_rows: Mapping[str, Mapping[str, Any]],
    attempts_per_parent: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if Chem is None:
        raise LLMAblationContractError("RDKit is required for BRICS proposal matching")
    if attempts_per_parent <= 0:
        raise LLMAblationContractError("attempts_per_parent must be positive")
    queries = []
    for record in vocabulary.records:
        query = Chem.MolFromSmiles(record.fragment_smiles)
        if query is None:
            raise LLMAblationContractError("BRICS vocabulary contains an invalid query")
        queries.append((record, query))

    candidates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    shortfall_by_parent: dict[str, int] = {}
    for parent_id in parent_ids:
        source = train_rows[parent_id]
        parent = Chem.MolFromSmiles(str(source["smiles"]))
        if parent is None:
            raise LLMAblationContractError(f"invalid proposal parent: {parent_id}")
        matching = [
            record
            for record, query in queries
            if parent.HasSubstructMatch(query, useChirality=True)
        ]
        emitted = min(attempts_per_parent, len(matching))
        shortfall = attempts_per_parent - emitted
        if shortfall:
            shortfall_by_parent[parent_id] = shortfall
        for attempt_index in range(attempts_per_parent):
            record = matching[attempt_index] if attempt_index < emitted else None
            row: dict[str, Any] = {
                "schema_version": "brics_attempt_record_v2",
                "variant": "BRICS_FIXED",
                "parent_id": parent_id,
                "source_label": 1,
                "attempt_index": attempt_index,
                "fragment_smiles": record.fragment_smiles if record else None,
                "proposal_shortfall": record is None,
                "selection_policy": "parent_match_then_train_frequency",
                "proposal_match_use_chirality": True,
                "oracle_used": False,
                "calibration_loaded": False,
                "test_loaded": False,
                "vocabulary_sha256": vocabulary.sha256,
            }
            row["attempt_id"] = canonical_json_sha256(
                {
                    "parent_id": parent_id,
                    "attempt_index": attempt_index,
                    "vocabulary_sha256": vocabulary.sha256,
                }
            )
            if record is not None:
                row.update(
                    {
                        "vocabulary_rank": record.vocabulary_rank,
                        "train_frequency": record.train_frequency,
                        "source_parent_count": record.source_parent_count,
                    }
                )
                candidate = dict(row)
                candidate["schema_version"] = "brics_candidate_record_v2"
                candidate["candidate_id"] = canonical_json_sha256(
                    {
                        "parent_id": parent_id,
                        "fragment_smiles": record.fragment_smiles,
                        "attempt_index": attempt_index,
                    }
                )
                candidates.append(candidate)
            attempts.append(row)
    total_attempts = len(parent_ids) * attempts_per_parent
    summary = {
        "parent_count": len(parent_ids),
        "attempts_per_parent": attempts_per_parent,
        "proposal_attempts": total_attempts,
        "valid_candidates": len(candidates),
        "proposal_shortfall": total_attempts - len(candidates),
        "parents_with_shortfall": len(shortfall_by_parent),
        "shortfall_by_parent": shortfall_by_parent,
    }
    return candidates, attempts, summary


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.workers <= 2:
        raise LLMAblationContractError("BRICS CPU workers must be in [1, 2]")
    train_csv = _physical_file(args.train_csv, role="train_csv")
    cohort_path = _physical_file(args.proposal_cohort_manifest, role="proposal_cohort")
    reference_path = _physical_file(args.reference_contract, role="reference_contract")
    _assert_train_path(train_csv, role="train_csv")
    _assert_train_path(cohort_path, role="proposal_cohort")
    train_sha = require_sha256(args.train_csv_sha256, field="train_csv_sha256")
    rows, input_columns = _read_full_train_csv(
        train_csv,
        expected_sha256=train_sha,
        expected_rows=args.expected_train_rows,
    )
    by_id = {str(row["parent_id"]): row for row in rows}
    cohort, parent_ids = _read_cohort(
        cohort_path,
        expected_sha256=args.proposal_cohort_sha256,
        expected_parent_count=args.expected_proposal_parents,
        train_csv=train_csv,
        train_sha256=train_sha,
        train_rows=by_id,
    )
    reference = _read_reference_contract(
        reference_path,
        expected_sha256=args.reference_contract_sha256,
        train_csv=train_csv,
        train_sha256=train_sha,
        cohort_path=cohort_path,
        cohort_sha256=sha256_file(cohort_path),
        expected_parent_count=args.expected_proposal_parents,
        attempts_per_parent=args.attempts_per_parent,
    )
    output = Path(args.output_root).expanduser()
    if not output.is_absolute() or output.exists():
        raise LLMAblationContractError("output root must be a fresh absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    # The existing deterministic builder owns decomposition and ordering.  It
    # remains a single-writer path even when the admission cap is set to two.
    vocabulary = build_train_only_brics_vocabulary(
        training_molecules_from_mappings(
            {
                "molecule_id": row["parent_id"],
                "smiles": row["smiles"],
                "split": "train",
                "label": row["label"],
            }
            for row in rows
        )
    )
    vocab_path = output / "brics_vocab.jsonl"
    _atomic_jsonl(vocab_path, (asdict(record) for record in vocabulary.records))
    candidates, attempts, summary = _build_proposals(
        vocabulary=vocabulary,
        parent_ids=parent_ids,
        train_rows=by_id,
        attempts_per_parent=args.attempts_per_parent,
    )
    pool_path = output / "brics_proposal_pool.jsonl"
    attempts_path = output / "brics_proposal_attempts.jsonl"
    _atomic_jsonl(pool_path, candidates)
    _atomic_jsonl(attempts_path, attempts)

    now = datetime.now(timezone.utc).isoformat()
    vocab_manifest = {
        "schema_version": "bace_brics_vocab_manifest_v2",
        "status": "PASS",
        "dataset": "bace",
        "source_split": "train",
        "train_csv": _file_identity(train_csv),
        "reference_contract": _file_identity(reference_path),
        "reference_contract_self_sha256": reference["reference_contract_sha256"],
        "expected_train_rows": args.expected_train_rows,
        "observed_train_rows": len(rows),
        "input_columns_present": input_columns,
        "input_columns_read": ["molecule_id", "parent_id", "smiles", "label", "split"],
        "molecule_id_equals_parent_id": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "oracle_fields_read": [],
        "ranking_policy": "train_frequency_only_no_oracle",
        "main_filter_provenance": MAIN_FILTER_PROVENANCE,
        "numeric_size_filter_applied": False,
        "numeric_size_filter_reason": "MAIN_B10_CONTRACT_HAS_NO_NUMERIC_SIZE_FILTER",
        "vocabulary_size": len(vocabulary.records),
        "vocabulary_contract_sha256": vocabulary.sha256,
        "vocabulary_file": _file_identity(vocab_path),
        "requested_workers": args.workers,
        "effective_workers": 1,
        "created_at": now,
    }
    vocab_manifest_path = output / "brics_vocab_manifest.json"
    _atomic_json(vocab_manifest_path, vocab_manifest)

    shortfall_receipt = {
        "schema_version": "bace_brics_proposal_shortfall_receipt_v2",
        "status": "PASS",
        **summary,
        "proposal_cohort_manifest": _file_identity(cohort_path),
        "reference_contract": _file_identity(reference_path),
        "proposal_parent_ids_sha256": cohort["parent_ids_sha256"],
        "vocabulary_contract_sha256": vocabulary.sha256,
        "candidate_pool": _file_identity(pool_path),
        "attempt_records": _file_identity(attempts_path),
        "candidate_duplication_used": False,
        "oracle_ranking_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "shortfall_is_not_backfilled": True,
    }
    receipt_path = output / "brics_proposal_shortfall_receipt.json"
    _atomic_json(receipt_path, shortfall_receipt)
    proposal_manifest = {
        "schema_version": "bace_brics_proposal_manifest_v2",
        "status": "PASS",
        **summary,
        "attempt_matched": True,
        "source_label": 1,
        "proposal_cohort_manifest": _file_identity(cohort_path),
        "vocabulary_manifest": _file_identity(vocab_manifest_path),
        "candidate_pool": _file_identity(pool_path),
        "attempt_records": _file_identity(attempts_path),
        "shortfall_receipt": _file_identity(receipt_path),
        "selection_policy": "parent_match_then_train_frequency",
        "proposal_match_use_chirality": True,
        "main_filter_provenance": MAIN_FILTER_PROVENANCE,
        "common_downstream_hard_deletion_required": True,
        "oracle_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    proposal_manifest_path = output / "brics_proposal_manifest.json"
    _atomic_json(proposal_manifest_path, proposal_manifest)

    inventory_paths = (
        vocab_path,
        vocab_manifest_path,
        pool_path,
        attempts_path,
        receipt_path,
        proposal_manifest_path,
    )
    sha_lines = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in inventory_paths
    )
    _atomic_bytes(output / "brics_vocab_sha256s.txt", sha_lines.encode("utf-8"))
    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "output_root": str(output),
        "vocabulary_size": len(vocabulary.records),
        **summary,
        "train_csv_sha256": train_sha,
        "proposal_cohort_sha256": sha256_file(cohort_path),
        "sha256_inventory": _file_identity(output / "brics_vocab_sha256s.txt"),
        "gpu_used": False,
        "oracle_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    _atomic_json(output / "terminal.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-csv-sha256", required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--proposal-cohort-manifest", required=True)
    parser.add_argument("--proposal-cohort-sha256", required=True)
    parser.add_argument("--expected-proposal-parents", type=int, default=386)
    parser.add_argument("--reference-contract", required=True)
    parser.add_argument("--reference-contract-sha256", required=True)
    parser.add_argument("--attempts-per-parent", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), sort_keys=True))
