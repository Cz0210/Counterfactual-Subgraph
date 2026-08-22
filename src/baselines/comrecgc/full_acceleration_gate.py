"""Fail-closed release gate for optimized 50k-step BACE COMRECGC.

The bounded RDKit/cache implementation is deliberately opt-in.  A paper-eligible
50k run may select it only after exact 500- and 1000-step legacy/optimized
replays have both passed.  This module binds those two replays to one runtime
configuration, parent cohort, frozen GINE, distance checkpoint, and dataset
audit.  The resulting immutable JSON is checked again immediately before a
full run creates its output directory.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .bace_preprocessing import PREPROCESS_ENGINE
from .contracts import (
    CF_MODE,
    UPSTREAM_COMMIT,
    GenerationParameters,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from .equivalence import EQUIVALENCE_SCHEMA, FLOAT_ABS_TOLERANCE


FULL_ACCELERATION_GATE_SCHEMA = "bace_comrecgc_full_acceleration_gate_v1"
EQUIVALENCE_BUDGETS = (500, 1000)
FULL_GENERATION_STEPS = 50_000
HEX64 = re.compile(r"[0-9a-f]{64}")

_RUNTIME_KEYS = (
    "engine",
    "batch_size",
    "workers",
    "max_inflight",
    "source_cache_capacity",
    "candidate_cache_capacity",
)


class FullAccelerationGateError(ValueError):
    """The optimized full route is not covered by exact replay evidence."""


def _read_object(path: Path) -> dict[str, Any]:
    logical = path.expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise FullAccelerationGateError(f"gate input must be absolute and physical: {logical}")
    source = logical.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FullAccelerationGateError(f"gate input is not a nonempty file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FullAccelerationGateError(f"invalid JSON gate input: {source}") from exc
    if not isinstance(value, dict):
        raise FullAccelerationGateError(f"gate input must contain one object: {source}")
    return value


def _checkpoint_file(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    candidate = source / "model.pt" if source.is_dir() else source
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise FullAccelerationGateError(f"checkpoint is missing: {candidate}")
    return candidate


def _distance_hash(manifest: Mapping[str, Any]) -> str:
    distance = manifest.get("distance_model")
    if not isinstance(distance, Mapping):
        raise FullAccelerationGateError("equivalence manifest lacks distance_model")
    for key in ("checkpoint_sha256", "checkpoint_hash", "model_sha256"):
        value = str(distance.get(key) or "")
        if HEX64.fullmatch(value):
            return value
    raise FullAccelerationGateError("equivalence manifest lacks a distance checkpoint hash")


def _manifest_batch_size(manifest: Mapping[str, Any]) -> int:
    raw = manifest.get("bace_preprocessing")
    value: Any = raw.get("batch_size") if isinstance(raw, Mapping) else None
    if value is None:
        prefix = "--batch-size="
        values = [
            str(item)[len(prefix) :]
            for item in (manifest.get("scientific_argv") or ())
            if str(item).startswith(prefix)
        ]
        if len(values) != 1:
            raise FullAccelerationGateError(
                "equivalence replay lacks one canonical GINE batch size"
            )
        try:
            value = json.loads(values[0])
        except json.JSONDecodeError as exc:
            raise FullAccelerationGateError(
                "equivalence replay batch-size CLI binding is invalid"
            ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
        raise FullAccelerationGateError(
            "equivalence replay GINE batch size must be a positive integer"
        )
    return int(value)


def _runtime_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("bace_preprocessing")
    if not isinstance(raw, Mapping):
        raise FullAccelerationGateError("optimized replay lacks bace_preprocessing")
    contract = {
        "engine": raw.get("engine"),
        "batch_size": _manifest_batch_size(manifest),
        **{key: raw.get(key) for key in _RUNTIME_KEYS[2:]},
    }
    if contract["engine"] != PREPROCESS_ENGINE:
        raise FullAccelerationGateError("optimized replay does not use the reviewed engine")
    for key in _RUNTIME_KEYS[1:]:
        value = contract[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise FullAccelerationGateError(f"optimized replay has invalid {key}")
    if int(contract["workers"]) <= 0 or int(contract["max_inflight"]) <= 0:
        raise FullAccelerationGateError("optimized replay worker/inflight limits must be positive")
    if int(contract["batch_size"]) <= 0:
        raise FullAccelerationGateError("optimized replay GINE batch size must be positive")
    if int(contract["source_cache_capacity"]) <= 0 or int(
        contract["candidate_cache_capacity"]
    ) <= 0:
        raise FullAccelerationGateError("optimized replay caches must be positive")
    return contract


def _scientific_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    oracle_hash = str(manifest.get("oracle_checkpoint_hash") or "")
    parent_hash = str(manifest.get("generation_parent_ids_sha256") or "")
    if HEX64.fullmatch(oracle_hash) is None:
        raise FullAccelerationGateError("equivalence manifest lacks frozen-GINE hash")
    if HEX64.fullmatch(parent_hash) is None:
        raise FullAccelerationGateError("equivalence manifest lacks parent-cohort hash")
    dataset_audit = manifest.get("dataset_audit")
    if not isinstance(dataset_audit, Mapping) or not dataset_audit:
        raise FullAccelerationGateError("equivalence manifest lacks dataset audit")
    gnn = manifest.get("gnn")
    if (
        not isinstance(gnn, Mapping)
        or gnn.get("checkpoint_sha256") != oracle_hash
        or gnn.get("oracle_checkpoint_hash") != oracle_hash
        or gnn.get("oracle_backend") != "gnn"
        or gnn.get("classifier_family") != "gine"
        or gnn.get("rf_oracle_used") is not False
    ):
        raise FullAccelerationGateError(
            "equivalence manifest GINE provenance/hash closure is invalid"
        )
    checks = {
        "dataset": str(manifest.get("dataset") or "").lower() == "bace",
        "oracle_backend": manifest.get("oracle_backend") == "gnn",
        "classifier_family": manifest.get("classifier_family") == "gine",
        "rf_oracle_used": manifest.get("rf_oracle_used") is False,
        "cf_mode": manifest.get("cf_mode") == CF_MODE,
        "parent_limit": int(manifest.get("parent_limit", -1)) == 360,
        "upstream_commit": manifest.get("upstream_commit") == UPSTREAM_COMMIT,
        "calibration_unloaded": manifest.get("calibration_loaded") is False,
        "test_unloaded": manifest.get("test_loaded") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise FullAccelerationGateError(
            "equivalence scientific contract failed: " + ",".join(failures)
        )
    return {
        "dataset": "bace",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": oracle_hash,
        "distance_checkpoint_hash": _distance_hash(manifest),
        "generation_parent_ids_sha256": parent_hash,
        "dataset_audit_sha256": stable_json_sha256(dataset_audit),
        "gnn_batch_size": _manifest_batch_size(manifest),
        "cf_mode": CF_MODE,
        "parent_limit": 360,
        "upstream_commit": UPSTREAM_COMMIT,
        "calibration_loaded": False,
        "test_loaded": False,
        "full_parameters": asdict(GenerationParameters.for_mode("full")),
    }


def _equivalence_evidence(root: Path, *, budget: int) -> dict[str, Any]:
    source = root.expanduser()
    if not source.is_absolute() or source.is_symlink():
        raise FullAccelerationGateError(f"equivalence root must be absolute/physical: {source}")
    source = source.resolve(strict=True)
    if not source.is_dir() or not (source / "PASS").is_file():
        raise FullAccelerationGateError(f"equivalence root is not PASS: {source}")
    audit_path = source / "audit/equivalence_summary.json"
    audit_pass = source / "audit/PASS"
    if not audit_pass.is_file():
        raise FullAccelerationGateError(f"equivalence audit PASS is absent: {source}")
    audit = _read_object(audit_path)
    unsigned_audit = dict(audit)
    recorded_summary_sha256 = str(unsigned_audit.pop("summary_sha256", ""))
    if (
        audit.get("schema_version") != EQUIVALENCE_SCHEMA
        or audit.get("status") != "PASS"
        or int(audit.get("expected_steps", -1)) != int(budget)
        or audit.get("paper_eligible") is not False
        or audit.get("failures") not in ([], None)
        or audit.get("identity_mismatches") != []
        or audit.get("selected_trace_chunks_exact") is not True
        or HEX64.fullmatch(recorded_summary_sha256) is None
        or recorded_summary_sha256 != stable_json_sha256(unsigned_audit)
    ):
        raise FullAccelerationGateError(f"equivalence audit is not clean for {budget}")
    optimized_path = source / "optimized/run_manifest.json"
    legacy_path = source / "legacy/run_manifest.json"
    optimized = _read_object(optimized_path)
    legacy = _read_object(legacy_path)
    if Path(str(audit.get("legacy_root") or "")).resolve(strict=True) != (
        source / "legacy"
    ) or Path(str(audit.get("optimized_root") or "")).resolve(strict=True) != (
        source / "optimized"
    ):
        raise FullAccelerationGateError("equivalence audit roots differ from gate roots")
    payload = audit.get("payload")
    if not isinstance(payload, Mapping):
        raise FullAccelerationGateError("equivalence audit lacks payload comparison")
    candidate_parity = payload.get("candidate_parity")
    payload_checks = {
        "payload_failures": payload.get("failures") == [],
        "candidate_trace_parity": isinstance(candidate_parity, Mapping)
        and candidate_parity.get("trace_parity_passed") is True,
        "graph_map_key_order": payload.get("graph_map_key_order_exact") is True,
        "graph_index_map": payload.get("graph_index_map_exact") is True,
        "traversed_hashes": payload.get("traversed_hashes_exact") is True,
        "input_coverage": float(
            payload.get("input_graphs_covered_max_abs_difference", math.inf)
        )
        == 0.0,
        "graph_identity": int(payload.get("graph_identity_mismatch_count", -1))
        == 0,
        "graph_values": float(
            payload.get("graph_embedding_or_element_max_abs_difference", math.inf)
        )
        <= FLOAT_ABS_TOLERANCE,
        "tolerance": float(payload.get("float_abs_tolerance", math.inf))
        == FLOAT_ABS_TOLERANCE,
    }
    if not all(payload_checks.values()):
        failures = [key for key, passed in payload_checks.items() if not passed]
        raise FullAccelerationGateError(
            "equivalence payload comparison is incomplete: " + ",".join(failures)
        )
    for role, manifest in (("legacy", legacy), ("optimized", optimized)):
        expected = replace(GenerationParameters.for_mode("full"), steps=int(budget))
        if (
            manifest.get("run_complete") is not True
            or manifest.get("diagnostic_only") is not True
            or manifest.get("paper_eligible") is not False
            or int(manifest.get("diagnostic_equivalence_steps", -1)) != int(budget)
            or manifest.get("equivalence_gate_role") != role
            or manifest.get("parameters") != asdict(expected)
        ):
            raise FullAccelerationGateError(
                f"{role} manifest is not an exact {budget}-step full-prefix replay"
            )
        role_root = source / role
        counterfactuals = role_root / "counterfactuals.pt"
        completion_path = role_root / "_RUN_COMPLETE.json"
        diagnostic_path = role_root / "DIAGNOSTIC_ONLY.json"
        actual_payload_sha256 = sha256_file(counterfactuals)
        completion = _read_object(completion_path)
        diagnostic = _read_object(diagnostic_path)
        if (
            manifest.get("counterfactuals_sha256") != actual_payload_sha256
            or completion.get("run_complete") is not True
            or completion.get("counterfactuals_sha256") != actual_payload_sha256
            or diagnostic.get("diagnostic_only") is not True
            or diagnostic.get("paper_eligible") is not False
            or int(diagnostic.get("steps", -1)) != int(budget)
            or diagnostic.get("role") != role
            or audit.get(f"{role}_counterfactuals_sha256")
            != actual_payload_sha256
        ):
            raise FullAccelerationGateError(
                f"{role} payload/completion diagnostic closure is invalid"
            )
        trace_rows = audit.get(f"{role}_trace_chunks")
        if not isinstance(trace_rows, list) or not trace_rows:
            raise FullAccelerationGateError(f"{role} trace evidence is absent")
        for trace_row in trace_rows:
            if not isinstance(trace_row, Mapping):
                raise FullAccelerationGateError(f"{role} trace row is invalid")
            relative = Path(str(trace_row.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise FullAccelerationGateError(f"{role} trace path is unsafe")
            candidates = (
                role_root / "_native_aux/trace" / relative,
                role_root / "trace" / relative,
            )
            matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
            if len(matches) != 1 or sha256_file(matches[0]) != trace_row.get("sha256"):
                raise FullAccelerationGateError(f"{role} trace chunk hash mismatch")
            row_count = int(trace_row.get("row_count", -1))
            actual_row_count = sum(
                1
                for line in matches[0].read_text(
                    encoding="utf-8", errors="strict"
                ).splitlines()
                if line.strip()
            )
            if row_count <= 0 or actual_row_count != row_count:
                raise FullAccelerationGateError(f"{role} trace row count is invalid")
    scientific = _scientific_contract(optimized)
    if _scientific_contract(legacy) != scientific:
        raise FullAccelerationGateError("legacy/optimized scientific contracts differ")
    return {
        "budget": int(budget),
        "root": str(source),
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "legacy_manifest": str(legacy_path),
        "legacy_manifest_sha256": sha256_file(legacy_path),
        "optimized_manifest": str(optimized_path),
        "optimized_manifest_sha256": sha256_file(optimized_path),
        "runtime_contract": _runtime_contract(optimized),
        "scientific_contract": scientific,
    }


def build_full_acceleration_gate(
    *, m500_root: str | Path, m1000_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Publish one immutable exact-500+1000 gate for the optimized full route."""

    evidence = [
        _equivalence_evidence(Path(m500_root), budget=500),
        _equivalence_evidence(Path(m1000_root), budget=1000),
    ]
    runtimes = {stable_json_sha256(row["runtime_contract"]) for row in evidence}
    sciences = {stable_json_sha256(row["scientific_contract"]) for row in evidence}
    if len(runtimes) != 1:
        raise FullAccelerationGateError("500/1000 optimized runtime contracts differ")
    if len(sciences) != 1:
        raise FullAccelerationGateError("500/1000 scientific contracts differ")
    root = require_empty_output(output_dir)
    payload = {
        "schema_version": FULL_ACCELERATION_GATE_SCHEMA,
        "status": "PASS",
        "marker": "[COMRECGC_PARALLEL_EQUIVALENCE_PASS]",
        "dataset": "bace",
        "method": "ComRecGC",
        "budgets": list(EQUIVALENCE_BUDGETS),
        "full_generation_steps": FULL_GENERATION_STEPS,
        "generation_index_sharded": False,
        "runtime_contract": evidence[0]["runtime_contract"],
        "runtime_contract_sha256": next(iter(runtimes)),
        "scientific_contract": evidence[0]["scientific_contract"],
        "scientific_contract_sha256": next(iter(sciences)),
        "equivalence_evidence": evidence,
    }
    payload["gate_payload_sha256"] = stable_json_sha256(payload)
    write_json(root / "FULL_ACCELERATION_GATE.json", payload)
    (root / "PASS").write_text(
        "[COMRECGC_PARALLEL_EQUIVALENCE_PASS]\n", encoding="utf-8"
    )
    return payload


def validate_full_acceleration_gate(
    gate_path: str | Path,
    *,
    expected_gate_sha256: str,
    gnn_checkpoint: str | Path,
    distance_checkpoint: str | Path,
    generation_parent_ids_sha256: str,
    dataset_audit: Mapping[str, Any],
    parent_limit: int,
    parameters: GenerationParameters,
    preprocess_engine: str,
    batch_size: int,
    preprocess_workers: int,
    preprocess_max_inflight: int,
    source_cache_capacity: int,
    candidate_cache_capacity: int,
) -> dict[str, Any]:
    """Revalidate gate bytes and all full-run identities before output creation."""

    path = Path(gate_path).expanduser()
    actual_gate_hash = sha256_file(path)
    if HEX64.fullmatch(str(expected_gate_sha256)) is None or actual_gate_hash != str(
        expected_gate_sha256
    ):
        raise FullAccelerationGateError("full acceleration gate SHA256 mismatch")
    payload = _read_object(path)
    unsigned_payload = dict(payload)
    recorded_payload_sha256 = str(
        unsigned_payload.pop("gate_payload_sha256", "")
    )
    if (
        payload.get("schema_version") != FULL_ACCELERATION_GATE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("marker") != "[COMRECGC_PARALLEL_EQUIVALENCE_PASS]"
        or payload.get("budgets") != list(EQUIVALENCE_BUDGETS)
        or int(payload.get("full_generation_steps", -1)) != FULL_GENERATION_STEPS
        or payload.get("generation_index_sharded") is not False
        or HEX64.fullmatch(recorded_payload_sha256) is None
        or recorded_payload_sha256 != stable_json_sha256(unsigned_payload)
    ):
        raise FullAccelerationGateError("full acceleration gate header is invalid")
    embedded_evidence = payload.get("equivalence_evidence")
    if not isinstance(embedded_evidence, list) or len(embedded_evidence) != 2:
        raise FullAccelerationGateError("full acceleration gate lacks two replay roots")
    rebuilt_evidence: list[dict[str, Any]] = []
    for row, budget in zip(embedded_evidence, EQUIVALENCE_BUDGETS, strict=True):
        if not isinstance(row, Mapping):
            raise FullAccelerationGateError(
                "full acceleration replay evidence is invalid"
            )
        rebuilt_evidence.append(
            _equivalence_evidence(Path(str(row.get("root") or "")), budget=budget)
        )
    if rebuilt_evidence != embedded_evidence:
        raise FullAccelerationGateError(
            "full acceleration replay evidence changed after gate publication"
        )
    runtime_hash = stable_json_sha256(payload.get("runtime_contract"))
    scientific_hash = stable_json_sha256(payload.get("scientific_contract"))
    if (
        payload.get("runtime_contract_sha256") != runtime_hash
        or payload.get("scientific_contract_sha256") != scientific_hash
        or any(
            row.get("runtime_contract") != payload.get("runtime_contract")
            or row.get("scientific_contract") != payload.get("scientific_contract")
            for row in rebuilt_evidence
        )
    ):
        raise FullAccelerationGateError(
            "full acceleration replay config/scientific aggregate is invalid"
        )
    if parameters != GenerationParameters.for_mode("full"):
        raise FullAccelerationGateError("optimized paper run must retain all 50k parameters")
    runtime = {
        "engine": preprocess_engine,
        "batch_size": int(batch_size),
        "workers": int(preprocess_workers),
        "max_inflight": int(preprocess_max_inflight),
        "source_cache_capacity": int(source_cache_capacity),
        "candidate_cache_capacity": int(candidate_cache_capacity),
    }
    if runtime != payload.get("runtime_contract"):
        raise FullAccelerationGateError("full optimized runtime differs from replay gate")
    scientific = payload.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise FullAccelerationGateError("gate lacks scientific contract")
    actual = {
        "oracle_checkpoint_hash": sha256_file(_checkpoint_file(gnn_checkpoint)),
        "distance_checkpoint_hash": sha256_file(_checkpoint_file(distance_checkpoint)),
        "generation_parent_ids_sha256": str(generation_parent_ids_sha256),
        "dataset_audit_sha256": stable_json_sha256(dataset_audit),
        "parent_limit": int(parent_limit),
    }
    failures = [key for key, value in actual.items() if scientific.get(key) != value]
    fixed = {
        "dataset": "bace",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "cf_mode": CF_MODE,
        "upstream_commit": UPSTREAM_COMMIT,
        "calibration_loaded": False,
        "test_loaded": False,
        "full_parameters": asdict(GenerationParameters.for_mode("full")),
    }
    failures.extend(key for key, value in fixed.items() if scientific.get(key) != value)
    if failures:
        raise FullAccelerationGateError(
            "full run differs from equivalence gate: " + ",".join(sorted(set(failures)))
        )
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": actual_gate_hash,
        "status": "PASS",
        "budgets": list(EQUIVALENCE_BUDGETS),
        "runtime_contract_sha256": payload.get("runtime_contract_sha256"),
        "scientific_contract_sha256": payload.get("scientific_contract_sha256"),
    }


__all__ = [
    "EQUIVALENCE_BUDGETS",
    "FULL_ACCELERATION_GATE_SCHEMA",
    "FULL_GENERATION_STEPS",
    "FullAccelerationGateError",
    "build_full_acceleration_gate",
    "validate_full_acceleration_gate",
]
