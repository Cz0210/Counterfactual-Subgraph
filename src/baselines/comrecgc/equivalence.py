"""Fail-closed legacy/optimized equivalence audit for BACE COMRECGC."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import require_empty_output, sha256_file, stable_json_sha256, write_json
from .graph_trace import assert_trace_parity, stable_untyped_graph_sha256


EQUIVALENCE_SCHEMA = "bace_comrecgc_generation_equivalence_v1"
FLOAT_ABS_TOLERANCE = 1e-6


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("COMRECGC equivalence audit requires PyTorch.") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location="cpu")


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    return value


def _numeric_difference(left: Any, right: Any) -> float:
    left_plain = _plain(left)
    right_plain = _plain(right)
    if isinstance(left_plain, list) and isinstance(right_plain, list):
        if len(left_plain) != len(right_plain):
            return math.inf
        return max(
            (_numeric_difference(a, b) for a, b in zip(left_plain, right_plain, strict=True)),
            default=0.0,
        )
    try:
        return abs(float(left_plain) - float(right_plain))
    except (TypeError, ValueError):
        return 0.0 if left_plain == right_plain else math.inf


def _trace_chunks(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "_native_aux/trace/selected_action_trace_manifest.json"
    if not manifest_path.is_file():
        # Direct CLI equivalence runs place trace at ``trace`` by default.
        manifest_path = root / "trace/selected_action_trace_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Selected action trace manifest is missing below {root}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    for row in manifest.get("chunks") or ():
        path = manifest_path.parent / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Selected trace chunk is missing: {path}")
        chunks.append(
            {
                "path": str(row["path"]),
                "row_count": int(row["row_count"]),
                "sha256": sha256_file(path),
            }
        )
    if sum(row["row_count"] for row in chunks) != int(manifest.get("row_count", -1)):
        raise ValueError("Selected action trace manifest row count is inconsistent.")
    return chunks


def _validate_run(root: Path, *, role: str, expected_steps: int) -> dict[str, Any]:
    required = (
        "counterfactuals.pt",
        "run_manifest.json",
        "_RUN_COMPLETE.json",
        "DIAGNOSTIC_ONLY.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{role} equivalence run is incomplete: {missing}")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "_RUN_COMPLETE.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        (root / "DIAGNOSTIC_ONLY.json").read_text(encoding="utf-8")
    )
    payload_sha256 = sha256_file(root / "counterfactuals.pt")
    checks = {
        "run_complete": manifest.get("run_complete") is True,
        "completion_marker": completion.get("run_complete") is True,
        "completion_payload_hash": completion.get("counterfactuals_sha256")
        == payload_sha256,
        "manifest_payload_hash": manifest.get("counterfactuals_sha256")
        == payload_sha256,
        "dataset_bace": str(manifest.get("dataset", "")).lower() == "bace",
        "diagnostic_only": manifest.get("diagnostic_only") is True,
        "diagnostic_marker": diagnostic.get("diagnostic_only") is True,
        "diagnostic_marker_paper_ineligible": diagnostic.get("paper_eligible")
        is False,
        "diagnostic_marker_steps": int(diagnostic.get("steps", -1))
        == int(expected_steps),
        "diagnostic_marker_role": diagnostic.get("role") == role,
        "paper_ineligible": manifest.get("paper_eligible") is False,
        "step_count": int(manifest.get("diagnostic_equivalence_steps", -1))
        == int(expected_steps),
        "role": manifest.get("equivalence_gate_role") == role,
        "traversed_steps": int(manifest.get("traversed_step_count", -1))
        == int(expected_steps),
        "oracle_gnn": manifest.get("oracle_backend") == "gnn",
        "classifier_gine": manifest.get("classifier_family") == "gine",
        "rf_forbidden": manifest.get("rf_oracle_used") is False,
        "calibration_unloaded": manifest.get("calibration_loaded") is False,
        "test_unloaded": manifest.get("test_loaded") is False,
        "dataset_audit_present": bool(manifest.get("dataset_audit")),
    }
    engine = str((manifest.get("bace_preprocessing") or {}).get("engine", ""))
    checks["engine_role"] = (
        (role == "legacy" and engine == "legacy_sequential_rdkit_v1")
        or (role == "optimized" and engine == "ordered_bounded_rdkit_process_pool_v1")
    )
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"{role} equivalence run failed provenance: {failures}")
    return manifest


def _payload_equivalence(
    legacy: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    parity = assert_trace_parity(legacy, optimized)
    legacy_keys = list((legacy.get("graph_map") or {}).keys())
    optimized_keys = list((optimized.get("graph_map") or {}).keys())
    failures: list[str] = []
    if legacy_keys != optimized_keys:
        failures.append("graph_map_key_order")
    legacy_index = _plain(legacy.get("graph_index_map") or {})
    optimized_index = _plain(optimized.get("graph_index_map") or {})
    if legacy_index != optimized_index:
        failures.append("graph_index_map")
    if _plain(legacy.get("traversed_hashes") or []) != _plain(
        optimized.get("traversed_hashes") or []
    ):
        failures.append("traversed_hashes")
    coverage_difference = _numeric_difference(
        legacy.get("input_graphs_covered"), optimized.get("input_graphs_covered")
    )
    if coverage_difference != 0.0:
        failures.append("input_graphs_covered")
    max_embedding_difference = 0.0
    graph_identity_mismatch_count = 0
    if legacy_keys == optimized_keys:
        legacy_map = legacy["graph_map"]
        optimized_map = optimized["graph_map"]
        for key in legacy_keys:
            left = legacy_map[key]
            right = optimized_map[key]
            if stable_untyped_graph_sha256(left[0]) != stable_untyped_graph_sha256(
                right[0]
            ):
                graph_identity_mismatch_count += 1
            max_embedding_difference = max(
                max_embedding_difference,
                _numeric_difference(left[1:], right[1:]),
            )
    if graph_identity_mismatch_count:
        failures.append("graph_content")
    if max_embedding_difference > FLOAT_ABS_TOLERANCE:
        failures.append("graph_embedding_or_element_values")
    return {
        "candidate_parity": parity,
        "graph_map_key_count": len(legacy_keys),
        "graph_map_key_order_exact": legacy_keys == optimized_keys,
        "graph_index_map_exact": legacy_index == optimized_index,
        "traversed_hashes_exact": _plain(legacy.get("traversed_hashes") or [])
        == _plain(optimized.get("traversed_hashes") or []),
        "input_graphs_covered_max_abs_difference": coverage_difference,
        "graph_identity_mismatch_count": graph_identity_mismatch_count,
        "graph_embedding_or_element_max_abs_difference": max_embedding_difference,
        "float_abs_tolerance": FLOAT_ABS_TOLERANCE,
        "failures": failures,
    }


def audit_generation_equivalence(
    *,
    legacy_root: str | Path,
    optimized_root: str | Path,
    output_dir: str | Path,
    expected_steps: int,
) -> dict[str, Any]:
    """Audit two fresh diagnostic prefixes and publish PASS only on parity."""

    if int(expected_steps) not in {500, 1000}:
        raise ValueError("The formal equivalence gate accepts only 500/1000 steps.")
    legacy_dir = Path(legacy_root).expanduser().resolve()
    optimized_dir = Path(optimized_root).expanduser().resolve()
    output = require_empty_output(output_dir)
    try:
        legacy_manifest = _validate_run(
            legacy_dir, role="legacy", expected_steps=expected_steps
        )
        optimized_manifest = _validate_run(
            optimized_dir, role="optimized", expected_steps=expected_steps
        )
        identity_fields = (
            "upstream_commit",
            "generation_parent_ids_sha256",
            "oracle_checkpoint_hash",
            "cf_mode",
            "parent_limit",
        )
        identity_mismatches = [
            field
            for field in identity_fields
            if legacy_manifest.get(field) != optimized_manifest.get(field)
        ]
        if legacy_manifest.get("parameters") != optimized_manifest.get("parameters"):
            identity_mismatches.append("parameters")
        if legacy_manifest.get("dataset_audit") != optimized_manifest.get(
            "dataset_audit"
        ):
            identity_mismatches.append("dataset_audit")
        if legacy_manifest.get("internal_prediction_counts") != optimized_manifest.get(
            "internal_prediction_counts"
        ):
            identity_mismatches.append("internal_prediction_counts")
        if legacy_manifest.get("distance_model") != optimized_manifest.get(
            "distance_model"
        ):
            identity_mismatches.append("distance_model")
        legacy_payload = _torch_load(legacy_dir / "counterfactuals.pt")
        optimized_payload = _torch_load(optimized_dir / "counterfactuals.pt")
        payload = _payload_equivalence(legacy_payload, optimized_payload)
        legacy_trace = _trace_chunks(legacy_dir)
        optimized_trace = _trace_chunks(optimized_dir)
        trace_exact = legacy_trace == optimized_trace
        failures = [*identity_mismatches, *payload["failures"]]
        if not trace_exact:
            failures.append("selected_action_trace_chunks")
        summary = {
            "schema_version": EQUIVALENCE_SCHEMA,
            "status": "PASS" if not failures else "FAIL",
            "expected_steps": int(expected_steps),
            "legacy_root": str(legacy_dir),
            "optimized_root": str(optimized_dir),
            "legacy_counterfactuals_sha256": sha256_file(
                legacy_dir / "counterfactuals.pt"
            ),
            "optimized_counterfactuals_sha256": sha256_file(
                optimized_dir / "counterfactuals.pt"
            ),
            "identity_fields": list(identity_fields),
            "identity_mismatches": identity_mismatches,
            "payload": payload,
            "selected_trace_chunks_exact": trace_exact,
            "legacy_trace_chunks": legacy_trace,
            "optimized_trace_chunks": optimized_trace,
            "failures": failures,
            "paper_eligible": False,
        }
    except Exception as exc:
        summary = {
            "schema_version": EQUIVALENCE_SCHEMA,
            "status": "FAIL",
            "expected_steps": int(expected_steps),
            "legacy_root": str(legacy_dir),
            "optimized_root": str(optimized_dir),
            "failures": [f"{type(exc).__name__}:{exc}"],
            "paper_eligible": False,
        }
    summary["summary_sha256"] = stable_json_sha256(summary)
    write_json(output / "equivalence_summary.json", summary)
    if summary["status"] == "PASS":
        (output / "PASS").write_text(
            "BACE COMRECGC legacy/optimized equivalence passed.\n",
            encoding="utf-8",
        )
        return summary
    write_json(output / "FAIL.json", summary)
    raise RuntimeError(
        "BACE COMRECGC equivalence gate failed: "
        + ",".join(str(value) for value in summary["failures"])
    )


__all__ = ["EQUIVALENCE_SCHEMA", "audit_generation_equivalence"]
