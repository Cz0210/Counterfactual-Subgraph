"""Fail-closed BACE B6--B14 dependency and data-split boundaries.

This module does not claim to implement B8--B14 science.  It gives the
controller one explicit release contract so downstream launchers cannot load
calibration or test early and cannot treat a READY decision as a scientific
PASS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE_SPLIT_CONTRACT = {
    "B6_PPO_SMOKE_V2": "train",
    "B7_PPO_FULL": "train",
    "B8_BASE_POOL": "train",
    "B9_HIGH_TEMP_POOL": "train",
    "B10_POOL_MERGE": "train",
    "B11_CALIBRATION_VERIFICATION": "calibration",
    "B12_SELECTOR_FREEZE": "calibration",
    "B13_TEST_EVALUATION": "test",
    "B14_FINAL_AUDIT": "manifest_only",
}

STAGE_DEPENDENCY_CONTRACT = {
    "B6_PPO_SMOKE_V2": ("B5_ORACLE_SMOKE",),
    "B7_PPO_FULL": ("B6_PPO_SMOKE_V2",),
    "B8_BASE_POOL": ("B7_PPO_FULL",),
    "B9_HIGH_TEMP_POOL": ("B7_PPO_FULL",),
    "B10_POOL_MERGE": ("B8_BASE_POOL",) * 4 + ("B9_HIGH_TEMP_POOL",) * 4,
    "B11_CALIBRATION_VERIFICATION": ("B10_POOL_MERGE",),
    "B12_SELECTOR_FREEZE": ("B11_CALIBRATION_VERIFICATION",),
    "B13_TEST_EVALUATION": ("B12_SELECTOR_FREEZE",),
    "B14_FINAL_AUDIT": ("B13_TEST_EVALUATION",),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selector_freeze_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = payload.get("ordered_rule_ids")
    required_nonempty = (
        "rule_hashes",
        "thresholds",
        "selector_config",
        "calibration_input_hash",
        "candidate_pool_hash",
        "gnn_checkpoint_hash",
        "molclr_checkpoint_hash",
    )
    failures = []
    if payload.get("selection_split") != "calibration":
        failures.append("selection_split")
    if payload.get("selector_fitted_on_calibration") is not True:
        failures.append("selector_fitted_on_calibration")
    if payload.get("test_used") is not False:
        failures.append("test_used")
    if payload.get("selection_frozen") is not True:
        failures.append("selection_frozen")
    if not isinstance(ordered, list) or len(ordered) != 20 or len(set(ordered)) != 20:
        failures.append("ordered_rule_ids")
    failures.extend(
        key for key in required_nonempty if payload.get(key) in (None, "", [], {})
    )
    if failures:
        raise ValueError("BACE selector freeze gate failed: " + ", ".join(failures))
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "manifest": payload,
    }


def validate_stage_data_access(
    *,
    stage: str,
    requested_split: str,
    selector_manifest: str | Path | None = None,
) -> dict[str, Any]:
    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in STAGE_SPLIT_CONTRACT:
        raise ValueError(f"Unknown BACE stage: {stage}")
    expected = STAGE_SPLIT_CONTRACT[normalized_stage]
    observed = str(requested_split).strip().lower()
    if observed != expected:
        raise ValueError(
            f"BACE {normalized_stage} may load {expected} only, not {observed}"
        )
    selector = None
    if expected == "test":
        if selector_manifest is None:
            raise ValueError("BACE test access requires the frozen B12 selector manifest")
        selector = validate_selector_freeze_manifest(selector_manifest)
    elif expected == "manifest_only":
        if selector_manifest is not None:
            selector = validate_selector_freeze_manifest(selector_manifest)
    elif selector_manifest is not None:
        raise ValueError("Selector manifest must not be used to unlock pre-test stages")
    return {
        "schema_version": "bace_stage_data_access_v1",
        "stage": normalized_stage,
        "requested_split": observed,
        "allowed": True,
        "selector_freeze": selector,
        "raw_test_loaded": expected == "test",
        "test_used_only_after_freeze": expected != "test" or selector is not None,
    }


def validate_pass_dependencies(
    manifests: Sequence[str | Path], *, expected_stages: Sequence[str]
) -> list[dict[str, Any]]:
    if len(manifests) != len(expected_stages):
        raise ValueError("BACE dependency manifest/stage cardinality mismatch")
    resolved: list[dict[str, Any]] = []
    for raw_path, expected_stage in zip(manifests, expected_stages, strict=True):
        path = Path(raw_path).expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage") != expected_stage or payload.get("status") != "PASS":
            raise ValueError(
                f"BACE dependency is not a PASS for {expected_stage}: {path}"
            )
        if payload.get("oracle_backend") != "gnn" or payload.get("rf_oracle_used") is not False:
            raise ValueError(f"BACE dependency violates the frozen-GNN/RF guard: {path}")
        resolved.append(
            {"path": str(path), "sha256": _sha256_file(path), "manifest": payload}
        )
    return resolved


__all__ = [
    "STAGE_SPLIT_CONTRACT",
    "STAGE_DEPENDENCY_CONTRACT",
    "validate_pass_dependencies",
    "validate_selector_freeze_manifest",
    "validate_stage_data_access",
]
