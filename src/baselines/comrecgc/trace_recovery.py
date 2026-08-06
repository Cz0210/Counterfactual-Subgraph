"""Adopt a complete COMRECGC trace run blocked only by an obsolete parity gate."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    UPSTREAM_COMMIT,
    GenerationParameters,
    require_empty_output,
    sha256_file,
    write_json,
)
from .graph_trace import (
    assert_trace_parity,
    load_selected_trace,
    recover_candidate_lineage_from_selected_trace,
)
from .runtime import validate_counterfactual_payload


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("COMRECGC trace recovery requires PyTorch.") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported PyTorch
        return torch.load(path, map_location="cpu")


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _materialize(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"Recovery destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        _atomic_copy(source, destination)
        mode = "atomic_copy"
    if destination.stat().st_size != source.stat().st_size:
        raise ValueError(f"Recovered artifact size mismatch: {destination}")
    if sha256_file(destination) != sha256_file(source):
        raise ValueError(f"Recovered artifact SHA256 mismatch: {destination}")
    return mode


def _validate_failed_source(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    failure = _load_json(source_root / "_RUN_FAILED.json")
    if failure.get("stage") != "project_generation":
        raise ValueError("Trace adoption requires a project_generation failure.")
    if failure.get("error_class") != "ValueError" or "Trace-on/off" not in str(
        failure.get("message")
    ):
        raise ValueError("Trace adoption refuses a non-parity algorithm failure.")
    if failure.get("calibration_loaded") is not False or failure.get("test_loaded") is not False:
        raise ValueError("Trace adoption source violates calibration/test isolation.")

    config = _load_json(source_root / "resolved_config.json")
    expected_parameters = GenerationParameters.for_mode("smoke").__dict__
    expected_fields: Mapping[str, Any] = {
        "dataset": "mutagenicity",
        "mode": "smoke",
        "parent_limit": 64,
        "parameters": expected_parameters,
        "upstream_commit": UPSTREAM_COMMIT,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for field, expected in expected_fields.items():
        if config.get(field) != expected:
            raise ValueError(
                f"Trace adoption source config mismatch: field={field}, "
                f"actual={config.get(field)!r}, expected={expected!r}."
            )
    parent_ids = list(config.get("generation_parent_ids") or [])
    if len(parent_ids) != 64 or len(set(parent_ids)) != 64:
        raise ValueError("Trace adoption source must contain 64 unique parent IDs.")
    return failure, config


def recover_mutagenicity_trace_run(
    *,
    source_failed_generation_dir: str | Path,
    reference_counterfactuals_path: str | Path,
    output_dir: str | Path,
    expected_reference_sha256: str,
    expected_candidate_count: int = 164,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and materialize an existing trace without rerunning COMRECGC."""

    source_root = Path(source_failed_generation_dir).expanduser().resolve()
    reference_path = Path(reference_counterfactuals_path).expanduser().resolve()
    output_root = require_empty_output(output_dir)
    failure, config = _validate_failed_source(source_root)
    if sha256_file(reference_path) != expected_reference_sha256:
        raise ValueError("Frozen trace parity reference SHA256 mismatch.")

    source_counterfactuals = source_root / "counterfactuals.pt"
    source_payload = _torch_load(source_counterfactuals)
    reference_payload = _torch_load(reference_path)
    graph_map, candidates = validate_counterfactual_payload(source_payload)
    if len(candidates) != int(expected_candidate_count):
        raise ValueError(
            "Recovered trace candidate count mismatch: "
            f"actual={len(candidates)}, expected={expected_candidate_count}."
        )
    parity = assert_trace_parity(reference_payload, source_payload)
    parity.update(
        {
            "reference_path": str(reference_path),
            "reference_sha256": expected_reference_sha256,
            "source_trace_counterfactuals_path": str(source_counterfactuals),
            "source_trace_counterfactuals_sha256": sha256_file(
                source_counterfactuals
            ),
        }
    )

    source_trace = source_root / "trace"
    source_trace_complete = _load_json(source_trace / "_TRACE_COMPLETE.json")
    if source_trace_complete.get("trace_complete") is not True:
        raise ValueError("Source trace does not have a valid completion marker.")
    selected_manifest = source_trace / "selected_action_trace_manifest.json"
    if sha256_file(selected_manifest) != str(
        source_trace_complete.get("selected_trace_manifest_sha256")
    ):
        raise ValueError("Source selected trace manifest SHA256 mismatch.")
    selected_events = load_selected_trace(selected_manifest)
    source_graphs_by_parent_id = None
    source_dataset_fingerprint = None
    if dataset_dir is not None:
        from .project_dataset import load_mutagenicity_generation_bundle

        source_bundle = load_mutagenicity_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=int(config["parent_limit"]),
        )
        if source_bundle.parent_ids != list(config["generation_parent_ids"]):
            raise ValueError(
                "Frozen trace source parent order differs from resolved generation config."
            )
        source_graphs_by_parent_id = dict(
            zip(source_bundle.parent_ids, source_bundle.graphs, strict=True)
        )
        source_dataset_fingerprint = source_bundle.dataset_fingerprint
    lineage = recover_candidate_lineage_from_selected_trace(
        source_payload,
        selected_events,
        source_graphs_by_parent_id=source_graphs_by_parent_id,
    )
    if len(lineage) != int(expected_candidate_count) or any(
        row.get("action_lineage_resolved") is not True for row in lineage
    ):
        raise ValueError("Recovered action lineage is incomplete.")
    inferred_action_count = sum(
        event.get("action_recovery") == "inferred_exact_graph_delta_v1"
        for row in lineage
        for event in row["actions"]
    )
    recorded_action_count = sum(
        event.get("action_recovery") == "recorded_exact"
        for row in lineage
        for event in row["actions"]
    )
    transition_count = sum(len(row["actions"]) for row in lineage)
    replay_exact_count = sum(
        event.get("action_replay_exact") is True
        for row in lineage
        for event in row["actions"]
    )
    if replay_exact_count != transition_count:
        raise ValueError("Recovered action lineage does not replay exactly.")
    zero_action_source_root_count = sum(
        row.get("zero_action_source_root") is True for row in lineage
    )
    lineage_replay = {
        "num_transitions": transition_count,
        "num_canonical_actions": transition_count,
        "num_ambiguous_reconstructions": 0,
        "num_replay_exact": replay_exact_count,
        "num_replay_failed": 0,
        "num_zero_action_source_roots": zero_action_source_root_count,
    }

    counterfactuals_path = output_root / "counterfactuals.pt"
    counterfactuals_mode = _materialize(
        source_counterfactuals, counterfactuals_path
    )
    resolved_config_path = output_root / "resolved_config.json"
    resolved_config_mode = _materialize(
        source_root / "resolved_config.json", resolved_config_path
    )
    output_trace = output_root / "trace"
    output_trace.mkdir(parents=True, exist_ok=True)
    selected_manifest_payload = _load_json(selected_manifest)
    trace_materialization_modes: dict[str, str] = {}
    for chunk in selected_manifest_payload.get("chunks") or []:
        relative = Path(str(chunk["path"]))
        source_chunk = source_trace / relative
        destination_chunk = output_trace / relative
        trace_materialization_modes[relative.as_posix()] = _materialize(
            source_chunk, destination_chunk
        )
    trace_materialization_modes[selected_manifest.name] = _materialize(
        selected_manifest, output_trace / selected_manifest.name
    )

    lineage_path = output_trace / "candidate_action_lineage.json"
    write_json(lineage_path, lineage)
    trace_summary = {
        **_load_json(source_trace / "trace_summary.json"),
        "candidate_lineage_path": str(lineage_path),
        "candidate_lineage_resolved_count": len(lineage),
        "selected_trace_path": str(output_trace / selected_manifest.name),
        "lineage_recovery_policy": "pinned_upstream_official_hash_source_root_v2",
        "source_dataset_dir": None if dataset_dir is None else str(Path(dataset_dir).expanduser().resolve()),
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "zero_action_source_root_count": zero_action_source_root_count,
        "inferred_action_count": inferred_action_count,
        "recorded_action_count": recorded_action_count,
        "algorithm_rerun": False,
        "lineage_replay": lineage_replay,
    }
    write_json(output_trace / "trace_summary.json", trace_summary)
    write_json(
        output_trace / "_TRACE_COMPLETE.json",
        {
            "trace_complete": True,
            "selected_trace_manifest_sha256": sha256_file(
                output_trace / selected_manifest.name
            ),
            "candidate_lineage_sha256": sha256_file(lineage_path),
            "recovered_existing_trace": True,
        },
    )
    write_json(output_root / "trace_parity.json", parity)

    manifest = {
        **config,
        "counterfactuals_path": str(counterfactuals_path),
        "counterfactuals_sha256": sha256_file(counterfactuals_path),
        "counterfactuals_bytes": counterfactuals_path.stat().st_size,
        "artifact_materialization_mode": counterfactuals_mode,
        "counterfactual_candidate_count": len(candidates),
        "visited_graph_count": len(graph_map),
        "traversed_step_count": len(source_payload.get("traversed_hashes") or []),
        "trace_enabled": True,
        "trace_summary": trace_summary,
        "trace_parity": parity,
        "lineage_replay": lineage_replay,
        "candidate_order_source": "official_frequency_reinforced_order",
        "algorithm_rerun": False,
        "source_algorithm_rerun": True,
        "recovered_existing_trace": True,
        "source_failed_generation_dir": str(source_root),
        "candidate_order_unchanged": True,
        "run_complete": True,
        "recovered_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "run_manifest.json", manifest)
    recovery = {
        "recovery_schema_version": 1,
        "recovery_validation_passed": True,
        "source_failed_generation_dir": str(source_root),
        "source_failure": failure,
        "source_counterfactuals_sha256": sha256_file(source_counterfactuals),
        "reference_counterfactuals_sha256": expected_reference_sha256,
        "candidate_count": len(candidates),
        "candidate_order_unchanged": True,
        "algorithm_rerun": False,
        "counterfactuals_materialization_mode": counterfactuals_mode,
        "resolved_config_materialization_mode": resolved_config_mode,
        "trace_materialization_modes": trace_materialization_modes,
        "lineage_recovery_policy": "pinned_upstream_official_hash_source_root_v2",
        "source_dataset_dir": None if dataset_dir is None else str(Path(dataset_dir).expanduser().resolve()),
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "zero_action_source_root_count": zero_action_source_root_count,
        "inferred_action_count": inferred_action_count,
        "recorded_action_count": recorded_action_count,
        "trace_parity": parity,
        "lineage_replay": lineage_replay,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    write_json(output_root / "recovery_manifest.json", recovery)
    write_json(
        output_root / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "recovered_existing_trace": True,
            "counterfactuals_sha256": manifest["counterfactuals_sha256"],
            "recovery_manifest_sha256": sha256_file(
                output_root / "recovery_manifest.json"
            ),
        },
    )
    return recovery
