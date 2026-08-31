"""Read-only recovery for the completed BACE GlobalGCE affine-edge failure.

The reviewed 100-epoch run completed native training and exact top-k mining,
but an older materializer interpreted the pinned official decoder's affine
bond-class scores as probabilities.  This module never resumes training.  It
reopens the failed train-only evidence, performs the official categorical
``argmax`` decode, hard-validates every native rule, and publishes at most
twenty semantically unique rules into a fresh candidate root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.bace_globalgce_k20_extension import (
    EXPECTED_EPOCHS,
    EXPECTED_MIN_FREQ,
    EXPECTED_SOURCE_PARENT_COUNT,
    ROUND_PLAN,
    _capture_science_contract,
    _validate_raw_manifest_binding,
    merge_unique_rules,
    validate_catalog_row,
)
from src.baselines.bace_gnn_baseline_contracts import (
    assert_gine_clean_manifest,
    oracle_provenance,
)
from src.baselines.bace_gnn_baseline_generic_adapter import (
    build_bace_baseline_generic_controller_fragment,
)
from src.baselines.globalgce_bace_adapter import (
    validate_bace_globalgce_terminal_artifacts,
)
from src.baselines.globalgce_mutagenicity_adapter import (
    OFFICIAL_AFFINE_EDGE_HARD_DECODE,
    materialize_frozen_gine_native_rule_rows,
)
from src.baselines.globalgce_resumable import (
    normalize_globalgce_training_resume_identity,
    validate_exact_top_k_proof_identity,
)
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    fresh_output_dir,
    sha256_file,
    stable_sha256,
    utc_now,
)


SCHEMA_VERSION = "bace_globalgce_affine_edge_terminal_recovery_v1"
PASS_MARKER = "[BACE_GLOBALGCE_AFFINE_EDGE_TERMINAL_RECOVERY_PASS]"
BASE_POOL_PASS_MARKER = "[BACE_GLOBALGCE_FROZEN_GINE_RULE_POOL_PASS]"
K_MAX = 20
MIN_RULES = 10
EXPECTED_SEED = 7
EXPECTED_RAW_BUDGET = 80
EXPECTED_OLD_REJECTION = (
    "GlobalGCENativeRuleError:rhs_edge_attr contains values outside [0,1]"
)
_REQUIRED_RULE_TENSORS = (
    "feat",
    "adj",
    "edge_attr",
    "features_reconst",
    "adj_reconst",
    "edge_attrs_reconst",
)


class BACEGlobalGCETerminalRecoveryError(RuntimeError):
    """Raised when the failed terminal cannot be adopted without retraining."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BACEGlobalGCETerminalRecoveryError(
            f"{label} is unavailable or malformed: {path}"
        ) from exc
    if type(payload) is not dict:
        raise BACEGlobalGCETerminalRecoveryError(f"{label} is not one JSON object")
    return payload


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if type(payload) is not dict:
                    raise BACEGlobalGCETerminalRecoveryError(
                        f"{label}:{line_number} is not one JSON object"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise BACEGlobalGCETerminalRecoveryError(
            f"{label} is unavailable or malformed: {path}"
        ) from exc
    return rows


def _require_physical_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BACEGlobalGCETerminalRecoveryError(
            f"{label} must be one physical regular file: {path}"
        )
    return path


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path): file_identity(_require_physical_file(path, label=path.name))
        for path in paths
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_no_open_write_descriptors(source: Path, proc_root: Path) -> None:
    """Reject a supposedly terminal source still held writable by any process."""

    if not proc_root.is_dir():
        raise BACEGlobalGCETerminalRecoveryError(
            f"proc root is unavailable for writer audit: {proc_root}"
        )
    writers: list[str] = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        fdinfo_root = process / "fdinfo"
        try:
            descriptors = list(fd_root.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target_text = os.readlink(descriptor)
                if target_text.endswith(" (deleted)"):
                    continue
                target = Path(target_text)
                if not target.is_absolute() or not _is_within(
                    target.resolve(strict=False), source
                ):
                    continue
                flags_line = next(
                    line
                    for line in (fdinfo_root / descriptor.name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.startswith("flags:")
                )
                flags = int(flags_line.split(":", 1)[1].strip(), 8)
            except (OSError, StopIteration, ValueError):
                continue
            if (flags & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR):
                writers.append(f"pid={process.name}:fd={descriptor.name}:{target}")
    if writers:
        raise BACEGlobalGCETerminalRecoveryError(
            "failed GlobalGCE source still has writable descriptors: "
            + ", ".join(writers[:10])
        )


def _codec_vocabulary(summary: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    metadata = summary.get("codec_metadata")
    if type(metadata) is not dict:
        raise BACEGlobalGCETerminalRecoveryError(
            "source codec summary lacks typed label metadata"
        )
    node_mapping = metadata.get("node_label_mapping")
    edge_mapping = metadata.get("edge_label_mapping")
    if type(node_mapping) is not dict or type(edge_mapping) is not dict:
        raise BACEGlobalGCETerminalRecoveryError(
            "source codec label mappings are malformed"
        )
    try:
        node_indices = sorted(int(value) for value in node_mapping)
        edge_indices = sorted(int(value) for value in edge_mapping)
    except (TypeError, ValueError) as exc:
        raise BACEGlobalGCETerminalRecoveryError(
            "source codec label indices are not integers"
        ) from exc
    if (
        node_indices != list(range(len(node_indices)))
        or edge_indices != list(range(len(edge_indices)))
        or node_mapping.get("0") != "padding"
        or str(edge_mapping.get("0") or "").lower() not in {"no_edge", "padding"}
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "source codec label mappings are not contiguous typed vocabularies"
        )
    atom_symbols = tuple(str(node_mapping[str(index)]) for index in node_indices[1:])
    bond_names = tuple(str(edge_mapping[str(index)]) for index in edge_indices)
    if (
        not atom_symbols
        or len(set(atom_symbols)) != len(atom_symbols)
        or len(bond_names) < 2
        or len(set(value.lower() for value in bond_names)) != len(bond_names)
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "source codec atom/bond vocabulary is empty or duplicated"
        )
    return atom_symbols, bond_names


def _load_rules(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency.
        raise BACEGlobalGCETerminalRecoveryError(
            "BACE GlobalGCE terminal recovery requires PyTorch"
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BACEGlobalGCETerminalRecoveryError(
            "saved GlobalGCE rules tensor bundle cannot be reopened read-only"
        ) from exc
    if not isinstance(payload, Mapping):
        raise BACEGlobalGCETerminalRecoveryError(
            "saved GlobalGCE rules tensor bundle is not a mapping"
        )
    return payload


def _validate_completed_training(
    *,
    source: Path,
    source_manifest: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    native = source / "native"
    core = _read_json(native / "training_core_summary.json", label="training core")
    heartbeat = _read_json(
        native / "globalgce_training_checkpoints" / "training_heartbeat.json",
        label="terminal training heartbeat",
    )
    try:
        resume_identity, resume_sha256 = normalize_globalgce_training_resume_identity(
            core.get("training_resume_identity")
        )
        proof = validate_exact_top_k_proof_identity(
            training.get("gspan_exact_top_k_proof") or {}
        )
        core_proof = validate_exact_top_k_proof_identity(
            core.get("gspan_exact_top_k_proof") or {}
        )
    except (TypeError, ValueError) as exc:
        raise BACEGlobalGCETerminalRecoveryError(
            "completed training lacks a valid resume/exact-top-k identity"
        ) from exc
    training_config = resume_identity.get("training_config")
    oracle_identity = resume_identity.get("oracle_identity")
    source_cohort = resume_identity.get("source_train_cohort")
    if (
        str(core.get("dataset_name") or "").lower() != "bace"
        or core.get("seed") != EXPECTED_SEED
        or core.get("epochs") != EXPECTED_EPOCHS
        or core.get("top_k_native") != EXPECTED_RAW_BUDGET
        or core.get("selected_parent_count") != EXPECTED_SOURCE_PARENT_COUNT
        or core.get("prediction_backend") != "frozen_gine_differentiable_bridge"
        or core.get("rf_oracle_used") is not False
        or core.get("trained_once") is not True
        or core.get("rule_selection_performed_once") is not True
        or core.get("gspan_exact_top_k_pruning") is not True
        or core.get("training_resume_identity_sha256") != resume_sha256
        or type(training_config) is not dict
        or training_config.get("seed") != EXPECTED_SEED
        or training_config.get("epochs") != EXPECTED_EPOCHS
        or training_config.get("top_k_native") != EXPECTED_RAW_BUDGET
        or training_config.get("min_freq") != EXPECTED_MIN_FREQ
        or training_config.get("gspan_exact_top_k_pruning") is not True
        or type(oracle_identity) is not dict
        or oracle_identity.get("checkpoint_id") != checkpoint_hash
        or type(source_cohort) is not dict
        or source_cohort.get("count") != EXPECTED_SOURCE_PARENT_COUNT
        or heartbeat.get("stage") != "complete"
        or heartbeat.get("next_epoch") != EXPECTED_EPOCHS + 1
        or heartbeat.get("resume_identity") != resume_identity
        or heartbeat.get("resume_identity_sha256") != resume_sha256
        or proof != core_proof
        or core.get("gnn_checkpoint_sha256") != checkpoint_hash
        or core.get("globalgce_model_checkpoint_sha256")
        != sha256_file(native / "globalgce_model.pt")
        or core.get("rules_checkpoint_sha256")
        != sha256_file(native / "globalgce_rules.pt")
        or source_manifest.get("config", {}).get("epochs") != EXPECTED_EPOCHS
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "saved GlobalGCE artifacts do not prove the exact completed 100-epoch run"
        )
    return resume_identity, proof


def validate_recovered_candidate_root(
    output_dir: str | Path,
    *,
    checkpoint_hash: str,
    require_pass: bool,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve(strict=True)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise BACEGlobalGCETerminalRecoveryError(
            "recovered candidate root contains a symbolic link"
        )
    manifest = _read_json(root / "run_manifest.json", label="recovery run manifest")
    summary = _read_json(root / "summary.json", label="recovery summary")
    training = _read_json(root / "training_summary.json", label="recovery training")
    complete = _read_json(root / "_RUN_COMPLETE.json", label="recovery completion")
    receipt = _read_json(root / "recovery_receipt.json", label="recovery receipt")
    candidates = _read_jsonl(root / "candidate_universe.jsonl", label="candidate universe")
    catalog = _read_jsonl(
        root / "native" / "native_rule_catalog.jsonl",
        label="recovered native catalog",
    )
    ids = [str(row.get("candidate_id") or "") for row in candidates]
    for row in candidates:
        validate_catalog_row(row, expected_checkpoint_hash=checkpoint_hash)
    if (
        not MIN_RULES <= len(candidates) <= K_MAX
        or candidates != catalog
        or any(not value for value in ids)
        or len(ids) != len(set(ids))
        or manifest.get("schema_version")
        != "bace_globalgce_frozen_gine_rule_pool_v1"
        or manifest.get("recovery_schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "PASS"
        or manifest.get("run_complete") is not True
        or manifest.get("stage") != "TRAIN_CANDIDATE_GENERATION"
        or manifest.get("candidate_count") != len(candidates)
        or manifest.get("effective_rule_count") != len(candidates)
        or manifest.get("K_MAX") != K_MAX
        or manifest.get("MIN_RULES_FOR_MAIN_TABLE") != MIN_RULES
        or manifest.get("candidate_universe_hash")
        != sha256_file(root / "candidate_universe.jsonl")
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("retrained") is not False
        or manifest.get("edge_score_hard_decode")
        != OFFICIAL_AFFINE_EDGE_HARD_DECODE
        or summary.get("valid_native_rule_count") != len(candidates)
        or training.get("valid_native_rule_count") != len(candidates)
        or complete.get("valid_native_rule_count") != len(candidates)
        or receipt.get("status") != "PASS"
        or receipt.get("selected_effective_rule_count") != len(candidates)
        or receipt.get("retrained") is not False
        or receipt.get("calibration_loaded") is not False
        or receipt.get("test_loaded") is not False
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "recovered GlobalGCE candidate terminal/hash closure failed"
        )
    assert_gine_clean_manifest(
        manifest,
        checkpoint_id=checkpoint_hash,
        require_train_only=True,
    )
    validate_bace_globalgce_terminal_artifacts(root, require_exact_top_k=True)
    marker = root / "PASS"
    if require_pass:
        if marker.read_text(encoding="utf-8") != BASE_POOL_PASS_MARKER + "\n":
            raise BACEGlobalGCETerminalRecoveryError(
                "recovered GlobalGCE PASS marker changed"
            )
    elif marker.exists():
        raise BACEGlobalGCETerminalRecoveryError(
            "recovered GlobalGCE PASS appeared before independent validation"
        )
    return manifest


def recover_failed_bace_globalgce_terminal(
    *,
    failed_controller_root: str | Path,
    source_round_root: str | Path,
    source_manifest: str | Path,
    native_train_csv: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Recover one failed affine-edge terminal into a fresh train-only root."""

    failed_root = Path(failed_controller_root).expanduser().resolve(strict=True)
    source = Path(source_round_root).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise BACEGlobalGCETerminalRecoveryError(
            "source round root must be one physical directory"
        )
    if source.parent.name != "rounds" or source.parent.parent != failed_root:
        raise BACEGlobalGCETerminalRecoveryError(
            "source round is not an exact child of the failed controller root"
        )
    failed = _read_json(failed_root / "FAILED.json", label="failed controller receipt")
    if failed.get("status") != "FAILED" or (failed_root / "PASS").exists():
        raise BACEGlobalGCETerminalRecoveryError(
            "source controller is not a failed, non-PASS terminal"
        )
    if any(
        (source / name).exists()
        for name in ("PASS", "_RUN_COMPLETE.json", "RAW_SHORTFALL", "K20_RAW_ROUND.json")
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "source round already has a conflicting terminal marker"
        )
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        raise BACEGlobalGCETerminalRecoveryError("output root must be absolute")
    destination = destination.resolve(strict=False)
    if _is_within(destination, failed_root) or _is_within(destination, source):
        raise BACEGlobalGCETerminalRecoveryError(
            "fresh recovery output cannot modify the failed source tree"
        )
    _assert_no_open_write_descriptors(source, Path(proc_root).expanduser())

    source_manifest_path = Path(source_manifest).expanduser().resolve(strict=True)
    native_path = Path(native_train_csv).expanduser().resolve(strict=True)
    official = Path(official_root).expanduser().resolve(strict=True)
    checkpoint_input = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    science, checkpoint = _capture_science_contract(
        source_manifest=source_manifest_path,
        native_train_csv=native_path,
        official_root=official,
        gnn_checkpoint=checkpoint_input,
    )
    checkpoint_hash = str((science.get("oracle") or {}).get("oracle_checkpoint_hash") or "")
    if len(checkpoint_hash) != 64:
        raise BACEGlobalGCETerminalRecoveryError(
            "frozen BACE GINE checkpoint identity is invalid"
        )
    source_run = _validate_raw_manifest_binding(
        source,
        science_contract=science,
        seed=EXPECTED_SEED,
        raw_budget=EXPECTED_RAW_BUDGET,
        complete=False,
    )

    native = source / "native"
    critical_paths = (
        failed_root / "FAILED.json",
        source / "run_manifest.json",
        source / "training_summary.json",
        native / "training_core_summary.json",
        native / "source_codec_summary.json",
        native / "globalgce_model.pt",
        native / "globalgce_rules.pt",
        native / "native_rule_catalog.jsonl",
        native / "native_rule_rejections.jsonl",
        native / "globalgce_training_checkpoints" / "training_heartbeat.json",
    )
    source_before = _snapshot(critical_paths)
    training = _read_json(source / "training_summary.json", label="source training summary")
    resume_identity, proof = _validate_completed_training(
        source=source,
        source_manifest=source_run,
        training=training,
        checkpoint_hash=checkpoint_hash,
    )
    codec = _read_json(native / "source_codec_summary.json", label="source codec summary")
    atom_symbols, bond_names = _codec_vocabulary(codec)
    old_catalog = _read_jsonl(
        native / "native_rule_catalog.jsonl", label="old native catalog"
    )
    old_rejections = _read_jsonl(
        native / "native_rule_rejections.jsonl", label="old native rejections"
    )
    if (
        old_catalog
        or len(old_rejections) != EXPECTED_RAW_BUDGET
        or {str(row.get("reason") or "") for row in old_rejections}
        != {EXPECTED_OLD_REJECTION}
        or training.get("native_rule_count") != EXPECTED_RAW_BUDGET
        or training.get("valid_native_rule_count") != 0
        or training.get("rejected_native_rule_count") != EXPECTED_RAW_BUDGET
        or training.get("codec_metadata") != codec.get("codec_metadata")
        or codec.get("source_codec_passed") is not True
        or codec.get("source_codec_checked_rows") != EXPECTED_SOURCE_PARENT_COUNT
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "failed source is not the isolated 80/80 affine-edge rejection"
        )

    rules = _load_rules(native / "globalgce_rules.pt")
    if any(rules.get(name) is None for name in _REQUIRED_RULE_TENSORS):
        raise BACEGlobalGCETerminalRecoveryError(
            "saved rules bundle lacks required typed tensors"
        )
    if any(
        int(rules[name].shape[0]) != EXPECTED_RAW_BUDGET
        for name in _REQUIRED_RULE_TENSORS
    ):
        raise BACEGlobalGCETerminalRecoveryError(
            "saved rules bundle does not contain the frozen 80-rule budget"
        )
    valid_rows, new_rejections = materialize_frozen_gine_native_rule_rows(
        rules=rules,
        atom_symbols=atom_symbols,
        bond_names=bond_names,
        oracle_checkpoint_hash=checkpoint_hash,
    )
    unique, dedup_audit = merge_unique_rules(
        [(ROUND_PLAN[0], valid_rows)],
        expected_checkpoint_hash=checkpoint_hash,
    )
    if len(unique) < MIN_RULES:
        raise BACEGlobalGCETerminalRecoveryError(
            f"typed edge decode yielded only {len(unique)} semantic-unique valid rules; "
            f"minimum is {MIN_RULES}"
        )
    selected: list[dict[str, Any]] = []
    for rank, row in enumerate(unique[:K_MAX], start=1):
        item = dict(row)
        item["rank"] = rank
        item["recovery_rank"] = rank
        selected.append(item)

    source_after_decode = _snapshot(critical_paths)
    if source_after_decode != source_before:
        raise BACEGlobalGCETerminalRecoveryError(
            "failed source bytes changed during read-only materialization"
        )
    output = fresh_output_dir(destination)
    (output / "native").mkdir()
    atomic_jsonl(output / "candidate_universe.jsonl", selected)
    atomic_jsonl(output / "native" / "native_rule_catalog.jsonl", selected)
    atomic_jsonl(output / "native" / "decoded_valid_rules_all.jsonl", valid_rows)
    atomic_jsonl(output / "native" / "typed_decode_rejections.jsonl", new_rejections)
    atomic_jsonl(output / "native" / "semantic_dedup_audit.jsonl", dedup_audit)
    atomic_jsonl(
        output / "candidate_filter_audit.jsonl",
        [
            {
                "candidate_id": row["candidate_id"],
                "native_rule_index": row["rule"]["native_rule_index"],
                "accepted": True,
                "reason": "valid_unique_typed_native_lhs_rhs_rule",
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
            }
            for row in selected
        ],
    )
    # ``science['oracle']`` is already the exact oracle_provenance payload
    # captured and independently reopened above.  Recompute once more and
    # require byte-for-byte semantic equality before publication.
    checkpoint_card = dict(science["oracle"])
    recomputed_provenance = oracle_provenance(
        {
            "checkpoint_id": checkpoint_hash,
        },
        checkpoint,
    )
    if recomputed_provenance != checkpoint_card:
        raise BACEGlobalGCETerminalRecoveryError(
            "frozen GINE provenance changed during recovery"
        )
    provenance = recomputed_provenance
    atomic_json(output / "oracle_provenance.json", provenance)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "failed_controller_root": str(failed_root),
        "source_round_root": str(source),
        "source_failed_receipt": source_before[str(failed_root / "FAILED.json")],
        "source_run_manifest": source_before[str(source / "run_manifest.json")],
        "source_training_summary": source_before[str(source / "training_summary.json")],
        "source_rules_checkpoint": source_before[str(native / "globalgce_rules.pt")],
        "source_model_checkpoint": source_before[str(native / "globalgce_model.pt")],
        "source_training_heartbeat": source_before[
            str(native / "globalgce_training_checkpoints" / "training_heartbeat.json")
        ],
        "source_evidence_unchanged": True,
        "source_native_rule_count": EXPECTED_RAW_BUDGET,
        "old_valid_native_rule_count": 0,
        "old_rejected_native_rule_count": EXPECTED_RAW_BUDGET,
        "old_rejection_reason": EXPECTED_OLD_REJECTION,
        "typed_decode_contract": "official_affine_bond_class_scores_argmax_to_one_hot",
        "edge_score_hard_decode": OFFICIAL_AFFINE_EDGE_HARD_DECODE,
        "decoded_valid_native_rule_count": len(valid_rows),
        "decoded_rejected_native_rule_count": len(new_rejections),
        "semantic_unique_valid_native_rule_count": len(unique),
        "selected_effective_rule_count": len(selected),
        "K_MAX": K_MAX,
        "MIN_RULES_FOR_MAIN_TABLE": MIN_RULES,
        "k_greater_than_effective_policy": "plateau_without_rule_copy",
        "retrained": False,
        "training_artifacts_reused_read_only": True,
        "classifier_parameters_frozen": True,
        "oracle_checkpoint_hash": checkpoint_hash,
        "resume_identity_sha256": stable_sha256(resume_identity),
        "gspan_exact_top_k_proof": proof,
        "calibration_loaded": False,
        "test_loaded": False,
        "candidate_universe": file_identity(output / "candidate_universe.jsonl"),
        "decoded_valid_rules_all": file_identity(
            output / "native" / "decoded_valid_rules_all.jsonl"
        ),
        "typed_decode_rejections": file_identity(
            output / "native" / "typed_decode_rejections.jsonl"
        ),
        "semantic_dedup_audit": file_identity(
            output / "native" / "semantic_dedup_audit.jsonl"
        ),
        "completed_at": utc_now(),
    }
    receipt["receipt_payload_sha256"] = stable_sha256(receipt)
    atomic_json(output / "recovery_receipt.json", receipt)

    recovered_training = dict(training)
    recovered_training.update(
        {
            "native_rule_count": EXPECTED_RAW_BUDGET,
            "decoded_valid_native_rule_count": len(valid_rows),
            "decoded_rejected_native_rule_count": len(new_rejections),
            "semantic_unique_valid_native_rule_count": len(unique),
            "valid_native_rule_count": len(selected),
            "rejected_native_rule_count": len(new_rejections),
            "native_rule_edge_score_contract": (
                "pinned_official_unbounded_affine_class_scores"
            ),
            "native_rule_edge_score_hard_decode": OFFICIAL_AFFINE_EDGE_HARD_DECODE,
            "native_rule_catalog": str(
                (output / "native" / "native_rule_catalog.jsonl").resolve()
            ),
            "native_rule_catalog_sha256": sha256_file(
                output / "native" / "native_rule_catalog.jsonl"
            ),
            "classifier_checkpoint_hash_before": checkpoint_hash,
            "classifier_checkpoint_hash_after": checkpoint_hash,
            "classifier_checkpoint_unchanged": True,
            "trainable_checkpoint_classifier_keys": [],
            "terminal_recovery": file_identity(output / "recovery_receipt.json"),
            "retrained": False,
        }
    )
    atomic_json(output / "training_summary.json", recovered_training)
    training_identity = file_identity(output / "training_summary.json")
    summary = {
        "status": "PASS",
        "run_complete": True,
        "source_parent_count": EXPECTED_SOURCE_PARENT_COUNT,
        "native_rule_count": EXPECTED_RAW_BUDGET,
        "decoded_valid_native_rule_count": len(valid_rows),
        "semantic_unique_valid_native_rule_count": len(unique),
        "valid_native_rule_count": len(selected),
        "effective_rule_count": len(selected),
        "K_MAX": K_MAX,
        "MIN_RULES_FOR_MAIN_TABLE": MIN_RULES,
        "candidate_universe_hash": sha256_file(output / "candidate_universe.jsonl"),
        "oracle_checkpoint_hash": checkpoint_hash,
        "calibration_loaded": False,
        "test_loaded": False,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "retrained": False,
        "edge_score_hard_decode": OFFICIAL_AFFINE_EDGE_HARD_DECODE,
        "training_summary": training_identity,
        "gspan_exact_top_k_proof": proof,
        "recovery_receipt": file_identity(output / "recovery_receipt.json"),
    }
    atomic_json(output / "summary.json", summary)
    summary_identity = file_identity(output / "summary.json")
    manifest = dict(source_run)
    manifest.update(
        {
            "schema_version": "bace_globalgce_frozen_gine_rule_pool_v1",
            "recovery_schema_version": SCHEMA_VERSION,
            "dataset": "bace",
            "method": "GlobalGCE",
            "method_id": "globalgce",
            "stage": "TRAIN_CANDIDATE_GENERATION",
            "status": "PASS",
            "run_complete": True,
            "action_kind": "lhs_rhs_graph_transformation_rule",
            "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
            **provenance,
            "candidate_universe_hash": summary["candidate_universe_hash"],
            "candidate_count": len(selected),
            "effective_rule_count": len(selected),
            "K_MAX": K_MAX,
            "MIN_RULES_FOR_MAIN_TABLE": MIN_RULES,
            "k_greater_than_effective_policy": "plateau_without_rule_copy",
            "classifier_parameters_frozen": True,
            "classifier_checkpoint_unchanged": True,
            "selector_fitted_on_calibration": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "retrained": False,
            "edge_score_contract": "pinned_official_unbounded_affine_class_scores",
            "edge_score_hard_decode": OFFICIAL_AFFINE_EDGE_HARD_DECODE,
            "training_summary": training_identity,
            "summary": summary_identity,
            "recovery_receipt": file_identity(output / "recovery_receipt.json"),
            "gspan_exact_top_k_proof": proof,
            "completed_at": utc_now(),
        }
    )
    assert_gine_clean_manifest(
        manifest,
        checkpoint_id=checkpoint_hash,
        require_train_only=True,
    )
    atomic_json(output / "run_manifest.json", manifest)
    complete = {
        **summary,
        "summary": summary_identity,
        "run_manifest": file_identity(output / "run_manifest.json"),
    }
    atomic_json(output / "_RUN_COMPLETE.json", complete)
    source_final = _snapshot(critical_paths)
    if source_final != source_before:
        raise BACEGlobalGCETerminalRecoveryError(
            "failed source bytes changed before recovery publication"
        )
    validate_recovered_candidate_root(
        output,
        checkpoint_hash=checkpoint_hash,
        require_pass=False,
    )
    atomic_marker(output / "PASS", BASE_POOL_PASS_MARKER)
    return validate_recovered_candidate_root(
        output,
        checkpoint_hash=checkpoint_hash,
        require_pass=True,
    )


def build_recovery_controller_fragment(
    *,
    python: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    failed_controller_root: str | Path,
    source_round_root: str | Path,
    source_manifest: str | Path,
    native_train_csv: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    dataset_dir: str | Path,
    calibration_split: str | Path,
    test_split: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
) -> dict[str, Any]:
    """Build one existing-controller queue starting at read-only recovery."""

    project = Path(project_root).expanduser()
    executable = Path(python).expanduser()
    root = Path(output_root).expanduser()
    for value, label in (
        (project, "project_root"),
        (executable, "python"),
        (root, "output_root"),
    ):
        if not value.is_absolute():
            raise BACEGlobalGCETerminalRecoveryError(f"{label} must be absolute")
    project = project.resolve(strict=False)
    executable = executable.resolve(strict=False)
    root = root.resolve(strict=False)
    fragment = build_bace_baseline_generic_controller_fragment(
        method="GlobalGCE",
        python=executable,
        project_root=project,
        output_root=root,
        gnn_checkpoint=gnn_checkpoint,
        dataset_dir=dataset_dir,
        calibration_split=calibration_split,
        test_split=test_split,
        molclr_root=molclr_root,
        molclr_checkpoint=molclr_checkpoint,
        neurosed_checkpoint=neurosed_checkpoint,
        official_root=official_root,
        globalgce_source_manifest=source_manifest,
        globalgce_native_train_csv=native_train_csv,
    )
    candidate_id = "bace_globalgce_train_candidates"
    final_id = "bace_globalgce_final_freeze"
    removed = {
        "bace_globalgce_preflight",
        "bace_globalgce_bridge_smoke",
    }
    retained = [dict(task) for task in fragment["tasks"] if task["id"] not in removed]
    candidate_index = next(
        (
            index
            for index, task in enumerate(retained)
            if task.get("id") == candidate_id
        ),
        None,
    )
    if candidate_index is None:
        raise BACEGlobalGCETerminalRecoveryError(
            "generic GlobalGCE fragment lacks the candidate task"
        )
    candidate_output = str(root / "train_candidates" / "attempt-{attempt}")
    retained[candidate_index] = {
        "id": candidate_id,
        "dataset": "bace",
        "stage": "BACE_GLOBALGCE_AFFINE_EDGE_TERMINAL_RECOVERY",
        "runner_dataset": "bace-baseline-globalgce",
        "runner_stage": "BACE_GLOBALGCE_AFFINE_EDGE_TERMINAL_RECOVERY",
        "depends_on": [],
        "resource": "cpu",
        "priority": 61,
        "enabled": True,
        "data_splits": ["train"],
        "manifest_only": False,
        "command": [
            str(executable),
            str(project / "scripts/autodl/recover_bace_globalgce_terminal.py"),
            "--config",
            str(project / "configs/hpc.yaml"),
            "--set",
            "inference.fallback_to_heuristic=false",
            "recover",
            "--failed-controller-root",
            str(Path(failed_controller_root).expanduser().resolve(strict=False)),
            "--source-round-root",
            str(Path(source_round_root).expanduser().resolve(strict=False)),
            "--source-manifest",
            str(Path(source_manifest).expanduser().resolve(strict=False)),
            "--native-train-csv",
            str(Path(native_train_csv).expanduser().resolve(strict=False)),
            "--official-root",
            str(Path(official_root).expanduser().resolve(strict=False)),
            "--gnn-checkpoint",
            str(Path(gnn_checkpoint).expanduser().resolve(strict=False)),
            "--output-dir",
            "{task_output}",
            "--proc-root",
            "/proc",
        ],
        "input_manifest": str(
            Path(source_round_root).expanduser().resolve(strict=False)
            / "run_manifest.json"
        ),
        "expected_output": candidate_output,
        "required_output_files": [
            "candidate_universe.jsonl",
            "candidate_filter_audit.jsonl",
            "oracle_provenance.json",
            "run_manifest.json",
            "training_summary.json",
            "summary.json",
            "recovery_receipt.json",
            "_RUN_COMPLETE.json",
            "PASS",
        ],
        "required_log_marker": PASS_MARKER,
        "environment": {
            "PYTHONPATH": "{project_root}",
            "RUN_TASTEMOLNET": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "native_action_kind": "lhs_rhs_graph_transformation_rule",
        "native_action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
        "read_only_adoption": True,
        "retraining_forbidden": True,
    }
    standard_id = "bace_globalgce_standardized"
    retained.append(
        {
            "id": standard_id,
            "dataset": "bace",
            "stage": "BACE_FROZEN_CELL_STANDARDIZATION",
            "runner_dataset": "paper-cell-bace-globalgce",
            "runner_stage": "BACE_FROZEN_CELL_STANDARDIZATION",
            "depends_on": [final_id],
            "resource": "cpu",
            "priority": 162,
            "enabled": True,
            "data_splits": [],
            "manifest_only": True,
            "command": [
                str(executable),
                str(project / "scripts/autodl/standardize_bace_frozen_cell.py"),
                "--config",
                str(project / "configs/hpc.yaml"),
                "--method",
                "GlobalGCE",
                "--source-final-root",
                "{dep_bace_globalgce_final_freeze_output}",
                "--gnn-checkpoint",
                str(Path(gnn_checkpoint).expanduser().resolve(strict=False)),
                "--output-dir",
                "{task_output}",
            ],
            "input_manifest": (
                "{dep_bace_globalgce_final_freeze_output}/FINAL_PASS.json"
            ),
            "expected_output": str(root / "standardized" / "attempt-{attempt}"),
            "required_output_files": [
                "figure3_coverage_vs_k.csv",
                "figure4_coverage_vs_threshold.csv",
                "prefix_metrics.csv",
                "prefix_metrics.json",
                "parent_best_distances.csv",
                "destination_distribution.csv",
                "table2_globalgce_k10.csv",
                "summary.json",
                "run_manifest.json",
                "oracle_manifest.json",
                "evaluation_manifest.json",
                "artifact_manifest.json",
                "freeze_manifest.json",
                "_FINALIZED.json",
                "final_artifact_audit.json",
                "PASS",
            ],
            "required_log_marker": (
                "[BACE_FROZEN_CELL_STANDARDIZATION_PASS] method=GlobalGCE"
            ),
            "environment": {
                "PYTHONPATH": "{project_root}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "RUN_TASTEMOLNET": "0",
            },
            "selector_parameters_frozen": True,
            "read_only_test": True,
        }
    )
    fragment.update(
        {
            "schema_version": "bace_globalgce_terminal_recovery_fragment_v1",
            "root_task_ids": [candidate_id],
            "terminal_task_ids": [standard_id],
            "tasks": retained,
            "source_training_reused_read_only": True,
            "retraining_forbidden": True,
            "K_MAX": K_MAX,
            "MIN_RULES_FOR_MAIN_TABLE": MIN_RULES,
        }
    )
    return fragment


__all__ = [
    "BASE_POOL_PASS_MARKER",
    "BACEGlobalGCETerminalRecoveryError",
    "K_MAX",
    "MIN_RULES",
    "PASS_MARKER",
    "SCHEMA_VERSION",
    "build_recovery_controller_fragment",
    "recover_failed_bace_globalgce_terminal",
    "validate_recovered_candidate_root",
]
