"""BACE facade over the established project GlobalGCE molecular adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .globalgce_mutagenicity_adapter import (
    NativeGeneratorProtocol,
    OfficialGlobalGCEMutagenicityGenerator,
    PoolBuildConfig,
    TrainParent,
    TeacherProtocol,
    audit_mutagenicity_train_pool,
    build_mutagenicity_train_pool,
)
from src.baselines.bace_gnn_baseline_contracts import (
    assert_gine_clean_manifest,
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    validate_official_globalgce_root,
)
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    sha256_file,
    stable_sha256,
    utc_now,
)


DATASET_NAME = "BACE"
EXPECTED_TRAIN_SOURCE_COUNT = 360
EXPECTED_NATIVE_TRAIN_COUNT = 869


def _read_source_manifest(
    path_like: str | Path,
    *,
    expected_parent_count: int,
) -> list[TrainParent]:
    path = Path(path_like).expanduser().resolve(strict=True)
    parents: list[TrainParent] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"BACE source manifest row {line_number} is not an object")
            parent_id = str(row.get("molecule_id") or "").strip()
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not parent_id or parent_id in seen or not smiles:
                raise ValueError(
                    f"BACE source manifest has missing/duplicate identity at row {line_number}"
                )
            if (
                str(row.get("split") or "").strip().lower() != "train"
                or int(row.get("source_label", -1)) != 1
                or int(row.get("gnn_label", -1)) != 1
                or int(row.get("target_label", -1)) != 0
            ):
                raise ValueError("BACE GlobalGCE source manifest is not frozen-GINE train-only")
            seen.add(parent_id)
            parents.append(TrainParent(parent_id, smiles, 1, "train"))
    parents.sort(key=lambda row: row.parent_id)
    if len(parents) != int(expected_parent_count):
        raise ValueError(
            "BACE GlobalGCE source-parent count mismatch: "
            f"actual={len(parents)}, expected={expected_parent_count}"
        )
    return parents


class OfficialGlobalGCEBACEGenerator(OfficialGlobalGCEMutagenicityGenerator):
    """Run unchanged official GlobalGCE components with BACE dataset identity."""

    def __init__(
        self,
        official_root: str | Path,
        *,
        native_train_csv: str | Path,
        min_freq: int,
        frozen_gine_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(
            official_root,
            native_train_csv=native_train_csv,
            dataset_name=DATASET_NAME,
            min_freq=int(min_freq),
            frozen_gine_checkpoint=frozen_gine_checkpoint,
            source_label=1,
            target_label=0,
        )


def build_bace_train_pool(
    *,
    train_csv: str | Path,
    teacher_path: str | Path,
    official_root: str | Path,
    output_dir: str | Path,
    teacher: TeacherProtocol,
    generator: NativeGeneratorProtocol,
    config: PoolBuildConfig | None = None,
) -> dict[str, Any]:
    return build_mutagenicity_train_pool(
        train_csv=train_csv,
        teacher_path=teacher_path,
        official_root=official_root,
        output_dir=output_dir,
        teacher=teacher,
        generator=generator,
        config=config,
        dataset_name=DATASET_NAME,
    )


def audit_bace_train_pool(
    run_dir: str | Path,
    *,
    train_csv: str | Path,
    expected_parent_count: int = EXPECTED_TRAIN_SOURCE_COUNT,
    expected_input_train_count: int | None = EXPECTED_TRAIN_SOURCE_COUNT,
    require_complete: bool = True,
) -> dict[str, Any]:
    return audit_mutagenicity_train_pool(
        run_dir,
        train_csv=train_csv,
        expected_parent_count=expected_parent_count,
        expected_input_train_count=expected_input_train_count,
        require_target_label_zero=True,
        require_unique_universe=True,
        forbid_calibration_test=True,
        require_complete=require_complete,
        dataset_name=DATASET_NAME,
    )


def build_bace_frozen_gine_rule_pool(
    *,
    source_manifest: str | Path,
    native_train_csv: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    min_freq: int,
    config: PoolBuildConfig | None = None,
) -> dict[str, Any]:
    """Train and freeze native GlobalGCE rules against the one BACE GINE.

    This route deliberately does not call the historical RF-backed BACE pool
    builder.  The official LHS/RHS decoder sees a straight-through view of the
    exact frozen GINE, while final rules remain hard native transformations
    that are re-applied and re-scored during calibration/test verification.
    """

    resolved = config or PoolBuildConfig(expected_parent_count=EXPECTED_TRAIN_SOURCE_COUNT)
    if int(resolved.expected_parent_count) != EXPECTED_TRAIN_SOURCE_COUNT:
        raise ValueError("BACE GlobalGCE requires the frozen 360-parent train cohort")
    source_path = Path(source_manifest).expanduser().resolve(strict=True)
    native_path = Path(native_train_csv).expanduser().resolve(strict=True)
    if any(token in source_path.name.lower() for token in ("test", "calibration")):
        raise ValueError("BACE GlobalGCE source manifest must be train-only")
    if any(token in native_path.name.lower() for token in ("test", "calibration")):
        raise ValueError("BACE GlobalGCE native vocabulary CSV must be train-only")
    with native_path.open(newline="", encoding="utf-8") as handle:
        native_train_count = sum(1 for _row in csv.DictReader(handle))
    if native_train_count != EXPECTED_NATIVE_TRAIN_COUNT:
        raise ValueError(
            "BACE GlobalGCE native train row count mismatch: "
            f"actual={native_train_count}, expected={EXPECTED_NATIVE_TRAIN_COUNT}"
        )
    parents = _read_source_manifest(
        source_path, expected_parent_count=EXPECTED_TRAIN_SOURCE_COUNT
    )
    source_dataset_manifest_path = source_path.parent / "run_manifest.json"
    if not source_dataset_manifest_path.is_file():
        raise FileNotFoundError(
            "BACE source cohort must be bound by its dataset run_manifest.json"
        )
    source_dataset_manifest = json.loads(
        source_dataset_manifest_path.read_text(encoding="utf-8")
    )
    declared_source = dict(
        (source_dataset_manifest.get("artifacts") or {}).get(
            source_path.name
        )
        or {}
    )
    if (
        source_dataset_manifest.get("run_complete") is not True
        or source_dataset_manifest.get("test_loaded") is not False
        or str(source_dataset_manifest.get("dataset") or "").lower() != "bace"
        or int(source_dataset_manifest.get("generation_source_rows") or -1)
        != EXPECTED_TRAIN_SOURCE_COUNT
        or declared_source.get("sha256") != sha256_file(source_path)
        or int(declared_source.get("bytes") or -1) != source_path.stat().st_size
    ):
        raise ValueError("BACE source cohort manifest/hash closure failed")
    checkpoint, card, _schema = validate_bace_frozen_gine(gnn_checkpoint)
    provenance = oracle_provenance(card, checkpoint)
    official_audit = validate_official_globalgce_root(official_root)
    generator = OfficialGlobalGCEBACEGenerator(
        official_root,
        native_train_csv=native_path,
        min_freq=int(min_freq),
        frozen_gine_checkpoint=checkpoint,
    )
    root = Path(output_dir).expanduser().resolve(strict=False)
    fingerprint_payload = {
        "schema_version": "bace_globalgce_frozen_gine_rule_pool_v1",
        "source_manifest": file_identity(source_path),
        "source_dataset_manifest": file_identity(source_dataset_manifest_path),
        "native_train_csv": file_identity(native_path),
        "native_train_row_count": native_train_count,
        "official_source_audit": official_audit,
        "native_generator": generator.config_identity(),
        "oracle": provenance,
        "source_parent_ids": [row.parent_id for row in parents],
        "config": {
            "seed": int(resolved.seed),
            "epochs": int(resolved.epochs),
            "top_k_native": int(resolved.top_k_native),
            "learning_rate": float(resolved.learning_rate),
            "dropout": float(resolved.dropout),
            "device": str(resolved.device),
            "min_freq": int(min_freq),
            "gspan_flush_every": int(resolved.gspan_flush_every),
            "gspan_max_in_memory_candidates": int(
                resolved.gspan_max_in_memory_candidates
            ),
        },
    }
    fingerprint = stable_sha256(fingerprint_payload)
    manifest_path = root / "run_manifest.json"
    if (root / "PASS").exists() or (root / "_RUN_COMPLETE.json").exists():
        raise FileExistsError(f"Completed BACE GlobalGCE rule pool is immutable: {root}")
    if root.exists() and any(root.iterdir()):
        if not bool(resolved.resume) or not manifest_path.is_file():
            raise FileExistsError("BACE GlobalGCE partial output requires explicit resume")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("BACE GlobalGCE resume fingerprint mismatch")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            manifest_path,
            {
                **fingerprint_payload,
                "dataset": "bace",
                "method": "GlobalGCE",
                "method_id": "globalgce",
                "stage": "TRAIN_CANDIDATE_GENERATION",
                "status": "RUNNING",
                "run_complete": False,
                "config_fingerprint": fingerprint,
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "calibration_loaded": False,
                "test_loaded": False,
                "created_at": utc_now(),
            },
        )

    before_checkpoint_hash = sha256_file(checkpoint / "model.pt")
    training_holder: dict[str, Any] = {}

    def record_training(summary: dict[str, Any]) -> None:
        training_holder.clear()
        training_holder.update(summary)
        atomic_json(root / "training_summary.json", training_holder)

    result = generator.generate(
        parents,
        output_dir=root / "native",
        seed=int(resolved.seed),
        epochs=int(resolved.epochs),
        top_k_native=int(resolved.top_k_native),
        learning_rate=float(resolved.learning_rate),
        dropout=float(resolved.dropout),
        device=str(resolved.device),
        resume=bool(resolved.resume),
        generation_chunk_size=int(resolved.generation_chunk_size),
        generation_num_workers=int(resolved.generation_num_workers),
        memory_log_every_chunks=int(resolved.memory_log_every_chunks),
        gspan_flush_every=int(resolved.gspan_flush_every),
        gspan_max_in_memory_candidates=int(resolved.gspan_max_in_memory_candidates),
        start_parent_offset=0,
        on_training_ready=record_training,
        on_chunk=None,
        rules_only=True,
    )
    training_holder.update(result.training_summary)
    catalog_path = root / "native" / "native_rule_catalog.jsonl"
    rows: list[dict[str, Any]] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                GlobalGCENativeRule.from_payload(row)
                rows.append(row)
    if len(rows) < 20 or len({str(row["candidate_id"]) for row in rows}) != len(rows):
        raise RuntimeError("BACE GlobalGCE did not freeze twenty unique valid rules")
    after_checkpoint_hash = sha256_file(checkpoint / "model.pt")
    if after_checkpoint_hash != before_checkpoint_hash:
        raise RuntimeError("Frozen BACE GINE checkpoint bytes changed during GlobalGCE")
    model_state_path = root / "native" / "globalgce_model.pt"
    try:
        import torch

        state = torch.load(model_state_path, map_location="cpu")
        forbidden_state_keys = [
            str(key)
            for key in state
            if "gt_gnn" in str(key).lower() or "bridge" in str(key).lower()
        ]
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("BACE GlobalGCE rule freeze requires PyTorch") from exc
    if forbidden_state_keys:
        raise RuntimeError(
            "GlobalGCE trainable checkpoint accidentally contains frozen classifier state"
        )
    atomic_jsonl(root / "candidate_universe.jsonl", rows)
    atomic_jsonl(
        root / "candidate_filter_audit.jsonl",
        [
            {
                "candidate_id": row["candidate_id"],
                "native_rule_index": row["rule"]["native_rule_index"],
                "accepted": True,
                "reason": "valid_native_lhs_rhs_rule",
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
            }
            for row in rows
        ],
    )
    training_holder.update(
        {
            "classifier_parameters_frozen": True,
            "classifier_checkpoint_hash_before": before_checkpoint_hash,
            "classifier_checkpoint_hash_after": after_checkpoint_hash,
            "classifier_checkpoint_unchanged": True,
            "trainable_checkpoint_classifier_keys": forbidden_state_keys,
        }
    )
    atomic_json(root / "training_summary.json", training_holder)
    summary = {
        "status": "PASS",
        "run_complete": True,
        "source_parent_count": len(parents),
        "native_rule_count": int(training_holder.get("native_rule_count") or 0),
        "valid_native_rule_count": len(rows),
        "candidate_universe_hash": sha256_file(root / "candidate_universe.jsonl"),
        "oracle_checkpoint_hash": card["checkpoint_id"],
        "calibration_loaded": False,
        "test_loaded": False,
    }
    atomic_json(root / "summary.json", summary)
    manifest = {
        **fingerprint_payload,
        "dataset": "bace",
        "method": "GlobalGCE",
        "method_id": "globalgce",
        "stage": "TRAIN_CANDIDATE_GENERATION",
        "status": "PASS",
        "run_complete": True,
        "config_fingerprint": fingerprint,
        "action_kind": "lhs_rhs_graph_transformation_rule",
        "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
        **provenance,
        "candidate_universe_hash": summary["candidate_universe_hash"],
        "candidate_count": len(rows),
        "classifier_parameters_frozen": True,
        "classifier_checkpoint_unchanged": True,
        "selector_fitted_on_calibration": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "completed_at": utc_now(),
    }
    assert_gine_clean_manifest(
        manifest, checkpoint_id=str(card["checkpoint_id"]), require_train_only=True
    )
    atomic_json(manifest_path, manifest)
    atomic_json(root / "oracle_provenance.json", provenance)
    atomic_json(root / "_RUN_COMPLETE.json", summary)
    atomic_marker(root / "PASS", "[BACE_GLOBALGCE_FROZEN_GINE_RULE_POOL_PASS]")
    return manifest


__all__ = [
    "DATASET_NAME",
    "EXPECTED_NATIVE_TRAIN_COUNT",
    "EXPECTED_TRAIN_SOURCE_COUNT",
    "OfficialGlobalGCEBACEGenerator",
    "audit_bace_train_pool",
    "build_bace_frozen_gine_rule_pool",
    "build_bace_train_pool",
]
