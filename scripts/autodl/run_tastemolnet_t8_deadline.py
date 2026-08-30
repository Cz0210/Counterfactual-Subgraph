#!/usr/bin/env python3
"""Deadline-only TasteMolNet T8 runner using the frozen two-target science."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.globalgce_bace_native_rules import validate_official_globalgce_root  # noqa: E402
from src.baselines.globalgce_mutagenicity_adapter import OfficialGlobalGCEMutagenicityGenerator  # noqa: E402
from src.baselines.tastemolnet_globalgce_smoke import (  # noqa: E402
    DATASET, GINE_PAYLOAD_FILES, NUM_CLASSES, PASS_MARKER, SOURCE_LABEL, STAGE,
    FrozenTasteGINEScorer, TasteGlobalGCESmokeConfig, TasteGlobalGCESmokeError,
    run_t8_science,
)
from src.utils.retained_output_directory import FreshOutputDirectory, prepare_terminal_output  # noqa: E402
from src.utils.tastemolnet_t8_globalgce_release import _checkpoint_train_contract  # noqa: E402

SCHEMA = "tastemolnet_t8_deadline_runner_v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 deadline cannot read {label}") from exc
    if type(value) is not dict:
        raise TasteGlobalGCESmokeError(f"T8 deadline {label} is not one object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: Path, label: str) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGlobalGCESmokeError(f"T8 deadline {label} must be absolute")
    return path.resolve(strict=True)


def _uuid4(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise TasteGlobalGCESmokeError("T8 attempt ID is not a UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise TasteGlobalGCESmokeError("T8 attempt ID must be canonical UUIDv4")
    return value


def _terminal_pass(root: Path, label: str) -> dict[str, Any]:
    gate = _read_json(root / "gate.json", f"{label} gate")
    verification = _read_json(root / "verification.json", f"{label} verification")
    if gate.get("status") != "PASS" or verification.get("status") != "PASS" or not (root / "PASS").is_file():
        raise TasteGlobalGCESmokeError(f"T8 deadline {label} is not PASS")
    science = verification.get("verification")
    if type(science) is not dict:
        raise TasteGlobalGCESmokeError(f"T8 deadline {label} science is absent")
    return science


def deadline_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, bytes], bytes]:
    if args.config.resolve(strict=True) != REPO_ROOT / "configs/hpc.yaml":
        raise TasteGlobalGCESmokeError("T8 deadline requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGlobalGCESmokeError("T8 deadline forbids heuristic fallback")
    attempt_id = _uuid4(args.attempt_id)
    t3_root = _absolute(args.t3_output, "T3 root")
    t4_root = _absolute(args.t4_output, "T4 root")
    t3 = _terminal_pass(t3_root, "T3")
    t4 = _terminal_pass(t4_root, "T4")
    if (t3.get("stage") != "T3_GINE_CALIBRATED" or t3.get("status") != "PASS"
            or t3.get("rf_oracle_used") is not False or t3.get("test_payload_loaded") is not False
            or t4.get("stage") != "T4_ORACLE_SMOKE" or t4.get("status") != "PASS"):
        raise TasteGlobalGCESmokeError("T8 deadline T3/T4 contract changed")

    binding_path = t4_root / "artifacts/t3_binding.json"
    binding = _read_json(binding_path, "T4-to-T3 binding")
    checkpoint = _absolute(args.gnn_checkpoint, "GINE checkpoint")
    model_path = checkpoint / "model.pt"
    if (binding.get("t3_root") != str(t3_root) or binding.get("checkpoint_dir") != str(checkpoint)
            or binding.get("rf_oracle_used") is not False or binding.get("test_payload_loaded") is not False
            or binding.get("model_sha256") != _sha256(model_path)):
        raise TasteGlobalGCESmokeError("T8 deadline T4 is not bound to frozen T3")

    payloads = {name: (checkpoint / name).read_bytes() for name in GINE_PAYLOAD_FILES}
    checkpoint_id = _sha256(model_path)
    train_contract, expected_train = _checkpoint_train_contract(
        payloads=payloads, checkpoint_evidence={"checkpoint_id": checkpoint_id})
    train_path = _absolute(args.train_csv, "train CSV")
    if train_path != expected_train or _sha256(train_path) != train_contract["sha256"]:
        raise TasteGlobalGCESmokeError("T8 deadline train CSV differs from frozen split")
    train_payload = train_path.read_bytes()
    model_card = json.loads(payloads["model_card.json"].decode())
    split = json.loads(payloads["split_manifest.json"].decode())
    if (model_card.get("dataset") != DATASET or model_card.get("backbone") != "gine"
            or model_card.get("num_classes") != NUM_CLASSES or model_card.get("source_label") != SOURCE_LABEL
            or model_card.get("rf_oracle_used") is not False
            or split.get("calibration_loaded_for_training") is not False
            or split.get("test_loaded_for_training") is not False
            or split.get("test_used_for_checkpoint_selection") is not False):
        raise TasteGlobalGCESmokeError("T8 deadline frozen GINE/split contract changed")

    official = validate_official_globalgce_root(_absolute(args.official_root, "official GlobalGCE"))
    evidence = {
        "schema_version": SCHEMA, "status": "READY", "attempt_id": attempt_id,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "stage": STAGE, "dataset": DATASET, "source_label": SOURCE_LABEL, "target_branches": [0, 2],
        "checkpoint_dir": str(checkpoint), "checkpoint_id": checkpoint_id,
        "train_csv": str(train_path), "train_sha256": train_contract["sha256"],
        "train_rows": train_contract["row_count"], "train_label_counts": train_contract["label_counts"],
        "t3_root": str(t3_root), "t3_verification_sha256": _sha256(t3_root / "verification.json"),
        "t4_root": str(t4_root), "t4_verification_sha256": _sha256(t4_root / "verification.json"),
        "t4_t3_binding_sha256": _sha256(binding_path), "official_root": official["official_root"],
        "official_commit": official["official_commit"],
        "official_runtime_source_inventory_sha256": official["runtime_source_inventory_sha256"],
        "official_runtime_source_authority": official["runtime_source_authority"],
        "num_classes": NUM_CLASSES, "rf_oracle_used": False, "calibration_loaded": False,
        "test_loaded": False, "gnn_ablation_started": False,
    }
    return evidence, payloads, train_payload


def _write_ready(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_json_bytes(evidence)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_deadline(args: argparse.Namespace) -> int:
    evidence, payloads, train_payload = deadline_preflight(args)
    if args.preflight_only:
        if args.ready_receipt is None:
            raise TasteGlobalGCESmokeError("T8 preflight requires --ready-receipt")
        receipt = {**evidence, "state": "READY_WAITING_FOR_CONTROLLER_OWNED_GPU_LOCK",
                   "project_gpu_lock_required": True, "direct_unlocked_launch_allowed": False,
                   "state_root": str(args.state_dir), "output_root": str(args.output_dir)}
        _write_ready(args.ready_receipt, receipt)
        print(json.dumps(receipt, sort_keys=True)); return 0
    if os.environ.get("TASTEMOLNET_T8_CONTROLLER_OWNED_GPU_SLOT") != "1":
        raise TasteGlobalGCESmokeError("T8 deadline refuses an unlocked direct launch")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TasteGlobalGCESmokeError("T8 deadline requires one controller-exposed GPU")
    for path, label in ((args.state_dir, "state"), (args.output_dir, "output")):
        if path.exists() or path.is_symlink():
            raise TasteGlobalGCESmokeError(f"T8 deadline {label} root must be fresh")
        path.parent.mkdir(parents=True, exist_ok=True)
    config = TasteGlobalGCESmokeConfig()
    scorer = FrozenTasteGINEScorer(payloads, device="cuda:0", batch_size=config.oracle_batch_size)
    state_root = FreshOutputDirectory.create(args.state_dir)
    state_tree = prepared = None
    try:
        def factory(target_label: int) -> OfficialGlobalGCEMutagenicityGenerator:
            return OfficialGlobalGCEMutagenicityGenerator(
                evidence["official_root"], native_train_csv=evidence["train_csv"], dataset_name=DATASET,
                min_freq=config.min_freq, frozen_gine_checkpoint=evidence["checkpoint_dir"],
                source_label=SOURCE_LABEL, target_label=target_label, num_classes=NUM_CLASSES,
                frozen_gine_payloads=payloads, native_train_payload=train_payload,
                official_source_authority=evidence["official_runtime_source_authority"], require_isolated_imports=True)
        science, state_tree = run_t8_science(
            train_payload=train_payload, expected_train_row_count=evidence["train_rows"],
            expected_train_label_counts=evidence["train_label_counts"], scorer=scorer,
            generator_factory=factory, state_root=state_root, config=config)
        rechecked, _, _ = deadline_preflight(args)
        if rechecked != evidence:
            raise TasteGlobalGCESmokeError("T8 deadline inputs changed during science")
        output = FreshOutputDirectory.create(args.output_dir)
        manifest = {**evidence, "status": "PASS", "science_sha256": hashlib.sha256(_json_bytes(science)).hexdigest(),
                    "strict_flip_count": science["strict_flip_validation"]["strict_flip_count"],
                    "destination_distribution": science["strict_flip_validation"]["destination_distribution"],
                    "canonical_rule_merge_complete": True, "canonical_candidate_dedup_complete": True,
                    "untargeted_strict_flip_complete": True}
        output.write_new("science.json", _json_bytes(science)); output.write_new("manifest.json", _json_bytes(manifest))
        output.write_new("gate.json", _json_bytes({"schema_version": SCHEMA, "status": "PASS", "marker": PASS_MARKER,
            "attempt_id": evidence["attempt_id"], "checkpoint_id": evidence["checkpoint_id"], "target_branches": [0, 2],
            "strict_flip_count": manifest["strict_flip_count"], "destination_distribution": manifest["destination_distribution"],
            "rf_oracle_used": False, "test_loaded": False}))
        prepared = prepare_terminal_output(output, marker_name="PASS", marker_payload=(PASS_MARKER + "\n").encode())
        prepared.commit(retained_input_closure=lambda: (state_root.revalidate(), state_tree.revalidate()))
        print(PASS_MARKER); print(json.dumps(manifest, sort_keys=True)); return 0
    finally:
        if prepared is not None: prepared.close()
        if state_tree is not None: state_tree.close()
        state_root.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--attempt-id", required=True); parser.add_argument("--t3-output", type=Path, required=True)
    parser.add_argument("--t4-output", type=Path, required=True); parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True); parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--ready-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_deadline(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
