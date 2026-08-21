#!/usr/bin/env python3
"""Run the bounded B4 calibration or B5 calibrated-oracle smoke payload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from scripts.calibrate_gnn_classifier import main as calibrate_main
from scripts.evaluate_molecular_gnn import main as evaluate_main
from src.chem.hard_deletion import (
    HardDeletionOutcome,
    apply_hard_deletion_match,
    enumerate_connected_hard_deletions,
)
from src.data.molecular_graph_dataset import MolecularGraphData, MolecularGraphDataset
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.oracles.gnn_oracle import (
    GNNOracle,
    sha256_file,
    update_checkpoint_sha256sums,
    verify_checkpoint_bundle,
)
from src.utils.autodl_runtime import atomic_write_json, fsync_directory

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required by the AutoDL environment gate.
    Chem = None


def _absolute(value: str, *, label: str, must_exist: bool) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute: {candidate}")
    return candidate.resolve(strict=must_exist)


def _fresh_output(value: str, *, label: str) -> Path:
    output = _absolute(value, label=label, must_exist=False)
    if output.exists():
        raise FileExistsError(f"{label} must be absent: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _copy_checkpoint_atomically(source: Path, destination: Path) -> None:
    if source == destination:
        raise ValueError("Calibration output must differ from the B3 checkpoint")
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.copy-", dir=destination.parent)
    )
    staged = temporary_parent / "checkpoint"
    try:
        shutil.copytree(source, staged, symlinks=False)
        os.replace(staged, destination)
        fsync_directory(destination.parent)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _config_arguments(configs: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in configs:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        result.extend(("--config", str(path)))
    return result


def validate_b3_validation_provenance(
    checkpoint: Path,
    validation_csv: Path,
) -> dict[str, str]:
    """Bind B4 to the exact validation path and bytes frozen by B3."""

    manifest_path = checkpoint / "split_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    row = files.get("validation") if isinstance(files, dict) else None
    if not isinstance(row, dict):
        raise ValueError("B3 split manifest has no validation file provenance")
    frozen_path_value = row.get("path")
    frozen_sha = row.get("sha256")
    if not isinstance(frozen_path_value, str) or not Path(
        frozen_path_value
    ).expanduser().is_absolute():
        raise ValueError("B3 validation provenance path is missing or non-absolute")
    frozen_path = Path(frozen_path_value).expanduser().resolve(strict=True)
    if frozen_path != validation_csv:
        raise ValueError(
            "B4 validation path differs from B3 frozen provenance: "
            f"B3={frozen_path}, B4={validation_csv}"
        )
    actual_sha = sha256_file(validation_csv)
    if not isinstance(frozen_sha, str) or frozen_sha != actual_sha:
        raise ValueError(
            "B4 validation SHA differs from B3 frozen provenance: "
            f"B3={frozen_sha!r}, B4={actual_sha}"
        )
    roles = payload.get("roles")
    if not isinstance(roles, dict) or roles.get("validation") != (
        "checkpoint_selection_and_temperature_calibration"
    ):
        raise ValueError("B3 split manifest does not authorize validation calibration")
    if payload.get("test_used_for_checkpoint_selection") is not False:
        raise ValueError("B3 split manifest has unsafe test-selection provenance")
    if payload.get("calibration_loaded_for_training") is not False:
        raise ValueError("B3 split manifest has unsafe calibration-training provenance")
    return {
        "split_manifest": str(manifest_path.resolve(strict=True)),
        "split_manifest_sha256": sha256_file(manifest_path),
        "validation_csv": str(validation_csv),
        "validation_csv_sha256": actual_sha,
    }


def validate_bace_model_card(model_card: dict[str, Any]) -> None:
    required = {
        "dataset": "bace",
        "num_classes": 2,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
    }
    failures = [
        f"{key}={model_card.get(key)!r}"
        for key, expected in required.items()
        if model_card.get(key) != expected
    ]
    if failures:
        raise ValueError("B5 BACE model-card contract failed: " + ", ".join(failures))


def require_every_parent_has_connected_deletion(
    counts: dict[str, int],
    *,
    expected_parents: int = 16,
) -> None:
    missing = sorted(parent_id for parent_id, count in counts.items() if count < 1)
    if len(counts) != expected_parents or missing:
        raise ValueError(
            "B5 requires at least one connected deletion for each of 16 parents: "
            f"observed_parents={len(counts)}, missing={missing}"
        )


def run_calibration(args: argparse.Namespace) -> int:
    source = _absolute(
        args.source_checkpoint, label="source checkpoint", must_exist=True
    )
    validation = _absolute(
        args.validation_csv, label="validation CSV", must_exist=True
    )
    output = _fresh_output(args.output_checkpoint, label="output checkpoint")
    if not source.is_dir() or not validation.is_file():
        raise FileNotFoundError("B4 requires a checkpoint directory and validation CSV")
    source_audit = verify_checkpoint_bundle(source)
    source_card = source_audit["model_card"]
    if (
        source_card.get("dataset") != "bace"
        or source_card.get("selection_split") != "validation"
        or source_card.get("temperature_calibration_split") != "validation"
    ):
        raise ValueError("B4 source is not a validation-selected BACE checkpoint")
    validation_provenance = validate_b3_validation_provenance(source, validation)
    source_temperature_path = source / "temperature_scaling.json"
    source_temperature = json.loads(source_temperature_path.read_text(encoding="utf-8"))
    if source_temperature.get("status") != "not_fit":
        raise ValueError("B4 requires an uncalibrated B3 checkpoint")
    if source_temperature.get("test_used_for_fit") is not False:
        raise ValueError("B3 checkpoint has unsafe temperature provenance")
    source_bundle_hash_before = sha256_file(source / "sha256sums.txt")
    source_temperature_hash_before = sha256_file(source_temperature_path)

    _copy_checkpoint_atomically(source, output)
    calibration_args = [
        *_config_arguments(args.config),
        "--checkpoint-dir",
        str(output),
        "--validation-csv",
        str(validation),
        "--split",
        "validation",
        "--device",
        "cpu",
        "--max-iter",
        str(args.max_iter),
    ]
    result = calibrate_main(calibration_args)
    if result != 0:
        raise RuntimeError(f"calibrate_gnn_classifier.py exited {result}")

    audit = verify_checkpoint_bundle(output)
    temperature = json.loads(
        (output / "temperature_scaling.json").read_text(encoding="utf-8")
    )
    required = {
        "status": "fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "argmax_invariant": True,
    }
    failures = [
        f"{key}={temperature.get(key)!r}"
        for key, expected in required.items()
        if temperature.get(key) != expected
    ]
    if failures:
        raise ValueError("Unsafe temperature result: " + ", ".join(failures))
    for key in (
        "temperature",
        "nll_before",
        "nll_after",
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
    ):
        value = float(temperature[key])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite calibration value: {key}={value}")
    if temperature.get("validation_csv_sha256") != sha256_file(validation):
        raise ValueError("B4 validation CSV hash mismatch")
    if source_bundle_hash_before != sha256_file(source / "sha256sums.txt"):
        raise RuntimeError("B4 mutated the B3 sha256 manifest")
    if source_temperature_hash_before != sha256_file(source_temperature_path):
        raise RuntimeError("B4 mutated the B3 temperature document")

    stage_result = {
        "schema_version": "bace_gnn_b4_calibration_v1",
        "status": "PASS",
        "source_checkpoint": str(source),
        "source_sha256sums_sha256": source_bundle_hash_before,
        "output_checkpoint": str(output),
        "checkpoint_id": audit["model_card"]["checkpoint_id"],
        "validation_csv": str(validation),
        "validation_csv_sha256": sha256_file(validation),
        "b3_validation_provenance": validation_provenance,
        "temperature_scaling": temperature,
        "test_used": False,
    }
    atomic_write_json(output / "b4_calibration.json", stage_result)
    update_checkpoint_sha256sums(output)
    verify_checkpoint_bundle(output)
    print(json.dumps(stage_result, sort_keys=True), flush=True)
    print("[BACE_GNN_CALIBRATION_PASS]", flush=True)
    return 0


def _graph_from_smiles(
    featurizer: MolecularGraphFeaturizer,
    smiles: str,
    molecule_id: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=-1,
        molecule_id=molecule_id,
        smiles=features.canonical_smiles,
        split="oracle_smoke",
        graph_sha256=features.graph_sha256,
    )


def select_correctly_predicted_source_indices(
    labels: Sequence[int],
    prediction_records: Sequence[dict[str, Any]],
    *,
    source_label: int,
    count: int,
) -> list[int]:
    """Freeze the first N calibration rows satisfying the main cohort rule."""

    if len(labels) != len(prediction_records):
        raise ValueError("Calibration labels and predictions have different lengths")
    selected = [
        index
        for index, (label, prediction) in enumerate(
            zip(labels, prediction_records, strict=True)
        )
        if int(label) == int(source_label)
        and int(prediction["predicted_label"]) == int(source_label)
    ][:count]
    if len(selected) != count:
        raise ValueError(
            "B5 requires exactly "
            f"{count} correctly-predicted source calibration parents; found "
            f"{len(selected)}"
        )
    return selected


def _real_connected_deletions(
    parent_smiles: str,
    *,
    parent_id: str,
    maximum: int,
) -> list[tuple[str, HardDeletionOutcome]]:
    """Enumerate bounded, connected one-atom/two-atom actions deterministically."""

    if Chem is None:
        raise RuntimeError("RDKit is required for B5 deletion smoke")
    parent = Chem.MolFromSmiles(parent_smiles, sanitize=True)
    if parent is None:
        return []
    atom_sets: list[tuple[int, ...]] = [
        (int(atom.GetIdx()),) for atom in parent.GetAtoms()
    ]
    atom_sets.extend(
        sorted(
            {
                tuple(
                    sorted(
                        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
                    )
                )
                for bond in parent.GetBonds()
            }
        )
    )
    retained: list[tuple[str, HardDeletionOutcome]] = []
    seen: set[tuple[tuple[int, ...], str]] = set()
    for attempt, atom_indices in enumerate(atom_sets):
        outcome = apply_hard_deletion_match(
            parent,
            atom_indices,
            parent_id=parent_id,
            candidate_id=f"b5-{parent_id}-{attempt}",
            match_id=attempt,
        )
        if not outcome.valid or not outcome.residual_smiles:
            continue
        try:
            fragment = Chem.MolFragmentToSmiles(
                parent,
                atomsToUse=list(atom_indices),
                canonical=True,
                isomericSmiles=True,
            )
        except Exception:
            continue
        if not fragment or Chem.MolFromSmiles(fragment, sanitize=True) is None:
            continue
        identity = (outcome.match_atom_indices, outcome.residual_smiles)
        if identity in seen:
            continue
        seen.add(identity)
        retained.append((fragment, outcome))
        if len(retained) >= maximum:
            break
    return retained


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def run_oracle_smoke(args: argparse.Namespace) -> int:
    if args.source_count != 16:
        raise ValueError("B5 source cohort is frozen at exactly 16 parents")
    if args.batch_size <= 0 or args.max_deletions_per_parent <= 0:
        raise ValueError("B5 batch/deletion limits must be positive")
    checkpoint = _absolute(
        args.checkpoint_dir, label="calibrated checkpoint", must_exist=True
    )
    calibration = _absolute(
        args.calibration_csv, label="calibration CSV", must_exist=True
    )
    output = _fresh_output(args.output_dir, label="oracle-smoke output")
    verify_checkpoint_bundle(checkpoint)
    temperature = json.loads(
        (checkpoint / "temperature_scaling.json").read_text(encoding="utf-8")
    )
    if (
        temperature.get("status") != "fit"
        or temperature.get("selection_split") != "validation"
        or temperature.get("test_used_for_fit") is not False
        or temperature.get("argmax_invariant") is not True
    ):
        raise ValueError("B5 requires a validation-only calibrated B4 checkpoint")

    evaluate_args = [
        *_config_arguments(args.config),
        "--checkpoint-dir",
        str(checkpoint),
        "--dataset-csv",
        str(calibration),
        "--dataset",
        "bace",
        "--split",
        "calibration",
        "--output-dir",
        str(output),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    result = evaluate_main(evaluate_args)
    if result != 0:
        raise RuntimeError(f"evaluate_molecular_gnn.py exited {result}")

    feature_schema_payload = json.loads(
        (checkpoint / "feature_schema.json").read_text(encoding="utf-8")
    )
    from src.data.molecular_graph_featurizer import MolecularFeatureSchema

    featurizer = MolecularGraphFeaturizer(
        MolecularFeatureSchema.from_dict(feature_schema_payload)
    )
    model_card = json.loads(
        (checkpoint / "model_card.json").read_text(encoding="utf-8")
    )
    validate_bace_model_card(model_card)
    dataset = MolecularGraphDataset.from_csv(
        calibration,
        num_classes=int(model_card["num_classes"]),
        featurizer=featurizer,
        expected_split="calibration",
    )
    oracle = GNNOracle.from_checkpoint(
        checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    calibration_graphs = [dataset[index] for index in range(len(dataset))]
    calibration_records = oracle.predict_records(
        calibration_graphs, batch_size=args.batch_size
    )
    selected_indices = select_correctly_predicted_source_indices(
        dataset.labels,
        calibration_records,
        source_label=oracle.source_label,
        count=args.source_count,
    )
    graphs = [dataset[index] for index in selected_indices]
    selected_records = [calibration_records[index] for index in selected_indices]
    batched = oracle.predict_proba(graphs, batch_size=args.batch_size)
    singles = np.vstack([oracle.predict_proba([graph]) for graph in graphs])
    if not np.isfinite(batched).all() or not np.allclose(
        batched, singles, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("B5 batch/single calibrated probabilities differ")
    records = oracle.predict_records(graphs, batch_size=args.batch_size)
    required_record_keys = {
        "predicted_label",
        "probabilities",
        "logits",
        "source_probability",
        "confidence",
        "checkpoint_id",
        "backbone",
        "num_classes",
        "temperature",
    }
    if any(not required_record_keys.issubset(record) for record in records):
        raise RuntimeError("B5 predict_records contract is incomplete")
    if any(int(record["predicted_label"]) != oracle.source_label for record in records):
        raise RuntimeError("B5 selected batch is not entirely predicted source")

    full_parent = enumerate_connected_hard_deletions("CC", "CC")
    invalid = enumerate_connected_hard_deletions("CC", "not-a-smiles")
    if not full_parent or any(outcome.valid for outcome in full_parent):
        raise RuntimeError("B5 empty residual did not fail closed")
    if invalid:
        raise RuntimeError("B5 invalid deletion did not fail closed")
    source_label = int(oracle.source_label)
    deletion_payloads: list[tuple[int, str, HardDeletionOutcome, MolecularGraphData]] = []
    parent_deletion_counts: dict[str, int] = {}
    for cohort_position, graph in enumerate(graphs):
        actions = _real_connected_deletions(
            graph.smiles,
            parent_id=graph.molecule_id,
            maximum=args.max_deletions_per_parent,
        )
        parent_deletion_counts[graph.molecule_id] = len(actions)
        for action_index, (fragment, outcome) in enumerate(actions):
            deletion_payloads.append(
                (
                    cohort_position,
                    fragment,
                    outcome,
                    _graph_from_smiles(
                        featurizer,
                        str(outcome.residual_smiles),
                        f"{graph.molecule_id}-residual-{action_index}",
                    ),
                )
            )
    require_every_parent_has_connected_deletion(
        parent_deletion_counts, expected_parents=args.source_count
    )
    pair_graphs: list[MolecularGraphData] = []
    for cohort_position, _fragment, _outcome, residual_graph in deletion_payloads:
        pair_graphs.extend((graphs[cohort_position], residual_graph))
    pair_records = oracle.predict_records(pair_graphs, batch_size=args.batch_size)
    deletion_rows: list[dict[str, Any]] = []
    for pair_index, (cohort_position, fragment, outcome, _residual_graph) in enumerate(
        deletion_payloads
    ):
        parent_record = pair_records[2 * pair_index]
        residual_record = pair_records[2 * pair_index + 1]
        pred_before = int(parent_record["predicted_label"])
        pred_after = int(residual_record["predicted_label"])
        probability_before = float(parent_record["source_probability"])
        probability_after = float(residual_record["source_probability"])
        if pred_before != source_label:
            raise RuntimeError("B5 deletion pair escaped the frozen source cohort")
        deletion_rows.append(
            {
                "parent_id": graphs[cohort_position].molecule_id,
                "parent_smiles": graphs[cohort_position].smiles,
                "source_label": source_label,
                "fragment_smiles": fragment,
                "match_atom_indices": list(outcome.match_atom_indices),
                "residual_smiles": outcome.residual_smiles,
                "residual_connected": outcome.residual_connected,
                "sanitize_ok": outcome.sanitize_ok,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "probabilities_before": parent_record["probabilities"],
                "probabilities_after": residual_record["probabilities"],
                "cf_drop": probability_before - probability_after,
                "cf_flip": pred_before == source_label and pred_after != source_label,
                "checkpoint_id": oracle.checkpoint_id,
                "temperature": oracle.temperature,
            }
        )
    _write_jsonl(output / "deletion_records.jsonl", deletion_rows)
    selected_parent_rows = [
        {
            "molecule_id": graph.molecule_id,
            "smiles": graph.smiles,
            "label": int(dataset.labels[index]),
            "pred_before": int(prediction["predicted_label"]),
            "source_probability": float(prediction["source_probability"]),
        }
        for graph, index, prediction in zip(
            graphs, selected_indices, selected_records, strict=True
        )
    ]
    smoke = {
        "schema_version": "bace_gnn_b5_oracle_smoke_v1",
        "status": "PASS",
        "checkpoint_dir": str(checkpoint),
        "checkpoint_id": oracle.checkpoint_id,
        "checkpoint_sha256sums_sha256": sha256_file(checkpoint / "sha256sums.txt"),
        "calibration_csv": str(calibration),
        "calibration_csv_sha256": sha256_file(calibration),
        "evaluation_split": "calibration",
        "test_loaded": False,
        "rf_guard_pass": True,
        "temperature": oracle.temperature,
        "selected_count": len(graphs),
        "selected_parents": selected_parent_rows,
        "all_selected_true_source": all(
            row["label"] == source_label for row in selected_parent_rows
        ),
        "all_selected_predicted_source": all(
            row["pred_before"] == source_label for row in selected_parent_rows
        ),
        "batch_examples": len(graphs),
        "batch_single_max_abs_difference": float(
            np.max(np.abs(batched - singles))
        ),
        "record_contract_keys": sorted(required_record_keys),
        "checkpoint_load_count_for_deletion_pairs": 1,
        "valid_deletion_count": len(deletion_rows),
        "parents_with_valid_deletion": sum(
            count > 0 for count in parent_deletion_counts.values()
        ),
        "parent_deletion_counts": parent_deletion_counts,
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
    }
    atomic_write_json(output / "oracle_smoke.json", smoke)
    print(json.dumps(smoke, sort_keys=True), flush=True)
    print("[BACE_GNN_ORACLE_SMOKE_PASS]", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    commands = parser.add_subparsers(dest="action", required=True)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--source-checkpoint", required=True)
    calibrate.add_argument("--output-checkpoint", required=True)
    calibrate.add_argument("--validation-csv", required=True)
    calibrate.add_argument("--max-iter", type=int, default=100)

    smoke = commands.add_parser("oracle-smoke")
    smoke.add_argument("--checkpoint-dir", required=True)
    smoke.add_argument("--calibration-csv", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--batch-size", type=int, default=32)
    smoke.add_argument("--source-count", type=int, default=16)
    smoke.add_argument("--max-deletions-per-parent", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "calibrate":
        return run_calibration(args)
    if args.action == "oracle-smoke":
        return run_oracle_smoke(args)
    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
