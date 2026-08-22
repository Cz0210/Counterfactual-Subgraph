"""Read-only B7-parallel preparation released by a passing B6-v2 gate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.eval.bace_frozen_gnn_contracts import (
    CLASSIFIER_TYPE,
    DATASET,
    NUM_CLASSES,
    NUM_SHARDS,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    assert_no_rf_provenance,
    assert_stage_data_boundary,
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    fixed_parent_shard_map,
    fresh_output_dir,
    load_bace_parents,
    read_json,
    sha256_file,
    stable_sha256,
    utc_now,
    validate_pass_manifest,
)
from src.eval.bace_frozen_gnn_pool import _checkpoint_contract
from src.eval.molclr_node_embeddings import MolCLRNodeEmbedder
from src.oracles.oracle_factory import build_oracle


PREP_ACTIONS = (
    "CALIBRATION_GNN_BEFORE_CACHE",
    "CALIBRATION_MOLCLR_PARENT_CACHE",
    "FIXED_SHARD_MANIFESTS",
    "OUTPUT_PREFLIGHT",
)


def _validate_b6(b6_output: Path) -> dict[str, Any]:
    manifest = read_json(b6_output / "ppo_smoke_manifest.json")
    assert_no_rf_provenance(manifest)
    required = {
        "dataset": DATASET,
        "stage": "B6_PPO_SMOKE_V2",
        "status": "PASS",
        "ppo_training_performed": True,
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    failures = [
        f"{key}={manifest.get(key)!r}"
        for key, expected in required.items()
        if manifest.get(key) != expected
    ]
    if int(manifest.get("ppo_update_count", 0)) < 5:
        failures.append(f"ppo_update_count={manifest.get('ppo_update_count')!r}")
    if failures:
        raise ValueError("B7-parallel prep requires a passing B6-v2: " + ", ".join(failures))
    return manifest


def _graph(featurizer: MolecularGraphFeaturizer, parent: Any) -> MolecularGraphData:
    features = featurizer.featurize(parent.smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=SOURCE_LABEL,
        molecule_id=parent.parent_id,
        smiles=features.canonical_smiles,
        split="calibration_before_cache",
        graph_sha256=features.graph_sha256,
    )


def run_b7_parallel_prep(
    *,
    action: str,
    b6_output: str | Path,
    output_dir: str | Path,
    calibration_split: str | Path | None = None,
    train_split: str | Path | None = None,
    gnn_checkpoint: str | Path | None = None,
    molclr_root: str | Path | None = None,
    molclr_checkpoint: str | Path | None = None,
    node_embedding_cache_dir: str | Path | None = None,
    planned_output_roots: Sequence[str | Path] = (),
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    normalized = str(action).strip().upper()
    if normalized not in PREP_ACTIONS:
        raise ValueError(f"Prep action must be one of {PREP_ACTIONS}")
    b6_root = Path(b6_output).expanduser().resolve(strict=True)
    b6 = _validate_b6(b6_root)
    output = fresh_output_dir(output_dir)
    payload: dict[str, Any] = {
        "schema_version": "bace_b7_parallel_prep_v1",
        "dataset": DATASET,
        "stage": "B7_PARALLEL_PREP",
        "action": normalized,
        "status": "PASS",
        "dependency_stage": "B6_PPO_SMOKE_V2",
        "b6_manifest_identity": file_identity(b6_root / "ppo_smoke_manifest.json"),
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "policy_checkpoint_loaded": False,
        "candidate_generation_performed": False,
        "selector_fitted": False,
        "calibration_loaded": normalized
        in {"CALIBRATION_GNN_BEFORE_CACHE", "CALIBRATION_MOLCLR_PARENT_CACHE", "FIXED_SHARD_MANIFESTS"},
        "test_loaded": False,
        "created_at": utc_now(),
    }
    if normalized == "CALIBRATION_GNN_BEFORE_CACHE":
        if calibration_split is None or gnn_checkpoint is None:
            raise ValueError("GNN-before prep requires calibration split and GNN checkpoint")
        calibration = Path(calibration_split).expanduser().resolve(strict=True)
        if "test" in calibration.name.lower():
            raise ValueError("B7 prep may not load test data")
        checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
        card, schema = _checkpoint_contract(checkpoint)
        b6_oracle_id = b6.get("oracle_checkpoint_hash") or b6.get("checkpoint_id")
        if b6_oracle_id != card["checkpoint_id"]:
            raise ValueError("B6 and prep use different GNN checkpoints")
        parents = load_bace_parents(calibration)
        oracle = build_oracle(
            dataset=DATASET,
            backend=ORACLE_BACKEND,
            checkpoint=checkpoint,
            device=device,
            batch_size=int(batch_size),
        )
        featurizer = MolecularGraphFeaturizer(schema)
        parent_graphs = [_graph(featurizer, parent) for parent in parents]
        predictions = oracle.predict_records(
            parent_graphs,
            batch_size=int(batch_size),
        )
        canonical_by_parent = {
            graph.molecule_id: graph.smiles for graph in parent_graphs
        }
        temperature_hash = sha256_file(checkpoint / "temperature_scaling.json")
        feature_schema_hash = sha256_file(checkpoint / "feature_schema.json")
        rows = [
            {
                "parent_id": parent.parent_id,
                "parent_smiles": parent.smiles,
                "canonical_smiles": canonical_by_parent[parent.parent_id],
                "source_label": SOURCE_LABEL,
                "pred_before": prediction["predicted_label"],
                "p_before": prediction["probabilities"],
                "source_prob_before": prediction["source_probability"],
                "oracle_backend": ORACLE_BACKEND,
                "rf_oracle_used": False,
                "oracle_checkpoint_hash": card["checkpoint_id"],
                "temperature_scaling_sha256": temperature_hash,
                "feature_schema_sha256": feature_schema_hash,
                "cache_key": stable_sha256(
                    {
                        "canonical_smiles": prediction.get("canonical_smiles")
                        or canonical_by_parent[parent.parent_id],
                        "oracle_checkpoint_hash": card["checkpoint_id"],
                        "temperature_scaling_sha256": temperature_hash,
                        "feature_schema_sha256": feature_schema_hash,
                    }
                ),
            }
            for parent, prediction in zip(parents, predictions, strict=True)
        ]
        atomic_jsonl(output / "calibration_parent_gnn_before.jsonl", rows)
        payload.update(
            {
                "parent_count": len(parents),
                "oracle_checkpoint_hash": card["checkpoint_id"],
                "gnn_sha256sums_sha256": sha256_file(checkpoint / "sha256sums.txt"),
                "temperature_scaling_sha256": temperature_hash,
                "feature_schema_sha256": feature_schema_hash,
                "calibration_split_identity": file_identity(calibration),
                "cache_identity": file_identity(
                    output / "calibration_parent_gnn_before.jsonl"
                ),
            }
        )
    elif normalized == "CALIBRATION_MOLCLR_PARENT_CACHE":
        required = (calibration_split, molclr_root, molclr_checkpoint, node_embedding_cache_dir)
        if any(value is None for value in required):
            raise ValueError("MolCLR prep requires calibration, project, checkpoint, and cache")
        calibration = Path(calibration_split).expanduser().resolve(strict=True)  # type: ignore[arg-type]
        if "test" in calibration.name.lower():
            raise ValueError("B7 prep may not load test data")
        parents = load_bace_parents(calibration)
        embedder = MolCLRNodeEmbedder(
            molclr_root=Path(molclr_root).expanduser().resolve(strict=True),  # type: ignore[arg-type]
            molclr_ckpt=Path(molclr_checkpoint).expanduser().resolve(strict=True),  # type: ignore[arg-type]
            node_emb_cache_dir=Path(node_embedding_cache_dir).expanduser().resolve(strict=False),  # type: ignore[arg-type]
            device=device,
        )
        rows = []
        for parent in parents:
            embedding = embedder.get(parent.smiles)
            rows.append(
                {
                    "parent_id": parent.parent_id,
                    "canonical_smiles": embedding.canonical_smiles,
                    "num_nodes": embedding.n_atoms,
                    "embedding_dim": int(embedding.H.shape[1]),
                    "cache_path": str(embedder.cache_path(embedding.canonical_smiles)),
                }
            )
        atomic_jsonl(output / "calibration_parent_molclr_cache.jsonl", rows)
        payload.update(
            {
                "parent_count": len(parents),
                "molclr_checkpoint_hash": sha256_file(molclr_checkpoint),  # type: ignore[arg-type]
                "molclr_embedding_checkpoint_identity": embedder.checkpoint_identity,
                "calibration_split_identity": file_identity(calibration),
                "cache_inventory_identity": file_identity(
                    output / "calibration_parent_molclr_cache.jsonl"
                ),
            }
        )
    elif normalized == "FIXED_SHARD_MANIFESTS":
        if calibration_split is None or train_split is None:
            raise ValueError("Shard prep requires train and calibration splits")
        calibration = Path(calibration_split).expanduser().resolve(strict=True)
        train = Path(train_split).expanduser().resolve(strict=True)
        if "test" in calibration.name.lower() or "test" in train.name.lower():
            raise ValueError("B7 shard prep may not load test data")
        train_parents = load_bace_parents(train)
        calibration_parents = load_bace_parents(calibration)
        manifests: dict[str, Any] = {}
        for name, parents, split in (
            ("train", train_parents, train),
            ("calibration", calibration_parents, calibration),
        ):
            mapping = fixed_parent_shard_map([parent.parent_id for parent in parents])
            shard_rows = [
                {
                    "parent_id": parent_id,
                    "shard_index": mapping[parent_id],
                    "num_shards": NUM_SHARDS,
                    "shard_rule": "sorted(parent_id)_position_mod_4",
                }
                for parent_id in sorted(mapping)
            ]
            path = output / f"{name}_parent_shards.jsonl"
            atomic_jsonl(path, shard_rows)
            parent_manifest_path = output / f"{name}_parent_ids.frozen.json"
            atomic_json(
                parent_manifest_path,
                {
                    "schema_version": "bace_frozen_parent_ids_v1",
                    "status": "FROZEN",
                    "dataset": DATASET,
                    "split": name,
                    "source_label": SOURCE_LABEL,
                    "num_classes": NUM_CLASSES,
                    "parent_ids": sorted(mapping),
                    "parent_count": len(parents),
                    "parent_ids_sha256": stable_sha256(sorted(mapping)),
                    "shard_count": NUM_SHARDS,
                    "shard_rule": "sorted(parent_id)_position_mod_4",
                    "split_identity": file_identity(split),
                },
            )
            manifests[name] = {
                "split_identity": file_identity(split),
                "parent_count": len(parents),
                "parent_ids_sha256": stable_sha256(sorted(mapping)),
                "shard_manifest_identity": file_identity(path),
                "parent_manifest_identity": file_identity(parent_manifest_path),
                "shard_counts": {
                    str(index): sum(row["shard_index"] == index for row in shard_rows)
                    for index in range(NUM_SHARDS)
                },
            }
        payload["fixed_shards"] = manifests
    else:
        if not planned_output_roots:
            raise ValueError("Output preflight requires planned fresh output roots")
        rows = []
        for path_like in planned_output_roots:
            path = Path(path_like).expanduser()
            if not path.is_absolute():
                raise ValueError(f"Planned scientific output must be absolute: {path}")
            resolved = path.resolve(strict=False)
            if resolved.exists():
                raise FileExistsError(f"Planned scientific output is not fresh: {resolved}")
            parent = resolved.parent
            parent.mkdir(parents=True, exist_ok=True)
            stats = os.statvfs(parent)
            free_bytes = int(stats.f_bavail * stats.f_frsize)
            if free_bytes <= 0:
                raise RuntimeError(f"No free space for planned output: {resolved}")
            rows.append(
                {
                    "planned_output_root": str(resolved),
                    "exists": False,
                    "parent": str(parent),
                    "free_bytes": free_bytes,
                }
            )
        atomic_jsonl(output / "output_preflight.jsonl", rows)
        payload.update(
            {
                "planned_output_count": len(rows),
                "output_preflight_identity": file_identity(
                    output / "output_preflight.jsonl"
                ),
            }
        )
    atomic_json(output / "prep_manifest.json", payload)
    atomic_marker(output / "PASS", "PASS")
    return payload


def run_postfreeze_test_shard_manifest(
    *,
    b12_output: str | Path,
    test_split: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze B13 parent IDs only after the B12 selector is immutable.

    The B12 gate is evaluated before the test path is resolved or opened.  The
    controller consumes ``test_parent_ids.frozen.json`` to materialize its four
    fixed instances; B7 prep must never create this artifact.
    """

    b12_root = Path(b12_output).expanduser().resolve(strict=True)
    frozen_path = (b12_root / "frozen_selection_manifest.json").resolve(strict=True)
    assert_stage_data_boundary(
        stage="B13_FINAL_EVAL",
        split_path=test_split,
        frozen_selection_manifest=frozen_path,
    )
    frozen = validate_pass_manifest(
        frozen_path,
        expected_stage="B12_SELECTOR",
        require_no_test=True,
    )
    test_path = Path(test_split).expanduser().resolve(strict=True)
    parents = load_bace_parents(test_path)
    parent_ids = sorted(parent.parent_id for parent in parents)
    mapping = fixed_parent_shard_map(parent_ids)
    output = fresh_output_dir(output_dir)
    document = {
        "schema_version": "bace_postfreeze_test_parent_ids_v1",
        "status": "FROZEN",
        "dataset": DATASET,
        "split": "test",
        "stage": "B13_TEST_PARENT_MANIFEST",
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "parent_ids": parent_ids,
        "parent_count": len(parent_ids),
        "parent_ids_sha256": stable_sha256(parent_ids),
        "shard_count": NUM_SHARDS,
        "shard_rule": "sorted(parent_id)_position_mod_4",
        "shards": {
            str(index): [
                parent_id for parent_id in parent_ids if mapping[parent_id] == index
            ]
            for index in range(NUM_SHARDS)
        },
        "test_split_identity": file_identity(test_path),
        "frozen_selection_manifest_identity": file_identity(frozen_path),
        "selection_frozen_before_test_load": True,
        "selector_fitted_on_calibration": frozen["selector_fitted_on_calibration"],
        "test_loaded": True,
        "created_at": utc_now(),
    }
    atomic_json(output / "test_parent_ids.frozen.json", document)
    manifest = {
        **document,
        "status": "PASS",
        "test_parent_ids_identity": file_identity(
            output / "test_parent_ids.frozen.json"
        ),
    }
    atomic_json(output / "test_shard_manifest.json", manifest)
    atomic_marker(output / "PASS", "PASS")
    return manifest


__all__ = [
    "PREP_ACTIONS",
    "run_b7_parallel_prep",
    "run_postfreeze_test_shard_manifest",
]
