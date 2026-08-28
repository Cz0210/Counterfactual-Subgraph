from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import pytest

from src.data.tastemolnet_neurosed_pairs import (
    TasteNeuroSEDPairError,
    build_connected_bfs_pairs,
    derive_feature_schema,
    pair_manifest,
    read_preparation_split_manifest,
    read_taste_split_rows,
    rows_to_graphs,
    split_boundary_manifest,
)
from src.models.tastemolnet_neurosed import (
    GCF_FORK_DISTANCE_SHA256,
    GCF_FORK_MODELS_SHA256,
    GREED_COMMIT,
    GREED_EXPERIMENTS_COMMIT,
    build_runner_model,
    build_training_model,
    model_contract,
)
from src.train.tastemolnet_neurosed import TasteNeuroSEDTrainConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_split(path: Path, split: str, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["molecule_id", "model_smiles", "label", "split"],
        )
        writer.writeheader()
        for molecule_id, smiles in rows:
            writer.writerow(
                {
                    "molecule_id": molecule_id,
                    "model_smiles": smiles,
                    "label": "1",
                    "split": split,
                }
            )


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "tastemolnet",
                "num_classes": 3,
                "label_map": {
                    "0": "Bitter",
                    "1": "Sweet",
                    "2": "Tasteless",
                },
                "source_label": 1,
                "seed": 7,
                "scaffold_overlap_gate_passed": True,
                "same_canonical_smiles_cross_split_forbidden": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_official_authority_and_runner_contract_are_pinned() -> None:
    contract = model_contract(11)
    assert GREED_COMMIT == "1c756f49625abb62c9f6de5b0059876a4c7499c1"
    assert GREED_EXPERIMENTS_COMMIT == "e85423dc943fda1979811e7449846efffec2a1e1"
    assert len(GCF_FORK_MODELS_SHA256) == len(GCF_FORK_DISTANCE_SHA256) == 64
    assert contract["training_model"] == "NormSEDModel"
    assert contract["runner_model"] == "NormGEDModel"
    assert contract["num_layers"] == 8
    assert contract["hidden_dim"] == contract["output_dim"] == 64
    assert contract["state_dict_isomorphic"] is True
    assert contract["distance_normalization"] == "divide_by_sum_graph_element_counts"
    assert hashlib.sha256(
        (
            PROJECT_ROOT
            / "baselines/gcfexplainer_official/neurosed/models.py"
        ).read_bytes()
    ).hexdigest() == GCF_FORK_MODELS_SHA256
    assert hashlib.sha256(
        (PROJECT_ROOT / "baselines/gcfexplainer_official/distance.py").read_bytes()
    ).hexdigest() == GCF_FORK_DISTANCE_SHA256


def test_formal_config_preserves_official_optimizer_and_boundary() -> None:
    config = (PROJECT_ROOT / "configs/autodl/tastemolnet_neurosed_v1.yaml").read_text(
        encoding="utf-8"
    )
    for needle in (
        "seed: 7",
        "max_grad_norm: 0.1",
        "calibration_loaded: false",
        "test_loaded: false",
        "worker_writes_pass: false",
        "auto_terminate_uncontrolled_children: false",
    ):
        assert needle in config
    TasteNeuroSEDTrainConfig().validate()


def test_formal_config_rejects_nonofficial_gradient_clip() -> None:
    with pytest.raises(ValueError, match="gradient clipping"):
        TasteNeuroSEDTrainConfig(max_grad_norm=1.0).validate()


def test_split_reader_accepts_only_declared_train_validation(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    _write_split(train, "train", [("t0", "CCO"), ("t1", "CCN")])
    _write_split(validation, "validation", [("v0", "CCC")])
    train_rows, train_evidence = read_taste_split_rows(
        train.resolve(), expected_split="train"
    )
    validation_rows, validation_evidence = read_taste_split_rows(
        validation.resolve(), expected_split="validation"
    )
    assert [row.split for row in train_rows] == ["train", "train"]
    assert [row.split for row in validation_rows] == ["validation"]
    assert len(train_evidence["graph_ids_hash"]) == 64
    assert len(validation_evidence["graph_ids_hash"]) == 64
    assert "graph_ids" not in train_evidence
    assert "smiles" not in train_evidence


@pytest.mark.parametrize("forbidden", ["calibration", "test"])
def test_split_reader_rejects_forbidden_payload_split(
    tmp_path: Path, forbidden: str
) -> None:
    path = tmp_path / f"{forbidden}.csv"
    _write_split(path, forbidden, [("x", "CCO")])
    with pytest.raises(TasteNeuroSEDPairError, match="only train and validation"):
        read_taste_split_rows(path.resolve(), expected_split=forbidden)


def test_split_reader_rejects_filename_alias_and_mixed_rows(tmp_path: Path) -> None:
    alias = tmp_path / "renamed.csv"
    _write_split(alias, "train", [("t0", "CCO")])
    with pytest.raises(TasteNeuroSEDPairError, match="must be named train.csv"):
        read_taste_split_rows(alias.resolve(), expected_split="train")
    mixed = tmp_path / "train.csv"
    _write_split(mixed, "validation", [("v0", "CCO")])
    with pytest.raises(TasteNeuroSEDPairError, match="contains split"):
        read_taste_split_rows(mixed.resolve(), expected_split="train")


def test_split_manifest_proof_never_embeds_ids_or_forbidden_payloads(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    _write_manifest(manifest_path)
    preparation = read_preparation_split_manifest(manifest_path.resolve())
    proof = split_boundary_manifest(
        train_evidence={
            "graph_ids_hash": "a" * 64,
            "row_count": 2,
            "source_csv_sha256": "b" * 64,
        },
        validation_evidence={
            "graph_ids_hash": "c" * 64,
            "row_count": 1,
            "source_csv_sha256": "d" * 64,
        },
        preparation_manifest=preparation,
        train_validation_intersection_empty=True,
    )
    encoded = json.dumps(proof, sort_keys=True)
    assert proof["opened_payload_splits"] == ["train", "validation"]
    assert proof["calibration_loaded"] is proof["test_loaded"] is False
    assert proof["calibration_graph_hashes_observed"] is False
    assert proof["test_graph_hashes_observed"] is False
    assert "CCO" not in encoded
    assert "graph_ids" not in proof


def _pyg_runtime() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("rdkit")


def test_connected_bfs_pairs_are_nested_exact_intervals(tmp_path: Path) -> None:
    _pyg_runtime()
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    _write_split(train, "train", [("t0", "CCCO"), ("t1", "CCCN")])
    _write_split(validation, "validation", [("v0", "CCCC")])
    train_rows, _ = read_taste_split_rows(train.resolve(), expected_split="train")
    validation_rows, _ = read_taste_split_rows(
        validation.resolve(), expected_split="validation"
    )
    schema = derive_feature_schema(train_rows, validation_rows)
    graphs = rows_to_graphs(train_rows, schema)
    pairs = build_connected_bfs_pairs(graphs, split="train", num_pairs=8, seed=7)
    manifest = pair_manifest(pairs, split="train")
    assert manifest["connected_queries"] is True
    assert manifest["all_lb_equal_ub"] is True
    assert manifest["cross_parent_pairs"] is False
    assert all(pair.lb == pair.ub == pair.removed_nodes + pair.removed_edges for pair in pairs)
    assert all(pair.query.num_nodes < pair.parent.num_nodes for pair in pairs)


def test_feature_schema_rejects_validation_unseen_atom(tmp_path: Path) -> None:
    _pyg_runtime()
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    _write_split(train, "train", [("t0", "CCO")])
    _write_split(validation, "validation", [("v0", "CCCl")])
    train_rows, _ = read_taste_split_rows(train.resolve(), expected_split="train")
    validation_rows, _ = read_taste_split_rows(
        validation.resolve(), expected_split="validation"
    )
    with pytest.raises(TasteNeuroSEDPairError, match="train-unseen"):
        derive_feature_schema(train_rows, validation_rows)


def test_normsed_and_runner_state_dicts_are_isomorphic() -> None:
    _pyg_runtime()
    training = build_training_model(input_dim=5, device="cpu")
    runner = build_runner_model(input_dim=5, device="cpu")
    assert list(training.state_dict()) == list(runner.state_dict())
    result = runner.load_state_dict(training.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []


def test_checkpoint_reload_batch_single_and_runner_load(tmp_path: Path) -> None:
    _pyg_runtime()
    import torch

    from src.eval.tastemolnet_neurosed_gate import checkpoint_health

    training = build_training_model(input_dim=5, device="cpu")
    checkpoint = tmp_path / "best.pt"
    torch.save(training.state_dict(), checkpoint)
    health = checkpoint_health(
        checkpoint,
        input_dim=5,
        require_cuda_tolerance=False,
    )
    assert health["checkpoint_reload"] is True
    assert health["gcf_runner_can_load"] is True
    assert health["gcf_distance_py_load_neurosed"] is True
    assert health["batch_single_agreement"] is True
    assert health["finite_distances"] is True


def test_checkpoint_cpu_gpu_numeric_tolerance_when_available(tmp_path: Path) -> None:
    _pyg_runtime()
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA health gate is exercised on AutoDL")
    from src.eval.tastemolnet_neurosed_gate import checkpoint_health

    training = build_training_model(input_dim=5, device="cpu")
    checkpoint = tmp_path / "best.pt"
    torch.save(training.state_dict(), checkpoint)
    health = checkpoint_health(
        checkpoint,
        input_dim=5,
        require_cuda_tolerance=True,
    )
    assert health["cpu_gpu_numeric_tolerance"] == "PASS"


def test_managed_worker_cannot_self_sign_neurosed_pass() -> None:
    worker = (PROJECT_ROOT / "scripts/autodl/train_tastemolnet_neurosed.py").read_text(
        encoding="utf-8"
    )
    verifier = (
        PROJECT_ROOT / "scripts/autodl/verify_tastemolnet_neurosed.py"
    ).read_text(encoding="utf-8")
    assert "[TASTE_GCF_NEUROSED_PASS]" not in worker
    assert "[TASTE_GCF_NEUROSED_PASS]" in verifier
    assert "verify_and_publish_sealed_attempt" in verifier
    assert "open_sealed_worker_artifact" in verifier


def _fake_held_managed_evidence(tmp_path: Path) -> tuple[object, dict[str, object], list[int]]:
    from src.utils.managed_execution_v2 import (
        WORKER_EXIT_SCHEMA,
        WORKER_RAW_EVIDENCE_SCHEMA,
    )

    attempt_id = str(uuid.uuid4())
    generation_token = str(uuid.uuid4())
    binding: dict[str, object] = {
        "source_execution_config_sha256": "a" * 64,
        "train_csv_sha256": "b" * 64,
        "validation_csv_sha256": "c" * 64,
        "preparation_split_manifest_sha256": "d" * 64,
        "execution_git_commit": "e" * 40,
    }
    attempt = {
        "task_id": "TASTE_GCF_NEUROSED",
        "controller_id": "controller",
        "attempt_id": attempt_id,
        "auto_terminate_uncontrolled_children": False,
        "input_hashes": {
            "train_csv": "b" * 64,
            "validation_csv": "c" * 64,
            "preparation_split_manifest": "d" * 64,
        },
        "config_hash": "a" * 64,
        "git_commit": "e" * 40,
    }
    payloads = {
        "raw_evidence.json": {
            "schema_version": WORKER_RAW_EVIDENCE_SCHEMA,
            "attempt_id": attempt_id,
            "generation_token": generation_token,
            "evidence": {
                "attempt_manifest": attempt,
                "scientific_command": ["python", "train.py", "--train-csv", "train.csv"],
                "process_lineage": {"schema_version": "managed_process_lineage_v2"},
            },
        },
        "worker_exit.json": {
            "schema_version": WORKER_EXIT_SCHEMA,
            "attempt_id": attempt_id,
            "generation_token": generation_token,
            "exit": {
                "exit_code": 0,
                "worker_closed_artifact_writers": True,
                "process_audit": {"state": "EXITED", "attempt_id": attempt_id},
            },
        },
    }
    descriptors: list[int] = []
    files: list[object] = []
    for name, payload in payloads.items():
        path = tmp_path / name
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        path.write_bytes(data)
        descriptor = os.open(path, os.O_RDONLY)
        descriptors.append(descriptor)
        files.append(
            SimpleNamespace(
                evidence=SimpleNamespace(relative_path=name, size=len(data)),
                descriptor=descriptor,
            )
        )
    held = SimpleNamespace(
        sealed=SimpleNamespace(
            attempt_id=attempt_id,
            generation_token=generation_token,
        ),
        files=tuple(files),
    )
    return held, {"managed_input_binding": binding}, descriptors


def test_independent_verifier_binds_managed_inputs(tmp_path: Path) -> None:
    from scripts.autodl.verify_tastemolnet_neurosed import _verify_managed_binding

    held, verification, descriptors = _fake_held_managed_evidence(tmp_path)
    try:
        result = _verify_managed_binding(held, verification)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    assert result["task_id"] == "TASTE_GCF_NEUROSED"
    assert result["worker_exit_code"] == 0


def test_independent_verifier_rejects_managed_input_drift(tmp_path: Path) -> None:
    from scripts.autodl.verify_tastemolnet_neurosed import _verify_managed_binding

    held, verification, descriptors = _fake_held_managed_evidence(tmp_path)
    verification["managed_input_binding"]["train_csv_sha256"] = "f" * 64  # type: ignore[index]
    try:
        with pytest.raises(ValueError, match="inputs differ"):
            _verify_managed_binding(held, verification)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_autodl_launcher_uses_managed_v2_and_never_signals() -> None:
    launcher = (
        PROJECT_ROOT / "scripts/autodl/launch_tastemolnet_neurosed.sh"
    ).read_text(encoding="utf-8")
    assert "managed_worker_v2.py" in launcher
    assert "verify_tastemolnet_neurosed.py" in launcher
    assert "gpu-index 1" in launcher
    assert "AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0" in launcher
    assert "RUN_GNN_ABLATION=0" in launcher
    assert re.search(r"\b(?:kill|pkill|killall)\b", launcher) is None
    assert "calibration.csv" not in launcher
    assert "test.csv" not in launcher


@pytest.mark.parametrize(
    "name",
    ["train_tastemolnet_neurosed.sh", "verify_tastemolnet_neurosed.sh"],
)
def test_slurm_wrappers_are_static_refusals(name: str) -> None:
    wrapper = (PROJECT_ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in wrapper
    assert "#SBATCH --gres=gpu:a800:1" in wrapper
    assert "--config configs/hpc.yaml" in wrapper
    assert "inference.fallback_to_heuristic=false" in wrapper
    assert "REFUSING_HPC_EXECUTION" in wrapper
    assert re.search(r"^exit (?:64|78)$", wrapper, flags=re.MULTILINE)
