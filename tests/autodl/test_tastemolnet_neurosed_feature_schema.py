from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from src.data.tastemolnet_neurosed_pairs import (
    TasteNeuroSEDPairError,
    TasteSplitRow,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts/autodl/build_tastemolnet_neurosed_feature_schema.py"
)
SLURM_PATH = (
    PROJECT_ROOT / "scripts/slurm/build_tastemolnet_neurosed_feature_schema.sh"
)
TRAIN_SHA256 = "a" * 64
VALIDATION_SHA256 = "b" * 64


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("taste_neurosed_schema_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema() -> dict[str, object]:
    return {
        "schema_version": "tastemolnet_gcf_neurosed_feature_schema_v1",
        "dataset": "tastemolnet",
        "node_feature_semantics": "one_hot_atomic_number",
        "feature_atomic_numbers": [1, 6, 7, 8],
        "input_dim": 4,
        "explicit_h_nodes": True,
        "native_adjacency_semantics": "binary_connectivity_directed_both_ways",
        "edge_features_used": False,
        "validation_unseen_atomic_numbers": [],
        "train_derived_only": True,
        "maximum_train_or_validation_nodes": 12,
    }


def _evidence(role: str, sha256: str, count: int) -> dict[str, object]:
    return {
        "split": role,
        "source_csv_sha256": sha256,
        "row_count": count,
        "graph_ids_hash": "c" * 64,
        "all_rows_declared_expected_split": True,
        "labels_opened_but_not_consumed": True,
    }


def _patch_inputs(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_id: str = "train-0",
    validation_id: str = "validation-0",
    validation_role: str = "validation",
    validation_observed_sha256: str = VALIDATION_SHA256,
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_reader(path: Path, *, expected_split: str):
        calls.append((Path(path).name, expected_split))
        if expected_split == "train":
            return (
                [TasteSplitRow(train_id, "CCO", "train")],
                _evidence("train", TRAIN_SHA256, 1),
            )
        return (
            [TasteSplitRow(validation_id, "CCN", validation_role)],
            _evidence(validation_role, validation_observed_sha256, 1),
        )

    def fake_derive(train_rows, validation_rows):
        assert [row.split for row in train_rows] == ["train"]
        assert [row.split for row in validation_rows] == ["validation"]
        return _schema()

    monkeypatch.setattr(module, "read_taste_split_rows", fake_reader)
    monkeypatch.setattr(module, "derive_feature_schema", fake_derive)
    return calls


def _argv(tmp_path: Path, output: Path) -> list[str]:
    return [
        "--config",
        str(PROJECT_ROOT / "configs/hpc.yaml"),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--expected-train-sha256",
        TRAIN_SHA256,
        "--validation-csv",
        str(tmp_path / "validation.csv"),
        "--expected-validation-sha256",
        VALIDATION_SHA256,
        "--output-json",
        str(output),
    ]


def test_cli_writes_only_canonical_schema_and_reports_no_heldout_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    calls = _patch_inputs(module, monkeypatch)
    output = tmp_path / "fresh" / "feature_schema.json"

    assert module.main(_argv(tmp_path, output)) == 0

    assert calls == [("train.csv", "train"), ("validation.csv", "validation")]
    assert json.loads(output.read_text(encoding="utf-8")) == _schema()
    assert output.read_bytes().endswith(b"\n")
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["opened_payload_splits"] == ["train", "validation"]
    assert receipt["forbidden_payload_splits_opened"] == []
    assert receipt["calibration_payload_opened"] is False
    assert receipt["test_payload_opened"] is False
    assert receipt["no_calibration_or_test_payload_access_evidence"] is True
    assert receipt["labels_used"] is False
    assert receipt["classifier_used"] is False
    assert receipt["feature_schema_atomic_no_replace"] is True
    assert receipt["feature_schema_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert set(json.loads(output.read_text(encoding="utf-8"))) == module.FEATURE_SCHEMA_FIELDS


@pytest.mark.parametrize(
    ("validation_role", "validation_sha", "message"),
    [
        ("test", VALIDATION_SHA256, "role/SHA"),
        ("validation", "d" * 64, "role/SHA"),
    ],
)
def test_cli_rejects_split_role_or_sha_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    validation_role: str,
    validation_sha: str,
    message: str,
) -> None:
    module = _load_script()
    _patch_inputs(
        module,
        monkeypatch,
        validation_role=validation_role,
        validation_observed_sha256=validation_sha,
    )
    output = tmp_path / "feature_schema.json"

    assert module.main(_argv(tmp_path, output)) == 78
    assert not output.exists()
    assert message in capsys.readouterr().err


def test_cli_rejects_cross_split_id_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    _patch_inputs(module, monkeypatch, train_id="same", validation_id="same")
    output = tmp_path / "feature_schema.json"

    assert module.main(_argv(tmp_path, output)) == 78
    assert not output.exists()
    assert "molecule IDs overlap" in capsys.readouterr().err


def test_cli_never_overwrites_existing_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    _patch_inputs(module, monkeypatch)
    output = tmp_path / "feature_schema.json"
    output.write_bytes(b"owned-before\n")

    assert module.main(_argv(tmp_path, output)) == 78
    assert output.read_bytes() == b"owned-before\n"
    assert "must be fresh" in capsys.readouterr().err


def test_cli_strictly_validates_compatibility_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    called = False

    def forbidden_reader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("split payload must not open")

    monkeypatch.setattr(module, "read_taste_split_rows", forbidden_reader)
    output = tmp_path / "feature_schema.json"
    argv = _argv(tmp_path, output)
    argv[argv.index("inference.fallback_to_heuristic=false")] = (
        "inference.fallback_to_heuristic=true"
    )
    assert module.main(argv) == 78
    assert called is False
    assert "--set must be exactly" in capsys.readouterr().err

    with pytest.raises(TasteNeuroSEDPairError, match="exactly one"):
        module._validate_config(
            [str(PROJECT_ROOT / "configs/hpc.yaml")] * 2
        )
    with pytest.raises(TasteNeuroSEDPairError, match="this checkout"):
        module._validate_config([str(PROJECT_ROOT / "configs/local.yaml")])


def _write_split(path: Path, role: str, rows: list[tuple[str, str]]) -> None:
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
                    "split": role,
                }
            )


def test_real_loader_and_derive_feature_schema_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rdkit import Chem

    from src.data import tastemolnet_neurosed_pairs as pair_module

    # derive_feature_schema itself needs only RDKit. The shared graph module's
    # broader runtime gate also imports torch/PyG for later graph conversion;
    # keep this focused producer fixture independent of those unused packages.
    monkeypatch.setattr(
        pair_module, "_require_chemistry", lambda: (None, None, Chem)
    )
    module = _load_script()
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    _write_split(train, "train", [("t0", "CCO"), ("t1", "CCN")])
    _write_split(validation, "validation", [("v0", "CCC")])
    output = tmp_path / "feature_schema.json"

    receipt = module.build_feature_schema(
        train_csv=train,
        expected_train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        validation_csv=validation,
        expected_validation_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
        output_json=output,
    )

    schema = json.loads(output.read_text(encoding="utf-8"))
    assert schema["schema_version"] == "tastemolnet_gcf_neurosed_feature_schema_v1"
    assert schema["feature_atomic_numbers"] == [1, 6, 7, 8]
    assert schema["input_dim"] == 4
    assert receipt["opened_payload_splits"] == ["train", "validation"]
    assert receipt["calibration_payload_opened"] is False
    assert receipt["test_payload_opened"] is False


def test_source_and_paired_slurm_contracts_parse() -> None:
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))
    slurm = SLURM_PATH.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
        "--expected-train-sha256",
        "--expected-validation-sha256",
        "--output-json",
    ):
        assert required in slurm
    refusal = slurm.index("exit 78")
    invocation = slurm.index(
        "python -B scripts/autodl/build_tastemolnet_neurosed_feature_schema.py"
    )
    assert refusal < invocation
