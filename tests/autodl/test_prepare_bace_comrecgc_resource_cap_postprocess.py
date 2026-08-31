from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import pytest

from scripts.autodl.prepare_bace_comrecgc_resource_cap_postprocess import prepare
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.baselines.comrecgc.contracts import sha256_file
from src.utils.autodl_bace_comrecgc_resource_cap_executor import (
    build_postprocess_fragment,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def physical_root() -> Path:
    # Controller validation intentionally treats any path segment named
    # ``test`` as held-out data.  Pytest includes the test function name in
    # tmp_path, so use a neutral physical root for the generated commands.
    root = Path(tempfile.mkdtemp(prefix="bace-cap-fixture-"))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _source_fragment(tmp_path: Path, *, effective: int = 20_000) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    python = tmp_path / "python"
    python.write_text("python\n", encoding="utf-8")
    checkpoint = tmp_path / "gnn"
    checkpoint.mkdir()
    root = tmp_path / "resource-cap"
    generation = root / "train_generation"
    generation.mkdir(parents=True)
    _write_json(
        generation / "run_manifest.json",
        {
            "M_configured_max": 20_000,
            "M_effective": effective,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
        },
    )
    _write_json(
        generation / "_RUN_COMPLETE.json",
        {"run_complete": True, "M_effective": effective},
    )
    receipt = root / "executor/resource_cap_receipt.json"
    _write_json(receipt, {"M_effective": effective, "test_loaded": False})
    fragment = build_postprocess_fragment(
        python=python,
        project_root=project,
        output_root=root,
        gnn_checkpoint=checkpoint,
        dataset_dir=tmp_path / "dataset",
        calibration_split=tmp_path / "calibration.csv",
        test_split=tmp_path / "test.csv",
        molclr_root=tmp_path / "molclr",
        molclr_checkpoint=tmp_path / "molclr.pt",
        neurosed_checkpoint=tmp_path / "neurosed.pt",
        official_root=tmp_path / "official",
        resource_cap_receipt=receipt,
    )
    source = root / "executor/postprocess.tasks.json"
    _write_json(source, fragment)
    return source


def test_prepare_adopts_exact_generation_and_routes_mutable_outputs(
    tmp_path: Path, physical_root: Path,
) -> None:
    source = _source_fragment(physical_root)
    generic = tmp_path / "control/comrecgc-generic.json"
    manifest = tmp_path / "control/comrecgc-controller.json"
    result = prepare(
        source_fragment=source,
        generic_fragment_output=generic,
        manifest_output=manifest,
        controller_id="bace-comrecgc-resource-cap-postprocess-test",
    )

    assert result["status"] == "PASS"
    payload = json.loads(generic.read_text(encoding="utf-8"))
    by_id = {task["id"]: task for task in payload["tasks"]}
    assert "bace_comrecgc_train_generation" not in by_id
    generation = str(source.resolve().parents[1] / "train_generation")
    recourse = by_id["bace_comrecgc_train_common_recourse"]
    assert generation in recourse["command"]
    assert "{task_output}/_native_aux/train_generation" not in json.dumps(
        recourse, sort_keys=True
    )
    assert recourse["depends_on"] == ["bace_comrecgc_preflight"]
    assert by_id["bace_comrecgc_standardized"]["read_only_test"] is True
    assert by_id["bace_comrecgc_standardized"]["required_log_marker"] == (
        "[BACE_FROZEN_CELL_STANDARDIZATION_PASS]"
    )
    serialized = json.dumps(payload, sort_keys=True)
    native_root = str(source.parents[1])
    assert f"{native_root}/calibration/cache/" not in serialized
    assert f"{native_root}/test/cache/" not in serialized
    assert payload["adopted_generation_root"] == generation
    assert payload["M_effective"] == 20_000
    assert payload["source_postprocess_fragment_sha256"] == sha256_file(source)

    loaded = load_controller_manifest(manifest)
    assert loaded.controller_id == "bace-comrecgc-resource-cap-postprocess-test"
    assert loaded.by_id["bace_comrecgc_standardized"].read_only_test is True
    assert loaded.by_id["bace_comrecgc_train_common_recourse"].resource == "gpu"


def test_prepare_rejects_receipt_drift(
    tmp_path: Path, physical_root: Path,
) -> None:
    source = _source_fragment(physical_root)
    fragment = json.loads(source.read_text(encoding="utf-8"))
    receipt = Path(fragment["resource_cap_receipt"])
    _write_json(receipt, {"M_effective": 20_000, "test_loaded": True})

    with pytest.raises(ValueError, match="hash changed"):
        prepare(
            source_fragment=source,
            generic_fragment_output=tmp_path / "generic.json",
            manifest_output=tmp_path / "manifest.json",
            controller_id="receipt-drift",
        )


def test_prepare_rejects_out_of_budget_materialization(
    tmp_path: Path, physical_root: Path,
) -> None:
    source = _source_fragment(physical_root, effective=25_001)

    with pytest.raises(ValueError, match="authorized train-only budget"):
        prepare(
            source_fragment=source,
            generic_fragment_output=tmp_path / "generic.json",
            manifest_output=tmp_path / "manifest.json",
            controller_id="outside-budget",
        )
