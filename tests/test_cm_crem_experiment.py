"""CM stage integration fixtures; no real model, official DB or scientific PASS."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from src.baselines.cm_crem_experiment import validate_contract, science_identity, load_parent_rows, pilot_indices
from src.baselines.cm_crem_runtime import digest, file_sha, atomic_json, checked_root, storage_probe


def spec_fixture():
    spec = yaml.safe_load((Path(__file__).parents[1]/"configs/baselines/cm_crem_global_v1.yaml").read_text())
    spec["upstream"]["commit"] = "b5816b502cde00ee24c652a02cbc54664583f773"
    spec["resolved_oracle"] = {"backbone": "gine", "model_sha256": "a"*64}
    spec["resolved_wnode"] = {"feature_cost": "cosine", "numerical_contract_sha256": "b"*64}
    spec["resolved_parents"] = {"train": {"count": 386}, "calibration": {"count": 66}, "test": {"count": 141}}
    spec["resolved_evaluation"] = {"theta": 0.008413518173529859, "cap": 0.02956508038627219}
    return spec


def test_original_gine_contract_not_gin():
    spec = spec_fixture()
    validate_contract(spec)
    spec["resolved_oracle"]["backbone"] = "gin"
    with pytest.raises(ValueError, match="GIN/A"):
        validate_contract(spec)


@pytest.mark.parametrize("field,value", [("radius", 2), ("raw_outputs_retained_per_parent", 256), ("top_level_calls_per_parent", 2)])
def test_no_silent_budget_expansion(field, value):
    spec = spec_fixture()
    spec["generation"][field] = value
    with pytest.raises(ValueError):
        validate_contract(spec)


def test_scope_and_path_relocation():
    spec = spec_fixture()
    other = deepcopy(spec)
    other["resolved_oracle"]["checkpoint_dir"] = "/different/physical/copy"
    assert science_identity(spec) == science_identity(other)
    other["resolved_evaluation"]["theta"] = .02
    assert science_identity(spec) != science_identity(other)


def test_authoritative_order_join_and_test_barrier(tmp_path):
    path = tmp_path/"parents.csv"
    path.write_text("molecule_id,smiles\nb,CCC\na,CC\nx,CCCC\n")
    bind = {"path": str(path), "format": "csv", "id_field": "molecule_id", "count": 2,
            "sha256": file_sha(path), "ordered_ids": ["a", "b"], "ordered_ids_sha256": digest(["a", "b"])}
    spec = {"resolved_parents": {"train": bind, "calibration": bind, "test": bind}}
    assert [r["parent_id"] for r in load_parent_rows(spec, "train", tmp_path)] == ["a", "b"]
    with pytest.raises(ValueError, match="before calibration"):
        load_parent_rows(spec, "test", tmp_path)
    with pytest.raises(ValueError, match="before train"):
        load_parent_rows(spec, "calibration", tmp_path)


def test_parent_duplicates_not_silently_deduped(tmp_path):
    p = tmp_path/"rows.json"
    p.write_text(json.dumps([{"parent_id": "a", "smiles": "CC"}]*2))
    spec = {"resolved_parents": {"train": {"path": str(p), "count": 2, "sha256": file_sha(p),
             "ordered_ids_sha256": digest(["a", "a"])}}}
    with pytest.raises(ValueError, match="duplicate"):
        load_parent_rows(spec, "train", tmp_path)


def test_fixed_train_only_pilot_structure():
    pytest.importorskip("rdkit")
    rows = [{"parent_id": f"p{i:03d}", "smiles": "C"*(i+2)} for i in range(64)]
    chosen, receipt = pilot_indices(rows, "a"*64)
    assert len(chosen) == len(set(chosen)) == 32
    assert all(len(row["parent_ids"]) == 8 for row in receipt["quartiles"])
    assert chosen == pilot_indices(rows, "a"*64)[0]
    assert receipt["test_loaded"] is False


def test_small_input_is_not_full_pilot():
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError, match="32"):
        pilot_indices([{"parent_id": "p", "smiles": "CC"}], "a"*64)


def test_immutable_receipt_rejects_rewrite(tmp_path):
    p = tmp_path/"sealed.json"
    atomic_json(p, {"x": 1}, immutable=True)
    atomic_json(p, {"x": 1}, immutable=True)
    with pytest.raises(ValueError, match="immutable"):
        atomic_json(p, {"x": 2}, immutable=True)
    assert json.loads(p.read_text()) == {"x": 1}


def test_allowroot_no_symlink_escape(tmp_path):
    (tmp_path/"link").symlink_to("/tmp")
    with pytest.raises(ValueError):
        checked_root(tmp_path/"link"/"x", tmp_path)
    with pytest.raises(ValueError):
        checked_root(tmp_path, tmp_path)


def test_bounded_real_io_fixture(tmp_path):
    result = storage_probe(tmp_path/"io", tmp_path)
    assert result["status"] == "PASS_BOUNDED_IO"
    assert result["bytes_checked"] == 65536


def test_cpu_slurm_matches_cli_and_no_gpu_request():
    repo = Path(__file__).parents[1]
    script = (repo/"scripts/slurm/run_cm_crem.sh").read_text()
    assert "#SBATCH --partition=intel" in script
    assert "#SBATCH --gres" not in script
    assert "--config configs/hpc.yaml" in script
    assert "-I -B scripts/run_cm_crem.py" in script
    assert "-s -B scripts/run_cm_crem.py" in script
    assert "unset PYTHONPATH" in script
    assert script.index("source ~/.bashrc") < script.index("set -euo pipefail")


def test_filter_reuses_sealed_pilot_units_and_measures_model_io(tmp_path, monkeypatch):
    from src.baselines import cm_crem_experiment as module
    monkeypatch.setattr(module, "require_compute_node", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    experiment = object.__new__(module.Experiment)
    experiment.root, experiment.sha = tmp_path, "fixture-science"
    experiment.spec = {"execution": {"execution_commit": "fixture"}}
    calls = []
    class Oracle:
        def filter_generated(self, parent, generation):
            calls.append(parent["parent_id"])
            return {"parent_id": parent["parent_id"], "accepted": [], "retained_raw_count": 0}
    experiment.oracle = lambda: Oracle()
    parents = [{"parent_id": "public-fixture-parent", "smiles": "CC"}]
    experiment.put("pilot/oracle.json", {"parents": parents})
    experiment.put("attribution.json", {"parents": parents})
    experiment.put(f"generation_units/{digest(parents[0]['parent_id'])[:20]}.json", {"status": "NO_NATIVE_REPLACEMENT"})
    experiment.stage_filter(pilot_only=True)
    assert calls == ["public-fixture-parent"]
    experiment.stage_filter(pilot_only=False)
    assert calls == ["public-fixture-parent"]
    assert experiment.get("filter_timing.json")["reused_sealed_parent_units"] == 1
    assert experiment.get("pilot/filter_timing.json")["total_filter_and_durable_io_seconds"] > 0
