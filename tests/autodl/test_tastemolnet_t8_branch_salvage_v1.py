import json
from pathlib import Path
from types import SimpleNamespace

import src.utils.tastemolnet_t8_branch_salvage_v1 as salvage
from src.utils.tastemolnet_t8_branch_salvage_v1 import (
    RERUN_SCHEMA,
    write_single_branch_rerun_request,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/autodl/salvage_tastemolnet_t8_branches_v1.py"
ORCHESTRATOR = ROOT / "scripts/autodl/run_tastemolnet_t8_salvage_release_v1.sh"
T13_RELAY = ROOT / "scripts/autodl/run_tastemolnet_t13_after_t8_salvage_v1.sh"
SLURM = ROOT / "scripts/slurm/salvage_tastemolnet_t8_branches_v1.sh"


def test_single_branch_failure_preserves_the_other_branch(tmp_path: Path) -> None:
    path = tmp_path / "rerun.json"
    receipt = write_single_branch_rerun_request(path, {2: "bad rule hash"})
    assert receipt["schema_version"] == RERUN_SCHEMA
    assert receipt["invalid_target_branches"] == [2]
    assert receipt["valid_target_branches_preserved"] == [0]
    assert receipt["rerun_both_branches"] is False
    assert receipt["source_artifacts_mutated"] is False
    assert json.loads(path.read_text(encoding="utf-8")) == receipt


def test_salvage_cli_exposes_only_read_only_source_roots() -> None:
    text = CLI.read_text(encoding="utf-8")
    for token in (
        "--target-0-root",
        "--target-2-root",
        "--state-root",
        "--output-root",
        "--rerun-request",
        "--device",
        "inference.fallback_to_heuristic=false",
    ):
        assert token in text


def test_native_materialization_keeps_target_provenance(monkeypatch) -> None:
    class FakeRule:
        @classmethod
        def from_payload(cls, row):
            return row

    monkeypatch.setattr(salvage, "GlobalGCENativeRule", FakeRule)
    monkeypatch.setattr(
        salvage,
        "apply_rule_to_parent",
        lambda smiles, rule: [
            {
                "valid": True,
                "canonical_smiles": f"{smiles}{rule['suffix']}",
            }
        ],
    )
    parents = [SimpleNamespace(parent_id="p1", smiles="CC")]
    records, audit = salvage.materialize_smoke_candidates(
        {0: [{"suffix": "N"}], 2: [{"suffix": "O"}]},
        parents=parents,
    )
    assert [row["raw_smiles"] for row in records[0]] == ["CCN"]
    assert [row["raw_smiles"] for row in records[2]] == ["CCO"]
    assert all(row["source_split"] == "train" for rows in records.values() for row in rows)
    assert audit["source_branch_mutated"] is False


def test_checkpoint_reload_is_restricted_and_branch_validator_is_scientific() -> None:
    text = (
        ROOT / "src/utils/tastemolnet_t8_branch_salvage_v1.py"
    ).read_text(encoding="utf-8")
    assert 'weights_only=True' in text
    assert 'weights_only=False' not in text
    for token in (
        "globalgce_model_checkpoint_sha256",
        "rules_checkpoint_sha256",
        "feature_schema_sha256",
        "gspan_exact_top_k_pruning",
        "source_train_cohort",
        "_writer_fds",
        "validate_globalgce_epoch_checkpoint_identity",
        "validate_candidates_with_original_gine",
    ):
        assert token in text


def test_salvage_adopts_then_persists_t13_relay() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "salvage_tastemolnet_t8_branches_v1.py" in text
    assert "adopt_tastemolnet_t8_deadline_v2.py" in text
    assert "launch_tastemolnet_t13_after_t8_salvage_v1.sh" in text
    assert "[TASTE_T8_SALVAGE_PASS]" in text
    assert "[TASTE_T8_GLOBALGCE_SMOKE_PASS]" in text
    assert "RUN_GNN_ABLATION" in text
    assert "run_tastemolnet_t8_deadline.py" not in text


def test_t13_relay_has_no_unrelated_taste_dependencies() -> None:
    text = T13_RELAY.read_text(encoding="utf-8")
    for forbidden in ("T10", "T11", "T12", "T14"):
        assert forbidden not in text
    assert "T8_PASS_ROOT" in text
    assert 'GPU_INDEX=${T13_GPU_INDEX:-1}' in text
    assert "adopt_tastemolnet_t8_deadline_v2.py" in text
    assert "--mode validate" in text
    assert "[TASTE_T13_GLOBALGCE_FULL_LAUNCHED]" in text
    assert "uuid.uuid4()" in text


def test_paired_slurm_keeps_hpc_baseline_and_science_flags() -> None:
    text = SLURM.read_text(encoding="utf-8")
    for token in (
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
    ):
        assert token in text
