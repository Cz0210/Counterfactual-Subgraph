from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.autodl import bace_frozen_gnn_route as route
from scripts.autodl.exp_run import (
    SCIENTIFIC_BLOCKED_EXIT_CODE,
    _validate_result_contract,
    run_worker,
)
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.models.molecular_gnn import build_molecular_gnn
from src.oracles.gnn_oracle import save_gnn_checkpoint_bundle, sha256_file
from src.oracles.oracle_factory import build_oracle
from src.utils.autodl_runtime import (
    build_runtime_layout,
    initialize_bace_stage_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_calibrated_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(7)
    schema = default_molecular_feature_schema()
    model = build_molecular_gnn(
        backbone="gine",
        num_classes=2,
        node_feature_schema=schema,
        edge_feature_schema=schema,
        num_layers=2,
        hidden_dim=16,
        dropout=0.0,
        pooling="mean",
        readout_layers=2,
        normalization="layer_norm",
        residual=True,
    )
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([-1.0, 1.0]))
    checkpoint = tmp_path / "b4"
    save_gnn_checkpoint_bundle(
        model=model,
        checkpoint_dir=checkpoint,
        feature_schema=schema,
        config={"gnn": model.config.to_dict()},
        model_card={
            "dataset": "bace",
            "source_label": 1,
            "seed": 7,
            "training_commit": "unit",
            "best_epoch": 1,
            "selection_metric": "roc_auc",
        },
        label_map={0: "Inactive", 1: "Active"},
        split_manifest={},
        training_metrics={"best_epoch": 1},
        test_evaluation_status={
            "status": "NOT_EVALUATED",
            "test_loaded": False,
            "reason": "held_out_until_frozen_final_evaluation",
            "path": "/frozen/test.csv",
            "sha256": "a" * 64,
        },
        temperature_scaling={
            "status": "fit",
            "selection_split": "validation",
            "test_used_for_fit": False,
            "argmax_invariant": True,
            "temperature": 1.5,
        },
        environment={"python": "unit"},
        git_state={"commit": "unit"},
    )
    return checkpoint


def test_b6_runs_real_batched_gnn_scoring_without_claiming_ppo(
    tmp_path: Path,
) -> None:
    checkpoint = _build_calibrated_checkpoint(tmp_path)
    card = json.loads((checkpoint / "model_card.json").read_text(encoding="utf-8"))
    schema = default_molecular_feature_schema()
    featurizer = route.MolecularGraphFeaturizer(schema)
    graphs = [
        route._graph_from_smiles(
            featurizer, smiles="CCO", molecule_id="parent"
        ),
        route._graph_from_smiles(
            featurizer, smiles="CC", molecule_id="residual"
        ),
    ]
    oracle = build_oracle(
        dataset="bace", backend="gnn", checkpoint=checkpoint, device="cpu"
    )
    predictions = oracle.predict_records(graphs)
    before, after = predictions
    cf_drop = float(before["source_probability"] - after["source_probability"])
    b5 = tmp_path / "b5"
    b5.mkdir()
    (b5 / "oracle_smoke.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "evaluation_split": "calibration",
                "test_loaded": False,
                "rf_guard_pass": True,
                "selected_count": 16,
                "checkpoint_id": card["checkpoint_id"],
                "checkpoint_dir": str(checkpoint),
                "checkpoint_sha256sums_sha256": sha256_file(
                    checkpoint / "sha256sums.txt"
                ),
            }
        ),
        encoding="utf-8",
    )
    (b5 / "deletion_records.jsonl").write_text(
        json.dumps(
            {
                "parent_id": "calibration-parent",
                "parent_smiles": "CCO",
                "source_label": 1,
                "fragment_smiles": "O",
                "residual_smiles": "CC",
                "residual_connected": True,
                "sanitize_ok": True,
                "pred_before": before["predicted_label"],
                "pred_after": after["predicted_label"],
                "probabilities_before": before["probabilities"],
                "probabilities_after": after["probabilities"],
                "cf_drop": cf_drop,
                "cf_flip": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "b6"
    assert route.main(
        [
            "scoring-preflight",
            "--checkpoint-dir",
            str(checkpoint),
            "--oracle-smoke-dir",
            str(b5),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--max-records",
            "1",
        ]
    ) == SCIENTIFIC_BLOCKED_EXIT_CODE
    summary = json.loads(
        (output / "b6_scoring_preflight.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "BLOCKED"
    assert summary["diagnostic_status"] == "PASS"
    assert summary["stage_gate_status"] == "BLOCKED"
    assert summary["blocker_code"] == "BLOCKED_MISSING_GNN_PPO_INTEGRATION"
    assert summary["secondary_blockers"] == [
        "BLOCKED_NO_GNN_CLEAN_BACE_POLICY_INITIALIZATION"
    ]
    assert summary["execution_mode"] == "gnn_scoring_preflight_not_ppo"
    assert summary["ppo_training_performed"] is False
    assert summary["ppo_pass_claimed"] is False
    assert summary["downstream_release_authorized"] is False
    assert summary["next_stage_launch_allowed"] is False
    assert summary["oracle_load_count"] == 1
    assert summary["test_loaded"] is False
    scored = json.loads(
        (output / "scored_candidates.jsonl").read_text(encoding="utf-8")
    )
    assert scored["oracle_backend"] == "gnn"
    assert scored["rf_oracle_used"] is False
    assert scored["ppo_checkpoint_hash"] is None
    assert np.isfinite(scored["diagnostic_score"])
    audit = json.loads(
        (output / "legacy_route_audit.json").read_text(encoding="utf-8")
    )
    classifications = {
        row["component"]: row["classification"] for row in audit["components"]
    }
    assert classifications["ChemLLM base proposer"] == "ORACLE_NEUTRAL"
    assert classifications["stable PPO reward"] == "RF_CONTAMINATED"
    assert classifications["candidate generation and scoring"] == "RF_CONTAMINATED"
    assert audit["historical_artifacts_promotable"] is False
    blocker = json.loads((output / "blocker.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "BLOCKED"
    assert blocker["ppo_update_count"] == 0
    assert blocker["policy_initialization_audit"] == {
        "historical_bace_ppo": "RF_CONTAMINATED_DIAGNOSIS_ONLY",
        "unknown_provenance_lora": "FORBIDDEN",
        "chemllm_base": "ORACLE_NEUTRAL_BUT_CURRENT_PPO_ENTRY_REQUIRES_LORA",
        "safe_reusable_policy_found": False,
    }
    requirements = json.loads(
        (output / "stage_requirements.json").read_text(encoding="utf-8")
    )
    assert requirements["required_to_pass"]["minimum_ppo_updates"] == 1
    assert requirements["policy_initialization_constraints"][
        "unknown_provenance_lora"
    ] == "REJECT"
    assert requirements["current_preflight_satisfies_stage"] is False


def test_b7_publishes_exact_blocker_and_never_promotes_legacy_ppo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "b4"
    checkpoint.mkdir()
    (checkpoint / "sha256sums.txt").write_text("checkpoint\n", encoding="utf-8")
    card = {
        "dataset": "bace",
        "num_classes": 2,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "backbone": "gine",
        "checkpoint_id": "gnn-checkpoint",
    }
    monkeypatch.setattr(
        route,
        "_validate_bace_checkpoint",
        lambda _path, **_kwargs: card,
    )
    predecessor = tmp_path / "b6"
    predecessor.mkdir()
    (predecessor / "ppo_smoke_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "ppo_training_performed": True,
                "ppo_update_count": 1,
                "reward_oracle_backend": "gnn",
            }
        ),
        encoding="utf-8",
    )
    (predecessor / "oracle_provenance.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "b7"
    assert route.main(
        [
            "stage-blocker",
            "--stage",
            "B7_PPO_FULL",
            "--checkpoint-dir",
            str(checkpoint),
            "--predecessor-output",
            str(predecessor),
            "--output-dir",
            str(output),
        ]
    ) == SCIENTIFIC_BLOCKED_EXIT_CODE
    blocker = json.loads((output / "blocker.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "BLOCKED"
    assert blocker["blocker_code"] == "BLOCKED_MISSING_GNN_PPO_INTEGRATION"
    assert blocker["legacy_artifact_classification"] == "RF_CONTAMINATED"
    assert blocker["legacy_reuse_allowed"] is False
    assert "CounterfactualTeacherScorer" in blocker["legacy_reason"]


def test_every_b7_b14_stage_has_concrete_fail_closed_contract() -> None:
    assert set(route.STAGE_BLOCKERS) == set(route.BLOCKED_STAGES)
    assert set(route.PREDECESSOR_OUTPUT_CONTRACT) == set(route.BLOCKED_STAGES)
    codes = []
    for stage in route.BLOCKED_STAGES:
        spec = route.STAGE_BLOCKERS[stage]
        codes.append(spec["code"])
        assert spec["missing_interface"]
        assert spec["legacy_entrypoints"]
        assert spec["legacy_reason"]
        assert spec["required_next_outputs"]
    assert len(codes) == len(set(codes))


def test_blocked_exit_is_registered_only_with_complete_evidence(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text("[BACE_GNN_STAGE_BLOCKED]\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "blocker.json").write_text("{}\n", encoding="utf-8")
    spec = {
        "log_path": str(log),
        "required_log_marker": "[BACE_GNN_STAGE_BLOCKED]",
        "expected_output": str(output),
        "required_output_files": ["blocker.json"],
    }
    assert _validate_result_contract(
        spec,
        exit_code=SCIENTIFIC_BLOCKED_EXIT_CODE,
        allow_scientific_blocker=True,
    ) == []
    failures = _validate_result_contract(
        spec,
        exit_code=SCIENTIFIC_BLOCKED_EXIT_CODE,
        allow_scientific_blocker=False,
    )
    assert failures == [f"scientific command exited {SCIENTIFIC_BLOCKED_EXIT_CODE}"]
    (output / "blocker.json").unlink()
    assert _validate_result_contract(
        spec,
        exit_code=SCIENTIFIC_BLOCKED_EXIT_CODE,
        allow_scientific_blocker=True,
    )
    incomplete = {
        **spec,
        "required_log_marker": None,
        "expected_output": None,
        "required_output_files": [],
    }
    incomplete_failures = _validate_result_contract(
        incomplete,
        exit_code=SCIENTIFIC_BLOCKED_EXIT_CODE,
        allow_scientific_blocker=True,
    )
    assert "scientific blocker requires a nonempty log marker" in incomplete_failures
    assert (
        "scientific blocker requires an expected output and evidence files"
        in incomplete_failures
    )


def test_worker_records_complete_scientific_blocker_as_state_and_gate_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    monkeypatch.delenv("AUTODL_CONTROL_ROOT", raising=False)
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    initialize_bace_stage_tree(layout)
    output = layout.artifacts_dir / "bace" / "b6-blocked"
    helper = tmp_path / "blocked_child.py"
    helper.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "root = Path(sys.argv[1])\n"
        "root.mkdir(parents=True)\n"
        "(root / 'blocker.json').write_text('{}\\n', encoding='utf-8')\n"
        "print('[BACE_GNN_STAGE_BLOCKED]', flush=True)\n"
        "raise SystemExit(78)\n",
        encoding="utf-8",
    )
    run_root = layout.runs_root / "unit-b6-blocked"
    run_root.mkdir(parents=True)
    log_path = layout.logs_dir / "unit-b6-blocked.log"
    spec = {
        "schema_version": 1,
        "run_id": "unit-b6-blocked",
        "project_root": str(project),
        "data_root": str(data),
        "control_root": str(layout.control_root),
        "python_executable": str(Path(sys.executable).resolve()),
        "dataset": "bace",
        "stage": "B6_PPO_SMOKE",
        "command": [sys.executable, str(helper), str(output)],
        "environment": {},
        "gpu_index": None,
        "gpu_uuid": None,
        "max_gpus": 2,
        "expected_output": str(output),
        "required_output_files": ["blocker.json"],
        "required_log_marker": "[BACE_GNN_STAGE_BLOCKED]",
        "log_path": str(log_path),
        "tmux_session": None,
        "git_commit": "unit",
        "config_hash": None,
        "input_hash": None,
    }
    spec_path = run_root / "launch_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert run_worker(spec_path) == SCIENTIFIC_BLOCKED_EXIT_CODE
    state = json.loads(
        (layout.stages_root / "B6_PPO_SMOKE" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    gate = json.loads(
        (layout.stages_root / "B6_PPO_SMOKE" / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["state"] == "BLOCKED"
    assert state["exit_code"] == SCIENTIFIC_BLOCKED_EXIT_CODE
    assert gate["status"] == "BLOCKED"
    assert gate["failures"] == []


def test_autodl_route_wrapper_and_paired_slurm_are_explicit() -> None:
    wrapper = PROJECT_ROOT / "scripts/autodl/run_bace_frozen_gnn_stage.sh"
    paired = PROJECT_ROOT / "scripts/slurm/bace_frozen_gnn_route.sh"
    driver = PROJECT_ROOT / "scripts/autodl/bace_frozen_gnn_route.py"
    assert wrapper.is_file()
    assert paired.is_file()
    wrapper_text = wrapper.read_text(encoding="utf-8")
    paired_text = paired.read_text(encoding="utf-8")
    driver_text = driver.read_text(encoding="utf-8")
    for stage in (route.B6_STAGE, *route.BLOCKED_STAGES):
        assert stage in wrapper_text
    assert "[BACE_GNN_SCORING_PREFLIGHT_PASS_NOT_PPO]" not in wrapper_text
    assert "[BACE_GNN_SCORING_PREFLIGHT_PASS_NOT_PPO]" in driver_text
    assert "[BACE_GNN_STAGE_BLOCKED]" in wrapper_text
    assert "sbatch" not in wrapper_text.lower()
    assert "#SBATCH --partition=A800" in paired_text
    assert "#SBATCH --gres=gpu:a800:1" in paired_text
    assert "#SBATCH --output=logs/%j.out" in paired_text
    assert "#SBATCH --error=logs/%j.err" in paired_text
    assert "source ~/.bashrc" in paired_text
    assert "conda activate smiles_pip118" in paired_text
    assert "cd /share/home/u20526/czx/counterfactual-subgraph" in paired_text
    assert "export PYTHONPATH=$PWD" in paired_text
    assert "--config configs/hpc.yaml" in paired_text
