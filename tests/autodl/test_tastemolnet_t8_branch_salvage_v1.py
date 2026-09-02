import json
from pathlib import Path
from types import SimpleNamespace

import src.utils.tastemolnet_t8_branch_salvage_v1 as salvage
from src.utils.tastemolnet_t8_branch_salvage_v1 import (
    RHS_CHEMISTRY_PREFLIGHT_SCHEMA,
    RERUN_SCHEMA,
    T8FinalizationError,
    preflight_rhs_rule_catalogs,
    read_single_branch_rerun_target,
    write_single_branch_rerun_request,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/autodl/salvage_tastemolnet_t8_branches_v1.py"
ORCHESTRATOR = ROOT / "scripts/autodl/run_tastemolnet_t8_salvage_release_v1.sh"
T13_RELAY = ROOT / "scripts/autodl/run_tastemolnet_t13_after_t8_salvage_v1.sh"
SLURM = ROOT / "scripts/slurm/salvage_tastemolnet_t8_branches_v1.sh"
RERUN_CLI = ROOT / "scripts/autodl/rerun_tastemolnet_t8_single_branch_v1.py"
RERUN_SLURM = ROOT / "scripts/slurm/rerun_tastemolnet_t8_single_branch_v1.sh"
SALVAGE_SOURCE = ROOT / "src/utils/tastemolnet_t8_branch_salvage_v1.py"
T13_SOURCE = ROOT / "src/baselines/tastemolnet_globalgce_full.py"


def _native_rule_payload(candidate_id: str, *, rhs_no_edge: bool = False) -> dict:
    return {
        "candidate_id": candidate_id,
        "native_rule_index": 0,
        "lhs_feature": [[0.0, 1.0], [0.0, 1.0]],
        "lhs_adjacency": [[0.0, 1.0], [1.0, 0.0]],
        "lhs_edge_attr": [[0.0, 1.0]],
        "rhs_feature": [[0.0, 1.0], [0.0, 1.0]],
        "rhs_adjacency": [[0.0, 1.0], [1.0, 0.0]],
        "rhs_edge_attr": [[1.0, 0.0] if rhs_no_edge else [0.0, 1.0]],
        "atom_symbols": ["C"],
        "bond_names": ["no_edge", "single"],
    }


def test_single_branch_failure_preserves_the_other_branch(tmp_path: Path) -> None:
    path = tmp_path / "rerun.json"
    receipt = write_single_branch_rerun_request(path, {2: "bad rule hash"})
    assert receipt["schema_version"] == RERUN_SCHEMA
    assert receipt["invalid_target_branches"] == [2]
    assert receipt["valid_target_branches_preserved"] == [0]
    assert receipt["rerun_both_branches"] is False
    assert receipt["source_artifacts_mutated"] is False
    assert json.loads(path.read_text(encoding="utf-8")) == receipt
    assert read_single_branch_rerun_target(path) == 2


def test_multi_branch_rerun_failure_is_typed(tmp_path: Path) -> None:
    path = tmp_path / "rerun.json"
    write_single_branch_rerun_request(path, {0: "no flip", 2: "no flip"})
    try:
        read_single_branch_rerun_target(path)
    except T8FinalizationError as exc:
        evidence = exc.to_dict()
    else:  # pragma: no cover - the typed failure is the contract under test
        raise RuntimeError("multi-branch rerun request unexpectedly passed")
    assert evidence == {
        "error_type": "T8FinalizationError",
        "code": "T8_RERUN_NOT_SINGLE_BRANCH",
        "field": "invalid_target_branches",
        "expected": "exactly one target branch",
        "actual": [0, 2],
        "source_manifest": RERUN_SCHEMA,
        "source_artifact": str(path),
        "stage": "T8_SINGLE_BRANCH_RERUN_SELECTION",
    }


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


def test_rhs_standalone_preflight_filters_inconsistent_rule(tmp_path: Path) -> None:
    artifact = tmp_path / "rhs-preflight.json"
    valid = _native_rule_payload("valid")
    invalid = _native_rule_payload("invalid", rhs_no_edge=True)
    approved, audit = preflight_rhs_rule_catalogs(
        {0: [invalid, valid], 2: [valid]},
        source_artifacts={
            0: tmp_path / "target0.jsonl",
            2: tmp_path / "target2.jsonl",
        },
        artifact_path=artifact,
    )
    assert approved == {0: [valid], 2: [valid]}
    assert audit["schema_version"] == RHS_CHEMISTRY_PREFLIGHT_SCHEMA
    assert audit["status"] == "PASS"
    assert audit["approved_rule_counts"] == {"0": 1, "2": 1}
    rejected = audit["rules"]["0"][0]
    assert rejected["candidate_id"] == "invalid"
    assert rejected["errors"][0]["code"] == "T8_RHS_BOND_NO_EDGE_MISMATCH"
    assert rejected["errors"][0]["field"] == "rhs_pair[0,1]"
    assert json.loads(artifact.read_text(encoding="utf-8")) == audit


def test_rhs_preflight_zero_usable_branch_is_typed_blocker(tmp_path: Path) -> None:
    artifact = tmp_path / "rhs-preflight.json"
    invalid = _native_rule_payload("invalid", rhs_no_edge=True)
    valid = _native_rule_payload("valid")
    try:
        preflight_rhs_rule_catalogs(
            {0: [invalid], 2: [valid]},
            source_artifacts={
                0: tmp_path / "target0.jsonl",
                2: tmp_path / "target2.jsonl",
            },
            artifact_path=artifact,
        )
    except T8FinalizationError as exc:
        evidence = exc.to_dict()
    else:  # pragma: no cover
        raise RuntimeError("RHS-empty target branch unexpectedly passed")
    assert evidence["code"] == "T8_RHS_PREFLIGHT_NO_USABLE_RULES"
    assert evidence["field"] == "branches.approved_rule_counts"
    assert evidence["actual"] == {"0": 0, "2": 1}
    audit = json.loads(artifact.read_text(encoding="utf-8"))
    assert audit["status"] == "BLOCKED"
    assert audit["invalid_target_branches"] == [0]
    assert audit["native_rule_application_started"] is False
    assert audit["gine_candidate_validation_started"] is False


def test_rhs_preflight_precedes_native_apply_and_gine() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    run_body = source[source.index("def run_salvage(") :]
    preflight = run_body.index("preflight_rhs_rule_catalogs(")
    scorer = run_body.index("FrozenTasteGINEScorer(")
    native_apply = run_body.index("materialize_smoke_candidates(")
    assert preflight < scorer < native_apply


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
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert 'gpu-$GPU_UUID.coordination.lock' in text
    assert "flock -n 8" in text
    assert "run_tastemolnet_t8_deadline.py" not in text


def test_salvage_has_one_bounded_target_specific_recovery() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "rerun_tastemolnet_t8_single_branch_v1.py" in text
    assert "read_single_branch_rerun_target" in text
    assert "assert " not in text
    assert "T8_SALVAGE_FAILED_NO_TARGET_SPECIFIC_RECOVERY" in text
    assert "T8_SALVAGE_FAILED_NON_SINGLE_BRANCH" in text
    assert "T8_SINGLE_BRANCH_RECOVERY_FAILED" in text
    assert "T8_BOUNDED_SINGLE_BRANCH_RECOVERY_FAILED" in text
    assert text.count("run_salvage_attempt") == 3  # declaration plus two calls
    assert "while true" in text  # GPU admission only, never a science retry loop
    recovery = RERUN_CLI.read_text(encoding="utf-8")
    assert 'choices=(0, 2)' in recovery
    assert "run_single_branch_recovery" in recovery
    assert "[TASTE_T8_SINGLE_BRANCH_RECOVERY_PASS]" in recovery


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
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert 'gpu-$GPU_UUID.coordination.lock' in text
    assert "flock -n 8" in text
    assert "initializer_mode=fresh_full_attempt" in text
    assert "t8_checkpoint_initializer_compatible=false" in text
    assert "K_MAX=20" in text
    assert "MIN_VALID_UNIQUE_RULES=10" in text


def test_salvage_emits_field_level_branch_and_smoke_audits() -> None:
    text = (
        ROOT / "src/utils/tastemolnet_t8_branch_salvage_v1.py"
    ).read_text(encoding="utf-8")
    for token in (
        "T8FinalizationError",
        '"field"',
        '"expected"',
        '"actual"',
        '"source_manifest"',
        '"source_artifact"',
        '"stage"',
        "target0_adoption_receipt.json",
        "target2_adoption_receipt.json",
        "branch_inventory.json",
        "merged_rules.json",
        "canonical_dedup.json",
        "strict_flip_smoke.json",
        "terminal.json",
        "final_audit.json",
        '"temperature_scaling_sha256"',
        '"test_loaded": False',
    ):
        assert token in text


def test_t8_assertions_are_typed() -> None:
    orchestrator = ORCHESTRATOR.read_text(encoding="utf-8")
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    assert "assert " not in orchestrator
    assert "T8FinalizationError" in source
    for field in (
        "field",
        "expected",
        "actual",
        "source_manifest",
        "source_artifact",
        "stage",
    ):
        assert f'"{field}"' in source


def test_t8_target0_salvage() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    assert '"target0_adoption_receipt.json"' in source
    assert '"target_label": target' in source
    assert "target not in TARGET_BRANCHES" in source


def test_t8_target2_salvage() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    assert '"target2_adoption_receipt.json"' in source
    assert '"target_branches": list(TARGET_BRANCHES)' in source


def test_t8_same_gine() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    for token in (
        '"oracle_checkpoint_hash"',
        '"oracle_identity_sha256"',
        '"temperature_hex"',
        '"temperature_scaling_sha256"',
        '"feature_schema_sha256"',
    ):
        assert token in source


def test_t8_merge_dedup() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    assert "merge_branch_rule_catalogs" in source
    assert "_deduplicate_generated_candidates" in source
    assert '"canonical_dedup.json"' in source


def test_t8_smoke_min_one_rule() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    assert 'rule_merge.get("merged_unique_rule_count", 0) < 1' in source
    assert 'expected=">=1"' in source


def test_t8_salvage_uses_merged_untargeted_flip_gate() -> None:
    source = SALVAGE_SOURCE.read_text(encoding="utf-8")
    assert "minimum_strict_flips_per_branch=0" in source
    assert "minimum_strict_flips_total=1" in source
    assert '"minimum_required_per_branch": 0' in source


def test_t13_full_min_ten_rules() -> None:
    source = T13_SOURCE.read_text(encoding="utf-8")
    relay = T13_RELAY.read_text(encoding="utf-8")
    assert "MIN_RULES = 10" in source
    assert "K_MAX = 20" in source
    assert "MIN_VALID_UNIQUE_RULES=10" in relay
    assert "K_MAX=20" in relay


def test_t13_initializer_compatibility() -> None:
    relay = T13_RELAY.read_text(encoding="utf-8")
    assert "initializer_mode=fresh_full_attempt" in relay
    assert "t8_checkpoint_initializer_compatible=false" in relay
    assert "smoke_exact_top_k_differs_from_full_contract" in relay


def test_t13_no_test_before_freeze() -> None:
    source = T13_SOURCE.read_text(encoding="utf-8")
    run_body = source[source.index("def run_t13_full(") :]
    freeze_position = run_body.index('phase="CALIBRATION_SELECTION_FROZEN"')
    test_position = run_body.index(
        "test_parents = authorize_and_load_test_after_freeze("
    )
    assert freeze_position < test_position


def test_paired_slurm_keeps_hpc_baseline_and_science_flags() -> None:
    for path in (SLURM, RERUN_SLURM):
        text = path.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --partition=A800",
            "#SBATCH --gres=gpu:a800:1",
            "#SBATCH --output=logs/%j.out",
            "#SBATCH --error=logs/%j.err",
            "source ~/.bashrc",
            "conda activate smiles_pip118",
            "cd /share/home/u20526/czx/counterfactual-subgraph",
            "export PYTHONPATH=$PWD",
            "export PYTHONDONTWRITEBYTECODE=1",
            "--config configs/hpc.yaml",
            "--set inference.fallback_to_heuristic=false",
        ):
            assert token in text
