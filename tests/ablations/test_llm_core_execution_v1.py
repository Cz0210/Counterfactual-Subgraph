from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest
import yaml

from src.ablations.llm.contracts import LLMAblationContractError, canonical_json_sha256
from src.ablations.llm.core_execution import (
    CORE_VARIANT_ORDER,
    SFT_AUXILIARY_REASON,
    SFT_AUXILIARY_STATE,
    CoreLLMVariant,
    CoreRunSpec,
    derive_core_reference,
    run_core_variant,
)
from src.ablations.llm.early_launch_gate import EarlyLaunchSnapshot
from src.ablations.llm.runtime_evidence import BACEReferenceEvidence
from src.ablations.llm.runtime_evidence import validate_off_the_shelf_7b_parameter_report


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


def _write(path: Path, value: object) -> dict[str, str]:
    path.write_text(
        value if isinstance(value, str) else json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _reference() -> BACEReferenceEvidence:
    return BACEReferenceEvidence(
        path="/runtime/reference.json",
        file_sha256="b" * 64,
        self_sha256="c" * 64,
        payload={
            "main_policy_scientific_name": (
                "CHEMLLM_7B_OFF_THE_SHELF_PLUS_FRESH_LORA_PPO"
            ),
            "ppo": {"optimizer_updates": 300},
            "stage_variants": {
                "A2_CHEMLLM_7B_PROJECT_SFT": {
                    "status": "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
                }
            },
        },
    )


def _stage_command() -> list[str]:
    code = r"""
import json, os
from pathlib import Path
root = Path(os.environ['LLM_ABLATION_OUTPUT_ROOT'])
stage = os.environ['LLM_ABLATION_STAGE']
names = {
 'candidate_pool': 'candidate_pool.jsonl',
 'common_verification': 'verification_manifest.json',
 'selector_freeze': 'selector_manifest.json',
 'heldout_test': 'heldout_test_metrics.json',
 'final_audit': 'final_audit.json',
}
payload = {
 'common_verification': {'status': 'PASS'},
 'selector_freeze': {'status': 'PASS', 'selection_frozen': True, 'test_loaded': False},
 'heldout_test': {'status': 'PASS', 'selection_frozen_before_test': True},
 'final_audit': {'status': 'PASS'},
}.get(stage)
path = root / names[stage]
if payload is None:
 path.write_text('{"candidate":true}\n')
else:
 path.write_text(json.dumps(payload) + '\n')
with (root / 'executed.txt').open('a') as handle:
 handle.write(stage + '\n')
"""
    return [sys.executable, "-c", code]


def _spec_payload(tmp_path: Path, *, variant: CoreLLMVariant) -> dict:
    run_id = str(uuid4())
    reference = _write(tmp_path / "reference.json", {"status": "PASS"})
    matrix = _write(tmp_path / "matrix.json", {"complete": 13})
    stages = []
    for name in (
        "candidate_pool",
        "common_verification",
        "selector_freeze",
        "heldout_test",
        "final_audit",
    ):
        stages.append(
            {
                "name": name,
                "action": "EXECUTE",
                "argv": _stage_command(),
                "adopted_artifacts": [],
            }
        )
    topology = "BASE_ONLY_NO_ADAPTER"
    if variant is CoreLLMVariant.BRICS_FIXED:
        source = _write(tmp_path / "brics-pool.jsonl", "{}\n")
        stages[0] = {
            "name": "candidate_pool",
            "action": "ADOPT",
            "argv": [],
            "adopted_artifacts": [source],
        }
        topology = "NO_MODEL"
    elif variant is CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN:
        topology = "BASE_PLUS_PPO_LORA_ADOPT_MAIN"
        for index, stage in enumerate(stages):
            source = _write(tmp_path / f"main-{index}.json", {"status": "PASS"})
            stage.update(action="ADOPT", argv=[], adopted_artifacts=[source])
    payload = {
        "schema_version": "llm_core_variant_run_spec_v1",
        "run_id": run_id,
        "variant": variant.value,
        "output_root": str((tmp_path / f"output-{run_id}").resolve()),
        "execution_commit": COMMIT,
        "reference_contract": reference,
        "matrix_authority": matrix,
        "adapter_topology": topology,
        "checkpoint_resume_supported": True,
        "stages": stages,
    }
    payload["run_spec_sha256"] = canonical_json_sha256(payload)
    return payload


def _snapshot(*, waiting: bool) -> EarlyLaunchSnapshot:
    return EarlyLaunchSnapshot(
        matrix_complete_cells=13,
        matrix_authority_path="/runtime/matrix.json",
        matrix_authority_sha256="a" * 64,
        t8_t13_state="RUNNING",
        t8_t13_science_pid=1,
        t12_healthy=True,
        t14_healthy=True,
        mut_passed_or_gpu_released=True,
        main_ready_waiting_gpu=("main",) if waiting else (),
        main_publishers_waiting_gpu=(),
        idle_gpu=0,
        idle_gpu_seconds=1200,
        persistent_free_gb=200,
        minimum_persistent_free_gb=100,
        memory_available_gb=128,
        minimum_memory_available_gb=64,
        checkpoint_resume_supported=True,
        requested_early_gpus=1,
        main_owner_registry_path="/runtime/final16-owner-registry.json",
        main_owner_registry_sha256="d" * 64,
        main_owner_registry_self_sha256="e" * 64,
        all_incomplete_main_cells_owned=True,
        unhealthy_or_unowned_main_cells=(),
        missing_main_publisher_cells=(),
        active_early_llm_ablation_gpus=(),
    )


def test_reference_truthfully_exposes_base_plus_ppo_and_no_sft() -> None:
    core = derive_core_reference(_reference())
    assert core["main_adaptation_path"] == "BASE_PLUS_PPO_LORA"
    assert core["project_sft_checkpoint_exists"] is False
    assert list(core["variants"]) == list(CORE_VARIANT_ORDER)
    assert core["sft_auxiliary"] == {
        "enabled": False,
        "state": "N/A",
        "reason": SFT_AUXILIARY_REASON,
    }


def test_core_variant_names_are_exact_and_legacy_sft_rows_are_absent() -> None:
    assert CORE_VARIANT_ORDER == (
        "BRICS_FIXED",
        "CHEMLLM_7B_OFF_THE_SHELF",
        "CHEMLLM_7B_PPO_LORA_MAIN",
        "CHEMLLM_2B_OFF_THE_SHELF",
    )
    assert "CHEMLLM_7B_PPO_MAIN" not in CORE_VARIANT_ORDER
    assert all("SFT" not in variant for variant in CORE_VARIANT_ORDER)


def test_core_config_has_only_four_real_rows_and_sft_auxiliary_disabled() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs/ablations/llm/bace_ours_core_ablation_v1.yaml")
        .read_text(encoding="utf-8")
    )
    assert [row["variant"] for row in config["variants"]] == list(CORE_VARIANT_ORDER)
    assert config["project_sft_checkpoint_exists"] is False
    assert config["sft_auxiliary"] == {
        "enabled": False,
        "state": SFT_AUXILIARY_STATE,
        "reason": SFT_AUXILIARY_REASON,
        "activation_requires": "NEW_USER_AUTHORIZATION_AND_MATCHED_TRAIN_ONLY_SFT_MANIFEST",
    }
    assert config["entrypoints"]["launcher_slurm"] == (
        "scripts/slurm/launch_llm_ablation_core_v1.sh"
    )


def test_core_entrypoints_and_required_slurm_pairs_are_real() -> None:
    files = (
        "scripts/autodl/run_llm_ablation_variant.py",
        "scripts/autodl/status_llm_ablation_core_v1.py",
        "scripts/autodl/launch_llm_ablation_core_v1.sh",
        "scripts/ablations/llm/audit_chemllm_2b_isolated_load.py",
        "scripts/slurm/run_llm_ablation_variant.sh",
        "scripts/slurm/status_llm_ablation_core_v1.sh",
        "scripts/slurm/launch_llm_ablation_core_v1.sh",
        "scripts/slurm/audit_chemllm_2b_isolated_load.sh",
    )
    for name in files:
        assert (REPO_ROOT / name).is_file()
    launcher = (REPO_ROOT / files[2]).read_text(encoding="utf-8")
    assert "run_llm_ablation_variant.py" in launcher
    assert 'exec "${run_args[@]}"' in launcher
    assert "BLOCKED_CONFIG_ONLY_NO_SCIENCE_ENTRYPOINT" not in launcher
    runner = (REPO_ROOT / files[0]).read_text(encoding="utf-8")
    assert "run_core_variant(" in runner
    isolated = (REPO_ROOT / files[3]).read_text(encoding="utf-8")
    assert 'choices=("metadata", "cpu-load")' in isolated
    for name in files[4:]:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "--config configs/hpc.yaml" in text
        assert "export PYTHONPATH=$PWD" in text


def test_real_runner_checkpoints_and_resume_is_idempotent(tmp_path: Path) -> None:
    spec = CoreRunSpec.from_mapping(
        _spec_payload(tmp_path, variant=CoreLLMVariant.CHEMLLM_7B_OFF_THE_SHELF)
    )
    result = run_core_variant(spec, resume=False, live_snapshot_loader=lambda: _snapshot(waiting=False))
    assert result["state"] == "PASS"
    assert result["completed_stages"] == [
        "candidate_pool",
        "common_verification",
        "selector_freeze",
        "heldout_test",
        "final_audit",
    ]
    assert (Path(spec.output_root) / "executed.txt").read_text().count("\n") == 5
    manifest = json.loads((Path(spec.output_root) / "run_manifest.json").read_text())
    assert manifest["project_sft_checkpoint_exists"] is False
    assert manifest["selector_frozen_before_heldout_test"] is True
    resumed = run_core_variant(spec, resume=True, live_snapshot_loader=lambda: _snapshot(waiting=False))
    assert resumed["state"] == "PASS"
    assert (Path(spec.output_root) / "executed.txt").read_text().count("\n") == 5


def test_main_priority_pauses_at_safe_boundary_then_resumes(tmp_path: Path) -> None:
    spec = CoreRunSpec.from_mapping(
        _spec_payload(tmp_path, variant=CoreLLMVariant.CHEMLLM_7B_OFF_THE_SHELF)
    )
    paused = run_core_variant(spec, resume=False, live_snapshot_loader=lambda: _snapshot(waiting=True))
    assert paused["state"] == "PAUSED_MAIN_PRIORITY"
    assert paused["completed_stages"] == []
    passed = run_core_variant(spec, resume=True, live_snapshot_loader=lambda: _snapshot(waiting=False))
    assert passed["state"] == "PASS"


def test_main_ppo_row_can_only_adopt_existing_artifacts(tmp_path: Path) -> None:
    payload = _spec_payload(
        tmp_path, variant=CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN
    )
    CoreRunSpec.from_mapping(payload)
    payload["stages"][0].update(action="EXECUTE", argv=_stage_command(), adopted_artifacts=[])
    payload["run_spec_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "run_spec_sha256"}
    )
    with pytest.raises(LLMAblationContractError, match="adopted without retraining"):
        CoreRunSpec.from_mapping(payload)


def test_main_ppo_lora_row_adopts_without_executing_or_retraining(tmp_path: Path) -> None:
    spec = CoreRunSpec.from_mapping(
        _spec_payload(
            tmp_path, variant=CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN
        )
    )
    result = run_core_variant(spec, resume=False)
    assert result["state"] == "PASS"
    assert not (Path(spec.output_root) / "executed.txt").exists()
    manifest = json.loads((Path(spec.output_root) / "run_manifest.json").read_text())
    assert manifest["science_retrained"] is False
    assert manifest["main_result_adopted"] is True
    assert manifest["main_result_retraining_permitted"] is False
    assert manifest["sft_auxiliary_state"] == "N/A"


def test_selector_must_freeze_before_test(tmp_path: Path) -> None:
    payload = _spec_payload(tmp_path, variant=CoreLLMVariant.CHEMLLM_7B_OFF_THE_SHELF)
    bad = """
import json, os
from pathlib import Path
root=Path(os.environ['LLM_ABLATION_OUTPUT_ROOT'])
stage=os.environ['LLM_ABLATION_STAGE']
if stage == 'candidate_pool': (root/'candidate_pool.jsonl').write_text('{}\\n')
elif stage == 'common_verification': (root/'verification_manifest.json').write_text('{\"status\":\"PASS\"}')
else: (root/'selector_manifest.json').write_text(json.dumps({'selection_frozen':False,'test_loaded':False}))
"""
    payload["stages"][2]["argv"] = [sys.executable, "-c", bad]
    payload["run_spec_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "run_spec_sha256"}
    )
    spec = CoreRunSpec.from_mapping(payload)
    with pytest.raises(LLMAblationContractError, match="freeze before held-out"):
        run_core_variant(spec, resume=False, live_snapshot_loader=lambda: _snapshot(waiting=False))


def test_7b_off_the_shelf_report_rejects_main_ppo_adapter(tmp_path: Path) -> None:
    report = {
        "schema_version": "actual_parameter_count_report_v1",
        "source": "ACTUAL_LOADED_WEIGHTS",
        "total_parameters": 7_737_708_544,
        "trainable_parameters": 0,
        "embedding_parameters": 758_120_448,
        "non_embedding_parameters": 6_979_588_096,
        "lora_trainable_parameters": 0,
        "trainable_fraction": 0.0,
        "dtype": ["torch.bfloat16"],
        "weight_bytes": 15_475_442_896,
        "config_hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 92544,
        "adapter_loaded": False,
    }
    report["parameter_report_sha256"] = canonical_json_sha256(report)
    identity = _write(tmp_path / "7b-ots.json", report)
    validate_off_the_shelf_7b_parameter_report(identity["path"], identity["sha256"])

    report["adapter_loaded"] = True
    report["parameter_report_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "parameter_report_sha256"}
    )
    identity = _write(tmp_path / "7b-with-adapter.json", report)
    with pytest.raises(LLMAblationContractError, match="loaded an adapter"):
        validate_off_the_shelf_7b_parameter_report(identity["path"], identity["sha256"])
