from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.autodl_mut_next_stage_executor_v1 import build_successor_spec
from src.utils.autodl_mut_first_divergence_v1 import stable_sha256
from src.utils.autodl_mut_same_contract_ab_v1 import (
    MutSameContractABSpecError,
    validate_same_contract_ab_spec,
)
from src.utils.autodl_mut_successor_template_v1 import (
    MutSuccessorTemplateError,
    render_successor_template,
    template_placeholders,
    validate_exact_mut_successor_template,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = PROJECT_ROOT / "configs/autodl/mut_next_stage_executor_v1.template.json"


def _template() -> dict[str, object]:
    value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _bindings(tmp_path: Path) -> dict[str, str]:
    predecessor = tmp_path / "ab/task-spec.json"
    predecessor.parent.mkdir(parents=True)
    predecessor.write_text("{}\n", encoding="utf-8")
    control = tmp_path / "ab/control"
    output = tmp_path / "ab/output"
    return {
        "__ADOPTION_ROOT__": str(tmp_path / "successor/adoption"),
        "__EXECUTION_COMMIT__": "a" * 40,
        "__EXECUTOR_LEASE_PATH__": str(tmp_path / "successor/owner.lease"),
        "__EXECUTOR_RUNTIME_ROOT__": str(tmp_path / "successor"),
        "__EXPORT_ROOT__": str(tmp_path / "successor/export"),
        "__MATRIX_AUTHORITY_ROOT__": str(tmp_path / "control/fast16_matrix_authority"),
        "__MATRIX_OUTPUT_ROOT__": str(tmp_path / "paper_matrix"),
        "__NEXT_ACTION_PATH__": str(tmp_path / "continuation/next_action.json"),
        "__OWNER_REGISTRY__": str(
            tmp_path / "control/final16-owner-registry/current.json"
        ),
        "__PREDECESSOR_TASK_ID__": "mut_same_contract_ab",
        "__PREDECESSOR_TASK_SPEC__": str(predecessor),
        "__PREDECESSOR_TERMINAL__": str(control / "terminal.json"),
        "__PROJECT_ROOT__": str(tmp_path / "immutable-project"),
        "__PUBLISHER_ID__": "mut-successor-publisher",
        "__PUBLISHER_LEASE_PATH__": str(tmp_path / "publisher/owner.lease"),
        "__PUBLISHER_LOCATOR__": str(tmp_path / "publisher/locator.json"),
        "__PUBLISH_ROOT__": str(tmp_path / "successor/publish"),
        "__PYTHON__": str(tmp_path / "env/bin/python"),
        "__ROUTE_B_ROOT__": str(tmp_path / "successor/route-b"),
        "__STANDARDIZED_ROOT__": str(tmp_path / "successor/standardized"),
        "__TASK_ID__": "mut-post-ab-successor",
        "__TRACE_MODE_GATE__": str(
            output / "trace_on_off_500_step_equivalence.json"
        ),
    }


def test_exact_template_has_no_superseded_instrumentation_or_memory_gate(
    tmp_path: Path,
) -> None:
    rendered = render_successor_template(_template(), _bindings(tmp_path))
    validated = validate_exact_mut_successor_template(rendered)
    argv = validated["adoption_pipeline"][0]["argv"]
    joined = " ".join(argv)
    assert "run_mut_same_contract_adoption_v1.py" in joined
    assert "--same-contract-gate" in argv
    assert "--ab-task-spec" in argv
    assert "--ab-owner-terminal" in argv
    assert "--instrumentation-gate" not in argv
    assert "--memory-receipt" not in argv
    assert "--trace-code-audit" not in argv


def test_rendered_template_builds_exact_ordered_successor_spec(tmp_path: Path) -> None:
    rendered = validate_exact_mut_successor_template(
        render_successor_template(_template(), _bindings(tmp_path))
    )
    spec = build_successor_spec(rendered, check_files=False)
    assert [row["stage"] for row in spec["adoption_pipeline"]] == [
        "HISTORICAL_50K_ADOPTION",
        "STANDARDIZED_EVALUATION",
        "FIGURE_TABLE_EXPORT",
        "MATRIX_PUBLISH",
    ]
    assert spec["route_b_pipeline"][0]["expected_terminal_status"] == [
        "BLOCKED_ADAPTER_MISSING"
    ]


def test_renderer_rejects_missing_binding(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path)
    bindings.pop(next(iter(template_placeholders(_template()))))
    with pytest.raises(MutSuccessorTemplateError, match="missing"):
        render_successor_template(_template(), bindings)


def _deployed_pre_gpu_lock_spec() -> dict[str, object]:
    return {
        "schema_version": "mut_same_contract_trace_ab_task_spec_v1",
        "task_id": "mut-same-contract-ab-7c9a0159",
        "attempt_uuid": "7c9a0159-108d-4980-a002-fc4370be3da9",
        "controller_project_root": "/root/autodl-tmp/worktrees/final-main16-02c8e032",
        "controller_commit": "02c8e032593e19893f7562ae9b9a8aa7ea72c3f0",
        "python": "/root/miniconda3/envs/smiles_pip118/bin/python3.10",
        "runner_path": "/root/autodl-tmp/worktrees/final-main16-02c8e032/scripts/autodl/run_mut_trace_mode_equivalence.py",
        "legacy_project_root": "/root/autodl-tmp/worktrees/run-mut-comrecgc-algorithm-7f7ed51",
        "execution_project_root": "/root/autodl-tmp/worktrees/run-mut-comrecgc-checkpoint-66487c0",
        "historical_artifact_root": "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z",
        "upstream_root": "/root/autodl-tmp/worktrees/vendor-comrecgc-122f9341-owned-20260904",
        "dataset_dir": "/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset",
        "gnn_checkpoint": "/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/outputs/hpc/mutagenicity/baselines/gcfexplainer/full_v2/gnn/model_best.pth",
        "distance_checkpoint": "/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt",
        "rf_oracle": "/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl",
        "run_root": "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_same_contract_trace_ab_v1_20260904T052257Z",
        "output_dir": "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/mut_same_contract_trace_ab_v1_20260904T052257Z",
        "control_root": "/autodl-fs/data/counterfactual-subgraph-runtime/control/mut_same_contract_ab_v1_20260904T052257Z_7c9a0159",
        "lease_path": "/autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-dispatch-leases/mut-gpu0-same-contract-ab-7c9a0159.lock",
        "gpu_index": 0,
        "source_algorithm_commit": "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4",
        "instrumentation_commit": "66487c062c86d53ef2f762ce04d0fb965af5af08",
        "upstream_commit": "122f9341a360e9f06bb58a2f5823bb596021f6bf",
        "steps": 500,
        "post_reload_steps": 10,
        "candidate_capacity": 100000,
        "trace_modes": ["on", "off"],
        "arms_sequential": True,
        "resume_parity_separate": True,
        "fresh_50k_started": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "required_environment": {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RUN_LLM_ABLATION": "0",
            "RUN_GNN_ABLATION": "0",
        },
        "bound_file_sha256s": {
            "runner": "2cd783ec72527cf20078abdd44cbffba2cecbb34fcc3e0d7cc3bd4555655f07b",
            "historical_manifest": "807bc5a7649b3546eb4b7c0e127ce609df99aa6ff110dcf855170c4accf8d878",
            "gnn_checkpoint": "22045e5a6a833d6ed980cef9834859859136a1e2f644d19d78bd63345585f239",
            "distance_checkpoint": "bc64c16340c9170388ff1b3951d2ee4cb9a372456b09691ecd6bb2a881f17648",
            "rf_oracle": "af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204",
            "dataset/dataset_summary.json": "18c3f75201702b7699cb7b0553aed68d6867a1bfc631110a1e7f7dd832ef2b4b",
            "dataset/generation_source_graphs.pt": "d28c85b24f7ae50c164ed79c68c7b77e9a780bce789a0c51e581961b32950f4e",
        },
        "created_at": "2026-09-04T05:25:59.682267+00:00",
        "spec_sha256": "45a52717e01d10171fbb53120d60f640ad2f43b82358080bcdd1aa0dc4558a97",
    }


def test_exact_deployed_pre_gpu_lock_spec_is_the_only_legacy_shape() -> None:
    deployed = _deployed_pre_gpu_lock_spec()
    assert validate_same_contract_ab_spec(deployed, check_files=False) == deployed
    forged = dict(deployed)
    forged["task_id"] = "another-pre-gpu-lock-task"
    forged["spec_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "spec_sha256"}
    )
    with pytest.raises(MutSameContractABSpecError, match="unrecognized"):
        validate_same_contract_ab_spec(forged, check_files=False)
