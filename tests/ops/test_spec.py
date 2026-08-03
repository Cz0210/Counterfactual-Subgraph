from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.ops.spec import SpecValidationError, load_task_spec


ROOT = Path(__file__).resolve().parents[2]


def test_example_and_clear_specs_validate() -> None:
    example = load_task_spec(ROOT / "ops/specs/example_smoke.yaml")
    clear = load_task_spec(ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml")
    assert example.task_id == "example_smoke"
    assert clear.data["permissions"]["allow_test"] is False
    assert clear.data["permissions"]["allow_calibration"] is False
    assert clear.data["permissions"]["allow_full"] is False
    assert clear.data["execution"]["stop_before"] == "phase_b_gpu_smoke"
    dirty = clear.data["remote_dirty_policy"]
    assert dirty["allowed_tracked_paths"] == [
        "baselines/clear_official",
        "docs/EXPERIMENT_LOG.md",
    ]
    assert dirty["allowed_untracked_paths"] == [
        "scripts/paper/plot_aids_mut_gcf_style.py"
    ]
    patched = dirty["allowed_patched_submodules"][0]
    assert patched["path"] == "baselines/clear_official"
    assert patched["allowed_modified_paths"] == [
        "src/data_preprocessing.py",
        "src/main.py",
        "src/models.py",
        "src/train_pred.py",
    ]
    assert patched["allowed_untracked_paths"] == [
        "dataset",
        "src/__pycache__",
    ]
    assert clear.data["proxy_policy"] == {
        "preserve_existing": True,
        "require_any_present_for_git_network": True,
        "required_for_stages": ["deploy_git_sync"],
    }
    adopt = clear.data["adopt_existing"]
    assert adopt["mode"] == "legacy_manifest_sha256"
    assert adopt["stages"] == [
        "phase_a_prepare",
        "phase_a_probe",
        "phase_a_audit",
    ]
    assert adopt["expected_generation_commit"] == (
        "f83f701a03306ba6ab0008ea61ce0cc34a2defca"
    )
    assert adopt["allowed_external_manifest_artifacts"] == [
        "baselines/clear_official/dataset/mutagenicity_full.pickle",
        "baselines/clear_official/dataset/mutagenicity_datasplit.pickle",
    ]
    assert "phase_b_gpu_smoke" not in adopt["stages"]


def test_clear_nested_allowlist_matches_patch_001_through_005() -> None:
    touched: set[str] = set()
    patch_root = ROOT / "patches/clear_official"
    for patch in sorted(patch_root.glob("00[1-5]_*.patch")):
        for line in patch.read_text(encoding="utf-8").splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                touched.add(line[6:])
    clear = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    configured = set(
        clear.data["remote_dirty_policy"]["allowed_patched_submodules"][0][
            "allowed_modified_paths"
        ]
    )
    assert configured == touched == {
        "src/data_preprocessing.py",
        "src/main.py",
        "src/models.py",
        "src/train_pred.py",
    }


def test_dependency_cycle_is_rejected(base_spec, write_spec) -> None:
    payload = deepcopy(base_spec)
    payload["stages"][0]["dependencies"] = ["second"]
    second = deepcopy(payload["stages"][0])
    second["id"] = "second"
    second["dependencies"] = ["local_gate"]
    payload["stages"].append(second)
    with pytest.raises(SpecValidationError, match="cycle"):
        load_task_spec(write_spec(payload))


@pytest.mark.parametrize(
    ("permission", "stage_mutation", "message"),
    [
        (
            "allow_test",
            {"command": ["python", "tool.py", "--test-csv", "data/test.csv"]},
            "test split",
        ),
        (
            "allow_calibration",
            {
                "command": [
                    "python",
                    "tool.py",
                    "--input",
                    "data/calibration.csv",
                ]
            },
            "calibration",
        ),
        (
            "allow_full",
            {"id": "gpu_full", "resources": {"tags": "full"}},
            "full run",
        ),
    ],
)
def test_split_and_full_permissions_block(
    base_spec, write_spec, permission, stage_mutation, message
) -> None:
    payload = deepcopy(base_spec)
    payload["permissions"][permission] = False
    stage = payload["stages"][0]
    for key, value in stage_mutation.items():
        if key == "resources":
            stage["resources"].update(value)
        else:
            stage[key] = value
    payload["execution"]["auto_until"] = stage["id"]
    with pytest.raises(SpecValidationError, match=message):
        load_task_spec(write_spec(payload))


@pytest.mark.parametrize(
    "remote_root", ["/", "$HOME", "/share/home/u20526"]
)
def test_dangerous_remote_root_is_rejected(
    base_spec, write_spec, remote_root
) -> None:
    payload = deepcopy(base_spec)
    payload["project"]["remote_root"] = remote_root
    with pytest.raises(SpecValidationError, match="Dangerous remote_root"):
        load_task_spec(write_spec(payload))


def test_sbatch_permission_is_required(base_spec, write_spec) -> None:
    payload = deepcopy(base_spec)
    stage = payload["stages"][0]
    stage.update(
        {
            "kind": "slurm_job",
            "command": [],
            "script": "scripts/slurm/ops/run_stage.sh",
        }
    )
    with pytest.raises(SpecValidationError, match="allow_sbatch=false"):
        load_task_spec(write_spec(payload))


def test_finalized_output_requires_overwrite_permission(
    base_spec, write_spec, tmp_path
) -> None:
    output = tmp_path / "final"
    output.mkdir()
    (output / "_FINALIZED.json").write_text("{}\n", encoding="utf-8")
    payload = deepcopy(base_spec)
    payload["stages"][0]["resources"]["expected_output_root"] = str(output)
    with pytest.raises(SpecValidationError, match="cannot be overwritten"):
        load_task_spec(write_spec(payload))


def test_command_must_be_argv_array(base_spec, write_spec) -> None:
    payload = deepcopy(base_spec)
    payload["stages"][0]["command"] = "python tool.py"
    with pytest.raises(SpecValidationError, match="JSON Schema"):
        load_task_spec(write_spec(payload))


def test_proxy_preservation_must_be_true(base_spec, write_spec) -> None:
    payload = deepcopy(base_spec)
    payload["permissions"]["preserve_proxy_environment"] = False
    with pytest.raises(SpecValidationError, match="preserve_proxy_environment"):
        load_task_spec(write_spec(payload))


def test_proxy_preservation_defaults_to_true(base_spec, write_spec) -> None:
    payload = deepcopy(base_spec)
    del payload["permissions"]["preserve_proxy_environment"]
    spec = load_task_spec(write_spec(payload))
    assert spec.data["permissions"]["preserve_proxy_environment"] is True


def test_remote_policy_paths_must_be_repository_relative(
    base_spec, write_spec
) -> None:
    payload = deepcopy(base_spec)
    payload["remote_dirty_policy"] = {
        "allowed_tracked_paths": ["docs/EXPERIMENT_LOG.md"],
        "allowed_patched_submodules": [
            {
                "path": "../outside",
                "allow_modified": True,
                "allow_untracked": False,
                "allow_staged": False,
                "required_markers": [
                    {"file": "src/main.py", "contains": "marker"}
                ],
                "allowed_modified_paths": ["src/main.py"],
            }
        ],
    }
    with pytest.raises(SpecValidationError, match="repository-relative"):
        load_task_spec(write_spec(payload))


def test_remote_tracked_dirty_allowlist_defaults_to_empty(
    base_spec, write_spec
) -> None:
    payload = deepcopy(base_spec)
    payload["remote_dirty_policy"] = {
        "allowed_patched_submodules": [],
    }
    spec = load_task_spec(write_spec(payload))
    assert spec.data["remote_dirty_policy"]["allowed_tracked_paths"] == []
    assert spec.data["remote_dirty_policy"]["allowed_untracked_paths"] == []


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/absolute/path",
        "../outside",
        "docs/../outside",
        "docs/*.md",
        "docs/file?.md",
        "docs/[abc].md",
        "docs//EXPERIMENT_LOG.md",
        "docs\\EXPERIMENT_LOG.md",
    ],
)
def test_remote_tracked_dirty_allowlist_rejects_non_exact_paths(
    base_spec, write_spec, value
) -> None:
    payload = deepcopy(base_spec)
    payload["remote_dirty_policy"] = {
        "allowed_tracked_paths": [value],
        "allowed_patched_submodules": [],
    }
    with pytest.raises(SpecValidationError):
        load_task_spec(write_spec(payload))


@pytest.mark.parametrize(
    "value",
    [
        "scripts/ops",
        "scripts/ops/experimentctl.py",
        "tests/ops/test_spec.py",
        "ops/specs/example_smoke.yaml",
        "ops/schemas/task_spec.schema.json",
    ],
)
def test_protected_automation_paths_cannot_be_allowlisted(
    base_spec, write_spec, value
) -> None:
    payload = deepcopy(base_spec)
    payload["remote_dirty_policy"] = {
        "allowed_tracked_paths": [value],
        "allowed_patched_submodules": [],
    }
    with pytest.raises(SpecValidationError, match="protected automation path"):
        load_task_spec(write_spec(payload))


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/absolute/file.py",
        "../outside.py",
        "scripts/../outside.py",
        "scripts/paper/*.py",
        "scripts/paper/file?.py",
        "scripts/paper/[abc].py",
        "scripts/paper/",
        "scripts//paper/file.py",
        "scripts\\paper\\file.py",
    ],
)
def test_remote_untracked_allowlist_rejects_non_exact_file_paths(
    base_spec, write_spec, value
) -> None:
    payload = deepcopy(base_spec)
    payload["remote_dirty_policy"] = {
        "allowed_tracked_paths": [],
        "allowed_untracked_paths": [value],
        "allowed_patched_submodules": [],
    }
    with pytest.raises(SpecValidationError):
        load_task_spec(write_spec(payload))


@pytest.mark.parametrize(
    "value",
    [
        "scripts/ops/experimentctl.py",
        "tests/ops/test_spec.py",
        "ops/specs/example_smoke.yaml",
        "ops/schemas/task_spec.schema.json",
    ],
)
def test_protected_automation_files_cannot_be_untracked_allowlisted(
    base_spec, write_spec, value
) -> None:
    payload = deepcopy(base_spec)
    payload["remote_dirty_policy"] = {
        "allowed_tracked_paths": [],
        "allowed_untracked_paths": [value],
        "allowed_patched_submodules": [],
    }
    with pytest.raises(SpecValidationError, match="protected automation path"):
        load_task_spec(write_spec(payload))
