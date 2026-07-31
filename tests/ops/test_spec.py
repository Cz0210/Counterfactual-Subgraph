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
