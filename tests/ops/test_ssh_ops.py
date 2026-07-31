from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from scripts.ops.ssh_ops import (
    PROXY_VARIABLES,
    SSHConfig,
    SSHSafetyError,
    _proxy_presence_lines,
    build_deploy_argv,
    build_preflight_argv,
    build_ssh_argv,
    build_status_argv,
    parse_preflight_output,
)
from scripts.ops.subprocess_utils import environment_audit, inherited_environment


def config(socket: str | None = None) -> SSHConfig:
    return SSHConfig(
        host="logini.tongji.edu.cn",
        port=10022,
        user="u20526",
        remote_root="/share/home/u20526/czx/counterfactual-subgraph",
        conda_env="smiles_pip118",
        control_socket=socket,
    )


def test_standard_ssh_argv_uses_port_and_batch_mode() -> None:
    argv = build_ssh_argv(config(), ["hostname"])
    assert argv[:3] == ["ssh", "-p", "10022"]
    assert "BatchMode=yes" in argv
    assert "ClearAllForwardings=yes" in argv
    assert "u20526@logini.tongji.edu.cn" in argv


def test_control_socket_child_clears_inherited_forward_requests() -> None:
    argv = build_ssh_argv(config("/tmp/tongji-codex.sock"), ["hostname"])
    assert argv[1:3] == ["-S", "/tmp/tongji-codex.sock"]
    joined = " ".join(argv)
    assert "BatchMode=yes" in joined
    assert "ClearAllForwardings=yes" in joined
    assert "39393" not in joined


def test_all_ordinary_ssh_builders_disable_inherited_forwarding() -> None:
    ssh = config("/tmp/tongji-codex.sock")
    commands = (
        build_preflight_argv(ssh),
        build_deploy_argv(
            ssh, branch="main", expected_commit="a" * 40
        ),
        build_status_argv(ssh, ["12345"]),
    )
    for argv in commands:
        assert ["-o", "BatchMode=yes"] == argv[
            argv.index("BatchMode=yes") - 1 : argv.index("BatchMode=yes") + 1
        ]
        assert ["-o", "ClearAllForwardings=yes"] == argv[
            argv.index("ClearAllForwardings=yes")
            - 1 : argv.index("ClearAllForwardings=yes")
            + 1
        ]


def test_preflight_activation_disables_nounset_around_bashrc() -> None:
    command = build_preflight_argv(config())[-1]
    assert command.index("set +u") < command.index("source ~/.bashrc")
    assert command.index("source ~/.bashrc") < command.index("conda activate")
    assert "set -u" not in command
    assert "command -v sbatch" in command
    assert "git rev-parse HEAD" in command


def test_preflight_blocks_finalized_outputs_by_default() -> None:
    command = build_preflight_argv(
        config(), protected_output_roots=["outputs/final"]
    )[-1]
    assert "outputs/final/_FINALIZED.json" in command
    assert "PREFLIGHT_FINALIZED" in command


def test_password_automation_is_forbidden() -> None:
    with pytest.raises(SSHSafetyError, match="Password"):
        build_ssh_argv(config(), ["sshpass", "--password", "secret"])


def test_proxy_environment_is_preserved_without_value_logging(monkeypatch) -> None:
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:39393")
    environment = inherited_environment(preserve_proxy_environment=True)
    audit = environment_audit(environment)
    assert environment["https_proxy"] == os.environ["https_proxy"]
    assert audit["proxy_present"]["https_proxy"] is True
    assert "https_proxy" not in audit["allowlisted_values"]


def test_ssh_module_never_unsets_proxy_or_modifies_user_files() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts/ops/ssh_ops.py"
    ).read_text(encoding="utf-8")
    assert "unset http_proxy" not in source
    assert "unset https_proxy" not in source
    assert ".ssh/config" not in source
    assert "unlink" not in source


def test_ssh_argv_has_exactly_one_bash_lc_layer() -> None:
    argv = build_preflight_argv(config())
    assert argv[-3:-1] == ["bash", "-lc"]
    assert "bash -lc" not in argv[-1]
    assert argv.count("bash") == 1
    assert argv.count("-lc") == 1


def test_preflight_script_is_read_only() -> None:
    script = build_preflight_argv(config())[-1]
    assert "git status --short" in script
    assert "git branch --show-current" in script
    assert "git rev-parse HEAD" in script
    for forbidden in (
        "git fetch",
        "git pull",
        "git merge",
        "git reset",
        "git clean",
        "git stash",
        "\nsbatch ",
        "exp_sbatch",
        "\nmkdir ",
        "\nmv ",
        "\nrm ",
    ):
        assert forbidden not in script


def test_preflight_proxy_checks_are_well_spaced_bash() -> None:
    script = build_preflight_argv(config())[-1]
    assert "thenecho" not in script
    assert "elseecho" not in script
    assert "-n${" not in script
    for variable in PROXY_VARIABLES:
        assert f"if [[ -n ${{{variable}+x}} ]]; then\n" in script
        assert f"  echo '[PREFLIGHT_PROXY_{variable}] true'" in script


def test_complete_preflight_remote_script_passes_bash_syntax() -> None:
    script = build_preflight_argv(config())[-1]
    result = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("variable", PROXY_VARIABLES)
@pytest.mark.parametrize("is_set", [False, True])
def test_proxy_presence_fragment_reports_only_presence(
    variable: str, is_set: bool
) -> None:
    environment = dict(os.environ)
    for key in PROXY_VARIABLES:
        environment.pop(key, None)
    secret = "http://127.0.0.1:39393/do-not-print"
    if is_set:
        environment[variable] = secret
    result = subprocess.run(
        ["bash"],
        input="\n".join(_proxy_presence_lines(variable)) + "\n",
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    expected = str(is_set).lower()
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"[PREFLIGHT_PROXY_{variable}] {expected}\n"
    assert secret not in result.stdout


def test_preflight_collects_patched_submodule_evidence_read_only() -> None:
    policy = {
        "path": "baselines/clear_official",
        "required_markers": [
            {
                "file": "src/main.py",
                "contains": "CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET",
            }
        ],
    }
    script = build_preflight_argv(
        config(), patched_submodules=[policy]
    )[-1]
    assert "git -C baselines/clear_official status --porcelain=v1" in script
    assert "git -C baselines/clear_official diff --name-only" in script
    assert "git -C baselines/clear_official diff --cached --name-only" in script
    assert "CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET" in script
    for forbidden in (" reset ", " clean ", " restore ", " stash "):
        assert forbidden not in script


def test_parse_preflight_records_only_proxy_presence() -> None:
    stdout = "\n".join(
        [
            "[PREFLIGHT_CONDA_READY] true",
            "[PREFLIGHT_HOSTNAME] logini02",
            "[PREFLIGHT_PWD] /remote/project",
            "[PREFLIGHT_BRANCH] main",
            "[PREFLIGHT_COMMIT] abc",
            "[PREFLIGHT_DIRTY_BEGIN]",
            " M docs/EXPERIMENT_LOG.md",
            "[PREFLIGHT_DIRTY_END]",
            "[PREFLIGHT_PYTHON] Python 3.10.18",
            "[PREFLIGHT_SBATCH_READY] true",
            "[PREFLIGHT_SACCT_READY] true",
            "[PREFLIGHT_PROXY_https_proxy] true",
            "[PREFLIGHT_PROXY_HTTP_PROXY] false",
            "[PREFLIGHT_FINALIZED_BLOCKED] false",
        ]
    )
    result = parse_preflight_output(stdout)
    assert result.commit == "abc"
    assert result.dirty_lines == (" M docs/EXPERIMENT_LOG.md",)
    assert result.proxy_present == {
        "https_proxy": True,
        "HTTP_PROXY": False,
    }
    assert "39393" not in str(result.to_dict())


def test_parse_preflight_records_nested_repository_evidence() -> None:
    stdout = "\n".join(
        [
            "[PREFLIGHT_SUBMODULE_STATUS_BEGIN] baselines/clear_official",
            " M src/main.py",
            "[PREFLIGHT_SUBMODULE_STATUS_END] baselines/clear_official",
            "[PREFLIGHT_SUBMODULE_MODIFIED_BEGIN] baselines/clear_official",
            "src/main.py",
            "[PREFLIGHT_SUBMODULE_MODIFIED_END] baselines/clear_official",
            "[PREFLIGHT_SUBMODULE_STAGED_BEGIN] baselines/clear_official",
            "[PREFLIGHT_SUBMODULE_STAGED_END] baselines/clear_official",
            "[PREFLIGHT_SUBMODULE_MARKER] "
            "baselines/clear_official|src/main.py|true",
        ]
    )
    result = parse_preflight_output(stdout)
    assert len(result.submodules) == 1
    nested = result.submodules[0]
    assert nested.path == "baselines/clear_official"
    assert nested.modified_paths == ("src/main.py",)
    assert nested.staged_paths == ()
    assert nested.marker_results == {"src/main.py": True}
