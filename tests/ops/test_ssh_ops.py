from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.ops.ssh_ops import (
    SSHConfig,
    SSHSafetyError,
    build_preflight_argv,
    build_ssh_argv,
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
    assert "u20526@logini.tongji.edu.cn" in argv


def test_control_socket_is_added_without_forward_changes() -> None:
    argv = build_ssh_argv(config("/tmp/tongji-codex.sock"), ["hostname"])
    assert argv[1:3] == ["-S", "/tmp/tongji-codex.sock"]
    joined = " ".join(argv)
    assert "ClearAllForwardings" not in joined
    assert "39393" not in joined


def test_preflight_activation_disables_nounset_around_bashrc() -> None:
    command = build_preflight_argv(config())[-1]
    assert command.index("set +u") < command.index("source ~/.bashrc")
    assert command.index("source ~/.bashrc") < command.index("conda activate")
    assert command.index("conda activate") < command.index("set -u")
    assert "command -v sbatch" in command
    assert "git rev-parse HEAD" in command


def test_preflight_blocks_finalized_outputs_by_default() -> None:
    command = build_preflight_argv(
        config(), protected_output_roots=["outputs/final"]
    )[-1]
    assert "outputs/final/_FINALIZED.json" in command
    assert "FINALIZED_OUTPUT_BLOCKED" in command


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
