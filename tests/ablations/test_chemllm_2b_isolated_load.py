from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ablations.llm.contracts import (
    LLMAblationContractError,
    canonical_json_sha256,
)
from src.ablations.llm.isolated_chemllm_load import (
    CHEMLLM_2B_REPOSITORY_ID,
    CHEMLLM_2B_REVISION,
    audit_remote_code,
    build_isolated_child_command,
    build_isolated_child_environment,
    pin_chemllm_2b_snapshot,
    prepare_fresh_output_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts/ablations/llm/audit_chemllm_2b_isolated_load.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path, *, source: str = "class Model:\n    pass\n"):
    root = tmp_path / "CHEMLLM-2b-1_5" / CHEMLLM_2B_REVISION
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoConfig": "modeling_chemllm.Config",
                    "AutoModelForCausalLM": "modeling_chemllm.Model",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "modeling_chemllm.py").write_text(source, encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"mock-safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}}
        ),
        encoding="utf-8",
    )
    manifest = root / "snapshot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "chemllm_snapshot_manifest_v1",
                "status": "PASS",
                "repository_id": CHEMLLM_2B_REPOSITORY_ID,
                "revision": CHEMLLM_2B_REVISION,
                "weights_downloaded": True,
                "parameters": {
                    "count_source": "downloaded_safetensors_tensor_headers",
                    "total_parameters": 1_889_110_016,
                },
            }
        ),
        encoding="utf-8",
    )
    return root, manifest, _sha(manifest)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("chemllm_2b_isolated_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_snapshot_and_remote_code_audit_are_byte_closed(tmp_path: Path) -> None:
    root, manifest, manifest_sha = _snapshot(tmp_path)
    snapshot = pin_chemllm_2b_snapshot(root, manifest, manifest_sha)
    audit = audit_remote_code(snapshot)
    assert snapshot.repository_id == CHEMLLM_2B_REPOSITORY_ID
    assert snapshot.revision == CHEMLLM_2B_REVISION
    assert len(snapshot.weight_files) == 2  # index plus its single physical shard
    assert audit["status"] == "PASS"
    assert audit["required_auto_map_modules"] == ["modeling_chemllm"]
    assert audit["violation_count"] == 0

    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(LLMAblationContractError, match="SHA256 changed"):
        pin_chemllm_2b_snapshot(root, manifest, manifest_sha)


def test_exact_unused_export_exception_is_hash_scoped_and_runtime_disabled(tmp_path, monkeypatch):
    from src.ablations.llm import isolated_chemllm_load as module
    source = "class Model:\n    def save_vocabulary(self):\n        open('/unused-export', 'wb')\n"
    root, manifest, digest = _snapshot(tmp_path, source=source)
    snapshot = pin_chemllm_2b_snapshot(root, manifest, digest)
    with pytest.raises(LLMAblationContractError, match="static audit failed"):
        audit_remote_code(snapshot)
    monkeypatch.setattr(module, "_AUDITED_UNUSED_TOKENIZER_EXPORTS", {
        ("modeling_chemllm.py", _sha(root / "modeling_chemllm.py"))})
    audit = audit_remote_code(snapshot)
    assert audit["audited_unused_export_methods"][0]["runtime_export_disabled"]
    tokenizer = SimpleNamespace()
    module.disable_tokenizer_exports(tokenizer)
    for method in (tokenizer.save_pretrained, tokenizer.save_vocabulary):
        with pytest.raises(LLMAblationContractError, match="export is disabled"):
            method("ignored")
    (root / "modeling_chemllm.py").write_text(source + "open('/not-export', 'wb')\n")
    with pytest.raises(LLMAblationContractError, match="static audit failed"):
        audit_remote_code(snapshot)


@pytest.mark.parametrize(
    "source",
    (
        "import socket\nclass Model: pass\n",
        "class Model:\n    def f(self):\n        return eval('1')\n",
        "class Model:\n    def f(self):\n        open('/tmp/x', 'w')\n",
        "import subprocess\nclass Model: pass\n",
    ),
)
def test_static_remote_code_audit_fails_closed_on_side_effects(
    tmp_path: Path, source: str
) -> None:
    root, manifest, manifest_sha = _snapshot(tmp_path, source=source)
    snapshot = pin_chemllm_2b_snapshot(root, manifest, manifest_sha)
    with pytest.raises(LLMAblationContractError, match="static audit failed"):
        audit_remote_code(snapshot)


def test_isolated_environment_and_command_hide_gpu_and_user_site(tmp_path: Path) -> None:
    root, manifest, manifest_sha = _snapshot(tmp_path)
    snapshot = pin_chemllm_2b_snapshot(root, manifest, manifest_sha)
    audit = audit_remote_code(snapshot)
    output = prepare_fresh_output_root(tmp_path / "evidence")
    environment = build_isolated_child_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "PYTHONPATH": "/untrusted",
            "PYTHONHOME": "/untrusted-home",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        output,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert Path(environment["HF_MODULES_CACHE"]).parent == output

    config = tmp_path / "hpc.yaml"
    config.write_text("project: test\n", encoding="utf-8")
    command = build_isolated_child_command(
        python="/usr/bin/python3",
        script=CLI_PATH,
        snapshot=snapshot,
        output_root=output,
        mode="metadata",
        tiny_forward=False,
        code_inventory_sha256=audit["code_inventory_sha256"],
        config=config,
        config_overrides=("inference.fallback_to_heuristic=false",),
    )
    assert command[:3] == ["/usr/bin/python3", "-I", "-B"]
    assert "--_isolated-child" in command
    assert "--tiny-forward" not in command
    assert "--snapshot-manifest-sha256" in command
    assert "--expected-code-inventory-sha256" in command


def test_parent_cli_uses_no_shell_gpu_or_real_transformer_in_mock_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, manifest_sha = _snapshot(tmp_path)
    config = tmp_path / "hpc.yaml"
    config.write_text("project: test\n", encoding="utf-8")
    output = tmp_path / "isolated-output"
    cli = _load_cli_module()
    observed: dict[str, object] = {}

    def fake_run(command, *, env, shell, check):
        observed.update(command=list(command), env=dict(env), shell=shell, check=check)
        assert shell is False and check is False
        assert command[1:3] == ["-I", "-B"]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["PYTHONNOUSERSITE"] == "1"
        receipt = {
            "schema_version": "chemllm_2b_isolated_load_receipt_v1",
            "status": "PASS",
            "repository_id": CHEMLLM_2B_REPOSITORY_ID,
            "revision": CHEMLLM_2B_REVISION,
            "snapshot_inventory_sha256": "a" * 64,
            "remote_code_audit_sha256": "b" * 64,
            "code_inventory_sha256": "c" * 64,
            "mode": "metadata",
            "isolated_import_pass": True,
            "trust_remote_code_enabled": True,
            "local_files_only": True,
            "offline_mode": True,
            "python_isolated_flag": True,
            "python_no_bytecode_flag": True,
            "python_no_user_site": True,
            "cuda_visible_devices": "",
            "cuda_available": False,
            "cuda_device_count": 0,
            "hf_modules_cache": str(output / "hf_modules_cache"),
            "main_gpu_lock_acquired": False,
            "main_output_root_written": False,
            "weights_loaded": False,
            "actual_parameter_report": None,
            "actual_parameter_report_file_sha256": None,
        }
        receipt["receipt_sha256"] = canonical_json_sha256(receipt)
        (output / "isolated_load_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc = cli.main(
        [
            "--config",
            str(config),
            "--set",
            "inference.fallback_to_heuristic=false",
            "--snapshot-root",
            str(root),
            "--snapshot-manifest",
            str(manifest),
            "--snapshot-manifest-sha256",
            manifest_sha,
            "--output-root",
            str(output),
            "--mode",
            "metadata",
        ]
    )
    assert rc == 0
    assert observed["shell"] is False
    assert (output / "terminal.json").is_file()
    adoption = json.loads(
        (output / "snapshot_runtime_adoption_manifest.json").read_text(encoding="utf-8")
    )
    assert adoption["isolated_import_pass"] is True
    assert adoption["trust_remote_code_enabled"] is True
    assert adoption["original_snapshot_modified"] is False
    assert adoption["actual_parameter_report"] is None


def test_metadata_mode_cannot_request_tiny_forward(tmp_path: Path) -> None:
    root, manifest, manifest_sha = _snapshot(tmp_path)
    snapshot = pin_chemllm_2b_snapshot(root, manifest, manifest_sha)
    audit = audit_remote_code(snapshot)
    output = prepare_fresh_output_root(tmp_path / "evidence")
    config = tmp_path / "hpc.yaml"
    config.write_text("project: test\n", encoding="utf-8")
    with pytest.raises(LLMAblationContractError, match="tiny forward"):
        build_isolated_child_command(
            python="/usr/bin/python3",
            script=CLI_PATH,
            snapshot=snapshot,
            output_root=output,
            mode="metadata",
            tiny_forward=True,
            code_inventory_sha256=audit["code_inventory_sha256"],
            config=config,
        )
