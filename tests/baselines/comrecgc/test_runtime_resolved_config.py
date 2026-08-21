from __future__ import annotations

import json

import pytest

from src.baselines.comrecgc.contracts import stable_json_sha256, write_json
from src.baselines.comrecgc.generation_checkpoint import (
    GenerationCheckpointError,
    scientific_command_sha256,
)
from src.baselines.comrecgc.runtime import (
    _load_persistent_resolved_config,
    _publish_persistent_resolved_config,
)


def _config(argv: tuple[str, ...], *, total_steps: int = 50_000) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "dataset": "bace",
        "mode": "full",
        "scientific_argv": list(argv),
        "command_sha256": scientific_command_sha256(argv),
        "total_steps": total_steps,
        "checkpoint_provenance": {
            "scientific_command_sha256": scientific_command_sha256(argv),
            "total_steps": str(total_steps),
        },
    }
    config["config_sha256"] = stable_json_sha256(config)
    return config


def test_resolved_config_is_durable_before_first_checkpoint_mirror(tmp_path) -> None:
    argv = ("run_generation.py", '--dataset="bace"', '--mode="full"')
    config = _config(argv)
    mirror = tmp_path / "persistent/resume/checkpoints"

    config_path, binding_path = _publish_persistent_resolved_config(
        config, mirror_root=mirror
    )

    # Fault injection: the process dies before creating or mirroring checkpoint 1.
    assert not mirror.exists()
    assert config_path.is_file()
    assert binding_path.is_file()
    assert (
        mirror.parent / "generation_resume_metadata/resolved_config.json"
    ).is_file()
    assert (
        mirror.parent / "generation_resume_metadata/resolved_config.binding.json"
    ).is_file()
    recovered = _load_persistent_resolved_config(
        mirror_root=mirror,
        expected_scientific_argv=argv,
        expected_command_sha256=scientific_command_sha256(argv),
        expected_total_steps=50_000,
    )
    assert recovered == config


def test_resolved_config_incomplete_publication_fails_closed(tmp_path) -> None:
    argv = ("run_generation.py", '--dataset="bace"', '--mode="full"')
    config = _config(argv)
    mirror = tmp_path / "persistent/resume/checkpoints"
    config_path = mirror.parent / "resolved_config.json"
    write_json(config_path, config)

    with pytest.raises(GenerationCheckpointError, match="no complete"):
        _load_persistent_resolved_config(
            mirror_root=mirror,
            expected_scientific_argv=argv,
            expected_command_sha256=scientific_command_sha256(argv),
            expected_total_steps=50_000,
        )


def test_resolved_config_cli_drift_fails_closed(tmp_path) -> None:
    argv = ("run_generation.py", '--dataset="bace"', '--batch-size=128')
    config = _config(argv)
    mirror = tmp_path / "persistent/resume/checkpoints"
    _publish_persistent_resolved_config(config, mirror_root=mirror)
    changed = ("run_generation.py", '--dataset="bace"', '--batch-size=64')

    with pytest.raises(GenerationCheckpointError, match="scientific argv differs"):
        _load_persistent_resolved_config(
            mirror_root=mirror,
            expected_scientific_argv=changed,
            expected_command_sha256=scientific_command_sha256(changed),
            expected_total_steps=50_000,
        )


def test_resolved_config_binding_tamper_fails_closed(tmp_path) -> None:
    argv = ("run_generation.py", '--dataset="bace"', '--mode="full"')
    config = _config(argv)
    mirror = tmp_path / "persistent/resume/checkpoints"
    _config_path, binding_path = _publish_persistent_resolved_config(
        config, mirror_root=mirror
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["total_steps"] = 49_999
    write_json(binding_path, binding)

    with pytest.raises(GenerationCheckpointError, match="binding mismatch"):
        _load_persistent_resolved_config(
            mirror_root=mirror,
            expected_scientific_argv=argv,
            expected_command_sha256=scientific_command_sha256(argv),
            expected_total_steps=50_000,
        )
