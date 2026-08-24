from __future__ import annotations

from scripts.baselines.comrecgc.run_generation import (
    build_parser,
    canonical_scientific_argv,
)
from src.baselines.comrecgc.generation_checkpoint import scientific_command_sha256


def _args(*extra: str):
    return build_parser().parse_args(
        [
            "--route",
            "project",
            "--dataset",
            "bace",
            "--output-dir",
            "/fast/output",
            *extra,
        ]
    )


def test_canonical_command_excludes_only_resume_transport_flag() -> None:
    fresh = canonical_scientific_argv(_args())
    resumed = canonical_scientific_argv(_args("--resume"))

    assert fresh == resumed
    assert all("--resume" not in value for value in fresh)
    assert scientific_command_sha256(fresh) == scientific_command_sha256(resumed)


def test_canonical_command_parameter_change_changes_identity() -> None:
    original = canonical_scientific_argv(_args("--batch-size", "128"))
    changed = canonical_scientific_argv(_args("--batch-size", "64"))

    assert original != changed
    assert scientific_command_sha256(original) != scientific_command_sha256(changed)


def test_split_gnn_and_distance_devices_are_explicit_command_identity() -> None:
    combined = canonical_scientific_argv(_args("--device", "cuda:0"))
    split = canonical_scientific_argv(
        _args(
            "--device",
            "cuda:0",
            "--gnn-device",
            "cpu",
            "--distance-device",
            "cuda:0",
        )
    )

    assert combined != split
    assert any('gnn-device="cpu"' in item for item in split)
    assert any('distance-device="cuda:0"' in item for item in split)
    assert scientific_command_sha256(combined) != scientific_command_sha256(split)


def test_preprocess_engine_and_worker_settings_are_checkpoint_identity() -> None:
    legacy = canonical_scientific_argv(_args())
    optimized = canonical_scientific_argv(
        _args(
            "--bace-preprocess-engine",
            "ordered_bounded_rdkit_process_pool_v1",
            "--bace-preprocess-workers",
            "4",
            "--bace-source-cache-capacity",
            "1024",
            "--bace-candidate-cache-capacity",
            "8192",
        )
    )

    assert legacy != optimized
    assert scientific_command_sha256(legacy) != scientific_command_sha256(
        optimized
    )


def test_full_acceleration_gate_path_and_hash_are_checkpoint_identity() -> None:
    original = canonical_scientific_argv(
        _args(
            "--mode",
            "full",
            "--bace-acceleration-gate",
            "/persistent/gate.json",
            "--bace-acceleration-gate-sha256",
            "a" * 64,
        )
    )
    changed = canonical_scientific_argv(
        _args(
            "--mode",
            "full",
            "--bace-acceleration-gate",
            "/persistent/gate.json",
            "--bace-acceleration-gate-sha256",
            "b" * 64,
        )
    )
    assert original != changed


def test_diagnostic_equivalence_prefix_is_explicit_command_identity() -> None:
    legacy = canonical_scientific_argv(
        _args(
            "--mode",
            "full",
            "--diagnostic-equivalence-steps",
            "500",
            "--equivalence-gate-role",
            "legacy",
        )
    )
    optimized = canonical_scientific_argv(
        _args(
            "--mode",
            "full",
            "--diagnostic-equivalence-steps",
            "500",
            "--equivalence-gate-role",
            "optimized",
            "--bace-preprocess-engine",
            "ordered_bounded_rdkit_process_pool_v1",
            "--bace-preprocess-workers",
            "4",
        )
    )

    assert legacy != optimized
    assert any("diagnostic-equivalence-steps=500" in item for item in legacy)
    assert any('equivalence-gate-role="legacy"' in item for item in legacy)


def test_canonical_command_redacts_sensitive_set_values() -> None:
    argv = canonical_scientific_argv(
        _args("--set", "api_token=do-not-persist", "--set", "seed=7")
    )
    rendered = "\n".join(argv)

    assert "do-not-persist" not in rendered
    assert "api_token=<redacted>" in rendered
    assert "seed=7" in rendered
