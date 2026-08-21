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


def test_canonical_command_redacts_sensitive_set_values() -> None:
    argv = canonical_scientific_argv(
        _args("--set", "api_token=do-not-persist", "--set", "seed=7")
    )
    rendered = "\n".join(argv)

    assert "do-not-persist" not in rendered
    assert "api_token=<redacted>" in rendered
    assert "seed=7" in rendered
