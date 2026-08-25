from __future__ import annotations

from pathlib import Path

import pytest

from scripts.autodl import run_aids_comrecgc_exact_recovery_stage as cli


@pytest.mark.parametrize(
    ("stage", "extra", "target"),
    [
        ("subset", ("--adoption-gate", "adoption"), "run_subset_stage"),
        (
            "exact",
            (
                "--adoption-gate",
                "adoption",
                "--subset-gate",
                "subset",
                "--resume",
            ),
            "run_exact_stage",
        ),
        (
            "downstream",
            ("--exact-gate", "exact", "--resume"),
            "run_downstream_stage",
        ),
        (
            "final",
            (
                "--adoption-gate",
                "adoption",
                "--subset-gate",
                "subset",
                "--exact-gate",
                "exact",
                "--downstream-gate",
                "downstream",
                "--resume",
            ),
            "run_final_stage",
        ),
    ],
)
def test_stage_cli_dispatches_only_manifest_bound_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    extra: tuple[str, ...],
    target: str,
) -> None:
    observed: dict[str, object] = {}

    def fake(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(cli, target, fake)
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "output"
    values = [
        "--config",
        "configs/hpc.yaml",
        stage,
        "--controller-manifest",
        str(manifest),
        "--output-dir",
        str(output),
    ]
    iterator = iter(extra)
    for token in iterator:
        values.append(token)
        if token.startswith("--") and token != "--resume":
            values.append(str(tmp_path / next(iterator)))
    assert cli.main(values) == 0
    assert observed["controller_manifest"] == manifest
    assert observed["output_dir"] == output
    assert "_PASS]" in capsys.readouterr().out


def test_stage_cli_requires_absolute_authority_paths() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "subset",
                "--controller-manifest",
                "relative.json",
                "--adoption-gate",
                "/absolute/gate.json",
                "--output-dir",
                "/absolute/output",
            ]
        )


def test_each_stage_help_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    for stage in ("subset", "exact", "downstream", "final"):
        with pytest.raises(SystemExit) as stopped:
            cli.build_parser().parse_args([stage, "--help"])
        assert stopped.value.code == 0
    assert "--controller-manifest" in capsys.readouterr().out
