from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl import run_comrecgc_standardized_continuation as continuation


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _file(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _inputs(tmp_path: Path, dataset: str = "mutagenicity") -> continuation.ContinuationInputs:
    source = tmp_path / "source"
    source.mkdir()
    payload = _file(source / "counterfactuals.pt")
    parent_limit = 1283 if dataset == "aids" else 1448
    _json(
        source / "run_manifest.json",
        {
            "dataset": dataset,
            "mode": "full",
            "parent_limit": parent_limit,
            "run_complete": True,
            "freeze_only_recovery": True,
            "algorithm_rerun": False,
            "upstream_commit": continuation.UPSTREAM_COMMIT,
            "generation_mode": "adopted_read_only_cache",
            "counterfactuals_path": str(payload),
            "counterfactuals_sha256": "a" * 64,
            "counterfactual_candidate_count": 100,
            "project_commit": "b" * 40,
        },
    )
    _json(
        source / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "freeze_only_recovery": True,
            "counterfactuals_sha256": "a" * 64,
        },
    )
    _json(
        source / "freeze_only_recovery.json",
        {
            "recovery_completed": True,
            "completed_steps": 50_000,
            "algorithm_rerun": False,
            "counterfactuals_sha256": "a" * 64,
        },
    )
    _json(
        source / "frozen_payload_closure_audit.json",
        {"closure_complete": True, "post_write_reload_verified": True},
    )
    _json(
        source / "adoption_manifest.json",
        {
            "generation_mode": "adopted_read_only_cache",
            "source_checksums": {"resolved_config.json": "c" * 64},
        },
    )
    _file(source / "trace/candidate_action_lineage.json")
    _file(source / "trace/trace_summary.json")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    molclr_root = tmp_path / "molclr"
    molclr_root.mkdir()
    return continuation.ContinuationInputs(
        dataset=dataset,
        source_generation_root=source,
        upstream_root=upstream,
        dataset_dir=dataset_dir,
        source_csv=_file(tmp_path / "source.csv") if dataset == "aids" else None,
        distance_checkpoint=_file(tmp_path / "distance.pt"),
        dataset_csv=_file(tmp_path / "test.csv"),
        teacher_path=_file(tmp_path / "teacher.pkl"),
        molclr_root=molclr_root,
        molclr_checkpoint=_file(molclr_root / "model.pth"),
        thresholds_path=_file(tmp_path / "thresholds.json", "{}"),
        output_root=tmp_path / "fresh-output",
        device="cuda:0",
        theta_star=0.05 if dataset == "aids" else None,
        cost_cap=0.0535 if dataset == "aids" else None,
    )


def test_adopts_recovered_generation_without_hashing_large_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    original = continuation.sha256_file
    hashed: list[Path] = []

    def recording_sha(path: str | Path) -> str:
        resolved = Path(path).resolve()
        hashed.append(resolved)
        if resolved.name == "counterfactuals.pt":
            raise AssertionError("adoption preflight must not rescan the multi-GB payload")
        return original(resolved)

    monkeypatch.setattr(continuation, "sha256_file", recording_sha)
    result = continuation.validate_adopted_generation(inputs)
    assert result["status"] == "PASS"
    assert result["generation_adopted"] is True
    assert result["generation_rerun"] is False
    assert result["counterfactuals_sha256_claimed"] == "a" * 64
    assert hashed


def test_parent_metadata_never_enters_generation_content_identity(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest_path = inputs.source_generation_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["representative_parent_id"] = "DIFFERENT_PROVENANCE_PARENT"
    _json(manifest_path, manifest)
    result = continuation.validate_adopted_generation(inputs)
    assert result["status"] == "PASS"
    assert "representative_parent_id" not in result


def test_rejects_payload_outside_frozen_recovery_root(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    outside = _file(tmp_path / "outside.pt")
    manifest_path = inputs.source_generation_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counterfactuals_path"] = str(outside)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not_inside_frozen_source"):
        continuation.validate_adopted_generation(inputs)


def test_commands_use_native_comrecgc_and_frozen_rf_contract(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, dataset="aids")
    commands = continuation.build_stage_commands(
        inputs,
        project_commit="d" * 40,
        candidate_count=100262,
        teacher_sha256="e" * 64,
    )
    names = [row[0] for row in commands]
    assert names == ["common_recourse", "chemistry", "unified_eval", "full_gate", "freeze"]
    common = commands[0][1]
    chemistry = commands[1][1]
    evaluation = commands[2][1]
    gate = commands[3][1]
    assert "run_common_recourse.py" in " ".join(common)
    assert common[common.index("--generation-dir") + 1] == str(
        inputs.source_generation_root
    )
    assert chemistry[chemistry.index("--expected-candidate-count") + 1] == "100262"
    assert evaluation[evaluation.index("--teacher-path") + 1] == str(
        inputs.teacher_path
    )
    assert evaluation[evaluation.index("--theta-star") + 1] == "0.050000000000000003"
    assert gate[gate.index("--expected-teacher-sha256") + 1] == "e" * 64


def test_pass_marker_is_published_only_after_all_frozen_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *_args, **_kwargs: {"passed": True, "actual_commit": continuation.UPSTREAM_COMMIT},
    )
    monkeypatch.setattr(continuation, "_git_head", lambda: "f" * 40)
    observed: list[str] = []

    def fake_stage(**kwargs) -> None:
        observed.append(str(kwargs["stage"]))
        if kwargs["stage"] == "freeze":
            standardized = inputs.output_root / "standardized"
            _json(
                standardized / "run_manifest.json",
                {
                    "dataset_key": "mutagenicity",
                    "cf_mode": continuation.CF_MODE,
                    "distance_line": continuation.DISTANCE_LINE,
                    "teacher_sha256": continuation.sha256_file(inputs.teacher_path),
                    "molclr_checkpoint_sha256": "1" * 64,
                    "dataset_csv_sha256": "2" * 64,
                },
            )
            _json(
                standardized / "freeze_manifest.json",
                {"dataset_key": "mutagenicity"},
            )

    monkeypatch.setattr(continuation, "_run_stage", fake_stage)
    result = continuation.run_continuation(inputs)
    assert observed == ["common_recourse", "chemistry", "unified_eval", "full_gate", "freeze"]
    assert result["status"] == "PASS"
    assert (inputs.output_root / "PASS").read_text(encoding="utf-8") == "PASS\n"
    assert json.loads((inputs.output_root / "_RUN_COMPLETE.json").read_text())["run_complete"] is True


def test_failure_keeps_evidence_and_never_publishes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(continuation, "_git_head", lambda: "f" * 40)

    def fail_stage(**_kwargs) -> None:
        raise RuntimeError("semantic lineage failure")

    monkeypatch.setattr(continuation, "_run_stage", fail_stage)
    with pytest.raises(RuntimeError, match="semantic lineage failure"):
        continuation.run_continuation(inputs)
    failure = json.loads((inputs.output_root / "FAILED.json").read_text())
    assert failure["status"] == "FAILED"
    assert not (inputs.output_root / "PASS").exists()
