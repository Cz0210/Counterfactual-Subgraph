from __future__ import annotations

import json
from pathlib import Path

from scripts.autodl import build_four_by_four_core_tasks as builder
from scripts.autodl.run_four_gpu_recovery_controller import _parse_task, validate_no_test_before_freeze


def _args(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    prepared = tmp_path / "taste"
    prepared.mkdir()
    (prepared / "provenance_manifest.json").write_text("{}")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    molclr = tmp_path / "molclr"
    molclr.mkdir()
    paths = {}
    for name in (
        "molclr_checkpoint",
        "mut_dataset_csv",
        "mut_teacher_path",
        "mut_distance_checkpoint",
        "mut_thresholds_path",
        "aids_dataset_csv",
        "aids_teacher_path",
        "aids_distance_checkpoint",
        "aids_thresholds_path",
        "aids_source_csv",
    ):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    for name in (
        "mut_source_generation_root",
        "mut_dataset_dir",
        "aids_source_generation_root",
        "aids_dataset_dir",
    ):
        path = tmp_path / name
        path.mkdir()
        if "source_generation" in name:
            (path / "run_manifest.json").write_text("{}")
        paths[name] = path
    return type(
        "Args",
        (),
        {
            "controller_id": "four-methods-four-datasets-v1",
            "runtime_root": runtime,
            "output": tmp_path / "fragment.json",
            "taste_prepared_root": prepared,
            "taste_approval_file": None,
            "taste_upstream_checkout": None,
            "comrecgc_upstream_root": upstream,
            "molclr_root": molclr,
            **paths,
        },
    )()


def test_builds_exact_core_fragment_and_keeps_taste_heavy_blocked(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    result = builder.build_tasks(args)
    payload = json.loads(args.output.read_text())
    by_id = {task["id"]: task for task in payload["tasks"]}

    assert result["status"] == "PASS"
    assert len(by_id) == 9
    assert by_id["tastemolnet_license_audit"]["resource"] == "cpu"
    assert by_id["tastemolnet_license_audit"]["manifest_only"] is True
    assert by_id["tastemolnet_license_audit"]["required_output_any"] == [
        ["PASS", "BLOCKED_LICENSE_REVIEW"]
    ]
    for method in builder.METHODS:
        task = by_id[f"tastemolnet_{method}"]
        assert task["command"] is None
        assert task["blocked_reason"] == "BLOCKED_LICENSE_REVIEW"
        assert task["manifest_only"] is True
        assert task["data_splits"] == []
    mut = by_id["mutagenicity_comrecgc_standardized"]
    aids = by_id["aids_comrecgc_standardized"]
    assert mut["runner_dataset"] == "paper-cell-mutagenicity-comrecgc"
    assert "SOURCE_CSV" not in mut["environment"]
    assert aids["environment"]["SOURCE_CSV"] == str(args.aids_source_csv)
    assert "{attempt}" in aids["expected_output"]
    assert mut["depends_on"] == ["mutagenicity_comrecgc_threshold_freeze"]
    assert aids["depends_on"] == ["aids_comrecgc_threshold_freeze"]
    assert mut["data_splits"] == ["test"]
    assert mut["selector_parameters_frozen"] is True
    assert mut["read_only_test"] is True
    assert by_id["mutagenicity_comrecgc_threshold_freeze"]["freezes_selector"] is True

    # pytest's own tmp directory contains a ``test_*`` component; replace that
    # harness-only prefix before exercising the production held-out-path guard.
    production_payload = json.loads(
        json.dumps(payload).replace(str(tmp_path), "/private/tmp/core-fixture")
    )
    parsed = [_parse_task(task) for task in production_payload["tasks"]]
    validate_no_test_before_freeze(parsed)


def test_refuses_to_overwrite_fragment(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output.write_text("occupied")
    try:
        builder.build_tasks(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("fresh fragment contract was bypassed")
