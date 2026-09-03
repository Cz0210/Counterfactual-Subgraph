from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest


from scripts.hpc.t8 import run_stress_followup as followup
from src.baselines.globalgce_hpc_exact import GlobalGCEHPCExactError, canonical_sha256


ROOT = Path(__file__).resolve().parents[3]
HPC = ROOT / "scripts" / "hpc" / "t8"


def _unit(
    unit_id: str,
    code: list[dict[str, object]],
    *,
    kind: str = "PREFIX_SUBTREE",
    support: int = 1,
    order: int = 0,
    shard: int = 0,
) -> dict[str, object]:
    return {
        "partition_id": unit_id,
        "partition_type": kind,
        "root_index": 0,
        "dfs_code": code,
        "dfs_code_sha256": "a" * 64,
        "support_hint": support,
        "global_partition_order": order,
        "shard_index": shard,
    }


def test_sacct_parser_uses_exact_base_job_state_not_batch_exit_code() -> None:
    text = "\n".join(
        (
            "2535373|t8-gspan-canary|intel|TIMEOUT|0:0|3627|01:00:00|8|64G||2026-09-03T20:07:04|2026-09-03T21:07:31|",
            "2535373.batch|batch||FAILED|15:0|3628||8|||||",
            "2535373.extern|extern||COMPLETED|0:0|3628||8|||||",
        )
    )
    parsed = followup.parse_sacct(text, "2535373")
    assert parsed["state"] == "TIMEOUT"
    assert parsed["exit_code"] == "0:0"
    assert parsed["elapsed_seconds"] == 3627


def test_sacct_parser_rejects_nonterminal_or_duplicate_base_rows() -> None:
    running = "12|x|intel|RUNNING|0:0|1|01:00:00|1|1G||||"
    with pytest.raises(GlobalGCEHPCExactError, match="not terminal"):
        followup.parse_sacct(running, "12")
    completed = "12|x|intel|COMPLETED|0:0|1|01:00:00|1|1G||||"
    with pytest.raises(GlobalGCEHPCExactError, match="found 2"):
        followup.parse_sacct(completed + "\n" + completed, "12")


def test_depth_plus_one_children_are_strict_and_heaviest_is_deterministic() -> None:
    a = {"frm": 0, "to": 1, "labels": [3, 1, 3]}
    b = {"frm": 1, "to": 2, "labels": [-1, 1, 3]}
    c = {"frm": 2, "to": 3, "labels": [-1, 1, 3]}
    x = {"frm": 3, "to": 4, "labels": [-1, 1, 3]}
    y = {"frm": 3, "to": 0, "labels": [-1, 1, -1]}
    parent = _unit("parent", [a, b, c], support=2957)
    parent_manifest = {"scientific_input_sha256": "1" * 64, "root_universe_sha256": "2" * 64}
    catalog = {
        "scope": "SELECTED_ROOTS_CANARY",
        "split_depth": 4,
        "included_root_indices": [0],
        "scientific_input_sha256": "1" * 64,
        "root_universe_sha256": "2" * 64,
        "partitions": [
            _unit("ancestor-header", [a], kind="PREFIX_HEADER", order=0),
            _unit("parent-header", [a, b, c], kind="PREFIX_HEADER", order=1),
            _unit("other", [a, y, c, x], support=9999, order=2),
            _unit("z-child", [a, b, c, x], support=200, order=4),
            _unit("a-child", [a, b, c, y], support=200, order=3),
            _unit("terminal-child", [a, b, c, {**x, "labels": [-1, 9, 3]}], kind="PREFIX_HEADER", support=300, order=5),
        ],
    }
    header, children, selected = followup.derive_refinement_children(
        parent_manifest, parent, catalog
    )
    assert header["partition_id"] == "parent-header"
    assert [row["partition_id"] for row in children] == ["a-child", "z-child", "terminal-child"]
    assert selected["partition_id"] == "a-child"
    assert all(len(row["dfs_code"]) == 4 for row in children)


def test_refinement_rejects_changed_science_or_missing_refinable_child() -> None:
    edge = {"frm": 0, "to": 1, "labels": [1, 1, 1]}
    parent = _unit("parent", [edge])
    base = {
        "scope": "SELECTED_ROOTS_CANARY",
        "split_depth": 2,
        "included_root_indices": [0],
        "scientific_input_sha256": "x",
        "root_universe_sha256": "r",
        "partitions": [
            _unit("parent-header", [edge], kind="PREFIX_HEADER"),
            _unit("terminal", [edge, edge], kind="PREFIX_HEADER", order=1),
        ],
    }
    with pytest.raises(GlobalGCEHPCExactError, match="not bound"):
        followup.derive_refinement_children(
            {"scientific_input_sha256": "s", "root_universe_sha256": "r"}, parent, base
        )
    base["scientific_input_sha256"] = "s"
    with pytest.raises(GlobalGCEHPCExactError, match=r"all depth\+1 children"):
        followup.derive_refinement_children(
            {"scientific_input_sha256": "s", "root_universe_sha256": "r"}, parent, base
        )


def test_admission_requires_both_walltime_and_storage() -> None:
    canary = {"partitions": [_unit("c", [], support=10)]}
    full = {
        "shard_count": 2,
        "partitions": [
            _unit("a", [], support=20, shard=0),
            _unit("b", [], support=10, shard=1),
        ],
    }
    passed = followup.compute_admission(
        canary_elapsed_seconds=100,
        canary_bytes=1_000,
        canary_manifest=canary,
        full_manifest=full,
        free_bytes=20 * 1024**3,
        walltime_limit_seconds=1_000,
        time_safety_factor=2.0,
        storage_safety_factor=2.0,
    )
    assert passed["projected_longest_shard_seconds"] == 400
    assert passed["projected_persistent_bytes"] == 6_000
    assert passed["persistent_reserve_bytes"] == 4 * 1024**3
    assert passed["admission_pass"] is True
    blocked = followup.compute_admission(
        canary_elapsed_seconds=100,
        canary_bytes=1_000,
        canary_manifest=canary,
        full_manifest=full,
        free_bytes=2 * 1024**3,
        walltime_limit_seconds=399,
        time_safety_factor=2.0,
        storage_safety_factor=2.0,
    )
    assert blocked["walltime_admission_pass"] is False
    assert blocked["storage_admission_pass"] is False
    assert blocked["admission_pass"] is False


def test_self_hashed_receipt_is_atomic_and_has_detached_sha(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = followup.atomic_write_self_hashed(
        path, {"schema_version": "tiny", "state": "PASS"}, hash_field="receipt_sha256"
    )
    assert receipt["receipt_sha256"] == canonical_sha256(
        {"schema_version": "tiny", "state": "PASS"}
    )
    assert followup.load_self_hashed(path, hash_field="receipt_sha256") == receipt
    assert path.with_suffix(".json.sha256").read_text().strip() == followup.sha256_file(path)
    assert not list(tmp_path.glob("*.tmp"))


def test_programmatic_slurm_logs_are_precreated_absolute_and_cwd_independent(
    tmp_path: Path,
) -> None:
    """Regression for jobs 2535893/2535894 failing before script startup.

    Their wrappers used ``logs/%x-%j`` while the immutable worktree had no
    ``logs`` directory.  Programmatic submissions must override that fallback
    with a directory which already exists before ``sbatch`` is called.
    """

    log_root = (tmp_path / "continuation" / "control" / "slurm-logs").resolve()
    assert not log_root.exists()
    scalar = followup._slurm_log_options(
        log_root, stem="refinement-canary"
    )
    array = followup._slurm_log_options(
        log_root, stem="full-array", array=True
    )

    assert log_root.is_dir()
    assert scalar == [
        "--output",
        str(log_root / "refinement-canary-%j.out"),
        "--error",
        str(log_root / "refinement-canary-%j.err"),
    ]
    assert array == [
        "--output",
        str(log_root / "full-array-%A_%a.out"),
        "--error",
        str(log_root / "full-array-%A_%a.err"),
    ]
    assert all(
        Path(value).is_absolute()
        for options in (scalar, array)
        for value in options[1::2]
    )


def test_execution_input_hash_calls_are_keyword_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    science = tmp_path / "science"
    controller = tmp_path / "controller"
    science_config = science / "configs" / "hpc.yaml"
    config = controller / "configs" / "hpc.yaml"
    science_config.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    science_config.write_text("seed: 7\n", encoding="utf-8")
    config.write_text("seed: 7\n", encoding="utf-8")
    graph = tmp_path / "graphs.jsonl"
    graph.write_text("{}\n", encoding="utf-8")
    config_sha = followup.sha256_file(config)
    graph_sha = followup.sha256_file(graph)
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(
        json.dumps(
            {
                "state": "PASS",
                "route_kind": "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD",
                "split_scope": "train_only",
                "calibration_payload_included": False,
                "test_payload_included": False,
                "matrix_publication_allowed_from_hpc": False,
                "mining_config_sha256": "d" * 64,
                "hpc_runtime_config": {"sha256": config_sha},
                "transaction_binding": {"graph_jsonl_sha256": graph_sha},
            }
        ),
        encoding="utf-8",
    )
    controller_commit = "b" * 40

    def fake_git(
        _command: list[str], *, cwd: Path, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text
        value = (
            followup.PINNED_SCIENCE_COMMIT if Path(cwd) == science else controller_commit
        )
        return subprocess.CompletedProcess([], 0, stdout=value + "\n", stderr="")

    monkeypatch.setattr(followup.subprocess, "run", fake_git)
    args = argparse.Namespace(
        config=config,
        set=[],
        expected_science_commit=followup.PINNED_SCIENCE_COMMIT,
        science_worktree=science,
        expected_controller_commit=controller_commit,
        controller_worktree=controller,
        input_manifest=input_manifest,
        expected_input_manifest_sha256=followup.sha256_file(input_manifest),
        expected_hpc_config_sha256=config_sha,
        expected_config_sha256="d" * 64,
        graphs_jsonl=graph,
    )
    assert followup._validate_execution_inputs(args)["state"] == "PASS"


def test_dry_run_timeout_never_calls_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    edge = {"frm": 0, "to": 1, "labels": [1, 1, 1]}
    parent_code = [edge, edge, edge]
    parent = _unit("parent", parent_code, support=10)
    parent_manifest = {
        "manifest_sha256": "m" * 64,
        "split_depth": 3,
        "scientific_input_sha256": "s" * 64,
        "root_universe_sha256": "r" * 64,
    }
    catalog = {
        "scope": "SELECTED_ROOTS_CANARY",
        "split_depth": 4,
        "included_root_indices": [0],
        "scientific_input_sha256": "s" * 64,
        "root_universe_sha256": "r" * 64,
        "manifest_sha256": "c" * 64,
        "partitions": [
            _unit("parent-header", parent_code, kind="PREFIX_HEADER"),
            _unit("child", [*parent_code, edge], support=9, order=1),
        ],
    }

    def fake_build(path: Path, **_: object) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalog), encoding="utf-8")
        return catalog

    monkeypatch.setattr(followup, "_validate_parent_timeout", lambda _: (parent_manifest, parent))
    monkeypatch.setattr(followup, "_build_manifest_or_adopt", fake_build)
    monkeypatch.setattr(
        followup,
        "submit_sbatch",
        lambda _: (_ for _ in ()).throw(AssertionError("dry run submitted sbatch")),
    )
    args = argparse.Namespace(
        upstream_canary_root=tmp_path / "canary",
        continuation_root=tmp_path / "continuation",
        output_root=tmp_path / "decisions",
        controller_worktree=tmp_path / "controller",
        expected_controller_commit="b" * 40,
        science_worktree=tmp_path / "science",
        submit=False,
        graphs_jsonl=tmp_path / "graphs",
        input_manifest=tmp_path / "input",
        expected_science_commit=followup.PINNED_SCIENCE_COMMIT,
        official_src=tmp_path / "official",
        min_support=2,
        min_vertices=3,
        max_vertices=20,
        top_k=20,
    )
    result = followup._handle_timeout(
        args,
        {"job_id": "2535373", "state": "TIMEOUT"},
        tmp_path / "decisions" / "upstream-2535373",
    )
    assert result["state"] == "READY_REFINEMENT_CANARY"
    assert result["dry_run"] is True
    assert result["fresh_canary_root"].endswith("DRY_RUN_FRESH_UUID")


def test_refinement_is_hard_bounded_to_four_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edge = {"frm": 0, "to": 1, "labels": [1, 1, 1]}
    parent = _unit("depth-seven", [edge] * 7, support=10)
    parent_manifest = {"manifest_sha256": "m" * 64}
    monkeypatch.setattr(
        followup,
        "_validate_parent_timeout",
        lambda _: (parent_manifest, parent),
    )
    monkeypatch.setattr(
        followup,
        "submit_sbatch",
        lambda _: (_ for _ in ()).throw(AssertionError("bounded route submitted")),
    )
    args = argparse.Namespace(
        upstream_canary_root=tmp_path / "canary",
        output_root=tmp_path / "decisions",
        controller_worktree=tmp_path / "controller",
        expected_controller_commit="b" * 40,
        science_worktree=tmp_path / "science",
        expected_science_commit=followup.PINNED_SCIENCE_COMMIT,
        submit=True,
    )
    report = followup._handle_timeout(
        args,
        {"job_id": "2535373", "state": "TIMEOUT"},
        tmp_path / "decisions" / "upstream-2535373",
    )
    assert report["state"] == "BLOCKED_MAX_REFINEMENT_LEVELS"
    assert report["requested_refinement_level"] == 5
    assert report["max_refinement_levels"] == 4


def test_telemetry_is_atomic_observational_and_reports_unavailable_dfs(
    tmp_path: Path,
) -> None:
    canary = tmp_path / "canary"
    scratch = tmp_path / "scratch"
    proc = tmp_path / "proc"
    (canary / "reference").mkdir(parents=True)
    scratch.mkdir()
    (canary / "events.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (scratch / "patterns.jsonl").write_text("{}\n", encoding="utf-8")
    (canary / "partition_manifest.json").write_text(
        json.dumps(
            {"partitions": [{"partition_id": "p", "dfs_code": [1, 2, 3, 4]}]}
        ),
        encoding="utf-8",
    )
    (canary / "reference" / "checkpoint.json").write_text(
        json.dumps({"current_unit_id": "p"}), encoding="utf-8"
    )
    for pid, ppid, rss_kib in ((123, 0, 100), (124, 123, 50)):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text(
            f"Name:\ttest\nPPid:\t{ppid}\nVmRSS:\t{rss_kib} kB\n",
            encoding="utf-8",
        )
    payload, _signature, last_progress = followup.collect_telemetry(
        canary_root=canary,
        scratch_roots=(scratch,),
        science_pid=123,
        persistent_cache={},
        scratch_cache={},
        previous_signature=None,
        last_progress_at=None,
        proc_root=proc,
    )
    assert payload["persistent"]["event_lines"] == 2
    assert payload["scratch"]["pattern_lines"] == 1
    assert payload["process_tree_rss_bytes"] == 150 * 1024
    assert payload["current_partition_prefix_depth"] == 4
    assert payload["current_dfs_depth"] is None
    assert payload["pure_observation"] is True
    assert payload["used_for_algorithm_control"] is False
    assert last_progress == payload["observed_at"]


def test_slurm_followup_is_cpu_only_scratch_bound_and_separates_chain() -> None:
    shell = (HPC / "slurm_stress_followup.sh").read_text(encoding="utf-8")
    python = (HPC / "run_stress_followup.py").read_text(encoding="utf-8")
    assert "#SBATCH --partition=intel" in shell
    assert "#SBATCH --gres" not in shell
    assert 'export CUDA_VISIBLE_DEVICES=""' in shell
    assert "SLURM_TMPDIR" in shell
    assert '--scratch-root "$job_tmp/merge"' in shell
    assert "merge)" in shell and "package)" in shell
    assert "afterany:" in python
    assert python.count("afterok:") >= 2
    assert 'parser.add_argument("--submit", action="store_true")' in python
    assert "submit_sbatch(" in python
    assert "shell=True" not in python
    assert '"T8_EXECUTION_WORKTREE": str(args.science_worktree)' in python
    assert 'args.science_worktree / "scripts/hpc/t8/slurm_array.sh"' in python
    assert 'args.controller_worktree / "scripts/hpc/t8/slurm_stress_followup.sh"' in python
    assert followup.PINNED_SCIENCE_COMMIT in python
    assert "refinement-canary" in shell
    assert 'bash "$T8_SCIENCE_WORKTREE/scripts/hpc/t8/slurm_canary.sh"' in shell
    assert "run_stress_followup.py monitor" in shell
    assert python.count("*_slurm_log_options(") == 5
    for stem in (
        "refinement-canary",
        "refinement-followup",
        "full-array",
        "full-merge",
        "result-package",
    ):
        assert f'stem="{stem}"' in python
    subprocess.run(["bash", "-n", str(HPC / "slurm_stress_followup.sh")], check=True)


def test_python_entrypoint_has_required_paired_cpu_wrapper() -> None:
    wrapper = ROOT / "scripts" / "slurm" / "run_stress_followup.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "CPU-only" in text
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --gres" not in text
    assert "scripts/hpc/t8/slurm_stress_followup.sh" in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "T8_CONTROLLER_WORKTREE" in text
    assert "T8_EXECUTION_WORKTREE" not in text
    subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_parser_is_dry_run_by_default() -> None:
    action = next(action for action in followup.build_parser()._actions if action.dest == "submit")
    assert action.default is False
    assert followup.build_parser().get_default("array_concurrency") == 4
    assert followup.MAX_ARRAY_CONCURRENCY == 8


def test_controller_and_science_exports_are_disjoint() -> None:
    args = argparse.Namespace(
        partition="intel",
        science_worktree=Path("/science-481"),
        expected_science_commit=followup.PINNED_SCIENCE_COMMIT,
        controller_worktree=Path("/controller-new"),
        expected_controller_commit="b" * 40,
        python=Path("/python"),
        graphs_jsonl=Path("/graphs.jsonl"),
        input_manifest=Path("/input.json"),
        expected_input_manifest_sha256="1" * 64,
        expected_config_sha256="2" * 64,
        expected_hpc_config_sha256="3" * 64,
        official_src=Path("/official"),
        min_support=2,
        min_vertices=3,
        max_vertices=20,
        top_k=20,
    )
    science = followup._science_export(args)
    controller = followup._controller_export(args)
    assert science["T8_EXECUTION_WORKTREE"] == "/science-481"
    assert science["T8_EXPECTED_COMMIT"] == followup.PINNED_SCIENCE_COMMIT
    assert "T8_CONTROLLER_WORKTREE" not in science
    assert controller["T8_CONTROLLER_WORKTREE"] == "/controller-new"
    assert controller["T8_EXPECTED_CONTROLLER_COMMIT"] == "b" * 40
    assert "T8_EXECUTION_WORKTREE" not in controller


def test_nonpinned_science_commit_fails_before_git_or_hash_checks(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        config=tmp_path / "configs/hpc.yaml",
        set=[],
        expected_science_commit="a" * 40,
        expected_controller_commit="b" * 40,
    )
    with pytest.raises(GlobalGCEHPCExactError, match="science commit must remain pinned"):
        followup._validate_execution_inputs(args)
