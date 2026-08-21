from __future__ import annotations

import cProfile
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.autodl import run_three_lines_stage as stage


def graph_diff_profile_probe() -> None:
    sum(range(4))


def transition_reconstruction_profile_probe() -> None:
    sum(range(5))


def canonical_hash_profile_probe() -> None:
    hash("canonical")


def rdkit_smiles_profile_probe() -> None:
    str("C")


def sqlite_execute_profile_probe() -> None:
    tuple(range(2))


def trace_serialization_profile_probe() -> None:
    json.dumps({"trace": 1})


def model_inference_profile_probe() -> None:
    max(1, 2)


def all_required_profile_probes() -> None:
    transition_reconstruction_profile_probe()
    rdkit_smiles_profile_probe()
    sqlite_execute_profile_probe()
    trace_serialization_profile_probe()
    model_inference_profile_probe()


def _context(tmp_path: Path, *, resume: bool = False) -> stage.Context:
    roots = {
        name: tmp_path / name
        for name in ("project", "step0", "external", "persistent", "fast")
    }
    for path in roots.values():
        path.mkdir(parents=True)
    python = tmp_path / "python"
    python.write_bytes(b"python")
    return stage.Context(
        project=roots["project"],
        step0=roots["step0"],
        external=roots["external"],
        persistent=roots["persistent"],
        fast=roots["fast"],
        python=python,
        resume=resume,
    )


def _snapshot(tmp_path: Path) -> stage.InputSnapshot:
    primary = tmp_path / "primary"
    static = tmp_path / "static"
    primary.mkdir()
    static.mkdir()
    (primary / "payload").write_text("primary\n", encoding="utf-8")
    (static / "payload").write_text("static\n", encoding="utf-8")
    primary_manifest = primary / "MANIFEST.sha256"
    static_manifest = static / "MANIFEST.sha256"
    primary_manifest.write_text(
        f"{stage.sha256_file(primary / 'payload')}  payload\n", encoding="utf-8"
    )
    static_manifest.write_text(
        f"{stage.sha256_file(static / 'payload')}  payload\n", encoding="utf-8"
    )
    required = tmp_path / "required.sha256"
    required.write_text(
        f"{stage.sha256_file(static / 'payload')}  {str(static / 'payload').lstrip('/')}\n",
        encoding="utf-8",
    )
    return stage.InputSnapshot(
        primary_root=primary,
        primary_manifest=primary_manifest,
        primary_digest=stage.sha256_file(primary_manifest),
        static_manifest=static_manifest,
        static_digest=stage.sha256_file(static_manifest),
        required_static_manifest=required,
        required_static_digest=stage.sha256_file(required),
        required_static_source_root=tmp_path,
    )


def _exact_recovery_fixture(
    root: Path,
    *,
    chunk_count: int,
    row_count: int,
    candidate_count: int,
    selected_transition_count: int | None = None,
    alias_count: int,
) -> None:
    transition_count = (
        candidate_count
        if selected_transition_count is None
        else selected_transition_count
    )
    trace = root / "trace"
    chunks_root = trace / "chunks"
    chunks_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    remaining = row_count
    for index in range(chunk_count):
        path = chunks_root / f"part-{index:06d}.jsonl"
        path.write_text(f'{{"chunk":{index}}}\n', encoding="utf-8")
        chunk_rows = min(512, remaining)
        remaining -= chunk_rows
        rows.append(
            {
                "index": index,
                "path": f"chunks/{path.name}",
                "row_count": chunk_rows,
                "bytes": path.stat().st_size,
                "sha256": stage.sha256_file(path),
            }
        )
    assert remaining == 0
    stage.atomic_write_json(
        trace / "selected_action_trace_manifest.json",
        {"row_count": row_count, "chunks": rows},
    )
    stage.atomic_write_json(
        trace / "trace_summary.json",
        {
            "algorithm_rerun": False,
            "selected_transition_count": transition_count,
            "recorded_action_present_count": transition_count,
            "recorded_action_replay_ok_count": transition_count,
            "recorded_action_replay_mismatch_count": 0,
            "legacy_missing_action_count": 0,
            "legacy_inference_called_count": 0,
            "legacy_inference_ambiguous_count": 0,
        },
    )
    graph_state = root / "graph_state"
    graph_state.mkdir()
    (graph_state / "authoritative_graph_store.sqlite3").write_bytes(b"sqlite")
    stage.atomic_write_json(
        root / "run_manifest.json",
        {
            "algorithm_rerun": False,
            "counterfactual_candidate_count": candidate_count,
        },
    )
    stage.atomic_write_json(
        root / "frozen_payload_closure_audit.json",
        {
            "closure_complete": True,
            "post_write_reload_verified": True,
            "original_trace_hash_roundtrip_verified": True,
            "canonical_graph_records_persisted": True,
            "alias_to_canonical_persisted": True,
            "original_trace_hashes_persisted": True,
            "unresolved_hash_count": 0,
            "graph_replacement_count": 0,
            "canonical_graph_record_count": max(candidate_count, 1),
            "alias_count": alias_count,
            "original_trace_hash_count": max(candidate_count, 1),
            "original_trace_hash_roundtrip_count": max(candidate_count, 1),
            "alias_cycle_count": 0,
            "dangling_alias_count": 0,
            "selected_trace_row_count": row_count,
        },
    )


class _FakeSqliteConnection:
    def __init__(self, entries: int) -> None:
        self.entries = entries

    def execute(self, statement: str) -> "_FakeSqliteConnection":
        self.statement = statement
        return self

    def fetchone(self) -> tuple[object]:
        return ("ok",) if "integrity_check" in self.statement else (self.entries,)

    def close(self) -> None:
        return None


def _source_recovery_validation(
    *, sqlite_entries: int, trace_chunks: int, trace_rows: int
) -> dict[str, object]:
    return {
        "completed_steps": 50_000,
        "backing_store_audit": {"entry_count": sqlite_entries},
        "selected_trace_audit": {
            "chunk_count": trace_chunks,
            "row_count": trace_rows,
        },
    }


def test_mut_formal_recovery_gate_consumes_all_exact_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "mut"
    _exact_recovery_fixture(
        generation,
        chunk_count=449,
        row_count=229_752,
        candidate_count=100_235,
        selected_transition_count=224_690,
        alias_count=0,
    )
    monkeypatch.setattr(
        stage.sqlite3, "connect", lambda *_args, **_kwargs: _FakeSqliteConnection(124_206)
    )
    evidence = stage._validate_recovered_generation_exact_gate(
        dataset="mutagenicity",
        generation=generation,
        source_validation=_source_recovery_validation(
            sqlite_entries=124_206, trace_chunks=449, trace_rows=229_752
        ),
    )
    assert evidence["candidate_count"] == 100_235
    assert evidence["selected_transition_count"] == 224_690
    assert evidence["recorded_action_replay_ok_count"] == 224_690
    summary = generation / "trace/trace_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["recorded_action_replay_ok_count"] = 224_689
    stage.atomic_write_json(summary, payload)
    with pytest.raises(stage.StageError, match="formal recovery counts changed"):
        stage._validate_recovered_generation_exact_gate(
            dataset="mutagenicity",
            generation=generation,
            source_validation=_source_recovery_validation(
                sqlite_entries=124_206, trace_chunks=449, trace_rows=229_752
            ),
        )

    # Restore the valid replay count, then prove that the old conflation of
    # unique payload candidates with selected transitions fails closed.
    payload["recorded_action_replay_ok_count"] = 224_690
    stage.atomic_write_json(summary, payload)
    run_manifest = generation / "run_manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["counterfactual_candidate_count"] = 224_690
    stage.atomic_write_json(run_manifest, payload)
    with pytest.raises(stage.StageError, match="formal recovery counts changed"):
        stage._validate_recovered_generation_exact_gate(
            dataset="mutagenicity",
            generation=generation,
            source_validation=_source_recovery_validation(
                sqlite_entries=124_206, trace_chunks=449, trace_rows=229_752
            ),
        )


def test_aids_formal_recovery_gate_requires_alias_roundtrip_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "aids"
    _exact_recovery_fixture(
        generation,
        chunk_count=450,
        row_count=229_976,
        candidate_count=7,
        alias_count=3,
    )
    monkeypatch.setattr(
        stage.sqlite3,
        "connect",
        lambda *_args, **_kwargs: _FakeSqliteConnection(87_160),
    )
    stage._validate_recovered_generation_exact_gate(
        dataset="aids",
        generation=generation,
        source_validation=_source_recovery_validation(
            sqlite_entries=87_160, trace_chunks=450, trace_rows=229_976
        ),
    )
    closure = generation / "frozen_payload_closure_audit.json"
    payload = json.loads(closure.read_text(encoding="utf-8"))
    payload["dangling_alias_count"] = 1
    stage.atomic_write_json(closure, payload)
    with pytest.raises(stage.StageError, match="frozen-closure gate failed"):
        stage._validate_recovered_generation_exact_gate(
            dataset="aids",
            generation=generation,
            source_validation=_source_recovery_validation(
                sqlite_entries=87_160, trace_chunks=450, trace_rows=229_976
            ),
        )


def test_aids_formal_recovery_gate_accepts_persisted_empty_alias_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "aids-direct"
    _exact_recovery_fixture(
        generation,
        chunk_count=450,
        row_count=229_976,
        candidate_count=7,
        alias_count=0,
    )
    monkeypatch.setattr(
        stage.sqlite3,
        "connect",
        lambda *_args, **_kwargs: _FakeSqliteConnection(87_160),
    )

    evidence = stage._validate_recovered_generation_exact_gate(
        dataset="aids",
        generation=generation,
        source_validation=_source_recovery_validation(
            sqlite_entries=87_160, trace_chunks=450, trace_rows=229_976
        ),
    )

    assert evidence["alias_count"] == 0
    assert evidence["alias_to_canonical_persisted"] is True

    closure = generation / "frozen_payload_closure_audit.json"
    payload = json.loads(closure.read_text(encoding="utf-8"))
    payload.pop("alias_count")
    stage.atomic_write_json(closure, payload)
    with pytest.raises(stage.StageError, match="frozen-closure gate failed"):
        stage._validate_recovered_generation_exact_gate(
            dataset="aids",
            generation=generation,
            source_validation=_source_recovery_validation(
                sqlite_entries=87_160, trace_chunks=450, trace_rows=229_976
            ),
        )


def test_external_vendor_lineage_rejects_dirty_code_and_binds_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    subprocess.run(["git", "init", "-q", str(context.external)], check=True)
    subprocess.run(
        ["git", "-C", str(context.external), "config", "user.email", "x@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(context.external), "config", "user.name", "test"],
        check=True,
    )
    tracked = context.external / "upstream.py"
    tracked.write_text("PINNED = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(context.external), "add", "upstream.py"], check=True)
    subprocess.run(
        ["git", "-C", str(context.external), "commit", "-qm", "pinned"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(context.external), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(stage, "UPSTREAM_COMMIT", commit)
    monkeypatch.setattr(stage, "_repair_code_closure_sha256", lambda _value: "7" * 64)
    monkeypatch.setattr(stage, "_git_head", lambda root: commit)

    clean = stage._current_code_lineage(context)
    assert clean["external_provenance_sha256"] == "ABSENT"
    assert len(clean["external_comrecgc_tree"]) == 40

    tracked.write_text("PINNED = 2\n", encoding="utf-8")
    with pytest.raises(stage.StageError, match="worktree has tracked/staged"):
        stage._current_code_lineage(context)
    tracked.write_text("PINNED = 1\n", encoding="utf-8")

    unexpected = context.external / "untracked_code.py"
    unexpected.write_text("bad = True\n", encoding="utf-8")
    with pytest.raises(stage.StageError, match="unapproved untracked"):
        stage._current_code_lineage(context)
    unexpected.unlink()

    provenance = context.external / "vendor_manifest.json"
    provenance.write_text('{"commit":"pinned"}\n', encoding="utf-8")
    bound = stage._current_code_lineage(context)
    assert bound["external_provenance_sha256"] == stage.sha256_file(provenance)


def test_bace_generation_exact_formal_wiring_and_resume(tmp_path: Path) -> None:
    context = _context(tmp_path)
    command = stage.bace_generation_command(context, resume=False)
    assert command[:3] == [
        str(context.python),
        str(context.project / "scripts/baselines/comrecgc/run_generation.py"),
        "--config",
    ]
    expected_pairs = {
        "--route": "project",
        "--dataset": "bace",
        "--mode": "full",
        "--parent-limit": "360",
        "--device": "cuda:0",
        "--batch-size": "128",
        "--checkpoint-interval-steps": "500",
        "--checkpoint-keep-last": "2",
        "--progress-interval-steps": "25",
        "--storage-min-free-gib": "8",
        "--storage-min-free-ratio": ".10",
        "--storage-min-free-inodes": "100000",
    }
    for option, expected in expected_pairs.items():
        assert command[command.index(option) + 1] == expected
    assert command[command.index("--trace-output-dir") + 1] == str(
        context.persistent
        / "outputs/bace_comrecgc/generation_resume_metadata/trace"
    )
    assert command[command.index("--checkpoint-root") + 1].startswith(
        str(context.fast)
    )
    assert command[command.index("--checkpoint-mirror-root") + 1].startswith(
        str(context.persistent)
    )
    assert "--resume" not in command
    assert stage.bace_generation_command(context, resume=True) == [*command, "--resume"]


def test_bace_generation_exact_gate_binds_trace_closure_and_checkpoint_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    output = context.persistent / "outputs/bace_comrecgc"
    generation = output / "generation"
    trace = generation / "trace"
    trace.mkdir(parents=True)
    chunk = trace / "selected_action_trace_chunks/part-000000.jsonl"
    chunk.parent.mkdir()
    chunk.write_text('{"move_index":1}\n', encoding="utf-8")
    stage.atomic_write_json(
        trace / "selected_action_trace_manifest.json",
        {
            "row_count": 1,
            "chunks": [
                {
                    "index": 0,
                    "path": "selected_action_trace_chunks/part-000000.jsonl",
                    "row_count": 1,
                    "bytes": chunk.stat().st_size,
                    "sha256": stage.sha256_file(chunk),
                }
            ],
        },
    )
    trace_manifest = trace / "selected_action_trace_manifest.json"
    stage.atomic_write_json(
        trace / "_TRACE_COMPLETE.json",
        {
            "trace_complete": True,
            "selected_trace_manifest_sha256": stage.sha256_file(trace_manifest),
        },
    )
    stage.atomic_write_json(
        generation / "progress.json",
        {
            "run_complete": True,
            "current_step": 50_000,
            "completed_step": 50_000,
            "next_step": 50_001,
            "total_steps": 50_000,
            "last_checkpoint_step": 50_000,
        },
    )
    stage.atomic_write_json(
        generation / "run_manifest.json",
        {
            "run_complete": True,
            "algorithm_rerun": True,
            "traversed_step_count": 50_000,
            "generation_checkpoint_interval_steps": 500,
        },
    )
    stage.atomic_write_json(
        generation / "resolved_config.json",
        {"generation_checkpoint_interval_steps": 500},
    )
    stage.atomic_write_json(
        generation / "frozen_payload_closure_audit.json",
        {
            "closure_complete": True,
            "post_write_reload_verified": True,
            "unresolved_hash_count": 0,
        },
    )
    metadata = output / "generation_resume_metadata"
    metadata.mkdir(parents=True)
    (metadata / "resolved_config.json").write_bytes(
        (generation / "resolved_config.json").read_bytes()
    )
    mirror = output / "generation_checkpoint_mirror"
    history = mirror / "retention_history"
    history.mkdir(parents=True)
    digest = "d" * 64
    for step_number in range(500, 49_001, 500):
        stage.atomic_write_json(
            history / f"step-{step_number:012d}.json",
            {
                "schema_version": "comrecgc_generation_checkpoint_retention_v1",
                "checkpoint_mirrored": True,
                "completed_step": step_number,
                "checkpoint_digest": digest,
            },
        )
    live = []
    for step_number in (49_500, 50_000):
        checkpoint = mirror / f"step-{step_number:012d}"
        checkpoint.mkdir()
        live.append(
            SimpleNamespace(
                completed_step=step_number,
                checkpoint_digest=digest,
                checkpoint_dir=checkpoint,
            )
        )
    module = SimpleNamespace(
        RETENTION_HISTORY_DIRNAME="retention_history",
        list_generation_checkpoints=lambda _root: [row.checkpoint_dir for row in live],
        validate_generation_checkpoint=lambda value, **_kwargs: (
            live[-1]
            if Path(value) == mirror
            else next(row for row in live if row.checkpoint_dir == Path(value))
        ),
    )
    monkeypatch.setattr(stage, "_checkpoint_module", lambda _context: module)
    monkeypatch.setattr(
        stage,
        "_validate_mirrored_checkpoint",
        lambda _module, checkpoint: next(
            row for row in live if row.checkpoint_dir == checkpoint
        ),
    )
    evidence = stage._validate_bace_generation_exact_gate(
        context=context,
        generation=generation,
        mirror=mirror,
        metadata=metadata,
    )
    assert evidence["published_checkpoint_count"] == 100
    assert evidence["duplicate_step_count"] == 0
    assert evidence["skipped_step_count"] == 0
    assert evidence["latest_checkpoint_digest"] == digest

    (history / "step-000000010000.json").unlink()
    with pytest.raises(stage.StageError, match="not contiguous"):
        stage._validate_bace_generation_exact_gate(
            context=context,
            generation=generation,
            mirror=mirror,
            metadata=metadata,
        )


def test_bace_globalgce_exact_preserved_input_wiring(tmp_path: Path) -> None:
    context = _context(tmp_path)
    command = stage.bace_globalgce_command(context)
    assert command[command.index("--candidate-path") + 1] == str(
        context.bace_input / "globalgce_selector/selected_top20_for_eval.csv"
    )
    assert command[command.index("--selection-manifest") + 1] == str(
        context.bace_input / "globalgce_selector/frozen_selection.json"
    )
    assert command[command.index("--reference-artifact-root") + 1].endswith(
        "/common4/ours"
    )
    assert command[command.index("--action-semantics-version") + 1] == (
        "connected_sanitized_residual_v1"
    )
    assert command[command.index("--match-selection-policy") + 1] == (
        "existential_min_wnode_among_valid_connected_strict_flips_v1"
    )
    assert "--resume" not in command
    assert stage.bace_globalgce_command(context, resume=True) == [*command, "--resume"]


def test_downstream_and_bace_artifact_commands_match_scientific_wrappers(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    base = context.persistent / "outputs/mut_comrecgc"
    common = stage._common_recourse_command(context, "mutagenicity", base)
    chemistry = stage._chemistry_command(context, "mutagenicity", base)
    evaluation = stage._slot_eval_command(context, "mutagenicity", base)
    assert Path(common[1]).name == "run_common_recourse.py"
    assert common[common.index("--parent-limit") + 1] == "1448"
    assert common[common.index("--distance-checkpoint") + 1].endswith(
        "/outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt"
    )
    assert Path(chemistry[1]).name == "audit_mutagenicity_chemistry.py"
    assert chemistry[chemistry.index("--trace-lineage-path") + 1].endswith(
        "/generation/trace/candidate_action_lineage.json"
    )
    assert Path(evaluation[1]).name == "run_slot_unified_eval.py"
    assert evaluation[evaluation.index("--expected-parent-count") + 1] == "217"
    paper = context.persistent / "outputs/bace_comrecgc/paper/comrecgc"
    artifact = stage._bace_artifact_gate_command(context, paper)
    assert Path(artifact[1]).name == "audit_bace_artifacts.py"
    assert artifact[artifact.index("--root") + 1] == str(paper)
    assert artifact[artifact.index("--thresholds-json") + 1] == str(
        context.bace_input / "common4/thresholds.json"
    )
    assert artifact[artifact.index("--expected-parent-count") + 1] == "116"


def test_freeze_only_requires_environment_and_never_invokes_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    with pytest.raises(stage.StageError, match="DISALLOW_GENERATION=1"):
        stage._run_freeze(context, "mutagenicity")
    monkeypatch.setenv("DISALLOW_GENERATION", "1")
    monkeypatch.setattr(
        stage, "_require_lineage_smoke_gate", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        stage,
        "_current_code_lineage",
        lambda _context: {
            "repair_code_closure_sha256": "7" * 64,
            "project_commit": "8" * 40,
            "external_comrecgc_commit": stage.UPSTREAM_COMMIT,
        },
    )
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(stage, "_all_input_gates", lambda *_args: snapshot)
    monkeypatch.setattr(
        stage,
        "_verify_all_input_gates",
        lambda _value: {
            "primary": snapshot.primary_digest,
            "static_project": snapshot.static_digest,
            "required_static": snapshot.required_static_digest,
        },
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(list(command))
        audit = Path(command[command.index("--audit-output") + 1])
        audit.parent.mkdir(parents=True, exist_ok=True)
        if "--validate-only" in command:
            stage.atomic_write_json(audit, {"FREEZE_ONLY_RECOVERY_SAFE": True})
            return
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        stage.atomic_write_json(
            output / "_RUN_COMPLETE.json",
            {"run_complete": True, "freeze_only_recovery": True},
        )
        stage.atomic_write_json(
            output / "run_manifest.json", {"algorithm_rerun": False}
        )
        stage.atomic_write_json(
            output / "frozen_payload_closure_audit.json",
            {"closure_complete": True, "post_write_reload_verified": True},
        )
        stage.atomic_write_json(
            output / "freeze_only_recovery.json",
            {
                "algorithm_rerun": False,
                "recovery_completed": True,
                "completed_steps": 50_000,
            },
        )
        stage.atomic_write_json(
            audit,
            {
                "FREEZE_ONLY_RECOVERY_SAFE": True,
                "recovery_completed": True,
                "algorithm_rerun": False,
            },
        )

    monkeypatch.setattr(stage, "_run_checked", fake_run)
    monkeypatch.setattr(
        stage,
        "_validate_recovered_generation_exact_gate",
        lambda **_kwargs: {"formal_exact_gate": True},
    )
    stage._run_freeze(context, "mutagenicity")
    assert len(commands) == 2
    assert "--validate-only" in commands[0]
    assert "--output-dir" in commands[1]
    assert all("run_generation.py" not in command for command in commands)
    sentinel = (
        context.persistent / "outputs/mut_comrecgc/MUT_FREEZE_RECOVERY_PASS.json"
    )
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["generation_rerun_performed"] is False
    assert payload["input_manifests_sha256"]["required_static"] == (
        snapshot.required_static_digest
    )


def test_run_checked_sets_hashseed_before_interpreter_and_uses_exact_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class Process:
        returncode = 0

        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured["argv"] = argv
            captured.update(kwargs)
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def send_signal(self, _signum: int) -> None:
            raise AssertionError("unexpected signal")

    monkeypatch.setattr(stage.subprocess, "Popen", Process)
    monkeypatch.setattr(stage.time, "sleep", lambda _value: None)
    project = tmp_path / "project"
    project.mkdir()
    command = ["/safe/python", "safe_stage.py", "--dataset", "bace"]
    stage._run_checked(command, cwd=project)
    assert captured["argv"] == command
    assert captured["env"]["PYTHONHASHSEED"] == "0"  # type: ignore[index]


def _assert_exact_child_reaped(pid: int) -> None:
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_run_checked_monitor_failure_terminates_term_resistant_group_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "child.ready"
    term_seen = tmp_path / "child.term"
    child_code = (
        "import os,pathlib,signal,sys,time; "
        "ready=pathlib.Path(sys.argv[1]); term=pathlib.Path(sys.argv[2]); "
        "signal.signal(signal.SIGTERM, lambda *_: term.write_text('TERM\\n')); "
        "ready.write_text(str(os.getpid())); "
        "time.sleep(600)"
    )
    original_cleanup = stage._terminate_child_process_group
    monkeypatch.setattr(
        stage,
        "_terminate_child_process_group",
        lambda process: original_cleanup(process, grace_seconds=0.25),
    )

    class MonitorFailure(RuntimeError):
        pass

    observed_pid: list[int] = []

    def fail_after_child_is_ready() -> None:
        if not ready.is_file():
            return
        pid = int(ready.read_text(encoding="utf-8"))
        observed_pid[:] = [pid]
        assert os.getpgid(pid) == pid
        raise MonitorFailure("primary-monitor-failure")

    with pytest.raises(MonitorFailure, match="primary-monitor-failure"):
        stage._run_checked(
            [sys.executable, "-c", child_code, str(ready), str(term_seen)],
            cwd=tmp_path,
            monitor=fail_after_child_is_ready,
        )
    assert observed_pid
    assert term_seen.read_text(encoding="utf-8") == "TERM\n"
    _assert_exact_child_reaped(observed_pid[0])


def test_profile_metric_failure_terminates_group_and_reaps_without_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    ready = tmp_path / "profile-child.ready"
    child_code = (
        "import os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(600)"
    )

    class MetricFailure(RuntimeError):
        pass

    class FailingCollector:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def observe(self, **_kwargs: object) -> None:
            deadline = time.monotonic() + 3
            while not ready.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not ready.is_file():
                raise AssertionError("profile child did not become ready")
            pid = int(ready.read_text(encoding="utf-8"))
            assert os.getpgid(pid) == pid
            raise MetricFailure("primary-metric-failure")

    monkeypatch.setattr(stage, "_ProfileObservationCollector", FailingCollector)
    mirror = tmp_path / "profile-mirror"
    mirror.mkdir()
    with pytest.raises(MetricFailure, match="primary-metric-failure"):
        stage._run_profile_until_checkpoint(
            context,
            command=[sys.executable, "-c", child_code, str(ready)],
            mirror_root=mirror,
            target_step=500,
        )
    _assert_exact_child_reaped(int(ready.read_text(encoding="utf-8")))


def test_resume_is_added_only_to_supported_nonempty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, resume=True)
    input_manifests = {
        "primary": "1" * 64,
        "static_project": "2" * 64,
        "required_static": "3" * 64,
    }
    monkeypatch.setattr(
        stage,
        "_scientific_reuse_lineage",
        lambda **kwargs: {
            "run_id": stage.RUN_ID,
            "command_sha256": "4" * 64,
            "input_manifests_sha256": dict(kwargs["input_manifests"]),
            "scientific_environment_sha256": "a" * 64,
            "repair_code_closure_sha256": "5" * 64,
            "project_commit": "6" * 40,
            "external_comrecgc_commit": stage.UPSTREAM_COMMIT,
        },
    )
    monkeypatch.setattr(
        stage,
        "_assert_stage_lineage_unchanged",
        lambda _context: {
            "repair_code_closure_sha256": "5" * 64,
            "project_commit": "6" * 40,
            "external_comrecgc_commit": stage.UPSTREAM_COMMIT,
        },
    )
    commands: list[list[str]] = []
    active_marker: list[Path] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        stage.atomic_write_json(active_marker[0], {"run_complete": True})

    monkeypatch.setattr(stage, "_run_checked", fake_run)
    empty = tmp_path / "empty"
    empty.mkdir()
    marker = empty / "_RUN_COMPLETE.json"
    active_marker[:] = [marker]
    stage._run_or_reuse(
        context=context,
        output=empty,
        marker=marker,
        command=["python", "stage.py"],
        resumable=True,
        input_manifests=input_manifests,
    )
    assert commands[-1] == ["python", "stage.py"]
    reuse_manifest = empty / "_AUTODL_REUSE_MANIFEST.sha256"
    assert reuse_manifest.is_file()
    stage.verify_sha256_manifest(empty, reuse_manifest)
    extra = empty / "unlisted.bin"
    extra.write_bytes(b"must-not-be-adopted")
    with pytest.raises(stage.StageError, match="inventory mismatch"):
        stage._run_or_reuse(
            context=context,
            output=empty,
            marker=marker,
            command=["python", "stage.py"],
            resumable=True,
            input_manifests=input_manifests,
        )
    extra.unlink()
    changed_inputs = {**input_manifests, "primary": "9" * 64}
    with pytest.raises(stage.StageError, match="reuse proof is stale"):
        stage._run_or_reuse(
            context=context,
            output=empty,
            marker=marker,
            command=["python", "stage.py"],
            resumable=True,
            input_manifests=changed_inputs,
        )
    (empty / "_RUN_COMPLETE.json").write_text(
        '{"run_complete":true,"tampered":true}\n', encoding="utf-8"
    )
    with pytest.raises(stage.StageError, match="SHA256 mismatch"):
        stage._run_or_reuse(
            context=context,
            output=empty,
            marker=marker,
            command=["python", "stage.py"],
            resumable=True,
            input_manifests=input_manifests,
        )
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "progress.json").write_text("{}\n", encoding="utf-8")
    active_marker[:] = [partial / "_RUN_COMPLETE.json"]
    stage._run_or_reuse(
        context=context,
        output=partial,
        marker=partial / "_RUN_COMPLETE.json",
        command=["python", "stage.py"],
        resumable=True,
        input_manifests=input_manifests,
    )
    assert commands[-1] == ["python", "stage.py", "--resume"]
    stage._run_or_reuse(
        context=context,
        output=partial,
        marker=partial / "_RUN_COMPLETE.json",
        command=["python", "stage.py"],
        resumable=True,
        input_manifests=input_manifests,
    )
    assert commands.count(["python", "stage.py", "--resume"]) == 1
    (partial / "_RUN_COMPLETE.json").unlink()
    with pytest.raises(stage.StageError, match="non-resumable"):
        stage._run_or_reuse(
            context=context,
            output=partial,
            marker=partial / "_RUN_COMPLETE.json",
            command=["python", "stage.py"],
            resumable=False,
            input_manifests=input_manifests,
        )

    marker_only = tmp_path / "marker-only"
    marker_only.mkdir()
    stage.atomic_write_json(
        marker_only / "_RUN_COMPLETE.json", {"run_complete": True}
    )
    with pytest.raises(stage.StageError, match="REUSE_PROOF"):
        stage._run_or_reuse(
            context=context,
            output=marker_only,
            marker=marker_only / "_RUN_COMPLETE.json",
            command=["python", "stage.py"],
            resumable=True,
            input_manifests=input_manifests,
        )


def test_profile_checkpoint_monitor_signals_once_and_persists_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    stage.atomic_write_json(mirror / "LATEST", {"completed_step": 500})
    checkpoint = mirror / "step-000000000500"
    checkpoint.mkdir()
    stage.atomic_write_json(
        checkpoint / "_CHECKPOINT_MIRRORED.json",
        {
            "checkpoint_mirrored": True,
            "checkpoint_digest": "d" * 64,
            "completed_step": 500,
        },
    )
    source_config = tmp_path / "fast/resolved_config.json"
    source_config.parent.mkdir(exist_ok=True)
    source_config.write_text('{"mode":"full"}\n', encoding="utf-8")
    persistent_config = tmp_path / "persistent-config/resolved_config.json"
    signals: list[int] = []

    class Process:
        returncode = 143

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls < 3 else self.returncode

        def send_signal(self, signum: int) -> None:
            signals.append(signum)

    validation = SimpleNamespace(
        checkpoint_digest="d" * 64,
        checkpoint_dir=checkpoint,
        completed_step=500,
    )
    module = SimpleNamespace(
        MIRRORED_FILENAME="_CHECKPOINT_MIRRORED.json",
        validate_generation_checkpoint=lambda *_args, **_kwargs: validation
    )
    monkeypatch.setattr(stage.subprocess, "Popen", Process)
    # Keep this process-control unit test independent from whether the host
    # happens to provide nvidia-smi.  Patching subprocess.Popen above is
    # intentionally narrow to the scientific child and cannot also serve as
    # the context-manager implementation used by subprocess.run.
    monkeypatch.setattr(
        stage,
        "_read_gpu_observation",
        lambda: (None, "gpu sampling is outside this unit test"),
    )
    monkeypatch.setattr(stage, "_checkpoint_module", lambda _context: module)
    monkeypatch.setattr(stage.time, "sleep", lambda _value: None)
    result = stage._run_profile_until_checkpoint(
        context,
        command=["python", "formal.py"],
        mirror_root=mirror,
        target_step=500,
        termination_signal=signal.SIGTERM,
        resolved_config_source=source_config,
        resolved_config_destination=persistent_config,
    )
    assert signals == [signal.SIGTERM]
    assert result["resolved_config"]["sha256"] == stage.sha256_file(source_config)


def test_standardized_freeze_validates_manifest_and_source_digest_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    gate = tmp_path / "gate"
    output = tmp_path / "standardized"
    source.mkdir()
    gate.mkdir()
    output.mkdir()
    (source / "run_manifest.json").write_text(
        '{"run_complete":true}\n', encoding="utf-8"
    )
    (gate / "gate_result.json").write_text(
        '{"audit_passed":true}\n', encoding="utf-8"
    )
    artifact = output / "summary.json"
    artifact.write_text('{"ok":true}\n', encoding="utf-8")
    stage.atomic_write_json(
        output / "freeze_manifest.json",
        {
            "source_run_manifest_sha256": stage.sha256_file(
                source / "run_manifest.json"
            ),
            "source_gate_result_sha256": stage.sha256_file(
                gate / "gate_result.json"
            ),
            "files": {
                "summary.json": {
                    "bytes": artifact.stat().st_size,
                    "sha256": stage.sha256_file(artifact),
                }
            },
        },
    )
    stage.atomic_write_json(
        output / "_FINALIZED.json",
        {
            "finalized": True,
            "gate_passed": True,
            "freeze_manifest_sha256": stage.sha256_file(
                output / "freeze_manifest.json"
            ),
        },
    )
    stage._validate_standardized_freeze(output, source=source, gate=gate)
    artifact.write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(stage.StageError, match="inventory mismatch"):
        stage._validate_standardized_freeze(output, source=source, gate=gate)


def test_abrupt_kill_waits_for_post_checkpoint_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    stage.atomic_write_json(mirror / "LATEST", {"completed_step": 500})
    checkpoint = mirror / "step-000000000500"
    checkpoint.mkdir()
    stage.atomic_write_json(
        checkpoint / "_CHECKPOINT_MIRRORED.json",
        {
            "checkpoint_mirrored": True,
            "checkpoint_digest": "d" * 64,
            "completed_step": 500,
        },
    )
    progress = tmp_path / "fast/progress.json"
    progress.parent.mkdir(exist_ok=True)
    signals: list[tuple[int, int]] = []

    class Process:
        returncode = -signal.SIGKILL

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            observed = 500 if self.polls == 1 else 525
            stage.atomic_write_json(progress, {"completed_step": observed})
            return None if self.polls < 3 else self.returncode

        def send_signal(self, signum: int) -> None:
            current = stage.read_json(progress)["completed_step"]
            signals.append((signum, int(current)))

    validation = SimpleNamespace(
        checkpoint_digest="d" * 64,
        checkpoint_dir=checkpoint,
        completed_step=500,
    )
    module = SimpleNamespace(
        MIRRORED_FILENAME="_CHECKPOINT_MIRRORED.json",
        validate_generation_checkpoint=lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(stage.subprocess, "Popen", Process)
    monkeypatch.setattr(
        stage,
        "_read_gpu_observation",
        lambda: (None, "gpu sampling is outside this unit test"),
    )
    monkeypatch.setattr(stage, "_checkpoint_module", lambda _context: module)
    monkeypatch.setattr(stage.time, "sleep", lambda _value: None)
    result = stage._run_profile_until_checkpoint(
        context,
        command=["python", "formal.py"],
        mirror_root=mirror,
        target_step=500,
        termination_signal=signal.SIGKILL,
        progress_path=progress,
        stop_after_progress_step=525,
    )
    assert signals == [(signal.SIGKILL, 525)]
    assert result["observed_process_step_at_signal"] == 525
    assert result["target_checkpoint_step"] == 500


def test_profile_observations_record_progress_gpu_cpu_iowait_and_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([10.0, 11.0])
    cpu = iter([{"cpu_seconds": 1.0}, {"cpu_seconds": 1.5}])
    process_io = iter(
        [
            {"read_bytes": 100, "write_bytes": 200},
            {"read_bytes": 160, "write_bytes": 290},
        ]
    )
    system = iter(
        [
            {"total_jiffies": 1_000, "iowait_jiffies": 50},
            {"total_jiffies": 1_200, "iowait_jiffies": 70},
        ]
    )
    gpu = iter(
        [
            [
                {
                    "gpu_index": 2,
                    "utilization_percent": 40.0,
                    "memory_used_mib": 1_024.0,
                    "memory_total_mib": 81_920.0,
                }
            ],
            [
                {
                    "gpu_index": 2,
                    "utilization_percent": 60.0,
                    "memory_used_mib": 2_048.0,
                    "memory_total_mib": 81_920.0,
                }
            ],
        ]
    )
    monkeypatch.setattr(stage.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        stage, "_read_process_cpu", lambda _pid: (next(cpu), None)
    )
    monkeypatch.setattr(
        stage, "_read_process_io", lambda _pid: (next(process_io), None)
    )
    monkeypatch.setattr(stage, "_read_system_iowait", lambda: (next(system), None))
    monkeypatch.setattr(stage, "_read_gpu_observation", lambda: (next(gpu), None))
    collector = stage._ProfileObservationCollector(
        pid=123, started_monotonic=10.0, metric_interval_seconds=1.0
    )
    collector.observe(progress_step=500, force_metrics=True)
    collector.observe(progress_step=510, force_metrics=True)
    result = collector.finish(elapsed_seconds=1.0)
    assert result["schema_version"] == stage.PROFILE_OBSERVATION_SCHEMA
    assert result["progress"]["per_step"]["seconds_per_step"] == pytest.approx(0.1)
    assert result["process_cpu"]["utilization_percent"]["mean"] == pytest.approx(
        50.0
    )
    assert result["process_io"]["byte_delta"] == {
        "status": "OBSERVED",
        "read_bytes": 60,
        "write_bytes": 90,
    }
    assert result["system_iowait"]["percent"]["mean"] == pytest.approx(10.0)
    assert result["gpu"]["devices"]["2"]["peak_memory_used_mib"] == 2_048.0


def test_gpu_observation_is_scoped_to_assigned_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(stage.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    result = SimpleNamespace(
        returncode=0,
        stdout=(
            "0, GPU-zero, 90, 100, 81920\n"
            "2, GPU-two, 40, 2048, 81920\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(stage.subprocess, "run", lambda *_args, **_kwargs: result)
    rows, reason = stage._read_gpu_observation()
    assert reason is None
    assert rows == [
        {
            "gpu_index": 2,
            "gpu_uuid": "GPU-two",
            "measurement_scope": "CUDA_VISIBLE_DEVICES",
            "utilization_percent": 40.0,
            "memory_used_mib": 2048.0,
            "memory_total_mib": 81920.0,
        }
    ]


def test_profile_observations_explicitly_mark_unavailable_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage.time, "monotonic", lambda: 5.0)
    monkeypatch.setattr(
        stage,
        "_read_process_cpu",
        lambda _pid: (None, "proc_process_stat_unavailable"),
    )
    monkeypatch.setattr(
        stage,
        "_read_process_io",
        lambda _pid: (None, "proc_process_io_unavailable"),
    )
    monkeypatch.setattr(
        stage,
        "_read_system_iowait",
        lambda: (None, "proc_system_stat_unavailable"),
    )
    monkeypatch.setattr(
        stage,
        "_read_gpu_observation",
        lambda: (None, "nvidia_smi_unavailable"),
    )
    collector = stage._ProfileObservationCollector(
        pid=None, started_monotonic=5.0, metric_interval_seconds=1.0
    )
    collector.observe(force_metrics=True)
    result = collector.finish(elapsed_seconds=0.0)
    for field in (
        "progress",
        "gpu",
        "process_cpu",
        "system_iowait",
        "process_io",
    ):
        assert result[field]["status"] == "NOT_OBSERVED"


def test_pstats_categories_record_real_calls_total_and_cumulative(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "probe.cprofile"
    profiler = cProfile.Profile()
    profiler.enable()
    graph_diff_profile_probe()
    transition_reconstruction_profile_probe()
    canonical_hash_profile_probe()
    rdkit_smiles_profile_probe()
    sqlite_execute_profile_probe()
    trace_serialization_profile_probe()
    model_inference_profile_probe()
    profiler.disable()
    profiler.dump_stats(profile_path)

    result = stage._aggregate_pstats([profile_path])
    assert result["status"] == "OBSERVED"
    assert set(result["categories"]) == set(stage.PROFILE_FUNCTION_PATTERNS)
    for category in stage.PROFILE_FUNCTION_PATTERNS:
        aggregation = result["categories"][category]
        assert aggregation["status"] == "OBSERVED"
        assert aggregation["calls"] >= 1
        assert aggregation["total_seconds"] >= 0.0
        assert aggregation["cumulative_seconds"] >= 0.0


def test_structured_profile_gate_requires_three_observed_progress_runs(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "probe.cprofile"
    profiler = cProfile.Profile()
    profiler.runcall(all_required_profile_probes)
    profiler.dump_stats(profile_path)
    second_profile_path = tmp_path / "probe-resume.cprofile"
    second_profiler = cProfile.Profile()
    second_profiler.runcall(all_required_profile_probes)
    second_profiler.dump_stats(second_profile_path)
    observed_functions = stage._aggregate_pstats([profile_path])
    resumed_functions = stage._aggregate_pstats([second_profile_path])
    combined_functions = stage._aggregate_pstats(
        [profile_path, second_profile_path]
    )
    observation = {
        "schema_version": stage.PROFILE_OBSERVATION_SCHEMA,
        "progress": {
            "status": "OBSERVED",
            "samples": [
                {"elapsed_seconds": 0.0, "completed_step": 0},
                {"elapsed_seconds": 1.0, "completed_step": 1},
            ],
            "per_step": {"status": "OBSERVED", "seconds_per_step": 1.0},
        },
        "gpu": {
            "status": "OBSERVED",
            "samples": [{"elapsed_seconds": 0.0, "devices": [{"gpu_index": 2}]}],
            "devices": {"2": {"sample_count": 1}},
        },
        "process_cpu": {
            "status": "OBSERVED",
            "samples": [
                {"elapsed_seconds": 0.0, "cpu_seconds": 0.0},
                {"elapsed_seconds": 1.0, "cpu_seconds": 0.5},
            ],
            "utilization_percent": {"status": "OBSERVED", "mean": 50.0},
        },
        "system_iowait": {
            "status": "OBSERVED",
            "samples": [
                {"elapsed_seconds": 0.0, "total_jiffies": 100, "iowait_jiffies": 1},
                {"elapsed_seconds": 1.0, "total_jiffies": 200, "iowait_jiffies": 2},
            ],
            "percent": {"status": "OBSERVED", "mean": 1.0},
        },
        "process_io": {
            "status": "OBSERVED",
            "samples": [
                {"elapsed_seconds": 0.0, "read_bytes": 0, "write_bytes": 0},
                {"elapsed_seconds": 1.0, "read_bytes": 10, "write_bytes": 20},
            ],
            "byte_delta": {
                "status": "OBSERVED",
                "read_bytes": 10,
                "write_bytes": 20,
            },
        },
    }
    report = {
        "structured_performance": {
            "schema_version": stage.PROFILE_PERFORMANCE_SCHEMA,
            "required_runtime_measurements": list(
                stage.PROFILE_REQUIRED_RUNTIME_MEASUREMENTS
            ),
            "required_combined_function_categories": list(
                stage.PROFILE_REQUIRED_FUNCTION_CATEGORIES
            ),
            "optional_function_categories": list(
                stage.PROFILE_OPTIONAL_FUNCTION_CATEGORIES
            ),
            "optional_function_absence_reasons": dict(
                stage.PROFILE_OPTIONAL_ABSENCE_REASONS
            ),
            "function_category_patterns": {
                key: list(value)
                for key, value in stage.PROFILE_FUNCTION_PATTERNS.items()
            },
            "runs": {
                "uninterrupted_0_to_1000": {
                    "runtime_observations": observation,
                    "function_profile": observed_functions,
                },
                "resume_path_0_to_post_checkpoint_kill": {
                    "trusted_checkpoint_step": 500,
                    "process_step_at_kill": 525,
                    "stop_step": 525,
                    "runtime_observations": observation,
                    "function_profile": stage._unobserved_function_profile(
                        "intentional_SIGKILL_prevents_durable_cprofile_dump"
                    ),
                },
                "resume_path_500_to_1000": {
                    "runtime_observations": observation,
                    "function_profile": resumed_functions,
                },
            },
            "combined_observed_function_profile": combined_functions,
        }
    }
    stage._validate_structured_profile_evidence(report)
    broken = json.loads(json.dumps(report))
    broken["structured_performance"]["runs"][
        "resume_path_0_to_post_checkpoint_kill"
    ][
        "runtime_observations"
    ]["progress"] = {"status": "NOT_OBSERVED", "samples": []}
    with pytest.raises(stage.StageError, match="did not observe progress"):
        stage._validate_structured_profile_evidence(broken)

    missing_resource = json.loads(json.dumps(report))
    missing_resource["structured_performance"]["runs"][
        "uninterrupted_0_to_1000"
    ]["runtime_observations"]["gpu"] = {
        "status": "NOT_OBSERVED",
        "reason": "nvidia_smi_unavailable",
        "samples": [],
    }
    with pytest.raises(stage.StageError, match="required resource was not observed"):
        stage._validate_structured_profile_evidence(missing_resource)

    missing_core_function = json.loads(json.dumps(report))
    missing_core_function["structured_performance"][
        "combined_observed_function_profile"
    ]["categories"]["sqlite"] = {
        "status": "NOT_OBSERVED",
        "reason": "no_profiled_function_matched_required_category_patterns",
        "matched_function_count": 0,
        "calls": 0,
    }
    with pytest.raises(stage.StageError, match="required category was not observed"):
        stage._validate_structured_profile_evidence(missing_core_function)


def test_mirror_selection_ignores_unmarked_latest_and_uses_prior(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "mirror"
    prior = mirror / "step-000000000500"
    crash_left = mirror / "step-000000001000"
    prior.mkdir(parents=True)
    crash_left.mkdir()
    stage.atomic_write_json(
        prior / "_CHECKPOINT_MIRRORED.json",
        {
            "checkpoint_mirrored": True,
            "checkpoint_digest": "5" * 64,
            "completed_step": 500,
        },
    )
    validations = {
        prior: SimpleNamespace(
            checkpoint_dir=prior,
            checkpoint_digest="5" * 64,
            completed_step=500,
        ),
        crash_left: SimpleNamespace(
            checkpoint_dir=crash_left,
            checkpoint_digest="a" * 64,
            completed_step=1000,
        ),
    }
    module = SimpleNamespace(
        MIRRORED_FILENAME="_CHECKPOINT_MIRRORED.json",
        list_generation_checkpoints=lambda _root: [prior, crash_left],
        validate_generation_checkpoint=lambda value: validations[Path(value)],
    )
    selected, ignored = stage._select_fully_mirrored_checkpoints(
        module, mirror, keep_last=2
    )
    assert [value.completed_step for value in selected] == [500]
    assert ignored == [
        {
            "checkpoint": str(crash_left),
            "reason": "valid_checkpoint_without_fully_mirrored_marker",
        }
    ]


@pytest.mark.parametrize("preexisting_fast_step", [500, 1000])
def test_formal_restore_materializes_selected_mirror_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting_fast_step: int,
) -> None:
    context = _context(tmp_path)
    mirror = (
        context.persistent
        / "outputs/bace_comrecgc/generation_checkpoint_mirror"
    )
    digests = {500: "5" * 64, 1000: "a" * 64}
    mirror_steps: list[Path] = []
    for completed_step in (500, 1000):
        checkpoint = mirror / f"step-{completed_step:012d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        stage.atomic_write_json(
            checkpoint / "_CHECKPOINT_MIRRORED.json",
            {
                "checkpoint_mirrored": True,
                "checkpoint_digest": digests[completed_step],
                "completed_step": completed_step,
            },
        )
        mirror_steps.append(checkpoint)

    fast = context.fast / "active/bace_comrecgc/generation_checkpoints"
    existing = fast / f"step-{preexisting_fast_step:012d}"
    existing.mkdir(parents=True)
    stage.atomic_write_json(
        existing / "_CHECKPOINT_MIRRORED.json",
        {
            "checkpoint_mirrored": True,
            "checkpoint_digest": digests[preexisting_fast_step],
            "completed_step": preexisting_fast_step,
        },
    )

    def validation(path: Path) -> SimpleNamespace:
        completed_step = int(path.name.removeprefix("step-"))
        return SimpleNamespace(
            checkpoint_dir=path,
            checkpoint_digest=digests[completed_step],
            completed_step=completed_step,
        )

    def fake_validate(value: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        path = Path(value)  # type: ignore[arg-type]
        if path == fast:
            pointer = stage.read_json(fast / "LATEST")
            return validation(fast / str(pointer["checkpoint_dir"]))
        return validation(path)

    module = SimpleNamespace(
        MIRRORED_FILENAME="_CHECKPOINT_MIRRORED.json",
        LATEST_FILENAME="LATEST",
        LATEST_SCHEMA_VERSION="comrecgc_generation_checkpoint_latest_v1",
        list_generation_checkpoints=lambda _root: mirror_steps,
        validate_generation_checkpoint=fake_validate,
    )
    monkeypatch.setattr(stage, "_checkpoint_module", lambda _context: module)
    monkeypatch.setattr(
        stage, "_reconcile_trace_to_checkpoint", lambda *_args, **_kwargs: {}
    )

    assert stage._restore_checkpoint_mirror(context) == 1000
    assert (fast / "step-000000000500").is_dir()
    assert (fast / "step-000000001000").is_dir()
    assert stage.read_json(fast / "LATEST") == {
        "schema_version": "comrecgc_generation_checkpoint_latest_v1",
        "checkpoint_dir": "step-000000001000",
        "completed_step": 1000,
        "checkpoint_digest": "a" * 64,
    }
    audit = stage.read_json(
        context.persistent
        / "outputs/bace_comrecgc/generation_resume_metadata"
        / "mirror_selection_audit.json"
    )
    assert [row["completed_step"] for row in audit["selected"]] == [500, 1000]


def test_formal_restore_rejects_same_step_fast_digest_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    mirror = (
        context.persistent
        / "outputs/bace_comrecgc/generation_checkpoint_mirror"
    )
    fast = context.fast / "active/bace_comrecgc/generation_checkpoints"
    mirror_step = mirror / "step-000000001000"
    fast_step = fast / mirror_step.name
    mirror_step.mkdir(parents=True)
    fast_step.mkdir(parents=True)
    stage.atomic_write_json(
        mirror_step / "_CHECKPOINT_MIRRORED.json",
        {
            "checkpoint_mirrored": True,
            "checkpoint_digest": "a" * 64,
            "completed_step": 1000,
        },
    )
    stage.atomic_write_json(
        fast_step / "_CHECKPOINT_MIRRORED.json",
        {
            "checkpoint_mirrored": True,
            "checkpoint_digest": "b" * 64,
            "completed_step": 1000,
        },
    )

    def validate(value: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        path = Path(value)  # type: ignore[arg-type]
        marker = stage.read_json(path / "_CHECKPOINT_MIRRORED.json")
        return SimpleNamespace(
            checkpoint_dir=path,
            checkpoint_digest=str(marker["checkpoint_digest"]),
            completed_step=int(marker["completed_step"]),
        )

    module = SimpleNamespace(
        MIRRORED_FILENAME="_CHECKPOINT_MIRRORED.json",
        LATEST_FILENAME="LATEST",
        LATEST_SCHEMA_VERSION="comrecgc_generation_checkpoint_latest_v1",
        list_generation_checkpoints=lambda _root: [mirror_step],
        validate_generation_checkpoint=validate,
    )
    monkeypatch.setattr(stage, "_checkpoint_module", lambda _context: module)
    with pytest.raises(stage.StageError, match="Conflicting fast checkpoint"):
        stage._restore_checkpoint_mirror(context)
    assert not (fast / "LATEST").exists()


def test_formal_restore_rejects_untrusted_fast_checkpoint_newer_than_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    mirror = (
        context.persistent
        / "outputs/bace_comrecgc/generation_checkpoint_mirror"
    )
    fast = context.fast / "active/bace_comrecgc/generation_checkpoints"
    mirror_step = mirror / "step-000000001000"
    fast_step = fast / "step-000000001500"
    mirror_step.mkdir(parents=True)
    fast_step.mkdir(parents=True)
    for checkpoint, completed_step, digest in (
        (mirror_step, 1000, "a" * 64),
        (fast_step, 1500, "f" * 64),
    ):
        stage.atomic_write_json(
            checkpoint / "_CHECKPOINT_MIRRORED.json",
            {
                "checkpoint_mirrored": True,
                "checkpoint_digest": digest,
                "completed_step": completed_step,
            },
        )

    def validate(value: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        path = Path(value)  # type: ignore[arg-type]
        marker = stage.read_json(path / "_CHECKPOINT_MIRRORED.json")
        return SimpleNamespace(
            checkpoint_dir=path,
            checkpoint_digest=str(marker["checkpoint_digest"]),
            completed_step=int(marker["completed_step"]),
        )

    module = SimpleNamespace(
        MIRRORED_FILENAME="_CHECKPOINT_MIRRORED.json",
        LATEST_FILENAME="LATEST",
        LATEST_SCHEMA_VERSION="comrecgc_generation_checkpoint_latest_v1",
        list_generation_checkpoints=lambda _root: [mirror_step],
        validate_generation_checkpoint=validate,
    )
    monkeypatch.setattr(stage, "_checkpoint_module", lambda _context: module)
    with pytest.raises(stage.StageError, match="newer than the latest verified"):
        stage._restore_checkpoint_mirror(context)
    assert not (fast / "LATEST").exists()


def test_trace_profile_parity_ignores_only_materialization() -> None:
    base = {
        "chunks": [
            {
                "index": 0,
                "path": "selected_action_trace_chunks/chunk.jsonl",
                "row_count": 10,
                "bytes": 100,
                "sha256": "a" * 64,
                "materialization": "atomic_write",
            }
        ],
        "pending_events": [{"event": "crash-window"}],
        "move_index": 500,
        "enumerated_transition_count": 800,
        "selected_transition_count": 700,
        "teleport_count": 2,
        "transition_cache_hit_count": 3,
        "transition_cache_miss_count": 4,
    }
    adopted = json.loads(json.dumps(base))
    adopted["chunks"][0]["materialization"] = "adopt_existing_identical"
    assert stage._trace_checkpoint_logical_audit(base) == (
        stage._trace_checkpoint_logical_audit(adopted)
    )
    adopted["pending_events"] = [{"event": "different-ephemeral-window"}]
    assert stage._trace_checkpoint_logical_audit(base) != (
        stage._trace_checkpoint_logical_audit(adopted)
    )


def test_lineage_smoke_replays_recorded_action_and_aids_alias_roundtrip(
    tmp_path: Path,
) -> None:
    from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256

    source = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
        comrecgc_parent_id="parent-1",
    )
    target = SimpleNamespace(
        x=[[0.0, 1.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
        comrecgc_parent_id="parent-1",
    )
    payload = {
        "dataset": "aids",
        "graph_map": {"source-original": [source], "target-canonical": [target]},
        "alias_to_canonical": {"target-original": "target-canonical"},
        "original_trace_hashes": ["source-original", "target-original"],
        "counterfactual_candidates": [],
    }
    trace = tmp_path / "trace"
    chunks = trace / "selected_action_trace_chunks"
    chunks.mkdir(parents=True)
    event = {
        "event": "selected_transition",
        "move_index": 0,
        "head_index": 0,
        "parent_id": "parent-1",
        "source_official_hash": "source-original",
        "target_official_hash": "target-original",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
        "action": ["NLC", 0, 1],
    }
    chunk = chunks / "chunk-000000.jsonl"
    chunk.write_text(json.dumps(event) + "\n", encoding="utf-8")
    manifest = trace / "selected_action_trace_manifest.json"
    stage.atomic_write_json(
        manifest,
        {
            "format": "chunked_jsonl",
            "row_count": 1,
            "chunks": [
                {
                    "index": 0,
                    "path": "selected_action_trace_chunks/chunk-000000.jsonl",
                    "row_count": 1,
                    "sha256": stage.sha256_file(chunk),
                }
            ],
        },
    )
    lineage, audit = stage._recorded_action_sample(payload, manifest)
    assert lineage[0]["action_lineage_resolved"] is True
    assert audit["recorded_action_replay_ok_count"] == 1
    assert audit["recorded_action_replay_mismatch_count"] == 0
    assert audit["legacy_inference_called_count"] == 0
    alias = stage._aids_alias_roundtrip(payload, tmp_path / "smoke")
    assert alias["alias_map_persisted"] is True
    assert alias["original_trace_roundtrip_ok_count"] == 2
    assert alias["alias_roundtrip_ok_count"] == 1
    assert (tmp_path / "smoke/alias_roundtrip_sample.pt").is_file()


def _direct_aids_roundtrip_payload() -> dict[str, object]:
    source = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
        comrecgc_parent_id="parent-1",
    )
    target = SimpleNamespace(
        x=[[0.0, 1.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
        comrecgc_parent_id="parent-1",
    )
    return {
        "dataset": "aids",
        "graph_map": {"source-direct": [source], "target-direct": [target]},
        "alias_to_canonical": {},
        "original_trace_hashes": ["source-direct", "target-direct"],
        "counterfactual_candidates": [],
    }


def test_aids_alias_roundtrip_accepts_direct_original_hashes_without_aliases(
    tmp_path: Path,
) -> None:
    result = stage._aids_alias_roundtrip(
        _direct_aids_roundtrip_payload(), tmp_path / "direct-smoke"
    )

    assert result["alias_map_persisted"] is True
    assert result["alias_map_entry_count"] == 0
    assert result["original_trace_roundtrip_sample_count"] == 2
    assert result["original_trace_roundtrip_ok_count"] == 2
    assert result["direct_roundtrip_ok_count"] == 2
    assert result["alias_roundtrip_ok_count"] == 0


def test_aids_lineage_smoke_gate_accepts_verified_empty_alias_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _output, _manifest, sentinel = stage._lineage_smoke_paths(context, "aids")
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    stage.atomic_write_json(
        sentinel,
        {
            "repair_code_closure_sha256": "a" * 64,
            "recorded_action_replay_mismatch_count": 0,
            "legacy_inference_called_count": 0,
            "alias_map_persisted": True,
            "alias_map_entry_count": 0,
            "original_trace_roundtrip_sample_count": 2,
            "original_trace_roundtrip_ok_count": 2,
            "original_trace_roundtrip_mismatch_count": 0,
            "alias_roundtrip_sample_count": 0,
            "alias_roundtrip_ok_count": 0,
            "alias_roundtrip_mismatch_count": 0,
        },
    )
    monkeypatch.setattr(stage, "_verify_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        stage, "_repair_code_closure_sha256", lambda _context: "a" * 64
    )

    stage._require_lineage_smoke_gate(
        context,
        "aids",
        input_manifests={
            "primary": "1" * 64,
            "static_project": "2" * 64,
            "required_static": "3" * 64,
        },
    )


def test_aids_alias_roundtrip_rejects_missing_original_hash(tmp_path: Path) -> None:
    payload = _direct_aids_roundtrip_payload()
    payload["original_trace_hashes"] = ["source-direct", "missing-original"]

    with pytest.raises(stage.StageError, match="original trace hash is absent"):
        stage._aids_alias_roundtrip(payload, tmp_path / "missing-smoke")


def test_aids_alias_roundtrip_rejects_graph_serialization_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.baselines.comrecgc import frozen_payload

    real_load = frozen_payload.torch_load_payload

    def load_with_graph_drift(path: Path) -> dict[str, object]:
        reloaded = real_load(path)
        reloaded["graph_map"]["source-direct"][0].x[0][0] = 99.0
        return reloaded

    monkeypatch.setattr(frozen_payload, "torch_load_payload", load_with_graph_drift)
    with pytest.raises(stage.StageError, match="serialization round trip changed"):
        stage._aids_alias_roundtrip(
            _direct_aids_roundtrip_payload(), tmp_path / "drift-smoke"
        )


def test_profile_command_keeps_formal_50000_protocol_and_uses_isolated_roots(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    command, paths = stage._profile_generation_command(
        context,
        label="resume_path",
        resume=True,
        profile_output=context.persistent / "profile.cprofile",
    )
    assert "--profile-target-steps" not in command
    assert command[command.index("--mode") + 1] == "full"
    assert command[command.index("--checkpoint-interval-steps") + 1] == "500"
    assert "--resume" in command
    assert str(paths["checkpoint"]).startswith(str(context.fast / "profile"))
    assert str(paths["mirror"]).startswith(
        str(context.persistent / "outputs/profile")
    )


def test_profile_parity_normalizes_only_operational_roots_and_resume(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    uninterrupted, _paths_a = stage._profile_generation_command(
        context,
        label="uninterrupted",
        resume=False,
        profile_output=context.persistent / "uninterrupted.cprofile",
    )
    resumed, _paths_b = stage._profile_generation_command(
        context,
        label="resume_path",
        resume=True,
        profile_output=context.persistent / "resumed.cprofile",
    )
    evidence_a = stage._profile_command_evidence(uninterrupted)
    evidence_b = stage._profile_command_evidence(resumed)
    assert evidence_a["raw_scientific_argv"] != evidence_b["raw_scientific_argv"]
    assert evidence_a["raw_scientific_command_sha256"] != (
        evidence_b["raw_scientific_command_sha256"]
    )
    assert evidence_a["parity_normalized_scientific_argv"] == (
        evidence_b["parity_normalized_scientific_argv"]
    )
    assert evidence_a["parity_normalized_scientific_command_sha256"] == (
        evidence_b["parity_normalized_scientific_command_sha256"]
    )

    scientific_drift = list(resumed)
    parent_limit = scientific_drift.index("--parent-limit") + 1
    scientific_drift[parent_limit] = "359"
    drift_evidence = stage._profile_command_evidence(scientific_drift)
    assert drift_evidence["parity_normalized_scientific_command_sha256"] != (
        evidence_a["parity_normalized_scientific_command_sha256"]
    )


def test_production_spec_has_no_placeholders_and_all_stage_hashseeds() -> None:
    spec_path = Path("ops/specs/autodl_three_lines_20260821.yaml")
    text = spec_path.read_text(encoding="utf-8")
    assert "__CONFIGURE_" not in text
    payload = json.loads(text)
    stages = [stage_row for lane in payload["lanes"] for stage_row in lane["stages"]]
    assert len(stages) == 8
    assert all(row["environment"]["PYTHONHASHSEED"] == "0" for row in stages)
    assert all("run_three_lines_stage.py" in row["command"][1] for row in stages)
    assert all(row["resume_command"][-1] == "--resume" for row in stages)
    globalgce = next(row for row in stages if row["id"] == "bace_globalgce_wnode")
    assert globalgce["output_manifest"].endswith(
        "/outputs/bace_globalgce_common4/common4/globalgce/MANIFEST.sha256"
    )


def test_freeze_and_bace_formal_stages_require_preflight_smoke_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    monkeypatch.setenv("DISALLOW_GENERATION", "1")
    with pytest.raises(stage.StageError, match="preserved-lineage smoke gate"):
        stage._run_freeze(context, "mutagenicity")
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(stage, "_all_input_gates", lambda *_args: snapshot)
    with pytest.raises(stage.StageError, match="BACE_PROFILE_SMOKE_PASS"):
        stage._run_bace_generate(context)


def test_stage_parser_exposes_nonformal_lineage_and_profile_smokes() -> None:
    parser = stage.build_parser()
    common = [
        "--project-root",
        "/project",
        "--step0-project-root",
        "/step0",
        "--external-root",
        "/vendor",
        "--persistent-root",
        "/persistent",
        "--fast-root",
        "/fast",
        "--python",
        "/python",
    ]
    for name in ("mut-lineage-smoke", "aids-lineage-smoke", "bace-profile-smoke"):
        assert parser.parse_args([name, *common]).stage == name


def test_secret_and_manifest_parent_symlink_escape_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(stage.StageError, match="secret"):
        stage._assert_no_secret(["safe", "--token", "redacted"])
    with pytest.raises(stage.StageError, match="secret"):
        stage._assert_no_secret(["safe", "--set", "openai_api_key=redacted"])
    root = tmp_path / "root"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    artifact = external / "artifact"
    artifact.write_text("outside\n", encoding="utf-8")
    (root / "escape").symlink_to(external, target_is_directory=True)
    manifest = root / "MANIFEST.sha256"
    manifest.write_text(
        f"{stage.sha256_file(artifact)}  escape/artifact\n", encoding="utf-8"
    )
    with pytest.raises(stage.StageError, match="escapes"):
        stage.verify_sha256_manifest(root, manifest)


def test_stage_subprocess_filters_inherited_credentials_and_rejects_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    class Process:
        returncode = 0

        def __init__(self, *_args: object, **kwargs: object) -> None:
            captured.update(kwargs["env"])

        def poll(self) -> int:
            return 0

        def send_signal(self, _signum: int) -> None:
            raise AssertionError("completed fake process must not receive a signal")

    monkeypatch.setenv("SAFE_STAGE_VALUE", "allowed")
    monkeypatch.setenv("HPC_PASSWORD", "credential-must-not-propagate")
    monkeypatch.setenv("OPENAI_API_KEY", "credential-must-not-propagate")
    monkeypatch.setenv("UNRELATED_VALUE", "token=credential-must-not-propagate")
    monkeypatch.setattr(stage.subprocess, "Popen", Process)
    stage._run_checked(
        ["python", "safe_stage.py"],
        cwd=tmp_path,
        env_extra={"EXPLICIT_SAFE": "yes"},
    )
    assert captured["SAFE_STAGE_VALUE"] == "allowed"
    assert captured["EXPLICIT_SAFE"] == "yes"
    assert "HPC_PASSWORD" not in captured
    assert "OPENAI_API_KEY" not in captured
    assert "UNRELATED_VALUE" not in captured

    with pytest.raises(stage.StageError, match="credential-named"):
        stage._run_checked(
            ["python", "safe_stage.py"],
            cwd=tmp_path,
            env_extra={"API_TOKEN": "not-logged"},
        )
    with pytest.raises(stage.StageError, match="credential-like"):
        stage._run_checked(
            ["python", "safe_stage.py"],
            cwd=tmp_path,
            env_extra={"SAFE_NAME": "password=not-logged"},
        )


def test_context_rejects_symlink_components_and_roots_reject_containment(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    project = real / "project"
    step0 = real / "step0"
    external = real / "external"
    persistent = real / "persistent"
    fast = real / "fast"
    for path in (project, step0, external, persistent, fast):
        path.mkdir(parents=True)
    python = real / "python"
    python.write_text("python\n", encoding="utf-8")
    python.chmod(0o755)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    args = SimpleNamespace(
        project_root=str(linked / "project"),
        step0_project_root=str(step0),
        external_root=str(external),
        persistent_root=str(persistent),
        fast_root=str(fast),
        python=str(python),
        resume=False,
    )
    with pytest.raises(stage.StageError, match="symlink component"):
        stage._context(args)

    interpreter_link = tmp_path / "python-link"
    interpreter_link.symlink_to(python)
    physical_args = SimpleNamespace(
        project_root=str(project),
        step0_project_root=str(step0),
        external_root=str(external),
        persistent_root=str(persistent),
        fast_root=str(fast),
        python=str(interpreter_link),
        resume=False,
    )
    resolved_context = stage._context(physical_args)
    assert resolved_context.python == python.resolve()
    stage._validate_roots(resolved_context)

    fast_inside_persistent = persistent / "nested-fast"
    fast_inside_persistent.mkdir()
    with pytest.raises(stage.StageError, match="must be disjoint"):
        stage._validate_roots(
            stage.Context(
                project=project,
                step0=step0,
                external=external,
                persistent=persistent,
                fast=fast_inside_persistent,
                python=python,
                resume=False,
            )
        )
    persistent_inside_fast = fast / "nested-persistent"
    persistent_inside_fast.mkdir()
    with pytest.raises(stage.StageError, match="must be disjoint"):
        stage._validate_roots(
            stage.Context(
                project=project,
                step0=step0,
                external=external,
                persistent=persistent_inside_fast,
                fast=fast,
                python=python,
                resume=False,
            )
        )


def test_sha_manifest_and_sentinel_fail_closed_after_output_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        stage,
        "_current_code_lineage",
        lambda _context: {
            "repair_code_closure_sha256": "d" * 64,
            "project_commit": "e" * 40,
            "external_comrecgc_commit": stage.UPSTREAM_COMMIT,
        },
    )
    root = tmp_path / "output"
    root.mkdir()
    artifact = root / "artifact.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = root / "MANIFEST.sha256"
    sentinel = root / "PASS.json"
    stage.write_sha256_manifest(
        base=root, items=[artifact], manifest=manifest, exclude=[manifest, sentinel]
    )
    digest = stage.sha256_file(manifest)
    stage._publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=root,
        input_digest_before="a" * 64,
        input_digest_after="a" * 64,
        input_manifests={
            "primary": "a" * 64,
            "static_project": "b" * 64,
            "required_static": "c" * 64,
        },
        payload={"schema_version": "scientific_report_v1", "status": "PASS"},
    )
    input_manifests = {
        "primary": "a" * 64,
        "static_project": "b" * 64,
        "required_static": "c" * 64,
    }
    assert stage._verify_sentinel(
        context,
        sentinel,
        manifest,
        {"status": "PASS"},
        input_manifests=input_manifests,
    )
    sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert sentinel_payload["schema_version"] == "autodl_three_lines_stage_v1"
    assert sentinel_payload["scientific_payload_schema_version"] == (
        "scientific_report_v1"
    )
    monkeypatch.setattr(
        stage,
        "_current_code_lineage",
        lambda _context: {
            "repair_code_closure_sha256": "f" * 64,
            "project_commit": "e" * 40,
            "external_comrecgc_commit": stage.UPSTREAM_COMMIT,
        },
    )
    with pytest.raises(stage.StageError, match="Existing sentinel is invalid"):
        stage._verify_sentinel(
            context,
            sentinel,
            manifest,
            {"status": "PASS"},
            input_manifests=input_manifests,
        )
    monkeypatch.setattr(
        stage,
        "_current_code_lineage",
        lambda _context: {
            "repair_code_closure_sha256": "d" * 64,
            "project_commit": "e" * 40,
            "external_comrecgc_commit": stage.UPSTREAM_COMMIT,
        },
    )
    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(stage.StageError, match="SHA256 mismatch"):
        stage._verify_sentinel(
            context,
            sentinel,
            manifest,
            {"status": "PASS"},
            input_manifests=input_manifests,
        )
    assert digest == json.loads(sentinel.read_text(encoding="utf-8"))[
        "output_manifest_sha256"
    ]
