from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile

import pytest


pytest.importorskip("networkx")
pytest.importorskip("pandas")

from src.baselines import globalgce_hpc_hierarchical as hierarchical
from src.baselines.globalgce_hpc_exact import (
    build_partition_manifest,
    merge_exact_shards,
    run_mining_shard,
    validate_merge_result,
)
from src.baselines.globalgce_hpc_hierarchical import (
    adopt_completed_array,
    build_group_plan,
    finalize_hierarchical_merge,
    monolithic_stream_parity,
    publish_hierarchical_evidence,
    run_group_merge,
)


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SRC = Path(
    os.environ.get(
        "GLOBALGCE_OFFICIAL_SRC", str(ROOT / "baselines/globalgce_official/src")
    )
)
EXECUTION_COMMIT = "b" * 40
OFFICIAL_COMMIT = "157e65c2850bc787f229a1ee8c60564906b933f2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_sha(payload: dict[str, object]) -> str:
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _tiny_shards(tmp_path: Path) -> tuple[Path, dict, Path]:
    graphs = tmp_path / "graphs.jsonl"
    graph_rows = [
        {
            "graph_id": 0,
            "nodes": [{"id": index, "label": 0} for index in range(3)],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
                {"source": 0, "target": 2, "label": 1},
            ],
        },
        {
            "graph_id": 1,
            "nodes": [{"id": index, "label": 0} for index in range(3)],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
            ],
        },
        {
            "graph_id": 2,
            "nodes": [{"id": index, "label": 2} for index in range(3)],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
            ],
        },
    ]
    graphs.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in graph_rows),
        encoding="utf-8",
    )
    mining: dict[str, object] = {
        "source_label": 1,
        "seed": 7,
        "epochs": 100,
        "min_support": 1,
        "min_vertices": 2,
        "max_vertices": 3,
        "top_k": 20,
        "root_count": 2,
        "exact": True,
        "approximate_pruning": False,
    }
    input_manifest = tmp_path / "input.json"
    payload: dict[str, object] = {
        "state": "PASS",
        "dataset": "tastemolnet",
        "method": "globalgce",
        "stage": "EXACT_GSPAN_CPU_INPUT",
        "route_kind": "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD",
        "split_scope": "train_only",
        "calibration_payload_included": False,
        "test_payload_included": False,
        "matrix_publication_allowed_from_hpc": False,
        "official_globalgce_commit": OFFICIAL_COMMIT,
        "source_commit": "a" * 40,
        "mining_config": mining,
        "mining_config_sha256": _line_sha(mining),
        "hpc_runtime_config": {"name": "hpc.yaml", "sha256": "c" * 64},
        "files": [{"role": "graph_jsonl", "sha256": _sha(graphs)}],
        "transaction_binding": {
            "shared_transaction_database": True,
            "target_labels": [0, 2],
            "target_semantics_do_not_modify_transaction_database": True,
            "graph_jsonl_sha256": _sha(graphs),
            "graph_count": 3,
        },
        "transfer_policy": {
            "source_data_is_train_only_derived": True,
            "hpc_may_modify_autodl_matrix": False,
        },
    }
    payload["manifest_sha256"] = _line_sha(payload)
    input_manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "partition-manifest.json"
    manifest = build_partition_manifest(
        graph_jsonl=graphs,
        input_manifest=input_manifest,
        expected_commit=EXECUTION_COMMIT,
        official_src=OFFICIAL_SRC,
        output=manifest_path,
        shard_count=2,
        min_support=1,
        min_vertices=2,
        max_vertices=3,
        top_k=20,
        split_root_indices=(0,),
        split_depth=2,
        canary_root_indices=(0,),
    )
    shards = tmp_path / "shards"
    for index in range(2):
        run_mining_shard(
            partition_manifest=manifest_path,
            shard_index=index,
            output_root=shards / f"shard-{index:03d}",
            flush_every=1,
        )
    return manifest_path, manifest, shards


def _plan(tmp_path: Path) -> tuple[Path, dict, Path, dict]:
    manifest_path, manifest, shards = _tiny_shards(tmp_path)
    adoption = tmp_path / "array-adoption.json"
    adopted = adopt_completed_array(
        partition_manifest=manifest_path, shards_root=shards, output=adoption
    )
    plan_path = tmp_path / "group-plan.json"
    plan = build_group_plan(
        partition_manifest=manifest_path,
        shards_root=shards,
        array_adoption=adoption,
        output=plan_path,
        group_count=2,
    )
    assert adopted["passed_shard_count"] == 2
    return plan_path, plan, shards, manifest


def test_hierarchical_final_streams_are_byte_identical_to_monolithic(
    tmp_path: Path,
) -> None:
    plan_path, plan, shards, manifest = _plan(tmp_path)
    groups = tmp_path / "groups"
    for group_index in range(plan["group_count"]):
        result = run_group_merge(
            group_plan=plan_path,
            group_index=group_index,
            output_root=groups / f"group-{group_index:02d}",
            scratch_root=tmp_path / "group-scratch",
            progress_seconds=1,
        )
        assert result["status"] == "PASS"
    monolithic = tmp_path / "monolithic"
    merge_exact_shards(
        partition_manifest=plan["partition_manifest"],
        shards_root=shards,
        output_root=monolithic,
    )
    final = tmp_path / "hierarchical"
    (tmp_path / "final-scratch").mkdir()
    result = finalize_hierarchical_merge(
        group_plan=plan_path,
        groups_root=groups,
        state_root=tmp_path / "final-state",
        scratch_root=tmp_path / "final-scratch",
        output_root=final,
    )
    assert result["status"] == "PASS"
    assert monolithic_stream_parity(final, monolithic)["status"] == "PASS"
    validated = validate_merge_result(
        final, manifest=manifest, allowed_scopes=("FULL_MANIFEST",)
    )
    assert validated["events_sha256"] == _sha(monolithic / "events.jsonl")
    assert validated["patterns_sha256"] == _sha(monolithic / "patterns.jsonl")


def test_group_merge_resumes_only_after_committed_source_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest, shards = _tiny_shards(tmp_path)
    adoption = tmp_path / "array-adoption.json"
    adopt_completed_array(
        partition_manifest=manifest_path, shards_root=shards, output=adoption
    )
    plan_path = tmp_path / "one-group-plan.json"
    build_group_plan(
        partition_manifest=manifest_path,
        shards_root=shards,
        array_adoption=adoption,
        output=plan_path,
        group_count=1,
    )
    original = hierarchical._write_partition_chunk
    failed = False

    def interrupt_second_shard(**kwargs):
        nonlocal failed
        if int(kwargs["unit"]["shard_index"]) == 1 and not failed:
            failed = True
            raise RuntimeError("simulated preemption")
        return original(**kwargs)

    monkeypatch.setattr(hierarchical, "_write_partition_chunk", interrupt_second_shard)
    root = tmp_path / "resumable-group"
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run_group_merge(
            group_plan=plan_path,
            group_index=0,
            output_root=root,
            scratch_root=tmp_path / "scratch",
        )
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    assert checkpoint["completed_shard_indices"] == [0]
    monkeypatch.setattr(hierarchical, "_write_partition_chunk", original)
    result = run_group_merge(
        group_plan=plan_path,
        group_index=0,
        output_root=root,
        scratch_root=tmp_path / "scratch",
    )
    assert result["status"] == "PASS"
    assert json.loads((root / "checkpoint.json").read_text())[
        "completed_shard_indices"
    ] == [0, 1]


def test_hierarchical_evidence_binds_adoption_groups_and_final(tmp_path: Path) -> None:
    plan_path, plan, _shards, _manifest = _plan(tmp_path)
    groups = tmp_path / "groups"
    for group_index in range(plan["group_count"]):
        run_group_merge(
            group_plan=plan_path,
            group_index=group_index,
            output_root=groups / f"group-{group_index:02d}",
            scratch_root=tmp_path / "group-scratch",
        )
    (tmp_path / "final-scratch").mkdir()
    merge = tmp_path / "merge"
    finalize_hierarchical_merge(
        group_plan=plan_path,
        groups_root=groups,
        state_root=tmp_path / "final-state",
        scratch_root=tmp_path / "final-scratch",
        output_root=merge,
    )
    package = tmp_path / "package"
    package.mkdir()
    result_archive = package / "t8_exact_result_bundle.tar.gz"
    result_archive.write_bytes(b"tiny result")
    receipt = {
        "receipt_sha256": "a" * 64,
        "archive_sha256": _sha(result_archive),
    }
    evidence_scratch = tmp_path / "evidence-scratch"
    evidence_scratch.mkdir()
    ready = publish_hierarchical_evidence(
        group_plan=plan_path,
        groups_root=groups,
        merge_root=merge,
        package_root=package,
        storage_safe_receipt=receipt,
        scratch_root=evidence_scratch,
    )
    assert ready["status"] == "PASS"
    assert ready["matrix_write_enabled"] is False
    with tarfile.open(package / "t8_hierarchical_evidence.tar.gz", "r:gz") as archive:
        names = archive.getnames()
    assert "array_adoption_manifest.json" in names
    assert "group_plan.json" in names
    assert "groups/group-00/group_manifest.json" in names
    assert (package / "HIERARCHICAL_PACKAGE_READY.json").is_file()


def test_chain_keeps_historical_jobs_held_and_uses_afterok() -> None:
    script = (ROOT / "scripts/hpc/t8/launch_hierarchical_merge_chain.sh").read_text()
    assert "JobHeldUser" in script
    assert "scontrol release" not in script
    assert 'afterok:${group_job_id}' in script
    assert 'afterok:${final_job_id}' in script


def test_mac_relay_is_resumable_and_never_publishes_matrix() -> None:
    script = (ROOT / "scripts/local/run_t8_result_relay_v1.sh").read_text()
    assert "--append-verify" in script
    assert ".partial" in script
    assert "HPC_PACKAGE_READY" in script
    assert "matrix_write_enabled\":False" in script
    assert "matrix_authority" not in script.lower()
