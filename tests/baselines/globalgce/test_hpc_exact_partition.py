from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest


pytest.importorskip("networkx")
pytest.importorskip("pandas")

from src.baselines.globalgce_hpc_exact import (
    GlobalGCEHPCExactError,
    TypedDFSEdge,
    build_partition_manifest,
    build_result_bundle,
    canonical_sha256,
    dfs_code_from_json,
    dfs_code_sha256,
    dfs_code_to_json,
    merge_exact_shards,
    load_graph_jsonl,
    run_exact_reference,
    run_mining_shard,
    validate_hpc_cli_contract,
    validate_partition_manifest,
    verify_exact_parity,
)
import src.baselines.globalgce_hpc_exact as hpc_exact


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SRC = PROJECT_ROOT / "baselines/globalgce_official/src"
EXECUTION_COMMIT = "b" * 40
OFFICIAL_COMMIT = "157e65c2850bc787f229a1ee8c60564906b933f2"


def _write_tiny_graphs(path: Path) -> None:
    rows = [
        {
            "graph_id": 0,
            "nodes": [
                {"id": 0, "label": 0},
                {"id": 1, "label": 0},
                {"id": 2, "label": 0},
            ],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
                {"source": 0, "target": 2, "label": 1},
            ],
        },
        {
            "graph_id": 1,
            "nodes": [
                {"id": 0, "label": 0},
                {"id": 1, "label": 0},
                {"id": 2, "label": 0},
            ],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
            ],
        },
        {
            "graph_id": 2,
            "nodes": [
                {"id": 0, "label": 2},
                {"id": 1, "label": 2},
                {"id": 2, "label": 2},
            ],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _line_sha(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def _write_input_manifest(path: Path, graphs: Path, *, min_vertices: int = 2) -> None:
    graph_sha = hashlib.sha256(graphs.read_bytes()).hexdigest()
    mining = {
        "source_label": 1,
        "seed": 7,
        "epochs": 100,
        "min_support": 1,
        "min_vertices": min_vertices,
        "max_vertices": 3,
        "top_k": 20,
        "root_count": 2,
        "exact": True,
        "approximate_pruning": False,
    }
    payload = {
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
        "files": [{"role": "graph_jsonl", "sha256": graph_sha}],
        "transaction_binding": {
            "shared_transaction_database": True,
            "target_labels": [0, 2],
            "target_semantics_do_not_modify_transaction_database": True,
            "graph_jsonl_sha256": graph_sha,
            "graph_count": 3,
        },
        "transfer_policy": {
            "source_data_is_train_only_derived": True,
            "hpc_may_modify_autodl_matrix": False,
        },
    }
    payload["manifest_sha256"] = _line_sha(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _build_manifest(
    tmp_path: Path,
    *,
    name: str,
    included: tuple[int, ...] | None,
    shard_count: int,
    included_unit_ids: tuple[str, ...] = (),
    canary_roots: tuple[int, ...] = (0,),
    min_vertices: int = 2,
) -> tuple[Path, dict]:
    graphs = tmp_path / "graphs.jsonl"
    if not graphs.exists():
        _write_tiny_graphs(graphs)
    input_manifest = tmp_path / f"input-manifest-minv{min_vertices}.json"
    if not input_manifest.exists():
        _write_input_manifest(input_manifest, graphs, min_vertices=min_vertices)
    output = tmp_path / name
    report = build_partition_manifest(
        graph_jsonl=graphs,
        input_manifest=input_manifest,
        expected_commit=EXECUTION_COMMIT,
        official_src=OFFICIAL_SRC,
        output=output,
        shard_count=shard_count,
        min_support=1,
        min_vertices=min_vertices,
        max_vertices=3,
        top_k=20,
        split_root_indices=(0,),
        split_depth=2,
        canary_root_indices=canary_roots,
        included_root_indices=included,
        included_unit_ids=included_unit_ids,
    )
    return output, report


def _run_all_shards(
    tmp_path: Path,
    manifest_path: Path,
    manifest: dict,
    name: str,
    *,
    use_scratch: bool = False,
) -> Path:
    shards = tmp_path / f"{name}-shards"
    for index in range(manifest["shard_count"]):
        root = shards / f"shard-{index:03d}"
        scratch = tmp_path / f"{name}-scratch" if use_scratch else None
        first = run_mining_shard(
            partition_manifest=manifest_path,
            shard_index=index,
            output_root=root,
            flush_every=1,
            scratch_root=scratch,
        )
        # A second invocation is a boundary-resume verification, not a replay.
        second = run_mining_shard(
            partition_manifest=manifest_path,
            shard_index=index,
            output_root=root,
            flush_every=1,
            scratch_root=scratch,
        )
        assert second["partition_result_sha256s"] == first["partition_result_sha256s"]
    merged = tmp_path / f"{name}-merged"
    merge_exact_shards(
        partition_manifest=manifest_path,
        shards_root=shards,
        output_root=merged,
        scratch_root=(tmp_path / f"{name}-merge-scratch") if use_scratch else None,
    )
    return merged


def test_typed_dfs_code_round_trip_preserves_integer_and_string_labels() -> None:
    code = (
        TypedDFSEdge(0, 1, (6, "single", 8)),
        TypedDFSEdge(1, 2, (-1, 2, "N")),
    )
    payload = dfs_code_to_json(code)
    assert dfs_code_from_json(payload) == code
    assert dfs_code_sha256(code) == canonical_sha256(payload)


def test_cli_contract_is_narrow_and_requires_hpc_config(tmp_path: Path) -> None:
    config = tmp_path / "configs/hpc.yaml"
    config.parent.mkdir()
    config.write_text("seed: 7\n", encoding="utf-8")
    assert validate_hpc_cli_contract(config, []) == config.resolve()
    validate_hpc_cli_contract(
        config, ["inference.fallback_to_heuristic=false"]
    )
    with pytest.raises(GlobalGCEHPCExactError, match="only --set"):
        validate_hpc_cli_contract(config, ["science.approximate=true"])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "graph_id": 9,
                "nodes": [{"id": 0, "label": 1}, {"id": 1, "label": 2}],
                "edges": [{"source": 0, "target": 1, "label": 1}],
            },
            "zero-based JSONL row index",
        ),
        (
            {
                "graph_id": 0,
                "nodes": [{"id": 0, "label": True}, {"id": 1, "label": 2}],
                "edges": [{"source": 0, "target": 1, "label": 1}],
            },
            "node label must be an integer",
        ),
        (
            {
                "graph_id": 0,
                "nodes": [{"id": 0, "label": 1}, {"id": 1, "label": 2}],
                "edges": [],
            },
            "must be connected",
        ),
    ],
)
def test_production_graph_contract_fails_closed(
    tmp_path: Path, row: dict, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(GlobalGCEHPCExactError, match=message):
        load_graph_jsonl(path)


def test_partitioned_official_dfs_is_event_exact_and_boundary_resumable(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _build_manifest(
        tmp_path, name="full-manifest.json", included=None, shard_count=2
    )
    assert manifest["scope"] == "FULL_ROOT_UNIVERSE"
    assert manifest["matrix_write_enabled"] is False
    assert manifest["completeness_proof"]["disjoint"] is True
    assert manifest["completeness_proof"]["complete"] is True
    types = {unit["partition_type"] for unit in manifest["partitions"]}
    assert {"ROOT_SUBTREE", "PREFIX_HEADER", "PREFIX_SUBTREE"} <= types
    assert validate_partition_manifest(manifest_path)["manifest_sha256"] == manifest[
        "manifest_sha256"
    ]

    merged = _run_all_shards(tmp_path, manifest_path, manifest, "full")
    reference = tmp_path / "reference"
    run_exact_reference(
        partition_manifest=manifest_path,
        output_root=reference,
        root_indices=(0,),
        flush_every=1,
    )
    parity_path = tmp_path / "parity.json"
    parity = verify_exact_parity(
        partition_manifest=manifest_path,
        reference_root=reference,
        merged_root=merged,
        output=parity_path,
    )
    assert parity["status"] == "PASS"
    assert parity["patterns_equal"] is True
    assert parity["supports_equal"] is True
    assert parity["stable_preorder_equal"] is True
    assert parity["candidate_inputs_equal"] is True
    assert parity["rejection_events_equal"] is True


def test_min_vertices_gate_excludes_two_vertex_root_patterns(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(
        tmp_path,
        name="minv3-manifest.json",
        included=None,
        shard_count=2,
        min_vertices=3,
    )
    merged = _run_all_shards(tmp_path, manifest_path, manifest, "minv3")
    patterns = [
        json.loads(line)
        for line in (merged / "patterns.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert patterns
    assert all(
        len(
            {
                endpoint
                for edge in row["dfs_code"]
                for endpoint in (edge["frm"], edge["to"])
            }
        )
        >= 3
        for row in patterns
    )
    assert any(
        len(row["dfs_code"]) == 1
        for row in (
            json.loads(line)
            for line in (merged / "events.jsonl").read_text(encoding="utf-8").splitlines()
        )
    )
    reference = tmp_path / "minv3-reference"
    run_exact_reference(
        partition_manifest=manifest_path,
        output_root=reference,
        root_indices=(0,),
        flush_every=1,
    )
    parity = verify_exact_parity(
        partition_manifest=manifest_path,
        reference_root=reference,
        merged_root=merged,
        output=tmp_path / "minv3-parity.json",
    )
    assert parity["status"] == "PASS"


def test_canary_receipt_binds_full_result_by_shared_scientific_input(
    tmp_path: Path,
) -> None:
    canary_path, canary = _build_manifest(
        tmp_path, name="canary-manifest.json", included=(0,), shard_count=2
    )
    canary_merged = _run_all_shards(tmp_path, canary_path, canary, "canary")
    reference = tmp_path / "canary-reference"
    run_exact_reference(
        partition_manifest=canary_path,
        output_root=reference,
        flush_every=1,
    )
    parity_path = tmp_path / "canary-parity.json"
    parity = verify_exact_parity(
        partition_manifest=canary_path,
        reference_root=reference,
        merged_root=canary_merged,
        output=parity_path,
    )
    assert parity["status"] == "PASS"

    full_path, full = _build_manifest(
        tmp_path, name="full-manifest.json", included=None, shard_count=2
    )
    assert full["manifest_sha256"] != canary["manifest_sha256"]
    assert full["scientific_input_sha256"] == canary["scientific_input_sha256"]
    full_merged = _run_all_shards(tmp_path, full_path, full, "full")
    archive = tmp_path / "result.tar"
    receipt_path = tmp_path / "result-manifest.json"
    evidence_paths = []
    for name in ("environment", "slurm", "resources"):
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(
            json.dumps({"schema_version": f"tiny_{name}_v1", "status": "PASS"}),
            encoding="utf-8",
        )
        evidence_paths.append(evidence_path)
    receipt = build_result_bundle(
        partition_manifest=full_path,
        merge_root=full_merged,
        parity_receipt=parity_path,
        output_tar=archive,
        output_manifest=receipt_path,
        environment_manifest=evidence_paths[0],
        slurm_inventory=evidence_paths[1],
        resource_metrics=evidence_paths[2],
    )
    assert receipt["status"] == "PASS"
    assert receipt["matrix_write_enabled"] is False
    assert archive.is_file()


def test_bounded_canary_uses_one_prefix_plus_one_small_root_and_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, catalog = _build_manifest(
        tmp_path, name="catalog.json", included=(0,), shard_count=1
    )
    del catalog_path
    prefix = next(
        unit for unit in catalog["partitions"] if unit["partition_type"] == "PREFIX_SUBTREE"
    )
    manifest_path, manifest = _build_manifest(
        tmp_path,
        name="bounded-canary.json",
        included=(1,),
        included_unit_ids=(prefix["partition_id"],),
        shard_count=2,
        canary_roots=(0, 1),
    )
    assert manifest["scope"] == "SELECTED_PARTITION_CANARY"
    assert manifest["whole_root_indices"] == [1]
    assert manifest["selected_partition_ids"] == [prefix["partition_id"]]
    assert {unit["partition_type"] for unit in manifest["partitions"]} == {
        "ROOT_SUBTREE",
        "PREFIX_SUBTREE",
    }

    merged = _run_all_shards(
        tmp_path, manifest_path, manifest, "bounded", use_scratch=True
    )
    reference = tmp_path / "bounded-reference"

    independent_unit = hpc_exact._run_independent_reference_unit
    call_count = 0

    def interrupt_after_first_boundary(*args: object, **kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated Slurm preemption")
        return independent_unit(*args, **kwargs)

    monkeypatch.setattr(
        hpc_exact, "_run_independent_reference_unit", interrupt_after_first_boundary
    )
    with pytest.raises(RuntimeError, match="simulated Slurm preemption"):
        run_exact_reference(
            partition_manifest=manifest_path,
            output_root=reference,
            scratch_root=tmp_path / "bounded-reference-scratch",
            flush_every=1,
        )
    first_unit = sorted(
        manifest["partitions"], key=lambda row: row["global_partition_order"]
    )[0]
    first_boundary_manifest = json.loads(
        (
            reference
            / "partitions"
            / first_unit["partition_id"]
            / "partition_manifest.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        hpc_exact, "_run_independent_reference_unit", independent_unit
    )

    def forbidden_production_path(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selected-prefix reference reused production executor/merge")

    monkeypatch.setattr(hpc_exact, "_execute_partition_unit", forbidden_production_path)
    monkeypatch.setattr(hpc_exact, "_merge_partition_results", forbidden_production_path)
    first_reference = run_exact_reference(
        partition_manifest=manifest_path,
        output_root=reference,
        scratch_root=tmp_path / "bounded-reference-scratch",
        flush_every=1,
    )
    second_reference = run_exact_reference(
        partition_manifest=manifest_path,
        output_root=reference,
        scratch_root=tmp_path / "bounded-reference-scratch",
        flush_every=1,
    )
    assert second_reference["result_sha256"] == first_reference["result_sha256"]
    assert first_reference["execution_engine"] == "INDEPENDENT_SERIAL_OFFICIAL_TRAVERSAL"
    resumed_boundary_manifest = json.loads(
        (
            reference
            / "partitions"
            / first_unit["partition_id"]
            / "partition_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert resumed_boundary_manifest["result_sha256"] == first_boundary_manifest[
        "result_sha256"
    ]
    parity = verify_exact_parity(
        partition_manifest=manifest_path,
        reference_root=reference,
        merged_root=merged,
        output=tmp_path / "bounded-parity.json",
    )
    assert parity["status"] == "PASS"
    assert parity["search_space_scope"] == "SELECTED_PARTITION_CANARY"
    assert parity["selected_partition_ids"] == [prefix["partition_id"]]
    for index in range(manifest["shard_count"]):
        checkpoint = json.loads(
            (
                tmp_path
                / "bounded-shards"
                / f"shard-{index:03d}"
                / "checkpoint.json"
            ).read_text(encoding="utf-8")
        )
        assert checkpoint["state"] == "COMPLETE"
        assert checkpoint["current_partition_id"] is None
        assert checkpoint["resume_boundary"] == "COMPLETED_PERSISTENT_PARTITION_ONLY"


def test_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_manifest(
        tmp_path, name="manifest.json", included=(0,), shard_count=1
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["partitions"][0]["dfs_code"][0]["to"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GlobalGCEHPCExactError, match="self-hash"):
        validate_partition_manifest(manifest_path)
