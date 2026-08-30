from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.baselines.bace_gnn_baseline_contracts import baseline_spec
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_jsonl,
    file_identity,
    sha256_file,
    stable_sha256,
)
from src.eval.bace_native_baseline_gnn import (
    CALIBRATION_STAGE,
    merge_fullgraph_verification_shards,
)


CHECKPOINT_ID = "a" * 64
MOLCLR_ID = "b" * 64


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_source(root: Path) -> tuple[list[dict[str, Any]], str]:
    root.mkdir(parents=True)
    spec = baseline_spec("gcfexplainer")
    candidates = [
        {
            "candidate_id": f"candidate-{index:02d}",
            "rank": index + 1,
            "native_rank": index + 1,
            "canonical_smiles": "C" * (index + 1),
            "action_kind": spec.action_kind,
            "action_semantics": spec.action_semantics,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "oracle_checkpoint_hash": CHECKPOINT_ID,
        }
        for index in range(20)
    ]
    atomic_jsonl(root / "candidate_universe.jsonl", candidates)
    candidate_sha256 = sha256_file(root / "candidate_universe.jsonl")
    atomic_json(
        root / "run_manifest.json",
        {
            "schema_version": "test_bace_gcf_generation_v1",
            "dataset": "bace",
            "method_id": "gcfexplainer",
            "stage": "TRAIN_CANDIDATE_GENERATION",
            "status": "PASS",
            "run_complete": True,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "candidate_universe_hash": candidate_sha256,
            "generation_attempt_id": "attempt-0",
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    return candidates, candidate_sha256


def _build_route(tmp_path: Path) -> dict[str, Any]:
    cell = tmp_path / "bace_gcf_cell"
    predecessor = cell / "train_candidates" / "attempt-0"
    candidates, candidate_sha256 = _write_source(predecessor)
    split = cell / "inputs" / "bace_calibration.csv"
    split.parent.mkdir(parents=True)
    split.write_text(
        "molecule_id,smiles,label\np0,C,1\np1,CC,1\np2,CCC,1\np3,CCCC,1\n",
        encoding="utf-8",
    )
    parent_ids = ["p0", "p1", "p2", "p3"]
    all_parent_sha256 = stable_sha256(parent_ids)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    shard_roots: list[Path] = []
    for index, parent_id in enumerate(parent_ids):
        root = cell / "calibration" / f"shard-{index}" / "attempt-0"
        root.mkdir(parents=True)
        pair_rows = [
            {
                "parent_id": parent_id,
                "candidate_id": candidate_id,
                "pair_strict_flip": False,
                "wnode_distance": None,
            }
            for candidate_id in candidate_ids
        ]
        atomic_jsonl(root / "pair_details.jsonl", pair_rows)
        manifest = {
            "schema_version": "bace_native_baseline_verification_shard_v1",
            "dataset": "bace",
            "method": "GCFExplainer",
            "method_id": "gcfexplainer",
            "stage": CALIBRATION_STAGE,
            "status": "PASS",
            "run_complete": True,
            "action_kind": baseline_spec("gcfexplainer").action_kind,
            "rf_oracle_used": False,
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": MOLCLR_ID,
            "candidate_source_hash": candidate_sha256,
            "candidate_pool_sha256": candidate_sha256,
            "candidate_ids": candidate_ids,
            "candidate_ids_sha256": stable_sha256(candidate_ids),
            "train_candidates_root": str(predecessor),
            "source_train_candidates_root": {"path": str(predecessor)},
            "generation_root": str(predecessor.parent),
            "generation_attempt_id": "attempt-0",
            "shard_index": index,
            "num_shards": 4,
            "shard_rule": "sorted(parent_id)_position_mod_4",
            "all_parent_ids_sha256": all_parent_sha256,
            "parent_ids": [parent_id],
            "parent_count": 1,
            "pair_count": len(pair_rows),
            "pair_details_identity": file_identity(root / "pair_details.jsonl"),
            "split_identity": file_identity(split),
        }
        atomic_json(root / "run_manifest.json", manifest)
        (root / "PASS").write_text("PASS\n", encoding="utf-8")
        shard_roots.append(root)
    return {
        "cell": cell,
        "predecessor": predecessor,
        "candidate_sha256": candidate_sha256,
        "shard_roots": shard_roots,
    }


def _merge(route: dict[str, Any], output: Path) -> dict[str, Any]:
    bad_hint = (
        route["cell"]
        / "calibration"
        / "merged"
        / "attempt-failed"
        / "_native_aux"
        / "train_candidates"
    )
    return merge_fullgraph_verification_shards(
        method="GCFExplainer",
        stage=CALIBRATION_STAGE,
        shard_dirs=route["shard_roots"],
        predecessor_output=bad_hint,
        output_dir=output,
    )


def test_fresh_merge_prefers_four_shard_lineage_over_bad_cli_hint(
    tmp_path: Path,
) -> None:
    route = _build_route(tmp_path)

    result = _merge(route, tmp_path / "fresh-merge")

    resolution = result["inputs"]["predecessor_resolution"]
    assert result["status"] == "PASS"
    assert resolution["authoritative_root"] == str(route["predecessor"])
    assert resolution["candidate_pool_sha256"] == route["candidate_sha256"]
    assert resolution["generation_attempt_id"] == "attempt-0"
    assert resolution["cli_hint_status"] == "ignored_forbidden_merged_or_native_aux"
    assert (tmp_path / "fresh-merge/PASS").is_file()


def test_legacy_shards_use_only_verified_attempt_zero_fallback(tmp_path: Path) -> None:
    route = _build_route(tmp_path)
    for root in route["shard_roots"]:
        manifest = _read_json(root / "run_manifest.json")
        for field in (
            "train_candidates_root",
            "source_train_candidates_root",
            "generation_root",
            "generation_attempt_id",
        ):
            manifest.pop(field)
        atomic_json(root / "run_manifest.json", manifest)

    result = _merge(route, tmp_path / "legacy-fresh-merge")

    sources = result["inputs"]["predecessor_resolution"][
        "shard_resolution_sources"
    ]
    assert set(sources) == {"0", "1", "2", "3"}
    assert all(value == ["verified_run_root_fallback"] for value in sources.values())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("oracle_checkpoint_hash", "c" * 64, "identities differ"),
        ("molclr_checkpoint_hash", "d" * 64, "identities differ"),
        (
            "split_identity",
            {"path": "/wrong/split.csv", "size": 1, "sha256": "e" * 64},
            "identities differ",
        ),
        ("candidate_pool_sha256", "f" * 64, "hashes conflict"),
    ],
)
def test_merge_rejects_cross_shard_science_identity_changes(
    tmp_path: Path, field: str, replacement: Any, message: str
) -> None:
    route = _build_route(tmp_path)
    manifest_path = route["shard_roots"][3] / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest[field] = replacement
    atomic_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _merge(route, tmp_path / f"rejected-{field}")


def test_merge_rejects_conflicting_source_roots_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    route = _build_route(tmp_path)
    other = route["cell"] / "other_train_candidates" / "attempt-0"
    _write_source(other)
    manifest_path = route["shard_roots"][3] / "run_manifest.json"
    manifest = _read_json(manifest_path)
    for field in (
        "train_candidates_root",
        "source_train_candidates_root",
        "generation_root",
    ):
        manifest[field] = str(other)
    atomic_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="do not share one predecessor root"):
        _merge(route, tmp_path / "rejected-source")


def test_merge_rejects_merged_native_aux_as_declared_source(tmp_path: Path) -> None:
    route = _build_route(tmp_path)
    forbidden = (
        route["cell"] / "calibration" / "merged" / "_native_aux" / "attempt-0"
    )
    for root in route["shard_roots"]:
        manifest = _read_json(root / "run_manifest.json")
        manifest["train_candidates_root"] = str(forbidden)
        manifest.pop("source_train_candidates_root")
        manifest.pop("generation_root")
        atomic_json(root / "run_manifest.json", manifest)

    with pytest.raises(ValueError, match="cannot resolve through merged/_native_aux"):
        _merge(route, tmp_path / "rejected-forbidden")


def test_merge_rejects_parent_ids_outside_fixed_shard_range(tmp_path: Path) -> None:
    route = _build_route(tmp_path)
    first_root, second_root = route["shard_roots"][:2]
    first_manifest = _read_json(first_root / "run_manifest.json")
    second_manifest = _read_json(second_root / "run_manifest.json")
    first_manifest["parent_ids"], second_manifest["parent_ids"] = (
        second_manifest["parent_ids"],
        first_manifest["parent_ids"],
    )
    for root, manifest in (
        (first_root, first_manifest),
        (second_root, second_manifest),
    ):
        pair_rows = [
            {
                "parent_id": manifest["parent_ids"][0],
                "candidate_id": candidate_id,
                "pair_strict_flip": False,
                "wnode_distance": None,
            }
            for candidate_id in manifest["candidate_ids"]
        ]
        atomic_jsonl(root / "pair_details.jsonl", pair_rows)
        manifest["pair_details_identity"] = file_identity(root / "pair_details.jsonl")
        atomic_json(root / "run_manifest.json", manifest)

    with pytest.raises(ValueError, match="differs from its frozen parent range"):
        _merge(route, tmp_path / "rejected-range")
