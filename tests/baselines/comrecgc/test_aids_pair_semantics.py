from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc import aids_pair_semantics as semantics_module
from src.baselines.comrecgc.aids_pair_semantics import (
    AIDSPairSemanticsError,
    _build_all_pairs_close_certificate,
    _formula_audit,
    _resolve_pair_chunk_authority,
    _sample_rows,
    _select_candidates,
    _write_terminal_pass,
)
from src.baselines.comrecgc.close_pair_scan import PairChunk, scan_theta_close_pairs
from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256


def test_candidate_cap_is_effective_after_classifier_and_graph_resolution() -> None:
    payload = {
        "counterfactual_candidates": [
            {"importance_parts": [0.7], "graph_hash": "g0"},
            {"importance_parts": [0.49], "graph_hash": "g1"},
            {"importance_parts": [0.8], "graph_hash": "missing"},
            {"importance_parts": [0.9], "graph_hash": "g2"},
            {"importance_parts": [0.6], "graph_hash": "g3"},
        ],
        "graph_map": {
            "g0": ["graph-0"],
            "g1": ["graph-1"],
            "g2": ["graph-2"],
            "g3": ["graph-3"],
        },
    }

    selected = _select_candidates(payload, cap=2)

    assert selected.raw_candidate_count == 5
    assert selected.classifier_pass_count == 4
    assert selected.graph_resolved_classifier_pass_count == 3
    assert selected.graphs == ["graph-0", "graph-2"]
    assert selected.graph_hashes == ["g0", "g2"]
    assert selected.generation_indices == [0, 3]
    assert selected.cap == 2


def test_production_sample_contract_includes_random_chunk_ends_and_boundary() -> None:
    chunks = [
        PairChunk(0, 0, 2, 6, (0, 0), (2, 1)),
        PairChunk(1, 2, 4, 6, (0, 2), (2, 3)),
    ]
    rows, sources = _sample_rows(
        total_rows=12,
        parent_count=3,
        chunks=chunks,
        boundary_rows=[2, 7],
        seed=7,
        random_count=4,
    )

    assert np.array_equal(rows, np.asarray(sorted(sources), dtype=np.int64))
    assert {0, 5, 6, 11}.issubset(sources)
    assert sources[0] >= {"chunk_first"}
    assert sources[5] >= {"chunk_last"}
    assert sources[6] >= {"chunk_first"}
    assert sources[11] >= {"chunk_last"}
    assert sources[2] >= {"theta_boundary_closest"}
    assert sources[7] >= {"theta_boundary_closest"}
    assert sum("random" in source for source in sources.values()) == 4


class _FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeBatch:
    def __init__(self, graphs: list[dict]) -> None:
        self.graphs = graphs

    @classmethod
    def from_data_list(cls, graphs: list[dict]) -> "_FakeBatch":
        return cls(graphs)

    def to(self, _device: str) -> "_FakeBatch":
        return self


class _FakeAdapter:
    def embed_model(self, batch: _FakeBatch) -> _FakeTensor:
        return _FakeTensor([graph["embedding"] for graph in batch.graphs])


class _FakeUpstreamUtil:
    @staticmethod
    def graph_element_counts(graphs: list[dict]) -> _FakeTensor:
        return _FakeTensor([graph["count"] for graph in graphs])


def test_formula_audit_proves_pair_axis_and_candidate_minus_parent(
    tmp_path: Path,
) -> None:
    candidate_graphs = [
        {"embedding": [3.0, 0.0], "count": 2},
        {"embedding": [0.0, 5.0], "count": 4},
    ]
    parent_embeddings = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    parent_counts = np.asarray([1, 3], dtype=np.int64)
    pairs = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    vectors = np.asarray(
        [
            (np.asarray(candidate_graphs[candidate]["embedding"], dtype=np.float32)
             - parent_embeddings[parent])
            / np.float32(candidate_graphs[candidate]["count"] + parent_counts[parent])
            for parent, candidate in pairs
        ],
        dtype=np.float32,
    )
    distances = np.linalg.norm(vectors, axis=1).astype(np.float32)
    pair_path = tmp_path / "pairs.npy"
    vector_path = tmp_path / "vectors.npy"
    distance_path = tmp_path / "distances.npy"
    np.save(pair_path, pairs, allow_pickle=False)
    np.save(vector_path, vectors, allow_pickle=False)
    np.save(distance_path, distances, allow_pickle=False)

    result = _formula_audit(
        sample_rows=np.arange(4, dtype=np.int64),
        sample_sources={index: {"fixture"} for index in range(4)},
        parent_count=2,
        candidate_graphs=candidate_graphs,
        parent_embeddings=_FakeTensor(parent_embeddings),
        parent_counts=_FakeTensor(parent_counts),
        adapter=_FakeAdapter(),
        upstream_util=_FakeUpstreamUtil(),
        pair_indices_path=pair_path,
        recourse_vectors_path=vector_path,
        normalized_distances_path=distance_path,
        batch_class=_FakeBatch,
        device="cpu",
        batch_size=1,
    )

    assert result["pair_axis_mismatch_count"] == 0
    assert result["candidate_minus_parent_max_absolute_error"] == 0.0
    assert result["parent_minus_candidate_max_absolute_error"] > 0.0
    assert result["distance_vs_recomputed_recourse_norm_max_absolute_error"] < 1e-7


def _snapshot_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    chunks = [
        {
            "chunk_index": 0,
            "row_count": 6,
            "first_pair": [0, 0],
            "last_pair": [2, 1],
            "scientific_identity": {
                "candidate_start": 0,
                "candidate_stop": 2,
            },
        }
    ]
    identity = {"candidate_count": 2, "parent_count": 3}
    source = {
        "run_complete": True,
        "chunk_count": 1,
        "chunks": chunks,
        "row_count": 6,
        "pairs_sha256": "pairs",
        "vectors_sha256": "vectors",
        "candidate_major_parent_minor_order": True,
        "scientific_identity": identity,
    }
    source_path = tmp_path / "source_manifest.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    wrapper = {
        "run_complete": True,
        "physical_snapshot": True,
        "physical_snapshot_schema": (
            "comrecgc_promoted_pair_store_physical_snapshot_v1"
        ),
        "chunk_count": 0,
        "chunks": [],
        "source_chunk_count": 1,
        "source_chunks_sha256": stable_json_sha256(chunks),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "row_count": 6,
        "pairs_sha256": "pairs",
        "vectors_sha256": "vectors",
        "candidate_major_parent_minor_order": True,
        "scientific_identity": identity,
    }
    wrapper_path = tmp_path / "wrapper_manifest.json"
    wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
    return wrapper_path, wrapper, source


def test_snapshot_wrapper_uses_hash_bound_source_chunks(tmp_path: Path) -> None:
    wrapper_path, wrapper, _source = _snapshot_fixture(tmp_path)

    chunks, authority = _resolve_pair_chunk_authority(wrapper_path, wrapper)

    assert chunks == [PairChunk(0, 0, 2, 6, (0, 0), (2, 1))]
    assert authority["physical_snapshot"] is True
    assert authority["source_chunk_count"] == 1
    assert authority["source_chunks_sha256"] == wrapper["source_chunks_sha256"]


def test_real_shape_snapshot_source_chunks_drive_bounded_two_chunk_scan(
    tmp_path: Path,
) -> None:
    parent_count = 1_283
    candidate_count = 71_642
    physical_rows = parent_count * candidate_count
    chunks: list[dict] = []
    for chunk_index, candidate_start in enumerate(range(0, candidate_count, 128)):
        candidate_stop = min(candidate_count, candidate_start + 128)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "row_count": (candidate_stop - candidate_start) * parent_count,
                "first_pair": [0, candidate_start],
                "last_pair": [parent_count - 1, candidate_stop - 1],
                "scientific_identity": {
                    "candidate_start": candidate_start,
                    "candidate_stop": candidate_stop,
                },
            }
        )
    assert len(chunks) == 560
    identity = {
        "candidate_count": candidate_count,
        "parent_count": parent_count,
    }
    source = {
        "run_complete": True,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "row_count": physical_rows,
        "pairs_sha256": "pairs",
        "vectors_sha256": "vectors",
        "candidate_major_parent_minor_order": True,
        "scientific_identity": identity,
    }
    source_path = tmp_path / "source-560.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    wrapper = {
        "run_complete": True,
        "physical_snapshot": True,
        "physical_snapshot_schema": (
            "comrecgc_promoted_pair_store_physical_snapshot_v1"
        ),
        "chunk_count": 0,
        "chunks": [],
        "source_chunk_count": len(chunks),
        "source_chunks_sha256": stable_json_sha256(chunks),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "row_count": physical_rows,
        "pairs_sha256": "pairs",
        "vectors_sha256": "vectors",
        "candidate_major_parent_minor_order": True,
        "scientific_identity": identity,
    }
    wrapper_path = tmp_path / "wrapper-560.json"
    wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
    resolved_chunks, _authority = _resolve_pair_chunk_authority(
        wrapper_path, wrapper
    )

    pair_path = tmp_path / "physical-pairs.npy"
    pairs = np.lib.format.open_memmap(
        pair_path,
        mode="w+",
        dtype=np.int64,
        shape=(physical_rows, 2),
    )
    bounded_rows = 2 * 128 * parent_count
    row_numbers = np.arange(bounded_rows, dtype=np.int64)
    pairs[:bounded_rows, 0] = row_numbers % parent_count
    pairs[:bounded_rows, 1] = row_numbers // parent_count
    pairs.flush()
    del pairs
    calls: list[tuple[int, int]] = []

    def provider(start: int, stop: int) -> np.ndarray:
        calls.append((start, stop))
        return np.zeros((stop - start, parent_count), dtype=np.float32)

    result = scan_theta_close_pairs(
        output_dir=tmp_path / "bounded-scan",
        pair_indices_path=pair_path,
        pair_chunks=resolved_chunks,
        parent_count=parent_count,
        candidate_count=candidate_count,
        theta=0.1,
        scientific_identity={"snapshot": "source-bound-560"},
        distance_provider=provider,
        max_chunks=2,
    )

    assert result is None
    assert calls == [(0, 128), (128, 256)]
    checkpoint = json.loads(
        (tmp_path / "bounded-scan/checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["next_chunk_index"] == 2
    assert checkpoint["rows_processed"] == 328_448
    assert checkpoint["logical_close_pair_count"] == 328_448


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_manifest_sha256", "wrong", "source manifest SHA256"),
        ("source_chunks_sha256", "wrong", "source binding failed"),
    ],
)
def test_snapshot_wrapper_rejects_unbound_source_metadata(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    wrapper_path, wrapper, _source = _snapshot_fixture(tmp_path)
    invalid = copy.deepcopy(wrapper)
    invalid[field] = value

    with pytest.raises(AIDSPairSemanticsError, match=match):
        _resolve_pair_chunk_authority(wrapper_path, invalid)


def test_snapshot_wrapper_rejects_scientific_identity_divergence(
    tmp_path: Path,
) -> None:
    wrapper_path, wrapper, _source = _snapshot_fixture(tmp_path)
    invalid = copy.deepcopy(wrapper)
    invalid["scientific_identity"] = {"candidate_count": 99, "parent_count": 3}

    with pytest.raises(AIDSPairSemanticsError, match="scientific_identity"):
        _resolve_pair_chunk_authority(wrapper_path, invalid)


def test_snapshot_wrapper_rejects_symlink_chunk_authority(tmp_path: Path) -> None:
    wrapper_path, wrapper, _source = _snapshot_fixture(tmp_path)
    link = tmp_path / "source_manifest_link.json"
    link.symlink_to(Path(wrapper["source_manifest_path"]))
    invalid = copy.deepcopy(wrapper)
    invalid["source_manifest_path"] = str(link)

    with pytest.raises(AIDSPairSemanticsError, match="must not be a symlink"):
        _resolve_pair_chunk_authority(wrapper_path, invalid)


def test_all_pairs_close_certificate_matches_exact_consumer_schema() -> None:
    certificate = _build_all_pairs_close_certificate(
        physical_pair_count=6,
        logical_close_pair_count=6,
        theta=0.1,
        count_distance_eq_theta=1,
        physical_vectors_sha256="vectors",
        normalized_distances_sha256="distances",
        close_bitmap_sha256="bitmap",
        distance_checkpoint_sha256="checkpoint",
        normalization_audit={
            "official_torch_vs_independent_numpy_exact": True,
            "max_absolute_error": 0.0,
        },
        formula_audit={
            "pair_axis_mismatch_count": 0,
            "candidate_minus_parent_max_absolute_error": 0.0,
            "distance_vs_recomputed_recourse_norm_max_absolute_error": 0.0,
        },
        formula_tolerance=1e-6,
        distance_norm_consistency_tolerance=1e-5,
    )

    assert certificate["schema_version"] == (
        "comrecgc_all_pairs_close_certificate_v1"
    )
    assert certificate["status"] == "PASS"
    assert certificate["pair_orientation"] == "col0_parent_col1_candidate"
    assert certificate["pair_order"] == "candidate_major_parent_minor"
    assert certificate["physical_store_rows"] == 6
    assert certificate["count_distance_le_theta"] == 6
    assert certificate["count_distance_gt_theta"] == 0
    assert certificate["count_distance_eq_theta"] == 1
    assert certificate["full_distance_scan_complete"] is True
    assert certificate["official_sample_comparison_pass"] is True
    assert certificate["normalization_audit_pass"] is True
    assert certificate["scale_contract"] == (
        "element_count(parent)+element_count(candidate)"
    )
    assert certificate["normalized_distance_contract"] == (
        "GREED/NeuroSED.predict_outer(parent,candidate)/"
        "(element_count(parent)+element_count(candidate))"
    )
    assert certificate["approximation_used"] is False


def test_terminal_pass_sentinel_is_exact_and_written_last(tmp_path: Path) -> None:
    final = {"physical_store_rows": 6, "logical_close_rows": 6}

    _write_terminal_pass(
        root=tmp_path,
        progress_path=tmp_path / "progress.json",
        final=final,
    )

    assert (tmp_path / "pair_semantics_audit.json").is_file()
    assert (tmp_path / "progress.json").is_file()
    assert (tmp_path / "PASS").read_bytes() == b"PASS\n"


def test_terminal_metadata_failure_never_writes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_progress(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated terminal metadata failure")

    monkeypatch.setattr(semantics_module, "_progress", fail_progress)
    with pytest.raises(RuntimeError, match="simulated terminal metadata failure"):
        _write_terminal_pass(
            root=tmp_path,
            progress_path=tmp_path / "progress.json",
            final={"physical_store_rows": 6, "logical_close_rows": 6},
        )
    assert not (tmp_path / "PASS").exists()
