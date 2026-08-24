from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.contracts import RecourseParameters, sha256_file, stable_json_sha256
from src.baselines.comrecgc import recourse
from src.baselines.comrecgc import external_memory_recourse as external_recourse
from src.baselines.comrecgc import external_pair_chunk_cache as chunk_cache
from src.baselines.comrecgc.external_memory_dbscan import _rss_bytes
from scripts.autodl import run_comrecgc_standardized_continuation as continuation


torch = pytest.importorskip("torch")
sklearn = pytest.importorskip("sklearn")


class FixtureGraph:
    def __init__(self, embedding, raw_distances, atom: float):
        self.embedding = torch.tensor(embedding, dtype=torch.float32)
        self.raw_distances = torch.tensor(raw_distances, dtype=torch.float32)
        self.x = torch.tensor([[atom]], dtype=torch.float32)
        self.edge_index = torch.empty((2, 0), dtype=torch.long)


class FixtureBatch:
    def __init__(self, graphs):
        self.graphs = list(graphs)

    @classmethod
    def from_data_list(cls, graphs):
        return cls(graphs)

    def to(self, _device):
        return self


class FixtureEmbedding:
    def eval(self):
        return self

    def embed_targets(self, _graphs):
        return None

    def embed_model(self, batch):
        return torch.stack([graph.embedding for graph in batch.graphs])

    def predict_outer_with_queries(self, graphs, batch_size):
        assert batch_size == 2
        return torch.stack([graph.raw_distances for graph in graphs])


def _greedy(counterfactual_covering, graphs_covered_by, k):
    selected = {}
    covered = set()
    for rank in range(1, k + 1):
        cluster, gains = max(
            counterfactual_covering.items(), key=lambda item: len(item[1])
        )
        covered.update(gains)
        counterfactual_covering.pop(cluster)
        for parent in gains:
            for other in graphs_covered_by[parent] - {cluster}:
                if other in counterfactual_covering:
                    counterfactual_covering[other].discard(parent)
        selected[rank] = (cluster, len(covered))
    return selected


def _official_coverage_summary(
    *, db_2, rec, idxs, radius, threshold_theta, recourse_size
):
    labels = np.asarray(db_2.labels_)
    covered = set()
    first_counterfactuals = set()
    points = rec[labels == 0]
    positions = np.flatnonzero(labels == 0)
    centroid = torch.mean(points, dim=0)
    distances = torch.norm(points - centroid, dim=-1)
    for local, distance in enumerate(distances):
        if distance < radius:
            parent, candidate = idxs[int(positions[local])]
            if int(parent) not in covered:
                covered.add(int(parent))
                first_counterfactuals.add(int(candidate))
    if torch.norm(centroid).item() >= threshold_theta or recourse_size <= 0:
        return ([], [], [])
    return (
        [len(covered)],
        [torch.norm(centroid).item()],
        [len(first_counterfactuals)],
    )


def test_full_runner_external_engine_is_pair_label_selection_hash_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zero = [0.0] * 64
    parent_one = [0.02, *([0.0] * 63)]
    candidate_zero = [0.01, *([0.0] * 63)]
    candidate_one = [0.015, *([0.0] * 63)]
    candidate_far = [1.0] * 64
    parents = [
        FixtureGraph(zero, [], 1.0),
        FixtureGraph(parent_one, [], 2.0),
    ]
    candidates = [
        FixtureGraph(candidate_zero, [0.02, 0.02], 3.0),
        FixtureGraph(candidate_one, [0.02, 0.02], 4.0),
        FixtureGraph(candidate_far, [1.0, 1.0], 5.0),
    ]
    payload = {
        "counterfactual_candidates": [
            {"importance_parts": [1.0], "graph_hash": f"hash-{index}"}
            for index in range(len(candidates))
        ],
        "graph_map": {
            f"hash-{index}": (graph, {}) for index, graph in enumerate(candidates)
        },
    }
    generation = tmp_path / "generation"
    generation.mkdir()
    counterfactuals = generation / "counterfactuals.pt"
    counterfactuals.write_bytes(b"fixture-payload")
    (generation / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_complete": True,
                "dataset": "aids",
                "mode": "full",
                "parent_limit": 2,
                "counterfactuals_path": str(counterfactuals),
                "counterfactuals_sha256": sha256_file(counterfactuals),
                "oracle_backend": "rf",
                "classifier_family": "random_forest",
                "rf_oracle_used": True,
            }
        ),
        encoding="utf-8",
    )
    distance = tmp_path / "distance.pt"
    distance.write_bytes(b"distance")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("smiles\nC\n", encoding="utf-8")
    bundle = SimpleNamespace(
        graphs=parents,
        parent_ids=["AIDS_0", "AIDS_1"],
        atom_vocabulary=["C"],
        dataset_fingerprint="fixture-dataset-fingerprint",
        audit=lambda: {
            "dataset_fingerprint": "fixture-dataset-fingerprint",
            "generation_parent_ids_sha256": stable_json_sha256(
                ["AIDS_0", "AIDS_1"]
            ),
            "source_files": [str(dataset_dir), str(source_csv)],
        },
    )

    monkeypatch.setattr(recourse, "_torch_load", lambda _path: payload)
    monkeypatch.setattr(recourse, "_torch_stack", lambda: (torch, FixtureBatch))
    monkeypatch.setattr(recourse, "load_aids_generation_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(
        recourse,
        "AIDSGreedEmbeddingAdapter",
        lambda *_args, **_kwargs: FixtureEmbedding(),
    )

    # The production upstream module is imported as an object, not a dict;
    # retain that shape for graph_element_counts.
    util = SimpleNamespace(
        graph_element_counts=lambda graphs: torch.ones(len(graphs), dtype=torch.float32)
    )

    @contextmanager
    def imported_modules(_root):
        yield {
            "util": util,
            "common_recourse": SimpleNamespace(
                coverage_summary=_official_coverage_summary,
                greedy_counterfactual_summary_from_covering_sets=_greedy,
            ),
        }

    monkeypatch.setattr(recourse, "imported_upstream", imported_modules)
    parameters = RecourseParameters.for_mode("full")
    assert (
        parameters.theta,
        parameters.delta,
        parameters.recourse_size,
        parameters.cf_size,
        parameters.cluster_size,
        parameters.seed,
    ) == (0.1, 0.02, 100, 100_000, 3, 0)
    common = dict(
        upstream_root=tmp_path,
        dataset="aids",
        dataset_dir=dataset_dir,
        source_csv=source_csv,
        generation_dir=generation,
        distance_checkpoint=distance,
        mode="full",
        parent_limit=2,
        parameters=parameters,
        device="cpu",
        batch_size=2,
    )
    legacy_root = tmp_path / "legacy"
    external_root = tmp_path / "external"
    legacy = recourse.run_common_recourse(
        **common,
        output_dir=legacy_root,
        engine="legacy_in_memory",
    )
    external_manifest = recourse.run_common_recourse(
        **common,
        output_dir=external_root,
        engine="external_memory_exact_v1",
        external_max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        external_query_block_size=2,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_shortcut_seed_count=3,
        external_shortcut_failure_cap=16,
        external_shortcut_query_block_size=2,
        external_exact_fallback_max_samples=0,
        external_summary_block_size=2,
        expected_sklearn_version=sklearn.__version__,
    )

    pair_manifest = json.loads(
        (external_root / "external_memory/pair_store/run_manifest.json").read_text()
    )
    pairs = np.load(pair_manifest["pairs_path"], allow_pickle=False)
    assert pairs.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = np.load(
        external_root / "external_memory/dbscan/labels.npy", allow_pickle=False
    )
    expected_vectors = np.load(pair_manifest["vectors_path"], allow_pickle=False)
    from sklearn.cluster import DBSCAN

    assert np.array_equal(
        labels,
        DBSCAN(eps=parameters.delta, min_samples=parameters.cluster_size).fit_predict(
            expected_vectors
        ),
    )
    dbscan_manifest = json.loads(
        Path(external_manifest["external_memory_artifacts"]["dbscan_manifest"])
        .read_text(encoding="utf-8")
    )
    proof = json.loads(
        Path(dbscan_manifest["shortcut_proof_path"]).read_text(encoding="utf-8")
    )
    assert proof["eps"] == 0.02
    assert proof["min_samples"] == 3
    assert proof["self_count_semantics"] == (
        "each sample index counts itself exactly once"
    )
    legacy_rows = json.loads((legacy_root / "selected_common_recourses.json").read_text())
    external_rows = json.loads(
        (external_root / "selected_common_recourses.json").read_text()
    )
    assert external_rows == legacy_rows
    assert stable_json_sha256(external_rows) == stable_json_sha256(legacy_rows)
    assert external_manifest["theta_eligible_pair_count"] == legacy["theta_eligible_pair_count"] == 4
    assert external_manifest["official_coverage_summary_result"] == legacy[
        "official_coverage_summary_result"
    ]
    assert external_manifest["common_recourse_engine"] == "external_memory_exact_v1"
    assert external_manifest["external_memory_artifacts"]["pair_indices_sha256"]
    assert external_manifest["external_memory_artifacts"][
        "one_cluster_summary_manifest_sha256"
    ]
    terminal_path = external_root / "_RUN_COMPLETE.json"
    terminal = json.loads(terminal_path.read_text())
    continuation._validate_common_recourse_completion(
        marker=terminal_path, terminal=terminal
    )
    checkpoint = external_root / "stage-checkpoint.json"
    argv = ["run_common_recourse", "--engine", "external_memory_exact_v1"]
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "RUNNING",
                "stage": "common_recourse",
                "argv_sha256": stable_json_sha256(argv),
                "marker": str(terminal_path),
                "required_field": "run_complete",
            }
        ),
        encoding="utf-8",
    )
    assert continuation._validate_completed_stage(
        stage="common_recourse",
        argv=argv,
        marker=terminal_path,
        required_field="run_complete",
        checkpoint_path=checkpoint,
    ) is True
    reconciled = json.loads(checkpoint.read_text())
    assert reconciled["status"] == "PASS"
    assert reconciled["reconciled_after_child_completion"] is True

    source_manifest = external_root / "external_memory/pair_store/run_manifest.json"
    source_pair = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_paths = [
        source_manifest,
        Path(source_pair["pairs_path"]),
        Path(source_pair["vectors_path"]),
    ]
    source_stats_before = {
        str(path): external_recourse._file_stat_identity(path) for path in source_paths
    }
    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    real_adopt = external_recourse.adopt_external_pair_store_read_only
    real_validate = external_recourse.validate_adopted_pair_store_read_only
    monkeypatch.setattr(
        recourse,
        "adopt_external_pair_store_read_only",
        lambda **kwargs: real_adopt(**kwargs, proc_root=fake_proc),
    )
    monkeypatch.setattr(
        continuation,
        "validate_adopted_pair_store_read_only",
        lambda path: real_validate(path, proc_root=fake_proc),
    )
    monkeypatch.setattr(
        recourse,
        "AIDSGreedEmbeddingAdapter",
        lambda *_args, **_kwargs: pytest.fail(
            "read-only pair-store adoption must not rerun pair inference"
        ),
    )
    adopted_root = tmp_path / "external-adopted"
    adopted_manifest = recourse.run_common_recourse(
        **common,
        output_dir=adopted_root,
        engine="external_memory_exact_v1",
        external_max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        external_query_block_size=2,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_shortcut_seed_count=3,
        external_shortcut_failure_cap=16,
        external_shortcut_query_block_size=2,
        external_exact_fallback_max_samples=0,
        external_summary_block_size=2,
        external_pair_store_source_manifest=source_manifest,
        external_pair_store_source_owner_root=external_root,
        expected_sklearn_version=sklearn.__version__,
    )
    assert not (adopted_root / "external_memory/pair_store").exists()
    adoption_path = (
        adopted_root / "external_memory/pair_store_adoption/run_manifest.json"
    )
    assert adoption_path.is_file()
    assert adopted_manifest["external_memory_artifacts"][
        "pair_store_adopted_read_only"
    ] is True
    assert adopted_manifest["external_memory_artifacts"]["pair_store_manifest"] == str(
        source_manifest
    )
    assert json.loads(
        (adopted_root / "selected_common_recourses.json").read_text()
    ) == external_rows
    assert {
        str(path): external_recourse._file_stat_identity(path) for path in source_paths
    } == source_stats_before
    adopted_terminal_path = adopted_root / "_RUN_COMPLETE.json"
    adopted_terminal = json.loads(adopted_terminal_path.read_text())
    assert (
        "external_memory/pair_store_adoption/run_manifest.json"
        in adopted_terminal["artifact_sha256"]
    )
    assert "external_memory/pair_store/run_manifest.json" not in adopted_terminal[
        "artifact_sha256"
    ]
    continuation._validate_common_recourse_completion(
        marker=adopted_terminal_path,
        terminal=adopted_terminal,
    )

    # A specialized source with every candidate-parent pair eligible proves
    # the chunk snapshot -> implicit Cartesian pair route end to end.
    payload_two = {
        "counterfactual_candidates": payload["counterfactual_candidates"][:2],
        "graph_map": {
            key: value
            for key, value in payload["graph_map"].items()
            if key in {"hash-0", "hash-1"}
        },
    }
    monkeypatch.setattr(recourse, "_torch_load", lambda _path: payload_two)
    monkeypatch.setattr(
        recourse,
        "AIDSGreedEmbeddingAdapter",
        lambda *_args, **_kwargs: FixtureEmbedding(),
    )
    chunk_source_root = tmp_path / "chunk-source"
    source_result = recourse.run_common_recourse(
        **common,
        output_dir=chunk_source_root,
        engine="external_memory_exact_v1",
        external_max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        external_query_block_size=2,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_shortcut_seed_count=3,
        external_shortcut_failure_cap=16,
        external_shortcut_query_block_size=2,
        external_exact_fallback_max_samples=0,
        external_summary_block_size=2,
        expected_sklearn_version=sklearn.__version__,
    )
    source_pair_root = chunk_source_root / "external_memory/pair_store"
    source_pair_manifest = json.loads(
        (source_pair_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    snapshot = {
        "schema_version": source_pair_manifest["schema_version"],
        "phase": "chunks",
        "scientific_identity": source_pair_manifest["scientific_identity"],
        "scientific_identity_sha256": source_pair_manifest[
            "scientific_identity_sha256"
        ],
        "next_chunk_index": source_pair_manifest["chunk_count"],
        "row_count": source_pair_manifest["row_count"],
        "chunks": source_pair_manifest["chunks"],
    }
    snapshot_path = tmp_path / "source-snapshot/chunk_checkpoint.snapshot.json"
    snapshot_path.parent.mkdir()
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    def portable_preallocate(path: Path, *, size: int) -> None:
        with path.open("r+b") as handle:
            handle.truncate(size)

    monkeypatch.setattr(chunk_cache, "_preallocate_file", portable_preallocate)
    monkeypatch.setattr(
        chunk_cache, "_allocated_bytes", lambda path: path.stat().st_size
    )
    monkeypatch.setattr(chunk_cache, "_statvfs_free_bytes", lambda _path: 10**12)
    monkeypatch.setattr(
        recourse,
        "AIDSGreedEmbeddingAdapter",
        lambda *_args, **_kwargs: pytest.fail(
            "Cartesian chunk adoption must not rerun pair inference"
        ),
    )
    chunk_adopted_root = tmp_path / "chunk-adopted"
    chunk_adopted = recourse.run_common_recourse(
        **common,
        output_dir=chunk_adopted_root,
        engine="external_memory_exact_v1",
        external_max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        external_query_block_size=2,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_shortcut_seed_count=3,
        external_shortcut_failure_cap=16,
        external_shortcut_query_block_size=2,
        external_exact_fallback_max_samples=0,
        external_summary_block_size=2,
        external_pair_store_source_checkpoint=snapshot_path,
        external_pair_store_source_owner_root=chunk_source_root,
        external_vector_cache_root=tmp_path / "local-vector-cache",
        external_vector_cache_lock=tmp_path / "local-vector-cache.lock",
        external_vector_cache_route_lock=tmp_path / "local-route.lock",
        external_vector_cache_min_free_bytes=128,
        external_vector_cache_proc_root=fake_proc,
        expected_sklearn_version=sklearn.__version__,
    )
    assert chunk_adopted["external_memory_artifacts"][
        "pair_chunks_adopted_read_only"
    ] is True
    assert chunk_adopted["external_memory_artifacts"][
        "pair_indices_materialized"
    ] is False
    assert not (
        chunk_adopted_root / "external_memory/chunk_vector_cache/pair_indices.npy"
    ).exists()
    assert json.loads(
        (chunk_adopted_root / "selected_common_recourses.json").read_text()
    ) == json.loads(
        (chunk_source_root / "selected_common_recourses.json").read_text()
    )
    assert chunk_adopted["official_coverage_summary_result"] == source_result[
        "official_coverage_summary_result"
    ]
    chunk_terminal_path = chunk_adopted_root / "_RUN_COMPLETE.json"
    continuation._validate_common_recourse_completion(
        marker=chunk_terminal_path,
        terminal=json.loads(chunk_terminal_path.read_text()),
    )
