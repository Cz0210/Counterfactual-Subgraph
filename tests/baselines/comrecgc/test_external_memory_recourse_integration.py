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
from src.baselines.comrecgc.external_memory_dbscan import (
    ExternalMemoryDBSCANError,
    _rss_bytes,
)
from src.baselines.comrecgc.close_pair_view import (
    ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
    FILTER_OPERATOR,
    NORMALIZED_DISTANCE_CONTRACT,
    PAIR_ORDER,
    PAIR_ORIENTATION,
    SCALE_CONTRACT,
    ThetaClosePairContract,
    materialize_theta_close_pair_view,
)
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


def _official_multi_coverage_summary(
    *, db_2, rec, idxs, radius, threshold_theta, recourse_size
):
    labels = np.asarray(db_2.labels_)
    common = {}
    norms = {}
    hashes = {}
    for label in range(int(labels.max()) + 1):
        positions = np.flatnonzero(labels == label)
        points = rec[labels == label]
        center = torch.mean(points, dim=0)
        distances = torch.norm(points - center, dim=-1)
        parents = set()
        candidates = set()
        for local, distance in enumerate(distances):
            if distance < radius:
                parent, candidate = idxs[int(positions[local])]
                if int(parent) not in parents:
                    parents.add(int(parent))
                    candidates.add(int(candidate))
        common[label] = parents
        norms[label] = torch.norm(center).item()
        hashes[label] = candidates
    filtered = {
        label: set(parents)
        for label, parents in common.items()
        if norms[label] < threshold_theta
    }
    covered_by = {}
    for label, parents in filtered.items():
        for parent in parents:
            covered_by.setdefault(parent, set()).add(label)
    selected = _greedy(
        counterfactual_covering=filtered,
        graphs_covered_by=covered_by,
        k=min(recourse_size, len(filtered)),
    )
    covering, costs, sizes = [], [], []
    cost = 0.0
    candidates = set()
    for rank in selected:
        label, cumulative = selected[rank]
        covering.append(cumulative)
        cost += norms[label]
        costs.append(cost)
        candidates.update(hashes[label])
        sizes.append(len(candidates))
    return covering, costs, sizes


@pytest.mark.parametrize(
    "candidate_embeddings,expected_clusters",
    [([0.0, 0.10], 2), ([0.0, 0.10, 0.18], 3)],
)
def test_full_runner_component_recovery_streams_multi_cluster_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_embeddings: list[float],
    expected_clusters: int,
) -> None:
    zero = [0.0] * 64
    parents = [FixtureGraph(zero, [], float(index + 1)) for index in range(3)]
    candidates = []
    for index, value in enumerate(candidate_embeddings):
        embedding = [value, *([0.0] * 63)]
        candidates.append(
            FixtureGraph(embedding, [0.02, 0.02, 0.02], float(index + 10))
        )
    payload = {
        "counterfactual_candidates": [
            {"importance_parts": [1.0], "graph_hash": f"hash-{index}"}
            for index in range(len(candidates))
        ],
        "graph_map": {
            f"hash-{index}": (graph, {})
            for index, graph in enumerate(candidates)
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
                "parent_limit": 3,
                "counterfactuals_path": str(counterfactuals),
                "counterfactuals_sha256": sha256_file(counterfactuals),
                "oracle_backend": "rf",
                "classifier_family": "random_forest",
                "rf_oracle_used": True,
            }
        )
    )
    distance = tmp_path / "distance.pt"
    distance.write_bytes(b"distance")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("smiles\nC\n")
    bundle = SimpleNamespace(
        graphs=parents,
        parent_ids=[f"AIDS_{index}" for index in range(3)],
        atom_vocabulary=["C"],
        dataset_fingerprint="multi-component-fixture",
        audit=lambda: {
            "dataset_fingerprint": "multi-component-fixture",
            "generation_parent_ids_sha256": stable_json_sha256(
                [f"AIDS_{index}" for index in range(3)]
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
    util = SimpleNamespace(
        graph_element_counts=lambda graphs: torch.ones(
            len(graphs), dtype=torch.float32
        )
    )

    @contextmanager
    def imported_modules(_root):
        yield {
            "util": util,
            "common_recourse": SimpleNamespace(
                coverage_summary=_official_multi_coverage_summary,
                greedy_counterfactual_summary_from_covering_sets=_greedy,
            ),
        }

    monkeypatch.setattr(recourse, "imported_upstream", imported_modules)
    parameters = RecourseParameters.for_mode("full")
    common = dict(
        upstream_root=tmp_path,
        dataset="aids",
        dataset_dir=dataset_dir,
        source_csv=source_csv,
        generation_dir=generation,
        distance_checkpoint=distance,
        mode="full",
        parent_limit=3,
        parameters=parameters,
        device="cpu",
        batch_size=2,
    )
    legacy_root = tmp_path / "legacy"
    external_root = tmp_path / "external"
    legacy = recourse.run_common_recourse(
        **common, output_dir=legacy_root, engine="legacy_in_memory"
    )
    external = recourse.run_common_recourse(
        **common,
        output_dir=external_root,
        engine="external_memory_exact_v1",
        external_max_rss_bytes=_rss_bytes() + 512 * 1024**2,
        external_query_block_size=2,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_shortcut_seed_count=3,
        external_shortcut_failure_cap=100,
        external_shortcut_query_block_size=2,
        external_exact_fallback_max_samples=0,
        external_summary_block_size=2,
        expected_sklearn_version=sklearn.__version__,
    )
    dbscan_manifest = json.loads(
        Path(external["external_memory_artifacts"]["dbscan_manifest"]).read_text()
    )
    component_summary = json.loads(
        Path(
            external["external_memory_artifacts"][
                "all_core_component_summary_manifest"
            ]
        ).read_text()
    )
    assert dbscan_manifest["cluster_count"] == expected_clusters
    assert component_summary["cluster_count"] == expected_clusters
    assert component_summary["large_cluster_advanced_index_copy"] is False
    assert external["official_coverage_summary_result"] == legacy[
        "official_coverage_summary_result"
    ]
    assert json.loads(
        (external_root / "selected_common_recourses.json").read_text()
    ) == json.loads((legacy_root / "selected_common_recourses.json").read_text())
    terminal_path = external_root / "_RUN_COMPLETE.json"
    continuation._validate_common_recourse_completion(
        marker=terminal_path,
        terminal=json.loads(terminal_path.read_text()),
    )

    # The completed component DBSCAN can be consumed by a fresh postprocess
    # root without creating another DBSCAN directory or changing source bytes.
    source_pair_manifest = (
        external_root / "external_memory/pair_store/run_manifest.json"
    )
    source_pair = json.loads(source_pair_manifest.read_text(encoding="utf-8"))
    pair_rows = int(source_pair["row_count"])
    distances = tmp_path / "postprocess-normalized-distances.npy"
    np.save(distances, np.full(pair_rows, 0.01, dtype=np.float32))
    pair_semantics = tmp_path / "postprocess-pair-semantics.json"
    pair_semantics.write_text(json.dumps({"fixture": "all-close"}))
    close_contract = ThetaClosePairContract(
        theta=parameters.theta,
        parent_count=3,
        candidate_count=len(candidate_embeddings),
        distance_checkpoint_sha256=sha256_file(distance),
        embedding_checkpoint_sha256=sha256_file(distance),
        scale_contract=SCALE_CONTRACT,
        normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
    )
    close_root = tmp_path / "postprocess-close-view"
    with pytest.raises(ExternalMemoryDBSCANError, match="ALL_PAIRS_CLOSE"):
        materialize_theta_close_pair_view(
            physical_vectors_path=source_pair["vectors_path"],
            normalized_distances_path=distances,
            output_dir=close_root,
            contract=close_contract,
            pair_semantics_contract_path=pair_semantics,
            block_size=2,
        )
    close_identity = json.loads(
        (close_root / "checkpoint.json").read_text(encoding="utf-8")
    )["scientific_identity"]
    close_certificate = tmp_path / "postprocess-all-close-certificate.json"
    close_certificate.write_text(
        json.dumps(
            {
                "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
                "status": "PASS",
                "all_pairs_close_proven": True,
                "full_distance_scan_complete": True,
                "official_sample_comparison_pass": True,
                "normalization_audit_pass": True,
                "physical_store_rows": pair_rows,
                "count_distance_le_theta": pair_rows,
                "count_distance_gt_theta": 0,
                "count_distance_eq_theta": 0,
                "theta": parameters.theta,
                "filter_operator": FILTER_OPERATOR,
                "pair_orientation": PAIR_ORIENTATION,
                "pair_order": PAIR_ORDER,
                "physical_vectors_sha256": close_identity[
                    "physical_vectors_sha256"
                ],
                "normalized_distances_sha256": close_identity[
                    "normalized_distances_sha256"
                ],
                "distance_checkpoint_sha256": sha256_file(distance),
                "embedding_checkpoint_sha256": sha256_file(distance),
                "scale_contract": SCALE_CONTRACT,
                "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
                "approximation_used": False,
            }
        )
    )
    close_view = materialize_theta_close_pair_view(
        physical_vectors_path=source_pair["vectors_path"],
        normalized_distances_path=distances,
        output_dir=close_root,
        contract=close_contract,
        pair_semantics_contract_path=pair_semantics,
        all_pairs_close_certificate_path=close_certificate,
        block_size=2,
        resume=True,
    )
    dbscan_path = Path(
        external["external_memory_artifacts"]["dbscan_manifest"]
    )
    dbscan_sha_before = sha256_file(dbscan_path)
    exact_receipt = tmp_path / "exact-recovery-receipt.json"
    exact_receipt.write_text(
        json.dumps(
            {
                "schema_version": recourse.AIDS_EXACT_RECOVERY_RECEIPT_SCHEMA,
                "status": "PASS",
                "run_complete": True,
                "recovery_only": True,
                "ordinary_pass_dependency_eligible": False,
                "dbscan_partition_proven": True,
                "dbscan_manifest_path": str(dbscan_path),
                "dbscan_manifest_sha256": dbscan_sha_before,
            }
        )
    )
    fake_proc = tmp_path / "postprocess-proc"
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
    source_dbscan = json.loads(dbscan_path.read_text(encoding="utf-8"))
    source_contract = source_dbscan["scientific_identity"]["contract"]
    postprocess_root = tmp_path / "fresh-postprocess"
    postprocess = recourse.run_common_recourse(
        **common,
        output_dir=postprocess_root,
        engine="external_memory_exact_v1",
        external_max_rss_bytes=int(source_contract["max_rss_bytes"]),
        external_query_block_size=int(source_contract["query_block_size"]),
        external_checkpoint_interval_blocks=int(
            source_contract["checkpoint_interval_blocks"]
        ),
        external_dbscan_shortcut_mode=str(source_contract["shortcut_mode"]),
        external_shortcut_seed_count=int(source_contract["shortcut_seed_count"]),
        external_shortcut_failure_cap=int(source_contract["shortcut_failure_cap"]),
        external_shortcut_query_block_size=int(
            source_contract["shortcut_query_block_size"]
        ),
        external_exact_fallback_max_samples=int(
            source_contract["exact_fallback_max_samples"]
        ),
        external_summary_block_size=2,
        external_pair_store_source_manifest=source_pair_manifest,
        external_pair_store_source_owner_root=source_pair_manifest.parent,
        external_close_pair_view_manifest=close_view.manifest_path,
        external_dbscan_source_manifest=dbscan_path,
        external_dbscan_source_receipt=exact_receipt,
        expected_sklearn_version=str(source_contract["expected_sklearn_version"]),
        resume=True,
    )
    assert sha256_file(dbscan_path) == dbscan_sha_before
    assert not (postprocess_root / "external_memory/dbscan").exists()
    adoption_path = (
        postprocess_root
        / "external_memory/dbscan_adoption/run_manifest.json"
    )
    assert adoption_path.is_file()
    assert postprocess["external_memory_artifacts"][
        "dbscan_adopted_read_only"
    ] is True
    assert json.loads(
        (postprocess_root / "selected_common_recourses.json").read_text()
    ) == json.loads((external_root / "selected_common_recourses.json").read_text())
    postprocess_terminal_path = postprocess_root / "_RUN_COMPLETE.json"
    continuation._validate_common_recourse_completion(
        marker=postprocess_terminal_path,
        terminal=json.loads(postprocess_terminal_path.read_text()),
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
    cartesian_source_root = tmp_path / "cartesian-terminal-source"
    cartesian_store = external_recourse.ExternalPairStore(
        root=cartesian_source_root / "pair_store",
        scientific_identity=source_pair["scientific_identity"],
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
    )
    cartesian_store.append(
        chunk_index=0,
        pairs=np.asarray(
            [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]],
            dtype=np.int64,
        ),
        vectors=np.zeros((6, 64), dtype=np.float32),
        chunk_identity={"fixture": "full-cartesian"},
    )
    cartesian_pair_result = cartesian_store.finalize()
    with pytest.raises(
        ExternalMemoryDBSCANError, match="UNPROVEN_CARTESIAN_DBSCAN_INPUT"
    ):
        recourse.run_common_recourse(
            **common,
            output_dir=tmp_path / "external-adopted-unproven",
            engine="external_memory_exact_v1",
            external_max_rss_bytes=_rss_bytes() + 512 * 1024**2,
            external_dbscan_shortcut_mode=(
                "all_core_one_component_adaptive_anchor_v1"
            ),
            external_pair_store_source_manifest=cartesian_pair_result.manifest_path,
            external_pair_store_source_owner_root=cartesian_source_root,
        )
    terminal_distances = tmp_path / "terminal-normalized-distances.npy"
    np.save(terminal_distances, np.full(6, 0.01, dtype=np.float32))
    terminal_contract = ThetaClosePairContract(
        theta=parameters.theta,
        parent_count=2,
        candidate_count=3,
        distance_checkpoint_sha256=sha256_file(distance),
        embedding_checkpoint_sha256=sha256_file(distance),
        scale_contract=SCALE_CONTRACT,
        normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
    )
    terminal_close_root = tmp_path / "terminal-close-view"
    terminal_pair_semantics_path = tmp_path / "terminal-pair-semantics.json"
    terminal_pair_semantics_path.write_text(
        json.dumps({"fixture": "terminal-cartesian-pair-semantics"})
    )
    with pytest.raises(ExternalMemoryDBSCANError, match="ALL_PAIRS_CLOSE"):
        materialize_theta_close_pair_view(
            physical_vectors_path=cartesian_pair_result.vectors_path,
            normalized_distances_path=terminal_distances,
            output_dir=terminal_close_root,
            contract=terminal_contract,
            pair_semantics_contract_path=terminal_pair_semantics_path,
            block_size=2,
        )
    terminal_identity = json.loads(
        (terminal_close_root / "checkpoint.json").read_text()
    )["scientific_identity"]
    terminal_certificate = {
        "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
        "status": "PASS",
        "all_pairs_close_proven": True,
        "full_distance_scan_complete": True,
        "official_sample_comparison_pass": True,
        "normalization_audit_pass": True,
        "physical_store_rows": 6,
        "count_distance_le_theta": 6,
        "count_distance_gt_theta": 0,
        "count_distance_eq_theta": 0,
        "theta": parameters.theta,
        "filter_operator": FILTER_OPERATOR,
        "pair_orientation": PAIR_ORIENTATION,
        "pair_order": PAIR_ORDER,
        "physical_vectors_sha256": terminal_identity["physical_vectors_sha256"],
        "normalized_distances_sha256": terminal_identity[
            "normalized_distances_sha256"
        ],
        "distance_checkpoint_sha256": sha256_file(distance),
        "embedding_checkpoint_sha256": sha256_file(distance),
        "scale_contract": SCALE_CONTRACT,
        "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
        "approximation_used": False,
    }
    terminal_certificate_path = tmp_path / "terminal-all-close-certificate.json"
    terminal_certificate_path.write_text(json.dumps(terminal_certificate))
    terminal_close_view = materialize_theta_close_pair_view(
        physical_vectors_path=cartesian_pair_result.vectors_path,
        normalized_distances_path=terminal_distances,
        output_dir=terminal_close_root,
        contract=terminal_contract,
        pair_semantics_contract_path=terminal_pair_semantics_path,
        all_pairs_close_certificate_path=terminal_certificate_path,
        block_size=2,
        resume=True,
    )
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
        external_pair_store_source_manifest=cartesian_pair_result.manifest_path,
        external_pair_store_source_owner_root=cartesian_source_root,
        external_close_pair_view_manifest=terminal_close_view.manifest_path,
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
        cartesian_pair_result.manifest_path
    )
    adopted_rows = json.loads(
        (adopted_root / "selected_common_recourses.json").read_text()
    )
    assert len(adopted_rows) == 1
    assert adopted_rows[0]["cluster_size"] == 6
    assert adopted_rows[0]["covered_parent_indices_native"] == [0, 1]
    assert adopted_rows[0]["member_counterfactual_indices"] == [0, 1, 2]
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
    local_vector_cache = tmp_path / "local-vector-cache"
    cache_result = chunk_cache.materialize_cartesian_chunk_vector_cache(
        source_checkpoint_path=snapshot_path,
        source_owner_root=chunk_source_root,
        persistent_root=chunk_adopted_root / "external_memory/chunk_vector_cache",
        local_cache_root=local_vector_cache,
        scratch_lock_path=tmp_path / "local-vector-cache.lock",
        route_lock_path=tmp_path / "local-route.lock",
        expected_scientific_identity=snapshot["scientific_identity"],
        expected_chunk_identities=[
            row["scientific_identity"] for row in snapshot["chunks"]
        ],
        parent_count=2,
        candidate_count=2,
        min_local_free_bytes=128,
        proc_root=fake_proc,
    )
    aligned_distances = tmp_path / "aligned-normalized-distances.npy"
    np.save(aligned_distances, np.full(4, 0.01, dtype=np.float32))
    close_contract = ThetaClosePairContract(
        theta=parameters.theta,
        parent_count=2,
        candidate_count=2,
        distance_checkpoint_sha256=sha256_file(distance),
        embedding_checkpoint_sha256=sha256_file(distance),
        scale_contract=SCALE_CONTRACT,
        normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
    )
    close_root = tmp_path / "all-close-view"
    chunk_pair_semantics_path = tmp_path / "chunk-pair-semantics.json"
    chunk_pair_semantics_path.write_text(
        json.dumps({"fixture": "chunk-cartesian-pair-semantics"})
    )
    with pytest.raises(ExternalMemoryDBSCANError, match="ALL_PAIRS_CLOSE"):
        materialize_theta_close_pair_view(
            physical_vectors_path=cache_result.vectors_path,
            normalized_distances_path=aligned_distances,
            output_dir=close_root,
            contract=close_contract,
            pair_semantics_contract_path=chunk_pair_semantics_path,
            block_size=2,
        )
    close_checkpoint = json.loads((close_root / "checkpoint.json").read_text())
    close_identity = close_checkpoint["scientific_identity"]
    all_close_certificate = {
        "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
        "status": "PASS",
        "all_pairs_close_proven": True,
        "full_distance_scan_complete": True,
        "official_sample_comparison_pass": True,
        "normalization_audit_pass": True,
        "physical_store_rows": 4,
        "count_distance_le_theta": 4,
        "count_distance_gt_theta": 0,
        "count_distance_eq_theta": 0,
        "theta": parameters.theta,
        "filter_operator": FILTER_OPERATOR,
        "pair_orientation": PAIR_ORIENTATION,
        "pair_order": PAIR_ORDER,
        "physical_vectors_sha256": close_identity["physical_vectors_sha256"],
        "normalized_distances_sha256": close_identity[
            "normalized_distances_sha256"
        ],
        "distance_checkpoint_sha256": sha256_file(distance),
        "embedding_checkpoint_sha256": sha256_file(distance),
        "scale_contract": SCALE_CONTRACT,
        "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
        "approximation_used": False,
    }
    all_close_certificate_path = tmp_path / "all-close-certificate.json"
    all_close_certificate_path.write_text(json.dumps(all_close_certificate))
    close_view = materialize_theta_close_pair_view(
        physical_vectors_path=cache_result.vectors_path,
        normalized_distances_path=aligned_distances,
        output_dir=close_root,
        contract=close_contract,
        pair_semantics_contract_path=chunk_pair_semantics_path,
        all_pairs_close_certificate_path=all_close_certificate_path,
        block_size=2,
        resume=True,
    )
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
        external_close_pair_view_manifest=close_view.manifest_path,
        external_vector_cache_root=local_vector_cache,
        external_vector_cache_lock=tmp_path / "local-vector-cache.lock",
        external_vector_cache_route_lock=tmp_path / "local-route.lock",
        external_vector_cache_min_free_bytes=128,
        external_vector_cache_proc_root=fake_proc,
        expected_sklearn_version=sklearn.__version__,
        resume=True,
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
