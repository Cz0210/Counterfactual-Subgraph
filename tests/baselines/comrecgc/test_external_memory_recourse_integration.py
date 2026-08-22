from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.contracts import RecourseParameters, sha256_file, stable_json_sha256
from src.baselines.comrecgc import recourse
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


def _official_coverage_summary(*, db_2, rec, idxs, **_kwargs):
    labels = np.asarray(db_2.labels_)
    return (
        [int(np.count_nonzero(labels >= 0))],
        [float(rec.sum().item())],
        [len(idxs)],
    )


def test_full_runner_external_engine_is_pair_label_selection_hash_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents = [
        FixtureGraph([0.0, 0.0], [], 1.0),
        FixtureGraph([0.02, 0.0], [], 2.0),
    ]
    candidates = [
        FixtureGraph([0.01, 0.0], [0.02, 0.02], 3.0),
        FixtureGraph([0.015, 0.0], [0.02, 0.02], 4.0),
        FixtureGraph([1.0, 1.0], [1.0, 1.0], 5.0),
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
