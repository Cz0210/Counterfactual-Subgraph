from __future__ import annotations

import itertools
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import src.baselines.globalgce_resumable as resumable_module
from src.baselines.globalgce_resumable import (
    _graph_input_fingerprint,
    resumable_gspan_root_chunks,
    validate_exact_top_k_audit,
    validate_exact_top_k_proof_identity,
)
from src.baselines.globalgce_bace_adapter import (
    validate_bace_globalgce_terminal_artifacts,
)
from src.eval.bace_frozen_gnn_contracts import file_identity


Projected = list


def PDFS(graph_id, edge, previous):
    return graph_id, edge, previous


def DFSedge(left, right, labels):
    return left, right, labels


@dataclass
class _TreeNode:
    name: str
    support: int
    children: list["_TreeNode"] = field(default_factory=list)
    to: int = 1
    elb: str = "single"


class _DFSCode:
    def __init__(self) -> None:
        self.values: list[tuple[int, int, object]] = []

    def append(self, value) -> None:
        self.values.append(value)

    def pop(self):
        return self.values.pop()

    def get_num_vertices(self) -> int:
        return len(self.values) + 1

    def to_graph(self, *, gid, is_undirected):
        del is_undirected
        label = self.values[-1][2]
        name = label.name if isinstance(label, _TreeNode) else label[-1]
        return gid, str(name)


class _FingerprintGraph:
    def __init__(self, name: str, *, reverse: bool = False) -> None:
        self.name = name
        self.reverse = reverse

    def nodes(self, data=False):
        assert data is True
        rows = [(0, {"label": self.name}), (1, {"label": f"{self.name}-target"})]
        return list(reversed(rows)) if self.reverse else rows

    def edges(self, data=False):
        assert data is True
        return [
            ((1, 0, {"label": "single"}) if self.reverse else (0, 1, {"label": "single"}))
        ]


class _InputGraph:
    def __init__(self, source: str, target: str, root: _TreeNode) -> None:
        self.vertices = {
            0: SimpleNamespace(vlb=source),
            1: SimpleNamespace(vlb=target),
        }
        self.root = root


class _ExactTopKFakeGSpan:
    crash_on_name: str | None = None
    reverse_fingerprint_order: bool = False

    def __init__(self) -> None:
        root_a = _TreeNode(
            "root-a",
            10,
            [
                _TreeNode("a-1", 9, [_TreeNode("a-1-1", 8), _TreeNode("a-1-2", 2)]),
                _TreeNode("a-2", 7, [_TreeNode("a-2-1", 6), _TreeNode("a-2-2", 5)]),
            ],
        )
        root_b = _TreeNode(
            "root-b",
            10,
            [_TreeNode("b-1", 9, [_TreeNode("b-1-1", 8)])],
        )
        self.graphs = {
            0: _InputGraph("A", "B", root_a),
            1: _InputGraph("C", "D", root_b),
        }
        self._nx_graph_list = [
            _FingerprintGraph("A", reverse=self.reverse_fingerprint_order),
            _FingerprintGraph("C"),
        ]
        self._max_num_vertices = 20
        self._min_num_vertices = 2
        self._min_support = 1
        self._is_undirected = True
        self._counter = itertools.count()
        self._DFScode = _DFSCode()
        self._frequent_subgraphs = []
        self.fs_collection: list[str] = []
        self.freq_collection: list[int] = []
        self._support = 0

    def _read_graphs(self) -> None:
        return None

    def _generate_1edge_frequent_subgraphs(self) -> None:
        self._counter = itertools.count()

    def _get_forward_root_edges(self, graph, vertex_id):
        return [graph.root] if vertex_id == 0 else []

    def _get_support(self, projected) -> int:
        return int(projected[0][1].support)

    def _is_min(self) -> bool:
        return True

    def _from_Graph_to_nx_Graph(self, graph):
        # The pinned implementation discards the report gid when it constructs
        # the NetworkX graph, so the fixture deliberately does the same.
        return graph[1]

    def _report(self, projected) -> None:
        del projected
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        graph = self._DFScode.to_graph(
            gid=next(self._counter), is_undirected=self._is_undirected
        )
        self.fs_collection.append(self._from_Graph_to_nx_Graph(graph))
        self.freq_collection.append(int(self._support))

    def _subgraph_mining(self, projected):
        node = projected[0][1]
        self._support = self._get_support(projected)
        if self._support < self._min_support or not self._is_min():
            return None
        self._report(projected)
        if self.crash_on_name == node.name:
            raise RuntimeError("fixture interruption")
        for child in node.children:
            self._DFScode.append((0, 1, child))
            self._subgraph_mining([PDFS(0, child, projected[0])])
            self._DFScode.pop()
        return self

    def run(self):
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        roots = defaultdict(Projected)
        for graph_id, graph in self.graphs.items():
            for vertex_id, vertex in graph.vertices.items():
                for edge in self._get_forward_root_edges(graph, vertex_id):
                    roots[(vertex.vlb, "single", graph.vertices[1].vlb)].append(
                        PDFS(graph_id, edge, None)
                    )
        for labels, projected in roots.items():
            self._DFScode.append(DFSedge(0, 1, labels))
            self._subgraph_mining(projected)
            self._DFScode.pop()
        ordered = sorted(
            zip(self.freq_collection, self.fs_collection),
            reverse=True,
            key=lambda row: row[0],
        )
        self.freq_collection = [int(row[0]) for row in ordered[:3]]
        self.fs_collection = [str(row[1]) for row in ordered[:3]]
        return self.fs_collection, self.freq_collection


def _module():
    return SimpleNamespace(gSpan=_ExactTopKFakeGSpan)


def _disable_storage_floor(monkeypatch) -> None:
    monkeypatch.setenv("GLOBALGCE_STORAGE_MIN_FREE_GIB", "0")
    monkeypatch.setenv("GLOBALGCE_STORAGE_MIN_FREE_RATIO", "0")
    monkeypatch.setenv("GLOBALGCE_STORAGE_MIN_FREE_INODES", "0")


def test_exact_topk_antimonotone_pruning_matches_stable_reference(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    reference = _ExactTopKFakeGSpan().run()
    candidate = _ExactTopKFakeGSpan()
    with resumable_gspan_root_chunks(
        _module(),
        checkpoint_root=tmp_path,
        top_k=3,
        flush_every=1,
        exact_top_k_pruning=True,
    ):
        observed = candidate.run()

    assert observed == reference
    database = next(tmp_path.glob("support_*/frequent_patterns.sqlite3"))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM patterns").fetchone()[0] == 3
        assert connection.execute(
            "SELECT COALESCE(SUM(pruned_branch_count), 0) FROM root_stats"
        ).fetchone()[0] > 0
    audit = json.loads(
        next(tmp_path.glob("support_*/exact_top_k_audit.json")).read_text(
            encoding="utf-8"
        )
    )
    assert audit["run_complete"] is True
    assert audit["status"] == "PASS"
    assert audit["selected_count"] == 3
    assert audit["retained_pattern_count"] == 3
    assert len(audit["selected_identity_sha256"]) == 64
    assert validate_exact_top_k_audit(
        next(tmp_path.glob("support_*/exact_top_k_audit.json"))
    )["status"] == "PASS"


def test_exact_topk_root_restart_restores_pre_root_snapshot(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    reference = _ExactTopKFakeGSpan().run()
    _ExactTopKFakeGSpan.crash_on_name = "root-b"
    try:
        interrupted = _ExactTopKFakeGSpan()
        with pytest.raises(RuntimeError, match="fixture interruption"):
            with resumable_gspan_root_chunks(
                _module(),
                checkpoint_root=tmp_path,
                top_k=3,
                flush_every=1,
                exact_top_k_pruning=True,
            ):
                interrupted.run()
    finally:
        _ExactTopKFakeGSpan.crash_on_name = None

    resumed = _ExactTopKFakeGSpan()
    with resumable_gspan_root_chunks(
        _module(),
        checkpoint_root=tmp_path,
        top_k=3,
        flush_every=1,
        exact_top_k_pruning=True,
    ):
        observed = resumed.run()
    assert observed == reference


def test_exact_topk_requires_positive_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive top_k"):
        with resumable_gspan_root_chunks(
            _module(),
            checkpoint_root=tmp_path,
            exact_top_k_pruning=True,
        ):
            pass


def test_fingerprint_binds_graph_node_and_edge_traversal_order() -> None:
    settings = {"spill_schema": "test", "root_order": ["A"]}
    forward = _graph_input_fingerprint(
        [_FingerprintGraph("A", reverse=False)], settings
    )
    reverse = _graph_input_fingerprint(
        [_FingerprintGraph("A", reverse=True)], settings
    )
    graph_list_reverse = _graph_input_fingerprint(
        [_FingerprintGraph("C"), _FingerprintGraph("A")], settings
    )
    graph_list_forward = _graph_input_fingerprint(
        [_FingerprintGraph("A"), _FingerprintGraph("C")], settings
    )

    assert forward != reverse
    assert graph_list_forward != graph_list_reverse


def test_cross_order_resume_never_adopts_the_other_traversal(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    _ExactTopKFakeGSpan.reverse_fingerprint_order = False
    _ExactTopKFakeGSpan.crash_on_name = "root-b"
    try:
        with pytest.raises(RuntimeError, match="fixture interruption"):
            with resumable_gspan_root_chunks(
                _module(),
                checkpoint_root=tmp_path,
                top_k=3,
                flush_every=1,
                exact_top_k_pruning=True,
            ):
                _ExactTopKFakeGSpan().run()
    finally:
        _ExactTopKFakeGSpan.crash_on_name = None

    _ExactTopKFakeGSpan.reverse_fingerprint_order = True
    try:
        with resumable_gspan_root_chunks(
            _module(),
            checkpoint_root=tmp_path,
            top_k=3,
            flush_every=1,
            exact_top_k_pruning=True,
        ):
            observed = _ExactTopKFakeGSpan().run()
    finally:
        _ExactTopKFakeGSpan.reverse_fingerprint_order = False

    assert observed == _ExactTopKFakeGSpan().run()
    support_roots = sorted(tmp_path.glob("support_*"))
    assert len(support_roots) == 2
    completed = [
        root for root in support_roots if (root / "exact_top_k_audit.json").is_file()
    ]
    assert len(completed) == 1
    assert json.loads((completed[0] / "checkpoint.json").read_text())["stage"] == "complete"


def test_complete_checkpoint_failure_never_publishes_audit_and_recovers(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    original = resumable_module._atomic_json
    failed = False

    def injected(path, payload):
        nonlocal failed
        if (
            not failed
            and path.name == "checkpoint.json"
            and payload.get("stage") == "complete"
        ):
            failed = True
            raise OSError("injected complete checkpoint failure")
        return original(path, payload)

    monkeypatch.setattr(resumable_module, "_atomic_json", injected)
    with pytest.raises(OSError, match="complete checkpoint failure"):
        with resumable_gspan_root_chunks(
            _module(), checkpoint_root=tmp_path, top_k=3,
            flush_every=1, exact_top_k_pruning=True,
        ):
            _ExactTopKFakeGSpan().run()
    assert not list(tmp_path.glob("support_*/exact_top_k_audit.json"))

    monkeypatch.setattr(resumable_module, "_atomic_json", original)
    with resumable_gspan_root_chunks(
        _module(), checkpoint_root=tmp_path, top_k=3,
        flush_every=1, exact_top_k_pruning=True,
    ):
        _ExactTopKFakeGSpan().run()
    assert validate_exact_top_k_audit(
        next(tmp_path.glob("support_*/exact_top_k_audit.json"))
    )["status"] == "PASS"


def test_audit_failure_leaves_complete_checkpoint_and_recovers(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    original = resumable_module._atomic_json
    failed = False

    def injected(path, payload):
        nonlocal failed
        if not failed and path.name == "exact_top_k_audit.json":
            failed = True
            raise OSError("injected exact audit failure")
        return original(path, payload)

    monkeypatch.setattr(resumable_module, "_atomic_json", injected)
    with pytest.raises(OSError, match="exact audit failure"):
        with resumable_gspan_root_chunks(
            _module(), checkpoint_root=tmp_path, top_k=3,
            flush_every=1, exact_top_k_pruning=True,
        ):
            _ExactTopKFakeGSpan().run()
    support_root = next(tmp_path.glob("support_*"))
    assert json.loads((support_root / "checkpoint.json").read_text())["stage"] == "complete"
    assert not (support_root / "exact_top_k_audit.json").exists()

    monkeypatch.setattr(resumable_module, "_atomic_json", original)
    with resumable_gspan_root_chunks(
        _module(), checkpoint_root=tmp_path, top_k=3,
        flush_every=1, exact_top_k_pruning=True,
    ):
        _ExactTopKFakeGSpan().run()
    assert validate_exact_top_k_audit(
        support_root / "exact_top_k_audit.json"
    )["status"] == "PASS"


def test_exact_proof_identity_rejects_deleted_or_tampered_audit(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    with resumable_gspan_root_chunks(
        _module(), checkpoint_root=tmp_path, top_k=3,
        flush_every=1, exact_top_k_pruning=True,
    ):
        _ExactTopKFakeGSpan().run()
    audit = next(tmp_path.glob("support_*/exact_top_k_audit.json"))
    identity = validate_exact_top_k_audit(audit)
    original = audit.read_bytes()

    audit.unlink()
    with pytest.raises(FileNotFoundError):
        validate_exact_top_k_proof_identity(identity)
    audit.write_bytes(original)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["selected_count"] += 1
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected-payload|identity/hash"):
        validate_exact_top_k_proof_identity(identity)


def test_bace_terminal_publications_hash_bind_and_revalidate_exact_proof(
    tmp_path, monkeypatch
) -> None:
    _disable_storage_floor(monkeypatch)
    proof_root = tmp_path / "proof"
    with resumable_gspan_root_chunks(
        _module(), checkpoint_root=proof_root, top_k=3,
        flush_every=1, exact_top_k_pruning=True,
    ):
        _ExactTopKFakeGSpan().run()
    audit = next(proof_root.glob("support_*/exact_top_k_audit.json"))
    proof = validate_exact_top_k_audit(audit)

    output = tmp_path / "upper"
    output.mkdir()
    training = {
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "gspan_exact_top_k_proof": proof,
    }
    (output / "training_summary.json").write_text(
        json.dumps(training, sort_keys=True), encoding="utf-8"
    )
    training_identity = file_identity(output / "training_summary.json")
    summary = {
        "status": "PASS",
        "run_complete": True,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "training_summary": training_identity,
        "gspan_exact_top_k_proof": proof,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    summary_identity = file_identity(output / "summary.json")
    manifest = {
        **summary,
        "summary": summary_identity,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    manifest_identity = file_identity(output / "run_manifest.json")
    complete = {
        **summary,
        "summary": summary_identity,
        "run_manifest": manifest_identity,
    }
    (output / "_RUN_COMPLETE.json").write_text(
        json.dumps(complete, sort_keys=True), encoding="utf-8"
    )

    assert validate_bace_globalgce_terminal_artifacts(
        output, require_exact_top_k=True
    )["status"] == "PASS"
    original = audit.read_bytes()
    audit.unlink()
    with pytest.raises(FileNotFoundError):
        validate_bace_globalgce_terminal_artifacts(
            output, require_exact_top_k=True
        )
    audit.write_bytes(original)
    tampered = json.loads(audit.read_text(encoding="utf-8"))
    tampered["selected_rows"][0]["support"] += 1
    audit.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_bace_globalgce_terminal_artifacts(
            output, require_exact_top_k=True
        )
