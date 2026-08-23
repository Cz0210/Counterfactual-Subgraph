from __future__ import annotations

import itertools
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.baselines.globalgce_resumable import resumable_gspan_root_chunks


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
    def __init__(self, name: str) -> None:
        self.name = name

    def nodes(self, data=False):
        assert data is True
        return [(0, {"label": self.name}), (1, {"label": f"{self.name}-target"})]

    def edges(self, data=False):
        assert data is True
        return [(0, 1, {"label": "single"})]


class _InputGraph:
    def __init__(self, source: str, target: str, root: _TreeNode) -> None:
        self.vertices = {
            0: SimpleNamespace(vlb=source),
            1: SimpleNamespace(vlb=target),
        }
        self.root = root


class _ExactTopKFakeGSpan:
    crash_on_name: str | None = None

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
        self._nx_graph_list = [_FingerprintGraph("A"), _FingerprintGraph("C")]
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
    assert audit["selected_count"] == 3
    assert audit["retained_pattern_count"] == 3
    assert len(audit["selected_identity_sha256"]) == 64


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
