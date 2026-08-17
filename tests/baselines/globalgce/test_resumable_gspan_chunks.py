from __future__ import annotations

import itertools
from collections import defaultdict
from types import SimpleNamespace

from src.baselines.globalgce_resumable import resumable_gspan_root_chunks


Projected = list


def PDFS(graph_id, edge, previous):
    return graph_id, edge, previous


def DFSedge(left, right, labels):
    return left, right, labels


class _NXGraph:
    def __init__(self, label: str) -> None:
        self.label = label

    def nodes(self, data=False):
        assert data is True
        return [(0, {"label": self.label}), (1, {"label": f"{self.label}x"})]

    def edges(self, data=False):
        assert data is True
        return [(0, 1, {"label": "single"})]


class _DFSCode:
    def __init__(self) -> None:
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)

    def pop(self):
        return self.values.pop()

    def get_num_vertices(self) -> int:
        return 2

    def to_graph(self, *, gid, is_undirected):
        return (gid, is_undirected, tuple(self.values[-1][2]))


class _OfficialGraph:
    def __init__(self, source_label: str, target_label: str) -> None:
        target = SimpleNamespace(vid=1, vlb=target_label)
        self.vertices = {
            0: SimpleNamespace(vid=0, vlb=source_label),
            1: target,
        }
        self.edge = SimpleNamespace(to=1, elb="single")


class _FakeGSpan:
    def __init__(self) -> None:
        self._max_num_vertices = 4
        self._min_num_vertices = 2
        self._min_support = 1
        self._is_undirected = True
        self._counter = itertools.count()
        self._DFScode = _DFSCode()
        self._frequent_subgraphs = []
        self.fs_collection = []
        self.freq_collection = []
        self._support = 0
        self.graphs = {
            0: _OfficialGraph("A", "B"),
            1: _OfficialGraph("C", "D"),
        }
        self._nx_graph_list = [_NXGraph("A"), _NXGraph("C")]

    def _read_graphs(self) -> None:
        return None

    def _generate_1edge_frequent_subgraphs(self) -> None:
        self._counter = itertools.count()

    def _get_forward_root_edges(self, graph, vertex_id):
        return [graph.edge] if vertex_id == 0 else []

    def _subgraph_mining(self, projected) -> None:
        self._support = len(projected)
        self._report(projected)

    def _from_Graph_to_nx_Graph(self, graph):
        return graph

    def _report(self, projected) -> None:
        self._frequent_subgraphs.append(tuple(self._DFScode.values))
        graph = self._DFScode.to_graph(
            gid=next(self._counter), is_undirected=self._is_undirected
        )
        self.fs_collection.append(self._from_Graph_to_nx_Graph(graph))
        self.freq_collection.append(self._support)

    def run(self):
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        root = defaultdict(Projected)
        for graph_id, graph in self.graphs.items():
            for vertex_id, vertex in graph.vertices.items():
                for edge in self._get_forward_root_edges(graph, vertex_id):
                    root[(vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)].append(
                        PDFS(graph_id, edge, None)
                    )
        for labels, projected in root.items():
            self._DFScode.append(DFSedge(0, 1, labels))
            self._subgraph_mining(projected)
            self._DFScode.pop()
        return self.fs_collection, self.freq_collection


def test_root_chunk_resume_matches_uninterrupted_reference(tmp_path) -> None:
    reference = _FakeGSpan().run()
    module = SimpleNamespace(gSpan=_FakeGSpan)

    first = _FakeGSpan()
    with resumable_gspan_root_chunks(module, checkpoint_root=tmp_path):
        first_result = first.run()

    second = _FakeGSpan()
    second._subgraph_mining = lambda _projected: (_ for _ in ()).throw(
        AssertionError("completed root should be loaded from checkpoint")
    )
    with resumable_gspan_root_chunks(module, checkpoint_root=tmp_path):
        resumed_result = second.run()

    assert first_result == reference
    assert resumed_result == reference
    parts = sorted(tmp_path.glob("support_*/root-*.pkl"))
    assert len(parts) == 2
    assert list(tmp_path.glob("support_*/checkpoint.json"))
