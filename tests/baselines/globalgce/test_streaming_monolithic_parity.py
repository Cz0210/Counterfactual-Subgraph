from __future__ import annotations

import itertools
from collections import defaultdict
from types import SimpleNamespace

from src.baselines.globalgce_resumable import resumable_gspan_root_chunks


class _Code:
    def __init__(self) -> None:
        self.value = []

    def append(self, value) -> None:
        self.value.append(value)

    def pop(self):
        return self.value.pop()

    def get_num_vertices(self) -> int:
        return 2

    def to_graph(self, **_kwargs):
        return tuple(self.value[-1][2])


class _GSpan:
    def __init__(self) -> None:
        self._max_num_vertices = 4
        self._min_num_vertices = 2
        self._min_support = 1
        self._is_undirected = True
        self._counter = itertools.count()
        self._DFScode = _Code()
        self._frequent_subgraphs = []
        self.fs_collection = []
        self.freq_collection = []
        self.graphs = {
            0: SimpleNamespace(vertices={0: SimpleNamespace(vlb="A"), 1: SimpleNamespace(vlb="B")}, edge=SimpleNamespace(to=1, elb="s")),
            1: SimpleNamespace(vertices={0: SimpleNamespace(vlb="C"), 1: SimpleNamespace(vlb="D")}, edge=SimpleNamespace(to=1, elb="s")),
        }
        self._nx_graph_list = [_NX("A"), _NX("C")]

    def _read_graphs(self):
        pass

    def _generate_1edge_frequent_subgraphs(self):
        self._counter = itertools.count()

    def _get_forward_root_edges(self, graph, vertex_id):
        return [graph.edge] if vertex_id == 0 else []

    def _subgraph_mining(self, projected):
        self._support = len(projected)
        self._report(projected)

    def _from_Graph_to_nx_Graph(self, graph):
        return graph

    def _report(self, projected):
        graph = self._DFScode.to_graph()
        self.fs_collection.append(graph)
        self.freq_collection.append(self._support)

    def run(self):
        roots = defaultdict(list)
        for graph_id, graph in self.graphs.items():
            roots[(graph.vertices[0].vlb, "s", graph.vertices[1].vlb)].append((graph_id, graph.edge, None))
        for labels, projected in roots.items():
            self._DFScode.append((0, 1, labels))
            self._subgraph_mining(projected)
            self._DFScode.pop()
        return self.fs_collection, self.freq_collection


class _NX:
    def __init__(self, label):
        self.label = label

    def nodes(self, data=False):
        return [(0, {"label": self.label})]

    def edges(self, data=False):
        return []


def test_sqlite_streaming_matches_monolithic_stable_order(tmp_path) -> None:
    reference = _GSpan().run()
    module = SimpleNamespace(
        gSpan=_GSpan,
        Projected=list,
        PDFS=lambda graph_id, edge, previous: (graph_id, edge, previous),
        DFSedge=lambda left, right, labels: (left, right, labels),
    )
    # The patch reads these helpers from run.__globals__.
    _GSpan.run.__globals__.update(
        Projected=list,
        PDFS=module.PDFS,
        DFSedge=module.DFSedge,
    )
    candidate = _GSpan()
    with resumable_gspan_root_chunks(module, checkpoint_root=tmp_path, top_k=20):
        streamed = candidate.run()
    assert streamed == reference
