from __future__ import annotations

import pytest

from src.baselines.comrecgc.shardability import audit_source_text


PINNED_STATE_MACHINE_FIXTURE = """
def move_to_next_graph(graphs_hash, start_graphs_hash, importance_args, teleport_probability):
    if random.uniform(0, 1) < teleport_probability:
        return None
    graph = graph_map[graphs_hash[0]]
    transition = transitions.get(graph)
    candidate = counterfactual_candidates[0]
    return random.choices([transition], weights=[candidate['frequency']])[0]

def restart_randomwalk(input_graphs, recourse_size, importance_args):
    probabilities = input_graphs_covered
    return random.choices(range(len(probabilities)), weights=probabilities, k=recourse_size)

def counterfactual_summary_with_randomwalk(dataset_name, input_graphs, importance_args, teleport_probability, max_steps, heads=5):
    current = restart_randomwalk(input_graphs, heads, importance_args)
    for step in range(max_steps):
        traversed_hashes.append(current)
        move_to_next_graph(current, current, importance_args, teleport_probability)
"""


def test_stateful_generation_is_explicitly_not_index_shardable() -> None:
    result = audit_source_text(PINNED_STATE_MACHINE_FIXTURE)
    assert result["status"] == "PASS"
    assert result["generation_index_shardable"] is False
    assert result["requested_shards"] == 8
    assert result["seed_merge_supported"] is False


def test_shardability_audit_fails_closed_if_rng_dependency_disappears() -> None:
    with pytest.raises(ValueError, match="re-audited manually"):
        audit_source_text(PINNED_STATE_MACHINE_FIXTURE.replace("random.uniform", "uniform"))
