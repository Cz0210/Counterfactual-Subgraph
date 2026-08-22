"""Static fail-closed audit of COMRECGC generation-index shardability."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .contracts import UPSTREAM_COMMIT, require_empty_output, sha256_file, write_json
from .upstream import validate_upstream_checkout


SHARDABILITY_SCHEMA = "comrecgc_generation_shardability_audit_v1"
STATEFUL_GLOBALS = {
    "counterfactual_candidates",
    "graph_index_map",
    "graph_map",
    "input_graphs_covered",
    "start",
    "transitions",
    "traversed_hashes",
}


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Pinned COMRECGC must define exactly one {name}().")
    return matches[0]


def audit_source_text(source: str) -> dict[str, Any]:
    """Prove that independent generation-index ranges cannot be merged."""

    tree = ast.parse(source)
    move = _function(tree, "move_to_next_graph")
    restart = _function(tree, "restart_randomwalk")
    outer = _function(tree, "counterfactual_summary_with_randomwalk")
    nodes = [*ast.walk(move), *ast.walk(restart), *ast.walk(outer)]
    referenced_globals = sorted(
        {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and node.id in STATEFUL_GLOBALS
        }
    )
    random_calls = sorted(
        {
            node.func.attr
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
        }
    )
    required_globals = {
        "counterfactual_candidates",
        "graph_map",
        "input_graphs_covered",
        "transitions",
    }
    required_random = {"choices", "uniform"}
    evidence_complete = required_globals.issubset(referenced_globals) and (
        required_random.issubset(random_calls)
    )
    if not evidence_complete:
        raise ValueError(
            "Pinned COMRECGC state/RNG structure changed; shardability must be "
            "re-audited manually before any split is permitted."
        )
    return {
        "schema_version": SHARDABILITY_SCHEMA,
        "status": "PASS",
        "generation_index_shardable": False,
        "requested_shards": 8,
        "forbidden_partition": "independent_step_ranges_0_through_49999",
        "referenced_stateful_globals": referenced_globals,
        "random_calls_in_state_machine": random_calls,
        "reason": (
            "Each step consumes one shared RNG stream and mutates global "
            "candidate frequencies/order, coverage, graph, and transition state; "
            "restart probabilities depend on all preceding steps."
        ),
        "allowed_parallel_boundary": (
            "pure ordered graph decode/featurization below the single producer"
        ),
        "seed_merge_supported": False,
        "cross_shard_lineage_merge_supported": False,
    }


def audit_generation_shardability(
    *, upstream_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    checkout = validate_upstream_checkout(upstream_root)
    source_path = checkout / "comrecgc.py"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    result = audit_source_text(source_path.read_text(encoding="utf-8"))
    result.update(
        {
            "upstream_root": str(checkout),
            "upstream_commit": UPSTREAM_COMMIT,
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
        }
    )
    root = require_empty_output(output_dir)
    write_json(root / "shardability_audit.json", result)
    write_json(root / "AUDIT_PASS.json", result)
    return result


__all__ = [
    "SHARDABILITY_SCHEMA",
    "audit_generation_shardability",
    "audit_source_text",
]
