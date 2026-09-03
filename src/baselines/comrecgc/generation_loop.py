"""Project-owned, checkpointable outer loop for pinned COMRECGC generation.

The order in :func:`run_generation_loop` intentionally mirrors
``COMRECGC@122f934:comrecgc.counterfactual_summary_with_randomwalk``.  In
particular, the pinned implementation does *not* assign the non-teleport
``next_graph_hash`` back to ``cur_graph_hash``.  That surprising behavior is
part of the frozen scientific contract and is preserved here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


LOOP_STATE_SCHEMA_VERSION = "comrecgc_project_outer_loop_v1"
OFFICIAL_STATE_SCHEMA_VERSION = "comrecgc_official_runtime_state_v1"


@dataclass(frozen=True, slots=True)
class GenerationLoopState:
    """Local state at one fully completed outer-loop iteration."""

    completed_step: int
    start_graph_hashes: tuple[Any, ...]
    current_graph_hashes: tuple[Any, ...]
    restart_indices: tuple[int, ...]

    @property
    def next_step(self) -> int:
        return int(self.completed_step) + 1

    def to_checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": LOOP_STATE_SCHEMA_VERSION,
            "completed_step": int(self.completed_step),
            "next_step": self.next_step,
            "start_graph_hashes": list(self.start_graph_hashes),
            "current_graph_hashes": list(self.current_graph_hashes),
            "restart_indices": list(self.restart_indices),
        }

    @classmethod
    def from_checkpoint_state(cls, value: Mapping[str, Any]) -> "GenerationLoopState":
        if value.get("schema_version") != LOOP_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported COMRECGC outer-loop checkpoint schema.")
        completed_step = int(value.get("completed_step", -1))
        if completed_step < 0 or int(value.get("next_step", -1)) != completed_step + 1:
            raise ValueError("COMRECGC outer-loop checkpoint step is inconsistent.")
        start_hashes = tuple(value.get("start_graph_hashes") or ())
        current_hashes = tuple(value.get("current_graph_hashes") or ())
        restart_indices = tuple(int(item) for item in value.get("restart_indices") or ())
        if not start_hashes or len(start_hashes) != len(current_hashes):
            raise ValueError("COMRECGC outer-loop head state is incomplete.")
        if len(restart_indices) != len(start_hashes):
            raise ValueError("COMRECGC outer-loop restart indices are incomplete.")
        return cls(
            completed_step=completed_step,
            start_graph_hashes=start_hashes,
            current_graph_hashes=current_hashes,
            restart_indices=restart_indices,
        )


def _restart_indices(value: Any) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(item) for item in value)


def run_generation_loop(
    module: Any,
    *,
    input_graphs: Any,
    importance_args: Mapping[str, Any],
    teleport_probability: float,
    max_steps: int,
    heads: int,
    initial_state: GenerationLoopState | None = None,
    on_step_complete: Callable[[GenerationLoopState], None] | None = None,
) -> GenerationLoopState:
    """Run the exact pinned outer-loop order from a completed-step boundary.

    ``on_step_complete`` is invoked only after teleport/restart handling and
    both pinned graph-map assertions.  It is therefore safe for durable
    checkpoints and must not be called from the lower-level move wrapper.
    """

    maximum = int(max_steps)
    if maximum <= 0 or int(heads) <= 0:
        raise ValueError("COMRECGC max_steps and heads must be positive.")
    if initial_state is None:
        start_graph_hashes, indices = module.restart_randomwalk(
            input_graphs, int(heads), importance_args
        )
        current_graph_hashes = copy.deepcopy(start_graph_hashes)
        completed_step = 0
        restart_indices = _restart_indices(indices)
    else:
        if initial_state.completed_step >= maximum:
            return initial_state
        if len(initial_state.start_graph_hashes) != int(heads):
            raise ValueError("Checkpoint head count differs from the current runtime.")
        start_graph_hashes = list(initial_state.start_graph_hashes)
        current_graph_hashes = list(initial_state.current_graph_hashes)
        restart_indices = tuple(initial_state.restart_indices)
        completed_step = int(initial_state.completed_step)

    last_state = GenerationLoopState(
        completed_step=completed_step,
        start_graph_hashes=tuple(start_graph_hashes),
        current_graph_hashes=tuple(current_graph_hashes),
        restart_indices=restart_indices,
    )
    for step in range(completed_step + 1, maximum + 1):
        # Do not deepcopy here: pinned upstream appends this exact list object.
        module.traversed_hashes.append(current_graph_hashes)
        (
            _next_graph_hash,
            is_teleported,
            _recourse,
            _next_importance,
            _diff,
        ) = module.move_to_next_graph(
            graphs_hash=current_graph_hashes,
            start_graphs_hash=start_graph_hashes,
            importance_args=importance_args,
            teleport_probability=teleport_probability,
        )
        if is_teleported:
            start_graph_hashes, indices = module.restart_randomwalk(
                input_graphs, int(heads), importance_args
            )
            current_graph_hashes = copy.deepcopy(start_graph_hashes)
            restart_indices = _restart_indices(indices)

        # These checks are deliberately before the completed-step callback.
        assert len(module.graph_map) == len(module.graph_index_map)
        assert set(module.graph_index_map.keys()) == set(module.graph_map.keys())

        last_state = GenerationLoopState(
            completed_step=step,
            start_graph_hashes=tuple(start_graph_hashes),
            current_graph_hashes=tuple(current_graph_hashes),
            restart_indices=restart_indices,
        )
        if on_step_complete is not None:
            on_step_complete(last_state)
    return last_state


def official_payload(module: Any) -> dict[str, Any]:
    """Build the payload emitted by the pinned upstream function."""

    return {
        "graph_map": module.graph_map,
        "graph_index_map": module.graph_index_map,
        "counterfactual_candidates": module.counterfactual_candidates,
        "MAX_COUNTERFACTUAL_SIZE": module.MAX_COUNTERFACTUAL_SIZE,
        "traversed_hashes": module.traversed_hashes,
        "input_graphs_covered": module.input_graphs_covered,
    }


def save_official_payload(module: Any, *, dataset_name: str, heads: int) -> Path:
    """Serialize to the same relative location/name as pinned upstream."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("COMRECGC result serialization requires PyTorch.") from exc
    destination = (
        Path("results")
        / str(dataset_name)
        / "counterfactuals"
        / f"comrecgc_k_{int(heads)}.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(official_payload(module), destination)
    return destination


def snapshot_official_state(module: Any) -> dict[str, Any]:
    """Capture all mutable pinned globals without serializing patch objects."""

    graph_map = dict(module.graph_map)
    covered = module.input_graphs_covered
    if hasattr(covered, "detach"):
        covered = covered.detach().cpu().clone()
    return {
        "schema_version": OFFICIAL_STATE_SCHEMA_VERSION,
        "graph_map": graph_map,
        "graph_index_map": dict(module.graph_index_map),
        "counterfactual_candidates": list(module.counterfactual_candidates),
        "input_graphs_covered": covered,
        "covering_graphs": set(module.covering_graphs),
        "start": dict(module.start),
        "is_sample": bool(module.is_sample),
        "starting_step": int(module.starting_step),
        "traversed_hashes": list(module.traversed_hashes),
        "sample_size": int(module.sample_size),
        "MAX_COUNTERFACTUAL_SIZE": int(module.MAX_COUNTERFACTUAL_SIZE),
    }


def restore_official_state(
    module: Any, value: Mapping[str, Any], *, consume: bool = False
) -> None:
    """Restore pinned globals before installing project runtime wrappers.

    ``consume`` is reserved for a trusted, already hash-validated checkpoint
    payload.  It transfers the large containers into the runtime and removes
    them from the deserialized checkpoint as it goes, avoiding a second live
    copy during T14 resume.  The default retains the historical copy semantics.
    """

    if value.get("schema_version") != OFFICIAL_STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported COMRECGC official runtime checkpoint schema.")
    required = {
        "graph_map",
        "graph_index_map",
        "counterfactual_candidates",
        "input_graphs_covered",
        "covering_graphs",
        "start",
        "traversed_hashes",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"COMRECGC official runtime checkpoint is incomplete: {missing}")
    if consume and not isinstance(value, dict):
        raise ValueError("Consumptive COMRECGC restore requires a mutable state mapping.")

    def take(name: str) -> Any:
        return value.pop(name) if consume else value[name]  # type: ignore[union-attr]

    graph_map = take("graph_map")
    graph_index_map = take("graph_index_map")
    candidates = take("counterfactual_candidates")
    covered = take("input_graphs_covered")
    covering_graphs = take("covering_graphs")
    start = take("start")
    traversed_hashes = take("traversed_hashes")
    module.graph_map = graph_map if consume else dict(graph_map)
    module.graph_index_map = graph_index_map if consume else dict(graph_index_map)
    module.counterfactual_candidates = candidates if consume else list(candidates)
    module.input_graphs_covered = (
        covered
        if consume
        else covered.clone() if hasattr(covered, "clone") else covered
    )
    module.covering_graphs = covering_graphs if consume else set(covering_graphs)
    module.transitions = {}
    module.start = start if consume else dict(start)
    module.is_sample = bool(value.get("is_sample", True))
    module.starting_step = int(value.get("starting_step", 1))
    module.traversed_hashes = traversed_hashes if consume else list(traversed_hashes)
    module.sample_size = int(value["sample_size"])
    module.MAX_COUNTERFACTUAL_SIZE = int(value["MAX_COUNTERFACTUAL_SIZE"])
    if consume:
        for name in (
            "schema_version",
            "is_sample",
            "starting_step",
            "sample_size",
            "MAX_COUNTERFACTUAL_SIZE",
        ):
            value.pop(name, None)  # type: ignore[union-attr]
