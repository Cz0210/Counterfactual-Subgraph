"""Stream the sealed T14 diagnostic evidence; never grant formal promotion."""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import pickle
import time

from src.baselines.t14_causal_diagnostic import atomic_json, first_difference, semantic


def expanded(value):
    """Only one bounded transition at a time; expose actual RNG/tensor values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": expanded(value.tolist())}
    if isinstance(value, dict):
        return {str(key): expanded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [expanded(item) for item in value]
    return semantic(value)


def diff(left, right):
    result = first_difference(expanded(left), expanded(right))
    if result and all(isinstance(result.get(key), (int, float)) and not isinstance(result[key], bool) for key in ("left", "right")):
        a, b = result["left"], result["right"]
        if math.isfinite(a) and math.isfinite(b):
            result["absolute_error"] = abs(a - b)
            result["relative_error_reference"] = abs(a-b) / abs(a) if a else None
    return result


def components(before, after):
    events = after["actual_sampling_events"]
    native = [row for row in events if row["api"] == "move_from_known_graph.return"]
    choices = [row for row in events if row["api"] == "Random.choices.return"]
    # Compare scientific event content, not elapsed time, pathname, pickle bytes,
    # or the storage representation of reference versus low-memory caches.
    return {
        "before_rng": before["rng"],
        "source_and_head": [{key: row.get(key) for key in ("actual_source_hash", "actual_lead_head")} for row in native],
        "candidate_order": [row["candidate_order"] for row in native],
        "ordered_native_actions": after["compact_candidate_actions"],
        "raw_importance": [{key: row.get(key) for key in ("raw_importances", "importance_values", "existing_candidate_frequency")} for row in native],
        "actual_probabilities": [row["actual_probabilities"] for row in native],
        "actual_cumulative_weights": [{key: row.get(key) for key in ("population", "actual_cumulative_weights", "actual_total", "k")} for row in choices],
        "actual_rng_draws": [row for row in events if row["api"] in {"Random.random", "Random.getrandbits"}],
        "selected_index": [row["selected_index"] for row in native],
        "selected_action": after["native_observation"].get("selected_transitions", ()),
        "after_rng": after["rng"],
        "loop_state": after["loop_state"],
    }


def rows(stream, step):
    before, after = pickle.load(stream), pickle.load(stream)
    if before.get("phase") != "BEFORE" or after.get("phase") != "AFTER" or before.get("step") != step or after.get("step") != step:
        raise ValueError(f"Missing/reordered T14 raw evidence at step {step}")
    return before, after


def _terminal(root):
    path = root / "terminal.json"
    if not path.is_file():
        raise ValueError(f"Unsealed T14 diagnostic root: {root}")
    row = json.loads(path.read_text())
    if row.get("status") != "BOUNDED_DIAGNOSTIC_335_COMPLETE" or row.get("completed_step") != 335 or row.get("completed_new_transitions") != 85 or row.get("started_new_transitions") != 85 or row.get("formal_dispatch_allowed") is not False:
        raise ValueError(f"Incomplete/non-diagnostic terminal: {root}")
    return row


def compare_replays(reference: Path, lowmemory: Path, output_root: Path, initial_comparison: Path | None = None) -> dict:
    """Comparison only. Inputs are trusted, locally produced sealed pickle rows."""
    terminals = [_terminal(root) for root in (reference, lowmemory)]
    initial = None
    if initial_comparison is not None:
        initial = json.loads(initial_comparison.read_text())
        if initial.get("new_transitions") != 0 or initial.get("status") not in {"STARTING_STATE_DIFFERENT", "STARTING_COMPONENTS_EQUAL"} or initial.get("formal_dispatch_allowed") is not False:
            raise ValueError("Invalid existing checkpoint250 comparison receipt")
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "causal_comparison.json"
    if result_path.exists():
        raise FileExistsError("Preserve existing diagnostic conclusion")
    first_by_component = {}
    first_selected = None
    first_rng = None
    selected_count = 0
    with gzip.open(reference / "raw_step_observations.pkl.gz", "rb") as left, gzip.open(lowmemory / "raw_step_observations.pkl.gz", "rb") as right, (output_root / "step_comparisons.jsonl").open("x") as trace:
        for step in range(251, 336):
            a = components(*rows(left, step))
            b = components(*rows(right, step))
            differences = {name: diff(a[name], b[name]) for name in a}
            for name, difference in differences.items():
                if difference is not None and name not in first_by_component:
                    first_by_component[name] = {"step": step, "component": name, **difference}
            if any(differences[name] for name in ("selected_index", "selected_action")):
                selected_count += 1
                if first_selected is None:
                    first_selected = {"step": step, "component_differences": differences,
                        "reference": {key: expanded(a[key]) for key in ("source_and_head", "candidate_order", "ordered_native_actions", "raw_importance", "actual_probabilities", "actual_cumulative_weights", "actual_rng_draws", "selected_index", "selected_action")},
                        "lowmemory": {key: expanded(b[key]) for key in ("source_and_head", "candidate_order", "ordered_native_actions", "raw_importance", "actual_probabilities", "actual_cumulative_weights", "actual_rng_draws", "selected_index", "selected_action")}}
                    atomic_json(output_root / "first_selected_action_difference.json", first_selected)
            if first_rng is None and (differences["before_rng"] or differences["after_rng"]):
                first_rng = {"step": step, "before": differences["before_rng"], "after": differences["after_rng"]}
            trace.write(json.dumps({"step": step, "component_first_differences": differences}, sort_keys=True, allow_nan=False) + "\n")
        for stream in (left, right):
            try:
                pickle.load(stream)
            except EOFError:
                continue
            raise ValueError("Extra raw transition evidence exceeds authorized170")
        trace.flush()
        os.fsync(trace.fileno())
    earliest = min(first_by_component.values(), key=lambda row: (row["step"], list(a).index(row["component"]))) if first_by_component else None
    result = {
        "status": "BOUNDED_CAUSAL_DIFFERENCES_RECORDED" if first_by_component else "BOUNDED_OBSERVATIONS_EQUAL",
        "reference": str(reference), "lowmemory": str(lowmemory),
        "scope": "OWN_CHECKPOINT250_TO335_SEQUENTIAL_DIAGNOSTIC",
        "transitions_completed": sum(row["completed_new_transitions"] for row in terminals),
        "total_transition_cap": 170, "new_transitions_computed_by_comparison": 0,
        "first_observed_difference": earliest, "first_by_component": first_by_component,
        "first_selected_action_difference_step": first_selected["step"] if first_selected else None,
        "first_rng_difference": first_rng, "selected_action_divergent_steps": selected_count,
        "starting_comparison_receipt": str(initial_comparison) if initial_comparison else None,
        "starting_state250_already_different": initial["status"] == "STARTING_STATE_DIFFERENT" if initial else None,
        "starting_difference_alone_proves_cause335": False,
        "numerical_tolerance_changed": False, "exact_comparison_is_not_a_new_parity_gate": True,
        "formal_dispatch_allowed": False, "retry3_started": False,
    }
    atomic_json(result_path, result)
    return result


def await_and_compare(campaign_path: Path, output_root: Path, max_wait_seconds: int, poll_seconds: int = 60, initial_comparison: Path | None = None):
    """One-shot postprocessing of this existing170 campaign; no science/lease."""
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("total_new_transition_cap") != 170 or set(campaign.get("arms", {})) != {"reference", "lowmemory"}:
        raise ValueError("Exact existing two-arm campaign required")
    if not 1 <= max_wait_seconds <= 48 * 3600 or poll_seconds < 30:
        raise ValueError("Bounded deadline and low-frequency checks required")
    roots = {key: Path(row["output_root"]) for key, row in campaign["arms"].items()}
    if any(not root.is_absolute() for root in roots.values()) or len(set(roots.values())) != 2:
        raise ValueError("Distinct absolute campaign roots required")
    output_root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    while True:
        for arm, root in roots.items():
            if (root / "failed.json").is_file():
                result = {"status": "UPSTREAM_DIAGNOSTIC_FAILED", "arm": arm, "failure_receipt": str(root / "failed.json"), "formal_dispatch_allowed": False, "retries_started": 0}
                atomic_json(output_root / "terminal.json", result)
                return result
        if all((root / "terminal.json").is_file() for root in roots.values()):
            result = compare_replays(roots["reference"], roots["lowmemory"], output_root, initial_comparison)
            atomic_json(output_root / "terminal.json", result)
            return result
        elapsed = time.monotonic() - start
        if elapsed >= max_wait_seconds:
            result = {"status": "WAIT_DEADLINE_REACHED", "upstream_signaled": False, "formal_dispatch_allowed": False}
            atomic_json(output_root / "terminal.json", result)
            return result
        atomic_json(output_root / "heartbeat.json", {"status": "WAITING_FOR_EXISTING_DIAGNOSTIC_TERMINALS", "sample_time": time.time(), "elapsed_seconds": elapsed, "upstream_roots": {key: str(root) for key, root in roots.items()}, "science_started": False, "gpu_lease_acquired": False})
        time.sleep(min(poll_seconds, max_wait_seconds-elapsed))
