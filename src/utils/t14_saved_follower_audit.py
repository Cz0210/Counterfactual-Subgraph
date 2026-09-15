"""CPU-only coverage audit of previously captured follower evidence.

No transition, RNG restoration, oracle, regeneration, or tolerance adjustment.
An action receipt is not treated as proof of a graph edit or a complete argmin.
"""
from __future__ import annotations
import gzip
import json
import pickle
from pathlib import Path


def plain(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    raise TypeError(f"UNSUPPORTED_SAVED_VALUE:{type(x).__name__}")


def load_records(path):
    result = {}
    with gzip.open(path, "rb") as stream:
        for _ in range(10000):
            try:
                row = pickle.load(stream)
            except EOFError:
                break
            key = (row["step"], row["phase"])
            if key in result:
                raise ValueError("DUPLICATE_SAVED_STEP_PHASE")
            result[key] = row
        else:
            raise ValueError("SAVED_RECORD_BOUND_EXCEEDED")
    return result


def analyze_after(row):
    compact = {r["source_hash"]: r for r in row.get("compact_candidate_actions", [])}
    observation = row.get("native_observation", {})
    transitions = observation.get("selected_transitions", [])
    checks = []
    for head, transition in enumerate(transitions):
        source, target = transition["source_graph_hash"], transition["target_graph_hash"]
        saved = compact.get(source)
        item = {"head": head, "source": source, "target": target,
                "selected_action_records": plain(transition.get("action_records", [])),
                "native_valid_fullgraph": transition.get("valid_fullgraph"),
                "captured_universe_for_this_source": saved is not None,
                "graph_one_edit_independently_proven": False}
        if saved is not None:
            ids, actions = saved["target_hashes"], saved["ordered_actions"]
            if len(ids) != len(actions):
                raise ValueError("CAPTURED_CANDIDATE_ACTION_LENGTH_MISMATCH")
            positions = [i for i, value in enumerate(ids) if value == target]
            item.update(captured_candidate_count=len(ids), target_positions=positions,
                        actions_at_target=[plain(actions[i]) for i in positions])
        # next_importance describes only the selected graph; never rename it
        # all-candidate follower score or infer a tie from equal chosen minima.
        item["complete_follower_scores_captured"] = False
        item["score_capture_keys"] = sorted(k for k in row if any(s in k.lower() for s in ("follower", "argmin", "score")))
        checks.append(item)
    importance = observation.get("next_importance")
    return {"step": row["step"], "selected_heads": len(transitions),
            "compact_sources": len(compact), "checks": checks,
            "saved_sampling_events": len(row.get("actual_sampling_events", [])),
            "selected_only_importance_count": None if importance is None else len(importance),
            "selected_only_importance_missing": importance is None}


def first_difference(left, right, path="$"):
    left, right = plain(left), plain(right)
    if type(left) is not type(right):
        return {"path": path, "left_type": type(left).__name__, "right_type": type(right).__name__}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {"path": path, "left_keys": list(left), "right_keys": list(right)}
        for key in left:
            hit = first_difference(left[key], right[key], f"{path}.{key}")
            if hit: return hit
    elif isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for i, (a, b) in enumerate(zip(left, right)):
            hit = first_difference(a, b, f"{path}[{i}]")
            if hit: return hit
    elif left != right:
        return {"path": path, "left": left, "right": right}
    return None


def audit(root):
    names = ["replay-reference-410ade5d", "replay-lowmemory-410ade5d"]
    arms = {name: load_records(root / name / "raw_step_observations.pkl.gz") for name in names}
    if set(arms[names[0]]) != set(arms[names[1]]):
        raise ValueError("ARM_CAPTURE_STEP_SETS_DIFFER")
    report = {"state": "EVIDENCE_INSUFFICIENT_NOT_SEMANTIC_PASS", "device": "cpu",
              "new_transitions": 0, "new_self_resume_budget_used": 0,
              "old_parity_rewritten": False, "oracle_calls": 0,
              "scientific_rng_restored": False, "arms": {}, "first_action_difference": None}
    for name, records in arms.items():
        rows = [analyze_after(row) for (_, phase), row in sorted(records.items()) if phase == "AFTER"]
        report["arms"][name] = {"saved_records": len(records), "after_records": len(rows),
            "selected_head_records": sum(r["selected_heads"] for r in rows),
            "heads_with_captured_source_universe": sum(c["captured_universe_for_this_source"] for r in rows for c in r["checks"]),
            "step335": next(r for r in rows if r["step"] == 335),
            "full_graph_payload_present": False}
    a, b = (arms[n] for n in names)
    for key in sorted(a):
        if key[1] != "AFTER": continue
        la = a[key].get("native_observation", {}).get("selected_transitions", [])
        lb = b[key].get("native_observation", {}).get("selected_transitions", [])
        delta = first_difference([r.get("action_records") for r in la], [r.get("action_records") for r in lb])
        if delta:
            report["first_action_difference"] = {"step": key[0], **delta}
            report["same_step_sampling_difference"] = first_difference(a[key].get("actual_sampling_events"), b[key].get("actual_sampling_events"))
            report["same_step_rng_difference"] = first_difference(a[key]["rng"], b[key]["rng"])
            break
    report["first_blocker"] = "COMPLETE_FOLLOWER_ARGMIN_INPUTS_AND_GRAPH_ONE_EDIT_NOT_IN_THIS_CAPTURE"
    report["next_evidence"] = "Locate corresponding immutable graph/action store and actual follower score producer; selected-only scores cannot close this gap."
    return report
