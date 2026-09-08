import gzip
import json
from pathlib import Path
import pickle

import pytest

from src.baselines.t14_causal_comparison import await_and_compare, compare_replays, diff


def fixture(root: Path, change=False, extra=False, missing=False):
    root.mkdir()
    terminal = {"status": "BOUNDED_DIAGNOSTIC_335_COMPLETE", "completed_step": 335, "completed_new_transitions": 85, "started_new_transitions": 85, "formal_dispatch_allowed": False}
    (root / "terminal.json").write_text(json.dumps(terminal))
    with gzip.open(root / "raw_step_observations.pkl.gz", "wb") as stream:
        for step in range(251, 336):
            if missing and step == 270:
                continue
            native = {"api": "move_from_known_graph.return", "candidate_order": ["a", "b"], "raw_importances": [[0.25], [0.75]], "actual_probabilities": [0.25, 0.75], "selected_index": 0, "actual_source_hash": "parent", "actual_lead_head": 0}
            chosen = ["NLC", 28, 15]
            if change and step >= 260:
                native["raw_importances"][0][0] += 2**-25
            if change and step == 335:
                native["selected_index"] = 1
                chosen = ["NLC", 28, 12]
            pickle.dump({"phase": "BEFORE", "step": step, "rng": {"python": [1, 2, 3]}}, stream)
            pickle.dump({"phase": "AFTER", "step": step, "rng": {"python": [3, 4, 5]}, "actual_sampling_events": [{"api": "Random.random", "u": 0.5}, native], "compact_candidate_actions": [{"source_hash": "parent", "ordered_actions": [["NLC", 28, 15], ["NLC", 28, 12]]}], "native_observation": {"selected_transitions": [chosen]}, "loop_state": {"completed_step": step}}, stream)
        if extra:
            pickle.dump({"phase": "BEFORE", "step": 336}, stream)


def test_first_raw_drift_distinguished_from_first_selected_action(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    fixture(left)
    fixture(right, change=True)
    result = compare_replays(left, right, tmp_path / "out")
    assert result["first_observed_difference"]["step"] == 260
    assert result["first_observed_difference"]["component"] == "raw_importance"
    assert result["first_selected_action_difference_step"] == 335
    assert result["first_rng_difference"] is None
    assert result["transitions_completed"] == 170
    assert result["formal_dispatch_allowed"] is False
    assert result["starting_state250_already_different"] is None
    evidence = json.loads((tmp_path / "out/first_selected_action_difference.json").read_text())
    assert evidence["reference"]["selected_action"] == [["NLC", 28, 15]]
    assert evidence["lowmemory"]["selected_action"] == [["NLC", 28, 12]]


@pytest.mark.parametrize("kwargs", [{"missing": True}, {"extra": True}])
def test_incomplete_or_excess_evidence_not_accepted(tmp_path, kwargs):
    fixture(tmp_path / "a")
    fixture(tmp_path / "b", **kwargs)
    with pytest.raises(ValueError):
        compare_replays(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    assert not (tmp_path / "out/causal_comparison.json").exists()


def test_array_difference_reports_actual_first_value():
    import numpy as np
    a = np.arange(100, dtype=np.uint8)
    b = a.copy()
    b[41] += 1
    assert diff(a, b)["path"] == "$.values[41]"
    assert diff(a, b)["left"] == 41


def test_waiter_respects_failure_without_starting_science(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    (tmp_path / "a/failed.json").write_text('{"status":"FAILED"}')
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"total_new_transition_cap": 170, "arms": {"reference": {"output_root": str(tmp_path / "a")}, "lowmemory": {"output_root": str(tmp_path / "b")}}}))
    result = await_and_compare(campaign, tmp_path / "out", 100)
    assert result["status"] == "UPSTREAM_DIAGNOSTIC_FAILED"
    assert result["retries_started"] == 0
    assert result["formal_dispatch_allowed"] is False
