from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.baselines.t14_semantic_review import REQUIRED, compare_rows, diagnostic_transition_admission, read_ledger, review_three_ledgers


def row(step=1):
    value = {key: "a" * 64 for key in REQUIRED}
    value.update(schema_version="tastemolnet_t14_route_c_step_state_v1", completed_step=step,
                 sequence_id=step, test_loaded=False, calibration_loaded=False,
                 selected={"action": ["NLC", 28, 15], "probability": 0.6})
    return value


def compare(a, b):
    return compare_rows([a], [b], start_step=1, end_step=1, require_complete_schema=True)


def test_observational_only_does_not_fail():
    a, b = row(), row()
    a["pid"], b["pid"] = 1, 2
    result = compare(a, b)
    assert result["status"] == "PASS"
    assert result["category_step_counts"] == {"observational": 1}


def test_selected_action_change_is_true_discrete_failure():
    a, b = row(), row()
    b["selected"]["action"][2] = 12
    result = compare(a, b)
    assert result["status"] == "FAILED"
    assert result["first_true_discrete_difference"]["field"] == "selected.action[2]"


def test_tiny_number_is_not_waived_or_claimed_algorithm_failure():
    a, b = row(), row()
    b["selected"]["probability"] += 1e-8
    result = compare(a, b)
    assert result["status"] == "EVIDENCE_INCOMPLETE"
    assert result["first_true_discrete_difference"] is None
    assert result["numeric_tolerance_changed"] is False


def test_rng_pickle_difference_needs_real_state():
    a, b = row(), row()
    b["rng_state_sha256"] = "b" * 64
    assert compare(a, b)["status"] == "EVIDENCE_INCOMPLETE"


def test_array_dtype_shape_checked_and_raw_gap_not_ignored():
    a, b = row(), row()
    a["selected"] = {"type": "array", "shape": [1], "dtype": "<f4", "bytes": 4, "sha256": "a" * 64}
    b["selected"] = dict(a["selected"], sha256="b" * 64)
    assert compare(a, b)["status"] == "EVIDENCE_INCOMPLETE"
    b["selected"]["dtype"] = "<i4"
    assert compare(a, b)["status"] == "FAILED"


def test_hex_float_parsed_not_compared_as_repr():
    a, b = row(), row()
    a["selected"] = {"type": "float", "value": "0x1.000p-1"}
    b["selected"] = {"type": "float", "value": "0x1.0p-1"}
    assert compare(a, b)["status"] == "PASS"


def test_unequal_mapping_type_or_array_axis_cannot_pass():
    a, b = row(), row()
    b["selected"]["action"] = tuple(b["selected"]["action"])
    assert compare(a, b)["status"] == "FAILED"


def test_missing_required_field_and_missing_step_rejected():
    a = row()
    del a["selected"]
    assert compare(a, a)["status"] == "EVIDENCE_INCOMPLETE"
    assert compare_rows([row()], [], start_step=1, end_step=1)["status"] == "FAILED"


def test_scan_continues_past_early_numeric_difference():
    a = [row(i) for i in range(1, 501)]
    b = deepcopy(a)
    b[0]["selected"]["probability"] += 1e-8
    b[334]["selected"]["action"][2] = 12
    result = compare_rows(a, b, start_step=1, end_step=500)
    assert result["first_semantic_divergence_step"] == 1
    assert result["first_true_discrete_difference"]["step"] == 335
    assert result["examined_step_count"] == 500


def test_three_failure_receipts_preserved_and_old_inputs_unchanged(tmp_path):
    sources = []
    originals = []
    for name in ("reference", "continuous", "reload"):
        path = tmp_path / f"{name}.jsonl"
        rows = [row(i) for i in range(1, 511)]
        if name != "reference":
            rows[334]["selected"]["action"][2] = 12
        if name == "reload":
            rows[500]["rng_state_sha256"] = "c" * 64
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        sources.append(path)
        originals.append(path.read_bytes())
    result = review_three_ledgers(*sources, output_root=tmp_path / "fresh-review")
    assert result["status"] == "FAILED"
    assert len(result["receipts"]) == 3
    assert all(Path(r["path"]).is_file() for r in result["receipts"].values())
    assert result["all_three_receipts_written_before_disposition"] is True
    assert result["new_science_transitions"] == 0
    assert originals == [p.read_bytes() for p in sources]
    with pytest.raises(FileExistsError):
        review_three_ledgers(*sources, output_root=tmp_path / "fresh-review")


def test_bad_sequence_and_test_inputs_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row(2)) + "\n")
    with pytest.raises(ValueError, match="noncontiguous"):
        read_ledger(path)
    r = row()
    r["test_loaded"] = True
    path.write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="non-train"):
        read_ledger(path)


def test_owner_writes_before_failed_gate_without_starting_science():
    source = (Path(__file__).resolve().parents[2] / "scripts/autodl/run_t14_route_c_owner.py").read_text()
    assert source.count("receipts = _compare_and_record_canaries(") == 2
    assert "review_three_ledgers(reference, continuous, reload, output_root=root)" in source


def test_access_replay_is_not_hidden_inside_budget():
    result = diagnostic_transition_admission(start_boundaries={"reference": 250, "continuous": 250}, end_step=335)
    assert result["requested"] == 170
    assert result["shortfall"] == 106
    assert result["status"] == "BLOCKED_DIAGNOSTIC_TRANSITION_BUDGET"
    assert result["science_launched"] is False
    assert diagnostic_transition_admission(start_boundaries={"a": 500, "b": 500}, end_step=501)["requested"] == 2
