from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from src.utils.main_ready_task_specs import atomic_json, stable_sha256
from src.utils.t12_shadow_recovery import (
    HARD_CAP, JointLedger, SelectedStepObserver, build_shadow_plan, cache_admission, compare_ledgers,
    finite_fresh_zero, read_ledger, require_natural_510, tensor_value,
    validate_full_parity, validate_plan,
)


class T12ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = build_shadow_plan(run_id="bounded-test", reference_root=str(self.root / "ref"),
            output_root=str(self.root / "out"), source_bindings={"active_reader": {"pid": 42, "start_ticks": 77}},
            existing_continuous_ledgers={}, activation_plan=str(self.root / "production.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_plan_540_only_when_both_missing(self):
        validate_plan(self.plan)
        self.assertEqual(self.plan["transitions_budgeted"], 540)
        self.assertEqual(self.plan["optional_extra_transitions"], 20)
        self.assertTrue(all(x["start"] >= 251 for x in self.plan["stages"]))

    def test_complete_existing_tail_reduces_budget(self):
        plan = build_shadow_plan(run_id="t", reference_root="/r", output_root="/o", source_bindings={"x": 1},
            existing_continuous_ledgers={"reference": {"complete": True}, "accelerated": {"complete": True}}, activation_plan="/p")
        self.assertEqual(plan["transitions_budgeted"], 520)

    def test_unverified_tail_not_treated_as_complete(self):
        with self.assertRaisesRegex(ValueError, "UNBOUND"):
            build_shadow_plan(run_id="t", reference_root="/r", output_root="/o", source_bindings={"x": 1},
                existing_continuous_ledgers={"reference": {"complete": False}}, activation_plan="/p")

    def test_plan_cannot_hide_transition(self):
        self.plan["stages"][0]["start"] = 250
        self.plan["plan_sha256"] = stable_sha256({k: v for k, v in self.plan.items() if k != "plan_sha256"})
        with self.assertRaisesRegex(ValueError, "HIDDEN"):
            validate_plan(self.plan)

    def test_active_reader_blocks_before_any_receipt_read(self):
        with self.assertRaisesRegex(ValueError, "MUST_FINISH_NATURALLY"):
            require_natural_510(self.plan, process_alive=lambda _pid, _ticks: True)

    def test_natural_510_metadata_and_receipt_binding(self):
        cp = Path(self.plan["reference_root"]) / "checkpoints/checkpoint-00000510.manifest.json"
        atomic_json(cp, {"checkpoint_cursor": 510})
        atomic_json(Path(self.plan["required_natural_510"]), {"status": "PASS", "reference_steps": 500,
            "reload_steps": [501, 510], "checkpoint_510": str(cp), "calibration_loaded": False, "test_loaded": False})
        self.assertEqual(require_natural_510(self.plan, process_alive=lambda *_: False)["status"], "PASS")

    def test_cache_accounting_includes_tmpfs(self):
        value = cache_admission(new_bytes=1 << 30, existing_volatile_bytes=1 << 30,
            cgroup_headroom=3 << 30, required_other_headroom=1 << 30)
        self.assertTrue(value["allowed"])
        self.assertFalse(value["persistent_checkpoint_in_tmpfs"])
        self.assertFalse(cache_admission(new_bytes=(1 << 30) + 1, existing_volatile_bytes=0,
            cgroup_headroom=10 << 30, required_other_headroom=0)["allowed"])

    def test_cache_aggregate_two_gib_cap(self):
        self.assertFalse(cache_admission(new_bytes=1 << 30, existing_volatile_bytes=(1 << 30) + 1,
            cgroup_headroom=10 << 30, required_other_headroom=0)["allowed"])

    def test_ledger_order_flush_seal_roundtrip(self):
        ledger = JointLedger(self.root / "ledger", start=251, end=252, binding_sha="bound")
        before = random.getstate()
        for step in (251, 252):
            ledger.append({"step": step, "selected_action": ["EA", 0, 1]})
        ledger.flush_before_checkpoint()
        cp = self.root / "checkpoint.json"
        atomic_json(cp, {"checkpoint_cursor": 252})
        receipt = ledger.seal(cp)
        rows = read_ledger(ledger.path, binding_sha="bound", start=251, end=252)
        self.assertEqual(len(rows), 2)
        self.assertTrue(receipt["joint_boundary_complete"])
        self.assertEqual(before, random.getstate())

    def test_gap_is_not_fabricated(self):
        ledger = JointLedger(self.root / "gap", start=251, end=252, binding_sha="x")
        with self.assertRaisesRegex(ValueError, "GAP"):
            ledger.append({"step": 252})
        ledger.close_failed()

    def test_incomplete_boundary_cannot_publish(self):
        ledger = JointLedger(self.root / "partial", start=251, end=252, binding_sha="x")
        ledger.append({"step": 251})
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            ledger.seal(self.root / "missing")
        ledger.close_failed()

    def test_missing_raw_logits_not_reconstructed(self):
        rows = [{"step": 251, "missing_raw_fields": ["RAW_CLASSIFIER_LOGITS"]}]
        self.assertEqual(compare_ledgers(rows, rows)["status"], "EVIDENCE_INCOMPLETE")

    def test_first_scientific_difference_located(self):
        result = compare_ledgers([{"step": 251, "selected_action": ["EA", 0, 1]}],
                                 [{"step": 251, "selected_action": ["ER", 0, 1]}])
        self.assertEqual(result["first_difference"], "selected_action")
        self.assertEqual(result["status"], "FAILED")

    def test_plain_pass_is_not_core_parity(self):
        with self.assertRaisesRegex(ValueError, "NOT_COMPLETE"):
            validate_full_parity({"status": "PASS"})

    def _parity(self):
        names = ("reference_500_binding", "accelerated_500_binding", "natural_510_binding",
            "reference_accelerated_251_500", "reference_reload_501_510", "accelerated_reload_501_510")
        comparisons = {}
        for name in names:
            path = self.root / (name + ".json")
            atomic_json(path, {"status": "PASS", "comparison": name, "raw_evidence_complete": True})
            comparisons[name] = {"status": "PASS", "evidence_path": str(path),
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        receipt = {"status": "T12_DIAGNOSTIC_PARITY_PASS", "comparisons": comparisons,
            "transitions_used": 540, "test_loaded": False, "checkpoint_promotion_allowed": False}
        receipt["self_sha256"] = stable_sha256(receipt)
        return receipt

    def test_independent_comparison_artifact_required(self):
        receipt = self._parity()
        validate_full_parity(receipt)
        Path(receipt["comparisons"]["natural_510_binding"]["evidence_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "ARTIFACT_INVALID"):
            validate_full_parity(receipt)

    def test_finite_activation_executes_exactly_fifteen_stages(self):
        stages, previous = [], None
        production = self.root / "science"
        for n in range(15):
            name = "T12_FRESH_FROM_ZERO_TO_00000100" if n == 0 else f"STAGE_{n}"
            row = {"stage_id": name, "required_predecessor": previous,
                "command": ["dummy", "--output-root", str(production)], "cwd": str(self.root),
                "matrix_write_allowed": False, "outputs": {"receipt": str(production / f"stage-{n}.json")}}
            row["stage_sha256"] = stable_sha256(row)
            stages.append(row)
            previous = name
        plan = {"fresh_from_zero": True, "source_checkpoint": None, "stages": stages,
                "publisher_handoff": {"canonical_locator": "/canonical/locator.json"}}
        plan["plan_sha256"] = stable_sha256(plan)
        seen = []
        def run(row):
            seen.append(row["stage_id"])
            atomic_json(Path(row["outputs"]["receipt"]), {"status": "fixture"})
            return 0
        result = finite_fresh_zero(plan, parity=self._parity(), output=self.root / "activation",
            run_stage=run, transfer_owner=lambda: {"exclusive_owner_transfer": True},
            stage_admission=lambda _: {"allowed": True})
        self.assertEqual(len(seen), 15)
        self.assertFalse(result["matrix_written"])
        self.assertEqual(result["status"], "FORMAL_CHAIN_COMPLETE_WAITING_CANONICAL_PUBLISHER")
        with self.assertRaises(FileExistsError):
            finite_fresh_zero(plan, parity=self._parity(), output=self.root / "activation",
                run_stage=run, transfer_owner=lambda: {"exclusive_owner_transfer": True},
                stage_admission=lambda _: {"allowed": True})

    def test_unbound_value_not_serialized_by_object_identity(self):
        with self.assertRaisesRegex(TypeError, "UNSUPPORTED_SCIENCE_VALUE"):
            tensor_value(object())

    def test_observer_real_action_boundary_does_not_consume_rng(self):
        # A small instrumented function with the real local variable names;
        # this is observational API coverage, not the production 3778-parent gate.
        source = '''
def move_from_known_graph(hashes, importances, coverage_matrices, importance_args):
    probabilities = [0.25, 0.75]
    selected_hash_idx = random.choices(range(len(hashes)), weights=probabilities)[0]
    return selected_hash_idx
def move_to_next_graph(graph_hash, importance_args, teleport_probability):
    selected_hash_idx = move_from_known_graph(['a', 'b'], [], [], {})
    selected_action = [('NOTHING', None, None), ('EA', 0, 1)][selected_hash_idx]
    selected_importance_parts = [0.7, 0.1]
    selected_graph = SimpleNamespace(x=[[1]], edge_index=[[], []], edge_attr=[], num_nodes=1)
    counterfactual_candidates[0]['frequency'] += 1
    return 'a', False
def counterfactual_summary_with_randomwalk():
    for step in range(251, 253):
        traversed_hashes.append('a')
        move_to_next_graph('a', {}, 0.0)
'''
        def scope():
            data = {"random": random, "SimpleNamespace": SimpleNamespace,
                "traversed_hashes": ['a'] * 250,
                "counterfactual_candidates": [{"graph_hash": 'a', "frequency": 1,
                    "importance_parts": [0.7, 0.1], "input_graphs_covering_list": [1]}],
                "graph_index_map": {"a": 0}, "input_graphs_covered": [1]}
            exec(compile(source, "/fixture/vrrw.py", "exec"), data)
            return data
        original = random.getstate()
        baseline = scope()
        baseline["counterfactual_summary_with_randomwalk"]()
        expected_rng = random.getstate()
        random.setstate(original)
        observed = scope()
        ledger = JointLedger(self.root / "observed", start=251, end=252, binding_sha="fixture")
        fake_np = SimpleNamespace(random=SimpleNamespace(get_state=lambda: ('fixture', [1, 2])))
        fake_torch = SimpleNamespace(get_rng_state=lambda: [2, 3], cuda=SimpleNamespace(is_initialized=lambda: False))
        with SelectedStepObserver(ledger, np=fake_np, torch=fake_torch).installed():
            observed["counterfactual_summary_with_randomwalk"]()
        self.assertEqual(expected_rng, random.getstate())
        self.assertEqual(baseline["counterfactual_candidates"], observed["counterfactual_candidates"])
        cp = self.root / "observed-cp.json"
        atomic_json(cp, {"checkpoint_cursor": 252})
        ledger.seal(cp)
        records = read_ledger(ledger.path, binding_sha="fixture", start=251, end=252)
        self.assertEqual(records[0]["sampling_probabilities"], [0.25, 0.75])
        self.assertIn(records[0]["selected_action"][0], ["NOTHING", "EA"])
        self.assertEqual(compare_ledgers(records, records)["status"], "EVIDENCE_INCOMPLETE")

    def test_tensor_snapshot_is_not_mutable_reference(self):
        original = {"nodes": [[1, 2]], "weights": [0.3]}
        frozen = tensor_value(original)
        original["nodes"][0][0] = 9
        self.assertEqual(frozen["nodes"], [[1, 2]])


if __name__ == "__main__":
    unittest.main()
