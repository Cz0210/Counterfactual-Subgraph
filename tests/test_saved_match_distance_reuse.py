"""Tiny completed JSON fixtures; no model, molecule inference or OT."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src.ablations.llm.saved_match_distance_reuse import SavedMatchDistanceReuse
from src.eval.bace_frozen_gnn_contracts import stable_sha256
from src.chem.hard_deletion import CONNECTED_MATCH_SELECTION_POLICY


DISTANCE_VERSION = "molclr_node_wasserstein_exact_emd2_v1"


class FakeDistance:
    def __init__(self):
        self.calls = []
        self.closed = False

    def distance_for_action(self, parent, residual, *, action_context):
        self.calls.append((parent, residual, action_context))
        return {"distance": 0.75, "ok": True, "cache_hit": False}

    def stats_dict(self):
        return {"pair_distance_cache_hits": 0}

    def close(self):
        self.closed = True


class SavedDistanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "completed"
        self.output = self.base / "fresh"
        self.source.mkdir()
        self.output.mkdir()
        (self.source / "writer.lock").touch()
        self.contract = dict(task_spec_sha256="task", reference_sha256="ref",
            bundle_sha256="bundle", variant="CHEMLLM_7B_OFF_THE_SHELF",
            selector_input_sha256="selector", cohort_definition="label1",
            pool={"pool_sha256": "pool", "receipt_sha256": "receipt"},
            scientific_sources={key: key for key in
                ("cpu_evaluation.py", "bace_frozen_gnn_pool.py", "bace_frozen_gnn_verification.py")})
        self.context = dict(parent_id="p", candidate_id="c", match_index=0,
            match_atom_indices=[1], teacher_sha256="oracle", oracle_checkpoint_id="oracle",
            action_semantics_version="connected_sanitized_residual_v1",
            match_selection_policy=CONNECTED_MATCH_SELECTION_POLICY,
            distance_implementation_version=DISTANCE_VERSION)
        self.row = {key: self.context[key] for key in
            ("parent_id", "candidate_id", "match_index", "match_atom_indices", "action_semantics_version")}
        self.row.update(parent_smiles="CCO", residual_smiles="CC", canonical_fragment="O",
            oracle_checkpoint_hash="oracle", wnode_distance=0.2, distance_ok=True,
            cf_flip=True, delete_valid=True, sanitize_ok=True, residual_connected=True,
            rf_oracle_used=False, oracle_backend="gnn")
        self.files = {}
        manifest = {**self.contract, "execution_commit": "a" * 40, "binding_sha256": "old"}
        manifest["self_sha256"] = stable_sha256(manifest)
        self.write("run_manifest.json", manifest)
        self.write("candidate_universe.jsonl", [{"candidate_id": "c", "canonical_fragment": "O"}], True)
        for split in ("calibration", "test"):
            science = {"match_rows": [self.row], "pair_rows": []}
            self.write("parent_checkpoints/" + split + "/p.json",
                       {"science": science, "science_sha256": stable_sha256(science)})
        self.audit = dict(state="PASS", main_matrix_write=False, binding_sha256="old", files=self.files)
        raw = json.dumps(self.audit).encode()
        (self.source / "final_audit.json").write_bytes(raw)
        self.audit_sha = hashlib.sha256(raw).hexdigest()
        repo = Path(__file__).resolve().parents[1]
        self.git = patch("src.ablations.llm.saved_match_distance_reuse.subprocess.check_output",
            side_effect=lambda args, **_: (repo / args[2].split(":", 1)[1]).read_bytes())
        self.git.start()
        fake = types.ModuleType("src.eval.bace_frozen_gnn_verification")
        fake.DISTANCE_IMPLEMENTATION_VERSION = DISTANCE_VERSION
        self.modules = patch.dict(sys.modules, {"src.eval.bace_frozen_gnn_verification": fake})
        self.modules.start()
        self.reuse = None

    def tearDown(self):
        if self.reuse is not None:
            self.reuse.close()
        self.modules.stop()
        self.git.stop()
        self.temp.cleanup()

    def write(self, name, value, jsonl=False):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (("\n".join(json.dumps(r) for r in value) + "\n") if jsonl else json.dumps(value)).encode()
        path.write_bytes(raw)
        self.files[name] = hashlib.sha256(raw).hexdigest()

    def construct(self, contract=None):
        self.reuse = SavedMatchDistanceReuse(FakeDistance(), source_root=self.source,
            audit_sha256=self.audit_sha, current_contract=contract or self.contract,
            output_root=self.output, current_binding="new")
        return self.reuse

    def test_exact_match_reuses_raw_distance_only(self):
        reuse = self.construct()
        reuse.prepare_split("calibration", [{"candidate_id": "c", "canonical_fragment": "O"}])
        result = reuse.distance_for_action("CCO", "CC", action_context=self.context)
        self.assertEqual(result["distance"], 0.2)
        self.assertNotIn("cf_flip", result)
        self.assertFalse(reuse.distance.calls)
        self.assertEqual(reuse.stats_dict()["saved_match_distance_reused"], 1)

    def test_different_match_or_graph_never_reuses(self):
        reuse = self.construct()
        reuse.prepare_split("calibration", [{"candidate_id": "c", "canonical_fragment": "O"}])
        reuse.distance_for_action("CCO", "CC", action_context={**self.context, "match_atom_indices": [2]})
        reuse.distance_for_action("CCO", "CO", action_context=self.context)
        self.assertEqual(len(reuse.distance.calls), 2)

    def test_test_records_unread_until_fresh_freeze(self):
        reuse = self.construct()
        with patch.object(reuse, "_member", wraps=reuse._member) as member:
            with self.assertRaises(FileNotFoundError):
                reuse.prepare_split("test", [{"candidate_id": "c", "canonical_fragment": "O"}])
            member.assert_not_called()
        frozen = dict(binding_sha256="new", test_loaded=False, selection_frozen=True)
        frozen["self_sha256"] = stable_sha256(frozen)
        (self.output / "selector_manifest.json").write_text(json.dumps(frozen))
        reuse.prepare_split("test", [{"candidate_id": "c", "canonical_fragment": "O"}])
        self.assertEqual(reuse.loaded_splits, ["test"])

    def test_old_freeze_binding_rejected(self):
        reuse = self.construct()
        frozen = dict(binding_sha256="old", test_loaded=False, selection_frozen=True)
        frozen["self_sha256"] = stable_sha256(frozen)
        (self.output / "selector_manifest.json").write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, "BEFORE_FRESH_FREEZE"):
            reuse.prepare_split("test", [])

    def test_changed_bundle_or_kernel_rejected(self):
        for key in ("bundle_sha256", "selector_input_sha256"):
            with self.assertRaisesRegex(ValueError, "CONTRACT_MISMATCH"):
                self.construct({**self.contract, key: "different"})
        contract = copy.deepcopy(self.contract)
        contract["scientific_sources"]["bace_frozen_gnn_verification.py"] = "different"
        with self.assertRaisesRegex(ValueError, "KERNEL_MISMATCH"):
            self.construct(contract)

    def test_source_active_writer_rejected(self):
        with (self.source / "writer.lock").open("r") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.construct()

    def test_corrupt_completed_record_rejected(self):
        reuse = self.construct()
        (self.source / "parent_checkpoints/calibration/p.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "MEMBER_HASH"):
            reuse.prepare_split("calibration", [{"candidate_id": "c", "canonical_fragment": "O"}])

    def test_canonical_identity_conflict_rejected(self):
        reuse = self.construct()
        with self.assertRaisesRegex(ValueError, "CANONICAL_RULE_CONFLICT"):
            reuse.prepare_split("calibration", [{"candidate_id": "c", "canonical_fragment": "N"}])


if __name__ == "__main__":
    unittest.main()
