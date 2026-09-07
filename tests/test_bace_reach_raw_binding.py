import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval.bace_frozen_gnn_contracts import sha256_file
from src.eval.bace_reach_raw_binding import current_raw_contract, wrap_raw_distance


class CurrentRawBindingTests(unittest.TestCase):
    def fixture(self, root):
        oracle = root / "oracle"
        source = root / "molclr"
        oracle.mkdir()
        (source / "models").mkdir(parents=True)
        (oracle / "feature_schema.json").write_text('{"atoms": [6, 8]}')
        (source / "models/ginet_molclr.py").write_text("# pinned actual encoder")
        weights = root / "model.pth"
        weights.write_bytes(b"already-receipted-weights")
        wn = {"solver": "exact_emd2"}
        ref = root / "reference.json"
        ref.write_text(json.dumps({"frozen_downstream": {"wnode_config": wn,
            "molclr_sha": "f"*64, "molclr_root": str(weights),
            "gine_checkpoint": str(oracle/"model.pt")}}))
        contract = {"paths": {"reference": str(ref), "oracle": str(oracle),
            "molclr_checkpoint": str(weights), "molclr_source": str(source)},
            "reference_sha256": sha256_file(ref), "molclr_sha256": "f"*64, "wnode_config": wn}
        def desc(p):
            return {"sha256": sha256_file(p), "size": p.stat().st_size}
        portable = {"wnode_config": wn, "feature_schema_path": "reference/gine/feature_schema.json",
            "molclr_checkpoint_path": "reference/molclr/model.pth", "molclr_source_root": "reference/molclr/source",
            "files": {"reference/gine/feature_schema.json": desc(oracle/"feature_schema.json"),
                "reference/molclr/model.pth": {"sha256": "f"*64, "size": weights.stat().st_size},
                "reference/molclr/source/models/ginet_molclr.py": desc(source/"models/ginet_molclr.py")}}
        return contract, portable

    def test_current_files_and_receipted_weights_not_index_self_assertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, portable = self.fixture(Path(tmp))
            def small_only(path):
                self.assertNotEqual(str(path), contract["paths"]["molclr_checkpoint"])
                return sha256_file(path)
            with patch("src.eval.bace_reach_raw_binding.sha256_file", side_effect=small_only):
                actual = current_raw_contract(contract, portable)
            self.assertEqual(actual["molclr_checkpoint"]["sha256"], "f"*64)
            (Path(contract["paths"]["molclr_source"])/"models/ginet_molclr.py").write_text("changed encoder")
            with self.assertRaisesRegex(ValueError, "DIFFERS_FROM_ACCEPTED"):
                current_raw_contract(contract, portable)

    def test_actual_schema_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, portable = self.fixture(Path(tmp))
            (Path(contract["paths"]["oracle"])/"feature_schema.json").write_text("changed schema")
            with self.assertRaisesRegex(ValueError, "DIFFERS_FROM_ACCEPTED"):
                current_raw_contract(contract, portable)

    def test_current_source_cannot_escape_portable_relative_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, portable = self.fixture(Path(tmp))
            portable["files"]["reference/molclr/source/../outside.py"] = {}
            with self.assertRaisesRegex(ValueError, "UNSAFE"):
                current_raw_contract(contract, portable)

    def test_train_search_cannot_read_old_cal_or_test_index(self):
        with self.assertRaisesRegex(ValueError, "MUST_NOT_GUIDE_TRAIN"):
            wrap_raw_distance(None, contract={}, descriptor={}, split="train", repo=Path("."))


if __name__ == "__main__":
    unittest.main()
