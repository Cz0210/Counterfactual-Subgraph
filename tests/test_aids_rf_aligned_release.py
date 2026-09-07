import json
from pathlib import Path
import tempfile
import unittest

from src.baselines.comrecgc.rf_aligned_pool import atomic_json
from src.baselines.comrecgc.rf_aligned_release import freeze_summary
from src.baselines.comrecgc.slot_evaluation import load_official_slots, build_internal_valid_candidates


class TestNativeSummaryRelease(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        atomic_json(source / 'terminal.json', {'state': 'RF_ALIGNED_NATIVE_SUMMARY_COMPLETE', 'old_cluster_labels_reused': False, 'selected_count': 2, 'source1_count': 1097})
        rows = [dict(cluster_id=i, rf={'prediction': 0}, canonical_smiles='CC' if i == 0 else 'CCC', graph={'labels': [1]}, original_candidate_index=i, original_parent_id='P') for i in range(2)]
        atomic_json(source / 'selected_native_recourses.json', rows)
        config = dict(execution_commit='code', official_upstream_commit='official', dataset_fingerprint='dataset', generation_parent_ids_sha256='parents', rf_sha256='rf', evaluation_contract={'theta_star': .05})
        return source, rows, config

    def test_all_native_clusters_preserved_and_no_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, rows, config = self.fixture(root)
            output = root / 'freeze'
            result = freeze_summary(config, recourse_root=source, output_root=output)
            slots = load_official_slots(output / 'medoid_validity.csv')
            candidates = build_internal_valid_candidates(slots, selection_method='original_native_summary_order', adaptation_mode='ComRecGC-RFAligned')
            self.assertEqual(len(candidates), 2)
            self.assertEqual([s['official_cluster_rank'] for s in slots], [1, 2])
            self.assertEqual(result['parent_count'], 1283)
            self.assertEqual(result['source_eligible_count'], 1097)
            self.assertFalse(result['additional_chemical_repair'])
            self.assertFalse(result['calibration_split_invented'])
            self.assertEqual(freeze_summary(config, recourse_root=source, output_root=output), result)

    def test_no_summary_no_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, rows, config = self.fixture(root)
            atomic_json(source / 'terminal.json', {'state': 'COUNT_COMPLETE'})
            with self.assertRaisesRegex(ValueError, 'Completed'):
                freeze_summary(config, recourse_root=source, output_root=root / 'freeze')

    def test_duplicates_or_rf_non_targets_cannot_publish(self):
        for mode in ('duplicate', 'rf_source'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, rows, config = self.fixture(root)
                if mode == 'duplicate':
                    rows[1]['cluster_id'] = rows[0]['cluster_id']
                else:
                    rows[0]['rf']['prediction'] = 1
                atomic_json(source / 'selected_native_recourses.json', rows)
                with self.assertRaises(ValueError):
                    freeze_summary(config, recourse_root=source, output_root=root / 'freeze')


if __name__ == '__main__':
    unittest.main()
