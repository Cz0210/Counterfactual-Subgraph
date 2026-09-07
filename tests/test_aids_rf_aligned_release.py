import json
from pathlib import Path
import tempfile
import unittest

from src.baselines.comrecgc.rf_aligned_pool import atomic_json
from src.baselines.comrecgc.rf_aligned_release import freeze_summary
from src.baselines.comrecgc.slot_evaluation import load_official_slots, build_internal_valid_candidates


class TestNativeSummaryRelease(unittest.TestCase):
    def test_publisher_uses_real_pointer_schema_and_original_cli(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from src.baselines.comrecgc.rf_aligned_release import publish_release
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'matrix_publication': {'worktree': str(root), 'execution_commit': 'code', 'state_path': str(root / 'state.json'), 'lock_path': str(root / 'publish.lock'), 'output_parent': str(root / 'versions')}}
            pointer = {'latest_authority_root': str(root / 'old'), 'latest_matrix_status_sha256': 'sha', 'latest_count': 13}
            with patch('src.eval.fast16_matrix_authority_pointer.read_authority_pointer', return_value=pointer) as read, patch('subprocess.check_output', return_value='code\n'), patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')) as run:
                result = publish_release(config, output_root=root)
            self.assertEqual(result['count_before'], 13)
            self.assertFalse(result['count_changed'])
            self.assertIsNone(read.call_args.kwargs['initial_authority_root'])
            self.assertIn('--supersede-existing', run.call_args.args[0])
            self.assertIn('sha', run.call_args.args[0])

    def test_predecessor_requires_real_native_terminal_and_owner_exit(self):
        from src.baselines.comrecgc.rf_aligned_release import predecessor_state
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner = root / 'owner.json'
            self.assertEqual(predecessor_state(root, owner), 'WAITING_NATIVE_OWNER')
            atomic_json(owner, {'returncode': 0})
            self.assertEqual(predecessor_state(root, owner), 'BLOCKED_NATIVE_SCIENCE_INCOMPLETE')
            atomic_json(root / 'terminal.json', {'state': 'RF_ALIGNED_NATIVE_SUMMARY_COMPLETE', 'old_cluster_labels_reused': False})
            self.assertEqual(predecessor_state(root, owner), 'READY')
            atomic_json(owner, {'returncode': 1})
            self.assertEqual(predecessor_state(root, owner), 'BLOCKED_NATIVE_OWNER_FAILED')

    def test_actual_action_chain_cannot_omit_or_invent_transition(self):
        import copy
        from src.baselines.comrecgc.rf_aligned_release import validate_lineage_records
        selected = [{'original_candidate_index': 3, 'original_parent_id': 'P', 'graph': {'labels': [1, 2]}}]
        event = {'event': 'selected_transition', 'action_resolution': 'exact', 'action': ['NA', 0, 2], 'parent_id': 'P', 'source_graph_sha256': 'parent', 'target_graph_sha256': 'child'}
        record = {'candidate_index': 3, 'parent_id': 'P', 'action_count': 1, 'stable_graph_sha256': 'child', 'actions': [event], 'replayed_compact_graph': selected[0]['graph']}
        validate_lineage_records({'records': [record]}, selected)
        for mutation in ('missing', 'inferred', 'wrong_parent', 'wrong_target'):
            bad = copy.deepcopy(record)
            if mutation == 'missing':
                bad['actions'] = []
            elif mutation == 'inferred':
                bad['actions'][0]['action_resolution'] = 'guessed_from_final_graph'
            elif mutation == 'wrong_parent':
                bad['actions'][0]['parent_id'] = 'Q'
            else:
                bad['actions'][0]['target_graph_sha256'] = 'other'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_lineage_records({'records': [bad]}, selected)

    def test_frozen_threshold_values_and_serialization_identity(self):
        import hashlib
        from src.baselines.comrecgc.rf_aligned_release import verify_reference_threshold_identity
        from src.baselines.comrecgc.rf_aligned_pool import file_sha
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'figure4.csv'
            path.write_text('threshold,coverage\n0.000,0.0\n0.0500,0.2\n')
            reference = {'figure4_path': str(path), 'figure4_sha256': file_sha(path), 'threshold_raw_string_sha256': hashlib.sha256(b'0.000\n0.0500\n').hexdigest()}
            evaluation = {'threshold_reference': reference, 'threshold_grid': [0., .05], 'threshold_config_hash': reference['threshold_raw_string_sha256']}
            self.assertEqual(verify_reference_threshold_identity(evaluation), reference)
            evaluation['threshold_grid'] = [0., .1]
            with self.assertRaisesRegex(ValueError, 'threshold'):
                verify_reference_threshold_identity(evaluation)

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
