import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import bace_gin_native_raw as raw
from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256


class NativeRawTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'sealed' / 'pair_matrix.jsonl'
        self.source.parent.mkdir()
        self.out = self.root / 'fresh' / 'index.json'
        self.row = dict(parent_id='p1', parent_smiles='CCO', candidate_id='c1', canonical_smiles='CC',
            action_kind=raw.OPERATIONS['gcfexplainer'][0], action_semantics=raw.OPERATIONS['gcfexplainer'][1],
            applicable=True, method_id='gcfexplainer', oracle_checkpoint_hash='old-gine',
            wnode_distance=.0125, distance_for_selection=99999, failure_reason=None,
            # These deliberately disagree with the finite cost: adoption may
            # not use them as the new GIN's predictions or strict-flip mask.
            pred_before=0, pred_after=1, cf_flip=False, pair_strict_flip=False)
        self.binding = dict(schema=raw.BINDING_SCHEMA, method_id='gcfexplainer', splits={},
            raw_contract={'wnode': {'solver': 'exact_emd2'}})
        self.doc = dict(action_kind=self.row['action_kind'], action_semantics=self.row['action_semantics'],
                        oracle_checkpoint_hash='old-gine')
        self.patchers = [patch.object(raw, '_source_docs', return_value=(self.doc, {'p1': 'CCO'},
                           {'c1': 'CC'}, {'kernel': 'sha'}, {'namespace_role': 'CACHE_ONLY'})),
                         patch.object(raw, '_canonical', side_effect=lambda x: x),
                         patch.object(raw, '_reject_writer')]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        self.bind([self.row])

    def bind(self, rows, finite=1):
        data = ''.join(json.dumps(x) + '\n' for x in rows).encode()
        self.source.write_bytes(data)
        item = dict(pair_file=dict(path=str(self.source), sha256=hashlib.sha256(data).hexdigest(), size=len(data)),
                    merge_manifest={'path': 'sealed/manifest', 'sha256': 'manifest'},
                    candidate_file={'path': str(self.source)},
                    pair_count=len(rows), finite_count=finite)
        self.binding['splits']['calibration'] = item
        self.binding['splits']['test'] = item

    def run_index(self, **kw):
        return raw.build_native_index(self.binding, split='calibration', output=self.out, repo=self.root, **kw)

    def test_exact_raw_only_not_flip_prediction_or_selection_cap(self):
        index = self.run_index()
        value = next(iter(index['graph_costs'].values()))
        self.assertEqual(value['distance'], .0125)
        self.assertEqual(value['source_records'], [{'source_member': 'original_pair_matrix', 'line': 1}])
        self.assertFalse(index['source_flip_masks_reused'])
        self.assertFalse(index['source_selected_match_minima_reused'])
        self.assertFalse(index['source_selection_caps_reused'])
        self.assertEqual(index['ot_recomputed'], 0)
        self.assertEqual(index['source_file_reads'], 1)

    def test_old_no_distance_is_gap_not_inf_or_zero(self):
        self.bind([{**self.row, 'wnode_distance': None, 'distance_for_selection': '+inf',
                    'failure_reason': 'frozen_gine_not_strict_flip'}], finite=0)
        value = self.run_index()
        self.assertEqual(value['graph_costs'], {})
        self.assertEqual(value['source_missing_raw_distance_rows'], 1)

    def test_bad_finite_never_accepted(self):
        for value in (float('nan'), float('inf'), -1, True, '0.0125'):
            with self.subTest(value=value):
                self.bind([{**self.row, 'wnode_distance': value}])
                with self.assertRaisesRegex(ValueError, 'NONFINITE_OR_FAILED'):
                    self.run_index()

    def test_failed_distance_never_accepted(self):
        self.bind([{**self.row, 'failure_reason': 'OT_FAILED'}])
        with self.assertRaisesRegex(ValueError, 'NONFINITE_OR_FAILED'):
            self.run_index()

    def test_changed_graph_or_native_mapping_rejected(self):
        for change in ({'canonical_smiles': 'CO'}, {'action_kind': 'delete_fragment'}, {'match_index': 0}):
            with self.subTest(change=change):
                self.bind([{**self.row, **change}])
                with self.assertRaisesRegex(ValueError, 'GRAPH_OR_OPERATION|NOT_UNIQUE_FULLGRAPH'):
                    self.run_index()

    def test_duplicate_pair_rejected(self):
        self.bind([self.row, self.row], finite=2)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_PARENT_CANDIDATE'):
            self.run_index()

    def test_stream_hash_binding_rejected(self):
        self.binding['splits']['calibration']['pair_file']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SOURCE_CHANGED_DURING'):
            self.run_index()

    def test_active_writer_rejected(self):
        raw._reject_writer.side_effect = ValueError('NATIVE_RAW_SOURCE_HAS_ACTIVE_WRITER:123')
        with self.assertRaisesRegex(ValueError, 'ACTIVE_WRITER'):
            self.run_index()

    def test_source_stat_change_rejected(self):
        old = raw._stat(self.source)
        with patch.object(raw, '_stat', side_effect=[old, {**old, 'st_mtime_ns': old['st_mtime_ns'] + 1}]):
            with self.assertRaisesRegex(ValueError, 'SOURCE_CHANGED_DURING'):
                self.run_index()

    def test_sealed_index_does_not_reread_source(self):
        first = self.run_index()
        self.source.unlink()
        raw._source_docs.reset_mock()
        self.assertEqual(self.run_index(), first)
        raw._source_docs.assert_not_called()

    def test_test_before_fresh_freeze_never_opens_source(self):
        with patch.object(raw, '_bound', side_effect=AssertionError('source should not open')):
            with self.assertRaisesRegex(ValueError, 'TEST_BEFORE_NEW_SELECTOR_FREEZE'):
                raw.build_native_index({}, split='test', output=self.out, repo=self.root,
                                       validate_test_freeze=True)

    def test_actual_freeze_validator_runs_before_test_source(self):
        freeze = self.root / 'new-freeze.json'
        atomic_json(freeze, {'own_scheme_a_freeze': True})
        digest = hashlib.sha256(freeze.read_bytes()).hexdigest()
        events = []
        raw._source_docs.side_effect = lambda *a: (events.append('source') or
            (self.doc, {'p1': 'CCO'}, {'c1': 'CC'}, {'kernel': 'sha'}, {}))
        result = raw.build_native_index(self.binding, split='test', output=self.out, repo=self.root,
            test_freeze_path=freeze, test_freeze_sha=digest,
            validate_test_freeze=lambda x: events.append('new-freeze-validated'))
        self.assertEqual(events, ['new-freeze-validated', 'source'])
        self.assertEqual(result['new_test_freeze_sha256'], digest)

    def test_old_or_invalid_freeze_refuses_before_source(self):
        freeze = self.root / 'old-freeze.json'
        atomic_json(freeze, {'old_gine_freeze': True})
        def reject(_):
            raise ValueError('SCHEME_A_ACTUAL_NEW_FREEZE_REQUIRED')
        with self.assertRaisesRegex(ValueError, 'ACTUAL_NEW_FREEZE'):
            raw.build_native_index(self.binding, split='test', output=self.out, repo=self.root,
                test_freeze_path=freeze, test_freeze_sha=hashlib.sha256(freeze.read_bytes()).hexdigest(),
                validate_test_freeze=reject)
        raw._source_docs.assert_not_called()

    def test_original_function_and_namespace_code_proof(self):
        repo = Path(__file__).resolve().parents[1]
        proof = raw._native_call_proof(repo, 'b8f978cbf90d124e4f317a8b925282b784ff36a7')
        self.assertFalse(proof['source_execution_commit_claimed'])
        self.assertEqual(proof['source_distance_namespace'], raw.NATIVE_NAMESPACE)

    def test_stream_sha_exact_existing_contract_and_atomic_roundtrip(self):
        for value in ({'a': [0., -0., 1e-250, '\u6e29\u5ea6'], 'z': None}, [True, False, {}, 'x'],
                      {str(i): {'distance': i / 9., 'source_records': [{'line': i}]} for i in range(3000)}):
            self.assertEqual(raw.stream_sha256(value), stable_sha256(value))
            raw._stream_atomic_json(self.out, value)
            self.assertEqual(json.loads(self.out.read_text()), value)

    def test_actual_full_gcf_layout_bound_below_four_gib(self):
        bound = raw.index_memory_bound(1361396, 1449228, 32220295)
        self.assertLess(bound['estimated_peak_rss_bound_bytes'], 4 * 1024**3)
        self.assertEqual(bound['serialized_full_copy_count'], 0)


if __name__ == '__main__':
    unittest.main()
