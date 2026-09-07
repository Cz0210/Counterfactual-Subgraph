import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval import fast16_matrix_authority_pointer as pointer


class TestCorrectiveCAS(unittest.TestCase):
    def run_fixture(self, temporary):
        root = Path(temporary)
        before_root, after_root = root / 'before', root / 'after'
        before_root.mkdir()
        after_root.mkdir()
        rows = {(d, m): {'dataset': d, 'method': m, 'status': 'FROZEN_PASS' if i < 13 else 'MISSING', 'version': 'old'} for i, (d, m) in enumerate((d, m) for d in pointer.DATASETS for m in pointer.METHODS)}
        before = {'root': before_root, 'complete': 13, 'matrix_sha256': 'a' * 64, 'combined_sha256': 'b' * 64, 'rows': rows}
        after = copy.deepcopy(before)
        after.update(root=after_root, matrix_sha256='c' * 64, combined_sha256='d' * 64)
        target = next(k for k in rows if k == ('AIDS', 'ComRecGC'))
        after['rows'][target]['version'] = 'corrected'
        state, lock = root / 'control/state.json', root / 'control/publish.lock'
        state.parent.mkdir()
        state.write_text(json.dumps(pointer._state_for(before)))
        values = {str(before_root): before, str(after_root): after}
        def verify(path, expected_complete=None):
            value = values[str(path)]
            if expected_complete is not None and expected_complete != value['complete']:
                raise ValueError('wrong complete count')
            return copy.deepcopy(value)
        kwargs = dict(state_path=state, lock_path=lock, requested_cell='AIDS/ComRecGC', expected_prior_authority_root=before_root, expected_prior_matrix_sha256='a' * 64, supersede=lambda _old: {'output_root': str(after_root)})
        return before, after, state, verify, kwargs

    def test_same_13_count_and_preserved_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after, state, verify, kwargs = self.run_fixture(temporary)
            old = copy.deepcopy(before)
            with patch.object(pointer, '_verify_authority', side_effect=verify):
                result = pointer.supersede_under_authority_pointer(**kwargs)
            self.assertEqual(before, old)
            self.assertEqual(result['authority_pointer_after']['latest_count'], 13)
            self.assertFalse(result['matrix_count_changed'])
            self.assertEqual(json.loads(state.read_text())['latest_authority_root'], str(after['root']))

    def test_stale_cas_does_not_invoke_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after, state, verify, kwargs = self.run_fixture(temporary)
            original_state = state.read_bytes()
            kwargs['expected_prior_matrix_sha256'] = 'stale'
            called = []
            kwargs['supersede'] = lambda old: called.append(old)
            with patch.object(pointer, '_verify_authority', side_effect=verify), self.assertRaisesRegex(pointer.MatrixAuthorityPointerError, 'STALE'):
                pointer.supersede_under_authority_pointer(**kwargs)
            self.assertFalse(called)
            self.assertEqual(state.read_bytes(), original_state)

    def test_nontarget_change_rejected_before_pointer_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after, state, verify, kwargs = self.run_fixture(temporary)
            after['rows'][('AIDS', 'Ours')]['version'] = 'accidental'
            original_state = state.read_bytes()
            with patch.object(pointer, '_verify_authority', side_effect=verify), self.assertRaisesRegex(pointer.MatrixAuthorityPointerError, 'non-target'):
                pointer.supersede_under_authority_pointer(**kwargs)
            self.assertEqual(state.read_bytes(), original_state)


if __name__ == '__main__':
    unittest.main()
