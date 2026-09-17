import unittest
import math
from src.eval.paper_source_closeout_v8 import metric, canonical_sha, verify_selector_bindings

class MetricTests(unittest.TestCase):
    def test_raw_not_cap_controls_coverage(self):
        m=metric([.12,.02,math.inf],.1,.03)
        self.assertEqual(m['covered'],1)
        self.assertEqual(m['finite'],2)
        self.assertAlmostEqual(m['conditional_median'],.07)
        self.assertAlmostEqual(m['capped_mean'],.08/3)
    def test_unknown_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):metric([math.nan],.1,.03)
    def test_no_finite_is_not_zero(self):
        m=metric([math.inf],.1,.03)
        self.assertIsNone(m['conditional_median'])
        self.assertEqual(m['covered'],0)

    def test_original_freeze_binding_is_required(self):
        spec={'dataset':'BACE'}
        files={'p0/input_binding.json':dict(spec=spec,spec_sha=canonical_sha(spec),matrix_semantic_sha='raw')}
        for i in range(10):
            value=dict(matrix_semantic_sha='raw',ordered_candidate_ids=['a'],variant=f'S{i}')
            files[('p0/' if i<7 else 'p1/')+f'S{i}_freeze.json']=dict(value,freeze_sha256=canonical_sha(value))
        verify_selector_bindings({'BACE':files})
        files['p0/S3_freeze.json']['ordered_candidate_ids']=['b']
        with self.assertRaisesRegex(ValueError,'FREEZE_DIGEST'):
            verify_selector_bindings({'BACE':files})

if __name__=='__main__':unittest.main()
