import unittest
from unittest.mock import patch
from src.eval.selector_timing_v8 import configuration, timed_selection


class TimingTest(unittest.TestCase):
    def test_original_two_by_two(self):
        self.assertFalse(configuration(2)['multi'])
        self.assertTrue(configuration(3)['multi'])
        self.assertFalse(configuration(3)['prefix'])
        self.assertTrue(configuration(4)['prefix'])
        self.assertTrue(configuration(5)['multi'])

    def test_greedy_included_even_for_s0(self):
        class Objective:
            def greedy(self, ids, **kw):return [0]
        with patch('src.eval.selector_timing_v8.time.perf_counter',side_effect=[10,12,12]):
            seq,t=timed_selection(Objective(),['a'],0)
        self.assertEqual(seq,[0])
        self.assertEqual(t['selection_seconds'],2)
        self.assertEqual(t['greedy_seconds'],2)

    def test_refine_uses_original_contract(self):
        class Objective:
            def greedy(self, ids, **kw):
                assert kw==dict(multi=True)
                return [0]
            def refine(self,initial,**kw):
                assert kw==dict(multi=True,prefix=True,reg=(1,1,1))
                return initial,dict(proposals=1)
        with patch('src.eval.selector_timing_v8.time.perf_counter',side_effect=[1,4,9]):
            _,t=timed_selection(Objective(),['a'],6)
        self.assertEqual(t['selection_seconds'],8)
        self.assertEqual(t['refinement_seconds'],5)


if __name__=='__main__':unittest.main()
