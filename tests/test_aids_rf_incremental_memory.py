import unittest
from src.baselines.comrecgc.rf_aligned_recourse import incremental_private_anon_reserve


class TestPrivateAnonIncrementalBudget(unittest.TestCase):
    def test_existing_private_only_not_double_counted(self):
        remaining, current = incremental_private_anon_reserve(14000, anonymous_bytes=1400, shared_clean_bytes=200, shared_dirty_bytes=100)
        self.assertEqual((remaining, current), (12900, 1100))

    def test_shared_larger_than_anon_discounts_nothing(self):
        self.assertEqual(incremental_private_anon_reserve(14000, anonymous_bytes=100, shared_clean_bytes=200, shared_dirty_bytes=100), (14000, 0))

    def test_missing_or_negative_measurement_not_zero_assumed(self):
        for value in (None, -1, 'unknown'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                incremental_private_anon_reserve(14000, anonymous_bytes=value, shared_clean_bytes=0, shared_dirty_bytes=0)

    def test_remaining_growth_cannot_be_negative(self):
        self.assertEqual(incremental_private_anon_reserve(1000, anonymous_bytes=1400, shared_clean_bytes=0, shared_dirty_bytes=0), (0, 1400))
