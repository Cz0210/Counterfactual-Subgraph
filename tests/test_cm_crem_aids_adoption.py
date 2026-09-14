import unittest
from src.baselines.cm_crem_aids_adoption import fixed32


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.parents = [{'parent_id': f'AIDS_CM_train_{i:03}', 'smiles': 'C'*((i%3)+1)} for i in range(40)]
        self.attrs = [{**p, 'before_label': 1} for p in self.parents[:16]]

    def test_keeps_old16_then_next16(self):
        self.assertEqual(fixed32(self.parents[::-1], self.attrs), self.parents[:32])

    def test_changed_old_source_rejected(self):
        self.attrs[0]['smiles'] = 'N'
        with self.assertRaises(ValueError): fixed32(self.parents, self.attrs)

    def test_outcome_resampling_rejected(self):
        self.attrs[0] = {**self.parents[17], 'before_label': 1}
        with self.assertRaises(ValueError): fixed32(self.parents, self.attrs)

    def test_not_sixteen_rejected(self):
        with self.assertRaises(ValueError): fixed32(self.parents, self.attrs[:12])
