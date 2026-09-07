import copy
import unittest
from src.experiments.bace_gin_globalgce import (
    adoption_manifest, ORIGINAL_MODEL_SHA, ORIGINAL_RULES_SHA,
)


def evidence():
    model = {'path': '/original/model.pt', 'sha256': ORIGINAL_MODEL_SHA, 'size': 198469}
    rules = {'path': '/original/rules.pt', 'sha256': ORIGINAL_RULES_SHA, 'size': 220183}
    summary = {'test_loaded': False, 'rules_checkpoint': rules['path'],
        'globalgce_model_checkpoint': model['path'],
        'gspan_exact_top_k_proof': {'selected_identity_sha256': 'mining'},
        'codec_metadata': {'node_label_mapping': {'0': 'padding', '1': 'C'},
                           'edge_label_mapping': {'0': 'no_edge', '1': 'single'}}}
    recovery = {'test_loaded': False, 'source_model_checkpoint': model,
                'source_rules_checkpoint': rules, 'candidate_universe': {'path': '/original/pool.jsonl'}}
    binding = {'model_retrained': False, 'mining_reused': True, 'source_model': model['path'],
        'source_rules': rules['path'], 'rule_count': 80,
        'train_ids': ['t'+str(i) for i in range(360)],
        'validation_ids': ['v'+str(i) for i in range(98)]}
    terminal = {'state': 'REMATERIALIZATION_COMPLETE', 'execution_valid': True,
                'test_loaded': False, 'weights_changed': False}
    for split, n, matched, maps in [('train', 360, 23397, 120289), ('validation', 98, 6493, 33841)]:
        terminal[split+'_index'] = {'parents': n, 'pairs': n*80, 'matched_pairs': matched, 'matches': maps}
        terminal[split] = {'rules': 80, 'counts': {'mappings': maps, 'DISCONNECTED_COMPLETE_PRODUCT': maps},
                           'test_loaded': False, 'calibration_loaded': False}
    catalog = [{'candidate_id': 'native'+str(i), 'rule': {'native_rule_index': i},
                'rule_content_hash': 'nativehash'+str(i), 'source_split': 'train'} for i in range(80)]
    return summary, recovery, binding, terminal, catalog


class ManifestTests(unittest.TestCase):
    def call(self, args=None):
        return adoption_manifest(*(args or evidence()), references={'completed_terminal': 'real path'})

    def test_all_invalid_is_blocked_not_zero(self):
        value = self.call()
        self.assertEqual(value['state'], 'BLOCKED_MATERIALIZATION')
        self.assertEqual(value['rule_count'], 80)
        self.assertIsNone(value['scientific_metrics'])
        self.assertFalse(value['zero_coverage_claimed'])
        self.assertEqual(value['train_validation_funnels']['train']['mapping_count'], 120289)
        self.assertFalse(value['train_validation_funnels']['train']['old_oracle_flip_counts_adopted'])

    def test_repair_weight_rejected(self):
        args = evidence(); args[1]['source_model_checkpoint']['sha256'] = 'repair'
        with self.assertRaisesRegex(ValueError, 'repair checkpoint'): self.call(args)

    def test_incomplete_attempt_cannot_be_zero(self):
        args = evidence(); args[3]['state'] = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.call(args)

    def test_mapping_coverage_missing_rejected(self):
        args = evidence(); args[3]['train_index']['matches'] += 1
        with self.assertRaisesRegex(ValueError, 'mapping coverage'): self.call(args)

    def test_rejection_count_not_padded(self):
        args = evidence(); args[3]['train']['counts']['DISCONNECTED_COMPLETE_PRODUCT'] -= 1
        with self.assertRaisesRegex(ValueError, 'funnel'): self.call(args)

    def test_duplicate_native_index_rejected(self):
        args = evidence(); args[4][-1]['rule']['native_rule_index'] = 0
        with self.assertRaisesRegex(ValueError, 'inventory'): self.call(args)

    def test_test_or_calibration_not_source(self):
        args = evidence(); args[4][0]['source_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'train proposal'): self.call(args)
        args = evidence(); args[3]['train']['calibration_loaded'] = True
        with self.assertRaisesRegex(ValueError, 'used calibration'): self.call(args)

    def test_valid_chemistry_is_not_gin_evaluation_pass(self):
        args = evidence(); counts = args[3]['train']['counts']
        counts['DISCONNECTED_COMPLETE_PRODUCT'] -= 1; counts['valid'] = 1
        value = self.call(args)
        self.assertEqual(value['state'], 'READY_FOR_FROZEN_GIN_CALIBRATION')
        self.assertIsNone(value['scientific_metrics'])
        self.assertFalse(value['new_oracle_inference_performed'])

    def test_no_input_mutation(self):
        args = evidence(); before = copy.deepcopy(args)
        self.call(args); self.assertEqual(args, before)


try:
    import torch
    from rdkit import Chem
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'tiny chemistry fixtures require existing torch/rdkit runtime')
class HardMaterializerTests(unittest.TestCase):
    def identity(self, smiles):
        from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule, build_parent_native_tensors
        atoms, bonds = ('C', 'O', 'N'), ('no_edge', 'single', 'double', 'triple')
        parent = build_parent_native_tensors(smiles, atom_symbols=atoms, bond_names=bonds)
        return parent, GlobalGCENativeRule('fixture', 0, parent.feature, parent.adjacency,
            parent.edge_attr, parent.feature, parent.adjacency, parent.edge_attr, atoms, bonds)

    def test_joint_none_not_forced_bond(self):
        from src.experiments.bace_gin_globalgce import joint_states, hard_state_tensors
        p, _ = self.identity('CC')
        q = joint_states(p.adjacency*.9, torch.tensor([[10., 0., 0., 0.]]))
        _, a, e = hard_state_tensors(p.feature, q)
        self.assertEqual(a.sum().item(), 0); self.assertEqual(e.argmax(-1).tolist(), [0])

    def test_disconnected_rhs_valid_via_attachments(self):
        from src.experiments.bace_gin_globalgce import materialize
        parent, _ = self.identity('C1CC1'); lhs, rule = self.identity('CC')
        result = materialize(parent, rule, {1: 1, 0: 0}, lhs.feature, torch.tensor([[1., 0., 0., 0.]]))
        self.assertEqual(result['canonical_smiles'], 'CCC')
        self.assertEqual(result['boundary_attachment_count'], 2)

    def test_complete_disconnect_rejected(self):
        from src.experiments.bace_gin_globalgce import materialize
        parent, rule = self.identity('CC')
        with self.assertRaisesRegex(ValueError, 'DISCONNECTED_COMPLETE_PRODUCT'):
            materialize(parent, rule, {0: 0, 1: 1}, parent.feature, torch.tensor([[1., 0., 0., 0.]]))

    def test_mapping_is_tensor_order_not_mapping_insertion(self):
        from src.experiments.bace_gin_globalgce import materialize
        parent, _ = self.identity('CCO'); lhs, rule = self.identity('CO')
        result = materialize(parent, rule, {2: 1, 1: 0}, lhs.feature, lhs.edge_attr)
        self.assertEqual(result['canonical_smiles'], 'CCO')

    def test_atom_attributes_rng_and_shared_parent_preserved(self):
        from src.experiments.bace_gin_globalgce import materialize
        parent, rule = self.identity('[NH3+]CC(=O)[O-]')
        old = (parent.feature.clone(), parent.adjacency.clone(), parent.edge_attr.clone())
        rng = torch.get_rng_state().clone()
        result = materialize(parent, rule, {i: i for i in range(len(parent.feature))}, parent.feature, parent.edge_attr)
        self.assertEqual(result['canonical_smiles'], parent.canonical_smiles)
        self.assertEqual(result['source_attributes_reset'], 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, (parent.feature, parent.adjacency, parent.edge_attr))))

    def test_bad_logits_domain_rejected(self):
        from src.experiments.bace_gin_globalgce import joint_states
        with self.assertRaisesRegex(ValueError, 'probability'):
            joint_states(torch.tensor([[0., 2.], [2., 0.]]), torch.zeros((1, 4)))


if __name__ == '__main__':
    unittest.main()
