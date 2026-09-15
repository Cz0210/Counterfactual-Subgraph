import unittest
from src.utils.t14_saved_follower_audit import analyze_after,first_difference

class SavedFollowerTests(unittest.TestCase):
    def row(self):
        return {'step':335,'compact_candidate_actions':[{'source_hash':'leader','target_hashes':['x'],'ordered_actions':[('NLC',28,15)]}],
                'native_observation':{'selected_transitions':[{'source_graph_hash':'follower','target_graph_hash':'y','action_records':[{'action':['NLC',28,12]}],'valid_fullgraph':True}], 'next_importance':[[.3,.2]]}}
    def test_selected_score_is_not_follower_argmin(self):
        x=analyze_after(self.row())['checks'][0]
        self.assertFalse(x['complete_follower_scores_captured'])
        self.assertFalse(x['captured_universe_for_this_source'])
        self.assertFalse(x['graph_one_edit_independently_proven'])
    def test_length_conflict_rejected(self):
        r=self.row();r['native_observation']['selected_transitions'][0]['source_graph_hash']='leader'
        r['compact_candidate_actions'][0]['ordered_actions']=[]
        with self.assertRaisesRegex(ValueError,'LENGTH'):analyze_after(r)
    def test_action_not_erased_by_tolerance(self):
        self.assertEqual(first_difference(['NLC',28,15],['NLC',28,12])['path'],'$[2]')
if __name__=='__main__':unittest.main()
