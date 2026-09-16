import numpy as np
from src.eval.ours_taste_theta010_search import choose_strata,search

def test_strata_counts_and_no_test():
    cohort=[{'parent_id':f'train{i}','scaffold':str(i%13)} for i in range(256)]
    best=np.r_[np.repeat(.12,33),np.repeat(np.inf,59),np.repeat(.01,164)]
    chosen,count=choose_strata(cohort,best)
    assert count=={'far':33,'none':59,'covered':36}
    assert len(chosen)==len(set(chosen))==128
    assert (chosen,count)==choose_strata(cohort,best)

def test_search_original_parent_budgets_and_theta():
    class Scorer:
        checkpoint_id='fixture'
        def score_smiles(self,items):
            return [{'logits':[2.,0.,1.],'predicted_label':0,'probabilities':[.66,.1,.24]} for _ in items]
    class Provider:
        def distance_for_action(self,parent,residual,**kw):
            assert parent=='CCCCCC'
            return {'ok':True,'distance':.08}
    rules,events,ledger=search({'parent_id':'trainfixture','smiles':'CCCCCC','pred_before':1},[],Scorer(),Provider(),{})
    assert events and rules and ledger['new_residual_oracle']<=64 and ledger['search_wnode']<=32
    assert all(len(r['deletion_atoms']) in (1,2,3,4,6) and r['original_parent_used'] for r in events)
    assert all(r['covered_theta010'] for r in events if r['raw_wnode'] is not None)
    assert ledger['new_lm_outputs']==0
