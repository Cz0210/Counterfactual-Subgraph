import numpy as np
import pytest
from src.baselines.cm_crem_k20 import freeze_k20
from src.baselines.cm_crem_selection import evaluate_frozen_test


def evaluate(values, states, mask):
    a=np.asarray(values,dtype=float).reshape(-1,1);s=np.asarray(states).reshape(-1,1)
    p=[str(i) for i in range(len(a))];m=np.asarray(mask,dtype=bool)
    f,_=freeze_k20(a,s,p,['own-cm-pool'],m,0.1,0.03416003659645076,[0.008,0.034],'a'*64,'b'*64)
    return evaluate_frozen_test(f,a,pair_status=s,parent_ids=p,candidate_ids=['own-cm-pool'],source_mask=m,contract_sha256='a'*64)


def test_uncapped_boundary_and_non_source():
    r=evaluate([.08,.12,.1,np.inf,np.inf],['OK','OK','OK','NO_STRICT_FLIP','BEFORE_NOT_SOURCE'],[1,1,1,1,0])
    k=r.prefix_metrics()[-1]
    assert k['covered_count']==2 and k['finite_recourse_count']==3
    assert k['conditional_median_cost']==.1
    assert k['effective_k']==1
    assert np.all(r.best_distances[0]==r.best_distances[-1])


@pytest.mark.parametrize('value,state',[(np.nan,'UNCOMPUTED'),(np.inf,'ERROR'),(np.inf,'UNKNOWN')])
def test_unknown_not_semantic_failure(value,state):
    with pytest.raises(ValueError):evaluate([value],[state],[True])


def test_next_float_above_boundary_not_rounded():
    r=evaluate([np.nextafter(.1,np.inf)],['OK'],[True])
    assert r.prefix_metrics()[-1]['covered_count']==0
