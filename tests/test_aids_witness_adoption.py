import json
import numpy as np
import pytest
import sklearn
from src.baselines.comrecgc.aids_global_witness import scan
from src.baselines.comrecgc.aids_witness_adoption import verify_edges


def example(tmp_path):
    x=np.asarray([[0.],[.001],[.002],[.003]],dtype=np.float32)
    binding={'seed_failure_ledger_complete':True,'anchor_ids':[0,1,2,3], 'failure_ids':[3],
             'seed_ids':[0,1,2],'eps':.02,'min_samples':3,'sklearn_version':sklearn.__version__}
    scan(x,anchors=binding['anchor_ids'],failures=[3],seeds=[0,1,2],eps=.02,min_samples=3,
         expected_version=sklearn.__version__,output=tmp_path,binding=binding,block_rows=2)
    state=json.loads((tmp_path/'witness_checkpoint.json').read_text())
    return x,binding,state


def test_independent_kernel_replay(tmp_path):
    x,b,s=example(tmp_path)
    proof=verify_edges(x,b,s)
    assert proof['status']=='PASS' and proof['single_epsilon_component_proven']
    assert proof['actual_witness_rows_replayed']==4


def test_changed_distance_rejected(tmp_path):
    x,b,s=example(tmp_path)
    s['witnesses']['3'][0]['distance']+=.001
    with pytest.raises(ValueError,match='distance differs'):verify_edges(x,b,s)


def test_55_core_not_enough_without_nonfailure_closure(tmp_path):
    x,b,s=example(tmp_path);b['seed_failure_ledger_complete']=False
    with pytest.raises(ValueError,match='closure'):verify_edges(x,b,s)


def test_forged_connectivity_and_duplicate_neighbors_rejected(tmp_path):
    x,b,s=example(tmp_path)
    s['outside_failure_neighbor']['3']['row_id']=3
    with pytest.raises(ValueError,match='False attachment'):verify_edges(x,b,s)
    x,b,s=example(tmp_path/'again')
    s['witnesses']['3'][1]=s['witnesses']['3'][0]
    with pytest.raises(ValueError,match='distinct'):verify_edges(x,b,s)


def test_false_anchor_edge_rejected(tmp_path):
    x,b,s=example(tmp_path)
    s['failure_edges']['3'].remove(0)
    with pytest.raises(ValueError,match='anchor graph'):verify_edges(x,b,s)
