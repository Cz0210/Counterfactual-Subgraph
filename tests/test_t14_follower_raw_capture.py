import pickle
import gzip
import random
from types import SimpleNamespace
import numpy as np
import pytest
from src.baselines.t14_causal_diagnostic import SamplingObserver, follower_field_preflight, follower_snapshot


def test_full_fields_survive_more_than_32_values(tmp_path):
    receipt=follower_field_preflight(tmp_path/'raw')
    assert receipt['new_transitions']==0 and receipt['captured_candidates']==40
    with gzip.open(receipt['raw_file'],'rb') as f:row=pickle.load(f)
    assert row['arrays']['target_graphs_embedding']['values'].shape==(40,64)
    assert 'content_sha256' not in row['arrays']['difference']


def test_observer_calls_same_argmin_and_does_not_consume_rng():
    def move_from_known_graph():pass
    def move_to_next_graph():
        k=1;i=11;select=0;target_graphs_hashes=[3,4]
        difference=np.array([.1,.1+1e-10],dtype=np.float64)
        recourse=np.zeros(2);matching_recourses=np.zeros((2,2));target_graphs_embedding=np.zeros((2,2))
        start_embedding=np.zeros(2);selected_elements=np.ones(2);start_elements=1
        target_graphs_importance_parts=np.ones((2,3))
        return np.argmin(difference)
    module=SimpleNamespace(move_from_known_graph=move_from_known_graph,move_to_next_graph=move_to_next_graph,np=np)
    state=random.getstate();native=np.argmin
    with SamplingObserver(module) as observer:
        observer.capture_follower=True
        assert move_to_next_graph()==0
    assert random.getstate()==state and np.argmin is native
    row=observer.events[0]
    assert row['exact_minimum_indices']==[0]
    assert row['arrays']['difference']['values'][1]>.1
    assert row['extra_rng_calls']==0


def test_missing_raw_is_not_a_pass():
    with pytest.raises(ValueError,match='RAW_LOCALS_MISSING'):
        follower_snapshot({'difference':np.zeros(2)},np.zeros(2),0)
