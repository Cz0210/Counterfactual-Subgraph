import inspect
from src.baselines.cm_crem_oracle_diagnostic import difference,input_only,diagnostic
def test_exact_contract():
    assert difference([1.],[1.])['exact']
    assert not difference([1.],[1.+1e-12])['exact']
    assert not difference([float('nan')],[1.])['exact']
def test_input_strip():
    assert input_only({'smiles':'CC','logits':[1,2]})=={'smiles':'CC'}
def test_capture_scope():
    s=inspect.getsource(diagnostic)
    assert 'fixed_spotcheck_pairs' in s
    assert "'ot_calls':0" in s
    assert "'scientific_pass_claimed':False" in s
    assert "'historical_saved_tensors':False" in s
    assert 'filter_generated(parent,generation)' in s
