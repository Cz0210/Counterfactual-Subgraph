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

def test_actual_portable_batch_roundtrip(tmp_path):
    import torch
    from src.data.molecular_graph_dataset import MolecularGraphBatch
    from src.baselines.cm_crem_oracle_diagnostic import tensor_fields
    b=MolecularGraphBatch(torch.ones((2,3),dtype=torch.long),torch.tensor([[0,1],[1,0]]),
       torch.ones((2,2),dtype=torch.long),torch.zeros(2,dtype=torch.long),torch.tensor([1]),
       ('a',),('CC',),('train',),('x',))
    torch.save(b.to('cpu'),tmp_path/'batch.pt')
    r=torch.load(tmp_path/'batch.pt',weights_only=False)
    assert all(torch.equal(v,getattr(r,k)) for k,v in tensor_fields(b).items())
    assert 'default_batch_size' in inspect.getsource(diagnostic)
