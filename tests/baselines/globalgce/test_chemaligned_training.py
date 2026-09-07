import copy
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('rdkit')
from src.baselines.bace_globalgce_chemaligned_training import (
    TRAINING_CONTRACT, reconstruction, train_update, snapshot_rng, restore_rng, atomic_torch, assert_semantic_equal,
)
from src.baselines.bace_globalgce_chemaligned import ChemAlignedBridge
from src.baselines.globalgce_bace_native_rules import build_parent_native_tensors, GlobalGCENativeRule
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig

def fixture():
    atoms=('C','O'); bonds=('no_edge','single','double','triple')
    p=build_parent_native_tensors('CCO',atom_symbols=atoms,bond_names=bonds)
    r=GlobalGCENativeRule('fixture',0,p.feature,p.adjacency,p.edge_attr,p.feature,p.adjacency,p.edge_attr,atoms,bonds)
    schema=default_molecular_feature_schema()
    frozen=MolecularGNN(MolecularGNNConfig(backbone='gine',num_classes=2,num_layers=1,hidden_dim=8,dropout=0.,
        pooling='mean',readout_layers=1,normalization='layer_norm',residual=True),
        node_cardinalities=schema.node_cardinalities,edge_cardinalities=schema.edge_cardinalities)
    bridge=ChemAlignedBridge(frozen,feature_schema=schema,atom_symbols=atoms,bond_names=bonds,checkpoint_id='fixture',temperature=1.2)
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.node=torch.nn.Parameter((p.feature.unsqueeze(0)*.96+.01).clone())
            self.edge=torch.nn.Parameter((p.edge_attr.unsqueeze(0)*15).clone())
            self.adj=torch.nn.Parameter((p.adjacency.unsqueeze(0)*.98).clone())
            self.mu=torch.nn.Parameter(torch.zeros((1,2)))
        def get_rules(self,fss):
            return dict(fss,features_reconst=self.node,adj_reconst=self.adj,
                        edge_attrs_reconst=self.edge,z_mu=self.mu,z_logvar=self.mu*0+.1)
    model=Generator()
    fss={'feat':p.feature.unsqueeze(0),'adj':p.adjacency.unsqueeze(0),'edge_attr':p.edge_attr.unsqueeze(0)}
    index=[({'id':'train-fixture','smiles':'CCO'},p,r,{0:0,1:1,2:2},{'predicted_label':1})]
    return model,fss,bridge,index

def test_actual_optimizer_update_frozen_oracle_and_atomic_reload(tmp_path):
    model,fss,bridge,index=fixture()
    old=copy.deepcopy(model.state_dict()); oracle=copy.deepcopy(bridge.model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.0001)
    result=train_update(model,opt,fss,bridge,index,torch.tensor([0]))
    assert result['counts']['valid_complete_product']==1
    assert any(not torch.equal(old[k],v) for k,v in model.state_dict().items())
    assert all(torch.equal(oracle[k],v) for k,v in bridge.model.state_dict().items())
    assert all(p.grad is None for p in bridge.model.parameters())
    atomic_torch(tmp_path/'checkpoint.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':snapshot_rng()})
    loaded=torch.load(tmp_path/'checkpoint.pt',weights_only=False)
    assert all(torch.equal(loaded['model'][k],v) for k,v in model.state_dict().items())
    assert not (tmp_path/'checkpoint.pt.partial').exists()

def test_invalid_products_never_call_frozen_oracle(monkeypatch):
    model,fss,bridge,index=fixture()
    with torch.no_grad(): model.edge[:,:,0]=100
    monkeypatch.setattr(bridge,'score_materialized',lambda _: (_ for _ in ()).throw(AssertionError('invalid oracle call')))
    opt=torch.optim.Adam(model.parameters(),lr=.0001)
    result=train_update(model,opt,fss,bridge,index,torch.tensor([0]))
    assert result['counts']['DISCONNECTED_COMPLETE_PRODUCT']==1

def test_rng_roundtrip_independent_tensor_state():
    state=snapshot_rng(); first=torch.rand(10); restore_rng(state); second=torch.rand(10)
    assert torch.equal(first,second)

def test_shared_joint_objective_finite_gradient():
    model,fss,_,_=fixture(); loss,states,components=reconstruction(model.get_rules(fss)); loss.backward()
    assert torch.isfinite(loss) and torch.allclose(states.sum(-1),torch.ones_like(states[...,0]))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert set(components)=={'node','edge','adjacency','kl'}

def test_predeclared_budget_not_extended_by_validation():
    assert TRAINING_CONTRACT['seed']==7 and TRAINING_CONTRACT['epochs']==100
    assert TRAINING_CONTRACT['optimizer_updates']==100
    assert TRAINING_CONTRACT['logical_batch_size']*TRAINING_CONTRACT['max_logical_batches']==2500
    assert TRAINING_CONTRACT['test_used'] is False

def test_semantic_reload_covers_optimizer_rng_and_counter(tmp_path):
    model,fss,bridge,index=fixture(); optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    train_update(model,optimizer,fss,bridge,index,torch.tensor([0]))
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':snapshot_rng(),'step':1}
    atomic_torch(tmp_path/'state.pt',payload)
    restored=torch.load(tmp_path/'state.pt',weights_only=False)
    assert_semantic_equal(payload,restored)
    restored['optimizer']['state'][0]['exp_avg'].flatten()[0]+=1
    with pytest.raises(ValueError,match='optimizer'): assert_semantic_equal(payload,restored)

def test_deployed_joint_rule_uses_full_parent_not_standalone_rhs():
    from src.baselines.bace_globalgce_chemaligned import apply_chemaligned_rule_to_parent
    from dataclasses import replace
    model,fss,bridge,index=fixture(); rule=index[0][2]
    # Identity deployment is the same complete graph used in the train bridge.
    rows=apply_chemaligned_rule_to_parent('CCO',rule)
    assert rows and all(row['valid'] and row['canonical_smiles']=='CCO' for row in rows)

def test_joint_catalog_deduplicates_identity_without_padding_or_lost_lineage():
    from src.baselines.bace_globalgce_chemaligned_export import joint_catalog
    from dataclasses import replace
    model,fss,bridge,index=fixture(); rule=index[0][2]
    rules=model.get_rules(fss)
    doubled={k:torch.cat((v,v),0) for k,v in rules.items()}
    rows,rejected=joint_catalog([rule,replace(rule,rule_id='other',native_rule_index=1)],doubled)
    assert not rejected and len(rows)==1 and rows[0]['generator_native_indices']==[0,1]
    assert rows[0]['standalone_RHS_chemical_validity_claimed'] is False

def test_joint_candidate_adapter_cannot_masquerade_as_legacy():
    from src.eval.bace_native_baseline_gnn import _molecular_adapter_metadata
    from src.baselines.bace_globalgce_chemaligned import SCHEMA
    assert _molecular_adapter_metadata({})=={}
    with pytest.raises(ValueError): _molecular_adapter_metadata({'molecular_adapter':SCHEMA,'method_id':'globalgce'})
    assert _molecular_adapter_metadata({'molecular_adapter':SCHEMA,'method_id':'globalgce','method_variant':'GlobalGCE-ChemAligned'})['rule_budget_semantics']=='AT_MOST_K'

def test_identity_canary_is_not_claimed_as_generated_recourse_and_restores_rng():
    from src.baselines.bace_globalgce_chemaligned_training import real_oracle_identity_canary
    model,fss,bridge,index=fixture(); rng=snapshot_rng(); weights=copy.deepcopy(model.state_dict())
    result=real_oracle_identity_canary(model,fss,bridge,index[0])
    assert result['state']=='PASS' and result['target_flip_claimed'] is False
    assert result['generator_gradient_l1']>0 and result['fixture_kind'].startswith('synthetic_train_identity')
    assert_semantic_equal(rng,snapshot_rng()); assert_semantic_equal(weights,model.state_dict())
