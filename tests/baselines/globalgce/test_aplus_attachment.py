import copy
import pytest
import torch
from rdkit import Chem

from src.baselines.bace_globalgce_aplus import (
    build_parent, materialize, apply_states, joint_states, GINAlignedBridge,
)
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule
from src.data.molecular_graph_featurizer import default_molecular_feature_schema, MolecularGraphFeaturizer
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig

ATOMS = ('C','O','N','F','S','Cl')
BONDS = ('no_edge','single','double','triple')

def fixture(smiles):
    p=build_parent(smiles,atom_symbols=ATOMS,bond_names=BONDS)
    r=GlobalGCENativeRule('fixture',0,p.feature,p.adjacency,p.edge_attr,
        p.feature,p.adjacency,p.edge_attr,ATOMS,BONDS)
    return p,r

@pytest.mark.parametrize('smiles',['CCO','C1CC1','N[C@@H](C)C(=O)O','F/C=C/F','F/C=C\\F','[NH3+]CC(=O)[O-]'])
def test_identity_preserves_full_attributed_graph(smiles):
    p,r=fixture(smiles)
    result=materialize(p,r,{i:i for i in range(len(p.feature))},p.feature,p.edge_attr)
    assert result.canonical_smiles == p.canonical_smiles
    assert Chem.MolToSmiles(p.source_molecule,isomericSmiles=True)==p.canonical_smiles

def test_multiboundary_disconnected_rhs_is_accepted_only_in_connected_product():
    p,_=fixture('C1CC1'); lhs,r=fixture('CC')
    no_bond=lhs.edge_attr.clone(); no_bond[0]=torch.tensor([1.,0.,0.,0.])
    product=materialize(p,r,{1:1,0:0},lhs.feature,no_bond)
    assert product.canonical_smiles == 'CCC'
    assert product.boundary_count == 2 and product.anchors == (0,1)

def test_anchor_padding_cannot_erase_external_bond_endpoint():
    p,_=fixture('CCO'); lhs,r=fixture('CO')
    f=lhs.feature.clone(); f[0]=0; f[0,0]=1
    product=materialize(p,r,{2:1,1:0},f,lhs.edge_attr)
    assert product.canonical_smiles == 'CCO'
    assert product.anchors == (1,)

def test_no_ad_hoc_connection_for_disconnected_generated_node():
    p,r=fixture('CC')
    no_bond=p.edge_attr.clone(); no_bond[0]=torch.tensor([1.,0.,0.,0.])
    with pytest.raises(ValueError,match='DISCONNECTED_COMPLETE_PRODUCT'):
        materialize(p,r,{0:0,1:1},p.feature,no_bond)

def test_boundary_mapping_uses_lhs_tensor_slot_not_mapping_order():
    p,_=fixture('CCO'); lhs,r=fixture('CO')
    product=materialize(p,r,{2:1,1:0},lhs.feature,lhs.edge_attr)
    assert product.canonical_smiles == 'CCO'
    assert product.mask_order == (1,2)

def test_parent_rng_and_input_immutable():
    p,r=fixture('CCO'); before=copy.deepcopy(p); rng=torch.get_rng_state().clone()
    materialize(p,r,{i:i for i in range(3)},p.feature,p.edge_attr)
    assert torch.equal(rng,torch.get_rng_state())
    for name in ('feature','adjacency','edge_attr'): assert torch.equal(getattr(p,name),getattr(before,name))
    assert Chem.MolToSmiles(p.source_molecule)==Chem.MolToSmiles(before.source_molecule)

def test_noedge_contradiction_not_forced_single():
    a=torch.tensor([[0.,.9],[.9,0.]])
    p=joint_states(a,torch.tensor([[10.,0.,0.,0.]]))
    assert p.argmax(-1).tolist()==[0]

@pytest.mark.parametrize('smiles',['CCO','C1CC1','F/C=C/F','N[C@@H](C)C(=O)O'])
def test_exact_gin_hard_forward_and_only_generator_gradients(smiles):
    torch.manual_seed(7); schema=default_molecular_feature_schema()
    model=MolecularGNN(MolecularGNNConfig(backbone='gin',num_classes=2,num_layers=2,
        hidden_dim=16,dropout=0.,pooling='mean',readout_layers=1,normalization='layer_norm',residual=True),
        node_cardinalities=schema.node_cardinalities,edge_cardinalities=schema.edge_cardinalities)
    bridge=GINAlignedBridge(model,feature_schema=schema,atom_symbols=ATOMS,bond_names=BONDS,
        checkpoint_id='fixture',temperature=1.141163362671653)
    p,r=fixture(smiles); features=p.feature.clone().requires_grad_(); states=p.edge_attr.clone().requires_grad_()
    product=materialize(p,r,{i:i for i in range(len(features))},features,states)
    actual=bridge.score_materialized(product)
    graph=MolecularGraphFeaturizer(schema).featurize(smiles)
    expected=model(x=torch.tensor(graph.node_features),edge_index=torch.tensor(graph.edge_index),
        edge_attr=torch.tensor(graph.edge_features))
    assert torch.allclose(actual['logits'],expected,atol=2e-6,rtol=0)
    (-actual['y_pred'][0,0]).backward()
    assert features.grad is not None and features.grad.abs().sum()>0
    assert states.grad is not None and states.grad.abs().sum()>0
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())

