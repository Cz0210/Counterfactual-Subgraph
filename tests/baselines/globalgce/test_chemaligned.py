import copy
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('rdkit')
from src.baselines.bace_globalgce_chemaligned import joint_states, hard_state_tensors, materialize, ChemAlignedBridge
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule, build_parent_native_tensors
from src.data.molecular_graph_featurizer import default_molecular_feature_schema, MolecularGraphFeaturizer
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig

ATOMS = ('C', 'O', 'N'); BONDS = ('no_edge', 'single', 'double', 'triple')

def identity(smiles):
    p = build_parent_native_tensors(smiles, atom_symbols=ATOMS, bond_names=BONDS)
    r = GlobalGCENativeRule('fixture', 0, p.feature, p.adjacency, p.edge_attr,
                           p.feature, p.adjacency, p.edge_attr, ATOMS, BONDS)
    return p, r

def test_joint_formula_and_gradient():
    a = torch.tensor([[0., .8], [.8, 0.]], requires_grad=True)
    e = torch.tensor([[1., 2., -1., .5]], requires_grad=True)
    q = e.softmax(-1); p = joint_states(a, e)
    assert torch.allclose(p[:, 0], .2+.8*q[:, 0])
    assert torch.allclose(p[:, 1:], .8*q[:, 1:])
    assert torch.allclose(p.sum(-1), torch.ones(1))
    p[:, 1].sum().backward()
    assert a.grad.abs().sum() > 0 and e.grad.abs().sum() > 0

def test_none_conflict_not_forced_single():
    p, _ = identity('CC')
    probabilities = joint_states(p.adjacency*.9, torch.tensor([[10., 0., 0., 0.]]))
    _, a, e = hard_state_tensors(p.feature, probabilities)
    assert a.sum() == 0 and e.argmax(-1).tolist() == [0]

def test_padding_and_zero_diagonal():
    p, _ = identity('CC')
    f = p.feature.clone(); f[1] = torch.tensor([1., 0., 0., 0.])
    _, a, e = hard_state_tensors(f, p.edge_attr)
    assert a.sum() == 0 and e[0, 0] == 1

@pytest.mark.parametrize('smiles', ['CCO', 'C1CC1', '[NH3+]CC(=O)[O-]', 'N[C@@H](C)C(=O)O'])
def test_identity_complete_graph_and_attributes(smiles):
    p, r = identity(smiles)
    product = materialize(p, r, {i: i for i in range(len(p.feature))}, p.feature, p.edge_attr)
    assert product.canonical_smiles == p.canonical_smiles
    assert product.attributes_inherited == len(p.feature)
    assert product.attributes_reset == 0

def test_disconnected_local_rhs_legal_via_parent_attachments():
    parent, _ = identity('C1CC1'); lhs, rule = identity('CC')
    states = torch.tensor([[1., 0., 0., 0.]])
    product = materialize(parent, rule, {1: 1, 0: 0}, lhs.feature, states)
    assert product.canonical_smiles == 'CCC'
    assert product.boundary_count == 2

def test_lhs_tensor_index_not_dict_insertion_order():
    parent, _ = identity('CCO'); lhs, rule = identity('CO')
    product = materialize(parent, rule, {2: 1, 1: 0}, lhs.feature, lhs.edge_attr)
    assert product.canonical_smiles == 'CCO'

def test_shared_parent_not_mutated_and_rng_not_consumed():
    p, r = identity('CCO'); before = (p.feature.clone(), p.adjacency.clone(), p.edge_attr.clone())
    rng = torch.get_rng_state().clone()
    materialize(p, r, {i: i for i in range(len(p.feature))}, p.feature, p.edge_attr)
    assert torch.equal(rng, torch.get_rng_state())
    for a, b in zip(before, (p.feature, p.adjacency, p.edge_attr)): assert torch.equal(a, b)

def test_invalid_full_graph_rejects():
    p, r = identity('CC')
    with pytest.raises(ValueError, match='DISCONNECTED_COMPLETE_PRODUCT'):
        materialize(p, r, {0: 0, 1: 1}, p.feature, torch.tensor([[1., 0., 0., 0.]]))

def test_shared_hard_forward_oracle_and_gradient_only_generator():
    torch.manual_seed(7); schema = default_molecular_feature_schema()
    model = MolecularGNN(MolecularGNNConfig(backbone='gine', num_classes=2, num_layers=2,
        hidden_dim=16, dropout=0., pooling='mean', readout_layers=1, normalization='layer_norm', residual=True),
        node_cardinalities=schema.node_cardinalities, edge_cardinalities=schema.edge_cardinalities)
    bridge = ChemAlignedBridge(model, feature_schema=schema, atom_symbols=ATOMS, bond_names=BONDS,
                               checkpoint_id='fixture', temperature=1.5)
    p, r = identity('[NH3+]CC(=O)[O-]')
    feature = p.feature.clone().requires_grad_()
    states = p.edge_attr.clone().requires_grad_()
    product = materialize(p, r, {i: i for i in range(len(p.feature))}, feature, states)
    result = bridge.score_materialized(product)
    graph = MolecularGraphFeaturizer(schema).featurize(p.canonical_smiles)
    expected = model(x=torch.tensor(graph.node_features), edge_index=torch.tensor(graph.edge_index),
                     edge_attr=torch.tensor(graph.edge_features))
    assert torch.allclose(result['logits'], expected, atol=2e-6, rtol=0)
    before = copy.deepcopy(model.state_dict())
    loss = -result['y_pred'][0, 0]; loss.backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert states.grad is not None and states.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

def test_domain_and_axis_errors():
    with pytest.raises(ValueError, match='probability'):
        joint_states(torch.tensor([[0., 2.], [2., 0.]]), torch.zeros((1, 4)))
    with pytest.raises(ValueError, match='lower-triangle'):
        joint_states(torch.zeros((2, 2)), torch.zeros((2, 4)))
