import csv
import pytest
from src.baselines.cm_crem_dataset_pilot import validate,read_train
from src.baselines.cm_crem_runtime import file_sha

def spec(dataset='tastemolnet'):
    return dict(dataset=dataset,num_classes=3 if dataset=='tastemolnet' else 2,
        oracle_backend='gine' if dataset=='tastemolnet' else 'rf',source_label=1,
        allowed_destinations=[0,2] if dataset=='tastemolnet' else [0],seed=7,
        mask_fraction=.2,radius=1,min_max_inc=3,max_replacements_per_component=64,
        raw_max_per_parent=128,parent_wall_seconds=900,parent_limit=32,full_library_cap=6000,
        deadline_utc='2026-09-16T16:27:44Z')

@pytest.mark.parametrize('dataset',['tastemolnet','mutagenicity','aids'])
def test_original_families_supported(dataset):validate(spec(dataset))

@pytest.mark.parametrize('key,value',[('allowed_destinations',[0]),('num_classes',2),
    ('oracle_backend','rf'),('parent_limit',16),('raw_max_per_parent',129),('mask_fraction',.1)])
def test_multiclass_and_fixed_budget_not_silently_changed(key,value):
    s=spec();s[key]=value
    with pytest.raises(ValueError):validate(s)

def test_train_labels_and_order_bound_without_test(tmp_path):
    path=tmp_path/'train.csv'
    path.write_text('molecule_id,model_smiles,label\na,CCO,1\nb,CCC,0\nc,CCN,2\n')
    s=spec();s['train']={'path':str(path),'sha256':file_sha(path),'id_field':'molecule_id','smiles_field':'model_smiles','label_field':'label'}
    rows,n=read_train(s)
    assert n==3 and [r['parent_id'] for r in rows]==['a'] and rows[0]['split']=='train'
    path.write_text(path.read_text()+'d,C,1\n')
    with pytest.raises(ValueError,match='content'):read_train(s)

def test_actual_three_class_gine_gradcam_keeps_all_class_axes():
    import torch
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
    from src.models.molecular_gnn import MolecularGNN,MolecularGNNConfig
    from src.oracles.gnn_oracle import GNNOracle
    from src.baselines.cm_crem_oracle import FrozenCMOracle
    torch.manual_seed(7)
    f=MolecularGraphFeaturizer();s=f.schema
    model=MolecularGNN(MolecularGNNConfig(backbone='gine',num_classes=3,num_layers=3,
        hidden_dim=8,dropout=.3,normalization='batch_norm',residual=True),
        node_cardinalities=s.node_cardinalities,edge_cardinalities=s.edge_cardinalities)
    with torch.no_grad():model.classifier[-1].bias[:]=torch.tensor([-4.,4.,-4.])
    oracle=GNNOracle(model,checkpoint_id='a'*64,backbone='gine',num_classes=3,
        source_label=1,temperature=1.9724769811393754,edge_feature_dim=len(s.edge_fields),device='cpu')
    a=FrozenCMOracle(oracle,f,binding={'dataset':'tastemolnet','allowed_destinations':[0,2]})
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=a.attribute_train_parent({'parent_id':'train-fixture','smiles':'CCOC(=O)N','split':'train'})
    assert len(result['prediction']['probabilities'])==3
    assert result['gradient_nonzero_count']>0 and a.allowed_destinations==[0,2]
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_rf_serial_execution_changes_only_reduction_schedule():
    from types import SimpleNamespace
    from src.baselines.cm_crem_dataset_pilot import serial_rf_execution
    trees=[object(),object()]
    model=SimpleNamespace(n_jobs=7,estimators_=trees,n_features_in_=2048)
    oracle=SimpleNamespace(model=model)
    assert serial_rf_execution(oracle) is oracle
    assert model.n_jobs==1 and model.estimators_ is trees and model.n_features_in_==2048
    assert oracle.cm_execution_receipt['original_n_jobs']==7
    assert not oracle.cm_execution_receipt['tolerance_changed']
