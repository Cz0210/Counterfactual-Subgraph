import numpy as np
import pytest
from src.baselines.cm_crem_rf_attribution import environment_features,attribute
from src.rewards.reward_calculator import smiles_to_morgan_array

def test_original_fp_and_collision_mapping():
    smi='CC1CCCCC1';x=smiles_to_morgan_array(smi,radius=2,n_bits=32)
    m,y,bits,env=environment_features(smi,radius=2,bits=32,original_features=x)
    assert np.array_equal(x,y) and len(bits)==m.GetNumAtoms()
    assert any(len(rows)>1 for rows in env.values())
    assert set(sum(bits,[]))==set(np.flatnonzero(x))
    with pytest.raises(ValueError):environment_features(smi,radius=2,bits=32,original_features=x+1)

def test_signed_scores_repeat_and_input_not_mutated():
    class Model:
        n_features_in_=32
        def predict_proba(self,x):
            p=x.sum(axis=1)/33
            return np.column_stack([1-p,p])
    class Oracle:
        model=Model();radius=2;n_bits=32;source_label=1;class_labels=(0,1);checkpoint_id='a'*64
        def predict_proba(self,rows):
            return self.model.predict_proba(np.array([smiles_to_morgan_array(s,radius=2,n_bits=32) for s in rows]))
    p={'smiles':'CCCOCCC','parent_id':'train_01'};a=attribute(Oracle(),p);b=attribute(Oracle(),p)
    assert a==b and a['not_gradcam'] and a['generation_request']['selected_atom_indices']
    assert a['fingerprint']['aux_features'] is False

@pytest.mark.parametrize('smi', ['CC=NNC(=O)c1ccccc1', 'C/C=N/NC(=O)c1ccccc1',
    'N[C@@H](C)C(=O)O', 'CC=NNC(=O)c1ccccc1.CS(=O)(=O)O'])
def test_explicit_stereo_transport_retains_unknown_and_known(smi):
    from rdkit import Chem
    from src.baselines.cm_crem_generation import make_parent_request,load_parent_mol,atom_order_sha256
    m=Chem.MolFromSmiles(smi)
    r=make_parent_request('train_fixture',m,[0],explicit_stereo_transport=True)
    restored=load_parent_mol(r)
    assert r['schema']=='cm_crem_parent_v3'
    assert atom_order_sha256(restored)==atom_order_sha256(m)
    assert Chem.MolToSmiles(restored)==Chem.MolToSmiles(m)

def test_v2_scratch_scope_is_explicit_and_permissions_not_relaxed(tmp_path,monkeypatch):
    from src.baselines import cm_crem_assets as assets
    monkeypatch.setattr(assets,'HPC_SCOPE',tmp_path)
    root=tmp_path/'counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2/rf16'
    root.mkdir(parents=True,mode=0o700)
    assert assets._cm_run_root(root)==root
    root.chmod(0o775)
    with pytest.raises(Exception,match='OWNERSHIP_UNSAFE'):assets._cm_run_root(root)
