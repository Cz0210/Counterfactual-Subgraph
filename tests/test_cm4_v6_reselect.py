import os
import numpy as np
import pytest
from src.eval.cm4_v6_reselect import CM4
from src.baselines.cm_crem_runtime import atomic_json,digest,file_sha
from src.baselines.cm_crem_k20 import freeze_k20
from src.baselines.cm_crem_postfilter import commit_npz


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();pilot=source/'pilot.json'
    atomic_json(pilot,dict(oracle_sha256='a'*64,resolved_wnode=dict(molclr_checkpoint_sha256='b'*64)))
    spec=dict(execution_commit='c'*40,output_root=str(source),pilot_spec=str(pilot),pilot_spec_sha256=file_sha(pilot),
        evaluation=dict(grid=[0.,.03,.1,.2],cap=.03),parent_block_size=8)
    old=digest({k:v for k,v in spec.items() if k not in {'execution_commit','output_root'}})
    atomic_json(source/'spec.json',spec)
    atomic_json(source/'audit/final_audit.json',dict(status='CM_DATASET_POSTFILTER_AUDIT_PASS',fixture=False,scientific_pass_claimed=True,contract_sha256=old))
    parents=['a','b','c','d'];ids=['r'+str(i) for i in range(8)];mask=[True,True,True,False]
    d=np.array([[.12,.08,.02,.09,.03,.05,.04,.01],[.13,.08,.03,.02,.04,.01,.06,.07],[.14,.11,.04,.03,.02,.01,.05,.06],[np.inf]*8])
    states=np.ones(d.shape,dtype=np.uint8);states[-1]=2
    st=np.where(states==1,'OK','BEFORE_NOT_SOURCE')
    f,_=freeze_k20(d,st,parents,ids,mask,.03,.03,spec['evaluation']['grid'],old,'d'*64)
    atomic_json(source/'selection_freeze.json',f.to_dict());atomic_json(source/'pool_binding.json',dict(contract_sha256=old,candidate_ids=ids))
    for split in ('calibration','test'):
        atomic_json(source/(split+'_prepared.json'),dict(contract_sha256=old,parent_ids=parents,candidate_ids=ids,source_mask=mask))
        p=source/split/'block-0000.npz';commit_npz(p,values=d,states=states)
        atomic_json(p.with_suffix('.json'),dict(contract_sha256=old,parent_ids=parents,candidate_ids=ids,npz_sha256=file_sha(p)))
    return CM4('Mutagenicity',source,tmp_path/'out')


def test_complete_fixture_select_evaluate_reduction(tmp_path,monkeypatch):
    x=fixture(tmp_path);monkeypatch.setattr(x,'admission',lambda: x.root.mkdir(exist_ok=True))
    x.select();f=x.freeze();assert f.theta==.1 and len(f.pool_candidate_ids)==8
    assert x.c['oracle_sha256']=='a'*64 and x.c['temperature'] is None
    x.evaluate();result=x.result();assert result.prefix_metrics()[-1]['covered_count']==3
    assert result.prefix_metrics()[-1]['base_parent_count']==4
    # The independent audit entry must not self-certify in the selecting process.
    with pytest.raises(ValueError,match='Independent process'):x.audit()


def test_changed_complete_pool_rejected(tmp_path,monkeypatch):
    x=fixture(tmp_path);x.pool_ids=x.pool_ids[:-1];monkeypatch.setattr(x,'admission',lambda: x.root.mkdir(exist_ok=True))
    with pytest.raises(ValueError,match='Full original calibration'):x.select()
