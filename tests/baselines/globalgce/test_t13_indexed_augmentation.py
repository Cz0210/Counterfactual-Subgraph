"""Exact small official eager/lazy parity; no real production training."""
from __future__ import annotations

import ast
import copy
from itertools import product,combinations
import os
from pathlib import Path
import random
from types import ModuleType
import sys

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader,Subset,default_collate

from src.baselines.t13_indexed_augmentation import (build_indexed_dataset,_eager_without_split,
    OFFICIAL_FSG_SHA,restore_mask)
from src.baselines.t13_indexed_canary import state_digest,rng_state,restore_rng,run_training_parity


def _defs(path,names,namespace):
    tree=ast.parse(path.read_text())
    body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    assert {node.name for node in body}==set(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)


@pytest.fixture
def official(monkeypatch):
    root=Path(os.environ.get('GLOBALGCE_OFFICIAL_ROOT','baselines/globalgce_official'))/'src'
    if not (root/'models/fsg.py').is_file():pytest.skip('pinned official checkout required via GLOBALGCE_OFFICIAL_ROOT')
    import hashlib
    assert hashlib.sha256((root/'models/fsg.py').read_bytes()).hexdigest()==OFFICIAL_FSG_SHA
    namespace=dict(torch=torch,F=F,nx=nx,iso=nx.algorithms.isomorphism,random=random,
        product=product,combinations=combinations,tqdm=lambda v:v,StratifiedShuffleSplit=StratifiedShuffleSplit)
    _defs(root/'utils.py',{'get_edge_index','get_nx_graph'},namespace)
    _defs(root/'data/dataset.py',{'get_train_val_test_idx','AugmentedDataset'},namespace)
    _defs(root/'models/fsg.py',{'FrequentSubgraphGenerator'},namespace)
    data=ModuleType('data.dataset');data.AugmentedDataset=namespace['AugmentedDataset']
    data.get_train_val_test_idx=namespace['get_train_val_test_idx']
    monkeypatch.setitem(sys.modules,'data.dataset',data)
    return namespace


class Parents:
    dataset_name='TasteMolNet';node_feat_dim=3;edge_attr_dim=4;max_num_nodes=5;num_classes=3
    def __init__(self):
        self.feat=F.one_hot(torch.tensor([[1,1,1,1,0]]*6),3).float()
        self.adj=torch.zeros(6,5,5)
        self.edge_attr=F.one_hot(torch.zeros(6,10,dtype=torch.long),4).float()
        for i in range(6):
            for a,b in ((0,1),(1,2),(2,3)):
                self.adj[i,a,b]=self.adj[i,b,a]=1
                self.edge_attr[i,(b-1)*b//2+a]=torch.tensor([0,1,0,0])
        self.labels=torch.zeros(6,dtype=torch.long)
        self.num_nodes=torch.full((6,),4,dtype=torch.long);self.num_edges=torch.full((6,),6,dtype=torch.long)
        self.train_idx=[3,1,5,0];self.val_idx=[4,2];self.test_idx=[]
    def __getitem__(self,i):
        return dict(feature=self.feat[i].clone(),adj=self.adj[i].clone(),edge_attr=self.edge_attr[i].clone(),
            label=self.labels[i].clone(),num_nodes=self.num_nodes[i].clone(),num_edges=self.num_edges[i].clone(),index=torch.tensor(i))


def setup(official):
    fsg=official['FrequentSubgraphGenerator'](2,3,'unused',2,False)
    fsg.fs_max_nodes=3;fsg.fs_min_nodes=2
    graphs={}
    for k,size in enumerate((2,3)):
        graph=nx.path_graph(size);nx.set_node_attributes(graph,1,'label');nx.set_edge_attributes(graph,1,'label');graphs[k]=graph
    return fsg,Parents(),graphs


def lazy(official,fsg,dataset,graphs):
    return build_indexed_dataset(fsg,dataset,graphs,get_nx_graph=official['get_nx_graph'],
        split_fn=official['get_train_val_test_idx'],eager_dataset_class=_eager_without_split(official['AugmentedDataset']))


def test_complete_official_eager_lazy_samples_split_and_rng_exact(official):
    fsg,parents,graphs=setup(official)
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    before=rng_state()
    eager=fsg.expand_data_by_fs(parents,graphs)
    after=state_digest(rng_state())
    restore_rng(before)
    indexed=lazy(official,fsg,parents,graphs)
    assert state_digest(rng_state())==after
    assert len(indexed)==len(eager)>20
    assert (indexed.train_idx,indexed.val_idx,indexed.test_idx)==(eager.train_idx,eager.val_idx,eager.test_idx)
    for i in range(len(eager)):
        assert state_digest(indexed[i])==state_digest(eager[i]),i
    assert indexed.identity['production_boundary_checks']==6
    assert not hasattr(indexed,'feat') and not hasattr(indexed,'adj')
    assert indexed.identity['compact_index_bytes'] < eager.fs_mask_list.numel()*eager.fs_mask_list.element_size()


def test_repeated_access_never_mutates_parent_or_cached_template(official):
    fsg,parents,graphs=setup(official);random.seed(7)
    indexed=lazy(official,fsg,parents,graphs)
    before=state_digest(parents.__dict__)
    expected=state_digest(indexed[0]);row=indexed[0]
    row['feature'].fill_(99);row['adj'].fill_(99);row['edge_attr'].fill_(99)
    assert state_digest(indexed[0])==expected and state_digest(parents.__dict__)==before


def test_lazy_forbids_full_augmented_repeat_interleave(official,monkeypatch):
    fsg,parents,graphs=setup(official)
    def forbidden(*_args,**_kwargs):raise AssertionError('global repeat_interleave is forbidden')
    monkeypatch.setattr(torch,'repeat_interleave',forbidden)
    indexed=lazy(official,fsg,parents,graphs)
    assert len(indexed)>0


def test_cartesian_axis_restores_order_and_sentinel_exactly():
    axes=torch.tensor([[3,1,8],[-1,-1,-1]],dtype=torch.int32)
    result=restore_mask(axes)
    assert torch.equal(result[0],torch.tensor(list(product([3,1,8],repeat=2))))
    assert torch.equal(result[1],-torch.ones(9,2,dtype=torch.long))


def test_deterministic_complete_index_digest_and_batch_order(official):
    fsg,parents,graphs=setup(official);random.seed(7)
    first=lazy(official,fsg,parents,graphs)
    random.seed(7);second=lazy(official,fsg,parents,graphs)
    assert first.identity==second.identity
    one=default_collate([first[i] for i in first.train_idx[:12]])
    two=next(iter(DataLoader(Subset(second,second.train_idx),batch_size=12)))
    assert state_digest(one)==state_digest(two)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.1));self.gt_gnn=torch.nn.Linear(1,1)
        self.device=torch.device('cpu')
    def get_rules(self,_fss):return {'noise':torch.rand(())}
    def run_one_batch(self,rules,data):
        value=(self.weight+rules['noise']+data['feature'].mean()).square()
        zero=value*0
        return value,zero,zero,value


def test_short_real_torch_forward_optimizer_scheduler_rng_and_reload_parity(official,tmp_path):
    fsg,parents,graphs=setup(official);random.seed(7)
    indexed=lazy(official,fsg,parents,graphs)
    report=run_training_parity(model=TinyModel(),fss={},
        train_loader=DataLoader(Subset(indexed,indexed.train_idx),batch_size=500,num_workers=0),
        learning_rate=0.1,output_root=tmp_path/'canary',resume_identity={'seed':7})
    assert report['checkpoint_reload_exact'] and report['model_optimizer_scheduler_rng_exact']
    assert report['optimizer_updates_per_arm']==2 and report['full_science_started'] is False
