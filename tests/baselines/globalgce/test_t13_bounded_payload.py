import pytest
import torch
from src.baselines.t13_bounded_payload import bounded_batches, incremental_admission, load_index, seal_payload


def test_missing_arrays_not_reseeded():
    with pytest.raises(ValueError,match='PAYLOAD_MISSING'):
        load_index(None,base_dataset=None,fsg=None,expected_identity={})


def test_actual_index_class_payload_roundtrip(tmp_path):
    from array import array
    from src.baselines.t13_indexed_augmentation import T13IndexedAugmentedDataset
    class Base:
        dataset_name='fixture';node_feat_dim=2;edge_attr_dim=0;max_num_nodes=2;num_classes=3
        labels=torch.tensor([1])
        def __getitem__(self,i):
            return dict(feature=torch.eye(2),adj=torch.eye(2),label=torch.tensor(1),num_nodes=torch.tensor(2),num_edges=torch.tensor(0))
    class Fsg:
        fs_max_nodes=2;fs_min_nodes=1
        def expand_graphs_size(self,f,a,e,n):return f,a,e
    base=Base();fsg=Fsg()
    ds=T13IndexedAugmentedDataset(dataset=base,fsg=fsg,graph_idxs=[0],positions=array('i',[0]*6),
        fs_indices=array('i',[0,1]*6),axes=array('i',[0,1,0,1]*6),
        split_fn=lambda n,labels:([0,1,2],[3,4],[5]),input_sha256='fixture',mask_sha256='fixture',
        rng_before_sha256='fixture',rng_after_sha256='fixture',boundary_checks=0)
    receipt=seal_payload(ds,tmp_path/'sealed')
    restored=load_index(receipt,base_dataset=base,fsg=fsg,expected_identity=ds.identity)
    assert restored.train_idx==ds.train_idx and restored.val_idx==ds.val_idx
    for i in range(6):
        for key,value in ds[i].items():
            actual=restored[i][key]
            assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual


def test_only_two_train_one_validation_not_all_dataset():
    class Data:
        identity={'sampler':{'num_workers':0}}
        train_idx=list(range(100000));val_idx=list(range(100000,150000))
        def __init__(self):self.seen=[]
        def __getitem__(self,i):self.seen.append(i);return {'index':i,'mask':torch.tensor([i,i])}
    d=Data();rng=torch.get_rng_state().clone();batches=bounded_batches(d)
    assert d.seen==list(range(1000))+list(range(100000,100500))
    assert all(b['mask'].shape==(500,2) for b in batches)
    assert torch.equal(torch.get_rng_state(),rng)


def test_incremental_not_double_current_rss():
    result=incremental_admission(headroom_bytes=100,canary_peak_increment_bytes=20,
        other_remaining_peaks={'aids':{'additional_bytes':14,'evidence':'sealed next stage'},
                               't12':{'additional_bytes':10,'evidence':'checkpoint bound'}},safety_margin_bytes=10)
    assert result['required_increment_bytes']==54 and result['admitted']
    with pytest.raises(ValueError,match='UNKNOWN'):
        incremental_admission(headroom_bytes=100,canary_peak_increment_bytes=20,
            other_remaining_peaks={'t12':{'additional_bytes':None,'evidence':'unknown'}},safety_margin_bytes=10)
