"""Read committed T13 compact arrays, never rebuild the augmented dataset.

The historical formal saved identities but not these arrays. A missing payload
is an explicit blocker: this module never substitutes reseeded masks.
"""
from array import array
from collections import OrderedDict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256


def seal_payload(indexed, root):
    """Future completed index only; no parent tensors or RNG calls."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    data=dict(identity=indexed.identity,buffers=[a.tobytes() for a in indexed._buffers],
        splits={name:list(getattr(indexed,name)) for name in ('train_idx','val_idx','test_idx')},
        byteorder=sys.byteorder)
    path=root/'compact_index.pkl.gz'
    with gzip.open(path,'wb') as stream:pickle.dump(data,stream,protocol=5)
    receipt=dict(schema='t13_committed_compact_arrays_v1',path=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        identity_sha256=indexed.identity['identity_sha256'],parent_tensors_saved=False,
        compact_bytes=sum(len(x) for x in data['buffers']),rng_used=False)
    atomic_json(root/'payload_manifest.json',receipt)
    return receipt


def load_index(descriptor, *, base_dataset, fsg, expected_identity):
    import torch
    from src.baselines.t13_indexed_augmentation import T13IndexedAugmentedDataset
    if not descriptor or not Path(descriptor['path']).is_file():
        raise ValueError('COMPACT_MASK_INDEX_PAYLOAD_MISSING_NOT_RESEEDABLE')
    path=Path(descriptor['path'])
    raw=path.read_bytes()
    if len(raw)!=descriptor['bytes'] or hashlib.sha256(raw).hexdigest()!=descriptor['sha256']:
        raise ValueError('COMPACT_PAYLOAD_BINDING')
    if descriptor['compact_bytes']>512*1024**2:
        raise ValueError('COMPACT_ARRAY_BUDGET_EXCEEDED')
    with gzip.open(path,'rb') as stream:data=pickle.load(stream)
    if data['identity']!=expected_identity or data['byteorder']!=sys.byteorder:
        raise ValueError('COMPACT_IDENTITY_OR_BYTEORDER_CHANGED')
    identity=data['identity'];buffers=[];digest=hashlib.sha256()
    for raw in data['buffers']:
        a=array('i');a.frombytes(raw)
        if a.itemsize!=4:raise ValueError('INT32_REQUIRED')
        buffers.append(a);digest.update(memoryview(a))
    if digest.hexdigest()!=identity['index_sha256'] or sum(len(x)*x.itemsize for x in buffers)!=descriptor['compact_bytes']:
        raise ValueError('COMPACT_INDEX_BYTES_CHANGED')
    splits=data['splits']
    if stable_sha256(dict(train=splits['train_idx'],validation=splits['val_idx'],test=splits['test_idx']))!=identity['split_sha256']:
        raise ValueError('COMPACT_SPLIT_ORDER_CHANGED')
    out=object.__new__(T13IndexedAugmentedDataset)
    out.dataset=base_dataset;out.fsg=fsg;out.graph_idxs=tuple(identity['graph_idxs']);out._buffers=tuple(buffers)
    out.graph_idx_list=torch.frombuffer(buffers[0],dtype=torch.int32)
    out.fs_idx_list=torch.frombuffer(buffers[1],dtype=torch.int32).reshape(-1,2)
    out.mask_axes=torch.frombuffer(buffers[2],dtype=torch.int32).reshape(-1,2,identity['fs_max_nodes'])
    if len(out.graph_idx_list)!=identity['sample_count'] or len(out.mask_axes)!=identity['sample_count']:
        raise ValueError('COMPACT_MULTIPLICITY')
    for name in ('dataset_name','node_feat_dim','edge_attr_dim','max_num_nodes','num_classes'):
        setattr(out,name,getattr(base_dataset,name))
    out.identity=identity;out.index=range(identity['sample_count']);out._cache=OrderedDict();out.cache_max_parents=8
    for name,values in splits.items():setattr(out,name,values)
    return out


def bounded_batches(indexed, *, batch_size=500):
    """Use original split order and batch width; only 2 train + 1 validation."""
    from torch.utils.data import default_collate
    if batch_size!=500 or indexed.identity['sampler']['num_workers']!=0:
        raise ValueError('ORIGINAL_BATCH_OR_WORKER_CONTRACT')
    if len(indexed.train_idx)<2*batch_size or len(indexed.val_idx)<batch_size:
        raise ValueError('REAL_FULL_BATCHES_UNAVAILABLE')
    selected=[indexed.train_idx[:batch_size],indexed.train_idx[batch_size:2*batch_size],indexed.val_idx[:batch_size]]
    return [default_collate([indexed[int(i)] for i in ids]) for ids in selected]


def incremental_admission(*, headroom_bytes, canary_peak_increment_bytes, other_remaining_peaks, safety_margin_bytes):
    """Current RSS is already in cgroup usage; only future growth is charged."""
    if any(type(x) is not int or x<0 for x in (headroom_bytes,canary_peak_increment_bytes,safety_margin_bytes)):
        raise ValueError('MEASURED_INCREMENT_REQUIRED')
    total=canary_peak_increment_bytes+safety_margin_bytes
    for task,row in other_remaining_peaks.items():
        if type(row.get('additional_bytes')) is not int or row['additional_bytes']<0 or not row.get('evidence'):
            raise ValueError('UNKNOWN_NEXT_BOUNDARY_PEAK:'+task)
        total+=row['additional_bytes']
    return dict(admitted=headroom_bytes>=total,required_increment_bytes=total,
        actual_headroom_bytes=headroom_bytes,current_rss_double_counted=False,
        scope='NEXT_BOUNDARY_INCREMENT_ONLY',other_remaining_peaks=other_remaining_peaks)
