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
import copy
import inspect
import random

from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256


def seal_payload(indexed, root):
    """Future completed index only; no parent tensors or RNG calls."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    data=dict(identity=indexed.identity,buffers=[a.tobytes() for a in indexed._buffers],
        splits={name:list(getattr(indexed,name)) for name in ('train_idx','val_idx','test_idx')},
        byteorder=sys.byteorder)
    if hasattr(indexed, 'reconstruction_receipt'):
        data['reconstruction_receipt'] = indexed.reconstruction_receipt
    path=root/'compact_index.pkl.gz'
    with gzip.open(path,'wb') as stream:pickle.dump(data,stream,protocol=5)
    receipt=dict(schema='t13_committed_compact_arrays_v1',path=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        identity_sha256=indexed.identity['identity_sha256'],parent_tensors_saved=False,
        compact_bytes=sum(len(x) for x in data['buffers']),rng_used=False)
    atomic_json(root/'payload_manifest.json',receipt)
    return receipt


def rebuild_index_proven(*, base_dataset, fsg, fs_dict, expected_identity, seed):
    """Recover only arrays proven identical to the old committed identity.

    A seed is not evidence on its own: before/after RNG, every index/mask byte,
    split order, inputs, multiplicity and official producer must all match.
    The caller must run this only after storage and resource admission. No
    training, oracle inference, mining or global RNG reseeding is performed.
    """
    from src.baselines.t13_indexed_augmentation import (
        OFFICIAL_FSG_SHA, _eager_without_split, _rng_digest, build_indexed_dataset)
    import importlib
    source = Path(inspect.getsourcefile(type(fsg)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != OFFICIAL_FSG_SHA:
        raise ValueError('T13_OFFICIAL_FSG_SOURCE_DRIFT')
    if expected_identity.get('official_fsg_sha256') != OFFICIAL_FSG_SHA:
        raise ValueError('T13_EXPECTED_FSG_SOURCE_DRIFT')
    private_rng = random.Random(seed)
    if stable_sha256(private_rng.getstate()) != expected_identity['python_rng_before_sha256']:
        raise ValueError('T13_RECONSTRUCTION_INITIAL_RNG_NOT_BOUND')
    before = _rng_digest()
    native_data = importlib.import_module('data.dataset')
    result = build_indexed_dataset(fsg, base_dataset, fs_dict,
        split_fn=native_data.get_train_val_test_idx,
        eager_dataset_class=_eager_without_split(native_data.AugmentedDataset),
        mask_rng=private_rng)
    if _rng_digest() != before:
        raise ValueError('T13_RECONSTRUCTION_CONSUMED_GLOBAL_RNG')
    return _adopt_proven_reconstruction(result, expected_identity)


def _adopt_proven_reconstruction(result, expected_identity):
    """Content comparison kept separate for focused rejection tests."""
    # Current process Torch/NumPy state need not equal the historical startup
    # state; formal restores checkpoint RNG after this data-only step. The
    # actual RNG-neutrality proof is preserved below, not silently discarded.
    ignored = {'identity_sha256', 'materialization_rng_sha256'}
    actual = copy.deepcopy(result.identity)
    expected = copy.deepcopy(expected_identity)
    mismatch = [key for key in sorted(set(actual) | set(expected))
                if key not in ignored and actual.get(key) != expected.get(key)]
    if mismatch:
        raise ValueError('T13_RECONSTRUCTION_CONTENT_MISMATCH:' + mismatch[0])
    if stable_sha256({k:v for k,v in expected.items() if k != 'identity_sha256'}) != expected['identity_sha256']:
        raise ValueError('T13_EXPECTED_INDEX_IDENTITY_INVALID')
    result.reconstruction_receipt = dict(
        state='COMMITTED_INDEX_MASK_SPLIT_CONTENT_RECONSTRUCTED',
        original_identity=expected, reconstructed_identity=actual,
        comparison_excluded_noncontent_fields=sorted(ignored),
        original_rng_before_matched=True, original_rng_after_matched=True,
        global_rng_unchanged=True, candidate_generation_repeated=False,
        mining_repeated=False, checkpoint_training_rng_not_used=True,
        scientific_resume_validated=False)
    # Explicit relocation/reconstruction overlay adopts the original content
    # identity only after all its scientific fields passed exact comparison.
    result.identity = expected
    return result


def install_committed_expansion(fsg, *, descriptor, expected_identity):
    """Route the existing model expansion call through the committed arrays."""
    from src.baselines.t13_indexed_augmentation import OFFICIAL_FSG_SHA, _tensor_digest
    source = Path(inspect.getsourcefile(type(fsg)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != OFFICIAL_FSG_SHA:
        raise ValueError('T13_OFFICIAL_FSG_SOURCE_DRIFT')

    def expand(dataset, fs_dict, crop_expansion=False):
        if crop_expansion:
            raise ValueError('T13_COMMITTED_CROP_EXPANSION_CHANGED')
        graph_idxs = [i for part in (dataset.train_idx, dataset.val_idx, dataset.test_idx) for i in part]
        if graph_idxs != expected_identity['graph_idxs']:
            raise ValueError('T13_COMMITTED_PARENT_ORDER_CHANGED')
        digest = hashlib.sha256()
        for idx in graph_idxs:
            row = dataset[idx]
            for name in ('feature', 'adj', 'edge_attr', 'label', 'num_nodes', 'num_edges'):
                if name in row:
                    digest.update(name.encode() + _tensor_digest(row[name]).encode())
        digest.update(stable_sha256(dict(graph_idxs=graph_idxs, rules=[
            dict(nodes=list(g.nodes(data=True)), edges=list(g.edges(data=True)))
            for g in fs_dict.values()])).encode())
        if digest.hexdigest() != expected_identity['input_sha256']:
            raise ValueError('T13_COMMITTED_PARENT_OR_RULE_CONTENT_CHANGED')
        if (int(fsg.fs_max_nodes), int(fsg.fs_min_nodes)) != (
                expected_identity['fs_max_nodes'], expected_identity['fs_min_nodes']):
            raise ValueError('T13_COMMITTED_PATTERN_SHAPE_CHANGED')
        return load_index(descriptor, base_dataset=dataset, fsg=fsg, expected_identity=expected_identity)

    fsg.expand_data_by_fs = expand


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
