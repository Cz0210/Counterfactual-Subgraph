"""Taste T13 storage-only expansion of the pinned official GlobalGCE dataset.

Mask enumeration and sampling remain official. Only persistent representation
changes: parent tensors are shared, augmented samples are integer indices, and
Cartesian masks are stored as their exact ordered node axes. Dense augmented
matrices are materialized only for the DataLoader's bounded batch.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256

SCHEMA = 't13_indexed_augmentation_v1'
OFFICIAL_FSG_SHA = '504cf5ba9ee1a6be32c6b201cf602b0eca9a50e52971cb9d76b62f0e896902bc'


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode() + value.numpy().tobytes()).hexdigest()


def _rng_digest():
    numpy_state=np.random.get_state()
    return stable_sha256(dict(python=random.getstate(),torch=_tensor_digest(torch.get_rng_state()),
        numpy=[numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:]]))


def restore_mask(axes):
    """Exact product(axis, repeat=2), including the upstream all-minus-one mask."""
    axes = axes.to(dtype=torch.long)
    size = axes.shape[-1]
    return torch.stack((axes.unsqueeze(-1).expand(-1, size, size),
                        axes.unsqueeze(-2).expand(-1, size, size)), dim=-1).reshape(2, size * size, 2)


class T13IndexedAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, *, dataset, fsg, graph_idxs, positions, fs_indices, axes, split_fn,
                 input_sha256, mask_sha256, rng_before_sha256, rng_after_sha256, boundary_checks):
        self.dataset = dataset
        self.fsg = fsg
        self.graph_idxs = tuple(graph_idxs)
        # The arrays retain their storage; torch.frombuffer never copies all masks.
        self._buffers = (positions, fs_indices, axes)
        self.graph_idx_list = torch.frombuffer(positions, dtype=torch.int32)
        self.fs_idx_list = torch.frombuffer(fs_indices, dtype=torch.int32).reshape(-1, 2)
        self.mask_axes = torch.frombuffer(axes, dtype=torch.int32).reshape(-1, 2, int(fsg.fs_max_nodes))
        self.index = range(len(self.graph_idx_list))
        for name in ('dataset_name', 'node_feat_dim', 'edge_attr_dim', 'max_num_nodes', 'num_classes'):
            setattr(self, name, getattr(dataset, name))
        labels = torch.tensor([int(dataset.labels[self.graph_idxs[int(p)]]) for p in self.graph_idx_list], dtype=torch.long)
        # Preserve the pinned stratified 33/33 split and full sample multiplicity.
        self.train_idx, self.val_idx, self.test_idx = split_fn(len(self), labels)
        self._cache = OrderedDict()
        self.cache_max_parents = 8
        index_digest=hashlib.sha256()
        for buffer in self._buffers:index_digest.update(memoryview(buffer))
        self.identity = dict(schema=SCHEMA, storage_only=True, parent_count=len(graph_idxs), sample_count=len(self),
            fs_max_nodes=int(fsg.fs_max_nodes),fs_min_nodes=int(fsg.fs_min_nodes),
            node_feat_dim=self.node_feat_dim,edge_attr_dim=self.edge_attr_dim,max_num_nodes=self.max_num_nodes,
            graph_idxs=list(graph_idxs), index_sha256=index_digest.hexdigest(),
            input_sha256=input_sha256, masks_sha256=mask_sha256,
            python_rng_before_sha256=rng_before_sha256, python_rng_after_sha256=rng_after_sha256,
            split_sha256=stable_sha256(dict(train=self.train_idx, validation=self.val_idx, test=self.test_idx)),
            sampler=dict(train_shuffle=False, validation_shuffle=False, test_shuffle=False,
                batch_size=500, num_workers=0, next_batch=0, epoch_boundary_resume=True),
            compact_index_bytes=sum(len(a)*a.itemsize for a in self._buffers),
            full_augmented_tensor_materialization=False, production_boundary_checks=boundary_checks)
        self.identity['identity_sha256'] = stable_sha256(self.identity)

    def __len__(self):
        return len(self.graph_idx_list)

    def _parent(self, position):
        if position in self._cache:
            self._cache.move_to_end(position)
            return self._cache[position]
        row = self.dataset[self.graph_idxs[position]]
        edges = row['edge_attr'].argmax(-1).unsqueeze(0) if 'edge_attr' in row else None
        features, adjacency, edges = self.fsg.expand_graphs_size(row['feature'].argmax(-1).unsqueeze(0),
            row['adj'].unsqueeze(0), edges, 2 * (self.fsg.fs_max_nodes - self.fsg.fs_min_nodes))
        parent = dict(feature=F.one_hot(features.long(), self.node_feat_dim).float()[0], adj=adjacency[0],
            label=row['label'], num_nodes=row['num_nodes'], num_edges=row['num_edges'])
        if edges is not None:
            parent['edge_attr'] = F.one_hot(edges.long(), self.edge_attr_dim).float()[0]
        self._cache[position] = parent
        if len(self._cache) > self.cache_max_parents:
            self._cache.popitem(last=False)
        return parent

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = int(self.graph_idx_list[index])
        # Official recourse mutates collated tensors. Never expose shared tensors.
        graph = {name: value.clone() for name, value in self._parent(position).items()}
        graph.update(max_num_nodes=self.max_num_nodes, index=index,
            fs_index=self.fs_idx_list[index].long(), mask_idx_list=restore_mask(self.mask_axes[index]),
            g_idx_list=torch.tensor(position, dtype=torch.long))
        return graph


def build_indexed_dataset(fsg, dataset, fs_dict, crop_expansion=False, *, output_root=None,
                          get_nx_graph=None, split_fn=None, eager_dataset_class=None, verify_boundaries=True):
    """Enumerate identical masks once, with at most one parent's dense scratch."""
    if get_nx_graph is None:
        get_nx_graph = importlib.import_module('utils').get_nx_graph
    official_data = importlib.import_module('data.dataset') if split_fn is None or eager_dataset_class is None else None
    split_fn = split_fn or official_data.get_train_val_test_idx
    eager_dataset_class = eager_dataset_class or official_data.AugmentedDataset
    graph_idxs = [i for part in (dataset.train_idx, dataset.val_idx, dataset.test_idx) for i in part]
    positions, fs_indices, axes = array('i'), array('i'), array('i')
    _require(positions.itemsize == 4, 'T13_INDEX_REQUIRES_INT32')
    inputs, masks_digest = hashlib.sha256(), hashlib.sha256()
    rng_before = stable_sha256(random.getstate())
    checks = []
    fs_max = int(fsg.fs_max_nodes)
    for position, idx in enumerate(graph_idxs):
        # Exactly one original getter per field value; no RNG is consumed here.
        row = dataset[idx]
        feat = row['feature'].argmax(-1)
        edge = row['edge_attr'].argmax(-1) if 'edge_attr' in row else None
        graph = get_nx_graph(feat, row['adj'], edge, remove_isolated_nodes=False)
        for name in ('feature', 'adj', 'edge_attr', 'label', 'num_nodes', 'num_edges'):
            if name in row:
                inputs.update(name.encode()+_tensor_digest(row[name]).encode())
        current_masks, current_fs = fsg.get_graph_masks(graph, fs_dict, crop_expansion)
        _require(len(current_masks) == len(current_fs), 'T13_MASK_INDEX_LENGTH')
        for pair, fs_pair in zip(current_masks, current_fs, strict=True):
            _require(len(pair) == len(fs_pair) == 2, 'T13_TWO_RULE_MASK_SHAPE')
            packed = []
            for mask in pair:
                _require(mask.dtype == torch.int64 and tuple(mask.shape) == (fs_max*fs_max, 2), 'T13_MASK_CONTRACT')
                packed.append(mask[::fs_max, 0].to(dtype=torch.int32))
                masks_digest.update(mask.contiguous().numpy().tobytes())
            reconstructed = restore_mask(torch.stack(packed))
            _require(all(torch.equal(reconstructed[i], pair[i]) for i in (0, 1)), 'T13_NON_CARTESIAN_MASK')
            positions.append(position)
            fs_indices.extend(int(x) for x in fs_pair)
            axes.extend(int(x) for axis in packed for x in axis)
        if verify_boundaries and current_masks:
            # A bounded two-record official eager fixture, never the full universe.
            picked = [0, len(current_masks)-1]
            feature, adjacency, expanded_edge = fsg.expand_graphs_size(feat.unsqueeze(0), row['adj'].unsqueeze(0),
                None if edge is None else edge.unsqueeze(0), 2 * (fsg.fs_max_nodes-fsg.fs_min_nodes))
            expected = eager_dataset_class(dataset, feature.expand(2, *feature.shape[1:]).clone(),
                adjacency.expand(2, *adjacency.shape[1:]).clone(),
                None if expanded_edge is None else expanded_edge.expand(2, *expanded_edge.shape[1:]).clone(),
                row['label'].repeat(2), row['num_nodes'].repeat(2), row['num_edges'].repeat(2),
                torch.tensor([position,position]), torch.tensor([current_fs[i] for i in picked]),
                torch.stack([torch.stack(current_masks[i]) for i in picked]))
            # Constructing official AugmentedDataset normally splits >=4 samples.
            # The canary adapter below uses the original __getitem__ without split.
            checks.append(dict(parent_position=position, first_sample=len(positions)-len(current_masks),
                last_sample=len(positions)-1, sample_count=len(current_masks),
                expected=[{k:_tensor_digest(v) if torch.is_tensor(v) else v for k,v in expected[i].items()} for i in (0,1)]))
        if output_root is not None and (position % 25 == 0 or position+1 == len(graph_idxs)):
            atomic_json(Path(output_root)/'index_progress.json', dict(parent_count=position+1,
                total_parents=len(graph_idxs), samples=len(positions), compact_bytes=sum(len(a)*a.itemsize for a in (positions,fs_indices,axes))))
    _require(len(positions)>0, 'T13_RULES_DO_NOT_EXIST')
    inputs.update(stable_sha256(dict(graph_idxs=graph_idxs, rules=[
        dict(nodes=list(g.nodes(data=True)), edges=list(g.edges(data=True))) for g in fs_dict.values()])).encode())
    before_materialization=_rng_digest()
    result = T13IndexedAugmentedDataset(dataset=dataset,fsg=fsg,graph_idxs=graph_idxs,positions=positions,
        fs_indices=fs_indices,axes=axes,split_fn=split_fn,input_sha256=inputs.hexdigest(),mask_sha256=masks_digest.hexdigest(),
        rng_before_sha256=rng_before,rng_after_sha256=stable_sha256(random.getstate()),boundary_checks=len(checks))
    for check in checks:
        for source_index, target_index in enumerate((check['first_sample'],check['last_sample'])):
            observed = result[target_index]
            expected = dict(check['expected'][source_index]);expected['index']=target_index
            _require({k:_tensor_digest(v) if torch.is_tensor(v) else v for k,v in observed.items()} == expected,
                'T13_PRODUCTION_BOUNDARY_EAGER_LAZY_DIFFERENCE')
    result._cache.clear()
    _require(_rng_digest()==before_materialization,'T13_MATERIALIZATION_CHANGED_RNG')
    result.identity['all_masks_reconstructed_exactly']=True
    result.identity['materialization_rng_unchanged']=True
    result.identity['materialization_rng_sha256']=before_materialization
    result.identity['official_fsg_sha256']=OFFICIAL_FSG_SHA
    result.identity['identity_sha256']=stable_sha256({k:v for k,v in result.identity.items() if k!='identity_sha256'})
    if output_root is not None:
        destination=Path(output_root)/'indexed_dataset_manifest.json'
        if destination.exists():
            _require(json.loads(destination.read_text())==result.identity,'T13_SEALED_INDEX_INPUT_OR_RNG_DRIFT')
        else:
            atomic_json(destination,result.identity)
    return result


def _eager_without_split(original_class):
    """Use the original eager item contract for a two-row production boundary."""
    class BoundedEager(original_class):
        def __init__(self, dataset, feat, adj, edge_attr, labels, num_nodes, num_edges, graph_idx, fs_idx, masks):
            self.dataset_name=dataset.dataset_name;self.node_feat_dim=dataset.node_feat_dim
            self.edge_attr_dim=dataset.edge_attr_dim;self.max_num_nodes=dataset.max_num_nodes;self.num_classes=dataset.num_classes
            self.feat=F.one_hot(feat.long().squeeze(),self.node_feat_dim).float();self.adj=adj
            self.edge_attr=F.one_hot(edge_attr.long().squeeze(),self.edge_attr_dim).float() if self.edge_attr_dim else None
            self.labels=labels;self.num_nodes=num_nodes;self.num_edges=num_edges;self.index=list(range(len(labels)))
            self.graph_idx_list=graph_idx;self.fs_idx_list=fs_idx;self.fs_mask_list=masks
    return BoundedEager


def install_t13_indexed_expansion(fsg, *, output_root):
    """Explicit single-instance storage adapter; never edits imported source."""
    source=Path(inspect.getsourcefile(type(fsg)))
    _require(hashlib.sha256(source.read_bytes()).hexdigest()==OFFICIAL_FSG_SHA, 'T13_OFFICIAL_FSG_SOURCE_DRIFT')
    original_data=importlib.import_module('data.dataset')
    def expand(dataset,fs_dict,crop_expansion=False):
        return build_indexed_dataset(fsg,dataset,fs_dict,crop_expansion,output_root=output_root,
            split_fn=original_data.get_train_val_test_idx,eager_dataset_class=_eager_without_split(original_data.AugmentedDataset))
    fsg.expand_data_by_fs=expand
