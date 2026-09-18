"""Scoped same-run checkpoint I/O: content and held-FD stability, not ctime order.

The default GlobalGCE guards are unchanged. This context is only installed by
the explicitly bound V10 continuation, never by a fresh trainer or other task.
"""
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path


def stable_checkpoint_bytes(path, *, expected_sha256=None):
    from src.baselines.globalgce_resumable import _regular_fd_evidence, _named_checkpoint_stat
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = _regular_fd_evidence(fd)
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data = b''.join(chunks)
        after = _regular_fd_evidence(fd)
        named = _named_checkpoint_stat(path)
        # FD metadata (including ctime) and bytes must remain stable during read.
        # Only the path-vs-FD ctime *ordering* is not scientific identity.
        if before != after or hashlib.sha256(data).hexdigest() != before['sha256']:
            raise ValueError('V10_CHECKPOINT_READ_MUTATION')
        if any(named[k] != before[k] for k in named if k != 'ctime_ns'):
            raise ValueError('V10_CHECKPOINT_PATH_VERSION_CHANGED')
        if expected_sha256 is not None and before['sha256'] != expected_sha256:
            raise ValueError('V10_CHECKPOINT_CONTENT_BINDING_CHANGED')
        return data, before
    finally:
        os.close(fd)


def validate_state(checkpoint, binding, *, tensor_finite):
    required = {'model_state','optimizer_state','scheduler_state','python_rng_state',
        'numpy_rng_state','torch_rng_state','cuda_rng_state','sampler_state',
        'augmented_dataset_identity','resume_identity','resume_identity_sha256',
        'next_epoch','best_loss','best_state_seen','checkpoint_schema_version'}
    if not required <= checkpoint.keys():
        raise ValueError('V10_INCOMPLETE_CHECKPOINT_STATE')
    if any(checkpoint[k] is None for k in required):
        raise ValueError('V10_INCOMPLETE_CHECKPOINT_STATE')
    if (binding['original_formal_attempt_id'] != '7b647a43-1919-4d8e-a05a-0f6071255f2e'
            or binding['formal_quota'] != '1/1' or binding['target'] != 0
            or checkpoint['checkpoint_schema_version'] != 'globalgce_epoch_checkpoint_v2'
            or checkpoint['next_epoch'] != binding['next_epoch']
            or binding['completed_epoch'] + 1 != binding['next_epoch']
            or not 30 < binding['next_epoch'] <= 101):
        raise ValueError('V10_SAME_RUN_LOGICAL_VERSION_MISMATCH')
    identity = checkpoint['resume_identity']
    if (identity['target_label'] != 0 or identity['source_label'] != 1
            or identity['training_config']['epochs'] != 100
            or checkpoint['resume_identity_sha256'] != binding['resume_identity_sha256']
            or checkpoint['augmented_dataset_identity']['identity_sha256'] != binding['compact_identity_sha256']):
        raise ValueError('V10_SCIENTIFIC_IDENTITY_CHANGED')
    if (checkpoint['sampler_state']['next_epoch'] != binding['next_epoch']
            or checkpoint['sampler_state']['next_batch'] != 0
            or checkpoint['scheduler_state']['last_epoch'] != binding['next_epoch']):
        raise ValueError('V10_SAMPLER_SCHEDULER_BOUNDARY_MISMATCH')
    steps = {int(float(v['step'])) for v in checkpoint['optimizer_state']['state'].values() if 'step' in v}
    if steps != {binding['next_epoch']} or not checkpoint['best_state_seen']:
        raise ValueError('V10_OPTIMIZER_OR_BEST_STATE_INCOMPLETE')
    if (binding.get('producer_phase') != 'AFTER_OPTIMIZER_SCHEDULER_AND_DUE_VALIDATION'
            or binding.get('historical_callback_rewritten') is not False
            or not binding.get('producer_evidence')):
        raise ValueError('V10_PRODUCER_PHASE_EVIDENCE_REQUIRED')
    def walk(value):
        if isinstance(value, dict):
            for v in value.values(): walk(v)
        elif isinstance(value, (list,tuple)):
            for v in value: walk(v)
        elif hasattr(value,'is_floating_point') and value.is_floating_point() and not tensor_finite(value):
            raise ValueError('V10_NONFINITE_TRAINING_STATE')
    walk(checkpoint['model_state']); walk(checkpoint['optimizer_state'])


def load_bound_checkpoint(torch, descriptor):
    from src.utils.t13_performance_dispatch import bound_json
    from src.eval.bace_frozen_gnn_contracts import sha256_file
    binding = bound_json(descriptor)
    for evidence in binding['producer_evidence']:
        if sha256_file(evidence['path']) != evidence['sha256']:
            raise ValueError('V10_PRODUCER_EVIDENCE_CHANGED')
    payload, physical = stable_checkpoint_bytes(binding['source_path'], expected_sha256=binding['source_sha256'])
    checkpoint = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    validate_state(checkpoint, binding, tensor_finite=lambda t: bool(torch.isfinite(t).all()))
    return checkpoint, binding, physical


@contextmanager
def checkpoint_io_scope(root):
    """Read only this fresh continuation's checkpoint leaves with scoped policy."""
    from src.baselines import globalgce_resumable as native
    root = Path(root).resolve()
    old_open, old_load = native._open_regular_file_evidence, native._load_torch_checkpoint_held
    def scoped(path):
        path = Path(path).resolve()
        return (path.is_relative_to(root) and path.parent.name == 'globalgce_training_checkpoints'
                and path.name in {'training_checkpoint.pt','training_heartbeat.json'})
    def evidence(path):
        if not scoped(path): return old_open(path)
        return stable_checkpoint_bytes(path)[1]
    def load(torch_module, path, *, map_location, expected_evidence):
        if not scoped(path):
            return old_load(torch_module,path,map_location=map_location,expected_evidence=expected_evidence)
        data, observed = stable_checkpoint_bytes(path)
        if expected_evidence is not None and observed != dict(expected_evidence):
            raise ValueError('V10_EXPECTED_PHYSICAL_LEAF_CHANGED')
        return torch_module.load(io.BytesIO(data),map_location=map_location,weights_only=False),observed
    native._open_regular_file_evidence, native._load_torch_checkpoint_held = evidence, load
    try: yield
    finally: native._open_regular_file_evidence, native._load_torch_checkpoint_held = old_open, old_load
