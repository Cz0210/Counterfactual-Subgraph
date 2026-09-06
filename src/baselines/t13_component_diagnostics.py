"""Bounded tensor-exact diagnostics for the T13 canary; never a tolerance gate."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os

import numpy as np
import torch


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def exact_difference(expected, observed, *, max_rows=64):
    """Return exact changed leaves and diagnostic magnitudes, never allclose."""
    differences = []
    changed_count = 0

    def record(path, reason, **details):
        nonlocal changed_count
        changed_count += 1
        if len(differences) < max_rows:
            differences.append(dict(path=path, reason=reason, **details))

    def visit(left, right, path):
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not (torch.is_tensor(left) and torch.is_tensor(right)):
                record(path, 'TYPE_CHANGED'); return
            a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
            if a.dtype != b.dtype or a.shape != b.shape:
                record(path, 'TENSOR_SCHEMA_CHANGED', expected_dtype=str(a.dtype), observed_dtype=str(b.dtype),
                       expected_shape=list(a.shape), observed_shape=list(b.shape)); return
            # Include signed zero / NaN payload changes, as checkpoint hashes do.
            if a.numpy().tobytes() == b.numpy().tobytes():
                return
            finite = torch.isfinite(a) & torch.isfinite(b)
            delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
            maximum = float(delta[finite].max()) if finite.any() else None
            record(path, 'TENSOR_BYTES_CHANGED', unequal_elements=int((a != b).sum()),
                   nonfinite_elements=int((~finite).sum()), max_absolute_difference=maximum,
                   expected_shape=list(a.shape), dtype=str(a.dtype)); return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
                record(path, 'TYPE_CHANGED'); return
            visit(torch.from_numpy(left.copy()), torch.from_numpy(right.copy()), path); return
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                record(path, 'KEYS_CHANGED', missing=[str(k) for k in left if k not in right],
                       extra=[str(k) for k in right if k not in left])
            for key in sorted(set(left) & set(right), key=str):
                visit(left[key], right[key], path + '/' + str(key))
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                record(path, 'SEQUENCE_SCHEMA_CHANGED'); return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, path + '/' + str(index))
            return
        if type(left) is not type(right) or left != right:
            record(path, 'VALUE_CHANGED', expected=left, observed=right)

    visit(expected, observed, '')
    return dict(exact=changed_count == 0, changed_leaf_count=changed_count,
                differences=differences, difference_rows_truncated=changed_count > len(differences),
                tolerance_used=False)


def numeric_runtime():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                torch_version=torch.__version__, cuda_version=torch.version.cuda)


@contextmanager
def strict_diagnostic_runtime(profile):
    if profile not in ('native', 'deterministic'):
        raise ValueError('T13_UNKNOWN_DIAGNOSTIC_PROFILE')
    before = numeric_runtime()
    if profile == 'deterministic':
        if torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('T13_DETERMINISTIC_FRESH_PROCESS_CUBLAS_REQUIRED')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Keep TF32/dtype exactly as the native profile; do not "improve" precision.
    try:
        yield numeric_runtime()
    finally:
        torch.use_deterministic_algorithms(before['deterministic_algorithms'],
                                          warn_only=before['deterministic_warn_only'])
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']
