"""Future-child-only, content-bound T12 history read location/buffering overlay.

No scientific module is edited. Original record decoder, chain validation and
first-seen restoration execute unchanged; no inference/transition is added.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from src.utils.t13_performance_dispatch import bound_json
from src.eval.bace_frozen_gnn_contracts import atomic_json


@contextmanager
def history_io_overlay(descriptor, output_root):
    policy = bound_json(descriptor)
    if (policy['authorization_id'] != 'T12_SINGLE_CONTENT_EQUIVALENT_IO_RELOCATION_V10_20260918'
            or policy['buffer_bytes'] != 1024*1024 or policy['model_fixture_calls'] != 0):
        raise ValueError('T12_V10_IO_ONLY_POLICY_REQUIRED')
    root = Path(output_root).absolute()
    mapping = {}
    for segment in policy['segments']:
        relative = Path(segment['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'bridge_history':
            raise ValueError('T12_V10_HISTORY_SCOPE')
        mapping[root/relative] = segment
    old_open = Path.open
    verified = set()
    opened = []
    def read_open(path, mode='r', buffering=-1, *args, **kwargs):
        path = path.absolute()
        segment = mapping.get(path)
        if segment is None or mode != 'rb':
            return old_open(path,mode,buffering,*args,**kwargs)
        actual = Path(segment.get('cache_path') or path)
        if actual != path and actual not in verified:
            if actual.is_symlink() or actual.stat().st_size != segment['bytes']:
                raise ValueError('T12_V10_CACHE_NOT_SEALED_REGULAR_FILE')
            before = actual.stat()
            h=hashlib.sha256()
            with old_open(actual,'rb',1024*1024) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            after=actual.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or h.hexdigest()!=segment['sha256']:
                raise ValueError('T12_V10_CACHE_CHANGED')
            verified.add(actual)
        opened.append(dict(requested_path=str(path),actual_path=str(actual),buffer_bytes=policy['buffer_bytes'],expected_prefix_sha256=segment['sha256']))
        atomic_json(root/'actual_history_io_v10.json',dict(policy_sha256=descriptor['sha256'],pid=os.getpid(),opens=opened,
            scientific_decoder_unchanged=True,model_calls=0,transitions_added=0))
        return old_open(actual,mode,policy['buffer_bytes'],*args,**kwargs)
    Path.open=read_open
    try:yield
    finally:Path.open=old_open
