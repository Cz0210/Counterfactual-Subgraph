"""Evidence-bound retirement of one failed physical stage, not its science gate."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from src.utils.final16_owner_registry_v1 import process_start_ticks


def verify_terminal_dependency(descriptor, registry, proc_root, *, held_lease=None):
    path = Path(descriptor['path'])
    if not path.is_absolute() or path.is_symlink() or path.stat().st_size > 2 * 1024**2:
        raise ValueError('UNSAFE_TERMINAL_RESOURCE_RECEIPT')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
        raise ValueError('TERMINAL_RESOURCE_RECEIPT_CHANGED')
    receipt = json.loads(raw)
    if (receipt.get('schema') != 't14_terminal_resource_release_v1'
            or receipt.get('science_dependency_for_global_aids') is not False
            or receipt.get('future_stage') != 'WAITING_PARITY'
            or receipt.get('scientific_status') != 'FAILED'
            or receipt.get('physical_lease_released') is not True):
        raise ValueError('TERMINAL_RESOURCE_SCOPE_INVALID')
    proc_root = Path(proc_root)
    if (proc_root / 'sys/kernel/random/boot_id').read_text().strip() != receipt['boot_id']:
        raise ValueError('TERMINAL_RESOURCE_BOOT_CHANGED')
    terminal = Path(receipt['terminal_path'])
    if terminal.is_symlink() or hashlib.sha256(terminal.read_bytes()).hexdigest() != receipt['terminal_sha256']:
        raise ValueError('FAILED_TERMINAL_CHANGED')
    if json.loads(terminal.read_text()).get('status') != 'FAILED':
        raise ValueError('FAILED_TERMINAL_REQUIRED')
    for identity in receipt['retired_processes']:
        if process_start_ticks(proc_root, identity['pid']) == identity['start_ticks']:
            raise ValueError('RETIRED_PROCESS_STILL_LIVE')
    rows = [r for r in registry['tasks'] if r['task_id'] == receipt['task_id']]
    if len(rows) != 1 or any(rows[0].get(k) != v for k, v in {
            'owner_state': 'BLOCKED', 'stage': 'FAILED_PARITY_WAITING_COMPONENT_EVIDENCE',
            'owner_pid': None, 'owner_start_ticks': None}.items()):
        raise ValueError('TERMINAL_TASK_HAS_NEW_OWNER_OR_STAGE')
    leases = [r for r in registry['gpu_leases'] if r['task_id'] == receipt['task_id']]
    if len(leases) != 1 or leases[0]['state'] != 'RELEASED' or leases[0]['lease_path'] != receipt['lease_path']:
        raise ValueError('TERMINAL_LEASE_NOT_RELEASED')
    lock = Path(receipt['lease_path'])
    # Before dispatch, independently acquire/release the old physical lease.
    # After dispatch, validate the existing owner's inherited open description:
    # opening it again must fail, not falsely classify our own lease as foreign.
    if held_lease is not None:
        fd = held_lease['fd']
        opened, named = os.fstat(fd), lock.lstat()
        if (not stat.S_ISREG(opened.st_mode) or lock.is_symlink()
                or [opened.st_dev, opened.st_ino] != receipt['lease_device_inode']
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)):
            raise ValueError('HELD_TERMINAL_LEASE_IDENTITY_CHANGED')
        metadata = json.loads(os.pread(fd, 65536, 0))
        if (metadata.get('pid') != held_lease['owner_pid']
                or metadata.get('run_id') != held_lease['run_id']
                or metadata.get('gpu_uuid') != held_lease['gpu_uuid']
                or metadata.get('ablation_family') != 't13_performance_diagnostic'
                or process_start_ticks(proc_root, held_lease['owner_pid']) != held_lease['owner_start_ticks']):
            raise ValueError('HELD_TERMINAL_OWNER_BINDING_CHANGED')
        with lock.open('r+') as competitor:
            try:
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return receipt['task_id']
            fcntl.flock(competitor, fcntl.LOCK_UN)
            raise ValueError('HELD_TERMINAL_FD_IS_NOT_EXCLUSIVE')
    fd = os.open(lock, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened, named = os.fstat(fd), lock.lstat()
        if (not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
                or [opened.st_dev, opened.st_ino] != receipt['lease_device_inode']):
            raise ValueError('TERMINAL_LEASE_IDENTITY_CHANGED')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return receipt['task_id']
