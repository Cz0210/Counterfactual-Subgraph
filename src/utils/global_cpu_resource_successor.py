"""Bound this GlobalGCE serial CPU successor, without operating its owner."""
from __future__ import annotations

import copy


def serial_chain_peak(gpu_peak, export_peak, evaluation_peak, waiting_peak=8):
    """Export exits before terminal publishes and the CPU evaluator may start.

    The independent waiting owner can overlap GPU/export. Its bounded atomic
    heartbeat/receipt margin is retained, even though most files already exist.
    No unrelated future stage is assumed to have zero resource requirements.
    """
    peaks = (gpu_peak, export_peak, evaluation_peak, waiting_peak)
    if any(type(p) is not int or p < 0 for p in peaks):
        raise ValueError('SERIAL_PEAK_UNKNOWN')
    return max(gpu_peak + waiting_peak, export_peak + waiting_peak,
               evaluation_peak)


def assert_export_only(owner_spec):
    successors = owner_spec.get('cpu_successors')
    if not isinstance(successors, list) or len(successors) != 1:
        raise ValueError('GLOBAL_CPU_EXPORT_SEQUENCE_CHANGED')
    stage = successors[0]
    command = stage.get('command', [])
    if (stage.get('opens_test') is not False or command.count('--action') != 1
            or command[command.index('--action') + 1] != 'export'):
        raise ValueError('GLOBAL_CPU_EXPORT_ONLY_PROOF_REQUIRED')


def prepare_cpu_spec(old_spec, *, resource_descriptor):
    """Only resource descriptor changes: original candidate/oracle/test bindings stay."""
    if old_spec.get('main_matrix_write') is not False:
        raise ValueError('INDEPENDENT_EXPERIMENT_ONLY')
    new = copy.deepcopy(old_spec)
    new['cpu_resource_config'] = copy.deepcopy(resource_descriptor)
    differences = {key for key in old_spec if old_spec[key] != new[key]}
    if differences != {'cpu_resource_config'}:
        raise ValueError('CPU_REBIND_MUST_CHANGE_ONLY_RESOURCE')
    return new


def assert_dynamic_config_only(old, new):
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    before.pop('stage_file_policy', None)
    after.pop('stage_file_policy', None)
    if before != after:
        raise ValueError('AIDS_DYNAMIC_REBIND_CHANGED_NONFILE_RESOURCE_CONTRACT')


def joint_memory_assessment(*, legacy_floor, concurrent_reserve, headroom,
                            proven_incremental_peak=None):
    """An old 8 GiB floor is not evidence that a whole evaluator peaks at 8 GiB."""
    known = (legacy_floor, concurrent_reserve, headroom)
    if any(type(v) is not int or v < 0 for v in known):
        raise ValueError('MEMORY_OBSERVATION_INVALID')
    if proven_incremental_peak is None:
        return {'state': 'CPU_JOINT_MEMORY_PEAK_EVIDENCE_INSUFFICIENT',
            'legacy_floor_bytes': legacy_floor, 'legacy_floor_is_peak_evidence': False,
            'concurrent_reserve_bytes': concurrent_reserve, 'headroom_bytes': headroom,
            'next_stage_incremental_peak_bytes': None, 'required_headroom_bytes': None,
            'full_resource_admission': False, 'shortfall_bytes': None,
            'activation_allowed': False}
    if type(proven_incremental_peak) is not int or proven_incremental_peak < 0:
        raise ValueError('MEMORY_PEAK_INVALID')
    required = max(legacy_floor, concurrent_reserve + proven_incremental_peak)
    return {'state': 'PASS' if headroom >= required else 'WAITING_JOINT_MEMORY',
        'required_headroom_bytes': required, 'headroom_bytes': headroom,
        'shortfall_bytes': max(0, required-headroom),
        'full_resource_admission': headroom >= required,
        'activation_allowed': headroom >= required}
