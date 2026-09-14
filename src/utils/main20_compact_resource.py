"""September14 user-authorized narrow stages, not a replacement global guard."""
import math


def admission(*, mode, available, own_peak, other_peak, external_decline_15m,
              new_bytes, io_pass, nvme_free=None, nvme_uncreated_peak=0):
    if mode not in {'SEALED_SMALL_RELEASE','BOUNDED_COMPACT_STAGE'}:
        raise ValueError('Unknown narrow policy; retain original science guard')
    values=[available,own_peak,other_peak,external_decline_15m,new_bytes,nvme_uncreated_peak]
    if any(v is None or isinstance(v,bool) or v<0 for v in values):
        raise ValueError('Unknown remaining peak is not zero')
    buffer=max(512,math.ceil(2*external_decline_15m))
    blockers=[]
    if not io_pass: blockers.append('ACTUAL_IO_PROBE_FAILED')
    if mode=='SEALED_SMALL_RELEASE' and (own_peak>128 or new_bytes>256*1024**2):
        blockers.append('TURN_SHARED_SMALL_RELEASE_BUDGET_EXCEEDED')
    if mode=='BOUNDED_COMPACT_STAGE' and (own_peak>256 or own_peak+other_peak>512):
        blockers.append('COMPACT_JOINT_BOUND_EXCEEDED')
    if available-own_peak-other_peak < 8192+buffer: blockers.append('NARROW_FILE_HEADROOM')
    if nvme_free is not None and nvme_free-nvme_uncreated_peak<2*1024**3:
        blockers.append('NVME_JOINT_2GIB_RESERVE')
    return dict(mode=mode,allowed=not blockers,blockers=blockers,available=available,
        own_uncreated_peak=own_peak,other_uncreated_peak=other_peak,dynamic_buffer=buffer,
        required_available=8192+buffer+own_peak+other_peak,original_20000_policy_unchanged=True)
