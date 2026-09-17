"""Narrow identity adapter for reference recovery on its original GPU.

The reviewed cross-GPU validator intentionally rejects equal UUIDs.  Recovery
on the reference GPU is a different transport case, not an excuse to relax
scientific identity or to claim algorithm parity.
"""
import copy


def validate_same_gpu_resume_identity(*, current, authority,
                                      scientific_source_equivalence=None):
    from src.utils.tastemolnet_t12_accelerated_from250 import (
        validate_scientific_source_equivalence_binding,
    )

    left, right = copy.deepcopy(current), copy.deepcopy(authority)
    identities = [x['identity_template'] for x in (left, right)]
    runtimes = [x['runtime'] for x in (left, right)]
    uuids = [x['gpu_uuid'] for x in identities]
    if not uuids[0] or uuids[0] != uuids[1]:
        raise ValueError('T12_SAME_GPU_UUID_REQUIRED')
    commits = [(x['execution_commit'], x['execution_tree']) for x in identities]
    for identity, runtime in zip(identities, runtimes):
        if (runtime['execution_commit'], runtime['execution_tree']) != (
                identity['execution_commit'], identity['execution_tree']):
            raise ValueError('T12_RUNTIME_SOURCE_IDENTITY_DISAGREES')
        if runtime['gpu']['gpu_uuid'] != identity['gpu_uuid']:
            raise ValueError('T12_RUNTIME_GPU_IDENTITY_DISAGREES')
    binding = None
    if commits[0] != commits[1]:
        binding = validate_scientific_source_equivalence_binding(
            scientific_source_equivalence,
            reference_commit=commits[1][0], reference_tree=commits[1][1],
            current_commit=commits[0][0], current_tree=commits[0][1])
    for value in (left, right):
        identity, runtime = value['identity_template'], value['runtime']
        for digest in (identity.pop('runtime_identity_sha256'),
                       value.pop('transition_contract_sha256')):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError('T12_DERIVED_IDENTITY_DIGEST_INVALID')
            int(digest, 16)
        # These two fields are transport selectors only. UUID and all hardware,
        # precision, deterministic, model, split, and algorithm fields remain.
        for field in ('visible_selector', 'physical_index'):
            runtime['gpu'].pop(field, None)
        if binding is not None:
            for record in (identity, runtime):
                record.pop('execution_commit')
                record.pop('execution_tree')
    if left != right:
        raise ValueError('T12_SAME_GPU_NONTRANSPORT_IDENTITY_CHANGED')
    return dict(schema_version='t12_same_gpu_source_bound_restore_v9',
        status='SAME_GPU_IDENTITY_VERIFIED_NOT_ALGORITHM_PARITY',
        authority_gpu_uuid=uuids[1], transport_gpu_uuid=uuids[0],
        authority_execution_commit=commits[1][0], transport_execution_commit=commits[0][0],
        cross_commit_source_equivalence_verified=binding is not None,
        scientific_source_equivalence_receipt_sha256=None if binding is None else binding['receipt_sha256'],
        checkpoint_identity_retained_from_authority=True,
        scientific_equivalence_claimed_before_parity=False)
