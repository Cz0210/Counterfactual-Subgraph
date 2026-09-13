"""Explicit, user-authorized IndexRebuilt continuation; never claim old parity.

Only the lost randomized mask/index layout may differ. Original inputs,
multiplicities, official construction, train state and epoch budget remain bound.
This module does not acquire a lease or authorize training on its own.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import random
from pathlib import Path

from src.eval.bace_frozen_gnn_contracts import stable_sha256

INVARIANTS = (
    'schema', 'parent_count', 'sample_count', 'fs_max_nodes', 'fs_min_nodes',
    'node_feat_dim', 'edge_attr_dim', 'max_num_nodes', 'graph_idxs', 'input_sha256',
    'sampler', 'compact_index_bytes', 'official_fsg_sha256',
    'full_augmented_tensor_materialization', 'all_masks_reconstructed_exactly',
)
LAYOUT_FIELDS = (
    'index_sha256', 'masks_sha256', 'split_sha256',
    'python_rng_before_sha256', 'python_rng_after_sha256',
)


def validate_rebuilt_identity(original, rebuilt):
    for label, value in [('original', original), ('rebuilt', rebuilt)]:
        if stable_sha256({k:v for k,v in value.items() if k != 'identity_sha256'}) != value.get('identity_sha256'):
            raise ValueError('T13_INDEX_IDENTITY_DIGEST:' + label)
    for key in INVARIANTS:
        if key not in original or key not in rebuilt or original[key] != rebuilt[key]:
            raise ValueError('T13_INDEX_REBUILT_NONLAYOUT_DIFFERENCE:' + key)
    if rebuilt['full_augmented_tensor_materialization'] or not rebuilt['all_masks_reconstructed_exactly']:
        raise ValueError('T13_INDEX_REBUILT_NOT_BOUNDED')
    return dict(scope='T13_INDEX_REBUILT_SAME_RUN_CONTINUATION',
        old_trajectory_parity_claimed=False, formal_fresh_start=False,
        invariant_fields=list(INVARIANTS), original_identity=original,
        rebuilt_identity=rebuilt,
        changed_layout_fields=[k for k in LAYOUT_FIELDS if original[k] != rebuilt[k]],
        original_checkpoint_mutated=False, training_state_adopted=False)


def rebuild_authorized(*, base_dataset, fsg, fs_dict, expected_identity, private_seed):
    from .t13_indexed_augmentation import OFFICIAL_FSG_SHA, _eager_without_split, _rng_digest, build_indexed_dataset
    source = Path(inspect.getsourcefile(type(fsg)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != OFFICIAL_FSG_SHA:
        raise ValueError('T13_INDEX_REBUILT_OFFICIAL_SOURCE_CHANGED')
    # A task-local RNG does not consume the committed training RNG. The original
    # fixed split implementation must also be globally RNG-neutral.
    before = _rng_digest()
    data = importlib.import_module('data.dataset')
    rebuilt = build_indexed_dataset(fsg, base_dataset, fs_dict,
        split_fn=data.get_train_val_test_idx,
        eager_dataset_class=_eager_without_split(data.AugmentedDataset),
        mask_rng=random.Random(private_seed))
    if _rng_digest() != before:
        raise ValueError('T13_INDEX_REBUILT_CHANGED_GLOBAL_RNG')
    receipt = validate_rebuilt_identity(expected_identity, rebuilt.identity)
    receipt.update(private_index_rng_seed=private_seed, global_rng_unchanged=True,
                   scientific_checkpoint_rng_not_used=True, remining=False)
    rebuilt.reconstruction_receipt = receipt
    return rebuilt


def adapted_checkpoint(original, rebuilt_identity, *, original_identity, formal_ledger):
    """New working checkpoint, preserving every non-layout training field.

    Caller must separately verify source checkpoint identity, runtime backend,
    input hashes, real-batch/reload evidence, physical lease, and storage commit.
    The output is not a formal start and is not marked durable here.
    """
    receipt = validate_rebuilt_identity(original_identity, rebuilt_identity)
    if formal_ledger.get('max_full_starts') != 1 or not formal_ledger.get('attempt_id'):
        raise ValueError('T13_ORIGINAL_FORMAL_LEDGER_REQUIRED')
    required = {'model_state','optimizer_state','scheduler_state','python_rng_state',
        'numpy_rng_state','torch_rng_state','cuda_rng_state','resume_identity',
        'resume_identity_sha256','sampler_state','augmented_dataset_identity',
        'next_epoch','best_loss','config'}
    if not required <= set(original):
        raise ValueError('T13_INCOMPLETE_TRAINING_STATE:' + ','.join(sorted(required-set(original))))
    if original['augmented_dataset_identity'] != original_identity:
        raise ValueError('T13_SOURCE_INDEX_CHECKPOINT_BINDING')
    identity = original['resume_identity']
    if identity.get('dataset') != 'TasteMolNet' or identity.get('source_label') != 1 or identity.get('target_label') != 0:
        raise ValueError('T13_INDEX_REBUILT_WRONG_DATASET_OR_TARGET')
    # The actual v2 producer stores next_epoch=epoch+1, not an epoch field.
    if original['next_epoch'] != 30 or original['config']['epochs'] != 100 or original['scheduler_state'].get('last_epoch') != 30:
        raise ValueError('T13_INDEX_REBUILT_NOT_AUTHORIZED_EPOCH29_BOUNDARY')
    if not original['model_state'] or not original['optimizer_state'].get('state'):
        raise ValueError('T13_MODEL_OR_OPTIMIZER_EMPTY')
    if original['sampler_state'] != dict(original_identity['sampler'], next_epoch=30):
        raise ValueError('T13_SOURCE_SAMPLER_CURSOR_INVALID')
    result = copy.deepcopy(original)
    result['augmented_dataset_identity'] = copy.deepcopy(rebuilt_identity)
    result['sampler_state'] = dict(rebuilt_identity['sampler'], next_epoch=30)
    receipt.update(original_formal_attempt_id=formal_ledger['attempt_id'],
        formal_quota_used='1/1', resume_epoch=30, validation_epoch30_required=True,
        training_state_adopted=True, persistent_committed=False,
        training_backend_validation_required=True, real_batch_reload_validation_required=True)
    # Additional metadata is honest and separate from the unchanged original
    # typed resume_identity. No PASS field is introduced.
    result['index_rebuilt_continuation'] = receipt
    return result, receipt
