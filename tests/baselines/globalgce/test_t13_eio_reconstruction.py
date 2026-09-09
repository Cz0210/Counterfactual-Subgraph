import copy
import random
from types import SimpleNamespace

import pytest

from src.baselines.t13_bounded_payload import _adopt_proven_reconstruction
from src.baselines.t13_indexed_augmentation import _with_independent_mask_rng
from src.baselines.t13_real_batch_performance import make_interceptor
from src.eval.bace_frozen_gnn_contracts import stable_sha256


def identity():
    value = dict(index_sha256='index', masks_sha256='masks', input_sha256='inputs',
                 split_sha256='split', python_rng_before_sha256='rng-start',
                 python_rng_after_sha256='rng-end', materialization_rng_sha256='old-runtime',
                 sample_count=1273625, graph_idxs=[0, 1],
                 all_masks_reconstructed_exactly=True, materialization_rng_unchanged=True)
    value['identity_sha256'] = stable_sha256(value)
    return value


def test_actual_recorded_rng_is_not_just_a_seed_claim():
    # Copied existing immutable manifest field, not an expected fit/result.
    expected = 'e9c954504f0ab33be368c2fd7e814a91a5324d67e695f4a74d12fbe3c072b5d2'
    assert stable_sha256(random.Random(7).getstate()) == expected
    assert stable_sha256(random.Random(8).getstate()) != expected


def test_private_sampling_matches_native_without_global_rng_or_method_mutation():
    class Fsg:
        def get_valid_masks(self, values):
            return random.sample(values, 3)
        def get_graph_masks(self, values):
            return self.get_valid_masks(values), random.sample(values, 2)
    original = Fsg()
    private = random.Random(7)
    isolated = _with_independent_mask_rng(original, private)
    state = random.getstate()
    result = isolated.get_graph_masks(list(range(20)))
    assert random.getstate() == state
    control = random.Random(7)
    assert result == (control.sample(list(range(20)), 3), control.sample(list(range(20)), 2))
    assert 'get_graph_masks' not in vars(original)
    assert Fsg.get_valid_masks.__globals__['random'] is random


@pytest.mark.parametrize('field', ['index_sha256', 'masks_sha256', 'input_sha256', 'split_sha256',
    'python_rng_before_sha256', 'python_rng_after_sha256', 'sample_count', 'graph_idxs',
    'all_masks_reconstructed_exactly', 'materialization_rng_unchanged'])
def test_each_scientific_reconstruction_mismatch_rejected(field):
    expected = identity()
    value = copy.deepcopy(expected)
    value[field] = 'wrong'
    rebuilt = SimpleNamespace(identity=value)
    with pytest.raises(ValueError, match='CONTENT_MISMATCH:' + field):
        _adopt_proven_reconstruction(rebuilt, expected)
    assert rebuilt.identity == value


def test_process_rng_digest_relocation_is_explicit_not_historical_rewrite():
    original = identity()
    expected = copy.deepcopy(original)
    actual = copy.deepcopy(original)
    actual['materialization_rng_sha256'] = 'new-runtime-state-unchanged'
    actual['identity_sha256'] = stable_sha256({k:v for k,v in actual.items() if k != 'identity_sha256'})
    result = _adopt_proven_reconstruction(SimpleNamespace(identity=actual), expected)
    assert result.identity == original == expected
    receipt = result.reconstruction_receipt
    assert receipt['reconstructed_identity'] == actual
    assert receipt['original_identity'] == original
    assert receipt['scientific_resume_validated'] is False


def test_expected_identity_tamper_rejected():
    expected = identity()
    expected['identity_sha256'] = 'invalid'
    with pytest.raises(ValueError, match='EXPECTED_INDEX_IDENTITY_INVALID'):
        _adopt_proven_reconstruction(SimpleNamespace(identity=copy.deepcopy(expected)), expected)


def test_canary_missing_payload_fails_before_any_expansion(tmp_path):
    called = []
    intercept = make_interceptor({}, {}, None, tmp_path, called.append)
    with pytest.raises(ValueError, match='PAYLOAD_MISSING_NOT_RESEEDABLE'):
        intercept(model=object(), gspan_adoption_proof='old-valid-proof')
    assert called == []
