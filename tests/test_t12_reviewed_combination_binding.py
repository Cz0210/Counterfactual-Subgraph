"""The authorized combination is exact; this is not a runtime parity waiver."""
import copy
import importlib.util
from pathlib import Path

import pytest

from src.utils import tastemolnet_t12_accelerated_from250 as mod


def reseal(r):
    for key in ('reference_inventory', 'current_inventory'):
        v = r[key]
        v['inventory_sha256'] = mod._stable_sha256({k: x for k, x in v.items() if k != 'inventory_sha256'})
    r['audited_differences_sha256'] = mod._stable_sha256({
        'schema_version': 'tastemolnet_t12_scientific_source_delta_audit_v1',
        'differences': r['audited_differences']})
    r['receipt_sha256'] = mod._stable_sha256({k: v for k, v in r.items() if k != 'receipt_sha256'})
    return r


def receipt():
    path = Path(__file__).parent / 'autodl/test_t12_accelerated_from250_v1.py'
    spec = importlib.util.spec_from_file_location('t12_old_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    r = fixture._source_equivalence_receipt(reference_commit=mod.REFERENCE_IMPLEMENTATION_COMMIT,
        reference_tree='2'*40, current_commit=mod.REVIEWED_FOUR_FILE_COMMIT, current_tree='4'*40)
    for key, sha_index, size_index in [('reference_inventory', 0, 1), ('current_inventory', 2, 3)]:
        for row in r[key]['paths']:
            if row['path'] in mod.REVIEWED_FOUR_FILES:
                pin = mod.REVIEWED_FOUR_FILES[row['path']]
                row.update(sha256=pin[sha_index], bytes=pin[size_index])
    r['changed_paths'] = sorted(mod.REVIEWED_FOUR_FILES)
    r['equivalence_basis'] = mod.REVIEWED_FOUR_FILE_BASIS
    r['audited_differences'] = mod._reviewed_four_file_audit(
        {v['path']: v for v in r['reference_inventory']['paths']},
        {v['path']: v for v in r['current_inventory']['paths']}, r['changed_paths'])
    return reseal(r)


def verify(r):
    return mod.validate_scientific_source_equivalence_binding(r, **{
        key: r[key] for key in ('reference_commit', 'reference_tree', 'current_commit', 'current_tree')})


def test_exact_reviewed_combination_preserves_producers_and_no_promotion():
    r = verify(receipt())
    assert r['runtime_parity_claimed'] is False
    assert len(r['audited_differences']) == 4
    assert all(not x['diagnostic_checkpoint_promotable'] for x in r['audited_differences'])
    assert r['audited_differences'][0]['producer_commit'] != r['current_commit']


@pytest.mark.parametrize('mutation', ['hash', 'producer', 'extra_file', 'mix_old_full', 'runtime', 'promotion'])
def test_resealed_unauthorized_content_or_claim_is_rejected(mutation):
    r = receipt()
    if mutation in ('hash', 'mix_old_full'):
        row = next(x for x in r['current_inventory']['paths'] if x['path'] == mod.AUDITED_TRANSPORT_GLUE_PATH)
        row['sha256'] = 'f'*64 if mutation == 'hash' else mod.AUDITED_CURRENT_SOURCE_SHA256
    elif mutation == 'extra_file':
        row = next(x for x in r['current_inventory']['paths'] if x['path'] not in mod.REVIEWED_FOUR_FILES)
        row['sha256'] = 'f'*64
        r['changed_paths'] = sorted([*r['changed_paths'], row['path']])
    elif mutation == 'producer':
        r['audited_differences'][0]['producer_commit'] = 'f'*40
    elif mutation == 'promotion':
        r['audited_differences'][0]['diagnostic_checkpoint_promotable'] = True
    else:
        r['runtime_parity_claimed'] = True
    with pytest.raises(mod.T12AcceleratedError):
        verify(reseal(r))


def test_pins_match_exact_reviewed_commit_not_arbitrary_head():
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parents[1]
    for path, pin in mod.REVIEWED_FOUR_FILES.items():
        body = subprocess.check_output(['git', 'show', mod.REVIEWED_FOUR_FILE_COMMIT+':'+path], cwd=root)
        assert (hashlib.sha256(body).hexdigest(), len(body)) == (pin[2], pin[3])
        body = subprocess.check_output(['git', 'show', pin[4]+':'+path], cwd=root)
        assert hashlib.sha256(body).hexdigest() == pin[2]
