"""Explicit bridge from the original AIDS16 attempt to the shared pilot32.

Reuse saved requests and products, including the independently diagnosed legal
empty result. No test input is read; the extra IDs are a fixed sorted prefix.
Artifact scope and historical per-parent generation RNG scope are distinct.
"""
from pathlib import Path

from .cm_crem_runtime import digest, file_sha, read_json


def generation_scope(spec):
    binding = spec.get('aids_legacy_adoption')
    if binding is None:
        return digest({k: v for k, v in spec.items()
                       if k not in {'execution_commit', 'execution_root', 'output_root'}})
    if spec['dataset'] != 'aids':
        raise ValueError('Legacy AIDS adoption cannot affect another dataset')
    old = read_json(binding['spec'])
    if file_sha(binding['spec']) != binding['spec_sha256']:
        raise ValueError('Legacy AIDS spec changed')
    scope = digest({k: v for k, v in old.items()
                    if k not in {'execution_commit', 'execution_root', 'output_root'}})
    if scope != binding['generation_scope_sha256']:
        raise ValueError('Legacy AIDS RNG namespace changed')
    if old['rf_sha'] != spec['oracle_sha256'] or old['source_csv'] != spec['train']['path']:
        raise ValueError('Legacy AIDS model/train input differs')
    return scope


def fixed32(eligible, records):
    by_id = {p['parent_id']: p for p in eligible}
    old_ids = [a['parent_id'] for a in records]
    if len(old_ids) != 16 or len(set(old_ids)) != 16:
        raise ValueError('Original 16 attribution IDs required')
    ordered = sorted(eligible, key=lambda p: p['parent_id'])
    if [p['parent_id'] for p in ordered[:16]] != old_ids:
        raise ValueError('Original source predicate / sorted prefix changed')
    for a in records:
        if a['smiles'] != by_id[a['parent_id']]['smiles'] or a['before_label'] != 1:
            raise ValueError('Original attribution train/source identity changed')
    if len(ordered) < 32:
        raise ValueError('Insufficient train-source roster')
    return ordered[:32]


def adopt(pilot, eligible):
    from .cm_crem_dataset_pilot import TERMINALS
    from .cm_crem_generation import parent_seed
    b = pilot.spec['aids_legacy_adoption']; scope = generation_scope(pilot.spec)
    old = read_json(b['spec']); root = Path(old['output_root'])
    attrs = read_json(root/'attribution.json')
    if attrs['scope_sha'] != scope or attrs['source_csv_sha'] != pilot.spec['train']['sha256'] or attrs['test_read']:
        raise ValueError('Original train attribution provenance differs')
    selected = fixed32(eligible, attrs['records'])
    pilot.put('pilot_manifest.json', {'parents': selected, 'selected_before_generation': True,
        'outcome_used_for_sampling': False, 'original16_preserved': True,
        'additional16_policy': 'NEXT_SORTED_TRAIN_SOURCE_IDS_FROZEN_BEFORE_NEW_GENERATION',
        'structure': 'LEGACY_FIXED_PREFIX_ADOPTION_NOT_RETROSPECTIVE_STRUCTURAL_RESAMPLING'})
    controls = read_json(b['controls']); diagnosis = read_json(b['diagnosis'])
    if controls['status'] != 'NATIVE_POSITIVE_AND_SAVED_INPUT_CONTROLS_PASS':
        raise ValueError('Native repair controls incomplete')
    adopted = []
    for a in attrs['records']:
        key = digest(a['parent_id'])[:20]
        pilot.put('attribution_units/'+key+'.json', {'record': a, 'adopted_from': str(root/'attribution.json')})
        source = root/'generated'/(a['parent_id']+'.json')
        g = read_json(source) if source.exists() else None
        if a['parent_id'] == diagnosis['parent_id']:
            if (controls['corrected_parent']['classification'] != 'VERIFIED_NO_REPLACEMENT'
                    or diagnosis['source_request_sha256'] != digest(a['generation_request'])
                    or diagnosis['result']['status'] != 'NO_NATIVE_REPLACEMENT'):
                raise ValueError('Legal empty result is not bound to saved request')
            g = diagnosis['result']; source = Path(b['diagnosis'])
        if g is None or g.get('status') not in TERMINALS:
            continue
        budgeted_timeout = (g.get('status') == 'TIMEOUT_BUDGETED' and g.get('scope_sha') == scope
                            and g.get('retained_raw') == [] and g.get('partial_adopted') is False
                            and 900 <= g.get('parent_wall_seconds', 0) < 960)
        if (g.get('parent_id') != a['parent_id'] or (not budgeted_timeout and
                (g.get('science_hash') != scope or g.get('seed') != parent_seed(scope, a['parent_id'])))):
            raise ValueError('Saved generation namespace / seed / parent differs')
        pilot.put('generated/'+key+'.json', {**{k:v for k,v in g.items() if k != 'scope_sha'},
            'adoption': {'source': str(source), 'source_record_sha256': digest(g),
                         'request_sha256': digest(a['generation_request']), 'generation_repeated': False,
                         'timeout_seed_not_serialized': budgeted_timeout,
                         'invocation_scope_sha256': scope}})
        adopted.append({'parent_id': a['parent_id'], 'status': g['status'], 'source': str(source)})
    pilot.put('legacy_adoption_receipt.json', {'adopted': adopted, 'generation_scope_sha256': scope,
        'remaining_pilot_ids': [p['parent_id'] for p in selected if p['parent_id'] not in {r['parent_id'] for r in adopted}],
        'test_read': False, 'old_attempt_modified': False})
    return selected
