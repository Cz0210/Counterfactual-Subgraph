import pytest
from src.baselines.cm_crem_dataset_full import partition_remaining, pilot_scope
from src.baselines.cm_crem_generation import parent_seed

def rows(n): return [{'parent_id': str(i), 'smiles': 'CC', 'split': 'train'} for i in range(n)]

def test_pilot_adoption_exact_and_no_duplicate_shards():
    r = rows(70); pilot = r[::2][:32]
    assigned = [partition_remaining(r, pilot, i, 4) for i in range(4)]
    ids = [p['parent_id'] for chunk in assigned for p in chunk]
    assert len(ids) == len(set(ids)) == 38
    assert set(ids).isdisjoint(p['parent_id'] for p in pilot)
    assert set(ids) | {p['parent_id'] for p in pilot} == {p['parent_id'] for p in r}

def test_pilot_changed_parent_rejected():
    r = rows(40); pilot = [dict(p) for p in r[:32]]; pilot[0]['smiles'] = 'CCC'
    with pytest.raises(ValueError, match='exact unchanged'): partition_remaining(r, pilot, 0, 2)

def test_no_rng_reset_from_deployment_paths():
    p = dict(seed=7, dataset='tastemolnet', output_root='/old', execution_commit='old')
    q = dict(p, output_root='/new', execution_commit='new')
    assert parent_seed(pilot_scope(p), 'same') == parent_seed(pilot_scope(q), 'same')

def test_duplicate_or_incomplete_pilot_rejected():
    for pilot in [rows(31), rows(31)+[rows(1)[0]]]:
        with pytest.raises(ValueError): partition_remaining(rows(40), pilot, 0, 1)
