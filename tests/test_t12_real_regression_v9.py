import json
from pathlib import Path
import pytest
from src.utils.t12_real_regression_v9 import charge,write,read,verify

def test_budget_counts_before_calls_and_keeps_failures(tmp_path):
    p=tmp_path/'budget.json'
    charge(p,'gine',63)
    with pytest.raises(ValueError,match='EXHAUSTED'):charge(p,'gine',2)
    assert read(p)['gine']==63
    charge(p,'gine',1)
    with pytest.raises(ValueError,match='EXHAUSTED'):charge(p,'gine',1)
    assert read(p)['gine']==64

def test_budgets_separate_and_transitions_never_allowed(tmp_path):
    p=tmp_path/'budget.json';charge(p,'neurosed_pairs',64);charge(p,'gine',1)
    with pytest.raises(ValueError,match='INVALID'):charge(p,'transitions',1)
    assert read(p)['transitions']==0
    charge(p,'fixture_groups',16)
    with pytest.raises(ValueError,match='EXHAUSTED'):charge(p,'fixture_groups',1)

def test_verifier_reopens_and_rejects_changed_arrays(tmp_path):
    write(tmp_path/'off.json',{'changed':True})
    write(tmp_path/'producer.json',{'files':{'off.json':'0'*64}})
    with pytest.raises(ValueError,match='EVIDENCE_CHANGED'):verify(tmp_path)

def test_actual_producer_uses_existing_models_and_observer():
    text=(Path(__file__).parents[1]/'src/utils/t12_real_regression_v9.py').read_text()
    for value in ('TasteFrozenGINENativeAdapter','load_neurosed','BoundSelectedStepObserver',
                  'observer.installed()',"charge(budget, 'gine'", "charge(budget, 'neurosed_pairs'",
                  "with lease.open",'LOCK_EX | fcntl.LOCK_NB'):
        assert value in text
    assert 'move_to_next_graph(' not in text
    assert '_prepare_canary(' not in text
    assert 'reset_rng' not in text

def test_cpu_slurm_refuses_fake_cpu_comparison():
    text=(Path(__file__).parents[1]/'scripts/slurm/run_t12_real_adapter_regression.sh').read_text()
    assert '--gres' not in text and '--gpus' not in text
    assert 'exit 64' in text
