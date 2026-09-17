from datetime import datetime,timezone
from pathlib import Path
import pytest
from src.utils import t13_v7_binding as v

def test_v7_deadline_and_no_budget_reset():
    assert datetime.fromisoformat(v.DEADLINE)>datetime(2026,9,24,13,tzinfo=timezone.utc)
    assert (datetime.fromisoformat(v.DEADLINE)-datetime.fromisoformat(v.CUTOFF)).total_seconds()==7200

def test_seal_fresh_and_bound(tmp_path):
    d=v.seal(tmp_path/'x.json',{'external_future_reserved_ram_bytes':0,'basis':'USER_AUTHORIZATION'})
    assert v.bound_json(d)['basis']=='USER_AUTHORIZATION'
    with pytest.raises(ValueError,match='FRESH'):v.seal(tmp_path/'x.json',{'x':2})

def test_old_results_and_policy_are_not_relabelled():
    source=Path(v.__file__).read_text()
    assert 'EXPLICIT_USER_V7_REPLACEMENT_NOT_A_KERNEL_MEASUREMENT' in source
    assert 'pair_chunk_entries=2*(468+468)' in source
    assert "cache_metadata_temp_entries='UNKNOWN'" in source
    assert 'run(root/\'dispatch.json\')' in source
    assert 'max_full_starts=1,new_fresh_start=False' in source
