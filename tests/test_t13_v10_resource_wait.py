from datetime import datetime, timezone
import pytest
from src.utils.t13_gap_recovery_owner import resource_wait_seconds

def test_finite_wait_capped_at_cutoff():
    spec=dict(resource_wait_seconds=86400,science_dispatch_cutoff_utc='2026-09-24T13:59:59+00:00')
    now=datetime(2026,9,24,13,0,tzinfo=timezone.utc)
    assert resource_wait_seconds(spec,now)==3599

def test_old_immediate_owner_unchanged():
    assert resource_wait_seconds({})==0

@pytest.mark.parametrize('value',[-1,86401,True,1.5])
def test_invalid_wait_rejected(value):
    with pytest.raises(ValueError):resource_wait_seconds(dict(resource_wait_seconds=value))

def test_cutoff_cannot_dispatch():
    with pytest.raises(ValueError,match='CUTOFF'):
        resource_wait_seconds(dict(resource_wait_seconds=60,science_dispatch_cutoff_utc='2026-09-24T13:59:59+00:00'),
                              datetime(2026,9,24,14,tzinfo=timezone.utc))
