import pytest
from src.utils.main20_compact_resource import admission

def base(**kw):
    return admission(**dict(mode='SEALED_SMALL_RELEASE',available=9299,own_peak=128,
        other_peak=0,external_decline_15m=0,new_bytes=60*1024**2,io_pass=True)|kw)

def test_release(): assert base()['allowed']
def test_io_failed(): assert not base(io_pass=False)['allowed']
def test_unknown():
    with pytest.raises(ValueError): base(other_peak=None)
def test_external_growth(): assert not base(external_decline_15m=600)['allowed']
def test_budget(): assert not base(own_peak=129)['allowed']
def test_nvme(): assert not base(nvme_free=3*1024**3,nvme_uncreated_peak=2*1024**3)['allowed']
