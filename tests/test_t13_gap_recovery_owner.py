from src.utils.t13_gap_recovery_owner import decision,SCHEMA
def test_exact_gpu1_and_resources():
    s=dict(schema=SCHEMA,gpu_index=1)
    e=dict(memory_safe=True,storage_safe=True,physical_gpu_safe=True)
    assert decision(s,e)['allowed']
    for key in e:
        assert not decision(s,dict(e,**{key:False}))['allowed']
    assert not decision(dict(s,gpu_index=0),e)['allowed']
    assert not decision(s,dict(e,source_blockers=['LIVE_T13_PREDECESSOR']))['allowed']
