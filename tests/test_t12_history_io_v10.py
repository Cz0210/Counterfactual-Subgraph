import hashlib,json,random
from pathlib import Path
import pytest
from src.utils.t12_history_io_v10 import history_io_overlay


def setup(tmp_path):
    root=tmp_path/'out';(root/'bridge_history').mkdir(parents=True)
    source=root/'bridge_history/history-x.bin';source.write_bytes(b'abc123'*30)
    cache=tmp_path/'cache';cache.write_bytes(source.read_bytes())
    policy=dict(authorization_id='T12_SINGLE_CONTENT_EQUIVALENT_IO_RELOCATION_V10_20260918',buffer_bytes=1024*1024,model_fixture_calls=0,segments=[dict(relative_path='bridge_history/history-x.bin',cache_path=str(cache),bytes=cache.stat().st_size,sha256=hashlib.sha256(cache.read_bytes()).hexdigest())])
    p=tmp_path/'policy.json';p.write_text(json.dumps(policy))
    return root,source,cache,dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def test_exact_bytes_order_rng_and_no_active_edit(tmp_path):
    root,source,cache,b=setup(tmp_path);old=Path.open;rng=random.getstate()
    with history_io_overlay(b,root):
        with source.open('rb',buffering=0) as f:rows=[f.read(6) for _ in range(30)]
        assert rows==[b'abc123']*30
    assert Path.open is old and random.getstate()==rng
    assert source.read_bytes()==cache.read_bytes()
    receipt=json.loads((root/'actual_history_io_v10.json').read_text())
    assert receipt['opens'][0]['actual_path']==str(cache)


def test_changed_cache_rejected(tmp_path):
    root,source,cache,b=setup(tmp_path);cache.write_bytes(b'123abc'*30)
    with history_io_overlay(b,root):
        with pytest.raises(ValueError,match='CACHE_CHANGED'):source.open('rb')


def test_unlisted_and_writes_not_redirected(tmp_path):
    root,source,cache,b=setup(tmp_path)
    p=tmp_path/'unrelated';p.write_bytes(b'outside')
    with history_io_overlay(b,root):
        assert p.read_bytes()==b'outside'
        p.write_bytes(b'new')
    assert p.read_bytes()==b'new'

def test_repeated_lookup_does_not_grow_or_rewrite_receipt(tmp_path,monkeypatch):
    from src.utils import t12_history_io_v10 as module
    root,source,cache,b=setup(tmp_path);calls=[];original=module.atomic_json
    def record(path,value):
        calls.append(path);return original(path,value)
    monkeypatch.setattr(module,'atomic_json',record)
    with history_io_overlay(b,root):
        for _ in range(100):
            with source.open('rb',buffering=0) as stream:assert stream.read(6)==b'abc123'
    assert len(calls)==1
    assert len(json.loads((root/'actual_history_io_v10.json').read_text())['opens'])==1
