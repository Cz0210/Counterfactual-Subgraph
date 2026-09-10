import gzip
import hashlib
import pytest
from src.baselines.cm_crem_database_prepare import unpack_verified

def sample(tmp_path):
    raw = b'SQLite format 3\0' + b'FIXTURE-NOT-DATABASE'*1000
    data = gzip.compress(raw)
    path = tmp_path/'fixture.gz'; path.write_bytes(data)
    expected = dict(compressed_bytes=len(data), published_md5=hashlib.md5(data).hexdigest(),
                    compressed_sha256=hashlib.sha256(data).hexdigest())
    return path, expected, raw

def test_single_stream_integrity_and_readonly(tmp_path):
    source, expected, raw = sample(tmp_path)
    target = tmp_path/'output.db'
    result = unpack_verified(source, target, expected, len(raw))
    assert source.exists() and target.read_bytes() == raw
    assert result['uncompressed_bytes'] == len(raw)
    assert result['uncompressed_sha256'] == hashlib.sha256(raw).hexdigest()
    assert target.stat().st_mode & 0o777 == 0o444

@pytest.mark.parametrize('field,value', [('published_md5','0'*32), ('compressed_sha256','0'*64), ('compressed_bytes',1)])
def test_identity_conflict_fails(tmp_path, field, value):
    source, expected, raw = sample(tmp_path); expected[field] = value
    with pytest.raises(ValueError, match='CONFLICT'):
        unpack_verified(source, tmp_path/'output.db', expected, len(raw))

def test_bound_no_silent_truncate(tmp_path):
    source, expected, raw = sample(tmp_path)
    with pytest.raises(ValueError, match='CAPACITY'):
        unpack_verified(source, tmp_path/'output.db', expected, 16)

def test_gzip_corruption_rejected(tmp_path):
    source, expected, raw = sample(tmp_path)
    data = bytearray(source.read_bytes()); data[-8] ^= 1; source.write_bytes(data)
    with pytest.raises(gzip.BadGzipFile):
        unpack_verified(source, tmp_path/'output.db', expected, len(raw))

def test_header_rejected(tmp_path):
    source = tmp_path/'bad.gz'; source.write_bytes(gzip.compress(b'HTTP403'))
    with pytest.raises(ValueError, match='HEADER'):
        unpack_verified(source, tmp_path/'output.db', {}, 100)
