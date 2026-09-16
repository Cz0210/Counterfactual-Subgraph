import pytest
from src.eval.cm4_v6_release import verify_raw
from src.eval.taste_final_v5_cm import write_csv
from src.baselines.cm_crem_runtime import atomic_json
from test_cm4_v6_reselect import fixture

def test_release_reduces_actual_columns_and_rejects_changed_block(tmp_path,monkeypatch):
    x=fixture(tmp_path);monkeypatch.setattr(x,'admission',lambda:x.root.mkdir(exist_ok=True))
    x.select();x.evaluate()
    atomic_json(x.root/'audit/final_audit.json',dict(status='CM4_V6_ACCEPTED',contract_sha256=x.sha))
    write_csv(x.root/'source_csv/prefix_metrics.csv',x.result().prefix_metrics())
    p,ids,mask,d,f,a=verify_raw(x.root)
    assert len(p)==20 and len(ids)==4 and int(p[-1]['covered_count'])==3
    path=x.root/'test/block-0000.npz'
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):verify_raw(x.root)
