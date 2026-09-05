import importlib.util
import json
from pathlib import Path
import pytest
from src.eval.bace_frozen_gnn_contracts import sha256_file

script=Path(__file__).resolve().parents[2]/'scripts/ablations/gnn/publish_corrected_seed7.py'
spec=importlib.util.spec_from_file_location('corrected_pub',script)
pub=importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)


def fixture(tmp_path):
    root=tmp_path/'import'; (root/'publication').mkdir(parents=True)
    audit={'state':'GNN_CORE_SEED7_CORRECTED_PASS'}
    for name in pub.TABLES:
        (root/'publication'/name).write_text(json.dumps(audit) if name.endswith('audit.json') else 'real,value\n')
    files={'publication/'+n:{'sha256':sha256_file(root/'publication'/n)} for n in pub.TABLES}
    (root/'package_manifest.json').write_text(json.dumps({'files':files}))
    acceptance={**audit,'main_matrix_write':False,'raw_ot_recomputed_count':0,
        'all_weights_unchanged':True,'gine_unchanged':True,'selectors_frozen_before_test':True,
        'package_sha256':'a'*64,'corrective_audit_sha256':files['publication/gnn_seed7_corrective_audit.json']['sha256']}
    (root/'corrective_location_overlay.json').write_text(json.dumps(acceptance))
    return root


def test_accepted_paper_publication_is_small_and_idempotent(tmp_path):
    root=fixture(tmp_path)
    before={p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    args=(root,sha256_file(root/'corrective_location_overlay.json'),tmp_path/'paper',tmp_path/'registry')
    assert pub.publish(*args)['state']=='GNN_CORE_SEED7_CORRECTED_PASS'
    assert pub.publish(*args)['archive_or_model_rehashed'] is False
    assert before=={p:p.read_bytes() for p in before}


def test_unaccepted_or_changed_table_not_published(tmp_path):
    root=fixture(tmp_path); (root/'publication'/pub.TABLES[0]).write_text('changed')
    with pytest.raises(ValueError,match='PAPER_TABLE_CHANGED'):
        pub.publish(root,sha256_file(root/'corrective_location_overlay.json'),tmp_path/'paper',tmp_path/'reg')


def test_main_control_registry_refused(tmp_path):
    root=fixture(tmp_path)
    with pytest.raises(ValueError,match='CANNOT_WRITE_MAIN_CONTROL'):
        pub.publish(root,sha256_file(root/'corrective_location_overlay.json'),tmp_path/'paper',tmp_path/'control/registry')
