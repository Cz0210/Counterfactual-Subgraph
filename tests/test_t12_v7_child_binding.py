import hashlib,json
from pathlib import Path
import pytest
from scripts.run_t12_v7_child import reviewed_view

def test_wrapper_and_science_commits_are_separate_and_content_checked(tmp_path):
    files={}
    for n in range(4):
        p=tmp_path/f'f{n}.py';p.write_text(str(n));files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(audited_differences=[dict(path=k,current_sha256=v) for k,v in files.items()],
                                      current_commit='scientific_commit',current_tree='scientific_tree')))
    sha=hashlib.sha256(receipt.read_bytes()).hexdigest()
    template=tmp_path/'template.json';template.write_text(json.dumps(dict(entrypoint='old_cli',science_contract=dict(scientific_source_equivalence_file_sha256=sha))))
    base=tmp_path/'base.json';base.write_text(json.dumps(dict(base_root=str(tmp_path),reviewed_files=files)))
    b=dict(template=str(template),template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
           source_base_binding=str(base),source_base_binding_sha256=hashlib.sha256(base.read_bytes()).hexdigest())
    spec=dict(execution_commit='wrapper_commit',repo_root='wrapper_root',science_contract=dict(v7_wrapper_binding=b,
        scientific_source_equivalence_receipt=str(receipt),scientific_source_equivalence_file_sha256=sha))
    view,_=reviewed_view(spec)
    assert view['execution_commit']=='scientific_commit' and spec['execution_commit']=='wrapper_commit'
    (tmp_path/'f0.py').write_text('changed')
    with pytest.raises(ValueError,match='SCIENTIFIC_BYTES_CHANGED'):reviewed_view(spec)
