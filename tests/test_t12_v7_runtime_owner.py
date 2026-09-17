from pathlib import Path
from src.utils import t12_v7_runtime_owner as v

def test_runtime_identity_and_fd_only_after_real_lock():
    text=Path(v.__file__).read_text()
    assert text.index('fcntl.flock(held,')<text.index("binding['owner_identity']=")
    assert 'pass_fds=(held.fileno(),)' in text
    assert "T12_OWNER_HELD_GPU_FD=str(held.fileno())" in text
    assert "'previous_stage'" not in text

def test_original_contract_and_finite_budget_preserved():
    text=Path(v.__file__).read_text()
    assert "spec=copy.deepcopy(old)" in text
    assert 'PREVIOUS_STAGE_BUDGET_REQUIRES_RECONCILIATION' in text
    assert "diagnostic_checkpoint_promotable=False" in text
    assert "source_template=str(template)" in text
    assert 'run_t12_generation_segment' not in text

def test_cli_string_template_is_converted_before_existing_loader(tmp_path,monkeypatch):
    import pytest
    def loader(path):
        assert isinstance(path,Path)
        raise RuntimeError('reached typed loader')
    monkeypatch.setattr(v,'load_spec',loader)
    with pytest.raises(RuntimeError,match='reached typed loader'):
        v.owner(template=str(tmp_path/'spec.json'),root=tmp_path/'out',registry='unused',code_root=tmp_path)

def test_posix_inherited_descriptor_and_exclusion(tmp_path):
    import fcntl,os,subprocess,sys
    path=tmp_path/'existing-lease'
    with path.open('a+b') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code='import os,sys;fd=int(sys.argv[1]);s=os.fstat(fd);t=os.stat(sys.argv[2]);assert (s.st_dev,s.st_ino)==(t.st_dev,t.st_ino);assert os.getppid()==int(sys.argv[3])'
        subprocess.run([sys.executable,'-I','-c',code,str(held.fileno()),str(path),str(os.getpid())],pass_fds=(held.fileno(),),check=True)
        code='import fcntl,sys;f=open(sys.argv[1],"rb");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(73)'
        assert subprocess.run([sys.executable,'-I','-c',code,str(path)]).returncode==73
