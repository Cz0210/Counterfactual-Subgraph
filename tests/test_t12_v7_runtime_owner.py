from pathlib import Path
from src.utils import t12_v7_runtime_owner as v

def test_runtime_identity_and_fd_only_after_real_lock():
    text=Path(v.__file__).read_text()
    assert text.index('fcntl.flock(held.fileno()')<text.index("binding['owner_identity']=")
    assert 'pass_fds=(held.fileno(),)' in text
    assert "T12_OWNER_HELD_GPU_FD=str(held.fileno())" in text
    assert "'previous_stage'" not in text

def test_original_contract_and_finite_budget_preserved():
    text=Path(v.__file__).read_text()
    assert "spec=copy.deepcopy(old)" in text
    assert 'PREVIOUS_STAGE_BUDGET_REQUIRES_RECONCILIATION' in text
    assert "diagnostic_checkpoint_promotable=False" in text
    assert "source_template=template" in text
    assert 'run_t12_generation_segment' not in text
