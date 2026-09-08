import ast
import json
from pathlib import Path
import pytest
from src.experiments import bace_eval_migration as m
from src.experiments.bace_gin_fixed_pool import prefix_metrics
from src.experiments.bace_gin_reach_selector import select
from src.eval.bace_reach_selector import ReachMasks


def test_new_scope_never_renames_old_experiments():
    assert '2659' in m.SCOPES['gnn_a']
    assert 'FIXED_OUTPUT' in m.SCOPES['llm_gin']
    assert 'END_TO_END' not in str(m.SCOPES)


def test_at_most_k_and_infinite_failures_not_ecdf_capped():
    rows=[dict(parent_id='p',candidate_id=str(i),pred_before=1,pred_after=0,
        pair_strict_flip=True,wnode_distance=.02) for i in range(15)]
    report=prefix_metrics(['p'],[str(i) for i in range(15)],rows,theta=.01,cap=.03,endpoints=[.01,.03])
    selected=[r for r in report['prefix_rows'] if r['cohort']=='fixed141']
    assert all(x['K_effective']==15 for x in selected[14:])
    assert selected[9]['coverage']==0 and selected[9]['strict_flip_availability']==1
    rows=[dict(x,pair_strict_flip=False) for x in rows]
    report=prefix_metrics(['p'],[str(i) for i in range(15)],rows,theta=.01,cap=.03,endpoints=[.01,.03])
    assert all(x['coverage']==0 for x in report['exact_ecdf'])
    assert all(x['best_valid_distance'] is None for x in report['parent_distances'])


def test_new_oracle_flip_must_be_real():
    with pytest.raises(ValueError,match='STRICT_FLIP'):
        prefix_metrics(['p'],['c'],[dict(parent_id='p',candidate_id='c',pred_before=0,
            pred_after=0,pair_strict_flip=True,wnode_distance=.0)],theta=.01,cap=.03,endpoints=[.03])


def test_freeze_required_not_boolean(tmp_path):
    spec=dict(output_root=str(tmp_path),roles={'L0':{}},selector_policy=m.POLICY)
    with pytest.raises(FileNotFoundError):m.require_freeze(spec)
    m.seal(tmp_path/'CALIBRATION_FREEZE.json',dict(spec_sha256='fake',test_loaded=False,
        policy=m.POLICY,selectors={'L0':{}}))
    with pytest.raises(ValueError,match='FREEZE'):m.require_freeze(spec)


def test_parent_checkpoint_cannot_be_rebound(tmp_path):
    p=tmp_path/'p.json';m.seal(p,{'model':'gin','binding':'one'})
    with pytest.raises(ValueError,match='IMMUTABLE'):m.seal(p,{'model':'gine','binding':'two'})


def test_no_training_search_or_main_publisher_entrypoint():
    tree=ast.parse(Path(m.__file__).read_text())
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not calls & {'search_parent','train','fit_temperature','matrix_append','generate'}
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    source=ast.unparse(run)
    assert "if split == 'test':\n                freeze(spec)" in source


def test_cpu_slurm_matches_entrypoint_without_gpu_request():
    p=Path(__file__).resolve().parents[1]/'scripts/slurm/run_bace_eval_migration.sh'
    s=p.read_text();assert '--partition=intel' in s and '--gres=' not in s
    assert 'run_bace_eval_migration.py' in s and '--config configs/hpc.yaml' in s
    assert s.index('source ~/.bashrc')<s.index('set -eo pipefail')


def test_raw_cache_is_job_scoped_not_backbone_scoped():
    import inspect
    source=inspect.getsource(m.evaluate_role)
    assert "_distance(bundle,manifest,root/'raw_cache')" in source
    assert "root/role/'raw_cache'" not in source
    assert 'oracle_checkpoint_id=oracle.checkpoint_id' in source
