"""No-network/no-science tests for this campaign's finite post-freeze relay."""
import hashlib
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.experiments import bace_gin_continuation as c


@pytest.fixture(autouse=True)
def local_renderer_fixture(tmp_path,monkeypatch):
    renderer=tmp_path/'renderer.py';renderer.write_text('# offline renderer fixture\n')
    monkeypatch.setattr(c,'LOCAL_RENDERER',str(renderer))


def plan(method='gcfexplainer'):
    return dict(schema='bace_gin_native_finite_continuation_v1', method=method,
        freeze_job_id='2596000', attempt_id='fixture-attempt', experiment_spec_sha256=c.SPEC_SHA,
        hpc_execution_commit=c.EXECUTION_COMMIT, main_matrix_write=False, repeat_calibration=False,
        repeat_freeze=False, native_source_binding_sha256='b'*64,
        continuation_worker_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(c.LOCAL_RENDERER).read_bytes()).hexdigest(),
        expected_calibration_entry=dict(path=c.CAMPAIGN+'/inputs/cal.json',sha256='a'*64),
        source_execution_binding=dict(spec_execution_commit='3f'*20,actual_driver_commit=c.EXECUTION_COMMIT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fake:
    def __init__(self, tmp, method='gcfexplainer'):
        self.tmp, self.method = tmp, method
        self.calls, self.submissions, self.job_receipts = [], [], {}
        self.files = {}
        self.freeze_job = dict(state='COMPLETED',exit_code='0:0')
        self.terminal = dict(state='COMPLETED',returncode=0,plan_sha=c.stable(plan(method)),freeze_sha='freeze')

    def putfile(self, remote, obj):
        data=json.dumps(obj).encode();self.files[remote]=data
        return dict(exists=True,value=obj,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))

    def rpc(self, host, **payload):
        self.calls.append((host,payload))
        op=payload['op']
        if op=='job': return self.freeze_job
        if op=='read':
            data=self.files.get(payload['path'])
            return dict(exists=False) if data is None else dict(exists=True,value=json.loads(data),sha256=hashlib.sha256(data).hexdigest())
        if op in ('mkdir','finalize','sidecar'): return dict(state='READY')
        if op=='launch':return dict(pid=12,start_ticks=123,plan_sha=payload['plan_sha'])
        if op=='alive':return dict(alive=True)
        if op=='index':return dict(sha256='index',bytes=10,raw_cost_count=2)
        if op=='result_manifest':
            return {r:dict(source_path=s,sha256=hashlib.sha256(self.files[s]).hexdigest(),bytes=len(self.files[s])) for r,s in payload['files'].items()}
        if op=='submit':
            key=payload['receipt']
            if key not in self.job_receipts:
                self.submissions.append(payload['argv'])
                self.job_receipts[key]=dict(job_id=str(2597000+len(self.submissions)),argv=payload['argv'])
            return self.job_receipts[key]
        raise AssertionError(op)

    def copy(self, source, dest):
        self.calls.append(('copy',dict(source=source,destination=dest)))
        if source.startswith('tongji-hpc:'):
            Path(dest).write_bytes(self.files[source.split(':',1)[1]])

    def run(self, argv):
        self.calls.append(('local-render',dict(argv=argv)))
        target=Path(argv[argv.index('--output')+1]);target.mkdir(parents=True,exist_ok=True)
        (target/'table2_bace_gin_fixed141.pdf').write_bytes(b'fixture pdf')
        (target/'table2_bace_gin_fixed141.tex').write_bytes(b'fixture tex')
        return ''


@pytest.fixture
def relay(tmp_path, monkeypatch):
    monkeypatch.setattr(c,'LOCAL',str(tmp_path/'local'))
    p=plan();fake=Fake(tmp_path);r=c.Continuation(p,fake)
    r.root.mkdir(parents=True)
    return r,fake


@pytest.mark.parametrize('field,value', [('method','ours'),('method','globalgce'),('freeze_job_id','TODO'),
    ('main_matrix_write',True),('repeat_calibration',True),('repeat_freeze',True),
    ('hpc_execution_commit','a'*40),('experiment_spec_sha256','x'*64)])
def test_narrow_contract_rejects_expansion(field,value):
    p=plan();p[field]=value
    with pytest.raises(ValueError):c.validate_plan(p)


def test_exact_cpu_test_migration_arguments():
    p=plan();args=c.migration_argv(p,'f'*64)
    assert args[:3]==[c.AUTODL_PYTHON,'-I','-B']
    assert args[args.index('--split')+1]=='test'
    assert args[args.index('--driver-commit')+1]==c.MIGRATION_COMMIT
    assert args[args.index('--base-commit')+1]==c.BASE_COMMIT
    assert args[args.index('--test-freeze-sha')+1]=='f'*64
    assert '--experiment-spec' in args and '--experiment-spec-sha' in args
    assert not any('generation' in x or 'calibration_index' in x for x in args)


def test_chain_only_test_aggregate_audit_export_no_calibration():
    previous='2596000'
    for stage in ('test','aggregate','audit','export'):
        args=c.job_argv(plan(),stage,previous)
        assert '--dependency=afterok:'+previous in args
        assert '--partition=intel' in args and not any('gres=' in x for x in args)
        assert '--chdir='+c.EXECUTION in args
        assert c.SPEC in args and 'verify-calibration' not in args and 'freeze' not in args
        previous=str(int(previous)+1)
    with pytest.raises(ValueError):c.job_argv(plan(),'freeze','2596000')


def test_running_freeze_and_permanent_dependency(relay):
    r,f=relay;f.freeze_job=dict(state='RUNNING',exit_code='0:0')
    assert r.advance() is False and r.state['phase']=='WAIT_FREEZE'
    assert not f.submissions
    f.freeze_job=dict(state='PENDING',reason='DependencyNeverSatisfied',exit_code='0:0')
    with pytest.raises(ValueError,match='PERMANENTLY'):r.advance()


def test_exit_zero_without_actual_freeze_rejected(relay):
    r,f=relay
    with pytest.raises(ValueError,match='ACTUAL_FREEZE'):r.advance()
    assert not f.submissions


def test_new_freeze_is_verified_before_any_test_transfer(relay,monkeypatch):
    r,f=relay
    spec={'pools':{'gcfexplainer':{'sha256':'p'*64}},'execution_commit':'3f'*20}
    f.putfile(c.SPEC,spec)
    f.putfile(r.paths['freeze'],dict(state='FROZEN',test_loaded=False))
    f.putfile(r.paths['sidecar'],dict(spec_sha256=c.SPEC_SHA,pool_manifest_sha256='p'*64,
        calibration=r.plan['expected_calibration_entry']))
    seen=[]
    monkeypatch.setattr(c,'validate_freeze',lambda *a:seen.append('validated'))
    assert r.advance()
    assert seen==['validated'] and r.state['phase']=='TRANSFER_BINDINGS'
    assert sha(r.root/'selection_freeze.json')==r.state['freeze_sha']
    assert not any(host=='autodl-a800' for host,_ in f.calls)


def test_no_duplicate_calibration_or_foreign_test_pointer(relay,monkeypatch):
    r,f=relay
    f.putfile(c.SPEC,{'pools':{'gcfexplainer':{'sha256':'p'*64}},'execution_commit':'3f'*20})
    f.putfile(r.paths['freeze'],{})
    f.putfile(r.paths['sidecar'],dict(spec_sha256=c.SPEC_SHA,pool_manifest_sha256='p'*64,
        calibration=r.plan['expected_calibration_entry'],test={'some_existing':'entry'}))
    monkeypatch.setattr(c,'validate_freeze',lambda *a:None)
    with pytest.raises(ValueError,match='TEST_ALREADY_BOUND'):r.advance()


def test_real_migration_failure_never_submits_test(relay):
    r,f=relay;r.state['phase']='WAIT_INDEX'
    f.putfile(r.paths['remote']+'/terminal.json',dict(state='FAILED',returncode=1))
    with pytest.raises(ValueError,match='MIGRATION_FAILED'):r.advance()
    assert f.submissions==[]


def test_resource_wait_does_not_claim_science(relay):
    r,f=relay;r.state.update(phase='WAIT_INDEX',migration_owner={'pid':12,'start_ticks':123})
    f.putfile(r.paths['remote']+'/heartbeat.json',dict(state='WAITING_RESOURCE',worker_pid=None))
    assert r.advance() is False
    assert r.state['migration_heartbeat']['worker_pid'] is None and not f.submissions


def test_sidecar_update_carries_existing_calibration_and_history(relay):
    r,f=relay;r.state.update(phase='INSTALL_SIDECAR',migrated_index={'sha256':'f'*64},freeze_sha='c'*64,
        sidecar_sha='d'*64,spec={'pools':{'gcfexplainer':{'sha256':'p'*64}}})
    r.advance();request=next(p for h,p in f.calls if h=='tongji-hpc' and p.get('op')=='sidecar')
    assert request['calibration']==r.plan['expected_calibration_entry']
    assert request['entry']['new_test_freeze_path']==r.paths['freeze']
    assert request['prior_receipt'].startswith(r.paths['hpc'])
    assert r.state['phase']=='SUBMIT_CHAIN'


def test_resume_mid_submission_adopts_receipt_never_resubmits(relay):
    r,f=relay;r.state['phase']='SUBMIT_CHAIN';r.advance()
    assert len(f.submissions)==4 and r.state['phase']=='WAIT_CHAIN'
    jobs=dict(r.state['jobs'])
    r.state.update(phase='SUBMIT_CHAIN',jobs={})
    r.advance()
    assert len(f.submissions)==4 and r.state['jobs']==jobs
    for i,args in enumerate(f.submissions):
        dep='2596000' if i==0 else str(2597000+i)
        assert '--dependency=afterok:'+dep in args


def test_completed_jobs_alone_do_not_claim_audit_pass(relay):
    r,f=relay;r.state.update(phase='WAIT_CHAIN',jobs={x:{'job_id':str(2597000+i)} for i,x in enumerate(('test','aggregate','audit','export'))},freeze_sha='x'*64)
    with pytest.raises(ValueError,match='ACTUAL_INDEPENDENT_AUDIT'):r.advance()


def test_remote_sidecar_real_atomic_install_preserves_original_bytes(tmp_path):
    original=dict(spec_sha256='s',pool_manifest_sha256='p',calibration={'path':'old','sha256':'c'},extra='preserve')
    side=tmp_path/'side.json';side.write_text(json.dumps(original,indent=2))
    old_bytes=side.read_bytes();index=tmp_path/'index.json';index.write_text('{}')
    freeze=tmp_path/'freeze.json';freeze.write_text('{}')
    entry=dict(path=str(index),sha256=sha(index),new_test_freeze_path=str(freeze),new_test_freeze_sha256=sha(freeze))
    payload=dict(op='sidecar',path=str(side),expected_sha256=sha(side),entry=entry,
        calibration=original['calibration'],spec_sha='s',pool_sha='p',prior_receipt=str(tmp_path/'prior.json'))
    result=subprocess.run([sys.executable,'-I','-B','-c',c.REMOTE,json.dumps(payload)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    after=json.loads(side.read_text());assert after==dict(original,test=entry)
    assert (tmp_path/'prior.json').read_bytes()==old_bytes
    again=subprocess.run([sys.executable,'-I','-B','-c',c.REMOTE,json.dumps(payload)],capture_output=True,text=True)
    assert again.returncode==0 and 'EXISTING_IDENTICAL' in again.stdout


def test_remote_submission_uncertainty_is_fail_closed(tmp_path):
    receipt=tmp_path/'submission.json';receipt.with_suffix('.intent.json').write_text('{}')
    payload=dict(op='submit',receipt=str(receipt),argv=['sbatch','DO_NOT_EXECUTE'],cwd='unused',commit='unused')
    result=subprocess.run([sys.executable,'-I','-B','-c',c.REMOTE,json.dumps(payload)],capture_output=True,text=True)
    assert result.returncode!=0 and 'OUTCOME_UNCERTAIN_DO_NOT_RESUBMIT' in result.stderr
    assert not receipt.exists()


def test_plan_change_cannot_reuse_checkpoint(relay):
    r,f=relay;r.checkpoint()
    changed=dict(r.plan,freeze_job_id='2596001')
    with pytest.raises(ValueError,match='PLAN_CHANGED'):c.Continuation(changed,f)


def test_resource_contract_is_real_cgroup_not_host_memory(monkeypatch):
    class CG:
        def __init__(self,p):self.p=p
        def __truediv__(self,n):return CG(self.p+'/'+n)
        def exists(self):return True
        def read_text(self):return str(500*1024**3 if 'limit' in self.p else 100*1024**3)
    class FS:
        f_bavail=4*1024**3;f_frsize=1;f_favail=30000
    monkeypatch.setattr(c,'Path',CG);monkeypatch.setattr(c.os,'statvfs',lambda _:FS())
    result=c.resource_snapshot('project')
    assert result['state']=='PASS' and result['required_headroom_bytes']==386*1024**3
    assert result['required_file_slots']==20600 and result['no_gpu']


def test_scoped_local_owner_lock_is_exclusive(relay):
    r,f=relay
    with (Path(c.LOCAL)/(r.plan['method']+'.relay.lock')).open('a') as other:
        fcntl.flock(other.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):r.run()
    assert not f.calls


def test_resource_wait_cannot_hide_changed_small_source_binding(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'NATIVE',str(tmp_path/'native'))
    p=plan();paths=c.locations(p);root=Path(paths['remote']);(root/'inputs').mkdir(parents=True)
    source=Path(paths['source']);source.parent.mkdir(parents=True);source.write_text('{}')
    p['native_source_binding_sha256']=sha(source)
    task=root/'inputs/worker_task.json';task.write_text(json.dumps(dict(plan=p,freeze_sha='f'*64)))
    readings=iter([dict(state='BLOCKED_RESOURCE'),dict(state='PASS')])
    monkeypatch.setattr(c,'resource_snapshot',lambda _:next(readings))
    monkeypatch.setattr(c.time,'sleep',lambda _:source.write_text('{"changed":true}'))
    monkeypatch.setattr(c.subprocess,'Popen',lambda *a,**k:pytest.fail('No child may start with changed binding'))
    with pytest.raises(ValueError,match='CHANGED_AFTER_RESOURCE_WAIT'):c.remote_index_worker(task)


def test_only_actual_independent_audit_completes_chain(relay):
    r,f=relay;r.state.update(phase='WAIT_CHAIN',jobs={x:{'job_id':str(2597000+i)} for i,x in enumerate(('test','aggregate','audit','export'))},freeze_sha='f'*64)
    f.putfile(r.paths['audit'],dict(state='RESULT_CONSISTENCY_PASS',method='gcfexplainer',
        spec_sha256=c.SPEC_SHA,selection_freeze_sha256='f'*64,main_matrix_write=False))
    assert r.advance() and r.state['phase']=='TRANSFER_RESULTS'
    assert r.state['main_matrix_write'] is False


def test_result_allowlist_excludes_science_inputs_and_runtime():
    files=c.result_files(plan())
    assert len(files)==12 and sum(k.startswith('source_csv/') for k in files)==7
    assert not any(x in path for path in files.values() for x in ('model.pt','checkpoint','applications.jsonl'))


def test_scoped_result_transfer_does_not_relay_whole_runtime(relay):
    r,f=relay;r.state['phase']='TRANSFER_RESULTS'
    for relative,source in c.result_files(r.plan).items():f.putfile(source,dict(result=relative))
    audit_source=c.result_files(r.plan)['provenance/independent_result_audit.json']
    r.state['audit']={'sha256':hashlib.sha256(f.files[audit_source]).hexdigest()}
    r.advance()
    assert r.state['phase']=='RENDER_RESULTS'
    for relative,source in c.result_files(r.plan).items():
        assert (r.root/'results'/relative).read_bytes()==f.files[source]
    assert len([1 for host,_ in f.calls if host=='copy'])==12
    assert not f.submissions
    r.advance()
    assert r.state['phase']=='COMPLETED' and r.state['autodl_result_publication'] is False
    result=json.loads((r.root/'results/delivery_manifest.json').read_text())
    assert result['state']=='LOCAL_PARTIAL_RESULTS_READY' and result['main_matrix_write'] is False
    assert result['independently_audited_method']=='gcfexplainer'
    r.state['phase']='RENDER_RESULTS';r.advance()
    assert len([1 for host,_ in f.calls if host=='local-render'])==1
