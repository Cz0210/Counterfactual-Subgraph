"""Finite Mac relay for this campaign's already-frozen native GIN methods.

No calibration, selector, generation, model training, GPU lease or main-matrix
operations. The only new science is the existing heldout evaluation chain.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone

HPC = '/share/home/u20526/czx'
HPC_RUNTIME = HPC + '/counterfactual-subgraph-hpc-runtime'
CAMPAIGN = HPC_RUNTIME + '/experiments/bace-gin-fixed-pool-v1-20260907/campaign-20260907T145923Z'
EVALUATION = CAMPAIGN + '/evaluation'
EXECUTION = HPC + '/worktrees/bace-gin-audit-d5ec293d'
EXECUTION_COMMIT = 'd5ec293d357e80f09c1f93eff861570ef8ee433d'
SPEC = CAMPAIGN + '/inputs/spec.json'
SPEC_SHA = '1cb7f568b0cc7e2186a6490bd788759594d2c97e8fe5b00b40c35ca1d6b71f72'
AUTODL = '/autodl-fs/data/counterfactual-subgraph-runtime'
NATIVE = AUTODL + '/outputs/autodl/experiments/bace-gin-fixed-pool-v1-20260907/native-raw-94394843-20260907T152400Z'
BASE = '/root/autodl-tmp/worktrees/bace-reach-v2-7e52ec41'
BASE_COMMIT = '7e52ec41cb471556d643fa4aa130477b88340bc8'
MIGRATION_COMMIT = 'c2164b940190a16d0639a0e07388d2c2815fc97f'
HPC_PYTHON = '/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python'
AUTODL_PYTHON = '/root/miniconda3/envs/smiles_pip118/bin/python'
LOCAL = '/private/tmp/bace-gin-direction-change-20260907/native-continuations'
LOCAL_PYTHON = '/Users/cz0210/miniconda3/envs/smiles_local/bin/python'
LOCAL_RENDERER = '/private/tmp/bace-gin-direction-change-20260907/partial-delivery/tools/replot_bace_gin.py'
METHODS = ('gcfexplainer', 'comrecgc')
CSV_NAMES = ('bace_gin_fixed141_table2.csv', 'bace_gin_fixed141_figure3.csv',
    'bace_gin_fixed141_figure4_exact_ecdf.csv', 'bace_gin_native_metrics.csv',
    'bace_gin_parent_predictions.csv', 'bace_gin_parent_best_distances.csv',
    'bace_gin_method_failure_funnel.csv')
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=2')


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def validate_plan(plan):
    require(plan.get('schema') == 'bace_gin_native_finite_continuation_v1', 'WRONG_CONTINUATION_SCHEMA')
    require(plan.get('method') in METHODS, 'ONLY_EXISTING_GCF_OR_COMREC_ALLOWED')
    require(str(plan.get('freeze_job_id', '')).isdigit(), 'REAL_FREEZE_JOB_REQUIRED')
    token = plan.get('attempt_id', '')
    require(8 <= len(token) <= 64 and all(c.isalnum() or c in '-_' for c in token), 'BAD_ATTEMPT_ID')
    require(plan.get('experiment_spec_sha256') == SPEC_SHA, 'ORIGINAL_CAMPAIGN_SPEC_REQUIRED')
    require(plan.get('hpc_execution_commit') == EXECUTION_COMMIT, 'ACTUAL_D5_EXECUTION_REQUIRED')
    require(plan.get('main_matrix_write') is False, 'NO_MAIN_MATRIX_AUTHORITY')
    require(plan.get('repeat_calibration') is False and plan.get('repeat_freeze') is False, 'NO_REPEAT_CALIBRATION_OR_FREEZE')
    require(len(plan.get('native_source_binding_sha256', '')) == 64, 'BOUND_NATIVE_SOURCE_REQUIRED')
    require(len(plan.get('continuation_worker_sha256', '')) == 64, 'IMMUTABLE_CONTINUATION_WORKER_REQUIRED')
    require(len(plan.get('renderer_sha256', '')) == 64, 'BOUND_EXISTING_OFFLINE_RENDERER_REQUIRED')
    cal = plan.get('expected_calibration_entry', {})
    require(set(cal) >= {'path', 'sha256'} and str(cal['path']).startswith(HPC_RUNTIME + '/')
            and '..' not in Path(cal['path']).parts and len(cal['sha256']) == 64, 'EXISTING_CALIBRATION_ENTRY_REQUIRED')
    execution=plan.get('source_execution_binding',{})
    require(isinstance(execution,dict) and execution.get('actual_driver_commit')==EXECUTION_COMMIT
            and len(execution.get('spec_execution_commit',''))==40,
            'SCIENCE_AND_ACTUAL_DRIVER_EXECUTION_BINDING_REQUIRED')
    return plan


def locations(plan):
    validate_plan(plan)
    method, token = plan['method'], plan['attempt_id']
    remote = NATIVE + '/' + method + '-test-continuation-' + token
    hpc = CAMPAIGN + '/continuations/' + method + '-' + token
    return dict(local=LOCAL + '/' + method + '-' + token, remote=remote, hpc=hpc,
        freeze=EVALUATION + '/' + method + '/selection_freeze.json',
        sidecar=EVALUATION + '/manifests/' + method + '_native_raw_adoption.json',
        source=NATIVE + '/inputs/' + method + '-raw-source.json',
        hpc_index=hpc + '/test_index.json', audit=EVALUATION + '/' + method + '/audits/' + token + '.json')


def validate_freeze(value, spec, plan):
    # Reuse the existing portable freeze contract; no historical test is opened.
    from src.experiments.bace_gin_native_raw import validate_portable_scheme_a_freeze
    require(stable(spec) == SPEC_SHA, 'COPIED_EXPERIMENT_SPEC_CHANGED')
    return validate_portable_scheme_a_freeze(value, experiment_spec=spec, method=plan['method'], expected_spec_sha256=SPEC_SHA)


def migration_argv(plan, freeze_sha):
    paths = locations(plan)
    return [AUTODL_PYTHON, '-I', '-B', NATIVE + '/overlay-c2164b94/scripts/experiments/migrate_bace_gin_native_raw.py',
        '--config', BASE + '/configs/hpc.yaml', '--binding', paths['source'], '--output', paths['remote'] + '/test_index.json',
        '--base-repo', BASE, '--base-commit', BASE_COMMIT, '--driver-commit', MIGRATION_COMMIT,
        '--split', 'test', '--experiment-spec', paths['remote'] + '/inputs/spec.json', '--experiment-spec-sha', SPEC_SHA,
        '--test-freeze', paths['remote'] + '/inputs/selection_freeze.json', '--test-freeze-sha', freeze_sha]


def job_argv(plan, phase, dependency):
    paths = locations(plan)
    require(phase in ('test', 'aggregate', 'audit', 'export') and str(dependency).isdigit(), 'FINITE_AFTEROK_STAGE_REQUIRED')
    resources = dict(test=('8', '32G', '12:00:00'), aggregate=('2', '8G', '01:00:00'),
                     audit=('2', '8G', '01:00:00'), export=('2', '8G', '00:30:00'))
    cpu, memory, wall = resources[phase]
    argv = ['sbatch', '--parsable', '--partition=intel', '--chdir=' + EXECUTION,
        '--cpus-per-task=' + cpu, '--mem=' + memory, '--time=' + wall,
        '--dependency=afterok:' + str(dependency), '--job-name=gin-' + plan['method'] + '-cont-' + phase,
        '--output=' + paths['hpc'] + '/%j-' + phase + '.out', '--error=' + paths['hpc'] + '/%j-' + phase + '.err']
    if phase == 'audit':
        return argv + ['scripts/slurm/audit_bace_gin_fixed_pool.sh', '--spec', SPEC, '--method', plan['method'], '--output', paths['audit']]
    argv += ['scripts/slurm/run_bace_gin_fixed_pool.sh', '--spec', SPEC, '--action',
             'evaluate-test' if phase == 'test' else phase]
    if phase != 'export':
        argv += ['--method', plan['method']]
    if phase == 'test':
        argv += ['--start', '0', '--stop', '141']
    return argv


def result_files(plan):
    """Exact export allowlist; never recursively relay runtime/model/checkpoints."""
    paths = locations(plan)
    files = {f'source_csv/{name}': EVALUATION+'/source_csv/'+name for name in CSV_NAMES}
    files.update({'provenance/independent_result_audit.json':paths['audit'],
        'provenance/spec.json':SPEC, 'provenance/selection_freeze.json':paths['freeze'],
        'provenance/method_metrics.json':EVALUATION+'/'+plan['method']+'/metrics.json',
        'provenance/experiment_registry.json':EVALUATION+'/experiment_registry.json'})
    return files


# A fixed set of metadata, transfer-finalize and submission operations. No shell
# interpolation, arbitrary command RPC, GPU operations or scientific computation.
REMOTE = r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.loads(sys.argv[1]); op=p['op']
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def save(path,d):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.partial')
 with tmp.open('w') as f:json.dump(d,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
if op=='read':
 path=Path(p['path'])
 if not path.exists(): out={'exists':False}
 else:
  if path.stat().st_size>p.get('limit',4*1024**2):raise ValueError('OVERSIZED_METADATA')
  b=path.read_bytes();out={'exists':True,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'value':json.loads(b)}
elif op=='mkdir':
 Path(p['path']).mkdir(parents=True,exist_ok=True);out={'state':'DIRECTORY_READY'}
elif op=='result_manifest':
 out={}
 for relative,source in p['files'].items():
  path=Path(source)
  if not path.is_file() or path.stat().st_size>128*1024**2:raise ValueError('MISSING_OR_OVERSIZED_SCOPED_RESULT:'+relative)
  out[relative]={'source_path':source,'sha256':sha(path),'bytes':path.stat().st_size}
elif op=='job':
 text=subprocess.check_output(['sacct','-X','-j',str(p['job']),'--format=JobIDRaw,State,ExitCode','-Pn'],text=True)
 rows=[x.split('|') for x in text.splitlines() if x.split('|')[0]==str(p['job'])]
 out={'state':rows[0][1].split()[0].rstrip('+'),'exit_code':rows[0][2]} if rows else {'state':'ACCOUNTING_NOT_YET_VISIBLE'}
 if out['state']=='PENDING':
  q=subprocess.check_output(['squeue','-h','-j',str(p['job']),'-o','%r'],text=True).strip()
  out['reason']=q
elif op=='alive':
 path=Path('/proc')/str(p['pid'])/'stat'
 if not path.exists():out={'alive':False}
 else:out={'alive':path.read_text().rsplit(')',1)[1].split()[19]==str(p['start_ticks'])}
elif op=='finalize':
 source,target=Path(p['source']),Path(p['target'])
 if target.exists():
  if sha(target)!=p['sha256']:raise ValueError('FRESH_TARGET_CONFLICT')
 elif sha(source)==p['sha256']:os.replace(source,target)
 else:raise ValueError('TRANSFER_HASH_MISMATCH')
 out={'state':'TRANSFER_VERIFIED','sha256':p['sha256']}
elif op=='index':
 path=Path(p['path']);d=json.loads(path.read_text())
 if d['schema']!='bace_reach_v2_raw_graph_cost_adoption_v1' or d['split']!='test' or d['new_test_freeze_sha256']!=p['freeze_sha']:raise ValueError('WRONG_TEST_INDEX')
 if d['ot_recomputed']!=0 or d['model_inference_performed'] or d['source_flip_masks_reused']:raise ValueError('NOT_RAW_ONLY_MIGRATION')
 if d['source_spec']['method_id']!=p['method']:raise ValueError('WRONG_NATIVE_METHOD_INDEX')
 body=dict(d);claimed=body.pop('self_sha256');digest=hashlib.sha256()
 for token in json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=True).iterencode(body):digest.update(token.encode())
 if claimed!=digest.hexdigest():raise ValueError('RAW_INDEX_SELF_BINDING_CHANGED')
 out={'sha256':sha(path),'bytes':path.stat().st_size,'state':d['state'],'raw_cost_count':d['raw_cost_count']}
elif op=='sidecar':
 path=Path(p['path']);old=json.loads(path.read_text());entry=p['entry']
 if old.get('test')==entry:out={'state':'EXISTING_IDENTICAL_TEST_ENTRY'}
 else:
  if sha(path)!=p['expected_sha256'] or old.get('test') is not None or old.get('calibration')!=p['calibration']:raise ValueError('SIDECAR_CAS_CONFLICT')
  if old.get('spec_sha256')!=p['spec_sha'] or old.get('pool_manifest_sha256')!=p['pool_sha']:raise ValueError('SIDECAR_WRONG_SOURCE')
  if sha(entry['path'])!=entry['sha256'] or sha(entry['new_test_freeze_path'])!=entry['new_test_freeze_sha256']:raise ValueError('SIDECAR_INPUT_CHANGED')
  prior=Path(p['prior_receipt'])
  if prior.exists():
   if sha(prior)!=p['expected_sha256']:raise ValueError('PRIOR_SIDECAR_RECEIPT_CONFLICT')
  else:
   with prior.open('xb') as f:f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
  old['test']=entry;save(path,old);out={'state':'TEST_ENTRY_ATOMICALLY_INSTALLED_CALIBRATION_PRESERVED'}
elif op=='launch':
 root=Path(p['root']);receipt=root/'launch.json';intent=root/'launch.intent.json'
 if receipt.exists():out=json.loads(receipt.read_text())
 elif intent.exists():raise ValueError('MIGRATION_LAUNCH_OUTCOME_UNCERTAIN_DO_NOT_DUPLICATE')
 else:
  if sha(p['worker'])!=p['worker_sha']:raise ValueError('RELAY_WORKER_TRANSFER_CHANGED')
  with intent.open('x') as f:json.dump({'plan_sha':p['plan_sha']},f);f.flush();os.fsync(f.fileno())
  with (root/'supervisor.out').open('ab') as outlog,(root/'supervisor.err').open('ab') as errlog:
   child=subprocess.Popen([sys.executable,'-I','-B',p['worker'],'--remote-index-worker',p['plan']],stdin=subprocess.DEVNULL,stdout=outlog,stderr=errlog,start_new_session=True,close_fds=True)
  stat=Path('/proc')/str(child.pid)/'stat'
  ticks=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else None
  out={'state':'CPU_INDEX_SUPERVISOR_LAUNCHED','pid':child.pid,'start_ticks':ticks,'plan_sha':p['plan_sha']};save(receipt,out)
elif op=='submit':
 path=Path(p['receipt']);intent=path.with_suffix('.intent.json')
 if path.exists():
  out=json.loads(path.read_text())
  if out['argv']!=p['argv']:raise ValueError('SUBMISSION_RECEIPT_COMMAND_CHANGED')
 elif intent.exists():raise ValueError('SLURM_SUBMISSION_OUTCOME_UNCERTAIN_DO_NOT_RESUBMIT')
 else:
  if subprocess.check_output(['git','rev-parse','HEAD'],cwd=p['cwd'],text=True).strip()!=p['commit']:raise ValueError('IMMUTABLE_DRIVER_CHANGED')
  path.parent.mkdir(parents=True,exist_ok=True)
  with intent.open('x') as f:json.dump({'argv':p['argv']},f);f.flush();os.fsync(f.fileno())
  job=subprocess.check_output(p['argv'],cwd=p['cwd'],text=True).strip().split(';')[0]
  if not job.isdigit():raise ValueError('INVALID_SBATCH_RECEIPT')
  out={'state':'SUBMITTED','job_id':job,'argv':p['argv'],'actual_execution_commit':p['commit']};save(path,out)
else:raise ValueError('UNSUPPORTED_FINITE_OPERATION')
print(json.dumps(out,sort_keys=True))
'''


class Transport:
    def run(self, argv):
        # SSH keepalives/rsync timeout end failed transports themselves. Never
        # use subprocess.run(timeout=...), which would SIGKILL a local child.
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError('TRANSPORT_FAILED:' + shlex.join(argv[:4]) + ':' + result.stderr[-2000:])
        return result.stdout

    def rpc(self, host, **payload):
        python = HPC_PYTHON if host == 'tongji-hpc' else AUTODL_PYTHON
        command = [python, '-I', '-B', '-c', REMOTE, json.dumps(payload, sort_keys=True)]
        text = self.run(['ssh', *SSH_OPTIONS, host, shlex.join(command)])
        return json.loads(text.strip().splitlines()[-1])

    def copy(self, source, destination):
        # Same supported rsync 2.6.9 options as the existing scoped GNN relay.
        self.run(['rsync', '-a', '--partial', '--timeout=300', '-e', shlex.join(['ssh', *SSH_OPTIONS]), source, destination])


def resource_snapshot(root):
    cg = Path('/sys/fs/cgroup/memory')
    if (cg / 'memory.limit_in_bytes').exists():
        limit, usage = [int((cg / name).read_text()) for name in ('memory.limit_in_bytes', 'memory.usage_in_bytes')]
    else:
        cg = Path('/sys/fs/cgroup')
        limit, usage = [int((cg / name).read_text()) for name in ('memory.max', 'memory.current')]
    fs = os.statvfs(root)
    required = 386 * 1024**3
    result = dict(sampled_at=now(), cgroup_limit_bytes=limit, cgroup_usage_bytes=usage,
        cgroup_headroom_bytes=limit-usage, other_main_task_headroom_reserve_bytes=384*1024**3,
        next_stage_peak_bytes=2*1024**3, required_headroom_bytes=required,
        available_bytes=fs.f_bavail*fs.f_frsize, required_bytes=2*1024**3,
        available_file_slots=fs.f_favail, required_file_slots=20600,
        file_slot_policy='20000 + 2*(268 active-stage bounded files + 32 own files)', no_gpu=True)
    result['state'] = 'PASS' if limit-usage >= required and result['available_bytes'] >= 2*1024**3 and fs.f_favail >= 20600 else 'BLOCKED_RESOURCE'
    return result


def remote_index_worker(plan_path):
    """One detached CPU supervisor. It exits after the single test-index child."""
    task = json.loads(Path(plan_path).read_text())
    plan = validate_plan(task['plan'])
    paths = locations(plan)
    root = Path(paths['remote'])
    require(Path(plan_path).resolve() == root / 'inputs/worker_task.json', 'UNEXPECTED_REMOTE_TASK_PATH')
    require(hashlib.sha256(Path(paths['source']).read_bytes()).hexdigest() == plan['native_source_binding_sha256'], 'NATIVE_SOURCE_BINDING_CHANGED')
    while True:
        admission = resource_snapshot(root)
        if admission['state'] == 'PASS':
            save(root/'resource_admission.json', admission)
            break
        save(root/'heartbeat.json', dict(state='WAITING_RESOURCE', owner_pid=os.getpid(), worker_pid=None,
            resource_sample=admission, heartbeat_at=now(), no_gpu=True, no_stage_started=True))
        time.sleep(60)
    # This small binding can outlive a long resource wait. Recheck immediately
    # before the immutable, independently gated migration child is launched.
    require(hashlib.sha256(Path(paths['source']).read_bytes()).hexdigest() == plan['native_source_binding_sha256'],
            'NATIVE_SOURCE_BINDING_CHANGED_AFTER_RESOURCE_WAIT')
    command = migration_argv(plan, task['freeze_sha'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    with (root/'stdout.log').open('xb') as stdout, (root/'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, close_fds=True, preexec_fn=lambda: os.nice(10))
        while True:
            save(root/'heartbeat.json', dict(state='CPU_TEST_INDEX_RUNNING', owner_pid=os.getpid(), worker_pid=child.pid,
                command=command, output_root=str(root), heartbeat_at=now(), no_gpu=True, model_inference=False, ot_recomputed=0))
            try:
                code = child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                continue
    save(root/'terminal.json', dict(state='COMPLETED' if code == 0 else 'FAILED', returncode=code,
        completed_at=now(), elapsed_seconds=time.monotonic()-started, plan_sha=stable(plan),
        freeze_sha=task['freeze_sha'], source_test_read=True if code == 0 else None,
        source_test_read_evidence='MIGRATION_COMPLETED' if code == 0 else 'UNKNOWN_GATE_OR_STREAM_STAGE',
        model_inference=False, ot_recomputed=0))
    return code


class Continuation:
    def __init__(self, plan, transport=None):
        self.plan = validate_plan(plan)
        self.paths = locations(plan)
        self.root = Path(self.paths['local'])
        self.transport = transport or Transport()
        self.worker_bytes = Path(__file__).read_bytes()
        require(hashlib.sha256(self.worker_bytes).hexdigest() == plan['continuation_worker_sha256'],
                'CONTINUATION_WORKER_CHANGED_FROM_SEALED_PLAN')
        self.renderer_bytes = Path(LOCAL_RENDERER).read_bytes()
        require(hashlib.sha256(self.renderer_bytes).hexdigest() == plan['renderer_sha256'],
                'OFFLINE_RENDERER_CHANGED_FROM_SEALED_PLAN')
        self.state_path = self.root/'checkpoint.json'
        self.mutex = threading.RLock()
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(plan_sha=stable(plan), phase='WAIT_FREEZE', jobs={})
        require(self.state['plan_sha'] == stable(plan), 'SEALED_CONTINUATION_PLAN_CHANGED')

    def checkpoint(self, **update):
        with self.mutex:
            self.state.update(update)
            self.state['heartbeat_at'] = now()
            self.state['owner_pid'] = os.getpid()
            save(self.state_path, self.state)

    def read(self, host, path):
        return self.transport.rpc(host, op='read', path=path)

    def put(self, local, host, destination):
        digest = hashlib.sha256(Path(local).read_bytes()).hexdigest()
        self.transport.copy(str(local), host + ':' + destination + '.partial')
        self.transport.rpc(host, op='finalize', source=destination+'.partial', target=destination, sha256=digest)
        return digest

    def advance(self):
        """At most one bounded business phase per call; waiting returns False."""
        p, t, paths, phase = self.plan, self.transport, self.paths, self.state['phase']
        if phase == 'WAIT_FREEZE':
            job = t.rpc('tongji-hpc', op='job', job=p['freeze_job_id'])
            self.checkpoint(freeze_job=job)
            require(job.get('reason') != 'DependencyNeverSatisfied', 'FREEZE_DEPENDENCY_PERMANENTLY_FAILED')
            if job['state'] in ('PENDING','RUNNING','COMPLETING','CONFIGURING','ACCOUNTING_NOT_YET_VISIBLE'):
                return False
            require(job['state'] == 'COMPLETED' and job['exit_code'] == '0:0', 'FREEZE_JOB_NOT_SUCCESSFUL_NO_SUCCESSOR')
            spec, freeze, side = [self.read('tongji-hpc', path) for path in (SPEC, paths['freeze'], paths['sidecar'])]
            require(all(x['exists'] for x in (spec, freeze, side)), 'ACTUAL_FREEZE_SPEC_SIDECAR_REQUIRED')
            validate_freeze(freeze['value'], spec['value'], p)
            require(spec['value']['execution_commit']==p['source_execution_binding']['spec_execution_commit'],
                    'SPEC_SCIENCE_IDENTITY_NOT_ACTUAL_DRIVER_COMMIT')
            require(side['value'].get('spec_sha256') == SPEC_SHA and side['value'].get('pool_manifest_sha256') == spec['value']['pools'][p['method']]['sha256']
                    and side['value'].get('calibration') == p['expected_calibration_entry'], 'EXISTING_NATIVE_CALIBRATION_BINDING_CHANGED')
            require(side['value'].get('test') is None, 'TEST_ALREADY_BOUND_DO_NOT_CREATE_DUPLICATE_CONTINUATION')
            self.root.mkdir(parents=True, exist_ok=True)
            # Retain exact source bytes, not just the equivalent parsed object.
            t.copy('tongji-hpc:'+SPEC, str(self.root/'spec.json'))
            require(hashlib.sha256((self.root/'spec.json').read_bytes()).hexdigest() == spec['sha256'], 'COPIED_SPEC_CHANGED')
            t.copy('tongji-hpc:'+paths['freeze'], str(self.root/'selection_freeze.json'))
            require(hashlib.sha256((self.root/'selection_freeze.json').read_bytes()).hexdigest() == freeze['sha256'], 'COPIED_FREEZE_CHANGED')
            self.checkpoint(phase='TRANSFER_BINDINGS', spec=spec['value'], freeze_sha=freeze['sha256'], sidecar_sha=side['sha256'])
        elif phase == 'TRANSFER_BINDINGS':
            t.rpc('autodl-a800', op='mkdir', path=paths['remote']+'/inputs')
            t.rpc('tongji-hpc', op='mkdir', path=paths['hpc'])
            self.put(self.root/'spec.json','autodl-a800',paths['remote']+'/inputs/spec.json')
            self.put(self.root/'selection_freeze.json','autodl-a800',paths['remote']+'/inputs/selection_freeze.json')
            worker = self.root/'continuation_worker.py'
            if worker.exists():
                require(hashlib.sha256(worker.read_bytes()).hexdigest()==p['continuation_worker_sha256'], 'SEALED_LOCAL_RELAY_WORKER_CHANGED')
            else:
                with worker.open('xb') as stream:
                    stream.write(self.worker_bytes);stream.flush();os.fsync(stream.fileno())
            worker_sha = self.put(worker,'autodl-a800',paths['remote']+'/inputs/continuation_worker.py')
            task = dict(plan=p, freeze_sha=self.state['freeze_sha'])
            save(self.root/'worker_task.json', task)
            self.put(self.root/'worker_task.json','autodl-a800',paths['remote']+'/inputs/worker_task.json')
            self.checkpoint(phase='LAUNCH_INDEX', worker_sha=worker_sha)
        elif phase == 'LAUNCH_INDEX':
            result = t.rpc('autodl-a800', op='launch', root=paths['remote'], worker=paths['remote']+'/inputs/continuation_worker.py',
                worker_sha=self.state['worker_sha'], plan=paths['remote']+'/inputs/worker_task.json', plan_sha=stable(p))
            require(result['plan_sha'] == stable(p), 'DIFFERENT_MIGRATION_OWNER_PRESENT')
            self.checkpoint(phase='WAIT_INDEX', migration_owner=result)
        elif phase == 'WAIT_INDEX':
            terminal = self.read('autodl-a800',paths['remote']+'/terminal.json')
            if not terminal['exists']:
                pulse = self.read('autodl-a800', paths['remote']+'/heartbeat.json')
                identity = self.state['migration_owner']
                live = t.rpc('autodl-a800', op='alive', pid=identity['pid'], start_ticks=identity['start_ticks'])
                if not live['alive']:
                    # Completion can race the first terminal read and /proc exit.
                    require(self.read('autodl-a800',paths['remote']+'/terminal.json')['exists'],
                            'MIGRATION_SUPERVISOR_EXITED_WITHOUT_TERMINAL_RETAIN_LOGS')
                    return True
                self.checkpoint(migration_heartbeat=pulse.get('value'))
                return False
            value = terminal['value']
            require(value['state'] == 'COMPLETED' and value['returncode'] == 0, 'MIGRATION_FAILED_NO_HELDOUT_SUBMISSION')
            require(value['plan_sha'] == stable(p) and value['freeze_sha'] == self.state['freeze_sha'], 'MIGRATION_TERMINAL_BINDING_CHANGED')
            index = t.rpc('autodl-a800', op='index', path=paths['remote']+'/test_index.json', freeze_sha=self.state['freeze_sha'], method=p['method'])
            self.checkpoint(phase='TRANSFER_INDEX', migrated_index=index)
        elif phase == 'TRANSFER_INDEX':
            local = self.root/'test_index.json'
            t.copy('autodl-a800:'+paths['remote']+'/test_index.json', str(local))
            require(hashlib.sha256(local.read_bytes()).hexdigest() == self.state['migrated_index']['sha256'], 'NEW_INDEX_TRANSFER_CHANGED')
            self.put(local,'tongji-hpc',paths['hpc_index'])
            self.checkpoint(phase='INSTALL_SIDECAR')
        elif phase == 'INSTALL_SIDECAR':
            entry = dict(path=paths['hpc_index'], sha256=self.state['migrated_index']['sha256'],
                new_test_freeze_path=paths['freeze'], new_test_freeze_sha256=self.state['freeze_sha'])
            result = t.rpc('tongji-hpc', op='sidecar', path=paths['sidecar'], expected_sha256=self.state['sidecar_sha'],
                entry=entry, calibration=p['expected_calibration_entry'], spec_sha=SPEC_SHA,
                pool_sha=self.state['spec']['pools'][p['method']]['sha256'], prior_receipt=paths['hpc']+'/native_raw_adoption_before.json')
            self.checkpoint(phase='SUBMIT_CHAIN', sidecar_install=result)
        elif phase == 'SUBMIT_CHAIN':
            dependency = str(p['freeze_job_id'])
            for stage in ('test','aggregate','audit','export'):
                if stage not in self.state['jobs']:
                    result = t.rpc('tongji-hpc', op='submit', receipt=paths['hpc']+'/'+stage+'_submission.json',
                        argv=job_argv(p,stage,dependency), cwd=EXECUTION, commit=EXECUTION_COMMIT)
                    with self.mutex:
                        self.state['jobs'][stage] = result
                        self.checkpoint()
                dependency = self.state['jobs'][stage]['job_id']
            self.checkpoint(phase='WAIT_CHAIN')
        elif phase == 'WAIT_CHAIN':
            jobs = {name:t.rpc('tongji-hpc',op='job',job=value['job_id']) for name,value in self.state['jobs'].items()}
            self.checkpoint(job_states=jobs)
            failures = {name:value for name,value in jobs.items() if value['state'] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED')}
            require(not failures, 'HELDOUT_CHAIN_FAILED_NO_AUTOMATIC_RERUN:'+json.dumps(failures))
            if not all(v['state']=='COMPLETED' and v['exit_code']=='0:0' for v in jobs.values()):
                return False
            audit = self.read('tongji-hpc', paths['audit'])
            require(audit['exists'] and audit['value']['state']=='RESULT_CONSISTENCY_PASS' and audit['value']['method']==p['method']
                    and audit['value']['spec_sha256']==SPEC_SHA and audit['value']['selection_freeze_sha256']==self.state['freeze_sha']
                    and audit['value']['main_matrix_write'] is False, 'ACTUAL_INDEPENDENT_AUDIT_REQUIRED')
            self.checkpoint(phase='TRANSFER_RESULTS', audit=dict(path=paths['audit'],sha256=audit['sha256']), main_matrix_write=False)
        elif phase == 'TRANSFER_RESULTS':
            manifest = self.state.get('result_manifest')
            if manifest is None:
                manifest = t.rpc('tongji-hpc',op='result_manifest',files=result_files(p))
                require(set(manifest)==set(result_files(p)), 'SCOPED_RESULT_SET_INCOMPLETE')
                require(manifest['provenance/independent_result_audit.json']['sha256']==self.state['audit']['sha256'],
                        'RESULT_AUDIT_CHANGED_AFTER_ACCEPTANCE')
                self.checkpoint(result_manifest=manifest)
            delivery = self.root/'results'
            for relative, identity in manifest.items():
                target = delivery/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    partial = target.with_name(target.name+'.partial')
                    t.copy('tongji-hpc:'+identity['source_path'],str(partial))
                    require(partial.stat().st_size==identity['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==identity['sha256'],
                            'SCOPED_RESULT_TRANSFER_CHANGED:'+relative)
                    os.replace(partial,target)
                else:
                    require(hashlib.sha256(target.read_bytes()).hexdigest()==identity['sha256'], 'LOCAL_RESULT_CONFLICT:'+relative)
            renderer = delivery/'tools/replot_bace_gin.py'
            renderer.parent.mkdir(parents=True,exist_ok=True)
            if renderer.exists():
                require(hashlib.sha256(renderer.read_bytes()).hexdigest()==p['renderer_sha256'], 'DELIVERY_RENDERER_CONFLICT')
            else:
                with renderer.open('xb') as stream:
                    stream.write(self.renderer_bytes);stream.flush();os.fsync(stream.fileno())
            self.checkpoint(phase='RENDER_RESULTS')
        elif phase == 'RENDER_RESULTS':
            delivery = self.root/'results'
            final_manifest = delivery/'delivery_manifest.json'
            if final_manifest.exists():
                old=json.loads(final_manifest.read_text())
                require(old['plan_sha']==stable(p) and old['independent_audit']==self.state['audit']
                    and old['files']==self.state['result_manifest'], 'LOCAL_DELIVERY_MANIFEST_CONFLICT')
                self.checkpoint(phase='COMPLETED',local_results=str(delivery),
                    result_state='LOCAL_PARTIAL_RESULTS_READY',autodl_result_publication=False)
                return True
            command = [LOCAL_PYTHON,'-I','-B',str(delivery/'tools/replot_bace_gin.py'),
                '--source-csv',str(delivery/'source_csv'),'--output',str(delivery/'figures')]
            t.run(command)
            tables = delivery/'tables'
            tables.mkdir(exist_ok=True)
            for source in (delivery/'source_csv/bace_gin_fixed141_table2.csv',
                    delivery/'figures/table2_bace_gin_fixed141.pdf',delivery/'figures/table2_bace_gin_fixed141.tex'):
                target=tables/source.name
                # Until the final manifest exists these are unsealed derived
                # local tables; a renderer-only retry does not repeat science.
                partial=target.with_name(target.name+'.partial')
                with partial.open('wb') as stream:
                    stream.write(source.read_bytes());stream.flush();os.fsync(stream.fileno())
                os.replace(partial,target)
            save(final_manifest,dict(state='LOCAL_PARTIAL_RESULTS_READY',plan_sha=stable(p),
                completed_at=now(),files=self.state['result_manifest'],renderer_sha256=p['renderer_sha256'],
                render_command=command,independently_audited_method=p['method'],
                independent_audit=self.state['audit'],main_matrix_write=False,autodl_result_publication=False,
                other_exported_rows_require_their_own_audits=True,source_science_repeated=False))
            self.checkpoint(phase='COMPLETED',local_results=str(delivery),
                result_state='LOCAL_PARTIAL_RESULTS_READY',autodl_result_publication=False)
        elif phase == 'COMPLETED':
            return False
        else:
            raise ValueError('UNKNOWN_FINITE_PHASE')
        return True

    def run(self, poll_seconds=60):
        require(30 <= poll_seconds <= 60, 'BOUNDED_REAL_STATUS_POLL_REQUIRED')
        self.root.mkdir(parents=True, exist_ok=True)
        # One local process lock per campaign+method, not a new scientific/GPU lock.
        with (Path(LOCAL)/(self.plan['method']+'.relay.lock')).open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            save(self.root/'plan.json',self.plan)
            stopped = threading.Event()
            def heartbeat():
                while not stopped.wait(60):
                    self.checkpoint()
            pulse = threading.Thread(target=heartbeat, daemon=True)
            pulse.start()
            try:
                while self.state['phase'] != 'COMPLETED':
                    changed = self.advance()
                    self.checkpoint()
                    if not changed:
                        time.sleep(poll_seconds)
            except Exception as exc:
                self.checkpoint(error=str(exc), state='BLOCKED_NO_AUTOMATIC_DUPLICATE_OR_SCIENCE_RETRY')
                raise
            finally:
                stopped.set()
                pulse.join(timeout=2)
        return self.state


if __name__ == '__main__':
    require(len(sys.argv)==3 and sys.argv[1]=='--remote-index-worker', 'ONLY_SCOPED_REMOTE_INDEX_WORKER_ENTRY')
    try:
        raise SystemExit(remote_index_worker(sys.argv[2]))
    except Exception as exc:
        task=json.loads(Path(sys.argv[2]).read_text())
        root=Path(locations(validate_plan(task['plan']))['remote'])
        save(root/'terminal.json', dict(state='FAILED_ENGINEERING', returncode=2, error=str(exc), completed_at=now()))
        raise
