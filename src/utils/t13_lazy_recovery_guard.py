"""Narrow admission for one authorized lazy T13 successor, inside its old owner."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from src.utils.t8_hpc_t13_successor_v1 import (
    atomic_json, atomic_json_no_replace, canonical_sha256, require_self_hash)

GIB = 1024**3
START_HEADROOM = 384*GIB
OTHER_MAIN_RESERVE = 192*GIB
CANARY_RSS_CAP = 96*GIB


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources(runtime):
    c = Path('/sys/fs/cgroup/memory')
    limit=int((c/'memory.limit_in_bytes').read_text())
    usage=int((c/'memory.usage_in_bytes').read_text())
    v=os.statvfs(runtime)
    return {'cgroup_limit_bytes':limit,'cgroup_usage_bytes':usage,
            'headroom_bytes':limit-usage,'failcnt':int((c/'memory.failcnt').read_text()),
            'free_bytes':v.f_frsize*v.f_bavail,'free_inodes':v.f_favail}


class T13LazyRecoveryGuard:
    def __init__(self, authorization, authorization_sha256, t13, owner_root):
        self.path=Path(authorization)
        if file_sha(self.path)!=authorization_sha256:
            raise ValueError('T13_REPAIR_AUTHORIZATION_HASH_CHANGED')
        self.auth=json.loads(self.path.read_text())
        require_self_hash(self.auth,'self_sha256','T13 lazy repair authorization')
        if (self.auth.get('schema_version')!='t13_lazy_repair_user_authorization_v1'
                or self.auth.get('authorized_by')!='user_project_owner'
                or self.auth.get('max_full_starts')!=1
                or self.auth.get('allow_lazy_indexed_dataset_repair') is not True
                or self.auth.get('require_exact_data_pipeline_parity') is not True
                or self.auth.get('allow_fresh_successor_after_parity') is not True):
            raise ValueError('T13_NARROW_REPAIR_AUTHORIZATION_REQUIRED')
        for key in ('remining','retransfer_6gb_package','reduce_cohort','change_decoder',
                    'relax_chemistry','test_used_in_training','gpu_borrow_enabled'):
            if self.auth.get(key) is not False:
                raise ValueError('T13_REPAIR_SCOPE_CHANGED:'+key)
        old_path=Path(self.auth['source_task_spec_path'])
        if file_sha(old_path)!=self.auth['source_task_spec']['sha256']:
            raise ValueError('T13_FAILED_SPEC_CHANGED')
        old=json.loads(old_path.read_text())
        if (self.auth['science_contract']!=t13['science_contract']
                or t13['gpu_uuid']!=self.auth['required_gpu_uuid']
                or t13['gpu_lease_path']!=self.auth['required_gpu_lease_path']
                or t13['publisher_id']!=self.auth['publisher_id']
                or t13['required_import_root']!=old['required_import_root']):
            raise ValueError('T13_LAZY_SCIENCE_OR_OWNER_CONTRACT_CHANGED')
        for key,value in old['input_paths'].items():
            if key not in ('wnode_cache_db','node_embedding_cache_dir') and t13['input_paths'][key]!=value:
                raise ValueError('T13_LAZY_INPUT_CHANGED:'+key)
        failure=json.loads((self.path.parent/'failure_audit.json').read_text())
        require_self_hash(failure,'self_sha256','T13 failure audit')
        for key in ('old_owner_pid','old_science_pid'):
            if Path('/proc',str(failure[key])).exists():
                raise ValueError('T13_PRIOR_PID_REQUIRES_IDENTITY_REVIEW')
        self.spec=t13; self.owner=Path(owner_root)
        self.canary=self.owner/'lazy-canary'
        self.runtime=self.path.parents[2]
        self.peak=0; self.min_headroom=None; self.baseline=None; self.samples=0
        self.authorization_sha256=authorization_sha256

    def admission(self):
        self.baseline=resources(self.runtime)
        if self.baseline['headroom_bytes']<START_HEADROOM:
            raise ValueError('T13_WAITING_384_GIB_HEADROOM')
        # This does not lower Mut's separate 100000 inode guard. T13 reserves
        # a compact index/checkpoint budget rather than per-sample files.
        if self.baseline['free_bytes']<100*GIB or self.baseline['free_inodes']<8192:
            raise ValueError('T13_COMPACT_OUTPUT_CAPACITY_UNAVAILABLE')
        atomic_json(self.owner/'lazy_memory_admission.json',dict(self.baseline,
            other_main_peak_reserve_bytes=OTHER_MAIN_RESERVE,
            t13_checkpoint_and_index_reserved_inodes=4096,
            state='CANARY_ADMITTED_NOT_FULL_ADMISSION'))

    def command(self):
        args=[self.spec['python'],'-I','-B',str(Path(self.spec['repo_root'])/'scripts/autodl/canary_t13_indexed_dataset.py'),
              '--config',str(Path(self.spec['repo_root'])/'configs/hpc.yaml'),
              '--set','inference.fallback_to_heuristic=false','--output-root',str(self.canary)]
        for option,key in (('--official-root','official_root'),('--train-csv','train_csv'),('--gnn-checkpoint','gnn_checkpoint')):
            args.extend((option,self.spec['input_paths'][key]))
        args.extend(('--gspan-adoption-proof',str(Path(self.spec['required_import_root'])/'adoption_proof.json'),
                     '--device','cuda:0','--targets','0,2'))
        return args

    def sample(self,pid):
        # Reuse the existing exact descendant-tree reader; no fuzzy signals.
        from scripts.autodl.run_t14_route_c_owner import _process_tree_snapshot
        rows=_process_tree_snapshot(pid)
        rss=sum(r['rss_bytes'] for r in rows)
        now=resources(self.runtime)
        self.peak=max(self.peak,rss); self.samples+=1
        self.min_headroom=min(now['headroom_bytes'],self.min_headroom or now['headroom_bytes'])
        payload={**now,'observed_at':datetime.now(timezone.utc).isoformat(),
                 'pid':pid,'process_tree':rows,'process_tree_rss_bytes':rss,
                 'process_tree_peak_bytes':self.peak,'min_headroom_bytes':self.min_headroom,
                 'samples':self.samples,'safe_reserve_bytes':OTHER_MAIN_RESERVE}
        atomic_json(self.owner/'lazy_memory_progress.json',payload)
        if (rss>CANARY_RSS_CAP or now['headroom_bytes']<OTHER_MAIN_RESERVE
                or now['failcnt']>self.baseline['failcnt']):
            raise ValueError('T13_MEMORY_GATE_PRESSURE_SAFE_STOP_REQUIRED')

    def accept_canary_and_claim_full(self):
        if (self.path.parent/'full_start.json').exists():
            raise ValueError('T13_FULL_START_ALREADY_CONSUMED_CHECKPOINT_RECOVERY_REQUIRED')
        path=self.canary/'canary.json'
        report=json.loads(path.read_text())
        if (report.get('state')!='T13_INDEXED_DATA_AND_SHORT_TRAINING_CANARY_PASS'
                or report.get('targets_order')!=[0,2] or report.get('seed')!=7
                or report.get('configured_epochs')!=100 or report.get('test_loaded') is not False
                or report.get('calibration_loaded') is not False or report.get('mining_recomputed') is not False
                or report.get('independent_reload_pass') is not True):
            raise ValueError('T13_COMPLETE_CANARY_CONTRACT_REQUIRED')
        for gate in ('index_contract_pass','mask_rng_batch_parity','training_step_parity','reload_parity'):
            if report.get(gate) is not True:
                raise ValueError('T13_CANARY_GATE_MISSING:'+gate)
        boundary=self.canary/'memory_samples.json'
        memory=json.loads(boundary.read_text())
        for row in memory.get('samples',[]):
            self.peak=max(self.peak,int(row.get('VmHWM_bytes',row.get('VmRSS_bytes',0))))
        # Per-step boundary probes capture peaks between the owner polls. Keep
        # this independent of the periodic process-tree samples above.
        for target in (0, 2):
            boundary_path=self.canary/f'target_{target}'/'training_canary'/'memory_boundaries.json'
            if boundary_path.is_file():
                evidence=json.loads(boundary_path.read_text())
                rows=evidence if isinstance(evidence,list) else evidence.get('samples',[])
                for row in rows:
                    self.peak=max(self.peak,int(row.get('VmHWM_bytes',row.get('VmRSS_bytes',0))))
        if self.peak<=0 or self.samples<1:
            raise ValueError('T13_MEASURED_PROCESS_TREE_PEAK_REQUIRED')
        now=resources(self.runtime)
        required=max(START_HEADROOM,OTHER_MAIN_RESERVE+2*self.peak)
        if now['headroom_bytes']<required or now['failcnt']!=self.baseline['failcnt']:
            raise ValueError('T13_MEASURED_FULL_PEAK_ADMISSION_FAILED')
        receipt={'state':'T13_LAZY_CANARY_AND_MEMORY_ADMISSION_PASS',
                 'authorization_sha256':self.authorization_sha256,'task_spec_sha256':self.spec['task_spec_sha256'],
                 'canary_sha256':file_sha(path),'process_tree_peak_bytes':self.peak,
                 'min_headroom_bytes':self.min_headroom,'required_full_headroom_bytes':required,
                 'current_headroom_bytes':now['headroom_bytes'],'other_main_peak_reserve_bytes':OTHER_MAIN_RESERVE,
                 'checkpoint_extra_peak_factor':2,'full_trajectory_parity_claimed':False}
        atomic_json(self.owner/'lazy_canary_acceptance.json',receipt)
        claim={'schema_version':'t13_one_lazy_full_start_v1','attempt_id':self.spec['attempt_id'],
               'task_id':self.spec['task_id'],'output_root':self.spec['output_root'],
               'task_spec_sha256':self.spec['task_spec_sha256'],'owner_pid':os.getpid(),
               'canary_acceptance_sha256':file_sha(self.owner/'lazy_canary_acceptance.json'),
               'authorization_sha256':self.authorization_sha256,'max_full_starts':1}
        atomic_json_no_replace(self.path.parent/'full_start.json',claim)
        return receipt
