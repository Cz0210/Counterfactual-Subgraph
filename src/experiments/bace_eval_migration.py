"""BACE-only evaluation migration. No training, proposals, scheduler or matrix IO."""
from __future__ import annotations
import copy
import fcntl
import math
import os
from pathlib import Path
import resource
import time
from dataclasses import asdict

from src.eval.bace_frozen_gnn_contracts import (atomic_json, atomic_csv, read_json,
    read_jsonl, sha256_file, stable_sha256, utc_now)
from src.experiments.bace_gin_reach_selector import POLICY, select
from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
from src.eval.bace_reach_selector import ReachMasks

SCOPES = {'gnn_a': 'GIN_SOURCE_APLUS2659_PROPOSAL_FIXED_BACKBONE_SENSITIVITY',
          'llm_gin': 'FIXED_OUTPUT_CROSS_ORACLE_PROPOSER_SENSITIVITY'}
BACKBONES = ('gin', 'gine', 'gatedgcn_plus', 'gcn', 'gatv2')


def bound(item):
    path = Path(item['path'])
    if sha256_file(path) != item['sha256']:
        raise ValueError('INPUT_BINDING_CHANGED:'+str(path))
    return read_jsonl(path) if path.suffix == '.jsonl' else read_json(path)


def seal(path, value):
    result = dict(value, self_sha256=stable_sha256(value))
    if Path(path).exists():
        if read_json(path) != result:
            raise ValueError('IMMUTABLE_MIGRATION_RECEIPT_CONFLICT:'+str(path))
    else:
        atomic_json(path, result)
    return result


def reopen(path):
    value = read_json(path)
    if value.get('self_sha256') != stable_sha256({k:v for k,v in value.items() if k!='self_sha256'}):
        raise ValueError('MIGRATION_SELF_HASH:'+str(path))
    return value


def validate(spec):
    family = spec['family']
    if family not in SCOPES or spec['scope'] != SCOPES[family] or spec['selector_policy'] != POLICY:
        raise ValueError('EXPERIMENT_SCOPE_OR_SELECTOR_CHANGED')
    if any(spec[k] is not False for k in ('training', 'temperature_fit', 'generation', 'main_matrix_write')):
        raise ValueError('EVALUATION_ONLY_AUTHORIZATION')
    roles = set(BACKBONES) if family == 'gnn_a' else {'L0','L1','L2','L3'}
    if set(spec['roles']) != roles:
        raise ValueError('ROLE_SET_CHANGED')
    root = Path(spec['output_root']).resolve()
    root.relative_to('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/experiments')
    if spec['cpu_threads'] != 8 or spec['max_concurrent_jobs'] != 2:
        raise ValueError('CPU_BOUND_CHANGED')
    for role, config in spec['roles'].items():
        if config['backbone'] != (role if family=='gnn_a' else 'gin'):
            raise ValueError('WRONG_ORACLE_ROLE')
        pool = bound(config['pool'])
        ids = [r['candidate_id'] for r in pool]
        if not ids or len(ids)!=len(set(ids)) or (family=='gnn_a' and len(ids)!=2659):
            raise ValueError('FROZEN_POOL_COUNT_OR_ID_CHANGED')
        if not config['old_orders'] or any(not order or not set(order)<=set(ids)
                or len(order)!=len(set(order)) for order in config['old_orders'].values()):
            raise ValueError('A_PLUS_CALIBRATION_FLOOR_ORDER_NOT_BOUND')
        if family=='llm_gin' and config['attempts'] != 3088:
            raise ValueError('ATTEMPT_BUDGET_CHANGED')
    for source in spec.get('saved_raw_sources',[]):
        if source['kind']=='aplus':
            validate_aplus_source_audit(bound(source['audit']),source)
    return root


def validate_aplus_source_audit(audit, source):
    """Use the actual accepted A+ audit schema, not a generic PASS string."""
    if (audit.get('state')!='SAVED_RECORD_AND_METRIC_CONSISTENCY_PASS'
            or audit.get('audit_scope')!='SAVED_APPLICATIONS_AND_INDEPENDENT_METRIC_REDUCER_NOT_MODEL_REEXECUTION'
            or audit.get('spec_sha256')!=source['spec_sha256']
            or audit.get('self_sha256')!=stable_sha256({k:v for k,v in audit.items() if k!='self_sha256'})
            or audit.get('main_matrix_write') is not False
            or audit.get('model_inference_rerun') is not False or audit.get('ot_recomputed')!=0):
        raise ValueError('A_PLUS_SOURCE_NOT_ACCEPTED')
    units=audit['parent_units']
    for split,count in (('calibration',66),('test',141)):
        ids=[r['parent_id'] for r in units if r['split']==split]
        if len(ids)!=count or len(set(ids))!=count:
            raise ValueError('A_PLUS_AUDITED_PARENT_COMPLETENESS')
    if len(units)!=207:
        raise ValueError('A_PLUS_UNEXPECTED_AUDITED_RECORD')
    return audit


def require_freeze(spec):
    frozen = reopen(Path(spec['output_root'])/'CALIBRATION_FREEZE.json')
    if (frozen['spec_sha256'] != stable_sha256(spec) or frozen['test_loaded'] is not False
            or frozen['policy'] != POLICY or set(frozen['selectors']) != set(spec['roles'])):
        raise ValueError('ALL_ROLE_CALIBRATION_FREEZE_REQUIRED')
    return frozen


def admission(spec):
    root = Path(spec['output_root'])
    v = os.statvfs(root)
    available = v.f_bavail*v.f_frsize
    # Joint two-job reserve derives from the explicit per-job compact bound;
    # observed files are not charged again. Sentinel inode counts remain unknown.
    owned=[]
    for directory in spec['joint_output_roots']:
        base=Path(directory).resolve()
        base.relative_to(root.parent)
        size=sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and not p.is_symlink()) if base.exists() else 0
        if size > spec['max_output_bytes']:
            raise ValueError('TASK_COMPACT_OUTPUT_BOUND_EXCEEDED')
        owned.append(size)
    required = sum(spec['max_output_bytes']-size for size in owned) + spec['storage_reserve_bytes']
    if available < required:
        raise ValueError(f'PATH_QUOTA_SHORTFALL:{available}<{required}')
    limit = 20000 + 2*spec['max_new_files']
    if v.f_files < 2**60 and v.f_favail < limit:
        raise ValueError('STAGE_FILE_BUDGET_SHORTFALL')
    return dict(path=str(root), available_bytes=available, required_bytes=required,
        inode_available=v.f_favail if v.f_files<2**60 else None,
        inode_source='statvfs_path_quota' if v.f_files<2**60 else 'sentinel_unknown',
        bounded_new_files=spec['max_new_files'])


def load_raw(spec, split, delegate, bundle_manifest):
    """Only complete graph costs. Current model's logits/masks/minima never reused."""
    from src.ablations.gnn.reach_raw_distance_reuse import (VerifiedRawGraphDistance,
        raw_contract_from_bundle, graph_key)
    if split=='test':
        require_freeze(spec)
    index = copy.deepcopy(bound(spec['raw_indexes'][split]))
    if index['split'] != split:
        raise ValueError('RAW_SPLIT_CHANGED')
    # Constructor validates full encoder/schema/numerical contract and kernel bytes.
    VerifiedRawGraphDistance(delegate, index=index, current_raw_contract=raw_contract_from_bundle(bundle_manifest),
        repo=Path(__file__).resolve().parents[2])
    added = 0
    for source in (spec.get('saved_raw_sources', []) if split!='train' else []):
        if source['kind']=='aplus':
            audit = validate_aplus_source_audit(bound(source['audit']),source)
            directory=Path(source['root'])/'aligned_pool'/split
            files=sorted(directory.glob('parent-*.json'))
            if len(files) != {'calibration':66,'test':141}[split]:
                raise ValueError('A_PLUS_SOURCE_PARENT_COUNT')
            records=[(str(p), reopen(p)) for p in files]
            for _, record in records:
                if record['spec_sha256'] != source['spec_sha256']:
                    raise ValueError('A_PLUS_PARENT_SOURCE_SPEC')
        elif source['kind']=='llm':
            audit=bound(source['audit'])
            if audit.get('state')!='PASS' or audit.get('main_matrix_write') is not False:
                raise ValueError('LLM_SOURCE_NOT_ACCEPTED')
            run=bound(dict(path=str(Path(source['root'])/'run_manifest.json'),sha256=audit['files']['run_manifest.json']))
            if run['bundle_sha256'] != source['original_bundle_sha256']:
                raise ValueError('LLM_SOURCE_ENCODER_FEATURE_BUNDLE_CHANGED')
            for rel,digest in source['kernel_identity'].items():
                if sha256_file(Path(__file__).resolve().parents[2]/rel)!=digest:
                    raise ValueError('LLM_RAW_NUMERICAL_IMPLEMENTATION_CHANGED')
            records=[]
            for rel,digest in audit['files'].items():
                if rel.startswith('parent_checkpoints/'+split+'/') and rel.endswith('.json'):
                    if Path(rel).name=='progress.json':
                        continue
                    raw=bound(dict(path=str(Path(source['root'])/rel),sha256=digest))
                    if raw.get('science_sha256')!=stable_sha256(raw['science']):
                        raise ValueError('LLM_RAW_SCIENCE_BINDING')
                    records.append((rel,raw['science']))
        else:
            raise ValueError('UNREVIEWED_RAW_SOURCE_SCHEMA')
        for path, record in records:
            for row in record['match_rows']:
                if row.get('distance_ok') is not True:
                    continue
                value=row['wnode_distance']
                if (not isinstance(value,(int,float)) or not math.isfinite(value) or value<0
                        or row.get('delete_valid') is not True or row.get('residual_connected') is not True):
                    raise ValueError('INVALID_RAW_DISTANCE_SOURCE')
                key,p,r=graph_key(row['parent_smiles'],row['residual_smiles'],index['raw_contract_sha256'])
                prior=index['graph_costs'].get(key)
                if prior and prior['distance']!=value:
                    raise ValueError('RAW_GRAPH_DISTANCE_CONFLICT:'+key)
                if not prior:
                    index['graph_costs'][key]=dict(parent=p,residual=r,distance=value,
                        source_records=[dict(source=path, source_audit_sha=source['audit']['sha256'],
                            match_sha=stable_sha256(row), match_atom_indices=row['match_atom_indices'])])
                    added+=1
    index['migration_raw_sources']=spec.get('saved_raw_sources',[])
    index['migration_additional_raw_costs']=added
    index['self_sha256']=stable_sha256({k:v for k,v in index.items() if k!='self_sha256'})
    return VerifiedRawGraphDistance(delegate,index=index,current_raw_contract=raw_contract_from_bundle(bundle_manifest),
        repo=Path(__file__).resolve().parents[2])


def evaluate_role(spec, role, split, *, timing=False):
    import torch
    from src.experiments.bace_gin_ours import original_bundle, fixed_source_parents
    from src.ablations.gnn.cpu_evaluation import _featurizer,_distance,_predict
    from src.ablations.llm.compact_node_cache import install_compact_node_cache
    from src.oracles.gnn_oracle import GNNOracle
    from src.eval.bace_reach_v2 import evaluate_pairs
    root=validate(spec); frozen=require_freeze(spec) if split=='test' else None
    cfg=spec['roles'][role]; pool=bound(cfg['pool'])
    if frozen:
        selected=set().union(*(set(v['ordered_rule_ids']) for v in frozen['selectors'][role].values()))
        pool=[r for r in pool if r['candidate_id'] in selected]
    parents=fixed_source_parents(spec,split,test_authorized=frozen is not None)
    if timing:
        if split!='train': raise ValueError('TIMING_TRAIN_ONLY')
        parents=parents[:1]
    out=root/role/split; out.mkdir(parents=True,exist_ok=True)
    if (out/'terminal.json').exists():
        return reopen(out/'terminal.json')
    torch.set_num_threads(spec['cpu_threads'])
    bundle, manifest=original_bundle(spec)
    for name,digest in cfg['model_files'].items():
        if sha256_file(Path(cfg['model_root'])/name)!=digest:
            raise ValueError('FROZEN_CLASSIFIER_CHANGED:'+role)
    oracle=GNNOracle.from_checkpoint(cfg['model_root'],device='cpu',batch_size=64,verify_hashes=False)
    if oracle.backbone!=cfg['backbone'] or oracle.checkpoint_id!=cfg['model_files']['model.pt']:
        raise ValueError('ACTUAL_LOADED_BACKBONE_CHANGED')
    for p in oracle.model.parameters():
        p.requires_grad_(False)
    oracle.model.eval()
    feat=_featurizer(bundle,manifest)
    # Roles run serially under this job's writer lease. Share only the delegate's
    # graph-pair raw-cost cache, never a model/action/flip result or another job's
    # active SQLite. Preserve newly computed exact costs across roles, too.
    dist=_distance(bundle,manifest,root/'raw_cache')
    install_compact_node_cache(dist)
    dist=load_raw(spec,split,dist,manifest)
    started=time.monotonic(); total_bytes=0
    try:
        for i,parent in enumerate(parents):
            path=out/f'parent-{i:04d}.json'
            binding=stable_sha256(dict(spec=stable_sha256(spec),role=role,split=split,
                parent=asdict(parent),pool=[r['candidate_id'] for r in pool],freeze=frozen))
            if path.exists():
                if reopen(path)['binding']!=binding:
                    raise ValueError('PARENT_RESUME_CONFLICT')
            else:
                admission(spec)
                tick=time.monotonic()
                pairs,matches=evaluate_pairs([parent],pool,oracle=oracle,featurizer=feat,
                    distance_provider=dist,split=split,oracle_checkpoint_id=oracle.checkpoint_id,
                    oracle_batch_size=64)
                seal(path,dict(binding=binding,parent_id=parent.parent_id,pair_rows=pairs,match_rows=matches,
                    elapsed_seconds=time.monotonic()-tick,old_flip_masks_adopted=False))
            total_bytes+=path.stat().st_size
            atomic_json(root/'progress.json',dict(state='RUNNING',role=role,split=split,completed_units=i+1,
                total_units=len(parents),pid=os.getpid(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                elapsed_seconds=time.monotonic()-started,updated_at=utc_now()))
            if total_bytes>spec['max_output_bytes']//2:
                raise ValueError('COMPACT_STAGE_OUTPUT_BOUND_REACHED_AFTER_CHECKPOINT')
        return seal(out/'terminal.json',dict(state='PARENT_EVALUATION_COMPLETE',role=role,split=split,
            parent_count=len(parents),pool_count=len(pool),spec_sha256=stable_sha256(spec),
            raw_distance_stats=dist.stats_dict(),elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    finally:
        dist.close()


def role_records(spec, role, split):
    root=Path(spec['output_root'])/role/split
    terminal=reopen(root/'terminal.json')
    if terminal['spec_sha256']!=stable_sha256(spec):
        raise ValueError('TERMINAL_SPEC_CHANGED')
    result=[reopen(root/f'parent-{i:04d}.json') for i in range(terminal['parent_count'])]
    if len({x['parent_id'] for x in result})!=len(result):
        raise ValueError('PARENT_PARTITION_DUPLICATE')
    return result


def freeze(spec):
    root=validate(spec)
    if (root/'CALIBRATION_FREEZE.json').exists():
        return require_freeze(spec)
    all_rows={r:role_records(spec,r,'calibration') for r in spec['roles']}
    natives={r:[x['parent_id'] for x in records if x['pair_rows'][0]['pred_before']==1]
             for r,records in all_rows.items()}
    common=set.intersection(*(set(ids) for ids in natives.values()))
    if spec['family']=='gnn_a' and not common:
        raise ValueError('BLOCKED_EMPTY_CALIBRATION_COMMON')
    thresholds=bound(spec['thresholds']); selections={}
    for role,records in all_rows.items():
        cfg=spec['roles'][role]; pool=bound(cfg['pool'])
        base=[x['parent_id'] for x in records]
        cohorts={'native':natives[role], 'common':[p for p in base if p in common]} if spec['family']=='gnn_a' else {'fixed141':base}
        selections[role]={}
        for mode,ids in cohorts.items():
            if not ids:
                raise ValueError('BLOCKED_EMPTY_CALIBRATION_COHORT')
            rows=[p for x in records if x['parent_id'] in ids for p in x['pair_rows']]
            matrix=matrix_from_pairs(ids,pool,rows,root=root,split='calibration')
            selected=select(ReachMasks.from_distances([r['candidate_id'] for r in pool],matrix.distances,thresholds),cfg['old_orders'][mode])
            selections[role][mode]=dict(selected,calibration_parent_ids=ids)
    return seal(root/'CALIBRATION_FREEZE.json',dict(state='ALL_SELECTORS_FROZEN',selectors=selections,
        spec_sha256=stable_sha256(spec),policy=POLICY,test_loaded=False,created_at=utc_now()))


def export(spec):
    from src.experiments.bace_gin_fixed_pool import prefix_metrics
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    root=validate(spec); frozen=require_freeze(spec); th=threshold_bundle_from_dict(bound(spec['thresholds']))
    data={r:role_records(spec,r,'test') for r in spec['roles']}
    native={r:{x['parent_id'] for x in records if x['pair_rows'][0]['pred_before']==1} for r,records in data.items()}
    common=set.intersection(*native.values()); curves=[]; ecdf=[]; distances=[]
    for role,records in data.items():
        base=[x['parent_id'] for x in records]
        for mode,selected in frozen['selectors'][role].items():
            ids=base if spec['family']=='llm_gin' else [p for p in base if p in (native[role] if mode=='native' else common)]
            order=selected['ordered_rule_ids']
            rows=[p for x in records if x['parent_id'] in ids for p in x['pair_rows'] if p['candidate_id'] in order]
            report=prefix_metrics(ids,order,rows,theta=th.theta_star,cap=th.cost_cap,endpoints=th.raw_thresholds)
            for source,target in [('prefix_rows',curves),('exact_ecdf',ecdf),('parent_distances',distances)]:
                for item in report[source]:
                    if spec['family']=='gnn_a' and item.get('cohort')=='gin_native':
                        continue
                    item=dict(item,role=role,selection_cohort=mode)
                    if spec['family']=='gnn_a' and 'cohort' in item: item['cohort']=mode
                    target.append(item)
    paths={}
    for name,rows in [('prefix_metrics.csv',curves),('exact_ecdf.csv',ecdf),('parent_distances.csv',distances)]:
        path=root/'source_csv'/name;atomic_csv(path,rows);paths[name]=sha256_file(path)
    atomic_csv(root/'source_csv/table_k10.csv',[r for r in curves if r['K_requested']==10])
    return seal(root/'final_audit.json',dict(state='EVALUATION_COMPLETE',scope=spec['scope'],
        spec_sha256=stable_sha256(spec),freeze_sha256=frozen['self_sha256'],source_csv=paths,
        calibration_before_test=True,training=False,generation=False,main_matrix_write=False,
        independent_oracle_rerun_claimed=False,completed_roles=list(data)))


def run(spec):
    root=validate(spec);root.mkdir(parents=True,exist_ok=True)
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1'):
        raise ValueError('HPC_CPU_COMPUTE_NODE_REQUIRED')
    with (root/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seal(root/'contract.json',spec)
        # This uses actual full-rule train-only timing, never a self-distance
        # substitute or a held-out parent; already sealed timing is resumed.
        evaluate_role(spec,next(iter(spec['roles'])),'train',timing=True)
        for split in ('calibration','test'):
            if split=='test': freeze(spec)
            for role in spec['roles']: evaluate_role(spec,role,split)
        return export(spec)
