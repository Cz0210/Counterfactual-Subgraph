"""Selected-union-only continuation using the existing RF/GINE/WNode kernels.

The calibration spec and ordered freezes are immutable. One committed block
per parent holds both pairs and matches. Partial or failed numerics never
become biological failures, and a resumed block is not reevaluated.
"""
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.eval.selector_controlled_v7 import (base_parent_ids, digest, load_matrix,
    prefix_rows, rows, write_csv, VALID_FAILURES)
from src.eval.bace_frozen_gnn_contracts import atomic_json


def frozen_inputs(spec_path, phase):
    spec=json.loads(Path(spec_path).read_text()); root=Path(spec['output_root'])/phase
    binding=json.loads((root/'input_binding.json').read_text())
    if binding['spec_sha']!=digest(spec):raise ValueError('SEALED_SPEC_CHANGED')
    frozen=json.loads((root/'ALL_CALIBRATION_FROZEN.json').read_text())
    orders={}
    for sid in frozen['variants']:
        f=json.loads((root/(sid+'_freeze.json')).read_text());sha=f.pop('freeze_sha256')
        if digest(f)!=sha:raise ValueError('FREEZE_CHANGED:'+sid)
        orders[sid]=f['ordered_candidate_ids']
    union=sorted({c for seq in orders.values() for c in seq})
    if set(union)!=set(frozen['selected_union']):raise ValueError('SELECTED_UNION_CHANGED')
    return spec,root,orders,union


def validate_new_block(block, requested):
    got=[(r['parent_id'],r['candidate_id']) for r in block['pairs']]
    if len(got)!=len(set(got)) or set(got)!=set(requested):
        raise ValueError('BLOCK_PAIR_SET_CHANGED')
    for r in block['pairs']:
        if r.get('pair_strict_flip') is False:
            if (r.get('failure_reason') not in VALID_FAILURES and not
                    (r.get('failure_reason')=='no_valid_strict_flip_with_finite_wnode'
                     and r.get('num_strict_flip_matches')==0)):
                raise ValueError('UNKNOWN_PAIR_NOT_INFINITY')
    for row in block['matches']:
        if row.get('teacher_strict_flip') or row.get('cf_flip'):
            d=row.get('wnode_distance')
            if d is None or not math.isfinite(float(d)) or float(d)<0:
                raise ValueError('STRICT_MATCH_NUMERICAL_FAILURE_NOT_NONFLIP')


def make_evaluator(spec, adapter, work):
    from src.eval.node_wasserstein_distance import MolCLRNodeWassersteinConfig, MolCLRNodeWassersteinDistance
    from src.eval.bace_frozen_gnn_contracts import sha256_file
    for role,expected in [('molclr_checkpoint','molclr_sha256'),('oracle_file','oracle_sha256')]:
        if sha256_file(adapter[role])!=spec[expected]:raise ValueError('ASSET_IDENTITY_CHANGED:'+role)
    # Only the new continuation cache is writable. Original raw pair matrices
    # remain the first-level reuse source; original database files are untouched.
    provider=MolCLRNodeWassersteinDistance(MolCLRNodeWassersteinConfig(
        molclr_root=adapter['molclr_root'],molclr_ckpt=adapter['molclr_checkpoint'],
        cache_db=work/'new-wnode.sqlite',node_emb_cache_dir=work/'new-node-embeddings',
        device='cpu',feature_cost='cosine',node_mass='uniform',size_penalty_beta=0.,
        distance_namespace=adapter['distance_namespace']))
    if spec['dataset']=='Mutagenicity':
        from src.rewards.teacher_semantic import TeacherSemanticScorer
        from src.eval.mutagenicity_wnode_matrix import CalibrationParent,evaluate_parent_candidate_pair
        from src.eval.close_counterfactual_coverage import predict_with_teacher
        teacher=TeacherSemanticScorer(adapter['oracle_file'],device='cpu')
        if not teacher.available:raise ValueError('RF_UNAVAILABLE:'+teacher.availability_reason)
        if adapter['distance_namespace']!='molclr_node_wasserstein_v1':raise ValueError('MUT_NATIVE_DISTANCE_NAMESPACE')
        def evaluate(row, candidates):
            p=CalibrationParent(row['id'],row['smiles'],int(row['label']),'test')
            before=predict_with_teacher(teacher,p.smiles,1)
            pairs=[];matches=[]
            for c in candidates:
                pair,ms=evaluate_parent_candidate_pair(p,c,teacher=teacher,distance_provider=provider,before_prediction=before)
                pairs.append(pair);matches.extend(ms)
            return dict(pairs=pairs,matches=matches)
    elif spec['dataset']=='BACE':
        from src.eval.bace_frozen_gnn_verification import _evaluate_rows
        from src.eval.bace_frozen_gnn_pool import _checkpoint_contract
        from src.eval.bace_frozen_gnn_contracts import BACEParent
        from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
        from src.chem.hard_deletion import CONNECTED_WNODE_CACHE_NAMESPACE
        from src.oracles.oracle_factory import build_oracle
        card,schema=_checkpoint_contract(Path(adapter['oracle_bundle']))
        oracle=build_oracle(dataset='bace',backend='gnn',checkpoint=adapter['oracle_bundle'],device='cpu',batch_size=256)
        if (oracle.checkpoint_id!=spec['oracle_sha256'] or oracle.temperature!=spec['temperature']
                or adapter['distance_namespace']!=CONNECTED_WNODE_CACHE_NAMESPACE):raise ValueError('BACE_GINE_CONTRACT_CHANGED')
        featurizer=MolecularGraphFeaturizer(schema)
        def evaluate(row,candidates):
            p=BACEParent(row['id'],row['smiles'],int(row['label']),row['index'])
            pairs,matches=_evaluate_rows([p],candidates,oracle=oracle,featurizer=featurizer,
                distance_provider=provider,oracle_batch_size=256,split='test',oracle_checkpoint_id=oracle.checkpoint_id)
            return dict(pairs=pairs,matches=matches)
    else:raise ValueError('ONLY_TWO_AUTHORIZED_DATASETS')
    return evaluate,provider


def complete(spec_path, adapter_path, *, phase='p0'):
    spec,root,orders,union=frozen_inputs(spec_path,phase)
    adapter=json.loads(Path(adapter_path).read_text())
    if adapter['frozen_spec_sha256']!=digest(spec):raise ValueError('ADAPTER_SPEC_BINDING')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_ONLY_ENV_REQUIRED')
    work=root/'selected-test-completion';work.mkdir(exist_ok=True)
    identity=dict(spec_sha=digest(spec),adapter_sha=digest(adapter),orders=orders)
    manifest=work/'execution_binding.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('TEST_RESUME_BINDING_CHANGED')
    if not manifest.exists():atomic_json(manifest,identity)
    if (work/'final_audit.json').exists():return json.loads((work/'final_audit.json').read_text())
    # Test is read only after all calibration freezes have been authenticated.
    pids=base_parent_ids(spec['test_parent_csv'],spec.get('test_label_filter'))
    if len(pids)!=spec['expected_test_count']:raise ValueError('BASE_CHANGED')
    with Path(spec['test_parent_csv']).open() as stream:
        parents={}
        for i,r in enumerate(csv.DictReader(stream)):
            pid=r.get('parent_id',r.get('molecule_id'))
            if pid in pids:parents[pid]=dict(id=pid,smiles=r['smiles'],label=r['label'],index=i)
    candidates={r['candidate_id']:r for r in rows(spec['candidate_universe'])}
    _,d,_=load_matrix(spec['saved_test_matrix'],union,expected_parents=pids,subset=True)
    # P1 adopts new P0 pairs, not just the old production20 cache.
    if phase=='p1':
        prior=Path(spec['output_root'])/'p0/selected-test-completion'
        bound=json.loads((prior/'execution_binding.json').read_text())
        if bound['spec_sha']!=digest(spec) or bound['adapter_sha']!=digest(adapter):raise ValueError('P0_CACHE_BINDING')
        pi={p:i for i,p in enumerate(pids)};ci={c:i for i,c in enumerate(union)}
        for path in sorted((prior/'parent-blocks').glob('*.json')):
            b=json.loads(path.read_text())
            validate_new_block(b,{(r['parent_id'],r['candidate_id']) for r in b['pairs']})
            for r in b['pairs']:
                if r['parent_id'] in pi and r['candidate_id'] in ci:
                    i,j=pi[r['parent_id']],ci[r['candidate_id']]
                    value=float(r['wnode_distance']) if r['pair_strict_flip'] else np.inf
                    if not np.isnan(d[i,j]) and d[i,j]!=value:raise ValueError('P0_RAW_CONFLICT')
                    d[i,j]=value
    if spec.get('test_predictions_csv'):
        with Path(spec['test_predictions_csv']).open() as stream:
            predictions={r['parent_id']:r for r in csv.DictReader(stream)}
        for i,pid in enumerate(pids):
            r=predictions[pid]
            if (r['checkpoint_id']!=spec['oracle_sha256'] or r['backbone']!='gine'
                    or float(r['temperature'])!=spec['temperature']):raise ValueError('BEFORE_PREDICTION_CHANGED')
            if int(r['predicted_label'])!=1:
                if np.isfinite(d[i]).any():raise ValueError('NONSOURCE_FINITE')
                d[i]=np.inf
    new_count=0;reuse_count=int((~np.isnan(d)).sum());start=time.monotonic()
    evaluate=provider=None;blocks=work/'parent-blocks';blocks.mkdir(exist_ok=True)
    try:
        for i,pid in enumerate(pids):
            missing=[union[j] for j in np.flatnonzero(np.isnan(d[i]))]
            if not missing:continue
            path=blocks/(hashlib.sha256(pid.encode()).hexdigest()+'.json')
            requested={(pid,c) for c in missing}
            if path.exists():
                block=json.loads(path.read_text());validate_new_block(block,requested)
            else:
                if datetime.now(timezone.utc)>=datetime.fromisoformat(spec['science_cutoff_utc']):raise ValueError('V7_CUTOFF')
                if evaluate is None:evaluate,provider=make_evaluator(spec,adapter,work)
                block=evaluate(parents[pid],[candidates[c] for c in missing])
                validate_new_block(block,requested)
                block['producer']=dict(dataset=spec['dataset'],oracle_sha=spec['oracle_sha256'],device='cpu',
                    oracle_batch_size=256 if spec['dataset']=='BACE' else 'native_RF',adapter_sha=digest(adapter))
                atomic_json(path,block);new_count+=len(missing)
            for r in block['pairs']:
                if r['pair_strict_flip']:
                    value=r.get('wnode_distance')
                    if value is None or not math.isfinite(float(value)):raise ValueError('MISSING_RAW')
                    d[i,union.index(r['candidate_id'])]=float(value)
                else:d[i,union.index(r['candidate_id'])]=np.inf
            atomic_json(work/'progress.json',dict(completed_parent_index=i+1,total_parents=len(pids),
                new_pairs_this_attempt=new_count,saved_raw_pairs_reused=reuse_count,elapsed_seconds=time.monotonic()-start,
                incomplete_slots=int(np.isnan(d).sum())))
    finally:
        if provider is not None:provider.close()
    if np.isnan(d).any():raise ValueError('INCOMPLETE_SELECTED_UNION')
    metrics=[];best_rows=[];idx={c:i for i,c in enumerate(union)}
    for sid,seq in orders.items():
        order=[idx[c] for c in seq];metrics.extend(prefix_rows(d,order,spec['cost_cap'],dataset=spec['dataset'],variant=sid,split='test'))
        best=np.minimum.accumulate(d[:,order],axis=1)
        for i,pid in enumerate(pids):
            for k in range(len(order)):
                best_rows.append(dict(dataset=spec['dataset'],variant=sid,parent_id=pid,k=k+1,
                    raw_best_distance=float(best[i,k]) if np.isfinite(best[i,k]) else 'INF_VERIFIED_FAILURE',
                    covered=bool(best[i,k]<=.1)))
    write_csv(work/'test_prefix.csv',metrics);write_csv(work/'parent_best_distances.csv',best_rows)
    result=dict(state='SELECTED_UNION_EVALUATED_STRUCTURAL_AUDIT_PASS',dataset=spec['dataset'],variants=list(orders),
        selected_union=len(union),base=len(pids),new_pairs_this_attempt=new_count,saved_pairs_reused=reuse_count,
        theta=.1,cost_cap=spec['cost_cap'],independent_numerical_audit='PENDING',
        new_generation=0,selector_rerun=False,elapsed_seconds=time.monotonic()-start,
        outputs=[str(work/'test_prefix.csv'),str(work/'parent_best_distances.csv')])
    atomic_json(work/'final_audit.json',result)
    return result
