"""Finite one-round Taste search -> delta matrices -> four frozen variants.

This is a dataset-specific stage inside the existing one-process GPU lease,
not a scheduler. It never starts another task or changes the main authority.
"""
from collections import Counter
from datetime import datetime,timezone
import csv,gzip,hashlib,json,os,time
from pathlib import Path
import numpy as np
from .ours_taste_focus_matrix import read_json,dump_json
from .ours_taste_focus_selector import ceilings,Selector,prefix,write_csv
from .ours_taste_search import search_parent


def append_segment(path,rows):
    with path.open('ab') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
            for row in rows:gz.write((json.dumps(row,sort_keys=True)+'\n').encode())
        raw.flush();os.fsync(raw.fileno())


def augment(original,additions):
    by_id={r['candidate_id']:dict(r,source_labels=['OLD_POOL']) for r in original}
    new={}
    for row in additions:
        cid=row['candidate_id']
        target=by_id.get(cid) or new.setdefault(cid,dict(row,source_parent_ids=[],source_modes=[],train_strict_flip_witnesses=0))
        for key in ('source_parent_ids','source_modes'):
            target[key]=sorted(set(target.get(key,[]))|set(row.get(key,[])))
        target['train_strict_flip_witnesses']=target.get('train_strict_flip_witnesses',0)+row['train_strict_flip_witnesses']
    # Frozen before calibration/test: train witness/support, small size, provenance, ID.
    ordered=sorted(new.values(),key=lambda r:(-r['train_strict_flip_witnesses'],-len(r['source_parent_ids']),
                         r['train_deletion_size'],tuple(r['source_modes']),r['candidate_id']))[:2048]
    return list(by_id.values())+ordered,{'new_unique_seen':len(new),'new_unique_retained':len(ordered),
            'old_retained':len(original),'pool_cut_uses_test':False,'pool_cut_uses_calibration':False}


def check_resource(spec):
    from src.ablations.llm.existing_gpu_owner import memory_headroom
    assert memory_headroom(Path('/proc'),Path(spec['cgroup_memory_root']))>=spec['runtime_safety_bytes'], 'RAM_BOUNDARY'
    assert os.statvfs(spec['output_root']).f_favail>=8192, 'SLOT_BOUNDARY'


def evaluate_delta(parents,candidates,split,scorer,provider,identity,out,pause,deadline,spec):
    from .tastemolnet_ours_full import evaluate_parent,TrainParent
    d=np.full((len(parents),len(candidates)),np.nan)
    predictions=np.full(len(parents),-1,dtype=np.int8)
    start=time.monotonic()
    for i,parent in enumerate(parents):
        if pause[0] or datetime.now(timezone.utc)>=deadline:raise RuntimeError('PAUSED_AT_COMPLETE_PARENT_BOUNDARY')
        check_resource(spec)
        rows=evaluate_parent(parent=TrainParent(parent['parent_id'],parent['smiles'],1,split),
                     candidates=candidates,scorer=scorer,distance=provider,split=split,evaluation_identity=identity)
        if rows:
            predictions[i]=rows[0]['pred_before']
            for j,row in enumerate(rows):d[i,j]=row['wnode_distance'] if row['pair_strict_flip'] else np.inf
            append_segment(out/f'{split}_new_pairs.jsonl.gz',rows)
        with (out/f'{split}_new_matrix.npz.tmp').open('wb') as f:
            np.savez_compressed(f,distances=d,predictions=predictions,parents=np.asarray([p['parent_id'] for p in parents]),
                           candidates=np.asarray([c['candidate_id'] for c in candidates]));f.flush();os.fsync(f.fileno())
        (out/f'{split}_new_matrix.npz.tmp').replace(out/f'{split}_new_matrix.npz')
        provider.embedder.commit()
        dump_json(out/'progress.json',{'state':f'{split.upper()}_DELTA_MATRIX','pid':os.getpid(),
                   'completed_parents':i+1,'parent_count':len(parents),'candidate_count':len(candidates),
                   'elapsed_seconds':time.monotonic()-start,'observed_at':datetime.now(timezone.utc).isoformat()})
    return d,predictions


def run(spec,out,compact,contract,scorer,provider,identity,pause):
    from rdkit import Chem
    from .tastemolnet_ours_full import load_prepared_split,select_on_calibration
    train=Path(spec['train_matrix_root'])
    assert read_json(train/'terminal.json')['state']=='TRAIN_P0_MATRIX_COMPLETE'
    cohort=read_json(train/'development_cohort.json')['parents']
    original=read_json(compact/'candidate_pool.json');ids0=[r['candidate_id'] for r in original]
    z=np.load(train/'train.npz',allow_pickle=False);d0=z['distances']
    assert not np.isnan(d0).any() and z['parents'].tolist()==[p['parent_id'] for p in cohort]
    theta,cap=contract['theta_star'],contract['cost_cap']
    train_bounds=ceilings(d0,z['predictions']==1,theta)
    dump_json(out/'train_P0_bounds.json',train_bounds)
    rank=[]
    for i,parent in enumerate(cohort):
        best=np.min(d0[i]);category=0 if not np.isfinite(best) else 1 if best>theta else 2
        rank.append((category,parent['scaffold'],parent['parent_id'],i))
    # Round-robin scaffold groups inside each confirmed difficulty category.
    chosen=[]
    for category in (0,1,2):
        groups={}
        for cat,scaf,pid,i in sorted(rank):
            if cat==category:groups.setdefault(scaf,[]).append(i)
        while groups and len(chosen)<128:
            for scaf in sorted(list(groups)):
                chosen.append(groups[scaf].pop(0))
                if not groups[scaf]:del groups[scaf]
                if len(chosen)==128:break
    dump_json(out/'round1_train_search_freeze.json',{'parent_ids':[cohort[i]['parent_id'] for i in chosen],
              'method':'difficulty_then_scaffold_round_robin_ID','max_queries_per_parent':64,'beam':8,
              'rounds_planned':1,'second_round_automatic':False,'lm_new_outputs':0,
              'scope':'SAVED_LM_SEEDED_PLUS_STRUCTURE_SEARCH','pool_selection_before_calibration':True,
              'reason_one_round':'bounded first-round implementation; preserve final evaluation window, no performance-conditioned test choice'})
    additions=[];queries=0;deadline=datetime.fromisoformat(spec['search_stop_at'])
    for number,i in enumerate(chosen):
        if pause[0] or datetime.now(timezone.utc)>=deadline:break
        check_resource(spec)
        candidates,events,ledger=search_parent(cohort[i],original,scorer)
        queries+=ledger['oracle_queries'];assert queries<=8192
        additions.extend(candidates);append_segment(out/'search_events.jsonl.gz',events)
        dump_json(out/'budget_ledger.json',{'round':1,'completed_search_parents':number+1,'lm_new_outputs':0,
              'search_oracle_queries':queries,'search_oracle_queries_authorized_max':16384,'shrink_oracle_queries':0,
              'oracle_query_unit':'new unique residual graphs scored; exact same-residual hits counted separately',
              'matrix_queries_separate':True,'historical_lm_attempt_records':30584})
        dump_json(out/'progress.json',{'state':'TRAIN_SEARCH','pid':os.getpid(),'completed_parents':number+1,
                     'total_parents':len(chosen),'oracle_queries':queries,'observed_at':datetime.now(timezone.utc).isoformat()})
    if pause[0]:raise RuntimeError('PAUSED_AFTER_SEARCH_PARENT')
    p1,summary=augment(original,additions);delta=p1[len(original):]
    dump_json(out/'candidate_pool_P1.json',p1)
    dump_json(out/'pool_P1_freeze.json',{**summary,'state':'TRAIN_ONLY_POOL_FROZEN','search_assisted':True,
              'lm_generated_new_rules':False,'new_ppo_updates':0,'test_read':False,'source_pool_count':len(original)})
    if not delta:
        dump_json(out/'terminal.json',{'state':'NO_NEW_POOL_REPORT_A_B_ONLY','new_unique_candidates':0,
             'search_oracle_queries':queries,'C_D_state':'NOT_CREATED','reason':'No new unique rule from the bounded train search',
             'main_matrix_write':False})
        return
    end=datetime.fromisoformat(spec['campaign_end_at'])
    train_delta,_=evaluate_delta(cohort,delta,'train',scorer,provider,identity,out,pause,end,spec)
    train_d1=np.hstack([d0,train_delta])
    dump_json(out/'train_reach_change.json',{'parents':len(cohort),
       'P0_cov':int((d0<=theta).any(1).sum()),'P1_cov':int((train_d1<=theta).any(1).sum()),
       'P0_finite':int(np.isfinite(d0).any(1).sum()),'P1_finite':int(np.isfinite(train_d1).any(1).sum())})
    cal_parents=load_prepared_split(Path(contract['calibration_path']),expected_split='calibration',expected_sha256=contract['calibration_sha256'])
    cal_rows=[{'parent_id':p.parent_id,'smiles':p.smiles} for p in cal_parents]
    cal0=np.load(compact/'calibration.npz',allow_pickle=False)
    cal_delta,pred=evaluate_delta(cal_rows,delta,'calibration',scorer,provider,identity,out,pause,end,spec)
    if delta:assert np.array_equal(pred,cal0['predictions']), 'CALIBRATION_SOURCE_CHANGED'
    cal=np.hstack([cal0['distances'],cal_delta]);ids=[r['candidate_id'] for r in p1]
    np.savez_compressed(out/'calibration_P1.npz',distances=cal,parents=cal0['parents'],candidates=np.asarray(ids),predictions=cal0['predictions'])
    bounds=ceilings(cal,cal0['predictions']==1,theta);dump_json(out/'calibration_P1_bounds.json',bounds)
    old_A=read_json(compact/'original_selection.json')['ordered_rule_ids']
    old_B=read_json(Path(spec['selection_B_path']))['candidate_ids']
    # Original function receives precisely its minimal semantic matrix inputs.
    records=[{'candidate_id':cid,'parent_id':parent,'split':'calibration','pair_strict_flip':bool(np.isfinite(cal[i,j])),
              'wnode_distance':float(cal[i,j])} for i,parent in enumerate(cal0['parents']) for j,cid in enumerate(ids)]
    selected_C,_=select_on_calibration(p1,records,theta_star=theta);del records
    c=[ids.index(r['candidate_id']) for r in selected_C]
    sizes=[Chem.MolFromSmiles(r['canonical_fragment']).GetNumHeavyAtoms() for r in p1]
    selector=Selector(cal,ids,theta,cap,contract['theta_grid'],sizes)
    d,trace=selector.optimize(c,[x['selected'] for x in bounds.values()])
    sequences={'A':[ids.index(x) for x in old_A],'B':[ids.index(x) for x in old_B],'C':c,'D':d}
    recommend=min(('B','C','D'),key=lambda v:selector.key(sequences[v]))
    freeze={'state':'ALL_VARIANTS_FROZEN_BEFORE_NEW_TEST','ordered_ids':{v:[ids[i] for i in s] for v,s in sequences.items()},
             'recommended_on_calibration':recommend,'selector_D_trace':trace,'test_used_for_selection':False,
             'test_previously_observed':True,'scope':'POST_HOC_SEARCH_ASSISTED_FIXED_ORACLE','P0_count':len(original),'P1_count':len(p1)}
    dump_json(out/'selection_freeze_manifest.json',freeze)
    write_csv(out/'fourway_calibration.csv',[{'variant':v,**row} for v,s in sequences.items() for row in prefix(cal,s,theta,cap)])
    # First and only new test pass, after all four sequences and recommendation.
    test_parents=load_prepared_split(Path(contract['test_path']),expected_split='test',expected_sha256=contract['declared_test_sha256'])
    test_rows=[{'parent_id':p.parent_id,'smiles':p.smiles} for p in test_parents]
    union=sorted({i for seq in sequences.values() for i in seq});assert len(union)<=80
    missing=[i for i in union if ids[i] not in old_A]
    test_new,pred=evaluate_delta(test_rows,[p1[i] for i in missing],'test',scorer,provider,identity,out,pause,end,spec)
    test_old=np.load(compact/'test.npz',allow_pickle=False)
    assert test_old['parents'].tolist()==[p.parent_id for p in test_parents]
    if missing:assert np.array_equal(pred,test_old['predictions']), 'TEST_SOURCE_CHANGED'
    combined=np.full((len(test_rows),len(union)),np.nan)
    for j,i in enumerate(union):combined[:,j]=test_old['distances'][:,old_A.index(ids[i])] if ids[i] in old_A else test_new[:,missing.index(i)]
    assert not np.isnan(combined).any()
    np.savez_compressed(out/'test_selected_union.npz',distances=combined,parents=test_old['parents'],candidates=np.asarray([ids[i] for i in union]),predictions=test_old['predictions'])
    metrics=[{'variant':v,**r} for v,seq in sequences.items() for r in prefix(combined,[union.index(i) for i in seq],theta,cap)]
    write_csv(out/'test_fourway_metrics.csv',metrics)
    write_csv(out/'table2_test_K20.csv',[r for r in metrics if r['k']==20])
    dump_json(out/'terminal.json',{'state':'FOURWAY_SCIENCE_COMPLETE_AWAITING_INDEPENDENT_EXPORT_AUDIT',
            'parent_count':len(test_rows),'selected_union_rules':len(union),'new_test_rules':len(missing),
            'recommended':recommend,'new_unique_candidates':len(delta),'search_oracle_queries':queries,
            'lm_new_outputs':0,'main_matrix_write':False,'oracle_changed':False,'scope':freeze['scope']})
