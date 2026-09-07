"""Saved-record-only diagnostics for the A+ calibration pool and frozen controls.

No model, candidate generation, selector, distance provider or OT is executed.
Unknown counts remain undefined alongside explicit known sums and gap counts.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

from src.eval.bace_frozen_gnn_contracts import atomic_csv, atomic_json, read_json, stable_sha256, utc_now

LAYERS = ('SOURCE_NOT_1','NO_MATCH','NO_VALID_RESIDUAL','NO_STRICT_FLIP',
          'NO_FINITE_DISTANCE','THETA_EXCEEDED','COVERED','UNKNOWN')
FIELDS = {'num_matches':'num_matches','num_valid_residuals':'num_valid_residuals',
          'strict_flip_matches':'num_strict_flip_matches'}


def count_value(row,key):
    value=row.get(key)
    if value is None:return None
    if type(value) is not int or value<0:raise ValueError('INVALID_SAVED_COUNT:'+key)
    return value


def finite_distance(row):
    if row.get('pair_strict_flip') is False:return None,True
    if row.get('pair_strict_flip') is not True:return None,False
    value=row.get('wnode_distance')
    if type(value) not in (float,int) or not math.isfinite(value) or value<0:return None,False
    if row.get('pred_before')!=1 or row.get('pred_after')!=0:
        raise ValueError('FINITE_PAIR_NOT_ACTUAL_SOURCE_TARGET_FLIP')
    return float(value),True


def first_failure(source,matches,valid,flips,distances,theta):
    if source==0:return 'SOURCE_NOT_1'
    if source!=1:return 'UNKNOWN'
    for value,reason in ((matches,'NO_MATCH'),(valid,'NO_VALID_RESIDUAL'),(flips,'NO_STRICT_FLIP')):
        if value is None:return 'UNKNOWN'
        if value==0:return reason
    if not distances:return 'NO_FINITE_DISTANCE'
    return 'COVERED' if min(distances)<=theta else 'THETA_EXCEEDED'


def summarize_scope(units,*,scope,split,order,theta,cap,expected_parents,expected_candidates):
    """Consume one complete saved parent at a time, with explicit set coverage."""
    sums=Counter();gaps=Counter();pair_failure=Counter();parent_failure=Counter()
    parent_rows=[];seen=set();reference_candidates=None;pair_samples={k:[] for k in LAYERS}
    for unit in units:
        pid=unit['parent_id']
        if pid in seen:raise ValueError('DUPLICATE_PARENT_UNIT')
        seen.add(pid)
        original=unit['pair_rows'];by_id={r['candidate_id']:r for r in original}
        if len(by_id)!=len(original):raise ValueError('DUPLICATE_SAVED_PAIR')
        selected=list(by_id) if order is None else list(order)
        if (len(selected)!=expected_candidates or len(selected)!=len(set(selected))
            or not set(selected)<=by_id.keys()):raise ValueError('SCOPE_CANDIDATE_COVERAGE_GAP')
        if order is None:
            if reference_candidates is None:reference_candidates=set(selected)
            if set(selected)!=reference_candidates:raise ValueError('CALIBRATION_POOL_CHANGED_BETWEEN_PARENTS')
        rows=[by_id[c] for c in selected]
        if any(r.get('parent_id')!=pid or r.get('split')!=split for r in rows):
            raise ValueError('SAVED_PAIR_PARENT_OR_SPLIT_MISMATCH')
        predictions={r.get('pred_before') for r in rows if r.get('pred_before') in (0,1)}
        if len(predictions)>1:raise ValueError('INCONSISTENT_SAVED_SOURCE_PREDICTION')
        source=next(iter(predictions)) if predictions else None
        sums['source_parents']+=source==1;gaps['source_parents']+=source is None
        values={name:[] for name in FIELDS};p_dist=[];distance_unknown=0
        p_app=[];p_flip=[];local_failure=Counter()
        for row in rows:
            sums['pair_count']+=1
            app=row.get('applicable');p_app.append(app if type(app) is bool else None)
            sums['applicable_pairs']+=app is True;gaps['applicable_pairs']+=type(app) is not bool
            flip=row.get('pair_strict_flip');p_flip.append(flip if type(flip) is bool else None)
            sums['strict_flip_pairs']+=flip is True;gaps['strict_flip_pairs']+=type(flip) is not bool
            for name,key in FIELDS.items():
                value=count_value(row,key);values[name].append(value)
                sums[name]+=value or 0;gaps[name]+=value is None
            distance,complete=finite_distance(row)
            if distance is not None:p_dist.append(distance)
            distance_unknown+=not complete
            sums['finite_pairs']+=distance is not None;gaps['finite_pairs']+=not complete
            sums['theta_pairs']+=distance is not None and distance<=theta;gaps['theta_pairs']+=not complete
            sums['cap_pairs']+=distance is not None and distance<=cap;gaps['cap_pairs']+=not complete
            failure=first_failure(row.get('pred_before'),values['num_matches'][-1],
                values['num_valid_residuals'][-1],values['strict_flip_matches'][-1],
                [] if distance is None else [distance],theta)
            pair_failure[failure]+=1;local_failure[failure]+=1
            if len(pair_samples[failure])<10:pair_samples[failure].append({'parent_id':pid,'candidate_id':row['candidate_id']})
        totals={key:sum(x for x in vals if x is not None) if all(x is not None for x in vals) else None
                for key,vals in values.items()}
        failure=first_failure(source,totals['num_matches'],totals['num_valid_residuals'],
            totals['strict_flip_matches'],p_dist,theta)
        parent_failure[failure]+=1
        for name,flags in (('parents_with_applicable',p_app),('parents_with_strict_flip',p_flip)):
            known_true=any(x is True for x in flags);unknown=not known_true and any(x is None for x in flags)
            sums[name]+=known_true;gaps[name]+=unknown
        for name,test in (('parents_with_matches',totals['num_matches']),
                          ('parents_with_valid_residuals',totals['num_valid_residuals'])):
            vals=values['num_matches' if name=='parents_with_matches' else 'num_valid_residuals']
            positive=any(v is not None and v>0 for v in vals)
            sums[name]+=positive;gaps[name]+=not positive and test is None
        for name,threshold in (('finite_parents',math.inf),('theta_parents',theta),('cap_parents',cap)):
            reached=any(d<=threshold for d in p_dist)
            sums[name]+=reached;gaps[name]+=not reached and distance_unknown>0
        parent_rows.append({'scope':scope,'split':split,'parent_id':pid,'pred_before':source,
            'pair_count':len(rows),**totals,'finite_pairs_available':len(p_dist),
            'unknown_pair_distances':distance_unknown,'best_available_distance':min(p_dist) if p_dist else None,
            'theta_reached':any(v<=theta for v in p_dist) if p_dist or not distance_unknown else None,
            'cap_reached':any(v<=cap for v in p_dist) if p_dist or not distance_unknown else None,
            'first_failure':failure,'pair_first_failure_counts':dict(local_failure)})
    if len(seen)!=expected_parents:raise ValueError('PARENT_DENOMINATOR_GAP')
    if sums['pair_count']!=expected_parents*expected_candidates:raise ValueError('FULL_SCOPE_CARTESIAN_GAP')
    output={'method':'Ours-GIN-Aplus','scope':scope,'split':split,'parent_denominator':expected_parents,
        'candidate_count_in_scope':expected_candidates,'pair_count':sums['pair_count'],
        'theta_star':theta,'cost_cap':cap,'test_full2659_evaluated':False if split=='test' else None}
    for name in ('source_parents','applicable_pairs','num_matches','num_valid_residuals','strict_flip_matches',
                 'strict_flip_pairs','finite_pairs','theta_pairs','cap_pairs','parents_with_applicable',
                 'parents_with_matches','parents_with_valid_residuals','parents_with_strict_flip',
                 'finite_parents','theta_parents','cap_parents'):
        output[name]=sums[name] if gaps[name]==0 else None
        output[name+'_known_sum']=sums[name];output[name+'_unknown_count']=gaps[name]
    failures=[{'scope':scope,'split':split,'first_failure':k,'pair_count':pair_failure[k],
        'parent_count':parent_failure[k],'pair_examples_first10':pair_samples[k]} for k in LAYERS]
    return output,failures,parent_rows


def summarize_many(units,scopes,*,split,theta,cap,expected_parents,source_candidates=None):
    """Read each potentially large parent container once, reduce all views."""
    collected={};seen=set();pool=source_candidates
    for unit in units:
        if unit['parent_id'] in seen:raise ValueError('DUPLICATE_PARENT_UNIT')
        seen.add(unit['parent_id'])
        ids={r['candidate_id'] for r in unit['pair_rows']}
        if pool is None:pool=ids
        if ids!=pool:raise ValueError('FULL_SOURCE_POOL_OR_FROZEN_UNION_CHANGED')
        for name,order,count in scopes:
            m,f,p=summarize_scope([unit],scope=name,split=split,order=order,theta=theta,cap=cap,
                expected_parents=1,expected_candidates=count)
            if name not in collected:
                collected[name]=(m,{r['first_failure']:r for r in f},p)
                continue
            total,failures,parents=collected[name]
            total['parent_denominator']+=1;total['pair_count']+=m['pair_count']
            for key,value in m.items():
                if key.endswith(('_known_sum','_unknown_count')):total[key]+=value
            for row in f:
                old=failures[row['first_failure']]
                old['pair_count']+=row['pair_count'];old['parent_count']+=row['parent_count']
                old['pair_examples_first10']=(old['pair_examples_first10']+row['pair_examples_first10'])[:10]
            parents.extend(p)
    if len(seen)!=expected_parents:raise ValueError('PARENT_DENOMINATOR_GAP')
    metrics=[];failures=[];parents=[]
    for name,_,_ in scopes:
        m,f,p=collected[name]
        for key in list(m):
            if key.endswith('_known_sum'):
                field=key[:-10];m[field]=m[key] if m[field+'_unknown_count']==0 else None
        metrics.append(m);failures.extend(f.values());parents.extend(p)
    return metrics,failures,parents


def export(spec,output):
    from src.experiments.bace_gin_reach_v2 import verified,_iter_units,campaign_group
    from src.experiments.bace_gin_reach_test_raw import validate_aplus_freeze
    from src.experiments.bace_gin_fixed_pool import bound_json
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    root=Path(spec['output_root']);destination=Path(output)
    if not destination.is_absolute() or destination==root or destination in root.parents:
        raise ValueError('FRESH_SCOPED_FUNNEL_OUTPUT_REQUIRED')
    contract=verified(root/'contract.json')
    if contract['spec_sha256']!=stable_sha256(spec):raise ValueError('ACTUAL_CAMPAIGN_CONTRACT_REQUIRED')
    group=campaign_group(spec);n=contract['adopted_candidate_count']
    thresholds=threshold_bundle_from_dict(bound_json(spec['thresholds']))
    def terminal(split):
        value=verified(root/group/split/'terminal.json')
        if (value.get('state')!='PARENT_EVALUATION_COMPLETE' or value['spec_sha256']!=stable_sha256(spec)
            or value['parent_count']!=spec['base_counts'][split]):raise ValueError('COMPLETE_BOUND_SPLIT_REQUIRED:'+split)
        return value
    cal=terminal('calibration')
    if cal['candidate_count']!=n:raise ValueError('COMPLETE_CALIBRATION_POOL_REQUIRED')
    test_scopes=[]
    test_state='PENDING_NO_COMPLETE_TEST_TERMINAL';freeze=None
    if (root/group/'test/terminal.json').exists():
        # This exact new freeze is checked before a single test parent file.
        freeze=verified(root/'selection_freeze.json')
        validate_aplus_freeze(freeze,spec=spec,evidence_root=root)
        test=terminal('test')
        union=set().union(*(set(v) for v in freeze['controls'].values()))
        if test['candidate_count']!=len(union):raise ValueError('TEST_CONTROL_UNION_COUNT_CONFLICT')
        for name,order in freeze['controls'].items():
            for k in (10,20):test_scopes.append((f'{name}_K{k}',order[:k],min(k,len(order))))
        test_state='THREE_FROZEN_CONTROLS_K10_K20_NOT_FULL2659'
    metrics,failures,parents=summarize_many(_iter_units(spec,group,'calibration'),
        [('calibration_full_pool',None,n)],split='calibration',theta=thresholds.theta_star,cap=thresholds.cost_cap,
        expected_parents=spec['base_counts']['calibration'])
    if test_scopes:
        more,stages,detail=summarize_many(_iter_units(spec,group,'test'),test_scopes,
            split='test',theta=thresholds.theta_star,cap=thresholds.cost_cap,
            expected_parents=spec['base_counts']['test'],source_candidates=union)
        metrics.extend(more);failures.extend(stages);parents.extend(detail)
    destination.mkdir(parents=True,exist_ok=False)
    atomic_csv(destination/'method_funnel.csv',metrics)
    atomic_csv(destination/'first_failure_funnel.csv',failures)
    atomic_csv(destination/'parent_failure_details.csv',parents)
    receipt={'state':'SAVED_RECORD_FUNNEL_EXPORTED','source_root':str(root),'spec_sha256':stable_sha256(spec),
        'calibration_pool_count':n,'calibration_parent_count':spec['base_counts']['calibration'],
        'test_state':test_state,'test_full_pool_claimed':False,'model_inference':False,'ot_computed':0,
        'unknown_counts':'null/blank with explicit known_sum and unknown_count; never filled as 0',
        'first_failure_policy':list(LAYERS),'distance_gap_is_not_valid_zero_result':True,
        'audit_scope':'SAVED_PAIR_RECORD_FUNNEL_NOT_MODEL_OR_CHEMISTRY_REEXECUTION',
        'large_parent_container_read_passes_per_split':1,
        'main_matrix_write':False,'created_at':utc_now()}
    atomic_json(destination/'funnel_manifest.json',receipt)
    return receipt
