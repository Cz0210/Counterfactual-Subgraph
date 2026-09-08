"""Independent saved-record reduction audit and small BACE migration package.

No model loading, OT, generation, selectors, or main-matrix publication.
Original parent records remain the scientific source; the portable package
contains their hash inventory, frozen contracts, CSVs, and this audit.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import fcntl
import math
from pathlib import Path
import statistics
import tarfile

from src.experiments.bace_eval_migration import reopen, seal
from src.eval.bace_frozen_gnn_contracts import read_json, sha256_file, stable_sha256, utc_now


def close(actual, expected, name):
    """CSV arithmetic closure only; not a relaxed model/OT parity tolerance."""
    if expected is None:
        if actual not in (None, ''):
            raise ValueError('EXPECTED_NA:'+name)
    elif actual in (None, '') or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-14):
        raise ValueError(f'CSV_REDUCTION:{name}:{actual}!={expected}')


def inspect_parent(record, candidate_ids, model_sha):
    pairs=record['pair_rows']; matches=defaultdict(list)
    for row in record['match_rows']:
        if row['parent_id'] != record['parent_id'] or row['oracle_checkpoint_hash'] != model_sha:
            raise ValueError('MATCH_PARENT_OR_MODEL_BINDING')
        strict=bool(row['delete_valid'] and row['residual_connected'] and
                    row['pred_before']==1 and row['pred_after']==0)
        if strict != row['teacher_strict_flip'] or strict != row['cf_flip']:
            raise ValueError('MATCH_STRICT_FLIP')
        if strict:
            distance=row['wnode_distance']
            if not row['distance_ok'] or not isinstance(distance,(int,float)) or not math.isfinite(distance) or distance<0:
                raise ValueError('STRICT_FLIP_RAW_DISTANCE_GAP')
            matches[row['candidate_id']].append(row)
    if len(pairs)!=len(candidate_ids) or [r['candidate_id'] for r in pairs] != candidate_ids:
        raise ValueError('PAIR_POOL_ORDER_OR_COMPLETENESS')
    predictions={r['pred_before'] for r in pairs}
    if len(predictions)!=1:
        raise ValueError('INCONSISTENT_PARENT_PREDICTION')
    result={}
    for pair in pairs:
        if pair['parent_id'] != record['parent_id'] or pair['oracle_checkpoint_hash'] != model_sha:
            raise ValueError('PAIR_PARENT_OR_MODEL_BINDING')
        eligible=matches[pair['candidate_id']]
        best=min(eligible,key=lambda r:(r['wnode_distance'],-r['cf_drop'],tuple(r['match_atom_indices']))) if eligible else None
        if pair['pair_strict_flip'] != bool(best):
            raise ValueError('PAIR_FLIP_REDUCTION')
        if best:
            if not pair['applicable'] or any(pair[k]!=best[k] for k in ('wnode_distance','pred_after','residual_smiles','cf_drop')):
                raise ValueError('PAIR_BEST_MATCH_REDUCTION')
            if pair['best_match_atom_indices'] != best['match_atom_indices']:
                raise ValueError('PAIR_MATCH_MAPPING')
        elif pair['wnode_distance'] is not None:
            raise ValueError('FAILED_PAIR_MUST_NOT_HAVE_RAW_COST')
        result[pair['candidate_id']]=best['wnode_distance'] if best else math.inf
    return next(iter(predictions)),result


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def audit(spec_path, output):
    spec=read_json(spec_path); root=Path(spec['output_root']); output=Path(output).resolve()
    output.relative_to(root.parent.resolve())
    if output==root.resolve() or root.resolve() in output.parents:
        raise ValueError('FRESH_SIBLING_ACCEPTANCE_REQUIRED')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'acceptance.json').exists():
        return reopen(output/'acceptance.json')
    with (root/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=reopen(root/'final_audit.json'); freeze=reopen(root/'CALIBRATION_FREEZE.json')
        identity=stable_sha256(spec)
        if final['state']!='EVALUATION_COMPLETE' or final['spec_sha256']!=identity or freeze['spec_sha256']!=identity or final['freeze_sha256']!=freeze['self_sha256'] or freeze['test_loaded'] is not False:
            raise ValueError('FINAL_FREEZE_SPEC_BINDING')
        if any(spec[k] is not False for k in ('training','temperature_fit','generation','main_matrix_write')):
            raise ValueError('EVALUATION_ONLY_SCOPE')
        for name,digest in final['source_csv'].items():
            if sha256_file(root/'source_csv'/name)!=digest:
                raise ValueError('CSV_BINDING:'+name)
        thresholds=read_json(spec['thresholds']['path'])
        if sha256_file(spec['thresholds']['path'])!=spec['thresholds']['sha256'] or thresholds['test_used'] is not False:
            raise ValueError('FROZEN_THRESHOLD_BINDING')
        inventory=[]; data={}; before={}; cal={}; reuse=None
        if spec['family']=='llm_gin':
            binding=reopen(root/'RAW_RECONCILIATION_BINDING.json')
            reuse=reopen(Path(binding['overlay_root'])/'reconciliation.json')
            if binding['freeze_sha256']!=freeze['self_sha256'] or reuse['calibration_units_reused']!=264 or reuse['calibration_values_changed'] or reuse['freeze_changed']:
                raise ValueError('CORRECTIVE_CALIBRATION_OR_FREEZE_CHANGED')
        for role,cfg in spec['roles'].items():
            with Path(cfg['pool']['path']).open() as f:
                import json
                pool=[json.loads(line)['candidate_id'] for line in f if line.strip()]
            if sha256_file(cfg['pool']['path'])!=cfg['pool']['sha256']:
                raise ValueError('POOL_BINDING')
            before[role]={};data[role]={};cal[role]={}
            for split,count in (('calibration',66),('test',141)):
                directory=root/role/split; terminal=reopen(directory/'terminal.json')
                if terminal['state']!='PARENT_EVALUATION_COMPLETE' or terminal['spec_sha256']!=identity or terminal['parent_count']!=count:
                    raise ValueError('TERMINAL_COMPLETENESS')
                selected=set().union(*(set(x['ordered_rule_ids']) for x in freeze['selectors'][role].values()))
                candidates=pool if split=='calibration' else [x for x in pool if x in selected]
                ids=[]
                for i in range(count):
                    path=directory/f'parent-{i:04d}.json'; record=reopen(path)
                    pred,values=inspect_parent(record,candidates,cfg['model_files']['model.pt'])
                    ids.append(record['parent_id'])
                    inventory.append(dict(path=str(path),self_sha256=record['self_sha256'],role=role,split=split,parent_id=record['parent_id']))
                    if split=='test':
                        if reuse and record.get('raw_reconciliation_sha256')!=binding['index_sha256']:
                            raise ValueError('TEST_CORRECTIVE_INDEX_BINDING')
                        data[role][record['parent_id']]=values;before[role][record['parent_id']]=pred
                    else:
                        cal[role][record['parent_id']]=pred
                if len(set(ids))!=count:
                    raise ValueError('DUPLICATE_PARENT')
        for split,source in (('test',before),('calibration',cal)):
            orders=[list(v) for v in source.values()]
            if any(ids!=orders[0] for ids in orders):
                raise ValueError('CROSS_ROLE_BASE_COHORT:'+split)
        common=set.intersection(*({p for p,pred in v.items() if pred==1} for v in before.values()))
        cal_common=set.intersection(*({p for p,pred in v.items() if pred==1} for v in cal.values()))
        for role,modes in freeze['selectors'].items():
            for mode,selected in modes.items():
                expected=list(cal[role]) if mode=='fixed141' else [p for p,pred in cal[role].items() if (pred==1 if mode=='native' else p in cal_common)]
                order=selected['ordered_rule_ids']
                if selected['calibration_parent_ids']!=expected or selected['test_loaded'] is not False or len(order)>20 or len(order)!=len(set(order)):
                    raise ValueError('CALIBRATION_SELECTION_SCOPE')
        distances={}
        for role,modes in freeze['selectors'].items():
            for mode,selection in modes.items():
                ids=list(before[role]) if spec['family']=='llm_gin' else [p for p,pred in before[role].items() if (pred==1 if mode=='native' else p in common)]
                for k in range(1,21):
                    order=selection['ordered_rule_ids'][:k]
                    for pid in ids:
                        distances[role,mode,k,pid]=(min((data[role][pid][c] for c in order),default=math.inf),len(order))
        seen=set()
        for row in read_csv(root/'source_csv/parent_distances.csv'):
            key=row['role'],row['selection_cohort'],int(row['K_requested']),row['parent_id']
            if key in seen or key not in distances:
                raise ValueError('PARENT_CSV_KEY')
            seen.add(key);value,n=distances[key]
            close(row['best_valid_distance'],value if math.isfinite(value) else None,'parent_min')
            if int(row['K_effective'])!=n or int(row['pred_before'])!=before[key[0]][key[3]]:
                raise ValueError('PARENT_CSV_EFFECTIVE_K_OR_PRED')
        if seen!=set(distances): raise ValueError('PARENT_CSV_COMPLETENESS')
        metrics=read_csv(root/'source_csv/prefix_metrics.csv');metric_keys=set()
        for row in metrics:
            key=row['role'],row['selection_cohort'],int(row['K_requested'])
            full=[(pid,v) for (r,m,k,pid),(v,_) in distances.items() if (r,m,k)==key]
            values=[v for pid,v in full if row['cohort']!='gin_native' or before[key[0]][pid]==1]
            n=len(values);finite=[v for v in values if math.isfinite(v)]
            theta=float(row['theta_star']);cap=float(row['cost_cap'])
            if theta!=thresholds['theta_star'] or cap!=thresholds['cost_cap']:
                raise ValueError('FROZEN_THRESHOLD_OR_CAP_CHANGED')
            if int(row['K_effective'])!=min(key[2],len(freeze['selectors'][key[0]][key[1]]['ordered_rule_ids'])):
                raise ValueError('METRIC_EFFECTIVE_K')
            covered=sum(v<=theta for v in values)
            if int(row['denominator'])!=n or int(row['covered_count'])!=covered or int(row['finite_strict_flip_count'])!=len(finite):
                raise ValueError('METRIC_COUNTS')
            for field,expected in [('coverage',covered/n if n else None),('strict_flip_availability',len(finite)/n if n else None),('fixed_capped_mean',math.fsum(min(v,cap) for v in values)/n if n else None),('conditional_median',statistics.median(finite) if finite else None)]:
                close(row[field],expected,field)
            mk=(*key,row['cohort'])
            if mk in metric_keys: raise ValueError('DUPLICATE_METRIC')
            metric_keys.add(mk)
        expected_metrics=200 if spec['family']=='gnn_a' else 160
        if len(metric_keys)!=expected_metrics: raise ValueError('PREFIX_COMPLETENESS')
        grids=defaultdict(list)
        for row in read_csv(root/'source_csv/exact_ecdf.csv'):
            key=row['role'],row['selection_cohort'],int(row['K_requested'])
            vals=[v for (r,m,k,p),(v,_) in distances.items() if (r,m,k)==key and (row['cohort']!='gin_native' or before[r][p]==1)]
            count=sum(v<=float(row['threshold']) for v in vals)
            if int(row['denominator'])!=len(vals) or int(row['covered_count'])!=count:
                raise ValueError('EXACT_ECDF_COUNT')
            close(row['coverage'],count/len(vals) if vals else None,'ECDF')
            grids[(*key,row['cohort'])].append(float(row['threshold']))
        endpoints=[float(r['threshold']) for r in thresholds['raw_quantile_thresholds']]
        for key in metric_keys:
            role,mode,k,cohort=key
            if k not in (10,20):continue
            vals=[v for (r,m,kk,p),(v,_) in distances.items() if (r,m,kk)==(role,mode,k) and (cohort!='gin_native' or before[r][p]==1)]
            expected=sorted({0.,thresholds['theta_star'],*endpoints,*(v for v in vals if math.isfinite(v) and v<=max(endpoints))})
            if grids[key]!=expected:raise ValueError('EXACT_ECDF_GRID_COMPLETENESS')
        if read_csv(root/'source_csv/table_k10.csv') != [r for r in metrics if int(r['K_requested'])==10]:
            raise ValueError('TABLE_CURVE_DISAGREEMENT')
        seal(output/'parent_inventory.json',dict(parent_units=inventory))
        return seal(output/'acceptance.json',dict(state='SAVED_RECORD_AND_METRIC_ACCEPTANCE_PASS',
            audit_scope='ALL_SAVED_MATCH_PAIR_PARENT_REDUCTIONS_AND_CSV_NOT_MODEL_REEXECUTION',
            spec_sha256=identity,source_final_sha256=final['self_sha256'],freeze_sha256=freeze['self_sha256'],
            family=spec['family'],scope=spec['scope'],calibration_units=66*len(spec['roles']),
            test_units=141*len(spec['roles']),common_test_count=len(common),
            source_root=str(root),source_spec=str(Path(spec_path).resolve()),
            native_test_counts={r:sum(v==1 for v in p.values()) for r,p in before.items()},
            prefix_metric_rows=len(metrics),csv_arithmetic_abs_tolerance=1e-14,
            model_inference_rerun=False,ot_recomputed=0,generation=False,training=False,
            main_matrix_write=False,created_at=utc_now()))


def package(spec_path, output):
    output=Path(output); accepted=audit(spec_path,output);root=Path(accepted['source_root'])
    files={'acceptance.json':output/'acceptance.json','parent_inventory.json':output/'parent_inventory.json',
           'spec.json':Path(spec_path),'CALIBRATION_FREEZE.json':root/'CALIBRATION_FREEZE.json',
           'source_final_audit.json':root/'final_audit.json'}
    files.update({'source_csv/'+p.name:p for p in (root/'source_csv').glob('*.csv')})
    if (root/'RAW_RECONCILIATION_BINDING.json').exists():
        b=reopen(root/'RAW_RECONCILIATION_BINDING.json')
        files['raw_reconciliation_binding.json']=root/'RAW_RECONCILIATION_BINDING.json'
        for name in ('scan.json','producer.json','reconciliation.json'):
            files['raw_reconciliation/'+name]=Path(b['overlay_root'])/name
    manifest=seal(output/'package_manifest.json',dict(files={k:sha256_file(p) for k,p in files.items()},
        source_records_external=True,source_records_retained=True,models_included=False,main_matrix_write=False))
    files['package_manifest.json']=output/'package_manifest.json'
    target=output/'result_package.tar.gz'
    if not target.exists():
        tmp=output/'result_package.tar.gz.partial'
        with tarfile.open(tmp,'w:gz') as tar:
            for name,path in files.items():tar.add(path,arcname=name,recursive=False)
        tmp.replace(target)
    return seal(output/'result_package.json',dict(state='PACKAGE_COMPLETE',path=str(target),bytes=target.stat().st_size,
        sha256=sha256_file(target),acceptance_sha256=accepted['self_sha256'],manifest_sha256=manifest['self_sha256']))
