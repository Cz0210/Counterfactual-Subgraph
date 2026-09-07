"""Map completed A+ saved metrics and unchanged V1 baselines to one renderer.

No selector, oracle or OT. The predeclared control is read from the frozen spec,
never picked by test performance. Missing A+ test/Global results stay PENDING.
"""
from pathlib import Path
import json
from src.eval.bace_frozen_gnn_contracts import sha256_file, stable_sha256
from src.experiments.bace_gin_reporting import INPUTS, METHODS, read_csv, write_csv, derive, render

LABELS = {"Ours":"Ours-Reach-v2", "GlobalGCE":"GlobalGCE-ChemAligned",
          "GCFExplainer":"GCFExplainer (V1)", "ComRecGC":"ComRecGC (V1)"}


def sealed(path):
    d=json.loads(Path(path).read_text())
    if d.get('self_sha256') != stable_sha256({k:v for k,v in d.items() if k!='self_sha256'}):
        raise ValueError('SOURCE_RECEIPT_CHANGED:'+str(path))
    return d


def reach_diagnostics(progress):
    """Only actual supplied counts; the new2659 train theta is a lower bound."""
    o=progress['observations']; result=[]
    for pool,split in [('old66','train'),('saved2607','train'),('old66','calibration'),
                       ('saved2607','calibration'),('final2659','calibration')]:
        key=f'{pool}_{split}'
        result.append(dict(method='Ours',candidate_pool=pool,split=split,
            base_denominator=o['train_denominator' if split=='train' else 'calibration_denominator'],
            gin_source_count=o['gin_source_train' if split=='train' else 'gin_source_calibration'],
            reach_count=o[key+'_reach'],reach_count_scope='FULL_SAVED_POOL_EXACT',
            theta_covered_count=o.get(key+'_theta','N/A'),theta_count_scope='FULL_SAVED_POOL_EXACT' if key+'_theta' in o else 'N/A',
            source='APLUS_ACTUAL_EXECUTION.json'))
    result.append(dict(method='Ours',candidate_pool='final2659',split='train',
        base_denominator=o['train_denominator'],gin_source_count=o['gin_source_train'],
        reach_count=o['saved2607_train_reach']+o['supplement_train_gap_parents_with_witness'],
        reach_count_scope='RETAINED_POOL_PLUS_DISTINCT_PREDECLARED_GAP_WITNESSES',
        theta_covered_count=o['saved2607_train_theta'],theta_count_scope='LOWER_BOUND_NOT_FULL2659_DISTANCE_EVALUATION',
        source='APLUS_ACTUAL_EXECUTION.json'))
    return result


def prepare(aplus_root, spec_path, v1_source, v1_spec_path, output, *, progress_path, funnel_root=None):
    aplus_root,spec_path,v1_source,v1_spec_path,output=map(Path,(aplus_root,spec_path,v1_source,v1_spec_path,output))
    spec=json.loads(spec_path.read_text()); prior=json.loads(v1_spec_path.read_text())
    if (sha256_file(v1_spec_path)!=spec['v1_spec']['sha256'] or prior['gin_files']!=spec['gin_files']
        or prior['thresholds']['sha256']!=spec['thresholds']['sha256']):
        raise ValueError('V1_BASELINES_NOT_SAME_FROZEN_GIN_AND_THRESHOLDS')
    derive(v1_source)  # source reducer consistency, not a repeated science audit
    frozen=sealed(aplus_root/'selection_freeze.json')
    from src.experiments.bace_gin_reach_test_raw import validate_aplus_freeze
    validate_aplus_freeze(frozen,spec=spec,evidence_root=aplus_root)
    selected=spec.get('new_control_name','adopted2607_new_selector')
    collected=[[r for r in read_csv(v1_source/name) if r['method'] in ('GCFExplainer','ComRecGC')] for name in INPUTS]
    for rows in collected:
        for r in rows:r['scientific_variant']='FROZEN_GIN_FIXED_POOL_V1_UNCHANGED'
    comparison=[];state='PENDING';audit_state='PENDING'
    if (aplus_root/'metrics.json').is_file():
        metrics=sealed(aplus_root/'metrics.json')
        if (metrics['spec_sha256']!=stable_sha256(spec) or metrics['freeze_sha256']!=frozen['self_sha256']
            or metrics['main_matrix_write'] is not False or set(metrics['results'])!=set(frozen['controls'])):
            raise ValueError('NEW_TEST_METRICS_NOT_PREDECLARED_FREEZE_BOUND')
        selected_metrics=metrics['results'][selected]
        def add(rows):return [dict(method='Ours',scientific_variant='OURS_REACH_V2_GIN_APLUS_2659',**r) for r in rows]
        fixed=[r for r in selected_metrics['prefix_rows'] if r['cohort']=='fixed141']
        collected[0]+=add([dict(r,state='EVALUATED') for r in fixed if r['K_requested']==10])
        collected[1]+=add(fixed)
        collected[2]+=add([r for r in selected_metrics['exact_ecdf'] if r['cohort']=='fixed141'])
        collected[3]+=add(selected_metrics['parent_distances'])
        comparison=[dict(method='Ours',variant=v,predeclared_primary=v==selected,**r)
            for v,result in metrics['results'].items() for r in result['prefix_rows'] if r['cohort']=='fixed141']
        state='EVALUATED'
        audit=aplus_root/'audit/final_audit.json'
        if audit.exists():
            value=sealed(audit)
            if value['spec_sha256']!=stable_sha256(spec) or value['freeze_sha256']!=frozen['self_sha256']:
                raise ValueError('NEW_TEST_AUDIT_BINDING_CHANGED')
            audit_state=value['state']
    else:
        collected[0].append(dict(method='Ours',state='PENDING',scientific_variant='OURS_REACH_V2_GIN_APLUS_2659'))
        comparison=[dict(method='Ours',variant=v,predeclared_primary=v==selected,state='PENDING',coverage='PENDING')
                    for v in frozen['controls']]
    collected[0].append(dict(method='GlobalGCE',state='PENDING',scientific_variant='GLOBALGCE_CHEMALIGNED_GIN_APLUS'))
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in zip(INPUTS,collected):write_csv(output/name,rows)
    checked=derive(output)
    progress=json.loads(Path(progress_path).read_text())
    write_csv(output/'reachability_and_pool_ceiling.csv',reach_diagnostics(progress))
    write_csv(output/'method_variant_comparison.csv',comparison)
    funnel=[]
    for method in METHODS:
        source=[r for r in collected[3] if r['method']==method and int(r['K_requested'])==10]
        row=next(r for r in checked['table2'] if r['method']==method)
        funnel.append(dict(method=method,variant=LABELS[method],split='test',K_requested=10,
            state=row['state'],base_parents=141 if source else 'PENDING',
            gin_source_parents=sum(int(r['pred_before'])==1 for r in source) if source else 'PENDING',
            selected_strict_flip_reached=row.get('finite_strict_flip_count','PENDING'),
            covered_at_theta=row.get('covered_count','PENDING'),
            applicable_pairs='N/A',chemically_valid_pairs='N/A',
            missing_evidence='PAIR_LEVEL_FAILURE_COUNTS_NOT_IN_AGGREGATE; NO_INFERENCE_RERUN',
            source='saved_parent_minima' if source else 'NO_COMPLETED_TEST_RESULT'))
    funnel_binding=None
    if funnel_root is not None:
        funnel_root=Path(funnel_root)
        receipt=sealed(funnel_root/'funnel_manifest.json')
        if (receipt.get('state')!='SAVED_RECORD_FUNNEL_EXPORTED'
            or receipt.get('spec_sha256')!=stable_sha256(spec)
            or receipt.get('model_inference') is not False
            or receipt.get('ot_computed')!=0 or receipt.get('main_matrix_write') is not False):
            raise ValueError('SAVED_FUNNEL_NOT_CURRENT_SPEC_BOUND')
        actual=read_csv(funnel_root/'method_funnel.csv')
        if not actual:
            raise ValueError('EMPTY_SAVED_FUNNEL')
        funnel=[dict(row,source='INDEPENDENT_SAVED_RECORD_FUNNEL') for row in actual]+[
            row for row in funnel if row['method']!='Ours']
        funnel_binding=dict(path=str(funnel_root),manifest_file_sha256=sha256_file(funnel_root/'funnel_manifest.json'),
                            csv_file_sha256=sha256_file(funnel_root/'method_funnel.csv'))
    write_csv(output/'method_failure_funnel.csv',funnel)
    manifest=dict(state='PARTIAL_A_PLUS_DISPLAY_SOURCES_PREPARED',selected_control=selected,
        ours_state=state,ours_audit_state=audit_state,global_state='PENDING',
        source_spec_file_sha256=sha256_file(spec_path),freeze_file_sha256=sha256_file(aplus_root/'selection_freeze.json'),
        baseline_source_root=str(v1_source),baseline_spec_file_sha256=sha256_file(v1_spec_path),
        progress_source_file_sha256=sha256_file(progress_path),main_matrix_write=False,
        different_method_versions_explicit=LABELS,selection_uses_test=False,
        saved_funnel_source=funnel_binding)
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def prepare_and_render(aplus_root,spec_path,v1_source,v1_spec_path,output,*,progress_path,funnel_root=None):
    output=Path(output)
    result=prepare(aplus_root,spec_path,v1_source,v1_spec_path,output/'source_csv',progress_path=progress_path,funnel_root=funnel_root)
    render(output/'source_csv',output/'figures',version_label='A+ Ours + unchanged V1 native baselines',display_labels=LABELS)
    return result
