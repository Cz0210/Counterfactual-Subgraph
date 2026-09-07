import json
from pathlib import Path
import pytest
from src.eval.bace_frozen_gnn_contracts import stable_sha256,sha256_file
from src.experiments.bace_gin_fixed_pool import prefix_metrics
from src.experiments.bace_gin_reporting import INPUTS,read_csv,write_csv
from src.experiments.bace_gin_aplus_reporting import prepare,reach_diagnostics
from src.experiments.bace_gin_reach_selector import POLICY


def write(path,data,seal=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if seal:data={**data,'self_sha256':stable_sha256(data)}
    path.write_text(json.dumps(data));return data


def case(root,complete):
    original=root/'v1';original.mkdir()
    ids=[f'p{i}' for i in range(141)];rules=[f'r{i}' for i in range(20)]
    def result(cost):
        pairs=[dict(parent_id=p,candidate_id=c,pred_before=int(p!='p0'),pred_after=0,
            pair_strict_flip=p!='p0',wnode_distance=cost if p!='p0' else None) for p in ids for c in rules]
        return prefix_metrics(ids,rules,pairs,theta=.02,cap=.03,endpoints=[0,.02,.03])
    old=result(.01);contents=[[],[],[],[]]
    for method in ('Ours','GCFExplainer','ComRecGC'):
        add=lambda rows:[dict(method=method,**r) for r in rows]
        fixed=[r for r in old['prefix_rows'] if r['cohort']=='fixed141']
        contents[0]+=add([dict(r,state='EVALUATED') for r in fixed if r['K_requested']==10])
        contents[1]+=add(fixed)
        contents[2]+=add([r for r in old['exact_ecdf'] if r['cohort']=='fixed141'])
        contents[3]+=add(old['parent_distances'])
    contents[0].append(dict(method='GlobalGCE',state='BLOCKED_MATERIALIZATION'))
    for name,rows in zip(INPUTS,contents):write_csv(original/name,rows)
    prior={'gin_files':{'model.pt':'gin','temperature_scaling.json':'T'},'thresholds':{'sha256':'threshold'}}
    write(root/'v1-spec.json',prior)
    spec={**prior,'v1_spec':{'sha256':sha256_file(root/'v1-spec.json')},'test_results_previously_observed':True,
          'new_control_name':'expanded_pool_new_selector','selector_policy':POLICY}
    write(root/'spec.json',spec)
    new=root/'new';new.mkdir()
    contract=write(new/'contract.json',dict(spec_sha256=stable_sha256(spec),old_order=rules,main_matrix_write=False),True)
    frozen=write(new/'selection_freeze.json',dict(state='CALIBRATION_SELECTOR_FROZEN',test_loaded=False,
        spec_sha256=stable_sha256(spec),policy=POLICY,contract_sha256=contract['self_sha256'],main_matrix_write=False,
        calibration_parent_ids=[f'cal{i}' for i in range(66)],
        controls={k:rules for k in ('old66_old_selector','old66_new_selector','expanded_pool_new_selector')}),True)
    if complete:
        write(new/'metrics.json',dict(state='EVALUATED_NOT_INDEPENDENT_AUDIT',spec_sha256=stable_sha256(spec),
            freeze_sha256=frozen['self_sha256'],main_matrix_write=False,
            results={k:result(.015 if k=='expanded_pool_new_selector' else .001) for k in frozen['controls']}),True)
    observations={'train_denominator':386,'calibration_denominator':66,'gin_source_train':318,'gin_source_calibration':45,
        'old66_train_reach':158,'saved2607_train_reach':313,'saved2607_train_theta':208,
        'old66_calibration_reach':19,'old66_calibration_theta':7,'saved2607_calibration_reach':43,
        'saved2607_calibration_theta':32,'final2659_calibration_reach':43,'final2659_calibration_theta':32,
        'supplement_train_gap_parents_with_witness':5}
    write(root/'progress.json',{'observations':observations})
    return dict(aplus_root=new,spec_path=root/'spec.json',v1_source=original,v1_spec_path=root/'v1-spec.json',
                output=root/'assembled',progress_path=root/'progress.json')


def test_pending_does_not_adopt_old_ours_or_plot_zero(tmp_path):
    args=case(tmp_path,False);manifest=prepare(**args)
    table=read_csv(args['output']/INPUTS[0])
    assert next(r for r in table if r['method']=='Ours')['state']=='PENDING'
    assert next(r for r in table if r['method']=='GlobalGCE')['state']=='PENDING'
    assert {r['method'] for r in read_csv(args['output']/INPUTS[1])}=={'GCFExplainer','ComRecGC'}
    assert manifest['ours_state']=='PENDING'


def test_completed_uses_predeclared_control_not_test_best(tmp_path):
    args=case(tmp_path,True);manifest=prepare(**args)
    table=read_csv(args['output']/INPUTS[0]);ours=next(r for r in table if r['method']=='Ours')
    assert float(ours['fixed_capped_mean'])==pytest.approx((140*.015+.03)/141)
    assert manifest['selected_control']=='expanded_pool_new_selector'
    assert manifest['ours_audit_state']=='PENDING'  # don't invent an independent audit
    comparison=read_csv(args['output']/'method_variant_comparison.csv')
    assert len(comparison)==60
    assert {r['variant'] for r in comparison if r['predeclared_primary']=='True'}=={'expanded_pool_new_selector'}


def test_baselines_require_same_actual_gin_spec_binding(tmp_path):
    args=case(tmp_path,True)
    spec=json.loads(args['spec_path'].read_text());spec['gin_files']['model.pt']='changed'
    write(args['spec_path'],spec)
    with pytest.raises(ValueError,match='SAME_FROZEN_GIN'):prepare(**args)


def test_2659_train_theta_retained_lower_bound_not_exact(tmp_path):
    args=case(tmp_path,False)
    rows=reach_diagnostics(json.loads(args['progress_path'].read_text()))
    row=next(r for r in rows if r['candidate_pool']=='final2659' and r['split']=='train')
    assert row['reach_count']==318 and row['theta_covered_count']==208
    assert row['theta_count_scope']=='LOWER_BOUND_NOT_FULL2659_DISTANCE_EVALUATION'


def test_actual_saved_funnel_is_preserved_not_overwritten_by_placeholder(tmp_path):
    args=case(tmp_path,True);funnel=tmp_path/'funnel';funnel.mkdir()
    spec=json.loads(args['spec_path'].read_text())
    write(funnel/'funnel_manifest.json',dict(state='SAVED_RECORD_FUNNEL_EXPORTED',
        spec_sha256=stable_sha256(spec),model_inference=False,ot_computed=0,main_matrix_write=False),True)
    write_csv(funnel/'method_funnel.csv',[dict(method='Ours-GIN-Aplus',scope='expanded_pool_new_selector_K10',
        applicable_pairs=75,chemically_valid_pairs=70,missing_raw_distance='N/A')])
    manifest=prepare(**args,funnel_root=funnel)
    rows=read_csv(args['output']/'method_failure_funnel.csv')
    ours=next(r for r in rows if r['method']=='Ours-GIN-Aplus')
    assert ours['applicable_pairs']=='75' and ours['missing_raw_distance']=='N/A'
    assert not any(r['method']=='Ours' for r in rows)
    assert manifest['saved_funnel_source']['csv_file_sha256']==sha256_file(funnel/'method_funnel.csv')
