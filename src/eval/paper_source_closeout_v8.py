"""Saved-record paper closeout. No model, selector, generation or OT execution."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def verify_selector_bindings(records):
    for dataset,files in records.items():
        binding=files['p0/input_binding.json']
        if canonical_sha(binding['spec'])!=binding['spec_sha']:
            raise ValueError('SELECTOR_SPEC_DIGEST_CHANGED:'+dataset)
        for i in range(10):
            key=('p0/' if i<7 else 'p1/')+f'S{i}_freeze.json'
            freeze=dict(files[key]);declared=freeze.pop('freeze_sha256')
            if canonical_sha(freeze)!=declared:
                raise ValueError('SELECTOR_FREEZE_DIGEST_CHANGED:'+dataset+'/'+key)
            if freeze['matrix_semantic_sha']!=binding['matrix_semantic_sha']:
                raise ValueError('SELECTOR_MATRIX_BINDING_CHANGED:'+dataset+'/'+key)
    return records


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def metric(values, theta, cap):
    values=np.asarray(values,dtype=float)
    if np.isnan(values).any() or (values<0).any():
        raise ValueError('UNKNOWN_OR_NEGATIVE_RAW')
    finite=values[np.isfinite(values)]
    return dict(denominator=len(values),covered=int((values<=theta).sum()),
        coverage=float((values<=theta).mean()),finite=len(finite),
        conditional_median=float(np.median(finite)) if len(finite) else None,
        capped_mean=float(np.minimum(values,cap).mean()))


def fmt(value, decimals=5):
    return 'N/A' if value is None or value=='' else f'{float(value):.{decimals}f}'


def tex_table(caption,label,columns,body,alignment):
    return ('\\begin{table}[t]\n\\centering\\small\n\\caption{'+caption+'}\n'
        '\\label{'+label+'}\n\\begin{tabular}{@{}'+alignment+'@{}}\n\\toprule\n'
        +' & '.join(columns)+' \\\\\n\\midrule\n'+'\n'.join(body)
        +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def export(base, output, paper):
    base,root,paper=Path(base),Path(output),Path(paper)
    for part in ('source_csv','figures','audit'):(root/part).mkdir(parents=True,exist_ok=True)
    cm=base/'cm4-taste-closeout-v6/run-20260916T093500Z/release-cm4-accepted'
    sel=base/'p0-first-p1-backfill-v7/run-20260917'
    migration=base/'main16-next-20260908-2218'
    binding_path=root/'audit/selector_frozen_bindings.json'
    selector_bindings=verify_selector_bindings(json.loads(binding_path.read_text()))
    mapping=[];checks=[]
    def bind(target,rowid,field,value,source,sourcefield,scope,oracle,pool,selector,N,K,theta,cost,audit,derivation='identity'):
        receipt=json.loads(Path(audit).read_text())
        mapping.append(dict(target=target,row_id=rowid,field=field,value=value,producer_source=str(source),
            producer_identity=receipt.get('source_root',str(Path(audit).parent.parent)),
            producer_commit=receipt.get('execution_commit',receipt.get('commit','NOT_RECORDED_IN_ADOPTED_RECEIPT')),
            producer_binding='saved record SHA plus original acceptance receipt; not a new model execution',
            input_manifest='',input_manifest_sha256='',matrix_semantic_sha256='',
            source_sha256=sha(source),source_field=sourcefield,derivation=derivation,
            oracle=oracle,pool=pool,selector=selector,scope=scope,denominator=N,K=K,theta=theta,
            cost_definition=cost,audit_path=str(audit),audit_sha256=sha(audit)))
    cm_table=read_csv(cm/'source_csv/table2_k20_theta010_versioned.csv')
    scopes={r['dataset']:r for r in read_csv(cm/'source_csv/cm_four_dataset_scope.csv')}
    for name in ('table2_k20_theta010_versioned.csv','figure2_coverage_cost_vs_k.csv',
                 'figure3_coverage_vs_theta.csv','parent_best_distances_selected.csv',
                 'cm_four_dataset_scope.csv','cm_taste_saturation_audit.csv'):
        shutil.copy2(cm/'source_csv'/name,root/'source_csv'/name)
    cm_prefix=read_csv(cm/'source_csv/figure2_coverage_cost_vs_k.csv')
    cm_ecdf=read_csv(cm/'source_csv/figure3_coverage_vs_theta.csv')
    for r in cm_table:
        key=(r['dataset'],r['method']);scope=scopes[r['dataset']]
        p=[x for x in cm_prefix if (x['dataset'],x['method'])==key and x['k']=='20']
        q=[x for x in cm_ecdf if (x['dataset'],x['method'])==key and float(x['theta'])==.1 and x['k']=='20']
        if len(p)!=1 or len(q)!=1:raise ValueError('MISSING_SAME_SOURCE_THETA_POINT:'+str(key))
        for field in ('covered_count','coverage'):
            if float(r[field])!=float(p[0][field]) or float(r[field])!=float(q[0][field]):
                raise ValueError('FIGURE_TABLE_CONFLICT:'+str(key))
        for field in ('conditional_median','fixed_capped_mean','finite_count'):
            if r[field]!=p[0][field]:raise ValueError('PREFIX_COST_CONFLICT')
        if float(r['coverage'])!=int(r['covered_count'])/int(r['N_base']):
            raise ValueError('DENOMINATOR_CONFLICT')
        for field in ('covered_count','coverage','finite_count','conditional_median','fixed_capped_mean'):
            bind('Table2/Figure2/Figure3', '/'.join(key),field,r[field],
                 cm/'source_csv/table2_k20_theta010_versioned.csv',field,r['scope'],
                 scope['oracle']+':'+scope['oracle_sha256'],str(scope['pool_count']) if r['method']!='Ours' else 'R3_original_pool_see_V5_adoption',
                 scope['freeze_sha'] if r['method']!='Ours' else json.loads((base/'taste-k20-theta010-final-v5/run-20260916T070500Z/ours_adoption.json').read_text())['ordered_rule_sha256'],
                 r['N_base'],20,.1,'all-finite median; separate fixed-base capped mean',cm/'audit/final_audit.json')
        checks.append(dict(check='TABLE2_K20_EQUALS_FIGURE2_K20_AND_FIGURE3_THETA010',row='/'.join(key),status='PASS'))
    table_body=[]
    for r in sorted(cm_table,key=lambda x:(x['dataset'],x['method'])):
        ds=r['dataset']+(' (descriptive)' if r['dataset']=='AIDS' else '')
        table_body.append(f"{ds} & {r['method'].replace('-Global','')} & {r['covered_count']}/{r['N_base']} & "
                          f"{100*float(r['coverage']):.2f} & {fmt(r['conditional_median'])} & {fmt(r['fixed_capped_mean'])} \\")
        table_body[-1]+='\\'
    (paper/'sections/generated/table2_results.tex').write_text(tex_table(
        'PARTIAL source-bound results at $K=20$, $\\theta=0.1$. Median is over all finite '
        'valid strict-flip distances, without threshold filtering; capped mean uses the original '
        'dataset cap. AIDS is source-descriptive, not held-out. Unbound legacy display values '
        'and unfinished Taste baselines are not substituted with zero.',
        'tab:main-results',['Dataset','Method','Covered/base','Cov (\\%)','Median','Capped mean'],table_body,'llrrrr'))
    # The accepted migration supplies saved valid parent distances. Validate the
    # original threshold reduction before reporting a threshold-only .1 view.
    ablations=[]
    for family in ('gnn_a','llm_gin'):
        delivery=migration/family/'delivery'
        receipt=json.loads((delivery/'acceptance.json').read_text())
        if receipt.get('state')!='SAVED_RECORD_AND_METRIC_ACCEPTANCE_PASS':
            raise ValueError('UNACCEPTED_MIGRATION_INPUT')
        if receipt.get('audit_scope')!='ALL_SAVED_MATCH_PAIR_PARENT_REDUCTIONS_AND_CSV_NOT_MODEL_REEXECUTION':
            raise ValueError('INCOMPLETE_SAVED_PARENT_AUDIT')
        manifest=json.loads((delivery/'package_manifest.json').read_text())
        for name in ('source_csv/parent_distances.csv','source_csv/prefix_metrics.csv','spec.json','acceptance.json'):
            if sha(delivery/name)!=manifest['files'][name]:raise ValueError('SMALL_SOURCE_DIGEST_CHANGED:'+name)
        spec=json.loads((delivery/'spec.json').read_text())
        parents=read_csv(delivery/'source_csv/parent_distances.csv')
        old=read_csv(delivery/'source_csv/prefix_metrics.csv')
        for r in old:
            if r['K_requested']!='20':continue
            raw=[x for x in parents if x['role']==r['role'] and x['selection_cohort']==r['selection_cohort']
                 and x['K_requested']=='20' and (r['cohort']!='gin_native' or x['pred_before']=='1')]
            if len({x['parent_id'] for x in raw})!=len(raw):raise ValueError('DUPLICATE_RAW_PARENT')
            if len(raw)!=int(r['denominator']):raise ValueError('ABLATION_DENOMINATOR_CHANGED')
            # This archived schema uses blank for an audited, fully evaluated
            # parent with no valid match. It is not a general UNKNOWN->INF rule:
            # receipt state, package digests and the original finite count are
            # all checked before accepting the reduction below.
            values=[float(x['best_valid_distance']) if x['best_valid_distance'] else math.inf for x in raw]
            cap=float(r['cost_cap']);prior=metric(values,float(r['theta_star']),cap)
            for oldkey,newkey in (('covered_count','covered'),('finite_strict_flip_count','finite'),
                                  ('conditional_median','conditional_median'),('fixed_capped_mean','capped_mean')):
                if not math.isclose(float(r[oldkey]),prior[newkey],rel_tol=0,abs_tol=1e-14):
                    raise ValueError('ACCEPTED_ABLATION_REDUCTION_MISMATCH:'+family+'/'+r['role'])
            revised=metric(values,.1,cap)
            result=dict(family=family,role=r['role'],cohort=r['cohort'],
                selection_cohort=r['selection_cohort'],K=20,effective_K=r['K_effective'],theta=.1,
                original_selection_theta=r['theta_star'],cost_cap=cap,**revised,
                scope=receipt['scope'],training_repeated=False,selection_repeated=False,
                reduction_scope='POST_HOC_THRESHOLD_ONLY_ON_SAVED_FROZEN_SEQUENCE')
            ablations.append(result)
            role=spec['roles'][r['role']]
            for field in ('coverage','covered','finite','conditional_median','capped_mean'):
                bind('Table3',family+'/'+r['role']+'/'+r['cohort'],field,result[field],
                    delivery/'source_csv/parent_distances.csv','best_valid_distance',receipt['scope'],
                    role['backbone']+':'+role['model_files']['model.pt'],role['pool']['sha256'],
                    receipt['freeze_sha256'],len(raw),20,.1,'all-finite median; fixed-base capped mean',
                    delivery/'acceptance.json','uncapped saved-parent reduction; no reselection')
    write_csv(root/'source_csv/table3_scoped_k20_theta010.csv',ablations)
    labels={'L0':'BRICS','L1':'7B off-the-shelf','L2':'7B PPO-LoRA','L3':'2B off-the-shelf',
            'gin':'GIN','gine':'GINE','gatedgcn_plus':'GatedGCN+','gatv2':'GATv2','gcn':'GCN'}
    body=[]
    for family,cohort,caption in [('llm_gin','fixed141','Fixed outputs, GIN, base 141'),
                                  ('gnn_a','common','A+ proposal-fixed, common 96')]:
        body.append('\\multicolumn{6}{l}{'+caption+'} \\\\')
        for r in ablations:
            if r['family']!=family or r['cohort']!=cohort:continue
            body.append(f"{labels[r['role']]} & {r['effective_K']} & {r['covered']}/{r['denominator']} & "
                        f"{100*r['coverage']:.2f} & {fmt(r['conditional_median'])} & {fmt(r['capped_mean'])} \\\\")
    (paper/'sections/generated/table3_ablation.tex').write_text(tex_table(
        'Archived BACE migrations, $K=20$, reported at $\\theta=0.1$ using saved uncapped '
        'parent distances. Original calibration order is unchanged; it was frozen under the '
        'earlier threshold contract. Panels have different pools and denominators and are '
        'not one end-to-end ablation. No project-SFT or multi-seed inference is established.',
        'tab:ablation-components',['Variant','Effective $K$','Covered/base','Cov (\\%)','Median','Capped mean'],body,'lrrrrr'))
    prefix_path=sel/'figures/selector_figure2_source.csv'
    prefix=read_csv(prefix_path);contrasts=[];selector_table=[]
    for ds in ('BACE','Mutagenicity'):
        def seq(sid):return sorted([r for r in prefix if r['dataset']==ds and r['variant']==sid],key=lambda r:int(r['k']))
        s3,s5=seq('S3'),seq('S5')
        if len(s3)!=20 or len(s5)!=20:raise ValueError('INCOMPLETE_CONTROLLED_PREFIX')
        for name,indices in [('MeanCov1_20',range(20)),('MeanCov1_10',range(10))]+[(f'Cov{k}',[k-1]) for k in (1,3,5,10,20)]:
            left=float(np.mean([float(s3[i]['coverage']) for i in indices]))
            right=float(np.mean([float(s5[i]['coverage']) for i in indices]))
            contrasts.append(dict(dataset=ds,contrast='S5-S3',metric=name,S3=left,S5=right,
                                  difference_pp=100*(right-left),scope='FROZEN_ORDER_HELDOUT_SAVED_REDUCTION'))
        for sid in (f'S{i}' for i in range(10)):
            sequence=seq(sid);r=sequence[-1]
            entry=dict(**r,MeanCov=float(np.mean([float(x['coverage']) for x in sequence])))
            selector_table.append(entry)
            for field in ('covered','coverage','finite','conditional_median','capped_mean','MeanCov'):
                bind('SelectorTables/Figures',ds+'/'+sid,field,entry[field],prefix_path,
                     field if field!='MeanCov' else 'mean(coverage[K=1..20])',
                     'ORIGINAL_POOL_CONTROLLED_V1',ds+('_GINE' if ds=='BACE' else '_RF'),
                     '66' if ds=='BACE' else '683',sid,r['denominator'],20,.1,
                     'all-finite median; separate fixed-base capped mean',sel/'figures/reduction_audit.json')
    write_csv(root/'source_csv/selector_s5_minus_s3.csv',contrasts)
    write_csv(root/'source_csv/selector_table_k20_with_meancov.csv',selector_table)
    body=[]
    for ds in ('BACE','Mutagenicity'):
        body.append('\\multicolumn{6}{l}{'+ds+'} \\\\')
        for r in selector_table:
            if r['dataset']!=ds or int(r['variant'][1:])>6:continue
            body.append(f"{r['variant']} & {100*float(r['coverage']):.2f} & {100*r['MeanCov']:.2f} & "
                        f"{r['finite']} & {fmt(r['conditional_median'])} & {fmt(r['capped_mean'])} \\\\")
    (paper/'sections/generated/table_selector_controlled_results.tex').write_text(tex_table(
        'Completed Controlled-v1 test results. MeanCov averages all twenty prefixes. '
        'S6 is Full-Controlled, not the production selector and not uniformly best. '
        'Cost columns have different populations; no test-time selection was performed.',
        'tab:selector-controlled-results',['Variant','Cov@20 (\\%)','MeanCov (\\%)','Finite','Median','Capped mean'],body,'lrrrrr'))
    full_body=[];diagnostic_body=[]
    for ds in ('BACE','Mutagenicity'):
        full_body.append('\\multicolumn{6}{l}{'+ds+'} \\\\')
        diagnostic_body.append('\\multicolumn{6}{l}{'+ds+'} \\\\')
        diag_path=sel/'selector_results'/ds/'saved-data-diagnostics/redundancy_size_efficiency_scoped.csv'
        diag_rows=read_csv(diag_path)
        for r in selector_table:
            if r['dataset']!=ds:continue
            full_body.append(f"{r['variant']} & {100*float(r['coverage']):.2f} & {100*r['MeanCov']:.2f} & "
                             f"{r['finite']} & {fmt(r['conditional_median'])} & {fmt(r['capped_mean'])} \\\\")
        for r in diag_rows:
            diagnostic_body.append(f"{r['variant']} & {fmt(r['calibration_coverage_jaccard'],3)} & "
                f"{fmt(r['structural_tanimoto'],3)} & {fmt(r['normalized_heavy_atom_size'],3)} & "
                f"{r['proposals']} & {r['accepted']} \\\\")
            for field in ('calibration_coverage_jaccard','structural_tanimoto','normalized_heavy_atom_size','proposals','accepted'):
                bind('SelectorDiagnostics',ds+'/'+r['variant'],field,r[field],diag_path,field,
                     'CALIBRATION_FIXED_POOL',ds+('_GINE' if ds=='BACE' else '_RF'),
                     '66' if ds=='BACE' else '683',r['variant'],'66' if ds=='BACE' else '235',20,.1,
                     'not a cost measurement',sel/'figures/reduction_audit.json')
    (paper/'sections/generated/table_selector_full_results.tex').write_text(tex_table(
        'All twenty completed Controlled-v1 units; no test-based variant selection.',
        'tab:selector-full-results',['Variant','Cov@20 (\\%)','MeanCov (\\%)','Finite','Median','Capped mean'],full_body,'lrrrrr'))
    (paper/'sections/generated/table_selector_diagnostics.tex').write_text(tex_table(
        'Saved calibration diagnostics. CovRed is coverage Jaccard at 0.1; Struct is '
        'fingerprint Tanimoto; Size is normalized heavy-atom count, not actual deleted-atom ratio. '
        'Complete selection timing is not inferred from the old refinement-only timer.',
        'tab:selector-diagnostics',['Variant','CovRed','Struct','Size','Proposals','Accepted'],diagnostic_body,'lrrrrr'))
    claims=[]
    for ds in ('BACE','Mutagenicity'):
        vals={r['metric']:r for r in contrasts if r['dataset']==ds}
        claims.append(f"On {ds}, S5 minus S3 changes MeanCov by {vals['MeanCov1_20']['difference_pp']:.3f} "
            f"percentage points and mean coverage over the first ten prefixes by "
            f"{vals['MeanCov1_10']['difference_pp']:.3f} points; the terminal coverage difference is "
            f"{vals['Cov20']['difference_pp']:.3f} points.")
    (paper/'sections/generated/v8_selector_findings.tex').write_text('\n'.join(claims)+'\n')
    for record in mapping:
        if not record['target'].startswith('Selector'):continue
        dataset,sid=record['row_id'].split('/')
        files=selector_bindings[dataset];binding=files['p0/input_binding.json']
        spec=binding['spec'];phase='p0' if int(sid[1:])<7 else 'p1'
        freeze=files[phase+'/'+sid+'_freeze.json']
        record.update(oracle=spec['oracle']+':'+spec['oracle_sha256'],
            pool=spec['pool_sha_from_existing_receipt'],
            selector=sid+':'+freeze['freeze_sha256'],
            input_manifest=str(binding_path),input_manifest_sha256=sha(binding_path),
            matrix_semantic_sha256=binding['matrix_semantic_sha'])
    write_csv(root/'source_csv/paper_result_source_map.csv',mapping)
    write_csv(root/'audit/source_checks.csv',checks)
    shutil.copy2(sel/'figures/selector_figure2_test.pdf',paper/'figures/selector_figure2_test.pdf')
    shutil.copy2(sel/'figures/selector_figure3_exact_ecdf.pdf',paper/'figures/selector_figure3_exact_ecdf.pdf')
    plot(root)
    for filename in ('figure2_coverage_cost_vs_k','figure3_coverage_vs_theta'):
        shutil.copy2(root/'figures'/(filename+'.pdf'),paper/'figures'/(filename+'.pdf'))
    audit=dict(state='SAVED_RECORD_PAPER_REDUCTION_PASS',source_bound_fields=len(mapping),
        main_rows=len(cm_table),ablation_rows=len(ablations),selector_rows=len(selector_table),
        source_figures_table_checks=len(checks),science_repeated=False,main_authority_written=False,
        cm_scope_metadata_warning='Mut reported ID-string overlap 217 requires namespace interpretation; no cross-scope aggregate claimed',
        manuscript_status='PARTIAL_NOT_20_CELL_FINAL',deadline='2026-09-24T23:59:59+08:00')
    (root/'audit/paper_reduction.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def plot(root, source_csv_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    source=Path(source_csv_dir) if source_csv_dir is not None else root/'source_csv'
    (root/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    datasets=('AIDS','Mutagenicity','BACE','TasteMolNet')
    rows=read_csv(source/'figure2_coverage_cost_vs_k.csv')
    ecdf=read_csv(source/'figure3_coverage_vs_theta.csv')
    styles={'CM-CReM-Global':dict(color='#333333',marker='s'), 'Ours':dict(color='#c32929',marker='o')}
    fig,axes=plt.subplots(2,4,figsize=(12.8,4.1),layout='constrained')
    for j,ds in enumerate(datasets):
        title=ds+('\nsource-descriptive' if ds=='AIDS' else '')
        axes[0,j].set_title(title,fontsize=10)
        for method,style in styles.items():
            rr=sorted([r for r in rows if r['dataset']==ds and r['method']==method],key=lambda r:int(r['k']))
            if not rr:continue
            for i,field in enumerate(('coverage','conditional_median')):
                y=[float(r[field])*(100 if i==0 else 1) if r[field] else np.nan for r in rr]
                axes[i,j].plot([int(r['k']) for r in rr],y,label=method.replace('-Global',''),
                              linewidth=1,markersize=3,markevery=[0,4,9,14,19],**style)
        for i in range(2):
            axes[i,j].set_xlim(1,20);axes[i,j].set_xticks([1,5,10,15,20]);axes[i,j].grid(alpha=.25)
            axes[i,j].set_xlabel('$K$' if i else '')
        maximum=max(float(r['coverage'])*100 for r in rows if r['dataset']==ds)
        axes[0,j].set_ylim(0,min(105,max(10,math.ceil((maximum+5)/10)*10)))
        if ds=='TasteMolNet':
            axes[0,j].axhline(100*285/468,ls=':',lw=.7,color='gray')
            axes[0,j].text(2,100*285/468+2,'source ceiling',fontsize=7,color='gray')
    axes[0,0].set_ylabel('Coverage (%)');axes[1,0].set_ylabel('Conditional median cost')
    axes[0,3].legend(fontsize=8,loc='lower right')
    fig.suptitle('PARTIAL — frozen accepted sequences; theta = 0.1',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(root/'figures'/('figure2_coverage_cost_vs_k.'+ext),dpi=320)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12.8,2.5),layout='constrained')
    for j,ds in enumerate(datasets):
        axes[j].set_title(ds+('\nsource-descriptive' if ds=='AIDS' else ''),fontsize=10)
        for method,style in styles.items():
            rr=sorted([r for r in ecdf if r['dataset']==ds and r['method']==method and r['k']=='20'],key=lambda r:float(r['theta']))
            if rr:axes[j].step([float(r['theta']) for r in rr],[100*float(r['coverage']) for r in rr],
                where='post',lw=1,color=style['color'],label=method.replace('-Global',''))
        axes[j].set_xlim(0,.2);axes[j].set_ylim(bottom=0);axes[j].grid(alpha=.25)
        axes[j].axvline(.1,ls=':',lw=.8,color='gray');axes[j].set_xlabel('Raw WNode threshold')
    axes[0].set_ylabel('Coverage (%)');axes[-1].legend(fontsize=8,loc='lower right')
    fig.suptitle('PARTIAL — K=20 exact empirical coverage, no smoothing',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(root/'figures'/('figure3_coverage_vs_theta.'+ext),dpi=320)
    plt.close(fig)
