"""Read-only 20-cell paper projection; never a publisher or scientific authority."""
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path

DATASETS=('AIDS','Mutagenicity','BACE','TasteMolNet')
METHODS=('GlobalGCE','ComRecGC','GCFExplainer','Ours','CM-CReM')

def read_csv(path):
    if not path.exists():return []
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=keys);out.writeheader();out.writerows(rows)

def number(value):
    if value in (None,'','N/A','PENDING'):return None
    x=float(value)
    if math.isnan(x):raise ValueError('NaN is not a missing or failed scientific record')
    return x

def reduce_distances(rows,theta,cap):
    ids=[r['parent_id'] for r in rows]
    if not ids or len(set(ids))!=len(ids) or 'N/A' in ids:raise ValueError('Missing/duplicate parent records')
    vals=[]
    for r in rows:
        flag=r['strict_recourse_available'].lower()
        if flag not in ('true','false'):raise ValueError('Unknown strict-flip mask')
        raw=number(r['best_distance']);value=math.inf if raw is None else raw
        if value<0 or math.isfinite(value)!=(flag=='true'):raise ValueError('Distance/recourse mismatch')
        vals.append(value)
    finite=sorted(x for x in vals if math.isfinite(x));n=len(vals)
    return dict(coverage=sum(x<=theta for x in vals)/n,covered_count=sum(x<=theta for x in vals),
        N=n,finite_recourse_count=len(finite),conditional_median_cost=statistics.median(finite) if finite else 'N/A',
        fixed_capped_mean_cost=math.fsum(min(x,cap) for x in vals)/n if cap is not None else 'N/A'),vals

def exact_curve(vals,theta):
    finite=[x for x in vals if math.isfinite(x)]
    return [dict(distance=x,coverage=sum(v<=x for v in finite)/len(vals))
        for x in sorted({0.,theta,*finite})]

def close(a,b):return a is not None and b is not None and math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12)

def build(input_root,output_dir,allow_partial=False):
    source=Path(input_root);out=Path(output_dir)
    index=json.loads((source/'source_index.json').read_text())
    pointer=index['pointer'];registered=set(pointer['applied_cells'])
    cells={(r['dataset'],r['method']):r for r in index['source_cells']}
    states=[];prefix=[];curves=[];issues=[]
    for d in DATASETS:
        for m in METHODS:
            meta=cells.get((d,m),{});root=source/'cells'/d/m
            reg=d+'/'+m in registered
            base=dict(dataset=d,method_family=m,variant=meta.get('registry_exception') or meta.get('status','PENDING'),
                oracle=meta.get('oracle_backend','PENDING'),source_root=meta.get('standardized_output_root','PENDING'),
                registered=reg,scientific_result_with_scope=False,paper_release=False,
                stage='PENDING',science_audit='PENDING',package_import='SOURCE_REFERENCE',replacement_of=d+'/'+m,
                scope=meta.get('registry_exception_waivers',''),pool='SOURCE_MANIFEST',selector='SOURCE_MANIFEST',
                cohort='SOURCE_MANIFEST',threshold='SOURCE_MANIFEST',cost='SOURCE_MANIFEST')
            cm=m=='CM-CReM' and d=='BACE'
            if m=='CM-CReM' and not cm:
                base.update(variant='CM-Global-K20-v2',oracle='gine' if d=='TasteMolNet' else 'RF_TO_VERIFY',
                    stage='RF_TRAIN_ADAPTER_IN_PROGRESS' if d=='AIDS' else 'PENDING_DATASET_PILOT')
                states.append(base);continue
            if cm:
                root=source/'cm_bace_v1';receipt=json.loads((root/'cm_import_receipt.json').read_text())
                publication=json.loads((root/'cm_run_publication.json').read_text())
                if receipt['status']!='CM_RESULT_IMPORT_VERIFIED' or publication['status']!='BACE_CM_CREM_RESULT_PUBLISHED':
                    raise ValueError('CM import/publication not accepted')
                base.update(registered=True,variant='CM-CReM-Global-Budgeted-v1',oracle='gine',
                    source_root=publication.get('import_root',str(root)),package_import='IMPORTED_PUBLISHED',
                    scope='POST_HOC_BACE_ORIGINAL_GINE_FULL_GRAPH_PROTOTYPES; NUMERIC_PRODUCER_BINDING_PENDING_FOR_COMBINED_PLOT')
                summary=json.loads((root/'export_manifest.json').read_text())
                theta=summary['theta'];cap=summary['cap'];rows=read_csv(root/'prefix_metrics.csv');parents=read_csv(root/'parent_best_distances.csv')
                parents=[dict(r,best_distance=r['best_distance_uncapped'],
                    strict_recourse_available=r['finite_strict_flip']) for r in parents]
            elif not reg:
                base.update(stage='PENDING_RECOVERY');states.append(base);continue
            else:
                summary=json.loads((root/'summary.json').read_text());rows=read_csv(root/'prefix_metrics.csv');parents=read_csv(root/'parent_best_distances.csv')
                theta=summary.get('theta_star',number(rows[0].get('theta',rows[0].get('threshold'))))
                cap=summary.get('cost_cap')
            base.update(threshold=theta,cost='FIXED_CAPPED_MEAN' if d in ('BACE','TasteMolNet') else 'ORIGINAL_CONDITIONAL',
                science_audit='EXISTING_ACCEPTANCE_WITH_RETAINED_SCOPE',stage='SAVED_RECORDS')
            broken=(d,m) in {('BACE','GlobalGCE'),('AIDS','ComRecGC')}
            if broken:
                base.update(stage='UNDER_REPAIR',science_audit='OLD_REGISTERED_RESULT_NOT_NEW_VALID_ZERO',
                    scope=base['scope']+'; '+('original GINE materialization unresolved' if d=='BACE' else 'RF-aligned summary successor running; old source-cohort retained only'))
            usable=not broken
            any_exact=False;all_valid=True
            for k in range(1,21):
                found=[r for r in rows if int(r['k'])==k]
                if len(found)!=1:raise ValueError(f'{d}/{m}/K{k}: missing/duplicate prefix')
                r=found[0];cov=number(r.get('coverage',r.get('close_cf_coverage')))
                cost=number(r.get('cost',r.get('conditional_median_cost')))
                metric='fixed_capped_mean_cost' if d in ('BACE','TasteMolNet') else 'conditional_median_cost'
                if metric=='fixed_capped_mean_cost':cost=number(r.get(metric,r.get('cost')))
                derived=[p for p in parents if p.get('k') and int(p['k'])==k]
                info=dict(N=summary.get('test_parent_count',summary.get('parent_count',r.get('num_parents','N/A'))),
                    covered_count=r.get('covered_count',r.get('num_close_cf_covered','N/A')),
                    finite_recourse_count=r.get('finite_recourse_count',r.get('num_any_strict_flip_parents','N/A')))
                exact=False
                if derived:
                    computed,vals=reduce_distances(derived,theta,cap);info.update(computed)
                    if not close(cov,computed['coverage']) or (cost is not None and not close(cost,computed[metric])):
                        issues.append(f'{d}/{m}/K{k}: prefix/minima disagreement');all_valid=False
                    exact=True;any_exact=True
                    if k in (10,20):
                        curves.extend(dict(dataset=d,method=m,k=k,theta_star=theta,source_root=base['source_root'],
                            plot_allowed=usable and not cm,**v) for v in exact_curve(vals,theta))
                row=dict(dataset=d,method=m,k=k,coverage=cov if cov is not None else 'N/A',cost=cost if cost is not None else 'N/A',
                    cost_definition=base['cost'],theta_star=theta,cap=cap if cap is not None else 'N/A',
                    N=info['N'],covered_count=info['covered_count'],finite_recourse_count=info['finite_recourse_count'],
                    effective_k=r.get('effective_k',r.get('valid_k',min(k,summary['effective_rule_count']) if summary.get('effective_rule_count') else 'N/A')),variant=base['variant'],scope=base['scope'],
                    record_reconciled=exact,plot_allowed=usable and not cm,stage=base['stage'],source_root=base['source_root'])
                prefix.append(row)
            if not any_exact:
                base.update(stage='LEGACY_NUMERIC_EXCEPTION_NOT_FRESH_REVIEW',science_audit='USER_APPROVED_FROZEN_V4',
                    scope=base['scope']+'; parent minima absent; K20 exact ECDF unavailable')
                issues.append(f'{d}/{m}: parent minima missing, no K20 exact ECDF')
            if not all_valid:
                base['stage']='SOURCE_CONFLICT'
                for r in prefix:
                    if r['dataset']==d and r['method']==m:r['plot_allowed']=False
                for r in curves:
                    if r['dataset']==d and r['method']==m:r['plot_allowed']=False
            base.update(scientific_result_with_scope=usable and any_exact and all_valid,
                paper_release=usable and any_exact and all_valid,cohort=info['N'])
            states.append(base)
    issues.append('CM/original BACE combined panel: numerical WNode producer binding still absent; CM standalone retained')
    if not allow_partial and (issues or not all(s['paper_release'] for s in states)):
        raise ValueError('Incomplete or unreconciled paper family; --allow-partial required')
    out.mkdir(parents=True,exist_ok=True);csvroot=out/'source_csv';csvroot.mkdir(exist_ok=True)
    write_csv(out/'matrix20_status.csv',states);write_csv(csvroot/'figure3.csv',prefix)
    for k in (10,20):
        write_csv(csvroot/f'figure4_k{k}_exact.csv',[r for r in curves if r['k']==k])
        table=[]
        for cell in states:
            row=next((r.copy() for r in prefix if r['dataset']==cell['dataset'] and r['method']==cell['method_family'] and r['k']==k),None)
            if row is None:row=dict(dataset=cell['dataset'],method=cell['method_family'],k=k,coverage='PENDING',cost='PENDING',
                finite_recourse_count='PENDING',N='PENDING',effective_k='PENDING',stage=cell['stage'])
            if cell['stage']=='UNDER_REPAIR':
                row.update(coverage='UNDER_REPAIR',cost='UNDER_REPAIR',finite_recourse_count='UNDER_REPAIR')
            table.append(row)
        write_csv(out/('table2_k20.csv' if k==20 else 'table2_k10_aux.csv'),table)
    counts=dict(registered_count=sum(s['registered'] for s in states),
        scientific_results_with_scope_count=sum(s['scientific_result_with_scope'] for s in states),
        paper_release_count=sum(s['paper_release'] for s in states),total_cells=20,
        paper_release_count_definition='per-cell saved-prefix/raw-minima consistency with retained scope; not complete combined20 release',
        all20_complete=False,scientific_authority_created=False,authority_written=False,read_only=True,
        original_authority_count=pointer['latest_count'],issues=issues,source_pointer=pointer)
    (out/'matrix20_status.json').write_text(json.dumps(dict(**counts,cells=states),indent=2))
    render(out,prefix,curves,states)
    (out/'paper_scope.md').write_text('''# PARTIAL original RF/GINE paper family — K20

This is a read-only projection, not a second authority. Registered counts do not
erase historic scientific exceptions. Original AIDS/Mut legacy numeric rows
retain USER_APPROVED_FROZEN_V4 scope; their missing parent records prevent a
new exact K20 ECDF claim. BACE original GlobalGCE materialization and AIDS
RF-aligned ComRecGC remain UNDER_REPAIR, not measured valid zero here.

CM v1 BACE is accepted as full-graph class-level prototypes (not one local
action applied to all parents). Its source CSV is included but its curve is
not overlaid until old-method numerical WNode producer bindings close.
Use the separately delivered accepted CM figures. RF/GINE and GIN/A+ families
are never mixed. CM v1/v2 replace the same cell; AIDS corrections do not add a cell.

CM-CReM is adapted to a class-level prototype summary. We use the same frozen
predictor as the other methods on each dataset. GNN predictors use Grad-CAM;
RF predictors use fingerprint-environment feature occlusion for mask proposal.
Final molecules are re-encoded and verified by the original predictor. The
global library cap and K20 selection protocol are reported explicitly. BACE
development followed earlier test inspection; no untouched-test model-selection
claim is made. RF adaptation is under implementation validation, not yet a
completed four-dataset experiment.
''')
    return counts

def render(out,prefix,curves,states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(GlobalGCE='#c33a36',ComRecGC='#26833b',GCFExplainer='#111111',Ours='#b42eb4',**{'CM-CReM':'#257aa8'})
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42})
    def save(fig,name,footer):
        fig.tight_layout(rect=(0,.11,1,.95));fig.text(.5,.035,footer,ha='center',fontsize=8)
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=220)
        plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,5),squeeze=False)
    for i,d in enumerate(DATASETS):
        axes[0,i].set_title(d)
        for m in METHODS:
            rows=[r for r in prefix if r['dataset']==d and r['method']==m and r['plot_allowed']]
            if not rows:continue
            for j,key in enumerate(('coverage','cost')):
                valid=[r for r in rows if isinstance(r[key],(int,float))]
                axes[j,i].plot([r['k'] for r in valid],[r[key]*(100 if j==0 else 1) for r in valid],color=colors[m],label=m,linewidth=1.1)
        for j in range(2):axes[j,i].set_xlim(1,20);axes[j,i].set_ylim(bottom=0);axes[j,i].grid(alpha=.25)
        axes[0,i].set_ylabel('Coverage (%)');axes[1,i].set_ylabel('Capped cost' if d in ('BACE','TasteMolNet') else 'Conditional cost')
        axes[1,i].set_xlabel('Size (K)')
    legend={}
    for ax in axes[0]:
        handles,labels=ax.get_legend_handles_labels();legend.update(zip(labels,handles))
    fig.legend(legend.values(),legend.keys(),loc='upper center',ncol=4,frameon=False)
    save(fig,'figure3','PARTIAL | legacy numeric exceptions retained | CM standalone; numerical binding pending | missing methods are not zero')
    for k in (10,20):
        fig,axs=plt.subplots(1,4,figsize=(14,3.1))
        for ax,d in zip(axs,DATASETS):
            actual=[]
            for m in METHODS:
                rs=[r for r in curves if r['dataset']==d and r['method']==m and r['k']==k and r['plot_allowed']]
                if not rs:continue
                actual.extend(rs);ax.step([r['distance'] for r in rs],[r['coverage']*100 for r in rs],where='post',color=colors[m],label=m,lw=1.1)
            if actual:
                ax.set_xlim(0,max(r['distance'] for r in actual)*1.025);ax.axvline(actual[0]['theta_star'],ls=':',color='gray',lw=.7)
            if not actual or d in ('AIDS','Mutagenicity'):
                ax.text(.03,.95,'Missing legacy raw minima\nNo fabricated K20 ECDF',transform=ax.transAxes,va='top',fontsize=8)
            ax.set_title(d);ax.set_xlabel('Distance threshold');ax.set_ylabel('Coverage (%)');ax.set_ylim(bottom=0);ax.grid(alpha=.25)
            if actual:ax.legend(fontsize=7,frameon=False)
        save(fig,f'figure4_k{k}',f'PARTIAL | exact saved-distance ECDF, K={k} | CM standalone; numerical binding pending')
    rows=read_csv(out/'table2_k20.csv');body=[]
    def fmt(v,percent=False):
        try:return f'{float(v)*(100 if percent else 1):.2f}' if percent else f'{float(v):.6f}'
        except (ValueError,TypeError):return str(v or 'N/A')
    for r in rows:body.append([r['dataset'],r['method'],fmt(r['coverage'],True),fmt(r['cost']),r.get('finite_recourse_count') or 'N/A',r.get('N') or 'N/A',r.get('effective_k') or 'N/A'])
    fig,ax=plt.subplots(figsize=(11.7,7));ax.axis('off')
    t=ax.table(cellText=body,colLabels=['Dataset','Method','Cov@20 (%)','Original Cost@20','Finite recourse','N','Effective K'],loc='center')
    t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.35)
    save(fig,'table2_k20','PARTIAL | RF/GINE family only | scope and producer gaps retained in source CSV / paper_scope.md')
    with (out/'table2_k20.tex').open('w') as f:
        f.write('\\begin{tabular}{llrrrrr}\nDataset & Method & Cov@20 & Cost@20 & Finite & N & Effective K \\\\\n')
        for r in body:f.write(' & '.join(str(x).replace('_',r'\_') for x in r)+' \\\\\n')
        f.write('\\end{tabular}\n')
