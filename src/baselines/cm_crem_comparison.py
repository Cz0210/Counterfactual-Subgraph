"""BACE-only saved-record comparison. No oracle, generation or selection calls."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from src.baselines.cm_crem_export import _file_sha, _read_csv, _write_csv, _write_json, replot
from src.baselines.cm_crem_selection import PAPER_LABEL, canonical_sha256

METHODS = ('GlobalGCE', 'ComRecGC', 'GCFExplainer', 'Ours')


def reduce_saved(rows, parent_ids, theta, cap):
    """Uncapped saved minima; a missing parent is never a failed recourse."""
    if len(rows) != len(parent_ids) or {r['parent_id'] for r in rows} != set(parent_ids):
        raise ValueError('Parent cohort mismatch or duplicate/missing row')
    values = []
    for row in rows:
        available = row['strict_recourse_available'].lower()
        if available not in ('true', 'false'):
            raise ValueError('Unknown strict-recourse status')
        raw = row['best_distance']
        value = math.inf if raw in ('N/A', 'inf', 'Infinity') else float(raw)
        if math.isnan(value) or value < 0 or (math.isfinite(value) != (available == 'true')):
            raise ValueError('Invalid uncapped minimum / strict-flip evidence')
        if float(row['theta_star']) != theta or float(row['cost_cap']) != cap:
            raise ValueError('Frozen threshold/cap mismatch')
        values.append(value)
    finite = sorted(v for v in values if math.isfinite(v))
    n = len(values)
    metrics = {'coverage': sum(v <= theta for v in values)/n,
               'fixed_capped_mean_cost': math.fsum(min(v, cap) for v in values)/n,
               'conditional_median_cost': statistics.median(finite) if finite else None,
               'finite_recourse_count': len(finite), 'base_parent_count': n}
    knots = sorted({0.0, *finite})
    ecdf = [{'distance': x, 'coverage': sum(v <= x for v in finite)/n} for x in knots]
    return metrics, ecdf


def legacy_records(spec, figure3, minima, excerpts):
    ids = spec['resolved_parents']['test']['ordered_ids']
    oracle, wn, ev = (spec[k] for k in ('resolved_oracle','resolved_wnode','resolved_evaluation'))
    if oracle['backbone'] != 'gine' or oracle['dataset'] != 'bace' or ev['cost_definition'] != 'FIXED_CAPPED_MEAN':
        raise ValueError('Only original BACE GINE fixed-capped comparison is permitted')
    metrics, curves, states = [], [], {}
    for method in METHODS:
        # Historical hard-materialization failure is NOT a measured valid zero.
        if method == 'GlobalGCE':
            states[method] = 'UNDER_REPAIR: original GINE hard-graph applicability evidence unresolved'
            continue
        source = excerpts['BACE/'+method]['summary']
        for key, value in {'classifier_family':'gine', 'source_label':1,
            'oracle_checkpoint_hash':oracle['model_sha256'],
            'molclr_checkpoint_hash':wn['molclr_checkpoint_sha256'],
            'distance_line':'MolCLR-Node-Wasserstein', 'theta_star':ev['theta'],
            'cost_cap':ev['cap'], 'test_parent_count':len(ids),
            'thresholds':ev['threshold_grid'], 'selection_frozen_before_test':True,
            'test_used_for_selection':False}.items():
            if source.get(key) != value:
                raise ValueError(f'{method}: incompatible or missing {key}')
        prior = None
        for k in range(1,21):
            rows = [r for r in minima if r['dataset']=='BACE' and r['method']==method and int(r['k'])==k]
            result, ecdf = reduce_saved(rows, ids, ev['theta'], ev['cap'])
            old = [r for r in figure3 if r['dataset']=='BACE' and r['method']==method and int(r['k'])==k]
            if len(old) != 1:
                raise ValueError(f'{method}/K{k}: source prefix missing/duplicated')
            for key in ('coverage', 'fixed_capped_mean_cost'):
                if not math.isclose(result[key],float(old[0][key]),abs_tol=1e-12,rel_tol=1e-12):
                    raise ValueError(f'{method}/K{k}: source CSV/minima conflict: {key}')
            current = {r['parent_id']:math.inf if r['best_distance'] in ('N/A','inf','Infinity') else float(r['best_distance']) for r in rows}
            if prior and any(current[p] > prior[p] for p in ids):
                raise ValueError('Non-nested saved parent minima')
            prior = current
            metrics.append({'dataset':'BACE','method':method,'k':k,**result})
            if k in (10,20):
                curves.extend({'dataset':'BACE','method':method,'k':k,**r} for r in ecdf)
        states[method] = 'SAVED_RECORDS_RECONCILED_NOT_NEW_SCIENTIFIC_REVIEW'
    return metrics, curves, states


def compare(*, spec_path, figure3_path, minima_path, excerpts_path, oracle_proof_path,
            output_dir, cm_import=None, allow_partial=False, distance_binding=None):
    spec = json.loads(Path(spec_path).read_text())
    proof = json.loads(Path(oracle_proof_path).read_text())
    temperature = proof['oracle_small_files']['temperature_scaling.json']
    card = proof['oracle_small_files']['model_card.json']
    # Original model-card/fit evidence, not manually rewritten classifier labels.
    if (temperature.get('temperature') != spec['resolved_oracle']['temperature'] or temperature.get('status') != 'fit'
            or card.get('checkpoint_id') != spec['resolved_oracle']['model_sha256']
            or temperature.get('selection_split') != 'validation' or temperature.get('test_used_for_fit') is not False):
        raise ValueError('Original validation-temperature proof mismatch')
    metrics, curves, states = legacy_records(spec, _read_csv(Path(figure3_path)), _read_csv(Path(minima_path)),
                                              json.loads(Path(excerpts_path).read_text()))
    output = Path(output_dir)
    if output.exists():
        raise ValueError('Use a fresh output directory')
    if cm_import is None and not allow_partial:
        raise ValueError('PENDING: accepted CM import required; use --allow-partial to show missing row explicitly')
    if cm_import is not None:
        require_distance_binding(distance_binding, spec)
    output.mkdir(parents=True)
    if cm_import is not None:
        root=Path(cm_import); receipt=json.loads((root/'cm_import_receipt.json').read_text())
        manifest=json.loads((root/'results/export_manifest.json').read_text())
        if (receipt.get('status')!='CM_RESULT_IMPORT_VERIFIED' or manifest.get('fixture') is not False
                or receipt.get('science_hash') != spec['science_hash']
                or manifest.get('contract_sha256') != spec['science_hash']
                or manifest.get('parent_ids_sha256') != canonical_sha256(spec['resolved_parents']['test']['ordered_ids'])
                or manifest.get('theta') != spec['resolved_evaluation']['theta']
                or manifest.get('cap') != spec['resolved_evaluation']['cap']):
            raise ValueError('CM import does not match this original-GINE campaign/cohort')
        # Existing authenticated replot performs all seven CM CSV binding checks.
        replot(root/'results/source_csv', output/'cm_only', dataset='bace')
        for row in _read_csv(root/'results/source_csv/prefix_metrics.csv'):
            metrics.append({'dataset':'BACE','method':PAPER_LABEL,'k':int(row['k']),
                **{key:(None if row[key]=='N/A' else float(row[key])) for key in ('coverage','fixed_capped_mean_cost','conditional_median_cost')},
                'finite_recourse_count':int(row['finite_recourse_count']), 'base_parent_count':int(row['base_parent_count'])})
        for k in (10,20):
            curves.extend({'dataset':'BACE','method':PAPER_LABEL,'k':k,'distance':float(r['distance']),'coverage':float(r['coverage'])}
                for r in _read_csv(root/f'results/source_csv/figure4_k{k}_exact.csv'))
        states[PAPER_LABEL]='ACCEPTED_CM_SAVED_RECORDS'
    else:
        states[PAPER_LABEL]='PENDING'
    source=output/'source_csv'; source.mkdir()
    _write_csv(source/'figure3.csv',metrics)
    for k in (10,20):
        _write_csv(source/f'figure4_k{k}_exact.csv',[r for r in curves if r['k']==k])
        table=[r for r in metrics if r['k']==k]
        for method,status in states.items():
            if not any(r['method']==method for r in table):
                table.append(dict.fromkeys(table[0],status)|{'dataset':'BACE','method':method,'k':k})
        _write_csv(source/f'table2_k{k}.csv',table)
    render(output,metrics,curves,states,spec['resolved_evaluation'])
    files=[Path(p) for p in (spec_path,figure3_path,minima_path,excerpts_path,oracle_proof_path)]
    result={'status':'PARTIAL_COMPARISON','scientific_pass_claimed':False,'main_matrix_written':False,
        'method_states':states,'source_files':{str(p):_file_sha(p) for p in files},
        'interpretation':'Original GINE saved records; GlobalGCE unresolved, never replaced by GIN/A+. CM endpoints are global prototypes.',
        'distance_producer_binding': 'CONFIRMED' if cm_import else 'REQUIRED_BEFORE_CM_OVERLAY',
        'oracle_calls':0,'ot_calls':0,'selector_calls':0,'output_dir':str(output)}
    _write_json(output/'comparison_receipt.json',result)
    return result


def require_distance_binding(path, spec):
    """Producer evidence is required; same encoder SHA alone is insufficient."""
    if path is None:
        raise ValueError('SOURCE_PROVENANCE_GAP: old methods need numeric WNode producer bindings before CM overlay')
    path=Path(path)
    binding=json.loads(path.read_text())
    expected=spec['resolved_wnode']
    for method in ('Ours','GCFExplainer','ComRecGC'):
        row=binding['methods'][method]
        producer=path.parent/row['evidence_file']
        if producer.resolve().parent != path.parent.resolve() or _file_sha(producer)!=row['evidence_sha256']:
            raise ValueError('Unbound numeric producer evidence')
        evidence=json.loads(producer.read_text())
        if not evidence.get('producer_commit') or evidence.get('method')!=method:
            raise ValueError('Missing actual old-method producer identity')
        for key in ('numerical_contract_sha256','molclr_checkpoint_sha256','feature_cost','node_mass','size_penalty_beta','distance_solver'):
            if evidence.get(key)!=expected[key]:
                raise ValueError(f'{method}: numeric WNode producer conflict/missing {key}')


def render(output,metrics,curves,states,ev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Ours':'#d62728','GCFExplainer':'black','ComRecGC':'#228b22',PAPER_LABEL:'#167D9A'}
    with plt.rc_context({'font.size':9,'font.family':'DejaVu Sans','savefig.facecolor':'white'}):
        def save(fig,name):
            fig.tight_layout(rect=(0,.1,1,.93))
            fig.text(.5,.025,'PARTIAL: GlobalGCE UNDER_REPAIR'+('; CM PENDING' if states[PAPER_LABEL]=='PENDING' else ''),ha='center',fontsize=8)
            for ext in ('png','pdf'): fig.savefig(output/(name+'.'+ext),dpi=200)
            plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(6.8,5.4),sharex=True)
        for method,color in colors.items():
            rows=[r for r in metrics if r['method']==method]
            if not rows: continue
            for ax,key in zip(axes,('coverage','fixed_capped_mean_cost')):
                ax.plot([r['k'] for r in rows],[r[key]*(100 if key=='coverage' else 1) for r in rows],label=method,color=color,linewidth=1.2)
                ax.grid(alpha=.25); ax.set_xlim(1,20)
        axes[0].set_ylabel('Coverage (%)'); axes[0].set_ylim(bottom=0); axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Size (K)',ylabel='Fixed-capped cost',ylim=(0,ev['cap']*1.08))
        fig.suptitle('BACE | original frozen GINE'); save(fig,'figure3')
        for k in (10,20):
            fig,ax=plt.subplots(figsize=(6.8,3.4))
            right=max(ev['cap'],max(r['distance'] for r in curves if r['k']==k))*1.02
            for method,color in colors.items():
                rows=[r for r in curves if r['method']==method and r['k']==k]
                if not rows: continue
                ax.step([r['distance'] for r in rows]+[right],[100*r['coverage'] for r in rows]+[100*rows[-1]['coverage']],where='post',label=method,color=color)
            ax.set(xlabel='Distance threshold',ylabel='Coverage (%)',xlim=(0,right),ylim=(0,100))
            ax.axvline(ev['theta'],color='gray',linestyle=':',linewidth=.8); ax.grid(alpha=.25); ax.legend(fontsize=8)
            fig.suptitle(f'BACE | original frozen GINE | K={k}'); save(fig,f'figure4_k{k}')
            rows=_read_csv(output/f'source_csv/table2_k{k}.csv')
            body=[[r['method'],f"{100*float(r['coverage']):.2f}" if r['coverage'] not in states.values() else r['coverage'].split(':')[0],
                   f"{float(r['fixed_capped_mean_cost']):.6f}" if r['fixed_capped_mean_cost'] not in states.values() else 'N/A'] for r in rows]
            fig,ax=plt.subplots(figsize=(9,2.8)); ax.axis('off')
            tab=ax.table(cellText=body,colLabels=['Method','Coverage (%)','Fixed-capped cost'],loc='center',colWidths=[.44,.30,.26])
            tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.5)
            fig.suptitle(f'Table 2 | BACE GINE | K={k}');save(fig,f'table2_k{k}')
            with (output/f'table2_k{k}.tex').open('x') as stream:
                stream.write('\\begin{tabular}{lrr}\nMethod & Coverage (\\%) & Fixed-capped cost \\\\\n')
                for row in body: stream.write(' & '.join(v.replace('_',r'\_') for v in row)+' \\\\\n')
                stream.write('\\end{tabular}\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','figure3','minima','excerpts','oracle-proof','output-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--cm-import',type=Path);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--distance-binding',type=Path)
    p.add_argument('--config');p.add_argument('--set',action='append',default=[])
    a=p.parse_args()
    print(json.dumps(compare(spec_path=a.spec,figure3_path=a.figure3,minima_path=a.minima,
        excerpts_path=a.excerpts,oracle_proof_path=a.oracle_proof,output_dir=a.output_dir,
        cm_import=a.cm_import,allow_partial=a.allow_partial,distance_binding=a.distance_binding),indent=2))
