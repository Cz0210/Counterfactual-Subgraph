#!/usr/bin/env python3
"""Saved-data-only fixed-membership, redundancy, size and efficiency diagnostics."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.eval.selector_controlled_v7 import (rows,load_matrix,complete_non_source_rows,
    base_parent_ids,Objective,prefix_rows,write_csv)
from src.eval.selector_selected_test_v7 import frozen_inputs
from src.eval.mutagenicity_wnode_selector import build_candidate_chemistry
from src.eval.bace_frozen_gnn_contracts import atomic_json

def process(spec_path):
    spec,root,orders,union=frozen_inputs(spec_path,'p0')
    _,_,other,_=frozen_inputs(spec_path,'p1');orders.update(other)
    dest=Path(spec['output_root'])/'saved-data-diagnostics';dest.mkdir(exist_ok=False)
    candidates=sorted(rows(spec['candidate_universe']),key=lambda r:r['candidate_id']);ids=[r['candidate_id'] for r in candidates]
    parents,d,_=load_matrix(spec['calibration_matrix'],ids)
    if spec.get('calibration_parent_csv'):parents,d,_=complete_non_source_rows(spec,parents,d)
    if np.isnan(d).any():raise ValueError('DCAL_INCOMPLETE')
    obj=Objective(d,build_candidate_chemistry(candidates));lookup={c:i for i,c in enumerate(ids)}
    report=[]
    for sid,seq in orders.items():
        order=[lookup[c] for c in seq];n=len(order);pairs=np.triu_indices(n,1)
        phase='p0' if int(sid[1:])<7 else 'p1'
        freeze=json.loads((Path(spec['output_root'])/phase/(sid+'_freeze.json')).read_text())
        report.append(dict(dataset=spec['dataset'],variant=sid,rule_count=n,
            calibration_coverage_jaccard=float(obj.covred[np.ix_(order,order)][pairs].mean()) if n>1 else 0.,
            structural_tanimoto=float(obj.struct[np.ix_(order,order)][pairs].mean()) if n>1 else 0.,
            normalized_heavy_atom_size=float(obj.size[order].mean()),
            empty_calibration_coverage_rules=int((~obj.covered[:,order].any(0)).sum()),
            proposals=freeze['stats']['proposals'],accepted=freeze['stats']['accepted'],
            selector_seconds=freeze['selector_seconds'],scope='CALIBRATION_FIXED_POOL'))
    write_csv(dest/'redundancy_size_efficiency.csv',report)
    if spec['dataset']=='BACE':
        f=json.loads((root/'fixed_membership_freeze.json').read_text());members=set(orders['S3'])
        if any(set(s)!=members for s in f['orders'].values()):raise ValueError('FIXED_MEMBERS_CHANGED')
        selected=sorted(members);base=base_parent_ids(spec['test_parent_csv'],spec.get('test_label_filter'))
        _,dt,_=load_matrix(spec['saved_test_matrix'],selected,expected_parents=base,subset=True)
        pi={x:i for i,x in enumerate(base)};ci={x:i for i,x in enumerate(selected)}
        for phase in ['p0','p1']:
            for path in sorted((Path(spec['output_root'])/phase/'selected-test-completion/parent-blocks').glob('*.json')):
                for r in json.loads(path.read_text())['pairs']:
                    if r['candidate_id'] not in ci:continue
                    i,j=pi[r['parent_id']],ci[r['candidate_id']]
                    value=float(r['wnode_distance']) if r['pair_strict_flip'] else np.inf
                    if not np.isnan(dt[i,j]) and dt[i,j]!=value:raise ValueError('RAW_PAIR_CONFLICT')
                    dt[i,j]=value
        with Path(spec['test_predictions_csv']).open() as stream:
            before={r['parent_id']:int(r['predicted_label']) for r in csv.DictReader(stream)}
        for i,pid in enumerate(base):
            if before[pid]!=1:dt[i]=np.inf
        if np.isnan(dt).any():raise ValueError('FIXED_MEMBERS_TEST_GAPS')
        output=[];terminals=[]
        for name,seq in f['orders'].items():
            metrics=prefix_rows(dt,[ci[c] for c in seq],spec['cost_cap'],dataset='BACE',variant=name,split='test')
            output.extend(metrics);terminals.append(tuple(metrics[-1][k] for k in ['covered','finite','conditional_median','capped_mean']))
        if any(x!=terminals[0] for x in terminals):raise ValueError('K20_INVARIANCE_FAILED')
        write_csv(dest/'fixed_membership_test_prefix.csv',output)
        atomic_json(dest/'fixed_membership_test_audit.json',dict(orders=len(terminals),same_s3_members=True,k20_exact_invariant=True,new_oracle=0,new_ot=0))
    atomic_json(dest/'terminal.json',dict(dataset=spec['dataset'],variants=len(orders),new_oracle=0,new_ot=0,new_generation=0,status='SAVED_DATA_DIAGNOSTICS_COMPLETE'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--set',action='append',default=[]);p.add_argument('--spec',nargs='+',required=True)
    a=p.parse_args()
    for spec in a.spec:process(spec)

if __name__=='__main__':main()
