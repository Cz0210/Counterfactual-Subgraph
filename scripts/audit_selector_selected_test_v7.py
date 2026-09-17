#!/usr/bin/env python3
"""Independent bounded CPU reexecution, then paired saved-distance statistics."""
import argparse,csv,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.eval.selector_selected_test_v7 import frozen_inputs,make_evaluator,validate_new_block
from src.eval.selector_controlled_v7 import digest,write_csv
from src.eval.bace_frozen_gnn_contracts import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--set',action='append',default=[])
    p.add_argument('--spec',required=True);p.add_argument('--adapter',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_ONLY')
    spec,root,orders,union=frozen_inputs(a.spec,'p0');adapter=json.loads(Path(a.adapter).read_text())
    if adapter['frozen_spec_sha256']!=digest(spec):raise ValueError('ADAPTER_BINDING')
    work=Path(spec['output_root'])/'independent-v7-audit'
    work.mkdir(exist_ok=False)
    candidates={r['candidate_id']:r for r in map(json.loads,Path(spec['candidate_universe']).read_text().splitlines())}
    with Path(spec['test_parent_csv']).open() as f:
        parents={}
        for i,r in enumerate(csv.DictReader(f)):
            pid=r.get('parent_id',r.get('molecule_id'));parents[pid]=dict(id=pid,smiles=r['smiles'],label=r['label'],index=i)
    completed=root/'selected-test-completion'
    paths=sorted((completed/'parent-blocks').glob('*.json'))
    # Deterministic bounded sample with at least one positive block if present;
    # no selection/choice of scientific version depends on audit outcomes.
    positive=[];negative=[]
    for path in paths:
        b=json.loads(path.read_text())
        (positive if any(r['pair_strict_flip'] for r in b['pairs']) else negative).append(path)
    selected=positive[:2]+negative[:2]
    if not selected:raise ValueError('NO_NEW_BLOCKS_TO_AUDIT')
    evaluate,provider=make_evaluator(spec,adapter,work)
    checked=0
    try:
        for path in selected:
            b=json.loads(path.read_text());pid=b['pairs'][0]['parent_id'];ids=[r['candidate_id'] for r in b['pairs']]
            actual=evaluate(parents[pid],[candidates[c] for c in ids])
            validate_new_block(actual,{(pid,c) for c in ids})
            before={r['candidate_id']:r for r in b['pairs']}
            for r in actual['pairs']:
                old=before[r['candidate_id']]
                for key in ['applicable','pair_strict_flip','pred_before','pred_after','best_match_index','residual_smiles',
                            'num_matches','num_valid_residuals','num_strict_flip_matches','p1_before','p1_after','wnode_distance','cf_drop']:
                    x,y=old.get(key),r.get(key)
                    same=(abs(x-y)<=1e-10 if isinstance(x,float) and isinstance(y,float) else x==y)
                    if not same:
                        atomic_json(work/'first_difference.json',dict(parent_id=pid,candidate_id=r['candidate_id'],field=key,saved=x,recomputed=y))
                        raise ValueError('INDEPENDENT_PAIR_MISMATCH:'+key)
                checked+=1
            atomic_json(work/(path.stem+'.json'),actual)
    finally:provider.close()
    # Original independent frozen-test audit tolerance is 1e-10; not fitted to differences.
    audit=dict(state='BOUNDED_INDEPENDENT_PAIR_REEXECUTION_PASS',pairs_checked=checked,
        parent_blocks=len(selected),comparison_abs_tolerance=1e-10,source_tolerance='mutagenicity_wnode_frozen_test._values_equal',
        whole_pool_oracle_reexecution=False,generation=0,training=0)
    atomic_json(work/'audit.json',audit)
    # Read only committed saved distances. Same resampled parent IDs for every contrast.
    tables=[];matrices={};parent_ids=None
    for phase in ['p0','p1']:
        path=Path(spec['output_root'])/phase/'selected-test-completion/parent_best_distances.csv'
        if not path.exists():continue
        with path.open() as f:rr=list(csv.DictReader(f))
        for sid in sorted({r['variant'] for r in rr}):
            v=[r for r in rr if r['variant']==sid and int(r['k'])==20]
            if parent_ids is None:parent_ids=[r['parent_id'] for r in v]
            if [r['parent_id'] for r in v]!=parent_ids:raise ValueError('PAIRED_PARENT_ORDER')
            matrices[sid]=np.array([float(r['raw_best_distance']) if r['raw_best_distance']!='INF_VERIFIED_FAILURE' else np.inf for r in v])
    rng=np.random.default_rng(7);sample=rng.integers(0,len(parent_ids),size=(1000,len(parent_ids)))
    for left,right in [('S5','S3'),('S5','S4'),('S6','S0'),('S6','S7'),('S6','S8'),('S6','S9')]:
        if left not in matrices or right not in matrices:continue
        dl,dr=matrices[left],matrices[right]
        for metric,x,y in [('coverage',dl<=.1,dr<=.1),('capped_mean',np.minimum(dl,spec['cost_cap']),np.minimum(dr,spec['cost_cap']))]:
            delta=np.asarray(x,float)-np.asarray(y,float);samples=delta[sample].mean(1)
            lo,hi=np.quantile(samples,[.025,.975]);tables.append(dict(dataset=spec['dataset'],contrast=left+'-'+right,metric=metric,
                estimate=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),replicates=1000,seed=7,denominator=len(parent_ids),
                scope='PAIRED_PARENTS_FIXED_MODELS_POOLS_AND_FROZEN_SELECTORS'))
    write_csv(work/'paired_bootstrap.csv',tables)
    print(json.dumps(audit))

if __name__=='__main__':main()
