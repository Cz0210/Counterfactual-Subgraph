"""Bounded maximum-coverage diagnostics and the authorized Taste selector.

Only calibration is opened during optimization. Published test data is read by
the separate old-baseline replay, never supplied to a selector or MILP.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack
from .ours_taste_focus_matrix import dump_json, read_json


def maximum_cover(a, k=20, seconds=120):
    """Incumbent is recounted. A minimization dual lower bound negates to a UB."""
    a = np.asarray(a, dtype=bool)
    n, p = a.shape
    union = int(a.any(axis=1).sum())
    if union == 0:
        return {'lower': 0, 'upper': 0, 'selected': [], 'status': 'OPTIMAL_EMPTY', 'gap': 0}
    constraints = vstack([hstack([-csr_matrix(a, dtype=float), eye(n, format='csr')]),
                          csr_matrix(np.r_[np.ones(p), np.zeros(n)][None,:])], format='csr')
    result = milp(np.r_[np.zeros(p), -np.ones(n)], integrality=np.ones(p+n),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(constraints, -np.inf, np.r_[np.zeros(n), k]),
                  options={'time_limit': float(min(seconds,120)), 'mip_rel_gap': 0.0})
    selected=[]
    if result.x is not None:
        assert np.max(np.abs(result.x-np.rint(result.x))) < 1e-5, 'NONINTEGRAL_INCUMBENT'
        selected = np.flatnonzero(result.x[:p] > .5).tolist()
        assert len(selected) <= k
    lower=int(a[:, selected].any(axis=1).sum()) if selected else 0
    dual=getattr(result,'mip_dual_bound',None)
    upper=union
    if dual is not None and np.isfinite(dual):
        upper=min(union, max(lower, math.ceil(-float(dual)+1e-8)))
    if result.status == 0:
        assert result.fun is not None and abs(-result.fun-lower)<1e-5
        upper=lower
    return {'lower':lower,'upper':upper,'selected':selected,'status':str(result.message),
            'scipy_status':int(result.status),'optimal':result.status==0,
            'negative_objective_dual_bound':float(dual) if dual is not None and np.isfinite(dual) else None,
            'gap':upper-lower, 'union_upper':union}


def ceilings(d, source, theta, k=20, seconds=120):
    unknown=np.isnan(d) & source[:,None]
    reports={}
    for name, a in [('theta', np.isfinite(d) & (d <= theta) & source[:,None]),
                    ('finite_recourse',np.isfinite(d) & source[:,None])]:
        result=maximum_cover(a,k,seconds)
        if unknown.any():
            result['upper']=int((a|unknown).any(axis=1).sum())
            result['upper_kind']='OPTIMISTIC_UNKNOWN_UNION'
            result['optimal']=False
        result['pool_lower']=int(a.any(axis=1).sum())
        result['pool_upper']=int((a|unknown).any(axis=1).sum())
        reports[name]=result
    return reports


def prefix(d, selected, theta, cap):
    best=np.full(d.shape[0], np.inf)
    result=[]
    for k in range(1,21):
        if k <= len(selected): best=np.minimum(best,d[:,selected[k-1]])
        finite=np.isfinite(best)
        result.append({'k':k,'effective_k':min(k,len(selected)),'parent_count':len(best),
                       'covered_count':int((best<=theta).sum()),'finite_count':int(finite.sum()),
                       'coverage':float((best<=theta).mean()),'reach':float(finite.mean()),
                       'capped_mean':float(np.minimum(best,cap).mean()),
                       'conditional_median':float(np.median(best[finite])) if finite.any() else 'N/A'})
    return result


class Selector:
    def __init__(self,d,ids,theta,cap,grid,sizes):
        assert not np.isnan(d).any(), 'SELECTION_REQUIRES_COMPLETE_MATRIX'
        self.d,self.ids,self.theta,self.cap,self.grid,self.sizes=d,ids,theta,cap,np.asarray(grid),sizes
        self.cover=d<=theta

    def key(self,seq):
        if not seq: return (0,0,self.cap,0,0,0,0,())
        best=np.minimum.accumulate(self.d[:,seq],axis=1)
        end=best[:,-1]
        sets=self.cover[:,seq].astype(np.int32)
        intersection=sets.T@sets
        count=sets.sum(axis=0)
        union=count[:,None]+count[None,:]-intersection
        mask=np.triu(np.ones(union.shape,dtype=bool),1)&(union>0)
        redundancy=float((intersection[mask]/union[mask]).mean()) if mask.any() else 0.
        mean_prefix=float(np.r_[np.mean(best<=self.theta,axis=0),
                                     np.repeat(np.mean(end<=self.theta),20-len(seq))].mean())
        return (-int((end<=self.theta).sum()), -int(np.isfinite(end).sum()),
                float(np.minimum(end,self.cap).mean()),
                -float((end[:,None]<=self.grid).mean()), -mean_prefix,
                redundancy,float(np.mean([self.sizes[i] for i in seq])),tuple(self.ids[i] for i in seq))

    def greedy(self,allowed=None,initial=None):
        selected=list(initial or [])
        remaining=set(range(len(self.ids)) if allowed is None else allowed)-set(selected)
        while remaining and len(selected)<20:
            winner=min(remaining,key=lambda i:self.key(selected+[i]))
            selected.append(winner); remaining.remove(winner)
        return selected

    def optimize(self,old,incumbents,seconds=300,proposals=10000,accepts=50):
        start=time.monotonic()
        initial=[old,self.greedy()]
        for inc in incumbents:
            initial.append(self.greedy(initial=self.greedy(allowed=inc)))
        selected=min(initial,key=self.key)
        log=[]; proposed=0
        while len(log)<accepts and proposed<proposals and time.monotonic()-start<seconds:
            best_key=self.key(selected); best_seq=None
            for out in range(len(selected)):
                for incoming in range(len(self.ids)):
                    if incoming in selected: continue
                    if proposed>=proposals or time.monotonic()-start>=seconds: break
                    proposed+=1
                    trial=selected.copy(); trial[out]=incoming
                    key=self.key(trial)
                    if key<best_key:
                        best_key,best_seq=key,trial
                if proposed>=proposals or time.monotonic()-start>=seconds: break
            if best_seq is None: break
            log.append({'kind':'1out_1in','before':[self.ids[i] for i in selected],
                        'after':[self.ids[i] for i in best_seq],'objective_before':self.key(selected)[:-1],
                        'objective_after':best_key[:-1]})
            selected=best_seq
        reordered=self.greedy(allowed=selected)
        if len(log)<accepts and self.key(reordered)<self.key(selected):
            log.append({'kind':'same_set_greedy_reorder','before':[self.ids[i] for i in selected],
                        'after':[self.ids[i] for i in reordered]}); selected=reordered
        # Explicit bounded position moves, preserving the final set.
        for i in range(len(selected)):
            for j in range(len(selected)):
                if proposed>=proposals or len(log)>=accepts or time.monotonic()-start>=seconds: break
                trial=selected.copy(); trial.insert(j,trial.pop(i)); proposed+=1
                if self.key(trial)<self.key(selected):
                    log.append({'kind':'same_set_move','from':i,'to':j}); selected=trial
        return selected, {'proposal_count':proposed,'accepted_changes':log,'elapsed_seconds':time.monotonic()-start,
                          'claim_global_optimal':False,'initialization_count':len(initial),
                          'objective':self.key(selected)[:-1]}


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(source,output):
    source,output=Path(source),Path(output)
    assert not output.exists(), 'FRESH_OUTPUT_REQUIRED'
    output.mkdir(parents=True)
    contract=read_json(source/'contract.json')
    data=np.load(source/'calibration.npz',allow_pickle=False)
    d=data['distances']; ids=data['candidates'].tolist(); native=data['predictions']==1
    assert np.all(~np.isfinite(d[~native,:]))
    theta,cap=contract['theta_star'],contract['cost_cap']
    bound=ceilings(d,native,theta)
    report={'scope':'P0_CALIBRATION_FINITE_WNODE_MATRIX_ONLY','parents':len(d),'source_count':int(native.sum()),
            'source_fraction_upper':float(native.mean()),'rules':len(ids),'unknown_pairs':int(np.isnan(d).sum()),
            'theta_star':theta,'cost_cap':cap,'bounds':bound,'full_delete_space_upper':'UNKNOWN',
            'train_full_pool_upper':'UNKNOWN_UNCOMPUTED','test_optimizer_used':False}
    dump_json(output/'bounds.json',report)
    old=[ids.index(x) for x in read_json(source/'original_selection.json')['ordered_rule_ids']]
    pools=read_json(source/'candidate_pool.json')
    from rdkit import Chem
    sizes=[Chem.MolFromSmiles(r['canonical_fragment']).GetNumHeavyAtoms() for r in pools]
    selector=Selector(d,ids,theta,cap,contract['theta_grid'],sizes)
    new,trace=selector.optimize(old,[b['selected'] for b in bound.values()])
    dump_json(output/'selection_B.json',{'state':'CALIBRATION_FROZEN_P0_B','candidate_ids':[ids[i] for i in new],
             'selector':'CoverageFirstMultiBudget-v1','pool':'P0','test_read':False,'trace':trace,
             'scope':'B_ONLY_NOT_FINAL_A_B_C_D_RECOMMENDATION','calibration_objective':selector.key(new)[:-1],
             'pool_content_sha':read_json(source/'original_selection.json')['candidate_universe_sha256']})
    rows=[{'variant':v,**r} for v,s in [('A',old),('B',new)] for r in prefix(d,s,theta,cap)]
    write_csv(output/'calibration_prefix.csv',rows)
    # This replay does not evaluate B or optimize anything on test.
    test=np.load(source/'test.npz',allow_pickle=False)
    assert test['candidates'].tolist()==[ids[i] for i in old]
    old_test=prefix(test['distances'],list(range(len(old))),theta,cap)
    write_csv(output/'old_test_prefix_replayed.csv',old_test)
    dump_json(output/'terminal.json',{'state':'P0_BOUNDS_AND_B_SELECTOR_COMPLETE','science_complete_for_ABCD':False,
             'train_matrix':'MISSING_FULL_MATRIX','new_pool':'NOT_GENERATED','new_test':'NOT_OPENED',
             'new_oracle_queries':0,'new_ot_queries':0,'old_test_k20':old_test[-1],
             'next_stage':'TRAIN_DEVELOPMENT_MATRIX_AND_BOUNDED_EXPANSION','main_authority_write':False})
    print(json.dumps({'state':'P0_BOUNDS_AND_B_SELECTOR_COMPLETE','bounds':report,
                      'calibration_B_K20':rows[-1],'old_test_K20':old_test[-1]},ensure_ascii=False))
