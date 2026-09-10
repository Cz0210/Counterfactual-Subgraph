"""Bounded calibration-only K20 selection; no model, generator or test access."""
from __future__ import annotations
import numpy as np
from .cm_crem_selection import SelectionFreeze, SelectionStep, _matrix, _matrix_sha, canonical_sha256


def objective(best, theta, cap, grid):
    best = np.asarray(best, dtype=np.float64)
    if np.isnan(best).any() or (best < 0).any():
        raise ValueError('Missing/NaN/negative distance is not semantic infinity')
    return (int(np.count_nonzero(best <= theta)), -float(np.minimum(best, cap).sum(dtype=np.float64)),
            float((len(grid)-np.searchsorted(grid, best, side='left')).sum()/ (len(best)*len(grid))))


def greedy(matrix, ids, theta, cap, grid, available=None):
    remaining = sorted(range(len(ids)) if available is None else available, key=lambda i: ids[i])
    chosen, best = [], np.full(matrix.shape[0], np.inf)
    while remaining and len(chosen) < 20:
        winner = min(remaining, key=lambda j: tuple(-x for x in objective(np.minimum(best,matrix[:,j]),theta,cap,grid))+(ids[j],))
        chosen.append(winner); remaining.remove(winner)
        best = np.minimum(best, matrix[:,winner])
    return chosen


def optimize(matrix, ids, theta, cap, grid, incumbent=()):
    """Greedy plus at most two complete best-improvement 1-out/1-in rounds."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (grid < 0).any() or np.any(grid[1:] <= grid[:-1]):
        raise ValueError('Actual frozen sorted threshold grid required')
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] != len(ids) or len(set(ids)) != len(ids) or len(ids)>6000:
        raise ValueError('Complete unique at-most6000 candidate matrix required')
    if np.isnan(matrix).any() or (matrix<0).any():raise ValueError('Unknown/invalid matrix')
    chosen = greedy(matrix,ids,theta,cap,grid)
    def score(s):return objective(np.min(matrix[:,s],axis=1) if s else np.full(matrix.shape[0],np.inf),theta,cap,grid)
    swaps=[]
    for iteration in range(2):
        if not chosen:break
        baseline=score(chosen); winning=None; winning_score=baseline
        selected=matrix[:,chosen]; argmin=selected.argmin(axis=1); first=selected.min(axis=1)
        second=np.partition(selected,1,axis=1)[:,1] if len(chosen)>1 else np.full(len(first),np.inf)
        for pos, dropped in enumerate(chosen):
            without=np.where(argmin==pos,second,first)
            for added in sorted(set(range(len(ids)))-set(chosen),key=lambda j:ids[j]):
                value=objective(np.minimum(without,matrix[:,added]),theta,cap,grid)
                if value <= baseline:continue
                candidate=chosen.copy();candidate[pos]=added
                tie=tuple(sorted(ids[j] for j in candidate))
                if value>winning_score or (value==winning_score and (winning is None or tie<winning[0])):
                    winning_score=value;winning=(tie,candidate,ids[dropped],ids[added])
        if winning is None:break
        chosen=winning[1];swaps.append({'round':iteration+1,'removed':winning[2],'added':winning[3],
                                     'before':baseline,'after':winning_score})
    if incumbent:
        if len(incumbent)!=min(20,len(ids)) or not set(incumbent)<=set(ids):raise ValueError('Incumbent not in pool')
        old=[ids.index(x) for x in incumbent]
        if score(old)>score(chosen) or (score(old)==score(chosen) and tuple(sorted(incumbent))<tuple(sorted(ids[j] for j in chosen))):
            chosen=old
    final=greedy(matrix,ids,theta,cap,grid,chosen)
    return final,{'objective':score(final),'swaps':swaps,'accepted_swap_count':len(swaps),
                 'pool_objective':objective(matrix.min(axis=1),theta,cap,grid),
                 'pool_finite_reach_count':int(np.isfinite(matrix.min(axis=1)).sum()),
                 'selection_used_test':False,'maximum_swap_rounds':2}


def freeze_k20(matrix, statuses, parents, ids, mask, theta, cap, grid, contract_sha, pool_sha, incumbent=()):
    matrix,statuses,mask=_matrix(matrix,statuses,parents,ids,mask)
    chosen,report=optimize(matrix,ids,theta,cap,grid,incumbent)
    best=np.full(len(parents),np.inf);steps=[]
    for k,j in enumerate(chosen,1):
        updated=np.minimum(best,matrix[:,j])
        steps.append(SelectionStep(k,ids[j],int(np.count_nonzero((updated<=theta)&(best>theta))),
            float(np.mean(np.minimum(best,cap)-np.minimum(updated,cap))),int(np.count_nonzero(updated<=theta)),
            float(np.minimum(updated,cap).mean())))
        best=updated
    raw=dict(schema_version='cm_crem_selection_freeze_k20_v2',method_id='CM-Global-K20-v2',
        contract_sha256=contract_sha,frozen_pool_sha256=pool_sha,pool_candidate_ids=ids,
        calibration_parent_ids=parents,calibration_source_mask=[bool(x) for x in mask],
        calibration_matrix_sha256=_matrix_sha(matrix,statuses),theta=theta,cap=cap,
        selected_candidate_ids=[ids[j] for j in chosen],steps=[vars(s) for s in steps])
    return SelectionFreeze.from_dict({**raw,'freeze_sha256':canonical_sha256(raw)}),report
