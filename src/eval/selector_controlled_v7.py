"""Calibration-only Controlled-v1 selectors over an authenticated complete matrix.

No generator, oracle or distance implementation is changed here. A missing or
failed numerical record is not a biological non-flip. Test records are opened
only after every requested calibration sequence has been sealed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.eval.bace_frozen_gnn_contracts import atomic_json
from src.eval.mutagenicity_wnode_selector import build_candidate_chemistry

THETAS = np.array([.025, .05, .075, .1, .15, .2, .4])
REGS = [(0,0,0)] * 6 + [(1,1,1), (0,1,1), (1,0,1), (1,1,0)]
VALID_FAILURES = {
    'no_substructure_match', 'no_teacher_strict_flip_match',
    'no_valid_connected_residual', 'no_valid_residual',
    'no_strict_flip_match', 'parent_not_source_class',
    'no_substructure_match_or_fragment_parse_failed',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_matrix(path, candidate_ids, *, expected_parents=None, subset=False):
    """Read actual terminal pair rows; retain UNKNOWN separately and reject it."""
    ci = {x: i for i, x in enumerate(candidate_ids)}
    parents, records, failures = [], {}, {}
    for row in rows(path):
        cid, pid = row['candidate_id'], row['parent_id']
        if cid not in ci:
            if subset:
                continue
            raise ValueError('UNBOUND_CANDIDATE:' + cid)
        if pid not in parents:
            parents.append(pid)
        key = (pid, cid)
        if key in records:
            raise ValueError('DUPLICATE_PAIR:' + str(key))
        if row.get('pair_strict_flip') is True:
            value = row.get('wnode_distance')
            if (not row.get('applicable') or value is None
                    or not np.isfinite(float(value)) or float(value) < 0):
                raise ValueError('INVALID_STRICT_RAW:' + str(key))
            records[key] = float(value)
        else:
            reason = row.get('failure_reason')
            # The older BACE producer shares this label between genuine
            # non-flips and strict flips with a missing numerical distance.
            # Only its explicit zero strict-match count proves the former.
            proven_bace_nonflip = (reason == 'no_valid_strict_flip_with_finite_wnode'
                                  and row.get('num_strict_flip_matches') == 0)
            if reason not in VALID_FAILURES and not proven_bace_nonflip:
                raise ValueError('UNCLASSIFIED_NOT_INFINITY:' + str(key) + ':' + str(reason))
            failures[reason] = failures.get(reason, 0) + 1
            records[key] = np.inf
    if expected_parents is not None:
        if set(parents) - set(expected_parents):
            raise ValueError('UNBOUND_PARENT')
        parents = list(expected_parents)
    d = np.full((len(parents), len(ci)), np.nan)
    for p, pid in enumerate(parents):
        for c, cid in enumerate(candidate_ids):
            if (pid, cid) in records:
                d[p, c] = records[pid, cid]
    return parents, d, failures


class Objective:
    def __init__(self, distances, chemistry):
        self.d = distances
        self.covered = distances <= .1
        x = self.covered.astype(np.int64)
        intersection = x.T @ x
        union = x.sum(0)[:, None] + x.sum(0)[None, :] - intersection
        self.covred = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union != 0)
        self.struct = chemistry.structural_similarity
        self.size = chemistry.normalized_sizes

    def value(self, sequence, *, multi, prefix, reg=(0,0,0)):
        best = np.minimum.accumulate(self.d[:, sequence], axis=1)
        theta = THETAS if multi else np.array([.1])
        coverage = (best[None, :, :] <= theta[:, None, None]).mean(axis=(0,1))
        if any(reg):
            ix = np.array(sequence)
            for k in range(1, len(ix)+1):
                sub = ix[:k]
                pair = np.triu_indices(k, 1)
                cr = self.covred[np.ix_(sub, sub)][pair].mean() if k > 1 else 0.
                sr = self.struct[np.ix_(sub, sub)][pair].mean() if k > 1 else 0.
                coverage[k-1] -= .05*reg[0]*cr + .05*reg[1]*sr + .02*reg[2]*self.size[sub].mean()
        if not prefix:
            return float(coverage[-1])
        rho = np.array([1/15 if k < 10 else 1/30 for k in range(len(sequence))])
        # Short pools retain the specified prefix weights (not silent renormalization).
        return float(rho @ coverage)

    def greedy(self, candidate_ids, *, multi, members=None):
        available = sorted(range(len(candidate_ids)) if members is None else members,
                           key=lambda i: candidate_ids[i])
        result = []
        while available and len(result) < 20:
            scores = [self.value(result+[x], multi=multi, prefix=False) for x in available]
            choice = available[int(np.argmax(scores))]
            result.append(choice)
            available.remove(choice)
        return result

    def refine(self, initial, *, multi, prefix, reg=(0,0,0), replacement=True,
               max_proposals=20000, max_accepted=50):
        sequence = list(initial)
        score = self.value(sequence, multi=multi, prefix=prefix, reg=reg)
        rng = np.random.default_rng(7)
        count, accepted = 0, []
        while count < max_proposals and len(accepted) < max_accepted:
            proposals = [('swap', i, j) for i in range(len(sequence)) for j in range(i+1, len(sequence))]
            proposals += [('relocate', i, j) for i in range(len(sequence)) for j in range(len(sequence)) if i != j]
            if replacement:
                proposals += [('replace', i, j) for i in range(len(sequence))
                              for j in range(self.d.shape[1]) if j not in sequence]
            improved = False
            for ix in rng.permutation(len(proposals)):
                move, i, j = proposals[int(ix)]
                trial = sequence.copy()
                if move == 'swap':
                    trial[i], trial[j] = trial[j], trial[i]
                elif move == 'relocate':
                    trial.insert(j, trial.pop(i))
                else:
                    trial[i] = j
                value = self.value(trial, multi=multi, prefix=prefix, reg=reg)
                count += 1
                if value > score + 1e-12:
                    accepted.append(dict(proposal=count, move=move, i=i, j=j, before=score, after=value))
                    sequence, score, improved = trial, value, True
                    break
                if count >= max_proposals:
                    break
            if not improved:
                break
        return sequence, dict(proposals=count, accepted=len(accepted), moves=accepted,
            objective=score, global_optimality_claimed=False,
            stop_reason='BUDGET' if count >= max_proposals or len(accepted) >= max_accepted else 'ENUMERATED_NEIGHBORHOOD_NO_IMPROVEMENT')


def prefix_rows(d, order, cap, **tags):
    best = np.minimum.accumulate(d[:, order], axis=1)
    result = []
    for k in range(1, len(order)+1):
        x = best[:, k-1]
        finite = x[np.isfinite(x)]
        result.append(dict(**tags, k=k, effective_k=k, denominator=len(x),
            covered=int((x<=.1).sum()), coverage=float((x<=.1).mean()), finite=int(len(finite)),
            conditional_median=None if not len(finite) else float(np.median(finite)),
            capped_mean=float(np.minimum(x, cap).mean()), theta_star=.1, cost_cap=cap))
    return result


def write_csv(path, values):
    if not values:
        return
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def run(spec_path, *, phase='p0'):
    spec = json.loads(Path(spec_path).read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(spec['science_cutoff_utc']):
        raise ValueError('V7_CUTOFF_REACHED')
    root = Path(spec['output_root']) / phase
    root.mkdir(parents=True, exist_ok=False)
    candidates = sorted(list(rows(spec['candidate_universe'])), key=lambda r:r['candidate_id'])
    ids = [r['candidate_id'] for r in candidates]
    if len(set(ids)) != len(ids) or len(ids) != spec['expected_candidate_count']:
        raise ValueError('POOL_IDENTITY_OR_COUNT_CHANGED')
    parents, d, failures = load_matrix(spec['calibration_matrix'], ids)
    if d.shape != (spec['expected_calibration_count'], len(ids)) or np.isnan(d).any():
        raise ValueError('INCOMPLETE_DCAL_UNKNOWN_NOT_INF:' + str(d.shape))
    # Actual matrix semantic identity is captured once, not repeated per variant.
    matrix_sha = hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()
    atomic_json(root/'input_binding.json', dict(spec=spec, spec_sha=digest(spec),
        calibration_parent_ids=parents, candidate_ids=ids, matrix_semantic_sha=matrix_sha,
        failure_funnel=failures, complete_pairs=int(d.size), raw_distance_reused=True,
        candidate_generation_repeated=False, test_loaded=False))
    objective = Objective(d, build_candidate_chemistry(candidates))
    sequences, all_metrics = {}, []
    initial = {0:objective.greedy(ids,multi=False), 1:objective.greedy(ids,multi=True)}
    variants = range(7) if phase == 'p0' else range(7,10)
    for variant in variants:
        started = time.monotonic()
        if variant < 2:
            seq, stats = initial[variant], dict(proposals=0, accepted=0, stop_reason='FIXED_LENGTH_GREEDY')
        else:
            seq, stats = objective.refine(initial[0 if variant in (2,4) else 1],
                multi=variant not in (2,4), prefix=variant >= 4, reg=REGS[variant])
        sid = 'S'+str(variant)
        sequences[sid] = seq
        freeze = dict(variant=sid, ordered_candidate_ids=[ids[i] for i in seq],
            theta_star=.1, thresholds=THETAS.tolist() if variant not in (0,2,4) else [.1],
            matrix_semantic_sha=matrix_sha, selector='Controlled-v1', seed=7,
            lambda_cost=0, reg=list(REGS[variant]), stats=stats,
            selector_seconds=time.monotonic()-started, test_loaded=False)
        freeze['freeze_sha256'] = digest(freeze)
        atomic_json(root/(sid+'_freeze.json'),freeze)
        all_metrics.extend(prefix_rows(d, seq, spec['cost_cap'], dataset=spec['dataset'], variant=sid, split='calibration'))
        atomic_json(root/'progress.json',dict(state='CALIBRATION_SELECTION', completed_variants=list(sequences)))
    if spec['dataset']=='BACE' and phase=='p0':
        members = sequences['S3']
        ordered = objective.greedy(ids,multi=True,members=members)
        optimized, stats = objective.refine(ordered,multi=True,prefix=True,replacement=False)
        orders={'within_set_greedy':ordered,'prefix_refined':optimized}
        orders.update({f'random_seed_{s}':np.random.default_rng(s).permutation(members).tolist() for s in range(20)})
        values=[]
        terminals=[]
        for name, seq in orders.items():
            metrics=prefix_rows(d,seq,spec['cost_cap'],dataset=spec['dataset'],variant=name,split='calibration')
            values.extend(metrics)
            terminals.append(tuple(metrics[-1][k] for k in ('covered','finite','conditional_median','capped_mean')))
        if any(v!=terminals[0] for v in terminals):
            raise ValueError('FIXED_MEMBERS_K20_CHANGED')
        write_csv(root/'fixed_membership_prefix.csv',values)
        atomic_json(root/'fixed_membership_freeze.json',dict(orders={k:[ids[i] for i in v] for k,v in orders.items()},
            source='S3',test_loaded=False,k20_invariant=True,refinement=stats))
    write_csv(root/'calibration_prefix.csv',all_metrics)
    union=sorted(set(i for seq in sequences.values() for i in seq))
    atomic_json(root/'ALL_CALIBRATION_FROZEN.json',dict(variants=list(sequences),
        selected_union=[ids[i] for i in union], maximum_union=140 if phase=='p0' else 60,
        test_selection_allowed=False, matrix_semantic_sha=matrix_sha))
    # Consume saved test columns only after all calibration freezes. Missing
    # selected columns remain a precisely enumerated evaluation worklist.
    with Path(spec['test_parent_csv']).open() as stream:
        test_parents=[r['parent_id'] for r in csv.DictReader(stream)
                      if spec.get('test_label_filter') is None or str(r['label'])==str(spec['test_label_filter'])]
    if len(test_parents)!=spec['expected_test_count']:
        raise ValueError('TEST_BASE_COUNT_CHANGED:'+str(len(test_parents)))
    union_ids=[ids[i] for i in union]
    ptest, dt, _ = load_matrix(spec['saved_test_matrix'],union_ids,expected_parents=test_parents,subset=True)
    missing=np.argwhere(np.isnan(dt))
    write_csv(root/'missing_selected_test_pairs.csv',[
        dict(parent_id=ptest[p],candidate_id=union_ids[c]) for p,c in missing])
    complete = not len(missing)
    if complete:
        index={cid:i for i,cid in enumerate(union_ids)}
        test_metrics=[]
        for sid,seq in sequences.items():
            test_metrics.extend(prefix_rows(dt,[index[ids[i]] for i in seq],spec['cost_cap'],
                dataset=spec['dataset'],variant=sid,split='test'))
        write_csv(root/'test_prefix.csv',test_metrics)
    terminal=dict(state='CALIBRATION_FROZEN_TEST_COMPLETE' if complete else 'CALIBRATION_FROZEN_MISSING_SELECTED_TEST_PAIRS',
        calibration_units=len(sequences), test_complete=complete, selected_union_count=len(union),
        missing_test_pairs=int(len(missing)), test_denominator=len(ptest),
        new_oracle_calls=0,new_ot_calls=0,new_generation=0, next_stage='independent_audit' if complete else 'evaluate_missing_selected_test_pairs')
    atomic_json(root/'terminal.json',terminal)
    return terminal
