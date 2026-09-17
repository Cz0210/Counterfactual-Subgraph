"""Independent selection-only timing, authenticated against V7 frozen orders.

This never opens test data and cannot replace a scientific freeze. Greedy
initialization is timed for every configuration, including S0 and S1.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.eval import selector_controlled_v7 as original


def configuration(variant):
    if variant not in range(10):
        raise ValueError("UNKNOWN_CONTROLLED_VARIANT")
    return dict(multi=variant not in (0, 2, 4), prefix=variant >= 4,
                reg=original.REGS[variant])


def timed_selection(objective, ids, variant):
    cfg = configuration(variant)
    start = time.perf_counter()
    initial = objective.greedy(ids, multi=cfg['multi'])
    greedy_end = time.perf_counter()
    if variant < 2:
        sequence, stats = initial, dict(proposals=0, accepted=0)
    else:
        sequence, stats = objective.refine(initial, **cfg)
    end = time.perf_counter()
    return sequence, dict(greedy_seconds=greedy_end-start,
                         refinement_seconds=end-greedy_end,
                         selection_seconds=end-start, stats=stats)


def validate_binding(spec, ids, parents, distances, binding):
    matrix_sha = hashlib.sha256(np.ascontiguousarray(distances).tobytes()).hexdigest()
    if (ids != binding['candidate_ids'] or parents != binding['calibration_parent_ids']
            or matrix_sha != binding['matrix_semantic_sha']
            or original.digest(spec) != binding['spec_sha']):
        raise ValueError('FROZEN_INPUT_BINDING_CHANGED')
    return matrix_sha


def run(spec_paths, output_root, *, wall_budget_seconds=6600):
    if not 0 < wall_budget_seconds <= 6600:
        raise ValueError('V8_TIMING_WALL_BUDGET_EXCEEDED')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    all_rows = []
    campaign_start = time.monotonic()
    source_sha = hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
    for spec_path in spec_paths:
        start_load = time.perf_counter()
        spec = json.loads(Path(spec_path).read_text())
        cutoff = datetime.fromisoformat(spec['science_cutoff_utc'])
        if datetime.now(timezone.utc) >= cutoff:
            raise ValueError('SCIENCE_CUTOFF_REACHED')
        frozen = Path(spec['output_root'])
        binding = json.loads((frozen/'p0/input_binding.json').read_text())
        candidates = sorted(original.rows(spec['candidate_universe']), key=lambda r:r['candidate_id'])
        ids = [r['candidate_id'] for r in candidates]
        parents, distances, _ = original.load_matrix(spec['calibration_matrix'], ids)
        if spec.get('calibration_parent_csv'):
            parents, distances, _ = original.complete_non_source_rows(spec, parents, distances)
        if np.isnan(distances).any():
            raise ValueError('UNKNOWN_DCAL')
        matrix_sha = validate_binding(spec, ids, parents, distances, binding)
        load_seconds = time.perf_counter()-start_load
        start_prepare = time.perf_counter()
        objective = original.Objective(distances, original.build_candidate_chemistry(candidates))
        prepare_seconds = time.perf_counter()-start_prepare
        for variant in range(10):
            if (time.monotonic()-campaign_start >= wall_budget_seconds
                    or datetime.now(timezone.utc) >= cutoff):
                original.atomic_json(root/'terminal.json', dict(state='BUDGET_STOP', completed=len(all_rows)))
                return all_rows
            sid = f'S{variant}'
            phase = 'p0' if variant < 7 else 'p1'
            prior = json.loads((frozen/phase/(sid+'_freeze.json')).read_text())
            supplied_sha = prior.pop('freeze_sha256')
            if original.digest(prior) != supplied_sha:
                raise ValueError('FREEZE_DIGEST_CHANGED')
            if prior['matrix_semantic_sha'] != matrix_sha:
                raise ValueError('PRIOR_FREEZE_MATRIX_CHANGED')
            sequence, timing = timed_selection(objective, ids, variant)
            order = [ids[i] for i in sequence]
            matches = order == prior['ordered_candidate_ids']
            out = dict(dataset=spec['dataset'], variant=sid, repetition=1,
                       scope='NEW_INDEPENDENT_SELECTION_ONLY_TIMING',
                       loading_seconds_shared=load_seconds,
                       objective_preparation_seconds_shared=prepare_seconds,
                       **timing, matrix_sha256=matrix_sha,
                       selector_source_sha256=source_sha,
                       ordered_candidate_ids=order,
                       order_sha256=original.digest(order),
                       prior_freeze_sha256=supplied_sha,
                       sequence_matches=matches, attachable=matches,
                       test_loaded=False, oracle_calls=0, ot_calls=0)
            write_start = time.perf_counter()
            original.atomic_json(root/(spec['dataset']+'-'+sid+'.json'), out)
            write_seconds = time.perf_counter()-write_start
            row = {k:v for k,v in out.items() if k not in ('stats','ordered_candidate_ids')}
            row['write_seconds'] = write_seconds
            all_rows.append(row)
            original.atomic_json(root/'progress.json', dict(state='TIMING', completed=len(all_rows),
                current=spec['dataset']+'/'+sid, last_sequence_matches=matches))
            if not matches:
                original.write_csv(root/'timing.csv',all_rows)
                original.atomic_json(root/'terminal.json',dict(state='ORDER_MISMATCH_NOT_ATTACHABLE',
                    dataset=spec['dataset'],variant=sid))
                return all_rows
    original.write_csv(root/'timing.csv', all_rows)
    original.atomic_json(root/'terminal.json', dict(state='PASS', configurations=len(all_rows),
        elapsed_seconds=time.monotonic()-campaign_start,
        measurement_scope='selection_only_not_end_to_end', repetitions=1,
        deadline_utc='2026-09-24T15:59:59+00:00'))
    return all_rows
