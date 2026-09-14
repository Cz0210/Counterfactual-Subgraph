"""Explicit source-cohort CM disclosure; no model, distance or selector calls."""
from __future__ import annotations
import hashlib
import math

EXPERIMENT = 'CM-AIDS-SOURCE-DESCRIPTIVE-v1'

def selection_ids(generation_ids):
    ids=list(generation_ids)
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('Generation IDs must be a nonempty unique frozen set')
    return sorted(ids,key=lambda p:(hashlib.sha256(('CM-AIDS-SOURCE-CAL-v1|7|'+p).encode()).hexdigest(),p))[:math.ceil(.20*len(ids))]

def validate_scope(scope, provenance, calibration_ids, evaluation_ids):
    if scope.get('experiment_id')!=EXPERIMENT or scope.get('heldout') is not False:
        raise ValueError('Source-descriptive result cannot claim held-out scope')
    generation=provenance['generation_ids']
    if selection_ids(generation)!=list(calibration_ids) or list(calibration_ids)!=scope['selection_ids']:
        raise ValueError('Authorized pre-outcome hash subset changed')
    if list(evaluation_ids)!=scope['evaluation_ids'] or len(set(evaluation_ids))!=len(evaluation_ids):
        raise ValueError('Original complete base IDs/order changed')
    for key,actual in [('generation_count',len(generation)),('selection_count',len(calibration_ids)),
                       ('evaluation_count',len(evaluation_ids)),
                       ('selection_evaluation_overlap',len(set(calibration_ids)&set(evaluation_ids))),
                       ('generation_evaluation_overlap',len(set(generation)&set(evaluation_ids)))]:
        if scope.get(key)!=actual:raise ValueError('Incorrect scope count: '+key)
    if scope.get('generated_again') is not False or scope.get('filter_repeated') is not False:
        raise ValueError('This delivery cannot authorize regeneration/filtering')
    if scope.get('original_deadline_utc')!='2026-09-16T16:27:44Z':
        raise ValueError('Original deadline changed')
    return dict(scope=EXPERIMENT,heldout=False,main_scope_compatible=False,
                selection_evaluation_overlap=scope['selection_evaluation_overlap'],
                generation_evaluation_overlap=scope['generation_evaluation_overlap'])
