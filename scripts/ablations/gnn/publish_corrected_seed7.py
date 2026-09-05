#!/usr/bin/env python3
"""Publish only small seed7 paper tables from an independently accepted import.

This is an additive publication adapter, not another science verifier. The
existing portable acceptance is adopted; weights/OT/archive are not rescanned.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.ablations.contracts import canonical_json_sha256

TABLES = ('gnn_seed7_classifier_table.csv', 'gnn_seed7_explanation_native.csv',
          'gnn_seed7_explanation_common.csv', 'gnn_seed7_temperature_fit_table.csv',
          'gnn_seed7_correction_diff.csv', 'gnn_seed7_table.tex',
          'gnn_seed7_corrective_audit.json')


def publish(import_root, acceptance_sha256, output_root, registry_root):
    source = Path(import_root).resolve(strict=True)
    acceptance = source / 'corrective_location_overlay.json'
    if sha256_file(acceptance) != acceptance_sha256:
        raise ValueError('ACCEPTANCE_RECEIPT_CHANGED')
    accepted = json.loads(acceptance.read_text())
    audit = json.loads((source / 'publication/gnn_seed7_corrective_audit.json').read_text())
    manifest = json.loads((source / 'package_manifest.json').read_text())
    if (accepted.get('state') != 'GNN_CORE_SEED7_CORRECTED_PASS'
            or audit.get('state') != accepted['state']
            or accepted.get('main_matrix_write') is not False
            or accepted.get('raw_ot_recomputed_count') != 0
            or not accepted.get('all_weights_unchanged') or not accepted.get('gine_unchanged')
            or not accepted.get('selectors_frozen_before_test')
            or accepted.get('corrective_audit_sha256') != sha256_file(source/'publication/gnn_seed7_corrective_audit.json')):
        raise ValueError('INDEPENDENT_CORRECTED_ACCEPTANCE_REQUIRED')
    output, registry = Path(output_root).absolute(), Path(registry_root).absolute()
    for p in (output, registry):
        if 'control' in p.parts or 'fast16_matrix_authority' in p.parts:
            raise ValueError('ABLATION_PUBLISHER_CANNOT_WRITE_MAIN_CONTROL')
        if p == source or source in p.parents or p in source.parents:
            raise ValueError('DO_NOT_WRITE_SEALED_IMPORT')
        if any(x.is_symlink() for x in (p, *p.parents)):
            raise ValueError('PHYSICAL_PUBLICATION_PATH_REQUIRED')
    files = {}
    for name in TABLES:
        rel = 'publication/' + name
        entry = manifest['files'][rel]
        actual = sha256_file(source/rel)
        if actual != entry['sha256']:
            raise ValueError('PAPER_TABLE_CHANGED:' + name)
        files[name] = actual
    payload = {'schema_version': 'bace_gnn_corrected_paper_publication_v1',
               'state': accepted['state'], 'dataset': 'bace', 'method': 'ours', 'seed': 7,
               'scope': 'PROPOSAL_FIXED_BACKBONE_SENSITIVITY',
               'accepted_import': str(source), 'acceptance_sha256': acceptance_sha256,
               'package_sha256': accepted['package_sha256'], 'files': files,
               'main_matrix_write': False, 'science_recomputed': False,
               'old_provisional_audit_modified': False, 'historical_missing_runtime': 'N/A',
               'frozen_trainable_count_is_training_parameter_count': False}
    payload['self_sha256'] = canonical_json_sha256(payload)
    if output.exists():
        if json.loads((output/'publication_receipt.json').read_text()) != payload:
            raise ValueError('PUBLICATION_CONFLICT')
    else:
        output.mkdir(parents=True)
        for name in TABLES:
            with (source/'publication'/name).open('rb') as src, (output/name).open('xb') as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
        atomic_json(output/'publication_receipt.json', payload)
    registry.mkdir(parents=True, exist_ok=True)
    with (registry/'publication.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = registry/'gnn_result_registry.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'schema_version':'bace_gnn_result_registry_v1', 'results': {}}
        if state['schema_version'] != 'bace_gnn_result_registry_v1':
            raise ValueError('GNN_REGISTRY_SCHEMA_MISMATCH')
        key = 'bace/ours/proposal_fixed/seed7/corrected'
        row = {'root':str(output), 'acceptance_sha256':acceptance_sha256,
               'publication_receipt_sha256':sha256_file(output/'publication_receipt.json'),
               'state':payload['state']}
        if key in state['results'] and state['results'][key] != row:
            raise ValueError('GNN_CORRECTED_REGISTRY_CONFLICT')
        state['results'][key] = row
        atomic_json(path, state)
    return {'state': payload['state'], 'paper_root': str(output),
            'registry':str(path), 'archive_or_model_rehashed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    for key in ('import-root','acceptance-sha256','output-root','registry-root'):
        p.add_argument('--'+key,required=True)
    a=vars(p.parse_args())
    if a.pop('config')!='configs/hpc.yaml': p.error('configs/hpc.yaml required')
    print(json.dumps(publish(**a),sort_keys=True))


if __name__=='__main__': main()
