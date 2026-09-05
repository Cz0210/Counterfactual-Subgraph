"""BACE seed7, four frozen-weight first temperature fits and exact correction.

This dataset-specific repair never imports an OT implementation. Historical
per-match costs are reopened through their sealed parent/graph/encoder contract.
The original test has already been evaluated; this is a disclosed correction.
"""
from __future__ import annotations

import ast
import copy
import csv
from dataclasses import asdict
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time
import warnings
from typing import Any

import numpy as np

from src.ablations.gnn import cpu_evaluation as core
from src.ablations.gnn import scientific_verification as audit
from src.ablations.gnn.cpu_training import load_bundle, bundle_file
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json, atomic_csv, read_json, sha256_file, stable_sha256, utc_now,
)

ALTERNATIVES = ('gin', 'gcn', 'gatv2', 'gatedgcn_plus')
ORIGINAL_SHA = 'e40c9ee7a3e53f0db9635040b7fb7f09cf3fac22174444a16f743a7696e8cf63'
OLD_REVIEW_SHA = '7368b8d5d0d4db709fde2fc9790f805d49cb170eade9b545c3e1ab54a78ab0f6'
FITTER_SHA = 'cddd4ba7117b4b7d75e593263affacdc1efc7a15f741db457728677c180c2d7d'
FITTER_CONFIG = dict(objective='unweighted_cross_entropy', dtype='float64',
    parameterization='exp(log_temperature).clamp(1e-3,1e3)', initial_temperature=1.0,
    optimizer='torch.optim.LBFGS', lr=0.25, max_iter=100, line_search_fn='strong_wolfe',
    max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sealed(path, value):
    value = copy.deepcopy(value)
    value['self_sha256'] = stable_sha256(value)
    if Path(path).exists():
        require(read_json(path) == value, f'Sealed correction differs: {path}')
    else:
        atomic_json(path, value)
    return value


def reopen(path):
    value = read_json(path)
    require(value.get('self_sha256') == stable_sha256({k: v for k,v in value.items() if k != 'self_sha256'}),
            f'Correction self hash differs: {path}')
    return value


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def vector(value):
    return np.asarray(ast.literal_eval(value) if isinstance(value, str) else value, dtype=np.float64)


def scaled_probabilities(logits, temperature):
    raw = np.asarray(logits, dtype=np.float64)
    require(raw.ndim == 2 and raw.shape[1] == 2 and np.isfinite(raw).all(), 'RAW_LOGITS_SHAPE_OR_FINITE')
    require(math.isfinite(temperature) and temperature > 0, 'TEMPERATURE_NOT_POSITIVE_FINITE')
    scaled = raw / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    require(np.array_equal(raw.argmax(axis=1), probs.argmax(axis=1)), 'ARGMAX_CHANGED_AFTER_SCALING')
    return probs


def safe_archive_manifest(path):
    """Integrity only: this must not promote the old uncalibrated package."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'DUPLICATE_ARCHIVE_MEMBER')
        require(all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),
                'UNSAFE_ARCHIVE_MEMBER')
        manifest = json.load(archive.extractfile('package_manifest.json'))
        from src.ablations.contracts import canonical_json_sha256
        require(manifest['manifest_sha256'] == canonical_json_sha256({k:v for k,v in manifest.items() if k != 'manifest_sha256'}),
                'ARCHIVE_MANIFEST_HASH')
        require(set(names) == set(manifest['files']) | {'package_manifest.json'}, 'ARCHIVE_INVENTORY')
        for member in members:
            if member.name == 'package_manifest.json':
                continue
            expected = manifest['files'][member.name]
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            require(member.size == expected['size'] and digest.hexdigest() == expected['sha256'], 'ARCHIVE_FILE_HASH')
    return manifest


def plan(*, source_spec, original_package, output_root, authorization, driver_commit):
    from src.ablations.gnn.sharded_evaluation import scoped
    target = scoped(output_root)
    require(not target.exists(), 'FRESH_CORRECTION_ROOT_REQUIRED')
    auth = reopen(authorization)
    require(auth.get('authorized_by') == 'user_project_owner' and auth.get('backbones') == list(ALTERNATIVES)
            and auth.get('fit_split') == 'validation' and auth.get('expected_examples') == 187
            and auth.get('first_fit_authorized') is True, 'NARROW_FIRST_FIT_AUTHORIZATION_REQUIRED')
    spec = read_json(source_spec)
    require(spec['driver_commit'] == 'fd98c5f23bf835f2b68799d03b7a2fd8b8b713f7', 'WRONG_EXACT_SOURCE')
    require(sha256_file(original_package) == ORIGINAL_SHA and Path(original_package).stat().st_size == 27013606,
            'ORIGINAL_PACKAGE_CHANGED')
    inventory = safe_archive_manifest(original_package)
    root, manifest = load_bundle(spec['bundle_root'])
    source = Path(spec['evaluation_root'])
    require(sha256_file(root / 'bundle_manifest.json') == spec['bundle_file_sha256'], 'ORIGINAL_INPUT_BUNDLE_CHANGED')
    # Hashing sealed test files is not parsing them or using performance to choose a repair.
    for rel, ident in inventory['files'].items():
        if rel.startswith('evaluation/'):
            require(sha256_file(source / rel.removeprefix('evaluation/')) == ident['sha256'], 'SEALED_SOURCE_DRIFT')
    model_files = {}
    for name, model in spec['models'].items():
        files = {p.name: sha256_file(p) for p in Path(model).iterdir() if p.is_file()}
        for leaf, digest in spec['model_files'][name].items():
            require(files[leaf] == digest, 'ORIGINAL_CLASSIFIER_CHANGED')
        for leaf in ('model.pt','temperature_scaling.json','validation_predictions.csv'):
            require(files[leaf] == inventory['files'][f'classifiers/{name}/{leaf}']['sha256'], 'PACKAGE_MODEL_BINDING')
        model_files[name] = files
    target.mkdir(parents=True)
    shutil.copyfile(authorization, target / 'authorization.json')
    shutil.copyfile(source_spec, target / 'original_task_spec.json')
    atomic_json(target / 'original_package_manifest.json', inventory)
    contract = dict(schema='bace_seed7_temperature_repair_v1', repair_reason='MISSING_VALIDATION_TEMPERATURE_FIT',
        repair_scope='FOUR_ALTERNATIVE_FROZEN_CLASSIFIERS', original_package=str(Path(original_package).resolve()),
        original_package_sha256=ORIGINAL_SHA, original_package_bytes=27013606,
        source_spec_sha256=sha256_file(source_spec), source_spec=spec, source_model_files=model_files,
        authorization_sha256=sha256_file(target / 'authorization.json'), driver_commit=driver_commit,
        original_scientific_commit=spec['original_scientific_commit'],
        original_publication_driver_commit='31391b261750fd901d953d46f7769a597ad3d7e9',
        original_exact_driver_commit=spec['driver_commit'], supersedes_review_sha256=OLD_REVIEW_SHA,
        weights_changed=False, gine_changed=False, candidate_pool_changed=False, split_changed=False,
        test_artifacts_previously_evaluated=True, repair_selected_using_test=False,
        fitter_config=FITTER_CONFIG, historical_fitter_source_sha256=FITTER_SHA,
        required_counts=dict(validation=187, calibration_parent_units=288, test_parent_units=614,
            common_calibration=41, common_test=96, test_classifier_examples=238),
        main_matrix_write=False, created_at=utc_now())
    result = sealed(target / 'repair_contract.json', contract)
    sealed(target / 'metric_dependency_audit.json', dict(
        classifier_probability='softmax(raw_logits / T) exactly once',
        source_cohort='true source and argmax source; positive scalar invariant checked',
        strict_flip='argmax source to other; checked per match',
        raw_wnode='independent of classifier/temperature; original graph+match+encoder+schema+solver bindings required',
        best_match='(raw WNode, negative calibrated CFDrop, atom mapping); replay required',
        selector='same global calibration objective and tie-break; replay required',
        classifier_metrics='NLL/ECE/Brier and full classification report regenerated from raw logits',
        no_ot_implementation_import=True, test_used_to_choose_repair=False,
        source_manifest_sha256=sha256_file(target / 'original_package_manifest.json')))
    return result


def context(output_root):
    output = Path(output_root).resolve(strict=True)
    contract = reopen(output / 'repair_contract.json')
    require(sha256_file(output / 'authorization.json') == contract['authorization_sha256'], 'AUTHORIZATION_DRIFT')
    spec = contract['source_spec']
    root, manifest = load_bundle(spec['bundle_root'])
    require(sha256_file(root / 'bundle_manifest.json') == spec['bundle_file_sha256'], 'INPUT_BUNDLE_DRIFT')
    for name, files in contract['source_model_files'].items():
        for leaf, digest in files.items():
            require(sha256_file(Path(spec['models'][name]) / leaf) == digest, 'SEALED_MODEL_CHANGED')
    return output, contract, spec, root, manifest


def validation_inputs(model_root, split_path, expected_weight):
    """Bind actual saved raw logits to exact validation order and selected bundle."""
    parents = core._all_parents(split_path)
    rows = csv_rows(Path(model_root) / 'validation_predictions.csv')
    require(len(parents) == len(rows) == 187, 'VALIDATION_MUST_HAVE_187_ROWS')
    ids = [r['molecule_id'] for r in rows]
    require(ids == [p.parent_id for p in parents] and len(set(ids)) == 187, 'VALIDATION_IDS_ORDER_OR_DUPLICATE')
    require(all(r['split'] == 'val' and int(r['label']) == p.label and r['smiles'] == p.smiles
                for r,p in zip(rows, parents, strict=True)), 'VALIDATION_LABEL_SMILES_SPLIT_BINDING')
    card = read_json(Path(model_root) / 'model_card.json')
    require(card['checkpoint_id'] == expected_weight == sha256_file(Path(model_root) / 'model.pt'), 'VALIDATION_SELECTED_WEIGHT_BINDING')
    logits = np.stack([vector(r['logits']) for r in rows])
    probs = np.stack([vector(r['probabilities']) for r in rows])
    # The historical validation writer uses float32 torch softmax, unlike the
    # oracle's float64 inference records. Reproduce that representation exactly.
    import torch
    historical_probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    require(np.array_equal(historical_probs.astype(np.float64), probs), 'VALIDATION_NOT_RAW_LOGITS')
    require(np.array_equal(logits.argmax(axis=1), [int(r['predicted_label']) for r in rows]), 'VALIDATION_ARGMAX_DRIFT')
    labels = np.asarray([p.label for p in parents], dtype=np.int64)
    mapping = read_json(Path(model_root) / 'label_map.json')
    require(mapping == {'0': 'Inactive', '1': 'Active'}, 'VALIDATION_CLASS_MAPPING_CHANGED')
    attempt = Path(model_root).parent
    terminal = read_json(attempt / 'training_terminal.json')
    latest = read_json(attempt / 'training_state' / 'latest_checkpoint.json')
    checkpoint_path = attempt / 'training_state' / latest['checkpoint_file']
    require(Path(latest['checkpoint_file']).name == latest['checkpoint_file']
        and latest['status'] == 'CHECKPOINT_COMPLETE' and terminal['status'] == 'TRAINING_PASS'
        and latest['checkpoint_sha256'] == terminal['final_checkpoint_sha256'] == sha256_file(checkpoint_path)
        and checkpoint_path.stat().st_size == latest['checkpoint_bytes']
        and terminal['model_sha256'] == expected_weight, 'BEST_WEIGHT_TERMINAL_BINDING')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    weights = torch.load(Path(model_root)/'model.pt', map_location='cpu', weights_only=True)['state_dict']
    require(checkpoint['best_epoch'] == terminal['best_epoch']
        and checkpoint['contract_sha256'] == terminal['training_contract_sha256'] == latest['contract_sha256']
        and set(weights) == set(checkpoint['best_state'])
        and all(torch.equal(v, checkpoint['best_state'][k]) for k,v in weights.items()), 'NOT_SELECTED_BEST_WEIGHTS')
    binding = dict(state='PASS', split='validation', saved_split_alias='val', count=187, sample_ids=ids,
        sample_id_manifest_sha=stable_sha256(ids), raw_logits_sha=stable_sha256(logits.tolist()),
        labels_sha=stable_sha256(labels.tolist()), model_weight_sha=expected_weight,
        validation_csv_sha256=sha256_file(split_path),
        validation_predictions_sha256=sha256_file(Path(model_root) / 'validation_predictions.csv'),
        class_mapping_sha=sha256_file(Path(model_root) / 'label_map.json'),
        selected_epoch=terminal['best_epoch'], terminal_sha256=sha256_file(attempt/'training_terminal.json'),
        selected_best_state_equal=True, terminal_checkpoint_sha256=latest['checkpoint_sha256'],
        raw_logit_inference_only_count=0,
        saved_prediction_provenance='trainer loads best_state then final_validation, same sealed bundle inventory',
        test_used=False, calibration_used=False)
    return logits, labels, binding


def fit(output_root):
    import torch
    from src.oracles.gnn_oracle import fit_temperature_scaling, classification_metrics, verify_checkpoint_bundle
    output, contract, spec, root, manifest = context(output_root)
    require(os.environ.get('SLURM_JOB_ID'), 'TEMPERATURE_FIT_REQUIRES_HPC_COMPUTE_JOB')
    torch.set_num_threads(2)
    require(hashlib.sha256(inspect.getsource(fit_temperature_scaling).rstrip('\n').encode()).hexdigest() == FITTER_SHA,
            'HISTORICAL_FITTER_SOURCE_DIFFERS')
    if (output/'fit_complete.json').exists():
        done = reopen(output/'fit_complete.json')
        for n in ALTERNATIVES:
            require(sha256_file(output/'classifiers'/n/'temperature_fit_receipt.json') == done['fit_receipts'][n], 'FIT_RECEIPT_DRIFT')
            verify_checkpoint_bundle(output/'classifiers'/n)
        return done
    validation_path = bundle_file(root, manifest, manifest['splits']['validation'])
    all_inputs = {n: validation_inputs(spec['models'][n], validation_path, spec['model_files'][n]['model.pt']) for n in ALTERNATIVES}
    sealed(output / 'validation_input_binding.json', {n: b for n, (_,_,b) in all_inputs.items()})
    table = []
    for name, (logits, labels, binding) in all_inputs.items():
        directory = output / 'classifiers' / name
        if (directory / 'temperature_fit_receipt.json').exists():
            receipt = reopen(directory / 'temperature_fit_receipt.json')
            require(receipt['raw_logits_sha'] == binding['raw_logits_sha'], 'FIT_RESUME_INPUT_DRIFT')
            verify_checkpoint_bundle(directory)
            table.append(dict(backbone=name, temperature=receipt['fitted_temperature'], num_examples=187,
                nll_before=receipt['objective_before'], nll_after=receipt['objective_after']))
            continue
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            fitted = fit_temperature_scaling(logits, labels, max_iter=100)
        require(fitted['status'] == 'fit' and fitted['argmax_invariant'], 'ACTUAL_FIT_FAILED')
        temperature = dict(fitted, validation_csv_sha256=binding['validation_csv_sha256'],
            validation_predictions_sha256=binding['validation_predictions_sha256'])
        directory.mkdir(parents=True, exist_ok=False)
        source = Path(spec['models'][name])
        for p in source.iterdir():
            if p.is_file() and p.name not in {'temperature_scaling.json','model_card.json','sha256sums.txt'}:
                shutil.copyfile(p, directory / p.name)
        atomic_json(directory / 'temperature_scaling.json', temperature)
        atomic_json(directory / 'temperature.json', temperature)
        card = read_json(source / 'model_card.json')
        card['temperature_calibration_fit_on_validation'] = True
        card['temperature_correction'] = dict(original_model_card_sha256=sha256_file(source / 'model_card.json'),
            repair_contract_sha256=sha256_file(output / 'repair_contract.json'), weights_changed=False,
            first_validation_fit=True, original_training_config_unchanged=True)
        atomic_json(directory / 'model_card.json', card)
        receipt = sealed(directory / 'temperature_fit_receipt.json', dict(binding,
            status='fit', method='temperature_scaling', fit_split='validation', num_examples=187,
            fitter_commit=contract['driver_commit'], fitter_config=FITTER_CONFIG,
            initial_temperature=1.0, fitted_temperature=fitted['temperature'],
            objective_before=fitted['nll_before'], objective_after=fitted['nll_after'],
            optimizer_termination='historical_LBFGS_step_returned_normally; internal_stop_reason_not_exported',
            numerical_warnings=[str(w.message) for w in observed], optimizer_parameter_count=1,
            model_optimizer_parameters=0, created_at=utc_now()))
        before = classification_metrics(labels, scaled_probabilities(logits, 1.0), num_classes=2)
        after = classification_metrics(labels, scaled_probabilities(logits, fitted['temperature']), num_classes=2)
        atomic_json(directory / 'validation_metrics_before_after.json', dict(before=before, after=after,
            nll_before=fitted['nll_before'], nll_after=fitted['nll_after']))
        atomic_json(directory / 'classifier_overlay_manifest.json', dict(original_root=str(source),
            source_files=contract['source_model_files'][name], weight_sha256=sha256_file(directory / 'model.pt'),
            weights_changed=False, effective_config_delta={'calibration.fit_on_validation': {'old': 'MISSING', 'new': True}},
            temperature_sha256=sha256_file(directory / 'temperature_scaling.json'),
            fit_receipt_sha256=sha256_file(directory / 'temperature_fit_receipt.json')))
        sums = ''.join(f'{sha256_file(p)}  {p.name}\n' for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'sha256sums.txt')
        from src.eval.bace_frozen_gnn_contracts import _atomic_text
        _atomic_text(directory / 'sha256sums.txt', sums)
        verify_checkpoint_bundle(directory)
        table.append(dict(backbone=name, temperature=receipt['fitted_temperature'], num_examples=187,
            nll_before=receipt['objective_before'], nll_after=receipt['objective_after']))
    atomic_csv(output / 'gnn_seed7_temperature_fit_table.csv', table)
    models = {n: str(output / 'classifiers' / n) if n in ALTERNATIVES else spec['models'][n] for n in core.BACKBONES}
    sealed(output / 'fit_complete.json', dict(state='PASS', models=models,
        fit_receipts={n: sha256_file(output / 'classifiers' / n / 'temperature_fit_receipt.json') for n in ALTERNATIVES},
        gine_temperature_sha256=contract['source_model_files']['gine']['temperature_scaling.json'],
        original_weights_unchanged=True, raw_logit_inference_only_count=0, completed_at=utc_now()))
    return read_json(output / 'fit_complete.json')


class BoundRawDistance:
    """Exact lookup, never a solver or a loosened cache-key lookup."""
    def __init__(self, old_matches, *, old_checkpoint, new_checkpoint, parent_checkpoint_sha, distance_contract_sha):
        require(old_checkpoint == new_checkpoint, 'RAW_OT_REUSE_REQUIRES_UNCHANGED_MODEL_WEIGHTS')
        self.matches = {(r['candidate_id'], r['match_index']): r for r in old_matches}
        require(len(self.matches) == len(old_matches), 'DUPLICATE_SOURCE_MATCH')
        self.checkpoint = old_checkpoint
        self.parent_sha = parent_checkpoint_sha
        self.contract_sha = distance_contract_sha
        self.used = []

    def distance_for_action(self, parent, residual, *, action_context):
        key = (action_context['candidate_id'], action_context['match_index'])
        row = self.matches.get(key)
        require(row is not None and row['parent_smiles'] == parent and row['residual_smiles'] == residual
                and row['parent_id'] == action_context['parent_id']
                and row['match_atom_indices'] == action_context['match_atom_indices']
                and row['oracle_checkpoint_hash'] == self.checkpoint == action_context['oracle_checkpoint_id']
                and row['action_semantics_version'] == action_context['action_semantics_version']
                and row['distance_ok'] is True and row['cf_flip'] is True,
                'CACHE_PROVENANCE_GAP_OR_FLIP_CHANGED')
        value = row['wnode_distance']
        require(isinstance(value, (int,float)) and math.isfinite(value) and value >= 0, 'CACHE_NONFINITE_DISTANCE')
        self.used.append(dict(parent_checkpoint_sha256=self.parent_sha, distance_contract_sha256=self.contract_sha,
            parent_smiles=parent, residual_smiles=residual, action_context=action_context,
            original_match_sha256=stable_sha256(row), raw_wnode=value))
        return {'ok': True, 'distance': value, 'cache_hit': True}


def corrected_predictions(source, parents, temperature, checkpoint):
    rows = csv_rows(source)
    require([r['parent_id'] for r in rows] == [p.parent_id for p in parents], 'PARENT_RAW_LOGIT_ORDER_DRIFT')
    raw = np.stack([vector(r['logits']) for r in rows])
    old_temperatures = {float(r['temperature']) for r in rows}
    require(len(old_temperatures) == 1 and all(r['checkpoint_id'] == checkpoint for r in rows), 'PARENT_LOGIT_CHECKPOINT_DRIFT')
    old_probs = np.stack([vector(r['probabilities']) for r in rows])
    require(np.allclose(scaled_probabilities(raw, next(iter(old_temperatures))), old_probs, rtol=1e-9, atol=1e-12),
            'PARENT_RAW_LOGIT_PROBABILITY_BINDING')
    new_probs = scaled_probabilities(raw, temperature)
    result = []
    for p,r,l,pr in zip(parents, rows, raw, new_probs, strict=True):
        require(int(r['label']) == p.label and int(r['predicted_label']) == int(pr.argmax()), 'SOURCE_ARGMAX_CHANGED')
        result.append(dict(predicted_label=int(pr.argmax()), probabilities=pr.tolist(), logits=l.tolist(),
            source_probability=float(pr[1]), confidence=float(pr.max()), checkpoint_id=checkpoint,
            backbone=r['backbone'], num_classes=2, temperature=temperature, source_label=1))
    return result


def evaluation_context(output_root):
    import torch
    from src.oracles.gnn_oracle import GNNOracle
    output, contract, spec, root, manifest = context(output_root)
    fit_state = reopen(output / 'fit_complete.json')
    torch.set_num_threads(8)
    oracles = {n: GNNOracle.from_checkpoint(p, device='cpu', batch_size=256) for n,p in fit_state['models'].items()}
    for oracle in oracles.values():
        for parameter in oracle.model.parameters():
            parameter.requires_grad_(False)
    candidates = core._candidates(root, manifest)
    selector = core.frozen_selector(root, manifest)
    payload = dict(schema=core.SCHEMA, execution_commit=manifest['execution_commit'],
        bundle_sha256=sha256_file(root / 'bundle_manifest.json'), oracle_batch_size=256, cpu_threads=8,
        checkpoints={n:o.checkpoint_id for n,o in oracles.items()}, temperatures={n:o.temperature for n,o in oracles.items()},
        selector_sha256=selector['input_sha256'], candidate_universe_sha256=stable_sha256(candidates))
    binding = stable_sha256(payload)
    evaluation = output / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    run = dict(payload, binding_sha256=binding, model_roots=fit_state['models'], seed=7,
        evaluation_driver_commit=contract['driver_commit'], correction_contract_sha256=sha256_file(output/'repair_contract.json'),
        test_artifacts_previously_evaluated=True, repair_selected_using_test=False, main_matrix_write=False)
    if (evaluation / 'run_manifest.json').exists():
        require(read_json(evaluation / 'run_manifest.json') == run, 'CORRECTED_RUN_BINDING_DRIFT')
    else:
        atomic_json(evaluation / 'run_manifest.json', run)
    return output, contract, spec, root, manifest, evaluation, oracles, candidates, selector, binding


def reconcile(output_root, *, split):
    from src.eval.bace_frozen_gnn_verification import _evaluate_rows
    require(os.environ.get('SLURM_JOB_ID'), 'RECONCILIATION_REQUIRES_COMPUTE_JOB')
    output, contract, spec, root, manifest, evaluation, oracles, candidates, selector, binding = evaluation_context(output_root)
    require(split in {'calibration', 'test'}, 'BAD_RECONCILIATION_SPLIT')
    if (output/f'{split}_reconciliation.json').exists():
        done = reopen(output/f'{split}_reconciliation.json')
        for rel,digest in done['receipt_files'].items():
            require(sha256_file(output/rel)==digest, 'RECONCILIATION_RECEIPT_DRIFT')
        return done
    if split == 'test':
        freeze = reopen(output / 'selector_replay_receipt.json')
        require(freeze['all_ten_frozen'] and sha256_file(evaluation / 'CALIBRATION_FREEZE.json') == freeze['freeze_sha256'],
                'TEST_BEFORE_NEW_GLOBAL_CALIBRATION_FREEZE')
    # No test parent or probability file is parsed above this point.
    parents = core._all_parents(bundle_file(root, manifest, manifest['splits'][split]))
    source = Path(spec['evaluation_root'])
    old_run = read_json(source / 'run_manifest.json')
    new_predictions = {n: corrected_predictions(source / n / f'{split}_classifier_predictions.csv', parents,
        oracles[n].temperature, oracles[n].checkpoint_id) for n in core.BACKBONES}
    cohorts = core.cohort_ids(parents, new_predictions)
    require(cohorts == read_json(source / f'{split}_cohorts.json'), 'SOURCE_COHORT_CHANGED')
    atomic_json(evaluation / f'{split}_cohorts.json', cohorts)
    encoder_binding = dict(wnode=manifest['wnode_config'], feature_schema=manifest['files'][manifest['feature_schema_path']],
        molclr_checkpoint=manifest['files'][manifest['molclr_checkpoint_path']],
        molclr_source={r:i for r,i in manifest['files'].items() if r.startswith(manifest['molclr_source_root'] + '/')})
    distance_sha = stable_sha256(encoder_binding)
    atomic_json(output / 'raw_distance_contract.json', encoder_binding)
    old_inventory = read_json(output / 'original_package_manifest.json')['files']
    expected_units = 288 if split == 'calibration' else 614
    parent_units, inference_count, reused_count = 0, 0, 0
    evidence = {}
    for name in core.BACKBONES:
        oracle = oracles[name]
        atomic_csv(evaluation / name / f'{split}_classifier_predictions.csv', [
            dict(parent_id=p.parent_id,label=p.label,**r) for p,r in zip(parents,new_predictions[name],strict=True)])
        chosen = candidates
        if split == 'test':
            old_freeze = read_json(source / 'CALIBRATION_FREEZE.json')
            new_freeze = read_json(evaluation / 'CALIBRATION_FREEZE.json')
            old_ids = set(old_freeze['selections'][name]['native']) | set(old_freeze['selections'][name]['common'])
            new_ids = set(new_freeze['selections'][name]['native']) | set(new_freeze['selections'][name]['common'])
            require(new_ids <= old_ids, 'CACHE_PROVENANCE_GAP_NEW_TEST_RULE_NOT_PREVIOUSLY_EVALUATED')
            # Exact old per-parent checkpoint identity binds the old selected union.
            chosen = [r for r in candidates if r['candidate_id'] in old_ids]
        before = dict(zip([p.parent_id for p in parents],new_predictions[name],strict=True))
        for parent in parents:
            if parent.parent_id not in cohorts['native'][name]:
                continue
            old_key = stable_sha256(dict(parent=asdict(parent), candidates=chosen, split=split,
                binding=old_run['binding_sha256'], checkpoint=oracle.checkpoint_id))
            rel = f'{name}/{split}/parents/{old_key}.json'
            original = source / rel
            expected_sha = old_inventory[f'evaluation/{rel}']['sha256']
            require(sha256_file(original) == expected_sha, 'ORIGINAL_PARENT_CHECKPOINT_DRIFT')
            old = read_json(original)
            require(old['binding'] == old_key and old['science_sha256'] == stable_sha256(old['science']), 'OLD_SCIENTIFIC_STATE_BINDING')
            use_candidates = chosen
            if split == 'test':
                use_candidates = [r for r in chosen if r['candidate_id'] in new_ids]
            new_key = stable_sha256(dict(parent=asdict(parent), candidates=use_candidates, split=split,
                binding=binding, checkpoint=oracle.checkpoint_id))
            target = evaluation / name / split / 'parents' / f'{new_key}.json'
            receipt_path = output / 'reuse' / split / name / f'{new_key}.json'
            if target.exists() and receipt_path.exists():
                receipt = reopen(receipt_path)
                require(receipt['original_parent_sha256'] == expected_sha and receipt['new_parent_sha256'] == sha256_file(target), 'PARENT_RESUME_DRIFT')
            else:
                old_science = old['science']
                pairs_by_id = {r['candidate_id']:r for r in old_science['pair_rows']}
                for pair in old_science['pair_rows']:
                    matches = [r for r in old_science['match_rows'] if r['candidate_id'] == pair['candidate_id']]
                    audit.verify_matches(pair, matches, checkpoint=oracle.checkpoint_id,
                        before={'predicted_label':pair['pred_before'], 'probabilities':
                            next((m['p_before'] for m in matches), [1-pair['p1_before'],pair['p1_before']])})
                selected_ids = {r['candidate_id'] for r in use_candidates}
                old_matches = [r for r in old_science['match_rows'] if r['candidate_id'] in selected_ids]
                distance = BoundRawDistance(old_matches, old_checkpoint=old_run['checkpoints'][name],
                    new_checkpoint=oracle.checkpoint_id, parent_checkpoint_sha=expected_sha, distance_contract_sha=distance_sha)
                captured = []
                if name == 'gine':
                    current = [r for r in old_science['pair_rows'] if r['candidate_id'] in selected_ids]
                    matches = old_matches
                else:
                    class Recorder:
                        def predict_records(self, graphs, **kwargs):
                            records = oracle.predict_records(graphs, **kwargs)
                            captured.extend(records)
                            return records
                    cache = {parent.parent_id:dict(parent_smiles=parent.smiles,
                        pred_before=before[parent.parent_id]['predicted_label'], p_before=before[parent.parent_id]['probabilities'])}
                    current, matches = _evaluate_rows([parent], use_candidates, oracle=Recorder(),
                        featurizer=core._featurizer(root,manifest), distance_provider=distance,
                        oracle_batch_size=256, split=split, oracle_checkpoint_id=oracle.checkpoint_id,
                        parent_prediction_cache=cache)
                    require(len(matches) == len(old_matches), 'MATCH_ENUMERATION_COUNT_CHANGED')
                    for old_match,new_match in zip(old_matches,matches,strict=True):
                        for field in ('parent_id','candidate_id','match_index','match_atom_indices','residual_smiles',
                                      'delete_valid','pred_before','pred_after','cf_flip','teacher_strict_flip','wnode_distance'):
                            require(old_match[field] == new_match[field], f'MATCH_OR_ARGMAX_OR_RAW_DISTANCE_CHANGED:{field}')
                    valid_matches = [m for m in matches if m['delete_valid'] and m.get('residual_smiles')]
                    require(len(valid_matches) == len(captured), 'RESIDUAL_LOGIT_RECORD_COUNT')
                    for old_match, record in zip([m for m in old_matches if m['delete_valid'] and m.get('residual_smiles')],captured,strict=True):
                        require(np.allclose(scaled_probabilities([record['logits']],1.0)[0], old_match['p_after'],rtol=1e-6,atol=1e-8),
                                'FROZEN_RESIDUAL_LOGIT_REPLAY_DIFFERS_FROM_OLD_MODEL')
                scientific = dict(pair_rows=current, match_rows=matches)
                atomic_json(target, dict(binding=new_key,science=scientific,science_sha256=stable_sha256(scientific)))
                receipt = sealed(receipt_path, dict(original_parent_sha256=expected_sha,
                    original_parent_relpath=rel, new_parent_sha256=sha256_file(target),
                    parent_id=parent.parent_id, backbone=name, split=split, raw_prediction_records=captured,
                    inference_only_count=len(captured), raw_distance_reuse=distance.used,
                    unchanged_gine_reuse_count=sum(bool(m['distance_ok']) for m in matches) if name=='gine' else 0,
                    source_cohort_changes=0, argmax_changes=0, strict_flip_changes=0,
                    raw_ot_recomputed_count=0, cache_provenance_gaps=[]))
            evidence[str(receipt_path.relative_to(output))] = sha256_file(receipt_path)
            inference_count += receipt['inference_only_count']
            reused_count += len(receipt['raw_distance_reuse']) + receipt['unchanged_gine_reuse_count']
            parent_units += 1
            atomic_json(output / f'{split}_progress.json', dict(parent_units=parent_units,
                expected=expected_units, backbone=name, parent_id=parent.parent_id, updated_at=utc_now()))
    require(parent_units == expected_units, 'CORRECTION_PARENT_UNIT_COVERAGE')
    return sealed(output / f'{split}_reconciliation.json', dict(state='PASS', split=split,
        completed_parent_units=parent_units, receipt_files=evidence, raw_logit_inference_only_count=inference_count,
        raw_ot_reused_count=reused_count, raw_ot_recomputed_count=0, cache_provenance_gaps=[],
        source_cohort_changes=0, argmax_changes=0, strict_flip_changes=0,
        completed_at=utc_now(), new_freeze_sha256=sha256_file(evaluation/'CALIBRATION_FREEZE.json') if split=='test' else None))


def replay_phase(output_root, *, phase):
    from src.oracles.gnn_oracle import classification_metrics
    output, contract, spec, root, manifest, evaluation, oracles, candidates, selector, binding = evaluation_context(output_root)
    if phase == 'calibration' and (output/'selector_replay_receipt.json').exists():
        done = reopen(output/'selector_replay_receipt.json')
        require(sha256_file(evaluation/'CALIBRATION_FREEZE.json') == done['freeze_sha256'], 'FROZEN_SELECTOR_DRIFT')
        return done
    split = 'calibration' if phase == 'calibration' else 'test'
    reopen(output / f'{split}_reconciliation.json')
    original_predict = core._predict
    def saved_prediction(parents, oracle, featurizer, actual_split, batch_size):
        if actual_split == 'test':
            require((output/'selector_replay_receipt.json').is_file(), 'TEST_BEFORE_FROZEN_SELECTOR_RECEIPT')
        rows = csv_rows(evaluation / oracle.backbone / f'{actual_split}_classifier_predictions.csv')
        require([r['parent_id'] for r in rows] == [p.parent_id for p in parents], 'SAVED_PREDICTION_ORDER')
        return [dict(predicted_label=int(r['predicted_label']),probabilities=vector(r['probabilities']).tolist(),
            logits=vector(r['logits']).tolist(),source_probability=float(r['source_probability']),
            confidence=float(r['confidence']),checkpoint_id=r['checkpoint_id'],backbone=r['backbone'],
            num_classes=2,temperature=float(r['temperature']),source_label=1) for r in rows]
    # Scoped replacement prevents unnecessary inference; original scientific kernels,
    # cached-pair hashes, global selector and metrics are unchanged in this process.
    core._predict = saved_prediction
    try:
        result = core._run_phases(root, manifest, evaluation, candidates, selector, oracles,
            {n:Path(p) for n,p in reopen(output/'fit_complete.json')['models'].items()},
            core._featurizer(root,manifest), None, binding,256,classification_metrics,
            phase=phase, require_cached_pairs=True)
    finally:
        core._predict = original_predict
    if phase == 'calibration':
        changes = []
        source = Path(spec['evaluation_root'])
        for name in core.BACKBONES:
            for mode in ('native','common'):
                old = read_json(source/name/mode/'selected_rules.json')['candidate_ids']
                new = read_json(evaluation/name/mode/'selected_rules.json')['candidate_ids']
                changes.append(dict(backbone=name,cohort=mode,old_order_sha256=stable_sha256(old),
                    new_order_sha256=stable_sha256(new),order_changed=old!=new))
        return sealed(output/'selector_replay_receipt.json', dict(state='PASS',all_ten_frozen=True,
            freeze_sha256=sha256_file(evaluation/'CALIBRATION_FREEZE.json'),
            new_test_probabilities_read=False, test_previously_evaluated=True,
            order_changes=changes, completed_at=utc_now()))
    return result


def _verify_fit_values(receipt, temperature, rows, model_sha, validation_sha):
    import torch
    raw = np.stack([vector(r['logits']) for r in rows])
    labels = np.asarray([int(r['label']) for r in rows], dtype=np.int64)
    require(len(rows) == 187 and len({r['molecule_id'] for r in rows}) == 187
        and {r['split'] for r in rows} == {'val'}, 'CORRECTIVE_VALIDATION_SCOPE')
    require(receipt['status'] == temperature['status'] == 'fit'
        and receipt['fit_split'] == 'validation' and receipt['num_examples'] == 187
        and receipt['model_weight_sha'] == model_sha
        and receipt['raw_logits_sha'] == stable_sha256(raw.tolist())
        and receipt['labels_sha'] == stable_sha256(labels.tolist())
        and receipt['sample_id_manifest_sha'] == stable_sha256([r['molecule_id'] for r in rows])
        and receipt['validation_csv_sha256'] == validation_sha
        and receipt['fitter_config'] == FITTER_CONFIG
        and receipt['optimizer_parameter_count'] == 1 and receipt['model_optimizer_parameters'] == 0,
        'CORRECTIVE_FIT_INPUT_OR_OPTIMIZATION_PROOF')
    t = float(temperature['temperature'])
    require(t == receipt['fitted_temperature'], 'FIT_RECEIPT_TEMPERATURE_MISMATCH')
    scaled_probabilities(raw,t)
    tensor = torch.tensor(raw,dtype=torch.float64)
    truth = torch.tensor(labels,dtype=torch.long)
    for field,denom in (('objective_before',1.0),('objective_after',t)):
        objective = float(torch.nn.functional.cross_entropy(tensor/denom,truth).item())
        require(math.isclose(objective,receipt[field],abs_tol=1e-12,rel_tol=1e-9), 'FIT_OBJECTIVE_REPLAY_FAILED')


def verify_package(output_root):
    """Independent science replay, fresh corrected tables, then immutable package."""
    output, contract, spec, root, manifest = context(output_root)
    require(os.environ.get('SLURM_JOB_ID'), 'INDEPENDENT_CORRECTIVE_AUDIT_REQUIRES_COMPUTE_JOB')
    require(sha256_file(contract['original_package']) == ORIGINAL_SHA, 'ORIGINAL_PACKAGE_NOT_PRESERVED')
    verified = output / 'verified'
    if (verified/'result_package.json').exists():
        previous = read_json(verified/'result_package.json')
        return verify_corrective_package(previous['archive'])
    evaluation = output/'evaluation'
    science = audit.verify_science(bundle_root=root,evaluation_root=evaluation)
    models = reopen(output/'fit_complete.json')['models']
    for n in ALTERNATIVES:
        model = Path(models[n])
        fit_receipt = reopen(model/'temperature_fit_receipt.json')
        _verify_fit_values(fit_receipt,read_json(model/'temperature_scaling.json'),
            csv_rows(model/'validation_predictions.csv'), spec['model_files'][n]['model.pt'],
            manifest['files'][manifest['splits']['validation']]['sha256'])
        require(sha256_file(model/'model.pt') == spec['model_files'][n]['model.pt'], 'CORRECTED_WEIGHT_CHANGED')
    cal = reopen(output/'calibration_reconciliation.json')
    test = reopen(output/'test_reconciliation.json')
    freeze = reopen(output/'selector_replay_receipt.json')
    require(test['new_freeze_sha256'] == freeze['freeze_sha256'] == sha256_file(evaluation/'CALIBRATION_FREEZE.json'),
        'CORRECTED_TEST_FREEZE_MISMATCH')
    require(cal['completed_parent_units']==288 and test['completed_parent_units']==614,
        'CORRECTIVE_COMPLETE_COUNTS')
    require(len(read_json(evaluation/'calibration_cohorts.json')['common'])==41
        and len(read_json(evaluation/'test_cohorts.json')['common'])==96, 'COMMON_COHORT_COUNT_DRIFT')
    verified.mkdir(exist_ok=False)
    atomic_json(verified/'independent_science_replay.json',science)
    diffs=[]
    source = Path(spec['evaluation_root'])
    for n in core.BACKBONES:
        before,after = read_json(source/n/'classifier_metrics.json'),read_json(evaluation/n/'classifier_metrics.json')
        for key in ('NLL','ece','brier_score','roc_auc','balanced_accuracy','macro_f1'):
            diffs.append(dict(backbone=n,metric=key,before=before.get(key),after=after.get(key),
                reason='validation_only_temperature_scaling; no test-selected remedy'))
    atomic_csv(verified/'gnn_seed7_correction_diff.csv',diffs)
    shutil.copyfile(evaluation/'gnn_seed7_classifier_table.csv',verified/'gnn_seed7_classifier_table.csv')
    shutil.copyfile(output/'gnn_seed7_temperature_fit_table.csv',verified/'gnn_seed7_temperature_fit_table.csv')
    rows=[]
    for mode in ('native','common'):
        current=[]
        for n in core.BACKBONES:
            values=read_json(evaluation/n/mode/'explanation_metrics.json')
            current.append(dict(backbone=n,cohort=mode,seed=7,
                **{k:v for k,v in values.items() if not isinstance(v,(dict,list))}))
        atomic_csv(verified/f'gnn_seed7_explanation_{mode}.csv',current)
        rows.extend(current)
    core._latex(verified/'gnn_seed7_table.tex',rows,
        ('backbone','cohort','seed','cohort_size','CCRCov@10','CCRCov@20','conditional_median_WNode'))
    result=dict(state='GNN_CORE_SEED7_CORRECTED_PASS', original_package_sha256=ORIGINAL_SHA,
        original_package_preserved=True, original_integrity_state='PASS', original_scientific_state='BLOCKED_TEMPERATURE_CONTRACT',
        supersedes_review_sha256=OLD_REVIEW_SHA,repair_contract_sha256=sha256_file(output/'repair_contract.json'),
        all_weights_unchanged=True,gine_unchanged=True,candidate_pool_unchanged=True,
        validation_counts={n:187 for n in ALTERNATIVES},counts=dict(calibration=288,test=614),
        native_common_metrics_replayed=True, raw_ot_recomputed_count=0,
        raw_ot_reused_count=cal['raw_ot_reused_count']+test['raw_ot_reused_count'],
        raw_logit_inference_only_count=cal['raw_logit_inference_only_count']+test['raw_logit_inference_only_count'],
        cache_provenance_gaps=[],argmax_changes=0,source_cohort_changes=0,strict_flip_changes=0,
        selectors_frozen_before_test=True, selector_order_changes=freeze['order_changes'],
        test_artifacts_previously_evaluated=True, repair_selected_using_test=False,
        scientific_engine_commit=contract['original_scientific_commit'],driver_commit=contract['driver_commit'],
        original_publication_driver_commit=contract['original_publication_driver_commit'],
        original_exact_driver_commit=contract['original_exact_driver_commit'],
        scope='PROPOSAL_FIXED_BACKBONE_SENSITIVITY',seed=7,
        secondary_seeds_required=False,main_matrix_write=False,
        independent_science_replay_sha256=sha256_file(verified/'independent_science_replay.json'),created_at=utc_now())
    sealed(verified/'gnn_seed7_corrective_audit.json',result)
    files={}
    def add(path,name):
        require(Path(path).is_file() and name not in files, 'PACKAGE_FILE_MISSING_OR_DUPLICATE')
        files[name]=Path(path)
    for leaf in ('repair_contract.json','authorization.json','original_package_manifest.json',
        'validation_input_binding.json','metric_dependency_audit.json','raw_distance_contract.json',
        'fit_complete.json','calibration_reconciliation.json','test_reconciliation.json','selector_replay_receipt.json'):
        add(output/leaf,'repair/'+leaf)
    for p in (output/'reuse').rglob('*.json'):
        add(p,'repair/'+str(p.relative_to(output)))
        receipt = reopen(p)
        rel=receipt['original_parent_relpath']
        if 'original/evaluation/'+rel not in files:
            add(source/rel,'original/evaluation/'+rel)
    for p in evaluation.rglob('*'):
        if p.is_file() and p.name!='writer.lock':
            add(p,'evaluation/'+str(p.relative_to(evaluation)))
    for n,model in models.items():
        for p in Path(model).iterdir():
            if p.is_file():
                add(p,f'classifiers/{n}/{p.name}')
    for p in verified.iterdir():
        if p.is_file():
            add(p,'publication/'+p.name)
    add(root/'bundle_manifest.json','inputs/bundle_manifest.json')
    add(bundle_file(root,manifest,manifest['splits']['validation']),'inputs/validation.csv')
    archive=verified/'bace_gnn_seed7_corrected.tar.gz'
    package_manifest=dict(schema='bace_gnn_corrective_package_v1', original_package_sha256=ORIGINAL_SHA,
        scientific_engine_commit=contract['original_scientific_commit'],driver_commit=contract['driver_commit'],
        files={n:dict(size=p.stat().st_size,sha256=sha256_file(p)) for n,p in files.items()},main_matrix_write=False)
    from src.ablations.contracts import canonical_json_sha256
    package_manifest['manifest_sha256']=canonical_json_sha256(package_manifest)
    buffer=json.dumps(package_manifest,indent=2,sort_keys=True).encode()+b'\n'
    partial=archive.with_suffix('.partial')
    with tarfile.open(partial,'w:gz') as tar:
        entry=tarfile.TarInfo('package_manifest.json');entry.size=len(buffer)
        tar.addfile(entry,io.BytesIO(buffer))
        for name,path in sorted(files.items()):
            tar.add(path,arcname=name,recursive=False)
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    os.rename(partial,archive)
    checked=verify_corrective_package(archive)
    atomic_json(verified/'result_package.json',dict(checked,archive=str(archive),
        path=str(archive),repair_driver_commit=contract['driver_commit'],
        bytes=archive.stat().st_size,sha256=sha256_file(archive),package_sha256=sha256_file(archive)))
    return read_json(verified/'result_package.json')


def verify_corrective_package(package, output_root=None):
    """Portable independent reopen of fits, raw logits and every reused match.

    Read only unless a fresh output_root is explicitly requested. The old
    provisional archive cannot satisfy the corrective schema or audit.
    """
    manifest=safe_archive_manifest(package)
    require(manifest.get('schema')=='bace_gnn_corrective_package_v1'
        and manifest.get('original_package_sha256')==ORIGINAL_SHA, 'CORRECTIVE_PACKAGE_REQUIRED_OLD_CORE_BLOCKED')
    with tarfile.open(package,'r:gz') as tar:
        def data(name):
            return json.load(tar.extractfile(name))
        def rows(name):
            return list(csv.DictReader(io.TextIOWrapper(tar.extractfile(name),encoding='utf-8')))
        proof=data('publication/gnn_seed7_corrective_audit.json')
        require(proof['self_sha256']==stable_sha256({k:v for k,v in proof.items() if k!='self_sha256'}), 'CORRECTIVE_AUDIT_SELF_HASH')
        require(proof['state']=='GNN_CORE_SEED7_CORRECTED_PASS' and proof['main_matrix_write'] is False,
                'CORRECTIVE_AUDIT_REQUIRED')
        contract=data('repair/repair_contract.json')
        require(contract['self_sha256']==stable_sha256({k:v for k,v in contract.items() if k!='self_sha256'})
            and proof['repair_contract_sha256']==manifest['files']['repair/repair_contract.json']['sha256']
            and contract['repair_selected_using_test'] is False and contract['test_artifacts_previously_evaluated'] is True,
            'CORRECTIVE_RESEARCH_SCOPE')
        original=data('repair/original_package_manifest.json')['files']
        inp=data('inputs/bundle_manifest.json')
        require(manifest['files']['inputs/validation.csv']['sha256']==inp['files'][inp['splits']['validation']]['sha256'],
                'PORTABLE_VALIDATION_INPUT_CHANGED')
        validation=rows('inputs/validation.csv')
        validation_ids=[r.get('parent_id') or r.get('molecule_id') for r in validation]
        for n in core.BACKBONES:
            weight=manifest['files'][f'classifiers/{n}/model.pt']['sha256']
            require(weight==original[f'classifiers/{n}/model.pt']['sha256']==contract['source_model_files'][n]['model.pt'],
                'CORRECTED_MODEL_WEIGHT_CHANGED')
            temp=data(f'classifiers/{n}/temperature_scaling.json')
            card=data(f'classifiers/{n}/model_card.json')
            predrows=rows(f'classifiers/{n}/validation_predictions.csv')
            require([r['molecule_id'] for r in predrows]==validation_ids,'PORTABLE_VALIDATION_IDS_DRIFT')
            audit.require_validation_fitted_temperature(temp,model_card=card,backbone=n,
                validation_sha256=manifest['files']['inputs/validation.csv']['sha256'],
                validation_predictions_sha256=manifest['files'][f'classifiers/{n}/validation_predictions.csv']['sha256'],validation_count=187)
            if n=='gine':
                require(manifest['files'][f'classifiers/{n}/temperature_scaling.json']['sha256']==
                    original[f'classifiers/{n}/temperature_scaling.json']['sha256'], 'GINE_TEMPERATURE_CHANGED')
            else:
                _verify_fit_values(data(f'classifiers/{n}/temperature_fit_receipt.json'),temp,predrows,weight,
                    manifest['files']['inputs/validation.csv']['sha256'])
        freeze=data('repair/selector_replay_receipt.json')
        require(freeze['all_ten_frozen'] is True and freeze['new_test_probabilities_read'] is False
            and freeze['freeze_sha256']==manifest['files']['evaluation/CALIBRATION_FREEZE.json']['sha256'],
            'PORTABLE_GLOBAL_FREEZE_BINDING')
        counts={'calibration':0,'test':0}; used=0; inferred=0
        for rel in sorted(manifest['files']):
            if not rel.startswith('repair/reuse/'):
                continue
            receipt=data(rel)
            require(receipt['self_sha256']==stable_sha256({k:v for k,v in receipt.items() if k!='self_sha256'}), 'REUSE_RECEIPT_SELF_HASH')
            source_rel='original/evaluation/'+receipt['original_parent_relpath']
            require(manifest['files'][source_rel]['sha256']==receipt['original_parent_sha256']==
                original['evaluation/'+receipt['original_parent_relpath']]['sha256'], 'RAW_OT_ORIGINAL_PARENT_BINDING')
            source=data(source_rel)
            require(source['science_sha256']==stable_sha256(source['science']), 'ORIGINAL_SCIENCE_HASH')
            name,split=receipt['backbone'],receipt['split']
            new_name=Path(rel).name
            current=data(f'evaluation/{name}/{split}/parents/{new_name}')
            require(current['science_sha256']==stable_sha256(current['science'])
                and manifest['files'][f'evaluation/{name}/{split}/parents/{new_name}']['sha256']==receipt['new_parent_sha256'],
                'CORRECTED_PARENT_BINDING')
            old_by_key={(m['candidate_id'],m['match_index']):m for m in source['science']['match_rows']}
            t=data(f'classifiers/{name}/temperature_scaling.json')['temperature']
            raw=iter(receipt['raw_prediction_records'])
            for match in current['science']['match_rows']:
                old=old_by_key[(match['candidate_id'],match['match_index'])]
                for key in ('parent_id','parent_smiles','residual_smiles','match_atom_indices','delete_valid',
                    'pred_before','pred_after','cf_flip','teacher_strict_flip','wnode_distance','distance_ok'):
                    require(match[key]==old[key], 'PORTABLE_ARGMAX_OR_RAW_OT_OR_GRAPH_CHANGED')
                if name!='gine' and match['delete_valid'] and match.get('residual_smiles'):
                    record=next(raw)
                    probs=scaled_probabilities([record['logits']],t)[0]
                    require(np.allclose(probs,match['p_after'],rtol=1e-12,atol=1e-12), 'PORTABLE_DOUBLE_SCALING_OR_SOFTMAX')
                    inferred+=1
                if match['distance_ok']:
                    used+=1
            require(next(raw,None) is None, 'EXTRA_RESIDUAL_LOGITS')
            counts[split]+=1
        require(counts=={'calibration':288,'test':614} and proof['counts']==counts,
            'PORTABLE_PARENT_UNITS_INCOMPLETE')
        require(proof['raw_ot_reused_count']==used and proof['raw_logit_inference_only_count']==inferred
            and proof['raw_ot_recomputed_count']==0 and proof['cache_provenance_gaps']==[], 'PORTABLE_REUSE_ACCOUNTING')
        require(proof['independent_science_replay_sha256']==manifest['files']['publication/independent_science_replay.json']['sha256'],
            'INDEPENDENT_CORRECTIVE_REPLAY_HASH')
        result=dict(proof,sha256=sha256_file(package),package_sha256=sha256_file(package),
            bytes=Path(package).stat().st_size,file_count=len(manifest['files'])+1,
            corrective_audit_sha256=manifest['files']['publication/gnn_seed7_corrective_audit.json']['sha256'])
    if output_root is not None:
        destination=Path(output_root)
        require(not destination.exists(), 'FRESH_CORRECTIVE_IMPORT_REQUIRED')
        require(not any(p.is_symlink() for p in (destination,*destination.parents)), 'IMPORT_SYMLINK_REFUSED')
        staging=destination.with_name(destination.name+'.partial')
        require(not staging.exists(), 'IMPORT_PARTIAL_EXISTS_REQUIRES_INSPECTION')
        staging.mkdir(parents=True)
        with tarfile.open(package,'r:gz') as tar:
            for member in tar:
                target=staging/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('xb') as sink:
                    shutil.copyfileobj(source,sink);sink.flush();os.fsync(sink.fileno())
        atomic_json(staging/'corrective_location_overlay.json',dict(result,main_matrix_write=False,source_archive=str(package)))
        os.rename(staging,destination)
    return result


def status(output_root):
    root=Path(output_root)
    result={'root':str(root),'state':'WAITING','main_matrix_write':False}
    for name in ('repair_contract','validation_input_binding','fit_complete','calibration_reconciliation',
                 'selector_replay_receipt','test_reconciliation'):
        path=root/f'{name}.json'
        result[name]=read_json(path) if path.exists() else None
    package=root/'verified/result_package.json'
    if package.exists():
        result['package']=read_json(package)
        result['state']=result['package']['state']
    return result
