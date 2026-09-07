"""Export a new ChemAligned native rule universe without rerunning mining."""
from pathlib import Path
from dataclasses import replace
import json
import torch

from src.baselines.bace_globalgce_chemaligned import SCHEMA, corrected_rule
from src.baselines.bace_globalgce_chemaligned_runner import setup, read_json
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, file_identity, stable_sha256, utc_now


def joint_catalog(templates, rules):
    """Stable semantic dedup; native indices remain genuine source lineage."""
    catalog = {}; rejected = []
    for i, template in enumerate(templates):
        try:
            rule = corrected_rule(template, rules['features_reconst'][i], rules['adj_reconst'][i], rules['edge_attrs_reconst'][i])
        except ValueError as exc:
            rejected.append({'native_rule_index': i, 'reason': str(exc)}); continue
        semantic = rule.to_payload(); semantic.pop('rule_id'); semantic.pop('native_rule_index')
        key = stable_sha256({'adapter': SCHEMA, 'rule': semantic})
        if key in catalog:
            catalog[key]['generator_native_indices'].append(i); continue
        cid = 'chemaligned-'+key[:24]
        catalog[key] = {'candidate_id': cid, 'rule': replace(rule, rule_id=cid).to_payload(),
            'molecular_adapter': SCHEMA, 'method_variant': 'GlobalGCE-ChemAligned',
            'action_kind': 'lhs_rhs_graph_transformation_rule',
            'action_semantics': 'native_lhs_to_rhs_attachment_aware_v1',
            'rf_oracle_used': False, 'generator_native_indices': [i],
            'candidate_source': 'frozen_official_LHS_and_joint_decoder_RHS',
            'chemical_validity_scope': 'complete_parent_application_checked_in_shared_evaluator',
            'standalone_RHS_chemical_validity_claimed': False}
    return list(catalog.values()), rejected


def export_pool(config_path, rematerialization_root, output_root, training_root=None):
    config = read_json(config_path); evidence = Path(rematerialization_root)
    initial = read_json(evidence/'terminal.json')
    if initial.get('state') != 'REMATERIALIZATION_COMPLETE':
        raise ValueError('complete train/validation rematerialization required')
    if training_root:
        trained = Path(training_root); result = read_json(trained/'terminal.json')
        feasibility = read_json(trained/'best_train_feasibility.json')
        if result.get('state') != 'REPAIR_TRAINING_COMPLETE' or feasibility.get('strict_flip_witness_found') is not True:
            raise ValueError('completed repair lacks real frozen-best train recourse')
        rule_path = trained/'best_rules.pt'; source_stage = str(trained)
    else:
        if initial['train']['counts'].get('strict_flip', 0) < 1:
            raise ValueError('original checkpoint has no train recourse; one repair required')
        rule_path = Path(read_json(config['training_summary'])['rules_checkpoint']); source_stage = str(evidence)
    summary, original, templates, train, val, bridge = setup(config, 'cpu')
    rules = torch.load(rule_path, map_location='cpu', weights_only=False)
    rows, rejected = joint_catalog(templates, rules)
    if not rows: raise ValueError('no well-formed joint native rule: no fake PASS pool')
    output = Path(output_root); output.mkdir(parents=True, exist_ok=False)
    atomic_jsonl(output/'candidate_universe.jsonl', rows)
    atomic_jsonl(output/'candidate_rejections.jsonl', rejected)
    old = read_json(config['source_manifest'])
    provenance_keys = ('oracle_backend','classifier_family','classifier_type','rf_oracle_used','source_label','num_classes',
        'cf_mode','oracle_checkpoint','oracle_checkpoint_hash','model_pt_sha256','temperature_scaling_sha256','feature_schema_sha256')
    manifest = {key: old[key] for key in provenance_keys if key in old}
    manifest.update(schema_version='bace_chemaligned_candidate_universe_v2', dataset='bace', method='GlobalGCE',
        method_id='globalgce', method_variant='GlobalGCE-ChemAligned', molecular_adapter=SCHEMA,
        stage='TRAIN_CANDIDATE_GENERATION', status='PASS', run_complete=True,
        action_kind='lhs_rhs_graph_transformation_rule', action_semantics='native_lhs_to_rhs_attachment_aware_v1',
        candidate_count=len(rows), generated_raw_rule_count=len(templates), rule_budget_semantics='AT_MOST_K',
        candidate_universe_hash=file_identity(output/'candidate_universe.jsonl')['sha256'],
        rule_checkpoint=file_identity(rule_path), source_completed_stage=source_stage,
        repair_contract_sha256=stable_sha256(config), joint_contract=config['joint_contract'],
        original_generation_manifest=config['source_manifest'], source_parent_ids=[r['id'] for r in train],
        mining_reused=True, generator_repair_finetune=bool(training_root), calibration_loaded=False, test_loaded=False,
        validation_used_for_checkpoint=bool(training_root), selector_fitted_on_calibration=False,
        benchmark_test_previously_seen=True, repair_selected_using_test=False, created_at=utc_now())
    atomic_json(output/'run_manifest.json',manifest)
    atomic_json(output/'summary.json',{'status':'PASS','candidate_count':len(rows),'raw_rule_count':len(templates),
        'semantic_duplicate_count':len(templates)-len(rejected)-len(rows),'rejected':len(rejected),'test_loaded':False})
    return manifest
