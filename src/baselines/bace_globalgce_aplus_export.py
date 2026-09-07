"""Freeze the validation-selected A+ native rule pool; no test or authority."""
from pathlib import Path
from dataclasses import replace
import torch
from src.baselines.bace_globalgce_aplus import SCHEMA, hard_state_tensors, joint_states
from src.baselines.bace_globalgce_aplus_inputs import setup, read_json
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, stable_sha256, sha256_file, utc_now

def export_pool(config_path, prepared_root, output_root, training_root):
    config=read_json(config_path); prepared=read_json(Path(prepared_root)/'terminal.json')
    trained=Path(training_root); terminal=read_json(trained/'terminal.json')
    if prepared.get('state')!='INPUTS_BOUND' or terminal.get('state')!='REPAIR_TRAINING_COMPLETE':
        raise ValueError('COMPLETED_INPUTS_AND_TRAINING_REQUIRED')
    if not read_json(trained/'best_train_feasibility.json').get('strict_flip_witness_found'):
        raise ValueError('NO_TRUE_TRAIN_RECOURSE_NOT_SCIENCE_ZERO')
    summary,original,templates,train,val,bridge=setup(config,'cpu')
    rules=torch.load(trained/'best_rules.pt',map_location='cpu',weights_only=False)
    catalog={}; rejected=[]
    for i,template in enumerate(templates):
        try:
            f,a,e=hard_state_tensors(rules['features_reconst'][i],joint_states(rules['adj_reconst'][i],rules['edge_attrs_reconst'][i]))
            rule=replace(template,rhs_feature=f,rhs_adjacency=a,rhs_edge_attr=e)
            rule.validate()
        except ValueError as exc:
            rejected.append({'native_rule_index':i,'reason':str(exc)});continue
        semantic=rule.to_payload();semantic.pop('rule_id');semantic.pop('native_rule_index')
        key=stable_sha256({'adapter':SCHEMA,'rule':semantic})
        if key in catalog: catalog[key]['generator_native_indices'].append(i);continue
        cid='globalgce-gin-aplus-'+key[:24]
        catalog[key]={'candidate_id':cid,'rule':replace(rule,rule_id=cid).to_payload(),
            'generator_native_indices':[i], 'molecular_adapter':SCHEMA,
            'action_kind':'lhs_rhs_graph_transformation_rule',
            'action_semantics':'native_replacement_explicit_parent_boundary_anchors_v1',
            'source_split':'train','generation_oracle':'GIN','method_variant':'GlobalGCE-ChemAligned-GIN-Aplus',
            'standalone_rhs_validity_claimed':False,'complete_parent_validation_required':True}
    if not catalog: raise ValueError('EMPTY_MATERIALIZED_POOL_NOT_ZERO_RESULT')
    root=Path(output_root);root.mkdir(parents=True,exist_ok=False)
    atomic_jsonl(root/'candidate_universe.jsonl',list(catalog.values()))
    atomic_jsonl(root/'candidate_rejections.jsonl',rejected)
    result={'state':'TRAIN_POOL_FROZEN','schema':'bace_globalgce_gin_aplus_pool_v1',
        'method_id':'globalgce','method_variant':'GlobalGCE-ChemAligned-GIN-Aplus',
        'candidate_count':len(catalog),'original_mined_lhs_count':80,
        'candidate_universe_sha256':sha256_file(root/'candidate_universe.jsonl'),
        'source_generator_checkpoint':config['warmstart_checkpoint'],'source_generator_oracle':'GINE',
        'training_oracle':'GIN','training_oracle_model_sha256':config['gin_model_sha256'],
        'training_oracle_temperature_sha256':config['gin_temperature_sha256'],
        'training_contract_sha256':stable_sha256(config), 'joint_contract':config['joint_contract'],
        'selected_best_rules_sha256':sha256_file(trained/'best_rules.pt'),
        'validation_selection':read_json(trained/'best_validation.json'),
        'new_campaign_optimizer_updates':terminal['optimizer_updates'],
        'original_epoch35_warmstart':True,'seamless_resume_claimed':False,
        'calibration_loaded':False,'test_loaded':False,'benchmark_test_previously_seen':True,
        'repair_selected_using_test':False,'mining_rerun':False,'main_matrix_write':False,
        'created_at':utc_now()}
    atomic_json(root/'run_manifest.json',result)
    return result
