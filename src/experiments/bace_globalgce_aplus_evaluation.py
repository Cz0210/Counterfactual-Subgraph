"""GlobalGCE A+ native replacement evaluation, not Ours deletion or new training.

The leaf reuses the original GlobalGCE calibration selector and common fixed141
reducer. Its test runtime is constructed only after this pool's own freeze.
"""
from __future__ import annotations
import collections,fcntl,json,math,os,time
from pathlib import Path
import torch
from src.baselines.bace_globalgce_aplus import SCHEMA,CONTRACT,build_parent,materialize
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule,enumerate_labeled_rule_matches
from src.eval.bace_frozen_gnn_contracts import atomic_json,atomic_jsonl,atomic_csv,read_json,read_jsonl,sha256_file,stable_sha256,utc_now,load_bace_parents
from src.eval.counterfactual_semantics import compute_counterfactual_semantics
from src.experiments.bace_gin_fixed_pool import prefix_metrics

def bound(item):
    p=Path(item['path'])
    if not p.is_absolute() or sha256_file(p)!=item['sha256']:raise ValueError('BOUND_APLUS_INPUT_CHANGED:'+str(p))
    return read_json(p)

def validate(spec):
    if spec.get('schema')!='bace_globalgce_aplus_evaluation_v1' or spec.get('main_matrix_write') is not False:
        raise ValueError('SEPARATE_APLUS_EXPERIMENT_REQUIRED')
    config=bound(spec['training_contract'])
    if config['joint_contract']!=CONTRACT or config['repair_kind']!='GIN_ALIGNED_EPOCH35_WARMSTART':
        raise ValueError('FROZEN_GIN_APLUS_TRAINING_CONTRACT')
    for key in ('output_root','pool_root','bundle_root','calibration_csv','test_csv'):
        if not Path(spec[key]).is_absolute():raise ValueError('ABSOLUTE_PATH_REQUIRED:'+key)
    if spec['base_counts']!={'calibration':66,'test':141}:raise ValueError('FIXED_COHORT_CHANGED')
    return config

def pool(spec):
    config=validate(spec);root=Path(spec['pool_root']);m=read_json(root/'run_manifest.json')
    if (m.get('state')!='TRAIN_POOL_FROZEN' or m.get('training_contract_sha256')!=stable_sha256(config)
        or m.get('test_loaded') is not False or m.get('calibration_loaded') is not False
        or m.get('training_oracle')!='GIN' or m.get('joint_contract')!=CONTRACT):
        raise ValueError('REAL_VALIDATION_SELECTED_POOL_FREEZE_REQUIRED')
    p=root/'candidate_universe.jsonl'
    if sha256_file(p)!=m['candidate_universe_sha256']:raise ValueError('POOL_CHANGED')
    candidates=read_jsonl(p)
    if not 1<=len(candidates)<=80 or len({c['candidate_id'] for c in candidates})!=len(candidates):
        raise ValueError('UNIQUE_NATIVE_POOL_REQUIRED')
    for c in candidates:
        rule=GlobalGCENativeRule.from_payload(c['rule']);rule.validate()
        if c['molecular_adapter']!=SCHEMA:raise ValueError('NATIVE_MATERIALIZER_CHANGED')
        c['selector_chemistry']=rule.selector_chemistry()
        c['canonical_fragment']='N/A'
    return m,candidates

def verified_freeze(spec):
    p=Path(spec['output_root'])/'selection_freeze.json';f=read_json(p)
    manifest,candidates=pool(spec)
    ids=f.get('ordered_rule_ids',[])
    if (f.get('state')!='FROZEN' or f.get('spec_sha256')!=stable_sha256(spec)
        or f.get('pool_manifest_sha256')!=stable_sha256(manifest) or f.get('test_loaded') is not False
        or not 1<=len(ids)<=20 or len(ids)!=len(set(ids))
        or not set(ids)<={c['candidate_id'] for c in candidates}):raise ValueError('OWN_GLOBAL_SELECTOR_FREEZE_REQUIRED')
    return f

def prediction(oracle,featurizer,smiles,pid,split):
    from src.eval.bace_native_baseline_gnn import _graph
    g=_graph(featurizer,smiles=smiles,molecule_id=pid,split=split)
    return oracle.predict_records([g],batch_size=1)[0]

def evaluate_parent(parent,candidates,oracle,featurizer,distance,split):
    if oracle.backbone!='gin' or parent.label!=1:raise ValueError('CORRECTED_GIN_TRUE_LABEL_COHORT')
    first=GlobalGCENativeRule.from_payload(candidates[0]['rule'])
    native=build_parent(parent.smiles,atom_symbols=first.atom_symbols,bond_names=first.bond_names)
    before=prediction(oracle,featurizer,parent.smiles,parent.parent_id,split)
    pairs,apps=[],[];prediction_cache={}
    for c in candidates:
        cid=c['candidate_id'];rule=GlobalGCENativeRule.from_payload(c['rule']);records=[]
        for index,mapping in enumerate(enumerate_labeled_rule_matches(native,rule)):
            a={'parent_id':parent.parent_id,'candidate_id':cid,'split':split,'match_index':index,
                'native_mapping':sorted([int(k),int(v)] for k,v in mapping.items()),
                'action_kind':'lhs_rhs_graph_transformation_rule','molecular_adapter':SCHEMA,
                'pred_before':before['predicted_label'],'p_before':before['probabilities'],
                'logits_before':before['logits'],'parent_smiles':parent.smiles,
                'applicable':False,'cf_flip':False,'pair_strict_flip':False,'distance_ok':False,
                'wnode_distance':None,'pred_after':None,'cf_drop':None,'residual_smiles':None}
            try:
                product=materialize(native,rule,mapping,rule.rhs_feature,rule.rhs_edge_attr)
            except ValueError as error:
                a['failure_reason']='materialization:'+str(error);records.append(a);continue
            smiles=product.canonical_smiles
            if smiles not in prediction_cache:prediction_cache[smiles]=prediction(oracle,featurizer,smiles,cid,split)
            after=prediction_cache[smiles]
            sem=compute_counterfactual_semantics(source_label=1,pred_before=before['predicted_label'],
                pred_after=after['predicted_label'],probabilities_before=before['probabilities'],
                probabilities_after=after['probabilities'],rule_id=cid)
            a.update(applicable=True,residual_smiles=smiles,pred_after=after['predicted_label'],
                p_after=after['probabilities'],logits_after=after['logits'],cf_drop=float(sem.cf_drop),
                cf_flip=bool(sem.cf_flip),boundary_count=product.boundary_count,
                failure_reason=None if sem.cf_flip else 'frozen_gin_not_strict_flip')
            if sem.cf_flip:
                cost=distance.distance(parent.smiles,smiles);v=cost.get('distance')
                if not cost.get('ok') or v is None or not math.isfinite(float(v)) or float(v)<0:
                    raise ValueError(f'STRICT_FLIP_RAW_DISTANCE_FAILURE_NOT_ZERO_COVERAGE:{parent.parent_id}:{cid}:{cost.get("error")}')
                a.update(wnode_distance=float(v),distance_ok=True,pair_strict_flip=True,
                    distance_cache_hit=bool(cost.get('cache_hit')))
            records.append(a)
        feasible=[r for r in records if r['pair_strict_flip']]
        legal=[r for r in records if r['applicable']]
        # Original all-legal-match minimum; temperature-sensitive CFDrop only
        # breaks equal-distance ties. Mapping then gives deterministic identity.
        chosen=min(feasible,key=lambda r:(r['wnode_distance'],-r['cf_drop'],r['native_mapping'])) if feasible else (
            min(legal,key=lambda r:(-r['cf_drop'],r['native_mapping'])) if legal else None)
        row={'parent_id':parent.parent_id,'parent_smiles':parent.smiles,'candidate_id':cid,
            'split':split,'method':'GlobalGCE-ChemAligned-GIN-Aplus','method_id':'globalgce',
            'canonical_fragment':'N/A','pred_before':before['predicted_label'],
            'p_before':before['probabilities'],'logits_before':before['logits'],
            'pred_after':chosen['pred_after'] if chosen else None,
            'cf_drop':chosen['cf_drop'] if chosen else 0.,'cf_flip':bool(feasible),
            'pair_strict_flip':bool(feasible),'applicable':bool(legal),
            'wnode_distance':chosen['wnode_distance'] if feasible else None,
            'distance_for_selection':chosen['wnode_distance'] if feasible else '+inf',
            'failure_reason':None if feasible else 'frozen_gin_not_strict_flip' if legal else 'no_legal_native_application' if records else 'lhs_unmatched',
            'match_count':len(records),'legal_application_count':len(legal),
            'selected_match_index':chosen['match_index'] if chosen else None,
            'oracle_checkpoint_hash':oracle.checkpoint_id,'oracle_backbone':'gin',
            'source_label':1,'cf_mode':'strict_flip','native_operation_not_deletion':True}
        pairs.append(row);apps.extend(records)
    return pairs,apps

def runtime(spec,split):
    config=validate(spec)
    if split=='test':verified_freeze(spec)
    from src.ablations.gnn.cpu_evaluation import _distance,_featurizer
    from src.ablations.gnn.reach_raw_distance_reuse import raw_contract_from_bundle
    from src.ablations.llm.compact_node_cache import install_compact_node_cache
    from src.experiments.bace_gin_ours import with_native_graph_distance
    from src.oracles.gnn_oracle import GNNOracle
    root=Path(spec['bundle_root']);manifest=bound(spec['bundle_manifest'])
    oracle=GNNOracle.from_checkpoint(config['gnn_checkpoint'],device='cpu',batch_size=64)
    if oracle.backbone!='gin' or oracle.checkpoint_id!=config['gin_model_sha256'] or oracle.temperature!=config['gin_temperature']:
        raise ValueError('ACTUAL_CORRECTED_GIN_DRIFT')
    for p in oracle.model.parameters():p.requires_grad_(False)
    distance=_distance(root,manifest,Path(spec['output_root'])/'runtime'/split)
    install_compact_node_cache(distance)
    # The test descriptor is not opened before verified_freeze above. All
    # borrowed values are graph-content raw costs, never teacher masks/minima.
    index=bound(spec['raw_cost_indexes'][split])
    if index['split']!=split:raise ValueError('RAW_SPLIT_CHANGED')
    if split=='test':
        atomic_json(Path(spec['output_root'])/'test_raw_adoption.json',{
            'state':'HASH_BOUND_RAW_GRAPH_COSTS_ADOPTED_AFTER_OWN_FREEZE',
            'source':spec['raw_cost_indexes']['test'],'source_prior_freeze_sha256':index.get('new_test_freeze_sha256'),
            'new_selector_freeze_sha256':sha256_file(Path(spec['output_root'])/'selection_freeze.json'),
            'source_flip_masks_adopted':False,'source_minima_adopted':False,
            'created_at':utc_now()})
    distance=with_native_graph_distance(distance,index=index,current_raw_contract=raw_contract_from_bundle(manifest),
        repo=Path(spec['raw_kernel_source_root']))
    return oracle,_featurizer(root,manifest),distance

def evaluate(spec,split):
    config=validate(spec);manifest,candidates=pool(spec);frozen=verified_freeze(spec) if split=='test' else None
    if split not in ('calibration','test'):raise ValueError('EXPLICIT_SCIENCE_SPLIT')
    if frozen:
        lookup={c['candidate_id']:c for c in candidates};candidates=[lookup[k] for k in frozen['ordered_rule_ids']]
    parents=load_bace_parents(spec[split+'_csv'],source_label=1)
    if len(parents)!=spec['base_counts'][split] or len({p.parent_id for p in parents})!=len(parents):raise ValueError('FIXED_COHORT_CHANGED')
    root=Path(spec['output_root'])/split;root.mkdir(parents=True,exist_ok=True)
    terminal=root/'terminal.json'
    if terminal.exists():
        old=read_json(terminal)
        if old['spec_sha256']!=stable_sha256(spec):raise ValueError('STAGE_BINDING_CHANGED')
        return old
    oracle,featurizer,distance=runtime(spec,split)
    try:
        with (root/'writer.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for i,parent in enumerate(parents):
                from src.baselines.bace_globalgce_aplus_owner import cpu_admission
                resource=bound(spec['cpu_resource_config']);admission,ok=cpu_admission(resource)
                if not ok:raise ValueError('CPU_BOUNDARY_RESOURCE_WAIT:'+json.dumps(admission))
                path=root/f'parent-{i:05d}.json'
                binding=stable_sha256(dict(spec=stable_sha256(spec),split=split,parent_id=parent.parent_id,
                    candidate_ids=[c['candidate_id'] for c in candidates],freeze=frozen))
                if path.exists():
                    if read_json(path).get('binding')!=binding:raise ValueError('PARENT_CHECKPOINT_DRIFT')
                    continue
                pairs,apps=evaluate_parent(parent,candidates,oracle,featurizer,distance,split)
                atomic_json(path,{'state':'COMPLETE','binding':binding,'parent_index':i,'parent_id':parent.parent_id,
                    'spec_sha256':stable_sha256(spec),'pair_rows':pairs,'application_rows':apps,
                    'applications_sha256':stable_sha256(apps),'pairs_sha256':stable_sha256(pairs),
                    'created_at':utc_now()})
                atomic_json(root/'heartbeat.json',{'state':'RUNNING','completed_parents':i+1,
                    'total_parents':len(parents),'split':split,'updated_at':utc_now()})
        result={'state':'EVALUATION_COMPLETE','spec_sha256':stable_sha256(spec),'split':split,
            'completed_parents':len(parents),'candidate_count':len(candidates),'raw_cost_statistics':distance.stats_dict(),
            'raw_cost_reuse':getattr(distance,'used',[]),'created_at':utc_now()}
        atomic_json(terminal,result);return result
    finally:distance.close()

def rows(spec,split):
    root=Path(spec['output_root'])/split;terminal=read_json(root/'terminal.json')
    if terminal.get('state')!='EVALUATION_COMPLETE' or terminal['spec_sha256']!=stable_sha256(spec):raise ValueError('INCOMPLETE_STAGE')
    for i in range(spec['base_counts'][split]):
        p=read_json(root/f'parent-{i:05d}.json')
        if p['parent_index']!=i or p['spec_sha256']!=stable_sha256(spec) or p['applications_sha256']!=stable_sha256(p['application_rows']) or p['pairs_sha256']!=stable_sha256(p['pair_rows']):raise ValueError('PARENT_UNIT_CHANGED')
        yield from p['pair_rows']

def freeze(spec):
    validate(spec);manifest,candidates=pool(spec);root=Path(spec['output_root']);fp=root/'selection_freeze.json'
    if fp.exists():return verified_freeze(spec)
    matrix=root/'calibration_matrix';matrix.mkdir(exist_ok=False)
    pairs=list(rows(spec,'calibration'))
    if len(pairs)!=66*len(candidates) or any(p['split']!='calibration' for p in pairs):raise ValueError('CALIBRATION_MATRIX_INCOMPLETE')
    atomic_jsonl(matrix/'pair_matrix.jsonl',pairs);atomic_jsonl(matrix/'selected_candidate_universe.jsonl',candidates)
    atomic_json(matrix/'summary.json',{'parent_count':66,'selected_candidate_count':len(candidates),'test_loaded':False})
    atomic_json(matrix/'run_manifest.json',{'inputs':{'cohort_name':'calibration'},'split':'calibration','test_loaded':False,'classifier_family':'gin'})
    original=bound(spec['original_global_selector_manifest']);v=bound(spec['original_global_variant_config'])
    from src.eval.mutagenicity_wnode_selector import preregistered_variant_configs
    from dataclasses import asdict
    if (original.get('method_id')!='globalgce' or original.get('test_loaded') is not False
        or original.get('stage')!='BASELINE_CALIBRATION_SELECTOR'
        or v['variants']!={x.name:asdict(x) for x in preregistered_variant_configs()}):raise ValueError('ORIGINAL_GLOBAL_SELECTOR_SOURCE_CHANGED')
    t=bound(spec['thresholds'])
    if t!=original['thresholds']:raise ValueError('ORIGINAL_GLOBAL_THRESHOLDS_CHANGED')
    from src.experiments.bace_gin_native_baselines import select_order
    config={k:v[k] for k in ('top_k','table_k','local_swap_passes','prefix_weights')}
    config.update(seed=13,parent_limit=0,candidate_limit=0,forbid_test=True)
    details=select_order(matrix,{'method':'globalgce','test_loaded':False,
        'native_attachment_contract':SCHEMA,'original_global_selector_verified':True,
        'original_selector_config':config,'thresholds':t,'threshold_provenance':original['threshold_provenance'],
        'output_root':str(root/'selector')})
    result={'state':'FROZEN','spec_sha256':stable_sha256(spec),'pool_manifest_sha256':stable_sha256(manifest),
        'ordered_rule_ids':details['ordered_rule_ids'],'test_loaded':False,'selector_details':details,
        'original_selector_manifest':spec['original_global_selector_manifest'],'original_variants':spec['original_global_variant_config'],
        'created_at':utc_now()}
    atomic_json(fp,result);return verified_freeze(spec)

def aggregate(spec):
    config=validate(spec);f=verified_freeze(spec);pairs=list(rows(spec,'test'))
    parents=list(dict.fromkeys(p['parent_id'] for p in pairs))
    if len(parents)!=141:raise ValueError('FIXED141_REQUIRED')
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    t=threshold_bundle_from_dict(bound(spec['thresholds']))
    metrics=prefix_metrics(parents,f['ordered_rule_ids'],pairs,theta=t.theta_star,cap=t.cost_cap,endpoints=t.raw_thresholds)
    root=Path(spec['output_root']);atomic_json(root/'metrics.json',metrics)
    for name,key in [('prefix_metrics','prefix_rows'),('parent_best_distances','parent_distances'),('exact_ecdf','exact_ecdf')]:atomic_csv(root/(name+'.csv'),metrics[key])
    # Saved-record consistency is explicitly narrower than rerunning oracle/OT.
    for split in ('calibration','test'):
        for i in range(spec['base_counts'][split]):
            record=read_json(root/split/f'parent-{i:05d}.json');apps=record['application_rows']
            for pair in record['pair_rows']:
                selected=[a for a in apps if a['candidate_id']==pair['candidate_id'] and a['pair_strict_flip']]
                if any(a['pred_before']!=1 or a['pred_after']!=0 or not a['applicable'] for a in selected):raise ValueError('APPLICATION_STRICT_FLIP_CONFLICT')
                best=min((a['wnode_distance'] for a in selected),default=None)
                if best!=pair['wnode_distance'] or bool(selected)!=pair['pair_strict_flip']:raise ValueError('APPLICATION_MINIMUM_CONFLICT')
    audit={'state':'APLUS_GLOBALGCE_EVALUATION_COMPLETE','training_contract_sha256':stable_sha256(config),
        'spec_sha256':stable_sha256(spec),'selector_freeze_sha256':sha256_file(root/'selection_freeze.json'),
        'fixed_test_parent_count':141,'selected_rules':len(f['ordered_rule_ids']),
        'saved_record_result_consistency':'PASS','oracle_reexecuted_by_this_audit':False,'ot_recomputed_by_this_audit':False,
        'benchmark_test_previously_seen':True,'repair_selected_using_test':False,'main_matrix_write':False,
        'metrics_sha256':sha256_file(root/'metrics.json'),'created_at':utc_now()}
    atomic_json(root/'final_audit.json',audit);atomic_json(root/'experiment_registry.json',audit)
    return audit
