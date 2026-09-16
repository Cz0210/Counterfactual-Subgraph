"""V6 final-evaluation bridge; native training/mining contracts are unchanged.

This bridge is called only after both original 100-epoch branches complete.
Its independent resource command must admit postprocessing, not reuse the
training probe's 32-GiB or 256-file envelope for a larger stage.
"""
from pathlib import Path
import json
import subprocess


def complete_zero_metrics(parent_ids, threshold):
    """A completed empty rule set is a measured zero, never a failed run."""
    from src.baselines.tastemolnet_globalgce_full import DATASET, METHOD
    prefix=[]; parents=[]
    for k in range(1,21):
        row=dict(dataset=DATASET,method=METHOD,k=k,SuppCov=0.0,CCRCov=0.0,
            coverage=0.0,cost=threshold.cost_cap,fixed_capped_mean_cost=threshold.cost_cap,
            conditional_mean_cost='N/A',conditional_median_cost='N/A',CFDrop='N/A',
            FlipRate=0.0,StructRed='N/A',CovRed='N/A',ValidRate=0.0,AvgSize='N/A',
            applicable_rate=0.0,effective_rule_count=0,plateau_after_effective_k=True)
        prefix.append(row)
        for pid in sorted(parent_ids):
            parents.append(dict(dataset=DATASET,method=METHOD,k=k,parent_id=pid,
                best_distance='N/A',capped_distance=threshold.cost_cap,best_candidate_id='N/A',
                destination_label='N/A',strict_recourse_available=False,theta_star_covered=False,
                applicable=False,effective_rule_count=0,plateau_after_effective_k=True))
    return dict(prefix=prefix,parent_best=parents,
        figure3=[{key:r[key] for key in ('dataset','method','k','coverage','cost')} for r in prefix],
        figure4=[dict(dataset=DATASET,method=METHOD,k=20,threshold=x,coverage=0.0,CCRCov=0.0) for x in threshold.values],
        table2=[dict(prefix[-1])],destination=[dict(dataset=DATASET,method=METHOD,destination_label=y,
            count=0,rate='N/A',denominator=0,distribution_scope='K20 finite untargeted strict flips') for y in (0,2)],
        parent_count=len(parent_ids),pair_count=0,effective_rule_count=0)


def run_final_successor(plan, output, sample):
    """Execute real existing evaluator/auditor/publisher, never a next_action."""
    from src.utils.t13_performance_dispatch import bound_json
    from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
    from src.baselines.tastemolnet_globalgce_full import (
        FINAL_EVAL_V6, TasteGlobalGCEFullConfig, load_input_authority,
        run_t13_full, verify_t13_output, write_checkpoint)
    binding=plan.get('final_evaluation_binding')
    if not binding:
        raise ValueError('V6_FINAL_EVALUATOR_AND_PUBLISHER_BINDING_REQUIRED')
    spec=bound_json(binding)
    if (spec['protocol']!=FINAL_EVAL_V6 or spec['training_plan_original_attempt_id']!=plan['original_formal_attempt_id']
            or Path(spec['output_root'])!=output or spec['formal_quota_used']!='1/1'):
        raise ValueError('V6_SAME_RUN_IDENTITY_CHANGED')
    sample('native_branches_complete')
    evidence=json.loads(subprocess.check_output(spec['resource_provider_command'],text=True,timeout=60))
    from datetime import datetime, timezone
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(evidence['observed_at'])).total_seconds()
    if (not 0<=age<=120 or evidence.get('allowed') is not True
            or evidence.get('stage')!='T13_FINAL_EVALUATION'
            or evidence.get('policy_sha256')!=spec['resource_policy_sha256']):
        raise ValueError('POST_TRAINING_STAGE_NOT_ADMITTED')
    authority=load_input_authority(**spec['input_authority'])
    if (authority.threshold.final_eval_protocol!=FINAL_EVAL_V6 or authority.threshold.theta_star!=0.1
            or authority.threshold.cost_cap!=0.03416003659645076
            or authority.checkpoint_id!=plan['gnn_checkpoint_id']):
        raise ValueError('ACTUAL_FINAL_EVALUATION_CONTRACT_CHANGED')
    atomic_json(output/'actual_final_evaluation_binding.json',dict(binding_sha256=binding['sha256'],
        threshold_file_sha256=sha256_file(spec['input_authority']['threshold_contract']),
        actual_threshold=authority.threshold.to_dict(),resource_evidence=evidence))
    config=TasteGlobalGCEFullConfig(seed=7,epochs=100)
    # Only the newly recovered root is initialized; original training checkpoint
    # and legacy sealed threshold file are never rewritten or promoted.
    write_checkpoint(output,phase='BOTH_NATIVE_BRANCHES_COMPLETE',resume_identity=authority.resume_identity(config))
    run_t13_full(authority=authority,output_dir=output,config=config,
        wnode_cache_db=spec['wnode_cache_db'],node_embedding_cache_dir=spec['node_embedding_cache_dir'],
        device='cuda:0',resume=True)
    audit=verify_t13_output(output)
    if audit.get('passed') is not True:raise ValueError('FINAL_AUDIT_NOT_PASS')
    from src.utils.t8_hpc_t13_successor_v1 import publish_verified_t13_locator
    publish_verified_t13_locator(spec_root=spec['publisher_spec_root'],terminal_root=output,final_eval_binding=binding)
    return 0
