#!/usr/bin/env python3
"""Seal an explicit V5 overlay from the accepted Ours contract (no science)."""
import argparse
from pathlib import Path
import sys
import shutil
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_runtime import read_json, atomic_json, digest, file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--ours-root',required=True);p.add_argument('--yaml',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--config',help='Site config for Slurm interface compatibility; no model is loaded')
    a=p.parse_args();root=Path(a.ours_root);out=Path(a.out_dir);y=yaml.safe_load(Path(a.yaml).read_text())
    x=read_json(root/'resolved_contract.json');old=x['old_contract'];e=y['evaluation']
    if y['campaign']!='TASTE_GINE_K20_THETA010_FINAL_V5' or e['primary_theta']!=.1 or e['cost_cap']!=old['cost_cap']:
        raise ValueError('Wrong replacement configuration')
    c=dict(experiment_id=y['campaign'],dataset='TasteMolNet',num_classes=3,source_label=x['source_label'],
        allowed_destinations=x['destinations'],destination_semantics='OR',require_pred_before_source=True,
        cf_mode='strict_flip',oracle_sha256=x['oracle_id'],temperature=x['temperature'],
        temperature_receipt_sha256=old['temperature_calibration_hash'],class_mapping=x['class_mapping'],
        calibration_ids=x['calibration_ids'],test_ids=x['test_ids'],
        calibration_ids_sha256=digest(x['calibration_ids']),test_ids_sha256=digest(x['test_ids']),
        calibration_file_sha256=old['calibration_sha256'],test_file_sha256=old['declared_test_sha256'],
        source_calibration=x['source_calibration'],source_test=x['source_test'],
        k_max=20,primary_report_k=20,k_mode=e['k_mode'],primary_theta=.1,final_selection_theta=.1,primary_export_theta=.1,
        theta_units='ORIGINAL_UNSCALED_WNODE',theta_old=x['theta_old'],cost_cap=old['cost_cap'],
        primary_cost=e['primary_cost'],auxiliary_cost=e['conditional_cost'],
        distance=dict(molclr_checkpoint_sha256=old['molclr_checkpoint_sha256'],feature_schema_sha256=old['feature_schema_hash'],
                      definition='cosine-node uniform-mass exact EMD; beta=0; original numerical producers'),
        deadline=y['deadline'],delivery_reserve_hours=2,scope='POST_HOC_THETA_REVISION_METHOD_NATIVE_OPERATIONS',
        original_contract_source=str(root/'resolved_contract.json'),original_contract_sha256=file_sha(root/'resolved_contract.json'),
        instruction_yaml_sha256=file_sha(a.yaml),test_results_previously_observed=True)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'taste_eval_contract.json',c,immutable=True)
    shutil.copy2(a.yaml,out/'instruction.yaml')
    print(str(out));print('contract_sha256='+digest(c))

if __name__=='__main__':main()
