#!/usr/bin/env python3
"""Read-only new GNN-core-before-LLM gate; no model/GPU/main writer."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.ablations.gnn.early_policy import gpu_allowed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/hpc.yaml')
    p.add_argument('--gnn-evaluation-root',required=True)
    p.add_argument('--gnn-verified-archive')
    p.add_argument('--gnn-verified-archive-sha256')
    p.add_argument('--main-resource-evidence')
    p.add_argument('--require-pass',action='store_true')
    a=p.parse_args()
    if a.config!='configs/hpc.yaml':p.error('Use configs/hpc.yaml')
    root=Path(a.gnn_evaluation_root)
    evidence=json.loads(Path(a.main_resource_evidence).read_text()) if a.main_resource_evidence else {}
    core_state='WAITING_GNN_CORE_SEED7'
    if a.gnn_verified_archive and a.gnn_verified_archive_sha256:
        from src.eval.bace_frozen_gnn_contracts import sha256_file
        from src.ablations.llm.corrected_core_gate import require_corrected_gnn_core
        if sha256_file(Path(a.gnn_verified_archive)) != a.gnn_verified_archive_sha256:
            raise ValueError('Independent GNN archive SHA mismatch')
        require_corrected_gnn_core(a.gnn_verified_archive,a.gnn_verified_archive_sha256)
        core_state='PASS'
    evidence['gnn_core_seed7_audit']=core_state
    result=gpu_allowed(evidence,family='llm')
    result.update(gnn_core_seed7_state=core_state,llm_cpu_preparation_allowed=True,
                  variants=['BRICS_FIXED','CHEMLLM_7B_OFF_THE_SHELF','CHEMLLM_7B_PPO_LORA_MAIN','CHEMLLM_2B_OFF_THE_SHELF'],
                  project_sft_checkpoint_exists=False,secondary_seeds_required=False)
    print(json.dumps(result,indent=2))
    return 0 if not a.require_pass or result['allowed'] else 3

if __name__=='__main__':raise SystemExit(main())
