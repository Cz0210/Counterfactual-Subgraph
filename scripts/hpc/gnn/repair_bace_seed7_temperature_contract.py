#!/usr/bin/env python3
"""First-fit four BACE frozen GNN temperatures and reconcile existing results."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn import temperature_repair as repair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--action',required=True,choices=('plan','fit','reconcile-calibration','freeze',
        'reconcile-test','finish','verify-package','status'))
    p.add_argument('--output-root',required=True)
    p.add_argument('--source-spec')
    p.add_argument('--original-package')
    p.add_argument('--authorization')
    p.add_argument('--driver-commit')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1'):
        raise ValueError('CPU_CORRECTION_MUST_NOT_SEE_GPU')
    if a.action=='plan':
        result=repair.plan(source_spec=a.source_spec,original_package=a.original_package,
            output_root=a.output_root,authorization=a.authorization,driver_commit=a.driver_commit)
    elif a.action=='fit': result=repair.fit(a.output_root)
    elif a.action=='reconcile-calibration': result=repair.reconcile(a.output_root,split='calibration')
    elif a.action=='freeze': result=repair.replay_phase(a.output_root,phase='calibration')
    elif a.action=='reconcile-test': result=repair.reconcile(a.output_root,split='test')
    elif a.action=='finish': result=repair.replay_phase(a.output_root,phase='finish')
    elif a.action=='verify-package': result=repair.verify_package(a.output_root)
    else: result=repair.status(a.output_root)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # CLI error is honest Slurm failure. No implicit retry or change to any
        # sealed source, model, temperature or main-table owner is performed.
        raise SystemExit(2) from exc
