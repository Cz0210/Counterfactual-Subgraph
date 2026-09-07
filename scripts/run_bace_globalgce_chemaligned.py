#!/usr/bin/env python3
"""BACE ChemAligned isolated CLI; train/validation only, no matrix writer."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Project runtime config, retained for paired Slurm CLI parity')
    p.add_argument('--repair-config', required=True)
    p.add_argument('--action', choices=('rematerialize', 'train', 'train-canary', 'export', 'status', 'owner'), required=True)
    p.add_argument('--owner-spec')
    p.add_argument('--rematerialization-root')
    p.add_argument('--training-root')
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output-root', required=True)
    p.add_argument('--set', action='append', default=[])
    args = p.parse_args()
    if not Path(args.config).is_file():
        p.error('runtime config does not exist')
    if args.action == 'status':
        root = Path(args.output_root)
        print(json.dumps({name: json.loads((root/name).read_text()) for name in
                          ('owner.json', 'heartbeat.json', 'terminal.json') if (root/name).is_file()}, indent=2))
        return
    if args.action == 'owner':
        if not args.owner_spec: p.error('owner requires the sealed --owner-spec')
        from src.baselines.bace_globalgce_chemaligned_owner import run_owner
        raise SystemExit(run_owner(args.owner_spec))
    if args.action == 'rematerialize' and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        p.error('CPU rematerialization requires CUDA_VISIBLE_DEVICES empty')
    boundary = lambda: None
    if args.device == 'cuda:0':
        from src.ablations.llm.existing_gpu_owner import receive_owner_binding, validate_inherited_lease
        from src.eval.bace_frozen_gnn_contracts import atomic_json, utc_now
        binding=receive_owner_binding()
        def boundary():
            try:
                evidence=json.loads(Path(binding['resource_live_evidence']).read_text())
                validate_inherited_lease(evidence,binding['held_gpu_lock_fd'],binding['held_project_slot_fd'])
            except (ValueError,OSError) as error:
                if Path(args.output_root).exists():
                    atomic_json(Path(args.output_root)/'paused.json',{'state':'PAUSED_BEFORE_NEXT_OPTIMIZER_UPDATE',
                        'reason':str(error),'updated_at':utc_now()})
                raise SystemExit(75)
        # Validate the actual inherited lease before Torch/CUDA initialization.
        boundary()
    import torch
    torch.set_num_threads(2)
    if args.action == 'rematerialize':
        from src.baselines.bace_globalgce_chemaligned_runner import run_rematerialization
        result = run_rematerialization(args.repair_config, args.output_root)
    elif args.action == 'export':
        from src.baselines.bace_globalgce_chemaligned_export import export_pool
        if not args.rematerialization_root: p.error('export requires completed --rematerialization-root')
        result = export_pool(args.repair_config,args.rematerialization_root,args.output_root,args.training_root)
    else:
        if not args.rematerialization_root:
            p.error('train requires completed --rematerialization-root')
        from src.baselines.bace_globalgce_chemaligned_training import run_training
        result = run_training(args.repair_config, args.rematerialization_root, args.output_root, args.device,
                              resume=args.resume, canary=args.action == 'train-canary',boundary_check=boundary)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
