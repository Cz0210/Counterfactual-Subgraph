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
    p.add_argument('--action', choices=('rematerialize', 'status'), required=True)
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
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        p.error('CPU rematerialization requires CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(2)
    from src.baselines.bace_globalgce_chemaligned_runner import run_rematerialization
    print(json.dumps(run_rematerialization(args.repair_config, args.output_root), indent=2))

if __name__ == '__main__':
    main()
