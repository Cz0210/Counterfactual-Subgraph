#!/usr/bin/env python3
"""V8 independent selection timing: no test, oracle, or distance computation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.selector_timing_v8 import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--spec',action='append',required=True)
    p.add_argument('--out-dir',required=True)
    args=p.parse_args()
    run(args.spec,args.out_dir)
