#!/usr/bin/env python3
"""Run one dataset's seven Controlled-v1 calibration selectors on CPU."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eval.selector_controlled_v7 import run

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--spec',required=True)
    p.add_argument('--phase',choices=['p0','p1'],default='p0')
    args=p.parse_args()
    print(json.dumps(run(args.spec,phase=args.phase),sort_keys=True))

if __name__=='__main__':main()
