#!/usr/bin/env python3
"""AutoDL real T12 adapter/observer evidence; verify is CPU-only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.t12_real_regression_v9 import produce,verify,write

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=['produce','verify'],required=True)
    p.add_argument('--template',type=Path)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--budget',type=Path)
    a=p.parse_args()
    if not a.config.is_file():raise ValueError('CONFIG_MISSING')
    try:
        if a.action=='produce':produce(a.template,a.root,a.budget)
        else:print(json.dumps(verify(a.root),sort_keys=True))
    except Exception as e:
        if a.root.is_dir():write(a.root/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
if __name__=='__main__':main()
