#!/usr/bin/env python3
"""Seal the existing T13 deterministic-evidence owner interface; no science."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.utils.t13_deterministic_execution import seal_formal_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--set',action='append',default=[])
    for name in ('source-spec-root','evidence-root','original-authorization','fresh-root',
                 'fresh-science-root','fresh-cache-root'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    if not args.config.is_file() or args.set!=['inference.fallback_to_heuristic=false']:
        parser.error('existing config and disabled heuristic fallback required')
    values=vars(args);values.pop('config');values.pop('set')
    print(json.dumps(seal_formal_spec(repo_root=ROOT,**values),sort_keys=True))


if __name__=='__main__':main()
