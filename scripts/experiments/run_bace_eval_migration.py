#!/usr/bin/env python3
"""BACE GNN-A / LLM-GIN frozen-input CPU evaluation; resume per parent."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.experiments.bace_eval_migration import run,validate,freeze,export

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--spec',required=True,type=Path)
    p.add_argument('--action',choices=('plan','run','resume','freeze','export','status'),required=True)
    a=p.parse_args()
    if not a.config.is_file() or a.set!=['inference.fallback_to_heuristic=false']:
        p.error('Actual config and fail-closed inference required')
    spec=json.loads(a.spec.read_text());root=validate(spec)
    if a.action=='status':
        value={x:json.loads((root/x).read_text()) for x in ('final_audit.json','progress.json') if (root/x).is_file()}
    elif a.action=='plan': value={'scope':spec['scope'],'roles':list(spec['roles']),'science_started':False}
    elif a.action in ('run','resume'): value=run(spec)
    else: value={'freeze':freeze,'export':export}[a.action](spec)
    print(json.dumps(value,default=str))
if __name__=='__main__':main()
