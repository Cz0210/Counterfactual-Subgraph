#!/usr/bin/env python3
"""Thin CPU entrypoint for the independent Taste theta0.1 protocol."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.ours_taste_theta010 import prepare,select

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','select','status'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path)
    p.add_argument('--protocol-config',type=Path)
    a=p.parse_args();assert a.config.resolve()==(ROOT/'configs/hpc.yaml').resolve()
    if a.action=='prepare':result=prepare(a.prior,a.protocol_config,a.root)
    elif a.action=='select':result=select(a.root)
    else:result={p.name:json.loads(p.read_text()) for p in a.root.glob('*receipt.json')}
    print(json.dumps({'state':'EXECUTED','action':a.action,'root':str(a.root)}))

if __name__=='__main__':main()
