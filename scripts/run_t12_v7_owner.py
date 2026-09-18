#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.t12_v7_runtime_owner import owner,provider

def main():
    p=argparse.ArgumentParser('Bind the real existing T12 restore500 stage owner')
    p.add_argument('--config',required=True);p.add_argument('--action',choices=['owner','provider','status'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--template');p.add_argument('--registry')
    p.add_argument('--observer-receipt')
    p.add_argument('--io-overlay',help='V10 authenticated history-only relocation/buffering policy')
    a=p.parse_args()
    if a.action=='owner':return owner(template=a.template,root=a.root,registry=a.registry,code_root=ROOT,observer_receipt=a.observer_receipt,io_overlay=a.io_overlay)
    if a.action=='provider':result=provider(a.root)
    else:
        result={name:json.loads((Path(a.root)/name).read_text()) for name in ['runtime_identity.json','admission.json','heartbeat.json','terminal.json'] if (Path(a.root)/name).exists()}
    print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
