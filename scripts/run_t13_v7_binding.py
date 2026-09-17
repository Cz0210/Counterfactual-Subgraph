#!/usr/bin/env python3
"""Bind and execute the existing T13 same-formal owner under V7 authority."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import t13_v7_binding as binding

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--action',choices=['prepare','owner','post-provider','status'],required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--old-plan-root');p.add_argument('--old-publisher-root');p.add_argument('--authorization-file')
    a=p.parse_args()
    if a.action=='owner':return binding.owner(a.root)
    if a.action=='prepare':
        value=binding.prepare(old_plan_root=a.old_plan_root,old_publisher_root=a.old_publisher_root,
            root=a.root,execution_root=ROOT,authorization_file=a.authorization_file)
    elif a.action=='post-provider':value=binding.post_provider(a.root)
    else:
        value={}
        for name in ['authorization_overlay.json','registry_heartbeat.json','owner/heartbeat.json','owner/terminal.json',
                     'science/formal_progress.json','science/runtime_backend_receipt.json','science/probe.json']:
            f=Path(a.root)/name
            if f.exists():value[name]=binding.read(f)
    print(json.dumps(value,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
