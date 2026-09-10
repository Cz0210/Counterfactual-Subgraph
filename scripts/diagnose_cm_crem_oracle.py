#!/usr/bin/env python3
"""Bounded CM oracle diagnostic, without regeneration or OT."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_oracle_diagnostic import diagnostic
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/hpc.yaml')
    p.add_argument('--spec',required=True)
    p.add_argument('--source-root',required=True)
    p.add_argument('--output-root',required=True)
    a=p.parse_args();r=diagnostic(a.spec,a.source_root,a.output_root)
    print(r['status'])
    sys.exit(0 if r['status']=='DIAGNOSTIC_CAPTURE_COMPLETE_NOT_ACCEPTANCE' else 2)
