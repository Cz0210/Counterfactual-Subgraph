#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.selector_selected_test_v7 import complete
from src.eval.selector_controlled_v7 import run

def main():
    p=argparse.ArgumentParser('Complete only frozen selected-union test gaps; CPU only')
    p.add_argument('--config',required=True);p.add_argument('--set',action='append',default=[])
    p.add_argument('--spec',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--phase',choices=['p0','p1'],default='p0');p.add_argument('--then-p1',action='store_true')
    a=p.parse_args();print(json.dumps(complete(a.spec,a.adapter,phase=a.phase)))
    if a.then_p1:
        spec=json.loads(Path(a.spec).read_text());root=Path(spec['output_root'])/'p1'
        if not (root/'ALL_CALIBRATION_FROZEN.json').exists():run(a.spec,phase='p1')
        print(json.dumps(complete(a.spec,a.adapter,phase='p1')))

if __name__=='__main__':main()
