#!/usr/bin/env python3
"""Export saved A+ calibration/full-pool and frozen-control test funnels."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if not Path(a.config).is_file():p.error('existing runtime config required')
    from src.experiments.bace_gin_aplus_funnel import export
    print(json.dumps(export(json.loads(Path(a.spec).read_text()),a.output),sort_keys=True))
if __name__=='__main__':main()
