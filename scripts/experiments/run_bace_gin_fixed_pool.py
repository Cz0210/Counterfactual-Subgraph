#!/usr/bin/env python3
"""Thin, isolated-mode-safe BACE fixed-pool evaluation CLI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Existing project runtime YAML (read only)')
    p.add_argument('--spec', required=True, help='Immutable BACE experiment JSON')
    p.add_argument('--action', required=True, choices=('plan','verify-calibration','freeze','evaluate-test','aggregate','export','status','resume'))
    p.add_argument('--method', choices=('ours','gcfexplainer','globalgce','comrecgc'))
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int)
    p.add_argument('--resume-split', choices=('calibration','test'), default='calibration')
    args = p.parse_args()
    if not Path(args.config).is_file(): p.error('Actual --config must exist')
    from src.experiments import bace_gin_fixed_pool as driver
    spec = json.loads(Path(args.spec).read_text())
    driver.validate_spec(spec)
    if args.action in ('plan','status','export'):
        result = getattr(driver,args.action)(spec)
    else:
        if not args.method: p.error('--method required for this action')
        if args.action in ('verify-calibration','evaluate-test','resume'):
            split = args.resume_split if args.action=='resume' else ('test' if args.action=='evaluate-test' else 'calibration')
            result = driver.evaluate(spec,args.method,split,args.start,args.stop)
        else:
            result = getattr(driver,args.action)(spec,args.method)
    print(json.dumps(result,sort_keys=True,default=str))

if __name__ == '__main__': main()
