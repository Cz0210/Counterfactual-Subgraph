#!/usr/bin/env python3
"""Audit an already-complete BACE frozen-GIN method without model/OT execution."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--method', required=True, choices=('ours', 'gcfexplainer', 'comrecgc', 'globalgce'))
    parser.add_argument('--output', required=True, help='Fresh independent JSON receipt, no overwrite')
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error('Actual read-only config must exist')
    from src.experiments.bace_gin_audit import audit_method
    result = audit_method(json.loads(Path(args.spec).read_text()), args.method, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'application_seals'}, sort_keys=True))


if __name__ == '__main__':
    main()
