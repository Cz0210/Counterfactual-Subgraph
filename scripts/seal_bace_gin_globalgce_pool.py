#!/usr/bin/env python3
"""Seal original80 metadata and existing train/validation chemistry, no science."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.bace_gin_globalgce import seal_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--rematerialization-root', required=True)
parser.add_argument('--output-root', required=True)
args = parser.parse_args()
if not Path(args.config).is_file():
    parser.error('config must exist')
result = seal_manifest(args.rematerialization_root, args.output_root)
print(json.dumps({'state': result['state'], 'rule_count': result['rule_count'],
    'manifest_sha256': result['manifest_sha256'], 'output_root': args.output_root}))
