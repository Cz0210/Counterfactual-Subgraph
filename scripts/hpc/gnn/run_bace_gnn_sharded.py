#!/usr/bin/env python3
"""Execute one exact BACE parent-partition or global closeout stage."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.sharded_evaluation import advance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--stage', choices=('regression', 'prepare-calibration', 'calibration-shard', 'freeze-calibration', 'test-shard', 'finish'), required=True)
    p.add_argument('--index', type=int)
    a = p.parse_args()
    index = a.index
    if a.stage.endswith('-shard') and index is None:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    result = advance(json.loads(Path(a.spec).read_text()), a.stage, index)
    print(json.dumps({k: v for k, v in result.items() if k not in {'predictions', 'parents', 'shards', 'files'}}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
