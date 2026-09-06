#!/usr/bin/env python3
"""Seal T14 formal identity overlays while its existing canary keeps running."""
from pathlib import Path
import argparse
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.utils.t14_formal_binding import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--task-spec', required=True, type=Path)
    parser.add_argument('--continuation-spec', required=True, type=Path)
    parser.add_argument('--authorization', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    if not args.config.is_file() or args.config.is_symlink():
        raise ValueError('physical configuration required')
    result = prepare(args.task_spec, args.continuation_spec, args.authorization,
                     args.output_root, driver_root=REPO_ROOT)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
