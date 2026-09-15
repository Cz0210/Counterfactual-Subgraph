#!/usr/bin/env python3
"""Taste-only, saved-matrix ceilings and selector; no main-authority writes."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--action', required=True, choices=['extract','bounds-select','status'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    assert a.config.resolve() == (ROOT/'configs/hpc.yaml').resolve()
    if a.action=='extract':
        from src.eval.ours_taste_focus_matrix import extract
        extract(a.source,a.output)
    elif a.action=='bounds-select':
        from src.eval.ours_taste_focus_selector import run
        run(a.source,a.output)
    else:
        for f in ['bounds.json','selection_B.json','terminal.json']:
            path=a.output/f
            print(f, path.read_text() if path.exists() else 'PENDING')

if __name__=='__main__':
    main()
