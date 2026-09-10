#!/usr/bin/env python3
"""Offline paper view from existing authority CSVs; no evaluation calls."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.main20_saved_view import build
p=argparse.ArgumentParser(__doc__);p.add_argument('--input-root',required=True);p.add_argument('--output-dir',required=True)
p.add_argument('--allow-partial',action='store_true');p.add_argument('--config');a=p.parse_args()
print(json.dumps(build(a.input_root,a.output_dir,a.allow_partial),indent=2))
