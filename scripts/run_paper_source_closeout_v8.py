#!/usr/bin/env python3
"""Source-bound saved-record paper export, or offline CSV-only replot."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.eval.paper_source_closeout_v8 import export,plot

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config')
    p.add_argument('--base')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--paper')
    p.add_argument('--source-csv-dir',help='Explicit saved CSV input directory for replot-only')
    p.add_argument('--replot-only',action='store_true')
    args=p.parse_args()
    if args.replot_only:plot(args.out_dir,args.source_csv_dir)
    else:
        if not args.base or not args.paper:p.error('--base and --paper required for export')
        print(json.dumps(export(args.base,args.out_dir,args.paper),sort_keys=True))
