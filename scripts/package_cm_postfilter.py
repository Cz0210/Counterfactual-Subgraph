#!/usr/bin/env python3
"""Record-only accepted CM Taste/Mut portable package using existing transport."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_release import package_run
p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--root',required=True)
a=p.parse_args();print(json.dumps(package_run(a.root)))
