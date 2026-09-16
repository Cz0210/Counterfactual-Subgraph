#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.eval.cm4_v6_reselect import CM4,first_prototype_audit
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--action',choices=['select','evaluate','audit','first-prototype'],required=True)
 p.add_argument('--dataset',choices=['AIDS','Mutagenicity','BACE']);p.add_argument('--source-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--contract');p.add_argument('--v5-cm-root');a=p.parse_args()
 if a.action=='first-prototype':first_prototype_audit(a.contract,a.source_root,a.v5_cm_root,a.output_root)
 else:getattr(CM4(a.dataset,a.source_root,a.output_root),a.action)()
if __name__=='__main__':main()
