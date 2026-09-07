#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.experiments.bace_gin_aplus_reporting import prepare_and_render

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Render frozen A+ Ours with unchanged V1 GCF/ComRec; no science execution')
    p.add_argument('--config',required=True)
    for name in ('aplus-root','spec','v1-source','v1-spec','output','progress'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--funnel-root',type=Path)
    a=p.parse_args()
    if not Path(a.config).is_file():p.error('config absent')
    print(json.dumps(prepare_and_render(a.aplus_root,a.spec,a.v1_source,a.v1_spec,a.output,progress_path=a.progress,funnel_root=a.funnel_root),sort_keys=True))
