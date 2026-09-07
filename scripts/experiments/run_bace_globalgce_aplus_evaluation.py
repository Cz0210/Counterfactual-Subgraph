#!/usr/bin/env python3
"""Independent GlobalGCE GIN A+ CPU calibration/freeze/test leaf."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--action',required=True,choices=('calibration','freeze','test','aggregate','cpu-handoff','status'))
    p.add_argument('--set',action='append',default=[])
    a=p.parse_args()
    if not Path(a.config).is_file():p.error('real runtime config required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):p.error('CPU evaluation requires no visible GPU')
    import torch;torch.set_num_threads(2)
    from src.experiments import bace_globalgce_aplus_evaluation as leaf
    spec=json.loads(Path(a.spec).read_text())
    if a.action=='cpu-handoff':
        from src.baselines.bace_globalgce_aplus_owner import run_cpu_handoff
        return run_cpu_handoff(a.spec)
    if a.action in ('calibration','test'):result=leaf.evaluate(spec,a.action)
    elif a.action=='status':
        root=Path(spec['output_root']);result={}
        for name in ('calibration/terminal.json','selection_freeze.json','test/terminal.json','final_audit.json'):
            if (root/name).exists():result[name]=json.loads((root/name).read_text())
    else:result=getattr(leaf,a.action)(spec)
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_cost_reuse',)},default=str))
if __name__=='__main__':
    try:sys.exit(main())
    except ValueError as error:
        if str(error).startswith('CPU_BOUNDARY_RESOURCE_WAIT:'):
            print(str(error),file=sys.stderr);sys.exit(75)
        raise
