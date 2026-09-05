import json
from pathlib import Path
import subprocess


def test_two_training_lanes_no_gpu_and_dependent_evaluation():
    root=Path(__file__).resolve().parents[2]
    source=(root/'scripts/hpc/gnn/submit_bace_gnn_seed7.py').read_text()
    assert "('gatedgcn_plus',None),('gin',None),('gcn','gatedgcn_plus'),('gatv2','gin')" in source
    assert "'afterany:'+receipt['jobs'][previous]" in source
    assert "'afterok:'+':'.join" in source
    for name in ('run_bace_gnn_cpu','preflight_bace_gnn_cpu','evaluate_bace_gnn_seed7','package_bace_gnn_seed7'):
        script=(root/f'scripts/slurm/{name}.sh').read_text()
        assert '#SBATCH --partition=intel' in script
        assert '#SBATCH --gres' not in script
        assert '#SBATCH --gpus' not in script
        assert 'export PYTHONPATH=$PWD' in script
        subprocess.run(['bash','-n',str(root/f'scripts/slurm/{name}.sh')],check=True)
