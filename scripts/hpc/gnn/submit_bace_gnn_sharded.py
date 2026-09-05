#!/usr/bin/env python3
"""Continue terminal CPU-admission-only evaluation as exact two-slot arrays."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.cpu_training import load_bundle
from src.ablations.gnn.sharded_evaluation import scoped
from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, sha256_file


def submit(campaign_root, worktree, expected_commit, output_root, chunk_size=32):
    campaign = scoped(campaign_root)
    output = scoped(output_root)
    worktree = Path(worktree).resolve(strict=True)
    if output.exists() or (campaign / 'sharded_submission.json').exists():
        raise FileExistsError('Inspect the existing exact-shard continuation; never duplicate it')
    if subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip() != expected_commit:
        raise ValueError('Evaluation driver commit differs')
    if subprocess.check_output(['git', '-C', str(worktree), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean immutable evaluation worktree required')
    for script in ('scripts/slurm/run_bace_gnn_sharded.sh', 'scripts/slurm/publish_bace_gnn_verified.sh',
                   'scripts/hpc/gnn/run_bace_gnn_sharded.py', 'scripts/hpc/gnn/publish_bace_gnn_verified.py'):
        if not (worktree / script).is_file():
            raise FileNotFoundError(f'Missing executable continuation: {script}')
    base = read_json(campaign / 'submission.json')
    previous = read_json(campaign / 'publication_submission.json')
    prior = Path(previous['evaluation_root'])
    admission = read_json(prior / 'cpu_admission.json')
    if (admission['state'] != 'READY_GNN_GPU_FALLBACK' or admission['core_pass'] is not False
            or admission['reason'] != 'MEASURED_RUNTIME_GATE' or admission['test_loaded'] is not False
            or admission['projected_total_seconds'] <= 12 * 3600):
        raise ValueError('Exact-shard continuation requires the measured terminal resource fallback')
    jobs = [previous['jobs']['evaluate'], previous['jobs']['package']]
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-j', ','.join(jobs), '-o', 'JobIDRaw,State', '-P'], text=True)
    states = dict(line.split('|')[:2] for line in accounting.strip().splitlines())
    if any(states.get(j) != 'COMPLETED' for j in jobs):
        raise ValueError('Prior evaluation/package must be terminal; healthy jobs are never duplicated')
    active = subprocess.check_output(['squeue', '-u', os.environ.get('USER', 'u20526'), '-h', '-o', '%j'], text=True)
    if any(n.startswith('gnn-bace-') for n in active.splitlines()):
        raise ValueError('Existing queued GNN job requires review, not another campaign')
    partition = subprocess.check_output(['scontrol', 'show', 'partition', 'intel'], text=True)
    if 'State=UP' not in partition:
        raise ValueError('CPU partition is not available')
    bundle, manifest = load_bundle(base['bundle'])
    models = read_json(campaign / 'model_roots.json')
    if not 1 <= chunk_size <= 64:
        raise ValueError('Bounded exact parent chunk size required')
    for name, digest in admission['probe_receipt_shas'].items():
        if sha256_file(prior / 'cpu_admission' / name / 'verification_timing.json') != digest:
            raise ValueError('Original real train-only timing changed')
    counts = manifest['split_row_counts']
    parent_seconds = admission['projected_evaluation_seconds'] / (5 * (counts['calibration'] + counts['test']))
    if parent_seconds * chunk_size > 12 * 3600:
        raise ValueError('Projected single shard exceeds twelve hours; use a smaller exact partition')
    free = shutil.disk_usage(campaign).free
    reserve = max(2 * 1024**3, math.ceil(free * 0.2))
    probe_sizes = [p.stat().st_size for p in (prior / 'cpu_admission').glob('*/parents/*.json') if p.name != 'progress.json']
    projected = max(2 * 1024**3, math.ceil(max(probe_sizes, default=0) * 5 * (counts['calibration'] + counts['test']) * 3))
    if projected > free - reserve:
        raise ValueError(f'HPC_GNN_STORAGE_SHORTFALL:required={projected},available={free-reserve}')
    publication = read_json(Path(previous['publication_root']) / 'finalization_receipt.json')
    if publication['status'] != 'PASS' or publication['errors']:
        raise ValueError('Completed classifier publication must independently PASS')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    spec = {'schema': 'bace_gnn_exact_parent_shards_v1', 'attempt_id': str(uuid.uuid4()),
        'driver_commit': expected_commit, 'original_scientific_commit': manifest['execution_commit'],
        'bundle_root': str(bundle), 'bundle_file_sha256': sha256_file(bundle / 'bundle_manifest.json'),
        'models': models, 'model_files': {n: {f: sha256_file(Path(p) / f) for f in ('model.pt', 'temperature_scaling.json', 'model_card.json', 'config.yaml', 'feature_schema.json', 'sha256sums.txt')} for n, p in models.items()},
        'evaluation_root': str(output / 'evaluation'), 'batch_size': 256, 'cpu_threads': 8,
        'chunk_size': chunk_size, 'slots': {s: math.ceil(counts[s] / chunk_size) for s in ('calibration', 'test')},
        'source_admission': str(prior / 'cpu_admission.json'), 'source_admission_sha256': sha256_file(prior / 'cpu_admission.json'),
        'reference_probe_parent_files': {n: {p.name: sha256_file(p) for p in (prior / 'cpu_admission' / n / 'parents').glob('*.json') if p.name != 'progress.json'} for n in models},
        'training_rerun': False, 'temperature_refit': False, 'candidate_regeneration': False,
        'max_concurrent_jobs': 2, 'main_matrix_write': False,
        'projected_single_shard_seconds': parent_seconds * chunk_size}
    spec_path = output / 'task_spec.json'
    atomic_json(spec_path, spec)
    receipt_path = campaign / 'sharded_submission.json'
    receipt = {'state': 'SUBMITTING', 'driver_commit': expected_commit, 'worktree': str(worktree),
        'source_bundle_manifest_sha256': manifest['manifest_sha256'], 'original_jobs': previous['jobs'],
        'output_root': str(output), 'evaluation_root': spec['evaluation_root'], 'spec': str(spec_path),
        'publication_root': str(output / 'verified'), 'package_root': str(output / 'verified'),
        'jobs': {}, 'commands': {}, 'storage': {'available_bytes': free, 'reserve_bytes': reserve, 'projected_peak_bytes': projected},
        'max_concurrent_heavy_jobs': 2, 'gpu_requested': False, 'main_matrix_write': False,
        'partition_evidence': partition}
    atomic_json(receipt_path, receipt)
    def job(name, script, args, dependency=None, array=None):
        export = f'ALL,GNN_EXECUTION_WORKTREE={worktree},GNN_EXECUTION_COMMIT={manifest["execution_commit"]},GNN_INPUT_BUNDLE={bundle},GNN_EVALUATION_ROOT={spec["evaluation_root"]},GNN_PUBLICATION_ROOT={output}/verified,GNN_PACKAGE_ROOT={output}/package,GNN_PUBLICATION_DRIVER_COMMIT={expected_commit},GNN_ENVIRONMENT_MANIFEST={campaign}/environment.json,CUDA_VISIBLE_DEVICES='
        cmd = ['sbatch', '--parsable', f'--job-name=gnn-bace-exact-{name}', f'--chdir={worktree}',
            f'--output={output}/logs/%A_%a.out', f'--error={output}/logs/%A_%a.err', f'--export={export}']
        if dependency:
            cmd += ['--dependency=afterok:' + dependency]
        if array:
            cmd += ['--array=' + array]
        cmd += [str(worktree / script), *map(str, args)]
        receipt['commands'][name] = cmd
        atomic_json(receipt_path, receipt)
        result = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        if not result.isdigit():
            raise ValueError('Ambiguous sbatch result; inspect receipt, do not retry blindly')
        receipt['jobs'][name] = result
        atomic_json(receipt_path, receipt)
        return result
    runner = 'scripts/slurm/run_bace_gnn_sharded.sh'
    regression = job('regression', runner, ['--spec', spec_path, '--stage', 'regression'])
    prep = job('prepare', runner, ['--spec', spec_path, '--stage', 'prepare-calibration'], regression)
    cal = job('calibration', runner, ['--spec', spec_path, '--stage', 'calibration-shard'], prep, f'0-{5*spec["slots"]["calibration"]-1}%2')
    freeze = job('freeze', runner, ['--spec', spec_path, '--stage', 'freeze-calibration'], cal)
    test = job('test', runner, ['--spec', spec_path, '--stage', 'test-shard'], freeze, f'0-{5*spec["slots"]["test"]-1}%2')
    finish = job('finish', runner, ['--spec', spec_path, '--stage', 'finish'], test)
    verified = job('verify_package', 'scripts/slurm/publish_bace_gnn_verified.sh', [], finish)
    receipt['jobs']['package'] = verified
    receipt['state'] = 'SUBMITTED'
    atomic_json(receipt_path, receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    for name in ('campaign-root', 'worktree', 'expected-commit', 'output-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--chunk-size', type=int, default=32)
    args = vars(p.parse_args()); args.pop('config')
    print(json.dumps(submit(**args), indent=2))
