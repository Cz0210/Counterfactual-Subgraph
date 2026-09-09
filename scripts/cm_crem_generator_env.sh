#!/usr/bin/env bash
# CM-CReM only: new independent environment, no shared-env/registry mutation.
set -eo pipefail

CM_TASK_ROOT="${1:-/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1}"
CM_CONDA="/share/home/u20526/anaconda3/bin/conda"
CM_BOOTSTRAP_PYTHON="/share/home/u20526/anaconda3/bin/python"
CM_ENV_ROOT="$CM_TASK_ROOT/environment-20260910-v1"
CM_PREFIX="$CM_ENV_ROOT/python3115-crem0214"

"$CM_BOOTSTRAP_PYTHON" -I -B - "$CM_TASK_ROOT" "$CM_ENV_ROOT" <<'PY'
import os, pathlib, sys
allowed = pathlib.Path('/share/home/u20526/czx').resolve(strict=True)
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
if not root.is_relative_to(allowed) or root == allowed:
    raise SystemExit('environment task root is outside the exact project scope')
if pathlib.Path(sys.argv[2]).exists():
    raise SystemExit('environment attempt already exists; inspect receipt, do not reinstall blindly')
v = os.statvfs(root)
if v.f_bavail * v.f_frsize < 10 * 1024**3:
    raise SystemExit('need >=10GiB free before package/cache reservation; no install performed')
PY

mkdir -p "$CM_ENV_ROOT" "$CM_ENV_ROOT/conda-pkgs" "$CM_ENV_ROOT/conda-envs" \
  "$CM_ENV_ROOT/pip-cache" "$CM_ENV_ROOT/tmp" "$CM_ENV_ROOT/xdg-cache"
export CONDA_PKGS_DIRS="$CM_ENV_ROOT/conda-pkgs"
export CONDA_ENVS_PATH="$CM_ENV_ROOT/conda-envs"
export CONDA_REGISTER_ENVS=false
export CONDA_AUTO_UPDATE_CONDA=false
export CONDA_NOTIFY_OUTDATED_CONDA=false
export CONDA_REPORT_ERRORS=false
export XDG_CACHE_HOME="$CM_ENV_ROOT/xdg-cache"
export PIP_CACHE_DIR="$CM_ENV_ROOT/pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$CM_ENV_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cm_write_terminal() {
  "$CM_BOOTSTRAP_PYTHON" -I -B - "$CM_ENV_ROOT" "$CM_PREFIX" "$1" "$2" <<'PY'
import datetime, json, os, pathlib, sys
root, prefix, state, code = sys.argv[1:]
record = dict(schema='cm_crem_generator_environment_v1', status=state, exit_code=int(code),
              prefix=prefix, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              shared_environment_modified=False, conda_register_envs=False,
              scope='generation_only_no_torch_no_difflinker', install_attempts=1)
target = pathlib.Path(root) / 'installation_terminal.json'
with target.open('x') as stream:
    json.dump(record, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
PY
}

CM_STAGE=conda_python
trap 'CM_CODE=$?; cm_write_terminal "FAILED_$CM_STAGE" "$CM_CODE"; exit "$CM_CODE"' ERR
timeout --signal=TERM 1200 "$CM_CONDA" create --yes --prefix "$CM_PREFIX" \
  --solver classic --override-channels --channel https://repo.anaconda.com/pkgs/main \
  python=3.11.5 pip > "$CM_ENV_ROOT/conda-install.log" 2>&1

CM_STAGE=pinned_pip
timeout --signal=TERM 600 "$CM_PREFIX/bin/python" -I -B -m pip install \
  --disable-pip-version-check --only-binary=:all: --retries 1 --timeout 30 \
  crem==0.2.14 rdkit==2023.9.6 numpy==1.26.4 > "$CM_ENV_ROOT/pip-install.log" 2>&1

CM_STAGE=actual_version_readback
"$CM_PREFIX/bin/python" -I -B - "$CM_ENV_ROOT" <<'PY'
import importlib.metadata, json, os, pathlib, sys
versions = {n: importlib.metadata.version(n) for n in ('crem', 'rdkit', 'numpy')}
versions['python'] = '.'.join(map(str, sys.version_info[:3]))
expected = {'python':'3.11.5','crem':'0.2.14','rdkit':'2023.9.6','numpy':'1.26.4'}
assert versions == expected, versions
from rdkit import Chem
from crem.crem import mutate_mol
assert Chem.MolFromSmiles('N[C@@H](C)C(=O)O') is not None and callable(mutate_mol)
with (pathlib.Path(sys.argv[1]) / 'actual_versions.json').open('x') as stream:
    json.dump(dict(versions=versions, executable=sys.executable,
                   import_smoke='PASS_NOT_CH_EMBL_GENERATION',
                   distributions=sorted((d.metadata['Name'], d.version)
                                        for d in importlib.metadata.distributions())), stream, indent=2)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
PY
cm_write_terminal INSTALLED_IMPORT_SMOKE_PASS 0
printf '%s\n' "$CM_PREFIX/bin/python"
