"""Exercise the cluster-shell bootstrap without loading local shell settings."""
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "preflight_bace_gnn_cpu.sh", "run_bace_gnn_cpu.sh",
    "evaluate_bace_gnn_seed7.sh", "package_bace_gnn_seed7.sh",
    "build_bace_gnn_bundle.sh",
    "resume_bace_gnn_cpu.sh",
    "finalize_bace_gnn_cpu.sh",
)


@pytest.mark.parametrize("name", WRAPPERS)
def test_site_bashrc_and_conda_optional_unset_vars_then_restore_nounset(tmp_path: Path, name: str) -> None:
    script = ROOT / "scripts/slurm" / name
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
    mock = tmp_path / "site.bashrc"
    mock.write_text(
        'if [ -z "$BASHRCSOURCED" ]; then BASHRCSOURCED=1; fi\n'
        'conda() {\n'
        '  [ "$1" = activate ] && [ "$2" = smiles_pip118 ]\n'
        '  if [ -z "$CONDA_OPTIONAL_UNSET" ]; then GNN_MOCK_CONDA_ACTIVE=1; fi\n'
        '}\n'
    )
    text = script.read_text()
    start = text.index("set -euo pipefail")
    end = text.index("set -u", text.index("conda activate smiles_pip118")) + len("set -u")
    prefix = text[start:end].replace("source ~/.bashrc", "source " + shlex.quote(str(mock)))
    checks = '\n[ "$BASHRCSOURCED" = 1 ]\n[ "$GNN_MOCK_CONDA_ACTIVE" = 1 ]\ncase "$-" in *u*) :;; *) exit 41;; esac\n'
    env = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(["bash", "-c", prefix + checks], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    restored = subprocess.run(
        ["bash", "-c", prefix + checks + '\n: "$GNN_POST_BOOTSTRAP_UNSET"\n'],
        capture_output=True, text=True, env=env,
    )
    assert restored.returncode != 0
    assert "unbound variable" in restored.stderr
