from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import src.baselines.gcfexplainer_mutagenicity_runtime as gcf_runtime

from scripts.baselines.gcfexplainer import run_mutagenicity_summary as summary_cli
from scripts.baselines.gcfexplainer.run_mutagenicity_vrrw import (
    _resolve_profile_defaults,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
SLURM_ROOT = ROOT / "scripts/slurm/gcfexplainer"
WRAPPERS = (
    SLURM_ROOT / "prepare_mutagenicity_dataset.sh",
    SLURM_ROOT / "train_mutagenicity_gnn.sh",
    SLURM_ROOT / "reproduce_mutagenicity_vrrw.sh",
    SLURM_ROOT / "reproduce_mutagenicity_summary.sh",
    SLURM_ROOT / "reproduce_mutagenicity_all.sh",
)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_wrapper_bash_syntax_and_resources(wrapper: Path) -> None:
    result = subprocess.run(["bash", "-n", str(wrapper)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    match = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    assert match and int(match.group(1)) <= 7
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "unset all_proxy" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
    assert "outputs/hpc/mutagenicity/baselines/gcfexplainer" in text or "RUN_ROOT" in text


def test_aids_wrappers_remain_dataset_specific_and_unchanged_in_contract() -> None:
    aids_train = (SLURM_ROOT / "train_aids_gnn.sh").read_text(encoding="utf-8")
    aids_vrrw = (SLURM_ROOT / "reproduce_aids_vrrw.sh").read_text(encoding="utf-8")
    aids_summary = (SLURM_ROOT / "reproduce_aids_summary.sh").read_text(encoding="utf-8")
    assert "gnn.py --dataset aids" in aids_train
    assert "python vrrw.py --dataset aids" in aids_vrrw
    assert "python summary.py --dataset aids" in aids_summary


def test_dataset_wrapper_uses_only_strict_train_and_val_files() -> None:
    text = WRAPPERS[0].read_text(encoding="utf-8")
    assert "train_source_label1_teacher_correct.csv" in text
    assert "train_target_label0_teacher_correct.csv" in text
    assert "val_source_label1_teacher_correct.csv" in text
    assert "val_target_label0_teacher_correct.csv" in text
    assert "calibration_source_label1_teacher_correct.csv" not in text
    assert "test_source_label1_teacher_correct.csv" not in text


def test_gnn_wrapper_profiles_and_checkpoint_isolation() -> None:
    text = WRAPPERS[1].read_text(encoding="utf-8")
    assert 'PROFILE="${PROFILE:-smoke}"' in text
    assert 'EPOCHS="${EPOCHS:-5}"' in text
    assert 'TRAIN_LIMIT="${TRAIN_LIMIT:-512}"' in text
    assert 'VAL_LIMIT="${VAL_LIMIT:-128}"' in text
    assert 'EPOCHS="${EPOCHS:-1000}"' in text
    assert 'TRAIN_LIMIT="${TRAIN_LIMIT:-2885}"' in text
    assert 'VAL_LIMIT="${VAL_LIMIT:-355}"' in text
    assert "ALLOW_FULL" not in text
    assert 'GNN_DIR="${GNN_DIR:-}"' in text
    assert "explicit_nonempty_path" in text
    assert "data/aids/gnn" not in text


def _run_gnn_wrapper(
    tmp_path: Path,
    **overrides: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    project_root = tmp_path / "project"
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "_PHASE_A_COMPLETE.json").write_text(
        '{"run_complete": true}\n', encoding="utf-8"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_log = tmp_path / "python_argv.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/bin/bash
set -e
if [[ "${1:-}" == *train_mutagenicity_gnn.py ]]; then
  printf '%s\n' "$@" > "$GNN_TEST_ARGV_LOG"
  output_dir=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--output-dir" ]]; then
      output_dir="$argument"
      break
    fi
    previous="$argument"
  done
  mkdir -p "$output_dir"
  printf '{"run_complete": true}\n' > "$output_dir/_RUN_COMPLETE.json"
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/bash\nprintf '0123456789abcdef0123456789abcdef01234567\\n'\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text(
        f'export PATH="{fake_bin}:$PATH"\nconda() {{ return 0; }}\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    for name in (
        "PROFILE",
        "EPOCHS",
        "TRAIN_LIMIT",
        "VAL_LIMIT",
        "SEED",
        "GNN_DIR",
        "RUN_ROOT",
        "ALLOW_FULL",
    ):
        env.pop(name, None)
    env.update(
        {
            "HOME": str(home),
            "PROJECT_ROOT": str(project_root),
            "DATASET_DIR": str(dataset_dir),
            "GNN_TEST_ARGV_LOG": str(argv_log),
            "RESUME": "false",
            **overrides,
        }
    )
    result = subprocess.run(
        ["bash", str(WRAPPERS[1])],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    return result, argv_log


def test_gnn_smoke_profile_still_uses_existing_contract(tmp_path: Path) -> None:
    result, argv_log = _run_gnn_wrapper(tmp_path, PROFILE="smoke")
    assert result.returncode == 0, result.stderr
    argv = argv_log.read_text(encoding="utf-8").splitlines()
    assert [argv[argv.index(flag) + 1] for flag in ("--profile", "--epochs", "--train-limit", "--val-limit", "--seed")] == [
        "smoke",
        "5",
        "512",
        "128",
        "13",
    ]


def test_gnn_full_profile_passes_exact_contract_and_cli_forwarding(
    tmp_path: Path,
) -> None:
    gnn_dir = tmp_path / "full_gnn"
    result, argv_log = _run_gnn_wrapper(
        tmp_path,
        PROFILE="full",
        EPOCHS="1000",
        TRAIN_LIMIT="2885",
        VAL_LIMIT="355",
        SEED="13",
        GNN_DIR=str(gnn_dir),
    )
    assert result.returncode == 0, result.stderr
    assert "[MUTAGENICITY_GCFEXPLAINER_GNN_CONFIG]" in result.stdout
    assert "profile=full" in result.stdout
    assert "epochs=1000" in result.stdout
    assert "train_limit=2885" in result.stdout
    assert "val_limit=355" in result.stdout
    assert "calibration_loaded=false" in result.stdout
    assert "test_loaded=false" in result.stdout
    argv = argv_log.read_text(encoding="utf-8").splitlines()
    expected = {
        "--profile": "full",
        "--epochs": "1000",
        "--train-limit": "2885",
        "--val-limit": "355",
        "--seed": "13",
        "--dataset-dir": str(tmp_path / "project" / "dataset"),
        "--output-dir": str(gnn_dir),
    }
    for flag, value in expected.items():
        assert argv[argv.index(flag) + 1] == value
    assert "smoke_v1" not in "\n".join(argv)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("EPOCHS", "5", "1000"),
        ("TRAIN_LIMIT", "2884", "2885"),
        ("VAL_LIMIT", "354", "355"),
    ),
)
def test_gnn_full_rejects_noncontract_values(
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    values = {
        "PROFILE": "full",
        "EPOCHS": "1000",
        "TRAIN_LIMIT": "2885",
        "VAL_LIMIT": "355",
        "SEED": "13",
        "GNN_DIR": str(tmp_path / "full_gnn"),
    }
    values[field] = value
    result, _argv_log = _run_gnn_wrapper(tmp_path, **values)
    assert result.returncode != 0
    assert "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" in result.stderr
    assert "profile=full" in result.stderr
    assert f"field={field.lower()}" in result.stderr
    assert f"actual={value}" in result.stderr
    assert f"expected={expected}" in result.stderr


def test_gnn_full_requires_explicit_output_dir(tmp_path: Path) -> None:
    result, _argv_log = _run_gnn_wrapper(
        tmp_path,
        PROFILE="full",
        EPOCHS="1000",
        TRAIN_LIMIT="2885",
        VAL_LIMIT="355",
        SEED="13",
    )
    assert result.returncode != 0
    assert "field=gnn_dir" in result.stderr
    assert "expected=explicit_nonempty_path" in result.stderr


def test_gnn_full_rejects_smoke_output_path(tmp_path: Path) -> None:
    result, _argv_log = _run_gnn_wrapper(
        tmp_path,
        PROFILE="full",
        EPOCHS="1000",
        TRAIN_LIMIT="2885",
        VAL_LIMIT="355",
        SEED="13",
        GNN_DIR=str(tmp_path / "smoke_v1" / "gnn"),
    )
    assert result.returncode != 0
    assert "field=gnn_dir" in result.stderr
    assert "expected=explicit_non_smoke_output_path" in result.stderr


def test_gnn_wrapper_rejects_unknown_profile(tmp_path: Path) -> None:
    result, _argv_log = _run_gnn_wrapper(tmp_path, PROFILE="experimental")
    assert result.returncode != 0
    assert "field=profile" in result.stderr
    assert "actual=experimental" in result.stderr
    assert "expected=smoke_or_full" in result.stderr


def test_later_stage_wrappers_keep_full_authorization_gate() -> None:
    for wrapper in (WRAPPERS[2], WRAPPERS[3]):
        text = wrapper.read_text(encoding="utf-8")
        assert '"${ALLOW_FULL:-false}" == "true"' in text
        assert "invalid/unauthorized PROFILE=$PROFILE" in text


def test_vrrw_wrapper_has_preregistered_mutagenicity_parameters() -> None:
    text = WRAPPERS[2].read_text(encoding="utf-8")
    assert 'VRRW_M="${VRRW_M:-500}"' in text
    assert 'VRRW_M="${VRRW_M:-50000}"' in text
    assert 'VRRW_ALPHA="${VRRW_ALPHA:-1.0}"' in text
    assert 'VRRW_THETA="${VRRW_THETA:-0.05}"' in text
    assert 'VRRW_SEED="${VRRW_SEED:-13}"' in text
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-64}"' in text
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-1448}"' in text
    assert '--parent-limit "$VRRW_PARENT_LIMIT"' in text
    assert '--m "$VRRW_M"' in text
    assert "[MUTAGENICITY_GCFEXPLAINER_VRRW_CONFIG]" in text
    assert 'echo "parent_limit=$VRRW_PARENT_LIMIT"' in text
    assert 'echo "M=$VRRW_M"' in text
    assert "--no-sample" in text
    assert "MAX_STEPS" not in text


def test_all_wrapper_explicitly_passes_profile_derived_vrrw_contract() -> None:
    text = WRAPPERS[4].read_text(encoding="utf-8")
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-64}"' in text
    assert 'VRRW_M="${VRRW_M:-500}"' in text
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-1448}"' in text
    assert 'VRRW_M="${VRRW_M:-50000}"' in text
    assert 'VRRW_ALPHA="${VRRW_ALPHA:-1.0}"' in text
    assert 'VRRW_THETA="${VRRW_THETA:-0.05}"' in text
    assert 'VRRW_SEED="${VRRW_SEED:-13}"' in text
    assert 'VRRW_PARENT_LIMIT="$VRRW_PARENT_LIMIT"' in text
    assert 'VRRW_M="$VRRW_M"' in text
    assert 'VRRW_ALPHA="$VRRW_ALPHA"' in text
    assert 'VRRW_THETA="$VRRW_THETA"' in text
    assert 'VRRW_SEED="$VRRW_SEED"' in text
    assert 'PARENT_LIMIT="$VRRW_PARENT_LIMIT"' in text
    assert 'EXPORT_DIR="${EXPORT_DIR:-}"' in text
    assert 'EXPORT_DIR="$EXPORT_DIR"' in text
    assert "MAX_STEPS" not in text


def _parse_vrrw(*extra: str):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-dir",
            "dataset",
            "--official-root",
            "official",
            "--gnn-checkpoint",
            "model_best.pth",
            "--neurosed-checkpoint",
            "best_model.pt",
            "--output-dir",
            "out",
            *extra,
        ]
    )
    _resolve_profile_defaults(args)
    return args


def test_vrrw_cli_defaults_are_profile_derived() -> None:
    smoke = _parse_vrrw()
    assert (smoke.profile, smoke.parent_limit, smoke.m) == ("smoke", 64, 500)
    assert (smoke.alpha, smoke.theta, smoke.seed) == (1.0, 0.05, 13)
    full = _parse_vrrw("--profile", "full")
    assert (full.parent_limit, full.m) == (1448, 50000)
    assert (full.alpha, full.theta, full.seed) == (1.0, 0.05, 13)


def test_shell_cli_and_runtime_use_one_to_one_vrrw_names() -> None:
    wrapper = WRAPPERS[2].read_text(encoding="utf-8")
    cli = (
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py"
    ).read_text(encoding="utf-8")
    assert '--profile "$PROFILE"' in wrapper
    assert '--parent-limit "$VRRW_PARENT_LIMIT"' in wrapper
    assert '--m "$VRRW_M"' in wrapper
    assert 'parser.add_argument("--m", type=int)' in cli
    assert "m=args.m" in cli
    assert "m: int," in runtime
    assert "max_steps=int(m)" in runtime
    assert 'parser.add_argument("--max-steps"' not in cli


def test_invalid_vrrw_config_logs_actual_values_and_writes_failure_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "vrrw"
    command = [
        sys.executable,
        str(ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py"),
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--official-root",
        str(tmp_path / "official"),
        "--gnn-checkpoint",
        str(tmp_path / "model_best.pth"),
        "--neurosed-checkpoint",
        str(tmp_path / "best_model.pt"),
        "--output-dir",
        str(output_dir),
        "--profile",
        "smoke",
        "--parent-limit",
        "1448",
        "--m",
        "50000",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "[MUTAGENICITY_GCFEXPLAINER_VRRW_CONFIG]" in result.stdout
    assert "parent_limit=1448" in result.stdout
    assert "M=50000" in result.stdout
    assert "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" in result.stderr
    assert "expected_parent_limit=64" in result.stderr
    assert "expected_M=500_or_1000" in result.stderr
    payload = json.loads(
        (output_dir / "failure_summary.json").read_text(encoding="utf-8")
    )
    assert payload["stage"] == "vrrw_config"
    assert payload["profile"] == "smoke"
    assert payload["parent_limit"] == 1448
    assert payload["M"] == 50000
    assert payload["expected_parent_limit"] == 64
    assert payload["expected_M"] == [500, 1000]
    assert payload["model_training_performed"] is False
    assert payload["calibration_loaded"] is False
    assert payload["test_loaded"] is False
    assert (output_dir / "_RUN_FAILED.json").is_file()


def test_all_resume_reuses_completed_gnn_before_vrrw() -> None:
    all_text = WRAPPERS[4].read_text(encoding="utf-8")
    gnn_text = WRAPPERS[1].read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-true}"' in all_text
    assert 'RESUME="$RESUME"' in all_text
    assert "train_mutagenicity_gnn.sh" in all_text
    assert "[MUTAGENICITY_GCFEXPLAINER_GNN_REUSED]" in gnn_text
    assert all_text.index("train_mutagenicity_gnn.sh") < all_text.index(
        "reproduce_mutagenicity_vrrw.sh"
    )


def test_vrrw_stage_never_trains_models_or_reads_held_out_splits() -> None:
    wrapper = WRAPPERS[2].read_text(encoding="utf-8")
    cli = (
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py"
    ).read_text(encoding="utf-8")
    assert "train_mutagenicity_gnn" not in wrapper
    assert "gnn.py" not in wrapper
    assert "calibration_source_label" not in wrapper
    assert "calibration_target_label" not in wrapper
    assert "test_source_label" not in wrapper
    assert "test_target_label" not in wrapper
    assert "--forbid-calibration-test" in wrapper
    assert '"model_training_performed": False' in cli


def test_mutagenicity_wrappers_do_not_unset_any_proxy_variable() -> None:
    forbidden = (
        "unset http_proxy",
        "unset https_proxy",
        "unset HTTP_PROXY",
        "unset HTTPS_PROXY",
        "unset all_proxy",
        "unset ALL_PROXY",
    )
    for wrapper in WRAPPERS:
        text = wrapper.read_text(encoding="utf-8")
        assert all(command not in text for command in forbidden)


def test_summary_wrapper_preserves_native_rank_then_rf_filters() -> None:
    text = WRAPPERS[3].read_text(encoding="utf-8")
    summary_position = text.index("run_mutagenicity_summary.py")
    export_position = text.index("export_mutagenicity_fullgraph_candidates.py")
    audit_position = text.index("audit_mutagenicity_run.py")
    assert summary_position < export_position < audit_position
    summary_command = text[summary_position:export_position]
    assert "--parent-limit" not in summary_command
    assert 'PARENT_LIMIT="${PARENT_LIMIT' not in text
    assert "EXPECTED_PARENT_LIMIT=64" in text
    assert "EXPECTED_PARENT_LIMIT=1448" in text
    assert 'MINIMUM_NATIVE_EXPORT="${MINIMUM_NATIVE_EXPORT:-100}"' in text
    assert 'TOP_K="${TOP_K:-20}"' in text
    assert "mutagenicity_rf_v1/mutagenicity_rf_model.pkl" in text
    assert 'EXPORT_DIR="${EXPORT_DIR:-}"' in text
    assert "EXPORT_DIR must be provided explicitly" in text
    assert '_SMOKE_AUDIT_COMPLETE.json' in text
    assert 'EXPORT_SUCCESS_MARKER="$EXPORT_DIR/_RUN_COMPLETE.json"' in text


def test_export_profile_gates_are_explicit_and_do_not_rerank() -> None:
    runtime = (
        ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py"
    ).read_text(encoding="utf-8")
    cli = (
        ROOT
        / "scripts/baselines/gcfexplainer/export_mutagenicity_fullgraph_candidates.py"
    ).read_text(encoding="utf-8")
    assert "candidate_filter_audit.jsonl" in runtime
    assert "filter_summary.json" in runtime
    assert "smoke_interface_gate_passed" in runtime
    assert "candidate_yield_gate_passed" in runtime
    assert "rf_reranking_performed" in runtime
    assert "wnode_reranking_performed" in runtime
    assert "_SMOKE_AUDIT_COMPLETE.json" in runtime
    assert "[MUTAGENICITY_GCFEXPLAINER_CHEMICAL_CODEC_ERROR]" in runtime
    assert "[MUTAGENICITY_GCFEXPLAINER_EXPORT_SMOKE_AUDIT_OK]" in cli
    assert "no_rf_target_candidate" in cli
    official_root = ROOT / "baselines/gcfexplainer_official"
    for path in official_root.glob("*.py"):
        official_text = path.read_text(encoding="utf-8")
        assert "candidate_filter_audit.jsonl" not in official_text


def test_summary_and_all_wrappers_require_explicit_export_dir() -> None:
    summary = WRAPPERS[3].read_text(encoding="utf-8")
    all_wrapper = WRAPPERS[4].read_text(encoding="utf-8")
    for text in (summary, all_wrapper):
        assert 'EXPORT_DIR="${EXPORT_DIR:-}"' in text
        assert "EXPORT_DIR must be provided explicitly" in text
        assert 'EXPORT_DIR="${EXPORT_DIR:-$RUN_ROOT/export}"' not in text
    assert 'test -s "$EXPORT_DIR/_SMOKE_AUDIT_COMPLETE.json"' in all_wrapper
    assert 'test -s "$EXPORT_DIR/_RUN_COMPLETE.json"' in all_wrapper


def test_summary_cli_and_runtime_use_vrrw_manifest_parent_lineage() -> None:
    cli = (
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_summary.py"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--parent-limit"' not in cli
    assert "_SummaryConfigError" in cli
    assert "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" in cli
    assert "[MUTAGENICITY_GCFEXPLAINER_SUMMARY_CONFIG]" in runtime
    assert 'vrrw_manifest["parent_limit"]' in runtime
    assert "vrrw_manifest_generation_parent_ids" in runtime
    assert '"vrrw_selected_parent_count"' in runtime
    assert '"summary_parent_count"' in runtime
    assert '"generation_parent_ids_sha256"' in runtime
    assert '"vrrw_manifest_sha256"' in runtime
    assert '"counterfactuals_sha256"' in runtime


def test_summary_config_failure_writes_structured_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    error = gcf_runtime._summary_config_error(
        field="summary_parent_count",
        actual=1448,
        expected=64,
        count_source="dataset_full_universe",
    )

    def fail_summary(**_kwargs):
        raise error

    monkeypatch.setattr(summary_cli, "build_native_summary", fail_summary)
    output_dir = tmp_path / "summary"
    return_code = summary_cli.main(
        [
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--official-root",
            str(tmp_path / "official"),
            "--vrrw-dir",
            str(tmp_path / "vrrw"),
            "--gnn-checkpoint",
            str(tmp_path / "model_best.pth"),
            "--neurosed-checkpoint",
            str(tmp_path / "best_model.pt"),
            "--output-dir",
            str(output_dir),
            "--profile",
            "smoke",
        ]
    )
    assert return_code == 2
    captured = capsys.readouterr()
    assert "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" in captured.err
    assert "field=summary_parent_count" in captured.err
    assert "actual=1448" in captured.err
    assert "expected=64" in captured.err
    failure = json.loads(
        (output_dir / "failure_summary.json").read_text(encoding="utf-8")
    )
    assert failure["stage"] == "summary_config"
    assert failure["field"] == "summary_parent_count"
    assert failure["actual"] == 1448
    assert failure["expected"] == 64
    assert (output_dir / "resolved_config.json").is_file()
    assert (output_dir / "_RUN_FAILED.json").is_file()
    assert not (output_dir / "_RUN_COMPLETE.json").exists()


def test_all_wrapper_defaults_to_smoke_and_requires_full_opt_in() -> None:
    text = WRAPPERS[4].read_text(encoding="utf-8")
    assert 'PROFILE="${PROFILE:-smoke}"' in text
    assert '"${ALLOW_FULL:-false}" == "true"' in text
    assert "full requires ALLOW_FULL=true" in text
    assert "prepare_mutagenicity_dataset.sh" in text
    assert "train_mutagenicity_gnn.sh" in text
    assert "reproduce_mutagenicity_vrrw.sh" in text
    assert "reproduce_mutagenicity_summary.sh" in text
    assert "[MUTAGENICITY_GCFEXPLAINER_SMOKE_OK]" in text


@pytest.mark.parametrize(
    "script",
    (
        ROOT / "scripts/baselines/gcfexplainer/prepare_mutagenicity_dataset.py",
        ROOT / "scripts/baselines/gcfexplainer/probe_mutagenicity_codec.py",
        ROOT / "scripts/baselines/gcfexplainer/train_mutagenicity_gnn.py",
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py",
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_summary.py",
        ROOT / "scripts/baselines/gcfexplainer/export_mutagenicity_fullgraph_candidates.py",
        ROOT / "scripts/baselines/gcfexplainer/audit_mutagenicity_run.py",
    ),
)
def test_python_entrypoints_forbid_calibration_test(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert "--forbid-calibration-test" in text


def test_runtime_reuses_official_algorithm_entrypoints() -> None:
    text = (ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py").read_text(encoding="utf-8")
    assert "modules[\"gnn\"].GNN" in text
    assert "vrrw.counterfactual_summary_with_randomwalk" in text
    assert "greedy_counterfactual_summary_from_covering_sets" in text
    assert "native_rank_reordered\": False" in text


def test_mutagenicity_runtime_scopes_alpha_endpoint_patch() -> None:
    runtime = (
        ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py"
    ).read_text(encoding="utf-8")
    official = (
        ROOT / "baselines/gcfexplainer_official/vrrw.py"
    ).read_text(encoding="utf-8")
    assert "vrrw_alpha_endpoint_none_safe_v1" in runtime
    assert "official_compatibility_patches" in runtime
    assert "alpha_endpoint_branch" in runtime
    assert "[GCFEXPLAINER_OFFICIAL_COMPAT_PATCH]" in runtime
    assert "with _official_vrrw_alpha_endpoint_patch(vrrw):" in runtime
    assert "vrrw.calculate_importance = original_calculate_importance" in runtime
    assert "vrrw_alpha_endpoint_none_safe_v1" not in official
