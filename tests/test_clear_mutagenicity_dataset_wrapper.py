from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts/baselines/clear/prepare_mutagenicity_dataset.py"
PROBE = ROOT / "scripts/baselines/clear/probe_mutagenicity_codec.py"
PATCH = ROOT / "patches/clear_official/005_support_mutagenicity_dataset.patch"
APPLY = ROOT / "scripts/baselines/clear/apply_clear_patches.sh"
COMMON = ROOT / "scripts/baselines/clear/common.sh"
RUN = ROOT / "scripts/baselines/clear/run_clear.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_phase_a_cli_help_and_probe_default() -> None:
    for script in (PREPARE, PROBE):
        result = subprocess.run(
            ["python3", str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--forbid-calibration-test" in result.stdout
    assert "default: 64" in subprocess.run(
        ["python3", str(PROBE), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_prepare_uses_only_strict_train_and_validation_inputs() -> None:
    text = _text(PREPARE)
    assert "train_source_label1_teacher_correct.csv" in text
    assert "train_target_label0_teacher_correct.csv" in text
    assert "val_source_label1_teacher_correct.csv" in text
    assert "val_target_label0_teacher_correct.csv" in text
    assert "mutagenicity_full.pickle" not in text
    assert "readiness_v1" not in text
    assert "max_num_nodes=30" not in text


def test_official_patch_is_minimal_binary_adjacency_support() -> None:
    text = _text(PATCH)
    assert "CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET" in text
    assert "mutagenicity_full.pickle" in text
    assert "mutagenicity_datasplit.pickle" in text
    assert "choices=['community', 'ogbg_molhiv', 'aids', 'mutagenicity'" in text
    assert "elif self.dataset == 'mutagenicity'" in text
    assert "x = x.clone()" in text
    lowered = text.lower()
    for forbidden in (
        "bond_type_decoder",
        "decoder_edge",
        "edge_class_decoder",
        "multichannel",
        "multi_channel",
    ):
        assert forbidden not in lowered


def test_apply_patch_registration_is_marker_idempotent() -> None:
    text = _text(APPLY)
    marker_check = text.index('grep -q "${patch_marker}" "${marker_file}"')
    apply_call = text.index('git -C "${CLEAR_DIR}" apply "${patch_file}"')
    assert marker_check < apply_call
    assert "005_support_mutagenicity_dataset.patch" in text
    assert "CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET" in text


def test_common_and_runner_support_mutagenicity_without_aids_regression() -> None:
    common = _text(COMMON)
    run = _text(RUN)
    assert "mutagenicity)" in common
    assert "mutagenicity_full.pickle" in common
    assert "mutagenicity_datasplit.pickle" in common
    assert "aids_full.pickle" in common
    assert "aids_datasplit.pickle" in common
    assert "community | ogbg_molhiv | aids | mutagenicity | imdb_m" in run
    assert 'CLEAR_BATCH_SIZE=8' in run
    assert 'CLEAR_NUM_WORKERS="${CLEAR_NUM_WORKERS:-0}"' in run
    assert '--num_workers "${CLEAR_NUM_WORKERS}"' in run
    assert "max_num_nodes=30" not in run


def test_shell_scripts_have_valid_syntax() -> None:
    for script in (COMMON, RUN, APPLY):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_official_decoder_remains_binary_adjacency() -> None:
    models = _text(ROOT / "baselines/clear_official/src/models.py")
    assert "self.decoder_a" in models
    assert ".view(-1, self.max_num_nodes, self.max_num_nodes)" in models
    assert (
        "nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), "
        "nn.Sigmoid()"
    ) in models
    assert "bond_type" not in models.lower()
    assert "edge_class" not in models.lower()


def test_patch_applies_after_existing_project_patches() -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT / "baselines/clear_official"),
            "apply",
            "--check",
            str(PATCH),
        ],
        check=True,
    )
