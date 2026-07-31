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
ADAPTER = ROOT / "src/baselines/clear_mutagenicity_adapter.py"


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


def test_atom_sidecar_v2_restores_source_hydrogen_attributes() -> None:
    text = _text(ADAPTER)
    for required in (
        "clear_mutagenicity_atom_sidecar_v2",
        '"num_explicit_hs"',
        '"num_implicit_hs"',
        '"no_implicit"',
        '"chiral_tag"',
        "SetNumExplicitHs",
        "SetNoImplicit",
        "SetChiralTag",
        "UpdatePropertyCache(strict=False)",
        "ambiguous_generated_atom_hydrogen_state",
    ):
        assert required in text
    assert "SetNumImplicitHs" not in text


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


def test_patch_applies_after_existing_project_patches(
    tmp_path: Path,
) -> None:
    official_root = ROOT / "baselines/clear_official"

    def git_output(repository: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    real_state_before = (
        git_output(official_root, "rev-parse", "HEAD"),
        git_output(official_root, "status", "--porcelain=v1"),
    )
    temporary_checkout = tmp_path / "clear_official"
    subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            str(official_root),
            str(temporary_checkout),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert git_output(temporary_checkout, "status", "--porcelain=v1") == ""
    assert git_output(temporary_checkout, "rev-parse", "HEAD") == (
        real_state_before[0]
    )

    patch_names = (
        "001_save_cfe_checkpoints.patch",
        "002_export_test_counterfactuals.patch",
        "003_support_aids_dataset.patch",
        "004_aids_weighted_graphpred.patch",
        "005_support_mutagenicity_dataset.patch",
    )
    for patch_name in patch_names:
        patch_path = ROOT / "patches/clear_official" / patch_name
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary_checkout),
                "apply",
                "--check",
                str(patch_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary_checkout),
                "apply",
                str(patch_path),
            ],
            check=True,
        )

    assert "CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET" in _text(
        temporary_checkout / "src/main.py"
    )
    real_state_after = (
        git_output(official_root, "rev-parse", "HEAD"),
        git_output(official_root, "status", "--porcelain=v1"),
    )
    assert real_state_after == real_state_before
