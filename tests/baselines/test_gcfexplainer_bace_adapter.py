from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.gcfexplainer_bace_adapter import (
    BACE_FEATURE_ATOMIC_NUMBERS,
    BACE_FEATURE_KEEP_INDICES,
    EXPECTED_GENERATION_SOURCE_ROWS,
    EXPECTED_MODEL_TRAIN_ROWS,
    EXPECTED_MODEL_VAL_ROWS,
    MUTAGENICITY_FEATURE_ATOMIC_NUMBERS,
    adapt_bace_neurosed_checkpoint,
    validate_bace_gnn_profile,
    validate_bace_vrrw_profile,
)


ROOT = Path(__file__).resolve().parents[2]


def test_bace_full_profiles_are_frozen() -> None:
    assert EXPECTED_MODEL_TRAIN_ROWS == 869
    assert EXPECTED_MODEL_VAL_ROWS == 162
    assert EXPECTED_GENERATION_SOURCE_ROWS == 360
    validate_bace_gnn_profile("full", epochs=1000, train_rows=869, val_rows=162)
    validate_bace_vrrw_profile(
        "full",
        parent_limit=360,
        m=50000,
        alpha=1.0,
        theta=0.05,
        seed=13,
    )


@pytest.mark.parametrize(
    ("field", "kwargs"),
    (
        ("parent_limit", {"parent_limit": 64}),
        ("M", {"m": 500}),
        ("alpha", {"alpha": 0.5}),
        ("theta", {"theta": 0.1}),
        ("seed", {"seed": 0}),
    ),
)
def test_bace_full_vrrw_rejects_protocol_drift(
    field: str, kwargs: dict[str, int | float]
) -> None:
    values: dict[str, int | float] = {
        "parent_limit": 360,
        "m": 50000,
        "alpha": 1.0,
        "theta": 0.05,
        "seed": 13,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=field):
        validate_bace_vrrw_profile("full", **values)


def test_bace_neurosed_projection_only_removes_phosphorus_channel(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    manifest_path = tmp_path / "projection.json"
    input_weight = torch.arange(40, dtype=torch.float32).reshape(4, 10)
    other_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(
        {
            "embed_model.pre.weight": input_weight,
            "embed_model.post.weight": other_weight,
        },
        source,
    )

    manifest = adapt_bace_neurosed_checkpoint(
        source_checkpoint=source,
        output_checkpoint=target,
        manifest_path=manifest_path,
    )
    projected = torch.load(target, map_location="cpu", weights_only=False)

    assert MUTAGENICITY_FEATURE_ATOMIC_NUMBERS == (
        6,
        8,
        17,
        1,
        7,
        9,
        35,
        16,
        15,
        53,
    )
    assert BACE_FEATURE_ATOMIC_NUMBERS == (6, 8, 17, 1, 7, 9, 35, 16, 53)
    assert BACE_FEATURE_KEEP_INDICES == (0, 1, 2, 3, 4, 5, 6, 7, 9)
    assert torch.equal(
        projected["embed_model.pre.weight"],
        input_weight[:, list(BACE_FEATURE_KEEP_INDICES)],
    )
    assert torch.equal(projected["embed_model.post.weight"], other_weight)
    assert manifest["removed_atomic_numbers"] == [15]
    assert manifest["training_performed"] is False
    assert manifest["calibration_loaded"] is False
    assert manifest["test_loaded"] is False
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_bace_adapter_does_not_modify_official_source_tree() -> None:
    official = ROOT / "baselines/gcfexplainer_official"
    for name in ("data.py", "gnn.py", "vrrw.py", "summary.py", "distance.py"):
        assert (official / name).is_file()
    production = (
        ROOT / "src/baselines/gcfexplainer_bace_adapter.py"
    ).read_text(encoding="utf-8") + (
        ROOT / "src/baselines/gcfexplainer_bace_runtime.py"
    ).read_text(encoding="utf-8")
    assert "write_text" not in production
    assert "gcfexplainer_official/vrrw.py" not in production
