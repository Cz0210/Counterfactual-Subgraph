"""BACE facade over the established project GlobalGCE molecular adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .globalgce_mutagenicity_adapter import (
    NativeGeneratorProtocol,
    OfficialGlobalGCEMutagenicityGenerator,
    PoolBuildConfig,
    TeacherProtocol,
    audit_mutagenicity_train_pool,
    build_mutagenicity_train_pool,
)


DATASET_NAME = "BACE"
EXPECTED_TRAIN_SOURCE_COUNT = 360
EXPECTED_NATIVE_TRAIN_COUNT = 869


class OfficialGlobalGCEBACEGenerator(OfficialGlobalGCEMutagenicityGenerator):
    """Run unchanged official GlobalGCE components with BACE dataset identity."""

    def __init__(
        self,
        official_root: str | Path,
        *,
        native_train_csv: str | Path,
        min_freq: int,
    ) -> None:
        super().__init__(
            official_root,
            native_train_csv=native_train_csv,
            dataset_name=DATASET_NAME,
            min_freq=int(min_freq),
        )


def build_bace_train_pool(
    *,
    train_csv: str | Path,
    teacher_path: str | Path,
    official_root: str | Path,
    output_dir: str | Path,
    teacher: TeacherProtocol,
    generator: NativeGeneratorProtocol,
    config: PoolBuildConfig | None = None,
) -> dict[str, Any]:
    return build_mutagenicity_train_pool(
        train_csv=train_csv,
        teacher_path=teacher_path,
        official_root=official_root,
        output_dir=output_dir,
        teacher=teacher,
        generator=generator,
        config=config,
        dataset_name=DATASET_NAME,
    )


def audit_bace_train_pool(
    run_dir: str | Path,
    *,
    train_csv: str | Path,
    expected_parent_count: int = EXPECTED_TRAIN_SOURCE_COUNT,
    expected_input_train_count: int | None = EXPECTED_TRAIN_SOURCE_COUNT,
    require_complete: bool = True,
) -> dict[str, Any]:
    return audit_mutagenicity_train_pool(
        run_dir,
        train_csv=train_csv,
        expected_parent_count=expected_parent_count,
        expected_input_train_count=expected_input_train_count,
        require_target_label_zero=True,
        require_unique_universe=True,
        forbid_calibration_test=True,
        require_complete=require_complete,
        dataset_name=DATASET_NAME,
    )


__all__ = [
    "DATASET_NAME",
    "EXPECTED_NATIVE_TRAIN_COUNT",
    "EXPECTED_TRAIN_SOURCE_COUNT",
    "OfficialGlobalGCEBACEGenerator",
    "audit_bace_train_pool",
    "build_bace_train_pool",
]
