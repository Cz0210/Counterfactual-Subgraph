from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.baselines.globalgce.build_bace_train_pool import (
    build_parser as build_legacy_rf_parser,
)
from scripts.autodl.run_bace_baseline_gnn_route import (
    build_parser as build_route_parser,
    main as route_main,
)
from scripts.autodl.run_four_gpu_recovery_controller import (
    initialize_controller_state,
    load_controller_manifest,
)
from src.baselines.bace_gnn_baseline_generic_adapter import (
    B11_SHARD_PRIORITY,
    GENERIC_FRAGMENT_SCHEMA,
    build_bace_baseline_generic_controller_fragment,
)
from src.baselines.bace_gnn_baseline_tasks import (
    build_bace_baseline_controller_fragment,
)


def _paths(tmp_path: Path) -> dict[str, Path]:
    fixture = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    root = Path("/persistent") / fixture
    return {
        "python": root / "env/bin/python",
        "project_root": root / "project",
        "output_root": root / "outputs/bace-baselines",
        "gnn_checkpoint": root / "gine",
        "dataset_dir": root / "dataset",
        "calibration_split": root / "bace_calibration.csv",
        "test_split": root / "bace_test.csv",
        "molclr_root": root / "molclr",
        "molclr_checkpoint": root / "molclr.pt",
        "neurosed_checkpoint": root / "neurosed.pt",
        "official_root": root / "official",
        "neurosed_manifest": root / "neurosed.json",
        "globalgce_source_manifest": root / "source_graph_manifest.jsonl",
        "globalgce_native_train_csv": root / "train.csv",
    }


def _generic(tmp_path: Path, method: str) -> dict:
    return build_bace_baseline_generic_controller_fragment(
        method=method, **_paths(tmp_path)
    )


def _production_manifest(tmp_path: Path, tasks: list[dict]) -> Path:
    path = tmp_path / "controller.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller_id": "bace-baseline-generic-adapter-test",
                "paper_frozen": True,
                "runtime": {
                    "max_gpus": 4,
                    "stable_idle_seconds": 60,
                    "sample_interval_seconds": 5,
                    "poll_seconds": 60,
                    "max_transient_retries": 1,
                },
                "resource_gates": {},
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_native_fragment_contract_is_not_changed(tmp_path: Path) -> None:
    native = build_bace_baseline_controller_fragment(
        method="GCFExplainer", **_paths(tmp_path)
    )
    first = native["tasks"][0]
    assert native["schema_version"] == "bace_baseline_controller_fragment_v1"
    assert "task_id" in first and "id" not in first
    assert "argv" in first and "command" not in first
    assert first["resource"] == {"kind": "cpu", "gpus": 0}


def test_generic_adapter_binds_attempt_roots_and_every_dependency(
    tmp_path: Path,
) -> None:
    fragment = _generic(tmp_path, "GCFExplainer")
    assert fragment["schema_version"] == GENERIC_FRAGMENT_SCHEMA
    tasks = {task["id"]: task for task in fragment["tasks"]}
    assert tasks["bace_gcfexplainer_train_vrrw"]["priority"] < B11_SHARD_PRIORITY
    assert tasks["bace_gcfexplainer_train_summary"]["priority"] < B11_SHARD_PRIORITY
    assert (
        tasks["bace_gcfexplainer_train_vrrw"]["priority"]
        < tasks["bace_gcfexplainer_train_summary"]["priority"]
        < tasks["bace_gcfexplainer_train_candidates"]["priority"]
        < B11_SHARD_PRIORITY
    )
    for task in tasks.values():
        assert task["runner_dataset"] == "bace-baseline-gcfexplainer"
        assert task["runner_dataset"] != "bace"
        assert task["resource"] in {"cpu", "gpu"}
        assert task["expected_output"].endswith("attempt-{attempt}")
        assert task["required_output_files"]
        assert task["required_log_marker"]
        serialized = json.dumps(task, sort_keys=True)
        for dependency in task["depends_on"]:
            assert f"{{dep_{dependency}_output}}" in serialized
        assert "{task_output}" in task["command"]
    summary = tasks["bace_gcfexplainer_train_summary"]
    assert (
        "{dep_bace_gcfexplainer_train_vrrw_output}" in summary["command"]
    )
    test_shard = tasks["bace_gcfexplainer_test_shard_0"]
    assert test_shard["selector_parameters_frozen"] is True
    assert test_shard["read_only_test"] is True


def test_all_three_native_routes_pass_production_loader(
    tmp_path: Path,
) -> None:
    gcf = _generic(tmp_path / "gcf", "GCFExplainer")
    comrecgc = _generic(tmp_path / "comrecgc", "ComRecGC")
    globalgce = _generic(tmp_path / "globalgce", "GlobalGCE")
    tasks = [*gcf["tasks"], *comrecgc["tasks"], *globalgce["tasks"]]
    manifest = load_controller_manifest(_production_manifest(tmp_path, tasks))

    by_id = manifest.by_id
    assert len(by_id) == len(tasks)
    assert by_id["bace_gcfexplainer_train_vrrw"].priority < B11_SHARD_PRIORITY
    assert by_id["bace_comrecgc_train_generation"].priority < B11_SHARD_PRIORITY
    assert (
        by_id["bace_comrecgc_train_generation"].priority
        < by_id["bace_comrecgc_train_common_recourse"].priority
        < by_id["bace_comrecgc_train_candidates"].priority
        < B11_SHARD_PRIORITY
    )
    bridge = by_id["bace_globalgce_bridge_smoke"]
    assert bridge.resource == "gpu"
    assert bridge.depends_on == ("bace_globalgce_preflight",)
    global_train = by_id["bace_globalgce_train_candidates"]
    assert global_train.resource == "gpu"
    assert global_train.depends_on == ("bace_globalgce_bridge_smoke",)
    root, states = initialize_controller_state(
        type("Layout", (), {"control_root": tmp_path / "control"})(), manifest
    )
    assert root.is_dir()
    assert states[bridge.task_id]["state"] == "NOT_STARTED"


def test_generated_globalgce_entrypoint_commands_parse_exactly(tmp_path: Path) -> None:
    fragment = _generic(tmp_path, "GlobalGCE")
    parser = build_route_parser()
    parsed_stages: set[str] = set()
    for task in fragment["tasks"]:
        command = task["command"]
        if not str(command[1]).endswith("run_bace_baseline_gnn_route.py"):
            continue
        parsed = parser.parse_args(command[2:])
        parsed_stages.add(str(parsed.stage))
    assert parsed_stages == {
        "preflight",
        "globalgce-bridge-smoke",
        "globalgce-train-rules",
        "verify-shard",
        "merge",
        "select",
        "freeze",
    }


def test_globalgce_exact_topk_flag_is_frozen_into_train_task(tmp_path: Path) -> None:
    fragment = build_bace_baseline_generic_controller_fragment(
        method="GlobalGCE",
        globalgce_exact_top_k_pruning=True,
        **_paths(tmp_path),
    )
    task = next(
        row
        for row in fragment["tasks"]
        if row["id"] == "bace_globalgce_train_candidates"
    )
    assert "--gspan-exact-top-k-pruning" in task["command"]
    assert "--gnn-checkpoint" in task["command"]
    assert "--teacher-path" not in task["command"]
    assert str(task["command"][1]).endswith(
        "scripts/autodl/run_bace_baseline_gnn_route.py"
    )
    parsed = build_route_parser().parse_args(task["command"][2:])
    assert parsed.gspan_exact_top_k_pruning is True
    assert parsed.gnn_checkpoint == str(_paths(tmp_path)["gnn_checkpoint"])


def test_exact_topk_is_not_exposed_by_legacy_rf_cli_or_slurm() -> None:
    with pytest.raises(SystemExit):
        build_legacy_rf_parser().parse_args(
            [
                "--train-csv", "train.csv",
                "--native-train-csv", "native.csv",
                "--teacher-path", "teacher.pkl",
                "--official-root", "official",
                "--output-dir", "output",
                "--gspan-exact-top-k-pruning",
            ]
        )
    project_root = Path(__file__).resolve().parents[2]
    legacy_slurm = (
        project_root / "scripts/slurm/build_bace_train_pool.sh"
    ).read_text(encoding="utf-8")
    exact_docs = (
        project_root / "docs/AUTODL_BACE_GLOBALGCE_EXACT_TOPK.md"
    ).read_text(encoding="utf-8")
    generic_slurm = (
        project_root / "scripts/slurm/run_bace_baseline_gnn_route.sh"
    ).read_text(encoding="utf-8")
    assert "GSPAN_EXACT_TOP_K_PRUNING" not in legacy_slurm
    assert "build_bace_train_pool.py \\\n" not in exact_docs
    assert "globalgce-train-rules" in exact_docs
    assert "--gnn-checkpoint" in exact_docs
    assert "globalgce-train-rules" in generic_slurm


def test_generic_fragment_cli_writes_fresh_composer_input(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    destination = tmp_path / "fragments/globalgce.json"
    argv = [
        "generic-task-fragment",
        "--method",
        "GlobalGCE",
        "--python",
        str(paths["python"]),
        "--project-root",
        str(paths["project_root"]),
        "--output-dir",
        str(paths["output_root"]),
        "--gnn-checkpoint",
        str(paths["gnn_checkpoint"]),
        "--dataset-dir",
        str(paths["dataset_dir"]),
        "--calibration-split",
        str(paths["calibration_split"]),
        "--test-split",
        str(paths["test_split"]),
        "--molclr-root",
        str(paths["molclr_root"]),
        "--molclr-checkpoint",
        str(paths["molclr_checkpoint"]),
        "--neurosed-checkpoint",
        str(paths["neurosed_checkpoint"]),
        "--official-root",
        str(paths["official_root"]),
        "--neurosed-manifest",
        str(paths["neurosed_manifest"]),
        "--globalgce-source-manifest",
        str(paths["globalgce_source_manifest"]),
        "--globalgce-native-train-csv",
        str(paths["globalgce_native_train_csv"]),
        "--fragment-output",
        str(destination),
    ]
    assert route_main(argv) == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == GENERIC_FRAGMENT_SCHEMA
    by_id = {row["id"]: row for row in payload["tasks"]}
    assert by_id["bace_globalgce_preflight"]["command"] is not None
    assert by_id["bace_globalgce_preflight"]["resource"] == "cpu"
    assert by_id["bace_globalgce_bridge_smoke"]["resource"] == "gpu"
    assert by_id["bace_globalgce_train_candidates"]["resource"] == "gpu"
