"""Generate dependency-complete controller tasks for BACE native baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.baselines.bace_gnn_baseline_contracts import baseline_spec
from src.baselines.globalgce_min_freq import BACE_PRIMARY_MIN_FREQ


NUM_SHARDS = 4


def _absolute(value: str | Path, *, field: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute: {path}")
    return str(path.resolve(strict=False))


def build_bace_baseline_controller_fragment(
    *,
    method: str,
    python: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    gnn_checkpoint: str | Path,
    dataset_dir: str | Path,
    calibration_split: str | Path,
    test_split: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
    official_root: str | Path | None = None,
    neurosed_manifest: str | Path | None = None,
    globalgce_source_manifest: str | Path | None = None,
    globalgce_native_train_csv: str | Path | None = None,
    globalgce_gspan_adoption_proof: str | Path | None = None,
    globalgce_exact_top_k_pruning: bool = False,
    omp_threads: int = 4,
) -> dict[str, Any]:
    """Return tasks ready to splice into a dependency-aware controller manifest.

    GPU assignment is deliberately absent from the environment.  The generic
    controller owns ``CUDA_VISIBLE_DEVICES`` and the UUID lock; each foreground
    command sees its assigned physical card as ``cuda:0``.
    """

    spec = baseline_spec(method)
    root = _absolute(output_root, field="output_root")
    project = _absolute(project_root, field="project_root")
    py = _absolute(python, field="python")
    checkpoint = _absolute(gnn_checkpoint, field="gnn_checkpoint")
    dataset = _absolute(dataset_dir, field="dataset_dir")
    calibration = _absolute(calibration_split, field="calibration_split")
    test = _absolute(test_split, field="test_split")
    molclr_project = _absolute(molclr_root, field="molclr_root")
    molclr_ckpt = _absolute(molclr_checkpoint, field="molclr_checkpoint")
    neurosed = _absolute(neurosed_checkpoint, field="neurosed_checkpoint")
    prefix = f"bace_{spec.method_id}"
    common_env = {
        "PYTHONPATH": project,
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(int(omp_threads)),
        "MKL_NUM_THREADS": str(int(omp_threads)),
        "RUN_TASTEMOLNET": "0",
        "PYTHONHASHSEED": "0",
    }

    script = f"{project}/scripts/autodl/run_bace_baseline_gnn_route.py"
    tasks: list[dict[str, Any]] = []

    def add(
        task_id: str,
        *,
        resource: str,
        argv: list[str],
        output: str,
        markers: list[str],
        dependencies: list[str],
        inputs: list[str],
    ) -> None:
        tasks.append(
            {
                "task_id": task_id,
                "dataset": "bace",
                "method": spec.method,
                "resource": {"kind": resource, "gpus": 1 if resource == "gpu" else 0},
                "argv": argv,
                "env": dict(common_env),
                "controller_injected_env": (
                    ["CUDA_VISIBLE_DEVICES"] if resource == "gpu" else []
                ),
                "inputs": inputs,
                "output_root": output,
                "fresh_output_required": True,
                "required_markers": markers,
                "dependencies": dependencies,
                "max_transient_retries": 1,
                "max_oom_retries": 1 if resource == "gpu" else 0,
                "retry_policy": (
                    "fresh_attempt_root" if resource == "gpu" else "no_automatic_retry"
                ),
            }
        )

    preflight_id = f"{prefix}_preflight"
    preflight_out = f"{root}/preflight"
    preflight_argv = [
        py,
        script,
        "preflight",
        "--method",
        spec.method,
        "--gnn-checkpoint",
        checkpoint,
        "--output-dir",
        preflight_out,
    ]
    preflight_inputs = [checkpoint]
    preflight_markers = ["READY"]
    comrecgc_official: str | None = None
    if spec.method_id == "globalgce":
        if official_root is None:
            raise ValueError("GlobalGCE task fragment requires official_root")
        globalgce_official = _absolute(official_root, field="official_root")
        preflight_argv.extend(["--official-root", globalgce_official])
        preflight_inputs.append(globalgce_official)
        preflight_markers = ["NATIVE_ACTION_READY", "READY"]
    elif spec.method_id == "comrecgc":
        comrecgc_official = _absolute(
            official_root or f"{project}/external/COMRECGC",
            field="official_root",
        )
        preflight_argv.extend(["--official-root", comrecgc_official])
        preflight_inputs.append(comrecgc_official)
    add(
        preflight_id,
        resource="cpu",
        argv=preflight_argv,
        output=preflight_out,
        markers=preflight_markers,
        dependencies=[],
        inputs=preflight_inputs,
    )

    if not spec.native_route_available:
        return {
            "schema_version": "bace_baseline_controller_fragment_v1",
            "dataset": "bace",
            "method": spec.method,
            "method_id": spec.method_id,
            "root_task_ids": [preflight_id],
            "terminal_task_ids": [f"{prefix}_terminal"],
            "tasks": tasks,
            "static_terminal": {
                "task_id": f"{prefix}_terminal",
                "state": "BLOCKED_CODE",
                "blocker_code": spec.blocker_code,
                "reason": spec.blocker_reason,
                "resource": {"kind": "none", "gpus": 0},
                "dependencies": [preflight_id],
                "argv": [],
                "output_root": None,
                "required_markers": [],
            },
        }

    if spec.method_id == "globalgce":
        if (
            official_root is None
            or globalgce_source_manifest is None
            or globalgce_native_train_csv is None
        ):
            raise ValueError(
                "GlobalGCE route requires official_root, source_manifest, and native_train_csv"
            )
        official = _absolute(official_root, field="official_root")
        source_manifest = _absolute(
            globalgce_source_manifest, field="globalgce_source_manifest"
        )
        native_train_csv = _absolute(
            globalgce_native_train_csv, field="globalgce_native_train_csv"
        )
        adoption_proof = (
            _absolute(
                globalgce_gspan_adoption_proof,
                field="globalgce_gspan_adoption_proof",
            )
            if globalgce_gspan_adoption_proof is not None
            else None
        )
        bridge_id = f"{prefix}_bridge_smoke"
        bridge_out = f"{root}/bridge_smoke"
        add(
            bridge_id,
            resource="gpu",
            argv=[
                py,
                script,
                "globalgce-bridge-smoke",
                "--method",
                spec.method,
                "--gnn-checkpoint",
                checkpoint,
                "--output-dir",
                bridge_out,
                "--parent-smiles",
                "CCO",
                "--atom-symbol",
                "C",
                "--atom-symbol",
                "O",
                "--atom-symbol",
                "Cl",
                "--atom-symbol",
                "H",
                "--atom-symbol",
                "N",
                "--atom-symbol",
                "F",
                "--atom-symbol",
                "Br",
                "--atom-symbol",
                "S",
                "--atom-symbol",
                "I",
                "--device",
                "cuda:0",
            ],
            output=bridge_out,
            markers=["PASS", "BRIDGE_PASS"],
            dependencies=[preflight_id],
            inputs=[checkpoint, official],
        )
        candidate_id = f"{prefix}_train_candidates"
        candidate_out = f"{root}/train_candidates"
        add(
            candidate_id,
            resource="gpu",
            argv=[
                py,
                script,
                "globalgce-train-rules",
                "--method",
                spec.method,
                "--gnn-checkpoint",
                checkpoint,
                "--source-manifest",
                source_manifest,
                "--native-train-csv",
                native_train_csv,
                "--official-root",
                official,
                "--output-dir",
                candidate_out,
                "--expected-parent-count",
                "360",
                "--seed",
                "13",
                "--min-freq",
                str(BACE_PRIMARY_MIN_FREQ),
                "--epochs",
                "100",
                "--top-k-native",
                "20",
                "--device",
                "cuda:0",
                "--resume",
            ],
            output=candidate_out,
            markers=["PASS"],
            dependencies=[bridge_id],
            inputs=[source_manifest, native_train_csv, official, checkpoint],
        )
        if adoption_proof is not None:
            tasks[-1]["argv"].extend(
                ["--gspan-adoption-proof", adoption_proof]
            )
            tasks[-1]["inputs"].append(adoption_proof)
        if globalgce_exact_top_k_pruning:
            tasks[-1]["argv"].append("--gspan-exact-top-k-pruning")
            tasks[-1]["env"]["GLOBALGCE_EXACT_TOP_K_PRUNING"] = "1"
        tasks[-1]["resume_argv"] = list(tasks[-1]["argv"])
        tasks[-1]["retry_policy"] = "resume_same_root_from_verified_checkpoint"
    elif spec.method_id == "gcfexplainer":
        if official_root is None or neurosed_manifest is None:
            raise ValueError("GCFExplainer task fragment requires official_root and neurosed_manifest")
        official = _absolute(official_root, field="official_root")
        neurosed_card = _absolute(neurosed_manifest, field="neurosed_manifest")
        generation_id = f"{prefix}_train_vrrw"
        generation_out = f"{root}/train_vrrw"
        add(
            generation_id,
            resource="gpu",
            argv=[
                py,
                f"{project}/scripts/baselines/gcfexplainer/run_bace_vrrw.py",
                "--dataset-dir",
                dataset,
                "--official-root",
                official,
                "--gnn-checkpoint",
                checkpoint,
                "--neurosed-checkpoint",
                neurosed,
                "--neurosed-manifest",
                neurosed_card,
                "--output-dir",
                generation_out,
                "--profile",
                "full",
                "--parent-limit",
                "360",
                "--m",
                "50000",
                "--device1",
                "cuda:0",
                "--device2",
                "cuda:0",
            ],
            output=generation_out,
            markers=["_RUN_COMPLETE.json"],
            dependencies=[preflight_id],
            inputs=[dataset, official, checkpoint, neurosed, neurosed_card],
        )
        summary_id = f"{prefix}_train_summary"
        summary_out = f"{root}/train_summary"
        add(
            summary_id,
            resource="gpu",
            argv=[
                py,
                f"{project}/scripts/baselines/gcfexplainer/run_bace_summary.py",
                "--dataset-dir",
                dataset,
                "--official-root",
                official,
                "--vrrw-dir",
                generation_out,
                "--gnn-checkpoint",
                checkpoint,
                "--neurosed-checkpoint",
                neurosed,
                "--output-dir",
                summary_out,
                "--profile",
                "full",
                "--native-candidate-limit",
                "0",
                "--device",
                "cuda:0",
            ],
            output=summary_out,
            markers=["_RUN_COMPLETE.json"],
            dependencies=[generation_id],
            inputs=[dataset, official, generation_out, checkpoint, neurosed],
        )
        native_dependency = summary_id
        candidate_id = f"{prefix}_train_candidates"
        candidate_out = f"{root}/train_candidates"
        add(
            candidate_id,
            resource="gpu",
            argv=[
                py,
                script,
                "gcf-export",
                "--method",
                spec.method,
                "--dataset-dir",
                dataset,
                "--summary-dir",
                summary_out,
                "--gnn-checkpoint",
                checkpoint,
                "--output-dir",
                candidate_out,
                "--profile",
                "full",
                "--parent-limit",
                "360",
                "--scan-limit",
                "0",
                "--device",
                "cuda:0",
            ],
            output=candidate_out,
            markers=["PASS"],
            dependencies=[native_dependency],
            inputs=[dataset, summary_out, checkpoint],
        )
    else:
        if comrecgc_official is None:  # pragma: no cover - guarded by method specs
            raise AssertionError("ComRecGC official checkout was not resolved")
        official = comrecgc_official
        generation_id = f"{prefix}_train_generation"
        generation_out = f"{root}/train_generation"
        add(
            generation_id,
            resource="gpu",
            argv=[
                py,
                f"{project}/scripts/baselines/comrecgc/run_generation.py",
                "--route",
                "project",
                "--dataset",
                "bace",
                "--mode",
                "full",
                "--project-root",
                project,
                "--upstream-root",
                official,
                "--dataset-dir",
                dataset,
                "--gnn-checkpoint",
                checkpoint,
                "--distance-checkpoint",
                neurosed,
                "--parent-limit",
                "360",
                "--device",
                "cuda:0",
                "--output-dir",
                generation_out,
                "--checkpoint-root",
                f"{root}/checkpoints",
                "--checkpoint-mirror-root",
                f"{root}/checkpoint_mirror",
                "--trace-output-dir",
                f"{root}/trace",
                "--graph-state-dir",
                f"{root}/graph_state",
                "--storage-guard-root",
                str(Path(root).parent),
            ],
            output=generation_out,
            markers=["_RUN_COMPLETE.json"],
            dependencies=[preflight_id],
            inputs=[project, official, dataset, checkpoint, neurosed],
        )
        tasks[-1]["resume_argv"] = [*tasks[-1]["argv"], "--resume"]
        tasks[-1]["retry_policy"] = "resume_same_root_from_verified_checkpoint"
        recourse_id = f"{prefix}_train_common_recourse"
        recourse_out = f"{root}/train_common_recourse"
        add(
            recourse_id,
            resource="gpu",
            argv=[
                py,
                f"{project}/scripts/baselines/comrecgc/run_common_recourse.py",
                "--dataset",
                "bace",
                "--mode",
                "full",
                "--upstream-root",
                official,
                "--dataset-dir",
                dataset,
                "--generation-dir",
                generation_out,
                "--distance-checkpoint",
                neurosed,
                "--output-dir",
                recourse_out,
                "--parent-limit",
                "360",
                "--device",
                "cuda:0",
            ],
            output=recourse_out,
            markers=["_RUN_COMPLETE.json"],
            dependencies=[generation_id],
            inputs=[official, dataset, generation_out, neurosed],
        )
        candidate_id = f"{prefix}_train_candidates"
        candidate_out = f"{root}/train_candidates"
        add(
            candidate_id,
            resource="gpu",
            argv=[
                py,
                script,
                "comrecgc-export",
                "--method",
                spec.method,
                "--common-recourse-dir",
                recourse_out,
                "--dataset-summary-json",
                f"{dataset}/dataset_summary.json",
                "--gnn-checkpoint",
                checkpoint,
                "--output-dir",
                candidate_out,
                "--device",
                "cuda:0",
            ],
            output=candidate_out,
            markers=["PASS"],
            dependencies=[recourse_id],
            inputs=[recourse_out, f"{dataset}/dataset_summary.json", checkpoint],
        )

    calibration_ids: list[str] = []
    for shard in range(NUM_SHARDS):
        task_id = f"{prefix}_calibration_shard_{shard}"
        shard_out = f"{root}/calibration/shard-{shard}"
        calibration_ids.append(task_id)
        add(
            task_id,
            resource="gpu",
            argv=[
                py,
                script,
                "verify-shard",
                "--method",
                spec.method,
                "--gnn-checkpoint",
                checkpoint,
                "--output-dir",
                shard_out,
                "--verification-stage",
                "BASELINE_CALIBRATION_VERIFY",
                "--split-path",
                calibration,
                "--predecessor-output",
                candidate_out,
                "--molclr-root",
                molclr_project,
                "--molclr-checkpoint",
                molclr_ckpt,
                "--shard-index",
                str(shard),
                "--wnode-cache-db",
                f"{root}/calibration/cache/shard-{shard}.sqlite3",
                "--node-embedding-cache-dir",
                f"{root}/calibration/cache/node-emb-shard-{shard}",
                "--device",
                "cuda:0",
            ],
            output=shard_out,
            markers=["PASS"],
            dependencies=[candidate_id],
            inputs=[calibration, candidate_out, checkpoint, molclr_project, molclr_ckpt],
        )
    calibration_merge_id = f"{prefix}_calibration_merge"
    calibration_merge_out = f"{root}/calibration/merged"
    merge_argv = [
        py,
        script,
        "merge",
        "--method",
        spec.method,
        "--verification-stage",
        "BASELINE_CALIBRATION_VERIFY",
        "--predecessor-output",
        candidate_out,
        "--output-dir",
        calibration_merge_out,
    ]
    for shard in range(NUM_SHARDS):
        merge_argv.extend(["--shard-dir", f"{root}/calibration/shard-{shard}"])
    add(
        calibration_merge_id,
        resource="cpu",
        argv=merge_argv,
        output=calibration_merge_out,
        markers=["PASS"],
        dependencies=calibration_ids,
        inputs=[f"{root}/calibration/shard-{shard}" for shard in range(NUM_SHARDS)],
    )
    selection_id = f"{prefix}_selection"
    selection_out = f"{root}/selection"
    add(
        selection_id,
        resource="cpu",
        argv=[
            py,
            script,
            "select",
            "--method",
            spec.method,
            "--matrix-output",
            calibration_merge_out,
            "--output-dir",
            selection_out,
        ],
        output=selection_out,
        markers=["PASS"],
        dependencies=[calibration_merge_id],
        inputs=[calibration_merge_out],
    )
    test_ids: list[str] = []
    for shard in range(NUM_SHARDS):
        task_id = f"{prefix}_test_shard_{shard}"
        shard_out = f"{root}/test/shard-{shard}"
        test_ids.append(task_id)
        add(
            task_id,
            resource="gpu",
            argv=[
                py,
                script,
                "verify-shard",
                "--method",
                spec.method,
                "--gnn-checkpoint",
                checkpoint,
                "--output-dir",
                shard_out,
                "--verification-stage",
                "BASELINE_TEST_EVAL",
                "--split-path",
                test,
                "--predecessor-output",
                selection_out,
                "--molclr-root",
                molclr_project,
                "--molclr-checkpoint",
                molclr_ckpt,
                "--shard-index",
                str(shard),
                "--wnode-cache-db",
                f"{root}/test/cache/shard-{shard}.sqlite3",
                "--node-embedding-cache-dir",
                f"{root}/test/cache/node-emb-shard-{shard}",
                "--device",
                "cuda:0",
            ],
            output=shard_out,
            markers=["PASS"],
            dependencies=[selection_id],
            inputs=[test, selection_out, checkpoint, molclr_project, molclr_ckpt],
        )
    test_merge_id = f"{prefix}_test_merge"
    test_merge_out = f"{root}/test/merged"
    test_merge_argv = [
        py,
        script,
        "merge",
        "--method",
        spec.method,
        "--verification-stage",
        "BASELINE_TEST_EVAL",
        "--predecessor-output",
        selection_out,
        "--output-dir",
        test_merge_out,
    ]
    for shard in range(NUM_SHARDS):
        test_merge_argv.extend(["--shard-dir", f"{root}/test/shard-{shard}"])
    add(
        test_merge_id,
        resource="cpu",
        argv=test_merge_argv,
        output=test_merge_out,
        markers=["PASS"],
        dependencies=test_ids,
        inputs=[f"{root}/test/shard-{shard}" for shard in range(NUM_SHARDS)],
    )
    final_id = f"{prefix}_final_freeze"
    final_out = f"{root}/final"
    add(
        final_id,
        resource="cpu",
        argv=[
            py,
            script,
            "freeze",
            "--method",
            spec.method,
            "--selection-output",
            selection_out,
            "--test-output",
            test_merge_out,
            "--output-dir",
            final_out,
        ],
        output=final_out,
        markers=["PASS", "FINAL_PASS.json"],
        dependencies=[selection_id, test_merge_id],
        inputs=[selection_out, test_merge_out],
    )
    return {
        "schema_version": "bace_baseline_controller_fragment_v1",
        "dataset": "bace",
        "method": spec.method,
        "method_id": spec.method_id,
        "root_task_ids": [preflight_id],
        "terminal_task_ids": [final_id],
        "tasks": tasks,
        "static_terminal": None,
    }


__all__ = ["build_bace_baseline_controller_fragment"]
