from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

from scripts.autodl import run_t13_from_hpc_import_v1 as science_cli
from scripts.autodl import run_t13_from_hpc_owner_v1 as t13_owner
from scripts.autodl import run_t8_hpc_import_owner_v1 as import_owner
from src.baselines import globalgce_hpc_autodl_import as importer
from src.baselines import globalgce_hpc_exact as exact
from src.baselines import globalgce_hpc_hierarchical as hierarchy
from src.baselines import globalgce_hpc_storage_safe as storage
from src.utils import t8_hpc_t13_successor_v1 as successor


ROOT = Path(__file__).resolve().parents[2]
IMPORT_LAUNCHER = ROOT / "scripts/autodl/launch_t8_hpc_import_owner_v1.sh"
T13_LAUNCHER = ROOT / "scripts/autodl/launch_t13_from_hpc_owner_v1.sh"


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _hashed(payload: dict, field: str) -> dict:
    value = dict(payload)
    value[field] = successor.canonical_sha256(value)
    return value


def _bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def _relay_package(tmp_path: Path, *, shard_count: int = 16) -> tuple[Path, dict]:
    package = tmp_path / f"relay-{shard_count}"
    package.mkdir()
    scientific_sha = "1" * 64
    partition_sha = "2" * 64
    merge_sha = "3" * 64
    execution_commit = "4" * 40
    code = [
        {"frm": 0, "to": 1, "vevlb": [6, 1, 6]},
        {"frm": 1, "to": 2, "vevlb": [-1, 1, 8]},
    ]
    selected = [
        {
            "support": 7,
            "global_preorder": 0,
            "dfs_code": code,
            "dfs_code_sha256": exact.dfs_code_sha256(exact.dfs_code_from_json(code)),
        }
    ]
    top = {
        "schema_version": "globalgce_hpc_exact_stable_top_k_v1",
        "top_k": 20,
        "selected_count": 1,
        "ordering": "SUPPORT_DESC_OFFICIAL_PREORDER_ASC",
        "selected": selected,
        "selected_sha256": successor.canonical_sha256(selected),
    }
    provenance = _hashed(
        {
            "execution_commit": execution_commit,
            "source_commit": "5" * 40,
            "route_kind": "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD",
            "dataset": "tastemolnet",
            "method": "globalgce",
            "split_scope": "train_only",
            "source_label": 1,
            "target_branches": [0, 2],
            "calibration_loaded": False,
            "test_loaded": False,
            "matrix_write_enabled": False,
        },
        "provenance_sha256",
    )
    proof = _hashed(
        {"disjoint": True, "complete": True, "partition_count": 16},
        "proof_sha256",
    )
    partition = _hashed(
        {
            "scope": "FULL_ROOT_UNIVERSE",
            "scientific_input_sha256": scientific_sha,
            "matrix_write_enabled": False,
            "provenance": provenance,
            "official_gspan": {"commit": exact.OFFICIAL_GLOBALGCE_COMMIT},
            "completeness_proof": proof,
            "configuration": {
                "min_support": 2,
                "min_vertices": 3,
                "max_vertices": 20,
                "top_k": 20,
            },
            "partitions": [{"partition_id": f"p-{index}"} for index in range(16)],
        },
        "manifest_sha256",
    )
    # This test binds the expected self-hash to the sealed partition itself.
    partition_sha = partition["manifest_sha256"]
    inner = _hashed(
        {"schema_version": storage.STORAGE_SAFE_BUNDLE_SCHEMA, "status": "PASS"},
        "bundle_content_sha256",
    )
    result_archive = package / "t8_exact_result_bundle.tar.gz"
    _tar(
        result_archive,
        [
            ("RESULT_MANIFEST.json", _bytes(inner)),
            ("partition_manifest.json", _bytes(partition)),
            ("merge/stable_top_k.json", _bytes(top)),
        ],
    )
    receipt = _hashed(
        {
            "schema_version": storage.STORAGE_SAFE_RECEIPT_SCHEMA,
            "status": "PASS",
            "matrix_write_enabled": False,
        },
        "receipt_sha256",
    )
    _json(package / "result_manifest.json", receipt)
    adoption = _hashed(
        {
            "schema_version": hierarchy.ARRAY_ADOPTION_SCHEMA,
            "status": "PASS",
            "shard_count": shard_count,
            "passed_shard_count": shard_count,
            "successful_shards_rerun": False,
            "shards": [
                {"shard_index": index, "result_sha256": f"{index + 10:064x}"}
                for index in range(shard_count)
            ],
            "matrix_write_enabled": False,
        },
        "array_adoption_sha256",
    )
    plan = _hashed(
        {
            "schema_version": hierarchy.GROUP_PLAN_SCHEMA,
            "status": "PASS",
            "partition_disjoint": True,
            "partition_complete": True,
            "official_global_order_preserved": True,
            "successful_shards_rerun": False,
            "matrix_write_enabled": False,
        },
        "group_plan_sha256",
    )
    final = _hashed(
        {
            "schema_version": hierarchy.FINAL_VERIFICATION_SCHEMA,
            "status": "PASS",
            "event_identity_unique": True,
            "pattern_identity_unique": True,
            "matrix_write_enabled": False,
            "group_plan_sha256": plan["group_plan_sha256"],
            "merge_result_sha256": merge_sha,
        },
        "verification_sha256",
    )
    identities = [
        {"name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(
            [
                ("array_adoption_manifest.json", _bytes(adoption)),
                ("final/hierarchical_verification.json", _bytes(final)),
                ("group_plan.json", _bytes(plan)),
            ]
        )
    ]
    evidence = _hashed(
        {
            "schema_version": hierarchy.EVIDENCE_MANIFEST_SCHEMA,
            "status": "PASS",
            "storage_safe_receipt_sha256": receipt["receipt_sha256"],
            "merge_result_sha256": merge_sha,
            "files": identities,
            "successful_shards_rerun": False,
            "matrix_write_enabled": False,
        },
        "evidence_manifest_sha256",
    )
    evidence_archive = package / "t8_hierarchical_evidence.tar.gz"
    _tar(
        evidence_archive,
        [
            ("EVIDENCE_MANIFEST.json", _bytes(evidence)),
            ("array_adoption_manifest.json", _bytes(adoption)),
            ("final/hierarchical_verification.json", _bytes(final)),
            ("group_plan.json", _bytes(plan)),
        ],
    )
    evidence_receipt = _hashed(
        {
            "schema_version": hierarchy.EVIDENCE_MANIFEST_SCHEMA,
            "status": "PASS",
            "archive_name": evidence_archive.name,
            "archive_bytes": evidence_archive.stat().st_size,
            "archive_sha256": _sha(evidence_archive),
            "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
            "merge_result_sha256": merge_sha,
            "matrix_write_enabled": False,
        },
        "receipt_sha256",
    )
    _json(package / "hierarchical_evidence_manifest.json", evidence_receipt)
    ready = _hashed(
        {
            "schema_version": hierarchy.PACKAGE_READY_SCHEMA,
            "status": "PASS",
            "result_archive_sha256": _sha(result_archive),
            "result_receipt_sha256": receipt["receipt_sha256"],
            "evidence_archive_sha256": _sha(evidence_archive),
            "evidence_receipt_sha256": evidence_receipt["receipt_sha256"],
            "merge_result_sha256": merge_sha,
            "matrix_write_enabled": False,
        },
        "package_ready_sha256",
    )
    _json(package / "HIERARCHICAL_PACKAGE_READY.json", ready)
    _json(
        package / "HPC_PACKAGE_READY.json",
        {
            "schema_version": importer.RELAY_READY_SCHEMA,
            "state": "HPC_PACKAGE_READY",
            "archive_sha256": _sha(result_archive),
            "hierarchical_evidence_sha256": _sha(evidence_archive),
            "matrix_write_enabled": False,
        },
    )
    return package, {
        "execution_commit": execution_commit,
        "scientific_sha": scientific_sha,
        "partition_sha": partition_sha,
        "merge_sha": merge_sha,
        "selected_sha": top["selected_sha256"],
    }


def _specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    monkeypatch.setattr(successor, "_clean_commit", lambda _root: "a" * 40)
    output = tmp_path / "specs"
    values = {
        "repo_root": ROOT,
        "python": Path(sys.executable),
        "output_root": output,
        "relay_import_parent": tmp_path / "relay",
        "import_output_root": tmp_path / "import",
        "t13_output_root": tmp_path / "t13",
        "t13_locator": tmp_path / "control" / "t13.locator.json",
        "matrix_authority_root": tmp_path / "matrix-authority",
        "publisher_lease_path": tmp_path / "control" / "publisher.lease",
        "gpu_lease_path": tmp_path / "control" / "gpu1.lease",
        "gnn_checkpoint": tmp_path / "inputs" / "gine",
        "train_csv": tmp_path / "inputs" / "train.csv",
        "calibration_csv": tmp_path / "inputs" / "calibration.csv",
        "test_csv": tmp_path / "inputs" / "test.csv",
        "official_root": tmp_path / "inputs" / "GlobalGCE",
        "molclr_root": tmp_path / "inputs" / "MolCLR",
        "molclr_checkpoint": tmp_path / "inputs" / "molclr.pt",
        "threshold_contract": tmp_path / "inputs" / "threshold.json",
        "wnode_cache_db": tmp_path / "cache" / "wnode.sqlite3",
        "node_embedding_cache_dir": tmp_path / "cache" / "embeddings",
        "expected_hpc_execution_commit": "4" * 40,
        "expected_scientific_input_sha256": "5" * 64,
        "expected_partition_manifest_sha256": "6" * 64,
        "gpu_index": 1,
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "import_attempt_id": "da80219a-0da6-4b7a-b522-68e1f4469caa",
        "t13_attempt_id": "f8f7ec0b-ab12-42de-965e-cf46f80deba6",
    }
    result = successor.build_spec_set(**values)
    return output, result


def test_predeployed_specs_keep_hpc_inert_and_publisher_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, specs = _specs(tmp_path, monkeypatch)
    assert sorted(path.name for path in root.iterdir()) == [
        "spec_set_manifest.json",
        "t13_from_hpc_task_spec.json",
        "t8_hpc_import_task_spec.json",
        "t8_publisher_task_spec.json",
    ]
    assert specs["manifest"]["publisher_claim_count_for_cell"] == 1
    assert specs["import"]["hpc_permissions"] == {
        "matrix_write": False,
        "gine_inference": False,
        "calibration": False,
        "test": False,
    }
    assert specs["t13"]["status"] == "PREDEPLOYED_WAITING_IMPORT_PASS"
    assert specs["t13"]["owner_acquires_gpu_only_after_import_pass"] is True
    assert "--hpc-import-root" in specs["t13"]["command"]
    assert "--t8-pass-root" not in specs["t13"]["command"]
    assert specs["publisher"]["matrix_writer"] == (
        "EXISTING_FAST16_MATRIX_PUBLISHER_QUEUE_ONLY"
    )
    assert not Path(specs["publisher"]["terminal_root_locator"]).exists()


def test_import_owner_absent_bundle_waits_without_import_or_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, specs = _specs(tmp_path, monkeypatch)
    Path(specs["import"]["relay_import_parent"]).mkdir(parents=True)
    heartbeat = tmp_path / "import-owner" / "heartbeat.json"
    release = tmp_path / "import-owner" / "release.json"
    observed = import_owner.run(
        spec_root=root,
        heartbeat=heartbeat,
        release=release,
        poll_seconds=5,
        once=True,
    )
    assert observed["state"] == "WAITING_HPC_PACKAGE"
    assert observed["science_started"] is False
    assert not Path(specs["import"]["output_root"]).exists()
    assert not release.exists()


def test_exact_relay_verifier_imports_only_complete_16_shard_train_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, identity = _relay_package(tmp_path)
    monkeypatch.setattr(
        importer,
        "stream_verify_storage_safe_bundle",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "receipt_verified": True,
            "matrix_write_enabled": False,
            "scientific_input_sha256": identity["scientific_sha"],
            "partition_manifest_sha256": identity["partition_sha"],
            "merge_result_sha256": identity["merge_sha"],
            "event_count": 1,
            "pattern_count": 1,
            "rejection_count": 0,
        },
    )
    verified = importer.validate_relayed_hpc_package(
        package,
        expected_execution_commit=identity["execution_commit"],
        expected_scientific_input_sha256=identity["scientific_sha"],
        expected_partition_manifest_sha256=identity["partition_sha"],
        proc_root=tmp_path / "absent-proc",
    )
    assert verified["status"] == "PASS"
    assert verified["hierarchy"]["shard_count"] == 16
    assert verified["hpc_gine_inference_used"] is False
    assert verified["hpc_calibration_or_test_used"] is False
    assert verified["matrix_write_enabled"] is False
    output = tmp_path / "autodl-import"
    manifest = importer.import_relayed_hpc_package(
        package,
        output,
        expected_execution_commit=identity["execution_commit"],
        expected_scientific_input_sha256=identity["scientific_sha"],
        expected_partition_manifest_sha256=identity["partition_sha"],
        proc_root=tmp_path / "absent-proc",
    )
    assert manifest["state"] == "IMPORTED_TRAIN_SIDE_GSPAN_PENDING_AUTODL_T13"
    assert manifest["all_16_shards_verified"] is True
    assert manifest["rhs_chemistry_pending_autodl"] is True
    assert manifest["gine_inference_pending_autodl"] is True
    assert manifest["calibration_test_export_pending_autodl"] is True
    assert not (output / "PASS").exists()
    assert not (output / "cell_root_locator.json").exists()


def test_relay_verifier_rejects_incomplete_array_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, identity = _relay_package(tmp_path, shard_count=15)
    monkeypatch.setattr(
        importer,
        "stream_verify_storage_safe_bundle",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "receipt_verified": True,
            "matrix_write_enabled": False,
            "scientific_input_sha256": identity["scientific_sha"],
            "partition_manifest_sha256": identity["partition_sha"],
            "merge_result_sha256": identity["merge_sha"],
            "event_count": 1,
            "pattern_count": 1,
            "rejection_count": 0,
        },
    )
    output = tmp_path / "forbidden-import"
    with pytest.raises(importer.T8HPCAutoDLImportError, match="array binding"):
        importer.import_relayed_hpc_package(
            package,
            output,
            expected_execution_commit=identity["execution_commit"],
            expected_scientific_input_sha256=identity["scientific_sha"],
            expected_partition_manifest_sha256=identity["partition_sha"],
            proc_root=tmp_path / "absent-proc",
        )
    assert not output.exists()


def test_t13_owner_does_not_request_gpu_before_import_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _spec = _specs(tmp_path, monkeypatch)

    @contextmanager
    def forbidden_lease(_path: Path):
        raise AssertionError("GPU lease requested before import release")
        yield None

    monkeypatch.setattr(t13_owner, "_nonblocking_lease", forbidden_lease)
    observed = t13_owner.run_once(
        spec_root=root,
        release_path=tmp_path / "absent-release.json",
        heartbeat=tmp_path / "t13-owner" / "heartbeat.json",
        owner_root=tmp_path / "t13-owner",
        poll_seconds=5,
    )
    assert observed["state"] == "WAITING_HPC_IMPORT_PASS"
    assert observed["science_started"] is False


def test_t13_owner_refuses_busy_gpu_without_starting_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, specs = _specs(tmp_path, monkeypatch)
    release_path = tmp_path / "release.json"
    release_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        t13_owner,
        "validate_t13_release",
        lambda **_kwargs: {"release_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        t13_owner,
        "_gpu_observation",
        lambda index, expected: {
            "gpu_index": index,
            "gpu_uuid": expected,
            "processes": [{"pid": 99, "process_name": "main-table"}],
        },
    )
    monkeypatch.setattr(
        t13_owner,
        "_run_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("science started on busy GPU")
        ),
    )
    observed = t13_owner.run_once(
        spec_root=root,
        release_path=release_path,
        heartbeat=tmp_path / "owner" / "heartbeat.json",
        owner_root=tmp_path / "owner",
        poll_seconds=5,
    )
    assert observed["state"] == "READY_WAITING_T13_GPU_IDLE"
    assert observed["science_started"] is False
    assert not Path(specs["t13"]["output_root"]).exists()


def test_verify_only_writes_standard_locator_only_after_autodl_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, specs = _specs(tmp_path, monkeypatch)
    terminal = Path(specs["t13"]["output_root"])
    terminal.mkdir(parents=True)
    _json(
        terminal / "run_manifest.json",
        {
            "status": "PASS",
            "run_complete": True,
            "upstream_kind": "hpc_exact_gspan_import",
            "hpc_mining_only": True,
            "rhs_chemistry_run_on_autodl": True,
            "gine_inference_run_on_autodl": True,
            "calibration_test_run_on_autodl": True,
            "test_used_for_selection": False,
        },
    )
    import src.baselines.tastemolnet_globalgce_full as full

    monkeypatch.setattr(
        full, "verify_t13_output", lambda _root: {"status": "PASS", "passed": True}
    )
    locator = successor.publish_verified_t13_locator(
        spec_root=root, terminal_root=terminal
    )
    assert locator == {
        "schema_version": "fast16_matrix_cell_root_locator_v1",
        "status": "READY",
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "terminal_root": str(terminal),
    }
    assert json.loads(
        Path(specs["publisher"]["terminal_root_locator"]).read_text(encoding="utf-8")
    ) == locator


def test_conflicting_locator_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, specs = _specs(tmp_path, monkeypatch)
    terminal = Path(specs["t13"]["output_root"])
    terminal.mkdir(parents=True)
    _json(
        terminal / "run_manifest.json",
        {
            "status": "PASS",
            "run_complete": True,
            "upstream_kind": "hpc_exact_gspan_import",
            "hpc_mining_only": True,
            "rhs_chemistry_run_on_autodl": True,
            "gine_inference_run_on_autodl": True,
            "calibration_test_run_on_autodl": True,
            "test_used_for_selection": False,
        },
    )
    import src.baselines.tastemolnet_globalgce_full as full

    monkeypatch.setattr(
        full, "verify_t13_output", lambda _root: {"status": "PASS", "passed": True}
    )
    locator_path = Path(specs["publisher"]["terminal_root_locator"])
    _json(locator_path, {"status": "READY", "terminal_root": "/conflict"})
    before = locator_path.read_bytes()
    with pytest.raises(successor.T8HPCT13SpecError, match="conflicts"):
        successor.publish_verified_t13_locator(spec_root=root, terminal_root=terminal)
    assert locator_path.read_bytes() == before


def test_science_cli_binds_hpc_import_not_managed_t8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "hpc.yaml"
    config.write_text("runtime: test\n", encoding="utf-8")
    observed: dict = {}

    def fake_authority(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(science_cli, "load_input_authority", fake_authority)
    monkeypatch.setattr(
        science_cli,
        "run_t13_full",
        lambda **_kwargs: {"status": "SEALED", "matrix_write_enabled": False},
    )
    spec_root = tmp_path / "specs"
    expected_paths = {
        "gnn_checkpoint": str((tmp_path / "gine").resolve()),
        "train_csv": str((tmp_path / "train.csv").resolve()),
        "calibration_csv": str((tmp_path / "cal.csv").resolve()),
        "test_csv": str((tmp_path / "test.csv").resolve()),
        "official_root": str((tmp_path / "official").resolve()),
        "molclr_root": str((tmp_path / "molclr").resolve()),
        "molclr_checkpoint": str((tmp_path / "molclr.pt").resolve()),
        "wnode_cache_db": str((tmp_path / "wnode.sqlite3").resolve()),
        "node_embedding_cache_dir": str((tmp_path / "embeddings").resolve()),
        "threshold_contract": str((tmp_path / "threshold.json").resolve()),
    }
    monkeypatch.setattr(
        science_cli,
        "validate_spec_set",
        lambda *_args, **_kwargs: {
            "t13": {
                "output_root": str((tmp_path / "output").resolve()),
                "required_import_root": str((tmp_path / "import").resolve()),
                "input_paths": expected_paths,
            }
        },
    )
    args = Namespace(
        config=config,
        set=["inference.fallback_to_heuristic=false"],
        output_dir=tmp_path / "output",
        spec_root=spec_root,
        verify_only=False,
        resume=False,
        hpc_import_root=tmp_path / "import",
        gnn_checkpoint=tmp_path / "gine",
        train_csv=tmp_path / "train.csv",
        calibration_csv=tmp_path / "cal.csv",
        test_csv=tmp_path / "test.csv",
        official_root=tmp_path / "official",
        molclr_root=tmp_path / "molclr",
        molclr_checkpoint=tmp_path / "molclr.pt",
        wnode_cache_db=tmp_path / "wnode.sqlite3",
        node_embedding_cache_dir=tmp_path / "embeddings",
        threshold_contract=tmp_path / "threshold.json",
        device="cuda:0",
        epochs=100,
        top_k_native=20,
        min_freq=2,
        learning_rate=0.1,
        dropout=0.5,
        generation_chunk_size=32,
        oracle_batch_size=256,
        gspan_flush_every=256,
        gspan_max_in_memory_candidates=256,
        seed=7,
    )
    assert science_cli.run(args) == 0
    assert observed["t8_pass_root"] is None
    assert observed["hpc_import_root"] == args.hpc_import_root


def test_launchers_are_predeploy_only_and_have_no_signal_or_matrix_write() -> None:
    import_text = IMPORT_LAUNCHER.read_text(encoding="utf-8")
    t13_text = T13_LAUNCHER.read_text(encoding="utf-8")
    owner_text = (
        ROOT / "scripts/autodl/run_t13_from_hpc_owner_v1.py"
    ).read_text(encoding="utf-8")
    assert "run_t8_hpc_import_owner_v1.py" in import_text
    assert "run_t13_from_hpc_owner_v1.py" in t13_text
    assert "READY_WAITING_T13_GPU" not in import_text
    for forbidden in ("SIGKILL", "pkill", "killall", "os.kill"):
        assert forbidden not in import_text
        assert forbidden not in t13_text
        assert forbidden not in owner_text
    assert "append_tastemolnet_cells" not in owner_text
    assert "matrix_authority" not in owner_text.lower()


@pytest.mark.parametrize(
    "name",
    [
        "build_t8_hpc_t13_successor_specs_v1.sh",
        "run_t8_hpc_import_owner_v1.sh",
        "status_t8_hpc_t13_successor_v1.sh",
        "run_t13_from_hpc_import_v1.sh",
        "run_t13_from_hpc_owner_v1.sh",
    ],
)
def test_autodl_entrypoints_have_static_refusal_slurm_pairs(name: str) -> None:
    text = (ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
        "REFUSING_HPC_EXECUTION",
    ):
        assert token in text
