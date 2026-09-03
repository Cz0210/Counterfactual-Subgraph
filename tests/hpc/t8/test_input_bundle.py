from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
NATIVE_TRAIN_CSV_SHA256 = hashlib.sha256(b"frozen train.csv bytes not transferred").hexdigest()
SCRIPT = ROOT / "scripts" / "hpc" / "t8" / "build_input_bundle.py"
SPEC = importlib.util.spec_from_file_location("t8_build_input_bundle", SCRIPT)
assert SPEC and SPEC.loader
BUNDLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUNDLE
SPEC.loader.exec_module(BUNDLE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: Any) -> str:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = tmp_path / "frozen-source"
    source.mkdir(parents=True)
    files = {
        "graph_jsonl": source / "production_graphs_target0.jsonl",
        "source_input_manifest": source / "input_manifest.json",
        "split_manifest": source / "split_manifest.json",
        "train_cohort_manifest": source / "train_cohort_manifest.json",
        "feature_schema": source / "feature_schema.json",
        "data_use_authorization": source / "hpc_data_authorization.json",
    }
    files["graph_jsonl"].write_text(
        '{"graph_id":0,"nodes":[{"id":0,"label":1}],"edges":[]}\n'
        '{"graph_id":1,"nodes":[{"id":0,"label":2}],"edges":[]}\n',
        encoding="utf-8",
    )
    _write_json(
        files["source_input_manifest"],
        {
            "train_only": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "graph_jsonl_sha256": _sha(files["graph_jsonl"]),
            "gspan_graph_count": 2,
            "selected_parent_count": 3,
            "seed": 7,
            "source_label": 1,
            "target_label": 0,
            "selected_parent_cohort_sha256": "c" * 64,
            "native_train_csv_sha256": NATIVE_TRAIN_CSV_SHA256,
        },
    )
    _write_json(
        files["split_manifest"],
        {
            "dataset": "tastemolnet",
            "files": {"train": {"sha256": NATIVE_TRAIN_CSV_SHA256}},
            "calibration_loaded_for_training": False,
            "test_loaded_for_training": False,
            "test_evaluated_during_training": False,
            "test_used_for_checkpoint_selection": False,
        },
    )
    _write_json(
        files["train_cohort_manifest"],
        {
            "train_only": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "selected_count": 3,
            "selected_cohort_sha256": "c" * 64,
        },
    )
    _write_json(files["feature_schema"], {"node": ["atomic_number"], "edge": ["bond_type"]})
    authorization: dict[str, object] = {
        "authorization_version": 1,
        "authorized_by": "user_project_owner",
        "allow_hpc_train_only_derived_t8_input_transfer": True,
        "route_kind": BUNDLE.ROUTE_KIND,
        "dataset": "tastemolnet",
        "method": "GlobalGCE",
        "split_scope": "train_only",
        "source_label": 1,
        "allowed_targets": [0, 2],
        "shared_transaction_database_authorized": True,
        "target_label_affects_graph_export": False,
        "source_graph_jsonl_sha256": _sha(files["graph_jsonl"]),
        "source_graph_count": 2,
        "native_train_csv_sha256": NATIVE_TRAIN_CSV_SHA256,
        "model_weights_included": False,
        "active_sqlite_or_wal_included": False,
        "checkpoint_tmp_included": False,
        "calibration_payload_included": False,
        "test_payload_included": False,
        "matrix_publication_allowed_from_hpc": False,
        "no_redistribution": True,
    }
    authorization["authorization_sha256"] = _canonical_sha(authorization)
    _write_json(files["data_use_authorization"], authorization)
    return source, files


def _args(tmp_path: Path, source: Path, files: dict[str, Path]) -> list[str]:
    arguments = [
        "--config",
        str(ROOT / "configs" / "hpc.yaml"),
        "--output-root",
        str(tmp_path / "bundle"),
        "--allowed-source-root",
        str(source),
        "--route-kind",
        BUNDLE.ROUTE_KIND,
        "--target",
        "0",
        "--target",
        "2",
        "--source-commit",
        "a" * 40,
        "--official-globalgce-commit",
        "b" * 40,
        "--graph-count",
        "2",
        "--selected-parent-count",
        "3",
        "--source-cohort-sha256",
        "c" * 64,
        "--production-fingerprint",
        "d" * 64,
        "--native-train-csv-sha256",
        NATIVE_TRAIN_CSV_SHA256,
        "--source-label",
        "1",
        "--seed",
        "7",
        "--epochs",
        "100",
        "--min-support",
        "2",
        "--min-vertices",
        "3",
        "--max-vertices",
        "20",
        "--top-k",
        "20",
        "--root-count",
        "50",
        "--stability-window-seconds",
        "0",
    ]
    for role, path in files.items():
        arguments.extend(("--source", f"{role}={path}"))
        arguments.extend(("--expected-sha256", f"{role}={_sha(path)}"))
    return arguments


def test_bundle_is_closed_deterministic_allowlist(tmp_path: Path) -> None:
    source, files = _fixture_inputs(tmp_path)
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    receipt = BUNDLE.build_bundle(args)
    output = tmp_path / "bundle"

    assert receipt["state"] == "PASS"
    manifest = json.loads((output / "input_bundle_manifest.json").read_text())
    declared = manifest.pop("manifest_sha256")
    assert declared == _canonical_sha(manifest)
    manifest["manifest_sha256"] = declared
    assert manifest["split_scope"] == "train_only"
    assert manifest["graph_payload_audit"]["ordered_graph_ids_sha256"] == _canonical_sha([0, 1])
    assert len(manifest["files"]) == 6
    assert manifest["native_train_csv_provenance"] == {
        "payload_included": False,
        "sha256": NATIVE_TRAIN_CSV_SHA256,
        "source_input_manifest_bound": True,
        "authorization_bound": True,
        "split_manifest_sha256": _sha(files["split_manifest"]),
    }
    assert manifest["transaction_binding"]["target_labels"] == [0, 2]
    target_hashes = manifest["transaction_binding"]["target_to_graph_jsonl_sha256"]
    assert target_hashes["0"] == target_hashes["2"] == _sha(files["graph_jsonl"])
    assert all(item["stat_before"] == item["stat_after"] for item in manifest["files"])
    assert all(item["source_sha256_before"] == item["source_sha256_after"] for item in manifest["files"])

    manifest_file_hash = _sha(output / "input_bundle_manifest.json")
    assert (output / "input_bundle_manifest.json.sha256").read_text().split()[0] == manifest_file_hash
    assert (output / "t8_hpc_input_bundle.tar.sha256").read_text().split()[0] == _sha(
        output / "t8_hpc_input_bundle.tar"
    )
    with tarfile.open(output / "t8_hpc_input_bundle.tar", "r") as archive:
        names = archive.getnames()
    assert names == sorted(names)
    assert set(names) == {
        "input_bundle_manifest.json",
        "input_bundle_manifest.json.sha256",
        *(f"payload/{name}" for name in BUNDLE.ROLE_TO_NAME.values()),
    }
    assert "payload/train.csv" not in names
    assert not any(name.endswith((".pt", ".pth", ".sqlite", ".safetensors", "-wal", "-shm")) for name in names)

    second_arguments = _args(tmp_path, source, files)
    output_index = second_arguments.index("--output-root") + 1
    second_arguments[output_index] = str(tmp_path / "bundle-second")
    BUNDLE.build_bundle(BUNDLE.parse_args(second_arguments))
    assert _sha(output / "t8_hpc_input_bundle.tar") == _sha(
        tmp_path / "bundle-second" / "t8_hpc_input_bundle.tar"
    )


def test_non_train_graph_payload_fails_closed(tmp_path: Path) -> None:
    source, files = _fixture_inputs(tmp_path)
    files["graph_jsonl"].write_text(
        '{"graph_id":0,"nodes":[{"id":0,"label":1}],"edges":[],"split":"test"}\n'
        '{"graph_id":1,"nodes":[{"id":0,"label":2}],"edges":[]}\n'
    )
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match="must contain exactly"):
        BUNDLE.build_bundle(args)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "rows",
    [
        (
            '{"graph_id":1,"nodes":[{"id":0,"label":1}],"edges":[]}\n'
            '{"graph_id":0,"nodes":[{"id":0,"label":2}],"edges":[]}\n'
        ),
        (
            '{"graph_id":0,"nodes":[{"id":0,"label":1}],"edges":[]}\n'
            '{"graph_id":2,"nodes":[{"id":0,"label":2}],"edges":[]}\n'
        ),
    ],
)
def test_graph_id_must_equal_zero_based_insertion_order(
    tmp_path: Path, rows: str
) -> None:
    source, files = _fixture_inputs(tmp_path)
    files["graph_jsonl"].write_text(rows, encoding="utf-8")
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match="zero-based JSONL insertion index"):
        BUNDLE.build_bundle(args)


@pytest.mark.parametrize(
    "rows, error",
    [
        (
            '{"graph_id":0,"nodes":[{"id":0,"label":"C"}],"edges":[]}\n'
            '{"graph_id":1,"nodes":[{"id":0,"label":2}],"edges":[]}\n',
            "invalid node value",
        ),
        (
            '{"graph_id":0,"nodes":[{"id":0,"label":1},{"id":1,"label":2}],'
            '"edges":[{"source":0,"target":1,"label":"single"}]}\n'
            '{"graph_id":1,"nodes":[{"id":0,"label":2}],"edges":[]}\n',
            "invalid edge value",
        ),
        (
            '{"graph_id":0,"nodes":[{"id":0,"label":1},{"id":1,"label":2}],'
            '"edges":[]}\n'
            '{"graph_id":1,"nodes":[{"id":0,"label":2}],"edges":[]}\n',
            "graph is disconnected",
        ),
    ],
)
def test_graph_labels_are_integer_and_each_graph_is_connected(
    tmp_path: Path, rows: str, error: str
) -> None:
    source, files = _fixture_inputs(tmp_path)
    files["graph_jsonl"].write_text(rows, encoding="utf-8")
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match=error):
        BUNDLE.build_bundle(args)


def test_source_manifest_must_prove_no_calibration_or_test_load(tmp_path: Path) -> None:
    source, files = _fixture_inputs(tmp_path)
    manifest = json.loads(files["source_input_manifest"].read_text())
    manifest["test_loaded"] = True
    _write_json(files["source_input_manifest"], manifest)
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match="test_loaded=false"):
        BUNDLE.build_bundle(args)


def test_source_outside_allowlist_fails_closed(tmp_path: Path) -> None:
    source, files = _fixture_inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    replacement = outside / "feature_schema.json"
    replacement.write_bytes(files["feature_schema"].read_bytes())
    files["feature_schema"] = replacement
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match="outside all allowed roots"):
        BUNDLE.build_bundle(args)


def test_symlink_and_weight_payloads_are_rejected(tmp_path: Path) -> None:
    source, files = _fixture_inputs(tmp_path)
    real = files["feature_schema"]
    link = source / "linked_feature_schema.json"
    link.symlink_to(real)
    files["feature_schema"] = link
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match="symlink source"):
        BUNDLE.build_bundle(args)

    source2, files2 = _fixture_inputs(tmp_path / "second")
    forbidden = source2 / "weights.pt"
    forbidden.write_bytes(files2["feature_schema"].read_bytes())
    files2["feature_schema"] = forbidden
    args2 = BUNDLE.parse_args(_args(tmp_path / "second", source2, files2))
    with pytest.raises(BUNDLE.BundleContractError, match="forbidden mutable/weight payload"):
        BUNDLE.build_bundle(args2)


def test_authorization_self_hash_and_target_contract_are_strict(tmp_path: Path) -> None:
    source, files = _fixture_inputs(tmp_path)
    authorization = json.loads(files["data_use_authorization"].read_text())
    authorization["no_redistribution"] = False
    _write_json(files["data_use_authorization"], authorization)
    args = BUNDLE.parse_args(_args(tmp_path, source, files))
    with pytest.raises(BUNDLE.BundleContractError, match="self-hash changed"):
        BUNDLE.build_bundle(args)

    source2, files2 = _fixture_inputs(tmp_path / "second")
    arguments = _args(tmp_path / "second", source2, files2)
    target_index = arguments.index("--target")
    del arguments[target_index : target_index + 2]
    args2 = BUNDLE.parse_args(arguments)
    with pytest.raises(BUNDLE.BundleContractError, match="targets must be exactly"):
        BUNDLE.build_bundle(args2)


def test_source_change_during_stability_window_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, files = _fixture_inputs(tmp_path)
    arguments = _args(tmp_path, source, files)
    arguments[arguments.index("--stability-window-seconds") + 1] = "0.01"
    original_sleep = BUNDLE.time.sleep
    changed = False

    def mutate_once(seconds: float) -> None:
        nonlocal changed
        if not changed:
            files["graph_jsonl"].write_text(
                files["graph_jsonl"].read_text(encoding="utf-8")
                + '{"graph_id":2,"nodes":[{"id":0,"label":3}],"edges":[]}\n',
                encoding="utf-8",
            )
            changed = True
        original_sleep(0)

    monkeypatch.setattr(BUNDLE.time, "sleep", mutate_once)
    args = BUNDLE.parse_args(arguments)
    with pytest.raises(BUNDLE.BundleContractError, match="(SHA-256 mismatch|source (stat|content) changed)"):
        BUNDLE.build_bundle(args)
    assert not (tmp_path / "bundle").exists()
