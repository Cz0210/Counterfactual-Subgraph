from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from scripts.audit_public_artifact_no_dataset_redistribution import main as audit_main
from src.utils.tastemolnet_public_artifacts import (
    AUDIT_MARKER,
    AUDIT_STATUS,
    TastePublicArtifactError,
    audit_tastemolnet_public_artifacts,
)
from src.utils.tastemolnet_research_policy import (
    ACTIVE_STATE,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    stable_json_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = PROJECT_ROOT / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_policy(tmp_path: Path) -> Path:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["authorization_basis"] = "explicit_user_instruction"
    payload["authorization_state"] = ACTIVE_STATE
    payload["authorization_source"] = "user_project_owner_instruction"
    payload["research_compute_allowed"] = True
    payload["paper_result_reporting_allowed"] = True
    payload["aggregated_metrics_release_allowed"] = True
    payload["figure_release_allowed"] = True
    payload["permissions"]["research_execution"] = "ALLOWED"
    payload["permissions"]["paper_reporting"] = "ALLOWED"
    payload["permissions"]["aggregate_publication"] = (
        "ALLOWED_AFTER_PUBLIC_ARTIFACT_AUDIT"
    )
    payload["execution"]["run_tastemolnet"] = 1
    path = tmp_path / "active-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _pending_policy(tmp_path: Path) -> Path:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["authorization_basis"] = "forwarded_user_instruction_pending_root_activation"
    payload["authorization_state"] = "PENDING_ROOT_ACTIVATION"
    payload["authorization_source"] = "PENDING_ROOT_ACTIVATION"
    payload["research_compute_allowed"] = False
    payload["paper_result_reporting_allowed"] = False
    payload["aggregated_metrics_release_allowed"] = False
    payload["figure_release_allowed"] = False
    payload["trained_model_release_allowed"] = False
    payload["permissions"]["research_execution"] = "PENDING_ROOT_ACTIVATION"
    payload["permissions"]["paper_reporting"] = "PENDING_ROOT_ACTIVATION"
    payload["permissions"]["aggregate_publication"] = (
        "ALLOWED_ONLY_AFTER_ACTIVATION_AND_PUBLIC_ARTIFACT_AUDIT"
    )
    payload["execution"]["run_tastemolnet"] = 0
    path = tmp_path / "pending-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _private_roots(tmp_path: Path) -> tuple[Path, Path]:
    prepared = tmp_path / "private/prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    protected = {
        "provenance_manifest.json": json.dumps(
            {"dataset": "tastemolnet", "private": True}, sort_keys=True
        ).encode(),
        "splits/train.csv": b"molecule_id,smiles,label\nTASTE_AAAAAAAA,CCO,1\n",
    }
    identities: dict[str, dict[str, object]] = {}
    for relative, data in protected.items():
        path = prepared / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        identities[relative] = {"bytes": len(data), "sha256": _sha(path)}
    output_manifest = {
        "schema_version": 1,
        "files": identities,
        "manifest_digest": stable_json_sha256(identities),
    }
    (prepared / "output_manifest.json").write_text(
        json.dumps(output_manifest, sort_keys=True), encoding="utf-8"
    )

    cache = tmp_path / "private/cache"
    cache.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "calibration", "test"):
        path = cache / f"{split}.pt"
        path.write_bytes(f"private-cache-{split}".encode())
        splits[split] = {
            "cache_file": path.name,
            "cache_sha256": _sha(path),
        }
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "molecular_graph_cache_manifest_v1",
                "dataset": "tastemolnet",
                "num_classes": 3,
                "split_order": ["train", "validation", "calibration", "test"],
                "splits": splits,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared, cache


def _public_root(tmp_path: Path, policy_path: Path) -> Path:
    policy = load_tastemolnet_research_policy(policy_path)
    root = tmp_path / "public"
    root.mkdir()
    artifacts = {
        "aggregate_metrics.json": (
            "aggregate_metrics",
            {
                "schema_version": "tastemolnet_public_aggregate_metrics_v1",
                "dataset": "tastemolnet",
                "num_classes": 3,
                "source_label": 1,
                "metrics": {"macro_f1": 0.72, "class_recall": [0.7, 0.8, 0.6]},
            },
        ),
        "confusion.json": (
            "aggregate_confusion_matrix",
            {
                "schema_version": "tastemolnet_public_confusion_matrix_v1",
                "dataset": "tastemolnet",
                "split": "test",
                "labels": [0, 1, 2],
                "matrix": [[3, 1, 0], [0, 4, 1], [1, 0, 5]],
            },
        ),
        "counts.json": (
            "aggregate_split_counts",
            {
                "schema_version": "tastemolnet_public_split_counts_v1",
                "dataset": "tastemolnet",
                "counts": {
                    split: {"0": 1, "1": 2, "2": 3}
                    for split in ("train", "validation", "calibration", "test")
                },
            },
        ),
    }
    rows = []
    for relative, (role, payload) in artifacts.items():
        path = root / relative
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        rows.append(
            {
                "path": relative,
                "role": role,
                "sha256": _sha(path),
                "contains_molecule_level_content": False,
            }
        )
    manifest = {
        "schema_version": "tastemolnet_public_release_manifest_v1",
        "dataset": "tastemolnet",
        "policy_file_sha256": policy.file_sha256,
        "policy_canonical_sha256": policy.canonical_sha256,
        "upstream_terms_status": "NOT_EXPLICITLY_STATED",
        "dataset_redistribution_allowed": False,
        "artifacts": rows,
    }
    (root / "public_release_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _audit(tmp_path: Path, public: Path, policy: Path) -> dict[str, object]:
    prepared, cache = _private_roots(tmp_path)
    return audit_tastemolnet_public_artifacts(
        public_root=public,
        policy_path=policy,
        expected_policy_sha256=_sha(policy),
        prepared_root=prepared,
        graph_cache_root=cache,
    )


def test_valid_aggregate_only_tree_passes_without_license_claim(tmp_path: Path) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    result = _audit(tmp_path, public, policy)
    assert result["status"] == AUDIT_STATUS
    assert result["audit_marker"] == AUDIT_MARKER
    assert result["dataset_redistribution_allowed"] is False
    assert result["upstream_terms_status"] == "NOT_EXPLICITLY_STATED"
    assert result["dataset_payloads_copied"] is False
    assert result["artifact_count"] == 3
    serialized = json.dumps(result, sort_keys=True)
    assert "LICENSE_PASS" not in serialized
    assert '"passed"' not in serialized


def test_pending_checked_policy_cannot_authorize_publication(tmp_path: Path) -> None:
    policy = _pending_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    prepared, cache = _private_roots(tmp_path)
    with pytest.raises(TasteResearchPolicyError, match="NOT_ACTIVATED"):
        audit_tastemolnet_public_artifacts(
            public_root=public,
            policy_path=policy,
            prepared_root=prepared,
            graph_cache_root=cache,
        )


def test_renamed_prepared_data_copy_is_rejected_by_hash(tmp_path: Path) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    prepared, cache = _private_roots(tmp_path)
    copied = public / "aggregate_metrics.json"
    copied.write_bytes((prepared / "splits/train.csv").read_bytes())
    manifest = json.loads((public / "public_release_manifest.json").read_text())
    manifest["artifacts"][0]["sha256"] = _sha(copied)
    (public / "public_release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TastePublicArtifactError, match="copies protected"):
        audit_tastemolnet_public_artifacts(
            public_root=public,
            policy_path=policy,
            prepared_root=prepared,
            graph_cache_root=cache,
        )


def test_nested_molecule_fields_are_rejected(tmp_path: Path) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    metrics = public / "aggregate_metrics.json"
    payload = json.loads(metrics.read_text())
    payload["metrics"]["debug"] = {"canonical_smiles": "CCO"}
    metrics.write_text(json.dumps(payload))
    manifest = json.loads((public / "public_release_manifest.json").read_text())
    manifest["artifacts"][0]["sha256"] = _sha(metrics)
    (public / "public_release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TastePublicArtifactError, match="molecule-level field"):
        _audit(tmp_path, public, policy)


def test_unregistered_symlink_or_extra_file_is_rejected(tmp_path: Path) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    (public / "extra.json").write_text("{}")
    with pytest.raises(TastePublicArtifactError, match="inventory changed"):
        _audit(tmp_path, public, policy)
    (public / "extra.json").unlink()
    (public / "extra.json").symlink_to(public / "aggregate_metrics.json")
    with pytest.raises(TastePublicArtifactError, match="symlink/special"):
        _audit(tmp_path, public, policy)


def test_hardlinked_artifact_and_opaque_checkpoint_are_rejected(tmp_path: Path) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    original = public / "aggregate_metrics.json"
    link = tmp_path / "second-link.json"
    os.link(original, link)
    with pytest.raises(TastePublicArtifactError, match="physical, unlinked regular"):
        _audit(tmp_path, public, policy)
    link.unlink()
    manifest = json.loads((public / "public_release_manifest.json").read_text())
    opaque = public / "model.pt"
    opaque.write_bytes(b"weights")
    manifest["artifacts"].append(
        {
            "path": "model.pt",
            "role": "model_checkpoint",
            "sha256": _sha(opaque),
            "contains_molecule_level_content": False,
        }
    )
    (public / "public_release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TastePublicArtifactError, match="role is forbidden"):
        _audit(tmp_path, public, policy)


def test_public_method_configuration_has_no_free_form_payload_channel(
    tmp_path: Path,
) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    path = public / "method.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "tastemolnet_public_method_configuration_v1",
                "dataset": "tastemolnet",
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "num_classes": 3,
                "source_label": 1,
                "source_label_name": "Sweet",
                "counterfactual_mode": "untargeted_strict_flip",
                "hyperparameters": {"batch_size": 64},
                "notes": "unregistered free-form content",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = public / "public_release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "path": path.name,
            "role": "method_configuration",
            "sha256": _sha(path),
            "contains_molecule_level_content": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TastePublicArtifactError, match="schema changed"):
        _audit(tmp_path, public, policy)


def test_cli_emits_audit_marker_and_no_pass_marker(tmp_path: Path, capsys) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    prepared, cache = _private_roots(tmp_path)
    output = tmp_path / "audit/audit.json"
    assert audit_main(
        [
            "--public-root",
            str(public),
            "--policy",
            str(policy),
            "--expected-policy-sha256",
            _sha(policy),
            "--prepared-root",
            str(prepared),
            "--graph-cache-root",
            str(cache),
            "--output",
            str(output),
        ]
    ) == 0
    stdout = capsys.readouterr().out
    assert f"[{AUDIT_MARKER}]" in stdout
    assert "LICENSE_PASS" not in stdout
    assert not (output.parent / "PASS").exists()


def test_cli_refuses_to_mutate_manifest_closed_public_root(
    tmp_path: Path,
) -> None:
    policy = _active_policy(tmp_path)
    public = _public_root(tmp_path, policy)
    prepared, cache = _private_roots(tmp_path)
    output = public / "audit.json"
    assert audit_main(
        [
            "--public-root",
            str(public),
            "--policy",
            str(policy),
            "--expected-policy-sha256",
            _sha(policy),
            "--prepared-root",
            str(prepared),
            "--graph-cache-root",
            str(cache),
            "--output",
            str(output),
        ]
    ) == 65
    assert not output.exists()
