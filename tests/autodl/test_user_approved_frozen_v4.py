from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from src.eval.four_by_four_main_results import audit_cell
from src.eval.four_by_four_registry import AuditConfig, audit_registry
import src.eval.user_approved_frozen_v4 as frozen_v4
from src.eval.user_approved_frozen_v4 import (
    APPROVED_CELLS,
    FrozenV4AdoptionError,
    adopt_user_approved_frozen_v4,
    load_approval_policy,
    validate_adopted_cell,
)


def _csv_bytes(fields: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _fixture_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / frozen_v4.SOURCE_ROOT_BASENAME
    source.mkdir()
    methods = ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
    datasets = ("AIDS", "Mutagenicity")
    figure3: list[dict[str, str]] = []
    figure4: list[dict[str, str]] = []
    table_rows: list[dict[str, str]] = []
    table_values: dict[tuple[str, str], tuple[str, str]] = {}
    for dataset_index, dataset in enumerate(datasets):
        for method_index, method in enumerate(methods):
            for k in range(1, 21):
                coverage = f"{min(0.99, 0.01 * k + 0.02 * method_index + 0.01 * dataset_index):.12f}"
                cost = f"{0.001 * (k + method_index + dataset_index):.12f}"
                figure3.append(
                    {
                        "Dataset": dataset,
                        "Method": method,
                        "K": str(k),
                        "Theta": "0.05",
                        "Coverage": coverage,
                        "Cost": cost,
                    }
                )
                if k == 10:
                    table_values[(dataset, method)] = (coverage, cost)
            for index in range(601):
                denominator = 100000 if dataset == "AIDS" else 50000
                figure4.append(
                    {
                        "Dataset": dataset,
                        "Method": method,
                        "K": "10",
                        "Threshold": f"{index / denominator:.8f}",
                        "Coverage": f"{min(0.99, index / 1000):.12f}",
                    }
                )
    for method in methods:
        table_rows.append(
            {
                "Method": method,
                "AIDS Coverage": table_values[("AIDS", method)][0],
                "AIDS Cost": table_values[("AIDS", method)][1],
                "NCI1 Coverage": "",
                "NCI1 Cost": "",
                "Mutagenicity Coverage": table_values[("Mutagenicity", method)][0],
                "Mutagenicity Cost": table_values[("Mutagenicity", method)][1],
                "Proteins Coverage": "",
                "Proteins Cost": "",
            }
        )
    csv_payloads = {
        "figure3_gcf_style_aids_mut_data.csv": _csv_bytes(
            ("Dataset", "Method", "K", "Theta", "Coverage", "Cost"), figure3
        ),
        "figure4_gcf_style_aids_mut_data.csv": _csv_bytes(
            ("Dataset", "Method", "K", "Threshold", "Coverage"), figure4
        ),
        "table2_gcf_style_aids_mut.csv": _csv_bytes(
            (
                "Method",
                "AIDS Coverage",
                "AIDS Cost",
                "NCI1 Coverage",
                "NCI1 Cost",
                "Mutagenicity Coverage",
                "Mutagenicity Cost",
                "Proteins Coverage",
                "Proteins Cost",
            ),
            table_rows,
        ),
    }
    for name, data in csv_payloads.items():
        (source / name).write_bytes(data)
    combined = {
        "schema_version": "aids_mut_gcf_style_csv_replay_v1",
        "render_only": True,
        "candidate_order_changed": False,
        "candidate_ranking_recomputed": False,
        "distance_recomputed": False,
        "teacher_recomputed": False,
        "selection_performed_in_plot": False,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "source_manifest_status": "advisory_not_used_as_numeric_source",
        "outputs": {
            name: {"bytes": len(data), "sha256": _sha(data)}
            for name, data in csv_payloads.items()
        },
        "source_csv_inventory": {
            name: {"bytes": len(data), "sha256": _sha(data)}
            for name, data in csv_payloads.items()
        },
    }
    combined_bytes = _json_bytes(combined)
    (source / "combined_manifest.json").write_bytes(combined_bytes)
    complete = {
        "run_complete": True,
        "render_only": True,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "manifest_sha256": _sha(combined_bytes),
    }
    complete_bytes = _json_bytes(complete)
    (source / "_RUN_COMPLETE.json").write_bytes(complete_bytes)
    payloads = {
        "_RUN_COMPLETE.json": complete_bytes,
        "combined_manifest.json": combined_bytes,
        **csv_payloads,
    }
    policy_payload = {
        "schema_version": "user_approved_frozen_v4_policy_v1",
        "approval_id": "USER_APPROVED_FROZEN_V4",
        "source_root_basename": frozen_v4.SOURCE_ROOT_BASENAME,
        "approval_scope": [f"{dataset}/{method}" for dataset, method in APPROVED_CELLS],
        "excluded_methods": ["CLEAR", "ComRecGC"],
        "registry_status": "ADOPTABLE_PASS",
        "source_files": {
            name: {"sha256": _sha(data)} for name, data in payloads.items()
        },
        "user_approved_waivers": [
            "legacy raw closure absent",
            "scientific identity hashes absent",
        ],
    }
    policy_path = tmp_path / "approval_policy.json"
    policy_path.write_bytes(_json_bytes(policy_payload))
    return source, policy_path


def _adopt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, policy_path = _fixture_source(tmp_path)
    policy = load_approval_policy(policy_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "outputs/adoption-v1"
    result = adopt_user_approved_frozen_v4(
        source_root=source,
        runtime_root=runtime,
        output_root=output,
        policy=policy,
        require_proc_writer_audit=False,
    )
    monkeypatch.setattr(frozen_v4, "DEFAULT_POLICY_PATH", policy_path)
    return source, policy, result


def test_production_policy_is_exact_and_excludes_clear_and_comrecgc() -> None:
    policy = load_approval_policy()
    assert policy.approval_scope == tuple(
        f"{dataset}/{method}" for dataset, method in APPROVED_CELLS
    )
    assert len(policy.source_hashes) == 5
    assert (
        policy.source_hashes["figure3_gcf_style_aids_mut_data.csv"]
        == "d63b36fb05d6f59dae9ce8adb5a1eeb28e3cfc03618ab0e181c7a9366f8acf73"
    )
    assert all("CLEAR" not in cell and "ComRecGC" not in cell for cell in policy.approval_scope)


def test_adoption_is_byte_bound_six_cell_fresh_and_registry_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, policy, result = _adopt(tmp_path, monkeypatch)
    inventory = json.loads(
        (result.output_root / "adopted_source_inventory.json").read_text()
    )
    assert inventory["all_content_read_once"] is True
    assert inventory["images_or_pdfs_read"] is False
    assert set(result.cell_roots) == set(policy.approval_scope)
    assert all(
        metadata["content_read_count"] == metadata["hash_scan_count"] == 1
        for metadata in inventory["files"].values()
    )
    for name, metadata in inventory["files"].items():
        copied = result.output_root / "source_bundle" / name
        assert copied.read_bytes() == (source / name).read_bytes()
        assert _sha(copied.read_bytes()) == metadata["sha256"]
    assert not any("clear" in path.as_posix().lower() for path in result.output_root.rglob("*"))
    assert not any("comrecgc" in path.as_posix().lower() for path in result.output_root.rglob("*"))
    for cell, root_text in result.cell_roots.items():
        ok, reasons, details = validate_adopted_cell(root_text, policy=policy)
        assert ok, (cell, reasons, details)
        root = Path(root_text)
        table = next(root.glob("table2_*_k10.csv"))
        row = next(csv.DictReader(table.open()))
        assert row["flip_rate"] == "N/A"
        assert row["cf_drop"] == "N/A"

    registry = audit_registry(
        AuditConfig(
            scan_roots=(result.output_root,),
            output_root=result.output_root / "unused-registry-root",
        )
    )
    assert registry.matrix_complete_cells == 6
    adopted = [row for row in registry.matrix_rows if row["status"] == "ADOPTABLE_PASS"]
    assert {(row["dataset"], row["method"]) for row in adopted} == set(APPROVED_CELLS)
    assert {row["registry_exception"] for row in adopted} == {
        "USER_APPROVED_FROZEN_V4"
    }
    assert all(not row["oracle_hash"] and not row["dataset_hash"] for row in adopted)
    audit_cell(adopted[0])


def test_tampered_projected_numeric_file_fails_exception_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, policy, result = _adopt(tmp_path, monkeypatch)
    root = Path(result.cell_roots["AIDS/Ours"])
    figure = root / "figure3_coverage_vs_k.csv"
    figure.write_text(figure.read_text().replace("0.010000000000", "0.110000000000", 1))
    ok, reasons, _ = validate_adopted_cell(root, policy=policy)
    assert not ok
    assert any("STANDARDIZED_SOURCE_PROJECTION_MISMATCH" in reason for reason in reasons)


def test_source_hash_mismatch_fails_without_publishing_output(tmp_path: Path) -> None:
    source, policy_path = _fixture_source(tmp_path)
    policy = load_approval_policy(policy_path)
    source.joinpath("table2_gcf_style_aids_mut.csv").write_text("tampered\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "outputs/adoption-v1"
    with pytest.raises(FrozenV4AdoptionError, match="source SHA mismatch"):
        adopt_user_approved_frozen_v4(
            source_root=source,
            runtime_root=runtime,
            output_root=output,
            policy=policy,
            require_proc_writer_audit=False,
        )
    assert not output.exists()


def test_refuses_nonfresh_or_paper_output(
    tmp_path: Path,
) -> None:
    source, policy_path = _fixture_source(tmp_path)
    policy = load_approval_policy(policy_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    existing = runtime / "outputs/existing"
    existing.mkdir(parents=True)
    with pytest.raises(FrozenV4AdoptionError, match="fresh and absent"):
        adopt_user_approved_frozen_v4(
            source_root=source,
            runtime_root=runtime,
            output_root=existing,
            policy=policy,
            require_proc_writer_audit=False,
        )
    with pytest.raises(FrozenV4AdoptionError, match="paper"):
        adopt_user_approved_frozen_v4(
            source_root=source,
            runtime_root=runtime,
            output_root=runtime / "paper/adoption",
            policy=policy,
            require_proc_writer_audit=False,
        )
