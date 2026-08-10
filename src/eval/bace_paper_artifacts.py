"""BACE paper artifacts computed from the existing GCF-style run contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.gcf_style_recourse_report import (
    best_recourse_by_parent,
    compute_k_curve,
    compute_prefix_metrics,
    load_method_run,
)


DISTANCE_LINE = "MolCLR-Node-Wasserstein"
DISTANCE_TYPE = "node_wasserstein"
CF_MODE = "strict_flip"
QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
FIGURE3_FIELDS = ("method", "k", "coverage", "cost")
FIGURE4_FIELDS = ("method", "threshold", "coverage")
TABLE2_FIELDS = ("method", "k", "coverage", "cost", "flip_rate", "cf_drop")
METHODS = {
    "ours": {
        "display": "Ours",
        "candidate_kind": "ours",
        "selection_method": "chemllm_stable300_external_selector",
    },
    "gcfexplainer": {
        "display": "GCFExplainer",
        "candidate_kind": "fullgraph",
        "selection_method": "native_gcf_summary_rank_filtered_by_validity",
    },
    "globalgce": {
        "display": "GlobalGCE",
        "candidate_kind": "fullgraph",
        "selection_method": "globalgce_frequency_top20_train_support_v1",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_sha256(
    identity: Mapping[str, Any] | None,
    *,
    fallback_path: str | Path | None = None,
) -> str | None:
    """Resolve a file identity without weakening SHA256 provenance checks."""

    record = dict(identity or {})
    declared = str(record.get("sha256") or "").strip().lower()
    if declared:
        if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
            raise ValueError(f"Invalid SHA256 in file identity: {declared!r}")
        return declared
    path_value = record.get("path") or fallback_path
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser().resolve()
    return sha256_file(path) if path.is_file() else None


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for source in rows:
                row = dict(source)
                writer.writerow(
                    {
                        field: ""
                        if row.get(field) is None
                        or (
                            isinstance(row.get(field), float)
                            and not math.isfinite(float(row[field]))
                        )
                        else row.get(field)
                        for field in fields
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot fit BACE thresholds without finite distances.")
    position = (len(ordered) - 1) * float(quantile)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def freeze_bace_connected_thresholds_from_matrix(
    *,
    calibration_matrix_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the existing q-grid from connected calibration actions only."""

    matrix_root = Path(calibration_matrix_dir).expanduser().resolve()
    manifest_path = matrix_root / "matrix_manifest.json"
    audit_path = matrix_root / "matrix_audit.json"
    pair_path = matrix_root / "pair_matrix.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("run_complete") is not True or audit.get("run_complete") is not True:
        raise ValueError("Connected calibration matrix is not complete.")
    if manifest.get("test_loaded") is not False:
        raise ValueError("Connected threshold source does not prove test_loaded=false.")
    if manifest.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
        raise ValueError("Threshold source is not a connected-residual matrix.")
    if manifest.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
        raise ValueError("Threshold source uses a different match policy.")
    if audit.get("disconnected_residual_used_count") != 0:
        raise ValueError("Threshold source used a disconnected residual.")
    if audit.get("all_winning_residuals_connected") is not True:
        raise ValueError("Threshold source has a disconnected winning residual.")
    rows = _read_jsonl(pair_path)
    finite: list[float] = []
    for row in rows:
        if not bool(row.get("pair_strict_flip")):
            continue
        if not (
            bool(row.get("residual_connected"))
            and bool(row.get("sanitize_ok"))
            and int(row.get("residual_num_components") or 0) == 1
            and not bool(row.get("contains_dot"))
        ):
            raise ValueError("Connected threshold source contains an invalid winner.")
        try:
            distance = float(row["wnode_distance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Connected strict-flip winner lacks WNode distance.") from exc
        if math.isfinite(distance) and distance >= 0.0:
            finite.append(distance)
    thresholds = [_linear_quantile(finite, quantile) for quantile in QUANTILES]
    inputs = dict(manifest.get("inputs") or {})
    payload = {
        "schema_version": "bace_wnode_thresholds_v2",
        "dataset": "BACE",
        "distance_line": DISTANCE_LINE,
        "distance_type": DISTANCE_TYPE,
        "cf_mode": CF_MODE,
        "threshold_source": "connected_ours_calibration_matrix",
        "threshold_protocol_version": "bace_common_quantile_grid_q05_q90_v1",
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "quantiles": list(QUANTILES),
        "thresholds": thresholds,
        "theta_star_quantile": 0.30,
        "theta_star": thresholds[3],
        "cost_cap_quantile": 0.90,
        "cost_cap": thresholds[-1],
        "finite_connected_strict_flip_pair_count": len(finite),
        "calibration_matrix_dir": str(matrix_root),
        "calibration_matrix_manifest_sha256": sha256_file(manifest_path),
        "calibration_matrix_audit_sha256": sha256_file(audit_path),
        "calibration_pair_matrix_sha256": sha256_file(pair_path),
        "calibration_cohort_hash": inputs.get("calibration_cohort_hash"),
        "calibration_parent_csv": (inputs.get("calibration_csv") or {}).get("path"),
        "calibration_parent_csv_sha256": (inputs.get("calibration_csv") or {}).get("sha256"),
        "teacher_sha256": (inputs.get("teacher_path") or {}).get("sha256"),
        "molclr_checkpoint_sha256": (inputs.get("molclr_checkpoint") or {}).get("sha256"),
        "distance_implementation_version": inputs.get("distance_implementation_version"),
        "size_penalty_beta": inputs.get("wnode_size_penalty_beta"),
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
        "shared_across_methods": True,
        "old_threshold_contaminated": True,
    }
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Frozen connected BACE threshold differs: {target}")
        return existing
    _write_json(target, payload)
    return payload


def freeze_bace_thresholds(
    *,
    calibration_run_dir: str | Path,
    output_path: str | Path,
    calibration_parent_csv: str | Path,
    action_semantics_version: str = "hard_delete_all_matches_v1",
    match_selection_policy: str = "min_wnode_then_cfdrop_then_match_index_v1",
) -> dict[str, Any]:
    run_root = Path(calibration_run_dir).expanduser().resolve()
    source = run_root / "distance_quantiles.csv"
    if not source.is_file():
        raise FileNotFoundError(f"BACE calibration quantiles are missing: {source}")
    rows = _read_csv(source)
    parsed: list[tuple[float, float]] = []
    for row in rows:
        try:
            quantile = float(row["quantile"])
            threshold = float(row["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid BACE distance quantile row: {row}") from exc
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"BACE threshold must be finite and non-negative: {row}")
        parsed.append((quantile, threshold))
    if len(parsed) != len(QUANTILES):
        raise ValueError(f"BACE calibration must yield {len(QUANTILES)} quantiles.")
    for (actual, _threshold), expected in zip(parsed, QUANTILES, strict=True):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"BACE quantile grid differs from the frozen protocol: {actual} != {expected}"
            )
    thresholds = [value for _quantile, value in parsed]
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("BACE calibration thresholds are not monotone.")
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    if str(run_config.get("threshold_source")) != "auto_quantile":
        raise ValueError("BACE thresholds must originate from Ours auto-quantile calibration.")
    if action_semantics_version == CONNECTED_ACTION_SEMANTICS:
        if match_selection_policy != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("Connected thresholds require the connected match policy.")
        if run_config.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
            raise ValueError("Connected thresholds cannot be frozen from a legacy run.")
        if run_config.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("Connected threshold run has the wrong match policy.")
    payload = {
        "schema_version": (
            "bace_wnode_thresholds_v2"
            if action_semantics_version == CONNECTED_ACTION_SEMANTICS
            else "bace_wnode_thresholds_v1"
        ),
        "dataset": "BACE",
        "distance_line": DISTANCE_LINE,
        "distance_type": DISTANCE_TYPE,
        "cf_mode": CF_MODE,
        "threshold_source": "frozen_ours_calibration",
        "quantiles": list(QUANTILES),
        "thresholds": thresholds,
        "theta_star_quantile": 0.30,
        "theta_star": thresholds[3],
        "cost_cap_quantile": 0.90,
        "cost_cap": thresholds[-1],
        "calibration_run_dir": str(run_root),
        "calibration_distance_quantiles_sha256": sha256_file(source),
        "calibration_parent_csv": str(Path(calibration_parent_csv).expanduser().resolve()),
        "calibration_parent_csv_sha256": sha256_file(calibration_parent_csv),
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
    }
    if action_semantics_version == CONNECTED_ACTION_SEMANTICS:
        payload.update(
            {
                "action_semantics_version": action_semantics_version,
                "match_selection_policy": match_selection_policy,
            }
        )
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Frozen BACE threshold contract already differs: {target}")
        return existing
    _write_json(target, payload)
    return payload


def load_bace_thresholds(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema not in {"bace_wnode_thresholds_v1", "bace_wnode_thresholds_v2"}:
        raise ValueError(f"Unsupported BACE threshold schema: {source}")
    if payload.get("distance_line") != DISTANCE_LINE or payload.get("cf_mode") != CF_MODE:
        raise ValueError("BACE threshold distance/strict-flip semantics changed.")
    quantiles = [float(value) for value in payload.get("quantiles", [])]
    thresholds = [float(value) for value in payload.get("thresholds", [])]
    if len(quantiles) != len(QUANTILES) or len(thresholds) != len(QUANTILES):
        raise ValueError("BACE frozen threshold vector must have seven values.")
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(quantiles, QUANTILES, strict=True)
    ):
        raise ValueError("BACE frozen quantile protocol changed.")
    if not math.isclose(
        float(payload["theta_star"]), thresholds[3], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("BACE theta_star is not q30.")
    if not math.isclose(
        float(payload["cost_cap"]), thresholds[-1], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("BACE cost_cap is not q90.")
    if schema == "bace_wnode_thresholds_v2":
        if payload.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
            raise ValueError("BACE v2 thresholds lack connected action semantics.")
        if payload.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("BACE v2 thresholds have the wrong match policy.")
        if payload.get("threshold_fitted_on_test") is not False:
            raise ValueError("BACE v2 thresholds do not prove calibration-only fitting.")
    return payload


def export_bace_method_artifacts(
    *,
    method: str,
    test_run_dir: str | Path,
    thresholds_json: str | Path,
    output_dir: str | Path,
    expected_parent_count: int,
    expected_top_k: int = 20,
    selection_manifest: str | Path | None = None,
    test_evaluation_count: int | None = None,
    reference_artifact_root: str | Path | None = None,
    action_semantics_version: str = "hard_delete_all_matches_v1",
    match_selection_policy: str = "min_wnode_then_cfdrop_then_match_index_v1",
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unsupported BACE method: {method}")
    spec = METHODS[method]
    thresholds = load_bace_thresholds(thresholds_json)
    if action_semantics_version == CONNECTED_ACTION_SEMANTICS:
        if match_selection_policy != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("Connected artifacts require the connected match policy.")
        if thresholds.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
            raise ValueError("Connected artifacts require connected frozen thresholds.")
        if thresholds.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("Connected artifacts and thresholds use different match policies.")
    run = load_method_run(
        str(spec["display"]),
        test_run_dir,
        expected_top_k=int(expected_top_k),
        expected_num_parents=int(expected_parent_count),
    )
    connected_candidate_count: int | None = None
    if (
        spec["candidate_kind"] == "fullgraph"
        and action_semantics_version == CONNECTED_ACTION_SEMANTICS
    ):
        connected_candidate_count = 0
        for candidate in run.candidates:
            molecule = Chem.MolFromSmiles(candidate.smiles)
            if molecule is None or molecule.GetNumAtoms() <= 0:
                raise ValueError(
                    f"BACE {spec['display']} candidate is not a nonempty molecule: "
                    f"rank={candidate.rank}."
                )
            try:
                Chem.SanitizeMol(molecule)
            except Exception as exc:
                raise ValueError(
                    f"BACE {spec['display']} candidate does not sanitize: "
                    f"rank={candidate.rank}."
                ) from exc
            if "." in candidate.smiles or len(Chem.GetMolFrags(molecule)) != 1:
                raise ValueError(
                    f"BACE {spec['display']} candidate is disconnected: "
                    f"rank={candidate.rank}."
                )
            connected_candidate_count += 1
    theta_star = float(thresholds["theta_star"])
    if (
        run.config.get("action_semantics_version") or "hard_delete_all_matches_v1"
    ) != action_semantics_version:
        raise ValueError("BACE evaluator and artifact action semantics differ.")
    if (
        run.config.get("match_selection_policy")
        or "min_wnode_then_cfdrop_then_match_index_v1"
    ) != match_selection_policy:
        raise ValueError("BACE evaluator and artifact match-selection policies differ.")
    if method == "ours" and action_semantics_version == CONNECTED_ACTION_SEMANTICS:
        if run.disconnected_residual_used_count != 0:
            raise ValueError("Connected Ours run used a disconnected residual.")
        if run.covered_residual_connected_rate not in {None, 1.0}:
            raise ValueError("Connected Ours run contains a non-connected covered residual.")
    selection_contract: dict[str, Any] | None = None
    selection_manifest_path: Path | None = None
    if selection_manifest is not None:
        selection_manifest_path = Path(selection_manifest).expanduser().resolve()
        selection_contract = json.loads(
            selection_manifest_path.read_text(encoding="utf-8")
        )
        if selection_contract.get("selection_frozen") is not True:
            raise ValueError("BACE selection manifest is not frozen.")
        if selection_contract.get("test_used") is not False:
            raise ValueError("BACE frozen selector does not prove test_used=false.")
        if selection_contract.get("gcf_result_used") is not False:
            raise ValueError("BACE frozen selector does not prove gcf_result_used=false.")
        expected_selection_split = "train" if method == "globalgce" else "calibration"
        if selection_contract.get("selection_split") != expected_selection_split:
            raise ValueError(
                "BACE frozen selector uses the wrong split: "
                f"actual={selection_contract.get('selection_split')!r}, "
                f"expected={expected_selection_split!r}."
            )
        if action_semantics_version == CONNECTED_ACTION_SEMANTICS:
            if selection_contract.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
                raise ValueError("Frozen selection lacks connected action semantics.")
            if selection_contract.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
                raise ValueError("Frozen selection uses a different match policy.")
        if int(test_evaluation_count or 0) != 1:
            raise ValueError("A frozen BACE selection requires test_evaluation_count=1.")
    prefix = compute_k_curve(run, threshold=theta_star, max_k=int(expected_top_k))
    figure3 = [
        {
            "method": spec["display"],
            "k": int(row["k"]),
            "coverage": float(row["coverage"]),
            "cost": row["conditional_median_cost"],
        }
        for row in prefix
    ]
    distances, _drops = best_recourse_by_parent(run, k=10)
    ordered_distances = [float(distances[parent_id]) for parent_id in run.parent_ids]
    figure4 = [
        {
            "method": spec["display"],
            "threshold": float(threshold),
            "coverage": sum(value <= float(threshold) for value in ordered_distances)
            / len(ordered_distances),
        }
        for threshold in thresholds["thresholds"]
    ]
    k10 = compute_prefix_metrics(run, k=10, threshold=theta_star)
    table2 = {
        "method": spec["display"],
        "k": 10,
        "coverage": float(k10["coverage"]),
        "cost": k10["conditional_median_cost"],
        "flip_rate": k10["flip_rate_among_covered"],
        "cf_drop": k10["mean_cf_drop_among_covered"],
    }
    if [int(row["k"]) for row in figure3] != list(range(1, int(expected_top_k) + 1)):
        raise AssertionError("BACE Figure 3 K grid changed.")
    if any(
        right["coverage"] + 1e-12 < left["coverage"]
        for left, right in zip(figure3, figure3[1:])
    ):
        raise AssertionError("BACE Figure 3 coverage is not monotone.")
    if any(
        right["coverage"] + 1e-12 < left["coverage"]
        for left, right in zip(figure4, figure4[1:])
    ):
        raise AssertionError("BACE Figure 4 coverage is not monotone.")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"BACE paper artifact root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        _write_csv(temporary / "figure3_coverage_vs_k.csv", figure3, FIGURE3_FIELDS)
        _write_csv(
            temporary / "figure4_coverage_vs_threshold.csv", figure4, FIGURE4_FIELDS
        )
        _write_csv(
            temporary / f"table2_{method}_k10.csv", [table2], TABLE2_FIELDS
        )
        run_config = run.config
        candidate_ids = [candidate.candidate_id for candidate in run.candidates]
        parent_ids_sha256 = stable_json_sha256(list(run.parent_ids))
        run_teacher_sha256 = _identity_sha256(
            run_config.get("teacher"),
            fallback_path=run_config.get("teacher_path"),
        )
        run_molclr_sha256 = _identity_sha256(
            run_config.get("molclr_checkpoint_identity"),
            fallback_path=run_config.get("molclr_checkpoint"),
        )
        selected_candidate_ids_exact = True
        teacher_identity_exact = True
        molclr_identity_exact = True
        threshold_identity_exact = True
        if selection_contract is not None:
            selected_candidate_ids_exact = candidate_ids == [
                str(value) for value in selection_contract.get("selected_candidate_ids", [])
            ]
            if not selected_candidate_ids_exact:
                raise ValueError("Evaluator candidate order differs from frozen selection.")
            expected_teacher = selection_contract.get("teacher_identity") or {}
            expected_molclr = selection_contract.get("molclr_identity") or {}
            teacher_identity_exact = bool(
                _identity_sha256(expected_teacher)
                and _identity_sha256(expected_teacher) == run_teacher_sha256
            )
            molclr_identity_exact = bool(
                _identity_sha256(expected_molclr)
                and _identity_sha256(expected_molclr) == run_molclr_sha256
            )
            threshold_identity_exact = bool(
                selection_contract.get("threshold_manifest_sha256")
                == sha256_file(thresholds_json)
            )
            if not teacher_identity_exact:
                raise ValueError("Evaluator teacher differs from frozen selection.")
            if not molclr_identity_exact:
                raise ValueError("Evaluator MolCLR checkpoint differs from frozen selection.")
            if not threshold_identity_exact:
                raise ValueError("Evaluator thresholds differ from frozen selection.")

        reference_protocol_exact = True
        reference_teacher_exact = True
        reference_molclr_exact = True
        reference_root: Path | None = None
        if reference_artifact_root is not None:
            reference_root = Path(reference_artifact_root).expanduser().resolve()
            reference_summary = json.loads(
                (reference_root / "summary.json").read_text(encoding="utf-8")
            )
            reference_manifest = json.loads(
                (reference_root / "run_manifest.json").read_text(encoding="utf-8")
            )
            reference_teacher_path = Path(
                str(reference_manifest.get("teacher_path") or "")
            ).expanduser()
            reference_molclr_path = Path(
                str(reference_manifest.get("molclr_checkpoint") or "")
            ).expanduser()
            reference_teacher_exact = bool(
                reference_teacher_path.is_file()
                and sha256_file(reference_teacher_path)
                == run_teacher_sha256
            )
            reference_molclr_exact = bool(
                reference_molclr_path.is_file()
                and sha256_file(reference_molclr_path)
                == run_molclr_sha256
            )
            reference_protocol_exact = bool(
                reference_summary.get("test_parent_ids_sha256") == parent_ids_sha256
                and math.isclose(
                    float(reference_summary.get("theta_star")),
                    theta_star,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(reference_summary.get("cost_cap")),
                    float(thresholds["cost_cap"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and reference_summary.get("cf_mode") == CF_MODE
                and reference_summary.get("distance_line") == DISTANCE_LINE
                and reference_manifest.get("thresholds_json_sha256")
                == sha256_file(thresholds_json)
                and reference_teacher_exact
                and reference_molclr_exact
            )
            if not reference_protocol_exact:
                raise ValueError(
                    "BACE v2 test cohort or frozen evaluation protocol differs from original Ours."
                )
        summary = {
            "schema_version": "bace_paper_method_summary_v1",
            "dataset": "BACE",
            "method": spec["display"],
            "source_label": 1,
            "target_label": 0,
            "distance_line": DISTANCE_LINE,
            "distance_type": DISTANCE_TYPE,
            "cf_mode": CF_MODE,
            "action_semantics_version": action_semantics_version,
            "match_selection_policy": match_selection_policy,
            "test_parent_count": len(run.parent_ids),
            "test_parent_ids_sha256": parent_ids_sha256,
            "candidate_count": len(run.candidates),
            "connected_candidate_count": connected_candidate_count,
            "all_candidates_connected": (
                connected_candidate_count == len(run.candidates)
                if connected_candidate_count is not None
                else None
            ),
            "pair_count": int(run.num_unique_parent_candidate_pairs),
            "theta_star": theta_star,
            "cost_cap": float(thresholds["cost_cap"]),
            "thresholds": list(thresholds["thresholds"]),
            "k10": table2,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "disconnected_residual_used_count": (
                run.disconnected_residual_used_count if method == "ours" else 0
            ),
            "covered_residual_connected_rate": (
                run.covered_residual_connected_rate if method == "ours" else None
            ),
            "selection_split": (
                selection_contract.get("selection_split")
                if selection_contract is not None
                else None
            ),
            "selected_sequence_sha256": (
                selection_contract.get("selected_sequence_sha256")
                if selection_contract is not None
                else None
            ),
            "test_evaluation_count": test_evaluation_count,
            "candidate_order_sha256": stable_json_sha256(candidate_ids),
            "selected_candidate_ids_exact": selected_candidate_ids_exact,
            "teacher_identity_exact": teacher_identity_exact,
            "molclr_identity_exact": molclr_identity_exact,
            "threshold_identity_exact": threshold_identity_exact,
            "reference_protocol_exact": reference_protocol_exact,
            "reference_teacher_exact": reference_teacher_exact,
            "reference_molclr_exact": reference_molclr_exact,
            "run_complete": True,
        }
        _write_json(temporary / "summary.json", summary)
        manifest = {
            "schema_version": "bace_paper_method_manifest_v1",
            "dataset": "BACE",
            "method": spec["display"],
            "source_evaluation_run": str(run.run_dir),
            "source_run_config_sha256": sha256_file(run.run_dir / "run_config.json"),
            "source_pair_details_sha256": sha256_file(
                run.run_dir / "details" / "pair_details.csv"
            ),
            "candidate_path": str(run.candidate_path),
            "candidate_path_sha256": (
                sha256_file(run.candidate_path)
                if run.candidate_path.is_file()
                else None
            ),
            "candidate_order_sha256": summary["candidate_order_sha256"],
            "test_parent_ids_sha256": parent_ids_sha256,
            "selection_method": run.selection_method,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "thresholds_json": str(Path(thresholds_json).expanduser().resolve()),
            "thresholds_json_sha256": sha256_file(thresholds_json),
            "teacher_path": run_config.get("teacher_path"),
            "molclr_checkpoint": run_config.get("molclr_checkpoint"),
            "distance_line": DISTANCE_LINE,
            "cf_mode": CF_MODE,
            "action_semantics_version": action_semantics_version,
            "match_selection_policy": match_selection_policy,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "disconnected_residual_used_count": summary[
                "disconnected_residual_used_count"
            ],
            "connected_candidate_count": connected_candidate_count,
            "all_candidates_connected": summary["all_candidates_connected"],
            "covered_residual_connected_rate": summary[
                "covered_residual_connected_rate"
            ],
            "selection_split": (
                selection_contract.get("selection_split")
                if selection_contract is not None
                else None
            ),
            "selection_manifest": (
                str(selection_manifest_path)
                if selection_manifest_path is not None
                else None
            ),
            "selection_manifest_sha256": (
                sha256_file(selection_manifest_path)
                if selection_manifest_path is not None
                else None
            ),
            "selected_sequence_sha256": (
                selection_contract.get("selected_sequence_sha256")
                if selection_contract is not None
                else None
            ),
            "test_evaluation_count": test_evaluation_count,
            "reference_artifact_root": str(reference_root) if reference_root else None,
            "reference_protocol_exact": reference_protocol_exact,
            "reference_teacher_exact": reference_teacher_exact,
            "reference_molclr_exact": reference_molclr_exact,
        }
        _write_json(temporary / "run_manifest.json", manifest)
        audit = {
            "schema_version": "bace_paper_method_audit_v1",
            "passed": True,
            "dataset": "BACE",
            "method": spec["display"],
            "figure3_schema": list(FIGURE3_FIELDS),
            "figure3_rows": len(figure3),
            "figure3_k": [int(row["k"]) for row in figure3],
            "figure4_schema": list(FIGURE4_FIELDS),
            "figure4_rows": len(figure4),
            "figure4_thresholds": [float(row["threshold"]) for row in figure4],
            "table2_schema": list(TABLE2_FIELDS),
            "table2_rows": 1,
            "strict_flip": True,
            "action_semantics_version": action_semantics_version,
            "match_selection_policy": match_selection_policy,
            "candidate_order_preserved": True,
            "selection_performed_in_eval": False,
            "coverage_vs_k_monotone": True,
            "coverage_vs_threshold_monotone": True,
            "selection_split": (
                selection_contract.get("selection_split")
                if selection_contract is not None
                else None
            ),
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "disconnected_residual_used_count": summary[
                "disconnected_residual_used_count"
            ],
            "covered_residual_connected_rate": summary[
                "covered_residual_connected_rate"
            ],
            "all_covered_residuals_connected": (
                method != "ours"
                or action_semantics_version != CONNECTED_ACTION_SEMANTICS
                or summary["covered_residual_connected_rate"] in {None, 1.0}
            ),
            "all_candidates_connected": summary["all_candidates_connected"],
            "test_evaluation_count": test_evaluation_count,
            "selected_candidate_ids_exact": selected_candidate_ids_exact,
            "teacher_identity_exact": teacher_identity_exact,
            "molclr_identity_exact": molclr_identity_exact,
            "threshold_identity_exact": threshold_identity_exact,
            "same_test_parents": reference_protocol_exact,
            "same_theta": reference_protocol_exact,
            "same_cost_definition": reference_protocol_exact,
            "same_reference_teacher": reference_teacher_exact,
            "same_reference_molclr": reference_molclr_exact,
        }
        _write_json(temporary / "final_artifact_audit.json", audit)
        file_inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        _write_json(
            temporary / "artifact_manifest.json",
            {"schema_version": "bace_paper_files_v1", "files": file_inventory},
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


__all__ = [
    "CF_MODE",
    "DISTANCE_LINE",
    "DISTANCE_TYPE",
    "FIGURE3_FIELDS",
    "FIGURE4_FIELDS",
    "METHODS",
    "QUANTILES",
    "TABLE2_FIELDS",
    "export_bace_method_artifacts",
    "freeze_bace_connected_thresholds_from_matrix",
    "freeze_bace_thresholds",
    "load_bace_thresholds",
    "sha256_file",
]
