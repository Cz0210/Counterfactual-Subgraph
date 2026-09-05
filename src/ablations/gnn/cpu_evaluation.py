"""BACE seed-7 proposal-fixed evaluation using the frozen main evaluator.

The five classifiers share an unchanged structural candidate universe, MolCLR,
thresholds and selector configuration.  Only classifier predictions and the
calibration-selected rule order vary.  No generator or main-matrix writer is
imported.  Parent-bound checkpoints allow long CPU evaluation to resume.
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.ablations.gnn.cpu_training import bundle_file, load_bundle
from src.eval.bace_frozen_gnn_contracts import (
    BACEParent, atomic_csv, atomic_json, atomic_jsonl, atomic_marker,
    load_bace_parents, read_json, read_jsonl, sha256_file, stable_sha256,
)

BACKBONES = ("gine", "gin", "gcn", "gatv2", "gatedgcn_plus")
SCHEMA = "bace_gnn_proposal_fixed_seed7_v1"


def _input(root: Path, manifest: Mapping[str, Any], key: str) -> Path:
    return bundle_file(root, manifest, str(manifest[key]))


def _candidates(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Adopt structural rows; do not reconstruct them from classifier scores."""
    rows = read_jsonl(_input(root, manifest, "candidate_universe_path"))
    pool = read_jsonl(_input(root, manifest, "candidate_pool_path"))
    if len(rows) != 66 or len(pool) != 1412:
        raise ValueError("Expected exact BACE main universe: 66 candidates / 1412 pool rows")
    ids = [str(row.get("candidate_id", "")) for row in rows]
    fragments = [str(row.get("canonical_fragment", "")) for row in rows]
    if any(not value for value in ids + fragments) or len(set(ids)) != len(ids):
        raise ValueError("Structural candidate identity is empty or duplicated")
    return rows


def frozen_selector(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Use the main selected variant and numeric thresholds without refitting."""
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict, VariantConfig
    selection = read_json(_input(root, manifest, "frozen_selection_manifest_path"))
    config = read_json(_input(root, manifest, "selector_variant_configs_path"))
    thresholds_payload = read_json(_input(root, manifest, "thresholds_path"))
    if selection.get("thresholds") != thresholds_payload:
        raise ValueError("Main selection/threshold payload mismatch")
    if selection.get("test_loaded") is not False or selection.get("selection_frozen") is not True:
        raise ValueError("Main selector is not a pre-test frozen selection")
    if config.get("top_k") != 20 or config.get("table_k") != 10:
        raise ValueError("BACE primary K contract must be 20/10")
    name = str(selection["selected_variant"])
    variant = VariantConfig(**config["variants"][name])
    if variant.name != name:
        raise ValueError("Selector variant name mismatch")
    weights = tuple(float(x) for x in config["prefix_weights"])
    if len(weights) != 20 or any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError("Invalid frozen prefix weights")
    return {"variant": variant, "thresholds": threshold_bundle_from_dict(thresholds_payload),
            "prefix_weights": weights, "local_swap_passes": int(config["local_swap_passes"]),
            "input_sha256": stable_sha256({"selection": selection, "config": config,
                                             "thresholds": thresholds_payload})}


def _all_parents(path: Path) -> list[BACEParent]:
    # Both classes are needed for classification metrics.  IDs/SMILES parsing
    # remains the same BACE loader used by the main route.
    rows = load_bace_parents(path, source_label=0) + load_bace_parents(path, source_label=1)
    if len({row.parent_id for row in rows}) != len(rows):
        raise ValueError("Duplicate parent ID across labels")
    return sorted(rows, key=lambda row: row.source_row_index)


def cohort_ids(parents: Sequence[BACEParent], predictions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if set(predictions) != set(BACKBONES):
        raise ValueError("Common cohort requires all five exact backbones")
    native: dict[str, list[str]] = {}
    for backbone in BACKBONES:
        records = predictions[backbone]
        if len(records) != len(parents):
            raise ValueError("Incomplete parent prediction sequence")
        native[backbone] = [p.parent_id for p, r in zip(parents, records, strict=True)
                            if p.label == 1 and int(r["predicted_label"]) == 1]
    common = sorted(set.intersection(*(set(native[name]) for name in BACKBONES)))
    return {"native": native, "common": common, "source_label": 1,
            "definition": "true_source_and_correctly_predicted_source", "backbones": list(BACKBONES)}


def matrix_from_pairs(parent_ids: Sequence[str], candidates: Sequence[Mapping[str, Any]],
                      pair_rows: Sequence[Mapping[str, Any]], *, root: Path, split: str) -> Any:
    from src.eval.mutagenicity_wnode_selector import MatrixData
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("Duplicate parent cohort ID")
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate ID")
    pidx = {v: i for i, v in enumerate(parent_ids)}
    cidx = {v: i for i, v in enumerate(candidate_ids)}
    d = np.full((len(pidx), len(cidx)), np.inf, dtype=np.float64)
    drops = np.full_like(d, np.nan)
    applicable = np.zeros(d.shape, dtype=bool)
    seen = set()
    for row in pair_rows:
        key = (str(row["parent_id"]), str(row["candidate_id"]))
        if key[0] not in pidx or key[1] not in cidx or key in seen:
            raise ValueError("Pair outside or duplicated within frozen Cartesian product")
        seen.add(key)
        pos = (pidx[key[0]], cidx[key[1]])
        applicable[pos] = bool(row["applicable"])
        if row["pair_strict_flip"]:
            value, drop = float(row["wnode_distance"]), float(row["cf_drop"])
            if not math.isfinite(value) or value < 0 or not math.isfinite(drop) or not applicable[pos]:
                raise ValueError("Invalid finite strict-flip distance")
            d[pos], drops[pos] = value, drop
    if len(seen) != len(pidx) * len(cidx):
        raise ValueError("Incomplete frozen parent/candidate Cartesian product")
    finite = d[np.isfinite(d)]
    return MatrixData(root, tuple(parent_ids), tuple(dict(row) for row in candidates), d, drops,
                      applicable, finite, len(pidx), len(cidx), len(seen), int(finite.size),
                      {"split": split}, {"split": split, "test_loaded": split == "test"},
                      tuple(dict(row) for row in candidates))


def select_calibration(matrix: Any, selector: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    if matrix.manifest.get("split") != "calibration" or matrix.manifest.get("test_loaded") is not False:
        raise ValueError("Selector accepts calibration-only matrices")
    if not matrix.parent_ids:
        raise ValueError("BLOCKED_EMPTY_CALIBRATION_COHORT")
    return _select_frozen_order(matrix, selector)


def _select_frozen_order(matrix: Any, selector: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    from src.eval.mutagenicity_wnode_selector import (
        build_candidate_chemistry, build_coverage_redundancy_matrix, _objective_callable,
        greedy_select, optimize_insertion_order, local_swap_search,
    )
    chemistry = build_candidate_chemistry(matrix.candidate_rows, size_normalization_rows=matrix.full_candidate_rows)
    redundancy = build_coverage_redundancy_matrix(matrix.distances, selector["thresholds"].levels)
    variant = selector["variant"]
    objective = _objective_callable(matrix=matrix, thresholds=selector["thresholds"],
        prefix_weights=selector["prefix_weights"], variant=variant,
        coverage_redundancy_matrix=redundancy, structural_similarity_matrix=chemistry.structural_similarity,
        normalized_sizes=chemistry.normalized_sizes)
    sequence, trace = greedy_select(range(len(matrix.candidate_rows)), top_k=20,
                                    objective_fn=objective, candidate_ids=matrix.candidate_ids)
    insertion, swap = [], []
    if variant.insertion_reorder:
        sequence, insertion = optimize_insertion_order(sequence, objective_fn=objective, candidate_ids=matrix.candidate_ids)
    if variant.local_swap:
        sequence, swap = local_swap_search(sequence, all_candidate_indices=range(len(matrix.candidate_rows)),
            objective_fn=objective, candidate_ids=matrix.candidate_ids, max_passes=selector["local_swap_passes"])
    return sequence, {"greedy": trace, "insertion": insertion, "swap": swap, "objective": objective(sequence)}


def explanation_metrics(matrix: Any, sequence: Sequence[int], thresholds: Any) -> dict[str, Any]:
    from src.eval.mutagenicity_wnode_selector import build_candidate_chemistry, build_coverage_redundancy_matrix, compute_prefix_metrics
    if len(sequence) != 20 or len(set(sequence)) != 20:
        raise ValueError("Explanation requires exactly 20 frozen unique rules")
    if not matrix.parent_ids:
        return {"state": "VALID_EMPTY_COHORT", "cohort_size": 0, "CCRCov@10": None,
                "CCRCov@20": None, "AUC_over_K_1_20": None, "conditional_median_WNode": None,
                "strict_flip_rate": None, "applicable_rate": None, "selected_rule_diversity": None,
                "covered_parent_ids": [], "prefix_rows": [], "parent_rows": [], "threshold_rows": []}
    chemistry = build_candidate_chemistry(matrix.candidate_rows, size_normalization_rows=matrix.full_candidate_rows)
    red = build_coverage_redundancy_matrix(matrix.distances, thresholds.levels)
    rows, parent_rows = compute_prefix_metrics(sequence, matrix=matrix, thresholds=thresholds,
        coverage_redundancy_matrix=red, structural_similarity_matrix=chemistry.structural_similarity)
    curve = [float(row["ccrcov_theta_star"]) for row in rows]
    best = np.min(matrix.distances[:, sequence], axis=1)
    grid = sorted(set(float(x) for x in thresholds.raw_thresholds))
    threshold_rows = [{"threshold": x, "CCRCov": float(np.mean(best <= x)), "K": 20} for x in grid]
    last = rows[-1]
    auc = float(sum((left + right) / 2 for left, right in zip(curve[:-1], curve[1:])))
    return {"state": "PASS", "cohort_size": len(matrix.parent_ids), "CCRCov@10": curve[9],
            "CCRCov@20": curve[19], "AUC_over_K_1_20": auc,
            "AUC_over_K_1_20_normalized": auc / 19,
            "conditional_median_WNode": last["conditional_median_cost"],
            "strict_flip_rate": last["strict_flip_parent_count"] / len(matrix.parent_ids),
            "applicable_rate": last["applicable_rate"],
            "selected_rule_diversity": 1 - last["structural_redundancy"],
            "covered_parent_ids": [p for p, value in zip(matrix.parent_ids, best) if value <= thresholds.theta_star],
            "prefix_rows": rows, "parent_rows": parent_rows, "threshold_rows": threshold_rows}


def _distance(root: Path, manifest: Mapping[str, Any], output: Path) -> Any:
    from src.eval.node_wasserstein_distance import MolCLRNodeWassersteinConfig, MolCLRNodeWassersteinDistance
    from src.chem.hard_deletion import CONNECTED_MATCH_SELECTION_POLICY
    frozen = dict(manifest["wnode_config"])
    required = {"distance_line", "distance_namespace", "distance_type", "feature_cost", "node_mass",
                "size_penalty_beta", "solver", "match_selection_policy", "no_valid_strict_flip_semantics"}
    if set(frozen) != required or frozen["solver"] != "exact_emd2":
        raise ValueError("WNode must use the complete frozen exact-EMD main contract")
    if (frozen["match_selection_policy"] != CONNECTED_MATCH_SELECTION_POLICY
            or frozen["distance_line"] != "MolCLR-Node-Wasserstein"
            or frozen["distance_type"] != "node_wasserstein"
            or frozen["no_valid_strict_flip_semantics"] != "+inf"):
        raise ValueError("Hard-deletion all-match aggregation differs from main")
    config = {key: frozen[key] for key in ("feature_cost", "node_mass", "size_penalty_beta", "distance_namespace")}
    config["encoder_type"] = str(manifest["molclr_encoder_type"])
    if config["encoder_type"] != "gin":
        raise ValueError("This node-state evaluator requires the main GIN MolCLR encoder")
    source = root / str(manifest["molclr_source_root"])
    source.resolve(strict=True).relative_to(root)
    for rel in manifest["files"]:
        if rel.startswith(str(manifest["molclr_source_root"]) + "/"):
            bundle_file(root, manifest, rel)
    return MolCLRNodeWassersteinDistance(MolCLRNodeWassersteinConfig(
        molclr_root=source, molclr_ckpt=_input(root, manifest, "molclr_checkpoint_path"),
        cache_db=output / "cache" / "wnode.sqlite", node_emb_cache_dir=output / "cache" / "nodes",
        device="cpu", **config))


def _featurizer(root: Path, manifest: Mapping[str, Any]) -> Any:
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer, MolecularFeatureSchema
    return MolecularGraphFeaturizer(MolecularFeatureSchema.from_dict(read_json(_input(root, manifest, "feature_schema_path"))))


def _predict(parents: Sequence[BACEParent], oracle: Any, featurizer: Any, split: str, batch_size: int) -> list[dict[str, Any]]:
    from src.eval.bace_frozen_gnn_verification import _graph
    graphs = [_graph(featurizer, smiles=p.smiles, molecule_id=p.parent_id, split=split) for p in parents]
    return oracle.predict_records(graphs, batch_size=batch_size) if graphs else []


def _pairs(parents: Sequence[BACEParent], candidates: Sequence[dict[str, Any]], *, oracle: Any,
           featurizer: Any, distance: Any, split: str, output: Path, binding: str,
           batch_size: int, predictions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    from src.eval.bace_frozen_gnn_verification import _evaluate_rows
    pairs = []
    output.mkdir(parents=True, exist_ok=True)
    for parent in parents:
        key = stable_sha256({"parent": asdict(parent), "candidates": candidates,
                             "split": split, "binding": binding, "checkpoint": oracle.checkpoint_id})
        path = output / f"{key}.json"
        if path.exists():
            checkpoint = read_json(path)
            science = checkpoint["science"]
            if checkpoint.get("binding") != key or checkpoint.get("science_sha256") != stable_sha256(science):
                raise ValueError("Evaluation parent checkpoint hash mismatch")
            current = science["pair_rows"]
        else:
            before = predictions[parent.parent_id]
            cache = {parent.parent_id: {"parent_smiles": parent.smiles,
                "pred_before": before["predicted_label"], "p_before": before["probabilities"]}}
            current, matches = _evaluate_rows([parent], candidates, oracle=oracle, featurizer=featurizer,
                distance_provider=distance, oracle_batch_size=batch_size, split=split,
                oracle_checkpoint_id=oracle.checkpoint_id, parent_prediction_cache=cache)
            if any(row.get("cf_flip") and row.get("distance_ok") is not True for row in matches):
                raise ValueError("STRICT_FLIP_WNODE_NUMERICAL_FAILURE")
            science = {"pair_rows": current, "match_rows": matches}
            atomic_json(path, {"binding": key, "science": science, "science_sha256": stable_sha256(science)})
        matrix_from_pairs([parent.parent_id], candidates, current, root=output, split=split)
        pairs.extend(current)
        atomic_json(output / "progress.json", {"parent_id": parent.parent_id,
            "completed_pair_rows": len(pairs), "expected_pair_rows": len(parents) * len(candidates),
            "split": split, "updated_unix": time.time()})
    return pairs


def benchmark_verification(bundle_root: str | Path, checkpoint_root: str | Path,
                           output_root: str | Path, parent_limit: int = 2,
                           candidate_limit: int = 8) -> dict[str, Any]:
    """Time real train-only deletion/oracle/WNode work; never infer a PASS ETA."""
    from src.oracles.gnn_oracle import GNNOracle
    root, manifest = load_bundle(bundle_root)
    output = Path(output_root).resolve()
    checkpoint = Path(checkpoint_root).resolve(strict=True)
    if any(output == p or p in output.parents or output in p.parents for p in (root, checkpoint)):
        raise ValueError("Timing output must be disjoint from immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    candidates = _candidates(root, manifest)[:candidate_limit]
    train = load_bace_parents(bundle_file(root, manifest, manifest["splits"]["train"]))[:parent_limit]
    oracle = GNNOracle.from_checkpoint(checkpoint_root, device="cpu")
    featurizer = _featurizer(root, manifest)
    begin = time.monotonic()
    distance = _distance(root, manifest, output)
    try:
        predictions = _predict(train, oracle, featurizer, "train", 256)
        pairs = _pairs(train, candidates, oracle=oracle, featurizer=featurizer, distance=distance,
            split="train", output=output / "parents", binding=sha256_file(root / "bundle_manifest.json"),
            batch_size=256, predictions={p.parent_id: r for p, r in zip(train, predictions, strict=True)})
        elapsed = time.monotonic() - begin
        # No-flip classifiers still require a real, non-self WNode timing probe.
        # These diagnostic distances never enter scientific pair rows.
        wnode_seconds = []
        from src.chem.hard_deletion import enumerate_connected_hard_deletions
        for parent in train:
            measured = False
            for candidate in candidates:
                for action in enumerate_connected_hard_deletions(parent.smiles, candidate["canonical_fragment"],
                    parent_id=parent.parent_id, candidate_id=candidate["candidate_id"]):
                    if action.valid and action.residual_smiles and action.residual_smiles != parent.smiles:
                        start = time.monotonic()
                        value = distance.distance(parent.smiles, action.residual_smiles)
                        wnode_seconds.append(time.monotonic() - start)
                        if value.get("ok") is not True:
                            raise ValueError("Train-only WNode timing probe failed")
                        measured = True
                        break
                if measured:
                    break
        selector_seconds = None
        if len(candidates) >= 20:
            matrix = matrix_from_pairs([p.parent_id for p in train], candidates, pairs, root=output, split="train")
            start = time.monotonic()
            _select_frozen_order(matrix, frozen_selector(root, manifest))
            selector_seconds = time.monotonic() - start
        receipt = {"schema_version": SCHEMA, "status": "PASS", "purpose": "TIMING_ONLY",
            "bundle_manifest_sha256": sha256_file(root / "bundle_manifest.json"),
            "checkpoint_sha256": oracle.checkpoint_id,
            "split": "train", "calibration_loaded": False, "test_loaded": False,
            "parent_count": len(train), "candidate_count": len(candidates), "pair_count": len(pairs),
            "seconds": elapsed, "seconds_per_pair": elapsed / len(pairs),
            "strict_flip_pair_count": sum(bool(r["pair_strict_flip"]) for r in pairs),
            "full_evaluation_estimate_is_extrapolation": True, "selector_timing_available": selector_seconds is not None,
            "selector_seconds": selector_seconds, "wnode_nonself_diagnostic_seconds": wnode_seconds,
            "wnode_nonself_diagnostic_only": True,
            "distance_statistics": distance.stats_dict()}
        atomic_json(output / "verification_timing.json", receipt)
        return receipt
    finally:
        distance.close()


def evaluate_with_cpu_admission(*, bundle_root: str | Path, model_roots: Mapping[str, str],
        output_root: str | Path, resume: bool = False, batch_size: int = 256,
        cpu_threads: int = 8, max_cpu_hours: float = 12) -> dict[str, Any]:
    """Runtime after-training gate based only on train probes and split metadata."""
    admission_started = time.monotonic()
    import torch
    torch.set_num_threads(cpu_threads)
    root, manifest = load_bundle(bundle_root)
    if set(model_roots) != set(BACKBONES):
        raise ValueError("CPU admission requires exactly five model roots")
    counts = manifest["split_row_counts"]
    if any(int(counts[s]) <= 0 for s in ("train", "calibration", "test")):
        raise ValueError("Missing frozen split row counts for evaluation admission")
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise FileExistsError("Fresh evaluation root or --resume required")
    output.mkdir(parents=True, exist_ok=True)
    probes = output / "cpu_admission"
    probes.mkdir(exist_ok=True)
    reports = {}
    missing = [name for name, path in model_roots.items() if not Path(path).is_dir()]
    if missing:
        decision = {"state": "READY_GNN_GPU_FALLBACK", "reason": "CLASSIFIER_CPU_TRAINING_INCOMPLETE",
                    "backbones": missing, "core_pass": False, "main_matrix_write": False}
        atomic_json(output / "cpu_admission.json", decision)
        return decision
    for name in BACKBONES:
        target = probes / name
        if (target / "verification_timing.json").is_file():
            report = read_json(target / "verification_timing.json")
        else:
            # A failed partial probe is preserved.  It cannot be silently used
            # as timing evidence or restarted inside the same output root.
            report = benchmark_verification(root, model_roots[name], target, parent_limit=2, candidate_limit=66)
        if (report.get("status") != "PASS" or report.get("test_loaded") is not False
                or report.get("bundle_manifest_sha256") != sha256_file(root / "bundle_manifest.json")
                or report.get("checkpoint_sha256") != sha256_file(Path(model_roots[name]) / "model.pt")):
            raise ValueError("Invalid CPU timing evidence")
        reports[name] = report
    complete = all(r.get("selector_timing_available") and r.get("wnode_nonself_diagnostic_seconds") for r in reports.values())
    projected = None
    if complete:
        pair_seconds = max(float(r["seconds_per_pair"]) + max(r["wnode_nonself_diagnostic_seconds"])
                           for r in reports.values())
        pairs_total = 5 * (int(counts["calibration"]) + int(counts["test"])) * 66
        selection_seconds = max(float(r["selector_seconds"]) / r["parent_count"] for r in reports.values())
        projected = 1.5 * (pairs_total * pair_seconds + 10 * int(counts["calibration"]) * selection_seconds)
    probe_elapsed = time.monotonic() - admission_started
    projected_total = None if projected is None else projected + probe_elapsed
    eligible = projected_total is not None and projected_total <= float(max_cpu_hours) * 3600
    decision = {"state": "CPU_FULL_ELIGIBLE" if eligible else "READY_GNN_GPU_FALLBACK",
        "reason": "MEASURED_RUNTIME_GATE" if complete else "WNode_OR_SELECTOR_TIMING_UNAVAILABLE",
        "projected_evaluation_seconds": projected, "actual_admission_seconds": probe_elapsed,
        "projected_total_seconds": projected_total, "ceiling_hours": max_cpu_hours,
        "safety_factor": 1.5, "all_parents_including_non_source_counted": True,
        "candidate_count": 66, "probe_splits": ["train"], "calibration_loaded": False,
        "test_loaded": False, "main_matrix_write": False, "core_pass": False,
        "probe_receipt_shas": {n: sha256_file(probes / n / "verification_timing.json") for n in BACKBONES}}
    atomic_json(output / "cpu_admission.json", decision)
    if not eligible:
        return decision
    return run_evaluation(bundle_root=root, model_roots=model_roots, output_root=output,
                          resume=True, batch_size=batch_size, cpu_threads=cpu_threads)


def run_evaluation(*, bundle_root: str | Path, model_roots: Mapping[str, str], output_root: str | Path,
                   resume: bool = False, batch_size: int = 256, cpu_threads: int = 8) -> dict[str, Any]:
    """Freeze all native/common calibration orders before first test CSV load."""
    import torch
    from src.oracles.gnn_oracle import GNNOracle, classification_metrics
    if set(model_roots) != set(BACKBONES) or cpu_threads < 1:
        raise ValueError("Evaluation requires all five classifier roots")
    torch.set_num_threads(cpu_threads)
    root, manifest = load_bundle(bundle_root)
    output = Path(output_root).resolve()
    models = {name: Path(path).resolve(strict=True) for name, path in model_roots.items()}
    for source in [root, *models.values()]:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Evaluation output must be disjoint from immutable inputs")
    if output.exists() and not resume:
        raise FileExistsError("Evaluation requires fresh root or explicit same-root resume")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidates = _candidates(root, manifest)
        selector = frozen_selector(root, manifest)
        oracles = {name: GNNOracle.from_checkpoint(path, device="cpu", batch_size=batch_size) for name, path in models.items()}
        schema = read_json(_input(root, manifest, "feature_schema_path"))["schema_sha256"]
        for name, oracle in oracles.items():
            card = read_json(models[name] / "model_card.json")
            model_schema = read_json(models[name] / "feature_schema.json")["schema_sha256"]
            if (oracle.backbone != name or oracle.num_classes != 2 or oracle.source_label != 1
                    or model_schema != schema or card.get("dataset") != "bace"):
                raise ValueError(f"Classifier backbone/dataset/schema drift: {name}")
        expected_gine = (root / manifest["gine_reference_root"]).resolve(strict=True)
        if models["gine"] != expected_gine:
            raise ValueError("GINE must adopt the bundled main checkpoint")
        binding_payload = {"schema": SCHEMA, "execution_commit": manifest["execution_commit"],
            "bundle_sha256": sha256_file(root / "bundle_manifest.json"),
            "oracle_batch_size": batch_size, "cpu_threads": cpu_threads,
            "checkpoints": {name: oracle.checkpoint_id for name, oracle in oracles.items()},
            "temperatures": {name: oracle.temperature for name, oracle in oracles.items()},
            "selector_sha256": selector["input_sha256"], "candidate_universe_sha256": stable_sha256(candidates)}
        binding = stable_sha256(binding_payload)
        run_path = output / "run_manifest.json"
        if run_path.exists() and read_json(run_path).get("binding_sha256") != binding:
            raise ValueError("Resume scientific input/config mismatch")
        if (output / "GNN_CORE_SEED7_PASS").is_file():
            return verify_evaluation(output)
        atomic_json(run_path, {**binding_payload, "binding_sha256": binding, "model_roots": {n: str(p) for n,p in models.items()},
            "main_matrix_write": False, "ChemLLM_loaded": False, "PPO_rerun": False,
            "proposal_fixed": True, "seed": 7, "cpu_only": True})
        featurizer = _featurizer(root, manifest)
        distance = _distance(root, manifest, output)
        try:
            return _run_phases(root, manifest, output, candidates, selector, oracles, models,
                               featurizer, distance, binding, batch_size, classification_metrics)
        finally:
            distance.close()


def _run_phases(root: Path, manifest: Mapping[str, Any], output: Path, candidates: list[dict[str, Any]],
                selector: Mapping[str, Any], oracles: Mapping[str, Any], models: Mapping[str, Path],
                featurizer: Any, distance: Any, binding: str, batch_size: int, metrics_fn: Any) -> dict[str, Any]:
    cal = _all_parents(bundle_file(root, manifest, manifest["splits"]["calibration"]))
    predictions = {name: _predict(cal, oracle, featurizer, "calibration", batch_size) for name, oracle in oracles.items()}
    cohorts = cohort_ids(cal, predictions)
    atomic_json(output / "calibration_cohorts.json", cohorts)
    selections: dict[str, dict[str, list[str]]] = {}
    for name in BACKBONES:
        ids = set(cohorts["native"][name])
        parents = [p for p in cal if p.parent_id in ids]
        pairs = _pairs(parents, candidates, oracle=oracles[name], featurizer=featurizer, distance=distance,
            split="calibration", output=output / name / "calibration" / "parents", binding=binding,
            batch_size=batch_size, predictions={p.parent_id: r for p,r in zip(cal,predictions[name],strict=True)})
        atomic_jsonl(output / name / "calibration" / "pair_matrix.jsonl", pairs)
        selections[name] = {}
        for mode in ("native", "common"):
            cohort = cohorts["native"][name] if mode == "native" else cohorts["common"]
            subset = [row for row in pairs if row["parent_id"] in set(cohort)]
            matrix = matrix_from_pairs(cohort, candidates, subset, root=output, split="calibration")
            sequence, trace = select_calibration(matrix, selector)
            selected = [candidates[i] for i in sequence]
            ids = [row["candidate_id"] for row in selected]
            selections[name][mode] = ids
            frozen = {"state": "FROZEN", "split": "calibration", "test_loaded": False,
                "binding_sha256": binding, "backbone": name, "cohort": mode, "candidate_ids": ids,
                "candidates": selected,
                "selected_rules": [{**row, "rule_id": row["candidate_id"],
                                    "fragment": row["canonical_fragment"]} for row in selected],
                "selector": asdict(selector["variant"]),
                "thresholds": selector["thresholds"].to_dict(), "trace": trace}
            target = output / name / mode / "selected_rules.json"
            if target.exists() and read_json(target) != frozen:
                raise ValueError("Deterministic calibration replay changed frozen selection")
            atomic_json(target, frozen)
    freeze = {"binding_sha256": binding, "all_five_calibration_orders_frozen": True, "test_loaded": False,
              "selections": selections, "source_files": {f"{n}/{m}": sha256_file(output / n / m / "selected_rules.json")
                                                          for n in BACKBONES for m in ("native", "common")}}
    atomic_json(output / "CALIBRATION_FREEZE.json", freeze)
    # This is deliberately the first scientific load of held-out test rows.
    test = _all_parents(bundle_file(root, manifest, manifest["splits"]["test"]))
    if {p.parent_id for p in cal} & {p.parent_id for p in test}:
        raise ValueError("Calibration/test parent identities overlap")
    test_predictions = {name: _predict(test, oracle, featurizer, "test", batch_size) for name, oracle in oracles.items()}
    test_cohorts = cohort_ids(test, test_predictions)
    atomic_json(output / "test_cohorts.json", test_cohorts)
    classifier_table, explanation_table, summaries = [], [], {}
    for name in BACKBONES:
        probs = np.asarray([r["probabilities"] for r in test_predictions[name]], dtype=np.float64)
        labels = np.asarray([p.label for p in test], dtype=np.int64)
        classifier = metrics_fn(labels, probs, num_classes=2)
        classifier["NLL"] = float(-np.mean(np.log(np.clip(probs[np.arange(len(test)), labels], 1e-12, 1))))
        classifier["parameter_count"] = sum(p.numel() for p in oracles[name].model.parameters())
        classifier["trainable_parameter_count"] = sum(p.numel() for p in oracles[name].model.parameters() if p.requires_grad)
        timing = models[name].parent / "training_terminal.json"
        resources = read_json(timing) if timing.is_file() else {"state": "NOT_AVAILABLE"}
        classifier["training_resources"] = resources
        classifier["reported_training_phase_wall_seconds"] = resources.get("elapsed_seconds")
        classifier["peak_RSS_native_units"] = resources.get("rss_peak_native_units")
        classifier["RSS_native_unit"] = resources.get("rss_native_unit")
        classifier["backbone"] = name
        atomic_json(output / name / "classifier_metrics.json", classifier)
        atomic_csv(output / name / "test_classifier_predictions.csv", [
            {"parent_id": p.parent_id, "label": p.label, **record}
            for p, record in zip(test, test_predictions[name], strict=True)])
        classifier_table.append({k: v for k,v in classifier.items() if not isinstance(v, (list,dict))})
        needed = set(selections[name]["native"]) | set(selections[name]["common"])
        selected_candidates = [row for row in candidates if row["candidate_id"] in needed]
        native_ids = set(test_cohorts["native"][name])
        native_parents = [p for p in test if p.parent_id in native_ids]
        pairs = _pairs(native_parents, selected_candidates, oracle=oracles[name], featurizer=featurizer,
            distance=distance, split="test", output=output / name / "test" / "parents", binding=binding,
            batch_size=batch_size, predictions={p.parent_id:r for p,r in zip(test,test_predictions[name],strict=True)})
        atomic_jsonl(output / name / "test" / "pair_matrix.jsonl", pairs)
        atomic_json(output / name / "verification_manifest.json", {
            "state": "PASS", "binding_sha256": binding, "backbone": name,
            "calibration_pair_matrix_sha256": sha256_file(output / name / "calibration" / "pair_matrix.jsonl"),
            "test_pair_matrix_sha256": sha256_file(output / name / "test" / "pair_matrix.jsonl"),
            "calibration_freeze_sha256": sha256_file(output / "CALIBRATION_FREEZE.json"),
            "test_loaded_after_freeze": True, "all_matches_evaluated": True,
            "canonical_candidate_generation_rerun": False, "method": "ours", "dataset": "bace"})
        summaries[name] = {}
        for mode in ("native", "common"):
            ids = test_cohorts["native"][name] if mode == "native" else test_cohorts["common"]
            chosen = set(selections[name][mode])
            mode_candidates = [row for row in candidates if row["candidate_id"] in chosen]
            subset = [row for row in pairs if row["parent_id"] in set(ids) and row["candidate_id"] in chosen]
            matrix = matrix_from_pairs(ids, mode_candidates, subset, root=output, split="test")
            sequence = [matrix.candidate_index[value] for value in selections[name][mode]]
            result = explanation_metrics(matrix, sequence, selector["thresholds"])
            summaries[name][mode] = result
            atomic_json(output / name / mode / "explanation_metrics.json", result)
            atomic_json(output / name / mode / "cohort_manifest.json", {"test_parent_ids": ids,
                "calibration_parent_ids": cohorts["native"][name] if mode == "native" else cohorts["common"],
                "split_sha256": sha256_file(bundle_file(root, manifest, manifest["splits"]["test"])),
                "freeze_sha256": sha256_file(output / "CALIBRATION_FREEZE.json")})
            for key, filename in (("prefix_rows", "figure3_coverage_vs_k.csv"),
                                  ("threshold_rows", "figure4_coverage_vs_threshold.csv")):
                if result[key]:
                    atomic_csv(output / name / mode / filename, result[key])
            if result["prefix_rows"]:
                atomic_csv(output / name / mode / "table2_k10.csv", [result["prefix_rows"][9]])
            explanation_table.append({"backbone": name, "cohort": mode,
                **{k:v for k,v in result.items() if not isinstance(v,(dict,list))}})
    stability = []
    from src.eval.rule_stability import compare_frozen_rule_selections
    for name in BACKBONES:
        for mode in ("native", "common"):
            result = compare_frozen_rule_selections(output / "gine" / mode / "selected_rules.json",
                                                    output / name / mode / "selected_rules.json")
            left = set(summaries["gine"][mode]["covered_parent_ids"])
            right = set(summaries[name][mode]["covered_parent_ids"])
            overlap = len(left & right) / len(left | right) if left | right else None
            result["covered_parent_jaccard"] = overlap
            atomic_json(output / name / mode / "rule_overlap_with_gine.json", result)
            stability.append({"backbone": name, "cohort": mode, "exact_rule_jaccard": result["exact_rule_jaccard"],
                              "covered_parent_jaccard": overlap})
    atomic_csv(output / "gnn_seed7_classifier_table.csv", classifier_table)
    atomic_csv(output / "gnn_seed7_explanation_table.csv", explanation_table)
    atomic_csv(output / "gnn_seed7_rule_stability.csv", stability)
    common_rows = [{"split": split, "parent_id": value} for split, c in (("calibration", cohorts),("test", test_cohorts)) for value in c["common"]]
    if common_rows:
        atomic_csv(output / "gnn_seed7_common_cohort.csv", common_rows)
    else:
        (output / "gnn_seed7_common_cohort.csv").write_text("split,parent_id\n")
    _latex(output / "gnn_seed7_classifier_table.tex", classifier_table, ("backbone", "roc_auc", "balanced_accuracy", "macro_f1", "NLL", "ece", "brier_score", "parameter_count"))
    _latex(output / "gnn_seed7_explanation_table.tex", explanation_table, ("backbone", "cohort", "cohort_size", "CCRCov@10", "CCRCov@20", "AUC_over_K_1_20", "conditional_median_WNode"))
    leaves = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*")
              if p.is_file() and not any(x in p.relative_to(output).parts for x in ("cache", "parents"))
              and p.name not in {"writer.lock", "gnn_seed7_final_audit.json", "GNN_CORE_SEED7_PASS"}}
    audit = {"schema_version": SCHEMA, "state": "PASS", "binding_sha256": binding,
             "backbones": list(BACKBONES), "seed": 7, "proposal_fixed": True, "candidate_count": 66,
             "main_matrix_write": False, "test_loaded_after_calibration_freeze": True,
             "classifier_temperature_refit_on_test": False, "selector_refit_on_test": False,
             "native_cohort_pass": True, "common_cohort_pass": True, "files": leaves}
    atomic_json(output / "gnn_seed7_final_audit.json", audit)
    atomic_marker(output / "GNN_CORE_SEED7_PASS", sha256_file(output / "gnn_seed7_final_audit.json"))
    return audit


def _latex(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    def render(value: Any) -> str:
        if value is None:
            return "N/A"
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        return text.replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}",
             " & ".join(render(c) for c in columns) + r" \\", r"\hline"]
    lines.extend(" & ".join(render(row.get(c)) for c in columns) + r" \\" for row in rows)
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def verify_evaluation(evaluation_root: str | Path) -> dict[str, Any]:
    """Independent output reopen before a result bundle may claim core PASS."""
    root = Path(evaluation_root).resolve(strict=True)
    audit = read_json(root / "gnn_seed7_final_audit.json")
    if audit.get("state") != "PASS" or not (root / "GNN_CORE_SEED7_PASS").is_file():
        raise ValueError("A complete five-backbone audit is required before packaging")
    if (tuple(audit.get("backbones", [])) != BACKBONES or audit.get("seed") != 7
            or audit.get("proposal_fixed") is not True or audit.get("candidate_count") != 66
            or audit.get("main_matrix_write") is not False
            or audit.get("test_loaded_after_calibration_freeze") is not True):
        raise ValueError("Incorrect core-audit scope")
    if (root / "GNN_CORE_SEED7_PASS").read_text().strip() != sha256_file(root / "gnn_seed7_final_audit.json"):
        raise ValueError("Final audit marker hash mismatch")
    for rel, digest in audit["files"].items():
        path = root / rel
        if Path(rel).is_absolute() or ".." in Path(rel).parts or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError("Final evaluation artifact drift")
    freeze = read_json(root / "CALIBRATION_FREEZE.json")
    if freeze.get("all_five_calibration_orders_frozen") is not True or freeze.get("test_loaded") is not False:
        raise ValueError("Missing pre-test calibration freeze")
    run = read_json(root / "run_manifest.json")
    if run.get("binding_sha256") != audit["binding_sha256"] or run.get("main_matrix_write") is not False:
        raise ValueError("Run/audit scientific binding mismatch")
    for name in BACKBONES:
        for mode in ("native", "common"):
            rules_path = root / name / mode / "selected_rules.json"
            rules = read_json(rules_path)
            ids = rules["candidate_ids"]
            if (len(ids) != 20 or len(set(ids)) != 20 or rules.get("test_loaded") is not False
                    or ids != freeze["selections"][name][mode]
                    or sha256_file(rules_path) != freeze["source_files"][f"{name}/{mode}"]):
                raise ValueError("Frozen rule selection drift")
            report = read_json(root / name / mode / "explanation_metrics.json")
            if report.get("state") not in {"PASS", "VALID_EMPTY_COHORT"}:
                raise ValueError("An explanation route is incomplete")
            cohort = read_json(root / name / mode / "cohort_manifest.json")
            if report["cohort_size"] != len(cohort["test_parent_ids"]):
                raise ValueError("Explanation cohort denominator drift")
    return audit


def package_evaluation(*, evaluation_root: str | Path, output_root: str | Path,
                       environment_manifest: str | Path, execution_commit: str) -> dict[str, Any]:
    """Package only sealed ablation outputs plus the exact classifier bundles."""
    root = Path(evaluation_root).resolve(strict=True)
    if not (root / "writer.lock").is_file():
        raise ValueError("A complete five-backbone audit/writer closure is required before packaging")
    with (root / "writer.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        audit = verify_evaluation(root)
        return _package_locked(root=root, audit=audit, output_root=output_root,
            environment_manifest=environment_manifest, execution_commit=execution_commit)


def _package_locked(*, root: Path, audit: Mapping[str, Any], output_root: str | Path,
                    environment_manifest: str | Path, execution_commit: str) -> dict[str, Any]:
    if len(execution_commit) != 40 or any(c not in "0123456789abcdef" for c in execution_commit):
        raise ValueError("Package execution commit must be an exact Git SHA")
    run = read_json(root / "run_manifest.json")
    if run.get("execution_commit") != execution_commit:
        raise ValueError("Package commit differs from science commit")
    output = Path(output_root).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Result package output must be disjoint from evaluation")
    output.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {f"evaluation/{rel}": root / rel for rel in audit["files"]}
    paths["evaluation/gnn_seed7_final_audit.json"] = root / "gnn_seed7_final_audit.json"
    paths["evaluation/GNN_CORE_SEED7_PASS"] = root / "GNN_CORE_SEED7_PASS"
    for name, raw in run["model_roots"].items():
        model = Path(raw).resolve(strict=True)
        checksum = model / "sha256sums.txt"
        for line in checksum.read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            rel = rel.lstrip("*")
            path = model / rel
            if Path(rel).is_absolute() or ".." in Path(rel).parts or path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("Classifier bundle drift during packaging")
            paths[f"classifiers/{name}/{rel}"] = path
        paths[f"classifiers/{name}/sha256sums.txt"] = checksum
        training_receipt = model.parent / "training_terminal.json"
        if name != "gine" and training_receipt.is_file():
            paths[f"classifiers/{name}/cpu_training_terminal.json"] = training_receipt
    paths["environment_manifest.json"] = Path(environment_manifest).resolve(strict=True)
    inventory = {name: {"sha256": sha256_file(path), "size": path.stat().st_size} for name,path in paths.items()}
    metadata = {"schema_version": SCHEMA, "state": "PASS", "execution_commit": execution_commit,
                "main_matrix_write": False, "files": inventory, "core_audit_sha256": sha256_file(root / "gnn_seed7_final_audit.json")}
    atomic_json(output / "package_manifest.json", metadata)
    partial = output / "bace_gnn_seed7.tar.gz.partial"
    with tarfile.open(partial, "w:gz") as archive:
        for name, path in sorted(paths.items()):
            archive.add(path, arcname=name, recursive=False)
        archive.add(output / "package_manifest.json", arcname="package_manifest.json", recursive=False)
    # Independent stream hashes detect corruption without extracting payloads.
    with tarfile.open(partial, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("Invalid package archive member")
            source = archive.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            expected = sha256_file(output / "package_manifest.json") if member.name == "package_manifest.json" else inventory[member.name]["sha256"]
            if digest.hexdigest() != expected:
                raise ValueError("Package round-trip hash mismatch")
    archive_path = output / "bace_gnn_seed7.tar.gz"
    os.replace(partial, archive_path)
    receipt = {"state": "PASS", "bundle": str(archive_path), "sha256": sha256_file(archive_path),
               "bytes": archive_path.stat().st_size, "main_matrix_write": False}
    atomic_json(output / "result_package.json", receipt)
    atomic_marker(output / "PASS", receipt["sha256"])
    return receipt
