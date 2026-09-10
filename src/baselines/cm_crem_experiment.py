"""BACE-only CM-CReM stage driver. Frozen data stages, not a general controller."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from src.baselines.cm_crem_runtime import atomic_json, checked_root, digest, file_sha, read_json, require_compute_node, utc_now

METHOD = "CM-CReM-Global-Budgeted-v1"
UPSTREAM = "b5816b502cde00ee24c652a02cbc54664583f773"
HPC_SCOPE = "/share/home/u20526/czx"
TERMINALS = {"GENERATED", "NO_NATIVE_REPLACEMENT", "NO_REPLACEABLE_CONTEXT", "TIMEOUT_BUDGETED"}


def validate_contract(spec: dict) -> None:
    checks = [
        (spec["method_id"], METHOD, "method"),
        (spec["campaign"]["seed"], 7, "seed"),
        (spec["campaign"]["write_original_main_matrix"], False, "matrix isolation"),
        (spec["campaign"]["primary"]["dataset"], "bace", "BACE first"),
        (spec["campaign"]["primary"]["oracle"], "original_frozen_gine", "original oracle"),
        (spec["generation"]["radius"], 1, "radius"),
        (spec["generation"]["min_max_inc"], 3, "increment"),
        (spec["generation"]["max_replacements_per_component"], 64, "component budget"),
        (spec["generation"]["raw_outputs_retained_per_parent"], 128, "raw budget"),
        (spec["generation"]["parent_wall_limit_seconds"], 900, "parent wall budget"),
        (spec["generation"]["top_level_calls_per_parent"], 1, "one call"),
        (spec["pool"]["max_candidates"], 2000, "pool budget"),
        (spec["summary"]["k_max"], 20, "K"),
        (spec["summary"]["at_most_k"], True, "at most K"),
        (spec["summary"]["selector"], "coverage_greedy_with_cost_tie_break", "selector"),
        (spec["upstream"]["commit"], UPSTREAM, "author pin"),
    ]
    for actual, expected, role in checks:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Frozen CM contract changed: {role}: {actual!r} != {expected!r}")
    if spec["resolved_oracle"]["backbone"] != "gine":
        raise ValueError("GIN/A+ must not enter original-GINE CM baseline")
    expected = spec["campaign"]["primary"]
    reference = spec["resolved_evaluation"]
    for key, field in (("theta", "theta_expected"), ("cap", "cap_expected")):
        if reference[key] != expected[field]:
            raise ValueError(f"Frozen BACE {key} does not match reference")
    for split, n in (("calibration", 66), ("test", 141)):
        if spec["resolved_parents"][split]["count"] != n:
            raise ValueError(f"Base cohort {split} count differs; do not trim/pad")


def science_identity(spec: dict) -> str:
    # Execution paths/Slurm IDs do not alter a deterministic scientific seed.
    binding = {k: spec[k] for k in ("method_id", "campaign", "attribution", "generation", "pilot", "pool", "summary", "metrics")}
    binding["upstream"] = {"commit": UPSTREAM, "environment": spec["upstream"]["generator_environment"],
                           "database_url": spec["upstream"]["database"]["url"]}
    binding["oracle"] = {k: v for k, v in spec["resolved_oracle"].items() if not k.endswith("dir") and k != "device"}
    binding["wnode"] = {k: v for k, v in spec["resolved_wnode"].items() if k.endswith("sha256") or k in ("feature_cost", "node_mass", "size_penalty_beta", "encoder_type")}
    binding["cohorts"] = {k: {p: v for p, v in row.items() if p not in {"path"}} for k, row in spec["resolved_parents"].items()}
    binding["evaluation"] = spec["resolved_evaluation"]
    return digest(binding)


def resolve_spec(template: Path, bindings: Path, output: Path) -> dict:
    import yaml
    spec = yaml.safe_load(template.read_text())
    spec["upstream"]["commit"] = UPSTREAM
    resolved = read_json(bindings)
    for key in ("resolved_oracle", "resolved_wnode", "resolved_parents", "resolved_evaluation", "execution", "source_receipts"):
        spec[key] = resolved[key]
    spec["resolved_attribution"] = {"source": "REVIEWED_UPSTREAM_FUNCTIONS", "upstream_commit": UPSTREAM,
        "node_layer": spec["resolved_oracle"]["node_layer"],
        "score": "calibrated_probability_source_class", "rounding": "max_1_floor_n_div_5",
        "ring_policy": "one_union_rings_intersecting_initial_mask", "relu": False,
        "atom_tie_break": "ascending_original_atom_index", "transport_schema": "cm_crem_parent_v2"}
    spec["campaign"]["start_time_utc"] = resolved["start_time_utc"]
    spec["template_sha256"] = file_sha(template)
    spec["bindings_sha256"] = file_sha(bindings)
    validate_contract(spec)
    spec["science_hash"] = science_identity(spec)
    atomic_json(output, spec, immutable=True)
    return spec


def load_parent_rows(spec: dict, split: str, root: Path) -> list[dict]:
    if split == "test" and not (root / "selection_freeze.json").is_file():
        raise ValueError("Test data cannot be opened before calibration selection freeze")
    if split == "calibration" and not (root / "pool_freeze.json").is_file():
        raise ValueError("Calibration data cannot be opened before train pool freeze")
    binding = spec["resolved_parents"][split]
    path = Path(binding["path"])
    if file_sha(path) != binding["sha256"]:
        raise ValueError(f"{split} source content changed")
    if binding.get("format") == "csv":
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif binding.get("format") == "jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = read_json(path)
        for key in binding.get("rows_key", []):
            rows = rows[key]
    id_field = binding.get("id_field", "parent_id")
    smiles_field = binding.get("smiles_field", "smiles")
    if binding.get("ordered_ids") is not None:
        by_id = {str(row[id_field]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate parent identity in source")
        rows = [by_id[str(identity)] for identity in binding["ordered_ids"]]
    result = [{**row, "parent_id": str(row[id_field]), "smiles": row[smiles_field], "split": split} for row in rows]
    if len(result) != binding["count"] or len({r["parent_id"] for r in result}) != len(result):
        raise ValueError(f"{split} cohort missing/duplicate rows")
    if digest([r["parent_id"] for r in result]) != binding["ordered_ids_sha256"]:
        raise ValueError(f"{split} parent order differs")
    return result


def pilot_indices(rows: list[dict], science_hash: str) -> tuple[list[int], dict]:
    """Structural train-only strata, no prediction/distance outcome ranking."""
    from rdkit import Chem
    info = []
    for i, row in enumerate(rows):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"Invalid authoritative parent {row['parent_id']}")
        info.append((mol.GetNumHeavyAtoms(), row["parent_id"], i,
                     bool(mol.GetRingInfo().NumRings()), len(Chem.GetMolFrags(mol))))
    info.sort()
    if len(info) < 32:
        raise ValueError("Fewer than 32 eligible train parents; no smaller pilot substituted")
    selected, bins = [], []
    for q in range(4):
        group = info[len(info)*q//4:len(info)*(q+1)//4]
        ordered = sorted(group, key=lambda x: (digest([science_hash, x[1]]), x[1]))
        # Guarantee examples of available structural categories in each size bin.
        choose = []
        for predicate in (lambda r: r[4] > 1, lambda r: r[3], lambda r: not r[3]):
            match = next((r for r in ordered if predicate(r) and r not in choose), None)
            if match is not None:
                choose.append(match)
        choose.extend(r for r in ordered if r not in choose)
        choose = choose[:8]
        selected.extend(r[2] for r in choose)
        bins.append({"quartile": q, "parent_ids": [r[1] for r in choose],
                     "heavy_atoms": [r[0] for r in choose], "has_ring": [r[3] for r in choose],
                     "molecular_components": [r[4] for r in choose]})
    return selected, {"quartiles": bins, "test_loaded": False,
                      "multicomponent_train_parents_available": sum(r[4] > 1 for r in info)}


def reusable_attributions(spec: dict) -> dict[str, dict]:
    """Adopt hash-sealed gradients from one terminal pilot; repair only transport."""
    reuse = spec["execution"].get("attribution_reuse")
    if not reuse:
        return {}
    source = checked_root(reuse["root"], HPC_SCOPE)
    old_spec = read_json(source / "spec.json")
    if old_spec["science_hash"] != spec["science_hash"] or old_spec["execution"]["execution_commit"] != reuse["execution_commit"]:
        raise ValueError("Attribution reuse source science/execution binding differs")
    job = str(reuse["job_id"])
    if not job.isdecimal():
        raise ValueError("Attribution reuse needs a real numeric Slurm producer")
    state = subprocess.check_output(["sacct", "-X", "-j", job, "--noheader", "--parsable2", "--format=JobIDRaw,State"], text=True)
    matches = [line.split("|") for line in state.splitlines() if line.split("|")[0] == job]
    if len(matches) != 1 or matches[0][1] not in {"FAILED", "COMPLETED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
        raise ValueError("Old attribution producer is not confirmed terminal; no adoption")
    receipts = [json.loads(line) for line in (source / "producer_receipts" / (job+".jsonl")).read_text().splitlines()]
    result = {}
    from rdkit import Chem
    from src.baselines.cm_crem_generation import make_parent_request
    for receipt in receipts:
        name = receipt.get("path", "")
        if not name.startswith("attribution_units/"):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Attribution unit escaped its old root")
        path = source / name
        if file_sha(path) != receipt["sha256"] or receipt["execution_commit"] != reuse["execution_commit"]:
            raise ValueError("Attribution unit differs from its original producer receipt")
        row = read_json(path)
        if (row.get("science_hash") != spec["science_hash"] or row.get("weights_bn_rng_unchanged") is not True
                or row.get("oracle_weight_sha256") != spec["resolved_oracle"]["model_sha256"]
                or row.get("temperature_sha256") != spec["resolved_oracle"]["temperature_sha256"]):
            raise ValueError("Unsealed/unbound gradient cannot be reused")
        before = row["generation_request"]
        after = make_parent_request(row["parent_id"], Chem.MolFromSmiles(row["input_smiles"]), before["selected_atom_indices"])
        if any(before[k] != after[k] for k in ("atom_order_sha256", "selected_atom_indices", "effective_atom_indices")):
            raise ValueError("Transport repair altered the original graph/mask")
        row["generation_request"] = after
        row["transport_adoption"] = {"source_path": str(path), "source_sha256": receipt["sha256"],
                                     "source_job_id": job, "new_gradcam_computation": False,
                                     "only_transport_schema_updated": True}
        result[row["parent_id"]] = row
    return result


class Experiment:
    def __init__(self, spec_path: Path, root: Path):
        self.spec_path = spec_path.resolve()
        self.spec = read_json(spec_path)
        validate_contract(self.spec)
        if self.spec["science_hash"] != science_identity(self.spec):
            raise ValueError("Science contract digest mismatch")
        self.root = checked_root(root, HPC_SCOPE)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sha = self.spec["science_hash"]

    def put(self, name: str, data: dict) -> dict:
        record = {"science_hash": self.sha, **data}
        atomic_json(self.root / name, record, immutable=True)
        # Compact append-only per-job producer records, not a new trust root.
        # Hash newly produced artifacts once; do not rescan old model packages.
        journal = self.root / "producer_receipts" / (str(os.environ.get("SLURM_JOB_ID", "control")) + ".jsonl")
        journal.parent.mkdir(exist_ok=True)
        item = {"path": name, "sha256": file_sha(self.root / name), "science_hash": self.sha,
                "execution_commit": self.spec["execution"].get("execution_commit"),
                "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(), "created_at": utc_now()}
        with journal.open("a") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def get(self, name: str) -> dict:
        record = read_json(self.root / name)
        if record.get("science_hash") != self.sha:
            raise ValueError(f"Stage contract mismatch: {name}")
        return record

    def oracle(self):
        from src.baselines.cm_crem_oracle import FrozenCMOracle
        return FrozenCMOracle.from_resolved(self.spec)

    def distance(self):
        from src.baselines.cm_crem_oracle import FrozenCMWNode
        return FrozenCMWNode.from_resolved(self.spec)

    def stage_preflight(self) -> dict:
        e = self.spec["execution"]
        missing = [str(p) for p in (e.get("generator_python"), e.get("database_path"), e.get("database_receipt"), e.get("upstream_root"), e.get("generator_environment_receipt")) if not p or not Path(p).exists()]
        return {"status": "ASSET_BLOCKED" if missing else "ASSETS_PRESENT_NOT_PILOT_PASS",
                "missing_assets": missing, "database_receipt": e.get("database_receipt"),
                "original_main_matrix_writes": False, "science_completed": False}

    def stage_attribution(self, *, pilot_only: bool = False) -> dict:
        require_compute_node()
        if not pilot_only:
            self.require_pilot()
        output = "pilot/oracle.json" if pilot_only else "attribution.json"
        if (self.root / output).exists():
            return self.get(output)
        started = time.monotonic()
        from src.baselines.cm_crem_runtime import cgroup_memory
        memory_start = cgroup_memory()
        oracle = self.oracle()
        rows = load_parent_rows(self.spec, "train", self.root)
        predictions = oracle.predict_rows(rows, split="train")
        eligible = [row for row in predictions if row["predicted_label"] == 1]
        indices, strata = pilot_indices(eligible, self.sha)
        selected = [eligible[i] for i in indices] if pilot_only else eligible
        self.put("pilot/design.json" if pilot_only else "attribution_design.json",
                 {"parents": selected, "train_predictions": predictions, "strata": strata,
                  "train_proposal_count": len(rows), "train_predicted_source_count": len(eligible),
                  "test_loaded": False, "selection_uses_coverage": False})
        records = []
        reused = reusable_attributions(self.spec)
        adopted = 0
        for i, row in enumerate(selected):
            part = f"attribution_units/{digest(row['parent_id'])[:20]}.json"
            if (self.root / part).exists():
                record = self.get(part)
            elif row["parent_id"] in reused:
                old = reused[row["parent_id"]]
                if old["input_smiles"] != row["smiles"]:
                    raise ValueError("Reusable attribution has a different original input")
                record = self.put(part, old)
                adopted += 1
            else:
                try:
                    record = self.put(part, oracle.attribute_train_parent(row))
                except Exception as error:
                    self.put(f"failures/attribution-{i}-{os.environ['SLURM_JOB_ID']}.json",
                             {"parent_id": row["parent_id"], "parent_input": row,
                              "error_type": type(error).__name__, "error": str(error),
                              "completed_before_failure": len(records), "stage": output})
                    raise
            records.append(record)
            atomic_json(self.root / "progress.json", {"stage": output, "completed_parent_units": i+1,
                         "total_parent_units": len(selected), "updated_at": utc_now(), "job_id": os.environ.get("SLURM_JOB_ID")})
        result = {"status": "ATTRIBUTION_COMPLETE", "parents": selected, "records": records,
                  "train_predictions": predictions,
                  "train_proposal_count": len(rows), "train_predicted_source_count": len(eligible),
                  "strata": strata, "seconds": time.monotonic()-started, "test_loaded": False,
                  "job_id": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
                  "sealed_attribution_units_reused": adopted,
                  "memory_start": memory_start, "memory_after_attribution": cgroup_memory()}
        if pilot_only:
            # Asset-independent portion only. These are not generated recourses.
            wnode = self.distance()
            encoded = wnode.encode_rows(selected, featurizer=oracle.featurizer)
            if len(encoded) < 12:
                raise ValueError("Insufficient distinct train graphs for 64 nonself pairs")
            from src.baselines.cm_crem_oracle import raw_distance_record
            pairs = []
            for offset in range(1, len(encoded)):
                for i in range(len(encoded)-offset):
                    start = time.monotonic()
                    row = raw_distance_record(encoded[i], encoded[i+offset], numerical_contract=self.spec["resolved_wnode"])
                    pairs.append({**row, "seconds": time.monotonic()-start})
                    if len(pairs) == 64:
                        break
                if len(pairs) == 64:
                    break
            result.update(nonself_pairs=pairs, encoded_graphs=encoded,
                          distance_sample_kind="TRAIN_PARENT_TO_DISTINCT_TRAIN_PARENT",
                          status="PILOT_ORACLE_DISTANCE_COMPLETE", generation_complete=False,
                          complete_end_to_end_pilot=False, full_campaign_eta_hours=None)
        import resource
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        result["memory_end"] = cgroup_memory()
        result["total_stage_seconds"] = time.monotonic() - started
        return self.put(output, result)

    def stage_generate(self, *, shard: int = 0, shards: int = 1, pilot_only: bool = False) -> dict:
        require_compute_node()
        if not pilot_only:
            self.require_pilot()
        preflight = self.stage_preflight()
        if preflight["missing_assets"]:
            raise FileNotFoundError(f"ASSET_BLOCKED: {preflight['missing_assets']}")
        e = self.spec["execution"]
        if sys.executable != e["generator_python"] and Path(sys.executable).resolve() != Path(e["generator_python"]).resolve():
            raise ValueError("generate must use its isolated fixed-version Python, never original GINE environment")
        db_receipt = read_json(e["database_receipt"])
        from src.baselines.cm_crem_assets import stage_static_database, validate_database_source, prepare_job_scratch
        source = validate_database_source(self.spec, db_receipt)
        scratch = prepare_job_scratch(self.root, required_bytes=db_receipt['uncompressed_bytes'], reserve_bytes=2*1024**3)
        if scratch['status'] != 'JOB_SCRATCH_READY':
            raise RuntimeError(scratch)
        staged = stage_static_database(e["database_path"], e["database_receipt"], reserve_bytes=2*1024**3,
                                       expected_url=source['actual_source_url'], scratch_receipt=scratch)
        self.put(f"assets/database-local-{os.environ['SLURM_JOB_ID']}.json", staged)
        if staged["status"] != "LOCAL_DATABASE_READY":
            raise RuntimeError(f"BLOCKED_LOCAL_DATABASE_STAGING: {staged.get('reason')}")
        from src.baselines.cm_crem_generation import generate_parent
        inputs = self.get("pilot/oracle.json" if pilot_only else "attribution.json")
        records = []
        for index, attr in enumerate(inputs["records"]):
            if index % shards != shard:
                continue
            path = f"generation_units/{digest(attr['parent_id'])[:20]}.json"
            if (self.root / path).exists():
                records.append(self.get(path))
                continue
            if attr["status"] == "NO_REPLACEABLE_CONTEXT":
                result = {"parent_id": attr["parent_id"], "status": "NO_REPLACEABLE_CONTEXT", "retained_raw": [], "top_level_calls": 0}
            else:
                log = self.root / "logs" / f"parent-{digest(attr['parent_id'])[:20]}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                result = generate_parent(attr["generation_request"], {"database_path": staged["database_path"],
                         "upstream_root": e["upstream_root"], "science_hash": self.sha, "parent_wall_limit_seconds": 900}, log_path=log)
            result["database_uncompressed_sha256"] = db_receipt["uncompressed_sha256"]
            result["database_staging_manifest"] = staged["manifest_path"]
            if result["status"] not in TERMINALS:
                self.put(f"failures/generate-{index}-{os.environ['SLURM_JOB_ID']}.json", result)
                raise RuntimeError(f"Generation {attr['parent_id']}: {result['status']}: {result.get('error')}")
            records.append(self.put(path, result))
        name = f"{'pilot' if pilot_only else 'full'}/generation-shard-{shard}.json"
        import resource
        rss_scale = 1 if sys.platform == "darwin" else 1024
        return self.put(name, {"status": "GENERATION_SHARD_COMPLETE", "shard": shard, "shards": shards,
                             "parent_count": len(records), "parent_ids": [r["parent_id"] for r in records],
                             "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale,
                             "largest_child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * rss_scale,
                             "rss_scope": "PARENT_PLUS_LARGEST_SERIAL_WORKER_NOT_CGROUP_TOTAL"})

    def stage_filter(self, *, pilot_only: bool = False) -> dict:
        require_compute_node()
        output = "pilot/pool.json" if pilot_only else "pool_freeze.json"
        if (self.root / output).exists():
            return self.get(output)
        started = time.monotonic()
        oracle = self.oracle()
        model_load_seconds = time.monotonic() - started
        attrs = self.get("pilot/oracle.json" if pilot_only else "attribution.json")
        accepted, funnels, reused_units = {}, [], 0
        for parent in attrs["parents"]:
            generation = self.get(f"generation_units/{digest(parent['parent_id'])[:20]}.json")
            unit = f"filter_units/{digest(parent['parent_id'])[:20]}.json"
            if (self.root / unit).exists():
                result = self.get(unit)
                reused_units += 1
            else:
                result = self.put(unit, oracle.filter_generated(parent, generation))
            funnels.append({k: v for k, v in result.items() if k != "accepted"})
            for graph in result["accepted"]:
                cid = graph["candidate_id"]
                if cid in accepted:
                    accepted[cid]["origins"].extend(graph["origins"])
                else:
                    accepted[cid] = graph
        ordered = sorted(accepted, key=lambda cid: (digest([self.sha, 7, cid]), cid))
        candidates = [accepted[cid] for cid in ordered[:2000]]
        payload = {"status": "TRAIN_POOL_FROZEN", "candidate_ids": [c["candidate_id"] for c in candidates],
                   "candidates": candidates, "unique_strict_flip_count_before_cap": len(accepted),
                   "retained_count": len(candidates), "funnels": funnels, "test_loaded": False,
                   "calibration_loaded": False, "selection": "DETERMINISTIC_HASH_BEFORE_CALIBRATION"}
        payload["pool_sha256"] = digest(payload)
        result = self.put(output, payload)
        import resource
        self.put("pilot/filter_timing.json" if pilot_only else "filter_timing.json",
                 {"status": "FILTER_TIMING_MEASURED", "parent_count": len(attrs["parents"]),
                  "reused_sealed_parent_units": reused_units, "model_load_seconds": model_load_seconds,
                  "total_filter_and_durable_io_seconds": time.monotonic() - started,
                  "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1 if sys.platform == "darwin" else 1024), "test_loaded": False})
        return result

    def require_pilot(self) -> dict:
        receipt = self.get("pilot/final_receipt.json")
        if receipt.get("status") != "PILOT_ENGINEERING_PROTOCOL_ACCEPTED":
            raise ValueError("Full CM generation requires complete end-to-end pilot and measured cost admission")
        return receipt

    def stage_pilot_closeout(self) -> dict:
        require_compute_node()
        import numpy as np
        oracle_result = self.get("pilot/oracle.json")
        pool = self.get("pilot/pool.json")
        filter_timing = self.get("pilot/filter_timing.json")
        generation_timing = self.get("pilot/generation-shard-0.json")
        if filter_timing.get("status") != "FILTER_TIMING_MEASURED" or filter_timing.get("parent_count") != 32:
            raise ValueError("Complete pilot needs actual32-parent model/filter/durable-I/O timing")
        parents = oracle_result["parents"]
        if len(parents) != 32:
            raise ValueError("Complete pilot must contain exactly32 prespecified train parents")
        generated = [self.get(f"generation_units/{digest(p['parent_id'])[:20]}.json") for p in parents]
        if any(g["status"] not in TERMINALS for g in generated):
            raise ValueError("Unresolved generation error is not a complete pilot")
        encoding_started = time.monotonic()
        oracle, distance = self.oracle(), self.distance()
        parent_encoded = distance.encode_rows(parents, featurizer=oracle.featurizer)
        prototype_encoded = distance.encode_rows(pool["candidates"], featurizer=oracle.featurizer)
        encoding_seconds = time.monotonic() - encoding_started
        from src.baselines.cm_crem_oracle import raw_distance_record
        raw_pairs = []
        for left in parent_encoded:
            for right in prototype_encoded:
                if left["candidate_id"] == right["candidate_id"]:
                    continue
                t = time.monotonic()
                row = raw_distance_record(left, right, numerical_contract=self.spec["resolved_wnode"])
                raw_pairs.append({**row, "seconds": time.monotonic()-t})
                if len(raw_pairs) >= 64:
                    break
            if len(raw_pairs) >= 64:
                break
        # If the pilot has genuinely too few CF prototypes, the mandatory real
        # nonself train-pair benchmark remains evidence, not a fabricated CF pair.
        measured = raw_pairs or oracle_result["nonself_pairs"]
        if len(oracle_result["nonself_pairs"]) < 64:
            raise ValueError("Required64 real nonself WNode samples missing")
        generation_times = [float(g.get("parent_wall_seconds", 0.0)) for g in generated]
        distance_p90 = float(np.quantile([r["seconds"] for r in measured], .9))
        gen_p90 = float(np.quantile(generation_times, .9))
        measured_io_started = time.monotonic()
        self.put("pilot/closeout_pair_timing.json", {"raw_pairs": raw_pairs,
                 "original_nonself_pair_receipt": "pilot/oracle.json",
                 "encoding_including_load_seconds": encoding_seconds,
                 "prototype_encoding_count": len(prototype_encoded), "test_loaded": False})
        measured_durable_io_seconds = time.monotonic() - measured_io_started
        # Worst-case pool=2000, two sequential CPU lanes; never adjust science
        # parameters based on the pilot's coverage, validity or rank.
        full_generation = oracle_result["train_predicted_source_count"] * gen_p90 / 2
        full_pairs = 66 * 2000 + 141 * 20
        # Include observed filter/model and encoding/serialization rather than
        # extrapolating only CM generation or OT. These are conservative cost
        # bounds, never observed complete-campaign wall time or scientific tuning.
        full_filter = filter_timing["total_filter_and_durable_io_seconds"] * oracle_result["train_predicted_source_count"] / 32
        full_encoding = encoding_seconds * (2000 + 66 + 141) / max(1, len(parent_encoded) + len(prototype_encoded))
        full_attribution = oracle_result["seconds"] * oracle_result["train_predicted_source_count"] / 32
        full_durable_io = measured_durable_io_seconds * full_pairs / max(1, len(raw_pairs))
        stages = {"generation_per_lane": full_generation, "filter": full_filter,
                  "encoding": full_encoding, "attribution": full_attribution,
                  "distance_per_lane": full_pairs * distance_p90 / 2,
                  "durable_serialization_and_io": full_durable_io}
        estimated = 2 * sum(stages.values()) / 3600 + 12
        from datetime import datetime, timezone
        elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(self.spec["campaign"]["start_time_utc"].replace("Z", "+00:00"))).total_seconds()/3600
        horizon_ok = estimated + elapsed <= self.spec["campaign"]["planning_horizon_hours"]
        stage_walltime_ok = 2 * max(stages.values()) <= 12 * 3600
        import resource
        gen_peak = int(generation_timing["process_peak_rss_bytes"]) + int(generation_timing["largest_child_peak_rss_bytes"])
        peak = max(gen_peak, int(filter_timing["process_peak_rss_bytes"]), int(oracle_result["process_peak_rss_bytes"]),
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        # Compute-node RSS is observed; unavailable cgroup limits are not invented.
        memory_ok = 2 * peak + 4 * 1024**3 <= 32 * 1024**3
        accepted = horizon_ok and stage_walltime_ok and memory_ok
        blocked = "BLOCKED_CAMPAIGN_COST" if not horizon_ok else (
            "BLOCKED_STAGE_WALLTIME_NEEDS_BOUNDED_SHARDS" if not stage_walltime_ok else "BLOCKED_PILOT_MEMORY_ADMISSION")
        return self.put("pilot/final_receipt.json", {"status": "PILOT_ENGINEERING_PROTOCOL_ACCEPTED" if accepted else blocked,
                       "parent_count": 32, "real_nonself_wnode_pairs": len(oracle_result["nonself_pairs"]),
                       "generated_prototype_pairs": raw_pairs, "generation_seconds": generation_times,
                       "generation_p50_seconds": float(np.median(generation_times)), "generation_p90_seconds": gen_p90,
                       "wnode_p90_seconds": distance_p90, "full_campaign_eta_hours_conservative": estimated,
                       "elapsed_hours": elapsed, "complete_end_to_end_pilot": True,
                       "phase_estimates_seconds_not_actual_full_run": stages,
                       "filter_timing": filter_timing, "encoding_including_load_seconds": encoding_seconds,
                       "generation_memory_receipt": generation_timing,
                       "measured_durable_serialization_io_seconds": measured_durable_io_seconds,
                       "process_peak_rss_bytes": peak, "slurm_requested_memory_bytes": 32 * 1024**3,
                       "memory_admission": memory_ok, "per_job_walltime_admission": stage_walltime_ok,
                       "safety_factor": 2, "additional_campaign_margin_hours": 12,
                       "generation_statuses": [r["status"] for r in generated], "test_loaded": False,
                       "scientific_parameters_tuned": False})

    def stage_encode(self) -> dict:
        require_compute_node()
        pool = self.get("pool_freeze.json")
        oracle = self.oracle()
        encoded = self.distance().encode_rows(pool["candidates"], featurizer=oracle.featurizer)
        return self.put("pool_encodings.json", {"status": "ENCODED", "pool_sha256": pool["pool_sha256"], "records": encoded})

    def stage_matrix(self, split: str, shard: int = 0, shards: int = 1) -> dict:
        require_compute_node()
        pool = self.get("pool_freeze.json")
        all_candidates = pool["candidates"]
        if split == "test":
            from src.baselines.cm_crem_selection import SelectionFreeze
            freeze = SelectionFreeze.from_dict(self.get("selection_freeze.json"))
            chosen = freeze.selected_candidate_ids
            if freeze.frozen_pool_sha256 != pool["pool_sha256"]:
                raise ValueError("Frozen selector has a different train pool")
            by_id = {c["candidate_id"]: c for c in all_candidates}
            all_candidates = [by_id[cid] for cid in chosen]
        parents = load_parent_rows(self.spec, split, self.root)
        oracle, wnode = self.oracle(), self.distance()
        prototypes = oracle.predict_rows(all_candidates, split="frozen_train_prototype")
        if any(c["predicted_label"] != 0 for c in prototypes):
            raise ValueError("Frozen target prototype prediction changed")
        encoded = {r["candidate_id"]: r for r in self.get("pool_encodings.json")["records"]}
        from src.baselines.cm_crem_oracle import raw_distance_record, full_graph_pair
        completed = []
        for i, parent in enumerate(parents):
            if i % shards != shard:
                continue
            name = f"{split}/parents/{digest(parent['parent_id'])[:20]}.json"
            if (self.root / name).exists():
                completed.append(self.get(name)["parent_id"])
                continue
            prediction = oracle.predict_rows([parent], split=split)[0]
            distances = []
            left = None
            if prediction["predicted_label"] == 1 and prototypes:
                left = wnode.encode_rows([parent], featurizer=oracle.featurizer)[0]
            for prototype in prototypes:
                raw = None if prediction["predicted_label"] != 1 else raw_distance_record(left, encoded[prototype["candidate_id"]], numerical_contract=self.spec["resolved_wnode"])
                pair = full_graph_pair(prediction, prototype, raw_distance=raw)
                distances.append({**pair, "raw_distance": raw, "pair_status": "OK" if pair["strict_flip"] else "BEFORE_NOT_SOURCE"})
            self.put(name, {"parent_id": parent["parent_id"], "prediction": prediction,
                           "parent_encoding": left, "pool_sha256": pool["pool_sha256"], "pairs": distances})
            completed.append(parent["parent_id"])
            atomic_json(self.root / "progress.json", {"stage": split, "shard": shard,
                        "completed_parent_units": len(completed), "updated_at": utc_now(), "job_id": os.environ["SLURM_JOB_ID"]})
        return self.put(f"{split}/shard-{shard}.json", {"status": "MATRIX_SHARD_COMPLETE", "parent_ids": completed,
                        "shard": shard, "shards": shards, "candidate_ids": [c["candidate_id"] for c in prototypes]})

    def matrix(self, split: str):
        import numpy as np
        parents = load_parent_rows(self.spec, split, self.root)
        pool = self.get("pool_freeze.json")
        cids = pool["candidate_ids"] if split == "calibration" else self.get("selection_freeze.json")["selected_candidate_ids"]
        values, statuses, masks = [], [], []
        for parent in parents:
            record = self.get(f"{split}/parents/{digest(parent['parent_id'])[:20]}.json")
            if record["parent_id"] != parent["parent_id"] or [r["candidate_id"] for r in record["pairs"]] != cids:
                raise ValueError("Matrix parent/candidate coverage/order changed")
            values.append([math.inf if r["distance"] is None else r["distance"] for r in record["pairs"]])
            statuses.append([r["pair_status"] for r in record["pairs"]])
            masks.append(record["prediction"]["predicted_label"] == 1)
        return np.asarray(values, dtype=float).reshape(len(parents), len(cids)), np.asarray(statuses).reshape(len(parents), len(cids)), [r["parent_id"] for r in parents], cids, masks

    def stage_select(self) -> dict:
        from src.baselines.cm_crem_selection import select_calibration
        values, statuses, pids, cids, masks = self.matrix("calibration")
        result = select_calibration(values, parent_ids=pids, candidate_ids=cids, source_mask=masks,
                 pair_status=statuses, theta=self.spec["resolved_evaluation"]["theta"], cap=self.spec["resolved_evaluation"]["cap"],
                 contract_sha256=self.sha, frozen_pool_sha256=self.get("pool_freeze.json")["pool_sha256"])
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        return self.put("selection_freeze.json", payload)

    def stage_audit(self) -> dict:
        from src.baselines.cm_crem_selection import evaluate_frozen_test, SelectionFreeze
        from src.baselines.cm_crem_audit import audit_bace_run, independent_spotcheck
        self.require_pilot()
        SelectionFreeze.from_dict(self.get("selection_freeze.json"))
        values, statuses, pids, cids, masks = self.matrix("test")
        freeze = self.get("selection_freeze.json")
        result = evaluate_frozen_test(freeze, values, parent_ids=pids, candidate_ids=cids,
                                     source_mask=masks, pair_status=statuses, contract_sha256=self.sha)
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self.put("test_evaluation.json", payload)
        if (self.root / 'audit/provenance_review.json').exists():
            # Immutable completed provenance contains its original auditor PID;
            # reuse it, do not overwrite it with a newly timestamped re-audit.
            # independent_spotcheck validates digest and bound inputs below.
            provenance = self.get('audit/provenance_review.json')
        else:
            provenance = audit_bace_run(self.spec, self.root)
            self.put("audit/provenance_review.json", provenance)
        spotcheck = independent_spotcheck(self.spec, self.root, provenance)
        if spotcheck.get("status") != "CM_CREM_INDEPENDENT_SPOTCHECK_PASS" or spotcheck.get("independent") is not True:
            raise ValueError("Independent real-science verification has not passed")
        self.put("audit/independent_spotcheck.json", spotcheck)
        return self.put("audit/final_audit.json", {"status": "BACE_CM_CREM_FINAL_AUDIT_PASS", "scientific_scope": METHOD,
                        "scientific_pass_claimed": True,
                        "test_base_count": len(pids), "candidate_count": len(cids),
                        "test_result_sha256": digest(payload), "selection_freeze_sha256": digest(freeze),
                        "weights_trained": False, "temperature_refitted": False, "main_matrix_written": False,
                        "independent_spotcheck": {"path": "audit/independent_spotcheck.json",
                            "sha256": file_sha(self.root / "audit/independent_spotcheck.json")},
                        "portable_acceptance_state": "PENDING_SCOPED_PACKAGE_TRANSFER_VERIFICATION"})

    def stage_export(self) -> dict:
        from src.baselines.cm_crem_export import export_results
        audit = self.get("audit/final_audit.json")
        if audit.get("status") != "BACE_CM_CREM_FINAL_AUDIT_PASS":
            raise ValueError("Final scientific audit has not passed")
        result = export_results(self.get("test_evaluation.json"), self.root, dataset="bace", oracle="gine")
        return {"status": "EXPORTED", "results": result}

    def stage_package(self) -> dict:
        from src.baselines.cm_crem_release import package_run
        return package_run(self.root)

    def status(self) -> dict:
        present = {}
        for name in ("pilot/oracle.json", "attribution.json", "pilot/pool.json", "pool_freeze.json", "pool_encodings.json", "selection_freeze.json", "audit/final_audit.json"):
            path = self.root / name
            if path.exists():
                r = self.get(name)
                present[name] = {k: r[k] for k in ("status", "retained_count", "test_base_count", "job_id", "generation_complete", "complete_end_to_end_pilot") if k in r}
        return {"method": METHOD, "root": str(self.root), "science_hash": self.sha,
                "stages": present, "assets": self.stage_preflight(), "progress": read_json(self.root/"progress.json") if (self.root/"progress.json").exists() else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=["resolve", "preflight", "pilot-oracle", "pilot-closeout", "attribution", "generate", "filter", "encode", "calibrate", "select", "test", "audit", "export", "package", "status"])
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    if any(v != "inference.fallback_to_heuristic=false" for v in args.set):
        raise ValueError("No unreviewed scientific CLI overrides")
    if not 1 <= args.shards <= 2 or not 0 <= args.shard < args.shards:
        raise ValueError("CM permits at most two stable shards")
    if args.action == "resolve":
        if args.bindings is None:
            parser.error("resolve requires --bindings")
        checked_root(args.run_root, HPC_SCOPE)
        spec = resolve_spec(args.spec, args.bindings, args.run_root/"spec.json")
        print(json.dumps({"status": "RESOLVED_NOT_SCIENCE", "science_hash": spec["science_hash"]}))
        return 0
    experiment = Experiment(args.spec, args.run_root)
    if args.action == "pilot-oracle":
        result = experiment.stage_attribution(pilot_only=True)
    elif args.action == "attribution":
        result = experiment.stage_attribution()
    elif args.action == "generate":
        result = experiment.stage_generate(shard=args.shard, shards=args.shards, pilot_only=args.pilot_only)
    elif args.action == "filter":
        result = experiment.stage_filter(pilot_only=args.pilot_only)
    elif args.action in {"calibrate", "test"}:
        result = experiment.stage_matrix("calibration" if args.action == "calibrate" else "test", args.shard, args.shards)
    elif args.action == "status":
        result = experiment.status()
    else:
        result = getattr(experiment, "stage_" + args.action.replace("-", "_"))()
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "parents", "encoded_graphs", "nonself_pairs", "candidates", "funnels", "train_predictions", "strata"}}, allow_nan=False))
    return 0
