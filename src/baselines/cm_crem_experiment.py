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
        records = []
        for i, row in enumerate(selected):
            part = f"attribution_units/{digest(row['parent_id'])[:20]}.json"
            if (self.root / part).exists():
                record = self.get(part)
            else:
                record = self.put(part, oracle.attribute_train_parent(row))
            records.append(record)
            atomic_json(self.root / "progress.json", {"stage": output, "completed_parent_units": i+1,
                         "total_parent_units": len(selected), "updated_at": utc_now(), "job_id": os.environ.get("SLURM_JOB_ID")})
        result = {"status": "ATTRIBUTION_COMPLETE", "parents": selected, "records": records,
                  "train_predictions": predictions,
                  "train_proposal_count": len(rows), "train_predicted_source_count": len(eligible),
                  "strata": strata, "seconds": time.monotonic()-started, "test_loaded": False,
                  "job_id": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
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
        if db_receipt.get("url") != self.spec["upstream"]["database"]["url"] or db_receipt.get("status") != "VERIFIED_STATIC_COPY":
            raise ValueError("Official database content provenance missing")
        if file_sha(e["database_path"]) != db_receipt["uncompressed_sha256"]:
            raise ValueError("Generator database differs from its one-transfer content proof")
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
                result = generate_parent(attr["generation_request"], {"database_path": e["database_path"],
                         "upstream_root": e["upstream_root"], "science_hash": self.sha, "parent_wall_limit_seconds": 900}, log_path=log)
            if result["status"] not in TERMINALS:
                self.put(f"failures/generate-{index}-{os.environ['SLURM_JOB_ID']}.json", result)
                raise RuntimeError(f"Generation {attr['parent_id']}: {result['status']}: {result.get('error')}")
            records.append(self.put(path, result))
        name = f"{'pilot' if pilot_only else 'full'}/generation-shard-{shard}.json"
        return self.put(name, {"status": "GENERATION_SHARD_COMPLETE", "shard": shard, "shards": shards,
                             "parent_count": len(records), "parent_ids": [r["parent_id"] for r in records]})

    def stage_filter(self, *, pilot_only: bool = False) -> dict:
        require_compute_node()
        oracle = self.oracle()
        attrs = self.get("pilot/oracle.json" if pilot_only else "attribution.json")
        accepted, funnels = {}, []
        for parent in attrs["parents"]:
            generation = self.get(f"generation_units/{digest(parent['parent_id'])[:20]}.json")
            result = oracle.filter_generated(parent, generation)
            self.put(f"filter_units/{digest(parent['parent_id'])[:20]}.json", result)
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
        return self.put("pilot/pool.json" if pilot_only else "pool_freeze.json", payload)

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
        parents = oracle_result["parents"]
        if len(parents) != 32:
            raise ValueError("Complete pilot must contain exactly32 prespecified train parents")
        generated = [self.get(f"generation_units/{digest(p['parent_id'])[:20]}.json") for p in parents]
        if any(g["status"] not in TERMINALS for g in generated):
            raise ValueError("Unresolved generation error is not a complete pilot")
        oracle, distance = self.oracle(), self.distance()
        parent_encoded = distance.encode_rows(parents, featurizer=oracle.featurizer)
        prototype_encoded = distance.encode_rows(pool["candidates"], featurizer=oracle.featurizer)
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
        # Worst-case pool=2000, two sequential CPU lanes; never adjust science
        # parameters based on the pilot's coverage, validity or rank.
        full_generation = oracle_result["train_predicted_source_count"] * gen_p90 / 2
        full_pairs = 66 * 2000 + 141 * 20
        estimated = 2 * (full_generation + full_pairs * distance_p90 / 2 + oracle_result["seconds"]) / 3600 + 12
        from datetime import datetime, timezone
        elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(self.spec["campaign"]["start_time_utc"].replace("Z", "+00:00"))).total_seconds()/3600
        accepted = estimated + elapsed <= self.spec["campaign"]["planning_horizon_hours"]
        return self.put("pilot/final_receipt.json", {"status": "PILOT_ENGINEERING_PROTOCOL_ACCEPTED" if accepted else "BLOCKED_CAMPAIGN_COST",
                       "parent_count": 32, "real_nonself_wnode_pairs": len(oracle_result["nonself_pairs"]),
                       "generated_prototype_pairs": raw_pairs, "generation_seconds": generation_times,
                       "generation_p50_seconds": float(np.median(generation_times)), "generation_p90_seconds": gen_p90,
                       "wnode_p90_seconds": distance_p90, "full_campaign_eta_hours_conservative": estimated,
                       "elapsed_hours": elapsed, "complete_end_to_end_pilot": True,
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
        from src.baselines.cm_crem_selection import evaluate_frozen_test, select_calibration, SelectionFreeze
        self.require_pilot()
        cv, cs, cp, cc, cm = self.matrix("calibration")
        replay = select_calibration(cv, pair_status=cs, parent_ids=cp, candidate_ids=cc, source_mask=cm,
                 theta=self.spec["resolved_evaluation"]["theta"], cap=self.spec["resolved_evaluation"]["cap"],
                 contract_sha256=self.sha, frozen_pool_sha256=self.get("pool_freeze.json")["pool_sha256"])
        if replay.freeze_sha256 != SelectionFreeze.from_dict(self.get("selection_freeze.json")).freeze_sha256:
            raise ValueError("Independent calibration replay differs from pre-test freeze")
        values, statuses, pids, cids, masks = self.matrix("test")
        freeze = self.get("selection_freeze.json")
        result = evaluate_frozen_test(freeze, values, parent_ids=pids, candidate_ids=cids,
                                     source_mask=masks, pair_status=statuses, contract_sha256=self.sha)
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self.put("test_evaluation.json", payload)
        return self.put("audit/final_audit.json", {"status": "RECORD_RECONCILIATION_COMPLETE", "scientific_scope": METHOD,
                        "test_base_count": len(pids), "candidate_count": len(cids),
                        "test_result_sha256": digest(payload), "selection_freeze_sha256": digest(freeze),
                        "weights_trained": False, "temperature_refitted": False, "main_matrix_written": False,
                        "portable_acceptance_state": "PENDING_INDEPENDENT_PROVENANCE_REVIEW"})

    def stage_export(self) -> dict:
        from src.baselines.cm_crem_export import export_results
        self.get("audit/final_audit.json")
        result = export_results(self.get("test_evaluation.json"), self.root, dataset="bace", oracle="gine")
        return {"status": "EXPORTED", "results": result}

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
    parser.add_argument("--action", required=True, choices=["resolve", "preflight", "pilot-oracle", "pilot-closeout", "attribution", "generate", "filter", "encode", "calibrate", "select", "test", "audit", "export", "status"])
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
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "parents", "encoded_graphs", "nonself_pairs", "candidates", "funnels"}}, allow_nan=False))
    return 0
