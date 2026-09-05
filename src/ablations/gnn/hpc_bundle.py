"""Small, immutable BACE proposal-fixed input transport; never writes a matrix."""
from __future__ import annotations

import gzip
import csv
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid

from src.ablations.contracts import canonical_json_sha256, sha256_file

BACKBONES = ("gine", "gin", "gcn", "gatv2", "gatedgcn_plus")


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(uuid.uuid4()) + ".tmp")
    with tmp.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_file(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular source file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"source changed while hashing: {path}")
    if expected and digest != expected:
        raise ValueError(f"source SHA mismatch: {path}")
    return {"sha256": digest, "size": after.st_size}


def build_bundle(*, reference_path, matrix_path, merged_pool_root, molclr_source,
                 project_root, output_root, execution_commit):
    from src.ablations.launch_gate import validate_matrix_authority_pointer

    ref = read_json(reference_path)
    expected = ref.pop("reference_contract_sha256")
    if canonical_json_sha256(ref) != expected or ref["status"] != "PASS":
        raise ValueError("invalid BACE reference")
    authority = validate_matrix_authority_pointer(read_json(matrix_path))
    if authority["complete_cells"] < 12 or authority["cell_roots"]["BACE/Ours"] != ref["main_final_root"]:
        raise ValueError("BACE reference is not the current published cell")
    project_root, output_root = Path(project_root), Path(output_root)
    if output_root.exists():
        raise ValueError("bundle requires a fresh output root")
    sources = {}
    split_row_counts = {}

    def add(relative, source, digest=None):
        identity = checked_file(source, digest)
        sources[relative] = {"source": str(source), **identity}

    for split, path in ref["dataset_split_paths"].items():
        add(f"data/{split}.csv", path, ref["dataset_split_hashes"][split])
        with Path(path).open() as stream:
            split_row_counts[split] = sum(1 for _ in csv.DictReader(stream))
    gine = Path(ref["gine_checkpoint_root"])
    for line in (gine / "sha256sums.txt").read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or Path(name).name != name:
            raise ValueError("malformed frozen GINE SHA inventory")
        checked_file(gine / name, digest)
    for name in ("model.pt", "config.yaml", "feature_schema.json", "split_manifest.json",
                 "temperature_scaling.json", "model_card.json", "label_map.json", "environment.json",
                 "sha256sums.txt", "training_metrics.json", "validation_predictions.csv", "git_state.json",
                 "test_evaluation_status.json", "b4_calibration.json"):
        digest = {"model.pt": ref["gine_checkpoint_sha"], "feature_schema.json": ref["feature_schema_sha"],
                  "temperature_scaling.json": ref["temperature_sha"]}.get(name)
        add(f"reference/gine/{name}", gine / name, digest)
    merge = read_json(Path(merged_pool_root) / "merge_manifest.json")
    frozen_pool = ref["proposal_contract"]["candidate_merge_dedup_policy"]
    for field, value in {"status": "PASS", "train_only": True, "test_loaded": False,
                         "candidate_universe_hash": frozen_pool["candidate_universe_sha"],
                         "candidate_pool_hash": frozen_pool["pool_sha"]}.items():
        if merge.get(field) != value:
            raise ValueError(f"merged candidate contract mismatch: {field}")
    for name, digest in (("candidate_pool.jsonl", frozen_pool["pool_sha"]),
                         ("candidate_universe.jsonl", frozen_pool["candidate_universe_sha"]),
                         ("merge_manifest.json", None)):
        add(f"candidates/{name}", Path(merged_pool_root) / name, digest)
    cohort = ref["proposal_contract"]["proposal_parent_cohort"]
    add("candidates/train_parent_ids.frozen.json", cohort["path"], cohort["sha256"])
    for key in ("selector_manifest", "thresholds", "variant_configs", "verified_matrix_manifest"):
        item = ref["selector_contract"][key]
        add(f"reference/selector/{Path(item['path']).name}", item["path"], item["sha256"])
    final = Path(ref["main_final_root"])
    add("reference/main/final_artifact_audit.json", final / "final_artifact_audit.json", ref["main_final_audit_sha"])
    add("reference/main/evaluation_manifest.json", final / "evaluation_manifest.json", ref["evaluation_config_sha"])
    add("reference/main/run_manifest.json", final / "run_manifest.json")
    if read_json(final / "run_manifest.json").get("test_used_for_selection") is not False:
        raise ValueError("main used test for selection")
    add("reference/molclr/model.pth", ref["molclr_root"], ref["molclr_sha"])
    molclr_source = Path(molclr_source)
    # Only the exact loader's model modules/configs; no weights/caches/history tree.
    for path in sorted((molclr_source / "models").glob("*.py")):
        add(f"reference/molclr/source/models/{path.name}", path)
    for name in ("config.yaml", "config_finetune.yaml", "config_pretrain.yaml"):
        if (molclr_source / name).is_file():
            add(f"reference/molclr/source/{name}", molclr_source / name)
    if not any("/source/models/" in name for name in sources):
        raise ValueError("missing exact MolCLR model source")
    for backbone in BACKBONES:
        add(f"configs/{backbone}.yaml", project_root / f"configs/gnn/{backbone}.yaml")
    for name in ("gatedgcn_plus_source_v1.yaml", "bace_gatedgcn_plus_parameter_match_v1.json", "bace_gine_reference_parameter_receipt_v1.json"):
        add(f"configs/{name}", project_root / "configs/ablations/gnn" / name)

    output_root.mkdir(parents=True)
    payload_root = output_root / "payload"
    payload_root.mkdir()
    for relative, identity in sources.items():
        destination = payload_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(identity["source"], destination)
        checked_file(destination, identity["sha256"])
        checked_file(identity["source"], identity["sha256"])
    contract = {"schema_version": "bace_ours_gnn_proposal_fixed_v1", "status": "PASS",
                "dataset": "bace", "method": "ours", "mode": "proposal_fixed", "source_label": ref["source_label"],
                "source_reference_sha256": checked_file(reference_path)["sha256"],
                "source_main_final_root": ref["main_final_root"], "main_cell_audit_sha256": ref["main_final_audit_sha"],
                "execution_commit": execution_commit, "candidate_pool_sha256": frozen_pool["pool_sha"],
                "candidate_universe_sha256": frozen_pool["candidate_universe_sha"],
                "candidate_universe_count": frozen_pool["candidate_universe_count"],
                "proposal_parent_count": ref["proposal_contract"]["proposal_parent_count"],
                "proposal_attempts_per_parent": ref["proposal_contract"]["candidate_attempts_per_parent"],
                "dataset_split_hashes": ref["dataset_split_hashes"], "gine_checkpoint_sha256": ref["gine_checkpoint_sha"],
                "temperature_sha256": ref["temperature_sha"], "feature_schema_sha256": ref["feature_schema_sha"],
                "molclr_sha256": ref["molclr_sha"], "wnode_config": ref["wnode_config"],
                "wnode_config_sha256": ref["wnode_config_sha"], "selector_config_sha256": ref["selector_config_sha"],
                "evaluation_config_sha256": ref["evaluation_config_sha"], "K": 20, "Table2_K": 10,
                "temperature_fit_split": "validation", "selector_fit_split": "calibration",
                "test_used_for_selection": False, "candidate_generation_allowed": False,
                "main_matrix_write_allowed": False, "main_matrix_observed_cells": authority["complete_cells"]}
    contract["contract_sha256"] = canonical_json_sha256(contract)
    atomic_json(payload_root / "reference_contract.json", contract)
    files = {name: {key: item[key] for key in ("sha256", "size")} for name, item in sources.items()}
    files["reference_contract.json"] = checked_file(payload_root / "reference_contract.json")
    manifest = {"schema_version": "bace_gnn_cpu_bundle_v1", "dataset": "bace", "seed": 7, "num_classes": 2,
                "execution_commit": execution_commit, "files": files,
                "splits": {s: f"data/{s}.csv" for s in ref["dataset_split_paths"]},
                "split_row_counts": split_row_counts,
                "feature_schema_path": "reference/gine/feature_schema.json", "training_config_path": "reference/gine/config.yaml",
                "backbone_configs": {b: f"configs/{b}.yaml" for b in BACKBONES},
                "gine_reference_root": "reference/gine", "reference_contract_path": "reference_contract.json",
                "candidate_pool_path": "candidates/candidate_pool.jsonl", "candidate_universe_path": "candidates/candidate_universe.jsonl",
                "candidate_manifest_path": "candidates/merge_manifest.json", "molclr_checkpoint": "reference/molclr/model.pth",
                "molclr_source": "reference/molclr/source", "selector_root": "reference/selector",
                "molclr_checkpoint_path": "reference/molclr/model.pth", "molclr_source_root": "reference/molclr/source",
                "frozen_selection_manifest_path": "reference/selector/frozen_selection_manifest.json",
                "selector_variant_configs_path": "reference/selector/variant_configs.json",
                "thresholds_path": "reference/selector/thresholds.json", "wnode_config": ref["wnode_config"],
                "molclr_encoder_type": "gin",
                "molclr_encoder_type_source": "src/eval/bace_frozen_gnn_verification.py: MolCLRNodeWassersteinConfig default gin",
                "main_matrix_write_allowed": False, "no_chemllm_weights": True}
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(payload_root / "bundle_manifest.json", manifest)
    with (output_root / "bace_gnn_hpc_allowlist.tsv").open("x") as stream:
        for name, item in sorted(sources.items()):
            stream.write(f"{name}\t{item['sha256']}\t{item['size']}\t{item['source']}\n")
    archive = output_root / "bace_gnn_hpc_input.tar.gz"
    with archive.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for path in sorted(payload_root.rglob("*")):
                if path.is_file():
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(payload_root).as_posix())
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
    receipt = {"state": "PASS", "archive": str(archive), **checked_file(archive),
               "payload_bytes": sum(item["size"] for item in files.values()), "manifest_sha256": manifest["manifest_sha256"],
               "reference_contract_sha256": contract["contract_sha256"], "main_matrix_write_allowed": False}
    atomic_json(output_root / "bundle_receipt.json", receipt)
    return receipt


def verify_bundle(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "bundle_manifest.json")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_json_sha256(body) != manifest.get("manifest_sha256"):
        raise ValueError("bundle manifest self hash mismatch")
    if manifest.get("schema_version") != "bace_gnn_cpu_bundle_v1" or manifest.get("main_matrix_write_allowed") is not False:
        raise ValueError("wrong bundle contract")
    expected = set(manifest["files"]) | {"bundle_manifest.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError("bundle inventory mismatch")
    for name, identity in manifest["files"].items():
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or (root / name).resolve().is_relative_to(root) is False:
            raise ValueError("bundle path escape")
        observed = checked_file(root / name, identity["sha256"])
        if observed["size"] != identity["size"]:
            raise ValueError("bundle size mismatch")
    return manifest


def unpack_bundle(archive, expected_sha, output_root):
    checked_file(archive, expected_sha)
    output_root = Path(output_root)
    if output_root.exists():
        verify_bundle(output_root)
        raise ValueError("destination already exists; adopt with verify, do not overwrite")
    staging = output_root.with_name(output_root.name + ".partial-" + str(uuid.uuid4()))
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r|gz") as tar:
        seen = set()
        total = 0
        for member in tar:
            pure = PurePosixPath(member.name)
            total += member.size
            if not member.isfile() or pure.is_absolute() or ".." in pure.parts or member.name in seen or total > 2 * 1024**3:
                raise ValueError("unsafe or oversized bundle member")
            seen.add(member.name)
            target = staging / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                shutil.copyfileobj(tar.extractfile(member), stream)
    manifest = verify_bundle(staging)
    os.rename(staging, output_root)
    return {"state": "PASS", "root": str(output_root), "manifest_sha256": manifest["manifest_sha256"], "archive_sha256": expected_sha}
