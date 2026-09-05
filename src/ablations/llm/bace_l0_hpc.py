"""Small, immutable BRICS inputs and a real CPU-only L0 successor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from src.ablations.llm.bace_native_runtime import verified_file
from src.ablations.llm.bace_readiness import generation_calls
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.portable_inputs import PortableInputs, SCHEMA
from src.ablations.llm.runtime_evidence import load_bace_reference_v2
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file


def build_inputs(*, task_spec: str | Path, task_spec_sha256: str,
                 gnn_input_bundle: str | Path, output_root: str | Path) -> dict[str, Any]:
    from src.ablations.gnn.cpu_training import load_bundle
    from src.ablations.llm.bace_common_downstream import load_attempts

    spec_identity = {"path": str(Path(task_spec).resolve(strict=True)), "sha256": task_spec_sha256}
    spec_path = verified_file(spec_identity)
    spec = json.loads(spec_path.read_text())
    if (spec.get("variant") != "BRICS_FIXED" or spec.get("parent_count") != 386
            or spec.get("attempts_per_parent") != 8
            or spec.get("task_spec_sha256") != canonical_json_sha256({
                k: v for k, v in spec.items() if k != "task_spec_sha256"})):
        raise ValueError("L0_REQUIRES_SEALED_386_BY_8_BRICS_TASK")
    reference = load_bace_reference_v2(spec["reference_contract"]["path"], spec["reference_contract"]["sha256"])
    if generation_calls(reference.payload) != spec["calls"]:
        raise ValueError("L0_PROPOSAL_CONTRACT_MISMATCH")
    bundle, _ = load_bundle(gnn_input_bundle)
    brics_root = verified_file(spec["adopted_brics"]["brics_proposal_manifest.json"]).parent
    attempts, evidence = load_attempts(spec, brics_root, reference.file_sha256)
    if len(attempts) != 3088:
        raise ValueError("L0_COMPLETE_ATTEMPT_LEDGER_REQUIRED")
    identities: list[tuple[str, Mapping[str, Any]]] = [
        ("reference", spec_identity), ("reference", spec["reference_contract"]),
        ("reference", reference.payload["candidate_generation"]["parent_manifest"]),
        ("reference", {"path": reference.payload["frozen_downstream"]["dataset_split_paths"]["train"],
                       "sha256": reference.payload["frozen_downstream"]["dataset_split_hashes"]["train"]}),
    ]
    identities.extend(("brics", identity) for identity in spec["adopted_brics"].values())
    proposal = json.loads(verified_file(spec["adopted_brics"]["brics_proposal_manifest.json"]).read_text())
    for key in ("attempt_records", "candidate_pool", "vocabulary_manifest", "shortfall_receipt"):
        identities.append(("brics", proposal[key]))
    # Do not traverse the reference's model/PPO identities: L0 needs no LLM
    # weights. The original vocabulary is copied only when explicitly bound.
    vocabulary = json.loads(verified_file(spec["adopted_brics"]["brics_vocab_manifest.json"]).read_text())
    if vocabulary.get("vocabulary_size") != 472:
        raise ValueError("L0_MUST_REUSE_THE_EXISTING_472_FRAGMENT_VOCABULARY")
    vocabulary_identity = vocabulary["vocabulary_file"]
    if Path(str(vocabulary_identity["path"])).name != "brics_vocab.jsonl":
        raise ValueError("L0_UNEXPECTED_VOCABULARY_FILE")
    identities.append(("brics", vocabulary_identity))
    sources = {}
    total = 0
    for group, identity in identities:
        path = verified_file(identity)
        if path.suffix not in {".json", ".jsonl", ".csv"}:
            raise ValueError("L0_INPUTS_MUST_BE_SMALL_DECLARED_JSON_CSV_ONLY")
        if str(path) in sources:
            if sources[str(path)]["sha256"] != identity["sha256"]:
                raise ValueError("L0_DUPLICATE_SOURCE_DIFFERENT_HASH")
            continue
        total += path.stat().st_size
        sources[str(path)] = {"relative": f"{group}/{len(sources):03d}-{path.name}",
                              "sha256": identity["sha256"], "size": path.stat().st_size}
    if total > 128 * 1024 * 1024:
        raise ValueError("L0_DECLARED_INPUT_BUDGET_EXCEEDED_128_MIB")
    output = Path(output_root).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("L0_PORTABLE_OUTPUT_MUST_BE_FRESH_PHYSICAL_ROOT")
    for protected in (bundle, brics_root, Path(reference.path).parent, spec_path.parent):
        if output == protected or protected in output.parents:
            raise ValueError("L0_PORTABLE_OUTPUT_MUST_NOT_WRITE_INTO_IMMUTABLE_INPUT_ROOT")
    output.mkdir(parents=True)
    for source, entry in sources.items():
        destination = output / entry["relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open("rb") as incoming, destination.open("xb") as out:
            shutil.copyfileobj(incoming, out)
            out.flush()
            os.fsync(out.fileno())
        if sha256_file(destination) != entry["sha256"]:
            raise ValueError("L0_COPY_HASH_MISMATCH")
    manifest = {"schema_version": SCHEMA, "variant": "BRICS_FIXED", "source_files": sources,
                "task_spec": spec_identity, "reference_contract": spec["reference_contract"],
                "gnn_input_bundle_manifest_sha256": sha256_file(bundle / "bundle_manifest.json"),
                "brics_root_relative": "brics", "parent_count": 386, "proposal_attempts": 3088,
                "vocabulary_size": 472,
                "proposal_evidence": evidence, "input_bytes": total,
                "model_weights_copied": False, "original_manifests_modified": False,
                "main_matrix_write": False, "science_started": False}
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(output / "portable_manifest.json", manifest)
    portable = PortableInputs(output)
    for source, entry in sources.items():
        portable.resolve({"path": source, "sha256": entry["sha256"]})
    # Exercise the same relocated provenance chain the HPC worker will use.
    load_attempts(spec, output / "brics", reference.file_sha256, file_resolver=portable.resolve)
    return manifest


def run_l0(*, portable_input_bundle: str | Path, gnn_input_bundle: str | Path,
           corrected_gnn_archive: str | Path, corrected_gnn_sha256: str,
           registry_root: str | Path, output_root: str | Path, resume: bool = False,
           cpu_threads: int = 2, batch_size: int = 64) -> dict[str, Any]:
    from src.ablations.llm.bace_common_downstream import run_downstream

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise ValueError("L0_HPC_CPU_REQUIRES_NO_VISIBLE_GPU")
    portable = PortableInputs(portable_input_bundle)
    spec = json.loads(portable.task_spec_path().read_text())
    if spec["variant"] != "BRICS_FIXED":
        raise ValueError("L0_CPU_RUNNER_REJECTS_LLM_MODEL_VARIANTS")
    return run_downstream(task_spec=portable.task_spec_path(),
        candidate_root=portable.root / portable.manifest["brics_root_relative"],
        gnn_input_bundle=gnn_input_bundle, gnn_verified_archive=corrected_gnn_archive,
        gnn_verified_sha256=corrected_gnn_sha256, registry_root=registry_root,
        output_root=output_root, resume=resume, device="cpu", cpu_threads=cpu_threads,
        batch_size=batch_size, portable_input_bundle=portable.root)
