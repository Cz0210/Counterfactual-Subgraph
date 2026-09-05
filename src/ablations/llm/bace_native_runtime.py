"""BACE-only native ChemLLM loading and four-sequence RNG-bound generation.

No training lives here.  The 7B PPO treatment loads the already-trained adapter;
all treatments use the same train parents, four shards and two decoding regimes.
Each native model's audited build_inputs is used instead of guessing a chat
template.  Parent-call checkpoints retain exact RNG state, including CUDA.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.isolated_chemllm_load import (
    audit_remote_code, disable_tokenizer_exports, validate_isolated_load_receipt,
)
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, sha256_file


VARIANTS = ("BRICS_FIXED", "CHEMLLM_7B_OFF_THE_SHELF",
            "CHEMLLM_7B_PPO_LORA_MAIN", "CHEMLLM_2B_OFF_THE_SHELF")
SOURCE_PINS = {
    "7b": {"revision": "b8b2ea19e48f53d190fe8dced94572717f8e89a2",
           "repository_id": "AI4Chem/ChemLLM-7B-Chat",
           "sources": {"configuration_internlm.py": "f5ad3ba053c540282ba974d9ff27df8334a0f36cc64453fb0535a4ce1c73a894",
               "modeling_internlm2.py": "49c880ca41d18d997548a145db9d437e3143ea2252b18e5741d26c3e9b66a0f5",
               "tokenization_internlm.py": "880e2cebff1d30db2acb485b8fc00299fda7a5efb2c4d8400bd9adf60d1158e0"}},
    "2b": {"revision": "215c0dbc89417a06bbc3bae43a3ad61e58f0a56e",
           "repository_id": "AI4Chem/CHEMLLM-2b-1_5",
           "sources": {"configuration_internlm2.py": "6a62401c95726d56e499142afbe40a383231c00ed96b07ad32cc9da9d8c876ff",
               "modeling_internlm2.py": "be2a7ea99ed402eb767d636624b2bd997b78344f850a06400c2d4dea8e7e78ca",
               "tokenization_internlm2.py": "444d4c2b0da158e61c34b3c727943f0ad454770c74b307f4d881f03603335eef"}},
}


def verified_file(identity: Mapping[str, Any]) -> Path:
    path = Path(str(identity["path"]))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Expected existing physical absolute file")
    if sha256_file(path) != identity["sha256"]:
        raise ValueError("Frozen runtime file SHA differs")
    return path


def audit_native_source(root: str | Path, size: str) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    pin = SOURCE_PINS[size]
    actual = {path.name: sha256_file(path) for path in root.glob("*.py")}
    if actual != pin["sources"]:
        raise ValueError("Native model source differs from its reviewed exact revision")
    return audit_remote_code(SimpleNamespace(root=str(root), config_path=str(root / "config.json"),
        repository_id=pin["repository_id"], revision=pin["revision"]))


def render_native_inputs(model: Any, tokenizer: Any, prompt: str, size: str) -> Mapping[str, Any]:
    native = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    if size == "2b":
        return native.build_inputs(tokenizer, prompt, history=[], meta_instruction="")
    if size == "7b":
        return native.build_inputs(tokenizer, prompt, history=[])
    raise ValueError("Only the two pinned ChemLLM scales are supported")


def verified_2b_runtime_proof(model_spec: Mapping[str, Any], proof: Mapping[str, Any] | None) -> dict[str, Any]:
    if not proof:
        raise ValueError("L3 requires actual isolated CPU weight/forward/generation PASS")
    receipt = validate_isolated_load_receipt(verified_file(proof), require_weights=True)
    tiny = receipt.get("tiny_forward") or {}
    if (tiny.get("status") != "PASS" or tiny.get("tiny_generation_only") is not True
            or tiny.get("tiny_generation_max_new_tokens") != 4
            or not 0 < tiny.get("tiny_generation_token_count", 0) <= 4
            or receipt.get("code_inventory_sha256") != model_spec["remote_code_audit"]["code_inventory_sha256"]):
        raise ValueError("L3 isolated native tiny generation/source proof differs")
    return receipt


class BACEHFNativeRuntime:
    def __init__(self, spec: Mapping[str, Any]):
        import torch
        from src.eval.full_candidate_pool import _build_base_model, _build_lora_model, _build_tokenizer
        from src.ablations.llm.parameter_count import count_actual_loaded_parameters
        model_spec = spec["model"]
        self.size = model_spec["size"]
        if self.size == "2b":
            verified_2b_runtime_proof(model_spec, spec.get("isolated_cpu_load_receipt"))
        if torch.cuda.device_count() != 1:
            raise ValueError("Formal generation requires exactly one already-leased visible GPU")
        root = Path(model_spec["root"])
        audit_native_source(root, self.size)
        for name, digest in model_spec["files"].items():
            if Path(name).name != name:
                raise ValueError("Model inventory path must be a direct child")
            verified_file({"path": str(root / name), "sha256": digest})
        self.tokenizer = _build_tokenizer(base_model_path=root, trust_remote_code=True, local_files_only=True)
        disable_tokenizer_exports(self.tokenizer)
        adapter = spec.get("ppo_adapter")
        if spec["variant"] == "CHEMLLM_7B_PPO_LORA_MAIN":
            if not adapter:
                raise ValueError("PPO row requires the exact existing 300-update adapter")
            for name, identity in adapter["files"].items():
                if verified_file(identity).parent != Path(adapter["root"]):
                    raise ValueError("PPO adapter file escaped pinned root")
            self.model = _build_lora_model(base_model_path=root, adapter_path=Path(adapter["root"]),
                trust_remote_code=True, local_files_only=True)
        else:
            if adapter is not None:
                raise ValueError("Off-the-shelf model must load no project adapter")
            self.model = _build_base_model(base_model_path=root, trust_remote_code=True, local_files_only=True)
        self.model.eval()
        self.parameter_report = count_actual_loaded_parameters(self.model).to_dict()
        self.torch = torch

    def generate_call(self, call: Mapping[str, Any]) -> list[dict[str, Any]]:
        from src.eval.full_candidate_pool import (FullPoolGenerationConfig, build_generation_kwargs,
                                                  generate_ids_with_sanitized_kwargs)
        from src.models.llm_generator import clean_generated_smiles
        encoded = render_native_inputs(self.model, self.tokenizer, call["prompt"], self.size)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        config = FullPoolGenerationConfig(num_return_sequences=4, max_new_tokens=96,
            generation_temperature=call["temperature"], generation_top_p=0.9,
            generation_do_sample=True, batch_size=1)
        kwargs = build_generation_kwargs(encoded=encoded, tokenizer=self.tokenizer, config=config)
        if self.size == "2b":
            kwargs["eos_token_id"] = [self.tokenizer.eos_token_id,
                                      self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        outputs = generate_ids_with_sanitized_kwargs(self.model, kwargs, torch_module=self.torch)
        raw = self.tokenizer.batch_decode(outputs[:, encoded["input_ids"].shape[1]:].detach().cpu().tolist(),
                                          skip_special_tokens=True)
        if len(raw) != 4:
            raise ValueError("Native generation must return four sequences in one call")
        stop = "<|im_end|>" if self.size == "2b" else "<eoa>"
        return [{"candidate_index": index, "raw_text": text.split(stop)[0],
                 "fragment_smiles": clean_generated_smiles(text.split(stop)[0])}
                for index, text in enumerate(raw)]


def _rng(torch: Any) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value: Mapping[str, Any], torch: Any) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA topology changed across generation resume")
        torch.cuda.set_rng_state_all(value["cuda"])


def validate_calls(calls: Sequence[Mapping[str, Any]]) -> None:
    parents, identities = {}, set()
    visited_groups, previous_group = set(), None
    for call in calls:
        group = (call["shard_id"], call["regime"])
        if group != previous_group:
            if group in visited_groups:
                raise ValueError("Shard/regime calls must be contiguous; never reset seed per parent")
            visited_groups.add(group)
            previous_group = group
        key = (call["parent_id"], call["regime"])
        if key in identities:
            raise ValueError("Duplicate parent/regime generation call")
        identities.add(key)
        if call["regime"] not in {"base", "high_temperature"}:
            raise ValueError("Unknown BACE regime")
        expected = (7, 0.3) if call["regime"] == "base" else (13, 0.7)
        if (call["seed"], call["temperature"]) != expected or call["shard_id"] not in range(4):
            raise ValueError("BACE decoding/partition contract changed")
        parents.setdefault(call["parent_id"], set()).add(call["regime"])
    if not parents or any(regimes != {"base", "high_temperature"} for regimes in parents.values()):
        raise ValueError("Each parent must retain both four-sequence regimes")


def _commit_checkpoint(output: Path, state: Mapping[str, Any], torch: Any) -> None:
    name = f"call-{state['next_call']:06d}.pt"
    fd, temporary = tempfile.mkstemp(prefix=".rng-", dir=output)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / name)
    finally:
        Path(temporary).unlink(missing_ok=True)
    atomic_json(output / "latest_checkpoint.json", {"schema_version": "bace_llm_call_checkpoint_v1",
        "next_call": state["next_call"], "checkpoint_file": name,
        "sha256": sha256_file(output / name), "spec_sha256": state["spec_sha256"]})


def run_generation(*, spec: Mapping[str, Any], output_root: str | Path, resume: bool = False,
                   runtime: Any = None, max_calls: int | None = None,
                   continue_guard: Any = None) -> dict[str, Any]:
    """Called only by the GNN-core/main-GPU authorized launcher, never a scheduler."""
    import torch
    if spec.get("variant") not in VARIANTS[1:] or spec.get("schema_version") != "bace_native_llm_task_v1":
        raise ValueError("Unsupported formal native LLM task")
    calls = list(spec["calls"])
    validate_calls(calls)
    digest = canonical_json_sha256(spec)
    output = Path(output_root)
    if output.is_symlink():
        raise ValueError("Generation output must be physical")
    output.mkdir(parents=True, exist_ok=resume)
    with (output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest_path = output / "latest_checkpoint.json"
        state = {"next_call": 0, "rows": [], "group": None, "rng": None}
        if resume:
            latest = json.loads(latest_path.read_text())
            checkpoint_name = latest["checkpoint_file"]
            if Path(checkpoint_name).name != checkpoint_name:
                raise ValueError("Checkpoint path escapes generation root")
            checkpoint = verified_file({"path": str((output / checkpoint_name).resolve()), "sha256": latest["sha256"]})
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["spec_sha256"] != digest or state["next_call"] != latest["next_call"]:
                raise ValueError("Generation resume contract/cursor changed")
            if len(state["rows"]) != state["next_call"] * 4:
                raise ValueError("Generation checkpoint omitted proposal attempts")
        runtime = runtime or BACEHFNativeRuntime(spec)
        if getattr(runtime, "parameter_report", None):
            atomic_json(output / "actual_parameter_count_report.json", runtime.parameter_report)
        if state["rng"] is not None:
            _restore_rng(state["rng"], torch)
        else:
            # Even a main-priority pause immediately after model load can resume
            # without inventing a fresh attempt or losing the initial RNG state.
            state.update(spec_sha256=digest, rng=_rng(torch))
            _commit_checkpoint(output, state, torch)
        stopped = False
        def stop_at_boundary(*_args: Any) -> None:
            nonlocal stopped
            stopped = True
        handlers = {sig: signal.signal(sig, stop_at_boundary) for sig in (signal.SIGTERM, signal.SIGUSR1)}
        started, completed_here = time.monotonic(), 0
        try:
            for index in range(state["next_call"], len(calls)):
                if continue_guard is not None and not continue_guard():
                    stopped = True
                    break
                call = calls[index]
                group = (call["shard_id"], call["regime"])
                if tuple(state["group"] or ()) != group:
                    # Once per original shard/regime, never once per parent.
                    from src.eval.full_candidate_pool import set_global_generation_seed
                    set_global_generation_seed(call["seed"])
                rows = runtime.generate_call(call)
                if len(rows) != 4:
                    raise ValueError("Runtime returned incomplete four-sequence call")
                records = [{**row, "parent_id": call["parent_id"], "parent_smiles": call["parent_smiles"],
                    "label": 1, "source_label": 1, "regime": call["regime"], "shard_id": call["shard_id"],
                    "attempt_index": (0 if call["regime"] == "base" else 4) + position,
                    "variant": spec["variant"], "train_only": True}
                    for position, row in enumerate(rows)]
                state = {"spec_sha256": digest, "next_call": index + 1,
                         "group": group, "rows": state["rows"] + records, "rng": _rng(torch)}
                _commit_checkpoint(output, state, torch)
                # Two committed boundaries are sufficient; raw science rows/RNG
                # are retained in both complete checkpoints, never lost to trimming.
                if index >= 1:
                    (output / f"call-{index - 1:06d}.pt").unlink(missing_ok=True)
                completed_here += 1
                atomic_json(output / "progress.json", {"next_call": index + 1, "total_calls": len(calls),
                    "rows": len(state["rows"]), "pid": os.getpid(), "spec_sha256": digest})
                if stopped or (max_calls is not None and completed_here >= max_calls):
                    break
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        complete = state["next_call"] == len(calls)
        if complete:
            atomic_jsonl(output / "candidate_pool.jsonl", state["rows"])
        receipt = {"status": "CANDIDATE_POOL_PASS" if complete else "PAUSED_AT_CALL_CHECKPOINT",
            "spec_sha256": digest, "next_call": state["next_call"], "total_calls": len(calls),
            "proposal_attempts": len(state["rows"]), "elapsed_this_invocation_seconds": time.monotonic() - started,
            "safe_pause_bound_seconds": None, "safe_pause_bound_measured": False,
            "checkpoint_resume_supported": True, "test_loaded": False, "calibration_loaded": False,
            "training_performed": False, "variant": spec["variant"],
            "candidate_pool_sha256": sha256_file(output / "candidate_pool.jsonl") if complete else None}
        atomic_json(output / "candidate_generation_receipt.json", receipt)
        return receipt
