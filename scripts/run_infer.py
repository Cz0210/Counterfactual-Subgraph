"""Config-driven inference runtime entrypoint for local and HPC use."""

from __future__ import annotations

import argparse
import json

from _runtime_common import (
    add_common_arguments,
    prepare_runtime,
    print_manifest_if_requested,
    write_named_manifest,
)

from src.data.schemas import MoleculeRecord
from src.eval.inference import (
    run_chemllm_inference,
    run_minimal_inference,
    sample_inference_record_from_aids,
)
from src.models import ChemLLMGenerator
from src.utils.paths import resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--smiles", help="Parent molecule SMILES for single-example inference.")
    parser.add_argument("--label", type=int, help="Optional parent label.")
    parser.add_argument("--dataset-path", help="Override data.raw_hiv_csv.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Override inference.sample_seed when sampling from the local AIDS/HIV CSV file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override inference.max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override inference.temperature.",
    )
    parser.add_argument("--top-p", type=float, help="Override inference.top_p.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.run_name:
        args.set.append("run.name=infer")
    if args.smiles:
        args.set.append(f"inference.smiles={args.smiles}")
    if args.label is not None:
        args.set.append(f"inference.label={args.label}")
    if args.dataset_path:
        args.set.append(f"data.raw_hiv_csv={args.dataset_path}")
    if args.sample_seed is not None:
        args.set.append(f"inference.sample_seed={args.sample_seed}")
    if args.max_new_tokens is not None:
        args.set.append(f"inference.max_new_tokens={args.max_new_tokens}")
    if args.temperature is not None:
        args.set.append(f"inference.temperature={args.temperature}")
    if args.top_p is not None:
        args.set.append(f"inference.top_p={args.top_p}")

    prepared = prepare_runtime(args, stage_name="infer", stage_config_name="eval.yaml")

    inference_cfg = prepared.config.get("inference", {})
    data_cfg = prepared.config.get("data", {})
    model_cfg = prepared.config.get("model", {})
    runtime_cfg = prepared.config.get("runtime", {})
    smiles = inference_cfg.get("smiles")
    label = inference_cfg.get("label")
    sample_seed = inference_cfg.get("sample_seed")
    record: MoleculeRecord | None = None

    if smiles:
        record = MoleculeRecord(record_id="cli", smiles=str(smiles).strip(), label=int(label or 0))
    elif inference_cfg.get("sample_from_dataset", True):
        dataset_path = resolve_path(
            data_cfg.get("raw_hiv_csv") or data_cfg.get("dataset_path"),
            base_dir=prepared.runtime_paths.data_root,
        )
        if dataset_path is None:
            raise ValueError(
                "data.raw_hiv_csv or data.dataset_path is required for dataset-backed inference."
            )
        record = sample_inference_record_from_aids(
            str(dataset_path),
            seed=int(sample_seed) if isinstance(sample_seed, int) else None,
            smiles_column=str(data_cfg.get("smiles_column", "smiles")),
            label_column=str(data_cfg.get("label_column", "HIV_active")),
            activity_column=str(data_cfg.get("activity_column", "activity")),
        )
        smiles = record.smiles
        label = record.label

    plan = {
        "config_files": prepared.config_files,
        "smiles": smiles,
        "label": label,
        "runtime_environment": prepared.config.get("runtime", {}).get("environment"),
        "run_dir": str(prepared.runtime_paths.run_dir),
    }
    write_named_manifest(prepared, "infer_plan.json", plan)

    if record is None or not record.smiles:
        prepared.logger.warning("No parent SMILES was provided; printing resolved config only.")
        print_manifest_if_requested(prepared, enabled=True)
        return

    result: dict[str, object]
    model_error: str | None = None
    if inference_cfg.get("use_llm", True):
        try:
            model_path = prepared.runtime_paths.model_path or resolve_path(
                model_cfg.get("model_path") or model_cfg.get("model_name_or_path"),
                base_dir=prepared.runtime_paths.repo_root,
            )
            tokenizer_path = prepared.runtime_paths.tokenizer_path or resolve_path(
                model_cfg.get("tokenizer_path")
                or model_cfg.get("model_path")
                or model_cfg.get("model_name_or_path"),
                base_dir=prepared.runtime_paths.repo_root,
            )
            if model_path is None:
                raise ValueError(
                    "model.model_path or model.model_name_or_path is required for ChemLLM inference."
                )
            generator = ChemLLMGenerator(
                model_name_or_path=str(model_path),
                tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
                device=str(runtime_cfg.get("device", "auto")),
                local_files_only=bool(runtime_cfg.get("local_files_only", True)),
                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
                max_new_tokens=int(inference_cfg.get("max_new_tokens", 64)),
                top_p=float(inference_cfg.get("top_p", 1.0)),
            )
            result = run_chemllm_inference(
                record,
                generator=generator,
                max_new_tokens=int(inference_cfg.get("max_new_tokens", 64)),
                temperature=float(inference_cfg.get("temperature", 0.0)),
                top_p=float(inference_cfg.get("top_p", 1.0)),
            )
            result["generator"] = {
                "backend": "chemllm_local_huggingface",
                "device": generator.assets.device,
                "model_path": str(generator.assets.model_path),
                "tokenizer_path": str(generator.assets.tokenizer_path),
                "local_files_only": bool(runtime_cfg.get("local_files_only", True)),
            }
        except Exception as exc:
            model_error = str(exc)
            prepared.logger.warning("ChemLLM initialization or inference failed: %s", exc)
            if not inference_cfg.get("fallback_to_heuristic", True):
                raise
            result = run_minimal_inference(
                record.smiles,
                label=record.label,
                max_new_tokens=int(inference_cfg.get("max_new_tokens", 64)),
                temperature=float(inference_cfg.get("temperature", 0.0)),
                top_p=float(inference_cfg.get("top_p", 1.0)),
            )
            result["mode"] = "heuristic_fallback_inference"
            result["model_error"] = model_error
    else:
        result = run_minimal_inference(
            record.smiles,
            label=record.label,
            max_new_tokens=int(inference_cfg.get("max_new_tokens", 64)),
            temperature=float(inference_cfg.get("temperature", 0.0)),
            top_p=float(inference_cfg.get("top_p", 1.0)),
        )

    result["dataset_record"] = {
        "record_id": record.record_id,
        "smiles": record.smiles,
        "label": record.label,
    }
    result["run"] = {
        "name": prepared.context.run_name,
        "stage": prepared.context.stage,
        "run_dir": str(prepared.runtime_paths.run_dir),
    }
    if args.print_config:
        result["resolved_config"] = prepared.manifest

    write_named_manifest(prepared, "infer_result.json", result)
    prepared.logger.info(
        "Generated fragment candidate '%s' for parent '%s'",
        result["fragment_candidate"],
        result["parent_smiles"],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
