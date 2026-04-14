"""Validate local dataset and ChemLLM assets without any network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.env import apply_dotlist_overrides, load_and_merge_config_files
from src.utils.paths import get_repo_root, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=["local", "hpc"],
        default="local",
        help="Which environment config overlay to use.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional config files to merge after base and environment configs.",
    )
    parser.add_argument("--dataset-path", help="Override data.raw_hiv_csv.")
    parser.add_argument("--model-path", help="Override model.model_path.")
    parser.add_argument("--tokenizer-path", help="Override model.tokenizer_path.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra dotted config overrides.",
    )
    return parser


def load_config(args: argparse.Namespace) -> dict[str, object]:
    repo_root = get_repo_root()
    config_files = [
        repo_root / "configs" / "base.yaml",
        repo_root / "configs" / f"{args.environment}.yaml",
    ]
    config_files.extend(Path(path).expanduser() for path in args.config)
    config = load_and_merge_config_files(config_files)

    overrides = list(args.set)
    if args.dataset_path:
        overrides.append(f"data.raw_hiv_csv={args.dataset_path}")
    if args.model_path:
        overrides.append(f"model.model_path={args.model_path}")
    if args.tokenizer_path:
        overrides.append(f"model.tokenizer_path={args.tokenizer_path}")
    if overrides:
        config = apply_dotlist_overrides(config, overrides)
    return config


def validate_assets(config: dict[str, object]) -> dict[str, object]:
    repo_root = get_repo_root()
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    runtime_cfg = dict(config.get("runtime", {}))

    dataset_path = resolve_path(
        data_cfg.get("raw_hiv_csv") or data_cfg.get("dataset_path"),
        base_dir=repo_root / "data",
    )
    model_path = resolve_path(
        model_cfg.get("model_path") or model_cfg.get("model_name_or_path"),
        base_dir=repo_root,
    )
    tokenizer_path = resolve_path(
        model_cfg.get("tokenizer_path")
        or model_cfg.get("model_path")
        or model_cfg.get("model_name_or_path"),
        base_dir=repo_root,
    )

    result: dict[str, object] = {
        "dataset_path": str(dataset_path) if dataset_path else None,
        "model_path": str(model_path) if model_path else None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "dataset_preview": [],
        "model_ready": False,
        "system_ready": False,
    }

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("pandas is required for scripts/test_assets.py.") from exc

    if dataset_path is None:
        raise ValueError("data.raw_hiv_csv or data.dataset_path is not configured.")
    dataframe = pd.read_csv(dataset_path)
    if "smiles" not in dataframe.columns:
        raise KeyError("Expected a 'smiles' column in the local AIDS/HIV CSV file.")

    preview = (
        dataframe["smiles"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    preview = [value for value in preview.tolist() if value][:3]
    if len(preview) < 3:
        raise ValueError("AIDS/HIV CSV did not contain at least 3 non-empty SMILES strings.")
    result["dataset_preview"] = preview

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("transformers is required for local ChemLLM asset validation.") from exc

    if model_path is None:
        raise ValueError("model.model_path or model.model_name_or_path is not configured.")
    if tokenizer_path is None:
        raise ValueError("model.tokenizer_path is not configured.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        use_fast=False,
    )
    config_obj = AutoConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    result["tokenizer_class"] = tokenizer.__class__.__name__
    result["model_config_class"] = config_obj.__class__.__name__
    result["local_files_only"] = bool(runtime_cfg.get("local_files_only", True))
    result["model_ready"] = True
    result["system_ready"] = True
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args)
        result = validate_assets(config)
    except Exception as exc:
        raise SystemExit(f"Asset validation failed: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print("System ready: local dataset and ChemLLM assets loaded successfully.")


if __name__ == "__main__":
    main()
