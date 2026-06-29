#!/usr/bin/env python3
"""Precompute MolCLR embeddings for parent, residual, and GT full graphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.molclr_distance import precompute_molclr_embeddings_for_ccrcov  # noqa: E402


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        try:
            expanded = path.expanduser().resolve()
        except Exception:
            continue
        if expanded.exists():
            return expanded
    return None


def discover_molclr_root(value: str | None) -> str:
    if value:
        return str(Path(value).expanduser().resolve())
    env_value = os.environ.get("MOLCLR_ROOT")
    if env_value:
        return str(Path(env_value).expanduser().resolve())
    candidates = [
        REPO_ROOT / "baselines" / "molclr",
        REPO_ROOT / "baselines" / "MolCLR",
        REPO_ROOT / "external" / "MolCLR",
        REPO_ROOT.parent / "MolCLR",
        Path("/share/home/u20526/czx/MolCLR"),
    ]
    found = _first_existing(candidates)
    if found is not None:
        return str(found)
    raise SystemExit(
        "[ERROR] MolCLR root not found. Pass --molclr-root or set MOLCLR_ROOT=/path/to/MolCLR."
    )


def discover_molclr_checkpoint(value: str | None) -> str:
    if value:
        return str(Path(value).expanduser().resolve())
    for env_name in ("MOLCLR_CKPT", "MOLCLR_CHECKPOINT", "MOLCLR_PRETRAINED_GIN_CKPT"):
        env_value = os.environ.get(env_name)
        if env_value:
            return str(Path(env_value).expanduser().resolve())
    direct_candidates = [
        REPO_ROOT / "outputs" / "hpc" / "molclr" / "pretrained_gin.pth",
        REPO_ROOT / "outputs" / "hpc" / "pretrained_gin" / "model.pth",
        REPO_ROOT / "checkpoints" / "molclr_gin.pth",
        Path("/share/home/u20526/czx/MolCLR/ckpt/pretrained_gin.pth"),
    ]
    found = _first_existing(direct_candidates)
    if found is not None:
        return str(found)
    search_roots = [REPO_ROOT / "outputs" / "hpc", REPO_ROOT / "checkpoints", REPO_ROOT / "baselines"]
    suffixes = {".pt", ".pth", ".ckpt"}
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            name = candidate.name.lower()
            if candidate.is_file() and candidate.suffix.lower() in suffixes and (
                "molclr" in name or "pretrained" in name or "gin" in name
            ):
                return str(candidate.resolve())
    raise SystemExit(
        "[ERROR] MolCLR checkpoint not found. Pass --molclr-checkpoint or set MOLCLR_CKPT=/path/to/checkpoint.pth."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ours-selected-path", required=True)
    parser.add_argument("--gt-fullgraph-candidates-path", required=True)
    parser.add_argument("--molclr-root", default=None)
    parser.add_argument("--molclr-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/hpc/molclr_ccrcov_embeddings")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--encoder-type", default="gin")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--invalid-policy", choices=["error", "skip", "zero"], default="skip")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    molclr_root = discover_molclr_root(args.molclr_root)
    molclr_checkpoint = discover_molclr_checkpoint(args.molclr_checkpoint)
    print(f"[MOLCLR_DISCOVERY] molclr_root={molclr_root}", flush=True)
    print(f"[MOLCLR_DISCOVERY] molclr_checkpoint={molclr_checkpoint}", flush=True)
    summary = precompute_molclr_embeddings_for_ccrcov(
        dataset_csv=args.dataset_csv,
        ours_selected_path=args.ours_selected_path,
        gt_fullgraph_candidates_path=args.gt_fullgraph_candidates_path,
        molclr_root=molclr_root,
        molclr_checkpoint=molclr_checkpoint,
        output_dir=args.output_dir,
        label=args.label,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        max_parents=args.max_parents,
        max_candidates=args.max_candidates,
        encoder_type=args.encoder_type,
        batch_size=args.batch_size,
        device=args.device,
        invalid_policy=args.invalid_policy,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
