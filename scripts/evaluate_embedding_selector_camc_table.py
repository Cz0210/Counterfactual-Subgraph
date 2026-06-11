#!/usr/bin/env python3
"""Build the final CAMC-style table for embedding-MMR selected top20 fragments.

The script evaluates the selected top20 against the evidence already present in
the selector candidate pools. It does not retrain or re-run the teacher model.
When GT rows use proxy-filled cf_drop=0.0, the report marks that field as not a
teacher re-evaluated counterfactual strength.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.candidate_pool_audit import _canonical_fragment_key  # noqa: E402
from src.eval.subgraph_similarity import (  # noqa: E402
    cosine_embedding_similarity,
    get_candidate_embedding,
)
from src.utils.io import read_jsonl  # noqa: E402

try:  # pragma: no cover - depends on runtime env
    from rdkit import DataStructs
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - depends on runtime env
    DataStructs = None
    Chem = None
    rdFingerprintGenerator = None


DEFAULT_OURS_POOL = Path(
    "outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/"
    "candidate_pool_with_embeddings.jsonl"
)
DEFAULT_GT_POOL_TEMPLATE = (
    "outputs/hpc/comparison/hiv_quick/{seed}/gt_fullgraph_candidate_pool_with_embeddings.jsonl"
)
GRID_PATTERN = re.compile(
    r"beta_(?P<beta>\d+(?:p\d+)?)_gamma_(?P<gamma>\d+(?:p\d+)?)$"
)
FRAGMENT_KEYS = (
    "fragment",
    "final_fragment",
    "final_fragment_smiles",
    "motif_smiles",
    "canonical_motif_smiles",
    "core_fragment",
)
PARENT_KEYS = (
    "parent_smiles",
    "input_smiles",
    "original_smiles",
    "molecule_smiles",
    "smiles",
    "SMILES",
    "Smiles",
)
LABEL_KEYS = ("label", "target_label", "y", "Y", "HIV_active", "hiv_active")
DISTANCE_KEYS = ("distance_proxy", "nearest_distance_proxy", "mcs_distance_proxy", "cost")
TABLE_COLUMNS = [
    "method",
    "seed",
    "beta_coverage",
    "gamma_redundancy",
    "selected_count",
    "candidate_rows_matched",
    "num_inputs",
    "support_coverage",
    "camc_flip_coverage",
    "camc_at_0p5",
    "mean_cf_drop_covered",
    "pairwise_embedding_cosine_mean",
    "pairwise_embedding_cosine_max",
    "pairwise_tanimoto_mean",
    "pairwise_tanimoto_max",
    "coverage_theta_0p20",
    "theta_coverage_source",
    "cf_drop_proxy_filled",
    "selector_dir",
    "pool_jsonl",
]
AGG_NUMERIC_COLUMNS = [
    "selected_count",
    "candidate_rows_matched",
    "num_inputs",
    "support_coverage",
    "camc_flip_coverage",
    "camc_at_0p5",
    "mean_cf_drop_covered",
    "pairwise_embedding_cosine_mean",
    "pairwise_embedding_cosine_max",
    "pairwise_tanimoto_mean",
    "pairwise_tanimoto_max",
    "coverage_theta_0p20",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for Slurm wrapper parity.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for Slurm wrapper parity.",
    )
    parser.add_argument("--ours-selector-dir", required=True)
    parser.add_argument("--gt-selector-root", required=True)
    parser.add_argument("--gt-seeds", default="label1_1594411,label1_1594412,label1_1594413")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.20)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ours-pool-jsonl", default=str(DEFAULT_OURS_POOL))
    parser.add_argument("--gt-pool-template", default=DEFAULT_GT_POOL_TEMPLATE)
    parser.add_argument(
        "--gt-selector-dir-name",
        default="",
        help="Optional GT grid directory name. Defaults to the basename of --ours-selector-dir.",
    )
    parser.add_argument("--embedding-field", default="final_fragment_embedding")
    return parser


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "nan"}:
        return None
    if text in {"1", "true", "t", "yes", "y", "on", "ok", "success"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "fail", "failed"}:
        return False
    numeric = _as_float(value)
    if numeric is not None:
        return bool(numeric)
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _coerce_label(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "null", "nan"}:
            return text
    return None


def _grid_from_dir(path: Path) -> tuple[float | None, float | None, str]:
    match = GRID_PATTERN.match(path.name)
    if not match:
        return None, None, path.name
    return (
        float(match.group("beta").replace("p", ".")),
        float(match.group("gamma").replace("p", ".")),
        path.name,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_fragments(selector_dir: Path) -> tuple[list[str], dict[str, str]]:
    selector_dir = selector_dir.expanduser().resolve()
    if not selector_dir.exists():
        raise FileNotFoundError(f"selector directory not found: {selector_dir}")

    discovered = {
        "selector_summary": str(selector_dir / "selector_summary.json")
        if (selector_dir / "selector_summary.json").exists()
        else "",
        "selected_subgraphs_json": str(selector_dir / "selected_subgraphs.json")
        if (selector_dir / "selected_subgraphs.json").exists()
        else "",
        "selected_fragments_json": str(selector_dir / "selected_fragments.json")
        if (selector_dir / "selected_fragments.json").exists()
        else "",
    }
    fragments: list[str] = []
    seen: set[str] = set()

    def add(fragment_value: Any) -> None:
        fragment = str(fragment_value or "").strip()
        if not fragment:
            return
        key = _canonical_fragment_key(fragment)
        dedup_key = key or fragment
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        fragments.append(fragment)

    selected_json = selector_dir / "selected_subgraphs.json"
    if selected_json.exists():
        payload = _read_json(selected_json)
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    add(_first_text(row, FRAGMENT_KEYS))
                else:
                    add(row)

    selected_fragments_json = selector_dir / "selected_fragments.json"
    if selected_fragments_json.exists():
        payload = _read_json(selected_fragments_json)
        if isinstance(payload, list):
            for item in payload:
                add(item.get("fragment") if isinstance(item, dict) else item)
        elif isinstance(payload, dict):
            selected = payload.get("selected_fragments") or payload.get("fragments")
            if isinstance(selected, list):
                for item in selected:
                    add(item)

    summary_json = selector_dir / "selector_summary.json"
    if summary_json.exists():
        payload = _read_json(summary_json)
        if isinstance(payload, dict) and isinstance(payload.get("selected_fragments"), list):
            for item in payload["selected_fragments"]:
                add(item)

    if not fragments:
        raise ValueError(
            f"No selected fragments found in {selector_dir}. Discovered files={discovered}"
        )
    return fragments, discovered


def _load_dataset_denominator(dataset_path: Path, label: int) -> tuple[int, dict[str, Any]]:
    path = dataset_path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset path not found: {path}")
    parent_keys: set[str] = set()
    row_count = 0
    label_matched = 0
    source = "row_count"

    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            row_count += 1
            row_label = _coerce_label(_first_text(row, LABEL_KEYS))
            if row_label is not None and row_label != int(label):
                continue
            label_matched += 1
            parent = _first_text(row, PARENT_KEYS) or str(index)
            parent_keys.add(parent)
        source = "unique_parent_keys_jsonl"
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            label_field = next((field for field in fieldnames if field.lower() in LABEL_KEYS), None)
            parent_field = next((field for field in fieldnames if field in PARENT_KEYS), None)
            for index, row in enumerate(reader):
                row_count += 1
                if label_field is not None:
                    row_label = _coerce_label(row.get(label_field))
                    if row_label is not None and row_label != int(label):
                        continue
                label_matched += 1
                parent = str(row.get(parent_field, "")).strip() if parent_field else ""
                parent_keys.add(parent or str(index))
            source = "unique_parent_keys_csv" if parent_field else "csv_row_count"

    denominator = len(parent_keys) if parent_keys else label_matched or row_count
    return denominator, {
        "dataset_path": str(path),
        "row_count": row_count,
        "label_matched_row_count": label_matched,
        "unique_parent_key_count": len(parent_keys),
        "denominator": denominator,
        "denominator_source": source,
    }


def _get_morgan_generator() -> Any | None:
    if rdFingerprintGenerator is None:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _pairwise(values: list[Any], sim_fn: Any) -> tuple[float | None, float | None]:
    sims: list[float] = []
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            try:
                sim = sim_fn(values[left_index], values[right_index])
            except Exception:
                sim = None
            if sim is not None and math.isfinite(float(sim)):
                sims.append(float(sim))
    if not sims:
        return None, None
    return sum(sims) / len(sims), max(sims)


def _pairwise_tanimoto(fragments: list[str]) -> tuple[float | None, float | None, str | None]:
    if Chem is None or DataStructs is None:
        return None, None, "RDKit is not available"
    generator = _get_morgan_generator()
    if generator is None:
        return None, None, "MorganGenerator could not be created"
    fps = []
    for fragment in fragments:
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            continue
        try:
            fps.append(generator.GetFingerprint(mol))
        except Exception:
            continue
    if len(fps) < 2:
        return None, None, "fewer than two valid Morgan fingerprints"
    mean_value, max_value = _pairwise(fps, DataStructs.TanimotoSimilarity)
    return mean_value, max_value, None


def _pick_representative_rows(
    selected_keys: set[str],
    pool_rows: list[dict[str, Any]],
    *,
    label: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    matched_rows: list[dict[str, Any]] = []
    reps: dict[str, dict[str, Any]] = {}
    for row in pool_rows:
        row_label = _coerce_label(row.get("label"))
        if row_label is not None and row_label != int(label):
            continue
        fragment = _first_text(row, FRAGMENT_KEYS)
        key = _canonical_fragment_key(fragment)
        if key is None or key not in selected_keys:
            continue
        matched_rows.append(row)
        old = reps.get(key)
        if old is None:
            reps[key] = row
            continue
        old_score = (
            _as_float(old.get("cf_drop")) or -math.inf,
            _as_float(old.get("reward_total")) or -math.inf,
        )
        new_score = (
            _as_float(row.get("cf_drop")) or -math.inf,
            _as_float(row.get("reward_total")) or -math.inf,
        )
        if new_score > old_score:
            reps[key] = row
    return reps, matched_rows


def _distance_value(row: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in DISTANCE_KEYS:
        value = _as_float(row.get(key))
        if value is not None:
            return value, key
    atom_ratio = _as_float(row.get("atom_ratio"))
    if atom_ratio is not None:
        return atom_ratio, "atom_ratio_proxy"
    return None, None


def evaluate_selector(
    *,
    method: str,
    seed: str,
    selector_dir: Path,
    pool_jsonl: Path,
    num_inputs: int,
    label: int,
    theta: float,
    embedding_field: str,
    beta: float | None,
    gamma: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_fragments, discovered = load_selected_fragments(selector_dir)
    selected_keys = {
        key for fragment in selected_fragments if (key := _canonical_fragment_key(fragment)) is not None
    }
    if not pool_jsonl.exists():
        raise FileNotFoundError(f"candidate pool not found: {pool_jsonl}")
    pool_rows = read_jsonl(pool_jsonl)
    reps, matched_rows = _pick_representative_rows(selected_keys, pool_rows, label=label)

    support_parents = {
        parent for row in matched_rows if (parent := _first_text(row, PARENT_KEYS)) is not None
    }
    flip_parents = {
        parent
        for row in matched_rows
        if (parent := _first_text(row, PARENT_KEYS)) is not None and bool(_as_bool(row.get("cf_flip")))
    }
    delta_parents = {
        parent
        for row in matched_rows
        if (parent := _first_text(row, PARENT_KEYS)) is not None
        and (_as_float(row.get("cf_drop")) or -math.inf) > 0.5
    }

    cf_drop_by_parent: dict[str, float] = {}
    for row in matched_rows:
        parent = _first_text(row, PARENT_KEYS)
        if parent is None or not bool(_as_bool(row.get("cf_flip"))):
            continue
        cf_drop = _as_float(row.get("cf_drop"))
        if cf_drop is None:
            continue
        cf_drop_by_parent[parent] = max(cf_drop_by_parent.get(parent, -math.inf), cf_drop)

    theta_source_counts: dict[str, int] = defaultdict(int)
    theta_parents: set[str] = set()
    for row in matched_rows:
        parent = _first_text(row, PARENT_KEYS)
        if parent is None:
            continue
        distance, source = _distance_value(row)
        if distance is None or source is None:
            continue
        theta_source_counts[source] += 1
        if distance <= float(theta):
            theta_parents.add(parent)

    ordered_rep_rows = []
    for fragment in selected_fragments:
        key = _canonical_fragment_key(fragment)
        if key is not None and key in reps:
            ordered_rep_rows.append(reps[key])

    rep_embeddings = []
    embedding_failures = 0
    for row in ordered_rep_rows:
        try:
            rep_embeddings.append(get_candidate_embedding(row, embedding_field).vector)
        except Exception:
            embedding_failures += 1
    emb_mean, emb_max = _pairwise(rep_embeddings, cosine_embedding_similarity)
    tanimoto_mean, tanimoto_max, tanimoto_warning = _pairwise_tanimoto(selected_fragments)

    cf_drop_missing_rows = sum(1 for row in matched_rows if bool(row.get("cf_drop_missing")))
    cf_drop_proxy_filled = bool(matched_rows) and cf_drop_missing_rows == len(matched_rows)

    def rate(count: int) -> float:
        return float(count) / float(num_inputs) if num_inputs else 0.0

    row = {
        "method": method,
        "seed": seed,
        "beta_coverage": beta,
        "gamma_redundancy": gamma,
        "selected_count": len(selected_fragments),
        "candidate_rows_matched": len(matched_rows),
        "num_inputs": int(num_inputs),
        "support_coverage": rate(len(support_parents)),
        "camc_flip_coverage": rate(len(flip_parents)),
        "camc_at_0p5": rate(len(delta_parents)),
        "mean_cf_drop_covered": (
            sum(cf_drop_by_parent.values()) / len(cf_drop_by_parent) if cf_drop_by_parent else None
        ),
        "pairwise_embedding_cosine_mean": emb_mean,
        "pairwise_embedding_cosine_max": emb_max,
        "pairwise_tanimoto_mean": tanimoto_mean,
        "pairwise_tanimoto_max": tanimoto_max,
        "coverage_theta_0p20": rate(len(theta_parents)),
        "theta_coverage_source": ",".join(
            f"{key}:{value}" for key, value in sorted(theta_source_counts.items())
        )
        or "unavailable",
        "cf_drop_proxy_filled": cf_drop_proxy_filled,
        "selector_dir": str(selector_dir),
        "pool_jsonl": str(pool_jsonl),
    }
    diagnostics = {
        "method": method,
        "seed": seed,
        "selected_fragments": selected_fragments,
        "selected_fragment_keys": sorted(selected_keys),
        "selector_files": discovered,
        "pool_row_count": len(pool_rows),
        "matched_candidate_rows": len(matched_rows),
        "support_parent_count": len(support_parents),
        "flip_parent_count": len(flip_parents),
        "delta_0p5_parent_count": len(delta_parents),
        "theta_parent_count": len(theta_parents),
        "theta_source_counts": dict(theta_source_counts),
        "embedding_failure_count": embedding_failures,
        "tanimoto_warning": tanimoto_warning,
        "cf_drop_missing_rows": cf_drop_missing_rows,
        "cf_drop_proxy_filled": cf_drop_proxy_filled,
    }
    return row, diagnostics


def _aggregate_gt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    output: list[dict[str, Any]] = []
    for suffix, reducer in (
        ("mean", lambda values: mean(values) if values else None),
        ("std", lambda values: stdev(values) if len(values) > 1 else (0.0 if values else None)),
    ):
        payload: dict[str, Any] = {
            "method": f"gt_fullgraph_greedy_proxy_{suffix}",
            "seed": "clean_multi_seed",
            "beta_coverage": rows[0].get("beta_coverage"),
            "gamma_redundancy": rows[0].get("gamma_redundancy"),
            "theta_coverage_source": "aggregate",
            "cf_drop_proxy_filled": all(bool(row.get("cf_drop_proxy_filled")) for row in rows),
            "selector_dir": "",
            "pool_jsonl": "",
        }
        for key in AGG_NUMERIC_COLUMNS:
            values = [float(value) for row in rows if (value := _as_float(row.get(key))) is not None]
            payload[key] = reducer(values)
        output.append(payload)
    return output


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in TABLE_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "method",
        "seed",
        "support_coverage",
        "camc_flip_coverage",
        "camc_at_0p5",
        "mean_cf_drop_covered",
        "pairwise_embedding_cosine_mean",
        "pairwise_embedding_cosine_max",
        "pairwise_tanimoto_mean",
        "coverage_theta_0p20",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    *,
    rows: list[dict[str, Any]],
    dataset_metadata: dict[str, Any],
    gt_grid_dir_name: str,
) -> str:
    lines = [
        "Final CAMC Table for Embedding-MMR Selector",
        "",
        "Experiment goal: selector redundancy uses embedding cosine similarity.",
        f"dataset_denominator: {dataset_metadata.get('denominator')}",
        f"denominator_source: {dataset_metadata.get('denominator_source')}",
        f"gt_grid_dir_name: {gt_grid_dir_name}",
        "",
        "Notes:",
        "- Metrics are recomputed from selected top20 fragments and their candidate-pool evidence.",
        "- GT cf_drop is proxy-filled and should not be interpreted as teacher re-evaluated CF strength when cf_drop_proxy_filled=true.",
        "- Coverage@theta uses distance_proxy when present and atom_ratio as a fallback proxy otherwise.",
        "",
        "Rows:",
    ]
    for row in rows:
        lines.append(
            "- {method}/{seed}: support={support} flip={flip} camc@0.5={delta} "
            "cf_drop={cf_drop} emb_mean={emb} theta_cov={theta_cov} theta_source={source}".format(
                method=row.get("method"),
                seed=row.get("seed"),
                support=_fmt(row.get("support_coverage")),
                flip=_fmt(row.get("camc_flip_coverage")),
                delta=_fmt(row.get("camc_at_0p5")),
                cf_drop=_fmt(row.get("mean_cf_drop_covered")),
                emb=_fmt(row.get("pairwise_embedding_cosine_mean")),
                theta_cov=_fmt(row.get("coverage_theta_0p20")),
                source=row.get("theta_coverage_source"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ours_selector_dir = Path(args.ours_selector_dir).expanduser()
    beta, gamma, ours_grid_dir_name = _grid_from_dir(ours_selector_dir)
    gt_grid_dir_name = args.gt_selector_dir_name.strip() or ours_grid_dir_name
    gt_seeds = [seed.strip() for seed in str(args.gt_seeds).split(",") if seed.strip()]

    num_inputs, dataset_metadata = _load_dataset_denominator(
        Path(args.dataset_path),
        int(args.label),
    )

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    ours_row, ours_diag = evaluate_selector(
        method="ours_merged",
        seed="merged",
        selector_dir=ours_selector_dir,
        pool_jsonl=Path(args.ours_pool_jsonl).expanduser(),
        num_inputs=num_inputs,
        label=int(args.label),
        theta=float(args.theta),
        embedding_field=str(args.embedding_field),
        beta=beta,
        gamma=gamma,
    )
    rows.append(ours_row)
    diagnostics.append(ours_diag)

    gt_rows: list[dict[str, Any]] = []
    gt_root = Path(args.gt_selector_root).expanduser()
    for seed in gt_seeds:
        selector_dir = gt_root / seed / gt_grid_dir_name
        pool_jsonl = Path(str(args.gt_pool_template).format(seed=seed)).expanduser()
        gt_row, gt_diag = evaluate_selector(
            method="gt_fullgraph_greedy_proxy",
            seed=seed,
            selector_dir=selector_dir,
            pool_jsonl=pool_jsonl,
            num_inputs=num_inputs,
            label=int(args.label),
            theta=float(args.theta),
            embedding_field=str(args.embedding_field),
            beta=beta,
            gamma=gamma,
        )
        rows.append(gt_row)
        gt_rows.append(gt_row)
        diagnostics.append(gt_diag)

    aggregate_gt_rows = _aggregate_gt_rows(gt_rows)
    rows.extend(aggregate_gt_rows)

    output_paths = {
        "final_camc_embedding_selector_table_tsv": str(
            out_dir / "final_camc_embedding_selector_table.tsv"
        ),
        "final_camc_embedding_selector_table_md": str(
            out_dir / "final_camc_embedding_selector_table.md"
        ),
        "final_camc_embedding_selector_summary_json": str(
            out_dir / "final_camc_embedding_selector_summary.json"
        ),
        "final_camc_embedding_selector_report_txt": str(
            out_dir / "final_camc_embedding_selector_report.txt"
        ),
    }
    write_tsv(out_dir / "final_camc_embedding_selector_table.tsv", rows)
    write_markdown(out_dir / "final_camc_embedding_selector_table.md", rows)
    report_text = render_report(
        rows=rows,
        dataset_metadata=dataset_metadata,
        gt_grid_dir_name=gt_grid_dir_name,
    )

    summary = {
        "run_config": {
            "ours_selector_dir": str(ours_selector_dir),
            "gt_selector_root": str(gt_root),
            "gt_grid_dir_name": gt_grid_dir_name,
            "gt_seeds": gt_seeds,
            "dataset_path": str(Path(args.dataset_path).expanduser()),
            "label": int(args.label),
            "theta": float(args.theta),
            "embedding_field": str(args.embedding_field),
            "ours_pool_jsonl": str(Path(args.ours_pool_jsonl).expanduser()),
            "gt_pool_template": str(args.gt_pool_template),
            "metric_source": "selected_top20_candidate_pool_evidence",
        },
        "dataset_metadata": dataset_metadata,
        "rows": rows,
        "diagnostics": diagnostics,
        "warnings": [
            "GT cf_drop is proxy-filled and should not be interpreted as teacher re-evaluated CF strength."
            if any(bool(row.get("cf_drop_proxy_filled")) for row in gt_rows)
            else "",
            "Coverage@theta uses atom_ratio_proxy when no distance_proxy field is available.",
        ],
        "output_paths": output_paths,
    }
    summary["warnings"] = [warning for warning in summary["warnings"] if warning]
    (out_dir / "final_camc_embedding_selector_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "final_camc_embedding_selector_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
    print(json.dumps(output_paths, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
