"""Mutagenicity adapter for the existing SFT v3 and stable-PPO data contracts."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.chem import parse_smiles
from src.data.hiv_dataset_utils import (
    HIVParentRecord,
    parent_atom_count_bin,
    sample_records_by_strata,
)
from src.data.mutagenicity import write_csv, write_json
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.data.sft_column_compat import normalize_completion_text
from src.data.sft_v3_builder import (
    ParentBuildResult,
    SFTV3BuilderConfig,
    select_reference_candidate_for_parent,
)
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer
from src.utils.io import write_jsonl


SOURCE_LABEL = 1
TARGET_LABEL = 0
SOURCE_SEMANTIC_LABEL = "mutagenic"
DEFAULT_EXPECTED_COUNTS = {
    "train": 1448,
    "val": 260,
    "calibration": 235,
    "test": 217,
}

BASE_OUTPUT_FIELDS = (
    "molecule_id",
    "parent_smiles",
    "smiles",
    "label",
    "source_label",
    "target_label",
    "semantic_label",
    "split",
    "scaffold_smiles",
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
)
SFT_OUTPUT_FIELDS = BASE_OUTPUT_FIELDS + (
    "prompt",
    "completion",
    "instruction",
    "output",
    "response",
    "reference_fragment",
    "raw_fragment",
    "core_fragment",
    "candidate_strategy",
    "atom_ratio",
    "residual_smiles",
    "cf_drop",
    "cf_flip",
    "oracle_ok",
)
PPO_OUTPUT_FIELDS = BASE_OUTPUT_FIELDS + ("prompt",)
MANIFEST_FIELDS = BASE_OUTPUT_FIELDS + (
    "source_row_index",
    "sft_target_available",
    "sft_drop_reason",
    "candidate_count",
    "valid_candidate_count",
    "candidate_strategy",
    "core_fragment",
)


@dataclass(frozen=True, slots=True)
class MutagenicityParent:
    """One validated teacher-consistent Mutagenicity source parent."""

    molecule_id: str
    source_row_index: int
    source_smiles: str
    parent_smiles: str
    label: int
    semantic_label: str
    split: str
    scaffold_smiles: str
    teacher_pred: int
    teacher_prob_0: float
    teacher_prob_1: float
    teacher_correct: bool
    parent_atom_count: int

    def to_sft_v3_parent(self) -> HIVParentRecord:
        """Adapt to the existing parent-derived SFT v3 candidate selector."""

        return HIVParentRecord(
            sample_id=self.molecule_id,
            source_row_index=self.source_row_index,
            source_smiles=self.source_smiles,
            parent_smiles=self.parent_smiles,
            label=self.label,
            raw_label=self.label,
            parent_atom_count=self.parent_atom_count,
            scaffold_smiles=self.scaffold_smiles,
            size_bin=parent_atom_count_bin(self.parent_atom_count),
        )

    def base_output_row(self) -> dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "parent_smiles": self.parent_smiles,
            "smiles": self.parent_smiles,
            "label": self.label,
            "source_label": SOURCE_LABEL,
            "target_label": TARGET_LABEL,
            "semantic_label": self.semantic_label,
            "split": self.split,
            "scaffold_smiles": self.scaffold_smiles,
            "teacher_pred": self.teacher_pred,
            "teacher_prob_0": self.teacher_prob_0,
            "teacher_prob_1": self.teacher_prob_1,
            "teacher_correct": self.teacher_correct,
        }


@dataclass(frozen=True, slots=True)
class MutagenicitySFTPPOConfig:
    """Controls that must stay aligned with the AIDS SFT v3 target builder."""

    seed: int = 42
    max_train_parents: int | None = None
    max_val_parents: int | None = None
    min_atom_ratio: float = 0.10
    max_atom_ratio: float = 0.55
    min_frag_atoms: int = 3
    max_frag_atoms: int = 30
    max_candidates_per_parent: int = 160
    use_teacher_ranking: bool = True

    def sft_v3_config(self, teacher_path: str | Path | None) -> SFTV3BuilderConfig:
        return SFTV3BuilderConfig(
            positive_label=SOURCE_LABEL,
            neg_pos_ratio=0.0,
            seed=self.seed,
            val_ratio=0.1,
            min_atom_ratio=self.min_atom_ratio,
            max_atom_ratio=self.max_atom_ratio,
            min_frag_atoms=self.min_frag_atoms,
            max_frag_atoms=self.max_frag_atoms,
            oracle_path=teacher_path,
            use_oracle_ranking=self.use_teacher_ranking,
            max_candidates_per_parent=self.max_candidates_per_parent,
            include_label_in_prompt=False,
        )


def _coerce_binary_label(value: Any, *, field: str, row_index: int) -> int:
    text = str(value).strip().lower()
    aliases = {"0": 0, "0.0": 0, "false": 0, "1": 1, "1.0": 1, "true": 1}
    if text not in aliases:
        raise ValueError(f"row={row_index} has invalid {field}={value!r}")
    return aliases[text]


def _coerce_bool(value: Any, *, field: str, row_index: int) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "no", "n"}:
        return False
    raise ValueError(f"row={row_index} has invalid {field}={value!r}")


def _coerce_probability(value: Any, *, field: str, row_index: int) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row={row_index} has invalid {field}={value!r}") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"row={row_index} has out-of-range {field}={value!r}")
    return probability


def load_teacher_consistent_parents(
    path: str | Path,
    *,
    expected_split: str,
    expected_count: int | None,
) -> list[MutagenicityParent]:
    """Load and strictly validate one teacher-consistent source-label split."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Teacher-consistent split is missing or empty: {source_path}")

    required = {
        "molecule_id",
        "smiles",
        "label",
        "semantic_label",
        "split",
        "scaffold_smiles",
        "teacher_pred",
        "teacher_prob_0",
        "teacher_prob_1",
        "teacher_correct",
    }
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{source_path} is missing required columns: {missing}")
        raw_rows = [dict(row) for row in reader]

    if expected_count is not None and expected_count > 0 and len(raw_rows) != expected_count:
        raise ValueError(
            f"{expected_split} parent count mismatch: expected={expected_count} actual={len(raw_rows)}"
        )

    parents: list[MutagenicityParent] = []
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    for row_index, row in enumerate(raw_rows):
        molecule_id = str(row.get("molecule_id") or "").strip()
        if not molecule_id:
            raise ValueError(f"row={row_index} has an empty molecule_id in {source_path}")
        if molecule_id in seen_ids:
            raise ValueError(f"duplicate molecule_id in {source_path}: {molecule_id}")

        raw_smiles = str(row.get("smiles") or "").strip()
        parsed = parse_smiles(
            raw_smiles,
            sanitize=True,
            canonicalize=True,
            allow_capped_fragments=False,
        )
        if not parsed.sanitized or parsed.mol is None or not parsed.canonical_smiles:
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has invalid SMILES: {raw_smiles!r}"
            )
        canonical_smiles = str(parsed.canonical_smiles).strip()
        if "." in raw_smiles or "." in canonical_smiles:
            raise ValueError(f"row={row_index} molecule_id={molecule_id} is multicomponent")
        if parsed.mol.GetNumAtoms() <= 0:
            raise ValueError(f"row={row_index} molecule_id={molecule_id} is empty")
        if any(atom.GetAtomicNum() == 0 for atom in parsed.mol.GetAtoms()):
            raise ValueError(f"row={row_index} molecule_id={molecule_id} contains a dummy atom")
        if canonical_smiles in seen_smiles:
            raise ValueError(f"duplicate canonical SMILES in {source_path}: {canonical_smiles}")

        label = _coerce_binary_label(row.get("label"), field="label", row_index=row_index)
        teacher_pred = _coerce_binary_label(
            row.get("teacher_pred"), field="teacher_pred", row_index=row_index
        )
        teacher_correct = _coerce_bool(
            row.get("teacher_correct"), field="teacher_correct", row_index=row_index
        )
        split = str(row.get("split") or "").strip().lower()
        semantic_label = str(row.get("semantic_label") or "").strip().lower()
        source_label_value = row.get("source_label")
        target_label_value = row.get("target_label")

        if split != expected_split:
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has split={split!r}; "
                f"expected {expected_split!r}"
            )
        if label != SOURCE_LABEL or teacher_pred != SOURCE_LABEL or not teacher_correct:
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} violates source-label/teacher filter: "
                f"label={label} teacher_pred={teacher_pred} teacher_correct={teacher_correct}"
            )
        if semantic_label != SOURCE_SEMANTIC_LABEL:
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has semantic_label={semantic_label!r}"
            )
        if source_label_value not in (None, "") and _coerce_binary_label(
            source_label_value, field="source_label", row_index=row_index
        ) != SOURCE_LABEL:
            raise ValueError(f"row={row_index} has source_label != {SOURCE_LABEL}")
        if target_label_value not in (None, "") and _coerce_binary_label(
            target_label_value, field="target_label", row_index=row_index
        ) != TARGET_LABEL:
            raise ValueError(f"row={row_index} has target_label != {TARGET_LABEL}")

        parents.append(
            MutagenicityParent(
                molecule_id=molecule_id,
                source_row_index=row_index,
                source_smiles=raw_smiles,
                parent_smiles=canonical_smiles,
                label=label,
                semantic_label=semantic_label,
                split=split,
                scaffold_smiles=str(row.get("scaffold_smiles") or "").strip(),
                teacher_pred=teacher_pred,
                teacher_prob_0=_coerce_probability(
                    row.get("teacher_prob_0"), field="teacher_prob_0", row_index=row_index
                ),
                teacher_prob_1=_coerce_probability(
                    row.get("teacher_prob_1"), field="teacher_prob_1", row_index=row_index
                ),
                teacher_correct=teacher_correct,
                parent_atom_count=int(parsed.mol.GetNumAtoms()),
            )
        )
        seen_ids.add(molecule_id)
        seen_smiles.add(canonical_smiles)
    return parents


def _set_audit(
    left: Sequence[MutagenicityParent],
    right: Sequence[MutagenicityParent],
) -> dict[str, list[str]]:
    left_ids = {row.molecule_id for row in left}
    right_ids = {row.molecule_id for row in right}
    left_smiles = {row.parent_smiles for row in left}
    right_smiles = {row.parent_smiles for row in right}
    left_scaffolds = {row.scaffold_smiles for row in left}
    right_scaffolds = {row.scaffold_smiles for row in right}
    return {
        "molecule_id_overlap": sorted(left_ids & right_ids),
        "canonical_smiles_overlap": sorted(left_smiles & right_smiles),
        "scaffold_overlap": sorted(left_scaffolds & right_scaffolds),
    }


def validate_source_isolation(
    train: Sequence[MutagenicityParent],
    val: Sequence[MutagenicityParent],
    calibration: Sequence[MutagenicityParent],
    test: Sequence[MutagenicityParent],
) -> dict[str, Any]:
    """Verify train/val isolation and calibration/test exclusion before output."""

    pair_audits = {
        "train_vs_val": _set_audit(train, val),
        "train_vs_calibration": _set_audit(train, calibration),
        "train_vs_test": _set_audit(train, test),
        "val_vs_calibration": _set_audit(val, calibration),
        "val_vs_test": _set_audit(val, test),
    }
    violations: list[str] = []
    for pair_name, audit in pair_audits.items():
        for field, values in audit.items():
            if values:
                violations.append(f"{pair_name}:{field}:{len(values)}")
    payload = {
        "source_kind": "teacher_consistent_source_label1",
        "calibration_and_test_usage": "exclusion_audit_only",
        "pair_audits": pair_audits,
        "violations": violations,
        "leakage_free": not violations,
    }
    if violations:
        raise ValueError("Mutagenicity split leakage detected: " + ", ".join(violations))
    return payload


def deterministic_parent_sample(
    parents: Sequence[MutagenicityParent],
    *,
    max_parents: int | None,
    seed: int,
) -> list[MutagenicityParent]:
    """Use the AIDS scaffold+size stratified sampler for deterministic smoke subsets."""

    if max_parents is None or max_parents <= 0 or max_parents >= len(parents):
        return list(parents)
    by_id = {parent.molecule_id: parent for parent in parents}
    sampled = sample_records_by_strata(
        [parent.to_sft_v3_parent() for parent in parents],
        sample_size=int(max_parents),
        seed=int(seed),
    )
    return [by_id[row.sample_id] for row in sampled]


def _build_teacher_scorer(
    teacher_path: str | Path | None,
    *,
    use_teacher_ranking: bool,
) -> CounterfactualTeacherScorer | None:
    if not use_teacher_ranking:
        return None
    if teacher_path is None:
        raise ValueError("Teacher ranking is enabled but no teacher_path was provided")
    resolved = Path(teacher_path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Mutagenicity RF teacher is missing or empty: {resolved}")
    scorer = CounterfactualTeacherScorer(resolved, device="cpu")
    if not scorer.available:
        raise RuntimeError(
            "Mutagenicity RF teacher could not be loaded for SFT target ranking: "
            f"{scorer.availability_reason}"
        )
    return scorer


def _materialize_split(
    parents: Sequence[MutagenicityParent],
    *,
    sft_config: SFTV3BuilderConfig,
    scorer: CounterfactualTeacherScorer | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sft_rows: list[dict[str, Any]] = []
    ppo_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    strategy_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()

    for parent in parents:
        result: ParentBuildResult = select_reference_candidate_for_parent(
            parent.to_sft_v3_parent(),
            config=sft_config,
            oracle_scorer=scorer,
        )
        base = parent.base_output_row()
        ppo_prompt = build_counterfactual_prompt(
            MoleculeRecord(
                record_id=parent.molecule_id,
                smiles=parent.parent_smiles,
                label=parent.label,
            ),
            include_label=True,
        )
        ppo_rows.append({**base, "prompt": ppo_prompt})

        candidate = result.selected_candidate
        if candidate is None:
            drop_reason = result.drop_reason or "no_sft_target"
            drop_counts[drop_reason] += 1
            manifest_rows.append(
                {
                    **base,
                    "source_row_index": parent.source_row_index,
                    "sft_target_available": False,
                    "sft_drop_reason": drop_reason,
                    "candidate_count": result.candidate_count,
                    "valid_candidate_count": result.valid_candidate_count,
                    "candidate_strategy": "",
                    "core_fragment": "",
                }
            )
            continue

        sft_prompt = build_counterfactual_prompt(
            MoleculeRecord(
                record_id=parent.molecule_id,
                smiles=parent.parent_smiles,
                label=parent.label,
            ),
            include_label=False,
        )
        completion = normalize_completion_text(candidate.core_fragment)
        sft_rows.append(
            {
                **base,
                "prompt": sft_prompt,
                "completion": completion,
                "instruction": sft_prompt,
                "output": candidate.core_fragment,
                "response": candidate.core_fragment,
                "reference_fragment": candidate.core_fragment,
                "raw_fragment": candidate.raw_fragment,
                "core_fragment": candidate.core_fragment,
                "candidate_strategy": candidate.candidate_strategy,
                "atom_ratio": candidate.atom_ratio,
                "residual_smiles": candidate.residual_smiles,
                "cf_drop": candidate.cf_drop,
                "cf_flip": candidate.cf_flip,
                "oracle_ok": candidate.oracle_ok,
            }
        )
        strategy_counts[candidate.candidate_strategy] += 1
        manifest_rows.append(
            {
                **base,
                "source_row_index": parent.source_row_index,
                "sft_target_available": True,
                "sft_drop_reason": "",
                "candidate_count": result.candidate_count,
                "valid_candidate_count": result.valid_candidate_count,
                "candidate_strategy": candidate.candidate_strategy,
                "core_fragment": candidate.core_fragment,
            }
        )

    unique_ppo_parents = {row["molecule_id"] for row in ppo_rows}
    unique_sft_parents = {row["molecule_id"] for row in sft_rows}
    summary = {
        "num_source_parents": len(parents),
        "num_sft_rows": len(sft_rows),
        "num_ppo_rows": len(ppo_rows),
        "num_unique_sft_parents": len(unique_sft_parents),
        "num_unique_ppo_parents": len(unique_ppo_parents),
        "sft_unique_parent_coverage": (
            len(unique_sft_parents) / len(parents) if parents else 0.0
        ),
        "ppo_unique_parent_coverage": (
            len(unique_ppo_parents) / len(parents) if parents else 0.0
        ),
        "candidate_strategy_counts": dict(sorted(strategy_counts.items())),
        "sft_drop_reason_counts": dict(sorted(drop_counts.items())),
    }
    return sft_rows, ppo_rows, manifest_rows, summary


def _validate_output_contract(
    train_parents: Sequence[MutagenicityParent],
    val_parents: Sequence[MutagenicityParent],
    train_sft: Sequence[Mapping[str, Any]],
    val_sft: Sequence[Mapping[str, Any]],
    train_ppo: Sequence[Mapping[str, Any]],
    val_ppo: Sequence[Mapping[str, Any]],
) -> None:
    for split_name, parents, sft_rows, ppo_rows in (
        ("train", train_parents, train_sft, train_ppo),
        ("val", val_parents, val_sft, val_ppo),
    ):
        parent_ids = {row.molecule_id for row in parents}
        ppo_ids = [str(row["molecule_id"]) for row in ppo_rows]
        if len(ppo_ids) != len(set(ppo_ids)) or set(ppo_ids) != parent_ids:
            raise ValueError(f"{split_name} PPO output must contain exactly one row per parent")
        for row in (*sft_rows, *ppo_rows):
            if int(row["label"]) != SOURCE_LABEL:
                raise ValueError(f"{split_name} output contains a non-source label")
            if int(row["teacher_pred"]) != SOURCE_LABEL or not bool(row["teacher_correct"]):
                raise ValueError(f"{split_name} output violates the teacher-consistent contract")
        for row in sft_rows:
            core = str(row.get("core_fragment") or "").strip()
            completion = str(row.get("completion") or "")
            if not core or "*" in core or completion.strip() != core:
                raise ValueError(f"{split_name} contains an unusable SFT completion")


def _report_text(summary: Mapping[str, Any], leakage_audit: Mapping[str, Any]) -> str:
    train = summary["splits"]["train"]
    val = summary["splits"]["val"]
    return "\n".join(
        [
            "# Mutagenicity SFT/PPO Dataset Report",
            "",
            "- Dataset: Mutagenicity v1 teacher-consistent source-label view",
            "- Direction: mutagenic (1) -> non_mutagenic (0)",
            "- SFT target source: existing AIDS SFT v3 parent-derived candidate selector",
            "- Calibration/test usage: exclusion audit only; never target construction",
            f"- Teacher ranking enabled: {summary['use_teacher_ranking']}",
            "",
            "## Counts",
            "",
            f"- Train source/PPO/SFT: {train['num_source_parents']} / "
            f"{train['num_ppo_rows']} / {train['num_sft_rows']}",
            f"- Validation source/PPO/SFT: {val['num_source_parents']} / "
            f"{val['num_ppo_rows']} / {val['num_sft_rows']}",
            f"- Train SFT target coverage: {train['sft_unique_parent_coverage']:.6f}",
            f"- Validation SFT target coverage: {val['sft_unique_parent_coverage']:.6f}",
            "",
            "## Leakage",
            "",
            f"- Leakage free: {leakage_audit['leakage_free']}",
            f"- Violations: `{leakage_audit['violations']}`",
            "",
            "PPO CSVs retain `molecule_id`, `parent_smiles`, `label`, and `prompt`; "
            "the existing stable PPO loader resolves these columns directly. Dataset-level "
            "parent coverage logging can therefore key future trainer instrumentation by "
            "the stable molecule ID without changing this build artifact.",
        ]
    ) + "\n"


def build_mutagenicity_sft_ppo_data(
    *,
    train_input: str | Path,
    val_input: str | Path,
    calibration_exclusion_input: str | Path,
    test_exclusion_input: str | Path,
    teacher_path: str | Path | None,
    output_dir: str | Path,
    config: MutagenicitySFTPPOConfig,
    expected_counts: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    """Build all requested artifacts without using calibration/test as examples."""

    expected = dict(DEFAULT_EXPECTED_COUNTS)
    if expected_counts is not None:
        expected.update(expected_counts)
    train_all = load_teacher_consistent_parents(
        train_input, expected_split="train", expected_count=expected.get("train")
    )
    val_all = load_teacher_consistent_parents(
        val_input, expected_split="val", expected_count=expected.get("val")
    )
    calibration = load_teacher_consistent_parents(
        calibration_exclusion_input,
        expected_split="calibration",
        expected_count=expected.get("calibration"),
    )
    test = load_teacher_consistent_parents(
        test_exclusion_input, expected_split="test", expected_count=expected.get("test")
    )
    leakage_audit = validate_source_isolation(train_all, val_all, calibration, test)

    train = deterministic_parent_sample(
        train_all, max_parents=config.max_train_parents, seed=config.seed
    )
    val = deterministic_parent_sample(
        val_all, max_parents=config.max_val_parents, seed=config.seed + 1
    )
    scorer = _build_teacher_scorer(
        teacher_path, use_teacher_ranking=config.use_teacher_ranking
    )
    sft_config = config.sft_v3_config(teacher_path)
    train_sft, train_ppo, train_manifest, train_summary = _materialize_split(
        train, sft_config=sft_config, scorer=scorer
    )
    val_sft, val_ppo, val_manifest, val_summary = _materialize_split(
        val, sft_config=sft_config, scorer=scorer
    )
    if not train_sft or not val_sft:
        raise ValueError(
            "The existing AIDS SFT v3 target constructor produced no usable targets for "
            "one or more Mutagenicity splits; no pseudo-target fallback is allowed"
        )
    _validate_output_contract(train, val, train_sft, val_sft, train_ppo, val_ppo)

    selected_audit = validate_source_isolation(train, val, calibration, test)
    leakage_audit.update(
        {
            "input_paths": {
                "train": str(Path(train_input).expanduser().resolve()),
                "val": str(Path(val_input).expanduser().resolve()),
                "calibration_exclusion": str(
                    Path(calibration_exclusion_input).expanduser().resolve()
                ),
                "test_exclusion": str(Path(test_exclusion_input).expanduser().resolve()),
            },
            "selected_pair_audits": selected_audit["pair_audits"],
            "selected_train_count": len(train),
            "selected_val_count": len(val),
            "calibration_exclusion_count": len(calibration),
            "test_exclusion_count": len(test),
        }
    )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "sft_train_csv": out / "mutagenicity_sft_train.csv",
        "sft_val_csv": out / "mutagenicity_sft_val.csv",
        "sft_train_jsonl": out / "mutagenicity_sft_train.jsonl",
        "sft_val_jsonl": out / "mutagenicity_sft_val.jsonl",
        "ppo_train_csv": out / "mutagenicity_ppo_prompts_train_label1.csv",
        "ppo_val_csv": out / "mutagenicity_ppo_prompts_val_label1.csv",
        "manifest_train_csv": out / "parent_manifest_train.csv",
        "manifest_val_csv": out / "parent_manifest_val.csv",
        "summary_json": out / "dataset_summary.json",
        "leakage_json": out / "leakage_audit.json",
        "report_md": out / "dataset_report.md",
    }
    write_csv(paths["sft_train_csv"], train_sft, SFT_OUTPUT_FIELDS)
    write_csv(paths["sft_val_csv"], val_sft, SFT_OUTPUT_FIELDS)
    write_jsonl(paths["sft_train_jsonl"], train_sft)
    write_jsonl(paths["sft_val_jsonl"], val_sft)
    write_csv(paths["ppo_train_csv"], train_ppo, PPO_OUTPUT_FIELDS)
    write_csv(paths["ppo_val_csv"], val_ppo, PPO_OUTPUT_FIELDS)
    write_csv(paths["manifest_train_csv"], train_manifest, MANIFEST_FIELDS)
    write_csv(paths["manifest_val_csv"], val_manifest, MANIFEST_FIELDS)

    summary: dict[str, Any] = {
        "dataset": "Mutagenicity",
        "dataset_version": "v1",
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "source_semantics": SOURCE_SEMANTIC_LABEL,
        "target_semantics": "non_mutagenic",
        "target_constructor": "src.data.sft_v3_builder.select_reference_candidate_for_parent",
        "target_format": "core_no_dummy",
        "raw_fragment_format": "attachment_points_may_use_dummy_atoms_for_audit",
        "use_teacher_ranking": config.use_teacher_ranking,
        "teacher_path": str(Path(teacher_path).expanduser().resolve()) if teacher_path else None,
        "inputs": dict(leakage_audit["input_paths"]),
        "build_config": {
            "seed": config.seed,
            "max_train_parents": config.max_train_parents,
            "max_val_parents": config.max_val_parents,
            "sampling_method": "scaffold_size_stratified_when_capped",
            "min_atom_ratio": config.min_atom_ratio,
            "max_atom_ratio": config.max_atom_ratio,
            "min_frag_atoms": config.min_frag_atoms,
            "max_frag_atoms": config.max_frag_atoms,
            "max_candidates_per_parent": config.max_candidates_per_parent,
        },
        "source_counts_before_smoke_sampling": {
            "train": len(train_all),
            "val": len(val_all),
            "calibration": len(calibration),
            "test": len(test),
        },
        "splits": {"train": train_summary, "val": val_summary},
        "output_columns": {
            "sft": list(SFT_OUTPUT_FIELDS),
            "ppo": list(PPO_OUTPUT_FIELDS),
            "parent_manifest": list(MANIFEST_FIELDS),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "calibration_test_used_for_training": False,
        "leakage_audit_passed": bool(leakage_audit["leakage_free"]),
    }
    write_json(paths["summary_json"], summary)
    write_json(paths["leakage_json"], leakage_audit)
    paths["report_md"].write_text(_report_text(summary, leakage_audit), encoding="utf-8")
    return summary


__all__ = [
    "BASE_OUTPUT_FIELDS",
    "DEFAULT_EXPECTED_COUNTS",
    "MANIFEST_FIELDS",
    "MutagenicityParent",
    "MutagenicitySFTPPOConfig",
    "PPO_OUTPUT_FIELDS",
    "SFT_OUTPUT_FIELDS",
    "SOURCE_LABEL",
    "TARGET_LABEL",
    "build_mutagenicity_sft_ppo_data",
    "deterministic_parent_sample",
    "load_teacher_consistent_parents",
    "validate_source_isolation",
]
