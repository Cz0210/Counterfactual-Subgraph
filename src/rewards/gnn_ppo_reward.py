"""Batched frozen-GNN reward adapter for the shared decoded-chemistry PPO loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.chem import (
    enumerate_connected_hard_deletions,
    is_parent_substructure,
    normalize_core_fragment,
    parse_smiles,
    project_fragment_to_parent_subgraph,
)
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeaturizer,
)
from src.eval.counterfactual_semantics import (
    compute_counterfactual_semantics,
    source_class_margin,
)
from src.oracles.base_oracle import BaseOracle
from src.oracles.gnn_oracle import GNNOracle, sha256_file
from src.oracles.oracle_factory import build_oracle


GNN_PPO_REWARD_SCHEMA = "bace_gnn_ppo_reward_v1"
TASTE_GNN_PPO_REWARD_SCHEMA = "tastemolnet_gnn_ppo_reward_v1"

_DATASET_CONTRACTS = {
    "bace": 2,
    "tastemolnet": 3,
}


@dataclass(frozen=True, slots=True)
class GNNPPORewardConfig:
    dataset: str = "bace"
    num_classes: int = 2
    source_label: int = 1
    valid_bonus: float = 0.25
    invalid_penalty: float = -2.0
    substructure_bonus: float = 1.0
    cf_drop_weight: float = 3.0
    strict_flip_bonus: float = 2.0
    size_bonus: float = 0.25
    size_penalty: float = -0.25
    preferred_min_atom_ratio: float = 0.10
    preferred_max_atom_ratio: float = 0.65
    projection_penalty: float = -0.50
    enable_projection: bool = True
    projection_min_score: float = 0.35
    projection_max_candidates: int = 128
    projection_min_atoms: int = 2
    projection_max_atom_ratio: float = 0.75
    oracle_batch_size: int = 256

    def validate(self) -> None:
        numeric = asdict(self)
        scalar_fields = set(numeric) - {
            "dataset",
            "enable_projection",
            "num_classes",
            "source_label",
            "projection_max_candidates",
            "projection_min_atoms",
            "oracle_batch_size",
        }
        for field in scalar_fields:
            value = numeric[field]
            if (
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"GNN PPO {field} must be one finite native numeric value"
                )
        if type(self.dataset) is not str or self.dataset not in _DATASET_CONTRACTS:
            raise ValueError("GNN PPO reward dataset is unsupported")
        if type(self.num_classes) is not int or (
            self.num_classes != _DATASET_CONTRACTS[self.dataset]
        ):
            raise ValueError("GNN PPO reward dataset/class contract differs")
        if type(self.source_label) is not int or self.source_label != 1:
            raise ValueError("GNN PPO reward requires source_label=1")
        if type(self.enable_projection) is not bool:
            raise ValueError("GNN PPO enable_projection must be one native bool")
        for field in (
            "projection_max_candidates",
            "projection_min_atoms",
            "oracle_batch_size",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"GNN PPO {field} must be one positive native int")
        if not 0.0 <= self.preferred_min_atom_ratio < self.preferred_max_atom_ratio < 1.0:
            raise ValueError("Invalid preferred fragment atom-ratio window")


@dataclass(slots=True)
class _PreparedCandidate:
    index: int
    parent_id: str
    parent_smiles: str
    label: int
    raw_fragment: str
    core_fragment: str | None
    final_fragment: str | None
    parse_ok: bool
    connected: bool
    direct_substructure: bool
    projection_used: bool
    projection_reason: str | None
    projection_score: float | None
    deletions: list[Any]


def _portable_graph(
    featurizer: MolecularGraphFeaturizer,
    smiles: str,
    *,
    molecule_id: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=-1,
        molecule_id=str(molecule_id),
        smiles=features.canonical_smiles,
        split="ppo_train_only_reward",
        graph_sha256=features.graph_sha256,
    )


def _canonical_smiles(smiles: str) -> str:
    parsed = parse_smiles(smiles, sanitize=True, canonicalize=True)
    if not parsed.sanitized or not parsed.canonical_smiles:
        raise ValueError(f"GNN PPO parent is not a valid molecule: {smiles!r}")
    return str(parsed.canonical_smiles)


def _oracle_vector(
    record: Mapping[str, Any],
    field: str,
    *,
    num_classes: int,
) -> list[float]:
    raw = record.get(field)
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
        or len(raw) != num_classes
    ):
        raise RuntimeError(f"GNN oracle {field} width differs from num_classes")
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"GNN oracle {field} contains non-finite values")
    return values


class BatchedGNNPPORewardAdapter:
    """Stable-loop compatible rewarder backed by one loaded calibrated GNN.

    Parent predictions are cached across PPO updates.  All valid residuals in
    one rollout are featurized first and then submitted to exactly one batched
    oracle call; a checkpoint is never loaded from inside candidate scoring.
    """

    def __init__(
        self,
        *,
        oracle: BaseOracle,
        featurizer: MolecularGraphFeaturizer,
        checkpoint_dir: str | Path,
        policy_initializer_hash: str,
        reference_policy_hash: str,
        config: GNNPPORewardConfig | None = None,
        temperature_calibration_hash: str | None = None,
        feature_schema_hash: str | None = None,
        checkpoint_path_is_retained: bool = False,
    ) -> None:
        self.oracle = oracle
        self.featurizer = featurizer
        requested_checkpoint = Path(checkpoint_dir).expanduser()
        if checkpoint_path_is_retained:
            normalized_checkpoint = Path(os.path.abspath(requested_checkpoint))
            if (
                not requested_checkpoint.is_absolute()
                or requested_checkpoint != normalized_checkpoint
            ):
                raise ValueError(
                    "Retained PPO GNN checkpoint path must be normalized and absolute"
                )
            self.checkpoint_dir = normalized_checkpoint
        else:
            self.checkpoint_dir = requested_checkpoint.resolve(strict=True)
        self.policy_initializer_hash = str(policy_initializer_hash)
        self.reference_policy_hash = str(reference_policy_hash)
        self.config = config or GNNPPORewardConfig()
        self.config.validate()
        backbone = str(oracle.backbone).lower()
        if (
            not isinstance(oracle, GNNOracle)
            and "gnn" not in backbone
            and backbone != "gine"
        ):
            raise ValueError("PPO reward requires a frozen GNN oracle")
        if (
            type(oracle.num_classes) is not int
            or oracle.num_classes != self.config.num_classes
            or type(oracle.source_label) is not int
            or oracle.source_label != self.config.source_label
        ):
            raise ValueError("PPO GNN oracle class/source contract differs")
        if "rf" in str(oracle.backbone).lower():
            raise ValueError("RF oracle is forbidden for GNN PPO")
        if not self.policy_initializer_hash or not self.reference_policy_hash:
            raise ValueError("Policy and reference hashes are required for reward provenance")
        schema = self.featurizer.schema.to_dict()
        self.feature_schema_hash = str(schema["schema_sha256"])
        temperature_path = self.checkpoint_dir / "temperature_scaling.json"
        self.temperature_calibration_hash = (
            sha256_file(temperature_path)
            if temperature_calibration_hash is None
            else str(temperature_calibration_hash)
        )
        if feature_schema_hash is not None:
            self.feature_schema_hash = str(feature_schema_hash)
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                self.temperature_calibration_hash,
                self.feature_schema_hash,
            )
        ):
            raise ValueError("PPO GNN payload hashes are malformed")
        self._parent_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.oracle_load_count = 1
        self.oracle_prediction_batches = 0
        self.parent_cache_hits = 0
        self.parent_cache_misses = 0
        self.scored_deletion_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | Any,
        policy_initializer_hash: str,
        reference_policy_hash: str,
        config: GNNPPORewardConfig | None = None,
        verify_hashes: bool = True,
    ) -> "BatchedGNNPPORewardAdapter":
        checkpoint = Path(checkpoint_dir).expanduser().resolve(strict=True)
        effective_config = config or GNNPPORewardConfig()
        effective_config.validate()
        oracle = build_oracle(
            dataset=effective_config.dataset,
            backend="gnn",
            checkpoint=checkpoint,
            device=device,
            batch_size=effective_config.oracle_batch_size,
            num_classes=effective_config.num_classes,
            source_label=effective_config.source_label,
            verify_hashes=verify_hashes,
        )
        schema = MolecularFeatureSchema.from_dict(
            json.loads((checkpoint / "feature_schema.json").read_text(encoding="utf-8"))
        )
        return cls(
            oracle=oracle,
            featurizer=MolecularGraphFeaturizer(schema),
            checkpoint_dir=checkpoint,
            policy_initializer_hash=policy_initializer_hash,
            reference_policy_hash=reference_policy_hash,
            config=effective_config,
        )

    @classmethod
    def from_payloads(
        cls,
        payloads: Mapping[str, bytes],
        *,
        checkpoint_dir: str | Path,
        device: str | Any,
        policy_initializer_hash: str,
        reference_policy_hash: str,
        config: GNNPPORewardConfig | None = None,
    ) -> "BatchedGNNPPORewardAdapter":
        """Load the frozen GNN from retained, descriptor-read payload bytes."""

        effective_config = config or GNNPPORewardConfig()
        effective_config.validate()
        oracle = GNNOracle.from_payloads(
            payloads,
            device=device,
            batch_size=effective_config.oracle_batch_size,
            checkpoint_dir=None,
        )
        try:
            schema_payload = json.loads(payloads["feature_schema.json"].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("PPO GNN retained feature schema is malformed") from exc
        if type(schema_payload) is not dict:
            raise ValueError("PPO GNN retained feature schema must be one object")
        feature_schema = MolecularFeatureSchema.from_dict(schema_payload)
        temperature_bytes = payloads.get("temperature_scaling.json")
        if type(temperature_bytes) is not bytes or not temperature_bytes:
            raise ValueError("PPO GNN retained temperature payload is missing")
        return cls(
            oracle=oracle,
            featurizer=MolecularGraphFeaturizer(feature_schema),
            checkpoint_dir=checkpoint_dir,
            policy_initializer_hash=policy_initializer_hash,
            reference_policy_hash=reference_policy_hash,
            config=effective_config,
            temperature_calibration_hash=hashlib.sha256(
                temperature_bytes
            ).hexdigest(),
            feature_schema_hash=hashlib.sha256(
                payloads["feature_schema.json"]
            ).hexdigest(),
            checkpoint_path_is_retained=True,
        )

    def _cache_key(self, canonical_parent: str) -> tuple[str, str, str, str]:
        return (
            canonical_parent,
            str(self.oracle.checkpoint_id),
            self.temperature_calibration_hash,
            self.feature_schema_hash,
        )

    def predict_parent_records(
        self,
        *,
        parent_smiles: Sequence[str],
        metas: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Score parents once so a smoke can select true source predictions.

        This does not evaluate a fragment or create a counterfactual candidate.
        It reuses the adapter's single frozen oracle and the same parent cache
        later consumed by ``score_batch``.
        """

        metadata = list(metas or ({} for _ in parent_smiles))
        if len(metadata) != len(parent_smiles):
            raise ValueError("GNN parent prediction metadata length mismatch")
        parent_ids = [
            str(row.get("id") or row.get("molecule_id") or row.get("index") or index)
            for index, row in enumerate(metadata)
        ]
        canonical = [_canonical_smiles(str(smiles)) for smiles in parent_smiles]
        self._ensure_parent_predictions(
            list(zip(parent_ids, parent_smiles, strict=True))
        )
        records: list[dict[str, Any]] = []
        for parent_id, canonical_smiles in zip(parent_ids, canonical, strict=True):
            record = dict(self._parent_cache[self._cache_key(canonical_smiles)])
            probabilities = _oracle_vector(
                record, "probabilities", num_classes=self.config.num_classes
            )
            logits = _oracle_vector(
                record, "logits", num_classes=self.config.num_classes
            )
            prediction = record.get("predicted_label")
            if (
                type(prediction) is not int
                or prediction not in range(self.config.num_classes)
                or max(range(self.config.num_classes), key=probabilities.__getitem__)
                != prediction
            ):
                raise RuntimeError("GNN parent prediction authority drifted")
            records.append(
                {
                    **record,
                    "parent_id": parent_id,
                    "canonical_parent_smiles": canonical_smiles,
                    "probabilities": probabilities,
                    "logits": logits,
                }
            )
        return records

    def _ensure_parent_predictions(
        self, parents: Sequence[tuple[str, str]]
    ) -> None:
        missing: list[tuple[tuple[str, str, str, str], MolecularGraphData]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for parent_id, smiles in parents:
            canonical = _canonical_smiles(smiles)
            key = self._cache_key(canonical)
            if key in self._parent_cache:
                self.parent_cache_hits += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            self.parent_cache_misses += 1
            missing.append(
                (key, _portable_graph(self.featurizer, canonical, molecule_id=parent_id))
            )
        if not missing:
            return
        records = self.oracle.predict_records(
            [graph for _key, graph in missing], batch_size=self.config.oracle_batch_size
        )
        self.oracle_prediction_batches += 1
        if len(records) != len(missing):
            raise RuntimeError("GNN parent prediction cardinality mismatch")
        for (key, _graph), record in zip(missing, records, strict=True):
            self._parent_cache[key] = dict(record)

    def _prepare_one(
        self,
        *,
        index: int,
        parent_id: str,
        parent_smiles: str,
        label: int,
        raw_fragment: str,
    ) -> _PreparedCandidate:
        if type(label) is not int or label != self.config.source_label:
            raise ValueError("GNN PPO accepts train source-class parents only")
        normalized = normalize_core_fragment(raw_fragment, keep_largest_component=True)
        core = (
            str(normalized.core_fragment_smiles).strip()
            if normalized.core_parse_ok and normalized.core_fragment_smiles
            else None
        )
        parse_ok = bool(core)
        connected = bool(normalized.core_connected and core)
        try:
            direct = bool(
                core and connected and is_parent_substructure(parent_smiles, core)
            )
        except Exception:
            direct = False
        final_fragment = core if direct else None
        projection_used = False
        projection_reason: str | None = None
        projection_score: float | None = None
        if core and connected and not direct and self.config.enable_projection:
            try:
                projection = project_fragment_to_parent_subgraph(
                    parent_smiles,
                    core,
                    min_score=self.config.projection_min_score,
                    max_candidates=self.config.projection_max_candidates,
                    min_atoms=self.config.projection_min_atoms,
                    max_atom_ratio=self.config.projection_max_atom_ratio,
                )
                projection_reason = projection.reason
                projection_score = projection.projection_score
                if projection.success and projection.projected_fragment_smiles:
                    final_fragment = str(projection.projected_fragment_smiles)
                    projection_used = True
            except Exception as exc:
                projection_reason = f"projection_error:{type(exc).__name__}"
        try:
            deletions = (
                enumerate_connected_hard_deletions(
                    parent_smiles,
                    final_fragment,
                    parent_id=parent_id,
                    candidate_id=f"ppo:{index}",
                )
                if final_fragment
                else []
            )
        except Exception:
            deletions = []
        return _PreparedCandidate(
            index=index,
            parent_id=parent_id,
            parent_smiles=parent_smiles,
            label=label,
            raw_fragment=str(raw_fragment or ""),
            core_fragment=core,
            final_fragment=final_fragment,
            parse_ok=parse_ok,
            connected=connected,
            direct_substructure=direct,
            projection_used=projection_used,
            projection_reason=projection_reason,
            projection_score=projection_score,
            deletions=[
                outcome
                for outcome in deletions
                if outcome.valid and outcome.residual_smiles
            ],
        )

    def _base_row(self, candidate: _PreparedCandidate) -> dict[str, Any]:
        return {
            "schema_version": (
                TASTE_GNN_PPO_REWARD_SCHEMA
                if self.config.dataset == "tastemolnet"
                else GNN_PPO_REWARD_SCHEMA
            ),
            "dataset": self.config.dataset,
            "num_classes": self.config.num_classes,
            "parent_id": candidate.parent_id,
            "parent_smiles": candidate.parent_smiles,
            "source_label": self.config.source_label,
            "raw_fragment": candidate.raw_fragment,
            "core_fragment": candidate.core_fragment,
            "final_fragment": candidate.final_fragment,
            "parse_ok": candidate.parse_ok,
            "valid": candidate.parse_ok and candidate.connected,
            "connected": candidate.connected,
            "direct_substructure": candidate.direct_substructure,
            "projection_used": candidate.projection_used,
            "projection_method": (
                "nearest_parent_subgraph" if candidate.projection_used else "none"
            ),
            "projection_reason": candidate.projection_reason,
            "projection_score": candidate.projection_score,
            "deletion_valid": bool(candidate.deletions),
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "oracle_checkpoint_hash": str(self.oracle.checkpoint_id),
            "temperature": float(self.oracle.temperature),
            "temperature_calibration_hash": self.temperature_calibration_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "policy_initializer_hash": self.policy_initializer_hash,
            "reference_policy_hash": self.reference_policy_hash,
            "rf_oracle_used": False,
            "calibration_loaded": False,
            "calibration_dataset_loaded": False,
            "frozen_temperature_calibration_loaded": True,
            "test_loaded": False,
            "reward_kl": 0.0,
            "reward_kl_accounted_in_ppo_token_objective": True,
        }

    def score_batch(
        self,
        *,
        parent_smiles: Sequence[str],
        generated_fragments: Sequence[str],
        labels: Sequence[int],
        metas: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        size = len(parent_smiles)
        if len(generated_fragments) != size or len(labels) != size:
            raise ValueError("GNN reward batch fields have different lengths")
        metadata = list(metas or ({} for _ in range(size)))
        if len(metadata) != size:
            raise ValueError("GNN reward metadata length mismatch")
        parent_ids = [
            str(row.get("id") or row.get("molecule_id") or row.get("index") or index)
            for index, row in enumerate(metadata)
        ]
        self._ensure_parent_predictions(list(zip(parent_ids, parent_smiles, strict=True)))
        prepared = [
            self._prepare_one(
                index=index,
                parent_id=parent_ids[index],
                parent_smiles=str(parent_smiles[index]),
                label=labels[index],
                raw_fragment=str(generated_fragments[index] or ""),
            )
            for index in range(size)
        ]
        residual_graphs: list[MolecularGraphData] = []
        residual_index: list[tuple[int, Any]] = []
        for candidate in prepared:
            for outcome in candidate.deletions:
                residual_graphs.append(
                    _portable_graph(
                        self.featurizer,
                        str(outcome.residual_smiles),
                        molecule_id=f"{candidate.parent_id}:residual:{outcome.match_id}",
                    )
                )
                residual_index.append((candidate.index, outcome))
        residual_records: list[dict[str, Any]] = []
        if residual_graphs:
            residual_records = self.oracle.predict_records(
                residual_graphs, batch_size=self.config.oracle_batch_size
            )
            self.oracle_prediction_batches += 1
        if len(residual_records) != len(residual_index):
            raise RuntimeError("GNN residual prediction cardinality mismatch")
        by_candidate: dict[int, list[tuple[Any, dict[str, Any]]]] = {}
        for (candidate_index, outcome), record in zip(
            residual_index, residual_records, strict=True
        ):
            by_candidate.setdefault(candidate_index, []).append((outcome, record))

        rows: list[dict[str, Any]] = []
        for candidate in prepared:
            row = self._base_row(candidate)
            canonical_parent = _canonical_smiles(candidate.parent_smiles)
            parent_record = self._parent_cache[self._cache_key(canonical_parent)]
            before_probabilities = _oracle_vector(
                parent_record,
                "probabilities",
                num_classes=self.config.num_classes,
            )
            before_logits = _oracle_vector(
                parent_record,
                "logits",
                num_classes=self.config.num_classes,
            )
            evaluated: list[tuple[Any, dict[str, Any], Any]] = []
            for outcome, after_record in by_candidate.get(candidate.index, []):
                semantics = compute_counterfactual_semantics(
                    source_label=self.config.source_label,
                    pred_before=int(parent_record["predicted_label"]),
                    pred_after=int(after_record["predicted_label"]),
                    probabilities_before=parent_record["probabilities"],
                    probabilities_after=after_record["probabilities"],
                )
                evaluated.append((outcome, after_record, semantics))
            selected = max(
                evaluated,
                key=lambda item: (
                    int(item[2].cf_flip),
                    float(item[2].cf_drop),
                    -int(item[0].match_id),
                ),
                default=None,
            )
            reward_valid = (
                self.config.valid_bonus
                if candidate.parse_ok and candidate.connected
                else self.config.invalid_penalty
            )
            reward_substructure = (
                self.config.substructure_bonus
                if candidate.direct_substructure
                else (self.config.substructure_bonus * 0.25 if candidate.projection_used else 0.0)
            )
            reward_projection = self.config.projection_penalty if candidate.projection_used else 0.0
            reward_cf = 0.0
            reward_size = self.config.size_penalty
            if selected is None:
                row.update(
                    {
                        "match_index": None,
                        "residual_smiles": None,
                        "pred_before": int(parent_record["predicted_label"]),
                        "pred_after": None,
                        "p_before": float(parent_record["source_probability"]),
                        "p_after": None,
                        "p_before_all_classes": before_probabilities,
                        "p_after_all_classes": None,
                        "logits_before_all_classes": before_logits,
                        "logits_after_all_classes": None,
                        "source_probability_before": float(
                            parent_record["source_probability"]
                        ),
                        "source_probability_after": None,
                        "cf_drop": None,
                        "cf_flip": False,
                        "destination_label": None,
                        "margin_before": source_class_margin(
                            before_probabilities, self.config.source_label
                        ),
                        "margin_after": None,
                        "margin_drop": None,
                        "oracle_ok": False,
                        "gnn_scored_deletion": False,
                        "fragment_atom_ratio": None,
                        "atom_ratio": None,
                        "final_fragment_atom_ratio": None,
                    }
                )
            else:
                outcome, after_record, semantics = selected
                after_logits = _oracle_vector(
                    after_record,
                    "logits",
                    num_classes=self.config.num_classes,
                )
                self.scored_deletion_count += 1
                ratio = float(outcome.atom_delete_ratio or 0.0)
                reward_cf = self.config.cf_drop_weight * float(semantics.cf_drop)
                if semantics.cf_flip:
                    reward_cf += self.config.strict_flip_bonus
                reward_size = (
                    self.config.size_bonus
                    if self.config.preferred_min_atom_ratio
                    <= ratio
                    <= self.config.preferred_max_atom_ratio
                    else self.config.size_penalty
                )
                row.update(
                    {
                        "match_index": int(outcome.match_id),
                        "match_atom_indices": list(outcome.match_atom_indices),
                        "residual_smiles": outcome.residual_smiles,
                        "pred_before": semantics.pred_before,
                        "pred_after": semantics.pred_after,
                        "p_before": semantics.source_prob_before,
                        "p_after": semantics.source_prob_after,
                        "p_before_all_classes": list(semantics.p_before_all_classes),
                        "p_after_all_classes": list(semantics.p_after_all_classes),
                        "logits_before_all_classes": before_logits,
                        "logits_after_all_classes": after_logits,
                        "source_probability_before": semantics.source_prob_before,
                        "source_probability_after": semantics.source_prob_after,
                        "cf_drop": semantics.cf_drop,
                        "cf_flip": semantics.cf_flip,
                        "destination_label": semantics.destination_label,
                        "margin_before": semantics.margin_before,
                        "margin_after": semantics.margin_after,
                        "margin_drop": semantics.margin_drop,
                        "oracle_ok": True,
                        "gnn_scored_deletion": True,
                        "fragment_atom_ratio": ratio,
                        "atom_ratio": ratio,
                        "final_fragment_atom_ratio": ratio,
                        "deletion_valid": True,
                        "valid_match_count": len(evaluated),
                    }
                )
            reward_total = (
                reward_valid
                + reward_substructure
                + reward_cf
                + reward_size
                + reward_projection
            )
            components = {
                "reward_valid": reward_valid,
                "reward_substructure": reward_substructure,
                "reward_cf": reward_cf,
                "reward_size": reward_size,
                "reward_projection": reward_projection,
                "reward_kl": 0.0,
                "reward_total": reward_total,
            }
            if any(not math.isfinite(float(value)) for value in components.values()):
                raise RuntimeError("GNN PPO reward produced a non-finite value")
            row.update(components)
            row["total"] = reward_total
            rows.append(row)
        return rows

    def compute_rewards_from_decoded(
        self,
        *,
        parent_smiles: Sequence[str],
        generated_fragments: Sequence[str],
        raw_outputs: Sequence[str] | None = None,
        labels: Sequence[int],
        metas: Sequence[Mapping[str, Any]] | None = None,
        device: str | Any = "cpu",
        step_index: int | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        del raw_outputs
        rows = self.score_batch(
            parent_smiles=parent_smiles,
            generated_fragments=generated_fragments,
            labels=labels,
            metas=metas,
        )
        for row in rows:
            row["step_index"] = step_index
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("GNN PPO reward requires torch") from exc
        reward = torch.tensor(
            [float(row["reward_total"]) for row in rows],
            dtype=torch.float32,
            device=device,
        )
        return reward, rows

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "tastemolnet_gnn_ppo_oracle_provenance_v1"
                if self.config.dataset == "tastemolnet"
                else "bace_gnn_ppo_oracle_provenance_v1"
            ),
            "dataset": self.config.dataset,
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_id": str(self.oracle.checkpoint_id),
            "backbone": str(self.oracle.backbone),
            "num_classes": int(self.oracle.num_classes),
            "source_label": int(self.oracle.source_label),
            "temperature": float(self.oracle.temperature),
            "temperature_calibration_hash": self.temperature_calibration_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "policy_initializer_hash": self.policy_initializer_hash,
            "reference_policy_hash": self.reference_policy_hash,
            "oracle_load_count": self.oracle_load_count,
            "oracle_prediction_batches": self.oracle_prediction_batches,
            "parent_cache_hits": self.parent_cache_hits,
            "parent_cache_misses": self.parent_cache_misses,
            "gnn_scored_deletion_count": self.scored_deletion_count,
            "calibration_loaded": False,
            "calibration_dataset_loaded": False,
            "frozen_temperature_calibration_loaded": True,
            "test_loaded": False,
        }


__all__ = [
    "BatchedGNNPPORewardAdapter",
    "GNNPPORewardConfig",
    "GNN_PPO_REWARD_SCHEMA",
    "TASTE_GNN_PPO_REWARD_SCHEMA",
]
