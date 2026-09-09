"""Frozen-GINE/Grad-CAM and original exact-WNode adapters for CM-CReM.

This module does not generate molecules, fit classifiers, or select prototypes.
It transports only plain records.  In particular an infrastructure failure is
never converted into an empty candidate or an infinite scientific distance.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularFeatureSchema, MolecularGraphFeaturizer
from src.oracles.gnn_oracle import GNNOracle, sha256_file


UPSTREAM_COMMIT = "b5816b502cde00ee24c652a02cbc54664583f773"
ATTRIBUTION_SCHEMA = "cm_crem_frozen_gine_gradcam_v1"
ENCODING_SCHEMA = "cm_crem_original_molclr_node_encoding_v1"
RAW_DISTANCE_SCHEMA = "cm_crem_raw_fullgraph_wnode_v1"


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _record(payload: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    path = Path(payload).expanduser().resolve(strict=True)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("Resolved configuration must be a JSON object")
    return loaded


def _pin(path: Path, expected: Any, role: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing resolved {role} SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} SHA256 mismatch: {path}")
    return actual


def _state_snapshot(model: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@contextmanager
def _unchanged_model(model: Any) -> Iterable[None]:
    """Check actual tensors/BN, modes, grad flags and RNG, not pickle bytes."""
    import torch

    states = _state_snapshot(model)
    modes = {name: module.training for name, module in model.named_modules()}
    flags = {name: param.requires_grad for name, param in model.named_parameters()}
    grads = {name: None if param.grad is None else param.grad.detach().clone()
             for name, param in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    if any(modes.values()):
        raise ValueError("CM attribution requires every frozen GINE module in eval mode")
    try:
        yield
    finally:
        for name, value in model.state_dict().items():
            if not torch.equal(states[name], value):
                raise RuntimeError(f"Frozen model/BN mutated: {name}")
        if modes != {name: module.training for name, module in model.named_modules()}:
            raise RuntimeError("Frozen model mode changed")
        for name, param in model.named_parameters():
            if flags[name] != param.requires_grad:
                raise RuntimeError(f"Frozen parameter grad flag changed: {name}")
            if (grads[name] is None) != (param.grad is None):
                raise RuntimeError(f"Attribution accumulated parameter gradients: {name}")
            if grads[name] is not None and not torch.equal(grads[name], param.grad):
                raise RuntimeError(f"Attribution changed parameter gradients: {name}")
        if not torch.equal(rng, torch.random.get_rng_state()):
            raise RuntimeError("Attribution consumed Torch CPU RNG")
        if cuda_rng is not None and any(not torch.equal(a, b) for a, b in
                                      zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)):
            raise RuntimeError("Attribution consumed CUDA RNG")


def upstream_cam(activation: Any, gradient: Any) -> Any:
    """Pinned author's formula: per-node channel mean; no ReLU.

    This intentionally is *not* the conventional channel-wise/node-mean CAM.
    See upstream source/explainability.py::GradCAM.node_importances.
    """
    import torch

    if activation.ndim != 2 or activation.shape != gradient.shape:
        raise ValueError("Node activation/gradient must have identical rank-2 shapes")
    if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite Grad-CAM activation/gradient")
    cam = (activation * gradient.mean(dim=1, keepdim=True)).sum(dim=1)
    shifted = cam - cam.min()
    return shifted / (shifted.max() + 1e-9)


def upstream_top_atoms(importances: Sequence[float]) -> list[int]:
    """Official CF branch floor(n/5), stable original-index tie break."""
    values = [float(value) for value in importances]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Attribution must contain finite importances")
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:max(1, len(values) // 5)]


def _smiles(row: Mapping[str, Any]) -> str:
    value = row.get("smiles") or row.get("parent_smiles") or row.get("model_smiles")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Explicit molecule SMILES is missing")
    return value.strip()


def _parent_id(row: Mapping[str, Any]) -> str:
    value = row.get("parent_id") or row.get("molecule_id") or row.get("id")
    if value is None or str(value) == "":
        raise ValueError("Explicit parent ID is missing")
    return str(value)


def graph_identity(smiles: str, featurizer: MolecularGraphFeaturizer,
                   *, require_connected: bool = True) -> dict[str, Any]:
    """Canonical full-graph identity for unique prototype/pair records."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("INVALID_OR_EMPTY_MOLECULE")
    if require_connected and len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("DISCONNECTED_MOLECULE")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("DUMMY_ATOM")
    Chem.SanitizeMol(mol)
    atom_field = next(field for field in featurizer.schema.node_fields if field.name == "atomic_num")
    for atom in mol.GetAtoms():
        if atom_field.encode(atom.GetAtomicNum()) == atom_field.unknown_index:
            raise ValueError(f"UNSUPPORTED_ELEMENT:{atom.GetAtomicNum()}")
    if any(str(bond.GetBondType()) not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
           for bond in mol.GetBonds()):
        raise ValueError("UNSUPPORTED_BOND")
    # No stale node ordering/RWPE: reconstruct every unique full graph anew.
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    features = featurizer.featurize(canonical)
    identity = {"canonical_smiles": canonical, "graph_sha256": features.graph_sha256,
                "feature_schema_sha256": features.schema_sha256}
    return {**identity, "candidate_id": _hash(identity), "num_atoms": len(features.node_features)}


class FrozenCMOracle:
    """Same original molecular-GINE forward, plus a removable node hook."""

    def __init__(self, oracle: GNNOracle, featurizer: MolecularGraphFeaturizer,
                 *, binding: Mapping[str, Any], forward_atol: float = 0.0,
                 forward_rtol: float = 0.0) -> None:
        self.oracle = oracle
        self.featurizer = featurizer
        self.binding = dict(binding)
        self.forward_atol = float(forward_atol)
        self.forward_rtol = float(forward_rtol)
        if str(oracle.backbone).lower() != "gine" or int(oracle.num_classes) != 2:
            raise ValueError("CM-CReM primary route requires original BACE GINE, not GIN/A+")
        if oracle.source_label != 1 or not math.isfinite(oracle.temperature) or oracle.temperature <= 0:
            raise ValueError("Original BACE source/temperature contract mismatch")
        if not hasattr(oracle.model, "layers") or not oracle.model.layers:
            raise ValueError("Unreviewed GINE node-layer implementation")
        self.node_layer_name = f"layers.{len(oracle.model.layers) - 1}"
        if self.binding.get("node_layer", self.node_layer_name) != self.node_layer_name:
            raise ValueError("Grad-CAM must bind the final message layer before residual/BN")
        self.node_layer = oracle.model.layers[-1]
        if self.node_layer.__class__.__name__ != "MolecularMessageLayer":
            raise ValueError("Grad-CAM final message layer has an unreviewed implementation")
        self.oracle.model.eval()
        for param in self.oracle.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_resolved(cls, payload: Mapping[str, Any] | str | Path) -> "FrozenCMOracle":
        spec = _record(payload)
        spec = dict(spec.get("resolved_oracle", spec))
        if spec.get("dataset") != "bace" or spec.get("backbone") != "gine":
            raise ValueError("Resolved original BACE GINE authority is required")
        if spec.get("source_label") != 1 or spec.get("allowed_destinations") != [0]:
            raise ValueError("BACE direction must remain 1 -> 0")
        root = Path(spec["checkpoint_dir"]).expanduser()
        if not root.is_absolute():
            raise ValueError("Resolved checkpoint_dir must be absolute")
        for name, field in (("temperature_scaling.json", "temperature_sha256"),
                            ("feature_schema.json", "feature_schema_sha256"),
                            ("label_map.json", "label_map_sha256")):
            _pin(root / name, spec.get(field), name)
        temperature = json.loads((root / "temperature_scaling.json").read_text())
        fit_split = temperature.get("fit_split", temperature.get("selection_split"))
        if (temperature.get("status") != "fit" or fit_split not in {"validation", "val"} or
                temperature.get("num_examples") != 187 or temperature.get("test_used_for_fit") is not False):
            raise ValueError("Original validation-fit temperature receipt is required")
        # Verify one adopted bundle at load; no repeated large-package scan.
        oracle = GNNOracle.from_checkpoint(root, device=spec.get("device", "cpu"),
                                          batch_size=int(spec.get("batch_size", 32)))
        if oracle.checkpoint_id != spec.get("model_sha256"):
            raise ValueError("Original frozen GINE weight identity changed")
        card = json.loads((root / "model_card.json").read_text())
        if card.get("dataset") != "bace" or card.get("backbone") != "gine":
            raise ValueError("Checkpoint is not the original BACE GINE")
        if "temperature" in spec and float(spec["temperature"]) != oracle.temperature:
            raise ValueError("Resolved temperature differs from frozen receipt")
        schema = MolecularFeatureSchema.from_dict(json.loads((root / "feature_schema.json").read_text()))
        # Forward is run on the same device/path. Exact is the default; any
        # existing nonzero tolerance must be supplied and tied to its receipt.
        if (spec.get("forward_atol", 0) or spec.get("forward_rtol", 0)) and not spec.get("numerical_contract_sha256"):
            raise ValueError("Nonzero tolerance lacks original numerical-contract binding")
        return cls(oracle, MolecularGraphFeaturizer(schema), binding=spec,
                   forward_atol=float(spec.get("forward_atol", 0)),
                   forward_rtol=float(spec.get("forward_rtol", 0)))

    def _graph(self, smiles: str, identity: str, split: str) -> MolecularGraphData:
        f = self.featurizer.featurize(smiles)
        return MolecularGraphData(f.node_features, f.edge_index, f.edge_features,
                                  self.oracle.source_label, identity, f.canonical_smiles,
                                  split, f.graph_sha256)

    def predict_rows(self, rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
        graphs = [self._graph(_smiles(row), str(row.get("candidate_id") or _parent_id(row)), split)
                  for row in rows]
        with _unchanged_model(self.oracle.model):
            predictions = self.oracle.predict_records(graphs) if graphs else []
        return [{**dict(row), **prediction, "oracle_weight_sha256": self.oracle.checkpoint_id,
                 "full_graph_id": graph_identity(_smiles(row), self.featurizer,
                                                 require_connected=False)["candidate_id"],
                 "temperature": self.oracle.temperature}
                for row, prediction in zip(rows, predictions, strict=True)]

    def attribute_train_parent(self, parent: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from rdkit import Chem
        from src.baselines.cm_crem_generation import make_parent_request

        if parent.get("split") != "train":
            raise ValueError("CM Grad-CAM attribution is train-only")
        parent_id, smiles = _parent_id(parent), _smiles(parent)
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("Invalid train parent")
        if any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()):
            raise ValueError("Explicit hydrogen-node parent requires an audited atom mapping")
        graph = self._graph(smiles, parent_id, "train")
        if graph.num_nodes != molecule.GetNumAtoms():
            raise ValueError("Featurizer/CM atom count mismatch")
        # Featurizer parses the original input and preserves its order. Its
        # canonical_smiles metadata must NOT be reparsed with old atom indices.
        before = self.oracle.predict_records([graph])[0]
        base = {"schema_version": ATTRIBUTION_SCHEMA, "parent_id": parent_id,
                "split": "train", "input_smiles": smiles, "prediction": before,
                "oracle_weight_sha256": self.oracle.checkpoint_id,
                "temperature_sha256": self.binding.get("temperature_sha256"),
                "source_score": "calibrated_probability_source_class",
                "node_layer": self.node_layer_name, "upstream_commit": UPSTREAM_COMMIT,
                "gradient_formula": "mean_channels_per_node_then_activation_sum_no_relu_minmax",
                "mask_rounding": "upstream_max_1_floor_n_div_5",
                "atom_mapping": list(range(graph.num_nodes))}
        if before["predicted_label"] != self.oracle.source_label:
            return {**base, "status": "BEFORE_NOT_SOURCE", "generation_allowed": False}
        saved: list[Any] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> Any:
            if not torch.is_tensor(output) or output.ndim != 2:
                raise ValueError("Grad-CAM hook did not capture node activations")
            # Frozen integer embeddings do not require gradients. Introduce an
            # activation leaf, never enable parameter training or input floats.
            activation = output if output.requires_grad else output.detach().requires_grad_(True)
            saved.append(activation)
            return activation

        started = time.monotonic()
        with _unchanged_model(self.oracle.model):
            handle = self.node_layer.register_forward_hook(capture)
            try:
                with torch.enable_grad():
                    batch = next(iter(self.oracle._batches([graph], 1))).to(self.oracle.device)
                    logits = self.oracle.model(batch)
                    if len(saved) != 1 or saved[0].shape[0] != graph.num_nodes:
                        raise ValueError("Node hook count/atom mapping changed")
                    reference = torch.tensor(before["logits"], dtype=logits.dtype, device=logits.device)
                    if not torch.allclose(logits[0], reference, atol=self.forward_atol,
                                          rtol=self.forward_rtol):
                        raise ValueError("Grad-CAM forward differs from original GINE")
                    if int(logits[0].argmax()) != before["predicted_label"]:
                        raise ValueError("Grad-CAM forward argmax changed")
                    # Upstream classification.predict is sigmoid probability;
                    # here use the same frozen oracle's single temperature.
                    probability = torch.softmax(logits / self.oracle.temperature, dim=1)[0]
                    gradient = torch.autograd.grad(probability[self.oracle.source_label], saved[0])[0]
                    importance = upstream_cam(saved[0], gradient)
                    scores = importance.detach().cpu().tolist()
                    raw_gradient = gradient.detach().cpu()
                    gradient_sha = hashlib.sha256(raw_gradient.contiguous().numpy().tobytes()).hexdigest()
                    activation_sha = hashlib.sha256(saved[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            finally:
                handle.remove()
        selected = upstream_top_atoms(scores)
        request = make_parent_request(parent_id=parent_id, mol=molecule,
                                      selected_atom_indices=selected, split="train")
        generation_allowed = len(request["effective_atom_indices"]) < graph.num_nodes
        return {**base, "status": "ATTRIBUTION_COMPLETE" if generation_allowed else "NO_REPLACEABLE_CONTEXT",
                "generation_allowed": generation_allowed, "generation_request": request,
                "importances": scores, "activation_sha256": activation_sha,
                "gradient_sha256": gradient_sha,
                "gradient_nonzero_count": int(torch.count_nonzero(raw_gradient)),
                "weights_bn_rng_unchanged": True, "forward_argmax_exact": True,
                "seconds": time.monotonic() - started}

    def filter_generated(self, parent: Mapping[str, Any], generated: Mapping[str, Any]) -> dict[str, Any]:
        if parent.get("split") != "train":
            raise ValueError("Generation filtering must use train parents")
        parent_id = _parent_id(parent)
        if str(generated.get("parent_id")) != parent_id:
            raise ValueError("Generation/parent identity mismatch")
        if generated.get("status") not in {"GENERATED", "GENERATION_COMPLETE", "NO_NATIVE_REPLACEMENT", "NO_REPLACEABLE_CONTEXT", "TIMEOUT_BUDGETED"}:
            raise ValueError(f"Generation stage is not scientific terminal: {generated.get('status')}")
        if generated.get("status") == "TIMEOUT_BUDGETED" and generated.get("retained_raw"):
            raise ValueError("Budgeted timeout cannot adopt partial candidates")
        if generated.get("status") in {"NO_NATIVE_REPLACEMENT", "NO_REPLACEABLE_CONTEXT"} and generated.get("retained_raw"):
            raise ValueError("Native empty/context terminal cannot carry raw candidates")
        raw = list(generated.get("retained_raw", []))
        if len(raw) > 128:
            raise ValueError("Raw-output truncation must precede oracle filtering")
        before = self.predict_rows([parent], split="train")[0]
        if before["predicted_label"] != 1 and generated.get("retained_raw"):
            raise ValueError("Generation was performed for a non-source parent")
        original = graph_identity(_smiles(parent), self.featurizer, require_connected=False)
        candidates, rejected = [], []
        for position, row in enumerate(raw):
            try:
                identity = graph_identity(_smiles(row), self.featurizer)
            except (ValueError, RuntimeError) as exc:
                rejected.append({"raw_id": row.get("raw_id"), "reason": str(exc)})
                continue
            if identity["candidate_id"] == original["candidate_id"]:
                rejected.append({"raw_id": row.get("raw_id"), "reason": "UNCHANGED_PARENT"})
                continue
            candidates.append({**identity, "smiles": identity["canonical_smiles"],
                               "origins": [{"parent_id": parent_id, "raw_id": row.get("raw_id"),
                                            "retained_raw_index": position}]})
        unique: dict[str, dict[str, Any]] = {}
        for row in candidates:
            if row["candidate_id"] in unique:
                unique[row["candidate_id"]]["origins"].extend(row["origins"])
            else:
                unique[row["candidate_id"]] = row
        evaluated = self.predict_rows(list(unique.values()), split="train_generated")
        accepted: dict[str, dict[str, Any]] = {}
        for row in evaluated:
            if row["predicted_label"] != 0:
                rejected.append({"candidate_id": row["candidate_id"], "reason": "NOT_DESTINATION",
                                 "prediction": row["predicted_label"], "origins": row["origins"]})
                continue
            key = row["candidate_id"]
            if key in accepted:
                accepted[key]["origins"].extend(row["origins"])
            else:
                accepted[key] = row
        return {"parent_id": parent_id, "status": "FILTER_COMPLETE", "source_prediction": before,
                "raw_count": len(raw), "chemically_valid_nonself_count": len(candidates),
                "strict_flip_count": sum(len(row["origins"]) for row in accepted.values()),
                "unique_target_count": len(accepted), "accepted": list(accepted.values()),
                "rejected": rejected, "test_loaded": False}


def encoding_digest(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "encoding_sha256"}
    return _hash(body)


class FrozenCMWNode:
    """Original MolCLR extraction and exact-EMD function, no new OT solver/DB."""

    def __init__(self, embedder: Any, *, binding: Mapping[str, Any]) -> None:
        self.embedder = embedder
        self.binding = dict(binding)
        if (self.binding.get("feature_cost") != "cosine" or
                self.binding.get("node_mass") != "uniform" or
                float(self.binding.get("size_penalty_beta", -1)) != 0.0):
            raise ValueError("Original BACE WNode numerical contract changed")
        if not self.binding.get("numerical_contract_sha256"):
            raise ValueError("Original WNode numerical-contract binding is required")

    @classmethod
    def from_resolved(cls, payload: Mapping[str, Any] | str | Path) -> "FrozenCMWNode":
        from src.eval.molclr_node_embeddings import MolCLRNodeEmbedder

        spec = _record(payload)
        spec = dict(spec.get("resolved_wnode", spec))
        _pin(Path(spec["molclr_ckpt"]), spec.get("molclr_checkpoint_sha256"), "MolCLR checkpoint")
        embedder = MolCLRNodeEmbedder(molclr_root=spec["molclr_root"], molclr_ckpt=spec["molclr_ckpt"],
                                     node_emb_cache_dir=spec["node_emb_cache_dir"],
                                     encoder_type=spec.get("encoder_type", "gin"), device=spec.get("device", "cpu"))
        return cls(embedder, binding=spec)

    def encode_rows(self, rows: Sequence[Mapping[str, Any]], *, featurizer: MolecularGraphFeaturizer) -> list[dict[str, Any]]:
        from src.eval.molclr_node_embeddings import NODE_EXTRACTION_VERSION, atom_numbers_for_smiles
        import torch
        import rdkit

        encoded: dict[str, dict[str, Any]] = {}
        for row in rows:
            # The frozen base cohort may contain multiple components. Its
            # original oracle input must not inherit the generated-prototype
            # connectedness filter; generated rows passed that filter already.
            identity = graph_identity(_smiles(row), featurizer, require_connected=False)
            graph_id = identity["candidate_id"]
            if graph_id in encoded:
                continue
            smiles = identity["canonical_smiles"]
            started = time.monotonic()
            # Same audited extraction, but results are returned to the stage's
            # compact shard; no per-candidate persistent SQLite/cache writer.
            H = self.embedder._compute_node_embeddings(smiles)
            atoms = atom_numbers_for_smiles(smiles)
            if H.ndim != 2 or H.shape[0] != len(atoms) or not np.isfinite(H).all():
                raise ValueError("Original MolCLR node output is invalid")
            result = {"schema_version": ENCODING_SCHEMA, **identity,
                      "H": np.asarray(H, dtype=np.float32).tolist(), "atom_numbers": atoms.tolist(),
                      "embedding_dtype": "float32", "node_extraction_version": NODE_EXTRACTION_VERSION,
                      "molclr_checkpoint_sha256": self.binding["molclr_checkpoint_sha256"],
                      "numerical_contract_sha256": self.binding["numerical_contract_sha256"],
                      "producer": {"device": str(self.embedder.loaded.device), "torch": torch.__version__,
                                   "numpy": np.__version__, "rdkit": rdkit.__version__,
                                   "architecture": self.embedder.architecture_identity}}
            result["encoding_sha256"] = encoding_digest(result)
            encoded[graph_id] = result
            # Timing is outside content identity and cannot change a graph key.
            result["timing_seconds"] = time.monotonic() - started
        return list(encoded.values())


def raw_distance_record(left: Mapping[str, Any], right: Mapping[str, Any],
                        *, numerical_contract: Mapping[str, Any], emd2_fn: Any = None) -> dict[str, Any]:
    """Compute one missing raw pair with the existing exact WNode function.

    Any oracle eligibility/mask belongs to a separate reduction, never this
    cache key. The function neither reads calibration/test nor chooses rules.
    """
    from src.eval.node_wasserstein_distance import compute_node_wasserstein_distance

    for side in (left, right):
        digest_body = {key: value for key, value in side.items() if key != "timing_seconds"}
        if encoding_digest(digest_body) != side.get("encoding_sha256"):
            raise ValueError("Node-encoding content identity mismatch")
        if side.get("schema_version") != ENCODING_SCHEMA:
            raise ValueError("Node encoding schema mismatch")
    for field in ("molclr_checkpoint_sha256", "numerical_contract_sha256", "node_extraction_version", "producer"):
        if left.get(field) != right.get(field):
            raise ValueError(f"Raw-distance encoding producer conflict: {field}")
    if numerical_contract.get("numerical_contract_sha256") != left["numerical_contract_sha256"]:
        raise ValueError("Raw-distance numerical contract mismatch")
    producer = {"numpy": np.__version__, "dtype": "float64_cost_uniform_mass",
                "implementation_sha256": hashlib.sha256(inspect.getsource(compute_node_wasserstein_distance).encode()).hexdigest(),
                "POT": importlib.metadata.version("POT") if emd2_fn is None else "UNIT_TEST_INJECTED_SOLVER"}
    science_contract = {key: numerical_contract[key] for key in
                        ("numerical_contract_sha256", "feature_cost", "node_mass", "size_penalty_beta")}
    key_body = {"schema_version": RAW_DISTANCE_SCHEMA,
                "encoding_ids": sorted([left["encoding_sha256"], right["encoding_sha256"]]),
                "numerical_contract": science_contract, "solver": "exact_emd2", "ot_producer": producer}
    value, metadata = compute_node_wasserstein_distance(
        np.asarray(left["H"], dtype=np.float32), np.asarray(right["H"], dtype=np.float32),
        feature_cost=numerical_contract["feature_cost"], node_mass=numerical_contract["node_mass"],
        size_penalty_beta=float(numerical_contract["size_penalty_beta"]), emd2_fn=emd2_fn)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Exact WNode produced invalid distance")
    return {**key_body, "raw_pair_key": _hash(key_body), "distance": value,
            "metadata": metadata, "production_solver_used": emd2_fn is None,
            "parent_graph_id": left["candidate_id"],
            "prototype_graph_id": right["candidate_id"], "distance_is_uncapped": True}


def full_graph_pair(parent_prediction: Mapping[str, Any], prototype_prediction: Mapping[str, Any],
                    *, raw_distance: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strict-flip full-prototype reduction; None means semantic infinity only."""
    if parent_prediction.get("oracle_weight_sha256") != prototype_prediction.get("oracle_weight_sha256"):
        raise ValueError("Parent/prototype oracle binding conflict")
    if parent_prediction.get("temperature") != prototype_prediction.get("temperature"):
        raise ValueError("Parent/prototype temperature binding conflict")
    before, after = int(parent_prediction["predicted_label"]), int(prototype_prediction["predicted_label"])
    reason = "BEFORE_NOT_SOURCE" if before != 1 else "NOT_DESTINATION" if after != 0 else None
    if reason is None and raw_distance is None:
        raise ValueError("Strict-flip pair is missing a computed raw distance; not infinity")
    if raw_distance is not None:
        if (not parent_prediction.get("full_graph_id") or not prototype_prediction.get("full_graph_id") or
                raw_distance.get("parent_graph_id") != parent_prediction["full_graph_id"] or
                raw_distance.get("prototype_graph_id") != prototype_prediction["full_graph_id"] or
                raw_distance.get("distance_is_uncapped") is not True):
            raise ValueError("Raw distance does not bind these complete parent/prototype graphs")
    distance = None if reason else float(raw_distance["distance"])
    if distance is not None and (not math.isfinite(distance) or distance < 0):
        raise ValueError("Nonfinite raw distance is an engineering error")
    return {"parent_id": _parent_id(parent_prediction), "candidate_id": prototype_prediction["candidate_id"],
            "pred_before": before, "pred_after": after, "strict_flip": reason is None,
            "distance": distance, "failure_reason": reason, "kept_in_base_denominator": True,
            "raw_pair_key": None if raw_distance is None else raw_distance["raw_pair_key"]}
