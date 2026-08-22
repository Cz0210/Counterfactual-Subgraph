"""Auditable smoke gate for the BACE GlobalGCE frozen-GINE bridge."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.baselines.bace_gnn_baseline_contracts import (
    DATASET,
    NUM_CLASSES,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.baselines.globalgce_bace_native_rules import build_parent_native_tensors
from src.baselines.globalgce_frozen_gine_bridge import (
    BRIDGE_SCHEMA_VERSION,
    FrozenGINEDifferentiableBridge,
)
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_marker,
    fresh_output_dir,
    sha256_file,
    utc_now,
)
from src.oracles.oracle_factory import build_oracle


BRIDGE_SMOKE_VERSION = "bace_globalgce_frozen_gine_bridge_smoke_v1"


def _graph(featurizer: MolecularGraphFeaturizer, smiles: str) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=SOURCE_LABEL,
        molecule_id="globalgce-bridge-smoke",
        smiles=features.canonical_smiles,
        split="preflight_canary",
        graph_sha256=features.graph_sha256,
    )


def run_frozen_gine_bridge_smoke(
    *,
    gnn_checkpoint: str | Path,
    parent_smiles: str,
    atom_symbols: tuple[str, ...],
    bond_names: tuple[str, ...],
    output_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Prove frozen weights, nonzero transformation gradient, and hard parity."""

    torch = __import__("torch")
    checkpoint, card, schema = validate_bace_frozen_gine(gnn_checkpoint)
    before_checkpoint_hash = sha256_file(checkpoint / "model.pt")
    bridge = FrozenGINEDifferentiableBridge.from_checkpoint(
        checkpoint,
        atom_symbols=atom_symbols,
        bond_names=bond_names,
        device=device,
    )
    native = build_parent_native_tensors(
        parent_smiles,
        atom_symbols=atom_symbols,
        bond_names=bond_names,
    )
    feature = native.feature.to(device).unsqueeze(0).clone().requires_grad_(True)
    adjacency = native.adjacency.to(device).unsqueeze(0).clone().requires_grad_(True)
    edge_attr = native.edge_attr.to(device).unsqueeze(0).clone().requires_grad_(True)
    classifier_before = {
        name: value.detach().cpu().clone()
        for name, value in bridge.model.state_dict().items()
    }
    bridge_result = bridge(feature, adjacency, edge_attr)

    oracle = build_oracle(
        dataset=DATASET,
        backend=ORACLE_BACKEND,
        checkpoint=checkpoint,
        device=device,
        batch_size=8,
        num_classes=NUM_CLASSES,
        source_label=SOURCE_LABEL,
    )
    graph = _graph(MolecularGraphFeaturizer(schema), parent_smiles)
    ordinary = oracle.predict_records([graph], batch_size=8)[0]
    ordinary_probabilities = tuple(float(value) for value in ordinary["probabilities"])
    bridge_probabilities = tuple(
        float(value)
        for value in bridge_result["y_pred"].detach().exp().cpu()[0].tolist()
    )
    parity_max_abs_error = max(
        abs(left - right)
        for left, right in zip(
            ordinary_probabilities, bridge_probabilities, strict=True
        )
    )
    parity_pass = parity_max_abs_error <= 2e-6

    target = 1 - int(ordinary["predicted_label"])
    loss = torch.nn.functional.nll_loss(
        bridge_result["y_pred"],
        torch.tensor([target], dtype=torch.long, device=bridge_result["y_pred"].device),
    )
    loss.backward()
    gradients = {
        "features": float(feature.grad.detach().abs().sum()) if feature.grad is not None else 0.0,
        "adjacency": (
            float(adjacency.grad.detach().abs().sum())
            if adjacency.grad is not None
            else 0.0
        ),
        "edge_attributes": (
            float(edge_attr.grad.detach().abs().sum())
            if edge_attr.grad is not None
            else 0.0
        ),
    }
    transformation_gradient_nonzero = sum(gradients.values()) > 0.0
    classifier_gradient_abs_sum = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in bridge.model.parameters()
        if parameter.grad is not None
    )
    classifier_parameters_unchanged = all(
        torch.equal(classifier_before[name], value.detach().cpu())
        for name, value in bridge.model.state_dict().items()
    )
    after_checkpoint_hash = sha256_file(checkpoint / "model.pt")
    checkpoint_unchanged = before_checkpoint_hash == after_checkpoint_hash
    finite = math.isfinite(float(loss.detach().cpu())) and all(
        math.isfinite(value) for value in gradients.values()
    )
    failures: list[str] = []
    for passed, name in (
        (parity_pass, "bridge_prediction_matches_ordinary_frozen_oracle"),
        (transformation_gradient_nonzero, "transformation_gradient_nonzero"),
        (classifier_gradient_abs_sum == 0.0, "classifier_gradient_zero"),
        (classifier_parameters_unchanged, "classifier_parameters_unchanged"),
        (checkpoint_unchanged, "checkpoint_hash_unchanged"),
        (finite, "finite_loss_and_gradients"),
        (
            int(bridge.last_audit.get("hard_graph_sanitized_count") or 0) == 1,
            "hard_graph_sanitized",
        ),
    ):
        if not passed:
            failures.append(name)
    if failures:
        raise ValueError("GlobalGCE frozen-GINE bridge smoke failed: " + ", ".join(failures))

    output = fresh_output_dir(output_dir)
    provenance = oracle_provenance(card, checkpoint)
    manifest = {
        "schema_version": BRIDGE_SMOKE_VERSION,
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        "status": "PASS",
        "dataset": DATASET,
        "method": "GlobalGCE",
        "stage": "FROZEN_GINE_DIFFERENTIABLE_BRIDGE_SMOKE",
        "action_kind": "lhs_rhs_graph_transformation_rule",
        "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
        "classifier_family": "gine",
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "classifier_requires_grad": False,
        "classifier_gradient_abs_sum": classifier_gradient_abs_sum,
        "classifier_parameters_unchanged": classifier_parameters_unchanged,
        "transformation_gradient_nonzero": transformation_gradient_nonzero,
        "transformation_gradient_abs_sums": gradients,
        "bridge_prediction_matches_ordinary_frozen_oracle": parity_pass,
        "prediction_parity_max_abs_error": parity_max_abs_error,
        "checkpoint_hash_before": before_checkpoint_hash,
        "checkpoint_hash_after": after_checkpoint_hash,
        "checkpoint_hash_unchanged": checkpoint_unchanged,
        "ordinary_prediction": dict(ordinary),
        "bridge_probabilities": list(bridge_probabilities),
        "hard_product_revalidated_by_same_frozen_gine": True,
        "hard_product_strict_flip": False,
        "hard_product_note": (
            "The bridge smoke is an identity hard graph. Full GlobalGCE output "
            "still requires native LHS->RHS application and strict-flip verification."
        ),
        "oracle_provenance": provenance,
        "test_loaded": False,
        "calibration_loaded": False,
        "created_at": utc_now(),
        "failures": [],
    }
    atomic_json(output / "bridge_gradient_audit.json", manifest)
    atomic_json(output / "oracle_provenance.json", provenance)
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "state.json", manifest)
    atomic_marker(output / "PASS", "PASS")
    atomic_marker(
        output / "BRIDGE_PASS",
        "[BACE_GLOBALGCE_BRIDGE_PASS]",
    )
    return manifest


__all__ = ["BRIDGE_SMOKE_VERSION", "run_frozen_gine_bridge_smoke"]
