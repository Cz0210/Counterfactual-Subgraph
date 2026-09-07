"""Bind unchanged current BACE raw-distance inputs to a portable old index.

The old index is evidence, not the authority for the current graph schema,
MolCLR weights or source. Small current files are checked; a prior immutable
reference receipt supplies the weights digest, avoiding repeated model hashing.
"""
from pathlib import Path

from src.eval.bace_frozen_gnn_contracts import read_json, sha256_file, stable_sha256


def bound_json(item):
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("RAW_REUSE_DESCRIPTOR_CHANGED:" + str(path))
    return read_json(path)


def current_raw_contract(contract, portable):
    """Derive values from actual current paths, never copy an index contract."""
    ref_path = Path(contract["paths"]["reference"])
    if sha256_file(ref_path) != contract["reference_sha256"]:
        raise ValueError("FROZEN_REFERENCE_RECEIPT_CHANGED")
    reference = read_json(ref_path)
    down = reference["frozen_downstream"]
    paths = contract["paths"]
    if (down["wnode_config"] != contract["wnode_config"]
        or down["molclr_sha"] != contract["molclr_sha256"]
        or Path(down["molclr_root"]).resolve() != Path(paths["molclr_checkpoint"]).resolve()
        or Path(down["gine_checkpoint"]).parent.resolve() != Path(paths["oracle"]).resolve()):
        raise ValueError("CURRENT_RAW_INPUTS_NOT_REFERENCE_BOUND")
    schema = Path(paths["oracle"]) / "feature_schema.json"
    weights = Path(paths["molclr_checkpoint"])
    prefix = portable["molclr_source_root"] + "/"
    source = {}
    for rel in portable["files"]:
        if not rel.startswith(prefix):
            continue
        suffix = Path(rel.removeprefix(prefix))
        if suffix.is_absolute() or ".." in suffix.parts:
            raise ValueError("UNSAFE_MOLCLR_SOURCE_SUFFIX")
        actual = Path(paths["molclr_source"]) / suffix
        source[rel] = {"sha256": sha256_file(actual), "size": actual.stat().st_size}
    if prefix + "models/ginet_molclr.py" not in source:
        raise ValueError("CURRENT_MOLCLR_IMPLEMENTATION_NOT_BOUND")
    actual = {"wnode": dict(contract["wnode_config"]),
        "feature_schema": {"sha256": sha256_file(schema), "size": schema.stat().st_size},
        "molclr_checkpoint": {"sha256": down["molclr_sha"], "size": weights.stat().st_size},
        "molclr_source": source}
    accepted = {"wnode": portable["wnode_config"],
        "feature_schema": portable["files"][portable["feature_schema_path"]],
        "molclr_checkpoint": portable["files"][portable["molclr_checkpoint_path"]],
        "molclr_source": {r: value for r, value in portable["files"].items() if r.startswith(prefix)}}
    if actual != accepted:
        raise ValueError("CURRENT_RAW_CONTRACT_DIFFERS_FROM_ACCEPTED_SOURCE")
    return actual


def wrap_raw_distance(delegate, *, contract, descriptor, split, repo, final_freeze=None):
    """A caller supplies the real new freeze for any old-test migration use."""
    if split not in {"calibration", "test"}:
        raise ValueError("OLD_RAW_INDEX_MUST_NOT_GUIDE_TRAIN_SEARCH")
    portable = bound_json(descriptor["portable_manifest"])
    source_spec = bound_json(descriptor["source_spec"])["raw_distance_source"]
    index = bound_json(descriptor["index"])
    if index.get("split") != split or index.get("source_spec") != source_spec:
        raise ValueError("RAW_INDEX_SPLIT_OR_SOURCE_CHANGED")
    if split == "test":
        if not final_freeze:
            raise ValueError("RAW_TEST_REUSE_REQUIRES_FINAL_FREEZE")
        from src.eval.bace_reach_v2 import unseal
        frozen = unseal(Path(final_freeze))
        if (frozen.get("state") != "REACH_V2_FINAL_CONFIGURATION_FROZEN"
            or frozen.get("test_opened") is not False
            or index.get("new_test_freeze_sha256") != sha256_file(final_freeze)):
            raise ValueError("RAW_TEST_INDEX_NOT_BOUND_TO_ACTUAL_NEW_FREEZE")
    actual = current_raw_contract(contract, portable)
    from src.ablations.gnn.reach_raw_distance_reuse import VerifiedRawGraphDistance
    result = VerifiedRawGraphDistance(delegate, index=index, current_raw_contract=actual, repo=Path(repo))
    result.current_input_binding = {"current_raw_contract_sha256": stable_sha256(actual),
        "frozen_reference_sha256": contract["reference_sha256"],
        "weights_digest_source": "EXISTING_IMMUTABLE_REFERENCE_RECEIPT",
        "weights_rehashed": False, "small_current_sources_verified": True,
        "source_descriptor": descriptor, "source_flip_masks_reused": False}
    return result
