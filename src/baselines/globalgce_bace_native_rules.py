"""Native GlobalGCE LHS->RHS rules for the frozen BACE GINE route.

The official GlobalGCE implementation applies a decoded rule locally.  It
does *not* replace a parent with an arbitrary full graph and it does not delete
an explanation fragment.  For every exact labelled LHS match it builds the
official mask order, overwrites only the mask square with the reconstructed
RHS tensors, and leaves all edges from matched nodes to nodes outside the mask
untouched.  New RHS nodes are appended after the parent nodes.

This module owns that action boundary.  The official checkout remains
read-only and is supplied explicitly at runtime; its pinned implementation is
used by :func:`run_official_tensor_parity` as a production preflight oracle.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from src.eval.bace_frozen_gnn_contracts import stable_sha256


OFFICIAL_GLOBALGCE_COMMIT = "157e65c2850bc787f229a1ee8c60564906b933f2"
OFFICIAL_SOURCE_SHA256 = {
    "src/models/GlobalGCE.py": "d744e26bd7a7d1f60777285d6bc1c9ef3d6d3641ef655cf748c5e3c914c6f33a",
    "src/models/fsg.py": "504cf5ba9ee1a6be32c6b201cf602b0eca9a50e52971cb9d76b62f0e896902bc",
    "src/models/models_utils.py": (
        "37d29df6ecf3ce7de5c560c57a25d950fc593c264167d071c860e1080c65fe75"
    ),
    "src/utils.py": "95a0d9f2953bdafa3cf1cb891958e45b4faaea876cf2ae86c5fb4f069745748a",
    "src/data/dataset.py": "e2e1e54197cf01c31f33f655d2d1dafd35ec7e99533ebb4ab6ccbbb2ff889aa0",
}
OFFICIAL_RUNTIME_FILES = frozenset(
    {
        "src/main.py",
        "src/models/GTGNN.py",
        "src/models/GlobalGCE.py",
        "src/models/models_utils.py",
        "src/models/fsg.py",
        "src/models/gSpan/gSpan.py",
        "src/data/data_preprocess.py",
        "src/data/dataset.py",
    }
)
ACTION_ENGINE_VERSION = "globalgce_native_attachment_rule_v2"
OFFICIAL_TENSOR_PARITY_VERSION = "globalgce_official_tensor_parity_v1"
RULE_SELECTOR_CHEMISTRY_VERSION = "globalgce_native_rule_selector_chemistry_v1"


class GlobalGCENativeRuleError(ValueError):
    """A rule, match, or application failed a native-semantics gate."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency.
        raise RuntimeError("Native GlobalGCE rule application requires PyTorch.") from exc
    return torch


def _rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - AutoDL dependency.
        raise RuntimeError("Native GlobalGCE rule application requires RDKit.") from exc
    return Chem


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_git_bytes(root: Path, *arguments: str) -> bytes:
    """Run Git without ambient configuration or replace-object authority."""

    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(root),
            *arguments,
        ],
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CEILING_DIRECTORIES": str(root.parent),
        },
    )
    return completed.stdout


def _official_runtime_source_authority(root: Path) -> dict[str, dict[str, Any]]:
    """Bind every tracked Python source byte actually reachable by imports."""

    hidden = _official_git_bytes(root, "ls-files", "-v", "-z")
    for raw in hidden.split(b"\0"):
        if not raw:
            continue
        record = raw.decode("utf-8")
        if len(record) < 3 or record[1] != " ":
            raise GlobalGCENativeRuleError("GlobalGCE Git index inventory is malformed")
        if record[0] == "S" or record[0].islower():
            raise GlobalGCENativeRuleError(
                "GlobalGCE checkout has skip-worktree/assume-unchanged entries"
            )
    if _official_git_bytes(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise GlobalGCENativeRuleError("GlobalGCE official checkout is not clean")
    ignored = tuple(
        raw.decode("utf-8")
        for raw in _official_git_bytes(
            root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            "src",
        ).split(b"\0")
        if raw
    )
    if any(
        "__pycache__" in Path(relative).parts
        or Path(relative).suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".dylib"}
        for relative in ignored
    ):
        raise GlobalGCENativeRuleError(
            "GlobalGCE source closure contains ignored runtime code"
        )

    tree_records = _official_git_bytes(
        root, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", "src"
    )
    tracked: dict[str, str] = {}
    for raw in tree_records.split(b"\0"):
        if not raw:
            continue
        metadata, separator, encoded_path = raw.partition(b"\t")
        fields = metadata.decode("ascii").split()
        relative = encoded_path.decode("utf-8") if separator else ""
        if len(fields) != 3 or fields[1] != "blob" or not relative:
            raise GlobalGCENativeRuleError("GlobalGCE HEAD source tree is malformed")
        if Path(relative).suffix == ".py":
            if fields[0] != "100644":
                raise GlobalGCENativeRuleError(
                    f"GlobalGCE Python source has unsafe Git mode: {relative}"
                )
            tracked[relative] = fields[2]
    if not OFFICIAL_RUNTIME_FILES.issubset(tracked):
        missing = sorted(OFFICIAL_RUNTIME_FILES - set(tracked))
        raise GlobalGCENativeRuleError(
            f"GlobalGCE runtime source closure is incomplete: {missing}"
        )

    authority: dict[str, dict[str, Any]] = {}
    for relative, blob in sorted(tracked.items()):
        path = root / relative
        observed = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size <= 0
        ):
            raise GlobalGCENativeRuleError(
                f"GlobalGCE runtime source is not one regular file: {relative}"
            )
        committed = _official_git_bytes(root, "cat-file", "blob", blob)
        digest = hashlib.sha256(committed).hexdigest()
        if len(committed) != observed.st_size or _sha256_file(path) != digest:
            raise GlobalGCENativeRuleError(
                f"GlobalGCE runtime source differs from pinned HEAD: {relative}"
            )
        source_relative = Path(relative).relative_to("src").as_posix()
        authority[source_relative] = {
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "bytes": int(observed.st_size),
            "sha256": digest,
        }
    return authority


def validate_official_globalgce_root(
    official_root: str | Path,
    *,
    expected_commit: str = OFFICIAL_GLOBALGCE_COMMIT,
) -> dict[str, Any]:
    """Fail closed unless the explicit checkout is the audited upstream tree."""

    root = Path(official_root).expanduser().resolve(strict=True)
    source = root / "src"
    if not source.is_dir():
        raise GlobalGCENativeRuleError(
            f"GlobalGCE official root must contain src/: {root}"
        )
    try:
        commit = _official_git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise GlobalGCENativeRuleError(
            f"Cannot resolve GlobalGCE official commit at {root}"
        ) from exc
    if commit != str(expected_commit):
        raise GlobalGCENativeRuleError(
            "GlobalGCE official commit mismatch: "
            f"actual={commit}, expected={expected_commit}"
        )
    identities: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in OFFICIAL_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise GlobalGCENativeRuleError(
                f"Pinned GlobalGCE source is missing or empty: {path}"
            )
        actual = _sha256_file(path)
        if actual != expected_sha:
            raise GlobalGCENativeRuleError(
                "Pinned GlobalGCE source hash mismatch: "
                f"file={relative}, actual={actual}, expected={expected_sha}"
            )
        identities[relative] = {
            "path": str(path),
            "sha256": actual,
            "size": path.stat().st_size,
        }
    runtime_source_authority = _official_runtime_source_authority(root)
    return {
        "official_root": str(root),
        "official_source_root": str(source),
        "official_commit": commit,
        "source_files": identities,
        "runtime_source_authority": runtime_source_authority,
        "runtime_source_inventory_sha256": stable_sha256(runtime_source_authority),
        "clean_checkout": True,
    }


def _as_float_tensor(value: Any, *, name: str) -> Any:
    torch = _torch()
    tensor = value.detach().clone() if hasattr(value, "detach") else torch.tensor(value)
    tensor = tensor.to(dtype=torch.float32, device="cpu")
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise GlobalGCENativeRuleError(f"{name} is empty or non-finite")
    return tensor


def _edge_position(left: int, right: int) -> int:
    high, low = max(int(left), int(right)), min(int(left), int(right))
    if high == low:
        raise GlobalGCENativeRuleError("Self loops have no native edge-vector slot")
    return (high - 1) * high // 2 + low


def _hard_label(row: Any) -> int:
    return int(row.argmax(-1).item())


def _validate_decoded_rows(value: Any, *, name: str) -> None:
    """Reject labels whose official argmax hard decode is not unique."""

    torch = _torch()
    if value.ndim != 2 or int(value.shape[1]) < 2:
        raise GlobalGCENativeRuleError(f"{name} must be a rank-2 label matrix")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise GlobalGCENativeRuleError(f"{name} contains values outside [0,1]")
    top = torch.topk(value, k=2, dim=-1).values
    if bool((torch.abs(top[:, 0] - top[:, 1]) <= 1e-8).any().item()):
        raise GlobalGCENativeRuleError(f"{name} contains an ambiguous hard label")


def _validate_adjacency(value: Any, *, name: str) -> None:
    torch = _torch()
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise GlobalGCENativeRuleError(f"{name} contains values outside [0,1]")
    if bool((torch.abs(value.diagonal()) > 1e-8).any().item()):
        raise GlobalGCENativeRuleError(f"{name} contains unsupported self loops")
    if bool((torch.abs(value - 0.5) <= 1e-8).any().item()):
        raise GlobalGCENativeRuleError(f"{name} contains ambiguous 0.5 edge values")


def _active_nodes(feature: Any, adjacency: Any) -> tuple[int, ...]:
    del adjacency
    return tuple(
        index
        for index in range(int(feature.shape[0]))
        if _hard_label(feature[index]) > 0
    )


def _labeled_graph(feature: Any, adjacency: Any, edge_attr: Any) -> nx.Graph:
    nodes = _active_nodes(feature, adjacency)
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(int(node), label=_hard_label(feature[node]))
    for position, left in enumerate(nodes):
        for right in nodes[position + 1 :]:
            forward = float(adjacency[left, right].item()) > 0.5
            reverse = float(adjacency[right, left].item()) > 0.5
            if forward != reverse:
                raise GlobalGCENativeRuleError(
                    f"Native adjacency is asymmetric at ({left},{right})"
                )
            if not forward:
                continue
            label = (
                _hard_label(edge_attr[_edge_position(left, right)])
                if edge_attr is not None
                else 1
            )
            if label <= 0:
                raise GlobalGCENativeRuleError(
                    "Native adjacency edge has explicit no-edge bond label"
                )
            graph.add_edge(int(left), int(right), label=label)
    return graph


@dataclass(frozen=True, slots=True)
class GlobalGCENativeRule:
    """One immutable official LHS and reconstructed RHS tensor rule."""

    rule_id: str
    native_rule_index: int
    lhs_feature: Any
    lhs_adjacency: Any
    lhs_edge_attr: Any
    rhs_feature: Any
    rhs_adjacency: Any
    rhs_edge_attr: Any
    atom_symbols: tuple[str, ...]
    bond_names: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GlobalGCENativeRule":
        source = payload.get("rule") if isinstance(payload.get("rule"), Mapping) else payload
        rule = cls(
            rule_id=str(payload.get("candidate_id") or source.get("rule_id") or ""),
            native_rule_index=int(source.get("native_rule_index", -1)),
            lhs_feature=_as_float_tensor(source.get("lhs_feature"), name="lhs_feature"),
            lhs_adjacency=_as_float_tensor(
                source.get("lhs_adjacency"), name="lhs_adjacency"
            ),
            lhs_edge_attr=_as_float_tensor(
                source.get("lhs_edge_attr"), name="lhs_edge_attr"
            ),
            rhs_feature=_as_float_tensor(source.get("rhs_feature"), name="rhs_feature"),
            rhs_adjacency=_as_float_tensor(
                source.get("rhs_adjacency"), name="rhs_adjacency"
            ),
            rhs_edge_attr=_as_float_tensor(
                source.get("rhs_edge_attr"), name="rhs_edge_attr"
            ),
            atom_symbols=tuple(str(value) for value in source.get("atom_symbols", [])),
            bond_names=tuple(str(value) for value in source.get("bond_names", [])),
        )
        rule.validate()
        return rule

    def validate(self) -> None:
        if not self.rule_id or self.native_rule_index < 0:
            raise GlobalGCENativeRuleError("Rule identity is empty or invalid")
        maximum = int(self.lhs_feature.shape[0])
        if self.lhs_feature.ndim != 2 or self.rhs_feature.shape != self.lhs_feature.shape:
            raise GlobalGCENativeRuleError("LHS/RHS feature shapes differ")
        if tuple(self.lhs_adjacency.shape) != (maximum, maximum):
            raise GlobalGCENativeRuleError("LHS adjacency shape is invalid")
        if tuple(self.rhs_adjacency.shape) != (maximum, maximum):
            raise GlobalGCENativeRuleError("RHS adjacency shape is invalid")
        if not bool((self.lhs_adjacency == self.lhs_adjacency.T).all().item()):
            raise GlobalGCENativeRuleError("LHS adjacency is asymmetric")
        if not bool(
            _torch().allclose(
                self.rhs_adjacency,
                self.rhs_adjacency.T,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise GlobalGCENativeRuleError("RHS adjacency is asymmetric")
        _validate_adjacency(self.lhs_adjacency, name="lhs_adjacency")
        _validate_adjacency(self.rhs_adjacency, name="rhs_adjacency")
        edge_slots = maximum * (maximum - 1) // 2
        if self.lhs_edge_attr.ndim != 2 or int(self.lhs_edge_attr.shape[0]) != edge_slots:
            raise GlobalGCENativeRuleError("LHS edge-attribute shape is invalid")
        if self.rhs_edge_attr.shape != self.lhs_edge_attr.shape:
            raise GlobalGCENativeRuleError("LHS/RHS edge-attribute shapes differ")
        if int(self.lhs_feature.shape[1]) != len(self.atom_symbols) + 1:
            raise GlobalGCENativeRuleError("Rule atom vocabulary width is invalid")
        if int(self.lhs_edge_attr.shape[1]) != len(self.bond_names):
            raise GlobalGCENativeRuleError("Rule bond vocabulary width is invalid")
        if (
            not self.atom_symbols
            or any(not value.strip() for value in self.atom_symbols)
            or len(set(self.atom_symbols)) != len(self.atom_symbols)
        ):
            raise GlobalGCENativeRuleError("Rule atom vocabulary is empty or duplicated")
        normalized_bonds = tuple(value.strip().lower() for value in self.bond_names)
        if (
            not normalized_bonds
            or normalized_bonds[0] != "no_edge"
            or any(not value for value in normalized_bonds)
            or len(set(normalized_bonds)) != len(normalized_bonds)
        ):
            raise GlobalGCENativeRuleError(
                "Rule bond vocabulary must start with unique no_edge"
            )
        _validate_decoded_rows(self.lhs_feature, name="lhs_feature")
        _validate_decoded_rows(self.rhs_feature, name="rhs_feature")
        _validate_decoded_rows(self.lhs_edge_attr, name="lhs_edge_attr")
        _validate_decoded_rows(self.rhs_edge_attr, name="rhs_edge_attr")
        lhs = _labeled_graph(self.lhs_feature, self.lhs_adjacency, self.lhs_edge_attr)
        if lhs.number_of_nodes() < 1 or not nx.is_connected(lhs):
            raise GlobalGCENativeRuleError("Rule LHS is empty or disconnected")

    @property
    def maximum_nodes(self) -> int:
        return int(self.lhs_feature.shape[0])

    @property
    def lhs_nodes(self) -> tuple[int, ...]:
        return _active_nodes(self.lhs_feature, self.lhs_adjacency)

    def content_hash(self) -> str:
        payload = self.to_payload()
        payload.pop("rule_id", None)
        return stable_sha256(payload)

    def to_payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "native_rule_index": self.native_rule_index,
            "lhs_feature": self.lhs_feature.tolist(),
            "lhs_adjacency": self.lhs_adjacency.tolist(),
            "lhs_edge_attr": self.lhs_edge_attr.tolist(),
            "rhs_feature": self.rhs_feature.tolist(),
            "rhs_adjacency": self.rhs_adjacency.tolist(),
            "rhs_edge_attr": self.rhs_edge_attr.tolist(),
            "atom_symbols": list(self.atom_symbols),
            "bond_names": list(self.bond_names),
        }

    def selector_chemistry(self, *, n_bits: int = 2048) -> dict[str, Any]:
        """Build a deterministic redundancy representation of the native rule.

        GlobalGCE actions are LHS-to-RHS transformations, not fragments.  The
        shared selector therefore fingerprints aligned labelled node and edge
        changes instead of inventing a deletion/full-graph SMILES.  These bits
        are calibration-only redundancy evidence; native application remains
        the sole scientific action.
        """

        bit_count = int(n_bits)
        if bit_count < 128:
            raise GlobalGCENativeRuleError(
                "Native rule selector fingerprint requires at least 128 bits"
            )
        lhs_nodes = [_hard_label(row) for row in self.lhs_feature]
        rhs_nodes = [_hard_label(row) for row in self.rhs_feature]
        tokens: list[str] = []
        for index, (lhs_label, rhs_label) in enumerate(
            zip(lhs_nodes, rhs_nodes, strict=True)
        ):
            tokens.append(f"node:{index}:{lhs_label}>{rhs_label}")
        for left in range(self.maximum_nodes):
            for right in range(left + 1, self.maximum_nodes):
                position = _edge_position(left, right)
                lhs_present = int(float(self.lhs_adjacency[left, right]) > 0.5)
                rhs_present = int(float(self.rhs_adjacency[left, right]) > 0.5)
                lhs_bond = (
                    _hard_label(self.lhs_edge_attr[position]) if lhs_present else 0
                )
                rhs_bond = (
                    _hard_label(self.rhs_edge_attr[position]) if rhs_present else 0
                )
                if lhs_present or rhs_present or lhs_bond != rhs_bond:
                    tokens.append(
                        f"edge:{left}:{right}:{lhs_present}:{lhs_bond}>"
                        f"{rhs_present}:{rhs_bond}"
                    )
        active_lhs = sum(label > 0 for label in lhs_nodes)
        active_rhs = sum(label > 0 for label in rhs_nodes)
        tokens.extend(
            (
                f"lhs_active:{active_lhs}",
                f"rhs_active:{active_rhs}",
                f"node_delta:{active_rhs - active_lhs}",
            )
        )
        bits: set[int] = set()
        for token in sorted(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 8, 2):
                bits.add(
                    int.from_bytes(digest[offset : offset + 2], "big") % bit_count
                )
        if not bits:
            raise GlobalGCENativeRuleError("Native rule selector fingerprint is empty")
        return {
            "schema_version": RULE_SELECTOR_CHEMISTRY_VERSION,
            "role": "native_lhs_rhs_rule_redundancy_only",
            "fingerprint_kind": "hashed_aligned_label_transition_bits",
            "fingerprint_n_bits": bit_count,
            "fingerprint_bits": sorted(bits),
            "heavy_atom_count": max(active_lhs, active_rhs),
            "canonical_fragment_applicable": False,
            "canonical_fragment_reason": (
                "GlobalGCE action is an attachment-aware LHS-to-RHS rule"
            ),
        }


@dataclass(frozen=True, slots=True)
class NativeParentTensors:
    canonical_smiles: str
    feature: Any
    adjacency: Any
    edge_attr: Any
    atom_attributes: tuple[dict[str, Any], ...]


def build_parent_native_tensors(
    smiles: str,
    *,
    atom_symbols: Sequence[str],
    bond_names: Sequence[str] = ("no_edge", "single", "double", "triple"),
) -> NativeParentTensors:
    """Encode one molecule in the exact dense vocabulary used by the rule."""

    Chem = _rdkit()
    torch = _torch()
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        raise GlobalGCENativeRuleError(f"Cannot parse parent SMILES: {smiles!r}")
    molecule = Chem.Mol(molecule)
    try:
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except Exception as exc:
        raise GlobalGCENativeRuleError(
            f"Cannot sanitize/kekulize parent SMILES: {smiles!r}"
        ) from exc
    symbols = {str(value): index + 1 for index, value in enumerate(atom_symbols)}
    unknown = sorted(
        {atom.GetSymbol() for atom in molecule.GetAtoms()} - set(symbols)
    )
    if unknown:
        raise GlobalGCENativeRuleError(
            f"Parent contains atoms absent from frozen GlobalGCE vocabulary: {unknown}"
        )
    bond_index = {str(value).lower(): index for index, value in enumerate(bond_names)}
    count = molecule.GetNumAtoms()
    feature = torch.zeros((count, len(atom_symbols) + 1), dtype=torch.float32)
    feature[:, 0] = 1.0
    adjacency = torch.zeros((count, count), dtype=torch.float32)
    edge_attr = torch.zeros(
        (count * (count - 1) // 2, len(bond_names)), dtype=torch.float32
    )
    edge_attr[:, 0] = 1.0
    attributes: list[dict[str, Any]] = []
    for atom in molecule.GetAtoms():
        index = int(atom.GetIdx())
        feature[index, 0] = 0.0
        feature[index, symbols[atom.GetSymbol()]] = 1.0
        attributes.append(
            {
                "native_node_index": index,
                "atomic_num": int(atom.GetAtomicNum()),
                "formal_charge": int(atom.GetFormalCharge()),
                "isotope": int(atom.GetIsotope()),
                "chiral_tag": int(atom.GetChiralTag()),
                "num_explicit_hs": int(atom.GetNumExplicitHs()),
                "no_implicit": bool(atom.GetNoImplicit()),
            }
        )
    for bond in molecule.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        adjacency[left, right] = adjacency[right, left] = 1.0
        if bond.GetBondType() == Chem.BondType.TRIPLE:
            name = "triple"
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            name = "double"
        else:
            name = "single"
        if name not in bond_index or bond_index[name] <= 0:
            raise GlobalGCENativeRuleError(f"Unsupported native bond: {name}")
        position = _edge_position(left, right)
        edge_attr[position, 0] = 0.0
        edge_attr[position, bond_index[name]] = 1.0
    return NativeParentTensors(
        canonical_smiles=canonical,
        feature=feature,
        adjacency=adjacency,
        edge_attr=edge_attr,
        atom_attributes=tuple(attributes),
    )


def enumerate_labeled_rule_matches(
    parent: NativeParentTensors, rule: GlobalGCENativeRule
) -> list[dict[int, int]]:
    """Return every exact node/edge-labelled official subgraph match."""

    parent_graph = _labeled_graph(parent.feature, parent.adjacency, parent.edge_attr)
    lhs_graph = _labeled_graph(
        rule.lhs_feature, rule.lhs_adjacency, rule.lhs_edge_attr
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        parent_graph,
        lhs_graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            ["label"], [None]
        ),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match(
            ["label"], [None]
        ),
    )
    matches: list[dict[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    lhs_nodes = set(lhs_graph.nodes)
    for raw in matcher.subgraph_isomorphisms_iter():
        mapping = {int(parent_node): int(lhs_node) for parent_node, lhs_node in raw.items()}
        if set(mapping.values()) != lhs_nodes or len(mapping) != len(lhs_nodes):
            raise GlobalGCENativeRuleError(
                "GlobalGCE subgraph isomorphism did not cover the exact LHS"
            )
        identity = tuple(mapping.items())
        if identity in seen:
            raise GlobalGCENativeRuleError(
                "Duplicate native GlobalGCE match mapping was emitted"
            )
        seen.add(identity)
        matches.append(mapping)
    matches.sort(key=lambda row: tuple(row.items()))
    return matches


def _pad_parent_for_rule(
    parent: NativeParentTensors,
    rule: GlobalGCENativeRule,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    parent_nodes = int(parent.feature.shape[0])
    pad = rule.maximum_nodes - len(rule.lhs_nodes)
    if pad < 0:
        raise GlobalGCENativeRuleError("Rule maximum nodes is smaller than its LHS")
    total = parent_nodes + pad
    feature = torch.zeros((total, int(parent.feature.shape[1])), dtype=torch.float32)
    feature[:, 0] = 1.0
    feature[:parent_nodes] = parent.feature
    adjacency = torch.zeros((total, total), dtype=torch.float32)
    adjacency[:parent_nodes, :parent_nodes] = parent.adjacency
    edge_attr = torch.zeros(
        (total * (total - 1) // 2, int(parent.edge_attr.shape[1])),
        dtype=torch.float32,
    )
    edge_attr[:, 0] = 1.0
    for left in range(parent_nodes):
        for right in range(left + 1, parent_nodes):
            edge_attr[_edge_position(left, right)] = parent.edge_attr[
                _edge_position(left, right)
            ]
    return feature, adjacency, edge_attr


def apply_official_rule_tensors(
    parent: NativeParentTensors,
    rule: GlobalGCENativeRule,
    mapping: Mapping[int, int],
) -> tuple[Any, Any, Any, tuple[int, ...]]:
    """Apply one match with the official mask-square overwrite semantics."""

    lhs_nodes = set(rule.lhs_nodes)
    normalized = {int(key): int(value) for key, value in mapping.items()}
    if set(normalized.values()) != lhs_nodes or len(normalized) != len(lhs_nodes):
        raise GlobalGCENativeRuleError("Rule mapping is not one exact LHS bijection")
    if any(index < 0 or index >= int(parent.feature.shape[0]) for index in normalized):
        raise GlobalGCENativeRuleError("Rule mapping points outside the parent graph")
    # ``generate_fs_mask`` in pinned upstream uses mapping.keys() in insertion
    # order, then appends the new-node indices.  Do not reorder this sequence.
    parent_nodes = int(parent.feature.shape[0])
    feature, adjacency, edge_attr = _pad_parent_for_rule(parent, rule)
    total = int(feature.shape[0])
    mask_order = tuple(normalized.keys()) + tuple(range(parent_nodes, total))
    if len(mask_order) != rule.maximum_nodes or len(set(mask_order)) != len(mask_order):
        raise GlobalGCENativeRuleError("Official mask order is not a unique RHS index map")
    for rhs_index, target in enumerate(mask_order):
        feature[target] = rule.rhs_feature[rhs_index]
    for rhs_left, target_left in enumerate(mask_order):
        for rhs_right, target_right in enumerate(mask_order):
            adjacency[target_left, target_right] = rule.rhs_adjacency[
                rhs_left, rhs_right
            ]
    for rhs_left in range(rule.maximum_nodes):
        for rhs_right in range(rhs_left + 1, rule.maximum_nodes):
            edge_attr[_edge_position(mask_order[rhs_left], mask_order[rhs_right])] = (
                rule.rhs_edge_attr[_edge_position(rhs_left, rhs_right)]
            )
    return feature, adjacency, edge_attr, mask_order


def _apply_atom_attributes(atom: Any, attributes: Mapping[str, Any]) -> None:
    Chem = _rdkit()
    atom.SetFormalCharge(int(attributes.get("formal_charge") or 0))
    atom.SetIsotope(int(attributes.get("isotope") or 0))
    atom.SetChiralTag(Chem.rdchem.ChiralType(int(attributes.get("chiral_tag") or 0)))
    atom.SetNumExplicitHs(int(attributes.get("num_explicit_hs") or 0))
    atom.SetNoImplicit(bool(attributes.get("no_implicit", False)))


def decode_applied_rule(
    *,
    parent: NativeParentTensors,
    rule: GlobalGCENativeRule,
    feature: Any,
    adjacency: Any,
    edge_attr: Any,
    mask_order: Sequence[int],
) -> dict[str, Any]:
    """Hard-decode, sanitize, and audit one native application."""

    Chem = _rdkit()
    graph = _labeled_graph(feature, adjacency, edge_attr)
    active = sorted(int(value) for value in graph.nodes)
    if not active or not nx.is_connected(graph):
        raise GlobalGCENativeRuleError("Applied GlobalGCE RHS is empty or disconnected")
    parent_count = int(parent.feature.shape[0])
    source_attributes = {int(row["native_node_index"]): row for row in parent.atom_attributes}
    editable = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    inherited = 0
    reset = 0
    for old_index in active:
        label = int(graph.nodes[old_index]["label"])
        if not 0 < label <= len(rule.atom_symbols):
            raise GlobalGCENativeRuleError(f"Unknown applied atom label: {label}")
        atom = Chem.Atom(rule.atom_symbols[label - 1])
        attributes = source_attributes.get(old_index)
        if (
            old_index < parent_count
            and attributes is not None
            and int(attributes["atomic_num"]) == int(atom.GetAtomicNum())
        ):
            _apply_atom_attributes(atom, attributes)
            inherited += 1
        else:
            # The official features encode atom type but not charge/chirality.
            # A changed or appended atom therefore receives explicit neutral
            # defaults; this is recorded and never guessed from another atom.
            reset += 1
        old_to_new[old_index] = int(editable.AddAtom(atom))
    for left, right, data in graph.edges(data=True):
        label = int(data["label"])
        if not 0 < label < len(rule.bond_names):
            raise GlobalGCENativeRuleError(f"Unknown applied bond label: {label}")
        name = rule.bond_names[label].strip().lower()
        types = {
            "single": Chem.BondType.SINGLE,
            "double": Chem.BondType.DOUBLE,
            "triple": Chem.BondType.TRIPLE,
            "aromatic": Chem.BondType.AROMATIC,
        }
        if name not in types:
            raise GlobalGCENativeRuleError(f"Unsupported applied bond label: {name}")
        editable.AddBond(old_to_new[left], old_to_new[right], types[name])
    molecule = editable.GetMol()
    try:
        Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise GlobalGCENativeRuleError("Applied molecule has multiple components")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except GlobalGCENativeRuleError:
        raise
    except Exception as exc:
        raise GlobalGCENativeRuleError("Applied GlobalGCE molecule failed sanitization") from exc
    if not canonical or "." in canonical:
        raise GlobalGCENativeRuleError("Applied GlobalGCE molecule is disconnected")
    mask = set(int(value) for value in mask_order)
    boundary_pairs = []
    boundary_preserved = True
    for inside in sorted(mask & set(range(parent_count))):
        for outside in range(parent_count):
            if outside in mask:
                continue
            before = float(parent.adjacency[inside, outside].item())
            after = float(adjacency[inside, outside].item())
            before_bond = (
                _hard_label(parent.edge_attr[_edge_position(inside, outside)])
                if before > 0.5
                else 0
            )
            after_bond = (
                _hard_label(edge_attr[_edge_position(inside, outside)])
                if after > 0.5
                else 0
            )
            if before > 0.5 or after > 0.5:
                boundary_pairs.append(
                    {
                        "inside": inside,
                        "outside": outside,
                        "adjacency_before": before,
                        "adjacency_after": after,
                        "bond_label_before": before_bond,
                        "bond_label_after": after_bond,
                    }
                )
            if (
                not math.isclose(before, after, rel_tol=0.0, abs_tol=0.0)
                or before_bond != after_bond
            ):
                boundary_preserved = False
    if not boundary_preserved:
        raise GlobalGCENativeRuleError("Native rule application changed a boundary edge")
    return {
        "canonical_smiles": canonical,
        "num_atoms": int(molecule.GetNumAtoms()),
        "num_bonds": int(molecule.GetNumBonds()),
        "mask_order": [int(value) for value in mask_order],
        "boundary_pairs": boundary_pairs,
        "boundary_attachments_preserved": True,
        "source_attributes_inherited": inherited,
        "source_attributes_reset": reset,
        "connected": True,
        "sanitized": True,
    }


def apply_rule_to_parent(
    parent_smiles: str, rule: GlobalGCENativeRule
) -> list[dict[str, Any]]:
    """Enumerate and apply every legal match; never choose an arbitrary first."""

    parent = build_parent_native_tensors(
        parent_smiles, atom_symbols=rule.atom_symbols, bond_names=rule.bond_names
    )
    matches = enumerate_labeled_rule_matches(parent, rule)
    results: list[dict[str, Any]] = []
    seen_match_ids: set[str] = set()
    for index, mapping in enumerate(matches):
        match_id = stable_sha256(
            {
                "rule_id": rule.rule_id,
                "parent_smiles": parent.canonical_smiles,
                "mapping": [[left, right] for left, right in mapping.items()],
            }
        )
        if match_id in seen_match_ids:
            raise GlobalGCENativeRuleError("Native match identity collision")
        seen_match_ids.add(match_id)
        row: dict[str, Any] = {
            "match_index": index,
            "match_id": match_id,
            "mapping": [[left, right] for left, right in mapping.items()],
            "valid": False,
        }
        try:
            feature, adjacency, edge_attr, mask = apply_official_rule_tensors(
                parent, rule, mapping
            )
            row.update(
                decode_applied_rule(
                    parent=parent,
                    rule=rule,
                    feature=feature,
                    adjacency=adjacency,
                    edge_attr=edge_attr,
                    mask_order=mask,
                )
            )
            row["valid"] = True
            row["failure_reason"] = None
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}:{exc}"
        results.append(row)
    return results


def _parity_fixture() -> tuple[NativeParentTensors, GlobalGCENativeRule, dict[int, int]]:
    torch = _torch()
    parent = NativeParentTensors(
        canonical_smiles="CCC",
        feature=torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32),
        adjacency=torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32
        ),
        edge_attr=torch.tensor([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32),
        atom_attributes=tuple(
            {"native_node_index": index, "atomic_num": 6} for index in range(3)
        ),
    )
    lhs_feature = torch.tensor([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    lhs_adjacency = torch.tensor(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32
    )
    lhs_edge = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    rhs_feature = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
    rhs_adjacency = torch.tensor(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32
    )
    rhs_edge = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    rule = GlobalGCENativeRule(
        rule_id="parity-rule",
        native_rule_index=0,
        lhs_feature=lhs_feature,
        lhs_adjacency=lhs_adjacency,
        lhs_edge_attr=lhs_edge,
        rhs_feature=rhs_feature,
        rhs_adjacency=rhs_adjacency,
        rhs_edge_attr=rhs_edge,
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double"),
    )
    rule.validate()
    return parent, rule, {1: 0, 2: 1}


def _load_audited_functions(
    source_path: Path,
    *,
    names: set[str],
    namespace: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    matches = {
        name: [
            node
            for node in ast.walk(parsed)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        for name in names
    }
    invalid = {name: len(nodes) for name, nodes in matches.items() if len(nodes) != 1}
    if invalid:
        raise GlobalGCENativeRuleError(
            f"Pinned GlobalGCE function identity is ambiguous or absent: {invalid}"
        )
    selected = [matches[name][0] for name in sorted(names)]
    loaded = dict(namespace)
    extracted = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    # ``exec`` is safe here because the whole source file was bound to an
    # audited commit and SHA-256 above, and only named function AST nodes are
    # retained (no module-level code/imports execute).
    exec(compile(extracted, str(source_path), "exec"), loaded)  # noqa: S102
    return {name: loaded[name] for name in names}


def run_official_tensor_parity(official_root: str | Path) -> dict[str, Any]:
    """Compare production tensor writes with the exact pinned upstream code.

    Importing ``models.GlobalGCE`` also imports the research training stack and
    therefore requires PyG even though its local tensor rewrite is pure
    PyTorch.  The preflight first verifies the complete audited checkout and
    file hashes, then AST-extracts only the three upstream tensor functions.
    This keeps parity tied to the exact audited source bytes without requiring
    an unrelated training dependency merely to validate action semantics.
    """

    audit = validate_official_globalgce_root(official_root)
    parent, rule, mapping = _parity_fixture()
    ours = apply_official_rule_tensors(parent, rule, mapping)
    mask_order = list(ours[3])
    feature, adjacency, edge_attr = _pad_parent_for_rule(parent, rule)

    torch = _torch()
    source_path = Path(audit["official_source_root"]) / "models" / "GlobalGCE.py"
    required = {
        "get_triu_indices",
        "get_3d_mask",
        "concate_inputs_with_local_recourse",
    }
    official_apply = _load_audited_functions(
        source_path,
        names=required,
        namespace={"torch": torch},
    )["concate_inputs_with_local_recourse"]

    from itertools import product

    fsg_path = Path(audit["official_source_root"]) / "models" / "fsg.py"
    official_mask = _load_audited_functions(
        fsg_path,
        names={"generate_fs_mask"},
        namespace={"torch": torch, "product": product},
    )["generate_fs_mask"]

    class _FSGFixture:
        fs_max_nodes = rule.maximum_nodes

    class _GraphFixture:
        @staticmethod
        def number_of_nodes() -> int:
            return int(parent.feature.shape[0])

    upstream_mask = official_mask(_FSGFixture(), mapping, _GraphFixture())
    expected_mask = torch.tensor(
        [[left, right] for left in mask_order for right in mask_order],
        dtype=torch.long,
    )
    mask_matches = torch.equal(expected_mask, upstream_mask)
    if not mask_matches:
        raise GlobalGCENativeRuleError("GlobalGCE official mask-order parity mismatch")

    mask = torch.tensor(
        [
            [
                [[left, right] for left in mask_order for right in mask_order],
                [[-1, -1] for _ in range(len(mask_order) ** 2)],
            ]
        ],
        dtype=torch.long,
    )
    rules = {
        "features_reconst": rule.rhs_feature.unsqueeze(0),
        "adj_reconst": rule.rhs_adjacency.unsqueeze(0),
        "edge_attrs_reconst": rule.rhs_edge_attr.unsqueeze(0),
    }
    upstream_feature, upstream_adjacency, upstream_edge_attr = official_apply(
        feature.unsqueeze(0),
        adjacency.unsqueeze(0),
        edge_attr.unsqueeze(0),
        torch.tensor([[0, -1]], dtype=torch.long),
        mask,
        rules,
        torch.device("cpu"),
    )
    boundary_matches = bool(
        ours[1][0, 1].item() == parent.adjacency[0, 1].item()
        and upstream_adjacency[0, 0, 1].item()
        == parent.adjacency[0, 1].item()
    )
    comparisons = {
        "mask": mask_matches,
        "feature": torch.equal(ours[0], upstream_feature[0]),
        "adjacency": torch.equal(ours[1], upstream_adjacency[0]),
        "edge_attr": torch.equal(ours[2], upstream_edge_attr[0]),
        "boundary_attachment": boundary_matches,
    }
    if not all(comparisons.values()):
        raise GlobalGCENativeRuleError(
            f"GlobalGCE official tensor parity mismatch: {comparisons}"
        )
    return {
        "schema_version": OFFICIAL_TENSOR_PARITY_VERSION,
        "status": "PASS",
        "comparisons": comparisons,
        "mask_order": mask_order,
        "boundary_attachment_checked": boundary_matches,
        "official_function_loading": "ast_extracted_from_hash_verified_source",
        **audit,
    }


def stable_rule_id(payload: Mapping[str, Any]) -> str:
    return "globalgce-rule-" + stable_sha256(dict(payload))[:20]


def iter_rule_payloads(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise GlobalGCENativeRuleError(
                    f"Rule catalog row {line_number} is not an object"
                )
            GlobalGCENativeRule.from_payload(payload)
            yield payload


__all__ = [
    "ACTION_ENGINE_VERSION",
    "GlobalGCENativeRule",
    "GlobalGCENativeRuleError",
    "NativeParentTensors",
    "OFFICIAL_GLOBALGCE_COMMIT",
    "OFFICIAL_SOURCE_SHA256",
    "OFFICIAL_RUNTIME_FILES",
    "RULE_SELECTOR_CHEMISTRY_VERSION",
    "apply_official_rule_tensors",
    "apply_rule_to_parent",
    "build_parent_native_tensors",
    "decode_applied_rule",
    "enumerate_labeled_rule_matches",
    "iter_rule_payloads",
    "run_official_tensor_parity",
    "stable_rule_id",
    "validate_official_globalgce_root",
]
