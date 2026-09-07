"""Train-only connected attributed-deletion beam search, with real query budgets.

No standalone-fragment sanitization is required for an attributed pattern.
Residual chemistry is delegated unchanged to the main hard-deletion kernel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from rdkit import Chem

from src.chem.hard_deletion import apply_hard_deletion_match, enumerate_connected_hard_deletions


@dataclass(frozen=True)
class SearchBudget:
    beam_width: int = 16
    initial_queries_per_parent: int = 128
    extra_parent_limit: int = 128
    extra_queries_per_parent: int = 384
    pool_limit: int = 4096
    minimum_atoms: int = 1
    maximum_deleted_fraction: float = 1.0

    def __post_init__(self):
        if (self.beam_width, self.initial_queries_per_parent, self.extra_parent_limit,
            self.extra_queries_per_parent, self.pool_limit) != (16, 128, 128, 384, 4096):
            raise ValueError("SEARCH_BUDGET_NOT_AUTHORIZED")
        if self.minimum_atoms < 1 or not 0 < self.maximum_deleted_fraction <= 1:
            raise ValueError("INVALID_FROZEN_SIZE_CONSTRAINT")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def connected(mol: Any, indices: Sequence[int]) -> bool:
    wanted = set(indices)
    if not wanted:
        return False
    seen, todo = set(), [min(wanted)]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        todo.extend(int(a.GetIdx()) for a in mol.GetAtomWithIdx(current).GetNeighbors()
                    if a.GetIdx() in wanted and a.GetIdx() not in seen)
    return seen == wanted


def pattern_from_match(mol: Any, indices: Sequence[int]) -> dict[str, Any]:
    """Canonical fragment graph identity has no parent ID or atom-map labels."""
    if not connected(mol, indices):
        raise ValueError("DELETION_PATTERN_NOT_CONNECTED")
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        atom.SetAtomMapNum(0)
    text = Chem.MolFragmentToSmiles(copied, atomsToUse=sorted(indices), canonical=True, isomericSmiles=True)
    pattern = Chem.MolFromSmiles(text, sanitize=False)
    if pattern is None or pattern.GetNumAtoms() != len(set(indices)):
        raise ValueError("ATTRIBUTED_PATTERN_ROUNDTRIP_FAILED")
    pattern.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(pattern)
    return _serialized_pattern(text, pattern)


def _serialized_pattern(text: str, pattern: Any) -> dict[str, Any]:
    """Bind the stored attributed graph, without standalone recanonicalization.

    A parent chiral center can lose distinguishing neighbours after a fragment
    cut. MolFragmentToSmiles on that isolated pattern can remove its original
    chiral tag and explicit-H annotation; that is not an identity verification.
    The original text, atom/bond attributes and all-match deletion stay bound.
    """
    atoms = [{"atomic_number": a.GetAtomicNum(), "formal_charge": a.GetFormalCharge(),
              "aromatic": a.GetIsAromatic(), "isotope": a.GetIsotope(),
              "chiral_tag": int(a.GetChiralTag()), "explicit_hydrogens": a.GetNumExplicitHs()}
             for a in pattern.GetAtoms()]
    bonds = [{"begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx(),
              "bond_type": str(b.GetBondType()), "aromatic": b.GetIsAromatic(),
              "stereo": int(b.GetStereo())} for b in pattern.GetBonds()]
    identity = "BACE_REACH_" + stable_hash({"graph_smiles": text, "schema": "attributed_graph_pattern_v1"})[:24].upper()
    return {"candidate_id": identity, "canonical_fragment": text, "graph_smiles": text,
            "representation": "attributed_graph_pattern_v1", "atoms": atoms, "bonds": bonds,
            "atom_count": len(atoms), "standalone_sanitization_required": False,
            "oracle_backend": "gnn", "classifier_type": "gnn", "rf_oracle_used": False,
            "source_method": "Ours-Reach-v2/train-only-deletion-repair"}


def validate_attributed_candidate(candidate: Mapping[str, Any]):
    if candidate.get("representation") != "attributed_graph_pattern_v1":
        raise ValueError("NOT_ATTRIBUTED_GRAPH_PATTERN")
    text = str(candidate["graph_smiles"])
    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        raise ValueError("ATTRIBUTED_GRAPH_UNREADABLE")
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    if not connected(mol, range(mol.GetNumAtoms())):
        raise ValueError("ATTRIBUTED_PATTERN_NOT_CONNECTED")
    expected = _serialized_pattern(text, mol)
    for key in ("candidate_id", "canonical_fragment", "graph_smiles", "atoms", "bonds", "atom_count"):
        if candidate.get(key) != expected[key]:
            raise ValueError("ATTRIBUTED_PATTERN_BINDING_CONFLICT:" + key)
    return mol


def all_matches(parent_mol: Any, candidate: Mapping[str, Any]) -> list[tuple[int, ...]]:
    text = str(candidate["canonical_fragment"])
    if candidate.get("representation") == "attributed_graph_pattern_v1":
        query = validate_attributed_candidate(candidate)
    else:
        query = Chem.MolFromSmiles(text)
    if query is None:
        raise ValueError("FROZEN_CANDIDATE_GRAPH_UNREADABLE:" + str(candidate["candidate_id"]))
    # maxMatches=0 is RDKit's unlimited count. Matches remain unique atom sets,
    # as in the main connected hard-deletion evaluator.
    matches = parent_mol.GetSubstructMatches(query, uniquify=True, useChirality=False, maxMatches=0)
    return sorted({tuple(sorted(int(x) for x in match)) for match in matches})


def deletion_outcomes(parent_smiles: str, candidate: Mapping[str, Any], parent_id: str) -> list[Any]:
    if candidate.get("representation") != "attributed_graph_pattern_v1":
        return enumerate_connected_hard_deletions(parent_smiles, str(candidate["canonical_fragment"]),
                                                  parent_id=parent_id, candidate_id=str(candidate["candidate_id"]))
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        raise ValueError("PARENT_INVALID")
    return [apply_hard_deletion_match(mol, atoms, parent_id=parent_id,
                candidate_id=str(candidate["candidate_id"]), match_id=i)
            for i, atoms in enumerate(all_matches(mol, candidate))]


def new_graph_key(smiles: str, oracle_binding: str) -> str:
    return stable_hash({"residual": smiles, "oracle_binding": oracle_binding})


def search_parent(*, parent_id: str, parent_smiles: str, before: Mapping[str, Any],
                  predict: Callable[[Sequence[str]], Sequence[Mapping[str, Any]]],
                  old_candidates: Sequence[Mapping[str, Any]], oracle_binding: str,
                  budget: SearchBudget, maximum_new_queries: int,
                  previous: Mapping[str, Any] | None = None,
                  source_label: int = 1,
                  initial_oracle_cache: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One deterministic parent stage; a saved first pass can receive +384.

    Invalid graphs and cache hits do not spend oracle queries. They still count
    as examined actions. Beam traversal is bounded, not an impossibility proof.
    """
    if maximum_new_queries not in (budget.initial_queries_per_parent, budget.extra_queries_per_parent):
        raise ValueError("INVALID_PARENT_PASS_QUERY_BUDGET")
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None or int(before["predicted_label"]) != source_label:
        raise ValueError("SEARCH_REQUIRES_VALID_TRAIN_SOURCE_ELIGIBLE_PARENT")
    binding = stable_hash({"parent_id": parent_id, "smiles": parent_smiles,
                           "oracle": oracle_binding, "budget": asdict(budget)})
    if previous and previous["binding"] != binding:
        raise ValueError("PARENT_RESUME_BINDING_CHANGED")
    if previous is not None and initial_oracle_cache is not None:
        raise ValueError("INITIAL_CACHE_ONLY_ON_FIRST_PARENT_PASS")
    records = list(previous.get("records", [])) if previous else []
    cache = dict(previous.get("oracle_cache", {})) if previous else {
        key: dict(value) for key, value in (initial_oracle_cache or {}).items()}
    examined = {tuple(r["match_atom_indices"]) for r in records}
    total_old = int(previous.get("new_graph_oracle_queries", 0)) if previous else 0
    queries, hits = 0, 0
    n = mol.GetNumAtoms()
    limit = min(n - 1, int(n * budget.maximum_deleted_fraction))
    seeds = {(i,) for i in range(n)}
    seeds.update(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds())
    seeds.update(tuple(sorted(ring)) for ring in mol.GetRingInfo().AtomRings())
    for candidate in old_candidates:
        seeds.update(all_matches(mol, candidate))
    if previous:
        frontier = [tuple(x) for x in previous.get("frontier", [])]
        seeds = set(frontier)
    else:
        frontier = []

    def priority(record: Mapping[str, Any]) -> tuple[Any, ...]:
        after = record.get("after")
        return (0 if record.get("strict_flip") else 1,
                float(after["probabilities"][source_label]) if after else 2.0,
                len(record["match_atom_indices"]), tuple(record["match_atom_indices"]))

    pending = sorted(seeds, key=lambda x: (len(x), x))
    stopped_for_budget = False
    while pending:
        layer: list[dict[str, Any]] = []
        unprocessed: list[tuple[int, ...]] = []
        for index, atoms in enumerate(pending):
            if atoms in examined or len(atoms) < budget.minimum_atoms or len(atoms) > limit or not connected(mol, atoms):
                continue
            outcome = apply_hard_deletion_match(mol, atoms, parent_id=parent_id)
            row = {**outcome.as_dict(), "stage": "train_search", "before": dict(before),
                   "strict_flip": False, "after": None, "new_oracle_query": False,
                   "oracle_binding": oracle_binding}
            if outcome.valid:
                key = new_graph_key(outcome.residual_smiles, oracle_binding)
                if key not in cache:
                    if queries >= maximum_new_queries:
                        unprocessed = pending[index:]
                        stopped_for_budget = True
                        break
                    prediction = dict(predict([outcome.residual_smiles])[0])
                    cache[key] = prediction
                    queries += 1
                    row["new_oracle_query"] = True
                else:
                    hits += 1
                row["after"] = cache[key]
                row["strict_flip"] = int(cache[key]["predicted_label"]) != source_label
                row["cf_drop"] = float(before["probabilities"][source_label]) - float(cache[key]["probabilities"][source_label])
                row["residual_graph_identity"] = key
                row["pattern"] = pattern_from_match(mol, atoms)
            examined.add(atoms)
            records.append(row)
            layer.append(row)
        # Advance the current beam, not its already expanded historical best.
        # Invalid residuals may lead to valid larger connected deletions (e.g.
        # complete aromatic-ring removal), so their connected states remain
        # eligible with lower priority, without spending an oracle query.
        ranked = sorted(layer, key=priority)
        beam = ranked[:budget.beam_width]
        next_states = set(unprocessed)
        for row in beam:
            atoms = tuple(row["match_atom_indices"])
            if len(atoms) < limit:
                adjacent = {a.GetIdx() for i in atoms for a in mol.GetAtomWithIdx(i).GetNeighbors()} - set(atoms)
                next_states.update(tuple(sorted((*atoms, a))) for a in adjacent)
            if row["strict_flip"]:
                # Witness minimalization uses the same oracle budget and valid
                # connected-removal predicate; no independent uncounted search.
                next_states.update(tuple(i for i in atoms if i != a) for a in atoms if len(atoms) > budget.minimum_atoms)
        next_states -= examined
        frontier = sorted(next_states, key=lambda x: (len(x), x))
        if stopped_for_budget or not frontier:
            break
        pending = frontier
    witnesses = sorted((r for r in records if r.get("strict_flip")), key=priority)
    return {"binding": binding, "parent_id": parent_id, "parent_smiles": parent_smiles,
            "before": dict(before), "records": records, "witnesses": witnesses,
            "oracle_cache": cache, "frontier": [list(x) for x in frontier],
            "new_graph_oracle_queries": total_old + queries, "pass_new_queries": queries,
            "pass_cache_hits": hits, "examined_action_count": len(examined),
            "beam_width": budget.beam_width, "test_loaded": False, "calibration_loaded": False,
            "search_state": "WITNESS_FOUND" if witnesses else "NOT_FOUND_WITHIN_BOUNDED_SEARCH",
            "budget_exhausted": stopped_for_budget, "impossibility_proven": False}


def retain_train_pool(old_candidates: Sequence[Mapping[str, Any]], parent_states: Sequence[Mapping[str, Any]],
                      limit: int = 4096) -> list[dict[str, Any]]:
    """Preserve old pool; cap new graphs by train witness marginal/support only."""
    old = [dict(r) for r in old_candidates]
    if len(old) > limit:
        raise ValueError("OLD_POOL_EXCEEDS_LIMIT")
    by_graph = {str(r["canonical_fragment"]): r for r in old}
    new: dict[str, dict[str, Any]] = {}
    for state in parent_states:
        for row in state.get("witnesses", []):
            pattern = row["pattern"]
            if pattern["canonical_fragment"] in by_graph:
                continue
            target = new.setdefault(pattern["candidate_id"], {**pattern, "source_parent_ids": []})
            if state["parent_id"] not in target["source_parent_ids"]:
                target["source_parent_ids"].append(state["parent_id"])
    remaining, selected, covered = set(new), [], set()
    while remaining and len(selected) + len(old) < limit:
        key = min(remaining, key=lambda k: (-len(set(new[k]["source_parent_ids"]) - covered),
            -len(new[k]["source_parent_ids"]), new[k]["atom_count"], k))
        row = new[key]
        row["source_parent_ids"].sort()
        row["source_parent_count"] = len(row["source_parent_ids"])
        row["train_support_scope"] = "observed_verified_train_witnesses_not_exhaustive_support"
        covered.update(row["source_parent_ids"])
        selected.append(row)
        remaining.remove(key)
    return old + selected
