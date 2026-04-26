"""Chemistry interfaces for parsing, validation, substructure checks, and deletion."""

from src.chem.component_salvage import salvage_connected_component
from src.chem.core_fragment import (
    BoundaryBondRecord,
    CoreFragmentNormalizationResult,
    ParentFragmentMatchResult,
    build_dummy_fragment_from_parent_match,
    match_core_fragment_to_parent,
    normalize_core_fragment,
)
from src.chem.deletion import (
    FragmentDeletionEngine,
    delete_fragment_from_parent,
    get_remainder_graph,
)
from src.chem.minimal_repair import repair_minimal_fragment_syntax
from src.chem.minimal_repair import generate_minimal_syntax_repair_candidates
from src.chem.projection import (
    ParentProjectionCandidate,
    build_parent_projection_candidates,
    project_fragment_to_parent_subgraph,
)
from src.chem.repair import repair_fragment_to_parent_subgraph
from src.chem.smiles_utils import (
    canonicalize_smiles,
    is_rdkit_available,
    parse_smiles,
    sanitize_molecule,
)
from src.chem.substructure import (
    SubstructureMatcher,
    find_parent_substructure_matches,
    is_connected_fragment,
    is_parent_substructure,
    is_valid_capped_subgraph,
)
from src.chem.types import (
    ChemistryFailureType,
    DeletionResult,
    FragmentComponentSalvageResult,
    FragmentProjectionResult,
    FragmentRepairResult,
    FragmentSyntaxRepairCandidate,
    FragmentSyntaxRepairResult,
    FragmentValidationResult,
    ParsedMolecule,
)
from src.chem.validation import FragmentValidator, validate_fragment_candidate

__all__ = [
    "ChemistryFailureType",
    "CoreFragmentNormalizationResult",
    "DeletionResult",
    "FragmentComponentSalvageResult",
    "FragmentDeletionEngine",
    "ParentFragmentMatchResult",
    "FragmentProjectionResult",
    "FragmentRepairResult",
    "FragmentSyntaxRepairCandidate",
    "FragmentSyntaxRepairResult",
    "FragmentValidationResult",
    "FragmentValidator",
    "BoundaryBondRecord",
    "ParentProjectionCandidate",
    "ParsedMolecule",
    "SubstructureMatcher",
    "build_dummy_fragment_from_parent_match",
    "build_parent_projection_candidates",
    "canonicalize_smiles",
    "delete_fragment_from_parent",
    "find_parent_substructure_matches",
    "generate_minimal_syntax_repair_candidates",
    "get_remainder_graph",
    "is_rdkit_available",
    "is_connected_fragment",
    "is_parent_substructure",
    "is_valid_capped_subgraph",
    "match_core_fragment_to_parent",
    "normalize_core_fragment",
    "parse_smiles",
    "project_fragment_to_parent_subgraph",
    "repair_minimal_fragment_syntax",
    "repair_fragment_to_parent_subgraph",
    "salvage_connected_component",
    "sanitize_molecule",
    "validate_fragment_candidate",
]
