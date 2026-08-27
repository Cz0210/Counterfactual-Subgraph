"""Stable consumer surface for the TasteMolNet T5 clean policy.

Downstream stages should import validators from this module rather than the
producer implementation. ``hold_clean_policy_output`` keeps the physical T5
root descriptor open and supports repeat validation while a consumer records
its own input authority.
"""

from src.train.tastemolnet_clean_policy_init import (
    HELD_LOAD_TOKEN_SCHEMA,
    HeldTasteCleanPolicyLoadAuthority,
    HeldTasteCleanPolicyOutput,
    HeldTasteCleanPolicySourceModel,
    PASS_MARKER,
    TasteCleanPolicyLoadToken,
    TasteCleanPolicyError,
    hold_clean_policy_load_authority,
    hold_clean_policy_output,
    hold_source_model_for_clean_policy,
    validate_clean_policy_output,
    validate_source_model_for_clean_policy,
)

__all__ = [
    "HELD_LOAD_TOKEN_SCHEMA",
    "HeldTasteCleanPolicyLoadAuthority",
    "HeldTasteCleanPolicyOutput",
    "HeldTasteCleanPolicySourceModel",
    "PASS_MARKER",
    "TasteCleanPolicyLoadToken",
    "TasteCleanPolicyError",
    "hold_clean_policy_load_authority",
    "hold_clean_policy_output",
    "hold_source_model_for_clean_policy",
    "validate_clean_policy_output",
    "validate_source_model_for_clean_policy",
]
