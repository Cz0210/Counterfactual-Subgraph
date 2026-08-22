from __future__ import annotations

import pytest

from scripts.autodl.exp_run import _parse_environment
from src.utils.autodl_runtime import AutoDLRuntimeError


def test_scheduler_tokenizers_parallelism_environment_is_allowed() -> None:
    assert _parse_environment(["TOKENIZERS_PARALLELISM=false"]) == {
        "TOKENIZERS_PARALLELISM": "false"
    }


@pytest.mark.parametrize(
    "key",
    (
        "TOKEN",
        "API_TOKEN",
        "SECRET",
        "PASSWORD",
        "AUTHORIZATION",
        "tokenizers_parallelism",
    ),
)
def test_credential_like_environment_keys_remain_rejected(key: str) -> None:
    with pytest.raises(AutoDLRuntimeError, match="Unsafe environment key"):
        _parse_environment([f"{key}=not-a-real-secret"])
