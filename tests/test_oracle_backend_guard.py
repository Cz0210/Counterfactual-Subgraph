import pytest

from src.data.dataset_registry import (
    OracleBackendNotAllowedError,
    active_dataset_ids,
    assert_oracle_backend_allowed,
    normalize_dataset_id,
)


def test_active_registry_replaces_bbbp_with_tastemolnet() -> None:
    assert active_dataset_ids() == ("aids", "mutagenicity", "bace", "tastemolnet")
    assert normalize_dataset_id("TasteMolNet") == "tastemolnet"
    assert normalize_dataset_id("bitter-sweet-tasteless") == "tastemolnet"


@pytest.mark.parametrize("dataset", ["bace", "tastemolnet", "taste", "bst"])
def test_formal_new_datasets_reject_rf(dataset: str) -> None:
    with pytest.raises(OracleBackendNotAllowedError, match="prohibited"):
        assert_oracle_backend_allowed(dataset, "rf")


def test_frozen_legacy_routes_keep_rf_compatibility() -> None:
    assert_oracle_backend_allowed("aids", "rf")
    assert_oracle_backend_allowed("mut", "rf")
    assert_oracle_backend_allowed("bace", "GNN")
