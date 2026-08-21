"""Classifier-oracle interfaces for counterfactual molecular experiments."""

from src.oracles.base_oracle import BaseOracle, OraclePredictionRecord
from src.oracles.gnn_oracle import GNNOracle
from src.oracles.oracle_factory import build_oracle, create_oracle, oracle_from_config

__all__ = [
    "BaseOracle",
    "GNNOracle",
    "OraclePredictionRecord",
    "build_oracle",
    "create_oracle",
    "oracle_from_config",
]
