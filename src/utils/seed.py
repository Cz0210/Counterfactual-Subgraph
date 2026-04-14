"""Reproducibility helpers kept independent of any training framework."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set process-level random seeds for common ML dependencies when present."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
