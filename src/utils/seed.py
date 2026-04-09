"""Reproducibility helpers kept independent of any training framework."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set process-level random seeds for standard Python and NumPy if present."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)
