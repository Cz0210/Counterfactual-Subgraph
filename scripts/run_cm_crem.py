#!/usr/bin/env python3
"""Thin isolated-mode CM-CReM CLI. Heavy work runs only inside Slurm."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cm_crem_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
