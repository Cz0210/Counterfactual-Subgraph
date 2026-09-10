#!/usr/bin/env python3
"""BACE saved-pool K20 entrypoint with explicit isolated-mode bootstrap."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_k20_experiment import main
if __name__=='__main__':main()
