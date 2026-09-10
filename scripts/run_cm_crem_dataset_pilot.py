#!/usr/bin/env python3
"""Shared actual 32-train CM pilot; no test or main-table mutation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_dataset_pilot import main
if __name__=='__main__':main()
