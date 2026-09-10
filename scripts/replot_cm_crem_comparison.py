#!/usr/bin/env python3
"""Offline CM/BACE original-GINE comparison; never executes science."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_comparison import main
if __name__=='__main__': main()
