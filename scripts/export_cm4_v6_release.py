#!/usr/bin/env python3
"""Export accepted CM4 raw receipts; CPU-only offline publication."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.eval.cm4_v6_release import main
if __name__=='__main__':main()
