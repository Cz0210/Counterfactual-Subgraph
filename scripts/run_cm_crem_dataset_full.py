#!/usr/bin/env python3
"""CM full continuation from an accepted dataset pilot; immutable stages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_dataset_full import main
if __name__ == '__main__': main()
