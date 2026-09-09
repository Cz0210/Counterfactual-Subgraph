#!/bin/bash
# Mac-scoped relay is deliberately NOT a schedulable HPC science process.
# Paired entrypoint documents the platform boundary; run the Python entry on Mac.
# Requires the mounted authorized external drive; 300s program polling, bounded
# 168h lifetime and 6h asset wait. It imports only this accepted CM result package.
set -euo pipefail
echo "run_cm_crem_relay.py is a bounded Mac-only relay, not an HPC job." >&2
exit 2
