# BACE candidate-universe v4 audit

The connected v3 merged pool contains 389 rows from 255 train parents and 151
canonical fragments.  Its historical matrix admitted 55 fragments because the
matrix entry path required source-parent oracle success, CFDrop at least 0.2,
and a source strict flip before cross-parent evaluation.

The v4 path keeps source outcomes as features and replays the exact connected
hard deletion on each source row.  On the frozen pool copy used for development,
132 canonical fragments pass the full chemistry and lineage gate.  Nineteen do
not: their recorded source residual status cannot be reproduced under the
current connected/sanitized hard-deletion implementation.  They remain in the
old artifact and in the attrition denominator but are not silently admitted.

The old 151-to-55 reduction decomposes into 41 canonical fragments removed by
the source CFDrop gate, 39 subsequently removed by the source flip gate, and 16
by other legacy source filters.  The formal HPC audit writes row-level evidence
before the v4 calibration matrix is submitted.

No final BACE v4 test evaluation is authorized by this code alone.  The current
connected Q30 value was fitted from an Ours calibration matrix.  The protocol
audit must either prove an existing cross-dataset, method-independent rule or
require pooled calibration Q30/Q50 thresholds to be frozen first.
