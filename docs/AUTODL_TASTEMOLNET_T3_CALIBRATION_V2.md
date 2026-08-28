# TasteMolNet T3 fresh calibration v2

T3 starts only from the independent T2 adoption receipt. The managed-v2
worker reads the frozen validation-prediction rows, performs a new scalar
temperature optimization, and writes a `SEALED_CANDIDATE` checkpoint. It
does not load calibration/test payloads and cannot write terminal PASS.

A separate verifier retains the complete SEALED tree plus the exact T2 receipt
and 19-file source bundle, repeats the fit, checks all metrics and provenance,
and proves that the model and feature schema are byte-identical. It then uses
the managed-v2 no-replace publisher. The authoritative checkpoint is:

```text
<calibrated-output>/artifacts/checkpoint/
```

The verifier records model, temperature-file, feature-schema-file, internal
feature-schema, validation-row-order, T2 gate, and T2 source-evidence hashes.
The old temperature embedded in T2 training/reload evidence remains historical;
the fresh `temperature_scaling.json` and updated oracle manifest are the T3
downstream authority.

The only T3 success marker is printed by the independent verifier after atomic
publication:

```text
[TASTE_T3_CALIBRATION_PASS]
```
