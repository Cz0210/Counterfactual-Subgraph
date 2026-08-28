# TasteMolNet T2 Scientific Adoption v2

This route adopts the completed TasteMolNet three-class GINE scientific result
without repairing or rerunning its failed historical controller.

The historical record remains exactly:

- terminal state: `FAILED`;
- reason: `WORKER_PROCESS_IDENTITY_DRIFT`;
- scientific training completion: `PASS`.

The independent verifier authenticates the exact 19-file bundle, its SHA-256
inventory, three-class label map, GINE/seed-7 configuration, train/validation
cache boundary, split and cache hashes, checkpoint reload evidence, finite
multiclass metrics, positive recall for every class, and the traceable training
commit. It then reloads `model.pt` and replays every saved validation logit from
the held validation graph-cache inode. Calibration and test are never loaded as
model inputs, and no RF oracle is permitted.

The old validation temperature is authenticated only as historical evidence.
It does not satisfy T3: the main v2 controller must still create a fresh T3
calibrated bundle by fitting one scalar temperature on validation only.

On PASS, the verifier writes a fresh UUID receipt below:

```text
$CONTROL/tastemolnet-main-v2/adoptions/T2_GINE/<receipt_id>/
```

The receipt records the source run/controller IDs, immutable artifact and input
hashes, training and verifier commits, the retained old failure, validation
replay tolerance, and these mandatory facts:

```text
old_failure_superseded_for_scientific_artifact=true
old_process_evidence_not_rewritten=true
state=ADOPTED_SCIENTIFIC_PASS
```

Run only from a clean immutable AutoDL checkout:

```bash
python -I -B scripts/autodl/adopt_tastemolnet_gine_pass_v2.py \
  --config configs/hpc.yaml \
  --control-root "$AUTODL_CONTROL_ROOT" \
  --artifact-root "$TASTE_T2_ARTIFACT_ROOT" \
  --controller-root "$TASTE_T2_CONTROLLER_ROOT" \
  --training-state-root "$TASTE_T2_TRAINING_STATE_ROOT" \
  --source-run-id "$TASTE_T2_SOURCE_RUN_ID" \
  --source-controller-id "$TASTE_T2_SOURCE_CONTROLLER_ID" \
  --device cpu
```

This command performs no training, sends no process signals, writes no matrix
cell, and does not redistribute TasteMolNet rows.
