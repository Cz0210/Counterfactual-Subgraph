# TasteMolNet T13 external locator follower

This is a narrow, CPU-only publication follower for the post-maintenance T8
dual-branch recovery chain.  The running chain was launched from an immutable
older checkout whose T13 relay correctly writes `completed_output_root` after
its independent verifier passes, but does not write the standard locator used
by the fast16 matrix publisher.

The follower does not launch, resume, stop, or signal T8 or T13.  It binds one
explicit `T8_DUAL_CONTROLLER_ROOT`, follows only that controller's
`downstream-salvage/t13-relay-launch.txt`, and requires:

- exact T8 `PASS_AND_T13_RELAY_PERSISTED` state and matching upstream/downstream
  managed-T8 roots;
- exact T13 controller ID, PID-file, launch-env, UUIDv4 attempt, GPU1, and
  fixed output-base bindings;
- T13 controller heartbeat/state `PASS` after `completed_output_root` appears;
- physical `SEALED` and `PASS` markers, PASS run manifest, independent final
  audit, PASS checkpoint, and the final-audit SHA binding;
- no `FAILED` terminal marker and no GNN ablation.

Only then does it create `cell_root_locator.json` with schema
`fast16_matrix_cell_root_locator_v1`, dataset `TasteMolNet`, and method
`GlobalGCE`.  Creation is no-replace and idempotent for the exact same payload.
The existing matrix publisher still performs the full method-specific terminal
validation and serializes the append through the unique fast16 authority.

## AutoDL launch

Use a clean immutable checkout containing the follower.  The T8 controller
path must be the exact already-running chain; never discover it by globbing.

```bash
export AUTODL_RUNTIME_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime
export AUTODL_CONTROL_ROOT=$AUTODL_RUNTIME_ROOT/control
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
export T13_LOCATOR_REPO_ROOT=/root/autodl-tmp/worktrees/<immutable-recovery-worktree>
export T8_DUAL_CONTROLLER_ROOT=$AUTODL_CONTROL_ROOT/tastemolnet-t8-dual-branch-recovery-20260901T141057Z-465145ec
export T13_LOCATOR_CONTROLLER_ROOT=$AUTODL_CONTROL_ROOT/tastemolnet-t13-external-locator-<fresh-uuid>
export T13_OUTPUT_BASE=$AUTODL_RUNTIME_ROOT/outputs/autodl/tastemolnet/globalgce/t13-full
export T13_LOCATOR_POLL_SECONDS=60
export RUN_GNN_ABLATION=0

bash "$T13_LOCATOR_REPO_ROOT/scripts/autodl/launch_tastemolnet_t13_external_locator_v1.sh"
```

Queue `T13_LOCATOR_CONTROLLER_ROOT/cell_root_locator.json` as the
`terminal_root_locator` for `TasteMolNet/GlobalGCE`.  An absent locator is a
normal waiting state.  Any identity, terminal, or hash mismatch is a hard
failure; the follower never repairs scientific output.
