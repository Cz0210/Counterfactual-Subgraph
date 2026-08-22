# Tests

This directory will eventually contain chemistry, reward, prompt, inference,
and evaluation tests.

The bootstrap phase includes only small smoke tests for the new interfaces.

`tests/autodl/test_four_gpu_recovery_controller.py` covers the controller-only
contracts: explicit four-GPU opt-in, dependency-safe work conservation,
deterministic fixed parent shards, one bounded OOM retry, delegation to
`exp_run` with the frozen interpreter, and no held-out test access before the
B12 selector freeze. It also covers strict cross-worktree run adoption,
four-shard B13 access, hidden test paths, absolute audit evidence, Commit-B
wrapper environment names, and the user-facing append-only registry fields.
