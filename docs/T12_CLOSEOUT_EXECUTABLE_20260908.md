# T12 closeout: implemented interfaces, not scientific PASS

The current reader and its execution tree are unchanged. No transition was
launched by this development change, including no replay of0–250.

## Actual implementation

- `t12_raw_evidence.py`: captures raw classifier output, exact query NeuroSED
  and normalizer from actual executed functions, binds them to structural/model
  graph and canonical-query identities, preserves observed versus authoritative
  canonical probabilities separately, persists compact gzip evidence. A cache
  hit without raw evidence remains `CACHE_RAW_EVIDENCE_MISSING`. It does not
  recompute or invert anything. Default serialized evidence bound is64MiB;
  Python/process RSS must still be included by the resource provider.
- `t12_shadow_execution.py`: runs the existing official segment using the
  original250 checkpoint fork, refuses an active original reader, requires the
  actual inherited GPUFD, contention/UUID/current-parent identity, fresh resource
  provider, source-equivalence and observational-regression evidence. Exclusive
  new ledger directories prevent silently repeating a segment.
- `run_live_tail`: seals the500 ledger before501, uses the same live
  walker/bridge/model/RNG objects, then seals510. This is an explicit conditional
  extra10-step segment, not an independent reload.
- Existing `run_t12_reference_500_v1.py` optionally calls
  `dispatch_inherited_activation` only after full parity exists. The actual FD
  is passed using `pass_fds`; child repeats the existing registry/resource checks.

## Source boundary

Only `tastemolnet_gcf_full.py` gains an optional diagnostic callback, default
`None`, rejected outside `diagnostic_only`. The original500 receipt is written
first. The callback executes while the original source/bridge contexts are still
alive. The default-body AST regression proves no other original executable AST
changed from f0dec58. Its new content pin and audit scope are separate from the
old transport-only pin; this source audit is not runtime parity.

## Still required before actual science

1. Current reader must finish its510 receipt naturally.
2. Immutable stage specs must use fresh fork roots and correct absolute paths;
   existing fork function copies only committed original250 prefixes.
3. Build the explicit new source-equivalence receipt against1ad12b56.
4. Run the allowed train-only observational regression with the actual adapter;
   CPU fixture tests are not that GPU production evidence.
5. Bind original owner PID/start ticks and resource-provider command in the
   fresh task's `science_contract.shadow_binding`; do not invent future PID.
6. Supply real original raw evidence where available. Missing pre250 raw logits
   or distances cannot be recovered by dividing probabilities or inverting masks.
7. Execute reference251–500+continuous501–510, then independent reload501–510;
   likewise accelerated arm. All planned steps remain capped at540, default520
   plus the predeclared missing continuous tails20. Only complete bound comparisons
   can release the existing fresh-zero15-stage plan and canonical publisher.

The CLI now has real `shadow-segment`, but current status remains
`WAITING_EXECUTION_BINDINGS`; it does not claim a science PID or unattended
completion until those runtime inputs are actually bound.
