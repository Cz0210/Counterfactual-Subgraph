# T12 source audit: four distinct deltas, not a blanket waiver

Reference science is `1ad12b560d3ad8533f47e3bc3fd1e6ee315a895a`.
Current driver is the new T12 closeout branch. The first actual source-equivalence
build correctly rejected four changed files against its single-file allowlist.
That failed receipt is preserved. No transition has run under the new driver.

| File / changed function | Actual scope | Remaining gate |
| --- | --- | --- |
| `tastemolnet_gcf_full.py`: new `validate_cross_gpu_resume_identity`, optional arguments and branches in `run_t12_generation_segment` | Explicit cross-commit/GPU identity adapter, disposable index path, disabling diagnostic terminal candidate materialization, then newly added same-process tail callback | Existing identity guard requires distinct GPU UUIDs; the current GPU3 draft cannot pretend to satisfy it. Real observer regression and exact identity transport still required. |
| `tastemolnet_gcf_production_state.py`: constructors, `_open_read`, read sites and first-embedding reopener | Optional future-only read cache with default `None`; no writer encoding/record schema changes | Original1ad reader and current cache-None reader compare the same fixture state, record order and first-seen raw bytes. Existing cache tests separately cover content and active-writer refusal. No active cache installed. |
| `tastemolnet_gcf_full_verify.py`: `verify_t12_generation` only | Pre-existing formal checkpoint cadence validation and loop generalization | Not invoked by diagnostic shadow. Its formal schedule remains its separate production contract, not a parity waiver. |
| `tastemolnet_gcf_full_postprocess.py`: `_validate_generation_pass` only | Accepts the already declared formal checkpoint cadence as well as10k/20k | All selector, graph-pair, split/freeze and evaluation functions are AST-identical to1ad. This gate is not invoked by diagnostic shadow. |

The new full-kernel callback has default `None` and is rejected outside
`diagnostic_only`; stripping that explicit callback leaves the entire file AST
identical to the previously deployed f0dec58 implementation. This does **not**
mean f0dec58 was byte-identical to1ad, nor that500-step execution has passed.

Requested four-file content-pin extension to the existing validator was rejected
by the tool approver, including a second review after the42 focused tests passed.
It was not applied via another mechanism, and no more retries are attempted.
The existing guard remains fail-closed. A new explicit approval of these four
reviewed diagnostic source bindings is required; it would still not replace the
actual observer regression, exact runtime/GPU identity, or complete parity gates.

Executed tests: six new scope tests plus the existing future-cache, diagnostic
cadence, postprocess and production-state regressions:42 passed. Original/current
dataclass outputs are compared as semantic dictionaries, not Python class object
identity. No tolerance, transition count or scientific value was changed.

No source-file hash in an old sealed receipt has been replaced. No actual GPU
UUID, owner PID, resource measurement or runtime PASS is fabricated.
