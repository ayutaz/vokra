# Candle four-pin diagnostic preparation — 2026-10-15

Accepted baseline is main `7701cc67414d9cb84e4fe3cd6cf228a10bb8b792`.
The [actual advisory result](kyutai-candle-source-advisory-result-2026-10-15.md)
has five findings; this document prepares a source-only dependency-resolution
experiment, not an advisory PASS or replacement owner approval. The
[paste candidate archive readback](kyutai-candle-pastey-source-readback-2026-10-15.md)
is separate and does not establish compatibility. All non-Apple prerequisites,
independent real-weight CPU parity and final clean-HEAD Apple packets remain
unfinished. No model or paid worker is used in this preparation.

## Exact experiment

The new helper allows only these four updates in the isolated upstream
reference projection:

| Package | Exact candidate | Purpose, not a proven outcome |
|---|---|---|
| crossbeam-epoch | 0.9.20 | Investigate the recorded vulnerability's patched range |
| anyhow | 1.0.103 | Investigate its recorded unsound-warning fix |
| rand | 0.9.3 | Investigate the recorded fix within the existing 0.9 series |
| backtrace | 0.3.76 | Investigate a maintained transitive miniz/adler route |

It cannot perform a broad update, build/check/test/run/fetch, model acquisition,
package execution or publication. Vokra's first-party-only runtime lock is
not the target. `paste 1.0.15` remains explicitly unresolved and not waived.
A successful resolution experiment would still need a new strict advisory
audit, exact license/native review and independently authorized compatibility
and real-weight validation.

The workflow uses one production helper invocation. That invocation creates
one monotonic deadline of at most 480 seconds, calls the unchanged source
collector with remaining time minus a two-second margin, and passes the same
absolute deadline to all later commands. The collector is authenticated at
SHA-256 `75bdaa3da0ce7a3add28a1018e7d319b48b9f18448d7c3093c55fa94fdb2773f`;
its old receipts, source files and earlier measurement are not edited.

Actual source bytes are checked against frozen Git-blob identities, not only
self-reported SHA-256 receipts. Original/base semantic projection and the
memmap-only patch are revalidated. The experiment starts from authenticated
patched/final projection bytes, preserving `memmap2 0.9.11`, rather than the
historical base lock's `0.9.5`. Genuine post-patch CPU and whole metadata are
captured before the four updates; the old initial native metadata is not
mislabelled as a post-patch baseline.

CPU metadata uses `--locked` with the exact Linux target and no default
features. Whole metadata uses the frozen collector's existing exact
`metadata --format-version 1 --all-features` argv, preserving its bounded
2 MiB capture path. Its lock fingerprints must match immediately before and
after each baseline/candidate whole command, including nonzero returns.
Whole-feature metadata is not CPU execution or GPU approval. Full raw native
metadata is retained once; canonical graph hashes and full change records
avoid duplicating it beyond the existing 2 MiB JSON cap.

The planned artifact has thirteen explicitly named files: the original seven
source-capture evidence files, post-patch baseline CPU/whole metadata,
candidate CPU/whole metadata, candidate lock and pin diagnostic. This is a
planned recovery contract, not a claim that a hosted run has produced them.
Nonzero Cargo results retain bounded actual return code/stdout/stderr hashes
and text. If the frozen capture helper cannot provide streams, the diagnostic
marks them unavailable rather than fabricating empty-stream evidence.

## Review found and corrected before actual execution

Manager rejects the first prototype's reset phase deadlines, vulnerable
base-lock reuse, receipt-only source authentication and lost nonzero-command
diagnostics. Subsequent recovered-data replay finds two further actual-size
defects, not hypothetical solver failures:

- Duplicated whole native metadata/package records serialize to 2,888,734
  bytes, exceeding the unchanged 2,097,152-byte writer cap. Retain raw native
  metadata once; preserve complete ordered lock/root/declaration deltas.
- Adding `--locked` to whole metadata misses the frozen runner's exact argv
  branch and selects its 524,288-byte default. Actual whole compact metadata
  is 728,260 bytes. Reuse the established whole argv and strictly reject any
  lock change; do not enlarge caps or mutate the frozen collector at runtime.

Additional causal tests cover consistently rehashed forged-base inputs,
rewritten source receipts, whole-lock mutation with return codes zero/one,
shared budget preservation, complete added/removed records and top-level lock
fields, and honest unavailable-stream diagnostics. A portable large-field
fixture tests the serialization fix without private recovered data. The
separate recovered-data sizing test runs locally and is explicitly skipped
when its private fixture is absent; do not report that case as a hosted pass.

Manager independently runs **57 tests in 6.895 seconds, all passing**, covering
the existing forty source-collector contracts and seventeen new diagnostic
tests, including the real recovered sizing case. This is model-free offline
test evidence, not new dependency resolution. The new workflow passes scoped
actionlint review. Reviewed implementation hashes are:

| File | SHA-256 |
|---|---|
| `.github/workflows/kyutai-candle-pin-remediation.yml` | `c8692c120e6109f6764367f3d19ebbc20b9e455df5270a0d6cc87de6da352167` |
| `pin_remediation_diagnostic.py` | `7104b0545cf389a018f4e3f32ede1a8e5acedc1129710cfff10474595e3ee1aa` |
| `test_pin_remediation_diagnostic.py` | `91d4fac0fc7ec0f0a78266c5b5e05c046cf8281499d6e9a73d4ae05b8de8ef2d` |

Next: integrate with current main and publish the reviewed source-only
diagnostic for an actual hosted attempt. Preserve all actual failures, recover
small evidence and review the complete resulting graph before a new security
verdict. PR #255 remains draft; unrelated drafts and owner/legal/source holds
are not promoted. Scaleway remains last, and no public artifact is changed.

## Integrated source-push boundary

Reviewed preparation is committed as `f9f90510cef31424bfe9e59ad4f2a26aa4abafb8`
with all five normal pre-commit checks passing. Manager separately passes all
eight compliance-scanner regressions, 51-workflow hygiene, documentation
references/community checks and the 144-block examples check (thirty Tier-C
blocks explicitly remain unverified). The evidence copy is independently
checked against the original plus its documented one terminal LF.

Before the corresponding source push, manager compares runtime/test/build
inputs with recorded green VAST source
`4263f41631bd15288f06d3cd5c6b83ea4a8fdf60`: no differences under `crates`,
`tests`, root Cargo files, `.cargo`, `.config` or Rust toolchain files.
The matching tree IDs are `cd87ca121ccf2f5c1249c1165476a66ebec3a7f1`
(`crates`) and `04b30c76bd82baf3229e861f839ad8b9c4610799` (`tests`).
Reuse that [recorded remote evidence](dia-urllib3-minimal-reconciliation-2026-10-15.md)
only for unchanged Rust inputs, following the documented source-push hook
path; do not launch a local deep Cargo run or claim a new VAST replay.
New wrapper/hosted diagnostics and fresh exact-head CI are separate gates.
