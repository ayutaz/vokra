# Candle four-pin diagnostic preparation — 2026-10-15

> **Corrected actual resolution succeeds:** run `38711006458` at `27a28413`
> produces all thirteen evidence files. Manager independently verifies the
> complete 161-to-154 record change, 151 identical retained records and
> 105-package CPU graph. This supersedes the resolution failure, not the
> original five-finding audit. The new candidate still needs its own strict
> audit; paste, compatibility/license/operator and real-weight gates remain.

> **Actual first attempt and cause correction:** published head `c538e390`
> reaches the real solver and stops at an unnecessary fourth update. The
> `anyhow` update already removes `backtrace` and `adler`; see the actual
> evidence below. The corrected next experiment uses three precise updates
> and explicitly rejects either removed package remaining. The four-pin
> preparation below is retained as history, not today's command sequence.

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

## Actual first attempt: removed package is not an update target

[Run 38709911065](https://github.com/ayutaz/vokra/actions/runs/38709911065),
job `116338436819`, uses published head
`c538e39062cd2bbde4044e6cb465ed68f9218b3e`. Its hosted seventeen-test step
passes with one explicit private-fixture skip (sixteen executed successes),
then verifies the host/toolchain and authenticates the source projection.
The real diagnostic stops at 09:03:37 UTC with wrapper exit 2, because the
fourth precise command returns Cargo exit **101**. The strict complete-set
step fails for the three absent final-candidate files; always-upload succeeds
with the ten files actually produced. It is not thirteen-file acceptance.

Artifact `11959593981` is **333,054 compressed bytes**. Manager recovers it
through the official CLI. The [unaltered failed diagnostic](kyutai-candle-pin-first-attempt-2026-10-15.json)
is **81,608 bytes**, SHA-256
`db9240ecf426d6729fb8f92344b804bd932fb8fbf9cb84ada7b54d43d37627ab`.
Manager independently revalidates the original/base semantic projection and
memmap-only patch. Patched lock SHA-256 remains
`303c49433187d5938d0f2a27d3f5ded9eb094388dd67e6702a0a8b7558eaec01`.
The genuine post-patch CPU/whole metadata contains 105/161 packages and
memmap2 0.9.11. Its two evidence wrappers are 849,497 / 1,205,830 bytes,
within the unchanged 2 MiB writer cap; command captures are 499,345 /
713,684 bytes. This verifies the earlier serialization/argv corrections on
the actual hosted path, not the final updated graph.

The first three actual update commands return zero. Their retained stderr
and checkpoint hashes establish the next action:

| Update | Actual observation / checkpoint SHA-256 |
|---|---|
| crossbeam-epoch 0.9.20 | Updated from 0.9.18; `4eebb62c16b9ddd8ecc720831d7e9fd7c5cc492ef0d60b9b11c44f3b369b64e4` |
| anyhow 1.0.103 | Updated from 1.0.98 and removed seven old transitive packages; `48181d41439c6262177a83b3bf03b67b0a903caffdec77506d30205934423001` |
| rand 0.9.3 | Updated from 0.9.0; 38,252-byte checkpoint `d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8` |
| backtrace 0.3.76 | Exit 101: the package ID no longer matches any package |

The anyhow command explicitly removes `addr2line 0.21.0`, `adler 1.0.2`,
`backtrace 0.3.71`, `gimli 0.28.1`, `miniz_oxide 0.7.4`, `object 0.32.2`
and `rustc-demangle 0.1.24`. The original backtrace/miniz upgrade hypothesis
was therefore unnecessary; do not retry it unchanged or replace the failed
receipt. The final checkpoint's raw lock was not recovered, so these logs
and hashes alone are not a complete lock-delta or fresh advisory verdict.

The next reviewed correction removes only the unnecessary backtrace command,
retains the three exact security candidates, and adds a fail-closed assertion
that neither `backtrace` nor `adler` exists in the final whole lock. All
source, memmap, graph, deadline, failure-evidence and thirteen-file gates
remain. `paste` remains unresolved. Fresh corrected-head resolution and
complete raw candidate evidence are required before any new strict audit.

The corrected causal fixture contains all seven observed old transitive
packages, retains them through the crossbeam update, removes them only at
the anyhow update and keeps them absent after rand. It asserts both
intermediate states and complete final removed records. Direct rejection
tests cover either forbidden name remaining; `adler2` is a different name,
not an automatic alias or a reason to waive a future audit finding.
Corrected helper/test/workflow hashes are respectively
`7bb61d1f2adaa58e64492584f67ecf7ca2b8d5f8f3943aa2a143b767e8661f6d`,
`1387ce6a1d32ffb6702916ed3d5d9c74fe136007703ceeae2f616f63ad1411b2`
and `b7c98ab376ef3e459bac7ac73d23d6cfb0ff73a4f929b16044800bde905782d5`.
Manager independently passes all eighteen focused tests, then **87 canonical
source-only tests in 10.650 seconds**, actionlint, 51-workflow hygiene and
documentation references. The extra canonical observer tests are not an
upstream/model run. Fresh GitHub main readback still equals `7701cc67`.

The unchanged old advisory workflow also automatically runs at this head
and finishes failure as run `38709910644`; a later cancellation request is
refused because it is already terminal. No cancellation or PASS is claimed.
The known old-lock failure is distinct from this removed-package error.
No model execution, CPU parity, owner approval, paid worker or publication
occurs; the full original step-4 objective remains active and incomplete.

## Corrected actual resolution and independent acceptance

[Run 38711006458](https://github.com/ayutaz/vokra/actions/runs/38711006458),
job `116342064821`, completes the exact corrected production path at
`27a2841384220c078aa37ff93ea8f9fc8a18594f`. Its source-only diagnostic,
strict complete-evidence check and upload all succeed. Artifact
`11961100426`, named `kyutai-candle-pin-remediation-38711006458-1`, is
**457,074 compressed bytes**. Manager recovers all **thirteen regular files**,
totaling **7,040,218 bytes**, and independently verifies:

- The [complete candidate lock](kyutai-candle-candidate-2026-10-15.lock) is
  **38,252 bytes**, SHA-256
  `d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8`.
  It exactly matches the first failed attempt's third-update checkpoint;
  the new run now recovers the actual bytes and final native graphs.
- Whole-lock records change from **161 to 154**. All **151 retained complete
  records**, their relative order and all top-level TOML fields are unchanged.
  The only version replacements are crossbeam-epoch 0.9.18 to 0.9.20,
  anyhow 1.0.98 to 1.0.103 and rand 0.9.0 to 0.9.3. Exactly the seven old
  anyhow-chain packages listed above disappear. No other package is added.
- The full memmap2 0.9.11 and paste 1.0.15 records are identical to the
  authenticated patched baseline. The removed adler/backtrace names are
  absent, not ignored or renamed in place.
- Candidate native CPU/whole metadata contains **105 / 154 packages**.
  Whole identities equal the complete candidate lock; CPU identities are
  contained in it, resolve-node identities/edges are valid and CPU nodes
  activate no GPU feature. All seven actual commands return zero.
- The [original successful resolution diagnostic](kyutai-candle-pin-success-2026-10-15.json)
  is **228,834 bytes**, SHA-256
  `5ee83d9cf3e2976b6350a7cd45346f27bfb40e4194616dcfb52e89c7f9385e44`.
  Its status remains `PIN_REMEDIATION_SOURCE_ONLY_REVIEW_PENDING`, paste is
  explicitly unresolved/not waived, approval is null, model execution is
  `NOT_AUTHORIZED` and publication is `NO_UPLOAD`.

The exact recovered member bindings below are for this producer only. They
do not replace the old seven-member producer or its five-finding receipt.
Paths are beneath the artifact root, using `capture` for
`vokra-candle-pin-capture-38711006458-1` and `result` for
`vokra-candle-pin-result-38711006458-1`.

| Member | Bytes | SHA-256 |
|---|---:|---|
| capture/base/Cargo.lock | 39,842 | `eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de` |
| capture/initial_metadata.json | 896,267 | `c4712fec75c50bf1b0c2813bc0456994c7e383e145d3a5ec41c42dec9095bbc5` |
| capture/metadata.json | 324,861 | `3a988e3219f9b25d128ec52c27798aaf93f07909e6ad9ef8269d9faee8319fff` |
| capture/patched/Cargo.lock | 39,843 | `303c49433187d5938d0f2a27d3f5ded9eb094388dd67e6702a0a8b7558eaec01` |
| capture/source/rust/Cargo.lock | 150,274 | `bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab` |
| capture/source_receipt.json | 8,320 | `157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec` |
| capture/whole_lock_metadata.json | 1,261,025 | `3ed8961b442e32541af940ab0b84b00f93de7942c8de34b852f4b59af68301a6` |
| result/baseline_cpu_metadata.json | 849,497 | `6964ac9b62708caa3cd794cf26055628e455a5768f9312476049c468ca3c7e0d` |
| result/baseline_whole_metadata.json | 1,205,830 | `cc3728fc52f80ea0554ca3cb9916de0bd45ebe7fa8d5379f541873ef0d7bed99` |
| result/candidate.lock | 38,252 | `d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8` |
| result/candidate_cpu_metadata.json | 849,118 | `9c351398903613d266f41d4d9769f488780aba5c4592312cee54b732c634f043` |
| result/candidate_whole_metadata.json | 1,148,255 | `959357f105d4118301fc0be3bf94bc6f49944e4df5122cb6a8124bfc2b6c2793` |
| result/pin_diagnostic.json | 228,834 | `5ee83d9cf3e2976b6350a7cd45346f27bfb40e4194616dcfb52e89c7f9385e44` |

Next is a separately authenticated strict advisory audit of this exact
154-record candidate, then the genuine maintained paste replacement path.
Do not filter to CPU packages, reuse the old lock's auditor verdict, waive a
warning or rewrite old receipts. Dependency resolution is not a source/macro
compatibility, license/operator or model parity verdict. PR #255 stays draft
and complete exact-head PR/CI acceptance remains separate. No paid worker,
model acquisition/execution, final Apple packet, inventory refresh or public
artifact update occurs. The original full step-4 objective is not complete.
