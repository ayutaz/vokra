# Candle CPU compatibility attempt — 2026-10-16 JST

> **Later separate confirmation:** the [new authorized job](kyutai-candle-cpu-confirmation-result-2026-10-16.md)
> succeeds for CPU library/API compatibility. This file still records the
> first failed attempt, not an overwritten verdict or real-weight result.

> **Subsequent authorization, not a changed verdict:** one additional
> model-free confirmation job is approved with a fresh activation label.
> See the [current six-stage readback](goal6-execution-readback-2026-10-16.md).
> The first run, failed receipt and evidence below remain immutable.

## Verdict: full library builds; API harness fails; overall not accepted

[Run 38838522884](https://github.com/ayutaz/vokra/actions/runs/38838522884),
attempt 1 / [job 116772969390](https://github.com/ayutaz/vokra/actions/runs/38838522884/job/116772969390),
executes exact PR head `ece1d9b2041ad325f3a1416d168f9d174a6a7450`.
The one permitted actual job is consumed. It starts at 02:07:01 UTC and
finishes at 02:08:27 UTC with failure; checker elapsed time is 54.113 seconds.
No retry, label reapplication or second actual job is performed.

All **16 hosted fixture tests pass without skips** in 8.642 seconds, including
the real Linux orphan-process regression skipped on the maintainer Mac.
Python/Rust bootstrap, original application-artifact download and final
evidence upload succeed. Final checker status remains
`CPU_COMPATIBILITY_FAILED`, wrapper rc 2. This is not an API compatibility,
model, owner/legal, real-weight CPU or Apple result.

## Independently recovered original evidence

Artifact **12020182704**, named
`kyutai-candle-cpu-compatibility-38838522884-1`, has API-reported compressed
size 56,526 bytes and ZIP digest
`sha256:d39dc7b1dc4085723163f712f2935b9443877103f4f9fa753f7c10ca981faf10`.
Manager recovers its thirteen original JSON/text members (651,863 bytes).
The 49,516-byte receipt has SHA-256
`ab4ca92d8f4ce6e611fe31c1352a48a2a24f3d61ede9dd5b6ce0fe7fa416da94`.
The API ZIP digest is recorded as reported, not claimed independently
reproduced from the already extracted download.

Manager independently rehashes all twelve original stdout/stderr streams
against the six completed command summaries. Every command is non-truncated
and non-timeout. One extra source-application entry records only planned
argv; it is not counted as a second execution.

| Actual command | rc | Seconds | Scope |
|---|---:|---:|---|
| Checkout HEAD | 0 | 0.001 | Matches exact published head |
| Frozen source application | 0 | 10.108 | Source-only replay |
| Namespace/toolchain preflight | 0 | 0.100 | Distinct network namespace, no-new-privs, Rust/Cargo 1.99.0 |
| Full CPU library build | 0 | 42.495 | Locked/offline/lib/no-default-features, explicit Linux target |
| Fresh CPU metadata | 0 | 0.058 | Same 105-node canonical graph |
| Uncalled API-signature probe | 1 | 0.018 | E0463: host proc-macro not found |

Native Cargo stdout (113,929 bytes) contains `build-finished: success=true`
and 122 compiler-artifact records, all bound to identities in the actual
post-build CPU metadata. Manager separately checks the native artifact set
contains all fourteen derived CPU gemm identities and pastey's proc macro.
The source, sixteen-parent alias application and
full Moshi/Candle CPU-library build succeed. Retained projection/vendor and
all ninety selected registry-source snapshots match before/after build.
Final lock remains
`21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6`.
Manager independently reconstructs the typed-edge/feature fingerprint from
native post-build metadata:
`99364d685eb1898f2ccf442c68fb51d70f0b5e42f90c04dd0171cb07933be51b`.
Original streams are privately retained; a temporary recovery directory is
not represented as permanent evidence storage.

## Cause and bounded correction

The original 324-byte API stderr reports E0463: `bytemuck_derive` cannot be
found when loading Moshi. Native build JSON proves that this proc macro
already compiled successfully into `target-build/debug/deps`, alongside
pastey. The probe supplied only `target-build/<triple>/debug/deps` as a
dependency search directory. This is a harness search-path omission, not an
unavailable dependency or a failed library/macro build.

[Cargo's documented host/target separation](https://doc.rust-lang.org/cargo/reference/build-cache.html)
explains separate build-script/proc-macro output when `--target` is used.
The [rustc dependency-search flag](https://doc.rust-lang.org/rustc/command-line-arguments.html#-l-add-a-directory-to-the-library-search-path)
permits the additional directory. Luna adds the fresh host dependency path
alongside the existing target path and validates regular host artifacts.
No dependency/version/feature, frozen producer, graph, tolerance, source or
workflow activation contract changes.

Manager reviews that two-file correction and independently runs its offline
suite: **17 enumerated, sixteen pass / one Linux-only orphan test skipped on
macOS**, 0.413 seconds. The new mocked API test checks both dependency paths,
explicit target and failure propagation; it executes no compiler or model.
This corrected candidate has **not** been run on Linux. It cannot turn the
original failed receipt into a pass. Another actual job requires a separate
bounded authorization and a fresh one-shot activation scope.

Reviewed corrected checker SHA-256:
`e86ccaa0d6763cf21a0c8cb0dfe38e86450a5470f23812cc72fd6b321a591183`.
Reviewed corrected test SHA-256:
`2cb57c17228f1e6d2b0cded43798daad88599887f7f21e9148409e0f20fb0483`.
The workflow stays byte-identical to the first attempt; its used label is
neither removed nor reapplied.

## Completion boundary

PR #255 remains draft/unmerged, with old baseline/candidate finding profiles
still genuine failures. Do not claim all CI green. No model acquisition,
construction/forward, paid worker, provider key, owner signature, publication
or inventory promotion occurs. Full non-Apple/source/license/native/operator,
independent real-weight CPU, final clean-HEAD/fresh Apple packets and Apple
CPU/Metal/no-fallback gates remain unfinished; Scaleway stays last.
