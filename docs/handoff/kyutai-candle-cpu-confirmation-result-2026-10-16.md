# Candle model-free CPU/API confirmation — 2026-10-16

## Technical verdict and exact scope

[Run 38885265971](https://github.com/ayutaz/vokra/actions/runs/38885265971),
attempt 1 / [job 116921359664](https://github.com/ayutaz/vokra/actions/runs/38885265971/job/116921359664),
executes exact reviewed head
`9b58b64f120c9c1f306c1f967e696c52b1038e9f`.
The job starts 10:12:10 UTC and finishes 10:13:44 UTC with success.
All **17 Linux fixture tests pass, no skips**, in 8.652 seconds.
The library and uncalled API-signature probe both return zero.

Manager accepts this bounded model-free CPU library/API compatibility
evidence after independent recovery and verification. It closes stage 1 of
the [six-stage objective](goal6-execution-readback-2026-10-16.md), not
real-weight CPU parity, complete PCM/reference behavior, model/operator/
license decisions, Apple/Metal, final packets or publication.
No model is constructed or executed.

The original [first failed run](kyutai-candle-cpu-compatibility-result-2026-10-16.md)
remains failed with its original receipt. Exactly one new confirmation label
is applied; no label reapplication, workflow retry or third actual job occurs.

## Original evidence and independent checks

Artifact `12038834001`,
`kyutai-candle-cpu-compatibility-38885265971-1`, has reported compressed
size 56,214 bytes and API ZIP digest
`sha256:015887f1dee41c58c2024683eb1a8d7a28e9ed9bfdeae394c07ba6a11563a0ef`.
The ZIP digest is API-reported, not independently reproduced from the
extracted download. Manager recovers thirteen original JSON/text files:
the 48,177-byte receipt and twelve original command streams. A byte-identical
[receipt is retained here](kyutai-candle-cpu-confirmation-receipt-2026-10-16.json)
at SHA-256
`4b668d3052fcb90a67bedc565ba1ea0374ee9b5dfff5363b95d92cfcc943daed`.
Temporary private recovery paths are not represented as permanent storage
of the twelve native streams.

Manager independently rehashes all twelve streams against the original
command summaries. Each of six completed commands returns zero with
`timed_out=false` and `truncated=false`. A separate planned
source-application argv entry is not counted as a second execution.

| Actual stage | rc | Seconds |
|---|---:|---:|
| Exact checkout HEAD | 0 | 0.002 |
| Authenticated frozen source application | 0 | 11.786 |
| Isolated namespace/toolchain preflight | 0 | 0.084 |
| Full locked/offline CPU library build | 0 | 44.981 |
| Fresh native CPU metadata | 0 | 0.319 |
| Uncalled API-signature compilation | 0 | 0.048 |

Checker elapsed time is 58.586 seconds. Full Cargo build remains
`--lib --locked --offline --no-default-features --target x86_64-unknown-linux-gnu`,
under a distinct network namespace and no-new-privs. Native build JSON
contains one successful build-finished record and 122 compiler-artifact
records, each bound to the actual fresh metadata IDs. Manager separately
checks all fourteen derived CPU gemm identities and all ten proc macros,
including pastey and bytemuck_derive in the host dependency directory.
The API argv includes both target and host dependency directories.
Its 262-byte stderr is only the expected warning that the signature-check
function is uncalled; no warning suppression is added.

Manager independently compares all package identity/source tuples,
selected features, dependency identities, typed edge kind/target fields
and workspace membership against the authenticated original overlay:
105 packages/nodes = registry90 + derived path14 + Moshi1, unchanged.
Relocated path IDs are matched by their authenticated name/version/source
identity, not by unrelated raw metadata hashes. One raw metadata field
differs: original `resolve.root=null`, while direct package-manifest
metadata names its sole Moshi workspace member as root. Manager explicitly
checks that member is path Moshi 0.6.4; no package/node/edge/feature differs.
The wrapper's retained canonical graph fingerprint is
`99364d685eb1898f2ccf442c68fb51d70f0b5e42f90c04dd0171cb07933be51b`.
This is an independently checked graph comparison, not a claim that every
raw metadata byte is identical or that this review separately recomputed
the wrapper's hash encoding.

Final lock remains
`21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6`.
The receipt's projection/vendor and all ninety selected registry-source
snapshots are equal before/after build. Source application proofs retain
308 checked members / 292 non-manifest members and 23 source files /
21 unchanged source files. Manager verifies the original report fields,
not a recovered complete remote filesystem.

## Remaining boundary

The original receipt intentionally retains
`CPU_COMPATIBILITY_REVIEW_PENDING`, null approval,
`model_execution=NOT_AUTHORIZED`, `source_only=true` and `NO_UPLOAD`.
Manager's technical acceptance is this separate record, not a restamped
receipt, owner/legal signature or VAST real-weight execution authority.
Ordinary exact-head CI and PR acceptance remain stage 2. All non-Apple
row prerequisites and real-weight CPU are still required before final clean
HEAD/packet and Scaleway Apple CPU/Metal/no-fallback.
