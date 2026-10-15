# Candle applied-lock advisory coverage — 2026-10-16 JST

## Scope and authorization

The owner explicitly authorizes the audit wrapper, focused tests and dedicated
GitHub workflow changes, related documentation and the hosted source-only
audit. This supersedes the preceding two-file/offline-test-only hold for this
narrow work. It does not authorize model acquisition/execution, a paid worker,
owner/legal sign-off, Apple provisioning or publication. PR #255 stays draft.

Use the [accepted actual application](kyutai-candle-paste-application-result-2026-10-16.md),
not a newly generated or registry-restamped lock. The producer is run
`38758745134`, attempt 1, head `89fbbc512beac5debbf4807a1dde3f2e8588096e`;
artifact `11985400172` is `kyutai-candle-paste-application-evidence-38758745134-1`
(259,361 compressed bytes). The API still reports it unexpired, with expiry
2026-10-16 15:16:14 UTC. All eight original member sizes and SHA-256 values
remain the immutable input contract recorded in the application result.

## Confirmed coverage defect, not a new audit result

Root resolves the official `cargo-audit/v0.22.2` annotated tag to
`281452c35cf0870969042374110f099a411bc185`. Its workspace lock contains
[`rustsec 0.33.0` and `cargo-lock 11.0.1`](https://github.com/rustsec/rustsec/blob/281452c35cf0870969042374110f099a411bc185/Cargo.lock).
Read-only primary-source inspection
establishes the complete call chain:

- [`Auditor::audit_lockfile` / `audit`](https://github.com/rustsec/rustsec/blob/281452c35cf0870969042374110f099a411bc185/cargo-audit/src/auditor.rs)
  load the original textual lock and call `Report::generate`.
- [`Database::query_vulnerabilities`](https://github.com/rustsec/rustsec/blob/281452c35cf0870969042374110f099a411bc185/rustsec/src/database.rs)
  skips packages whose source is absent or is not the default registry,
  before constructing the package query.
- [`find_warnings`](https://github.com/rustsec/rustsec/blob/281452c35cf0870969042374110f099a411bc185/rustsec/src/report.rs)
  uses that same query function. Its `dependency-count` is merely the length
  of the parsed lock's package vector, not the number of queried packages.

Root independently rehashes and parses the recovered `final.lock` without
running Cargo or any upstream code. SHA-256 remains
`21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6`.
Its **154** records partition as follows:

| Scope | Records | Required coverage |
|---|---:|---|
| Default-registry packages | 137 | Genuine pinned cargo-audit, all warnings denied |
| Modified source-absent parents | 16 | Explicit same-database supplemental check |
| Source-absent Moshi 0.6.4 workspace | 1 | Explicit separate workspace accounting |

The sixteen parents are `gemm`, `gemm-c32`, `gemm-c64`, `gemm-common`,
`gemm-f16`, `gemm-f32` and `gemm-f64`, each at 0.17.1 and 0.18.2, plus
`metal` 0.27.0 and 0.29.0. All seventeen source-absent records also have no
checksum. No other source kind occurs.

The **137 unchanged complete records** in the earlier delta proof and the
**137 registry records** here are different sets: the former includes Moshi,
whereas the latter includes the new pastey registry package. Do not equate
these numerically equal counts.

## Conservative supplemental contract

The fixed database commit remains
`edb58f5974b83eb8302332821cb6d389777aa351`, tree
`9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63`. GitHub's [full recursive response](https://api.github.com/repos/RustSec/advisory-db/git/trees/9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63?recursive=1)
for the exact tree is not truncated and contains 2,315 entries, including
1,309 crate-advisory and 20 Rust-advisory Markdown files. None of the nine
package names above has a `crates/<name>` directory or descendant in that
fixed tree. This is an observed fixed-snapshot fact, not a claim that these
packages are generally vulnerability-free.

The pinned [`Entries::load_file`](https://github.com/rustsec/rustsec/blob/281452c35cf0870969042374110f099a411bc185/rustsec/src/database/entries.rs)
requires the package name inside each advisory to equal its parent directory.
Therefore an absent package directory proves that the same valid database
has no crate advisory for that name, independently of a version-range parser.

The additive profile must authenticate the original eight-file artifact and
the exact 154-record partition, then pair the genuine audit with a conservative
absence proof for **every one of the seventeen source-absent identities**.
Require a real `crates` directory in the authenticated clean database;
fail closed if any expected package path exists as a directory, file or
symlink, including a dangling symlink. Do not silently parse away, ignore or
waive such a new path. It requires explicit additional advisory review.

Both legs must use the same fixed database, verified before and after, and
the original lock must remain byte-identical. Preserve native RustSec JSON,
warning denial, empty ignore/filter settings, bounded output and the shared
480-second deadline. A successful 154-record native report without complete
supplemental coverage is an error, not PASS. No shadow lock, forged registry
source or fabricated checksum is part of this design.

## Acceptance and remaining boundaries

Root reviews the additive profile, source-free coverage helper, original lock
rehashes after the native command and final provenance, and the focused fault
tests. The sixteen-parent/source-root partition is explicit; the original
baseline/candidate contracts and failed receipts remain intact. The new
workflow includes the old workflow file required by legacy self-tests.

Reviewed SHA-256 values:

- `audit_lock.py`: `995c9969fecf4bcbb5857c0d2758a8d7b314b52554e6d4171a78cd2a697a7375`.
- `test_applied_audit_lock.py`: `e3a5e221be3678fac00fb4762148323fb8d3e8ba4320d5635f031bdbb2882fce`.
- `kyutai-candle-applied-advisory.yml`: `ffbce2c9d66e01eff8e1e52a6c85f0e0bc8eee9b2de57c1e37656e3714631d08`.

Root's separate read-only AST/hash comparison authenticates all eight literal
profile member hashes against the original recovered bytes (4,333,048 bytes),
without importing the audit module. The private original evidence Git tree
remains clean. Implementer syntax checks and direct evidence validation pass;
they are not substituted for root's or hosted full unittest execution.

Root's first local legacy regression invocation is refused by the local-model
PreToolUse guard for `test_candidate_audit_lock.py`. It is not retried through
a different runner, flag or path, and no hook rule is changed. The owner has
explicitly authorized hosted CI, so test execution moves to that permitted
remote environment; no current local unittest result is inferred from the
aborted parallel invocation. Read-only review identifies the actual source
receipt's `files` as a 23-entry dictionary, not a list, before hosted execution.
A subsequent spelling command naming the new test/script/workflow paths is
also refused by that guard and is not repackaged or retried. The earlier
documentation-only spelling result is not a code-spelling result.

Local static checks pass: the eight NVIDIA compliance-scanner regressions,
actionlint for the new workflow, workflow hygiene (55 workflows / 36 cron
entries), documentation references, community-document checks and spelling.
The first workflow-hygiene invocation hits the inaccessible default uv cache;
using the established writable `/private/tmp/vokra-uv-check` cache succeeds.
Rust/Cargo/test inputs remain identical to recorded green VAST `4263f416`;
that is reused input evidence, not a new whole-HEAD VAST result.

Implementation and independent regression review precede the hosted run.
The actual applied-lock audit has **not yet run** at this preparation stage;
the GitHub tree readback is not its execution receipt. Existing baseline and
candidate profiles and their failed findings remain unchanged.

Even a future successful result establishes only the recorded known-advisory
scope at the fixed database. It does not approve modified manifests/macros,
native/API compatibility, licenses, operator execution, independent full-PCM
reference, real-weight CPU parity or Apple CPU/Metal/no-fallback behavior.
The original step-4 objective still includes all unresolved non-Apple
prerequisites, authorized real-weight CPU work and final clean-HEAD/fresh
Apple packets. Scaleway and publication remain later, separately gated work.
