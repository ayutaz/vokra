# Candle applied-lock advisory result — 2026-10-16 JST

## Verdict and exact execution

**ACCEPTED_FIXED_DB_ADVISORY_SCOPE_ONLY.** The genuine pinned registry audit
and conservative same-database source-absent coverage succeed. This supersedes
the not-yet-run status in the [reviewed preparation](kyutai-candle-applied-advisory-preparation-2026-10-16.md),
not the old baseline/candidate failed receipts or any compatibility, license,
owner, model or Apple gate. PR #255 remains draft and unmerged.

- Published source: `04574e71bf9f63c40ad62bf3c351bf65db8e6c51`.
- Base: accepted main `f8d859dc9cf284372ae758c0fa06763d93e4ebdc`.
- [Run 38765578560](https://github.com/ayutaz/vokra/actions/runs/38765578560),
  attempt 1, [job 116527781072](https://github.com/ayutaz/vokra/actions/runs/38765578560/job/116527781072),
  terminal **SUCCESS**, completed 2026-10-15 16:03:56 UTC.
- Actual PR merge checkout:
  `63769cf472dde779c1914d9b6fcc8461d6c8544e`. The decoded log explicitly
  identifies the named published source merged into the named base.
- Native wrapper return code **0**; every job step, including evidence upload,
  succeeds. No paid worker, target-package/model execution or publication occurs.

## Actual hosted tests

The log distinguishes unavailable-fixture checks from actual artifact tests:

- Legacy self-tests: **12 passed**, no skips, 5.572 seconds.
- Applied tests before artifact download: 17 enumerated, **3 passed / 14
  skipped**, 0.001 seconds. Skips are not passes.
- After the exact original application artifact is downloaded: **17 passed**,
  no skips, 0.163 seconds. This includes original eight-file authentication,
  CPU/whole/lock binding, source-absent identity and checksum mutations,
  missing/malformed/symlinked database paths, positive native orchestration,
  post-command path injection, vulnerability failure, lock mutation during
  the command and final provenance, and supplemental deadline rejection.

Local full unittest/spelling refusals and their no-bypass handling remain
recorded in the preparation. Hosted results, not a relabelled local run,
close those test-execution gaps for this source candidate.

## Recovery and independent original-byte comparison

Artifact **11989905461** is
`kyutai-candle-applied-advisory-38765578560-1`, **2,552 compressed bytes**,
bound by GitHub's API to run/head above. API and upload log agree on ZIP
SHA-256 `d4a9991771eba79f2f02450a869bca625a9a231bf9f693f892ffe5b2f422af0b`;
expiry is 2026-10-16 16:03:51 UTC. Root downloads/extracts through the GitHub
CLI and does not claim a separate local ZIP-byte rehash.

Its sole regular, non-symlink, single-link `advisory-report.json` is
**12,919 bytes**, SHA-256
`b740e2e3331fd0c5befd82b3d92e87aa2e995613bf81ce7c2da1e5a2ee825805`.
Original bytes are retained at
`/private/tmp/vokra-candle-applied-audit-045-20261016.wyySLH`, private Git
commit `a9d9bd35f6a3548b3bccee434b7c0d6141a774e5`, clean, with no remote.
This records current retention, not permanent availability of a temporary
directory or availability from a public clone.

Root's separate read-only UV/Python 3.12 JSON/TOML/AST analysis does not
import the audit implementation, run Cargo, or execute target packages. It
independently verifies:

- All eight producer member names, lengths and hashes against both the
  literal reviewed profile and original recovered application bytes;
  **4,333,048 bytes** total. Producer run/head/artifact bindings remain exact.
- Final lock SHA-256 remains
  `21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6`.
  Its partition is **137 registry + 16 modified source-absent parents + 1
  source-absent Moshi workspace**, not 154 native auditor queries.
- Native stdout is **314 bytes**, SHA-256
  `e69700d8f3148910541de1d1d6e74017a79a98288e613b78ee2756829f88dc44`.
  Its parsed JSON equals the embedded report. Stderr is empty; neither
  stream is truncated and the process does not time out.
- Native return code is zero, with **zero vulnerabilities / zero warnings**
  against the fixed **1,329-advisory** database. Ignore/target/severity
  filters are absent and all informational warning classes remain enabled.
  The actual argv retains `--no-fetch --no-yanked --deny warnings`.
- Clean DB Git identity before and after agrees on commit
  `edb58f5974b83eb8302332821cb6d389777aa351`, tree
  `9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63`. Native null DB commit is
  preserved, not fabricated from that external provenance.
- Both supplemental records agree on every one of the **17 exact identities**,
  version, `crates/<name>` path, absent status and parent/workspace role.
  The independently obtained complete fixed Git tree also has no matching
  package path. This is the explicit narrow absence proof required because
  the pinned native auditor skips source-absent records.
- Approval remains null, license review is `NOT_PERFORMED`, model execution
  is `NOT_AUTHORIZED`, upload is `NO_UPLOAD`, and yanked checking is
  `NOT_CHECKED`. A known-advisory result is not a full security guarantee.

The actual binary reports `cargo-audit-audit 0.22.2`; its recorded SHA-256 is
`66f9c8f530950d106c1869ce27ef5a4008bcea53c7716c3ed8117586337ee7d8`.
Install action/release identities match the reviewed fixed contract.

## Legacy findings, ordinary CI and next gate

Fresh legacy jobs at the same source remain honest failures: baseline
`116527782483` passes twelve tests then returns wrapper code 1; candidate
`116527781968` passes twelve tests and, after its own download, all nine
candidate tests, then returns wrapper code 1. Their original locks/findings
are not ignored or retroactively promoted by this changed-input result.

The ordinary-check snapshot has 66 enumerated checks: seven successful,
one skipped, seven running, 49 queued and the two legacy failures above.
This transient snapshot is not an all-green PR or merge authorization.

Next resolve source/license/native/operator and modified-manifest/macro/API
compatibility prerequisites, then independently captured full-PCM reference
and separately authorized real-weight CPU work. The original step-4 scope
still includes all unresolved non-Apple rows and final reviewed clean-HEAD,
fresh Apple packets. No model row/count, frozen owner decision or protected
manifest changes. Scaleway and publication remain later, separately gated.
