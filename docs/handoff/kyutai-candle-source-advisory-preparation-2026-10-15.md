# Candle source-only advisory preparation — 2026-10-15

> Actual result supersession: the third attempt at `6a002fe6` now produces
> an authenticated `FAIL_FINDINGS` report with one vulnerability/four warnings.
> See the [retained result and remediation boundary](kyutai-candle-source-advisory-result-2026-10-15.md).
> Source PR #254 is separately accepted as main `7701cc67`; audit PR #255
> remains draft. Earlier preparation/pending statements below are historical.

> Subsequent actual attempts are retained in the [hosted result record](kyutai-candle-source-advisory-result-2026-10-15.md).
> The first stops in a causal test and the second at version verification;
> neither reaches a completed advisory audit. The corrected candidate below
> still requires fresh hosted evidence. Earlier preparation-time non-execution
> statements are not later execution verdicts.

This prepares a separate advisory leg at accepted main
`5ceb432545cc1d9bdf2ff5c5ff289d683276539a`. It does not repeat the authenticated
Candle source/metadata capture and does not execute a target crate or model.
The accepted [104-archive facts](kyutai-candle-primary-archive-facts-2026-10-15.md)
and the subsequent private 14-archive supplement are separate evidence. The
supplement is being reviewed in the dependency/documentation branch; its
presence is not an execution approval.

## Exact inputs and security boundary

The new workflow reuses artifact `11951567227`, producer run `38690241603`,
attempt 1, head `b111f377bfbfc2f8d079f10791d61befb2d3ff5d`. It checks the
successful producer, exact repository/ID/name/size/expiry and all seven
bounded regular member hashes before using the patched lock. Manager
independently runs the production artifact validator against the actual
recovered seven-member packet: **2,726,428 bytes**, all hashes match.

Audit scope is the **161-record patched lock**, SHA-256
`303c49433187d5938d0f2a27d3f5ded9eb094388dd67e6702a0a8b7558eaec01`.
The selected **105-package CPU graph** is separately recorded, not relabeled
as the audited universe. Neither the zero-dependency root runtime audit nor
an integrations audit covers this lock.

Tool release is cargo-audit 0.22.2 at
`281452c35cf0870969042374110f099a411bc185`, installed through the pinned
checksum-verifying action without source-build fallback. Database commit is
`edb58f5974b83eb8302332821cb6d389777aa351`, tree
`9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63`, observed from official HEAD at
2026-10-15; the earlier `6aed8466` observation remains historical.

The explicit lock/database arguments, no-fetch and no-yanked flags prevent
target dependency resolution/acquisition. The auditor runs only on hosted
Linux x86_64, from an empty directory and task-specific empty `CARGO_HOME`;
the runner's `HOME` is unchanged. Child commands do not inherit the GitHub
token or user Git overrides. Findings, tool failures, malformed reports,
timeouts and bounded/truncated output remain distinct, with no ignored
advisories or fabricated success. Yanked-package checking is explicitly
`NOT_CHECKED`, not represented as a complete security guarantee.

## Pinned-source correction: native DB commit is null

The [pinned auditor](https://raw.githubusercontent.com/rustsec/rustsec/281452c35cf0870969042374110f099a411bc185/cargo-audit/src/auditor.rs)
calls `Database::open` when `--no-fetch` is present. The
[pinned implementation](https://raw.githubusercontent.com/rustsec/rustsec/281452c35cf0870969042374110f099a411bc185/rustsec/src/database.rs)
sets `latest_commit: None` in `open`; only `load_from_repo` fills it.
Therefore this release's raw report must retain null `database.last-commit`.
The manager rejects the initial latest-doc inference that a valid checkout
would populate it automatically. The wrapper instead proves the exact Git
commit/tree and clean status before and after the audit, preserving the raw
null rather than inserting a hash into the upstream JSON.

## Review and execution status

The manager independently passes the corrected eleven stdlib tests,
including blocking API reads, process-group deadline cleanup, exact version,
artifact mutation, report/counter/truncation classification, config isolation
and workflow pins, initially in 2.820 seconds, and actionlint passes. The exceptional
I/O/interruption path now terminates its owned process group before
unwinding and reaps its direct child. Before the actual audit, further review
finds that the Linux-only mock patched the same global `os.read` used by
Python's `Popen` bootstrap. It could fail before entering the intended pipe
loop, so the initial Mac result does not prove that causal path.

The correction delegates bootstrap/non-target reads to the original function
and captures target pipe descriptors only after successful process creation.
Both the exited-parent/descendant-held-pipe timeout case and the injected-I/O
case retain a forked SIGTERM-ignoring descendant, using heartbeat liveness
rather than zombie-PID absence. Neither case is skipped on the Mac. Selector
creation/registration is inside the cleanup-protected lifecycle. Manager
independently reruns all eleven tests in **4.167 seconds**, including both
causal paths, and actionlint/diff checks pass. Actual hosted execution and the
advisory verdict still require separate evidence.

After the actual hosted version-contract failure, the verifier requires the
observed exact `cargo-audit-audit 0.22.2` line and retains bounded mismatch
diagnostics. Manager independently passes all **12 tests in 4.179 seconds**,
including wrong-prefix/version, prerelease/build/extra-line, nonzero/truncated
output and orchestration evidence regressions. Tool/action/database pins,
audit scope and fail-closed findings policy remain unchanged.

Final reviewed workflow/wrapper/test SHA-256 values respectively are
`cf48f1c684bc517a64494ed0ae1b8b830c20cae11f7aada79558b87e10b6939b`,
`bb2a5afeee95c27a788093dcd85f54fc6be2cfae14c742fbae3eaa0d8dca176d`
and `b2f6876f6b0785c38cd630dd47fd877f15b9eaee0a8ddd547af2bc0eaf80b0ab`.
These are source and synthetic-test identities, not an advisory verdict.

No actual hosted advisory run, cargo-audit installation/execution, target
package import/build/proc-macro, model/config/tokenizer acquisition, paid
worker or publication has occurred in this preparation. License review,
owner/operator decisions, full-PCM reference, real-weight CPU and final
clean-HEAD Apple packets remain separate prerequisites. Every resulting
receipt keeps null approval, `NOT_AUTHORIZED` model execution and `NO_UPLOAD`.
