# Candle source-only advisory preparation — 2026-10-15

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

The manager independently passes the final eleven corrected stdlib tests,
including blocking API reads, process-group deadline cleanup, exact version,
artifact mutation, report/counter/truncation classification, config isolation
and workflow pins, in 2.820 seconds, and actionlint passes. The exceptional
I/O/interruption path now terminates its owned process group before
unwinding and reaps its direct child. The additional injected-I/O descendant
heartbeat assertion is Linux-only and has not run on the maintainer Mac;
the actual hosted self-test step must prove that branch before audit.

Reviewed workflow/wrapper/test SHA-256 values respectively are
`cf48f1c684bc517a64494ed0ae1b8b830c20cae11f7aada79558b87e10b6939b`,
`68e4ee25064b58566d73960523e976bb8c70a9b008933b2faa07356e1637a4b5`
and `a7ea44920435bb68dd437133afa30ee0bee6c6518622d46aa72878bfc3b7b5b0`.
These are source and synthetic-test identities, not an advisory verdict.

No actual hosted advisory run, cargo-audit installation/execution, target
package import/build/proc-macro, model/config/tokenizer acquisition, paid
worker or publication has occurred in this preparation. License review,
owner/operator decisions, full-PCM reference, real-weight CPU and final
clean-HEAD Apple packets remain separate prerequisites. Every resulting
receipt keeps null approval, `NOT_AUTHORIZED` model execution and `NO_UPLOAD`.
