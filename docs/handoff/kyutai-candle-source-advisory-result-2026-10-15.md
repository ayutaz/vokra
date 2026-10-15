# Candle source-only advisory attempts — 2026-10-15

This records actual hosted attempts of the [reviewed advisory preparation](kyutai-candle-source-advisory-preparation-2026-10-15.md),
not a successful vulnerability verdict. Target crate/model execution,
license/operator approval, real-weight CPU parity and final Apple packets
remain outside these attempts. Existing null approval and `NO_UPLOAD` remain.

## First attempt: causal test stopped before audit

PR #255 head `12793120fee47e7ca68de9c9aa61b8f1a8759b14`,
[run 38702495030](https://github.com/ayutaz/vokra/actions/runs/38702495030),
job `116313816116`, fails its self-test at 08:04:27 UTC. The injected I/O
mock changes the same global `os.read` used by Python's subprocess bootstrap,
so it raises before the wrapper owns the successfully started child. The
actual log retains a running-subprocess warning and a changing descendant
heartbeat. This independently confirms the manager's pre-audit diagnosis.

Artifact download, database checkout, auditor installation and actual audit
are skipped. The always-upload step separately fails because an audit report
does not yet exist. Neither failure is a RustSec finding. The correction
captures only stdout/stderr descriptors after successful `Popen`, delegates
bootstrap reads and preserves both causal descendant scenarios. All eleven
corrected tests are independently replayed on the Mac, without a Linux skip.

## Second attempt: actual version-output contract mismatch

PR #255 head `b2aba5900eef477b81eef697f42e0a6ad0e8ed00`,
[run 38703182444](https://github.com/ayutaz/vokra/actions/runs/38703182444),
job `116316066260`, passes all eleven hosted self-tests and the exact
producer-artifact download, fixed database checkout and checksum-verifying
prebuilt auditor installation. At 08:07:30 UTC the wrapper exits **2 / ERROR**
at version verification, before producer/member authentication or the actual
advisory command. The report's `producer`, `artifact` and `audit` are null.
Downloading the input in the workflow is not a substitute for its validator.

The installer log actually runs `cargo-audit audit --version` and records:

```text
cargo-audit-audit 0.22.2
```

The wrapper expected a different command-name prefix. This is a consumer
contract defect, not evidence that the installed version is wrong or that
the lock has vulnerabilities. The pinned upstream subcommand itself derives
its version through Clap. Preserve the exact version/action/database pins;
correct the full-line comparison and retain bounded version stdout/stderr,
return code and truncation flags on future errors instead of weakening the
version gate or rerunning the unchanged candidate.

The uploaded artifact is `11957531271`, named
`kyutai-candle-source-advisory-38703182444-1`, compressed size **752 bytes**.
Manager recovers its only small JSON report through the official CLI. The
report is **1,019 bytes**, SHA-256
`3966069f96c67f25dfdc49258a276e240a77e13180d12baa4fd44be4c389678a`.
It explicitly preserves `ERROR`, `NOT_PERFORMED` license review,
`NOT_AUTHORIZED` model execution, null approval and `NO_UPLOAD`. No audit
PASS or vulnerability count can be derived from an absent audit report.

## Acceptance still required

The version-contract correction passes manager's independent review and
**12 stdlib tests in 4.179 seconds**, including the actual version-output
positive case, wrong prefix/version/prerelease/build/extra-line negatives
and retained error diagnostics. Actionlint and diff checks pass. It still
needs a fresh hosted run. A successful source-only advisory result, if obtained,
would still not approve the whole license/native/operator closure or prove
independent full-PCM/real-weight CPU parity. Retain all actual failures and
their source identities; do not rewrite a failed receipt as a later success.
Scaleway remains last, after every non-Apple prerequisite and the reviewed
clean HEAD/fresh final transfer packets. No paid worker, local model run,
credential/provider operation or publication is performed here.
