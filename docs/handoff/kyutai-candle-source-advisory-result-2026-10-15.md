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

## Third attempt: authenticated actual findings, not a wrapper error

PR #255 head `6a002fe606f3ae73566d8147c2e1f2eec42a1529`,
[run 38704070263](https://github.com/ayutaz/vokra/actions/runs/38704070263),
job `116318999323`, passes the corrected twelve hosted tests, producer/input
authentication and fixed tool/database checks. At 08:17:22 UTC the actual
auditor returns **1 / FAIL_FINDINGS**. The strict workflow fails as intended;
bounded evidence upload succeeds. No ignore, warning waiver or filter was
added to make the result green.

The [original recovered report](kyutai-candle-source-advisory-2026-10-15.json)
is **28,055 bytes**, SHA-256
`1c087752a55dfca61c46efeb567afc9557609b5f0bb1eee299664bee0089e0d1`.
Uploaded artifact `11957314113`, named
`kyutai-candle-source-advisory-38704070263-1`, is **6,334 compressed bytes**.
Manager independently replays report validation, checks raw native JSON
equality and stdout length/hash, all seven original input hashes totaling
2,726,428 bytes, exact producer and tool identities, and unchanged clean
database commit/tree before and after the audit. Raw JSON retains its native
null database commit and separate external Git provenance; 1,329 database
advisories and all 161 lock records are counted. There is no truncation or
timeout, stderr is empty, and the expected no-fetch/no-yanked/no-ignore
command is preserved verbatim.

| Class | Exact package | Fixed RustSec advisory | Recorded patched range |
|---|---|---|---|
| Vulnerability | crossbeam-epoch 0.9.18 | RUSTSEC-2026-0204 | `>=0.9.20` |
| Unsound warning | anyhow 1.0.98 | RUSTSEC-2026-0190 | `>=1.0.103` |
| Unsound warning | rand 0.9.0 | RUSTSEC-2026-0097 | `>=0.9.3, <0.10.0` for this series |
| Unmaintained warning | adler 1.0.2 | RUSTSEC-2025-0056 | None listed; investigate maintained dependency route |
| Unmaintained warning | paste 1.0.15 | RUSTSEC-2024-0436 | None listed; investigate maintained dependency route |

Manager independently reads all five official entries at the exact database
commit, rather than assuming current online notices match the frozen audit:
[crossbeam-epoch](https://raw.githubusercontent.com/RustSec/advisory-db/edb58f5974b83eb8302332821cb6d389777aa351/crates/crossbeam-epoch/RUSTSEC-2026-0204.md),
[anyhow](https://raw.githubusercontent.com/RustSec/advisory-db/edb58f5974b83eb8302332821cb6d389777aa351/crates/anyhow/RUSTSEC-2026-0190.md),
[rand](https://raw.githubusercontent.com/RustSec/advisory-db/edb58f5974b83eb8302332821cb6d389777aa351/crates/rand/RUSTSEC-2026-0097.md),
[adler](https://raw.githubusercontent.com/RustSec/advisory-db/edb58f5974b83eb8302332821cb6d389777aa351/crates/adler/RUSTSEC-2025-0056.md)
and [paste](https://raw.githubusercontent.com/RustSec/advisory-db/edb58f5974b83eb8302332821cb6d389777aa351/crates/paste/RUSTSEC-2024-0436.md).

These findings concern the external, isolated Candle reference closure,
not newly introduced dependencies in Vokra's first-party-only runtime.
The selected 105-package CPU graph remains distinct from this whole-lock
audit. Revised dependencies require a new authenticated source/metadata and
license/native review scope; the failed receipt and original archive facts
must not be restamped or silently reduced to a more convenient denominator.
PR #255 is draft until remediation and fresh acceptance gates succeed.

## Acceptance still required

The version-contract correction passes manager's independent review and
**12 stdlib tests in 4.179 seconds**, including the actual version-output
positive case, wrong prefix/version/prerelease/build/extra-line negatives
and retained error diagnostics. Actionlint and diff checks pass. It
needed a fresh hosted run; the third attempt above now supplies an actual
failed findings report. Remediation of all five findings is the next gate,
not another unchanged audit or an ignore rule. A successful source-only
advisory result, if obtained,
would still not approve the whole license/native/operator closure or prove
independent full-PCM/real-weight CPU parity. Retain all actual failures and
their source identities; do not rewrite a failed receipt as a later success.
Scaleway remains last, after every non-Apple prerequisite and the reviewed
clean HEAD/fresh final transfer packets. No paid worker, local model run,
credential/provider operation or publication is performed here.
