# Candle candidate strict advisory result — 2026-10-15

The actual candidate audit completes with **`FAIL_FINDINGS`, return code 1**,
not an auditor/bootstrap error and not PASS. At the same fixed RustSec
database as the original 161-record audit, the authenticated 154-record
candidate has zero reported vulnerabilities and one unmaintained warning:
`paste 1.0.15`, `RUSTSEC-2024-0436`. The strict `--deny warnings` policy
correctly keeps this run red. This closes four specific prior findings;
it does not establish security against later databases or model readiness.

## Actual execution identity

- PR head: `09d2003b6abae3aef7089748a0572fe218ed3eae`; base main:
  `7701cc67414d9cb84e4fe3cd6cf228a10bb8b792`.
- GitHub executes the PR merge checkout
  `d7e7a86f552d0e80ed00b8330eefc15056407e8f`, not a claimed direct-head checkout.
- [Run 38713528857](https://github.com/ayutaz/vokra/actions/runs/38713528857),
  attempt 1; job `116350535383`; actual wrapper return code 1.
- Artifact `11961234042`,
  `kyutai-candle-candidate-advisory-38713528857-1`, compressed 3,400 bytes.
  Uploaded archive digest from the job log:
  `902aef1ea7ffd94424a9d6d0504b5d06d521b744768b2ee0200ad8a232d27961`.
- The [recovered report](kyutai-candle-candidate-advisory-2026-10-15.json)
  is exactly **10,037 bytes**, SHA-256
  `c67367f549c76e1c9e338738edd0c834da96a2b91ac1992a85f71436f9f5d6ea`.
  Its original terminal newline is preserved; this is not a restamped report.
- The embedded native stdout is 1,732 bytes, SHA-256
  `ed99fcfdf31e0a245627ded5a372f94f18ce15893037338cfe5e22dd4c31831e`.
  Stderr is empty. Neither stream is truncated and the auditor does not time out.

## Independent recovered-input and report acceptance

Manager rehashes all thirteen producer files against the report's exact
member names, lengths and hashes: 7,040,218 bytes in total. The producer
remains successful run `38711006458`, attempt 1, head
`27a2841384220c078aa37ff93ea8f9fc8a18594f`, artifact `11961100426`.
The [candidate lock](kyutai-candle-candidate-2026-10-15.lock) is 38,252 bytes,
SHA-256 `d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8`.
The report audits all **154 whole-lock records**, not the 105-package CPU
subset. The [resolution acceptance](kyutai-candle-pin-remediation-2026-10-15.md#corrected-actual-resolution-and-independent-acceptance)
retains the complete lock-delta proof separately.

The native stdout parses to the exact embedded report. Its counters,
warning identities, empty ignore/target/severity filters, all informational
warning classes, nonzero return code and `FAIL_FINDINGS` classification
agree. The auditor is cargo-audit 0.22.2, binary SHA-256
`66f9c8f530950d106c1869ce27ef5a4008bcea53c7716c3ed8117586337ee7d8`.
Before and after the run the database is clean at commit
`edb58f5974b83eb8302332821cb6d389777aa351`, tree
`9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63`, with 1,329 advisories.
Native `database.last-commit` remains explicit null; external Git evidence
authenticates the fixed database without rewriting native JSON.

Hosted execution passes all twelve legacy tests in 5.544 seconds. The
pre-download candidate suite explicitly skips three unavailable-artifact
cases; after the exact artifact download, all nine candidate tests pass
in 0.057 seconds with no skips. Those tests are preparation/contract
evidence, not a substitute for the actual nonzero audit result.

## Original-to-candidate comparison at the same database

| Original finding | Candidate observation |
|---|---|
| crossbeam-epoch / RUSTSEC-2026-0204 | No longer reported after precise update. |
| anyhow / RUSTSEC-2026-0190 | No longer reported after precise update. |
| rand / RUSTSEC-2026-0097 | No longer reported after precise update. |
| adler / RUSTSEC-2025-0056 | Package removed with the old anyhow/backtrace chain; no longer reported. |
| paste / RUSTSEC-2024-0436 | Still present and reported; no waiver. |

The [old five-finding receipt](kyutai-candle-source-advisory-2026-10-15.json)
and failed diagnostic remain unchanged. No unchanged retry is needed:
the next action is genuine maintained-replacement compatibility/source work.
The [pastey primary-source facts](kyutai-candle-pastey-source-readback-2026-10-15.md)
do not prove equivalence. Fourteen selected CPU parent packages plus two
whole-lock-only Metal packages require an exact dependency route; a root-only
alias does not rewrite their transitive dependencies. Do not hide the
warning with an ignore, renamed fake legacy crate, target filter or reduced
audit universe.

## Full-goal boundary

[PR #255](https://github.com/ayutaz/vokra/pull/255) stays draft. License/native/
operator decisions, independent full-PCM reference, real-weight CPU parity,
all-row non-Apple closure and final clean-HEAD/fresh Apple packets remain
incomplete. This result does not advance a model inventory row or owner
signature. `approval` remains null, `model_execution` is `NOT_AUTHORIZED`,
`license_review` is `NOT_PERFORMED`, and upload is `NO_UPLOAD`.

Recovery and independent checks use only small source/audit evidence. No new
paid worker, model/config/tokenizer acquisition, local model execution,
Scaleway allocation or public artifact update occurs. Continue the original
step-4 objective; do not narrow completion to this dependency audit.
