# Documentation refresh — 2026-10-13

## Current facts and source boundary

Local and GitHub `main` both resolve to
`7d38061c3dcb56f6a16bcf4b8085d966b3cdab09`. This is a documentation-only
refresh over a dirty worktree, not a new clean release or accepted code HEAD.
The [2026-10-08 audit](documentation-refresh-2026-10-08.md) remains dated
evidence; the present record supersedes its current-status pointers only.

A fresh metadata/card-only `tools/audit/hf_mac_coverage.py --workers 6` run
returned exit 0: **194 public repositories / 193 GGUF-bearing / 198 files**;
CPU `full=136`, `partial=43`, `no-runtime-binder=14`, `not-artifact=1`;
Metal `full=136`, `blocked-by-cpu=57`, `not-artifact=1`.
The **58 unresolved public rows** and **136 code/artifact-full rows** are
not a full-catalog real-weight or Apple hardware verdict. No model bytes were
downloaded. The frozen 63-row owner packet and literal M5 checklist counts
**53 checked / 29 unchecked** are different denominators and are unchanged.

Fresh all-pages GitHub reads report **199 open Dependabot alerts**:
183 name a patched version / 16 do not; severity critical 3, high 67,
medium 74, low 55. There are **8 open Code Scanning findings**: four Scorecard
(`CIIBestPracticesID`, `CodeReviewID`, `SASTID`, `VulnerabilitiesID`) and four
CodeQL `rust/log-injection` alerts #68–#71 at
`integrations/vokra-server/tests/wyoming_compat.rs`. Their test location does
not establish a false positive or runtime exploitability; triage remains.
No alert was dismissed or changed by this refresh.

Main protection retains 16 required context/application pairs, strict
up-to-date checking, admin enforcement, linear history and conversation
resolution. These settings were read, not changed.

| Open PR | Head | State at readback |
|---|---|---|
| [#229](https://github.com/ayutaz/vokra/pull/229) | `9db7d7b512888d3ea9834b10aaca767881309ae1` | Non-draft; CLEAN/MERGEABLE; 74 SUCCESS / 3 SKIPPED; no pending/failure |
| [#198](https://github.com/ayutaz/vokra/pull/198) | `5614a92c1938a0cee44ee333bec2b8cbbaa7d969` | Draft; owner/legal |
| [#191](https://github.com/ayutaz/vokra/pull/191) | `63cadf3cf356e703175fe12696e7691fc8ac5af0` | Draft; Kyutai DSM PCM |
| [#182](https://github.com/ayutaz/vokra/pull/182) | `a3afc1ecc38db67434c564bbea424e0366e0cdd9` | Draft; Realtime consolidation |
| [#169](https://github.com/ayutaz/vokra/pull/169) | `b3e9d5c1cd2c75bcc0b2cbe81ce8b115f00e3690` | Draft; CPU reference closure |
| [#152](https://github.com/ayutaz/vokra/pull/152) | `dcb24a0e3d979d31fe838356da1944f8f2087fc0` | Draft; XCodec2 gates |

There are six open PRs, five drafts. Draft CI totals are not freshly asserted
here. PR #199 was merged at `2026-10-08T08:36:36Z`, merge commit
`edc2ab5404f1b9d23dab121f9b4d7e1657ea9db9`, reviewed head
`cf36b448a63105fc8d97147982212cabf08485b5`. PRs #220–#225 are merged;
#219/#226/#227/#228 are CLOSED but **not individually merged**. The current
release/tag API still reports one release and one tag, `v0.3.0`, published
2026-09-20 with 18 assets. A merged packaging PR is not proof that those
published assets were replaced or that six-platform distribution is complete.

## New execution evidence and remaining work

The [2026-10-13 VibeVoice static closure audit](vibevoice-realtime-static-closure-audit-2026-10-13.md)
records the single static-only VAST run, exact seven private inputs, 41-wheel
findings, manager validation, worker/data destruction and original local
guard restoration. Controller PASS coexists with
`OWNER_REVIEW_REQUIRED_NO_UPLOAD`; it is not main-environment runtime,
real-weight, Apple, license, owner or publication approval.

The next sequence remains exact dependency/license/native/upstream/owner
resolution → bounded approved compatibility work → VAST real-weight
conversion/independent reference/CPU parity → final clean HEAD/packet →
Scaleway Apple CPU/Metal/no-fallback → follow-up results → separately
approved publication. External waits may be skipped for other independent
rows but never marked complete. There is no reason to repeat this same
static paid run without a new scope or changed inputs.

## Coverage ledger and preserved history

The starting tracked inventory contains **328 Markdown/MDX files**. This is
an inventory count, not a claim of new line-by-line review of every unrelated
historical record. The related campaign ledgers and current source/checker
boundaries were inspected; stale baseline/PR/security statements were
searched across the tracked documentation. Repository-wide static gates
cover other documents within their implemented coverage, not runtime truth.

| Related surface | Disposition |
|---|---|
| Root README EN/JA; documentation index | Current snapshot, latest audit links and incomplete-catalog boundary updated |
| API reference and backend guide EN/JA | Metadata date/HEAD refreshed; source/API verification distinguished from metadata-only readback |
| VibeVoice reference README | Private static inputs distinguished from main; no runtime or upload promotion |
| Pre-Scaleway ledger, execution plan, residual and full-coverage ledgers | New supersession pointers; all old per-row results, packet hashes and failures retained |
| Owner packet and owner/legal decision record | Scope-bound approvals preserved; static audit grants no new decision |
| Apple results | Existing bounded hardware evidence preserved; no new hardware pass implied |
| Catalog/security completion and security remediation | Current counts, new CodeQL triage and remaining gates updated |
| PR review plan and step-3 prerequisite evidence | Accepted/closed dispositions and present queue separated from dated snapshots |
| License audit and M5 checklist | Static evidence gaps recorded without changing sign-offs, model status or checkboxes |
| Desktop release, platform matrix and release-preparation record | #199 acceptance recorded without inventing later native assembly or publication evidence |
| ABI changelog and M5-13 handoff | Current routing added; old ABI/source test evidence and advisory/owner gate retained |
| Quarterly-review index, metrics runbook, Go/No-go and DoD templates | Current release/main pointer; blank owner verdicts remain blank |
| CONTRIBUTING, security/legal policy, install/architecture/migration/tutorial and unrelated dated records | No new behavior/policy or hardware evidence from this task; existing facts retained and static references checked |

Ignored local planning files are not promoted to tracked public documents.
The three pre-existing protected-path repair files remain untouched by the
documentation edits and uncommitted; the restored local-model guard retains
its original SHA256. The protected CosyVoice2 license manifest is excluded
from inspection and all edits. No commit, push, PR mutation/merge, model
execution/download, new VAST/Scaleway allocation or publication occurs in
this documentation task.

## Verification boundary

Integration checks against the reviewed documentation diff passed as follows.
The update covers **31 existing documents and two new dated records**.
Static links, examples, citations and hook regressions
do not execute the documented models or establish numerical parity. Newly
created unstaged documents require separate link coverage because some
repository checkers enumerate only tracked files. Failed DNS/network attempts
were not counted as successful reads or zero-alert results; the fresh counts
above come from terminal exit-0 readbacks.

| Check | Actual result |
|---|---|
| `tools/docs/check_doc_examples.py --self-test` and default | Exit 0; default processes 144 blocks across 26 documents, with 30 Tier C blocks explicitly deferred/not verified |
| `scripts/check-doc-references.sh --self-test` and default, through UV/Python 3.12 | 17 parser cases; 110 glossary IDs, 33 architecture anchors, 12 hierarchy documents and six EN/JA heading pairs pass |
| `scripts/check-runbook-path-citations.sh` through UV | 1,734 anchored citations across 132 runbooks pass, including explicitly marked missing/local paths |
| `scripts/check-community-docs.sh` through UV | All six legs pass, including funnel links and twin pairs |
| `scripts/check-parity-sidecar-citations.sh` through UV | 468 citations across 1,238 source files pass within the checker's marked-path boundary |
| `scripts/check-owner-checklist-drift.sh` | 28 claim/evidence anchors pass; this checker does not decide M5 owner checkboxes |
| `scripts/check-platform-support.sh` | 53 anchors pass; not a hardware verification |
| `scripts/check-abi-changelog.sh` | Changed C symbols and 124 converter prefixes have rows; historical baseline delta remains advisory, no snapshot advanced |
| `scripts/check-workflow-hygiene.sh` | 48 workflows / 36 crons pass |
| `scripts/check-codex-hooks.sh` | PASS: eight shell files, schema/targets, syntax, self-tests and diff hygiene |
| `scripts/check-protected-path-metadata.sh` | Focused 21-case regression PASS; no protected model manifest inspected |
| Added/new Markdown relative-link path audit, inline standard-library Python through UV | All 62 added relative targets across the 31 modified and two new documents resolve; external URLs/fragments are outside this additional path-only check |
| `git diff --check`, excluding the protected manifest | PASS |

The first standalone workflow-hygiene attempt could not initialize the default
UV cache under the filesystem sandbox. It was rerun with the existing writable
temporary `UV_CACHE_DIR` and `UV_OFFLINE=1`; the actual terminal result is PASS.
No safety hook or repository rule was disabled to resolve this environment
issue. Negative-fixture FAIL messages printed by the example self-test are
expected rejection cases; its terminal exit is 0.

Workspace/model Cargo, model/API execution, native build/device tests and
new paid cloud jobs were deliberately not run. This static documentation
verification does not claim those gates passed.
