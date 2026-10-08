# PR review and disposition plan — 2026-10-08

> **2026-10-08 12:51:46 UTC disposition readback:** the frozen 20-PR batch
> now has four actual merges, eleven incorporated closures and five retained
> OPEN/Draft entries. Accepted main is `f5902db5cc164a292d0095a30a1151cd075eee82`;
> the maintainer checkout is clean and synchronized. The
> [final disposition audit](#final-frozen-scope-disposition-audit) supersedes
> earlier pending observations without rewriting their dated evidence.
> Publication and acceptance of this management record remain the last
> delivery step; model/legal/Apple/publication gates are not closed by PR cleanup.

## Scope and baseline

This is the management plan for the owner's request to handle the normal PRs
and classify/advance the existing Draft PRs before new model work. The frozen
starting scope is 20 open PRs: five normal PRs and 15 Drafts. Later bot arrivals
are a separate batch, not a reason to expand this batch indefinitely.

Starting `main`: `0163cea79cd2f6dcad302fb2bb13178799951367`. The maintainer
checkout is clean. The earlier #201–#206 batch has been handled: #201, #202 and
#206 merged; #203–#205 were incorporated through #202 and closed, not separately
merged. Unmerged PR contents are not part of this main baseline.

This plan does not authorize model execution, an owner/legal sign-off, artifact
publication or Scaleway allocation. The [current documentation index](../README.md),
[remaining-model ledger](mac-pre-scaleway-remaining-tasks-2026-09-05.md),
[execution plan](mac-cpu-metal-execution-plan-2026-09-07.md) and
[owner disposition packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md)
retain their distinct evidence boundaries. The recorded 136 code/artifact-full
and 58 unresolved rows are not a whole-catalog Apple hardware verdict.

## 1. Normal PR acceptance

| PR | Starting issue | Action and acceptance condition |
|---|---|---|
| #207 | NanoCodec Hydra update; behind main; documentation-links failed | Review together with #209, including contextual marker reachability. Integrate the reviewed updates, then require fresh current-main CI. |
| #208 | Pyannote diarization Mako update; behind main | Check exact package/artifact/license changes and contextual marker equivalence; integrate only the reviewed scope and obtain fresh CI. |
| #209 | NanoCodec Werkzeug update; current checks have no failure | Review the collateral marker normalization, not just the named version. Combine with #207 only if the integrated graph is verified. |
| #210 | NeuCodec multidict update; current checks have no failure | Review all changed artifact identities, package metadata and primary license bytes; include in the reviewed batch if independent. |
| #199 | Desktop packaging; main conflict and old link failure | Preserve both dated support-matrix notes, verify native packaging gates, push the reconciliation and require fresh required CI plus Desktop release completeness. |

Green checks on an old HEAD are not evidence for an updated candidate. Normal
protected merges must use the reviewed exact head; no admin merge, protection
change, global HTTP403 acceptance or blind identical rerun is part of this plan.

### Actual #199 reconciliation

Reviewed/pushed head: `cf36b448a63105fc8d97147982212cabf08485b5`, merging the
starting main into old PR head `8b95d3709107715ad8ff5f586ab04e02f13a21bd`.
The only conflict was the support-matrix note; both records are retained.
The manager independently passed desktop unit tests 16/16, handoff oracle
24/24 and compliance-scanner regression 8/8. Rust/Cargo/build inputs and the
release-tool tree match the previously accepted VAST source `3468a48c`;
this is source identity plus dated remote evidence, not a fresh VAST execution.
The local deep Cargo hook was replaced by that corresponding recorded remote
evidence for the non-force push, not run on the maintainer Mac.

Fresh workflows actually exist for this head, including Desktop release
completeness `37746580632`, CI `37746580579` and Security `37746580134`.
At the initial readback they were queued, not accepted as green. Old link
failures remain historical; this source update alone is not merge acceptance.

### Actual dependency-batch integration

The four reviewed source PRs were integrated through GitHub-native branch merges
into existing #209, without changing main or force-pushing. The integrated head
is `0292b6f4847a298d05a2aa41b621210fcec5d13b`; only the three intended lockfiles
change and their blobs exactly match the Luna-reviewed candidate. The manager
independently compared 780 marker contexts for all three reachable graphs,
including package variants, dependency edges and activated extras: only the
four intended package-version changes differ. Non-target artifacts and all
top-level metadata are unchanged. The 28 target artifact metadata entries and
four representative wheel license texts were verified against primary sources;
this is not a fresh whole-native-closure audit or a model-execution verdict.

Fresh CI `37748187779`, Security `37748187439` and Quality `37748187438` exist
for that integrated head and were initially queued. #207, #208 and #210 remain
open until accepted main integration and exact scoped readback. They will be
closed as incorporated, not individually merged.

## 2. Draft responsibilities and disposition

| Draft PR | Responsibility / disposition | Required next condition |
|---|---|---|
| #197 | Consolidated, stopped dependency/security preparation for seven families | Review the preserved execution stops, integrate current main and pass fresh required CI before merging preparation only. |
| #150 | MAGNeT preparation included in #197 | Close as superseded only after #197 is in main and current-head responsibility coverage is verified; vendored-license and execution holds remain. |
| #158 | BigVGAN preparation included in #197 | Same supersession check; retain Linux/Darwin/native/owner and real-weight obligations. |
| #159 | CLAP preparation included in #197 | Same supersession check; retain dependency/native/operator and real-weight obligations. |
| #173 | Chatterbox preparation included in #197 | Same supersession check; retain unresolved license/native terms, stopped workers and real-weight duties. |
| #174 | WeSpeaker preparation included in #197 | Same supersession check; retain component/provenance/replacement and owner decisions. |
| #176 | Ultravox preparation included in #197 | Same supersession check; retain companion/license/native/owner gates and NO_UPLOAD. |
| #180 | Dia preparation included in #197 | Same supersession check; the upstream/candidate version difference still blocks real execution. |
| #191 | Kyutai DSM PCM scheduling and independent-reference authentication | Keep Draft while secure supported upstream/dependency/native closure, owner decisions and independent real-weight comparison are missing. Source/synthetic CI does not close them. |
| #182 | VibeVoice Realtime native composite, CLI guards and reference tooling | Keep Draft; the latest record still blocks the installed 41-package audit at locked Torch wheel HTTP403 and lacks full authenticated E2E/CPU/Apple evidence. Diagnose before another paid replay. |
| #169 | Stronger VibeVoice collector/security preparation | Verify its responsibilities are preserved by #182; close only after the accepted replacement is actually in main. |
| #147 | SpeechBrain Lang-ID reference preparation and scoped urllib3 update | Reconcile current main without losing hash-bound gates; package/native/source/model/fixture and operator review remain separate from staging-tooling acceptance. |
| #195 | Bot urllib3 update overlaps #147 and has collateral lock deletions | Do not merge unchanged. Close only after #147's scoped update is verified in main. |
| #152 | XCodec2 pre-import evidence and decoder-tooling audit | Preserve Draft/BLOCKED/NO_UPLOAD: primary archive/build/RECORD/API binding, required copyleft/native paths and exact owner/legal disposition remain unresolved. |
| #198 | Isolated California primary-source documentation reconciliation | Review exact official-source changes and required link CI; do not infer applicability or compliance, accept403 globally, or rewrite historical observations. |

The seven source PRs are not yet closed by their presence in #197's history.
Likewise #169 and #195 are not superseded merely because a replacement Draft
exists. A current-head ancestry or scoped-content audit plus actual accepted
main integration is required for each closure. Closing a duplicate review
entry does not dispose of its outstanding model/legal duties.

### Actual #197 reconciliation and current-head coverage

GitHub's main update produced head
`9c63402ed33e5a3b0a1851b9959a973965270d19`. Its tree
`2147c08884cc28e4af285e475fbe4e0f3303ec55` exactly matches Luna's locally
verified `7302800b85b4da1e33ac12687770ad97b75b509c` integration. Fresh CI
`37746838643`, Security `37746838154` and Quality `37746838119` exist but
are not yet accepted as green. The PR remains Draft during review.

All seven source PRs' current heads are ancestors of the reviewed local
integration, not merely their historical heads: #150 `373c853e`, #158
`c38ae4ed`, #159 `40653c7e`, #173 `ddcdd74a`, #174 `edc1e6d0`, #176
`501645ac`, #180 `f39d4a96`. Scoped review retains their original functional
responsibilities. Later MAGNeT vendored-license holds and Chatterbox worker
identity checks strengthen the execution stops; shared documentation refreshes
do not supply missing model, legal or hardware approvals.

### Subsequent reviewed corrections and current heads

The later #197 child `def974d3eb942a107f9262306bbc52ed646019b9` corrects a
concrete source-only reproducibility gap found in final review: matching HEAD
and candidate hashes alone did not reject a dirty Vokra probe/source-contract
file. The Dia worker now requires successful, well-formed exact HEAD and clean
tracked/staged/untracked state before source acquisition and after the probe.
Model-free regression covers wrong HEAD, dirty states, invalid repository,
failed Git status and malformed/empty HEAD output. Luna and manager self-tests,
ShellCheck and diff hygiene pass. Normal commit gates 5/5 and normal pre-push
compliance 8/8 pass; the existing Rust-neutral classifier skipped local Cargo
without a bypass. Fresh CI `37749256154`, Security `37749255699` and Quality
`37749255847` exist for this concrete correction; the old 9c checks are not
child-head evidence. Draft, upstream compatibility hold and NO_UPLOAD remain.

#147 was reconciled with the starting main in a separate worktree, reviewed,
committed and non-force pushed as
`ac31162fde438d0efd257ef8ad2e71a9fc0d3d0f`. The original dirty worktree was
not changed. Both dated legal-source records and main's non-reauthentication
boundary remain distinct, with no new link exemption or owner/legal decision.
Manager independently passed the three stdlib-only SpeechBrain self-tests,
workflow hygiene and diff checks. An initial workflow-check invocation could
not access the default UV cache; the established writable cache rerun passed.
No environment sync or third-party import occurred. The excluded benchmark
crate tree `ab134dbc7884c67e97cffeebcd5060768e43623f` matches accepted VAST
`6a02a03e`; root Rust/Cargo/test inputs match accepted workspace VAST `0a11d8d9`.
Those corresponding dated remote inputs supported the push instead of local
deep Cargo; this is not a fresh ac311 VAST replay. Fresh CI `37749579773`,
Security `37749579383` and Quality `37749579516` remain to be accepted.
Draft, official-link access concerns, all pending reviews and NO_UPLOAD remain.

The 2026-10-08 readback of official Moshi main is still
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`; upstream
[issue #429](https://github.com/kyutai-labs/moshi/issues/429) remains open with
zero comments. This does not provide a supported patched dependency range for
#191. #182's public head remains `477ddd32`, Draft; its independent model and
audit duties are not absorbed by #197. #152 current-main lock/identity
reconciliation is delegated without promoting approval or execution state.

At this readback no PR in the frozen 20-PR scope has newly merged or closed.
The maintainer main remains the starting baseline, and the management record
is committed separately. Current-head CI remains an actual wait, not a green
claim. The explicitly superseded old #197 run `37746838643` received a normal
cancel request; its first readback was still queued, so cancellation completion
is not inferred from the accepted request. Current-head verification is retained.

### Subsequent #199 acceptance and current-main refresh

PR #199 was normally squash-merged at 2026-10-08 08:36:36 UTC as
`edc2ab5404f1b9d23dab121f9b4d7e1657ea9db9`. The reviewed exact head remained
`cf36b448a63105fc8d97147982212cabf08485b5`; all 16 required checks succeeded.
Desktop release completeness `37746580632` succeeded on Windows, macOS and
Linux and in `assemble-and-verify`. No failed current-head check was present
at acceptance; other non-required checks were still nonterminal, so this is
not a claim that the entire rollup was terminal. No admin merge, protection
change, package-registry release or model publication occurred. GitHub's
merged-state/commit readback and local fast-forward verify the actual main
integration. This supersedes the earlier unmerged #199 observations above.

The remaining reviewed candidates were updated through normal GitHub branch
merges. The manager independently verifies that each parent-to-child raw Git
diff is exactly the accepted nine-file #199 change and that both the reviewed
parent and new main are ancestors:

| PR | Current head | Fresh CI / Security / Quality |
|---|---|---|
| #209 | `f61c5d0e707fdb0adff65ea26ac4e859b2251922` | `37751100055` / `37751099734` / `37751099758` |
| #197 | `90585b100c75ea19bc305a38cb6958538073c406` | `37751101589` / `37751101264` / `37751101245` |
| #147 | `b18fcc6dc317fef9ce1390ac922902f5d0a65d62` | `37751105191` / `37751104870` / `37751104790` |

These fresh runs are queued/pending, not accepted green. The superseded #197
run `37746838643` is now authoritatively completed/cancelled, while current-head
verification is retained. In the frozen scope, one PR has actually merged;
the other 19 remain open, and no incorporated-original closure is yet made.

PR #152's five-file lock/identity reconciliation is independently reviewed:
only the multidict package record changes, it exactly matches main's 6.9.1
record, all bound hashes agree, and approval/execution/publication states are
unchanged. The normal merge commit `b8148ab3654d01496951969b673d5bdc0b5ae90f`
passed all five commit gates. A new main merge and explicit hosted stdlib-test
wiring are being reviewed, not yet pushed or passed. Local full unittest was
refused by the maintainer safety hook and not bypassed. Accepted historical
native/archive evidence is not relabelled as evidence for the changed lock.
PR #198's dated-source conflict reconciliation is separately under review;
no fresh legal reauthentication, applicability verdict or sign-off is inferred.

### Subsequent independent Draft updates published

PR #152 is now non-force pushed as
`66b5a7c10a95184351af8d0ab75a03143fbc5422`, with reviewed parent
`b8148ab3` and main `edc2ab54`. The parent-to-child diff has the accepted
nine-file #199 change plus two reviewed additions: the explicit hosted
XCodec2 regression step and a dated identity supersession note. The three
stdlib/synthetic suites contain 22, seven and six named cases (35 total);
their fresh hosted results are pending, not substituted by local self-tests.
Normal commit gates 5/5 pass. Manager independently passes actionlint,
workflow hygiene, diff checks and compliance regression 8/8. Runtime/Cargo/
test inputs match accepted dated VAST `0a11d8d9`; release tools match accepted
VAST `3468a48c` and merged main. These identities support the non-force push
instead of prohibited local deep Cargo, not a fresh whole-HEAD VAST verdict.
Fresh CI `37752090942`, Security `37752090609` and Quality `37752090602`
exist and are queued. Draft and all mandatory dependency/native/owner/model
and publication gates remain unchanged; historical archive receipts retain
their original identities.

PR #198 is non-force pushed as
`799a407b2501fbda41f92cf94f025cf43af4bf87`, with original `ab33cd6c` and
main `edc2ab54` parents. Its main-relative scope is exactly the two existing
documents. The manager reviews both conflict resolutions and verifies exact
workflow/release-tool identity with main and unchanged runtime/Cargo/test
inputs against the corresponding dated VAST source. Normal commit gates 5/5,
focused example/reference/runbook checks and diff hygiene pass. Its push uses
the same corresponding source-evidence boundary, not new local deep Cargo
or a fresh legal-source authentication. Fresh CI `37752095434`, Security
`37752095100` and Quality `37752095005` exist and are queued. Keep Draft;
Oct6, Oct7 and Oct8 source/date boundaries, checklist warnings and old 403
failures are retained, with no new exclusion, sign-off or compliance claim.

The existing PR bodies have current-head supplements without removing old
receipts. In this batch no new provider was allocated and no local model,
third-party environment synchronization/import or artifact publication was
performed. Superseded old CI runs `37749256154` and `37749579773` are now
completed/cancelled. The old #209 run `37748187779` has a cancellation request
but its first readback remains queued; do not infer terminal cancellation.
All five current reviewed candidate heads retain their own live verification.

### Subsequent required-link diagnosis and scope separation

At unchanged #147 head `b18fcc6dc317fef9ce1390ac922902f5d0a65d62`, required
[documentation-links job113224260463](https://github.com/ayutaz/vokra/actions/runs/37751104870/job/113224260463)
completed FAILURE at 2026-10-08 08:55:29 UTC. The manager reads the completed
log: nine HTTP403 citations in `docs/legal-compliance.md`, covering five unique
official California endpoints (AB853 billCompare/billStatus, AB2713 billNav,
BPC section22757.6 and BPC Chapter25). This is a retrieval failure, not a
runtime/parity verdict or legal reauthentication. The existing PR body now
records the actual failure without removing earlier receipts. No new link
exemption, accepted403 status, unchanged retry or admin merge was performed.

Read-only Luna comparison finds overlapping but non-identical legal changes
in #147 and the isolated legal reconciliation #198. The next bounded change
is responsibility separation: preserve every unique #147 legal record and
source in #198, integrate its separate October7 dated observations, then
restore only #147's legal-document blob to accepted main. The security
workflow, SpeechBrain preparation, pending manifests and benchmark changes
remain in #147. The transfer is not yet reviewed, committed or pushed at this
record. A per-hunk preservation audit is required before either push; moving
the review scope must not silently drop facts or resolve the primary-source
access failure. #198 retains that explicit unresolved duty and required CI.

The superseded old #209 run `37748187779` still ignored the earlier ordinary
cancel request. After exact old-head and queued-job revalidation, a targeted
force-cancel request was made. Authoritative readback now reports
`completed/cancelled` at old head `0292b6f4`; all latest reviewed-head runs
remain intact. No workflow run/log was deleted and no main run was cancelled.

The #152 hosted `python-parity-oracles` job `113227528167` in Quality
`37752090602` is still queued with no step results at the 09:03 UTC readback;
its new three suites / 35 cases are not claimed as passed. #209, #197 and
#152 latest rollups show no failed checks, but remain nonterminal. #198 also
remains queued; no further merge or incorporated-original closure is inferred.
Main remains `edc2ab54`, clean. No local model execution, provider allocation
or publication occurred in this diagnosis.

The normal management-record commit `6b06148f` subsequently exposed a local
hook side effect: its stdlib lint uses `uv run --project tools/parity`, which
automatically created this new worktree's Python environment and installed
146 packages (about 1.3GB). This is environment synchronization, so no
no-sync claim applies to that commit. No model load/forward was performed.
Only that newly created task-owned `.venv` was deleted after exact-path and
creation-time inspection; it is reproducible from the existing lock. Future
local static hooks use no-sync/offline operation with a validated existing
environment rather than silently installing a new dependency closure.

### Reviewed scope transfer published; hosted XCodec2 regression accepted

The legal-scope transfer above is now reviewed, normally committed and
non-force pushed, in preservation-first order:

- #198 `6b567fedc41f1c8ebdffe31e0a54d649d2e1531a` changes only the legal
  guide and dated documentation audit. The manager's first review found
  omitted concrete definition/exception/operative-date details; Luna restored
  them as October6 observations before acceptance. All unique #147 legal URLs
  are independently verified present, with the October7 observations and
  October8 non-reauthentication boundary preserved. Official access, current
  applicability and owner/legal approval remain unresolved here.
- #147 `82f8544f2f2ef82409c493024b9372785bed2ae4` changes only the legal
  guide back to the exact accepted `edc2ab54` main blob. All SpeechBrain,
  benchmark, workflow and pending approval inputs are unchanged. This removes
  the unrelated legal review from this PR, not from the project scope.

Both normal commit gates pass 5/5 and both normal push compliance regressions
pass 8/8. The actual public tracking-branch comparisons classify the one-/
two-document child diffs as Rust-build-neutral; no hook bypass or compiling
Cargo was used. Hook execution used the validated existing Python3.12
environment with no-sync/offline settings and no installation output.
GH readback confirms both published exact heads, still Draft/unmerged.

Fresh #147 CI/Security/Quality are `37754825184` / `37754824988` /
`37754825118`; fresh #198 equivalents are `37754807269` / `37754807054` /
`37754806868`. They exist but are queued/pending, not accepted green. PR bodies
and #198's broader EU/California title now describe the retained responsibilities
and fresh heads. Old #147 CI `37751105191` and old #198 CI `37752095434` received
normal cancellation requests after exact-head supersession checks; immediate
readbacks remained queued, so terminal cancellation is not yet claimed.

At unchanged #152 head `66b5a7c10a95184351af8d0ab75a03143fbc5422`, hosted
[python-parity-oracles job113227528167](https://github.com/ayutaz/vokra/actions/runs/37752090602/job/113227528167)
actually runs from 2026-10-08 09:11:01 to 09:11:16 UTC and completes SUCCESS.
Manager reads the specific completed log: the newly wired XCodec2 suites run
22, seven and six tests, each OK (35 total). The parent run's exact head is
independently confirmed. This closes the added hosted regression obligation,
not all CI or package/native/model/owner/independent-real-weight/Apple gates.
The body records the result without overwriting the previous pending receipt.

After both ordinary cancel requests remained queued, exact old-head job
readback found only the queued `python-wheel-build` leaf in each superseded
run and no active leaf. Targeted force-cancel requests were then made for
`37751105191` and `37752095434` only. Both now authoritatively report
completed/cancelled. Latest #147 CI `37754825184` and #198 CI `37754807269`
remain present at their reviewed new heads and progressed from pending to
queued; no logs/runs were deleted and no main/current-head run was cancelled.
This supersedes only the immediate nonterminal cancellation observations above.

The frozen batch remains one actual merge (#199), 19 open PRs and no
incorporated-original closure. Main remains `edc2ab54`, clean and synced.

### New-head link outcomes and bounded primary-source investigation

At unchanged #198 head `6b567fed`, required documentation-links job
`113236626598` completes FAILURE at 2026-10-08 09:24:05 UTC. Manager reads
the completed job log: 929 total, 899 successful, zero timeouts and 15 HTTP403
citations across seven unique official California URLs (three citations in
the dated audit, 12 in the legal guide). The five earlier #147 URLs and the
AB2713/SB1000 chaptered billText URLs are all retained. No identical retry,
expanded exemption, accepted403 or current-law conclusion is made; the body
now records this actual result instead of inferring green from scope transfer.

At unchanged #147 head `82f8544f`, required documentation-links job
`113236681971` completes SUCCESS at 09:26:05 UTC. Manager reads its log:
897 total, 884 successful, zero timeouts and zero errors. The exact-head
Security run `37754824988` is completed/success. Other required checks remain
pending, and the separately retained #198 legal/source duty is not waived.

A new bounded investigation finds the separately published
[official PUBINFO index](https://downloads.leginfo.legislature.ca.gov/) and
[provider instructions](https://downloads.leginfo.legislature.ca.gov/pubinfo_Readme.pdf).
Direct index GET succeeds HTTP200 (8519 bytes). A one-byte Range GET of the
listed session ZIP succeeds HTTP206 with total length 1288351194 and
Last-Modified 2026-10-05 04:26:12 GMT. This proves technical accessibility,
not the relevant bill/code record identity or current applicability. Luna's
temporary selective inspector is bounded to 64MB fetched and 128MB memory,
fixed official origin, exact range/ETag and archive-integrity checks. No full
1.2GB download, database installation, model activity or repository-source
edit is authorized by this investigation. Its result is still pending here.

Current #209/#197/#152 verification continues without failed required checks;
running/queued jobs are live waits, not merge permission. #152's 35-case
acceptance above does not replace its outstanding required checks or primary
archive/native/upstream/model/owner duties. Main remains clean at `edc2ab54`.

### Subsequent temporary-reader safety review and CI coverage boundary

The manager reviews all 513 lines of the corrected temporary official-source
reader before authorizing more retrieval. Streaming curl and exact Range/ETag
checks improve the earlier implementation, but the requested 64MiB fetch and
128MiB memory limits are not yet proved: ZIP member decompression remains
unbounded, table parsing materializes all rows, and local-header, archive
bounds, header-length and cleanup checks remain incomplete. Further network
retrieval is stopped pending bounded decompression, sequential selected-row
processing and adversarial offline tests. The older metadata-only output is
not accepted as proof of enforced limits or authenticated bill/code records.
This supersedes the prospective bounded-inspector description above; it does
not replace the independently verified index/one-byte Range observations.
No full ZIP, statutory payload, model, database or paid provider is activated.

The manager independently reads completed #197 Quality jobs through their
individual log API while the parent workflow remains live. At exact head
`90585b10`, `python-parity-oracles` job `113224749219` succeeds, but the log's
test list is the existing Whisper/workflow-oracle suite; it does not execute
the new Dia compatibility or Chatterbox worker-contract tests. Similarly,
successful `shell-self-tests` job `113224749229` covers the named generic
checks and publication/hook self-tests, not the Dia inspection worker's
source-clean regression. Retain the separately recorded focused source-only
verification for those changes; do not label these two hosted jobs as new
family-specific regression acceptance or model parity.

The current required checks are independently read back against all 16 strict
protected contexts. #209 has 13 successful checks and three queued checks;
#197 has 11 successful checks with two running and three queued. Exact jobs
`113229896294` / `113229896422` / `113245511567` (#209) and
`113228158789` / `113228158655` (#197) verify genuine queue/execution state.
No failed required check is present in either candidate, but neither can be
merged yet. Main remains clean and matches live `edc2ab54`; no additional
merge or incorporated-original closure is inferred.

### Current #169 / #182 responsibility-coverage audit

Read-only Luna audit confirms current #169 `b3e9d5c1` and #182 `477ddd32`
remain open Drafts and neither is incorporated into accepted main `edc2ab54`.
#169's project, lock and five-case lock-contract regression are byte-identical
in #182. The collector's installed-closure bounds and
`OWNER_REVIEW_REQUIRED_NO_UPLOAD` status are preserved; #182 adds strict JSON
duplicate-key rejection and strengthens the collector regression from 31 to
33 cases. Its README retains the earlier collector responsibilities and the
separate outstanding license/native/owner/reference/real-weight/Apple gates.
One current command-description sentence still says 31 tests despite naming
the current 33-case suite; a bounded correction is delegated without rewriting
the earlier 31-case receipts. No model/composite/Metal completion is inferred
from the added runtime, CLI, manifest or backend error handling.

Both branches still need current-main reconciliation and fresh acceptance
evidence. A new owned #182 worktree may prepare the normal main merge and that
current-description correction, but no public push or additional CI launch is
made before manager review and the current normal-PR acceptance wave. Closure
of #169 still requires an actually merged replacement and a scoped readback
of the retained and strengthened responsibilities, not this coverage report.

### Actual bounded official metadata acceptance

After the safety review, the temporary reader gains a separate
`--metadata-only` route. Manager independently passes its offline adversarial
self-test and reviews its call path: no table payload, LOB or selected-record
decompression is called. The authorized single actual retrieval exits zero in
17.98 seconds, fetching 2,491,728 bytes with measured peak RSS 33,161,216 bytes.
The exact reader SHA-256 is
`9f6b839356fb7c4486a963f26896b78b7b3b5259a25ae38d4cb4861f9aa62939`.
The fresh task-owned metadata receipt is distinct from the unaccepted old
output and the manager verifies its fields and byte-accounting sum.

The official session ZIP still reports 1,288,351,194 bytes, ETag
`"4ccaadda-65d10482c8100"` and Last-Modified 2026-10-05 04:26:12 GMT. The
one-MiB central-directory probe observes 10,985 of 205,033 expected entries;
the full count and payload hashes remain unverified. Selected table metadata
is now known: BILL_TBL compressed/uncompressed 100,852/954,288 bytes;
BILL_VERSION_TBL 756,287/4,512,719; LAW_SECTION_TBL 6,058,294/43,276,148.
The law table exceeds the new eight-MiB member cap and is not acquired.

This changes the next safe action: authenticate the official loader's column
schema without executing SQL, then separately retrieve just the two bounded
bill tables and select the three named bill/version rows. That preparation
is delegated, with new network activity held until manager review. Bill-text
LOBs and the larger law table need a later bounded design; there is no full
ZIP, database, model or paid-worker operation. #198's required link failure,
statutory-byte/record-chain work, current-law/applicability questions and all
sign-off/CI protection boundaries remain unresolved.

### Reviewed local #182 clean preparation

Luna prepares the conflict-free normal merge of public #182 `477ddd32` with
accepted main `edc2ab54` in a new owned worktree. Manager independently checks
that the staged 92 paths exactly equal the merge-base-to-main delta: 90 blobs
are identical to main, and the two different documents retain only #182's
dated Realtime history and scope caveats. The additional README change is
the reviewed current-command description from 31 to 33 collector tests;
manager's AST inspection confirms 33 collector and five lock-contract cases.
The earlier 31-case historical receipt remains intact. Diff hygiene and all
documentation-reference legs pass independently.

The normal local merge commit is
`a492cc7cd54d092ba6134ee119df06a1e223f3cf`, with parents `477ddd32` and
`edc2ab54`, tree `7a46b138b18fbbeeefc0771538d3a46407758996`. Normal commit
hooks pass 5/5 using the existing environment with synchronization disabled.
Manager verifies the parents, tree, exact README delta and clean owned
worktree. This candidate is not pushed and no fresh CI is launched yet;
the public PR remains `477ddd32`. Update it after the current normal-PR
acceptance wave to avoid an unnecessary serial whole-CI restart. Historical
source/tokenizer/`21dded9b` Rust/`7e11e027` API receipts keep their original
scope; no fresh whole-HEAD VAST, real-model, approval or Apple verdict is made.
No original dirty worktree or protected owner manifest is changed.

### Actual official bill/version source-only advancement

After manager review and an independent offline self-test, the separately
authorized `--bill-records-only` retrieval exits zero in 10.87 seconds. It
fetches 3,349,015 bytes with measured peak RSS 47,480,832 bytes. The official
loader SQL members (500 and 530 bytes) supply the static column schemas;
SQL is not executed. The two authenticated table payloads retain their
declared sizes/CRC and have SHA-256 values
`28f3e439d5119054f1c79e60409438054dc603956115d52a274f40bbc690c227`
and `2c4eebab90988ec5c569f3db33ab0ade9f6683458fc986e87676a46839482044`.
Archive length/ETag/date match the earlier actual metadata receipt.

Manager reads the fresh bill-records receipt: the selected table rows report
AB853 chapter 674 (2025), AB2713 chapter 856 (2026) and SB1000 chapter 861
(2026), with `Chaptered` status. Their latest version IDs are respectively
`20250AB85393CHP`, `20250AB271393CHP`, `20250SB100094CHP`, pointing to
`BILL_VERSION_TBL_9524.lob`, `_19416.lob` and `_19249.lob`. These are dated
official table facts, not authentication of the LOB/statutory text or a legal
applicability judgment. All earlier receipts remain separate; no full ZIP,
LAW_SECTION, bill-text LOB, model, Cargo or paid provider is acquired.

The next evidence requirement is preservation/readback of the six selected
full column/value rows from the actual output, then a separately reviewed
bounded retrieval of the three referenced LOBs. Missing raw-line bytes must
be reported rather than reconstructed as retrieval evidence. #198 stays
Draft with its actual required-link failure; no primary URL is removed,
HTTP403 accepted, approval invented or model count changed.

### Source-row preservation and exact-summary correction

The actual completed output's six selected column/value rows are preserved
separately for comparison. Their original raw UTF-8 lines were not retained;
all six explicitly say `UNAVAILABLE`, and raw-row hashes are not recomputed
from reconstructed values. Manager comparison finds a summary-only error:
the original receipt shortened SESSION_YEAR `20252026` to `2025` and version
action timestamps to dates. A separate v2 receipt restores the exact column
values, retaining the original receipt and comparison packet. Chapter,
version and LOB facts are unchanged. No additional network or statutory text
is acquired; the raw-byte preservation gap remains explicit for the next
source-collection step rather than invented into evidence.

### Actual dependency acceptance and normal-PR closeout

At reviewed #209 head `f61c5d0e707fdb0adff65ea26ac4e859b2251922`, all 16
strict required contexts actually succeed. Manager confirms unchanged head,
base `edc2ab54`, mergeability and no failed/cancelled current-head checks.
Other non-required jobs are still nonterminal; entire-rollup termination is
not claimed. The main-relative scope remains exactly the three reviewed
lockfiles. A normal squash merge with exact-head match succeeds at
2026-10-08 09:57:37 UTC, producing main
`bf428b40779752728e00f9d63d332981aa7ba3c7`. GitHub state/commit readback and
the clean local fast-forward prove actual incorporation.

All three main lock blobs exactly match the accepted candidate. Manager's
stdlib-only locked-metadata readback confirms Hydra 1.3.7 / Werkzeug 3.1.9,
Mako 1.4.2 and multidict 6.9.1. Current #207, #208 and #210 heads are
ancestors of that reviewed integration. #207 is already closed at 09:57:38;
manager closes #208 and #210 as incorporated at 09:59:25/09:59:28, with
scoped disposition comments. None is separately merged or branch-deleted.
No model/reference/native/owner/Apple/publication verdict is inferred.

The frozen normal-PR scope is now complete: two actual merges (#199/#209)
and three incorporated closures (#207/#208/#210). The authoritative open list
has exactly 15 Drafts and zero normal PRs. #197/#147/#152 still need the new
main's three-lock delta and fresh exact-head CI. Their conflict/delta review
is delegated before public branch updates; old-head checks stay historical.
#182's clean local preparation remains unpushed. Original-family closures,
retained-Draft gates and the overall Draft-disposition finish audit remain
open; the persistent goal is not complete.

### Latest accepted-main Draft integrations

After the actual #209 acceptance, manager independently reviews three normal
merge candidates against `bf428b40779752728e00f9d63d332981aa7ba3c7`. Each
adds only the three accepted dependency lockfiles; all three blobs match main.
The staged candidates have no runtime, Cargo, source-test or approval-input
changes. GitHub-native updates use the expected old head and normal merge
semantics, without force, rebase, admin acceptance or manual CI cancellation.
Manager then independently verifies the published parents and exact tree:

| PR | Published head | Parent heads | Tree |
|---|---|---|---|
| #197 | `00d019210248b3e06dbbef269e5a7f07337cc3ee` | `90585b10`, `bf428b40` | `5381322dfff1ea58fc86faaf35ce99e980cba585` |
| #147 | `7b596dd9477a121d6411cc1214c6c90b899b3d42` | `82f8544f`, `bf428b40` | `c0950c0c910faf7cc6f45d6e1352daac892993c8` |
| #152 | `fe1f5fe27fe4016d6a6fd76fe131fa3d62e1f307` | `66b5a7c1`, `bf428b40` | `edd4a32e4066965781295538d4b4efcfbd500bf4` |

Fresh exact-head CI / Security / Quality runs are respectively:

- #197: `37760742831` / `37760742350` / `37760742366`.
- #147: `37760770589` / `37760770161` / `37760770044`.
- #152: `37760793532` / `37760793150` / `37760793324`.

All are real registered runs, initially queued/pending, not accepted green.
The old CI runs `37751101589`, `37754825184` and `37752090942` are independently
confirmed completed/cancelled after the updates; no live run remains for those
three old heads. Their logs and receipts remain historical. The public bodies
now name each new head and retain Draft, execution/owner/license/reference/
Apple holds and `NO_UPLOAD`. Manager verifies those body/head readbacks.

A subsequent live readback still finds 15 open Drafts and zero normal PRs.
It also observes a changed #195 head (`bf54a6c2`), so the earlier collateral
scope and duplicate closure proof must be revalidated against that current
head before any eventual closure. No changed bot head is silently treated as
the older reviewed head. The #182 local candidate remains unpushed; its
three-lock latest-main preparation is delegated for independent review.
These updates advance source/CI preparation, not any model or Apple verdict.

### Actual #182 current-main update and retained local correction

Manager independently checks the separate public-update candidate: its 95
incoming paths exactly equal the `d100d937`-to-`bf428b40` accepted-main delta.
Ninety-three staged blobs match main; the two remaining document differences
retain only #182's dated Realtime records and caveats. This candidate does
not contain the separately prepared 31-to-33 README correction or any new
runtime authorship. A normal GitHub-native update with expected head succeeds;
the published head is `a6ff40225b975eff7b17ce4ef5c429abd47e7418`, parents
`477ddd32` / `bf428b40`, tree `4b4d90d766e00dc491adf8b6af12b7e70f5112b4`.
Manager independently verifies the public parents/tree and Draft state.

Fresh CI `37762123156`, Security `37762122931` and Quality `37762123290`
exist and start queued/pending, not green. The updated public body matches
the owned body exactly and preserves the previous 64,929-character history.
The local `a492cc7c` plus three-lock staged candidate remains unpushed, including
its README correction. The literal normal pre-push classifier requires the
deep path for that local candidate; no local heavy Cargo or hook override is
used. GitHub's ordinary base integration is not fresh whole-HEAD VAST evidence.
Installed-closure Torch HTTP403, dependency/native/license/voice-rights/owner,
full E2E/CPU/reference and Apple/publication duties remain open.

### Current bot and upstream scoped readbacks

At current #195 `bf54a6c2`, the lock has 63 additions and 63 deletions, rather
than silently retaining the older bot-head disposition. Independent read-only
comparison proves its complete urllib3 2.8.0 block (source and both artifact
hashes/sizes) exactly matches #147 `7b596dd9`. #195 also removes edge markers
and Torch/TorchAudio wheel size fields while preserving the project's Linux
x86_64 and Python 3.12 restrictions. Effective contextual graph equivalence
of that collateral normalization is not proved here. #195 remains open until
the reviewed replacement is actually in main and scoped responsibility proof
is repeated; that eventual dependency-only closure does not complete #147's
separately retained model/reference/owner/Apple duties. Fresh primary issue
readbacks confirm Moshi #429 and X-Codec-2.0 #41 are open with zero comments.

### Actual bill-text retrieval and capture failure diagnosis

After correction review and manager's own offline adversarial PASS, literal
`--bill-text-only` at reader hash
`d9c63609e74e1c56bd19ec77e20958378244876251328d5b2e07f6f4ccf0035d`
exits zero after 37.40 seconds, with measured peak RSS 60,784,640 bytes and
22,937,132 fetched bytes. The visible output reports the fixed archive identity,
three expected bill/version/LOB chains and `complete-bill-text-only`.
However, the execution tool truncates stdout. The separate v3 receipt records
`stdout-truncated-by-tool`; it is not a complete preserved source receipt and
does not prove the complete LOB/hash set or independent raw-byte replay.
No missing bytes are reconstructed, and no identical acquisition is retried.

The concrete defect is the stdout capture boundary, not archive retrieval.
A bounded generated-receipt file mode is delegated for offline validation
before any further source run. It must preserve the complete original rows
and three LOB payloads directly, refuse overwrites/symlinks, and print only a
small summary. Prior receipts remain intact. #198's required link failure,
current-law/applicability and sign-off boundaries remain unresolved; no full
ZIP, LAW_SECTION payload, SQL/database, model, package installation or paid
worker is part of this source-only work.

### Corrected generated capture: actual source acceptance boundary

The manager finds and rejects a further writer cleanup race before execution:
an `O_EXCL` failure must not unlink a file created by a competing process.
Luna's correction limits cleanup to its own successful creation, rejects a
non-owned/symlink parent and adds a causal competing-creator regression.
Manager reads the corrected functions and independently passes the offline
capture/scanner adversarial suite at exact reader hash
`ea117df6d76d8e55e894c1a785a1fb7e2ca64b640c4ba76dbfe6f1300ca6b439`.

One changed capture-procedure run exits zero after 51.73 seconds, peak RSS
60,112,896 bytes. It generates the complete v4 receipt directly (95,161 bytes,
SHA-256 `33e0853f58e4a8501fec6de4d3d229935049fe8dda1f1bbb5c2e376adea9b11e`)
and emits only a bounded summary. Source requests total 22,937,132 bytes;
archive ETag/size remain the frozen values. The older v3 truncated capture
remains unaccepted and intact; this is a corrected producer, not an identical
blind retrieval retry.

Manager independently reads the saved complete JSON, re-parses all six original
UTF-8 rows (including terminators), and verifies raw byte lengths/hashes,
column values and all three bill/latest-version/LOB chains. Original decoded
LOB byte sizes/hashes match their payload receipts:

| Bill | LOB member | Original bytes | SHA-256 |
|---|---|---:|---|
| AB853 | `BILL_VERSION_TBL_9524.lob` | 21,326 | `f69387024f914945379706b944361dce5870573c9e0ecddde08b51daaeff6dda` |
| AB2713 | `BILL_VERSION_TBL_19416.lob` | 9,915 | `34bdebb75d57c3be1f5c935d260e33a7115a84e7ae8eb0337caaef0120f54e60` |
| SB1000 | `BILL_VERSION_TBL_19249.lob` | 28,045 | `4fe006c5fc88097a27e39b9b99d132039c83e495bfb4e48808e7cfb28fb1b61d` |

The saved archive fields still describe the initial directory probe: 10,985
of expected 205,033 entries, `entry_count_verified=false`. The literal bill-text
route requires a complete scanner/count check before member retrieval, but a
separate full-directory proof/digest is not serialized; initial probe flags
are not rewritten or represented as that proof. Manager's XML readback confirms
the expected chaptered measure identities and named sections. This closes
the observed full-text/raw-row preservation gap only. LAW_SECTION is not
acquired; integrated current-law/applicability, approval and #198's required
link failure remain open. No full ZIP, model, package installation, Cargo or
provider operation runs. The next source task is comparison with the proposed
legal wording and the remaining official current-code chain, not re-acquisition
of these now-preserved three bill texts.

### Preserved bill-text wording review and bounded LAW source preparation

At unchanged Draft #198 `6b567fed`, the read-only wording review reuses the
accepted original v4 receipt rather than reacquiring the three bills. Its
chaptered AB853/AB2713/SB1000 bytes support the named chapter identities,
dates, section changes and statutory date observations in the proposed
document. The EU observations and the historical browser's current-code
display remain separate sources; the California bill-text receipt does not
authenticate either. No wording is promoted to current-law applicability,
compliance or owner/legal approval. The public body preserving the v4 facts
is independently confirmed at the same unmerged head.

Before a LAW_SECTION acquisition, manager rejects and corrects three concrete
producer problems: stripping arbitrary path prefixes before checking a LOB
basename; checking the 512-KiB LOB output cap only after retrieval; and using
zlib `flush(length)` as though its initial buffer size were a hard allocation
limit. The separate Luna implementation validates the full official basename,
rejects oversized entries before fetch, consumes deflate output in <=64-KiB
pieces and finalizes only at clean EOF without `flush`. The receipt includes
the complete 504-byte official loader SQL member, whose source SHA-256 is
`212274f8ccbc93ea8e48258a0c062ac337c26cce357986912caccef1290329d6`;
SQL is parsed statically, never executed.

Manager independently reads the corrected implementation and passes its
offline adversarial suite at producer SHA-256
`b1edf092185e26cb4a210b3d5b3a06ac35eb27a3222623d28ddfb57246cf6f22`.
The frozen bill-text helper remains
`ea117df6d76d8e55e894c1a785a1fb7e2ca64b640c4ba76dbfe6f1300ca6b439`.
The input fetch uses <=4-MiB slabs, split into <=64-KiB inflater pieces,
rather than 93 individual payload requests. Full-table size/CRC/hash and
complete directory scans remain mandatory; only selected BPC 22757 rows and
their exact LOBs may be preserved. Aggregate transfer stays <=64 MiB, each
selected LOB <=512 KiB and the exclusive receipt <=4 MiB. Existing receipts,
the 8-MiB member cap and the protected owner manifest remain untouched.

The actual source-only run has started at session `83888`; it is confirmed
live, not yet accepted as source evidence. This is not a full ZIP download,
database import, model/package execution, paid provider or legal approval.
Current #197/#147 exact-head CI handles `37760742831`/`37760770589` are also
confirmed live. Required parity is not registered yet; pending/queued checks
are not PASS. Branch protection still requires all 16 contexts with strict
base freshness. #198 has 15 successful required contexts and the actual
documentation-links failure; it remains Draft with no waiver or blind retry.

### Actual selected LAW_SECTION preservation and independent acceptance

Session `83888` terminates zero after 84.98 seconds, with peak RSS 54,280,192
bytes and 47,711,457 bytes fetched. The generated exclusive receipt is
102,182 bytes, SHA-256
`26ca867f9ac1a60b62571e597e24ee99819fa348c7fca503dd6470afacb638a3`.
The fixed archive ETag/size do not change. Both complete central-directory
scans are serialized separately from the initial 10,985-entry probe; the
scanner checks the full expected 205,033 records and archive boundary before
returning. No complete central-directory byte digest is claimed.

The streamed LAW_SECTION member has compressed SHA-256
`c9d8e3b2d1de20ada218c954b90aa3872d56c8dc89682937bd8c15922dcc54a5`
and uncompressed SHA-256
`dde1d3d09d161856d7e35906f6ba855724da6b878ea9b8a7454f75c9d8eac3bb`.
Its declared 6,058,294/43,276,148-byte sizes and CRC32 `ae791104` pass the
producer's streaming integrity checks. Manager independently parses the saved
complete JSON, authenticates the 504-byte SQL member, re-parses all 18 original
selected rows and verifies their byte lengths, hashes, exact column values
and referents. All 18 original base64 LOBs independently pass decoded size,
SHA-256, CRC32 and strict UTF-8 checks; the exact referenced/preserved sets agree.
The whole 43-MiB table is not preserved or independently re-inflated locally.

The prefix selection includes 11 Chapter 25 sections and seven SB53/Chapter26
sections (22757.10 through 22757.16). The latter are outside this PR's legal
wording review: 18 selected prefix rows is not 18 Chapter25 obligations.
The saved Chapter25 XML and row histories support the dated SB1000 source
observations for sections 22757.1/.2/.3/.4/.4.1/.5, and distinguish section
22757.6's January1,2026 effective history from its August2,2026 operative
text. Section 22757.3.1 remains the AB853/2025 Chapter674 version in this
official table, while the preserved chaptered AB2713/2026 Chapter856 bill
amends that section. This version boundary is retained for authoritative
legal reconciliation; neither source is silently overwritten or promoted
to an integrated latest-law/applicability verdict.

The new receipt closes the selected original current-code-source preservation
gap, not EU reauthentication, legal review/sign-off, role-specific
applicability, required-link CI or model/Apple gates. Source-boundary-only
additions to #198's two existing documents are delegated for review before
publication. The protected manifest, all prior receipts and model counts
remain unchanged. No full ZIP, SQL/database, model, package installation,
Cargo verification, paid provider or upload runs.

### Published #198 source-only documents on actual current main

Manager catches and rejects two concrete wording errors in Luna's initial
document diff: the LAW receipt references 18 LOBs, not three, and bill/LAW
preservation comes from separate actual runs. The corrected two-file diff
also distinguishes a BPC section from a subsection. Manager reviews the
corrected diff and independently passes example self-tests, the 144-block
source/example check and doc-reference gates. The example checker explicitly
retains 30 deferred/unverified tiers; this is not a runtime/toolchain pass.

Normal hooks pass for local document commit `14de2a0e` and the reviewed main
merge `421ab7b1` (tree `92347fbf87ec83c70ac32ab7770bb50dc56194ef`). A normal
push is correctly refused before Cargo: the upstream is still `6b567fed`, so
the incoming accepted-main nested NanoCodec lock triggers the conservative
deep classifier. Compliance regression passes 8/8; neither hook bypass nor
local Cargo is used. The initial sandboxed `git write-tree` inspection is
also refused; actual committed-tree readback subsequently proves the expected
tree, rather than claiming that denied command passed.

The corrected normal workflow first performs GitHub's ordinary base update
with expected head `6b567fed`. The actual published integration is
`ac64ec9a12a93bc855bf0eb238bc8533584faa7d`, parents `6b567fed` / `bf428b40`.
Its three incoming lock blobs exactly equal main. A separate owned worktree
replays only the reviewed two-document commit on that actual public head;
the earlier local candidates remain intact. Manager sets the correct existing
PR upstream, verifies the unchanged literal classifier now sees only those
two documents, and commits normally at
`99676662c438d44e43914ccd005d4aca6cf38192`. Parent is `ac64ec9a`; tree remains
`92347fbf87ec83c70ac32ab7770bb50dc56194ef`. Ordinary non-force push passes the
compliance regression and the existing Rust-neutral fast path, without
allow-list/configuration changes, deep Cargo or a paid worker.

GitHub independently confirms #198 at `99676662`, still Draft/unmerged.
Fresh CI `37766087586`, Security `37766087153` and Quality `37766087141`
exist and are queued/pending, not accepted green. The updated body exactly
matches the owned file (15,418 characters) and preserves all 13,154 characters
of the previous body. Earlier LAW-not-acquired statements are explicitly
retained as superseded narrow source scopes, not erased. The historical
`6b567fed` required link failure is not a new-head verdict; every new-head
required check must independently pass. The source-display/version,
EU/applicability/counsel/owner and CI holds remain.

Read-only final preparation review of #197 `00d01921` and #147 `7b596dd9`
finds no additional source-preparation defect in their accepted-main
reconciliation: only the three reviewed locks change from `90585b10` /
`82f8544f`, and every resulting blob equals main. Actual fresh required CI,
normal acceptance and post-main responsibility/duplicate coverage remain
mandatory. No original family, #169 or #195 is closed in this step.

### SpeechBrain bot audit review: initial result not accepted

The bounded #195 audit helper is reviewed against exact main `bf428b40`,
#195 `bf54a6c2` and #147 `7b596dd9`. Manager does not accept its initial
supported-context/extras PASS claim. The helper substitutes `CPython` / `PyPy`
for `implementation_name`, while the [primary PyPA marker specification](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#defined-environment-marker-fields)
defines `cpython` / `pypy` for that field; the capitalized values belong to
`platform_python_implementation`. A literal string partition is not actual
interpreter support evidence. Its extras handling records an edge label but
does not activate optional dependencies, so that test cannot prove extras
closure. Its version-wildcard handling also needs an explicit bounded contract.

Corrections are delegated to the same Luna owner, limited to the two existing
temporary audit-source/test files. The request retains the strict serialized
non-target-row failure, requires explicit project/top-level-lock and artifact
identity checks, and rejects graph forms outside the helper's proven scope.
No dependency resolver, installation, package import, model or Cargo execution
is authorized by this static audit. #195 remains open: even corrected
dependency-only evidence cannot replace actual #147 incorporation into main
and a fresh scoped responsibility readback.

The independently read #191 link job `112586733021` in Security run
`37557412972` is terminal FAILURE for two occurrences of the same official
SB942 comparison URL returning HTTP403, with zero timeouts. This is a dated
external-link failure, not a model or numerical failure. No unchanged retry,
citation removal, global accepted403 rule or owner/upstream decision is made.

### Corrected #195 static responsibility audit: limited acceptance

Luna freezes the corrected temporary helper at SHA-256
`dbeca6a3bb89744bdec2d8e481460643012b5c4e412c0187905db48bf4960da2`
and its tests at
`22246b2a11d54bfa3d5c73dbd17162055d33ae6346c274c8c5c72eaf40aca227`.
Manager reviews the changed contracts and independently runs the eight
unittests through offline Python 3.12/uv: all pass. The aggregate closure now
uses the CPython partition explicitly, rather than the last loop's synthetic
partition. Unevaluated extras, optional/group/dev inputs and string-field
wildcards fail closed. This helper is a bounded static comparator for these
exact Git inputs, not a general Python resolver or interpreter-support test.

Manager's independent actual audit uses main
`bf428b40779752728e00f9d63d332981aa7ba3c7`, #195
`bf54a6c256854708ca0802353373e42649fd5d70` and #147
`7b596dd9477a121d6411cc1214c6c90b899b3d42`. Main and #195 have identical
complete project TOML/direct dependencies and top-level lock metadata.
Both lowercase `cpython` and `pypy` marker partitions have identical
38-package closures and no edge delta; other/empty strings also have 38,
and the explicitly hypothetical titlecase `PyPy` partition has 37. These
are symbolic marker facts, not proof that the locked CPython-ABI Torch wheel
supports PyPy or that any package/model was installed or executed.

All 15 non-target serialized row differences are independently classified
as dependency-marker changes, plus `torch.wheels[0].size` and
`torchaudio.wheels[0].size` omissions. No other non-target version, registry,
artifact URL/hash or field delta is reported by the exact comparison.
The strict audit intentionally exits 1, with `audit_pass=false`; this is
the expected scoped-change rejection, not a failed numerical test. #147's
complete urllib3 artifact contract exactly matches #195, while its separate
SpeechBrain/Torch/TorchAudio dependency updates remain intentional and held
behind their own execution gates. There is no general closure/license,
real-weight, owner, Apple or publication PASS. #195 remains open until the
reviewed replacement is actually incorporated into main and this scoped
responsibility proof is repeated against the then-current live heads.

### Published #195 bounded-review evidence

The accepted narrow audit is recorded in [#195's review comment](https://github.com/ayutaz/vokra/pull/195#issuecomment-6058697186)
at 2026-10-08 11:18:16 UTC. Manager independently compares the API's actual
JSON body with the reviewed 2,071-character local document: exact match.
An initial CLI-display diff shows only its extra printed newline; it is not
used to claim exact equality. Fresh PR readback proves the original Dependabot
body and `bf54a6c2` head are unchanged, and #195 remains OPEN/Draft. The comment
retains the strict scoped-change failure, no-runtime/no-owner/no-publication
boundary and actual-main-before-closure condition. No PR is merged or closed
by posting this review evidence.

### Actual SpeechBrain acceptance and incorporated duplicate closure

PR #147 was normally squash-merged at 2026-10-08 11:22:56 UTC as
`2da1b3fa1cb6ecb15fa1bf3db97b8b369bb5c02a`, parent accepted main
`bf428b40779752728e00f9d63d332981aa7ba3c7`. The reviewed head is
`7b596dd9477a121d6411cc1214c6c90b899b3d42`. All 16 required checks passed
with their exact required application identities; the final parity job
`113274115012` succeeded. Two non-required packaging checks were still pending
at acceptance, so this is not a whole-rollup terminal-success claim.
The merged tree `c0950c0c910faf7cc6f45d6e1352daac892993c8` equals the reviewed
head tree. Normal API merged-state readback and local fast-forward confirm
the actual integration; local main is clean and synced. No admin/force merge,
protection change, model execution, owner sign-off or publication occurred.

Before acceptance, manager independently reran the three stdlib-only
SpeechBrain synthetic self-tests and the forbidden-import/zero-dependency
static gates: all passed.
These do not authorize the pending package/native/source/model/fixture and
operator rows. Corresponding dated benchmark/workspace VAST source identities
remain dated evidence, not a fresh whole-head VAST replay.

The corrected #195 comparator is repeated against actual accepted main
`2da1b3fa`, original bot head `bf54a6c2` and base `bf428b40`. Main contains
the exact complete urllib3 version/source/sdist/wheel artifact contract from
#195. Its 15 non-target serialized differences still cause the strict audit
to exit 1 / `audit_pass=false`; their bounded classification is retained,
not recast as a general resolver, native-license or runtime PASS. Symbolic
marker reachability is equal within the proven scope. Main's separate
SpeechBrain/Torch/TorchAudio changes are intentional, not silently imported
bot collateral. No unique #195 responsibility remains outside accepted main.

PR #195 is actually CLOSED at 2026-10-08 11:24:39 UTC as incorporated through
#147, with `mergeCommit=null` and unchanged head. It is not individually
merged and its branch is not deleted. #147's updated 68,481-character body
is independently read back exactly and preserves its entire prior body.
An initially generated trailing blank line was corrected before final
whitespace validation/readback; no historical evidence is rewritten.

### Published #197 and #198 refresh after actual #147 main integration

Both ordinary GitHub main updates preserve the reviewed old heads and add
accepted main `2da1b3fa` as a parent. Their trees exactly match the separate
Luna-prepared, manager-reviewed no-conflict candidates. The incoming delta is
only the nine accepted #147 paths; original family/legal documents and all
stopped-execution, owner, real-weight, Apple and publication duties remain.

| PR | Current published head | Reviewed tree | Fresh CI / Security / Quality |
|---|---|---|---|
| #197 | `fb4eb91324292bc8e159cebe11613aa75e07b911` | `75f8aa72f2ac6728b0cf9bd1c89bb5ab0b87b3c5` | `37770169016` / `37770168808` / `37770168555` |
| #198 | `5614a92c1938a0cee44ee333bec2b8cbbaa7d969` | `48559583688128d1f94ebb6db09f58601c87f0e2` | `37770169594` / `37770169295` / `37770169314` |

These actual fresh handles are queued/nonterminal, not accepted green. The
readback has 58 queued / one running / one skipped checks for #197 and
33 queued / one skipped registered checks for #198; unregistered checks
are not presumed passed. Both remain OPEN/Draft. Seven original family PRs
remain open until actual #197 acceptance in main and fresh live-head
responsibility coverage. #169 likewise remains until accepted #182 in main.

At old #198 head `99676662`, actual documentation-links job `113273993157`
in Security `37766087153` completed FAILURE at 2026-10-08 11:21:01 UTC:
932 total, 902 successful, zero timeouts and 15 HTTP403 citation errors across
seven official California URLs. This is an old-head external source-access
result, not a numerical failure or a verdict on `5614a92c`. The new accepted
main includes reviewed bounded link throttling; no global accepted403 rule,
extra exclusion, citation deletion or identical failed-job rerun is made.
Original source receipts/version caveats and EU/applicability/counsel/owner
holds are preserved.

At 2026-10-08 11:34 UTC, manager publishes updated #197/#198 bodies and
independently verifies API exact equality (21,920 / 17,261 characters), full
prior-body inclusion, unchanged current heads and OPEN/Draft states. A
generated #198 EOF-blank issue is corrected while preserving the prior
body's literal trailing newlines before the final bounded end note; both
body files pass whitespace checks before publication.

The frozen 20-PR batch now has three actual merges (#199/#209/#147), four
incorporated closures (#207/#208/#210/#195), and 13 still-open Drafts. New
#211/#212/#213 are a separate later bot batch. Original item 1 is complete;
item 2 remains incomplete while #197 acceptance, original-family dispositions
and #198 current-head outcome are unresolved. No provider is allocated by
this PR work; this does not claim that an unqueried cloud account is empty.

### Scoped current-head coverage and redundant-run cancellation requests

Manager fetches the actual public #197/#198 Git objects and independently
confirms their exact parents/trees above. Both parent-to-child changes list
only the nine accepted #147 paths and pass `git diff --check`. Fresh live
original heads #150 `373c853e`, #158 `c38ae4ed`, #159 `40653c7e`, #173
`ddcdd74a`, #174 `edc1e6d0`, #176 `501645ac` and #180 `f39d4a96` are all
ancestors of published #197 `fb4eb913`. This confirms current-head inclusion,
not acceptance in main or proof that ancestry alone preserves every duty.
Their prior scoped responsibility review remains separate; no original is
closed before actual accepted-main incorporation and another live readback.

The current CI list still contains redundant old #198 CI `37766087586` at
`99676662` and closed #195 CI `37761058312` at unchanged `bf54a6c2`.
Manager confirms #198's replacement `37770169594` exists at `5614a92c` with
the same workflow identity `305986097`, event and branch, and independently
reads #195's actual CLOSED state. Ordinary cancellation requests for only
these two redundant runs are accepted at 2026-10-08 11:36 UTC. No run is
restarted; current heads, accepted-main CI and unrelated PR runs are kept.

The 11:38:40 UTC API poll still reports both old runs queued/nonterminal,
#198 new CI pending, and #197 new CI and both new Security runs queued.
Cancellation completion or freed capacity is therefore not claimed. An
intermediate `gh run view` result contains a display warning and cannot be
parsed as JSON; direct run-API readback supplies the authoritative state.
Queue delay is observed, but its provider/capacity cause is not established.
These confirmed live handles are a verified wait, not a stopped/missing-run
condition and not a reason to rerun unchanged jobs or waive required checks.
Local main and the management worktree remain independently clean.

### Redundant old CI cancellation completed; current #198 CI released

The two ordinary cancellations above remain nonterminal on the fresh
2026-10-08 11:39 UTC readback. Job-level evidence narrows the remainder in
both runs to queued `python-wheel-build` with no assigned runner; their other
jobs are terminal. Manager reads the exact old #198 workflow source and
confirms that this aggregator has `if: always()`. The
[official GitHub API documentation](https://docs.github.com/en/rest/actions/workflow-runs#force-cancel-a-workflow-run)
describes force-cancel for runs not responding to ordinary cancellation,
including `always()` conditions. This identifies the surviving condition,
not a general explanation of all runner queue delays.

After another exact-ID/head/nonterminal check, manager uses the official
force-cancel endpoint only for replaced #198 `37766087586` / `99676662` and
closed #195 `37761058312` / `bf54a6c2`. Both requests succeed. Independent
API readback now proves `completed/cancelled`, respectively at 2026-10-08
11:40:44 and 11:40:46 UTC. No logs/artifacts are deleted; their terminal
cancelled states are not recorded as PASS. This is bounded old-run cleanup,
not force merging, weakening branch protection or changing workflow code.

The retained #198 current-head CI `37770169594` moves from pending to queued
at 11:40:44 UTC. The 11:41:02 UTC readback confirms 13 actual registered jobs,
all queued; #197 `37770169016` also has 13 queued jobs. Latest heads, main and
unrelated PR runs remain intact, with no unchanged rerun. The latest Security
checks are likewise not accepted green. #197/#198 remain Draft/unmerged
until their independent exact-head required gates pass; original-family
closure and the frozen-batch completion audit are still outstanding.

### New-head #198 link failure diagnosed and published; #197 advances

At current #198 head `5614a92c1938a0cee44ee333bec2b8cbbaa7d969`, required
[documentation-links job `113287531384`](https://github.com/ayutaz/vokra/actions/runs/37770169295/job/113287531384)
in Security `37770169295` actually completes FAILURE at 2026-10-08
11:48:17 UTC. Manager reads the completed job's final report and independently
confirms its failed link step: 932 total, 902 successful, zero timeouts and
15 HTTP403 citation errors across the same seven official California LegInfo
URLs in `docs/legal-compliance.md` and the 2026-10-04 documentation record.
This is fresh failure after reviewed bounded throttling, not old-head evidence
substituted for the current head. It is source-access failure, not a runtime,
numerical or legal-conformance verdict.

The initial run-level CLI log request is refused because the overall run
still has nonterminal jobs. The completed-job-specific API returns the actual
log successfully; job status and failed step are then separately read back.
No signed log URL, credentials or raw authenticated diagnostic is recorded.
There is no unchanged retry, citation deletion, extra exclusion, accepted403
rule or protection change. The original citations, bounded authenticated bulk
receipts, integrated current-law/version caveat and EU/applicability/counsel/
owner holds remain intact. #198 stays OPEN/Draft as an explicitly held lane.
Its link gate needs an actual passing required result against the preserved
sources; the other source/legal duties are not closed by HTTP availability.

At 2026-10-08 11:50:57 UTC, manager publishes the dated current-head failure
header and independently verifies API full-body equality (18,837 characters),
preservation of the complete previous 17,261-character body, unchanged head
and OPEN/Draft state. Earlier pending/old-failure entries remain dated history.

#197's current-head link job `113287526813` succeeds independently. The
11:48:34 UTC exact required-name/application audit has five required successes
(dependency-review, documentation-links, workflow-security, pins sync and
gitleaks), no current-head failure, and remaining build/test/parity/license/
CodeQL checks still incomplete. The 16 required contexts and app IDs are
re-read from strict main protection, not assumed from prose. #197 is not yet
accepted or used to close its seven source PRs; its independent CI progresses
while #198's genuine external hold is retained.

A full check-run payload exceeds the bounded tool output and cannot be
accepted as JSON evidence. The corrected read projects only check name,
application ID, state, conclusion and ID/URL before returning output; all
registered checks fit, and total-count equality is verified. Missing parity
or CodeQL contexts are not treated as passed. This is an observation fix,
not a job restart or evidence manufactured from truncated output.

The live frozen-scope audit still confirms three actual merges, four
incorporated closures and 13 OPEN/Draft entries. Existing Moshi #429 and
XCodec2 #41 remain open with zero comments and unchanged 2026-10-05 updates;
no new upstream answer or owner/legal decision is inferred. #147 CI
`37760770589` is now actually completed/success at 11:38:53 UTC, superseding
only its residual pending-packaging observation, not recording a model run.

### Seven-original scoped file coverage before #197 acceptance

Manager independently reads each original PR's changed-file list and current
OPEN head, then compares all 53 original changed-path entries with published
#197 `fb4eb913`. All seven actual heads remain ancestors. No original target
path is deleted: 44 entries are byte-identical, and nine changed entries are
individually reviewed below. This is current candidate coverage, not accepted
main incorporation or a claim that all remaining model duties are complete.

| Original PR | Current head | Changed paths / equal in #197 | Reviewed subsequent delta |
|---|---|---|---|
| #150 | `373c853e` | 7 / 2 | Shared AudioCraft gate plus two MAGNeT project policies and READMEs: retained unresolved vendored LGPL evidence and stronger unconditional stop. Both locks remain identical. |
| #158 | `c38ae4ed` | 9 / 9 | All original changed-path contents are identical. |
| #159 | `40653c7e` | 9 / 8 | Only shared dated catalog record changes: named-head snapshot and corrected historical anchor; original evidence remains. |
| #173 | `ddcdd74a` | 3 / 3 | All original changed-path contents are identical. |
| #174 | `edc1e6d0` | 5 / 4 | Only README gains dated exact-wheel metadata facts with ABI/native/owner/NO_UPLOAD holds preserved. |
| #176 | `501645ac` | 10 / 10 | All original changed-path contents are identical. |
| #180 | `f39d4a96` | 10 / 8 | README retains exact-wheel metadata as non-approval; source-only worker adds clean expected-HEAD checks before and after probe. |

The MAGNeT delta adds exact evidence/schema/record matching, changes the
setuptools row from MIT-only to `UNRESOLVED_VENDORED_LGPLV3`, includes that
evidence in the approval scope, and refuses execution even with a synthetic
signed approval. Unknown retained member-path/SPDX details remain unknown.
This stages enforcement only; it does not install/import the locked package,
approve its license or introduce a runtime crate. Existing forbidden-package,
NC/owner and NO_UPLOAD stops remain. Dia retains the upstream-version mismatch,
VAST-only route, expected exit 2 and compatibility stop while rejecting dirty,
invalid, malformed or changed checkouts. No new legal decision is made.

Manager reads the license-audit guidance, deny policy, contributor dependency/
model boundaries and owner sign-off semantics before accepting this narrow
stopped-preparation distinction. Full original-family source, native/license,
real-weight, independent-reference, Apple and publication duties survive any
later duplicate closure; ancestry alone is not their completion evidence.

Two path-rich read-only Git commands are refused by the maintainer model-safety
hook, not bypassed and not counted as executed tests. The permitted candidate
status and whole-diff-name readback instead show only the nine known staged
incoming #147 paths and no content delta against published `fb4eb913`; these
checks do not read/hash the protected CosyVoice manifest. On that matching
candidate, explicit stdlib/synthetic AudioCraft self-test passes. Dia's explicit
inspection self-test also passes: inspection, source contract, model-free
dependency audit/approval, wrapper and clean-head/tamper regressions. Its expected
negative-test diagnostics are not runtime failures. Forbidden-symbol and
first-party-only Cargo.lock static gates pass. No package/model acquisition,
model execution, broad Cargo, paid worker or upload occurs.

The 2026-10-08 11:56:50 UTC exact-name/application readback gives #197 13/16
required successes with zero current-head failures. Remaining macOS build/test
are queued and parity is not yet registered; none is presumed passed. Main is
still `2da1b3fa`. No original PR is closed before actual #197 acceptance and a
fresh post-main head/content responsibility readback.

### Actual #197 acceptance and seven incorporated closures

At 2026-10-08 12:47:48 UTC, manager independently re-reads strict main
protection and matches all 16 required check names to their exact application
IDs (GitHub Actions `15368`, CodeQL `57789`) on reviewed head
`fb4eb91324292bc8e159cebe11613aa75e07b911`. Every required context has exactly
one completed/success result; no failed, cancelled, timed-out or
action-required current-head check is present. The final macOS build
`113287545935` and test `113287546009` both succeed. Advisory `unity-package`
`113316887550` is still in progress at acceptance; this is not a claim that
the entire rollup is terminal. CI is `37770169016`, Security `37770168808`
and Quality `37770168555` at the accepted head.

Manager normally marks #197 ready and squash-merges with the exact-head guard,
without admin/force, branch deletion or protection changes. API readback proves
actual MERGED at 2026-10-08 12:48:04 UTC, with main commit
`f5902db5cc164a292d0095a30a1151cd075eee82`, parent `2da1b3fa`. Its complete tree
`75f8aa72f2ac6728b0cf9bd1c89bb5ab0b87b3c5` equals the independently reviewed
consolidation tree. A whole-tree diff-name comparison has no changed paths.
The maintainer main is fetched and fast-forwarded, then independently read
back clean and equal to origin/main.

Before closure, all seven originals' latest OPEN heads are read again and
match the heads in the scoped coverage review above. Fresh changed-path lists
also exactly match that review: 53 entries, with the same 44 identical and
nine individually reviewed subsequent changes inherited into accepted main.
There is no protected-path target. The original heads' ancestry belongs to
the reviewed consolidation head, not to squash main; equal accepted trees
and unchanged original scopes establish incorporation instead.

Each original receives a scoped incorporated comment retaining its model,
dependency/native/license, owner, independent-reference, CPU/Apple and
publication obligations. Actual CLOSED readbacks have null mergedAt and
mergeCommit; no original is claimed individually merged or model-complete.

| Original | Verified unchanged head | Actual closure, UTC |
|---|---|---|
| #150 | `373c853e` | 2026-10-08 12:49:54 |
| #158 | `c38ae4ed` | 2026-10-08 12:50:00 |
| #159 | `40653c7e` | 2026-10-08 12:50:12 |
| #173 | `ddcdd74a` | 2026-10-08 12:50:19 |
| #174 | `edc1e6d0` | 2026-10-08 12:50:26 |
| #176 | `501645ac` | 2026-10-08 12:50:33 |
| #180 | `f39d4a96` | 2026-10-08 12:50:40 |

### Final frozen-scope disposition audit

The 2026-10-08 12:51:46 UTC API audit enumerates every original PR, not only
the accepted subset. The denominator remains 20:

| Actual disposition | Original PRs | Count |
|---|---|---:|
| MERGED | #199, #209, #147, #197 | 4 |
| CLOSED as incorporated, not individually merged | #207, #208, #210, #195, #150, #158, #159, #173, #174, #176, #180 | 11 |
| OPEN/Draft with an explicit retained next condition | #152, #169, #182, #191, #198 | 5 |

Accepted main commits are #199 `edc2ab54`, #209 `bf428b40`, #147 `2da1b3fa`
and #197 `f5902db5`, with the actual acceptance evidence recorded in their
dated sections. The eleven closures retain null merge commits. Every retained
entry remains Draft; none receives an invented owner/legal or execution pass.

| Retained Draft | Current head | Evidence required before further acceptance |
|---|---|---|
| #152 | `fe1f5fe2` | Authoritative XCodec2 primary archive/build/RECORD/API and native/license evidence, supported upstream/checkpoint route, and exact owner disposition before independent real-weight validation. |
| #169 | `b3e9d5c1` | Its replacement responsibilities are in #182, not accepted main. Close only after #182 is accepted and fresh scoped coverage is verified. |
| #182 | `a6ff4022` | Diagnose the locked Torch wheel HTTP403 and complete the authenticated 41-package audit; full E2E/reference, streaming/KV lifecycle, CPU/Apple and voice/license/owner conditions remain. |
| #191 | `63cadf3c` | Supported patched upstream/dependency/native and owner scope plus independent real-weight PCM reference; an open upstream issue and synthetic scheduling tests are insufficient. |
| #198 | `5614a92c` | Current required link job `113287531384` actually fails on official California citations with HTTP403 after bounded throttling. Preserve citations and source receipts; obtain a genuinely passing required result and retain current-law/version, EU/applicability/counsel/owner holds. |

This closes the original normal-PR acceptance and Draft disposition work,
not the real models behind the retained or incorporated entries. The frozen
batch is not expanded to later bot arrivals such as #211–#213, or to the
58-row unresolved-model campaign. Those are separately planned work.

The remaining delivery requirement is to integrate accepted main into the
management-only branch, review its sole handoff-file delta, pass relevant
static/normal hook gates, publish its PR and obtain fresh required CI before
normal acceptance. Until that record is actually delivered, the overall
PR-handling goal is not declared complete. No new model download/execution,
broad local Cargo, paid worker, HF publication or owner/legal approval is
introduced by this disposition audit; the protected manifest remains outside
all reads and edits.

## Execution and finish conditions

1. Work on the dependency batch and desktop reconciliation in separate owned
   worktrees; the manager reviews every implementation diff.
2. Review and integrate #197's preparation without enabling blocked execution;
   carry every source family's remaining responsibility forward in main.
3. Review the independent Draft lanes and advance safe source/tooling work.
   External waits receive an explicit named blocker and next evidence request,
   not an invented approval or a compute retry.
4. Diagnose current-head CI failures from the actual logs. A transient external
   request failure is not a numerical/runtime verdict; neither is it CI PASS.
5. Merge only reviewed eligible PRs after required exact-head checks pass,
   reconcile duplicate dispositions, and leave local main clean and synced.
6. Record the accepted commits, fresh CI handles, closures and genuinely
   retained Draft gates before calling this PR-handling batch complete.

No maintainer-Mac model download/execution, workspace/model-crate Cargo, HF
upload or provider allocation has occurred in the actions recorded above.
The protected owner CosyVoice manifest is not read, hashed, edited or staged.
