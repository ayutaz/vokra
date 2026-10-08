# PR review and disposition plan — 2026-10-08

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
