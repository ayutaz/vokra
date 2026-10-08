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
