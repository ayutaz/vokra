# Security remediation plan (2026-09-21)

**Status:** active. This is the execution plan for the open Dependabot and
OpenSSF Scorecard findings after the `v0.3.0` GitHub release. It records the
current routing boundary; it is not evidence that an alert is fixed, and it
does not authorize model publication or a weaker numerical oracle.

## 2026-10-02 current security readback

A fresh all-pages Dependabot API read reports 209 open alerts: 182 with a
published patched version and 27 without one. This supersedes the older
queue counts below; no alert was dismissed or reference environment declared
numerically verified by this readback.

The current Code Scanning API reports three open Scorecard findings:
`CIIBestPracticesID`, `CodeReviewID` and `VulnerabilitiesID`. `MaintainedID`
is `fixed` since 2026-09-30 02:49:50 UTC and `SASTID` is `fixed` since
2026-09-30 05:07:01 UTC; neither has a dismissal record. The latest successful
[Scorecard run](https://github.com/ayutaz/vokra/actions/runs/36847827407)
evaluated main `97447185361a37af64c1b30fe87e8e2618d96e20` on 2026-10-01.
Root downloaded only its small `scorecard-results` artifact and inspected
the actual SARIF: it contains exactly those three remaining findings,
59 distinct vulnerability IDs, 0/29 independently approved changesets and no
Best Practices badge. The SARIF SHA-256 is
`e183baf1266029bf4b1e759bf84a195d7ff8caa6da6e62da5d1012c5c34a65b8`.
The corresponding [main CodeQL run](https://github.com/ayutaz/vokra/actions/runs/36842462753)
also succeeded at that exact head. These facts supersede the earlier five-open
snapshots, not the remaining remediation or independent-review requirements.

## 2026-09-22 execution update

PR #115 was squash-merged as
`1d9fd7de7f12762c543d8e478787b359ecc49917`, completing S0 and merging the
S1 JIT-containment layer. Dependabot alert #531 intentionally remains open
until the separate VAST source/import-closure and synchronized dependency
replacement evidence exists. A fresh read-only API snapshot still reports
275 open Dependabot alerts with the same severity and patched/unpatched totals
below.

[OpenSSF Scorecard run
#35670001603](https://github.com/ayutaz/vokra/actions/runs/35670001603)
re-evaluated clean `main` at that commit and uploaded SARIF successfully. The
overall score is 6.4 and the same five tracked findings remain open. The run
still reports 0/23 approved changesets, repository age below 90 days, no
OpenSSF Best Practices registration, CodeQL coverage on 26/30 recent commits,
and 76 distinct existing OSV vulnerability IDs. These are current evidence,
not a reason to dismiss or weaken any finding.

## 2026-09-28 execution update

Three bounded security updates were squash-merged after the 2026-09-22
snapshot:

- PR #118, `f87f55c4c688457891526a174d835d1fc7ac00e7`, synchronized
  `lightning` and `pytorch-lightning` at exact 2.6.6 in the nanocodec,
  pyannote-diarization, and pyannote-segmentation reference environments.
  The three remote checkpoint-regression jobs and the required PR checks
  passed. Dependabot alerts #532, #534, and #535 are fixed.
- PR #119, `bcd7de4a69969fe7ed644c4faf5a6f90d1bf15d7`, updated the UTMOS
  safe-state-dict environment to Torch 2.13.0 and made dependency-only changes
  trigger `parity-utmos`. The current-base model-free self-tests passed. The
  numeric job remained an explicit skip because authenticated tensor-only
  inputs were absent; no UTMOS numeric parity is claimed.
- PR #121, `107a49a923c3e32b8a3a2bed45c16fe29e5a6cbc`, synchronized the root
  `tools/parity` Lightning pair at exact 2.6.6. The fail-closed inventory guard
  rejected unrelated package/version/dependency-edge changes, dependency
  review and all required checks passed, and Dependabot alert #533 is fixed.

PR #120 tested the proposed nanocodec NLTK 3.10.3 replacement but was closed
without merging. Dependency review correctly rejected 3.10.3 because
GHSA-8mgp-746c-j5xp affects NLTK `<= 3.10.3` and no fixed release is
published. No allow-list entry or dismissal was added, the remote branch was
deleted, and the unpatched NLTK and Accelerate alerts remain visible.

After GitHub's configured dependency-graph update completed, the read-only API
snapshot contains 270 open alerts: 6 critical, 48 high, 101 medium, and 115
low. 238 name a first patched version and 32 still have none. The decrease from
the dated 275-alert baseline is recorded here rather than rewriting that
historical snapshot.

[OpenSSF Scorecard run
#36329299240](https://github.com/ayutaz/vokra/actions/runs/36329299240)
completed successfully on `main` at `107a49a9`. The same five code-scanning
findings remain open (`CIIBestPracticesID`, `MaintainedID`, `CodeReviewID`,
`SASTID`, and `VulnerabilitiesID`); none was dismissed to improve the score.

A later clean-main [Scorecard rerun
#36333491104](https://github.com/ayutaz/vokra/actions/runs/36333491104), after
the corresponding CodeQL run completed, reports SAST coverage on 29 of the 30
reviewed changesets. The remaining miss is not a failed or absent CodeQL run:
PR #114 head `1bb61c0d` has a successful GitHub Advanced Security CodeQL check,
but Scorecard v5.5.0 requests only the first 30 check suites and that CodeQL
suite is the 33rd suite for the commit. Its squash commit `cb51b7e6` was the
eighth of the 30 mainline changesets in this snapshot, so the finding will age
out only after 23 subsequent meaningful commits. Do not create empty commits,
rerun an already successful check, or dismiss the finding to change the score.
The other four finding classes remain unchanged.

### Later 2026-09-28 execution update

Five more bounded updates are merged on current `main`:

- PR #125, `ff14d1b65f1ec4731932120ec60b48dc924fcba3`, updated the
  production GitHub Actions group. The repository workflow-hygiene guard and
  its fixtures now require the same immutable `setup-uv` v10.2.0 commit used
  by the workflows.
- PR #126, `af188a42b88d011385d6bb34c3c789cdad6c2d07`, updated the AST
  reference environment to Torch 2.13.0 and Transformers 5.10.4. Exact-head
  VAST replay passed the existing real-GGUF CPU parity bounds and closed
  alerts #101-#104 and #174-#175. The disposable worker and storage were
  destroyed after evidence recovery.
- PR #127, `850ffbe370f29dd29233ba55bf6df31c22ad4ed7`, updated the
  Deepfake reference environment to Transformers 5.10.4. Exact-head VAST
  real-weight measurement passed without inventing a numerical gate and
  closed alerts #180 and #181. The disposable worker and storage were
  destroyed after evidence recovery.
- PR #128, `d70c8587367417f03c731f65408b43f88d425f73`, updated the T5
  encoder and MusicGen reference route to Transformers 5.10.4. Exact-head VAST
  CPU parity, delay-pattern checks, native T5 tests, and the Apple-target
  Metal-feature cross-check passed, closing alert #195. No artifact was
  published, and every temporary worker and volume was destroyed.
- PR #129, `a1a769c71d4b78482e204a3624cbff1108b36a83`, updated the
  Whisper-Medusa oracle to Transformers 5.10.4. Exact-head VAST parity passed
  at the unchanged `atol=5e-4`, the greedy token matched exactly, the
  Apple-target Metal-feature cross-check passed, and the disposable worker
  was destroyed. No artifact was uploaded. GitHub still reports lock alert
  #199 as open with `fixed_at=null`; keep it visible until the dependency graph
  rescans the 5.10.4 lock rather than dismissing it manually.

The read-only Dependabot API snapshot after these merges contains 261 open
alerts: 6 critical, 43 high, 101 medium, and 111 low. 229 name a patched
version and the same 32 still have no published fix, across 49 manifest files.
This is a new current snapshot; the 275- and 270-alert dated snapshots below
remain unchanged as historical evidence.

## Baseline and current inventory

The reviewed source baseline is clean public `main`
`0df21558c0a8f699a4b2b11c108f413ee21fc8c2`. The read-only GitHub API
snapshot on 2026-09-21 contains 275 open Dependabot alerts:

| Severity | Open alerts |
|---|---:|
| Critical | 6 |
| High | 53 |
| Medium | 101 |
| Low | 115 |

Of those alerts, 243 name a first patched version and 32 do not. All are in
Python dependency manifests. 274 belong to isolated `tools/parity/**`
reference environments; one belongs to the opt-in, out-of-workspace
`integrations/vokra-misaki-g2p` bridge. None changes the root runtime's
first-party-only `Cargo.lock` invariant, but an isolated oracle is still
security-sensitive and must not be treated as disposable evidence.

The 275 alerts cover 58 manifest files and 46 logical Python environments:

| Package | Alerts | Patched version named | No patched version |
|---|---:|---:|---:|
| `torch` | 189 | 162 | 27 |
| `transformers` | 29 | 29 | 0 |
| `nltk` | 21 | 19 | 2 |
| `onnx` | 20 | 20 | 0 |
| `diffusers` | 6 | 6 | 0 |
| `lightning` | 5 | 5 | 0 |
| `accelerate` | 3 | 0 | 3 |
| `modelscope` | 2 | 2 | 0 |

The 32 alerts without a published fixed version are not to be bulk-dismissed:

- 27 Torch alerts across `chatterbox_t3`, `cosyvoice2_reference`,
  `cosyvoice3_reference`, `dia_1_6b_reference`,
  `owsm_v4_medium_1b_reference`, `speecht5_tts`, and `xcodec2`;
- three Accelerate alerts across `nanocodec`, `neutts_air`, and `ultravox`;
- two NLTK alerts across `nanocodec` and
  `integrations/vokra-misaki-g2p`.

## Execution rules

1. One logical reference environment is the default update and review unit.
   Do not combine all 275 alerts into one lockfile wave.
2. Use the environment's `pyproject.toml` and `uv.lock`; all Python commands
   run through `uv` with Python 3.12. Do not use bare Python, pip, or conda.
3. A dependency bump is not complete when the lock resolves. Re-run its
   independent source/API smoke and numerical reference/parity contract.
   Torch, Transformers, Diffusers, ONNX, Lightning, and ModelScope changes
   require remote numerical evidence before acceptance.
4. Model downloads, real-weight execution, workspace or `vokra-models` Cargo,
   and artifacts of at least 2 GB run only on disposable VAST workers. Recover
   small evidence and destroy the worker and storage afterward.
5. Never read, edit, stage, revert, or clean
   `tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json`.
6. Preserve fixed upstream revisions and preregistered numerical bounds. A
   dependency update must not silently replace the independent oracle, loosen
   tolerances, or convert a blocked result into a pass.
7. Alerts without a fixed version remain open until an upstream fix, dependency
   removal, or an explicit reviewed risk disposition exists. Isolation alone
   is not a vulnerability fix.

## Ordered remediation waves

### S0 — synchronize public status documentation

Update the current documentation to record the published `v0.3.0` release,
the current M5 `53 checked / 29 unchecked` count, and the completed Qwen3-TTS
VAST CPU-parity leg. Keep dated historical counts unchanged. Synchronize the
English/Japanese security-policy pair so it no longer claims that no release
exists.

Exit gate: documentation references, owner-checklist drift, community-doc
parity, workflow hygiene, and diff hygiene pass.

### S1 — disposition the existing Zonos update safely

[Dependabot PR #112](https://github.com/ayutaz/vokra/pull/112) was the first
bounded update. It changed only:

- `tools/parity/zonos_v0_1_reference/pyproject.toml`;
- `tools/parity/zonos_v0_1_reference/uv.lock`.

It updated Torch `2.11.0` to `2.13.0` for a patched advisory and was mergeable
but behind `main`. Its diff left Torchaudio at `2.11.0`, however, while
the checked-in Zonos environment deliberately requires a synchronized
Torch/Torchaudio pair. The old-base CI does not exercise the independent
reference environment deeply enough to approve that mixed pair. Do not rebase
or merge PR #112 as written. It was closed on 2026-09-21 with the synchronized
pair and remote-evidence requirements recorded in the closing rationale.

The advisory is limited to `torch.jit.script`. This branch now makes both the
model-free compatibility probe and the full reference worker refuse that API
fail closed. The full-worker guard spans official-source import through
evidence output, detects guard replacement, restores the original callable,
and emits a hash-bound `jit_script_guard` record that the inspector requires.
Model-free self-tests cover refusal, restoration, guard drift, missing and
tampered evidence, the compatibility wrapper, and the complete inspection
wrapper. Keep the alert visible after this containment merges: a disposable
VAST worker must still establish source/import closure and test a fully
compatible Torch/Torchaudio replacement (or justified Torchaudio removal).
Only after that evidence is reviewed may the alert receive a narrowly reasoned
`vulnerable_code_not_present` disposition; do not manufacture an unsupported
mixed install.

Exit gate: PR #112 is replaced or closed rather than merged unchanged; the
affected API is fail-closed with a regression test; any dependency replacement
has current-base source/API and independent VAST evidence; recovered checksums
and worker destruction are recorded.

### S2 — patched-only environment batches

After S1, process patched-only environments separately. Start with small,
single-purpose environments such as `ast`, `moss_audio`,
`vibevoice_1_5b_reference`, and the Transformers-only `bark`,
`deepfake_detection`, `t5_encoder`, and `whisper_medusa` projects. Keep
Torch-only environments separate and exclude the seven no-fixed-version Torch
environments from automatic update waves.

Each batch must record the alert numbers it closes, old and new locked
versions, immutable upstream source identity, reference/API compatibility,
numerical results, exact source head, and cleanup result. Stop and split the
batch when an API or numerical result changes.

Exit gate: the GitHub API confirms the intended alerts closed and no new alert
was introduced for that environment.

### S3 — no-fixed-version and mixed environments

Keep the 32 no-fixed-version alerts visible. Re-check upstream advisories and
available releases after each patched-only wave. For a mixed environment,
avoid a partial update that produces an unreviewable oracle state; either
prove the complete locked closure or defer it with the exact unresolved
advisory recorded.

The opt-in Misaki bridge does not call NLTK's affected model-artifact
serialization APIs, but that usage fact is not a patched dependency. Continue
to track the upstream fix unless a separate owner/security decision authorizes
a narrowly reasoned disposition.

## OpenSSF Scorecard routing

The five historical findings and their remediation routes are listed below.
The 2026-10-02 readback above identifies which three remain open on `main`;
the older causes here are retained as dated context, not current verdicts.

| Alert | Current cause | Resolution route |
|---|---|---|
| `CIIBestPracticesID` | No OpenSSF Best Practices registration | Register the project and complete the external questionnaire; add a badge only after the external record exists. |
| `MaintainedID` | Repository was created within the last 90 days | Keep active maintenance and re-run after the 90-day boundary, approximately 2026-10-01. |
| `CodeReviewID` | 0 of 23 recent changesets have an approval | Adopt at least one independent approving reviewer before changing branch protection from zero required approvals; do not create an owner self-approval fiction. |
| `SASTID` | CodeQL covered 26 of the latest 30 commits | Keep CodeQL required on PRs and `main`; this clears as the four older unscanned commits leave the rolling window. |
| `VulnerabilitiesID` | Scorecard detected vulnerable Python dependencies | Reduce through S1-S3, then re-run Scorecard and inspect the new OSV set. |

The existing Scorecard workflow is already SHA-pinned, least-privilege,
scheduled, run on `main` pushes and branch-protection changes, and uploads
SARIF. The existing CodeQL workflow runs on pull requests, `main`, schedule,
and manual dispatch, and `CodeQL` is a required branch-protection check. Do not
dismiss the five findings merely to improve the score.

## Completion proof

This plan is complete only when:

- all 243 currently patchable alerts are closed by reviewed dependency changes
  with environment-specific evidence, or a later API snapshot proves that the
  upstream advisory no longer applies;
- every alert without a fixed version has a current upstream status and an
  explicit, evidence-backed disposition without bulk suppression;
- Scorecard is re-run after dependency remediation and the remaining four
  process findings reflect their actual external, historical, or review-policy
  state;
- the current public documentation and English/Japanese security policies
  agree with the released repository state; and
- the final worktree is reviewed, relevant gates are green, remote workers and
  storage are destroyed, and no model was executed on the maintainer Mac.
