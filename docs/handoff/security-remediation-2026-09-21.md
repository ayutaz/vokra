# Security remediation plan (2026-09-21)

**Status:** active. This is the execution plan for the open Dependabot and
OpenSSF Scorecard findings after the `v0.3.0` GitHub release. It records the
current routing boundary; it is not evidence that an alert is fixed, and it
does not authorize model publication or a weaker numerical oracle.

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

The advisory is limited to `torch.jit.script`. The model-free Zonos
compatibility probe already refuses checkpoint construction and
`torch.jit.load`; first add and test an explicit fail-closed refusal for
`torch.jit.script`. Keep the alert visible until that containment is merged
and a security review confirms that the affected API is absent from every
authorized Zonos worker. Only then may the alert receive a narrowly reasoned
`vulnerable_code_not_present` disposition. In parallel, test a fully
compatible Torch/Torchaudio upgrade or the justified removal of Torchaudio on
a disposable VAST worker; do not manufacture an unsupported mixed install.

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

Five process alerts are open on `main`:

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
