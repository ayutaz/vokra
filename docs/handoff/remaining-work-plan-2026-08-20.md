# Remaining-work execution plan (2026-08-20)

This is the execution order derived from the live repository, merged PR #38, GitHub
settings, the M5 owner checklist, and the platform-support matrix. It is a
route map, not a completion claim. A checked historical ticket or a green
gated-skip does not replace the evidence named below.

## Completion boundary

The remaining work is complete only when all of the following are true:

1. The PR #38 CI/branch-protection reconciliation remains intact on `main`.
2. Release surfaces advertised as usable match the current 41-function C ABI;
   package jobs produce installable, tested artifacts rather than scaffolds.
3. Every enabled real-weight family records an independent numerical verdict;
   a disabled repository variable or clean skip is not a pass.
4. Every published model passes the repository publication chain and live
   verification; unpublished models retain an explicit withholding decision.
5. Platform and hardware claims have measurements from the required real
   target, not cross-build or emulation evidence substituted for hardware.
6. M5 legal, commercial, DoD, and ABI-freeze gates have owner evidence. The
   v1.0.0 tag is last; it must not be used to manufacture completion.

## Phase A — PR #38 reconciliation (complete)

| Item | State on 2026-08-20 | Evidence required to close |
|---|---|---|
| Retire obsolete branches | Complete: 13 old refs were audited, archived in a verified local Git bundle, and obsolete remote branches were deleted | Archive/delete evidence from the working session |
| Unique required context ownership | Complete: Vulkan is `vulkan-parity`; main CI exclusively owns required `parity` | Final PR run contained distinct contexts |
| PR check fanout | Complete: dynamic `nightly-full-parity` planning and prose-only workflow filters landed | Final-head workflow checks |
| Branch-protection documentation | Complete: 14 required contexts and four security contexts are current | Branch-protection API and workflow index |
| Workflow inventory | Complete for PR #38 at 44 workflows / 36 cron entries; this Python branch intentionally adds workflow 45 without adding a cron | `scripts/check-workflow-hygiene.sh` reports 45 / 36 on this branch |
| M5 tracking | Complete for the reconciliation scope: live routing index added and stale Accepted decisions corrected | Documentation gates in PR #38 |
| Final verification | Complete and merged | PR #38 had 110 checks (99 success, 11 intentional skips, zero failures) and was squash-merged as `234d368` |

PR #38 is no longer a pending dependency. New work must remain on separate,
reviewable branches and preserve the merged required-check ownership.

## Phase B — make the release surface truthful

1. **Python binding / wheel (active).** The separate local branch
   `agent/python-bindings-capi-2026-08-20` is rebased onto merged PR #38 and extends
   `gen-py-bindings.py` to parse the current C structs, four enum families,
   seven opaque handles, integer widths, plain-`bool` and struct-pointer
   returns, and all 41 functions. Its uv tests pass on Python 3.9 and 3.12.
   This branch also replaces unsafe CI-artifact retag/reuse with a reusable
   same-run native build: pinned manylinux 2.28 + auditwheel on Linux, separate
   arm64/x86_64 macOS builds + delocate, and Windows x86_64 + delvewheel. Each
   wheel is `py3-none-<platform>`, archive/RECORD/native architecture checked,
   clean-installed on Python 3.9/3.12, and collected into an exact four-wheel
   manifest. Local static/source gates are green; GitHub native jobs and a VAST
   Rust verification remain required before the branch is ready to land.
2. **Desktop distribution.** Replace the T32-gated scaffold with native
   Windows/macOS/Linux library + CLI builds, a completeness manifest, install
   smoke, and fail-loud publication. A Linux-only best-effort archive is not a
   three-platform release.
3. **Android distribution.** Replace the manifest + arm64 `.so` scaffold with
   a real standalone AAR containing the required helper/classes.jar, declared
   ABI set, Gradle consumer test, and fail-loud completeness gate; keep Android
   real-device RTF separate from cross-build success.
4. **Godot distribution.** Require every advertised platform slice before
   publishing the AssetLib zip; crossbuild reuse may not be best-effort at a
   release tag.
5. **Registry channels.** Exercise dry-runs first for PyPI, crates.io, npm,
   OpenUPM/Godot AssetLib as applicable. Tokens, project reservations, and
   production uploads require exact owner authorization.

Each implementation belongs in a focused branch after PR #38. The active
Python branch is the first such release-surface branch.

## Phase C — real parity and runtime depth

Execute in increasing cost order:

1. Activate the already-proven scheduled legs for DeepFilterNet3 and
   DeBERTa-v3-large after owner approval of the repository variables.
2. For NeMo ASR, Whisper extras, TTS DAC, HiFTNet family, Qwen3-TTS,
   continuous-VAE TTS, and Japanese TTS, generate independent upstream
   reference outputs, implement any missing native consumer, and record a
   PASS/FAIL per family. Structural GGUF checks are not numerical parity.
3. Close the five explicit implementation follow-ups: Charsiu checkpoint
   binding, microWakeWord emitted quantization metadata + real fixture, native
   BF16 compute, full HiFTNet GPU generator, and full BigVGAN GPU path.
4. Keep Matcha-TTS dormant until its three documented triggers are met; do not
   create a converter merely to reduce an unchecked count.

Explicit non-checklist implementation holes also remain:

- implement Godot `session_vad_open_stream` object creation and add headless +
  editor smoke evidence before claiming full runtime dispatch;
- replace the flow-sampler fixture-triggered panic with an independent
  reference decoder/comparison;
- parse and compare the openWakeWord reference JSON against a real fixture;
- finish real-checkpoint DeBERTa-v2 tensor mapping (distinct from the completed
  DeBERTa-v3 scheduled parity);
- decide whether vLLM completion generation is in the GA server scope; until
  implemented, retain the explicit contract-only/501 wording at top level.

Read the numerical-parity skill before writing a reference dumper, changing a
bound, or diagnosing a parity failure. All model artifacts totaling at least
2 GB and every `vokra-models`/workspace Cargo command run on VAST.

## Phase D — publication and destination decisions

1. Decide whether OpenVoiceV2, knn-vc, FreeVC, and MeanVC belong in the
   separate experimental repository or are rejected from public distribution.
2. Complete license/provenance/destination records before any upload.
3. Correct the live Voxtral-Small-24B repository using only
   `publish-one.sh`; the stopped VAST instance `47955178` is the retained
   artifact source as of this plan. Rotate the exposed VAST API credential
   before reuse, live-verify the corrected repo, then destroy the retained
   instance/volume. If publication is declined, record withholding and destroy
   it instead of paying indefinitely for retained storage.
4. Verify LICENSE, NOTICE where required, SOURCE.md, and GGUF producer/schema
   provenance for every uploaded repository.

HF credential transfer and `--push` are irreversible publication actions and
require approval for the exact repository/artifact even when conversion or
dry-run work was already approved.

## Phase E — hardware, legal, and GA critical path

The dependency order is fixed:

1. Capture the M5-14/M5-15 same-rig CPU, quantization, and quality evidence.
2. Run CoreML/ANE and QNN/Hexagon placement + RTF bakeoffs against matched CPU
   baselines; record PASS/FAIL/INSUFFICIENT DATA.
3. Decide delegate and WFST C exports; ratify the WFST ADR.
4. Run Cortex-M55/FVP Silero VAD and decide Tier-3/Helium investment.
5. Complete iOS, Android, Web, Godot, server-latency, and self-hosted CUDA
   measurements required by their owning acceptance criteria.
6. Complete console NDA/SDK verification, voice-clone legal/trust-root work,
   SynthID-contract-or-alternative decision, EU certification, and commercial
   adoption evidence.
7. Supply the X-05-T04 owner contact points and land the four pending
   `CODE_OF_CONDUCT` / `SECURITY` English/Japanese community files.
8. Stabilize the release train and community/maintainer DoD inputs.
9. Promote `abi-surface` only after a green observation window, record all
   M5-12 DoD evidence, then prepare v1.0.0 and fire the irreversible ABI freeze.

## Continuous controls

- Python always runs through uv.
- Never run workspace-wide Cargo or compiling/testing `-p vokra-models` on the
  maintainer Mac.
- Never convert, validate, or publish aggregate model artifacts of 2 GB or more
  on the Mac.
- Never interpret a scheduled clean skip, scaffold test, synthetic fixture, or
  converter-only check as real-weight/native parity.
- Never publish or change legal/license decisions without the named owner
  authorization and primary-source evidence.

## Operational cleanup that must not be mistaken for product work

- The running VAST `piper-v11` instance was migrated on 2026-08-20 from a
  bare `python3 -m venv`/pip bootstrap to an uv-managed environment synced
  from the pinned piper-plus `uv.lock`; the resume-capable dataset download
  continues. Preserve its checkpoint/dataset evidence, then destroy the
  instance after the smoke/train lifecycle finishes.
- Inventory the five stopped VAST volumes before destruction. The Voxtral
  volume has an explicit publication/withhold decision path; the older Piper
  and Moshi volumes need an artefact manifest first, while the PR #38 verify
  volume should be checked for unique evidence before requesting destruction.
- Three local stashes include entries likely superseded by later merged PRs, but none is
  deletion-safe yet. Compare evolved hunks semantically against PRs #27/#28
  and #8, then request explicit destructive approval before dropping them.
- GitHub CLI is authenticated as `ayutaz`. Rotate the VAST API credential that
  appeared in prior command output before further credential-sensitive use. Never
  record replacement credentials in this repository or a VAST `.env` file.
