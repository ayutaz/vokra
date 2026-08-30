# Remaining-work execution plan (2026-08-20)

> **2026-08-30 current-state notice:** this is a dated execution plan, not the
> current ledger. Its historical counts, branch/HEAD references and completion
> boundaries are superseded by the authenticated current snapshot in
> `docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md` and the live
> `docs/m5-owner-verification-checklist.md`. The pre-documentation
> implementation/code baseline observed on 2026-08-30 was branch
> `feat/mac-cpu-metal-full-coverage-2026-08-28` at
> `c64b7b7237b70c5dc70ffd60394af325016d9a8d`, workspace `0.2.0`, with
> C ABI 57 functions / 15 typedefs and M5 49 checked / 33 unchecked. The
> GitHub `main` reference remains `41ce9ffdd4b0959497f55afa5016822f77a8a7b6`.

> 2026-08-21 refinement: the runtime portion of Phase C is superseded by
> `docs/handoff/runtime-gap-execution-plan-2026-08-21.md`. The newer audit
> corrects the stale DeBERTa-v2 mapping item, records the public RNNoise
> opaque-blob issue, and groups all 79 blocker rows into executable waves.

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
| Branch-protection documentation | Complete: 15 required contexts (core 10 + security 4 + pins-sync 1), strict checks, conversation resolution, linear history, administrator enforcement, and Actions SHA pinning are current | Branch-protection and Actions-policy APIs plus workflow index |
| Workflow inventory | Complete for PR #38 at 44 workflows / 36 cron entries; this Python branch adds workflow 45 and promotes its secret scan with one nightly full-history cron | `scripts/check-workflow-hygiene.sh` reports 45 / 37 on this branch |
| M5 tracking | Complete for the reconciliation scope: live routing index added and stale Accepted decisions corrected | Documentation gates in PR #38 |
| Final verification | Complete and merged | PR #38 had 110 checks (99 success, 11 intentional skips, zero failures) and was squash-merged as `234d368` |

PR #38 is no longer a pending dependency. New work must remain on separate,
reviewable branches and preserve the merged required-check ownership.

## Phase B — make the release surface truthful

1. **Python binding / wheel (active).** The separate local branch
   `agent/python-bindings-capi-2026-08-20` extends
   `gen-py-bindings.py` to parse the current C structs, four enum families,
   seven opaque handles, integer widths, plain-`bool` and struct-pointer
   returns, and all 41 functions. Its uv tests pass on Python 3.9 and 3.12.
   This branch also replaces unsafe CI-artifact retag/reuse with a reusable
   same-run native build: pinned manylinux 2.28 + auditwheel on Linux, separate
   arm64/x86_64 macOS builds + delocate, and Windows x86_64 + delvewheel. Each
   wheel is `py3-none-<platform>`, archive/RECORD/native architecture checked,
   clean-installed on Python 3.9/3.12, and collected into an exact four-wheel
   manifest. GitHub's four native jobs are green. VAST independently built and
   tested `vokra-capi`, then clean-installed an actual Linux native wheel on
   Python 3.9 and 3.12 and resolved all 41 symbols. The branch is rebased onto
   the post-PR #39/#40/#43 `main`. The implementation head's CI is green (72
   checks: 71 success, one
   intentional full-history skip, zero failures); later documentation-only
   heads retain the same required-check coverage. Review is the remaining
   integration work. The branch promotes
   `gitleaks` after 18/18 green main runs and adds a nightly full-history scan.
   PyPI/TestPyPI publication remains out of scope.
2. **Desktop distribution.** Replace the T32-gated scaffold with native
   Windows/macOS/Linux library + CLI builds, a completeness manifest, install
   smoke, and fail-loud publication. A Linux-only best-effort archive is not a
   three-platform release.
3. **Android distribution and binding depth.** Replace the manifest + arm64
   `.so` scaffold with a real standalone AAR containing the required
   helper/classes.jar, declared ABI set, Gradle consumer test, and fail-loud
   completeness gate. The current raw-JNI crate exposes only five lifecycle /
   error entry points; ratify JNA vs raw JNI, then add ASR, TTS, VAD,
   streaming, AEC, and S2S wrappers, the `AssetManager` → `filesDir` helper,
   coroutine wrappers, and Maven publication. Keep Android real-device RTF
   separate from cross-build success.
4. **Godot distribution.** Require every advertised platform slice before
   publishing the AssetLib zip; crossbuild reuse may not be best-effort at a
   release tag.
5. **Registry channels.** Exercise dry-runs first for PyPI, crates.io, npm,
   OpenUPM/Godot AssetLib as applicable. Tokens, project reservations, and
   production uploads require exact owner authorization. As of 2026-08-20 the
   repository has no repo-level Actions secrets and no environments, so the
   external registry tokens and Unity license referenced by workflows are not
   configured here (the automatic `GITHUB_TOKEN` is separate).
6. **Dashboard hosting.** Decide whether the generated performance dashboard
   should be public. The artifact path already works; public deployment still
   requires enabling GitHub Pages and setting `VOKRA_PAGES_ENABLED=true`.

Each implementation belongs in a focused branch after PR #38. The active
Python branch is the first such release-surface branch.

## Phase C — real parity and runtime depth

The runtime inventory is mechanically complete but is not a claim that every
model executes. On 2026-08-20 `check-bound-arch-coverage.sh` accounted for all
89 distinct binder architectures (14 directly routed and 75 represented by 79
`BOUND_ARCHES` rows), `check-arch-handshake.sh` reconciled 111 converter and 89
binder discoveries plus 570 required metadata-key reads, and
`check-m5-residual-blockers.sh` passed. The honest blocker ledger in
`crates/vokra-cli/src/engine.rs` currently splits the 79 unrouted rows into:

| Blocker class | Rows | Next action |
|---|---:|---|
| `RealForwardNoCliTask` | 3 | Wire `wetextprocessing` normalization (with `vokra-wfst`), FCPE extraction, and CREPE extraction into explicit CLI tasks and tests |
| `NeedsPairedInput` | 1 | Add a paired mic/reference input contract before routing NKF-AEC |
| `NoCliShapedOutput` | 2 | Define an honest presentation/input contract for Mimi codes and CT-Punc token/id input, or keep them library-only |
| `NoGgufLoader` | 17 | Add real artifact loaders before advertising CLI execution: Parakeet-TDT, CosyVoice3, Chatterbox (three variants), Dia, Irodori-TTS, Qwen3-TTS, VibeVoice, VITS-JA, VoxCPM2, Zonos, BigVGAN, Vocos, HiFi-GAN vocoder, SpeechT5 HiFi-GAN, and Charsiu |
| `LoudPartialForward` | 56 | Implement and independently parity-test the primitive named by each binder; do not collapse these into one checkbox or count a loader/converter as a forward |

Treat the first three CLI adapters as the lowest-cost code slice. The 17
loader and 56 forward rows need model-family scope, upstream references, and
real artifacts; they must not be bulk-marked complete from structural tests.

Execute in increasing cost order:

1. Activate the already-proven scheduled legs for DeepFilterNet3 and
   DeBERTa-v3-large after owner approval of the repository variables. The
   repo-level Actions Variables list is empty as of 2026-08-20, so these and
   every other `vars.*`-guarded real-parity / Pages / self-hosted leg currently
   clean-skip; enabling one requires supplying its complete artifact/runner
   contract rather than only flipping an enable flag.
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

- ~~implement Godot `session_vad_open_stream` Object creation and add headless
  smoke evidence~~ — **closed 2026-08-22**. The official Godot 4.7.1 gate now
  checks the Object return, real Silero load, push/poll, interrupt drain, and
  deterministic reset. Interactive editor demo confirmation remains a manual
  release check rather than an implementation gap;
- implement a real `TtsEngine::synthesize_stream` override before advertising
  incremental streaming; the trait default intentionally returns
  `UnsupportedOp`, so a one-chunk synchronous wrapper is not completion;
- replace the flow-sampler fixture-triggered panic with an independent
  reference decoder/comparison;
- parse and compare the openWakeWord reference JSON against a real fixture;
- finish real-checkpoint DeBERTa-v2 tensor mapping (distinct from the completed
  DeBERTa-v3 scheduled parity);
- replace the RNNoise opaque-blob checkpoint prep with the Xiph v0.2 per-layer
  split, bind a real GGUF runtime, and promote the env-gated full-denoise test
  from its intentional panic marker to independent C-reference waveform
  parity;
- implement PyIN HMM/Viterbi temporal smoothing. The current
  `viterbi_smooth_todo` deliberately returns the framewise pitch unchanged;
- stamp and strictly read the Conv-TasNet topology chunk group instead of
  relying on transcribed runtime constants;
- replace JASCO's provisional chord/drum vocabulary and sampler defaults with
  values pinned from the upstream AudioCraft config before any real artifact
  claim;
- resolve the SBV2 language-row ordering and spline `num_bins` value with
  checkpoint/config evidence, then finish production Mandarin segmentation /
  word-boundary handling; the completed ZH numerical fixture does not close
  production G2P;
- verify the Windows NVRTC DLL suffix list on a real Windows CUDA image before
  claiming that the dynamic CUDA loader covers that platform;
- decide whether vLLM completion generation is in the GA server scope; until
  implemented, retain the explicit contract-only/501 wording at top level.
- decide the GA compatibility boundary for the other explicit server 501s:
  OpenAI segment timestamps (`verbose_json`, SRT, VTT), compressed/headerless
  speech formats and per-request speed, plus Piper per-request overrides.
  Preserve fail-loud responses for every feature kept out of scope.

The marker audit found no executable `todo!()` or `unimplemented!()` macro in
production code; the three literal hits are historical RED-phase prose in
tests. That does not make the runtime complete: the explicit error paths,
placeholder metadata, and deferred helper bodies above are the actionable
forms used by this repository.

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
2. Implement the delegate execution prerequisite before scheduling NPU
   measurements. `vokra-backend-coreml` and `vokra-backend-qnn` currently
   report zero supported ops: non-empty graphs return `UnsupportedOp`, while
   `execute` on an empty graph is `NotImplemented`. Ratify the CoreML model-
   supply ADR, transcribe the QNN SDK-gated graph path, bind the selected
   delegate submodel, and add fail-loud graph/parity tests.
3. Run CoreML/ANE and QNN/Hexagon placement + RTF bakeoffs against matched CPU
   baselines only after that execution path exists; record
   PASS/FAIL/INSUFFICIENT DATA.
4. Decide delegate and WFST C exports; ratify the WFST ADR.
5. Run Cortex-M55/FVP Silero VAD and decide Tier-3/Helium investment.
6. Complete iOS, Android, Web, Godot, server-latency, and self-hosted CUDA
   measurements required by their owning acceptance criteria.
7. Complete console NDA/SDK verification, voice-clone legal/trust-root work,
   SynthID-contract-or-alternative decision, EU certification, and commercial
   adoption evidence.
8. Supply the X-05-T04 owner contact points and land the four pending
   `CODE_OF_CONDUCT` / `SECURITY` English/Japanese community files.
9. Confirm that GitHub Sponsors is configured for `@ayutaz`, or disable/remove
   `.github/FUNDING.yml` until it is; do not leave the repository funding
   surface in an owner-unverified state.
10. Stabilize the release train and community/maintainer DoD inputs.
11. Promote `abi-surface` only after a green observation window, record all
   M5-12 DoD evidence, then prepare v1.0.0 and fire the irreversible ABI freeze.

CI quality debt remains independently visible: `rustdoc (advisory)` currently
uses a 266-warning ceiling rather than a zero-warning gate. Drain that baseline
in a dedicated documentation-hygiene WP, ratchet the ceiling down as warnings
are removed, and only then consider required-check promotion. The gitleaks
advisory window is no longer debt: this branch records 18/18 green main runs,
promotes the working-tree scan, and adds the nightly full-history companion.

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
  from the pinned piper-plus `uv.lock`. Its 300,443-entry dataset and F0 cache
  are complete, and the v11 H-src-r2 smoke is the active retained evidence.
  Preserve its checkpoint/dataset evidence, then destroy the instance after
  the smoke/train lifecycle finishes.
- The live VAST inventory is one running instance (`48184676`, `piper-v11`)
  plus five **stopped instances with attached disks**, not five detached
  volumes; `vastai show volumes` returns an empty list. Before requesting
  destruction, inventory the stopped Voxtral (`47955178`), Piper v10b
  (`48000459`), Moshi (`48178589` / `48187958`), and PR #38 verification
  (`48186199`) disks. The Voxtral disk has an explicit publication/withhold
  decision path; the older Piper/Moshi disks need artifact manifests, and the
  PR #38 disk must be checked for unique evidence.
- The three local stashes were compared against the current Python PR head and
  `main`. `stash@{0}` has all 17 paths accounted for (10 byte-identical, seven
  evolved by the native-wheel follow-ups); `stash@{1}` has all 307 paths
  accounted for (181 byte-identical, 126 evolved through PRs #27/#28 and later
  fixes); `stash@{2}` has all 10 server paths evolved through PR #8 and later
  server work. No stash-only path is missing from the current trees. They are
  deletion candidates, but dropping them remains a destructive action that
  requires explicit approval.
- GitHub CLI is authenticated as `ayutaz`. Rotate the VAST API credential and
  any active instance access/Jupyter token exposed by raw CLI inventory output
  before further credential-sensitive use. Never record replacement
  credentials in this repository or a VAST `.env` file.
