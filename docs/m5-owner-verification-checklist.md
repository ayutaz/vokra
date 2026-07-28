# M5 (v1.0 GA) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — real-hardware verification, real-weight sourcing, legal sign-off, external contracts / infra provisioning, ADR ratification, and the v1.0 GA tag decision.

**CC-side status (2026-07-21)**: this checklist covers the owner tasks left by the M5 WPs whose CC-side work has landed on branch `feat/m5-plan-and-wave1`. It is the input to the **v1.0 GA** decision (commercial GA + C ABI freeze). It is NOT a GA declaration and NOT a freeze — the freeze FIRES at the owner's v1.0 GA tag (M5-13).

**Tracking**: this file (`docs/m5-owner-verification-checklist.md`) is **tracked (public)**, same convention as `docs/m3-` / `docs/m4-owner-verification-checklist.md`. Referenced handoffs `docs/handoff/m5-*.md` are tracked/public; specs `docs/tickets/m5/*.md` and ADRs `docs/adr/M5-*.md` are gitignore-local internal docs (referenced by ID).

Each task: **(a)** what / **(b)** why owner-only / **(c)** reference / **(d)** done-when.

---

## 1. M5-13 — C ABI freeze firing (the one-way v1.0 GA action)

The freeze machinery is landed (`abi-diff.sh --gate`, proven to fail on a blocking delta). Firing it is owner-gated. See `docs/handoff/m5-13.md` §(c) for the full procedure.

### 1.1 T17 — fire the v1.0 GA tag (= freeze trigger + M5 close)

- **(a)**: after M5-01…M5-12 complete and the NPU bakeoff (1.5) decides the delegate question, tag the GA commit (must carry `version = "1.0.0"`); roll `CHANGELOG.md [Unreleased]` → `[1.0.0]` as the tag-preparation step.
- **(b)**: an owner milestone decision, not a WP deliverable; needs the real-hardware bakeoff.
- **(c)**: `docs/handoff/m5-13.md` §(c) T17; spec M5-13-T17.
- **(d)**: a v1.0.0 GA tag exists on a `version = "1.0.0"` commit; CC then runs the freeze-firing sequence (handoff §(d)) against it.

### 1.2 T18 — promote the ABI gate to a required check

- **(a)**: after `abi-diff.sh --gate` runs green in CI for a stretch, register the `abi-surface` job as a required branch-protection context and drop its `continue-on-error`.
- **(b)**: branch-protection contexts are a repo-admin action.
- **(c)**: `docs/handoff/m5-13.md` §(c) T18.
- **(d)**: the ABI gate is required; a PR adding an unrecorded/breaking C symbol is blocked. (The gate's teeth are proven by the T14a negative test — not an empty gate.)

### 1.3 T19 — GO/NO-GO on the C-export candidates

- **(a)**: decide per candidate whether it is a frozen C symbol: (1) NPU delegate selector (integrate the M5-01/M5-02 bakeoff verdicts), (2) `wfst_decode` (M5-06 delegated the C-export call here).
- **(b)**: the delegate decision needs the bakeoff; a frozen C symbol's trust/scope is an owner call.
- **(c)**: `docs/handoff/m5-13.md` §(c) T19; `docs/handoff/m4-12.md` §(f)-4.
- **(d)**: a recorded GO/NO-GO for **both**. "Do not decide" is not allowed. NO-GO is recoverable post-GA via an additive MINOR bump.

## 1.5 NPU bakeoff (M5-01 CoreML/ANE + M5-02 QNN/Hexagon)

- **(a)**: run the CoreML (Apple ANE) and QNN (Qualcomm Hexagon) delegates on real hardware and measure the NFR-PF-12 acceptance criterion (≥2× over the CPU baseline). Feeds T19.
- **(b)**: needs real ANE / Hexagon silicon; this machine has neither an NPU bakeoff rig nor the delegate runtimes.
- **(c)**: spec M5-01-T24 / M5-02-T12 (gitignore-local).
- **(d)**: a pass/fail vs the 2× bar is recorded for each delegate.

---

## 2. M5-03 — IoT Tier 3 (Cortex-M55 no_std Silero VAD)

CC landed the no_std subset + `vokra-vad-micro` crate + thumbv8m cross-build + host-executable bit-identical differential + memory budget. See `docs/handoff/m5-03.md`.

### 2.1 T02 — ratify the crate-topology ADR

- **(a)**: ratify `docs/adr/M5-03-iot-tier3-nostd.md` (Status=Proposed): topology (案1 new `vokra-vad-micro` crate is CC's proposed default, vs 案2 in-place feature-gate), the all-target transcendental unification, the sqrt route (Newton default vs `asm! vsqrt`), and the Helium investment (scalar default vs raw-asm MVE).
- **(b)**: an architecture decision with a large downstream cost (案2 is a large refactor); an owner call.
- **(c)**: `docs/handoff/m5-03.md`; spec M5-03-T02.
- **(d)**: ADR is Accepted with the topology + transcendental + sqrt + Helium choices recorded.

### 2.2 T17 — real Cortex-M55 silicon / Arm FVP run

- **(a)**: run the no_std Silero VAD on real Cortex-M55 silicon (devboard) or an Arm FVP / Corstone-300, measure RTF + RAM. CC's host-executable differential is the reference oracle.
- **(b)**: this machine has no Cortex-M55 board and no FVP license; QEMU `mps3-an547` is not installed.
- **(c)**: `docs/handoff/m5-03.md`; spec M5-03-T17.
- **(d)**: Silero VAD demonstrably runs on M55/FVP (SRS §6 acceptance) with real RTF/RAM.
- **honest note**: the both-rate weight heap (3.15 MiB) does not fit a typical M55 on-chip SRAM as-is; single-rate is 1.29–1.86 MiB (borderline). The reduction options (drop `weight_t`, single-rate bind, XIP zero-copy borrow) are recorded in the handoff as owner follow-ups.

### 2.3 T18 — Tier-3 positioning + Helium investment sign-off

- **(a)**: sign off the "opt-in / community-maintained" Tier-3 positioning and decide whether to invest in raw-asm Helium/MVE acceleration (scalar meets the acceptance criterion; MVE intrinsics are absent on stable Rust).
- **(b)**: a market-positioning + cost/benefit call informed by the T17 real numbers.
- **(c)**: `docs/handoff/m5-03.md`; spec M5-03-T18.
- **(d)**: positioning + Helium decision recorded.

---

## 3. M5-05 — voice-clone separation + watermark-dependency resolution

CC landed the contradiction ADR (Proposed), the consent schema/validator, the flag gate, and the `vokra-voiceclone-experimental` scaffold seed. See `docs/adr/M5-05-watermark-dependency.md`.

### 3.1 T04 — resolution option + legal judgment + ADR ratification

- **(a)**: choose the resolution option ((i) un-defer watermark embedding / (ii) amend the completion criteria to what the code holds / (iii) M5-defer), judge EU AI Act Article 50 / SB 942 / ELVIS Act / NO FAKES sufficiency, decide the consent-signature trust root (whose key / distribution / revocation), and set the ADR to Accepted. CC's recommendation is "(提案) (ii)" (matches the current honest posture: core does not embed, the deployer discloses per §1.4).
- **(b)**: a legal-sufficiency + trust-root decision; not a code judgment.
- **(c)**: `docs/adr/M5-05-watermark-dependency.md` §5 (blank); spec M5-05-T04.
- **(d)**: ADR Accepted with the option, legal record, and signature-verification policy filled in.

### 3.2 T15 — publish the separate repo + f0_extract + sign-off + doc propagation

- **(a)**: create/publish `vokra-voiceclone-experimental` from the scaffold seed (`staging/vokra-voiceclone-experimental/`, gitignored); confirm the `f0_extract` (FR-OP-83) implementation site (core vs separate repo) AND its landing WP (its only assignment `milestones.md:56` M5-05 is invalidated by this defer — pick a WP number, CC will not invent one); fill the `docs/license-audit.md` §3.1 RVC v2 / GPT-SoVITS sign-off rows (blank = fail-closed); approve the CLAUDE.md `otonx-` → `vokra-` rename.
- **(b)**: repo creation/publish, legal sign-off, and the WP-number/SSOT decisions are owner-only.
- **(c)**: `docs/adr/M5-05-watermark-dependency.md`; spec M5-05-T15.
- **(d)**: repo published (flag + consent enforced; the watermark-forced leg follows T04); f0_extract site + landing WP recorded; sign-off rows filled; rename approved.

**honest note (watermark leg)**: the "watermark forced-embed" completion leg is honest-UNMET — `WatermarkConfig::backend_status()` is permanently Deferred (2026-07-04 drop, BIG-8 held). The scaffold test positively asserts this UNMET state rather than faking a pass. It becomes MET only if T04 picks option (i).

---

## 4. M5-04 — console-portability static-link base

CC landed `scripts/check-console-static.sh` (C-ABI-completeness + FFI-panic-firewall + no-dynamic-load gate, self-tested). See `docs/handoff/m5-04.md`.

### 4.1 Console NDA + SDK build + ADR ratification

- **(a)**: sign the console-platform NDA, install the SDK toolchain, run `VOKRA_STATIC_TRIPLE=<sdk-triple> scripts/check-console-static.sh` against the real target, and ratify `docs/adr/M5-04-console-portability.md` (Proposed).
- **(b)**: the static-link SDK is only obtainable under NDA; the real target triple must not be written into any tracked file.
- **(c)**: `docs/handoff/m5-04.md` §(c).
- **(d)**: the gate passes for the real console triple; ADR Accepted.

---

## 5. M5-07 — Bark / StyleTTS 2 / Matcha-TTS license sign-off

CC landed the audit material (fail-closed, docs-only) in `docs/license-audit.md` §3 / §3.1 / §CC-verified and `docs/legal-compliance.md` §9.

### 5.1 Adoption sign-off

- **(a)**: for Bark / StyleTTS 2 / Matcha-TTS, make the Commercial / Research-only / Rejected decision, pass the legal-compliance checklist, and fix the §9 ✅/⚠️ tier. Fill the §3.1 sign-off rows (blank = fail-closed = not for official distribution; CC did not pre-fill).
- **(b)**: an adoption + legal-sufficiency judgment.
- **(c)**: `docs/license-audit.md` §3.1; spec M5-07-T09/T10.
- **(d)**: each model has a recorded tier + signed-off row.
- **honest note**: Bark = current MIT (was CC-BY-NC → MIT 2023-05-01) but the HF card says "research purposes only"; StyleTTS 2 weight is a voice-consent usage agreement → registry `Unknown` (fail-closed); Matcha checkpoint has no separate license file (owner primary-source check pending). These are owner legal calls.

---

## SoTA plan Phase 1-4 + JA + BF16 fleet (2026-07-25, PR #20)

CC landed the SoTA plan Phase 1-4 + JA + BF16 fleet scaffolds on branch `feat/sota-phase1-2026-07-23`. The following are owner-only actions. All checkboxes are unchecked; fail-closed default (blank / unchecked → no publish, no promote) applies until yousan-signed with primary-source verification.

### 6.1 PR #20 review + merge

- [ ] Review PR #20 (branch `feat/sota-phase1-2026-07-23` → `main`) and merge.

### 6.2 License sign-off in `docs/license-audit.md` §3.1

Fail-closed default (blank → no publish) applies until yousan-signed with primary-source verification. CC did not pre-fill any row.

BF16 fleet families:

- [ ] `kimi_audio`
- [ ] `step_audio2_mini`
- [ ] `baichuan_audio`
- [ ] `speechtokenizer`
- [ ] `funcodec`
- [ ] `xy_tokenizer`
- [ ] `bicodec`
- [ ] `neucodec`
- [ ] `openvoice_v2`
- [ ] `knn_vc`
- [ ] `freevc`
- [ ] `meanvc`
- [ ] `ecapa_tdnn`
- [ ] `wespeaker`
- [ ] `speaker_3d`
- [ ] `emotion2vec`

Phase 1-4 + JA families:

- [ ] Dia
- [ ] Zonos
- [ ] Kyutai-STT
- [ ] Parakeet-TDT
- [ ] Parakeet-CTC
- [ ] Canary
- [ ] OmniASR-CTC
- [ ] Distil-Large
- [ ] CosyVoice3
- [ ] Chatterbox × 3 variants (sign off all three at their respective rows)
- [ ] Qwen3-TTS
- [ ] VoxCPM2
- [ ] VibeVoice
- [ ] kotoba-whisper
- [ ] Irodori
- [ ] vits-ja (audit only — weight publication is separately excluded, see §6.8)

### 6.3 Parity CI activation (9 workflows)

Full runbook: `docs/handoff/parity-ci-flip-switch.md`. Per family: read the HF card → complete §3.1 sign-off (§6.2) if publishable → set the `VOKRA_<PREFIX>_ENABLE=1` repo/environment variable → `gh workflow run parity-<family>-real.yml` → confirm the workflow reports a PASS verdict.

Original SoTA Phase 1-4 seven families:

- [ ] Family 1 (NeMo-ASR, `VOKRA_NEMO_ASR_ENABLE`): HF-card read → §6.2 row signed → `VOKRA_<PREFIX>_ENABLE=1` set → `gh workflow run parity-<family>-real.yml` → PASS verdict confirmed.
- [ ] Family 2 (whisper-extras, `VOKRA_WHISPER_EXTRAS_ENABLE`): same sequence.
- [ ] Family 3 (tts-dac, `VOKRA_TTS_DAC_ENABLE`): same sequence.
- [ ] Family 4 (tts-hiftnet, `VOKRA_TTS_HIFTNET_ENABLE`): same sequence.
- [ ] Family 5 (Qwen3-TTS, `VOKRA_QWEN3_TTS_ENABLE`): same sequence.
- [ ] Family 6 (tts-continuous-vae, `VOKRA_TTS_CONT_VAE_ENABLE`): same sequence.
- [ ] Family 7 (tts-japanese, `VOKRA_TTS_JA_ENABLE`): same sequence.

2026-07-28 follow-up additions (bringing total to 9):

- [ ] Family 8 (deepfilternet3, `VOKRA_DFN3_ENABLE`): HF-card read (Rikorose/DeepFilterNet MIT/Apache-2.0 dual, §3.1 row 258 already ☑ Commercial) → set `VOKRA_DFN3_ENABLE=1` → `gh workflow run parity-deepfilternet3-real.yml` → PASS verdict confirmed. Phase B byte-parity leg additionally needs `VOKRA_DFN3_DATA_URL` populated with a pre-baked reference bundle — see `docs/handoff/parity-deepfilternet3-real.md` §Phase B.
- [ ] Family 9 (deberta-v3-large, `VOKRA_DEBERTA_V3_ENABLE`): HF-card read (microsoft/deberta-v3-large MIT, §3.1 row 304 already ☑ Commercial 2026-07-27 yousan) → set `VOKRA_DEBERTA_V3_ENABLE=1` → `gh workflow run parity-deberta-v3-large-real.yml` → PASS verdict confirmed. Phase B (Rust numerical parity vs reference dumper) opt-in on `VOKRA_DEBERTA_V3_HARNESS_READY=1` — currently honest-skips with `::notice::` since no consumer harness exists yet. See `docs/handoff/parity-deberta-v3-large-real.md`.

### 6.4 Real-weight parity harness fire

For each landed scaffold that ships a flip-the-switch harness, point the per-family `REFERENCE_DIR` env var (e.g. `VOKRA_HIFTNET_REFERENCE_DIR`) at the real dumped reference tensors and re-run the harness. Per-family env-var names are recorded in the parity CI YAMLs (`.github/workflows/parity-*.yml`).

- [ ] Enumerate landed flip-the-switch scaffolds from the parity CI YAMLs.
- [ ] For each, dump the reference tensors from the real upstream weights.
- [ ] For each, set the `VOKRA_*_REFERENCE_DIR` env var and re-run the harness.
- [ ] Record PASS / FAIL per family.

### 6.5 misaki venv setup (Kokoro G2P)

- [ ] Create a Python venv and install `misaki[en,ja,zh,ko]`.
- [ ] Export `VOKRA_MISAKI_VENV` = venv path in the runner / dev environment.

### 6.6 Follow-up WPs (CC-tracked, not owner-blocking)

These are tracked on the CC side for future waves; listed here for owner visibility only. Not gating for GA.

- [ ] F0 op real CNN forward (replace placeholder implementation).
- [ ] `align` real Viterbi implementation.
- [ ] `vokra-kws-micro` real model (replace scaffold).
- [ ] BF16 native compute in runtime (currently upcast-to-f32 shim).
- [ ] GPU kernel land for HiFTNet.
- [ ] GPU kernel land for BigVGAN.
- [ ] GPU kernel land for SNAC.
- [ ] GPU kernel land for Qwen3-TTS-codec.

### 6.7 Publication decisions (huggingface.co/vokra)

Each of the ~30 new families requires yousan sign-off before upload. Per memory [[project-huggingface-vokra-publication]] the 5-gate posture applies: catalog-reality / redistributable / provenance / §3.1 sign-off / allow-noncommercial. Publication is default "not published — will decide at publish time" per fail-closed policy.

- [ ] For each family in §6.2 (BF16 fleet + Phase 1-4 + JA), decide upload / withhold per the 5-gate posture and record the verdict alongside the §3.1 sign-off row.
- [ ] Confirm each uploaded repo carries: LICENSE (upstream file, not just an SPDX tag), NOTICE (if attribution-required), `SOURCE.md` (upstream URL + re-convert recipe), and `vokra.schema.version` / `vokra.schema.producer` provenance in the GGUF.
- [ ] Run `publish-one.sh` (never the manual upload path) for every published family.

### 6.8 VITS-JA weight — excluded from vokra publication

VITS-JA weight is `RedistributionForbidden` (JSUT / JVS training data forbid weight redistribution). It is excluded from `huggingface.co/vokra` irrespective of §6.2 audit sign-off. §6.2 covers the audit record only; §6.8 covers the publication exclusion.

- [ ] Confirm VITS-JA weight remains excluded from `huggingface.co/vokra` regardless of the §6.2 audit outcome.
- [ ] Verify the `check-catalog-reality.sh` / `LicenseClass::redistributable()` gate rejects any accidental attempt to publish the VITS-JA weight.
