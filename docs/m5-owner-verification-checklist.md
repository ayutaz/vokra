# M5 (v1.0 GA) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — real-hardware verification, real-weight sourcing, legal sign-off, external contracts / infra provisioning, ADR ratification, and the v1.0 GA tag decision.

**CC-side status (2026-07-21)**: this checklist covers the owner tasks left by the M5 WPs whose CC-side work has landed on branch `feat/m5-plan-and-wave1`. It is the input to the **v1.0 GA** decision (commercial GA + C ABI freeze). It is NOT a GA declaration and NOT a freeze — the freeze FIRES at the owner's v1.0 GA tag (M5-13).

**2026-08-10 addendum (SBV2 v2 4-Blocker + Blocker 2c residual + ZH BERT publish + H100 FA v3 bakeoff; PR #27 merged 2026-08-11)**: the following is the pre-merge wave ledger from branch `feat/sbv2-voxtral-real-verify-2026-08-06` (then 18 commits ahead of `origin/main`, tip `8d469eb`). PR #27 merged as `0937ef874495465bdadf18d5511f14e6e2a0ab71`; the follow-up audit PR #29 merged as `8e048d8afd95d7d26bfa5121eef7533178b854d1` on 2026-08-17.
- **Wave 1** (2026-08-10, 9 commits `16a8410..9cb4d52`): SBV2 4 Blockers closed — Blocker 5 (SentencePiece proto parser + WordPiece + DeBERTa v2/v3 sibling tokenizer discovery, `cb2cd7b`/`e7dc2e4`/`7242f94`), Blocker 3 (`SbV2Model::speaker_projection()` accessor, `1a90e0d`), Blocker 2b (TDD-hardening 3 commits: flow rename table + metadata-key contract + converter spelling, `296dba1`/`672ef5b`/`922d3f5`), Blocker 2c Wave 1 (rational-quadratic spline math primitive, `f1b7815`).
- **Wave 2** (2026-08-10, 3 commits `5027b2b..c8e2777`): Blocker 2c residual — `.sqrt()` routed through `vokra_math` (`5027b2b`), from_gguf loud-fail defensive check for `sbv2.sdp.flows.<even>.*` unread tensors (`879ba8e`), `#[ignore]`d `sdp_body_matches_torch_ref` scaffold as owner-fixture-待ち gate (`c8e2777`).
- **Wave 3** (2026-08-10, 3 commits `315b8f7..3f76abf`): the license-audit.md §3.1 entry for ZH BERT `hfl/chinese-roberta-wwm-ext-large` changed blank → ☑ Commercial by owner delegation (`315b8f7`), CLI `bert-base` arm + `nemo_pt_to_safetensors.py` shared-tensor dedup (`1ea38bd`), fixture sidecar populate for WP-19 4-file loader (`3f76abf`).
- **Wave 4** (2026-08-10, 1 commit `8d469eb`): M4-07 T17/T18 H100 FA v3 bakeoff on vast.ai H100 PCIe (60 min, $1.73, offer #31427212). See §7 (SoTA cross-cutting) for owner ripples — the M4-07 owner ripple is tracked in `docs/m4-owner-verification-checklist.md` §2.1 (dashboard registration only remains).

**Verify snapshot at pre-merge branch tip `8d469eb`**: `cargo test --workspace` = 5447 passed / 0 failed / 22 ignored / 199 suites (baseline 5446/21, +1 test +1 scaffold). All gates green: `cargo fmt` / `cargo clippy -D warnings` / `scripts/check-zero-deps.sh` (root Cargo.lock = `vokra-*` only, NFR-DS-02 preserved) / `scripts/check-abi-changelog.sh` / `scripts/gen-c-abi.sh --check` (no drift, v1.0-rc baseline 33 fn + 11 typedef unchanged, no new C ABI). This is historical evidence; PR #27 is merged.

**2026-08-17 current-state rule**: the earlier **94 unchecked boxes** were a historical owner ledger, **not** a count of 94 missing implementations. This reconciliation leaves **41 unchecked boxes**, which remain an action ledger rather than an implementation metric: a box can mean an external legal/infra decision, real-weight access, a deliberately fail-closed policy, a future backend, or a partially landed implementation that still lacks its real-checkpoint proof. Only mark a box complete when its literal done-condition is evidenced; do not infer implementation status from the unchecked total. The reconciliation below removes stale merge/sign-off/implementation claims while retaining genuine follow-up work.

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

- **(a)**: run the CoreML (Apple ANE) and QNN (Qualcomm Hexagon) delegates on real hardware and measure the NFR-PF-12 acceptance criterion (≥ 2× over the CPU baseline). Feeds T19.
- **(b)**: needs real ANE / Hexagon silicon; this machine has neither an NPU bakeoff rig nor the delegate runtimes.
- **(c)**: spec M5-01-T24 / M5-02-T12 (gitignore-local); runbook is the sub-sections below.
- **(d)**: a `PASS` / `FAIL` / `INSUFFICIENT DATA` verdict vs the 2× bar is recorded for each delegate in the sibling template files.

### 1.5.1 Baseline discipline (NFR-PF-12 protocol)

The 2× ratio compares an NPU RTF to a **CPU baseline captured on the same host in the same session**. The CPU baseline is **M5-14-post CPU** (SIMD hot-path optimised, libm-route — the leg landed by M5-14). An NPU RTF captured without a matched CPU baseline **cannot** feed the 2× verdict; a matched pair collected minutes apart on the same box is what NFR-PF-12 acceptance requires. This is codified in `docs/system-requirements.md` NFR-PF-12 (footnote + hazard clause), mirrored in the tracked glossary at `docs/requirement-ids.md` NFR-PF-12, and cross-referenced from `docs/handoff/m5-02.md` §"NFR-PF-12 baseline".

Silent-CPU-fallback (< 90 % placement on the target NPU) **disqualifies** the run — the analyzer surfaces this as a `WARN`, and the template forces the owner to record `INSUFFICIENT DATA` rather than a numeric 2× ratio. This is the FR-EX-08 hazard clause, not a soft warning.

### 1.5.2 CC-side machinery landed

CC has landed the CC-actionable prep (WP-15) for owner NPU bakeoff. No hardware access was needed for the prep itself; all artifacts are docs + shell/python tooling. The owner-visible pieces are:

- `tools/parity/npu_rtf_variance.sh` — generic NPU RTF variance harness (mirrors `cuda_rtf_variance.sh`, `--backend {coreml|qnn|cuda|cpu}`, folds an optional placement probe's JSON into each iteration line).
- `tools/parity/npu_rtf_analyze.py` — analyzer (stdlib only, matches the `cuda_rtf_analyze.py` red-line — never asserts an RTF ceiling — but adds a `WARN` on CV > 0.20 or placement < 90 %).
- `tools/parity/test_npu_rtf_analyze.py` — Python unit tests covering clean / flaky-fallback / noisy runs plus the QNN `htp_frac` vs legacy `dsp_frac` alias.
- `tools/parity/provision-h100.sh` — H100 provisioning script for the M4-07 FA v3 bench, sibling of `scripts/publish/vast-ai/provision.sh`; includes a Hopper compute-cap gate (exit 1 if < 9.0).
- `docs/handoff/m5-01-coreml-bakeoff-template.md` — CoreML/ANE bakeoff report template.
- `docs/handoff/m5-02-qnn-bakeoff-template.md` — QNN/Hexagon bakeoff report template.

### 1.5.3 Owner runbook (per delegate)

Run this loop once per delegate (CoreML then QNN). Both loops end with a
recorded verdict feeding **§1.3 T19 GO/NO-GO** on the C-ABI symbol call.

**Prep**
1. Wire up a **placement probe** for the delegate — a shell wrapper that
   emits `{"ane_frac": …, "gpu_frac": …, "cpu_frac": …}` (CoreML — from
   Xcode Instruments MLModel trace) or `{"htp_frac": …, "cpu_frac": …}`
   (QNN — from `qnn-net-run --profiling_option=op`). The analyzer refuses
   to declare a 2× verdict without one — a missing probe is
   `INSUFFICIENT DATA` per FR-EX-08.
2. Copy the template file next to the runbook artifact:
   `cp docs/handoff/m5-0{1,2}-*-bakeoff-template.md docs/handoff/m5-0{1,2}-*-bakeoff-YYYY-MM-DD.md`.
3. Fill in §1 (hardware fingerprint) before running the harness — makes
   the run reproducible even if the box gets destroyed later.

**Baseline capture** (§2 of the template)
4. Run `tools/parity/npu_rtf_variance.sh --backend cpu --iters 10` on the
   target hardware. The `cpu` arm invokes vokra-cli's M5-14-post CPU
   path (SIMD hot-path optimised, libm-route). Save the JSONL.
5. Run `tools/parity/npu_rtf_analyze.py` on the JSONL. Confirm
   `Analyzer CV verdict = OK` before recording the mean.

**Delegate capture** (§3 of the template)
6. Run `tools/parity/npu_rtf_variance.sh --backend {coreml|qnn} --iters 10
   --placement-probe /path/to/probe.sh` on the target hardware. Save the
   JSONL.
7. Run `tools/parity/npu_rtf_analyze.py` on the JSONL. Confirm both
   `Analyzer CV verdict = OK` and `Analyzer placement verdict = OK`
   before recording the mean. If placement < 90 %, treat as
   `INSUFFICIENT DATA` and hand the failing-op inventory back to CC as
   an M5-01 / M5-02 follow-up ticket.

**Verdict** (§4 of the template)
8. Compute `speedup = CPU_median / NPU_median` and compare to 2.0.
9. Record `PASS` / `FAIL` / `INSUFFICIENT DATA` and the reason.

**Commit** (§6 of the template)
10. Commit the JSONL + report + filled template under
    `docs/bench-baselines/m5-0{1,2}-*-bakeoff-YYYY-MM-DD/` and
    `docs/handoff/m5-0{1,2}-*-bakeoff-YYYY-MM-DD.md`.
11. Tick the §1.5 checkbox below and feed the verdict into §1.3 T19.

### 1.5.4 Bakeoff checklist

- [ ] CoreML placement probe (Xcode Instruments MLModel trace wrapper) is wired up + emits the expected JSON.
- [ ] CoreML baseline captured (`cpu`, N=10, CV ≤ 0.20).
- [ ] CoreML NPU captured (`coreml`, N=10, CV ≤ 0.20, mean placement ≥ 0.90).
- [ ] CoreML verdict recorded in `docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md`.
- [ ] QNN placement probe (`qnn-net-run --profiling_option=op` wrapper) is wired up + emits the expected JSON.
- [ ] QNN baseline captured (`cpu`, N=10, CV ≤ 0.20).
- [ ] QNN NPU captured (`qnn`, N=10, CV ≤ 0.20, mean placement ≥ 0.90).
- [ ] QNN verdict recorded in `docs/handoff/m5-02-qnn-bakeoff-YYYY-MM-DD.md`.
- [ ] Both verdicts fed into §1.3 T19 (GO/NO-GO on the delegate selector C-ABI symbol).

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

## 5. M5-07 — Bark / StyleTTS 2 / Matcha-TTS license decision (reconciled)

The current `docs/license-audit.md` §3.1 rows record Bark and Matcha-TTS as Commercial (2026-07-23) and StyleTTS 2 as Rejected. The earlier owner-decision wording in this checklist was a historical audit snapshot, not an active blank-sign-off gate.

### 5.1 Recorded decision

- **(a)**: retain the current decisions: Bark = Commercial, Matcha-TTS = Commercial, StyleTTS 2 = Rejected. Reopen a row only if a primary source changes.
- **(b)**: these are already-recorded adoption + legal-sufficiency judgments; real-weight verification and implementation remain separate gates.
- **(c)**: `docs/license-audit.md` §3.1; spec M5-07-T09/T10.
- **(d)**: satisfied — all three rows have a recorded tier and decision.
- **honest note**: Bark is current MIT (was CC-BY-NC → MIT 2023-05-01) while the HF card says "research purposes only"; the recorded §3.1 decision treats the MIT license as governing. StyleTTS 2 remains excluded because its weight carries a voice-consent usage agreement. Matcha's checkpoint provenance remains a real-weight verification concern even though its §3.1 commercial row is recorded.

---

## SoTA plan Phase 1-4 + JA + BF16 fleet (reconciled 2026-08-17)

PR #20 has merged. This section is now a mixed owner/action ledger: its remaining unchecked boxes are not a claim that the corresponding model or converter is absent. Fail-closed still applies where a row has no decision or where a publish gate has not been completed.

### 6.1 PR #20 review + merge

- [x] Review and merge PR #20 (`feat/sota-phase1-2026-07-23` → `main`), merged as `7ed054825bbd51d8c0b7556657db5000059de922` on 2026-07-25.

### 6.2 License sign-off in `docs/license-audit.md` §3.1

`docs/license-audit.md` §3.1 is the source of truth. The old per-family unchecked list was stale: it treated every row as blank even after decisions were recorded. These are **license decisions only**, not claims that a converter is publish-ready or that a real-weight parity run has completed.

- [x] **Commercial decision recorded** for `kimi_audio`, `step_audio2_mini`, `baichuan_audio`, `speechtokenizer`, `funcodec`, `xy_tokenizer`, `neucodec`, `ecapa_tdnn`, `wespeaker`, `speaker_3d`, `emotion2vec`; and for Dia, Zonos, Kyutai-STT, Parakeet-TDT/CTC, Canary, OmniASR-CTC, Distil-Whisper Large, CosyVoice3, all three Chatterbox variants, Qwen3-TTS, VoxCPM2, kotoba-whisper, and Irodori.
- [x] **Research-only decision recorded** for `bicodec` (CC-BY-NC-SA-4.0; its T4/T3 publication record is separate from normal commercial publishing).
- [x] **Rejected/withheld decision recorded** for VITS-JA (explicit corpus redistribution prohibition; see §6.8) and for the withdrawn VibeVoice-Large upstream. The available VibeVoice-1.5B and Realtime-0.5B variants have Commercial decisions.
- [ ] **Voice-conversion scope decision remains** for `openvoice_v2`, `knn_vc`, `freevc`, and `meanvc`: choose an experimental-repository destination or explicitly reject them for the public main repository. Their blank §3.1 decisions are intentional pending that policy decision, not missing converter implementations.

### 6.3 Parity CI activation (10 workflows: 9 variable gates + SBV2 sidecar gate)

Full runbook: `docs/handoff/parity-ci-flip-switch.md`. For the nine variable-gated families: read the HF card → complete §3.1 sign-off (§6.2) if publishable → set the `VOKRA_<PREFIX>_ENABLE=1` repo/environment variable → `gh workflow run parity-<family>-real.yml` → confirm the workflow reports a PASS verdict. SBV2 is the deliberate exception: its three required sidecars control the current JA/EN numerical-parity leg.

**2026-08-17 status**: scheduled workflow successes exist for the listed families, but a green scheduled run alone does not prove that the required real-weight leg was enabled, downloaded, and produced the required reference artifact. Keep these boxes open until the per-family run output demonstrates the full PASS verdict without an honest skip.

Original SoTA Phase 1-4 seven families:

- [ ] Family 1 (NeMo-ASR, `VOKRA_NEMO_ASR_ENABLE`): HF-card read → applicable §6.2 decision recorded → `VOKRA_<PREFIX>_ENABLE=1` set → `gh workflow run parity-<family>-real.yml` → PASS verdict confirmed.
- [ ] Family 2 (whisper-extras, `VOKRA_WHISPER_EXTRAS_ENABLE`): same sequence.
- [ ] Family 3 (tts-dac, `VOKRA_TTS_DAC_ENABLE`): same sequence.
- [ ] Family 4 (tts-hiftnet, `VOKRA_TTS_HIFTNET_ENABLE`): same sequence.
- [ ] Family 5 (Qwen3-TTS, `VOKRA_QWEN3_TTS_ENABLE`): same sequence.
- [ ] Family 6 (tts-continuous-vae, `VOKRA_TTS_CONT_VAE_ENABLE`): same sequence.
- [ ] Family 7 (tts-japanese, `VOKRA_TTS_JA_ENABLE`): same sequence.

2026-07-28 follow-up additions (bringing the variable-gated total to 9):

- [ ] Family 8 (deepfilternet3, `VOKRA_DFN3_ENABLE`): HF-card read (Rikorose/DeepFilterNet MIT/Apache-2.0 dual; §3.1 Commercial decision recorded) → set `VOKRA_DFN3_ENABLE=1` → `gh workflow run parity-deepfilternet3-real.yml` → PASS verdict confirmed. Phase B byte-parity leg additionally needs `VOKRA_DFN3_DATA_URL` populated with a pre-baked reference bundle — see `docs/handoff/parity-deepfilternet3-real.md` §Phase B.
- [ ] Family 9 (deberta-v3-large, `VOKRA_DEBERTA_V3_ENABLE`): HF-card read (microsoft/deberta-v3-large MIT; §3.1 Commercial decision recorded 2026-07-27) → set `VOKRA_DEBERTA_V3_ENABLE=1` → `gh workflow run parity-deberta-v3-large-real.yml` → PASS verdict confirmed. Phase B (Rust numerical parity vs reference dumper) opt-in on `VOKRA_DEBERTA_V3_HARNESS_READY=1` — currently honest-skips with `::notice::` since no consumer harness exists yet. See `docs/handoff/parity-deberta-v3-large-real.md`.

- [ ] Family 10 (SBV2, sidecar-hash gate): `parity-sbv2-real.yml` first validates the three real sidecars for the current main + JA BERT + EN BERT numerical path, then runs the real dump and `parity_sbv2_real`. The published ZH BERT sidecar proves the separate WP-19 four-file loader input is available; it is **not** a ZH numerical-parity PASS because the ZH reference dumper/harness does not exist yet. See `docs/handoff/parity-ci-flip-switch.md`.

### 6.4 Real-weight parity harness fire

For each landed scaffold that ships a flip-the-switch harness, point the per-family `REFERENCE_DIR` env var (e.g. `VOKRA_HIFTNET_REFERENCE_DIR`) at the real dumped reference tensors and re-run the harness. Per-family env-var names are recorded in the parity CI YAMLs (`.github/workflows/parity-*.yml`).

- [x] Enumerate the current inventory: nine `VOKRA_*_ENABLE` variable-gated workflows (Families 1–9 in §6.3) plus the SBV2 sidecar-hash-gated workflow (Family 10). A scheduled `success` is not evidence of a real run while its gate is closed.
- [ ] For each of the nine variable-gated families, dump reference tensors from real upstream weights, set its `VOKRA_*_REFERENCE_DIR` where applicable, and record the numerical harness result.
- [ ] For SBV2, populate/verify the three current numerical-path inputs, run the JA/EN dump and `parity_sbv2_real`, and separately add the missing ZH reference-dump/harness before claiming four-file numerical parity.
- [ ] Record PASS / FAIL per family.

### 6.5 misaki venv setup (Kokoro G2P)

- [ ] Create the Python venv with `uv venv` and install with `uv pip install 'misaki[en,ja,zh,ko]'`.
- [ ] Export `VOKRA_MISAKI_VENV` = venv path in the runner / dev environment.

### 6.6 Follow-up WPs (CC-tracked, not owner-blocking)

These are tracked on the CC side for future waves; listed here for owner visibility only. Not gating for GA.

- [x] F0 / CREPE real 6-block CNN forward landed (`crates/vokra-models/src/f0/crepe.rs`); targeted F0 tests pass. Real external-checkpoint parity remains a separate §6.4 task.
- [ ] Charsiu `align` real-checkpoint binding and reference parity. CTC segmentation/Viterbi and synthesized-weight forward are implemented; the remaining work is the upstream tensor manifest/GGUF bind, not a replacement of a placeholder Viterbi algorithm.
- [ ] `vokra-kws-micro` upstream-model binding and real `hey_jarvis` fixture. The INT8 pipeline and synthetic/parity tests are landed; emitted quantization metadata plus a real checkpoint remain.
- [ ] BF16 native compute in runtime (currently upcast-to-f32 shim).
- [ ] Full HiFTNet GPU generator path. Metal primitives are landed, but the complete generator and non-Metal backends remain.
- [ ] Full BigVGAN GPU path. Metal activation/upsampling primitives are landed, but the complete generator and non-Metal backends remain.
- [x] SNAC Metal MSL kernel and CPU-parity coverage landed. CUDA/Vulkan/WebGPU equivalents remain future backend work.
- [x] Qwen3-TTS-codec Metal MSL kernel and CPU-parity coverage landed. CUDA/Vulkan/WebGPU equivalents remain future backend work.

### 6.7 Publication decisions (huggingface.co/vokra)

Each unpublished family still requires the 5-gate posture: catalog-reality / redistributable / provenance / §3.1 sign-off / allow-noncommercial. Completed model-level decisions and uploads are recorded in §6.9 and §7; they are not evidence that every family is published.

- [ ] Complete the remaining publication/destination decisions, beginning with the four voice-conversion families in §6.2; record every future upload or withholding verdict beside the §3.1 row.
- [ ] Confirm each uploaded repo carries: LICENSE (upstream file, not just an SPDX tag), NOTICE (if attribution-required), `SOURCE.md` (upstream URL + re-convert recipe), and `vokra.schema.version` / `vokra.schema.producer` provenance in the GGUF.
- [ ] Run `publish-one.sh` (never the manual upload path) for every published family.

### 6.8 VITS-JA weight — excluded from vokra publication

VITS-JA weight is `RedistributionForbidden` (JSUT / JVS training data forbid weight redistribution). It is excluded from `huggingface.co/vokra` irrespective of §6.2 audit sign-off. §6.2 covers the audit record only; §6.8 covers the publication exclusion.

- [x] VITS-JA remains excluded from `huggingface.co/vokra`: its §3.1 row is explicitly Rejected for the JSUT redistribution terms, and its parity workflow disables HF auto-fetch by design.
- [x] The default gate resolves `vits-ja` aliases to `LicenseClass::RedistributionForbidden`, whose `redistributable()` result is false; `scripts/publish/check-catalog-reality.sh` supplies the catalog-side drift gate. Any exceptional, separately licensed retraining requires a new model row and review.

### 6.9 Publish sign-off queue (2026-07-28)

This queue is historical plus active backlog. It must not be read as a blanket "all sign-offs blank" list: the five ASR entries below are published, the BF16 fleet has current §3.1 decisions except the four intentionally deferred voice-conversion rows, and several remaining entries are blocked by converter/configuration/compute work rather than legal review.

Each unchecked entry below therefore names its actual remaining condition (real-checkpoint validation, converter/configuration work, infrastructure, or policy), not a generic missing sign-off.

**Row-number note**: the audit table grows over time. Any historical `row N` reference retained below is a dated snapshot; the model identifier and its current §3.1 decision are authoritative.

**Phase 2 ASR family (5 entries) — 5 published 2026-07-28**:

Per 2026-07-28 owner explicit go-signal ("Wave 3 の 22 owner-signoff モデル + Voxtral-Small-24B publish を進めてください"), CC has signed all 5 rows and pushed to huggingface.co/vokra. NVIDIA-EULA overlay decision resolved as: NVIDIA-EULA governs runtime binaries (cuDNN/cuBLAS bundles), the CC-BY-4.0 weight redistribution is governed by the model card's license tag.

- [x] **kyutai/stt-2.6b-en** (row 266) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED**: `huggingface.co/vokra/kyutai-stt-2.6b-en` = live, ~5.23 GB / 323 tensors, BF16 direct (no strip). Mimi sibling already at `vokra/mimi`.
- [x] **nvidia/parakeet-tdt-0.6b-v3** (row 267) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED**: `huggingface.co/vokra/parakeet-tdt-0.6b-v3` = live, ~2.51 GB / 699 tensors, `num_batches_tracked` 24 stripped via `tools/parity/strip_int_tensors.py` (inference-inert BatchNorm counter). NVIDIA-EULA overlay decision: weight redistribution governed by CC-BY-4.0 card.
- [x] **nvidia/parakeet-ctc-1.1b** (row 268) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED**: `huggingface.co/vokra/parakeet-ctc-1.1b` = live, ~4.25 GB / 1652 tensors, `num_batches_tracked` 42 stripped.
- [x] **nvidia/canary-1b-v2** (row 269) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED (vast.ai)**: `huggingface.co/vokra/canary-1b-v2` = live. Upstream distributes `.nemo` only (2.5 GB tar); `tools/parity/nemo_pt_to_safetensors.py` extracts the inner `timestamps_asr_model_weights.ckpt` (688 float tensors kept, 24 int tensors stripped as inference-inert) into safetensors, then `vokra-cli convert --model canary` produces the GGUF. NVIDIA-EULA overlay decision: weight redistribution governed by the CC-BY-4.0 card (NOTICE §7 carries NVIDIA credit).
- [x] **facebook/omniASR-CTC-1B** (row 270) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED (vast.ai)**: `huggingface.co/vokra/omniasr-ctc-1b` = live. Upstream distributes `omniASR-CTC-1B.pt` (3.9 GB — a regular pickled `state_dict`, not TorchScript); `tools/parity/nemo_pt_to_safetensors.py` handles it via `torch.load` + `model` wrapper unwrap (807 float tensors kept, 0 int tensors stripped) into safetensors, then `vokra-cli convert --model omniasr-ctc` produces the GGUF. HONEST DISCREPANCY pin ratified in same wave: `facebook/omniASR-CTC-1B` is canonical, the SoTA-plan-listed `suno/omniASR-CTC-1B-v1` is a 401 dead reference.

**BF16 fleet skeletons (16 entries, PR #20 Wave E landing)**:

*The twelve signed, non-voice-conversion converters are wired into `ModelKind`, licensed `convert_file` dispatch, and `vokra-cli`; package and CLI parsing tests cover the paths. Their remaining work is real-weight preparation/binding/parity and, where applicable, the five-gate publish run. The four voice-conversion rows remain a policy-destination decision instead.*

- [x] **moonshotai/Kimi-Audio-7B-Instruct** — Commercial signed; CLI dispatch is complete. category=s2s. Remaining: real-weight preparation/binding/parity and publish gate. Candidate: `vokra/kimi-audio-7b-instruct`. ~14 GB BF16.
- [x] **stepfun-ai/Step-Audio-2-mini** — Commercial signed; CLI dispatch is complete. category=s2s. Remaining: real-weight preparation/binding/parity and publish gate. Candidate: `vokra/step-audio-2-mini`.
- [x] **baichuan-inc/Baichuan-Audio-Instruct** — Commercial signed; CLI dispatch is complete. category=s2s. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **fnlp/SpeechTokenizer** — Commercial signed; CLI dispatch is complete. category=codec. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **alibaba-damo/audio_codec-encodec-zh_en-…** (FunCodec) — Commercial signed; CLI dispatch is complete. category=codec. FunCodec is not Meta EnCodec; the existing allowlist is intentional. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **fnlp/XY_Tokenizer_TTSD_V0** — Commercial signed; CLI dispatch is complete. category=codec. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **SparkAudio/Spark-TTS-0.5B** (BiCodec) — Research-only signed (CC-BY-NC-SA-4.0); CLI dispatch is complete and its provenance now retains the NC + share-alike class. Remaining: real-weight preparation/binding/parity and the T4/T3 publish path. category=codec.
- [x] **neuphonic/neucodec** — Commercial signed; CLI dispatch is complete. category=codec. Remaining: real-weight preparation/binding/parity and publish gate.
- [ ] **myshell-ai/OpenVoiceV2** — public-main voice-conversion policy/destination decision remains; do not wire or publish here until it is resolved.
- [ ] **bshall/knn-vc** — public-main voice-conversion policy/destination decision remains; its upstream license also needs primary-source resolution.
- [ ] **OlaWod/FreeVC** — public-main voice-conversion policy/destination decision remains; its upstream license also needs primary-source resolution.
- [ ] **ASLP-lab/MeanVC** — public-main voice-conversion policy/destination decision remains; do not wire or publish here until it is resolved.
- [x] **speechbrain/spkrec-ecapa-voxceleb** (ECAPA-TDNN candidate) — Commercial signed; CLI dispatch is complete. The exact upstream candidate remains recorded in §3.1. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **Wespeaker/wespeaker-voxceleb-resnet34-LM** — Commercial signed; CLI dispatch is complete. category=speaker. Its current CC-BY-4.0 provenance includes FR-MD-09 attribution. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **iic/speech_eres2net_sv_zh-cn_16k-common** (3D-Speaker) — Commercial signed; CLI dispatch is complete. category=speaker. Remaining: real-weight preparation/binding/parity and publish gate.
- [x] **emotion2vec/emotion2vec_plus_large** — Commercial signed; CLI dispatch is complete. category=emotion. Remaining: real-weight preparation/binding/parity and publish gate.

**Copyleft (1 entry) — SKU rename + PUBLISHED 2026-07-28**:

- [x] **~~litagin02/style_bert_vits2~~ → `litagin/Style-Bert-VITS2-2.0-base-JP-Extra`** (SBV2 v2 JP-Extra 2.0 base, license-audit.md §3.1 **row 315**, replaces the deprecated row 302 reference above) — AGPL-3.0 ☑ Commercial 2026-07-28 yousan (依頼者許可 = CC 判断). **PUBLISHED**: `huggingface.co/vokra/sbv2-v2-jp-extra-base` = live. SKU rename rationale (per row 315 audit note): original `litagin02/style_bert_vits2` = typo (correct author = `litagin`, upstream returns 404) + actual distribution is JP-Extra 2.0 base (the current SBV2 v2 mainline), not the 1.0 multilingual base (which has no HF `cardData.license` and is defer-blocked fail-closed). Publish path used = T3 Copyleft gate (`publish-one.sh --license-spdx agpl-3.0 --acknowledge-copyleft --push` = LICENSE full text bundled + NOTICE + SOURCE.md + `--acknowledge-copyleft` opt-in flag). Fixture-status prerequisite (Blocker 2b/2c per `tests/fixtures/sbv2/README.md`) is fully resolved by 2026-08-10 Waves 1-2 (see §7 below); the residual `#[ignore]`d `sdp_body_matches_torch_ref` scaffold (commit `c8e2777`) remains an owner-fixture-待ち gate for the SDP parity flip, but is not a publish blocker for the AGPL-3.0 weight itself.

**Remaining converter / real-weight gates (signed)**:

- [x] **Suno Bark** — MIT signed 2026-07-23 yousan. `models::bark`, `ModelKind::{Bark,BarkSmall}`, and CLI dispatch support full and small variants. The upstream torch-pickle checkpoint must be flattened to safetensors by a UV-managed sidecar before the real-weight round-trip; the EnCodec companion remains research-only. Any conversion/publish run with material memory use follows the vast.ai policy.
- [ ] **Matcha-TTS** — MIT signed 2026-07-23 yousan, but the maintained design remains a conditional Draft and explicitly forbids landing a converter while the defer decision holds. Re-open only with (1) an owner-recorded trigger, (2) a ≥95% piper-plus phoneme-set coverage report, and (3) primary-source confirmation for the paired LJ Speech HiFi-GAN; then follow W0–W9 in `docs/superpowers/specs/2026-07-28-matcha-tts-design.md`. Until then `matcha.rs`, registry registration, parity workflow, and publish stay absent by design.
- [x] **WavTokenizer** — the `ModelKind::WavTokenizer` converter and CLI dispatch landed in the 2026-08-01 codec wave. The remaining future work is Lightning `.ckpt` → safetensors preparation and real-checkpoint parity, not implementation of the converter.

**VoxCPM2-2B — converter extension complete; real-weight/publish gate remains**:

- [x] **openbmb/VoxCPM2-2B** (row 280) — Apache-2.0 signed 2026-07-28 yousan. `VoxCpm2Variant::TwoB` detects the 2048-wide LM embedding, emits 2B-specific hparams/provenance, and accepts `voxcpm2-2b` CLI aliases. The remaining task is the 4.96-GB real-weight conversion/parity and five-gate publish run on vast.ai; design/runbook: `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md` and `docs/handoff/vast-ai-publish-voxcpm2-2b.md`.

**Deferred by RAM constraint (implemented + signed, host infrastructure blocked)**:

- [ ] **Voxtral-Small-24B-2507** (row 251) — Apache-2.0 signed 2026-07-23 yousan. The CLI routes unquantized config-backed conversion to `convert_voxtral_file_streaming`, but the 48-GB/11-shard real conversion, verification, and upload must run on a vast.ai instance with at least 64 GB RAM; never run it on the M1 iMac. Remaining after conversion: real ASR/runtime parity and the five-gate publish run. HF slug: `vokra/voxtral-small-24b-2507`.

**BF16 fleet — dispatch complete; real-weight and policy work remain**:

The twelve signed non-voice-conversion converters (`crates/vokra-convert/src/models/kimi_audio.rs` etc.) have `ModelKind` entries, licensed `convert_file` dispatch arms, and `vokra-cli` model parsing/help coverage (PR #27). The four voice-conversion families remain deliberately excluded from the main public distribution path pending their destination decision. Dispatch completion does not claim a runnable native forward: the remaining follow-up is real checkpoint preparation, tensor binding/parity, and the applicable five-gate publish run.

**Voice-clone territory (4 rows: openvoice_v2 / knn_vc / freevc / meanvc) — ELVIS Act policy defer**:

Per CLAUDE.md 設計判断 8, voice-cloning is intentionally excluded from the `ayutaz/vokra` public repo to avoid tool-distributor liability under ELVIS Act §3 (Tennessee, 2024-07-01) + NO FAKES Act (federal). These 4 converters should either be moved to `vokra-voiceclone-experimental` (M5-05 T15 owner-only) or explicitly Rejected in §3.1. Owner action: choose destination.

---

## 7. SBV2 v2 3-language full publish (2026-08-10; reconciled 2026-08-17)

**Status**: PR #27 merged as `0937ef874495465bdadf18d5511f14e6e2a0ab71` on 2026-08-11. All 4 SBV2 Blockers (2b / 2c / 3 / 5) closed on CC side + ZH BERT license sign-off delivered via owner delegation. SBV2 v2 3-language full publish achieved (JA / EN / ZH BERT + base = 4 models on `huggingface.co/vokra`). See the header addendum for the pre-merge wave ledger.

### 7.1 Published models (4 SKUs, all live on huggingface.co/vokra)

- [x] **`huggingface.co/vokra/sbv2-v2-jp-extra-base`** (SBV2 v2 base, AGPL-3.0, license-audit.md §3.1 row 315) — signed 2026-07-28 yousan, published via T3 Copyleft gate (`publish-one.sh --acknowledge-copyleft --license-spdx agpl-3.0 --push`). Owner ripple: none (publish complete, SA cascade obligation documented in NOTICE + README front-matter).
- [x] **`huggingface.co/vokra/deberta-v2-large-japanese-char-wwm`** (SBV2 v2 JA BERT, CC-BY-SA-4.0, license-audit.md §3.1 row 316) — signed 2026-08-06 yousan (owner delegation, T3 Copyleft path with SA cascade disclosure in NOTICE + README). Owner ripple: none (publish complete).
- [x] **`huggingface.co/vokra/deberta-v3-large`** (SBV2 v2 EN BERT, MIT, license-audit.md §3.1 row 317) — signed 2026-07-27 yousan, published via standard permissive path. Owner ripple: none (publish complete).
- [x] **`huggingface.co/vokra/chinese-roberta-wwm-ext-large`** (SBV2 v2 ZH BERT, apache-2.0, license-audit.md §3.1 row 318) — signed 2026-08-10 yousan (owner delegation "モデルは公開してください（code license に影響がない限り）"), published via standard permissive path. Runbook = `docs/handoff/zh-bert-publish-2026-08-10.md` (gitignore-local). Fixture sidecar `tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf.sha256` populated for WP-19 4-file loader (commit `3f76abf`). Owner ripple: none (publish complete).

### 7.2 Blocker 2c residual — owner-fixture-待ち gate

- [ ] **`sdp_body_matches_torch_ref` VAST real-parity gate** (Blocker 2c residual, commit `c8e2777` = `#[ignore]`d scaffold). VAST execution on 2026-08-18 against the public JP-Extra v2 checkpoint generated the independent MIT-reference body fixture and passed the Rust comparison in 85.36 s: `max |Δ| = 9.536743164e-6` at channel 96 / time 31, below the strict `1e-5` candidate bound (commit `ea3aef6` prints this measurement). The real GGUFs and derived raw fixture bytes remain gitignored under the SBV2 artifact policy, so flipping this to an unconditional local `#[test]` would manufacture failures on a clean checkout. Remaining owner action: provision a repeatable VAST-only gate that stages the three GGUFs, regenerates the fixture, runs this explicit `--ignored` test, and collects repeated measurements before any bound change. This does not gate the AGPL-3.0 weight publish (already done, §7.1) or the 4-Blocker close-out itself.

### 7.3 PR #27 merge

- [x] **Review and merge PR #27** (`feat/sbv2-voxtral-real-verify-2026-08-06` → `main`), merged as `0937ef874495465bdadf18d5511f14e6e2a0ab71` on 2026-08-11. The historical branch-tip verification snapshot above remains the evidence for the merge.

### 7.4 M4-07 X-06 nightly dashboard registration (cross-cutting)

- [x] **Register the FA v3 vs FA v2 dashboard row**: `tools/bench/build_dashboard.py` now renders `e2e_speedup_summary.fa_v3_vs_fa_v2_e2e_median` from `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` as `1.0573x` in the GPU table; its test pins the value. This satisfies the code/artifact part of M4-07 T18 without inventing a benchmark result.
- [ ] **Owner deployment gate**: enable GitHub Pages and set `VOKRA_PAGES_ENABLED=true` if the dashboard must be publicly deployed. Until then `dashboard.yml` still produces the downloadable dashboard artifact, and no public-deployment claim is made.
