# M5 (v1.0 GA) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — real-hardware verification, real-weight sourcing, legal sign-off, external contracts / infra provisioning, ADR ratification, and the v1.0 GA tag decision.

**Pre-documentation implementation/code baseline (reconciled 2026-08-31)**:
the branch observed before the documentation commits was
`feat/mac-cpu-metal-full-coverage-2026-08-28` at
`9f69277d8a0d5df574c1ee95563bd1f005de91d0`; the pre-refresh
evidence/package checkpoint was
`5cd97d124bc9eb9d2bb7b0367541dcd1492e4d1e`. The workspace is version `0.2.0`, with
57 C ABI functions / 15 typedefs and 49 checked / 33 unchecked literal boxes.
The GitHub `main` reference remains `41ce9ffdd4b0959497f55afa5016822f77a8a7b6`.
This checklist is the remaining action ledger feeding the **v1.0 GA** decision
(commercial GA + C ABI freeze). It is NOT a GA declaration and NOT a freeze —
the freeze FIRES at the owner's v1.0 GA tag (M5-13). The 2026-08-18
branch/operation history remains in
`docs/handoff/codex-operations-2026-08-18.md`; the later runtime evidence is in
`docs/handoff/runtime-gap-execution-plan-2026-08-21.md`.

**2026-08-10 addendum (SBV2 v2 4-Blocker + Blocker 2c residual + ZH BERT publish + H100 FA v3 bakeoff; PR #27 merged 2026-08-11)**: the following is the pre-merge wave ledger from branch `feat/sbv2-voxtral-real-verify-2026-08-06` (then 18 commits ahead of `origin/main`, tip `8d469eb`). PR #27 merged as `0937ef874495465bdadf18d5511f14e6e2a0ab71`; the follow-up audit PR #29 merged as `8e048d8afd95d7d26bfa5121eef7533178b854d1` on 2026-08-17.
- **Wave 1** (2026-08-10, 9 commits `16a8410..9cb4d52`): SBV2 4 Blockers closed — Blocker 5 (SentencePiece proto parser + WordPiece + DeBERTa v2/v3 sibling tokenizer discovery, `cb2cd7b`/`e7dc2e4`/`7242f94`), Blocker 3 (`SbV2Model::speaker_projection()` accessor, `1a90e0d`), Blocker 2b (TDD-hardening 3 commits: flow rename table + metadata-key contract + converter spelling, `296dba1`/`672ef5b`/`922d3f5`), Blocker 2c Wave 1 (rational-quadratic spline math primitive, `f1b7815`).
- **Wave 2** (2026-08-10, 3 commits `5027b2b..c8e2777`): Blocker 2c residual — `.sqrt()` routed through `vokra_math` (`5027b2b`), from_gguf loud-fail defensive check for `sbv2.sdp.flows.<even>.*` unread tensors (`879ba8e`), `#[ignore]`d `sdp_body_matches_torch_ref` scaffold as owner-fixture-待ち gate (`c8e2777`).
- **Wave 3** (2026-08-10, 3 commits `315b8f7..3f76abf`): the license-audit.md §3.1 entry for ZH BERT `hfl/chinese-roberta-wwm-ext-large` changed blank → ☑ Commercial by owner delegation (`315b8f7`), CLI `bert-base` arm + `nemo_pt_to_safetensors.py` shared-tensor dedup (`1ea38bd`), fixture sidecar populate for WP-19 4-file loader (`3f76abf`).
- **Wave 4** (2026-08-10, 1 commit `8d469eb`): M4-07 T17/T18 H100 FA v3 bakeoff on vast.ai H100 PCIe (60 min, $1.73, offer #31427212). See §7 (SoTA cross-cutting) for owner ripples — the M4-07 owner ripple is tracked in `docs/m4-owner-verification-checklist.md` §2.1 (dashboard registration only remains).

**Verify snapshot at pre-merge branch tip `8d469eb`**: `cargo test --workspace` = 5447 passed / 0 failed / 22 ignored / 199 suites (baseline 5446/21, +1 test +1 scaffold). All gates green: `cargo fmt` / `cargo clippy -D warnings` / `scripts/check-zero-deps.sh` (root Cargo.lock = `vokra-*` only, NFR-DS-02 preserved) / `scripts/check-abi-changelog.sh` / `scripts/gen-c-abi.sh --check` (no drift, v1.0-rc baseline 33 fn + 11 typedef unchanged, no new C ABI). This is historical evidence; PR #27 is merged.

**2026-08-31 current-state rule**: the earlier **94 unchecked boxes** and the
2026-08-18 **42 checked / 36 unchecked** are historical owner ledgers, not
current counts of missing implementations. The current literal ledger is
**49 checked / 33 unchecked**. Those counts are not an exhaustive task count:
the M5-03/M5-04/M5-05/M5-06 and M5-10…M5-15 GA gates were written as prose
rather than Markdown boxes. The live index below includes both sets. A box can
mean an external legal/infra decision, real-weight access, a deliberately
fail-closed policy, a future backend, or a partially landed implementation that
still lacks its real-checkpoint proof. Only mark a condition complete when its
literal done-condition is evidenced; do not infer implementation status from
the unchecked total.

**Tracking**: this file (`docs/m5-owner-verification-checklist.md`) is **tracked (public)**, same convention as `docs/m3-` / `docs/m4-owner-verification-checklist.md`. Referenced handoffs `docs/handoff/m5-*.md` are tracked/public; specs `docs/tickets/m5/*.md` and ADRs `docs/adr/M5-*.md` are gitignore-local internal docs (referenced by ID).

Each task: **(a)** what / **(b)** why owner-only / **(c)** reference / **(d)** done-when.

---

## 0. Live remaining-work index (2026-08-31)

This table is the complete M5 routing index. The 33 unchecked Markdown boxes
live mainly in §1.5 and §6; the prose-only rows below are equally real and must
not disappear from planning merely because `rg '\[ \]'` cannot count them.

| Scope | Current state | Remaining done-condition / route |
|---|---|---|
| M5-01 / M5-02 | CoreML now executes the complete Whisper encoder submodel; its 2026-08-24 M1 bakeoff recorded 99.63% ANE placement but parity and 2x speed FAIL. QNN remains an SDK-gated zero-op scaffold. | Implement the QNN delegate graph, capture the Hexagon result, then combine its verdict with the recorded CoreML NO-GO for the M5-13 C-export decision (§1.5). |
| M5-03 | ADR **Accepted**; `vokra-vad-micro`, cross-build, host differential, and memory budget landed | Real Cortex-M55/FVP run plus Tier-3/Helium investment decision (§2.2/§2.3) |
| M5-04 | Static-link and no-dynamic-load gate landed | Console NDA, real SDK triple build, and ADR ratification (§4) |
| M5-05 | ADR option (ii) **Accepted**; `f0_extract` placement = core / M5-16; naming migration applied | Legal sufficiency, consent trust root, separate-repository publication, and RVC/GPT-SoVITS sign-off (§3) |
| M5-06 | Decode-only `wfst_decode` and independent OpenFST parity landed | Ratify the Proposed WFST ADR; decide WFST C export at M5-13; decide Google SynthID contract vs OSS alternative |
| M5-07 | Bark/Matcha commercial and StyleTTS2 rejected decisions recorded | No unconditional license-decision task; Matcha implementation remains trigger-gated (§6.9) |
| M5-08 | CPU+Vulkan critical-safe build/SBOM machinery landed | Market positioning, B2B requirements, and M5-11 commercial bundle decision |
| M5-09 | M4-09 ADR chose piper-plus G2P reuse | **Skipped by design**; no Rust-port task while that Accepted decision holds |
| M5-10 | Compliance configuration/documentation exists | Owner legal work and EU AI Act certification evidence |
| M5-11 | Technical product surface exists | Commercial adoption evidence; fundraising is tracked but is not itself a DoD pass criterion |
| M5-12 | GA/DoD machinery and review runbook exist | Satisfy all DoD inputs, record the Go/No-Go result, and declare GA only after they are evidenced |
| M5-13 | Freeze tooling and negative test landed; ABI remains unfrozen | v1.0.0 tag/freeze, `abi-surface` required promotion, delegate/WFST C-export GO/NO-GO (§1.1–§1.3) |
| M5-14 / M5-15 | CPU/quant/UTMOS implementation waves and advisory gates landed to their documented scope | Final same-rig performance/quality sweeps and GA-quality evidence before the NPU bakeoff |
| M5-16 / M5-17 | Explicit trigger-gated homes | Implement only when a named consumer/model/toolchain/hardware trigger fires; currently open concrete implementations are listed in §6.6 |
| Mac CPU/Metal model closure | Five Apple-ready models have strict native CPU routes and independent official VAST evidence: GigaAM v3, GigaAM Multilingual, OmniASR CTC 1B, ReazonSpeech NeMo v2 and BiCodec. Three authenticated packets are held on stopped/exited VAST instances `49168183` and `49261078`; compute is off and storage billing continues. Live inventory is CPU `full=131`, `partial=42`, `no-runtime-binder=20`, `not-artifact=1`; Metal `full=129`, `blocked-by-cpu=62`, `cpu-only=2`, `not-artifact=1`. | Provision the 32 GiB-or-larger Scaleway Apple host, transfer and verify all three packets, run the five Apple CPU/Metal workers, preserve evidence and destroy both VAST instances. This closes only the prepared rows; it does not close the other 62 CPU-blocked repositories. |
| SoTA / parity / publish | Converters and many structural proofs landed | The 33 literal boxes cover NPU capture, parity families, implementation follow-ups, publication/destination policy, Voxtral live correction, and optional Pages deployment |

The cross-milestone Python binding, package distribution, and real-device lab
gaps are tracked outside this file in
`docs/platform-support/v1.0-rc-support-matrix.md`; they still feed M5-12 DoD
and are not waived by this M5 index.

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

- **(a)**: first land executable delegate graph/submodel paths, then run the CoreML (Apple ANE) and QNN (Qualcomm Hexagon) delegates on real hardware and measure the NFR-PF-12 acceptance criterion (≥ 2× over the CPU baseline). Feeds T19. The CoreML complete-Whisper-encoder path and its M1 result are now recorded in `docs/handoff/m5-01-coreml-bakeoff-2026-08-24.md`; QNN still deliberately reports zero executable ops until its official SDK ABI is transcribed.
- **(b)**: QNN implementation needs the SDK-gated graph API transcription and its measurement needs real Hexagon silicon. The current Apple M1 machine supplied the CoreML/ANE result but has neither the Qualcomm SDK/runtime nor a Snapdragon/Hexagon target.
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
- `tools/coreml/generate_whisper_encoder.py` — strict offline GGUF to CoreML
  MLProgram converter for the complete Whisper encoder.
- `tools/coreml/check_placement.sh` — official `MLComputePlan` estimated-cost
  placement gate (constants remain visible but contribute zero cost).
- `vokra-cli npu-bakeoff` — release-only alternating CPU/delegate exact-submodel
  harness that reuses one process, model, input feature tensor, and delegate
  session while enforcing parity and the 2x threshold.

### 1.5.3 Owner runbook (per delegate)

Run this loop once per delegate (CoreML then QNN). Both loops end with a
recorded verdict feeding **§1.3 T19 GO/NO-GO** on the C-ABI symbol call.

**CoreML 2026-08-24 protocol note:** the completed M5-01 run uses the exact
delegated Whisper-encoder unit rather than separate whole-ASR CLI processes.
Its alternating same-session samples satisfy the baseline-discipline intent
more directly. Because the measured encoder-only ratio is 1.422828x, it is an
upper bound on the full hybrid-ASR ratio: adding the same non-negative CPU
decoder time to numerator and denominator can only move the ratio toward 1x.
The clean encoder-only FAIL therefore decides the 2x question without a
second whole-ASR RTF capture. See the dated report for the proof and raw data.

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

- [x] CoreML complete-Whisper-encoder submodel execution and official
  `MLComputePlan` estimated-cost placement probe are wired up.
- [x] CoreML same-session CPU encoder baseline captured (`N=10`, CV
  `0.148269320`).
- [x] CoreML same-session delegate encoder captured (`N=10`, CV
  `0.131539500`, ANE estimated-cost placement `0.996281551`).
- [x] CoreML FAIL / C ABI NO-GO recorded in
  `docs/handoff/m5-01-coreml-bakeoff-2026-08-24.md`.
- [ ] QNN graph construction/execution and placement probe (`qnn-net-run --profiling_option=op` wrapper) are wired up + emit the expected JSON.
- [ ] QNN baseline captured (`cpu`, N=10, CV ≤ 0.20).
- [ ] QNN NPU captured (`qnn`, N=10, CV ≤ 0.20, mean placement ≥ 0.90).
- [x] QNN prerequisite verdict recorded as `INSUFFICIENT DATA` in
  `docs/handoff/m5-02-qnn-bakeoff-2026-08-24.md`; no performance number was
  fabricated without SDK/runtime/Hexagon hardware.
- [x] CoreML FAIL plus QNN INSUFFICIENT DATA fed into §1.3 T19: NPU delegate
  selector C ABI = **NO-GO** for v1.0.

---

## 2. M5-03 — IoT Tier 3 (Cortex-M55 no_std Silero VAD)

CC landed the no_std subset + `vokra-vad-micro` crate + thumbv8m cross-build + host-executable bit-identical differential + memory budget. See `docs/handoff/m5-03.md`.

### 2.1 T02 — crate-topology ADR (complete)

- **(a)**: `docs/adr/M5-03-iot-tier3-nostd.md` is **Accepted** (2026-07-22): topology = new `vokra-vad-micro` crate, all-target transcendental unification, Newton sqrt accepted, scalar Helium default/defer, and split per-PR/weekly cadence.
- **(b)**: completed owner architecture decision; do not reopen it without new evidence and a superseding ADR.
- **(c)**: `docs/handoff/m5-03.md`; spec M5-03-T02.
- **(d)**: satisfied. Real-device performance and the optional raw-asm investment remain T17/T18, not T02.

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

CC landed the contradiction ADR, the consent schema/validator, the flag gate,
and the `vokra-voiceclone-experimental` scaffold seed. The ADR was subsequently
Accepted with option (ii). See `docs/adr/M5-05-watermark-dependency.md`.

### 3.1 T04 — accepted resolution; legal sufficiency + trust root remain

- **(a)**: resolution option **(ii) requirement-side amendment is Accepted** (2026-07-22). Remaining owner work is to judge EU AI Act Article 50 / SB 942 / ELVIS Act / NO FAKES sufficiency and decide the consent-signature trust root (whose key / distribution / revocation).
- **(b)**: a legal-sufficiency + trust-root decision; not a code judgment.
- **(c)**: `docs/adr/M5-05-watermark-dependency.md` §5; spec M5-05-T04.
- **(d)**: satisfied for the resolution option; still open for the legal record and signature-verification policy.

### 3.2 T15 — publish the separate repo + sign-off

- **(a)**: create/publish `vokra-voiceclone-experimental` from the scaffold seed (`staging/vokra-voiceclone-experimental/`, gitignored) and fill the `docs/license-audit.md` §3.1 RVC v2 / GPT-SoVITS sign-off rows (blank = fail-closed). The `f0_extract` site (core), landing WP (M5-16), and `otonx-` → `vokra-` naming migration were already decided/applied on 2026-07-22.
- **(b)**: repo creation/publish and legal sign-off are owner-only; the former WP-number/SSOT decisions are complete.
- **(c)**: `docs/adr/M5-05-watermark-dependency.md`; spec M5-05-T15.
- **(d)**: repo published (flag + consent enforced; the watermark-forced leg follows the accepted option (ii) contract) and sign-off rows filled.

**honest note (watermark leg)**: real watermark embedding remains honestly
Deferred (`WatermarkConfig::backend_status()`, 2026-07-04 drop, BIG-8 held).
Accepted option (ii) amended M5-05 to require the enforced configuration
surface plus deployer-visible disclosure; it did not fabricate an embed. Real
embedding can become active only through a separately approved follow-up WP.

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

**2026-08-18 compute policy**: the recipe above describes the repository's
activation mechanism, not the machine on which new model work should be done.
All multi-GB checkpoint download/conversion/reference work and every
`vokra-models` cargo run now execute on VAST; do not dispatch a heavy hosted
runner leg merely to close this checklist. A VAST result may establish the
technical evidence, while setting each repository variable remains a separate
owner scheduling decision. No Hugging Face upload was performed in the VAST
campaign below.

Original SoTA Phase 1-4 seven families:

- [ ] Family 1 (NeMo-ASR, `VOKRA_NEMO_ASR_ENABLE`): VAST `47955178` converted pinned Parakeet-TDT 0.6B v3 (`7c35754d…`) at branch `afe9b50`: the auditable integer normalizer kept 699 float tensors and removed exactly 24 scalar BatchNorm `num_batches_tracked` counters, then produced a 2,508,284,704-byte GGUF with 38 metadata keys and zero converter-side skips. The corrected exact test selection ran one `parakeet_tdt` test and passed the GGUF/metadata surface. VAST `48335615` later completed Parakeet-CTC-1.1B strict conversion, native JFK transcription and independent Transformers parity with unchanged bounds; see the runtime-gap ledger and committed fixture README. Keep the family open because Kyutai-STT, Canary, and OmniASR-CTC remain unproven in this campaign, and repository scheduling is also undecided.
- [ ] Family 2 (whisper-extras, `VOKRA_WHISPER_EXTRAS_ENABLE`): VAST converted pinned Distil-Whisper Large v3.5 (`728a…`, 539 tensors, 3,025,666,272-byte GGUF) and Kotoba-Whisper v2.2 (`9d334…`, 539 tensors, 3,025,666,304-byte GGUF); both targeted harnesses passed their current GGUF metadata/hparam checks and loudly confirmed the native loader is still absent. Keep open because native transcription and independent output parity are not implemented, and repository scheduling is undecided.
- [ ] Family 3 (tts-dac, `VOKRA_TTS_DAC_ENABLE`): VAST converted pinned Dia 1.6B (`257bc…`, 343 tensors, 6,444,673,088-byte GGUF) and Zonos v0.1 transformer (`9d833…`, 246 tensors, 3,248,843,808-byte GGUF); both targeted scaffold harnesses passed. Keep open because no reference-stage/output numerical parity ran and native synthesis remains a scaffold; repository scheduling is undecided.
- [ ] Family 4 (tts-hiftnet, `VOKRA_TTS_HIFTNET_ENABLE`): VAST converted and passed the current targeted GGUF harness for Chatterbox multilingual (292 tensors, 2,143,980,064 bytes), turbo (299 tensors, 1,915,470,144 bytes), and nano (155 tensors, 869,895,424 bytes), all at pinned revisions. Keep open: reference stage taps were unset, CosyVoice3 still lacks its required torch-to-safetensors sidecar, and repository scheduling is undecided.
- [ ] Family 5 (Qwen3-TTS, `VOKRA_QWEN3_TTS_ENABLE`): VAST converted the pinned 0.6B release (`5d839924…`) to a 478-tensor, 1,829,328,672-byte GGUF. Its targeted harness passed 12 tests and matched the upstream talker (13 axes) and code-predictor (10 axes) config exactly. Conversion now embeds and authenticates the fixed-revision config, byte-BPE and generation sidecars for all five official main checkpoints. The runtime implements the exact Base/CustomVoice/VoiceDesign prompt boundary, bounded mmap autoregressive talker with KV cache, all fifteen code-predictor rows, frame-major sixteen-codebook generation, and an explicit same-backend main + 12-Hz waveform-decoder API/CLI join on CPU or Metal. The separately released pinned tokenizer has a strict 271-tensor decode-only GGUF contract and complete native mapped waveform graph. Keep open because independent real-weight CPU parity and Apple-hardware Metal parity have not run, and the four historical public main GGUFs plus absent public companion still require separately authorized gated replacement/publication; repository scheduling is also undecided.
- [ ] Family 6 (tts-continuous-vae, `VOKRA_TTS_CONT_VAE_ENABLE`): in addition to the prior VAST VoxCPM2 proof, VAST merged all three pinned VibeVoice-1.5B shards with the fail-loud checkpoint merger (1,204 tensors, 2,704,021,987 parameters, zero dropped/shared tensors), then converted the full model to a 5,408,160,960-byte GGUF and passed the targeted harness. The workflow now mirrors that proven full-shard path instead of selecting only the first shard. Keep open because byte-reference taps/native synthesis remain absent and repository scheduling is undecided.
- [ ] Family 7 (tts-japanese, `VOKRA_TTS_JA_ENABLE`): VAST converted pinned Irodori-TTS-500M-v3 (`236c…`) to a 637-tensor, 2,048,247,584-byte GGUF and passed the current targeted harness. Keep open because its byte-reference directory was unset; VITS-JA remains intentionally unfetched and publication-blocked by the JSUT/JVS redistribution terms (§6.8), and repository scheduling is undecided.

2026-07-28 follow-up additions (bringing the variable-gated total to 9):

- [ ] Family 8 (deepfilternet3, `VOKRA_DFN3_ENABLE`): HF-card read (Rikorose/DeepFilterNet MIT/Apache-2.0 dual; §3.1 Commercial decision recorded) → set `VOKRA_DFN3_ENABLE=1` → `gh workflow run parity-deepfilternet3-real.yml -f force_parity=true` → PASS verdict confirmed. The old `VOKRA_DFN3_DATA_URL` blocker is closed: the workflow now creates the independent upstream bundle from a checked-in uv lock. VAST `47955178` proved the exact path on 2026-08-18, then PR #33 run `32069035682` independently completed the first real GitHub Actions verdict with all 21 stage/output bounds green (`enhanced` max |Δ| `4.172e-7`, upstream/Vokra SI-SNR both `14.768 dB`, no tolerance change). The only remaining activation action is setting `VOKRA_DFN3_ENABLE=1` so scheduled runs execute the real leg; the first GitHub Actions proof itself is complete.
- [ ] Family 9 (deberta-v3-large, `VOKRA_DEBERTA_V3_ENABLE`): HF-card read (microsoft/deberta-v3-large MIT; §3.1 Commercial decision recorded 2026-07-27) → set `VOKRA_DEBERTA_V3_ENABLE=1` → `gh workflow run parity-deberta-v3-large-real.yml` → PASS verdict confirmed. The old “no Rust consumer” description was stale: `crates/vokra-bert/tests/deberta_v3_real.rs` landed on 2026-07-29 and consumes the upstream `input_ids` + `final_hidden` dump. The workflow now uses a dedicated Linux-x86_64 uv lock, removes the obsolete `VOKRA_DEBERTA_V3_HARNESS_READY` gate, and defaults matching PRs / enabled schedules / manual dispatches to the real final-hidden numerical leg. VAST `47955178` proved the full path on 2026-08-18, then PR #33 run `32069035556` independently completed the first real GitHub Actions verdict: converter smoke passed, the tokenizer loaded 128,000 pieces, and final-hidden max |Δ| was `1.049042e-5` under the unchanged `6.0e-3` bound. The only remaining activation action is setting `VOKRA_DEBERTA_V3_ENABLE=1` so scheduled runs execute the real leg. Per-layer hidden/attention taps remain a separately disclosed extension, not a missing final-output consumer. See `docs/handoff/parity-deberta-v3-large-real.md`.

- [x] Family 10 (SBV2, sidecar-hash gate): the default `parity-sbv2-real.yml` leg validates main + JA BERT + EN BERT, while explicit `include_zh=true` adds the fourth ZH BERT sidecar and routes a `Language::ZH` request through the four-file loader. VAST `47955178` proved the JA path on 2026-08-18; PR #33 run `32069035448` then supplied the first corrected GitHub Actions verdict. VAST `47977839` subsequently proved the new ZH leg at commit `e564186`: all four regenerated GGUF hashes matched, the upstream `transformers` WordPiece/plain-BERT reference produced `bert_hidden_zh [5,1024]`, and the named non-ignored Rust consumer passed `1/1` in 1026.70 s with no tolerance change (`bert_hidden_zh` max |Δ| `1.907349e-5`, bridge/mel `5.960464e-6`, latent `1.096725e-5`, waveform `1.031446e-1`, mel-loss `1.820711e-1`). The optional UTMOS sub-gate was explicitly skipped and is not a quality PASS; the fixture-only Mandarin input row is numerical replay evidence, not production G2P validation. See `docs/handoff/parity-sbv2-real-vast-2026-08-18.md` and `docs/handoff/parity-ci-flip-switch.md`.

### 6.4 Real-weight parity harness fire

For each landed scaffold that ships a flip-the-switch harness, point the per-family `REFERENCE_DIR` env var (e.g. `VOKRA_HIFTNET_REFERENCE_DIR`) at the real dumped reference tensors and re-run the harness. Per-family env-var names are recorded in the parity CI YAMLs (`.github/workflows/parity-*.yml`).

- [x] Enumerate the current inventory: nine `VOKRA_*_ENABLE` variable-gated workflows (Families 1–9 in §6.3) plus the SBV2 sidecar-hash-gated workflow (Family 10). A scheduled `success` is not evidence of a real run while its gate is closed.
- [ ] For each of the nine variable-gated families, dump reference tensors from real upstream weights, set its `VOKRA_*_REFERENCE_DIR` where applicable, and record the numerical harness result.
- [x] For SBV2, both selected bundles are real-weight proven as of 2026-08-18: the default three-file JA path and the explicit four-file ZH path. The ZH run used the real upstream WordPiece/plain-BERT forward plus the same clean-room downstream SBV2 oracle, matched all four GGUF sidecars, and passed the full Rust manifest consumer without changing a bound. This closes numerical parity only; production Mandarin G2P and JP-Extra multilingual audio quality remain separately disclosed limitations.
- [ ] Record PASS / FAIL per family.

### 6.5 misaki venv setup (Kokoro G2P)

- [x] Provision the Python 3.12 environment from the checked-in lock with `uv sync --project integrations/vokra-misaki-g2p --group parity --frozen`. The old ad-hoc `uv venv` + `uv pip install` wording was stale: `pyproject.toml` / `uv.lock` now pin `misaki[en,ja,zh,ko]==0.9.4`, the English spaCy model, and the Korean `python-mecab-ko` provider omitted by upstream's `ko` extra. VAST instance `47955178` synced the final lock, then EN / JA / ZH / KO real G2P all passed; a second `uv --offline` frozen sync and four-language run passed after explicitly staging NLTK `cmudict`. Open JTalk and NLTK language data remain documented first-use inputs outside the Python wheel lock.
- [x] Export `VOKRA_MISAKI_VENV` = venv path in the runner / dev environment. `parity-kokoro-real.yml` now exports the uv-managed `/tmp/vokra-py-parity`; the VAST verification used `/root/scratchpad/misaki-uv-venv`. Both point at the same frozen project graph rather than a mutable pip environment.

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
- [x] **nvidia/parakeet-ctc-1.1b** (row 268) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED**: `huggingface.co/vokra/parakeet-ctc-1.1b` = live, ~4.25 GB / 1652 tensors, `num_batches_tracked` 42 stripped. **2026-08-22 runtime proof**: strict re-conversion on VAST produced a 4,251,045,248-byte GGUF (`sha256=8cbe063d…13f6d`); native JFK encoder/logits stayed inside the predeclared bounds, 138 raw ids + 26 tokens + full text matched pinned Transformers 5.15.0 exactly, and the public CLI route returned the same transcript. No upload occurred in this runtime run.
- [x] **nvidia/canary-1b-v2** (row 269) — ☑ Commercial 2026-07-28 yousan. **PUBLISHED BUT ARTIFACT-PARTIAL (audit 2026-08-26)**: `huggingface.co/vokra/canary-1b-v2` is live, but both filenames resolve to the same 688-F32-tensor / 24-layer timestamp auxiliary CTC manifest (`sha256(name,shape)=2318daa8…5d65bcb`), not the advertised 32-layer FastConformer + eight-layer Transformer AED. The pinned upstream revision `87bc5265…54bf` is a 6,358,958,080-byte tar (`sha256=ae5ef1bf…431094`) containing `./timestamps_asr_model_weights.ckpt` (2,503,310,314 bytes) **and a separate correct** `./model_weights.ckpt` (3,853,798,427 bytes). The old extractor missed the latter because of its `./` prefix and silently selected the first `.ckpt`; checkpoint selection now normalizes member paths, prefers the unique main member and refuses ambiguity. A tensor-payload-free audit of that main member pins 1,510 state tensors = 1,478 float inference tensors + 32 I64 BatchNorm counters and strict float manifest `a7a50151…ae34`; against the Flash release it has exactly four additional 26-tensor decoder layers and three 16,384-vocab shape changes. The strict complete-main converter, native CPU/Metal binder, 25-language CLI/bench route, and independent official-NeMo worker are implemented on this branch; the partial live GGUF still fails the 1,478-tensor gate. VAST compile/CPU token parity and Apple-silicon real-weight Metal parity remain pending, and no replacement upload is authorized. NVIDIA-EULA overlay decision remains unchanged: weight redistribution is governed by the CC-BY-4.0 card (NOTICE §7 carries NVIDIA credit).
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
- [x] **emotion2vec/emotion2vec_plus_large** — Commercial signed; strict
  real-weight preparation, binding and native CPU/Metal CLI dispatch are
  complete in the 2026-08-26 wave. category=emotion. Remaining: VAST official
  CPU parity and Apple-device Metal parity. The existing public GGUF is the
  destination; no republish is planned by this runtime wave.

**Copyleft (1 entry) — SKU rename + PUBLISHED 2026-07-28**:

- [x] **~~litagin02/style_bert_vits2~~ → `litagin/Style-Bert-VITS2-2.0-base-JP-Extra`** (SBV2 v2 JP-Extra 2.0 base, license-audit.md §3.1 **row 315**, replaces the deprecated row 302 reference above) — AGPL-3.0 ☑ Commercial 2026-07-28 yousan (依頼者許可 = CC 判断). **PUBLISHED**: `huggingface.co/vokra/sbv2-v2-jp-extra-base` = live. SKU rename rationale (per row 315 audit note): original `litagin02/style_bert_vits2` = typo (correct author = `litagin`, upstream returns 404) + actual distribution is JP-Extra 2.0 base (the current SBV2 v2 mainline), not the 1.0 multilingual base (which has no HF `cardData.license` and is defer-blocked fail-closed). Publish path used = T3 Copyleft gate (`publish-one.sh --license-spdx agpl-3.0 --acknowledge-copyleft --push` = LICENSE full text bundled + NOTICE + SOURCE.md + `--acknowledge-copyleft` opt-in flag). Fixture-status prerequisite (Blocker 2b/2c per `tests/fixtures/sbv2/README.md`) is fully resolved by 2026-08-10 Waves 1-2 (see §7 below); the residual `#[ignore]`d `sdp_body_matches_torch_ref` scaffold (commit `c8e2777`) remains an owner-fixture-待ち gate for the SDP parity flip, but is not a publish blocker for the AGPL-3.0 weight itself.

**Remaining converter / real-weight gates (signed)**:

- [x] **Suno Bark** — MIT signed 2026-07-23 yousan. `models::bark`, `ModelKind::{Bark,BarkSmall}`, and CLI dispatch support full and small variants. The upstream torch-pickle checkpoint must be flattened to safetensors by a UV-managed sidecar before the real-weight round-trip; the EnCodec companion remains research-only. Any conversion/publish run with material memory use follows the vast.ai policy.
- [ ] **Matcha-TTS** — MIT signed 2026-07-23 yousan, but the maintained design remains a conditional Draft and explicitly forbids landing a converter while the defer decision holds. Re-open only with (1) an owner-recorded trigger, (2) a ≥95% piper-plus phoneme-set coverage report, and (3) primary-source confirmation for the paired LJ Speech HiFi-GAN; then follow W0–W9 in `docs/superpowers/specs/2026-07-28-matcha-tts-design.md`. Until then `matcha.rs`, registry registration, parity workflow, and publish stay absent by design.
- [x] **WavTokenizer** — the `ModelKind::WavTokenizer` converter and CLI dispatch landed in the 2026-08-01 codec wave. The remaining future work is Lightning `.ckpt` → safetensors preparation and real-checkpoint parity, not implementation of the converter.

**VoxCPM2-2B — complete real-weight conversion done; numerical parity and
destination-gated publish remain**:

- [x] **openbmb/VoxCPM2-2B** (license-audit row 296) — Apache-2.0 signed
  2026-07-28 yousan. VAST execution on 2026-08-18 found that the upstream
  loader merges two required weight files: 577 BF16 main tensors plus 311 FP32
  `audio_vae.*` tensors from `audiovae.pth`; the earlier main-only conversion
  was incomplete. Commit `5bc62ae` added a hash/count/dtype/config-pinned
  UV preparer, required tokenizer embedding, and strict rejection of incomplete
  2B artifacts. Instance `47955178` produced a 4,956,973,816-byte complete
  safetensors (`f8c8ed28…`) and a GGUF v3 with 888 tensors / 60 metadata keys /
  4,960,621,760 bytes (`1cdea939…`). Independent header verification confirmed
  577 BF16 + 311 F32, all AudioVAE sentinels, tokenizer length 3,676,772, exact
  upstream revision, and Apache-2.0 provenance. The raw 577-tensor checkpoint
  now fails with no output. The real Rust structural parity leg initially found
  a BOOL-vs-integer bug in its own `residual_lm.no_rope` assertion; `e8d016f`
  corrected it and the VAST rerun passed (888 tensors / 2B runtime config /
  explicit synthesize refusal). No `REFDIR` or native forward was present, so
  numerical-output parity remains open. `publish-one.sh` was run without `--push` and stopped
  at `UNKNOWN_REPO`: the signed license row permits redistribution, but the
  official voice-cloning positioning still needs an owner decision against the
  M5-05 separate-repository policy before any destination slug is registered.
  The instance is stopped with the artifact retained. Remaining work is
  independent upstream numerical output parity, destination/legal ratification,
  explicit upload authorization, live verification, and CI flip. This already-
  checked implementation row does not change the 36-box action-ledger total.
  Full evidence/runbook: `docs/handoff/vast-ai-publish-voxcpm2-2b.md`.

**Deferred by RAM constraint (implemented + signed, host infrastructure blocked)**:

- [ ] **Voxtral-Small-24B-2507** (row 251) — Apache-2.0 signed 2026-07-23 yousan. **Adapter-aware 48-GB conversion, publish dry-run, and real ASR/runtime parity completed on vast.ai 2026-08-18** from pinned upstream commit `da5b42409f279fdd92febee0511a6c32828569c1` (11 shards only; duplicate `consolidated.safetensors` excluded). The first provenance-only dry-run artifact (`52f860…`) lacked active adapter metadata and was deliberately not uploaded. The corrected streaming conversion uses the tracked Small-24B side-car and produced 852 tensors / 54 metadata keys / 851 exact BF16 passthrough / 0 skipped / tokenizer embedded / `adapter=frame_stack_mlp` / 48,542,409,248 bytes / SHA-256 `91f2733492dd49b8e8f810192c77538d7d6d2f4c1c568098e11c3ad91f752c87`; peak RSS was 1,780.18 MiB with a 1,280 MiB largest tensor. Header, §3.1, model-card, LICENSE, NOTICE, SOURCE and all no-credential publish gates pass. Independent upstream fixtures are committed under `tests/parity/voxtral-small-24b-2507/`: mandatory two-layer orchestration self-check was bitwise; Vokra tower parity measured mel `1.311e-5` (atol `5e-5`), encoder `2.956e-5` (atol `1.5e-3`), projector `1.812e-5` (atol `6e-5`); decoder logits measured `6.356e-4` (unchanged atol `1e-2`), and all 27 greedy ids matched exactly with EOS in 5,292.21 s. Reference peak was 130.43 GiB; the cgroup peak was 139,312,283,648 bytes, entirely on VAST. Commit `7640a02` was fast-forwarded back to VAST and its bounded fixture smoke passed 4/4. The existing live HF artifact at `vokra/voxtral-small-24b-2507` remains invalid for completion because it carries stale false Mini-3B provenance. Instance `47955178` is stopped (`exited`) with the corrected staged artifact retained pending explicit authorization to transfer the HF credential and run `publish-one.sh --push`; never move the 48-GB artifact or upload work to the M1 iMac. The only remaining literal done-condition is corrected-artifact upload/live verification, so this box remains open and the action-ledger total remains 36.

**BF16 fleet — dispatch complete; real-weight and policy work remain**:

The twelve signed non-voice-conversion converters (`crates/vokra-convert/src/models/kimi_audio.rs` etc.) have `ModelKind` entries, licensed `convert_file` dispatch arms, and `vokra-cli` model parsing/help coverage (PR #27). The four voice-conversion families remain deliberately excluded from the main public distribution path pending their destination decision. Dispatch completion does not claim a runnable native forward: the remaining follow-up is real checkpoint preparation, tensor binding/parity, and the applicable five-gate publish run.

**Voice-clone territory (4 rows: openvoice_v2 / knn_vc / freevc / meanvc) — ELVIS Act policy defer**:

Per [`legal-compliance.md`](legal-compliance.md) and the M5-05 separation ADR,
voice-cloning is intentionally excluded from the `ayutaz/vokra` public repo
pending owner legal ratification under ELVIS Act §3 (Tennessee, 2024-07-01)
and the federal NO FAKES proposal. These four converters should either move to
`vokra-voiceclone-experimental` (M5-05 T15 owner-only) or be explicitly
Rejected in §3.1. Owner action: choose the destination.

---

## 7. SBV2 v2 3-language full publish (2026-08-10; reconciled 2026-08-17)

**Status**: PR #27 merged as `0937ef874495465bdadf18d5511f14e6e2a0ab71` on 2026-08-11. All 4 SBV2 Blockers (2b / 2c / 3 / 5) closed on CC side + ZH BERT license sign-off delivered via owner delegation. SBV2 v2 3-language full publish achieved (JA / EN / ZH BERT + base = 4 models on `huggingface.co/vokra`). See the header addendum for the pre-merge wave ledger.

### 7.1 Published models (4 SKUs, all live on huggingface.co/vokra)

- [x] **`huggingface.co/vokra/sbv2-v2-jp-extra-base`** (SBV2 v2 base, AGPL-3.0, license-audit.md §3.1 row 315) — signed 2026-07-28 yousan, published via T3 Copyleft gate (`publish-one.sh --acknowledge-copyleft --license-spdx agpl-3.0 --push`). Owner ripple: none (publish complete, SA cascade obligation documented in NOTICE + README front-matter).
- [x] **`huggingface.co/vokra/deberta-v2-large-japanese-char-wwm`** (SBV2 v2 JA BERT, CC-BY-SA-4.0, license-audit.md §3.1 row 316) — signed 2026-08-06 yousan (owner delegation, T3 Copyleft path with SA cascade disclosure in NOTICE + README). Owner ripple: none (publish complete).
- [x] **`huggingface.co/vokra/deberta-v3-large`** (SBV2 v2 EN BERT, MIT, license-audit.md §3.1 row 317) — signed 2026-07-27 yousan, published via standard permissive path. Owner ripple: none (publish complete).
- [x] **`huggingface.co/vokra/chinese-roberta-wwm-ext-large`** (SBV2 v2 ZH BERT, apache-2.0, license-audit.md §3.1 row 318) — signed 2026-08-10 yousan (owner delegation "モデルは公開してください（code license に影響がない限り）"), published via standard permissive path. Runbook = `docs/handoff/zh-bert-publish-2026-08-10.md` (gitignore-local). Fixture sidecar `tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf.sha256` populated for WP-19 4-file loader (commit `3f76abf`). Owner ripple: none (publish complete).

### 7.2 Blocker 2c residual — repeatable VAST real-parity gate

- [x] **`sdp_body_matches_torch_ref` VAST real-parity gate** (Blocker 2c residual, commit `c8e2777` = `#[ignore]`d scaffold). The repeatable worker `scripts/publish/vast-ai/run-sbv2-sdp-parity.sh` and lifecycle runbook landed in `223bb13`, with non-interactive tool-path hardening in `485d690` and fail-closed CPU model/ISA provenance in `cdfb3e2`. Its final VAST execution on 2026-08-18 pinned the three upstream revisions (`a731761…` / `547b0e8…` / `64a8c8e…`), regenerated and hash-verified all three GGUFs against the committed sidecars, generated the independent MIT-reference body fixture, recorded Xeon E5-2699 v4 + AVX2 + torch 2.13.0 before the numeric result, and passed the explicit ignored Rust test in 79.90 s: `max |Δ| = 8.583068848e-6` at channel 118 / time 48, below the unchanged strict `1e-5` candidate bound. The earlier manual VAST run measured `9.536743164e-6`; this environment-qualified spread is why the bound was not changed. Real GGUFs and derived raw bytes remain gitignored, and the test correctly remains explicit `#[ignore]` on clean checkouts. Text logs were collected, instance `47953638` was destroyed, and the account was verified at zero running instances. Evidence/runbook: `docs/handoff/sbv2-sdp-vast-parity.md`.

### 7.3 PR #27 merge

- [x] **Review and merge PR #27** (`feat/sbv2-voxtral-real-verify-2026-08-06` → `main`), merged as `0937ef874495465bdadf18d5511f14e6e2a0ab71` on 2026-08-11. The historical branch-tip verification snapshot above remains the evidence for the merge.

### 7.4 M4-07 X-06 nightly dashboard registration (cross-cutting)

- [x] **Register the FA v3 vs FA v2 dashboard row**: `tools/bench/build_dashboard.py` now renders `e2e_speedup_summary.fa_v3_vs_fa_v2_e2e_median` from `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` as `1.0573x` in the GPU table; its test pins the value. This satisfies the code/artifact part of M4-07 T18 without inventing a benchmark result.
- [ ] **Owner deployment gate**: enable GitHub Pages and set `VOKRA_PAGES_ENABLED=true` if the dashboard must be publicly deployed. Until then `dashboard.yml` still produces the downloadable dashboard artifact, and no public-deployment claim is made.

---

## 8. Parity-oracle dependency upgrade (2026-08-28)

**Status**: every pinned reference toolchain under `tools/parity/**` was audited
against the GitHub Advisory Database — 401 unique package/version pairs across
27 trees, of which 16 carried a moderate-or-higher advisory. Fourteen of the
seventeen affected trees were upgraded and now resolve clean: `bark`, `dac`,
`deepfake_detection`, `facodec`, `funcodec`, `moss_audio`, `nanocodec`,
`neutts_air`, `pyannote_diarization`, `pyannote_segmentation`, `speecht5_tts`,
`speechtokenizer`, `t5_encoder`, and `ultravox`. Transformers moved to `5.5.0`,
torch to `2.13.0`, `sentencepiece` to `0.2.2`, `setuptools` to `84.0.0`, and
`hydra-core` to `1.3.5`. Each dumper's fail-closed `TRANSFORMERS_VERSION` guard
and Bark's pinned Transformers source revision were updated with them.

`dac` moved only `protobuf` (3.19.6 to 7.36.0) through a `tool.uv` override:
its numeric path is unchanged, verified by diffing the lockfiles, so the
committed 16/24/44.1 kHz fixtures stay valid.

Three trees cannot be upgraded because their newest upstream release still
hard-pins a vulnerable dependency:

| Tree | Upstream pin | Residual |
| --- | --- | --- |
| `qwen3_asr` | `qwen-asr==0.0.6` (latest) requires `transformers==4.57.6` | 3 advisories |
| `parler_tts` | `parler-tts==0.2.2` requires `transformers==4.46.1` | 16 advisories |
| `xcodec2` | `xcodec2==0.1.5` (latest) requires `torch==2.5.0` | 4 advisories |

`xcodec2` would accept Transformers 5.5.0, but its torch pin keeps the tree
flagged either way and the bump would void five committed fixtures, so it was
left alone. The union of the three residues — 20 exact GHSA ids — is
allow-listed in `.github/workflows/ci-security.yml`. torch and Transformers
appear nowhere outside `tools/parity/**`: the Rust runtime carries no
dependencies (enforced by `scripts/check-zero-deps.sh`) and the published
Python wheel declares `dependencies = []`, so no shipped artefact is exposed.

- [x] **VAST verification of the upgraded oracles** (2026-08-28, instance
      `48950897`, destroyed after log recovery, account verified at zero running
      instances). All sixteen touched trees installed from their committed
      lockfiles, executed every `transformers`/`torch` import their dumper
      declares — at module scope and inside functions — and ran that dumper's
      argument parser: 16 pass, 0 fail. Log SHA-256
      `f4f295abe4140bb6d87087608082657a0c4ac651fe170ad63af71548d830c1c3`.

      The run found one real defect, and it predates this branch's dependency
      work: `parler_tts` resolved `torch` from the `pytorch-cpu` index while
      `torchaudio` came from PyPI, so `_torchaudio.abi3.so` could not load and
      the oracle could not start. The same cross-index split existed at
      `torch 2.5.1+cpu`, so that oracle had never run on this branch. Routing
      `torchaudio` through the same index fixes it. Measurement, not version
      arithmetic, settled this: `torch 2.13.0` with `torchaudio 2.11.0` loads
      fine in six other trees where both come from one index.

      This exercises installation and the import surface. It does not re-derive
      any reference tensor, so a numerical change inside Transformers 5.5.0
      would not be caught here.
- [ ] Re-check the three blocked trees when `qwen-asr`, `parler-tts`, or
      `xcodec2` publish a release that relaxes its pin, and drop the
      corresponding ids from the allow-list.

---

## 9. GGUF producer stamp after the 0.2.0 bump (2026-08-28)

Every GGUF carries `general.schema_producer = "vokra-core <CARGO_PKG_VERSION>"`,
written by `GgufWriter` so the stamp always describes the build that produced
the bytes (`vokra-core::gguf::schema::tests::every_builder_written_gguf_is_stamped`
pins this). Opening the 0.2.0 line therefore changes the bytes of every
regenerated GGUF, and with them the committed SHA-256 sidecars.

`parity-sbv2-real` passed on `32efad34`, the commit immediately before the
bump, and the only change from there to `4e59e12b` is Cargo version metadata,
so the producer string is the whole delta. The three sidecars that job
regenerates on a pull request were re-pinned to the values that run measured.

- [ ] **Re-pin `tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf.sha256`**.
      Its artefact is only rebuilt when `parity-sbv2-real` is dispatched with
      `RUN_ZH=true`, so no run has produced the 0.2.0 value yet and the sidecar
      still holds the 0.1.0 one. The next ZH dispatch will fail on it, which is
      the intended fail-closed behaviour; re-pin from that run.
- [ ] Published GGUFs on `huggingface.co/vokra` carry the 0.1.0 stamp. Nothing
      republishes them automatically, but any future re-upload will differ in
      this field from the artefact currently recorded in
      `docs/license-audit.md`.
