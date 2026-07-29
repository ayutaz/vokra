# Residual wave 3 (2026-07-30) — owner handoff

Tracked / public. Comprehensive summary of the M5-gap "residual wave 3" that
landed on branch `feat/model-publish-and-m5-gap-2026-07-29` between the
previous wave 2 tip (`aa9df4e chore(ci): fix 2 wave-2 advisory reds`,
2026-07-29) and this handoff. The wave picked up the CC-side pass on the
"public-facing model coverage gap" that the 2026-07-28 audit
(`docs/handoff/model-publish-and-parity-2026-07-28.md`) enumerated but
which the two earlier CC-side waves had left un-actioned.

Convention: this file is the operational counterpart to the wave commits.
For per-model publication readiness see `docs/handoff/publish-unhandled-2026-07-28.md`,
for per-parity-family status see `docs/handoff/parity-ci-flip-switch.md`,
for the licensing SoT see `docs/license-audit.md` §3.1. `CLAUDE.md`
(gitignore-local internal SoT) mirrors this handoff's summary bullet in
its "現在のタスク状態" section.

## Executive summary

* **11 items landed, 1 partial, 0 skipped, 0 blocked** across 17 physical
  commits (some wave items ship as commit clusters — see per-wave table).
* **New model coverage** — 3 F0 pitch extractors (RMVPE / FCPE / CREPE,
  all MIT), FSMN-VAD (FunASR MIT — first-class alternative to Silero),
  Silero v6.2.1 (backward-compat variant switch, v5 still selectable),
  TitaNet-L (NVIDIA CC-BY-4.0), VoxCPM2-2B (Apache-2.0 multilingual),
  StyleTTS 2 scaffold, Canary-Qwen-2.5B ASR, omniASR-CTC {300M, 7B}
  variants, Qwen3-TTS 1.7B variant, Charsiu forced-alignment (real
  wav2vec2 CTC forward), and 3 new first-class encoder / decoder ops for
  the JA-ASR family (Zipformer, E-Branchformer, hybrid CTC/Attention
  decode with LSTM LM shallow fusion).
* **1 partial** — RMVPE — the module + converter + real mel front-end +
  360-class → Hz decoder are all real (not fake), but the U-Net + GRU
  internal forward binding to real weight is deferred to an owner
  real-checkpoint parity wave. `RMVPE::extract_real` returns
  `VokraError::UnsupportedOp` (FR-EX-08 loud-partial), not a silent
  approximation. See per-wave notes below for the honesty rationale.
* **Zero new C ABI symbols** — every new surface is Rust-only. The
  v1.0-rc baseline (33 exported functions + 11 typedefs) is unchanged.
  `check-abi-changelog.sh` and `gen-c-abi.sh --check` both green.
* **Zero-dep NFR-DS-02 preserved** — root `Cargo.lock` remains `vokra-*`
  only; every Python dependency lives in `tools/parity/` uv-managed
  venvs (Python 3.12 per memory `feedback-python-3-12`); every workflow
  ends with a `git diff --exit-code Cargo.lock` tripwire.
* **License sign-off (yousan-delegated)** — the audit rows for the F0
  family (RMVPE / FCPE / CREPE) landed with primary-source-verified
  `☑ Commercial 2026-07-30 yousan` per memory
  `feedback-license-signoff-primary-source`; the batch commit
  (`b15f69b`) also progressed the BF16 fleet + Wave 2 backlog.

## Per-wave table

Each row is one wave item (some items map to multiple physical commits).
The Wave A/B/C/D letters record the parallel-worktree bucket the CC side
used when driving the ultracode dispatch — they carry no long-term
meaning and can be dropped once the M5 GA cut is done.

Original worktree-branch SHAs (from the wave orchestrator's initial ledger) are noted in parentheses. Nine of the twelve items had already been cherry-picked onto the integration branch by earlier waves; the remaining three bundles (JA-ASR ops / StyleTTS 2 + Charsiu / Canary-Qwen bundle) sat on the isolated worktree branches (`worktree-wf_779368da-fa3-14/15/16`) with the correct commit content but were not yet reachable from the integration branch tip at wave-3 hand-in. **This handoff commit is preceded by 8 cherry-picks (`d3f30a0` / `9c65324` / `7537c37` / `f7838e4` / `c53a742` / `6cb59ff` / `136a1dd` / `3742570`) that reconcile the wave's ledger with the integration branch reality.** Each of these preserves the original commit author, message, and file-level diff; the only merge-conflict resolutions were the auto-generated `<<<<<<<` markers in the `vokra-convert` CLI help strings (both edits merge `styletts2` + `canary-qwen` in the model-kind alternation) and the `docs/license-audit.md` section-header ordering (both blocks retained, no data lost).

| # | Wave | Integration SHA (worktree SHA) | Item | Status | Deferred_reason (owner) |
|---|------|---------------------------------|------|--------|-------------------------|
| 1 | A | `b15f69b` | `docs/license-audit.md` §3.1 batch sign-off — BF16 fleet + Wave 2 (owner-delegated) | landed | — |
| 2 | A | `5f7cb15` | Wire 5 BF16 fleet converters (openvoice_v2, freevc, meanvc, ecapa_tdnn candidate, wespeaker) into `ModelKind` + CLI dispatch — per §6.9 of `docs/m5-owner-verification-checklist.md`, converter skeletons remain landed but publishing is still owner-signoff-gated | landed | Per-row §3.1 sign-off for voice-clone territory (openvoice_v2 / freevc / meanvc) still owner-only — ELVIS Act separation to `vokra-voiceclone-experimental` |
| 3 | A | `e369dde` | VoxCPM2-2B multilingual variant — Option C hybrid (single-module + config-dispatch, per gitignore-local spec) — closes the "Publish 前提条件" on §3.1 row 280 | landed | Publish upload is owner (real checkpoint fetch + upstream 30-lang list re-verify) |
| 4 | A | `f84729f` | TitaNet-L converter + §3.1 sign-off (NVIDIA CC-BY-4.0 primary source) — reserves `titanet_speaker_encode` residual op, NOTICE §7 records NVIDIA attribution | landed | Real-weight publish (vast.ai for large sizes) + `TITANET_SPEAKER_ENCODE_OP` fixed-op landing (currently residual anchor per `m5_residual_ops.rs`) |
| 5 | B | `4e02235` | RMVPE F0 pitch extractor — converter, config, weight loader, real mel front-end (STFT + 128-mel filterbank), 360-class → Hz decoder with local-centroid refinement, parity harness scaffold | **partial** | U-Net + GRU internal forward kernel binding to real weight + owner real-checkpoint parity dumper (`tools/parity/rmvpe_dump_reference.py`) + `huggingface.co/vokra/rmvpe` publish. **Honesty**: `RMVPE::extract_real` returns `VokraError::UnsupportedOp` (FR-EX-08); we did **not** ship a best-guess topology with `from_gguf` shape-mismatch as the loud-fail path because that would commit permanently-red code. See "Wave B honesty note" below. |
| 6 | B | `06a6b53` | FCPE F0 pitch extractor — real Conformer forward + converter (MIT), `tools/parity/fcpe_prepare_checkpoint.py` bridges upstream `.pt` | landed | Owner-side real-checkpoint parity run once `parity-fcpe-real.yml` is landed (follow-up wave) |
| 7 | B | `61ca106` | CREPE F0 pitch extractor — real 6-block CNN + converter (MIT) — closes the 5-size (`tiny/small/medium/large/full`) MIT single-sign-off; adds `tools/parity/keras_h5_to_safetensors.py` Keras export layer | landed | Publish upload for any of the 5 sizes (owner decision per §3.1 sign-off row 306) |
| 8 | C | `9ba2202` | Silero v6.2.1 upgrade with backward-compat variant switch — new variant `silero_v6_2_1`, prior v5 remains selectable, GGUF `vokra.silero.model_version` chunk | landed | Real-weight round-trip verify + `parity-silero-real.yml` matrix update (owner cron) |
| 9 | C | `a5763ce` | FSMN-VAD backend (FunASR MIT) — first-class VAD op alternative to Silero, `crates/vokra-ops/src/fsmn_vad.rs` + `crates/vokra-models/src/fsmn_vad/` + converter + SPEC.md | landed | Real-weight round-trip + `vokra/fsmn-vad` publish |
| 10 | C | `6cb59ff` (`0a45ec3`) + `136a1dd` (`88ad467`) + `3742570` (`6b0effc`) | Canary-Qwen-2.5B via FastConformer + Voxtral-style Qwen decoder + omniASR-CTC 300M / 7B variants of the 1B loader (`capacity factor`) + Qwen3-TTS 1.7B fork of 0.6B loader (`hidden-size fork`) | landed | omniASR-CTC 7B publish (vast.ai) / Canary-Qwen real Qwen decoder tokenizer verify / Qwen3-TTS 1.7B publish |
| 11 | D | `d3f30a0` (`71ca015`) + `9c65324` (`e3f543c`) + `7537c37` (`7a8796f`) | JA-ASR encoder / decoder op family — Zipformer (`vokra-ops/src/zipformer.rs`) + E-Branchformer (`vokra-ops/src/ebranchformer.rs`) + Hybrid CTC/Attention decode with LSTM LM shallow fusion (`vokra-ops/src/hybrid_ctc_attention.rs`) | landed | Model-side wiring into ReazonSpeech-k2 / OWSM binders is a follow-up wave; owner publish of any bound model requires §3.1 sign-off per family |
| 12 | D | `f7838e4` (`735fe9d`) + `c53a742` (`3ef8f57`) | StyleTTS 2 scaffold with weight-consent fail-closed gate + Charsiu real wav2vec2 CTC forward (replaces M5-04 alignment scaffold with full acoustic-model consuming forward) | landed | StyleTTS 2 real weight (voice-consent condition = `Unknown` per §3.1 row 260 = fail-closed until owner accepts); Charsiu real-checkpoint alignment parity run against upstream (owner or CC follow-up wave) |

**Wave B honesty note (RMVPE partial)** — CC deliberately chose the
"loud-partial" posture (`VokraError::UnsupportedOp` from
`RMVPE::extract_real`) over two alternatives that would have looked
"more complete" but sacrificed honesty:

1. **Best-guess topology with `from_gguf` shape-mismatch loud-fail** —
   this pattern would commit code that is 100% loud-fail against every
   real checkpoint until the owner-side real-weight parity wave lands.
   That means CC ships permanently-red code with no CI signal that the
   topology guess is right or wrong. Worse than a defer.
2. **Silent placeholder returning zeros / mean-pitch** — this violates
   FR-EX-08 (no silent fallback) and is exactly the anti-pattern the
   RMVPE PR text calls out ("silent-wrong risk high").

The landed shape (real converter + real mel front-end + real decoder,
loud-partial on the internal forward) preserves every downstream
consumer's ability to load the GGUF and iterate the mel/decoder halves
against synthetic tests today, while making the owner-side real-weight
parity wave the single knob that flips `extract_real` on. Same pattern
the M4-20 T17 DeepFilterNet3 harness uses for the "prep_noisy" Phase B
gate (`docs/handoff/parity-deepfilternet3-real.md`).

## Verify results (as of this handoff commit)

Every gate was re-run on the actual integrated HEAD (not on isolated
worktree agent reports), following the [[project-m4-implementation]]
"verify-on-actual-HEAD" rule.

| Gate | Result | Notes |
|---|---|---|
| `cargo test --workspace` (default features) | see `verify_results.test_default` in return JSON | full workspace including new f0/{fcpe,crepe,rmvpe}, fsmn_vad, styletts2, canary_qwen, omniasr_ctc, align/charsiu, zipformer, ebranchformer, hybrid_ctc_attention tests |
| `cargo test --workspace --all-features` | see `verify_results.test_all_features` | metal + cuda + vulkan feature crates |
| `cargo fmt --check` | pass | clean at handoff commit |
| `cargo clippy --workspace --all-targets -- -D warnings` | see `verify_results.clippy` | |
| `scripts/check-zero-deps.sh` | pass | root `Cargo.lock` = `vokra-*` only |
| `scripts/check-abi-changelog.sh` | pass | v1.0-rc baseline = 33 fn + 11 typedef unchanged |
| `scripts/gen-c-abi.sh --check` | pass | `include/vokra.h` up-to-date (Rust surface additions only) |

**Zero new C ABI symbols** — all wave additions live in Rust surface
(new modules, new ops, converter dispatch entries, CLI subcommands).
The v1.0-rc C ABI freeze prep (M5-13) baseline is unaffected.

## Owner critical path (priority-ordered)

The following owner tasks are **blocked-on-owner** and cannot be
progressed by CC without explicit further permission or physical / paid
infrastructure. Each references its long-form owner-checklist item.

### 1. vast.ai instance provisioning for large-model convert/publish
- **What**: provision a vast.ai instance per `docs/handoff/vast-ai-large-model-publish.md`
  §2 (>= 64 GB RAM, >= 200 GB disk, cheap GPU box for the bandwidth).
- **Why owner-only**: paid infra + HF token export + SSH lifecycle.
- **What this wave adds to the queue**: **Voxtral-Small-24B-2507** (row
  251, 48 GB BF16, still queued from wave 1 A-3) / **omniASR-CTC-7B**
  (large variant landed in commit `88ad467` via `6b0effc` cluster) /
  **Canary-Qwen-2.5B** (`0a45ec3`, ~4.96 GB per `check-model-size.sh`
  verdict `LOCAL_BORDERLINE`) / **VoxCPM2-2B** (converter now supports
  2B via `e369dde`; 2B checkpoint is ~4 GB BF16 = borderline).
- **Reference**: `docs/handoff/vast-ai-large-model-publish.md` §1 sizing
  guide, `docs/m5-owner-verification-checklist.md` §6.9 Publish sign-off
  queue.
- **Done when**: each model is either published (row updated in
  `docs/handoff/publish-unhandled-2026-07-28.md`) or rejected with a
  §3.1 Notes entry.

### 2. Initial `workflow_dispatch` for parity CI (post-PR-#24 merge)
- **What**: GitHub Actions `workflow_dispatch` requires the workflow
  file to be on the default branch (`main`) — the two parity workflows
  landed in wave 1 (`parity-deberta-v3-large-real.yml` — DeBERTa v3
  Phase B) and wave 2 (extended coverage) still cannot be fired from
  the feature branch.
- **Why owner-only**: PR #24 merge decision + workflow-variable +
  secret provisioning (see `docs/handoff/parity-deepfilternet3-real.md`
  §Phase A / §Phase B for the `VOKRA_DFN3_ENABLE` / `VOKRA_DFN3_DATA_URL`
  pattern; DeBERTa v3 uses `VOKRA_DEBERTA_V3_HARNESS_READY=1` per PR #24
  body).
- **Reference**: `docs/handoff/parity-ci-flip-switch.md` (canonical
  flip-switch procedure) + `docs/handoff/parity-deberta-v3-large-real.md`
  + `docs/handoff/parity-deepfilternet3-real.md`.
- **Done when**: each workflow runs at least once on `main` with
  `run_conversion=true` and Phase B active-or-honest-skip, and the
  cron begins hitting the noise floor.

### 3. M5-05 T04 ADR ratification (AudioSeal 0.2 revival)
- **What**: ratify `docs/adr/M5-05-voice-cloning-separation.md`
  (currently Proposed) with the "watermark 強制 leg =
  honest-UNMET" reasoning that CC landed in commit `6dc9f86` (M5
  terminal wave 2026-07-21). CC-side AudioSeal 0.2 revival is code-
  complete but the legal-sufficiency question against EU AI Act
  Article 50 (2026-08-02 applies) + California SB 942 (already in
  force) is a lawyer / product decision, not a CC one.
- **Why owner-only**: legal sufficiency + product policy.
- **Reference**: `docs/handoff/m5-05.md` (gitignore-local ADR) +
  `docs/legal-compliance.md` §1.4 (deployer disclosure).
- **Done when**: ADR Status → Accepted with the strict-embedding vs
  deployer-disclosure trade-off recorded.

### 4. `vokra-voiceclone-experimental` separate-repo publish
- **What**: create the separate repository per `CLAUDE.md` 設計判断 8
  and move the 4 voice-clone converters (openvoice_v2, knn_vc, freevc,
  meanvc) into it — this wave landed the last of the ModelKind wiring
  (`5f7cb15`) so they now compile against the main repo dispatch, but
  the ELVIS Act separation policy says they must not ship in
  `ayutaz/vokra`.
- **Why owner-only**: new repo creation + tool-distributor liability
  policy decision + org access + separate license posture.
- **Reference**: `docs/m5-owner-verification-checklist.md` §6.9 "Voice-
  clone territory (4 rows: openvoice_v2 / knn_vc / freevc / meanvc) —
  ELVIS Act policy defer"; `CLAUDE.md` 設計判断 8.
- **Done when**: 4 converter modules live in `vokra-voiceclone-experimental`,
  the 4 §3.1 rows here are either moved or marked `☑ Rejected — moved
  to voiceclone repo`, and the `ayutaz/vokra` publish tooling refuses
  those slugs.

### 5. NDA / NPU bakeoff / Cortex-M55 real hardware (long-lead M5 items)
- **What**: run the M5-01 CoreML/ANE + M5-02 QNN/Hexagon delegate
  bakeoff on real silicon (the `≥ 2× over CPU baseline` NFR-PF-12 gate)
  and the M5-03 T17 Cortex-M55 silicon / FVP run. This is the long-
  lead item that ultimately unblocks the M5-13 C ABI freeze firing at
  the v1.0 GA tag.
- **Why owner-only**: this machine has neither ANE / Hexagon NPU
  bakeoff rig nor Cortex-M55 devboard.
- **Reference**: `docs/m5-owner-verification-checklist.md` §1.5 (NPU
  bakeoff) + §2.2 (Cortex-M55).
- **Done when**: pass/fail vs the 2× bar is recorded per delegate;
  RTF + RAM measured on Cortex-M55 vs CC's host-executable differential.

### v1.0 GA tag (M5-13 T17) — final owner action, not blocked by this wave
The v1.0 GA tag remains the single one-way owner action that fires the
C ABI freeze. It is downstream of #5 (NPU bakeoff must complete first
to inform the M5-13 T19 GO/NO-GO on `wfst_decode` + NPU delegate C
symbols) and #4 (voice-clone separation must complete first to avoid
freezing the ELVIS-Act-tainted surface). Nothing in this wave changes
the M5-13 gating: the freeze machinery is in place
(`abi-diff.sh --gate`), the baseline is `docs/abi/vokra.h.v1.0-rc-baseline.symbols`,
and this wave added zero C symbols to it.

## Non-goals (do not re-open)

* **Matcha-TTS** — 見送り posture from `docs/m4-scope-expansion-2026-07-13.md`
  is preserved. The design spec at `docs/superpowers/specs/2026-07-28-matcha-tts-design.md`
  is Draft-only (塩漬け) per M5-07. Do not re-open in a residual wave.
* **AudioSeal / watermark strict-embedding leg** — the strict-embedding
  path was intentionally left at "honest-UNMET" in the M5 terminal
  wave (`6dc9f86` 2026-07-21). Owner must pick the trade-off in
  §Owner critical path #3 above before any CC-side re-attempt.
* **RVC v2 / GPT-SoVITS in `ayutaz/vokra`** — these live in the future
  `vokra-voiceclone-experimental` repo, never in the main repo. Same
  ELVIS Act separation as §Owner critical path #4.
* **NNAPI backend** — Google-deprecated (Android 15, 2024-10). Do not
  re-open. Vulkan (M3-02) covers Android GPU.
* **Piper (OHF-Voice/piper1-gpl)** — GPL-3.0 + eSpeak-NG (GPL-3.0)
  double contamination. Use `piper-plus` (owner's MIT fork) only.

## References

* This wave's commits — see per-wave table above (all reachable from
  branch `feat/model-publish-and-m5-gap-2026-07-29`).
* Previous wave handoffs on the same branch —
  wave 1 = `bcf4557` + `12efb13` (M5 gap A-1 / A-2 / A-3), documented
  inline in PR #24 body;
  wave 2 = `b13a3c0` + `aa9df4e` (M5 gap A-4 / A-5 / A-6 / B-1 / B-2
  + CI advisory-red fix), no dedicated handoff (rolled into `CLAUDE.md`
  "現在のタスク状態" internal SoT).
* Adjacent handoffs —
  `docs/handoff/model-publish-and-parity-2026-07-28.md` §5 owner critical
  path is the parent structure this wave feeds into;
  `docs/handoff/publish-unhandled-2026-07-28.md` tracks per-model
  publication readiness (this handoff updates it separately);
  `docs/handoff/vast-ai-large-model-publish.md` is the runbook for
  §Owner critical path #1;
  `docs/handoff/parity-deepfilternet3-real.md` +
  `docs/handoff/parity-deberta-v3-large-real.md` +
  `docs/handoff/parity-ci-flip-switch.md` for §Owner critical path #2.
* License SoT — `docs/license-audit.md` §3.1 (yousan sign-off table);
  the F0 family sign-off rows landed in this wave are rows 305 (RMVPE)
  and 306 (CREPE) per the batch commit `b15f69b`.
* Design specs (gitignore-local internal) —
  `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md` (Option C
  hybrid; landed as commit `e369dde`);
  `docs/superpowers/specs/2026-07-28-wavtokenizer-design.md` (still
  塩漬け, not landed here);
  `docs/superpowers/specs/2026-07-28-matcha-tts-design.md` (見送り, do
  not re-open).
* Milestones —
  `docs/milestones.md` §9 M5-14 / M5-15 for the perf phase this wave
  precedes; §9 M5-16 (f0_extract, this wave lands the CC-side coverage
  minus the RMVPE internal forward);
  §7.2 M3-06 for the Mimi codec attribution (unchanged by this wave).

---

**Handoff commit**: this file lands together with the workspace-wide
verify record as `docs(handoff): residual wave 3 comprehensive summary
+ verify (2026-07-30)` — same commit that closes the wave on the CC side.
