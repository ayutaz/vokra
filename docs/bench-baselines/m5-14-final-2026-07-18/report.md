# M5-14 final measurement record — CPU hot-path optimization (Wave 3 close, 2026-07-18)

- Rig: Apple M1 iMac (8 CPU: 4P+4E, 16 GB), macOS. `active_isa = neon-dotprod`, 8-thread pool.
- Tree: branch `feat/m4-plan-and-wave1`, HEAD `3d71d0d` (M5-14 Wave 2). Release profile (opt3 + fat LTO + 1 CGU).
- Methodology: campaign-1 (`m1-real-weight-eval-2026-07-16`) — **inference-only window** (model load + input read excluded), **1 warmup + 3 measured, median**, RTF = wall / audio seconds. Loadavg recorded per run by the harness (raw logs in `raw/`).
- Quiet windows: light legs ran at 1-min loadavg **1.56–1.83**. whisper base/small/beam ran back-to-back at 1.65–3.6 (residual **self**-load from preceding legs; per-iteration spreads ≤3% except small 6.6%). whisper **medium/turbo were re-run standalone** because their first sweep slots started at loadavg 3.6/4.4 (kept as `raw/*.sweepslot.log`): the medium re-run started at loadavg **1.59** (iters 8.503/8.270/8.344, spread 2.8%), the turbo re-run at **1.86** (iters 10.091/10.100/10.381, spread 2.9%, iter0 ≈ median). End-loadavg values (3.9/4.2) are the runs' own 8-thread self-load, not foreign contamination — the tight iteration spreads are the cleanliness evidence. `raw/*-quiet.log` are the table numbers.
- ORT / torch reference numbers are the **frozen same-rig references** (campaign-1 2026-07-16 + Wave-0 2026-07-18 re-measure, onnxruntime 1.19.2 CPU EP); they were **not** re-measured in Wave 3. Where campaign-1 and the Wave-0 fresh two-phase re-measure disagree (whisper-small: 2.52 s vs 1.842 s) both are shown and the gap is a range.

## Final table (jfk-30s.wav = 11.00 s primary; piper = 2.055 s JA utterance; Mimi = 10.96 s @ 24 kHz; DFN3 = 11.0 s @ 48 kHz)

| model (leg) | Wave-0 baseline | now (Wave 3, quiet) | speedup | vs-ORT gap before → after | target (ADR D6) | met? |
|---|---|---|---|---|---|---|
| whisper base greedy | 1.943 s (RTF 0.177, campaign-1) | **0.810 s** (RTF 0.0737) | 2.40x | 0.98x → **0.41x** (now 2.5x FASTER than ORT 1.99 s) | none (was already ~parity) | MET (bonus) |
| whisper small greedy | 7.358 s (RTF 0.669) | **3.126 s** (RTF 0.284) | 2.35x | 2.9–4.0x → **1.24x** (vs campaign ORT 2.52 s) / 1.70x (vs Wave-0 fresh ORT 1.842 s) | <=1.5x | MET vs frozen campaign ORT; 1.70x vs the tighter Wave-0 ORT re-measure |
| whisper medium greedy | 46.28 s (RTF 4.21) | **8.344 s** (RTF 0.759) | 5.55x | 6.5x → **1.17x** (ORT 7.15 s) | <=1.5x | MET |
| whisper turbo greedy | 49.1 s (campaign; Wave-0 staged 39.5 s) | **10.100 s** (RTF 0.918) | 4.86x (vs campaign) / 3.91x (vs staged) | 1.45–1.81x → **0.37x** (now 2.7x FASTER than ORT 27.2 s) | <=1.3x | MET |
| whisper small beam=5 | 35.68 s (4.85x greedy) | **8.460 s** (2.71x greedy) | 4.22x | — (no ORT beam leg) | beam5 <=1.6x greedy | NOT MET — near-miss #2 (2.71x vs 1.6x; mechanism below) |
| whisper small beam=1 | — | 3.143 s (+0.6% vs greedy) | — | — | beam1 == greedy (bit-identical) | MET (asserted in tests + 0.6% overhead) |
| piper css10-JA | 338 ms (RTF 0.165) | **144.3 ms** (RTF 0.0702) | 2.34x | 5.1x → **2.17x** (ORT RTF 0.0324) | <=2x primary / <=2.5x pre-authorized fallback (RTF <=0.081, ADR D6) | MET on fallback (0.0702 <= 0.081); primary 2x missed by 8% |
| Silero VAD (jfk stream) | 241.0 ms (RTF 0.0219) | **37.7 ms** (RTF 0.0034) | 6.39x | 2.7x → **0.43x** (now 2.3x FASTER than ORT 88.6 ms) | <=1.5x | MET |
| CAM++ embed (jfk) | 293.8 ms (+20.1 fbank) | **145.1 ms** (+19.9 fbank) | 2.03x | 4.4x → **2.18x** (ORT default 66.6 ms); 1.7x → **0.83x** vs ORT intra=1 (175.5 ms) | <=2x (<=133 ms) | NOT MET — near-miss #1 (145 vs 133 = +9%; mechanism below) |
| Mimi encode (10.96 s) | 5.488 s (RTF 0.501) | **1.273 s** (RTF 0.116) | 4.31x | 11.6x → **2.70x** (1T-torch 0.472 s, AMX caveat) | <=3x vs 1T-torch (RTF <=~0.13) | MET |
| Mimi decode | 4.167 s (RTF 0.380) | **1.062 s** (RTF 0.0969) | 3.92x | 9.6x → **2.44x** (1T-torch 0.435 s, AMX caveat) | <=3x (RTF <=0.12) | MET |
| DFN3 (11 s, 48 kHz) | 1.974 s (RTF 0.180) | **0.452 s** (RTF 0.0411) | 4.37x | 7.7x → **1.77x** (torch RTF 0.0233, milder AMX exposure) | RTF <=0.06 | MET |
| Kokoro | not measured (out of WP scope) | not measured | — | — | no target (BiLstm1d scalar pin respected, ADR D5) | n/a — parity 8/8 re-verified instead (see below) |

Transcripts: whisper greedy token sequences are identical across all iterations and match the campaign baselines (harness asserts iteration-equality; the gated `greedy_transcript_parity` suite re-verified all 4 sizes against ORT-matching references on real weights this session).

## Near-misses (explicit Wave-3+ backlog, mechanisms measured)

1. **CAM++ embed 145 ms vs 133 ms target (+9%)** — residual is the thin-m (m<=8) B re-pack inside the Wave-1 packed-GEMM driver: every call re-packs the same B panels because pack reuse is per-call, not per-weight. Fix (Wave-3+): pack-once-share — cache packed B keyed by weight identity for conv/linear layers called repeatedly with the same B.
2. **whisper beam5 = 2.71x greedy vs 1.6x target** — Wave-2's incremental per-beam KV (snapshot/restore + generational cache) removed the O(L^2) full-prefix recompute (4.85x → 2.71x), but each step still issues `beam_width` separate m=1 decoder forwards. Fix (Wave-3+): batched-beam forward — fold the per-beam steps into one m=beam_width GEMM family per layer (the Wave-1 driver's thin-m path already handles m=2..8; the remaining work is model-side glue).

## Not-taken decisions (deliberate, recorded)

- **SIMD log-mel rewrite: not taken.** log-mel is ~0.7% of the whisper-small wall (65.8 ms of 7.36 s at Wave 0; absolute cost unchanged now) and a vectorized rewrite would change STFT/mel accumulation order — a numerics change on the bit-exact `frontend_spec` surface for a sub-1% total win. Rejected on parity-risk/benefit grounds.
- **Kokoro: untouched end-to-end.** BiLstm1d scalar pin (M2 T17-fixup #2) respected; no Kokoro-path kernels were rerouted. Gated parity suite re-run this session: 8/8 tensors within bounds (text_encoder 1.31e-6, bert 1.01e-5, prosody f0 3.01e-3 / n 5.13e-5 / hidden 8.53e-5 at atol 0.01; decoder mag 2.51e-1 / phase 1.49e-1 / pcm 1.92e-2 within mag 0.5 / phase 0.3 / pcm 0.04 bounds).
- **mel_frontend_baseline.json (CI gate file): NOT regenerated.** See "Baseline JSONs" below.

## Total regression matrix (T26, this HEAD, all green)

| suite | result |
|---|---|
| full workspace default (`cargo test --release`, 116 suites incl. doctests) | **2551 passed / 0 failed / 4 ignored** |
| Metal feature suite (`-p vokra-models --features metal`, real M1 GPU) | **838 passed / 0 failed** (GPU-vs-CPU parity unshifted — Waves 1/2 are bit-identical on the CPU side) |
| vokra-server excluded workspace (10 suites) | **245 passed / 0 failed** (vllm_compat loopback flake did not fire) |
| real-weight gated: whisper base/small/medium/turbo rows (`parity_whisper`) | 8 passed / 0 failed (319.8 s; encoder / greedy-transcript / decoder-logits / beam1==greedy on real GGUFs) |
| real-weight gated: Kokoro (`parity_kokoro`) | 8 passed / 0 failed — **8/8 tensors within bounds** (unshifted) |
| real-weight gated: DFN3 (`parity_denoise_dfn3`, real weights + staged taps) | 1 passed / 0 failed |
| real-weight gated: Mimi PCM roundtrip (freshly re-converted GGUF, byte-identical to the campaign artifact) | 1 passed / 0 failed |
| real-weight gated: Mimi + DAC full-table codec parity (`real_codec_parity`) | 2 passed / 0 failed |
| real-weight gated: CosyVoice2 LLM forward vs transformers reference | 6 passed / 0 failed (see note) |
| real-weight gated: Voxtral real-GGUF load + greedy step (9.4 GB BF16) | 1 passed / 0 failed (164 s); `parity_voxtral` smokes skip cleanly (reference fixture dir `tests/parity/voxtral` never landed — pre-existing owner item) |
| real-weight gated: Moshi staged reference (truncated 2-layer real dump) | 2 passed / 0 failed |
| real-weight gated: CAM++ (`speaker::parity`) + Silero real-speech ctx576 | passed (env-gated CAM++ leg ran; Silero ctx576 fixtures in default suite) |
| gates | check-zero-deps OK / check-abi-changelog OK (v1.0-rc baseline 33 fn + 11 typedefs unchanged — Waves 1–3 added no C ABI) / gen-c-abi --check OK / check-platform-support OK (50 anchors) / check-fa-v3-confinement OK |

Note (CosyVoice2): the first gated run failed on a **stale cache artifact** — `~/.cache/vokra-eval/gguf/cosyvoice2-0.5b-llm.gguf` is the pre-fix conversion with 0-hparams metadata. Re-converting the same checkpoint with the upstream Qwen2 `config.json` (exactly what the campaign fix `7336079` requires) makes the full suite pass, including forward parity vs the with-bias transformers dump at atol 3e-4. Not a Wave-1/2 regression; the cache GGUF should be refreshed (owner note).

## Baseline JSONs (`vokra-cli bench --baseline` consumables)

Five M1-rig-scoped baseline files were generated in this directory with `vokra-cli bench --format json` (quiet window, loadavg 1.85 at start; raw command log = `raw/` + scratch `bench-json.log`), each with a `provenance` key prepended (the parser tolerates extra keys; consumption verified with an actual `--baseline` run):

| file | task / input | rtf (mean-based, as bench emits) | cross-check vs harness median |
|---|---|---|---|
| `whisper-base-asr.m1.baseline.json` | asr, jfk-30s.wav, 10 iters | 0.072007 | p50 788.8 ms vs harness 810.4 ms |
| `whisper-small-asr.m1.baseline.json` | asr, jfk-30s.wav, 10 iters | 0.284845 | p50 3106.9 ms vs harness 3125.7 ms |
| `silero-vad.m1.baseline.json` | vad, jfk-30s.wav, 10 iters | 0.003343 | p50 36.76 ms vs harness 37.75 ms |
| `piper-tts-default-text.m1.baseline.json` | tts, DEFAULT_BENCH_TEXT (EN, char-tokenizer route — NOT the JA phoneme leg in the table above; deterministic regression reference only) | 0.103656 | n/a (different input than the JA G2P leg) |
| `mel-frontend.m1.baseline.json` | mel-frontend standalone, 30 iters | 0.001527 | consumption-verified (ratio 0.98, green) |

Usage on this rig: `vokra-cli bench --model <same.gguf> --input tests/fixtures/audio/jfk-30s.wav --baseline docs/bench-baselines/m5-14-final-2026-07-18/<file>` — exits non-zero on a >5% RTF regression (NFR-PF-13).

**`docs/bench-baselines/mel_frontend_baseline.json` was deliberately NOT regenerated.** That file feeds the CI `bench-regression` job which is **locked to ubuntu-latest** (ci.yml comment: NEON hardware skew keeps the 5% gate off macOS). This M1 measures mel-frontend RTF ~0.0026 vs the committed ubuntu value 0.003115 — overwriting it with the faster M1 number would make the ubuntu runner appear >5% regressed and falsely redden main CI. M5-14 only made the mel path equal-or-faster, so the existing gate stays valid and green. The M1-scoped JSONs above are the post-M5-14 regression references for this rig.

## Caveats / footnotes

- **AMX caveat (Mimi/DFN3 references):** torch on macOS arm64 routes linears through Accelerate = the AMX coprocessor (measured 848–1400 GFLOP/s 1T on this rig, 10–30x any NEON kernel). A zero-dep NEON runtime cannot legally match that per-op; the honest kernel-quality reference is ORT-MLAS NEON (Wave-0 T03), which the Wave-1 packed GEMM now sits at 92–96% of at 1T and beats at 8T. Read the Mimi "vs 1T-torch" gaps with that hardware asymmetry in mind (DFN3's torch path is conv/GRU-loop-bound, so its reference is fairer).
- **x86 tier is unmeasured here:** AVX2/AVX-512 packed kernels are compile-verified + runtime-detect differential-tested only. Cloud-VM perf run = owner ticket M5-14-T29; real-device sweep (Android / iOS / Web / Pi) trigger = owner ticket M5-14-T30.
- **whisper-small ORT ambiguity:** campaign-1 measured ORT small full greedy at 2.52 s; the Wave-0 fresh two-phase re-measure got 1.842 s (both same rig; delta = ambient load + feature-extraction placement, documented in Wave-0 `target-table.md`). The <=1.5x verdict is met against the frozen campaign number the target was set with; against 1.842 s the ratio is 1.70x. A same-session ORT re-measure alongside the vokra run would settle it — folded into the owner T29 pass.
- **beam word_timestamps:** Wave-2 measured +3.9% overhead with `word_timestamps: true` (within the +2–6% band); this sweep ran beam with timestamps off (campaign parity).
- The Wave-0 raw baseline (logs, hotspot tables, ORT phase measurements, target-table rationale) lives at `~/.cache/vokra-eval/out/m5-14-wave0/` (local, untracked); its numbers are reproduced in the table above and in `docs/adr/M5-14-cpu-hotpath.md` (local ADR). This report + `raw/` is the tracked record.
