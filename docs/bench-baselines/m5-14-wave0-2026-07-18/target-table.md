# M5-14 Wave-0 T03 — ORT contrast + honest optimization target table

Same rig (M1 iMac 8 CPU), same jfk-30s.wav input, onnxruntime 1.19.2
CPUExecutionProvider. ORT method: (a) encoder-only = raw InferenceSession.run on
the cached campaign `encoder_model.onnx`, features precomputed, 1+3 median,
intra_op ∈ {0 = ORT default, 1}; (b) full decode = optimum
ORTModelForSpeechSeq2Seq.generate (greedy, campaign prompt, max_new_tokens=444),
features outside the window (campaign pass2 windows INCLUDED feature extraction
— small deltas vs campaign are explained by that plus ambient load).
ORT profiling API was not needed: the two-phase split + the vokra-side stage
tables identify the gap sources unambiguously.

## Fresh two-phase ORT measurements (this session)

| leg | ORT median | vokra median (Wave 0) | gap |
|---|---|---|---|
| whisper-small encoder-only (intra default) | 1.217 s | 3.740 s | 3.07× |
| whisper-small encoder-only (intra=1) | 4.200 s | 8.949 s (1T) | 2.13× |
| whisper-small full greedy | 1.842 s (21 tok) | 7.358 s (26 tok) | 4.0× wall / 3.3× per-token |
| whisper-medium encoder-only (intra default) | 4.258 s | 36.263 s | **8.52×** |
| whisper-medium encoder-only (intra=1) | 13.651 s | (1T not run — microbench: ffn shapes 5× behind MLAS 1T) | ~2.7× (derived) |
| whisper-medium full greedy | 7.146 s (22 tok) | 46.28 s (27 tok) | 6.5× wall / 2.8× per-token decode (131 vs 369 ms/tok) |

Campaign references retained where re-measurement was redundant (same rig,
2026-07-16): piper ORT synthesis-only RTF 0.0324 (ja); Silero ORT jfk ctx576
88.6 ms (default threads); CAM++ ORT 66.6 ms (default) / 175.5 ms (intra=1);
whisper-turbo ORT jfk RTF 2.4723 (27.2 s — the fp32 turbo export is itself
slow, which is why the campaign ratio is only 1.81×); Mimi upstream 1T-torch
encode RTF 0.0431 / decode 0.0397 (macOS torch = Accelerate/AMX — see caveat);
DFN3 upstream torch RTF 0.0233 (same caveat, milder: conv/GRU loops).

## FINAL: per-model current / ORT / gap / hot spot / mechanism / target

| model (leg) | vokra now | ORT same rig | gap | dominant hot spot (T02) | mechanism | proposed target (honest) | rationale |
|---|---|---|---|---|---|---|---|
| whisper small (jfk full) | 7.36 s (RTF 0.669) | 1.84–2.52 s | 2.9–4.0× | encoder GEMM 86% of enc; enc = 59% of wall; dec m=1 GEMMs 4.9 GF/s | 1T kernel 2.1× behind MLAS (no packing) × pool scaling 2.1× vs MLAS 3.7×; m=1 row-tail | **≤1.5× — SUPPORTED** | MLAS proves 91–96 GF/s 1T / 330–355 GF/s 8T on this silicon for exactly these shapes; encoder is ≥86% GEMM so kernel parity ≈ encoder parity; decoder m=1→GEMV alone is ~3× on 45% of decode |
| whisper medium (jfk full) | 46.3 s (RTF 4.21) | 7.15–7.20 s | 6.4–7.8× | fc1/fc2 (n or k = 4096) = 83% of encoder GEMM at **18 GF/s**; enc = 78% of wall | 16 KiB power-of-2 B-row stride = L1 set-aliasing in the unpacked kernel (naive loop beats it); packing removes it entirely (MLAS: 95.6 GF/s same shape) | **≤1.5× — SUPPORTED** | the pathology is 100% a packing artifact; after packing, medium reduces to the small case (same 2.1× 1T + scaling story) |
| whisper turbo (jfk full) | 39.5 s staged (this run) / 49.1 s (campaign) | 27.2 s (campaign) | 1.45–1.81× | encoder 94% of wall (32L d=1280); GEMM at ~65 GF/s aggregate | same unpacked-kernel gap; no 16 KiB aliasing (5 KiB/20 KiB strides) — hence milder | **≤1.3× — SUPPORTED** (likely <1.0×) | ORT's fp32 turbo export is itself slow (RTF 2.47); vokra needs only ~1.4× encoder speedup to reach parity, packing alone yields ~2× on these shapes |
| whisper beam=5 (small) | 35.7 s vs greedy 7.36 s (4.9×; campaign +9.7–93% was smaller widths) | — | — | full-prefix re-forward per beam per step (m=5…29 GEMMs ×360 observed) | O(L²) recompute; logits computed for ALL prefix rows | T13: per-beam incremental KV; **beam5 ≤ 1.6× greedy** | after incremental KV, beam cost ≈ beam_width × per-token m=1 cost, amortized by shared encoder; beam=1 ≡ greedy bit-identity is the gate |
| piper css10-JA (2.06 s utt) | 338 ms (RTF 0.165) | 66 ms (RTF 0.0324) | 5.1× | MB-iSTFT decoder 67% of wall; **82% of wall outside kernels** (im2col fills, scatter+bias, iSTFT) | glue-dominated: per-conv col alloc+gather is scalar single-thread; GEMMs themselves small | **≤2× — SUPPORTED with scope** (needs BOTH conv-glue arena/vectorized gather AND packed GEMM; glue is the bigger half) | halving glue (vectorized im2col + arena) + 2× GEMM ⇒ ~2.2×; residual iSTFT scalar may hold it above 2× on short utterances — if Wave-2 measurement shows a floor, re-propose 2.5× with data |
| Silero VAD (jfk stream) | 241 ms (RTF 0.0219) | 88.6 ms | 2.7× (jfk; up to 4.6× corpus) | encoder convs 57% + pseudo-STFT 34%, all private scalar, 0 kernel calls | 未接続 (deliberate M0-05 1:1 scalar; SIMD deferred and never wired) | **≤1.5× — SUPPORTED** | plain NEON vectorization of conv/LSTM in-place (within-op, 1:1 semantics preserved) typically 3–4× on these sizes; absolute budget tiny (0.7 ms/frame) |
| CAM++ (jfk embed) | 294 ms embed (+20 ms fbank) | 66.6 ms (default) / 175.5 ms (intra=1) | 4.4× / 1.7× | 60% of embed outside kernels (50 MB im2col cols, per-call alloc); thin-m huge-n GEMMs | conv glue + no packing | **≤2× — SUPPORTED** | vs intra=1 ORT the gap is already 1.7×; arena + packed GEMM + fbank vectorization close the default-threads gap to ~2× |
| Mimi encode / decode (10.96 s) | 5.49 s (RTF 0.501) / 4.17 s (RTF 0.380) | 1T-torch 0.472 s / 0.435 s (RTF 0.043 / 0.040) — **AMX-backed** | 11.6× / 9.6× | per-position transformer loop: m=1 GEMMs 2.88 s (52% of encode) at 4.6 GF/s; RVQ argmin scalar ~1.0 s; SEANet glue ~1.0 s | model issues `gemm_f32(1,d,·)` per frame by streaming design; RVQ + glue private scalar | **≤3× vs 1T-torch (RTF ≤ ~0.13 enc / ≤0.12 dec) — SUPPORTED; the spec-目安 ≤2× is NOT supported without batching** | m=1→GEMV (3×) on 52% + RVQ vectorization (dot-product argmin is NEON-friendly, ~3×) + conv glue ⇒ ~2.8–3.3× whole-run ⇒ ~3.5–4× vs torch. Torch's AMX linears (~10× our NEON 1T) cannot be matched per-op under zero-dep; reaching ≤2× additionally requires batching the whole-buffer transformer path (projections are position-independent — allowed numerically, needs the streaming-state seam rework; propose as Wave-2 stretch, re-measure then) |
| DFN3 (11 s, 48 kHz) | 1.974 s (RTF 0.180) | torch 0.0233 RTF (1.28 s wall equiv? — campaign torch wall basis) | 7.7× | GRU-bearing stages 82% (df_dec 43% + erb_dec 27% + enc_gru 12%), 0 kernel calls | 100% private scalar conv2d/GRU (fresh code — no legacy fixture constraint) | **RTF ≤0.06 (≈3× improvement; ≈2.5× vs torch) — SUPPORTED; ≤2× (RTF 0.047) is stretch** | GRU gates = 4 GEMV per frame — route onto the GEMV kernel (18 GF/s vs scalar ~2–3) ⇒ 3–5× on 82% of wall; conv2d im2col onto packed GEMM; torch's conv/GRU CPU path is not AMX-dominated so this reference is fairer than Mimi's |
| Kokoro | (not re-measured — Wave-0 required set excludes it; campaign: 4.27×, sentence RTF ~2.3) | — | — | BiLstm1d scalar pin (parity-mandated) + NSF/iSTFT | pin respected per M2 T17-fixup #2 | **no target this WP** (T24 = low-risk glue only, parity 8/8 + WER 0.0 re-verified) | ADR D5 red-line |

## Where the spec 目安 is / is not supported by this data

- small/medium ≤1.5×: SUPPORTED (see rows; the enabling fact is MLAS-parity
  GEMM is demonstrably reachable on this hardware, and both models are
  ≥78%-encoder, ≥86%-GEMM).
- turbo ≤1.3×: SUPPORTED (ORT's own turbo leg is slow; encoder-only work).
- 短発話系 ≤2×: SUPPORTED for Silero / CAM++; piper is borderline (82%
  non-kernel — propose keeping ≤2× as the target but pre-authorize a
  data-backed revision to ≤2.5× if the Wave-2 glue rework measures a floor).
- Mimi (spec lists 11.7×/9.5× vs 1T-torch): a blanket ≤2× vs AMX-backed torch
  is NOT supported without the batching rework; propose ≤3× now, revisit
  after Wave-2 batching spike.
- DFN3: no spec 目安 given; propose RTF ≤0.06 (torch ×2.5) from the GRU→GEMV
  arithmetic above.

## Caveats
- Ambient desktop load (Chrome/WindowServer) was present for parts of the
  session; loadavg is recorded in every log. Single-thread numbers, kernel
  shares, and within-run ratios are robust; 8T absolute walls carry ±10–15%
  noise (e.g. whisper-small staged sum 6.38 s vs one-call 7.36 s across load
  conditions). T27 re-baselines everything in a quiet window.
- torch/numpy on macOS arm64 = Accelerate = AMX coprocessor (measured 848–1400
  GFLOP/s 1T): NOT a zero-dep-legal target; ORT-MLAS NEON is the honest one.
- ORT decode-only figures are full−encoder derivations (method documented
  above), not isolated decoder sessions.
- Turbo was staged once (enc 37.0 s at load ~2–8); campaign total was 49.1 s —
  day-to-day variance flagged, ratio bounds given as a range.
