# M5-14 Wave-0 T02 — per-model per-stage hot-spot tables (M1 iMac, 8 CPU)

Rig: Apple M1 iMac 8-CPU 16 GB, worktree detached @ 9b718d1, release (opt3 + fat
LTO + 1 CGU), `active_isa = neon-dotprod` (f32 table = baseline NEON kernels).
Protocol: 1 warmup + 3 measured (prof disabled), median; then 1 profiled pass for
within-stage attribution (instrumentation overhead measured ≤ 1.5% of stage wall).
Stage rows marked [P] come from the profiled pass (throwaway `vokra_core::prof`
instrumentation compiled into the worktree; NOT in the shipped runtime).
Ambient load: the user's desktop (Chrome/WindowServer) was active during parts of
the session; per-run 1-min loadavg is noted (embedded in every raw log).
Raw logs: `~/.cache/vokra-eval/out/m5-14-wave0/*.log`.

## 1. whisper-small (jfk-30s.wav = 11.0 s speech, greedy, 26 tokens, 8T) [load ~1.7–2.0]

One-call window (WhisperAsr::transcribe_tokens): median 7.358 s, RTF 0.669
(campaign pass2: 7.319 s / 0.665 — reproduced) [that run at load 2–6].
Staged medians (sum 6.38 s at load ~1.9):

| stage | ms | % of staged sum |
|---|---|---|
| log-mel frontend | 46.2 | 0.7% |
| encoder | 3740.4 | 58.6% |
| decoder loop (incl. decoder init) | 2590.9 | 40.6% |
| detokenize | 0.006 | ~0% |

Encoder inside (profiled pass, enc 3.687 s) [P]:

| component | ms | % of encoder | note |
|---|---|---|---|
| k.gemm total | 3159.0 | 85.7% | 360 calls |
| — mlp fc1 (1500,3072,768) ×12 | 960.3 | | 88 GF/s (8T) |
| — mlp fc2 (1500,768,3072) ×12 | 848.2 | | 100 GF/s |
| — q/k/v/out proj (1500,768,768) ×48 | 708.6 | | 120 GF/s |
| — attn AV (1500,64,1500) ×144 | 344.2 | | per-head |
| — attn scores (1500,1500,64) ×144 | 297.7 | | per-head |
| k.softmax ×144 | 311.8 | 8.5% | |
| conv stem (2×conv1d+gelu+transpose) | 51.9 | 1.4% | |
| k.layer_norm ×25 | 21.3 | 0.6% | |
| stage split: enc.mlp 1877 / enc.attn 1748 | | | |

Decoder inside (profiled pass, dec 2.627 s; 26 steps × 12 layers) [P]:

| component | ms | % of decoder | note |
|---|---|---|---|
| cross-attn stage ×312 | 1185.0 | 45.1% | fixed t_kv=1500 |
| mlp stage ×312 | 671.2 | 25.5% | |
| self-attn stage ×312 | 335.7 | 12.8% | growing t_kv≤29 |
| cross-K/V precompute (1500,768,768) ×24 | 356.1 | 13.6% | decoder init |
| logits (ln + tied head) ×26 | 75.8 | 2.9% | gemv (51865,768) 2.87 ms/step |
| kv_append ×312 | 0.31 | ~0% | NOT a hot spot |
| m=1 projections (1,768,768) ×1800 | 433.8 | | 4.9 GF/s — latency-bound row tail |
| m=1 mlp fc1/fc2 (1,3072,768)/(1,768,3072) ×600 | 590.7 | | 4.8 GF/s |
| per-head cross (1,64,1500)+(1,1500,64) ×7200 | 182.8 | | 30 µs/call |

Per-token decode: 99.6 ms/token (vokra) vs ~29.8 ms/token (ORT derived, §T03).

## 2. whisper-medium (same input, 27 tokens, 8T) [load 2.2→8.5 across run — flagged]

Staged medians (sum 46.28 s, RTF 4.21; campaign 4.26 — reproduced):

| stage | ms | % |
|---|---|---|
| log-mel | 46.2 | 0.1% |
| encoder | 36262.7 | 78.4% |
| decoder loop | 9971.8 | 21.5% |
| detokenize | 0.008 | ~0% |

Encoder inside (profiled pass 41.9 s — heavier ambient load; use shares) [P]:

| component | s | % of encoder GEMM | GF/s (8T) |
|---|---|---|---|
| k.gemm total | 40.23 | 96% of encoder | 28 aggregate |
| — fc1 (1500,4096,1024) ×24 | 16.86 | 42% | **17.9** |
| — fc2 (1500,1024,4096) ×24 | 16.69 | 41% | **18.1** |
| — q/k/v/out (1500,1024,1024) ×96 | 4.29 | 11% | 71.9 |
| — attn (1500,64,1500)+(1500,1500,64) ×768 | 2.39 | 6% | |
| k.softmax ×384 | 1.08 | | |
| conv stem | 0.12 | | |

The two ffn shapes (n or k = 4096 = 16 KiB row stride) run at 18 GF/s and are
83% of encoder GEMM time — the medium-specific pathology (§microbench).

Decoder inside (profiled pass 13.21 s at high load; shares) [P]:
mlp stage 46% / cross-attn 21% / self-attn 12% / cross-K/V precompute
(1500,1024,1024)×48 = 2.49 s ≈ 19% / logits 1.5%. m=1 fc1+fc2
(1,4096,1024)+(1,1024,4096) = 5.14 s; m=1 proj (1,1024,1024)×3744 = 1.93 s.
Per-token: 369 ms/token (vokra) vs ~131 ms/token (ORT derived).

## 3. piper-plus css10-JA (campaign phoneme input, 2.055 s audio) [load ~2.0]

Median wall 338.2 ms, RTF 0.165 (campaign synthesis-only est 0.167 — reproduced).

| stage | ms | % | note |
|---|---|---|---|
| MB-iSTFT decoder | 225.7 | 66.7% | |
| text encoder | 59.3 | 17.5% | |
| flow (reverse) | 47.0 | 13.9% | |
| duration predictor | 5.6 | 1.7% | |
| length regulate + noise | 0.44 | 0.1% | |
| **k.gemm total (all stages)** | **62.1** | **18.4% of wall** | 2608 calls |

⇒ 81.6% of piper wall is OUTSIDE the dispatched kernels: model-side im2col
gather fills (`piper_plus/nn.rs::conv1d` fills a fresh `col` + scatters with
bias per conv), iSTFT/multiband synthesis, activation glue. Top GEMM:
(384,177,960)×16 = 30.8 ms. This is a glue-dominated model, not a kernel-
dominated one.

## 4. Mimi (jfk 24 kHz 10.96 s, mimi-neural.gguf, real weights)

Encode: median 5.488 s, **RTF 0.501** (campaign 11.7× vs 1T-torch 0.0431 — reproduced).

| stage | s | % | note [P] |
|---|---|---|---|
| bottleneck transformer | 3.128 | 57% | per-position loop (`process_inplace`: `for i in 0..t { step }`), issues `gemm_f32(1,d,·)` — m=1 GEMMs total 2.88 s at 4.6–4.8 GF/s |
| frame resample + RVQ quantize | 1.258 | 23% | GEMV only 4.2 ms → ~1.0 s is the private scalar `rvq_quantize_chain` argmin |
| SEANet conv stack | 1.159 | 21% | GEMM inside ~0.16 s → ~1.0 s im2col/ELU/copy glue |

Decode: median 4.167 s, **RTF 0.380** (campaign 9.5× — reproduced).
transformer 3.016 s (72%, same m=1 pattern) / SEANet decoder 1.024 s (25%) /
feature head 2.4 ms.

## 5. DFN3 (real dfn3.gguf, 11 s noisy 48 kHz)

Median wall 1.974 s, **RTF 0.180** (campaign 0.181 — reproduced). ZERO dispatched
kernel calls — 100% private scalar in `vokra-ops/src/denoise.rs`.

| stage | ms | % [P] |
|---|---|---|
| DF decoder (df_gru + skip + convp + per-frame df_out) | 854.1 | 43.3% |
| ERB decoder (dec_emb_gru + conv/convt chain) | 540.4 | 27.4% |
| encoder convs (erb_conv0-3 + df_conv0-1 + fc_emb) | 297.0 | 15.0% |
| encoder GRU + lsnr | 235.3 | 11.9% |
| synthesis iSTFT | 29.0 | 1.5% |
| analysis STFT | 14.8 | 0.7% |
| mask + deep-filter apply | 2.3 | 0.1% |
| features (erb/spec) | 0.7 | ~0% |

GRU-bearing stages (df_dec + erb_dec + enc_gru) = 82% ⇒ GRU/Linear per-frame
scalar loops are the target (T22/T23).

## 6. Silero VAD (jfk 11 s, 512-sample stream, ctx576 path) [load ~2.2]

Median wall 241.0 ms, RTF 0.0219, 343 frames (0.70 ms/frame). ZERO dispatched
kernel calls — 100% private scalar (`silero_vad/math.rs`; SIMD deliberately
deferred at M0-05 and never wired).

| stage | ms | % [P] |
|---|---|---|
| encoder convs (4× conv1d+relu) | 138.4 | 57.4% |
| pseudo-STFT (reflect pad + conv) | 82.1 | 34.1% |
| LSTM + head | 20.5 | 8.5% |

Per-call overhead: private conv allocates its output `Vec` (and pad buffer) per
call — but the compute itself is scalar, so the gap is kernel-quality-first,
allocation-second at this size. vs ORT (campaign timing.tsv, same rig): jfk
ctx576 88.6 ms default-threads → 2.7×; worst corpus file 4.6×.

## 7. CAM++ (jfk 11 s) [load ~2.1]

fbank 20.1 ms + embed 293.8 ms.

| component | ms | % of embed |
|---|---|---|
| k.gemm total | 118.5 | 40.3% |
| non-kernel (im2col gather + col alloc + scatter/BN/pool glue) | ~175 | ~60% |

Top GEMM shapes are thin-m/huge-n: (32,43920,288) 38.5 ms/4, (32,21960,288)
23.6 ms/4 — im2col col matrices up to ~50 MB, allocated per call.
vs ORT campplus.onnx (campaign): 66.6 ms (default threads) → 4.4×; 175.5 ms
(intra=1) → 1.7×.

## 8. Thread scaling (whisper-small encoder, VOKRA_CPU_THREADS=1/2/4/8) [load ~3–4]

| threads | median s | speedup |
|---|---|---|
| 1 | 8.949 | 1.00× |
| 2 | 5.187 | 1.73× |
| 4 | 3.916 | 2.29× |
| 8 | 4.229 | **2.12× (SLOWER than 4T)** |

Mechanism: `pool.rs` splits output rows EVENLY and statically; on M1 (4P+4E)
the E-cores become the critical path, and ambient load amplifies stragglers.
GEMM is the only parallel op; decoder-step GEMMs are below `GEMM_MIN_MACS`
(2^20) so decode is single-threaded (m=1·768·768 = 0.59 M MACs).

## 9. GEMM microbench (vokra kernels::gemm_f32 vs naive vs references)

vokra = dispatch path (NEON MR=8×NR=8, unpacked B); 1 warmup + 3 (9 for m=1),
median GFLOP/s. torch/numpy = **Accelerate = AMX coprocessor** (not a legal
zero-dep target; shown for scale). ORT-MLAS = MatMul-node session, B as
initializer (pre-packable), same rig.

| shape (m,n,k) | vokra 8T | vokra 1T | naive 1T | MLAS 1T | MLAS 8T(def) | torch-AMX 1T |
|---|---|---|---|---|---|---|
| 1500,3072,768 (small fc1) | 90.1 | 40.9 | 23.1 | 91.3 | 336.0 | 1180 |
| 1500,768,3072 (small fc2) | 100.9 | 40.5 | 21.7 | 90.7 | 328.4 | 1070 |
| 1500,768,768 (small qkv) | 122.9 | 44.8 | 23.7 | 96.5 | 355.0 | 1400 |
| **1500,4096,1024 (med fc1)** | **21.0** | **19.3** | **22.7** | **95.6** | **343.9** | 915 |
| 1500,1024,1024 (med qkv) | 92.1 | 46.2 | 24.1 | 96.3 | 353.5 | 1174 |
| 1500,1024,4096 (med fc2) | (in-model ~18) | — | — | 90.8 | 328.5 | 848 |
| 1,768,768 (dec proj) | 6.5 | — | — | 25.2 | 72.4 | 79.5 |
| 1,3072,768 (dec fc1) | 6.2 | — | — | 24.8 | 73.3 | 76.8 |
| 1,4096,1024 (med dec fc1) | 3.1 | — | — | 23.5 | 32.0 | 20.3 |
| gemv 51865×768 (logits) | 28.6 | 18.2 | — | — | — | — |
| gemv 1×768 | 18.3 | — | — | — | — | — |

Readings:
- MLAS 1T is shape-INVARIANT (~91–96): packing defeats the stride pathology.
- vokra 1T = 41–46 on friendly shapes (2.1–2.2× behind MLAS 1T), 19–21 on the
  16 KiB-stride medium ffn shapes (5× behind) where the NAIVE loop wins.
- m=1 through the gemm row-tail = 6.5 GF/s; the same op through vokra's own
  GEMV kernel = 18.3 GF/s (≈3× free); MLAS 1T = 25.
- logits GEMV at 8T moves 51865·768·4 B ≈ 159 MB in 2.79 ms ≈ 57 GB/s —
  already near M1 bandwidth; not a compute target.
