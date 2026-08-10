# Whisper large-v3 CUDA RTF baseline — H100 PCIe (2026-08-10)

M4-07 FlashAttention v3 Hopper hand-back — see
`docs/m4-07-hopper-bench-handover.md` §3 (owner run).

## Hardware / driver

- **GPU**: NVIDIA H100 PCIe (SM 9.0, 80 GB VRAM)
- **Driver**: 550.163.01
- **CUDA**: 12.4.1 (nvidia/cuda:12.4.1-devel-ubuntu22.04)
- **Host**: vast.ai offer #31427212 (container 6f11f5642828)
- **Session**: 2026-08-10, ~60 min, $1.73 total

## Workload

- **Model**: `vokra/whisper-large-v3` (2.9 GB GGUF, DL 2026-08-10)
- **Audio**: `tests/fixtures/audio/jfk-30s.wav` (11.0 s of speech)
- **Backend**: CUDA (dlopen libcuda.so.1 + libnvrtc.so.12, zero-dep)
- **Repeats**: N=10 per mode, one warmup, iters sequential (no concurrency)

## Results

### e2e RTF (`cuda_rtf_variance.sh`, one host, three modes)

| mode | median RTF | mean RTF | CV | speedup vs decomposed |
|------|-----------:|---------:|----:|----------------------:|
| **decomposed** | 0.965622 | 0.966009 | 0.10% | 1.0000× (baseline) |
| **v2** (gated) | 0.965551 | 0.965814 | 0.21% | 1.0001× (gate off @ t_q=1) |
| **v3** | **0.913262** | **0.913127** | **0.23%** | **1.0574× (5.7% e2e gain)** |

### FA v3 kernel parity (`parity_kernels_cuda.rs::flash_attn_v3_*`)

| face | worst \|Δ\| | atol pre-registered | status |
|------|-----------:|--------------------:|:-------|
| causal vs decomposed | 1.206e-2 | 0.02 | PASS (60% of atol) |
| non-causal vs decomposed | 1.026e-2 | 0.02 | PASS (51% of atol) |
| validation (cuda-less) | n/a | n/a | PASS |

Sweep: t_q ∈ {1, 17, 63, 64, 65, 96, 448, 1500}, t_kv = t_q, q_offset = 0.

### NVRTC feasibility (`fa_v3_nvrtc_feasibility.rs`, all 4 PASS)

- **compute_90a snippet**: 4225 B PTX, `wgmma.mma_async` present
- **compute_89 snippet**: 4224 B PTX (unexpectedly accepted — arch check
  deferred to module load; the SM≥9.0 lazy-compile gate is the
  load-time firewall)
- **compute_90a full program**: 69917 B PTX
- **NUL byte rejection**: PASS

## Interpretation

- **FA v3 is empirically positive on this workload.** The 5.7% median
  e2e RTF speedup vs FA v2 (decoder-dominant) is small in absolute
  terms because the FA v3 encoder pass runs once per 30 s audio while
  decoder self-attention (t_q=1) stays below `FA_V3_MIN_TQ = 64` by
  design. This is the first mode to move the e2e number measurably;
  earlier RTX 4090 sessions saw FA v2 honest-negative.
- **FA v2 continues to be honest-negative on decoder-step Whisper**
  (mean delta 0.00019 vs decomposed, well inside the 0.0021 CV band).
  This is the same result as the 2026-07-10 RTX 4090 run
  (`docs/bench-baselines/vast-2026-07-10/rtf-fa-v2.jsonl`) — the
  Hopper hardware does not change the algorithmic gate.
- **Parity holds comfortably inside atol**: both causal and non-causal
  are ≤ 60% of the pre-registered 0.02 bound. A follow-up session on
  a different driver / CUDA version could tighten the bound to 0.015.

## Files

- `rtf-h100-decomposed.jsonl` / `rtf-h100-decomposed.report.md`
- `rtf-h100-fa-v2.jsonl` / `rtf-h100-fa-v2.report.md`
- `rtf-h100-fa-v3.jsonl` / `rtf-h100-fa-v3.report.md`

## Cross-references

- `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` — machine-readable
  baseline (all numbers above + parity findings + NVRTC findings)
- `docs/perf/cuda-large-v3-baseline.json` — RTX 4090 formal gate
  baseline (M2-14 / M3-01, MUST NOT be merged with H100 numbers)
- `docs/adr/M4-07-fa-v3-hopper.md` — ADR (owner: add T17 findings)
- `docs/m4-07-hopper-bench-handover.md` — the recipe this run followed
- `docs/bench-baselines/vast-2026-07-10/` — RTX 4090 predecessor
