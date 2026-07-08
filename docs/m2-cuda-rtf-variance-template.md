# M2-03 CUDA large-v3 RTF variance report — template

_This is a **template**. The owner fills in the values after a two-mode
`tools/parity/cuda_rtf_variance.sh` run on a vast.ai RTX 4090 (or any
CUDA host). The filled report is committed alongside the raw JSONL under
`docs/bench-baselines/` for provenance._

**Position**: reference measurement, **not** the formal `RTF < 0.10`
always-on gate. That always-on decision belongs to **M2-14** (owner
self-hosted CUDA runner) + **M3-01** (5% regression gate) per
[`docs/adr/M2-03-followup-rtf.md`](adr/M2-03-followup-rtf.md) §D6.

---

## Meta

| field | value |
|-------|-------|
| Report date (UTC)     | `YYYY-MM-DDTHH:MM:SSZ` |
| Owner                 | `ayutaz` |
| Vokra commit          | `<git rev-parse HEAD>` |
| Toolchain             | `rustc 1.86.0 (05f9846f8 2025-03-31)` |
| Harness version       | `tools/parity/cuda_rtf_variance.sh` @ `<commit>` |
| Related ADR           | [`docs/adr/M2-03-followup-rtf.md`](adr/M2-03-followup-rtf.md) |
| Related checklist row | [`docs/m2-owner-verification-checklist.md`](m2-owner-verification-checklist.md) §2 |
| Baseline JSON updated | [`docs/bench-baselines/whisper_large_v3_cuda_rtf.json`](bench-baselines/whisper_large_v3_cuda_rtf.json) — yes / no |

## Hardware / environment

| field | value |
|-------|-------|
| GPU                     | `NVIDIA GeForce RTX 4090` |
| GPU memory (MiB)        | `24564` (or whatever `nvidia-smi --query-gpu=memory.total` reports) |
| Driver version          | `<nvidia-smi --query-gpu=driver_version --format=csv,noheader>` |
| CUDA toolkit version    | `<nvcc --version | head -4>` |
| Host                    | `vast.ai spot / self-hosted / other` |
| Instance offer / id     | `vast.ai offer <id>, instance <id>` |
| Cost per hour (USD)     | `0.4` (typical spot) |
| Ambient / cooling notes | free-form — was the case airflow OK? Was the SM temp stable? |

## GGUF / audio inputs

| field | value |
|-------|-------|
| GGUF path (on host)      | `/root/whisper-large-v3.gguf` |
| GGUF SHA-256             | `2ebfc46a95ad3831377ae5f4d9d30e35dd2d87fb0526769a02f78b237d30e761` (baseline) — must match, otherwise the RTF is not comparable |
| GGUF size (bytes)        | `3087545600` |
| Audio path               | `/root/jfk-30s.wav` |
| Audio SHA-256            | `<sha256sum>` |
| Audio duration (s)       | `30.0` (must be 30 s for baseline-comparable RTF) |
| Audio format             | `PCM_16BIT / IEEE_FLOAT32`, 16 kHz mono |
| Converter command        | `vokra-cli convert --model whisper --input model.safetensors --output whisper-large-v3.gguf` |

## Harness invocation

```bash
# Path A: decomposed 2 + 7·n_head chain (VOKRA_CUDA_DISABLE_FA_V2=1)
./tools/parity/cuda_rtf_variance.sh \
    --gguf   /root/whisper-large-v3.gguf \
    --audio  /root/jfk-30s.wav \
    --iters  10 \
    --warmup 1 \
    --fa-v2  off \
    --label  decomposed \
    --output /root/rtf-decomposed.jsonl

# Path B: default FA v2 gated wrapper (t_q >= 16)
./tools/parity/cuda_rtf_variance.sh \
    --gguf   /root/whisper-large-v3.gguf \
    --audio  /root/jfk-30s.wav \
    --iters  10 \
    --warmup 1 \
    --fa-v2  on \
    --label  gated_fa_v2 \
    --output /root/rtf-fa-v2.jsonl

./tools/parity/cuda_rtf_analyze.py /root/rtf-decomposed.jsonl \
    --output /root/rtf-decomposed.report.md
./tools/parity/cuda_rtf_analyze.py /root/rtf-fa-v2.jsonl \
    --output /root/rtf-fa-v2.report.md
```

## Path A — decomposed (VOKRA_CUDA_DISABLE_FA_V2=1)

_Paste the `RTF statistics` and `Coefficient-of-variation warning` sections
from `rtf-decomposed.report.md` here._

### RTF statistics

| metric | value |
|--------|-------|
| n (successful samples) | `10` |
| mean                   | `` |
| median                 | `` |
| stddev (population)    | `` |
| CV (stddev / mean)     | `` |
| p50                    | `` |
| p95                    | `` |
| p99                    | `` |
| min                    | `` |
| max                    | `` |
| iters failed           | `0` |

### CV verdict

`OK / WARNING` — `<paste the report's CV block verbatim>`

### Comparison to baseline

Baseline (from `docs/bench-baselines/whisper_large_v3_cuda_rtf.json`
`measurement.statistics`): median RTF `0.1133`, stddev estimate `0.00016`.

| comparison | value |
|------------|-------|
| Absolute delta (this median − baseline median) | `` |
| Relative delta (% of baseline median)          | `` |
| Within 5% baseline-noise band?                 | `yes / no` |

## Path B — gated FA v2 (default)

_Paste the `RTF statistics` and `Coefficient-of-variation warning` sections
from `rtf-fa-v2.report.md` here._

### RTF statistics

| metric | value |
|--------|-------|
| n (successful samples) | `10` |
| mean                   | `` |
| median                 | `` |
| stddev (population)    | `` |
| CV (stddev / mean)     | `` |
| p50                    | `` |
| p95                    | `` |
| p99                    | `` |
| min                    | `` |
| max                    | `` |
| iters failed           | `0` |

### CV verdict

`OK / WARNING` — `<paste the report's CV block verbatim>`

### Comparison to baseline

Baseline (from `docs/bench-baselines/whisper_large_v3_cuda_rtf.json`
`gated_fa_v2.measurement.statistics`): median RTF `0.1323`, stddev
estimate `0.00086`.

| comparison | value |
|------------|-------|
| Absolute delta (this median − baseline median) | `` |
| Relative delta (% of baseline median)          | `` |
| Within 5% baseline-noise band?                 | `yes / no` |

## Owner judgment (formal `<0.10` gate)

The formal `RTF < 0.10` always-on gate is **NOT** decided by the harness.
This section is the owner's write-up of what the variance report implies.

**Question**: does the observed variance make the M2-03 single-shot range
`0.081 – 0.115` explainable as hardware variability (thermal / boost /
PCIe / vast.ai spot roommate contention), or is it a single-shot outlier?

Fill one of:

- [ ] **Promote formal `<0.10` gate now** — CV is low (`< 0.01`), mean is
      well below 0.10 in both paths, p99 is well below 0.10 too. The
      single-shot 0.081 – 0.115 range appears to be an outlier / cold-run
      artifact. Land the formal gate as an always-on test at M2-14
      (self-hosted runner) and the 5% regression at M3-01.
- [ ] **Defer formal gate to M2-14** — variance is moderate to high
      (CV `> 0.01`) OR mean / p99 straddles 0.10. The single-shot range
      reflects hardware variability, not a static perf number. Keep the
      current `<0.15` sanity gate; land the formal `<0.10` gate only
      after M2-14 self-hosted runner is stood up (dedicated GPU, no
      spot-tenant contention) and M3-01 can enforce the 5% regression
      band.
- [ ] **Investigate further** — variance is high enough to be
      inconclusive. Re-run with `--iters 30 --warmup 3` on a dedicated
      instance before deciding.

**Rationale** (owner free-form):

```
<why the checkbox above was picked — link to any nvtx traces, thermal
logs, or vast.ai instance state that informed the decision>
```

## Attachments (commit alongside)

- [ ] `rtf-decomposed.jsonl` — raw per-iter JSONL, decomposed path
- [ ] `rtf-fa-v2.jsonl` — raw per-iter JSONL, gated FA v2 path
- [ ] `rtf-decomposed.report.md` — analyzer output, decomposed
- [ ] `rtf-fa-v2.report.md` — analyzer output, FA v2
- [ ] Screenshot of `nvidia-smi -q -d TEMPERATURE,CLOCK,POWER` mid-run (optional but useful for CV WARN cases)
- [ ] `docs/bench-baselines/whisper_large_v3_cuda_rtf.json` — updated with new `variance_analysis` node (schema: two `statistics` + `per_iter_rtf` subnodes, one per FA v2 mode)

## Red-lines re-confirmed

- [ ] Zero-dep (NFR-DS-02) — no `pip install`, no `cudarc`, no cuBLAS/cuDNN linkage, no candle-kernels linkage
- [ ] NVIDIA EULA — `libcuda.so.1` + `libnvrtc.so.12` via `dlopen`; nothing shipped by Vokra
- [ ] FA v3 (Hopper WGMMA/TMA) — NOT pulled forward from v1.5+ (`CLAUDE.md` L77 red-line)
- [ ] FR-EX-08 — no silent CPU fallback; every failed iteration is emitted as an `error` JSONL line
- [ ] `git status --short` clean before the run (so the `Vokra commit` field above is meaningful)

## References

- ADR: [`docs/adr/M2-03-followup-rtf.md`](adr/M2-03-followup-rtf.md) (§D6, §D7, §D8)
- Checklist row: [`docs/m2-owner-verification-checklist.md`](m2-owner-verification-checklist.md) §2
- Existing sanity test: [`crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs`](../crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs)
- Baseline JSON: [`docs/bench-baselines/whisper_large_v3_cuda_rtf.json`](bench-baselines/whisper_large_v3_cuda_rtf.json)
- Harness README: [`tools/parity/README-cuda-rtf-variance.md`](../tools/parity/README-cuda-rtf-variance.md)
