# `docs/bench-baselines/` — index & provenance (X-06-T17)

This directory mixes three *kinds* of file that must never be confused. The
2026-07-18 M5-14 report recorded the cost of confusing them: overwriting a
**CI-gate** baseline with an **M1-rig** number falsely reddens main CI,
because the CI gate is locked to ubuntu-latest and an M1/NEON rig measures a
different RTF (mel-frontend ~0.0026 on M1 vs the committed ubuntu 0.003115 —
overwriting the gate file would make the ubuntu runner look >5% regressed).

**Rule: a baseline JSON that a CI gate reads is measured on ubuntu-latest and
only there. Everything else here is a reference, not a gate.**

## The three kinds

| kind | who reads it | where it must be measured | may it fail CI? |
|------|--------------|---------------------------|-----------------|
| **CI gate** | a `.github/workflows/*.yml` step via `vokra-cli bench --baseline` | **ubuntu-latest only** (hardware skew breaks the 5% gate off it) | yes (required or advisory) |
| **rig reference** | humans, and `vokra-cli bench --baseline` *on that same rig* | the rig named in its `provenance` (e.g. M1 iMac) | no — never wired to CI |
| **reference dump / report** | humans; parity/eval evidence | wherever it was captured (recorded in the file) | no |

## CI-gate baselines (ubuntu-latest)

| file | gate | posture |
|------|------|---------|
| `mel_frontend_baseline.json` | `ci.yml` bench-regression, `--task mel-frontend` (M2-04) | **required** (PR-blocking) |
| `silero_vad_baseline.json` | `ci.yml` bench-regression, Silero VAD (X-06-T09) | advisory; **placeholder** until seeded on ubuntu-latest via `bench-baseline-capture.yml` |
| `whisper_base_asr_nightly_baseline.json` | `nightly-asr-wer.yml` RTF companion (X-06-T11) | advisory record-only; **placeholder** |
| `piper_tts_nightly_baseline.json` | `nightly-asr-wer.yml` RTF companion (X-06-T12) | advisory record-only; **placeholder** (no piper GGUF committed — open question #7) |

Placeholders carry `"$placeholder": true` and `"rtf": null`. The gate step
classifies them with `tools/bench/baseline_gate.py` and **clean-skips** (no
verdict claimed — FR-EX-08). The owner seeds the real number from an
ubuntu-latest run and commits it (X-06-T20). Do **not** seed from an
M1/aarch64 rig.

GPU-side CI-gate baselines live under `docs/perf/` (separate dir):
`cuda-large-v3-baseline.json` (RTX 4090, `gpu-cuda-rtf.yml`) and
`cuda-large-v3-h100-fa-v3-baseline.json` (H100 reference, M4-07). Both are
`workflow_dispatch` + weekly cron, not required checks.

## Rig references (NOT CI gates)

| path | rig | note |
|------|-----|------|
| `m5-14-final-2026-07-18/mel-frontend.m1.baseline.json` | M1 iMac | mirror of the mel gate for M1 regression testing; **NOT** the CI file |
| `m5-14-final-2026-07-18/silero-vad.m1.baseline.json` | M1 iMac | |
| `m5-14-final-2026-07-18/whisper-base-asr.m1.baseline.json` | M1 iMac | |
| `m5-14-final-2026-07-18/whisper-small-asr.m1.baseline.json` | M1 iMac | |
| `m5-14-final-2026-07-18/piper-tts-default-text.m1.baseline.json` | M1 iMac | EN char-tokenizer route, deterministic reference |
| `m5-14-final-2026-07-18/report.md` + `raw/` | M1 iMac | M5-14 outcome (RTF-vs-ORT table, the AMX caveat) |
| `m5-14-wave0-2026-07-18/` | M1 iMac | M5-14 Wave-0 profiling (hotspot + target tables) |

Every `*.m1.baseline.json` carries an `M1-RIG-SCOPED` marker in its
`provenance` key. The benchmark dashboard (`tools/bench/build_dashboard.py`)
labels these "M1-rig reference (NOT a CI gate)".

## Reference dumps & reports

| path | what |
|------|------|
| `m1-real-weight-eval-2026-07-16/` | real upstream checkpoint → GGUF → e2e eval vs onnxruntime (report.md + report-campaign2.md + agent-results*.json) |
| `rtf-decomposed-2026-07-08.jsonl` / `.report.md` | early CUDA RTF probe (decomposed attention) |
| `rtf-fa-v2-2026-07-08.jsonl` / `.report.md` | early CUDA RTF probe (FA v2) |
| `vast-2026-07-10/` | vast.ai RTX 4090 N=10 references (rtf-decomposed / rtf-fa-v2 jsonl + reports, in-process latency, server-help) |
| `whisper_large_v3_cuda_rtf.json` | CUDA large-v3 RTF sanity capture |
| `metal-transcript-parity-2026-07-19/` | Metal-vs-CPU transcript parity |
| `silero-8k-ctx288-2026-07-19/` | Silero 8 kHz ctx288 measurement |
| `sbom-reproducibility-2026-07-19/` | SBOM reproducibility (spdx) |
| `web-2026-07-15/` | web/WASM notes |
| `eval-cache-artifacts-2026-07-19/` | eval-cache artifact notes |
| `m4-05-csm-fixture-reference.md` | CSM fixture reference |
| `m4-19-wyoming-real-gguf-2026-07-19/` | Wyoming real-GGUF report |

The dashboard aggregates the machine-readable subset of these (the `rtf-*.jsonl`
runs, `docs/perf/*.json`, and the committed baselines) into one page; the
markdown reports remain the narrative source.
