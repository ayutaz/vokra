# Web (WASM / WebGPU) RTF baselines — M4-01-T24 / Kill switch G input

WP M4-01 measurement hook: browser timing uses the JS `performance.now()`
import (`vokra_webgpu_now_ms` in the glue; `std::time::Instant` panics on
wasm32-unknown-unknown), Node timing uses `performance.now()` directly.
RTF = wall-clock seconds / audio seconds; **no threshold assert anywhere**
(NFR-PF-08 requires "動作" only — honest measurement + reporting, the
`cuda_rtf_analyze.py` no-ceiling posture).

## Kill switch G hand-over

CLAUDE.md Kill switch **G**: "ORT Web + WebGPU が音声モデルを実用速度
(RTF < 1.0 on Whisper base) で動かせる" (12-18 month window — overlapping
this phase, milestones §7.4). **The Vokra-side numbers in this file are the
comparison input for the quarterly Go/No-go review**; collecting the ORT Web
side and making the G call is the owner's quarterly review task. Reference
path: quarterly review → this file + the ORT Web release notes / demos.

## How to measure

```sh
scripts/build-wasm.sh all               # 2 artifacts + web/pkg
# Node (CPU path; simd128 + base, JFK fixture; prints RTF per artifact):
VOKRA_WHISPER_GGUF=models/whisper-base.gguf node tools/wasm/run-whisper-wasm.mjs
# Browser (CPU + WebGPU; RTF shown in the demo UI / parity harness):
node web/demo/serve.mjs                 # then open http://localhost:8788/web/demo/
#   parity harness: http://localhost:8788/tools/wasm/parity.html
```

## Measurements

| date | environment | artifact / backend | audio | RTF | notes |
|------|-------------|--------------------|-------|-----|-------|
| 2026-07-15 | Node 24.16.0, Apple M1 iMac (authoring host) | simd128 / cpu | jfk-30s.wav (11.0 s speech) | **0.876** | `tools/wasm/run-whisper-wasm.mjs`, whisper-base fp32 GGUF (290.9 MB, converted from openai/whisper-base `model.safetensors`), wall 9.64 s. Transcript = exact parity-whisper-real expectation (" And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."). FR-EX-08 negative leg also green: explicit `backend=webgpu` refused with the BackendUnavailable text on this adapterless host. |
| 2026-07-15 | Node 24.16.0, Apple M1 iMac (authoring host) | base (scalar) / cpu | jfk-30s.wav (11.0 s speech) | **1.888** | Same run, wall 20.77 s; identical transcript. **SIMD128 vs scalar = 2.15x** on the whisper-base forward. |
| (owner) | M1 iMac Chrome (real WebGPU) | simd128 / cpu | jfk-30s.wav | _fill from demo UI_ | M4-01-T24 |
| (owner) | M1 iMac Chrome (real WebGPU) | simd128 / webgpu | jfk-30s.wav | _fill from demo UI_ | per-op upload/dispatch/readback mode — expect SLOWER than cpu at whisper-base scale until device-resident chains land (the honest M2-01 per-op-stage precedent: Metal per-op was 913 ms vs CPU 128 ms before session residency); record what is measured |
| (owner, T28) | Chrome / Edge / Safari spot check | both backends | jfk-30s.wav | _record per browser_ | Safari WebGPU support: **record the observed state, do not invent it** |

Environment columns to note with every row: browser + version / Node
version, OS, GPU (about://gpu), COOP/COEP on/off, artifact
(`session.meta.artifact`), backend.

## Per-kernel parity numbers (record alongside RTF)

`tools/wasm/parity.html` prints per-kernel max |Δ| vs the CPU oracle
(atol = 0.01, NFR-QL-01). Paste the table here on the first real-GPU run;
if any kernel exceeds atol, follow the honest per-tensor atol procedure
(Kokoro `PROSODY_F0_ATOL` precedent: architectural-bound rationale in
rustdoc + ADR + CI, never a fabricated pass).
