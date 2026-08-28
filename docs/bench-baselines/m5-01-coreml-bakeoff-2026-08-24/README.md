# M5-01 CoreML/ANE bakeoff evidence — 2026-08-24 JST

This directory records the real Apple M1 run used by
`docs/handoff/m5-01-coreml-bakeoff-2026-08-24.md`. The tested unit is the
complete Whisper encoder (`log_mel` to `encoder_hidden`), which is exactly the
submodel delegated to CoreML. CPU and CoreML samples alternate inside one
release process while reusing the same Rust model, input features, and loaded
`MLModel` session.

The GGUF and compiled CoreML bundle are deliberately not committed here. The
hashes below bind the evidence to the local artifacts:

- source GGUF SHA-256:
  `7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27`
- FP16 compiled `.mlmodelc` tree SHA-256:
  `477dd5393d57eb38139b88e3e48f79b7e92d8498a9b8b73d4862175cc6f4f7a9`
- FP32 diagnostic compiled tree SHA-256:
  `9cde2ab4ad712f147deeccb0e0d61a3c255777cdfd09d97aa68a26c3b903aef9`
- audio fixture SHA-256:
  `58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f`
- source tree base commit:
  `5d93d3daa533427b375a18930b72f006e820fcef`

`bakeoff.txt` is the complete stdout of the formal FP16 release run. Exit code
1 is expected because the command is a gate and both parity and 2x speed fail.
`placement.txt` is the stable key/value subset of the official CoreML
`MLComputePlan` output; the omitted per-constant name repetition does not
affect its operation counts or estimated-cost placement fraction.
`fp32-diagnostic.txt` proves the numerical discrepancy is tied to the FP16
ANE-capable path: FP32 passes the CPU oracle but CoreML places its estimated
compute cost entirely on CPU.

Reproduction shape (paths are operator-local):

```bash
CARGO_BUILD_JOBS=1 RUSTC_WRAPPER= cargo run --release \
  -p vokra-cli --features coreml -- npu-bakeoff \
  --model "$GGUF" --input tests/fixtures/audio/jfk-30s.wav \
  --delegate coreml --warmup 2 --iters 10 \
  --atol 0.01 --min-speedup 2.0

tools/coreml/check_placement.sh \
  "$GGUF.coreml/whisper-encoder.mlmodelc" 0.90
```
