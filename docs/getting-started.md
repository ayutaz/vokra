# Getting Started

**English** | [日本語](getting-started.ja.md)

A 5-minute quick start for **Vokra**. This guide gets VAD → ASR → TTS running
on CPU. For GPU backends (Metal / CUDA) and distribution artefacts
(iOS / Unity / Python), see the "Next steps" section at the end.

## Prerequisites

- **Rust toolchain**: 1.89 or newer (`rustup default stable`). 1.89 is the
  *effective* MSRV: the workspace declares `rust-version = "1.85"` (the floor
  for edition 2024), but `vokra-backend-cpu` raises its own floor to 1.89 for
  the AVX-512 intrinsics stabilized there, and that crate is in every build.
  CI verifies this in the `msrv` job.
- **git**: for cloning the repository
- **curl or wget**: for downloading the checkpoint files used by the sample
  conversion recipes
- **uv + Python 3.12**: only needed to prepare checkpoints consumed by the
  converter. Run every Python command through uv; the Vokra runtime itself
  has no Python dependency (`FR-LD-05`).
- **Disk**: 2–4 GB for the sample models (Whisper base + a piper-plus voice)

The Vokra runtime is **zero external dependency** — the root `Cargo.lock`
only contains `vokra-*` crates — so no system packages beyond the Rust
toolchain are required.

## 1 min: Build

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
cargo build --release -p vokra-cli
```

This produces the CLI at `target/release/vokra-cli`. Build the C ABI
separately with `cargo build --release -p vokra-capi` when needed. Reference
CLI build time: ~2 minutes on a MacBook Air M2 (cold).

## 2 min: Convert models to GGUF

The Vokra runtime loads **GGUF only** — ONNX graphs are never loaded at
runtime, only through the offline conversion tool. Three common recipes:

### Silero VAD v5

```sh
# Upstream ONNX → GGUF
wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
./target/release/vokra-cli convert \
  --model silero-vad \
  --input silero_vad.onnx \
  --output silero_vad.gguf
```

### Whisper base (ASR)

```sh
# Hugging Face safetensors → GGUF (size auto-detected from checkpoint shape)
uv run --no-project --python 3.12 \
  --with transformers --with safetensors python - <<'PY'
from transformers import WhisperForConditionalGeneration
m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')
m.save_pretrained('whisper-base', safe_serialization=True)
PY
./target/release/vokra-cli convert \
  --model whisper \
  --input whisper-base/model.safetensors \
  --output whisper-base.gguf
```

Run conversion, validation, and publication on VAST when the aggregate model
artefacts are 2 GB or larger. Count all checkpoint shards and do not stage a
large converted artefact back onto a memory-constrained development Mac.

To **K-quantize** for smaller footprint:

```sh
./target/release/vokra-cli convert \
  --model whisper \
  --input whisper-base/model.safetensors \
  --output whisper-base.q4_k.gguf \
  --quantize q4_k
```

### piper-plus (TTS)

```sh
# piper-plus voice ONNX + config.json → GGUF
# Example voice: en_US-lessac-medium
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
  -O en_US-lessac-medium.config.json
./target/release/vokra-cli convert \
  --model piper-plus \
  --input en_US-lessac-medium.onnx \
  --config en_US-lessac-medium.config.json \
  --output en_US-lessac-medium.gguf
```

## 3 min: Run

`vokra-cli run` auto-selects the task from the GGUF's `vokra.model.arch`
metadata (Whisper→ASR / Silero VAD→VAD / piper-plus→TTS).

### VAD

```sh
./target/release/vokra-cli run \
  --model silero_vad.gguf \
  --input speech.wav
# Output: per-frame speech probabilities
```

### ASR

```sh
./target/release/vokra-cli run \
  --model whisper-base.gguf \
  --input speech.wav
# Output: transcribed text
```

### TTS

```sh
./target/release/vokra-cli run \
  --model en_US-lessac-medium.gguf \
  --text "Hello from Vokra." \
  --output hello.wav
# Output: hello.wav (22050 Hz mono PCM)
```

## 4 min: Benchmarks

```sh
# CPU RTF
./target/release/vokra-cli bench --model whisper-base.gguf --input speech.wav

# GPU (Metal on macOS / CUDA on Linux with hardware)
cargo build --release -p vokra-cli --features metal   # macOS
cargo build --release -p vokra-cli --features cuda    # Linux with system CUDA
./target/release/vokra-cli bench --model whisper-large-v3.gguf \
  --input speech30s.wav --backend cuda
```

RTF < 1.0 means real-time. The values below are acceptance targets, not a
guaranteed result on every machine. The RTX 4090 range 0.081–0.115 is a dated,
single-shot VAST reference from 2026-07-07, retained in
[`docs/perf/cuda-large-v3-baseline.json`](perf/cuda-large-v3-baseline.json) and
the [variance report](m2-cuda-rtf-variance-2026-07-08.md); it is not a current
CPU or Metal benchmark. Re-run `bench` on the target hardware before making a
performance claim.

## 5 min: Call from C ABI

Include `include/vokra.h` and link `libvokra` (see the [C ABI
example](../README.md#library-integration) in the top-level README):

```sh
cargo build --release -p vokra-capi
```

```c
#include "vokra.h"

vokra_session_t *s = NULL;
vokra_session_create_from_file("whisper-base.gguf", &s);

char *text = NULL;
vokra_asr_transcribe(s, pcm, num_samples, 16000, &text);
printf("%s\n", text);
vokra_string_free(text);
vokra_session_destroy(s);
```

## Next steps

- **Per-platform tutorials**: [`docs/tutorials/`](tutorials/)
  - [Unity + IL2CPP](tutorials/unity.md)
  - [iOS Swift Package](tutorials/ios.md)
  - [Python bindings](tutorials/python.md)
  - [Web (WASM / WebGPU)](tutorials/web.md)
  - [Android (native JNI)](tutorials/android.md)
  - [Godot GDExtension](tutorials/godot.md)
  - [Desktop CLI deep-dive](tutorials/cli.md)
  - [Server (four compatibility APIs)](tutorials/server.md)
- **Adding a backend**: [backend-guide.md](backend-guide.md)
- **API reference**: [api-reference.md](api-reference.md)
- **Migrating from another runtime**: [Migration Guide](migration-guide.md)
  (from ONNX Runtime / whisper.cpp / sherpa-onnx)
- **License / Compliance**: [`docs/license-audit.md`](license-audit.md),
  [`docs/legal-compliance.md`](legal-compliance.md)

## Troubleshooting

- **`error: model file has no vokra.model.arch metadata`**: the GGUF was
  produced by a non-Vokra tool (e.g. `llama.cpp`). The Vokra runtime only
  accepts GGUFs written by its own converter — regenerate with the
  `vokra-cli convert` recipes above.
- **`error: backend does not implement op X`**: GPU backends do not
  silently fall back to CPU (FR-EX-08). Retry with `--backend cpu` or open
  an issue with the model / op name.
- **`error: research flag required for CC-BY-NC weight`**: non-commercial
  weights (F5-TTS / Fish-Speech / EnCodec) are refused by the compliance
  gate. Explicit opt-in is required for research use.
