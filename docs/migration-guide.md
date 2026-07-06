# Migration Guide

**English** | [日本語](migration-guide.ja.md)

If you already have a speech-inference pipeline built on **ONNX Runtime**,
**whisper.cpp**, or **sherpa-onnx**, this guide covers what changes when
you switch to Vokra, the API mapping table, model conversion, and rough
performance expectations.

## 1. What changes conceptually

| Concern | ONNX Runtime / sherpa-onnx | whisper.cpp | Vokra |
|---|---|---|---|
| Loaded file format | ONNX (Protobuf) | GGUF (ggml-audio) | GGUF (`vokra.*` audio chunks) |
| Speech ops (STFT, iSTFT, mel, VAD state, flow-matching sampler, KV cache) | Ad-hoc host code + graph glue | Whisper-specific inline | **First-class native operators** |
| Backend seams | Execution Providers (asymmetric op coverage) | CPU + optional CUDA/Metal | CPU + Metal + CUDA (staged Vulkan / WebGPU / CoreML / QNN) |
| Silent CPU fallback | Sometimes | No | **No — explicit error (FR-EX-08)** |
| ONNX at runtime | Yes | No | **No** — ONNX is offline conversion only |
| Weight license enforcement | External | External | Built-in `vokra.provenance.*` gate (CC-BY-NC refused without a research flag) |
| Distribution | ORT binaries + your app | Single binary | Single binary; **root `Cargo.lock` has only `vokra-*` crates** |

The design goal is to move speech-specific concerns (frame-accurate STFT,
per-layer KV cache, streaming state, bit-exact frontend) out of your app
code and into the runtime.

## 2. From ONNX Runtime / sherpa-onnx

### 2.1 API mapping

| ONNX Runtime | Vokra |
|---|---|
| `Ort::Session(env, "model.onnx", ...)` | `vokra_session_create_from_file("model.gguf", &s)` |
| `session.Run(inputs, outputs)` | `vokra_asr_transcribe(s, pcm, n, sr, &out)` (task auto-selected from `vokra.model.arch`) |
| `SessionOptions::SetExecutionProviderCUDA(...)` | `vokra_session_set_backend(s, VOKRA_BACKEND_CUDA)` (build with `--features cuda`) |
| Custom op registration | The op is either first-class already or reported as an explicit error |
| `sherpa_onnx_offline_recognizer_*` | `vokra_session_create_from_file` (Whisper GGUF) + `vokra_asr_transcribe` |
| `sherpa_onnx_online_recognizer_*` | `vokra_stream_open` + `vokra_stream_push_pcm` + `vokra_stream_poll` |

### 2.2 Model conversion

Vokra loads GGUF only. ONNX models are converted **offline** — the
runtime never links `onnxruntime` / `protobuf` / `abseil` (FR-LD-05):

```sh
# Silero VAD v5
vokra-cli convert --model silero-vad --input silero_vad.onnx --output silero_vad.gguf

# piper-plus voice
vokra-cli convert --model piper-plus \
  --input voice.onnx --config voice.config.json --output voice.gguf

# CAM++ speaker encoder
vokra-cli convert --model campplus --input campplus.onnx --output campplus.gguf
```

For Whisper, prefer the safetensors path (upstream `openai/whisper-*`):

```sh
vokra-cli convert --model whisper \
  --input model.safetensors --output whisper.gguf
```

### 2.3 What you gain by leaving ONNX behind

- **Correct STFT / iSTFT semantics**: window / hop / normalization / RFFT
  are explicit attributes, not opset drift.
- **Streaming KV cache managed by the runtime**: no more manual
  three-piece decoder splits or 6-GB static caches.
- **No opset upgrade risk**: `torch.onnx.export`'s dynamo /
  scriptmodule split does not apply — Vokra reads safetensors /
  checkpoints directly through the native model implementation
  (whisper.cpp-style).

### 2.4 What you have to reconcile

- **Custom ONNX ops** (contrib operators, `com.microsoft.*`, etc.) do
  **not** carry over. If your pipeline depends on one, either open an
  issue with the op signature so we can consider promoting it, or keep
  that step in your host code.
- **Windows Bitcode** and other ORT-specific packaging concerns fall
  away — Vokra ships a single `libvokra.dll` / `.dylib` / `.so`.

## 3. From whisper.cpp

### 3.1 API mapping

| whisper.cpp | Vokra |
|---|---|
| `whisper_init_from_file("model.bin")` | `vokra_session_create_from_file("whisper.gguf", &s)` |
| `whisper_full_default_params(WHISPER_SAMPLING_GREEDY)` + `whisper_full` | `vokra_asr_transcribe(s, pcm, n, 16000, &out)` |
| `whisper_full_get_segment_text` | Full text returned as a single `char*`; per-segment API is planned |
| `whisper_state` reuse across chunks | `vokra_stream_open` + `vokra_stream_push_pcm` (streaming ASR is v0.5+) |
| `whisper.cpp` GGUF | **Not compatible** — Vokra GGUFs carry `vokra.*` audio metadata chunks that whisper.cpp does not read, and vice versa. Convert from safetensors. |

### 3.2 Model conversion

```sh
# From Hugging Face safetensors — supports base / small / medium / large-v3 / turbo.
vokra-cli convert --model whisper \
  --input openai_whisper-large-v3/model.safetensors \
  --output whisper-large-v3.gguf
```

Quantization presets available: `--quantize q4_k` / `q5_k` / `q6_k`
(alias for `--policy-preset whisper_q4_k` etc.). See
`docs/design/quantization-policy.md` for the layer-level policy scheme.

### 3.3 Performance expectations

- **Whisper base CPU**: RTF < 0.3 target (parity with whisper.cpp on the
  same CPU; both use K-quant and a hand-tuned kernel path).
- **Whisper large-v3 on RTX 4090**: **RTF < 0.15** end-to-end (measured
  0.081–0.115). This includes device-resident encoder + per-decoder-step
  device residency + a fused FA v2 causal-attention kernel. whisper.cpp's
  CUDA path is typically 2–3× slower on the same GPU because it goes
  through cuBLAS rather than a fused kernel.

### 3.4 What you have to reconcile

- **Language auto-detect** is currently emitted through the standard
  Whisper prompt; a first-class `detect_language()` shortcut is planned.
- **Word-level timestamps**: not yet emitted through the C ABI; use the
  `beam_search` op path via the Rust API in the meantime.

## 4. From `faster-whisper` (Python)

The Python binding (`vokra` on PyPI) maps `faster-whisper`'s common
surface almost 1:1:

```python
# faster-whisper
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda")
segments, _ = m.transcribe("speech.wav")
text = " ".join(s.text for s in segments)

# vokra
from vokra import Session
with Session.open("whisper-large-v3.gguf") as s:
    pcm, sr = read_wav_mono_f32(open("speech.wav", "rb"))
    text = s.transcribe(pcm, sr)
```

For a **drop-in HTTP client**, run
[`integrations/vokra-server`](../integrations/vokra-server) — it exposes
OpenAI Whisper's `/v1/audio/transcriptions` (faster-whisper drop-in),
vLLM's `/v1/completions` + `/v1/chat/completions`, piper-plus HTTP
`/api/tts`, and the Wyoming Protocol for Home Assistant. Change the URL,
leave your client code alone.

## 5. From `piper` / `piper-plus`

The Vokra piper-plus integration is a **native reimplementation** of
`piper-plus`'s inference stack (MB-iSTFT-VITS2). Voice models are
converted **offline** from the upstream ONNX + `config.json` pair; the
runtime carries no `onnxruntime` dependency.

```sh
vokra-cli convert --model piper-plus \
  --input voice.onnx --config voice.config.json --output voice.gguf
```

The 8-language G2P (JA/EN/ZH/ES/FR/PT/SV/KO) from piper-plus is reused as
a preprocessing layer; a Rust port is on the roadmap but not required
for using the Vokra native TTS path.

## 6. Compliance changes

If your pipeline used **CC-BY-NC / CC-BY-NC-SA weights** (F5-TTS,
Fish-Speech, EnCodec) directly, Vokra refuses them by default. Opt in
with an explicit `research_flag: true` in `ComplianceLevel` (or the
equivalent CLI switch). This is a design decision to protect commercial
users of the runtime — see [`docs/legal-compliance.md`](legal-compliance.md).

## 7. What Vokra does not (yet) do

- **Speaker diarization** (`pyannote`-equivalent): planned; not shipped
  in v0.5.
- **Bark / StyleTTS 2**: planned for v2.0+ after license audit.
- **Voice cloning (RVC v2 / GPT-SoVITS)**: intentionally moved to the
  separate `vokra-voiceclone-experimental` repository for legal reasons
  (ELVIS Act / NO FAKES Act).
- **ONNX at runtime**: never (design decision — see
  [`docs/onnx-alternative-research.md`](onnx-alternative-research.md)).

## Next steps

- [Getting Started](getting-started.md) for a 5-minute quick start.
- [Tutorials](tutorials/) for Unity, iOS, and Python integrations.
- [License audit](license-audit.md) for the full weight-license table.
