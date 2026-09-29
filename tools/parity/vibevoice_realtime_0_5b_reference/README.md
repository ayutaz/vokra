# VibeVoice Realtime 0.5B reference contract

This directory records the model-free contract used by
`tools/parity/vibevoice_realtime_0_5b_inspect.py`. It is an inspection gate,
not a runtime or parity implementation.

## Fixed upstream identities

The gate binds all evidence to these immutable revisions:

- HF model: `microsoft/VibeVoice-Realtime-0.5B`
  `6bce5f06044837fe6d2c5d7a71a84f0416bd57e4`
- VibeVoice source: `microsoft/VibeVoice`
  `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`
- Transformers: `v4.51.3`
  `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`
- Base tokenizer: `Qwen/Qwen2.5-0.5B`
  `060db6499f32faf8b98477b0a26969ef7d8b9987`

The VibeVoice source checkout is authenticated by exact Git blob identities
for the streaming processor, acoustic tokenizer processor, text tokenizer,
and acoustic tokenizer implementation. The gate additionally checks source
markers to document and enforce the expected streaming API surface; the exact
Git blob identities remain the source of file identity and provenance.

## Tokenizer roles

Only the following selected Qwen tokenizer files are structurally inspected;
the recursive server walk is still authenticated in full:

| File | Model-free role contract |
| --- | --- |
| `tokenizer_config.json` | Qwen2 tokenizer configuration with a positive `model_max_length` |
| `tokenizer.json` | Fast-tokenizer JSON whose model is a non-empty BPE vocabulary and merge table |
| `vocab.json` | Non-empty token-string to non-negative integer map |
| `merges.txt` | UTF-8 BPE pairs with no duplicates; the fixed snapshot is headerless, while an optional `#version: 0.2` line is tolerated |
| `LICENSE` | Non-empty text retained for a separate license review |

Tokenizer model weights are not selected or downloaded. The tokenizer license
and redistribution decision remain `SEPARATE_REVIEW_REQUIRED`.

The upstream text tokenizer defines slow and fast Qwen2 variants and adds
`<|vision_start|>`, `<|vision_end|>`, and `<|vision_pad|>` as additional
special tokens. The pinned source records their Realtime speech boundaries as
token strings; the exact pinned tokenizer sidecars authenticate their IDs as
`151652`, `151653`, and `151654`, respectively. The Rust
`vokra_models::vibevoice_streaming::tokenizer::VibeVoiceRealtimeTokenizer`
primitive accepts only the four byte-authenticated sidecars below and reuses
the first-party byte-level BPE implementation; any size, hash, JSON role, or
GGUF metadata drift fails closed:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer_config.json` | 7,228 | `c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9` |
| `tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |

The primitive's `streaming_text_ids` applies the upstream streaming boundary
(`text.strip()` followed by one newline and no added special tokens).
Historical tokenizer evidence at exact implementation HEAD
`89a211738705f603a88b6996d4151cea689ae6db` is bound only to that commit and
must not be reused as proof for a later rebased or integration HEAD. That
disposable VAST run compiled the Rust package, bound the exact sidecars, and
matched independently obtained pinned Transformers Qwen2TokenizerFast IDs for
representative ASCII, Unicode, whitespace, and punctuation inputs. Reserved
speech-boundary literals are rejected by policy; their official IDs are
checked separately against the authenticated sidecar records. A fresh
integration-HEAD receipt is required for the branch that carries this
contract. These are model-free tokenizer results, not model execution,
synthesis, or full parity.

## Streaming input/output contract

The fixed upstream processor intentionally rejects the ordinary `__call__`
path for realtime input. Streaming callers must use
`process_input_with_cached_prompt(...)` with one text input and a non-null
cached prompt. The model-free source contract records the output field names
`input_ids`, `tts_lm_input_ids`, `tts_text_ids`, `speech_tensors`,
`speech_input_mask`, `attention_mask`, `tts_lm_attention_mask`, and
`speech_masks` where applicable.

The acoustic processor requires audio, uses the upstream 24 kHz configuration,
normalizes to the configured target, and returns its processor field as
`audio`. The realtime preprocessor declares a speech compression ratio of
3200; speech mask lengths are therefore derived from the upstream ceiling
operation. The pinned `process_input_with_cached_prompt` path itself sets
`speech_inputs` to `None`; the native
`vokra_models::vibevoice_streaming::state` module therefore binds only its
model-free cached-prompt text boundary. It takes the two authenticated
hidden-state lengths, authenticates the VibeVoice Fast tokenizer's
`<|image_pad|>`-based `pad_id` from the exact sidecars (this intentionally
differs from the ordinary Qwen `pad_token` field), emits pad-filled pseudo
LM/TTS-LM IDs, and tracks newline-terminated text steps without owning a KV
cache. Audio preparation, sampling-rate conversion,
staged model execution, EOS/diffusion/acoustic decoding, and output numerical
parity are not implemented by this gate. The state module must not be read as
evidence that synthesis or real-weight parity is complete.

## Native prediction-head boundary

The native `vibevoice_streaming::diffusion` module implements one
source-derived prediction-head forward step on an explicitly selected
`Compute` backend. Its small scalar-oracle test checks the calculation
contract only. A fixed range read of the HF safetensors header is now
authenticated: the 79,432-byte JSON header declares all 605 tensors as BF16,
and the 26 `prediction_head` tensor names/shapes match the native loader in
PR #160. The authenticated range SHA-256 is
`73c4658be17469d62e22a0f4b7f042cc10aa83d409055a5e3a350d5b8d8f26cb`.
This authenticates only the header range; the complete payload hash and
real-weight tensor values remain unverified.

The native `vibevoice_streaming::sampler::sample_vibevoice_realtime_cfg`
boundary now contains the fixed 20-step CPU-only CFG loop and an explicit
non-CPU rejection. Its tests use a deterministic synthetic predictor to check
branch order, shared latents, scheduler reset, and 20 steps. This is a
model-free contract test only: it is not an official numerical reference,
real-weight CPU parity, Apple CPU/Metal parity, or a full acoustic synthesis
route. The tokenizer, staged language-model execution, and publication gates
remain separate; the acoustic decoder boundary is described below.

At exact implementation commit `21dded9b7b0630266ee6c223b87d5ff7e569e2bc`,
a disposable VAST instance ran `cargo test -p vokra-models --lib
vibevoice_streaming::diffusion` (5 passed) and `cargo test -p vokra-models
--lib vibevoice_streaming` (20 passed, 2 ignored because the authenticated
Qwen sidecars were absent). Both used Rust 1.98.1 and the instance was
destroyed with an exact-ID readback of `instances: null`. This is model-free
Rust test evidence, not a speed comparison or real-weight parity receipt.

## Native Realtime acoustic decoder boundary

The fixed HF header range also records 276 BF16 tensors under
`model.acoustic_tokenizer.decoder.*`; its canonical descriptor digest is
`4576ebac2def6293d72668f3fa068461b7c256c6d75e9f12eb6c16207ccfbfbb`.
The descriptor shapes match the already-native causal VibeVoice acoustic
decoder (stem 64→2048, stage widths 2048/1024/512/256/128/64/32 with depths
8/3/3/3/3/3/3, six transposed convolutions, and a 32→1 head). The dedicated
Realtime constructor first applies
`VibeVoiceStreamingCheckpoint::from_gguf`, enforces exactly 276 decoder-prefix
tensors, then reuses the 1.5B decoder loader and scalar conversion logic. The
rank-0 `speech_bias_factor`/`speech_scaling_factor` tensors are loaded and
validated by the Realtime-specific `VibeVoiceLatentScale` loader after the
composite checkpoint gate; they are not part of the descriptor-only composite
tensor gate itself.

`VibeVoiceRealtimeAcousticDecoderStream::decode_scaled_latent` accepts one
scaled `[64]` frame, applies the official
`latent / scaling_factor - bias_factor` conversion, and forwards it through
the existing causal decoder stream to one 3,200-sample 24 kHz mono chunk. It
rejects non-CPU backends explicitly; no CPU fallback is implied. The tests are
model-free contract tests only. The header range does not authenticate the
full payload, and no real-weight CPU parity, independent waveform reference,
Metal parity, or complete synthesis claim follows from this boundary.

## Verification boundary

Run only the model-free checks:

```text
uv run --no-project --python 3.12 python tools/parity/vibevoice_realtime_0_5b_inspect.py --self-test
uv run --no-project --python 3.12 python tools/parity/vibevoice_realtime_0_5b_inspect.py --gate-self-test
```

No model, tokenizer snapshot, or tensor body is loaded by these checks. The
HF model contains a roughly 2 GB safetensors payload and remains VAST-only;
the inspector only accepts a future authenticated remote evidence packet. A
real-weight reference run, independent native parity, owner/legal approval,
dataset provenance review, and public publication are still blocked and must
not be inferred from this structural contract.
