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
markers for the streaming API so a role file cannot be replaced by a
same-shaped placeholder while retaining a superficially valid inventory.

## Tokenizer roles

Only the following selected Qwen tokenizer files are structurally inspected;
the recursive server walk is still authenticated in full:

| File | Model-free role contract |
| --- | --- |
| `tokenizer_config.json` | Qwen2 tokenizer configuration with a positive `model_max_length` |
| `tokenizer.json` | Fast-tokenizer JSON whose model is a non-empty BPE vocabulary and merge table |
| `vocab.json` | Non-empty token-string to non-negative integer map |
| `merges.txt` | UTF-8 BPE pairs with an exact `#version: 0.2` header and no duplicates |
| `LICENSE` | Non-empty text retained for a separate license review |

Tokenizer model weights are not selected or downloaded. The tokenizer license
and redistribution decision remain `SEPARATE_REVIEW_REQUIRED`.

The upstream text tokenizer defines slow and fast Qwen2 variants and adds
`<|vision_start|>`, `<|vision_end|>`, and `<|vision_pad|>` as additional
special tokens. The gate records the source contract only; it does not claim
that either tokenizer has been executed here.

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
operation. Sampling-rate conversion, cached-prompt construction, staged model
execution, and output numerical parity are not implemented by this gate.

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
