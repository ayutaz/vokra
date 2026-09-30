# Official VibeVoice Realtime streaming reference

`run_streaming_reference.py` is the independent reference caller for the
complete Microsoft streaming path. It invokes the pinned
`VibeVoiceStreamingForConditionalGenerationInference.generate` implementation
from the clean Microsoft checkout; it does not copy the generation loop,
diffusion sampler, scheduler, acoustic decoder, or tokenizer algorithm.

The real-weight path is intentionally VAST-only (`Linux/x86_64` with
`VOKRA_PUBLISH_ON_VAST=1`). It refuses to import `torch`, load the checkpoint,
or load the Carter preset until all of these have passed:

1. Microsoft/VibeVoice source revision `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`,
   clean tree and authenticated origin.
2. The exact Realtime checkpoint (`6bce5f06044837fe6d2c5d7a71a84f0416bd57e4`,
   2,035,332,888 bytes, SHA-256
   `7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514`) and
   config (`2,117` bytes, SHA-256
   `caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57`).
3. The exact four Qwen sidecars at revision
   `060db6499f32faf8b98477b0a26969ef7d8b9987`, with the byte/hash contract
   already consumed by `vokra_models::vibevoice_streaming::tokenizer`:
   `vocab.json`, `merges.txt`, `tokenizer_config.json`, and `tokenizer.json`.
   The directory must contain exactly those four non-symlink files.
4. The fixed Carter preset at the authenticated Git blob
   `1d795ef667e6641eecb8b22452bb853b089bfdbe` and payload SHA-256
   `a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897`.
5. A bounded text input whose SHA-256 is bound in the owner scope.
6. An owner-supplied execution scope whose file SHA-256 is supplied separately
   as `--owner-scope-sha256`. The scope also binds this script, `run_reference.py`,
   `uv.lock`, the clean Vokra HEAD/tree, the source revision, model/config/preset
   identities, tokenizer sidecars, fixed seed/noise policy, and compatibility
   route. No scope file or approval is created by this repository.

The scope must record an exact owner, UTC approval timestamp, either
`APPROVE_COMMERCIAL_EXECUTION` or `APPROVE_RESEARCH_ONLY_EXECUTION`,
`publication: NO_UPLOAD`, Carter consent evidence reference and hash,
`model_forward: APPROVED`, `audio_generation: APPROVED`, `max_new_tokens` in
the bounded range, and exactly 20 official DDPM steps. It must preserve the
upstream disclaimer and explicitly retain the current deferred watermark
status (`DEFERRED_NO_EMBEDDED_CLAIM`); this runner does not claim to embed a
watermark. The owner scope must also bind the measured `uv.lock` hash and carry
explicit license and security dispositions approved for reference execution,
with `REFERENCE_ONLY_NO_RUNTIME_REUSE` and hash-bound evidence. These are
owner decisions, not claims manufactured by this runner; without them the
execution remains closed.

## Cache compatibility and dtype

The Carter `.pt` is loaded with the upstream demo's `weights_only=True` safe
globals (`BaseModelOutputWithPast` and `DynamicCache`). The pinned official
source helper `_ensure_cache_has_layers` is then applied to each of the four
legacy cache objects. The runner verifies that every key/value tensor is
unchanged byte-for-byte at the tensor-value level, that layer counts and
position lengths are unchanged, and that the official layer `update` and
`get_mask_sizes` API is present. It does not create an ad-hoc pickle shim or
reconstruct a cache from guessed shapes. If this official compatibility route
fails, the run stops as `SOURCE_COMPATIBILITY_OPEN`.

The authenticated BF16 hidden/KV tensors are cast to F32 for the F32 model
replay only after a BF16 round-trip equality check. The original preset is not
rewritten. The packet records the cast and every initial diffusion draw.

Static inspection of the pinned upstream source shows that its
`_init_cache_for_generation` deliberately returns `None` when the installed
`DynamicCache` constructor exposes `config`, and its own
`_ensure_cache_has_layers` adapter supplies the layer interface used by the
official model. The fixed 5.10.4 environment has already loaded the four
legacy cache lists through the safe loader, but full `generate` compatibility
is still an execution-time gate. The runner therefore records the trusted
`run_reference.py` and lock hashes and remains `SOURCE_COMPATIBILITY_OPEN`
until the disposable VAST replay proves the official helper preserves all
four cache values/lengths and the official forward accepts the resulting API.

## Observational traces and device choice

Temporary hooks wrap only official instance methods/modules and restore them in
`finally`, including partial hook-installation failures. They record all four
authenticated prefilled LM/TTS branches, positive LM calls, and
positive/negative TTS calls by authenticated cache-object identity,
sampled diffusion latents, the official decoder input after its
`speech_latent / scale - bias` transform, connector input/output, decoder chunks,
EOS outputs, and cache lengths. The official return values are returned
unchanged.

The CPU trace establishes a tape of the actual initial `torch.randn` values.
The CUDA trace replays that tape while also checking that CUDA's original draw
at each step is identical; a changed draw count, shape, or value is a hard
failure. This is a controlled same-input replay, recorded separately from
observational hooks. One no-trace warmup followed by three no-trace repeats per
device measures timing without
trace-file I/O. CUDA is selected only when all trace stages, cache lengths,
waveform shapes/finiteness, and initial noise match and the unchanged
provisional `5e-2` guard passes, and its median is lower than CPU. That guard
is device selection evidence only, not an independent Rust parity tolerance or
waveform release bound.

The output is `reference.json`, CPU/CUDA PCM arrays, and stage trace arrays.
`reference.json` records both the canonical owner-scope hash and the externally
supplied owner-scope file SHA, plus the measured Vokra HEAD/tree and the exact
reference script, `uv.lock`, and trusted-loader identities.
The output directory must be absent before the run; symlinked ancestors and
clobbering are rejected. Publication remains `NO_UPLOAD`.

## Current status

The model-free self-test covers canonical hashing, duplicate-key rejection,
dependency disposition binding, scope fail-closed behavior,
NaN/bool/budget/timestamp tampering, and hook restoration after both an
exception and partial installation failure. Its dependency/consent values are
ephemeral synthetic fixtures only. No actual
generation is authorized or claimed until the external owner scope is
provided and the exact dependency/source compatibility route is replayed on a
disposable VAST worker. The existing narrow `run_reference.py` remains the
trusted loader/API smoke and exact 605-tensor binding helper.
