# Native Realtime parity consumer

## 2026-10-06 runtime identity binding

Before preset/native model execution, the consumer now binds
`reference.json.runtime.vokra_head` and the canonical lowercase
40-character `reference.json.runtime.vokra_tree_sha1` to the corresponding
`runtime` identities in the authenticated owner scope. Both the packet and the
owner scope HEAD must also equal the externally supplied
`VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_HEAD`; a missing, malformed,
uppercase, truncated, or single-field-mutated identity fails closed. These are
source/packet identity checks only. They do not establish numerical parity,
real-weight completion, or Apple CPU/Metal support; those gates remain
`NOT_RUN` until the separately authorized VAST and Scaleway runs.

crates/vokra-models/tests/parity_vibevoice_realtime_streaming.rs is the
VAST-only consumer for a real packet emitted by
run_streaming_reference.py. It is ignored by default and has no committed
model, preset, text, tokenizer, or waveform fixture. A missing packet is
therefore a correct skip condition, not a synthetic PASS.

The official packet is an independent caller of Microsoft's pinned
VibeVoiceStreamingForConditionalGenerationInference.generate path. Its
reference.json records the source revision, the fixed checkpoint identity,
the Carter preset identity, the authenticated text hash, the Vokra HEAD, and
the CPU packet. The CPU packet contains:

- traces: ordered records for the four prefilled outputs (`lm.positive`,
  `lm.negative`, `tts.positive`, and `tts.negative`), LM/TTS cache
  updates, diffusion predictions, sampled latent calls, acoustic connector and
  decoder calls, and EOS-classifier calls;
- diffusion_initial_noise: the exact [2, 64] tensors drawn by the official
  CFG call. The native sampler consumes the first [64] row, matching the
  official speech[:len(speech)//2] return boundary. The consumer also
  authenticates `noise_draws`, `noise_matching`, and each record's declared
  shape against the actual NPY shape. `noise_hashes` are the official raw
  float32 payload digests; they are checked separately from each TensorRecord
  SHA, which covers the complete NPY file.
- pcm: the official concatenated [1, time] float32 waveform.

Before native execution the test rejects duplicate JSON keys, verifies the
external packet and GGUF SHA-256 values, checks every trace/noise/PCM NPY hash
and shape, binds the exact four-output Carter safetensors cache through
VibeVoiceRealtimePresetCache, binds the four fixed Qwen sidecars through
VibeVoiceRealtimeTokenizer, and verifies the input text hash against the
packet. The official source, checkpoint, Carter payload, and Vokra HEAD must
match the fixed/external identities; publication remains NO_UPLOAD.

## 2026-10-07 diagnostic observer status

The native runtime now has an explicitly opt-in, borrowed, source-ordered
diagnostic observer. The default synthesis/session path installs no observer
and does not copy diagnostic tensors. The observer reuses the single
production sampler and causal decoder computations, and records the actual
LM/TTS hidden outputs, cache positions, diffusion conditional/unconditional
predictions, sampled latent, decoder input/chunk, connector input/output, and
EOS classifier calls. It is session-scoped; observer errors use the existing
poison/reset path, and dropping a session releases the borrowed observer.

The EOS trace preserves text-window calls followed by the positive TTS,
negative TTS, and positive stop-classifier calls for each cached speech step.
Therefore, with `S` sampled speech steps, `U` in `{0,1}` uncached terminal
max-length chunks, and `W` observed text windows, the ordered EOS call count is
`W + 3 * (S - U)`. The sampler output is `[1, 64]`; decoder/connector inputs
are `[1, 1, 64]`, connector output is `[1, 1, 896]`, decoder chunks are
`[1, 1, 3200]`, and EOS output is `[1, 1]`. These layouts are authenticated
source/config contracts and are rejected when a same-numel rank/layout differs.

The consumer now prints per-stage/ordinal worst-bin index, reference/native
values, shape, and aggregate error as `MEASURED_NOT_GATED`; PCM remains
`MEASURED_NOT_GATED`. The added source/model-free tests cover observer ordering,
validation, reset/poison behavior, cache/EOS drain ordering, and borrowed
observer reuse. They do not run real weights, CPU numerical parity, or Apple
CPU/Metal validation. Those remain `NOT_RUN` pending the separately authorized
VAST and Scaleway evidence.

The current run_streaming_reference.py packet schema records the owner-scope
digest but does not copy cfg_scale, max_new_tokens, or ddpm_steps into
reference.json. The consumer therefore requires the same authenticated
owner-scope JSON used by the runner, its externally supplied file SHA-256, and
an externally reviewed canonical scope SHA-256. It binds the packet's two
owner-scope fields to those digests, then reads cfg_scale, max_new_tokens, and
ddpm_steps from the scope itself. A scope's self-declared digest is never used
alone as provenance.

`VOKRA_VIBEVOICE_REALTIME_MAX_SPEECH_STEPS` is a separate native caller
budget, not an official VibeVoice setting. It must cover the observed tape but
is not required to equal `speech_count`: EOS may stop generation early, and
`speech_count` is not evidence from which a caller budget may be inferred. The
consumer does not claim that this native control has been recorded by the
official runner.

The native VibeVoiceRealtimeRuntime::step API still exposes PCM chunks,
generated positions, draining events, and the terminal reason to ordinary
callers. The separately opt-in observer exposes the intermediate values to the
diagnostic consumer without changing default synthesis behavior:

- stage names, ordinals, counts, inference-step count, and all cache-layer
  lengths are checked against the official trace. The observed LM text-window
  prefix may be shorter than the full authenticated input plan when the
  official run reaches EOS or max length; a non-prefix or empty observation is
  rejected as an unsupported packet. With `S` sampled speech steps, `U` in
  `{0,1}` uncached max-length terminal chunks, and `W` observed text windows,
  the expected positive TTS counts are `W + S - U`, negative TTS counts are
  `S - U`, and ordered EOS classifier calls are `W + 3 * (S - U)` (text,
  positive speech, negative speech, then positive stop-classifier calls);
- native `session.generated_positions()` values are checked against the
  official positive TTS cache positions after subtracting the authenticated
  preset's initial position, negative cache progression, noise-tape
  consumption, EOS step, and source-faithful EOS drain count. A source
  max-length terminal audio chunk is recognized separately because the
  official implementation decodes/connects it without appending a TTS cache
  row.
- native PCM length and finiteness are checked, and max-absolute/RMSE waveform
  differences are printed as MEASURED_NOT_GATED. Intermediate stage/ordinal
  values and worst-bin locations are likewise diagnostic-only and remain
  MEASURED_NOT_GATED.

No full-waveform bound is registered here. The result must not be described as
numerical PCM parity, production synthesis completion, voice-consent approval,
or Apple CPU/Metal parity. A future bound needs independent evidence and
manager/owner review; widening a default FP32 gate from one packet would be
fail-closed incorrectly.

## VAST packet contract

The ignored test requires all of the following environment variables:

    VOKRA_PUBLISH_ON_VAST=1
    VOKRA_VIBEVOICE_REALTIME_GGUF
    VOKRA_VIBEVOICE_REALTIME_GGUF_SHA256
    VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR
    VOKRA_VIBEVOICE_REALTIME_REFERENCE_SHA256
    VOKRA_VIBEVOICE_REALTIME_PRESET_DIR
    VOKRA_VIBEVOICE_REALTIME_PRESET_MANIFEST_SHA256
    VOKRA_VIBEVOICE_REALTIME_PRESET_SAFETENSORS_SHA256
    VOKRA_VIBEVOICE_TOKENIZER_DIR
    VOKRA_VIBEVOICE_REALTIME_INPUT_TEXT_FILE
    VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_HEAD
    VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE
    VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE_SHA256
    VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE_CANONICAL_SHA256
    VOKRA_VIBEVOICE_REALTIME_MAX_SPEECH_STEPS

VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR contains the official reference.json
and its cpu/*.npy records. VOKRA_VIBEVOICE_REALTIME_PRESET_DIR contains the
exporter's cache.safetensors and manifest.json. The manifest SHA-256 is
supplied externally because the manifest's self-declared provenance is not
accepted as proof. The actual checkpoint is not loaded by this consumer; its
fixed upstream SHA-256 is checked in reference.json, while the converted GGUF
is authenticated separately by its externally supplied digest.

The VAST command is:

    CARGO_BUILD_JOBS=1 cargo test -p vokra-models \
      --test parity_vibevoice_realtime_streaming \
      -- --ignored --nocapture \
      vibevoice_realtime_native_matches_official_streaming_structure_and_pcm_diagnostic

This command is intentionally not a local-Mac command. It executes the native
CPU runtime on real weights and remains subject to the VAST lifecycle,
license/owner gates, and final instance destruction. No model download,
publication, waveform promotion, or consent claim follows from a green
structural run.
