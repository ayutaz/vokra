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
official runner. The CLI receives this value explicitly through the required
`--realtime-max-speech-steps <N>` flag; it is not copied from the owner scope's
logical `max_new_tokens` budget.

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

The separate Apple Silicon diagnostic is also ignored by default and requires
an authorized macOS/aarch64 worker with the `metal` feature and
`VOKRA_VIBEVOICE_REALTIME_APPLE_AUTHORIZED=1`:

    CARGO_BUILD_JOBS=1 cargo test -p vokra-models --features metal \
      --test parity_vibevoice_realtime_streaming \
      -- --ignored --nocapture \
      vibevoice_realtime_native_cpu_metal_structural_diagnostic

It reuses the same authenticated packet, owner scope, GGUF, preset, tokenizer,
text, and noise tape for sequential CPU and Metal sessions. It requires backend
identity, event order/layout, cache/EOS/drain, terminal reason, and PCM-length
agreement, while printing CPU/Metal value differences only as
`MEASURED_NOT_GATED` and `STRUCTURAL_ONLY`. It is source preparation, not
evidence that the Apple run has occurred.

## 2026-10-07 GuardedRealtimeCLI route

### 2026-10-07 source capability correction

The production `VibeVoiceRealtimeRuntime` route is distinct from the legacy
model-free `VibeVoiceRealtimeGenerationSession`. The production runtime,
sampler, acoustic connector, and causal decoder select `BackendKind::Cpu` or
`BackendKind::Metal` through the first-party `Compute` registry and reject
uncovered backends before binding; they do not silently fall back to CPU. The
legacy generation control plane remains CPU-only. This source capability does
not constitute Apple hardware execution, CPU/Metal numerical parity, or a
completion/publication decision; those remain separately authorized gates.

The dedicated `vibevoice-realtime` CLI route is source-, reference-, and
owner-bound. Its preflight completes before `VibeVoiceRealtimeRuntime::from_gguf`
is called, so the native binder does not get a chance to substitute an
unbound artifact or a generic tokenizer path. The route rejects generic
`--text` and `--tokenizer`; callers must use the dedicated text file and
tokenizer directory.

The preflight binds all of these inputs independently:

- the mapped/derived GGUF bytes and their externally supplied
  `--realtime-gguf-sha256`; this is deliberately separate from the raw
  Microsoft checkpoint/source `.pt` identity;
- `reference.json` and its CPU trace/noise records through
  `--realtime-reference-dir` and `--realtime-reference-sha256`;
- the owner scope and canonical owner payload, each with its own externally
  supplied digest. The scope supplies the source owner/reference decisions,
  execution limits, Vokra HEAD/tree binding, text digest, and the no-upload
  disposition; this CLI does not invent or self-attest those facts;
- the derived Carter preset cache (`--realtime-preset-safetensors`) and its
  manifest plus external `--realtime-preset-manifest-sha256`. The source
  `.pt` identity and this derived safetensors cache are distinct artifacts and
  must not be conflated;
- exactly the four fixed tokenizer files (`vocab.json`, `merges.txt`,
  `tokenizer_config.json`, and `tokenizer.json`) under
  `--realtime-tokenizer-dir`, with their source-authenticated sizes and
  digests;
- the bounded UTF-8 text file, whose measured digest is compared with the
  owner scope; and
- the externally supplied `--realtime-vokra-head` and
  `--realtime-vokra-tree`, reference-runner, `uv.lock`, and trusted-runner
  digests. These expected HEAD/tree values are bindings from the external
  packet, not binary self-attestation.

The complete authenticated noise tape is loaded from the reference directory
before the native bind. An empty tape, a tape longer than the owner limit, or
an owner `ddpm_steps` value different from the native inference-step contract
fails preflight. The route then creates the output with an exclusive
create-new operation, consumes finite PCM at 24,000 Hz, and explicitly drains
the terminal `Finished` event. It does not overwrite an existing output, make
a synthetic waveform, or fall back to another runtime when the sequence is
invalid.

The current generation control-plane session is CPU-only: its `require_cpu`
check rejects a Metal selection with an explicit backend error. The production
Realtime composite route is separately source-capable on CPU and Metal as
described above; unavailable backends still fail explicitly and never fall
back to CPU. This is a support/error contract, not a claim that Metal parity
is complete.

The following is the complete CLI invocation template from the dedicated
`run.rs` usage. It is a VAST-only template: every value below is an external,
reviewed placeholder, not an invented source owner, SHA-256, or provenance
receipt. The actual command must remain behind the existing source, owner,
license, and no-upload gates.

```sh
# VAST-only; do not run this template on the maintainer Mac.
: "${VOKRA_VIBEVOICE_REALTIME_GGUF:?set externally reviewed mapped GGUF path}"
: "${VOKRA_VIBEVOICE_REALTIME_GGUF_SHA256:?set externally reviewed GGUF SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR:?set externally reviewed reference packet}"
: "${VOKRA_VIBEVOICE_REALTIME_REFERENCE_SHA256:?set externally reviewed reference SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE:?set externally reviewed owner scope JSON}"
: "${VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE_SHA256:?set externally reviewed owner-scope SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_OWNER_CANONICAL:?set externally reviewed canonical owner JSON}"
: "${VOKRA_VIBEVOICE_REALTIME_OWNER_CANONICAL_SHA256:?set externally reviewed canonical SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_PRESET_SAFETENSORS:?set externally reviewed derived preset cache}"
: "${VOKRA_VIBEVOICE_REALTIME_PRESET_MANIFEST:?set externally reviewed preset manifest}"
: "${VOKRA_VIBEVOICE_REALTIME_PRESET_MANIFEST_SHA256:?set externally reviewed manifest SHA-256}"
: "${VOKRA_VIBEVOICE_TOKENIZER_DIR:?set externally reviewed four-file tokenizer directory}"
: "${VOKRA_VIBEVOICE_REALTIME_INPUT_TEXT_FILE:?set externally reviewed text file}"
: "${VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_HEAD:?set external packet HEAD binding}"
: "${VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_TREE:?set external packet tree binding}"
: "${VOKRA_VIBEVOICE_REALTIME_REFERENCE_SCRIPT_SHA256:?set external runner SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_UV_LOCK_SHA256:?set external uv.lock SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_TRUSTED_RUNNER_SHA256:?set external trusted-runner SHA-256}"
: "${VOKRA_VIBEVOICE_REALTIME_MAX_SPEECH_STEPS:?set caller-supplied positive native speech budget}"
: "${VOKRA_VIBEVOICE_REALTIME_OUTPUT:?set fresh output path}"

vokra-cli run --model "$VOKRA_VIBEVOICE_REALTIME_GGUF" --backend cpu \
  --realtime-text-file "$VOKRA_VIBEVOICE_REALTIME_INPUT_TEXT_FILE" \
  --realtime-gguf-sha256 "$VOKRA_VIBEVOICE_REALTIME_GGUF_SHA256" \
  --realtime-reference-dir "$VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR" \
  --realtime-reference-sha256 "$VOKRA_VIBEVOICE_REALTIME_REFERENCE_SHA256" \
  --realtime-owner-scope "$VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE" \
  --realtime-owner-scope-sha256 "$VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE_SHA256" \
  --realtime-owner-canonical "$VOKRA_VIBEVOICE_REALTIME_OWNER_CANONICAL" \
  --realtime-owner-canonical-sha256 "$VOKRA_VIBEVOICE_REALTIME_OWNER_CANONICAL_SHA256" \
  --realtime-preset-safetensors "$VOKRA_VIBEVOICE_REALTIME_PRESET_SAFETENSORS" \
  --realtime-preset-manifest "$VOKRA_VIBEVOICE_REALTIME_PRESET_MANIFEST" \
  --realtime-preset-manifest-sha256 "$VOKRA_VIBEVOICE_REALTIME_PRESET_MANIFEST_SHA256" \
  --realtime-tokenizer-dir "$VOKRA_VIBEVOICE_TOKENIZER_DIR" \
  --realtime-vokra-head "$VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_HEAD" \
  --realtime-vokra-tree "$VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_TREE" \
  --realtime-reference-script-sha256 "$VOKRA_VIBEVOICE_REALTIME_REFERENCE_SCRIPT_SHA256" \
  --realtime-uv-lock-sha256 "$VOKRA_VIBEVOICE_REALTIME_UV_LOCK_SHA256" \
  --realtime-trusted-runner-sha256 "$VOKRA_VIBEVOICE_REALTIME_TRUSTED_RUNNER_SHA256" \
  --realtime-max-speech-steps "$VOKRA_VIBEVOICE_REALTIME_MAX_SPEECH_STEPS" \
  --output "$VOKRA_VIBEVOICE_REALTIME_OUTPUT"
```

This documentation adds no numerical parity, completion, consent, owner
approval, publication, or model-upload claim. The combined HEAD's new Rust
tests and Clippy checks were not run in this documentation refresh; they
remain pending a new combined-head remote proof. No local model download,
native execution, or real-weight CLI invocation was performed.
