# Runtime gap execution plan (2026-08-21)

This plan refines Phase C of
`docs/handoff/remaining-work-plan-2026-08-20.md` from the live source tree. It
is a task ledger, not a completion claim. The inventory source of truth is
`BOUND_ARCHES` in `crates/vokra-cli/src/engine.rs`.

Baseline: `3464de7cc832` (`origin/main` when the working branch was created).

## Audit result and Wave 1 status

At the baseline, the 79 unrouted runtime rows were partitioned as follows:

| Blocker | Baseline | After Wave 1 | After Wave 2 | Current | Meaning |
|---|---:|---:|---:|---:|---|
| `RealForwardNoCliTask` | 3 | 0 | 0 | 0 | Real forward exists; CLI routing is absent |
| `NeedsPairedInput` | 1 | 0 | 0 | 0 | The CLI has no honest two-audio-input contract |
| `NoCliShapedOutput` | 2 | 2 | 0 | 0 | Input/output serialization must be specified first |
| `NoGgufLoader` | 17 | 17 | 17 | 16 | No real artifact can be bound, even if a constructor/forward exists |
| `LoudPartialForward` | 56 | 56 | 56 | 56 | Loading can succeed, but the named forward stops explicitly |
| **Total** | **79** | **75** | **73** | **72** | `BOUND_ARCHES` registry rows |

Wave 1 on `feat/runtime-gap-closure-2026-08-21` closes the four low-cost CLI
registry rows and implements the separate Godot VAD Object-return gap:

- `crepe` and `fcpe` now share the established F0 tabular output contract;
- `wetextprocessing` consumes `--text`, retains the feature-off refusal, and
  is also tested with `--features vokra-wfst`;
- `nkf_aec` requires `--input` plus `--far-end`, rejects sample-rate or length
  mismatch, and emits an exact-input-length WAV on the microphone timebase;
- `session_vad_open_stream` now validates the Godot Int range, opens the C API
  stream, constructs a `VokraStream` Object with populated Rust instance data,
  and packs the Object Variant. Mock-level Object construction/lifetime and
  Variant-constructor tests pass. Real headless/editor smoke remains required
  owner evidence because this crate intentionally carries no VAD GGUF fixture.

These are blocker rows, not 79 equally sized implementation tickets. In
particular, adding a loader must not close a row whose forward still refuses,
and a structural/synthesized-weight test must not close real-weight parity.

## Corrections to the previous ledger

### DeBERTa-v2 mapping is closed for the supported Japanese checkpoint

`ku-nlp/deberta-v2-large-japanese-char-wwm` is already mapped by
`crates/vokra-convert/src/models/deberta_v2.rs`, including the per-layer
attention/FFN/normalization tensors, convolution tensors, relative embeddings,
and the checkpoint's shared relative-attention projections. The pinned real
VAST run in `docs/handoff/parity-sbv2-real-vast-2026-08-18.md` regenerated the
GGUF, matched the committed hash, and passed `bert_hidden_ja` at max absolute
error `2.956390e-5` against the existing `0.05` bound without widening it.

Therefore “finish real-checkpoint DeBERTa-v2 tensor mapping” is not remaining
runtime work for this target. Remaining cleanup is only to remove stale prose,
including the synthetic test comment that names a non-existent
`parity_deberta_v2_layer_bisect.rs`. Supporting another DeBERTa-v2 variant
would be a new, separately scoped model task.

### RNNoise real-weight network closure (completed 2026-08-21)

The old prep script wrote one opaque `rnnoise.weights_blob_f32` tensor and the
converter silently accepted it. That route has been removed. The canonical
prep now verifies the official v0.2 release tarball SHA-256 and parses the 36
arrays embedded in `src/rnnoise_data.c`; the converter rejects missing, extra,
renamed, wrong-sized, or non-F32-container arrays before writing GGUF.

The public `vokra/rnnoise-v0.2` repository was checked on 2026-08-21. Its Hub
API revision was `bedd79292105b7975ddb2383c24c06d4390c100b`, it remained
public, and its 1.4 MB GGUF exposed `rnnoise.weights_blob_f32` but no
`prep_status` marker. The runtime has RNNoise DSP/RNN primitives and a
pass-through converter, but no `vokra-models` RNNoise binder; the env-gated
denoise parity test intentionally panicked when a GGUF was supplied.

The replacement binder implements Xiph's causal Conv1d pair, three 384-wide
GRUs, signed-int8 8×4 sparse matrix walk, scale/bias/recurrent-diagonal
handling, `[z,r,h]` gates, rational activations, 32 gain outputs, and VAD head.
The real GGUF (36 tensors, 4,469,280 bytes) passed four sequential frames
against an independent executable compiled from unmodified Xiph v0.2 C at
`max_abs=1.4901161e-7`, within the unchanged `2e-5` bound. The completed
network tier is gated by `VOKRA_RNNOISE_V02_REAL_GGUF`; the pending pitch and
waveform tier has a separate `VOKRA_RNNOISE_V02_PITCH_REFERENCE` opt-in so a
real network GGUF cannot accidentally trigger an unimplemented test.

Remaining RNNoise work is now narrower and explicit: replace the legacy
22-band/42-feature waveform primitives in `vokra-ops::rnnoise` with v0.2's
32-band/65-feature analysis, official downsampled pitch search and
`remove_doubling`, delayed-spectrum gain application, and overlap-add
synthesis; then add waveform parity. Auditing or replacing the already-public
Hub artifact remains a separate publication action requiring explicit upload
authorization and `publish-one.sh`.

## Wave 1 — small runtime surfaces (implemented on this branch)

These can proceed before the model-family loader/forward waves.

### CLI-01: three low-cost routes

- `crepe`: add an explicit F0 task and reuse the existing F0 tabular renderer.
- `fcpe`: add the sibling F0 task with the same output/timebase contract.
- `wetextprocessing`: route existing `--text` input to `normalize`; retain the
  loud feature-off error and cover a `--features vokra-wfst` build.

Keep CREPE/FCPE rendering shared so both emit the same columns and do not
invent confidence values beyond what the model exposes. Update task dispatch,
loader probes, help text, benchmark classification, and the `BOUND_ARCHES`
rows in the same change.

### CLI-02: NKF-AEC paired-input contract

Add a dedicated far-end/reference WAV option rather than overloading the
speaker-only `--compare` flag. The contract must specify mono handling, equal
sample-rate validation, sample alignment, length mismatch behavior, and the
output WAV timebase. Route `AecEngine::open_stream`/`push_paired`, add parser
and mismatch tests, then reclassify `nkf_aec`. This contract decision precedes
the wiring; `dtln_aec` remains a separate partial-forward row.

### GODOT-01: VAD stream object construction

`session_vad_open_stream` validates its argument and then always reports
pending. The binding layer already has `create_bound_object`,
`object_set_instance`, `create_stream_instance`, and an Object-to-Variant
constructor, so the old claim that instance binding infrastructure is absent
is stale.

Refactor object creation so the new `VokraStream` is constructed with
`StreamInstance { inner: Some(stream) }`, wrap the Godot Object in a Variant,
and clean up correctly on construction/wrapping failures. Cover success,
invalid session/model/sample-rate, constructor failure, Variant failure,
`push_pcm`, `poll`, and `interrupt`. Package-scoped tests may run locally;
headless and editor smoke evidence is still required before advertising full
Godot runtime dispatch.

## Wave 2 — specify the two display contracts (implemented on this branch)

These were design/API tasks first, not formatting-only patches. The branch now
implements both and removes their registry rows:

1. `ct_punc`: `--tokens` reads `vokra-ct-punc-tsv-v1`. Every record pairs one
   `u32` id with one escaped UTF-8 token, so length divergence between separate
   token/id arrays is unrepresentable. Literal Unicode and `\\`, `\t`, `\n`,
   `\r`, `\u{HEX}` escapes are specified; malformed, empty, extra-column and
   out-of-range records fail loudly. Restored text is labelled on stdout or
   written as exact UTF-8 bytes through `--output`.
2. `mimi`: `--codec-mode encode|decode` uses the fixed little-endian
   `VKRMCODE` v1 container. It pins `[frame, codebook]` order, unsigned 32-bit
   codes, mono sample rate, milli-Hz frame rate, frame/sample counts, codebook
   axes, feature width, and SHA-256 of the GGUF's effective codebook table.
   Decode refuses topology/hash/length/range mismatches; encode does not
   resample, pad, or trim.

Parser, binary round-trip, SHA-256 NIST-vector, dispatch, flag-scope, and CLI
package tests are present. Real-weight CT-Punc/Mimi execution remains part of
the final VAST evidence pass; it is not replaced by the structural tests.

## Wave 3 — 16 of 17 GGUF loaders remaining

Implement loaders in dependency-aware families. Every loader needs a writer ↔
reader tensor/metadata handshake, a real pinned checkpoint, license/provenance
evidence, negative shape/key tests, and an independent numerical consumer.

1. Alignment first (1): `charsiu` — **closed in PR #44 on 2026-08-21**.
   The canonical converter consumed all 213 upstream tensors and emitted 211
   runtime tensors, the strict loader bound the real 360 MiB GGUF, and the
   paired CLI route accepts 16 kHz mono WAV plus exact whitespace-delimited
   phone labels and emits versioned TSV. Independent Transformers reference
   parity on VAST measured `max_abs=7.629394531e-6` over the 42 one-frame
   logits (fixed gate `2e-4`). The same VAST run completed the full
   `vokra-models` package: lib `2533 passed / 0 failed / 1 ignored`, with all
   integration/doc-test suites green. Instance `48290692` was destroyed after
   the run; no weight upload occurred.
2. Vocoder family (4): `bigvgan`, `hifigan_vocoder`,
   `speecht5_hifigan`, `vocos`. Share only genuinely identical tensor
   conventions. Vocos also has a second ConvNeXt-V2 forward blocker; loading
   it does not make decoding complete. BigVGAN CPU loader/forward work is
   separate from the full GPU path follow-up.
3. ASR (1): `parakeet-tdt`. Add the artifact binder before exposing its
   existing transcription entry point.
4. TTS scaffolds (11): `chatterbox`, `chatterbox_nano`,
   `chatterbox_turbo`, `cosyvoice3`, `dia`, `irodori-tts`, `qwen3_tts`,
   `vibevoice`, `vits-ja`, `voxcpm2`, `zonos`. Several currently use
   synthesized weights, zero-placeholder axes, or a missing terminal codec.
   Split each into config transcription, real-weight load, forward, terminal
   codec/vocoder, and parity; do not land blanket loaders that only reveal the
   next refusal.

Use the `add-speech-model`, `numerical-parity`, and `license-audit` repository
skills when implementing these waves. Any artifact set of at least 2 GB and
every compiling/testing `vokra-models` Cargo command belongs on VAST.

## Wave 4 — 56 partial forwards

The exact rows are grouped below to expose shared work without treating a
family as one completion checkbox.

| Family | Count | Rows |
|---|---:|---|
| ASR / speech transcription | 13 | `canary`, `canary-1b-flash`, `canary-qwen`, `firered_asr_aed_l`, `gigaam_multilingual`, `kyutai-stt`, `moonshine`, `omniasr-ctc`, `parakeet-ctc`, `parakeet-tdt-1_1b`, `sber_gigaam_v3`, `sensevoicesmall`, `whisper-medusa-v1` |
| VAD / KWS / turn taking | 4 | `firered_vad`, `openwakeword_op`, `smart_turn`, `ten_vad` |
| TTS / speech-to-speech | 6 | `chattts`, `cosyvoice2`, `diffsinger`, `llama_omni2`, `styletts2`, `voila` |
| Music generation/transcription | 6 | `audiogen`, `audioldm2`, `beat-this`, `jasco_400m_chords_drums`, `mt3`, `musicgen` |
| Enhancement / separation / AEC | 8 | `audiosr`, `conv_tasnet`, `demucs`, `dtln_aec`, `facebook_denoiser`, `gtcrn`, `sepformer`, `storm` |
| Speaker / diarization | 3 | `redimnet`, `sortformer`, `speaker_3d` |
| Representation / classification | 11 | `atst`, `clap`, `deepfake_detection`, `eat`, `emotion2vec`, `lang_id_ecapa`, `m2d`, `maest`, `panns`, `w2v-bert-2`, `wavlm_sv` |
| Quality estimation | 4 | `dnsmos`, `nisqa_v2_weight`, `torchaudio_squim`, `utmosv2` |
| Codec | 1 | `snac` |

Recommended dependency order:

1. Repair oracle harnesses and metadata contracts first: parse/compare the
   openWakeWord reference JSON, replace the flow-sampler fixture panic, stamp
   Conv-TasNet topology metadata, and pin JASCO vocabulary/sampler values from
   upstream config.
2. Land shared encoder/decoder primitives with independent op parity, then one
   representative end-to-end model: Conformer/FastConformer + CTC/RNN-T/TDT
   for ASR; frozen speech embedding for KWS; recurrent/stateful blocks for
   enhancement; audio Transformer/feature front ends for representation.
3. Expand each shared primitive only after the representative real checkpoint
   passes. Record a verdict per row; sibling architecture similarity is not
   parity evidence.
4. Defer the largest generative TTS/music/S2S paths until their text/token
   front ends, sampler contract, and terminal codec/vocoder are independently
   proven. Keep explicit errors in place meanwhile.

## PyIN temporal smoothing — completed 2026-08-21

The identity `viterbi_smooth_todo` scaffold was removed. `pyin_detailed`
retains every CMNDF trough, integrates exact Beta(2, 18) CDF interval masses,
builds voiced/unvoiced pitch-bin observations, and decodes the canonical local
triangle/switch transition model with Viterbi. `pyin(...) -> Vec<f32>` remains
the compatibility wrapper, while the CLI now renders the real voiced
probability instead of binary confidence for PyIN.

Independent evidence is committed under
`crates/vokra-ops/tests/fixtures/pyin`: `librosa.pyin==0.11.0` at revision
`af8c839fb15317fa2712ea66e7a22da6a9267b32`, 121 frames covering steady
tones, a short octave spike, silence boundaries, and voiced/unvoiced
transitions. The fixed gates are exact voiced-state agreement, F0 absolute
error <= 1e-3 Hz, and voiced-probability absolute error <= 2e-4. The focused
unit suite and independent parity test pass locally with `CARGO_BUILD_JOBS=1`.

## Other explicit non-row holes retained

- Real `TtsEngine::synthesize_stream`; a one-chunk synchronous wrapper is not
  streaming completion.
- microWakeWord emitted quantization metadata plus a real fixture.
- native BF16 compute.
- full HiFTNet GPU generator and full BigVGAN GPU path, with no silent CPU
  fallback.
- SBV2 language-row ordering, spline `num_bins`, and production Mandarin
  segmentation/word boundaries. This is separate from the completed
  DeBERTa-v2 mapping and ZH numerical fixture.
- vLLM completion generation scope and the other explicit server 501
  contracts. Preserve the 501s until scope and implementation are real.

## Definition of done and verification routing

For every row moved out of `BOUND_ARCHES`:

1. The real entry point completes for a pinned, licensed checkpoint; no
   synthesized/placeholder data is accepted as a production artifact.
2. Converter metadata and loader requirements round-trip, with missing/wrong
   keys and shapes failing loudly.
3. An independent upstream reference fixture exercises the implemented path;
   tolerances are justified and not widened to make a failure disappear.
4. CLI/API help, snapshots, docs, provenance, license rows, and the registry
   classification change together.
5. `scripts/check-bound-arch-coverage.sh`,
   `scripts/check-arch-handshake.sh`, the relevant focused checks, and
   `git diff --check` pass.
6. Local verification stays package-scoped and serial. All `vokra-models` or
   workspace-scale Cargo work runs on a disposable VAST instance, which is
   destroyed after evidence is recorded.

Model publication remains outside this execution branch. In particular, the
RNNoise public-repository audit does not authorize replacement or upload.
