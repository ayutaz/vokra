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
| `NoGgufLoader` | 17 | 17 | 17 | 0 | No real artifact can be bound, even if a constructor/forward exists |
| `LoudPartialForward` | 56 | 56 | 56 | 65 | Loading can succeed, but the named forward stops explicitly |
| **Total** | **79** | **75** | **73** | **65** | `BOUND_ARCHES` registry rows |

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

### 2026-08-22 openWakeWord closure

`openwakeword_op` is no longer a partial-forward row. Audit of the official
v0.5.1 ONNX graphs corrected two scaffold errors: the frontend is a learned
512-sample DFT rather than a 1024-point Hann STFT, and the Alexa classifier is
`16×96 → 128 → 128 → 1`, not a two-layer `96 → hidden → 1` head. The offline
Python 3.12 bridge now extracts the learned DFT/mel tensors, all 20 embedding
convolutions, and every Gemm in graph order. It also passes PCM16 values to
the reference rather than normalized floats that upstream would quantize to
almost all zeros.

The native runtime binds all 48 canonical tensors and implements the exact
1280-sample streaming state, 480-sample raw context, 76-frame mel window,
16-embedding classifier window, first-five suppression, and per-head DNN.
The embedding CNN uses the first-party CPU GEMM seam; the initial scalar path
took 117.68 s for 37 debug hops, while the GEMM path took 12.16 s. The optimized
release CLI processed the same 3 s / 37-hop WAV in 0.376 s (RTF 0.125). Both
implementations measured max probability error `5.960464478e-8` against direct
ONNX Runtime, under the unchanged `1e-4` gate. Official CC-BY-NC-SA-4.0 weights
remained on the VAST validation instance and were not uploaded. The CLI now
routes the arch to a real KWS task and prints detections at the documented 0.5
threshold.

### 2026-08-22 FireRedVAD closure

`firered_vad` is no longer a partial-forward row. The official implementation
at FireRedTeam/FireRedVAD revision
`c30ec49e8cc69642b0ee65362eba11b9d11c6e54` disproved the old transformer
scaffold: Stream-VAD is an eight-stage causal DFSMN over 80-bin Kaldi fbank,
checkpoint CMVN, and a one-column sigmoid head. The offline Python 3.12 bridge
maps every one of the official ONNX graph's 37 floating initializers plus the
two CMVN vectors into a strict 39-tensor F32 bundle. The converter refuses all
missing, extra, retyped, or reshaped tensors and stamps the complete frontend,
DFSMN, cache, variant, provenance, and required-tensor contracts.

The native runtime carries 19 projected history frames for each DFSMN stage,
preserves incomplete PCM framing across pushes, and explicitly converts
normalized runtime PCM back to the int16-valued scale used by the official
Kaldi frontend before checkpoint CMVN. On VAST, the 2,278,176-byte official
GGUF matched direct ONNX Runtime across 222 frames at
`max_abs=1.788139343e-7` for precomputed normalized features and
`3.039836884e-6` for the complete PCM path, under the pre-registered `1e-4`
bound. The same PCM error held when the WAV was pushed in 173-sample chunks,
pinning recurrent and frontend remainder state. The CLI now routes the arch
through the shared VAD run/bench contract.
No model artifact was uploaded or published.

### 2026-08-22 SmartTurn v2 closure

`smart_turn` is no longer a partial-forward row. Inspection of the official
`pipecat-ai/smart-turn-v2` checkpoint at revision
`3267e96b50db03fe030b9869eb35f849a5eea1fa` corrected the stale scaffold: the
model is raw-waveform Wav2Vec2-base, not w2v-BERT/Conformer. Its head is learned
attention pooling followed by `768 → 256 → 64 → 1` classification and sigmoid.
The strict converter consumes the exact 223-tensor F32 source manifest, folds
the parametrized positional-convolution weight norm, omits only the eval-unused
SpecAugment mask vector, and emits a canonical 221-tensor / 379,147,680-byte
GGUF with pinned checkpoint, config, processor, and Pipecat source revisions.

The native binder implements the complete convolutional feature encoder,
feature projection, positional convolution, 12 Transformer blocks, pooling,
and classifier. Pipecat right-pads every utterance to 16 seconds: the first
Wav2Vec2 convolution's GroupNorm therefore depends on the padded time axis.
The new first-party right-padding frontend path reproduces those statistics
analytically without convolving the constant zero tail, while the Transformer
evaluates only the 49 valid keys and 50 ratio-mask-selected queries for the
one-second fixture. A focused op test matches a fully materialized zero tail
within `1e-6`.

The independent pinned Transformers reference produced completion probability
`0.106821209192276`; its trimmed-query check differed from the full 799-query
forward by only `4.470348358e-8`. The ratio-index fixture also pins PyTorch's
`torch.float32` promotion, including the rare input-length boundaries where an
F64 reimplementation would retain one wrong query. The native real-GGUF
forward produced `0.1068216413`, for `max_abs=4.321336746e-7` under the fixed
`1e-4` gate. The CLI now routes SmartTurn as an utterance-level endpoint task
rather than pretending it is a frame-level `VadEngine`. No model artifact was
uploaded or published.

### 2026-08-22 TEN-VAD v1.0 closure

`ten_vad` is no longer a partial-forward row. The official `v1.0-ONNX` source
at commit `8e96899ba05a8e8c0e883ec7417e7a144bd9dec0` fixes the topology at a
`3 × 41` feature context, three separable convolutions, two 64-unit ONNX LSTM
layers, and a `128 → 32 → 1` sigmoid head. The offline Python 3.12 sidecar
verifies the released ONNX SHA-256, maps exactly its 19 float initializers,
and rejects graph-manifest drift. The converter accepts only that complete F32
manifest and stamps every topology, revision, source-hash, and frontend-license
axis consumed by the strict runtime binder.

The native stream implements the official 16 kHz / 256-sample-hop frontend:
pre-emphasis, a 768-point periodic Hann inside a 1024-point FFT, 40 log-mel
features, the LPCNet-derived LPC/pitch tracker, three-frame context, and both
recurrent states. The neural graph matched direct ONNX Runtime across four
stateful steps at `max_abs=5.960464478e-8`. Against the official prebuilt C ABI
over 40 deterministic PCM16 frames, the independent frontend bound is
`3e-4` (measured `2.753734589e-4`) and the complete probability bound is
`1e-3` (measured `6.932020187e-4`). These bounds account for the released
binary's fixed-coefficient Ooura f32 FFT versus Vokra's independently
implemented mixed-radix f32 FFT; they were set after isolating the neural
graph, window coefficients, official internal power spectrum, and feature
context rather than by weakening the exact-network gate.

The license audit also corrected the old plain-Apache assumption. Upstream's
`LICENSE` adds non-compete, application-only deployment conditions and binds
derivatives to them. Canonical GGUFs therefore stamp
`LicenseRef-Agora-TEN-VAD-Open-Source-License-2025` as
`RedistributionForbidden`; no official weight is bundled, uploaded, or
eligible for the Vokra model zoo. Both BSD-2-Clause and BSD-3-Clause LPCNet
notices are preserved for the native frontend.

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

The network binder is intentionally registered in `BOUND_ARCHES` as a
`LoudPartialForward`: `forward_features` is real and parity-verified, but a
`rnnoise` GGUF is not advertised as a runnable denoiser until the waveform DSP
chain above lands. This newly bound arch raised the partial-forward inventory
by one at the time; the current table also includes later loader waves.

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

## Wave 3 — all 17 GGUF loader gaps closed

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
2. Vocoder family: all four rows are closed. Shared code is limited to
   genuinely identical tensor conventions; each checkpoint keeps its own
   preprocessing, padding, normalization, and output-rate contract.

   `bigvgan` closed on 2026-08-21: the strict loader binds the official base
   checkpoint's 448-tensor folded manifest, including all 146 stored
   alias-free Kaiser filter buffers. The native ratio-2 Activation1d matches
   an upstream-import fixture, and the 56,103,872-byte GGUF passed independent
   one-frame mel-to-256-sample waveform parity at
   `max_abs=5.736947e-6` under the tightened `2e-5` bound. The CLI consumes
   explicit channel-major little-endian f32 mel data and emits 22.05/24/44.1
   kHz WAV according to the pinned variant. The VAST CLI smoke used the same
   100-value mel row and produced an IEEE-float32 24,000 Hz WAV with exactly
   256 samples; parsing that WAV independently measured
   `max_abs=5.736514e-6` against the upstream output row. The full GPU
   generator remains a separate backend task; unsupported GPU execution has
   no CPU fallback.

   `speecht5_hifigan` closed on 2026-08-21: the strict loader binds the
   official revision `bb6f429406e86a9992357a972c0698b22043307d` 158-tensor
   manifest, applies its learned 80-bin `mean` / `scale`, and rejects every
   missing, extra, renamed, or wrong-shaped tensor. The 50,636,640-byte GGUF
   (`sha256=96a262c6cd5b222feaca490486c10d9f6d2d9e274f4c2b35e38ce670bcf4a6bc`)
   passed the official Transformers two-frame mel-to-512-sample forward at
   `max_abs=2.7298927e-5` under a `5e-5` FP32 bound. The bound accounts for
   accumulation-order differences between PyTorch CPU convolutions and the
   native scalar kernels across 78 convolutions; it is 1.8x the observed
   delta. The CLI smoke emitted an IEEE-float32 16 kHz WAV with exactly 512
   samples and independently measured `max_abs=2.7299363e-5` after parsing.

   `hifigan_vocoder` closed on 2026-08-21 against SpeechBrain revision
   `4188503131602dc234f48d7f22eebea93d788736`. The audited sidecar validates
   the 234-tensor `generator.ckpt` and folds every `weight_g` / `weight_v`
   pair exactly like the official `remove_weight_norm()`, producing the strict
   156-tensor runtime manifest. The binder preserves both SpeechBrain-only
   contracts: five-frame replicate padding around the mel input and reflect
   padding in every stride-1 Conv1d. The 55,714,976-byte GGUF
   (`sha256=fb11fd44e97344eb6ecca2b2cddf314a1eb443b28424a873b4dc3dec92b5d11e`)
   passed the official SpeechBrain 1.0.3 two-frame mel-to-3072-sample forward
   at `max_abs=3.1374395e-5` under the predeclared `5e-5` FP32 bound. The CLI
   smoke emitted mono IEEE-float32 WAV at 22,050 Hz with exactly 3072 samples;
   independent WAV parsing measured the same `max_abs=3.137439489e-5`.

   `vocos` closed on 2026-08-21 for both official releases. Inspection of
   `vocos==0.1.0` corrected the scaffold's stale “ConvNeXt V2 + GRN” claim:
   the releases use eight ConvNeXt 1D blocks with LayerScale, plain LayerNorm
   for Mel, and four-row bandwidth-conditioned AdaLayerNorm for Encodec. The
   Mel revision `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` binds an exact
   83-tensor manifest; its 54,346,208-byte GGUF
   (`sha256=00586c971bb14d0b96aed1eebd2fc94637619fd718d52d78808d60cbc51116b9`)
   matched the official five-frame feature decode over 1,024 PCM samples at
   `max_abs=2.873130143e-7`. The Encodec revision
   `4e61d082c08045a4c11e5b148ad93b1d0c591a14` binds its distinct 82-tensor
   manifest, including `[16384,128]` codebook embeddings and AdaLayerNorm
   tables; its 40,337,152-byte GGUF
   (`sha256=53faa308126852eb8cea836651e54a5e3151091d1a1b9a746b32cb6037c4a6d0`)
   matched bandwidth id 2 over 1,600 samples at
   `max_abs=3.680586815e-6`. Both pass the fixed `1e-5` bound. The CLI consumes
   channel-major little-endian f32 features, requires explicit
   `--bandwidth-id 0..3` for Encodec, and emitted mono IEEE-float32 24 kHz WAVs
   with the same sample counts and errors under independent parsing. No
   Encodec neural encoder weights are bundled or silently substituted.
3. `parakeet-tdt` loader closed on 2026-08-21. The strict binder validates
   every name and shape in the published 699-float-tensor artifact and rejects
   the 24 training-only BatchNorm `num_batches_tracked` counters. The audited
   upstream revision is `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; the
   existing public GGUF revision is
   `e2448d380310b49b74a6776e9903929ae5a4467d`, with size 2,508,284,704 bytes
   and SHA-256
   `df5e044b040fa27447de23912694b462c6e97b8d5510c24e8c1ed6090dcc0a18`.
   No upload occurred.

   A real numerical consumer now runs the encoder projector, blank/token
   embedding, two-layer zero-state LSTM prediction network, decoder
   projector, ReLU join, and combined 8,198-output token/duration head through
   the shared CPU GEMV kernel. Against the official Transformers implementation
   on VAST, token ids 0, 1, 4096, and blank 8192 measured worst
   `max_abs=5.493164062e-4` and `mean_abs=9.052013047e-5`; all four joint
   argmax values matched. The fixed gates are `max_abs <= 1.2e-3`,
   `mean_abs <= 2e-4`, and exact argmax.

   This moves the row to `LoudPartialForward`, not completion. The remaining
   path is the exact 128-bin log-mel front end, three-stage depthwise-separable
   Conv2D subsampler, relative-position FastConformer attention, eval
   BatchNorm convolution modules, recurrent TDT decode state, and
   SentencePiece detokenization. The generic stacking/RoPE Conformer scaffold
   is not numerically equivalent and is not substituted.
4. The remaining TTS loader slice closed on 2026-08-22. Eight redistributable
   official GGUFs were mmap-bound and exercised on VAST instance `48305436`;
   every binder pins the complete sorted tensor name/shape manifest rather
   than admitting a count-only match. No artifact was uploaded.

   | Arch | Upstream revision | Bytes | Tensors | GGUF SHA-256 | Manifest SHA-256 | Real consumer / measured `max_abs` |
   |---|---|---:|---:|---|---|---|
   | `chatterbox` | `95c8bf4409c237de930c2eec0274fb2b99a21a09` | 2,143,980,064 | 292 | `32733495d1379fc495e091f527139d2b0b5a0fbaf7ec8a53c03f0cebbf939d32` | `4c62a90e6241765f742f27917ac05c08f66623f0b1768e48efd7f7f6bbf84c79` | 256→1024 speaker projection / `8.195639e-8` |
   | `chatterbox_nano` | `49b2f3612ec3e479eb64ce49ab27ae82cbf0b206` | 869,895,424 | 155 | `624bec40b1f590ecf3e336f1ffe0deb42b49089b24abdfc3e1944ff5154cc39d` | `ecc33b97887ddc77e21d06ad225b323afcd10daae15eb82e4b3fdb25350b9798` | 256→768 speaker projection / `1.4901161e-7` |
   | `chatterbox_turbo` | `10fee774c6c5ed890e39cea76d0ae1a320f7a4eb` | 1,915,470,144 | 299 | `ab1a266a42e41a9b4c2ab48fc60040abd9f1c320f807df154c08da986cd601b5` | `c21cfd336cb9b0f70179fcf2308ec66e239a8272193464ba2a78251d8182b880` | 256→1024 speaker projection / `1.2852252e-7` |
   | `cosyvoice3` | `37e7d22a665d96dd7eb2e10e43ff4571783670cc` | 2,577,517,280 | 293 | `d581891f7b25f8b3da80a73b750098108f065f03421e23acf0722f716c3cc84f` | `fb6e0c2c37f12343bd3c7ad52bc6a1551b7ed8945c7bab0e011e666fcbca7705` | layer-0 896→896 Q projection / `5.722046e-6` |
   | `dia` | `dd1df2a129fed7d15c365caeabaae227ccfe8537` | 6,444,673,088 | 343 | `a90733e9e6806cae66abf3eca1d575ecf6dab9298c07d39fc4217a509c952a6d` | `55fce2a39cafba838bd800f6a6aefe63a8e3b1dd86f2727f9a20d87fe6d252f7` | `[256,1024]` text embedding / exact |
   | `vibevoice` | `dec190628f58928fc247b1205b9da2dabc58b9da` | 5,408,160,960 | 1,204 | `8ef5f259dfab0b048151ce52d27468040f72b35b6909528e6db7fbb332ccaeac` | `45cb011420fdb114c7ad61d80888663bcc861e33b7945873836aee2450eb5702` | 64→1536 acoustic projection / `1.4901161e-8` |
   | `voxcpm2` | `ee0ca6d6728c947ecf170e6711bdfbd6decaf0d5` | 1,304,607,744 | 377 | `2c5c3b2509368db3545ea44e66ddd3ef5050ceacd5b5a431d8d8acf1300c6cce` | `d364689d5593ed8886029907a5d17e7659b94f7f310fe95b133c545b6901c509` | 1024→1024 stop projection / `5.9604645e-7` |
   | `zonos` | `b1bf5c56d470eb9097e9b04f9deca364576574ba` | 3,248,843,808 | 246 | `12d542bd219f7f31c91b893810d85b0d810285e603029c69fbd19fd3c7da2c5c` | `6543af3747d3e85bde862c3337744eea31f0105f9df6d8617c1c9afdae805847` | 128→2048 speaker conditioner / `4.4703484e-8` |

   Fixed bounds are respectively `5e-7`, `5e-7`, `5e-7`, `1e-5`, `1e-8`,
   `1e-7`, `1e-6`, and `1e-7`; none was widened after measurement. The
   `voxcpm2` binder accepts only the known published name `voxcpm-0.5b` and
   the current converter alias `voxcpm2-0.5b` under the same exact manifest.
   The official Chatterbox Nano artifact is GPT-2-shaped (12 layers, hidden
   768), exposing stale Llama-shaped scaffold prose/config that remains a
   forward-wave correction rather than being silently treated as equivalent.

   `vits-ja` is the ninth row and has a different evidence boundary. The
   official ESPnet README identifies the 22.05 kHz release as Zenodo record
   5521354 trained at commit
   `628b46282537ce532d613d6bafb75e826e8455de`; JSUT terms prohibit
   redistribution, so Vokra neither fetched nor uploaded the weight. The new
   offline preparer accepts only an operator-held checkpoint, extracts and
   normalizes only the `VITSGenerator` state, and requires the 885-tensor,
   42,011,890-parameter manifest
   `b5d039b6f6febfcb93f2ad17f1647311bb0c37869f54b5e5ceac23f7b951b284`.
   A generator instantiated from that exact official commit and recipe was
   wrapped like an ESPnet training checkpoint, passed through the preparer and
   converter, then passed the strict Rust binder and real 43×192 embedding
   consumer on VAST. This is structural loader evidence, not real-weight
   numerical parity; the operator-provisioned real leg remains fail-closed in
   `parity-tts-japanese-real.yml` until a lawful local artifact is supplied.
5. `qwen3_tts` loader closed on 2026-08-21 for the official
   `Qwen/Qwen3-TTS-12Hz-0.6B-Base` checkpoint. The strict binder validates the
   exact 478-tensor name/shape manifest, including the 76-tensor speaker
   encoder, 28 bias-free talker layers with per-head Q/K RMSNorm, and the
   5-layer code predictor with fifteen residual codebook embeddings and
   heads. It refuses the structurally different 1.7B variants rather than
   admitting them under the 0.6B contract.

   On disposable VAST instance `48305436`, the pinned 1,829,344,272-byte BF16
   safetensors converted to a 1,829,328,672-byte GGUF with 478 tensors and 35
   metadata keys; zero tensors were skipped. The mmap real-artifact harness
   passed the strict binder, decoded real BF16 talker and code-predictor layer
   0 weights, and matched all 23 stamped topology axes against the pinned
   official `config.json`. An independent fixture generated by the official
   `Qwen3TTSTalkerDecoderLayer.forward` at Git revision
   `022e286b98fbec7e1e916cb940cdf532cd9f488e` measured
   `max_abs=3.576278687e-7` and `mean_abs=9.322927023e-8` against the native
   decoder block; fixed gates are `1e-6` and `2e-7`.

   This moves `qwen3_tts` to `LoudPartialForward`, not completion. End-to-end
   PCM still requires the Qwen2 BPE sidecars, multi-codebook autoregressive
   generation loop, and separate 12-Hz neural speech-tokenizer decoder. The
   existing RVQ table fold is not substituted for that decoder.

6. `irodori-tts` loader closed on 2026-08-21 against the published
   `vokra/irodori-tts-500m-v3` revision
   `28e3efaf41f0890784d88f4744c34269e80bdd41`. The 2,048,247,584-byte F32
   GGUF has SHA-256
   `b64d970cf6a7b7cb81579147fa4b661761ee2c224c8da542926dc764fe04e09e`
   and exactly 637 tensors: 384 RF-DiT, 121 text encoder, 98 speaker encoder,
   24 duration predictor, and 10 global projection/norm tensors. The binder
   requires all topology metadata and every tensor name/shape before loading.

   A native text-encoder block now implements the official non-causal masked
   attention, per-head Q/K RMSNorm, adjacent-pair RoPE, sigmoid output gate,
   SwiGLU, and residuals. An independent fixture generated by
   `irodori_tts.model.TextBlock.forward` at upstream Git revision
   `8224dafb46d0aba89209a8f905f1cb7e3299d9c1` measured
   `max_abs=1.192092896e-7` and `mean_abs=2.312784394e-8`; fixed gates are
   `5e-7` and `1e-7`. The real artifact also decoded and executed layer 0
   with finite, non-identity output on VAST.

   This moves `irodori-tts` to `LoudPartialForward`, not completion. PCM still
   requires the LLM-JP tokenizer sidecars, full text/reference stacks,
   duration head, RF-DiT joint-attention sampling loop, and separately
   distributed Semantic-DACVAE-Japanese-32dim decoder.

Use the `add-speech-model`, `numerical-parity`, and `license-audit` repository
skills when implementing these waves. Any artifact set of at least 2 GB and
every compiling/testing `vokra-models` Cargo command belongs on VAST.

## Wave 4 — 65 partial forwards

The exact rows are grouped below to expose shared work without treating a
family as one completion checkbox.

| Family | Count | Rows |
|---|---:|---|
| ASR / speech transcription | 14 | `canary`, `canary-1b-flash`, `canary-qwen`, `firered_asr_aed_l`, `gigaam_multilingual`, `kyutai-stt`, `moonshine`, `omniasr-ctc`, `parakeet-ctc`, `parakeet-tdt`, `parakeet-tdt-1_1b`, `sber_gigaam_v3`, `sensevoicesmall`, `whisper-medusa-v1` |
| TTS / speech-to-speech | 17 | `chattts`, `chatterbox`, `chatterbox_nano`, `chatterbox_turbo`, `cosyvoice2`, `cosyvoice3`, `dia`, `diffsinger`, `irodori-tts`, `llama_omni2`, `qwen3_tts`, `styletts2`, `vibevoice`, `vits-ja`, `voila`, `voxcpm2`, `zonos` |
| Music generation/transcription | 6 | `audiogen`, `audioldm2`, `beat-this`, `jasco_400m_chords_drums`, `mt3`, `musicgen` |
| Enhancement / separation / AEC | 9 | `audiosr`, `conv_tasnet`, `demucs`, `dtln_aec`, `facebook_denoiser`, `gtcrn`, `rnnoise`, `sepformer`, `storm` |
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

### Wave 4 prerequisite repair — completed 2026-08-21

The four harness/metadata blockers named in item 1 are repaired. This does not
move any model row out of `LoudPartialForward`; it makes the later real-weight
work testable and self-describing.

- `openwakeword_op`: the real branch now parses the owner-generated reference
  JSON with `vokra_core::json`, validates sample rate, 1,280-sample prediction
  chunk width, wake-word ordering, non-empty finite probabilities, and compares
  every emitted score at the existing `1e-4` bound. The old "embedding became
  real, now panic and implement the parser" branch is gone.
- flow sampler: the fixture-presence panic is replaced by committed independent
  PyTorch float32 references for linear Euler and sway-scheduled Heun with
  dual-forward CFG. The fixed maximum absolute-error gate is `2e-6`; all seven
  focused `vokra-ops` tests pass locally.
- `conv_tasnet`: the converter now stamps all twelve topology axes and the
  runtime strictly reads and validates them, including the 50% overlap and
  causal flag contracts. Metadata-free artifacts fail with the exact missing
  key instead of silently taking constructor defaults.
- `jasco_400m_chords_drums`: placeholder values were replaced from official
  AudioCraft revision `896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`.
  The chord conditioner has card 194 plus one null row (vocabulary 195), the
  drum conditioner consumes 128-wide EnCodec latents rather than a General-MIDI
  vocabulary, the Euler fallback is 100 steps, and the all-condition CFG
  coefficient is 5.0. The converter stamps the resulting topology group.
  AudioCraft's normal default remains adaptive Dopri5 with `rtol=atol=1e-5`;
  the actual JASCO forward must preserve that distinction when it lands.

### Wave 4 prerequisite VAST verification — completed 2026-08-21

The code-bearing prerequisite commit `062c209f` was transferred without an
unverified push to disposable VAST instance `48303876`
(`vokra-pr44-runtime-gap-verify`: RTX A4000, 16 effective CPU cores, 125.5 GiB
RAM) by git bundle. The following commands completed with exit status 0:

- `cargo test --workspace`, including all workspace integration tests and
  doctests;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo fmt --all -- --check`;
- focused Flow sampler, Conv-TasNet, JASCO, openWakeWord, and converter tests;
- zero-dependency, forbidden-symbol, bound-arch, arch-handshake, zoo-manifest,
  M5 no-C-ABI, and catalog-reality gates.

The first ABI changelog gate run correctly rejected the two newly stamped
metadata groups. Commit `93c2b60f` records `vokra.conv_tasnet.*` and
`vokra.jasco.*` as additive persisted on-disk schema; the gate then passed at
that exact HEAD. Instance `48303876` was destroyed immediately after evidence
collection and the API returned no remaining instance record. No model
artifacts were downloaded, converted, published, or uploaded during this
verification.

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

- `fsmn-vad` conversion still admits an identity-CMVN placeholder instead of
  requiring checkpoint statistics, while its docs overstate real end-to-end
  parity. Replace that producer path with pinned real CMVN extraction and an
  independent Kaldi-fbank/LFR/CMVN/encoder fixture before treating its PCM
  path as validated.
- The Godot VAD stream Object bridge has mock-level coverage but still needs
  real headless/editor construction, lifetime, and push/reset smoke evidence.
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
