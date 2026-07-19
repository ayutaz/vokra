# Silero VAD v5 parity fixtures (M0-05, NFR-QL-01)

Reference data for the native Silero VAD v5 subgraph in
`crates/vokra-models/src/silero_vad/`. Ground truth is **onnxruntime** running
the upstream `snakers4/silero-vad` `silero_vad.onnx`; the native model's GGUF
weights are extracted from the *same* ONNX, so ORT is a faithful oracle.

Model/architecture details and the pinned-down numeric facts live in the code
SPEC: `crates/vokra-models/src/silero_vad/SPEC.md`.

## Files

| file | contents |
|------|----------|
| `gen_reference.py` | regenerates everything below from the upstream ONNX |
| `silero-vad-v5.gguf` | corrected **both-rate** GGUF (30 tensors, `sr8k.*` / `sr16k.*`); the model loads this |
| `test_16k.wav`, `test_8k.wav` | deterministic mono float32 PCM (silence / noise / tone); shared by the streaming test and the `vad_demo` example |
| `probs_16k.txt`, `probs_8k.txt` | ORT speech probability per fixed frame (512 @16k / 256 @8k), LSTM state carried — the **raw** bare-frame e2e reference |
| `probs_16k_ctx.txt`, `probs_8k_ctx.txt` | same clips through the **official** wrapper semantics (`utils_vad.py OnnxWrapper`): rolling 64-sample (@16k; 32 @8k) audio context prepended, ORT fed `[1,576]` / `[1,288]` — the default-stream e2e reference |
| `probs_jfk30s_ctx.txt` | official-context ORT reference over the real-speech clip `tests/fixtures/audio/jfk-30s.wav` (PCM16 mono 16 kHz, sha256 `58adb4ea…`; 343 complete frames) — backs the P1 real-speech regression test |
| `jfk-30s-8k.wav` | the same real speech decimated to 8 kHz (PCM16 mono, sha256 `5c8c1ad4…`, 88 000 samples) by `gen_reference.py::halfband_decimate` — an anti-aliased Kaiser-sinc lowpass then 2:1 decimation, **not** `x[::2]`. Committed (rather than resampled inside the test) so the Rust stream and ORT score byte-identical samples |
| `probs_jfk8k_ctx.txt` | official-context (**ctx288**) ORT reference over `jfk-30s-8k.wav`; 343 complete frames — backs the 8 kHz real-speech regression test |
| `step_stftconv_<rate>.txt` | ORT pseudo-STFT conv output, first frame / zero state (T04) |
| `step_magnitude_<rate>.txt` | ORT magnitude spectrogram, first frame (T05) |
| `step_encoder_<rate>.txt` | ORT encoder output, first frame (T06) |

Float fixtures are one value per line, parsed with Rust `str::parse` (never
`strtod` — NFR-RL-01).

**Provenance**: all references were generated from the upstream **master**
`silero_vad.onnx` (`src/silero_vad/data/silero_vad.onnx`, sha256 `1a153a22…`),
the exact ONNX the fixture GGUF's weights are extracted from. (The `v5.0`
release-tag ONNX carries *different* weight values — the 2026-07-16 eval
confirmed only master reproduces this GGUF byte-identically.) The `_ctx` /
jfk references were generated with onnxruntime 1.19.2 on 2026-07-16 and are
byte-identical to the eval campaign's independent ctx576 dumps
(`docs/bench-baselines/m1-real-weight-eval-2026-07-16/`); the raw/step
references regenerate byte-identically from the original M0 run.

## Tolerance

FP32 `atol = 0.01` (NFR-QL-01). Measured max abs error of the native
implementation vs ORT: e2e streaming raw 7.9e-8 (16k) / 3.2e-6 (8k), official
context 2.1e-6 (16k) / 3.3e-7 (8k), real-speech jfk 6.1e-6 @16k and 1.2e-6
@8k (both max prob 1.0000, 4/4 segments matching upstream
`get_speech_timestamps` at threshold 0.5); pseudo-STFT and magnitude are
bit-exact; encoder ≤ 4.9e-4. See SPEC for the full table.

**Why two prob interfaces**: the official silero-vad python wrapper prepends a
64-sample rolling context to every 512-sample frame (`[1,576]` into the
graph). Bare `[1,512]` frames are numerically valid but collapse on real
speech (max prob 0.0037 on jfk → zero detections — the 2026-07-16 real-weight
eval P1). The runtime's public VAD entry points therefore use the official
context; the raw fixtures pin the 1:1 graph port via the in-crate raw stream.
At 8 kHz the raw interface instead peaks at 0.646 while still producing zero
segments (2026-07-19 measurement, SPEC §"8 kHz quadrant") — same product-level
gap, different symptom.

**Weight provenance matters more than it looks**: the v5.0 release-tag ONNX and
the pinned master ONNX produce probability streams that differ by up to **0.81**
on the same clip — a factor of ~7e5 over the ~1e-6 engine-vs-ORT agreement. A
GGUF built from one ONNX must never be compared against ORT running the other:
the mismatch looks exactly like a catastrophic parity failure. The local eval
cache contains one such trap (`silero-vad-v5-bothrate.gguf` is v5.0-release
weights despite the converter-fix name) — see
`docs/bench-baselines/silero-8k-ctx288-2026-07-19/report.md` §4.2.

## Regenerating

Requires the upstream `silero_vad.onnx` and a Python env with
`onnx` / `onnxruntime` / `numpy` (see `../parity-requirements.txt`):

```sh
python gen_reference.py /path/to/silero_vad.onnx
```

The step-level intermediates are obtained by lifting each ONNX `If` branch
(8 kHz / 16 kHz) into a standalone graph and exposing internal tensors as
outputs — the values are ORT ground truth, not re-derived by hand.

## Note on the GGUF

`silero-vad-v5.gguf` is written here directly from the ONNX by
`gen_reference.py`. As of 2026-07-16 the production converter emits the same
corrected both-rate scheme: `vokra-cli convert --model silero-vad` on the
master `silero_vad.onnx` reproduces this fixture **byte-identically**
(sha256 `9de80aca…`; the old converter de-duped the two `If` branches and
dropped the 16 kHz model — see SPEC "Conversion"). The fixture stays committed
so parity tests run without the upstream ONNX.
