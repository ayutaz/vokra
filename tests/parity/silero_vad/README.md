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
| `probs_16k.txt`, `probs_8k.txt` | ORT speech probability per fixed frame (512 @16k / 256 @8k), LSTM state carried — the e2e reference |
| `step_stftconv_<rate>.txt` | ORT pseudo-STFT conv output, first frame / zero state (T04) |
| `step_magnitude_<rate>.txt` | ORT magnitude spectrogram, first frame (T05) |
| `step_encoder_<rate>.txt` | ORT encoder output, first frame (T06) |

Float fixtures are one value per line, parsed with Rust `str::parse` (never
`strtod` — NFR-RL-01).

## Tolerance

FP32 `atol = 0.01` (NFR-QL-01). Measured max abs error of the native
implementation vs ORT: e2e streaming 7.9e-8 (16k) / 3.2e-6 (8k); pseudo-STFT and
magnitude are bit-exact; encoder ≤ 4.9e-4. See SPEC for the full table.

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

`silero-vad-v5.gguf` is written here directly from the ONNX because the current
`vokra-convert` de-dups the two `If` branches by name and keeps only the 8 kHz
weights, dropping the 16 kHz model. See SPEC "known conversion gap"; fixing
`vokra-convert` to emit `sr8k.*` / `sr16k.*` is the recommended follow-up, after
which this fixture can come straight from the converter.
