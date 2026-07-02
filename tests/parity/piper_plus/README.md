# piper-plus MB-iSTFT-VITS2 parity fixtures (M0-07-T21)

Numerical-parity reference for the native piper-plus TTS (`vokra-models`
`piper_plus`), generated **offline** from the distributed piper-plus voice via
onnxruntime. onnxruntime is used only here, never in the runtime or CI
(FR-LD-05); the fixtures are committed so the parity tests need no ONNX.

## Files

- `gen_reference.py` — the generator (needs `onnx` + `onnxruntime` + `numpy`).
- `manifest.txt` — `key = value` generation conditions (input phoneme ids,
  scales, shapes, sample rate, `piper_version`).
- `*.f32` — little-endian `f32` reference arrays:
  - `pcm.f32` — final PCM `[T_samples]`.
  - `durations.f32` — pre-ceil durations `[T_phonemes]`.
  - `m_p.f32` / `logs_p.f32` — encoder prior stats `[HIDDEN, T_phonemes]`.
  - `dec_input.f32` — post-flow decoder-input latent `z·y_mask`
    `[HIDDEN, T_frames]`.
  - `sdp_body.f32` — stochastic-duration-predictor body (proj output)
    `[DP_FILTER, T_phonemes]`.

## Determinism

The VITS noise is disabled by zeroing the noise scales
(`scales = [0, length_scale, 0]`, `docs/piper-plus-integration.md` §5), so the
reference is fully deterministic. The distributed voice is FP16 ONNX, which
onnxruntime casts to FP32 for every op, so it is compared against the FP32
native implementation at the FP32 parity bound (NFR-QL-01 `atol = 0.01`).

## Regenerating

```sh
python gen_reference.py <tsukuyomi-6lang-fp16.onnx> <config.json> .
```

The voice model itself is **not committed** (~40 MB FP16 / ~77 MB FP32 GGUF).
The native parity tests (`crates/vokra-models/src/piper_plus/parity.rs`) are
gated on `$VOKRA_PIPER_GGUF` and skip cleanly when it is unset (e.g. in CI),
mirroring the Whisper parity tests (`$VOKRA_WHISPER_GGUF`). To run them locally,
convert the voice and point the env var at the GGUF:

```sh
cargo run -p vokra-convert -- --model piper-plus \
    --input tsukuyomi-6lang-fp16.onnx --config config.json --output voice.gguf
VOKRA_PIPER_GGUF=voice.gguf cargo test -p vokra-models piper
```

The voice: `ayousanz/piper-plus-tsukuyomi-chan` (6-language multilingual medium,
MB-iSTFT-VITS2, `piper_version` 1.11.0), MIT.
