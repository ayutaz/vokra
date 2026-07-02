# tests/parity — numerical parity harness (`vokra-parity`)

Test-only workspace crate (`publish = false`) hosting Vokra's numerical
parity suites. It backs the `parity` CI check wired by M0-01-T13.

## Criteria (NFR-QL-01)

Numerical parity against the **PyTorch reference** implementations is
verified in CI for every PR:

- **FP32: `atol = 0.01`** (constant `FP32_ATOL` in `tests/harness_smoke.rs`)
- INT8: `atol = 0.05` (becomes relevant once quantization lands)
- **Per-model criteria are stated explicitly in this directory** — every
  parity suite must document its reference implementation, input fixtures
  and tolerance here (this README and/or a per-suite doc header).

## M0 status

Only the placeholder smoke test `tests/harness_smoke.rs` exists so the CI
check stays green while ops/models are still being implemented. **The real
parity suites are added by the owning work packages:**

| Suite (added by) | Target | Reference | Tolerance |
|---|---|---|---|
| M0-04 | `stft` / `istft` / `mel_filterbank` / `mfcc` / `dct` | PyTorch / librosa / torchaudio | FP32 `atol = 0.01` |
| M0-05 | Silero VAD subgraph (LSTM state 1:1) | Silero VAD reference | per-suite doc (M0-05) |
| M0-06 | Whisper base encoder / decoder / beam search | PyTorch (openai/whisper) | FP32 `atol = 0.01` |
| M0-07 | piper-plus native TTS (MB-iSTFT-VITS2) | piper-plus reference implementation | per-suite doc (M0-07) |

Later milestones extend the table (Kokoro, CosyVoice2, ... and INT8
`atol = 0.05` once quantized paths exist).
