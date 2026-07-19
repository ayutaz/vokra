# Silero VAD v5 — implementation spec (M0-05)

Single source for the 1:1-preserved Silero VAD v5 subgraph: architecture, the
GGUF weight map, the exact numeric details pinned against the onnxruntime oracle,
and the parity methodology. Source of truth for the code in this directory.

Upstream: `snakers4/silero-vad` `silero_vad.onnx` (v5, 2024-06-27; v4 ≈ 1.7 MB,
v5 ≈ 2 MB, LSTM-based; single model handles 8 kHz & 16 kHz — research
`docs/_research/03-speech-specialized-runtimes.md` §3.1). License MIT / MIT,
★ official zoo (`docs/license-audit.md` §3) — commercial use OK, no new audit
entry required for M0-05.

## Design red lines (permanent)

- **1:1 preservation (FR-LD-06 / FR-OP-50).** Kept as a dedicated subgraph, not
  lowered to generic audio-dialect ops and not itself an audio-dialect op. The
  LSTM `h`/`c` and the pseudo-STFT are hidden behind the stream handle; no `pub`
  field exposes them.
- **No librosa/FFT STFT approximation (NFR-QL-05).** The pseudo-STFT is a
  *learned* `Conv1d`; it runs through the module-private `math::conv1d`, never
  the `vokra-ops` `stft` op (FR-OP-01) or any FFT. Verified by grep (see README).

## Architecture

Silero v5 is **two independently-trained networks with identical topology**, one
per sample rate, selected in the ONNX by a top-level `If(sr == 16000)` (verified:
the compare constant is `16000`, `then` = 16 kHz, `else` = 8 kHz).

```
input  [ctx + frame = 576 @16k / 288 @8k]   official interface (default stream)
       [bare frame  = 512 @16k / 256 @8k]   raw 1:1 graph interface (parity only)
  -> reflect-pad RIGHT by n_fft/4            (64 @16k / 32 @8k)
  -> Conv1d(1, 2*bins, k=n_fft, stride=n_fft/2)     learned pseudo-STFT
       n_fft = 256 @16k / 128 @8k;  2*bins = 258 / 130
  -> magnitude = sqrt(real^2 + imag^2)              real=ch[0..bins], imag=ch[bins..2*bins]
       -> [bins, 4]  (official) / [bins, 3]  (raw)  bins = 129 @16k / 65 @8k
  -> encoder: Conv1d(k=3,pad=1)+ReLU x4, strides 1,2,2,1
       channels bins->128->64->64->128;  time 4->4->2->1->1 (or 3->3->2->1->1)  -> [128, 1]
  -> LSTM(128,128)   h/c carried across frames        -> [128]
  -> ReLU -> Conv1d(128,1,k=1) -> Sigmoid             -> probability
```

The ONNX graph's time axis is dynamic and its only *state* input is
`[2,1,128]` = h/c — but **official usage carries audio context too, outside
the graph**: the upstream python wrapper (`utils_vad.py OnnxWrapper`) prepends
the previous frame's last 64 samples (@ 16 kHz; 32 @ 8 kHz) to every fixed
frame, zeros before the first, and resets that context together with the LSTM
state (`reset_states`). `stream.rs` reproduces exactly that (the default,
`ContextMode::Official`); `forward_chunk` uses the zero context of a fresh
stream.

**2026-07-16 real-weight eval P1**: feeding bare `[1, 512]` frames (the
original M0 implementation) is numerically faithful to the raw graph interface
(ORT on the identical inputs matches ≤ 1.13e-6) but semantically wrong — on
real speech the probabilities collapse (max prob 0.0037 on `jfk-30s.wav`,
zero segments at the official threshold 0.5, vs 4 segments / max prob 0.9999
through the official interface). The raw interface is therefore kept only as
`ContextMode::Raw` / `SileroVadV5::open_raw_stream` (test-gated, for parity
against the bare-frame fixtures); every public entry point uses the official
context.

**2026-07-19 (8 kHz quadrant)**: the same comparison at 8 kHz shows the raw
interface fails *differently* but no less completely. On the decimated clip
`jfk-30s-8k.wav` bare `[1, 256]` frames peak at **0.646** — they do cross the
0.5 threshold, unlike the 16 kHz collapse to 0.0037 — yet never sustain speech
for the default `min_speech` (250 ms), so the segment count is still **zero**
against the official interface's four. Any consumer that lowered `min_speech`
or thresholded per frame would therefore get spurious detections at 8 kHz
where 16 kHz merely gets silence; this is a second, independent reason the raw
interface is not a supported entry point.

## GGUF weight map (per rate)

Tensor names are the upstream PyTorch parameter names. Every one of the 15
tensors **differs in value between the two rates** (not only the two
rate-shaped ones) — see "known conversion gap".

| stage | GGUF tensor | shape 16 k | shape 8 k |
|-------|-------------|-----------|-----------|
| pseudo-STFT | `stft.forward_basis_buffer` | `[258,1,256]` | `[130,1,128]` |
| encoder 0 | `encoder.0.reparam_conv.weight` / `.bias` | `[128,129,3]` / `[128]` | `[128,65,3]` / `[128]` |
| encoder 1 | `encoder.1.reparam_conv.weight` / `.bias` | `[64,128,3]` / `[64]` | same |
| encoder 2 | `encoder.2.reparam_conv.weight` / `.bias` | `[64,64,3]` / `[64]` | same |
| encoder 3 | `encoder.3.reparam_conv.weight` / `.bias` | `[128,64,3]` / `[128]` | same |
| LSTM | `decoder.rnn.weight_ih` / `weight_hh` | `[512,128]` ×2 | same |
| LSTM | `decoder.rnn.bias_ih` / `bias_hh` | `[512]` ×2 | same |
| head | `decoder.decoder.2.weight` / `.bias` | `[1,128,1]` / `[1]` | same |

`weights.rs` binds these under two accepted naming schemes:

- **corrected (both rates)** — `sr8k.<param>` / `sr16k.<param>` (the fixture
  GGUF and the target `vokra-convert` output);
- **legacy (single rate)** — bare `<param>` (current `vokra-convert` output),
  the rate inferred from the `stft.forward_basis_buffer` kernel length.

## Exact details pinned against onnxruntime (NFR-QL-05 / T01)

Determined empirically by matching against ORT ground truth — never guessed
(CLAUDE.md "ハルシネーション厳禁"). Intermediates were obtained by lifting each
`If` branch into a standalone ONNX graph and exposing internal tensors.

| item | value | how confirmed |
|------|-------|---------------|
| `If` selector | `sr == 16000` → then(16k), else(8k) | top-level compare constant = 16000 |
| reflection pad | **right side only**, width `n_fft/4` | conv frame count 3 (not 4); side/amount matched to conv ground truth (err ≈ 3e-6) |
| conv real/imag | `real = ch[0..bins]`, `imag = ch[bins..2*bins]` | ONNX `Slice` bounds `[0:bins]` / `[bins:]`, magnitude = `sqrt(re²+im²)` |
| encoder | k=3, pad=1, strides 1,2,2,1 | ONNX `Conv` attrs |
| **LSTM gate order** | **PyTorch `ifgo`** (input, forget, cell, output) | prob err vs ORT 5e-8 with `ifgo`, ~1e-1 with ONNX `iofc` |
| LSTM bias | apply both `bias_ih` + `bias_hh` | PyTorch convention; confirmed by e2e match |
| head | ReLU → Conv1d(k=1) → Sigmoid; time frame = 1 so ONNX `ReduceMean` is a no-op | e2e match |
| state | `h`/`c` zero-initialised; `[2,1,128]` in ONNX = (h,c) | ORT reset reproduces first frame |
| **rolling context** | official wrapper prepends previous frame's last 64 samples @16k / 32 @8k (zeros at start, reset with state); graph then sees 576 / 288 and 4 STFT frames | upstream `utils_vad.py OnnxWrapper` source; e2e ctx parity vs ORT fed `[1,576]` (2.1e-6 synthetic, 6.1e-6 jfk); same dynamic-length Rust path, no op change |

## Parity (NFR-QL-01, atol = 0.01 FP32)

Reference: onnxruntime on the upstream ONNX (weights come from the same ONNX,
so ORT is a faithful oracle). Fixtures + regeneration: `tests/parity/silero_vad/`.
Tests: `parity.rs` (in-crate, run by `cargo test -p vokra-models silero`).

Measured max abs error vs ORT (this implementation, FP32):

| check | 16 kHz | 8 kHz |
|-------|--------|-------|
| pseudo-STFT conv (T04) | 0.0 (bit-exact) | 0.0 (bit-exact) |
| magnitude (T05) | 0.0 (bit-exact) | 0.0 (bit-exact) |
| encoder output (T06) | 4.9e-4 | 1.0e-5 |
| **e2e streaming prob, raw interface (T09)** | **7.9e-8** | **3.2e-6** |
| e2e streaming prob, official context | 2.1e-6 | 3.3e-7 |
| e2e real speech `jfk-30s.wav`, official context | 6.1e-6 (max prob 1.0000; segments at 0.5 = 4, spans identical to upstream `get_speech_timestamps` on the ORT reference) | 1.2e-6 on `jfk-30s-8k.wav` (max prob 1.0000; 4 segments, spans identical to the ORT reference and within one 8 kHz frame of the rate-normalised 16 kHz spans) |

Layer intermediates (conv/magnitude/encoder) are true ORT ground truth (lifted
branch graphs). The final probability is true ORT ground truth (full model,
state carried; the official-context references feed ORT `[1,576]` / `[1,288]`
exactly as the upstream wrapper does). No intermediate was fabricated; nothing
had to be deferred to e2e for these stages.

## Conversion (both-rate, fixed 2026-07-16)

Historical gap: the original M0 `vokra-convert` Silero path stripped the
`If`-branch context and de-duped the colliding embedded parameter names, on the
assumption the two branches "recompute the same network". **They do not** — the
8 kHz and 16 kHz branches carry different weights for *all 15* tensors — so it
silently dropped the entire 16 kHz model (confirmed empirically in the
2026-07-16 real-weight eval: 16 kHz input → explicit "model has no weights for
16000 Hz").

Fixed: `crates/vokra-convert/src/models/silero.rs` now recovers each
`Constant`'s branch from its scope-qualified *output* name
(`If_0_then_branch__Inline_0__…` = 16 kHz / `If_0_else_branch__…` = 8 kHz, per
the `If(sr == 16000)` selector) and emits the corrected rate-namespaced scheme
(`sr16k.*` / `sr8k.*`, 30 tensors, name-sorted; op-scope float strays and
non-float control-flow constants are skipped with per-category counts).
`vokra-cli convert --model silero-vad` on the master `silero_vad.onnx`
(sha256 `1a153a22…`) reproduces the fixture GGUF
`tests/parity/silero_vad/silero-vad-v5.gguf` **byte-identically**
(sha256 `9de80aca…`), so converter and fixture generation are now the same
artifact. `from_gguf` still accepts the legacy single-rate (bare-name) scheme
for old GGUFs.
