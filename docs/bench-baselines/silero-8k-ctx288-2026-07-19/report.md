# Silero VAD — 8 kHz (ctx288) real-speech evaluation (2026-07-19)

**cc-25** from the M4-residual audit. The official rolling-context fix
(`7639dc0`) closed three of the four `{16 kHz, 8 kHz} x {synthetic, real}`
quadrants: both rates were verified on synthetic audio, and 16 kHz was verified
on real speech (`jfk-30s.wav`). **8 kHz x real speech was never measured.** This
report measures it and lands a committed regression test so it stays measured.

Result: **the 8 kHz ctx288 path is correct.** Vokra reproduces onnxruntime to
**1.2e-6** on real speech and detects the identical speech segments. Three
findings came out of the run (§4).

## 1. Method

### Producing a legitimate 8 kHz signal

Halving the sample rate halves Nyquist (8 kHz -> 4 kHz), so everything above
4 kHz must be **removed before** decimation. Plain `x[::2]` instead folds it
back into the audible band and yields an 8 kHz *file* that is not an 8 kHz
*signal*. The fixture path uses `gen_reference.py::halfband_decimate`: a 41-tap
linear-phase Kaiser-windowed sinc lowpass (cutoff 0.5 x Nyquist, beta 5.0,
DC-normalised), symmetric zero-padding to cancel group delay, then 2:1
decimation. It is implemented in numpy so the parity tooling gains no new
dependency (`tests/parity/parity-requirements.txt` is unchanged).

Evidence that this is a real 8 kHz signal, not a decimated artifact:

| check | jfk-30s | LibriSpeech 1272-128104-0002 |
|---|---|---|
| source power above 4 kHz (the energy that would alias) | 0.006 % | **18.80 %** |
| agreement with `scipy.signal.resample_poly(x, 1, 2)` | **3.3e-16** | **2.2e-16** |
| passband 50-3500 Hz gain, median (p05..p95) | +0.005 dB (-0.013..+0.015) | +0.004 dB (-0.024..+0.028) |
| max abs difference from naive `x[::2]` | 0.055 | **0.412** |

The numpy filter is bit-equal to scipy's default polyphase design at double
precision, the passband is flat to hundredths of a dB, and on the wideband
LibriSpeech clip the output differs from naive decimation by 0.41 full-scale —
the anti-aliasing filter is doing substantial, necessary work there.

Stated precisely rather than absolutely, the 41-tap response is:

| band | worst-case gain |
|---|---|
| 0-3500 Hz (passband) | +0.02 / -0.26 dB |
| 4000 Hz | -6.02 dB (the halfband half-power point) |
| 4000-4400 Hz | -6.0 dB |
| 4400-5000 Hz | -22.6 dB |
| above 5000 Hz | -57.3 dB or better |

So rejection is **not** uniformly deep: a narrow 4.0-4.4 kHz sliver folds back
to 3.6-4.0 kHz attenuated by only 6-22 dB. That is inherent to a halfband
design of this length and is exactly what `scipy.signal.resample_poly` does by
default — which is the point of the 3.3e-16 equivalence above: the fixture is
resampled the way the standard tool resamples, not by an ad-hoc filter chosen
here. On jfk the residue is negligible (0.006 % of source power lies above
4 kHz at all). The claim this supports is "a standard, correctly anti-aliased
8 kHz signal", not "an alias-free one".

Honest caveat: on **jfk** the filter barely matters, because that clip carries
only 0.006 % of its power above 4 kHz (an already band-limited historical
recording). The wideband LibriSpeech clip was added specifically so the
resampling claim rests on a case where the filter is load-bearing.

### Running both engines

Both engines score **byte-identical samples**: the 8 kHz WAV is written as mono
PCM16 and read back before either engine sees it, so the reference is computed
on the quantised signal the Rust reader gets, not on pre-quantisation floats.

- **Reference**: onnxruntime 1.19.2, CPU EP, official wrapper semantics
  (`utils_vad.py OnnxWrapper`) — a rolling 32-sample audio context prepended to
  every 256-sample frame, graph fed `[1, 288]`, LSTM state carried, trailing
  partial frame dropped.
- **Vokra**: `SileroVadV5::open(...).open_stream()` (default
  `ContextMode::Official`), same clip, same frame count.
- **Segments**: upstream `get_speech_timestamps` at defaults (threshold 0.5,
  neg 0.35, min_speech 250 ms, min_silence 100 ms, pad 30 ms), applied to both
  probability streams.

## 2. Result — 8 kHz real speech matches ORT

| leg | weights | frames | max abs delta vs ORT | max prob | segments | speech |
|---|---|---|---|---|---|---|
| jfk-30s-8k | master | 343 | **1.192e-6** | 1.0000 | **4/4 identical** | 7.98 s |
| LibriSpeech 0002 @ 8k | master | 390 | **2.488e-6** | 1.0000 | **2/2 identical** | 11.00 s |
| jfk-30s-8k | v5.0 release | 343 | **1.192e-6** | 0.9954 | **4/4 identical** | 6.90 s |

"segments identical" means Vokra's probability stream and ORT's produce exactly
the same spans, and that those spans equal what upstream `get_speech_timestamps`
yields on the ORT reference. Every leg is well inside `atol = 0.01` (NFR-QL-01).

### Cross-rate agreement

Same utterance at both rates, spans rate-normalised (16 kHz / 2):

| segment | 8 kHz | 16 kHz / 2 | delta |
|---|---|---|---|
| 1 | (2320, 18416) | (2576, 18160) | 256 / 256 samples (1 frame) |
| 2 | (26128, 35568) | (26128, 35568) | **exact** |
| 3 | (43024, 61424) | (43024, 61424) | **exact** |
| 4 | (65296, 85232) | (65296, 84976) | 0 / 256 samples |

4/4 segments at both rates, two spans exactly equal, the rest within one 8 kHz
frame (32 ms). Total detected speech 7.98 s @ 8k vs 7.89 s @ 16k. The 8 kHz
branch is a **separate weight set** (`sr8k.*`, selected by the ONNX's
`If(sr == 16000)`), so this exercises weights and code the 16 kHz leg never
touches.

## 3. What landed

- `tests/parity/silero_vad/jfk-30s-8k.wav` — the decimated clip (PCM16 mono,
  88 000 samples, sha256 `5c8c1ad4...`), derived from the already-committed
  `tests/fixtures/audio/jfk-30s.wav`. No new third-party audio enters the repo.
- `tests/parity/silero_vad/probs_jfk8k_ctx.txt` — the ctx288 ORT reference.
- `gen_reference.py` — `halfband_decimate` + `write_wav_pcm16` + the 8 kHz
  block. **All pre-existing fixtures regenerate byte-for-byte**, which also
  re-confirms the master-ONNX provenance recorded in that directory's README.
- `crates/vokra-models/src/silero_vad/parity.rs` —
  `vad_real_speech_jfk_8k_official_context`.
- `.gitattributes` — LFS-free pin for `tests/parity/silero_vad/*.{wav,gguf}`
  (that directory's binary fixtures were previously unpinned).

The LibriSpeech and v5.0 legs are **measurement only** — reproducible with
`measure.py` here, but not committed as fixtures (LibriSpeech audio is
third-party CC-BY-4.0 content; adding it is a licence-audit decision, not a
test-coverage one).

## 4. Findings

### 4.1 The raw interface fails *differently* at 8 kHz (docs were 16 kHz-specific)

The known 2026-07-16 P1 is that the raw bare-frame interface collapses on real
speech: max prob **0.0037** at 16 kHz, zero detections. At 8 kHz the raw
interface instead peaks at **0.646** — it *does* cross the 0.5 threshold — yet
still never sustains speech long enough to satisfy `min_speech` (250 ms), so it
still yields **zero** segments against the official interface's four.

The product-level conclusion is unchanged (the official context is required),
but the symptom is not: a consumer who lowered `min_speech` or thresholded per
frame would get **spurious detections at 8 kHz** where 16 kHz merely gives
silence. `SPEC.md` and the fixture README said "collapses (max prob 0.0037)"
without qualifying the rate; both now record the 8 kHz behaviour.

The test asserts only the product-level fact (zero segments) and *prints* the
peak rather than thresholding it — pinning 0.646 with an invented bound would be
a tuned-to-pass assertion, not a measurement.

### 4.2 `silero-vad-v5-bothrate.gguf` is built from the v5.0 release ONNX, not master

The lane brief named `~/.cache/vokra-eval/gguf/silero-vad-v5-bothrate.gguf` as
the file to test. It is **not** the model the repo's oracle describes.
Rebuilding a GGUF from each upstream ONNX with the committed `gen_reference.py`
writer identifies both cache files exactly:

| ONNX | sha256 | -> GGUF sha256 | cache file |
|---|---|---|---|
| `silero_vad_master.onnx` | `1a153a22...` | `9de80aca...` | `silero-vad-v5-master.gguf` **= the committed fixture** |
| `silero_vad.onnx` (v5.0 tag) | `6b99cbfd...` | `f24a814c...` | `silero-vad-v5-bothrate.gguf` |

The name encodes the *both-sample-rate* converter fix, which it has — but not
its *weight provenance*, which is the v5.0 release tag. The repo README already
warns that the v5.0 ONNX "carries different weight values"; this quantifies it:
the two weight sets differ by up to **0.811** in per-frame probability on the
same clip — a factor of ~7e5 over the ~1e-6 engine-vs-ORT agreement.
Comparing `-bothrate.gguf` against a master-ONNX reference would have produced a
spurious ~0.8 "parity failure".

This report therefore ran **both** pairings correctly matched, which is why §2
has two weight columns — and it is why the v5.0 leg is *useful*: it
independently confirms the GGUF conversion is faithful for a second, unrelated
weight set. Operational follow-up:
`docs/bench-baselines/eval-cache-artifacts-2026-07-19/`.

### 4.3 No disagreement between Vokra and ORT was found

Both engines agree to ~1e-6 on every leg and produce identical segments. There
was nothing to tune, and nothing was tuned.

## 5. Reproducing

```sh
# fixtures (committed) — regenerates every fixture in the directory
tests/parity/silero_vad/gen_reference.py <path to silero_vad_master.onnx>
cargo test --release -p vokra-models --lib silero_vad

# the measurement legs in section 2/4 (needs the local eval cache + an ORT venv)
python measure.py <output dir>
```

Environment: Apple M1 iMac, macOS arm64; onnxruntime 1.19.2 / numpy 2.0.2 /
scipy 1.13.1 (scipy used only for the cross-check and spectra in section 1,
never on the fixture path).
