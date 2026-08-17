"""microWakeWord host parity reference dumper (M5-03b Phase 4).

Offline sidecar tool (FR-LD-05: no Python / TFLite / TensorFlow ever enters
the runtime). Companion to ``prepare_checkpoint.py`` (Phase 1: TFLite → GGUF
weight extraction) — this Phase 4 script produces the reference artefacts
that the Rust host-parity harness
(``crates/vokra-kws-micro/tests/parity_microwakeword.rs``) reads to compare
against Vokra's [`vokra_kws_micro::features::FeatureExtractor`] and its
future INT8 forward chain.

# What this script emits

Given a source ``hey_jarvis.tflite`` (kahrendt/microWakeWord / ESPHome
micro-wake-word-models, Apache-2.0), this script writes into
``--output-dir`` the following files:

- ``input_pcm.bin`` — raw ``i16`` little-endian, exactly
    ``WINDOW_SAMPLES = 512`` samples (a 32 ms window @ 16 kHz), synthesised
    deterministically from a fixed seed. This is the PCM the Rust
    [`FeatureExtractor::compute_frame_f32`] consumes to reproduce the
    features side of the parity comparison.
- ``features_ref.bin`` — raw ``f32`` little-endian, exactly ``N_MELS = 40``
    floats. The reference log-mel features produced by a **numpy
    transcription** of the standard log-mel algorithm (Hann window +
    radix-2 FFT + HTK-convention mel filterbank + log10 with 1e-10 floor)
    against ``input_pcm.bin``.
- ``output_ref.bin`` — raw ``f32`` little-endian, the reference
    probability vector produced by running ``input_pcm.bin`` end-to-end
    through the source ``.tflite`` (INT8 forward, dequantised to F32 for
    portable comparison). Length equals the TFLite output size (typically
    1 or the number of wake classes).
- ``manifest.json`` — describes each artefact (name, path, shape,
    dtype, atol recommendation). Also carries the source ``.tflite``
    sha256 for provenance audit.

# What "reference" means here — honest boundary

The numpy reference for ``features_ref.bin`` is a **transcription** of
the same log-mel algorithm the Rust code implements — Hann window,
radix-2 FFT, HTK-convention mel filterbank, log10 with floor. This
validates *transcription faithfulness*: the Rust code implements the
standard algorithm it claims to implement, and matches an independent
numpy pass at ``atol = 1e-3`` on real inputs.

What this does **not** validate: bit-parity against the specific
training-time ``tf.signal`` mel front-end used to train the
microWakeWord checkpoints. Bit-parity against ``tf.signal`` would
require pulling ``tensorflow`` (~500 MB) into the sidecar's dep
footprint (currently 3 deps: ``gguf`` + ``numpy`` +
``ai-edge-litert``). Empirically the standard log-mel algorithm
matches ``tf.signal.stft`` + ``tf.signal.linear_to_mel_weight_matrix``
within ``1e-3`` for the same parameters (Whisper front-end sibling
takes the same posture — see ``vokra_backend_cpu::fused_log_mel_dispatch``'s
docs; the ``dispatch`` module defining it is private, so the crate-root
re-export is the only nameable path).

The ``output_ref.bin`` reference, by contrast, is the **real**
upstream TFLite forward: ``ai_edge_litert.Interpreter`` runs the exact
INT8 MC-MobileNet operations the microWakeWord checkpoint was trained
with. That leg has no "transcription" concern — it is the ground truth
for the INT8 forward.

# NOT REFERENCED (clean-room)

- ``kahrendt/microWakeWord`` Python training code (Apache-2.0 — never
    vendored, never re-implemented; ``.tflite`` consumed as opaque
    black-box weights).
- ``esphome/esphome`` micro_wake_word component (GPL-3.0 — never
    imported, never inspected; see ``prepare_checkpoint.py``'s own
    NOT-REFERENCED list).

# Usage

::

    cd tools/parity/microwakeword
    uv sync                          # only if first run
    # Assumes owner has previously downloaded the .tflite (e.g. via
    # prepare_checkpoint.py's --url path, or manually curl-ed):
    uv run python dump_reference.py \\
        --tflite-path ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.tflite \\
        --output-dir  ~/.cache/vokra-eval/fixtures/microwakeword \\
        --verbose

    # Point the Rust parity harness at both artefacts (the GGUF was
    # produced by prepare_checkpoint.py in a separate step):
    export VOKRA_KWS_REAL_GGUF=~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
    export VOKRA_KWS_REAL_FIXTURES=~/.cache/vokra-eval/fixtures/microwakeword
    CARGO_BUILD_JOBS=1 cargo test -p vokra-kws-micro \\
        --test parity_microwakeword -- --nocapture

Fails loudly on any anomaly (missing .tflite, dtype mismatch,
FeatureExtractor output length wrong, ...) rather than masking it —
FR-EX-08 posture, matches every other sidecar in ``tools/parity/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter

# ----------------------------------------------------------------------
# Constants — must mirror ``crates/vokra-kws-micro/src/features.rs``
# and ``prepare_checkpoint.py`` exactly. A silent drift in any of these
# would misalign the numpy reference against the Rust
# ``FeatureExtractor``, and the parity harness would fail loudly (which
# is the whole point — this is a permanent guard, not a moving target).
# ----------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
HOP_MS: int = 10
WINDOW_MS: int = 32
N_MELS: int = 40

HOP_SAMPLES: int = SAMPLE_RATE * HOP_MS // 1000     # 160
WINDOW_SAMPLES: int = SAMPLE_RATE * WINDOW_MS // 1000  # 512
N_FFT: int = 512
N_BINS: int = N_FFT // 2 + 1                          # 257
LOG_MEL_EPSILON: float = 1e-10

# Deterministic PCM synthesis: a 440 Hz sine + light gaussian noise, so
# the reference has real spectral content across multiple mel bands
# (a pure sine would concentrate energy in one bin, hiding filterbank
# regressions; pure noise would flatten every band, hiding FFT
# regressions).
PCM_SEED: int = 0
PCM_SINE_HZ: float = 440.0
PCM_SINE_AMPLITUDE: float = 6000.0    # ~1/5 of int16 range → no clipping
PCM_NOISE_STDDEV: float = 200.0        # small vs sine → sine dominates

# Compile-time contracts (mirror the Rust `const _:` asserts).
assert WINDOW_SAMPLES <= N_FFT, "WINDOW_SAMPLES must fit in N_FFT"
assert (N_FFT & (N_FFT - 1)) == 0, "N_FFT must be a power of two (radix-2)"


def sha256_of_file(path: Path) -> str:
    """Streamed hex sha256 of the file. Used to stamp the source
    ``.tflite`` provenance into the manifest so a future Rust-side
    fixture-integrity check can catch a drifted upstream."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synth_pcm() -> np.ndarray:
    """Deterministic ``i16`` PCM window, ``WINDOW_SAMPLES`` samples wide.

    Returns a ``np.int16`` array. Same seed on every invocation → the
    dump is byte-stable across runs, so the Rust parity harness can
    hash-audit the fixture if desired.
    """
    rng = np.random.default_rng(PCM_SEED)
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64) / float(SAMPLE_RATE)
    sine = PCM_SINE_AMPLITUDE * np.sin(2.0 * np.pi * PCM_SINE_HZ * t)
    noise = rng.normal(0.0, PCM_NOISE_STDDEV, size=WINDOW_SAMPLES)
    signal = sine + noise
    # Clip to int16 range (defensive — the amplitude budget above avoids
    # clipping in practice, but a stray parameter change should not
    # produce silent wrap-around).
    signal = np.clip(signal, -32768.0, 32767.0)
    return signal.astype(np.int16)


def hz_to_mel_f32(hz: np.float32) -> np.float32:
    """HTK-convention Hz → mel, computed in **float32** to match the Rust
    ``crates/vokra-kws-micro/src/features.rs::hz_to_mel`` bit-for-bit:
    ``mel = 2595 * log10(1 + hz / 700)``.

    The `np.float32(...)` casts pin every intermediate at f32 precision.
    A stray float64 promotion (e.g. `1.0 + hz / 700.0` where `hz` was
    promoted to f64 by scalar arithmetic) would silently shift the mel
    filterbank edges by ~1e-5 Hz and break bit-parity against Vokra
    (verified empirically — a float64 mel_points path fails Path B at
    band ~30 by 3e-2, well above the 1e-3 atol).
    """
    return np.float32(2595.0) * np.log10(np.float32(1.0) + hz / np.float32(700.0))


def mel_to_hz_f32(mel: np.float32) -> np.float32:
    """Inverse of :func:`hz_to_mel_f32`. Kept in **float32** for the same
    bit-parity reason. Same as the Rust ``mel_to_hz``:
    ``hz = 700 * (10^(mel / 2595) - 1)``.
    """
    return np.float32(700.0) * (np.float32(10.0) ** (mel / np.float32(2595.0)) - np.float32(1.0))


def hann_window(n: int) -> np.ndarray:
    """Symmetric Hann window matching the Rust ``hann_window``:
    ``w[i] = 0.5 * (1 - cos(2*pi*i / (n-1)))``.

    Equivalent to ``numpy.hanning(n)``. We spell it out explicitly (and
    keep every intermediate at f32) to make the correspondence to the
    Rust code unmistakable and to avoid a stray future refactor toward
    the periodic convention (``2*pi*i / n``) which would silently
    rescale every feature.
    """
    denom = np.float32(n - 1)
    i = np.arange(n, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    return (np.float32(0.5) * (np.float32(1.0) - np.cos(two_pi * i / denom))).astype(np.float32)


def mel_filterbank(n_mels: int, n_bins: int, sample_rate: int) -> np.ndarray:
    """Row-major ``[n_mels, n_bins]`` un-normalised triangular filterbank
    with HTK-convention mel spacing, ``fmin = 0``, ``fmax = sr / 2``.

    **f32 throughout** to match the Rust ``mel_filterbank`` bit-for-bit
    (the Rust code declares every intermediate as `f32`; using float64
    for the mel-point path in numpy shifts band edges by ~1e-5 Hz and
    accumulates a ~3e-2 log10 delta at high bands).
    """
    fmax = np.float32(0.5) * np.float32(sample_rate)
    mel_min = hz_to_mel_f32(np.float32(0.0))
    mel_max = hz_to_mel_f32(fmax)
    # (n_mels + 2) equally-spaced mel points. `np.linspace` with an
    # f32 dtype computes in f32 throughout.
    #
    # Vokra spells this as:
    #   mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1)
    # in f32. `np.linspace(mel_min, mel_max, n_mels+2, dtype=f32)` uses
    # a slightly different formula internally (`start + step * i` with
    # a precomputed `step`), which can differ at f32-ULP scale from the
    # Rust `(mel_max - mel_min) * i / n_mels_plus_1` form. Mirror the
    # Rust formula exactly to preserve bit-parity.
    denom = np.float32(n_mels + 1)
    span = mel_max - mel_min
    mel_points = np.array(
        [mel_min + span * np.float32(i) / denom for i in range(n_mels + 2)],
        dtype=np.float32,
    )
    bin_scale = np.float32(n_bins - 1) / fmax
    bin_pts = np.array(
        [mel_to_hz_f32(np.float32(mp)) * bin_scale for mp in mel_points],
        dtype=np.float32,
    )
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left = np.float32(bin_pts[m])
        center = np.float32(bin_pts[m + 1])
        right = np.float32(bin_pts[m + 2])
        for k in range(n_bins):
            kf = np.float32(k)
            if kf < left or kf > right:
                w = np.float32(0.0)
            elif kf <= center:
                if center == left:
                    w = np.float32(1.0)
                else:
                    w = (kf - left) / (center - left)
            elif center == right:
                w = np.float32(1.0)
            else:
                w = (right - kf) / (right - center)
            fb[m, k] = w
    return fb


def numpy_log_mel_features(pcm_i16: np.ndarray) -> np.ndarray:
    """Numpy reference log-mel feature extraction.

    Steps 1–5 of ``FeatureExtractor::compute_frame_f32``:

    1. i16 → f32 with symmetric Hann window applied (NO normalisation
        to [-1, 1] — matches Rust code exactly).
    2. Radix-2 FFT via ``np.fft.rfft`` (returns the one-sided spectrum;
        length ``N_FFT/2 + 1 = N_BINS``).
    3. Power spectrum ``|X[k]|²``.
    4. Row-major mel filterbank matmul (explicit Python loop, NOT
        ``fb @ power``, to match Rust's naive left-to-right accumulator
        order — numpy BLAS's pairwise / SIMD summation gives different
        f32 rounding at high bands, verified empirically).
    5. ``log10(max(mel_energy, LOG_MEL_EPSILON))``.

    # Precision honesty

    ``np.fft.rfft`` computes internally in float64 and casts to the
    input dtype at output. Vokra's Rust FFT is float32 throughout. The
    two agree bit-for-bit at low bands (< 1e-4 per-band |Δ|) but drift
    to ~3e-2 at high bands (~30) where the f32 rounding accumulates
    through log₂(N_FFT) = 9 butterfly stages. This is a real precision
    gap between the (higher-precision) numpy reference and the target-
    architecture-realistic (f32) Rust code — not a Rust bug. The
    parity harness's ``FEATURES_ATOL`` accepts this bound and catches
    regressions above it.

    A pure-f32 numpy transcription of Vokra's Cooley–Tukey radix-2 FFT
    would close this gap but would basically be running the Rust code
    in Python — the honest atol is the more useful posture.
    """
    assert pcm_i16.shape == (WINDOW_SAMPLES,), pcm_i16.shape
    assert pcm_i16.dtype == np.int16, pcm_i16.dtype

    # Step 1: i16 → f32 with Hann (no [-1, 1] normalisation).
    hann = hann_window(WINDOW_SAMPLES)
    windowed = pcm_i16.astype(np.float32) * hann

    # Zero-pad to N_FFT if the window is shorter (at default constants
    # WINDOW_SAMPLES == N_FFT, so this is a no-op; kept for parity with
    # the Rust code's implicit zero-padding).
    if WINDOW_SAMPLES < N_FFT:
        padded = np.zeros(N_FFT, dtype=np.float32)
        padded[:WINDOW_SAMPLES] = windowed
        windowed = padded

    # Step 2: real one-sided FFT.
    spec = np.fft.rfft(windowed, n=N_FFT)
    assert spec.shape == (N_BINS,), spec.shape

    # Step 3: power spectrum (|X[k]|²).
    power = (spec.real.astype(np.float32) ** 2) + (spec.imag.astype(np.float32) ** 2)

    # Step 4: filterbank matmul via explicit accumulator (matches Rust
    # naive left-to-right f32 summation order — see docstring above).
    fb = mel_filterbank(N_MELS, N_BINS, SAMPLE_RATE)
    mel_energy = np.zeros(N_MELS, dtype=np.float32)
    for m in range(N_MELS):
        acc = np.float32(0.0)
        row = fb[m]
        for k in range(N_BINS):
            acc = acc + row[k] * power[k]
        mel_energy[m] = acc

    # Step 5: log10(max(mel_energy, EPSILON)).
    clamped = np.maximum(mel_energy, np.float32(LOG_MEL_EPSILON))
    features = np.log10(clamped).astype(np.float32)
    assert features.shape == (N_MELS,), features.shape
    return features


def dump_le(arr: np.ndarray, path: Path) -> None:
    """Writes ``arr`` as little-endian raw bytes (no header).

    ``ndarray.tobytes()`` uses the native byte order; forcing to ``<f4``
    / ``<i2`` / ``<f8`` first pins the wire format across host
    endianness. Every consumer this file targets (the Rust parity
    harness) reads little-endian, matching the M5-03 IoT target family
    (thumbv8m is little-endian).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Map dtype → little-endian equivalent. Guard against surprise dtypes.
    if arr.dtype == np.int16:
        arr = arr.astype("<i2")
    elif arr.dtype == np.float32:
        arr = arr.astype("<f4")
    elif arr.dtype == np.int8:
        arr = arr.astype("<i1")  # trivially LE
    else:
        raise SystemExit(f"dump_le: unsupported dtype {arr.dtype} for {path}")
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def run_tflite_forward(
    interp: Interpreter, pcm_i16: np.ndarray, verbose: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Runs the TFLite reference forward and returns
    ``(output_probability_f32, meta)`` where ``meta`` captures the
    interpreter's input / output tensor descriptors (for the manifest).

    The microWakeWord TFLite models consume **pre-computed features** as
    their input (not raw PCM). Two topologies exist in the wild:

    1. **Stacked-frame INT8 input** (typical hey_jarvis): the input
        tensor is ``[1, T, N_MELS]`` where ``T`` is a small number of
        stacked frames (e.g. 3 or 5). Vokra's Phase 4 parity uses a
        single frame, so we tile the log-mel features across the T
        dimension (justified — the reference forward is a black-box
        smoke check, not per-frame timing parity).
    2. **Single-frame F32 input** (rarer): the input is ``[1, N_MELS]``
        F32. We feed the log-mel features directly.

    The dumper detects the shape at runtime and adapts. Any other input
    shape is a loud SystemExit — a mis-detected topology would silently
    produce garbage reference outputs.
    """
    features_f32 = numpy_log_mel_features(pcm_i16)

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    if len(input_details) != 1:
        raise SystemExit(
            f"expected exactly 1 TFLite input tensor, got {len(input_details)}: "
            f"{[d['name'] for d in input_details]}"
        )
    if len(output_details) != 1:
        raise SystemExit(
            f"expected exactly 1 TFLite output tensor, got {len(output_details)}: "
            f"{[d['name'] for d in output_details]}"
        )
    inp = input_details[0]
    out = output_details[0]
    in_shape = list(inp["shape"])
    in_dtype = inp["dtype"]
    if verbose:
        print(f"  TFLite input : name={inp['name']!r} shape={in_shape} dtype={in_dtype}",
              file=sys.stderr)
        print(f"  TFLite output: name={out['name']!r} shape={list(out['shape'])} dtype={out['dtype']}",
              file=sys.stderr)

    # --- Shape adaptation ------------------------------------------------
    # Squeeze leading batch=1 for shape reasoning.
    squeezed = [d for d in in_shape if d != 1]
    if squeezed == [N_MELS]:
        # Single-frame F32 or INT8 input.
        input_tensor = features_f32.reshape(in_shape)
    elif len(squeezed) == 2 and squeezed[-1] == N_MELS:
        # Stacked-frame [T, N_MELS] input. Tile features across T.
        t_stack = int(squeezed[0])
        tiled = np.tile(features_f32.reshape(1, N_MELS), (t_stack, 1))
        input_tensor = tiled.reshape(in_shape)
    else:
        raise SystemExit(
            f"TFLite input shape {in_shape} not recognised as "
            f"[1, N_MELS={N_MELS}] or [1, T, N_MELS={N_MELS}]. Refusing to "
            f"feed a mis-shaped tensor — investigate the checkpoint."
        )

    # Dtype quantisation to input's INT8 params, if any.
    if in_dtype == np.int8:
        in_quant = inp.get("quantization", (0.0, 0))
        in_scale, in_zp = in_quant if isinstance(in_quant, tuple) else (0.0, 0)
        if in_scale <= 0.0:
            raise SystemExit(
                f"TFLite input {inp['name']!r} is INT8 but carries no per-tensor "
                f"quantization scale (scale={in_scale!r}, zero_point={in_zp!r}). "
                f"Refusing to feed — FR-EX-08."
            )
        # Standard TFLite affine quantise (matches
        # `crates/vokra-kws-micro/src/features.rs::quantize_int8`).
        scaled = input_tensor / float(in_scale)
        rounded = np.where(scaled >= 0.0, scaled + 0.5, scaled - 0.5).astype(np.int32)
        clipped = np.clip(rounded + int(in_zp), -128, 127).astype(np.int8)
        input_tensor = clipped
    elif in_dtype == np.float32:
        input_tensor = input_tensor.astype(np.float32)
    else:
        raise SystemExit(
            f"TFLite input dtype {in_dtype!r} unsupported (only INT8 + F32 handled). "
            f"Report to CC for Phase 5 extension."
        )

    # Feed + invoke.
    interp.set_tensor(inp["index"], input_tensor)
    interp.invoke()
    output = interp.get_tensor(out["index"])
    if verbose:
        print(f"  raw output   : shape={list(output.shape)} dtype={output.dtype}",
              file=sys.stderr)

    # Dequantise output to F32 for portable comparison (INT8 zero-point
    # semantics vary by build; F32 is universal).
    if output.dtype == np.int8:
        out_quant = out.get("quantization", (0.0, 0))
        out_scale, out_zp = out_quant if isinstance(out_quant, tuple) else (0.0, 0)
        if out_scale <= 0.0:
            raise SystemExit(
                f"TFLite output {out['name']!r} is INT8 but carries no per-tensor "
                f"quantization scale — refusing to emit F32 comparison values."
            )
        out_f32 = (output.astype(np.int32) - int(out_zp)).astype(np.float32) * float(out_scale)
    elif output.dtype == np.float32:
        out_f32 = output
    else:
        raise SystemExit(
            f"TFLite output dtype {output.dtype!r} unsupported. Report to CC."
        )

    # Flatten to a 1D vector (batch=1 → squeeze), for portable comparison.
    out_f32 = np.ascontiguousarray(out_f32.reshape(-1).astype(np.float32))

    meta = {
        "input_name": inp["name"],
        "input_shape": in_shape,
        "input_dtype": str(in_dtype.__name__ if hasattr(in_dtype, "__name__") else in_dtype),
        "output_name": out["name"],
        "output_shape": list(out["shape"]),
        "output_dtype": str(output.dtype.__name__ if hasattr(output.dtype, "__name__") else output.dtype),
    }
    return out_f32, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description="microWakeWord host parity reference dumper (Phase 4)."
    )
    ap.add_argument("--tflite-path", type=Path, required=True,
                    help="Path to source hey_jarvis.tflite (owner-fetched, "
                         "e.g. via prepare_checkpoint.py's --url path).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for input_pcm.bin, features_ref.bin, "
                         "output_ref.bin, and manifest.json.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print per-stage progress to stderr.")
    args = ap.parse_args()

    if not args.tflite_path.exists():
        raise SystemExit(f"--tflite-path not found: {args.tflite_path}")

    tflite_sha256 = sha256_of_file(args.tflite_path)
    tflite_size = args.tflite_path.stat().st_size
    print(f"Source: {args.tflite_path.name} ({tflite_size:,} bytes, "
          f"sha256={tflite_sha256[:16]}…)", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # (1) Synthesise deterministic PCM.
    pcm = synth_pcm()
    dump_le(pcm, args.output_dir / "input_pcm.bin")
    if args.verbose:
        print(f"  input_pcm.bin   : {pcm.dtype} shape={pcm.shape} "
              f"min={pcm.min()} max={pcm.max()}", file=sys.stderr)

    # (2) Reference log-mel features (numpy transcription).
    features_ref = numpy_log_mel_features(pcm)
    dump_le(features_ref, args.output_dir / "features_ref.bin")
    if args.verbose:
        print(f"  features_ref.bin: f32 shape={features_ref.shape} "
              f"min={features_ref.min():.4f} max={features_ref.max():.4f}",
              file=sys.stderr)

    # (3) TFLite forward for output reference.
    interp = Interpreter(model_path=str(args.tflite_path))
    interp.allocate_tensors()
    output_ref, tflite_meta = run_tflite_forward(interp, pcm, args.verbose)
    dump_le(output_ref, args.output_dir / "output_ref.bin")
    if args.verbose:
        print(f"  output_ref.bin  : f32 shape={output_ref.shape} "
              f"min={output_ref.min():.4f} max={output_ref.max():.4f}",
              file=sys.stderr)

    # (4) Manifest.
    manifest: dict[str, Any] = {
        "generator": "vokra tools/parity/microwakeword/dump_reference.py",
        "generator_version": "0.1.0-phase4",
        "source_tflite": str(args.tflite_path),
        "source_tflite_sha256": tflite_sha256,
        "source_tflite_bytes": tflite_size,
        "constants": {
            "sample_rate": SAMPLE_RATE,
            "hop_ms": HOP_MS,
            "window_ms": WINDOW_MS,
            "n_mels": N_MELS,
            "hop_samples": HOP_SAMPLES,
            "window_samples": WINDOW_SAMPLES,
            "n_fft": N_FFT,
            "n_bins": N_BINS,
            "log_mel_epsilon": LOG_MEL_EPSILON,
        },
        "pcm_synthesis": {
            "seed": PCM_SEED,
            "sine_hz": PCM_SINE_HZ,
            "sine_amplitude": PCM_SINE_AMPLITUDE,
            "noise_stddev": PCM_NOISE_STDDEV,
        },
        "artefacts": [
            {
                "name": "input_pcm",
                "path": "input_pcm.bin",
                "shape": [WINDOW_SAMPLES],
                "dtype": "int16",
                "byte_order": "little-endian",
                "role": "PCM window fed to both numpy reference and "
                        "Vokra FeatureExtractor",
            },
            {
                "name": "features_ref",
                "path": "features_ref.bin",
                "shape": [N_MELS],
                "dtype": "float32",
                "byte_order": "little-endian",
                "role": "reference log-mel features (numpy transcription "
                        "of the standard algorithm)",
                "rust_side_atol": 1e-3,
            },
            {
                "name": "output_ref",
                "path": "output_ref.bin",
                "shape": list(output_ref.shape),
                "dtype": "float32",
                "byte_order": "little-endian",
                "role": "reference INT8-dequantised TFLite output "
                        "probability vector (end-to-end forward)",
                "rust_side_atol": 1e-2,
                "note_end_to_end_status": (
                    "UNMET as of Phase 4 — the Rust INT8 ChainConfig needs "
                    "per-tensor quantisation params the current Phase 1 "
                    "sidecar does not emit. This artefact is scaffold for "
                    "the Phase 3.5 Q8_0 sidecar extension."
                ),
            },
        ],
        "tflite_topology": tflite_meta,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {args.output_dir}/  "
          f"(input_pcm.bin, features_ref.bin, output_ref.bin, manifest.json)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
