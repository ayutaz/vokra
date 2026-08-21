#!/usr/bin/env python3
"""Generate the DeepFilterNet3 parity reference audio bundle (M4-20 T17 /
M5 gap A-2, landed 2026-07-29).

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the runtime).
Produces the ``clean_48k.f32`` / ``noisy_48k.f32`` triplet that
``parity_denoise_dfn3::dfn3_real_weight_stage_and_output_parity`` demands as
its input, and — when ``--enhance`` is passed — the ``enhanced_upstream.f32``
that the same harness compares Vokra's output against.

# Why this file exists

Prior to this script the recipe lived only in
``docs/bench-baselines/m1-real-weight-eval-2026-07-16/agent-results-campaign2.json``
(``dfn3-real`` leg) and in the M4-20 owner-local
``~/.cache/vokra-eval/out/dfn3-real/prep_noisy.py``.
``docs/handoff/parity-deepfilternet3-real.md`` §Phase B, Path 1 spelled out
the fix: land ``tools/parity/dfn3_prep_noisy.py`` here so the parity CI can
call it inline and drop the ``VOKRA_DFN3_DATA_URL`` gate. This is that
script; the CI wiring change is a follow-up in the same land.

The reproduced recipe (bit-exact vs the 2026-07-17 measured baseline that
sets the harness's ``snr_noisy = 5.002 ± 0.01`` and ``snr_up = 14.768 ± 0.01``
bounds):

1. Read ``--clean-source`` (default ``tests/fixtures/audio/jfk-30s.wav`` —
   actually 11.0 s, mono, PCM16 @ 16 kHz).
2. Convert to float32 mono, then sinc-resample 16 kHz → 48 kHz with
   ``torchaudio.functional.resample`` from the pinned torch/torchaudio oracle.
   This preserves the 2026-07-17 baseline path; replacing it with scipy's
   different polyphase kernel changes the clean/noisy bytes and the downstream
   quality anchor.
3. Draw an additive white-noise vector from
   ``np.random.default_rng(20260717)`` — seed matches the campaign-2
   measured run byte-for-byte.
4. Scale the float64 noise from raw full-signal powers so the construction
   SNR = 5.000 dB. The Rust harness separately reports zero-mean SI-SNR
   (5.002 dB); conflating those two definitions changes the fixture bytes.
5. Write ``clean_48k.f32`` (the 48 kHz clean signal) and ``noisy_48k.f32``
   (clean + scaled noise) as raw little-endian float32 — the same layout
   ``read_f32`` in ``parity_denoise_dfn3.rs`` expects.

Optionally (``--enhance``) also runs the upstream ``deepfilternet`` package
from the checked-in ``tools/parity/dfn3`` uv lock over ``noisy_48k`` to produce
``enhanced_upstream.f32``. Kept opt-in because the ``deepfilternet`` install
pulls torch + torchaudio (~200 MB of CPU wheels); CI callers that only need the
input tensors skip it.

Fails loudly on any anomaly (missing file, wrong sample rate, corrupt
tensor) rather than masking it — FR-EX-08 posture, matches
``dfn3_prepare_checkpoint.py`` sibling.

# NOT REFERENCED (clean-room)

- ``github.com/Rikorose/DeepFilterNet`` code (dual MIT/Apache-2.0 — we call
  the installed ``deepfilternet`` package as a black box in ``--enhance``
  mode, we do not vendor or re-implement any of its Rust / Python source).
  The recipe above is derived from the ``analyze.py`` in the M4-20 leg
  bench report + the ``si_snr`` methodology the harness itself uses; no
  DFN3 source code is transliterated here.

# Usage

::

    uv sync --project tools/parity/dfn3 --frozen --python 3.11

    # inputs only (fast — used by the parity CI):
    uv run --project tools/parity/dfn3 --frozen python \\
        tools/parity/dfn3_prep_noisy.py \\
        --clean-source tests/fixtures/audio/jfk-30s.wav \\
        --out-dir ${RUNNER_TEMP}/dfn3-refdata

    # inputs + upstream enhancement (owner-local, closes Phase B fully):
    uv run --project tools/parity/dfn3 --frozen python \\
        tools/parity/dfn3_prep_noisy.py \\
        --clean-source tests/fixtures/audio/jfk-30s.wav \\
        --out-dir ${RUNNER_TEMP}/dfn3-refdata \\
        --enhance --model-dir /path/to/DeepFilterNet3

Then ``tools/parity/dfn3_dump_reference.py`` produces ``taps/*.f32`` on top
of the ``noisy_48k.f32`` this script writes, and the parity harness runs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

LOG_PREFIX = "[dfn3-prep-noisy]"

# Seed reproduces the 2026-07-17 measured run byte-for-byte. Bumping this
# breaks the harness's `snr_noisy = 5.002 ± 0.01` bound and every per-stage
# tap tolerance downstream — do not touch without re-calibrating all bounds.
DEFAULT_NOISE_SEED = 20260717

# Target signal-to-noise ratio in dB for the additive white noise.
# 5.000 dB is the campaign-2 measurement anchor.
DEFAULT_SNR_DB = 5.0

# Fixed output sample rate — DFN3 was trained at 48 kHz and the parity taps
# reference this rate. The input file is expected at 16 kHz (the JFK
# fixture); other rates fall through to a warning + explicit exit so a
# silent resample doesn't quietly change the SNR calibration.
OUTPUT_SR = 48_000


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr)


def read_wav_mono_f32(path: Path):
    """Read a PCM WAV as float32 mono. Returns (samples, sample_rate)."""
    import soundfile as sf  # deferred: soundfile is not in the stdlib

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        # Down-mix to mono the same way the fixture's own preparation did
        # (equal-weight average across channels — the JFK fixture ships mono
        # already, so this branch is only exercised on user-supplied inputs).
        import numpy as np

        data = data.mean(axis=1, dtype="float32")
    return data, int(sr)


def resample_16k_to_48k(samples):
    """Run the locked torchaudio sinc resampler used by the baseline."""
    import numpy as np
    import torch
    import torchaudio.functional as audio_functional

    tensor = torch.from_numpy(np.asarray(samples, dtype="float32"))
    return audio_functional.resample(tensor, 16_000, OUTPUT_SR).numpy()


def scale_noise_to_snr(clean, noise, snr_db: float):
    """Scale float64 noise using the baseline's raw full-signal powers."""
    import numpy as np

    clean64 = np.asarray(clean, dtype="float64")
    noise64 = np.asarray(noise, dtype="float64")
    p_clean = float((clean64**2).mean())
    p_noise = float((noise64**2).mean())
    if p_noise <= 0.0:
        raise SystemExit(
            f"{LOG_PREFIX} noise vector has zero power — cannot scale to SNR"
        )
    target_p_noise = p_clean / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(target_p_noise / p_noise)
    return noise64 * scale


def write_f32(path: Path, samples) -> None:
    """Write a raw little-endian float32 file + record its sha256 in stderr."""
    import numpy as np

    arr = np.asarray(samples, dtype="<f4")  # little-endian float32, explicit
    path.write_bytes(arr.tobytes())
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    log(f"wrote {path}  ({arr.nbytes} B, {arr.size} f32 samples, sha256 {digest[:16]}…)")


def measure_snr_db(clean, noisy) -> float:
    """Raw-power SNR check matching the fixture construction definition."""
    import numpy as np

    clean64 = np.asarray(clean, dtype="float64")
    diff64 = np.asarray(noisy, dtype="float64") - clean64
    p_clean = float((clean64**2).mean())
    p_noise = float((diff64**2).mean())
    if p_noise <= 0.0:
        return float("inf")
    return 10.0 * math.log10(p_clean / p_noise)


def run_upstream_enhance(noisy, out_path: Path, model_dir: Path) -> None:
    """Run the upstream ``deepfilternet`` package over ``noisy_48k`` and
    write ``enhanced_upstream.f32``. Opt-in (``--enhance``).

    Fails loudly if ``deepfilternet`` / ``torch`` / ``torchaudio`` are not
    installed — never falls back to an approximation.
    """
    try:
        import numpy as np
        import torch

        from dfn3_torchaudio_compat import install_deepfilternet_import_compat

        install_deepfilternet_import_compat()
        from df.enhance import enhance, init_df, load_audio  # noqa: F401 — presence check
    except ImportError as e:
        raise SystemExit(
            f"{LOG_PREFIX} --enhance requires torch + torchaudio + deepfilternet in the "
            f"uv environment (uv sync --project tools/parity/dfn3 --frozen) — {e}"
        ) from e

    from df.enhance import enhance as df_enhance, init_df as df_init

    log(f"initializing upstream DeepFilterNet from {model_dir}")
    model, df_state, _ = df_init(str(model_dir))

    # df.enhance wants (channels, samples) at df_state.sr(); noisy is
    # already at 48 kHz mono.
    if df_state.sr() != OUTPUT_SR:
        raise SystemExit(
            f"{LOG_PREFIX} upstream df_state sr={df_state.sr()} != {OUTPUT_SR}; refusing to resample"
        )

    tensor = torch.from_numpy(np.asarray(noisy, dtype="float32")).unsqueeze(0)
    with torch.inference_mode():
        enhanced = df_enhance(model, df_state, tensor)
    enhanced_np = enhanced.squeeze(0).cpu().numpy().astype("float32", copy=False)

    if enhanced_np.shape != noisy.shape:
        raise SystemExit(
            f"{LOG_PREFIX} upstream output shape {enhanced_np.shape} != input {noisy.shape}"
        )
    write_f32(out_path, enhanced_np)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DFN3 parity reference audio (clean_48k.f32 / noisy_48k.f32 / enhanced_upstream.f32)"
    )
    p.add_argument(
        "--clean-source",
        type=Path,
        default=Path("tests/fixtures/audio/jfk-30s.wav"),
        help="Input WAV (16 kHz mono expected). Default: tests/fixtures/audio/jfk-30s.wav",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write clean_48k.f32 + noisy_48k.f32 (+ enhanced_upstream.f32 if --enhance)",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help=(
            f"RNG seed for the white-noise vector (default: {DEFAULT_NOISE_SEED}, "
            "matches campaign-2 measured baseline)"
        ),
    )
    p.add_argument(
        "--snr-db",
        type=float,
        default=DEFAULT_SNR_DB,
        help=f"Target SNR in dB (default: {DEFAULT_SNR_DB}, matches harness bound)",
    )
    p.add_argument(
        "--enhance",
        action="store_true",
        help="Also run upstream deepfilternet over the noisy signal to produce enhanced_upstream.f32",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to unpacked DeepFilterNet3 directory (required with --enhance)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.enhance and args.model_dir is None:
        raise SystemExit(f"{LOG_PREFIX} --enhance requires --model-dir")

    if not args.clean_source.is_file():
        raise SystemExit(f"{LOG_PREFIX} --clean-source not a file: {args.clean_source}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"reading {args.clean_source}")
    clean_16k, sr = read_wav_mono_f32(args.clean_source)
    log(f"input: sr={sr} Hz, {clean_16k.size} samples ({clean_16k.size / sr:.3f} s)")
    if sr != 16_000:
        raise SystemExit(
            f"{LOG_PREFIX} expected 16 kHz input (harness calibrations assume it); "
            f"got sr={sr}. Resample your source to 16 kHz mono PCM16 first."
        )

    log(f"resampling 16 kHz → {OUTPUT_SR} Hz via torchaudio.functional.resample")
    clean_48k = resample_16k_to_48k(clean_16k)
    log(f"resampled: {clean_48k.size} samples ({clean_48k.size / OUTPUT_SR:.3f} s)")

    log(f"drawing white noise (seed={args.noise_seed}) and scaling to {args.snr_db:.3f} dB SNR")
    import numpy as np

    rng = np.random.default_rng(args.noise_seed)
    raw_noise = rng.standard_normal(clean_48k.size)
    noise_scaled = scale_noise_to_snr(clean_48k, raw_noise, args.snr_db)
    noisy_48k = (clean_48k.astype("float64") + noise_scaled).astype("float32")

    measured = measure_snr_db(clean_48k, noisy_48k)
    log(f"measured SNR: {measured:.4f} dB (target {args.snr_db:.3f} dB)")
    if abs(measured - args.snr_db) > 0.01:
        raise SystemExit(
            f"{LOG_PREFIX} SNR calibration drift: measured {measured:.4f} vs target {args.snr_db:.3f} "
            f"(> 0.01 dB) — a floor-shift, refuse rather than let downstream taps break silently"
        )

    write_f32(args.out_dir / "clean_48k.f32", clean_48k)
    write_f32(args.out_dir / "noisy_48k.f32", noisy_48k)

    if args.enhance:
        log("running upstream deepfilternet over noisy_48k → enhanced_upstream.f32")
        run_upstream_enhance(noisy_48k, args.out_dir / "enhanced_upstream.f32", args.model_dir)

    log("done")


if __name__ == "__main__":
    main()
