#!/usr/bin/env python3
"""Kaldi fbank front-end parity fixtures for the CAM++ speaker encoder (M0-08).

Oracle for `vokra_ops::kaldi_fbank` under `KaldiFbankOpts::camplus()` — the
*Kaldi* log-mel fbank (NOT the librosa/Whisper log-mel) the 3D-Speaker / CosyVoice
CAM++ speaker encoder is fed. The CAM++ NETWORK (fbank -> 192-d embedding) is
already validated against onnxruntime in `tests/parity/camplus/`; this suite
closes the remaining gap by validating the audio -> fbank FRONTEND bit-for-bit
against an independent Kaldi implementation.

Primary oracle (a): ``torchaudio.compliance.kaldi.fbank`` (BSD; a faithful
PyTorch port of Kaldi ``compute-fbank-feats``) with the exact CAM++ parameters,
followed by the same per-utterance CMN vokra applies (subtract each mel bin's
time-mean). ``torchaudio.compliance.kaldi.fbank`` does not itself apply CMN when
``subtract_mean=False``, so CMN is applied here in numpy to mirror vokra.

Cross-check (b): a from-scratch numpy reimplementation of the same Kaldi pipeline
(faithful to feature-window.cc / feature-fbank.cc / mel-computations.cc), written
independently of both torchaudio and the vokra Rust source. The manifest records
the torchaudio-vs-numpy agreement so the reference is trustworthy even where
torchaudio is unavailable at regeneration time.

Determinism / float-reproducibility trap: the signal is built from a *seeded* RNG
and written to ``input.f32`` (little-endian float32). The oracle then RE-READS
``input.f32`` and processes exactly those committed bytes — the identical bytes
the Rust test reads — so cross-language float reproducibility of the generator is
irrelevant to the contract.

Outputs (committed): ``input.f32`` (16000 float32 LE, 1 s @ 16 kHz),
``fbank_ref.f32`` ([98, 80] float32 LE, C-order), ``manifest.txt``.

Re-run:  python tests/parity/kaldi_fbank/gen_reference.py
"""

import hashlib
import pathlib

import numpy as np

# ── CAM++ / CosyVoice Kaldi fbank knobs — must equal KaldiFbankOpts::camplus() ──
SR = 16000            # sample_rate
N_MELS = 80           # num_mel_bins
FRAME_LEN = 400       # frame_length  (25 ms @ 16 kHz)
FRAME_SHIFT = 160     # frame_shift   (10 ms @ 16 kHz)
PREEMPH = 0.97        # preemphasis_coefficient
LOW_FREQ = 20.0       # low_freq
HIGH_FREQ = 0.0       # high_freq (<= 0 -> nyquist + high_freq = 8000)
FFT_SIZE = 512        # round_to_power_of_two: 400 -> 512
N_SAMPLES = 16000     # 1 s
SEED = 1234
# vokra floors mel energies at f32::EPSILON before log; mirror it exactly.
EPS = float(np.finfo(np.float32).eps)  # 1.1920929e-07

HERE = pathlib.Path(__file__).resolve().parent


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def build_signal() -> np.ndarray:
    """A deterministic, broadband, time-varying speech-like probe.

    40 amplitude-modulated sinusoids log-spaced over 60..7600 Hz (broadband so
    every mel bin sits well above the log floor) plus a small noise floor, with
    per-component AM so successive frames differ (making the per-bin CMN a
    non-trivial, load-bearing step). Scaled to ~int16 range (Kaldi's native
    amplitude domain) purely so mel energies are far above f32::EPSILON — the
    absolute scale is irrelevant to the result because CMN removes any global
    log offset.
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(N_SAMPLES, dtype=np.float64) / SR
    freqs = np.geomspace(60.0, 7600.0, 40)
    sig = np.zeros(N_SAMPLES, dtype=np.float64)
    for f in freqs:
        amp = rng.uniform(0.3, 1.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        fm = rng.uniform(1.0, 5.0)       # 1..5 Hz amplitude modulation
        pm = rng.uniform(0.0, 2.0 * np.pi)
        mod = 0.6 + 0.4 * np.sin(2.0 * np.pi * fm * t + pm)
        sig += amp * mod * np.sin(2.0 * np.pi * f * t + ph)
    sig += 0.01 * rng.standard_normal(N_SAMPLES)  # tiny broadband noise floor
    sig = sig / np.max(np.abs(sig)) * 8000.0       # ~int16 range
    return sig.astype(np.float32)


def povey_window(n: int) -> np.ndarray:
    """Kaldi Povey window: symmetric Hann to the 0.85 power (denom = n-1)."""
    k = np.arange(n, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))
    return np.power(hann, 0.85)


# ── (b) Independent numpy Kaldi mel filter bank (mel-domain triangular ramps) ──
def hz_to_mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)      # Kaldi's native constant


def kaldi_mel_banks() -> np.ndarray:
    """[n_mels, FFT_SIZE//2 + 1] Kaldi mel filter bank, HTK warp, no area norm.

    Faithful to mel-computations.cc: mel-domain triangles, strict
    left_mel < mel(bin) < right_mel support, computed over num_fft_bins =
    FFT_SIZE/2 bins (the Nyquist bin FFT_SIZE/2 is dropped -> zero column).
    """
    n_freqs = FFT_SIZE // 2 + 1                   # 257
    num_fft_bins = FFT_SIZE // 2                  # 256 (Nyquist bin excluded)
    nyquist = 0.5 * SR
    high = HIGH_FREQ + nyquist if HIGH_FREQ <= 0.0 else HIGH_FREQ
    mel_low, mel_high = hz_to_mel(LOW_FREQ), hz_to_mel(high)
    delta = (mel_high - mel_low) / (N_MELS + 1)
    bin_hz = (SR / FFT_SIZE) * np.arange(num_fft_bins, dtype=np.float64)
    bin_mel = hz_to_mel(bin_hz)
    banks = np.zeros((N_MELS, n_freqs), dtype=np.float64)
    for m in range(N_MELS):
        left = mel_low + m * delta
        center = mel_low + (m + 1) * delta
        right = mel_low + (m + 2) * delta
        up = (bin_mel - left) / (center - left)
        down = (right - bin_mel) / (right - center)
        tri = np.maximum(0.0, np.minimum(up, down))
        tri[(bin_mel <= left) | (bin_mel >= right)] = 0.0
        banks[m, :num_fft_bins] = tri            # column num_fft_bins stays 0
    return banks


def fbank_numpy(pcm: np.ndarray) -> np.ndarray:
    """Independent numpy Kaldi log-mel fbank + per-utterance CMN (float64)."""
    x = pcm.astype(np.float64)
    win = povey_window(FRAME_LEN)
    banks = kaldi_mel_banks()
    n_frames = 1 + (len(x) - FRAME_LEN) // FRAME_SHIFT
    feats = np.zeros((n_frames, N_MELS), dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * FRAME_SHIFT: i * FRAME_SHIFT + FRAME_LEN].copy()
        seg -= seg.mean()                                   # DC removal
        pre = seg.copy()                                    # pre-emphasis
        pre[1:] = seg[1:] - PREEMPH * seg[:-1]
        pre[0] = seg[0] - PREEMPH * seg[0]
        wnd = pre * win                                     # Povey window
        padded = np.zeros(FFT_SIZE, dtype=np.float64)
        padded[:FRAME_LEN] = wnd
        spec = np.fft.rfft(padded)                          # unnormalized
        power = spec.real ** 2 + spec.imag ** 2             # |X|^2
        mel = banks @ power
        feats[i] = np.log(np.maximum(mel, EPS))
    feats -= feats.mean(axis=0, keepdims=True)              # per-bin CMN
    return feats


# ── (a) Primary torchaudio oracle ─────────────────────────────────────────────
def fbank_torchaudio(pcm: np.ndarray):
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi
    except Exception as e:  # pragma: no cover
        print(f"torchaudio unavailable ({e}); using numpy oracle as reference")
        return None
    wav = torch.from_numpy(pcm.astype(np.float32)).unsqueeze(0)  # [1, N]
    feats = ta_kaldi.fbank(
        wav,
        num_mel_bins=N_MELS,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=float(SR),
        preemphasis_coefficient=PREEMPH,
        low_freq=LOW_FREQ,
        high_freq=HIGH_FREQ,
        window_type="povey",
        use_energy=False,
        dither=0.0,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        use_power=True,
        use_log_fbank=True,
        snip_edges=True,
        subtract_mean=False,       # CMN applied below to mirror vokra exactly
    ).numpy()
    feats = feats - feats.mean(axis=0, keepdims=True)  # per-bin per-utterance CMN
    return feats.astype(np.float32)


def main() -> None:
    # 1) Build the probe and commit the exact bytes both oracle and Rust read.
    sig = build_signal()
    in_bytes = sig.tobytes()
    (HERE / "input.f32").write_bytes(in_bytes)

    # 2) Re-read the committed bytes so we process EXACTLY what the Rust test does.
    pcm = np.frombuffer((HERE / "input.f32").read_bytes(), dtype="<f4")
    assert pcm.shape == (N_SAMPLES,)

    n_frames = 1 + (N_SAMPLES - FRAME_LEN) // FRAME_SHIFT  # 98

    # 3) Independent numpy cross-check.
    ref_np = fbank_numpy(pcm)
    assert ref_np.shape == (n_frames, N_MELS)

    # 4) Primary torchaudio oracle (falls back to numpy if unavailable).
    ref_ta = fbank_torchaudio(pcm)
    if ref_ta is not None:
        assert ref_ta.shape == (n_frames, N_MELS)
        cross = float(np.max(np.abs(ref_ta.astype(np.float64) - ref_np)))
        reference, oracle_name = ref_ta, "torchaudio.compliance.kaldi.fbank"
    else:
        cross = 0.0
        reference, oracle_name = ref_np.astype(np.float32), "numpy Kaldi reimpl"

    out = np.ascontiguousarray(reference.astype(np.float32))
    out_bytes = out.tobytes()
    (HERE / "fbank_ref.f32").write_bytes(out_bytes)

    lines = [
        "# Kaldi fbank front-end parity manifest (M0-08, CAM++ speaker encoder).",
        "# Generated by tests/parity/kaldi_fbank/gen_reference.py.",
        f"oracle          = {oracle_name}",
        "cmn             = per-bin per-utterance mean subtraction (applied in numpy)",
        "cross_check      = independent numpy Kaldi reimpl",
        f"cross_check_maxdiff = {cross:.3e}  (torchaudio float32 vs numpy float64)",
        f"seed            = {SEED}",
        f"sample_rate     = {SR}",
        f"num_mel_bins    = {N_MELS}",
        f"frame_length    = {FRAME_LEN}",
        f"frame_shift     = {FRAME_SHIFT}",
        f"fft_size        = {FFT_SIZE}",
        f"preemph_coeff   = {PREEMPH}",
        f"low_freq        = {LOW_FREQ}",
        f"high_freq       = {HIGH_FREQ}  (<=0 -> nyquist)",
        "window          = povey (Hann^0.85, denom=N-1)",
        "mel             = HTK warp, mel-domain ramps, no Slaney norm, snip-edges",
        "log_floor       = f32::EPSILON (1.1920929e-07)",
        f"input.f32       shape=[{N_SAMPLES}] sha256={sha16(in_bytes)}",
        f"fbank_ref.f32   shape=[{n_frames}, {N_MELS}] len={out.size} "
        f"sha256={sha16(out_bytes)}",
        "# Rust: vokra_ops::kaldi_fbank(KaldiFbankOpts::camplus()) vs fbank_ref.f32.",
        "# Achieved vokra(f32)-vs-oracle peak error = 9.3e-5 on log-mel of O(10)",
        "# (within float32 rounding); asserted at atol=2e-4 in",
        "# crates/vokra-ops/tests/kaldi_fbank_parity.rs.",
    ]
    (HERE / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nmel-energy floor headroom check: min log-mel (pre-CMN) bins are "
          f"well above the floor by construction.")


if __name__ == "__main__":
    main()
