#!/usr/bin/env python3
"""Dump Kokoro-82M reference tensors for parity — TRUE-UPSTREAM edition.

This is an **offline** tool (FR-LD-05: no Python / PyTorch is ever pulled into
the runtime). It regenerates the fixtures under ``tests/parity/kokoro/`` that
the Rust parity tests (``crates/vokra-models/tests/parity_kokoro.rs``) compare
against.

REWRITTEN 2026-07-16 (P1 fidelity fix — real-weight eval finding #2)
=====================================================================

The previous revision of this script was a **from-scratch PyTorch
re-implementation that mirrored the Rust code layer-by-layer**. That made the
fixtures a *self-consistency* check, not a *fidelity* check: both sides shared
the same author and the same misreadings of the architecture (LeakyReLU 0.1
vs upstream 0.2, 4 ALBERT layers + erf-GELU vs 12 + gelu_new, affine-only
"LayerNorm", γ-only AdaIN, zero harmonic source, `ceil(exp(log_dur))`
durations, `sin(x)·π` phase, BERT features into the decoder, zero F0/N).
9/9 tensors "passed" while the synthesized audio was unintelligible
(whisper-base round-trip WER 1.0 vs the true upstream's 0.0).

**This revision imports the REAL upstream ``kokoro`` pip package and hooks
``KModel.forward_with_tokens`` directly.** The reference tensors ARE the
upstream model's activations; there is no re-implementation left in this
file, and there must never be one again. If ``import kokoro`` fails, the
script aborts loudly — it does NOT fall back to any local re-forward.

Upstream-truth adjudication (2026-07-16 eval, ``oracle_kmodel.py``): the
package (kokoro==0.9.4 + the pinned ``kokoro-v1_0.pth``) and the independent
``onnx-community/Kokoro-82M-v1.0-ONNX`` export agree to
``bert_encoder max|Δ| = 1.7e-5``, ``text_encoder max|Δ| = 3.7e-6`` and
bit-equal ``pred_dur`` on the fixture input — two independently-derived
implementations agreeing pins this package as ground truth.

Environment
-----------

Requires ``kokoro==0.9.4`` (which pulls torch / transformers / misaki /
loguru / huggingface_hub) + ``numpy``. Use a dedicated venv, e.g.::

    python3.11 -m venv kokoro-venv
    kokoro-venv/bin/pip install "kokoro==0.9.4" numpy
    kokoro-venv/bin/python tools/parity/dump_kokoro_reference.py \
        --checkpoint /path/to/kokoro-v1_0.pth \
        --config /path/to/config.json \
        [tests/parity/kokoro]

Without ``--checkpoint`` / ``--config``, both are fetched from the pinned
``hexgrad/Kokoro-82M`` HF repo (cached).

Determinism — SineGen RNG substitution
--------------------------------------

Upstream ``SineGen`` draws two RNG tensors per forward (``istftnet.py``):

* ``torch.rand`` initial phase offsets for harmonics ≥ 2 (line ~150);
* ``noise_amp · torch.randn_like`` additive dither (line ~205).

These make the upstream waveform **irreproducible even against itself**
(the eval measured oracle-vs-its-own-ONNX waveform ``max|Δ| = 0.44`` from
exactly this). The Vokra native decoder pins the phase offsets to ZERO (a
valid draw) and replaces the dither with a **counter-based SplitMix64 +
Box–Muller normal** at the upstream amplitude — the dither is designed
excitation (the sole source energy in unvoiced regions), so it must keep
its N(0, 1) statistics, just with reproducible bits (see
``crates/vokra-models/src/kokoro/decoder/generator.rs``
``deterministic_gauss``). This dumper monkeypatches ``torch.rand`` to zeros
and ``torch.randn_like`` to the SAME deterministic generator for the
duration of the reference forward, so fixtures compare
deterministic-to-deterministic with identical noise bits. The substitution
is recorded in the manifest (``sinegen_rng``) — documented and bounded, not
a silent divergence. All pre-decoder taps (bert / text_encoder / durations /
f0 / n / hidden) are unaffected by the patch.

Tensor taps (9)
---------------

============================  =========================================  =============================
fixture file                  upstream tap                               shape (fixture input)
============================  =========================================  =============================
``bert.f32``                  ``model.bert_encoder`` output              first ENC_POS rows of [T, 512]
``text_encoder.f32``          ``model.text_encoder`` output (t_en)       first ENC_POS rows of [T, 512]
``prosody_durations.i64``     ``pred_dur`` return value                  [T]
``prosody_f0.f32``            ``model.predictor.F0_proj`` output         [2·T_frames]
``prosody_n.f32``             ``model.predictor.N_proj`` output          [2·T_frames]
``prosody_hidden.f32``        ``model.predictor.shared`` LSTM output     [512, T_frames] channel-major
``decoder_pre_istft_mag.f32``   ``generator.conv_post`` out rows :n_half   [n_half, t_gen] channel-major
``decoder_pre_istft_phase.f32`` ``generator.conv_post`` out rows n_half:   [n_half, t_gen] channel-major
``decoder_pcm.f32``           ``forward_with_tokens`` audio return       [hop · (t_gen − 1)]
============================  =========================================  =============================

Inputs: ``phoneme_ids.i64`` (24 seed-derived ids) + ``style.f32`` (128-dim
unit-norm, seed-derived). ``ref_s = concat([style, style])`` [1, 256] — the
Rust side's 128-dim style path applies the same vector to both the decoder
and prosody halves, so the two conventions coincide on the fixture. Existing
input files in the out dir are REUSED byte-for-byte (regeneration from the
fixed seeds produces identical bytes).

The legacy placeholder files (``prosody.f32`` / ``mel_pre_istft.f32`` /
``pcm.f32``) predate the per-module taps and remain seed-derived noise —
kept only so the fixture-shape tests stay stable; the Rust harness never
byte-compares them in ``mode = full``.

.. note::

   Kokoro-82M is Apache 2.0 code + weight (docs/license-audit.md), so the
   fixtures can be committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEED = 1234
NUM_PHONEMES = 24  # short deterministic sequence: enough for parity, small on disk
ENC_POS = 8  # text encoder / bert positions to dump
DEC_FRAMES = 64  # legacy placeholder mel frames
PCM_LEN = 16000  # legacy placeholder pcm samples
ATOL = 0.01

# Pinned upstream identity.
MODEL_REPO = "hexgrad/Kokoro-82M"
CHECKPOINT_FILE = "kokoro-v1_0.pth"

# Repo-relative default output directory.
DEFAULT_OUT_DIR = Path("tests/parity/kokoro")


def synth_phoneme_ids(vocab_size: int) -> np.ndarray:
    """Deterministic in-range phoneme id sequence, seed-derived.

    Byte-identical to the pre-rewrite dumper's generator so the committed
    ``phoneme_ids.i64`` stays stable across the true-upstream migration.
    """
    rng = np.random.default_rng(SEED)
    # Reserve id 0 (blank / pad in most Kokoro-style vocabs); use [1, vocab_size).
    upper = max(vocab_size, 2)
    return rng.integers(1, upper, size=NUM_PHONEMES, dtype=np.int64)


def synth_style(dim: int) -> np.ndarray:
    """Deterministic unit-norm style vector (byte-stable, see above)."""
    rng = np.random.default_rng(SEED + 1)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / (n if n > 0.0 else 1.0)


def synth_legacy_placeholders(hidden_dim: int, mel_channels: int):
    """Legacy placeholder tensors (`prosody.f32` etc.), seed-derived.

    Never byte-compared in ``mode = full``; regenerated with the exact
    pre-rewrite logic so the committed bytes do not churn.
    """
    rng = np.random.default_rng(SEED + 2)
    _text_encoder = rng.standard_normal((ENC_POS, hidden_dim)).astype(np.float32)
    prosody = rng.standard_normal((ENC_POS, 2)).astype(np.float32)
    mel_pre_istft = rng.standard_normal((DEC_FRAMES, mel_channels)).astype(np.float32)
    pcm = rng.standard_normal(PCM_LEN).astype(np.float32) * 0.1
    return prosody, mel_pre_istft, pcm


def det_gauss(n: int) -> np.ndarray:
    """Counter-based standard normal — SplitMix64 + Box–Muller.

    BIT-MIRRORED by the Rust runtime
    (``crates/vokra-models/src/kokoro/decoder/generator.rs``
    ``deterministic_gauss``): flat index ``m`` seeds two SplitMix64 outputs
    (``2m``, ``2m+1``); ``u1 ∈ (0, 1]``, ``u2 ∈ [0, 1)``;
    ``z = sqrt(−2·ln u1) · cos(2π·u2)`` in f64, narrowed to f32. Any change
    here MUST land in the Rust twin in the same commit.
    """
    with np.errstate(over="ignore"):
        m = np.arange(n, dtype=np.uint64)

        def sm(z):
            z = z + np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            return z ^ (z >> np.uint64(31))

        x = sm(m * np.uint64(2))
        y = sm(m * np.uint64(2) + np.uint64(1))
    two53 = 9007199254740992.0  # 2^53
    u1 = ((x >> np.uint64(11)).astype(np.float64) + 1.0) / two53
    u2 = (y >> np.uint64(11)).astype(np.float64) / two53
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z.astype(np.float32)


def write_f32(path: Path, arr) -> None:
    a = np.asarray(arr, dtype="<f4").reshape(-1)
    path.write_bytes(a.tobytes())


def write_i64(path: Path, arr) -> None:
    a = np.asarray(arr, dtype="<i8").reshape(-1)
    path.write_bytes(a.tobytes())


def load_or_synth_inputs(out_dir: Path, vocab_size: int, style_dim: int):
    """Reuse committed inputs byte-for-byte; synthesize only when absent."""
    ids_path = out_dir / "phoneme_ids.i64"
    style_path = out_dir / "style.f32"
    if ids_path.is_file():
        ids = np.frombuffer(ids_path.read_bytes(), dtype="<i8")
        print(f"[dump_kokoro] reusing existing {ids_path} ({ids.size} ids)")
    else:
        ids = synth_phoneme_ids(vocab_size)
    if style_path.is_file():
        style = np.frombuffer(style_path.read_bytes(), dtype="<f4")
        print(f"[dump_kokoro] reusing existing {style_path} ({style.size} dims)")
    else:
        style = synth_style(style_dim)
    if ids.size != NUM_PHONEMES:
        sys.exit(f"phoneme_ids.i64 holds {ids.size} ids, expected {NUM_PHONEMES}")
    if style.size != style_dim:
        sys.exit(f"style.f32 holds {style.size} dims, expected style_dim {style_dim}")
    return ids.copy(), style.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoint",
        default=None,
        help=f"path to {CHECKPOINT_FILE} (default: hf_hub_download from {MODEL_REPO})",
    )
    ap.add_argument(
        "--config",
        default=None,
        help=f"path to the upstream config.json (default: hf_hub_download from {MODEL_REPO})",
    )
    ap.add_argument(
        "out_dir",
        nargs="?",
        default=str(DEFAULT_OUT_DIR),
        help=f"fixture output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = ap.parse_args()

    # ---- Hard requirement: the TRUE upstream package -----------------------
    try:
        import torch
        import kokoro as kokoro_pkg
        from kokoro import KModel
    except ImportError as e:  # pragma: no cover - environment guard
        sys.exit(
            "FATAL: the upstream `kokoro` package (and torch) is REQUIRED — "
            "`pip install kokoro==0.9.4`. This dumper never falls back to a "
            f"re-implementation (that was the P1 self-consistency bug): {e}"
        )

    config_path = args.config
    ckpt_path = args.checkpoint
    if config_path is None or ckpt_path is None:
        from huggingface_hub import hf_hub_download

        if config_path is None:
            config_path = hf_hub_download(repo_id=MODEL_REPO, filename="config.json")
        if ckpt_path is None:
            ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)

    with open(config_path, "r", encoding="utf-8") as r:
        config = json.load(r)
    vocab_size = int(config["n_token"])
    hidden_dim = int(config["hidden_dim"])
    style_dim = int(config["style_dim"])
    mel_channels = int(config["n_mels"])
    istft_n_fft = int(config["istftnet"]["gen_istft_n_fft"])
    istft_hop = int(config["istftnet"]["gen_istft_hop_size"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = KModel(repo_id=MODEL_REPO, config=config_path, model=ckpt_path).eval()

    ids, style = load_or_synth_inputs(out_dir, vocab_size, style_dim)

    # ref_s = [decoder_half | prosody_half]; the fixture uses ONE 128-dim
    # vector for both halves (see module docstring).
    ref_s = torch.from_numpy(np.concatenate([style, style]).astype(np.float32))[None, :]
    input_ids = torch.from_numpy(ids.astype(np.int64))[None, :]

    # ---- Hook the 9 taps ----------------------------------------------------
    caps: dict[str, torch.Tensor] = {}

    def cap(name):
        def hook(_m, _inp, out):
            caps[name] = out[0] if isinstance(out, tuple) else out

        return hook

    hooks = [
        model.bert_encoder.register_forward_hook(cap("bert_encoder")),
        model.text_encoder.register_forward_hook(cap("text_encoder")),
        model.predictor.shared.register_forward_hook(cap("shared")),
        model.predictor.F0_proj.register_forward_hook(cap("f0_proj")),
        model.predictor.N_proj.register_forward_hook(cap("n_proj")),
        model.decoder.generator.conv_post.register_forward_hook(cap("conv_post")),
    ]

    # ---- Deterministic forward: substitute the SineGen RNG ------------------
    # (module docstring §Determinism; recorded in the manifest.) `torch.rand`
    # (harmonic phase offsets) → zeros; `torch.randn_like` (dither) → the
    # shared SplitMix64 + Box–Muller generator, per-call counter from 0 in
    # row-major order — exactly the Rust runtime's indexing over the
    # [t_full, n_harm] dither tensor.
    torch.manual_seed(SEED)
    orig_rand, orig_randn_like = torch.rand, torch.randn_like

    def zero_rand(*size, **kw):
        kw = {k: v for k, v in kw.items() if k in ("dtype", "device", "layout")}
        return torch.zeros(*size, **kw)

    def det_randn_like(t, **_kw):
        flat = det_gauss(t.numel())
        return torch.from_numpy(flat).reshape(t.shape).to(t.dtype)

    torch.rand, torch.randn_like = zero_rand, det_randn_like
    try:
        with torch.no_grad():
            audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, 1.0)
    finally:
        torch.rand, torch.randn_like = orig_rand, orig_randn_like
        for h in hooks:
            h.remove()

    audio = audio.reshape(-1).cpu().numpy().astype(np.float32)
    pred_dur = pred_dur.reshape(-1).cpu().numpy().astype(np.int64)
    t = int(ids.size)
    t_frames = int(pred_dur.sum())

    # bert: bert_encoder output [1, T, 512] → first ENC_POS rows.
    bert_out = caps["bert_encoder"].detach()[0].cpu().numpy()
    assert bert_out.shape == (t, hidden_dim), f"bert_encoder shape {bert_out.shape}"
    # text_encoder: t_en [1, 512, T] channel-major → [T, 512] rows.
    t_en = caps["text_encoder"].detach()[0].cpu().numpy()
    assert t_en.shape == (hidden_dim, t), f"text_encoder shape {t_en.shape}"
    t_en_rows = t_en.T
    # prosody hidden: shared LSTM out [1, F, 512] → channel-major [512, F].
    shared = caps["shared"].detach()[0].cpu().numpy()
    assert shared.shape == (t_frames, hidden_dim), f"shared shape {shared.shape}"
    hidden_cm = shared.T
    # F0 / N: proj outputs [1, 1, 2F] → flat [2F].
    f0 = caps["f0_proj"].detach().reshape(-1).cpu().numpy()
    n_ten = caps["n_proj"].detach().reshape(-1).cpu().numpy()
    assert f0.size == 2 * t_frames, f"f0 len {f0.size} != 2·{t_frames}"
    assert n_ten.size == 2 * t_frames, f"n len {n_ten.size} != 2·{t_frames}"
    # decoder: conv_post [1, 2·n_half, t_gen] → mag/phase logits channel-major.
    conv_post = caps["conv_post"].detach()[0].cpu().numpy()
    n_half = istft_n_fft // 2 + 1
    assert conv_post.shape[0] == 2 * n_half, f"conv_post ch {conv_post.shape}"
    t_gen = int(conv_post.shape[1])
    mag = conv_post[:n_half, :]
    phase = conv_post[n_half:, :]
    # torch.istft(center=True) emits hop · (t_gen − 1) samples.
    assert audio.size == istft_hop * (t_gen - 1), (
        f"audio len {audio.size} != hop·(t_gen−1) = {istft_hop * (t_gen - 1)}"
    )

    # ---- Write fixtures ------------------------------------------------------
    write_i64(out_dir / "phoneme_ids.i64", ids)
    write_f32(out_dir / "style.f32", style)
    write_f32(out_dir / "bert.f32", bert_out[:ENC_POS, :])
    write_f32(out_dir / "text_encoder.f32", t_en_rows[:ENC_POS, :])
    write_i64(out_dir / "prosody_durations.i64", pred_dur)
    write_f32(out_dir / "prosody_f0.f32", f0)
    write_f32(out_dir / "prosody_n.f32", n_ten)
    write_f32(out_dir / "prosody_hidden.f32", hidden_cm)
    write_f32(out_dir / "decoder_pre_istft_mag.f32", mag)
    write_f32(out_dir / "decoder_pre_istft_phase.f32", phase)
    write_f32(out_dir / "decoder_pcm.f32", audio)

    prosody_ph, mel_ph, pcm_ph = synth_legacy_placeholders(hidden_dim, mel_channels)
    write_f32(out_dir / "prosody.f32", prosody_ph)
    write_f32(out_dir / "mel_pre_istft.f32", mel_ph)
    write_f32(out_dir / "pcm.f32", pcm_ph)

    manifest = f"""# Kokoro-82M parity manifest (M2-07 / 2026-07-16 true-upstream rewrite).
# Generated by tools/parity/dump_kokoro_reference.py. `key = value`;
# list values are space-separated.
# reference = upstream-kokoro-package: every `mode = full` tensor is a
# HOOKED ACTIVATION of the real `kokoro` pip package's KModel — never a
# re-implementation (the pre-2026-07-16 self-consistency bug).
# sinegen_rng = deterministic-splitmix64: torch.rand (SineGen harmonic phase
# offsets) was patched to zeros and torch.randn_like (dither) to the shared
# counter-based SplitMix64 + Box-Muller normal the Rust runtime mirrors
# (generator.rs::deterministic_gauss). Pre-decoder taps are unaffected.
model = {MODEL_REPO}
checkpoint_file = {CHECKPOINT_FILE}
reference = upstream-kokoro-package
kokoro_version = {kokoro_pkg.__version__}
torch_version = {torch.__version__}
sinegen_rng = deterministic-splitmix64
seed = {SEED}
atol = {ATOL}
mode = full
text_encoder_mode = full
bert_mode = full
prosody_mode = full
decoder_mode = full
vocab_size = {vocab_size}
hidden_dim = {hidden_dim}
style_dim = {style_dim}
voice_id = 0
voice_name =
num_phonemes = {NUM_PHONEMES}
enc_pos = {ENC_POS}
dec_frames = {DEC_FRAMES}
prosody_channels = 2
mel_channels = {mel_channels}
bert_out_dim = {hidden_dim}
pcm_len = {PCM_LEN}
sample_rate = 24000
prosody_t_frames = {t_frames}
prosody_d_model = {hidden_dim}
prosody_f0_len = {2 * t_frames}
prosody_n_len = {2 * t_frames}
decoder_n_half = {n_half}
decoder_t_gen = {t_gen}
decoder_pcm_len = {audio.size}
istft_n_fft = {istft_n_fft}
istft_hop = {istft_hop}
istft_win_length = {istft_n_fft}
"""
    (out_dir / "manifest.txt").write_text(manifest, encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "dumped",
                "out_dir": str(out_dir),
                "pred_dur": pred_dur.tolist(),
                "t_frames": t_frames,
                "t_gen": t_gen,
                "pcm_len": int(audio.size),
                "kokoro": kokoro_pkg.__version__,
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
