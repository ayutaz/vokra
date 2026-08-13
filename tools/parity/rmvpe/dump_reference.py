#!/usr/bin/env python3
"""RMVPE reference dumper — Path B fixture generator for `parity_rmvpe.rs`.

Fair-use verbatim reference: instantiates the **upstream** yxlllc/RMVPE
(or Dream-High/RMVPE fork, same architecture) ``nn.Module`` in-process,
loads the released ``.pt`` pickle, runs the real mel + U-Net forward on
a supplied PCM (or canned sine sweep), captures the post-CNN hidden
state via a pre-forward hook on the GRU submodule, then dumps three
artefacts the vokra Rust-side parity harness reads:

* ``hidden.f32``  — raw little-endian f32 buffer, shape
  ``[n_frames * feature_dim]`` (row-major, `[n_frames, feature_dim]`)
* ``argmax.u32``  — raw little-endian u32 buffer, shape ``[n_frames]``
  (per-frame pitch-class index; 0 == unvoiced per the upstream head
  saturation convention)
* ``meta.json``   — provenance + shape metadata (n_frames, feature_dim,
  sample_rate, hop, upstream variant / class name, ``.pt`` sha256, PCM
  sha256, dumper version)

**IMPORTANT — CC does NOT run this script.** Per依頼者ルール #3, the
Vokra CC pipeline provides converter + parity harness + this dumper
*script*, and the owner runs the actual dump on the machine that has
the fetched upstream ``.pt`` (see ``fetch_rmvpe_pt.sh``) and a checkout
of the upstream repository (see ``--upstream-src``). The dumper writes
raw binary — nothing shipped as part of vokra-* runtime (FR-LD-05).

# Path B contract (parity_rmvpe.rs L82-104)

The Rust-side harness reads the dumped buffers via
``std::fs::read`` + ``chunks_exact(4)`` — the byte layout is little-
endian contiguous with **no** header (numpy ``.npy`` magic would be a
loud parity failure). The owner sets these env vars on the CI runner
or local machine before ``cargo test -p vokra-models --test parity_rmvpe``:

    export VOKRA_RMVPE_REAL_GGUF=~/rmvpe.gguf                # Path A + B
    export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe-fixtures/hidden.f32
    export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=<from meta.json>
    export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe-fixtures/argmax.u32

The ``feature_dim`` env var is required because the raw buffer carries
no shape header; it is echoed in ``meta.json`` for owner copy-paste.

# Owner usage

::

    cd tools/parity/rmvpe
    uv sync

    # 1. Fetch upstream .pt (owner-side, see fetch_rmvpe_pt.sh):
    bash ./fetch_rmvpe_pt.sh --output ~/rmvpe-fixtures/rmvpe.pt

    # 2. Clone the upstream repo for the nn.Module class:
    git clone https://github.com/yxlllc/RMVPE.git ~/rmvpe-upstream

    # 3. Dump against a canned 30 s sine sweep (no external WAV needed):
    uv run python dump_reference.py \\
        --pt-path      ~/rmvpe-fixtures/rmvpe.pt \\
        --upstream-src ~/rmvpe-upstream \\
        --canned \\
        --out-dir      ~/rmvpe-fixtures/dump

    # 4. Or dump against a real WAV clip (16 kHz mono PCM16 preferred;
    #    the dumper resamples if needed):
    uv run python dump_reference.py \\
        --pt-path      ~/rmvpe-fixtures/rmvpe.pt \\
        --upstream-src ~/rmvpe-upstream \\
        --pcm          ~/my-clip.wav \\
        --out-dir      ~/rmvpe-fixtures/dump

    # 5. Read feature_dim from meta.json and wire the harness:
    export VOKRA_RMVPE_REAL_GGUF=~/rmvpe-fixtures/rmvpe.gguf
    export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe-fixtures/dump/hidden.f32
    export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/rmvpe-fixtures/dump/meta.json"))["feature_dim"])')
    export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe-fixtures/dump/argmax.u32
    cargo test -p vokra-models --test parity_rmvpe -- --nocapture

# Upstream API expectations

The dumper works against the **standard** yxlllc/RMVPE and
Dream-High/RMVPE topology (they are the same architecture; yxlllc is a
maintenance fork). The Python module class is discovered dynamically:

1. ``src.model.E2E``  — yxlllc/RMVPE canonical name
2. ``src.model.RMVPE`` — Dream-High/RMVPE canonical name
3. Any ``nn.Module`` subclass in ``src.model`` that has both a mel
   front-end submodule (any of ``mel_extractor`` / ``mel`` / ``spec``)
   and a GRU submodule (any of ``gru`` / ``bigru`` / ``rnn``)

If none of the above match (rare fork), pass ``--module-import
<mod.path.ClassName>`` to import a specific class, and ``--gru-attr
<name>`` to point at the GRU submodule that consumes the post-CNN
hidden state.

The pre-forward hook on the GRU captures ``input[0]`` which is the
``[B, T, feature_dim]`` tensor the BiGRU consumes; that is exactly
what ``RMVPE::forward_from_hidden`` in ``crates/vokra-models/src/f0/rmvpe.rs``
expects on the Vokra side.

# License / distribution note

The **yxlllc/RMVPE** upstream and its parent **Dream-High/RMVPE** both
ship MIT license. The Vokra runtime consumes only the produced raw
binary fixtures as opaque numeric artefacts; **no upstream Python /
PyTorch code enters the runtime** (FR-LD-05 sidecar isolation). This
script imports the upstream ``nn.Module`` for offline dumping only —
the same fair-use pattern as ``dfn3_dump_reference.py`` (which imports
``df.enhance``) and ``dump_kokoro_reference.py`` (which imports the
real ``kokoro`` package).

.pt pickle security: PyTorch pickle allows arbitrary code execution.
Fetch only from the verified upstream releases (see
``fetch_rmvpe_pt.sh`` for the pinned sha256s).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


DUMPER_VERSION = "0.1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canned_pcm(duration_sec: float, sample_rate: int) -> np.ndarray:
    """Deterministic canned PCM: half sine sweep 100 → 800 Hz, half silence.

    Chosen so a working RMVPE forward yields a mostly-voiced first half
    (pitch clearly in-range) and a mostly-unvoiced second half; owner can
    eyeball ``argmax.u32`` to sanity-check the dump before wiring
    into ``cargo test``.
    """
    n = int(duration_sec * sample_rate)
    half = n // 2

    # Linear sweep 100 Hz -> 800 Hz over the first half.
    t = np.arange(half, dtype=np.float64) / float(sample_rate)
    f_start, f_end = 100.0, 800.0
    # Instantaneous freq f(t) = f_start + k * t; phase = 2π ∫ f dt
    # = 2π (f_start * t + 0.5 * k * t^2), k = (f_end - f_start) / T.
    T = half / float(sample_rate)
    k = (f_end - f_start) / T
    phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t * t)
    sweep = 0.5 * np.sin(phase)

    silence = np.zeros(n - half, dtype=np.float64)
    pcm = np.concatenate([sweep, silence]).astype(np.float32)
    return pcm


def load_pcm(path: Path, target_sr: int) -> np.ndarray:
    """Load WAV/FLAC as f32 mono at target_sr. Simple linear resample if needed."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        # Downmix to mono by averaging channels.
        data = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        # Simple linear resample — for parity dumps we only need the
        # mel front-end to see the same waveform on both sides; if
        # the owner supplies a mismatched-rate WAV, this is a best-
        # effort resample. A high-quality resample belongs offline.
        # (For 44.1 kHz -> 16 kHz reference clips, prefer sox / ffmpeg
        # to preprocess and pass the 16 kHz WAV to --pcm.)
        n_out = int(round(len(data) * target_sr / sr))
        idx = np.linspace(0, len(data) - 1, n_out).astype(np.float64)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, len(data) - 1)
        frac = (idx - i0).astype(np.float32)
        data = ((1.0 - frac) * data[i0] + frac * data[i1]).astype(np.float32)
    return data


def add_upstream_src_to_syspath(upstream_src: Path) -> None:
    """Adds `<upstream_src>` and `<upstream_src>/src` to sys.path so
    ``from src.model import E2E`` and similar imports resolve."""
    if not upstream_src.is_dir():
        raise FileNotFoundError(
            f"--upstream-src {upstream_src} does not exist; expected a "
            f"checkout of github.com/yxlllc/RMVPE (or Dream-High/RMVPE)."
        )
    for entry in (upstream_src, upstream_src.parent):
        s = str(entry.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def instantiate_upstream_module(
    module_import: str | None,
    upstream_src: Path,
    hop: int,
    n_fft: int,
    n_mels: int,
    sample_rate: int,
) -> torch.nn.Module:
    """Discovers and instantiates the upstream RMVPE / E2E nn.Module class.

    Tries in order:

    1. Explicit ``--module-import <mod.path.ClassName>`` if provided.
    2. Common yxlllc/RMVPE convention: ``src.model.E2E``.
    3. Common Dream-High/RMVPE convention: ``src.model.RMVPE``.
    4. Fallback scan: any ``nn.Module`` in ``src.model`` that has both
       a mel-ish submodule and a GRU-ish submodule.

    Constructor signature is typically ``(hop, n_fft, n_mels, sr,
    n_class=360)``; adapt via subclassing if the owner's fork uses a
    different signature.
    """
    add_upstream_src_to_syspath(upstream_src)

    if module_import:
        mod_path, _, cls_name = module_import.rpartition(".")
        if not mod_path or not cls_name:
            raise ValueError(
                f"--module-import must be a dotted path like 'src.model.E2E', "
                f"got {module_import!r}"
            )
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise AttributeError(
                f"class {cls_name!r} not found in module {mod_path!r} "
                f"(loaded from {mod.__file__})"
            )
        candidates = [(cls_name, cls)]
    else:
        try:
            mod = importlib.import_module("src.model")
        except ImportError as e:
            raise ImportError(
                f"failed to import 'src.model' from --upstream-src "
                f"{upstream_src}; if your fork uses a different module layout "
                f"pass --module-import <mod.path.ClassName>. Original error: {e}"
            ) from e
        candidates = []
        for preferred in ("E2E", "RMVPE"):
            cls = getattr(mod, preferred, None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, torch.nn.Module):
                candidates.append((preferred, cls))
        if not candidates:
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                    candidates.append((name, obj))

    if not candidates:
        raise RuntimeError(
            f"could not find any nn.Module subclass in src.model "
            f"(upstream_src={upstream_src}); pass --module-import explicitly."
        )

    # Try each candidate constructor in turn — the standard upstream
    # signature is (hop, n_fft, n_mels, sr, ...). If yours differs, the
    # loud TypeError below tells the owner exactly what to adapt.
    last_err: Exception | None = None
    for cls_name, cls in candidates:
        try:
            model = cls(hop, n_fft, n_mels, sample_rate)
            print(
                f"[dumper] instantiated upstream class {cls_name!r} "
                f"from {getattr(cls, '__module__', '?')} "
                f"with (hop={hop}, n_fft={n_fft}, n_mels={n_mels}, "
                f"sr={sample_rate})"
            )
            return model
        except TypeError as e:
            last_err = e
            print(
                f"[dumper] candidate {cls_name!r} did not accept the "
                f"standard (hop, n_fft, n_mels, sr) signature: {e}",
                file=sys.stderr,
            )
            continue

    raise RuntimeError(
        f"none of the discovered candidates ({[c[0] for c in candidates]}) "
        f"accepted the standard (hop, n_fft, n_mels, sr) constructor. "
        f"Last TypeError: {last_err}. If your fork uses a different "
        f"constructor signature, either subclass to adapt or edit "
        f"instantiate_upstream_module() to match."
    )


def discover_gru_submodule(model: torch.nn.Module, gru_attr: str | None) -> torch.nn.Module:
    """Finds the BiGRU submodule to attach the pre-forward hook to."""
    if gru_attr:
        sub = model
        for part in gru_attr.split("."):
            if not hasattr(sub, part):
                raise AttributeError(
                    f"--gru-attr {gru_attr!r} not resolvable; "
                    f"stopped at {part!r} on {type(sub).__name__}"
                )
            sub = getattr(sub, part)
        if not isinstance(sub, torch.nn.Module):
            raise TypeError(
                f"--gru-attr {gru_attr!r} did not resolve to nn.Module "
                f"(got {type(sub).__name__})"
            )
        return sub

    # Standard yxlllc / Dream-High RMVPE: model.gru is the BiGRU
    # (nn.GRU with bidirectional=True); model.bigru or model.rnn on
    # some forks.
    for name in ("gru", "bigru", "rnn"):
        sub = getattr(model, name, None)
        if isinstance(sub, torch.nn.Module):
            print(f"[dumper] found GRU submodule at attribute {name!r}")
            return sub

    # Last resort: named_modules() scan for the first nn.GRU.
    for name, sub in model.named_modules():
        if isinstance(sub, torch.nn.GRU):
            print(f"[dumper] found nn.GRU via named_modules() scan at {name!r}")
            return sub

    raise RuntimeError(
        "could not locate a GRU submodule on the upstream model; pass "
        "--gru-attr <dotted.path.to.gru> to point at it explicitly."
    )


def dump_raw_f32(path: Path, arr: np.ndarray) -> None:
    """Writes ``arr`` as raw little-endian f32 (no .npy header)."""
    a = np.ascontiguousarray(arr.astype("<f4"))
    a.tofile(str(path))


def dump_raw_u32(path: Path, arr: np.ndarray) -> None:
    """Writes ``arr`` as raw little-endian u32 (no .npy header)."""
    a = np.ascontiguousarray(arr.astype("<u4"))
    a.tofile(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pt-path",
        required=True,
        type=Path,
        help="Path to the upstream RMVPE .pt checkpoint (see fetch_rmvpe_pt.sh).",
    )
    ap.add_argument(
        "--upstream-src",
        required=True,
        type=Path,
        help="Path to a checkout of the upstream yxlllc/RMVPE (or "
        "Dream-High/RMVPE) repository — the nn.Module class is imported "
        "from '<upstream-src>/src/model.py'.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for hidden.f32 / argmax.u32 / meta.json.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pcm",
        type=Path,
        help="WAV/FLAC file to run the reference forward on (mono preferred; "
        "resampled to --sample-rate if needed).",
    )
    group.add_argument(
        "--canned",
        action="store_true",
        help="Use a deterministic canned 30 s sine sweep 100->800 Hz + 30 s "
        "silence PCM (no external file required — good for owner smoke).",
    )
    ap.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Target sample rate; must match the RMVPE training rate "
        "(default 16000).",
    )
    ap.add_argument(
        "--hop", type=int, default=160,
        help="Mel hop (default 160 samples == 10 ms at 16 kHz).",
    )
    ap.add_argument(
        "--n-fft", type=int, default=2048,
        help="STFT n_fft (default 2048, matches upstream config).",
    )
    ap.add_argument(
        "--n-mels", type=int, default=128,
        help="Mel bands (default 128, matches upstream config).",
    )
    ap.add_argument(
        "--canned-duration", type=float, default=30.0,
        help="--canned duration in seconds (default 30; first half sweep, "
        "second half silence).",
    )
    ap.add_argument(
        "--module-import", default=None,
        help="Explicit dotted-path to the upstream class (e.g. "
        "'src.model.E2E'); auto-discovers if unset.",
    )
    ap.add_argument(
        "--gru-attr", default=None,
        help="Explicit attribute path to the BiGRU submodule on the "
        "upstream model (e.g. 'gru' or 'rnn.bigru'); auto-discovers if unset.",
    )
    ap.add_argument(
        "--voiced-threshold", type=float, default=0.03,
        help="Sigmoid probability threshold below which a frame's peak class "
        "is reported as unvoiced (argmax = 0). Matches upstream + Vokra "
        "runtime (crates/vokra-models/src/f0/rmvpe.rs VOICED_THRESHOLD).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. PCM (canned deterministic or real WAV).
    if args.canned:
        pcm = canned_pcm(args.canned_duration, args.sample_rate)
        pcm_sha256 = sha256_bytes(pcm.tobytes())
        pcm_source = f"canned sweep 100->800 Hz + silence, {args.canned_duration}s"
    else:
        pcm = load_pcm(args.pcm, args.sample_rate)
        pcm_sha256 = sha256_file(args.pcm)
        pcm_source = str(args.pcm.resolve())

    print(
        f"[dumper] PCM: {len(pcm)} samples at {args.sample_rate} Hz "
        f"({len(pcm) / args.sample_rate:.2f} s), sha256={pcm_sha256[:16]}..."
    )

    # 2. Load upstream nn.Module + checkpoint.
    model = instantiate_upstream_module(
        args.module_import,
        args.upstream_src,
        args.hop,
        args.n_fft,
        args.n_mels,
        args.sample_rate,
    )

    ckpt = torch.load(str(args.pt_path), map_location="cpu", weights_only=False)
    # Upstream releases sometimes wrap the state_dict under 'model' or
    # 'state_dict'; try common wrappers before assuming raw state_dict.
    state_dict: dict[str, Any] | None = None
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "module", "params"):
            v = ckpt.get(key)
            if isinstance(v, dict) and all(hasattr(t, "shape") for t in v.values()):
                state_dict = v
                break
        if state_dict is None and all(hasattr(t, "shape") for t in ckpt.values()):
            state_dict = ckpt  # type: ignore[assignment]
    if state_dict is None:
        raise RuntimeError(
            f"could not locate a state_dict inside {args.pt_path}; "
            f"top-level type = {type(ckpt).__name__}; if the upstream fork "
            f"wraps weights under a nonstandard key, unwrap it before "
            f"passing to torch.load()"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[dumper] WARN: state_dict missing {len(missing)} keys: {missing[:5]}...")
    if unexpected:
        print(f"[dumper] WARN: state_dict unexpected {len(unexpected)} keys: {unexpected[:5]}...")

    model.eval()

    # 3. Register a pre-forward hook on the GRU to capture the post-CNN
    #    hidden state (BiGRU input == exactly what
    #    RMVPE::forward_from_hidden expects on the Vokra side).
    gru = discover_gru_submodule(model, args.gru_attr)
    captured: dict[str, torch.Tensor] = {}

    def gru_pre_hook(_module: torch.nn.Module, inputs: tuple) -> None:
        # inputs[0] is [B, T, feature_dim] (torch.nn.GRU with batch_first=True)
        # or [T, B, feature_dim] (batch_first=False, the PyTorch default).
        # We normalise to [B, T, feature_dim] before storing.
        t = inputs[0]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"GRU input[0] is {type(t).__name__}, expected Tensor")
        if t.dim() != 3:
            raise ValueError(
                f"GRU input[0] has {t.dim()} dims (shape {tuple(t.shape)}); "
                f"expected 3 dims [B, T, F] or [T, B, F]"
            )
        # nn.GRU exposes .batch_first as a bool attribute.
        if isinstance(_module, torch.nn.GRU) and not _module.batch_first:
            t = t.transpose(0, 1)  # [T, B, F] -> [B, T, F]
        captured["hidden"] = t.detach().cpu().clone()

    hook_handle = gru.register_forward_pre_hook(gru_pre_hook)

    # 4. Run the reference forward.
    audio = torch.from_numpy(pcm).unsqueeze(0)  # [1, N]
    with torch.no_grad():
        try:
            logits = model(audio)
        except Exception as e:
            hook_handle.remove()
            raise RuntimeError(
                f"upstream model forward failed on audio [1, {len(pcm)}]: {e}. "
                f"If your fork's forward signature is different (e.g. requires "
                f"a mel spectrogram directly, or a positional argument), adapt "
                f"the model(audio) call in dump_reference.py:main()."
            ) from e

    hook_handle.remove()

    # Some upstream forwards return (logits, hidden) or (mel, logits);
    # we expect logits to be [B, T, n_class] with n_class == 360.
    if isinstance(logits, tuple):
        # Pick the [B, T, 360] tensor by shape sniff.
        picked = None
        for t in logits:
            if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[-1] == 360:
                picked = t
                break
        if picked is None:
            raise RuntimeError(
                f"upstream forward returned a tuple of {len(logits)} tensors "
                f"but none has shape [B, T, 360]; adapt the extraction logic."
            )
        logits = picked
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"upstream forward returned {type(logits).__name__}, expected Tensor")

    if "hidden" not in captured:
        raise RuntimeError(
            "GRU pre-forward hook did not fire — the upstream forward did not "
            "call the discovered GRU submodule. Pass --gru-attr to point at the "
            "correct submodule."
        )

    hidden = captured["hidden"]  # [1, T, feature_dim]
    if hidden.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got shape {tuple(hidden.shape)}")
    hidden = hidden.squeeze(0)   # [T, feature_dim]

    n_frames = int(hidden.shape[0])
    feature_dim = int(hidden.shape[1])

    # Logits shape sanity: [B, T, n_class] where T == n_frames and
    # n_class == 360 (upstream RMVPE head width). Some upstreams emit
    # sigmoid probabilities directly; others emit pre-sigmoid logits and
    # the caller applies sigmoid. Either way, argmax is invariant to a
    # monotonic point-wise transform, so we compute argmax on the raw
    # output.
    if logits.dim() != 3 or logits.shape[0] != 1 or logits.shape[2] != 360:
        raise ValueError(
            f"upstream logits shape {tuple(logits.shape)} != expected "
            f"[1, {n_frames}, 360]; either the head width is nonstandard "
            f"or the frame axis is not aligned with the GRU hidden state"
        )
    if int(logits.shape[1]) != n_frames:
        raise ValueError(
            f"upstream logits frame count {int(logits.shape[1])} != GRU "
            f"hidden frame count {n_frames}; forward is inconsistent"
        )

    # Per-frame argmax with a voiced gate — a peak below
    # `voiced_threshold` (after sigmoid) reports 0 (unvoiced), matching
    # upstream + Vokra runtime convention (rmvpe.rs `decode_class_to_hz`).
    probs = torch.sigmoid(logits.squeeze(0))  # [T, 360]
    peak_prob, peak_idx = probs.max(dim=-1)   # [T], [T]
    argmax = torch.where(
        peak_prob >= args.voiced_threshold,
        peak_idx.to(torch.int64),
        torch.zeros_like(peak_idx, dtype=torch.int64),
    )

    # 5. Dump raw buffers + meta.
    hidden_np = hidden.numpy().astype("<f4")
    argmax_np = argmax.numpy().astype("<u4")

    hidden_path = args.out_dir / "hidden.f32"
    argmax_path = args.out_dir / "argmax.u32"
    meta_path = args.out_dir / "meta.json"

    dump_raw_f32(hidden_path, hidden_np)
    dump_raw_u32(argmax_path, argmax_np)

    pt_sha256 = sha256_file(args.pt_path)
    upstream_module = getattr(type(model), "__module__", "?")
    upstream_class = type(model).__name__

    meta = {
        "dumper_version": DUMPER_VERSION,
        "n_frames": n_frames,
        "feature_dim": feature_dim,
        "n_class": 360,
        "sample_rate": args.sample_rate,
        "hop": args.hop,
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
        "voiced_threshold": args.voiced_threshold,
        "upstream_module": upstream_module,
        "upstream_class": upstream_class,
        "pt_path": str(args.pt_path.resolve()),
        "pt_sha256": pt_sha256,
        "pcm_source": pcm_source,
        "pcm_sha256": pcm_sha256,
        "voiced_frame_count": int((argmax_np != 0).sum()),
        "path_b_env": {
            "VOKRA_RMVPE_REAL_HIDDEN": str(hidden_path.resolve()),
            "VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM": str(feature_dim),
            "VOKRA_RMVPE_REAL_ARGMAX": str(argmax_path.resolve()),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(
        f"[dumper] wrote hidden.f32 ({hidden_np.nbytes} bytes, "
        f"[{n_frames}, {feature_dim}])"
    )
    print(
        f"[dumper] wrote argmax.u32 ({argmax_np.nbytes} bytes, "
        f"[{n_frames}], voiced={int((argmax_np != 0).sum())} / {n_frames})"
    )
    print(f"[dumper] wrote meta.json: {meta_path}")
    print("[dumper] export the following env vars into your parity run:")
    print(f'  export VOKRA_RMVPE_REAL_HIDDEN="{hidden_path.resolve()}"')
    print(f'  export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM="{feature_dim}"')
    print(f'  export VOKRA_RMVPE_REAL_ARGMAX="{argmax_path.resolve()}"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
