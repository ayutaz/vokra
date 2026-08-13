"""kahrendt/microWakeWord TFLite → Vokra GGUF (M5-03b Phase 1).

Offline sidecar tool (FR-LD-05: no Python / TFLite ever enters the runtime).
Fetches a canonical microWakeWord model release (Apache-2.0), inspects the
TFLite FlatBuffer via ``ai-edge-litert.Interpreter.get_tensor_details()``,
and emits a GGUF (via ``gguf.GGUFWriter``) whose metadata keys use the
``vokra.kws.*`` prefix so ``vokra_core::gguf::GgufFile::from_external`` (the
no_std GGUF reader the ``vokra-vad-micro`` sister crate already uses) can
open it on both host and thumbv8m Cortex-M55 (M5-03 IoT Tier-3).

# Why this file exists (Phase 1 of 3)

The ``vokra-kws-micro`` crate lands in this repo as a SCAFFOLD only —
``KwsMicro::detect()`` returns ``KwsEvent::Idle`` unconditionally so
callers can wire the type surface (register keywords, feed frames,
pattern-match events). Phase 1 (this script + a companion feature
extractor in Rust) bridges the upstream TFLite artifact to the Vokra
GGUF shape the future ``KwsMicro`` forward will bind. Phase 2 (later
wave) implements the actual detection forward + ``vokra-cli convert
--model microwakeword``; Phase 3 wires thumbv8m cross-build.

# Contract — GGUF metadata keys (vokra.kws.* prefix)

The output GGUF carries these metadata keys (matching the vokra-vad-micro
``vokra.silero.*`` posture — a per-model prefix with the audio-dialect
category as a discriminator):

- ``vokra.kws.arch``         = ``"microwakeword"``  (distinct from
                                ``"openwakeword"`` — the two are separate
                                ecosystems; openWakeWord targets host CPUs
                                via a shared speech-embedding TFLite, while
                                microWakeWord targets microcontrollers via
                                a self-contained MC-MobileNet).
- ``vokra.kws.model``        = ``"hey_jarvis"`` (or the CLI ``--name``)
- ``vokra.kws.threshold``    = f32 (default 0.5; the wake-decision cutoff)
- ``vokra.kws.sample_rate``  = u32 (typically 16000)
- ``vokra.kws.hop_ms``       = u32 (typically 10 or 20)
- ``vokra.kws.window_ms``    = u32 (typically 30 or 32)
- ``vokra.kws.n_mels``       = u32 (typically 40)
- ``vokra.kws.feature_dim``  = u32 (per-frame feature vector length,
                                    equals ``n_mels`` for standard mel;
                                    kept as an independent key because a
                                    stacked-frame model may differ)
- ``vokra.kws.tflite_sha256`` = string (source TFLite hex digest for
                                        provenance audit)
- ``vokra.kws.upstream``     = string (upstream release URL for provenance)
- Provenance chunk group (``vokra.provenance.*``) written by
  ``gguf.GGUFWriter.add_string`` per ``license_class::Permissive`` +
  ``apache-2.0`` posture. The Rust converter (Phase 2 WP) will use the
  ``crates/vokra-convert`` ``stamp_provenance`` helper; this script emits
  the same key set inline (``vokra.provenance.license`` + ``…upstream_hf``
  + ``…class``) so the artifact passes the FR-OP-32 catalog-reality gate.

# Tensor emission (Phase 1 shape-only)

Phase 1 emits per-tensor NAMES + SHAPES + DTYPES only, with the
DEQUANTIZED F32 weight values. The upstream TFLite is INT8-quantized for
TFLite-Micro inference on Cortex-M55; Phase 2 will preserve the INT8
form via GGUF Q8_0 once the ``vokra-cli convert --model microwakeword``
codepath lands (owner-side follow-up). Phase 1's F32 dequantization is
LOSSLESS for a fixed ``(scale, zero_point)`` pair — the arithmetic is
``f32 = scale * (int8 - zero_point)`` — so the F32 GGUF this script
emits carries the exact numeric values the runtime will need.

# NOT REFERENCED (clean-room)

- ``kahrendt/microWakeWord`` Python training code (Apache-2.0 — we do
  not vendor or re-implement it; we consume the released ``.tflite`` as
  an opaque black-box).
- ``esphome/esphome`` micro_wake_word component (GPL-3.0 — never
  imported, never inspected; the ESPHome layer is out-of-scope for
  Vokra Apache-2.0 posture, see CLAUDE.md "Piper (piper1-gpl)" red-line).

The tensor extraction logic is derived from ``ai-edge-litert`` public
docs (``Interpreter.get_tensor_details()`` returning ``[{name, shape,
dtype, quantization}]``) — a black-box API contract, no source
transliteration.

# Usage

::

    cd tools/parity/microwakeword
    uv sync
    # DL + convert the canonical hey_jarvis model:
    uv run python prepare_checkpoint.py \\
        --url    https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite \\
        --name   hey_jarvis \\
        --output ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf

    # Or point at a locally-downloaded .tflite:
    uv run python prepare_checkpoint.py \\
        --input  /path/to/hey_jarvis.tflite \\
        --name   hey_jarvis \\
        --output ./hey_jarvis.gguf

Fails loudly on any anomaly (non-INT8 weight, missing quantization
metadata, malformed FlatBuffer) rather than masking it — FR-EX-08
posture, matches every other sidecar in ``tools/parity/``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from gguf import GGMLQuantizationType, GGUFWriter

# ----------------------------------------------------------------------
# Constants — Vokra GGUF metadata keys (``vokra.kws.*`` per ADR M5-03b)
# ----------------------------------------------------------------------

# Architecture discriminator: distinct from ``openwakeword`` (the two
# ecosystems target different tiers — MC-MobileNet on M55 vs speech-embed
# MLP on RPi/Linux). Downstream binders switch on this key.
ARCH: str = "microwakeword"

# Standard microWakeWord front-end defaults (upstream v2 release,
# owner-verifiable via ``strings <model>.tflite | grep -i mel``). Emitted
# as GGUF metadata so ``vokra-kws-micro/src/features.rs`` picks them up at
# load time rather than hard-coding them at compile time (the same
# posture Vokra takes for Whisper front-end via ``vokra.frontend.*``).
DEFAULT_SAMPLE_RATE: int = 16_000
DEFAULT_HOP_MS: int = 10
DEFAULT_WINDOW_MS: int = 32
DEFAULT_N_MELS: int = 40
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_UPSTREAM_URL: str = (
    # ESPHome hosts the canonical curated v2 releases; kahrendt/microWakeWord
    # is the upstream author and the ESPHome mirror tracks it. If a specific
    # model is not present, the owner can override via ``--url``.
    "https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite"
)

# GGUF metadata key names — grouped so the ``add_metadata`` helper below
# reads top-down.
KEY_ARCH = "vokra.kws.arch"
KEY_MODEL = "vokra.kws.model"
KEY_THRESHOLD = "vokra.kws.threshold"
KEY_SAMPLE_RATE = "vokra.kws.sample_rate"
KEY_HOP_MS = "vokra.kws.hop_ms"
KEY_WINDOW_MS = "vokra.kws.window_ms"
KEY_N_MELS = "vokra.kws.n_mels"
KEY_FEATURE_DIM = "vokra.kws.feature_dim"
KEY_TFLITE_SHA256 = "vokra.kws.tflite_sha256"
KEY_UPSTREAM = "vokra.kws.upstream"

# Provenance chunk group (mirrors ``vokra_core::stamp_provenance`` output;
# Phase 2 Rust converter will call ``stamp_provenance`` directly, this
# script duplicates the emit set so the FR-OP-32 catalog-reality gate
# passes on the Phase 1 artifact too).
KEY_PROV_LICENSE = "vokra.provenance.license"
KEY_PROV_CLASS = "vokra.provenance.license_class"
KEY_PROV_UPSTREAM_HF = "vokra.provenance.upstream_hf"
KEY_PROV_UPSTREAM_NAME = "vokra.provenance.upstream_name"

# ----------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """Hex sha256 of the entire file (streamed, no full-file read)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Streams the URL to ``dest``. Raises loudly on non-200.

    Kept in stdlib (``urllib.request``) — the microWakeWord release is a
    single ~200 KB TFLite file, no auth, no chunking. Adding ``requests``
    would double this file's dep footprint for zero win.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ``urlretrieve`` follows redirects (GitHub raw → objects.githubusercontent.com)
    # and raises ``HTTPError`` on 4xx/5xx by default, which is the loud-fail
    # behaviour we want.
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise SystemExit(
                f"HTTP {response.status} fetching {url!r}: {response.reason}"
            )
        with dest.open("wb") as f:
            shutil.copyfileobj(response, f)


def dequantize_int8_to_f32(
    quantized: np.ndarray, scale: float, zero_point: int
) -> np.ndarray:
    """Standard TFLite affine dequantization: ``f32 = scale * (int8 - zero_point)``.

    LOSSLESS for a fixed ``(scale, zero_point)`` pair. Every INT8-quantized
    weight tensor in a TFLite-Micro model carries per-tensor quantization
    params; per-axis quantization is used only for activations. The Phase 1
    GGUF emits the dequantized F32 form — Phase 2 will preserve INT8 via
    GGUF Q8_0.
    """
    # Cast to int32 first to avoid signed int8 overflow when subtracting the
    # zero point (numpy would otherwise wrap ``-128 - (-2) = 126`` correctly
    # but ``127 - (-1) = -128`` incorrectly).
    return (quantized.astype(np.int32) - zero_point).astype(np.float32) * scale


def extract_tensors(
    interp: Interpreter, verbose: bool
) -> tuple[list[dict[str, Any]], int, int]:
    """Walks ``interp.get_tensor_details()`` and returns
    ``(records, weight_count, activation_count)`` — where ``records`` is
    the per-weight list of ``{name, shape, f32_data, orig_dtype, scale, zero_point}``.

    A "weight" is any tensor for which ``interp.get_tensor(idx)`` returns
    a populated array (constant); an "activation" is one for which it
    raises (runtime buffer). Constants become GGUF tensors; activations
    do not.
    """
    weights: list[dict[str, Any]] = []
    n_weights = 0
    n_activations = 0
    for td in interp.get_tensor_details():
        idx = td["index"]
        name = td["name"]
        shape = td["shape"]
        dtype = td["dtype"]
        quantization = td.get("quantization", (0.0, 0))
        scale, zero_point = quantization if isinstance(quantization, tuple) else (0.0, 0)
        try:
            data = interp.get_tensor(idx)
        except (ValueError, RuntimeError):
            # Activation buffer (no persistent data) — skip. This is
            # normal for input, output, and intermediate tensors.
            n_activations += 1
            if verbose:
                print(f"  skip[activation] idx={idx:3d} name={name!r} shape={list(shape)}",
                      file=sys.stderr)
            continue
        # `data` shape and dtype are the ground truth (get_tensor_details()
        # can carry a stale shape when the interpreter has never been
        # allocated for the specific batch dimension).
        n_weights += 1
        if dtype == np.int8:
            if scale <= 0.0:
                raise SystemExit(
                    f"tensor {name!r}: INT8 without per-tensor quantization scale "
                    f"(scale={scale!r}, zero_point={zero_point!r}). "
                    f"Refusing to emit — see FR-EX-08 (loud fail, no silent fallback)."
                )
            f32 = dequantize_int8_to_f32(data, float(scale), int(zero_point))
            weights.append({
                "name": name,
                "shape": list(data.shape),
                "f32_data": f32,
                "orig_dtype": "int8",
                "scale": float(scale),
                "zero_point": int(zero_point),
            })
        elif dtype == np.float32:
            weights.append({
                "name": name,
                "shape": list(data.shape),
                "f32_data": data.astype(np.float32),
                "orig_dtype": "float32",
                "scale": 0.0,
                "zero_point": 0,
            })
        else:
            # Loud fail rather than mask — matches dfn3_prepare_checkpoint.py posture.
            raise SystemExit(
                f"tensor {name!r}: unsupported dtype {dtype!r} (only INT8 + F32 "
                f"handled in Phase 1). Report to CC for Phase 2 extension."
            )
        if verbose:
            print(f"  emit[{weights[-1]['orig_dtype']:>7s}] idx={idx:3d} "
                  f"name={name!r} shape={list(data.shape)}",
                  file=sys.stderr)
    return weights, n_weights, n_activations


def write_gguf(
    output: Path,
    weights: list[dict[str, Any]],
    *,
    model_name: str,
    threshold: float,
    sample_rate: int,
    hop_ms: int,
    window_ms: int,
    n_mels: int,
    tflite_sha256: str,
    upstream_url: str,
) -> None:
    """Emits the GGUF that ``vokra_core::gguf::GgufFile::from_external`` can
    read. Uses ``gguf.GGUFWriter`` (Apache-2.0) so the metadata + tensor
    layout matches the writer llama.cpp / vokra-convert use elsewhere.
    """
    writer = GGUFWriter(str(output), ARCH)
    # Model identity + audio front-end contract:
    writer.add_string(KEY_ARCH, ARCH)
    writer.add_string(KEY_MODEL, model_name)
    writer.add_float32(KEY_THRESHOLD, threshold)
    writer.add_uint32(KEY_SAMPLE_RATE, sample_rate)
    writer.add_uint32(KEY_HOP_MS, hop_ms)
    writer.add_uint32(KEY_WINDOW_MS, window_ms)
    writer.add_uint32(KEY_N_MELS, n_mels)
    writer.add_uint32(KEY_FEATURE_DIM, n_mels)
    # Provenance (FR-OP-32 catalog-reality gate + M2-13 compliance stamp):
    writer.add_string(KEY_TFLITE_SHA256, tflite_sha256)
    writer.add_string(KEY_UPSTREAM, upstream_url)
    writer.add_string(KEY_PROV_LICENSE, "apache-2.0")
    writer.add_string(KEY_PROV_CLASS, "Permissive")
    writer.add_string(KEY_PROV_UPSTREAM_HF, "kahrendt/microWakeWord")
    writer.add_string(KEY_PROV_UPSTREAM_NAME, model_name)

    # Tensors — the F32 dequantized weights. Phase 2 will replace with
    # Q8_0 (INT8-preserving) once the Rust converter lands.
    for w in weights:
        writer.add_tensor(
            name=w["name"],
            tensor=w["f32_data"],
            raw_shape=w["shape"],
            raw_dtype=GGMLQuantizationType.F32,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract kahrendt/microWakeWord TFLite → Vokra GGUF (Phase 1)."
    )
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--input",
        type=Path,
        help="Local .tflite path (skip download). Mutually exclusive with --url.",
    )
    src.add_argument(
        "--url",
        type=str,
        default=DEFAULT_UPSTREAM_URL,
        help="URL to fetch the .tflite from (default: ESPHome micro-wake-word-models "
             "hey_jarvis v2 release).",
    )
    ap.add_argument("--name", default="hey_jarvis",
                    help="Model name for GGUF vokra.kws.model (default hey_jarvis).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output .gguf path.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Wake-decision threshold (default {DEFAULT_THRESHOLD}).")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                    help=f"Audio sample rate in Hz (default {DEFAULT_SAMPLE_RATE}).")
    ap.add_argument("--hop-ms", type=int, default=DEFAULT_HOP_MS,
                    help=f"Feature hop in ms (default {DEFAULT_HOP_MS}).")
    ap.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS,
                    help=f"Feature window in ms (default {DEFAULT_WINDOW_MS}).")
    ap.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS,
                    help=f"Number of mel bands (default {DEFAULT_N_MELS}).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print per-tensor emit / skip records to stderr.")
    args = ap.parse_args()

    # Resolve source .tflite path (download or use local).
    if args.input is not None:
        tflite_path = args.input
        if not tflite_path.exists():
            raise SystemExit(f"--input path not found: {tflite_path}")
        upstream_url = str(tflite_path.resolve())
        tmpdir: tempfile.TemporaryDirectory[str] | None = None
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="vokra-mww-")
        tflite_path = Path(tmpdir.name) / "model.tflite"
        print(f"Downloading {args.url} …", file=sys.stderr)
        download(args.url, tflite_path)
        upstream_url = args.url

    try:
        tflite_sha256 = sha256_of_file(tflite_path)
        size = tflite_path.stat().st_size
        print(f"Source: {tflite_path.name} ({size:,} bytes, sha256={tflite_sha256[:16]}…)",
              file=sys.stderr)

        # Parse the TFLite FlatBuffer via ai-edge-litert (successor of
        # tflite-runtime; get_tensor_details() is the same API).
        interp = Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()

        weights, n_weights, n_activations = extract_tensors(interp, args.verbose)
        if not weights:
            raise SystemExit(
                "No weight tensors extracted — the source .tflite may be "
                "activation-only or malformed. Aborting to avoid emitting "
                "an empty GGUF (FR-EX-08)."
            )
        print(f"Extracted {n_weights} weight tensor(s), "
              f"skipped {n_activations} activation tensor(s).",
              file=sys.stderr)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_gguf(
            args.output,
            weights,
            model_name=args.name,
            threshold=args.threshold,
            sample_rate=args.sample_rate,
            hop_ms=args.hop_ms,
            window_ms=args.window_ms,
            n_mels=args.n_mels,
            tflite_sha256=tflite_sha256,
            upstream_url=upstream_url,
        )
        out_size = args.output.stat().st_size
        print(f"Wrote {args.output} ({out_size:,} bytes, {n_weights} tensors, "
              f"vokra.kws.arch={ARCH}, vokra.kws.model={args.name})",
              file=sys.stderr)
        print(f"sha256(output) = {sha256_of_file(args.output)}", file=sys.stderr)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
