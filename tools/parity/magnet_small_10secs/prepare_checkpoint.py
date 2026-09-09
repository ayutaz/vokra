#!/usr/bin/env python3
"""Bridge facebook/magnet-small-10secs upstream bundle → flat safetensors.

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). The upstream Meta AudioCraft ``facebook/magnet-small-10secs``
release ships as a **compressed torch pickle bundle** (`state_dict.bin`)
containing the 300M-parameter non-autoregressive masked-LM
transformer, plus a bundled EnCodec 32 kHz audio codec, plus a frozen
T5-base text encoder — total 1,076,848,566 bytes per authenticated HF
primary source. The Rust converter
(``crates/vokra-convert/src/models/magnet_small_10secs.rs``) consumes
**single-file safetensors** by design — the runtime never grows a
shard-index reader or a pickle parser (NFR-DS-02 zero-dep + FR-LD-05
no pickle in runtime). This script bridges the two by:

1. **Reading** a previously authenticated VAST-local release directory;
   this sidecar never performs model/source downloads itself.
2. **Loading** the exact authenticated ``state_dict.bin``
   via the one permitted ``torch.load(..., weights_only=True)`` path.
   Unsupported/failed safe loading is a hard error; this bridge never
   retries with an unsafe pickle loader. Native safetensors and mirror
   substitutions are deliberately not accepted.
3. **Dedupes shared-storage tensors** via a ``data_ptr → canonical
   name`` pass (memory ``[[reference-safetensors-shared-tensor-dedup]]``):
   ``safetensors.torch.save_file`` refuses two names pointing at the
   same storage, so every duplicate is cloned + made contiguous into
   genuinely independent storage. The alias graph is preserved in
   ``<output>.shared_pairs.json`` so a downstream runtime binder can
   restore ties (e.g. tied text embedding + lm_head — an audiocraft-
   family posture).
4. **Drops non-float training-scaffold entries** explicitly and reports
   each one:

   - ``.num_batches_tracked`` BatchNorm I64 counters (no inference
     role — eval-mode BatchNorm consumes only running_mean /
     running_var / weight / bias).
   - ``.total_ops`` / ``.total_params`` ``torch.profiler`` bookkeeping.

5. **Rejects unexpected non-float dtypes** (I32 / I64 / F64 / Bool)
   that are NOT in the drop-list — FR-EX-08 no silent fallback. This
   catches upstream state that a future runtime binder needs to know
   about explicitly.
6. **F32 / F16 / BF16 pass through** under their upstream dtype (the
   Rust converter's ``GgmlType::F32 | GgmlType::F16 | GgmlType::BF16``
   pass-through arm handles all three; the runtime widens BF16 → f32
   losslessly at load via ``crates/vokra-core/src/gguf/quant/mod.rs
   decode_bf16``).
7. **Emits** a ``<output>.sha256`` line + parameter count + tensor
   count to stdout for the fixture / workflow logs.

# Memory footprint

The checkpoint and its conversion are **VAST-only**: the aggregate
checkpoint artefact is at/above the repository's 2 GB threshold. Do not
download, pickle-load, prepare, or convert it on the maintainer Mac.
The VAST worker performs the license and lock gates before creating a
work directory, syncing, or acquiring any model/source data.

# Usage

Managed through ``uv`` per the tools/parity contract (memory
``[[feedback-python-uses-uv]] / [[feedback-python-3-12]]``).

::

    cd tools/parity/magnet_small_10secs
    # Execute only on an authorized VAST worker after its preflight gate.
    uv sync

    # From the authenticated three-file bundle collected by the VAST worker:
    uv run python prepare_checkpoint.py \\
        --input-dir /path/to/magnet-small-10secs \\
        --output    /path/to/magnet-small-10secs/flat.safetensors

    # The resulting file is handed to a separately authorized converter
    # workflow. This sidecar never converts, uploads, or publishes artifacts.

# Rust converter contract

The output of this script is fed to
``crates/vokra-convert/src/models/magnet_small_10secs.rs`` unchanged.
That converter walks the safetensors file, pass-through-encodes each
F32 / F16 / BF16 tensor into GGUF, and stamps ``vokra.model.arch =
"magnet_small_10secs"`` + ``vokra.model.category = "music"`` +
``vokra.provenance.upstream_hf = "facebook/magnet-small-10secs"`` +
``vokra.provenance.license = "cc-by-nc-4.0"`` +
``vokra.provenance.weight_license = "noncommercial"``. Conversion and
publication are separate, owner-authorized workflows; this sidecar does
not publish or upload artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


DROPPED_SUFFIXES = (
    # BatchNorm training-scaffold counters (no inference role).
    ".num_batches_tracked",
    # torch.profiler bookkeeping (no inference role).
    ".total_ops",
    ".total_params",
)

ACCEPTED_FLOAT_DTYPES = frozenset(
    {"torch.float32", "torch.float16", "torch.bfloat16"}
)


def _dtype_name(t: Any) -> str:
    """Return a canonical ``torch.<name>`` string for a torch dtype."""
    return str(t.dtype) if hasattr(t, "dtype") else "unknown"


EXPECTED_BUNDLE = {
    "README.md": (10693, "e7fba2ce044a85fdcff253fa250e661044d9071d6f5033e5eab3f2ca42ce16e4"),
    "compression_state_dict.bin": (236003715, "91598c7da3d183eb8e0cc19cbbdc4f64f2d0c53069f9c8aa84185d0e33873c67"),
    "state_dict.bin": (840844851, "0594e551ed9c40464b5918f5ddcce348e491e912e61d69f4d5d64d4ddd1a6ade"),
}


def _reject_symlink_ancestors(path: Path) -> None:
    candidate = Path(os.path.abspath(path))
    while True:
        if candidate.is_symlink():
            raise SystemExit(f"prepare_checkpoint: symlink path component is forbidden: {candidate}")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def _validate_input_bundle(input_dir: Path | None) -> Path:
    if input_dir is not None:
        _reject_symlink_ancestors(input_dir)
    if input_dir is None or input_dir.is_symlink() or not input_dir.is_dir():
        raise SystemExit("prepare_checkpoint: --input-dir must be a real directory")
    entries = list(input_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise SystemExit("prepare_checkpoint: input bundle contains a symlink/non-regular entry")
    if {entry.name for entry in entries} != set(EXPECTED_BUNDLE):
        raise SystemExit("prepare_checkpoint: input bundle file set is not exact")
    for name, (size, expected) in EXPECTED_BUNDLE.items():
        path = input_dir / name
        if path.stat().st_size != size or _sha256_file(path) != expected:
            raise SystemExit(f"prepare_checkpoint: authenticated input identity mismatch: {name}")
    return input_dir


def _validate_output(output: Path, input_dir: Path) -> None:
    sidecars = (output.with_suffix(output.suffix + ".sha256"), output.with_suffix(output.suffix + ".shared_pairs.json"))
    _reject_symlink_ancestors(output)
    for sidecar in sidecars:
        _reject_symlink_ancestors(sidecar)
    if any(path.exists() or path.is_symlink() for path in (output, *sidecars)):
        raise SystemExit("prepare_checkpoint: output or sidecar already exists")
    input_real = input_dir.resolve()
    output_real = output.resolve(strict=False)
    if output_real == input_real or input_real in output_real.parents or output_real in input_real.parents:
        raise SystemExit("prepare_checkpoint: output overlaps authenticated input bundle")


def _load_state_dict(input_dir: Path | None, *, validated: bool = False) -> dict[str, Any]:
    """Load the upstream checkpoint into a state dict.

    The only accepted input is the authenticated ``state_dict.bin`` in the
    exact three-file release bundle.
    """
    if not validated:
        input_dir = _validate_input_bundle(input_dir)
    assert input_dir is not None
    import torch  # deferred (uv-managed)
    ckpt_path = input_dir / "state_dict.bin"
    print(f"[load] torch pickle: {ckpt_path}", file=sys.stderr)
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "prepare_checkpoint: torch.load(weights_only=True) is unsupported; "
            "refusing an unsafe retry"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "prepare_checkpoint: restricted torch.load(weights_only=True) failed; "
            "refusing an unsafe retry"
        ) from exc

    # audiocraft state_dict.bin sometimes wraps the raw state under a
    # ``"best_state"`` / ``"state_dict"`` / ``"model"`` key; unwrap.
    for k in ("best_state", "state_dict", "model", "state"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            print(f"[load]   unwrapped .{k}", file=sys.stderr)
            break

    if not isinstance(obj, dict):
        raise SystemExit(
            f"prepare_checkpoint: loaded object is {type(obj).__name__}, "
            f"expected dict-like state_dict"
        )
    return obj


def _dedup_and_filter(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str]], list[str], list[tuple[str, str]]]:
    """Dedupe shared storage + drop training-scaffold + reject unknowns.

    Returns (safe_state_dict, shared_pairs, dropped, rejected).
    """
    import torch  # deferred

    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []
    dropped: list[str] = []
    rejected: list[tuple[str, str]] = []
    safe_state: dict[str, Any] = {}

    for name, tensor in state_dict.items():
        # Drop training-scaffold explicitly.
        if any(name.endswith(suf) for suf in DROPPED_SUFFIXES):
            dropped.append(name)
            continue

        if not isinstance(tensor, torch.Tensor):
            # Non-tensor entries (e.g. lists of int) — reject loudly.
            rejected.append((name, f"non-tensor {type(tensor).__name__}"))
            continue

        # Accept float dtypes only. Reject anything else loudly.
        dt = _dtype_name(tensor)
        if dt not in ACCEPTED_FLOAT_DTYPES:
            rejected.append((name, dt))
            continue

        # Dedupe shared storage (safetensors.torch.save_file refuses
        # aliases). data_ptr() is 0 for meta / uninitialized tensors —
        # those get their own copies too.
        ptr = tensor.data_ptr()
        if ptr != 0 and ptr in seen:
            canonical = seen[ptr]
            shared_pairs.append((name, canonical))
            safe_state[name] = tensor.detach().clone().contiguous()
        else:
            if ptr != 0:
                seen[ptr] = name
            # Always contiguous — safetensors requires it.
            safe_state[name] = (
                tensor.detach().contiguous()
                if tensor.is_contiguous()
                else tensor.detach().clone().contiguous()
            )

    if rejected:
        # Report every rejected entry, then bail. FR-EX-08 no silent
        # fallback — a future binder must decide how to handle each.
        for name, why in rejected:
            print(f"[reject] {name}  ({why})", file=sys.stderr)
        raise SystemExit(
            f"prepare_checkpoint: {len(rejected)} unexpected non-float "
            f"tensors — refusing to write output (FR-EX-08)"
        )

    return safe_state, shared_pairs, dropped, rejected


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 of ``path``'s contents (streaming)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _self_test_body(bundle: Path) -> int:
    """Prove the restricted loader never retries after a TypeError."""
    import types

    checkpoint = bundle / "state_dict.bin"
    original_bundle = EXPECTED_BUNDLE.copy()
    EXPECTED_BUNDLE.clear()
    EXPECTED_BUNDLE.update({name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in (("README.md", b"readme"), ("compression_state_dict.bin", b"compression"), ("state_dict.bin", b"not-a-checkpoint"))})
    for name, data in (("README.md", b"readme"), ("compression_state_dict.bin", b"compression"), ("state_dict.bin", b"not-a-checkpoint")):
        bundle.joinpath(name).write_bytes(data)
    # An unrelated pickle-looking file must never affect exact selection.
    checkpoint.with_name("alternate.bin").write_bytes(b"not-a-checkpoint")
    try:
        _validate_input_bundle(bundle)
    except SystemExit:
        pass
    else:
        raise AssertionError("extra checkpoint file was accepted")
    checkpoint.with_name("alternate.bin").unlink()
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"tampered")
    try:
        _validate_input_bundle(bundle)
    except SystemExit:
        pass
    else:
        raise AssertionError("checkpoint hash/size tamper was accepted")
    checkpoint.write_bytes(original)
    input_link = bundle.with_name("bundle-link")
    input_link.symlink_to(bundle, target_is_directory=True)
    try:
        _validate_input_bundle(input_link)
    except SystemExit:
        pass
    else:
        raise AssertionError("input directory symlink was accepted")
    input_link.unlink()
    input_parent = bundle.parent / f"input-real-{bundle.name}"
    input_parent.mkdir()
    input_parent_link = bundle.parent / f"input-link-{bundle.name}"
    input_parent_link.symlink_to(input_parent, target_is_directory=True)
    try:
        _validate_input_bundle(input_parent_link / "bundle")
    except SystemExit:
        pass
    else:
        raise AssertionError("input bundle under symlink ancestor was accepted")
    input_parent_link.unlink()
    output = bundle.parent / "flat.safetensors"
    _validate_output(output, bundle)
    output.with_suffix(output.suffix + ".sha256").write_text("occupied", encoding="utf-8")
    try:
        _validate_output(output, bundle)
    except SystemExit:
        pass
    else:
        raise AssertionError("existing output sidecar was accepted")
    output.with_suffix(output.suffix + ".sha256").unlink()
    output.symlink_to(bundle / "missing-output")
    try:
        _validate_output(output, bundle)
    except SystemExit:
        pass
    else:
        raise AssertionError("dangling output symlink was accepted")
    output.unlink()
    output_parent = bundle.parent / f"output-real-{bundle.name}"
    output_parent.mkdir()
    output_parent_link = bundle.parent / f"output-link-{bundle.name}"
    output_parent_link.symlink_to(output_parent, target_is_directory=True)
    try:
        _validate_output(output_parent_link / "nested" / "flat.safetensors", bundle)
    except SystemExit:
        pass
    else:
        raise AssertionError("output under symlink ancestor was accepted")
    output_parent_link.unlink()
    try:
        _validate_output(bundle / "nested" / "flat.safetensors", bundle)
    except SystemExit:
        pass
    else:
        raise AssertionError("overlapping output was accepted")
    calls = 0

    def load(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise TypeError("fake torch without weights_only")

    fake_torch = types.ModuleType("torch")
    fake_torch.load = load  # type: ignore[attr-defined]
    previous = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        try:
            _load_state_dict(checkpoint.parent)
        except RuntimeError as exc:
            assert "unsafe retry" in str(exc)
        else:
            raise AssertionError("restricted loader unexpectedly succeeded")
        assert calls == 1, f"torch.load called {calls} times"
        checkpoint.rename(bundle / "state_dict.bad")
        try:
            _validate_input_bundle(bundle)
        except SystemExit:
            pass
        else:
            raise AssertionError("missing exact state_dict.bin was accepted")
    finally:
        EXPECTED_BUNDLE.clear()
        EXPECTED_BUNDLE.update(original_bundle)
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous
    print("magnet-small-10secs prepare self-test: PASS")
    return 0


def _self_test() -> int:
    import tempfile

    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="magnet-small-prepare-", dir=temporary_root) as directory:
        return _self_test_body(Path(directory))


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the upstream checkpoint bundle "
        "exact state_dict.bin is required; discovery is disabled.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the flat safetensors output file.",
    )
    args = ap.parse_args()

    if args.input_dir is None:
        print("prepare_checkpoint: --input-dir is required", file=sys.stderr)
        return 2

    input_dir = _validate_input_bundle(args.input_dir)
    _validate_output(args.output, input_dir)
    state_dict = _load_state_dict(input_dir, validated=True)
    print(
        f"[load] loaded {len(state_dict)} raw entries",
        file=sys.stderr,
    )

    safe_state, shared_pairs, dropped, _rejected = _dedup_and_filter(state_dict)
    print(
        f"[filter] {len(safe_state)} float tensors, "
        f"{len(shared_pairs)} shared aliases, "
        f"{len(dropped)} training-scaffold dropped",
        file=sys.stderr,
    )
    for name in dropped:
        print(f"[drop]  {name}", file=sys.stderr)
    for alias, canonical in shared_pairs:
        print(f"[alias] {alias}  ->  {canonical}", file=sys.stderr)

    # Serialize.
    from safetensors.torch import save_file as safe_save

    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_save(safe_state, str(args.output))
    print(f"[write] {args.output}", file=sys.stderr)

    # Sidecar sha256 + shared-pair audit trail.
    sha = _sha256_file(args.output)
    (args.output.with_suffix(args.output.suffix + ".sha256")).write_text(
        f"{sha}  {args.output.name}\n"
    )
    if shared_pairs:
        (args.output.with_suffix(args.output.suffix + ".shared_pairs.json")).write_text(
            json.dumps(
                [{"alias": a, "canonical": c} for a, c in shared_pairs],
                indent=2,
            )
            + "\n"
        )

    # Summary line for the CI log.
    total_params = sum(t.numel() for t in safe_state.values())
    print(
        f"[done] wrote {args.output.name}: "
        f"{len(safe_state)} tensors / {total_params:,} params / sha256 {sha[:16]}...",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
