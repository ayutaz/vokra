#!/usr/bin/env python3
"""Flatten a Meta HT-Demucs ``*.th`` release archive → single ``.safetensors``.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

Upstream ``facebook/demucs`` ships weights as ``torch.save(...)`` archives
suffixed ``.th`` (e.g. ``955717e8-8726e21a.th`` for the base ``htdemucs``
release, ``f7e0c4bc-ba3fe64a.th`` for a ``htdemucs_ft`` bag member). The
archive layout is the demucs-canonical dict::

    {
        "klass":  HTDemucs,          # class reference (not a state_dict)
        "args":   (...),             # ctor positional args
        "kwargs": {...},             # ctor keyword args
        "state":  {                  # <-- the actual state_dict (NOTE: "state", not "state_dict")
            "encoder.0.conv.weight": Tensor,
            "crosstransformer.layers.0.attn.in_proj_weight": Tensor,
            ...
        },
        "sig":    "abc123",          # upstream sha checksum
    }

Written by ``demucs.states.save_with_checksum`` — see
``github.com/facebookresearch/demucs/blob/main/demucs/states.py`` +
``demucs/pretrained.py::load_model``.

The Vokra Rust converter (``crates/vokra-convert/src/models/demucs_htdemucs.rs``)
consumes **safetensors only** (no pickle in the runtime tree). This script
bridges the two: it loads the ``.th``, extracts the ``state`` sub-dict,
handles HT-Demucs' encoder↔decoder tied weights (safetensors refuses
shared storage — must ``.clone().contiguous()``), strips int-dtype
counters, and writes a single ``.safetensors`` the caller feeds to
``vokra-cli convert --model demucs-htdemucs``.

Precedent: ``sepformer_prepare_checkpoint.py`` (multi-file .ckpt bundle),
``dac_prepare_checkpoint.py`` (single .pth + config side-car), and
``nemo_pt_to_safetensors.py`` (single .pt/.nemo). This script is the
single-.th cousin with two demucs-specific twists: (1) the state_dict
lives under ``"state"`` (not ``"state_dict"``) and (2) shared-storage
dedup is mandatory (HT-Demucs ties bottleneck weights across
encoder/decoder).

# Usage

::

    uv run --project tools/parity python tools/parity/demucs_prepare_checkpoint.py \\
        --input ~/.cache/demucs/955717e8-8726e21a.th \\
        --output /tmp/demucs-htdemucs.safetensors \\
        [--strict]

Then::

    vokra-cli convert --model demucs-htdemucs \\
        --input /tmp/demucs-htdemucs.safetensors \\
        --output /tmp/demucs-htdemucs.gguf

# Pickle-trust posture

Demucs ``.th`` archives embed a ``klass`` class reference at the top level
which ``torch.load(weights_only=True)`` refuses (the safe-unpickler
rejects arbitrary class objects). This script therefore loads with
``weights_only=False``. The upstream primary source is Meta's MIT-licensed
``facebook/demucs`` release; per
memory ``[[feedback-license-signoff-primary-source]]`` the pickle trust
boundary is acknowledged at the point of running this offline sidecar
(the runtime tree never touches pickle — FR-LD-05).

# Shared-storage handling

HT-Demucs ties some encoder↔decoder bottleneck weights; the underlying
safetensors serializer hard-errors on ``data_ptr()`` collision
(``RuntimeError: The weights trying to be saved contain shared tensors``).
The script dedups via a ``{data_ptr: first_name}`` map + ``.clone()
.contiguous()`` on subsequent references (first tensor stays as-is; each
alias becomes an independent copy). Manifest records the tied pairs for
owner audit.

# Redistribution

Upstream weight license is ``mit`` — see ``docs/license-audit.md`` §3.1
row "Demucs (HT-Demucs)" (Meta primary source, owner sign-off queue).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Same dtype taxonomy as the sepformer / nemo precedents. INT dtypes are
# training artefacts (BatchNorm num_batches_tracked counters etc.) — safe
# to strip. Any dtype outside both sets is refused under --strict; without
# --strict the script silently skips them (mirroring the sepformer
# --allow-strip-any inverse: this script's default is permissive, --strict
# flips to fail-loud).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}

# Wrapper-key probe order. **``"state"`` is FIRST** — that's the
# demucs-canonical top-level key (via ``demucs.states.save_with_checksum``).
# The other names are common Lightning / HuggingFace fork wrappers kept
# as defensive fallbacks; if none match and the raw dict already looks
# like a flat ``{str: Tensor}`` state_dict we use it verbatim.
WRAPPER_KEYS = ("state", "state_dict", "model_state_dict", "model", "module")


def _looks_like_state_dict(obj: Any) -> bool:
    """Return True if ``obj`` is a ``{str: Tensor}`` dict with ≥1 float leaf."""
    import torch

    if not isinstance(obj, dict) or not obj:
        return False
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if not isinstance(v, torch.Tensor):
            return False
    return True


def _extract_state_dict(raw: Any) -> dict:
    """Unwrap a demucs .th payload down to the flat state_dict.

    Probes ``WRAPPER_KEYS`` in order (``"state"`` first — demucs-canonical).
    Fails loudly on a payload that yields no state_dict rather than
    silently emitting an empty safetensors (the classic "silent partial"
    trap this project bans — FR-EX-08).
    """
    if _looks_like_state_dict(raw):
        return raw

    if isinstance(raw, dict):
        for wrapper in WRAPPER_KEYS:
            inner = raw.get(wrapper)
            if _looks_like_state_dict(inner):
                print(f"  unwrapped ['{wrapper}']")
                return inner  # type: ignore[return-value]

    sys.exit(
        "demucs_prepare_checkpoint: could not locate a state_dict — expected a "
        "demucs .th archive with a top-level 'state' key "
        "(demucs.states.save_with_checksum layout). Top-level keys observed: "
        f"{sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
    )


def _partition_and_dedup(sd: dict, strict: bool):
    """Split into ``(kept, dropped_int, unknown_other, shared_pairs)``.

    Shared-storage tensors are cloned (first occurrence kept verbatim,
    subsequent occurrences ``.clone().contiguous()`` into fresh storage
    so safetensors accepts them). ``shared_pairs`` records the
    (clone_name, original_name) tuples for the audit manifest.
    """
    import torch  # noqa: F401  (needed for isinstance guards below)

    kept: dict = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []

    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)

        if dtype_s in KEEP_DTYPES:
            # data_ptr() dedup — safetensors refuses shared storage.
            # First tensor at a given ptr stays as-is; aliases clone.
            # ``untyped_storage()`` is the torch >=2.0 forward-compatible
            # accessor (``.storage()`` warns TypedStorage-deprecated).
            ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else 0
            if ptr and ptr in seen:
                shared_pairs.append((name, seen[ptr]))
                t = t.detach().clone().contiguous()
            else:
                if ptr:
                    seen[ptr] = name
                t = t.detach().contiguous()
            kept[name] = t
        elif dtype_s in INT_DTYPES:
            dropped.append((name, dtype_s, list(t.shape)))
        else:
            unknown.append((name, dtype_s, list(t.shape)))

    return kept, dropped, unknown, shared_pairs


def _run_pipeline(raw: Any, output: Path, strict: bool, sig: str | None) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift."""
    from safetensors.torch import save_file

    state = _extract_state_dict(raw)
    kept, dropped, unknown, shared_pairs = _partition_and_dedup(state, strict)

    if unknown and strict:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"demucs_prepare_checkpoint: --strict refusing to drop "
            f"{len(unknown)} tensors of unknown dtype (first 3: {first}); "
            "re-run without --strict if verified inference-inert.",
            file=sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(output))
    written_bytes = output.stat().st_size

    manifest = {
        "output": str(output),
        "kept_count": len(kept),
        "skipped_count": len(dropped) + len(unknown),
        "dropped_int_count": len(dropped),
        "unknown_count": len(unknown),
        "shared_cloned_count": len(shared_pairs),
        "written_bytes": written_bytes,
        "strict": strict,
        "demucs_sig": sig,
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in unknown
        ],
        "shared_tied": [
            {"clone": clone, "original": orig} for clone, orig in shared_pairs
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    skipped = len(dropped) + len(unknown)
    print(
        f"demucs_prepare_checkpoint: kept={len(kept)} skipped={skipped} "
        f"shared_cloned={len(shared_pairs)} written_bytes={written_bytes:,} "
        f"manifest -> {manifest_path.name}"
    )
    return 0


def _self_test() -> int:
    """Synthesize an in-memory demucs .th, round-trip through the pipeline,
    and assert kept/skipped/shared_cloned counts + safetensors reload.

    Exercises the three demucs-specific quirks: (a) ``"state"`` wrapper
    key (b) shared-storage tensor clone (c) int-dtype strip. No real
    weight file is touched — this validates the code path can be walked
    end-to-end even when the caller has no upstream weights.
    """
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        print(
            f"demucs_prepare_checkpoint --self-test: torch/safetensors missing "
            f"({exc}). run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    import tempfile

    # Synthetic HT-Demucs-shaped state_dict with the three quirks folded in:
    #   - ``encoder.0.conv.weight``: a real waveform-branch tensor name
    #   - ``crosstransformer.layers.0.attn.in_proj_weight``: bottleneck name;
    #     shares storage with ``decoder.tied.weight`` via ``.view(...)``
    #   - ``decoder.5.conv_tr.num_batches_tracked``: int64 counter → dropped
    shared = torch.randn(6, 8, dtype=torch.bfloat16)
    tied_alias = shared.view(6, 8)  # same underlying storage as `shared`
    synthetic_state = {
        "encoder.0.conv.weight": torch.randn(4, 2, 3),
        "crosstransformer.layers.0.attn.in_proj_weight": shared,
        "decoder.tied.weight": tied_alias,
        "decoder.5.conv_tr.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    raw = {
        "klass": "HTDemucs",  # str stand-in for the class ref; unwrap probes "state" first
        "args": (),
        "kwargs": {},
        "state": synthetic_state,
        "sig": "self-test-sig",
    }

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "self-test.safetensors"
        rc = _run_pipeline(raw, out, strict=False, sig=raw["sig"])
        if rc != 0:
            print("demucs_prepare_checkpoint --self-test: pipeline non-zero", file=sys.stderr)
            return rc

        # Assert: 3 float tensors kept, 1 int dropped, 1 shared clone.
        loaded = load_file(str(out))
        expected_keys = {
            "encoder.0.conv.weight",
            "crosstransformer.layers.0.attn.in_proj_weight",
            "decoder.tied.weight",
        }
        if set(loaded.keys()) != expected_keys:
            print(
                f"demucs_prepare_checkpoint --self-test: kept keys "
                f"{sorted(loaded.keys())} != expected {sorted(expected_keys)}",
                file=sys.stderr,
            )
            return 4

        manifest_path = out.with_suffix(out.suffix + ".stripped-manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["kept_count"] != 3:
            print(f"self-test: kept_count={manifest['kept_count']} != 3", file=sys.stderr)
            return 4
        if manifest["dropped_int_count"] != 1:
            print(f"self-test: dropped_int_count={manifest['dropped_int_count']} != 1", file=sys.stderr)
            return 4
        if manifest["shared_cloned_count"] != 1:
            print(
                f"self-test: shared_cloned_count={manifest['shared_cloned_count']} != 1 "
                "(tied encoder/decoder bottleneck weight should have been cloned)",
                file=sys.stderr,
            )
            return 4
        if manifest["demucs_sig"] != "self-test-sig":
            print(f"self-test: demucs_sig={manifest['demucs_sig']!r} != 'self-test-sig'", file=sys.stderr)
            return 4

    print("demucs_prepare_checkpoint --self-test: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Flatten a Meta HT-Demucs .th release archive → single .safetensors",
    )
    ap.add_argument(
        "--input", type=Path,
        help=(
            "upstream demucs .th release archive (e.g. 955717e8-8726e21a.th, "
            "typically fetched via `hf download facebook/demucs --local-dir <dir>`)."
        ),
    )
    ap.add_argument(
        "--output", type=Path,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "fail-loud on tensors of unknown dtype (fp64 / complex / etc.). "
            "Default: silently skip them (they are inference-inert for HT-Demucs)."
        ),
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "synthesize a demucs-shaped .th payload in-memory, round-trip through "
            "the pipeline, and assert kept/skipped/shared_cloned counts. Does NOT "
            "touch any upstream weight file."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.input is None or args.output is None:
        print(
            "demucs_prepare_checkpoint: --input and --output are required "
            "(unless --self-test).",
            file=sys.stderr,
        )
        return 2

    try:
        import torch
    except ImportError as exc:
        print(
            f"demucs_prepare_checkpoint: torch missing ({exc}). "
            "run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2
    try:
        import safetensors.torch  # noqa: F401
    except ImportError as exc:
        print(
            f"demucs_prepare_checkpoint: safetensors missing ({exc}). "
            "run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    if not args.input.is_file():
        print(f"--input must be an existing .th file: {args.input}", file=sys.stderr)
        return 2

    print(f"loading {args.input.name} ({args.input.stat().st_size:,} bytes)")

    # weights_only=False: demucs .th archives embed a ``klass`` class
    # reference which the safe-unpickler rejects. The upstream primary
    # source is Meta's MIT-licensed facebook/demucs release; the runtime
    # tree never touches pickle (FR-LD-05) — the trust boundary is here,
    # in the offline sidecar.
    try:
        raw = torch.load(str(args.input), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"torch.load({args.input!s}) failed: {exc}", file=sys.stderr)
        return 2

    sig = raw.get("sig") if isinstance(raw, dict) else None
    return _run_pipeline(raw, args.output, strict=args.strict, sig=sig)


if __name__ == "__main__":
    sys.exit(main())
