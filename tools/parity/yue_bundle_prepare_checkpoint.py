#!/usr/bin/env python3
"""Merge YuE bundle (upsampler / xcodec-mini) checkpoint → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The YuE full-song music-generation system (Yuan et al. 2025,
arXiv:2503.08638, ``github.com/multimodal-art-projection/YuE``) publishes
its **codec / vocoder half** across two sibling HF repos, each of which
ships **only torch pickle** (``.pth`` / ``.bin``) — there is no
``model.safetensors`` mirror on either side (verified 2026-08-01 via
``https://huggingface.co/api/models/m-a-p/YuE-upsampler`` and
``https://huggingface.co/api/models/m-a-p/xcodec_mini_infer``):

===================================================   ======   ============================================
Variant                                              Size     Payload
===================================================   ======   ============================================
``upsampler``   (`m-a-p/YuE-upsampler`)              145 MB   Vocos backbone + iSTFT head — 44.1 kHz
                                                              vocoder decoding YuE codec latents
``xcodec-mini`` (`m-a-p/xcodec_mini_infer`)          2.2 GB   SoundStream RVQ codec + HuBERT-base
                                                              semantic encoder + Vocos decoder head
===================================================   ======   ============================================

The Rust converter (``crates/vokra-convert/src/models/yue_bundle.rs``)
consumes **single-file safetensors only** (no pickle in the runtime
tree, no multi-file reader). This script bridges the two: for a
caller-selected ``--variant`` it loads the corresponding sub-parts,
prefixes every key with its role (``codec.`` / ``semantic.`` /
``decoder.`` for xcodec-mini; bare ``backbone.`` / ``head.`` for
upsampler because there is only one sub-module) so the native
``YueXcodecMini::from_gguf`` binder can locate sub-modules, and writes a
single ``.safetensors`` the caller feeds to
``vokra-cli convert --model {yue-upsampler,yue-xcodec-mini}``.

Precedent: ``naturalspeech3_facodec_prepare_checkpoint.py`` (multi-file
cousin — same shape, same INT-dtype filter, same
``.stripped-manifest.json`` sidecar, same fail-loud posture) +
``sepformer_prepare_checkpoint.py`` (SpeechBrain ``.ckpt`` bundle) +
``bin_to_safetensors.py`` (single-file pickle → safetensors).

# Usage

::

    # Vocoder half (145 MB)
    uv run --project tools/parity python \\
        tools/parity/yue_bundle_prepare_checkpoint.py \\
        --ckpt-dir /path/to/m-a-p--YuE-upsampler \\
        --variant upsampler \\
        --snapshot 151000 \\
        --output /tmp/yue-upsampler.safetensors \\
        [--allow-strip-any]

    # Codec bundle (~1.81 GB selected weights; source repo total >2 GB)
    uv run --project tools/parity python \\
        tools/parity/yue_bundle_prepare_checkpoint.py \\
        --ckpt-dir /path/to/m-a-p--xcodec_mini_infer \\
        --variant xcodec-mini \\
        --snapshot 151000 \\
        --output /tmp/yue-xcodec-mini.safetensors \\
        [--allow-strip-any]

The upsampler is small enough for local metadata work.  The xcodec-mini
source set is larger than the repository's 2 GB aggregate-artifact guard,
so preparation, conversion, validation, and publication MUST run through
the documented vast.ai workflow rather than on the maintainer Mac.

# Snapshot selection (131k vs 151k)

Both upstream repos ship two training snapshots of the Vocos decoder
head: ``decoder_131000.pth`` and ``decoder_151000.pth``. These files
are **byte-identical** between the two sibling repos (same xet
content-address hash: ``c030b262…`` for 131k / ``70e4fbd9…`` for
151k, verified 2026-08-01 via HF API). The 151k snapshot is the
later training step and is typically the "final" one — default here.

# Determinism

Keys are ordered by (variant subset order, per-file dict-iteration
order — Python dict preserves insertion order since 3.7, and
``torch.load(weights_only=True)`` is deterministic). Identical
``--ckpt-dir`` + ``--variant`` + ``--snapshot`` input produces
byte-identical output (safetensors serialization is deterministic
for fixed key ordering).

# Sub-tree skip list (xcodec-mini)

The upstream ``m-a-p/xcodec_mini_infer`` repo ships full source-tree
copies of RepCodec and Descript-Audio-Codec at ``RepCodec/`` and
``descriptaudiocodec/dac/``. The bundled RepCodec license material is
mixed (MIT text plus CC-BY-NC-4.0 text/source headers), while DAC is MIT. These
are inference-tree artefacts of the upstream release process — they
are **not** loaded weights and this bridge does NOT recurse into them.
NOTICE credit is preserved (their code informed the YuE codec
design), but no tensors are lifted from them. Same policy for the
``__pycache__/`` bytecode cache leaked into the release.

# Redistribution

The two weight repositories declare apache-2.0 — see
``docs/license-audit.md`` §3.1 rows "YuE-upsampler" and
"YuE xcodec-mini". Upstream YuE code at
``github.com/multimodal-art-projection/YuE`` is also apache-2.0. This bridge
loads only the three fixed weight files; it neither imports nor redistributes
the bundled RepCodec/DAC source trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Which upstream files each --variant reads, and under what role
# prefix they land in the merged safetensors. The upsampler variant
# has only one sub-module (Vocos decoder head), so its role prefix
# is empty ("" = bare `backbone.*` / `head.*` names, matching the
# sibling `charactr/vocos-mel-24khz` and `charactr/vocos-encodec-24khz`
# tensor-name contract). The xcodec-mini variant has three
# sub-modules (SoundStream codec, HuBERT semantic encoder, Vocos
# decoder head), so each gets a distinct role prefix.
#
# Each tuple = (relative_path_template, role_prefix).
# {snapshot} is substituted with the --snapshot argument (131000 or
# 151000) for the .pth files that come in two training snapshots.
VARIANT_SUBSETS: dict[str, list[tuple[str, str]]] = {
    "upsampler": [
        ("decoder_{snapshot}.pth", ""),  # bare backbone.* / head.* names
    ],
    "xcodec-mini": [
        # SoundStream/XCodec checkpoint. The public wrapper uses the exact
        # top-level key `codec_model`; optimizer_g/optimizer_d are training
        # artefacts and must never enter a fresh conversion.
        ("final_ckpt/ckpt_00360000.pth", "codec"),
        # HuBERT-base semantic encoder (HF-transformers pickle).
        ("semantic_ckpts/hf_1_325000/pytorch_model.bin", "semantic"),
        # Vocos decoder head (byte-identical to the sibling repo's
        # decoder_{snapshot}.pth — same xet content-address hash).
        ("decoders/decoder_{snapshot}.pth", "decoder"),
    ],
}

# Mirrors the classification in ``nemo_pt_to_safetensors.py`` /
# ``sepformer_prepare_checkpoint.py`` /
# ``naturalspeech3_facodec_prepare_checkpoint.py``. INT dtypes come
# from BatchNorm ``num_batches_tracked`` counters and similar training
# artefacts — safe to strip. Any dtype outside both sets is refused
# unless --allow-strip-any is passed (fail-loud posture: the runtime
# forward path would refuse them anyway).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}

UPSAMPLER_REVISION = "c6d7494a60555672be09ca809a40be400d682a53"
UPSAMPLER_CHECKPOINT_FILE = "decoder_151000.pth"
UPSAMPLER_CHECKPOINT_BYTES = 72_610_550
UPSAMPLER_CHECKPOINT_SHA256 = (
    "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998"
)
UPSAMPLER_TENSOR_COUNT = 81

XCODEC_REVISION = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"
XCODEC_FIXED_FILES: dict[str, tuple[int, str]] = {
    "final_ckpt/ckpt_00360000.pth": (
        1_360_444_883,
        "c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c",
    ),
    "semantic_ckpts/hf_1_325000/pytorch_model.bin": (
        377_555_286,
        "c5ddbd7fa2468483cb9b2aa53117813471543dd278e65870333a56c54305f527",
    ),
    "decoders/decoder_151000.pth": (
        72_610_550,
        "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998",
    ),
}


def _upsampler_manifest() -> dict[str, list[int]]:
    out = {
        "backbone.embed.weight": [512, 1024, 7],
        "backbone.embed.bias": [512],
        "backbone.norm.weight": [512],
        "backbone.norm.bias": [512],
        "backbone.final_layer_norm.weight": [512],
        "backbone.final_layer_norm.bias": [512],
        "head.istft.window": [3528],
        "head.out.weight": [3530, 512],
        "head.out.bias": [3530],
    }
    for layer in range(8):
        prefix = f"backbone.convnext.{layer}"
        out.update(
            {
                f"{prefix}.dwconv.weight": [512, 1, 7],
                f"{prefix}.dwconv.bias": [512],
                f"{prefix}.norm.weight": [512],
                f"{prefix}.norm.bias": [512],
                f"{prefix}.pwconv1.weight": [1536, 512],
                f"{prefix}.pwconv1.bias": [1536],
                f"{prefix}.pwconv2.weight": [512, 1536],
                f"{prefix}.pwconv2.bias": [512],
                f"{prefix}.gamma": [512],
            }
        )
    assert len(out) == UPSAMPLER_TENSOR_COUNT
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_official_upsampler(path: Path) -> None:
    if path.name != UPSAMPLER_CHECKPOINT_FILE:
        sys.exit(
            f"YuE-upsampler public contract requires {UPSAMPLER_CHECKPOINT_FILE}, "
            f"got {path.name}"
        )
    size = path.stat().st_size
    if size != UPSAMPLER_CHECKPOINT_BYTES:
        sys.exit(
            f"{path} has {size} bytes; expected {UPSAMPLER_CHECKPOINT_BYTES} "
            f"from m-a-p/YuE-upsampler@{UPSAMPLER_REVISION}"
        )
    actual = _sha256_file(path)
    if actual != UPSAMPLER_CHECKPOINT_SHA256:
        sys.exit(
            f"{path} SHA-256 {actual}; expected {UPSAMPLER_CHECKPOINT_SHA256}"
        )


def _verify_official_xcodec(path: Path, relative_path: str) -> None:
    """Authenticate every fixed 151k xcodec-mini source payload."""
    expected = XCODEC_FIXED_FILES.get(relative_path)
    if expected is None:
        # The converter still exposes the upstream 131k decoder selector, but
        # it is not the pinned canonical/public runtime identity.
        return
    expected_bytes, expected_sha256 = expected
    size = path.stat().st_size
    if size != expected_bytes:
        sys.exit(
            f"{path} has {size} bytes; expected {expected_bytes} "
            f"from m-a-p/xcodec_mini_infer@{XCODEC_REVISION}"
        )
    actual = _sha256_file(path)
    if actual != expected_sha256:
        sys.exit(f"{path} SHA-256 {actual}; expected {expected_sha256}")


def _flatten(prefix: str, obj: Any) -> dict:
    """Flatten a nested dict into dotted-key ``{name: Tensor}``.

    The upstream files carry mostly-flat state dicts, but the
    SoundStream ``ckpt_00360000.pth`` file is training-wrapped (it
    contains ``codec_model`` / ``optimizer_g`` / ``optimizer_d`` top-level
    keys); the caller-visible unwrap happens in ``_load_one``
    before this walk is invoked, but this recursive flatten still
    protects against any residual nesting.
    """
    import torch

    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(key, v))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj
    # else: silently drop non-Tensor scalars/arrays — safetensors would
    # refuse them anyway; the runtime doesn't need them.
    return out


def _load_one(path: Path, role: str) -> dict:
    """Load one .pth / .bin state_dict and return a flat ``{role.name: Tensor}``.

    Fail-loud on any load failure OR on a payload that yields zero tensors —
    better a hard exit than a silently-empty prefix (the downstream Rust
    converter would then emit a GGUF with a valid header but no weights and
    the runtime forward would only fail much later at first-forward, which
    is the classic "silent partial" trap this project bans, mirror of the
    sepformer / facodec / dfn3 posture).

    The xcodec-mini SoundStream file has a stricter contract: role ``codec``
    MUST unwrap the exact non-empty ``codec_model`` mapping. Its sibling
    ``optimizer_g`` / ``optimizer_d`` mappings are training artefacts. A
    missing/renamed inference wrapper is an error rather than permission to
    recursively flatten the entire checkpoint.
    """
    import torch

    # First try weights_only=True (default in torch 2.6+). Upstream YuE
    # xcodec_mini training snapshots embed `omegaconf.listconfig.ListConfig`
    # (Hydra config wrapper) in the state dict — safe unpickler refuses it.
    # We fall back to weights_only=False for m-a-p/xcodec_mini_infer +
    # m-a-p/YuE-upsampler because we trust the upstream HF org (verified
    # 2026-08-01: apache-2.0 org, cardData sha256 match) and there's no
    # available torch.serialization.safe_globals entry for OmegaConf types.
    # This is a well-known accepted trade-off for legacy Hydra-based
    # training snapshots — see torch documentation.
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        try:
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"torch.load({path!s}) failed: {exc}")

    if role == "codec":
        if not isinstance(raw, dict):
            sys.exit(f"{path!s} codec payload is not a mapping")
        inner = raw.get("codec_model")
        if not isinstance(inner, dict) or not inner:
            sys.exit(
                f"{path!s} is missing the non-empty `codec_model` mapping; "
                "refusing to flatten optimizer_g/optimizer_d training state"
            )
        print(f"  {path.name}: unwrapped ['codec_model']; optimizer state excluded")
        raw = inner
    elif isinstance(raw, dict):
        # Common wrappers for the HuBERT and Vocos files. These roles have no
        # optimizer-bearing special case, but still unwrap only a tensor map.
        for wrapper in ("state_dict", "model_state_dict", "model", "module"):
            inner = raw.get(wrapper)
            if isinstance(inner, dict) and inner:
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    print(f"  {path.name}: unwrapped [{wrapper!r}]")
                    raw = inner
                    break

    flat_local = _flatten("", raw)
    if not flat_local:
        sys.exit(
            f"{path!s} yielded no tensors — expected an m-a-p YuE bundle "
            f"state_dict (see github.com/multimodal-art-projection/YuE)."
        )
    # Namespace under the role prefix so ``YueXcodecMini::from_gguf``
    # can locate the sub-module a tensor belongs to (``codec.encoder.conv.weight``
    # vs. ``semantic.encoder.layer_norm.weight`` vs. ``decoder.head.out.weight``).
    # For the upsampler variant the role is empty ("") = bare backbone.* /
    # head.* names (matches the sibling Charactr AI Vocos tensor-name
    # contract so the native runtime binder can share the loader path).
    if role:
        prefixed = {f"{role}.{k}": v for k, v in flat_local.items()}
    else:
        prefixed = dict(flat_local)
    return prefixed


def _partition(sd: dict, allow_strip_any: bool):
    """Split into ``(kept, dropped_int, unknown_other)`` — same taxonomy the
    ``nemo_pt_to_safetensors.py`` / ``sepformer_prepare_checkpoint.py`` /
    ``naturalspeech3_facodec_prepare_checkpoint.py`` precedents use."""
    kept: dict = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)
        if dtype_s in KEEP_DTYPES:
            if hasattr(t, "contiguous"):
                t = t.contiguous()
            if hasattr(t, "detach"):
                t = t.detach()
            kept[name] = t
        elif dtype_s in INT_DTYPES:
            dropped.append((name, dtype_s, list(t.shape)))
        else:
            unknown.append((name, dtype_s, list(t.shape)))
    return kept, dropped, unknown


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge YuE bundle (upsampler or xcodec-mini) upstream pickle "
            "checkpoint(s) → single .safetensors for vokra-convert. "
            "Variant selects which HF repo layout to expect."
        ),
    )
    ap.add_argument(
        "--ckpt-dir", required=True, type=Path,
        help=(
            "directory holding the upstream files — typically the output of "
            "`huggingface-cli download m-a-p/YuE-upsampler --local-dir <dir>` "
            "or `huggingface-cli download m-a-p/xcodec_mini_infer --local-dir <dir>`."
        ),
    )
    ap.add_argument(
        "--variant", required=True,
        choices=sorted(VARIANT_SUBSETS.keys()),
        help=(
            "which of the two YuE bundle variants to merge: "
            "'upsampler' = m-a-p/YuE-upsampler (Vocos vocoder head, 145 MB); "
            "'xcodec-mini' = m-a-p/xcodec_mini_infer (SoundStream RVQ codec + "
            "HuBERT semantic encoder + Vocos decoder head bundle, ~1.88 GB)."
        ),
    )
    ap.add_argument(
        "--snapshot", required=False, default="151000",
        choices=("131000", "151000"),
        help=(
            "which Vocos decoder training snapshot to pick "
            "(default: 151000 = later training step, typically the 'final' one). "
            "The two snapshots are byte-identical between the two sibling repos."
        ),
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--allow-strip-any", action="store_true",
        help="also strip fp64 / complex tensors (default: refuse them loudly).",
    )
    args = ap.parse_args()

    if args.variant == "upsampler" and args.snapshot != "151000":
        ap.error(
            "the strict public yue-upsampler contract pins snapshot 151000; "
            "snapshot 131000 is not published under the canonical runtime identity"
        )

    try:
        from safetensors.torch import save_file
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            f"missing dep {exc}. run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    ckpt_dir: Path = args.ckpt_dir
    if not ckpt_dir.is_dir():
        print(f"--ckpt-dir must be an existing directory: {ckpt_dir}", file=sys.stderr)
        return 2

    subset = VARIANT_SUBSETS[args.variant]

    merged: dict = {}
    per_file_counts: dict[str, int] = {}
    for rel_template, role in subset:
        rel = rel_template.format(snapshot=args.snapshot)
        f_path = ckpt_dir / rel
        if not f_path.is_file():
            print(
                f"required checkpoint missing: {f_path} "
                f"(variant={args.variant!r}, snapshot={args.snapshot!r})",
                file=sys.stderr,
            )
            return 2
        if args.variant == "upsampler":
            _verify_official_upsampler(f_path)
        else:
            _verify_official_xcodec(f_path, rel)
        role_suffix = f" as role {role!r}" if role else " (no role prefix)"
        print(
            f"  loading {rel} ({f_path.stat().st_size:,} bytes){role_suffix}"
        )
        sub = _load_one(f_path, role)
        # Duplicate-key guard: the role prefix makes cross-file collision
        # impossible in practice for xcodec-mini, but this redundant assert
        # catches an accidental within-file duplicate the flatten walk
        # cannot see (a nested dict that re-uses a leaf name under two
        # branches). For the upsampler variant which has only one file,
        # this is a no-op.
        overlap = set(merged) & set(sub)
        if overlap:
            print(
                f"  duplicate keys after prefix (first 5): {sorted(overlap)[:5]}",
                file=sys.stderr,
            )
            return 3
        merged.update(sub)
        per_file_counts[rel] = len(sub)

    kept, dropped, unknown = _partition(merged, args.allow_strip_any)

    if args.variant == "upsampler":
        expected = _upsampler_manifest()
        if len(kept) != UPSAMPLER_TENSOR_COUNT:
            print(
                f"strict YuE-upsampler checkpoint has {len(kept)} float tensors; "
                f"expected exactly {UPSAMPLER_TENSOR_COUNT}",
                file=sys.stderr,
            )
            return 3
        actual_names = set(kept)
        expected_names = set(expected)
        wrong_shapes = [
            (name, list(kept[name].shape), expected[name])
            for name in sorted(actual_names & expected_names)
            if list(kept[name].shape) != expected[name]
        ]
        if actual_names != expected_names or wrong_shapes:
            print(
                "strict YuE-upsampler manifest mismatch: "
                f"missing={sorted(expected_names - actual_names)[:5]}, "
                f"extra={sorted(actual_names - expected_names)[:5]}, "
                f"wrong_shapes={wrong_shapes[:3]}",
                file=sys.stderr,
            )
            return 3

    if unknown and not args.allow_strip_any:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"refusing to drop {len(unknown)} tensors of unknown dtype "
            f"(first 3: {first}); re-run with --allow-strip-any if verified "
            f"inference-inert.",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))

    manifest = {
        "input_dir": str(ckpt_dir),
        "variant": args.variant,
        "snapshot": args.snapshot,
        "upstream_revision": (
            UPSAMPLER_REVISION if args.variant == "upsampler" else None
        ),
        "checkpoint_sha256": (
            UPSAMPLER_CHECKPOINT_SHA256 if args.variant == "upsampler" else None
        ),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "per_file": per_file_counts,
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_stripped": (
            [{"name": n, "dtype": d, "shape": s} for n, d, s in unknown]
            if args.allow_strip_any else []
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"yue_bundle_prepare_checkpoint: variant={args.variant!r}, "
        f"snapshot={args.snapshot!r}, kept {len(kept)}, "
        f"dropped {len(dropped)} int, "
        f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
        f"per-file {per_file_counts}; "
        f"manifest -> {manifest_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
