#!/usr/bin/env python3
"""Merge OpenMOSS MOSS-Audio-Tokenizer sharded safetensors → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``OpenMOSS-Team/MOSS-Audio-Tokenizer`` release ships as **2
sharded safetensors** files (``model-00001-of-00002.safetensors`` +
``model-00002-of-00002.safetensors``) plus a ``model.safetensors.index.json``
weight-map (~6.6 GB total, verified 2026-08-01 via HF cardData API
``https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-Tokenizer``).
The sibling ``OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`` ships as **1
sharded safetensors** (``model-00001-of-00001.safetensors``) with the same
``model.safetensors.index.json`` weight-map layout (~88 MB total).
``OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`` ships as **3 shards** containing
8,494,804,992 F32 tensor bytes. It must be prepared on vast.ai, pinned to
revision ``f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169``.

Vokra's Rust converter (``crates/vokra-convert/src/models/moss_audio_tokenizer.rs``)
consumes **single-file safetensors** by design — the runtime never grows
a shard-index reader (NFR-DS-02 zero-dep). This script bridges the two:
it walks the weight-map, loads every shard, merges them into a single
in-memory state_dict, and re-serializes as one safetensors the caller
feeds to ``vokra-cli convert --model moss-audio-tokenizer``.

Both variants are **safetensors-native** — no pickle bridge is required
(contrast the sibling ``MOSS-TTS-Nano-100M`` which requires
``bin_to_safetensors.py``). Both are ``private: false`` and
``gated: false`` with ``license: apache-2.0`` (HF cardData primary source
verified 2026-08-01 via authenticated API).

Precedent: this posture mirrors ``granite_speech_prepare_checkpoint.py``
(3-shard merge) — Vokra converters keep the runtime tree free of the
shard-index-json reader by pre-flattening offline.

# Determinism

Shards are loaded in the order declared by ``model.safetensors.index.json``'s
``weight_map`` (Python dict insertion order preserves iteration since
3.7). Identical ``--hf-repo`` input produces byte-identical output
(safetensors serialization is deterministic for a fixed key ordering).

# Redistribution

Upstream weight license is ``apache-2.0`` end-to-end (verified 2026-08-01
via HF cardData API + upstream README) — see ``docs/license-audit.md``
§3.1 rows "MOSS-Audio-Tokenizer (Full)" / "MOSS-Audio-Tokenizer (Nano)",
both ☑ Commercial as of 2026-08-01.

# Custom code / trust_remote_code

Both variants ship ``modeling_moss_audio_tokenizer.py`` +
``configuration_moss_audio_tokenizer.py`` requiring
``trust_remote_code=True`` for the reference Python forward. Vokra never
touches Python at runtime, so this only affects the owner-side parity
dumper (mirror ``kokoro_prepare_checkpoint.py``). This script reads
tensor bytes verbatim without invoking the modeling code.

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/moss_audio_tokenizer_prepare_checkpoint.py \\
        --hf-repo OpenMOSS-Team/MOSS-Audio-Tokenizer \\
        --output /tmp/moss-audio-tokenizer-full.safetensors

    uv run --project tools/parity python \\
        tools/parity/moss_audio_tokenizer_prepare_checkpoint.py \\
        --hf-repo OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano \\
        --output /tmp/moss-audio-tokenizer-nano.safetensors

    uv run --project tools/parity python \\
        tools/parity/moss_audio_tokenizer_prepare_checkpoint.py \\
        --hf-repo OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \\
        --revision f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169 \\
        --output /tmp/moss-audio-tokenizer-v2.safetensors

The optional ``--local-dir`` argument accepts a pre-downloaded checkpoint
directory (skips the HF download); useful when the operator has already
snapshotted the release for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LOG_PREFIX = "moss_audio_tokenizer_prepare_checkpoint:"
V2_REPO = "openmoss-team/moss-audio-tokenizer-v2"
V2_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
V2_FILE_SHA256 = {
    "LICENSE": "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a",
    "config.json": "aeb9a0e9d88c74bf9fbaa81ee54443d463e09b5f335b3306bb798e282a10e564",
    "configuration_moss_audio_tokenizer.py": "f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529",
    "model.safetensors.index.json": "912f52f053e04ff7e9abc8f05aa75dfbb40b31c86a0f4ad5c5a36e4aa28a624f",
    "modeling_moss_audio_tokenizer.py": "7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9",
    "model-00001-of-00003.safetensors": "2d9f9182f17b143a23937feb87c63c08221bd28e685e4bc2fa55dcdce17fcde7",
    "model-00002-of-00003.safetensors": "d4e48106d0254fe3b00ea0707e88fc6aee076993825e108dd9cef847f9db236e",
    "model-00003-of-00003.safetensors": "d0449fe1b0ef1f6045946867148d8166b9a91a58d0feca4a18b641494d0b22da",
}
V2_FILE_BYTES = {
    "LICENSE": 11_324,
    "config.json": 10_166,
    "configuration_moss_audio_tokenizer.py": 19_772,
    "model.safetensors.index.json": 191_718,
    "modeling_moss_audio_tokenizer.py": 105_970,
    "model-00001-of-00003.safetensors": 3_978_639_168,
    "model-00002-of-00003.safetensors": 3_992_738_352,
    "model-00003-of-00003.safetensors": 523_681_336,
}

# Files the downstream Vokra converter needs. We deliberately fetch the
# custom-code Python files too so an owner-side parity dumper (see the
# module docstring) can invoke them with trust_remote_code=True.
HF_ALLOW_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "configuration_moss_audio_tokenizer.py",
    "modeling_moss_audio_tokenizer.py",
    "LICENSE",
    "*.md",
]

# INT dtypes come from training-artifact counters (BatchNorm
# num_batches_tracked etc.). Safe to strip. Any dtype outside both sets
# is refused unless --allow-strip-any is passed (fail-loud posture: the
# runtime forward path would refuse them anyway).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _partition(sd: dict, allow_strip_any: bool):
    """Split into ``(kept, dropped_int, unknown_other)`` — same taxonomy
    the ``nemo_pt_to_safetensors.py`` / ``sepformer_prepare_checkpoint.py``
    precedents use."""
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


def _load_shard(path: Path) -> dict:
    """Load one .safetensors file into a flat state_dict.

    Fail-loud on any load failure OR on a payload that yields zero
    tensors — better a hard exit than a silently-empty prefix (the
    downstream Rust converter would then emit a GGUF with a valid header
    but no weights and the runtime forward would only fail much later at
    first-forward, which is the classic "silent partial" trap this
    project bans).
    """
    from safetensors.torch import load_file

    try:
        sd = load_file(str(path), device="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"safetensors.torch.load_file({path!s}) failed: {exc}")
    if not sd:
        sys.exit(
            f"{path!s} yielded no tensors — expected a MOSS-Audio-Tokenizer "
            f"shard. Corrupt download or wrong file?"
        )
    return sd


def _download_repo(repo: str, revision: str | None, out_dir: Path) -> Path:
    """Fetch the HF repo into ``out_dir`` via ``huggingface_hub.snapshot_download``."""
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(out_dir),
            allow_patterns=HF_ALLOW_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"huggingface_hub.snapshot_download({repo}) failed: {exc}")
    return Path(local_dir)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_v2_contract(src_dir: Path) -> None:
    """Fail closed unless every fixed-revision source and shard matches."""
    for relative, expected in V2_FILE_SHA256.items():
        path = src_dir / relative
        if not path.is_file():
            sys.exit(f"{LOG_PREFIX} v2 contract file is missing: {path}")
        actual_bytes = path.stat().st_size
        expected_bytes = V2_FILE_BYTES[relative]
        if actual_bytes != expected_bytes:
            sys.exit(
                f"{LOG_PREFIX} v2 contract size mismatch for {relative}: "
                f"got {actual_bytes}, expected {expected_bytes} at revision "
                f"{V2_REVISION}"
            )
        actual = _sha256_file(path)
        if actual != expected:
            sys.exit(
                f"{LOG_PREFIX} v2 contract hash mismatch for {relative}: "
                f"got {actual}, expected {expected} at revision {V2_REVISION}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge OpenMOSS MOSS-Audio-Tokenizer sharded safetensors → single "
            ".safetensors for consumption by vokra-cli convert --model "
            "moss-audio-tokenizer{,-nano}."
        ),
    )
    ap.add_argument(
        "--hf-repo",
        default=None,
        help=(
            "HF repo id (e.g. OpenMOSS-Team/MOSS-Audio-Tokenizer, "
            "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano). Required unless "
            "--local-dir is supplied."
        ),
    )
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=(
            "Pre-downloaded checkpoint directory (must contain "
            "model.safetensors.index.json and every shard listed therein). "
            "Skips the HF download. Mutually exclusive with --hf-repo unless "
            "both are supplied — in which case the local-dir wins and "
            "--hf-repo is used only for the provenance log."
        ),
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help=(
            "immutable HF revision to download. Required for v2; use "
            "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169."
        ),
    )
    ap.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "directory to snapshot into when --local-dir is not supplied. "
            "Defaults to a sibling of --output named '<stem>-src/'."
        ),
    )
    ap.add_argument(
        "--allow-strip-any", action="store_true",
        help="also strip fp64 / complex tensors (default: refuse them loudly).",
    )
    args = ap.parse_args()

    if args.hf_repo is None and args.local_dir is None:
        print("either --hf-repo or --local-dir is required", file=sys.stderr)
        return 2
    if (
        args.hf_repo is not None
        and args.hf_repo.lower() == V2_REPO
        and args.revision != V2_REVISION
    ):
        print(
            f"{LOG_PREFIX} v2 requires --revision "
            "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
            file=sys.stderr,
        )
        return 2

    try:
        from safetensors.torch import save_file  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            f"{LOG_PREFIX} missing dep {exc}. run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    # Resolve source directory.
    if args.local_dir is not None:
        src_dir: Path = args.local_dir
        if not src_dir.is_dir():
            print(f"{LOG_PREFIX} --local-dir must exist: {src_dir}", file=sys.stderr)
            return 2
    else:
        download_dir = args.download_dir
        if download_dir is None:
            stem = args.output.stem
            download_dir = args.output.parent / f"{stem}-src"
        download_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"{LOG_PREFIX} downloading {args.hf_repo} → {download_dir}",
            file=sys.stderr,
        )
        src_dir = _download_repo(args.hf_repo, args.revision, download_dir)

    v2_config = src_dir / "config.json"
    is_v2 = (
        args.hf_repo is not None and args.hf_repo.lower() == V2_REPO
    ) or (
        v2_config.is_file()
        and _sha256_file(v2_config) == V2_FILE_SHA256["config.json"]
    )
    if is_v2 and args.revision != V2_REVISION:
        print(
            f"{LOG_PREFIX} v2 local input also requires --revision {V2_REVISION}",
            file=sys.stderr,
        )
        return 2
    if is_v2:
        _validate_v2_contract(src_dir)

    # Locate the weight-map.
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        # Some releases ship a single un-sharded model.safetensors and
        # omit the index. Fall back to that if present.
        single = src_dir / "model.safetensors"
        if single.is_file():
            print(
                f"{LOG_PREFIX} no weight-map found; single-shard release detected "
                f"({single}). Loading directly.",
                file=sys.stderr,
            )
            merged = _load_shard(single)
        else:
            print(
                f"{LOG_PREFIX} neither {index_path.name} nor model.safetensors "
                f"found in {src_dir}",
                file=sys.stderr,
            )
            return 3
    else:
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        wm = index.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            print(
                f"{LOG_PREFIX} weight_map is missing or empty in {index_path}",
                file=sys.stderr,
            )
            return 3

        # Load every unique shard listed in the weight_map, preserving
        # first-seen order (Python dict insertion order since 3.7).
        seen: dict[str, None] = {}
        for shard_rel in wm.values():
            if not isinstance(shard_rel, str):
                continue
            seen.setdefault(shard_rel, None)

        merged: dict = {}
        for shard_rel in seen:
            shard_path = src_dir / shard_rel
            if not shard_path.is_file():
                print(
                    f"{LOG_PREFIX} weight_map references missing shard: "
                    f"{shard_path}",
                    file=sys.stderr,
                )
                return 3
            print(
                f"{LOG_PREFIX}   loading {shard_rel} "
                f"({shard_path.stat().st_size:,} bytes)",
                file=sys.stderr,
            )
            sub = _load_shard(shard_path)
            overlap = set(merged) & set(sub)
            if overlap:
                # A well-formed index should never map a tensor name to
                # two shards; assert loudly if it does.
                print(
                    f"{LOG_PREFIX}   duplicate keys across shards "
                    f"(first 5): {sorted(overlap)[:5]}",
                    file=sys.stderr,
                )
                return 3
            merged.update(sub)

        # Sanity: every declared weight ought to be present after merge.
        missing = [k for k in wm if k not in merged]
        if missing:
            print(
                f"{LOG_PREFIX} weight_map declared {len(missing)} tensors "
                f"absent from merged state_dict (first 5: {missing[:5]})",
                file=sys.stderr,
            )
            return 3

    kept, dropped, unknown = _partition(merged, args.allow_strip_any)

    if unknown and not args.allow_strip_any:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"{LOG_PREFIX} refusing to drop {len(unknown)} tensors of unknown "
            f"dtype (first 3: {first}); re-run with --allow-strip-any if "
            f"verified inference-inert.",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(kept, str(args.output))

    manifest = {
        "hf_repo": args.hf_repo,
        "input_dir": str(src_dir),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "sha256": _sha256_file(args.output),
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
        f"{LOG_PREFIX} kept {len(kept)}, dropped {len(dropped)} int, "
        f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
        f"sha256={manifest['sha256'][:16]}...; "
        f"manifest -> {manifest_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
