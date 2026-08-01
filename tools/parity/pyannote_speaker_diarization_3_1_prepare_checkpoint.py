#!/usr/bin/env python3
"""Download the `pyannote/speaker-diarization-3.1` pipeline config.yaml.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream `pyannote/speaker-diarization-3.1` HF repo is a
**pipeline orchestration definition**, not a weight checkpoint. It ships
**only** a ~2 KB ``config.yaml`` — no ``.bin`` / ``.safetensors`` /
``.ckpt`` weight files. The pipeline delegates every forward-pass
computation to two sibling MIT weight repos:

- ``pyannote/segmentation-3.0`` (PyanNet VAD / speaker-segmentation
  backbone — separately staged with the sibling
  ``pyannote_segmentation`` converter + the shared
  ``bin_to_safetensors.py`` bridge).
- ``pyannote/wespeaker-voxceleb-resnet34-LM`` (WeSpeaker speaker
  embedding — separately staged with the sibling ``wespeaker``
  converter path).

The Vokra converter (``crates/vokra-convert/src/models/
pyannote_speaker_diarization_3_1.rs``) accepts the raw ``config.yaml``
as a sanity buffer (it does NOT parse the YAML — the pipeline
hparams are transcribed from primary source as Rust compile-time
constants, keeping NFR-DS-02 zero-dep). This prep script:

1. Downloads the config.yaml + LICENSE + README.md into an output
   directory (via ``huggingface_hub.snapshot_download``).
2. Prints sha256 of the config.yaml so a downstream publish can
   include it in the model card + verify against upstream drift.
3. Emits nothing that requires a torch or YAML parser — the file
   is passed to ``vokra-cli convert`` as-is.

# Determinism

``huggingface_hub`` caches the download and reuses on repeat runs
(idempotent); the emitted ``config.yaml`` is byte-identical to the
HF-hosted upstream (no repackaging). sha256 is stable given a stable
upstream commit revision.

# Redistribution

Upstream ``config.yaml`` license is ``mit`` (verified 2026-07-30 via
authenticated HF API cardData tag ``license: mit`` + ``gated: auto``
= access control only, no extra obligations). See
``docs/license-audit.md`` §3.1 row
``pyannote-speaker-diarization-3.1 (pyannote/speaker-diarization-3.1)``
= ☑ Commercial 2026-08-01 yousan (依頼者許可 = CC 判断).

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/pyannote_speaker_diarization_3_1_prepare_checkpoint.py \\
        --output-dir /tmp/pyannote-speaker-diarization-3.1

The Vokra converter is then invoked as::

    vokra-cli convert \\
        --model pyannote-speaker-diarization-3.1 \\
        --input /tmp/pyannote-speaker-diarization-3.1/config.yaml \\
        --output /tmp/vokra-pyannote-speaker-diarization-3.1.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

LOG_PREFIX = "pyannote_speaker_diarization_3_1_prepare:"
UPSTREAM_REPO = "pyannote/speaker-diarization-3.1"

# What we ask HF to include in the snapshot. Deliberately small — the
# repo is only ~2 KB, and we want the config + license + card so the
# operator can inspect provenance offline. Excludes anything the Rust
# converter does not need.
HF_ALLOW_PATTERNS = [
    "config.yaml",
    "LICENSE",
    "LICENSE.txt",
    "README.md",
    "handler.py",  # upstream ships a small inference wrapper — kept for reference only
]


def sha256_of(path: Path) -> str:
    """SHA256 of `path`, streaming — no full-file read into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pipeline_config(hf_repo: str, output_dir: Path, revision: "str | None") -> Path:
    """Mirror of ``sepformer_prepare_checkpoint.download_checkpoint`` +
    ``bin_to_safetensors.download_checkpoint`` — token resolved from the
    environment by ``huggingface_hub``, never accepted as a CLI flag
    (argv leaks via ``ps``)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover — env-config error
        sys.exit(
            f"{LOG_PREFIX} missing Python dep ({exc}); install with "
            "`uv add huggingface_hub` in tools/parity/"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=HF_ALLOW_PATTERNS,
    )
    return Path(snapshot_dir)


def verify_pipeline_config(config_path: Path) -> None:
    """Fail-loud sanity check.

    Not a YAML parser — a raw-bytes substring scan mirroring the Rust
    converter's ``is_speaker_diarization_3_1_config`` helper. If the
    upstream repo layout drifts, this catches it up-front rather than
    letting the Rust converter emit a weightless GGUF against a wrong
    config.
    """
    if not config_path.exists():
        sys.exit(
            f"{LOG_PREFIX} expected {config_path} in snapshot, not found — "
            f"upstream repo layout may have changed"
        )
    payload = config_path.read_bytes()
    if b"SpeakerDiarization" not in payload:
        sys.exit(
            f"{LOG_PREFIX} {config_path} does not carry the SpeakerDiarization "
            f"marker — this is NOT the {UPSTREAM_REPO} pipeline config"
        )
    if b"3.1" not in payload:
        sys.exit(
            f"{LOG_PREFIX} {config_path} does not carry the 3.1 version marker "
            f"— upstream may have released a new pipeline version, verify the "
            f"primary source before proceeding"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download the pyannote/speaker-diarization-3.1 pipeline "
            "config.yaml (offline side-car for the Vokra Rust converter)."
        )
    )
    ap.add_argument(
        "--hf-repo",
        default=UPSTREAM_REPO,
        help=f"HF repo id (default: {UPSTREAM_REPO})",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="HF commit SHA / tag / branch (default: latest main)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to snapshot the config into (created if absent)",
    )
    args = ap.parse_args()

    snapshot = download_pipeline_config(args.hf_repo, args.output_dir, args.revision)
    print(f"{LOG_PREFIX} snapshot at {snapshot}")

    config = snapshot / "config.yaml"
    verify_pipeline_config(config)
    digest = sha256_of(config)
    print(f"{LOG_PREFIX} config.yaml sha256 = {digest}")
    print(f"{LOG_PREFIX} pass to `vokra-cli convert --model pyannote-speaker-diarization-3.1`")


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    main()
