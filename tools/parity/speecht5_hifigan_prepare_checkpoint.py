#!/usr/bin/env python3
"""Download `microsoft/speecht5_hifigan` and convert its `pytorch_model.bin`
to `model.safetensors` for the Vokra converter.

Why a thin wrapper over ``bin_to_safetensors.py``
-------------------------------------------------

Upstream ships **only** ``pytorch_model.bin`` + ``config.json`` (verified
2026-07-31 via ``https://huggingface.co/api/models/microsoft/speecht5_hifigan``
— the sibling filenames are exactly ``[".gitattributes", "README.md",
"config.json", "pytorch_model.bin"]``). Vokra's Rust converter under
``crates/vokra-convert/`` is safetensors-only by design (keeps zero-dep
NFR-DS-02: no pickle parser in the runtime, and no arbitrary-code-exec
risk on load), so we do the bin→safetensors conversion in Python — where
torch already knows how to safely deserialize its own pickle — up-front,
as an offline sidecar tool.

The heavy lifting (safe ``torch.load(weights_only=True)``, non-str-key /
non-tensor / empty-state-dict fail-loud, tied-weight aliasing detection,
contiguous-layout enforcement, sha256 output) lives in
``tools/parity/bin_to_safetensors.py`` (the shared DeBERTa-v3-large /
VoxCPM-0.5B / Fun-CosyVoice3 prep tool). This wrapper only pins
``--hf-repo microsoft/speecht5_hifigan`` so the caller does not have to
re-type the slug.

Usage
-----

    uv run python speecht5_hifigan_prepare_checkpoint.py \\
        --output-dir /tmp/speecht5_hifigan-safetensors

    # optional: pin a revision for reproducible fixtures
    uv run python speecht5_hifigan_prepare_checkpoint.py \\
        --output-dir /tmp/speecht5_hifigan-safetensors \\
        --revision bb6f429406e86a9992357a972c0698b22043307d

The output directory receives ``pytorch_model.bin`` + ``config.json`` +
``model.safetensors`` side-by-side. Feed ``model.safetensors`` to
``vokra-cli convert --model speecht5-hifigan``.

Zero-dep posture
----------------

This script is python3-stdlib-only itself (argparse + subprocess);
transitively it inherits the ``torch`` + ``safetensors`` + ``huggingface-hub``
deps declared under ``tools/parity/pyproject.toml`` (per
[[feedback-python-uses-uv]] + [[feedback-python-3-12]]). None of these
show up in the Vokra runtime ``Cargo.lock`` — the runtime never grows a
pickle parser.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

UPSTREAM_HF = "microsoft/speecht5_hifigan"
LOG_PREFIX = "speecht5_hifigan_prep:"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download microsoft/speecht5_hifigan and convert its "
            "pytorch_model.bin to model.safetensors (thin wrapper over "
            "bin_to_safetensors.py Mode a). See module docstring for the "
            "FR-EX-08 fail-loud posture inherited from the underlying tool."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory to download the checkpoint into (created if absent). "
            "Will contain pytorch_model.bin + config.json + model.safetensors "
            "after a successful run."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Optional HF revision (branch / tag / commit sha) to pin. "
            "Default is the current `main`. Pinning the commit sha makes "
            "downstream parity fixtures reproducible."
        ),
    )
    parser.add_argument(
        "--out-basename",
        default="model.safetensors",
        help=(
            "Output filename for the converted safetensors "
            "(default: model.safetensors, mirrors HF convention)."
        ),
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    bin_to_safetensors = here / "bin_to_safetensors.py"
    if not bin_to_safetensors.is_file():
        sys.exit(
            f"{LOG_PREFIX} missing sibling script {bin_to_safetensors} — "
            "this wrapper is a thin driver around it."
        )

    cmd = [
        sys.executable,
        str(bin_to_safetensors),
        "--hf-repo",
        UPSTREAM_HF,
        "--output-dir",
        str(args.output_dir),
        "--out-basename",
        args.out_basename,
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])

    print(f"{LOG_PREFIX} exec: {' '.join(cmd)}")
    # Inherit env: HF_TOKEN etc. flows through untouched.
    return subprocess.run(cmd, check=False, env=os.environ.copy()).returncode


if __name__ == "__main__":
    sys.exit(main())
