#!/usr/bin/env python3
"""Flatten a DAC release ``weights_*.pth`` → safetensors + config side-car (M4-04 T11).

This is an **offline** sidecar tool (FR-LD-05: no Python / PyTorch ever enters
the runtime). The upstream descript-audio-codec releases ship torch-pickle
``.pth`` files (an audiotools ``BaseModel`` dict of ``{"state_dict": …,
"metadata": {"kwargs": …}}``); the Rust converter
(``crates/vokra-convert/src/models/dac.rs``) consumes safetensors + a JSON
config only, so this script bridges the two:

* Loads the ``.pth`` with ``torch.load(..., weights_only=True)`` (the release
  checkpoints load cleanly under the safe loader — verified on the 24 kHz
  tag 0.0.4 file; a checkpoint that does not is refused rather than
  falling back to unsafe unpickling).
* Writes the flat ``state_dict`` verbatim via
  ``safetensors.torch.save_file`` (dotted keys preserved:
  ``quantizer.quantizers.{i}.codebook.weight`` /
  ``.out_proj.{weight_g,weight_v,bias}`` / encoder / decoder chain —
  weight-norm is NOT folded here; the Rust converter owns that math so the
  fold is covered by its unit tests).
* Emits the config side-car the converter requires, derived from the
  checkpoint's own ``metadata.kwargs`` (nothing invented):

  - ``n_codebooks`` / ``codebook_size`` / ``codebook_dim`` — verbatim;
  - ``d_model`` — ``kwargs["latent_dim"]`` when present, else the DAC class
    default derivation ``encoder_dim * 2 ** len(encoder_rates)``
    (descriptinc/descript-audio-codec ``dac/model/dac.py`` L169-170);
  - ``sample_rate`` — verbatim;
  - ``hop_length`` — ``prod(encoder_rates)`` (dac.py L174).

* Prints a sha256 manifest line per output for the fixture / workflow logs.

Fails loudly on any anomaly (missing kwargs key, non-tensor state entry,
list-valued ``codebook_dim`` with mixed values) rather than masking it —
FR-EX-08 posture.

# Usage

::

    tools/parity/parity-venv/bin/python tools/parity/dac_prepare_checkpoint.py \\
        --pth ~/.cache/descript/dac/weights_24khz_8kbps_0.0.4.pth \\
        --output /tmp/dac-24khz.safetensors \\
        --config-out /tmp/dac-24khz-config.json

Then:

::

    vokra-cli convert --model dac --input /tmp/dac-24khz.safetensors \\
        --config /tmp/dac-24khz-config.json --output /tmp/dac-24khz.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pth", type=Path, required=True, help="Upstream DAC weights_*.pth")
    ap.add_argument("--output", type=Path, required=True, help="Flattened safetensors out")
    ap.add_argument("--config-out", type=Path, required=True, help="Config JSON side-car out")
    args = ap.parse_args()

    import torch  # noqa: PLC0415  (offline tool; venv-isolated)
    from safetensors.torch import save_file  # noqa: PLC0415

    try:
        payload = torch.load(args.pth, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"dac_prepare_checkpoint: refusing to load {args.pth} without weights_only=True "
            f"({type(e).__name__}: {e}). The official release checkpoints load under the safe "
            "loader; do not disable it.",
            file=sys.stderr,
        )
        return 2

    if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
        print(
            "dac_prepare_checkpoint: unexpected checkpoint layout — expected "
            "{'state_dict': …, 'metadata': {'kwargs': …}} (audiotools BaseModel dict)",
            file=sys.stderr,
        )
        return 2

    kwargs = payload["metadata"].get("kwargs", {})
    required = ["n_codebooks", "codebook_size", "codebook_dim", "sample_rate", "encoder_rates"]
    missing = [k for k in required if k not in kwargs]
    if missing:
        print(f"dac_prepare_checkpoint: metadata.kwargs missing {missing}", file=sys.stderr)
        return 2

    codebook_dim = kwargs["codebook_dim"]
    if isinstance(codebook_dim, list):
        uniq = set(codebook_dim)
        if len(uniq) != 1:
            print(
                f"dac_prepare_checkpoint: per-quantizer codebook_dim varies ({sorted(uniq)}) — "
                "the Vokra DacRvqAttrs carries a single codebook_dim; this variant is not "
                "supported (file an issue with the variant name).",
                file=sys.stderr,
            )
            return 2
        codebook_dim = codebook_dim[0]

    encoder_rates = list(kwargs["encoder_rates"])
    hop_length = math.prod(encoder_rates)
    if "latent_dim" in kwargs and kwargs["latent_dim"]:
        d_model = int(kwargs["latent_dim"])
    else:
        # DAC class default: latent_dim = encoder_dim * 2 ** len(encoder_rates)
        # (dac/model/dac.py L169-170). encoder_dim is required in that case.
        if "encoder_dim" not in kwargs:
            print(
                "dac_prepare_checkpoint: kwargs has neither latent_dim nor encoder_dim — "
                "cannot derive d_model",
                file=sys.stderr,
            )
            return 2
        d_model = int(kwargs["encoder_dim"]) * (2 ** len(encoder_rates))

    state = payload["state_dict"]
    tensors = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            print(f"dac_prepare_checkpoint: non-tensor state entry `{k}` ({type(v)})", file=sys.stderr)
            return 2
        tensors[k] = v.contiguous().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))

    config = {
        "n_codebooks": int(kwargs["n_codebooks"]),
        "codebook_size": int(kwargs["codebook_size"]),
        "codebook_dim": int(codebook_dim),
        "d_model": d_model,
        "sample_rate": int(kwargs["sample_rate"]),
        "hop_length": int(hop_length),
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n")

    print(f"tensors: {len(tensors)}")
    print(f"config: {json.dumps(config)}")
    print(f"sha256 {args.output.name} {_sha256(args.output)}")
    print(f"sha256 {args.config_out.name} {_sha256(args.config_out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
