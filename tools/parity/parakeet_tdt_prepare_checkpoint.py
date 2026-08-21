#!/usr/bin/env python3
"""Prepare the audited Parakeet-TDT-0.6B-v3 checkpoint for conversion.

The official safetensors contains 24 scalar int64
``encoder.layers.*.conv.norm.num_batches_tracked`` training counters. Vokra's
runtime accepts only float weights, and eval BatchNorm never reads those
counters. This wrapper pins the upstream revision, delegates the auditable
copy to ``strip_int_tensors.py``, then verifies that precisely those 24 slots
were removed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

UPSTREAM_HF = "nvidia/parakeet-tdt-0.6b-v3"
AUDITED_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
COUNTER = re.compile(r"^encoder\.layers\.(\d+)\.conv\.norm\.num_batches_tracked$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--revision",
        default=AUDITED_REVISION,
        help="Override only after an explicit upstream re-audit",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream_dir = args.output_dir / "upstream"
    upstream_dir.mkdir(exist_ok=True)
    input_path = Path(
        hf_hub_download(
            repo_id=UPSTREAM_HF,
            filename="model.safetensors",
            revision=args.revision,
            local_dir=upstream_dir,
        )
    )
    for filename in ["config.json", "generation_config.json", "processor_config.json"]:
        hf_hub_download(
            repo_id=UPSTREAM_HF,
            filename=filename,
            revision=args.revision,
            local_dir=upstream_dir,
        )

    output_path = args.output_dir / "model.safetensors"
    helper = Path(__file__).resolve().with_name("strip_int_tensors.py")
    command = [
        sys.executable,
        str(helper),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    print("parakeet_tdt_prepare_checkpoint: " + " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode

    manifest_path = output_path.with_suffix(output_path.suffix + ".stripped-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    dropped = manifest["dropped_tensors"]
    layers = []
    for tensor in dropped:
        match = COUNTER.fullmatch(tensor["name"])
        if match is None or tensor["dtype"] != "torch.int64" or tensor["shape"] != []:
            sys.exit(f"unexpected stripped tensor: {tensor}")
        layers.append(int(match.group(1)))
    if manifest["kept_count"] != 699 or sorted(layers) != list(range(24)):
        sys.exit(
            "expected 699 floats and exactly one scalar BatchNorm counter for layers 0..23; "
            f"got kept={manifest['kept_count']} layers={sorted(layers)}"
        )
    print(
        f"parakeet_tdt_prepare_checkpoint: revision={args.revision} kept=699 dropped=24 output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
