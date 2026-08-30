# NanoCodec conversion contract

> **2026-08-30 current-state boundary:** The conversion contract and the
> checkpoint inventory below are dated audit records (the inventory is
> explicitly 2026-08-22). They preserve the source revisions, validation
> rules, and license posture established at that audit; they do not by
> themselves claim a newly run conversion, parity result, or publication. For
> current implementation and remaining-gate status, consult the converter
> source and the current license/handoff records. Any future checkpoint still
> needs its own immutable revision, provenance, license, and parity evidence.

NanoCodec conversion is intentionally split at the `.nemo` pickle boundary:

1. `tools/parity/nanocodec/prepare_checkpoint.py` runs in the pinned Python
   3.12/NeMo environment, verifies the imported distribution's PEP 610 source
   as the official `NVIDIA-NeMo/Speech` repository at commit
   `4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`, and emits decoder-only F32
   safetensors plus JSON.
2. `vokra-cli convert --model nanocodec` consumes only those two
   dependency-free files, verifies the complete tensor manifest and shapes,
   and writes GGUF.

```sh
uv run --project tools/parity/nanocodec \
  python tools/parity/nanocodec/prepare_checkpoint.py \
  --checkpoint /path/to/model.nemo \
  --model-id nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps \
  --revision 5c8e22ed763c14d81337fbe6ca74062f3d10f7e5 \
  --output /tmp/nanocodec.decoder.safetensors \
  --config-output /tmp/nanocodec.decoder.json

vokra-cli convert --model nanocodec \
  --input /tmp/nanocodec.decoder.safetensors \
  --config /tmp/nanocodec.decoder.json \
  --output /tmp/nanocodec.gguf
```

Every public metadata field is read from the restored checkpoint: number of
FSQ groups, levels per group, embedding width, sample rate, frame hop, decoder
channels/kernels/dilations, and upsample rates. Weight normalization is folded
by reading PyTorch's effective parametrized weight. Grouped transposed
convolutions are expanded to dense `[in, out, kernel]` tensors with explicit
zeros, matching the pinned official NeMo-Speech.cpp transform.

The sidecar and Rust converter independently require the model ID/revision and
the audited sample-rate, group/level, embedding, base-channel, frame-hop, and
upsample-rate tuple to agree. This prevents a valid checkpoint from one profile
being mislabeled as another. The 0.6 kbps checkpoint additionally has the
audited file SHA-256
`bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870`;
the sidecar checks it before opening the `.nemo` pickle.

## Audited checkpoint inventory (2026-08-22)

The official NVIDIA Hugging Face namespace exposes these three repositories:

| Repository | Immutable revision | FSQ groups | FSQ levels | Embed dim | Frame hop | Upsample rates |
|---|---|---:|---|---:|---:|---|
| `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps` | `5c8e22ed763c14d81337fbe6ca74062f3d10f7e5` | 4 | `[9,8,8,7]` | 16 | 1764 | `[7,7,6,3,2]` |
| `nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps` | `c4ab84a92c8d36a8b5a79eaea807cfaf7f03ed86` | 13 | `[8,7,6,6]` | 52 | 1764 | `[7,7,6,3,2]` |
| `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` | `fc00890b604aa2de298d2641ffc6c5f6caf8c4d7` | 8 | `[8,7,6,6]` | 32 | 1024 | `[8,8,4,2,2]` |

Issue #47 also names
`nvidia/nemo-nano-codec-22khz-0.8kbps-12.5fps`. That repository was not
published in the NVIDIA namespace at audit time, and NVIDIA's NanoCodec search
returned only the three repositories above. That id is deliberately rejected
by both the sidecar and converter, and is absent from the license registry and
publish sign-off map. A future official release needs its own immutable
revision, license audit, fixture, and parity result before the exact alias can
be added.

The 1.89 kbps checkpoint declares `frame_hop = 1024`; its decoder rates
`[8,8,4,2,2]` also multiply to 1024. The sidecar records both values and the
Rust converter requires them to agree, so topology drift fails closed before a
GGUF is written.

All three weights use the NVIDIA Open Model License Agreement (June 14, 2024).
The converter stamps `AttributionRequired` and the agreement's exact NOTICE
sentence. Model publication remains blocked until the separate license audit
owner sign-off succeeds; conversion and a green parity run do not authorize an
upload.
