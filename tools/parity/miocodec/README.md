# MioCodec official decoder parity

`miocodec_dump_reference.py` generates an independent reference from the
official `Aratako/MioCodec` implementation. It imports a clean checkout at
`77473544375d57e96cbdfd5d7d257e8f280fa8e3`, verifies the exact public v2
config and safetensors digests, and calls `MioCodecModel.decode`. Forward hooks
capture official module outputs; the oracle never imports Vokra or reimplements
the forward equations.

Run this workflow only on VAST. The 528 MB checkpoint is below the standalone
2 GB threshold, but instantiating the official config also materializes the
WavLM bundle and PyTorch runtime. Keeping both those payloads and the full
`vokra-models` build off the maintainer Mac avoids avoidable memory pressure.

```sh
git clone https://github.com/Aratako/MioCodec /workspace/MioCodec
git -C /workspace/MioCodec checkout 77473544375d57e96cbdfd5d7d257e8f280fa8e3
test -z "$(git -C /workspace/MioCodec status --porcelain --untracked-files=all)"

uv sync --python 3.12 --project /workspace/MioCodec --frozen

uv run --python 3.12 --project /workspace/MioCodec --frozen \
  hf download Aratako/MioCodec-25Hz-44.1kHz-v2 \
  config.yaml model.safetensors \
  --revision 67faba34153fe74e6665991c432a7327e23c5c1c \
  --local-dir /workspace/miocodec-v2

uv run --python 3.12 --project /workspace/MioCodec --frozen \
  python /workspace/vokra/tools/parity/miocodec_dump_reference.py \
  --source /workspace/MioCodec \
  --config /workspace/miocodec-v2/config.yaml \
  --weights /workspace/miocodec-v2/model.safetensors \
  --output /workspace/reference/miocodec-v2
```

The default eight-code input deliberately includes both codebook boundaries
and selects a target length whose official two-stage flooring interpolates 16
post-ConvTranspose frames down to 15 pre-upsampler frames. The resulting
fixture contains the versioned `VKRMIO01` input, stage taps, final PCM and a
digest-bearing manifest. Numerical tolerances are not declared here: they must
be calibrated from the first independent CPU and Metal runs rather than
invented before evidence exists.
