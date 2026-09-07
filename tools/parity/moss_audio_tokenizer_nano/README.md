# MOSS Audio Tokenizer Nano reference gate

This is a dedicated Python 3.12, Linux/x86_64 VAST oracle project for
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` at revision
`6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`. It is separate from the general
parity environment and contains a resolver-generated 52-package closure for
Linux/x86_64 Python 3.12: Torch 2.7.1+cu126 from the official PyTorch CUDA
index and the isolated security pin Transformers 5.10.4 from PyPI. The prior
5.5.0 pin is previous isolated-reference provenance only; no upstream API
compatibility is claimed.
Every non-virtual lock row carries
resolver URL, SHA-256, and positive artifact-size metadata. No package sync is
performed by the local gate.

The exact upstream payload contract is seven files: `LICENSE`, `README.md`,
`config.json`, `configuration_moss_audio_tokenizer.py`,
`modeling_moss_audio_tokenizer.py`, `model.safetensors.index.json`, and
`model-00001-of-00001.safetensors`. This checkout does not contain authenticated
byte/SHA-256 evidence for those files at the fixed revision. The manifest
therefore records those identities as unresolved and the dependency/reference
route as unresolved. Transformers 5.10.4 is above the GHSA-xrqw-3rrv-vx5w
patched minimum of 5.10.0, but no authenticated API smoke has been run. The
decoder tap count/shapes and quantizer output shape
are also explicitly unresolved contract fields; they are not wildcards.
`license_gate.py` intentionally exits 2 before any uv
sync, source/model acquisition, conversion, Cargo, or CUDA work.

The first public `vokra/moss-audio-tokenizer-nano` GGUF is historically
mis-stamped with Full metadata and is never accepted by this gate. A corrected
replacement may only be converted on VAST and remains `MEASURED_NOT_GATED`
until an owner reviews the independent official reference and numerical bound.

Run only the dependency-free gate and its self-test locally:

```text
uv run --no-project --python 3.12 python license_gate.py --self-test
```

Before owner approval, the fixed-revision source contract can be collected on
a disposable VAST Linux host without running a model:

```text
scripts/publish/vast-ai/run-moss-audio-tokenizer-nano-inspection.sh \
  --expected-head <40-hex-commit>
```

The inspection materializes only the six non-weight files on VAST. It does not
download the model shard: that shard is authenticated solely from the expanded
HF server-tree Git/LFS identity and the checkpoint index reference. The report
records materialized SHA-256 and canonical Git-blob SHA-1 values for the six
files and server size/LFS identity for the shard, then checks the official
`AutoConfig.from_pretrained`
plus meta-device
`AutoModel.from_config` route. Decoder and audio shapes are observed by
meta-device shape propagation; no safetensors tensor is loaded or executed.
The output remains `BLOCKED` with `OWNER_SIGNOFF_REQUIRED`, `NOT_RUN`, and
`NO_UPLOAD`. A complete inspection intentionally exits 2 so its evidence must
be recovered and reviewed before any conversion or parity worker is started.
An `INSPECTION_ERROR` manifest is never treated as complete.

Evidence output is no-clobber: the inspector refuses an existing output path,
including a prior evidence directory, and all blocked/error outcomes remain
exit status 2.

The owner approval path is `MOSS_AUDIO_TOKENIZER_NANO_LICENSE_APPROVAL`; the
tracked manifest remains `OWNER_SIGNOFF_REQUIRED` and cannot be self-approved.
