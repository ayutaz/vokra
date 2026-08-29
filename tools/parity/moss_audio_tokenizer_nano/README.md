# MOSS Audio Tokenizer Nano reference gate

This is a dedicated Python 3.12, Linux/x86_64 VAST oracle project for
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` at revision
`6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`. It is separate from the general
parity environment and contains a resolver-generated 52-package closure for
Linux/x86_64 Python 3.12: Torch 2.7.1+cu126 from the official PyTorch CUDA
index and Transformers 5.5.0 from PyPI. Every non-virtual lock row carries
resolver URL, SHA-256, and positive artifact-size metadata. No package sync is
performed by the local gate.

The exact upstream payload contract is seven files: `LICENSE`, `README.md`,
`config.json`, `configuration_moss_audio_tokenizer.py`,
`modeling_moss_audio_tokenizer.py`, `model.safetensors.index.json`, and
`model-00001-of-00001.safetensors`. This checkout does not contain authenticated
byte/SHA-256 evidence for those files at the fixed revision. The manifest
therefore records those identities as unresolved and the dependency/reference
route as unresolved. The decoder tap count/shapes and quantizer output shape
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

The owner approval path is `MOSS_AUDIO_TOKENIZER_NANO_LICENSE_APPROVAL`; the
tracked manifest remains `OWNER_SIGNOFF_REQUIRED` and cannot be self-approved.
