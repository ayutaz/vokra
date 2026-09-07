# MOSS Audio Tokenizer Nano reference gate

This is a dedicated Python 3.12, Linux/x86_64 VAST oracle project for
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` at revision
`6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`. It is separate from the general
parity environment and contains a resolver-generated 52-lock-row closure for
Linux/x86_64 Python 3.12: Torch 2.7.1+cu126 from the official PyTorch CUDA
index and the isolated security pin Transformers 5.10.4 from PyPI. The prior
5.5.0 pin is previous isolated-reference provenance only; no upstream API
compatibility is claimed.
The 52 lock rows comprise 51 active installed distributions plus one virtual
project row; the virtual row is not an installed package.
Every non-virtual lock row carries
resolver URL, SHA-256, and positive artifact-size metadata. No package sync is
performed by the local gate.

The exact upstream payload contract at the fixed revision is eight files: the
seven non-weight files `.gitattributes`, `README.md`, `__init__.py`,
`config.json`, `configuration_moss_audio_tokenizer.py`,
`modeling_moss_audio_tokenizer.py`, and `model.safetensors.index.json`, plus
`model-00001-of-00001.safetensors`. There is no `LICENSE` file in that
complete tree. The authenticated HF model card reports `cardData.license` as
`apache-2.0`; this is recorded separately from the absent license file. The
2026-08-01 owner sign-off by yousan in `docs/license-audit.md:671` records
Apache-2.0 / Commercial for source and weights; it does not approve the Python
closure, API route, or parity. The manifest binds authenticated non-weight
byte/SHA-256 and canonical Git-blob SHA-1 identities, while the shard remains
server-identity-only with `content_not_downloaded=true`, server size 87922568,
LFS payload SHA-256, and LFS pointer Git-blob SHA-1. Transformers 5.10.4 is above the GHSA-xrqw-3rrv-vx5w
patched minimum of 5.10.0, but no authenticated API smoke has been run. The
meta-device inspection authenticated quantizer shape `1x768x2` and nine
decoder taps through `decoder_8`; real-weight/API compatibility and numerical
parity remain unresolved and blocked.
The fixed source `config.json` also binds the official Transformers mapping
`AutoConfig -> MossAudioTokenizerConfig` and `AutoModel ->
MossAudioTokenizerModel`. The inspection checks this mapping and the
shape-bearing decoder layout before construction, then requires the official
model methods `encode`, `decode`, `forward`, and `create_decode_session`.
Its ordered meta taps are `quantizer 1x768x2`, `decoder_0 1x192x8`,
`decoder_1 1x768x8`, `decoder_2 1x384x16`, `decoder_3 1x768x16`,
`decoder_4 1x384x32`, `decoder_5 1x768x32`, `decoder_6 1x384x64`,
`decoder_7 1x240x64`, and `decoder_8 1x1x15360`; after official channel
restoration the audio shape must be `1x2x7680`.
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

The inspection materializes only the seven non-weight files on VAST. It does not
download the model shard: that shard is authenticated solely from the expanded
HF server-tree Git/LFS identity and the checkpoint index reference. The report
records materialized SHA-256 and canonical Git-blob SHA-1 values for the seven
files and server size/LFS identity for the shard, then checks the official
`AutoConfig.from_pretrained`
plus meta-device
`AutoModel.from_config` route. Decoder and audio shapes are observed by
meta-device shape propagation; no safetensors tensor is loaded or executed.
The output remains `BLOCKED` with source/weight `REVIEWED` (owner sign-off
`docs/license-audit.md:671`) but unresolved Python closure, API/runtime
approval, and `NOT_RUN` numerical parity; publication remains `NO_UPLOAD`. The
evidence also binds the exact clean Vokra checkout `{expected_head, head,
clean}`. A complete inspection intentionally exits 2 so its evidence must be
recovered and reviewed before any conversion or parity worker is started.
An `INSPECTION_ERROR` manifest is never treated as complete.

Evidence output is no-clobber: the inspector refuses an existing output path,
including a prior evidence directory, and all blocked/error outcomes remain
exit status 2.

The owner approval path is `MOSS_AUDIO_TOKENIZER_NANO_LICENSE_APPROVAL`; the
tracked manifest still cannot be self-approved because Python closure, API,
runtime, and parity gates remain unresolved.

The dependency/native-payload audit is a separate no-model VAST phase.  After
the exact project has been synced on the disposable Linux/x86_64 host, run:

```text
scripts/publish/vast-ai/audit-moss-audio-tokenizer-nano-dependencies.sh \
  --expected-head <40-hex-commit> \
  --output /dev/shm/moss-audio-tokenizer-nano-dependency-audit.json
```

It uses `--no-sync` and records every locked artifact URL/hash/size, installed
package license/EULA bytes, and hashes plus ELF `NEEDED` facts for native
payloads.  CUDA/NVIDIA and Triton distributions are called out explicitly.
It never requests model files, imports model code, invokes Cargo, converts, or
publishes.  A blocked report is expected until the owner reviews the package
and native-payload rows; it is not a Python/API/runtime/parity approval.  The
required `--expected-head` is checked against a clean Vokra checkout and is
bound into the report.  Report creation is atomic and no-replace; a concurrent
creator cannot be overwritten.
