# WeSpeaker ResNet34-LM parity staging

This is a dedicated Python 3.12 reference project for the pinned official
WeSpeaker loader. The lock is immutable for the staged run:

- uv.lock SHA-256:
  996f10762498f29a8f6c24d3403ebac4734118f8150137b716ddf5d54e512b6e
- pyproject.toml SHA-256:
  4d5a2bae9fdd3dff3d1224235c6e125995f32e491e3f42bb2063281d2a9d1850
- The current 14-package closure is recorded and reviewed as one exact graph;
  it is not interchangeable with a generic parity project.
- The direct closure is limited to numpy, safetensors, torch, and torchaudio;
  their transitive runtime packages are resolver-recorded with complete HTTPS
  artifact URLs, SHA-256 values, and positive sizes. No dynamic `uv --with`
  dependencies are permitted.
- The resolver is constrained to Python 3.12 on Linux x86_64. Torch and
  torchaudio 2.9.1 are sourced only from the official PyTorch CPU index; the
  lock records the official x86_64 wheel sizes and hashes: torch
  184,378,187 bytes / `7417d8c565f219d3455654cb431c6d892a3eb40246055e14d645422de13b9ea1`,
  torchaudio 495,619 bytes /
  `43cf20a2965cf081945c91d2dc8844377e5e3f1b172c0d0c18399ca3ecf1f899`.

The independent dumper imports official wenet-e2e/wespeaker revision
45941e7cba2c3ea99e232d02bedf617fc71b0dad, loads only the pinned
Wespeaker/wespeaker-voxceleb-resnet34-LM avg_model at revision
f0c48c298fd835726c27956a5d617bad7115627e, and uses weights_only=True.
Its authenticated source contract includes `LICENSE`,
`wespeaker/models/resnet.py`, and `wespeaker/models/pooling_layers.py`
(10,255 bytes, SHA-256
768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9,
Git blob 47120eead47a511939267470496539804c17b7d3).
The checkpoint identity is SHA-256
9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449.

The corrected public replacement is separately bound to
vokra/pyannote-wespeaker-voxceleb-resnet34-lm, revision
8e27acd8a875088f1a7321f40610397bf964a446, file
pyannote-wespeaker.restamped.gguf, 26,584,064 bytes, SHA-256
6dccbc026e9c32a8f99f3441e64f1ff52e36afb055442595c86cda8021c78c39.
The old vokra/wespeaker artifact is retained only as a rejected, mislicensed
identity; its Apache stamp is not inherited by the replacement.

preflight_gate.py runs before host checks, scratch/cache creation, sync,
network, conversion, or Cargo. Production is currently expected to exit 2:
dependency and component review rows are pending. No approval or license value
is fabricated. The public-domain fixture input is
tests/fixtures/audio/jfk-30s.wav (352,078 bytes, SHA-256
58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f).

The VAST worker is no-upload and emits a portable quoted Apple invocation
containing expected GGUF and reference-manifest hashes. Apple validation is
separate, requires explicit hashes and an exact reference file set, and
requires singleton CPU/Metal result and parity sentinels before recording
evidence. Do not run the model, dependency sync, Cargo, VAST, Apple worker,
or upload on the maintainer machine.
