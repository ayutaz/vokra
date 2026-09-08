# OWSM v4 medium 1B official frontend reference project

This is a dedicated Python 3.12 project for the VAST-only official ESPnet
frontend reference. It is not a Vokra runtime dependency and must not be used
to execute a model checkpoint locally.

The project is intentionally fail-closed until the pinned dependency closure
has an owner-reviewed primary-source license audit. Its lock is metadata-only
until a disposable VAST worker installs it. The CPU-only PyTorch index is
explicit so a worker cannot resolve CUDA/Triton wheels.

The direct packages are the statically authenticated import closure of the
pinned ESPnet roles: `humanfriendly`, `librosa`, `numpy`, `packaging`,
`torch`, `torch-complex`, `typeguard`, and `PyYAML`. Their exact versions are
recorded by `uv.lock`; transitive packages remain part of the pending audit.

The worker must use `uv run --frozen --project
tools/parity/owsm_v4_medium_1b_reference` and must not run `uv sync` on the
maintainer workstation. This project does not authorize checkpoint/BPE
retrieval, upload, or model execution.
