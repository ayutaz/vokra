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

`dependency_audit.py` is the model-free VAST audit for the same 43 lock rows
(40 Linux distributions, two platform-selected rows, and the virtual project
row). It follows dependency edges from the virtual root using the Linux
CPython marker set; the Windows-only `pyreadline3` edge is therefore excluded
from the Linux closure.
It records exact installed distribution metadata, publisher license-file
hashes, locked PyPI-sdist license bytes when a wheel has no license file, and
ELF `NEEDED` facts for native payloads. The audit is intentionally
`BLOCKED_OWNER_REVIEW` (or `BLOCKED_FACTUAL_AUDIT` when evidence is missing):
it never changes the pending gate to `AUDITED_ALLOW`, and it does not import
ESPnet/model code or request a checkpoint. The VAST source-only worker syncs
this frozen project on the disposable host, writes the no-replace audit report,
then stops before source, BPE, or model acquisition so the evidence can be
reviewed independently. For the pinned `torch-complex==0.4.4`, the official
PyPI sdist/wheel and the official v0.4.4 source tag contain no license bytes;
the report consequently remains the exact factual blocker
`primary license bytes unavailable: torch-complex==0.4.4`. A metadata
classifier or `setup.py` declaration is not substituted for license bytes, and
this blocker must not be removed by inference.

Run only the dependency-free self-test locally:

```text
uv run --no-project --offline --python 3.12 python dependency_audit.py --self-test
```
