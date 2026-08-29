# MMS-1B-All dedicated staging closure

This directory is intentionally not a copy of the broad `tools/parity`
project.  The MMS worker is allowed to run only after an owner supplies a
dedicated Python 3.12/Linux x86_64 CPU-only `pyproject.toml` and `uv.lock`
whose every registry artifact records an authenticated URL, SHA-256, byte
size, upload time, and approved registry host.

Those closure bytes are not present in the current checkout.  No dependency
version, artifact URL, hash, size, upload time, license, or approval is
invented here.  Consequently `license_gate.py` deliberately returns
controlled exit 2 for a normal run before VAST host probing, cache creation,
or network access.  CUDA, NVIDIA, and Triton dependencies are forbidden when
the closure is supplied.

The eventual manifest must cover the complete backbone (`model.safetensors`),
exactly one explicit `adapter.<language>.safetensors`, and its matching
`vocabs/<language>.txt`; the public ~8.9 MB adapter must never be represented as
the 1B backbone.  Package license/native-bundled review rows and independent
owner approval are mandatory.  Publication remains `NO_UPLOAD`, and all
reference/runtime statuses remain `INSPECTION_ONLY` until separate real
evidence is reviewed.
