# MOSS Audio Tokenizer v2 reference gate

This dedicated Linux/x86_64 Python 3.12 VAST oracle resolves a 52-package
closure with Torch 2.7.1+cu126 from the official PyTorch CUDA index and
Transformers 5.5.0 from PyPI. The checked-in lock is resolver-generated and
records URL, SHA-256, and positive size metadata for every non-virtual
artifact. The dependency-free gate verifies those fields before any host,
scratch, cache, sync, model, or Cargo operation.

The upstream identity remains the immutable
`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2@f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169`
three-shard contract in `license_gate_manifest.json`. Package, source, weight,
and publication reviews remain fail-closed (`NO_UPLOAD`) until separately
authenticated owner evidence is supplied; a signed manifest cannot override
unresolved rows.

Run only the dependency-free self-test locally:

```text
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python license_gate.py --self-test
```

After an owner-authorized frozen sync on a clean Linux/x86_64 VAST checkout,
run `scripts/publish/vast-ai/audit-moss-audio-tokenizer-v2-dependencies.sh`.
The audit uses `--no-sync`, records the exact installed closure and native ELF
facts, fetches only the pinned upstream `LICENSE` and exact locked PyPI sdist
license candidates when publisher files are absent, and never downloads model
weights, imports Torch/model code, or invokes Cargo. It must complete before
the validation worker acquires the model or starts Cargo; unresolved factual
license evidence remains a blocker.
