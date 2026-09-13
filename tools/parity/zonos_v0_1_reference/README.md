# Zonos-v0.1 reference environment

This is a dedicated Python 3.12 project for the exact upstream Zonos commit
`bc40d98e1e1ab54fc65c483be127a90e3c7c0645`. It is separate from
`tools/parity/pyproject.toml` so unrelated parity tools cannot silently widen
the reference closure.

The project intentionally contains only the pinned import/runtime closure for
`Zonos.from_local`, the official conditioning modules, generation, and the
official DAC decoder. It does not include `phonemizer`, eSpeak, `librosa`,
`soundfile`, `soxr`, `cffi`, `libsndfile`, ONNX, or any optional hybrid/UI
packages. Zonos conditioning packets carry offline phoneme IDs; if the pinned
source attempts text phonemization, the reference import policy raises an
explicit error.

`uv.lock` is generated with `uv lock` and is bound by SHA-256 in every Zonos
consumer. The dependency audit remains `BLOCKED_UNREVIEWED_TRANSITIVE` until
an owner/legal decision reviews the Linux x86_64 closure, native ELF files,
and publisher LICENSE/NOTICE bytes. No model/source/checkpoint acquisition,
execution, or publication is authorized by this project alone; all workers
remain `NO_UPLOAD`.

The security-fixed closure pins `transformers==5.10.4` and its compatible
`huggingface-hub==1.5.0`; the previous `transformers==4.48.1` was in the
`GHSA-xrqw-3rrv-vx5w` affected range. The upstream Zonos source/API has not
yet been smoke-tested against this patched closure. Consequently the worker
stops with `BLOCKED_UNVERIFIED_TRANSFORMERS_API_SMOKE` before any source or
checkpoint acquisition, and no compatibility result is inferred from the
lock alone.

## Dependency approval transition

The model-free audit emits `BLOCKED_UNREVIEWED_TRANSITIVE` because dependency
facts and owner/legal classification are separate decisions. A later worker
may cross the pre-acquisition boundary only when an owner supplies an external
`vokra-zonos-dependency-approval-v1` record. The record is distinct from the
source/model approval and must bind the audit file SHA-256, its exact
`candidate_scope_sha256`, exact clean HEAD, and every scope digest: lock,
prepared NumPy, no-forbidden-BLAS policy/runtime, installed closure, native
files, publisher files, NumPy RECORD, and publisher archive manifest. It must
also carry an explicit `owner/legal` attestation and a canonical signature.

`run-zonos-inspection.sh` reads this record from
`ZONOS_DEPENDENCY_APPROVAL` and binds its raw digest with
`ZONOS_DEPENDENCY_APPROVAL_SHA256`. Missing, stale, unsigned, duplicate-key,
placeholder-signer, wrong-head/hash, `NO_UPLOAD`-inconsistent, symlinked, or
overlapping records remain fail-closed. Successful validation authorizes only
the next pre-acquisition stage; the factual audit status and publication
disposition do not change.
