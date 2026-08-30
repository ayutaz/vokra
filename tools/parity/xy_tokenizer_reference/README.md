# XY-Tokenizer official reference project

This is an offline sidecar for the pinned official XY-Tokenizer source. It is
VAST-only and is not part of the zero-dependency Rust runtime. The upstream
README advertises Python 3.10+ and PyTorch 2.0+, while its tracked
`requirements.txt` is unversioned. The project therefore fixes the repository
runtime policy to Python 3.12 and Linux x86_64, and routes only Torch and
Torchaudio through the official CPU wheel index. CUDA, NVIDIA and Triton are
not permitted.

The authenticated model route imports only `numpy`, `torch`, `torchaudio`,
`einops`, `librosa`, `pyyaml`, `transformers`, and `scipy`. `debugpy` appears
only in `utils/helpers.py`, which is not imported by `xy_tokenizer/model.py`
or its frontend/encoder/RVQ/Vocos route; it is excluded from this closure.

The exact VAST-generated CPU-only `uv.lock` is tracked at the pinned digest
`ba26854d2cd1d695195fc906dde3d02f1fbf7ccc1d154e6015aaaa0aec44c049` (57 package
rows). `license_evidence.json` is tracked only as an empty template. VAST must collect
publisher or locked-sdist license bytes plus native bundled payload evidence.
Run `scripts/publish/vast-ai/run-xy-tokenizer-dependency-audit.sh` on a clean
Linux/x86_64 VAST checkout to collect exact lock artifact bytes into a separate
evidence directory; the tracked template remains untouched and the final audit
uses an explicit evidence override. It emits a report-only result. Run
`audit.py` and `collect_evidence.py` only with fake fixtures locally. The audit emits `BLOCKED`,
`NO_UPLOAD`, and owner sign-off required; it cannot self-assert
`AUDITED_ALLOW`.

The approval workflow is: VAST generates `uv.lock` and evidence, `audit.py`
binds both digests and emits the blocked review manifest, then an independent
owner supplies an external approval record containing the exact project scope
digest, signatory, decision, and reviewed evidence digest. Only that recorded
decision may permit a separate Luna implementation/review to materialize a
dumper-compatible `dependency_audit.json` with `AUDITED_ALLOW`; this is not
automatic functionality, and the generator never creates that status itself.

The source route remains blocked by the separate topology/native-runtime and
numerical-parity gates. No model, checkpoint, dependency sync, or network
operation is allowed on the maintainer Mac.
