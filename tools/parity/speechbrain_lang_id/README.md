# SpeechBrain VoxLingua107 Lang-ID validation

This is a dedicated Python 3.12 Linux/x86_64 VAST oracle project for
`speechbrain/lang-id-voxlingua107-ecapa` at revision
`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`. It is intentionally separate from
the broad parity environment. The exact official loader closure is pinned in
`pyproject.toml` and `uv.lock`.

The three upstream payloads are `embedding_model.ckpt` (84474355 bytes,
`ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a`),
`classifier.ckpt` (762555 bytes,
`a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc`), and
`label_encoder.txt` (2204 bytes,
`9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f`). The
loader config is `hyperparams.yaml` (1519 bytes,
`88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f`) and
metadata is `config.json` (51 bytes,
`a861f8fbc2e23c0fc0823b3c0fd2b3d1e839563c2d4e3f9663a1237cce62bc89`). The
complete dependency/license review remains unresolved, so `preflight_gate.py`
exits 2 before host probing, cache creation, sync, network, model acquisition,
conversion, Cargo, or CUDA. Owner signoff cannot override an identity or
closure gate.

The checked-in lock is a genuine `uv lock` resolution for Linux/x86_64 Python
3.12, with 38 package rows and resolver-emitted artifact URL/hash/size rows.
The official CPU Torch index reports Content-Length 191817609 for the pinned
Torch 2.13.0+cpu wheel and 341338 for the pinned TorchAudio 2.11.0+cpu wheel;
those values are bound into the lock and checked by the fail-closed preflight.
It uses the official CPU Torch index (`torch==2.13.0+cpu` and
`torchaudio==2.11.0+cpu`). GitHub Dependabot's four open Torch alerts for this
manifest report patched floors through 2.13.0 (alerts #434--#437). TorchAudio's
official installation documentation states that 2.11 is built against the
stable ABI and supports PyTorch 2.11 and every later release, including 2.13
(<https://docs.pytorch.org/audio/main/installation.html>). The complete
dependency/license review remains unresolved, so the production gate still
exits 2.

The fixed local fixture is `tests/fixtures/audio/jfk-30s.wav` (352078 bytes,
SHA-256 `58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f`).
No unsafe pickle loading is
allowed: the official SpeechBrain loader remains the only checkpoint reader.

`model_free_api_probe.py` is an import-only, source-ready probe for the
official SpeechBrain 1.1.1 release. It verifies the pinned Torch/TorchAudio
versions, imports SpeechBrain without a compatibility shim, and checks that
the official classifier exposes the `encode_batch` and `classify_batch` APIs
used by the current dumper. It forces Hub offline and never downloads weights,
instantiates a model, or executes inference. `MODEL_FREE_API_VALIDATED` means
only that this import/API contract is available; it is not a CPU parity or
Metal parity result. The dumper's former `torchaudio.list_audio_backends`
monkey-patch was removed; any import failure now blocks the source-ready
candidate instead of being hidden by a compatibility shim.

## SpeechBrain 1.1.1 source-ready candidate

SpeechBrain 1.1.1 is the official PyPI release from 2026-08-27 and is
Apache-2.0. Its official backend implementation guards the removed
`torchaudio.list_audio_backends` API for TorchAudio 2.9+:
[release backend](https://github.com/speechbrain/speechbrain/blob/v1.1.1/speechbrain/utils/torch_audio_backend.py)
and [PyPI release](https://pypi.org/project/speechbrain/1.1.1/). The fixed
PCM input path in the current dumper remains the stdlib `wave` reader; no input
fixture or parity tolerance was changed here.

The release also changes SpeechBrain's internal audio loading from
`torchaudio.load/info` to its `soundfile`-based `audio_io` layer and adds
`requests` and `soundfile` to the locked SpeechBrain dependency closure. The
existing direct `soundfile` and `requests` pins remain, but the native
`libsndfile` payload and the complete Python closure still require the
separate owner/license review. The manifest therefore keeps every dependency
and native-bundled review row unresolved and keeps `publication_decision` at
`NO_UPLOAD`.

This is a source-ready candidate only. Real-weight CPU parity, independent
reference evidence, and subsequent Apple CPU/Metal validation have not been
run for SpeechBrain 1.1.1; no existing fixture or numeric bound is promoted.

All real conversion and measurements are VAST-only and no-upload. Numeric
bounds remain unset; evidence is measurement-only until CPU and Metal results
are independently reviewed.
