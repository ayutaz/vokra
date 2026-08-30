# Parler-TTS official-reference parity

This directory is a VAST-only independent oracle for the public
`vokra/parler-tts-mini-v1` and `vokra/parler-tts-mini-multilingual` GGUFs. It
imports `ParlerTTSForConditionalGeneration` from the official
`huggingface/parler-tts` repository at commit
`d108732cd57788ec86bc857d99a6cabd66663d68`, with Transformers 4.46.1. It does
not import Vokra or reproduce the Rust graph in Python.

The exact Python closure is the reviewed `uv.lock`: Python 3.12, NumPy 1.26.4,
Torch 2.11.0+cpu and TorchAudio 2.11.0+cpu from the official PyTorch CPU index,
and Transformers 4.46.1. TorchAudio's 2.11 stable-ABI contract supports the
matching 2.11 pair; this project does not infer compatibility from an omitted
lock dependency. This is based on the [official TorchAudio installation
matrix](https://pytorch.org/audio/stable/installation.html) and the
[official PyTorch previous-versions table](https://pytorch.org/get-started/previous-versions/),
not an ABI-mismatch guess. The official source's local `parler_tts.dac_wrapper` is the
runtime DAC path. The official setup metadata declares `descript-audio-codec`,
`descript-audiotools`, `librosa`, `soxr`, `soundfile`, and `protobuf`, but this
fixed dumper path does not import them; they are not silently pulled into this
inference closure. Any future change must be reviewed and added explicitly.

The fixed model identities are English
`parler-tts/parler-tts-mini-v1@0392b9451a601e528fd863bbb0598431fee810d9` and
Multilingual
`parler-tts/parler-tts-mini-multilingual-v1.1@11b27d57855dec1ce0914ba1f12363bf2ea75ba3`,
both Apache-2.0, plus the MIT DAC component
`parler-tts/dac_44khZ_8kbps@5cf6b8ad50fbb17e52c341410a1d00083201b6a9`.
Public GGUF revisions, file identities, and hashes are recorded in
`license_gate_manifest.json`; its DAC model entry is the immutable 306,642,416
byte LFS SHA-256 identity reported by the pinned HF tree, and the 227-byte
config is bound to its immutable Git blob identity. DAC license/native/bundled
review still requires owner approval, so production is intentionally
fail-closed.

Before any worker scratch creation, source checkout, sync, or model download,
`preflight_gate.py` binds exact project/lock bytes, canonical package
version/source/marker/dependency rows, model/source/DAC identities, and
version-keyed license/native/bundled review rows to authenticated operator
evidence. `PENDING_REVIEW`, `UNRESOLVED`, null, and placeholder rows cannot
pass, even if editable signer fields are populated.

The no-upload worker is:

```sh
bash scripts/publish/vast-ai/run-parler-tts-validation.sh
```

For each release the oracle runs a fixed, explicit description-token sequence
and a separate fixed prompt-token sequence. It records the official FLAN-T5
hidden states, four-frame greedy delayed code packet, and official embedded-DAC
PCM. The Rust gate compares codes exactly, compares the T5 hidden states under
the recorded FP32 ceiling, then decodes the official packet independently and
applies `max_abs <= 0.01` to PCM. These legs separate text-encoder, LM/schedule,
and codec drift.

The worker has no upload or push option. Pull its `logs/` and `reference/`
directories before destroying the VAST instance. Metal execution is a separate
remote Apple Silicon run using the same GGUF and reference; never execute these
real models on the maintainer Mac.

After the authorized frozen `uv sync`, the worker invokes
`scripts/publish/vast-ai/audit-parler-tts-dependencies.sh` before acquiring any
checkpoint or building Cargo. The audit is VAST/Linux x86_64-only and uses
`--frozen --no-sync`; it records every lock row, the exact normalized installed
name/version multiset, package license/NOTICE bytes and hashes, and native ELF
`readelf -d` `NEEDED` entries. It fetches only the exact `LICENSE` paths for the
Parler source, two pinned models, and DAC revision. Missing or redirected
non-license paths are factual blockers, and no license class is inferred from
the returned bytes. The audit can inspect only an environment synchronized by a
separately authorized, named VAST job; it does not download weights, import
model/Torch code, or invoke Cargo.
