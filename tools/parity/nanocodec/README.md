# NanoCodec checkpoint preparation, Group-FSQ, and causal HiFi-GAN parity

This isolated Python 3.12 uv project prepares audited checkpoints for issue
#47 and regenerates the independent Group-FSQ and causal HiFi-GAN reference
fixtures for issues #45 and #46. It is not part of the Rust runtime and does
not alter the root `Cargo.lock`.

The lock pins `nemo-toolkit[tts]` directly to NVIDIA-NeMo/Speech commit
`4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`, matching the version recorded by
the released checkpoint. The pinned TTS package initializer imports
`EasyMagpieTTSInferenceModel`, whose import-time SpeechLM2 helper imports
`peft`, and PEFT 0.20.0 imports `accelerate`; both therefore remain explicit
in the isolated lock. In contrast, the only TTS source path that imports
NLTK is the G2P implementation, which is not in the AudioCodecModel import
graph. The lock omits only `nltk` with an impossible marker override; a
model-free closure guard fails closed if NLTK reappears or the required
PEFT/Accelerate edges disappear. The dumper additionally checks
that the imported `GroupFiniteScalarQuantizer` source is inside a checkout at
that exact commit and refuses any fallback implementation.

The VAST import probe was model-free and blocked NLTK in-process. It also used
process-local compatibility shims for the pinned NeMo/Lightning OneLogger
`overrides` signature mismatch and the missing optional `NeptuneLogger` symbol.
Those shims changed neither installed source nor dependency metadata and are
audit-only; they must not be added to production or reference execution. A
normal unshimmed import remains blocked by that upstream API mismatch, so the
probe proves only that the official `AudioCodecModel` import path does not
require NLTK under the audited environment. It does not claim that the full
checkpoint/reference execution is currently runnable.

The dependency closure can be checked without installing NeMo or opening a
checkpoint:

```sh
uv run --no-project --python 3.12 \
  python tools/parity/nanocodec/dependency_guard.py --self-test
```

When an exact source checkout is available, add
`--nemo-source-root /path/to/NVIDIA-NeMo-Speech` to audit the pinned TTS source
boundary as well. The guard intentionally fails closed when that source root
is omitted from the full audit mode.

Run from the repository root:

```sh
uv sync --python 3.12 --project tools/parity/nanocodec --frozen
uv run --python 3.12 --project tools/parity/nanocodec \
  python tools/parity/fsq_dump.py nanocodec \
  --checkpoint /path/to/nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo \
  --nemo-source-root /path/to/NVIDIA-NeMo-Speech \
  --out tests/parity/fsq/nanocodec \
  --time 16
```

The checkpoint must be the file at HF repo commit
`5c8e22ed763c14d81337fbe6ca74062f3d10f7e5`, SHA-256
`bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870`.
The checkpoint remains a temporary local/VAST input and is never committed.

## Checkpoint preparation

This directory is the only trusted Python/pickle boundary for NVIDIA NeMo
NanoCodec conversion. The project is locked to Python 3.12, the official
`NVIDIA-NeMo/Speech` commit recorded in `pyproject.toml`. The dependency guard
keeps the unused NLTK branch out while retaining the required PEFT/Accelerate
import closure.

Prepare one of the three audited public checkpoints:

```sh
uv sync --project tools/parity/nanocodec --locked --python 3.12
uv run --project tools/parity/nanocodec --python 3.12 \
  python tools/parity/nanocodec/prepare_checkpoint.py \
  --checkpoint /path/to/model.nemo \
  --model-id nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps \
  --revision 5c8e22ed763c14d81337fbe6ca74062f3d10f7e5 \
  --output /tmp/nanocodec.decoder.safetensors \
  --config-output /tmp/nanocodec.decoder.json
```

The script verifies the installed NeMo PEP 610 source, immutable model
revision, audited geometry, and the 0.6 kbps checkpoint SHA-256 before opening
the `.nemo` pickle. The emitted safetensors and JSON are then consumed by the
dependency-free Rust converter. No model is uploaded by this workflow.

## Causal HiFi-GAN reference bridge

This is an offline, independent reference path for issue #46. It imports the
official NVIDIA NeMo implementation pinned at commit
`4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`; it does not reproduce the Rust
forward in Python. The isolated lock excludes only the unused NLTK branch from
the upstream TTS extra while retaining the required PEFT/Accelerate import
closure; the reference module loads a real `.nemo` checkpoint,
materializes NeMo's weight-normalized decoder tensors, runs
`CausalHiFiGANDecoder.forward`, and writes a temporary binary fixture consumed
by `crates/vokra-models/tests/parity_nanocodec_causal_hifigan.rs`.

The dumper fails unless the imported `nemo_toolkit` PEP 610 provenance names
the official `NVIDIA-NeMo/Speech` repository at that exact commit. It also
requires the audited 0.6 kbps checkpoint at HF revision
`5c8e22ed763c14d81337fbe6ca74062f3d10f7e5` and verifies SHA-256
`bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870`
before unpickling. The fixture records those identifiers together with the
actual torch version, CPU capability, platform, CPU count, and torch thread
count so a parity result remains attributable to its execution environment.

The fixture contains the real decoder weights and can be large. Do not commit
it. A model artifact at or above 2 GB and every `vokra-models` build/test belong
on VAST under the repository safety rules.

```sh
uv sync --project tools/parity/nanocodec --python 3.12
uv run --project tools/parity/nanocodec --python 3.12 tools/parity/nanocodec/dump_reference.py \
  --checkpoint /workspace/model.nemo \
  --checkpoint-id nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps \
  --checkpoint-revision 5c8e22ed763c14d81337fbe6ca74062f3d10f7e5 \
  --output /workspace/nanocodec-causal-hifigan.vknp

VOKRA_NANOCODEC_PARITY_FIXTURE=/workspace/nanocodec-causal-hifigan.vknp \
  CARGO_BUILD_JOBS=1 cargo test -p vokra-models \
  --test parity_nanocodec_causal_hifigan -- --nocapture
```

The Rust gate was pre-registered before the first reference run:

- maximum absolute waveform error: `2e-4`;
- waveform RMSE: `2e-5`.

These bounds cover f32 accumulation plus the at-most-ULP scalar sine difference
while remaining far below 16-bit PCM quantization. A failure requires operator
or fixture investigation; do not relax the bounds from a failing observation.

The published 21.5 fps archive declares `samples_per_frame: 1024` and
`up_sample_rates: [8, 8, 4, 2, 2]`, whose product is also 1024. This was
verified directly from `model_config.yaml` in the fixed official checkpoint;
the bridge accepts that consistent geometry. It still fails closed whenever a
declared hop and the generator product differ, rather than discarding or
inventing reference audio.
