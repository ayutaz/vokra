# SpeechBrain Lang-ID CPU/Metal parity runbook

Status on 2026-08-26: source harness prepared; real CPU and Metal measurements
pending. No numeric parity result or public-artifact readiness is claimed by
this document.

## What the measurement proves

The independent oracle is
`tools/parity/speechbrain_lang_id_dump_reference.py`. It loads the immutable
official SpeechBrain release through `EncoderClassifier` and records four
stage boundaries:

1. sentence-normalized SpeechBrain fbank features;
2. the complete ECAPA embedding;
3. the official XVector/log-softmax or cosine classifier output;
4. the ordered label encoder and winning language.

`crates/vokra-models/tests/parity_speechbrain_lang_id_real.rs` checks the GGUF
source/revision, tensor count, exact axes, label order and checkpoint hashes
before executing a model. It reports stage metrics without a guessed numeric
bound. CPU runs on VAST. Metal runs on a remote Apple-silicon runner because
repository policy forbids `vokra-models` Cargo work and model inference on the
maintainer Mac.

## Immutable inputs

| Variant | Source | Revision | Axes | Head |
|---|---|---|---|---|
| VoxLingua107 | `speechbrain/lang-id-voxlingua107-ecapa` | `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9` | 60 mel / 256 embedding / 107 classes | XVector MLP + log-softmax |
| CommonLanguage | `speechbrain/lang-id-commonlanguage_ecapa` | `70a742bbc513f693efcf73d6d64a5ed14b3a34a4` | 80 mel / 192 embedding / 45 classes | cosine classifier |

Both Python tools default to the source-specific revision above. Passing an
override still requires a full 40-hex commit. The test rejects a reference and
GGUF produced from different revisions.

The fixed audio input is `tests/fixtures/audio/jfk-30s.wav` (mono PCM16,
16 kHz). The dumper validates that contract before inference.

## VAST CPU recipe

Follow `.agents/skills/vast-ai-workflow/SKILL.md` for rent, bundle transfer,
provisioning, evidence pull and unconditional destroy. Do not start an
authenticated VAST operation until the local VAST API key has been rotated.
No `--push` or Hugging Face upload is part of this recipe.

On the instance, from the exact bundled commit:

```sh
set -eu
uname -a
lscpu
rustc -Vv
uv sync --project tools/parity --frozen --python 3.12
cargo build --locked --release -p vokra-cli
```

Run VoxLingua107:

```sh
set -eu
case_root=/root/vokra-lang-id/voxlingua107
mkdir -p "$case_root"

uv run --project tools/parity --frozen --python 3.12 python \
  tools/parity/speechbrain_lang_id_prepare_checkpoint.py \
  --source speechbrain/lang-id-voxlingua107-ecapa \
  --revision 0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9 \
  --savedir "$case_root/upstream" \
  --output "$case_root/prepared.safetensors"

uv run --project tools/parity --frozen --python 3.12 python \
  tools/parity/speechbrain_lang_id_dump_reference.py \
  --source speechbrain/lang-id-voxlingua107-ecapa \
  --revision 0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9 \
  --savedir "$case_root/upstream" \
  --wav tests/fixtures/audio/jfk-30s.wav \
  --output-dir "$case_root/reference"

./target/release/vokra-cli convert \
  --model lang-id-voxlingua107 \
  --input "$case_root/prepared.safetensors" \
  --output "$case_root/model.gguf"

VOKRA_LANG_ID_GGUF="$case_root/model.gguf" \
VOKRA_LANG_ID_REFERENCE_DIR="$case_root/reference" \
  cargo test --locked --release -p vokra-models \
    --test parity_speechbrain_lang_id_real \
    measure_cpu_against_independent_speechbrain -- --ignored --nocapture
```

Run CommonLanguage with a separate savedir so checkpoint files cannot mix:

```sh
set -eu
case_root=/root/vokra-lang-id/commonlanguage
mkdir -p "$case_root"

uv run --project tools/parity --frozen --python 3.12 python \
  tools/parity/speechbrain_lang_id_prepare_checkpoint.py \
  --source speechbrain/lang-id-commonlanguage_ecapa \
  --revision 70a742bbc513f693efcf73d6d64a5ed14b3a34a4 \
  --savedir "$case_root/upstream" \
  --output "$case_root/prepared.safetensors"

uv run --project tools/parity --frozen --python 3.12 python \
  tools/parity/speechbrain_lang_id_dump_reference.py \
  --source speechbrain/lang-id-commonlanguage_ecapa \
  --revision 70a742bbc513f693efcf73d6d64a5ed14b3a34a4 \
  --savedir "$case_root/upstream" \
  --wav tests/fixtures/audio/jfk-30s.wav \
  --output-dir "$case_root/reference"

./target/release/vokra-cli convert \
  --model lang-id-commonlanguage \
  --input "$case_root/prepared.safetensors" \
  --output "$case_root/model.gguf"

VOKRA_LANG_ID_GGUF="$case_root/model.gguf" \
VOKRA_LANG_ID_REFERENCE_DIR="$case_root/reference" \
  cargo test --locked --release -p vokra-models \
    --test parity_speechbrain_lang_id_real \
    measure_cpu_against_independent_speechbrain -- --ignored --nocapture
```

Preserve the command log, both manifests, `sha256sum` output for the prepared
checkpoint/GGUF/reference files, and every `LANG_ID_MEASURE` line. Pull that
small evidence bundle before destroying the instance. Do not pull model
weights or GGUFs back to the maintainer Mac.

## Apple-silicon Metal recipe

Move the VAST-produced prepared checkpoint/reference bundle to a remote
Apple-silicon runner, or regenerate it there from the same pins. Do not use the
maintainer Mac. Record `system_profiler SPHardwareDataType`, `sw_vers`,
`rustc -Vv`, and the exact commit before measuring.

For each variant:

```sh
VOKRA_LANG_ID_GGUF=/remote/case/model.gguf \
VOKRA_LANG_ID_REFERENCE_DIR=/remote/case/reference \
  cargo test --locked --release -p vokra-models --features metal \
    --test parity_speechbrain_lang_id_real \
    measure_metal_against_cpu_and_independent_speechbrain \
    -- --ignored --nocapture
```

The Metal test separately measures embedding, classifier, complete network and
end-to-end scores against both SpeechBrain and CPU. Failure to create Metal or
execute Conv1d, Softmax or GEMV is a hard error; the runtime has no Metal-to-CPU
fallback.

## Turning measurements into gates

1. Inspect the worst element and its neighborhood for every stage, especially
   before treating a max-error outlier as a platform effect.
2. If CPU differs, rerun on the same VAST host with `VOKRA_CPU_ISA=scalar`.
   Scalar close to SIMD points away from a micro-kernel bug; scalar materially
   closer than SIMD points to the selected ISA implementation.
3. Establish a bound only after both variants' CPU measurements and the
   Apple-silicon Metal measurements are recorded. Do not widen the repository
   FP32 policy to fit an observation.
4. Commit the independent fixtures, measured floors, environment/evidence
   record and the derived max/aggregate/cosine gates together. Only then remove
   the `MEASUREMENT_ONLY` posture and claim numerical parity.

## Current blockers

- VAST CPU execution waits for the compromised/stale local VAST credential to
  be rotated.
- Metal execution waits for a remote Apple-silicon run; local execution is
  intentionally prohibited.
- The historical public `vokra/lang-id-voxlingua107` artifact is
  embedding-only and cannot satisfy the complete classifier/label contract.
  Replacing or uploading an artifact requires separate explicit permission.
