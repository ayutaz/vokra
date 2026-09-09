# UTMOS parity fixtures (M5-15)

## Current status (2026-09-09)

The committed fixture and native UTMOS harness remain useful historical
artifacts, but the legacy SaruLab Lightning checkpoint cannot currently be
prepared safely. `tools/parity/utmos_dump_reference.py` is an explicit
`BLOCKED_UNSAFE_PICKLE` stub: it does not import torch, download upstream
sources, read a checkpoint, or write reference output. The current
`tools/parity/utmos_prepare_checkpoint.py` uses only an explicit restricted
`weights_only=True` loader and stops when that loader rejects the checkpoint;
there is no unsafe fallback.

Therefore current CI claims only the model-free self-tests and the
safe-loader refusal boundary; it claims no numeric UTMOS parity. Re-enabling
real reference generation requires owner-approved safe state-dict wiring,
separate from the MIT license sign-off in `docs/license-audit.md` §3.1.

The checkpoint is never committed and Vokra ships no UTMOS weights. Future
checkpoint preparation, reference generation, conversion, or `vokra-models`
Cargo verification must run in the approved VAST workflow, not on the
maintainer Mac. Publication remains `NO_UPLOAD` until separately authorized.

## Historical fixture record

The files below were generated from the real upstream implementation before
the safe-loader boundary was tightened. They are retained as historical,
non-rerunnable evidence and must not be presented as current executable proof.

What is committed here:

| file | what |
|---|---|
| `ref-clip.wav` | 2 s mono 16 kHz PCM16, cut from `tests/fixtures/audio/jfk-30s.wav` (offset 0.5 s). Small enough to keep the 99-frame parity run fast, long enough to exercise the whole stack. |
| `score.json` | The upstream score for that clip + the honest tolerance and its derivation. |

**Still not committed, deliberately:** the checkpoint itself. The weights stay
owner-gated pending the `docs/license-audit.md` §3.1 UTMOS sign-off, and Vokra
ships no weights.

## The two harnesses

| test | what it checks | env |
|---|---|---|
| `crates/vokra-eval/tests/parity_utmos.rs` | final score vs `score.json` | `VOKRA_UTMOS_GGUF` |
| `crates/vokra-eval/tests/parity_utmos_stages.rs` | **every stage** vs the upstream hook points | `VOKRA_UTMOS_GGUF` + `VOKRA_UTMOS_REFDIR` |

The stage harness is the load-bearing one: a single scalar cannot localize a
fault (a swapped `ln1`/`ln2` mapping, a mis-folded weight-norm and a backwards
LSTM direction all just read as "wrong number"), so the per-stage comparison is
what turns a failure into a named stage.

## Historical regeneration recipe (currently blocked)

The following recipe documents the former VAST procedure. Do not run it until
owner-approved safe state-dict wiring is supplied. The current safe loader must
not be bypassed.

```bash
# 0. environment — measured, not assumed (M5-15 T38; docs/adr/M5-15-utmos.md §(d)).
#    Python 3.9 + torch 2.8.0 + fairseq @ d03f4e77 + pytorch-lightning 1.9.5 + omegaconf 2.1.2.
#    Python 3.11 does NOT work (fairseq@2022 trips 3.11's tightened dataclass check);
#    the upstream pin torch==1.11.0 has no macOS-arm64 wheel at all.
tools/parity/utmos_env_probe.sh          # records which branch this machine lands on

# 1. flatten the upstream .ckpt → safetensors + config side-car
uv run --project tools/parity --frozen --python 3.12 python tools/parity/utmos_prepare_checkpoint.py \
    --ckpt "$CKPT" --output /tmp/utmos.safetensors --config-out /tmp/utmos-config.json

# 2. convert to a vokra.utmos.* GGUF (v1 variant)
cargo run --release -p vokra-convert -- --model utmos \
    --input /tmp/utmos.safetensors --config /tmp/utmos-config.json --output /tmp/utmos.gguf

# 3. dump the upstream reference — this IMPORTS the real implementation
uv run --project tools/parity --frozen --python 3.12 python tools/parity/utmos_dump_reference.py \
    --ckpt "$CKPT" --w2v "$W2V" --clip tests/parity/utmos/ref-clip.wav \
    --outdir ~/.cache/vokra-eval/out/utmos-flip/reference

# 4. run both harnesses
VOKRA_UTMOS_GGUF=/tmp/utmos.gguf \
VOKRA_UTMOS_REFDIR=~/.cache/vokra-eval/out/utmos-flip/reference \
    cargo test --release -p vokra-eval --test parity_utmos_stages -- --nocapture
```

The conversion and model Cargo commands above are VAST-only. The former
Python 3.9/fairseq environment is historical and is not a reason to bypass the
current `weights_only=True` boundary.

## The honesty rules this directory enforces

- **The reference must import upstream.** `utmos_dump_reference.py` fetches the
  `sarulab-speech/UTMOS-demo` sources at a pinned, sha256-verified revision and
  lets *them* build the network (the real `fairseq` `Wav2Vec2Model` at
  `d03f4e77`). If the import fails it aborts loudly. Writing a local
  re-implementation to produce the "reference" is banned: a mirror agrees with
  the port by construction, so parity goes green while the audio is wrong. That
  is exactly what happened to Kokoro (fixed in `92dbc92`, round-trip WER
  1.0 → 0.0).
- **Synthesized weights are refused on the parity path** (`parity_utmos.rs`).
- **`atol` is derived, not chosen.** Each stage's bound is the *measured*
  worst-case delta × 2, and the measurements are tabulated in the `STAGE_ATOL`
  rustdoc in `parity_utmos_stages.rs` and in `docs/adr/M5-15-utmos.md` §(e).
  Never widen one to chase a green — localize the stage instead. If an
  architectural bound genuinely forces a wider value, record the derivation
  (Kokoro `PROSODY_F0_ATOL` precedent).
- **ISA caveat.** The bounds are calibrated on arm64/NEON. Kokoro showed that a
  different CPU class can shift a parity delta *deterministically* (AVX2
  4.34e-2 vs AVX-512 1.58e-2 on the same tensor). An x86 excursion is an ISA
  re-derivation, not automatically a regression — measure and add a second
  calibrated row rather than widening one bound to cover both.

## Historical measured result (2026-07-20, M1 iMac / arm64)

These values are retained for provenance and are not a current rerunnable
parity claim. The current status is `BLOCKED_UNSAFE_PICKLE`.

Every stage and the final score agreed with upstream:

| stage | max \|Δ\| |
|---|---|
| `conv_out` | 1.378e-7 |
| `feature_ln` | 2.384e-6 |
| `feat_proj` | 7.391e-6 |
| `pos_conv` (this stage also validates the offline weight-norm fold) | 1.621e-5 |
| `enc_in_ln` | 3.759e-6 |
| `enc_block_last` | 1.311e-6 |
| `blstm_out` | 4.172e-7 |
| `head_out` | 7.153e-7 |
| **score** | **1.192e-7** |

Cross-clip check (6 clips spanning MOS 1.27 … 4.50, native vs upstream): all
within 9.3e-7. See `docs/adr/M5-15-utmos.md` §(f).
