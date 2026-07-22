# UTMOS parity fixtures (M5-15 — flipped; the reference is upstream-generated)

**Status change vs M4-18.** This directory used to say "no fixture is
committed, deliberately" because the UTMOS weights were owner-gated and writing
an `expected_score` without running upstream would have fabricated the very
number the gate exists to verify. The 2026-07-18 owner un-defer resolved that:
the checkpoint is anonymously obtainable and permissively licensed (campaign-2
`utmos-probe`), so M5-15 generated the reference **by importing the real
upstream implementation**.

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

## Regenerating the reference (the whole recipe)

```bash
# 0. environment — measured, not assumed (M5-15 T38; docs/adr/M5-15-utmos.md §(d)).
#    Python 3.9 + torch 2.8.0 + fairseq @ d03f4e77 + pytorch-lightning 1.9.5 + omegaconf 2.1.2.
#    Python 3.11 does NOT work (fairseq@2022 trips 3.11's tightened dataclass check);
#    the upstream pin torch==1.11.0 has no macOS-arm64 wheel at all.
tools/parity/utmos_env_probe.sh          # records which branch this machine lands on

# 1. flatten the upstream .ckpt → safetensors + config side-car
~/.cache/vokra-eval/venv-utmos-e/bin/python tools/parity/utmos_prepare_checkpoint.py \
    --ckpt "$CKPT" --output /tmp/utmos.safetensors --config-out /tmp/utmos-config.json

# 2. convert to a vokra.utmos.* GGUF (v1 variant)
cargo run --release -p vokra-convert -- --model utmos \
    --input /tmp/utmos.safetensors --config /tmp/utmos-config.json --output /tmp/utmos.gguf

# 3. dump the upstream reference — this IMPORTS the real implementation
~/.cache/vokra-eval/venv-utmos-e/bin/python tools/parity/utmos_dump_reference.py \
    --ckpt "$CKPT" --w2v "$W2V" --clip tests/parity/utmos/ref-clip.wav \
    --outdir ~/.cache/vokra-eval/out/utmos-flip/reference

# 4. run both harnesses
VOKRA_UTMOS_GGUF=/tmp/utmos.gguf \
VOKRA_UTMOS_REFDIR=~/.cache/vokra-eval/out/utmos-flip/reference \
    cargo test --release -p vokra-eval --test parity_utmos_stages -- --nocapture
```

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

## Measured result (2026-07-20, M1 iMac / arm64)

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
