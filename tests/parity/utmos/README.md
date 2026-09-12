# UTMOS parity fixtures (M5-15)

## Current status (2026-09-12)

The committed fixture and native UTMOS harness remain useful historical
artifacts. The canonical preparation input is now a fixed, authenticated
tensor-only `.safetensors` state-dict: it contains no pickle program or Python
objects. `tools/parity/utmos_prepare_checkpoint.py --state-dict ...` accepts
that path and derives the side-car only after strict tensor/key/shape checks.

The historical SaruLab Lightning checkpoint is permanently refused. It is a
pickle container with training objects, and no class allowlist or
`weights_only=True` exception is permitted for this route: `--ckpt` returns
`BLOCKED_UNSAFE_PICKLE`. A fixed, authenticated tensor-only state-dict is
accepted only for the safe preparation milestone. The current
`tools/parity/utmos_dump_reference.py` inspects that safe input but returns
`BLOCKED_SAFE_REFERENCE` until an independently authenticated wav2vec source
and owner-approved safe upstream model-construction path are available.

Therefore current CI claims only the model-free self-tests and the
safe state-dict boundary; it claims no numeric UTMOS parity. Re-enabling real
reference generation requires owner-approved safe state-dict wiring for both
the UTMOS and wav2vec inputs, separate from the MIT license sign-off in
`docs/license-audit.md` §3.1.

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

**Still not committed, deliberately:** the checkpoint itself. The legacy
pickle stays unusable; only an owner-provided, fixed `.safetensors` export may
enter the preparation route. Vokra ships no UTMOS weights here.

## The two harnesses

| test | what it checks | env |
|---|---|---|
| `crates/vokra-eval/tests/parity_utmos.rs` | final score vs `score.json` | `VOKRA_UTMOS_GGUF` |
| `crates/vokra-eval/tests/parity_utmos_stages.rs` | **every stage** vs the upstream hook points | `VOKRA_UTMOS_GGUF` + `VOKRA_UTMOS_REFDIR` |

The stage harness is the load-bearing one: a single scalar cannot localize a
fault (a swapped `ln1`/`ln2` mapping, a mis-folded weight-norm and a backwards
LSTM direction all just read as "wrong number"), so the per-stage comparison is
what turns a failure into a named stage.

## Safe regeneration recipe (currently blocked)

The following recipe documents the former VAST procedure. Do not run it until
the owner supplies both fixed state-dict URLs and SHA-256 manifests. The
legacy Lightning pickle must not be bypassed or opened.

```bash
# 0. The legacy Lightning checkpoint is not an input. The VAST worker must
#    first receive both an authenticated tensor-only state-dict and its SHA.

# 1. flatten an authenticated tensor-only state-dict → config side-car
uv run --project tools/parity/utmos --frozen --python 3.12 python tools/parity/utmos_prepare_checkpoint.py \
    --state-dict "$STATE_DICT" --output /tmp/utmos.safetensors --config-out /tmp/utmos-config.json

# 2. convert to a vokra.utmos.* GGUF (v1 variant)
cargo run --release -p vokra-convert -- --model utmos \
    --input /tmp/utmos.safetensors --config /tmp/utmos-config.json --output /tmp/utmos.gguf

# 3. dump the upstream reference — currently blocked until both safe inputs
#    and the owner-approved upstream construction path are supplied
uv run --project tools/parity/utmos --frozen --python 3.12 python tools/parity/utmos_dump_reference.py \
    --state-dict "$STATE_DICT" --w2v "$W2V_STATE_DICT" --clip tests/parity/utmos/ref-clip.wav \
    --outdir ~/.cache/vokra-eval/out/utmos-flip/reference

# 4. run both harnesses
VOKRA_UTMOS_GGUF=/tmp/utmos.gguf \
VOKRA_UTMOS_REFDIR=~/.cache/vokra-eval/out/utmos-flip/reference \
    cargo test --release -p vokra-eval --test parity_utmos_stages -- --nocapture
```

The conversion and model Cargo commands above are VAST-only. No legacy
Lightning pickle route may be used to obtain a state-dict.

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
parity claim. The legacy `--ckpt` status is `BLOCKED_UNSAFE_PICKLE`; the safe
state-dict reference path remains `BLOCKED_SAFE_REFERENCE`.

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
