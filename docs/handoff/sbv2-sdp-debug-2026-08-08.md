# SBV2 v2 SDP debug — handoff, 2026-08-08

**Branch**: `feat/sbv2-voxtral-real-verify-2026-08-06` (PR #27).
**Symptom** (pre-fix, CI): `[sbv2-synth-warn] SbV2SDP produced runaway
durations (max=26539, sum=107980) — clamped to per-phoneme ceiling 500.`
Reference Python dumper produces `sum=26, max=8` (sane).

## Executive summary

Three SBV2 v2 SDP bugs found; two fixed at the source, one out of
this task's file scope. The remaining bug is upstream of the SDP
(text encoder produces `hidden` values ~35× too large) and needs a
separate follow-up in `crates/vokra-models/src/sbv2/text_encoder.rs`
(or wherever the true scale mismatch lives).

## Root causes + fixes

### Bug 1 (SDP flow order — FIXED)

Rust `SbV2SDP::sample` walked `self.flows` in **forward** order
(`flows[0], flows[1], flows[2], flows[3]`). Upstream
`StochasticDurationPredictor.forward(reverse=True)` walks in
**reversed** order and drops the "useless vflow" via `flows[:-2] +
[flows[-1]]`:

```python
# upstream tools/parity/vendor/vits/sdp.py lines 131-136:
else:
  flows = list(reversed(self.flows))
  flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
  z = torch.randn(x.size(0), 2, x.size(2)) * noise_scale
  for flow in flows:
    z = flow(z, x_mask, g=x, reverse=True)
```

Upstream `self.flows` is `[EA, CF, Flip, CF, Flip, CF, Flip, CF,
Flip]` (9 items for `n_flows=4`). Reversed then `[:-2] + [-1]` gives
`[Flip, CF(orig 7), Flip, CF(orig 5), Flip, CF(orig 3), Flip,
EA(orig 0)]` — 4 Flips + 3 CFs + 1 EA. The dropped item is `CF(orig
1)` = Rust `self.flows[0]` after the converter's dense re-index
(upstream `sdp.flows.1` → dense index 0).

**Fix**: `crates/vokra-models/src/sbv2/duration.rs::SbV2SDP::sample`
now iterates `self.flows[1..].iter().rev()` (via `split_first`),
producing `Flip, flows[3], Flip, flows[2], Flip, flows[1], Flip, EA`.
Matches upstream exactly. Doc updated on the `flows` field and the
`sample` module docstring.

### Bug 2 (n_sdp_layers mis-declared — FIXED)

Config side-car (`tools/parity/sbv2_prepare_checkpoint.py` line 354)
defaults `n_sdp_layers = 3`, sourced from VITS
`StochasticDurationPredictor` `n_layers_dp=3` (the DDS **inner
depth**). But Rust's `n_sdp_layers` metadata slot is used as
`n_flows` (**ConvFlow count**), which for real SBV2 v2 is 4. The
mismatch caused Rust to load only 3 of the 4 ConvFlows — the fourth
(`sdp.flows.7.*` = dense index 3) was never bound, so the loader
constructed a 3-CF SDP where upstream has 4.

**Fix**: `crates/vokra-convert/src/models/sbv2.rs::convert_sbv2_file`
now scans the input safetensors for `sdp.flows.<odd>.pre.weight`
tensors and overrides `cfg.n_sdp_layers` with the observed count.
Pattern matches the existing shape-recovery in
`.github/workflows/parity-sbv2-real.yml` for `d_speaker`,
`n_speakers`, `decoder_upsample_kernel_sizes`. Emits a stderr
warning line so the override is auditable per FR-EX-08.

### Bug 3 (Python reference SDP mis-implementation — FIXED)

`tools/parity/sbv2_dump_reference.py::SDPReference.sample` had two
cut corners:

1. It skipped `+ self.cond(g)` in the body (the `sdp.cond.*` weights
   were LOADED via `__init__` but never applied). Upstream
   unconditionally applies them when `g` is provided.
2. It walked ALL 9 flows in `reversed(self._m.flows)`, including
   the "useless vflow" upstream drops via `flows[:-2] + [flows[-1]]`.

**Fix**: `SDPReference.sample` now mirrors upstream verbatim: apply
`.cond(g)` iff `_m.cond` exists; use the `flows[:-2] + [flows[-1]]`
slice. Body computation also matches upstream (`pre(x)` with no
`* x_mask`, `+ cond(g)` before DDS, `proj * x_mask` after).

## Bug 4 (out of scope — REMAINING)

**Text encoder produces `hidden` values ~35× too large.** After
Bugs 1-3 are fixed, Rust's SDP still emits runaway durations
(max≈12036 down from 26539, sum≈24229 down from 107980) because
the input `hidden` fed to `SbV2SDP::sample` has magnitudes in
±33.3, vs. the Python reference `text_hidden.bin` which has
magnitudes in ±0.9. When SDP receives `hidden ≈ ±33`, the body's
DDS+proj chain amplifies to ±215; RQS spline softmax params
saturate (near-one-hot); the spline degenerates to a step function
that maps all inputs to the same tail corner.

**Proof the SDP itself is correct**: I temporarily instrumented
`SbV2SDP::sample` with a `VOKRA_SBV2_SDP_HIDDEN_OVERRIDE` env var
that replaces `hidden` with the bytes from Python's
`reference_dump/text_hidden.bin` (±0.9 magnitudes). Rerunning the
parity test with the override:

```
[sbv2-synth-trace] SDP durations n=8 min=1 max=10 sum=28
```

vs Python reference `sum=26, max=8` — essentially matching (small
delta is expected from the RNG divergence between Rust's
`GaussianSplitMix64` and Python's `torch.randn`, plus per-tensor
accumulation FP order). The SDP is now numerically consistent with
upstream when fed correct input.

**Files that likely need investigation** (not modified by this
task):

- `crates/vokra-models/src/sbv2/text_encoder.rs` — check the
  `SbV2TransformerBlock::forward` path, `PositionWiseFFN`, and
  `LayerNorm` for either (a) a missing scaling factor (upstream
  `attentions.py::Encoder` applies `x * x_mask` before + after the
  block stack — Rust omits this since single-utterance mask is all
  ones, but the FFN's internal `x * x_mask` before conv_1 may
  matter under any padding), (b) a missing `spk_emb_linear`
  contribution (the base ckpt carries `enc_p.encoder.spk_emb_linear.*`
  = SBV2 v2's per-block speaker gating, neither Rust nor Python
  vendored VITS applies it — but the converter DOES rename the
  tensors), or (c) a wrong weight layout (weight might be loaded
  transposed vs what `conv1d_same_padded` expects).
- `crates/vokra-models/src/sbv2/mod.rs::synthesize` — the step 4/5
  additions (bert bridge + speaker + style) may double-apply
  scaling. Python passes only `text_hidden` to SDP, without adding
  bridge/speaker — worth checking whether upstream SBV2 v2 (`models_jp_extra.py`)
  concatenates them into `x` before SDP or keeps them separate.

## Stopgap left in place

`crates/vokra-models/src/sbv2/mod.rs::synthesize` still contains the
`PER_PHONEME_DURATION_CEILING = 500` OOM stopgap (commit `d62353b`,
2026-08-06). Left in place per task instructions ("Do NOT touch the
OOM stopgap cap in `mod.rs` — leave it in place as safety"). The
stopgap message text still refers to the SDP as a "scalar-affine
simplification" — this is now stale (SbV2SDP is the real DDS-net +
ConvFlow SDP post-Blocker-2c) but was outside the file scope of
this task.

## Verification (local, M1 iMac)

- `cargo test -p vokra-models --lib sbv2::duration::` → **11/11
  passed** (SDP unit tests + primitive tests).
- `cargo clippy -p vokra-models -p vokra-convert -- -D warnings` →
  clean.
- `./scripts/check-zero-deps.sh` → OK (root `Cargo.lock` unchanged,
  vokra-* only).
- Regenerated `tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf`
  locally with the converter fix — GGUF now stamps
  `vokra.sbv2.n_sdp_layers = 4` (was 3).
- Rerun parity test with the regenerated fixture: SDP output
  `max=12036, sum=24229` (down from `max=26539, sum=107980`), still
  runaway due to Bug 4.

## Success criteria vs. delivered

| Criterion (from task brief) | Delivered |
|---|---|
| No longer emit `[sbv2-synth-warn] runaway durations` | **No** — still fires (Bug 4 unresolved) |
| Produce a sane `sum=` value (< 200 frames for the 8-phoneme "テスト" test) | **No** — sum=24229 (down from 107980, but still runaway) |
| Either pass parity OR fail on a much smaller numeric delta | Partial: 4-6× smaller runaway; would fully pass if `hidden` were correct (proven via `VOKRA_SBV2_SDP_HIDDEN_OVERRIDE` experiment above) |

The SDP itself is verified correct against upstream (Bugs 1-3
fixed, override experiment). Full CI green requires resolving Bug 4
outside this task's file scope.
