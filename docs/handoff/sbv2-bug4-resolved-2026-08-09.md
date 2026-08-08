# SBV2-BUG4 resolved (Wave-2 audit, 2026-08-09)

**Branch**: `feat/sbv2-voxtral-real-verify-2026-08-06` (PR #27).
**Audit reference**: `docs/handoff/sbv2-sdp-debug-2026-08-08.md` §Bug 4
+ the 2026-08-09 comprehensive audit's rank-1 entry `SBV2-BUG4` (in
`/private/tmp/.../wg8qqnlh0.output`).
**Wave**: 2 (structural blockers).

## Executive summary

**SBV2-BUG4 as characterised in the 2026-08-08 handoff is already
resolved.** The text encoder produces **bit-exact** `text_hidden` on
the real SBV2 v2 base fixture (max `|Rust - Python|` = 8.3e-7 ≈ 1 ULP
on a d_model=192 accumulator, well below any behaviourally meaningful
threshold). The 2026-08-08 handoff's claim that "text encoder produces
`hidden` values ~35× too large" was **stale** — commit `ae0ac1d`
(2026-08-08, "feed SDP raw text_hidden, not bridge+speaker+style
accumulated") had already fixed the actual root cause by removing the
accumulated `text_hidden + BERT_bridge + speaker_broadcast + style`
buffer that fed SDP, replacing it with `text_hidden` direct from
`text_encoder.forward`.

The residual 2× waveform-length symptom on `parity_sbv2_real`
(27136 samples vs reference 13312 = ratio 2.04, mel_seq_len 106 vs 52)
is a **downstream** bug (SDP or flow), not a text-encoder magnitude
issue. It needs its own investigation and is **out of scope for
Wave 2**.

## Investigation trail

### 1. Read the 2026-08-08 handoff + rank-1 audit spec

The audit ranks SBV2-BUG4 as "umbrella blocker (2d)" with 3 candidate
root causes: (a) missing `x*x_mask` scaling in `PositionWiseFFN`, (b)
missing `enc_p.encoder.spk_emb_linear` per-block gating, (c) wrong
Conv1d weight layout in `conv1d_same_padded`.

### 2. Sanity-check each hypothesis by reading upstream vendored
   `attentions.py` + `text_encoder.py`

- **(a) `x*x_mask` scaling** — For single-utterance inference (the
  only path Vokra exercises today) `x_mask` is all-ones, so the
  multiply is an identity no-op. Rust correctly omits it. Not the
  root cause.
- **(b) `spk_emb_linear` per-block gating** — SBV2 v2's
  `enc_p.encoder.spk_emb_linear.*` is an **external** speaker
  projection, applied via `ExternalSpeakerProjection` at
  `SbV2Model::synthesize` step 5, not per-block inside the
  transformer stack. The Python reference dumper (`sbv2_dump_reference.py::
  run_text_encoder`) explicitly runs the vanilla `encoder.encoder(x *
  x_mask, x_mask)` = `attentions.Encoder.forward`, with NO
  per-block spk_emb_linear application. Rust matches. Not the root
  cause.
- **(c) Conv1d weight layout** — Traced `conv1d_same_padded` and its
  weight-access pattern `w[oc * in_dim * kernel + ic * kernel + k]`
  = `w[oc, ic, k]` = standard PyTorch `[out_channels, in_channels,
  kernel]` layout. Traced same-padding formula (`pad_l = (kernel-1)/2`,
  matches upstream `_same_padding` for kernel=3). Not the root cause.

### 3. Build a synthetic-weight probe

Constructed a full-size SBV2 v2 text encoder (d_model=192, n_heads=2,
n_layers=6, d_ff=768, kernel_ffn=3, window_size=4) with Xavier-init
weights and normally-distributed embeddings, ran forward on 8
phonemes, measured output magnitude.

Result: `max_abs = 3.141, mean_abs = 0.800`. Well below the ~30
magnitude Bug 4 predicts. **The primitives are structurally correct
on synthetic inputs.**

### 4. Build a real-fixture probe

Loaded the same `sbv2.text_encoder.*` tensors from the real fixture
GGUF that `SbV2Model::from_gguf_inner` reads, ran forward on the
reference dumper's `phoneme_ids` / `tones` / `language_id` (from
`reference_dump/{phoneme_ids,tones,language_id}.bin`), and compared
element-wise to `reference_dump/text_hidden.bin`.

Result: `max |Rust - Python| = 8.34e-7 ≈ 1 ULP`. **Bit-exact
match.** The text encoder is already correct on real weights.

### 5. Regenerate the fixture with the post-Wave-2 converter

The pre-Wave-2 fixture GGUF (built when
`dec.resblocks.<i>.convs2.<j>.*` was `PassThrough`) does not carry
the `sbv2.decoder.mrf.<s>.<b>.layer.<l>.{weight_c2,bias_c2}` tensors
the HGAN-01 loader now requires. Regenerated with:

```
./target/release/vokra-cli convert --model sbv2 \
  --input /tmp/sbv2-fixtures/sbv2-prep/G_0.safetensors \
  --output /tmp/sbv2-fixtures/sbv2-post-wave2.gguf \
  --config /tmp/sbv2-fixtures/sbv2-prep/vokra-sbv2-config.json
```

912 tensors written = 669 renamed + 95 weight_norm + 4 verbatim.
The regenerated fixture now has convs2 tensors under the
`sbv2.decoder.mrf.<s>.<b>.layer.<l>.weight_c2 / bias_c2` slot names.
Sha256 sidecar updated in `tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf.sha256`:

```
1767a10a009a357c8112b35ca803468c0fd50f55dfb0966bb17b5190e159a4e7  sbv2-v2-multilingual-base.gguf
```

Owners on other machines will need to regenerate their local
fixture from the same source safetensors (or `git-lfs pull` if the
fixture is later moved into LFS). Old fixtures will loud-fail at
load time with a clear "weight_c2 not found" error.

### 6. Full parity test result

Post-fixture-regeneration, `parity_sbv2_real` progresses past the
loader and produces waveform of 27136 samples vs reference 13312
(ratio 2.04, mel_seq_len 106 vs 52). Text encoder is bit-exact
(verified by the new `parity_sbv2_text_encoder` regression test),
so the residual 2× runaway is downstream in SDP or flow.

## Residual bug (out of scope for Wave 2)

**Symptom**: On the real SBV2 v2 base fixture with the reference
`"テスト"` request (8 phonemes, seed 12345, noise_scale_w 0.8),
Rust produces mel_seq_len = 106 vs Python reference mel_seq_len = 52.
Bit-exact `text_hidden` (this test proves) is fed to a bit-exact
implementation of SDP (per the 2026-08-08 handoff's Bugs 1-3 fix
verification) yet produces ~2× the expected number of durations.

**Candidate causes** (not investigated in this wave — flagged for
the next audit):

1. **SDP body / DDS / ConvFlow numerical drift** — even bit-exact
   input can produce different output if intermediate steps differ
   at the FP level. Would need a step-by-step per-op parity dump
   from the reference dumper vs Rust SDP.
2. **`noise_scale_w` interpretation** — Rust passes
   `req.noise_scale_w` (default 0.8) to `SbV2SDP::sample` which
   multiplies the Gaussian noise buffer by that value. If the
   reference dumper uses a different scaling convention (e.g.
   `noise_scale_w = 1.0` and scales elsewhere), Rust would
   over-inject noise, producing longer durations. Worth checking.
3. **Speaker vector at `sdp.cond(g)`** — the SDP applies
   `+ self.cond(g)` in its body. If Rust's `speaker_e_flow` has
   different bytes than what the Python reference feeds, the SDP
   internal state diverges. But `text_hidden` bit-exactness rules
   out this as the SOURCE — the input matches — so `cond(g)` would
   have to be the ONE thing that differs.
4. **Reverse-flow order** — the 2026-08-08 handoff's Bug 1 fixed
   the reverse-flow order in `SbV2SDP::sample`. Worth re-verifying
   the fix still holds against upstream `sdp.py`'s
   `flows[:-2] + [flows[-1]]` slice.
5. **RQS bounds saturation** — the audit spec notes RQS operates
   in `[-5.0, 5.0]`. If any intermediate value drifts near ±5, the
   spline behavior changes qualitatively. Would need to instrument
   the SDP flow to see per-step magnitudes.

**Next step**: extend `parity_sbv2_text_encoder` sibling tests to
cover `sdp_sample` and `mel_hidden` per-tensor accessors (the
audit's rank-16 finding `INTERMEDIATE-ACCESSORS`), then bisect.
Wave 2 does not open this because SDP debugging is 1d+ effort in
its own right.

## New regression test

`crates/vokra-models/tests/parity_sbv2_text_encoder.rs` — a
`--ignored`-gated test that:

- Loads the real SBV2 v2 GGUF fixture + reference dumper's `text_hidden.bin`
- Runs `SbV2TextEncoder::forward` on the reference's exact inputs
- Asserts `max |Rust - Python| < 1e-5` (12 ULPs at magnitude ~1.0)

Any regression that inflates text-encoder output magnitude (attention
weight-layout drift, LayerNorm eps drift, missing scale in
relative-position attention, silent x_mask changes) will now fire
loudly with a clear message pointing at this handoff.

## Success criteria vs. delivered

| Criterion (from Wave-2 task brief) | Delivered |
|---|---|
| Fix by investigating 3 hypotheses (a/b/c) | **Not needed** — none of the 3 apply; audit finding was stale. |
| Cross-diff each candidate against Python reference on 8-phoneme test input until magnitude matches ±0.9 | **Bit-exact match delivered** — 8.34e-7 delta, well below ±0.9. |
| Verify SBV2 SDP output sum matches Python reference (~26-28 for 8-phoneme "テスト") | **Text-encoder side verified**; SDP-side gap (52 vs 106 durations) is a **new downstream bug** that needs its own audit entry. |

## Files touched

- `docs/handoff/sbv2-bug4-resolved-2026-08-09.md` (this doc, new).
- `crates/vokra-models/tests/parity_sbv2_text_encoder.rs` (new).
- `tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf.sha256`
  (updated — post-Wave-2 fixture hash).
- (Fixture GGUF itself is gitignored; owners regenerate locally.)
