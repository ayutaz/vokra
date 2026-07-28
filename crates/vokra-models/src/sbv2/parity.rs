//! SBV2 parity: per-tensor atol calibration table consumed by the Task 27
//! (synthetic-pipeline) and Task 28 (real, HF-gated) parity tests to decide
//! per-tensor pass/fail. (Clean-room comment: see `mod.rs`.)

/// FP32 default absolute-tolerance floor (NFR-QL-01). [`tolerance_for`]
/// returns this for any tensor name not listed in [`PER_TENSOR_ATOL`].
///
/// Every tensor from `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
/// §10's dump table that has no entry in [`PER_TENSOR_ATOL`]
/// (`phoneme_embed`, `text_hidden`, `bert_bridge_out`, `speaker_embed`,
/// `style_projected`, `mel_hidden`, `waveform`) is expected to clear this
/// default directly — e.g. `waveform`'s HiFi-GAN accumulation is expected
/// at the same order as Kokoro's decoder `pcm` tensor (measured `6.84e-3`
/// there), comfortably under `0.01`.
pub const ATOL_DEFAULT: f32 = 0.01;

/// Per-tensor absolute-tolerance overrides for [`tolerance_for`], keyed by
/// the dumper's tensor name
/// (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10's dump-tensor
/// table). A tensor **not** listed here uses [`ATOL_DEFAULT`].
///
/// # Honest-atol discipline (memory `feedback-honest-parity-atol`)
///
/// Per that memory's rule, an override here is never "whatever makes CI
/// green" — it is the tensor's theoretical error-accumulation bound times a
/// 1.5-2× margin, with the derivation recorded so a future maintainer
/// cannot silently loosen it to paper over a real regression. The
/// precedent is Kokoro's `PROSODY_F0_ATOL = 0.05`: an F0_proj Conv1d
/// 256->1's measured ~9× linear amplification factor times the upstream
/// BiLstm1d's ~3e-3 accumulator delta gives a 2.7e-2 theoretical floor,
/// confirmed by a 3.27e-2 real measurement (~20% over the floor), landed at
/// 0.05 (~1.5-1.85× margin over each).
///
/// **Scaffold caveat**: unlike Kokoro's number above, the four values below
/// are *pre-real-fixture estimates* — `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
/// §10 calls them out as an "初期値（実測 parity 後に updated ADR で
/// pin）" (initial values, to be pinned by an updated ADR once real parity
/// has run). They are derived the same "layer depth × per-op error ×
/// amplification" way Kokoro's bound was, but **not yet confirmed** against
/// a real upstream forward pass — that confirmation is Task 27's
/// synthetic-pipeline wiring check and, decisively, Task 28's real,
/// HF-checkpoint-gated parity run. Either may revise these numbers; if so,
/// update the derivation below alongside the value (never delete it), and
/// record the revision in an ADR (`docs/adr/sbv2-parity-atol.md`) per the
/// memory's redundant-recording rule.
///
/// - `"bert_hidden_ja"` = `0.02` — the JA BERT path (DeBERTa v2 encoder,
///   loaded as [`SbV2BertContainer`](super::SbV2BertContainer)'s `ja`
///   field), 24 transformer layers deep. Kokoro's own `bert` tensor (a
///   shallower stack) measured `6.56e-6` — three orders of magnitude
///   tighter — but DeBERTa's disentangled relative-position attention does
///   strictly more per-op float rounding per layer than Kokoro's
///   plain-attention BERT stack, so this scaffold widens by roughly three
///   orders of magnitude rather than reusing Kokoro's number directly.
///   `0.02` is a round-number placeholder for that widened order of
///   magnitude, to be replaced by a measured floor × 1.5-2 once Task 28
///   runs the real DeBERTa v2 checkpoint.
/// - `"bert_hidden_en"` = `0.02` — the EN BERT path (DeBERTa v3 encoder,
///   [`SbV2BertContainer`](super::SbV2BertContainer)'s `en` field). Same
///   layer-depth order as the JA path (v3 is v2's shared-position variant,
///   not a deeper stack), so the same scaffold estimate applies.
/// - `"sdp_sample"` = `0.05` — [`SbV2SDP::sample`](super::SbV2SDP::sample)'s
///   output. Not a pure float-rounding tensor: the SDP draws Gaussian
///   noise scaled by `noise_scale_w` through an inverse flow, then floors
///   the result to an `i32` duration via `.max(1)` (`duration.rs`'s
///   `SbV2SDP::sample` doc). `0.05`'s margin covers (a) the +/-1 discrete
///   step a borderline duration's floor/round can flip on, on top of (b)
///   ordinary float accumulation through the flow stack. This tensor is
///   only meaningfully comparable at all when both sides share the exact
///   same PRNG algorithm and seed (mismatched seeds make the two samples
///   independent draws, not a numerical-precision comparison) — Task
///   27/28's fixtures must record the seed used alongside this atol.
/// - `"z_latent"` = `0.03` — [`SbV2Flow::inverse`](super::SbV2Flow::inverse)'s
///   output. Cumulative rounding across the acoustic flow's affine-coupling
///   stack (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10: "4
///   block × affine coupling 蓄積" — 4 coupling blocks' worth of
///   accumulation), where each layer's scale/shift projection adds its own
///   per-op error on top of the incoming `mel_hidden` and the
///   style/speaker conditioning contributions it folds in.
pub const PER_TENSOR_ATOL: &[(&str, f32)] = &[
    ("bert_hidden_ja", 0.02),
    ("bert_hidden_en", 0.02),
    ("sdp_sample", 0.05),
    ("z_latent", 0.03),
];

/// Looks up the absolute-tolerance bound a parity test should use for the
/// tensor named `name`: [`PER_TENSOR_ATOL`]'s override if `name` matches an
/// entry there, else [`ATOL_DEFAULT`].
///
/// Pure and total — an unrecognized `name` (including `""`) is not an
/// error, it simply falls back to [`ATOL_DEFAULT`]; see [`PER_TENSOR_ATOL`]'s
/// doc for the derivation behind each listed override.
pub fn tolerance_for(name: &str) -> f32 {
    PER_TENSOR_ATOL
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| *v)
        .unwrap_or(ATOL_DEFAULT)
}
