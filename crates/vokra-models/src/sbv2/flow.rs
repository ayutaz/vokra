//! SBV2 normalizing flow (VITS2 latent flow): inverse-transforms a
//! Gaussian-prior latent `z` into the vocoder-facing acoustic latent,
//! conditioned on per-utterance style + speaker embeddings.
//! (Clean-room comment: see `mod.rs` — the affine-coupling structure below
//! follows the generic two-block affine-coupling normalizing-flow
//! construction of RealNVP (arXiv:1605.08803, Dinh, Sohl-Dickstein & Bengio
//! 2016) and the VITS2 paper's latent-flow role (arXiv:2307.16430, matching
//! `mod.rs`'s top-level reference list's bare citation for this paper); no
//! SBV2/BV2 source referenced.)
//!
//! `AffineCouplingLayer` reuse decision (Task 21): **(B) new local
//! [`SbV2AffineCouplingLayer`]**, not a `piper_plus` reuse. `piper_plus`'s
//! flow types (`crates/vokra-models/src/piper_plus/flow.rs`) are doubly
//! unreachable from `sbv2`: the per-layer coupling type, `Coupling`
//! (`flow.rs:106`), has **no visibility modifier at all** (private to
//! `piper_plus::flow`), and its owner, `Flow` (`flow.rs:137`), is only
//! `pub(super)` (visible in `piper_plus`, the parent module of
//! `piper_plus::flow` — not in the sibling top-level module `sbv2`). Even
//! set aside visibility, both `Coupling::reverse` (`flow.rs:116`) and
//! `Flow::reverse` (`flow.rs:218`) take a `&Compute` handle and operate on
//! a channel-major `[hidden, T]` whole-utterance buffer driven by a
//! WaveNet-style gated dilated-conv conditioner (`Wn`, `flow.rs:25`) with
//! `n_layers` learned convolutions per coupling — architecturally heavier
//! than this task needs, and `Compute`-tangled in a way `sbv2` (which
//! stays `Compute`-free throughout, like every prior `sbv2` module) does
//! not want to depend on. This is the identical shape of reasoning Task 17
//! applied to `TransformerBlock` and Task 19 applied to `ConvFlow`/
//! `Coupling` — see `text_encoder.rs`'s and `duration.rs`'s module docs
//! for the parallel writeups.
//!
//! # Flow structure
//!
//! [`SbV2Flow::inverse`] holds a stack of [`SbV2AffineCouplingLayer`]s,
//! applied **in order** (not reversed — the layers themselves already
//! implement the inference/reverse direction, matching
//! [`SbV2SDP`](super::duration::SbV2SDP)'s and `piper_plus::flow::Flow`'s
//! convention of only ever implementing a `reverse`/inverse pass, never a
//! forward). Each `[mel_seq_len, d_z]` row is split into two `[mel_seq_len,
//! half_d_z]` channel halves, `z_a` (first `half_d_z` channels) and `z_b`
//! (last `half_d_z` channels), `half_d_z = d_z / 2`. Per layer:
//!
//! 1. `z_a` is the **conditioning** half — read, never written.
//! 2. `z_b` is the **transformed** half:
//!    `z_b[p, d] ← (z_b[p, d] − shift(cond_a)[d]) · exp(−log_scale(cond_a)[d])`,
//!    where `cond_a[p, d] = z_a[p, d] + style_delta[d] + speaker_delta[d]`
//!    (the style/speaker deltas are per-utterance projections, computed
//!    once per layer and broadcast over every position `p` — see
//!    [`SbV2AffineCouplingLayer`]'s `inverse` doc for the exact formula).
//! 3. `z_a` and `z_b` are **swapped** before the next layer — the
//!    standard VITS/RealNVP-style alternating-coupling trick, so every
//!    layer gets a turn conditioning on (and transforming) each half.
//!
//! After the last layer, the current `(z_a, z_b)` pair is concatenated
//! back into one `[mel_seq_len, d_z]` row-major buffer (in whichever slot
//! order the final swap left them). An empty `coupling_layers` stack is a
//! legitimate, exercised no-op configuration: the split-then-immediately
//! -merge round trip (zero layers run in between, so zero swaps happen
//! too) is a bit-exact identity, since it performs no arithmetic — only
//! slice copies (see this crate's `tests/sbv2_flow.rs`), matching
//! [`SbV2SDP`](super::duration::SbV2SDP)'s and
//! [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s empty-stack
//! precedent.
//!
//! # Layout convention
//!
//! `z` and its two split halves are flat, row-major, **position-major**
//! (matching [`text_encoder.rs`](super::text_encoder)'s and
//! [`style.rs`](super::style)'s convention, not `piper_plus`'s
//! channel-major `[channels, time]` layout): a `[rows, cols]` buffer
//! addresses row `r` as `buf[r * cols .. (r + 1) * cols]`.

/// A single VITS2-style affine-coupling flow layer — see the module doc's
/// `AffineCouplingLayer` reuse decision for why this is a fresh, minimal
/// type rather than a `piper_plus` reuse, and the "Flow structure" section
/// for how [`SbV2Flow::inverse`] chains a stack of these with a
/// halves-swap between layers.
///
/// Unlike [`SbV2CouplingLayer`](super::duration::SbV2CouplingLayer) (a
/// **scalar** coupling between one duration latent and its conditioning
/// vector), this coupling splits a whole `[mel_seq_len, d_z]`
/// acoustic-latent buffer into two equal-width channel halves — the
/// classic two-block affine-coupling construction (RealNVP,
/// arXiv:1605.08803): one half (`z_a`) conditions the transform of the
/// other (`z_b`); `z_a` itself passes through this layer unchanged.
pub struct SbV2AffineCouplingLayer {
    /// Row-major `[half_d_z, half_d_z]`: projects the (style/speaker
    /// -conditioned) `z_a` half to the per-channel log-scale (`log_scale`
    /// in [`inverse`](Self::inverse)'s doc), bias-free.
    scale_weight: Vec<f32>,
    /// Row-major `[half_d_z, half_d_z]`: projects the conditioned `z_a`
    /// half to the per-channel shift, bias-free.
    shift_weight: Vec<f32>,
    /// Row-major `[half_d_z, d_style]`: projects the per-utterance
    /// `style_vec` to an additive delta over `z_a`'s `half_d_z` channels
    /// (broadcast identically to every `mel_seq_len` position), bias-free
    /// — same "linear map, no bias" convention as
    /// [`StyleVectorInjector`](super::style::StyleVectorInjector)'s
    /// projections.
    style_proj: Vec<f32>,
    /// Row-major `[half_d_z, d_speaker]`: as `style_proj`, but projects
    /// the per-utterance `speaker_embed`.
    speaker_proj: Vec<f32>,
    /// Half the flow's latent channel width (`d_z / 2`) — the row width
    /// of `z_a`/`z_b` and the output row count of both projections.
    half_d_z: usize,
    /// Style-vector input dimensionality (`style_proj.len() == half_d_z *
    /// d_style`, and every [`inverse`](Self::inverse) call's
    /// `style_vec.len()` must equal this).
    d_style: usize,
    /// Speaker-embedding input dimensionality (`speaker_proj.len() ==
    /// half_d_z * d_speaker`, and every [`inverse`](Self::inverse) call's
    /// `speaker_embed.len()` must equal this).
    d_speaker: usize,
}

impl SbV2AffineCouplingLayer {
    /// Builds a coupling layer from pre-trained projection weights.
    /// Crate-internal: no caller constructs a non-empty `coupling_layers`
    /// stack yet — the future `converter` module (see `mod.rs`'s roadmap
    /// comment) loads real GGUF weights and will call this (mirrors
    /// [`SbV2CouplingLayer::new`](super::duration::SbV2CouplingLayer)'s and
    /// [`SbV2TransformerBlock::new`](super::text_encoder::SbV2TransformerBlock)'s
    /// identical precedent).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, hot inner-loop constructor per this
    /// crate's established convention — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs) if `scale_weight.len() != half_d_z * half_d_z`,
    /// `shift_weight.len() != half_d_z * half_d_z`, `style_proj.len() !=
    /// half_d_z * d_style`, or `speaker_proj.len() != half_d_z *
    /// d_speaker`.
    #[allow(dead_code)] // constructed by the future converter once real GGUF-loaded flow weights are wired
    pub(crate) fn new(
        scale_weight: Vec<f32>,
        shift_weight: Vec<f32>,
        style_proj: Vec<f32>,
        speaker_proj: Vec<f32>,
        half_d_z: usize,
        d_style: usize,
        d_speaker: usize,
    ) -> Self {
        debug_assert_eq!(
            scale_weight.len(),
            half_d_z * half_d_z,
            "scale_weight must be [half_d_z, half_d_z]"
        );
        debug_assert_eq!(
            shift_weight.len(),
            half_d_z * half_d_z,
            "shift_weight must be [half_d_z, half_d_z]"
        );
        debug_assert_eq!(
            style_proj.len(),
            half_d_z * d_style,
            "style_proj must be [half_d_z, d_style]"
        );
        debug_assert_eq!(
            speaker_proj.len(),
            half_d_z * d_speaker,
            "speaker_proj must be [half_d_z, d_speaker]"
        );
        Self {
            scale_weight,
            shift_weight,
            style_proj,
            speaker_proj,
            half_d_z,
            d_style,
            d_speaker,
        }
    }

    /// Inverse affine-coupling transform, applied in place to `z_b`
    /// (`z_a` is the read-only conditioning half — see the type doc). Both
    /// `z_a` and `z_b` are `[mel_seq_len, half_d_z]` row-major.
    ///
    /// Per position `p`: `cond_a[d] = z_a[p, d] + style_delta[d] +
    /// speaker_delta[d]`, where `style_delta = style_proj · style_vec` and
    /// `speaker_delta = speaker_proj · speaker_embed` are per-utterance
    /// projections computed once and broadcast over every position
    /// (mirrors
    /// [`StyleVectorInjector::inject`](super::style::StyleVectorInjector::inject)'s
    /// identical project-once-broadcast pattern). Then `log_scale =
    /// scale_weight · cond_a`, `shift = shift_weight · cond_a` (each a
    /// bias-free `[half_d_z, half_d_z]` linear map), and finally `z_b[p,
    /// d] = (z_b[p, d] − shift[d]) · exp(−log_scale[d])` — the exact
    /// inverse of the canonical affine-coupling forward `y = x ·
    /// exp(log_scale) + shift`.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `z_a.len()` or `z_b.len()` differ
    /// from `mel_seq_len * self.half_d_z`, if `style_vec.len() !=
    /// self.d_style`, or if `speaker_embed.len() != self.d_speaker`.
    fn inverse(
        &self,
        z_a: &[f32],
        z_b: &mut [f32],
        mel_seq_len: usize,
        style_vec: &[f32],
        speaker_embed: &[f32],
    ) {
        let half = self.half_d_z;
        debug_assert_eq!(
            z_a.len(),
            mel_seq_len * half,
            "z_a must be [mel_seq_len, half_d_z]"
        );
        debug_assert_eq!(
            z_b.len(),
            mel_seq_len * half,
            "z_b must be [mel_seq_len, half_d_z]"
        );
        debug_assert_eq!(style_vec.len(), self.d_style, "style_vec must be [d_style]");
        debug_assert_eq!(
            speaker_embed.len(),
            self.d_speaker,
            "speaker_embed must be [d_speaker]"
        );

        // Per-utterance conditioning delta, shared across every position
        // (computed once here, not per-position below).
        let mut cond_delta = vec![0.0_f32; half];
        for (d, delta) in cond_delta.iter_mut().enumerate() {
            let style_row = &self.style_proj[d * self.d_style..(d + 1) * self.d_style];
            let speaker_row = &self.speaker_proj[d * self.d_speaker..(d + 1) * self.d_speaker];
            let s: f32 = style_row.iter().zip(style_vec).map(|(w, x)| w * x).sum();
            let p: f32 = speaker_row
                .iter()
                .zip(speaker_embed)
                .map(|(w, x)| w * x)
                .sum();
            *delta = s + p;
        }

        let mut cond_a = vec![0.0_f32; half];
        let mut log_scale = vec![0.0_f32; half];
        let mut shift = vec![0.0_f32; half];
        for (za_row, zb_row) in z_a.chunks_exact(half).zip(z_b.chunks_exact_mut(half)) {
            for ((c, &za), &delta) in cond_a.iter_mut().zip(za_row).zip(cond_delta.iter()) {
                *c = za + delta;
            }
            for (d2, (sc, sh)) in log_scale.iter_mut().zip(shift.iter_mut()).enumerate() {
                let srow = &self.scale_weight[d2 * half..(d2 + 1) * half];
                let trow = &self.shift_weight[d2 * half..(d2 + 1) * half];
                *sc = srow.iter().zip(&cond_a).map(|(w, x)| w * x).sum();
                *sh = trow.iter().zip(&cond_a).map(|(w, x)| w * x).sum();
            }
            for ((zb, &sh), &sc) in zb_row.iter_mut().zip(shift.iter()).zip(log_scale.iter()) {
                *zb = (*zb - sh) * (-sc).exp();
            }
        }
    }
}

/// SBV2's VITS2 normalizing flow: inverse-transforms a Gaussian-prior
/// latent `z` into the vocoder-facing acoustic latent by walking a stack
/// of [`SbV2AffineCouplingLayer`]s, conditioned on per-utterance style and
/// speaker embeddings. See the module doc's "Flow structure" section for
/// the exact per-layer transform and the halves-swap between layers.
pub struct SbV2Flow {
    /// Affine-coupling stack, applied **in order** (see the module doc's
    /// "Flow structure" section). An empty stack is a legitimate,
    /// exercised no-op configuration — see [`inverse`](Self::inverse)'s
    /// doc and this crate's `tests/sbv2_flow.rs`.
    coupling_layers: Vec<SbV2AffineCouplingLayer>,
    /// Latent channel dimension (`z.len() == mel_seq_len * d_z` in every
    /// [`inverse`](Self::inverse) call). Must be even — see
    /// [`from_layers`](Self::from_layers)'s panic docs.
    d_z: usize,
}

impl SbV2Flow {
    /// Builds a flow from a pre-trained affine-coupling stack. The default
    /// VITS2 flow depth is 4 layers, but this constructor takes whatever
    /// `coupling_layers` the caller provides — it is the source of truth
    /// for the stack's actual depth (Task 24-27's converter reads the real
    /// depth from the checkpoint's config and constructs accordingly).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs for why this crate uses `debug_assert!` rather than
    /// `Result` for constructor shape checks) if `d_z == 0` or `d_z` is
    /// odd — VITS2 affine coupling splits `z` into two equal-width
    /// channel halves (see the module doc's "Flow structure" section), so
    /// `d_z` must be even, and `d_z == 0` would make `half_d_z == 0`, which
    /// [`inverse`](Self::inverse)'s internal `chunks_exact(half_d_z)` calls
    /// cannot accept (mirrors
    /// [`BertBridge::forward`](super::text_encoder::BertBridge::forward)'s
    /// `bert_seq_len == 0` guard, added in Task 17's review for the
    /// identical class of zero-width-dimension bug).
    pub fn from_layers(coupling_layers: Vec<SbV2AffineCouplingLayer>, d_z: usize) -> Self {
        debug_assert!(d_z > 0, "d_z must be non-zero");
        debug_assert_eq!(
            d_z % 2,
            0,
            "d_z must be even (VITS2 affine coupling splits into two equal halves)"
        );
        Self {
            coupling_layers,
            d_z,
        }
    }

    /// Inverse VITS2 normalizing flow: transforms a Gaussian-prior latent
    /// `z` (`[mel_seq_len, d_z]` row-major) into the vocoder-facing
    /// acoustic latent, conditioned on `style_vec` and `speaker_embed`.
    /// Returns a `[mel_seq_len, d_z]` row-major buffer of the same shape
    /// as `z`. See the module doc's "Flow structure" section for the full
    /// per-layer algorithm, including the empty-stack identity-pass case.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `z.len() != mel_seq_len *
    /// self.d_z`. `style_vec`/`speaker_embed` are validated per-layer
    /// (each [`SbV2AffineCouplingLayer`] carries its own `d_style`/
    /// `d_speaker`, not `SbV2Flow` itself), so with an empty
    /// `coupling_layers` stack they are never read or checked at all.
    pub fn inverse(
        &self,
        z: &[f32],
        mel_seq_len: usize,
        style_vec: &[f32],
        speaker_embed: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(
            z.len(),
            mel_seq_len * self.d_z,
            "z must be [mel_seq_len, d_z]"
        );

        let half = self.d_z / 2;
        let (mut z_a, mut z_b) = split_halves(z, self.d_z, half);
        for layer in &self.coupling_layers {
            layer.inverse(&z_a, &mut z_b, mel_seq_len, style_vec, speaker_embed);
            std::mem::swap(&mut z_a, &mut z_b);
        }
        merge_halves(&z_a, &z_b, half)
    }
}

/// Splits `z` (`[mel_seq_len, d_z]` row-major, `d_z = 2 * half`) into two
/// `[mel_seq_len, half]` row-major halves: `z_a` holds each row's first
/// `half` channels, `z_b` holds each row's last `half` channels. Exactly
/// undone by [`merge_halves`] (pure slice copies, no arithmetic — so
/// `merge_halves(split_halves(z, ..))` reproduces `z` bit-for-bit).
fn split_halves(z: &[f32], d_z: usize, half: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = z.len() / d_z;
    let mut z_a = Vec::with_capacity(rows * half);
    let mut z_b = Vec::with_capacity(rows * half);
    for row in z.chunks_exact(d_z) {
        z_a.extend_from_slice(&row[..half]);
        z_b.extend_from_slice(&row[half..]);
    }
    (z_a, z_b)
}

/// Inverse of [`split_halves`]: interleaves two `[mel_seq_len, half]`
/// row-major halves back into one `[mel_seq_len, 2 * half]` row-major
/// buffer (row `p`'s output is `z_a`'s row `p` followed by `z_b`'s row
/// `p`).
fn merge_halves(z_a: &[f32], z_b: &[f32], half: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(z_a.len() + z_b.len());
    for (a_row, b_row) in z_a.chunks_exact(half).zip(z_b.chunks_exact(half)) {
        out.extend_from_slice(a_row);
        out.extend_from_slice(b_row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weight_coupling_layer_inverse_is_identity() {
        let half_d_z = 3;
        let d_style = 2;
        let d_speaker = 2;
        let layer = SbV2AffineCouplingLayer::new(
            vec![0.0; half_d_z * half_d_z],
            vec![0.0; half_d_z * half_d_z],
            vec![0.0; half_d_z * d_style],
            vec![0.0; half_d_z * d_speaker],
            half_d_z,
            d_style,
            d_speaker,
        );
        let mel_seq_len = 2;
        let z_a = vec![0.3, -1.2, 5.0, 0.1, 2.2, -0.4]; // [mel_seq_len, half_d_z]
        let mut z_b = vec![-2.5_f32, 0.0, 1.0, 7.25, 3.3, -6.6];
        let expected = z_b.clone();
        let style_vec = [0.9, -0.4];
        let speaker_embed = [1.1, -2.2];

        layer.inverse(&z_a, &mut z_b, mel_seq_len, &style_vec, &speaker_embed);

        assert_eq!(
            z_b, expected,
            "zero-weight coupling layer must be the identity on z_b"
        );
    }

    #[test]
    fn coupling_layer_inverse_matches_hand_computed_affine() {
        // half_d_z = 2, d_style = 1, d_speaker = 1, mel_seq_len = 1.
        // scale_weight = identity [[1,0],[0,1]], shift_weight = zero.
        // style_proj = [[1],[0]] (style only affects cond channel 0),
        // speaker_proj = [[0],[1]] (speaker only affects cond channel 1).
        // z_a = [0.5, 0.5], style_vec = [0.0], speaker_embed = [0.0]
        //   => cond_a = z_a + 0 = [0.5, 0.5]
        //   => log_scale = identity * cond_a = [0.5, 0.5], shift = [0, 0]
        // z_b = [2.0, 3.0]
        //   => z_b' = (z_b - 0) * exp(-[0.5, 0.5])
        //           = [2.0 * exp(-0.5), 3.0 * exp(-0.5)]
        let layer = SbV2AffineCouplingLayer::new(
            vec![1.0, 0.0, 0.0, 1.0], // scale_weight: identity
            vec![0.0, 0.0, 0.0, 0.0], // shift_weight: zero
            vec![1.0, 0.0],           // style_proj: [[1],[0]]
            vec![0.0, 1.0],           // speaker_proj: [[0],[1]]
            2,
            1,
            1,
        );
        let z_a = [0.5_f32, 0.5];
        let mut z_b = [2.0_f32, 3.0];
        layer.inverse(&z_a, &mut z_b, 1, &[0.0], &[0.0]);

        let expected0 = 2.0_f32 * (-0.5_f32).exp();
        let expected1 = 3.0_f32 * (-0.5_f32).exp();
        assert!(
            (z_b[0] - expected0).abs() < 1e-6,
            "got {}, expected {expected0}",
            z_b[0]
        );
        assert!(
            (z_b[1] - expected1).abs() < 1e-6,
            "got {}, expected {expected1}",
            z_b[1]
        );
    }

    #[test]
    fn coupling_layer_inverse_reads_style_and_speaker_conditioning() {
        // Same shapes as the previous test, but with nonzero style_vec /
        // speaker_embed, exercising the additive conditioning delta
        // (`cond_a = z_a + style_delta + speaker_delta`) rather than just
        // the z_a contribution.
        let layer = SbV2AffineCouplingLayer::new(
            vec![1.0, 0.0, 0.0, 1.0], // scale_weight: identity
            vec![0.0, 0.0, 0.0, 0.0], // shift_weight: zero
            vec![1.0, 0.0],           // style_proj: [[1],[0]]
            vec![0.0, 1.0],           // speaker_proj: [[0],[1]]
            2,
            1,
            1,
        );
        // style_vec = [0.2] -> style_delta = [0.2, 0.0]
        // speaker_embed = [0.3] -> speaker_delta = [0.0, 0.3]
        // z_a = [0.0, 0.0] -> cond_a = [0.2, 0.3]
        // log_scale = cond_a = [0.2, 0.3], shift = [0, 0]
        let z_a = [0.0_f32, 0.0];
        let mut z_b = [1.0_f32, 1.0];
        layer.inverse(&z_a, &mut z_b, 1, &[0.2], &[0.3]);

        let expected0 = 1.0_f32 * (-0.2_f32).exp();
        let expected1 = 1.0_f32 * (-0.3_f32).exp();
        assert!(
            (z_b[0] - expected0).abs() < 1e-6,
            "got {}, expected {expected0}",
            z_b[0]
        );
        assert!(
            (z_b[1] - expected1).abs() < 1e-6,
            "got {}, expected {expected1}",
            z_b[1]
        );
    }

    #[test]
    fn flow_inverse_single_identity_layer_swaps_halves() {
        // A single zero-weight (identity-on-z_b) coupling layer still
        // performs the mandatory end-of-layer halves-swap (see the module
        // doc's "Flow structure" section), so the output must be z_a and
        // z_b *swapped* relative to the input, not z unchanged — this
        // exercises SbV2Flow::inverse's split/loop/swap/merge machinery
        // (the external `tests/sbv2_flow.rs` integration tests can only
        // build an *empty* stack, since SbV2AffineCouplingLayer::new is
        // `pub(crate)` — see this module's doc for why).
        let half_d_z = 2;
        let d_z = 4;
        let layer = SbV2AffineCouplingLayer::new(
            vec![0.0; half_d_z * half_d_z],
            vec![0.0; half_d_z * half_d_z],
            vec![0.0; half_d_z],
            vec![0.0; half_d_z],
            half_d_z,
            1,
            1,
        );
        let flow = SbV2Flow::from_layers(vec![layer], d_z);
        let z = vec![1.0_f32, 2.0, 3.0, 4.0]; // one row: z_a=[1,2], z_b=[3,4]
        let out = flow.inverse(&z, 1, &[0.0], &[0.0]);

        assert_eq!(
            out,
            vec![3.0, 4.0, 1.0, 2.0],
            "single identity layer must still swap halves once"
        );
    }

    #[test]
    fn flow_inverse_two_identity_layers_returns_to_original_order() {
        // Two zero-weight layers: each is an identity on the half it
        // transforms, and two halves-swaps cancel out, so the output must
        // equal the input exactly — a *different* code path than the
        // empty-stack identity (this one actually runs the loop body
        // twice), giving non-redundant coverage of the swap-cancels-out
        // invariant.
        let half_d_z = 2;
        let d_z = 4;
        let make_layer = || {
            SbV2AffineCouplingLayer::new(
                vec![0.0; half_d_z * half_d_z],
                vec![0.0; half_d_z * half_d_z],
                vec![0.0; half_d_z],
                vec![0.0; half_d_z],
                half_d_z,
                1,
                1,
            )
        };
        let flow = SbV2Flow::from_layers(vec![make_layer(), make_layer()], d_z);
        let z = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 rows
        let out = flow.inverse(&z, 2, &[0.0], &[0.0]);

        assert_eq!(out, z, "two identity layers must cancel their halves-swaps");
    }
}
