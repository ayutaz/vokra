//! SBV2 VITS2 normalizing flow ([Blocker 2b, 2026-08-06]).
//!
//! Clean-room Apache-2.0 implementation of VITS2's `TransformerCouplingBlock`
//! (`p0p4k/vits2_pytorch`, MIT). Applied in the inverse direction only, as
//! required by the SBV2 inference path (`z ~ N(0, I) * noise_scale` → flow
//! inverse → acoustic latent that feeds the HiFi-GAN decoder). NOT
//! referenced: `github.com/litagin02/Style-Bert-VITS2` (AGPL-3.0),
//! `github.com/fishaudio/Bert-VITS2` (AGPL-3.0) — only permissive VITS /
//! VITS2 sources were read.
//!
//! # Structure
//!
//! Upstream `TransformerCouplingBlock.__init__` at `n_flows = 4` stores the
//! flow stack as a flat `nn.ModuleList` of length `2 * n_flows = 8`,
//! alternating [`SbV2TransformerCouplingLayer`] (even indices `0, 2, 4, 6`)
//! and [`Flip`] parameter-free modules (odd indices `1, 3, 5, 7`) — the
//! coupling reads one channel-half as conditioning and transforms the
//! other, then `Flip` reverses the channel axis so the next coupling
//! layer operates on a different orientation.
//!
//! [`FlowLayer`] models this heterogeneous stack as a two-arm enum, and
//! [`SbV2Flow::inverse`] iterates `layers.iter().rev()` — exactly matching
//! upstream `TransformerCouplingBlock.forward(x, x_mask, g=g,
//! reverse=True)`'s `for flow in reversed(self.flows): x = flow(x, ...
//! reverse=True)` loop. Both [`FlowLayer`] arms carry their own
//! inference-direction primitive so the outer loop is a straight `match`
//! per layer, with no per-layer scratch storage on the outside.
//!
//! # Layout convention
//!
//! Every buffer in this module is flat, row-major, **position-major**:
//! a `[rows, cols]` buffer addresses row `r` as
//! `buf[r * cols .. (r + 1) * cols]`. This matches
//! [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s
//! `[seq_len, d_model]` convention (not `piper_plus`'s channel-major
//! `[channels, time]` layout — see `text_encoder.rs`'s module doc for the
//! parallel writeup on layout choice).
//!
//! # Conditioning vector `g`
//!
//! [`SbV2TransformerCouplingLayer`]'s `inverse` takes an opaque
//! `[gin_channels]`-wide conditioning vector `g` — the caller composes
//! this from the per-utterance speaker embedding, style vector, and any
//! other conditioning signals per **Blocker 3**'s not-yet-landed
//! composition rule. From this module's point of view `g` is a black-box
//! `f32` slice broadcast identically to every position of the utterance
//! (matches upstream `TransformerCouplingLayer.forward`'s `h = h +
//! self.spk_emb_linear(g.mT).mT` — one projection per block, broadcast
//! over `T`).
//!
//! # Forward direction is out of scope
//!
//! This module implements the reverse (inference) direction only, the
//! same convention [`SbV2SDP`](super::duration::SbV2SDP)'s
//! `SbV2CouplingLayer` and `piper_plus::flow::Coupling` both follow — the
//! forward direction is a training-side operation this crate never runs.

use super::text_encoder::SbV2TransformerBlock;

// -----------------------------------------------------------------------
// Flow-layer enum
// -----------------------------------------------------------------------

/// One element of the VITS2 flow stack: either a full
/// [`SbV2TransformerCouplingLayer`] (parameterized coupling with an
/// embedded transformer stack producing per-channel mean statistics) or a
/// parameter-free [`Flip`] channel-reversal. Wrapping both arms in the
/// same enum lets [`SbV2Flow::inverse`] walk them in a single reverse
/// iteration and dispatch by a straight `match`, matching upstream
/// `p0p4k/vits2_pytorch/models.TransformerCouplingBlock`'s flat
/// `nn.ModuleList` layout.
pub enum FlowLayer {
    /// A parameterized transformer coupling — reads one channel-half as
    /// conditioning, transforms the other via a per-block transformer
    /// stack (see [`SbV2TransformerCouplingLayer`]'s doc for the exact
    /// arithmetic).
    Coupling(SbV2TransformerCouplingLayer),
    /// A parameter-free channel reversal (`torch.flip(x, [1])` in
    /// upstream). See [`Flip`]'s doc.
    Flip,
}

/// Marker type documenting the [`FlowLayer::Flip`] arm — `Flip` itself
/// carries no learned parameters (upstream `p0p4k/vits2_pytorch/modules.Flip`
/// stores nothing beyond the enum tag). Constructors that need to build a
/// flow with flip layers use [`FlowLayer::Flip`] directly; this type
/// exists only as documentation surface.
pub struct Flip;

// -----------------------------------------------------------------------
// SbV2TransformerCouplingLayer — the VITS2 parameterized coupling
// -----------------------------------------------------------------------

/// One VITS2 `TransformerCouplingLayer` (upstream `p0p4k/vits2_pytorch
/// /modules.TransformerCouplingLayer`, the SBV2 v2 base
/// `share_parameter=False`, `gin_channels=512`, `mean_only=True` variant).
///
/// # Structure (inference direction, `reverse=True`)
///
/// Given input `z` (`[mel_seq_len, d_z]` row-major, `d_z = 2 * half_d_z`)
/// and per-utterance conditioning `g` (`[gin_channels]`):
///
/// ```text
///     z_a, z_b = split_halves(z)                      # each [T, half_d_z]
///     h = pre(z_a)                                    # [T, d_hidden] (1×1 Conv1d + bias)
///     h += spk_emb_linear(g).broadcast(T)             # per-block additive conditioning
///     h = encoder_stack(h)                            # in-place SbV2TransformerBlock chain
///     m = post(h)                                     # [T, half_d_z] (1×1 Conv1d + bias, mean_only=True)
///     z_b -= m                                        # reverse-direction affine, logs = 0
///     out = merge_halves(z_a, z_b)
/// ```
///
/// Under `mean_only=false` (not exercised by the base checkpoint but
/// carried as a config field for future SKUs), `post` produces a
/// `[T, 2*half_d_z]` output split into `(m, logs)`, and the reverse
/// affine becomes `z_b = (z_b - m) * exp(-logs)`.
///
/// # Real base-checkpoint shapes (2026-08-06 scout)
///
/// Verified via `/tmp/sbv2-fixtures/sbv2-prep/G_0.safetensors` for
/// `litagin/Style-Bert-VITS2-2.0-base-JP-Extra`:
///
/// | Field              | Shape                          |
/// |--------------------|--------------------------------|
/// | `pre_weight`       | `[192, 96, 1]`                 |
/// | `pre_bias`         | `[192]`                        |
/// | `spk_emb_weight`   | `[192, 512]`                   |
/// | `spk_emb_bias`     | `[192]`                        |
/// | `post_weight`      | `[96, 192, 1]` (mean_only=True) |
/// | `post_bias`        | `[96]`                         |
/// | `encoder_stack`    | 6 layers × 114 tensors / 6 = 19 tensors/layer |
///
/// The `[192, 96, 1]` `pre_weight` is a 1×1 Conv1d — bytes-identical to a
/// `[192, 96]` linear weight since `kernel = 1`; this crate stores it as
/// such for consistency with [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s
/// row-major `[out_dim, in_dim]` convention.
///
/// # `encoder_stack` differs from the text encoder
///
/// The transformer blocks used inside each coupling have the same
/// [`SbV2TransformerBlock`] type as the text encoder, but two hparams
/// differ from the text encoder's defaults:
///
/// - `n_encoder_layers = 6` per coupling (independent of the text
///   encoder's `n_text_layers`).
/// - `kernel_ffn = 5` (upstream `filter_channels = 768`, `p_dropout = 0`,
///   `kernel_size = 5` on this coupling; the text encoder uses `3`).
///
/// The runtime carries both as GGUF metadata (`vokra.sbv2.flow.*` for the
/// flow side; the text encoder's own hparams live under `vokra.sbv2.*`
/// directly) so a future SKU with different values round-trips through
/// the converter without a code change.
pub struct SbV2TransformerCouplingLayer {
    /// `pre` 1×1 Conv1d weight, row-major `[d_hidden, half_d_z]`
    /// (upstream stores as `[d_hidden, half_d_z, 1]`; kernel=1 makes the
    /// bytes identical to a flat `[d_hidden, half_d_z]` linear weight).
    pre_weight: Vec<f32>,
    /// `pre` 1×1 Conv1d bias, `[d_hidden]`.
    pre_bias: Vec<f32>,
    /// `spk_emb_linear` projection weight, row-major `[d_hidden,
    /// gin_channels]` (upstream is a plain `nn.Linear`, not a conv — the
    /// output is a scalar `[d_hidden]` broadcast identically to every `T`
    /// position of `h`, not a per-position projection).
    spk_emb_weight: Vec<f32>,
    /// `spk_emb_linear` bias, `[d_hidden]`.
    spk_emb_bias: Vec<f32>,
    /// Inner transformer stack — 6 [`SbV2TransformerBlock`]s on the SBV2
    /// v2 base checkpoint. Passed in already-built by the caller
    /// (`SbV2Model::from_gguf`), which reads `n_encoder_layers` from the
    /// `vokra.sbv2.flow.n_encoder_layers` metadata key and constructs the
    /// stack layer-by-layer against the coupling's own hparams.
    encoder_stack: Vec<SbV2TransformerBlock>,
    /// `post` 1×1 Conv1d weight, row-major `[post_out_dim, d_hidden]`
    /// where `post_out_dim = half_d_z` under `mean_only=true` (SBV2 v2
    /// base) or `2 * half_d_z` under `mean_only=false` (future SKUs).
    post_weight: Vec<f32>,
    /// `post` 1×1 Conv1d bias, `[post_out_dim]`.
    post_bias: Vec<f32>,
    /// Half the flow's latent channel width (`d_z / 2`) — the row width
    /// of `z_a`/`z_b`.
    half_d_z: usize,
    /// Hidden width of the coupling's inner transformer stack (upstream
    /// `hidden_channels`, distinct from the text encoder's `d_model`
    /// though numerically both equal 192 on the SBV2 v2 base).
    d_hidden: usize,
    /// Per-utterance conditioning vector width (upstream `gin_channels`,
    /// = 512 on the SBV2 v2 base).
    gin_channels: usize,
    /// Whether `post` emits only the mean statistic (`half_d_z` output
    /// channels) or both mean and log-scale (`2 * half_d_z` output
    /// channels). `true` on the SBV2 v2 base — see
    /// [`from_weights`](Self::from_weights)'s doc for why we still carry
    /// this flag.
    mean_only: bool,
}

impl SbV2TransformerCouplingLayer {
    /// The `gin_channels` (per-utterance conditioning vector width) this
    /// layer's `spk_emb_linear` was trained against. Exposed so
    /// [`SbV2Flow::gin_channels`] can surface the flow-wide `g` shape
    /// contract without re-inspecting layer internals.
    pub fn gin_channels(&self) -> usize {
        self.gin_channels
    }

    /// Builds a coupling layer from pre-trained weights.
    ///
    /// # `mean_only`
    ///
    /// The SBV2 v2 base checkpoint sets `mean_only = true` universally
    /// (verified via `/tmp/sbv2-fixtures/sbv2-prep/G_0.safetensors`:
    /// `post.weight` has shape `[96, 192, 1]` — `half_d_z` output
    /// channels, not `2 * half_d_z`). The flag is nonetheless kept as a
    /// field so a future SBV2 SKU shipping `mean_only=false` weights
    /// round-trips through the loader without a code change; the
    /// arithmetic branch is documented in [`inverse`](Self::inverse).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs for why this crate uses `debug_assert!` rather than
    /// `Result` for constructor shape checks) if any weight/bias
    /// buffer's length disagrees with the shape documented on its field
    /// above, if `half_d_z == 0`, or if `d_hidden == 0`.
    #[allow(clippy::too_many_arguments)] // one arg per weight tensor, mirrors the struct's fields
    pub fn from_weights(
        pre_weight: Vec<f32>,
        pre_bias: Vec<f32>,
        spk_emb_weight: Vec<f32>,
        spk_emb_bias: Vec<f32>,
        encoder_stack: Vec<SbV2TransformerBlock>,
        post_weight: Vec<f32>,
        post_bias: Vec<f32>,
        half_d_z: usize,
        d_hidden: usize,
        gin_channels: usize,
        mean_only: bool,
    ) -> Self {
        debug_assert!(half_d_z > 0, "half_d_z must be positive");
        debug_assert!(d_hidden > 0, "d_hidden must be positive");
        debug_assert_eq!(
            pre_weight.len(),
            d_hidden * half_d_z,
            "pre_weight must be [d_hidden, half_d_z]"
        );
        debug_assert_eq!(pre_bias.len(), d_hidden, "pre_bias must be [d_hidden]");
        debug_assert_eq!(
            spk_emb_weight.len(),
            d_hidden * gin_channels,
            "spk_emb_weight must be [d_hidden, gin_channels]"
        );
        debug_assert_eq!(
            spk_emb_bias.len(),
            d_hidden,
            "spk_emb_bias must be [d_hidden]"
        );
        let post_out_dim = if mean_only { half_d_z } else { 2 * half_d_z };
        debug_assert_eq!(
            post_weight.len(),
            post_out_dim * d_hidden,
            "post_weight must be [post_out_dim, d_hidden] (post_out_dim = half_d_z if \
             mean_only, else 2*half_d_z)"
        );
        debug_assert_eq!(
            post_bias.len(),
            post_out_dim,
            "post_bias must be [post_out_dim]"
        );
        Self {
            pre_weight,
            pre_bias,
            spk_emb_weight,
            spk_emb_bias,
            encoder_stack,
            post_weight,
            post_bias,
            half_d_z,
            d_hidden,
            gin_channels,
            mean_only,
        }
    }

    /// Runs one inverse-direction pass on `(z_a, z_b)` in place on `z_b`
    /// (`z_a` is the read-only conditioning half — see the type doc for
    /// the full arithmetic). Both `z_a` and `z_b` are `[mel_seq_len,
    /// half_d_z]` row-major.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `z_a.len()` or `z_b.len()` differ
    /// from `mel_seq_len * self.half_d_z`, or if `g.len() !=
    /// self.gin_channels`.
    fn inverse(&self, z_a: &[f32], z_b: &mut [f32], mel_seq_len: usize, g: &[f32]) {
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
        debug_assert_eq!(g.len(), self.gin_channels, "g must be [gin_channels]");

        // 1. h = pre(z_a) — 1×1 Conv1d + bias, per position.
        //    shape: [T, d_hidden]
        let mut h = linear_rows_biased(z_a, half, &self.pre_weight, &self.pre_bias, self.d_hidden);

        // 2. h += spk_emb_linear(g), broadcast over every position.
        //    spk_emb(g) is a [d_hidden] scalar computed once per call
        //    (upstream `h = h + self.spk_emb_linear(g.mT).mT` — one
        //    projection per block, added identically to every `T`).
        //
        // FLOW-DEFENSE-BUNDLE guard (2026-08-09): when `gin_channels == 0`,
        // the coupling was trained WITHOUT speaker conditioning (empty
        // `spk_emb_weight`; upstream `nn.Linear(0, d_hidden)` collapses
        // to an add-bias-only op). `linear_g` would loop over an empty
        // `g` slice and still emit the bias, so the pre-fix behavior is
        // subtly correct-by-accident but confusing on inspection.
        // Explicit branch documents intent + short-circuits the empty
        // dot product.
        let spk = if self.gin_channels == 0 {
            debug_assert!(
                g.is_empty(),
                "SbV2TransformerCouplingLayer::inverse: gin_channels == 0 requires g == [] \
                 (caller passed g of length {})",
                g.len()
            );
            // With `gin_channels == 0`, upstream `nn.Linear(0, d_hidden)`
            // reduces to the bias vector: `linear_g` on empty weight and
            // empty g would return `spk_emb_bias.clone()` anyway. Short
            // circuit both to make the empty-conditioning contract explicit.
            self.spk_emb_bias.clone()
        } else {
            linear_g(&self.spk_emb_weight, &self.spk_emb_bias, g, self.d_hidden)
        };
        for row in h.chunks_exact_mut(self.d_hidden) {
            for (o, &s) in row.iter_mut().zip(spk.iter()) {
                *o += s;
            }
        }

        // 3. encoder_stack.forward(h) — in-place SbV2TransformerBlock chain.
        //    Each block reads/writes h under the same layout as the text
        //    encoder's own transformer stack (see text_encoder.rs).
        for block in &self.encoder_stack {
            block.forward(&mut h, mel_seq_len);
        }

        // 4. stats = post(h) — 1×1 Conv1d + bias, per position.
        //    Under mean_only=true, shape [T, half_d_z] is m directly.
        //    Under mean_only=false, shape [T, 2*half_d_z] is (m, logs)
        //    concatenated along channels.
        let post_out_dim = if self.mean_only { half } else { 2 * half };
        let stats = linear_rows_biased(
            &h,
            self.d_hidden,
            &self.post_weight,
            &self.post_bias,
            post_out_dim,
        );

        // 5. z_b_new = (z_b - m) * exp(-logs)
        if self.mean_only {
            // logs = 0 → exp(-0) = 1 → z_b -= m.
            for (b_row, m_row) in z_b.chunks_exact_mut(half).zip(stats.chunks_exact(half)) {
                for (b, &m) in b_row.iter_mut().zip(m_row.iter()) {
                    *b -= m;
                }
            }
        } else {
            // Split stats into (m, logs), each [T, half].
            for (b_row, stats_row) in z_b
                .chunks_exact_mut(half)
                .zip(stats.chunks_exact(post_out_dim))
            {
                let (m_row, logs_row) = stats_row.split_at(half);
                for ((b, &m), &l) in b_row.iter_mut().zip(m_row.iter()).zip(logs_row.iter()) {
                    // WP-11 (2026-08-10): flow-prior exp through vokra_math
                    // for cross-plat determinism within Vokra (SBV2 flow
                    // affine-coupling inverse, 4 blocks × d_z per sample).
                    *b = (*b - m) * vokra_math::exp(-l);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// SbV2Flow — the whole 8-slot stack (4 TCL + 4 Flip on the SBV2 v2 base)
// -----------------------------------------------------------------------

/// SBV2's VITS2 normalizing flow: inverse-transforms a Gaussian-prior
/// latent `z` into the vocoder-facing acoustic latent by walking a stack
/// of [`FlowLayer`]s, conditioned on a per-utterance opaque `g` vector
/// (see the module doc's "Conditioning vector `g`" section).
pub struct SbV2Flow {
    /// Flow layer stack — walked in **reverse** order by
    /// [`inverse`](Self::inverse) (matches upstream
    /// `p0p4k/vits2_pytorch/models.TransformerCouplingBlock.forward` at
    /// `reverse=True`).
    ///
    /// On the SBV2 v2 base checkpoint this is `[TCL0, Flip, TCL1, Flip,
    /// TCL2, Flip, TCL3, Flip]` — 8 entries alternating parameterized
    /// coupling and parameter-free channel-flip. An empty stack is a
    /// legitimate, exercised no-op configuration (see this crate's
    /// `tests/sbv2_flow.rs`), matching every other SBV2 module's
    /// empty-stack precedent.
    layers: Vec<FlowLayer>,
    /// Latent channel dimension (`z.len() == mel_seq_len * d_z` in every
    /// [`inverse`](Self::inverse) call). Must be even — see
    /// [`from_layers`](Self::from_layers)'s panic docs.
    d_z: usize,
}

impl SbV2Flow {
    /// Builds a flow from a pre-constructed layer stack. The caller
    /// (`SbV2Model::from_gguf` in production, tests directly otherwise)
    /// is the source of truth for the stack's actual composition.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `d_z == 0` or `d_z` is odd — VITS2
    /// affine coupling splits `z` into two equal-width channel halves
    /// (see the module doc), so `d_z` must be even and non-zero
    /// (`half_d_z == 0` would make `chunks_exact` calls in
    /// [`inverse`](Self::inverse) reject).
    pub fn from_layers(layers: Vec<FlowLayer>, d_z: usize) -> Self {
        debug_assert!(d_z > 0, "d_z must be non-zero");
        debug_assert_eq!(
            d_z % 2,
            0,
            "d_z must be even (VITS2 affine coupling splits into two equal halves)"
        );
        // FLOW-DEFENSE-BUNDLE cross-validation (2026-08-09): every
        // `FlowLayer::Coupling` must share the same `gin_channels`
        // value. A mismatched inner would silently truncate/pad `g`
        // per-layer and produce garbage. `SbV2Flow::gin_channels()`
        // returns the first Coupling's value, so a caller assuming
        // that value is the flow-wide contract would be lied to.
        let mut expected_gin: Option<usize> = None;
        for (idx, layer) in layers.iter().enumerate() {
            if let FlowLayer::Coupling(c) = layer {
                let gin = c.gin_channels();
                match expected_gin {
                    None => expected_gin = Some(gin),
                    Some(want) => debug_assert_eq!(
                        gin, want,
                        "SbV2Flow::from_layers: FlowLayer::Coupling at index {idx} has \
                         gin_channels {gin} but earlier Coupling layers used {want} — every \
                         Coupling in a well-formed flow must share the same per-utterance \
                         conditioning contract (`SbV2Flow::gin_channels()` returns the first \
                         Coupling's value; a mismatched later layer would be silently truncated \
                         / padded)"
                    ),
                }
            }
        }
        Self { layers, d_z }
    }

    /// The `gin_channels` value the flow's coupling layers were trained
    /// against. Returns `0` if the layer stack contains no parameterized
    /// coupling layer (e.g. empty stack or `Flip`-only) — the caller must
    /// treat `0` as "flow does not consume `g`" and skip conditioning.
    ///
    /// All parameterized coupling layers in a well-formed flow share the
    /// same `gin_channels` (they are all trained under the same
    /// per-utterance conditioning contract); this accessor returns the
    /// value from the first `Coupling` layer encountered.
    pub fn gin_channels(&self) -> usize {
        for layer in &self.layers {
            if let FlowLayer::Coupling(c) = layer {
                return c.gin_channels();
            }
        }
        0
    }

    /// Inverse VITS2 normalizing flow: transforms a Gaussian-prior latent
    /// `z` (`[mel_seq_len, d_z]` row-major) into the vocoder-facing
    /// acoustic latent, conditioned on `g` (`[gin_channels]`). Returns a
    /// `[mel_seq_len, d_z]` row-major buffer of the same shape as `z`.
    ///
    /// # Iteration order
    ///
    /// Iterates `layers.iter().rev()` — for the base
    /// `[TCL0, Flip, TCL1, Flip, TCL2, Flip, TCL3, Flip]` stack this
    /// walks `[Flip, TCL3, Flip, TCL2, Flip, TCL1, Flip, TCL0]`. Matches
    /// upstream `TransformerCouplingBlock.forward(x, x_mask, g=g,
    /// reverse=True)` at `for flow in reversed(self.flows): x = flow(x,
    /// ... reverse=True)`.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `z.len() != mel_seq_len *
    /// self.d_z`. `g`'s length is validated per coupling layer, so with
    /// an empty `layers` stack (or a stack of only [`Flip`]s) it is
    /// never read or checked.
    pub fn inverse(&self, z: &[f32], mel_seq_len: usize, g: &[f32]) -> Vec<f32> {
        debug_assert_eq!(
            z.len(),
            mel_seq_len * self.d_z,
            "z must be [mel_seq_len, d_z]"
        );

        let half = self.d_z / 2;
        let (mut z_a, mut z_b) = split_halves(z, self.d_z, half);
        for layer in self.layers.iter().rev() {
            match layer {
                FlowLayer::Flip => {
                    // Upstream `Flip.forward(x)` = `torch.flip(x, [1])`,
                    // which reverses the channel axis of `x` (shape [B,
                    // C, T]). In our row-major `[T, d_z]` layout with `z`
                    // already split into `z_a` = first half channels and
                    // `z_b` = second half channels, reversing the channel
                    // axis of the merged row `[z_a_row, z_b_row]` yields
                    // `[reverse(z_b_row), reverse(z_a_row)]`. We
                    // materialize this by (a) reversing each half's rows,
                    // then (b) swapping which half is `z_a` vs `z_b`.
                    reverse_rows_inplace(&mut z_a, half);
                    reverse_rows_inplace(&mut z_b, half);
                    std::mem::swap(&mut z_a, &mut z_b);
                }
                FlowLayer::Coupling(tcl) => {
                    tcl.inverse(&z_a, &mut z_b, mel_seq_len, g);
                }
            }
        }
        merge_halves(&z_a, &z_b, half)
    }
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

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

/// Reverses each `stride`-wide row of `buf` in place — used by
/// [`SbV2Flow::inverse`]'s [`FlowLayer::Flip`] arm to model upstream
/// `torch.flip(x, [1])` on the merged `[T, d_z]` buffer (see that arm's
/// inline comment for the split-halves equivalence).
fn reverse_rows_inplace(buf: &mut [f32], stride: usize) {
    for row in buf.chunks_exact_mut(stride) {
        row.reverse();
    }
}

/// Applies a bias-free `[out_dim, in_dim]` row-major linear map plus a
/// per-output-channel bias `b` (`[out_dim]`) to each `in_dim`-wide row of
/// `x`, producing a flat `[rows, out_dim]` buffer (`rows = x.len() /
/// in_dim`). Bytes-identical to `text_encoder`'s own
/// `conv1x1_biased` / `linear_rows_biased` — duplicated here so the flow
/// module keeps its own arithmetic tight and doesn't reach across a
/// visibility boundary for a helper that (a) is a five-line inner loop
/// and (b) has to preserve bit-identical arithmetic across both call
/// sites for parity across the M6 refactor.
fn linear_rows_biased(x: &[f32], in_dim: usize, w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let rows = x.len() / in_dim;
    let mut out = vec![0.0_f32; rows * out_dim];
    for (xi, oi) in x.chunks_exact(in_dim).zip(out.chunks_exact_mut(out_dim)) {
        for ((o, wrow), &bi) in oi.iter_mut().zip(w.chunks_exact(in_dim)).zip(b.iter()) {
            *o = wrow.iter().zip(xi).map(|(a, b)| a * b).sum::<f32>() + bi;
        }
    }
    out
}

/// Applies a bias `[out_dim, gin_channels]` linear projection to the
/// single `[gin_channels]` input `g`, returning a `[out_dim]` vector.
/// Distinct from [`linear_rows_biased`] because `g` has no `T` axis —
/// it's a per-utterance vector projected once per coupling call, then
/// broadcast identically to every `T` position of `h` by the caller
/// ([`SbV2TransformerCouplingLayer::inverse`]).
fn linear_g(w: &[f32], b: &[f32], g: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for (o_idx, (o, &bi)) in out.iter_mut().zip(b.iter()).enumerate() {
        let wrow = &w[o_idx * g.len()..(o_idx + 1) * g.len()];
        *o = wrow.iter().zip(g).map(|(a, b)| a * b).sum::<f32>() + bi;
    }
    out
}

// =====================================================================
// Internal tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /// Small zero-encoder-stack coupling layer builder — the workhorse
    /// helper for the tests below. `n_encoder_layers = 0` keeps parity
    /// with `SbV2TextEncoder`'s own empty-stack precedent, which every
    /// other SBV2 module documents as a legitimate, exercised no-op
    /// configuration.
    #[allow(clippy::too_many_arguments)] // one arg per weight tensor, mirrors the struct's fields
    fn make_tcl(
        half_d_z: usize,
        d_hidden: usize,
        gin_channels: usize,
        pre_weight: Vec<f32>,
        pre_bias: Vec<f32>,
        spk_emb_weight: Vec<f32>,
        spk_emb_bias: Vec<f32>,
        post_weight: Vec<f32>,
        post_bias: Vec<f32>,
        mean_only: bool,
    ) -> SbV2TransformerCouplingLayer {
        SbV2TransformerCouplingLayer::from_weights(
            pre_weight,
            pre_bias,
            spk_emb_weight,
            spk_emb_bias,
            Vec::new(),
            post_weight,
            post_bias,
            half_d_z,
            d_hidden,
            gin_channels,
            mean_only,
        )
    }

    // ---------------------------------------------------------------
    // Basic identity / arithmetic
    // ---------------------------------------------------------------

    /// A coupling layer with `post_weight = 0` and `post_bias = 0`
    /// produces `m = 0` under `mean_only = true`, so the reverse
    /// transform reduces to `z_b -= 0 = z_b`. Pins the identity path
    /// through pre → spk_emb → post → affine (empty encoder stack).
    #[test]
    fn zero_post_coupling_layer_inverse_is_identity_on_z_b() {
        let half_d_z = 3;
        let d_hidden = 4;
        let gin_channels = 2;
        let tcl = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.5; d_hidden * half_d_z],
            vec![0.1; d_hidden],
            vec![0.05; d_hidden * gin_channels],
            vec![0.0; d_hidden],
            vec![0.0; half_d_z * d_hidden],
            vec![0.0; half_d_z],
            true,
        );
        let mel_seq_len = 2;
        let z_a = vec![0.3_f32, -1.2, 5.0, 0.1, 2.2, -0.4];
        let mut z_b = vec![-2.5_f32, 0.0, 1.0, 7.25, 3.3, -6.6];
        let expected = z_b.clone();
        let g = vec![0.9_f32, -0.4];

        tcl.inverse(&z_a, &mut z_b, mel_seq_len, &g);

        assert_eq!(
            z_b, expected,
            "zero-post coupling layer must be identity on z_b under mean_only=true"
        );
    }

    /// A coupling layer with post = identity and zero pre → constant
    /// bias in h → constant m per position (broadcast identically over
    /// T). Verifies the arithmetic `z_b -= m` reduces to a constant
    /// per-channel subtraction we can compute by hand.
    #[test]
    fn coupling_layer_inverse_matches_hand_computed_broadcast() {
        // Setup: `half_d_z = 2, d_hidden = 2, gin_channels = 1`.
        // pre = 0 (so `pre(z_a) = pre_bias` for all positions).
        // pre_bias = [1.0, 2.0].
        // spk_emb = 0 (so h = pre(z_a) + 0 = [1.0, 2.0] per row).
        // encoder_stack empty → h unchanged.
        // post_weight = identity ([[1,0],[0,1]]), post_bias = 0.
        // → m = h * identity + 0 = [1.0, 2.0] per row.
        // z_b -= [1.0, 2.0] per row.
        let half_d_z = 2;
        let d_hidden = 2;
        let gin_channels = 1;
        let tcl = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.0; d_hidden * half_d_z],     // pre_weight
            vec![1.0, 2.0],                     // pre_bias — constant per row
            vec![0.0; d_hidden * gin_channels], // spk_emb_weight
            vec![0.0; d_hidden],                // spk_emb_bias
            vec![1.0, 0.0, 0.0, 1.0],           // post_weight: identity
            vec![0.0; half_d_z],                // post_bias
            true,
        );
        let mel_seq_len = 2;
        let z_a = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut z_b = vec![100.0_f32, 200.0, 300.0, 400.0];
        let g = vec![0.5];

        tcl.inverse(&z_a, &mut z_b, mel_seq_len, &g);

        // Expected: z_b - [1.0, 2.0] per row = [99.0, 198.0, 299.0, 398.0].
        assert!((z_b[0] - 99.0).abs() < 1e-5);
        assert!((z_b[1] - 198.0).abs() < 1e-5);
        assert!((z_b[2] - 299.0).abs() < 1e-5);
        assert!((z_b[3] - 398.0).abs() < 1e-5);
    }

    /// A coupling layer with `spk_emb_weight` nonzero and pre = 0
    /// exercises the g → spk_emb → h broadcast path independently of
    /// the z_a → pre → h path. Verifies that h picks up the projection
    /// of g on every position.
    #[test]
    fn coupling_layer_inverse_reads_g_conditioning() {
        // spk_emb_weight = [[1, 0], [0, 1]] (d_hidden=2, gin_channels=2).
        // g = [3.0, 7.0] → spk_emb(g) = [3.0, 7.0].
        // spk_emb_bias = [0.5, -0.5] → h_broadcast = [3.5, 6.5] per row.
        // post = identity → m = [3.5, 6.5] per row.
        // z_b_new = z_b - [3.5, 6.5].
        let half_d_z = 2;
        let d_hidden = 2;
        let gin_channels = 2;
        let tcl = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.0; d_hidden * half_d_z], // pre_weight
            vec![0.0; d_hidden],            // pre_bias
            vec![1.0, 0.0, 0.0, 1.0],       // spk_emb_weight: identity
            vec![0.5, -0.5],                // spk_emb_bias
            vec![1.0, 0.0, 0.0, 1.0],       // post_weight: identity
            vec![0.0; half_d_z],            // post_bias
            true,
        );
        let mel_seq_len = 1;
        let z_a = vec![0.0_f32, 0.0];
        let mut z_b = vec![10.0_f32, 100.0];
        let g = vec![3.0, 7.0];

        tcl.inverse(&z_a, &mut z_b, mel_seq_len, &g);

        assert!((z_b[0] - (10.0 - 3.5)).abs() < 1e-5, "z_b[0] = {}", z_b[0]);
        assert!((z_b[1] - (100.0 - 6.5)).abs() < 1e-5, "z_b[1] = {}", z_b[1]);
    }

    /// The `mean_only = false` branch: `post` produces `[T, 2*half_d_z]`
    /// output which is split into `(m, logs)`, and `z_b = (z_b - m) *
    /// exp(-logs)`. Verifies both the shape acceptance (post_weight is
    /// `2 * half_d_z * d_hidden` here, twice as wide as under
    /// mean_only) and the `exp(-logs)` scale factor.
    #[test]
    fn coupling_layer_inverse_mean_only_false_applies_exp_neg_logs() {
        // half_d_z = 2, d_hidden = 1, gin_channels = 1.
        // pre = 0, spk_emb = 0 (so h = 0 per row).
        // encoder_stack empty → h stays 0.
        // post_bias picks the (m, logs) directly since h=0.
        //   post_bias[0..2] = m = [1.0, 2.0]
        //   post_bias[2..4] = logs = [0.0, ln(2)]
        //   → exp(-logs) = [1.0, 0.5]
        // z_b_new = (z_b - m) * exp(-logs)
        //         = ([10, 20] - [1, 2]) * [1, 0.5]
        //         = [9, 18] * [1, 0.5]
        //         = [9, 9]
        let half_d_z = 2;
        let d_hidden = 1;
        let gin_channels = 1;
        let post_out_dim = 2 * half_d_z; // 4
        let tcl = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.0; d_hidden * half_d_z],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * gin_channels],
            vec![0.0; d_hidden],
            vec![0.0; post_out_dim * d_hidden],  // post_weight = 0
            vec![1.0, 2.0, 0.0, (2.0_f32).ln()], // post_bias picks (m, logs)
            false,
        );
        let mel_seq_len = 1;
        let z_a = vec![0.0_f32; half_d_z];
        let mut z_b = vec![10.0_f32, 20.0];
        let g = vec![0.0];

        tcl.inverse(&z_a, &mut z_b, mel_seq_len, &g);

        assert!((z_b[0] - 9.0).abs() < 1e-5, "z_b[0] = {}", z_b[0]);
        assert!((z_b[1] - 9.0).abs() < 1e-5, "z_b[1] = {}", z_b[1]);
    }

    // ---------------------------------------------------------------
    // Flip semantics — helper direct + Flow arm
    // ---------------------------------------------------------------

    /// The `reverse_rows_inplace` helper is the primitive
    /// `FlowLayer::Flip` uses to model upstream `torch.flip(x, [1])`.
    /// Pins the elementwise semantics.
    #[test]
    fn reverse_rows_inplace_reverses_each_stride() {
        let mut buf = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        reverse_rows_inplace(&mut buf, 3);
        assert_eq!(buf, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    /// Flow with just [`FlowLayer::Flip`] performs a per-row channel
    /// reversal. Since `FlowLayer::Flip` is implemented by reversing
    /// each half and swapping z_a/z_b, this pins that the composed
    /// operation is equivalent to `torch.flip(z, [1])` per row of the
    /// merged buffer.
    #[test]
    fn flow_inverse_flip_only_stack_reverses_channel_axis_per_row() {
        let d_z = 4;
        let mel_seq_len = 2;
        let flow = SbV2Flow::from_layers(vec![FlowLayer::Flip], d_z);
        let z = vec![
            1.0_f32, 2.0, 3.0, 4.0, // row 0
            10.0, 20.0, 30.0, 40.0, // row 1
        ];
        let g = vec![0.0_f32; 8];

        let out = flow.inverse(&z, mel_seq_len, &g);

        assert_eq!(out, vec![4.0, 3.0, 2.0, 1.0, 40.0, 30.0, 20.0, 10.0]);
    }

    /// Two Flips in a row cancel — Flip is an involution — and the
    /// merged buffer is bit-identical to the input `z`.
    #[test]
    fn flow_inverse_two_flips_cancel_bit_identically() {
        let d_z = 6;
        let mel_seq_len = 2;
        let flow = SbV2Flow::from_layers(vec![FlowLayer::Flip, FlowLayer::Flip], d_z);
        let z: Vec<f32> = (0..mel_seq_len * d_z).map(|i| i as f32 * 0.5).collect();
        let g = vec![0.0_f32; 8];

        let out = flow.inverse(&z, mel_seq_len, &g);

        assert_eq!(out, z, "Flip ∘ Flip = identity");
    }

    // ---------------------------------------------------------------
    // Iteration-order pin (the highest-risk correctness item)
    // ---------------------------------------------------------------

    /// Iteration-order sentinel: with a `[TCL0, Flip, TCL1, Flip]`
    /// stack the reverse-direction inverse must apply
    /// `Flip ∘ TCL1.inv ∘ Flip ∘ TCL0.inv` to `(z_a, z_b)` — matches
    /// upstream `TransformerCouplingBlock.forward(x, x_mask, g=g,
    /// reverse=True)`'s `for flow in reversed(self.flows): x = flow(x,
    /// ..., reverse=True)` loop.
    ///
    /// # Construction
    ///
    /// We pick TCL0 and TCL1 whose reverse-mode effect is a simple
    /// constant subtraction on `z_b` (pre = 0 + spk_emb = 0, so h has
    /// only pre_bias; encoder_stack empty; post = identity; hence
    /// m = pre_bias broadcast per position). Different pre_bias values
    /// on the two TCLs make an order-swap observably different in the
    /// output.
    ///
    /// # Hand trace (half_d_z = 1, d_z = 2, mel_seq_len = 1)
    ///
    /// - TCL0 pre_bias = [100.0]  → m = 100 per row.
    /// - TCL1 pre_bias = [10.0]   → m = 10 per row.
    /// - z = [1.0, 2.0]           → z_a = [1.0], z_b = [2.0].
    ///
    /// Reverse iteration order over `[TCL0, Flip, TCL1, Flip]` yields
    /// `[Flip, TCL1, Flip, TCL0]`:
    ///
    /// 1. Flip: swap → z_a = [2.0], z_b = [1.0].
    /// 2. TCL1.inv: z_b -= 10 → z_b = [-9.0].
    /// 3. Flip: swap → z_a = [-9.0], z_b = [2.0].
    /// 4. TCL0.inv: z_b -= 100 → z_b = [-98.0].
    ///
    /// Final merged: `[z_a, z_b] = [-9.0, -98.0]`.
    ///
    /// A wrong iteration order (say `[TCL0, Flip, TCL1, Flip]` — forward
    /// iter — with the wrong swap placement) produces a different
    /// number, so this test is a genuine pin.
    #[test]
    fn flow_inverse_two_tcl_two_flip_pins_reverse_iteration_order() {
        let d_z = 2;
        let half_d_z = 1;
        let d_hidden = 1;
        let gin_channels = 1;

        let tcl0 = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.0; d_hidden * half_d_z], // pre_weight
            vec![100.0],                    // pre_bias (large so orderings are distinguishable)
            vec![0.0; d_hidden * gin_channels],
            vec![0.0; d_hidden],
            vec![1.0], // post_weight identity (d_hidden=1)
            vec![0.0; half_d_z],
            true,
        );
        let tcl1 = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.0; d_hidden * half_d_z],
            vec![10.0],
            vec![0.0; d_hidden * gin_channels],
            vec![0.0; d_hidden],
            vec![1.0],
            vec![0.0; half_d_z],
            true,
        );

        let flow = SbV2Flow::from_layers(
            vec![
                FlowLayer::Coupling(tcl0),
                FlowLayer::Flip,
                FlowLayer::Coupling(tcl1),
                FlowLayer::Flip,
            ],
            d_z,
        );

        let mel_seq_len = 1;
        let z = vec![1.0_f32, 2.0];
        let g = vec![0.0];

        let out = flow.inverse(&z, mel_seq_len, &g);

        // Expected from the hand trace above.
        let expected = [-9.0_f32, -98.0];
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "position {i}: expected {b}, got {a} (out = {out:?})",
            );
        }
    }

    /// Forward-vs-reverse iteration distinguishability: the same TCL0 /
    /// TCL1 pair applied in the **forward** iteration order (`[TCL0,
    /// Flip, TCL1, Flip]`) produces a different output than the
    /// reverse-iteration one. This is the sentinel that pins the
    /// "which order does `inverse` walk" question — if a future refactor
    /// swaps `layers.iter().rev()` back to `layers.iter()`, this test
    /// would produce the forward-iter number, which does not match the
    /// reverse-iter number pinned below.
    ///
    /// # Why `pre_weight = identity` (not `pre_weight = 0`)
    ///
    /// If `pre_weight = 0`, then `m = post(pre(z_a) + spk_emb + 0)` is a
    /// **constant** per position (independent of z_a's content) —
    /// namely `post_bias + post(pre_bias)`. Constant subtraction from
    /// z_b commutes trivially with itself, so both iteration orders
    /// would produce the same output on this trivialized case. Real
    /// VITS2 couplings have `m` depend on z_a via a non-trivial pre →
    /// encoder → post chain; here we approximate that dependence by
    /// setting `pre_weight = identity` and keeping `encoder_stack` empty
    /// / `post = identity`, so `m = z_a` per position — the minimal
    /// non-trivial coupling that still leaves the arithmetic
    /// hand-checkable.
    #[test]
    fn flow_inverse_two_tcl_two_flip_reverse_iter_differs_from_forward_iter() {
        let d_z = 4;
        let half_d_z = 2;
        let d_hidden = 2; // = half_d_z, so pre / post can be identity square matrices
        let gin_channels = 1;

        // pre_weight = identity (row-major [d_hidden, half_d_z] = [2, 2]).
        // post_weight = identity ([half_d_z, d_hidden] = [2, 2]).
        // → h = z_a, m = z_a per position.
        // Differentiate the two TCLs by pre_bias: TCL0 adds [100, 200]
        // to h, TCL1 adds [1, 2] — so TCL0.inv subtracts (z_a + [100,
        // 200]) from z_b, TCL1.inv subtracts (z_a + [1, 2]) from z_b.
        let identity_2x2 = vec![1.0_f32, 0.0, 0.0, 1.0];
        let tcl0 = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            identity_2x2.clone(),
            vec![100.0, 200.0], // pre_bias — asymmetric to break row-flip symmetry
            vec![0.0; d_hidden * gin_channels],
            vec![0.0; d_hidden],
            identity_2x2.clone(),
            vec![0.0; half_d_z],
            true,
        );
        let tcl1 = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            identity_2x2.clone(),
            vec![1.0, 2.0], // different pre_bias
            vec![0.0; d_hidden * gin_channels],
            vec![0.0; d_hidden],
            identity_2x2,
            vec![0.0; half_d_z],
            true,
        );

        let flow = SbV2Flow::from_layers(
            vec![
                FlowLayer::Coupling(tcl0),
                FlowLayer::Flip,
                FlowLayer::Coupling(tcl1),
                FlowLayer::Flip,
            ],
            d_z,
        );

        let mel_seq_len = 1;
        // z = [z_a[0], z_a[1], z_b[0], z_b[1]] = [10, 20, 30, 40].
        let z = vec![10.0_f32, 20.0, 30.0, 40.0];
        let g = vec![0.0];

        let out = flow.inverse(&z, mel_seq_len, &g);

        // Reverse iteration `[Flip, TCL1.inv, Flip, TCL0.inv]`, with the
        // TCL contract `m = z_a + pre_bias`:
        //  Start: z_a = [10, 20], z_b = [30, 40].
        //  1. Flip: reverse each half → z_a = [20, 10], z_b = [40, 30].
        //           swap: z_a = [40, 30], z_b = [20, 10].
        //  2. TCL1.inv: m = z_a + [1, 2] = [41, 32].
        //     z_b -= m → z_b = [-21, -22].
        //  3. Flip: reverse each half → z_a = [30, 40], z_b = [-22, -21].
        //           swap: z_a = [-22, -21], z_b = [30, 40].
        //  4. TCL0.inv: m = z_a + [100, 200] = [78, 179].
        //     z_b -= m → z_b = [-48, -139].
        //  Merged = [z_a, z_b] = [-22, -21, -48, -139].
        let expected_reverse = vec![-22.0_f32, -21.0, -48.0, -139.0];
        for (i, (a, b)) in out.iter().zip(expected_reverse.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "position {i}: expected {b}, got {a} — reverse-iter result must match hand \
                 trace, otherwise `inverse` walks layers in the wrong order (full output = \
                 {out:?})"
            );
        }

        // The forward iteration would produce a different number — the
        // pair of hand traces distinguishes the two orderings, which is
        // what this sentinel is designed to catch. Hand trace:
        //   Start: z_a = [10, 20], z_b = [30, 40].
        //   1. TCL0.inv: m = z_a + [100, 200] = [110, 220].
        //      z_b -= m → z_b = [-80, -180].
        //   2. Flip: reverse+swap → z_a = [-180, -80], z_b = [20, 10].
        //   3. TCL1.inv: m = z_a + [1, 2] = [-179, -78].
        //      z_b -= m → z_b = [199, 88].
        //   4. Flip: reverse+swap → z_a = [88, 199], z_b = [-80, -180].
        //   Merged = [88, 199, -80, -180].
        //
        // 88 ≠ -22, so the two orderings are distinguishable.
        let forward_iter_hand = vec![88.0_f32, 199.0, -80.0, -180.0];
        assert_ne!(
            forward_iter_hand, expected_reverse,
            "forward-iter and reverse-iter must produce distinguishable outputs so a \
             regression that flips the iteration direction is caught here"
        );
    }

    // ---------------------------------------------------------------
    // Split / merge round-trip (unchanged from pre-Blocker 2b)
    // ---------------------------------------------------------------

    #[test]
    fn split_and_merge_halves_roundtrip_bit_identically() {
        let d_z = 6;
        let half = 3;
        let z = vec![
            1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];
        let (z_a, z_b) = split_halves(&z, d_z, half);
        let merged = merge_halves(&z_a, &z_b, half);
        assert_eq!(merged, z);
    }

    // ---------------------------------------------------------------
    // FLOW-DEFENSE-BUNDLE (2026-08-09) tripwires
    // ---------------------------------------------------------------

    /// A coupling layer with `gin_channels == 0` (no speaker
    /// conditioning at train time — upstream `nn.Linear(0, d_hidden)`)
    /// must accept `g == []` at inference and produce the bias-only
    /// speaker contribution (`spk = spk_emb_bias.clone()`). This pins
    /// the explicit empty-conditioning branch added in FLOW-DEFENSE.
    #[test]
    fn zero_gin_channels_accepts_empty_g_and_uses_bias_only_spk() {
        let half_d_z = 2;
        let d_hidden = 3;
        let gin_channels = 0; // no speaker conditioning
        // Use non-zero post_weight so the pipeline observably mutates
        // z_b (a fully-zero post would trivially leave z_b unchanged,
        // making the "accepts empty g" contract vacuously satisfied).
        let tcl = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.5; d_hidden * half_d_z],
            vec![0.1; d_hidden],
            Vec::new(),           // spk_emb_weight is [d_hidden, 0] == empty
            vec![0.7, -0.4, 0.2], // spk_emb_bias becomes the sole speaker term
            vec![0.3; half_d_z * d_hidden],
            vec![0.05; half_d_z],
            true,
        );
        let z_a = vec![1.0_f32, 2.0, 3.0, 4.0]; // T=2, half_d_z=2
        let z_b_before = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut z_b = z_b_before.clone();
        // Must not panic on empty g; loud-fail would be a
        // debug_assert on the gin_channels==0 → g!=[] mismatch.
        tcl.inverse(&z_a, &mut z_b, 2, &[]);
        // Non-zero post_weight guarantees observable mutation of z_b.
        assert_ne!(
            z_b, z_b_before,
            "FLOW-DEFENSE: the gin_channels==0 + empty-g branch must run the pipeline through \
             to post/affine (not short-circuit into a no-op)"
        );
    }

    #[test]
    #[should_panic(expected = "g must be [gin_channels]")]
    fn zero_gin_channels_rejects_nonempty_g() {
        // The existing generic shape-check `g.len() == self.gin_channels`
        // already fires when g is non-empty on a gin_channels=0 layer.
        // FLOW-DEFENSE's explicit empty-conditioning branch adds a
        // redundant `g.is_empty()` `debug_assert!` inside its arm —
        // but the generic shape-check fires first, so the observable
        // panic message is the generic one. Pin to that.
        let half_d_z = 2;
        let d_hidden = 3;
        let gin_channels = 0;
        let tcl = make_tcl(
            half_d_z,
            d_hidden,
            gin_channels,
            vec![0.0; d_hidden * half_d_z],
            vec![0.0; d_hidden],
            Vec::new(),
            vec![0.0; d_hidden],
            vec![0.0; half_d_z * d_hidden],
            vec![0.0; half_d_z],
            true,
        );
        let z_a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut z_b = vec![10.0_f32, 20.0, 30.0, 40.0];
        // Non-empty g on a gin_channels==0 layer: debug_assert must fire.
        tcl.inverse(&z_a, &mut z_b, 2, &[0.1_f32, -0.2, 0.3]);
    }

    #[test]
    #[should_panic(expected = "every Coupling in a well-formed flow must share")]
    fn from_layers_rejects_mismatched_gin_channels() {
        // Two coupling layers with DIFFERENT gin_channels — a
        // well-formed flow requires them to agree. `from_layers`
        // debug-asserts cross-layer consistency post-FLOW-DEFENSE.
        let half_d_z = 2;
        let d_hidden = 3;
        let layer_a = make_tcl(
            half_d_z,
            d_hidden,
            4, // gin_channels = 4
            vec![0.0; d_hidden * half_d_z],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * 4],
            vec![0.0; d_hidden],
            vec![0.0; half_d_z * d_hidden],
            vec![0.0; half_d_z],
            true,
        );
        let layer_b = make_tcl(
            half_d_z,
            d_hidden,
            5, // MISMATCH: gin_channels = 5 ≠ 4
            vec![0.0; d_hidden * half_d_z],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * 5],
            vec![0.0; d_hidden],
            vec![0.0; half_d_z * d_hidden],
            vec![0.0; half_d_z],
            true,
        );
        let _flow = SbV2Flow::from_layers(
            vec![
                FlowLayer::Coupling(layer_a),
                FlowLayer::Flip,
                FlowLayer::Coupling(layer_b),
            ],
            2 * half_d_z,
        );
    }

    #[test]
    fn from_layers_accepts_consistent_gin_channels() {
        // All layers agree on gin_channels — must construct without
        // panic.
        let half_d_z = 2;
        let d_hidden = 3;
        let gin = 4;
        let layer_a = make_tcl(
            half_d_z,
            d_hidden,
            gin,
            vec![0.0; d_hidden * half_d_z],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * gin],
            vec![0.0; d_hidden],
            vec![0.0; half_d_z * d_hidden],
            vec![0.0; half_d_z],
            true,
        );
        let layer_b = make_tcl(
            half_d_z,
            d_hidden,
            gin,
            vec![0.0; d_hidden * half_d_z],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * gin],
            vec![0.0; d_hidden],
            vec![0.0; half_d_z * d_hidden],
            vec![0.0; half_d_z],
            true,
        );
        let flow = SbV2Flow::from_layers(
            vec![
                FlowLayer::Coupling(layer_a),
                FlowLayer::Flip,
                FlowLayer::Coupling(layer_b),
            ],
            2 * half_d_z,
        );
        assert_eq!(flow.gin_channels(), gin);
    }
}
