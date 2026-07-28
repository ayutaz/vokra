//! SBV2 style-vector injector: AdaIN-style scale + bias conditioning.
//! (Clean-room comment: see `mod.rs` — references VITS/VITS2 papers and
//! the AdaIN paper arXiv:1703.06868 (Huang & Belongie 2017), no SBV2/BV2
//! source referenced.)
//!
//! # Weight layout
//!
//! [`StyleVectorInjector`] holds two separate `[d_target, d_style]`
//! projection matrices — `proj_scale` and `proj_bias` — rather than one
//! combined `[2*d_target, d_style]` matrix. This matches the struct's
//! field names literally (`proj_scale: Vec<f32>`, `proj_bias: Vec<f32>`)
//! and keeps the "scale half" and "bias half" as two independently
//! addressable tensors, so a future GGUF-loading constructor (Task 25
//! converter) can read them as two separate named GGUF tensors instead
//! of slicing a fused one.
//!
//! Each projection is a **linear map without a bias term**:
//! `scale_delta = proj_scale · style_vec`, `bias = proj_bias · style_vec`
//! (no `+ b`). Both a zero `style_vec` (for any weights) and zero
//! weights (for any `style_vec`) independently guarantee identity
//! injection, since the projection is a pure linear map with no bias
//! term (`f(0) = 0` for any `W`, and `0·x = 0` for any `x`). A nonzero
//! constant bias on the projection itself would break the first
//! invariant — it would be absorbed into upstream layers per standard
//! AdaIN practice, which is why this implementation omits it. The
//! module's tests exercise the zero-weights case
//! (`zero_projections_produce_identity`). If a real SBV2 checkpoint
//! turns out to carry a projection bias, that can be added to this
//! struct additively (new `Option<Vec<f32>>` fields) without breaking
//! this constructor's existing callers.

/// AdaIN-style per-utterance style conditioning: projects a `d_style`-dim
/// style vector into a per-channel `(scale, bias)` pair over `d_target`
/// channels, then applies `h[i, d] = h[i, d] * (1 + scale[d]) + bias[d]`
/// to every sequence position `i` of a `[seq_len, d_target]` hidden-state
/// buffer (same `scale`/`bias` broadcast across all positions).
///
/// The two projections (`proj_scale`, `proj_bias`) are independent
/// `[d_target, d_style]` row-major weight matrices — see the module docs
/// for why this crate uses two matrices rather than one fused
/// `[2*d_target, d_style]` matrix, and why neither carries a bias term.
pub struct StyleVectorInjector {
    /// Row-major `[d_target, d_style]` weights projecting `style_vec` to
    /// the additive scale delta (`scale_delta = proj_scale · style_vec`).
    proj_scale: Vec<f32>,
    /// Row-major `[d_target, d_style]` weights projecting `style_vec` to
    /// the bias (`bias = proj_bias · style_vec`).
    proj_bias: Vec<f32>,
    /// Input style-vector dimensionality (`style_vec.len()`).
    d_style: usize,
    /// Target hidden-state channel count (`hidden.len() == seq_len *
    /// d_target`).
    d_target: usize,
}

impl StyleVectorInjector {
    /// Builds an injector from two pre-trained `[d_target, d_style]`
    /// row-major projection weight matrices.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — this is a
    /// hot inner-loop constructor, not a public API validation boundary)
    /// if `proj_scale.len() != d_target * d_style` or `proj_bias.len() !=
    /// d_target * d_style`.
    pub fn from_projections(
        proj_scale: Vec<f32>,
        proj_bias: Vec<f32>,
        d_style: usize,
        d_target: usize,
    ) -> Self {
        debug_assert_eq!(
            proj_scale.len(),
            d_target * d_style,
            "proj_scale must be [d_target, d_style]"
        );
        debug_assert_eq!(
            proj_bias.len(),
            d_target * d_style,
            "proj_bias must be [d_target, d_style]"
        );
        Self {
            proj_scale,
            proj_bias,
            d_style,
            d_target,
        }
    }

    /// The style-vector input dimensionality every
    /// [`inject`](Self::inject) call's `style_vec` must match. Task 23
    /// (`SbV2Model`'s `TtsEngine` adapter) uses this to size a default
    /// identity (all-zero) style vector for callers of the cross-engine
    /// [`vokra_core::SynthesisRequest`] shape, which carries no style-vector
    /// field of its own — mirrors
    /// [`SbV2TextEncoder::d_model`](super::text_encoder::SbV2TextEncoder::d_model)'s
    /// identical private-field-accessor precedent.
    pub fn d_style(&self) -> usize {
        self.d_style
    }

    /// Applies AdaIN-style style conditioning to `hidden` in place.
    ///
    /// `hidden` is a flat `[seq_len, d_target]` row-major buffer (i.e.
    /// `hidden[i * d_target + d]` addresses position `i`, channel `d`).
    /// `style_vec` is a `[d_style]` per-utterance style embedding,
    /// projected once into a `(scale, bias)` pair over `d_target`
    /// channels and then broadcast identically to every position `i`:
    ///
    /// ```text
    /// scale[d] = sum_s proj_scale[d, s] * style_vec[s]
    /// bias[d]  = sum_s proj_bias[d, s]  * style_vec[s]
    /// hidden[i, d] = hidden[i, d] * (1 + scale[d]) + bias[d]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`; see [`from_projections`]'s panic
    /// docs for why this is a `debug_assert!` rather than a `Result`) if
    /// `hidden.len() != seq_len * self.d_target` or `style_vec.len() !=
    /// self.d_style`.
    ///
    /// [`from_projections`]: StyleVectorInjector::from_projections
    pub fn inject(&self, hidden: &mut [f32], seq_len: usize, style_vec: &[f32]) {
        debug_assert_eq!(
            hidden.len(),
            seq_len * self.d_target,
            "hidden must be [seq_len, d_target]"
        );
        debug_assert_eq!(style_vec.len(), self.d_style, "style_vec must be [d_style]");

        // Project style_vec -> (scale, bias), each [d_target], once —
        // shared across all seq_len positions below.
        let mut scale = vec![0.0_f32; self.d_target];
        let mut bias = vec![0.0_f32; self.d_target];
        for (d, (scale_d, bias_d)) in scale.iter_mut().zip(bias.iter_mut()).enumerate() {
            let row = d * self.d_style;
            let scale_row = &self.proj_scale[row..row + self.d_style];
            let bias_row = &self.proj_bias[row..row + self.d_style];
            let mut s = 0.0_f32;
            let mut b = 0.0_f32;
            for ((&sw, &bw), &x) in scale_row.iter().zip(bias_row.iter()).zip(style_vec.iter()) {
                s += sw * x;
                b += bw * x;
            }
            *scale_d = s;
            *bias_d = b;
        }

        let n_active = seq_len * self.d_target;
        for hidden_row in hidden[..n_active].chunks_exact_mut(self.d_target) {
            for ((h, &sc), &bi) in hidden_row.iter_mut().zip(scale.iter()).zip(bias.iter()) {
                *h = *h * (1.0 + sc) + bi;
            }
        }
    }
}
