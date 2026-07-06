//! Kokoro-82M prosody predictor (M2-07-T13/T15) — scaffold.
//!
//! Input surface: text-encoder output `encoded` `[hidden_dim, T]` (channel-major,
//! same layout as every other piper / kokoro tensor) + a style vector `style`
//! `[style_dim]` looked up from the voicepack. Output surface: three per-frame
//! streams `(log_durations, f0, energy)` each `[T]`, in the shape the T14 length
//! regulator + T16 decoder input processing consume.
//!
//! This file is the T13 landing point: interface + shape validation +
//! **deterministic** forward path, exercised by synthetic-weight shape /
//! determinism tests. The concrete Kokoro prosody-predictor topology (per-head
//! conv stacks, style-conditioned FiLM / AdaIN — see
//! `docs/adr/0007-kokoro-native.md` §Op gap) lands once T02 upstream inspection
//! pins the real weight names + hparams; the plan explicitly withholds those to
//! avoid inventing constants (M2-07 §Design decisions, "shape-driven, never
//! silent-defaulted"). Until then the deterministic path is a **fixed-function
//! reduction** over the inputs — the shape / no-RNG surface future callers
//! (T14/T15/T18) depend on is stable, and the numerics are documented in the
//! forward comment so the T02 rewrite is a drop-in.
//!
//! T15 additionally lands the **F0-conditioning conv path** and
//! **energy-conditioning conv path** the iSTFTNet decoder consumes at its
//! input-processing stage
//! (`crates/vokra-models/src/kokoro/decoder.rs`,
//! `z += F0_proj(f0)`, `z += Energy_proj(energy)` — the StyleTTS 2 / Kokoro
//! iSTFTNet pattern). Both are 1×1 (kernel=1, in_ch=1) 1-D convolutions
//! projecting a per-frame scalar contour `[t]` onto `[hidden_dim, t]`
//! channel-major, via the shared [`super::nn::conv1d`] im2col+GEMM path.
//!
//! FR-EX-08: every shape mismatch is a loud [`VokraError::InvalidArgument`],
//! and the not-yet-wired stochastic path is a loud
//! [`VokraError::NotImplemented`] — never a silent zero output.

use vokra_core::{Result, VokraError};

use super::config::KokoroConfig;
use super::weights::TensorStore;
use crate::compute::Compute;

/// Kokoro prosody predictor (duration / F0 / energy heads).
///
/// Fields (`hidden_dim`, `style_dim`) come from the [`KokoroConfig`] the
/// converter wrote — never a hard-coded default. The T02-follow-on rewrite
/// adds per-head conv stacks + a style projection here; the T13 landing only
/// needs the two shape parameters for the deterministic-reduction forward
/// path.
pub(crate) struct ProsodyPredictor {
    hidden_dim: usize,
    style_dim: usize,
}

impl ProsodyPredictor {
    /// Loads the prosody predictor from a voice GGUF.
    ///
    /// T13 landing: no per-head weights are pinned here yet — the real Kokoro
    /// weight names are TBD at T02 upstream inspection, and inventing
    /// placeholder names risks a rename churn once a real checkpoint arrives.
    /// The shape-only capture (hidden / style dims from
    /// [`KokoroConfig`]) is what every downstream call needs to shape-check
    /// its inputs. Kept infallible so
    /// [`super::KokoroTts::from_gguf_with_policy`] can exercise the config +
    /// weight-license gate path end-to-end.
    #[allow(dead_code)] // called from KokoroTts::from_gguf_with_policy at T18
    pub(crate) fn load(_store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        Ok(Self {
            hidden_dim: config.hidden_dim,
            style_dim: config.style_dim,
        })
    }

    /// Predicts `(log_durations, f0, energy)` from the encoded features
    /// `encoded` `[hidden_dim, T]` under a style vector `style` `[style_dim]`.
    ///
    /// - `t` is `T` (the encoder / frame axis); `encoded.len()` must equal
    ///   `hidden_dim · t`, `style.len()` must equal `style_dim`. Any mismatch
    ///   is a loud [`VokraError::InvalidArgument`] (FR-EX-08).
    /// - `deterministic = true` is the parity path: the reduction below runs
    ///   without any RNG, so two calls with identical inputs return
    ///   bit-identical outputs (the determinism-test's core assertion).
    ///   `deterministic = false` returns [`VokraError::NotImplemented`] — the
    ///   stochastic path is deferred to the T02 follow-on together with the
    ///   real Kokoro prosody topology.
    ///
    /// # Deterministic reduction (T13 scaffold)
    ///
    /// A style bias `s[c] = tanh(sum_k(style[k] · w[c, k]))` with
    /// `w[c, k] = (k + 1) · (c + 1) / (style_dim · hidden_dim)` is a bounded,
    /// deterministic per-channel scalar that (a) depends on **every** style
    /// entry through a position-weighted projection (permuting `style` yields
    /// a different bias, so the style input is never silently dropped) and (b)
    /// stays finite (`tanh` bounds the sum). Per frame `ti`:
    ///
    /// - `log_duration[ti] = mean_c(encoded[c, ti] + s[c])`
    /// - `f0[ti]           = mean_c(encoded[c, ti] · s[c])`
    /// - `energy[ti]       = mean_c(encoded[c, ti]²)`
    ///
    /// This is **not** the upstream Kokoro topology; it is a shape-preserving,
    /// deterministic stand-in that keeps the caller surface stable until T02.
    /// The T02 rewrite will swap the body while the signature is preserved.
    #[allow(dead_code)] // called by the T18 e2e path
    pub(crate) fn forward(
        &self,
        encoded: &[f32],
        style: &[f32],
        t: usize,
        deterministic: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if !deterministic {
            return Err(VokraError::NotImplemented(
                "kokoro stochastic prosody path wired at M2-07-T13 follow-on",
            ));
        }
        if encoded.len() != self.hidden_dim * t {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: encoded len {} != hidden_dim ({}) · t ({})",
                encoded.len(),
                self.hidden_dim,
                t,
            )));
        }
        if style.len() != self.style_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: style len {} != style_dim ({})",
                style.len(),
                self.style_dim,
            )));
        }

        // Per-channel style bias — a deterministic bounded function of the
        // whole style vector via a position-weighted projection
        // `w[c, k] = (k + 1)(c + 1) / (style_dim · hidden_dim)` (see forward
        // comment). Permutation-sensitive because each `k` carries a distinct
        // coefficient.
        let mut style_bias = vec![0.0f32; self.hidden_dim];
        if self.style_dim > 0 && self.hidden_dim > 0 {
            let norm = 1.0 / (self.style_dim as f32 * self.hidden_dim as f32);
            for (c, sb) in style_bias.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                let c_w = (c + 1) as f32;
                for (k, &sk) in style.iter().enumerate() {
                    acc += sk * (k + 1) as f32 * c_w;
                }
                *sb = (acc * norm).tanh();
            }
        }

        let mut log_dur = vec![0.0f32; t];
        let mut f0 = vec![0.0f32; t];
        let mut energy = vec![0.0f32; t];
        if self.hidden_dim == 0 || t == 0 {
            return Ok((log_dur, f0, energy));
        }
        let inv_c = 1.0 / self.hidden_dim as f32;
        for ti in 0..t {
            let mut sum = 0.0f32;
            let mut prod = 0.0f32;
            let mut sq = 0.0f32;
            for c in 0..self.hidden_dim {
                let v = encoded[c * t + ti];
                let s = style_bias[c];
                sum += v + s;
                prod += v * s;
                sq += v * v;
            }
            log_dur[ti] = sum * inv_c;
            f0[ti] = prod * inv_c;
            energy[ti] = sq * inv_c;
        }
        Ok((log_dur, f0, energy))
    }
}

// --- T15: F0 / energy conditioning conv paths (decoder input processing) ----
//
// These free functions land the two explicit "conditioning conv paths" the
// iSTFTNet decoder consumes at its input-processing stage: the per-frame
// prosody predictor outputs (F0 contour, energy contour) are each projected
// from `[t]` scalar streams up to `[hidden_dim, t]` channel-major tensors via
// a 1×1 (kernel=1, in_ch=1) 1-D convolution, so the decoder can fold them
// into its input latent (`z += f0_proj`, `z += energy_proj`) at T18 wire-up.
//
// The 1×1 conv is lowered through the shared [`super::nn::conv1d`] im2col+GEMM
// path — the same primitive the T16 upsample stack uses — so the numerics
// stay consistent across every kokoro conv site.

/// Projects a per-frame scalar contour (F0 or energy) onto `[hidden_dim, t]`
/// channel-major via a 1×1 (kernel=1, in_ch=1) 1-D convolution — the shared
/// conditioning conv path the decoder consumes at its input processing stage
/// (`crates/vokra-models/src/kokoro/decoder.rs`).
///
/// `contour` is `[t]`; `weight` is `[hidden_dim, 1, 1]` flattened to
/// `hidden_dim` floats (PyTorch `Conv1d` layout with `out_ch=hidden_dim,
/// in_ch=1, kernel=1`); `bias` (when `Some`) is `[hidden_dim]`. Returns
/// `[hidden_dim · t]` channel-major, matching every other kokoro tensor.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if `hidden_dim == 0`, if `weight`
/// is not `hidden_dim` floats long, or if `bias` (when present) is not
/// `hidden_dim` floats long (FR-EX-08: never a silent zero-fill).
#[allow(dead_code)] // consumed by the T18 decoder input-processing wire-up
pub(crate) fn conditioning_conv_1x1(
    compute: &Compute,
    contour: &[f32],
    hidden_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if hidden_dim == 0 {
        return Err(VokraError::InvalidArgument(
            "kokoro conditioning conv: hidden_dim must be > 0".to_owned(),
        ));
    }
    if weight.len() != hidden_dim {
        return Err(VokraError::InvalidArgument(format!(
            "kokoro conditioning conv weight: expected {hidden_dim} floats \
             (out_ch=hidden_dim, in_ch=1, kernel=1), got {}",
            weight.len(),
        )));
    }
    if let Some(b) = bias {
        if b.len() != hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro conditioning conv bias: expected {hidden_dim} floats, got {}",
                b.len(),
            )));
        }
    }
    let t = contour.len();
    let (out, out_t) = super::nn::conv1d(
        compute, contour, /* in_ch */ 1, /* in_len */ t, weight,
        /* out_ch */ hidden_dim, /* kernel */ 1, bias, /* stride */ 1,
        /* pad */ 0, /* dilation */ 1, /* groups */ 1,
    );
    debug_assert_eq!(
        out_t, t,
        "conditioning_conv_1x1: kernel=1 must preserve time axis"
    );
    debug_assert_eq!(out.len(), hidden_dim * t);
    Ok(out)
}

/// F0-conditioning conv path — projects the per-frame F0 contour onto the
/// decoder's hidden channel count via [`conditioning_conv_1x1`]. See that
/// function for the shape and error contract.
#[allow(dead_code)] // consumed by the T18 decoder input-processing wire-up
pub(crate) fn f0_conditioning(
    compute: &Compute,
    f0: &[f32],
    hidden_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    conditioning_conv_1x1(compute, f0, hidden_dim, weight, bias)
}

/// Energy-conditioning conv path — projects the per-frame energy contour onto
/// the decoder's hidden channel count via [`conditioning_conv_1x1`]. See that
/// function for the shape and error contract.
#[allow(dead_code)] // consumed by the T18 decoder input-processing wire-up
pub(crate) fn energy_conditioning(
    compute: &Compute,
    energy: &[f32],
    hidden_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    conditioning_conv_1x1(compute, energy, hidden_dim, weight, bias)
}

#[cfg(test)]
mod f0 {
    use super::*;

    /// The F0 and energy conditioning conv paths both project `[t]` → `[hidden_dim, t]`
    /// channel-major, so the decoder's input-processing `z += f0_proj` /
    /// `z += energy_proj` operates on matching shapes. This test pins that
    /// shape contract with synthetic (non-hparam) weights, and additionally
    /// pins the 1×1 conv semantics per (channel, time) so an im2col axis swap
    /// would surface at test time rather than mid-forward at T18.
    #[test]
    fn f0_energy_predictors_shape_match() {
        // Tiny synthetic sizes — no real Kokoro hparams asserted (M2-07 rule:
        // shape-driven, never invent constants).
        let compute = Compute::cpu();
        let hidden_dim = 8;
        let t = 5;

        let f0: Vec<f32> = (0..t).map(|i| 0.1 + i as f32 * 0.05).collect();
        let energy: Vec<f32> = (0..t).map(|i| 0.2 - i as f32 * 0.03).collect();
        // weight = [hidden_dim, 1, 1] flat = hidden_dim floats.
        let f0_w: Vec<f32> = (0..hidden_dim).map(|i| (i + 1) as f32 * 0.5).collect();
        let en_w: Vec<f32> = (0..hidden_dim).map(|i| -((i + 1) as f32) * 0.25).collect();
        let en_b: Vec<f32> = (0..hidden_dim).map(|i| 0.125 * i as f32).collect();

        let f0_out = f0_conditioning(&compute, &f0, hidden_dim, &f0_w, None).expect("f0 ok");
        let en_out = energy_conditioning(&compute, &energy, hidden_dim, &en_w, Some(&en_b))
            .expect("energy ok");

        // (a) Shape match: both paths must produce `[hidden_dim, t]`
        // channel-major, which the T18 wire-up depends on.
        assert_eq!(f0_out.len(), hidden_dim * t, "f0 output shape mismatch");
        assert_eq!(en_out.len(), hidden_dim * t, "energy output shape mismatch");

        // (b) 1×1 conv semantics: each (channel c, time i) entry equals
        // `weight[c] · contour[i]  (+ bias[c] when present)`. Pins that the
        // paths did not swap axes through im2col+GEMM.
        for c in 0..hidden_dim {
            for i in 0..t {
                let f0_expected = f0_w[c] * f0[i];
                let en_expected = en_w[c] * energy[i] + en_b[c];
                let f0_actual = f0_out[c * t + i];
                let en_actual = en_out[c * t + i];
                assert!(
                    (f0_actual - f0_expected).abs() < 1e-6,
                    "f0[c={c},i={i}]: {f0_actual} vs {f0_expected}",
                );
                assert!(
                    (en_actual - en_expected).abs() < 1e-6,
                    "energy[c={c},i={i}]: {en_actual} vs {en_expected}",
                );
            }
        }
    }

    /// FR-EX-08: a weight vector whose length is not `hidden_dim` is a loud
    /// [`VokraError::InvalidArgument`], never silently zero-padded.
    #[test]
    fn f0_conditioning_rejects_wrong_weight_shape() {
        let compute = Compute::cpu();
        let f0 = vec![0.0f32; 3];
        let hidden_dim = 4;
        let bad_weight = vec![1.0f32; hidden_dim + 1];

        let err = f0_conditioning(&compute, &f0, hidden_dim, &bad_weight, None).unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("weight"), "error must name `weight`: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// FR-EX-08: a bias vector whose length is not `hidden_dim` is a loud
    /// [`VokraError::InvalidArgument`], never silently dropped.
    #[test]
    fn energy_conditioning_rejects_wrong_bias_shape() {
        let compute = Compute::cpu();
        let energy = vec![0.0f32; 3];
        let hidden_dim = 4;
        let weight = vec![1.0f32; hidden_dim];
        let bad_bias = vec![0.0f32; hidden_dim - 1];

        let err = energy_conditioning(&compute, &energy, hidden_dim, &weight, Some(&bad_bias))
            .unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("bias"), "error must name `bias`: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// FR-EX-08: `hidden_dim == 0` is a loud [`VokraError::InvalidArgument`],
    /// never a silent empty output.
    #[test]
    fn conditioning_conv_rejects_zero_hidden_dim() {
        let compute = Compute::cpu();
        let contour = vec![0.0f32; 3];
        let weight: Vec<f32> = Vec::new();
        let err = conditioning_conv_1x1(&compute, &contour, 0, &weight, None).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(hidden_dim: usize, style_dim: usize) -> ProsodyPredictor {
        ProsodyPredictor {
            hidden_dim,
            style_dim,
        }
    }

    /// `t`-long streams for each head, matching the input frame axis. The
    /// exact numerical values are a T13 scaffold; the shape contract is the
    /// stable surface downstream callers (T14 length regulator, T18 e2e) key
    /// off.
    #[test]
    fn predicts_expected_shapes() {
        let p = build(4, 3);
        let t = 5;
        let encoded: Vec<f32> = (0..4 * t).map(|i| i as f32 * 0.05).collect();
        let style = vec![0.1, -0.2, 0.3];
        let (dur, f0, energy) = p.forward(&encoded, &style, t, true).expect("forward ok");
        assert_eq!(dur.len(), t);
        assert_eq!(f0.len(), t);
        assert_eq!(energy.len(), t);
        // Every entry is finite — the reduction never produces NaN/Inf for a
        // finite input, which the T18 wire-up depends on.
        for v in dur.iter().chain(f0.iter()).chain(energy.iter()) {
            assert!(v.is_finite(), "prosody head produced non-finite: {v}");
        }
    }

    /// The deterministic path is **bit-exact** reproducible: two calls with
    /// identical inputs and `deterministic = true` return equal
    /// `(log_dur, f0, energy)`. This is the parity-testable surface — a hidden
    /// RNG here would defeat the T20 quality-gate rerun-and-diff pattern.
    #[test]
    fn deterministic_is_reproducible() {
        let p = build(6, 4);
        let t = 3;
        let encoded: Vec<f32> = (0..6 * t).map(|i| ((i * 7) % 11) as f32 * 0.03).collect();
        let style = vec![0.4, -0.1, 0.25, -0.35];
        let a = p.forward(&encoded, &style, t, true).expect("first ok");
        let b = p.forward(&encoded, &style, t, true).expect("second ok");
        assert_eq!(a.0, b.0, "log_dur differs across calls");
        assert_eq!(a.1, b.1, "f0 differs across calls");
        assert_eq!(a.2, b.2, "energy differs across calls");
    }

    /// The style bias reads **every** entry (not just one), so a permutation of
    /// `style` yields different outputs. Guards against the "silently drops the
    /// style" regression a fresh port could introduce.
    #[test]
    fn style_permutation_changes_outputs() {
        let p = build(4, 3);
        let t = 4;
        let encoded: Vec<f32> = (0..4 * t).map(|i| i as f32 * 0.02).collect();
        let a = p
            .forward(&encoded, &[0.1, 0.2, 0.3], t, true)
            .expect("first ok");
        let b = p
            .forward(&encoded, &[0.3, 0.1, 0.2], t, true)
            .expect("permuted ok");
        assert_ne!(a.1, b.1, "permuted style must reach f0");
    }

    /// FR-EX-08: an encoded vector whose length is not `hidden_dim · t` is a
    /// loud [`VokraError::InvalidArgument`], never silently truncated.
    #[test]
    fn rejects_encoded_shape_mismatch() {
        let p = build(4, 3);
        let t = 5;
        // 4 · 5 = 20 expected; ship 19.
        let encoded = vec![0.0f32; 4 * t - 1];
        let style = vec![0.0; 3];
        match p.forward(&encoded, &style, t, true) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("encoded"), "error must name `encoded`: {msg}");
            }
            other => panic!("expected InvalidArgument for encoded, got {other:?}"),
        }
    }

    /// FR-EX-08: a style vector whose length is not `style_dim` is a loud
    /// [`VokraError::InvalidArgument`], never zero-padded.
    #[test]
    fn rejects_style_shape_mismatch() {
        let p = build(4, 3);
        let t = 2;
        let encoded = vec![0.0f32; 4 * t];
        let style = vec![0.0f32; 5];
        match p.forward(&encoded, &style, t, true) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("style"), "error must name `style`: {msg}");
            }
            other => panic!("expected InvalidArgument for style, got {other:?}"),
        }
    }

    /// FR-EX-08: the stochastic path is not silently degraded to a
    /// deterministic zero — it is a loud [`VokraError::NotImplemented`].
    #[test]
    fn rejects_non_deterministic_path() {
        let p = build(4, 3);
        let t = 2;
        let encoded = vec![0.0f32; 4 * t];
        let style = vec![0.0f32; 3];
        assert!(matches!(
            p.forward(&encoded, &style, t, false),
            Err(VokraError::NotImplemented(_))
        ));
    }
}
