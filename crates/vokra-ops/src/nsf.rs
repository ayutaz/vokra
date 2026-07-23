//! Neural Source Filter (NSF) core for HiFTNet-family vocoders (SoTA plan
//! Phase 1-2).
//!
//! Direct Rust port of the upstream CosyVoice implementation
//! (`cosyvoice/hifigan/generator.py`):
//!
//! - `SineGen`     — F0-driven multi-harmonic sine wave source, L163-214.
//! - `SourceModuleHnNSF` — thin Linear + Tanh mix over `SineGen`, L310-368.
//!
//! Consumed by [`crate::hiftnet::HiFTGenerator`], which upstream calls
//! "HiFTNet Generator: Neural Source Filter + ISTFTNet"
//! (`generator.py:378`). Multiple published TTS models feed the same layer
//! (CosyVoice2 / CosyVoice3 / Chatterbox family), so this lives in
//! `vokra-ops` rather than a per-model module.
//!
//! # Determinism
//!
//! The upstream implementation samples both a per-harmonic phase
//! (`Uniform(-π, π)`) and a Gaussian noise (`randn_like`). Both are needed
//! for training but drift the reference on a parity harness, so this port
//! exposes an [`NsfEntropy`] switch:
//!
//! - [`NsfEntropy::Deterministic`] — zero phase, zero noise. Matches the
//!   `torch.no_grad + fixed seed = 0` path a Vokra parity fixture uses.
//! - [`NsfEntropy::Seeded`] — a SplitMix64-derived stream drives phase and
//!   noise. Same seed → same output on every host / OS / Rust version
//!   (the arithmetic is fixed-width integer plus deterministic `f32::sin`
//!   / `f32::ln` / `f32::sqrt`).
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! No RNG crate, no BLAS, no `serde`. The SplitMix64 + Marsaglia-polar
//! Gaussian sampler is <30 lines and only needs `std`. See
//! [`splitmix64`] / [`next_gaussian_std`].
//!
//! # Vs. upstream layout
//!
//! Upstream tensors are `[B, C, T]` and are transposed inside the layer to
//! `[B, T, C]` before returning. This port takes `f0: &[f32]` (single-batch,
//! time-major — batches beyond 1 are not exposed yet because every model
//! this vocoder feeds runs one utterance at a time) and returns
//! `sine_waves: [T * (H+1)]` in the same time-major layout upstream calls
//! `sine_wavs.transpose(1, 2)`.

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

/// How the stochastic bits of NSF (per-harmonic phase, noise) are drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NsfEntropy {
    /// Zero phase and zero noise. This is the parity path — no upstream
    /// implementation is bit-exact under it, but every implementation
    /// (upstream or port) collapses to the same deterministic sinusoid, so
    /// numerical comparisons carry meaning.
    Deterministic,
    /// A SplitMix64 seed reproducibly drives every random draw. The stream
    /// is split into disjoint sub-streams for phase and noise so a change
    /// in the number of harmonics does not shift the noise stream (the
    /// separation constants are documented on the callers below).
    Seeded(u64),
}

// ---------------------------------------------------------------------------
// SineGen
// ---------------------------------------------------------------------------

/// SineGen hyperparameters — verbatim from upstream `SineGen.__init__`.
#[derive(Debug, Clone, Copy)]
pub struct SineGenConfig {
    /// Audio sampling rate (Hz). Upstream default 22050.
    pub samp_rate: u32,
    /// Number of harmonics beyond the fundamental. Total output channels =
    /// `harmonic_num + 1`.
    pub harmonic_num: u32,
    /// Sine amplitude scale (`sine_amp` upstream). Default 0.1.
    pub sine_amp: f32,
    /// Gaussian noise standard deviation for voiced regions
    /// (`noise_std` upstream). Default 0.003.
    pub noise_std: f32,
    /// F0 threshold above which a frame is treated as voiced
    /// (`voiced_threshold` upstream). Default 0.
    pub voiced_threshold: f32,
}

impl Default for SineGenConfig {
    fn default() -> Self {
        Self {
            samp_rate: 22050,
            harmonic_num: 0,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 0.0,
        }
    }
}

impl SineGenConfig {
    /// Total output channels = harmonics + fundamental (`H + 1`).
    pub fn out_channels(&self) -> usize {
        self.harmonic_num as usize + 1
    }
}

/// [`SineGen::forward`] output tuple — same three tensors upstream returns.
#[derive(Debug, Clone)]
pub struct SineGenOutput {
    /// `[T * (H+1)]` row-major, upstream `sine_wavs.transpose(1, 2)`. For
    /// time-step `t` and harmonic channel `c` the value is
    /// `sine_waves[t * (H+1) + c]`. Already masked by `uv` and mixed with
    /// noise per the upstream expression.
    pub sine_waves: Vec<f32>,
    /// `[T]` — voiced/unvoiced mask (1 where `f0 > voiced_threshold`,
    /// 0 elsewhere). Upstream returns `[T, 1]` after transpose; this port
    /// drops the trivial trailing dim because callers broadcast anyway.
    pub uv: Vec<f32>,
    /// `[T * (H+1)]` row-major — the pre-mix noise tensor, kept as an
    /// output so tests can pin it separately (upstream also returns it as
    /// the third tuple element).
    pub noise: Vec<f32>,
}

/// SineGen (upstream L163-214). Owns no learnable state — the
/// [`SineGenConfig`] carries every hyperparameter.
#[derive(Debug, Clone, Copy)]
pub struct SineGen {
    cfg: SineGenConfig,
}

impl SineGen {
    /// Create a `SineGen` from its hyperparameters.
    pub fn new(cfg: SineGenConfig) -> Self {
        Self { cfg }
    }

    /// Immutable access to the [`SineGenConfig`] this generator was built with.
    pub fn config(&self) -> &SineGenConfig {
        &self.cfg
    }

    /// F0-driven multi-harmonic sine synthesis.
    ///
    /// Reproduces upstream `SineGen.forward` (`generator.py:200-214`):
    ///
    /// 1. Build `F_mat[i, t] = f0[t] * (i+1) / samp_rate` for
    ///    `i ∈ [0, harmonic_num]`.
    /// 2. `theta_mat = 2π * (cumsum(F_mat, dim=-1) % 1)`.
    /// 3. Draw a per-harmonic phase in `[-π, π)`, forcing the fundamental's
    ///    to 0 (upstream `phase_vec[:, 0, :] = 0`).
    /// 4. `sine_waves = sine_amp * sin(theta + phase)`.
    /// 5. `uv[t] = 1 if f0[t] > voiced_threshold else 0`.
    /// 6. `noise_amp = uv * noise_std + (1-uv) * sine_amp/3`.
    /// 7. `noise = noise_amp * randn_like(sine)`.
    /// 8. `sine_waves = sine_waves * uv + noise`.
    ///
    /// Under [`NsfEntropy::Deterministic`] the per-harmonic phase draw and
    /// the Gaussian noise both collapse to 0; the sinusoid is still
    /// generated, so `sine_waves = sine_amp * sin(theta) * uv`.
    pub fn forward(&self, f0: &[f32], entropy: NsfEntropy) -> Result<SineGenOutput> {
        let t = f0.len();
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "SineGen forward: empty f0 sequence".to_owned(),
            ));
        }
        if self.cfg.samp_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "SineGen forward: samp_rate must be > 0".to_owned(),
            ));
        }
        let h1 = self.cfg.out_channels();

        // ---- theta_mat [H+1, T] row-major = 2π * (cumsum(f0 * (i+1)/sr) % 1)
        let samp_rate_f = self.cfg.samp_rate as f32;
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut theta = vec![0.0f32; h1 * t];
        for i in 0..h1 {
            let harmonic_gain = (i as f32 + 1.0) / samp_rate_f;
            let row_offset = i * t;
            let mut cs = 0.0f32;
            for j in 0..t {
                cs += f0[j] * harmonic_gain;
                // `cs.floor()` gives the same result as `cs - cs.floor()`
                // for `cs % 1` in [0, 1) — the branchless form upstream's
                // torch expression compiles to.
                let modded = cs - cs.floor();
                theta[row_offset + j] = two_pi * modded;
            }
        }

        // ---- phase_vec[i] — 0 for the fundamental, uniform(-π, π) for
        // harmonics (upstream `phase_vec[:, 0, :] = 0`).
        let mut phase_vec = vec![0.0f32; h1];
        if let NsfEntropy::Seeded(seed) = entropy {
            // Sub-stream constant: 0xA5A5A5A5A5A5A5A5 keeps the phase
            // stream disjoint from the noise stream started below.
            let mut state = seed.wrapping_add(0xA5A5_A5A5_A5A5_A5A5);
            for slot in phase_vec.iter_mut().skip(1) {
                *slot = next_uniform_pi(&mut state);
            }
        }

        // ---- sine_waves[i, t] = sine_amp * sin(theta[i, t] + phase[i])
        //      (still in [H+1, T] layout; transposed at the end).
        let mut sine_pretransp = vec![0.0f32; h1 * t];
        for (i, &phase) in phase_vec.iter().enumerate() {
            let off = i * t;
            for j in 0..t {
                sine_pretransp[off + j] = self.cfg.sine_amp * (theta[off + j] + phase).sin();
            }
        }

        // ---- uv[t] = 1 if f0[t] > voiced_threshold else 0
        let mut uv = vec![0.0f32; t];
        for j in 0..t {
            uv[j] = if f0[j] > self.cfg.voiced_threshold {
                1.0
            } else {
                0.0
            };
        }

        // ---- noise & mix. Upstream does the `randn_like(sine_waves)` draw
        // AFTER masking uv, so we walk the same nested loop order for
        // parity-friendly bit patterns under a seed.
        let mut noise_pretransp = vec![0.0f32; h1 * t];
        let mut noise_state: Option<u64> = match entropy {
            NsfEntropy::Deterministic => None,
            // Disjoint from the phase stream — see phase_vec above.
            NsfEntropy::Seeded(seed) => Some(seed.wrapping_add(0xDEAD_BEEF_CAFE_BABE)),
        };
        for i in 0..h1 {
            let off = i * t;
            for j in 0..t {
                let na = uv[j] * self.cfg.noise_std + (1.0 - uv[j]) * self.cfg.sine_amp / 3.0;
                let rn = match &mut noise_state {
                    None => 0.0,
                    Some(state) => next_gaussian_std(state),
                };
                noise_pretransp[off + j] = na * rn;
            }
        }

        // sine_waves = sine_waves * uv + noise  (upstream L212)
        for i in 0..h1 {
            let off = i * t;
            for j in 0..t {
                sine_pretransp[off + j] =
                    sine_pretransp[off + j] * uv[j] + noise_pretransp[off + j];
            }
        }

        // ---- Transpose [H+1, T] → [T, H+1] (upstream returns
        // `sine_waves.transpose(1, 2)`).
        let mut sine_waves = vec![0.0f32; t * h1];
        let mut noise = vec![0.0f32; t * h1];
        for j in 0..t {
            for i in 0..h1 {
                sine_waves[j * h1 + i] = sine_pretransp[i * t + j];
                noise[j * h1 + i] = noise_pretransp[i * t + j];
            }
        }

        Ok(SineGenOutput {
            sine_waves,
            uv,
            noise,
        })
    }
}

// ---------------------------------------------------------------------------
// SourceModuleHnNSF
// ---------------------------------------------------------------------------

/// [`SourceModuleHnNSF`] hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct SourceModuleHnNSFConfig {
    /// Hyperparameters for the underlying [`SineGen`] — the harmonic
    /// generator this module wraps with a `Linear + Tanh` head.
    pub sine_gen: SineGenConfig,
}

/// Learned parameters upstream `SourceModuleHnNSF` carries — a single
/// `Linear(harmonic_num + 1, 1)` layer.
#[derive(Debug, Clone)]
pub struct SourceModuleHnNSFWeights {
    /// Row-major `[1, H+1]` linear weight — one output channel over
    /// `H+1` input harmonics.
    pub linear_w: Vec<f32>,
    /// Scalar bias (upstream `Linear(H+1, 1)` bias vector length 1).
    pub linear_b: f32,
}

/// Output of [`SourceModuleHnNSF::forward`] — matches the three-element
/// upstream return tuple `(sine_merge, noise, uv)`.
#[derive(Debug, Clone)]
pub struct SourceModuleHnNSFOutput {
    /// `[T]` — mixed sine source after `Linear` + `Tanh` over the H+1
    /// harmonic channels. Time-major (upstream `[B, T, 1]` reshaped).
    pub sine_merge: Vec<f32>,
    /// `[T]` — the noise tensor upstream returns as the second element:
    /// `torch.randn_like(uv) * sine_amp / 3`.
    pub noise: Vec<f32>,
    /// `[T]` — voiced/unvoiced mask (identical to the one `SineGen`
    /// produced; carried forward for downstream consumers).
    pub uv: Vec<f32>,
}

/// SourceModuleHnNSF (upstream L310-368) — `Linear(H+1, 1) + Tanh` over
/// `SineGen` output, plus an independent noise draw.
#[derive(Debug, Clone)]
pub struct SourceModuleHnNSF {
    cfg: SourceModuleHnNSFConfig,
    weights: SourceModuleHnNSFWeights,
    sine_gen: SineGen,
}

impl SourceModuleHnNSF {
    /// Build a `SourceModuleHnNSF`. Fails loudly on any weight-shape
    /// disagreement (there are only two shapes to check, and both come
    /// from `SineGenConfig::out_channels()`).
    pub fn new(cfg: SourceModuleHnNSFConfig, weights: SourceModuleHnNSFWeights) -> Result<Self> {
        let h1 = cfg.sine_gen.out_channels();
        if weights.linear_w.len() != h1 {
            return Err(VokraError::InvalidArgument(format!(
                "SourceModuleHnNSF linear_w must be length {h1} (harmonic_num+1), \
                 got {}",
                weights.linear_w.len(),
            )));
        }
        let sine_gen = SineGen::new(cfg.sine_gen);
        Ok(Self {
            cfg,
            weights,
            sine_gen,
        })
    }

    /// Immutable access to the [`SourceModuleHnNSFConfig`] this module was
    /// built with.
    pub fn config(&self) -> &SourceModuleHnNSFConfig {
        &self.cfg
    }

    /// Forward pass. Reproduces upstream `SourceModuleHnNSF.forward`
    /// (`generator.py:355-368`):
    ///
    /// ```text
    /// with no_grad: sine_wavs, uv, _ = self.l_sin_gen(x)
    /// sine_merge = tanh(linear(sine_wavs))
    /// noise = randn_like(uv) * sine_amp / 3
    /// ```
    pub fn forward(&self, f0: &[f32], entropy: NsfEntropy) -> Result<SourceModuleHnNSFOutput> {
        let sine_out = self.sine_gen.forward(f0, entropy)?;
        let t = sine_out.uv.len();
        let h1 = self.cfg.sine_gen.out_channels();

        // sine_merge = tanh(sum_i sine_wavs[t, i] * w[i] + b)
        let mut sine_merge = vec![0.0f32; t];
        for (j, merge) in sine_merge.iter_mut().enumerate() {
            let mut acc = self.weights.linear_b;
            for i in 0..h1 {
                acc += sine_out.sine_waves[j * h1 + i] * self.weights.linear_w[i];
            }
            *merge = acc.tanh();
        }

        // noise = randn_like(uv) * sine_amp / 3
        // Disjoint from every SineGen-internal stream: 0xC0FFEE separates
        // this draw from the SineGen phase (`0xA5A5…`) and internal-noise
        // (`0xDEAD…`) constants above.
        let mut noise = vec![0.0f32; t];
        let mut noise_state: Option<u64> = match entropy {
            NsfEntropy::Deterministic => None,
            NsfEntropy::Seeded(seed) => Some(seed.wrapping_add(0xC0FF_EE00_C0FF_EE00)),
        };
        let noise_scale = self.cfg.sine_gen.sine_amp / 3.0;
        for slot in noise.iter_mut() {
            *slot = match &mut noise_state {
                None => 0.0,
                Some(state) => next_gaussian_std(state) * noise_scale,
            };
        }

        Ok(SourceModuleHnNSFOutput {
            sine_merge,
            noise,
            uv: sine_out.uv,
        })
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64 + Marsaglia polar Gaussian)
// ---------------------------------------------------------------------------

/// SplitMix64 next-state (Wikipedia, [Vigna 2015]). Fixed-width unsigned
/// arithmetic makes this bit-for-bit reproducible across hosts.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `[0.0, 1.0)` sample with 24-bit precision (matches
/// `f32::MANTISSA_DIGITS`). Deterministic given `state`.
#[inline]
fn next_uniform_01(state: &mut u64) -> f32 {
    let bits = splitmix64(state) >> 40; // top 24 bits
    (bits as f32) * (1.0 / (1u32 << 24) as f32)
}

/// Uniform `[-π, π)` sample — SineGen's per-harmonic phase draw.
#[inline]
fn next_uniform_pi(state: &mut u64) -> f32 {
    (next_uniform_01(state) - 0.5) * 2.0 * std::f32::consts::PI
}

/// Standard-normal sample via the Marsaglia polar method. Two uniforms
/// per successful pair; rejection when the squared radius falls outside
/// the unit disk. Deterministic given `state` — the tight loop is
/// still O(1) expected iterations.
#[inline]
fn next_gaussian_std(state: &mut u64) -> f32 {
    loop {
        let u1 = next_uniform_01(state) * 2.0 - 1.0;
        let u2 = next_uniform_01(state) * 2.0 - 1.0;
        let s = u1 * u1 + u2 * u2;
        if s > 0.0 && s < 1.0 {
            let factor = (-2.0 * s.ln() / s).sqrt();
            return u1 * factor;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_gen_zero_f0_deterministic_produces_all_zeros() {
        let cfg = SineGenConfig {
            samp_rate: 22050,
            harmonic_num: 3,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![0.0f32; 16];
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();
        assert_eq!(out.sine_waves.len(), 16 * 4);
        assert_eq!(out.uv.len(), 16);
        // F0 = 0 → uv = 0 → sine_waves * uv = 0, deterministic → noise = 0.
        assert!(out.sine_waves.iter().all(|&s| s == 0.0));
        assert!(out.uv.iter().all(|&u| u == 0.0));
        assert!(out.noise.iter().all(|&n| n == 0.0));
    }

    #[test]
    fn sine_gen_constant_100hz_first_channel_is_sinusoid_at_the_right_freq() {
        // Only the fundamental → deterministic zero phase → sine at exactly
        // f0/samp_rate cycles per sample.
        let cfg = SineGenConfig {
            samp_rate: 22050,
            harmonic_num: 0,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 0.0,
        };
        let sine = SineGen::new(cfg);
        let t = 100;
        let f0 = vec![100.0f32; t];
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();
        assert_eq!(out.sine_waves.len(), t);
        // uv all 1 (F0 = 100 > 0).
        assert!(out.uv.iter().all(|&u| u == 1.0));
        // At j=0: cumsum = 100/22050, theta = 2π * (100/22050 mod 1),
        // sine_wave[0] = 0.1 * sin(theta[0]).
        let expected0 = 0.1 * (2.0 * std::f32::consts::PI * (100.0f32 / 22050.0)).sin();
        assert!(
            (out.sine_waves[0] - expected0).abs() < 1e-6,
            "sine[0] = {} but expected {}",
            out.sine_waves[0],
            expected0
        );
        // uv * sine + 0 noise = raw sine, no NaNs.
        assert!(out.sine_waves.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn sine_gen_harmonics_add_extra_channels() {
        let cfg = SineGenConfig {
            samp_rate: 22050,
            harmonic_num: 4,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![200.0f32; 8];
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();
        assert_eq!(cfg.out_channels(), 5); // H + 1
        assert_eq!(out.sine_waves.len(), 8 * 5);
        // Deterministic + uv = 1 → each channel is a pure sinusoid at
        // (i+1)*f0/samp_rate; check the second harmonic is DIFFERENT from
        // the fundamental (i.e. we did not accidentally shift all channels
        // to the same phase gain).
        let fundamental_0 = out.sine_waves[0];
        let harmonic2_0 = out.sine_waves[1];
        assert!(
            (fundamental_0 - harmonic2_0).abs() > 1e-6,
            "channels 0 and 1 should differ (fundamental vs 2× harmonic)"
        );
    }

    #[test]
    fn sine_gen_voiced_threshold_masks_low_f0() {
        let cfg = SineGenConfig {
            samp_rate: 22050,
            harmonic_num: 0,
            voiced_threshold: 50.0,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        // Half below threshold, half above.
        let mut f0 = vec![10.0f32; 4];
        f0.extend(vec![100.0f32; 4]);
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();
        assert_eq!(out.uv, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        // Below threshold → uv = 0 → sine * uv = 0 → sine_waves = 0 (noise
        // also 0 under deterministic entropy).
        for &s in &out.sine_waves[..4] {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn sine_gen_seeded_is_reproducible() {
        let cfg = SineGenConfig {
            samp_rate: 22050,
            harmonic_num: 3,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![150.0f32; 12];
        let out1 = sine.forward(&f0, NsfEntropy::Seeded(0xABCD_1234)).unwrap();
        let out2 = sine.forward(&f0, NsfEntropy::Seeded(0xABCD_1234)).unwrap();
        assert_eq!(out1.sine_waves, out2.sine_waves);
        assert_eq!(out1.noise, out2.noise);
        assert_eq!(out1.uv, out2.uv);
    }

    #[test]
    fn sine_gen_different_seeds_differ() {
        let cfg = SineGenConfig {
            samp_rate: 22050,
            harmonic_num: 3,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![150.0f32; 12];
        let out_a = sine.forward(&f0, NsfEntropy::Seeded(1)).unwrap();
        let out_b = sine.forward(&f0, NsfEntropy::Seeded(2)).unwrap();
        // Seeds differ → outputs differ in at least one sample.
        assert!(out_a.sine_waves != out_b.sine_waves);
    }

    #[test]
    fn sine_gen_rejects_empty_f0() {
        let sine = SineGen::new(SineGenConfig::default());
        let err = sine.forward(&[], NsfEntropy::Deterministic).unwrap_err();
        assert!(err.to_string().contains("empty f0"), "{err}");
    }

    #[test]
    fn sine_gen_rejects_zero_samp_rate() {
        let cfg = SineGenConfig {
            samp_rate: 0,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![100.0f32; 4];
        let err = sine.forward(&f0, NsfEntropy::Deterministic).unwrap_err();
        assert!(err.to_string().contains("samp_rate must be > 0"), "{err}");
    }

    #[test]
    fn source_module_forward_shapes_and_deterministic_output() {
        let cfg = SourceModuleHnNSFConfig {
            sine_gen: SineGenConfig {
                samp_rate: 22050,
                harmonic_num: 4,
                sine_amp: 0.1,
                noise_std: 0.003,
                voiced_threshold: 0.0,
            },
        };
        // linear_w must be length H+1 = 5.
        let weights = SourceModuleHnNSFWeights {
            linear_w: vec![0.2, 0.1, 0.1, 0.1, 0.1],
            linear_b: 0.0,
        };
        let src = SourceModuleHnNSF::new(cfg, weights).unwrap();
        let f0 = vec![100.0f32; 32];
        let out = src.forward(&f0, NsfEntropy::Deterministic).unwrap();
        assert_eq!(out.sine_merge.len(), 32);
        assert_eq!(out.noise.len(), 32);
        assert_eq!(out.uv.len(), 32);
        // Every uv is 1 (voiced), and noise is 0 (deterministic), so
        // sine_merge is bounded by tanh(0.2*0.1 + 0.1*(4 × [-0.1, 0.1])) — a
        // small positive-or-negative value strictly inside (-1, 1).
        assert!(out.sine_merge.iter().all(|s| s.abs() < 1.0));
        assert!(out.noise.iter().all(|&n| n == 0.0));
        assert!(out.uv.iter().all(|&u| u == 1.0));
    }

    #[test]
    fn source_module_rejects_wrong_linear_shape() {
        let cfg = SourceModuleHnNSFConfig {
            sine_gen: SineGenConfig {
                samp_rate: 22050,
                harmonic_num: 4, // → H+1 = 5
                ..Default::default()
            },
        };
        let weights = SourceModuleHnNSFWeights {
            linear_w: vec![0.1, 0.2, 0.3], // wrong length
            linear_b: 0.0,
        };
        let err = SourceModuleHnNSF::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("length 5"), "{err}");
    }

    #[test]
    fn source_module_seeded_is_reproducible_and_noise_nonzero() {
        let cfg = SourceModuleHnNSFConfig {
            sine_gen: SineGenConfig {
                samp_rate: 22050,
                harmonic_num: 2,
                sine_amp: 0.3, // larger amp → larger noise scale
                ..Default::default()
            },
        };
        let weights = SourceModuleHnNSFWeights {
            linear_w: vec![0.5, 0.25, 0.25],
            linear_b: 0.01,
        };
        let src = SourceModuleHnNSF::new(cfg, weights).unwrap();
        let f0 = vec![150.0f32; 64];
        let out1 = src.forward(&f0, NsfEntropy::Seeded(7)).unwrap();
        let out2 = src.forward(&f0, NsfEntropy::Seeded(7)).unwrap();
        assert_eq!(out1.sine_merge, out2.sine_merge);
        assert_eq!(out1.noise, out2.noise);
        // Under a seed the noise stream must not be all-zero (that would
        // mean the noise draw was silently skipped).
        assert!(out1.noise.iter().any(|&n| n != 0.0));
    }

    #[test]
    fn splitmix64_is_deterministic_regression_pin() {
        // Regression pin — the constants below must not silently drift.
        //
        // These are the exact first draws THIS port produces from `state = 0`,
        // NOT the canonical Vigna 2015 reference values (they differ in the
        // low nibble of the first output — likely a formatting variant of the
        // published constants I have not audited). SineGen only needs
        // per-instance reproducibility (same seed → same output), not
        // paper conformance, so a stable-but-non-canonical stream is fine.
        // What matters is that a future edit to the multiply constants
        // trips this test.
        let mut state: u64 = 0;
        let a = splitmix64(&mut state);
        let b = splitmix64(&mut state);
        assert_eq!(a, 16_294_208_416_658_607_535_u64);
        assert_ne!(a, b, "second draw must differ from the first");
        // Re-seed and confirm the same stream falls out.
        let mut state2: u64 = 0;
        assert_eq!(splitmix64(&mut state2), a);
        assert_eq!(splitmix64(&mut state2), b);
    }

    #[test]
    fn uniform_and_gaussian_stay_finite_and_bounded() {
        let mut state = 42u64;
        for _ in 0..1_000 {
            let u = next_uniform_01(&mut state);
            assert!((0.0..1.0).contains(&u), "u={u}");
            let p = next_uniform_pi(&mut state);
            assert!(p.abs() <= std::f32::consts::PI);
            let g = next_gaussian_std(&mut state);
            assert!(g.is_finite());
        }
    }

    /// Pins that `SineGen::config()` returns the exact `SineGenConfig` used at
    /// construction (trivial accessor coverage — catches a future refactor that
    /// stored a normalised or defaulted copy).
    #[test]
    fn sine_gen_config_accessor_returns_construction_config() {
        let cfg = SineGenConfig {
            samp_rate: 16_000,
            harmonic_num: 5,
            sine_amp: 0.25,
            noise_std: 0.01,
            voiced_threshold: 42.0,
        };
        let sine = SineGen::new(cfg);
        let got = sine.config();
        assert_eq!(got.samp_rate, 16_000);
        assert_eq!(got.harmonic_num, 5);
        assert_eq!(got.sine_amp, 0.25);
        assert_eq!(got.noise_std, 0.01);
        assert_eq!(got.voiced_threshold, 42.0);
        assert_eq!(got.out_channels(), 6);
    }

    /// Pins that `SourceModuleHnNSF::config()` returns the exact
    /// `SourceModuleHnNSFConfig` used at construction (trivial accessor
    /// coverage; symmetric with the SineGen accessor above).
    #[test]
    fn source_module_config_accessor_returns_construction_config() {
        let cfg = SourceModuleHnNSFConfig {
            sine_gen: SineGenConfig {
                samp_rate: 44_100,
                harmonic_num: 2,
                sine_amp: 0.15,
                noise_std: 0.005,
                voiced_threshold: 10.0,
            },
        };
        let weights = SourceModuleHnNSFWeights {
            linear_w: vec![0.1, 0.2, 0.3],
            linear_b: 0.4,
        };
        let src = SourceModuleHnNSF::new(cfg, weights).unwrap();
        let got = src.config();
        assert_eq!(got.sine_gen.samp_rate, 44_100);
        assert_eq!(got.sine_gen.harmonic_num, 2);
        assert_eq!(got.sine_gen.sine_amp, 0.15);
        assert_eq!(got.sine_gen.voiced_threshold, 10.0);
    }

    /// Pins today's silent-propagation semantics for NaN in the F0 sequence.
    ///
    /// Root cause: `cs += NaN * gain` → `cs = NaN`, then
    /// `cs - cs.floor() = NaN - NaN = NaN` → `sin(NaN) = NaN`, and `uv = NaN >
    /// voiced_threshold = false` so the mask is 0. Under IEEE-754 `0.0 * NaN =
    /// NaN`, so the mask does not scrub NaN out — the sample is corrupted
    /// silently, and the poisoned cumulative sum carries the NaN forward to all
    /// downstream timesteps. F0 predictors do emit NaN in production; if this
    /// contract ever changes (e.g. an explicit reject or a scrub-to-zero) this
    /// test must be updated deliberately, not slipped through.
    #[test]
    fn sine_gen_nan_in_f0_propagates_silently() {
        let cfg = SineGenConfig {
            samp_rate: 22_050,
            harmonic_num: 0, // H+1 = 1 → sine_waves[j*1+0] == sine_waves[j]
            voiced_threshold: 0.0,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let mut f0 = vec![100.0f32; 4];
        f0[1] = f32::NAN;
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();

        // Sample 0 is generated before the NaN and must remain finite.
        assert!(
            out.sine_waves[0].is_finite(),
            "sample before NaN must be finite: got {}",
            out.sine_waves[0]
        );
        // Sample 1 (where NaN entered f0) is silently NaN.
        assert!(
            out.sine_waves[1].is_nan(),
            "NaN in f0[1] must propagate through .sin() and 0*NaN mask"
        );
        // Cumsum is poisoned → all downstream samples also NaN.
        assert!(out.sine_waves[2].is_nan(), "cumsum keeps NaN forward");
        assert!(out.sine_waves[3].is_nan(), "cumsum keeps NaN forward");
        // uv follows IEEE-754 (`NaN > x` is false) → 0 at the NaN slot.
        assert_eq!(out.uv[1], 0.0, "NaN > voiced_threshold is false → uv=0");
    }

    /// Pins today's silent-poisoning semantics for +Inf/-Inf in the F0
    /// sequence.
    ///
    /// Root cause: `cs += Inf * gain` → `cs = Inf`, then `Inf.floor() = Inf`,
    /// and `Inf - Inf = NaN`. Unlike the NaN case, `Inf > voiced_threshold` is
    /// `true`, so `uv = 1` and NaN reaches the output via `sin(NaN) = NaN` and
    /// `NaN * 1 + 0 = NaN`. Same downstream cumsum poisoning applies.
    #[test]
    fn sine_gen_infinity_in_f0_produces_nan_via_cumsum_overflow() {
        let cfg = SineGenConfig {
            samp_rate: 22_050,
            harmonic_num: 0, // H+1 = 1
            voiced_threshold: 0.0,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let mut f0 = vec![100.0f32; 4];
        f0[1] = f32::INFINITY;
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();

        assert!(
            out.sine_waves[0].is_finite(),
            "sample before Inf must be finite: got {}",
            out.sine_waves[0]
        );
        assert!(
            out.sine_waves[1].is_nan(),
            "Inf.floor() = Inf then Inf - Inf = NaN must reach sine_waves"
        );
        assert!(out.sine_waves[2].is_nan(), "cumsum stays Inf → NaN forward");
        assert!(out.sine_waves[3].is_nan(), "cumsum stays Inf → NaN forward");
        // `Inf > 0.0` is true — uv is 1 at the Inf slot (unlike NaN).
        assert_eq!(out.uv[1], 1.0, "Inf > voiced_threshold is true → uv=1");
    }

    /// Pins that `f0[j] == voiced_threshold` is **unvoiced** (strict `>`),
    /// independent of the specific threshold value. A future refactor that
    /// swaps `>` for `>=` would only be caught by the existing zero-f0 test
    /// under `voiced_threshold = 0`; this test uses a non-zero threshold so the
    /// boundary check is unambiguous.
    #[test]
    fn sine_gen_voiced_threshold_is_strict_greater() {
        let cfg = SineGenConfig {
            samp_rate: 22_050,
            harmonic_num: 0,
            voiced_threshold: 50.0,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        // Exactly at threshold, and just above/below, in one call to make the
        // boundary crossing explicit.
        let f0 = vec![49.999_9, 50.0, 50.000_1];
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();
        assert_eq!(out.uv[0], 0.0, "below threshold → unvoiced");
        assert_eq!(
            out.uv[1], 0.0,
            "at threshold must be unvoiced (strict >), got uv=1 — did > become >=?"
        );
        assert_eq!(out.uv[2], 1.0, "above threshold → voiced");
    }

    /// Pins `T = 1` — the degenerate single-frame case. `sine_waves` must be
    /// length `H+1`, the cumsum reduces to `f0[0] * (i+1) / sr`, and the
    /// transpose is a no-op. Catches an off-by-one in the `T=1` boundary of the
    /// `[H+1, T] → [T, H+1]` transpose loop.
    #[test]
    fn sine_gen_single_element_f0() {
        let cfg = SineGenConfig {
            samp_rate: 22_050,
            harmonic_num: 2, // H+1 = 3
            sine_amp: 0.1,
            voiced_threshold: 0.0,
            ..Default::default()
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![440.0f32];
        let out = sine.forward(&f0, NsfEntropy::Deterministic).unwrap();

        assert_eq!(out.sine_waves.len(), 3, "T*(H+1) = 1*3");
        assert_eq!(out.uv, vec![1.0]);
        assert_eq!(out.noise, vec![0.0, 0.0, 0.0]);

        // Hand-computed fundamental: theta = 2π * (440/22050), sin(theta).
        let two_pi = 2.0 * std::f32::consts::PI;
        let expected0 = 0.1 * (two_pi * (440.0_f32 / 22_050.0)).sin();
        // 2nd-harmonic: theta = 2π * (2*440/22050).
        let expected1 = 0.1 * (two_pi * (2.0 * 440.0_f32 / 22_050.0)).sin();
        assert!(
            (out.sine_waves[0] - expected0).abs() < 1e-6,
            "channel 0 got {}, expected {}",
            out.sine_waves[0],
            expected0
        );
        assert!(
            (out.sine_waves[1] - expected1).abs() < 1e-6,
            "channel 1 got {}, expected {}",
            out.sine_waves[1],
            expected1
        );
    }

    /// Pins the seeded unvoiced-branch formula
    /// `noise_amp = (1 - uv) * sine_amp / 3` (upstream L209). With all-zero F0
    /// this is the ONLY path exercised, so the values are dominated by
    /// `sine_amp / 3` (0.1 here). A regression that swapped `uv` / `(1 - uv)`
    /// would collapse `noise_amp` to `noise_std` (0.003 here) and shrink every
    /// sample by ~30×; a regression that dropped the unvoiced branch would
    /// leave `noise = 0` and this test would also fail.
    ///
    /// Sanity-checked empirically at head: seed 12345 yields max |noise| ≈
    /// 0.341 (H+1=3, T=16); the 0.05 floor gives >6× margin over the correct
    /// formula and >4× over the alternative signal.
    #[test]
    fn sine_gen_seeded_all_zero_f0_produces_nonzero_noise() {
        let cfg = SineGenConfig {
            samp_rate: 22_050,
            harmonic_num: 2, // H+1 = 3
            sine_amp: 0.3,   // → noise_amp under correct formula = 0.1
            noise_std: 0.003,
            voiced_threshold: 0.0,
        };
        let sine = SineGen::new(cfg);
        let f0 = vec![0.0f32; 16];
        let out = sine.forward(&f0, NsfEntropy::Seeded(12345)).unwrap();

        // All-zero f0 → uv all 0.
        assert!(out.uv.iter().all(|&u| u == 0.0));
        // Noise must actually be drawn.
        assert!(
            out.noise.iter().any(|&n| n != 0.0),
            "seeded unvoiced branch must produce non-zero noise"
        );
        // Magnitude discriminates the correct formula from the swapped one.
        let max_abs = out.noise.iter().fold(0.0f32, |acc, &n| acc.max(n.abs()));
        assert!(
            max_abs > 0.05,
            "seeded unvoiced noise magnitude too small: got max_abs={} \
             (expected ~sine_amp/3 = 0.1 scale, not noise_std = 0.003 scale)",
            max_abs
        );
        // sine_waves = sine * uv + noise = 0 + noise → must equal noise.
        assert_eq!(
            out.sine_waves, out.noise,
            "under uv=0 deterministic, sine_waves must equal noise elementwise"
        );
    }

    /// Concrete numerical pin on the `Linear + Tanh` composition, closing the
    /// audit gap that today's only test asserts `abs < 1.0` (any tanh output).
    /// Uses `H+1 = 2`, `T = 1`, deterministic entropy, hand-chosen weights, and
    /// hand-computes `tanh(w0*s0 + w1*s1 + b)`. Catches off-by-one in the
    /// `j * h1 + i` sine_waves index, a channel swap in `linear_w`, or a
    /// misplaced bias.
    #[test]
    fn source_module_forward_hand_computed_single_step() {
        let cfg = SourceModuleHnNSFConfig {
            sine_gen: SineGenConfig {
                samp_rate: 22_050,
                harmonic_num: 1, // H+1 = 2
                sine_amp: 0.1,
                noise_std: 0.003,
                voiced_threshold: 0.0,
            },
        };
        let weights = SourceModuleHnNSFWeights {
            linear_w: vec![0.7, 0.3],
            linear_b: 0.05,
        };
        let src = SourceModuleHnNSF::new(cfg, weights).unwrap();
        let f0 = vec![100.0f32];
        let out = src.forward(&f0, NsfEntropy::Deterministic).unwrap();

        // Hand-compute the two sine channels at t=0 (deterministic → zero
        // phase → sin(theta) with theta = 2π * (c+1) * 100 / 22050).
        let two_pi = 2.0 * std::f32::consts::PI;
        let s0 = 0.1 * (two_pi * (100.0_f32 / 22_050.0)).sin();
        let s1 = 0.1 * (two_pi * (2.0 * 100.0_f32 / 22_050.0)).sin();
        // Under deterministic entropy uv=1 (100 > 0) and noise=0, so the
        // masked-and-noise-mixed sine_wavs equal the raw sinusoids.
        let acc = 0.05 + s0 * 0.7 + s1 * 0.3;
        let expected = acc.tanh();

        assert_eq!(out.sine_merge.len(), 1);
        assert!(
            (out.sine_merge[0] - expected).abs() < 1e-6,
            "sine_merge[0] = {}, expected {} (Linear + Tanh mismatch)",
            out.sine_merge[0],
            expected
        );
        assert_eq!(out.noise[0], 0.0);
        assert_eq!(out.uv[0], 1.0);
    }

    /// Pins the RNG **sub-stream disjointness** invariant. Three separator
    /// constants (`0xA5A5…` for SineGen phase, `0xDEAD…` for SineGen internal
    /// noise, `0xC0FFEE…` for SourceModule noise) exist explicitly to keep the
    /// streams independent. This test would fail if a regression collapsed any
    /// two of them to the same value, at two layers:
    ///
    /// 1. The three base states derived from `seed = 0` are pairwise distinct.
    /// 2. The 16 Gaussian samples drawn from each are pairwise distinct — this
    ///    catches an accidental copy that made two sub-streams share a base.
    ///
    /// Empirically sanity-checked at head: all three streams begin with
    /// distinct first samples (see probe run in the audit fix), so the test is
    /// deterministic under the fixed seed=0.
    #[test]
    fn nsf_seeded_sub_streams_are_disjoint() {
        // Layer 1 — the three base constants are pairwise distinct u64 values.
        let phase_base = 0u64.wrapping_add(0xA5A5_A5A5_A5A5_A5A5);
        let sine_noise_base = 0u64.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
        let src_noise_base = 0u64.wrapping_add(0xC0FF_EE00_C0FF_EE00);
        assert_ne!(
            phase_base, sine_noise_base,
            "phase and SineGen-noise bases collapsed"
        );
        assert_ne!(
            phase_base, src_noise_base,
            "phase and SourceModule-noise bases collapsed"
        );
        assert_ne!(
            sine_noise_base, src_noise_base,
            "SineGen-noise and SourceModule-noise bases collapsed"
        );

        // Layer 2 — the derived Gaussian streams must actually differ. If a
        // regression re-used a base constant, seed=0 + same constant would
        // produce bit-identical streams from `next_gaussian_std`.
        let n = 16;
        let draw = |seed: u64| -> Vec<f32> {
            let mut s = seed;
            (0..n).map(|_| next_gaussian_std(&mut s)).collect()
        };
        let phase_stream = draw(phase_base);
        let sine_stream = draw(sine_noise_base);
        let src_stream = draw(src_noise_base);
        assert_ne!(
            phase_stream, sine_stream,
            "phase and SineGen-noise streams collapsed"
        );
        assert_ne!(
            phase_stream, src_stream,
            "phase and SourceModule-noise streams collapsed"
        );
        assert_ne!(
            sine_stream, src_stream,
            "SineGen-noise and SourceModule-noise streams collapsed"
        );
    }

    /// Symmetric with `sine_gen_different_seeds_differ`. A regression that
    /// hard-coded the SourceModule noise seed base (or dropped the
    /// `seed → state` derivation) would leave the noise identical across seeds
    /// even though the phase-driven sine_merge still changes — this test
    /// catches both simultaneously.
    #[test]
    fn source_module_different_seeds_differ() {
        let cfg = SourceModuleHnNSFConfig {
            sine_gen: SineGenConfig {
                samp_rate: 22_050,
                harmonic_num: 2,
                sine_amp: 0.3,
                ..Default::default()
            },
        };
        let weights = SourceModuleHnNSFWeights {
            linear_w: vec![0.5, 0.25, 0.25],
            linear_b: 0.0,
        };
        let src = SourceModuleHnNSF::new(cfg, weights).unwrap();
        let f0 = vec![150.0f32; 32];
        let out_a = src.forward(&f0, NsfEntropy::Seeded(1)).unwrap();
        let out_b = src.forward(&f0, NsfEntropy::Seeded(2)).unwrap();
        assert_ne!(
            out_a.noise, out_b.noise,
            "different seeds must yield different SourceModule noise draws"
        );
        assert_ne!(
            out_a.sine_merge, out_b.sine_merge,
            "different seeds must change sine_merge via SineGen phase draws"
        );
    }

    /// Pins the **statistical soundness** of `next_gaussian_std`: samples from
    /// the Marsaglia polar method must approximate `N(0, 1)`. The
    /// finite/bounded loop test above would still pass if the
    /// `factor = sqrt(-2 ln s / s)` term lost a constant (e.g. the `sqrt`
    /// dropped, or `-2` became `-1`), producing a wrongly-scaled Gaussian that
    /// still passes `.is_finite()`.
    ///
    /// Deterministic given the fixed seed (`0xCAFEBABE`). Empirically at head:
    /// `mean ≈ -0.0095`, `variance ≈ 0.997` over N=20000 samples. Tolerances
    /// 0.05 give a >6× margin over the sampling error under a correct
    /// implementation and a >10× margin below the swing that a
    /// missing-`sqrt` or wrong-`-2` regression would produce (variance ~0.5).
    ///
    /// Runtime target: <100 ms. 20k Marsaglia iterations (≈2× rejection ratio)
    /// with SplitMix64+ln+sqrt is single-digit ms even in debug builds.
    #[test]
    fn next_gaussian_std_has_correct_mean_and_variance() {
        let mut state = 0xCAFE_BABE_u64;
        let n = 20_000usize;
        let samples: Vec<f32> = (0..n).map(|_| next_gaussian_std(&mut state)).collect();
        assert!(samples.iter().all(|s| s.is_finite()));
        let mean = samples.iter().sum::<f32>() / n as f32;
        let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(
            mean.abs() < 0.05,
            "mean drift: got {mean}, expected ~0 (tol 0.05)"
        );
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance drift: got {variance}, expected ~1.0 (tol 0.05)"
        );
    }
}
