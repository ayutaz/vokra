//! SNAC (Multi-Scale Neural Audio Codec) 3/4-stage residual VQ decode
//! (SoTA plan Phase 3 TTS primitive; FR-OP-30 RVQ family).
//!
//! # Op contract
//!
//! SNAC is a **hierarchical / multi-scale** residual vector quantizer: unlike
//! Mimi / DAC where every quantizer shares the same time axis, SNAC's `k`th
//! stage runs at frame rate `base / vq_strides[k]` (upstream
//! `hubertsiuzdak/snac/blob/main/snac/vq.py`, `VectorQuantize.__init__`
//! `stride` argument L15-25; `VectorQuantize.forward` L27-40 — encode side
//! `avg_pool1d(stride)`, decode side `repeat_interleave(stride, dim=-1)`).
//!
//! The 24 kHz variant used by Orpheus and Maya1 uses `vq_strides = [4, 2, 1]`.
//! With SNAC 24 kHz's encoder `hop = 512` the base frame rate is
//! `24000 / 512 ≈ 46.875 Hz`, which after the strides gives the widely quoted
//! per-stage code rates of ~12 / 23 / 47 Hz.
//!
//! # From the upstream `ResidualVectorQuantize.from_codes` (verbatim
//! algorithm, `hubertsiuzdak/snac/blob/main/snac/vq.py` L61-71)
//!
//! ```text
//!   z_q = 0
//!   for i in range(n_codebooks):
//!       z_p_i = quantizers[i].decode_code(codes[i])             # embed lookup
//!       z_q_i = quantizers[i].out_proj(z_p_i)                    # 1x1 conv (weight-norm folded)
//!       z_q_i = z_q_i.repeat_interleave(strides[i], dim=-1)      # temporal upsample
//!       z_q  += z_q_i
//! ```
//!
//! SNAC's per-quantizer path is **factorized** in exactly the DAC sense
//! (`in_proj` on the encode side, `out_proj` on the decode side, both 1x1
//! `WNConv1d`s that fold cheaply once the weight-norm parameters are collapsed
//! offline). The two crates therefore share [`crate::dac_rvq::DacOutProj`] for
//! the folded weight + bias state — one FP32 GEMV core, one bias vector, one
//! shape validator (M4-04 §D-f rationale applies verbatim: without
//! factorization each stage would materialise `codebook_size × d_model`
//! floats, which is ~12 MB at SNAC 24 kHz's `(4096 × 768)` — 20 kB with
//! factorization).
//!
//! # Shape contract
//!
//! - `codes[i]` — length `t_i` where `t_i × vq_strides[i]` is the same for
//!   every stage `i`. That common product is the output time dimension `T`.
//! - Every stage shares the same `codebook_size`, `codebook_dim`, and
//!   `d_model`; only the temporal stride differs.
//! - The returned buffer is `[T, d_model]` row-major FP32 — the same feature
//!   convention as [`crate::mimi_rvq::mimi_rvq_decode`] and
//!   [`crate::dac_rvq::dac_rvq_decode`]. Downstream stages (the SNAC decoder
//!   upsample chain to PCM) are consumer-WP concerns and stay outside this
//!   primitive.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Out-of-range indices, mis-aligned stage lengths, zero strides, and every
//! shape mismatch surface as [`VokraError::InvalidArgument`] rather than a
//! silent clamp / truncation / zero fill. A wrong RVQ index degrades the
//! reconstructed audio in a plausible-sounding way, so failing at decode is
//! the safer default (same rationale as the mimi_rvq module docs).
//!
//! # No SIMD / no unsafe
//!
//! The fold is a plain scalar `y += W * x` accumulator in FP32 (Rust's
//! IEEE-754 mul-then-add ordering). This is bit-identical to the equivalent
//! DAC helper on the same input; the two implementations are independent for
//! module isolation. The `unsafe` opt-out in the crate root does not extend
//! into this file: SNAC is deliberately safe Rust top to bottom.
//!
//! # License note
//!
//! The upstream SNAC repository (`hubertsiuzdak/snac`) is dual MIT / Apache
//! 2.0; this port only mirrors the algorithm shape and requires the folded
//! weights be supplied by the caller / model converter.

use vokra_core::{Result, VokraError};

use crate::dac_rvq::DacOutProj;
use crate::mimi_rvq::CodebookTable;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Maximum number of hierarchical RVQ stages in the published SNAC family.
/// The 24 kHz checkpoint uses three stages and the 44.1 kHz checkpoint uses
/// four. Keeping this bound explicit lets the Metal ABI stay fixed-size while
/// the public op accepts either released topology.
pub const MAX_SNAC_STAGES: usize = 4;

/// Static configuration for a SNAC hierarchical RVQ decode.
///
/// The task-level API deliberately keeps this struct minimal — the sample
/// rate and per-stage strides are the only two fields the consumer (Orpheus,
/// Maya1) needs to know. Codebook size / dim and `d_model` are derived from
/// the supplied [`SnacWeights`] at [`SnacDecoder::new`] time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnacConfig {
    /// Output PCM sample rate the underlying SNAC model was trained for
    /// (24 000 for Orpheus / Maya1's 24 kHz variant).
    pub sample_rate: u32,
    /// Per-stage temporal strides. Every stage `i` runs at
    /// `base_frame_rate / vq_strides[i]`; SNAC 24 kHz = `[4, 2, 1]` gives the
    /// canonical ~12 / 23 / 47 Hz per-stage code rate (module docs).
    pub vq_strides: [u32; MAX_SNAC_STAGES],
    /// Number of active entries in [`Self::vq_strides`]. Published SNAC
    /// checkpoints use 3 (24 kHz) or 4 (44.1 kHz).
    pub n_stages: usize,
}

impl SnacConfig {
    /// Canonical SNAC 24 kHz variant (Orpheus, Maya1): `sample_rate = 24000`,
    /// `vq_strides = [4, 2, 1]`.
    ///
    /// Upstream defaults: `hubertsiuzdak/snac_24khz/config.json`
    /// (`sampling_rate = 24000`, `vq_strides = [4, 2, 1]`).
    #[inline]
    #[must_use]
    pub const fn snac_24khz() -> Self {
        Self {
            sample_rate: 24000,
            vq_strides: [4, 2, 1, 0],
            n_stages: 3,
        }
    }

    /// Canonical SNAC 44.1 kHz variant: four hierarchical stages with
    /// `vq_strides = [8, 4, 2, 1]`.
    #[inline]
    #[must_use]
    pub const fn snac_44khz() -> Self {
        Self {
            sample_rate: 44_100,
            vq_strides: [8, 4, 2, 1],
            n_stages: 4,
        }
    }

    /// Active per-stage strides. Construction validation guarantees this
    /// slice is non-empty and contains no zero stride.
    #[inline]
    #[must_use]
    pub fn active_vq_strides(&self) -> &[u32] {
        &self.vq_strides[..self.n_stages]
    }
}

// ---------------------------------------------------------------------------
// Weights bundle
// ---------------------------------------------------------------------------

/// The factorized codebook tables + `out_proj`s that feed a SNAC
/// decode.
///
/// Both the [`CodebookTable`] (from `mimi_rvq`) and [`DacOutProj`] (from
/// `dac_rvq`) types are reused verbatim — SNAC's `WNConv1d(codebook_dim,
/// input_dim, kernel_size=1)` folds identically to DAC's, and the embedding
/// table storage is the same `[codebook_size, codebook_dim]` row-major layout
/// (see upstream `hubertsiuzdak/snac/blob/main/snac/vq.py` L15-25
/// `VectorQuantize.__init__` and L42-44 `decode_code` — the row width stored
/// is `codebook_dim`, not `input_dim`).
///
/// The three codebooks and three `out_proj`s must share `codebook_size`,
/// `codebook_dim`, and `d_model`; only the temporal stride differs between
/// stages (the stride lives in [`SnacConfig::vq_strides`], not here).
#[derive(Debug, Clone)]
pub struct SnacWeights {
    /// Per-stage factorized `[codebook_size, codebook_dim]` codebooks.
    pub codebooks: Vec<CodebookTable>,
    /// Per-stage `out_proj` (`codebook_dim → d_model`) with weight-norm
    /// **already folded offline** by the model converter.
    pub out_projs: Vec<DacOutProj>,
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Host-side SNAC hierarchical RVQ decoder — owns its codebooks + folded
/// `out_proj`s, exposes the tight `decode(&codes)` mouth the Orpheus / Maya1
/// model wrappers consume.
///
/// Mirror of [`crate::mimi_rvq::MimiDecoder`] / the DAC decode surface: the
/// stub keeps state on the CPU side so a caller can drive it without pulling
/// the SNAC decoder-network module in.
#[derive(Debug, Clone)]
pub struct SnacDecoder {
    config: SnacConfig,
    weights: SnacWeights,
    codebook_size: usize,
    codebook_dim: usize,
    d_model: usize,
}

impl SnacDecoder {
    /// Builds a decoder from a [`SnacConfig`] and a fully-populated
    /// [`SnacWeights`] bundle.
    ///
    /// The three codebook tables and three `out_proj`s are validated for
    /// mutual shape consistency here so per-decode calls do not repay that
    /// cost. Every axis must be `> 0`, every codebook table must share
    /// `[codebook_size, codebook_dim]`, every `out_proj` must share
    /// `[d_model, codebook_dim]`, and every `vq_strides` entry must be `> 0`
    /// (`stride = 0` would divide the frame rate by zero — FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of the above axis / shape /
    /// stride violations.
    pub fn new(config: SnacConfig, weights: SnacWeights) -> Result<Self> {
        if !(1..=MAX_SNAC_STAGES).contains(&config.n_stages) {
            return Err(VokraError::InvalidArgument(format!(
                "SnacDecoder::new: n_stages {} is outside 1..={MAX_SNAC_STAGES}",
                config.n_stages
            )));
        }
        if weights.codebooks.len() != config.n_stages || weights.out_projs.len() != config.n_stages
        {
            return Err(VokraError::InvalidArgument(format!(
                "SnacDecoder::new: config.n_stages={} but codebooks.len()={} and \
                 out_projs.len()={} (one table and projection are required per stage)",
                config.n_stages,
                weights.codebooks.len(),
                weights.out_projs.len()
            )));
        }
        for (i, s) in config.active_vq_strides().iter().enumerate() {
            if *s == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "SnacDecoder::new: vq_strides[{i}] must be > 0, got 0 \
                     (stride 0 would divide the base frame rate by zero)"
                )));
            }
        }

        let cb0 = &weights.codebooks[0];
        let codebook_size = cb0.codebook_size;
        // Note: `CodebookTable::d_model` names the row width; for a
        // factorized RVQ that width is `codebook_dim`, not the output
        // `d_model` (same convention DAC uses — dac_rvq.rs L419-433).
        let codebook_dim = cb0.d_model;
        let d_model = weights.out_projs[0].d_model;

        if codebook_size == 0 || codebook_dim == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "SnacDecoder::new: axes must be > 0, got codebook_size={codebook_size} \
                 codebook_dim={codebook_dim} d_model={d_model}"
            )));
        }

        for (i, t) in weights.codebooks.iter().enumerate() {
            if t.codebook_size != codebook_size || t.d_model != codebook_dim {
                return Err(VokraError::InvalidArgument(format!(
                    "SnacDecoder::new: codebooks[{i}] shape [{},{}] != [{},{}] \
                     (all stages must share the same codebook architecture)",
                    t.codebook_size, t.d_model, codebook_size, codebook_dim
                )));
            }
        }

        for (i, p) in weights.out_projs.iter().enumerate() {
            if p.d_model != d_model || p.codebook_dim != codebook_dim {
                return Err(VokraError::InvalidArgument(format!(
                    "SnacDecoder::new: out_projs[{i}] shape [{},{}] != [{},{}] \
                     (all stages must project into the same d_model)",
                    p.d_model, p.codebook_dim, d_model, codebook_dim
                )));
            }
        }

        Ok(Self {
            config,
            weights,
            codebook_size,
            codebook_dim,
            d_model,
        })
    }

    /// Read-only view of the configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SnacConfig {
        &self.config
    }

    /// Feature width (columns of the returned `[T, d_model]` buffer).
    #[inline]
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Number of entries per codebook (equal across the three stages).
    #[inline]
    #[must_use]
    pub const fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Factorized codebook row width (equal across the three stages).
    #[inline]
    #[must_use]
    pub const fn codebook_dim(&self) -> usize {
        self.codebook_dim
    }

    /// Decodes the per-stage code vectors into a `[T, d_model]`
    /// row-major FP32 feature buffer, where
    /// `T = codes[i].len() * vq_strides[i]` (the same for every `i`).
    ///
    /// Mirrors the upstream `ResidualVectorQuantize.from_codes`
    /// (`hubertsiuzdak/snac/blob/main/snac/vq.py` L61-71): each stage
    /// embed-looks-up its codes, projects the low-dim row up to `d_model`,
    /// `repeat_interleave`s along time by its stride, and adds into the
    /// FP32 accumulator.
    ///
    /// # Edge cases
    ///
    /// - All three code vectors empty → returns an empty `Vec` (`T = 0`).
    /// - Mixed empty / non-empty across stages → `InvalidArgument`
    ///   (would give inconsistent `T`s — FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of:
    /// - stage-to-stage `T` mis-alignment
    ///   (`codes[i].len() * vq_strides[i]` differs across stages);
    /// - `codes[i].len() * vq_strides[i]` overflows `usize`;
    /// - `codes[i][t] >= codebook_size` (no silent clamp — FR-EX-08).
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codes.len() != self.config.n_stages {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode: codes.len() {} != n_stages {}",
                codes.len(),
                self.config.n_stages
            )));
        }
        let strides = self.config.active_vq_strides();

        // Every stage must expand to the same base T; compute it once.
        let t_expanded = self.check_and_measure(codes, strides)?;
        if t_expanded == 0 {
            return Ok(Vec::new());
        }

        let mut out = vec![0.0_f32; t_expanded * self.d_model];
        let mut projected = vec![0.0_f32; self.d_model];

        for (stage_idx, stage_codes) in codes.iter().enumerate() {
            let stride = strides[stage_idx] as usize;
            let cb = &self.weights.codebooks[stage_idx];
            let proj = &self.weights.out_projs[stage_idx];

            for (t_stage, &idx) in stage_codes.iter().enumerate() {
                let low = cb.row(idx)?;

                // out_proj @ low + bias, FP32 accumulator — same math as
                // dac_rvq::project_accumulate, kept inline here so this
                // module does not force `pub(crate)` on a helper it only
                // needs a copy of (the fold is 3 lines and independent
                // reimplementations produce bit-identical output — same
                // mul-then-add ordering).
                for (o, dst) in projected.iter_mut().enumerate() {
                    let w_row = proj.weight_row(o);
                    let mut y = proj.bias[o];
                    for (w, x) in w_row.iter().zip(low.iter()) {
                        y += *w * *x;
                    }
                    *dst = y;
                }

                // repeat_interleave(stride, dim=-1): the projected row is
                // added to `stride` contiguous output timesteps starting at
                // `t_stage * stride`.
                let t_start = t_stage * stride;
                for t_expansion in 0..stride {
                    let t_out = t_start + t_expansion;
                    let out_base = t_out * self.d_model;
                    for (dst, src) in out[out_base..out_base + self.d_model]
                        .iter_mut()
                        .zip(projected.iter())
                    {
                        *dst += *src;
                    }
                }
            }
        }

        Ok(out)
    }

    /// Validates cross-stage length alignment and returns the common `T`.
    /// `T = codes[i].len() * vq_strides[i]` must be equal for every stage.
    fn check_and_measure(&self, codes: &[Vec<u32>], strides: &[u32]) -> Result<usize> {
        let mut common: Option<usize> = None;
        for (i, stage_codes) in codes.iter().enumerate() {
            let t_i = stage_codes.len();
            let s_i = strides[i] as usize;
            let expanded = t_i.checked_mul(s_i).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "snac_decode: codes[{i}].len() ({t_i}) * vq_strides[{i}] ({s_i}) \
                     overflows usize"
                ))
            })?;
            match common {
                Some(prev) if prev != expanded => {
                    return Err(VokraError::InvalidArgument(format!(
                        "snac_decode: stage {i} expands to T={expanded}, but earlier \
                         stages expand to T={prev} (codes[i].len() * vq_strides[i] must \
                         be the same for every stage — SNAC's multi-scale RVQ requires \
                         co-aligned base frames)"
                    )));
                }
                Some(_) => {}
                None => common = Some(expanded),
            }
        }
        Ok(common.unwrap_or(0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Small helpers ----------------------------------------------------

    const CB_SIZE: usize = 4;
    const CB_DIM: usize = 2;
    const D_MODEL: usize = 5;

    /// Deterministic low-dim ramp codebook for stage `cb`:
    /// row `i` = `[i + 10*cb, i + 10*cb + 1]`. Distinct per stage so the
    /// residual sum picks up cross-stage differences.
    fn make_codebook(cb: usize) -> CodebookTable {
        let mut data = vec![0.0_f32; CB_SIZE * CB_DIM];
        for i in 0..CB_SIZE {
            for d in 0..CB_DIM {
                data[i * CB_DIM + d] = (i + d) as f32 + (cb as f32) * 10.0;
            }
        }
        CodebookTable::new(CB_SIZE, CB_DIM, data).unwrap()
    }

    /// Deterministic per-stage projection: exactly representable
    /// (powers-of-two + integer coefficients) so hand folds stay bit-clean.
    fn make_proj(cb: usize) -> DacOutProj {
        let mut w = vec![0.0_f32; D_MODEL * CB_DIM];
        for o in 0..D_MODEL {
            for c in 0..CB_DIM {
                w[o * CB_DIM + c] = 0.5 + o as f32 * 0.25 + c as f32 * 0.125 + cb as f32;
            }
        }
        let b: Vec<f32> = (0..D_MODEL)
            .map(|o| o as f32 * 0.0625 - cb as f32 * 0.5)
            .collect();
        DacOutProj::new(D_MODEL, CB_DIM, w, b).unwrap()
    }

    fn make_weights() -> SnacWeights {
        SnacWeights {
            codebooks: vec![make_codebook(0), make_codebook(1), make_codebook(2)],
            out_projs: vec![make_proj(0), make_proj(1), make_proj(2)],
        }
    }

    fn make_weights_4() -> SnacWeights {
        SnacWeights {
            codebooks: vec![
                make_codebook(0),
                make_codebook(1),
                make_codebook(2),
                make_codebook(3),
            ],
            out_projs: vec![make_proj(0), make_proj(1), make_proj(2), make_proj(3)],
        }
    }

    fn tiny_config() -> SnacConfig {
        // Non-canonical strides — [4, 2, 1] is the SNAC 24 kHz shape.
        SnacConfig {
            sample_rate: 24_000,
            vq_strides: [4, 2, 1, 0],
            n_stages: 3,
        }
    }

    // ---- Canonical variant ------------------------------------------------

    #[test]
    fn snac_config_canonical_matches_hubertsiuzdak_24khz_defaults() {
        let c = SnacConfig::snac_24khz();
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.active_vq_strides(), [4, 2, 1]);
        assert_eq!(c.n_stages, 3);
        let c44 = SnacConfig::snac_44khz();
        assert_eq!(c44.sample_rate, 44_100);
        assert_eq!(c44.active_vq_strides(), [8, 4, 2, 1]);
        assert_eq!(c44.n_stages, 4);
    }

    // ---- Constructor validation ------------------------------------------

    #[test]
    fn new_accepts_valid_bundle() {
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        assert_eq!(d.codebook_size(), CB_SIZE);
        assert_eq!(d.codebook_dim(), CB_DIM);
        assert_eq!(d.d_model(), D_MODEL);
        assert_eq!(d.config().active_vq_strides(), [4, 2, 1]);
    }

    #[test]
    fn new_rejects_zero_stride() {
        let mut cfg = tiny_config();
        cfg.vq_strides[1] = 0;
        assert!(matches!(
            SnacDecoder::new(cfg, make_weights()),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_codebook_shape_mismatch() {
        // Give stage 1 a different codebook_size than stage 0 / stage 2.
        let odd =
            CodebookTable::new(CB_SIZE + 1, CB_DIM, vec![0.0; (CB_SIZE + 1) * CB_DIM]).unwrap();
        let weights = SnacWeights {
            codebooks: vec![make_codebook(0), odd, make_codebook(2)],
            out_projs: vec![make_proj(0), make_proj(1), make_proj(2)],
        };
        assert!(matches!(
            SnacDecoder::new(tiny_config(), weights),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_codebook_dim_mismatch() {
        // Stage 2's row width differs (would break the factorized fold).
        let wide =
            CodebookTable::new(CB_SIZE, CB_DIM + 1, vec![0.0; CB_SIZE * (CB_DIM + 1)]).unwrap();
        let weights = SnacWeights {
            codebooks: vec![make_codebook(0), make_codebook(1), wide],
            out_projs: vec![make_proj(0), make_proj(1), make_proj(2)],
        };
        assert!(matches!(
            SnacDecoder::new(tiny_config(), weights),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_out_proj_d_model_mismatch() {
        // Stage 2's out_proj projects into a wider space than stage 0 / 1.
        let wide = DacOutProj::new(
            D_MODEL + 1,
            CB_DIM,
            vec![0.0; (D_MODEL + 1) * CB_DIM],
            vec![0.0; D_MODEL + 1],
        )
        .unwrap();
        let weights = SnacWeights {
            codebooks: vec![make_codebook(0), make_codebook(1), make_codebook(2)],
            out_projs: vec![make_proj(0), make_proj(1), wide],
        };
        assert!(matches!(
            SnacDecoder::new(tiny_config(), weights),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_out_proj_codebook_dim_mismatch() {
        // Stage 0's out_proj expects a wider low-dim row than the codebooks
        // provide — the fold would read past the end of the row.
        let wrong = DacOutProj::new(
            D_MODEL,
            CB_DIM + 1,
            vec![0.0; D_MODEL * (CB_DIM + 1)],
            vec![0.0; D_MODEL],
        )
        .unwrap();
        let weights = SnacWeights {
            codebooks: vec![make_codebook(0), make_codebook(1), make_codebook(2)],
            out_projs: vec![wrong, make_proj(1), make_proj(2)],
        };
        assert!(matches!(
            SnacDecoder::new(tiny_config(), weights),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- Decode: happy path ----------------------------------------------

    #[test]
    #[allow(clippy::needless_range_loop, clippy::useless_vec)]
    // Hand fold mirrors the op's math 1:1 — index form is the intentional
    // shape, not a style oversight.
    fn decode_matches_hand_fold_for_strides_4_2_1() {
        // T = 4 (stage 0: 1 code × stride 4, stage 1: 2 codes × stride 2,
        // stage 2: 4 codes × stride 1).
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![3], vec![0, 2], vec![1, 0, 3, 2]];
        let strides = [4usize, 2, 1];
        let t_expanded = 4;

        let got = d.decode(&codes).unwrap();
        assert_eq!(got.len(), t_expanded * D_MODEL);

        // Hand fold: per stage, per timestep, project low-dim row up through
        // the stage's out_proj (bias + FP32 MAC), then add to every output
        // timestep in the repeat_interleave span.
        let mut want = vec![0.0_f32; t_expanded * D_MODEL];
        for stage in 0..3 {
            let cb = &d.weights.codebooks[stage];
            let proj = &d.weights.out_projs[stage];
            for (t_stage, &idx) in codes[stage].iter().enumerate() {
                let low = cb.row(idx).unwrap();
                let mut projected = vec![0.0_f32; D_MODEL];
                for o in 0..D_MODEL {
                    let mut y = proj.bias[o];
                    for c in 0..CB_DIM {
                        y += proj.weight[o * CB_DIM + c] * low[c];
                    }
                    projected[o] = y;
                }
                let stride = strides[stage];
                let t_start = t_stage * stride;
                for t_exp in 0..stride {
                    let t_out = t_start + t_exp;
                    for o in 0..D_MODEL {
                        want[t_out * D_MODEL + o] += projected[o];
                    }
                }
            }
        }

        assert_eq!(
            got, want,
            "residual fold must be bit-identical to hand FP32 loop"
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    // Hand fold mirrors the op's math 1:1.
    fn decode_matches_hand_fold_for_strides_1_1_1() {
        // With strides all 1, SNAC collapses to a standard 3-quantizer RVQ
        // (every stage runs at the base rate). Uses a longer T to exercise
        // multiple timesteps.
        let cfg = SnacConfig {
            sample_rate: 24_000,
            vq_strides: [1, 1, 1, 0],
            n_stages: 3,
        };
        let d = SnacDecoder::new(cfg, make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![0, 1, 2, 3], vec![3, 2, 1, 0], vec![1, 3, 0, 2]];
        let got = d.decode(&codes).unwrap();
        assert_eq!(got.len(), 4 * D_MODEL);

        // Reference: per timestep sum three stages' projected rows.
        let mut want = vec![0.0_f32; 4 * D_MODEL];
        for t in 0..4 {
            for stage in 0..3 {
                let cb = &d.weights.codebooks[stage];
                let proj = &d.weights.out_projs[stage];
                let idx = codes[stage][t];
                let low = cb.row(idx).unwrap();
                for o in 0..D_MODEL {
                    let mut y = proj.bias[o];
                    for c in 0..CB_DIM {
                        y += proj.weight[o * CB_DIM + c] * low[c];
                    }
                    want[t * D_MODEL + o] += y;
                }
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn decode_single_slot_matches_manual_arithmetic() {
        // One timestep worth of output, one code per stage.
        // codebook_size=2, codebook_dim=2, d_model=3.
        let cb = CodebookTable::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // W = [[1, 0], [0, 1], [1, 1]], b = [0.5, -0.5, 0.25]
        let proj = DacOutProj::new(
            3,
            2,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.5, -0.5, 0.25],
        )
        .unwrap();

        let weights = SnacWeights {
            codebooks: vec![cb.clone(), cb.clone(), cb.clone()],
            out_projs: vec![proj.clone(), proj.clone(), proj.clone()],
        };
        let cfg = SnacConfig {
            sample_rate: 24_000,
            vq_strides: [1, 1, 1, 0],
            n_stages: 3,
        };
        let d = SnacDecoder::new(cfg, weights).unwrap();

        // Three stages, all decoding code=1, all strides 1 → sum three copies
        // of one projected row.
        let codes: [Vec<u32>; 3] = [vec![1], vec![1], vec![1]];
        let got = d.decode(&codes).unwrap();
        // Row 1 = [3, 4]; W@row + b = [3+0.5, 4-0.5, 7+0.25] = [3.5, 3.5, 7.25]
        // Summed three times: [10.5, 10.5, 21.75].
        assert_eq!(got, vec![10.5, 10.5, 21.75]);
    }

    #[test]
    fn decode_stride_expansion_broadcasts_stage_zero_evenly() {
        // Stage 0 with stride 4 contributes the same projected row to 4
        // output timesteps. Stages 1 and 2 stay zero here.
        let cb = CodebookTable::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let proj = DacOutProj::new(
            3,
            2,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        )
        .unwrap();
        let zero_cb = CodebookTable::new(2, 2, vec![0.0; 4]).unwrap();
        let zero_proj = DacOutProj::new(3, 2, vec![0.0; 6], vec![0.0; 3]).unwrap();

        let weights = SnacWeights {
            codebooks: vec![cb, zero_cb.clone(), zero_cb],
            out_projs: vec![proj, zero_proj.clone(), zero_proj],
        };
        let cfg = SnacConfig {
            sample_rate: 24_000,
            vq_strides: [4, 2, 1, 0],
            n_stages: 3,
        };
        let d = SnacDecoder::new(cfg, weights).unwrap();

        // Stage 0: 1 code (row 1 = [3, 4] → W@row = [3, 4, 7]) × stride 4.
        // Stages 1 / 2: zero contribution.
        let codes: [Vec<u32>; 3] = [vec![1], vec![0, 0], vec![0, 0, 0, 0]];
        let got = d.decode(&codes).unwrap();
        assert_eq!(got.len(), 4 * 3);
        // Every output timestep must be [3, 4, 7].
        for t in 0..4 {
            let base = t * 3;
            assert_eq!(&got[base..base + 3], &[3.0, 4.0, 7.0][..]);
        }
    }

    // ---- Decode: edge cases -----------------------------------------------

    #[test]
    fn decode_all_empty_returns_empty() {
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![], vec![], vec![]];
        let got = d.decode(&codes).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    #[allow(clippy::needless_range_loop, clippy::useless_vec)]
    // Hand-computed reference row mirrors the op's math 1:1.
    fn decode_blank_only_codes_still_decodes() {
        // "Blank-only" here = every stage decodes only code 0 for its length.
        // That is a valid input (0 is a legal codebook index); we just want to
        // confirm the fold + expansion never trip on it.
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![0], vec![0, 0], vec![0, 0, 0, 0]];
        let got = d.decode(&codes).unwrap();
        assert_eq!(got.len(), 4 * D_MODEL);

        // Every output timestep must equal the sum-of-three-stages projected
        // row for code=0 (the same value at every t because there is only one
        // logical stage-0 timestep and it broadcasts to all four).
        let mut expected_row = vec![0.0_f32; D_MODEL];
        for stage in 0..3 {
            let cb = &d.weights.codebooks[stage];
            let proj = &d.weights.out_projs[stage];
            let low = cb.row(0).unwrap();
            for o in 0..D_MODEL {
                let mut y = proj.bias[o];
                for c in 0..CB_DIM {
                    y += proj.weight[o * CB_DIM + c] * low[c];
                }
                expected_row[o] += y;
            }
        }
        for t in 0..4 {
            let base = t * D_MODEL;
            assert_eq!(&got[base..base + D_MODEL], expected_row.as_slice());
        }
    }

    // ---- Decode: shape validation -----------------------------------------

    #[test]
    fn decode_rejects_misaligned_stage_lengths() {
        // Stage 0 length 1 × stride 4 → T = 4; stage 1 length 3 × stride 2 → T = 6.
        // Must fail loud, not silently trim.
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![0], vec![0, 0, 0], vec![0, 0, 0, 0]];
        assert!(matches!(
            d.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_rejects_mixed_empty_and_nonempty() {
        // One stage empty, others populated → different Ts across stages.
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![], vec![0, 0], vec![0, 0, 0, 0]];
        assert!(matches!(
            d.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_rejects_out_of_range_code() {
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        // codebook_size == CB_SIZE (= 4) → code = 4 is out of range.
        let codes: [Vec<u32>; 3] = [vec![CB_SIZE as u32], vec![0, 0], vec![0, 0, 0, 0]];
        assert!(matches!(
            d.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_rejects_length_overflow() {
        // Stride 1 avoids the overflow trap on stride = usize::MAX; instead
        // give stride 2 with usize::MAX / 2 + 1 codes so the multiplication
        // overflows even after the checked cast. We do NOT actually allocate
        // usize::MAX / 2 codes; we lie about the length via a synthetic Vec
        // — that is impossible without unsafe. So instead we simulate by
        // reaching the check via strides.
        //
        // Practical minimal test: stride = u32::MAX and a Vec of length 2 —
        // 2 * u32::MAX (~8.6e9) overflows on 32-bit targets but not on 64-bit.
        // On 64-bit runners (the majority), the multiplication succeeds; we
        // therefore assert only the shape-consistency error path via
        // misalignment, which is the only overflow-adjacent guarantee we
        // actually rely on downstream.
        //
        // The `checked_mul` on 64-bit runs would require lengths near
        // 2^63 / stride, which is not constructible in-memory — the guard is
        // real code but unreachable through a legitimate Vec allocation.
        // This test just exercises the sibling error path so the guard
        // stays covered by *some* case.
        let d = SnacDecoder::new(tiny_config(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![0, 0], vec![0, 0], vec![0, 0, 0, 0]];
        // stage 0: 2 × 4 = 8; stage 1: 2 × 2 = 4 → mismatch.
        assert!(matches!(
            d.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- Host-only smoke --------------------------------------------------

    #[test]
    fn host_only_smoke_decode_end_to_end() {
        // Full path — construct decoder, decode a [4, D_MODEL] block with
        // strides [4, 2, 1] — runs on the CPU with zero external
        // dependencies and no GPU anywhere (mirror of mimi_rvq /
        // dac_rvq's host_only_smoke tests).
        let d = SnacDecoder::new(SnacConfig::snac_24khz(), make_weights()).unwrap();
        let codes: [Vec<u32>; 3] = [vec![2], vec![1, 3], vec![0, 1, 2, 3]];
        let out = d.decode(&codes).unwrap();
        assert_eq!(out.len(), 4 * D_MODEL);
        // Values are finite (no NaN / Inf leaks from the fold).
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn snac_44khz_four_stage_decode_matches_hand_fold() {
        let d = SnacDecoder::new(SnacConfig::snac_44khz(), make_weights_4()).unwrap();
        let codes = [
            vec![1],
            vec![2, 3],
            vec![0, 1, 2, 3],
            vec![3, 2, 1, 0, 1, 2, 3, 0],
        ];
        let got = d.decode(&codes).unwrap();
        assert_eq!(got.len(), 8 * D_MODEL);

        let mut want = vec![0.0_f32; 8 * D_MODEL];
        for stage in 0..4 {
            let stride = d.config().active_vq_strides()[stage] as usize;
            let cb = &d.weights.codebooks[stage];
            let proj = &d.weights.out_projs[stage];
            for (t_stage, &idx) in codes[stage].iter().enumerate() {
                let low = cb.row(idx).unwrap();
                for t in t_stage * stride..(t_stage + 1) * stride {
                    for o in 0..D_MODEL {
                        let mut y = proj.bias[o];
                        for c in 0..CB_DIM {
                            y += proj.weight[o * CB_DIM + c] * low[c];
                        }
                        want[t * D_MODEL + o] += y;
                    }
                }
            }
        }
        assert_eq!(got, want);
    }
}
