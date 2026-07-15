//! DAC (Descript Audio Codec) factorized residual VQ decode
//! (M4-04; FR-OP-30, third and last op of the RVQ family).
//!
//! # Why DAC needs its own op (and is not an alias of `mimi_rvq`)
//!
//! DAC's quantizers are **factorized** ("Factorized codes" trick from
//! Improved VQGAN — upstream `dac/nn/quantize.py` L13-32, see the shapes /
//! structure source note below): each quantizer owns
//!
//! - a **low-dimensional** codebook `[codebook_size, codebook_dim]`
//!   (`codebook_dim = 8` for every released variant), and
//! - a per-quantizer 1×1 conv `out_proj` (`codebook_dim → d_model`,
//!   weight-normed, **with bias**) applied **before** the residual sum:
//!
//! ```text
//!   decoded[t, :] = sum_{cb} ( W_cb @ codebook_cb[codes[t, cb]] + b_cb )
//! ```
//!
//! (upstream `ResidualVectorQuantize.from_codes`, quantize.py L200-220 —
//! projection per quantizer, *then* sum; unlike Mimi where the projection is
//! applied after the per-split sum). A plain-table op would require the
//! converter to materialise `n_codebooks × codebook_size × d_model` effective
//! tables (~134 MB for the 24 kHz variant) versus ~0.5 MB factorized, so the
//! runtime op mirrors the factorized architecture 1:1 (ADR M4-04 §D-f /
//! alternatives).
//!
//! The GEMV + residual fold is FP32-accumulated (same audio-dialect rule as
//! [`crate::mimi_rvq`]: "BF16 mantissa loss is the real problem", CLAUDE.md).
//!
//! # Shapes / structure source (nothing invented — ADR M4-04 §T02)
//!
//! The one and only source for shapes and structure is the upstream
//! descript-audio-codec implementation (MIT), pinned in ADR M4-04:
//! `dac/nn/quantize.py` (`VectorQuantize` / `ResidualVectorQuantize`) at
//! descriptinc/descript-audio-codec `c7cfc5d2`. The 24 kHz zoo variant's
//! verified checkpoint metadata (`weights_24khz.pth`, tag 0.0.4):
//! `n_codebooks = 32`, `codebook_size = 1024`, `codebook_dim = 8`,
//! `d_model (latent_dim) = 1024`, `hop = 320` at 24 kHz ⇒ **75 Hz** frame
//! rate.
//!
//! # Paged variant (75–86 Hz → `block_size=4` primary)
//!
//! [`dac_rvq_decode_paged`] mirrors [`crate::mimi_rvq::mimi_rvq_decode_paged`]:
//! per-codebook, pre-sum **projected** features are written into a
//! [`PagedKvCache<f32>`] `[time, stream, codebook]` arena (K side = feature,
//! V side = zero, `n_head = 1`, `d_head = d_model`). [`BlockSize::Four`] is
//! the **primary** block size for DAC — every released DAC variant runs at
//! 50–86.13 Hz (16 kHz → 50 Hz, 24 kHz → 75 Hz, 44.1 kHz → 86.13 Hz), so a
//! 4-step block spans 46–80 ms. [`BlockSize::Two`] also works (FR-EX-03
//! "block 2-4"; the enum deliberately has nothing larger — M3-03 ADR §D2).
//!
//! # No silent fallback (FR-EX-08)
//!
//! Out-of-range indices, shape mismatches (codes / tables / projections /
//! paged cache) are explicit [`VokraError::InvalidArgument`] — never a silent
//! clamp. A wrong RVQ index corrupts the downstream feature stream in a
//! plausible-looking way, so the decode is where the error must surface.
//!
//! # Runtime function — not an `OpKind` variant
//!
//! Same two reasons as `mimi_rvq` (its module docs L61-77; ADR M4-04 §D-b):
//! live `&mut PagedKvCache` state and imperative consumer shape (M4-05 CSM /
//! M4-06 Moshi).
//!
//! # GPU seam
//!
//! `Compute::dac_rvq_f32` (vokra-models) has a real CPU arm delegating here;
//! Metal / CUDA / Vulkan arms are explicit [`VokraError::UnsupportedOp`]
//! until a kernel lands (naive gather + GEMV + fold layout — the
//! `mimi_rvq` L104-106 kernel note applies to all three RVQ ops).
//!
//! [`PagedKvCache<f32>`]: vokra_core::cache::paged::PagedKvCache
//! [`BlockSize::Four`]: vokra_core::cache::paged::BlockSize
//! [`BlockSize::Two`]: vokra_core::cache::paged::BlockSize

use vokra_core::cache::paged::{KvDims, PagedKvCache};
use vokra_core::{Result, VokraError};

use crate::mimi_rvq::CodebookTable;

// ---------------------------------------------------------------------------
// Op attributes
// ---------------------------------------------------------------------------

/// Static shape attributes for a DAC factorized RVQ decode.
///
/// The four numbers come from the checkpoint (converter emits
/// `vokra.dac.{n_codebooks,codebook_size,codebook_dim,d_model}` — M4-04 T11);
/// at decode time the runtime just consumes them here. Unlike
/// [`crate::mimi_rvq::MimiRvqAttrs`] there is an extra `codebook_dim` axis
/// because the codebook rows live in the factorized low-dimensional space and
/// are projected up to `d_model` per quantizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DacRvqAttrs {
    /// Number of quantizers (base + residuals). 24 kHz variant = 32.
    pub n_codebooks: usize,
    /// Number of entries per codebook. Every released DAC variant = 1024.
    pub codebook_size: usize,
    /// Factorized codebook row width (pre-projection). Every released DAC
    /// variant = 8.
    pub codebook_dim: usize,
    /// Output feature width per timestep (= DAC `latent_dim`). 24 kHz
    /// variant = 1024.
    pub d_model: usize,
}

impl DacRvqAttrs {
    /// Builds the verified 24 kHz / 8 kbps zoo-variant shape
    /// (32 × 1024 × 8 → 1024).
    ///
    /// Values verified from the actual `weights_24khz.pth` (tag 0.0.4)
    /// checkpoint metadata — ADR M4-04 §T02. Callers with a different DAC
    /// variant build the struct field-by-field from their checkpoint's
    /// `vokra.dac.*` metadata.
    #[inline]
    #[must_use]
    pub const fn dac_24khz() -> Self {
        Self {
            n_codebooks: 32,
            codebook_size: 1024,
            codebook_dim: 8,
            d_model: 1024,
        }
    }
}

/// One quantizer's output projection: a 1×1 conv `codebook_dim → d_model`
/// with bias, weight-norm **already folded offline** by the converter
/// (`W = g * v / ||v||₂` per output row — torch `weight_norm` with `dim=0`;
/// the runtime never sees `weight_g` / `weight_v`).
#[derive(Debug, Clone, PartialEq)]
pub struct DacOutProj {
    /// Output width (rows of `weight`).
    pub d_model: usize,
    /// Input width (columns of `weight`).
    pub codebook_dim: usize,
    /// Row-major `[d_model, codebook_dim]` folded weight.
    pub weight: Vec<f32>,
    /// `[d_model]` bias (upstream `WNConv1d` keeps `nn.Conv1d`'s default
    /// `bias=True`).
    pub bias: Vec<f32>,
}

impl DacOutProj {
    /// Constructs a projection, validating both buffer lengths.
    pub fn new(
        d_model: usize,
        codebook_dim: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self> {
        if d_model == 0 || codebook_dim == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "DacOutProj::new: d_model and codebook_dim must be > 0, got \
                 d_model={d_model} codebook_dim={codebook_dim}"
            )));
        }
        let expected_w = d_model * codebook_dim;
        if weight.len() != expected_w {
            return Err(VokraError::InvalidArgument(format!(
                "DacOutProj::new: weight.len() {} != d_model * codebook_dim {expected_w}",
                weight.len()
            )));
        }
        if bias.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "DacOutProj::new: bias.len() {} != d_model {d_model}",
                bias.len()
            )));
        }
        Ok(Self {
            d_model,
            codebook_dim,
            weight,
            bias,
        })
    }

    /// Row `o` of the folded weight (`codebook_dim` long).
    #[inline]
    #[must_use]
    pub fn weight_row(&self, o: usize) -> &[f32] {
        let base = o * self.codebook_dim;
        &self.weight[base..base + self.codebook_dim]
    }
}

// ---------------------------------------------------------------------------
// Core op
// ---------------------------------------------------------------------------

/// Decodes a `[time, n_codebooks]` row-major `codes` block into a
/// `[time, d_model]` row-major feature buffer:
/// `out[t, :] = Σ_cb ( W_cb @ codebook_cb[codes[t, cb]] + b_cb )`,
/// FP32-accumulated.
///
/// The output is `[time, d_model]` **features** — not PCM; the DAC
/// feature→PCM decoder chain is a consumer-WP concern (ADR M4-04 §D-g).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - shape mismatch in `codes` / `codebook_tables` / `out_projs` vs `attrs`;
/// - `codes[t, cb] >= attrs.codebook_size` (no silent clamp — FR-EX-08).
pub fn dac_rvq_decode(
    codes: &[u32],
    time: usize,
    codebook_tables: &[CodebookTable],
    out_projs: &[DacOutProj],
    attrs: &DacRvqAttrs,
) -> Result<Vec<f32>> {
    check_shapes(codes, time, codebook_tables, out_projs, attrs)?;

    let mut out = vec![0.0_f32; time * attrs.d_model];
    for t in 0..time {
        let out_base = t * attrs.d_model;
        let code_base = t * attrs.n_codebooks;
        let acc = &mut out[out_base..out_base + attrs.d_model];
        for cb in 0..attrs.n_codebooks {
            let idx = codes[code_base + cb];
            let low = codebook_tables[cb].row(idx)?;
            project_accumulate(&out_projs[cb], low, acc);
        }
    }
    Ok(out)
}

/// `acc[o] += bias[o] + Σ_c W[o, c] * low[c]` — the per-(timestep, quantizer)
/// factorized projection fold. FP32 only (module docs).
#[inline]
fn project_accumulate(proj: &DacOutProj, low: &[f32], acc: &mut [f32]) {
    for (o, dst) in acc.iter_mut().enumerate() {
        let w_row = proj.weight_row(o);
        let mut y = proj.bias[o];
        for (w, x) in w_row.iter().zip(low.iter()) {
            y += *w * *x;
        }
        *dst += y;
    }
}

// ---------------------------------------------------------------------------
// Paged variant (M3-03 layout; block_size=4 primary for 75 Hz DAC)
// ---------------------------------------------------------------------------

/// Writes per-codebook (pre-sum) **projected** feature vectors into a
/// [`PagedKvCache<f32>`] with `[time, stream, codebook]` addressing.
///
/// Mirror of [`crate::mimi_rvq::mimi_rvq_decode_paged`] with the factorized
/// projection applied at write time, so each `(t, stream, cb)` slot holds
/// `W_cb @ codebook_cb[codes[t, cb]] + b_cb` (a `d_model`-long K row; V side
/// zero). The paged cache must be sized `n_layer = 1`, `n_head = 1`,
/// `d_head = attrs.d_model`, `n_codebook >= attrs.n_codebooks`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on shape / dim / range violations;
/// [`VokraError::KvCacheExhausted`] surfaced verbatim from
/// [`PagedKvCache::append_step`] if the arena is out of pages.
///
/// [`PagedKvCache<f32>`]: vokra_core::cache::paged::PagedKvCache
/// [`PagedKvCache::append_step`]: vokra_core::cache::paged::PagedKvCache::append_step
#[allow(clippy::too_many_arguments)]
pub fn dac_rvq_decode_paged(
    codes: &[u32],
    time: usize,
    codebook_tables: &[CodebookTable],
    out_projs: &[DacOutProj],
    attrs: &DacRvqAttrs,
    stream: usize,
    cache: &mut PagedKvCache<f32>,
    time_start: usize,
) -> Result<()> {
    check_shapes(codes, time, codebook_tables, out_projs, attrs)?;
    crate::mimi_rvq::check_cache_dims(cache, attrs.n_codebooks, attrs.d_model, "dac_rvq paged")?;

    let dims = cache.dims();
    if stream >= dims.n_stream {
        return Err(VokraError::InvalidArgument(format!(
            "dac_rvq_decode_paged: stream {stream} >= cache.n_stream {}",
            dims.n_stream
        )));
    }
    let end = time_start.checked_add(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "dac_rvq_decode_paged: time_start ({time_start}) + time ({time}) overflows"
        ))
    })?;
    if end > dims.max_time {
        return Err(VokraError::InvalidArgument(format!(
            "dac_rvq_decode_paged: time_start + time = {end} > cache.max_time {}",
            dims.max_time
        )));
    }

    let v_zeros = vec![0.0_f32; attrs.d_model];
    let mut projected = vec![0.0_f32; attrs.d_model];

    for t in 0..time {
        let code_base = t * attrs.n_codebooks;
        for cb in 0..attrs.n_codebooks {
            let idx = codes[code_base + cb];
            let low = codebook_tables[cb].row(idx)?;
            projected.iter_mut().for_each(|x| *x = 0.0);
            project_accumulate(&out_projs[cb], low, &mut projected);
            cache.append_step(0, time_start + t, stream, cb, &projected, &v_zeros)?;
        }
    }
    Ok(())
}

/// Reads and sums the per-codebook projected features previously written by
/// [`dac_rvq_decode_paged`] for `(stream, t)` — the mirror of
/// [`dac_rvq_decode`]'s output row.
///
/// Unwritten codebook slots contribute zero on the *read* side (same
/// asymmetry as `mimi_rvq_read_summed` — FR-EX-08 governs writes, reads
/// treat the paged store as an arena; mimi_rvq.rs L369-372 規約).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on axis / dim violations.
pub fn dac_rvq_read_summed(
    cache: &PagedKvCache<f32>,
    attrs: &DacRvqAttrs,
    stream: usize,
    t: usize,
) -> Result<Vec<f32>> {
    crate::mimi_rvq::read_summed_core(
        cache,
        attrs.n_codebooks,
        attrs.d_model,
        stream,
        t,
        "dac_rvq_read_summed",
    )
}

/// Builds a [`KvDims`] with the shape [`dac_rvq_decode_paged`] expects
/// (mirror of [`crate::mimi_rvq::mimi_paged_dims`]).
#[must_use]
pub fn dac_paged_dims(attrs: &DacRvqAttrs, n_stream: usize, max_time: usize) -> KvDims {
    KvDims {
        n_layer: 1,
        n_head: 1,
        d_head: attrs.d_model,
        n_stream,
        n_codebook: attrs.n_codebooks,
        max_time,
    }
}

// ---------------------------------------------------------------------------
// Shared shape checks
// ---------------------------------------------------------------------------

fn check_shapes(
    codes: &[u32],
    time: usize,
    codebook_tables: &[CodebookTable],
    out_projs: &[DacOutProj],
    attrs: &DacRvqAttrs,
) -> Result<()> {
    if attrs.n_codebooks == 0
        || attrs.codebook_size == 0
        || attrs.codebook_dim == 0
        || attrs.d_model == 0
    {
        return Err(VokraError::InvalidArgument(format!(
            "dac_rvq: attrs must have every axis > 0, got n_codebooks={} \
             codebook_size={} codebook_dim={} d_model={}",
            attrs.n_codebooks, attrs.codebook_size, attrs.codebook_dim, attrs.d_model,
        )));
    }
    if codebook_tables.len() != attrs.n_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "dac_rvq: codebook_tables.len() {} != attrs.n_codebooks {}",
            codebook_tables.len(),
            attrs.n_codebooks
        )));
    }
    for (i, t) in codebook_tables.iter().enumerate() {
        // The low-dim table re-uses `CodebookTable`, whose `d_model` field is
        // the row width — here that row width must be the factorized
        // `codebook_dim`, not the output `d_model`.
        if t.codebook_size != attrs.codebook_size || t.d_model != attrs.codebook_dim {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq: codebook_tables[{i}] shape [{},{}] != attrs [{},{}] \
                 (row width must be the factorized codebook_dim)",
                t.codebook_size, t.d_model, attrs.codebook_size, attrs.codebook_dim
            )));
        }
    }
    if out_projs.len() != attrs.n_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "dac_rvq: out_projs.len() {} != attrs.n_codebooks {}",
            out_projs.len(),
            attrs.n_codebooks
        )));
    }
    for (i, p) in out_projs.iter().enumerate() {
        if p.d_model != attrs.d_model || p.codebook_dim != attrs.codebook_dim {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq: out_projs[{i}] shape [{},{}] != attrs [{},{}]",
                p.d_model, p.codebook_dim, attrs.d_model, attrs.codebook_dim
            )));
        }
    }
    let expected = time.checked_mul(attrs.n_codebooks).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "dac_rvq: time ({time}) * n_codebooks ({}) overflows usize",
            attrs.n_codebooks
        ))
    })?;
    if codes.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "dac_rvq: codes.len() {} != time * n_codebooks {expected}",
            codes.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::cache::paged::{BlockSize, PagedKvCache};

    /// Tiny attrs: 3 quantizers, 4 entries, factorized dim 2, output dim 5.
    fn tiny_attrs() -> DacRvqAttrs {
        DacRvqAttrs {
            n_codebooks: 3,
            codebook_size: 4,
            codebook_dim: 2,
            d_model: 5,
        }
    }

    /// Deterministic low-dim ramp codebooks: row `i` of codebook `cb` is
    /// `[i + 10*cb, i + 10*cb + 1]`.
    fn make_low_tables(attrs: DacRvqAttrs) -> Vec<CodebookTable> {
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut data = vec![0.0_f32; attrs.codebook_size * attrs.codebook_dim];
            for i in 0..attrs.codebook_size {
                for d in 0..attrs.codebook_dim {
                    data[i * attrs.codebook_dim + d] = (i + d) as f32 + (cb as f32) * 10.0;
                }
            }
            tables.push(CodebookTable::new(attrs.codebook_size, attrs.codebook_dim, data).unwrap());
        }
        tables
    }

    /// Deterministic projections: `W_cb[o, c] = 0.5 + o as f32 * 0.25 + c as
    /// f32 * 0.125 + cb as f32`, `b_cb[o] = o as f32 * 0.0625 - cb as f32 *
    /// 0.5`. Exactly representable in f32 (powers of two) so hand folds stay
    /// bit-clean.
    fn make_projs(attrs: DacRvqAttrs) -> Vec<DacOutProj> {
        let mut projs = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut w = vec![0.0_f32; attrs.d_model * attrs.codebook_dim];
            for o in 0..attrs.d_model {
                for c in 0..attrs.codebook_dim {
                    w[o * attrs.codebook_dim + c] =
                        0.5 + o as f32 * 0.25 + c as f32 * 0.125 + cb as f32;
                }
            }
            let b: Vec<f32> = (0..attrs.d_model)
                .map(|o| o as f32 * 0.0625 - cb as f32 * 0.5)
                .collect();
            projs.push(DacOutProj::new(attrs.d_model, attrs.codebook_dim, w, b).unwrap());
        }
        projs
    }

    // ---- T03: attrs + canonical variant -----------------------------------

    #[test]
    fn dac_attrs_canonical_matches_verified_24khz_checkpoint() {
        // ADR M4-04 §T02: verified from weights_24khz.pth metadata.kwargs.
        let a = DacRvqAttrs::dac_24khz();
        assert_eq!(a.n_codebooks, 32);
        assert_eq!(a.codebook_size, 1024);
        assert_eq!(a.codebook_dim, 8);
        assert_eq!(a.d_model, 1024);
    }

    #[test]
    fn out_proj_new_validates_shapes() {
        assert!(matches!(
            DacOutProj::new(5, 2, vec![0.0; 9], vec![0.0; 5]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            DacOutProj::new(5, 2, vec![0.0; 10], vec![0.0; 4]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            DacOutProj::new(0, 2, vec![], vec![]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(DacOutProj::new(5, 2, vec![0.0; 10], vec![0.0; 5]).is_ok());
    }

    // ---- T04: decode vs hand-fold oracle ----------------------------------

    #[test]
    #[allow(clippy::needless_range_loop)] // index-form hand fold mirrors the op's math 1:1
    fn decode_is_bit_identical_to_hand_fold() {
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let time = 3;
        let codes: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 1, 0, 3];

        let got = dac_rvq_decode(&codes, time, &tables, &projs, &attrs).unwrap();

        // Hand fold: same scalar loop, written independently.
        let mut want = vec![0.0_f32; time * attrs.d_model];
        for t in 0..time {
            for cb in 0..attrs.n_codebooks {
                let idx = codes[t * attrs.n_codebooks + cb] as usize;
                let low =
                    &tables[cb].data[idx * attrs.codebook_dim..(idx + 1) * attrs.codebook_dim];
                for o in 0..attrs.d_model {
                    let mut y = projs[cb].bias[o];
                    for c in 0..attrs.codebook_dim {
                        y += projs[cb].weight[o * attrs.codebook_dim + c] * low[c];
                    }
                    want[t * attrs.d_model + o] += y;
                }
            }
        }
        assert_eq!(
            got, want,
            "factorized decode must be a bit-identical FP32 fold"
        );
    }

    #[test]
    fn decode_single_slot_matches_manual_arithmetic() {
        // One timestep, one quantizer — fully hand-computed expectation.
        let attrs = DacRvqAttrs {
            n_codebooks: 1,
            codebook_size: 2,
            codebook_dim: 2,
            d_model: 3,
        };
        // codebook rows: row0 = [1, 2], row1 = [3, 4]
        let tables = vec![CodebookTable::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()];
        // W = [[1, 0], [0, 1], [1, 1]], b = [0.5, -0.5, 0.25]
        let projs = vec![
            DacOutProj::new(
                3,
                2,
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![0.5, -0.5, 0.25],
            )
            .unwrap(),
        ];
        // code 1 → row [3, 4] → W@row + b = [3+0.5, 4-0.5, 7+0.25]
        let got = dac_rvq_decode(&[1], 1, &tables, &projs, &attrs).unwrap();
        assert_eq!(got, vec![3.5, 3.5, 7.25]);
    }

    #[test]
    fn decode_rejects_shape_mismatches_and_bad_index() {
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);

        // codes.len() mismatch.
        assert!(matches!(
            dac_rvq_decode(&[0u32; 2], 1, &tables, &projs, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Wrong table count.
        assert!(matches!(
            dac_rvq_decode(
                &[0u32; 3],
                1,
                &tables[..attrs.n_codebooks - 1],
                &projs,
                &attrs
            ),
            Err(VokraError::InvalidArgument(_))
        ));
        // Wrong table row width (built as d_model instead of codebook_dim).
        let wrong_width = vec![
            CodebookTable::new(
                attrs.codebook_size,
                attrs.d_model,
                vec![0.0; attrs.codebook_size * attrs.d_model]
            )
            .unwrap();
            attrs.n_codebooks
        ];
        assert!(matches!(
            dac_rvq_decode(&[0u32; 3], 1, &wrong_width, &projs, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Wrong projection count.
        assert!(matches!(
            dac_rvq_decode(&[0u32; 3], 1, &tables, &projs[..2], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Out-of-range code (== codebook_size) — FR-EX-08, no silent clamp.
        let over = {
            let mut v = vec![0u32; attrs.n_codebooks];
            v[2] = attrs.codebook_size as u32;
            v
        };
        assert!(matches!(
            dac_rvq_decode(&over, 1, &tables, &projs, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Zero-axis attrs.
        let zero = DacRvqAttrs {
            n_codebooks: 0,
            ..attrs
        };
        assert!(matches!(
            dac_rvq_decode(&[], 0, &[], &[], &zero),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T05: paged variant ------------------------------------------------

    #[test]
    fn paged_block_size_four_matches_direct_decode() {
        // BlockSize::Four is the DAC primary (75 Hz — module docs).
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let time = 6;
        let codes: Vec<u32> = (0..time as u32 * attrs.n_codebooks as u32)
            .map(|i| (i * 3) % attrs.codebook_size as u32)
            .collect();

        let dims = dac_paged_dims(&attrs, 1, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();
        dac_rvq_decode_paged(&codes, time, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();

        let direct = dac_rvq_decode(&codes, time, &tables, &projs, &attrs).unwrap();
        for t in 0..time {
            let summed = dac_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            let want = &direct[t * attrs.d_model..(t + 1) * attrs.d_model];
            assert_eq!(summed, want, "paged block_size=4, t={t}");
        }
        assert_eq!(cache.page_of(5), 1, "block_size=4 → t=5 on page 1");
    }

    #[test]
    fn paged_block_size_two_matches_direct_decode() {
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let time = 5;
        let codes: Vec<u32> = (0..time as u32 * attrs.n_codebooks as u32)
            .map(|i| i % attrs.codebook_size as u32)
            .collect();

        let dims = dac_paged_dims(&attrs, 1, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
        dac_rvq_decode_paged(&codes, time, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();

        let direct = dac_rvq_decode(&codes, time, &tables, &projs, &attrs).unwrap();
        for t in 0..time {
            let summed = dac_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            let want = &direct[t * attrs.d_model..(t + 1) * attrs.d_model];
            assert_eq!(summed, want, "paged block_size=2, t={t}");
        }
    }

    #[test]
    fn paged_layout_keeps_codebook_dim_contiguous() {
        // M3-03 row layout [block_offset, stream, codebook, head, d_head]:
        // each (t, stream, cb) slot must hold exactly that quantizer's
        // projected row (the M3-06 T07 contiguity assert, DAC edition).
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let codes: Vec<u32> = (0..attrs.n_codebooks as u32).collect();

        let dims = dac_paged_dims(&attrs, 1, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();
        dac_rvq_decode_paged(&codes, 1, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();

        for cb in 0..attrs.n_codebooks {
            let (k, _v) = cache.read_step(0, 0, 0, cb).expect("slot written");
            let low = tables[cb].row(cb as u32).unwrap();
            let mut want = vec![0.0_f32; attrs.d_model];
            project_accumulate(&projs[cb], low, &mut want);
            assert_eq!(k, want.as_slice(), "codebook slot cb={cb}");
        }
    }

    #[test]
    fn paged_multi_stream_write_isolation() {
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let dims = dac_paged_dims(&attrs, 2, 2);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();

        let codes0 = vec![0u32; attrs.n_codebooks];
        let codes1 = vec![3u32; attrs.n_codebooks];
        dac_rvq_decode_paged(&codes0, 1, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();
        dac_rvq_decode_paged(&codes1, 1, &tables, &projs, &attrs, 1, &mut cache, 0).unwrap();

        let s0 = dac_rvq_read_summed(&cache, &attrs, 0, 0).unwrap();
        let s1 = dac_rvq_read_summed(&cache, &attrs, 1, 0).unwrap();
        assert_ne!(s0, s1, "streams must not alias each other");

        // And each stream matches its own direct decode.
        let d0 = dac_rvq_decode(&codes0, 1, &tables, &projs, &attrs).unwrap();
        let d1 = dac_rvq_decode(&codes1, 1, &tables, &projs, &attrs).unwrap();
        assert_eq!(s0, d0);
        assert_eq!(s1, d1);
    }

    #[test]
    fn paged_rejects_bad_cache_shape_stream_and_overflow() {
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let codes = vec![0u32; attrs.n_codebooks];

        // Wrong d_head.
        let bad = KvDims {
            d_head: attrs.d_model + 1,
            ..dac_paged_dims(&attrs, 1, 2)
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(bad, BlockSize::Four).unwrap();
        assert!(matches!(
            dac_rvq_decode_paged(&codes, 1, &tables, &projs, &attrs, 0, &mut cache, 0),
            Err(VokraError::InvalidArgument(_))
        ));

        // Out-of-range stream.
        let dims = dac_paged_dims(&attrs, 1, 2);
        let mut cache2 = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();
        assert!(matches!(
            dac_rvq_decode_paged(&codes, 1, &tables, &projs, &attrs, 7, &mut cache2, 0),
            Err(VokraError::InvalidArgument(_))
        ));

        // time_start + time > max_time.
        assert!(matches!(
            dac_rvq_decode_paged(&codes, 1, &tables, &projs, &attrs, 0, &mut cache2, 2),
            Err(VokraError::InvalidArgument(_))
        ));

        // NOTE: `KvCacheExhausted` is surfaced verbatim from `append_step`
        // (see rustdoc) but is not constructible through this API with a
        // `pre_allocate`d cache — the arena is sized to cover `max_time`
        // exactly, and the axis checks above reject every out-of-arena write
        // first. The variant is exercised by the vokra-core paged tests.
    }

    // ---- T19: host-only fallback smoke -------------------------------------

    #[test]
    fn host_only_smoke_decode_end_to_end() {
        // The full DAC path — factorized decode + paged decode + summed read
        // — runs on the CPU with zero external dependencies and no GPU
        // anywhere (mirror of mimi_rvq's host_only_smoke; any GPU-only path
        // must stay an explicit opt-in, FR-EX-08).
        let attrs = DacRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            codebook_dim: 2,
            d_model: 4,
        };
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let time = 2;
        let codes = vec![0u32, 1, 2, 0];

        let flat = dac_rvq_decode(&codes, time, &tables, &projs, &attrs).unwrap();
        assert_eq!(flat.len(), time * attrs.d_model);

        let dims = dac_paged_dims(&attrs, 1, time);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();
        dac_rvq_decode_paged(&codes, time, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();
        for t in 0..time {
            let want = &flat[t * attrs.d_model..(t + 1) * attrs.d_model];
            let got = dac_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            assert_eq!(got, want);
        }
    }

    // ---- read_summed negative axes -----------------------------------------

    #[test]
    fn read_summed_rejects_bad_axes() {
        let attrs = tiny_attrs();
        let dims = dac_paged_dims(&attrs, 1, 2);
        let cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();
        assert!(matches!(
            dac_rvq_read_summed(&cache, &attrs, 5, 0),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            dac_rvq_read_summed(&cache, &attrs, 0, 99),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
