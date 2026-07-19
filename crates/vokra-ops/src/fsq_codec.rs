//! FSQ codec family decode — `wavtokenizer_vq` + `xcodec2_fsq`
//! (M4-16; FR-OP-31, **separate subgraph from the RVQ family**).
//!
//! # FSQ vs RVQ — the structural distinction (FR-OP-31 vs FR-OP-30)
//!
//! CLAUDE.md's audio-dialect chapter mandates that **RVQ (paged block size
//! 2-4 for 音声レート整合)** and **FSQ (65k+ vocab embedding, 単段 GEMV
//! bound)** be implemented as *separate subgraphs*. This module is the FSQ
//! side; [`crate::mimi_rvq`] / [`crate::dac_rvq`] / [`crate::encodec_rvq`]
//! are the RVQ side. The structural difference:
//!
//! - **RVQ (FR-OP-30)** — base + N residual codebooks, decoded by a
//!   **cross-codebook FP32 residual sum**
//!   (`decoded[t,:] = Σ_cb tables[cb].row(codes[t,cb])`), plus a paged
//!   `[time, stream, codebook]` variant (block size 2-4). Signature-level
//!   marker: the RVQ decodes take `&[CodebookTable]` (a *slice*).
//! - **FSQ family (FR-OP-31, this module)** — **single-stage**: there is
//!   **no cross-codebook residual-sum loop and no paged variant**.
//!   - [`wavtokenizer_vq_decode`] gathers rows from **one** large-vocab
//!     codebook (`&CodebookTable`, *singular* — not a slice).
//!   - [`xcodec2_fsq_decode`] has **no codebook tensor at all**: the code is
//!     decomposed onto an implicit per-dimension finite-scalar grid and
//!     projected up by one output GEMV.
//!
//! **No adapter between the families is provided** (混同ヘルパー禁止, M4-16
//! T06): feeding FSQ codes into `mimi_rvq_decode` / `dac_rvq_decode` (or RVQ
//! codes into these ops) is a modelling error, and the singular-vs-slice /
//! no-table signatures are the first line of defence. FSQ has no paged
//! (M3-03) dependency either — a single-stage decode has no residual streams
//! to page out (ADR M4-16 §D-a).
//!
//! # Upstream sources (nothing invented — ADR M4-16 §D-c, verified 2026-07-15)
//!
//! - **`wavtokenizer_vq`** — WavTokenizer (jishengpeng/WavTokenizer, MIT).
//!   The released configs run an EnCodec-style `ResidualVectorQuantizer`
//!   with `num_quantizers: 1`, `vq_bins: 4096`, `dimension=512`
//!   (`decoder/feature_extractors.py`; config
//!   `wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml`;
//!   released checkpoints `WavTokenizer-{small,medium}-*-24k-4096`). Decode
//!   is a raw embedding row gather with **no normalization** and an
//!   **Identity** `project_out` (`encoder/quantization/core_vq.py`:
//!   `EuclideanCodebook.dequantize = F.embedding(embed_ind, self.embed)`;
//!   `codebook_dim = default(codebook_dim, dim)`), and `n_q = 1` collapses
//!   the residual loop to a single lookup. **Honest note on "65k+ vocab"
//!   (FR-OP-31)**: the released WavTokenizer vocab is **4096**; the 65k+
//!   family headline is realized by the X-Codec 2 side (4^8 = 65536). The op
//!   is shape-generic and the 65k+ path is pinned by a 65,537-row synthetic
//!   test below.
//! - **`xcodec2_fsq`** — X-Codec 2 (HKUSTAudio/xcodec2, HF repo; PyPI
//!   `xcodec2==0.1.5` pins `vector-quantize-pytorch==1.17.8` +
//!   `torch==2.5.0`). `vq/codec_decoder_vocos.py` constructs
//!   `ResidualFSQ(dim = vq_dim /*= 2048 default, used as-is*/,
//!   levels = [4, 4, 4, 4, 4, 4, 4, 4], num_quantizers = 1)`; decode is
//!   `get_output_from_indices` (`modeling_xcodec2.py::decode_code`). At the
//!   pinned vector-quantize-pytorch 1.17.8:
//!
//!   ```text
//!     basis[d]       = Π_{k<d} levels[k]          (cumprod([1] + levels[:-1]))
//!     level_index[d] = (index / basis[d]) % levels[d]
//!     half_width[d]  = levels[d] / 2               (integer division)
//!     code[d]        = (level_index[d] − half_width[d]) / half_width[d]
//!     out[t, :]      = W_proj @ code + b_proj      (Linear len(levels) → dim,
//!                                                   bias=True; scale = (levels−1)^0 = 1
//!                                                   for the single quantizer)
//!   ```
//!
//!   (1.17.8 has **no** `preserve_symmetry`; the current upstream master
//!   defaults differ and are deliberately *not* used — the pin is what the
//!   released X-Codec 2 checkpoints decode with.) The downstream
//!   `fc_post_a` / Vocos backbone are the consumer model's layers, outside
//!   this op boundary (same cut as ADR M4-04 §D-g "features, not PCM").
//!
//! # Single-stage GEMV bound
//!
//! Both decodes are **single-stage**: per timestep, `wavtokenizer_vq` is one
//! large-table gather (the 65k+-vocab gather *is* the dominant cost) and
//! `xcodec2_fsq` is one implicit-grid decompose + one `d_model × n_dims`
//! GEMV. Sums are FP32-accumulated (the "BF16 mantissa loss is the real
//! problem" audio-dialect rule — no FP16/BF16 accumulator, NFR-QL-01).
//!
//! # No silent fallback (FR-EX-08)
//!
//! Out-of-range codes (`codes[t] >= vocab_size`, `codes[t] >=
//! Π levels`), shape mismatches, `levels` entries `< 2`, and `Π levels`
//! overflow are explicit [`VokraError::InvalidArgument`] — never a silent
//! clamp. A wrong codec index produces plausible-looking wrong audio
//! downstream, so decode time is where the error must surface.
//!
//! # Runtime functions — not `OpKind` variants
//!
//! Same rationale as [`crate::mimi_rvq`] (its module docs) / ADR M4-04 §D-b,
//! carried by ADR M4-16 §D-b: the heterogeneous signatures (`u32` codes +
//! table / projection operands → `Vec<f32>`) do not fit the
//! `OpValue::Real/Complex` dispatch surface, and the planned consumers
//! (future WavTokenizer / X-Codec 2 model WPs) are imperative models that
//! want the tight function API, not a graph-node round-trip (FR-EX-10
//! 精神).
//!
//! # GPU seam (Compute-seam awareness wired; kernels deferred)
//!
//! `vokra-models/src/compute.rs` exposes `HotOp::WavTokenizerVq` /
//! `HotOp::Xcodec2Fsq` whose CPU arms delegate here; the Metal / CUDA arms
//! return an explicit `VokraError::UnsupportedOp` until real kernels land
//! (FR-EX-08 — never a silent CPU fall back), and the `for_backend`
//! coverage gate rejects models listing these ops against Metal / CUDA /
//! Vulkan. Because FSQ is single-stage GEMV bound, the future kernels can
//! reuse the existing M2-01 (Metal MSL) / M2-03 (CUDA NVRTC) gemv + gather
//! kernels — simpler than the RVQ fold (no shared-memory tile to design).
//!
//! # GGUF metadata contract (documented — M4-16 T07; EXPERIMENTAL at M5-13)
//!
//! The converter of a future WavTokenizer / X-Codec 2 model WP bakes the
//! attrs below into `vokra.wavtokenizer.*` / `vokra.xcodec2.*` chunks (1:1
//! with the attr fields; see the attr struct docs). Both namespaces are
//! recorded in `docs/abi-changelog.md` ("GGUF Metadata additions", status
//! `documented`) with the intent to declare them **EXPERIMENTAL** at the
//! M5-13 C ABI / GGUF schema freeze (`docs/handoff/m4-12.md` §(e)-2), so
//! schema evolution stays legal at minor bumps until the codec API
//! stabilizes.

use vokra_core::{Result, VokraError};

use crate::mimi_rvq::CodebookTable;

// ---------------------------------------------------------------------------
// wavtokenizer_vq — single-stage large-vocab VQ lookup
// ---------------------------------------------------------------------------

/// Static shape attributes for a WavTokenizer single-codebook VQ decode.
///
/// A future WavTokenizer model WP's converter bakes these into the GGUF
/// metadata chunks (M4-16 T07, status `documented` — see the module docs):
///
/// - `vokra.wavtokenizer.vocab_size` (`u32`) ↔ [`Self::vocab_size`]
/// - `vokra.wavtokenizer.d_model` (`u32`) ↔ [`Self::d_model`]
///
/// The released WavTokenizer checkpoints use `vocab_size = 4096`,
/// `d_model = 512` (upstream `vq_bins: 4096` / `dimension=512` — module
/// docs source table); the op itself is shape-generic and handles the
/// FR-OP-31 "65k+ vocab" scale (see [`wavtokenizer_vq_decode`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavTokenizerVqAttrs {
    /// Number of codebook entries. Released WavTokenizer configs = 4096.
    pub vocab_size: usize,
    /// Embedding width per entry. Released WavTokenizer configs = 512.
    pub d_model: usize,
}

impl WavTokenizerVqAttrs {
    /// Builds the released 24 kHz WavTokenizer shape (4096 × 512) — the
    /// `WavTokenizer-{small,medium}-*-24k-4096` checkpoints (`vq_bins: 4096`,
    /// `dimension=512`; upstream sources in the module docs).
    ///
    /// Callers with a different variant build the struct field-by-field from
    /// their checkpoint's `vokra.wavtokenizer.*` metadata.
    #[inline]
    #[must_use]
    pub const fn wavtokenizer_24k_4096() -> Self {
        Self {
            vocab_size: 4096,
            d_model: 512,
        }
    }
}

/// Decodes `[time]` single-codebook WavTokenizer VQ codes into a
/// `[time, d_model]` row-major feature buffer:
/// `decoded[t, :] = codebook_table.row(codes[t])`.
///
/// **Single-stage** (FR-OP-31): exactly one gather per timestep from one
/// codebook — no cross-codebook residual sum (that is the RVQ family,
/// [`crate::mimi_rvq`]), which is why this signature takes a *singular*
/// [`CodebookTable`] where the RVQ decodes take `&[CodebookTable]`. The
/// gather is bit-exact (no arithmetic, no normalization — upstream decodes
/// via a raw `F.embedding` lookup, module docs).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of (FR-EX-08 — no silent clamp):
/// - zero-sized `attrs` axes;
/// - `codebook_table` shape ≠ `[vocab_size, d_model]`;
/// - `codes.len() != time`;
/// - `codes[t] >= vocab_size`.
///
/// # Example
///
/// ```
/// use vokra_ops::{CodebookTable, WavTokenizerVqAttrs, wavtokenizer_vq_decode};
///
/// let attrs = WavTokenizerVqAttrs { vocab_size: 3, d_model: 2 };
/// // Rows: [0,1], [10,11], [20,21].
/// let table = CodebookTable::new(3, 2, vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]).unwrap();
/// let out = wavtokenizer_vq_decode(&[2, 0], 2, &table, &attrs).unwrap();
/// assert_eq!(out, vec![20.0, 21.0, 0.0, 1.0]);
/// ```
pub fn wavtokenizer_vq_decode(
    codes: &[u32],
    time: usize,
    codebook_table: &CodebookTable,
    attrs: &WavTokenizerVqAttrs,
) -> Result<Vec<f32>> {
    if attrs.vocab_size == 0 || attrs.d_model == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "wavtokenizer_vq: attrs must have every axis > 0, got vocab_size={} d_model={}",
            attrs.vocab_size, attrs.d_model
        )));
    }
    if codebook_table.codebook_size != attrs.vocab_size || codebook_table.d_model != attrs.d_model {
        return Err(VokraError::InvalidArgument(format!(
            "wavtokenizer_vq: codebook_table shape [{},{}] != attrs [{},{}]",
            codebook_table.codebook_size, codebook_table.d_model, attrs.vocab_size, attrs.d_model
        )));
    }
    if codes.len() != time {
        return Err(VokraError::InvalidArgument(format!(
            "wavtokenizer_vq: codes.len() {} != time {time} (single codebook — one code per \
             timestep; the [time, n_codebooks] layout is the RVQ family's)",
            codes.len()
        )));
    }

    let d = attrs.d_model;
    let mut out = vec![0.0_f32; time * d];
    for (t, &code) in codes.iter().enumerate() {
        if (code as usize) >= attrs.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "wavtokenizer_vq: codes[{t}] = {code} >= vocab_size {} (no silent clamp — \
                 FR-EX-08)",
                attrs.vocab_size
            )));
        }
        // Single-stage gather: one row copy per timestep, no residual fold.
        out[t * d..(t + 1) * d].copy_from_slice(codebook_table.row(code)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// xcodec2_fsq — single-stage finite scalar quantization dequant
// ---------------------------------------------------------------------------

/// Static shape attributes for an X-Codec 2 FSQ dequant.
///
/// A future X-Codec 2 model WP's converter bakes these into the GGUF
/// metadata chunks (M4-16 T07, status `documented` — see the module docs):
///
/// - `vokra.xcodec2.levels` (`u32` array) ↔ [`Self::levels`]
/// - `vokra.xcodec2.d_model` (`u32`) ↔ [`Self::d_model`]
///
/// The released X-Codec 2 checkpoint uses `levels = [4; 8]` (effective
/// vocab 4^8 = 65536 — the FR-OP-31 "65k+ vocab") and `d_model = 2048`
/// (`vq_dim` default, used as-is by `modeling_xcodec2.py`); sources in the
/// module docs. Every level must be ≥ 2 (a 1-level dimension has
/// `half_width = 0` and cannot represent any grid).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xcodec2FsqAttrs {
    /// Quantization levels per code dimension (the FSQ levels tuple).
    /// X-Codec 2 released checkpoint = `[4, 4, 4, 4, 4, 4, 4, 4]`
    /// (upstream `vq/codec_decoder_vocos.py`).
    pub levels: Vec<u32>,
    /// Output feature width after the out-projection GEMV (= upstream
    /// `vq_dim`). X-Codec 2 released checkpoint = 2048.
    pub d_model: usize,
}

/// Validates an FSQ `levels` tuple and returns `Π levels` (the effective
/// vocab). Shared by [`Xcodec2FsqAttrs::effective_vocab`] and the
/// [`fsq_index_to_grid_codes`] wrapper so their validation can never diverge
/// — and so neither path has to allocate a `Vec` just to reach it (the old
/// `fsq_index_to_grid_codes` built a throwaway `Xcodec2FsqAttrs` via
/// `levels.to_vec()` on every call). Explicit error on an empty tuple, a
/// level `< 2`, or a product that overflows `usize` (FR-EX-08 — no silent
/// wrap).
fn effective_vocab_of(levels: &[u32]) -> Result<usize> {
    if levels.is_empty() {
        return Err(VokraError::InvalidArgument(
            "xcodec2_fsq: levels tuple must be non-empty".to_owned(),
        ));
    }
    let mut vocab = 1usize;
    for (d, &level) in levels.iter().enumerate() {
        if level < 2 {
            return Err(VokraError::InvalidArgument(format!(
                "xcodec2_fsq: levels[{d}] = {level} < 2 (half_width would be 0 — a 1-level \
                 dimension cannot represent any grid value)"
            )));
        }
        vocab = vocab.checked_mul(level as usize).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "xcodec2_fsq: Π levels overflows usize at levels[{d}] = {level} (no silent \
                 wrap — FR-EX-08)"
            ))
        })?;
    }
    Ok(vocab)
}

impl Xcodec2FsqAttrs {
    /// Builds the released X-Codec 2 shape (`levels = [4; 8]`,
    /// `d_model = 2048`) — verified from `vq/codec_decoder_vocos.py` +
    /// `modeling_xcodec2.py` (module docs source table).
    #[must_use]
    pub fn xcodec2() -> Self {
        Self {
            levels: vec![4; 8],
            d_model: 2048,
        }
    }

    /// Number of code dimensions (= `len(levels)`; upstream
    /// `codebook_dim = len(levels)` at vector-quantize-pytorch 1.17.8).
    #[inline]
    #[must_use]
    pub fn n_dims(&self) -> usize {
        self.levels.len()
    }

    /// Effective vocabulary size = `Π levels` (65536 for the released
    /// X-Codec 2). Explicit error on an empty tuple, a level < 2, or a
    /// product that overflows `usize` (FR-EX-08 — no silent wrap). Thin
    /// wrapper over [`effective_vocab_of`] (shared with the
    /// [`fsq_index_to_grid_codes`] validation path).
    pub fn effective_vocab(&self) -> Result<usize> {
        effective_vocab_of(&self.levels)
    }
}

/// The FSQ output projection: one Linear `n_dims → d_model` **with bias**
/// (upstream `ResidualFSQ.project_out = nn.Linear(codebook_dim, dim)` at the
/// pinned vector-quantize-pytorch 1.17.8; `requires_projection` because
/// `len(levels) = 8 != vq_dim = 2048`).
///
/// Same folded-weight layout discipline as [`crate::dac_rvq::DacOutProj`]
/// (row-major `[d_model, n_dims]`), but deliberately a **separate type**: the
/// FSQ family must not be plumbing-compatible with the RVQ per-quantizer
/// projections (no cross-family adapter — module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct FsqOutProj {
    /// Output width (rows of `weight`).
    pub d_model: usize,
    /// Input width (columns of `weight`) — must equal `len(levels)`.
    pub n_dims: usize,
    /// Row-major `[d_model, n_dims]` weight.
    pub weight: Vec<f32>,
    /// `[d_model]` bias (upstream `nn.Linear` default `bias=True`).
    pub bias: Vec<f32>,
}

impl FsqOutProj {
    /// Constructs a projection, validating both buffer lengths.
    pub fn new(d_model: usize, n_dims: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self> {
        if d_model == 0 || n_dims == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "FsqOutProj::new: d_model and n_dims must be > 0, got d_model={d_model} \
                 n_dims={n_dims}"
            )));
        }
        let expected_w = d_model * n_dims;
        if weight.len() != expected_w {
            return Err(VokraError::InvalidArgument(format!(
                "FsqOutProj::new: weight.len() {} != d_model * n_dims {expected_w}",
                weight.len()
            )));
        }
        if bias.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "FsqOutProj::new: bias.len() {} != d_model {d_model}",
                bias.len()
            )));
        }
        Ok(Self {
            d_model,
            n_dims,
            weight,
            bias,
        })
    }

    /// Row `o` of the weight (`n_dims` long).
    #[inline]
    #[must_use]
    pub fn weight_row(&self, o: usize) -> &[f32] {
        let base = o * self.n_dims;
        &self.weight[base..base + self.n_dims]
    }
}

/// Decomposes one FSQ `index` onto the implicit per-dimension grid,
/// writing `len(levels)` normalized values into `out`:
///
/// ```text
///   basis[d]       = Π_{k<d} levels[k]
///   level_index[d] = (index / basis[d]) % levels[d]
///   out[d]         = (level_index[d] − levels[d]/2) / (levels[d]/2)
/// ```
///
/// (integer `levels[d]/2` — the pinned vector-quantize-pytorch 1.17.8
/// `indices_to_level_indices` + `_scale_and_shift_inverse`, module docs.)
/// For `levels[d] = 4` the grid values are `{-1.0, -0.5, 0.0, 0.5}`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on an empty `levels`, a level < 2, a
/// `Π levels` overflow, `out.len() != levels.len()`, or
/// `index >= Π levels` (FR-EX-08).
pub fn fsq_index_to_grid_codes(index: u32, levels: &[u32], out: &mut [f32]) -> Result<()> {
    // Public validating entry point for callers that do not already hold a
    // validated `Π levels`. Validate the borrowed slice directly through
    // `effective_vocab_of` — the same path the attrs use, so the two never
    // diverge — with no `levels.to_vec()` (the old body built a throwaway
    // `Xcodec2FsqAttrs` here on every call). The hot loop in
    // `xcodec2_fsq_decode` never reaches this wrapper: it computes `vocab`
    // once and calls `fsq_index_to_grid_codes_into` per frame.
    let vocab = effective_vocab_of(levels)?;
    if out.len() != levels.len() {
        return Err(VokraError::InvalidArgument(format!(
            "fsq_index_to_grid_codes: out.len() {} != levels.len() {}",
            out.len(),
            levels.len()
        )));
    }
    if (index as usize) >= vocab {
        return Err(VokraError::InvalidArgument(format!(
            "fsq_index_to_grid_codes: index {index} >= Π levels {vocab} (no silent clamp — \
             FR-EX-08)"
        )));
    }
    fsq_index_to_grid_codes_into(index, levels, vocab, out);
    Ok(())
}

/// Grid-decomposes `index` into `out`, **assuming the tuple is already
/// validated** — the allocation-free hot path for [`xcodec2_fsq_decode`],
/// which computes `vocab = Π levels` (validating `levels`) exactly once
/// before the per-timestep loop rather than re-deriving it for every frame
/// (the previous code re-ran `levels.to_vec()` + `effective_vocab` per frame).
///
/// Contract the caller MUST uphold — enforced by [`fsq_index_to_grid_codes`]
/// for external callers, and by `xcodec2_fsq_decode`'s once-per-decode
/// [`effective_vocab_of`] validation plus its per-frame `code < vocab` guard:
/// - `levels` is non-empty with every entry `>= 2` (so `half_width >= 1`);
/// - `vocab == Π levels` from a prior [`effective_vocab_of`] on the *same*
///   `levels` — trusted here, not recomputed (and no `levels.to_vec()`);
/// - `out.len() == levels.len()`;
/// - `index < vocab`.
///
/// The invariants are `debug_assert!`ed — a contract guard in debug/test
/// builds, compiled out of release so the runtime bounds check lives *once*
/// in the caller instead of being doubled here (FR-EX-08 is satisfied by the
/// caller's single explicit check). Infallible by construction.
fn fsq_index_to_grid_codes_into(index: u32, levels: &[u32], vocab: usize, out: &mut [f32]) {
    debug_assert_eq!(
        out.len(),
        levels.len(),
        "fsq_index_to_grid_codes_into: caller must size out to len(levels)"
    );
    debug_assert!(
        (index as usize) < vocab,
        "fsq_index_to_grid_codes_into: caller must guarantee index < Π levels"
    );

    // Mixed-radix decompose + normalize (vector-quantize-pytorch 1.17.8
    // `indices_to_level_indices` + `_scale_and_shift_inverse`; module docs).
    // `basis` is folded into `rem` instead of materializing the cumprod:
    // walking dims low→high, `rem / basis[d] % levels[d]` == `rem % levels[d]`
    // followed by `rem /= levels[d]` — same integers, no alloc.
    let mut rem = index as usize;
    for (d, &level) in levels.iter().enumerate() {
        let level = level as usize;
        let level_index = rem % level;
        rem /= level;
        let half_width = level / 2; // integer division (>= 1: level >= 2 validated)
        out[d] = (level_index as f32 - half_width as f32) / half_width as f32;
    }
}

/// Decodes `[time]` X-Codec 2 FSQ codes into a `[time, d_model]` row-major
/// feature buffer: per timestep, the index is decomposed onto the implicit
/// per-dimension grid ([`fsq_index_to_grid_codes`]) and projected by **one**
/// GEMV (`out[t, :] = W @ grid + b`, FP32 accumulator) — the FR-OP-31
/// "single-stage GEMV bound". No residual loop, no codebook tensor, no
/// paged variant (the RVQ family is [`crate::mimi_rvq`]).
///
/// `out_proj = None` mirrors the upstream `requires_projection = false`
/// Identity case and requires `attrs.d_model == attrs.n_dims()` (explicit
/// error otherwise). The released X-Codec 2 always projects
/// (8 → 2048), so real-checkpoint callers pass `Some`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of (FR-EX-08):
/// - invalid `attrs` (empty `levels`, level < 2, `Π levels` overflow,
///   `d_model == 0`);
/// - `codes.len() != time`;
/// - `codes[t] >= Π levels`;
/// - projection shape mismatch (`n_dims`, `d_model`), or `None` with
///   `d_model != n_dims()`.
///
/// # Example
///
/// ```
/// use vokra_ops::{Xcodec2FsqAttrs, xcodec2_fsq_decode};
///
/// // Identity-projection case: d_model == n_dims == 2, levels [4, 4].
/// let attrs = Xcodec2FsqAttrs { levels: vec![4, 4], d_model: 2 };
/// // index 7 → level indices (3, 1) → grid (0.5, -0.5).
/// let out = xcodec2_fsq_decode(&[7], 1, None, &attrs).unwrap();
/// assert_eq!(out, vec![0.5, -0.5]);
/// ```
pub fn xcodec2_fsq_decode(
    codes: &[u32],
    time: usize,
    out_proj: Option<&FsqOutProj>,
    attrs: &Xcodec2FsqAttrs,
) -> Result<Vec<f32>> {
    let vocab = attrs.effective_vocab()?; // validates levels (non-empty, >= 2, no overflow)
    let n_dims = attrs.n_dims();
    if attrs.d_model == 0 {
        return Err(VokraError::InvalidArgument(
            "xcodec2_fsq: attrs.d_model must be > 0".to_owned(),
        ));
    }
    match out_proj {
        Some(proj) => {
            if proj.n_dims != n_dims || proj.d_model != attrs.d_model {
                return Err(VokraError::InvalidArgument(format!(
                    "xcodec2_fsq: out_proj shape [{},{}] != attrs [d_model={}, n_dims={n_dims}]",
                    proj.d_model, proj.n_dims, attrs.d_model
                )));
            }
        }
        None => {
            // Mirrors upstream `requires_projection = codebook_dim != dim`:
            // Identity is only legal when the two widths agree.
            if attrs.d_model != n_dims {
                return Err(VokraError::InvalidArgument(format!(
                    "xcodec2_fsq: out_proj = None (Identity) requires d_model == len(levels), \
                     got d_model={} len(levels)={n_dims} — the released X-Codec 2 projects \
                     8 → 2048 and must pass Some(&FsqOutProj)",
                    attrs.d_model
                )));
            }
        }
    }
    if codes.len() != time {
        return Err(VokraError::InvalidArgument(format!(
            "xcodec2_fsq: codes.len() {} != time {time} (single-stage — one code per timestep; \
             the [time, n_codebooks] layout is the RVQ family's)",
            codes.len()
        )));
    }

    let d_model = attrs.d_model;
    let mut out = vec![0.0_f32; time * d_model];
    let mut grid = vec![0.0_f32; n_dims];
    for (t, &code) in codes.iter().enumerate() {
        // Single per-frame bounds check against the once-computed `vocab`
        // (FR-EX-08 — no silent clamp; the timestep `t` is named here for the
        // diagnostic). The decompose then goes through the allocation-free
        // inner: no per-frame `levels.to_vec()`, no `effective_vocab`
        // revalidation, and no second (doubled) bounds check.
        if (code as usize) >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "xcodec2_fsq: codes[{t}] = {code} >= Π levels {vocab} (no silent clamp — \
                 FR-EX-08)"
            )));
        }
        fsq_index_to_grid_codes_into(code, &attrs.levels, vocab, &mut grid);
        let row = &mut out[t * d_model..(t + 1) * d_model];
        match out_proj {
            Some(proj) => {
                // Single-stage GEMV, FP32 accumulator (module docs — no
                // FP16/BF16 fold): row[o] = bias[o] + Σ_d W[o, d] * grid[d].
                for (o, dst) in row.iter_mut().enumerate() {
                    let w_row = proj.weight_row(o);
                    let mut y = proj.bias[o];
                    for (w, g) in w_row.iter().zip(grid.iter()) {
                        y += *w * *g;
                    }
                    *dst = y;
                }
            }
            None => row.copy_from_slice(&grid),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------ helpers ---------------------------------------------------

    /// Deterministic ramp table: row `i` is `[i, i+1, ..., i+d-1]` as f32.
    fn ramp_table(vocab: usize, d: usize) -> CodebookTable {
        let mut data = vec![0.0_f32; vocab * d];
        for i in 0..vocab {
            for j in 0..d {
                data[i * d + j] = (i + j) as f32;
            }
        }
        CodebookTable::new(vocab, d, data).unwrap()
    }

    // ---- T02: attrs / signature ------------------------------------------

    #[test]
    fn wavtokenizer_attrs_canonical_matches_released_config() {
        // vq_bins: 4096 / dimension=512 — upstream sources in module docs.
        let a = WavTokenizerVqAttrs::wavtokenizer_24k_4096();
        assert_eq!(a.vocab_size, 4096);
        assert_eq!(a.d_model, 512);
    }

    // ---- T03: single-stage lookup, bit-identical + bounds -----------------

    #[test]
    fn wavtokenizer_lookup_returns_expected_rows_bit_identical() {
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 6,
            d_model: 4,
        };
        let table = ramp_table(attrs.vocab_size, attrs.d_model);
        let codes = vec![2u32, 0, 5];
        let got = wavtokenizer_vq_decode(&codes, 3, &table, &attrs).unwrap();
        // Hand oracle: rows 2, 0, 5 of the ramp — single gather per timestep,
        // no residual fold (the RVQ sum loop must NOT exist here).
        let mut want = Vec::new();
        for &c in &codes {
            want.extend_from_slice(table.row(c).unwrap());
        }
        assert_eq!(got, want, "gather must be bit-identical (pure lookup)");
    }

    #[test]
    fn wavtokenizer_lookup_is_per_timestep_independent() {
        // Single-stage property: decoding [c0, c1] equals decoding [c0] and
        // [c1] separately (no cross-timestep / cross-codebook coupling).
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 4,
            d_model: 3,
        };
        let table = ramp_table(4, 3);
        let joint = wavtokenizer_vq_decode(&[1, 3], 2, &table, &attrs).unwrap();
        let a = wavtokenizer_vq_decode(&[1], 1, &table, &attrs).unwrap();
        let b = wavtokenizer_vq_decode(&[3], 1, &table, &attrs).unwrap();
        assert_eq!(joint[..3], a[..]);
        assert_eq!(joint[3..], b[..]);
    }

    #[test]
    fn wavtokenizer_rejects_out_of_range_and_shape_mismatch() {
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 6,
            d_model: 4,
        };
        let table = ramp_table(6, 4);
        // Out-of-range code (== vocab_size) — silent clamp forbidden.
        assert!(matches!(
            wavtokenizer_vq_decode(&[6], 1, &table, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // codes.len() != time.
        assert!(matches!(
            wavtokenizer_vq_decode(&[0, 1], 1, &table, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Table shape != attrs.
        let small = ramp_table(5, 4);
        assert!(matches!(
            wavtokenizer_vq_decode(&[0], 1, &small, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        let narrow = ramp_table(6, 3);
        assert!(matches!(
            wavtokenizer_vq_decode(&[0], 1, &narrow, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Zero-axis attrs.
        let zero = WavTokenizerVqAttrs {
            vocab_size: 0,
            d_model: 4,
        };
        assert!(matches!(
            wavtokenizer_vq_decode(&[0], 1, &table, &zero),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn wavtokenizer_65k_plus_vocab_path_is_exact() {
        // FR-OP-31 pins the family at "65k+ vocab embedding, 単段 GEMV
        // bound". The committed fixtures stay small, so the 65k+ scale is
        // pinned here with an in-memory synthetic table: vocab 65537
        // (= X-Codec 2's 65536 + 1 so the extreme row is asymmetric),
        // d_model 4. Row i col j = (i*31 + j) mod 1009 — deterministic and
        // cheap (~1 MB).
        let vocab = 65_537usize;
        let d = 4usize;
        let mut data = vec![0.0_f32; vocab * d];
        for i in 0..vocab {
            for j in 0..d {
                data[i * d + j] = ((i as u64 * 31 + j as u64) % 1009) as f32;
            }
        }
        let table = CodebookTable::new(vocab, d, data).unwrap();
        let attrs = WavTokenizerVqAttrs {
            vocab_size: vocab,
            d_model: d,
        };
        // Highest valid row: 65536 → (65536*31 + j) % 1009 = 499 + j.
        let got = wavtokenizer_vq_decode(&[65_536], 1, &table, &attrs).unwrap();
        assert_eq!(got, vec![499.0, 500.0, 501.0, 502.0]);
        // One past the end is an explicit error (FR-EX-08).
        assert!(matches!(
            wavtokenizer_vq_decode(&[65_537], 1, &table, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T04: xcodec2 attrs ------------------------------------------------

    #[test]
    fn xcodec2_attrs_canonical_matches_released_checkpoint() {
        // levels [4; 8] / vq_dim 2048 — upstream sources in module docs.
        let a = Xcodec2FsqAttrs::xcodec2();
        assert_eq!(a.levels, vec![4u32; 8]);
        assert_eq!(a.n_dims(), 8);
        assert_eq!(a.d_model, 2048);
        assert_eq!(a.effective_vocab().unwrap(), 65_536);
    }

    #[test]
    fn xcodec2_effective_vocab_rejects_degenerate_and_overflowing_levels() {
        // Empty tuple.
        let empty = Xcodec2FsqAttrs {
            levels: vec![],
            d_model: 4,
        };
        assert!(matches!(
            empty.effective_vocab(),
            Err(VokraError::InvalidArgument(_))
        ));
        // A 1-level dimension has half_width = 0 (division by zero in the
        // dequant) — explicit error, not a NaN.
        let one = Xcodec2FsqAttrs {
            levels: vec![4, 1],
            d_model: 4,
        };
        assert!(matches!(
            one.effective_vocab(),
            Err(VokraError::InvalidArgument(_))
        ));
        // Π levels overflows usize — explicit error, no silent wrap.
        let huge = Xcodec2FsqAttrs {
            levels: vec![u32::MAX; 5],
            d_model: 4,
        };
        assert!(matches!(
            huge.effective_vocab(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T05: grid decompose + GEMV ---------------------------------------

    #[test]
    fn fsq_grid_decompose_matches_hand_math_even_levels() {
        // levels [4, 4]: basis [1, 4], half_width [2, 2].
        let levels = [4u32, 4];
        let mut out = [0.0_f32; 2];
        // index 0 → (0, 0) → (-1, -1).
        fsq_index_to_grid_codes(0, &levels, &mut out).unwrap();
        assert_eq!(out, [-1.0, -1.0]);
        // index 7 → (7 % 4, 7 / 4) = (3, 1) → (0.5, -0.5).
        fsq_index_to_grid_codes(7, &levels, &mut out).unwrap();
        assert_eq!(out, [0.5, -0.5]);
        // index 15 (max) → (3, 3) → (0.5, 0.5).
        fsq_index_to_grid_codes(15, &levels, &mut out).unwrap();
        assert_eq!(out, [0.5, 0.5]);
        // Grid values for L=4 are exactly {-1.0, -0.5, 0.0, 0.5}.
    }

    #[test]
    fn fsq_grid_decompose_matches_hand_math_odd_levels() {
        // levels [3, 5]: basis [1, 3], half_width [1, 2].
        let levels = [3u32, 5];
        let mut out = [0.0_f32; 2];
        // index 14 → (14 % 3, 14 / 3) = (2, 4) → ((2-1)/1, (4-2)/2) = (1, 1).
        fsq_index_to_grid_codes(14, &levels, &mut out).unwrap();
        assert_eq!(out, [1.0, 1.0]);
        // index 4 → (1, 1) → (0.0, -0.5).
        fsq_index_to_grid_codes(4, &levels, &mut out).unwrap();
        assert_eq!(out, [0.0, -0.5]);
        // index 0 → (-1, -1).
        fsq_index_to_grid_codes(0, &levels, &mut out).unwrap();
        assert_eq!(out, [-1.0, -1.0]);
    }

    #[test]
    fn fsq_grid_decompose_rejects_out_of_range_and_bad_shapes() {
        let levels = [4u32, 4];
        let mut out = [0.0_f32; 2];
        // index == Π levels is out of range.
        assert!(matches!(
            fsq_index_to_grid_codes(16, &levels, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        // out width mismatch.
        let mut narrow = [0.0_f32; 1];
        assert!(matches!(
            fsq_index_to_grid_codes(0, &levels, &mut narrow),
            Err(VokraError::InvalidArgument(_))
        ));
        // level < 2.
        assert!(matches!(
            fsq_index_to_grid_codes(0, &[4, 1], &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        // empty levels.
        assert!(matches!(
            fsq_index_to_grid_codes(0, &[], &mut []),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn xcodec2_decode_with_projection_matches_hand_fold() {
        // levels [4, 4] (n_dims 2) → d_model 3. index 7 → grid (0.5, -0.5).
        // W = [[1,2],[3,4],[5,6]], b = [0.5, -0.5, 0.25]:
        //   out = [0.5 + (0.5 - 1.0), -0.5 + (1.5 - 2.0), 0.25 + (2.5 - 3.0)]
        //       = [0.0, -1.0, -0.25]
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 3,
        };
        let proj = FsqOutProj::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![0.5, -0.5, 0.25],
        )
        .unwrap();
        let got = xcodec2_fsq_decode(&[7], 1, Some(&proj), &attrs).unwrap();
        assert_eq!(got, vec![0.0, -1.0, -0.25]);
    }

    #[test]
    fn xcodec2_decode_identity_projection_passes_grid_through() {
        // None mirrors upstream requires_projection = false (Identity):
        // d_model must equal n_dims.
        let attrs = Xcodec2FsqAttrs {
            levels: vec![3, 5],
            d_model: 2,
        };
        let got = xcodec2_fsq_decode(&[14, 0, 4], 3, None, &attrs).unwrap();
        assert_eq!(got, vec![1.0, 1.0, -1.0, -1.0, 0.0, -0.5]);
    }

    #[test]
    fn xcodec2_decode_is_single_stage_per_timestep() {
        // Single-stage property (the RVQ residual loop must not exist):
        // decoding a batch equals per-timestep decodes concatenated.
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 2,
        };
        let joint = xcodec2_fsq_decode(&[7, 15], 2, None, &attrs).unwrap();
        let a = xcodec2_fsq_decode(&[7], 1, None, &attrs).unwrap();
        let b = xcodec2_fsq_decode(&[15], 1, None, &attrs).unwrap();
        assert_eq!(joint[..2], a[..]);
        assert_eq!(joint[2..], b[..]);
    }

    #[test]
    fn xcodec2_decode_rejects_bad_inputs() {
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 3,
        };
        let proj = FsqOutProj::new(3, 2, vec![0.0; 6], vec![0.0; 3]).unwrap();
        // Out-of-range code (>= 16) — explicit error (FR-EX-08).
        assert!(matches!(
            xcodec2_fsq_decode(&[16], 1, Some(&proj), &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // codes.len() != time.
        assert!(matches!(
            xcodec2_fsq_decode(&[0, 1], 1, Some(&proj), &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Projection n_dims mismatch.
        let wrong_in = FsqOutProj::new(3, 4, vec![0.0; 12], vec![0.0; 3]).unwrap();
        assert!(matches!(
            xcodec2_fsq_decode(&[0], 1, Some(&wrong_in), &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Projection d_model mismatch vs attrs.
        let wrong_out = FsqOutProj::new(4, 2, vec![0.0; 8], vec![0.0; 4]).unwrap();
        assert!(matches!(
            xcodec2_fsq_decode(&[0], 1, Some(&wrong_out), &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // None (Identity) with d_model != n_dims.
        assert!(matches!(
            xcodec2_fsq_decode(&[0], 1, None, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // d_model = 0.
        let zero = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 0,
        };
        assert!(matches!(
            xcodec2_fsq_decode(&[0], 1, None, &zero),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn fsq_out_proj_new_validates_shape() {
        assert!(matches!(
            FsqOutProj::new(3, 2, vec![0.0; 5], vec![0.0; 3]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            FsqOutProj::new(3, 2, vec![0.0; 6], vec![0.0; 2]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            FsqOutProj::new(0, 2, vec![], vec![]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T13: host-only smoke (explicit CPU, zero external deps) ----------

    #[test]
    fn host_only_smoke_fsq_family_end_to_end() {
        // Both FSQ-family decodes run to completion on the host with zero
        // external dependencies — the explicit-CPU baseline every backend
        // must match (FR-EX-08: a GPU path is an explicit opt-in, never a
        // silent substitute). Mirrors mimi_rvq's host_only_smoke.
        let wt_attrs = WavTokenizerVqAttrs {
            vocab_size: 8,
            d_model: 3,
        };
        let table = ramp_table(8, 3);
        let wt = wavtokenizer_vq_decode(&[0, 7, 3], 3, &table, &wt_attrs).unwrap();
        assert_eq!(wt.len(), 3 * wt_attrs.d_model);

        let fsq_attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4, 4],
            d_model: 2,
        };
        let proj = FsqOutProj::new(2, 3, vec![0.5; 6], vec![0.0; 2]).unwrap();
        let fs = xcodec2_fsq_decode(&[0, 63, 21], 3, Some(&proj), &fsq_attrs).unwrap();
        assert_eq!(fs.len(), 3 * fsq_attrs.d_model);
    }

    // ---- alloc-refactor guards: inner fast-path == public validating path -

    #[test]
    fn fsq_inner_and_public_wrapper_agree_over_full_vocab() {
        // The refactor split the grid decompose into a private inner
        // (`fsq_index_to_grid_codes_into`, taking the once-computed `vocab`)
        // and kept the public `fsq_index_to_grid_codes` as a validating
        // wrapper. Both must produce identical grids for every valid index —
        // the inner drops the per-call `levels.to_vec()` + `effective_vocab`
        // revalidation, never the arithmetic. `effective_vocab_of` is the
        // shared borrow-slice validator both paths reach.
        let levels = [3u32, 4, 5]; // Π levels = 60
        let vocab = effective_vocab_of(&levels).unwrap();
        assert_eq!(vocab, 60);
        let mut via_inner = [0.0_f32; 3];
        let mut via_wrapper = [0.0_f32; 3];
        for index in 0..vocab as u32 {
            fsq_index_to_grid_codes_into(index, &levels, vocab, &mut via_inner);
            fsq_index_to_grid_codes(index, &levels, &mut via_wrapper).unwrap();
            assert_eq!(
                via_inner, via_wrapper,
                "inner fast-path diverged from the public wrapper at index {index}",
            );
        }
    }

    #[test]
    fn xcodec2_decode_hot_path_output_unchanged_vs_per_frame_wrapper() {
        // Property test (M4-16 alloc-refactor): the batched decode — whose hot
        // loop now calls the private inner with a once-computed `vocab` — must
        // be bit-identical to the *pre-refactor* path, which decoded each frame
        // through the public `fsq_index_to_grid_codes` wrapper and the same
        // sequential FP32 GEMV. Reconstruct that old path and assert exact
        // equality, so removing the per-frame alloc + revalidation changed
        // nothing observable (no fabricated tolerance — exact `==`).
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4, 4, 4],
            d_model: 5,
        };
        let vocab = attrs.effective_vocab().unwrap(); // 256
        let n_dims = attrs.n_dims();
        let d_model = attrs.d_model;

        // Deterministic synthetic projection + codes (SplitMix64, zero-dep).
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let f = |v: u64| (v % 4000) as f32 / 1000.0 - 2.0; // ~[-2, 2)
        let weight: Vec<f32> = (0..d_model * n_dims).map(|_| f(next())).collect();
        let bias: Vec<f32> = (0..d_model).map(|_| f(next())).collect();
        let proj = FsqOutProj::new(d_model, n_dims, weight, bias).unwrap();

        let time = 128;
        let codes: Vec<u32> = (0..time).map(|_| (next() % vocab as u64) as u32).collect();

        let got = xcodec2_fsq_decode(&codes, time, Some(&proj), &attrs).unwrap();

        // Pre-refactor reconstruction: public wrapper per frame + hand GEMV.
        let mut want = vec![0.0_f32; time * d_model];
        let mut grid = vec![0.0_f32; n_dims];
        for (t, &code) in codes.iter().enumerate() {
            fsq_index_to_grid_codes(code, &attrs.levels, &mut grid).unwrap();
            for o in 0..d_model {
                let w_row = proj.weight_row(o);
                let mut y = proj.bias[o];
                for (w, g) in w_row.iter().zip(grid.iter()) {
                    y += *w * *g;
                }
                want[t * d_model + o] = y;
            }
        }
        assert_eq!(
            got, want,
            "hot-path (inner, vocab computed once) output must be bit-identical to the \
             pre-refactor per-frame public-wrapper path",
        );
    }
}
