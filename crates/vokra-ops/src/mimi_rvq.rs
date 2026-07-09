//! Mimi (Kyutai) residual vector quantization (RVQ) codec decode
//! (M3-06; FR-OP-30, first RVQ family op).
//!
//! # Op contract
//!
//! Given
//!
//! - `codes` — a `[time, n_codebooks]` row-major slice of `u32` codebook
//!   indices (one index per (timestep, codebook) pair);
//! - `codebook_tables` — one [`CodebookTable`] per codebook, each shaped
//!   `[codebook_size, d_model]` row-major;
//!
//! `mimi_rvq_decode` returns a `[time, d_model]` row-major `Vec<f32>` of
//! feature vectors reconstructed by summing every codebook's contribution:
//!
//! ```text
//!   decoded[t, :] = sum_{cb=0..n_codebooks} codebook_tables[cb].row(codes[t, cb])
//! ```
//!
//! The sum is done in FP32 (`f32` accumulator) — the "BF16 mantissa loss is
//! the real problem" note in the CLAUDE.md audio-dialect chapter applies to
//! any codec-side accumulator, so we never fold in FP16 / BF16 (all-FP32
//! path is the only path shipped from M3-06).
//!
//! # Paged variant (Mimi 12.5 Hz → `block_size=2`)
//!
//! [`mimi_rvq_decode_paged`] writes **per-codebook, pre-sum** features into
//! a [`PagedKvCache<f32>`] using its `[time, stream, codebook]` addressing.
//! The paged cache row width is one `d_model`-long feature (`n_head=1`,
//! `d_head=d_model` in [`KvDims`]); the K side of the KV row stores the
//! feature and the V side is zeroed. This matches the M3-03 layout contract
//! and keeps the codebook axis contiguous inside a page (M3-06 T07 assert).
//!
//! Two block sizes are supported:
//!
//! - [`BlockSize::Two`] — **primary** for Mimi (12.5 Hz → 160 ms per block).
//! - [`BlockSize::Four`] — variant for 25 / 50 Hz codecs (DAC, X-Codec 2,
//!   WavTokenizer). M4+ consumers.
//!
//! [`mimi_rvq_read_summed`] reads a per-timestep-per-stream contiguous span
//! across codebooks and returns the residual sum — the mirror of the direct
//! `mimi_rvq_decode` output.
//!
//! # Consumer stub — [`MimiDecoder`]
//!
//! [`MimiDecoder`] is a host-side session-like helper that owns the codebook
//! tables. It is the standalone stand-in for the future
//! `Session::mimi_decode` (M3-09 wires the same shape into
//! `vokra_core::Session`). M3-06 keeps everything in `vokra-ops` so the crate
//! dependency edge does not have to reverse.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Any out-of-range index (`codes[t, cb] >= attrs.codebook_size`), shape
//! mismatch, or paged-cache mis-shape is an explicit
//! [`VokraError::InvalidArgument`] — never a silent 0 clamp or garbage row.
//! A wrong RVQ index causes Flow Matching to diverge silently downstream, so
//! surfacing the error at decode time is safer than producing plausible-
//! looking wrong audio.
//!
//! # Runtime function — not an `OpKind` variant (same rationale as
//! `flow_sampler`)
//!
//! `mimi_rvq_decode` / `mimi_rvq_decode_paged` are runtime functions, not
//! [`vokra_core::OpKind`] variants. Two reasons:
//!
//! 1. **Live state**: the paged variant writes into a `&mut PagedKvCache`;
//!    the `OpKind` dispatch surface (`OpValue::Real` / `OpValue::Complex`)
//!    has no place for a borrowed cache handle. Threading one through
//!    every `dispatch` call site just to serve this op would tax every
//!    other op.
//! 2. **Consumer shape**: the only planned consumer (CosyVoice2, M3-09) is
//!    an imperative model that already threads its own compute seam; it
//!    wants the tight `MimiDecoder` API, not a graph-node round-trip.
//!
//! Same design pattern as [`crate::flow_sampler`] (FR-EX-10 精神). See ADR
//! M3-06 §D4 for the paged-cache re-use rationale.
//!
//! # GPU seam (Wave 5 follow-up — TODO)
//!
//! The Metal / CUDA `mimi_rvq` kernels are deferred to Wave 5 (see the M3-06
//! T14 / T15 tickets). The imperative `Compute` seam in
//! `vokra-models/src/compute.rs` does not yet know about this op. When it
//! lands, the same "one kernel per (backend, op)" pattern the M2 seam uses
//! applies here:
//!
//! - Add a `HotOp::MimiRvq` arm to `vokra-models/src/compute.rs`;
//! - Add `Compute::mimi_rvq_f32(...)` returning `Result<Vec<f32>>` that
//!   dispatches to `kernels::mimi_rvq_f32` on CPU, `MetalContext::mimi_rvq`
//!   on Metal (M2-01 pattern — MSL kernel `vokra_mimi_rvq_f32.metal`), and
//!   `libcuda`-loaded PTX (`vokra_mimi_rvq_f32.cu`) on CUDA (M2-03 pattern
//!   — NVRTC compile via `CudaContext::compile_kernel`);
//! - RVQ is embedding-lookup + FP32 fold, so a naive `blockDim.x = d_model,
//!   gridDim.x = time, gridDim.y = n_stream` layout is enough; there is no
//!   shared-memory tile to design (Wave 5 T14 rationale).
//! - Update `HotOp::covered_by_metal` / `covered_by_cuda` accordingly; the
//!   `metal_coverage_is_consistent` / `cuda_coverage_is_consistent` tests
//!   pin the coverage table.
//!
//! Silent host fallback is forbidden (FR-EX-08); an incomplete GPU arm
//! must remain `UnsupportedOp` until the kernel lands.
//!
//! [`PagedKvCache<f32>`]: vokra_core::cache::paged::PagedKvCache
//! [`KvDims`]: vokra_core::cache::paged::KvDims

use vokra_core::cache::paged::{KvDims, PagedKvCache};
use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Op attributes
// ---------------------------------------------------------------------------

/// Static shape attributes for a Mimi RVQ decode.
///
/// The three numbers are baked into the model checkpoint by the M3-09
/// converter (proposed metadata chunks `vokra.mimi.n_codebooks` /
/// `vokra.mimi.codebook_size` / `vokra.mimi.d_model`); at decode time the
/// runtime just consumes them here. Mimi's canonical shape is
/// `n_codebooks = 8`, `codebook_size = 2048`, `d_model = 512`, but the op
/// itself is shape-generic — any RVQ codec whose codebooks share
/// `[codebook_size, d_model]` slots into the same code path (this is why
/// FR-OP-30 groups Mimi / DAC / X-Codec 2 / WavTokenizer under one op family).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimiRvqAttrs {
    /// Number of codebooks (base + residuals). Mimi = 8.
    pub n_codebooks: usize,
    /// Number of entries per codebook (each row is a `d_model` feature).
    /// Mimi = 2048.
    pub codebook_size: usize,
    /// Feature dimension per codebook entry. Mimi = 512.
    pub d_model: usize,
}

impl MimiRvqAttrs {
    /// Builds a canonical Mimi-shaped attribute (8 × 2048 × 512).
    ///
    /// Callers with a different RVQ codec build the struct field-by-field.
    #[inline]
    #[must_use]
    pub const fn mimi() -> Self {
        Self {
            n_codebooks: 8,
            codebook_size: 2048,
            d_model: 512,
        }
    }
}

/// One codebook table, shape `[codebook_size, d_model]` row-major.
///
/// Owns the row data as a flat `Vec<f32>` so the runtime can populate it
/// from a GGUF tensor slice without copying. The M3-09 converter is the
/// intended producer.
#[derive(Debug, Clone, PartialEq)]
pub struct CodebookTable {
    /// Number of entries (rows).
    pub codebook_size: usize,
    /// Feature width (columns).
    pub d_model: usize,
    /// Row-major `[codebook_size, d_model]` data.
    pub data: Vec<f32>,
}

impl CodebookTable {
    /// Constructs a table, validating that `data.len() == codebook_size *
    /// d_model` and both axes are non-zero.
    pub fn new(codebook_size: usize, d_model: usize, data: Vec<f32>) -> Result<Self> {
        if codebook_size == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "CodebookTable::new: codebook_size and d_model must be > 0, got \
                 codebook_size={codebook_size} d_model={d_model}"
            )));
        }
        let expected = codebook_size * d_model;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "CodebookTable::new: data.len() {} != codebook_size * d_model {expected}",
                data.len()
            )));
        }
        Ok(Self {
            codebook_size,
            d_model,
            data,
        })
    }

    /// Returns the `d_model`-long row at `index`, or an explicit error if the
    /// index is out of range (FR-EX-08 — no silent clamp to 0).
    #[inline]
    pub fn row(&self, index: u32) -> Result<&[f32]> {
        let idx = index as usize;
        if idx >= self.codebook_size {
            return Err(VokraError::InvalidArgument(format!(
                "CodebookTable::row: index {idx} >= codebook_size {}",
                self.codebook_size
            )));
        }
        let base = idx * self.d_model;
        Ok(&self.data[base..base + self.d_model])
    }
}

// ---------------------------------------------------------------------------
// Core op
// ---------------------------------------------------------------------------

/// Codebook lookup for a single `(codebook_id, index)` pair — the elementary
/// building block behind [`mimi_rvq_decode`].
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - `codebook_id >= attrs.n_codebooks`;
/// - `codebook_tables.len() != attrs.n_codebooks`;
/// - a codebook table shape that does not match `(codebook_size, d_model)`;
/// - `index >= attrs.codebook_size` (delegated to [`CodebookTable::row`]).
pub fn codebook_lookup<'a>(
    codebook_tables: &'a [CodebookTable],
    codebook_id: usize,
    index: u32,
    attrs: &MimiRvqAttrs,
) -> Result<&'a [f32]> {
    check_tables_shape(codebook_tables, attrs)?;
    if codebook_id >= attrs.n_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "codebook_lookup: codebook_id {codebook_id} >= n_codebooks {}",
            attrs.n_codebooks
        )));
    }
    codebook_tables[codebook_id].row(index)
}

/// Decodes a `[time, n_codebooks]` row-major `codes` block into a
/// `[time, d_model]` row-major feature buffer, summing every codebook's
/// contribution in FP32.
///
/// `codes.len()` must equal `time * attrs.n_codebooks`. Any out-of-range
/// index or shape mismatch is an explicit error (FR-EX-08).
///
/// The residual sum is FP32-accumulated even if a future variant of this op
/// stores codebook tables in FP16 — the mixing precision follows the audio-
/// dialect rule of thumb ("BF16 mantissa loss is the real problem",
/// CLAUDE.md).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - shape mismatch in `codes` / `codebook_tables`;
/// - `codes[t, cb] >= attrs.codebook_size` (no silent clamp — FR-EX-08).
pub fn mimi_rvq_decode(
    codes: &[u32],
    time: usize,
    codebook_tables: &[CodebookTable],
    attrs: &MimiRvqAttrs,
) -> Result<Vec<f32>> {
    check_tables_shape(codebook_tables, attrs)?;
    check_codes_shape(codes, time, attrs)?;

    let mut out = vec![0.0_f32; time * attrs.d_model];
    for t in 0..time {
        let out_base = t * attrs.d_model;
        let code_base = t * attrs.n_codebooks;
        for cb in 0..attrs.n_codebooks {
            let idx = codes[code_base + cb];
            let row = codebook_tables[cb].row(idx)?;
            // FP32 fold (see module docs — no FP16 / BF16 accumulator here).
            for (dst, src) in out[out_base..out_base + attrs.d_model]
                .iter_mut()
                .zip(row.iter())
            {
                *dst += *src;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Paged variant (M3-03 layout)
// ---------------------------------------------------------------------------

/// Writes per-codebook (pre-sum) decoded feature vectors into a
/// [`PagedKvCache<f32>`] with `[time, stream, codebook]` addressing (M3-03).
///
/// The paged cache must be sized with `n_layer = 1`, `n_head = 1`,
/// `d_head = attrs.d_model`, `n_codebook >= attrs.n_codebooks`, and enough
/// `max_time` to accommodate `time_start + time`. `stream` must be within
/// `n_stream`.
///
/// This is the write-side companion of [`mimi_rvq_read_summed`]. Callers who
/// only need the summed features can use [`mimi_rvq_decode`] and skip the
/// paged store; the paged variant exists so CosyVoice2 (M3-09) can page-out
/// per-codebook features and stream them into its Flow Matching CFM chunks.
///
/// K side of each KV row = feature; V side = zero (the paged cache is
/// re-used purely for its addressing / arena — Mimi decode has no K/V
/// semantics, and the V zero-fill is what
/// [`PagedKvCache::pre_allocate`] gives every row by default).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on shape / dim / range violations;
/// [`VokraError::KvCacheExhausted`] if the paged arena is out of pages
/// (both surfaced verbatim from [`PagedKvCache::append_step`]).
pub fn mimi_rvq_decode_paged(
    codes: &[u32],
    time: usize,
    codebook_tables: &[CodebookTable],
    attrs: &MimiRvqAttrs,
    stream: usize,
    cache: &mut PagedKvCache<f32>,
    time_start: usize,
) -> Result<()> {
    check_tables_shape(codebook_tables, attrs)?;
    check_codes_shape(codes, time, attrs)?;
    check_cache_shape(cache, attrs)?;

    let dims = cache.dims();
    if stream >= dims.n_stream {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq_decode_paged: stream {stream} >= cache.n_stream {}",
            dims.n_stream
        )));
    }
    let end = time_start.checked_add(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "mimi_rvq_decode_paged: time_start ({time_start}) + time ({time}) overflows"
        ))
    })?;
    if end > dims.max_time {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq_decode_paged: time_start + time = {end} > cache.max_time {}",
            dims.max_time
        )));
    }

    // Zero-filled V row reused across writes (`append_step` copies, so the
    // slice does not need per-call reallocation).
    let v_zeros = vec![0.0_f32; attrs.d_model];

    for t in 0..time {
        let code_base = t * attrs.n_codebooks;
        for cb in 0..attrs.n_codebooks {
            let idx = codes[code_base + cb];
            let row = codebook_tables[cb].row(idx)?;
            cache.append_step(0, time_start + t, stream, cb, row, &v_zeros)?;
        }
    }
    Ok(())
}

/// Reads and sums the per-codebook feature vectors previously written by
/// [`mimi_rvq_decode_paged`] for `(stream, t)` — the mirror of
/// [`mimi_rvq_decode`]'s output.
///
/// Any codebook slot that has never been written contributes zero (silent
/// zero-fill is fine on the *read* side; the *write* side is where FR-EX-08
/// forbids silent behaviour). The reader treats the paged store as an arena
/// so a partial write followed by a partial read is well-defined.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on axis / dim violations.
pub fn mimi_rvq_read_summed(
    cache: &PagedKvCache<f32>,
    attrs: &MimiRvqAttrs,
    stream: usize,
    t: usize,
) -> Result<Vec<f32>> {
    check_cache_shape(cache, attrs)?;
    let dims = cache.dims();
    if stream >= dims.n_stream {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq_read_summed: stream {stream} >= cache.n_stream {}",
            dims.n_stream
        )));
    }
    if t >= dims.max_time {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq_read_summed: t {t} >= cache.max_time {}",
            dims.max_time
        )));
    }
    let mut acc = vec![0.0_f32; attrs.d_model];
    for cb in 0..attrs.n_codebooks {
        if let Some((k, _v)) = cache.read_step(0, t, stream, cb) {
            for (dst, src) in acc.iter_mut().zip(k.iter()) {
                *dst += *src;
            }
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Consumer stub — MimiDecoder (session-like helper for M3-09)
// ---------------------------------------------------------------------------

/// Host-side helper that owns a set of codebook tables and exposes the same
/// entry points as the eventual `Session::mimi_decode` (M3-09).
///
/// Keeping the helper here in `vokra-ops` avoids a reverse crate dependency
/// (`vokra-core` must not depend on `vokra-ops`). M3-09 will forward its
/// `vokra_core::Session` API to a `MimiDecoder` loaded from the GGUF's
/// `vokra.mimi.*` metadata + tensor chunks.
///
/// The stub is deliberately opinionated: identity codebooks (row `i` =
/// one-hot at position `i mod d_model`) give a trivial end-to-end path that
/// the smoke test exercises without touching a real Kyutai checkpoint.
#[derive(Debug, Clone)]
pub struct MimiDecoder {
    attrs: MimiRvqAttrs,
    tables: Vec<CodebookTable>,
}

impl MimiDecoder {
    /// Builds a decoder from an already-loaded set of codebook tables.
    ///
    /// The tables are validated for shape at construction (each is
    /// `[codebook_size, d_model]`) so per-decode calls do not repay that
    /// cost.
    pub fn new(attrs: MimiRvqAttrs, tables: Vec<CodebookTable>) -> Result<Self> {
        check_tables_shape(&tables, &attrs)?;
        Ok(Self { attrs, tables })
    }

    /// Builds a decoder whose codebook tables are the identity fixture
    /// (row `i` puts a single `1.0` at column `i mod d_model`, zero
    /// elsewhere). Every codebook is the *same* identity so the summed
    /// output at a given code is `n_codebooks` times the one-hot row —
    /// making the invariant test-friendly without any weight file.
    ///
    /// The identity fixture is the only host-only smoke path that ships
    /// today; a real Kyutai / CosyVoice2 checkpoint is loaded by M3-09.
    pub fn identity(attrs: MimiRvqAttrs) -> Result<Self> {
        let cb_size = attrs.codebook_size;
        let d_model = attrs.d_model;
        if cb_size == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "MimiDecoder::identity: codebook_size and d_model must be > 0, got \
                 codebook_size={cb_size} d_model={d_model}"
            )));
        }
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for _ in 0..attrs.n_codebooks {
            let mut data = vec![0.0_f32; cb_size * d_model];
            for i in 0..cb_size {
                data[i * d_model + (i % d_model)] = 1.0;
            }
            tables.push(CodebookTable::new(cb_size, d_model, data)?);
        }
        Self::new(attrs, tables)
    }

    /// Attribute snapshot.
    #[inline]
    #[must_use]
    pub const fn attrs(&self) -> &MimiRvqAttrs {
        &self.attrs
    }

    /// Read-only view of the codebook tables (used by tests and by the M3-09
    /// converter for round-trip audits).
    #[inline]
    #[must_use]
    pub fn tables(&self) -> &[CodebookTable] {
        &self.tables
    }

    /// Decodes a full `[time, n_codebooks]` code block and returns
    /// `[time, d_model]` row-major features.
    pub fn decode(&self, codes: &[u32], time: usize) -> Result<Vec<f32>> {
        mimi_rvq_decode(codes, time, &self.tables, &self.attrs)
    }

    /// Streams `[time, n_codebooks]` codes into a paged store. See
    /// [`mimi_rvq_decode_paged`] for the shape contract.
    pub fn decode_paged(
        &self,
        codes: &[u32],
        time: usize,
        stream: usize,
        cache: &mut PagedKvCache<f32>,
        time_start: usize,
    ) -> Result<()> {
        mimi_rvq_decode_paged(
            codes,
            time,
            &self.tables,
            &self.attrs,
            stream,
            cache,
            time_start,
        )
    }
}

// ---------------------------------------------------------------------------
// Shared shape checks
// ---------------------------------------------------------------------------

fn check_tables_shape(codebook_tables: &[CodebookTable], attrs: &MimiRvqAttrs) -> Result<()> {
    if attrs.n_codebooks == 0 || attrs.codebook_size == 0 || attrs.d_model == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq: attrs must have every axis > 0, got n_codebooks={} \
             codebook_size={} d_model={}",
            attrs.n_codebooks, attrs.codebook_size, attrs.d_model,
        )));
    }
    if codebook_tables.len() != attrs.n_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq: codebook_tables.len() {} != attrs.n_codebooks {}",
            codebook_tables.len(),
            attrs.n_codebooks
        )));
    }
    for (i, t) in codebook_tables.iter().enumerate() {
        if t.codebook_size != attrs.codebook_size || t.d_model != attrs.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "mimi_rvq: codebook_tables[{i}] shape [{},{}] != attrs [{},{}]",
                t.codebook_size, t.d_model, attrs.codebook_size, attrs.d_model
            )));
        }
    }
    Ok(())
}

fn check_codes_shape(codes: &[u32], time: usize, attrs: &MimiRvqAttrs) -> Result<()> {
    let expected = time.checked_mul(attrs.n_codebooks).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "mimi_rvq: time ({time}) * n_codebooks ({}) overflows usize",
            attrs.n_codebooks
        ))
    })?;
    if codes.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq: codes.len() {} != time * n_codebooks {expected}",
            codes.len()
        )));
    }
    Ok(())
}

fn check_cache_shape(cache: &PagedKvCache<f32>, attrs: &MimiRvqAttrs) -> Result<()> {
    let dims = cache.dims();
    if dims.n_layer != 1 {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq paged: cache.n_layer must be 1 (Mimi is single-layer), got {}",
            dims.n_layer
        )));
    }
    if dims.n_head != 1 {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq paged: cache.n_head must be 1, got {}",
            dims.n_head
        )));
    }
    if dims.d_head != attrs.d_model {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq paged: cache.d_head {} != attrs.d_model {}",
            dims.d_head, attrs.d_model
        )));
    }
    if dims.n_codebook < attrs.n_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "mimi_rvq paged: cache.n_codebook {} < attrs.n_codebooks {}",
            dims.n_codebook, attrs.n_codebooks
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// KvDims helper: build the shape the paged variant expects (test/consumer
// convenience — every field is derivable but this reads better at call site).
// ---------------------------------------------------------------------------

/// Builds a [`KvDims`] with the shape [`mimi_rvq_decode_paged`] expects.
///
/// Callers who already have a [`KvDims`] for their model's decoder can
/// override the codebook axis in place; this helper is for the common case
/// of a Mimi-only cache.
#[must_use]
pub fn mimi_paged_dims(attrs: &MimiRvqAttrs, n_stream: usize, max_time: usize) -> KvDims {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::cache::paged::{BlockSize, PagedKvCache};

    // ------------ Small helpers --------------------------------------------

    /// Simple deterministic codebook: row `i` is `[i, i+1, i+2, ..., i+d-1]`
    /// cast to f32. Each codebook is offset by `cb_offset * 100` so different
    /// codebooks yield different rows for the same index.
    fn make_ramp_tables(attrs: MimiRvqAttrs) -> Vec<CodebookTable> {
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
            for i in 0..attrs.codebook_size {
                for d in 0..attrs.d_model {
                    data[i * attrs.d_model + d] = (i + d) as f32 + (cb as f32) * 100.0;
                }
            }
            tables.push(CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap());
        }
        tables
    }

    /// A tiny attrs shape used throughout the tests — big enough to exercise
    /// bounds, small enough to reason about by hand.
    fn tiny_attrs() -> MimiRvqAttrs {
        MimiRvqAttrs {
            n_codebooks: 3,
            codebook_size: 4,
            d_model: 5,
        }
    }

    // ---- T02: op signature is stable, doctest-friendly --------------------

    #[test]
    fn mimi_attrs_canonical_matches_kyutai_default() {
        let a = MimiRvqAttrs::mimi();
        assert_eq!(a.n_codebooks, 8);
        assert_eq!(a.codebook_size, 2048);
        assert_eq!(a.d_model, 512);
    }

    // ---- T03: codebook_lookup exact-match ---------------------------------

    #[test]
    fn codebook_lookup_returns_expected_row() {
        // Ramp table: row 2 of codebook 1 is [102+0, 102+1, ...] = [102, 103, 104, 105, 106]
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let row = codebook_lookup(&tables, 1, 2, &attrs).unwrap();
        let expected: Vec<f32> = (0..attrs.d_model).map(|d| 102.0 + d as f32).collect();
        assert_eq!(row, expected.as_slice());
    }

    #[test]
    fn codebook_lookup_rejects_out_of_range() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        // codebook_id too high
        assert!(matches!(
            codebook_lookup(&tables, 42, 0, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // index too high
        assert!(matches!(
            codebook_lookup(&tables, 0, 42, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T04: residual sum bit-identical to a hand-rolled fold ------------

    #[test]
    fn residual_sum_is_bit_identical_to_hand_fold() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let time = 3;
        // Fixed codes: `[0,1,2, 3,2,1, 1,0,3]`.
        let codes: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 1, 0, 3];
        let got = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();

        // Hand-fold oracle: sum every codebook's row for every timestep in a
        // scalar FP32 loop, which is exactly what the impl does.
        let mut want = vec![0.0_f32; time * attrs.d_model];
        for t in 0..time {
            for cb in 0..attrs.n_codebooks {
                let idx = codes[t * attrs.n_codebooks + cb];
                let row = tables[cb].row(idx).unwrap();
                for d in 0..attrs.d_model {
                    want[t * attrs.d_model + d] += row[d];
                }
            }
        }
        assert_eq!(got, want, "residual sum must be bit-identical FP32 fold");
    }

    #[test]
    fn decode_rejects_shape_mismatches_and_bad_index() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        // codes.len() != time * n_codebooks.
        let short = vec![0u32; attrs.n_codebooks - 1];
        assert!(matches!(
            mimi_rvq_decode(&short, 1, &tables, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Wrong table count.
        let bad_tables = tables[..attrs.n_codebooks - 1].to_vec();
        let codes = vec![0u32; attrs.n_codebooks];
        assert!(matches!(
            mimi_rvq_decode(&codes, 1, &bad_tables, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Out-of-range index (silent-clamp is forbidden — FR-EX-08).
        let over = {
            let mut v = vec![0u32; attrs.n_codebooks];
            v[1] = attrs.codebook_size as u32; // == codebook_size (bad)
            v
        };
        assert!(matches!(
            mimi_rvq_decode(&over, 1, &tables, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T05: block_size=2 paging (Mimi 12.5 Hz primary) ------------------

    #[test]
    fn paged_block_size_two_matches_direct_decode() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let time = 5;
        let stream = 0;
        let codes: Vec<u32> = (0..time as u32 * attrs.n_codebooks as u32)
            .map(|i| i % attrs.codebook_size as u32)
            .collect();

        let dims = mimi_paged_dims(&attrs, /*n_stream=*/ 1, /*max_time=*/ 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

        // Paged decode.
        let decoder = MimiDecoder::new(attrs, tables.clone()).unwrap();
        decoder
            .decode_paged(&codes, time, stream, &mut cache, 0)
            .unwrap();

        // Direct decode (oracle).
        let direct = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();

        // Compare per-timestep summed reads.
        for t in 0..time {
            let summed = mimi_rvq_read_summed(&cache, &attrs, stream, t).unwrap();
            let want = &direct[t * attrs.d_model..(t + 1) * attrs.d_model];
            assert_eq!(summed, want, "paged block_size=2, t={t}");
        }

        // Sanity: block_size=2 → time step 3 lives on page index 1.
        assert_eq!(cache.page_of(3), 1);
    }

    // ---- T06: block_size=4 paging (variant) -------------------------------

    #[test]
    fn paged_block_size_four_matches_direct_decode() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let time = 6;
        let stream = 0;
        let codes: Vec<u32> = (0..time as u32 * attrs.n_codebooks as u32)
            .map(|i| (i * 3) % attrs.codebook_size as u32)
            .collect();

        let dims = mimi_paged_dims(&attrs, /*n_stream=*/ 1, /*max_time=*/ 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();

        let decoder = MimiDecoder::new(attrs, tables.clone()).unwrap();
        decoder
            .decode_paged(&codes, time, stream, &mut cache, 0)
            .unwrap();

        let direct = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();
        for t in 0..time {
            let summed = mimi_rvq_read_summed(&cache, &attrs, stream, t).unwrap();
            let want = &direct[t * attrs.d_model..(t + 1) * attrs.d_model];
            assert_eq!(summed, want, "paged block_size=4, t={t}");
        }
        // Sanity: block_size=4 → time step 5 lives on page index 1.
        assert_eq!(cache.page_of(5), 1);
    }

    // ---- T07: codebook dim contiguous inside the row ---------------------

    #[test]
    fn paged_layout_keeps_codebook_dim_contiguous() {
        // The M3-03 row layout is [block_offset, stream, codebook, head, d_head].
        // For our (n_head=1, d_head=d_model, n_stream=1) shape, two adjacent
        // codebooks (cb and cb+1) at the same (layer, t, stream) live in
        // adjacent d_model-long slots — i.e. moving cb by 1 advances the
        // start pointer by exactly d_model floats. This is the "codebook
        // stride = 1 slot" invariant.
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let time = 1;
        let stream = 0;
        // Distinct codes per codebook so we can distinguish rows.
        let codes: Vec<u32> = (0..attrs.n_codebooks as u32).collect();

        let dims = mimi_paged_dims(&attrs, 1, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
        mimi_rvq_decode_paged(&codes, time, &tables, &attrs, stream, &mut cache, 0).unwrap();

        // Read every codebook slot: they must equal `tables[cb].row(cb)`.
        for (cb, table) in tables.iter().enumerate().take(attrs.n_codebooks) {
            let (k, _v) = cache.read_step(0, 0, stream, cb).expect("row written");
            let expected = table.row(cb as u32).unwrap();
            assert_eq!(k, expected, "codebook slot cb={cb} must be contiguous");
        }
    }

    // ---- T08 / T09: MimiDecoder session-like helper -----------------------

    #[test]
    fn mimi_decoder_identity_yields_predictable_sum() {
        // Identity codebook: row i puts 1.0 at column (i mod d_model), 0 else.
        // Every codebook is the same identity, so decoding codes = [c0, c1, ..., c_{K-1}]
        // sums K one-hots. If all `c_k = c`, the summed row has n_codebooks
        // at column (c mod d_model) and 0 elsewhere.
        let attrs = tiny_attrs();
        let decoder = MimiDecoder::identity(attrs).unwrap();
        let time = 1;
        // Every codebook gets code = 2 → summed row = n_codebooks at col 2, 0 else.
        let codes = vec![2u32; attrs.n_codebooks];
        let out = decoder.decode(&codes, time).unwrap();
        let mut want = vec![0.0_f32; attrs.d_model];
        want[2 % attrs.d_model] = attrs.n_codebooks as f32;
        assert_eq!(out, want);
    }

    #[test]
    fn mimi_decoder_new_validates_shape() {
        let attrs = tiny_attrs();
        // Wrong number of tables → validated at construction (not at decode).
        let short = make_ramp_tables(MimiRvqAttrs {
            n_codebooks: attrs.n_codebooks - 1,
            ..attrs
        });
        assert!(matches!(
            MimiDecoder::new(attrs, short),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn mimi_decoder_paged_matches_flat_decode() {
        let attrs = tiny_attrs();
        let decoder = MimiDecoder::identity(attrs).unwrap();
        let time = 4;
        let codes: Vec<u32> = (0..time as u32 * attrs.n_codebooks as u32)
            .map(|i| i % attrs.codebook_size as u32)
            .collect();

        let flat = decoder.decode(&codes, time).unwrap();

        let dims = mimi_paged_dims(&attrs, /*n_stream=*/ 1, /*max_time=*/ time);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
        decoder
            .decode_paged(&codes, time, 0, &mut cache, 0)
            .unwrap();
        for t in 0..time {
            let sum = mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            let want = &flat[t * attrs.d_model..(t + 1) * attrs.d_model];
            assert_eq!(sum, want, "paged/flat mismatch at t={t}");
        }
    }

    // ---- T09: multi-stream isolation --------------------------------------

    #[test]
    fn paged_multi_stream_write_isolation() {
        // Two streams share the same cache; a write on stream 0 must not
        // affect stream 1's summed read.
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let dims = mimi_paged_dims(&attrs, /*n_stream=*/ 2, /*max_time=*/ 2);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

        // stream 0 codes: all zeros.
        let codes0 = vec![0u32; attrs.n_codebooks];
        // stream 1 codes: all threes.
        let codes1 = vec![3u32; attrs.n_codebooks];
        mimi_rvq_decode_paged(&codes0, 1, &tables, &attrs, 0, &mut cache, 0).unwrap();
        mimi_rvq_decode_paged(&codes1, 1, &tables, &attrs, 1, &mut cache, 0).unwrap();

        let s0 = mimi_rvq_read_summed(&cache, &attrs, 0, 0).unwrap();
        let s1 = mimi_rvq_read_summed(&cache, &attrs, 1, 0).unwrap();
        // The two streams differ (codes differ), which is the isolation
        // signal we care about.
        assert_ne!(s0, s1, "streams must not alias each other");
    }

    // ---- T16: host-only fallback smoke ------------------------------------

    #[test]
    fn host_only_smoke_decode_end_to_end() {
        // The full path — identity decoder + flat decode + paged decode +
        // summed read — runs on the CPU with zero external dependencies.
        // Ignition test for the "silent fallback forbidden" contract: any
        // future GPU-only path must be an explicit opt-in.
        let attrs = MimiRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 4,
        };
        let decoder = MimiDecoder::identity(attrs).unwrap();
        let time = 2;
        let codes = vec![0u32, 1, 2, 0];
        let flat = decoder.decode(&codes, time).unwrap();
        assert_eq!(flat.len(), time * attrs.d_model);

        let dims = mimi_paged_dims(&attrs, 1, time);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
        decoder
            .decode_paged(&codes, time, 0, &mut cache, 0)
            .unwrap();
        for t in 0..time {
            let want = &flat[t * attrs.d_model..(t + 1) * attrs.d_model];
            let got = mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn paged_rejects_bad_cache_shape() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);

        // Wrong d_head.
        let bad_dims = KvDims {
            d_head: attrs.d_model + 1,
            ..mimi_paged_dims(&attrs, 1, 2)
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(bad_dims, BlockSize::Two).unwrap();
        let codes = vec![0u32; attrs.n_codebooks];
        assert!(matches!(
            mimi_rvq_decode_paged(&codes, 1, &tables, &attrs, 0, &mut cache, 0),
            Err(VokraError::InvalidArgument(_))
        ));

        // n_layer must be 1.
        let bad_dims2 = KvDims {
            n_layer: 2,
            ..mimi_paged_dims(&attrs, 1, 2)
        };
        let mut cache2 = PagedKvCache::<f32>::pre_allocate(bad_dims2, BlockSize::Two).unwrap();
        assert!(matches!(
            mimi_rvq_decode_paged(&codes, 1, &tables, &attrs, 0, &mut cache2, 0),
            Err(VokraError::InvalidArgument(_))
        ));

        // Stream out of range.
        let dims = mimi_paged_dims(&attrs, /*n_stream=*/ 1, 2);
        let mut cache3 = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
        assert!(matches!(
            mimi_rvq_decode_paged(
                &codes,
                1,
                &tables,
                &attrs,
                /*stream=*/ 5,
                &mut cache3,
                0
            ),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn paged_rejects_time_overflow() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let dims = mimi_paged_dims(&attrs, 1, /*max_time=*/ 2);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
        let codes = vec![0u32; attrs.n_codebooks * 3]; // time=3 > max_time=2
        assert!(matches!(
            mimi_rvq_decode_paged(&codes, 3, &tables, &attrs, 0, &mut cache, 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn codebook_table_new_validates_shape() {
        // Non-multiple data length → error.
        let bad = CodebookTable::new(4, 5, vec![0.0; 4 * 5 - 1]);
        assert!(matches!(bad, Err(VokraError::InvalidArgument(_))));

        // Zero axis → error.
        let bad2 = CodebookTable::new(0, 5, vec![]);
        assert!(matches!(bad2, Err(VokraError::InvalidArgument(_))));
    }
}
