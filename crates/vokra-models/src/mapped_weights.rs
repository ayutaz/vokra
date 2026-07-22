//! Shared primitives for **mapped-lazy** weight stores.
//!
//! A mapped-lazy store leaves a model's bulk weights inside the GGUF mapping
//! (`vokra_mmap::open_gguf`) and widens one layer at a time into a reused
//! scratch block during the forward pass, instead of materialising every
//! tensor as owned `f32` at load. The trade is recurring per-step decode
//! bandwidth for a bounded memory ceiling — the difference between "needs
//! ~30 GiB resident" and "runs on a 16 GiB machine".
//!
//! This module holds the parts that are **model-independent**: the widen +
//! transpose kernels, the tile constant that gives them locality, and the
//! bind-time tensor validator. The per-model stores
//! (`moshi::backbone::MappedTemporalBlocks`,
//! `voxtral::text_decoder::MappedTextBlocks`) own their own layer-location
//! tables and `materialize_into` bodies, because the tensor names, the
//! fused-vs-split layout and the shape parameters differ per architecture.
//!
//! Every function takes a `model` tag purely so errors name the caller
//! (`"moshi"` / `"voxtral"`) — FR-EX-08 messages must say which loader failed
//! and what the alternative is.
//!
//! # Bit-identity contract
//!
//! The widening formulas here are byte-for-byte the ones
//! `gguf::quant::dequantize` uses (BF16: the stored `u16` is the *top half* of
//! the f32 bit pattern, so `bits << 16` is exact and lossless; F32:
//! `from_le_bytes`). A mapped store must therefore produce **bit-identical**
//! values to the corresponding resident loader — pinned per model by a
//! `mapped_*_match_resident_bitwise*` test.

use std::sync::{Mutex, MutexGuard};

use vokra_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use vokra_core::{Result, VokraError};

/// Identifies the model a mapped store belongs to, for error messages only.
///
/// `resident_entry` is the constructor a caller should reach for when the
/// mapped path genuinely cannot serve a payload (a quantized or F16 GGUF).
/// Naming it is what makes the refusal actionable rather than a dead end —
/// the mapped path is an optimization, and there is always a resident route.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MappedModel {
    /// Short model name used as the error prefix (`"moshi"` / `"voxtral"`).
    pub(crate) name: &'static str,
    /// Fully-qualified resident constructor to recommend on refusal.
    pub(crate) resident_entry: &'static str,
}

/// Square tile edge for [`transpose_widen`], in elements.
///
/// The naive `dst[c * rows + r]` walk writes with a `rows * 4`-byte stride, so
/// at multi-thousand `rows` (Moshi-7B: 4096 / 11264; Voxtral-Mini: 3072 /
/// 4096 / 8192) essentially every element costs a cache miss *and* a TLB miss.
/// Tiling bounds the working set of an inner pass to `TILE` short runs on each
/// side: per tile the destination touches `TILE` contiguous runs of `TILE` f32
/// (128 B each at `TILE = 32`) and the source `TILE` contiguous runs of `TILE`
/// elements — together a few KiB, i.e. L1-resident, and each destination page
/// is revisited `TILE` times while it is still mapped rather than once per
/// full sweep.
///
/// 32 is chosen so a destination run (32 x 4 B = 128 B) is an exact multiple
/// of the 64 B cache line on both aarch64 and x86-64, with no partial-line
/// writes at tile boundaries for aligned starts.
pub(crate) const TRANSPOSE_TILE: usize = 32;

/// Widens + transposes a `[rows, cols]` row-major `F32`/`BF16` payload into
/// `dst` as `[cols, rows]` — the fused equivalent of `quant::dequantize`
/// followed by a transpose, and byte-formula-identical to both.
///
/// The traversal is **tiled** ([`TRANSPOSE_TILE`]) purely for locality: every
/// destination element is written exactly once, from the same source element,
/// through the same widening formula as the untiled walk. Reordering
/// independent writes cannot change any value, so the result is
/// **bit-identical** to the naive order — pinned by
/// `tiled_transpose_matches_naive_bitwise` below (odd shapes, both dtypes).
///
/// # Errors
///
/// [`VokraError::ModelLoad`] if `src` is not exactly `rows * cols` elements
/// wide for `dtype`, or if `dtype` is outside `{F32, BF16}` (checked *before*
/// `dst` is touched, so a rejected call leaves the destination untouched).
pub(crate) fn transpose_widen(
    src: &[u8],
    dtype: GgmlType,
    rows: usize,
    cols: usize,
    dst: &mut Vec<f32>,
    model: MappedModel,
) -> Result<()> {
    let n = rows * cols;
    if src.len() != n * dtype.type_size() {
        return Err(VokraError::ModelLoad(format!(
            "{} mapped blocks: payload is {} bytes, expected {} ({n} x {:?})",
            model.name,
            src.len(),
            n * dtype.type_size(),
            dtype
        )));
    }
    if !matches!(dtype, GgmlType::F32 | GgmlType::BF16) {
        return Err(unsupported_dtype(model, dtype));
    }
    dst.clear();
    dst.resize(n, 0.0);
    let is_f32 = dtype == GgmlType::F32;
    let esz = dtype.type_size();
    for r0 in (0..rows).step_by(TRANSPOSE_TILE) {
        let r_end = (r0 + TRANSPOSE_TILE).min(rows);
        for c0 in (0..cols).step_by(TRANSPOSE_TILE) {
            let c_end = (c0 + TRANSPOSE_TILE).min(cols);
            for r in r0..r_end {
                let row_base = r * cols;
                for c in c0..c_end {
                    let i = (row_base + c) * esz;
                    dst[c * rows + r] = if is_f32 {
                        f32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]])
                    } else {
                        f32::from_bits(u32::from(u16::from_le_bytes([src[i], src[i + 1]])) << 16)
                    };
                }
            }
        }
    }
    Ok(())
}

/// Widens a dense `F32`/`BF16` payload into `dst` in storage order (the γ
/// vectors — no transpose).
///
/// # Errors
///
/// [`VokraError::ModelLoad`] if `dtype` is outside `{F32, BF16}`.
pub(crate) fn widen_into(
    src: &[u8],
    dtype: GgmlType,
    dst: &mut Vec<f32>,
    model: MappedModel,
) -> Result<()> {
    let esz = dtype.type_size();
    match dtype {
        GgmlType::F32 => {
            dst.clear();
            dst.reserve(src.len() / esz);
            for c in src.chunks_exact(4) {
                dst.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(())
        }
        GgmlType::BF16 => {
            dst.clear();
            dst.reserve(src.len() / esz);
            for c in src.chunks_exact(2) {
                dst.push(f32::from_bits(
                    u32::from(u16::from_le_bytes([c[0], c[1]])) << 16,
                ));
            }
            Ok(())
        }
        other => Err(unsupported_dtype(model, other)),
    }
}

/// The one message every mapped path uses for a dtype it cannot widen in
/// place — quantized payloads have no per-element byte offset, so the mapped
/// store genuinely cannot serve them and the caller must fall back to the
/// resident loader (which dequantizes block-wise).
fn unsupported_dtype(model: MappedModel, dtype: GgmlType) -> VokraError {
    let MappedModel {
        name,
        resident_entry,
    } = model;
    VokraError::ModelLoad(format!(
        "{name} mapped blocks: unsupported dtype {dtype:?}; the bounded-memory \
         mapped path serves F32 and BF16 payloads only — load through \
         {resident_entry} (resident) for this GGUF (FR-EX-08: explicit, not \
         silent)"
    ))
}

/// Resolves a tensor descriptor for a mapped store: present, exact element
/// count, dtype in `{F32, BF16}` (loud otherwise — FR-EX-08).
///
/// Doing this for **every** layer at bind time is what keeps a malformed GGUF
/// from failing halfway through a stream: the load fails, not the forward.
///
/// # Errors
///
/// [`VokraError::ModelLoad`] naming the tensor on a miss, a count mismatch or
/// an unsupported dtype.
pub(crate) fn mapped_info(
    file: &GgufFile,
    name: &str,
    want_elems: usize,
    model: MappedModel,
) -> Result<GgufTensorInfo> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("{}: tensor `{name}`: missing", model.name))
    })?;
    let elems = info
        .element_count()
        .map_err(|e| VokraError::ModelLoad(format!("{}: tensor `{name}`: {e}", model.name)))?;
    if elems != want_elems as u64 {
        return Err(VokraError::ModelLoad(format!(
            "{}: tensor `{name}` has {elems} elements, expected {want_elems}",
            model.name
        )));
    }
    match info.dtype {
        GgmlType::F32 | GgmlType::BF16 => Ok(info.clone()),
        other => Err(unsupported_dtype(model, other)),
    }
}

/// Locks a mapped store's materialization scratch, converting a poisoned
/// mutex into a loud, actionable error (a poisoned scratch means a previous
/// forward panicked mid-materialization, so its contents are undefined).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when the mutex is poisoned.
pub(crate) fn lock_scratch<'a, T>(
    scratch: &'a Mutex<T>,
    model: MappedModel,
) -> Result<MutexGuard<'a, T>> {
    scratch.lock().map_err(|_| {
        VokraError::InvalidArgument(format!(
            "{} mapped blocks: materialization scratch mutex poisoned (a \
             prior panic mid-forward); reload the engine",
            model.name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The untiled reference walk `transpose_widen` replaced — kept verbatim
    /// as the oracle so the tiling can never drift into a value change.
    fn transpose_widen_naive(
        src: &[u8],
        dtype: GgmlType,
        rows: usize,
        cols: usize,
        dst: &mut Vec<f32>,
    ) {
        dst.clear();
        dst.resize(rows * cols, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = match dtype {
                    GgmlType::F32 => {
                        let i = (r * cols + c) * 4;
                        f32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]])
                    }
                    GgmlType::BF16 => {
                        let i = (r * cols + c) * 2;
                        f32::from_bits(u32::from(u16::from_le_bytes([src[i], src[i + 1]])) << 16)
                    }
                    other => panic!("oracle covers F32/BF16 only, got {other:?}"),
                };
            }
        }
    }

    /// A deterministic bit pattern exercising sign / exponent / mantissa,
    /// including values whose BF16 top-half is negative.
    const TEST_MODEL: MappedModel = MappedModel {
        name: "test",
        resident_entry: "TestModel::from_gguf",
    };

    fn word(k: usize) -> u16 {
        ((k as u32).wrapping_mul(2_654_435_761) >> 16) as u16
    }

    /// Tiling is a pure traversal reorder: every destination element is
    /// written once, from the same source element, through the same formula.
    /// Shapes deliberately straddle [`TRANSPOSE_TILE`] (partial edge tiles on
    /// both axes, and the degenerate 1-row / 1-col cases) so an off-by-one in
    /// the tile clamp cannot hide. Non-square shapes cover the GQA / FFN
    /// rectangles Voxtral needs.
    #[test]
    fn tiled_transpose_matches_naive_bitwise() {
        for &(rows, cols) in &[
            (1, 1),
            (1, 77),
            (77, 1),
            (TRANSPOSE_TILE, TRANSPOSE_TILE),
            (TRANSPOSE_TILE + 1, TRANSPOSE_TILE - 1),
            (33, 65),
            (70, 37),
            (129, 31),
        ] {
            let n = rows * cols;

            let mut bf16 = Vec::with_capacity(n * 2);
            for k in 0..n {
                bf16.extend_from_slice(&word(k).to_le_bytes());
            }
            let (mut got, mut want) = (Vec::new(), Vec::new());
            transpose_widen(&bf16, GgmlType::BF16, rows, cols, &mut got, TEST_MODEL).unwrap();
            transpose_widen_naive(&bf16, GgmlType::BF16, rows, cols, &mut want);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "BF16 {rows}x{cols}: tiled transpose must be bit-identical"
            );

            let mut f32b = Vec::with_capacity(n * 4);
            for k in 0..n {
                let bits = (u32::from(word(k)) << 16) | u32::from(word(k + 1));
                f32b.extend_from_slice(&bits.to_le_bytes());
            }
            let (mut got, mut want) = (Vec::new(), Vec::new());
            transpose_widen(&f32b, GgmlType::F32, rows, cols, &mut got, TEST_MODEL).unwrap();
            transpose_widen_naive(&f32b, GgmlType::F32, rows, cols, &mut want);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "F32 {rows}x{cols}: tiled transpose must be bit-identical"
            );
        }
    }

    /// A short payload is a loud error, and an unsupported dtype is rejected
    /// before the destination is filled (so a rejected call cannot leave a
    /// half-written scratch behind).
    #[test]
    fn transpose_widen_rejects_bad_payload_and_dtype() {
        let mut dst = Vec::new();
        let err =
            transpose_widen(&[0u8; 6], GgmlType::BF16, 2, 2, &mut dst, TEST_MODEL).unwrap_err();
        assert!(
            format!("{err}").contains("payload is 6 bytes"),
            "short payload must name the byte counts, got: {err}"
        );
        // The payload check runs first, so reach the dtype arm with a
        // correctly-sized (if semantically meaningless) quantized payload.
        let q4k = vec![0u8; 4 * GgmlType::Q4K.type_size()];
        let err = transpose_widen(&q4k, GgmlType::Q4K, 2, 2, &mut dst, TEST_MODEL).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported dtype")
                && msg.contains(TEST_MODEL.name)
                && msg.contains(TEST_MODEL.resident_entry),
            "an unsupported dtype must be explicit, name the model, and point at \
             the resident constructor (otherwise the refusal is a dead end), got: {msg}"
        );
        assert!(
            dst.is_empty(),
            "an unsupported dtype must be rejected before the destination is filled"
        );
    }

    /// `widen_into` is the no-transpose sibling; it must use the identical
    /// BF16 formula (top-half shift) and reject the same dtypes.
    #[test]
    fn widen_into_is_exact_and_rejects_quantized() {
        let mut src = Vec::new();
        for k in 0..64 {
            src.extend_from_slice(&word(k).to_le_bytes());
        }
        let mut dst = Vec::new();
        widen_into(&src, GgmlType::BF16, &mut dst, TEST_MODEL).unwrap();
        assert_eq!(dst.len(), 64);
        for (k, v) in dst.iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                u32::from(word(k)) << 16,
                "BF16 widening is the exact top-half shift"
            );
        }
        let err = widen_into(&[0u8; 144], GgmlType::Q4K, &mut dst, TEST_MODEL).unwrap_err();
        assert!(format!("{err}").contains("unsupported dtype"), "{err}");
    }
}
