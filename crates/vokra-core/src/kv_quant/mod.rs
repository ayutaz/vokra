//! Runtime KV cache quantization (M3-04, FR-QT-05).
//!
//! # Scope
//!
//! This module carries the **runtime** quantization arm for the paged KV
//! cache (`PagedKvCache`, M3-03) — a completely separate layer from the M1
//! **weight** quantization (K-quants `Q4_K` / `Q5_K` / `Q6_K` under
//! `crates/vokra-core/src/gguf/quant/`). Do not confuse them:
//!
//! | Layer            | Element size    | Storage location            | Scope  |
//! |------------------|-----------------|-----------------------------|--------|
//! | Weight (K-quant) | 256 (super-block) | disk (GGUF tensor payload) | M1     |
//! | KV cache (Q_0)   | 32 (single scale) | runtime page memory only  | M3-04  |
//!
//! See `docs/adr/M3-04-kv-cache-quantization.md` §D1 for the rationale.
//!
//! # Formats
//!
//! Three symmetric block layouts (llama.cpp-compatible; `_0` suffix = no
//! zero-point), one per module file:
//!
//! - [`Q4_0`](q4_0::BlockQ4_0): 4-bit signed + FP16 scale, 18 bytes / block
//!   (compression ratio 8× vs FP32).
//! - [`Q5_0`](q5_0::BlockQ5_0): 5-bit signed + FP16 scale, 22 bytes / block
//!   (~5.8× vs FP32).
//! - [`Q8_0`](q8_0::BlockQ8_0): 8-bit signed + FP16 scale, 34 bytes / block
//!   (~3.8× vs FP32).
//!
//! Asymmetric `_1` variants are intentionally out of scope; the K-quant
//! super-block already covers that on the weight side.
//!
//! # Zero-dep + safe
//!
//! Every file in this module is safe Rust (no `unsafe`), std-only, no external
//! deps (NFR-DS-02). Bit-packing uses plain `u8` / `u16` arithmetic. The FP16
//! helper ([`half`]) duplicates `gguf/quant/mod.rs::f16_to_f32` on purpose
//! (M3-04 ADR §D1) so the two layers cannot silently drift.
//!
//! # Example
//!
//! ```
//! use vokra_core::kv_quant::{BlockQ8_0, KvQuant};
//!
//! // A single 32-element window round-trips through the pack/unpack seam
//! // within one quantization step.
//! let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
//! let block = BlockQ8_0::pack(&input);
//! let mut out = vec![0.0_f32; 32];
//! block.unpack(&mut out);
//!
//! // KvQuant is what a Session builder accepts via `.with_kv_quant(...)`.
//! assert_eq!(KvQuant::Q8_0.block_bytes(), 34);
//! assert_eq!(KvQuant::Q8_0.block_size(), 32);
//! ```

pub mod dequant_gemm;
pub mod gpu_ops;
pub mod half;
pub mod q4_0;
pub mod q5_0;
pub mod q8_0;
pub mod verify;

pub use dequant_gemm::{
    dense_gemv_f32, dequant_gemv_scalar, max_relative_error, pack_matrix_to_bytes,
    quant_gemv_round_trip,
};
pub use gpu_ops::{KvQuantDequantGemvOps, validate_dequant_gemv};
pub use half::{F16Bits, f16_bits_to_f32, f32_to_f16_bits};
pub use q4_0::BlockQ4_0;
pub use q5_0::BlockQ5_0;
pub use q8_0::BlockQ8_0;
pub use verify::{DEGRADATION_THRESHOLD, KvQuantMetric, KvQuantVerifyReport};

/// Universal 32-element block size shared by every Q_0 format.
///
/// Kept as a top-level const so the paged KV cache (`PagedKvCache`) can align
/// its `time` axis with the quant block boundary without pulling in a
/// per-format constant.
pub const KV_QUANT_BLOCK_SIZE: usize = 32;

/// Runtime KV cache quantization mode (FR-QT-05).
///
/// Selected by the `Session` builder via [`SessionBuilder::with_kv_quant`]. The
/// default across every builder is [`Self::Fp32`] (no quantization) so
/// existing call sites keep bit-identical behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvQuant {
    /// Uncompressed FP32 — the pre-M3-04 behaviour, kept as the default.
    Fp32,
    /// 4-bit signed + FP16 scale, 32-elem block, 18 bytes.
    Q4_0,
    /// 5-bit signed + FP16 scale, 32-elem block, 22 bytes.
    Q5_0,
    /// 8-bit signed + FP16 scale, 32-elem block, 34 bytes.
    Q8_0,
}

impl KvQuant {
    /// Number of on-wire bytes per 32-element block. Included for the FR-QT-05
    /// footprint bench (`vokra-cli bench`) and for consumer crates that need
    /// to reason about page sizing.
    #[inline]
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Fp32 => 32 * 4, // 128 bytes = 32 × FP32
            Self::Q4_0 => q4_0::BLOCK_BYTES,
            Self::Q5_0 => q5_0::BLOCK_BYTES,
            Self::Q8_0 => q8_0::BLOCK_BYTES,
        }
    }

    /// Elements per block. Always 32 for every non-`Fp32` variant. FP32 also
    /// reports 32 so a caller can uniformly compute pages / capacity.
    #[inline]
    #[must_use]
    pub const fn block_size(self) -> usize {
        KV_QUANT_BLOCK_SIZE
    }

    /// The mapped `QuantKind` discriminant (returns `None` for `Fp32`).
    #[inline]
    #[must_use]
    pub const fn quant_kind(self) -> Option<QuantKind> {
        match self {
            Self::Fp32 => None,
            Self::Q4_0 => Some(QuantKind::Q4_0),
            Self::Q5_0 => Some(QuantKind::Q5_0),
            Self::Q8_0 => Some(QuantKind::Q8_0),
        }
    }

    /// Human-readable tag for logs, error messages, and `vokra-cli` output.
    #[inline]
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32",
            Self::Q4_0 => "Q4_0",
            Self::Q5_0 => "Q5_0",
            Self::Q8_0 => "Q8_0",
        }
    }

    /// Compression ratio vs FP32 (bytes-per-element ratio).
    ///
    /// Used by the M3-04 verify pipeline (T12) to summarise footprint savings
    /// per mode. FP32 reports `1.0`; every quantized variant reports the ratio
    /// as a `f32` for a clean-looking log line.
    #[inline]
    #[must_use]
    pub fn compression_ratio_vs_fp32(self) -> f32 {
        (KV_QUANT_BLOCK_SIZE * 4) as f32 / self.block_bytes() as f32
    }
}

impl Default for KvQuant {
    /// Default is [`Self::Fp32`] — every existing caller is unaffected by the
    /// M3-04 land.
    fn default() -> Self {
        Self::Fp32
    }
}

/// Non-`Fp32` KV quantization discriminant.
///
/// Kept separate from [`KvQuant`] because runtime code paths that already
/// know they are quantized (Metal / CUDA fused dequant kernels — deferred to
/// M3-04 phase 2) never need to match on the FP32 arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)] // matches llama.cpp on-wire naming
pub enum QuantKind {
    /// 4-bit signed + FP16 scale.
    Q4_0,
    /// 5-bit signed + FP16 scale.
    Q5_0,
    /// 8-bit signed + FP16 scale.
    Q8_0,
}

impl QuantKind {
    /// On-wire block size in bytes for this quant format.
    #[inline]
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => q4_0::BLOCK_BYTES,
            Self::Q5_0 => q5_0::BLOCK_BYTES,
            Self::Q8_0 => q8_0::BLOCK_BYTES,
        }
    }

    /// The [`KvQuant`] variant that wraps this discriminant.
    #[inline]
    #[must_use]
    pub const fn as_kv_quant(self) -> KvQuant {
        match self {
            Self::Q4_0 => KvQuant::Q4_0,
            Self::Q5_0 => KvQuant::Q5_0,
            Self::Q8_0 => KvQuant::Q8_0,
        }
    }
}

/// Trait every KV-cache-storable quantized block implements.
///
/// **Not** a replacement for [`KvElement`](crate::KvElement) — the FP32 path
/// keeps its `KvElement` trait unchanged, and the quantized path lives in a
/// parallel type ([`QuantizedPagedKvCache`]). Splitting the traits keeps every
/// existing consumer of `PagedKvCache<f32>` unaffected (M3-03 backward
/// compatibility, ADR M3-03 §D3).
pub trait KvQuantBlock: Copy + Sized + 'static {
    /// Number of FP32 elements per block. Fixed to
    /// [`KV_QUANT_BLOCK_SIZE`] for every M3-04 variant.
    const BLOCK_SIZE: usize = KV_QUANT_BLOCK_SIZE;
    /// The [`QuantKind`] discriminant.
    const KIND: QuantKind;

    /// Encodes a 32-element FP32 window. Panics if `input.len() != BLOCK_SIZE`.
    fn pack(input: &[f32]) -> Self;
    /// Decodes the block into `output`. Panics if `output.len() != BLOCK_SIZE`.
    fn unpack(&self, output: &mut [f32]);
}

impl KvQuantBlock for BlockQ4_0 {
    const KIND: QuantKind = QuantKind::Q4_0;
    fn pack(input: &[f32]) -> Self {
        BlockQ4_0::pack(input)
    }
    fn unpack(&self, output: &mut [f32]) {
        BlockQ4_0::unpack(self, output);
    }
}

impl KvQuantBlock for BlockQ5_0 {
    const KIND: QuantKind = QuantKind::Q5_0;
    fn pack(input: &[f32]) -> Self {
        BlockQ5_0::pack(input)
    }
    fn unpack(&self, output: &mut [f32]) {
        BlockQ5_0::unpack(self, output);
    }
}

impl KvQuantBlock for BlockQ8_0 {
    const KIND: QuantKind = QuantKind::Q8_0;
    fn pack(input: &[f32]) -> Self {
        BlockQ8_0::pack(input)
    }
    fn unpack(&self, output: &mut [f32]) {
        BlockQ8_0::unpack(self, output);
    }
}

/// Pack an arbitrary FP32 slice into a `Vec` of quantized blocks.
///
/// `input.len()` **must** be a multiple of [`KV_QUANT_BLOCK_SIZE`]; a partial
/// tail would violate the block-aligned page invariant. The caller is expected
/// to pad up-front (the paged KV cache reserves multiples of 32 by
/// construction, so this constraint is trivially satisfied for M3-04 call
/// sites).
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`](crate::VokraError::InvalidArgument)
/// on a length mismatch.
pub fn pack_slice<B: KvQuantBlock>(input: &[f32]) -> crate::Result<Vec<B>> {
    if input.len() % KV_QUANT_BLOCK_SIZE != 0 {
        return Err(crate::VokraError::InvalidArgument(format!(
            "kv_quant::pack_slice: input.len()={} not a multiple of {KV_QUANT_BLOCK_SIZE}",
            input.len()
        )));
    }
    let n_blocks = input.len() / KV_QUANT_BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks);
    for chunk in input.chunks_exact(KV_QUANT_BLOCK_SIZE) {
        out.push(B::pack(chunk));
    }
    Ok(out)
}

/// Unpack a slice of quantized blocks into a `Vec<f32>` of `blocks.len() * 32`
/// FP32 values.
#[must_use]
pub fn unpack_slice<B: KvQuantBlock>(blocks: &[B]) -> Vec<f32> {
    let mut out = vec![0.0f32; blocks.len() * KV_QUANT_BLOCK_SIZE];
    for (i, block) in blocks.iter().enumerate() {
        block.unpack(&mut out[i * KV_QUANT_BLOCK_SIZE..(i + 1) * KV_QUANT_BLOCK_SIZE]);
    }
    out
}

/// Dequantize any [`KvQuant`] discriminant against `bytes` interpreted as a
/// packed on-wire block array.
///
/// This is the **differential oracle** for the M3-04 T11 CPU scalar path: the
/// M3-04-phase-2 Metal / CUDA fused kernels must produce bit-identical output
/// against this reference (modulo FP32 GEMM rounding).
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`](crate::VokraError::InvalidArgument)
/// if `bytes.len()` is not a whole multiple of the format's block size, or if
/// `mode == KvQuant::Fp32` (FP32 payloads are not "dequantized"; the caller
/// should short-circuit that branch).
pub fn dequantize_bytes(mode: KvQuant, bytes: &[u8]) -> crate::Result<Vec<f32>> {
    match mode {
        KvQuant::Fp32 => Err(crate::VokraError::InvalidArgument(
            "kv_quant::dequantize_bytes: FP32 has no on-wire block layout".into(),
        )),
        KvQuant::Q4_0 => decode_q4_0_bytes(bytes),
        KvQuant::Q5_0 => decode_q5_0_bytes(bytes),
        KvQuant::Q8_0 => decode_q8_0_bytes(bytes),
    }
}

fn check_multiple(bytes_len: usize, block_bytes: usize) -> crate::Result<usize> {
    if bytes_len % block_bytes != 0 {
        return Err(crate::VokraError::InvalidArgument(format!(
            "kv_quant::dequantize_bytes: bytes.len()={bytes_len} not a multiple of block_bytes={block_bytes}"
        )));
    }
    Ok(bytes_len / block_bytes)
}

/// In-place variant of [`dequantize_bytes`] that writes into a caller-provided
/// slice, avoiding a per-call `Vec` allocation. Used by
/// [`dequant_gemm::dequant_gemv_scalar`] on the hot GEMV loop.
///
/// `out.len()` must equal `bytes.len() / block_bytes * KV_QUANT_BLOCK_SIZE`
/// where `block_bytes` is the format's block byte size. Rejects `mode ==
/// KvQuant::Fp32`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`](crate::VokraError::InvalidArgument)
/// on shape mismatch or `Fp32` mode.
pub fn dequantize_bytes_into(mode: KvQuant, bytes: &[u8], out: &mut [f32]) -> crate::Result<()> {
    match mode {
        KvQuant::Fp32 => Err(crate::VokraError::InvalidArgument(
            "kv_quant::dequantize_bytes_into: FP32 has no on-wire block layout".into(),
        )),
        KvQuant::Q4_0 => decode_q4_0_bytes_into(bytes, out),
        KvQuant::Q5_0 => decode_q5_0_bytes_into(bytes, out),
        KvQuant::Q8_0 => decode_q8_0_bytes_into(bytes, out),
    }
}

fn check_out_size(bytes_len: usize, block_bytes: usize, out_len: usize) -> crate::Result<usize> {
    let n_blocks = check_multiple(bytes_len, block_bytes)?;
    let expected = n_blocks * KV_QUANT_BLOCK_SIZE;
    if out_len != expected {
        return Err(crate::VokraError::InvalidArgument(format!(
            "dequantize_bytes_into: out.len()={out_len} != n_blocks({n_blocks})*BLOCK_SIZE = {expected}"
        )));
    }
    Ok(n_blocks)
}

fn parse_q4_0_block(chunk: &[u8]) -> BlockQ4_0 {
    let mut qs = [0u8; 16];
    qs.copy_from_slice(&chunk[2..18]);
    BlockQ4_0 {
        d: F16Bits(u16::from_le_bytes([chunk[0], chunk[1]])),
        qs,
    }
}

fn parse_q5_0_block(chunk: &[u8]) -> BlockQ5_0 {
    let mut qh = [0u8; 4];
    let mut qs = [0u8; 16];
    qh.copy_from_slice(&chunk[2..6]);
    qs.copy_from_slice(&chunk[6..22]);
    BlockQ5_0 {
        d: F16Bits(u16::from_le_bytes([chunk[0], chunk[1]])),
        qh,
        qs,
    }
}

fn parse_q8_0_block(chunk: &[u8]) -> BlockQ8_0 {
    let mut qs = [0i8; 32];
    for (dst, src) in qs.iter_mut().zip(chunk[2..34].iter()) {
        *dst = *src as i8;
    }
    BlockQ8_0 {
        d: F16Bits(u16::from_le_bytes([chunk[0], chunk[1]])),
        qs,
    }
}

fn decode_q4_0_bytes_into(bytes: &[u8], out: &mut [f32]) -> crate::Result<()> {
    check_out_size(bytes.len(), q4_0::BLOCK_BYTES, out.len())?;
    for (block_idx, chunk) in bytes.chunks_exact(q4_0::BLOCK_BYTES).enumerate() {
        let block = parse_q4_0_block(chunk);
        block.unpack(
            &mut out[block_idx * KV_QUANT_BLOCK_SIZE..(block_idx + 1) * KV_QUANT_BLOCK_SIZE],
        );
    }
    Ok(())
}

fn decode_q5_0_bytes_into(bytes: &[u8], out: &mut [f32]) -> crate::Result<()> {
    check_out_size(bytes.len(), q5_0::BLOCK_BYTES, out.len())?;
    for (block_idx, chunk) in bytes.chunks_exact(q5_0::BLOCK_BYTES).enumerate() {
        let block = parse_q5_0_block(chunk);
        block.unpack(
            &mut out[block_idx * KV_QUANT_BLOCK_SIZE..(block_idx + 1) * KV_QUANT_BLOCK_SIZE],
        );
    }
    Ok(())
}

fn decode_q8_0_bytes_into(bytes: &[u8], out: &mut [f32]) -> crate::Result<()> {
    check_out_size(bytes.len(), q8_0::BLOCK_BYTES, out.len())?;
    for (block_idx, chunk) in bytes.chunks_exact(q8_0::BLOCK_BYTES).enumerate() {
        let block = parse_q8_0_block(chunk);
        block.unpack(
            &mut out[block_idx * KV_QUANT_BLOCK_SIZE..(block_idx + 1) * KV_QUANT_BLOCK_SIZE],
        );
    }
    Ok(())
}

fn decode_q4_0_bytes(bytes: &[u8]) -> crate::Result<Vec<f32>> {
    let n_blocks = check_multiple(bytes.len(), q4_0::BLOCK_BYTES)?;
    let mut out = vec![0.0f32; n_blocks * KV_QUANT_BLOCK_SIZE];
    decode_q4_0_bytes_into(bytes, &mut out)?;
    Ok(out)
}

fn decode_q5_0_bytes(bytes: &[u8]) -> crate::Result<Vec<f32>> {
    let n_blocks = check_multiple(bytes.len(), q5_0::BLOCK_BYTES)?;
    let mut out = vec![0.0f32; n_blocks * KV_QUANT_BLOCK_SIZE];
    decode_q5_0_bytes_into(bytes, &mut out)?;
    Ok(out)
}

fn decode_q8_0_bytes(bytes: &[u8]) -> crate::Result<Vec<f32>> {
    let n_blocks = check_multiple(bytes.len(), q8_0::BLOCK_BYTES)?;
    let mut out = vec![0.0f32; n_blocks * KV_QUANT_BLOCK_SIZE];
    decode_q8_0_bytes_into(bytes, &mut out)?;
    Ok(out)
}

/// Serialize a [`BlockQ4_0`] to its 18-byte on-wire representation. Used by
/// the CPU differential oracle (T11) and by anyone constructing a golden test
/// vector.
#[must_use]
pub fn block_q4_0_bytes(block: &BlockQ4_0) -> [u8; q4_0::BLOCK_BYTES] {
    let mut out = [0u8; q4_0::BLOCK_BYTES];
    out[0..2].copy_from_slice(&block.d.0.to_le_bytes());
    out[2..18].copy_from_slice(&block.qs);
    out
}

/// Serialize a [`BlockQ5_0`] to its 22-byte on-wire representation.
#[must_use]
pub fn block_q5_0_bytes(block: &BlockQ5_0) -> [u8; q5_0::BLOCK_BYTES] {
    let mut out = [0u8; q5_0::BLOCK_BYTES];
    out[0..2].copy_from_slice(&block.d.0.to_le_bytes());
    out[2..6].copy_from_slice(&block.qh);
    out[6..22].copy_from_slice(&block.qs);
    out
}

/// Serialize a [`BlockQ8_0`] to its 34-byte on-wire representation.
#[must_use]
pub fn block_q8_0_bytes(block: &BlockQ8_0) -> [u8; q8_0::BLOCK_BYTES] {
    let mut out = [0u8; q8_0::BLOCK_BYTES];
    out[0..2].copy_from_slice(&block.d.0.to_le_bytes());
    for i in 0..32 {
        out[2 + i] = block.qs[i] as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_quant_tags_and_sizes() {
        assert_eq!(KvQuant::Fp32.tag(), "FP32");
        assert_eq!(KvQuant::Q4_0.tag(), "Q4_0");
        assert_eq!(KvQuant::Q5_0.tag(), "Q5_0");
        assert_eq!(KvQuant::Q8_0.tag(), "Q8_0");

        assert_eq!(KvQuant::Fp32.block_bytes(), 128);
        assert_eq!(KvQuant::Q4_0.block_bytes(), 18);
        assert_eq!(KvQuant::Q5_0.block_bytes(), 22);
        assert_eq!(KvQuant::Q8_0.block_bytes(), 34);

        // Every non-FP32 variant reports 32 elements per block.
        assert_eq!(KvQuant::Q4_0.block_size(), 32);
        assert_eq!(KvQuant::Q5_0.block_size(), 32);
        assert_eq!(KvQuant::Q8_0.block_size(), 32);
    }

    #[test]
    fn compression_ratios_are_ordered() {
        // Q4_0 > Q5_0 > Q8_0 in compression ratio (smaller block = more
        // compression). FP32 = 1.0.
        assert!(
            KvQuant::Q4_0.compression_ratio_vs_fp32() > KvQuant::Q5_0.compression_ratio_vs_fp32()
        );
        assert!(
            KvQuant::Q5_0.compression_ratio_vs_fp32() > KvQuant::Q8_0.compression_ratio_vs_fp32()
        );
        assert!(KvQuant::Q8_0.compression_ratio_vs_fp32() > 1.0);
        assert_eq!(KvQuant::Fp32.compression_ratio_vs_fp32(), 1.0);
    }

    #[test]
    fn kv_quant_default_is_fp32() {
        assert_eq!(KvQuant::default(), KvQuant::Fp32);
        assert!(KvQuant::default().quant_kind().is_none());
    }

    #[test]
    fn quant_kind_round_trip_via_kv_quant() {
        for q in [KvQuant::Q4_0, KvQuant::Q5_0, KvQuant::Q8_0] {
            let kind = q.quant_kind().unwrap();
            assert_eq!(kind.as_kv_quant(), q);
        }
    }

    #[test]
    fn pack_slice_all_three_formats_round_trip() {
        // 2 blocks × 32 elems = 64 elem window.
        let input: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0 * 2.0 - 1.0).collect();

        let q4 = pack_slice::<BlockQ4_0>(&input).unwrap();
        let q5 = pack_slice::<BlockQ5_0>(&input).unwrap();
        let q8 = pack_slice::<BlockQ8_0>(&input).unwrap();
        assert_eq!(q4.len(), 2);
        assert_eq!(q5.len(), 2);
        assert_eq!(q8.len(), 2);

        let q4_out = unpack_slice(&q4);
        let q5_out = unpack_slice(&q5);
        let q8_out = unpack_slice(&q8);
        assert_eq!(q4_out.len(), 64);
        assert_eq!(q5_out.len(), 64);
        assert_eq!(q8_out.len(), 64);

        // Q4_0 is coarsest; use its bound.
        let amax = input.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        let tol_q4 = 0.6 * (amax / 7.0);
        for (x, y) in input.iter().zip(&q4_out) {
            assert!((x - y).abs() <= tol_q4);
        }
    }

    #[test]
    fn pack_slice_rejects_non_multiple_length() {
        let input = vec![0.0f32; 33];
        let err = pack_slice::<BlockQ8_0>(&input).unwrap_err();
        assert!(matches!(err, crate::VokraError::InvalidArgument(_)));
    }

    #[test]
    fn dequantize_bytes_round_trip_all_three() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let q4 = BlockQ4_0::pack(&input);
        let q5 = BlockQ5_0::pack(&input);
        let q8 = BlockQ8_0::pack(&input);

        let q4_bytes = block_q4_0_bytes(&q4);
        let q5_bytes = block_q5_0_bytes(&q5);
        let q8_bytes = block_q8_0_bytes(&q8);

        let q4_out = dequantize_bytes(KvQuant::Q4_0, &q4_bytes).unwrap();
        let q5_out = dequantize_bytes(KvQuant::Q5_0, &q5_bytes).unwrap();
        let q8_out = dequantize_bytes(KvQuant::Q8_0, &q8_bytes).unwrap();
        assert_eq!(q4_out.len(), 32);
        assert_eq!(q5_out.len(), 32);
        assert_eq!(q8_out.len(), 32);

        // Compare against a direct unpack.
        let mut q4_direct = vec![0.0f32; 32];
        let mut q5_direct = vec![0.0f32; 32];
        let mut q8_direct = vec![0.0f32; 32];
        q4.unpack(&mut q4_direct);
        q5.unpack(&mut q5_direct);
        q8.unpack(&mut q8_direct);
        assert_eq!(q4_out, q4_direct);
        assert_eq!(q5_out, q5_direct);
        assert_eq!(q8_out, q8_direct);
    }

    #[test]
    fn dequantize_bytes_rejects_fp32() {
        let err = dequantize_bytes(KvQuant::Fp32, &[0u8; 128]).unwrap_err();
        assert!(matches!(err, crate::VokraError::InvalidArgument(_)));
    }

    #[test]
    fn dequantize_bytes_rejects_misaligned_length() {
        // Q8_0 block is 34 bytes; 33 is misaligned.
        let err = dequantize_bytes(KvQuant::Q8_0, &[0u8; 33]).unwrap_err();
        assert!(matches!(err, crate::VokraError::InvalidArgument(_)));
    }

    /// Serialize a block, deserialize it back, then check the round-trip is
    /// bit-identical against the original block's `unpack` output. This is the
    /// contract the M3-04-phase-2 Metal / CUDA fused dequant kernels must
    /// preserve (differential oracle).
    #[test]
    fn on_wire_serialization_is_bit_identical() {
        // Q8_0 has full 8-bit precision — exact after ser/de.
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let block = BlockQ8_0::pack(&input);
        let bytes = block_q8_0_bytes(&block);
        let decoded = dequantize_bytes(KvQuant::Q8_0, &bytes).unwrap();
        let mut direct = vec![0.0f32; 32];
        block.unpack(&mut direct);
        assert_eq!(decoded, direct);
    }
}
