//! Minimal, std-only K-quant quantizer (offline; `vokra-convert` only).
//!
//! Encodes dense `f32` weights into ggml `Q4_K` / `Q5_K` / `Q6_K` super-blocks
//! (the same on-disk layout the runtime's [`vokra_core::gguf::quant`] decoder
//! reads). This is the *offline* counterpart to the runtime dequantizer: it
//! lets a real quantized Whisper GGUF be produced in-repo from the F32
//! safetensors, and — crucially — gives a fully-internal **quantize → dequant
//! round-trip oracle** (bounded per-block error) so K-quant loader correctness
//! is proven with zero external artifact (NFR-QL-01).
//!
//! # Scope: correctness, not compression quality
//!
//! Each super-block uses a single affine scale (uniform 6-bit sub-scales), not
//! ggml's error-minimizing `make_qkx2_quants` search. The output is a *valid*
//! K-quant block that both this crate's decoder and ggml/llama.cpp read
//! identically; the reconstruction error is bounded by one quantization step
//! (plus the `f16` scale-storage error), which is exactly what the oracle
//! asserts. A quality-optimizing quantizer is a documented follow-up.

use std::fmt;

use vokra_core::gguf::GgmlType;
use vokra_core::gguf::tensor::QK_K;

/// Error from the offline K-quant quantizer.
#[derive(Debug)]
pub enum QuantizeError {
    /// The input length is not a whole number of [`QK_K`] super-blocks.
    NotBlockAligned {
        /// Requested target dtype.
        dtype: GgmlType,
        /// The (mis-sized) element count.
        len: usize,
    },
    /// The requested dtype is not a supported K-quant target.
    UnsupportedTarget(GgmlType),
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBlockAligned { dtype, len } => write!(
                f,
                "cannot quantize {len} elements to {dtype:?}: not a multiple of {QK_K}"
            ),
            Self::UnsupportedTarget(d) => {
                write!(f, "{d:?} is not a K-quant target (use Q4_K / Q5_K / Q6_K)")
            }
        }
    }
}

impl std::error::Error for QuantizeError {}

/// Quantizes dense `f32` `data` into the on-disk payload for `dtype`.
///
/// `dtype` must be `Q4_K` / `Q5_K` / `Q6_K` and `data.len()` a multiple of
/// [`QK_K`] (256). The returned bytes are exactly
/// [`dtype.payload_size(data.len())`](GgmlType::payload_size) long and decode
/// back through [`vokra_core::gguf::quant::dequantize`].
pub fn quantize(dtype: GgmlType, data: &[f32]) -> Result<Vec<u8>, QuantizeError> {
    match dtype {
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {}
        other => return Err(QuantizeError::UnsupportedTarget(other)),
    }
    if data.len() % QK_K != 0 {
        return Err(QuantizeError::NotBlockAligned {
            dtype,
            len: data.len(),
        });
    }
    Ok(match dtype {
        GgmlType::Q4K => quantize_q4_k(data),
        GgmlType::Q5K => quantize_q5_k(data),
        GgmlType::Q6K => quantize_q6_k(data),
        _ => unreachable!("guarded above"),
    })
}

/// Minimum and maximum of a non-empty block.
fn min_max(block: &[f32]) -> (f32, f32) {
    let mut lo = block[0];
    let mut hi = block[0];
    for &v in &block[1..] {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Encodes an asymmetric K-quant super-block (`Q4_K` / `Q5_K`): one affine map
/// `y = step·q + base`, `base = min(lo, 0) ≤ 0`, `q ∈ [0, max_q]`, with uniform
/// 6-bit sub-scales (`sc = m = 63`, i.e. `scales = [0xFF; 12]`). Returns the
/// per-element quants plus the `f16` `d` / `dmin` super-block scales.
fn affine_quants(block: &[f32], max_q: u8) -> (Vec<u8>, u16, u16) {
    let (lo, hi) = min_max(block);
    let base = lo.min(0.0);
    let range = hi - base;
    let step = range / f32::from(max_q);
    // d·63 == step and dmin·63 == -base, so every sub-block reconstructs the
    // same affine map (sc = m = 63 everywhere).
    let d = if step > 0.0 { step / 63.0 } else { 0.0 };
    let dmin = if base < 0.0 { -base / 63.0 } else { 0.0 };

    let quants = block
        .iter()
        .map(|&v| {
            if step > 0.0 {
                (((v - base) / step).round()).clamp(0.0, f32::from(max_q)) as u8
            } else {
                0
            }
        })
        .collect();
    (quants, f32_to_f16(d), f32_to_f16(dmin))
}

/// Quantizes `data` (a multiple of 256) into `Q4_K` blocks (144 bytes each).
fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK_K * 144);
    for block in data.chunks_exact(QK_K) {
        let (q, d, dmin) = affine_quants(block, 15);
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&dmin.to_le_bytes());
        out.extend_from_slice(&[0xFFu8; 12]); // sc = m = 63 for all 8 sub-blocks
        // Interleave 4-bit quants: within each 64-element chunk, element (64k+l)
        // is a low nibble and (64k+32+l) a high nibble of qs[32k+l].
        let mut qs = [0u8; 128];
        for k in 0..4 {
            for l in 0..32 {
                qs[32 * k + l] = (q[64 * k + l] & 0xF) | ((q[64 * k + 32 + l] & 0xF) << 4);
            }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantizes `data` (a multiple of 256) into `Q5_K` blocks (176 bytes each).
fn quantize_q5_k(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK_K * 176);
    for block in data.chunks_exact(QK_K) {
        let (q, d, dmin) = affine_quants(block, 31);
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for k in 0..4 {
            for l in 0..32 {
                let lo = q[64 * k + l]; // 0..31
                let hi = q[64 * k + 32 + l];
                qs[32 * k + l] = (lo & 0xF) | ((hi & 0xF) << 4);
                if lo & 0x10 != 0 {
                    qh[l] |= 1 << (2 * k);
                }
                if hi & 0x10 != 0 {
                    qh[l] |= 1 << (2 * k + 1);
                }
            }
        }
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&dmin.to_le_bytes());
        out.extend_from_slice(&[0xFFu8; 12]);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantizes `data` (a multiple of 256) into `Q6_K` blocks (210 bytes each).
///
/// `Q6_K` is symmetric (`y = d·sc·q`, no min): the block uses a single scale
/// over `[-amax, amax]` with uniform `int8` sub-scales `sc = 31`.
fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK_K * 210);
    for block in data.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let step = amax / 31.0;
        let d = if step > 0.0 { step / 31.0 } else { 0.0 }; // d·31 == step

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for (n, &v) in block.iter().enumerate() {
            let q = if step > 0.0 {
                (v / step).round().clamp(-32.0, 31.0) as i32
            } else {
                0
            };
            let stored = (q + 32) as u8; // 0..63
            let low4 = stored & 0xF;
            let hi2 = (stored >> 4) & 3;
            // Inverse of the decoder's (half, quarter, l) addressing.
            let half = n / 128;
            let o = n % 128;
            let quarter = o / 32;
            let l = o % 32;
            match quarter {
                0 => {
                    ql[64 * half + l] |= low4;
                    qh[32 * half + l] |= hi2;
                }
                1 => {
                    ql[64 * half + l + 32] |= low4;
                    qh[32 * half + l] |= hi2 << 2;
                }
                2 => {
                    ql[64 * half + l] |= low4 << 4;
                    qh[32 * half + l] |= hi2 << 4;
                }
                _ => {
                    ql[64 * half + l + 32] |= low4 << 4;
                    qh[32 * half + l] |= hi2 << 6;
                }
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&[31u8; 16]); // int8 sub-scale 31 for all 16
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

/// IEEE-754 single → half precision, round-to-nearest-ties-to-even.
///
/// The encoder counterpart to [`vokra_core::gguf::quant::f16_to_f32`]; handles
/// normals, subnormals, overflow → inf and NaN. Pinned by round-tripping
/// exactly-representable values back through the decoder in the tests.
fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let aexp = ((x >> 23) & 0xff) as i32; // biased f32 exponent
    let mant = x & 0x007f_ffff;

    if aexp == 0xff {
        // Inf (mant == 0) or NaN.
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }

    let e = aexp - 127 + 15; // rebias to f16
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → signed zero
        }
        // Subnormal: restore the implicit leading 1, then shift into range.
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32; // in [14, 24]
        let mut h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }

    // Normal f16: drop 13 mantissa bits, round to nearest even.
    let mut h = ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // a carry into the exponent field is intentional and correct
    }
    sign | h
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::quant;

    /// Deterministic, dependency-free pseudo-random f32 stream spanning
    /// negatives and positives (mimicking zero-centred weights).
    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                // SplitMix64 step → a value in roughly [-1.5, 1.5].
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 40) as f32 / (1u32 << 24) as f32; // [0,1)
                (u - 0.5) * 3.0
            })
            .collect()
    }

    /// Round-trip oracle: for each 256-block, the max |dequant − original| must
    /// not exceed one quantization step plus the small `f16` scale-storage
    /// slack. `q_step` is the per-level width for the format.
    fn assert_roundtrip(dtype: GgmlType, data: &[f32], levels: f32) {
        let bytes = quantize(dtype, data).unwrap();
        // Payload length matches the block-aware size the reader expects.
        assert_eq!(
            bytes.len() as u64,
            dtype.payload_size(data.len() as u64).unwrap()
        );
        let back = quant::dequantize(dtype, &bytes, data.len()).unwrap();
        assert_eq!(back.len(), data.len());

        for (bi, block) in data.chunks_exact(QK_K).enumerate() {
            let (lo, hi) = min_max(block);
            let maxabs = lo.abs().max(hi.abs());
            // Asymmetric formats span [min(lo,0), hi]; symmetric spans amax.
            let range = if dtype == GgmlType::Q6K {
                2.0 * maxabs
            } else {
                hi - lo.min(0.0)
            };
            let q_step = range / levels;
            // one step (covers ≤ half-step rounding + f16 scale error on q) plus
            // a relative floor for the min-term f16 storage error.
            let tol = q_step + maxabs * 5.0e-3 + 1.0e-6;
            for (i, &x) in block.iter().enumerate() {
                let y = back[bi * QK_K + i];
                assert!(
                    (y - x).abs() <= tol,
                    "{dtype:?} block {bi} elem {i}: |{y} - {x}| = {} > {tol}",
                    (y - x).abs()
                );
            }
        }
    }

    #[test]
    fn q4_k_roundtrip_is_within_one_step() {
        let data = synth(256 * 3, 1);
        assert_roundtrip(GgmlType::Q4K, &data, 15.0);
    }

    #[test]
    fn q5_k_roundtrip_is_tighter_than_q4() {
        let data = synth(256 * 3, 2);
        assert_roundtrip(GgmlType::Q5K, &data, 31.0);
    }

    #[test]
    fn q6_k_roundtrip_is_within_one_step() {
        let data = synth(256 * 3, 3);
        assert_roundtrip(GgmlType::Q6K, &data, 31.0);
    }

    #[test]
    fn q5_k_error_is_no_worse_than_q4_k_on_same_data() {
        // More levels ⇒ no larger max error: a direct differential between the
        // two asymmetric formats on identical input.
        let data = synth(256 * 2, 7);
        let err = |dt: GgmlType| -> f32 {
            let bytes = quantize(dt, &data).unwrap();
            let back = quant::dequantize(dt, &bytes, data.len()).unwrap();
            data.iter()
                .zip(&back)
                .map(|(&x, &y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(err(GgmlType::Q5K) <= err(GgmlType::Q4K) + 1e-6);
    }

    #[test]
    fn constant_and_zero_blocks_roundtrip() {
        // All-zero and constant (negative) blocks are the degenerate range==0
        // cases; they must reconstruct within the f16 relative slack.
        for dtype in [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let zeros = vec![0.0f32; QK_K];
            let back = quant::dequantize(dtype, &quantize(dtype, &zeros).unwrap(), QK_K).unwrap();
            assert!(back.iter().all(|&v| v == 0.0), "{dtype:?} zeros");

            let c = vec![-1.5f32; QK_K];
            let back = quant::dequantize(dtype, &quantize(dtype, &c).unwrap(), QK_K).unwrap();
            assert!(
                back.iter().all(|&v| (v + 1.5).abs() < 5.0e-3),
                "{dtype:?} constant"
            );
        }
    }

    #[test]
    fn f32_to_f16_roundtrips_exactly_representable_values() {
        // Values exactly representable in f16 must survive encode→decode intact.
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 2.0, -2.0, 63.0, 0.25, 100.0] {
            assert_eq!(quant::f16_to_f32(f32_to_f16(v)), v, "roundtrip {v}");
        }
        assert!(quant::f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(quant::f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn rejects_unaligned_and_unsupported() {
        assert!(matches!(
            quantize(GgmlType::Q4K, &[0.0; 100]),
            Err(QuantizeError::NotBlockAligned { .. })
        ));
        assert!(matches!(
            quantize(GgmlType::F32, &[0.0; 256]),
            Err(QuantizeError::UnsupportedTarget(_))
        ));
    }
}
