//! Q5_0 KV cache quantization (M3-04-T03).
//!
//! # Layout (llama.cpp `block_q5_0`, 1 block = 22 bytes, 32 elements)
//!
//! ```text
//!   { d: f16, qh: [u8; 4], qs: [u8; 16] }
//! ```
//!
//! - `d`: FP16 scale (`|max| / 15`, 5-bit signed range `[-16, 15]`).
//! - `qh` (4 bytes = 32 bits): high bit of each 5-bit element. `qh[i / 8] &
//!   (1 << (i % 8))` is bit 5 of element `i` (0 or 1). Little-endian bit
//!   packing along `i`.
//! - `qs[i]` (i ∈ 0..16): low nibble = low 4 bits of element `2·i`, high
//!   nibble = low 4 bits of element `2·i + 1`. Each nibble is stored **biased
//!   by +16** internally so the on-wire representation and the reconstructed
//!   signed range line up (`(qs_lo | qh_lo << 4) - 16`).
//!
//! Symmetric quantization only (`_0` suffix); zero-point is not stored.

use super::QuantKind;
use super::half::{F16Bits, f16_bits_to_f32, f32_to_f16_bits};

/// Number of elements packed in one Q5_0 block.
pub const BLOCK_SIZE: usize = 32;

/// On-wire size of one Q5_0 block in bytes.
pub const BLOCK_BYTES: usize = 22;

/// One Q5_0 block. `repr(C)` because a KV cache page stores blocks as an array
/// and downstream Metal / CUDA kernels indirect through the same layout.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BlockQ5_0 {
    /// FP16 scale.
    pub d: F16Bits,
    /// One high bit per element (32 elements × 1 bit = 4 bytes).
    pub qh: [u8; 4],
    /// Low 4 bits per element, packed low/high within each byte.
    pub qs: [u8; 16],
}

impl BlockQ5_0 {
    /// Packs a 32-element FP32 window (symmetric, `d = |max| / 15`).
    ///
    /// `input.len()` **must** equal [`BLOCK_SIZE`] (32).
    #[must_use]
    pub fn pack(input: &[f32]) -> Self {
        assert_eq!(
            input.len(),
            BLOCK_SIZE,
            "Q5_0 pack: input.len()={} must equal BLOCK_SIZE ({BLOCK_SIZE})",
            input.len()
        );
        // Symmetric range → scale = |max| / 15 (5-bit signed range [-16, 15]).
        let amax = input.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 15.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qh = [0u8; 4];
        let mut qs = [0u8; 16];
        for i in 0..BLOCK_SIZE {
            let biased = quantize_5bit_biased(input[i] * inv_d);
            // Low 4 bits into the packed nibble; bit 4 (bit 5 of the signed
            // range) into the high-bit byte.
            let lo4 = biased & 0x0F;
            let hi1 = (biased >> 4) & 0x01;
            if i % 2 == 0 {
                qs[i / 2] |= lo4;
            } else {
                qs[i / 2] |= lo4 << 4;
            }
            qh[i / 8] |= hi1 << (i % 8);
        }
        Self {
            d: F16Bits(f32_to_f16_bits(d)),
            qh,
            qs,
        }
    }

    /// Writes 32 dequantized FP32 values into `output`.
    ///
    /// `output.len()` **must** equal [`BLOCK_SIZE`] (32).
    pub fn unpack(&self, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            BLOCK_SIZE,
            "Q5_0 unpack: output.len()={} must equal BLOCK_SIZE ({BLOCK_SIZE})",
            output.len()
        );
        let d = f16_bits_to_f32(self.d.0);
        for (i, slot) in output.iter_mut().enumerate() {
            let lo4 = if i % 2 == 0 {
                self.qs[i / 2] & 0x0F
            } else {
                (self.qs[i / 2] >> 4) & 0x0F
            };
            let hi1 = (self.qh[i / 8] >> (i % 8)) & 0x01;
            let biased = (hi1 << 4) | lo4;
            let signed = biased as i32 - 16;
            *slot = signed as f32 * d;
        }
    }

    /// Q5_0 discriminant for the runtime KV element trait.
    #[inline]
    #[must_use]
    pub const fn quant_kind() -> QuantKind {
        QuantKind::Q5_0
    }
}

/// Round-half-away-from-zero into the 5-bit signed range `[-16, 15]`, returned
/// as a biased-by-16 value in `[0, 31]`.
#[inline]
fn quantize_5bit_biased(scaled: f32) -> u8 {
    let clamped = scaled.clamp(-16.0, 15.0);
    let rounded = clamped.round() as i32;
    (rounded + 16).clamp(0, 31) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic oracle: pack → unpack must round-trip within one quantization
    /// step (`d = amax / 15`, error bound = `d / 2`).
    #[test]
    fn round_trip_within_quantization_step() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let block = BlockQ5_0::pack(&input);
        let mut out = vec![0.0f32; 32];
        block.unpack(&mut out);
        let amax = input.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        let d = amax / 15.0;
        let tol = 0.6 * d;
        for (x, y) in input.iter().zip(out.iter()) {
            assert!((x - y).abs() <= tol, "|{x} - {y}| > {tol}");
        }
    }

    /// Q5_0 error bound is strictly tighter than Q4_0 (5-bit vs 4-bit) for the
    /// same input window. Regression-anchors the two formats' relative
    /// precision.
    #[test]
    fn q5_0_precision_beats_q4_0() {
        use super::super::q4_0::BlockQ4_0;
        // Same ramp as above.
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let q4 = BlockQ4_0::pack(&input);
        let q5 = BlockQ5_0::pack(&input);
        let mut q4_out = vec![0.0f32; 32];
        let mut q5_out = vec![0.0f32; 32];
        q4.unpack(&mut q4_out);
        q5.unpack(&mut q5_out);
        // Sum of squared errors should be lower for Q5_0.
        let sse_q4: f32 = input
            .iter()
            .zip(&q4_out)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let sse_q5: f32 = input
            .iter()
            .zip(&q5_out)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(
            sse_q5 <= sse_q4,
            "Q5_0 SSE {sse_q5} not ≤ Q4_0 SSE {sse_q4}"
        );
    }

    #[test]
    fn all_zeros_round_trip_exactly() {
        let input = [0.0f32; 32];
        let block = BlockQ5_0::pack(&input);
        let mut out = [1.0f32; 32];
        block.unpack(&mut out);
        assert!(out.iter().all(|&x| x == 0.0));
        assert_eq!(block.d.0, 0);
    }

    #[test]
    fn saturation_at_range_edges() {
        // Symmetric design (`d = amax / 15`) — most positive saturates at +15
        // (biased 31), most negative at -15 (biased 1). The -16 sentinel is
        // documented as unreachable for typical inputs (llama.cpp Q5_0
        // parity, same rationale as Q4_0 / Q8_0).
        let mut input = [0.0f32; 32];
        input[0] = 1000.0;
        input[1] = -1000.0;
        let block = BlockQ5_0::pack(&input);
        let biased_0 = (block.qs[0] & 0x0F) | ((block.qh[0] & 0x01) << 4);
        assert_eq!(biased_0, 31); // +15
        let biased_1 = ((block.qs[0] >> 4) & 0x0F) | (((block.qh[0] >> 1) & 0x01) << 4);
        assert_eq!(biased_1, 1); // -15 (not -16)
    }

    #[test]
    fn block_size_bytes_pin() {
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), BLOCK_BYTES);
    }

    #[test]
    fn quant_kind_pin() {
        assert!(matches!(BlockQ5_0::quant_kind(), QuantKind::Q5_0));
    }
}
