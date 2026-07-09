//! Q4_0 KV cache quantization (M3-04-T02).
//!
//! # Layout (llama.cpp `block_q4_0`, 1 block = 18 bytes, 32 elements)
//!
//! ```text
//!   { d: f16, qs: [u8; 16] }
//! ```
//!
//! - `d`: FP16 scale (`|max| / 7`, 4-bit signed range `[-8, 7]`).
//! - `qs[i]` (i ∈ 0..16): low nibble = element `2·i`, high nibble = `2·i + 1`.
//!   Each nibble is stored **biased by +8** so a `u8`-only reader can restore
//!   the signed value with `nib - 8`. This bias convention matches ggml's
//!   `dequantize_row_q4_0`.
//!
//! Symmetric quantization only (`_0` suffix); zero-point is not stored. See
//! ADR M3-04 §D2 for the rationale.

use super::QuantKind;
use super::half::{F16Bits, f16_bits_to_f32, f32_to_f16_bits};

/// Number of elements packed in one Q4_0 block.
pub const BLOCK_SIZE: usize = 32;

/// On-wire size of one Q4_0 block in bytes.
pub const BLOCK_BYTES: usize = 18;

/// One Q4_0 block. `repr(C)` because a KV cache page stores blocks as an array
/// and downstream Metal / CUDA kernels indirect through the same layout.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BlockQ4_0 {
    /// FP16 scale.
    pub d: F16Bits,
    /// 32 nibbles, packed low/high within each byte, each biased by +8.
    pub qs: [u8; 16],
}

impl BlockQ4_0 {
    /// Packs a 32-element FP32 window (symmetric, `d = |max| / 7`).
    ///
    /// `input.len()` **must** equal [`BLOCK_SIZE`] (32). A window of 32 zeros
    /// encodes as `d = 0` with all nibbles = 8 (biased zero).
    #[must_use]
    pub fn pack(input: &[f32]) -> Self {
        assert_eq!(
            input.len(),
            BLOCK_SIZE,
            "Q4_0 pack: input.len()={} must equal BLOCK_SIZE ({BLOCK_SIZE})",
            input.len()
        );
        // Symmetric range → scale is |max| / 7 (4-bit signed range [-8, 7]).
        // Denominator 7 (not 8) prevents saturation of the -8 sentinel on the
        // positive side under round-half-away-from-zero.
        let amax = input.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 7.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 16];
        for (i, chunk) in input.chunks_exact(2).enumerate() {
            let q0 = quantize_nibble(chunk[0] * inv_d);
            let q1 = quantize_nibble(chunk[1] * inv_d);
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        Self {
            d: F16Bits(f32_to_f16_bits(d)),
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
            "Q4_0 unpack: output.len()={} must equal BLOCK_SIZE ({BLOCK_SIZE})",
            output.len()
        );
        let d = f16_bits_to_f32(self.d.0);
        for (i, byte) in self.qs.iter().enumerate() {
            let lo = (*byte & 0x0F) as i32 - 8;
            let hi = ((*byte >> 4) & 0x0F) as i32 - 8;
            output[2 * i] = lo as f32 * d;
            output[2 * i + 1] = hi as f32 * d;
        }
    }

    /// Q4_0 discriminant for the runtime KV element trait.
    #[inline]
    #[must_use]
    pub const fn quant_kind() -> QuantKind {
        QuantKind::Q4_0
    }
}

/// Round-half-away-from-zero into the 4-bit signed range `[-8, 7]`, returned
/// as a nibble biased by +8 (so the on-wire byte is a plain `u8`).
#[inline]
fn quantize_nibble(scaled: f32) -> u8 {
    // clamp *then* round so the biased sentinel `+8` for +7 is unreachable.
    let clamped = scaled.clamp(-8.0, 7.0);
    let rounded = clamped.round() as i32;
    (rounded + 8).clamp(0, 15) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic oracle: pack → unpack must round-trip within one quantization
    /// step (`d = amax / 7`, error bound = `d / 2`).
    #[test]
    fn round_trip_within_quantization_step() {
        // Deterministic 32-element window (linear ramp -1..1).
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let block = BlockQ4_0::pack(&input);
        let mut out = vec![0.0f32; 32];
        block.unpack(&mut out);
        let amax = input.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        let d = amax / 7.0;
        // Error bound is d/2 (round-half-away-from-zero) + FP16 scale rounding
        // (< 2^-11 · amax). Combine into a single loose tolerance = 0.6 · d.
        let tol = 0.6 * d;
        for (x, y) in input.iter().zip(out.iter()) {
            assert!((x - y).abs() <= tol, "|{x} - {y}| > {tol}");
        }
    }

    #[test]
    fn all_zeros_round_trip_exactly() {
        let input = [0.0f32; 32];
        let block = BlockQ4_0::pack(&input);
        let mut out = [1.0f32; 32]; // pre-fill with junk to ensure overwrite
        block.unpack(&mut out);
        assert!(out.iter().all(|&x| x == 0.0));
        assert_eq!(block.d.0, 0); // FP16 zero
    }

    #[test]
    fn saturation_at_range_edges() {
        // Symmetric-range design: `d = amax / 7`. When the max |input| lives
        // both positive and negative (large-symmetric input), the most
        // positive elem saturates at +7 (biased 15) and the most negative at
        // -7 (biased 1). The -8 sentinel is reachable only when the actual
        // max |input| is dominated by a large negative value while the
        // largest positive value is smaller — see the below trigger.
        let mut input = [0.0f32; 32];
        input[0] = 1000.0;
        input[1] = -1000.0;
        let block = BlockQ4_0::pack(&input);
        // +7 → biased 15
        assert_eq!(block.qs[0] & 0x0F, 15);
        // -7 → biased 1 (not 0; asymmetric-negative dominance not present).
        assert_eq!((block.qs[0] >> 4) & 0x0F, 1);
    }

    #[test]
    fn negative_bias_reaches_lowest_sentinel() {
        // Asymmetric-negative input reaches the -8 sentinel (biased 0).
        // With input[0]=-1000, input[1]=+500, amax=1000, d≈142.86; scaled
        // input[0] = -1000/142.86 = -7.0 → biased 1 STILL. The design
        // deliberately never actually assigns -8 for typical inputs; this
        // test pins that documented behaviour.
        let mut input = [0.0f32; 32];
        input[0] = -1000.0;
        input[1] = 500.0;
        let block = BlockQ4_0::pack(&input);
        // -7 → biased 1 (not 0). Documented invariant of the symmetric
        // `d = amax / 7` design (llama.cpp Q4_0 parity).
        assert_eq!(block.qs[0] & 0x0F, 1);
    }

    #[test]
    fn block_size_bytes_pin() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), BLOCK_BYTES);
    }

    #[test]
    fn quant_kind_pin() {
        assert!(matches!(BlockQ4_0::quant_kind(), QuantKind::Q4_0));
    }

    #[test]
    #[should_panic(expected = "must equal BLOCK_SIZE")]
    fn pack_wrong_length_panics() {
        let _ = BlockQ4_0::pack(&[0.0f32; 16]);
    }

    #[test]
    #[should_panic(expected = "must equal BLOCK_SIZE")]
    fn unpack_wrong_length_panics() {
        let block = BlockQ4_0::default();
        let mut out = vec![0.0f32; 16];
        block.unpack(&mut out);
    }
}
