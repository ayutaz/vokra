//! Q8_0 KV cache quantization (M3-04-T04).
//!
//! # Layout (llama.cpp `block_q8_0`, 1 block = 34 bytes, 32 elements)
//!
//! ```text
//!   { d: f16, qs: [i8; 32] }
//! ```
//!
//! - `d`: FP16 scale (`|max| / 127`, 8-bit signed range `[-128, 127]`).
//! - `qs`: 32 signed 8-bit values, stored as `i8` directly (no bias). The
//!   dequantized value is `qs[i] as f32 * d`.
//!
//! Symmetric quantization only (`_0` suffix); zero-point is not stored.

use super::QuantKind;
use super::half::{F16Bits, f16_bits_to_f32, f32_to_f16_bits};

/// Number of elements packed in one Q8_0 block.
pub const BLOCK_SIZE: usize = 32;

/// On-wire size of one Q8_0 block in bytes.
pub const BLOCK_BYTES: usize = 34;

/// One Q8_0 block. `repr(C)` because a KV cache page stores blocks as an array
/// and downstream Metal / CUDA kernels indirect through the same layout.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BlockQ8_0 {
    /// FP16 scale.
    pub d: F16Bits,
    /// 32 signed 8-bit quantized values.
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    /// Packs a 32-element FP32 window (symmetric, `d = |max| / 127`).
    ///
    /// `input.len()` **must** equal [`BLOCK_SIZE`] (32).
    #[must_use]
    pub fn pack(input: &[f32]) -> Self {
        assert_eq!(
            input.len(),
            BLOCK_SIZE,
            "Q8_0 pack: input.len()={} must equal BLOCK_SIZE ({BLOCK_SIZE})",
            input.len()
        );
        // Symmetric range → scale = |max| / 127 (8-bit signed range [-128, 127]).
        // Denominator 127 (not 128) prevents saturation of the -128 sentinel
        // on the positive side under round-half-away-from-zero.
        let amax = input.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0i8; 32];
        for (i, &x) in input.iter().enumerate() {
            let scaled = x * inv_d;
            let clamped = scaled.clamp(-127.0, 127.0);
            qs[i] = clamped.round() as i8;
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
            "Q8_0 unpack: output.len()={} must equal BLOCK_SIZE ({BLOCK_SIZE})",
            output.len()
        );
        let d = f16_bits_to_f32(self.d.0);
        for (slot, q) in output.iter_mut().zip(self.qs.iter()) {
            *slot = *q as f32 * d;
        }
    }

    /// Q8_0 discriminant for the runtime KV element trait.
    #[inline]
    #[must_use]
    pub const fn quant_kind() -> QuantKind {
        QuantKind::Q8_0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_within_quantization_step() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let block = BlockQ8_0::pack(&input);
        let mut out = vec![0.0f32; 32];
        block.unpack(&mut out);
        let amax = input.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        let d = amax / 127.0;
        let tol = 0.6 * d;
        for (x, y) in input.iter().zip(out.iter()) {
            assert!((x - y).abs() <= tol, "|{x} - {y}| > {tol}");
        }
    }

    /// Q8_0 precision is strictly better than Q5_0 and Q4_0 for the same
    /// input.
    #[test]
    fn q8_0_precision_beats_q5_0_and_q4_0() {
        use super::super::q4_0::BlockQ4_0;
        use super::super::q5_0::BlockQ5_0;
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0 * 2.0 - 1.0).collect();
        let q4 = BlockQ4_0::pack(&input);
        let q5 = BlockQ5_0::pack(&input);
        let q8 = BlockQ8_0::pack(&input);
        let mut q4_out = vec![0.0f32; 32];
        let mut q5_out = vec![0.0f32; 32];
        let mut q8_out = vec![0.0f32; 32];
        q4.unpack(&mut q4_out);
        q5.unpack(&mut q5_out);
        q8.unpack(&mut q8_out);
        let sse =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum() };
        let sse_q4 = sse(&input, &q4_out);
        let sse_q5 = sse(&input, &q5_out);
        let sse_q8 = sse(&input, &q8_out);
        assert!(
            sse_q8 <= sse_q5,
            "Q8_0 SSE {sse_q8} not ≤ Q5_0 SSE {sse_q5}"
        );
        assert!(
            sse_q8 <= sse_q4,
            "Q8_0 SSE {sse_q8} not ≤ Q4_0 SSE {sse_q4}"
        );
    }

    #[test]
    fn all_zeros_round_trip_exactly() {
        let input = [0.0f32; 32];
        let block = BlockQ8_0::pack(&input);
        let mut out = [1.0f32; 32];
        block.unpack(&mut out);
        assert!(out.iter().all(|&x| x == 0.0));
        assert_eq!(block.d.0, 0);
    }

    #[test]
    fn saturation_at_range_edges() {
        let mut input = [0.0f32; 32];
        input[0] = 1000.0;
        input[1] = -1000.0;
        let block = BlockQ8_0::pack(&input);
        // Positive saturates to +127; negative saturates to -127 (not -128, by
        // convention — same as llama.cpp `ggml_quantize_row_q8_0`).
        assert_eq!(block.qs[0], 127);
        assert_eq!(block.qs[1], -127);
    }

    #[test]
    fn block_size_bytes_pin() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), BLOCK_BYTES);
    }

    #[test]
    fn quant_kind_pin() {
        assert!(matches!(BlockQ8_0::quant_kind(), QuantKind::Q8_0));
    }
}
