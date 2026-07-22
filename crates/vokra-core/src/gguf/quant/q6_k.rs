//! `Q6_K` dequantization (ggml type tag 14).
//!
//! On-disk `block_q6_K` (210 bytes, 256 elements), transcribed from ggml
//! `k_quants.h`:
//!
//! | field    | bytes | meaning                                            |
//! |----------|-------|----------------------------------------------------|
//! | `ql`     | 128   | low 4 bits of each 6-bit quant (two per byte)       |
//! | `qh`     | 64    | high 2 bits of each 6-bit quant (four per byte)     |
//! | `scales` | 16    | one `int8` scale per 16-element sub-block           |
//! | `d`      | 2     | `f16` super-block scale                             |
//!
//! `Q6_K` is symmetric (no min): `y = d · sc_is · q`, where the 6-bit quant is
//! reassembled as `(ql_low4 | (qh_hi2 << 4)) − 32 ∈ [−32, 31]`, and `sc_is` is
//! the signed sub-block scale (`is` walks the 16 scales in the ggml order
//! `is+0, is+2, is+4, is+6` across the four quarters of each 128-element half).

use super::{f16_to_f32, n_blocks};
use crate::gguf::tensor::QK_K;
// M5-03-T05: `Vec` and the `vec!` macro are `alloc` (core-clean); the no_std
// subset imports them (inert under std, where they are in the prelude).
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// 128 low-nibble bytes + 64 high-2-bit bytes + 16 `int8` scales + `f16` `d`.
const BLOCK_BYTES: usize = 210;
const QL_OFF: usize = 0;
const QH_OFF: usize = 128;
const SCALES_OFF: usize = 192;
const D_OFF: usize = 208;

/// Dequantizes `n_elements` (a whole multiple of [`QK_K`]) `Q6_K` values.
///
/// The caller guarantees `bytes.len() == n_blocks * 210`, so every indexed
/// range is in bounds.
pub(super) fn dequantize(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let nb = n_blocks(n_elements);
    let mut out = vec![0.0f32; n_elements];

    for i in 0..nb {
        let block = &bytes[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[D_OFF], block[D_OFF + 1]]));
        let ql_all = &block[QL_OFF..QL_OFF + 128];
        let qh_all = &block[QH_OFF..QH_OFF + 64];
        let sc_all = &block[SCALES_OFF..SCALES_OFF + 16];
        let y = &mut out[i * QK_K..(i + 1) * QK_K];

        // Two 128-element halves; each advances ql by 64, qh by 32, scales by 8.
        for half in 0..2 {
            let ql = &ql_all[half * 64..half * 64 + 64];
            let qh = &qh_all[half * 32..half * 32 + 32];
            let sc = &sc_all[half * 8..half * 8 + 8];
            let y = &mut y[half * 128..half * 128 + 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * f32::from(sc[is] as i8) * q1 as f32;
                y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
                y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
                y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Maps an output index to its `(half, quarter, l)` coordinates. `n =
    /// 128*half + 32*quarter + l`.
    fn coords(n: usize) -> (usize, usize, usize) {
        let half = n / 128;
        let o = n % 128;
        (half, o / 32, o % 32)
    }

    /// The signed sub-block scale index for output element `n` (matches the
    /// `is + 2*quarter` schedule inside the decoder, offset by the half).
    fn scale_index(n: usize) -> usize {
        let (half, quarter, l) = coords(n);
        8 * half + (l / 16) + 2 * quarter
    }

    /// Assembles one 210-byte `Q6_K` block from 256 chosen signed quants
    /// (`q ∈ [-32, 31]`) and 16 signed sub-block scales.
    fn build_block(d: u16, scales: [i8; 16], quants: [i32; 256]) -> Vec<u8> {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for (n, &qn) in quants.iter().enumerate() {
            let (half, quarter, l) = coords(n);
            let stored = (qn + 32) as u8; // 0..63
            let low4 = stored & 0xF;
            let hi2 = (stored >> 4) & 3;
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
        let mut b = Vec::with_capacity(BLOCK_BYTES);
        b.extend_from_slice(&ql);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&scales.map(|s| s as u8));
        b.extend_from_slice(&d.to_le_bytes());
        b
    }

    #[test]
    fn closed_form_matches_signed_quants_and_scales() {
        let d = 0x3C00; // 1.0
        // Distinct signed scales so a wrong `is + 2*quarter` schedule is caught.
        let mut scales = [0i8; 16];
        for (k, s) in scales.iter_mut().enumerate() {
            *s = (k as i8) - 8; // -8..7
        }
        // Full signed range, cycling through [-32, 31].
        let mut quants = [0i32; 256];
        for (n, q) in quants.iter_mut().enumerate() {
            *q = (n % 64) as i32 - 32;
        }
        let block = build_block(d, scales, quants);

        let out = dequantize(&block, 256);
        let df = f16_to_f32(d);
        for (n, &got) in out.iter().enumerate() {
            let want = df * f32::from(scales[scale_index(n)]) * quants[n] as f32;
            assert!(
                (got - want).abs() < 1e-3,
                "element {n}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn zero_quant_is_negative_offset() {
        // stored nibble 0 => q = -32; with d=1, sc=1 => y = -32 everywhere.
        let block = build_block(0x3C00, [1i8; 16], [-32i32; 256]);
        let out = dequantize(&block, 256);
        assert!(out.iter().all(|&v| (v + 32.0).abs() < 1e-3));
    }

    #[test]
    fn negative_scale_flips_sign() {
        // q = 16, sc = -2, d = 1 => y = -32.
        let block = build_block(0x3C00, [-2i8; 16], [16i32; 256]);
        let out = dequantize(&block, 256);
        assert!(out.iter().all(|&v| (v + 32.0).abs() < 1e-3));
    }
}
