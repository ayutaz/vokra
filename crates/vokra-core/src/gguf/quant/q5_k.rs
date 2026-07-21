//! `Q5_K` dequantization (ggml type tag 13).
//!
//! On-disk `block_q5_K` (176 bytes, 256 elements), transcribed from ggml
//! `k_quants.h`:
//!
//! | field    | bytes | meaning                                            |
//! |----------|-------|----------------------------------------------------|
//! | `d`      | 2     | `f16` super-block scale for the 6-bit sub-scales    |
//! | `dmin`   | 2     | `f16` super-block scale for the 6-bit sub-mins      |
//! | `scales` | 12    | eight 6-bit sub-scales + eight 6-bit sub-mins       |
//! | `qh`     | 32    | one high bit per element (the 5th quant bit)        |
//! | `qs`     | 128   | 256 × low-4 quant bits (two per byte)               |
//!
//! Identical to `Q4_K` except each quant gains a 5th (high) bit drawn from
//! `qh`: `y = d·sc_j·(q5) − dmin·m_j` with `q5 ∈ [0,31]`. The sub-scales share
//! `Q4_K`'s [`get_scale_min_k4`](super) packing. Within each 64-element chunk
//! `k`, the low 32 elements take `qh` bit `2k` and the high 32 take bit `2k+1`.

use super::{f16_to_f32, get_scale_min_k4, n_blocks};
use crate::gguf::tensor::QK_K;
// M5-03-T05: `Vec` and the `vec!` macro are `alloc` (core-clean); the no_std
// subset imports them (inert under std, where they are in the prelude).
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// `f16` `d` + `f16` `dmin` + 12 scale bytes + 32 high-bit bytes + 128 quants.
const BLOCK_BYTES: usize = 176;
const SCALES_OFF: usize = 4;
const QH_OFF: usize = 16;
const QS_OFF: usize = 48;

/// Dequantizes `n_elements` (a whole multiple of [`QK_K`]) `Q5_K` values.
///
/// The caller guarantees `bytes.len() == n_blocks * 176`, so every indexed
/// range is in bounds.
pub(super) fn dequantize(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let nb = n_blocks(n_elements);
    let mut out = vec![0.0f32; n_elements];

    for i in 0..nb {
        let block = &bytes[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[SCALES_OFF..SCALES_OFF + 12];
        let qh = &block[QH_OFF..QH_OFF + 32];
        let qs = &block[QS_OFF..QS_OFF + 128];
        let y = &mut out[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut ql = 0usize; // running offset into qs
        let mut base = 0usize; // running offset into y
        // `u1`/`u2` select which `qh` bit feeds the low/high 32 of this chunk;
        // they walk bits (0,1) → (2,3) → (4,5) → (6,7) across the 4 chunks.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        while base < QK_K {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * f32::from(sc0);
            let m1 = dmin * f32::from(m0);
            let (sc1, m1s) = get_scale_min_k4(is + 1, scales);
            let d2 = d * f32::from(sc1);
            let m2 = dmin * f32::from(m1s);

            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16u16 } else { 0 };
                let q = u16::from(qs[ql + l] & 0xF) + hi;
                y[base + l] = d1 * f32::from(q) - m1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16u16 } else { 0 };
                let q = u16::from(qs[ql + l] >> 4) + hi;
                y[base + 32 + l] = d2 * f32::from(q) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
            base += 64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs sub-scales/mins into the 12-byte layout (inverse of
    /// `get_scale_min_k4`; shared shape with `Q4_K`).
    fn pack_scales(sc: [u8; 8], m: [u8; 8]) -> [u8; 12] {
        let mut s = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                s[j] = sc[j] & 63;
                s[j + 4] = m[j] & 63;
            } else {
                s[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
                s[j - 4] |= (sc[j] >> 4) << 6;
                s[j] |= (m[j] >> 4) << 6;
            }
        }
        s
    }

    /// Assembles one 176-byte `Q5_K` block from 256 chosen 5-bit quants.
    fn build_block(d: u16, dmin: u16, scales: [u8; 12], quants5: [u8; 256]) -> Vec<u8> {
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for k in 0..4 {
            for l in 0..32 {
                let lo_elem = quants5[64 * k + l]; // 0..31
                let hi_elem = quants5[64 * k + 32 + l];
                qs[32 * k + l] = (lo_elem & 0xF) | ((hi_elem & 0xF) << 4);
                if lo_elem & 0x10 != 0 {
                    qh[l] |= 1 << (2 * k);
                }
                if hi_elem & 0x10 != 0 {
                    qh[l] |= 1 << (2 * k + 1);
                }
            }
        }
        let mut b = Vec::with_capacity(BLOCK_BYTES);
        b.extend_from_slice(&d.to_le_bytes());
        b.extend_from_slice(&dmin.to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&qs);
        b
    }

    #[test]
    fn closed_form_exercises_the_fifth_bit() {
        let d = 0x3C00; // 1.0
        let dmin = 0x3800; // 0.5
        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let m = [2u8, 2, 2, 2, 2, 2, 2, 2];
        let scales = pack_scales(sc, m);
        // Quant value = index mod 32, so half the elements set the 5th bit.
        let mut quants5 = [0u8; 256];
        for (idx, q) in quants5.iter_mut().enumerate() {
            *q = (idx % 32) as u8;
        }
        let block = build_block(d, dmin, scales, quants5);

        let out = dequantize(&block, 256);
        let df = f16_to_f32(d);
        let dminf = f16_to_f32(dmin);
        for (n, &got) in out.iter().enumerate() {
            let k = n / 64;
            let sub = if n % 64 < 32 { 2 * k } else { 2 * k + 1 };
            let q = (n % 32) as f32;
            let want = df * f32::from(sc[sub]) * q - dminf * f32::from(m[sub]);
            assert!(
                (got - want).abs() < 1e-4,
                "element {n}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn max_quant_uses_high_bit() {
        // q = 31 everywhere (5th bit set) with sc=1,m=0,d=1 => y = 31.
        let block = build_block(0x3C00, 0x0000, pack_scales([1; 8], [0; 8]), [31u8; 256]);
        let out = dequantize(&block, 256);
        assert!(out.iter().all(|&v| (v - 31.0).abs() < 1e-4));
    }
}
