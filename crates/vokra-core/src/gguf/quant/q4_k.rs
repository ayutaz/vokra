//! `Q4_K` dequantization (ggml type tag 12).
//!
//! On-disk `block_q4_K` (144 bytes, 256 elements), transcribed from ggml
//! `k_quants.h`:
//!
//! | field    | bytes | meaning                                            |
//! |----------|-------|----------------------------------------------------|
//! | `d`      | 2     | `f16` super-block scale for the 6-bit sub-scales    |
//! | `dmin`   | 2     | `f16` super-block scale for the 6-bit sub-mins      |
//! | `scales` | 12    | eight 6-bit sub-scales + eight 6-bit sub-mins       |
//! | `qs`     | 128   | 256 × 4-bit quants (two per byte)                   |
//!
//! Each of the eight 32-element sub-blocks `j` reconstructs
//! `y = d·sc_j·q − dmin·m_j`, with `q ∈ [0,15]`, `sc_j, m_j ∈ [0,63]`. The
//! 6-bit sub-scales/mins are bit-packed across the 12 `scales` bytes exactly as
//! ggml `get_scale_min_k4` unpacks them.

use super::{f16_to_f32, get_scale_min_k4, n_blocks};
use crate::gguf::tensor::QK_K;
// M5-03-T05: `Vec` and the `vec!` macro are `alloc` (core-clean); the no_std
// subset imports them (inert under std, where they are in the prelude).
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// `f16` `d` + `f16` `dmin` + 12 packed scale bytes + 128 quant bytes.
const BLOCK_BYTES: usize = 144;
const SCALES_OFF: usize = 4;
const QS_OFF: usize = 16;

/// Dequantizes `n_elements` (a whole multiple of [`QK_K`]) `Q4_K` values.
///
/// The caller guarantees `bytes.len() == n_blocks * 144` (the dispatch in
/// [`super::dequantize`] validates this), so every indexed range is in bounds.
pub(super) fn dequantize(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let nb = n_blocks(n_elements);
    let mut out = vec![0.0f32; n_elements];

    for i in 0..nb {
        let block = &bytes[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[SCALES_OFF..SCALES_OFF + 12];
        let qs = &block[QS_OFF..QS_OFF + 128];
        let y = &mut out[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut q = 0usize; // running offset into qs (advances 32 per 64-block)
        let mut base = 0usize; // running offset into y
        while base < QK_K {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * f32::from(sc0);
            let m1 = dmin * f32::from(m0);
            let (sc1, m1s) = get_scale_min_k4(is + 1, scales);
            let d2 = d * f32::from(sc1);
            let m2 = dmin * f32::from(m1s);

            for l in 0..32 {
                y[base + l] = d1 * f32::from(qs[q + l] & 0xF) - m1;
            }
            for l in 0..32 {
                y[base + 32 + l] = d2 * f32::from(qs[q + l] >> 4) - m2;
            }
            q += 32;
            is += 2;
            base += 64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs eight 6-bit sub-scales and eight 6-bit sub-mins into the 12-byte
    /// `scales` layout — the exact inverse of [`get_scale_min_k4`] (ggml's
    /// forward packing). Used as an independent in-test encoder so the
    /// closed-form oracle pins the bit-packing without any external reference.
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

    /// Assembles one 144-byte `Q4_K` block from semantic fields.
    fn build_block(d: u16, dmin: u16, scales: [u8; 12], quants: [u8; 256]) -> Vec<u8> {
        let mut b = Vec::with_capacity(BLOCK_BYTES);
        b.extend_from_slice(&d.to_le_bytes());
        b.extend_from_slice(&dmin.to_le_bytes());
        b.extend_from_slice(&scales);
        // Pack 256 4-bit quants into 128 bytes with the sub-block interleave the
        // dequant expects: within each 64-element chunk k, element (64k+l) is a
        // low nibble and (64k+32+l) is a high nibble of qs[32k+l].
        let mut qs = [0u8; 128];
        for k in 0..4 {
            for l in 0..32 {
                let lo = quants[64 * k + l] & 0xF;
                let hi = quants[64 * k + 32 + l] & 0xF;
                qs[32 * k + l] = lo | (hi << 4);
            }
        }
        b.extend_from_slice(&qs);
        b
    }

    #[test]
    fn closed_form_matches_distinct_scales_and_quants() {
        // Chosen so d·sc·q and dmin·m land on exact binary fractions (no f32
        // rounding), letting us assert bit-exact equality with the formula.
        let d = 0x3C00; // 1.0
        let dmin = 0x3800; // 0.5
        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let m = [8u8, 7, 6, 5, 4, 3, 2, 1];
        let scales = pack_scales(sc, m);
        // Quant value = element index mod 16, so every nibble position differs.
        let mut quants = [0u8; 256];
        for (idx, q) in quants.iter_mut().enumerate() {
            *q = (idx % 16) as u8;
        }
        let block = build_block(d, dmin, scales, quants);

        let out = dequantize(&block, 256);

        // Recompute the expected value with the ggml sub-block schedule: element
        // n is in 64-chunk k=n/64; the first 32 of a chunk use sub-scale index
        // 2k, the second 32 use 2k+1.
        let df = f16_to_f32(d);
        let dminf = f16_to_f32(dmin);
        for (n, &got) in out.iter().enumerate() {
            let k = n / 64;
            let sub = if n % 64 < 32 { 2 * k } else { 2 * k + 1 };
            let q = (n % 16) as f32;
            let want = df * f32::from(sc[sub]) * q - dminf * f32::from(m[sub]);
            assert!(
                (got - want).abs() < 1e-4,
                "element {n}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn uniform_block_is_a_known_constant() {
        // Constant-block oracle: all sub-scales/mins equal and all quants equal
        // => every output equals d·sc·q − dmin·m, one known value.
        let d = 0x4000; // 2.0
        let dmin = 0x3C00; // 1.0
        let scales = pack_scales([3; 8], [5; 8]);
        let quants = [4u8; 256]; // q = 4 everywhere
        let block = build_block(d, dmin, scales, quants);

        let out = dequantize(&block, 256);
        // 2.0*3*4 - 1.0*5 = 24 - 5 = 19.
        assert!(out.iter().all(|&v| (v - 19.0).abs() < 1e-4));
    }

    #[test]
    fn two_blocks_decode_independently() {
        let block0 = build_block(0x3C00, 0x0000, pack_scales([1; 8], [0; 8]), [7u8; 256]);
        let block1 = build_block(0x3C00, 0x0000, pack_scales([2; 8], [0; 8]), [3u8; 256]);
        let mut bytes = block0;
        bytes.extend_from_slice(&block1);

        let out = dequantize(&bytes, 512);
        // Block 0: 1.0*1*7 = 7; block 1: 1.0*2*3 = 6.
        assert!(out[..256].iter().all(|&v| (v - 7.0).abs() < 1e-4));
        assert!(out[256..].iter().all(|&v| (v - 6.0).abs() < 1e-4));
    }
}
