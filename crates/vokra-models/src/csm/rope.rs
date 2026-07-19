//! Llama-3 scaled RoPE (M4-05, ADR M4-05 §D3).
//!
//! CSM-1B builds both stacks on torchtune's `llama3_2` with
//! `scale_factor=32` (`SesameAILabs/csm` `models.py`), which selects
//! torchtune's `Llama3ScaledRoPE` (`torchtune/models/llama3_1/
//! _position_embeddings.py`). Two properties differ from the plain
//! [`crate::voxtral::text_decoder::rope_apply`]:
//!
//! 1. **Frequency rescale** — each base frequency
//!    `freq_j = base^(-2j/head_dim)` is remapped by wavelength bands:
//!    - `wavelen < old_context_len / high_freq_factor` → unchanged;
//!    - `wavelen > old_context_len / low_freq_factor` → `freq / scale_factor`;
//!    - in between → `smooth`-interpolated:
//!      `smooth = (old_context_len / wavelen - low_freq_factor) /
//!                (high_freq_factor - low_freq_factor)`,
//!      `freq' = (1 - smooth) * freq / scale_factor + smooth * freq`.
//! 2. **Adjacent-pair rotation** — torchtune reshapes the head vector to
//!    `[head_dim/2, 2]` and rotates `(x[2j], x[2j+1])` pairs, whereas the
//!    Voxtral/CosyVoice2 `rope_apply` rotates the half-split pair
//!    `(x[j], x[j+half])` (HF Llama layout). The two conventions are
//!    permutations of each other; CSM parity (T23/T24) compares against
//!    the torchtune-based reference, so this module implements the
//!    **adjacent-pair** convention and documents it here. Bit-level
//!    confirmation against the real checkpoint is the T29 flip-the-switch
//!    (no upstream value is invented before then; the formulas above are
//!    verbatim transcriptions — ADR M4-05 §D3).
//!
//! `scaling = None` keeps the base frequencies untouched (plain RoPE
//! frequencies, adjacent-pair application).

use vokra_core::{Result, VokraError};

use super::config::CsmRopeScaling;

/// Precomputes the per-pair rotation frequencies for a head of `head_dim`
/// channels: `freq_j = base^(-2j/head_dim)` for `j = 0..head_dim/2`,
/// remapped by the Llama-3 wavelength bands when `scaling` is present
/// (ADR M4-05 §D3 formula, transcribed from torchtune).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on an odd / zero `head_dim`, a
/// non-positive `base`, or an ill-formed scaling triple.
pub fn llama3_inv_freqs(
    head_dim: usize,
    base: f32,
    scaling: Option<&CsmRopeScaling>,
) -> Result<Vec<f32>> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "llama3_inv_freqs: head_dim ({head_dim}) must be even and > 0"
        )));
    }
    if base <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "llama3_inv_freqs: rope base must be > 0, got {base}"
        )));
    }
    if let Some(s) = scaling {
        if s.scale_factor <= 0.0
            || s.old_context_len == 0
            || s.high_freq_factor <= s.low_freq_factor
        {
            return Err(VokraError::InvalidArgument(format!(
                "llama3_inv_freqs: ill-formed scaling (scale_factor={}, low={}, high={}, \
                 old_context_len={})",
                s.scale_factor, s.low_freq_factor, s.high_freq_factor, s.old_context_len
            )));
        }
    }
    let half = head_dim / 2;
    let mut freqs = Vec::with_capacity(half);
    for j in 0..half {
        let freq = base.powf(-2.0 * (j as f32) / (head_dim as f32));
        freqs.push(match scaling {
            None => freq,
            Some(s) => scale_one_freq(freq, s),
        });
    }
    Ok(freqs)
}

/// The torchtune `Llama3ScaledRoPE.apply_scaling` band map for one base
/// frequency (ADR M4-05 §D3).
fn scale_one_freq(freq: f32, s: &CsmRopeScaling) -> f32 {
    let wavelen = 2.0 * std::f32::consts::PI / freq;
    let low_freq_wavelen = s.old_context_len as f32 / s.low_freq_factor;
    let high_freq_wavelen = s.old_context_len as f32 / s.high_freq_factor;
    if wavelen < high_freq_wavelen {
        freq
    } else if wavelen > low_freq_wavelen {
        freq / s.scale_factor
    } else {
        let smooth = (s.old_context_len as f32 / wavelen - s.low_freq_factor)
            / (s.high_freq_factor - s.low_freq_factor);
        (1.0 - smooth) * freq / s.scale_factor + smooth * freq
    }
}

/// Applies adjacent-pair RoPE in place over `x = [seq_len, head_dim]`
/// row-major, using precomputed `inv_freqs` (one per pair —
/// [`llama3_inv_freqs`]). Row `i` rotates by angle
/// `(position_offset + i) * inv_freqs[j]` on the pair
/// `(x[2j], x[2j+1])` — the torchtune `reshape(..., -1, 2)` convention
/// (module docs).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any shape mismatch.
pub fn rope_apply_adjacent(
    x: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    inv_freqs: &[f32],
    position_offset: usize,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply_adjacent: head_dim ({head_dim}) must be even"
        )));
    }
    let half = head_dim / 2;
    if inv_freqs.len() != half {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply_adjacent: inv_freqs len {} != head_dim/2 {}",
            inv_freqs.len(),
            half
        )));
    }
    if x.len() != seq_len * head_dim {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply_adjacent: x len {} != seq_len*head_dim {}",
            x.len(),
            seq_len * head_dim
        )));
    }
    for i in 0..seq_len {
        let m = (position_offset + i) as f32;
        let row = &mut x[i * head_dim..(i + 1) * head_dim];
        for (j, &f) in inv_freqs.iter().enumerate() {
            let angle = m * f;
            let (s, c) = angle.sin_cos();
            let a = row[2 * j];
            let b = row[2 * j + 1];
            row[2 * j] = a * c - b * s;
            row[2 * j + 1] = a * s + b * c;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaling() -> CsmRopeScaling {
        CsmRopeScaling {
            scale_factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            old_context_len: 8192,
        }
    }

    #[test]
    fn unscaled_freqs_match_standard_formula() {
        let f = llama3_inv_freqs(8, 500_000.0, None).expect("freqs");
        for (j, got) in f.iter().enumerate() {
            let want = 500_000.0f32.powf(-2.0 * (j as f32) / 8.0);
            assert_eq!(*got, want, "pair {j}");
        }
    }

    #[test]
    fn scaling_keeps_high_freq_band_untouched_and_divides_low_band() {
        // freq_0 = 1.0 → wavelen 2π ≪ 8192/4: high band, unchanged.
        let s = scaling();
        let unscaled = llama3_inv_freqs(64, 500_000.0, None).expect("freqs");
        let scaled = llama3_inv_freqs(64, 500_000.0, Some(&s)).expect("freqs");
        assert_eq!(scaled[0], unscaled[0], "high-frequency band must not move");
        // The lowest frequency of a 500k-base, 64-dim head has wavelength
        // 2π·500000^(62/64) ≈ 1.7e6 ≫ 8192: low band → divided by 32.
        let last = unscaled.last().unwrap();
        assert!(
            (scaled.last().unwrap() - last / 32.0).abs() <= f32::EPSILON * last,
            "low-frequency band divides by scale_factor"
        );
        // Every scaled frequency stays within [freq/32, freq].
        for (j, (&sc, &un)) in scaled.iter().zip(unscaled.iter()).enumerate() {
            assert!(
                sc <= un + f32::EPSILON && sc >= un / 32.0 - f32::EPSILON,
                "pair {j}: {sc} outside [{}, {un}]",
                un / 32.0
            );
        }
    }

    #[test]
    fn smooth_band_matches_transcribed_formula() {
        // Pick a frequency whose wavelength lands strictly between
        // old/4 = 2048 and old/1 = 8192, then check the interpolation
        // against the ADR §D3 formula evaluated by hand here.
        let s = scaling();
        let freq = 2.0 * std::f32::consts::PI / 4000.0; // wavelen = 4000
        let got = scale_one_freq(freq, &s);
        let smooth = (8192.0 / 4000.0 - 1.0) / (4.0 - 1.0);
        let want = (1.0 - smooth) * freq / 32.0 + smooth * freq;
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn position_zero_is_identity() {
        let f = llama3_inv_freqs(6, 10_000.0, None).expect("freqs");
        let mut x = vec![0.3, -0.7, 1.5, 2.5, -0.1, 0.9];
        let orig = x.clone();
        rope_apply_adjacent(&mut x, 1, 6, &f, 0).expect("apply");
        assert_eq!(x, orig, "angle 0 rotation is the identity");
    }

    #[test]
    fn rotation_preserves_pair_norms() {
        let s = scaling();
        let f = llama3_inv_freqs(8, 500_000.0, Some(&s)).expect("freqs");
        let mut x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let orig = x.clone();
        rope_apply_adjacent(&mut x, 2, 8, &f, 5).expect("apply");
        for row in 0..2 {
            for j in 0..4 {
                let o = &orig[row * 8 + 2 * j..row * 8 + 2 * j + 2];
                let n = &x[row * 8 + 2 * j..row * 8 + 2 * j + 2];
                let no = (o[0] * o[0] + o[1] * o[1]).sqrt();
                let nn = (n[0] * n[0] + n[1] * n[1]).sqrt();
                assert!((no - nn).abs() < 1e-5, "row {row} pair {j}: {no} vs {nn}");
            }
        }
        assert_ne!(x, orig, "non-zero position must rotate");
    }

    #[test]
    fn incremental_offset_matches_bulk() {
        // rope(x, seq=2, offset=3) row 1 == rope(row1, seq=1, offset=4).
        let f = llama3_inv_freqs(4, 500_000.0, None).expect("freqs");
        let row0 = [0.1f32, 0.2, 0.3, 0.4];
        let row1 = [-0.5f32, 0.6, -0.7, 0.8];
        let mut bulk = [row0, row1].concat();
        rope_apply_adjacent(&mut bulk, 2, 4, &f, 3).expect("bulk");
        let mut inc = row1.to_vec();
        rope_apply_adjacent(&mut inc, 1, 4, &f, 4).expect("inc");
        assert_eq!(&bulk[4..8], inc.as_slice());
    }

    #[test]
    fn shape_errors_are_loud() {
        let f = llama3_inv_freqs(4, 10_000.0, None).expect("freqs");
        let mut x = vec![0.0f32; 7];
        assert!(rope_apply_adjacent(&mut x, 2, 4, &f, 0).is_err());
        assert!(llama3_inv_freqs(5, 10_000.0, None).is_err(), "odd head_dim");
        assert!(llama3_inv_freqs(4, 0.0, None).is_err(), "zero base");
        let bad = CsmRopeScaling {
            scale_factor: 0.0,
            ..scaling()
        };
        assert!(llama3_inv_freqs(4, 10_000.0, Some(&bad)).is_err());
    }
}
