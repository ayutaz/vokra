//! Discrete cosine transform, type II (M0-04-T15; FR-OP-03).
//!
//! DCT-II over the innermost axis: `C[k] = Σ_i x[i]·cos(π·k·(2i+1) / (2N))`.
//! The [`Normalization`] attribute selects the scaling:
//!
//! - [`Normalization::Ortho`] — the orthonormal DCT-II (`scipy` `norm="ortho"`,
//!   torchaudio's MFCC matrix): `C[0]·√(1/N)`, `C[k]·√(2/N)` for `k ≥ 1`;
//! - [`Normalization::Backward`] — the unnormalized DCT-II (`scipy` default):
//!   `2·C[k]`;
//! - [`Normalization::Forward`] — `Backward` scaled by `1/N`.
//!
//! Only DCT-II is implemented; it is the type the MFCC path
//! ([`crate::mfcc`]) needs. Other DCT types are added if a model requires them.

use vokra_core::ir::graph::{DctAttrs, Normalization};

/// Applies a DCT-II to each row of `input`.
///
/// `input` is row-major `[rows, n]`; the result is row-major `[rows, n_out]`
/// where `n_out = attrs.n_out.unwrap_or(n)` (leading coefficients kept).
///
/// # Panics
///
/// Panics if `input.len() != rows * n`, if `n == 0`, or if `attrs.n_out`
/// exceeds `n`.
pub fn dct(input: &[f32], rows: usize, n: usize, attrs: &DctAttrs) -> Vec<f32> {
    assert!(n > 0, "DCT length must be non-zero");
    assert_eq!(input.len(), rows * n, "DCT input shape mismatch");
    let n_out = attrs.n_out.unwrap_or(n);
    assert!(n_out <= n, "n_out exceeds transform length");

    // Basis[k*n + i] = cos(π·k·(2i+1) / (2N)), precomputed in f64.
    let mut basis = vec![0.0f64; n_out * n];
    for k in 0..n_out {
        for i in 0..n {
            let angle = std::f64::consts::PI * (k as f64) * (2 * i + 1) as f64 / (2.0 * n as f64);
            basis[k * n + i] = angle.cos();
        }
    }

    // Per-coefficient scale.
    let scale = coeff_scales(n, n_out, attrs.normalization);

    let mut out = vec![0.0f32; rows * n_out];
    for r in 0..rows {
        let row = &input[r * n..(r + 1) * n];
        for k in 0..n_out {
            let mut acc = 0.0f64;
            let brow = &basis[k * n..(k + 1) * n];
            for (b, x) in brow.iter().zip(row) {
                acc += b * (*x as f64);
            }
            out[r * n_out + k] = (acc * scale[k]) as f32;
        }
    }
    out
}

/// Per-coefficient scale factors for the selected normalization.
fn coeff_scales(n: usize, n_out: usize, norm: Normalization) -> Vec<f64> {
    let nf = n as f64;
    (0..n_out)
        .map(|k| match norm {
            Normalization::Ortho => {
                if k == 0 {
                    (1.0 / nf).sqrt()
                } else {
                    (2.0 / nf).sqrt()
                }
            }
            Normalization::Backward => 2.0,
            Normalization::Forward => 2.0 / nf,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_row_has_only_dc_component() {
        // DCT-II of a constant sequence puts all energy in coefficient 0.
        let n = 8;
        let input = vec![2.0f32; n];
        let out = dct(&input, 1, n, &DctAttrs::new());
        // Ortho DC = sqrt(1/N)·Σ x = sqrt(1/8)·16 = 16/√8 ≈ 5.65685.
        assert!((out[0] - 16.0 / (8.0f32).sqrt()).abs() < 1e-4, "{}", out[0]);
        for c in &out[1..] {
            assert!(c.abs() < 1e-4, "nonzero AC {c}");
        }
    }

    #[test]
    fn ortho_dct_is_energy_preserving() {
        // Orthonormal DCT preserves the L2 norm (Parseval).
        let n = 16;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let out = dct(&input, 1, n, &DctAttrs::new());
        let e_in: f32 = input.iter().map(|v| v * v).sum();
        let e_out: f32 = out.iter().map(|v| v * v).sum();
        assert!((e_in - e_out).abs() < 1e-3, "{e_in} vs {e_out}");
    }

    #[test]
    fn known_two_point_dct() {
        // N=2, ortho: C0 = (x0+x1)/√2, C1 = (x0−x1)/√2.
        let input = [3.0f32, 1.0];
        let out = dct(&input, 1, 2, &DctAttrs::new());
        assert!((out[0] - 4.0 / (2.0f32).sqrt()).abs() < 1e-5);
        assert!((out[1] - 2.0 / (2.0f32).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn n_out_truncates_coefficients() {
        let n = 10;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let attrs = DctAttrs {
            n_out: Some(4),
            normalization: Normalization::Ortho,
        };
        let out = dct(&input, 1, n, &attrs);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn multiple_rows_are_independent() {
        let n = 4;
        let input = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let out = dct(&input, 2, n, &DctAttrs::new());
        // Row 0 constant → only DC.
        assert!(out[1].abs() < 1e-5 && out[2].abs() < 1e-5 && out[3].abs() < 1e-5);
        // Row 1 non-constant → some AC energy.
        assert!(out[5].abs() > 1e-3);
    }

    #[test]
    fn backward_and_forward_norm_two_point_dct() {
        // Re-derived from the DCT-II definition C[k] = Σ_i x[i]·cos(π·k·(2i+1)/2N).
        // For x = [3, 1], N = 2:
        //   C[0] = 3·cos0 + 1·cos0                        = 4
        //   C[1] = 3·cos(π/4) + 1·cos(3π/4) = (3−1)·√2/2  = √2
        // Backward scales every coefficient by 2      → [8, 2√2] = [8.0, 2.8284271]
        // Forward  scales every coefficient by 2/N = 1 → [4, √2 ] = [4.0, 1.4142135]
        use std::f32::consts::SQRT_2;
        let input = [3.0f32, 1.0];

        let backward = dct(
            &input,
            1,
            2,
            &DctAttrs {
                n_out: None,
                normalization: Normalization::Backward,
            },
        );
        assert!((backward[0] - 8.0).abs() < 1e-5, "{}", backward[0]);
        assert!((backward[1] - 2.0 * SQRT_2).abs() < 1e-5, "{}", backward[1]);

        let forward = dct(
            &input,
            1,
            2,
            &DctAttrs {
                n_out: None,
                normalization: Normalization::Forward,
            },
        );
        assert!((forward[0] - 4.0).abs() < 1e-5, "{}", forward[0]);
        assert!((forward[1] - SQRT_2).abs() < 1e-5, "{}", forward[1]);
    }

    #[test]
    fn norm_modes_relate_to_ortho_by_analytic_scale() {
        // All three normalizations share the same raw sum C[k]; they differ
        // only by a per-coefficient scale, so (exact algebraic identities):
        //   backward[0]   == ortho[0] · 2·√N
        //   backward[k≥1] == ortho[k] · √(2N)
        //   forward[k]    == backward[k] / N
        // Deterministic non-trivial rows stand in for "random" input; the
        // identities hold for every input, so a fixed varied signal suffices.
        let n = 12usize;
        let rows = 3usize;
        let input: Vec<f32> = (0..rows * n)
            .map(|i| (i as f32 * 0.7).sin() + 0.3 * (i as f32 * 0.13).cos())
            .collect();

        let ortho = dct(&input, rows, n, &DctAttrs::new());
        let backward = dct(
            &input,
            rows,
            n,
            &DctAttrs {
                n_out: None,
                normalization: Normalization::Backward,
            },
        );
        let forward = dct(
            &input,
            rows,
            n,
            &DctAttrs {
                n_out: None,
                normalization: Normalization::Forward,
            },
        );

        let nf = n as f32;
        let two_sqrt_n = 2.0 * nf.sqrt();
        let sqrt_2n = (2.0 * nf).sqrt();
        for r in 0..rows {
            for k in 0..n {
                let idx = r * n + k;
                let expect_bw = if k == 0 {
                    ortho[idx] * two_sqrt_n
                } else {
                    ortho[idx] * sqrt_2n
                };
                assert!(
                    (backward[idx] - expect_bw).abs() <= 1e-4 * (1.0 + expect_bw.abs()),
                    "backward[{idx}]={} expect {expect_bw}",
                    backward[idx]
                );
                let expect_fw = backward[idx] / nf;
                assert!(
                    (forward[idx] - expect_fw).abs() <= 1e-4 * (1.0 + expect_fw.abs()),
                    "forward[{idx}]={} expect {expect_fw}",
                    forward[idx]
                );
            }
        }
    }
}
