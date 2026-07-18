//! Small, self-contained numeric helpers for the Silero VAD subgraph.
//!
//! These are deliberately **private** to the subgraph (FR-LD-06 / T04): the
//! pseudo-STFT `Conv1d` and every other op run through this module's own
//! `conv1d`, never through the generic `vokra-ops` `stft` op (NFR-QL-05) or an
//! external crate. Implementations are portable scalar Rust; SIMD tuning is out
//! of scope for M0-05 (that is M0-08). Compute is done in `f32` to match the
//! model's storage and keep the streaming handle cheap.

/// Reflection-pads `x` on the **right** by `n` samples (NumPy `mode="reflect"`:
/// the edge sample is not duplicated).
///
/// Silero's front-end pads a fixed frame (512 @ 16 kHz / 256 @ 8 kHz) by
/// `n_fft/4` on the right only (verified against the ONNX `Pad` node); this is
/// the exact operation the learned pseudo-STFT expects. Requires `n < x.len()`.
pub(super) fn reflect_pad_right(x: &[f32], n: usize) -> Vec<f32> {
    debug_assert!(
        n < x.len(),
        "reflect pad {n} needs len > {n}, got {}",
        x.len()
    );
    let l = x.len();
    let mut out = Vec::with_capacity(l + n);
    out.extend_from_slice(x);
    for j in 0..n {
        out.push(x[l - 2 - j]);
    }
    out
}

/// 1-D convolution matching ONNX / PyTorch `Conv1d` (cross-correlation).
///
/// * `x`: input `[c_in, l]` row-major (channel-major, length fastest);
/// * `weight`: `[c_out, c_in, k]` row-major;
/// * `bias`: optional `[c_out]`;
/// * zero-padded by `pad` on both sides, then strided by `stride`.
///
/// Returns `[c_out, l_out]` with `l_out = (l + 2*pad - k) / stride + 1`.
///
/// A low-level primitive: the many dimension arguments are the shape of the
/// convolution, kept explicit rather than bundled into a struct.
///
/// Since M5-14 Wave-2 this per-channel scalar reduction is the **test-side
/// reference oracle** for the transposed production path ([`conv1d_wt`]).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn conv1d(
    x: &[f32],
    c_in: usize,
    l: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), c_in * l);
    debug_assert_eq!(weight.len(), c_out * c_in * k);
    let lp = l + 2 * pad;
    let l_out = (lp - k) / stride + 1;
    let mut out = vec![0.0f32; c_out * l_out];
    for co in 0..c_out {
        let b = bias.map_or(0.0, |v| v[co]);
        let w_co = &weight[co * c_in * k..(co + 1) * c_in * k];
        for t in 0..l_out {
            // Window start in padded coordinates; map back to real x indices.
            let base = t * stride;
            let mut acc = b;
            for ci in 0..c_in {
                let w_ci = &w_co[ci * k..(ci + 1) * k];
                let x_ci = &x[ci * l..(ci + 1) * l];
                for (kk, &wk) in w_ci.iter().enumerate() {
                    let p = base + kk;
                    // Skip taps that fall in the zero pad region.
                    if p < pad || p >= pad + l {
                        continue;
                    }
                    acc += wk * x_ci[p - pad];
                }
            }
            out[co * l_out + t] = acc;
        }
    }
    out
}

/// [`conv1d`] with a pre-transposed weight (`weight_t = [c_in·k, c_out]`,
/// tap-major, output channel fastest) — the M5-14 Wave-2 (T21) hot path.
///
/// **Bit-identical to [`conv1d`]** by construction: each output element
/// `out[co, t]` is still one accumulator seeded from `bias[co]`, advanced
/// over the taps `(ci, kk)` in the same ascending order with the same
/// unfused `w · x` multiply-add, skipping the same zero-pad taps. Only the
/// loop NESTING changes — taps outer, output channels inner — which turns
/// the inner loop into a contiguous axpy over a `weight_t` row that the
/// compiler auto-vectorizes (`acc[co] += w[co] · xv`, no reduction), instead
/// of the latency-bound scalar reduction. The differential test below pins
/// `==` (not a tolerance) against [`conv1d`], which is what keeps the
/// committed parity fixtures (7.9e-8 anchor + ctx576) byte-stable
/// (NFR-QL-05: 1:1 subgraph semantics untouched).
#[allow(clippy::too_many_arguments)]
pub(super) fn conv1d_wt(
    x: &[f32],
    c_in: usize,
    l: usize,
    weight_t: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), c_in * l);
    debug_assert_eq!(weight_t.len(), c_in * k * c_out);
    let lp = l + 2 * pad;
    let l_out = (lp - k) / stride + 1;
    let mut out = vec![0.0f32; c_out * l_out];
    let mut acc = vec![0.0f32; c_out];
    for t in 0..l_out {
        match bias {
            Some(b) => acc.copy_from_slice(b),
            None => acc.fill(0.0),
        }
        let base = t * stride;
        for ci in 0..c_in {
            let x_ci = &x[ci * l..(ci + 1) * l];
            for kk in 0..k {
                let p = base + kk;
                // Skip taps that fall in the zero pad region (same guard,
                // same surviving-tap order as `conv1d`).
                if p < pad || p >= pad + l {
                    continue;
                }
                let xv = x_ci[p - pad];
                let wrow = &weight_t[(ci * k + kk) * c_out..(ci * k + kk + 1) * c_out];
                for (a, &wv) in acc.iter_mut().zip(wrow) {
                    *a += wv * xv;
                }
            }
        }
        for (co, &a) in acc.iter().enumerate() {
            out[co * l_out + t] = a;
        }
    }
    out
}

/// Matrix-vector product `y = w @ x` with `w` stored `[m, n]` row-major.
pub(super) fn matvec(w: &[f32], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(x.len(), n);
    let mut y = vec![0.0f32; m];
    for i in 0..m {
        let row = &w[i * n..(i + 1) * n];
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Logistic sigmoid `1 / (1 + e^-x)` (`f32`; saturates cleanly at the extremes).
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU in place.
pub(super) fn relu_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_right_matches_numpy() {
        // np.pad([1,2,3,4], (0,2), 'reflect') == [1,2,3,4,3,2]
        assert_eq!(
            reflect_pad_right(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn conv1d_identity_kernel() {
        // c_in=1, k=1 identity weight, stride 1, no pad -> passthrough * w.
        let x = [1.0, 2.0, 3.0];
        let w = [2.0]; // [1,1,1]
        let y = conv1d(&x, 1, 3, &w, None, 1, 1, 1, 0);
        assert_eq!(y, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn conv1d_pad_and_stride() {
        // c_in=1,c_out=1,k=3,pad=1,stride=2 over [1,2,3,4]:
        // padded [0,1,2,3,4,0]; windows @0,@2 -> taps (0,1,2),(2,3,4).
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0];
        let y = conv1d(&x, 1, 4, &w, Some(&[10.0]), 1, 3, 2, 1);
        // l_out = (4+2-3)/2+1 = 2; sums = 0+1+2+10=13, 2+3+4+10=19
        assert_eq!(y, vec![13.0, 19.0]);
    }

    #[test]
    fn conv1d_accumulates_over_input_channels() {
        // c_in=2, c_out=1, k=1, stride 1, pad 0. Channel-major input:
        // ch0 = [1,2], ch1 = [3,4]; weight [c_out,c_in,k] = [10, 1].
        // y = 10*ch0 + 1*ch1 = [10*1+1*3, 10*2+1*4] = [13, 24].
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [10.0, 1.0];
        let y = conv1d(&x, 2, 2, &w, None, 1, 1, 1, 0);
        assert_eq!(y, vec![13.0, 24.0]);
    }

    #[test]
    fn conv1d_strides_output_channels() {
        // Same input; c_out=2, weight [c_out,c_in,k] = [[10,1],[1,0]].
        // co0 = 10*ch0 + 1*ch1 = [13, 24]; co1 = 1*ch0 + 0*ch1 = [1, 2].
        // Output is [c_out, l_out] row-major = [13,24, 1,2].
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [10.0, 1.0, 1.0, 0.0];
        let y = conv1d(&x, 2, 2, &w, None, 2, 1, 1, 0);
        assert_eq!(y, vec![13.0, 24.0, 1.0, 2.0]);
    }

    /// M5-14 Wave-2 bit-identity pin: `conv1d_wt` (taps-outer axpy over the
    /// pre-transposed weight) must equal `conv1d` **exactly** — same
    /// per-element accumulation chain, only the loop nesting differs.
    #[test]
    fn conv1d_wt_bitwise_matches_conv1d() {
        // Deterministic pseudo-random values (xorshift, no external RNG).
        fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
            let mut x = seed | 1;
            (0..n)
                .map(|_| {
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
                    bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
                })
                .collect()
        }
        // (c_in, l, c_out, k, stride, pad) — the real subgraph shapes at
        // reduced sizes (pseudo-STFT k=8 s=4 p=0; encoder k=3 p=1 s∈{1,2})
        // plus ragged corners.
        let cases = [
            (1, 40, 10, 8, 4, 0),
            (5, 4, 8, 3, 1, 1),
            (8, 4, 6, 3, 2, 1),
            (6, 2, 4, 3, 2, 1),
            (6, 1, 4, 3, 1, 1),
            (3, 7, 5, 1, 1, 0),
        ];
        for (i, &(c_in, l, c_out, k, stride, pad)) in cases.iter().enumerate() {
            let x = rand_vec(31 + i as u64, c_in * l);
            let w = rand_vec(131 + i as u64, c_out * c_in * k);
            let bias = rand_vec(231 + i as u64, c_out);
            // Transpose [c_out, c_in*k] -> [c_in*k, c_out].
            let taps = c_in * k;
            let mut w_t = vec![0.0f32; taps * c_out];
            for co in 0..c_out {
                for tap in 0..taps {
                    w_t[tap * c_out + co] = w[co * taps + tap];
                }
            }
            for b in [None, Some(bias.as_slice())] {
                let want = conv1d(&x, c_in, l, &w, b, c_out, k, stride, pad);
                let got = conv1d_wt(&x, c_in, l, &w_t, b, c_out, k, stride, pad);
                assert_eq!(got, want, "case {i} bias={}", b.is_some());
            }
        }
    }

    #[test]
    fn matvec_basic() {
        // [[1,2],[3,4]] @ [1,1] = [3,7]
        let y = matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0, 1.0]);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn sigmoid_midpoint_and_saturation() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(40.0) > 0.999);
        assert!(sigmoid(-40.0) < 0.001);
    }
}
