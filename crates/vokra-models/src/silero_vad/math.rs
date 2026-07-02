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
