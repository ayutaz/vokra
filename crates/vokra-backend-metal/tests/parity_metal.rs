//! Metal GEMM numerical parity (M2-01-T17/T18): the FP32 GPU GEMM vs the CPU
//! backend's `gemm_f32` kernel (M0-08), the same differential oracle the
//! scalar⇔SIMD harness uses. Ceiling is the NFR-QL-01 FP32 bound `atol = 0.01`
//! (the observed error is far smaller and logged per shape).
//!
//! Runs only where a Metal device is available: [`vokra_metal_probe`] gates the
//! suite, so a non-Apple / Metal-less host skips rather than fails (the same
//! "runner must have the device" policy as the GGUF-gated model parity tests).
//! The macOS Metal CI job (M2-01-T21) runs it for real.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use vokra_backend_cpu::kernels as cpu;
use vokra_backend_metal::{MetalContext, vokra_metal_probe};

/// NFR-QL-01 FP32 parity ceiling.
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32 in roughly [-1, 1) (xorshift64*), matching
/// the CPU backend's bench/differential generator so inputs are reproducible.
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

/// Naive f64-accumulated GEMM — an independent second oracle (so parity is not
/// judged solely against the CPU backend's own FMA reduction order).
fn naive_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] as f64 * b[l * n + j] as f64;
            }
            if let Some(bias) = bias {
                acc += bias[j] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// One (m, k, n) case with optional bias, checked against both oracles.
fn check_case(ctx: &MetalContext, m: usize, k: usize, n: usize, with_bias: bool) -> f32 {
    let a = rand_vec(0x1234 ^ ((m * 131 + k) as u64), m * k);
    let b = rand_vec(0x9E37 ^ ((k * 17 + n) as u64), k * n);
    let bias_vec = with_bias.then(|| rand_vec(0xABCD ^ (n as u64), n));
    let bias = bias_vec.as_deref();

    // GPU result.
    let mut gpu = vec![f32::NAN; m * n];
    ctx.gemm_f32(m, n, k, &a, &b, bias, &mut gpu)
        .expect("metal gemm must succeed");

    // CPU backend oracle (M0-08) + independent naive f64 oracle.
    let mut cpu_out = vec![0.0f32; m * n];
    cpu::gemm_f32(m, n, k, &a, &b, bias, &mut cpu_out).expect("cpu gemm oracle");
    let naive = naive_gemm(m, n, k, &a, &b, bias);

    let d_cpu = max_abs_diff(&gpu, &cpu_out);
    let d_naive = max_abs_diff(&gpu, &naive);
    let worst = d_cpu.max(d_naive);
    eprintln!(
        "GEMM parity  m={m:<4} k={k:<4} n={n:<4} bias={:<5}  max|Δ| vs cpu={d_cpu:.3e}  vs naive={d_naive:.3e}",
        with_bias
    );
    assert!(
        d_cpu <= ATOL,
        "metal vs cpu max|Δ| {d_cpu:.3e} exceeds atol {ATOL} (m={m} k={k} n={n} bias={with_bias})"
    );
    assert!(
        d_naive <= ATOL,
        "metal vs naive max|Δ| {d_naive:.3e} exceeds atol {ATOL} (m={m} k={k} n={n} bias={with_bias})"
    );
    worst
}

#[test]
fn gemm_metal_matches_cpu_and_naive_across_shapes() {
    // Gate: only run with a real Metal device (else skip, do not fail).
    let caps = match vokra_metal_probe() {
        Ok(caps) => caps,
        Err(e) => {
            eprintln!("no Metal device ({e}); skipping Metal GEMM parity");
            return;
        }
    };
    eprintln!("Metal device: {}", caps.summary());

    let ctx = MetalContext::new().expect("build Metal context");

    // Square powers of two, ragged non-multiples of the 16x16 threadgroup, thin
    // vectors (n=1 / m=1), identity-ish tiny cases, and a large-K reduction.
    let shapes = [
        (1usize, 1usize, 1usize),
        (2, 3, 4),
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
        (16, 32, 24),
        (33, 17, 9),
        (1, 128, 64),
        (64, 1, 64),
        (64, 64, 64),
        (37, 100, 41),
        (128, 128, 128),
    ];

    let mut global_worst = 0.0f32;
    for &(m, k, n) in &shapes {
        global_worst = global_worst.max(check_case(&ctx, m, k, n, false));
    }
    // Bias path on a representative subset.
    for &(m, k, n) in &[(2usize, 3usize, 4usize), (16, 16, 16), (37, 100, 41)] {
        global_worst = global_worst.max(check_case(&ctx, m, k, n, true));
    }

    eprintln!("Metal GEMM parity: global max|Δ| = {global_worst:.3e} (atol = {ATOL})");
    assert!(global_worst <= ATOL);
}

/// Shape mismatches / zero dims are explicit `InvalidArgument`, not a GPU fault.
#[test]
fn gemm_rejects_bad_shapes_explicitly() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("no Metal device; skipping shape-validation test");
        return;
    };
    // a should be m*k = 4 long, but is 2.
    let mut out = [0.0f32; 4];
    assert!(
        ctx.gemm_f32(2, 2, 2, &[1.0, 2.0], &[1.0; 4], None, &mut out)
            .is_err()
    );
    // zero dimension.
    assert!(
        ctx.gemm_f32(0, 2, 2, &[], &[1.0; 4], None, &mut [0.0; 0])
            .is_err()
    );
    // bias length != n.
    assert!(
        ctx.gemm_f32(2, 2, 2, &[1.0; 4], &[1.0; 4], Some(&[1.0; 1]), &mut out)
            .is_err()
    );
}

/// HTDemucs' spatial convolutions use grouped, dilated, and transposed
/// channel-major kernels. This is a real device-vs-CPU check; a Metal-less
/// runner skips it rather than claiming parity without a GPU.
#[test]
fn conv2d_and_transpose2d_metal_match_cpu() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("no Metal device; skipping Conv2d parity");
        return;
    };
    let input = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    ];
    let weight = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, -1.0];
    let bias = [0.5, -1.0];
    let mut cpu_conv = [0.0f32; 6];
    let mut gpu_conv = [0.0f32; 6];
    cpu::conv2d_f32(
        &input,
        2,
        2,
        3,
        &weight,
        2,
        2,
        2,
        Some(&bias),
        (1, 1),
        (0, 1),
        (1, 2),
        2,
        &mut cpu_conv,
    )
    .expect("CPU Conv2d reference");
    ctx.conv2d_f32(
        &input,
        2,
        2,
        3,
        &weight,
        2,
        2,
        2,
        Some(&bias),
        (1, 1),
        (0, 1),
        (1, 2),
        2,
        &mut gpu_conv,
    )
    .expect("Metal Conv2d");
    let conv_diff = max_abs_diff(&cpu_conv, &gpu_conv);
    assert!(
        conv_diff <= ATOL,
        "Conv2d max|Δ| {conv_diff:.3e} exceeds {ATOL}"
    );

    let trans_input = [1.0, 2.0, 10.0, 20.0];
    let trans_weight = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
    let trans_bias = [0.5, -1.0];
    // Formula: output channels × height × width = 2 × 3 × 6 = 36.  The
    // spatial extents are (1 - 1) × 2 + 1 × (2 - 1) + 1 + 1 = 3 and
    // (2 - 1) × 2 + 2 × (2 - 1) + 1 + 1 = 6, with zero padding.
    let mut cpu_trans = [0.0f32; 36];
    let mut gpu_trans = [0.0f32; 36];
    cpu::conv_transpose2d_f32(
        &trans_input,
        2,
        1,
        2,
        &trans_weight,
        2,
        2,
        2,
        Some(&trans_bias),
        (2, 2),
        (0, 0),
        (1, 2),
        (1, 1),
        2,
        &mut cpu_trans,
    )
    .expect("CPU ConvTranspose2d reference");
    ctx.conv_transpose2d_f32(
        &trans_input,
        2,
        1,
        2,
        &trans_weight,
        2,
        2,
        2,
        Some(&trans_bias),
        (2, 2),
        (0, 0),
        (1, 2),
        (1, 1),
        2,
        &mut gpu_trans,
    )
    .expect("Metal ConvTranspose2d");
    let trans_diff = max_abs_diff(&cpu_trans, &gpu_trans);
    assert!(
        trans_diff <= ATOL,
        "ConvTranspose2d max|Δ| {trans_diff:.3e} exceeds {ATOL}"
    );
}

/// OUVE predictor and annealed-Langevin corrector are checked against a small
/// independent scalar oracle and the resident API. The resident assertions
/// make a device-to-host round-trip between sampler steps observable.
#[test]
fn ouve_sampler_metal_matches_independent_oracle_and_stays_resident() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("no Metal device; skipping OUVE parity");
        return;
    };
    let theta = 1.5f32;
    let sigma_min = 0.05f32;
    let sigma_max = 0.5f32;
    let t = 0.7f32;
    let step = 0.01f32;
    let snr = 0.2f32;
    let x = [0.2f32, -0.4, 0.8, -1.1];
    let y = [0.1f32, 0.3, -0.2, 0.7];
    let score = [0.4f32, -0.2, 0.1, 0.6];
    let noise = [0.5f32, -0.25, 0.75, -0.125];
    let log_ratio = (sigma_max / sigma_min).ln();
    let diffusion = sigma_min * (log_ratio * t).exp() * (2.0 * log_ratio).sqrt();
    let score_scale = diffusion * diffusion;
    let mut expected_mean = [0.0f32; 4];
    let mut expected = [0.0f32; 4];
    for i in 0..x.len() {
        let drift = theta * (y[i] - x[i]);
        expected_mean[i] = x[i] - (drift - score_scale * score[i]) * step;
        expected[i] = expected_mean[i] + diffusion * step.sqrt() * noise[i];
    }
    let mut actual = [f32::NAN; 4];
    let mut actual_mean = [f32::NAN; 4];
    ctx.ouve_reverse_diffusion_f32(
        theta,
        sigma_min,
        sigma_max,
        &x,
        &y,
        &score,
        t,
        step,
        &noise,
        false,
        &mut actual,
        &mut actual_mean,
    )
    .expect("Metal OUVE predictor");
    assert!(max_abs_diff(&actual, &expected) <= ATOL);
    assert!(max_abs_diff(&actual_mean, &expected_mean) <= ATOL);

    let x_dev = ctx.upload(&x).expect("upload x");
    let y_dev = ctx.upload(&y).expect("upload y");
    let score_dev = ctx.upload(&score).expect("upload score");
    let noise_dev = ctx.upload(&noise).expect("upload noise");
    let mut out_dev = ctx.alloc_dev(x.len()).expect("alloc out");
    let mut mean_dev = ctx.alloc_dev(x.len()).expect("alloc mean");
    let before = ctx.readback_count();
    ctx.ouve_reverse_diffusion_dev(
        &mut out_dev,
        &mut mean_dev,
        &x_dev,
        &y_dev,
        &score_dev,
        &noise_dev,
        theta,
        sigma_min,
        sigma_max,
        t,
        step,
        false,
    )
    .expect("resident Metal OUVE predictor");
    assert_eq!(ctx.readback_count(), before);

    let variance = sigma_min
        * sigma_min
        * (-2.0 * theta * t).exp()
        * ((2.0 * (theta + log_ratio) * t).exp() - 1.0)
        * log_ratio
        / (theta + log_ratio);
    let corrector_step = 2.0 * (snr * variance.sqrt()) * (snr * variance.sqrt());
    let corrector_noise_scale = (2.0 * corrector_step).sqrt();
    let mut expected_corrector = [0.0f32; 4];
    let mut expected_corrector_mean = [0.0f32; 4];
    for i in 0..x.len() {
        expected_corrector_mean[i] = expected[i] + corrector_step * score[i];
        expected_corrector[i] = expected_corrector_mean[i] + corrector_noise_scale * noise[i];
    }
    let mut corrected_dev = ctx.alloc_dev(x.len()).expect("alloc corrected");
    let mut corrected_mean_dev = ctx.alloc_dev(x.len()).expect("alloc corrected mean");
    ctx.ouve_annealed_langevin_dev(
        &mut corrected_dev,
        &mut corrected_mean_dev,
        &out_dev,
        &score_dev,
        &noise_dev,
        theta,
        sigma_min,
        sigma_max,
        t,
        snr,
    )
    .expect("resident Metal OUVE corrector");
    assert_eq!(
        ctx.readback_count(),
        before,
        "sampler steps must not read back device state"
    );
    let mut resident_corrected = [f32::NAN; 4];
    let mut resident_corrected_mean = [f32::NAN; 4];
    ctx.download(&corrected_dev, &mut resident_corrected)
        .expect("download corrected result");
    ctx.download(&corrected_mean_dev, &mut resident_corrected_mean)
        .expect("download corrected mean");
    assert_eq!(ctx.readback_count(), before + 2);
    assert!(max_abs_diff(&resident_corrected, &expected_corrector) <= ATOL);
    assert!(max_abs_diff(&resident_corrected_mean, &expected_corrector_mean) <= ATOL);

    let mut actual_corrector = [f32::NAN; 4];
    let mut actual_corrector_mean = [f32::NAN; 4];
    ctx.ouve_annealed_langevin_f32(
        theta,
        sigma_min,
        sigma_max,
        &expected,
        &score,
        t,
        snr,
        &noise,
        &mut actual_corrector,
        &mut actual_corrector_mean,
    )
    .expect("Metal OUVE corrector");
    assert!(max_abs_diff(&actual_corrector, &expected_corrector) <= ATOL);
    assert!(max_abs_diff(&actual_corrector_mean, &expected_corrector_mean) <= ATOL);
}

#[test]
fn ncsnpp_group_norm_groups_matches_independent_oracle() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("no Metal device; skipping multi-group GroupNorm parity");
        return;
    };
    // NCSN++ source-shaped width: 128 channels split into 32 groups. This
    // exercises a group reduction over multiple channels, not merely the
    // degenerate one-channel case, and the Rust launcher dispatches exactly
    // one Metal thread per group.
    let channels = 128;
    let positions = 3;
    let groups = 32;
    let input: Vec<f32> = (0..channels * positions)
        .map(|index| index as f32 * 0.25 - 3.0)
        .collect();
    let gamma: Vec<f32> = (0..channels)
        .map(|channel| 0.5 + channel as f32 * 0.03125)
        .collect();
    let beta: Vec<f32> = (0..channels)
        .map(|channel| -0.25 + channel as f32 * 0.015625)
        .collect();
    let mut actual = vec![f32::NAN; input.len()];
    ctx.group_norm_groups_f32(
        &input,
        &mut actual,
        channels,
        positions,
        groups,
        &gamma,
        &beta,
        1.0e-6,
    )
    .expect("Metal multi-group GroupNorm");

    // Independent scalar oracle, deliberately separate from either backend.
    let mut expected = vec![0.0; input.len()];
    let channels_per_group = channels / groups;
    for group in 0..groups {
        let first_channel = group * channels_per_group;
        let count = channels_per_group * positions;
        let mut sum = 0.0f32;
        for channel in first_channel..first_channel + channels_per_group {
            let base = channel * positions;
            for position in 0..positions {
                sum += input[base + position];
            }
        }
        let mean = sum / count as f32;
        let mut variance_sum = 0.0f32;
        for channel in first_channel..first_channel + channels_per_group {
            let base = channel * positions;
            for position in 0..positions {
                let delta = input[base + position] - mean;
                variance_sum += delta * delta;
            }
        }
        let variance = variance_sum / count as f32;
        let inv_std = (variance + 1.0e-6).sqrt().recip();
        for channel in first_channel..first_channel + channels_per_group {
            let base = channel * positions;
            for position in 0..positions {
                expected[base + position] =
                    (input[base + position] - mean) * inv_std * gamma[channel] + beta[channel];
            }
        }
    }
    assert!(max_abs_diff(&actual, &expected) <= ATOL);
}
