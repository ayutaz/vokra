//! Synthetic end-to-end parity for the public CosyVoice2 [`HiFTChain`]
//! Metal adapter.
//!
//! This is deliberately a small, synthetic-model test.  It does not claim
//! real-checkpoint support or model quality.  Its purpose is to exercise the
//! complete resident graph through [`HiFTChain::forward_with_backend`], using
//! the same non-zero topology bundle as the HiFTNet module/ops tests, and to
//! compare its one final host result with the scalar CPU oracle.
//!
//! The Metal path owns its readback counter internally.  A successful call
//! therefore also proves the adapter's public contract that exactly one final
//! download was selected; the invalid odd-FFT call below proves the public
//! failure contract without exposing backend internals (a non-zero readback
//! would be wrapped as `BackendUnavailable` by the chain).

use vokra_core::{BackendKind, VokraError};
use vokra_models::cosyvoice2::{HiFTChain, HiFTChainConfig, HiFTChainWeights};
use vokra_ops::hiftnet::{F0PredictorWeights, ResBlockWeights};

const IN_CHANNELS: usize = 4;
const T_MEL: usize = 2;

// Registered from the existing Metal primitive contracts: Snake/SineGen use
// 5e-4 for transcendental differences, while the resident linear/conv/STFT
// paths are tighter.  Keep this bound fixed until a failing run is diagnosed;
// it must not be widened to make a parity run green.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
const METAL_ATOL: f32 = 5e-4;

/// The small HiFTNet shape used by `vokra-ops`'s resident graph tests and the
/// `HiFTChain` module tests.  Every graph weight is non-zero but deliberately
/// tiny, so all branches are exercised without making this a real model claim.
fn small_nonzero_bundle() -> (HiFTChainConfig, HiFTChainWeights) {
    let cfg = HiFTChainConfig {
        in_channels: IN_CHANNELS as u32,
        base_channels: 8,
        nb_harmonics: 2,
        sampling_rate: 16_000,
        nsf_alpha: 0.1,
        nsf_sigma: 0.003,
        nsf_voiced_threshold: 10.0,
        upsample_rates: vec![2, 2],
        upsample_kernel_sizes: vec![4, 4],
        istft_n_fft: 8,
        istft_hop_len: 2,
        resblock_kernel_sizes: vec![3],
        resblock_dilation_sizes: vec![vec![1]],
        source_resblock_kernel_sizes: vec![3, 3],
        source_resblock_dilation_sizes: vec![vec![1], vec![1]],
        lrelu_slope: 0.1,
        audio_limit: 0.99,
    };

    let mut f0_conv_weights = vec![vec![0.001; 8 * IN_CHANNELS * 3]];
    for _ in 1..5 {
        f0_conv_weights.push(vec![0.001; 8 * 8 * 3]);
    }
    let f0_weights = F0PredictorWeights {
        conv_weights: f0_conv_weights,
        conv_biases: vec![vec![0.001; 8]; 5],
        linear_w: vec![0.001; 8],
        linear_b: vec![0.1],
    };

    let ups_w = vec![vec![0.002; 8 * 4 * 4], vec![0.002; 4 * 2 * 4]];
    let ups_b = vec![vec![0.001; 4], vec![0.001; 2]];

    // n_fft + 2 = 10; downsample_us for [2, 2] is [2, 1].
    let source_downs_w = vec![vec![0.002; 4 * 10 * 4], vec![0.002; 2 * 10]];
    let source_downs_b = vec![vec![0.001; 4], vec![0.001; 2]];

    let make_res = |channels: usize| ResBlockWeights {
        convs1_w: vec![vec![0.001; channels * channels * 3]],
        convs1_b: vec![vec![0.001; channels]],
        convs2_w: vec![vec![0.001; channels * channels * 3]],
        convs2_b: vec![vec![0.001; channels]],
        activations1_alpha: vec![vec![0.2; channels]],
        activations2_alpha: vec![vec![0.2; channels]],
    };
    let source_resblock_weights = vec![make_res(4), make_res(2)];
    let resblock_weights = vec![make_res(4), make_res(2)];

    let weights = HiFTChainWeights {
        conv_pre_w: vec![0.002; 8 * IN_CHANNELS * 7],
        conv_pre_b: vec![0.001; 8],
        ups_w,
        ups_b,
        source_downs_w,
        source_downs_b,
        source_resblock_weights,
        resblock_weights,
        conv_post_w: vec![0.002; 10 * 2 * 7],
        conv_post_b: vec![0.001; 10],
        m_source_linear_w: vec![0.1; 3],
        m_source_linear_b: 0.001,
        f0_predictor_weights: f0_weights,
    };
    (cfg, weights)
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
#[test]
fn metal_backend_is_explicitly_unavailable_without_apple_metal() {
    let (cfg, weights) = small_nonzero_bundle();
    let chain = HiFTChain::new(cfg, weights).expect("synthetic HiFT bundle builds");
    let err = chain
        .forward_with_backend(&[0.1; IN_CHANNELS * T_MEL], T_MEL, BackendKind::Metal)
        .expect_err("non-Apple/no-feature Metal must not fall back to CPU");
    assert!(matches!(err, VokraError::BackendUnavailable(_)), "{err:?}");
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal {
    use super::*;

    #[test]
    fn hift_chain_metal_matches_cpu_with_one_final_readback_or_clean_skip() {
        let (cfg, weights) = small_nonzero_bundle();
        let cpu = HiFTChain::new(cfg.clone(), weights.clone()).expect("CPU chain builds");
        let metal = HiFTChain::new(cfg.clone(), weights.clone()).expect("Metal chain builds");
        let mel: Vec<f32> = (0..IN_CHANNELS * T_MEL)
            .map(|i| 0.1 + i as f32 * 0.01)
            .collect();

        let cpu_pcm = cpu
            .forward_with_backend(&mel, T_MEL, BackendKind::Cpu)
            .expect("CPU scalar oracle must succeed");
        let metal_pcm = match metal.forward_with_backend(&mel, T_MEL, BackendKind::Metal) {
            Ok(pcm) => pcm,
            Err(VokraError::BackendUnavailable(message))
                if message == "no system default Metal device" =>
            {
                println!("skip: Metal device unavailable ({message})");
                return;
            }
            Err(error) => panic!("Metal HiFT resident graph failed: {error}"),
        };

        assert_eq!(metal_pcm.len(), cpu_pcm.len());
        assert!(
            metal_pcm.iter().any(|sample| sample.abs() > 1e-7),
            "non-zero synthetic graph must not collapse to an all-zero waveform"
        );
        let delta = max_delta(&metal_pcm, &cpu_pcm);
        assert!(
            delta <= METAL_ATOL,
            "HiFTChain Metal vs CPU max |Δ| = {delta:e} > {METAL_ATOL:e}"
        );

        // Perturb the terminal pre-iSTFT STFT components via `conv_post_b`,
        // rather than the mel input.  The tiny synthetic upstream weights
        // intentionally make the mel-to-waveform sensitivity very small, so a
        // changed mel is not a reliable discriminator here.  A material
        // final-bias perturbation must still move the scalar oracle beyond the
        // fixed parity bound; otherwise a close result could be vacuous
        // zero-output agreement.  Keep this control on a separate chain so
        // the CPU/Metal comparison above uses the exact same bundle.
        let mut control_weights = weights.clone();
        for bias in &mut control_weights.conv_post_b {
            *bias += 0.5;
        }
        let control = HiFTChain::new(cfg, control_weights).expect("control chain builds");
        let control_pcm = control
            .forward_with_backend(&mel, T_MEL, BackendKind::Cpu)
            .expect("CPU negative control must succeed");
        let control_delta = max_delta(&cpu_pcm, &control_pcm);
        assert!(
            control_delta > METAL_ATOL,
            "negative control moved CPU output only {control_delta:e}, not beyond {METAL_ATOL:e}"
        );
    }

    #[test]
    fn hift_chain_invalid_resident_graph_fails_before_any_readback() {
        let (mut cfg, mut weights) = small_nonzero_bundle();
        // The public chain accepts this shape, while the resident graph
        // explicitly rejects odd FFT sizes because its [Re F; Im F] contract
        // differs from the scalar odd-FFT layout.  Adjust the dependent
        // tensors so construction reaches the resident-forward guard.
        cfg.istft_n_fft = 7;
        weights.source_downs_w = vec![vec![0.002; 4 * 9 * 4], vec![0.002; 2 * 9]];
        weights.conv_post_w = vec![0.002; 9 * 2 * 7];
        weights.conv_post_b = vec![0.001; 9];
        let chain = HiFTChain::new(cfg, weights).expect("odd-FFT bundle builds for guard test");
        let error = match chain.forward_with_backend(
            &[0.1; IN_CHANNELS * T_MEL],
            T_MEL,
            BackendKind::Metal,
        ) {
            Ok(_) => panic!("resident graph must reject odd FFT"),
            Err(VokraError::BackendUnavailable(message))
                if message == "no system default Metal device" =>
            {
                println!("skip: Metal device unavailable ({message})");
                return;
            }
            Err(error) => error,
        };
        assert!(
            matches!(error, VokraError::InvalidArgument(ref message) if message.contains("even")),
            "unexpected invalid-graph result: {error:?}"
        );
    }
}
