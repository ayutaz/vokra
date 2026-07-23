//! CosyVoice2 → HiFTNet vocoder chain (SoTA plan §1(a) 訂正, Phase 1-3).
//!
//! # Correct upstream pipeline (arXiv:2412.10117 + cosyvoice/hifigan/generator.py)
//!
//! ```text
//! FSQ tokens → Qwen2.5-0.5B AR decoder → chunk-aware CFM → mel → HiFTNet → PCM
//! ```
//!
//! `HiFTGenerator` (upstream `cosyvoice/hifigan/generator.py:378`,
//! `class HiFTGenerator` — docstring: `"HiFTNet Generator: Neural Source
//! Filter + ISTFTNet"`) is the terminal vocoder that consumes the mel
//! spectrogram produced by the chunk-aware CFM and emits 24 kHz PCM. Vokra's
//! port lives in [`vokra_ops::hiftnet`] (Waves 3c-2/3c-3 + Wave 4 harness).
//!
//! # Why this seam (and not the [`super::mimi_bridge`] module)
//!
//! The 2026-07-22 SoTA-plan §1(a) 訂正 identified that the previous
//! CosyVoice2 → Mimi wiring was built on a wrong premise: **CosyVoice2 does
//! not use the Mimi codec** — that is exclusive to Moshi and CSM. The
//! upstream `cosyvoice/hifigan/generator.py` file makes no reference to
//! Mimi at any point; instead it composes `SourceModuleHnNSF` (NSF) with a
//! `torch.istft` post-conv (ISTFTNet) and a `Snake` activation stack (see
//! `:320 SourceModuleHnNSF`, `:503 torch.istft`, `:102 Snake`). The
//! [`super::mimi_bridge`] module is therefore [`#[deprecated]`][super::mimi_bridge]
//! and kept only to avoid breaking existing test imports and the
//! [`super::chunk_pipeline`] scaffold; new callers use [`HiFTChain`].
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! [`HiFTChain`] holds an owned [`vokra_ops::hiftnet::HiFTGenerator`] — the
//! op crate is a first-party `vokra-*` workspace member, so the root
//! `Cargo.lock` stays `vokra-*` only. No external crate is added by this
//! module.
//!
//! # Fail-loud contract (FR-EX-08)
//!
//! Every shape mismatch is caught inside [`vokra_ops::hiftnet::HiFTGenerator::new`]
//! (config-side: `upsample_kernel_sizes.len() != upsample_rates.len()`,
//! `resblock_dilation_sizes.len() != resblock_kernel_sizes.len()`, F0
//! predictor / ResBlock branch counts, conv layout mismatches). This module
//! propagates those errors verbatim rather than swallowing or re-wrapping
//! them, so a mis-supplied weight bundle surfaces the exact upstream failure.

use vokra_core::Result;
use vokra_ops::hiftnet::{HiFTGenerator, HiFTGeneratorConfig, HiFTGeneratorWeights};

/// Configuration for the HiFTNet vocoder chain.
///
/// A newtype re-alias of [`vokra_ops::hiftnet::HiFTGeneratorConfig`] so the
/// public surface stays namespaced under the CosyVoice2 module (matching the
/// `text_encoder::CosyVoice2Tokenizer` / `llm::LlmBackboneConfig` pattern in
/// this crate) without introducing a shape-drift wrapper that would have to
/// be kept in sync with the op-crate config.
pub type HiFTChainConfig = HiFTGeneratorConfig;

/// Learned parameters for the HiFTNet vocoder chain.
///
/// A newtype re-alias of [`vokra_ops::hiftnet::HiFTGeneratorWeights`] for the
/// same reason as [`HiFTChainConfig`]. Every tensor layout (conv_pre,
/// per-stage ConvTranspose1d ups, per-stage source_downs, per-stage
/// source_resblocks, row-major `[num_upsamples * num_kernels]` resblocks,
/// conv_post, m_source_linear, F0 predictor) is documented on the op-crate
/// type.
pub type HiFTChainWeights = HiFTGeneratorWeights;

/// CosyVoice2 → HiFTNet vocoder chain — the mel → PCM seam.
///
/// Owns a single [`HiFTGenerator`] produced from a caller-supplied
/// [`HiFTChainConfig`] + [`HiFTChainWeights`] bundle. The forward path
/// delegates to [`HiFTGenerator::forward`] verbatim; this wrapper exists so
/// [`super::CosyVoice2Tts`] can carry an `Option<HiFTChain>` field with a
/// stable name across the T24 codec-migration work — the top-level engine
/// does not have to know which vocoder is bound, and a future caller who
/// wires the CFM head to this chain gets a single `hift_chain` seam.
///
/// # Construction
///
/// [`HiFTChain::new`] validates the config/weights bundle and builds an
/// internal [`HiFTGenerator`]. Every shape check surfaces the op-crate's
/// error verbatim (see the module docstring's fail-loud contract note).
///
/// # Forward
///
/// [`HiFTChain::forward`] takes a mel spectrogram `[in_channels, t_mel]`
/// row-major and returns a `Vec<f32>` PCM waveform. The sample rate of that
/// waveform is [`HiFTChainConfig::sampling_rate`] and the length is exactly
/// `t_mel * total_upsample_factor()` (upstream contract, spelled out in the
/// op-crate rustdoc for [`HiFTGenerator::forward`]).
#[derive(Debug, Clone)]
pub struct HiFTChain {
    generator: HiFTGenerator,
}

impl HiFTChain {
    /// Builds a [`HiFTChain`] from its config + weights bundle.
    ///
    /// # Errors
    ///
    /// Propagates every [`HiFTGenerator::new`] validation error verbatim.
    /// The op-crate rustdoc enumerates them; the common ones are:
    ///
    /// - [`VokraError::InvalidArgument`] on empty `upsample_rates`, mismatched
    ///   `upsample_kernel_sizes` / `resblock_dilation_sizes` lengths, or a
    ///   conv weight vector whose length does not match the expected
    ///   `[out_ch, in_ch, kernel]` layout.
    ///
    /// # Zero-argument sanity check
    ///
    /// A caller who accidentally builds a [`HiFTChainWeights`] whose
    /// `ups_w` length disagrees with `upsample_rates` gets a loud error at
    /// construction time — never a mid-forward panic (FR-EX-08).
    pub fn new(cfg: HiFTChainConfig, weights: HiFTChainWeights) -> Result<Self> {
        let generator = HiFTGenerator::new(cfg, weights)?;
        Ok(Self { generator })
    }

    /// Immutable access to the generator config the chain was built with —
    /// convenient for the caller to read the sample rate off the same
    /// source the vocoder used, without holding a duplicate copy.
    #[must_use]
    pub fn config(&self) -> &HiFTChainConfig {
        self.generator.config()
    }

    /// Output PCM sample rate in Hz — mirror of `config().sampling_rate`,
    /// kept as a first-class accessor so a caller wiring the result into a
    /// [`vokra_core::SynthesizedAudio`] does not have to walk into the
    /// config.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.generator.config().sampling_rate
    }

    /// Runs the HiFTNet vocoder forward on a mel spectrogram.
    ///
    /// `mel` is row-major `[in_channels, t_mel]`. Returns the reconstructed
    /// PCM as a `Vec<f32>` of length `t_mel * total_upsample_factor()`.
    ///
    /// # Errors
    ///
    /// Propagates [`HiFTGenerator::forward`] errors verbatim:
    ///
    /// - [`VokraError::InvalidArgument`] on `t_mel == 0` or
    ///   `mel.len() != in_channels * t_mel`.
    /// - Any downstream op error surfaces with the op-crate's original
    ///   message.
    pub fn forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        self.generator.forward(mel, t_mel)
    }
}

// -----------------------------------------------------------------------------
// SAFETY / posture notes for consumers
// -----------------------------------------------------------------------------
// This module contains ZERO `unsafe`. The generator forward is pure-safe Rust
// living in `vokra_ops::hiftnet`. `HiFTChain` is `Debug + Clone` because
// `HiFTGenerator` is (all owned f32 weight vectors, no interior mutability).

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::VokraError;
    use vokra_ops::hiftnet::{F0PredictorWeights, ResBlockWeights};

    /// Wave-4 harness pattern: a small-shape config so the synthesized-weight
    /// build path is exercised without a real HiFTNet checkpoint. The
    /// numbers here mirror `small_hift_config()` in the op-crate parity
    /// harness; keeping them in sync with a Wave 3c-3 helper is left to the
    /// respective owner (Wave 4 tests already pin the shapes, and any change
    /// there would surface as a build error here).
    fn small_hift_chain_bundle() -> (HiFTChainConfig, HiFTChainWeights) {
        let cfg = HiFTChainConfig {
            in_channels: 4,
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

        // F0Predictor (cond_channels = base_channels = 8, num_layers = 5).
        let mut f0_conv_weights: Vec<Vec<f32>> = vec![vec![0.0; 8 * 4 * 3]];
        for _ in 1..5 {
            f0_conv_weights.push(vec![0.0; 8 * 8 * 3]);
        }
        let f0_weights = F0PredictorWeights {
            conv_weights: f0_conv_weights,
            conv_biases: vec![vec![0.0; 8]; 5],
            linear_w: vec![0.0; 8],
            linear_b: vec![0.0; 1],
        };

        // ups: stage 0 [in=8, out=4, k=4], stage 1 [in=4, out=2, k=4]
        let ups_w = vec![vec![0.0; 8 * 4 * 4], vec![0.0; 4 * 2 * 4]];
        let ups_b = vec![vec![0.0; 4], vec![0.0; 2]];

        // source_downs: n_fft + 2 = 10; upstream downsample_us for
        // upsample_rates=[2, 2] resolves to [2, 1]. Stage 0: k=4/stride=2/pad=1;
        // stage 1: k=1/stride=1/pad=0 → 2*10*1 = 20.
        let n_fft_plus_2 = 10;
        let source_downs_w = vec![
            vec![0.0; 4 * n_fft_plus_2 * 4], // stage 0
            vec![0.0; 2 * n_fft_plus_2],     // stage 1
        ];
        let source_downs_b = vec![vec![0.0; 4], vec![0.0; 2]];

        let make_res_zero = |ch: usize, k: usize, n_branches: usize| ResBlockWeights {
            convs1_w: vec![vec![0.0; ch * ch * k]; n_branches],
            convs1_b: vec![vec![0.0; ch]; n_branches],
            convs2_w: vec![vec![0.0; ch * ch * k]; n_branches],
            convs2_b: vec![vec![0.0; ch]; n_branches],
            activations1_alpha: vec![vec![0.0; ch]; n_branches],
            activations2_alpha: vec![vec![0.0; ch]; n_branches],
        };

        let source_resblock_weights = vec![
            make_res_zero(4, 3, 1), // stage 0
            make_res_zero(2, 3, 1), // stage 1
        ];
        // resblocks: row-major [num_ups * num_kernels], num_kernels = 1.
        let resblock_weights = vec![make_res_zero(4, 3, 1), make_res_zero(2, 3, 1)];

        let weights = HiFTChainWeights {
            conv_pre_w: vec![0.0; 8 * 4 * 7],
            conv_pre_b: vec![0.0; 8],
            ups_w,
            ups_b,
            source_downs_w,
            source_downs_b,
            source_resblock_weights,
            resblock_weights,
            conv_post_w: vec![0.0; n_fft_plus_2 * 2 * 7],
            conv_post_b: vec![0.0; n_fft_plus_2],
            m_source_linear_w: vec![0.0; 3], // nb_harmonics + 1
            m_source_linear_b: 0.0,
            f0_predictor_weights: f0_weights,
        };

        (cfg, weights)
    }

    /// The chain builds from a valid synthesized-weight bundle and its
    /// config/sample-rate accessors surface the same values the caller
    /// supplied.
    #[test]
    fn hift_chain_new_accepts_small_synthesized_bundle() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg, weights).expect("small bundle must build");
        assert_eq!(chain.config().in_channels, 4);
        assert_eq!(chain.config().base_channels, 8);
        assert_eq!(chain.sample_rate(), 16_000);
    }

    /// The forward pass produces the exact upstream length contract:
    /// `t_mel * total_upsample_factor()` PCM samples. This is the shape
    /// invariant a caller needs when packing the result into a
    /// `SynthesizedAudio` — a silent shift here would be undetectable
    /// downstream.
    #[test]
    fn hift_chain_forward_output_length_matches_upstream_contract() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        for &t_mel in &[1usize, 2, 3, 5] {
            let mel = vec![0.0f32; cfg.in_channels as usize * t_mel];
            let audio = chain.forward(&mel, t_mel).expect("forward must succeed");
            assert_eq!(
                audio.len(),
                t_mel * cfg.total_upsample_factor() as usize,
                "t_mel = {t_mel}"
            );
        }
    }

    /// A mis-shaped mel must surface as an explicit InvalidArgument — not
    /// a silent shorter output or a panic. This mirrors the op-crate's
    /// `hift_generator_forward_rejects_wrong_mel_shape` pin at the
    /// integration boundary.
    #[test]
    fn hift_chain_forward_rejects_wrong_mel_shape() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        let bogus = vec![0.0f32; cfg.in_channels as usize * 4 - 1];
        let err = chain
            .forward(&bogus, 4)
            .expect_err("wrong-length must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    /// `t_mel = 0` is rejected up front (zero-frame synthesis is not a
    /// valid HiFTNet input — the op-crate's own contract). The chain must
    /// propagate that verbatim, not swallow it.
    #[test]
    fn hift_chain_forward_rejects_zero_t_mel() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg, weights).expect("build");
        let err = chain.forward(&[], 0).expect_err("t_mel=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    /// Same input → same output, twice. The chain does not introduce
    /// hidden RNG; upstream's `NsfEntropy::Deterministic` posture holds
    /// through the wrapper.
    #[test]
    fn hift_chain_forward_is_deterministic_on_same_input() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        let t_mel = 4;
        let mel: Vec<f32> = (0..(cfg.in_channels as usize * t_mel))
            .map(|i| ((i % 7) as f32) * 0.03 - 0.05)
            .collect();
        let a = chain.forward(&mel, t_mel).expect("forward 1");
        let b = chain.forward(&mel, t_mel).expect("forward 2");
        assert_eq!(a, b, "wrapper must not introduce hidden state");
    }

    /// A weight bundle whose `ups_w` length disagrees with `upsample_rates`
    /// must be caught at `new` time. This is the fail-loud construction
    /// contract; the op-crate's own new() surfaces the error, and the
    /// wrapper propagates it verbatim.
    #[test]
    fn hift_chain_new_rejects_ups_weight_count_mismatch() {
        let (cfg, mut weights) = small_hift_chain_bundle();
        // Drop one ups weight — the count no longer matches
        // upsample_rates.len() == 2.
        weights.ups_w.pop();
        weights.ups_b.pop();
        let err = HiFTChain::new(cfg, weights).expect_err("mismatched ups must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }
}
