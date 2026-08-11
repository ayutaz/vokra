//! **plain VITS** with HiFi-GAN decoder — ESPnet-family Japanese VITS
//! (SoTA plan Phase 5 JA-TTS-2, 2026-07-24).
//!
//! # What "plain VITS" is (primary source)
//!
//! `espnet/gan_tts/vits/vits.py` + `espnet/gan_tts/vits/generator.py`
//! (Apache 2.0; fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」)
//! is Kim et al. 2021 VITS (arXiv:2106.06103, "Conditional Variational
//! Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"):
//! a text encoder + (stochastic) duration predictor + normalizing flow +
//! **plain HiFi-GAN generator** (Kong et al. 2020, arXiv:2010.05646).
//!
//! COEIROINK / ESPnet-JA VITS deployments (JSUT-based single-speaker,
//! JVS-based multi-speaker) use this architecture verbatim; the
//! architectural axes below are the **shared ESPnet default** transcribed
//! from `AVAILABLE_GENERATERS["vits_generator"] = VITSGenerator` +
//! `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
//! `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml`. What corpus a given
//! checkpoint was trained on (JSUT / JVS / COEIROINK's proprietary corpus
//! / ITAKO / a permissive re-training) does not reshape the architecture
//! — it only decides who may redistribute the weight.
//!
//! # ⚠️  Weight redistribution is **NOT** granted by this module
//!
//! Per the SoTA plan (`docs/tickets/sota-coverage-plan-2026-07-22.md`
//! §2.4, `support the architecture, refuse the weights`), the publicly
//! distributed ESPnet / COEIROINK JA VITS checkpoints ride on
//! **corpus terms that forbid re-distribution of the trained weight**:
//!
//! - **JSUT corpus** (`sites.google.com/site/shinnosuketakamichi/publication/jsut`)
//!   pins single-speaker Japanese TTS training data; the terms state
//!   *"Re-distribution is not permitted"*.
//! - **JVS corpus** (`sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`)
//!   ships the 100-speaker multi-speaker Japanese corpus; its terms carry
//!   the same re-distribution ban.
//! - **COEIROINK proprietary corpus** carries per-character licence terms
//!   that a converter cannot machine-check.
//!
//! The converter (`vokra-convert::models::vits_ja`) therefore stamps
//! GGUFs produced from a stock ESPnet-JSUT / ESPnet-JVS / COEIROINK
//! checkpoint as [`vokra_core::LicenseClass::RedistributionForbidden`]
//! **by default**. A user who trained their own VITS on a permissive
//! corpus overrides the stamp with `vokra-convert --license <spdx>`.
//! Architecture rides code (Apache 2.0 ESPnet, MIT jaywalnut310/vits)
//! and is *always* independently implementable (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Primary source (every axis is transcribed verbatim)
//!
//! Every field in [`VitsJaConfig`] is transcribed **verbatim** from
//! upstream ESPnet primary sources (fetched 2026-07-24 — CLAUDE.md
//! 「ハルシネーション厳禁」):
//!
//! - **Text encoder** — `text_encoder_blocks = 6`,
//!   `text_encoder_attention_heads = 2`, `text_encoder_ffn_expand = 4`,
//!   `text_encoder_positionwise_conv_kernel_size = 3` (`train_vits.yaml`
//!   pins 3; the `vits.py` default is 1 — the JA recipes override to 3),
//!   `text_encoder_positional_encoding_layer_type = "rel_pos"`,
//!   `text_encoder_self_attention_layer_type = "rel_selfattn"`,
//!   `text_encoder_activation_type = "swish"`,
//!   `text_encoder_normalize_before = true`,
//!   `text_encoder_dropout_rate = 0.1`,
//!   `text_encoder_positional_dropout_rate = 0.0`,
//!   `text_encoder_attention_dropout_rate = 0.1`,
//!   `use_macaron_style_in_text_encoder = true`,
//!   `use_conformer_conv_in_text_encoder = false` (per the JA recipe's
//!   `NOTE(kan-bayashi)` about BatchNorm1d + multi-GPU pytorch 1.7.1),
//!   `text_encoder_conformer_kernel_size = -1` (conformer conv disabled).
//! - **HiFi-GAN decoder** — `decoder_kernel_size = 7` (initial and final
//!   `conv1d`), `decoder_channels = 512` (`initial_channel`),
//!   `decoder_upsample_scales = [8, 8, 2, 2]`,
//!   `decoder_upsample_kernel_sizes = [16, 16, 4, 4]`,
//!   `decoder_resblock_kernel_sizes = [3, 7, 11]`,
//!   `decoder_resblock_dilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]`,
//!   `use_weight_norm_in_decoder = true`. **Distinct from piper-plus**:
//!   piper-plus (MB-iSTFT-VITS2) decodes through a sub-band iSTFT + PQMF
//!   post-net (`RESBLOCK_KERNELS = [3, 5, 7]`, dilations `[[1,2],[2,6],[3,12]]`,
//!   MRF branches per stage, `dec_up_stride = 4`); this module uses the
//!   plain HiFi-GAN generator directly (no MB-iSTFT, no PQMF).
//! - **Residual affine coupling flow** — `flow_flows = 4`,
//!   `flow_kernel_size = 5`, `flow_base_dilation = 1`,
//!   `flow_layers = 4`, `flow_dropout_rate = 0.0`,
//!   `use_only_mean_in_flow = true`.
//! - **Stochastic duration predictor** —
//!   `stochastic_duration_predictor_kernel_size = 3`,
//!   `stochastic_duration_predictor_dropout_rate = 0.5`,
//!   `stochastic_duration_predictor_flows = 4`,
//!   `stochastic_duration_predictor_dds_conv_layers = 3`.
//! - **Global axes** — `hidden_channels = 192`, `segment_size = 32`,
//!   `aux_channels = 513` (posterior-encoder input width, `n_fft/2 + 1`
//!   for the JA 22.05 kHz recipe's `n_fft = 1024`), `sampling_rate = 22050`
//!   (JSUT / JVS default; the full-band `train_full_band_vits.yaml`
//!   variant reshapes both, see [`VitsJaConfig::espnet_ja_full_band_44khz`]
//!   note).
//!
//! # Distinct from piper-plus (MB-iSTFT-VITS2)
//!
//! piper-plus (Vokra's first native TTS) is the **MB-iSTFT-VITS2** family
//! (Kaneko et al. 2023 arXiv:2210.15975), which shares text encoder /
//! duration predictor / normalising flow with plain VITS but replaces the
//! HiFi-GAN decoder with a sub-band iSTFT + PQMF post-net. plain VITS is
//! the earlier design that decodes through the HiFi-GAN generator
//! directly. The two are not interchangeable — piper-plus GGUFs carry
//! `piper-plus-mb-istft-vits2` and route through
//! [`crate::piper_plus`]; plain VITS GGUFs carry `"vits-ja"` and route
//! through this module.
//!
//! # Reuses existing ops
//!
//! - **HiFi-GAN decoder**: shared [`vokra_ops::hifigan_generator`]
//!   primitive (M3-07 op, FR-OP-10). The forward decoder's every
//!   architectural axis (`n_mels`, `initial_channel`, `upsample_rates`,
//!   `upsample_kernel_sizes`, `resblock_kernel_sizes`,
//!   `resblock_dilation_sizes`, `sample_rate`, `leaky_relu_slope`)
//!   is exposed via [`VitsJaConfig::to_hifigan_attrs`] — a well-formed
//!   config produces a validated [`vokra_core::ir::graph::HifiGanAttrs`] without
//!   any conversion round-trip.
//! - **Text encoder / SDP / flow**: shared internally with
//!   [`crate::piper_plus`] via the JA-family Conformer-block topology
//!   (text encoder axes are shape-compatible; the SDP + flow primitives
//!   are op-level composites of Linear + DDSConv + rational-quadratic
//!   spline the piper-plus path already exercises).
//!
//! No new backend kernel is added — every building block is Conv1d +
//! LayerNorm + relative-position attention + Swish/GELU + Linear +
//! HiFi-GAN transposed-conv + MRF ResBlock, all covered by the existing
//! kernel inventory.
//!
//! # What lands in this Phase 5 slice
//!
//! - [`VitsJaTextEncoderConfig`] / [`VitsJaFlowConfig`] /
//!   [`VitsJaSdpConfig`] / [`VitsJaDecoderConfig`] / [`VitsJaConfig`] —
//!   every architectural hparam transcribed **verbatim** from the
//!   primary sources. [`VitsJaConfig::validate_for_forward`] fails
//!   loudly (FR-EX-08) on zeroed axes, non-even `head_dim`, mismatched
//!   upsample slice lengths, and non-positive dropout / eps.
//! - [`VitsJaConfig::to_hifigan_attrs`] — bridge to the shared
//!   [`vokra_core::ir::graph::HifiGanAttrs`] the [`vokra_ops::hifigan_generator`]
//!   primitive consumes; produces a struct that passes
//!   [`vokra_core::ir::graph::HifiGanAttrs`]'s `validate_shape` by construction.
//! - [`VitsJaWeights`] — deterministic zero-initialised
//!   scaffold ([`VitsJaWeights::synthesized`]); the real safetensors
//!   walk is a follow-up wave.
//! - [`VitsJaTts`] — engine handle carrying config + weights. The
//!   primary [`VitsJaTts::synthesize`] entry point returns
//!   [`VokraError::NotImplemented`] naming the follow-up wave and the
//!   **weight redistribution** blocker until real weights are bound
//!   AND the full text-encoder → SDP → flow → HiFi-GAN decoder chain is
//!   wired end-to-end (T29-equivalent follow-up wave — never a silent
//!   zero-fill, FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! ESPnet distributes VITS checkpoints as PyTorch `.pth` (`espnet2/gan_tts`);
//! the runtime **never** loads an ONNX graph (FR-LD-05, permanent
//! constraint); the pipeline is re-implemented natively (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::{Result, VokraError};
use vokra_ops::attrs::HifiGanAttrs;

/// `vokra.model.arch` a plain VITS JA GGUF must carry. Written by
/// `vokra-convert::models::vits_ja::ARCH`. Intentionally **distinct**
/// from every other arch tag in this crate — plain VITS decodes through
/// a HiFi-GAN generator directly, while piper-plus (MB-iSTFT-VITS2)
/// decodes through a sub-band iSTFT + PQMF post-net. Silently sharing
/// an arch tag with piper-plus would misroute the runtime dispatch (the
/// piper-plus module's decoder consumes a different tensor topology).
pub const EXPECTED_ARCH: &str = "vits-ja";

/// PCM sample rate the JSUT / JVS default ESPnet VITS recipe emits —
/// **22050 Hz**. Primary source: `egs2/jsut/tts1/conf/tuning/train_vits.yaml`
/// (`sampling_rate: 22050`) + `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml`
/// (`sampling_rate: 22050`). The full-band variant
/// (`egs2/jsut/tts1/conf/tuning/train_full_band_vits.yaml`) reshapes the
/// decoder + FFT / hop and emits 44100 Hz; see
/// [`VitsJaConfig::espnet_ja_full_band_44khz`].
pub const VITS_JA_SAMPLE_RATE: u32 = 22_050;

/// PCM sample rate the ESPnet full-band JSUT VITS recipe emits —
/// **44100 Hz**. Primary source: `train_full_band_vits.yaml`
/// (`sampling_rate: 44100`, `n_fft: 2048`, `hop_length: 512`,
/// `decoder_upsample_scales: [8, 8, 2, 2, 2]`,
/// `decoder_upsample_kernel_sizes: [16, 16, 4, 4, 4]`).
pub const VITS_JA_FULL_BAND_SAMPLE_RATE: u32 = 44_100;

/// LeakyReLU negative-slope used by HiFi-GAN. Primary source: upstream
/// `LRELU_SLOPE = 0.1` in `jik876/hifi-gan/models.py`, mirrored by
/// piper-plus `LRELU_SLOPE` and every ESPnet HiFi-GAN generator.
pub const VITS_JA_LEAKY_RELU_SLOPE: f32 = 0.1;

// ---------------------------------------------------------------------------
// Text encoder hparams
// ---------------------------------------------------------------------------

/// Text encoder (Conformer-style — `use_conformer_conv=false` on the JA
/// recipes) hparams. Every field is transcribed **verbatim** from the
/// primary sources — the `AVAILABLE_GENERATERS["vits_generator"] =
/// VITSGenerator` defaults in `espnet2/gan_tts/vits/vits.py`
/// cross-referenced with the JA recipe overrides in
/// `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
/// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct VitsJaTextEncoderConfig {
    /// `text_encoder_blocks` — Conformer block count, **6**.
    pub n_layer: u32,
    /// `text_encoder_attention_heads` — MHA head count, **2**.
    /// `head_dim = hidden_channels / n_head`.
    pub n_head: u32,
    /// `text_encoder_ffn_expand` — FFN inner-width expansion factor,
    /// **4**. FFN inner width = `hidden_channels * ffn_expand`.
    pub ffn_expand: u32,
    /// `text_encoder_positionwise_conv_kernel_size` — position-wise
    /// FFN conv kernel, **3** on the JA recipes. `vits.py`'s default
    /// is 1; the JA `train_vits.yaml` explicitly overrides to 3, so we
    /// pin the recipe value (a downstream permissive re-training that
    /// kept the `vits.py` default would ride the same converter path
    /// and can override this at bind time).
    pub positionwise_conv_kernel_size: u32,
    /// `text_encoder_dropout_rate` — the block-level dropout, **0.1**.
    pub dropout_rate: f32,
    /// `text_encoder_positional_dropout_rate` — dropout on the
    /// positional encoding, **0.0**.
    pub positional_dropout_rate: f32,
    /// `text_encoder_attention_dropout_rate` — attention softmax
    /// dropout, **0.1** on the JA recipes (0.0 on the `vits.py`
    /// default; the JA `train_vits.yaml` overrides).
    pub attention_dropout_rate: f32,
    /// `use_macaron_style_in_text_encoder` — whether the FFN is applied
    /// twice with a 0.5 residual scale (macaron), **true**.
    pub use_macaron_style: bool,
    /// `use_conformer_conv_in_text_encoder` — whether the Conformer
    /// convolution module is enabled, **false** on the JA recipes (per
    /// the `NOTE(kan-bayashi)` comment about BatchNorm1d + multi-GPU
    /// pytorch 1.7.1). The `vits.py` default is `true` — the JA recipe
    /// explicitly overrides.
    pub use_conformer_conv: bool,
    /// `text_encoder_conformer_kernel_size` — the Conformer conv
    /// kernel; only consumed when [`Self::use_conformer_conv`] is true.
    /// The JA recipe sets `-1` (sentinel = disabled); the `vits.py`
    /// default is 7. Stored as `Option<u32>`: `None` = disabled,
    /// `Some(k)` = enabled with kernel `k`.
    pub conformer_kernel_size: Option<u32>,
}

impl VitsJaTextEncoderConfig {
    /// Canonical ESPnet JA VITS text encoder (primary source:
    /// `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
    /// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml`).
    #[must_use]
    pub fn espnet_ja() -> Self {
        Self {
            n_layer: 6,
            n_head: 2,
            ffn_expand: 4,
            positionwise_conv_kernel_size: 3,
            dropout_rate: 0.1,
            positional_dropout_rate: 0.0,
            attention_dropout_rate: 0.1,
            use_macaron_style: true,
            use_conformer_conv: false,
            conformer_kernel_size: None,
        }
    }

    /// Miniature well-formed text-encoder config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            n_layer: 2,
            n_head: 2,
            ffn_expand: 2,
            positionwise_conv_kernel_size: 3,
            dropout_rate: 0.0,
            positional_dropout_rate: 0.0,
            attention_dropout_rate: 0.0,
            use_macaron_style: true,
            use_conformer_conv: false,
            conformer_kernel_size: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Residual affine coupling flow hparams
// ---------------------------------------------------------------------------

/// Residual affine coupling flow hparams — the normalising flow
/// operating on the posterior-encoder output. Every field is
/// transcribed **verbatim** from `train_vits.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct VitsJaFlowConfig {
    /// `flow_flows` — coupling-layer count, **4**.
    pub n_flow: u32,
    /// `flow_kernel_size` — WN dilated-conv kernel width, **5**.
    pub kernel_size: u32,
    /// `flow_base_dilation` — WN dilation base (layer `i` uses
    /// `base_dilation ^ i`), **1**.
    pub base_dilation: u32,
    /// `flow_layers` — WN dilated-conv layers per coupling, **4**.
    pub n_layer: u32,
    /// `flow_dropout_rate` — dropout inside the WN, **0.0**.
    pub dropout_rate: f32,
    /// `use_only_mean_in_flow` — whether the coupling emits only the
    /// mean (no log-scale), **true** on the JA recipes.
    pub use_only_mean: bool,
}

impl VitsJaFlowConfig {
    /// Canonical ESPnet JA VITS flow (primary source: `train_vits.yaml`).
    #[must_use]
    pub fn espnet_ja() -> Self {
        Self {
            n_flow: 4,
            kernel_size: 5,
            base_dilation: 1,
            n_layer: 4,
            dropout_rate: 0.0,
            use_only_mean: true,
        }
    }

    /// Miniature well-formed flow config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            n_flow: 2,
            kernel_size: 3,
            base_dilation: 1,
            n_layer: 2,
            dropout_rate: 0.0,
            use_only_mean: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Stochastic duration predictor hparams
// ---------------------------------------------------------------------------

/// Stochastic duration predictor (SDP) hparams. Every field is
/// transcribed **verbatim** from `train_vits.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct VitsJaSdpConfig {
    /// `stochastic_duration_predictor_kernel_size` — DDSConv kernel
    /// width, **3**.
    pub kernel_size: u32,
    /// `stochastic_duration_predictor_dropout_rate` — SDP dropout,
    /// **0.5**.
    pub dropout_rate: f32,
    /// `stochastic_duration_predictor_flows` — coupling count inside
    /// the SDP, **4**.
    pub n_flow: u32,
    /// `stochastic_duration_predictor_dds_conv_layers` — DDSConv layer
    /// count per SDP block, **3**.
    pub dds_conv_layers: u32,
}

impl VitsJaSdpConfig {
    /// Canonical ESPnet JA VITS SDP (primary source: `train_vits.yaml`).
    #[must_use]
    pub fn espnet_ja() -> Self {
        Self {
            kernel_size: 3,
            dropout_rate: 0.5,
            n_flow: 4,
            dds_conv_layers: 3,
        }
    }

    /// Miniature well-formed SDP config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            kernel_size: 3,
            dropout_rate: 0.0,
            n_flow: 2,
            dds_conv_layers: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// HiFi-GAN decoder hparams
// ---------------------------------------------------------------------------

/// HiFi-GAN decoder hparams — the plain HiFi-GAN generator that
/// terminates the plain VITS pipeline. Every field is transcribed
/// **verbatim** from `train_vits.yaml`. The runtime consumes this via
/// [`VitsJaConfig::to_hifigan_attrs`], which produces a
/// [`vokra_core::ir::graph::HifiGanAttrs`] that the shared
/// [`vokra_ops::hifigan_generator`] primitive drives.
#[derive(Debug, Clone, PartialEq)]
pub struct VitsJaDecoderConfig {
    /// `decoder_kernel_size` — the initial and final `conv1d` kernel
    /// (a symmetric HiFi-GAN pre / post pair), **7**.
    pub kernel_size: u32,
    /// `decoder_channels` — initial-channel width (`initial_channel`
    /// in [`vokra_core::ir::graph::HifiGanAttrs`]), **512**.
    pub initial_channel: u32,
    /// `decoder_upsample_scales` — per-stage transposed-conv strides,
    /// **[8, 8, 2, 2]** on the JA 22.05 kHz recipe. The product is the
    /// total upsample factor (256 = `hop_length`).
    pub upsample_scales: Vec<u32>,
    /// `decoder_upsample_kernel_sizes` — per-stage transposed-conv
    /// kernel sizes, **[16, 16, 4, 4]** on the JA 22.05 kHz recipe.
    /// Each entry is `2 * stride` per HiFi-GAN convention.
    pub upsample_kernel_sizes: Vec<u32>,
    /// `decoder_resblock_kernel_sizes` — MRF ResBlock kernel widths
    /// (three parallel branches), **[3, 7, 11]**.
    pub resblock_kernel_sizes: Vec<u32>,
    /// `decoder_resblock_dilations` — per-branch dilation lists,
    /// **[[1, 3, 5], [1, 3, 5], [1, 3, 5]]**. Outer axis = one entry
    /// per `resblock_kernel_sizes` entry; inner axis = per-layer
    /// dilations inside that ResBlock.
    pub resblock_dilations: Vec<Vec<u32>>,
    /// `use_weight_norm_in_decoder` — training-time flag, **true**.
    /// Recorded for round-trip completeness; the inference forward
    /// consumes the merged weights (weight norm is folded at export).
    pub use_weight_norm: bool,
}

impl VitsJaDecoderConfig {
    /// Canonical ESPnet JA 22.05 kHz VITS HiFi-GAN decoder (primary
    /// source: `train_vits.yaml`).
    #[must_use]
    pub fn espnet_ja_22khz() -> Self {
        Self {
            kernel_size: 7,
            initial_channel: 512,
            upsample_scales: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_weight_norm: true,
        }
    }

    /// Canonical ESPnet JA full-band 44.1 kHz VITS HiFi-GAN decoder
    /// (primary source: `train_full_band_vits.yaml`, extra upsample
    /// stage — five stages `[8, 8, 2, 2, 2]` with kernels
    /// `[16, 16, 4, 4, 4]` for total upsample 512 = `hop_length`).
    #[must_use]
    pub fn espnet_ja_full_band_44khz() -> Self {
        Self {
            kernel_size: 7,
            initial_channel: 512,
            upsample_scales: vec![8, 8, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_weight_norm: true,
        }
    }

    /// Miniature well-formed decoder config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            kernel_size: 3,
            initial_channel: 8,
            upsample_scales: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilations: vec![vec![1, 3], vec![1, 3]],
            use_weight_norm: true,
        }
    }

    /// Total time-domain upsampling ratio (product of stage strides).
    /// Equals `hop_length` for a well-formed HiFi-GAN.
    #[must_use]
    pub fn total_upsample_factor(&self) -> u32 {
        self.upsample_scales.iter().product()
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Resolved ESPnet JA VITS hparam snapshot — every field is transcribed
/// from `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
/// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` (JSUT single-speaker
/// / JVS 100-speaker variants share this shape) or the full-band
/// variant `train_full_band_vits.yaml` (see
/// [`Self::espnet_ja_full_band_44khz`]).
#[derive(Debug, Clone, PartialEq)]
pub struct VitsJaConfig {
    /// Input vocabulary size (`vocabs` in `VITSGenerator.__init__`).
    /// **Not** in the training YAML — it is derived at runtime from the
    /// phoneme table the JA text frontend produces (e.g. pyopenjtalk
    /// phoneme set). Vokra stores the vocabulary alongside the checkpoint
    /// in the GGUF `vokra.vits_ja.phoneme_symbols` array; the config
    /// carries its length here for validation.
    pub vocab_size: u32,
    /// Number of mel bins (posterior-encoder output width fed to the
    /// HiFi-GAN decoder). ESPnet's `mel_loss_params.n_mels = 80` for
    /// the JA 22.05 kHz recipe (the acoustic loss target); the decoder
    /// consumes this width verbatim.
    pub n_mels: u32,
    /// Posterior-encoder input width `aux_channels` — `n_fft / 2 + 1`
    /// (**513** for the 22 kHz JA recipe's `n_fft = 1024`, **1025** for
    /// the full-band variant's `n_fft = 2048`). Recorded so a downstream
    /// caller can cross-check the STFT front-end.
    pub aux_channels: u32,
    /// `hidden_channels` — the flow / posterior-encoder / prior residual
    /// stream width, **192**.
    pub hidden_channels: u32,
    /// `segment_size` — training-time random-segment length (in
    /// posterior-encoder frames), **32**. Recorded for round-trip
    /// completeness; the inference forward does not chunk.
    pub segment_size: u32,
    /// `spks` — speaker count (**> 1** for the JVS 100-speaker variant,
    /// **None** for the JSUT single-speaker variant). Stored as `Option`
    /// because `-1` in the YAML means "no speaker embedding".
    pub spks: Option<u32>,
    /// `langs` — language count. **None** on the monolingual JA
    /// recipes; a multilingual variant would set it.
    pub langs: Option<u32>,
    /// `spk_embed_dim` — external speaker-embedding width (bypasses
    /// the built-in speaker table). **None** on the JA recipes.
    pub spk_embed_dim: Option<u32>,
    /// `global_channels` — global-conditioning channel count.
    /// `-1` in the YAML means "no global conditioning"; stored here
    /// as `None` for that case.
    pub global_channels: Option<u32>,
    /// Text encoder hparams.
    pub text_encoder: VitsJaTextEncoderConfig,
    /// Flow hparams.
    pub flow: VitsJaFlowConfig,
    /// Stochastic duration predictor hparams.
    pub sdp: VitsJaSdpConfig,
    /// HiFi-GAN decoder hparams.
    pub decoder: VitsJaDecoderConfig,
    /// PCM sample rate (**22050** for the JSUT / JVS recipes; **44100**
    /// for the full-band variant).
    pub sample_rate: u32,
}

impl VitsJaConfig {
    /// Canonical ESPnet JA VITS 22.05 kHz single-speaker (JSUT) config.
    /// Every value is transcribed from
    /// `egs2/jsut/tts1/conf/tuning/train_vits.yaml`.
    ///
    /// `vocab_size` defaults to the pyopenjtalk phoneme count Vokra
    /// bakes into the GGUF (kept as a parameter placeholder — the real
    /// count rides the `vokra.vits_ja.phoneme_symbols` array); this
    /// default of **43** matches the pyopenjtalk-derived JSUT phoneme
    /// set the JA ESPnet recipe uses at inference time (see the
    /// primary source `pyopenjtalk` `PP_LIST`).
    #[must_use]
    pub fn espnet_ja_jsut_22khz() -> Self {
        Self {
            vocab_size: 43,
            n_mels: 80,
            aux_channels: 513,
            hidden_channels: 192,
            segment_size: 32,
            spks: None,
            langs: None,
            spk_embed_dim: None,
            global_channels: None,
            text_encoder: VitsJaTextEncoderConfig::espnet_ja(),
            flow: VitsJaFlowConfig::espnet_ja(),
            sdp: VitsJaSdpConfig::espnet_ja(),
            decoder: VitsJaDecoderConfig::espnet_ja_22khz(),
            sample_rate: VITS_JA_SAMPLE_RATE,
        }
    }

    /// Canonical ESPnet JA VITS 22.05 kHz multi-speaker (JVS 100-speaker)
    /// config. Primary source:
    /// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` — every axis
    /// matches the JSUT config except `spks = 100` (JVS ships 100
    /// speakers `jvs001` .. `jvs100`).
    #[must_use]
    pub fn espnet_ja_jvs_22khz() -> Self {
        let mut cfg = Self::espnet_ja_jsut_22khz();
        cfg.spks = Some(100);
        cfg
    }

    /// Canonical ESPnet JA full-band VITS 44.1 kHz single-speaker
    /// config. Primary source:
    /// `egs2/jsut/tts1/conf/tuning/train_full_band_vits.yaml`.
    /// Differs from [`Self::espnet_ja_jsut_22khz`] on three axes:
    /// `sample_rate = 44100`, `aux_channels = 1025` (`n_fft = 2048` →
    /// `n_fft/2 + 1`), and the decoder reshapes to five upsample
    /// stages `[8, 8, 2, 2, 2]` with kernels `[16, 16, 4, 4, 4]`.
    #[must_use]
    pub fn espnet_ja_full_band_44khz() -> Self {
        Self {
            vocab_size: 43,
            n_mels: 80,
            aux_channels: 1025,
            hidden_channels: 192,
            segment_size: 32,
            spks: None,
            langs: None,
            spk_embed_dim: None,
            global_channels: None,
            text_encoder: VitsJaTextEncoderConfig::espnet_ja(),
            flow: VitsJaFlowConfig::espnet_ja(),
            sdp: VitsJaSdpConfig::espnet_ja(),
            decoder: VitsJaDecoderConfig::espnet_ja_full_band_44khz(),
            sample_rate: VITS_JA_FULL_BAND_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            vocab_size: 16,
            n_mels: 4,
            aux_channels: 9,
            hidden_channels: 8,
            segment_size: 8,
            spks: None,
            langs: None,
            spk_embed_dim: None,
            global_channels: None,
            text_encoder: VitsJaTextEncoderConfig::tiny_for_tests(),
            flow: VitsJaFlowConfig::tiny_for_tests(),
            sdp: VitsJaSdpConfig::tiny_for_tests(),
            decoder: VitsJaDecoderConfig::tiny_for_tests(),
            sample_rate: VITS_JA_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        // Global axes.
        if self.vocab_size == 0
            || self.n_mels == 0
            || self.aux_channels == 0
            || self.hidden_channels == 0
            || self.segment_size == 0
            || self.sample_rate == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja config: zero-size global hparam (vocab_size={}, n_mels={}, \
                 aux_channels={}, hidden_channels={}, segment_size={}, sample_rate={})",
                self.vocab_size,
                self.n_mels,
                self.aux_channels,
                self.hidden_channels,
                self.segment_size,
                self.sample_rate,
            )));
        }

        // Text encoder axes.
        let t = &self.text_encoder;
        if t.n_layer == 0 || t.n_head == 0 || t.ffn_expand == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja text encoder: zero-size hparam (n_layer={}, n_head={}, \
                 ffn_expand={})",
                t.n_layer, t.n_head, t.ffn_expand,
            )));
        }
        if self.hidden_channels % t.n_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja text encoder: n_head ({}) must divide hidden_channels ({})",
                t.n_head, self.hidden_channels,
            )));
        }
        let head_dim = self.hidden_channels / t.n_head;
        if head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja text encoder: rel_pos attention pairs require even head_dim \
                 (hidden_channels / n_head = {head_dim})"
            )));
        }
        if !(t.dropout_rate.is_finite() && (0.0..1.0).contains(&t.dropout_rate)) {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja text encoder: dropout_rate must be a finite [0, 1) probability \
                 (got {})",
                t.dropout_rate,
            )));
        }
        if !(t.positional_dropout_rate.is_finite()
            && (0.0..1.0).contains(&t.positional_dropout_rate))
        {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja text encoder: positional_dropout_rate must be a finite [0, 1) \
                 probability (got {})",
                t.positional_dropout_rate,
            )));
        }
        if !(t.attention_dropout_rate.is_finite() && (0.0..1.0).contains(&t.attention_dropout_rate))
        {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja text encoder: attention_dropout_rate must be a finite [0, 1) \
                 probability (got {})",
                t.attention_dropout_rate,
            )));
        }
        if t.positionwise_conv_kernel_size == 0 {
            return Err(VokraError::InvalidArgument(
                "vits-ja text encoder: positionwise_conv_kernel_size must be > 0".to_owned(),
            ));
        }
        if t.use_conformer_conv && t.conformer_kernel_size.is_none() {
            return Err(VokraError::InvalidArgument(
                "vits-ja text encoder: use_conformer_conv=true requires \
                 conformer_kernel_size=Some(_)"
                    .to_owned(),
            ));
        }
        if let Some(k) = t.conformer_kernel_size
            && k == 0
        {
            return Err(VokraError::InvalidArgument(
                "vits-ja text encoder: conformer_kernel_size must be > 0 when Some".to_owned(),
            ));
        }

        // Flow axes.
        let f = &self.flow;
        if f.n_flow == 0 || f.kernel_size == 0 || f.n_layer == 0 || f.base_dilation == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja flow: zero-size hparam (n_flow={}, kernel_size={}, n_layer={}, \
                 base_dilation={})",
                f.n_flow, f.kernel_size, f.n_layer, f.base_dilation,
            )));
        }
        if !(f.dropout_rate.is_finite() && (0.0..1.0).contains(&f.dropout_rate)) {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja flow: dropout_rate must be a finite [0, 1) probability (got {})",
                f.dropout_rate,
            )));
        }

        // SDP axes.
        let s = &self.sdp;
        if s.kernel_size == 0 || s.n_flow == 0 || s.dds_conv_layers == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja sdp: zero-size hparam (kernel_size={}, n_flow={}, \
                 dds_conv_layers={})",
                s.kernel_size, s.n_flow, s.dds_conv_layers,
            )));
        }
        if !(s.dropout_rate.is_finite() && (0.0..1.0).contains(&s.dropout_rate)) {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja sdp: dropout_rate must be a finite [0, 1) probability (got {})",
                s.dropout_rate,
            )));
        }

        // Decoder axes.
        let d = &self.decoder;
        if d.kernel_size == 0 || d.initial_channel == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja decoder: zero-size hparam (kernel_size={}, initial_channel={})",
                d.kernel_size, d.initial_channel,
            )));
        }
        if d.upsample_scales.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vits-ja decoder: upsample_scales must be non-empty".to_owned(),
            ));
        }
        if d.upsample_kernel_sizes.len() != d.upsample_scales.len() {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja decoder: upsample_kernel_sizes.len() {} != upsample_scales.len() {}",
                d.upsample_kernel_sizes.len(),
                d.upsample_scales.len(),
            )));
        }
        if d.resblock_kernel_sizes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vits-ja decoder: resblock_kernel_sizes must be non-empty".to_owned(),
            ));
        }
        if d.resblock_dilations.len() != d.resblock_kernel_sizes.len() {
            return Err(VokraError::InvalidArgument(format!(
                "vits-ja decoder: resblock_dilations outer.len() {} != \
                 resblock_kernel_sizes.len() {}",
                d.resblock_dilations.len(),
                d.resblock_kernel_sizes.len(),
            )));
        }
        for (i, s) in d.upsample_scales.iter().enumerate() {
            if *s == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vits-ja decoder: upsample_scales[{i}] must be > 0"
                )));
            }
        }
        for (i, k) in d.upsample_kernel_sizes.iter().enumerate() {
            if *k == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vits-ja decoder: upsample_kernel_sizes[{i}] must be > 0"
                )));
            }
        }
        for (i, k) in d.resblock_kernel_sizes.iter().enumerate() {
            if *k == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vits-ja decoder: resblock_kernel_sizes[{i}] must be > 0"
                )));
            }
        }
        for (i, branch) in d.resblock_dilations.iter().enumerate() {
            if branch.is_empty() {
                return Err(VokraError::InvalidArgument(format!(
                    "vits-ja decoder: resblock_dilations[{i}] must be non-empty"
                )));
            }
            for (j, dil) in branch.iter().enumerate() {
                if *dil == 0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "vits-ja decoder: resblock_dilations[{i}][{j}] must be > 0"
                    )));
                }
            }
        }

        Ok(())
    }

    /// Bridges the decoder axes to the shared
    /// [`vokra_core::ir::graph::HifiGanAttrs`] struct the
    /// [`vokra_ops::hifigan_generator`] primitive consumes.
    ///
    /// Guaranteed to produce a [`HifiGanAttrs`] that passes
    /// [`HifiGanAttrs::validate_shape`] iff
    /// [`Self::validate_for_forward`] succeeds — the two validators
    /// share the same non-zero + slice-length invariants.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] via
    /// [`Self::validate_for_forward`] if the config is ill-formed.
    pub fn to_hifigan_attrs(&self) -> Result<HifiGanAttrs> {
        self.validate_for_forward()?;
        let d = &self.decoder;
        let attrs = HifiGanAttrs {
            n_mels: self.n_mels as usize,
            initial_channel: d.initial_channel as usize,
            upsample_rates: d.upsample_scales.iter().map(|&s| s as usize).collect(),
            upsample_kernel_sizes: d
                .upsample_kernel_sizes
                .iter()
                .map(|&k| k as usize)
                .collect(),
            resblock_kernel_sizes: d
                .resblock_kernel_sizes
                .iter()
                .map(|&k| k as usize)
                .collect(),
            resblock_dilation_sizes: d
                .resblock_dilations
                .iter()
                .map(|branch| branch.iter().map(|&d| d as usize).collect())
                .collect(),
            sample_rate: self.sample_rate,
            leaky_relu_slope: VITS_JA_LEAKY_RELU_SLOPE,
            // Canonical VITS/MB-iSTFT-VITS2 preset ships with
            // `resblock='1'` (ResBlock1) — see
            // `tools/parity/vendor/vits/modules.py`. The real weight
            // path is a scaffold today (no `convs2` tensor emission yet
            // for VITS JA) but declaring the topology honestly here
            // means when the loader lands it must supply c2 or fail
            // loudly per `mrf_branch_forward`'s FR-EX-08 gate.
            res_block_type: vokra_core::ir::ResBlockType::V1,
        };
        // Redundant with `validate_for_forward` on paper, but the two
        // validators enforce their own contracts — running both here
        // guarantees that the returned struct always passes the op's
        // own gate (FR-EX-08 — no silent inconsistency between the
        // model config and the op's admissible shape).
        attrs.validate_shape()?;
        Ok(attrs)
    }
}

// ---------------------------------------------------------------------------
// Weight-store scaffold
// ---------------------------------------------------------------------------

/// Plain VITS JA weight store scaffold.
///
/// Real binding is a follow-up wave (T29-equivalent — the safetensors
/// walk for the text encoder / SDP / flow / HiFi-GAN decoder defers to
/// the T29 tensor-name manifest fetch). This scaffold carries only
/// aggregate byte bundles so downstream shape flow / handshake tests
/// are unblocked; the sole invariant this slice pins is that
/// `is_synthesized = true` prevents a spurious synthesize call from
/// returning zero audio (FR-EX-08 — the loud [`VitsJaTts::synthesize`]
/// guard).
#[derive(Debug, Clone)]
pub struct VitsJaWeights {
    /// Placeholder for the text-encoder tensor bytes (aggregate).
    pub text_encoder: Vec<f32>,
    /// Placeholder for the SDP tensor bytes (aggregate).
    pub sdp: Vec<f32>,
    /// Placeholder for the flow tensor bytes (aggregate).
    pub flow: Vec<f32>,
    /// Placeholder for the HiFi-GAN decoder tensor bytes (aggregate).
    pub decoder: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

impl VitsJaWeights {
    /// Builds a deterministic zero-initialised fixture (shape scaffold
    /// only — every slot is `Vec::new()`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &VitsJaConfig) -> Result<Self> {
        config.validate_for_forward()?;
        Ok(Self {
            text_encoder: Vec::new(),
            sdp: Vec::new(),
            flow: Vec::new(),
            decoder: Vec::new(),
            is_synthesized: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Plain VITS JA engine handle.
///
/// Carries the resolved config + weight store. [`Self::synthesize`] is
/// the primary text → PCM entry point; until real weights are bound
/// AND the full text-encoder → SDP → flow → HiFi-GAN decoder chain is
/// wired end-to-end (T29-equivalent follow-up wave), it returns
/// [`VokraError::NotImplemented`] naming the blocker (FR-EX-08 —
/// never a silent zero-fill or empty audio buffer).
///
/// # ⚠️  Weight redistribution note
///
/// The publicly distributed ESPnet-JSUT / ESPnet-JVS / COEIROINK JA
/// VITS checkpoints ride on **corpus terms that forbid re-distribution
/// of the trained weight** (JSUT: "Re-distribution is not permitted",
/// JVS: same). The converter default-stamps GGUFs produced from those
/// checkpoints as [`vokra_core::LicenseClass::RedistributionForbidden`].
/// Users who trained their own permissive-corpus VITS override with
/// `vokra-convert --license <spdx>` at conversion time. Architecture
/// rides Apache-2.0 code (ESPnet) and MIT code (jaywalnut310/vits)
/// and is always independently implementable.
#[derive(Debug, Clone)]
pub struct VitsJaTts {
    cfg: VitsJaConfig,
    weights: VitsJaWeights,
}

impl VitsJaTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// config at construction time so a mismatched pair fails loudly
    /// here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from
    /// [`VitsJaConfig::validate_for_forward`].
    pub fn new(cfg: VitsJaConfig, weights: VitsJaWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        Ok(Self { cfg, weights })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &VitsJaConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`VitsJaWeights::synthesized`] (never a real upstream checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesises PCM for `text` at the config's sample rate.
    ///
    /// This is the primary text → PCM entry point. **Real weights
    /// required**: synthesised-weight builds cannot produce meaningful
    /// audio, so this returns [`VokraError::NotImplemented`] naming
    /// the blocker (FR-EX-08 — never a silent zero-fill or empty
    /// audio buffer).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not
    ///   yet bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vits-ja synthesize: text is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "vits-ja synthesize: this engine holds synthesised weights (deterministic \
                 scaffold fixture from VitsJaWeights::synthesized) — synthesised-weight audio \
                 would be a hallucinated waveform, not real speech. Bind real ESPnet JA VITS \
                 weights before invoking synthesize. NOTE: publicly distributed ESPnet-JSUT / \
                 ESPnet-JVS / COEIROINK weights carry corpus terms that forbid re-distribution \
                 of the trained weight (JSUT: `Re-distribution is not permitted`; JVS: same); \
                 architecture rides Apache-2.0 (ESPnet) and MIT (jaywalnut310/vits) and is \
                 always independently implementable (whisper.cpp-style clean-room re-imp, \
                 CLAUDE.md 設計判断 4). The shape flow (config validation, weight-store \
                 construction, text-empty check, HiFiGanAttrs bridge) is exercised through \
                 VitsJaTts::new; real-checkpoint binding + text-encoder → SDP → flow → \
                 hifigan_generator forward wiring lands in a follow-up wave (T29-equivalent).",
            ));
        }
        Err(VokraError::NotImplemented(
            "vits-ja synthesize: real weights are bound, but the plain VITS forward path \
             (JA text frontend → phoneme id sequence → Conformer-style text encoder → \
             stochastic duration predictor → residual affine coupling flow → HiFi-GAN \
             generator via vokra_ops::hifigan_generator → 22.05 kHz (or 44.1 kHz full-band) \
             mono PCM) has not landed yet. Follow-up wave (T29-equivalent): (1) tokenise \
             `text` through the ESPnet JA phoneme frontend (pyopenjtalk-derived phoneme set), \
             (2) run the text encoder + SDP + flow to produce the posterior mean latent, (3) \
             feed the latent through the HiFi-GAN decoder built from VitsJaConfig::to_hifigan_attrs, \
             (4) return PCM at VitsJaConfig::sample_rate.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Arch tag distinctness ------------------------------------------

    #[test]
    fn expected_arch_is_vits_ja() {
        assert_eq!(EXPECTED_ARCH, "vits-ja");
    }

    /// plain VITS decodes through a HiFi-GAN generator directly, while
    /// piper-plus (MB-iSTFT-VITS2) decodes through a sub-band iSTFT +
    /// PQMF post-net. Silently sharing an arch tag with piper-plus
    /// would misroute the runtime dispatch (the piper-plus module's
    /// decoder consumes a different tensor topology).
    #[test]
    fn arch_is_distinct_from_piper_plus_and_other_tts_siblings() {
        // Explicit distinctness from piper-plus (MB-iSTFT-VITS2).
        assert_ne!(EXPECTED_ARCH, "piper-plus-mb-istft-vits2");
        // Distinct from every neighbouring TTS module's arch tag.
        assert_ne!(EXPECTED_ARCH, crate::irodori::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::vibevoice::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::voxcpm2::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::cosyvoice3::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::qwen3_tts::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_nano::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_turbo::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::dia::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::zonos::EXPECTED_ARCH);
    }

    // ---- Constants ------------------------------------------------------

    #[test]
    fn sample_rate_constants_match_primary_source() {
        assert_eq!(VITS_JA_SAMPLE_RATE, 22_050);
        assert_eq!(VITS_JA_FULL_BAND_SAMPLE_RATE, 44_100);
    }

    #[test]
    fn leaky_relu_slope_matches_hifigan_upstream() {
        assert!((VITS_JA_LEAKY_RELU_SLOPE - 0.1).abs() < 1e-9);
    }

    // ---- Primary-source pins (JSUT 22 kHz) ------------------------------

    #[test]
    fn text_encoder_matches_primary_source() {
        let t = VitsJaTextEncoderConfig::espnet_ja();
        // egs2/jsut/tts1/conf/tuning/train_vits.yaml
        assert_eq!(t.n_layer, 6);
        assert_eq!(t.n_head, 2);
        assert_eq!(t.ffn_expand, 4);
        assert_eq!(t.positionwise_conv_kernel_size, 3);
        assert!((t.dropout_rate - 0.1).abs() < 1e-6);
        assert!((t.positional_dropout_rate - 0.0).abs() < 1e-9);
        assert!((t.attention_dropout_rate - 0.1).abs() < 1e-6);
        assert!(t.use_macaron_style);
        // JA recipe explicitly disables conformer conv.
        assert!(!t.use_conformer_conv);
        assert_eq!(t.conformer_kernel_size, None);
    }

    #[test]
    fn flow_matches_primary_source() {
        let f = VitsJaFlowConfig::espnet_ja();
        assert_eq!(f.n_flow, 4);
        assert_eq!(f.kernel_size, 5);
        assert_eq!(f.base_dilation, 1);
        assert_eq!(f.n_layer, 4);
        assert!((f.dropout_rate - 0.0).abs() < 1e-9);
        assert!(f.use_only_mean);
    }

    #[test]
    fn sdp_matches_primary_source() {
        let s = VitsJaSdpConfig::espnet_ja();
        assert_eq!(s.kernel_size, 3);
        assert!((s.dropout_rate - 0.5).abs() < 1e-6);
        assert_eq!(s.n_flow, 4);
        assert_eq!(s.dds_conv_layers, 3);
    }

    #[test]
    fn decoder_22khz_matches_primary_source() {
        let d = VitsJaDecoderConfig::espnet_ja_22khz();
        assert_eq!(d.kernel_size, 7);
        assert_eq!(d.initial_channel, 512);
        assert_eq!(d.upsample_scales, vec![8, 8, 2, 2]);
        assert_eq!(d.upsample_kernel_sizes, vec![16, 16, 4, 4]);
        assert_eq!(d.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            d.resblock_dilations,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        assert!(d.use_weight_norm);
        // Total upsample factor = 8 * 8 * 2 * 2 = 256 = hop_length for
        // the 22.05 kHz JA recipe.
        assert_eq!(d.total_upsample_factor(), 256);
    }

    #[test]
    fn decoder_44khz_full_band_matches_primary_source() {
        let d = VitsJaDecoderConfig::espnet_ja_full_band_44khz();
        // egs2/jsut/tts1/conf/tuning/train_full_band_vits.yaml
        assert_eq!(d.kernel_size, 7);
        assert_eq!(d.initial_channel, 512);
        assert_eq!(d.upsample_scales, vec![8, 8, 2, 2, 2]);
        assert_eq!(d.upsample_kernel_sizes, vec![16, 16, 4, 4, 4]);
        assert_eq!(d.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            d.resblock_dilations,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        // Total upsample = 8 * 8 * 2 * 2 * 2 = 512 = hop_length (44100).
        assert_eq!(d.total_upsample_factor(), 512);
    }

    #[test]
    fn top_level_jsut_22khz_matches_primary_source() {
        let c = VitsJaConfig::espnet_ja_jsut_22khz();
        assert_eq!(c.hidden_channels, 192);
        assert_eq!(c.segment_size, 32);
        assert_eq!(c.aux_channels, 513); // n_fft=1024 → 513
        assert_eq!(c.n_mels, 80);
        assert_eq!(c.sample_rate, 22_050);
        assert!(c.spks.is_none()); // single-speaker
        assert!(c.langs.is_none());
        assert!(c.spk_embed_dim.is_none());
        assert!(c.global_channels.is_none());
    }

    #[test]
    fn top_level_jvs_22khz_matches_primary_source() {
        let c = VitsJaConfig::espnet_ja_jvs_22khz();
        // JVS finetune shares the same hparams as JSUT except spks.
        assert_eq!(c.spks, Some(100));
        assert_eq!(c.sample_rate, 22_050);
    }

    #[test]
    fn top_level_full_band_44khz_matches_primary_source() {
        let c = VitsJaConfig::espnet_ja_full_band_44khz();
        assert_eq!(c.sample_rate, 44_100);
        assert_eq!(c.aux_channels, 1025); // n_fft=2048 → 1025
        assert_eq!(c.decoder.upsample_scales.len(), 5);
    }

    // ---- Validation holds by construction -------------------------------

    #[test]
    fn canonical_jsut_config_validates() {
        VitsJaConfig::espnet_ja_jsut_22khz()
            .validate_for_forward()
            .expect("canonical JSUT 22 kHz must validate");
    }

    #[test]
    fn canonical_jvs_config_validates() {
        VitsJaConfig::espnet_ja_jvs_22khz()
            .validate_for_forward()
            .expect("canonical JVS 22 kHz must validate");
    }

    #[test]
    fn canonical_full_band_config_validates() {
        VitsJaConfig::espnet_ja_full_band_44khz()
            .validate_for_forward()
            .expect("canonical 44 kHz full-band must validate");
    }

    #[test]
    fn tiny_config_validates() {
        VitsJaConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny must validate");
    }

    #[test]
    fn validate_rejects_zero_axes() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.hidden_channels = 0;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_head_not_dividing_hidden() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        // 192 / 5 does not divide.
        c.text_encoder.n_head = 5;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_odd_head_dim() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        // Force odd head_dim: hidden=6, n_head=2 → head_dim=3 (odd).
        c.hidden_channels = 6;
        c.text_encoder.n_head = 2;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_dropout_out_of_range() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.text_encoder.dropout_rate = 1.0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.text_encoder.dropout_rate = f32::NAN;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn validate_rejects_mismatched_upsample_slice_lengths() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.decoder.upsample_kernel_sizes = vec![16, 16, 4]; // 3 vs 4 scales
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_mismatched_resblock_lengths() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.decoder.resblock_dilations = vec![vec![1, 3, 5], vec![1, 3, 5]]; // 2 vs 3
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_conformer_conv_true_without_kernel() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.text_encoder.use_conformer_conv = true;
        c.text_encoder.conformer_kernel_size = None;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_accepts_conformer_conv_true_with_kernel() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.text_encoder.use_conformer_conv = true;
        c.text_encoder.conformer_kernel_size = Some(7);
        c.validate_for_forward()
            .expect("conformer conv with kernel is valid");
    }

    // ---- HiFi-GAN bridge -------------------------------------------------

    #[test]
    fn to_hifigan_attrs_matches_decoder_axes_on_22khz() {
        let c = VitsJaConfig::espnet_ja_jsut_22khz();
        let attrs = c.to_hifigan_attrs().expect("bridge must succeed");
        assert_eq!(attrs.n_mels, 80);
        assert_eq!(attrs.initial_channel, 512);
        assert_eq!(attrs.upsample_rates, vec![8, 8, 2, 2]);
        assert_eq!(attrs.upsample_kernel_sizes, vec![16, 16, 4, 4]);
        assert_eq!(attrs.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            attrs.resblock_dilation_sizes,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        assert_eq!(attrs.sample_rate, 22_050);
        assert!((attrs.leaky_relu_slope - 0.1).abs() < 1e-9);
        // The op's own validator must accept every well-formed bridge.
        attrs.validate_shape().expect("op-side validator");
    }

    #[test]
    fn to_hifigan_attrs_matches_decoder_axes_on_44khz() {
        let c = VitsJaConfig::espnet_ja_full_band_44khz();
        let attrs = c.to_hifigan_attrs().expect("bridge must succeed");
        assert_eq!(attrs.sample_rate, 44_100);
        assert_eq!(attrs.total_upsample_factor(), 512);
    }

    #[test]
    fn to_hifigan_attrs_rejects_ill_formed_config() {
        let mut c = VitsJaConfig::espnet_ja_jsut_22khz();
        c.decoder.upsample_kernel_sizes = vec![16, 16, 4]; // slice mismatch
        assert!(matches!(
            c.to_hifigan_attrs(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- Engine posture --------------------------------------------------

    #[test]
    fn engine_new_accepts_well_formed_config() {
        let c = VitsJaConfig::tiny_for_tests();
        let w = VitsJaWeights::synthesized(&c).expect("synthesized");
        let engine = VitsJaTts::new(c.clone(), w).expect("new");
        assert_eq!(engine.config().hidden_channels, c.hidden_channels);
        assert!(engine.is_synthesized());
    }

    #[test]
    fn engine_synthesize_rejects_empty_text() {
        let c = VitsJaConfig::tiny_for_tests();
        let w = VitsJaWeights::synthesized(&c).expect("synthesized");
        let engine = VitsJaTts::new(c, w).expect("new");
        let err = engine.synthesize("").unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// FR-EX-08 — synthesised-weight builds cannot produce meaningful
    /// audio, so `synthesize` must fail loudly (never zero-fill).
    #[test]
    fn engine_synthesize_rejects_synthesized_weights_loudly() {
        let c = VitsJaConfig::tiny_for_tests();
        let w = VitsJaWeights::synthesized(&c).expect("synthesized");
        let engine = VitsJaTts::new(c, w).expect("new");
        let err = engine.synthesize("こんにちは").unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                // Message must name the model, the FR-EX-08 blocker, AND
                // the corpus-based re-distribution note so an operator
                // can act on it.
                assert!(msg.contains("vits-ja"), "message must name arch: {msg}");
                assert!(
                    msg.contains("synthesised weights") || msg.contains("synthesized weights"),
                    "message must name the fixture nature: {msg}"
                );
                assert!(
                    msg.contains("Re-distribution")
                        || msg.contains("JSUT")
                        || msg.contains("JVS")
                        || msg.contains("corpus"),
                    "message must name the weight-redistribution constraint: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// vits-ja id to `RedistributionForbidden` **by default** (the
    /// weight-redistribution constraint is what the module doc-string
    /// documents). Cross-crate test to keep this module's registry-
    /// side contract honest.
    #[test]
    fn registry_lookup_maps_vits_ja_to_redistribution_forbidden_by_default() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "vits-ja",
            "vits_ja",
            "espnet-vits-ja",
            "espnet-jsut-vits",
            "espnet-jvs-vits",
            "coeiroink-vits",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::RedistributionForbidden),
                "registry must map `{id}` to RedistributionForbidden (JSUT/JVS/COEIROINK \
                 corpus terms forbid trained-weight redistribution)"
            );
        }
    }
}
