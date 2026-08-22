//! Standalone **HiFi-GAN vocoder** runtime binder — turns a mel
//! spectrogram into a raw PCM waveform via the shared
//! [`vokra_ops::hifigan::hifigan_generator`] op (HiFi-GAN family neural
//! vocoder — Kong et al. 2020, arXiv:2010.05646; jik876/hifi-gan, MIT).
//!
//! # What this module is (vs the SBV2 wrapper)
//!
//! `crates/vokra-models/src/sbv2/decoder.rs::SbV2Decoder` is the same
//! shape but attached to the SBV2 v2 (Style-Bert-VITS2 JP-Extra)
//! text-encoder → SDP → flow → HiFi-GAN chain, and it lives inside the
//! SBV2 model tree so that consumer builds its decoder from its own
//! `SbV2Config`. This module is the **standalone** binder for the two
//! HiFi-GAN-family GGUFs the converters
//! `crates/vokra-convert/src/models/{hifigan_vocoder, speecht5_hifigan}.rs`
//! already emit today — `speechbrain/tts-hifigan-libritts-22050Hz`
//! (apache-2.0, 22.05 kHz) and `microsoft/speecht5_hifigan` (MIT,
//! 16 kHz). Both converters forward-reference this module in their own
//! docstrings (search
//! `crates/vokra-models/src/hifigan_vocoder/` and
//! `crates/vokra-models/src/speecht5_hifigan/`); this binder is the
//! landing site those references pointed at.
//!
//! # Design mirrors `SbV2Decoder`
//!
//! [`HiFiGan`] bundles a pre-built [`HifiGanWeights`] + [`HifiGanAttrs`]
//! + [`HifiGanConfig`] triple plus a `sample_rate` metadata field, and
//!   validates them up-front in [`HiFiGan::new`] so a mismatched pair
//!   fails loudly at construction — never deep inside a forward
//!   (FR-EX-08).
//!
//! The `vokra_ops::hifigan` op is a **free function** over those three
//! separately-built bundles (not a method on some `HiFiGanGenerator`
//! struct — see `SbV2Decoder`'s module docstring for the same
//! observation); [`HiFiGan::decode`] is a thin
//! [`vokra_ops::hifigan::hifigan_generator`] call that keeps the wrapper's surface
//! stable even as the op grows internal knobs (INT8 opt-in, `cond`
//! layer, etc.).
//!
//! # Weight-load posture
//!
//! `speecht5_hifigan` is strict and complete: the official 16 kHz config is
//! transcribed from `microsoft/speecht5_hifigan/config.json`, all 158 tensors
//! are name/shape checked with no extras, and learned 80-bin `mean` / `scale`
//! normalization is applied before the shared generator. The independent
//! fixture runs the official Transformers forward.
//!
//! `hifigan_vocoder` is also strict and complete. Its offline prep sidecar
//! folds SpeechBrain's weight-normalized 234-tensor training checkpoint into
//! the exact 156 effective tensors consumed here. The binder additionally
//! preserves SpeechBrain's `inference_padding = 5` replicate-padding contract;
//! this is deliberately distinct from SpeechT5's learned normalization.
//!
//! # Fixture posture — [`HiFiGan::synthesized`]
//!
//! [`HiFiGan::synthesized`] is the **test-only** deterministic fixture
//! (same "zero-init scaffold" role as
//! `crates/vokra-models/src/vits_ja/mod.rs::VitsJaWeights::synthesized`)
//! that materialises a validated [`HifiGanWeights`] bundle whose shape
//! matches the caller's [`HifiGanAttrs`] — every conv / MRF branch is
//! zero-initialised (so the forward is deterministic but produces
//! near-zero audio via the terminal `tanh`), and the topology honours
//! [`HifiGanAttrs::res_block_type`] verbatim (V1 populates
//! `weight_c2 + bias_c2` on every layer per `mrf_branch_forward`'s
//! FR-EX-08 gate, V2 leaves them `None`). Fixtures never carry a
//! `cond` layer (`weights.cond = None`) — the single-speaker
//! unconditioned path — so [`HiFiGan::decode`] can exercise the op's
//! full forward without a per-utterance conditioning vector.

use vokra_core::gguf::GgufFile;
use vokra_core::gguf::chunks;
use vokra_core::{Result, VokraError};
use vokra_ops::attrs::{HifiGanAttrs, ResBlockType};
use vokra_ops::hifigan::{
    HifiGanConfig, HifiGanConvPadding, HifiGanWeights, MrfBranchWeights, ResBlockLayer,
    UpsampleStageWeights, hifigan_generator_with_conv_padding,
};

/// `vokra.model.arch` value emitted by the SpeechBrain LibriTTS 22.05 kHz
/// converter (`crates/vokra-convert/src/models/hifigan_vocoder.rs::ARCH`).
///
/// Pinned by the [`arch_tags_are_distinct_and_match_converters`] test —
/// a converter rename must land here in the same commit or the pin
/// fails loudly rather than silently mis-routing dispatch.
pub const ARCH_HIFIGAN_VOCODER: &str = "hifigan_vocoder";

/// `vokra.model.arch` value emitted by the Microsoft SpeechT5 16 kHz
/// converter (`crates/vokra-convert/src/models/speecht5_hifigan.rs::ARCH`).
///
/// Intentionally distinct from [`ARCH_HIFIGAN_VOCODER`] — the two GGUFs
/// walk different tensor trees (`generator.*` vs `upsampler.*` /
/// `resblocks.*` at top level, with `mean` / `scale` scalars for
/// `normalize_before = true` on the SpeechT5 side); silently sharing an
/// arch tag would mis-route runtime dispatch.
pub const ARCH_SPEECHT5_HIFIGAN: &str = "speecht5_hifigan";

/// Standalone HiFi-GAN vocoder handle: owns a [`HifiGanWeights`] +
/// [`HifiGanAttrs`] + [`HifiGanConfig`] triple plus the intended output
/// sample rate. Exposes [`decode`](Self::decode) as the primary
/// mel → PCM entry point.
///
/// # Sibling wrappers
///
/// `crates/vokra-models/src/sbv2/decoder.rs::SbV2Decoder` is the same
/// shape but attached to the SBV2 v2 flow-output-to-waveform bridge;
/// this binder is the standalone counterpart for the two HiFi-GAN
/// family GGUFs the sibling converters emit.
pub struct HiFiGan {
    /// Pre-built HiFi-GAN weight bundle (conv_pre / upsample stack /
    /// MRF branches / conv_post — see [`HifiGanWeights`]'s field docs).
    weights: HifiGanWeights,
    /// Shape + upsample-ladder metadata. [`decode`](Self::decode)'s
    /// `mel.len()` must equal `n_frames * attrs.n_mels`, checked inside
    /// [`vokra_ops::hifigan::hifigan_generator`] itself.
    attrs: HifiGanAttrs,
    /// Precision policy — FP32 (default) or mixed-precision FP16 with
    /// FP32 accumulator. INT8 opt-in stays gated behind
    /// [`HifiGanConfig::with_int8_opt_in`] + `spectral_check_passed`
    /// on the op boundary.
    config: HifiGanConfig,
    /// Output sample rate in Hz — informational metadata mirroring
    /// [`HifiGanAttrs::sample_rate`]; neither field feeds the forward
    /// math. Both exist so a caller can cross-check against the
    /// frontend_spec `sample_rate` (FR-LD-03). [`Self::new`] rejects
    /// a mismatched pair.
    sample_rate: u32,
    /// Optional per-mel normalization carried by SpeechT5 HiFi-GAN.
    /// The standalone SpeechBrain sibling has no equivalent tensors.
    normalization: Option<HifiGanInputNormalization>,
    /// Number of edge-replicated mel frames applied on both sides before the
    /// generator. SpeechBrain's public `inference()` contract uses 5;
    /// SpeechT5 feeds the unpadded mel directly.
    inference_padding: usize,
    /// Boundary mode for every stride-1 Conv1d. SpeechBrain's wrapper uses
    /// reflect; SpeechT5 and the public constructor retain zero padding.
    conv_padding: HifiGanConvPadding,
}

#[derive(Debug, Clone)]
struct HifiGanInputNormalization {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

fn normalize_hifigan_mel(
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    norm: &HifiGanInputNormalization,
) -> Result<Vec<f32>> {
    if mel.len() != n_mels * n_frames {
        return Err(VokraError::InvalidArgument(format!(
            "HiFiGan::decode: mel.len() {} != n_mels * n_frames = {} * {} = {}",
            mel.len(),
            n_mels,
            n_frames,
            n_mels * n_frames
        )));
    }
    let mut normalized = Vec::with_capacity(mel.len());
    for channel in 0..n_mels {
        let mean = norm.mean[channel];
        let scale = norm.scale[channel];
        let start = channel * n_frames;
        normalized.extend(
            mel[start..start + n_frames]
                .iter()
                .map(|value| (*value - mean) / scale),
        );
    }
    Ok(normalized)
}

fn replicate_pad_hifigan_mel(
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    if mel.len() != n_mels * n_frames {
        return Err(VokraError::InvalidArgument(format!(
            "HiFiGan::decode: mel.len() {} != n_mels * n_frames = {} * {} = {}",
            mel.len(),
            n_mels,
            n_frames,
            n_mels * n_frames
        )));
    }
    if n_frames == 0 {
        return Err(VokraError::InvalidArgument(
            "HiFiGan::decode: n_frames must be positive for replicate padding".to_owned(),
        ));
    }
    if padding == 0 {
        return Ok(mel.to_vec());
    }
    let padded_frames = n_frames + padding * 2;
    let mut padded = Vec::with_capacity(n_mels * padded_frames);
    for channel in 0..n_mels {
        let row = &mel[channel * n_frames..(channel + 1) * n_frames];
        padded.extend(std::iter::repeat_n(row[0], padding));
        padded.extend_from_slice(row);
        padded.extend(std::iter::repeat_n(row[n_frames - 1], padding));
    }
    Ok(padded)
}

impl HiFiGan {
    /// Assembles a vocoder handle from a pre-built weight bundle, its
    /// shape metadata, a precision policy, and the intended output
    /// sample rate.
    ///
    /// Runs [`HifiGanAttrs::validate_shape`] + [`HifiGanConfig::validate`]
    /// up front (SbV2Decoder / VitsJaTts::new precedent) so a
    /// mismatched pair fails loudly here rather than deep inside a
    /// forward — FR-EX-08. Cross-checks `sample_rate == attrs.sample_rate`
    /// because the two are expected to always agree; a converter that
    /// wrote one but not the other would silently produce audio at the
    /// wrong rate.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from
    ///   [`HifiGanAttrs::validate_shape`] on a structurally invalid attrs
    ///   bundle.
    /// - [`VokraError::InvalidArgument`] when
    ///   `sample_rate != attrs.sample_rate`.
    /// - [`VokraError::HifiganInt8VerifyMissing`] from
    ///   [`HifiGanConfig::validate`] when `int8_enabled == true` but the
    ///   calibration table or spectral-check verdict is missing.
    pub fn new(
        weights: HifiGanWeights,
        attrs: HifiGanAttrs,
        config: HifiGanConfig,
        sample_rate: u32,
    ) -> Result<Self> {
        Self::new_with_preprocessing(
            weights,
            attrs,
            config,
            sample_rate,
            None,
            0,
            HifiGanConvPadding::Zero,
        )
    }

    fn new_with_preprocessing(
        weights: HifiGanWeights,
        attrs: HifiGanAttrs,
        config: HifiGanConfig,
        sample_rate: u32,
        normalization: Option<HifiGanInputNormalization>,
        inference_padding: usize,
        conv_padding: HifiGanConvPadding,
    ) -> Result<Self> {
        attrs.validate_shape()?;
        config.validate()?;
        if sample_rate != attrs.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "HiFiGan::new: sample_rate {sample_rate} != attrs.sample_rate {}",
                attrs.sample_rate
            )));
        }
        if let Some(norm) = normalization.as_ref() {
            if norm.mean.len() != attrs.n_mels || norm.scale.len() != attrs.n_mels {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFiGan::new: normalization mean/scale lengths {}/{} != n_mels {}",
                    norm.mean.len(),
                    norm.scale.len(),
                    attrs.n_mels
                )));
            }
            if norm.mean.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(
                    "HiFiGan::new: normalization mean must be finite".to_owned(),
                ));
            }
            if norm
                .scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(VokraError::InvalidArgument(
                    "HiFiGan::new: normalization scale must be positive and finite".to_owned(),
                ));
            }
        }
        Ok(Self {
            weights,
            attrs,
            config,
            sample_rate,
            normalization,
            inference_padding,
            conv_padding,
        })
    }

    /// Output sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The shape-metadata bundle (n_mels, upsample_rates, MRF branches).
    #[must_use]
    pub fn attrs(&self) -> &HifiGanAttrs {
        &self.attrs
    }

    /// Deterministic zero-initialised fixture — **test scaffold only**.
    ///
    /// Materialises a validated [`HifiGanWeights`] bundle whose shape
    /// matches `attrs` verbatim, so [`Self::new`] + [`Self::decode`]
    /// can exercise the full forward path on a synthesised checkpoint
    /// without needing an actual upstream GGUF. Every conv / MRF-branch
    /// weight cell is zero, so the forward emits near-zero audio
    /// bounded by the terminal `tanh` — the test contract is finiteness
    /// and shape, not fidelity.
    ///
    /// Honours [`HifiGanAttrs::res_block_type`] verbatim: V1 populates
    /// `weight_c2 + bias_c2` on every MRF layer (per
    /// `mrf_branch_forward`'s FR-EX-08 gate: V1 attrs *must* have
    /// `weight_c2 + bias_c2` on every layer); V2 leaves them `None`.
    /// Fixtures never carry a `cond` (speaker conditioning) layer
    /// (`weights.cond = None`) so [`Self::decode`] exercises the
    /// single-speaker unconditioned path.
    ///
    /// The internal channel schedule mirrors
    /// `vokra_ops::hifigan::tests::tiny_weights` (`out_ch =
    /// max(3, in_ch / 2)` per stage) — proven to satisfy
    /// `validate_weights`'s cross-stage chain (each stage's
    /// `in_ch` == previous stage's `out_ch`) for any
    /// `validate_shape`-passing attrs. `conv_pre_kernel = 3` and
    /// `conv_post_kernel = 3` (test-fixture sizing — real
    /// [`hifigan_vocoder`](ARCH_HIFIGAN_VOCODER) checkpoints use
    /// kernel 7, but the fixture never rides a real
    /// [`from_gguf`](Self::from_gguf) path).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from
    /// [`HifiGanAttrs::validate_shape`] on a structurally invalid `attrs`
    /// (zero-axis dims, mismatched slice lengths, etc.) or when
    /// `sample_rate != attrs.sample_rate`.
    pub fn synthesized(attrs: HifiGanAttrs, sample_rate: u32) -> Result<Self> {
        attrs.validate_shape()?;
        if sample_rate != attrs.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "HiFiGan::synthesized: sample_rate {sample_rate} != attrs.sample_rate {}",
                attrs.sample_rate
            )));
        }

        let conv_pre_kernel = 3;
        let conv_post_kernel = 3;

        // conv_pre: [initial_channel, n_mels, conv_pre_kernel] zero-init.
        let conv_pre_weight = vec![0.0f32; attrs.initial_channel * attrs.n_mels * conv_pre_kernel];
        let conv_pre_bias = vec![0.0f32; attrs.initial_channel];

        // Walk the upsample stack in `validate_weights` chain order:
        // every stage's `in_ch` must equal the previous stage's
        // `out_ch`, with the first stage's `in_ch == initial_channel`.
        let mut upsample_weights = Vec::with_capacity(attrs.n_upsample_stages());
        let mut mrf_stage_weights = Vec::with_capacity(attrs.n_upsample_stages());
        let mut in_ch = attrs.initial_channel;
        for stage in 0..attrs.n_upsample_stages() {
            // Mirror the op-side tiny_weights schedule; the `max(3, ..)`
            // floor keeps every layer's channel count non-zero even
            // when a caller passes a tiny attrs like initial_channel = 2.
            let out_ch = 3.max(in_ch / 2);
            let kernel = attrs.upsample_kernel_sizes[stage];
            let stride = attrs.upsample_rates[stage];
            upsample_weights.push(UpsampleStageWeights {
                weight: vec![0.0f32; in_ch * out_ch * kernel],
                bias: vec![0.0f32; out_ch],
                in_ch,
                out_ch,
                kernel,
                stride,
            });

            let mut branches = Vec::with_capacity(attrs.n_mrf_branches());
            for branch_idx in 0..attrs.n_mrf_branches() {
                let branch_kernel = attrs.resblock_kernel_sizes[branch_idx];
                let dilations = &attrs.resblock_dilation_sizes[branch_idx];
                let mut layers = Vec::with_capacity(dilations.len());
                for &dilation in dilations {
                    let weight = vec![0.0f32; out_ch * out_ch * branch_kernel];
                    let bias = vec![0.0f32; out_ch];
                    let (weight_c2, bias_c2) = match attrs.res_block_type {
                        // V1 requires both c2 buffers populated per
                        // `mrf_branch_forward`'s FR-EX-08 gate — a
                        // missing c2 on a V1 attrs is loud-fail there.
                        ResBlockType::V1 => (
                            Some(vec![0.0f32; out_ch * out_ch * branch_kernel]),
                            Some(vec![0.0f32; out_ch]),
                        ),
                        // V2 mandates both `None` — carrying c2 on a V2
                        // attrs is also loud-fail per the same gate.
                        ResBlockType::V2 => (None, None),
                    };
                    layers.push(ResBlockLayer {
                        weight,
                        bias,
                        weight_c2,
                        bias_c2,
                        dilation,
                        kernel: branch_kernel,
                        channels: out_ch,
                    });
                }
                branches.push(MrfBranchWeights { layers });
            }
            mrf_stage_weights.push(branches);
            in_ch = out_ch;
        }

        // conv_post: [1, ch_last, conv_post_kernel]. Emit the pre-HGAN-04
        // explicit-zero bias shape `[1]` (validate_weights accepts
        // `len == 0` for upstream bias=False and `len == 1` for
        // the explicit-zero form; the two are numerically identical).
        let conv_post_weight = vec![0.0f32; in_ch * conv_post_kernel];
        let conv_post_bias = vec![0.0f32; 1];

        let weights = HifiGanWeights {
            conv_pre_weight,
            conv_pre_bias,
            conv_pre_kernel,
            upsample_weights,
            mrf_stage_weights,
            conv_post_weight,
            conv_post_bias,
            conv_post_kernel,
            cond: None,
        };

        Self::new(weights, attrs, HifiGanConfig::fp32(), sample_rate)
    }

    /// Runs the HiFi-GAN forward on `mel` (`[n_mels, n_frames]`
    /// row-major, `mel.len() == attrs.n_mels * n_frames`) and returns
    /// the raw PCM waveform bounded to `(−1, 1)` by the op's terminal
    /// `tanh`.
    ///
    /// This is a thin unconditioned wrapper — the multi-speaker `cond`
    /// path lives on [`hifigan_generator_conditioned`] and would
    /// require an additional API entry once a `cond`-carrying
    /// converter lands (SBV2 v2 already goes through the sibling
    /// `SbV2Decoder::generate_conditioned` for that).
    ///
    /// # Errors
    ///
    /// See [`vokra_ops::hifigan::hifigan_generator`]. In practice, once `self` has passed
    /// [`Self::new`], the only reachable errors are
    /// [`VokraError::InvalidArgument`] on a `mel.len()` mismatch
    /// (caller sent the wrong `n_frames`) and
    /// [`VokraError::UnsupportedOp`] when INT8 is opt-in-enabled (the
    /// INT8 forward kernel is deferred to the consumer WP per
    /// `hifigan_generator`'s own error surface).
    ///
    /// [`hifigan_generator_conditioned`]: vokra_ops::hifigan::hifigan_generator_conditioned
    pub fn decode(&self, mel: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        let normalized;
        let preprocessed = if let Some(norm) = self.normalization.as_ref() {
            normalized = normalize_hifigan_mel(mel, n_frames, self.attrs.n_mels, norm)?;
            normalized.as_slice()
        } else {
            mel
        };
        let padded;
        let (input, input_frames) = if self.inference_padding == 0 {
            (preprocessed, n_frames)
        } else {
            padded = replicate_pad_hifigan_mel(
                preprocessed,
                n_frames,
                self.attrs.n_mels,
                self.inference_padding,
            )?;
            (padded.as_slice(), n_frames + self.inference_padding * 2)
        };
        hifigan_generator_with_conv_padding(
            input,
            input_frames,
            &self.weights,
            &self.attrs,
            &self.config,
            self.conv_padding,
        )
    }

    /// Dispatches on the `vokra.model.arch` metadata chunk and loads a
    /// [`HiFiGan`] from a GGUF file.
    ///
    /// SpeechT5 HiFi-GAN binds its exact 158-tensor manifest and learned input
    /// normalization. SpeechBrain HiFi-GAN binds its exact 156-tensor folded
    /// manifest and applies its five-frame replicate padding.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is missing,
    ///   not a UTF-8 string, or does not match either supported arch
    ///   ([`ARCH_HIFIGAN_VOCODER`] or [`ARCH_SPEECHT5_HIFIGAN`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "HiFiGan::from_gguf: missing or non-string GGUF metadata key `{}` — the \
                     hifigan_vocoder / speecht5_hifigan converters both stamp this key; a GGUF \
                     without it is either not a HiFi-GAN vocoder or was produced by a converter \
                     that predates the arch-dispatch discipline. Rebuild via the appropriate \
                     `vokra-cli convert --model {{hifigan_vocoder,speecht5_hifigan}}` path.",
                    chunks::KEY_MODEL_ARCH
                ))
            })?;
        match arch {
            ARCH_HIFIGAN_VOCODER => load_speechbrain_hifigan(file),
            ARCH_SPEECHT5_HIFIGAN => load_speecht5_hifigan(file),
            other => Err(VokraError::ModelLoad(format!(
                "HiFiGan::from_gguf: unsupported `vokra.model.arch` = {other:?}. This binder \
                 accepts only {ARCH_HIFIGAN_VOCODER:?} (SpeechBrain LibriTTS 22.05 kHz) and \
                 {ARCH_SPEECHT5_HIFIGAN:?} (Microsoft SpeechT5 16 kHz). Other HiFi-GAN-family \
                 GGUFs (bigvgan, piper-plus, sbv2 decoders folded inside their consumer \
                 models) route through their own binder modules."
            ))),
        }
    }
}

fn speecht5_hifigan_attrs() -> HifiGanAttrs {
    HifiGanAttrs {
        n_mels: 80,
        initial_channel: 512,
        upsample_rates: vec![4, 4, 4, 4],
        upsample_kernel_sizes: vec![8, 8, 8, 8],
        resblock_kernel_sizes: vec![3, 7, 11],
        resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        sample_rate: 16_000,
        leaky_relu_slope: 0.1,
        res_block_type: ResBlockType::V1,
    }
}

fn speechbrain_hifigan_attrs() -> HifiGanAttrs {
    HifiGanAttrs {
        n_mels: 80,
        initial_channel: 512,
        upsample_rates: vec![8, 8, 2, 2],
        upsample_kernel_sizes: vec![16, 16, 4, 4],
        resblock_kernel_sizes: vec![3, 7, 11],
        resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        sample_rate: 22_050,
        leaky_relu_slope: 0.1,
        res_block_type: ResBlockType::V1,
    }
}

fn load_hifigan_tensor(
    file: &GgufFile,
    name: &str,
    expected_shape: &[usize],
    expected_names: &mut std::collections::BTreeSet<String>,
) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("HiFiGan: required tensor `{name}` is missing"))
    })?;
    let actual_shape: Vec<usize> = info.dimensions.iter().map(|&dim| dim as usize).collect();
    if actual_shape != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "HiFiGan: tensor `{name}` shape {actual_shape:?}, expected {expected_shape:?}"
        )));
    }
    expected_names.insert(name.to_owned());
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("HiFiGan: tensor `{name}` decode failed: {error}"))
    })
}

fn load_speechbrain_hifigan(file: &GgufFile) -> Result<HiFiGan> {
    use std::collections::BTreeSet;

    let attrs = speechbrain_hifigan_attrs();
    let mut expected_names = BTreeSet::new();
    let conv_pre_weight = load_hifigan_tensor(
        file,
        "conv_pre.weight",
        &[attrs.initial_channel, attrs.n_mels, 7],
        &mut expected_names,
    )?;
    let conv_pre_bias = load_hifigan_tensor(
        file,
        "conv_pre.bias",
        &[attrs.initial_channel],
        &mut expected_names,
    )?;

    let mut upsample_weights = Vec::with_capacity(attrs.n_upsample_stages());
    let mut mrf_stage_weights = Vec::with_capacity(attrs.n_upsample_stages());
    for stage in 0..attrs.n_upsample_stages() {
        let in_ch = attrs.initial_channel >> stage;
        let out_ch = attrs.initial_channel >> (stage + 1);
        let kernel = attrs.upsample_kernel_sizes[stage];
        upsample_weights.push(UpsampleStageWeights {
            weight: load_hifigan_tensor(
                file,
                &format!("ups.{stage}.weight"),
                &[in_ch, out_ch, kernel],
                &mut expected_names,
            )?,
            bias: load_hifigan_tensor(
                file,
                &format!("ups.{stage}.bias"),
                &[out_ch],
                &mut expected_names,
            )?,
            in_ch,
            out_ch,
            kernel,
            stride: attrs.upsample_rates[stage],
        });

        let mut branches = Vec::with_capacity(attrs.n_mrf_branches());
        for branch in 0..attrs.n_mrf_branches() {
            let block = stage * attrs.n_mrf_branches() + branch;
            let kernel = attrs.resblock_kernel_sizes[branch];
            let mut layers = Vec::with_capacity(attrs.resblock_dilation_sizes[branch].len());
            for (layer, &dilation) in attrs.resblock_dilation_sizes[branch].iter().enumerate() {
                let conv1 = format!("resblocks.{block}.convs1.{layer}");
                let conv2 = format!("resblocks.{block}.convs2.{layer}");
                layers.push(ResBlockLayer {
                    weight: load_hifigan_tensor(
                        file,
                        &format!("{conv1}.weight"),
                        &[out_ch, out_ch, kernel],
                        &mut expected_names,
                    )?,
                    bias: load_hifigan_tensor(
                        file,
                        &format!("{conv1}.bias"),
                        &[out_ch],
                        &mut expected_names,
                    )?,
                    weight_c2: Some(load_hifigan_tensor(
                        file,
                        &format!("{conv2}.weight"),
                        &[out_ch, out_ch, kernel],
                        &mut expected_names,
                    )?),
                    bias_c2: Some(load_hifigan_tensor(
                        file,
                        &format!("{conv2}.bias"),
                        &[out_ch],
                        &mut expected_names,
                    )?),
                    dilation,
                    kernel,
                    channels: out_ch,
                });
            }
            branches.push(MrfBranchWeights { layers });
        }
        mrf_stage_weights.push(branches);
    }

    let last_channels = attrs.initial_channel >> attrs.n_upsample_stages();
    let conv_post_weight = load_hifigan_tensor(
        file,
        "conv_post.weight",
        &[1, last_channels, 7],
        &mut expected_names,
    )?;
    let conv_post_bias = load_hifigan_tensor(file, "conv_post.bias", &[1], &mut expected_names)?;

    let actual_names: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|info| info.name.clone())
        .collect();
    if actual_names != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual_names).take(4).collect();
        let extra: Vec<&String> = actual_names.difference(&expected_names).take(4).collect();
        return Err(VokraError::ModelLoad(format!(
            "HiFiGan::from_gguf(hifigan_vocoder): tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual_names.len()
        )));
    }

    let weights = HifiGanWeights {
        conv_pre_weight,
        conv_pre_bias,
        conv_pre_kernel: 7,
        upsample_weights,
        mrf_stage_weights,
        conv_post_weight,
        conv_post_bias,
        conv_post_kernel: 7,
        cond: None,
    };
    HiFiGan::new_with_preprocessing(
        weights,
        attrs,
        HifiGanConfig::fp32(),
        22_050,
        None,
        5,
        HifiGanConvPadding::Reflect,
    )
    .map_err(|error| {
        VokraError::ModelLoad(format!(
            "HiFiGan::from_gguf(hifigan_vocoder): loaded tensor tree failed validation: {error}"
        ))
    })
}

fn load_speecht5_hifigan(file: &GgufFile) -> Result<HiFiGan> {
    use std::collections::BTreeSet;

    let attrs = speecht5_hifigan_attrs();
    let mut expected_names = BTreeSet::new();
    let conv_pre_weight = load_hifigan_tensor(
        file,
        "conv_pre.weight",
        &[attrs.initial_channel, attrs.n_mels, 7],
        &mut expected_names,
    )?;
    let conv_pre_bias = load_hifigan_tensor(
        file,
        "conv_pre.bias",
        &[attrs.initial_channel],
        &mut expected_names,
    )?;

    let mut upsample_weights = Vec::with_capacity(attrs.n_upsample_stages());
    let mut mrf_stage_weights = Vec::with_capacity(attrs.n_upsample_stages());
    for stage in 0..attrs.n_upsample_stages() {
        let in_ch = attrs.initial_channel >> stage;
        let out_ch = attrs.initial_channel >> (stage + 1);
        let kernel = attrs.upsample_kernel_sizes[stage];
        upsample_weights.push(UpsampleStageWeights {
            weight: load_hifigan_tensor(
                file,
                &format!("upsampler.{stage}.weight"),
                &[in_ch, out_ch, kernel],
                &mut expected_names,
            )?,
            bias: load_hifigan_tensor(
                file,
                &format!("upsampler.{stage}.bias"),
                &[out_ch],
                &mut expected_names,
            )?,
            in_ch,
            out_ch,
            kernel,
            stride: attrs.upsample_rates[stage],
        });

        let mut branches = Vec::with_capacity(attrs.n_mrf_branches());
        for branch in 0..attrs.n_mrf_branches() {
            let block = stage * attrs.n_mrf_branches() + branch;
            let kernel = attrs.resblock_kernel_sizes[branch];
            let mut layers = Vec::with_capacity(attrs.resblock_dilation_sizes[branch].len());
            for (layer, &dilation) in attrs.resblock_dilation_sizes[branch].iter().enumerate() {
                let conv1 = format!("resblocks.{block}.convs1.{layer}");
                let conv2 = format!("resblocks.{block}.convs2.{layer}");
                layers.push(ResBlockLayer {
                    weight: load_hifigan_tensor(
                        file,
                        &format!("{conv1}.weight"),
                        &[out_ch, out_ch, kernel],
                        &mut expected_names,
                    )?,
                    bias: load_hifigan_tensor(
                        file,
                        &format!("{conv1}.bias"),
                        &[out_ch],
                        &mut expected_names,
                    )?,
                    weight_c2: Some(load_hifigan_tensor(
                        file,
                        &format!("{conv2}.weight"),
                        &[out_ch, out_ch, kernel],
                        &mut expected_names,
                    )?),
                    bias_c2: Some(load_hifigan_tensor(
                        file,
                        &format!("{conv2}.bias"),
                        &[out_ch],
                        &mut expected_names,
                    )?),
                    dilation,
                    kernel,
                    channels: out_ch,
                });
            }
            branches.push(MrfBranchWeights { layers });
        }
        mrf_stage_weights.push(branches);
    }

    let last_channels = attrs.initial_channel >> attrs.n_upsample_stages();
    let conv_post_weight = load_hifigan_tensor(
        file,
        "conv_post.weight",
        &[1, last_channels, 7],
        &mut expected_names,
    )?;
    let conv_post_bias = load_hifigan_tensor(file, "conv_post.bias", &[1], &mut expected_names)?;
    let normalization = HifiGanInputNormalization {
        mean: load_hifigan_tensor(file, "mean", &[attrs.n_mels], &mut expected_names)?,
        scale: load_hifigan_tensor(file, "scale", &[attrs.n_mels], &mut expected_names)?,
    };

    let actual_names: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|info| info.name.clone())
        .collect();
    if actual_names != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual_names).take(4).collect();
        let extra: Vec<&String> = actual_names.difference(&expected_names).take(4).collect();
        return Err(VokraError::ModelLoad(format!(
            "HiFiGan::from_gguf(speecht5_hifigan): tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual_names.len()
        )));
    }

    let weights = HifiGanWeights {
        conv_pre_weight,
        conv_pre_bias,
        conv_pre_kernel: 7,
        upsample_weights,
        mrf_stage_weights,
        conv_post_weight,
        conv_post_bias,
        conv_post_kernel: 7,
        cond: None,
    };
    HiFiGan::new_with_preprocessing(
        weights,
        attrs,
        HifiGanConfig::fp32(),
        16_000,
        Some(normalization),
        0,
        HifiGanConvPadding::Zero,
    )
    .map_err(|error| {
        VokraError::ModelLoad(format!(
            "HiFiGan::from_gguf(speecht5_hifigan): loaded tensor tree failed validation: {error}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_ops::hifigan::{CalibrationTable, HifiGanPrecision};

    /// Tiny V1 attrs — canonical HiFi-GAN / SBV2 v2 topology (per-layer
    /// `weight_c2 + bias_c2`), small enough for a decoding smoke test.
    /// Mirrors the shape of `vokra_ops::hifigan::tests::tiny_attrs_v1`
    /// so the synthesized-weight schedule (`out_ch = max(3, in_ch/2)`)
    /// exactly matches the op-side fixture's chain expectations.
    fn tiny_attrs_v1() -> HifiGanAttrs {
        HifiGanAttrs {
            n_mels: 4,
            initial_channel: 6,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
            sample_rate: 16_000,
            leaky_relu_slope: 0.1,
            res_block_type: ResBlockType::V1,
        }
    }

    /// V2 counterpart (single-conv per layer, `weight_c2`/`bias_c2 =
    /// None`). Same shape as [`tiny_attrs_v1`], different topology tag.
    fn tiny_attrs_v2() -> HifiGanAttrs {
        HifiGanAttrs {
            res_block_type: ResBlockType::V2,
            ..tiny_attrs_v1()
        }
    }

    /// Task-spec pin: the arch tags this binder dispatches on MUST
    /// match verbatim the `ARCH` constants the sibling converters emit
    /// (`crates/vokra-convert/src/models/{hifigan_vocoder,
    /// speecht5_hifigan}.rs`). A converter rename that skipped this
    /// module would silently route to the unknown-arch error path
    /// instead of the deferred-loader loud path — this test catches
    /// that drift.
    #[test]
    fn arch_tags_are_distinct_and_match_converters() {
        assert_eq!(ARCH_HIFIGAN_VOCODER, "hifigan_vocoder");
        assert_eq!(ARCH_SPEECHT5_HIFIGAN, "speecht5_hifigan");
        assert_ne!(
            ARCH_HIFIGAN_VOCODER, ARCH_SPEECHT5_HIFIGAN,
            "the two vocoders walk different tensor trees — sharing an arch would mis-route"
        );
    }

    /// Primary integration test: a `synthesized(tiny_attrs_v1)`
    /// fixture decodes a zero-mel input to a finite, bounded PCM
    /// buffer of the shape `hifigan_generator`'s transposed-conv
    /// formula predicts. Exercises the full forward (conv_pre →
    /// per-stage transposed-conv + MRF branch fusion → conv_post →
    /// tanh) with a V1 topology's per-layer `c2` chain populated.
    #[test]
    fn synthesized_v1_decodes_zero_mel_to_finite_bounded_pcm() {
        let attrs = tiny_attrs_v1();
        let sr = attrs.sample_rate;
        let hifigan = HiFiGan::synthesized(attrs.clone(), sr).expect("synthesized V1");
        assert_eq!(hifigan.sample_rate(), sr);
        assert_eq!(hifigan.attrs().res_block_type, ResBlockType::V1);

        let n_frames = 4;
        let mel = vec![0.0f32; attrs.n_mels * n_frames];
        let pcm = hifigan.decode(&mel, n_frames).expect("decode zero mel");

        // Expected sample count from the transposed-conv shape formula
        // used by `hifigan_generator`'s per-stage upsample:
        //   out_len = (in_len - 1) * stride + kernel - 2 * ((kernel - stride) / 2)
        // For every stage in tiny_attrs_v1 (kernel=4, stride=2), padding
        // = (4-2)/2 = 1, so out_len = (in-1)*2 + 4 - 2 = in*2 exactly.
        // Chained over 2 stages: n_frames * 4. Compute explicitly rather
        // than hard-coding so a future tiny_attrs shape change does not
        // silently drift.
        let mut expected_len = n_frames;
        for stage in 0..attrs.n_upsample_stages() {
            let kernel = attrs.upsample_kernel_sizes[stage];
            let stride = attrs.upsample_rates[stage];
            let padding = kernel.saturating_sub(stride) / 2;
            expected_len = (expected_len - 1) * stride + kernel - 2 * padding;
        }
        assert_eq!(
            pcm.len(),
            expected_len,
            "PCM length matches transposed-conv formula"
        );

        for (i, &v) in pcm.iter().enumerate() {
            assert!(v.is_finite(), "PCM[{i}] = {v} must be finite (tanh output)");
            assert!(v > -1.0 && v < 1.0, "PCM[{i}] = {v} must lie in (-1, 1)");
        }
    }

    /// V2 counterpart of the primary integration test — same shape but
    /// single-conv per MRF layer (no c2 chain). Guards the
    /// `res_block_type` branch inside [`HiFiGan::synthesized`] against
    /// a regression that populates c2 unconditionally (which would
    /// loud-fail inside `mrf_branch_forward`'s V2 arm).
    #[test]
    fn synthesized_v2_decodes_zero_mel_to_finite_bounded_pcm() {
        let attrs = tiny_attrs_v2();
        let sr = attrs.sample_rate;
        let hifigan = HiFiGan::synthesized(attrs.clone(), sr).expect("synthesized V2");
        assert_eq!(hifigan.attrs().res_block_type, ResBlockType::V2);

        let n_frames = 4;
        let mel = vec![0.0f32; attrs.n_mels * n_frames];
        let pcm = hifigan.decode(&mel, n_frames).expect("decode zero mel V2");
        assert!(!pcm.is_empty(), "V2 forward must emit a non-empty waveform");
        for &v in &pcm {
            assert!(v.is_finite() && v > -1.0 && v < 1.0);
        }
    }

    /// Structural precondition: [`HiFiGan::synthesized`] must reject a
    /// zero-axis attrs (empty `upsample_rates`) via
    /// `HifiGanAttrs::validate_shape` before touching any weight
    /// allocation. FR-EX-08 — construction fails loudly, not silently.
    #[test]
    fn synthesized_rejects_zero_axis_attrs() {
        let mut attrs = tiny_attrs_v1();
        attrs.upsample_rates.clear();
        attrs.upsample_kernel_sizes.clear();
        let Err(err) = HiFiGan::synthesized(attrs, 16_000) else {
            panic!("expected InvalidArgument from validate_shape on zero-axis attrs");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument from validate_shape, got: {err}"
        );
    }

    /// [`HiFiGan::synthesized`] must reject a `sample_rate` that
    /// disagrees with `attrs.sample_rate`. The two are expected to
    /// always agree; a silent disagreement would produce audio at the
    /// wrong rate.
    #[test]
    fn synthesized_rejects_sample_rate_disagreement() {
        let attrs = tiny_attrs_v1();
        // attrs.sample_rate = 16_000; caller supplies 22_050 by mistake.
        let Err(err) = HiFiGan::synthesized(attrs, 22_050) else {
            panic!("expected InvalidArgument from sample_rate mismatch");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// [`HiFiGan::new`] must reject a config that manually flips
    /// `int8_enabled = true` without attaching a calibration table +
    /// spectral-check verdict, via [`HifiGanConfig::validate`]. The
    /// atomic constructor [`HifiGanConfig::with_int8_opt_in`] pairs
    /// both proofs; a hand-built config that skips it must fail loudly.
    #[test]
    fn new_rejects_broken_int8_opt_in() {
        let attrs = tiny_attrs_v1();
        let synth = HiFiGan::synthesized(attrs.clone(), attrs.sample_rate).unwrap();
        let broken = HifiGanConfig {
            precision: HifiGanPrecision::Fp32,
            int8_enabled: true,
            calibration_data: None,
            spectral_check_passed: true,
        };
        let Err(err) = HiFiGan::new(synth.weights, synth.attrs, broken, attrs.sample_rate) else {
            panic!("expected HifiganInt8VerifyMissing from broken INT8 opt-in config");
        };
        assert!(
            matches!(err, VokraError::HifiganInt8VerifyMissing),
            "expected HifiganInt8VerifyMissing from HifiGanConfig::validate, got: {err}"
        );
    }

    /// [`HiFiGan::new`] must accept an INT8 opt-in config that satisfies
    /// both proofs — validation passes at construction, but decoding
    /// still hits the deferred-kernel [`VokraError::UnsupportedOp`] on
    /// the op boundary (the M3-07 op-only WP does not ship the INT8
    /// forward kernel). Guards that the gate distinguishes "invalid
    /// config" (rejected at construction) from "valid config, kernel
    /// deferred" (rejected at forward).
    #[test]
    fn new_accepts_int8_opt_in_but_decode_returns_unsupported_kernel() {
        let attrs = tiny_attrs_v1();
        let synth = HiFiGan::synthesized(attrs.clone(), attrs.sample_rate).unwrap();
        let calibration =
            CalibrationTable::new(vec![1.0; 3], vec![0; 3], 3).expect("valid calibration");
        let cfg = HifiGanConfig::fp32().with_int8_opt_in(calibration, true);
        let hifigan = HiFiGan::new(synth.weights, synth.attrs, cfg, attrs.sample_rate)
            .expect("new must accept fully-authorised INT8 config");
        let n_frames = 2;
        let mel = vec![0.0f32; attrs.n_mels * n_frames];
        let err = hifigan.decode(&mel, n_frames).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp (INT8 kernel deferred to consumer WP), got: {err}"
        );
    }

    /// A GGUF that does not carry `vokra.model.arch` at all must fail
    /// with [`VokraError::ModelLoad`] — never a silent success on a
    /// zero-tensor fixture, never a panic.
    #[test]
    fn from_gguf_missing_arch_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.name", "no-arch-here");
        let bytes = b.to_bytes().expect("build minimal GGUF");
        let file = GgufFile::parse(bytes).expect("parse minimal GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming the missing arch key on unset arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(chunks::KEY_MODEL_ARCH));
            }
            other => panic!("expected ModelLoad naming the missing arch key, got: {other}"),
        }
    }

    /// A GGUF carrying an arch tag this binder does not recognise must
    /// fail with [`VokraError::ModelLoad`] naming both accepted arches
    /// so a downstream caller can pick the right converter.
    #[test]
    fn from_gguf_unknown_arch_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "bigvgan_v2");
        let bytes = b.to_bytes().expect("build GGUF with wrong arch");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming supported arches on unknown arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(ARCH_HIFIGAN_VOCODER));
                assert!(msg.contains(ARCH_SPEECHT5_HIFIGAN));
                assert!(msg.contains("bigvgan_v2"));
            }
            other => panic!("expected ModelLoad naming supported arches, got: {other}"),
        }
    }

    /// `arch = hifigan_vocoder` now enters the strict 156-tensor folded
    /// manifest walk. A metadata-only file must fail at the first tensor.
    #[test]
    fn from_gguf_hifigan_vocoder_requires_real_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_HIFIGAN_VOCODER);
        let bytes = b.to_bytes().expect("build hifigan_vocoder-arch GGUF");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("metadata-only SpeechBrain HiFi-GAN must not bind");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains("conv_pre.weight"));
                assert!(msg.contains("missing"));
            }
            other => panic!("expected strict tensor ModelLoad error, got: {other}"),
        }
    }

    /// `arch = speecht5_hifigan` now enters the strict 158-tensor walk.
    /// A metadata-only file must therefore fail on the first required
    /// tensor, not return the old deferred-loader error.
    #[test]
    fn from_gguf_speecht5_hifigan_requires_real_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_SPEECHT5_HIFIGAN);
        let bytes = b.to_bytes().expect("build speecht5_hifigan-arch GGUF");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("metadata-only SpeechT5 HiFi-GAN must not bind");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains("conv_pre.weight"));
                assert!(msg.contains("missing"));
            }
            other => panic!("expected strict tensor ModelLoad error, got: {other}"),
        }
    }

    #[test]
    fn speecht5_normalization_is_per_mel_channel_in_channel_major_layout() {
        let norm = HifiGanInputNormalization {
            mean: vec![1.0, -2.0],
            scale: vec![2.0, 4.0],
        };
        let normalized = normalize_hifigan_mel(&[1.0, 3.0, -2.0, 6.0], 2, 2, &norm)
            .expect("valid channel-major mel");
        assert_eq!(normalized, vec![0.0, 1.0, 0.0, 2.0]);

        let err = normalize_hifigan_mel(&[0.0; 3], 2, 2, &norm).unwrap_err();
        assert!(err.to_string().contains("mel.len() 3"));
    }

    #[test]
    fn speechbrain_replicate_padding_preserves_channel_major_rows() {
        let padded = replicate_pad_hifigan_mel(&[1.0, 2.0, 10.0, 20.0], 2, 2, 2)
            .expect("valid channel-major mel");
        assert_eq!(
            padded,
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0
            ]
        );
        assert!(
            replicate_pad_hifigan_mel(&[], 0, 2, 5)
                .unwrap_err()
                .to_string()
                .contains("n_frames must be positive")
        );
    }

    #[test]
    fn speechbrain_attrs_match_pinned_hyperparams() {
        let attrs = speechbrain_hifigan_attrs();
        assert_eq!(attrs.n_mels, 80);
        assert_eq!(attrs.initial_channel, 512);
        assert_eq!(attrs.upsample_rates, [8, 8, 2, 2]);
        assert_eq!(attrs.upsample_kernel_sizes, [16, 16, 4, 4]);
        assert_eq!(attrs.total_upsample_factor(), 256);
        assert_eq!(attrs.sample_rate, 22_050);
    }
}
