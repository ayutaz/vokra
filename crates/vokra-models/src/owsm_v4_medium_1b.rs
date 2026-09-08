//! Strict GGUF contract for ESPnet OWSM v4 medium 1B.
//!
//! The VAST inspection packet at the pinned ESPnet/Hugging Face revisions
//! authenticates 1,172 F32 tensors and the composite frontend/encoder/decoder
//! topology.  This module binds that evidence without guessing a tensor name
//! or axis.  The PCM frontend, autoregressive decoder, SentencePiece bridge,
//! and independent numerical parity are deliberately still explicit
//! `NotImplemented` blockers; a manifest-only bind must never masquerade as a
//! working ASR engine.

use std::collections::BTreeSet;
use std::path::Path;

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::ir::graph::{PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::{AsrEngine, BackendKind, Result, Transcription, VokraError};
use vokra_ops::stft;

/// GGUF architecture tag written by the reviewed OWSM converter.
pub const ARCH: &str = "owsm-v4-medium-1b";
/// Human-readable model name.
pub const NAME: &str = "OWSM v4 medium 1B";
/// Fixed Hugging Face repository and revision authenticated by VAST.
pub const HF_REPOSITORY: &str = "espnet/owsm_v4_medium_1B";
/// Full-length immutable Hugging Face revision used by the inspected source tree.
pub const HF_REVISION: &str = "e10985c8f1d592e905c24d2ac2b2c53e3feb24dc";
/// Fixed ESPnet source revision authenticated by VAST.
pub const SOURCE_REVISION: &str = "cccc29023d43a3f504e28df7d1324bb4eb6daedd";
/// Original PyTorch checkpoint identity from the inspection packet.
pub const CHECKPOINT_SHA256: &str =
    "b02d79f29a4daa31dd49ce145d9bb4cda0a1b68cdad91ae0af170ec3a4e92e09";
/// Number of checkpoint tensors in the authenticated structural inventory.
pub const CHECKPOINT_TENSOR_COUNT: usize = 1172;
/// SHA-256 of the complete VAST inspection manifest JSON.
pub const INSPECTION_MANIFEST_SHA256: &str =
    "82de20eea3cf3a247624c76cd8e108e562addda0c8582577515cf88abb3053d9";
/// SHA-256 of the VAST inspection validation log.
pub const INSPECTION_LOG_SHA256: &str =
    "4df29428ea8ce381311c5e407d937b6a517750f4edcbc88b8c606cdef82dc93b";
/// SHA-256 of the fixed BPE sidecar evidence recorded by inspection.
pub const BPE_SHA256: &str = "7ddb01f03dab493c18ab69391e98744c090f897890d8b529b30cae52a8d9eef4";
/// SHA-256 of the fixed global-MVN statistics sidecar evidence.
pub const STATS_SHA256: &str = "00c22dba27594df8f1d8f74a491b20c6e6e8c17e92159f81dfd634f98c098654";
/// SHA-256 of the fixed 50,002-entry token-list sidecar evidence.
pub const TOKEN_LIST_SHA256: &str =
    "e19396ec012b0294a11fe85c35e36a1d903bc83e60ea602ddf6cc59b7c0e92f9";

const KEY_SAMPLE_RATE: &str = "vokra.owsm_v4_medium_1b.sample_rate";
const KEY_N_FFT: &str = "vokra.owsm_v4_medium_1b.frontend.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.owsm_v4_medium_1b.frontend.win_length";
const KEY_HOP_LENGTH: &str = "vokra.owsm_v4_medium_1b.frontend.hop_length";
const KEY_N_MELS: &str = "vokra.owsm_v4_medium_1b.frontend.n_mels";
const KEY_ENCODER_LAYERS: &str = "vokra.owsm_v4_medium_1b.encoder.n_layers";
const KEY_ENCODER_DIM: &str = "vokra.owsm_v4_medium_1b.encoder.d_model";
const KEY_ENCODER_HEADS: &str = "vokra.owsm_v4_medium_1b.encoder.n_heads";
const KEY_ENCODER_FFN: &str = "vokra.owsm_v4_medium_1b.encoder.ffn_dim";
const KEY_ENCODER_CGMLP: &str = "vokra.owsm_v4_medium_1b.encoder.cgmlp_dim";
const KEY_ENCODER_CGMLP_KERNEL: &str = "vokra.owsm_v4_medium_1b.encoder.cgmlp_kernel";
const KEY_ENCODER_MERGE_KERNEL: &str = "vokra.owsm_v4_medium_1b.encoder.merge_kernel";
const KEY_DECODER_LAYERS: &str = "vokra.owsm_v4_medium_1b.decoder.n_layers";
const KEY_DECODER_DIM: &str = "vokra.owsm_v4_medium_1b.decoder.d_model";
const KEY_DECODER_HEADS: &str = "vokra.owsm_v4_medium_1b.decoder.n_heads";
const KEY_DECODER_FFN: &str = "vokra.owsm_v4_medium_1b.decoder.ffn_dim";
const KEY_VOCAB_SIZE: &str = "vokra.owsm_v4_medium_1b.vocab_size";
const KEY_REVISION: &str = "vokra.owsm_v4_medium_1b.revision";
const KEY_SOURCE_REVISION: &str = "vokra.owsm_v4_medium_1b.source_revision";
const KEY_CHECKPOINT_SHA256: &str = "vokra.owsm_v4_medium_1b.checkpoint_sha256";
const KEY_BPE_SHA256: &str = "vokra.owsm_v4_medium_1b.bpe_sha256";
const KEY_STATS_SHA256: &str = "vokra.owsm_v4_medium_1b.stats_sha256";
const KEY_TOKEN_LIST_SHA256: &str = "vokra.owsm_v4_medium_1b.token_list_sha256";
const KEY_INSPECTION_MANIFEST_SHA256: &str = "vokra.owsm_v4_medium_1b.inspection_manifest_sha256";
const KEY_INSPECTION_LOG_SHA256: &str = "vokra.owsm_v4_medium_1b.inspection_log_sha256";

/// The fixed source-level log floor in ESPnet's `LogMel` layer.
///
/// This is a source contract, not a tolerance or a substitute for a missing
/// tensor.  The independent source inspector authenticates the clamp/log
/// expressions before this runtime path is enabled.
pub const LOG_MEL_FLOOR: f32 = 1.0e-10;
/// Epsilon used by ESPnet's `GlobalMVN` when deriving standard deviation.
pub const GLOBAL_MVN_EPS: f32 = 1.0e-20;

const FRONTEND_MELMAT: &str = "frontend.logmel.melmat";
const NORMALIZE_MEAN: &str = "normalize.mean";
const NORMALIZE_STD: &str = "normalize.std";

/// The source-authenticated STFT contract for OWSM's default ESPnet frontend.
///
/// Values here mirror the fixed `DefaultFrontend`/`Stft` source API and the
/// OWSM config metadata.  No caller may override padding, centering, window,
/// normalization, or spectrum side: doing so would be a different frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwsmV4Medium1bFrontendConfig {
    /// FFT size in samples.
    pub n_fft: usize,
    /// Window size in samples.
    pub win_length: usize,
    /// Frame hop in samples.
    pub hop_length: usize,
    /// Number of one-sided STFT frequency bins in the mel matrix input.
    pub n_freqs: usize,
    /// Number of mel bands in each frontend output frame.
    pub n_mels: usize,
}

impl OwsmV4Medium1bFrontendConfig {
    fn from_model_config(config: &OwsmV4Medium1bConfig) -> Self {
        Self {
            n_fft: config.n_fft as usize,
            win_length: config.win_length as usize,
            hop_length: config.hop_length as usize,
            n_freqs: config.n_fft as usize / 2 + 1,
            n_mels: config.n_mels as usize,
        }
    }

    fn stft_attrs(&self) -> StftAttrs {
        let mut attrs = StftAttrs::new(self.n_fft, self.hop_length);
        attrs.win_length = self.win_length;
        attrs.window = Window::Hann;
        attrs.window_symmetry = WindowSymmetry::Periodic;
        attrs.center = true;
        attrs.pad_mode = PadMode::Reflect;
        attrs.causal = false;
        attrs.real_input = true;
        attrs
    }
}

/// PCM frontend output in the row-major `[frames, n_mels]` layout expected by
/// ESPnet's E-Branchformer input projection.
#[derive(Debug, Clone, PartialEq)]
pub struct OwsmV4Medium1bFrontendOutput {
    /// Global-MVN normalized features, row-major `[frames, n_mels]`.
    pub features: Vec<f32>,
    /// Number of valid frames in `features`.
    pub frames: usize,
    /// Number of mel channels in each frame.
    pub n_mels: usize,
}

/// Strictly bound frontend tensors from the existing GGUF artifact.
///
/// The frontend owns no fallback mel construction: `melmat`, `mean`, and
/// `std` must all be present with the authenticated shapes and F32 payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct OwsmV4Medium1bFrontend {
    config: OwsmV4Medium1bFrontendConfig,
    melmat: Vec<f32>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl OwsmV4Medium1bFrontend {
    /// Binds the model's existing frontend and GlobalMVN payloads.
    pub fn from_gguf(file: &GgufFile, model_config: &OwsmV4Medium1bConfig) -> Result<Self> {
        let config = OwsmV4Medium1bFrontendConfig::from_model_config(model_config);
        let melmat = frontend_tensor(file, FRONTEND_MELMAT, &[config.n_freqs, config.n_mels])?;
        let mean = frontend_tensor(file, NORMALIZE_MEAN, &[config.n_mels])?;
        let std = frontend_tensor(file, NORMALIZE_STD, &[config.n_mels])?;
        validate_frontend_values(&melmat, &mean, &std, config.n_freqs, config.n_mels)?;
        Ok(Self {
            config,
            melmat,
            mean,
            std,
        })
    }

    /// Returns the fixed frontend dimensions.
    pub fn config(&self) -> &OwsmV4Medium1bFrontendConfig {
        &self.config
    }

    /// Computes STFT → power → bound mel matrix → ESPnet LogMel → GlobalMVN.
    ///
    /// This method handles one finite mono PCM sequence and returns only its
    /// valid frames.  It deliberately does not pad a batch, resample audio,
    /// apply SpecAugment, or run the OWSM encoder/decoder.
    pub fn extract(&self, pcm: &[f32]) -> Result<OwsmV4Medium1bFrontendOutput> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "OWSM v4 medium 1B frontend: PCM must not be empty".to_owned(),
            ));
        }
        if pcm.len() <= self.config.n_fft / 2 {
            return Err(VokraError::InvalidArgument(
                "OWSM v4 medium 1B frontend: centered reflect padding requires PCM longer than n_fft/2"
                    .to_owned(),
            ));
        }
        if pcm.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "OWSM v4 medium 1B frontend: PCM contains a non-finite sample".to_owned(),
            ));
        }

        let spectrum = stft(pcm, &self.config.stft_attrs())?;
        // The pinned ESPnet `Stft` source adds `2 * (n_fft / 2)` samples when
        // `center` is enabled, then computes
        // `(ilens - n_fft) // hop_length + 1`.  Validate the backend result
        // against that authenticated length contract instead of accepting a
        // silently different padding/frame convention.
        let expected_frames = pcm
            .len()
            .checked_add(self.config.n_fft)
            .and_then(|length| length.checked_sub(self.config.n_fft))
            .and_then(|length| length.checked_div(self.config.hop_length))
            .and_then(|frames| frames.checked_add(1))
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "OWSM v4 medium 1B frontend: source frame-length calculation overflowed"
                        .to_owned(),
                )
            })?;
        if spectrum.frames != expected_frames {
            return Err(VokraError::ModelLoad(format!(
                "OWSM v4 medium 1B frontend: STFT produced {} frames, expected {} from the authenticated centered source formula",
                spectrum.frames, expected_frames
            )));
        }
        let power = spectrum.power();
        let expected_power = spectrum.frames * self.config.n_freqs;
        if spectrum.bins != self.config.n_freqs || power.len() != expected_power {
            return Err(VokraError::ModelLoad(format!(
                "OWSM v4 medium 1B frontend: STFT produced {} bins and {} values, expected {} bins and {} values",
                spectrum.bins,
                power.len(),
                self.config.n_freqs,
                expected_power
            )));
        }

        let mut features = vec![0.0f32; spectrum.frames * self.config.n_mels];
        for frame in 0..spectrum.frames {
            let power_row = &power[frame * self.config.n_freqs..(frame + 1) * self.config.n_freqs];
            let feature_row =
                &mut features[frame * self.config.n_mels..(frame + 1) * self.config.n_mels];
            for mel in 0..self.config.n_mels {
                let mut energy = 0.0f32;
                for (frequency, &power_value) in power_row.iter().enumerate() {
                    energy += power_value * self.melmat[frequency * self.config.n_mels + mel];
                }
                // ESPnet LogMel clamps before applying natural logarithm.
                feature_row[mel] = energy.max(LOG_MEL_FLOOR).ln();
            }
            for (mel, value) in feature_row.iter_mut().enumerate() {
                *value = (*value - self.mean[mel]) / self.std[mel];
            }
        }

        Ok(OwsmV4Medium1bFrontendOutput {
            features,
            frames: spectrum.frames,
            n_mels: self.config.n_mels,
        })
    }
}

fn validate_frontend_values(
    melmat: &[f32],
    mean: &[f32],
    std: &[f32],
    n_freqs: usize,
    n_mels: usize,
) -> Result<()> {
    if melmat
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !melmat.iter().any(|value| *value > 0.0)
        || mean.iter().any(|value| !value.is_finite())
        || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(VokraError::ModelLoad(
            "OWSM v4 medium 1B: frontend mel matrix must be finite, non-negative, and non-zero; GlobalMVN tensors must be finite with positive std"
                .to_owned(),
        ));
    }
    if n_freqs == 0 || n_mels == 0 || melmat.len() != n_freqs * n_mels {
        return Err(VokraError::ModelLoad(
            "OWSM v4 medium 1B: frontend mel matrix dimensions are inconsistent".to_owned(),
        ));
    }
    for mel in 0..n_mels {
        if !(0..n_freqs).any(|frequency| melmat[frequency * n_mels + mel] > 0.0) {
            return Err(VokraError::ModelLoad(format!(
                "OWSM v4 medium 1B: frontend mel column {mel} has no positive filter weight"
            )));
        }
    }
    Ok(())
}

fn frontend_tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "OWSM v4 medium 1B frontend: required tensor `{name}` is missing"
        ))
    })?;
    let shape: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect();
    if shape != expected {
        return Err(VokraError::ModelLoad(format!(
            "OWSM v4 medium 1B frontend: tensor `{name}` shape {shape:?}, expected {expected:?}"
        )));
    }
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "OWSM v4 medium 1B frontend: tensor `{name}` is {:?}, expected F32",
            info.dtype
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "OWSM v4 medium 1B frontend: tensor `{name}` decode failed: {error}"
        ))
    })
}

/// Fixed topology read from converter-stamped GGUF metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwsmV4Medium1bConfig {
    /// Input audio sample rate in hertz; the authenticated value is 16,000.
    pub sample_rate: u32,
    /// Short-time Fourier transform size in input samples; authenticated as 512.
    pub n_fft: u32,
    /// Analysis-window length in input samples; authenticated as 400.
    pub win_length: u32,
    /// Analysis-hop length in input samples; authenticated as 160.
    pub hop_length: u32,
    /// Number of log-mel feature bands; authenticated as 128.
    pub n_mels: u32,
    /// Number of Transformer encoder blocks; authenticated as 18.
    pub encoder_layers: u32,
    /// Encoder hidden width; authenticated as 1,024.
    pub encoder_dim: u32,
    /// Number of attention heads in each encoder block; authenticated as 16.
    pub encoder_heads: u32,
    /// Encoder feed-forward intermediate width; authenticated as 4,096.
    pub encoder_ffn: u32,
    /// Encoder convolutional GLU intermediate width; authenticated as 4,096.
    pub encoder_cgmlp_dim: u32,
    /// Encoder convolutional GLU kernel width in feature steps; authenticated as 31.
    pub encoder_cgmlp_kernel: u32,
    /// Encoder fusion kernel width in feature steps; authenticated as 31.
    pub encoder_merge_kernel: u32,
    /// Number of Transformer decoder blocks; authenticated as 18.
    pub decoder_layers: u32,
    /// Decoder hidden width; authenticated as 1,024.
    pub decoder_dim: u32,
    /// Number of attention heads in each decoder block; authenticated as 16.
    pub decoder_heads: u32,
    /// Decoder feed-forward intermediate width; authenticated as 4,096.
    pub decoder_ffn: u32,
    /// Decoder/CTC vocabulary cardinality; authenticated as 50,002 entries.
    pub vocab_size: u32,
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "OWSM v4 medium 1B: missing or non-u32 metadata `{key}`"
        ))),
    }
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("OWSM v4 medium 1B: missing metadata `{key}`"))
        })
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let found = required_string(file, key)?;
    if found != expected {
        return Err(VokraError::ModelLoad(format!(
            "OWSM v4 medium 1B: metadata `{key}` is `{found}`, expected `{expected}`"
        )));
    }
    Ok(())
}

impl OwsmV4Medium1bConfig {
    /// Reads all required OWSM axes. No primary-source fallback is used.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str())
            != Some(ARCH)
        {
            return Err(VokraError::ModelLoad(format!(
                "OWSM v4 medium 1B: expected `{}` in `{}`, refusing misroute",
                ARCH,
                chunks::KEY_MODEL_ARCH
            )));
        }
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            "attribution-required",
        )?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "CC-BY-4.0")?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, ARCH)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_SOURCE,
            "https://huggingface.co/espnet/owsm_v4_medium_1B",
        )?;
        require_string(file, KEY_REVISION, HF_REVISION)?;
        require_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
        require_string(file, KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)?;
        require_string(file, KEY_BPE_SHA256, BPE_SHA256)?;
        require_string(file, KEY_STATS_SHA256, STATS_SHA256)?;
        require_string(file, KEY_TOKEN_LIST_SHA256, TOKEN_LIST_SHA256)?;
        require_string(
            file,
            KEY_INSPECTION_MANIFEST_SHA256,
            INSPECTION_MANIFEST_SHA256,
        )?;
        require_string(file, KEY_INSPECTION_LOG_SHA256, INSPECTION_LOG_SHA256)?;

        let config = Self {
            sample_rate: required_u32(file, KEY_SAMPLE_RATE)?,
            n_fft: required_u32(file, KEY_N_FFT)?,
            win_length: required_u32(file, KEY_WIN_LENGTH)?,
            hop_length: required_u32(file, KEY_HOP_LENGTH)?,
            n_mels: required_u32(file, KEY_N_MELS)?,
            encoder_layers: required_u32(file, KEY_ENCODER_LAYERS)?,
            encoder_dim: required_u32(file, KEY_ENCODER_DIM)?,
            encoder_heads: required_u32(file, KEY_ENCODER_HEADS)?,
            encoder_ffn: required_u32(file, KEY_ENCODER_FFN)?,
            encoder_cgmlp_dim: required_u32(file, KEY_ENCODER_CGMLP)?,
            encoder_cgmlp_kernel: required_u32(file, KEY_ENCODER_CGMLP_KERNEL)?,
            encoder_merge_kernel: required_u32(file, KEY_ENCODER_MERGE_KERNEL)?,
            decoder_layers: required_u32(file, KEY_DECODER_LAYERS)?,
            decoder_dim: required_u32(file, KEY_DECODER_DIM)?,
            decoder_heads: required_u32(file, KEY_DECODER_HEADS)?,
            decoder_ffn: required_u32(file, KEY_DECODER_FFN)?,
            vocab_size: required_u32(file, KEY_VOCAB_SIZE)?,
        };
        if config.sample_rate != 16_000
            || config.n_fft != 512
            || config.win_length != 400
            || config.hop_length != 160
            || config.n_mels != 128
            || config.encoder_layers != 18
            || config.encoder_dim != 1024
            || config.encoder_heads != 16
            || config.encoder_ffn != 4096
            || config.encoder_cgmlp_dim != 4096
            || config.encoder_cgmlp_kernel != 31
            || config.encoder_merge_kernel != 31
            || config.decoder_layers != 18
            || config.decoder_dim != 1024
            || config.decoder_heads != 16
            || config.decoder_ffn != 4096
            || config.vocab_size != 50_002
        {
            return Err(VokraError::ModelLoad(
                "OWSM v4 medium 1B: metadata axes do not match the authenticated VAST config"
                    .to_owned(),
            ));
        }
        Ok(config)
    }
}

fn push_pair(manifest: &mut Vec<(String, Vec<u64>)>, prefix: &str, name: &str, shape: &[u64]) {
    manifest.push((format!("{prefix}.{name}.weight"), shape.to_vec()));
    manifest.push((format!("{prefix}.{name}.bias"), vec![shape[0]]));
}

/// Exact 1,172-entry F32 inventory derived from the authenticated VAST packet.
pub fn expected_tensor_manifest() -> Vec<(String, Vec<u64>)> {
    let mut manifest = Vec::with_capacity(CHECKPOINT_TENSOR_COUNT);
    manifest.push(("frontend.logmel.melmat".to_owned(), vec![257, 128]));
    manifest.push(("normalize.mean".to_owned(), vec![128]));
    manifest.push(("normalize.std".to_owned(), vec![128]));
    manifest.push((
        "encoder.embed.conv.0.weight".to_owned(),
        vec![1024, 1, 3, 3],
    ));
    manifest.push(("encoder.embed.conv.0.bias".to_owned(), vec![1024]));
    manifest.push((
        "encoder.embed.conv.2.weight".to_owned(),
        vec![1024, 1024, 3, 3],
    ));
    manifest.push(("encoder.embed.conv.2.bias".to_owned(), vec![1024]));
    manifest.push((
        "encoder.embed.conv.4.weight".to_owned(),
        vec![1024, 1024, 3, 3],
    ));
    manifest.push(("encoder.embed.conv.4.bias".to_owned(), vec![1024]));
    manifest.push(("encoder.embed.out.weight".to_owned(), vec![1024, 15_360]));
    manifest.push(("encoder.embed.out.bias".to_owned(), vec![1024]));
    for layer in 0..18 {
        let prefix = format!("encoder.encoders.{layer}");
        for projection in [
            "attn.linear_q",
            "attn.linear_k",
            "attn.linear_v",
            "attn.linear_out",
        ] {
            push_pair(&mut manifest, &prefix, projection, &[1024, 1024]);
        }
        push_pair(
            &mut manifest,
            &prefix,
            "cgmlp.channel_proj1.0",
            &[4096, 1024],
        );
        manifest.push((format!("{prefix}.cgmlp.csgu.norm.weight"), vec![2048]));
        manifest.push((format!("{prefix}.cgmlp.csgu.norm.bias"), vec![2048]));
        manifest.push((
            format!("{prefix}.cgmlp.csgu.conv.weight"),
            vec![2048, 1, 31],
        ));
        manifest.push((format!("{prefix}.cgmlp.csgu.conv.bias"), vec![2048]));
        push_pair(&mut manifest, &prefix, "cgmlp.channel_proj2", &[1024, 2048]);
        for branch in ["feed_forward", "feed_forward_macaron"] {
            push_pair(
                &mut manifest,
                &prefix,
                &format!("{branch}.w_1"),
                &[4096, 1024],
            );
            push_pair(
                &mut manifest,
                &prefix,
                &format!("{branch}.w_2"),
                &[1024, 4096],
            );
        }
        for norm in [
            "norm_ff",
            "norm_ff_macaron",
            "norm_mha",
            "norm_mlp",
            "norm_final",
        ] {
            manifest.push((format!("{prefix}.{norm}.weight"), vec![1024]));
            manifest.push((format!("{prefix}.{norm}.bias"), vec![1024]));
        }
        manifest.push((
            format!("{prefix}.depthwise_conv_fusion.weight"),
            vec![2048, 1, 31],
        ));
        manifest.push((format!("{prefix}.depthwise_conv_fusion.bias"), vec![2048]));
        push_pair(&mut manifest, &prefix, "merge_proj", &[1024, 2048]);
    }
    manifest.push(("encoder.after_norm.weight".to_owned(), vec![1024]));
    manifest.push(("encoder.after_norm.bias".to_owned(), vec![1024]));
    manifest.push(("decoder.embed.0.weight".to_owned(), vec![50_002, 1024]));
    manifest.push(("decoder.after_norm.weight".to_owned(), vec![1024]));
    manifest.push(("decoder.after_norm.bias".to_owned(), vec![1024]));
    manifest.push(("decoder.output_layer.weight".to_owned(), vec![50_002, 1024]));
    manifest.push(("decoder.output_layer.bias".to_owned(), vec![50_002]));
    for layer in 0..18 {
        let prefix = format!("decoder.decoders.{layer}");
        for attention in ["self_attn", "src_attn"] {
            for projection in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                push_pair(
                    &mut manifest,
                    &prefix,
                    &format!("{attention}.{projection}"),
                    &[1024, 1024],
                );
            }
        }
        push_pair(&mut manifest, &prefix, "feed_forward.w_1", &[4096, 1024]);
        push_pair(&mut manifest, &prefix, "feed_forward.w_2", &[1024, 4096]);
        for norm in ["norm1", "norm2", "norm3"] {
            manifest.push((format!("{prefix}.{norm}.weight"), vec![1024]));
            manifest.push((format!("{prefix}.{norm}.bias"), vec![1024]));
        }
    }
    manifest.push(("ctc.ctc_lo.weight".to_owned(), vec![50_002, 1024]));
    manifest.push(("ctc.ctc_lo.bias".to_owned(), vec![50_002]));
    manifest
}

/// Strictly bound the full checkpoint tensor inventory without copying it.
#[derive(Debug, Clone)]
pub struct OwsmV4Medium1bWeights {
    tensor_names: BTreeSet<String>,
}

impl OwsmV4Medium1bWeights {
    /// Validates the exact 1,172-name F32 structural inventory in a GGUF file.
    ///
    /// This binds names, shapes, dtypes, and count only; it does not claim
    /// independent per-payload hash or numerical-forward authentication.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let expected = expected_tensor_manifest();
        if expected.len() != CHECKPOINT_TENSOR_COUNT {
            return Err(VokraError::ModelLoad(
                "OWSM v4 medium 1B: internal manifest count is inconsistent".to_owned(),
            ));
        }
        if file.tensors().len() != CHECKPOINT_TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "OWSM v4 medium 1B: expected {CHECKPOINT_TENSOR_COUNT} tensors, found {}",
                file.tensors().len()
            )));
        }
        let mut names = BTreeSet::new();
        for info in file.tensors() {
            if info.dtype != GgmlType::F32 {
                return Err(VokraError::ModelLoad(format!(
                    "OWSM v4 medium 1B: tensor `{}` is {:?}, expected F32",
                    info.name, info.dtype
                )));
            }
            if !names.insert(info.name.clone()) {
                return Err(VokraError::ModelLoad(format!(
                    "OWSM v4 medium 1B: duplicate tensor `{}`",
                    info.name
                )));
            }
        }
        let expected_names: BTreeSet<String> =
            expected.iter().map(|(name, _)| name.clone()).collect();
        if names != expected_names {
            return Err(VokraError::ModelLoad(
                "OWSM v4 medium 1B: tensor names do not match the authenticated inventory"
                    .to_owned(),
            ));
        }
        for (name, shape) in expected {
            let Some(found) = file.tensor_info(&name) else {
                return Err(VokraError::ModelLoad(format!(
                    "OWSM v4 medium 1B: tensor `{name}` disappeared during validation"
                )));
            };
            if found.dimensions != shape {
                return Err(VokraError::ModelLoad(format!(
                    "OWSM v4 medium 1B: tensor `{name}` has shape {:?}, expected {shape:?}",
                    found.dimensions
                )));
            }
        }
        Ok(Self {
            tensor_names: names,
        })
    }

    /// Returns the number of structurally validated tensors in this handle.
    pub fn tensor_count(&self) -> usize {
        self.tensor_names.len()
    }
}

/// Strict manifest-bound model handle. Forward is intentionally blocked.
#[derive(Debug, Clone)]
pub struct OwsmV4Medium1b {
    config: OwsmV4Medium1bConfig,
    weights: OwsmV4Medium1bWeights,
    frontend: OwsmV4Medium1bFrontend,
}

impl OwsmV4Medium1b {
    /// Opens a GGUF path and applies the strict OWSM metadata/inventory gate.
    ///
    /// No PCM input is executed while binding; transcription remains blocked
    /// until the independent VAST execution contract is complete.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let file =
            GgufFile::open(path).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
        Self::from_file(&file)
    }

    /// Applies the strict OWSM license, metadata, name, shape, and dtype gates
    /// to an already parsed GGUF without running model inference.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        vokra_core::check_weight_license(file, &vokra_core::CompliancePolicy::strict())?;
        let config = OwsmV4Medium1bConfig::from_gguf(file)?;
        let weights = OwsmV4Medium1bWeights::from_gguf(file)?;
        let frontend = OwsmV4Medium1bFrontend::from_gguf(file, &config)?;
        Ok(Self {
            config,
            weights,
            frontend,
        })
    }

    /// Returns the immutable, structurally authenticated OWSM configuration.
    pub fn config(&self) -> &OwsmV4Medium1bConfig {
        &self.config
    }

    /// Returns the number of structurally authenticated tensor entries.
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Computes the authenticated PCM frontend without entering the model
    /// encoder/decoder.
    pub fn frontend(&self, pcm: &[f32]) -> Result<OwsmV4Medium1bFrontendOutput> {
        self.frontend.extract(pcm)
    }

    /// Native PCM-to-text is not enabled by a manifest-only bind.
    pub fn transcribe(&self, _pcm: &[f32]) -> Result<String> {
        Err(VokraError::NotImplemented(
            "OWSM v4 medium 1B frontend/decoder/tokenizer forward requires independent VAST parity",
        ))
    }
}

impl AsrEngine for OwsmV4Medium1b {
    fn transcribe(&self, _pcm: &[f32]) -> Result<Transcription> {
        Err(VokraError::NotImplemented(
            "OWSM v4 medium 1B frontend/decoder/tokenizer forward requires independent VAST parity",
        ))
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticated_manifest_has_exact_count_and_anchor_shapes() {
        let manifest = expected_tensor_manifest();
        assert_eq!(manifest.len(), CHECKPOINT_TENSOR_COUNT);
        assert_eq!(
            manifest[0],
            ("frontend.logmel.melmat".to_owned(), vec![257, 128])
        );
        assert!(manifest.iter().any(|(name, shape)| {
            name == "encoder.encoders.17.cgmlp.csgu.conv.weight" && shape == &[2048, 1, 31]
        }));
        assert!(manifest.iter().any(|(name, shape)| {
            name == "decoder.decoders.17.src_attn.linear_out.weight" && shape == &[1024, 1024]
        }));
        assert_eq!(manifest.last().unwrap().0, "ctc.ctc_lo.bias");
    }

    #[test]
    fn missing_arch_rejects_before_any_runtime_forward() {
        let builder = vokra_core::gguf::GgufBuilder::new();
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = OwsmV4Medium1bConfig::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains("refusing misroute"));
    }

    fn test_frontend() -> OwsmV4Medium1bFrontend {
        let config = OwsmV4Medium1bFrontendConfig {
            n_fft: 4,
            win_length: 4,
            hop_length: 2,
            n_freqs: 3,
            n_mels: 2,
        };
        OwsmV4Medium1bFrontend {
            config,
            melmat: vec![1.0; 6],
            mean: vec![10.0, -10.0],
            std: vec![1.0, 1.0],
        }
    }

    #[test]
    fn frontend_rejects_empty_nonfinite_and_short_reflect_inputs() {
        let frontend = test_frontend();
        assert!(frontend.extract(&[]).is_err());
        assert!(frontend.extract(&[f32::NAN; 8]).is_err());
        let error = frontend.extract(&[0.0; 2]).unwrap_err();
        assert!(error.to_string().contains("reflect padding"));
    }

    #[test]
    fn frontend_returns_finite_nonzero_shape_and_applies_normalization() {
        let frontend = test_frontend();
        let pcm = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut unnormalized = test_frontend();
        unnormalized.mean = vec![0.0, 0.0];
        let reference = unnormalized.extract(&pcm).unwrap();
        let output = frontend.extract(&pcm).unwrap();
        assert_eq!(output.frames, 5);
        assert_eq!(output.n_mels, 2);
        assert_eq!(output.features.len(), output.frames * output.n_mels);
        assert!(output.features.iter().all(|value| value.is_finite()));
        assert!((output.features[0] - (reference.features[0] - 10.0)).abs() < 1.0e-5);
        assert!((output.features[1] - (reference.features[1] + 10.0)).abs() < 1.0e-5);
    }

    #[test]
    fn frontend_rejects_nonpositive_or_nonfinite_global_mvn_std() {
        let melmat = [1.0f32; 2];
        let mean = [0.0f32; 1];
        assert!(validate_frontend_values(&melmat, &mean, &[0.0], 2, 1).is_err());
        assert!(validate_frontend_values(&melmat, &mean, &[f32::NAN], 2, 1).is_err());
        assert!(validate_frontend_values(&melmat, &mean, &[1.0], 2, 1).is_ok());
        assert!(validate_frontend_values(&[-1.0, 1.0], &mean, &[1.0], 2, 1).is_err());
        assert!(validate_frontend_values(&[0.0, 0.0], &mean, &[1.0], 2, 1).is_err());
        assert!(validate_frontend_values(&[1.0, 0.0], &mean, &[1.0], 2, 2).is_err());
        assert!(
            validate_frontend_values(&[1.0, 0.0, 0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], 2, 2)
                .is_err()
        );
    }
}
