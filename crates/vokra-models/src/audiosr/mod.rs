#![allow(clippy::doc_lazy_continuation)]
//! **AudioSR** (`haoheliu/versatile_audio_super_resolution`, **MIT**) —
//! versatile **audio super-resolution / bandwidth extension** runtime
//! binder for the `audiosr` converter arch (Wave D 2026-08-15,
//! **brand-new capability category**: Vokra had no audio
//! super-resolution model before this landing).
//!
//! # Primary sources
//!
//! - Code (MIT, the config authority):
//!   <https://github.com/haoheliu/versatile_audio_super_resolution>
//! - Paper: Liu, Chen, Tian, Wang & Plumbley, *"AudioSR: Versatile Audio
//!   Super-resolution at Scale"*, arXiv:2309.07314.
//! - Weights: <https://huggingface.co/haoheliu/audiosr_basic> and
//!   <https://huggingface.co/haoheliu/audiosr_speech>.
//!
//! Per the paper abstract, AudioSR accepts input signals with a
//! bandwidth **between 2 kHz and 16 kHz** and upsamples them to a
//! **24 kHz bandwidth at a 48 kHz sampling rate**, across "versatile
//! audio types, including sound effects, music, and speech".
//!
//! # Architecture (transcribed from upstream, fetched 2026-08-15)
//!
//! ```text
//! band-limited PCM (any rate, 2-16 kHz usable bandwidth)
//!   -> resample to 48 kHz                             ← `vokra_ops::resample` LANDED
//!   -> STFT (n_fft 2048, hop 480, win 2048)           ← `vokra_ops::stft` LANDED
//!   -> mel filterbank (256 bands, 20 Hz .. 24 kHz)    ← **WIRED HERE**
//!        [`AudioSr::mel_filterbank`] builds the real bank from the
//!        transcribed axes via `vokra_ops::mel::MelFilterbank`.
//!   -> VAE encode -> latent [16, 128, 32]             ← **loud-partial**
//!        (`first_stage_config` upstream; `vokra_ops::vae_continuous`
//!         exists as an anchor but AudioSR's 2-D mel VAE tensor-name
//!         walk is NOT pinned — the checkpoint is pickle and was not
//!         downloaded.)
//!   -> latent-diffusion U-Net, DDIM                   ← **loud-partial**
//!        (2-D U-Net: model_channels 128, channel_mult [1,2,3,5],
//!         attention_resolutions [8,4,2], num_res_blocks 2,
//!         num_head_channels 32, transformer_depth 1. The condition is
//!         the **low-pass latent concatenated channel-wise** — upstream
//!         config carries a literal `concat_lowpass_cond` key and
//!         `in_channels 32` = 2 x `channels 16`.
//!         The **β schedule + ᾱ table is WIRED HERE** via
//!         [`AudioSr::alphas_cumprod`] over `vokra_ops::ddpm_sampler`;
//!         only the U-Net forward body is missing.)
//!   -> VAE decode -> mel                              ← **loud-partial**
//!   -> vocoder -> 48 kHz PCM                          ← **loud-partial**
//!        (`vokra_ops::hifigan` exists as an anchor, but
//!         `audiosr/utils.py` carries **no vocoder config block** —
//!         verified ABSENT — so the vocoder identity and its tensor
//!         prefix are not transcribed.)
//! ```
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! **Real in this landing** — not scaffolding, actually computed:
//!
//! - [`AudioSrVariant::from_name`] discriminates `basic` vs `speech`.
//! - [`AudioSrConfig::from_gguf`] is a **strict** reader: every axis in
//!   the `vokra.audiosr.*` chunk group must be present or the bind
//!   fails loud naming the missing key (the `emotion2vec` /
//!   `redimnet` strict-read precedent; the converter in this same
//!   commit stamps all of them, so a fallback path would only ever
//!   mask a mis-produced artifact).
//! - [`AudioSr::from_gguf`] enforces strict `vokra.model.arch` equality
//!   and names both the expected and the actual tag on mismatch.
//! - [`AudioSrWeights::tensor`] is a **loud** tensor accessor: a
//!   missing tensor produces a [`VokraError::ModelLoad`] naming the
//!   tensor. This is the accessor the follow-up tensor-name walk will
//!   use, and it is exercised by tests today.
//! - [`AudioSr::mel_filterbank`] builds the **real** 256-band mel
//!   filterbank from the transcribed axes through
//!   `vokra_ops::mel::MelFilterbank` — a genuine computation, not a
//!   placeholder.
//! - [`AudioSr::alphas_cumprod`] builds the **real** cumulative-ᾱ
//!   diffusion table through `vokra_ops::ddpm_sampler` from the
//!   transcribed `timesteps: 1000` + `beta_schedule: "cosine"`.
//!
//! **Loud-partial** — [`AudioSr::super_resolve`] returns
//! [`VokraError::UnsupportedOp`] naming four missing pieces plus the
//! manifest gap. No fabricated waveform is ever emitted (FR-EX-08).
//!
//! # Two axes deliberately NOT claimed as transcribed
//!
//! Honesty markers, so a follow-up parity wave knows exactly where to
//! look rather than trusting this module:
//!
//! 1. **Mel filterbank convention.** The *numeric* axes (sample rate,
//!    `n_fft`, band count, `fmin`, `fmax`) are transcribed verbatim.
//!    The *convention* axes — [`MelScale`], [`MelNorm`],
//!    [`MelInterp`] — are set to Vokra's librosa-compatible defaults
//!    (Slaney scale, Slaney norm, Hz-domain ramps) and were **not**
//!    independently transcribed from upstream. See
//!    [`AudioSr::mel_attrs`].
//! 2. **Diffusion prediction target and cosine parameterisation.**
//!    `prediction_type` does not appear in the transcribed config and
//!    is **not** stamped by the converter; the cosine schedule's `s`
//!    offset and β clip ceiling were likewise not transcribed.
//!    [`AudioSr::alphas_cumprod`] therefore uses Nichol & Dhariwal's
//!    canonical `s = 0.008` / `β_max = 0.999`, which is what
//!    `beta_schedule: "cosine"` denotes. Crucially the ᾱ table does
//!    **not** depend on `prediction_type` at all, so the table this
//!    module computes is well-defined; bit-exact parity against
//!    upstream's own cosine code remains a follow-up parity item.
//!
//! # Sibling distinctness — `audioldm2` is the dangerous neighbour
//!
//! [`ARCH`] = `"audiosr"` shares an author and a latent-diffusion
//! lineage with `audioldm2` but is the **opposite task**:
//!
//! | | `audiosr` | `audioldm2` |
//! |---|---|---|
//! | task | restoration (bandwidth extension) | text-to-audio generation |
//! | condition | low-pass latent, **concatenated** | T5 + CLAP + GPT-2, **cross-attention** |
//! | output rate | 48 kHz | 16 kHz |
//! | mel bands | 256 | (different) |
//!
//! The tensor shapes are incompatible. Silently sharing an arch tag
//! would misroute runtime dispatch — FR-EX-08.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Upstream ships `pytorch_model.bin` (torch pickle) — the HF file
//! listing for `haoheliu/audiosr_basic` is exactly `.gitattributes`,
//! `README.md`, `pytorch_model.bin`. This runtime **never** touches
//! pickle or ONNX (FR-LD-05 / NFR-DS-02); callers bridge to safetensors
//! offline through `tools/parity/nemo_pt_to_safetensors.py` (uv-managed
//! Python 3.12 sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), which is a developer tool, not part of
//! the shipped runtime.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME_BASIC`] / [`NAME_SPEECH`] / [`CATEGORY`] mirror
//! `crates/vokra-convert/src/models/audiosr.rs` so `vokra-models` does
//! not gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`, `vokra-core →
//! GGUF reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::ir::graph::{MelAttrs, MelInterp, MelNorm, MelScale};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::ddpm_sampler::{
    BetaSchedule, CfgMode, CfgScaleProfile, DdpmSamplerConfig, PredictionType, build_alphas_cumprod,
};
use vokra_ops::mel::MelFilterbank;

// ---------------------------------------------------------------------------
// Identity constants — mirror of the converter. See module doc for the
// cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model audiosr` (or `--model audiosr-speech`).
///
/// Shared across both AudioSR checkpoints — upstream
/// `audiosr/utils.py::get_basic_config()` takes no `model_name`
/// argument, so `basic` and `speech` are the same topology with
/// different training data. Variant discrimination happens via
/// [`AudioSrVariant::from_name`] against `vokra.model.name`.
pub const ARCH: &str = "audiosr";

/// Expected `vokra.model.name` for the **basic** (general audio)
/// checkpoint — upstream `--model_name basic`, the CLI default.
pub const NAME_BASIC: &str = "audiosr-basic";

/// Expected `vokra.model.name` for the **speech** checkpoint —
/// upstream `--model_name speech`.
pub const NAME_SPEECH: &str = "audiosr-speech";

/// Expected `vokra.model.category` — a new taxonomy tag, deliberately
/// distinct from the `enhancement` cohort (which removes additive noise
/// within a fixed bandwidth, whereas AudioSR synthesises new spectral
/// content above the input cutoff).
pub const CATEGORY: &str = "super-resolution";

/// Primary-source anchor: the HF weight repo for the basic checkpoint.
pub const PRIMARY_SOURCE_HF_BASIC: &str = "https://huggingface.co/haoheliu/audiosr_basic";

/// Primary-source anchor: the HF weight repo for the speech checkpoint.
pub const PRIMARY_SOURCE_HF_SPEECH: &str = "https://huggingface.co/haoheliu/audiosr_speech";

/// Primary-source anchor: the upstream MIT code repository — the
/// authority for every axis in the `vokra.audiosr.*` chunk group and
/// the tensor-name-walk anchor for the follow-up wave.
pub const PRIMARY_SOURCE_GITHUB: &str =
    "https://github.com/haoheliu/versatile_audio_super_resolution";

/// Primary-source anchor: the paper.
pub const PRIMARY_SOURCE_ARXIV: &str = "https://arxiv.org/abs/2309.07314";

// ---------------------------------------------------------------------------
// `vokra.audiosr.*` chunk-group keys — mirror of the converter.
// ---------------------------------------------------------------------------

/// `vokra.audiosr.sample_rate`.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.audiosr.sample_rate";
/// `vokra.audiosr.duration_ms`.
pub const GGUF_KEY_DURATION_MS: &str = "vokra.audiosr.duration_ms";
/// `vokra.audiosr.n_mel_channels`.
pub const GGUF_KEY_N_MEL_CHANNELS: &str = "vokra.audiosr.n_mel_channels";
/// `vokra.audiosr.n_fft`.
pub const GGUF_KEY_N_FFT: &str = "vokra.audiosr.n_fft";
/// `vokra.audiosr.hop_length`.
pub const GGUF_KEY_HOP_LENGTH: &str = "vokra.audiosr.hop_length";
/// `vokra.audiosr.win_length`.
pub const GGUF_KEY_WIN_LENGTH: &str = "vokra.audiosr.win_length";
/// `vokra.audiosr.mel_fmin`.
pub const GGUF_KEY_MEL_FMIN: &str = "vokra.audiosr.mel_fmin";
/// `vokra.audiosr.mel_fmax`.
pub const GGUF_KEY_MEL_FMAX: &str = "vokra.audiosr.mel_fmax";
/// `vokra.audiosr.num_train_timesteps`.
pub const GGUF_KEY_NUM_TRAIN_TIMESTEPS: &str = "vokra.audiosr.num_train_timesteps";
/// `vokra.audiosr.beta_schedule` (string chunk).
pub const GGUF_KEY_BETA_SCHEDULE: &str = "vokra.audiosr.beta_schedule";
/// `vokra.audiosr.linear_start_scaled_1e6`.
pub const GGUF_KEY_LINEAR_START_SCALED_1E6: &str = "vokra.audiosr.linear_start_scaled_1e6";
/// `vokra.audiosr.linear_end_scaled_1e6`.
pub const GGUF_KEY_LINEAR_END_SCALED_1E6: &str = "vokra.audiosr.linear_end_scaled_1e6";
/// `vokra.audiosr.latent_t_size`.
pub const GGUF_KEY_LATENT_T_SIZE: &str = "vokra.audiosr.latent_t_size";
/// `vokra.audiosr.latent_f_size`.
pub const GGUF_KEY_LATENT_F_SIZE: &str = "vokra.audiosr.latent_f_size";
/// `vokra.audiosr.latent_channels`.
pub const GGUF_KEY_LATENT_CHANNELS: &str = "vokra.audiosr.latent_channels";
/// `vokra.audiosr.unet_in_channels`.
pub const GGUF_KEY_UNET_IN_CHANNELS: &str = "vokra.audiosr.unet_in_channels";
/// `vokra.audiosr.unet_out_channels`.
pub const GGUF_KEY_UNET_OUT_CHANNELS: &str = "vokra.audiosr.unet_out_channels";
/// `vokra.audiosr.unet_model_channels`.
pub const GGUF_KEY_UNET_MODEL_CHANNELS: &str = "vokra.audiosr.unet_model_channels";
/// `vokra.audiosr.unet_num_res_blocks`.
pub const GGUF_KEY_UNET_NUM_RES_BLOCKS: &str = "vokra.audiosr.unet_num_res_blocks";
/// `vokra.audiosr.unet_num_head_channels`.
pub const GGUF_KEY_UNET_NUM_HEAD_CHANNELS: &str = "vokra.audiosr.unet_num_head_channels";
/// `vokra.audiosr.unet_transformer_depth`.
pub const GGUF_KEY_UNET_TRANSFORMER_DEPTH: &str = "vokra.audiosr.unet_transformer_depth";
/// `vokra.audiosr.ddim_sampling_steps` (config-side default, 200).
pub const GGUF_KEY_DDIM_SAMPLING_STEPS: &str = "vokra.audiosr.ddim_sampling_steps";
/// `vokra.audiosr.cli_ddim_steps` (CLI-side default, 50).
pub const GGUF_KEY_CLI_DDIM_STEPS: &str = "vokra.audiosr.cli_ddim_steps";
/// `vokra.audiosr.guidance_scale_scaled_1e3`.
pub const GGUF_KEY_GUIDANCE_SCALE_SCALED_1E3: &str = "vokra.audiosr.guidance_scale_scaled_1e3";
/// `vokra.audiosr.latent_t_per_second_scaled_1e3`.
pub const GGUF_KEY_LATENT_T_PER_SECOND_SCALED_1E3: &str =
    "vokra.audiosr.latent_t_per_second_scaled_1e3";
/// `vokra.audiosr.input_bandwidth_min_hz`.
pub const GGUF_KEY_INPUT_BANDWIDTH_MIN_HZ: &str = "vokra.audiosr.input_bandwidth_min_hz";
/// `vokra.audiosr.input_bandwidth_max_hz`.
pub const GGUF_KEY_INPUT_BANDWIDTH_MAX_HZ: &str = "vokra.audiosr.input_bandwidth_max_hz";
/// `vokra.audiosr.output_bandwidth_hz`.
pub const GGUF_KEY_OUTPUT_BANDWIDTH_HZ: &str = "vokra.audiosr.output_bandwidth_hz";

// ---------------------------------------------------------------------------
// Variant
// ---------------------------------------------------------------------------

/// Which AudioSR checkpoint a GGUF represents.
///
/// Upstream ships exactly two (`--model_name` `choices=["basic",
/// "speech"]`, `default="basic"`). Both share [`ARCH`] and the whole
/// topology chunk group; only the trained weights differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSrVariant {
    /// `haoheliu/audiosr_basic` — general audio (sound effects, music,
    /// speech). The upstream CLI default.
    Basic,
    /// `haoheliu/audiosr_speech` — speech-specialised checkpoint.
    Speech,
}

impl AudioSrVariant {
    /// Discriminates a variant from `vokra.model.name`. Returns `None`
    /// for any string that is not one of the two upstream checkpoints.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            NAME_BASIC => Some(Self::Basic),
            NAME_SPEECH => Some(Self::Speech),
            _ => None,
        }
    }

    /// Canonical `vokra.model.name` string for this variant.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Basic => NAME_BASIC,
            Self::Speech => NAME_SPEECH,
        }
    }

    /// The primary-source HF weight-repo URL for this variant, so the
    /// loud-partial error points at the right card.
    #[must_use]
    pub const fn primary_source_hf(self) -> &'static str {
        match self {
            Self::Basic => PRIMARY_SOURCE_HF_BASIC,
            Self::Speech => PRIMARY_SOURCE_HF_SPEECH,
        }
    }
}

// ---------------------------------------------------------------------------
// Config — STRICT read
// ---------------------------------------------------------------------------

/// Reads a required `u32` axis, failing loud when absent.
fn read_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "audiosr: GGUF is missing the required topology key `{key}`. Every axis \
                 in the `vokra.audiosr.*` chunk group is stamped by \
                 `vokra-cli convert --model audiosr` / `--model audiosr-speech`, so an \
                 artifact missing it was not produced by the AudioSR converter (or was \
                 produced by an older build). Re-convert rather than loading a \
                 partially-described model (FR-EX-08 — no silent default)."
            ))
        })
}

/// AudioSR hyperparameters as they ride the `vokra.audiosr.*` chunk
/// group.
///
/// [`from_gguf`](Self::from_gguf) is **strict**: every axis must be
/// present. The converter in this same commit stamps all of them, so a
/// fallback path could only ever mask a mis-produced artifact
/// (`emotion2vec` / `redimnet` strict-read precedent).
///
/// Three upstream axes are fractional and ride **scaled** integer keys
/// (see [`Self::linear_start`], [`Self::linear_end`],
/// [`Self::guidance_scale`], [`Self::latent_t_per_second`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSrConfig {
    /// Output PCM sample rate, Hz (upstream `sampling_rate: 48000`).
    pub sample_rate: u32,
    /// Fixed generation window in milliseconds (upstream
    /// `duration: 10.24` s).
    pub duration_ms: u32,
    /// Mel band count (upstream `n_mel_channels: 256`).
    pub n_mel_channels: u32,
    /// STFT FFT length (upstream `filter_length: 2048`).
    pub n_fft: u32,
    /// STFT hop in samples (upstream `hop_length: 480`).
    pub hop_length: u32,
    /// STFT window length in samples (upstream `win_length: 2048`).
    pub win_length: u32,
    /// Mel lowest band edge, Hz (upstream `mel_fmin: 20`).
    pub mel_fmin: u32,
    /// Mel highest band edge, Hz (upstream `mel_fmax: 24000`).
    pub mel_fmax: u32,
    /// Diffusion training horizon (upstream `timesteps: 1000`).
    pub num_train_timesteps: u32,
    /// β schedule name (upstream `beta_schedule: "cosine"`).
    pub beta_schedule: String,
    /// `linear_start` x 1e6 (upstream `0.0015`).
    pub linear_start_scaled_1e6: u32,
    /// `linear_end` x 1e6 (upstream `0.0195`).
    pub linear_end_scaled_1e6: u32,
    /// Latent time extent (upstream `latent_t_size: 128`).
    pub latent_t_size: u32,
    /// Latent frequency extent (upstream `latent_f_size: 32`).
    pub latent_f_size: u32,
    /// Latent channel width (upstream `channels: 16`).
    pub latent_channels: u32,
    /// U-Net input channels (upstream `in_channels: 32`).
    pub unet_in_channels: u32,
    /// U-Net output channels (upstream `out_channels: 16`).
    pub unet_out_channels: u32,
    /// U-Net base channel width (upstream `model_channels: 128`).
    pub unet_model_channels: u32,
    /// Residual blocks per resolution (upstream `num_res_blocks: 2`).
    pub unet_num_res_blocks: u32,
    /// Channels per attention head (upstream `num_head_channels: 32`).
    pub unet_num_head_channels: u32,
    /// Spatial-transformer depth (upstream `transformer_depth: 1`).
    pub unet_transformer_depth: u32,
    /// Config-side DDIM step count (upstream
    /// `ddim_sampling_steps: 200`).
    pub ddim_sampling_steps: u32,
    /// CLI-side DDIM step default (upstream
    /// `--ddim_steps default=50`) — genuinely different from
    /// [`Self::ddim_sampling_steps`] upstream, so both are carried.
    pub cli_ddim_steps: u32,
    /// `unconditional_guidance_scale` x 1e3 (upstream `3.5`).
    pub guidance_scale_scaled_1e3: u32,
    /// `latent_t_per_second` x 1e3 (upstream `12.8`).
    pub latent_t_per_second_scaled_1e3: u32,
    /// Minimum supported input bandwidth, Hz (paper: 2 kHz).
    pub input_bandwidth_min_hz: u32,
    /// Maximum supported input bandwidth, Hz (paper: 16 kHz).
    pub input_bandwidth_max_hz: u32,
    /// Output bandwidth, Hz (paper: 24 kHz).
    pub output_bandwidth_hz: u32,
}

impl AudioSrConfig {
    /// Strict read of every `vokra.audiosr.*` axis this binder consumes.
    ///
    /// **Scope note (precision, not an omission).** The converter also
    /// stamps two *array* axes — `attention_resolutions` (`[8, 4, 2]`)
    /// and `channel_mult` (`[1, 2, 3, 5]`), as indexed scalar keys plus
    /// a count. This reader does **not** bind them, because the only
    /// consumer would be the 2-D U-Net forward, which is loud-partial
    /// (see [`AudioSr::super_resolve`]). Binding them now would mean
    /// carrying fields no code reads. The wave that lands the U-Net
    /// body adds them here in the same commit as the code that uses
    /// them.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the first missing key.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let beta_schedule = gguf
            .get(GGUF_KEY_BETA_SCHEDULE)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "audiosr: GGUF is missing the required topology key \
                     `{GGUF_KEY_BETA_SCHEDULE}` (a string chunk — upstream value is \
                     `cosine`). Re-convert rather than guessing the diffusion schedule \
                     (FR-EX-08 — no silent default)."
                ))
            })?
            .to_owned();

        Ok(Self {
            sample_rate: read_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            duration_ms: read_u32(gguf, GGUF_KEY_DURATION_MS)?,
            n_mel_channels: read_u32(gguf, GGUF_KEY_N_MEL_CHANNELS)?,
            n_fft: read_u32(gguf, GGUF_KEY_N_FFT)?,
            hop_length: read_u32(gguf, GGUF_KEY_HOP_LENGTH)?,
            win_length: read_u32(gguf, GGUF_KEY_WIN_LENGTH)?,
            mel_fmin: read_u32(gguf, GGUF_KEY_MEL_FMIN)?,
            mel_fmax: read_u32(gguf, GGUF_KEY_MEL_FMAX)?,
            num_train_timesteps: read_u32(gguf, GGUF_KEY_NUM_TRAIN_TIMESTEPS)?,
            beta_schedule,
            linear_start_scaled_1e6: read_u32(gguf, GGUF_KEY_LINEAR_START_SCALED_1E6)?,
            linear_end_scaled_1e6: read_u32(gguf, GGUF_KEY_LINEAR_END_SCALED_1E6)?,
            latent_t_size: read_u32(gguf, GGUF_KEY_LATENT_T_SIZE)?,
            latent_f_size: read_u32(gguf, GGUF_KEY_LATENT_F_SIZE)?,
            latent_channels: read_u32(gguf, GGUF_KEY_LATENT_CHANNELS)?,
            unet_in_channels: read_u32(gguf, GGUF_KEY_UNET_IN_CHANNELS)?,
            unet_out_channels: read_u32(gguf, GGUF_KEY_UNET_OUT_CHANNELS)?,
            unet_model_channels: read_u32(gguf, GGUF_KEY_UNET_MODEL_CHANNELS)?,
            unet_num_res_blocks: read_u32(gguf, GGUF_KEY_UNET_NUM_RES_BLOCKS)?,
            unet_num_head_channels: read_u32(gguf, GGUF_KEY_UNET_NUM_HEAD_CHANNELS)?,
            unet_transformer_depth: read_u32(gguf, GGUF_KEY_UNET_TRANSFORMER_DEPTH)?,
            ddim_sampling_steps: read_u32(gguf, GGUF_KEY_DDIM_SAMPLING_STEPS)?,
            cli_ddim_steps: read_u32(gguf, GGUF_KEY_CLI_DDIM_STEPS)?,
            guidance_scale_scaled_1e3: read_u32(gguf, GGUF_KEY_GUIDANCE_SCALE_SCALED_1E3)?,
            latent_t_per_second_scaled_1e3: read_u32(
                gguf,
                GGUF_KEY_LATENT_T_PER_SECOND_SCALED_1E3,
            )?,
            input_bandwidth_min_hz: read_u32(gguf, GGUF_KEY_INPUT_BANDWIDTH_MIN_HZ)?,
            input_bandwidth_max_hz: read_u32(gguf, GGUF_KEY_INPUT_BANDWIDTH_MAX_HZ)?,
            output_bandwidth_hz: read_u32(gguf, GGUF_KEY_OUTPUT_BANDWIDTH_HZ)?,
        })
    }

    /// Upstream `linear_start` (`0.0015`), un-scaled.
    #[must_use]
    pub fn linear_start(&self) -> f32 {
        self.linear_start_scaled_1e6 as f32 / 1e6
    }

    /// Upstream `linear_end` (`0.0195`), un-scaled.
    #[must_use]
    pub fn linear_end(&self) -> f32 {
        self.linear_end_scaled_1e6 as f32 / 1e6
    }

    /// Upstream `unconditional_guidance_scale` (`3.5`), un-scaled.
    #[must_use]
    pub fn guidance_scale(&self) -> f32 {
        self.guidance_scale_scaled_1e3 as f32 / 1e3
    }

    /// Upstream `latent_t_per_second` (`12.8`), un-scaled.
    #[must_use]
    pub fn latent_t_per_second(&self) -> f32 {
        self.latent_t_per_second_scaled_1e3 as f32 / 1e3
    }

    /// The fixed generation window in seconds (upstream `10.24`).
    #[must_use]
    pub fn duration_secs(&self) -> f32 {
        self.duration_ms as f32 / 1e3
    }

    /// Number of real-input STFT bins, `n_fft / 2 + 1`.
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_fft as usize / 2 + 1
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Weight tensors bound from an AudioSR GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step — a GGUF carrying zero tensors is rejected rather than silently
/// running an all-zero forward (FR-EX-08).
///
/// No upstream tensor-name manifest has been transcribed (the release
/// ships pickle and was not downloaded), so this struct stores the
/// discovered names + dims and exposes the loud [`tensor`](Self::tensor)
/// accessor the follow-up tensor-name walk will use. Naming a
/// speculative manifest here would be fabrication.
#[derive(Debug)]
pub struct AudioSrWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl AudioSrWeights {
    /// Scans `gguf` for tensors, refusing to bind an empty manifest.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(
                "audiosr: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model audiosr` (or \
                 `--model audiosr-speech`) against a checkpoint bridged from the \
                 upstream `pytorch_model.bin`; a legitimate AudioSR artifact carries \
                 the VAE, the latent-diffusion U-Net and the vocoder weight groups."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Every bound tensor name, in GGUF order. Diagnostic accessor for
    /// the follow-up manifest transcription.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Looks up a tensor's dimensions by upstream `state_dict` name.
    ///
    /// This is the **loud** accessor the follow-up VAE / U-Net /
    /// vocoder tensor-name walks will use: a missing tensor produces a
    /// [`VokraError::ModelLoad`] naming the tensor rather than a silent
    /// zero-fill (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming `name` when it is absent.
    pub fn tensor(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "audiosr: required tensor `{name}` is absent from the GGUF ({} \
                     tensors bound). The artifact does not carry the weight group this \
                     lookup needs — re-convert from a complete upstream checkpoint \
                     rather than proceeding with a zero-filled tensor (FR-EX-08).",
                    self.tensors.len()
                ))
            })
    }
}

// ---------------------------------------------------------------------------
// AudioSr — the runtime binder handle
// ---------------------------------------------------------------------------

/// AudioSR audio super-resolution / bandwidth-extension runtime binder
/// (`haoheliu/audiosr_basic` / `haoheliu/audiosr_speech`, MIT).
///
/// Bind with [`from_gguf`](Self::from_gguf). The mel front-end
/// ([`mel_filterbank`](Self::mel_filterbank)) and the diffusion ᾱ table
/// ([`alphas_cumprod`](Self::alphas_cumprod)) are **real** and usable
/// today; the end-to-end [`super_resolve`](Self::super_resolve) is
/// loud-partial pending the U-Net / VAE / vocoder bodies.
#[derive(Debug)]
pub struct AudioSr {
    config: AudioSrConfig,
    variant: AudioSrVariant,
    weights: AudioSrWeights,
    weight_license: LicenseClass,
}

impl AudioSr {
    /// Binds an AudioSR GGUF: validates arch strictly, discriminates the
    /// variant, strict-reads the topology chunk group, discovers
    /// tensors, and surfaces the stamped weight-license class.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   is not `"audiosr"` — the message names **both** the actual and
    ///   the expected tag, and calls out `audioldm2` explicitly because
    ///   it is the nearest neighbour (same author, same latent-diffusion
    ///   family, opposite task).
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is absent or
    ///   is not one of the two upstream checkpoint names.
    /// - [`VokraError::ModelLoad`] naming the first missing
    ///   `vokra.audiosr.*` topology key.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, so a mis-typed model fails with a
        //    specific message instead of a downstream shape error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "audiosr: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model audiosr` or \
                     `--model audiosr-speech`? The nearest neighbour is `audioldm2` — \
                     same first author and the same latent-diffusion family, but the \
                     OPPOSITE task: AudioLDM 2 is text-to-audio GENERATION conditioned \
                     by T5 + CLAP + GPT-2 tokens through cross-attention at 16 kHz, \
                     whereas AudioSR is bandwidth-extension RESTORATION conditioned by \
                     a low-pass latent concatenated channel-wise at 48 kHz with 256 mel \
                     bands. Their tensor shapes are incompatible. Other siblings that \
                     are NOT this model: `denoise` / `gtcrn` / `nsnet2` / `rnnoise` / \
                     `storm` (noise removal at fixed bandwidth, not bandwidth \
                     extension), `sepformer` / `conv_tasnet` / `demucs` / `bs_roformer` \
                     (source separation), `musicgen` / `stable_audio_open_small` / \
                     `ace_step` (generation), `bigvgan` / `hifigan_vocoder` (vocoders). \
                     Silently aliasing arch would misroute runtime dispatch — \
                     FR-EX-08.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "audiosr: GGUF is missing `vokra.model.arch` (the converter did not \
                     stamp it — this is not a Vokra-native audiosr GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant discrimination. Both checkpoints share the arch
        //    tag; `vokra.model.name` is the discriminator.
        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "audiosr: GGUF is missing `vokra.model.name` (cannot discriminate \
                     the AudioSR checkpoint between `{NAME_BASIC}` (general audio, the \
                     upstream CLI default) and `{NAME_SPEECH}` (speech-specialised))"
                ))
            })?;
        let variant = AudioSrVariant::from_name(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "audiosr: NAME `{name}` is not a recognised AudioSR checkpoint. Expected \
                 one of `{NAME_BASIC}` or `{NAME_SPEECH}` — upstream ships exactly two \
                 (`audiosr --model_name` has `choices=[\"basic\",\"speech\"]`). Primary \
                 source: {PRIMARY_SOURCE_GITHUB}."
            ))
        })?;

        // 3. Strict topology read — every axis or a loud failure.
        let config = AudioSrConfig::from_gguf(file)?;

        // 4. Tensor manifest with the non-emptiness gate.
        let weights = AudioSrWeights::from_gguf(file)?;

        // 5. Provenance surfacing. A GGUF missing the stamp reads back
        //    as `Unknown`, which is fail-closed at the M2-13 gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            variant,
            weights,
            weight_license,
        })
    }

    /// The bound topology axes (strict read of `vokra.audiosr.*`).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &AudioSrConfig {
        &self.config
    }

    /// The bound checkpoint variant.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> AudioSrVariant {
        self.variant
    }

    /// The bound weight manifest (exposes the loud
    /// [`AudioSrWeights::tensor`] accessor).
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &AudioSrWeights {
        &self.weights
    }

    /// The stamped weight-license class. `mit` (repo `LICENSE`) and
    /// `apache-2.0` (the `haoheliu/audiosr_basic` HF card tag) both
    /// normalise to [`LicenseClass::Permissive`], so the verdict is
    /// robust to that upstream discrepancy. A GGUF missing the stamp
    /// reads back as [`LicenseClass::Unknown`], fail-closed at M2-13.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The mel-filterbank attributes implied by the bound config.
    ///
    /// The **numeric** axes — `sample_rate`, `n_fft`, `n_mels`, `fmin`,
    /// `fmax` — are transcribed verbatim from upstream
    /// `audiosr/utils.py::get_basic_config()`.
    ///
    /// The **convention** axes — [`MelScale`], [`MelNorm`],
    /// [`MelInterp`] — are Vokra's librosa-compatible defaults (Slaney
    /// scale, Slaney normalisation, Hz-domain triangular ramps) and were
    /// **not** independently transcribed from upstream. They are the
    /// obvious candidates if a future real-weight parity run disagrees
    /// with the reference mel, and this rustdoc exists so that
    /// investigation starts in the right place rather than assuming the
    /// numeric axes are wrong.
    #[must_use]
    pub fn mel_attrs(&self) -> MelAttrs {
        MelAttrs {
            sample_rate: self.config.sample_rate,
            n_fft: self.config.n_fft as usize,
            n_mels: self.config.n_mel_channels as usize,
            fmin: self.config.mel_fmin as f32,
            fmax: Some(self.config.mel_fmax as f32),
            scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
            interp: MelInterp::Hz,
        }
    }

    /// Builds the **real** AudioSR mel filterbank.
    ///
    /// This is a genuine computation over `vokra_ops::mel`, not a
    /// placeholder: with the upstream axes it yields a bank of
    /// `n_mel_channels` (256) filters over `n_fft / 2 + 1` (1025)
    /// real-input STFT bins. Composable today with `vokra_ops::stft`
    /// to produce an AudioSR-shaped mel spectrogram; only the VAE and
    /// downstream stages are missing.
    ///
    /// See [`mel_attrs`](Self::mel_attrs) for the one caveat about the
    /// filterbank *convention* axes.
    #[must_use]
    pub fn mel_filterbank(&self) -> MelFilterbank {
        MelFilterbank::new(&self.mel_attrs())
    }

    /// Builds the DDPM sampler config implied by the bound topology.
    ///
    /// `num_train_steps` and the β schedule come straight from the
    /// stamped axes (upstream `timesteps: 1000`,
    /// `beta_schedule: "cosine"`). `beta_start` / `beta_end` carry the
    /// stamped `linear_start` / `linear_end` so a caller who overrides
    /// the schedule to [`BetaSchedule::Linear`] gets upstream's own
    /// bounds rather than Ho et al.'s generic ones.
    ///
    /// Two axes are **not** transcribed from upstream and are flagged
    /// rather than hidden:
    ///
    /// - `prediction_type` does not appear in the transcribed config and
    ///   is not stamped by the converter. [`PredictionType::Epsilon`] is
    ///   used as the DDPM default. **The ᾱ table does not depend on it**
    ///   (see [`alphas_cumprod`](Self::alphas_cumprod)), so this choice
    ///   cannot corrupt the schedule; it matters only once a real U-Net
    ///   forward exists, and must be confirmed then.
    /// - `cosine_offset` / `cosine_beta_max` use Nichol & Dhariwal's
    ///   canonical `0.008` / `0.999`, which is what
    ///   `beta_schedule: "cosine"` denotes.
    ///
    /// CFG is left at [`CfgMode::None`]; a real caller sets it from
    /// [`AudioSrConfig::guidance_scale`] (upstream `3.5`) per render
    /// intent.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `num_inference_steps` is
    /// zero or exceeds the stamped training horizon — surfaced by
    /// `vokra_ops::ddpm_sampler`'s own validation.
    pub fn sampler_config(&self, num_inference_steps: u32) -> Result<DdpmSamplerConfig> {
        let beta_schedule = match self.config.beta_schedule.as_str() {
            "cosine" => BetaSchedule::Cosine,
            "linear" => BetaSchedule::Linear,
            other => {
                return Err(VokraError::UnsupportedOp(format!(
                    "audiosr: stamped beta_schedule `{other}` is not supported by \
                     `vokra_ops::ddpm_sampler`, which implements `cosine` (Nichol & \
                     Dhariwal 2021) and `linear` (Ho et al. 2020). Upstream AudioSR \
                     stamps `cosine`. Primary source: {PRIMARY_SOURCE_GITHUB}."
                )));
            }
        };
        // Validate the step count here so the caller gets an
        // AudioSR-specific message naming the upstream defaults, rather
        // than the generic `ddpm_sample:` prefix from the op's own
        // validator (which would still fire later, but reads as if the
        // sampler were already running).
        if num_inference_steps == 0 {
            return Err(VokraError::InvalidArgument(
                "audiosr sampler_config: num_inference_steps must be > 0 (upstream \
                 defaults are 200 in `get_basic_config` and 50 on the CLI)"
                    .to_owned(),
            ));
        }
        if num_inference_steps > self.config.num_train_timesteps {
            return Err(VokraError::InvalidArgument(format!(
                "audiosr sampler_config: num_inference_steps ({num_inference_steps}) must \
                 be <= the stamped training horizon ({})",
                self.config.num_train_timesteps
            )));
        }
        Ok(DdpmSamplerConfig {
            num_train_steps: self.config.num_train_timesteps,
            num_inference_steps,
            // NOT transcribed upstream — see the rustdoc above. Does
            // not affect the ᾱ table.
            prediction_type: PredictionType::Epsilon,
            beta_schedule,
            beta_start: self.config.linear_start(),
            beta_end: self.config.linear_end(),
            cosine_offset: 0.008,
            cosine_beta_max: 0.999,
            cfg_mode: CfgMode::None,
            cfg_scale: CfgScaleProfile::Constant(1.0),
        })
    }

    /// Builds the **real** cumulative-ᾱ diffusion table.
    ///
    /// A genuine computation over `vokra_ops::ddpm_sampler` from the
    /// stamped `timesteps` + `beta_schedule`. The table is a function of
    /// the schedule alone — it does **not** depend on the model's
    /// prediction target — so it is well-defined even though upstream's
    /// `prediction_type` was not transcribed. Bit-exact parity against
    /// upstream's own cosine implementation remains a follow-up parity
    /// item (see [`sampler_config`](Self::sampler_config)).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an invalid step count;
    /// [`VokraError::UnsupportedOp`] on an unsupported stamped schedule.
    pub fn alphas_cumprod(&self, num_inference_steps: u32) -> Result<Vec<f32>> {
        let cfg = self.sampler_config(num_inference_steps)?;
        build_alphas_cumprod(&cfg)
    }

    /// Super-resolves `pcm` (sampled at `input_sample_rate`) to the
    /// model's 48 kHz output rate.
    ///
    /// # Input validation (runs BEFORE the loud-partial gate)
    ///
    /// A caller with a legitimate bug sees a specific
    /// [`VokraError::InvalidArgument`] rather than the generic
    /// loud-partial [`VokraError::UnsupportedOp`]. The loud-partial gate
    /// is not an escape from input validation.
    ///
    /// - empty `pcm`;
    /// - zero `input_sample_rate`;
    /// - non-finite samples (NaN / ±infinity);
    /// - `ddim_steps` of zero or above the stamped training horizon;
    /// - non-finite or non-positive `guidance_scale`.
    ///
    /// Note the input rate is deliberately **not** rejected for being
    /// outside the paper's 2-16 kHz usable-bandwidth window: that window
    /// describes the *content bandwidth* the model was trained to
    /// restore, not a hard constraint on the container sample rate, and
    /// upstream's own README warns about unfamiliar cutoff patterns
    /// (e.g. MP3 loss) degrading quality rather than erroring. Inventing
    /// a hard gate here would be stricter than the primary source.
    ///
    /// # Loud-partial
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the four missing
    /// forward bodies plus the tensor-manifest gap, and citing the
    /// pieces that ARE landed so the follow-up wave knows what it can
    /// compose against. **No fabricated waveform is ever emitted**
    /// (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on input-validation failure;
    /// [`VokraError::UnsupportedOp`] for the loud-partial gate.
    pub fn super_resolve(
        &self,
        pcm: &[f32],
        input_sample_rate: u32,
        ddim_steps: u32,
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "audiosr super_resolve: pcm must not be empty (there is no band-limited \
                 signal to extend)"
                    .to_owned(),
            ));
        }
        if input_sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "audiosr super_resolve: input_sample_rate must be non-zero".to_owned(),
            ));
        }
        if let Some(pos) = pcm.iter().position(|s| !s.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "audiosr super_resolve: pcm[{pos}] is not finite ({}) — NaN / ±infinity \
                 is never a legitimate audio sample",
                pcm[pos]
            )));
        }
        if ddim_steps == 0 {
            return Err(VokraError::InvalidArgument(
                "audiosr super_resolve: ddim_steps must be > 0 (upstream defaults are \
                 200 in the config and 50 on the CLI)"
                    .to_owned(),
            ));
        }
        if ddim_steps > self.config.num_train_timesteps {
            return Err(VokraError::InvalidArgument(format!(
                "audiosr super_resolve: ddim_steps ({ddim_steps}) must be <= the stamped \
                 training horizon ({})",
                self.config.num_train_timesteps
            )));
        }
        if !guidance_scale.is_finite() || guidance_scale <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "audiosr super_resolve: guidance_scale must be finite and positive, got \
                 {guidance_scale} (upstream default is 3.5)"
            )));
        }
        Err(super_resolve_loud_partial(
            &self.config,
            self.variant,
            pcm.len(),
            input_sample_rate,
            ddim_steps,
        ))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`AudioSr::super_resolve`] until the U-Net / VAE / vocoder bodies
/// land.
///
/// Names every missing piece, distinguishes "greenfield" from "anchor
/// exists but the walk is not pinned", cites the pieces that ARE landed,
/// and lists all three primary sources. Mirrors the sortformer / RMVPE /
/// musicgen / audioldm2 loud-partial-message precedent — CLAUDE.md
/// 教訓 (a).
fn super_resolve_loud_partial(
    cfg: &AudioSrConfig,
    variant: AudioSrVariant,
    pcm_len: usize,
    input_sample_rate: u32,
    ddim_steps: u32,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "audiosr super_resolve: latent-diffusion super-resolution chain pending. \
         MISSING: (a) the 2-D latent-diffusion U-Net forward — GREENFIELD, no \
         equivalent primitive in `vokra_ops` today (model_channels={mc}, \
         channel_mult=[1,2,3,5], attention_resolutions=[8,4,2], num_res_blocks={nrb}, \
         num_head_channels={nhc}, transformer_depth={td}; the condition is the low-pass \
         latent CONCATENATED channel-wise, which is why unet_in_channels={uic} is twice \
         latent_channels={lc}); (b) the VAE encode/decode forward mapping the {nm}-band \
         mel to the [{lc}, {lt}, {lf}] latent and back — `vokra_ops::vae_continuous` \
         exists as an ANCHOR but AudioSR's 2-D mel-VAE tensor-name walk is NOT pinned; \
         (c) the vocoder forward (mel -> {sr} Hz PCM) — `vokra_ops::hifigan` exists as \
         an ANCHOR but upstream `audiosr/utils.py` carries NO vocoder config block \
         (verified absent), so the vocoder identity and its state_dict prefix are not \
         transcribed; (d) NO VERIFIED TENSOR-NAME MANIFEST — the upstream release ships \
         `pytorch_model.bin` (torch pickle) and was not downloaded, so no state_dict \
         prefix walk can be pinned without fabrication. ALREADY LANDED and composable \
         today: `vokra_ops::resample` (input rate -> {sr} Hz), `vokra_ops::stft` / \
         `vokra_ops::istft` (n_fft={nfft}, hop={hop}, win={win}), the {nm}-band mel \
         filterbank via `AudioSr::mel_filterbank`, and the cosine-schedule cumulative-a \
         table via `AudioSr::alphas_cumprod` over `vokra_ops::ddpm_sampler` \
         (beta_schedule={bs}, timesteps={tt}) — so the follow-up wave is ONE greenfield \
         U-Net body plus two tensor-name walks, NOT four greenfield kernels. Config: \
         variant={variant_short}, sample_rate={sr}, n_mel_channels={nm}, \
         duration_secs={dur}, input bandwidth window {bwmin}-{bwmax} Hz -> output \
         bandwidth {bwout} Hz. Request: pcm_len={pcm_len}, \
         input_sample_rate={input_sample_rate}, ddim_steps={ddim_steps}. Primary \
         sources: {github} + {hf} + {paper}. Loud pending (CLAUDE.md 教訓 (a) — \
         'loud-partial は fake-complete より honest') — no silent fabricated waveform \
         is ever emitted (FR-EX-08).",
        mc = cfg.unet_model_channels,
        nrb = cfg.unet_num_res_blocks,
        nhc = cfg.unet_num_head_channels,
        td = cfg.unet_transformer_depth,
        uic = cfg.unet_in_channels,
        lc = cfg.latent_channels,
        lt = cfg.latent_t_size,
        lf = cfg.latent_f_size,
        nm = cfg.n_mel_channels,
        sr = cfg.sample_rate,
        nfft = cfg.n_fft,
        hop = cfg.hop_length,
        win = cfg.win_length,
        bs = cfg.beta_schedule,
        tt = cfg.num_train_timesteps,
        dur = cfg.duration_secs(),
        bwmin = cfg.input_bandwidth_min_hz,
        bwmax = cfg.input_bandwidth_max_hz,
        bwout = cfg.output_bandwidth_hz,
        variant_short = match variant {
            AudioSrVariant::Basic => "Basic",
            AudioSrVariant::Speech => "Speech",
        },
        pcm_len = pcm_len,
        input_sample_rate = input_sample_rate,
        ddim_steps = ddim_steps,
        github = PRIMARY_SOURCE_GITHUB,
        hf = variant.primary_source_hf(),
        paper = PRIMARY_SOURCE_ARXIV,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the AudioSR runtime binder.
    //!
    //! What is honestly testable here: variant discrimination, the
    //! strict config round-trip, the **real** mel filterbank and ᾱ
    //! table (genuine computations, not placeholders), the loud
    //! tensor accessor, and negative-space round-trips on every stated
    //! failure surface. Fabricating a real-inference output would
    //! violate CLAUDE.md 教訓 (a).

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Upstream-transcribed axis values, duplicated here so the fixture
    /// is self-describing and a converter drift shows up as a test
    /// disagreement rather than propagating silently.
    fn upstream_config() -> AudioSrConfig {
        AudioSrConfig {
            sample_rate: 48_000,
            duration_ms: 10_240,
            n_mel_channels: 256,
            n_fft: 2048,
            hop_length: 480,
            win_length: 2048,
            mel_fmin: 20,
            mel_fmax: 24_000,
            num_train_timesteps: 1000,
            beta_schedule: "cosine".to_owned(),
            linear_start_scaled_1e6: 1_500,
            linear_end_scaled_1e6: 19_500,
            latent_t_size: 128,
            latent_f_size: 32,
            latent_channels: 16,
            unet_in_channels: 32,
            unet_out_channels: 16,
            unet_model_channels: 128,
            unet_num_res_blocks: 2,
            unet_num_head_channels: 32,
            unet_transformer_depth: 1,
            ddim_sampling_steps: 200,
            cli_ddim_steps: 50,
            guidance_scale_scaled_1e3: 3_500,
            latent_t_per_second_scaled_1e3: 12_800,
            input_bandwidth_min_hz: 2_000,
            input_bandwidth_max_hz: 16_000,
            output_bandwidth_hz: 24_000,
        }
    }

    fn stamp_config(b: &mut GgufBuilder, c: &AudioSrConfig) {
        b.add_u32(GGUF_KEY_SAMPLE_RATE, c.sample_rate);
        b.add_u32(GGUF_KEY_DURATION_MS, c.duration_ms);
        b.add_u32(GGUF_KEY_N_MEL_CHANNELS, c.n_mel_channels);
        b.add_u32(GGUF_KEY_N_FFT, c.n_fft);
        b.add_u32(GGUF_KEY_HOP_LENGTH, c.hop_length);
        b.add_u32(GGUF_KEY_WIN_LENGTH, c.win_length);
        b.add_u32(GGUF_KEY_MEL_FMIN, c.mel_fmin);
        b.add_u32(GGUF_KEY_MEL_FMAX, c.mel_fmax);
        b.add_u32(GGUF_KEY_NUM_TRAIN_TIMESTEPS, c.num_train_timesteps);
        b.add_string(GGUF_KEY_BETA_SCHEDULE, &c.beta_schedule);
        b.add_u32(GGUF_KEY_LINEAR_START_SCALED_1E6, c.linear_start_scaled_1e6);
        b.add_u32(GGUF_KEY_LINEAR_END_SCALED_1E6, c.linear_end_scaled_1e6);
        b.add_u32(GGUF_KEY_LATENT_T_SIZE, c.latent_t_size);
        b.add_u32(GGUF_KEY_LATENT_F_SIZE, c.latent_f_size);
        b.add_u32(GGUF_KEY_LATENT_CHANNELS, c.latent_channels);
        b.add_u32(GGUF_KEY_UNET_IN_CHANNELS, c.unet_in_channels);
        b.add_u32(GGUF_KEY_UNET_OUT_CHANNELS, c.unet_out_channels);
        b.add_u32(GGUF_KEY_UNET_MODEL_CHANNELS, c.unet_model_channels);
        b.add_u32(GGUF_KEY_UNET_NUM_RES_BLOCKS, c.unet_num_res_blocks);
        b.add_u32(GGUF_KEY_UNET_NUM_HEAD_CHANNELS, c.unet_num_head_channels);
        b.add_u32(GGUF_KEY_UNET_TRANSFORMER_DEPTH, c.unet_transformer_depth);
        b.add_u32(GGUF_KEY_DDIM_SAMPLING_STEPS, c.ddim_sampling_steps);
        b.add_u32(GGUF_KEY_CLI_DDIM_STEPS, c.cli_ddim_steps);
        b.add_u32(
            GGUF_KEY_GUIDANCE_SCALE_SCALED_1E3,
            c.guidance_scale_scaled_1e3,
        );
        b.add_u32(
            GGUF_KEY_LATENT_T_PER_SECOND_SCALED_1E3,
            c.latent_t_per_second_scaled_1e3,
        );
        b.add_u32(GGUF_KEY_INPUT_BANDWIDTH_MIN_HZ, c.input_bandwidth_min_hz);
        b.add_u32(GGUF_KEY_INPUT_BANDWIDTH_MAX_HZ, c.input_bandwidth_max_hz);
        b.add_u32(GGUF_KEY_OUTPUT_BANDWIDTH_HZ, c.output_bandwidth_hz);
    }

    /// Builds a well-formed AudioSR GGUF with one representative tensor.
    fn audiosr_gguf(name: &str, license: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, name);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cls) = license {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        stamp_config(&mut b, &upstream_config());
        b.add_tensor(
            "model.diffusion_model.input_blocks.0.0.weight",
            GgmlType::F32,
            vec![128, 32, 3, 3],
            vec![0u8; 128 * 32 * 3 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------
    // Test 1 — variant discrimination + primary-source routing
    // -----------------------------------------------------------------

    #[test]
    fn variant_discrimination_and_primary_source_urls() {
        assert_eq!(
            AudioSrVariant::from_name(NAME_BASIC),
            Some(AudioSrVariant::Basic)
        );
        assert_eq!(
            AudioSrVariant::from_name(NAME_SPEECH),
            Some(AudioSrVariant::Speech)
        );
        assert_eq!(AudioSrVariant::from_name("audiosr"), None);
        assert_eq!(AudioSrVariant::from_name("audioldm2"), None);
        assert_eq!(AudioSrVariant::from_name(""), None);

        assert_eq!(AudioSrVariant::Basic.name(), NAME_BASIC);
        assert_eq!(AudioSrVariant::Speech.name(), NAME_SPEECH);

        assert_eq!(
            AudioSrVariant::Basic.primary_source_hf(),
            "https://huggingface.co/haoheliu/audiosr_basic"
        );
        assert_eq!(
            AudioSrVariant::Speech.primary_source_hf(),
            "https://huggingface.co/haoheliu/audiosr_speech"
        );
        assert_ne!(
            AudioSrVariant::Basic.primary_source_hf(),
            AudioSrVariant::Speech.primary_source_hf(),
            "a copy-paste regression on the enum arm would land here"
        );
        assert_eq!(PRIMARY_SOURCE_ARXIV, "https://arxiv.org/abs/2309.07314");
    }

    // -----------------------------------------------------------------
    // Test 2 — a well-formed synthetic GGUF binds, round-tripping every
    //          transcribed axis
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_binds_and_round_trips_every_axis() {
        let file = audiosr_gguf(NAME_BASIC, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).expect("valid basic GGUF must bind");

        assert_eq!(m.variant(), AudioSrVariant::Basic);
        assert_eq!(*m.config(), upstream_config());
        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "MIT (repo LICENSE) and apache-2.0 (HF card tag) both normalise to \
             Permissive — the verdict is robust to the upstream tag discrepancy"
        );
        assert_eq!(m.tensor_count(), 1);

        // Un-scaling accessors recover the exact upstream fractionals.
        let c = m.config();
        assert!((c.linear_start() - 0.0015).abs() < 1e-9, "linear_start");
        assert!((c.linear_end() - 0.0195).abs() < 1e-9, "linear_end");
        assert!((c.guidance_scale() - 3.5).abs() < 1e-9, "guidance_scale");
        assert!(
            (c.latent_t_per_second() - 12.8).abs() < 1e-6,
            "latent_t_per_second"
        );
        assert!((c.duration_secs() - 10.24).abs() < 1e-6, "duration_secs");
        assert_eq!(c.n_freqs(), 1025, "n_fft 2048 -> 1025 real-input bins");

        // The config-side and CLI-side DDIM defaults genuinely differ
        // upstream; both must survive the round trip.
        assert_eq!(c.ddim_sampling_steps, 200);
        assert_eq!(c.cli_ddim_steps, 50);
        assert_ne!(
            c.ddim_sampling_steps, c.cli_ddim_steps,
            "upstream config default (200) and CLI default (50) differ — carrying only \
             one would make a reader guess which a given render used"
        );
    }

    // -----------------------------------------------------------------
    // Test 3 — the mel filterbank is REAL
    // -----------------------------------------------------------------

    /// The filterbank is a genuine computation over `vokra_ops::mel`,
    /// not a placeholder: it must have the transcribed band count over
    /// the transcribed bin count, and carry non-zero energy.
    #[test]
    fn mel_filterbank_is_real_at_transcribed_axes() {
        let file = audiosr_gguf(NAME_BASIC, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).unwrap();

        let attrs = m.mel_attrs();
        assert_eq!(attrs.sample_rate, 48_000);
        assert_eq!(attrs.n_fft, 2048);
        assert_eq!(attrs.n_mels, 256);
        assert!((attrs.fmin - 20.0).abs() < 1e-6);
        assert_eq!(attrs.fmax, Some(24_000.0));

        let fb = m.mel_filterbank();
        assert_eq!(fb.n_mels, 256, "256 mel bands per upstream n_mel_channels");
        assert_eq!(fb.n_freqs, 1025, "n_fft 2048 -> 1025 real-input bins");
        assert_eq!(
            fb.weights.len(),
            256 * 1025,
            "filter matrix must be fully materialised (layout-independent size check)"
        );
        assert!(
            fb.weights.iter().any(|w| *w > 0.0),
            "a real filterbank must carry non-zero triangular weights — an all-zero \
             matrix would mean the bank silently failed to build"
        );
        assert!(
            fb.weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "mel filter weights must be finite and non-negative"
        );
    }

    // -----------------------------------------------------------------
    // Test 4 — the diffusion alpha-bar table is REAL
    // -----------------------------------------------------------------

    /// `alphas_cumprod` genuinely walks `vokra_ops::ddpm_sampler` with
    /// the transcribed cosine schedule. Assertions are the
    /// schedule-independent mathematical invariants of a cumulative
    /// alpha table, so they cannot be satisfied by a stub.
    #[test]
    fn alphas_cumprod_is_real_and_monotone() {
        let file = audiosr_gguf(NAME_BASIC, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).unwrap();

        // Upstream's own config-side DDIM step count.
        let alphas = m.alphas_cumprod(200).expect("cosine schedule must build");
        assert!(!alphas.is_empty(), "the table must not be empty");
        assert!(
            (alphas[0] - 1.0).abs() < 1e-6,
            "alpha_bar_0 must be exactly 1.0 (no noise at t=0), got {}",
            alphas[0]
        );
        assert!(
            alphas
                .iter()
                .all(|a| a.is_finite() && (0.0..=1.0).contains(a)),
            "every alpha_bar must be a finite probability in [0, 1]"
        );
        for w in alphas.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-6,
                "alpha_bar must be monotonically non-increasing (it is a cumulative \
                 product of (1 - beta) with beta >= 0), saw {} -> {}",
                w[0],
                w[1]
            );
        }
        assert!(
            *alphas.last().unwrap() < alphas[0],
            "the schedule must actually add noise across the horizon — a constant table \
             would mean the schedule silently did nothing"
        );

        // The sampler config carries the stamped axes through.
        let cfg = m.sampler_config(50).expect("CLI-default step count");
        assert_eq!(cfg.num_train_steps, 1000);
        assert_eq!(cfg.num_inference_steps, 50);
        assert_eq!(cfg.beta_schedule, BetaSchedule::Cosine);
        assert!((cfg.beta_start - 0.0015).abs() < 1e-9);
        assert!((cfg.beta_end - 0.0195).abs() < 1e-9);
    }

    // -----------------------------------------------------------------
    // Test 5 — bad inference-step counts fail loud
    // -----------------------------------------------------------------

    #[test]
    fn sampler_config_rejects_out_of_range_step_counts() {
        let file = audiosr_gguf(NAME_BASIC, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).unwrap();

        assert!(
            m.sampler_config(0).is_err(),
            "zero inference steps must fail loud"
        );
        assert!(
            m.sampler_config(1001).is_err(),
            "more inference steps than the 1000-step training horizon must fail loud"
        );
        assert!(m.sampler_config(1000).is_ok(), "the boundary must be valid");
    }

    // -----------------------------------------------------------------
    // Test 6 — wrong arch fails loud naming BOTH tags
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // `audioldm2` is the dangerous neighbour: same author, same
        // latent-diffusion family, opposite task.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "audioldm2");
        b.add_string(chunks::KEY_MODEL_NAME, "audioldm2");
        b.add_tensor("some.tensor", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad on foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`audioldm2`"),
                    "message must name the ACTUAL arch tag, got `{m}`"
                );
                assert!(
                    m.contains("`audiosr`"),
                    "message must name the EXPECTED arch tag, got `{m}`"
                );
                assert!(
                    m.contains("OPPOSITE task") || m.contains("opposite task"),
                    "message should explain why the nearest neighbour is still wrong, \
                     got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 7 — missing arch chunk fails loud
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch_chunk() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, NAME_BASIC);
        b.add_tensor("some.tensor", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("audiosr"),
                    "message must name the complaining binder, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 8 — unknown / missing variant name fails loud
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_and_unknown_variant_name() {
        // (a) missing name chunk
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_tensor("t", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing name");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains("missing `vokra.model.name`")
                    && m.contains(NAME_BASIC)
                    && m.contains(NAME_SPEECH),
                "message must name the missing key AND both valid checkpoints, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (b) unrecognised name
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "audiosr-xl");
        b.add_tensor("t", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad on unknown variant");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains("audiosr-xl") && m.contains(NAME_BASIC),
                "message must echo the offending name and list the valid ones, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 9 — the strict config reader names the missing axis
    // -----------------------------------------------------------------

    /// Strict-read pin: dropping any single axis must fail loud naming
    /// that exact key, never fall back to a silent default.
    #[test]
    fn strict_config_read_names_the_missing_axis() {
        // Drop `latent_channels` from an otherwise complete artifact.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_BASIC);
        let mut c = upstream_config();
        stamp_config(&mut b, &c);
        b.add_tensor("t", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        // Sanity: the complete fixture binds.
        let complete = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(AudioSr::from_gguf(&complete).is_ok());

        // Now rebuild WITHOUT the latent-channels key.
        let mut b2 = GgufBuilder::new();
        b2.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b2.add_string(chunks::KEY_MODEL_NAME, NAME_BASIC);
        c.beta_schedule = "cosine".to_owned();
        // Stamp everything, then rely on a fresh builder that simply
        // omits the one key by stamping the others individually.
        b2.add_u32(GGUF_KEY_SAMPLE_RATE, c.sample_rate);
        b2.add_u32(GGUF_KEY_DURATION_MS, c.duration_ms);
        b2.add_u32(GGUF_KEY_N_MEL_CHANNELS, c.n_mel_channels);
        b2.add_u32(GGUF_KEY_N_FFT, c.n_fft);
        b2.add_u32(GGUF_KEY_HOP_LENGTH, c.hop_length);
        b2.add_u32(GGUF_KEY_WIN_LENGTH, c.win_length);
        b2.add_u32(GGUF_KEY_MEL_FMIN, c.mel_fmin);
        b2.add_u32(GGUF_KEY_MEL_FMAX, c.mel_fmax);
        b2.add_u32(GGUF_KEY_NUM_TRAIN_TIMESTEPS, c.num_train_timesteps);
        b2.add_string(GGUF_KEY_BETA_SCHEDULE, &c.beta_schedule);
        b2.add_u32(GGUF_KEY_LINEAR_START_SCALED_1E6, c.linear_start_scaled_1e6);
        b2.add_u32(GGUF_KEY_LINEAR_END_SCALED_1E6, c.linear_end_scaled_1e6);
        b2.add_u32(GGUF_KEY_LATENT_T_SIZE, c.latent_t_size);
        b2.add_u32(GGUF_KEY_LATENT_F_SIZE, c.latent_f_size);
        // GGUF_KEY_LATENT_CHANNELS deliberately omitted.
        b2.add_u32(GGUF_KEY_UNET_IN_CHANNELS, c.unet_in_channels);
        b2.add_u32(GGUF_KEY_UNET_OUT_CHANNELS, c.unet_out_channels);
        b2.add_u32(GGUF_KEY_UNET_MODEL_CHANNELS, c.unet_model_channels);
        b2.add_u32(GGUF_KEY_UNET_NUM_RES_BLOCKS, c.unet_num_res_blocks);
        b2.add_u32(GGUF_KEY_UNET_NUM_HEAD_CHANNELS, c.unet_num_head_channels);
        b2.add_u32(GGUF_KEY_UNET_TRANSFORMER_DEPTH, c.unet_transformer_depth);
        b2.add_u32(GGUF_KEY_DDIM_SAMPLING_STEPS, c.ddim_sampling_steps);
        b2.add_u32(GGUF_KEY_CLI_DDIM_STEPS, c.cli_ddim_steps);
        b2.add_u32(
            GGUF_KEY_GUIDANCE_SCALE_SCALED_1E3,
            c.guidance_scale_scaled_1e3,
        );
        b2.add_u32(
            GGUF_KEY_LATENT_T_PER_SECOND_SCALED_1E3,
            c.latent_t_per_second_scaled_1e3,
        );
        b2.add_u32(GGUF_KEY_INPUT_BANDWIDTH_MIN_HZ, c.input_bandwidth_min_hz);
        b2.add_u32(GGUF_KEY_INPUT_BANDWIDTH_MAX_HZ, c.input_bandwidth_max_hz);
        b2.add_u32(GGUF_KEY_OUTPUT_BANDWIDTH_HZ, c.output_bandwidth_hz);
        b2.add_tensor("t", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b2.to_bytes().unwrap()).unwrap();

        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad when a topology axis is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_LATENT_CHANNELS),
                    "message must name the exact missing key, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08 (no silent default), got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // The string axis is strict too.
        let mut b3 = GgufBuilder::new();
        b3.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b3.add_string(chunks::KEY_MODEL_NAME, NAME_BASIC);
        b3.add_tensor("t", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b3.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad when the whole chunk group is absent");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    // -----------------------------------------------------------------
    // Test 10 — empty tensor manifest fails loud
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_BASIC);
        stamp_config(&mut b, &upstream_config());
        // No tensors.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = AudioSr::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
                assert!(
                    m.contains("audiosr"),
                    "message must name the converter slug so the reader can re-produce \
                     the artifact, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 11 — the loud tensor accessor names the missing tensor
    // -----------------------------------------------------------------

    /// This is the accessor the follow-up VAE / U-Net / vocoder walks
    /// will use, so its loud behaviour is pinned today.
    #[test]
    fn tensor_accessor_is_loud_and_names_the_tensor() {
        let file = audiosr_gguf(NAME_BASIC, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).unwrap();

        // Present: returns real dims.
        let dims = m
            .weights()
            .tensor("model.diffusion_model.input_blocks.0.0.weight")
            .expect("the fixture tensor must resolve");
        assert_eq!(dims, &[128, 32, 3, 3]);
        assert_eq!(
            m.weights().tensor_names(),
            vec!["model.diffusion_model.input_blocks.0.0.weight"]
        );

        // Absent: loud, naming the tensor.
        let Err(err) = m
            .weights()
            .tensor("first_stage_model.decoder.conv_out.weight")
        else {
            panic!("expected ModelLoad when a required tensor is absent");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("first_stage_model.decoder.conv_out.weight"),
                    "message must name the missing tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08 (no zero-fill), got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 12 — super_resolve loud-partials, naming every gap and anchor
    // -----------------------------------------------------------------

    #[test]
    fn super_resolve_loud_partials_naming_gaps_and_landed_anchors() {
        let file = audiosr_gguf(NAME_SPEECH, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).unwrap();

        // Well-shaped inputs so the loud-partial gate fires rather than
        // the input-validation gate.
        let pcm = vec![0.1_f32; 16_000];
        let Err(err) = m.super_resolve(&pcm, 16_000, 50, 3.5) else {
            panic!("super_resolve must loud-partial on well-shaped inputs");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("audiosr super_resolve"), "surface: {msg}");
                // Every missing piece named.
                assert!(msg.contains("U-Net"), "U-Net gap missing: {msg}");
                assert!(msg.contains("VAE"), "VAE gap missing: {msg}");
                assert!(msg.contains("vocoder"), "vocoder gap missing: {msg}");
                assert!(
                    msg.contains("TENSOR-NAME MANIFEST"),
                    "manifest gap missing: {msg}"
                );
                // Greenfield vs anchor-exists distinction preserved.
                assert!(
                    msg.contains("GREENFIELD"),
                    "greenfield marker missing: {msg}"
                );
                assert!(msg.contains("ANCHOR"), "anchor marker missing: {msg}");
                // Landed primitives cited so the follow-up wave knows
                // what it can compose against.
                assert!(
                    msg.contains("ALREADY LANDED"),
                    "landed-anchor section missing: {msg}"
                );
                assert!(
                    msg.contains("vokra_ops::stft"),
                    "stft anchor missing: {msg}"
                );
                assert!(
                    msg.contains("vokra_ops::ddpm_sampler"),
                    "ddpm_sampler anchor missing: {msg}"
                );
                assert!(
                    msg.contains("mel_filterbank"),
                    "mel filterbank anchor missing: {msg}"
                );
                // Config echoed for cross-checking.
                assert!(msg.contains("sample_rate=48000"), "sample_rate: {msg}");
                assert!(msg.contains("n_mel_channels=256"), "n_mel_channels: {msg}");
                assert!(msg.contains("variant=Speech"), "variant: {msg}");
                assert!(msg.contains("ddim_steps=50"), "ddim_steps: {msg}");
                // All three primary sources.
                assert!(
                    msg.contains("github.com/haoheliu/versatile_audio_super_resolution"),
                    "GitHub source missing: {msg}"
                );
                assert!(
                    msg.contains("huggingface.co/haoheliu/audiosr_speech"),
                    "variant-correct HF source missing: {msg}"
                );
                assert!(msg.contains("2309.07314"), "paper source missing: {msg}");
                // Honesty clauses.
                assert!(msg.contains("FR-EX-08"), "FR-EX-08 missing: {msg}");
                assert!(
                    msg.contains("教訓 (a)") || msg.contains("loud-partial は fake-complete"),
                    "教訓 (a) citation missing: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 13 — input validation precedes the loud-partial gate
    // -----------------------------------------------------------------

    #[test]
    fn super_resolve_validates_inputs_before_loud_partial() {
        let file = audiosr_gguf(NAME_BASIC, Some(LicenseClass::Permissive));
        let m = AudioSr::from_gguf(&file).unwrap();
        let good = vec![0.1_f32; 128];

        // Empty PCM.
        let Err(err) = m.super_resolve(&[], 16_000, 50, 3.5) else {
            panic!("empty pcm must be rejected");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("empty"), "{msg}");
                assert!(
                    !msg.contains("U-Net"),
                    "the validation path must NOT surface the loud-partial primitive \
                     list, got `{msg}`"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }

        // Zero sample rate.
        assert!(matches!(
            m.super_resolve(&good, 0, 50, 3.5),
            Err(VokraError::InvalidArgument(_))
        ));

        // Non-finite samples.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut pcm = good.clone();
            pcm[7] = bad;
            let Err(err) = m.super_resolve(&pcm, 16_000, 50, 3.5) else {
                panic!("non-finite sample {bad} must be rejected");
            };
            match err {
                VokraError::InvalidArgument(msg) => assert!(
                    msg.contains("pcm[7]"),
                    "message should name the offending index, got `{msg}`"
                ),
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }

        // Bad step counts.
        assert!(matches!(
            m.super_resolve(&good, 16_000, 0, 3.5),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            m.super_resolve(&good, 16_000, 1001, 3.5),
            Err(VokraError::InvalidArgument(_))
        ));

        // Bad guidance scales.
        for bad in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    m.super_resolve(&good, 16_000, 50, bad),
                    Err(VokraError::InvalidArgument(_))
                ),
                "guidance_scale {bad} must be rejected"
            );
        }
    }

    // -----------------------------------------------------------------
    // Test 14 — missing provenance stamp is fail-closed
    // -----------------------------------------------------------------

    #[test]
    fn missing_provenance_stamp_falls_back_to_unknown() {
        let file = audiosr_gguf(NAME_BASIC, None);
        let m = AudioSr::from_gguf(&file).expect("bind without provenance");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing provenance stamp must read back as Unknown, which is fail-closed \
             at the M2-13 compliance gate"
        );
    }

    // -----------------------------------------------------------------
    // Test 15 — arch / category distinctness pins (FR-EX-08)
    // -----------------------------------------------------------------

    #[test]
    fn arch_and_category_distinct_from_siblings() {
        assert_eq!(ARCH, "audiosr");
        assert_eq!(NAME_BASIC, "audiosr-basic");
        assert_eq!(NAME_SPEECH, "audiosr-speech");
        assert_eq!(CATEGORY, "super-resolution");

        assert_ne!(
            ARCH, "audioldm2",
            "same author + same latent-diffusion family, but generation vs restoration \
             with incompatible tensor shapes (FR-EX-08)"
        );
        for sibling in [
            "denoise",
            "gtcrn",
            "nsnet2",
            "rnnoise",
            "storm",
            "metricgan_plus",
            "mp_senet_dns",
            "frcrn",
            "facebook_denoiser",
            "sepformer",
            "conv_tasnet",
            "demucs",
            "bs_roformer",
            "tiger_separator",
            "musicgen",
            "stable_audio_open_small",
            "ace_step",
            "bigvgan",
            "hifigan_vocoder",
        ] {
            assert_ne!(
                ARCH, sibling,
                "audiosr (latent-diffusion bandwidth extension) and `{sibling}` are \
                 distinct arches — sharing an arch tag would misroute runtime dispatch \
                 (FR-EX-08)"
            );
        }
        assert_ne!(
            CATEGORY, "enhancement",
            "AudioSR synthesises new spectral content above the input cutoff; the \
             `enhancement` cohort removes additive noise within a fixed bandwidth"
        );
    }
}
