#![allow(clippy::doc_lazy_continuation)]
//! **AudioSR** (`haoheliu/versatile_audio_super_resolution`, **MIT**) —
//! versatile **audio super-resolution / bandwidth extension**:
//! safetensors → GGUF conversion (Wave D 2026-08-15, **brand-new
//! capability category** — Vokra had no audio super-resolution model
//! before this converter).
//!
//! # Model class — latent-diffusion audio super-resolution
//!
//! AudioSR (Liu, Chen, Tian, Wang & Plumbley, arXiv:2309.07314
//! *"AudioSR: Versatile Audio Super-resolution at Scale"*) takes a
//! band-limited input signal and regenerates the missing high band.
//! Per the paper abstract (fetched 2026-08-15 from
//! <https://arxiv.org/abs/2309.07314>) it accepts input signals with a
//! bandwidth **between 2 kHz and 16 kHz** and upsamples them to a
//! **24 kHz bandwidth at a 48 kHz sampling rate**, across "versatile
//! audio types, including sound effects, music, and speech". The
//! abstract additionally frames it as "a plug-and-play module to
//! enhance the generation quality" of AudioLDM, FastSpeech2 and
//! MusicGen.
//!
//! Architecturally it is a **latent-diffusion model** in the AudioLDM
//! lineage (same first author as sibling `audioldm2`): a mel front-end
//! feeds a VAE that compresses to a 2-D latent, a U-Net diffuses in
//! that latent space conditioned on the **low-pass (low-resolution)
//! latent concatenated channel-wise**, and a vocoder returns 48 kHz
//! waveform. The concat conditioning is not guesswork — the upstream
//! config carries a literal `"concat_lowpass_cond"` key, and the U-Net
//! `in_channels = 32` is exactly twice the latent `channels = 16`
//! while `out_channels = 16` matches the latent width.
//!
//! # Distinct arch tag from every sibling
//!
//! [`ARCH`] = `"audiosr"` is **deliberately distinct** from every
//! sibling arch tag. The nearest neighbours and why they are NOT the
//! same thing:
//!
//! - `audioldm2` — CVSSP AudioLDM 2, *same first author, same
//!   latent-diffusion family*, but a **text-to-audio generator**
//!   (T5 + CLAP + GPT-2 triple-fusion **cross-attention** condition,
//!   16 kHz output, 64-bin mel). AudioSR is a **restoration** model
//!   with a **concatenated low-pass latent** condition, 48 kHz output
//!   and a 256-bin mel. Same family, opposite task, incompatible
//!   tensor shapes — this is precisely the pair where a silent arch
//!   alias would be most tempting and most destructive (FR-EX-08).
//! - `denoise` (DeepFilterNet3), `gtcrn`, `nsnet2`, `rnnoise`,
//!   `metricgan_plus`, `mp_senet_dns`, `frcrn`, `facebook_denoiser`,
//!   `storm` — enhancement models that **remove additive noise** at a
//!   fixed bandwidth. AudioSR **adds spectral content above the input
//!   cutoff**; it is not a denoiser and shares no topology with them.
//! - `sepformer`, `conv_tasnet`, `demucs`, `bs_roformer`,
//!   `tiger_separator`, `mossformer2_ss_16k` — source separation.
//! - `musicgen*`, `magnet_*`, `melodyflow_t24_30secs`,
//!   `audiogen_medium`, `jasco_400m_chords_drums`,
//!   `stable_audio_open_small`, `ace_step` — generation families.
//! - `bigvgan`, `hifigan_vocoder` — vocoders (mel → waveform only).
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route an
//! AudioSR checkpoint onto a wrong-topology loader. FR-EX-08 forbids
//! the silent shape misroute.
//!
//! # Two variants — `basic` and `speech`
//!
//! Upstream ships exactly two checkpoints, selected by the CLI's
//! `--model_name` flag (`choices=["basic","speech"]`, `default="basic"`
//! — transcribed verbatim from `audiosr/__main__.py`):
//!
//! - [`NAME_BASIC`] = `audiosr-basic` ← `haoheliu/audiosr_basic`
//! - [`NAME_SPEECH`] = `audiosr-speech` ← `haoheliu/audiosr_speech`
//!
//! Both share the [`ARCH`] tag and the whole `vokra.audiosr.*` topology
//! chunk group: `audiosr/utils.py::get_basic_config()` takes **no
//! `model_name` argument**, so the two checkpoints are the same
//! topology with different training data. Variant discrimination
//! happens on `vokra.model.name` in the runtime binder
//! (`crates/vokra-models/src/audiosr/mod.rs`).
//!
//! # License — MIT (upstream repo `LICENSE`), with a documented
//! # HF-tag discrepancy that does NOT change the class
//!
//! The upstream GitHub repo
//! <https://github.com/haoheliu/versatile_audio_super_resolution>
//! ships an **MIT** `LICENSE` (GitHub's own license API classifies the
//! repo as MIT). One honest caveat worth recording rather than hiding:
//! the `LICENSE` file's copyright line reads *"Copyright (c) 2012-2023
//! Scott Chacon and others"* — an unedited MIT **template** artifact
//! (that line belongs to the `git` project's MIT template), not a
//! statement that Scott Chacon owns AudioSR. The grant text is
//! standard MIT; only the attribution line was never customised.
//!
//! Separately, the weight repos disagree with each other on the HF tag:
//!
//! - `haoheliu/audiosr_basic` — HF API `cardData.license` = `apache-2.0`
//!   (fetched 2026-08-15 via `huggingface.co/api/models/...`);
//! - `haoheliu/audiosr_speech` — **no model card, no license tag**.
//!
//! **The class verdict is robust to this discrepancy**: `mit` and
//! `apache-2.0` both normalise to [`LicenseClass::Permissive`] through
//! [`LicenseClass::from_license_str`], so whichever of the two an owner
//! ultimately ratifies, the runtime compliance gate behaves identically
//! (commercial OK, no attribution obligation, no research flag). The
//! converter defaults to [`DEFAULT_LICENSE_SPDX`] = `"mit"` (the code
//! repo's own `LICENSE`), and the `license` override parameter exists
//! for a repackager who wants to stamp `apache-2.0` to match the HF
//! card instead.
//!
//! §3.1 sign-off in `docs/license-audit.md` is **BLANK** — fail-closed
//! default. CC does NOT sign a license row; that is owner-only per
//! memory `[[feedback-license-signoff-primary-source]]`. The converter
//! + runtime binder land today (structural testing is unblocked);
//! *publish* stays blocked until the owner signs.
//!
//! # `vokra.audiosr.*` topology chunk group
//!
//! Every axis below is transcribed **verbatim** from upstream source
//! fetched 2026-08-15 (CLAUDE.md「ハルシネーション厳禁」 — nothing here
//! is inferred or remembered):
//!
//! `audiosr/utils.py::get_basic_config()`:
//!
//! | upstream key | value | GGUF key |
//! |---|---|---|
//! | `sampling_rate` | `48000` | [`KEY_SAMPLE_RATE`] |
//! | `duration` | `10.24` (s) | [`KEY_DURATION_MS`] (`10240`) |
//! | `n_mel_channels` | `256` | [`KEY_N_MEL_CHANNELS`] |
//! | `filter_length` | `2048` | [`KEY_N_FFT`] |
//! | `hop_length` | `480` | [`KEY_HOP_LENGTH`] |
//! | `win_length` | `2048` | [`KEY_WIN_LENGTH`] |
//! | `mel_fmin` | `20` | [`KEY_MEL_FMIN`] |
//! | `mel_fmax` | `24000` | [`KEY_MEL_FMAX`] |
//! | `timesteps` | `1000` | [`KEY_NUM_TRAIN_TIMESTEPS`] |
//! | `beta_schedule` | `"cosine"` | [`KEY_BETA_SCHEDULE`] |
//! | `linear_start` | `0.0015` | [`KEY_LINEAR_START_SCALED_1E6`] (`1500`) |
//! | `linear_end` | `0.0195` | [`KEY_LINEAR_END_SCALED_1E6`] (`19500`) |
//! | `latent_t_size` | `128` | [`KEY_LATENT_T_SIZE`] |
//! | `latent_f_size` | `32` | [`KEY_LATENT_F_SIZE`] |
//! | `channels` | `16` | [`KEY_LATENT_CHANNELS`] |
//! | `in_channels` (unet) | `32` | [`KEY_UNET_IN_CHANNELS`] |
//! | `out_channels` (unet) | `16` | [`KEY_UNET_OUT_CHANNELS`] |
//! | `model_channels` | `128` | [`KEY_UNET_MODEL_CHANNELS`] |
//! | `num_res_blocks` | `2` | [`KEY_UNET_NUM_RES_BLOCKS`] |
//! | `num_head_channels` | `32` | [`KEY_UNET_NUM_HEAD_CHANNELS`] |
//! | `transformer_depth` | `1` | [`KEY_UNET_TRANSFORMER_DEPTH`] |
//! | `attention_resolutions` | `[8, 4, 2]` | [`KEY_ATTENTION_RESOLUTION_PREFIX`]`{0,1,2}` |
//! | `channel_mult` | `[1, 2, 3, 5]` | [`KEY_CHANNEL_MULT_PREFIX`]`{0,1,2,3}` |
//! | `unconditional_guidance_scale` | `3.5` | [`KEY_GUIDANCE_SCALE_SCALED_1E3`] (`3500`) |
//! | `ddim_sampling_steps` | `200` | [`KEY_DDIM_SAMPLING_STEPS`] |
//!
//! `audiosr/pipeline.py`: `latent_t_per_second=12.8` →
//! [`KEY_LATENT_T_PER_SECOND_SCALED_1E3`] (`12800`).
//!
//! `audiosr/__main__.py` (CLI default, deliberately **distinct** from
//! the config's `ddim_sampling_steps = 200`): `--ddim_steps default=50`
//! → [`KEY_CLI_DDIM_STEPS`]. Both are stamped because they genuinely
//! differ upstream and a reader must not have to guess which one a
//! given render used.
//!
//! arXiv:2309.07314 abstract → [`KEY_INPUT_BANDWIDTH_MIN_HZ`] (`2000`),
//! [`KEY_INPUT_BANDWIDTH_MAX_HZ`] (`16000`),
//! [`KEY_OUTPUT_BANDWIDTH_HZ`] (`24000`).
//!
//! ## Why some axes are scaled integers
//!
//! GGUF hparam chunks in this tree are `u32` by convention. Three
//! upstream axes are fractional, so they ride **scaled** integer keys
//! with the scale factor in the key name — the same
//! `vokra.wavlm.layer_norm_eps_scaled_1e9` /
//! `vokra.wavlm.hidden_dropout_scaled_1e3` precedent. The runtime
//! binder divides back out. No precision is lost: `0.0015`, `0.0195`,
//! `3.5` and `12.8` are all exact at their stated scales.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 ([`GgmlType::BF16`]); the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. The
//! [`AudioSrReport::bf16_passthrough`] counter records how many BF16
//! tensors landed on the arm so a silent widen cannot slip in.
//!
//! # Upstream ships pickle — offline safetensors bridge required
//!
//! The HF file listing for `haoheliu/audiosr_basic` (fetched
//! 2026-08-15) is exactly `.gitattributes`, `README.md`,
//! **`pytorch_model.bin`** — i.e. a torch **pickle**, no safetensors.
//! This converter accepts **safetensors only**; pickle never enters the
//! Rust runtime (FR-LD-05 "no arbitrary code execution at load",
//! NFR-DS-02 zero-dep). Callers bridge offline through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` uv-managed Python 3.12
//! sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), which is a developer tool, not part of
//! the shipped runtime.
//!
//! # Scale — local convert is fine
//!
//! Unlike sibling `audioldm2` (~8.5 GB → vast.ai handoff), the AudioSR
//! `pytorch_model.bin` is a single-file checkpoint well under the 8 GB
//! M1-iMac threshold from memory `[[feedback-large-models-on-vast-ai]]`,
//! so local conversion is expected to be viable. The exact byte size is
//! **not** transcribed here because the HF API response consulted did
//! not include per-file sizes — an owner converting for real should
//! read it off the repo rather than trust a number invented here.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `state_dict` names verbatim**.
//! No tensor-name manifest has been transcribed (the checkpoint is
//! pickle and was not downloaded — see the standing "no multi-GB
//! downloads" rule), so the runtime binder's forward path is
//! **loud-partial**: `crates/vokra-models/src/audiosr/mod.rs`
//! `AudioSr::super_resolve` returns [`VokraError::UnsupportedOp`]
//! naming precisely which primitives are missing. No fabricated
//! waveform is ever emitted (FR-EX-08).
//!
//! [`VokraError::UnsupportedOp`]: vokra_core::VokraError::UnsupportedOp

// Skeleton-only allowance: the public API is exercised by the in-module
// tests + wired to the CLI + `ModelKind` + `pub mod` re-export in the
// same commit. Removed once callers exercise the API outside tests.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

// ---------------------------------------------------------------------------
// Identity constants
// ---------------------------------------------------------------------------

/// `vokra.model.arch` = `audiosr` — distinct from every sibling arch
/// tag, most importantly from `audioldm2` (same author + same
/// latent-diffusion family, but text-to-audio **generation** with
/// cross-attention conditioning at 16 kHz / 64 mel bins, versus
/// AudioSR's **restoration** with concatenated low-pass conditioning at
/// 48 kHz / 256 mel bins). FR-EX-08 forbids the silent shape misroute.
pub const ARCH: &str = "audiosr";

/// `vokra.model.name` for the **basic** checkpoint (general audio:
/// sound effects, music, speech) — upstream `--model_name basic`, the
/// CLI default.
pub const NAME_BASIC: &str = "audiosr-basic";

/// `vokra.model.name` for the **speech** checkpoint — upstream
/// `--model_name speech`.
pub const NAME_SPEECH: &str = "audiosr-speech";

/// `vokra.model.category` = `super-resolution`.
///
/// A **new taxonomy tag**: Vokra had no audio super-resolution /
/// bandwidth-extension model before this converter. It is deliberately
/// NOT `enhancement` — the existing `enhancement` cohort (`denoise`,
/// `gtcrn`, `nsnet2`, `rnnoise`, `storm`, ...) *removes* additive noise
/// within a fixed bandwidth, whereas AudioSR *synthesises new spectral
/// content above the input cutoff*. Collapsing the two would make the
/// zoo manifest advertise a bandwidth extender as a denoiser. Singleton
/// category tags are established precedent in this tree (`watermark`,
/// `f0`, `beat-tracking`, `emotion`, `diarize`, `kws`).
pub const CATEGORY: &str = "super-resolution";

/// Upstream HF weight repo for the [`NAME_BASIC`] checkpoint.
pub const UPSTREAM_HF_BASIC: &str = "haoheliu/audiosr_basic";

/// Upstream HF weight repo for the [`NAME_SPEECH`] checkpoint.
pub const UPSTREAM_HF_SPEECH: &str = "haoheliu/audiosr_speech";

/// Upstream code repository (MIT) — the tensor-name-walk anchor and the
/// authority for every axis in the `vokra.audiosr.*` chunk group.
pub const UPSTREAM_GITHUB: &str = "https://github.com/haoheliu/versatile_audio_super_resolution";

/// Paper anchor — Liu et al., *"AudioSR: Versatile Audio
/// Super-resolution at Scale"*.
pub const PRIMARY_SOURCE_ARXIV: &str = "https://arxiv.org/abs/2309.07314";

/// Default weight-license SPDX (`mit`) per the upstream repo `LICENSE`.
///
/// See the module docstring for the honest caveat about the unedited
/// template copyright line and the `apache-2.0` tag on
/// `haoheliu/audiosr_basic` — both normalise to
/// [`LicenseClass::Permissive`], so the compliance-gate verdict is the
/// same either way.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Ad-hoc metadata key for the model category (converter-side constant,
/// mirroring the sibling `gtcrn` / `nsnet2` / `audioldm2` posture until
/// a first-class `category` consumer lands in `vokra-core`).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream HF repo id.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Ad-hoc metadata key for the upstream code repository (GitHub). The
/// weights live on HF but the *config authority* is the GitHub tree, so
/// both are stamped.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

const UPSTREAM_SOURCE_BASIC: &str = "haoheliu/audiosr_basic (AudioSR: Versatile Audio Super-resolution at Scale, \
     arXiv:2309.07314, latent diffusion, 2-16 kHz input bandwidth -> 24 kHz bandwidth \
     at 48 kHz, general audio checkpoint, MIT)";

const UPSTREAM_SOURCE_SPEECH: &str = "haoheliu/audiosr_speech (AudioSR: Versatile Audio Super-resolution at Scale, \
     arXiv:2309.07314, latent diffusion, 2-16 kHz input bandwidth -> 24 kHz bandwidth \
     at 48 kHz, speech-specialised checkpoint, MIT)";

// ---------------------------------------------------------------------------
// `vokra.audiosr.*` hparam chunk group keys
// ---------------------------------------------------------------------------

/// PCM sample rate, Hz (`sampling_rate`).
pub const KEY_SAMPLE_RATE: &str = "vokra.audiosr.sample_rate";
/// Fixed generation window in **milliseconds** (`duration` = 10.24 s).
pub const KEY_DURATION_MS: &str = "vokra.audiosr.duration_ms";
/// Mel band count (`n_mel_channels`).
pub const KEY_N_MEL_CHANNELS: &str = "vokra.audiosr.n_mel_channels";
/// STFT FFT length (`filter_length`).
pub const KEY_N_FFT: &str = "vokra.audiosr.n_fft";
/// STFT hop, samples (`hop_length`).
pub const KEY_HOP_LENGTH: &str = "vokra.audiosr.hop_length";
/// STFT window length, samples (`win_length`).
pub const KEY_WIN_LENGTH: &str = "vokra.audiosr.win_length";
/// Mel lowest band edge, Hz (`mel_fmin`).
pub const KEY_MEL_FMIN: &str = "vokra.audiosr.mel_fmin";
/// Mel highest band edge, Hz (`mel_fmax`).
pub const KEY_MEL_FMAX: &str = "vokra.audiosr.mel_fmax";
/// Diffusion training horizon (`timesteps`).
pub const KEY_NUM_TRAIN_TIMESTEPS: &str = "vokra.audiosr.num_train_timesteps";
/// β schedule name (`beta_schedule`) — a **string** chunk, unlike every
/// other axis in the group.
pub const KEY_BETA_SCHEDULE: &str = "vokra.audiosr.beta_schedule";
/// `linear_start` × 1e6 (`0.0015` → `1500`).
pub const KEY_LINEAR_START_SCALED_1E6: &str = "vokra.audiosr.linear_start_scaled_1e6";
/// `linear_end` × 1e6 (`0.0195` → `19500`).
pub const KEY_LINEAR_END_SCALED_1E6: &str = "vokra.audiosr.linear_end_scaled_1e6";
/// Latent time extent (`latent_t_size`).
pub const KEY_LATENT_T_SIZE: &str = "vokra.audiosr.latent_t_size";
/// Latent frequency extent (`latent_f_size`).
pub const KEY_LATENT_F_SIZE: &str = "vokra.audiosr.latent_f_size";
/// Latent channel width (`channels`).
pub const KEY_LATENT_CHANNELS: &str = "vokra.audiosr.latent_channels";
/// U-Net input channels (`in_channels`).
pub const KEY_UNET_IN_CHANNELS: &str = "vokra.audiosr.unet_in_channels";
/// U-Net output channels (`out_channels`).
pub const KEY_UNET_OUT_CHANNELS: &str = "vokra.audiosr.unet_out_channels";
/// U-Net base channel width (`model_channels`).
pub const KEY_UNET_MODEL_CHANNELS: &str = "vokra.audiosr.unet_model_channels";
/// Residual blocks per resolution (`num_res_blocks`).
pub const KEY_UNET_NUM_RES_BLOCKS: &str = "vokra.audiosr.unet_num_res_blocks";
/// Channels per attention head (`num_head_channels`).
pub const KEY_UNET_NUM_HEAD_CHANNELS: &str = "vokra.audiosr.unet_num_head_channels";
/// Spatial-transformer depth (`transformer_depth`).
pub const KEY_UNET_TRANSFORMER_DEPTH: &str = "vokra.audiosr.unet_transformer_depth";
/// Number of `attention_resolutions` entries stamped.
pub const KEY_ATTENTION_RESOLUTIONS_COUNT: &str = "vokra.audiosr.attention_resolutions_count";
/// Prefix for the indexed `attention_resolutions` entries — the wavlm
/// `vokra.wavlm.conv_dim_{i}` precedent for array axes.
pub const KEY_ATTENTION_RESOLUTION_PREFIX: &str = "vokra.audiosr.attention_resolution_";
/// Number of `channel_mult` entries stamped.
pub const KEY_CHANNEL_MULT_COUNT: &str = "vokra.audiosr.channel_mult_count";
/// Prefix for the indexed `channel_mult` entries.
pub const KEY_CHANNEL_MULT_PREFIX: &str = "vokra.audiosr.channel_mult_";
/// Config-side DDIM step count (`ddim_sampling_steps` = 200).
pub const KEY_DDIM_SAMPLING_STEPS: &str = "vokra.audiosr.ddim_sampling_steps";
/// CLI-side DDIM step default (`--ddim_steps default=50`) — genuinely
/// different from [`KEY_DDIM_SAMPLING_STEPS`] upstream.
pub const KEY_CLI_DDIM_STEPS: &str = "vokra.audiosr.cli_ddim_steps";
/// `unconditional_guidance_scale` × 1e3 (`3.5` → `3500`).
pub const KEY_GUIDANCE_SCALE_SCALED_1E3: &str = "vokra.audiosr.guidance_scale_scaled_1e3";
/// `latent_t_per_second` × 1e3 (`12.8` → `12800`).
pub const KEY_LATENT_T_PER_SECOND_SCALED_1E3: &str = "vokra.audiosr.latent_t_per_second_scaled_1e3";
/// Minimum supported input bandwidth, Hz (paper abstract: 2 kHz).
pub const KEY_INPUT_BANDWIDTH_MIN_HZ: &str = "vokra.audiosr.input_bandwidth_min_hz";
/// Maximum supported input bandwidth, Hz (paper abstract: 16 kHz).
pub const KEY_INPUT_BANDWIDTH_MAX_HZ: &str = "vokra.audiosr.input_bandwidth_max_hz";
/// Output bandwidth, Hz (paper abstract: 24 kHz).
pub const KEY_OUTPUT_BANDWIDTH_HZ: &str = "vokra.audiosr.output_bandwidth_hz";

// ---------------------------------------------------------------------------
// Primary-source-transcribed defaults
// ---------------------------------------------------------------------------

/// `sampling_rate: 48000` — `audiosr/utils.py::get_basic_config`.
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;
/// `duration: 10.24` seconds → 10 240 ms.
pub const DEFAULT_DURATION_MS: u32 = 10_240;
/// `n_mel_channels: 256`.
pub const DEFAULT_N_MEL_CHANNELS: u32 = 256;
/// `filter_length: 2048`.
pub const DEFAULT_N_FFT: u32 = 2048;
/// `hop_length: 480` (= 100 mel frames/s at 48 kHz).
pub const DEFAULT_HOP_LENGTH: u32 = 480;
/// `win_length: 2048`.
pub const DEFAULT_WIN_LENGTH: u32 = 2048;
/// `mel_fmin: 20`.
pub const DEFAULT_MEL_FMIN: u32 = 20;
/// `mel_fmax: 24000` (= Nyquist at 48 kHz).
pub const DEFAULT_MEL_FMAX: u32 = 24_000;
/// `timesteps: 1000`.
pub const DEFAULT_NUM_TRAIN_TIMESTEPS: u32 = 1000;
/// `beta_schedule: "cosine"`.
pub const DEFAULT_BETA_SCHEDULE: &str = "cosine";
/// `linear_start: 0.0015` × 1e6.
pub const DEFAULT_LINEAR_START_SCALED_1E6: u32 = 1_500;
/// `linear_end: 0.0195` × 1e6.
pub const DEFAULT_LINEAR_END_SCALED_1E6: u32 = 19_500;
/// `latent_t_size: 128`.
pub const DEFAULT_LATENT_T_SIZE: u32 = 128;
/// `latent_f_size: 32`.
pub const DEFAULT_LATENT_F_SIZE: u32 = 32;
/// `channels: 16`.
pub const DEFAULT_LATENT_CHANNELS: u32 = 16;
/// unet `in_channels: 32` (= 2 × [`DEFAULT_LATENT_CHANNELS`], the
/// concatenated low-pass conditioning latent).
pub const DEFAULT_UNET_IN_CHANNELS: u32 = 32;
/// unet `out_channels: 16`.
pub const DEFAULT_UNET_OUT_CHANNELS: u32 = 16;
/// `model_channels: 128`.
pub const DEFAULT_UNET_MODEL_CHANNELS: u32 = 128;
/// `num_res_blocks: 2`.
pub const DEFAULT_UNET_NUM_RES_BLOCKS: u32 = 2;
/// `num_head_channels: 32`.
pub const DEFAULT_UNET_NUM_HEAD_CHANNELS: u32 = 32;
/// `transformer_depth: 1`.
pub const DEFAULT_UNET_TRANSFORMER_DEPTH: u32 = 1;
/// `attention_resolutions: [8, 4, 2]`.
pub const DEFAULT_ATTENTION_RESOLUTIONS: [u32; 3] = [8, 4, 2];
/// `channel_mult: [1, 2, 3, 5]`.
pub const DEFAULT_CHANNEL_MULT: [u32; 4] = [1, 2, 3, 5];
/// `ddim_sampling_steps: 200` (config-side).
pub const DEFAULT_DDIM_SAMPLING_STEPS: u32 = 200;
/// `--ddim_steps default=50` (CLI-side).
pub const DEFAULT_CLI_DDIM_STEPS: u32 = 50;
/// `unconditional_guidance_scale: 3.5` × 1e3. The CLI's
/// `--guidance_scale default=3.5` agrees.
pub const DEFAULT_GUIDANCE_SCALE_SCALED_1E3: u32 = 3_500;
/// `latent_t_per_second=12.8` × 1e3 (`audiosr/pipeline.py`).
pub const DEFAULT_LATENT_T_PER_SECOND_SCALED_1E3: u32 = 12_800;
/// Paper abstract: input bandwidth lower bound, 2 kHz.
pub const DEFAULT_INPUT_BANDWIDTH_MIN_HZ: u32 = 2_000;
/// Paper abstract: input bandwidth upper bound, 16 kHz.
pub const DEFAULT_INPUT_BANDWIDTH_MAX_HZ: u32 = 16_000;
/// Paper abstract: output bandwidth, 24 kHz.
pub const DEFAULT_OUTPUT_BANDWIDTH_HZ: u32 = 24_000;

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Outcome of an AudioSR conversion.
///
/// Counter shape mirrors the sibling BF16-pass-through converters
/// ([`super::gtcrn::GtcrnReport`], `super::audioldm2::AudioLdm2Report`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudioSrReport {
    /// Total tensors surfaced by the safetensors reader (the sum of
    /// `written + skipped_non_float`).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]).
    pub bf16_passthrough: usize,
}

// ---------------------------------------------------------------------------
// Conversion entry points
// ---------------------------------------------------------------------------

/// Converts a `haoheliu/audiosr_basic` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`.
///
/// The upstream release ships `pytorch_model.bin` (torch pickle), so
/// callers bridge to safetensors **offline** first via
/// `tools/parity/nemo_pt_to_safetensors.py` — pickle never enters the
/// Rust runtime (FR-LD-05 / NFR-DS-02).
///
/// `license` overrides [`DEFAULT_LICENSE_SPDX`] (`"mit"`); the class is
/// re-derived via [`LicenseClass::from_license_str`]. Pass
/// `Some("apache-2.0")` to match the `haoheliu/audiosr_basic` HF card
/// tag instead of the repo `LICENSE` — both land on
/// [`LicenseClass::Permissive`].
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// cannot be assembled.
pub fn convert_audiosr_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioSrReport, ConvertError> {
    convert_audiosr_family_file(
        input,
        output,
        license,
        NAME_BASIC,
        UPSTREAM_HF_BASIC,
        UPSTREAM_SOURCE_BASIC,
    )
}

/// Converts a `haoheliu/audiosr_speech` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`.
///
/// The speech-specialised sibling of [`convert_audiosr_file`]. The
/// topology is **identical** — `audiosr/utils.py::get_basic_config()`
/// takes no `model_name` argument, so both checkpoints share every axis
/// in the `vokra.audiosr.*` chunk group; only the training data (and
/// therefore the weights, the `vokra.model.name` stamp and the upstream
/// repo id) differ.
///
/// Note that `haoheliu/audiosr_speech` carries **no HF model card and
/// no license tag**; the stamped default therefore falls back to the
/// upstream code repo's MIT `LICENSE` exactly as for the basic
/// checkpoint. See the module docstring.
///
/// # Errors
///
/// Same failure modes as [`convert_audiosr_file`].
pub fn convert_audiosr_speech_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioSrReport, ConvertError> {
    convert_audiosr_family_file(
        input,
        output,
        license,
        NAME_SPEECH,
        UPSTREAM_HF_SPEECH,
        UPSTREAM_SOURCE_SPEECH,
    )
}

/// Shared implementation for the AudioSR family (basic + speech — same
/// arch, same topology chunk group, different `vokra.model.name` +
/// provenance stamps).
///
/// Kept `pub(crate)` so a future variant can piggyback without
/// duplicating the BF16 pass-through dispatch; external callers route
/// through [`convert_audiosr_file`] / [`convert_audiosr_speech_file`]
/// so the built-in defaults stay in one place (the sibling
/// `audioldm2::convert_audioldm2_family_file` posture).
pub(crate) fn convert_audiosr_family_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    name: &str,
    upstream_hf: &str,
    upstream_source: &str,
) -> Result<AudioSrReport, ConvertError> {
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, name);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default `mit` per the upstream repo LICENSE; the
    // override knob exists for a repackager who prefers the
    // `apache-2.0` tag on the `haoheliu/audiosr_basic` HF card. Both
    // normalise to `LicenseClass::Permissive`, so the compliance-gate
    // verdict is unchanged either way — see the module docstring.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(name),
        Some(upstream_source),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, upstream_hf);
    // The weights live on HF but the *config authority* (and the MIT
    // LICENSE) is the GitHub tree, so both provenance anchors ride the
    // artifact.
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_GITHUB);

    stamp_hparams(&mut b);

    let mut report = AudioSrReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (`docs/adr/qwen3-tts-bf16.md`, strategy A_passthrough); the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

/// Stamps the full `vokra.audiosr.*` topology chunk group.
///
/// Every value is a primary-source transcription (see the module
/// docstring's table). Both AudioSR checkpoints share these axes
/// because `get_basic_config()` takes no `model_name` argument
/// upstream — the two releases differ only in trained weights.
fn stamp_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_DURATION_MS, DEFAULT_DURATION_MS);
    b.add_u32(KEY_N_MEL_CHANNELS, DEFAULT_N_MEL_CHANNELS);
    b.add_u32(KEY_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_HOP_LENGTH, DEFAULT_HOP_LENGTH);
    b.add_u32(KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH);
    b.add_u32(KEY_MEL_FMIN, DEFAULT_MEL_FMIN);
    b.add_u32(KEY_MEL_FMAX, DEFAULT_MEL_FMAX);
    b.add_u32(KEY_NUM_TRAIN_TIMESTEPS, DEFAULT_NUM_TRAIN_TIMESTEPS);
    b.add_string(KEY_BETA_SCHEDULE, DEFAULT_BETA_SCHEDULE);
    b.add_u32(KEY_LINEAR_START_SCALED_1E6, DEFAULT_LINEAR_START_SCALED_1E6);
    b.add_u32(KEY_LINEAR_END_SCALED_1E6, DEFAULT_LINEAR_END_SCALED_1E6);
    b.add_u32(KEY_LATENT_T_SIZE, DEFAULT_LATENT_T_SIZE);
    b.add_u32(KEY_LATENT_F_SIZE, DEFAULT_LATENT_F_SIZE);
    b.add_u32(KEY_LATENT_CHANNELS, DEFAULT_LATENT_CHANNELS);
    b.add_u32(KEY_UNET_IN_CHANNELS, DEFAULT_UNET_IN_CHANNELS);
    b.add_u32(KEY_UNET_OUT_CHANNELS, DEFAULT_UNET_OUT_CHANNELS);
    b.add_u32(KEY_UNET_MODEL_CHANNELS, DEFAULT_UNET_MODEL_CHANNELS);
    b.add_u32(KEY_UNET_NUM_RES_BLOCKS, DEFAULT_UNET_NUM_RES_BLOCKS);
    b.add_u32(KEY_UNET_NUM_HEAD_CHANNELS, DEFAULT_UNET_NUM_HEAD_CHANNELS);
    b.add_u32(KEY_UNET_TRANSFORMER_DEPTH, DEFAULT_UNET_TRANSFORMER_DEPTH);

    // Array axes ride indexed scalar keys (the wavlm
    // `vokra.wavlm.conv_dim_{i}` precedent) plus an explicit count so a
    // reader never has to probe for the end of the run.
    b.add_u32(
        KEY_ATTENTION_RESOLUTIONS_COUNT,
        DEFAULT_ATTENTION_RESOLUTIONS.len() as u32,
    );
    for (i, v) in DEFAULT_ATTENTION_RESOLUTIONS.iter().enumerate() {
        b.add_u32(&format!("{KEY_ATTENTION_RESOLUTION_PREFIX}{i}"), *v);
    }
    b.add_u32(KEY_CHANNEL_MULT_COUNT, DEFAULT_CHANNEL_MULT.len() as u32);
    for (i, v) in DEFAULT_CHANNEL_MULT.iter().enumerate() {
        b.add_u32(&format!("{KEY_CHANNEL_MULT_PREFIX}{i}"), *v);
    }

    b.add_u32(KEY_DDIM_SAMPLING_STEPS, DEFAULT_DDIM_SAMPLING_STEPS);
    b.add_u32(KEY_CLI_DDIM_STEPS, DEFAULT_CLI_DDIM_STEPS);
    b.add_u32(
        KEY_GUIDANCE_SCALE_SCALED_1E3,
        DEFAULT_GUIDANCE_SCALE_SCALED_1E3,
    );
    b.add_u32(
        KEY_LATENT_T_PER_SECOND_SCALED_1E3,
        DEFAULT_LATENT_T_PER_SECOND_SCALED_1E3,
    );
    b.add_u32(KEY_INPUT_BANDWIDTH_MIN_HZ, DEFAULT_INPUT_BANDWIDTH_MIN_HZ);
    b.add_u32(KEY_INPUT_BANDWIDTH_MAX_HZ, DEFAULT_INPUT_BANDWIDTH_MAX_HZ);
    b.add_u32(KEY_OUTPUT_BANDWIDTH_HZ, DEFAULT_OUTPUT_BANDWIDTH_HZ);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + monotonically increasing
    /// sequence — the sibling gtcrn / sepformer pattern; no external
    /// `tempfile` dep, preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-audiosr-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    // -----------------------------------------------------------------
    // Test 1 — BF16 round-trip + full topology + provenance stamps
    // -----------------------------------------------------------------

    /// BF16 pass-through pin plus the full metadata surface. Non-zero
    /// BF16 bit patterns so a silent widen / downcast is caught by the
    /// byte-identity assert (a zeroed payload would round-trip
    /// trivially through an F32 widen too).
    #[test]
    fn bf16_tensor_passes_through_and_full_metadata_lands() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements x 2 bytes BF16 payload");

        // A plausible latent-diffusion U-Net tensor name. The real
        // upstream manifest has NOT been transcribed (upstream ships
        // pickle and was not downloaded), so this fixture only
        // exercises the byte-copy path — it is not a claim about the
        // upstream state-dict naming.
        let input_bytes = safetensors_one(
            "model.diffusion_model.input_blocks.0.0.weight",
            "BF16",
            &[2, 3],
            &bf16,
        );
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_audiosr_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("model.diffusion_model.input_blocks.0.0.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Identity + provenance chunks.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_BASIC)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins AudioSR as `super-resolution`, NOT `enhancement`"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT weight license normalises to LicenseClass::Permissive"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_BASIC)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_GITHUB),
            "the GitHub tree is the config authority and must ride the artifact"
        );
        // Schema stamp is written unconditionally by the GGUF writer.
        assert!(file.get(chunks::KEY_SCHEMA_VERSION).is_some());
        assert!(file.get(chunks::KEY_SCHEMA_PRODUCER).is_some());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 2 — every transcribed axis lands with its exact value
    // -----------------------------------------------------------------

    /// Pin every `vokra.audiosr.*` axis to the value transcribed from
    /// the upstream source on 2026-08-15. A drift in any default lands
    /// here in the same commit or fails this test
    /// (CLAUDE.md「ハルシネーション厳禁」).
    #[test]
    fn all_transcribed_axes_emit_expected_upstream_values() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("dummy.weight", "F32", &[1], &payload);
        let input = scratch_path("axes-in");
        let output = scratch_path("axes-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_audiosr_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        for (k, want) in [
            (KEY_SAMPLE_RATE, 48_000u32),
            (KEY_DURATION_MS, 10_240),
            (KEY_N_MEL_CHANNELS, 256),
            (KEY_N_FFT, 2048),
            (KEY_HOP_LENGTH, 480),
            (KEY_WIN_LENGTH, 2048),
            (KEY_MEL_FMIN, 20),
            (KEY_MEL_FMAX, 24_000),
            (KEY_NUM_TRAIN_TIMESTEPS, 1000),
            (KEY_LINEAR_START_SCALED_1E6, 1_500),
            (KEY_LINEAR_END_SCALED_1E6, 19_500),
            (KEY_LATENT_T_SIZE, 128),
            (KEY_LATENT_F_SIZE, 32),
            (KEY_LATENT_CHANNELS, 16),
            (KEY_UNET_IN_CHANNELS, 32),
            (KEY_UNET_OUT_CHANNELS, 16),
            (KEY_UNET_MODEL_CHANNELS, 128),
            (KEY_UNET_NUM_RES_BLOCKS, 2),
            (KEY_UNET_NUM_HEAD_CHANNELS, 32),
            (KEY_UNET_TRANSFORMER_DEPTH, 1),
            (KEY_ATTENTION_RESOLUTIONS_COUNT, 3),
            (KEY_CHANNEL_MULT_COUNT, 4),
            (KEY_DDIM_SAMPLING_STEPS, 200),
            (KEY_CLI_DDIM_STEPS, 50),
            (KEY_GUIDANCE_SCALE_SCALED_1E3, 3_500),
            (KEY_LATENT_T_PER_SECOND_SCALED_1E3, 12_800),
            (KEY_INPUT_BANDWIDTH_MIN_HZ, 2_000),
            (KEY_INPUT_BANDWIDTH_MAX_HZ, 16_000),
            (KEY_OUTPUT_BANDWIDTH_HZ, 24_000),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(u64::from(want)),
                "hparam `{k}` must be stamped as {want}"
            );
        }

        // The one string axis in the group.
        assert_eq!(
            file.get(KEY_BETA_SCHEDULE).and_then(|v| v.as_str()),
            Some("cosine"),
            "beta_schedule is `cosine` upstream, NOT linear — the linear_start / \
             linear_end axes are stamped too but the cosine schedule is what \
             `timesteps: 1000` is walked with"
        );

        // Indexed array axes.
        for (i, want) in DEFAULT_ATTENTION_RESOLUTIONS.iter().enumerate() {
            let k = format!("{KEY_ATTENTION_RESOLUTION_PREFIX}{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(*want)),
                "attention_resolutions[{i}] must be {want}"
            );
        }
        for (i, want) in DEFAULT_CHANNEL_MULT.iter().enumerate() {
            let k = format!("{KEY_CHANNEL_MULT_PREFIX}{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(*want)),
                "channel_mult[{i}] must be {want}"
            );
        }

        // Structural invariants that hold on the transcribed values.
        // These are arithmetic checks on what upstream states, not
        // independent claims about the architecture.
        assert_eq!(
            DEFAULT_UNET_IN_CHANNELS,
            2 * DEFAULT_LATENT_CHANNELS,
            "unet in_channels (32) is exactly twice the latent width (16) — \
             consistent with the `concat_lowpass_cond` conditioning key"
        );
        assert_eq!(
            DEFAULT_UNET_OUT_CHANNELS, DEFAULT_LATENT_CHANNELS,
            "unet out_channels must match the latent width"
        );
        assert_eq!(
            DEFAULT_MEL_FMAX,
            DEFAULT_SAMPLE_RATE / 2,
            "mel_fmax (24 kHz) is Nyquist at 48 kHz"
        );
        assert_eq!(
            DEFAULT_OUTPUT_BANDWIDTH_HZ,
            DEFAULT_SAMPLE_RATE / 2,
            "the paper's 24 kHz output bandwidth is Nyquist at the 48 kHz output rate"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 3 — speech sibling flips name + upstream, keeps topology
    // -----------------------------------------------------------------

    /// The speech checkpoint shares every topology axis with the basic
    /// checkpoint (`get_basic_config()` takes no `model_name` argument
    /// upstream); only `vokra.model.name` + the upstream HF repo flip.
    #[test]
    fn speech_variant_flips_name_and_upstream_but_shares_topology() {
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("x", "F32", &[1], &payload);
        let input = scratch_path("speech-in");
        let output = scratch_path("speech-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_audiosr_speech_file(&input, &output, None).expect("convert speech");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // Arch is SHARED across the family.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        // Name + upstream flip.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_SPEECH)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_SPEECH)
        );
        assert_ne!(
            NAME_BASIC, NAME_SPEECH,
            "the two checkpoint names must be distinct so the runtime binder can \
             discriminate them"
        );
        assert_ne!(UPSTREAM_HF_BASIC, UPSTREAM_HF_SPEECH);

        // Topology is IDENTICAL to the basic variant.
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(48_000)
        );
        assert_eq!(
            file.get(KEY_N_MEL_CHANNELS).and_then(|v| v.as_u64()),
            Some(256)
        );
        assert_eq!(
            file.get(KEY_LATENT_CHANNELS).and_then(|v| v.as_u64()),
            Some(16)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 4 — license override swaps the SPDX, class stays Permissive
    // -----------------------------------------------------------------

    /// The documented HF-tag discrepancy in action: overriding to
    /// `apache-2.0` (the `haoheliu/audiosr_basic` card tag) flips the
    /// stamped SPDX but lands on the SAME `LicenseClass::Permissive`
    /// verdict as the repo's `mit` LICENSE. This is why the discrepancy
    /// is documented rather than blocking.
    #[test]
    fn license_override_swaps_spdx_and_class_stays_permissive() {
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("x", "F32", &[1], &payload);
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_audiosr_file(&input, &output, Some("apache-2.0")).expect("convert with override");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX lands verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 (HF card tag) and mit (repo LICENSE) both normalise to \
             LicenseClass::Permissive — the compliance verdict is robust to the \
             upstream tag discrepancy"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 5 — malformed safetensors input fails loud
    // -----------------------------------------------------------------

    /// A truncated / garbage input must surface a loud
    /// [`ConvertError`], never a silently-empty GGUF.
    #[test]
    fn malformed_safetensors_input_fails_loud() {
        let input = scratch_path("bad-in");
        let output = scratch_path("bad-out");
        // An 8-byte header length that promises far more bytes than the
        // file actually carries.
        let mut bad = Vec::new();
        bad.extend_from_slice(&(4096u64).to_le_bytes());
        bad.extend_from_slice(b"{\"x\":");
        std::fs::write(&input, &bad).expect("write malformed input");

        let err = convert_audiosr_file(&input, &output, None);
        assert!(
            err.is_err(),
            "a malformed safetensors header must fail loud rather than emit an \
             empty GGUF (FR-EX-08)"
        );
        assert!(
            !output.exists(),
            "no output artifact may be written when the input cannot be parsed"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 6 — non-existent input path fails loud with Io
    // -----------------------------------------------------------------

    #[test]
    fn missing_input_path_fails_loud_with_io() {
        let input = scratch_path("does-not-exist");
        let output = scratch_path("never-written");
        let Err(err) = convert_audiosr_file(&input, &output, None) else {
            panic!("expected an error when the input path does not exist");
        };
        assert!(
            matches!(err, ConvertError::Io(_)),
            "a missing input file must surface ConvertError::Io, got {err:?}"
        );
        assert!(!output.exists());
    }

    // -----------------------------------------------------------------
    // Test 7 — arch tag distinct from every sibling family (FR-EX-08)
    // -----------------------------------------------------------------

    /// Pin `ARCH = "audiosr"` and assert distinctness against every
    /// sibling arch string, with `audioldm2` called out first because
    /// it is the nearest neighbour (same author, same latent-diffusion
    /// family, opposite task).
    #[test]
    fn arch_tag_distinct_from_sibling_families() {
        assert_eq!(ARCH, "audiosr");
        assert_eq!(CATEGORY, "super-resolution");
        assert_ne!(
            ARCH, "audioldm2",
            "audiosr (super-resolution, concat low-pass latent condition, 48 kHz, \
             256 mel bins) and audioldm2 (text-to-audio generation, T5+CLAP+GPT-2 \
             cross-attention condition, 16 kHz) are the SAME latent-diffusion family \
             by the same author but opposite tasks with incompatible tensor shapes — \
             sharing an arch tag would misroute runtime dispatch (FR-EX-08)"
        );
        for sibling in [
            "denoise",                 // DeepFilterNet3
            "gtcrn",                   // GTCRN
            "nsnet2",                  // Microsoft DNS baseline
            "rnnoise",                 // Xiph RNNoise
            "storm",                   // StoRM diffusion enhancement
            "metricgan_plus",          // MetricGAN+
            "mp_senet_dns",            // MP-SENet
            "frcrn",                   // FRCRN
            "facebook_denoiser",       // Meta Denoiser
            "mossformer2_ss_16k",      // MossFormer2
            "sepformer",               // SpeechBrain SepFormer
            "conv_tasnet",             // Asteroid ConvTasNet
            "demucs",                  // Demucs / HT-Demucs
            "bs_roformer",             // BS-Roformer
            "tiger_separator",         // TIGER
            "musicgen",                // Meta MusicGen
            "magnet_small_10secs",     // Meta MAGNeT
            "melodyflow_t24_30secs",   // Meta MelodyFlow
            "audiogen_medium",         // Meta AudioGen
            "jasco_400m_chords_drums", // Meta JASCO
            "stable_audio_open_small", // Stability Stable Audio Open
            "ace_step",                // ACE-Step
            "bigvgan",                 // BigVGAN vocoder
            "hifigan_vocoder",         // HiFi-GAN vocoder
        ] {
            assert_ne!(
                ARCH, sibling,
                "audiosr (latent-diffusion audio super-resolution / bandwidth \
                 extension) and `{sibling}` are distinct arches — sharing an arch tag \
                 would misroute the runtime dispatch (FR-EX-08)"
            );
        }
        // The category tag is deliberately NOT the enhancement cohort's.
        assert_ne!(
            CATEGORY, "enhancement",
            "AudioSR synthesises new spectral content above the input cutoff; the \
             `enhancement` cohort removes additive noise within a fixed bandwidth. \
             Collapsing the two would advertise a bandwidth extender as a denoiser"
        );
        assert_ne!(CATEGORY, "denoise");
    }
}
