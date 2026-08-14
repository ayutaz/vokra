//! **AudioLDM 2** (`cvssp/audioldm2` / `cvssp/audioldm2-large`,
//! CC-BY-NC-SA-4.0 — T4 tier NonCommercialShareAlike) — text-to-audio
//! **latent-diffusion** generator runtime binder for the `audioldm2`
//! converter arch (Wave 6 2026-08-14 audit follow-up, loud-partial
//! landing per RMVPE / MusicGen / Charsiu / MOSS-Audio-Tokenizer
//! precedent).
//!
//! # Primary source
//!
//! - HF model card (base, ~350M UNet):
//!   <https://huggingface.co/cvssp/audioldm2>
//! - HF model card (large, ~750M UNet, sibling — same arch tag,
//!   different `vokra.model.name`): <https://huggingface.co/cvssp/audioldm2-large>
//! - CVSSP reference implementation:
//!   <https://github.com/haoheliu/AudioLDM2>
//! - Paper: Liu et al., *"AudioLDM 2: Learning Holistic Audio
//!   Generation with Self-supervised Pretraining"*, ICML 2024
//!   (arXiv:2308.05734).
//! - Weight license: **CC-BY-NC-SA-4.0** per CVSSP primary source
//!   (README + paper §Ethics), doubly restrictive (NC gate + SA
//!   cascade) — `docs/license-audit.md` §3.1 row 400 = ☑
//!   Research-only 2026-08-01 yousan. HF cardData shows the looser
//!   `license: cc-by-nc-4.0` tag; the converter follows the more
//!   restrictive CVSSP-owned primary source per the Fish-Speech
//!   precedent (silently downgrading the SA cascade would strip
//!   downstream re-license obligations).
//!
//! # Architecture (transcribed from primary sources)
//!
//! ```text
//! text prompt (UTF-8 string)
//!   -> frozen T5-base text encoder                   ← **loud-partial**
//!        (google-t5/t5-base — HF transformers `T5EncoderModel`; no
//!         reusable primitive in `vokra_ops` today; the follow-up wave
//!         lands a T5-base body or a shared `t5_text_encode` op.
//!         Shared with sibling MusicGen family — a future landing of
//!         the T5-base primitive unblocks both AudioLDM 2 and MusicGen
//!         simultaneously.)
//!   -> CLAP text encoder                             ← **loud-partial**
//!        (LAION CLAP text tower — <https://github.com/LAION-AI/CLAP>,
//!         joint audio-text embedding; no reusable primitive in
//!         `vokra_ops` today. AudioLDM 2's paper §3 fuses T5 tokens +
//!         CLAP text embedding + a GPT-2 audio-caption LM's "language
//!         of audio" tokens as the U-Net cross-attention condition —
//!         this triple fusion is the paper's novel contribution vs
//!         AudioLDM 1 which used only CLAP.)
//!   -> latent-diffusion U-Net                        ← **loud-partial**
//!        (2D U-Net operating on a VAE-compressed latent, time-embed
//!         conditioning + cross-attention over the fused T5 + CLAP +
//!         GPT-2 tokens; `diffusers.AudioLDM2UNet2DConditionModel`.
//!         The DDIM / DPM++ ODE step is available via
//!         `vokra_ops::flow_sampler` — this is **ONE landed anchor**
//!         the loud-partial message cites so the reader knows the
//!         sampler composition anchor.)
//!   -> VAE decoder                                   ← **loud-partial**
//!        (mel-latent → mel spectrogram —
//!         `diffusers.AutoencoderKL`. The audio VAE is distinct from
//!         the SD family's image VAE; the follow-up wave lands the
//!         audio-latent inversion path.)
//!   -> HiFi-GAN vocoder                              ← **primitive
//!      (mel → waveform, 16 kHz PCM output)             partially exists**
//!        (Vokra has a native `hifigan_generator` op landed in M3-07,
//!         but the tensor-name walk from the AudioLDM 2 vocoder
//!         state_dict prefixes to that op's inputs has NOT been
//!         pinned pending the manifest fetch. See `crates/vokra-models/src/hifigan/`.)
//!   -> PCM (mono f32, 16 kHz)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`AudioLdm2Variant`] enum discrimination via
//!     [`AudioLdm2Variant::from_name`] (Base + Large — the two
//!     variants the converter emits with matching NAME_BASE / NAME_LARGE
//!     stamps).
//!   - [`AudioLdm2::from_gguf`] with strict `vokra.model.arch ==
//!     "audioldm2"` validation + name-based variant dispatch. Unknown
//!     AudioLDM 2 family variants (`audioldm2-music` / `-music-665k`
//!     shipped in the CVSSP release family but not bound here yet)
//!     fail with a distinct error naming the "follow-up wave adds
//!     Music + Music-665k to `AudioLdm2Variant` enum" so a reader
//!     diagnosing the mismatch has an anchor.
//!   - [`AudioLdm2Config::from_gguf`] with primary-source constant
//!     fallback (the AudioLDM 2 converter does NOT currently stamp the
//!     `vokra.audioldm2.*` chunk group — only arch / name / category /
//!     upstream_hf / provenance — so a *strict* reader would refuse
//!     already-produced AudioLDM 2 GGUFs; primary-source axes are
//!     transcribed from HF `config.json` on the base variant, so
//!     fallback does not fabricate values). A future converter sub-wave
//!     that starts stamping the chunk group upgrades this to real-
//!     stamped reads per-key with no runtime code change — mirror of
//!     the Sortformer / PyanNet / MusicGen / Conv-TasNet fallback
//!     precedent).
//!   - [`AudioLdm2Weights::from_gguf`] with a floor of non-empty
//!     tensor count enforced loud (a GGUF that carries zero tensors is
//!     refused rather than silently running an all-zero forward —
//!     FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::NonCommercialShareAlike`] per the AudioLDM 2
//!     converter's stamped `cc-by-nc-sa-4.0` — T4 tier, fail-closed at
//!     the runtime compliance gate M2-13 for both NC + SA obligations).
//!
//! - **Loud-partial (this WP)**: [`AudioLdm2::generate`] returns
//!   [`VokraError::UnsupportedOp`] naming **five** deferred pieces:
//!   1. the frozen **T5-base text encoder** forward (upstream
//!      `transformers.T5EncoderModel`; no reusable primitive in
//!      `vokra_ops` today; the follow-up wave lands a T5-base
//!      implementation or a first-class `t5_text_encode` op — shared
//!      with sibling MusicGen family);
//!   2. the **CLAP text encoder** forward (LAION CLAP text tower;
//!      no reusable primitive; the paper §3 novel triple-fusion
//!      condition depends on this);
//!   3. the **latent-diffusion U-Net** forward (2D U-Net over the
//!      VAE latent with time-embed conditioning + cross-attention over
//!      the fused text tokens) — the DDIM / DPM++ ODE step is
//!      **available via `vokra_ops::flow_sampler`** (M3-05), cited
//!      as one composition anchor;
//!   4. the **VAE decoder** forward (mel-latent → mel spectrogram —
//!      audio VAE distinct from SD image VAE);
//!   5. the **HiFi-GAN vocoder** forward (mel → 16 kHz PCM) —
//!      `crates/vokra-models/src/hifigan/` has the native op, but
//!      the tensor-name walk from the AudioLDM 2 vocoder state_dict
//!      prefixes to that op's inputs has NOT been pinned pending the
//!      manifest fetch.
//!
//! The error names **all three** primary source URLs (HF card +
//! CVSSP GitHub + paper) so a reader diagnosing this gap has exactly
//! three places to walk. **No fabricated PCM stream is ever emitted**
//! (FR-EX-08).
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / sortformer / musicgen loud-partial precedent,
//! CLAUDE.md 教訓 (a) — "loud-partial は fake-complete より honest"):
//! the surrounding scaffold + `from_gguf` arch validation +
//! `AudioLdm2Config` primary-source fallback + FR-EX-08 loud-fails
//! land today so a follow-up wave can flip the switch by (i) landing
//! the T5-base text encoder body against a real T5-base state_dict
//! (the converter already emits the T5 weights under
//! `text_encoder.*` — see the converter's tensor-name contract at
//! `crates/vokra-convert/src/models/audioldm2.rs`), (ii) landing the
//! LAION CLAP text encoder body, (iii) implementing the latent-
//! diffusion U-Net forward wired to the existing
//! `vokra_ops::flow_sampler` ODE integrator, (iv) implementing the
//! audio VAE decoder, and (v) wiring the tensor-name walk from
//! `vocoder.*` to the existing `hifigan` native op. Two of the five
//! primitives (flow_sampler + hifigan) already exist so the follow-up
//! wave is composition + three greenfield forward bodies, NOT five
//! greenfield kernels.
//!
//! # Input-validation contract (loud-partial with pre-validation)
//!
//! [`AudioLdm2::generate`] validates its inputs **before** the loud-
//! partial fire path so a caller with a legitimate bug (empty prompt,
//! non-positive duration, NaN / infinite duration) sees a specific
//! [`VokraError::InvalidArgument`] instead of the generic loud-partial
//! [`VokraError::UnsupportedOp`]. This ordering is a matter of
//! diagnostic quality — the loud-partial gate is not an "escape from
//! input validation" excuse. Only well-shaped inputs reach the
//! loud-partial fire path.
//!
//! # Sibling family distinctness (music-generation neighbourhood)
//!
//! [`ARCH`] = `"audioldm2"` is **deliberately distinct** from every
//! sibling music-generation arch tag — this binder's loud-error
//! disambiguation table sits atop the same distinctness contract the
//! converter documents:
//!
//! - `musicgen` — Meta MusicGen family (**autoregressive** transformer
//!   LM over EnCodec RVQ tokens; sampler surface = AR decode); AudioLDM 2
//!   is **latent diffusion** (sampler surface = DDIM / DPM++ ODE step);
//! - `magnet_small_10secs` / `magnet_medium_30secs` — Meta MAGNeT
//!   (non-autoregressive masked-LM parallel decoding);
//! - `melodyflow_t24_30secs` — Meta MelodyFlow (DiT flow-matching
//!   music editing);
//! - `audiogen_medium` — Meta AudioGen (sound-effects AR LM);
//! - `jasco_400m_chords_drums` — Meta JASCO (chord/drum-conditioned
//!   AR LM);
//! - `stable_audio_open_small` — Stability AI Stable Audio Open (also
//!   latent diffusion but distinct topology + T5 conditioner without
//!   the CLAP + GPT-2 audio-caption LM triple fusion);
//! - `ace_step` — ACE-Step (chunked-AR);
//! - `bs_roformer` — Band-Split Roformer (source separation, not
//!   generation).
//!
//! Silently sharing arch would let runtime dispatch mis-route an
//! AudioLDM 2 checkpoint onto an AR / masked-LM / different-diffusion
//! loader — the sampler surfaces are completely different and the
//! tensor-name walks would fail with a downstream missing-tensor error
//! instead of a specific arch-mismatch message. FR-EX-08 forbids the
//! silent shape misroute across generation families.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME_BASE`] /
//! [`NAME_LARGE`] / [`CATEGORY`] — same rule the sibling BF16 pass-
//! through binders (`sortformer_diar_4spk_v1` / `pyannote` / `snac` /
//! `hifigan` / `beat_this` / `mt3` / `musicgen` / `conv_tasnet` /
//! `sepformer`) use so `vokra-models` does not gain a dependency edge
//! onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! AudioLDM 2 ships safetensors + PyTorch pickle upstream; this
//! runtime **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02).
//! If the upstream release ships pickle only, callers pre-flatten +
//! merge offline via `tools/parity/audioldm2_prepare_checkpoint.py`
//! (a thin wrapper over `bin_to_safetensors.py`; an uv-managed
//! Python 3.12 sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]` — not part of the runtime), mirroring
//! the SpeechT5-HiFi-GAN / Sortformer / MusicGen bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/audioldm2.rs`. See module docstring
// for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model audioldm2` or
/// `vokra-cli convert --model audioldm2-large`.
///
/// **Shared across every AudioLDM 2 family variant** (base / large /
/// music / music-665k) — the family shares the latent-diffusion +
/// U-Net + VAE + HiFi-GAN + T5 + CLAP + GPT-2 topology, only model
/// dims + optional variant-specific weights differ. Variant
/// discrimination happens via [`AudioLdm2Variant::from_name`] against
/// `vokra.model.name`.
///
/// Deliberately distinct from every sibling music-generation arch —
/// see the module docstring "Sibling family distinctness" section for
/// the FR-EX-08 rationale.
pub const ARCH: &str = "audioldm2";

/// Expected `vokra.model.name` value for the **Base** (~350M UNet)
/// variant — matches the `huggingface.co/cvssp/audioldm2` upstream
/// slug + the converter's `NAME` constant.
pub const NAME_BASE: &str = "audioldm2";

/// Expected `vokra.model.name` value for the **Large** (~750M UNet)
/// variant — matches the `huggingface.co/cvssp/audioldm2-large`
/// upstream slug + the converter's `LARGE_NAME` constant.
pub const NAME_LARGE: &str = "audioldm2-large";

/// Expected `vokra.model.category` value — AudioLDM 2 is text-to-audio
/// (music / general audio) generator, sharing the `music` taxonomy
/// tag with sibling MusicGen / MAGNeT / MelodyFlow / AudioGen /
/// JASCO / Stable Audio families per the 2026-07-30 scope expansion
/// (`[[project-scope-expansion-2026-07-30]]`).
pub const CATEGORY: &str = "music";

// GGUF key names for the topology chunk group. The AudioLDM 2
// converter (as of 2026-08-01) does NOT stamp these — the binder
// reads them with per-variant primary-source-transcribed fallback.
// A future converter sub-wave that adds the stamps upgrades this
// reader to real-stamped reads per-key with no runtime code change.

/// `vokra.audioldm2.sample_rate` — output PCM sample rate (Hz).
/// Primary-source default: 16000 (16 kHz — CVSSP AudioLDM 2 README).
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.audioldm2.sample_rate";
/// `vokra.audioldm2.num_train_timesteps` — diffusion training
/// horizon (number of forward-noise steps the model was trained
/// with). Primary-source default: 1000 (diffusers LDM family default,
/// used by cvssp/audioldm2 per `scheduler/scheduler_config.json`).
pub const GGUF_KEY_NUM_TRAIN_TIMESTEPS: &str = "vokra.audioldm2.num_train_timesteps";

/// Primary-source anchor for the AudioLDM 2 **Base** HF release.
/// Cited in the loud-partial error so a reader diagnosing the gap
/// knows the definitive artifact source.
pub const PRIMARY_SOURCE_HF: &str = "https://huggingface.co/cvssp/audioldm2";

/// Primary-source anchor for the AudioLDM 2 **Large** sibling HF
/// release. Cited alongside [`PRIMARY_SOURCE_HF`] when the bound
/// variant is Large so the reader gets the right HF card.
pub const PRIMARY_SOURCE_HF_LARGE: &str = "https://huggingface.co/cvssp/audioldm2-large";

/// Primary-source anchor for the CVSSP reference implementation
/// (MIT code — the tensor-name walk anchor). Cited in the loud-
/// partial error so a reader knows the code reference.
pub const PRIMARY_SOURCE_GITHUB: &str = "https://github.com/haoheliu/AudioLDM2";

/// Paper anchor (Liu et al. ICML 2024) — cited alongside the HF card
/// + CVSSP repo so a reader has the theoretical context as well.
pub const PRIMARY_SOURCE_ARXIV: &str = "https://arxiv.org/abs/2308.05734";

// ---------------------------------------------------------------------------
// AudioLdm2Variant — the variant discriminator (name-based; every
// AudioLDM 2 variant shares the `audioldm2` arch tag).
// ---------------------------------------------------------------------------

/// Which AudioLDM 2 family variant a GGUF represents. Determined by
/// [`AudioLdm2Variant::from_name`] against `vokra.model.name`.
///
/// **This WP is scoped to Base + Large only**, matching the two
/// variants the converter currently emits. Sibling variants (`-music` /
/// `-music-665k` — CVSSP release-family variants that would ride the
/// same shared converter helper) map to `None` in
/// [`from_name`](Self::from_name) so [`AudioLdm2::from_gguf`] can emit
/// a specific "converter emits it but runtime enum extension pending"
/// error rather than a generic bind failure. Mirror of
/// [`crate::musicgen::MusicGenVariant`] posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioLdm2Variant {
    /// `cvssp/audioldm2` — ~350M-parameter UNet base variant.
    Base,
    /// `cvssp/audioldm2-large` — ~750M-parameter UNet large sibling.
    Large,
}

impl AudioLdm2Variant {
    /// Discriminates an AudioLDM 2 variant from `vokra.model.name`.
    /// Returns `None` for AudioLDM 2 family variants not bound in
    /// this WP (`audioldm2-music` / `audioldm2-music-665k`) so
    /// [`AudioLdm2::from_gguf`] can emit a specific "follow-up wave
    /// adds Music variants to the enum" error, and for any string
    /// that isn't an AudioLDM 2 family name at all.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            NAME_BASE => Some(Self::Base),
            NAME_LARGE => Some(Self::Large),
            _ => None,
        }
    }

    /// Canonical `vokra.model.name` string for this variant. Matches
    /// the upstream HF slug + the converter's `NAME` / `LARGE_NAME`
    /// constants.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Base => NAME_BASE,
            Self::Large => NAME_LARGE,
        }
    }

    /// The primary-source HF card URL for this variant. Cited in the
    /// loud-partial error so a reader lands on the correct HF card.
    #[must_use]
    pub const fn primary_source_hf(self) -> &'static str {
        match self {
            Self::Base => PRIMARY_SOURCE_HF,
            Self::Large => PRIMARY_SOURCE_HF_LARGE,
        }
    }
}

// ---------------------------------------------------------------------------
// AudioLdm2Config — the composite topology axes read from the
// `vokra.audioldm2.*` chunk group, with primary-source constant
// fallback (the AudioLDM 2 converter does not currently stamp this
// chunk group — the fallback is honest because the axes are well-
// established primary-source constants; a future converter sub-wave
// that adds the stamps upgrades this reader to real-stamped reads
// seamlessly). Mirror of [`crate::musicgen::MusicGenConfig::from_gguf`]
// + [`crate::conv_tasnet::ConvTasnetConfig`] hold posture.
//
// **Axis-scope discipline**: this WP holds only the two axes that
// are verifiable from the CVSSP README + the diffusers scheduler
// default (`sample_rate = 16000`, `num_train_timesteps = 1000`).
// Additional axes named in the loud-partial error (n_mels /
// latent_channels / unet_in_channels / flan_t5_dim / clap_text_dim /
// vae_scaling_factor / ...) are pending the primary-source manifest
// fetch — mirror of the Conv-TasNet / MusicGen posture where a
// documented deviation from the RMVPE / ReDimNet strict-read
// precedent is honest for post-launch chunk-group additions.
// Fabricating unverified axis values would violate CLAUDE.md 教訓 (a).
// ---------------------------------------------------------------------------

/// AudioLDM 2 hyperparameters as they ride the `vokra.audioldm2.*`
/// chunk group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key — a GGUF that never carried the chunk
/// still loads with the upstream-verified defaults transcribed from
/// the CVSSP README + diffusers scheduler config. Every numeric axis
/// is `u32` in the GGUF (mirror of MusicGen / Conv-TasNet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioLdm2Config {
    /// Output PCM sample rate (Hz). Primary-source default: 16000
    /// (CVSSP AudioLDM 2 README).
    pub sample_rate: u32,
    /// Diffusion training horizon — the number of forward-noise steps
    /// the model was trained with. Primary-source default: 1000
    /// (diffusers LDM family default, used by cvssp/audioldm2 per
    /// `scheduler/scheduler_config.json`). This is the maximum step
    /// count the DDIM / DPM++ sampler can address; the follow-up wave
    /// exposes a caller-supplied `num_inference_steps` (typically
    /// 200 for AudioLDM 2 per the paper's evaluation setup) as an
    /// argument to `generate` distinct from this train-time constant.
    pub num_train_timesteps: u32,
}

impl Default for AudioLdm2Config {
    /// The primary-source-transcribed CVSSP AudioLDM 2 defaults. Used
    /// by [`AudioLdm2::from_gguf`] via [`Self::cvssp_base_default`]
    /// as the fallback path when the topology chunk group is absent.
    fn default() -> Self {
        Self::cvssp_base_default()
    }
}

impl AudioLdm2Config {
    /// The CVSSP AudioLDM 2 primary-source axes as a `const` —
    /// transcribed from the CVSSP README + diffusers scheduler config.
    /// Shared between the Base and Large variants (both use 16 kHz
    /// output at the same 1000-step training horizon; only the U-Net
    /// scale differs).
    #[must_use]
    pub const fn cvssp_base_default() -> Self {
        Self {
            sample_rate: 16000,
            num_train_timesteps: 1000,
        }
    }

    /// Reads every `vokra.audioldm2.*` chunk from `gguf`, falling back
    /// to the primary-source defaults per absent key.
    ///
    /// The AudioLDM 2 converter does not currently stamp this chunk
    /// group (only arch / name / category / upstream_hf / provenance),
    /// so on an already-published GGUF every axis falls through to its
    /// primary-source default. A future converter sub-wave that adds
    /// the stamps upgrades this reader to real-stamped reads per-key
    /// with no runtime code change.
    ///
    /// Mirror of
    /// [`crate::musicgen::MusicGenConfig::from_gguf`] +
    /// [`crate::pyannote::PyanNetConfig::from_gguf`] — the same
    /// fallback pattern used for converters whose topology-stamp
    /// sub-wave is still queued.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::cvssp_base_default();
        Self {
            sample_rate: gguf
                .get(GGUF_KEY_SAMPLE_RATE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.sample_rate),
            num_train_timesteps: gguf
                .get(GGUF_KEY_NUM_TRAIN_TIMESTEPS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_train_timesteps),
        }
    }
}

// ---------------------------------------------------------------------------
// AudioLdm2Weights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// T5 + CLAP + U-Net + VAE + HiFi-GAN composition is deferred. Mirror
// of `MusicGenWeights` / `ConvTasnetWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an AudioLDM 2 GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a
/// valid AudioLDM 2 checkpoint; the multi-encoder bundle must carry
/// at least the VAE + U-Net + vocoder + T5 + CLAP + GPT-2 groups).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future
/// `AudioLdm2Weights::bind_t5_encoder_weights` /
/// `bind_clap_encoder_weights` / `bind_unet_weights` /
/// `bind_vae_decoder_weights` / `bind_vocoder_weights` tensor walks
/// can find their inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct AudioLdm2Weights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// T5 + CLAP + U-Net + VAE + HiFi-GAN composition wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl AudioLdm2Weights {
    /// Scans `gguf` for the AudioLDM 2 state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid AudioLDM 2 checkpoint).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(
                "audioldm2: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model audioldm2` (or \
                 `--model audioldm2-large` for the ~750M UNet sibling) against a \
                 `cvssp/audioldm2` or `cvssp/audioldm2-large` safetensors checkpoint (the \
                 upstream release ships a bundle of VAE encoder/decoder + U-Net + HiFi-GAN \
                 vocoder + T5-base text encoder + CLAP text encoder + GPT-2 audio-caption \
                 LM — every group must be present; the whole bundle is ~8.5 GB base / ~7 GB \
                 large per the converter docstring)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the T5 + CLAP + U-Net + VAE + HiFi-GAN forward wave
    /// uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// AudioLdm2 — the runtime binder handle
// ---------------------------------------------------------------------------

/// CVSSP AudioLDM 2 text-to-audio latent-diffusion generator runtime
/// binder (`cvssp/audioldm2` / `cvssp/audioldm2-large`,
/// CC-BY-NC-SA-4.0 T4 tier NonCommercialShareAlike).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`generate`](Self::generate) with a text prompt + duration to
/// obtain a `Vec<f32>` of 16 kHz PCM samples. See the module doc for
/// the current implementation-status matrix and the FR-EX-08 loud-
/// error contract on the T5 + CLAP + U-Net + VAE + HiFi-GAN
/// composition.
#[derive(Debug)]
pub struct AudioLdm2 {
    config: AudioLdm2Config,
    variant: AudioLdm2Variant,
    // The bound weights are held (real, counted) but the T5 + CLAP +
    // U-Net + VAE + HiFi-GAN composition is a follow-up wave; the
    // field is deliberately `#[allow(dead_code)]` until the composition
    // lands so a reader is not misled by an unused field. Same posture
    // as RMVPE / pyannote / mt3 / beat_this / sortformer / musicgen.
    #[allow(dead_code)]
    weights: AudioLdm2Weights,
    weight_license: LicenseClass,
}

impl AudioLdm2 {
    /// Binds an AudioLDM 2 GGUF: validates arch, discriminates the
    /// variant from `vokra.model.name`, reads the topology chunk group
    /// (with primary-source constant fallback per key), discovers
    /// tensors, and surfaces the stamped weight-license class for
    /// compliance-gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"audioldm2"` (a sibling music-generation GGUF handed
    ///   here by mistake — `musicgen` / `magnet_small_10secs` /
    ///   `melodyflow_t24_30secs` / `audiogen_medium` /
    ///   `stable_audio_open_small` / `bs_roformer` / ... — fails with
    ///   a clear message instead of a downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is absent,
    ///   or when it identifies an AudioLDM 2 family variant not bound
    ///   in this WP (`audioldm2-music` / `audioldm2-music-665k` —
    ///   would ride the same converter helper but the runtime enum
    ///   extension is a follow-up wave).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`AudioLdm2Weights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "audioldm2: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model audioldm2` or \
                     `--model audioldm2-large`? Note that the sibling music-generation arch \
                     tags — `musicgen` (Meta MusicGen autoregressive transformer LM over \
                     EnCodec RVQ tokens, sampler = AR decode), `magnet_small_10secs` / \
                     `magnet_medium_30secs` (Meta MAGNeT non-autoregressive masked-LM \
                     parallel decoding), `melodyflow_t24_30secs` (Meta MelodyFlow DiT \
                     flow-matching music editing), `audiogen_medium` (Meta AudioGen \
                     sound-effects AR LM), `jasco_400m_chords_drums` (chord/drum-conditioned \
                     AR LM), `stable_audio_open_small` (Stability AI latent diffusion, \
                     different conditioner stack), `ace_step` (chunked-AR), `bs_roformer` \
                     (source separation — not generation at all) — all live in the same \
                     music-generation neighbourhood but have completely different sampler \
                     surfaces + text-conditioner stacks. AudioLDM 2's latent-diffusion \
                     U-Net over a VAE-compressed audio latent with T5 + CLAP + GPT-2 \
                     audio-caption LM triple-fusion cross-attention condition has no \
                     analog in any sibling — silently aliasing arch would misroute the \
                     runtime dispatch, FR-EX-08.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "audioldm2: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native audioldm2 GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant discrimination via `vokra.model.name`. Every
        //    AudioLDM 2 variant shares the `audioldm2` arch tag; the
        //    name chunk is the discriminator. Unknown AudioLDM 2
        //    family variants (`audioldm2-music` / `audioldm2-music-665k`
        //    — CVSSP release-family variants that would ride the same
        //    converter helper but this WP does not bind them) get a
        //    specific "runtime enum extension pending" error rather
        //    than a generic bind failure.
        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "audioldm2: GGUF is missing `vokra.model.name` (converter did not \
                     stamp it — cannot discriminate the AudioLDM 2 family variant \
                     between `audioldm2` (base ~350M UNet) and `audioldm2-large` \
                     (large ~750M UNet))"
                        .to_owned(),
                )
            })?;
        let variant = AudioLdm2Variant::from_name(name).ok_or_else(|| {
            if name == "audioldm2-music" || name == "audioldm2-music-665k" {
                VokraError::ModelLoad(format!(
                    "audioldm2: NAME `{name}` is not yet bound in the runtime — the \
                     CVSSP release family includes it but this WP scopes to base + large \
                     (2026-08-14 audit follow-up Wave 6 task spec). A follow-up wave \
                     adds Music + Music-665k to the `AudioLdm2Variant` enum. Primary \
                     source: https://huggingface.co/cvssp/{name}."
                ))
            } else {
                VokraError::ModelLoad(format!(
                    "audioldm2: NAME `{name}` is not a recognised AudioLDM 2 family \
                     variant. Expected one of `{NAME_BASE}` or `{NAME_LARGE}`. \
                     (Was this GGUF produced by the AudioLDM 2 converter? The converter \
                     stamps `vokra.model.name` = `audioldm2` / `audioldm2-large`.)"
                ))
            }
        })?;

        // 3. Topology axes from the `vokra.audioldm2.*` chunk group
        //    (fallback-friendly — see the module doc for the
        //    AudioLDM 2 converter's stamp posture).
        let config = AudioLdm2Config::from_gguf(file);

        // 4. Load the tensor manifest with the non-emptiness gate.
        let weights = AudioLdm2Weights::from_gguf(file)?;

        // 5. Provenance surfacing — read the stamped weight-license
        //    class for compliance-gate cross-checks. The AudioLDM 2
        //    converter defaults to `NonCommercialShareAlike` per the
        //    CVSSP `cc-by-nc-sa-4.0` primary source (doubly restrictive
        //    T4 tier — NC gate + SA cascade both fail-closed). A GGUF
        //    missing the stamp reads back as `LicenseClass::Unknown`
        //    which is also fail-closed at the M2-13 compliance gate —
        //    same posture as MusicGen / Conv-TasNet / MT3 / Sortformer.
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

    /// The bound topology axes (from `vokra.audioldm2.*` chunk group
    /// with primary-source constant fallback — see module doc for the
    /// fallback rationale).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &AudioLdm2Config {
        &self.config
    }

    /// The bound AudioLDM 2 family variant.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> AudioLdm2Variant {
        self.variant
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The AudioLDM 2
    /// converter stamps `NonCommercialShareAlike` by default per the
    /// CVSSP `cc-by-nc-sa-4.0` primary source (T4 tier — doubly
    /// restrictive: NC gate + SA cascade). A GGUF missing the stamp
    /// reads back as [`LicenseClass::Unknown`] which is also fail-
    /// closed at the M2-13 compliance gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the T5 + CLAP + U-Net + VAE + HiFi-GAN forward wave
    /// uses it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Generates a `duration_secs`-length 16 kHz PCM stream conditioned
    /// on the text `prompt`.
    ///
    /// # Input validation (pre-loud-partial)
    ///
    /// Validates inputs **before** the loud-partial fire path so a
    /// caller with a legitimate bug sees a specific
    /// [`VokraError::InvalidArgument`] rather than the generic loud-
    /// partial [`VokraError::UnsupportedOp`]. The loud-partial gate
    /// is not an "escape from input validation" excuse.
    ///
    /// - Empty prompt → [`VokraError::InvalidArgument`].
    /// - Non-positive `duration_secs` (0.0 or negative) →
    ///   [`VokraError::InvalidArgument`].
    /// - Non-finite `duration_secs` (NaN or ±infinity) →
    ///   [`VokraError::InvalidArgument`].
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — AudioLDM 2's inference
    /// path requires **five** deferred pieces:
    ///
    /// 1. **Frozen T5-base text encoder forward** (upstream
    ///    `transformers.T5EncoderModel` — no reusable primitive in
    ///    `vokra_ops` today; the follow-up wave lands either a T5-base
    ///    implementation dedicated to AudioLDM 2 or a first-class
    ///    `t5_text_encode` op that other future consumers can share).
    /// 2. **CLAP text encoder forward** (LAION CLAP text tower —
    ///    <https://github.com/LAION-AI/CLAP>, joint audio-text
    ///    embedding; no reusable primitive today).
    /// 3. **Latent-diffusion U-Net forward** (2D U-Net over the VAE
    ///    latent with time-embed conditioning + cross-attention over
    ///    the fused T5 + CLAP + GPT-2 tokens) — the DDIM / DPM++ ODE
    ///    step is **available via `vokra_ops::flow_sampler`** (M3-05),
    ///    one composition anchor.
    /// 4. **VAE decoder forward** (mel-latent → mel spectrogram —
    ///    `diffusers.AutoencoderKL`, audio VAE distinct from SD image
    ///    VAE).
    /// 5. **HiFi-GAN vocoder forward** (mel → 16 kHz PCM) — Vokra has
    ///    a native `hifigan_generator` op landed in M3-07, but the
    ///    tensor-name walk from the AudioLDM 2 vocoder state_dict
    ///    prefixes to that op's inputs has NOT been pinned.
    ///
    /// The error names **all three** primary-source URLs (HF card
    /// for the bound variant + CVSSP GitHub repo + paper) so a reader
    /// diagnosing this gap has exactly three places to walk. **No
    /// fabricated PCM stream is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on input-validation failure
    ///   (see above).
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred T5 + CLAP + U-Net + VAE + HiFi-GAN composition.
    pub fn generate(&self, prompt: &str, duration_secs: f32) -> Result<Vec<f32>> {
        // Input validation must precede the loud-partial fire path so a
        // caller with a legitimate bug (empty prompt, non-positive
        // duration, NaN / infinite duration) sees a specific
        // InvalidArgument rather than the generic UnsupportedOp.
        if prompt.is_empty() {
            return Err(VokraError::InvalidArgument(
                "audioldm2 generate: prompt must not be empty (a text-to-audio \
                 generator with no prompt has no condition to diffuse against)"
                    .to_owned(),
            ));
        }
        if !duration_secs.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "audioldm2 generate: duration_secs must be finite, got {duration_secs} \
                 (NaN / ±infinity is never a legitimate PCM stream length)"
            )));
        }
        if duration_secs <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "audioldm2 generate: duration_secs must be positive, got {duration_secs} \
                 (a zero-length or negative PCM stream cannot be diffused)"
            )));
        }
        Err(generate_forward_loud_partial(
            &self.config,
            self.variant,
            prompt,
            duration_secs,
        ))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`AudioLdm2::generate`] until the T5 + CLAP + U-Net + VAE +
/// HiFi-GAN composition lands.
///
/// Names **all three** primary source URLs (HF card for the bound
/// variant + CVSSP GitHub repo + paper) so a reader diagnosing the
/// gap has exactly three places to walk. Mirrors the sortformer / MT3
/// / beat_this / RMVPE / pyannote / snac / hifigan / musicgen /
/// conv_tasnet Wave 3-5 loud-partial-message precedent — CLAUDE.md
/// 教訓 (a).
fn generate_forward_loud_partial(
    cfg: &AudioLdm2Config,
    variant: AudioLdm2Variant,
    prompt: &str,
    duration_secs: f32,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "audioldm2 generate: T5-base text encoder + CLAP text encoder + latent-diffusion \
         U-Net + VAE decoder + HiFi-GAN vocoder composition pending. What is missing is \
         (a) the frozen T5-base text encoder forward (upstream `transformers.T5EncoderModel` \
         — no reusable primitive in `vokra_ops` today; the follow-up wave lands either a \
         T5-base implementation dedicated to AudioLDM 2 or a first-class `t5_text_encode` \
         op, shared with sibling MusicGen family), (b) the CLAP text encoder forward \
         (LAION CLAP text tower — no reusable primitive today; the paper §3 novel \
         triple-fusion condition depends on both T5 + CLAP + GPT-2 audio-caption LM tokens), \
         (c) the latent-diffusion U-Net forward (2D U-Net over the VAE-compressed audio \
         latent with time-embed conditioning + cross-attention over the fused text tokens \
         — the DDIM / DPM++ ODE step is available via `vokra_ops::flow_sampler` from M3-05, \
         so the follow-up wave is a UNet forward body + sampler composition, not a \
         greenfield ODE integrator), (d) the VAE decoder forward (mel-latent → mel \
         spectrogram — audio VAE distinct from SD image VAE), and (e) the HiFi-GAN vocoder \
         forward (mel → 16 kHz PCM) — Vokra has a native `hifigan_generator` op landed in \
         M3-07 but the tensor-name walk from the AudioLDM 2 vocoder state_dict prefixes \
         has NOT been pinned pending the manifest fetch. Config: variant={variant_short}, \
         sample_rate={sample_rate}, num_train_timesteps={num_train_timesteps}. Requested \
         prompt_len={prompt_len} chars, duration_secs={duration_secs}. Primary sources: \
         {hf} + {github} + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は \
         fake-complete より honest') — no silent fabricated PCM stream ever emitted \
         (FR-EX-08).",
        variant_short = match variant {
            AudioLdm2Variant::Base => "Base",
            AudioLdm2Variant::Large => "Large",
        },
        sample_rate = cfg.sample_rate,
        num_train_timesteps = cfg.num_train_timesteps,
        prompt_len = prompt.len(),
        duration_secs = duration_secs,
        hf = variant.primary_source_hf(),
        github = PRIMARY_SOURCE_GITHUB,
        paper = PRIMARY_SOURCE_ARXIV,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the AudioLDM 2 runtime binder — variant discrimination
    //! + config round-trip + negative-space round-trip on the
    //! loud-partial gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real inference this
    //! would be `generate(...)` returning real 16 kHz PCM, but the
    //! T5 + CLAP + U-Net + VAE + HiFi-GAN composition is deferred (see
    //! the module doc + [`AudioLdm2::generate`] rustdoc). Fabricating a
    //! real-inference output would violate CLAUDE.md 教訓 (a)
    //! ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Variant discrimination**: name → enum → primary-source
    //!    HF URL.
    //! 2. **Config round-trip**: `from_gguf` reads every axis stamped
    //!    by the converter (via the fallback path today; the strict
    //!    path when a future converter sub-wave stamps the topology
    //!    chunk group).
    //! 3. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing name /
    //!    unsupported variant / empty tensor list / empty prompt /
    //!    non-positive duration / non-finite duration / unsupported
    //!    forward surface) fires at its documented surface point, in
    //!    the documented error variant.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds an AudioLDM 2 GGUF carrying the arch tag + name +
    /// category + one representative U-Net tensor. Optionally
    /// stamps the topology chunk group + `weight_license_class`.
    fn audioldm2_gguf(
        name: &str,
        cfg: Option<AudioLdm2Config>,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, name);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cfg) = cfg {
            b.add_u32(GGUF_KEY_SAMPLE_RATE, cfg.sample_rate);
            b.add_u32(GGUF_KEY_NUM_TRAIN_TIMESTEPS, cfg.num_train_timesteps);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative U-Net tensor so the non-emptiness gate
        // passes. Uses a realistic AudioLDM 2 state-dict-like name
        // (`unet.conv_in.weight` = input conv of the latent-diffusion
        // U-Net per `diffusers.AudioLDM2UNet2DConditionModel`, matching
        // the converter test's chosen sample tensor for consistency).
        b.add_tensor(
            "unet.conv_in.weight",
            GgmlType::F32,
            vec![320, 8, 3, 3],
            vec![0u8; 320 * 8 * 3 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Variant discrimination + primary-source HF URL routing
    // -----------------------------------------------------------------------

    #[test]
    fn variant_discrimination_and_primary_source_urls() {
        // Both known variant names discriminate to the enum arms.
        assert_eq!(
            AudioLdm2Variant::from_name(NAME_BASE),
            Some(AudioLdm2Variant::Base)
        );
        assert_eq!(
            AudioLdm2Variant::from_name(NAME_LARGE),
            Some(AudioLdm2Variant::Large)
        );
        // Unknown / future-variant names return None so from_gguf can
        // emit a specific "runtime enum extension pending" error.
        assert_eq!(AudioLdm2Variant::from_name("audioldm2-music"), None);
        assert_eq!(AudioLdm2Variant::from_name("audioldm2-music-665k"), None);
        assert_eq!(AudioLdm2Variant::from_name("musicgen-small"), None);
        assert_eq!(AudioLdm2Variant::from_name(""), None);

        // Round-trip: enum → name → enum.
        assert_eq!(AudioLdm2Variant::Base.name(), NAME_BASE);
        assert_eq!(AudioLdm2Variant::Large.name(), NAME_LARGE);

        // The primary-source HF URL differs between variants so the
        // loud-partial message points to the correct HF card.
        assert_eq!(
            AudioLdm2Variant::Base.primary_source_hf(),
            "https://huggingface.co/cvssp/audioldm2"
        );
        assert_eq!(
            AudioLdm2Variant::Large.primary_source_hf(),
            "https://huggingface.co/cvssp/audioldm2-large"
        );
        // The base + large URLs must NOT be identical — a copy-paste
        // regression on the enum arm would land here.
        assert_ne!(
            AudioLdm2Variant::Base.primary_source_hf(),
            AudioLdm2Variant::Large.primary_source_hf(),
            "base + large HF URLs must be distinct"
        );

        // Primary-source constants exposed as pub const so downstream
        // consumers + tests can assert-substring against them.
        assert_eq!(PRIMARY_SOURCE_HF, "https://huggingface.co/cvssp/audioldm2");
        assert_eq!(PRIMARY_SOURCE_ARXIV, "https://arxiv.org/abs/2308.05734");
    }

    // -----------------------------------------------------------------------
    // Test 2 — Config default matches primary-source CVSSP axes
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_matches_cvssp_primary_source_axes() {
        // Pin the primary-source-transcribed axes. A rename or axis-
        // value drift would land here in the same commit or fail this
        // test.
        let cfg = AudioLdm2Config::cvssp_base_default();
        assert_eq!(cfg.sample_rate, 16000, "sample_rate primary-source pin");
        assert_eq!(
            cfg.num_train_timesteps, 1000,
            "num_train_timesteps primary-source pin (diffusers LDM family default)"
        );
        // Sanity: Default matches cvssp_base_default (both must be
        // primary-source-transcribed constants; no silent divergence).
        assert_eq!(AudioLdm2Config::default(), cfg);
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf metadata round-trip (base variant, fallback
    //          config, provenance stamp present)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip_base_variant() {
        // Build a legitimate GGUF (arch + name + category + provenance
        // license class + one representative tensor). The binder must
        // bind, hold the primary-source config via fallback, surface
        // the NC-SA license class, and report at least one tensor
        // bound.
        let file = audioldm2_gguf(NAME_BASE, None, Some(LicenseClass::NonCommercialShareAlike));
        let a = AudioLdm2::from_gguf(&file).expect("valid base GGUF must bind");
        // Variant discrimination: NAME_BASE → Base.
        assert_eq!(a.variant(), AudioLdm2Variant::Base);
        // Config fallback: absent topology chunks fall through to the
        // primary-source defaults.
        assert_eq!(*a.config(), AudioLdm2Config::cvssp_base_default());
        // License-class surface: NonCommercialShareAlike per HF
        // `cc-by-nc-sa-4.0` CVSSP primary source (T4 tier doubly
        // restrictive).
        assert_eq!(
            a.weight_license(),
            LicenseClass::NonCommercialShareAlike,
            "audioldm2 converter defaults to NonCommercialShareAlike per CVSSP \
             cc-by-nc-sa-4.0 — the runtime binder must surface it so the M2-13 \
             compliance gate can enforce both NC + SA cascade"
        );
        assert!(
            a.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4 — from_gguf reads stamped topology axes (round-trip)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_reads_stamped_topology_axes() {
        // Stamp non-default axis values so we can verify the read
        // picks them up vs falling through to defaults. This test
        // pins the future-proofing invariant: when a converter sub-
        // wave adds the topology stamps, this reader picks them up
        // per-key.
        let cfg = AudioLdm2Config {
            sample_rate: 48000,
            num_train_timesteps: 500,
        };
        let file = audioldm2_gguf(
            NAME_LARGE,
            Some(cfg),
            Some(LicenseClass::NonCommercialShareAlike),
        );
        let a = AudioLdm2::from_gguf(&file).expect("valid large GGUF must bind");
        // Variant: LARGE.
        assert_eq!(a.variant(), AudioLdm2Variant::Large);
        // Stamped topology axes must round-trip exactly (no silent
        // fallback to the primary-source defaults when the chunk
        // group is present).
        assert_eq!(a.config().sample_rate, 48000);
        assert_eq!(a.config().num_train_timesteps, 500);
    }

    // -----------------------------------------------------------------------
    // Test 5 — from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `musicgen` GGUF handed to the AudioLDM 2 binder by
        // mistake must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08). AudioLDM 2's latent
        // diffusion and MusicGen's AR-over-EnCodec are completely
        // different sampler surfaces, so silent aliasing would
        // misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "musicgen");
        b.add_string(chunks::KEY_MODEL_NAME, "musicgen-small");
        b.add_tensor("some.tensor", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioLdm2::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`musicgen`") && m.contains("`audioldm2`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("latent-diffusion") || m.contains("latent diffusion"),
                    "message should call out AudioLDM 2's latent-diffusion topology so \
                     the reader knows why the arches are distinct, got `{m}`"
                );
                assert!(
                    m.contains("autoregressive"),
                    "message should mention AR as the distinguishing sampler surface \
                     for MusicGen, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch_chunk() {
        // A GGUF that carries no `vokra.model.arch` at all (e.g. a
        // hand-assembled fixture from an unrelated pipeline) must
        // fail loud rather than mis-bind.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "not-audioldm2");
        // No `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioLdm2::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("audioldm2"),
                    "message must name the audioldm2 binder so a reader knows which \
                     loader complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — from_gguf rejects missing name chunk
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_name_chunk() {
        // A GGUF that carries the arch tag but NO `vokra.model.name`
        // must fail loud — the arch tag alone cannot distinguish base
        // from large.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        // No `vokra.model.name`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioLdm2::from_gguf(&file) else {
            panic!("expected ModelLoad on missing name");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.name`"),
                    "message must call out the missing name key, got `{m}`"
                );
                assert!(
                    m.contains("audioldm2") && m.contains("audioldm2-large"),
                    "message should name both base + large variants so the reader \
                     knows the discrimination axis, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — from_gguf rejects unrecognised variant name (future
    //          variants get a specific "runtime enum extension pending"
    //          error)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_unsupported_variant() {
        // `audioldm2-music` is a CVSSP release-family variant not
        // bound in this WP; the error must call out the "runtime enum
        // extension pending" scope rather than a generic bind failure.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "audioldm2-music");
        b.add_tensor(
            "unet.conv_in.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioLdm2::from_gguf(&file) else {
            panic!("expected ModelLoad on unsupported variant");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("audioldm2-music"),
                    "message must echo the offending variant name, got `{m}`"
                );
                assert!(
                    m.contains("follow-up wave") || m.contains("runtime enum"),
                    "message must call out the runtime enum extension pending, got `{m}`"
                );
                assert!(
                    m.contains("cvssp/audioldm2-music"),
                    "message should link to the CVSSP primary source for the future \
                     variant, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — Empty tensor manifest fails loud (never binds all-zero
    //          forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + name but zero tensors — the AudioLdm2Weights
        // non-emptiness gate must fire with an FR-EX-08 clause.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_BASE);
        b.add_string("vokra.model.category", CATEGORY);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioLdm2::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                assert!(
                    m.contains("audioldm2"),
                    "message should name the converter --model slug so the reader knows \
                     how to re-produce the GGUF, got `{m}`"
                );
                // The message must name at least a representative
                // subset of the five-encoder bundle so a reader knows
                // what a legitimate GGUF must carry.
                assert!(
                    m.contains("VAE") && m.contains("U-Net"),
                    "message must call out at least VAE + U-Net so the reader knows the \
                     bundle shape, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10 — generate loud-partial names five primitives + all three
    //           primary sources + config axes + FR-EX-08 + 教訓 (a)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_loud_partial_names_five_primitives_and_primary_sources() {
        let file = audioldm2_gguf(NAME_BASE, None, Some(LicenseClass::NonCommercialShareAlike));
        let a = AudioLdm2::from_gguf(&file).unwrap();
        // Legitimate inputs so the loud-partial gate fires (not the
        // input-validation gate).
        let Err(err) = a.generate("piano melody with soft drums", 5.0) else {
            panic!("generate must loud-partial on well-shaped inputs");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("audioldm2 generate"),
                    "message must call out the audioldm2 generate surface, got `{m}`"
                );
                // All FIVE deferred pieces must be named so the follow-
                // up wave knows the composition anchors.
                assert!(
                    m.contains("T5-base") || m.contains("FLAN-T5") || m.contains("T5"),
                    "message must name the T5 text encoder piece, got `{m}`"
                );
                assert!(
                    m.contains("CLAP"),
                    "message must name the CLAP text encoder piece, got `{m}`"
                );
                assert!(
                    m.contains("U-Net") || m.contains("UNet"),
                    "message must name the latent-diffusion U-Net piece, got `{m}`"
                );
                assert!(
                    m.contains("VAE"),
                    "message must name the VAE decoder piece, got `{m}`"
                );
                assert!(
                    m.contains("HiFi-GAN") || m.contains("hifigan"),
                    "message must name the HiFi-GAN vocoder piece, got `{m}`"
                );
                // All three primary-source anchors must be cited (HF
                // card + CVSSP GitHub + arXiv paper).
                assert!(
                    m.contains("huggingface.co/cvssp/audioldm2"),
                    "message must contain the HF card URL, got `{m}`"
                );
                assert!(
                    m.contains("github.com/haoheliu/AudioLDM2"),
                    "message must contain the CVSSP GitHub repo URL, got `{m}`"
                );
                assert!(
                    m.contains("2308.05734"),
                    "message must contain the arXiv paper id, got `{m}`"
                );
                // Config axes must be echoed so the reader can cross-
                // check what topology the follow-up wave targets.
                assert!(
                    m.contains("sample_rate=16000"),
                    "sample_rate axis missing: {m}"
                );
                assert!(
                    m.contains("num_train_timesteps=1000"),
                    "num_train_timesteps axis missing: {m}"
                );
                // Request context (prompt len + duration) echoed so
                // an operator can correlate a log line to a call site.
                assert!(m.contains("prompt_len="), "prompt_len missing: {m}");
                assert!(m.contains("duration_secs="), "duration_secs missing: {m}");
                // Sampler-composition anchor (flow_sampler) named as
                // an existing primitive so the reader knows the
                // ODE-integrator half is already available.
                assert!(
                    m.contains("flow_sampler") || m.contains("DDIM"),
                    "message should cite the flow_sampler / DDIM ODE step as a landed \
                     composition anchor, got `{m}`"
                );
                // FR-EX-08 clause + honesty-rationale citations.
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                assert!(
                    m.contains("教訓 (a)") || m.contains("loud-partial は fake-complete"),
                    "message must cite CLAUDE.md 教訓 (a), got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11 — generate rejects empty prompt (input validation
    //           precedes loud-partial gate)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_rejects_empty_prompt() {
        let file = audioldm2_gguf(NAME_BASE, None, Some(LicenseClass::NonCommercialShareAlike));
        let a = AudioLdm2::from_gguf(&file).unwrap();
        let Err(err) = a.generate("", 5.0) else {
            panic!("generate must reject empty prompt with InvalidArgument, not fire loud-partial");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(
                    m.contains("prompt") && m.contains("empty"),
                    "message must call out the empty-prompt validation, got `{m}`"
                );
                // Must NOT be a loud-partial (input validation runs
                // before the loud-partial gate).
                assert!(
                    !m.contains("T5") && !m.contains("CLAP"),
                    "empty-prompt path must NOT surface the loud-partial primitive names, \
                     got `{m}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 12 — generate rejects non-positive duration
    //           (0.0 and negative)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_rejects_nonpositive_duration() {
        let file = audioldm2_gguf(NAME_BASE, None, Some(LicenseClass::NonCommercialShareAlike));
        let a = AudioLdm2::from_gguf(&file).unwrap();
        for &bad in &[0.0f32, -1.0, -0.001] {
            let Err(err) = a.generate("prompt", bad) else {
                panic!("generate must reject non-positive duration {bad} with InvalidArgument");
            };
            match err {
                VokraError::InvalidArgument(m) => {
                    assert!(
                        m.contains("duration") && m.contains("positive"),
                        "message must call out the non-positive-duration validation for \
                         {bad}, got `{m}`"
                    );
                }
                other => {
                    panic!("expected VokraError::InvalidArgument for duration {bad}, got {other:?}")
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 13 — generate rejects non-finite duration (NaN + ±infinity)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_rejects_nonfinite_duration() {
        let file = audioldm2_gguf(NAME_BASE, None, Some(LicenseClass::NonCommercialShareAlike));
        let a = AudioLdm2::from_gguf(&file).unwrap();
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let Err(err) = a.generate("prompt", bad) else {
                panic!("generate must reject non-finite duration {bad} with InvalidArgument");
            };
            match err {
                VokraError::InvalidArgument(m) => {
                    assert!(
                        m.contains("duration") && m.contains("finite"),
                        "message must call out the non-finite-duration validation for \
                         {bad}, got `{m}`"
                    );
                }
                other => {
                    panic!("expected VokraError::InvalidArgument for duration {bad}, got {other:?}")
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 14 — Missing provenance stamp falls back to Unknown
    //           (fail-closed at M2-13 compliance gate)
    // -----------------------------------------------------------------------

    #[test]
    fn missing_provenance_stamp_falls_back_to_unknown_license_class() {
        // A GGUF that carries the arch tag + one tensor but NO
        // `vokra.provenance.weight_license` chunk must fall back to
        // `LicenseClass::Unknown` — the same fail-closed posture the
        // MusicGen / Conv-TasNet / MT3 / Sortformer binders take.
        // `Unknown` is refused at the M2-13 compliance gate so a
        // mis-stamped artifact cannot slip past commercial-mode
        // dispatch.
        let file = audioldm2_gguf(NAME_BASE, None, None);
        let a = AudioLdm2::from_gguf(&file).expect("bind without provenance");
        assert_eq!(
            a.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed at M2-13)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 15 — Arch tag distinct from sibling music-generation arches
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_music_generation_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. Every sibling music-generation
        // arch tag MUST NOT collide with ours — the sampler surfaces
        // + text-conditioner stacks are fundamentally different across
        // families.
        assert_eq!(ARCH, "audioldm2");
        assert_eq!(NAME_BASE, "audioldm2");
        assert_eq!(NAME_LARGE, "audioldm2-large");
        assert_eq!(CATEGORY, "music");
        // Direct string comparisons against sibling music-generation
        // arch tags so a future sibling rename would land here in the
        // same commit or fail this test.
        assert_ne!(
            ARCH, "musicgen",
            "audioldm2 (latent diffusion) and musicgen (AR-over-EnCodec) are distinct \
             music-gen arches — sharing arch tag would misroute the runtime dispatch \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "magnet_small_10secs",
            "audioldm2 (latent diffusion) and MAGNeT (non-autoregressive masked-LM) are \
             distinct music-gen arches — sharing arch tag would misroute the runtime \
             dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "melodyflow_t24_30secs",
            "audioldm2 (latent diffusion) and MelodyFlow (DiT flow-matching editing) \
             are distinct music-gen arches — sharing arch tag would misroute the runtime \
             dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "audiogen_medium",
            "audioldm2 (latent diffusion) and AudioGen (sound-effects AR LM) are distinct \
             music-gen arches — sharing arch tag would misroute the runtime dispatch \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "jasco_400m_chords_drums",
            "audioldm2 (latent diffusion, text condition) and JASCO (chord/drum-conditioned \
             AR LM) are distinct music-gen arches — sharing arch tag would misroute the \
             runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "stable_audio_open_small",
            "audioldm2 (T5 + CLAP + GPT-2 triple-fusion condition) and Stable Audio Open \
             (T5-only condition) are distinct latent-diffusion arches — sharing arch tag \
             would misroute the runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "ace_step",
            "audioldm2 (latent diffusion) and ACE-Step (chunked-AR) are distinct music-gen \
             arches — sharing arch tag would misroute the runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "bs_roformer",
            "audioldm2 (music generation) and BS-Roformer (music source separation) are \
             completely different tasks — sharing arch tag would misroute the runtime \
             dispatch (FR-EX-08)"
        );
    }
}
