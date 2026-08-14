//! **Meta MusicGen** (`facebook/musicgen-{small,medium}`,
//! CC-BY-NC-4.0 — T4 tier) — text-to-music autoregressive transformer
//! LM runtime binder (2026-08-14 audit follow-up Wave 5, first
//! **music generation** runtime binder in the tree).
//!
//! # Primary source
//!
//! - HF model card (Small, 300M params):
//!   <https://huggingface.co/facebook/musicgen-small>
//! - HF model card (Medium, 1.5B params):
//!   <https://huggingface.co/facebook/musicgen-medium>
//! - AudioCraft reference implementation (MIT code, non-commercial
//!   weights): <https://github.com/facebookresearch/audiocraft>
//!   (`audiocraft/models/musicgen.py` +
//!   `audiocraft/models/lm.py` — the `MusicGen` handle + `LMModel`
//!   autoregressive transformer LM).
//! - Paper: Copet et al., *"Simple and Controllable Music Generation"*,
//!   NeurIPS 2023 (arXiv:2306.05284).
//! - Weight license: **CC-BY-NC-4.0** (Meta AudioCraft weight policy;
//!   the code layer at `github.com/facebookresearch/audiocraft` is MIT
//!   but the trained weights are non-commercial). `docs/license-audit.md`
//!   §3.1 row 399 = ☑ Research-only 2026-08-01 yousan for medium;
//!   the small variant is covered by the shared **X-Codec-2 T4
//!   precedent** (2026-07-28).
//!
//! # Architecture (transcribed from primary sources)
//!
//! ```text
//! text prompt (UTF-8 string)
//!   -> frozen T5-base text encoder                   ← **loud-partial**
//!        (google-t5/t5-base — HF transformers `T5EncoderModel`,
//!         encoder-only, no reusable primitive in `vokra_ops` today;
//!         the follow-up wave lands a T5-base implementation or a
//!         first-class `t5_text_encode` op — see the loud-partial
//!         message for the concrete plan.)
//!   -> autoregressive transformer LM with            ← **loud-partial**
//!      4-codebook delay pattern and text-conditioned
//!      cross-attention to T5 tokens
//!        (Copet et al. Algorithm 1 — the MusicGen-specific
//!         "delay pattern" interleave over the 4 codebook streams,
//!         plus cross-attention to the T5-encoded prompt tokens; the
//!         single-codebook AR LMs already in the tree — cosyvoice2 /
//!         moshi / voxtral — cover neither the delay pattern nor the
//!         text-encoder cross-attention.)
//!   -> EnCodec RVQ decode (4 codebooks, 50 Hz frame ← **primitive
//!      rate, 32 kHz PCM output)                        exists**
//!        (available via `vokra_ops::encodec_rvq_decode` — this is the
//!         ONE landed anchor the loud-partial message cites so the
//!         reader knows the composition anchor.)
//!   -> PCM (mono f32, 32 kHz)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`MusicGenVariant`] enum discrimination via
//!     [`MusicGenVariant::from_name`] (Small / Medium only — the
//!     task-scoped variant set).
//!   - [`MusicGen::from_gguf`] with strict `vokra.model.arch == "musicgen"`
//!     validation + name-based variant dispatch. Unknown MusicGen family
//!     variants (`musicgen-large` / `musicgen-melody` shipped by the
//!     converter but not bound here yet) fail with a distinct error
//!     naming the "follow-up wave adds Large + Melody to `MusicGenVariant`
//!     enum" so a reader diagnosing the mismatch has an anchor.
//!   - [`MusicGenConfig::from_gguf`] with primary-source constant
//!     fallback **per variant** (the MusicGen converter does NOT
//!     currently stamp the `vokra.musicgen.*` chunk group — only arch /
//!     name / category / upstream_hf / provenance — so a *strict*
//!     reader would refuse the already-published
//!     `huggingface.co/facebook/musicgen-{small,medium}` GGUFs. Primary
//!     source is well-established (HF `config.json` on both variants +
//!     AudioCraft code + paper), so fallback does not fabricate axes;
//!     a future converter sub-wave that starts stamping the chunk group
//!     upgrades this to real-stamped reads seamlessly — mirror of the
//!     Sortformer / PyanNet fallback pattern).
//!   - [`MusicGenWeights::from_gguf`] with a floor of non-empty tensor
//!     count enforced loud (a GGUF that carries zero tensors is refused
//!     rather than silently running an all-zero forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::NonCommercial`] per the MusicGen converter's
//!     stamped `cc-by-nc-4.0` — T4 tier, fail-closed at the runtime
//!     compliance gate M2-13).
//!
//! - **Loud-partial (this WP)**: [`MusicGen::generate`] returns
//!   [`VokraError::UnsupportedOp`] naming **three** deferred pieces:
//!   1. the frozen T5-base text encoder forward (upstream
//!      `transformers.T5EncoderModel`; no reusable primitive in
//!      `vokra_ops` today; the follow-up wave lands the T5-base body or
//!      a first-class `t5_text_encode` op);
//!   2. the autoregressive transformer LM decode with the **4-codebook
//!      delay pattern** (the MusicGen-specific interleave over 4 RVQ
//!      streams) + text-conditioned **cross-attention** to the T5-encoded
//!      prompt tokens (the single-codebook AR LMs already in the tree —
//!      cosyvoice2 / moshi / voxtral — cover neither the delay pattern
//!      nor the text-encoder cross-attention);
//!   3. the EnCodec RVQ decode from the 4 codebook streams to 32 kHz PCM
//!      — **available via `vokra_ops::encodec_rvq_decode`**, cited as
//!      the one landed piece so a reader diagnosing this gap knows the
//!      composition anchor.
//!
//! The error names the **three primary source URLs** (HF card for the
//! bound variant + AudioCraft repo + paper), the config axes echoed
//! (`variant`, `d_model`, `num_layers`, `n_heads`, `num_codebooks`,
//! `sample_rate_hz`), and the prompt length + duration so a reader
//! diagnosing this gap has exactly three places to walk. **No fabricated
//! PCM stream is ever emitted** (FR-EX-08).
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / sortformer loud-partial precedent, CLAUDE.md 教訓
//! (a) — "loud-partial は fake-complete より honest"): the surrounding
//! scaffold + `from_gguf` chunk-group validation + `MusicGenVariant`
//! enum + FR-EX-08 loud-fails land today so a follow-up wave can flip
//! the switch by (i) landing the T5-base text encoder body against a
//! real T5-base state_dict (the converter already emits the T5 weights
//! under `text_encoder.*` — see the converter's tensor-name contract),
//! (ii) implementing the delay-pattern + cross-attention AR transformer
//! LM decode, and (iii) wiring the composed loop to the existing
//! `vokra_ops::encodec_rvq_decode` primitive. The primitive for (iii)
//! already exists so the follow-up wave is composition + two greenfield
//! forward bodies, NOT three greenfield kernels.
//!
//! # `vokra.musicgen.*` chunk group (read here — fallback-friendly)
//!
//! The MusicGen converters
//! (`crates/vokra-convert/src/models/musicgen_small.rs` /
//! `crates/vokra-convert/src/models/musicgen_medium.rs`) currently stamp
//! only the arch / name / category / upstream_hf / provenance chunks.
//! The topology chunk group is READ by this binder but any absent key
//! falls back to the **per-variant primary-source constant** so an
//! already-published GGUF loads correctly. A future converter sub-wave
//! that adds `vokra.musicgen.*` stamps will override the fallback
//! automatically per-key with no runtime code change.
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"musicgen"`).
//!   Deliberately distinct from every sibling music-generation arch —
//!   `magnet_small_10secs` / `magnet_medium_30secs` / `melodyflow_t24_30secs`
//!   / `audiogen_medium` / `jasco_400m_chords_drums` / `audioldm2` /
//!   `stable_audio_open_small` / `ace_step` / `bs_roformer`. MusicGen is
//!   the **AR-LM** family (autoregressive transformer LM over EnCodec RVQ
//!   tokens); silently sharing an arch tag with a sibling music-gen
//!   family would mis-route runtime dispatch to a wrong-shape forward
//!   (MAGNeT is non-autoregressive masked-LM; MelodyFlow is DiT flow-
//!   matching; AudioLDM2 / Stable-Audio-Open are diffusion; ACE-Step is
//!   yet another decoder topology; BS-Roformer is source-separation, not
//!   generation). FR-EX-08 forbids the silent-wrong shape mismatch.
//! - `vokra.model.name` (`String`): [`NAME_SMALL`] (`"musicgen-small"`)
//!   or [`NAME_MEDIUM`] (`"musicgen-medium"`) — the variant discriminator
//!   under the shared `musicgen` arch tag. Sibling variants shipped by
//!   the converter but NOT yet bound here (`musicgen-large` /
//!   `musicgen-melody`) fail loudly with a "follow-up wave" anchor.
//! - `vokra.musicgen.{d_model, num_layers, n_heads, ffn_dim, vocab_size,
//!   num_codebooks, codec_frame_rate_hz, sample_rate_hz}` (`u32` each):
//!   the composite topology axes. Fallback constants transcribed from
//!   HF `config.json` on both variants (see the `DEFAULT_*` constants
//!   for the primary-source anchors).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03 / M2-13) can classify the
//!   artifact without re-inspecting the safetensors provenance. The
//!   MusicGen converters stamp `NonCommercial` by default per the HF
//!   cards' `license: cc-by-nc-4.0` — a caller who legitimately holds
//!   the weight under a distinct SPDX overrides at
//!   `vokra-cli convert --license <spdx>` and the stamped class re-
//!   derives via `LicenseClass::from_license_str`.
//!
//! # Cross-crate constant duplication (mirror of the converters'
//! [`ARCH`] / [`NAME_SMALL`] / [`NAME_MEDIUM`] / topology keys) — same
//! rule the sibling BF16 pass-through binders (`sortformer` / `pyannote`
//! / `snac` / `hifigan` / `beat_this` / `mt3`) use so `vokra-models`
//! does not gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`, `vokra-core →
//! GGUF reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! MusicGen ships safetensors + PyTorch pickle upstream; this runtime
//! **never** touches ONNX (FR-LD-05 / NFR-DS-02). If the upstream release
//! ships pickle only, callers pre-flatten offline via
//! `tools/parity/musicgen_medium_prepare_checkpoint.py` (a thin wrapper
//! over `bin_to_safetensors.py`; an uv-managed Python 3.12 sidecar per
//! memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]` —
//! not part of the runtime), mirroring the SpeechT5-HiFi-GAN /
//! Sortformer / Charsiu bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/musicgen_small.rs` +
// `crates/vokra-convert/src/models/musicgen_medium.rs`. See module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model musicgen-{small|medium}`.
///
/// **Shared across every MusicGen family variant** (small / medium /
/// large / melody / stereo-*) — the family shares the AR-LM +
/// text-conditioned cross-attention + delay-pattern EnCodec topology,
/// only the model dims + optional chroma-conditioning frontend
/// (melody-only) differ. Variant discrimination happens via
/// [`MusicGenVariant::from_name`] against `vokra.model.name`.
///
/// Deliberately distinct from every sibling music-generation arch —
/// `magnet_small_10secs` / `magnet_medium_30secs` (non-autoregressive
/// masked-LM; MAGNeT sibling), `melodyflow_t24_30secs` (DiT flow-
/// matching editing backbone), `audiogen_medium` (sound-effects LM),
/// `jasco_400m_chords_drums` (chord/drum-conditioned AR LM),
/// `audioldm2` (latent diffusion), `stable_audio_open_small` (latent
/// diffusion), `ace_step` (chunked-AR), `bs_roformer` (source-
/// separation — not generation at all). Silently sharing an arch tag
/// with any of them would mis-route the runtime dispatch to a wrong-
/// shape forward — FR-EX-08.
pub const ARCH: &str = "musicgen";

/// Expected `vokra.model.name` value for the **Small** (300M params)
/// variant — matches the `huggingface.co/facebook/musicgen-small`
/// upstream slug + the converter's `NAME` constant.
pub const NAME_SMALL: &str = "musicgen-small";

/// Expected `vokra.model.name` value for the **Medium** (1.5B params)
/// variant — matches the `huggingface.co/facebook/musicgen-medium`
/// upstream slug + the converter's `NAME` constant.
pub const NAME_MEDIUM: &str = "musicgen-medium";

/// `vokra.musicgen.d_model` — transformer LM hidden dim (per-variant).
/// Primary-source defaults: 1024 (small) / 1536 (medium).
pub const GGUF_KEY_D_MODEL: &str = "vokra.musicgen.d_model";
/// `vokra.musicgen.num_layers` — transformer LM depth (per-variant).
/// Primary-source defaults: 24 (small) / 48 (medium).
pub const GGUF_KEY_NUM_LAYERS: &str = "vokra.musicgen.num_layers";
/// `vokra.musicgen.n_heads` — multi-head attention head count
/// (per-variant). Primary-source defaults: 16 (small) / 24 (medium).
/// `head_dim = d_model / n_heads = 64` for both variants (a coincidence
/// of the family's design that keeps head_dim stable across scale-ups).
pub const GGUF_KEY_N_HEADS: &str = "vokra.musicgen.n_heads";
/// `vokra.musicgen.ffn_dim` — feedforward inner dimension (per-variant).
/// Primary-source defaults: 4096 (small) / 6144 (medium). Both variants
/// use the "4× hidden" AudioCraft convention.
pub const GGUF_KEY_FFN_DIM: &str = "vokra.musicgen.ffn_dim";
/// `vokra.musicgen.vocab_size` — per-codebook token vocabulary size.
/// Shared across variants: 2048 (the EnCodec RVQ codebook size, one
/// entry per codebook — the LM emits `num_codebooks` streams each of
/// this vocab size).
pub const GGUF_KEY_VOCAB_SIZE: &str = "vokra.musicgen.vocab_size";
/// `vokra.musicgen.num_codebooks` — number of RVQ codebook streams the
/// LM emits per frame. Shared across variants: 4 (the EnCodec 32 kHz
/// codec configuration MusicGen bundles).
pub const GGUF_KEY_NUM_CODEBOOKS: &str = "vokra.musicgen.num_codebooks";
/// `vokra.musicgen.codec_frame_rate_hz` — the EnCodec 32 kHz output
/// frame rate. Shared across variants: 50 Hz.
pub const GGUF_KEY_CODEC_FRAME_RATE_HZ: &str = "vokra.musicgen.codec_frame_rate_hz";
/// `vokra.musicgen.sample_rate_hz` — the paired EnCodec sample rate.
/// Shared across variants: 32000 Hz (32 kHz).
pub const GGUF_KEY_SAMPLE_RATE_HZ: &str = "vokra.musicgen.sample_rate_hz";

// Per-variant primary-source constants transcribed from the HF model
// cards' `config.json` (fetched 2026-08-14 — CLAUDE.md
// 「ハルシネーション厳禁」). The MusicGen family fixes head_dim = 64
// across all sizes, so d_model / n_heads = 64 for every variant.

/// Small variant transformer LM hidden dim (`d_model`). Primary source:
/// `huggingface.co/facebook/musicgen-small/config.json`
/// (`decoder.hidden_size`).
pub const DEFAULT_D_MODEL_SMALL: u32 = 1024;
/// Small variant transformer LM depth (`num_hidden_layers`).
/// Primary source: `musicgen-small/config.json`
/// (`decoder.num_hidden_layers`).
pub const DEFAULT_NUM_LAYERS_SMALL: u32 = 24;
/// Small variant attention head count (`num_attention_heads`).
/// Primary source: `musicgen-small/config.json`
/// (`decoder.num_attention_heads`). `head_dim = 1024 / 16 = 64`.
pub const DEFAULT_N_HEADS_SMALL: u32 = 16;
/// Small variant feedforward inner dimension (`ffn_dim`). Primary
/// source: `musicgen-small/config.json` (`decoder.ffn_dim`). AudioCraft
/// "4× hidden" convention: `4096 = 4 × 1024`.
pub const DEFAULT_FFN_DIM_SMALL: u32 = 4096;

/// Medium variant transformer LM hidden dim (`d_model`). Primary source:
/// `huggingface.co/facebook/musicgen-medium/config.json`
/// (`decoder.hidden_size`).
pub const DEFAULT_D_MODEL_MEDIUM: u32 = 1536;
/// Medium variant transformer LM depth (`num_hidden_layers`).
/// Primary source: `musicgen-medium/config.json`
/// (`decoder.num_hidden_layers`).
pub const DEFAULT_NUM_LAYERS_MEDIUM: u32 = 48;
/// Medium variant attention head count (`num_attention_heads`).
/// Primary source: `musicgen-medium/config.json`
/// (`decoder.num_attention_heads`). `head_dim = 1536 / 24 = 64`.
pub const DEFAULT_N_HEADS_MEDIUM: u32 = 24;
/// Medium variant feedforward inner dimension (`ffn_dim`). Primary
/// source: `musicgen-medium/config.json` (`decoder.ffn_dim`). AudioCraft
/// "4× hidden" convention: `6144 = 4 × 1536`.
pub const DEFAULT_FFN_DIM_MEDIUM: u32 = 6144;

/// Shared per-codebook vocabulary size across MusicGen variants. Primary
/// source: HF `config.json` (`decoder.vocab_size`) + AudioCraft
/// `EncodecModel.quantizer.bins`. The EnCodec 32 kHz codec MusicGen
/// bundles is a 4-codebook RVQ with 2048 entries per codebook.
pub const DEFAULT_VOCAB_SIZE: u32 = 2048;

/// Number of RVQ codebook streams the LM emits per frame. Shared across
/// every MusicGen variant. Primary source: AudioCraft
/// `MusicGen(...).lm.n_q = 4`.
pub const NUM_CODEBOOKS: u32 = 4;

/// EnCodec output frame rate for the bundled 32 kHz codec (matches
/// AudioCraft `EncodecModel.frame_rate = 50`).
pub const CODEC_FRAME_RATE_HZ: u32 = 50;

/// EnCodec sample rate for the bundled 32 kHz codec (matches AudioCraft
/// `EncodecModel.sample_rate = 32000`).
pub const SAMPLE_RATE_HZ: u32 = 32_000;

/// Primary-source anchor for the **Small** variant's HF model card.
/// Cited in the loud-partial error so a reader diagnosing the gap knows
/// the definitive artifact source.
pub const PRIMARY_SOURCE_HF_CARD_SMALL: &str = "huggingface.co/facebook/musicgen-small";
/// Primary-source anchor for the **Medium** variant's HF model card.
/// Cited in the loud-partial error so a reader diagnosing the gap knows
/// the definitive artifact source.
pub const PRIMARY_SOURCE_HF_CARD_MEDIUM: &str = "huggingface.co/facebook/musicgen-medium";
/// Primary-source anchor for the AudioCraft reference repository
/// (MIT code — the tensor-name walk anchor). Cited in the loud-partial
/// error so a reader knows the code reference.
pub const PRIMARY_SOURCE_AUDIOCRAFT_REPO: &str = "github.com/facebookresearch/audiocraft";
/// Paper anchor (Copet et al. NeurIPS 2023) — cited alongside the HF
/// card + AudioCraft repo so a reader has the theoretical context as
/// well.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2306.05284";

// ---------------------------------------------------------------------------
// MusicGenVariant — the variant discriminator (name-based, not
// arch-based, because every MusicGen variant shares the `musicgen`
// arch tag).
// ---------------------------------------------------------------------------

/// Which MusicGen family variant a GGUF represents. Determined by
/// [`MusicGenVariant::from_name`] against `vokra.model.name`.
///
/// **This WP is scoped to Small + Medium only**, per the 2026-08-14
/// audit follow-up task spec ("Add runtime binder for both small (300M)
/// and medium (1.5B) variants"). Sibling variants shipped by the
/// converter but NOT bound here yet (`musicgen-large` /
/// `musicgen-melody`) map to `None` in [`from_name`](Self::from_name)
/// so [`MusicGen::from_gguf`] can emit a specific "converter exists but
/// runtime enum extension pending in this WP" error rather than a
/// generic bind failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicGenVariant {
    /// `facebook/musicgen-small` — 300M-parameter LM decoder
    /// (`d_model=1024`, `num_layers=24`, `n_heads=16`, `ffn_dim=4096`).
    Small,
    /// `facebook/musicgen-medium` — 1.5B-parameter LM decoder
    /// (`d_model=1536`, `num_layers=48`, `n_heads=24`, `ffn_dim=6144`).
    Medium,
}

impl MusicGenVariant {
    /// Discriminates a MusicGen variant from `vokra.model.name`. Returns
    /// `None` for MusicGen family variants that exist as converters but
    /// are not bound in the runtime yet (`musicgen-large` /
    /// `musicgen-melody`) so [`MusicGen::from_gguf`] can emit a specific
    /// "follow-up wave adds Large + Melody to the enum" error, and for
    /// any string that isn't a MusicGen family name at all.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            NAME_SMALL => Some(Self::Small),
            NAME_MEDIUM => Some(Self::Medium),
            _ => None,
        }
    }

    /// Canonical `vokra.model.name` string for this variant. Matches
    /// the upstream HF slug + the converter's `NAME` constant.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Small => NAME_SMALL,
            Self::Medium => NAME_MEDIUM,
        }
    }

    /// The primary-source HF card URL for this variant. Cited in the
    /// loud-partial error.
    #[must_use]
    pub const fn primary_source_hf_card(self) -> &'static str {
        match self {
            Self::Small => PRIMARY_SOURCE_HF_CARD_SMALL,
            Self::Medium => PRIMARY_SOURCE_HF_CARD_MEDIUM,
        }
    }

    /// Primary-source-transcribed axes for this variant as a const
    /// [`MusicGenConfig`]. Used by [`MusicGenConfig::from_gguf`] as the
    /// per-key fallback when the topology chunk group is absent (the
    /// current converter's default).
    #[must_use]
    pub const fn default_config(self) -> MusicGenConfig {
        match self {
            Self::Small => MusicGenConfig {
                variant: Self::Small,
                d_model: DEFAULT_D_MODEL_SMALL,
                num_layers: DEFAULT_NUM_LAYERS_SMALL,
                n_heads: DEFAULT_N_HEADS_SMALL,
                ffn_dim: DEFAULT_FFN_DIM_SMALL,
                vocab_size: DEFAULT_VOCAB_SIZE,
                num_codebooks: NUM_CODEBOOKS,
                codec_frame_rate_hz: CODEC_FRAME_RATE_HZ,
                sample_rate_hz: SAMPLE_RATE_HZ,
            },
            Self::Medium => MusicGenConfig {
                variant: Self::Medium,
                d_model: DEFAULT_D_MODEL_MEDIUM,
                num_layers: DEFAULT_NUM_LAYERS_MEDIUM,
                n_heads: DEFAULT_N_HEADS_MEDIUM,
                ffn_dim: DEFAULT_FFN_DIM_MEDIUM,
                vocab_size: DEFAULT_VOCAB_SIZE,
                num_codebooks: NUM_CODEBOOKS,
                codec_frame_rate_hz: CODEC_FRAME_RATE_HZ,
                sample_rate_hz: SAMPLE_RATE_HZ,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// MusicGenConfig — the composite topology axes read from the
// `vokra.musicgen.*` chunk group, with primary-source constant fallback
// **per variant** (the MusicGen converter does not currently stamp this
// chunk group — the fallback is honest because the per-variant primary
// source is well-established; a future converter sub-wave that adds the
// stamps upgrades this reader to real-stamped reads seamlessly). Mirror
// of [`crate::sortformer_diar_4spk_v1::SortformerConfig::from_gguf`] +
// [`crate::pyannote::PyanNetConfig::from_gguf`].
// ---------------------------------------------------------------------------

/// MusicGen hyperparameters as they ride the `vokra.musicgen.*` chunk
/// group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key using [`MusicGenVariant::default_config`]
/// as the per-variant baseline — a GGUF that never carried the chunk
/// still loads with the upstream defaults transcribed from HF
/// `config.json`. Every numeric axis is `u32` in the GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicGenConfig {
    /// Which MusicGen family variant this config represents.
    pub variant: MusicGenVariant,
    /// Transformer LM hidden dim (default 1024 small / 1536 medium).
    pub d_model: u32,
    /// Transformer LM depth (default 24 small / 48 medium).
    pub num_layers: u32,
    /// Multi-head attention head count (default 16 small / 24 medium).
    /// `head_dim = d_model / n_heads = 64` for both variants.
    pub n_heads: u32,
    /// Feedforward inner dimension (default 4096 small / 6144 medium).
    /// AudioCraft "4× hidden" convention: `ffn_dim = 4 × d_model`.
    pub ffn_dim: u32,
    /// Per-codebook vocabulary size (shared: 2048).
    pub vocab_size: u32,
    /// Number of RVQ codebook streams the LM emits per frame
    /// (shared: 4).
    pub num_codebooks: u32,
    /// EnCodec output frame rate (shared: 50 Hz).
    pub codec_frame_rate_hz: u32,
    /// EnCodec sample rate (shared: 32000 Hz = 32 kHz).
    pub sample_rate_hz: u32,
}

impl MusicGenConfig {
    /// Primary-source-transcribed Small variant axes as a `const` —
    /// alias for `MusicGenVariant::Small.default_config()`.
    #[must_use]
    pub const fn v_small_default() -> Self {
        MusicGenVariant::Small.default_config()
    }

    /// Primary-source-transcribed Medium variant axes as a `const` —
    /// alias for `MusicGenVariant::Medium.default_config()`.
    #[must_use]
    pub const fn v_medium_default() -> Self {
        MusicGenVariant::Medium.default_config()
    }

    /// Reads every `vokra.musicgen.*` chunk from `gguf`, falling back to
    /// the per-variant primary-source defaults per absent key.
    ///
    /// The MusicGen converter does not currently stamp this chunk group
    /// (only arch / name / category / upstream_hf / provenance), so on
    /// an already-published GGUF every axis falls through to its
    /// primary-source default for the resolved variant. A future
    /// converter sub-wave that adds the stamps upgrades this reader to
    /// real-stamped reads per-key with no runtime code change.
    ///
    /// Mirror of
    /// [`crate::sortformer_diar_4spk_v1::SortformerConfig::from_gguf`]
    /// + [`crate::pyannote::PyanNetConfig::from_gguf`] — the same
    /// fallback pattern used for converters whose topology-stamp
    /// sub-wave is still queued.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile, variant: MusicGenVariant) -> Self {
        let default = variant.default_config();
        Self {
            variant,
            d_model: gguf
                .get(GGUF_KEY_D_MODEL)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.d_model),
            num_layers: gguf
                .get(GGUF_KEY_NUM_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_layers),
            n_heads: gguf
                .get(GGUF_KEY_N_HEADS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.n_heads),
            ffn_dim: gguf
                .get(GGUF_KEY_FFN_DIM)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.ffn_dim),
            vocab_size: gguf
                .get(GGUF_KEY_VOCAB_SIZE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.vocab_size),
            num_codebooks: gguf
                .get(GGUF_KEY_NUM_CODEBOOKS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_codebooks),
            codec_frame_rate_hz: gguf
                .get(GGUF_KEY_CODEC_FRAME_RATE_HZ)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.codec_frame_rate_hz),
            sample_rate_hz: gguf
                .get(GGUF_KEY_SAMPLE_RATE_HZ)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.sample_rate_hz),
        }
    }
}

// ---------------------------------------------------------------------------
// MusicGenWeights — bound the tensor manifest with a non-emptiness gate.
// Under the loud-partial WP the weights are counted but the T5-base
// encoder + AR LM (delay pattern + cross-attention) + EnCodec RVQ
// decode composition is deferred. Mirror of `SortformerWeights` /
// `Mt3Weights` / `BeatThisWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a MusicGen GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// MusicGen checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future
/// `MusicGenWeights::bind_t5_encoder_weights` /
/// `MusicGenWeights::bind_lm_decoder_weights` tensor walk can find its
/// inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct MusicGenWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// T5-encoder + AR-LM + EnCodec-decode composition wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl MusicGenWeights {
    /// Scans `gguf` for the MusicGen state_dict tensors. Refuses to bind
    /// if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is
    /// never a valid MusicGen checkpoint).
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
                "musicgen: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model \
                 musicgen-{small|medium}` against a `facebook/musicgen-{small,medium}` \
                 safetensors checkpoint (the upstream release ships a bundle of LM \
                 decoder + T5-base text encoder + EnCodec RVQ codec — every group \
                 must be present)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder / decoder / codec-decode forward wave
    /// uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load-time shape gate — validates that at least one bound tensor
    /// has an axis matching `config.d_model`. Under the current landing
    /// this is a **soft** gate (mismatch is silently ignored) because
    /// the T5-encoder + AR-LM + EnCodec-decode tensor-name walk has not
    /// yet been pinned pending the follow-up wave's manifest fetch —
    /// a hard shape assertion today would fail against every
    /// legitimate future manifest.
    ///
    /// The follow-up wave will replace this soft accessor with a hard
    /// pin against the primary-source-verified tensor-name walk
    /// (mirror of `pyannote::PyanNetWeights::verify_core_shapes`).
    ///
    /// Kept as a `#[must_use]` accessor so the read is deliberate.
    #[must_use]
    pub fn matches_config(&self, config: &MusicGenConfig) -> bool {
        let d = config.d_model as usize;
        self.tensors
            .iter()
            .any(|(_, dims)| dims.iter().any(|&x| x == d))
    }
}

// ---------------------------------------------------------------------------
// MusicGen — the runtime binder handle
// ---------------------------------------------------------------------------

/// Meta MusicGen text-to-music autoregressive transformer LM runtime
/// binder (`facebook/musicgen-{small,medium}`, CC-BY-NC-4.0 T4 tier).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`generate`](Self::generate) with a text prompt + duration to obtain
/// a `Vec<f32>` of 32 kHz PCM samples. See the module doc for the
/// current implementation-status matrix and the FR-EX-08 loud-error
/// contract on the T5-base + AR-LM + EnCodec-decode composition.
#[derive(Debug)]
pub struct MusicGen {
    config: MusicGenConfig,
    variant: MusicGenVariant,
    // The bound weights are held (real, counted) but the T5-encoder +
    // AR-LM + EnCodec-decode composition is a follow-up wave; the field
    // is deliberately `#[allow(dead_code)]` until the composition lands
    // so a reader is not misled by an unused field. Same posture as
    // RMVPE / pyannote / mt3 / beat_this / sortformer.
    #[allow(dead_code)]
    weights: MusicGenWeights,
    weight_license: LicenseClass,
}

impl MusicGen {
    /// Binds a MusicGen GGUF: validates arch, discriminates the variant
    /// from `vokra.model.name`, reads the topology chunk group (with
    /// per-variant primary-source constant fallback per key), discovers
    /// tensors, and surfaces the stamped weight-license class for
    /// compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"musicgen"` (a sibling music-generation GGUF handed to us
    ///   by mistake — `magnet_small_10secs` / `melodyflow_t24_30secs` /
    ///   `audiogen_medium` / … — fails with a clear message instead of a
    ///   downstream missing-tensor).
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is absent, or
    ///   when it identifies a MusicGen family variant not bound in this
    ///   WP (`musicgen-large` / `musicgen-melody` — converters exist but
    ///   the runtime enum extension is a follow-up wave).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`MusicGenWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "musicgen: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model musicgen-{{small|medium}}`? \
                     Note that the sibling music-generation arch tags — \
                     `magnet_small_10secs` / `magnet_medium_30secs` (Meta AudioCraft \
                     non-autoregressive masked-LM), `melodyflow_t24_30secs` (Meta \
                     AudioCraft DiT flow-matching editing), `audiogen_medium` (Meta \
                     AudioCraft sound-effects LM), `jasco_400m_chords_drums` \
                     (chord/drum-conditioned AR LM), `audioldm2` / \
                     `stable_audio_open_small` (latent diffusion), `ace_step` \
                     (chunked-AR), `bs_roformer` (source-separation) — all live in the \
                     same music-generation neighbourhood but have completely different \
                     forward topologies; MusicGen's autoregressive transformer LM with \
                     4-codebook delay pattern + T5 text-encoder cross-attention has no \
                     analog in any sibling and silently aliasing arch would misroute \
                     the runtime dispatch, FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "musicgen: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native musicgen GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant discrimination via `vokra.model.name`. Every
        //    MusicGen variant shares the `musicgen` arch tag; the name
        //    chunk is the discriminator. Unknown MusicGen family
        //    variants (`musicgen-large` / `musicgen-melody` — the
        //    converter ships them but this WP does not bind them) get
        //    a specific "runtime enum extension pending" error rather
        //    than a generic bind failure.
        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "musicgen: GGUF is missing `vokra.model.name` (converter did not \
                     stamp it — cannot discriminate the MusicGen family variant \
                     between `musicgen-small` and `musicgen-medium`)"
                        .to_owned(),
                )
            })?;
        let variant = MusicGenVariant::from_name(name).ok_or_else(|| {
            if name == "musicgen-large" {
                VokraError::ModelLoad(format!(
                    "musicgen: NAME `{name}` is not yet bound in the runtime — the \
                     converter exists but this WP scopes to small + medium (2026-08-14 \
                     audit follow-up task spec). A follow-up wave adds Large + Melody \
                     to the `MusicGenVariant` enum. Primary source: \
                     huggingface.co/facebook/{name}."
                ))
            } else if name == "musicgen-melody" {
                VokraError::ModelLoad(format!(
                    "musicgen: NAME `{name}` is not yet bound in the runtime — the \
                     converter exists but this WP scopes to small + medium (2026-08-14 \
                     audit follow-up task spec). A follow-up wave adds Large + Melody \
                     to the `MusicGenVariant` enum. Melody also needs a chroma \
                     conditioning frontend in addition to the LM. Primary source: \
                     huggingface.co/facebook/{name}."
                ))
            } else {
                VokraError::ModelLoad(format!(
                    "musicgen: NAME `{name}` is not a recognised MusicGen family \
                     variant. Expected one of `{NAME_SMALL}` or `{NAME_MEDIUM}`. \
                     (Was this GGUF produced by a MusicGen converter? The converters \
                     stamp `vokra.model.name` = `musicgen-small` / `musicgen-medium` \
                     / `musicgen-large` / `musicgen-melody`.)"
                ))
            }
        })?;

        // 3. Topology axes from the `vokra.musicgen.*` chunk group
        //    (fallback-friendly — see the module doc for the MusicGen
        //    converter's stamp posture).
        let config = MusicGenConfig::from_gguf(file, variant);

        // 4. Load the tensor manifest with the non-emptiness gate.
        let weights = MusicGenWeights::from_gguf(file)?;

        // 5. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The MusicGen
        //    converters default to `NonCommercial` per the HF cards'
        //    `license: cc-by-nc-4.0`; a caller override at `--license
        //    <spdx>` re-derives the class. Missing provenance falls
        //    back to `Unknown` which is fail-closed at the M2-13
        //    compliance gate — same posture as MT3 / Sortformer.
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

    /// The bound topology axes (from `vokra.musicgen.*` chunk group with
    /// per-variant primary-source constant fallback).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &MusicGenConfig {
        &self.config
    }

    /// The bound MusicGen family variant.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> MusicGenVariant {
        self.variant
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The MusicGen converters
    /// stamp `NonCommercial` by default per the HF cards' `license:
    /// cc-by-nc-4.0` (T4 tier — fail-closed at the M2-13 compliance
    /// gate; owner must pass `--allow-noncommercial` to publish and the
    /// runtime refuses commercial-mode load). A GGUF missing the stamp
    /// reads back as [`LicenseClass::Unknown`] which is also
    /// fail-closed.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder / decoder / codec-decode forward wave
    /// uses it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Generates a `duration_secs`-length 32 kHz PCM stream conditioned
    /// on the text `prompt`.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — MusicGen's inference path
    /// requires **three** deferred pieces:
    ///
    /// 1. **Frozen T5-base text encoder forward**: the upstream release
    ///    freezes a `google-t5/t5-base` encoder for text conditioning
    ///    (HF transformers `T5EncoderModel`); no reusable primitive
    ///    exists in `vokra_ops` today. The follow-up wave lands either
    ///    a T5-base implementation dedicated to MusicGen or a first-
    ///    class `t5_text_encode` op that other future consumers can
    ///    share.
    /// 2. **Autoregressive transformer LM decode with 4-codebook delay
    ///    pattern + text-conditioned cross-attention**: Copet et al.
    ///    Algorithm 1 describes the MusicGen-specific "delay pattern"
    ///    interleave across the 4 RVQ codebook streams, plus the
    ///    cross-attention over the T5-encoded prompt tokens. The
    ///    single-codebook autoregressive LMs already in the tree
    ///    (`cosyvoice2` / `moshi` / `voxtral`) cover neither the delay
    ///    pattern nor the text-encoder cross-attention.
    /// 3. **EnCodec RVQ decode**: available via
    ///    `vokra_ops::encodec_rvq_decode` — this is the **ONE landed
    ///    anchor** the loud-partial message cites so a reader
    ///    diagnosing this gap knows the composition anchor.
    ///
    /// The error names **three** primary source URLs (HF card for the
    /// bound variant + AudioCraft repo + paper) so a reader diagnosing
    /// this gap has exactly three places to walk. **No fabricated PCM
    /// stream is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred T5-encoder + AR-LM + EnCodec-decode composition.
    pub fn generate(&self, prompt: &str, duration_secs: f32) -> Result<Vec<f32>> {
        // Bind unused args so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future real
        // implementation will consume both.
        let _ = (prompt, duration_secs);
        Err(generate_forward_loud_partial(
            &self.config,
            self.variant,
            prompt,
            duration_secs,
        ))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`MusicGen::generate`] until the T5-encoder + AR-LM + EnCodec-decode
/// composition lands.
///
/// Names **all three** primary source URLs (HF card for the bound
/// variant + AudioCraft repo + paper) so a reader diagnosing the gap
/// has exactly three places to walk. Mirrors the sortformer / MT3 /
/// beat_this / RMVPE / pyannote / snac / hifigan Wave 3-4 loud-partial-
/// message precedent — CLAUDE.md 教訓 (a).
fn generate_forward_loud_partial(
    cfg: &MusicGenConfig,
    variant: MusicGenVariant,
    prompt: &str,
    duration_secs: f32,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "musicgen generate: T5-base text encoder forward + autoregressive transformer \
         LM decode (with 4-codebook delay pattern + text-conditioned cross-attention) \
         + EnCodec RVQ decode composition pending. What is missing is (a) the frozen \
         T5-base text encoder forward (upstream `transformers.T5EncoderModel` — no \
         reusable primitive in `vokra_ops` today; the follow-up wave lands either a \
         T5-base implementation dedicated to MusicGen or a first-class \
         `t5_text_encode` op), (b) the autoregressive transformer LM decode with the \
         MusicGen-specific 4-codebook delay pattern (Copet et al. Algorithm 1 — the \
         interleave across the 4 RVQ codebook streams) plus text-conditioned \
         cross-attention over the T5-encoded prompt tokens (the single-codebook AR \
         LMs already in the tree — cosyvoice2 / moshi / voxtral — cover neither the \
         delay pattern nor the text-encoder cross-attention), and (c) the EnCodec RVQ \
         decode step — this is available via `vokra_ops::encodec_rvq_decode` (the ONE \
         landed anchor of the composition; the follow-up wave wires the AR LM output \
         onto this primitive). Config: variant={variant_short}, d_model={d_model}, \
         num_layers={num_layers}, n_heads={n_heads}, num_codebooks={num_codebooks}, \
         sample_rate_hz={sample_rate_hz}. Requested prompt_len={prompt_len} chars, \
         duration_secs={duration_secs}. Primary sources: {hf_card} + {audiocraft_repo} \
         + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
         より honest') — no silent fabricated PCM stream ever emitted (FR-EX-08).",
        variant_short = match variant {
            MusicGenVariant::Small => "Small",
            MusicGenVariant::Medium => "Medium",
        },
        d_model = cfg.d_model,
        num_layers = cfg.num_layers,
        n_heads = cfg.n_heads,
        num_codebooks = cfg.num_codebooks,
        sample_rate_hz = cfg.sample_rate_hz,
        prompt_len = prompt.len(),
        duration_secs = duration_secs,
        hf_card = variant.primary_source_hf_card(),
        audiocraft_repo = PRIMARY_SOURCE_AUDIOCRAFT_REPO,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the MusicGen runtime binder — variant discrimination +
    //! per-variant config round-trip + negative-space round-trip on the
    //! loud-partial gates.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real inference this
    //! would be `generate(...)` returning real 32 kHz PCM, but the
    //! T5-base + AR-LM (delay pattern + cross-attention) + EnCodec-
    //! decode composition is deferred (see the module doc +
    //! [`MusicGen::generate`] rustdoc). Fabricating a real-inference
    //! output would violate CLAUDE.md 教訓 (a) ("loud-partial は
    //! fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Variant discrimination**: name → enum → per-variant default
    //!    config.
    //! 2. **Config round-trip**: `from_gguf` reads every axis stamped by
    //!    the converter (via the fallback path today; the strict path
    //!    when a future converter sub-wave stamps the topology chunk
    //!    group).
    //! 3. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / missing name / unsupported
    //!    variant / empty tensor list / unsupported forward surface)
    //!    fires at its documented surface point, in the documented
    //!    error variant.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a MusicGen GGUF carrying the arch tag + name + one
    /// representative LM decoder tensor whose outer dim matches
    /// `d_model`. The topology chunk group is optionally stamped
    /// (`stamp_topology = true`) — when omitted the runtime binder
    /// falls back to the per-variant primary-source defaults per key.
    ///
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn musicgen_gguf(
        name: &str,
        cfg: MusicGenConfig,
        stamp_topology: bool,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, name);
        if stamp_topology {
            b.add_u32(GGUF_KEY_D_MODEL, cfg.d_model);
            b.add_u32(GGUF_KEY_NUM_LAYERS, cfg.num_layers);
            b.add_u32(GGUF_KEY_N_HEADS, cfg.n_heads);
            b.add_u32(GGUF_KEY_FFN_DIM, cfg.ffn_dim);
            b.add_u32(GGUF_KEY_VOCAB_SIZE, cfg.vocab_size);
            b.add_u32(GGUF_KEY_NUM_CODEBOOKS, cfg.num_codebooks);
            b.add_u32(GGUF_KEY_CODEC_FRAME_RATE_HZ, cfg.codec_frame_rate_hz);
            b.add_u32(GGUF_KEY_SAMPLE_RATE_HZ, cfg.sample_rate_hz);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative LM decoder tensor so the non-emptiness gate
        // passes and the shape-consistency accessor has something to
        // walk. The `d_model` dim is deliberately at axis 0 so
        // `matches_config` returns true. The tensor name mirrors the
        // upstream `MusicgenForConditionalGeneration` decoder Q
        // projection.
        let d = cfg.d_model as u64;
        b.add_tensor(
            "decoder.model.decoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Variant default configs match primary-source HF config.json axes
    // -----------------------------------------------------------------------

    #[test]
    fn variant_default_configs_match_primary_source_hf_config_json_axes() {
        // Pin the Small axes transcribed from
        // huggingface.co/facebook/musicgen-small/config.json. A rename
        // or axis-value drift would land here in the same commit or
        // fail this test.
        let small = MusicGenConfig::v_small_default();
        assert_eq!(small.variant, MusicGenVariant::Small);
        assert_eq!(small.d_model, 1024);
        assert_eq!(small.num_layers, 24);
        assert_eq!(small.n_heads, 16);
        assert_eq!(small.ffn_dim, 4096);
        assert_eq!(small.vocab_size, 2048);
        assert_eq!(small.num_codebooks, 4);
        assert_eq!(small.codec_frame_rate_hz, 50);
        assert_eq!(small.sample_rate_hz, 32_000);

        // Pin the Medium axes transcribed from
        // huggingface.co/facebook/musicgen-medium/config.json.
        let medium = MusicGenConfig::v_medium_default();
        assert_eq!(medium.variant, MusicGenVariant::Medium);
        assert_eq!(medium.d_model, 1536);
        assert_eq!(medium.num_layers, 48);
        assert_eq!(medium.n_heads, 24);
        assert_eq!(medium.ffn_dim, 6144);
        assert_eq!(medium.vocab_size, 2048);
        assert_eq!(medium.num_codebooks, 4);
        assert_eq!(medium.codec_frame_rate_hz, 50);
        assert_eq!(medium.sample_rate_hz, 32_000);

        // MusicGen family design invariant: `head_dim = 64` for every
        // variant (a deliberate choice that keeps the attention kernel
        // stable across scale-ups). A future variant that violates
        // this would need to be added deliberately, not by accidental
        // silent misconfiguration.
        assert_eq!(small.d_model / small.n_heads, 64, "small: head_dim = 64");
        assert_eq!(medium.d_model / medium.n_heads, 64, "medium: head_dim = 64");

        // AudioCraft "4× hidden" FFN convention invariant.
        assert_eq!(
            small.ffn_dim,
            4 * small.d_model,
            "small: ffn_dim = 4 × d_model"
        );
        assert_eq!(
            medium.ffn_dim,
            4 * medium.d_model,
            "medium: ffn_dim = 4 × d_model"
        );

        // Variant discrimination via from_name matches the enum arms.
        assert_eq!(
            MusicGenVariant::from_name(NAME_SMALL),
            Some(MusicGenVariant::Small)
        );
        assert_eq!(
            MusicGenVariant::from_name(NAME_MEDIUM),
            Some(MusicGenVariant::Medium)
        );
        // Sibling variants (converters exist but not bound here).
        assert_eq!(MusicGenVariant::from_name("musicgen-large"), None);
        assert_eq!(MusicGenVariant::from_name("musicgen-melody"), None);
        // Random arbitrary string.
        assert_eq!(MusicGenVariant::from_name("not-musicgen"), None);
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf Small variant full topology chunk-group round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_small_round_trips_stamped_chunk_group() {
        let cfg = MusicGenConfig::v_small_default();
        let file = musicgen_gguf(
            NAME_SMALL,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let mg = MusicGen::from_gguf(&file).expect("valid Small GGUF must bind");
        assert_eq!(mg.variant(), MusicGenVariant::Small);
        // Config round-trip — every stamped axis reads back into the
        // same MusicGenConfig value (converter follow-up sub-wave
        // path).
        assert_eq!(*mg.config(), cfg);
        assert_eq!(mg.config().d_model, 1024);
        assert_eq!(mg.config().num_layers, 24);
        // NC weight license is the primary-source default per the HF
        // card (`license: cc-by-nc-4.0`) — the runtime must surface it
        // verbatim from the provenance chunk. The M2-13 compliance gate
        // refuses this artifact in commercial mode (T4 tier —
        // `--allow-noncommercial` opt-in required).
        assert_eq!(mg.weight_license(), LicenseClass::NonCommercial);
        assert!(mg.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf Medium variant full topology chunk-group round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_medium_round_trips_stamped_chunk_group() {
        let cfg = MusicGenConfig::v_medium_default();
        let file = musicgen_gguf(
            NAME_MEDIUM,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let mg = MusicGen::from_gguf(&file).expect("valid Medium GGUF must bind");
        assert_eq!(mg.variant(), MusicGenVariant::Medium);
        assert_eq!(*mg.config(), cfg);
        // Sanity: Medium axes differ from Small — a variant mix-up
        // would land here.
        assert_eq!(mg.config().d_model, 1536);
        assert_eq!(mg.config().num_layers, 48);
        assert_eq!(mg.config().n_heads, 24);
        assert_eq!(mg.config().ffn_dim, 6144);
        assert_ne!(mg.config().d_model, DEFAULT_D_MODEL_SMALL);
        assert_ne!(mg.config().num_layers, DEFAULT_NUM_LAYERS_SMALL);
        assert_eq!(mg.weight_license(), LicenseClass::NonCommercial);
    }

    // -----------------------------------------------------------------------
    // 4. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `magnet_small_10secs` GGUF handed to the MusicGen binder by
        // mistake must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08). Both `magnet_small_10secs`
        // and `musicgen` live in the music-generation family but have
        // completely different forward topologies (MAGNeT is non-
        // autoregressive masked-LM; MusicGen is autoregressive
        // transformer LM), so silent aliasing would misroute.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "magnet_small_10secs");
        b.add_string(chunks::KEY_MODEL_NAME, NAME_SMALL);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MusicGen::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`magnet_small_10secs`") && m.contains("`musicgen`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("4-codebook delay pattern"),
                    "message should disambiguate MusicGen's AR-LM + delay-pattern topology \
                     to help the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. from_gguf rejects unsupported variant (musicgen-large) with
    //    "pending enum extension" anchor
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_unsupported_variant_large_pending_enum_extension() {
        // A `musicgen-large` GGUF handed to this WP's binder must fail
        // loud with the "converter exists but runtime enum extension
        // pending" anchor so the reader knows the follow-up wave path.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "musicgen-large");
        b.add_tensor(
            "decoder.model.decoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MusicGen::from_gguf(&file) else {
            panic!("expected ModelLoad on unsupported variant");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("musicgen-large"),
                    "message must name the unsupported variant, got `{m}`"
                );
                assert!(
                    m.contains("follow-up wave"),
                    "message must anchor the reader on the follow-up wave path, got `{m}`"
                );
                assert!(
                    m.contains("MusicGenVariant"),
                    "message must name the enum being extended, got `{m}`"
                );
                assert!(
                    m.contains("huggingface.co/facebook/musicgen-large"),
                    "message must name the primary source URL for the unsupported variant, \
                     got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Same check for `musicgen-melody` — the melody variant needs
        // a chroma-conditioning frontend on top of the LM, and the
        // error message must call that out.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "musicgen-melody");
        b.add_tensor(
            "decoder.model.decoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MusicGen::from_gguf(&file) else {
            panic!("expected ModelLoad on melody variant");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("musicgen-melody"),
                    "message must name the melody variant, got `{m}`"
                );
                assert!(
                    m.contains("chroma"),
                    "message must call out the additional chroma conditioning frontend, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6. Empty tensor manifest fails loud (never binds all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + name + full chunk group but zero tensors —
        // the MusicGenWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_SMALL);
        b.add_u32(GGUF_KEY_D_MODEL, DEFAULT_D_MODEL_SMALL);
        b.add_u32(GGUF_KEY_NUM_LAYERS, DEFAULT_NUM_LAYERS_SMALL);
        b.add_u32(GGUF_KEY_N_HEADS, DEFAULT_N_HEADS_SMALL);
        b.add_u32(GGUF_KEY_FFN_DIM, DEFAULT_FFN_DIM_SMALL);
        b.add_u32(GGUF_KEY_VOCAB_SIZE, DEFAULT_VOCAB_SIZE);
        b.add_u32(GGUF_KEY_NUM_CODEBOOKS, NUM_CODEBOOKS);
        b.add_u32(GGUF_KEY_CODEC_FRAME_RATE_HZ, CODEC_FRAME_RATE_HZ);
        b.add_u32(GGUF_KEY_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MusicGen::from_gguf(&file) else {
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
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. generate returns UnsupportedOp with primary-source anchors +
    //    all three URLs (task-hint requirement)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_loud_partial_returns_unsupported_op_with_primary_source_urls() {
        let cfg = MusicGenConfig::v_small_default();
        let file = musicgen_gguf(
            NAME_SMALL,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let mg = MusicGen::from_gguf(&file).unwrap();
        // A legitimate prompt + duration — the loud-partial gate must
        // fire on the composition surface, not on pre-generate arg
        // validation.
        let Err(err) = mg.generate("a jazz piano solo", 5.0) else {
            panic!("generate must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("musicgen generate"),
                    "message must call out the musicgen generate surface, got `{m}`"
                );
                // The three deferred pieces MUST all be named so the
                // follow-up wave has an unambiguous work anchor.
                assert!(
                    m.contains("T5"),
                    "message must name the T5-base text encoder deferred piece, got `{m}`"
                );
                assert!(
                    m.contains("delay pattern"),
                    "message must name the MusicGen-specific delay pattern, got `{m}`"
                );
                assert!(
                    m.contains("encodec_rvq_decode"),
                    "message must name the ONE landed anchor (`vokra_ops::encodec_rvq_decode`), \
                     got `{m}`"
                );
                // All three primary source URLs must be cited — task
                // hint requires this.
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD_SMALL),
                    "message must contain the Small HF card URL substring \
                     ({PRIMARY_SOURCE_HF_CARD_SMALL}), got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_AUDIOCRAFT_REPO),
                    "message must contain the AudioCraft repo URL substring \
                     ({PRIMARY_SOURCE_AUDIOCRAFT_REPO}), got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_PAPER),
                    "message must contain the paper URL substring \
                     ({PRIMARY_SOURCE_PAPER}), got `{m}`"
                );
                // Every config axis must be echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(m.contains("variant=Small"), "variant axis missing: {m}");
                assert!(m.contains("d_model=1024"), "d_model axis missing: {m}");
                assert!(m.contains("num_layers=24"), "num_layers axis missing: {m}");
                assert!(m.contains("n_heads=16"), "n_heads axis missing: {m}");
                assert!(
                    m.contains("num_codebooks=4"),
                    "num_codebooks axis missing: {m}"
                );
                assert!(
                    m.contains("sample_rate_hz=32000"),
                    "sample_rate_hz axis missing: {m}"
                );
                // The prompt length + duration must be echoed so a
                // caller can cross-check the request.
                assert!(
                    m.contains("prompt_len=17"),
                    "prompt_len (17 = len(\"a jazz piano solo\")) missing: {m}"
                );
                assert!(m.contains("duration_secs=5"), "duration_secs missing: {m}");
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08 no-silent-fabrication clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }

        // Same assertion against a Medium GGUF — the message must swap
        // the HF card URL to the medium variant's card.
        let cfg = MusicGenConfig::v_medium_default();
        let file = musicgen_gguf(
            NAME_MEDIUM,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let mg = MusicGen::from_gguf(&file).unwrap();
        let Err(err) = mg.generate("classical string quartet", 10.0) else {
            panic!("generate must loud-partial for medium");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("variant=Medium"),
                    "medium variant axis missing: {m}"
                );
                assert!(
                    m.contains("d_model=1536"),
                    "medium d_model axis missing: {m}"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD_MEDIUM),
                    "medium variant's message must contain the Medium HF card URL substring \
                     ({PRIMARY_SOURCE_HF_CARD_MEDIUM}), got `{m}`"
                );
                // The AudioCraft repo + paper are shared across
                // variants — still cited.
                assert!(
                    m.contains(PRIMARY_SOURCE_AUDIOCRAFT_REPO),
                    "message must contain the AudioCraft repo URL substring, got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_PAPER),
                    "message must contain the paper URL substring, got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 8. Default weight license is NonCommercial (T4 tier fail-closed)
    // -----------------------------------------------------------------------

    #[test]
    fn default_weight_license_stamps_noncommercial_t4_tier() {
        // The MusicGen converter's DEFAULT_LICENSE_SPDX is
        // `cc-by-nc-4.0` → LicenseClass::NonCommercial. The runtime
        // must surface this verbatim so the M2-13 compliance gate
        // refuses commercial-mode load (T4 tier — the same fail-
        // closed posture as X-Codec-2 precedent 2026-07-28 +
        // Sortformer 2026-08-04).
        let cfg = MusicGenConfig::v_small_default();
        let file = musicgen_gguf(
            NAME_SMALL,
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let mg = MusicGen::from_gguf(&file).expect("bind");
        assert_eq!(
            mg.weight_license(),
            LicenseClass::NonCommercial,
            "the MusicGen converter defaults to NonCommercial per the HF card's \
             `license: cc-by-nc-4.0` — the runtime binder must surface it so the \
             M2-13 compliance gate can refuse commercial-mode load (T4 tier)"
        );
        // Missing provenance stamp falls back to Unknown (also
        // fail-closed at the gate).
        let file_no_license = musicgen_gguf(NAME_SMALL, cfg, /*stamp_topology=*/ false, None);
        let mg_no_license =
            MusicGen::from_gguf(&file_no_license).expect("bind without license stamp");
        assert_eq!(
            mg_no_license.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed)"
        );
    }

    // -----------------------------------------------------------------------
    // 9. Arch tag is stable and distinct from sibling music-generation
    //    arches (reciprocal defense — sortformer sibling test mirror)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_music_generation_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. The sibling music-generation arch
        // tags MUST NOT collide with ours — they live in the same
        // taxonomy neighbourhood but have completely different forward
        // topologies.
        assert_eq!(ARCH, "musicgen");
        assert_eq!(NAME_SMALL, "musicgen-small");
        assert_eq!(NAME_MEDIUM, "musicgen-medium");
        // Direct string comparisons against sibling arch tags to
        // document the "which sibling should NOT be aliased" contract
        // at test time (a future rename of any sibling arch would
        // land here in the same commit or fail this test).
        assert_ne!(
            ARCH, "magnet_small_10secs",
            "musicgen (AR LM) and magnet_small_10secs (non-AR masked-LM) share the \
             music-generation taxonomy but have completely different forward topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "magnet_medium_30secs",
            "musicgen and magnet_medium_30secs — same rationale as small — mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "melodyflow_t24_30secs",
            "musicgen (AR LM) and melodyflow_t24_30secs (DiT flow-matching editing) \
             both generate music but have completely different forward topologies — \
             mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "audiogen_medium",
            "musicgen (text-to-music) and audiogen_medium (sound-effects LM) share \
             the AR-LM structure but produce different modalities — mis-route \
             (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 10. Config from_gguf falls back to per-variant primary-source
    //     defaults on absent chunks
    // -----------------------------------------------------------------------

    #[test]
    fn config_from_gguf_falls_back_to_primary_source_defaults_when_chunk_group_absent() {
        // The MusicGen converter does NOT currently stamp the
        // `vokra.musicgen.*` chunk group (only arch / name / category /
        // upstream_hf / provenance). An already-published GGUF must
        // still load — the fallback path reads the per-variant
        // primary-source constants transcribed from HF config.json.
        // Mirror of SortformerConfig::from_gguf fallback pattern.
        let cfg_small = MusicGenConfig::v_small_default();
        let file_small = musicgen_gguf(
            NAME_SMALL,
            cfg_small,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let mg_small =
            MusicGen::from_gguf(&file_small).expect("chunk-free Small GGUF must bind via fallback");
        // Every axis fell through to its primary-source Small default —
        // the loader returns the same values as v_small_default().
        assert_eq!(mg_small.config().d_model, DEFAULT_D_MODEL_SMALL);
        assert_eq!(mg_small.config().num_layers, DEFAULT_NUM_LAYERS_SMALL);
        assert_eq!(mg_small.config().n_heads, DEFAULT_N_HEADS_SMALL);
        assert_eq!(mg_small.config().ffn_dim, DEFAULT_FFN_DIM_SMALL);
        assert_eq!(mg_small.config().vocab_size, DEFAULT_VOCAB_SIZE);
        assert_eq!(mg_small.config().num_codebooks, NUM_CODEBOOKS);
        assert_eq!(mg_small.config().codec_frame_rate_hz, CODEC_FRAME_RATE_HZ);
        assert_eq!(mg_small.config().sample_rate_hz, SAMPLE_RATE_HZ);

        // Same for Medium — falls back to the Medium primary-source
        // defaults, distinct from Small.
        let cfg_medium = MusicGenConfig::v_medium_default();
        let file_medium = musicgen_gguf(
            NAME_MEDIUM,
            cfg_medium,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let mg_medium = MusicGen::from_gguf(&file_medium)
            .expect("chunk-free Medium GGUF must bind via fallback");
        assert_eq!(mg_medium.config().d_model, DEFAULT_D_MODEL_MEDIUM);
        assert_eq!(mg_medium.config().num_layers, DEFAULT_NUM_LAYERS_MEDIUM);
        assert_eq!(mg_medium.config().n_heads, DEFAULT_N_HEADS_MEDIUM);
        assert_eq!(mg_medium.config().ffn_dim, DEFAULT_FFN_DIM_MEDIUM);
        // Cross-variant sanity — Medium fallback path did NOT slip
        // into Small defaults.
        assert_ne!(mg_medium.config().d_model, DEFAULT_D_MODEL_SMALL);
        assert_ne!(mg_medium.config().num_layers, DEFAULT_NUM_LAYERS_SMALL);
    }

    // -----------------------------------------------------------------------
    // 11. Missing name chunk fails loud (never silently mis-routes to a
    //     default variant)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_name_chunk() {
        // A GGUF with correct arch but no name chunk cannot be
        // discriminated between Small and Medium — silently defaulting
        // would misroute half the artifacts. The loader must fail loud.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        // NO name chunk.
        b.add_tensor(
            "decoder.model.decoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MusicGen::from_gguf(&file) else {
            panic!("expected ModelLoad on missing name");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.name`"),
                    "message must call out the missing name chunk, got `{m}`"
                );
                assert!(
                    m.contains("small") && m.contains("medium"),
                    "message must name both variants the loader would need to \
                     discriminate, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. matches_config soft accessor honestly reflects shape presence
    // -----------------------------------------------------------------------

    #[test]
    fn matches_config_soft_accessor_finds_d_model_axis() {
        // The soft accessor should return true when at least one bound
        // tensor has an axis matching `d_model`. The fixture LM decoder
        // tensor's rows/cols are both `d_model` so this must pass.
        let cfg = MusicGenConfig::v_small_default();
        let file = musicgen_gguf(
            NAME_SMALL,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let mg = MusicGen::from_gguf(&file).unwrap();
        assert!(
            mg.weights.matches_config(mg.config()),
            "at least one bound tensor must have an axis matching config.d_model"
        );
        // Sanity: a stale config (bogus d_model) does NOT match the
        // fixture — pins the accessor as a real check (not a stub that
        // always returns true).
        let stale = MusicGenConfig {
            d_model: 99_999,
            ..cfg
        };
        assert!(
            !mg.weights.matches_config(&stale),
            "matches_config must return false for a d_model with no matching axis"
        );
    }
}
