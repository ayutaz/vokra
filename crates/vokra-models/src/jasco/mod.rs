//! **Meta AudioCraft JASCO 400M Chords+Drums**
//! (`facebook/jasco-chords-drums-400M`, CC-BY-NC-4.0 — T4 tier) —
//! joint-audio-symbolic-conditioning music generation runtime binder
//! (2026-08-14 audit follow-up Wave 6, first **flow-matching + symbolic
//! conditioning** music runtime binder in the tree).
//!
//! # Primary source
//!
//! - HF model card (400M Chords+Drums):
//!   <https://huggingface.co/facebook/jasco-chords-drums-400M>
//! - AudioCraft reference implementation (MIT code, CC-BY-NC-4.0
//!   weights): <https://github.com/facebookresearch/audiocraft>
//!   (`audiocraft/models/jasco.py` — the `JASCO` handle +
//!   `SymbolicConditioner` chord/drum encoder stack).
//! - Paper: Tal et al., *"Joint Audio and Symbolic Conditioning for
//!   Temporally Controlled Text-to-Music Generation"* (2024,
//!   arXiv:2406.10970).
//! - Weight license: **CC-BY-NC-4.0** (Meta AudioCraft weight policy;
//!   the code layer at `github.com/facebookresearch/audiocraft` is MIT
//!   but the trained weights are non-commercial). `docs/license-audit.md`
//!   §3.1 row 458 = ☑ Research-only 2026-08-04 yousan (T4, X-Codec-2
//!   precedent + MusicGen family T4 precedent).
//!
//! # Architecture (transcribed from primary sources)
//!
//! JASCO is a **flow-matching** music generator conditioned **jointly**
//! on three signal streams: (1) natural-language text, (2) a chord
//! progression as a discrete-symbol sequence, and (3) a drum track as a
//! discrete-symbol sequence. The paper's key contribution is the joint
//! symbolic-conditioning encoder stack — chord and drum tracks are
//! encoded to time-aligned prefixes that condition a flow-matching DiT
//! backbone, alongside a frozen T5-base text prefix. The forward pass is:
//!
//! ```text
//! text prompt (UTF-8 string)  chords ([u32])  drums ([u32])
//!   |                            |               |
//!   v                            v               v
//!   [frozen T5-base text        [JASCO chord    [JASCO drum      ← **loud-partial**
//!    encoder]                    encoder]        encoder]
//!   (google-t5/t5-base — HF   (audiocraft/     (audiocraft/
//!    transformers              models/jasco.py   models/jasco.py
//!    T5EncoderModel;           `SymbolicConditioner` chord head +
//!    native body landed;       drum head — greenfield, no analog
//!    composition pending)      in-tree; distinct from musicgen's
//!                              single-modality text-only cross-attention)
//!   |__________________________|_______________|
//!                              |
//!                              v
//!   AudioCraft flow-matching transformer stack with              ← **loud-partial**
//!   joint text + chord + drum prefix cross-attention
//!   (audiocraft transformer body with the JASCO-specific dual-
//!    conditioning cross-attention; distinct from melodyflow's
//!    dual-prefix editing use-case and musicgen's AR + delay
//!    pattern.)
//!   |
//!   v
//!   flow-matching sampler + EnCodec RVQ decode composition   ← **primitives exist**
//!   (`vokra_ops::flow_sampler::flow_sample` from M3-05 +
//!    `vokra_ops::encodec_rvq_decode` from M4-04 — both are the
//!    LANDED composition anchors this loud-partial message cites so a
//!    reader knows the two primitives already exist.)
//!   |
//!   v
//! PCM (mono f32, 32 kHz — bundled AudioCraft EnCodec 32 kHz codec)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`JascoVariant`] enum discrimination via
//!     [`JascoVariant::from_name`] (Chords400mDrums bound — the only
//!     converter in-tree; the 3 known-unbound siblings
//!     `NAME_CHORDS_DRUMS_1B` / `NAME_MELODY_400M` / `NAME_MELODY_1B`
//!     map to `None` for a specific "converter not yet in-tree" error).
//!   - [`Jasco::from_gguf`] with strict
//!     `vokra.model.arch == "jasco_400m_chords_drums"` validation. The
//!     sibling music-gen family arch tags (`musicgen` / `audiogen` /
//!     `magnet_small_10secs` / `magnet_medium_30secs` /
//!     `melodyflow_t24_30secs` / `audioldm2` / `stable_audio_open_small`
//!     / `ace_step` / `bs_roformer`) fail with a specific "sibling
//!     mis-route" [`VokraError::ModelLoad`] enumerating the whole
//!     neighbourhood — silent aliasing would misroute the runtime
//!     dispatch to a family with a different decoder loop (FR-EX-08).
//!   - [`JascoConfig::from_gguf`] with stamped topology reads on current
//!     artifacts and primary-source constant fallback **per variant** for
//!     GGUFs produced before the chunk contract landed.
//!   - [`JascoWeights::from_gguf`] with a floor of non-empty tensor
//!     count enforced loud (a GGUF that carries zero tensors is refused
//!     rather than silently running an all-zero forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::NonCommercial`] per the JASCO converter's
//!     stamped `cc-by-nc-4.0` — T4 tier, fail-closed at the runtime
//!     compliance gate M2-13).
//!   - Argument validation ordering: the empty-chords + empty-drums
//!     "symbolic conditioning is not optional" gate + finite / positive
//!     duration + non-empty prompt gates fire BEFORE the loud-partial
//!     [`VokraError::UnsupportedOp`], so callers with bad args get a
//!     specific [`VokraError::InvalidArgument`] rather than a generic
//!     "not implemented" fall-through.
//!
//! - **Loud-partial (this WP)**: [`Jasco::generate`] returns
//!   [`VokraError::UnsupportedOp`] naming **three** deferred pieces:
//!   1. the joint symbolic conditioning encoder stack — frozen T5-base
//!      text encoder (upstream `transformers.T5EncoderModel`; no
//!      reusable primitive in `vokra_ops` today, shared with the
//!      musicgen follow-up) PLUS the JASCO-specific chord encoder + drum
//!      encoder (audiocraft `SymbolicConditioner` heads — greenfield, no
//!      analog in-tree; sibling musicgen has text-only cross-attention);
//!   2. the AudioCraft flow-matching transformer stack with joint
//!      conditioning — audiocraft transformer body plus JASCO's dual-
//!      conditioning cross-attention over text + chord + drum prefix
//!      (distinct from melodyflow's dual-prefix editing use-case and
//!      musicgen's AR + delay pattern);
//!   3. the flow-matching sampler + EnCodec RVQ decode composition —
//!      **`vokra_ops::flow_sampler::flow_sample` (M3-05) +
//!      `vokra_ops::encodec_rvq_decode` (M4-04)** are the LANDED
//!      composition anchors the message cites so a reader diagnosing
//!      the gap knows the two primitives already exist.
//!
//! The error names the **three primary source URLs** (HF card for the
//! bound variant + AudioCraft repo + paper) so a reader diagnosing this
//! gap has exactly three places to walk. **No fabricated PCM stream is
//! ever emitted** (FR-EX-08).
//!
//! Rationale (musicgen / vocos / bigvgan / snac / mt3 / RMVPE / pyannote
//! / hifigan / beat_this / sortformer / melodyflow loud-partial
//! precedent, CLAUDE.md 教訓 (a) — "loud-partial は fake-complete より
//! honest"): the surrounding scaffold + `from_gguf` chunk-group
//! validation + `JascoVariant` enum + FR-EX-08 loud-fails land today so
//! a follow-up wave can flip the switch by (i) landing the T5-base text
//! encoder body against a real T5-base `state_dict` (the converter
//! already emits the T5 weights under the `text_encoder.*` verbatim
//! HuggingFace safetensors keys — see the converter's tensor-name
//! contract), (ii) implementing the JASCO chord + drum encoders + the
//! joint-conditioning cross-attention AudioCraft transformer decode, and
//! (iii) wiring the composed loop to the LANDED `flow_sampler` +
//! `encodec_rvq_decode` primitives. The primitive for (iii) already
//! exists so the follow-up wave is composition + two greenfield forward
//! bodies (T5 + JASCO chord/drum encoder pair + audiocraft transformer),
//! NOT three greenfield kernels.
//!
//! # `vokra.jasco.*` chunk group (read here — fallback-friendly)
//!
//! The JASCO converter
//! (`crates/vokra-convert/src/models/jasco_400m_chords_drums.rs`)
//! stamps the complete topology chunk group. The binder retains the
//! per-variant fallback only for artifacts produced before that contract
//! landed.
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`]
//!   (`"jasco_400m_chords_drums"`). Deliberately distinct from every
//!   sibling music-generation arch — `musicgen` / `audiogen`
//!   (AR-over-EnCodec) / `magnet_small_10secs` / `magnet_medium_30secs`
//!   (non-autoregressive masked-LM) / `melodyflow_t24_30secs` (DiT
//!   flow-matching editing — same op family but different conditioning
//!   stack, dual text + audio prefix, not chord + drum) / `audioldm2` /
//!   `stable_audio_open_small` (latent diffusion) / `ace_step`
//!   (chunked-AR) / `bs_roformer` (source-separation, not generation at
//!   all). Silently sharing an arch tag would mis-route runtime
//!   dispatch to a wrong-shape forward — FR-EX-08.
//! - `vokra.model.name` (`String`): [`NAME_CHORDS_DRUMS_400M`]
//!   (`"jasco_400m_chords_drums"`) — the sole bound variant under the
//!   shared arch tag today. Sibling variants (`jasco-chords-drums-1B`
//!   / `jasco-melody-400M` / `jasco-melody-1B`) referenced in the paper
//!   are NOT yet in-tree as converters, so the runtime enum extension
//!   pending path emits a specific "converter not yet in-tree" anchor.
//! - `vokra.jasco.{d_model, num_layers, n_heads, ffn_dim, num_codebooks,
//!   codec_frame_rate_hz, sample_rate_hz, text_prefix_len,
//!   chord_vocab_size, drum_vocab_size, num_flow_steps, cfg_scale}`
//!   (`u32` for the counts + `f32` for `cfg_scale`): the composite
//!   topology axes. Fallback constants transcribed from paper §3 (Tal et
//!   al. 2024) + AudioCraft family convention (musicgen-small style
//!   transformer axes) + the bundled EnCodec 32 kHz codec params.
//!   Values are pinned to AudioCraft revision
//!   `896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`: chord card 194 plus one
//!   null token (vocabulary 195), 128-wide drum EnCodec latents, the
//!   100-step Euler fallback, and CFG(all)=5.0.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03 / M2-13) can classify the
//!   artifact without re-inspecting the safetensors provenance. The
//!   JASCO converter stamps `NonCommercial` by default per the HF card's
//!   `license: cc-by-nc-4.0` — a caller who legitimately holds the
//!   weight under a distinct SPDX overrides at `vokra-cli convert
//!   --license <spdx>` and the stamped class re-derives via
//!   `LicenseClass::from_license_str`.
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`NAME_CHORDS_DRUMS_400M`] / topology keys) — same rule
//! the sibling BF16 pass-through binders (`musicgen` / `sortformer` /
//! `pyannote` / `snac` / `hifigan` / `beat_this` / `mt3` / `melodyflow`
//! / `magnet`) use so `vokra-models` does not gain a dependency edge
//! onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! JASCO ships safetensors + PyTorch pickle upstream; this runtime
//! **never** touches ONNX (FR-LD-05 / NFR-DS-02). If the upstream release
//! ships pickle only, callers pre-flatten offline via a future
//! `tools/parity/jasco_400m_chords_drums_prepare_checkpoint.py` (not
//! yet written — a thin wrapper over `bin_to_safetensors.py`; an
//! uv-managed Python 3.12 sidecar per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]` — not
//! part of the runtime), mirroring the
//! MusicGen / MelodyFlow / SpeechT5-HiFi-GAN / Sortformer / Charsiu
//! bridge pattern.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/jasco_400m_chords_drums.rs`. See module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model jasco-400m-chords-drums`.
///
/// **Currently the only converter in-tree** — the sibling variants
/// (1B chords/drums, 400M melody, 1B melody) referenced in Tal et al.
/// 2024 §4 are not yet in-tree as converters; when they land they will
/// carry their own arch tags (following the family convention that a
/// music-gen variant with a distinct decoder-topology stanza gets its
/// own arch tag — e.g. MAGNeT's `magnet_small_10secs` /
/// `magnet_medium_30secs` split).
///
/// Deliberately distinct from every sibling music-generation arch —
/// `musicgen` (Meta AudioCraft AR-LM over EnCodec, sibling AR family),
/// `audiogen` (Meta AudioCraft sound-effects LM), `magnet_small_10secs`
/// / `magnet_medium_30secs` (Meta AudioCraft non-autoregressive
/// masked-LM), `melodyflow_t24_30secs` (Meta AudioCraft DiT flow-
/// matching editing — same op family but different conditioning stack:
/// dual text + audio prefix, not chord + drum), `audioldm2` (score-based
/// latent diffusion U-Net), `stable_audio_open_small` (DiT + audio VAE
/// with different conditioning), `ace_step` (chunked-AR),
/// `bs_roformer` (music-source separation, not generation at all).
/// Silent-sharing an arch tag with any of them would mis-route the
/// runtime dispatch to a wrong-shape forward — FR-EX-08.
pub const ARCH: &str = "jasco_400m_chords_drums";

/// Expected `vokra.model.name` value for the **400M Chords+Drums**
/// variant — matches the `huggingface.co/facebook/jasco-chords-drums-400M`
/// upstream slug + the converter's `NAME` constant.
pub const NAME_CHORDS_DRUMS_400M: &str = "jasco_400m_chords_drums";

/// Future-proof anchor: canonical `vokra.model.name` for the **1B
/// Chords+Drums** variant — no converter in-tree yet. Named in
/// [`JascoVariant::from_name`] so a mis-routed 1B GGUF gets a specific
/// "converter not yet in-tree" anchor rather than a generic bind
/// failure. Primary source: `huggingface.co/facebook/jasco-chords-drums-1B`
/// (referenced in Tal et al. 2024 §4, not yet in-tree as a converter).
pub const NAME_CHORDS_DRUMS_1B: &str = "jasco_1b_chords_drums";

/// Future-proof anchor: canonical `vokra.model.name` for the **400M
/// Melody** variant — no converter in-tree yet. The melody variant also
/// carries a chroma-conditioning frontend on top of the LM (following
/// the sibling `musicgen-melody` convention), which the error message
/// calls out for the reader. Primary source:
/// `huggingface.co/facebook/jasco-melody-400M` (referenced in Tal et
/// al. 2024 §4).
pub const NAME_MELODY_400M: &str = "jasco_400m_melody";

/// Future-proof anchor: canonical `vokra.model.name` for the **1B
/// Melody** variant — no converter in-tree yet. Same "chroma
/// conditioning frontend" note as [`NAME_MELODY_400M`]. Primary source:
/// `huggingface.co/facebook/jasco-melody-1B` (referenced in Tal et al.
/// 2024 §4).
pub const NAME_MELODY_1B: &str = "jasco_1b_melody";

/// Upstream HF repository slug (`org/name`) for the **400M
/// Chords+Drums** variant. Mirror of the converter's `UPSTREAM_HF`
/// constant.
pub const UPSTREAM_HF_400M_CHORDS_DRUMS: &str = "facebook/jasco-chords-drums-400M";

/// Primary-source anchor for the **400M Chords+Drums** variant's HF
/// model card. Cited in the loud-partial error so a reader diagnosing
/// the gap knows the definitive artifact source.
pub const PRIMARY_SOURCE_HF_CARD_400M_CHORDS_DRUMS: &str =
    "huggingface.co/facebook/jasco-chords-drums-400M";

/// Primary-source anchor for the AudioCraft reference repository
/// (MIT code — the tensor-name walk anchor for T5 encoder + AudioCraft
/// transformer + JASCO `SymbolicConditioner` head references). Cited in
/// the loud-partial error so a reader knows the code reference.
pub const PRIMARY_SOURCE_AUDIOCRAFT_REPO: &str = "github.com/facebookresearch/audiocraft";

/// Paper anchor (Tal et al. 2024, "Joint Audio and Symbolic Conditioning
/// for Temporally Controlled Text-to-Music Generation") — cited
/// alongside the HF card + AudioCraft repo so a reader has the
/// theoretical context as well.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2406.10970";

// ---------------------------------------------------------------------------
// GGUF metadata key constants for the `vokra.jasco.*` chunk group.
// Mirrored by the converter; fallback via `JascoVariant::default_config()`
// remains for older artifacts.
// ---------------------------------------------------------------------------

/// `vokra.jasco.d_model` — flow-matching transformer hidden dim.
/// Primary-source default (400M): 1024, pinned by AudioCraft's
/// `config/model/lm/model_scale/small.yaml`.
pub const GGUF_KEY_D_MODEL: &str = "vokra.jasco.d_model";
/// `vokra.jasco.num_layers` — flow-matching transformer depth.
/// Primary-source default (400M): 24 (audiocraft musicgen-small style
/// convention).
pub const GGUF_KEY_NUM_LAYERS: &str = "vokra.jasco.num_layers";
/// `vokra.jasco.n_heads` — multi-head attention head count.
/// Primary-source default (400M): 16 (audiocraft musicgen-small style
/// convention; `head_dim = 1024 / 16 = 64`).
pub const GGUF_KEY_N_HEADS: &str = "vokra.jasco.n_heads";
/// `vokra.jasco.ffn_dim` — feedforward inner dimension.
/// Primary-source default (400M): 4096 (AudioCraft "4× hidden"
/// convention: `4096 = 4 × 1024`).
pub const GGUF_KEY_FFN_DIM: &str = "vokra.jasco.ffn_dim";
/// `vokra.jasco.num_codebooks` — number of RVQ codebook streams the
/// bundled EnCodec 32 kHz codec emits. Shared across variants: 4
/// (matches audiocraft `EncodecModel(...).quantizer.n_q = 4`).
pub const GGUF_KEY_NUM_CODEBOOKS: &str = "vokra.jasco.num_codebooks";
/// `vokra.jasco.codec_frame_rate_hz` — bundled EnCodec 32 kHz codec
/// output frame rate. Shared across variants: 50 Hz (matches audiocraft
/// `EncodecModel.frame_rate = 50`).
pub const GGUF_KEY_CODEC_FRAME_RATE_HZ: &str = "vokra.jasco.codec_frame_rate_hz";
/// `vokra.jasco.sample_rate_hz` — bundled EnCodec 32 kHz codec sample
/// rate. Shared across variants: 32000 Hz (matches audiocraft
/// `EncodecModel.sample_rate = 32000`).
pub const GGUF_KEY_SAMPLE_RATE_HZ: &str = "vokra.jasco.sample_rate_hz";
/// `vokra.jasco.text_prefix_len` — frozen T5-base text encoder
/// conditioning prefix length in tokens (u32). Primary-source default:
/// 512 (T5-base `max_position_embeddings = 512`).
pub const GGUF_KEY_TEXT_PREFIX_LEN: &str = "vokra.jasco.text_prefix_len";
/// `vokra.jasco.chord_vocab_size` — chord embedding vocabulary including
/// the null/dropout row. AudioCraft pins `card: 194` and constructs
/// `nn.Embedding(card + 1, 16)`, so this value is exactly 195.
pub const GGUF_KEY_CHORD_VOCAB_SIZE: &str = "vokra.jasco.chord_vocab_size";
/// `vokra.jasco.drum_vocab_size` — legacy key name for the drum
/// conditioner input width. JASCO does not use a General-MIDI token
/// vocabulary here: the official config feeds 128-wide EnCodec latents.
pub const GGUF_KEY_DRUM_VOCAB_SIZE: &str = "vokra.jasco.drum_vocab_size";
/// `vokra.jasco.num_flow_steps` — Euler-mode fallback step count. The
/// official API defaults to adaptive Dopri5; when `euler=true`, it pins
/// `euler_steps=100`.
pub const GGUF_KEY_NUM_FLOW_STEPS: &str = "vokra.jasco.num_flow_steps";
/// `vokra.jasco.cfg_scale` — official all-condition CFG coefficient.
/// `JASCO::set_generation_params` pins `cfg_coef_all=5.0` and
/// `cfg_coef_txt=0.0`; this legacy single-scale field records the all term.
pub const GGUF_KEY_CFG_SCALE: &str = "vokra.jasco.cfg_scale";

// Per-variant primary-source constants. The 400M Chords+Drums variant
// axes are transcribed from paper §3 (Tal et al. 2024 arXiv:2406.10970)
// + AudioCraft family convention (musicgen-small style transformer)
// + the bundled EnCodec 32 kHz codec params. Conditioner and sampling
// values are pinned to AudioCraft revision
// 896ec7c47f5e5d1e5aa1e4b260c4405328bf009d.

/// 400M Chords+Drums variant flow-matching transformer hidden dim
/// (`d_model`).
pub const DEFAULT_D_MODEL_400M_CHORDS_DRUMS: u32 = 1024;
/// 400M Chords+Drums variant flow-matching transformer depth.
pub const DEFAULT_NUM_LAYERS_400M_CHORDS_DRUMS: u32 = 24;
/// 400M Chords+Drums variant attention head count.
/// `head_dim = 1024 / 16 = 64` — the audiocraft family invariant.
pub const DEFAULT_N_HEADS_400M_CHORDS_DRUMS: u32 = 16;
/// 400M Chords+Drums variant feedforward inner dimension. AudioCraft
/// "4× hidden" convention: `4096 = 4 × 1024`.
pub const DEFAULT_FFN_DIM_400M_CHORDS_DRUMS: u32 = 4096;

/// Shared number of RVQ codebook streams the bundled EnCodec 32 kHz
/// codec emits per frame. Shared across every JASCO variant. Primary
/// source: audiocraft `EncodecModel(...).quantizer.n_q = 4`.
pub const NUM_CODEBOOKS: u32 = 4;

/// EnCodec output frame rate for the bundled 32 kHz codec (matches
/// audiocraft `EncodecModel.frame_rate = 50`).
pub const CODEC_FRAME_RATE_HZ: u32 = 50;

/// EnCodec sample rate for the bundled 32 kHz codec (matches audiocraft
/// `EncodecModel.sample_rate = 32000`).
pub const SAMPLE_RATE_HZ: u32 = 32_000;

/// Frozen T5-base text encoder conditioning prefix length in tokens.
/// Primary source: T5-base `max_position_embeddings = 512`.
pub const DEFAULT_TEXT_PREFIX_LEN: u32 = 512;

/// Chord embedding row count: config card 194 plus one null/dropout row.
pub const DEFAULT_CHORD_VOCAB_SIZE: u32 = 195;

/// Official drum-condition EnCodec latent width. The public field/key name
/// is retained for compatibility; this is not a discrete drum vocabulary.
pub const DEFAULT_DRUM_VOCAB_SIZE: u32 = 128;

/// Official Euler fallback step count (`FlowMatchingModel.generate`).
pub const DEFAULT_NUM_FLOW_STEPS: u32 = 100;

/// Official all-condition CFG coefficient (`JASCO::set_generation_params`).
pub const DEFAULT_CFG_SCALE: f32 = 5.0;

// ---------------------------------------------------------------------------
// JascoVariant — the variant discriminator (name-based)
// ---------------------------------------------------------------------------

/// Which JASCO family variant a GGUF represents. Determined by
/// [`JascoVariant::from_name`] against `vokra.model.name`.
///
/// **This WP is scoped to the 400M Chords+Drums variant only** — it is
/// the only JASCO converter in-tree. Sibling variants referenced in
/// Tal et al. 2024 §4 (1B chords/drums, 400M melody, 1B melody) map to
/// `None` in [`from_name`](Self::from_name) so [`Jasco::from_gguf`] can
/// emit a specific "converter not yet in-tree" error rather than a
/// generic bind failure. When those converters ship, the enum extends
/// with new arms without changing this bound-variant behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JascoVariant {
    /// `facebook/jasco-chords-drums-400M` — 400M-parameter flow-matching
    /// transformer with joint text + chord + drum symbolic conditioning
    /// (`d_model=1024`, `num_layers=24`, `n_heads=16`, `ffn_dim=4096`).
    Chords400mDrums,
}

impl JascoVariant {
    /// Discriminates a JASCO variant from `vokra.model.name`. Returns
    /// `None` for JASCO family variants that exist in the paper but are
    /// not yet bound (`jasco_1b_chords_drums` / `jasco_400m_melody` /
    /// `jasco_1b_melody`) so [`Jasco::from_gguf`] can emit a specific
    /// "converter not yet in-tree" error naming the paper as the
    /// primary source, and for any string that isn't a JASCO family
    /// name at all.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            NAME_CHORDS_DRUMS_400M => Some(Self::Chords400mDrums),
            _ => None,
        }
    }

    /// Canonical `vokra.model.name` string for this variant. Matches
    /// the upstream HF slug + the converter's `NAME` constant.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Chords400mDrums => NAME_CHORDS_DRUMS_400M,
        }
    }

    /// Canonical short display name for logs / errors.
    #[must_use]
    pub const fn short(self) -> &'static str {
        match self {
            Self::Chords400mDrums => "Chords400mDrums",
        }
    }

    /// The `vokra.model.arch` string this variant carries.
    #[must_use]
    pub const fn arch(self) -> &'static str {
        match self {
            Self::Chords400mDrums => ARCH,
        }
    }

    /// The primary-source HF card URL for this variant. Cited in the
    /// loud-partial error.
    #[must_use]
    pub const fn primary_source_hf_card(self) -> &'static str {
        match self {
            Self::Chords400mDrums => PRIMARY_SOURCE_HF_CARD_400M_CHORDS_DRUMS,
        }
    }

    /// Primary-source-transcribed axes for this variant as a const
    /// [`JascoConfig`]. Used by [`JascoConfig::from_gguf`] as the
    /// per-key fallback when the topology chunk group is absent in an
    /// artifact produced before the current converter contract.
    #[must_use]
    pub const fn default_config(self) -> JascoConfig {
        match self {
            Self::Chords400mDrums => JascoConfig {
                variant: Self::Chords400mDrums,
                d_model: DEFAULT_D_MODEL_400M_CHORDS_DRUMS,
                num_layers: DEFAULT_NUM_LAYERS_400M_CHORDS_DRUMS,
                n_heads: DEFAULT_N_HEADS_400M_CHORDS_DRUMS,
                ffn_dim: DEFAULT_FFN_DIM_400M_CHORDS_DRUMS,
                num_codebooks: NUM_CODEBOOKS,
                codec_frame_rate_hz: CODEC_FRAME_RATE_HZ,
                sample_rate_hz: SAMPLE_RATE_HZ,
                text_prefix_len: DEFAULT_TEXT_PREFIX_LEN,
                chord_vocab_size: DEFAULT_CHORD_VOCAB_SIZE,
                drum_vocab_size: DEFAULT_DRUM_VOCAB_SIZE,
                num_flow_steps: DEFAULT_NUM_FLOW_STEPS,
                cfg_scale: DEFAULT_CFG_SCALE,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// JascoConfig — composite topology axes from the `vokra.jasco.*` chunk
// group, with primary-source constant fallback **per variant**. Mirror
// of `MusicGenConfig` / `SortformerConfig` / `PyanNetConfig`
// fallback pattern.
// ---------------------------------------------------------------------------

/// JASCO hyperparameters as they ride the `vokra.jasco.*` chunk group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key using [`JascoVariant::default_config`] as
/// the per-variant baseline — a GGUF that never carried the chunk still
/// loads with the upstream defaults transcribed from paper §3 + the
/// AudioCraft family convention. Numeric-count axes are `u32` in the
/// GGUF; `cfg_scale` is `f32`.
///
/// The chord vocabulary and drum latent width are pinned to the official
/// AudioCraft conditioner config; neither is inferred from a paper diagram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JascoConfig {
    /// Which JASCO family variant this config represents.
    pub variant: JascoVariant,
    /// Flow-matching transformer hidden dim (400M default: 1024).
    pub d_model: u32,
    /// Flow-matching transformer depth (400M default: 24).
    pub num_layers: u32,
    /// Multi-head attention head count (400M default: 16;
    /// `head_dim = d_model / n_heads = 64`).
    pub n_heads: u32,
    /// Feedforward inner dimension (400M default: 4096; AudioCraft
    /// "4× hidden" convention `ffn_dim = 4 × d_model`).
    pub ffn_dim: u32,
    /// Number of RVQ codebook streams the bundled EnCodec 32 kHz codec
    /// emits per frame (shared: 4).
    pub num_codebooks: u32,
    /// EnCodec output frame rate (shared: 50 Hz).
    pub codec_frame_rate_hz: u32,
    /// EnCodec sample rate (shared: 32000 Hz = 32 kHz).
    pub sample_rate_hz: u32,
    /// Frozen T5-base text encoder conditioning prefix length in tokens
    /// (default: 512 = T5-base `max_position_embeddings`).
    pub text_prefix_len: u32,
    /// Chord embedding vocabulary size including the null row (195).
    pub chord_vocab_size: u32,
    /// Legacy-named drum conditioner width (128 EnCodec latent channels;
    /// not a discrete vocabulary).
    pub drum_vocab_size: u32,
    /// Euler fallback step count (100; normal official default is adaptive
    /// Dopri5 with 1e-5 rtol/atol).
    pub num_flow_steps: u32,
    /// All-condition classifier-free-guidance coefficient (5.0; text-only
    /// coefficient is 0.0 in the official JASCO wrapper).
    pub cfg_scale: f32,
}

impl JascoConfig {
    /// Primary-source-transcribed 400M Chords+Drums axes as a `const` —
    /// alias for `JascoVariant::Chords400mDrums.default_config()`.
    #[must_use]
    pub const fn v_400m_chords_drums_default() -> Self {
        JascoVariant::Chords400mDrums.default_config()
    }

    /// Reads every `vokra.jasco.*` chunk from `gguf`, falling back to
    /// the per-variant primary-source defaults per absent key.
    ///
    /// The current converter stamps this complete chunk group. Per-key
    /// fallback is retained only for artifacts produced before the
    /// metadata contract landed.
    ///
    /// Mirror of
    /// [`crate::musicgen::MusicGenConfig::from_gguf`] fallback pattern.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile, variant: JascoVariant) -> Self {
        let default = variant.default_config();
        Self {
            variant,
            d_model: gguf
                .get(GGUF_KEY_D_MODEL)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.d_model),
            num_layers: gguf
                .get(GGUF_KEY_NUM_LAYERS)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.num_layers),
            n_heads: gguf
                .get(GGUF_KEY_N_HEADS)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.n_heads),
            ffn_dim: gguf
                .get(GGUF_KEY_FFN_DIM)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.ffn_dim),
            num_codebooks: gguf
                .get(GGUF_KEY_NUM_CODEBOOKS)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.num_codebooks),
            codec_frame_rate_hz: gguf
                .get(GGUF_KEY_CODEC_FRAME_RATE_HZ)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.codec_frame_rate_hz),
            sample_rate_hz: gguf
                .get(GGUF_KEY_SAMPLE_RATE_HZ)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.sample_rate_hz),
            text_prefix_len: gguf
                .get(GGUF_KEY_TEXT_PREFIX_LEN)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.text_prefix_len),
            chord_vocab_size: gguf
                .get(GGUF_KEY_CHORD_VOCAB_SIZE)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.chord_vocab_size),
            drum_vocab_size: gguf
                .get(GGUF_KEY_DRUM_VOCAB_SIZE)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.drum_vocab_size),
            num_flow_steps: gguf
                .get(GGUF_KEY_NUM_FLOW_STEPS)
                .and_then(GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default.num_flow_steps),
            cfg_scale: match gguf.get(GGUF_KEY_CFG_SCALE) {
                Some(GgufMetadataValue::F32(v)) => *v,
                _ => default.cfg_scale,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// JascoWeights — bound the tensor manifest with a non-emptiness gate.
// Mirror of `MusicGenWeights` / `SortformerWeights` / `Mt3Weights` /
// `BeatThisWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a JASCO GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// JASCO checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future
/// `JascoWeights::bind_t5_encoder_weights` /
/// `JascoWeights::bind_symbolic_encoder_weights` /
/// `JascoWeights::bind_flow_transformer_weights` tensor walk can find
/// its inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct JascoWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// T5-encoder + JASCO-symbolic-encoder + audiocraft-flow-transformer
    /// + EnCodec-decode composition wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl JascoWeights {
    /// Scans `gguf` for the JASCO state_dict tensors. Refuses to bind
    /// if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is
    /// never a valid JASCO checkpoint).
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
                "jasco: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model \
                 jasco-400m-chords-drums` against a `facebook/jasco-chords-drums-400M` \
                 safetensors checkpoint (the upstream release ships a bundle of the \
                 flow-matching transformer + T5-base text encoder + JASCO chord + drum \
                 symbolic encoders + bundled EnCodec 32 kHz codec — every group must \
                 be present)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the T5 + JASCO-symbolic + flow-transformer + codec-
    /// decode forward wave uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load-time shape gate — validates that at least one bound tensor
    /// has an axis matching `config.d_model`. Under the current landing
    /// this is a **soft** gate (mismatch is silently ignored) because
    /// the T5-encoder + JASCO-symbolic-encoder + audiocraft-flow-
    /// transformer + EnCodec-decode tensor-name walk has not yet been
    /// pinned pending the follow-up wave's manifest fetch — a hard
    /// shape assertion today would fail against every legitimate future
    /// manifest.
    ///
    /// The follow-up wave will replace this soft accessor with a hard
    /// pin against the primary-source-verified tensor-name walk (mirror
    /// of `pyannote::PyanNetWeights::verify_core_shapes`).
    #[must_use]
    pub fn matches_config(&self, config: &JascoConfig) -> bool {
        let d = config.d_model as usize;
        self.tensors.iter().any(|(_, dims)| dims.contains(&d))
    }
}

// ---------------------------------------------------------------------------
// Jasco — the runtime binder handle
// ---------------------------------------------------------------------------

/// Meta AudioCraft JASCO joint-audio-symbolic-conditioning music
/// generation runtime binder (`facebook/jasco-chords-drums-400M`,
/// CC-BY-NC-4.0 T4 tier).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`generate`](Self::generate) with a text prompt + chord sequence +
/// drum sequence + duration to obtain a `Vec<f32>` of 32 kHz PCM
/// samples. See the module doc for the current implementation-status
/// matrix and the FR-EX-08 loud-error contract on the T5-base + JASCO-
/// symbolic-encoder + audiocraft-flow-transformer + EnCodec-decode
/// composition.
#[derive(Debug)]
pub struct Jasco {
    config: JascoConfig,
    variant: JascoVariant,
    // The bound weights are held (real, counted) but the T5-encoder +
    // JASCO-symbolic-encoder + flow-transformer + EnCodec-decode
    // composition is a follow-up wave; the field is deliberately
    // `#[allow(dead_code)]` until the composition lands so a reader is
    // not misled by an unused field. Same posture as musicgen / RMVPE /
    // pyannote / mt3 / beat_this / sortformer.
    #[allow(dead_code)]
    weights: JascoWeights,
    weight_license: LicenseClass,
}

impl Jasco {
    /// Binds a JASCO GGUF: validates arch, discriminates the variant
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
    ///   not `"jasco_400m_chords_drums"` (a sibling music-generation
    ///   GGUF handed to us by mistake — `musicgen` / `audiogen` /
    ///   `magnet_small_10secs` / `magnet_medium_30secs` /
    ///   `melodyflow_t24_30secs` / … — fails with a clear message
    ///   instead of a downstream missing-tensor).
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is absent, or
    ///   when it identifies a JASCO family variant not bound in this WP
    ///   (`jasco_1b_chords_drums` / `jasco_400m_melody` /
    ///   `jasco_1b_melody` — referenced in the paper but converters not
    ///   yet in-tree).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`JascoWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "jasco: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model jasco-400m-chords-drums`? \
                     Note that the sibling music-generation arch tags — `musicgen` \
                     (Meta AudioCraft AR-LM over EnCodec, sibling AR family), \
                     `audiogen` (Meta AudioCraft sound-effects LM), \
                     `magnet_small_10secs` / `magnet_medium_30secs` (Meta AudioCraft \
                     non-autoregressive masked-LM), `melodyflow_t24_30secs` (Meta \
                     AudioCraft DiT flow-matching editing — same op family but \
                     different conditioning stack: dual text + audio prefix, not \
                     chord + drum), `audioldm2` / `stable_audio_open_small` (latent \
                     diffusion), `ace_step` (chunked-AR), `bs_roformer` (source-\
                     separation, not generation at all) — all live in the same \
                     music-generation neighbourhood but have completely different \
                     forward topologies; JASCO's flow-matching transformer with joint \
                     text + chord + drum symbolic prefix cross-attention has no analog \
                     in any sibling and silently aliasing arch would misroute the \
                     runtime dispatch, FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "jasco: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native jasco GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant discrimination via `vokra.model.name`. The 400M
        //    Chords+Drums variant carries the shared arch tag; sibling
        //    JASCO family variants (`jasco_1b_chords_drums` /
        //    `jasco_400m_melody` / `jasco_1b_melody` — referenced in
        //    Tal et al. 2024 §4 but not yet in-tree as converters) get
        //    a specific "converter not yet in-tree" error rather than a
        //    generic bind failure.
        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "jasco: GGUF is missing `vokra.model.name` (converter did not \
                     stamp it — cannot discriminate the JASCO family variant; expected \
                     `jasco_400m_chords_drums` for the only variant bound in this WP)"
                        .to_owned(),
                )
            })?;
        let variant = JascoVariant::from_name(name).ok_or_else(|| {
            if name == NAME_CHORDS_DRUMS_1B {
                VokraError::ModelLoad(format!(
                    "jasco: NAME `{name}` is not yet bound in the runtime — the paper \
                     (Tal et al. 2024 arXiv:2406.10970 §4) references a 1B \
                     Chords+Drums variant but no converter for it exists in-tree yet. \
                     A follow-up wave adds the converter alongside a new \
                     `JascoVariant::Chords1BDrums` arm. Primary source: \
                     huggingface.co/facebook/jasco-chords-drums-1B (converter not yet \
                     in-tree)."
                ))
            } else if name == NAME_MELODY_400M {
                VokraError::ModelLoad(format!(
                    "jasco: NAME `{name}` is not yet bound in the runtime — the paper \
                     (Tal et al. 2024 arXiv:2406.10970 §4) references a 400M Melody \
                     variant but no converter for it exists in-tree yet. The melody \
                     variant also needs a chroma-conditioning frontend on top of the \
                     flow-matching transformer (following the sibling `musicgen-melody` \
                     convention). A follow-up wave adds the converter alongside a new \
                     `JascoVariant::Melody400m` arm. Primary source: \
                     huggingface.co/facebook/jasco-melody-400M (converter not yet \
                     in-tree)."
                ))
            } else if name == NAME_MELODY_1B {
                VokraError::ModelLoad(format!(
                    "jasco: NAME `{name}` is not yet bound in the runtime — the paper \
                     (Tal et al. 2024 arXiv:2406.10970 §4) references a 1B Melody \
                     variant but no converter for it exists in-tree yet. Same chroma \
                     conditioning frontend note as the 400M Melody sibling. A follow-\
                     up wave adds the converter alongside a new \
                     `JascoVariant::Melody1B` arm. Primary source: \
                     huggingface.co/facebook/jasco-melody-1B (converter not yet \
                     in-tree)."
                ))
            } else {
                VokraError::ModelLoad(format!(
                    "jasco: NAME `{name}` is not a recognised JASCO family variant. \
                     Expected `{NAME_CHORDS_DRUMS_400M}` (the only variant bound in \
                     this WP). (Was this GGUF produced by a JASCO converter? The \
                     converter stamps `vokra.model.name` = `jasco_400m_chords_drums`.)"
                ))
            }
        })?;

        // 3. Topology axes from the `vokra.jasco.*` chunk group
        //    (fallback-friendly — see the module doc for the JASCO
        //    converter's stamp posture).
        let config = JascoConfig::from_gguf(file, variant);

        // 4. Load the tensor manifest with the non-emptiness gate.
        let weights = JascoWeights::from_gguf(file)?;

        // 5. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The JASCO converter
        //    defaults to `NonCommercial` per the HF card's
        //    `license: cc-by-nc-4.0`; a caller override at
        //    `--license <spdx>` re-derives the class. Missing provenance
        //    falls back to `Unknown` which is fail-closed at the M2-13
        //    compliance gate — same posture as musicgen / MT3 /
        //    Sortformer.
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

    /// The bound topology axes (from `vokra.jasco.*` chunk group with
    /// per-variant primary-source constant fallback).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &JascoConfig {
        &self.config
    }

    /// The bound JASCO family variant.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> JascoVariant {
        self.variant
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The JASCO converter
    /// stamps `NonCommercial` by default per the HF card's
    /// `license: cc-by-nc-4.0` (T4 tier — fail-closed at the M2-13
    /// compliance gate; owner must pass `--allow-noncommercial` to
    /// publish and the runtime refuses commercial-mode load). A GGUF
    /// missing the stamp reads back as [`LicenseClass::Unknown`] which
    /// is also fail-closed.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the T5 + JASCO-symbolic + flow-transformer + codec-
    /// decode forward wave uses it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Generates a `duration_secs`-length 32 kHz PCM stream conditioned
    /// jointly on the text `prompt`, discrete `chords` symbol sequence,
    /// and discrete `drums` symbol sequence.
    ///
    /// # Argument semantics
    ///
    /// - `prompt` — natural-language text description (UTF-8).
    /// - `chords` — discrete chord-progression symbol sequence (paper
    ///   §3.2 — indices into the discrete chord vocabulary).
    /// - `drums` — discrete drum-track symbol sequence (paper §3.2 —
    ///   indices into the discrete drum vocabulary).
    /// - `duration_secs` — target output horizon in seconds.
    ///
    /// **Symbolic conditioning is not optional in JASCO** — the paper's
    /// key contribution is joint text + chord + drum conditioning, so
    /// at least one of `chords` / `drums` MUST be non-empty. Passing
    /// both empty is an [`VokraError::InvalidArgument`] (semantic gate
    /// — silent fall-through to a text-only generation would
    /// misrepresent the model per FR-EX-08).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — JASCO's inference path
    /// requires **three** deferred pieces:
    ///
    /// 1. **Joint symbolic conditioning encoders**: the frozen T5-base
    ///    text encoder (upstream `transformers.T5EncoderModel`) PLUS
    ///    the JASCO-specific **chord encoder + drum encoder** (audiocraft
    ///    `SymbolicConditioner` heads). The shared native T5-base body
    ///    now lives in [`crate::t5_encoder`], but JASCO has not bound it
    ///    or its tokenizer metadata; the chord + drum heads are greenfield with
    ///    no analog in-tree (sibling musicgen has text-only cross-
    ///    attention).
    /// 2. **AudioCraft flow-matching transformer stack with joint
    ///    conditioning**: audiocraft transformer body plus JASCO's dual-
    ///    conditioning cross-attention over text + chord + drum prefix
    ///    (distinct from melodyflow's dual-prefix editing use-case and
    ///    musicgen's AR + delay pattern).
    /// 3. **Flow-matching sampler + EnCodec RVQ decode composition**:
    ///    `vokra_ops::flow_sampler::flow_sample` (M3-05) +
    ///    `vokra_ops::encodec_rvq_decode` (M4-04) are the **TWO landed
    ///    anchors** the loud-partial message cites so a reader
    ///    diagnosing this gap knows the two primitives already exist.
    ///
    /// The error names **three** primary source URLs (HF card for the
    /// bound variant + AudioCraft repo + paper) so a reader diagnosing
    /// this gap has exactly three places to walk. **No fabricated PCM
    /// stream is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `prompt` is empty, when
    ///   `chords` AND `drums` are both empty (symbolic conditioning is
    ///   not optional), or when `duration_secs` is not finite or is
    ///   not strictly positive. These gates fire BEFORE the loud-
    ///   partial so a caller with bad args gets a specific validation
    ///   error rather than a generic "not implemented" fall-through.
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred T5-encoder + JASCO-symbolic-encoder + flow-transformer
    ///   + EnCodec-decode composition.
    pub fn generate(
        &self,
        prompt: &str,
        chords: &[u32],
        drums: &[u32],
        duration_secs: f32,
    ) -> Result<Vec<f32>> {
        // Argument validation runs BEFORE the loud-partial so the
        // caller cannot confuse "bad args → wrong output" with
        // "loud-partial gate → no output at all". Mirror of
        // melodyflow's argument-first-then-loud-partial pattern.
        if prompt.is_empty() {
            return Err(VokraError::InvalidArgument(
                "jasco.generate: `prompt` is empty — JASCO's text prefix path expects \
                 at least one UTF-8 code point (upstream `JASCO.generate` rejects \
                 empty text; silent zero-fill would misrepresent the run per FR-EX-08)"
                    .to_owned(),
            ));
        }
        if chords.is_empty() && drums.is_empty() {
            return Err(VokraError::InvalidArgument(
                "jasco.generate: both `chords` and `drums` are empty — JASCO's \
                 symbolic conditioning is not optional (paper §3.2 — the model's \
                 key contribution is joint text + chord + drum conditioning). Pass \
                 at least one non-empty symbolic conditioning sequence; a silent \
                 fall-through to text-only generation would misrepresent the model \
                 per FR-EX-08. For pure text-to-music generation, use the sibling \
                 MusicGen family (`vokra-models::musicgen`) which is designed for \
                 text-only conditioning."
                    .to_owned(),
            ));
        }
        if !duration_secs.is_finite() || duration_secs <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "jasco.generate: `duration_secs` = {duration_secs} — must be finite \
                 and strictly positive (config default num_flow_steps = {}; \
                 sample_rate_hz = {})",
                self.config.num_flow_steps, self.config.sample_rate_hz,
            )));
        }

        // Bind unused args explicitly so a `#[warn(unused_variables)]`
        // change does not silently mask the loud-partial fire path; the
        // future real implementation will consume all four.
        let _ = (prompt, chords, drums, duration_secs);
        Err(generate_forward_loud_partial(
            &self.config,
            self.variant,
            prompt,
            chords,
            drums,
            duration_secs,
        ))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Jasco::generate`] until the T5-encoder + JASCO-symbolic-encoder +
/// flow-transformer + EnCodec-decode composition lands.
///
/// Names **all three** primary source URLs (HF card for the bound
/// variant + AudioCraft repo + paper) so a reader diagnosing the gap
/// has exactly three places to walk. Names the three deferred pieces
/// (chord encoder + drum encoder + flow-matching + AudioCraft
/// transformer stack) + the two LANDED composition anchors
/// (`vokra_ops::flow_sampler::flow_sample` from M3-05 +
/// `vokra_ops::encodec_rvq_decode` from M4-04). Mirrors the musicgen /
/// sortformer / MT3 / beat_this / RMVPE / pyannote / snac / hifigan /
/// melodyflow Wave 3-5 loud-partial-message precedent — CLAUDE.md 教訓
/// (a).
///
/// **Wave 5 lesson**: `VokraError::UnsupportedOp(String)` MUST be used
/// here (not `NotImplemented(&'static str)`) because the message is
/// dynamic (`format!()` with runtime state); `NotImplemented` would
/// fail E0308 on the `format!()` result.
fn generate_forward_loud_partial(
    cfg: &JascoConfig,
    variant: JascoVariant,
    prompt: &str,
    chords: &[u32],
    drums: &[u32],
    duration_secs: f32,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "jasco generate: T5-base text encoder forward + JASCO chord encoder + JASCO \
         drum encoder + AudioCraft flow-matching transformer stack (with joint text + \
         chord + drum prefix cross-attention) + EnCodec RVQ decode composition \
         pending. What is missing is (a) the joint symbolic conditioning encoder stack \
         — the frozen T5-base text encoder (`crate::t5_encoder` supplies the landed \
         native CPU/Metal body, but tokenizer and JASCO binding remain pending) PLUS \
         the JASCO-specific chord encoder + drum encoder (audiocraft \
         `SymbolicConditioner` heads — greenfield, no analog in-tree; sibling musicgen \
         has text-only cross-attention), (b) the AudioCraft flow-matching transformer \
         stack with joint conditioning — audiocraft transformer body plus the JASCO \
         dual-conditioning cross-attention over text + chord + drum prefix (distinct \
         from melodyflow's dual-prefix editing use-case and musicgen's AR + delay \
         pattern), and (c) the flow-matching sampler + EnCodec RVQ decode composition \
         — this is available via `vokra_ops::flow_sampler::flow_sample` (M3-05) + \
         `vokra_ops::encodec_rvq_decode` (M4-04) (the TWO landed anchors of the \
         composition; the follow-up wave wires the flow-transformer output onto these \
         primitives). Config: variant={variant_short}, d_model={d_model}, \
         num_layers={num_layers}, n_heads={n_heads}, num_codebooks={num_codebooks}, \
         sample_rate_hz={sample_rate_hz}, chord_vocab_size={chord_vocab_size}, \
         drum_vocab_size={drum_vocab_size}. Requested prompt_len={prompt_len} chars, \
         chords_len={chords_len} symbols, drums_len={drums_len} symbols, \
         duration_secs={duration_secs}. Primary sources: {hf_card} + \
         {audiocraft_repo} + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial \
         は fake-complete より honest') — no silent fabricated PCM stream ever emitted \
         (FR-EX-08).",
        variant_short = variant.short(),
        d_model = cfg.d_model,
        num_layers = cfg.num_layers,
        n_heads = cfg.n_heads,
        num_codebooks = cfg.num_codebooks,
        sample_rate_hz = cfg.sample_rate_hz,
        chord_vocab_size = cfg.chord_vocab_size,
        drum_vocab_size = cfg.drum_vocab_size,
        prompt_len = prompt.len(),
        chords_len = chords.len(),
        drums_len = drums.len(),
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
    //! Tests for the JASCO runtime binder — variant discrimination +
    //! per-variant config round-trip + negative-space round-trip on the
    //! loud-partial gates.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real inference this
    //! would be `generate(...)` returning real 32 kHz PCM, but the
    //! T5-base + JASCO-symbolic-encoder + AudioCraft flow-transformer +
    //! EnCodec-decode composition is deferred (see the module doc +
    //! [`Jasco::generate`] rustdoc). Fabricating a real-inference output
    //! would violate CLAUDE.md 教訓 (a) ("loud-partial は fake-complete
    //! より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Variant discrimination**: name → enum → per-variant default
    //!    config.
    //! 2. **Config round-trip**: `from_gguf` reads every axis stamped by
    //!    the converter; a separate test pins compatibility fallback for
    //!    older chunk-free artifacts.
    //! 3. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / missing name / unsupported
    //!    variant / empty tensor list / empty symbolic conditioning /
    //!    unsupported forward surface) fires at its documented surface
    //!    point, in the documented error variant.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a JASCO GGUF carrying the arch tag + name + one
    /// representative flow-matching transformer tensor whose outer dim
    /// matches `d_model`. The topology chunk group is optionally
    /// stamped (`stamp_topology = true`) — when omitted the runtime
    /// binder falls back to the per-variant primary-source defaults per
    /// key.
    ///
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn jasco_gguf(
        name: &str,
        cfg: JascoConfig,
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
            b.add_u32(GGUF_KEY_NUM_CODEBOOKS, cfg.num_codebooks);
            b.add_u32(GGUF_KEY_CODEC_FRAME_RATE_HZ, cfg.codec_frame_rate_hz);
            b.add_u32(GGUF_KEY_SAMPLE_RATE_HZ, cfg.sample_rate_hz);
            b.add_u32(GGUF_KEY_TEXT_PREFIX_LEN, cfg.text_prefix_len);
            b.add_u32(GGUF_KEY_CHORD_VOCAB_SIZE, cfg.chord_vocab_size);
            b.add_u32(GGUF_KEY_DRUM_VOCAB_SIZE, cfg.drum_vocab_size);
            b.add_u32(GGUF_KEY_NUM_FLOW_STEPS, cfg.num_flow_steps);
            b.add_f32(GGUF_KEY_CFG_SCALE, cfg.cfg_scale);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative flow-matching transformer tensor so the
        // non-emptiness gate passes and the shape-consistency accessor
        // has something to walk. The `d_model` dim is deliberately at
        // axis 0 so `matches_config` returns true. The tensor name
        // mirrors an audiocraft-style transformer projection weight.
        let d = cfg.d_model as u64;
        b.add_tensor(
            "flow_transformer.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // T1. Variant default config matches primary-source axes + invariants
    // -----------------------------------------------------------------------

    #[test]
    fn variant_default_config_matches_primary_source_axes_and_invariants() {
        // Pin the 400M Chords+Drums axes transcribed from paper §3
        // + AudioCraft family convention (musicgen-small style
        // transformer). A rename or axis-value drift would land here in
        // the same commit or fail this test.
        let cfg = JascoConfig::v_400m_chords_drums_default();
        assert_eq!(cfg.variant, JascoVariant::Chords400mDrums);
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.ffn_dim, 4096);
        assert_eq!(cfg.num_codebooks, 4);
        assert_eq!(cfg.codec_frame_rate_hz, 50);
        assert_eq!(cfg.sample_rate_hz, 32_000);
        assert_eq!(cfg.text_prefix_len, 512);

        // AudioCraft family design invariant: head_dim = 64 (a
        // deliberate choice that keeps the attention kernel stable
        // across scale-ups). A future variant that violates this would
        // need to be added deliberately, not by accidental silent
        // misconfiguration.
        assert_eq!(cfg.d_model / cfg.n_heads, 64, "head_dim = 64");

        // AudioCraft "4× hidden" FFN convention invariant.
        assert_eq!(cfg.ffn_dim, 4 * cfg.d_model, "ffn_dim = 4 × d_model");

        assert_eq!(cfg.chord_vocab_size, 195, "194 chords + null row");
        assert_eq!(cfg.drum_vocab_size, 128, "EnCodec drum latent width");
        assert_eq!(cfg.num_flow_steps, 100, "official Euler fallback steps");
        assert_eq!(cfg.cfg_scale, 5.0, "official all-condition CFG coefficient");

        // Variant discrimination via from_name matches the enum arm +
        // rejects known-unbound siblings + garbage strings.
        assert_eq!(
            JascoVariant::from_name(NAME_CHORDS_DRUMS_400M),
            Some(JascoVariant::Chords400mDrums)
        );
        assert_eq!(JascoVariant::from_name(NAME_CHORDS_DRUMS_1B), None);
        assert_eq!(JascoVariant::from_name(NAME_MELODY_400M), None);
        assert_eq!(JascoVariant::from_name(NAME_MELODY_1B), None);
        assert_eq!(JascoVariant::from_name("not-jasco"), None);

        // Accessors round-trip.
        assert_eq!(JascoVariant::Chords400mDrums.name(), NAME_CHORDS_DRUMS_400M);
        assert_eq!(JascoVariant::Chords400mDrums.arch(), ARCH);
        assert_eq!(JascoVariant::Chords400mDrums.short(), "Chords400mDrums");
        assert_eq!(
            JascoVariant::Chords400mDrums.primary_source_hf_card(),
            PRIMARY_SOURCE_HF_CARD_400M_CHORDS_DRUMS
        );
    }

    // -----------------------------------------------------------------------
    // T2. from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch_chunk() {
        // A GGUF with NO arch chunk cannot be routed to JASCO. Fail
        // loud (FR-EX-08 — never silently pretend it's JASCO).
        let mut b = GgufBuilder::new();
        // NO arch chunk.
        b.add_string(chunks::KEY_MODEL_NAME, NAME_CHORDS_DRUMS_400M);
        b.add_tensor(
            "flow_transformer.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch chunk, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // T3. from_gguf rejects sibling music-gen arch tag (never silently
    //     mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_sibling_music_gen_arch_tag() {
        // A `musicgen` GGUF handed to the JASCO binder by mistake must
        // fail loud with a specific message rather than silently mis-
        // binding (FR-EX-08). Both `musicgen` and `jasco_400m_chords_drums`
        // live in the music-generation family but have completely
        // different forward topologies (MusicGen is AR-over-EnCodec;
        // JASCO is flow-matching with joint symbolic conditioning), so
        // silent aliasing would misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "musicgen");
        b.add_string(chunks::KEY_MODEL_NAME, NAME_CHORDS_DRUMS_400M);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`musicgen`") && m.contains("`jasco_400m_chords_drums`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("joint text + chord + drum"),
                    "message should disambiguate JASCO's flow-matching + joint symbolic \
                     conditioning topology to help the reader, got `{m}`"
                );
                // Sibling neighbourhood enumeration must include the
                // full music-gen family so the reader has a checklist.
                assert!(
                    m.contains("magnet_small_10secs") && m.contains("melodyflow_t24_30secs"),
                    "message must enumerate sibling music-gen arch tags for context, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Also verify `magnet_medium_30secs` gets the same specific
        // mis-route message — pin every enumerated sibling.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "magnet_medium_30secs");
        b.add_string(chunks::KEY_MODEL_NAME, NAME_CHORDS_DRUMS_400M);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on magnet_medium_30secs arch");
        };
        assert!(
            matches!(err, VokraError::ModelLoad(_)),
            "sibling music-gen arch tag must fail with ModelLoad, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // T4. from_gguf rejects missing name chunk + unsupported variant
    //     name (with per-sibling "converter not yet in-tree" anchors)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_name_and_unsupported_variant_names() {
        // (a) Missing name chunk — correct arch but no name means we
        // can't discriminate the variant. The 400M Chords+Drums is the
        // only bound variant today, but the loader must still fail loud
        // rather than silently defaulting.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        // NO name chunk.
        b.add_tensor(
            "flow_transformer.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on missing name");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.name`"),
                    "message must call out the missing name chunk, got `{m}`"
                );
                assert!(
                    m.contains("jasco_400m_chords_drums"),
                    "message must name the expected variant name for the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (b) Unsupported variant name — `jasco_1b_chords_drums` gets
        // the "converter not yet in-tree" anchor with the paper URL.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_CHORDS_DRUMS_1B);
        b.add_tensor(
            "flow_transformer.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on unsupported 1B variant");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(NAME_CHORDS_DRUMS_1B),
                    "message must name the unsupported variant, got `{m}`"
                );
                assert!(
                    m.contains("converter not yet in-tree") || m.contains("follow-up wave"),
                    "message must anchor the reader on the follow-up path, got `{m}`"
                );
                assert!(
                    m.contains("huggingface.co/facebook/jasco-chords-drums-1B"),
                    "message must name the primary source URL for the unsupported variant, \
                     got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (c) The melody variant name must call out the chroma-
        // conditioning frontend the melody sibling needs.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_MELODY_400M);
        b.add_tensor(
            "flow_transformer.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on 400M melody variant");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(NAME_MELODY_400M),
                    "message must name the melody variant, got `{m}`"
                );
                assert!(
                    m.contains("chroma"),
                    "message must call out the additional chroma-conditioning frontend, \
                     got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (d) Garbage name that isn't in the JASCO family at all.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "not-a-jasco-name");
        b.add_tensor(
            "flow_transformer.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
            panic!("expected ModelLoad on garbage variant");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not-a-jasco-name"),
                    "message must echo the garbage name so the reader can spot the typo, \
                     got `{m}`"
                );
                assert!(
                    m.contains(NAME_CHORDS_DRUMS_400M),
                    "message must name the only supported variant for the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // T5. Empty tensor manifest fails loud (never binds all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        // Correct arch + name + full chunk group but zero tensors —
        // the JascoWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_CHORDS_DRUMS_400M);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Jasco::from_gguf(&file) else {
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
                    m.contains("jasco-400m-chords-drums"),
                    "message must name the converter to re-run, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // T6. Happy path: 400M Chords+Drums binds config + weights + variant
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_happy_path_400m_chords_drums_binds_config_and_weights() {
        // Full stamp + tensor list — the loader binds cleanly and every
        // axis rounds through.
        let cfg = JascoConfig::v_400m_chords_drums_default();
        let file = jasco_gguf(
            NAME_CHORDS_DRUMS_400M,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let j = Jasco::from_gguf(&file).expect("valid Chords400mDrums GGUF must bind");
        assert_eq!(j.variant(), JascoVariant::Chords400mDrums);
        // Config round-trip — every stamped axis reads back into the
        // same JascoConfig value (converter follow-up sub-wave path).
        assert_eq!(*j.config(), cfg);
        assert_eq!(j.config().d_model, 1024);
        assert_eq!(j.config().num_layers, 24);
        // NC weight license is the primary-source default per the HF
        // card (`license: cc-by-nc-4.0`) — the runtime must surface it
        // verbatim from the provenance chunk. The M2-13 compliance gate
        // refuses this artifact in commercial mode (T4 tier —
        // `--allow-noncommercial` opt-in required).
        assert_eq!(j.weight_license(), LicenseClass::NonCommercial);
        assert!(j.tensor_count() >= 1);
        // matches_config soft accessor picks up the fixture tensor.
        // (Access via `Jasco` accessors — the `weights` field is
        // deliberately private, mirror of the sibling musicgen /
        // sortformer / pyannote binders.)
        // We can still verify the accessor indirectly by rebuilding
        // JascoWeights with `from_gguf` and checking matches_config.
        let weights = JascoWeights::from_gguf(&file).expect("re-bind weights");
        assert!(
            weights.matches_config(j.config()),
            "at least one bound tensor must have an axis matching config.d_model"
        );
        // Sanity: a stale config (bogus d_model) does NOT match the
        // fixture — pins the accessor as a real check (not a stub that
        // always returns true).
        let stale = JascoConfig {
            d_model: 99_999,
            ..cfg
        };
        assert!(
            !weights.matches_config(&stale),
            "matches_config must return false for a d_model with no matching axis"
        );
    }

    // -----------------------------------------------------------------------
    // T7. generate loud-partial names all three primary sources + all
    //     three deferred pieces + all echoed config axes
    // -----------------------------------------------------------------------

    #[test]
    fn generate_returns_loud_partial_naming_all_three_primary_sources() {
        let cfg = JascoConfig::v_400m_chords_drums_default();
        let file = jasco_gguf(
            NAME_CHORDS_DRUMS_400M,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let j = Jasco::from_gguf(&file).unwrap();
        // A legitimate prompt + non-empty symbolic conditioning + valid
        // duration — the loud-partial gate must fire on the composition
        // surface, not on pre-generate arg validation.
        let Err(err) = j.generate("a jazz piano solo", &[0, 4, 5, 0], &[36, 40, 42, 46], 5.0)
        else {
            panic!("generate must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("jasco generate"),
                    "message must call out the jasco generate surface, got `{m}`"
                );
                // The three deferred pieces MUST all be named so the
                // follow-up wave has an unambiguous work anchor. Task
                // requires "chord encoder" / "drum encoder" /
                // "flow-matching" substrings.
                assert!(
                    m.contains("chord encoder"),
                    "message must name the JASCO chord encoder deferred piece, got `{m}`"
                );
                assert!(
                    m.contains("drum encoder"),
                    "message must name the JASCO drum encoder deferred piece, got `{m}`"
                );
                assert!(
                    m.contains("flow-matching"),
                    "message must name the flow-matching aspect, got `{m}`"
                );
                assert!(
                    m.contains("T5"),
                    "message must name the T5-base text encoder deferred piece, got `{m}`"
                );
                // The two LANDED composition anchors MUST be named so a
                // reader knows the primitives already exist.
                assert!(
                    m.contains("flow_sampler::flow_sample") && m.contains("encodec_rvq_decode"),
                    "message must name BOTH landed composition anchors \
                     (`vokra_ops::flow_sampler::flow_sample` + \
                     `vokra_ops::encodec_rvq_decode`), got `{m}`"
                );
                // All three primary source URLs must be cited — task
                // hint requires this.
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD_400M_CHORDS_DRUMS),
                    "message must contain the 400M Chords+Drums HF card URL substring \
                     ({PRIMARY_SOURCE_HF_CARD_400M_CHORDS_DRUMS}), got `{m}`"
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
                assert!(
                    m.contains("variant=Chords400mDrums"),
                    "variant axis missing: {m}"
                );
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
                assert!(
                    m.contains("chord_vocab_size="),
                    "chord_vocab_size axis missing: {m}"
                );
                assert!(
                    m.contains("drum_vocab_size="),
                    "drum_vocab_size axis missing: {m}"
                );
                // The call args must be echoed so a caller can cross-
                // check the request.
                assert!(
                    m.contains("prompt_len=17"),
                    "prompt_len (17 = len(\"a jazz piano solo\")) missing: {m}"
                );
                assert!(
                    m.contains("chords_len=4"),
                    "chords_len (4 = len([0,4,5,0])) missing: {m}"
                );
                assert!(
                    m.contains("drums_len=4"),
                    "drums_len (4 = len([36,40,42,46])) missing: {m}"
                );
                assert!(m.contains("duration_secs=5"), "duration_secs missing: {m}");
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08 no-silent-fabrication clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // T8. generate rejects empty symbolic conditioning BEFORE loud-partial
    //     (arg-validation ordering)
    // -----------------------------------------------------------------------

    #[test]
    fn generate_rejects_empty_symbolic_conditioning_before_loud_partial() {
        // Bind cleanly, then attempt generate with both `chords` and
        // `drums` empty — the InvalidArgument gate must fire BEFORE
        // the UnsupportedOp loud-partial so callers can distinguish
        // "bad args" from "not implemented yet".
        let cfg = JascoConfig::v_400m_chords_drums_default();
        let file = jasco_gguf(
            NAME_CHORDS_DRUMS_400M,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let j = Jasco::from_gguf(&file).unwrap();

        // (a) Both symbolic conditioning inputs empty → InvalidArgument.
        let Err(err) = j.generate("a jazz piano solo", &[], &[], 5.0) else {
            panic!("expected InvalidArgument on empty symbolic conditioning");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(
                    m.contains("symbolic conditioning is not optional"),
                    "message must call out the JASCO-specific 'symbolic conditioning \
                     is not optional' contract, got `{m}`"
                );
                assert!(
                    m.contains("paper §3.2"),
                    "message must cite the paper §3.2 source, got `{m}`"
                );
                assert!(
                    m.contains("musicgen"),
                    "message must name the sibling family (musicgen) as the correct \
                     text-only alternative, got `{m}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        // (b) Empty prompt → InvalidArgument (before any conditioning
        // check).
        let Err(err) = j.generate("", &[0, 4, 5, 0], &[], 5.0) else {
            panic!("expected InvalidArgument on empty prompt");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "empty prompt must be InvalidArgument, got {err:?}"
        );

        // (c) Non-finite duration → InvalidArgument.
        let Err(err) = j.generate("valid", &[0, 4, 5, 0], &[], f32::NAN) else {
            panic!("expected InvalidArgument on NaN duration");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "NaN duration must be InvalidArgument, got {err:?}"
        );

        // (d) Zero duration → InvalidArgument.
        let Err(err) = j.generate("valid", &[0, 4, 5, 0], &[], 0.0) else {
            panic!("expected InvalidArgument on zero duration");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "zero duration must be InvalidArgument, got {err:?}"
        );

        // (e) Negative duration → InvalidArgument.
        let Err(err) = j.generate("valid", &[0, 4, 5, 0], &[], -1.5) else {
            panic!("expected InvalidArgument on negative duration");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "negative duration must be InvalidArgument, got {err:?}"
        );

        // (f) Positive control: chords-only symbolic conditioning
        // (drums empty) succeeds past the arg-gate and hits the
        // loud-partial. Verifies the "at least one non-empty" contract
        // is honored on the OR side.
        let Err(err) = j.generate("valid", &[0, 4, 5, 0], &[], 5.0) else {
            panic!("chords-only should still fire loud-partial");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "chords-only + valid args must reach the loud-partial gate, got {err:?}"
        );

        // (g) Positive control: drums-only symbolic conditioning
        // (chords empty) succeeds past the arg-gate and hits the
        // loud-partial.
        let Err(err) = j.generate("valid", &[], &[36, 40], 5.0) else {
            panic!("drums-only should still fire loud-partial");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "drums-only + valid args must reach the loud-partial gate, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // T9. weight_license defaults to NonCommercial (T4 tier fail-closed)
    // -----------------------------------------------------------------------

    #[test]
    fn weight_license_defaults_to_noncommercial_from_provenance_stamp() {
        // The JASCO converter's DEFAULT_LICENSE_SPDX is `cc-by-nc-4.0`
        // → LicenseClass::NonCommercial. The runtime must surface this
        // verbatim so the M2-13 compliance gate refuses commercial-mode
        // load (T4 tier — the same fail-closed posture as X-Codec-2
        // precedent 2026-07-28 + MusicGen family + Sortformer
        // 2026-08-04).
        let cfg = JascoConfig::v_400m_chords_drums_default();
        let file = jasco_gguf(
            NAME_CHORDS_DRUMS_400M,
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let j = Jasco::from_gguf(&file).expect("bind with NC provenance");
        assert_eq!(
            j.weight_license(),
            LicenseClass::NonCommercial,
            "the JASCO converter defaults to NonCommercial per the HF card's \
             `license: cc-by-nc-4.0` — the runtime binder must surface it so the \
             M2-13 compliance gate can refuse commercial-mode load (T4 tier)"
        );

        // Missing provenance stamp falls back to Unknown (also fail-
        // closed at the gate).
        let file_no_license = jasco_gguf(
            NAME_CHORDS_DRUMS_400M,
            cfg,
            /*stamp_topology=*/ false,
            None,
        );
        let j_no_license = Jasco::from_gguf(&file_no_license).expect("bind without license stamp");
        assert_eq!(
            j_no_license.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed at the \
             M2-13 compliance gate)"
        );
    }

    // -----------------------------------------------------------------------
    // T10. Config from_gguf falls back to primary-source defaults on
    //      absent chunk group (converter's current stamp posture)
    // -----------------------------------------------------------------------

    #[test]
    fn config_from_gguf_falls_back_to_primary_source_defaults_when_chunk_group_absent() {
        // A GGUF produced before the `vokra.jasco.*` chunk group landed
        // must still load — the fallback path reads the per-variant
        // primary-source constants transcribed from paper §3 +
        // AudioCraft family convention. Mirror of MusicGenConfig::from_gguf
        // fallback pattern.
        let cfg_400m = JascoConfig::v_400m_chords_drums_default();
        let file = jasco_gguf(
            NAME_CHORDS_DRUMS_400M,
            cfg_400m,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let j = Jasco::from_gguf(&file).expect("chunk-free GGUF must bind via fallback");
        // Every axis fell through to its primary-source 400M default —
        // the loader returns the same values as v_400m_chords_drums_default().
        assert_eq!(j.config().d_model, DEFAULT_D_MODEL_400M_CHORDS_DRUMS);
        assert_eq!(j.config().num_layers, DEFAULT_NUM_LAYERS_400M_CHORDS_DRUMS);
        assert_eq!(j.config().n_heads, DEFAULT_N_HEADS_400M_CHORDS_DRUMS);
        assert_eq!(j.config().ffn_dim, DEFAULT_FFN_DIM_400M_CHORDS_DRUMS);
        assert_eq!(j.config().num_codebooks, NUM_CODEBOOKS);
        assert_eq!(j.config().codec_frame_rate_hz, CODEC_FRAME_RATE_HZ);
        assert_eq!(j.config().sample_rate_hz, SAMPLE_RATE_HZ);
        assert_eq!(j.config().text_prefix_len, DEFAULT_TEXT_PREFIX_LEN);
        assert_eq!(j.config().chord_vocab_size, DEFAULT_CHORD_VOCAB_SIZE);
        assert_eq!(j.config().drum_vocab_size, DEFAULT_DRUM_VOCAB_SIZE);
        assert_eq!(j.config().num_flow_steps, DEFAULT_NUM_FLOW_STEPS);
        // f32 comparison uses bit-identical match — DEFAULT_CFG_SCALE
        // was written verbatim.
        assert!(
            (j.config().cfg_scale - DEFAULT_CFG_SCALE).abs() < f32::EPSILON,
            "cfg_scale fallback must match DEFAULT_CFG_SCALE"
        );
    }

    // -----------------------------------------------------------------------
    // T11. Arch tag is stable and distinct from sibling music-generation
    //      arches (reciprocal defense — sortformer / musicgen sibling test
    //      mirror)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_music_generation_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. The sibling music-generation arch
        // tags MUST NOT collide with ours — they live in the same
        // taxonomy neighbourhood but have completely different forward
        // topologies.
        assert_eq!(ARCH, "jasco_400m_chords_drums");
        assert_eq!(NAME_CHORDS_DRUMS_400M, "jasco_400m_chords_drums");
        assert_eq!(NAME_CHORDS_DRUMS_1B, "jasco_1b_chords_drums");
        assert_eq!(NAME_MELODY_400M, "jasco_400m_melody");
        assert_eq!(NAME_MELODY_1B, "jasco_1b_melody");

        // Direct string comparisons against sibling arch tags to
        // document the "which sibling should NOT be aliased" contract
        // at test time (a future rename of any sibling arch would land
        // here in the same commit or fail this test).
        assert_ne!(
            ARCH, "musicgen",
            "jasco (flow-matching + joint symbolic) and musicgen (AR-over-EnCodec) \
             share the music-generation taxonomy but have completely different \
             forward topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "magnet_small_10secs",
            "jasco (flow-matching) and magnet_small_10secs (non-AR masked-LM) — \
             different forward topologies, mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "magnet_medium_30secs",
            "jasco and magnet_medium_30secs — same rationale as small — mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "melodyflow_t24_30secs",
            "jasco (flow-matching + chord/drum symbolic conditioning) and \
             melodyflow_t24_30secs (flow-matching + text/audio dual-prefix editing) \
             are the closest siblings but have different conditioning stacks — \
             sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "audiogen",
            "jasco (text-to-music with symbolic conditioning) and audiogen (sound-\
             effects LM) share the AR-LM taxonomy at the coarser level but produce \
             different modalities — mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "audioldm2",
            "jasco (flow-matching DiT) and audioldm2 (score-based latent diffusion \
             U-Net) — different sampler stacks — mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "stable_audio_open_small",
            "jasco (flow-matching + chord/drum symbolic) and stable_audio_open_small \
             (DiT + audio VAE) — different conditioning stacks — mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "bs_roformer",
            "jasco (music generation) and bs_roformer (music source separation) — \
             different task entirely — mis-route (FR-EX-08)"
        );
    }
}
