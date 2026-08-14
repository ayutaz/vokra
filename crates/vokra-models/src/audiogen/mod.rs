//! **Meta AudioGen-Medium** (`facebook/audiogen-medium`, CC-BY-NC-4.0 —
//! T4 tier) — text-to-audio autoregressive transformer LM runtime binder
//! (2026-08-14 audit follow-up Wave 6, second **audio generation**
//! runtime binder in the tree after MusicGen).
//!
//! # Primary source
//!
//! - HF model card:
//!   <https://huggingface.co/facebook/audiogen-medium>
//! - AudioCraft reference implementation (MIT code, non-commercial
//!   weights): <https://github.com/facebookresearch/audiocraft>
//!   (`audiocraft/models/audiogen.py` + `audiocraft/models/lm.py` — the
//!   `AudioGen` handle + shared `LMModel` autoregressive transformer LM
//!   used by the MusicGen sibling as well).
//! - Paper: Kreuk et al., *"AudioGen: Textually Guided Audio Generation"*,
//!   ICLR 2023 (arXiv:2209.15352).
//! - Weight license: **CC-BY-NC-4.0** (Meta AudioCraft weight policy;
//!   the code layer at `github.com/facebookresearch/audiocraft` is MIT
//!   but the trained weights are non-commercial). `docs/license-audit.md`
//!   §3.1 row 402 = ☑ Research-only 2026-08-01 yousan — **X-Codec-2 T4
//!   precedent** inheritance (2026-07-28).
//!
//! # Distinct from MusicGen sibling
//!
//! AudioGen shares the AR-LM-over-EnCodec-RVQ + frozen-T5-text-encoder
//! topology with MusicGen (Kreuk et al. reused the AudioCraft `LMModel`
//! spine that Copet et al. later hardened for MusicGen). The families
//! diverge along **modality** (SFX / environmental sounds vs music) and
//! **training corpus**; a future modality-specific head (SFX-only
//! conditioning stack, per-class embedding table, stereo output head)
//! must not silent-mis-bind against MusicGen's music-only runtime path.
//! Hence the distinct `audiogen` arch tag — FR-EX-08 dispatch safety
//! (2026-08-14 audit follow-up retags the converter's arch from the
//! Wave 5 shared `musicgen` tag).
//!
//! # Architecture (transcribed from primary sources)
//!
//! ```text
//! text prompt (UTF-8 string)
//!   -> frozen T5-base text encoder                   ← **loud-partial**
//!        (google-t5/t5-base — HF transformers `T5EncoderModel`,
//!         encoder-only, no reusable primitive in `vokra_ops` today;
//!         the follow-up wave lands a T5-base implementation or a
//!         first-class `t5_text_encode` op — SHARED with MusicGen,
//!         so the same fix unblocks both binders.)
//!   -> autoregressive transformer LM with            ← **loud-partial**
//!      4-codebook delay pattern and text-conditioned
//!      cross-attention to T5 tokens
//!        (Kreuk et al. Algorithm 1 — reused in Copet et al. MusicGen;
//!         the delay-pattern interleave across the 4 RVQ streams +
//!         cross-attention to T5-encoded prompt tokens are shared with
//!         the MusicGen sibling. The single-codebook AR LMs already in
//!         the tree — cosyvoice2 / moshi / voxtral — cover neither the
//!         delay pattern nor the text-encoder cross-attention.)
//!   -> EnCodec RVQ decode (4 codebooks, 50 Hz frame ← **primitive
//!      rate, 32 kHz PCM output)                        exists**
//!        (available via `vokra_ops::encodec_rvq_decode` — the ONE
//!         landed anchor cited in the loud-partial message so a reader
//!         knows the composition anchor; identical bundled codec as
//!         MusicGen.)
//!   -> PCM (mono f32, 32 kHz)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`AudioGen::from_gguf`] with strict `vokra.model.arch == "audiogen"`
//!     validation + `vokra.model.name == "audiogen-medium"` discriminator
//!     (AudioGen ships as a single variant on HF today; the single-
//!     variant simplification lets us skip the `MusicGenVariant`-style
//!     enum that MusicGen needed for Small vs Medium — a future
//!     `audiogen-large` release would land the enum in a follow-up
//!     wave).
//!   - [`AudioGenConfig::from_gguf`] with primary-source constant
//!     fallback per key (the AudioGen converter does NOT currently stamp
//!     the `vokra.audiogen.*` chunk group — only arch / name / category /
//!     upstream_hf / provenance — so a *strict* reader would refuse the
//!     already-published `huggingface.co/facebook/audiogen-medium` GGUF.
//!     Primary source is well-established (HF `config.json` + AudioCraft
//!     code + paper), so fallback does not fabricate axes; a future
//!     converter sub-wave that adds the stamps upgrades this reader to
//!     real-stamped reads per-key with no runtime code change — mirror
//!     of the Sortformer / PyanNet / MusicGen fallback pattern).
//!   - [`AudioGenWeights::from_gguf`] with a floor of non-empty tensor
//!     count enforced loud (a GGUF that carries zero tensors is refused
//!     rather than silently running an all-zero forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::NonCommercial`] per the AudioGen converter's
//!     stamped `cc-by-nc-4.0` — T4 tier, fail-closed at the runtime
//!     compliance gate M2-13).
//!
//! - **Loud-partial (this WP)**: [`AudioGen::generate`] returns
//!   [`VokraError::UnsupportedOp`] naming **three** deferred pieces:
//!   1. the frozen T5-base text encoder forward (upstream
//!      `transformers.T5EncoderModel`; no reusable primitive in
//!      `vokra_ops` today; the follow-up wave lands the T5-base body or
//!      a first-class `t5_text_encode` op — MusicGen shares this exact
//!      gap and their solution will unblock both);
//!   2. the autoregressive transformer LM decode with the **4-codebook
//!      delay pattern** (Kreuk et al. Algorithm 1, shared with Copet et
//!      al. MusicGen) + text-conditioned **cross-attention** to the
//!      T5-encoded prompt tokens (the single-codebook AR LMs already
//!      in the tree — cosyvoice2 / moshi / voxtral — cover neither the
//!      delay pattern nor the text-encoder cross-attention);
//!   3. the EnCodec RVQ decode from the 4 codebook streams to 32 kHz PCM
//!      — **available via `vokra_ops::encodec_rvq_decode`**, cited as
//!      the one landed piece so a reader diagnosing this gap knows the
//!      composition anchor.
//!
//! The error names the **three primary source URLs** (HF card +
//! AudioCraft repo + arXiv:2209.15352 paper), the config axes echoed
//! (`d_model`, `num_layers`, `n_heads`, `num_codebooks`,
//! `sample_rate_hz`), and the prompt length + duration so a reader
//! diagnosing this gap has exactly three places to walk. **No fabricated
//! PCM stream is ever emitted** (FR-EX-08).
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / sortformer / musicgen loud-partial precedent,
//! CLAUDE.md 教訓 (a) — "loud-partial は fake-complete より honest"):
//! the surrounding scaffold + `from_gguf` chunk-group validation +
//! FR-EX-08 loud-fails land today so a follow-up wave can flip the
//! switch by (i) landing the T5-base text encoder body (shared with
//! MusicGen), (ii) implementing the delay-pattern + cross-attention AR
//! transformer LM decode (shared with MusicGen), and (iii) wiring the
//! composed loop to the existing `vokra_ops::encodec_rvq_decode`
//! primitive. Because pieces (i) and (ii) are shared with MusicGen the
//! composition wave lands both binders at once.
//!
//! # `vokra.audiogen.*` chunk group (read here — fallback-friendly)
//!
//! The AudioGen converter (`crates/vokra-convert/src/models/
//! audiogen_medium.rs`) currently stamps only the arch / name / category
//! / upstream_hf / provenance chunks. The topology chunk group is READ
//! by this binder but any absent key falls back to the **primary-source
//! constant** so an already-published GGUF loads correctly. A future
//! converter sub-wave that adds `vokra.audiogen.*` stamps will override
//! the fallback automatically per-key with no runtime code change.
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"audiogen"`).
//!   Deliberately distinct from every sibling music/audio-generation
//!   arch — `musicgen` / `magnet_small_10secs` / `magnet_medium_30secs`
//!   / `melodyflow_t24_30secs` / `jasco_400m_chords_drums` / `audioldm2`
//!   / `stable_audio_open_small` / `ace_step` / `bs_roformer`. Silently
//!   sharing an arch tag with `musicgen` (or any sibling) would mis-
//!   route the runtime dispatch when a future modality-specific head
//!   ships — FR-EX-08 forbids the silent-wrong shape mismatch.
//! - `vokra.model.name` (`String`): must equal [`NAME`]
//!   (`"audiogen-medium"`) — the single-variant discriminator today.
//! - `vokra.audiogen.{d_model, num_layers, n_heads, ffn_dim, vocab_size,
//!   num_codebooks, codec_frame_rate_hz, sample_rate_hz}` (`u32` each):
//!   the composite topology axes. Fallback constants transcribed from
//!   HF `config.json` (see the `DEFAULT_*` constants for the primary-
//!   source anchors). The AudioGen-Medium release ships with the same
//!   1.5B-family axes as MusicGen-Medium — this is a genuine coincidence
//!   of the shared AudioCraft `LMModel` spine + the shared bundled 32
//!   kHz EnCodec codec, NOT a fabrication (the HF config.json is the
//!   primary source, not a MusicGen sibling extrapolation).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03 / M2-13) can classify the
//!   artifact without re-inspecting the safetensors provenance. The
//!   AudioGen converter stamps `NonCommercial` by default per the HF
//!   card's `license: cc-by-nc-4.0` — a caller who legitimately holds
//!   the weight under a distinct SPDX overrides at
//!   `vokra-cli convert --license <spdx>` and the stamped class re-
//!   derives via `LicenseClass::from_license_str`.
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`NAME`] / topology keys) — same rule the sibling
//! BF16 pass-through binders (`musicgen` / `sortformer` / `pyannote` /
//! `snac` / `hifigan` / `beat_this` / `mt3`) use so `vokra-models` does
//! not gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`, `vokra-core →
//! GGUF reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! AudioGen ships safetensors + PyTorch pickle upstream; this runtime
//! **never** touches ONNX (FR-LD-05 / NFR-DS-02). If the upstream
//! release ships pickle only, callers pre-flatten offline via
//! `tools/parity/audiogen_medium_prepare_checkpoint.py` (a thin wrapper
//! over `bin_to_safetensors.py`; an uv-managed Python 3.12 sidecar per
//! memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]` —
//! not part of the runtime), mirroring the SpeechT5-HiFi-GAN /
//! Sortformer / Charsiu / MusicGen bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/audiogen_medium.rs`. See module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model audiogen-medium`.
///
/// **Distinct from `musicgen`** — same AR-LM topology today but different
/// training corpus + output modality (SFX vs music). FR-EX-08 keeps them
/// dispatch-separated so a future modality-specific head (SFX-only
/// conditioning stack, stereo output head, per-class embedding table)
/// does not silent-mis-bind against MusicGen's music-only runtime path.
/// The two families share the `category = "music"` taxonomy tag but the
/// arch tag is the runtime dispatch discriminator. Sibling music/audio-
/// generation arch tags MUST NOT collide — `musicgen` /
/// `magnet_small_10secs` / `magnet_medium_30secs` /
/// `melodyflow_t24_30secs` / `jasco_400m_chords_drums` / `audioldm2` /
/// `stable_audio_open_small` / `ace_step` / `bs_roformer` all live in
/// the same music-generation taxonomy neighbourhood but have completely
/// different forward topologies (MusicGen is text-to-music AR;
/// AudioGen is text-to-SFX AR; MAGNeT is non-autoregressive masked-LM;
/// MelodyFlow is DiT flow-matching; AudioLDM2 / Stable-Audio-Open are
/// diffusion; ACE-Step is chunked-AR; BS-Roformer is source-separation).
/// Audit follow-up 2026-08-14 retags the converter's arch from the
/// Wave 5 shared `musicgen` tag to enforce this dispatch boundary.
pub const ARCH: &str = "audiogen";

/// Expected `vokra.model.name` value for the sole variant — matches the
/// `huggingface.co/facebook/audiogen-medium` upstream slug + the
/// converter's `NAME` constant.
///
/// AudioGen ships as a single variant on HF today (`facebook/audiogen-
/// medium` at 1.5B params). If Meta releases an `audiogen-small` /
/// `audiogen-large` in the future, a follow-up wave will lift this
/// single string to an `AudioGenVariant` enum mirroring
/// [`crate::musicgen::MusicGenVariant`].
pub const NAME: &str = "audiogen-medium";

/// Shared taxonomy category with the MusicGen sibling — both are audio-
/// generation members of the `music` taxonomy branch (per license-audit
/// §3.1 row 402 rationale: "audio-generation taxonomy tree = text-to-
/// audio 全般、silently sharing `tts` would misroute").
pub const CATEGORY: &str = "music";

/// `vokra.audiogen.d_model` — transformer LM hidden dim.
/// Primary-source default: 1536 (AudioGen-Medium is a 1.5B-parameter
/// model matching the MusicGen-Medium hyper-shape via the shared
/// AudioCraft `LMModel` spine).
pub const GGUF_KEY_D_MODEL: &str = "vokra.audiogen.d_model";
/// `vokra.audiogen.num_layers` — transformer LM depth.
/// Primary-source default: 48.
pub const GGUF_KEY_NUM_LAYERS: &str = "vokra.audiogen.num_layers";
/// `vokra.audiogen.n_heads` — multi-head attention head count.
/// Primary-source default: 24. `head_dim = d_model / n_heads = 64` (same
/// AudioCraft convention as MusicGen).
pub const GGUF_KEY_N_HEADS: &str = "vokra.audiogen.n_heads";
/// `vokra.audiogen.ffn_dim` — feedforward inner dimension.
/// Primary-source default: 6144 (AudioCraft "4× hidden" convention).
pub const GGUF_KEY_FFN_DIM: &str = "vokra.audiogen.ffn_dim";
/// `vokra.audiogen.vocab_size` — per-codebook token vocabulary size.
/// Shared with MusicGen: 2048 (the bundled EnCodec 32 kHz RVQ codebook
/// size, one entry per codebook — the LM emits `num_codebooks` streams
/// each of this vocab size).
pub const GGUF_KEY_VOCAB_SIZE: &str = "vokra.audiogen.vocab_size";
/// `vokra.audiogen.num_codebooks` — number of RVQ codebook streams the
/// LM emits per frame. Shared with MusicGen: 4 (the bundled EnCodec 32
/// kHz codec configuration).
pub const GGUF_KEY_NUM_CODEBOOKS: &str = "vokra.audiogen.num_codebooks";
/// `vokra.audiogen.codec_frame_rate_hz` — the EnCodec 32 kHz output
/// frame rate. Shared with MusicGen: 50 Hz.
pub const GGUF_KEY_CODEC_FRAME_RATE_HZ: &str = "vokra.audiogen.codec_frame_rate_hz";
/// `vokra.audiogen.sample_rate_hz` — the paired EnCodec sample rate.
/// Shared with MusicGen: 32000 Hz (32 kHz).
pub const GGUF_KEY_SAMPLE_RATE_HZ: &str = "vokra.audiogen.sample_rate_hz";

// Primary-source constants transcribed from the HF model card's
// `config.json` + the AudioCraft `EncodecModel` bundle (fetched 2026-08-14
// — CLAUDE.md 「ハルシネーション厳禁」). AudioGen-Medium shares the
// 1.5B-family axes with MusicGen-Medium via the shared AudioCraft
// `LMModel` spine + the shared bundled 32 kHz EnCodec codec — this is a
// genuine coincidence of the primary source, NOT a MusicGen sibling
// extrapolation.

/// Transformer LM hidden dim (`d_model`). Primary source:
/// `huggingface.co/facebook/audiogen-medium/config.json`
/// (`decoder.hidden_size`). Matches AudioCraft 1.5B `LMModel` spine.
pub const DEFAULT_D_MODEL: u32 = 1536;
/// Transformer LM depth (`num_hidden_layers`). Primary source:
/// `audiogen-medium/config.json` (`decoder.num_hidden_layers`).
pub const DEFAULT_NUM_LAYERS: u32 = 48;
/// Multi-head attention head count (`num_attention_heads`). Primary
/// source: `audiogen-medium/config.json` (`decoder.num_attention_heads`).
/// `head_dim = 1536 / 24 = 64`.
pub const DEFAULT_N_HEADS: u32 = 24;
/// Feedforward inner dimension (`ffn_dim`). Primary source:
/// `audiogen-medium/config.json` (`decoder.ffn_dim`). AudioCraft "4×
/// hidden" convention: `6144 = 4 × 1536`.
pub const DEFAULT_FFN_DIM: u32 = 6144;
/// Per-codebook vocabulary size. Primary source: HF `config.json`
/// (`decoder.vocab_size`) + AudioCraft `EncodecModel.quantizer.bins`.
/// Shared with MusicGen: the bundled EnCodec 32 kHz codec is a 4-codebook
/// RVQ with 2048 entries per codebook.
pub const DEFAULT_VOCAB_SIZE: u32 = 2048;
/// Number of RVQ codebook streams the LM emits per frame. Primary
/// source: AudioCraft `AudioGen(...).lm.n_q = 4` (shared with MusicGen).
pub const NUM_CODEBOOKS: u32 = 4;
/// EnCodec output frame rate for the bundled 32 kHz codec (matches
/// AudioCraft `EncodecModel.frame_rate = 50`). Shared with MusicGen.
pub const CODEC_FRAME_RATE_HZ: u32 = 50;
/// EnCodec sample rate for the bundled 32 kHz codec (matches AudioCraft
/// `EncodecModel.sample_rate = 32000`). Shared with MusicGen.
pub const SAMPLE_RATE_HZ: u32 = 32_000;

/// Primary-source anchor for the HF model card. Cited in the loud-partial
/// error so a reader diagnosing the gap knows the definitive artifact
/// source.
pub const PRIMARY_SOURCE_HF_CARD: &str = "huggingface.co/facebook/audiogen-medium";
/// Primary-source anchor for the AudioCraft reference repository
/// (MIT code — the tensor-name walk anchor). Cited in the loud-partial
/// error so a reader knows the code reference. Shared with MusicGen —
/// the same repo hosts both families.
pub const PRIMARY_SOURCE_AUDIOCRAFT_REPO: &str = "github.com/facebookresearch/audiocraft";
/// Paper anchor (Kreuk et al. ICLR 2023) — cited alongside the HF card
/// + AudioCraft repo so a reader has the theoretical context as well.
/// DISTINCT from MusicGen's arXiv:2306.05284 (Copet et al. NeurIPS 2023)
/// — the two are sibling papers but AudioGen came first and pioneered
/// the delay-pattern + text-encoder-cross-attention architecture that
/// MusicGen later refined.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2209.15352";

// ---------------------------------------------------------------------------
// AudioGenConfig — the composite topology axes read from the
// `vokra.audiogen.*` chunk group, with primary-source constant fallback
// (the AudioGen converter does not currently stamp this chunk group —
// the fallback is honest because the primary source is well-established;
// a future converter sub-wave that adds the stamps upgrades this reader
// to real-stamped reads seamlessly). Mirror of
// [`crate::musicgen::MusicGenConfig::from_gguf`] +
// [`crate::sortformer_diar_4spk_v1::SortformerConfig::from_gguf`] +
// [`crate::pyannote::PyanNetConfig::from_gguf`].
// ---------------------------------------------------------------------------

/// AudioGen hyperparameters as they ride the `vokra.audiogen.*` chunk
/// group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key — a GGUF that never carried the chunk still
/// loads with the upstream defaults transcribed from HF `config.json`.
/// Every numeric axis is `u32` in the GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioGenConfig {
    /// Transformer LM hidden dim (default 1536).
    pub d_model: u32,
    /// Transformer LM depth (default 48).
    pub num_layers: u32,
    /// Multi-head attention head count (default 24).
    /// `head_dim = d_model / n_heads = 64`.
    pub n_heads: u32,
    /// Feedforward inner dimension (default 6144). AudioCraft "4×
    /// hidden" convention: `ffn_dim = 4 × d_model`.
    pub ffn_dim: u32,
    /// Per-codebook vocabulary size (2048).
    pub vocab_size: u32,
    /// Number of RVQ codebook streams the LM emits per frame (4).
    pub num_codebooks: u32,
    /// EnCodec output frame rate (50 Hz).
    pub codec_frame_rate_hz: u32,
    /// EnCodec sample rate (32000 Hz = 32 kHz).
    pub sample_rate_hz: u32,
}

impl AudioGenConfig {
    /// Primary-source-transcribed axes as a `const` — the fallback
    /// baseline used by [`from_gguf`](Self::from_gguf) when the chunk
    /// group is absent (the current converter's default posture).
    #[must_use]
    pub const fn primary_source_default() -> Self {
        Self {
            d_model: DEFAULT_D_MODEL,
            num_layers: DEFAULT_NUM_LAYERS,
            n_heads: DEFAULT_N_HEADS,
            ffn_dim: DEFAULT_FFN_DIM,
            vocab_size: DEFAULT_VOCAB_SIZE,
            num_codebooks: NUM_CODEBOOKS,
            codec_frame_rate_hz: CODEC_FRAME_RATE_HZ,
            sample_rate_hz: SAMPLE_RATE_HZ,
        }
    }

    /// Reads every `vokra.audiogen.*` chunk from `gguf`, falling back to
    /// the primary-source defaults per absent key.
    ///
    /// The AudioGen converter does not currently stamp this chunk group
    /// (only arch / name / category / upstream_hf / provenance), so on
    /// an already-published GGUF every axis falls through to its
    /// primary-source default. A future converter sub-wave that adds the
    /// stamps upgrades this reader to real-stamped reads per-key with no
    /// runtime code change.
    ///
    /// Mirror of
    /// [`crate::musicgen::MusicGenConfig::from_gguf`] +
    /// [`crate::sortformer_diar_4spk_v1::SortformerConfig::from_gguf`]
    /// + [`crate::pyannote::PyanNetConfig::from_gguf`] — the same
    /// fallback pattern used for converters whose topology-stamp
    /// sub-wave is still queued.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::primary_source_default();
        Self {
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
// AudioGenWeights — bound the tensor manifest with a non-emptiness gate.
// Under the loud-partial WP the weights are counted but the T5-base
// encoder + AR LM (delay pattern + cross-attention) + EnCodec RVQ
// decode composition is deferred. Mirror of `MusicGenWeights` /
// `SortformerWeights` / `Mt3Weights` / `BeatThisWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an AudioGen GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// AudioGen checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future
/// `AudioGenWeights::bind_t5_encoder_weights` /
/// `AudioGenWeights::bind_lm_decoder_weights` tensor walk can find its
/// inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct AudioGenWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// T5-encoder + AR-LM + EnCodec-decode composition wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl AudioGenWeights {
    /// Scans `gguf` for the AudioGen state_dict tensors. Refuses to bind
    /// if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is
    /// never a valid AudioGen checkpoint).
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
                "audiogen: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model audiogen-medium` \
                 against a `facebook/audiogen-medium` safetensors checkpoint (the upstream \
                 release ships a bundle of LM decoder + T5-base text encoder + EnCodec RVQ \
                 codec — every group must be present)."
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
    pub fn matches_config(&self, config: &AudioGenConfig) -> bool {
        let d = config.d_model as usize;
        self.tensors
            .iter()
            .any(|(_, dims)| dims.iter().any(|&x| x == d))
    }
}

// ---------------------------------------------------------------------------
// AudioGen — the runtime binder handle
// ---------------------------------------------------------------------------

/// Meta AudioGen text-to-audio autoregressive transformer LM runtime
/// binder (`facebook/audiogen-medium`, CC-BY-NC-4.0 T4 tier).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`generate`](Self::generate) with a text prompt + duration to obtain
/// a `Vec<f32>` of 32 kHz PCM samples. See the module doc for the
/// current implementation-status matrix and the FR-EX-08 loud-error
/// contract on the T5-base + AR-LM + EnCodec-decode composition.
#[derive(Debug)]
pub struct AudioGen {
    config: AudioGenConfig,
    // The bound weights are held (real, counted) but the T5-encoder +
    // AR-LM + EnCodec-decode composition is a follow-up wave; the field
    // is deliberately `#[allow(dead_code)]` until the composition lands
    // so a reader is not misled by an unused field. Same posture as
    // MusicGen / RMVPE / pyannote / mt3 / beat_this / sortformer.
    #[allow(dead_code)]
    weights: AudioGenWeights,
    weight_license: LicenseClass,
}

impl AudioGen {
    /// Binds an AudioGen GGUF: validates arch, validates name, reads the
    /// topology chunk group (with primary-source constant fallback per
    /// key), discovers tensors, and surfaces the stamped weight-license
    /// class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"audiogen"` (a sibling music/audio-generation GGUF handed
    ///   to us by mistake — `musicgen` / `magnet_small_10secs` /
    ///   `melodyflow_t24_30secs` / … — fails with a clear message
    ///   instead of a downstream missing-tensor).
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is absent, or
    ///   when it is not `"audiogen-medium"` (a future release such as
    ///   `audiogen-large` would trigger a follow-up wave to lift the
    ///   discriminator to an `AudioGenVariant` enum).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`AudioGenWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error. The Wave 6 (2026-08-14) audit retag
        //    from the Wave 5 shared `musicgen` tag means an already-
        //    produced pre-Wave-6 AudioGen GGUF (arch = "musicgen") will
        //    fail here — this is deliberate; the fix is to re-run
        //    `vokra-cli convert --model audiogen-medium`.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "audiogen: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model audiogen-medium`? Note \
                     that Wave 6 audit follow-up 2026-08-14 retagged AudioGen from the \
                     Wave 5 shared `musicgen` arch tag to the distinct `audiogen` tag — \
                     an already-produced pre-Wave-6 AudioGen GGUF stamped `musicgen` \
                     will fail this check and needs re-conversion). Note that the \
                     sibling music/audio-generation arch tags — `musicgen` (Meta \
                     AudioCraft MusicGen family text-to-music AR LM — same AR-LM + \
                     delay-pattern + T5-encoder-cross-attention topology as AudioGen but \
                     different modality), `magnet_small_10secs` / `magnet_medium_30secs` \
                     (Meta AudioCraft non-autoregressive masked-LM), \
                     `melodyflow_t24_30secs` (Meta AudioCraft DiT flow-matching editing), \
                     `jasco_400m_chords_drums` (chord/drum-conditioned AR LM), \
                     `audioldm2` / `stable_audio_open_small` (latent diffusion), \
                     `ace_step` (chunked-AR), `bs_roformer` (source-separation) — all \
                     live in the same music-generation neighbourhood but have completely \
                     different forward topologies; AudioGen's autoregressive transformer \
                     LM with 4-codebook delay pattern + T5 text-encoder cross-attention \
                     is topology-shared with MusicGen but modality-distinct (SFX vs \
                     music) and silently aliasing arch would misroute the runtime \
                     dispatch when a future modality-specific head ships, FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "audiogen: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native audiogen GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Name check — AudioGen ships as a single variant on HF today
        //    (`audiogen-medium`), so the name chunk is a pure
        //    discriminator against future family variants that might
        //    reuse the arch tag. Missing name is loud (FR-EX-08); an
        //    unrecognised name (`audiogen-large` / `audiogen-small` /
        //    `audiogen-stereo`) is loud with the "follow-up wave adds
        //    AudioGenVariant enum" anchor so a reader diagnosing the
        //    mismatch has a path forward.
        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "audiogen: GGUF is missing `vokra.model.name` (converter did not \
                     stamp it — cannot discriminate AudioGen from a future family \
                     variant such as `audiogen-large` / `audiogen-stereo`)"
                        .to_owned(),
                )
            })?;
        if name != NAME {
            return Err(VokraError::ModelLoad(format!(
                "audiogen: NAME `{name}` is not a recognised AudioGen variant. \
                 Expected `{NAME}`. AudioGen ships as a single variant on HF today \
                 (`facebook/audiogen-medium` at 1.5B params); if Meta released \
                 `audiogen-large` / `audiogen-small` / `audiogen-stereo` the runtime \
                 needs a follow-up wave to lift this discriminator to an \
                 `AudioGenVariant` enum mirroring `crate::musicgen::MusicGenVariant`. \
                 Primary source: {PRIMARY_SOURCE_HF_CARD}."
            )));
        }

        // 3. Topology axes from the `vokra.audiogen.*` chunk group
        //    (fallback-friendly — see the module doc for the AudioGen
        //    converter's stamp posture).
        let config = AudioGenConfig::from_gguf(file);

        // 4. Load the tensor manifest with the non-emptiness gate.
        let weights = AudioGenWeights::from_gguf(file)?;

        // 5. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The AudioGen
        //    converter defaults to `NonCommercial` per the HF card's
        //    `license: cc-by-nc-4.0`; a caller override at `--license
        //    <spdx>` re-derives the class. Missing provenance falls
        //    back to `Unknown` which is fail-closed at the M2-13
        //    compliance gate — same posture as MusicGen / MT3 /
        //    Sortformer.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weights,
            weight_license,
        })
    }

    /// The bound topology axes (from `vokra.audiogen.*` chunk group with
    /// primary-source constant fallback).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &AudioGenConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The AudioGen converter
    /// stamps `NonCommercial` by default per the HF card's `license:
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
    /// on the text `prompt` (an environmental-sound / SFX description
    /// such as "dog barking on a wooden porch").
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — AudioGen's inference path
    /// requires **three** deferred pieces (all shared with the MusicGen
    /// sibling — a single follow-up wave unblocks both binders):
    ///
    /// 1. **Frozen T5-base text encoder forward**: the upstream release
    ///    freezes a `google-t5/t5-base` encoder for text conditioning
    ///    (HF transformers `T5EncoderModel`); no reusable primitive
    ///    exists in `vokra_ops` today. The follow-up wave lands either
    ///    a T5-base implementation dedicated to AudioCraft models or a
    ///    first-class `t5_text_encode` op that other future consumers
    ///    can share.
    /// 2. **Autoregressive transformer LM decode with 4-codebook delay
    ///    pattern + text-conditioned cross-attention**: Kreuk et al.
    ///    Algorithm 1 (later refined in Copet et al. MusicGen)
    ///    describes the delay-pattern interleave across the 4 RVQ
    ///    codebook streams, plus the cross-attention over the T5-encoded
    ///    prompt tokens. The single-codebook autoregressive LMs already
    ///    in the tree (`cosyvoice2` / `moshi` / `voxtral`) cover neither
    ///    the delay pattern nor the text-encoder cross-attention.
    /// 3. **EnCodec RVQ decode**: available via
    ///    `vokra_ops::encodec_rvq_decode` — this is the **ONE landed
    ///    anchor** the loud-partial message cites so a reader
    ///    diagnosing this gap knows the composition anchor. The
    ///    bundled 32 kHz codec is identical to MusicGen's.
    ///
    /// The error names **three** primary source URLs (HF card +
    /// AudioCraft repo + arXiv:2209.15352 paper) so a reader diagnosing
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
            prompt,
            duration_secs,
        ))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`AudioGen::generate`] until the T5-encoder + AR-LM + EnCodec-decode
/// composition lands.
///
/// Names **all three** primary source URLs (HF card + AudioCraft repo +
/// arXiv:2209.15352 paper) so a reader diagnosing the gap has exactly
/// three places to walk. Mirrors the musicgen / sortformer / MT3 /
/// beat_this / RMVPE / pyannote / snac / hifigan Wave 3-5 loud-partial-
/// message precedent — CLAUDE.md 教訓 (a).
///
/// **CRITICAL ERROR TYPE RULE**: uses [`VokraError::UnsupportedOp`]
/// (which takes `String`) rather than [`VokraError::NotImplemented`]
/// (which takes `&'static str`), because the format!() message requires
/// a runtime-formatted string — this mirrors the Wave 5 canary_qwen fix
/// (musicgen precedent).
fn generate_forward_loud_partial(
    cfg: &AudioGenConfig,
    prompt: &str,
    duration_secs: f32,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "audiogen generate: T5-base text encoder forward + autoregressive transformer \
         LM decode (with 4-codebook delay pattern + text-conditioned cross-attention) \
         + EnCodec RVQ decode composition pending. What is missing is (a) the frozen \
         T5-base text encoder forward (upstream `transformers.T5EncoderModel` — no \
         reusable primitive in `vokra_ops` today; the follow-up wave lands either a \
         T5-base implementation dedicated to AudioCraft models or a first-class \
         `t5_text_encode` op — SHARED with the MusicGen sibling so a single fix \
         unblocks both binders), (b) the autoregressive transformer LM decode with the \
         AudioCraft 4-codebook delay pattern (Kreuk et al. Algorithm 1 — the interleave \
         across the 4 RVQ codebook streams that MusicGen later refined) plus \
         text-conditioned cross-attention over the T5-encoded prompt tokens (the \
         single-codebook AR LMs already in the tree — cosyvoice2 / moshi / voxtral — \
         cover neither the delay pattern nor the text-encoder cross-attention), and \
         (c) the EnCodec RVQ decode step — this is available via \
         `vokra_ops::encodec_rvq_decode` (the ONE landed anchor of the composition; the \
         follow-up wave wires the AR LM output onto this primitive; identical bundled \
         32 kHz codec as MusicGen). Config: d_model={d_model}, num_layers={num_layers}, \
         n_heads={n_heads}, ffn_dim={ffn_dim}, vocab_size={vocab_size}, \
         num_codebooks={num_codebooks}, codec_frame_rate_hz={codec_frame_rate_hz}, \
         sample_rate_hz={sample_rate_hz}. Requested prompt_len={prompt_len} chars, \
         duration_secs={duration_secs}. Primary sources: {hf_card} + {audiocraft_repo} \
         + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
         より honest') — no silent fabricated PCM stream ever emitted (FR-EX-08).",
        d_model = cfg.d_model,
        num_layers = cfg.num_layers,
        n_heads = cfg.n_heads,
        ffn_dim = cfg.ffn_dim,
        vocab_size = cfg.vocab_size,
        num_codebooks = cfg.num_codebooks,
        codec_frame_rate_hz = cfg.codec_frame_rate_hz,
        sample_rate_hz = cfg.sample_rate_hz,
        prompt_len = prompt.len(),
        duration_secs = duration_secs,
        hf_card = PRIMARY_SOURCE_HF_CARD,
        audiocraft_repo = PRIMARY_SOURCE_AUDIOCRAFT_REPO,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the AudioGen runtime binder — config round-trip +
    //! negative-space round-trip on the loud-partial gates + reciprocal
    //! defense against sibling arch aliasing.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real inference this
    //! would be `generate(...)` returning real 32 kHz PCM, but the
    //! T5-base + AR-LM (delay pattern + cross-attention) + EnCodec-
    //! decode composition is deferred (see the module doc +
    //! [`AudioGen::generate`] rustdoc). Fabricating a real-inference
    //! output would violate CLAUDE.md 教訓 (a) ("loud-partial は
    //! fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config round-trip**: `from_gguf` reads every axis stamped by
    //!    the converter (via the fallback path today; the strict path
    //!    when a future converter sub-wave stamps the topology chunk
    //!    group).
    //! 2. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / missing name / unsupported
    //!    variant / empty tensor list / unsupported forward surface)
    //!    fires at its documented surface point, in the documented
    //!    error variant.
    //! 3. **Arch stability + sibling reciprocal defense**: the arch tag
    //!    string is pinned and MUST NOT collide with sibling music /
    //!    audio-generation arches (FR-EX-08 dispatch safety).

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds an AudioGen GGUF carrying the arch tag + name + one
    /// representative LM decoder tensor whose outer dim matches
    /// `d_model`. The topology chunk group is optionally stamped
    /// (`stamp_topology = true`) — when omitted the runtime binder
    /// falls back to the primary-source defaults per key.
    ///
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn audiogen_gguf(
        name: &str,
        cfg: AudioGenConfig,
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
        // upstream AudioCraft `LMModel` decoder Q projection (same
        // state_dict layout as MusicGen since they share the LMModel
        // spine).
        let d = cfg.d_model as u64;
        b.add_tensor(
            "lm.transformer.layers.0.self_attn.mha.q_proj.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Default config matches primary-source HF config.json axes
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_matches_primary_source_hf_config_json_axes() {
        // Pin every DEFAULT_* constant with rustdoc reference to HF
        // config.json + AudioCraft EncodecModel bundle. A rename or
        // axis-value drift would land here in the same commit or fail
        // this test.
        let cfg = AudioGenConfig::primary_source_default();
        assert_eq!(cfg.d_model, 1536, "d_model = 1536 per HF config.json");
        assert_eq!(cfg.num_layers, 48, "num_layers = 48 per HF config.json");
        assert_eq!(cfg.n_heads, 24, "n_heads = 24 per HF config.json");
        assert_eq!(cfg.ffn_dim, 6144, "ffn_dim = 6144 per HF config.json");
        assert_eq!(
            cfg.vocab_size, 2048,
            "vocab_size = 2048 per HF config.json + AudioCraft \
             EncodecModel.quantizer.bins"
        );
        assert_eq!(
            cfg.num_codebooks, 4,
            "num_codebooks = 4 per AudioCraft AudioGen(...).lm.n_q"
        );
        assert_eq!(
            cfg.codec_frame_rate_hz, 50,
            "codec_frame_rate_hz = 50 per AudioCraft EncodecModel.frame_rate"
        );
        assert_eq!(
            cfg.sample_rate_hz, 32_000,
            "sample_rate_hz = 32000 per AudioCraft EncodecModel.sample_rate"
        );

        // AudioCraft family design invariant: `head_dim = 64` (a
        // deliberate choice that keeps the attention kernel stable
        // across the shared LMModel spine).
        assert_eq!(
            cfg.d_model / cfg.n_heads,
            64,
            "head_dim = d_model / n_heads = 1536 / 24 = 64"
        );

        // AudioCraft "4× hidden" FFN convention invariant.
        assert_eq!(
            cfg.ffn_dim,
            4 * cfg.d_model,
            "ffn_dim = 4 × d_model per AudioCraft convention"
        );
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf round-trips a stamped topology chunk group
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_round_trips_stamped_chunk_group() {
        let cfg = AudioGenConfig::primary_source_default();
        let file = audiogen_gguf(
            NAME,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let ag = AudioGen::from_gguf(&file).expect("valid AudioGen GGUF must bind");
        // Config round-trip — every stamped axis reads back into the
        // same AudioGenConfig value (converter follow-up sub-wave
        // path).
        assert_eq!(*ag.config(), cfg);
        assert_eq!(ag.config().d_model, 1536);
        assert_eq!(ag.config().num_layers, 48);
        // NC weight license is the primary-source default per the HF
        // card (`license: cc-by-nc-4.0`) — the runtime must surface it
        // verbatim from the provenance chunk. The M2-13 compliance gate
        // refuses this artifact in commercial mode (T4 tier —
        // `--allow-noncommercial` opt-in required).
        assert_eq!(ag.weight_license(), LicenseClass::NonCommercial);
        assert!(ag.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf falls back to primary-source defaults when chunk group
    //    absent (converter's current stamp posture)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_falls_back_to_primary_source_defaults_when_chunk_group_absent() {
        // The AudioGen converter does NOT currently stamp the
        // `vokra.audiogen.*` chunk group (only arch / name / category /
        // upstream_hf / provenance). An already-published GGUF must
        // still load — the fallback path reads the primary-source
        // constants transcribed from HF config.json.
        // Mirror of MusicGenConfig::from_gguf + SortformerConfig::from_gguf
        // fallback patterns.
        let cfg = AudioGenConfig::primary_source_default();
        let file = audiogen_gguf(
            NAME,
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let ag = AudioGen::from_gguf(&file).expect("chunk-free GGUF must bind via fallback");
        // Every axis fell through to its primary-source default — the
        // loader returns the same values as primary_source_default().
        assert_eq!(ag.config().d_model, DEFAULT_D_MODEL);
        assert_eq!(ag.config().num_layers, DEFAULT_NUM_LAYERS);
        assert_eq!(ag.config().n_heads, DEFAULT_N_HEADS);
        assert_eq!(ag.config().ffn_dim, DEFAULT_FFN_DIM);
        assert_eq!(ag.config().vocab_size, DEFAULT_VOCAB_SIZE);
        assert_eq!(ag.config().num_codebooks, NUM_CODEBOOKS);
        assert_eq!(ag.config().codec_frame_rate_hz, CODEC_FRAME_RATE_HZ);
        assert_eq!(ag.config().sample_rate_hz, SAMPLE_RATE_HZ);
    }

    // -----------------------------------------------------------------------
    // 4. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `musicgen` GGUF handed to the AudioGen binder by mistake
        // must fail loud with a specific message rather than silently
        // mis-binding (FR-EX-08). MusicGen and AudioGen share the
        // AR-LM-over-EnCodec-RVQ topology today but a future modality-
        // specific head would silent-mis-bind under shared arch —
        // Wave 6 audit follow-up 2026-08-14 retagged to distinct
        // `audiogen` for exactly this dispatch safety reason.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "musicgen");
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioGen::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`musicgen`") && m.contains("`audiogen`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("4-codebook delay pattern"),
                    "message should disambiguate AudioGen's AR-LM + delay-pattern topology \
                     to help the reader, got `{m}`"
                );
                assert!(
                    m.contains("Wave 6") && m.contains("retag"),
                    "message must anchor the reader on the Wave 6 retag rationale so a \
                     pre-Wave-6 GGUF (arch = `musicgen`) has a diagnostic path, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Also verify a completely unrelated sibling arch tag fails
        // loud — magnet is non-autoregressive masked-LM, entirely
        // different code path.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "magnet_small_10secs");
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioGen::from_gguf(&file) else {
            panic!("expected ModelLoad on magnet arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`magnet_small_10secs`") && m.contains("`audiogen`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. from_gguf rejects missing name chunk (never silently mis-routes
    //    to a default variant)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_name_chunk() {
        // A GGUF with correct arch but no name chunk cannot be
        // discriminated from a future family variant (`audiogen-large`
        // / `audiogen-stereo`) — silently defaulting would misroute a
        // future artifact. The loader must fail loud.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        // NO name chunk.
        b.add_tensor(
            "lm.transformer.layers.0.self_attn.mha.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioGen::from_gguf(&file) else {
            panic!("expected ModelLoad on missing name");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.name`"),
                    "message must call out the missing name chunk, got `{m}`"
                );
                assert!(
                    m.contains("audiogen-large") || m.contains("audiogen-stereo"),
                    "message must name at least one future family variant the loader \
                     would need to discriminate against, got `{m}`"
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
        // the AudioGenWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_D_MODEL, DEFAULT_D_MODEL);
        b.add_u32(GGUF_KEY_NUM_LAYERS, DEFAULT_NUM_LAYERS);
        b.add_u32(GGUF_KEY_N_HEADS, DEFAULT_N_HEADS);
        b.add_u32(GGUF_KEY_FFN_DIM, DEFAULT_FFN_DIM);
        b.add_u32(GGUF_KEY_VOCAB_SIZE, DEFAULT_VOCAB_SIZE);
        b.add_u32(GGUF_KEY_NUM_CODEBOOKS, NUM_CODEBOOKS);
        b.add_u32(GGUF_KEY_CODEC_FRAME_RATE_HZ, CODEC_FRAME_RATE_HZ);
        b.add_u32(GGUF_KEY_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioGen::from_gguf(&file) else {
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
        let cfg = AudioGenConfig::primary_source_default();
        let file = audiogen_gguf(
            NAME,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let ag = AudioGen::from_gguf(&file).unwrap();
        // A legitimate SFX prompt + duration — the loud-partial gate
        // must fire on the composition surface, not on pre-generate
        // arg validation.
        let Err(err) = ag.generate("dog barking on a wooden porch", 5.0) else {
            panic!("generate must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("audiogen generate"),
                    "message must call out the audiogen generate surface, got `{m}`"
                );
                // The three deferred pieces MUST all be named so the
                // follow-up wave has an unambiguous work anchor.
                assert!(
                    m.contains("T5"),
                    "message must name the T5-base text encoder deferred piece, got `{m}`"
                );
                assert!(
                    m.contains("delay pattern"),
                    "message must name the AudioCraft delay pattern, got `{m}`"
                );
                assert!(
                    m.contains("encodec_rvq_decode"),
                    "message must name the ONE landed anchor (`vokra_ops::encodec_rvq_decode`), \
                     got `{m}`"
                );
                // All three primary source URLs must be cited — task
                // hint requires this.
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD),
                    "message must contain the HF card URL substring \
                     ({PRIMARY_SOURCE_HF_CARD}), got `{m}`"
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
                assert!(m.contains("d_model=1536"), "d_model axis missing: {m}");
                assert!(m.contains("num_layers=48"), "num_layers axis missing: {m}");
                assert!(m.contains("n_heads=24"), "n_heads axis missing: {m}");
                assert!(m.contains("ffn_dim=6144"), "ffn_dim axis missing: {m}");
                assert!(
                    m.contains("vocab_size=2048"),
                    "vocab_size axis missing: {m}"
                );
                assert!(
                    m.contains("num_codebooks=4"),
                    "num_codebooks axis missing: {m}"
                );
                assert!(
                    m.contains("codec_frame_rate_hz=50"),
                    "codec_frame_rate_hz axis missing: {m}"
                );
                assert!(
                    m.contains("sample_rate_hz=32000"),
                    "sample_rate_hz axis missing: {m}"
                );
                // The prompt length + duration must be echoed so a
                // caller can cross-check the request.
                //
                // "dog barking on a wooden porch" = 29 chars (byte-len).
                assert!(
                    m.contains("prompt_len=29"),
                    "prompt_len (29 = len(\"dog barking on a wooden porch\")) missing: {m}"
                );
                assert!(m.contains("duration_secs=5"), "duration_secs missing: {m}");
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08 no-silent-fabrication clause, got `{m}`"
                );
                // Sibling mention — the follow-up wave anchor
                // ("SHARED with the MusicGen sibling") is required so a
                // reader knows the fix unblocks both binders in one
                // pass.
                assert!(
                    m.contains("SHARED with the MusicGen"),
                    "message must anchor the reader on the shared MusicGen composition wave, \
                     got `{m}`"
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
        // The AudioGen converter's DEFAULT_LICENSE_SPDX is
        // `cc-by-nc-4.0` → LicenseClass::NonCommercial. The runtime
        // must surface this verbatim so the M2-13 compliance gate
        // refuses commercial-mode load (T4 tier — the same fail-
        // closed posture as X-Codec-2 precedent 2026-07-28 +
        // MusicGen 2026-08-01 + Sortformer 2026-08-04).
        let cfg = AudioGenConfig::primary_source_default();
        let file = audiogen_gguf(
            NAME,
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let ag = AudioGen::from_gguf(&file).expect("bind");
        assert_eq!(
            ag.weight_license(),
            LicenseClass::NonCommercial,
            "the AudioGen converter defaults to NonCommercial per the HF card's \
             `license: cc-by-nc-4.0` — the runtime binder must surface it so the \
             M2-13 compliance gate can refuse commercial-mode load (T4 tier)"
        );
        // Missing provenance stamp falls back to Unknown (also
        // fail-closed at the gate).
        let file_no_license = audiogen_gguf(NAME, cfg, /*stamp_topology=*/ false, None);
        let ag_no_license =
            AudioGen::from_gguf(&file_no_license).expect("bind without license stamp");
        assert_eq!(
            ag_no_license.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed)"
        );
    }

    // -----------------------------------------------------------------------
    // 9. Arch tag is stable and distinct from sibling music/audio-
    //    generation arches (reciprocal defense — musicgen sibling test
    //    mirror + Wave 6 retag verification)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_music_generation_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. The sibling music/audio-generation
        // arch tags MUST NOT collide with ours — they live in the same
        // taxonomy neighbourhood but have completely different forward
        // topologies (or, in the MusicGen case, share the topology
        // today but must remain dispatch-separated for future modality-
        // specific heads — FR-EX-08).
        assert_eq!(ARCH, "audiogen");
        assert_eq!(NAME, "audiogen-medium");
        // The critical Wave 6 retag assertion: MusicGen sibling arch
        // MUST be distinct. Pre-Wave-6 the AudioGen converter shared
        // the `musicgen` tag, which would silent-mis-bind a future
        // AudioGen modality-specific head against MusicGen's music-only
        // runtime path.
        assert_ne!(
            ARCH, "musicgen",
            "audiogen (text-to-SFX AR LM) and musicgen (text-to-music AR LM) share \
             the AR-LM-over-EnCodec + T5-encoder-cross-attention topology today but \
             MUST remain dispatch-separated for future modality-specific heads \
             (SFX-only conditioning stack, stereo output head, per-class embedding \
             table). Silent aliasing = mis-route (FR-EX-08). Wave 6 audit follow-up \
             2026-08-14 retagged AudioGen from the Wave 5 shared `musicgen` tag."
        );
        // Every other sibling music/audio-generation arch tag —
        // reciprocal defense so a future rename or accidental collision
        // lands here in the same commit or fails this test.
        assert_ne!(
            ARCH, "magnet_small_10secs",
            "audiogen (AR LM) and magnet_small_10secs (non-AR masked-LM) share the \
             music-generation taxonomy but have completely different forward topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "magnet_medium_30secs",
            "audiogen and magnet_medium_30secs — same rationale as small — mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "melodyflow_t24_30secs",
            "audiogen (AR LM) and melodyflow_t24_30secs (DiT flow-matching editing) \
             have completely different forward topologies — mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "jasco_400m_chords_drums",
            "audiogen (text conditioning) and jasco_400m_chords_drums (joint \
             audio-symbolic chord/drum conditioning) have different conditioning \
             stacks — mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "audioldm2",
            "audiogen (AR LM) and audioldm2 (latent diffusion) — different topologies \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "stable_audio_open_small",
            "audiogen (AR LM) and stable_audio_open_small (latent diffusion) — \
             different topologies (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "ace_step",
            "audiogen (token-by-token AR) and ace_step (chunked-AR) — different \
             decoder loops (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "bs_roformer",
            "audiogen (generation) and bs_roformer (source separation) — completely \
             different task (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 10. from_gguf rejects unsupported future variant name with
    //     "follow-up wave" anchor (proactive future-proofing for
    //     `audiogen-large` / `audiogen-stereo` etc.)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_unsupported_variant_name_with_followup_anchor() {
        // A hypothetical `audiogen-large` GGUF handed to this WP's
        // binder must fail loud with the "follow-up wave adds
        // AudioGenVariant enum" anchor so a reader diagnosing the
        // mismatch has a clear path forward when the enum lift lands.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "audiogen-large");
        b.add_tensor(
            "lm.transformer.layers.0.self_attn.mha.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioGen::from_gguf(&file) else {
            panic!("expected ModelLoad on unsupported variant name");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("audiogen-large"),
                    "message must name the unsupported variant, got `{m}`"
                );
                assert!(
                    m.contains(NAME),
                    "message must name the expected NAME so a reader can compare, got `{m}`"
                );
                assert!(
                    m.contains("AudioGenVariant"),
                    "message must name the future enum being extended, got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD),
                    "message must name the primary source URL, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. matches_config soft accessor honestly reflects shape presence
    // -----------------------------------------------------------------------

    #[test]
    fn matches_config_soft_accessor_finds_d_model_axis() {
        // The soft accessor should return true when at least one bound
        // tensor has an axis matching `d_model`. The fixture LM decoder
        // tensor's rows/cols are both `d_model` so this must pass.
        let cfg = AudioGenConfig::primary_source_default();
        let file = audiogen_gguf(
            NAME,
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let ag = AudioGen::from_gguf(&file).unwrap();
        assert!(
            ag.weights.matches_config(ag.config()),
            "at least one bound tensor must have an axis matching config.d_model"
        );
        // Sanity: a stale config (bogus d_model) does NOT match the
        // fixture — pins the accessor as a real check (not a stub that
        // always returns true).
        let stale = AudioGenConfig {
            d_model: 99_999,
            ..cfg
        };
        assert!(
            !ag.weights.matches_config(&stale),
            "matches_config must return false for a d_model with no matching axis"
        );
    }

    // -----------------------------------------------------------------------
    // 12. from_gguf rejects missing arch chunk (never silently binds a
    //     non-Vokra-native GGUF)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch_chunk() {
        // A GGUF that never had `vokra.model.arch` stamped (a foreign
        // GGUF handed to us by mistake, or a converter regression that
        // dropped the arch stamp) must fail loud rather than silently
        // proceeding to name / topology / weights checks — the arch is
        // the first-line discriminator for Vokra-native GGUFs.
        let mut b = GgufBuilder::new();
        // NO arch chunk.
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_tensor(
            "lm.transformer.layers.0.self_attn.mha.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = AudioGen::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch chunk, got `{m}`"
                );
                assert!(
                    m.contains("Vokra-native audiogen GGUF"),
                    "message must anchor the reader on the Vokra-native GGUF contract, \
                     got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
