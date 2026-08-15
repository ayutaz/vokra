//! **Meta HT-Demucs** (`facebook/demucs`, MIT) — Hybrid Transformer
//! Demucs music source separation runtime binder for the `demucs`
//! converter arch (2026-08-14 audit follow-up Wave 5 loud-partial).
//!
//! # Primary source
//!
//! - HF model card: <https://huggingface.co/facebook/demucs>
//! - GitHub reference:
//!   <https://github.com/facebookresearch/demucs>
//!   (`demucs/htdemucs.py` — the hybrid model definition; MIT LICENSE).
//! - Paper: Rouard, Massa, Défossez, *"Hybrid Transformers for Music
//!   Source Separation"*, ICASSP 2023 (arXiv:2211.08553).
//! - Weight license: **MIT** (Permissive; `docs/license-audit.md` §3.1
//!   row 422 ☑ Commercial 2026-08-01 yousan — first music-source-
//!   separation permissive land after BS-Roformer Rejected).
//!
//! # Architecture (transcribed from primary sources)
//!
//! HT-Demucs is a *hybrid* U-Net operating on **two parallel branches**
//! that meet at a shared bottleneck:
//!
//! ```text
//! mixed_music_pcm (stereo f32, 44.1 kHz)
//!   -> (branch A) waveform-domain U-Net encoder      ← **loud-partial**
//!        (`depth=4` down-sample stages of Conv1D + GELU + GLU;
//!         base channel width `channels=48`; kernel_size=8, stride=4;
//!         local attention stubs at each stage.)
//!   -> (branch B) STFT + spectrogram-domain U-Net encoder ← **loud-partial**
//!        (`nfft=4096`, `hop_length=1024`, Hann window; matching
//!         `depth=4` down-sample stages of Conv2D + GELU + GLU with
//!         cross-domain skip.)
//!   -> shared bottleneck                                  ← **loud-partial**
//!        (BiLSTM stack, `lstm_layers=2`, hidden=384 — waveform side;
//!         Transformer stack, `t_layers=5` cross-domain self-attention
//!         + `cross_attn=true` connecting waveform ↔ spectrogram
//!         tokens; SwiGLU FFN, RoPE-free classical MHA.)
//!   -> (branch A^-1) waveform-domain U-Net decoder    ← **loud-partial**
//!        (mirror of encoder — 4 up-sample transposed-Conv1D stages,
//!         cross-domain gated skip-connections consume matching
//!         spectrogram-domain skip tensors.)
//!   -> (branch B^-1) spectrogram-domain U-Net decoder + iSTFT ← **loud-partial**
//!        (mirror of encoder — 4 up-sample transposed-Conv2D stages
//!         → iSTFT with matching Hann window → time-domain sum onto
//!         the waveform-branch output — the "hybrid" step.)
//!   -> per-source PCM stems
//!   -> `DemucsStems { drums, bass, vocals, other }`
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`Demucs::from_gguf`] with strict
//!   `vokra.model.arch == "demucs"` validation +
//!   [`DemucsConfig::from_gguf`] with primary-source constant fallback
//!   (the Demucs converter does NOT currently stamp the
//!   `vokra.demucs.*` chunk group — only arch / name / category /
//!   upstream_hf / provenance — so a *strict* reader would refuse the
//!   already-published `huggingface.co/vokra/demucs-htdemucs` GGUF.
//!   Primary source is well-established (HF card + upstream repo +
//!   paper) so fallback does not fabricate axes; a future converter
//!   sub-wave that starts stamping the chunk group upgrades this to
//!   real-stamped reads seamlessly — mirror of `SortformerConfig::from_gguf`
//!   pattern), [`DemucsWeights::from_gguf`] with a floor of non-empty
//!   tensor count enforced loud (a GGUF that carries zero tensors is
//!   refused rather than silently running an all-zero forward — FR-EX-08),
//!   [`DemucsStems`] public output surface pin (fields match the
//!   MUSDB18 4-stem taxonomy `drums`/`bass`/`vocals`/`other` — pinned
//!   in the surface pin test), and weight-license class surfacing
//!   (defaults to [`LicenseClass::Permissive`] per the converter's
//!   stamped `mit`).
//! - **Loud-partial (this WP)**: [`Demucs::separate`] returns
//!   [`VokraError::UnsupportedOp`] naming **all five** deferred pieces:
//!   1. the waveform-branch U-Net encoder + decoder + tensor-name walk
//!      from the upstream `facebook/demucs` state_dict prefixes;
//!   2. the spectrogram-branch U-Net encoder + decoder + STFT/iSTFT
//!      wiring (STFT/iSTFT primitives exist in `vokra_ops::stft` /
//!      `vokra_ops::istft` — what is pending is the *composition*);
//!   3. the BiLSTM bottleneck primitive (a new `vokra_ops::lstm` op —
//!      the one public LSTM today is
//!      `vokra_ops::hybrid_ctc_attention::LstmLmCell`, which is
//!      LM-shaped (token id in, one log-probability out) and so is
//!      the wrong function for a bottleneck; Silero's is a
//!      `pub(crate)` `lstm_forward` fixed at `HIDDEN = 128` in the
//!      separate `vokra-vad-micro` crate. Extraction of a generic
//!      sequence LSTM as an op is a follow-up wave)
//!      **plus** the Transformer bottleneck composition (composable
//!      from existing softmax + GEMM + LayerNorm — no new op needed);
//!   4. cross-domain self-attention between the waveform and
//!      spectrogram trunks;
//!   5. the per-stem sum path (waveform-decoder output + iSTFT-decoded
//!      spectrogram-decoder output).
//!
//! The error names the primary source URLs (HF card + upstream repo +
//! paper) so a reader diagnosing this gap has exactly three places to
//! walk — mirror of the `SortformerDiar::diarize` / `Mt3::transcribe`
//! Wave 3-4 loud-partial-message precedent.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / redimnet / sortformer loud-partial precedent,
//! CLAUDE.md 教訓 (a)): the surrounding scaffold + `from_gguf`
//! chunk-group validation + `DemucsStems` surface + FR-EX-08 loud-fails
//! land today so a follow-up wave can flip the switch by (i) landing
//! the tensor-name walk against a real HT-Demucs state_dict via
//! `tools/parity/*_prepare_checkpoint.py` (uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) + (ii) wiring the two U-Net branches +
//! STFT/iSTFT composition + (iii) extracting the shared BiLSTM
//! primitive (from the `vokra-vad-micro` Silero cell, which is
//! `pub(crate)` and fixed-width) into `vokra_ops::lstm` + (iv) porting
//! the cross-domain self-attention + stem-sum output stage. STFT /
//! iSTFT / softmax / GEMM / LayerNorm primitives already exist —
//! the follow-up wave is composition + tensor walk + one new op
//! (BiLSTM extraction), NOT a greenfield kernel farm.
//!
//! # HT-Demucs multi-variant sibling
//!
//! The **base `demucs` arch tag** targeted here is the 4-stem HT-Demucs
//! variant (MUSDB18 `drums`/`bass`/`vocals`/`other`). The sibling
//! converter `htdemucs_multi.rs` (arch tag `htdemucs_multi`, category
//! `source-separation`, single `ModelKind::HtdemucsMulti` covers both
//! 4-source `htdemucs_ft` and 6-source `htdemucs_6s` via tensor-shape-
//! derived source count) is a **distinct arch tag** and requires a
//! **distinct runtime binder**, per sortformer's arch-tag discipline
//! ("silently sharing arch would misroute — FR-EX-08"). The
//! multi-source binder is a future WP; this module handles only the
//! base 4-stem `demucs` arch tag.
//!
//! # `vokra.demucs.*` chunk group (read here — fallback-friendly)
//!
//! The Demucs converter
//! (`crates/vokra-convert/src/models/demucs_htdemucs.rs`) currently
//! stamps only the arch / name / category / upstream_hf / provenance
//! chunks. The topology chunk group is READ by this binder but any
//! absent key falls back to the primary-source constant so an
//! already-published GGUF loads correctly. A future converter sub-wave
//! that adds `vokra.demucs.*` stamps will override the fallback
//! automatically per-key. Mirror of `SortformerConfig::from_gguf` /
//! `PyanNetConfig::from_gguf` pattern.
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"demucs"`).
//!   Deliberately distinct from every sibling separator arch
//!   (`sepformer` — dual-path time-domain speech separation /
//!   `tiger_separator` — time-frequency dialog/effects/music /
//!   `mp_senet` — magnitude-phase parallel speech enhancement /
//!   `bs_roformer` — STFT-domain band-split /
//!   `htdemucs_multi` — 4/6-source HT-Demucs multi-variant /
//!   `mossformer2_ss_16k` — FSMN + gated attention 16 kHz cocktail-
//!   party separation) — silently sharing would misroute the runtime
//!   dispatch (FR-EX-08).
//! - `vokra.model.name` (`String`): `"demucs-htdemucs"` — the
//!   versioned identifier that matches the `huggingface.co/vokra/`
//!   publish slug.
//! - `vokra.demucs.{audio_channels, samplerate, sources, channels,
//!   depth, nfft, lstm_layers, transformer_layers}` (`u32` each):
//!   the composite topology axes. Fallback constants transcribed from
//!   the HF card + upstream `demucs/htdemucs.py` + paper (see the
//!   `DEFAULT_*` constants for the primary-source anchors).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03 / M2-13) can classify the
//!   artifact without re-inspecting the safetensors provenance. The
//!   Demucs converter stamps `Permissive` by default per the upstream
//!   `LICENSE` file's MIT — a caller override at
//!   `vokra-cli convert --license <spdx>` re-derives the class.
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`NAME`] / topology keys) — same rule the sibling
//! loud-partial binders (`sortformer_diar_4spk_v1` / `mt3` /
//! `beat_this` / `redimnet` / `pyannote` / `snac` / `hifigan` /
//! `bigvgan` / `vocos`) use so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! HT-Demucs is distributed as `.th` (Torch pickle) on Meta's mirror
//! and as safetensors on HF; this runtime **never** touches ONNX
//! (FR-LD-05 / NFR-DS-02). The `.th` → safetensors bridge for
//! HT-Demucs is a future uv-managed Python 3.12 sidecar per
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]` — not
//! part of this WP.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/demucs_htdemucs.rs`. See module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model demucs-htdemucs`.
///
/// Deliberately distinct from every sibling separator arch tag
/// (`sepformer` / `tiger_separator` / `mp_senet` / `bs_roformer` /
/// `htdemucs_multi` / `mossformer2_ss_16k`). Silently sharing would
/// let runtime dispatch bind a loader for the wrong topology (an
/// `sepformer` loader would look for `masker.*` / `separator.*`
/// tensors that HT-Demucs never emits; HT-Demucs's hybrid two-branch
/// U-Net + cross-domain attention has no sibling analog) — FR-EX-08
/// forbids the silent-wrong shape mismatch. Version-neutral (a future
/// HT-Demucs-v5 or hybrid-drums-only variant keeps the tag; only
/// [`NAME`] is versioned).
pub const ARCH: &str = "demucs";

/// Expected `vokra.model.name` value — matches the
/// `huggingface.co/vokra/demucs-htdemucs` publish slug.
pub const NAME: &str = "demucs-htdemucs";

/// `vokra.demucs.audio_channels` — input audio channel count
/// (stereo = 2 for the standard 4-stem release, per HF card).
pub const GGUF_KEY_AUDIO_CHANNELS: &str = "vokra.demucs.audio_channels";
/// `vokra.demucs.samplerate` — input sample rate (Hz).
/// Primary-source default: 44_100 (Meta upstream release + HF card).
pub const GGUF_KEY_SAMPLERATE: &str = "vokra.demucs.samplerate";
/// `vokra.demucs.sources` — output stem count. Primary-source default:
/// 4 (MUSDB18 `drums`/`bass`/`vocals`/`other`). The sibling
/// `htdemucs_multi` binder handles 4-source `htdemucs_ft` + 6-source
/// `htdemucs_6s` variants with a distinct arch tag.
pub const GGUF_KEY_SOURCES: &str = "vokra.demucs.sources";
/// `vokra.demucs.channels` — U-Net base channel width. Primary-source
/// default: 48 (Rouard et al. §3 + upstream `demucs/htdemucs.py`
/// `channels=48`).
pub const GGUF_KEY_CHANNELS: &str = "vokra.demucs.channels";
/// `vokra.demucs.depth` — U-Net encoder / decoder depth (matched for
/// both branches). Primary-source default: 4 (Rouard et al. §3 +
/// upstream `demucs/htdemucs.py` `depth=4`).
pub const GGUF_KEY_DEPTH: &str = "vokra.demucs.depth";
/// `vokra.demucs.nfft` — STFT `n_fft` for the spectrogram branch.
/// Primary-source default: 4096 (upstream `demucs/htdemucs.py`
/// `nfft=4096`).
pub const GGUF_KEY_NFFT: &str = "vokra.demucs.nfft";
/// `vokra.demucs.lstm_layers` — BiLSTM bottleneck depth on the
/// waveform-branch side. Primary-source default: 2 (upstream
/// `demucs/htdemucs.py` `lstm_layers=2`).
pub const GGUF_KEY_LSTM_LAYERS: &str = "vokra.demucs.lstm_layers";
/// `vokra.demucs.transformer_layers` — cross-domain Transformer
/// bottleneck depth. Primary-source default: 5 (upstream
/// `demucs/htdemucs.py` `t_layers=5`).
pub const GGUF_KEY_TRANSFORMER_LAYERS: &str = "vokra.demucs.transformer_layers";

// Primary-source constants transcribed from the HF model card
// (huggingface.co/facebook/demucs), the upstream repository
// (github.com/facebookresearch/demucs `demucs/htdemucs.py`), and the
// paper (arXiv:2211.08553 §3 "Model Architecture", fetched 2026-08-14
// — CLAUDE.md「ハルシネーション厳禁」).

/// Input audio channel count (stereo = 2). Primary source: HF card +
/// upstream release manifest.
pub const DEFAULT_AUDIO_CHANNELS: u32 = 2;
/// Input sample rate (Hz). Primary source: HF card + upstream release
/// manifest.
pub const DEFAULT_SAMPLERATE: u32 = 44_100;
/// Output stem count (MUSDB18 4-stem `drums`/`bass`/`vocals`/`other`).
/// Primary source: HF card + paper §3.
pub const DEFAULT_SOURCES: u32 = 4;
/// U-Net base channel width. Primary source: upstream
/// `demucs/htdemucs.py` `channels=48` + Rouard et al. §3.
pub const DEFAULT_CHANNELS: u32 = 48;
/// U-Net encoder / decoder depth. Primary source: upstream
/// `demucs/htdemucs.py` `depth=4` + Rouard et al. §3.
pub const DEFAULT_DEPTH: u32 = 4;
/// STFT `n_fft` for the spectrogram branch. Primary source: upstream
/// `demucs/htdemucs.py` `nfft=4096`.
pub const DEFAULT_NFFT: u32 = 4096;
/// BiLSTM bottleneck depth. Primary source: upstream
/// `demucs/htdemucs.py` `lstm_layers=2`.
pub const DEFAULT_LSTM_LAYERS: u32 = 2;
/// Cross-domain Transformer bottleneck depth. Primary source:
/// upstream `demucs/htdemucs.py` `t_layers=5` + Rouard et al. §3.
pub const DEFAULT_TRANSFORMER_LAYERS: u32 = 5;

/// Primary-source anchor for the HF model card. Cited in the
/// loud-partial error so a reader diagnosing the gap knows the
/// definitive artifact source.
const PRIMARY_SOURCE_HF_CARD: &str = "huggingface.co/facebook/demucs";
/// Primary-source anchor for the upstream Meta repository. Cited in
/// the loud-partial error so a reader diagnosing the gap knows the
/// tensor-name walk anchor (`demucs/htdemucs.py`).
const PRIMARY_SOURCE_GITHUB_REPO: &str = "github.com/facebookresearch/demucs";
/// Paper anchor (Rouard, Massa, Défossez, ICASSP 2023) — cited
/// alongside the two artefact URLs so a reader has the theoretical
/// context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2211.08553";

// ---------------------------------------------------------------------------
// DemucsConfig — the composite topology axes read from the
// `vokra.demucs.*` chunk group, with primary-source constant fallback
// (the Demucs converter does not currently stamp this chunk group —
// the fallback is honest because the primary source is well-established;
// a future converter sub-wave that adds the stamps upgrades this
// reader to real-stamped reads seamlessly). Mirror of
// [`crate::sortformer_diar_4spk_v1::SortformerConfig::from_gguf`].
// ---------------------------------------------------------------------------

/// HT-Demucs hyperparameters as they ride the `vokra.demucs.*` chunk
/// group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key — a GGUF that never carried the chunk
/// still loads with the upstream defaults transcribed from the HF card
/// + upstream `htdemucs.py` + paper. Every numeric axis is `u32` in
///   the GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemucsConfig {
    /// Input audio channel count (stereo = 2 for the standard 4-stem
    /// release).
    pub audio_channels: u32,
    /// Input sample rate (Hz, 44_100 for the standard release).
    pub samplerate: u32,
    /// Output stem count (4 for MUSDB18 base HT-Demucs).
    pub sources: u32,
    /// U-Net base channel width (48 per Rouard et al. §3).
    pub channels: u32,
    /// U-Net encoder / decoder depth (matched for both branches,
    /// 4 per Rouard et al. §3).
    pub depth: u32,
    /// STFT `n_fft` for the spectrogram branch (4096 per upstream
    /// `demucs/htdemucs.py`).
    pub nfft: u32,
    /// BiLSTM bottleneck depth on the waveform-branch side (2 per
    /// upstream `demucs/htdemucs.py`).
    pub lstm_layers: u32,
    /// Cross-domain Transformer bottleneck depth (5 per Rouard et al.
    /// §3 + upstream `demucs/htdemucs.py`).
    pub transformer_layers: u32,
}

impl Default for DemucsConfig {
    /// The primary-source-transcribed HT-Demucs 4-stem axes. Used by
    /// [`Self::from_gguf`] as the per-key fallback.
    fn default() -> Self {
        Self {
            audio_channels: DEFAULT_AUDIO_CHANNELS,
            samplerate: DEFAULT_SAMPLERATE,
            sources: DEFAULT_SOURCES,
            channels: DEFAULT_CHANNELS,
            depth: DEFAULT_DEPTH,
            nfft: DEFAULT_NFFT,
            lstm_layers: DEFAULT_LSTM_LAYERS,
            transformer_layers: DEFAULT_TRANSFORMER_LAYERS,
        }
    }
}

impl DemucsConfig {
    /// The primary-source-transcribed HT-Demucs 4-stem axes as a
    /// `const` — an alias for the [`Default`] impl useful in contexts
    /// that need a `const` (e.g. `const` initializers, doc examples).
    /// Never used silently by the loader; every axis passes through
    /// [`Self::from_gguf`]'s per-key fallback path.
    #[must_use]
    pub const fn v4stem_default() -> Self {
        Self {
            audio_channels: DEFAULT_AUDIO_CHANNELS,
            samplerate: DEFAULT_SAMPLERATE,
            sources: DEFAULT_SOURCES,
            channels: DEFAULT_CHANNELS,
            depth: DEFAULT_DEPTH,
            nfft: DEFAULT_NFFT,
            lstm_layers: DEFAULT_LSTM_LAYERS,
            transformer_layers: DEFAULT_TRANSFORMER_LAYERS,
        }
    }

    /// Reads every `vokra.demucs.*` chunk from `gguf`, falling back to
    /// the primary-source [`Default`] constants per absent key.
    ///
    /// The Demucs converter does not currently stamp this chunk group
    /// (only arch / name / category / upstream_hf / provenance), so
    /// on an already-published GGUF every axis falls through to its
    /// primary-source default. A future converter sub-wave that adds
    /// the stamps upgrades this reader to real-stamped reads per-key
    /// with no runtime code change.
    ///
    /// Mirror of [`crate::sortformer_diar_4spk_v1::SortformerConfig::from_gguf`]
    /// and [`crate::pyannote::PyanNetConfig::from_gguf`] — the same
    /// fallback pattern used for the sibling loud-partial binders
    /// whose converter chunk groups are likewise post-launch. Distinct
    /// from [`crate::mt3::Mt3Config::from_gguf`] which is strict (fails
    /// loud on any missing chunk) because MT3's upstream release ships
    /// no first-class config anywhere and fallback would fabricate.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::default();
        Self {
            audio_channels: gguf
                .get(GGUF_KEY_AUDIO_CHANNELS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.audio_channels),
            samplerate: gguf
                .get(GGUF_KEY_SAMPLERATE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.samplerate),
            sources: gguf
                .get(GGUF_KEY_SOURCES)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.sources),
            channels: gguf
                .get(GGUF_KEY_CHANNELS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.channels),
            depth: gguf
                .get(GGUF_KEY_DEPTH)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.depth),
            nfft: gguf
                .get(GGUF_KEY_NFFT)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.nfft),
            lstm_layers: gguf
                .get(GGUF_KEY_LSTM_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.lstm_layers),
            transformer_layers: gguf
                .get(GGUF_KEY_TRANSFORMER_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.transformer_layers),
        }
    }
}

// ---------------------------------------------------------------------------
// DemucsWeights — bound the tensor manifest with a non-emptiness gate.
// Under the loud-partial WP the weights are counted but the two
// U-Net branches + BiLSTM + Transformer bottleneck + cross-domain
// attention + stem-sum output forward is deferred. Mirror of
// `SortformerWeights` / `Mt3Weights` / `BeatThisWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an HT-Demucs GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid HT-Demucs checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future `DemucsWeights::bind_waveform_branch_weights`
/// / `bind_spectrogram_branch_weights` / `bind_bottleneck_weights`
/// tensor walks can find their inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct DemucsWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up branch-wiring
    /// wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl DemucsWeights {
    /// Scans `gguf` for the HT-Demucs state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid HT-Demucs checkpoint).
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
                "demucs: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model demucs-htdemucs` \
                 against an upstream safetensors checkpoint (either the direct \
                 `facebook/demucs` safetensors on HF or the upstream `.th` bundle \
                 flattened via a `tools/parity/*_prepare_checkpoint.py` sidecar)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the branch-forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load-time shape gate — validates that at least one bound
    /// tensor has an axis matching the U-Net encoder-branch base
    /// channel width (`config.channels`). Under the current landing
    /// this is a **soft** gate (mismatch is silently ignored) because
    /// the two U-Net branch + bottleneck + head tensor-name walk has
    /// not yet been pinned pending the upstream tensor-name manifest
    /// fetch — a hard shape assertion today would fail against every
    /// legitimate future manifest.
    ///
    /// The follow-up wave will replace this soft accessor with a
    /// hard pin against the primary-source-verified tensor-name walk
    /// (mirror of `pyannote::PyanNetWeights::verify_core_shapes`).
    ///
    /// Kept as a `#[must_use]` accessor so the read is deliberate.
    #[must_use]
    pub fn matches_config(&self, config: &DemucsConfig) -> bool {
        let base = config.channels as usize;
        self.tensors.iter().any(|(_, dims)| dims.contains(&base))
    }
}

// ---------------------------------------------------------------------------
// DemucsStems — the public output surface for `Demucs::separate` once
// the tensor-name walk + two-branch U-Net composition + STFT/iSTFT
// wiring + cross-domain attention + stem-sum path land. Defined here
// per the task hint ("Ship `Demucs::separate(pcm) -> Result<DemucsStems
// { drums, bass, vocals, other }>`") — pinned as the surface a future
// forward wave binds against.
//
// Field spelling `drums`/`bass`/`vocals`/`other` matches the MUSDB18
// 4-stem taxonomy (per HF card + Rouard et al. §4 evaluation set).
// A future 6-source HT-Demucs (guitar / piano extras — see sibling
// `htdemucs_multi` binder for `htdemucs_6s`) needs a distinct output
// type; silently unifying would break the 4-stem contract here.
// ---------------------------------------------------------------------------

/// A 4-stem separation result emitted by HT-Demucs's hybrid U-Net.
///
/// Fields match the task-hint spelling AND the upstream MUSDB18 4-stem
/// taxonomy — pinned as a **surface pin**: a rename or field-shape
/// change would need to land here in the same commit or fail the
/// surface pin test at the bottom of this module.
///
/// Each stem is a mono-or-stereo interleaved `Vec<f32>` matching the
/// input `audio_channels` × input length. The caller is responsible
/// for de-interleaving into per-channel buffers as needed.
#[derive(Debug, Clone, PartialEq)]
pub struct DemucsStems {
    /// Drum-kit stem (percussion — kick, snare, hats, cymbals, toms).
    pub drums: Vec<f32>,
    /// Bass-line stem (electric bass, upright bass, synth bass).
    pub bass: Vec<f32>,
    /// Vocal stem (lead + backing vocals).
    pub vocals: Vec<f32>,
    /// Residual / all-other stem (guitars, keys, synths, strings,
    /// effects, room ambience — anything not in `drums`/`bass`/`vocals`).
    pub other: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Demucs — the runtime binder handle
// ---------------------------------------------------------------------------

/// Meta HT-Demucs hybrid music source-separation runtime binder
/// (`facebook/demucs`, MIT — Permissive T1 tier).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`separate`](Self::separate) on a mixed-music PCM buffer to obtain
/// a `DemucsStems { drums, bass, vocals, other }`. See the module doc
/// for the current implementation-status matrix and the FR-EX-08
/// loud-error contract on the two U-Net branches + BiLSTM + Transformer
/// bottleneck + cross-domain attention + stem-sum path composition.
#[derive(Debug)]
pub struct Demucs {
    config: DemucsConfig,
    // The bound weights are held (real, counted) but the two-branch
    // composition + bottleneck + head + region-merging is a follow-up
    // wave; the field is deliberately `#[allow(dead_code)]` until the
    // composition lands so a reader is not misled by an unused field.
    // Same posture as sortformer_diar_4spk_v1 / RMVPE / pyannote /
    // mt3 / beat_this / redimnet.
    #[allow(dead_code)]
    weights: DemucsWeights,
    weight_license: LicenseClass,
}

impl Demucs {
    /// Binds an HT-Demucs GGUF: validates arch, reads the topology
    /// chunk group (with primary-source constant fallback per key),
    /// discovers tensors, and surfaces the stamped weight-license
    /// class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"demucs"` (a `sepformer` / `tiger_separator` /
    ///   `mp_senet` / `bs_roformer` / `htdemucs_multi` /
    ///   `mossformer2_ss_16k` GGUF handed to us by mistake fails with
    ///   a clear message instead of a downstream missing-tensor —
    ///   every sibling separator arch has a different hybrid /
    ///   time-frequency / STFT-domain topology, so the runtime dispatch
    ///   discipline forbids silent aliasing).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`DemucsWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "demucs: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model demucs-htdemucs`? Note that \
                     the sibling separator arches — `sepformer` (dual-path time-domain \
                     speech separation), `tiger_separator` (time-frequency dialog / \
                     effects / music), `mp_senet` (magnitude-phase parallel speech \
                     enhancement), `bs_roformer` (STFT-domain band-split), \
                     `htdemucs_multi` (4/6-source HT-Demucs multi-variant, distinct \
                     from this 4-stem base binder), `mossformer2_ss_16k` (FSMN + gated \
                     attention 16 kHz cocktail-party separation) — each has a \
                     fundamentally different topology; HT-Demucs's hybrid waveform + \
                     spectrogram U-Net + cross-domain Transformer bottleneck has no \
                     sibling analog and silently aliasing arch would misroute the \
                     runtime dispatch, FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "demucs: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native demucs GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Topology axes from the `vokra.demucs.*` chunk group
        //    (fallback-friendly — see the module doc for the Demucs
        //    converter's stamp posture).
        let config = DemucsConfig::from_gguf(file);

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = DemucsWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The Demucs
        //    converter defaults to `Permissive` per the upstream
        //    `LICENSE` file's MIT; a caller override at
        //    `--license <spdx>` re-derives the class. Missing
        //    provenance falls back to `Unknown` which is fail-closed
        //    at the M2-13 compliance gate — same posture as MT3 /
        //    sortformer.
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

    /// The bound topology axes (from `vokra.demucs.*` chunk group with
    /// primary-source constant fallback).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &DemucsConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The Demucs converter
    /// stamps `Permissive` by default per the upstream `LICENSE`
    /// file's MIT. A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] which is fail-closed at the M2-13
    /// compliance gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the branch-forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Separates a mixed-music PCM buffer (stereo f32 44.1 kHz per the
    /// HT-Demucs release spec) into a `DemucsStems { drums, bass,
    /// vocals, other }`.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — HT-Demucs's inference
    /// path requires **five** deferred pieces:
    ///
    /// 1. **Waveform-branch U-Net composition + tensor walk**: the
    ///    encoder + decoder stack of Conv1D + GELU + GLU stages,
    ///    plus the tensor-name walk from the upstream
    ///    `facebook/demucs` state_dict prefixes to the encoder /
    ///    decoder inputs (pending the upstream tensor-name manifest
    ///    fetch — same posture as sortformer / Charsiu real-weight
    ///    bind).
    /// 2. **Spectrogram-branch U-Net + STFT/iSTFT composition**: STFT
    ///    (n_fft=4096) → Conv2D + GELU + GLU encoder / decoder →
    ///    iSTFT wiring; the STFT / iSTFT primitives already exist in
    ///    `vokra_ops::stft` / `vokra_ops::istft` — what is pending is
    ///    the composition.
    /// 3. **BiLSTM bottleneck + Transformer bottleneck composition**:
    ///    the BiLSTM primitive is *not yet extracted* into
    ///    `vokra_ops::lstm`. The one public LSTM,
    ///    `vokra_ops::hybrid_ctc_attention::LstmLmCell`, is LM-shaped
    ///    (token id in, one log-probability out, embedding + vocab
    ///    projection bundled in) and so is the wrong function for a
    ///    bottleneck; Silero's is a `pub(crate)` `lstm_forward` fixed
    ///    at `HIDDEN = 128` in the separate `vokra-vad-micro` crate.
    ///    Extraction is a follow-up wave. The Transformer bottleneck (SwiGLU + MHA
    ///    + LayerNorm) is composable from Vokra's existing softmax +
    ///      GEMM + LayerNorm primitives (no new op needed).
    /// 4. **Cross-domain self-attention** between the waveform-branch
    ///    and spectrogram-branch trunks (the "hybrid" step — attention
    ///    heads read from both branches' tokens simultaneously).
    /// 5. **Per-stem sum path**: waveform-decoder output + iSTFT-decoded
    ///    spectrogram-decoder output → 4 stems (`drums` / `bass` /
    ///    `vocals` / `other`).
    ///
    /// The error names **three** primary source URLs (HF card +
    /// upstream repo + paper) so a reader diagnosing this gap has
    /// exactly three places to walk. **No fabricated stem stream is
    /// ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred two-branch U-Net + bottleneck + cross-domain
    ///   attention + stem-sum composition.
    pub fn separate(&self, mixed_music_pcm: &[f32]) -> Result<DemucsStems> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = mixed_music_pcm;
        Err(separate_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`Demucs::separate`] until the tensor-name walk + two-branch
/// U-Net composition + STFT/iSTFT wiring + bottleneck + cross-domain
/// attention + stem-sum output stage land.
///
/// Names **all three** primary source URLs (HF card + upstream repo +
/// paper) so a reader diagnosing the gap has exactly three places to
/// walk. Mirrors the sortformer / MT3 / beat_this / RMVPE / pyannote /
/// snac / hifigan Wave 3-4 loud-partial-message precedent — CLAUDE.md
/// 教訓 (a).
fn separate_forward_loud_partial(cfg: &DemucsConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "demucs separate: hybrid two-branch U-Net + BiLSTM + Transformer bottleneck + \
         cross-domain self-attention + stem-sum composition pending. What is missing: \
         (a) the waveform-branch U-Net encoder + decoder (depth={depth} Conv1D + GELU + \
         GLU stages, base channels={channels}) + tensor-name walk from the upstream \
         `facebook/demucs` state_dict prefixes (pending the upstream tensor-name manifest \
         fetch — same posture as sortformer / Charsiu real-weight bind), (b) the \
         spectrogram-branch U-Net encoder + decoder (STFT nfft={nfft} → Conv2D + GELU + \
         GLU depth={depth} → iSTFT — the STFT/iSTFT primitives `vokra_ops::stft` / \
         `vokra_ops::istft` already exist, what is pending is the composition), \
         (c) the BiLSTM bottleneck (lstm_layers={lstm_layers} — a new `vokra_ops::lstm` \
         op is needed. The one public LSTM in vokra-ops is \
         `vokra_ops::hybrid_ctc_attention::LstmLmCell`, which is LM-shaped (token id \
         in, one log-probability out, embedding + vocab projection bundled in) and is \
         therefore the wrong function for a bottleneck rather than a missing one; \
         Silero's LSTM is a `pub(crate)` `lstm_forward` fixed at `HIDDEN = 128` in \
         the separate `vokra-vad-micro` crate, not in `silero_vad`, which is only a \
         std veneer over it. Extraction is a follow-up wave) plus the Transformer \
         bottleneck (transformer_layers={transformer_layers} — composable from existing \
         softmax + GEMM + LayerNorm, no new op needed), (d) the cross-domain \
         self-attention connecting waveform and spectrogram trunks (the 'hybrid' step \
         per Rouard et al. §3), and (e) the per-stem sum path emitting {sources} stems \
         `drums`/`bass`/`vocals`/`other`. Config: audio_channels={audio_channels}, \
         samplerate={samplerate}, sources={sources}, channels={channels}, depth={depth}, \
         nfft={nfft}, lstm_layers={lstm_layers}, transformer_layers={transformer_layers}. \
         Primary sources: {hf_card} + {github_repo} + {paper}. Loud pending (CLAUDE.md \
         教訓 (a) — 'loud-partial は fake-complete より honest') — no silent fabricated \
         DemucsStems stream ever emitted (FR-EX-08).",
        audio_channels = cfg.audio_channels,
        samplerate = cfg.samplerate,
        sources = cfg.sources,
        channels = cfg.channels,
        depth = cfg.depth,
        nfft = cfg.nfft,
        lstm_layers = cfg.lstm_layers,
        transformer_layers = cfg.transformer_layers,
        hf_card = PRIMARY_SOURCE_HF_CARD,
        github_repo = PRIMARY_SOURCE_GITHUB_REPO,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the HT-Demucs runtime binder — round-trip on the
    //! topology chunk group + negative-space round-trip on the
    //! loud-partial gates + `DemucsStems` surface pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `separate(...)` returning real 4-stem separation results,
    //! but the tensor-name walk + two-branch U-Net composition +
    //! STFT/iSTFT wiring + cross-domain attention + stem-sum path are
    //! all deferred (see the module doc + [`Demucs::separate`]
    //! rustdoc). Fabricating a real-PCM output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
    //! honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config round-trip**: `from_gguf` reads every axis stamped
    //!    by the converter, and falls back cleanly to the
    //!    primary-source defaults for any absent key.
    //! 2. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / empty tensor list /
    //!    unsupported forward surface) fires at its documented
    //!    surface point, in the documented error variant.
    //! 3. **DemucsStems surface pin**: the field shape matches the
    //!    task-hint spelling AND the MUSDB18 4-stem taxonomy.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds an HT-Demucs GGUF carrying the arch tag + name + one
    /// representative encoder tensor whose outer dim matches
    /// `channels`. The topology chunk group is optionally stamped
    /// (`stamp_topology = true`) — when omitted the runtime binder
    /// falls back to the primary-source defaults per key.
    ///
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn demucs_gguf(
        cfg: DemucsConfig,
        stamp_topology: bool,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        if stamp_topology {
            b.add_u32(GGUF_KEY_AUDIO_CHANNELS, cfg.audio_channels);
            b.add_u32(GGUF_KEY_SAMPLERATE, cfg.samplerate);
            b.add_u32(GGUF_KEY_SOURCES, cfg.sources);
            b.add_u32(GGUF_KEY_CHANNELS, cfg.channels);
            b.add_u32(GGUF_KEY_DEPTH, cfg.depth);
            b.add_u32(GGUF_KEY_NFFT, cfg.nfft);
            b.add_u32(GGUF_KEY_LSTM_LAYERS, cfg.lstm_layers);
            b.add_u32(GGUF_KEY_TRANSFORMER_LAYERS, cfg.transformer_layers);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative encoder tensor so the non-emptiness
        // gate passes and the shape-consistency accessor has
        // something to walk. The `channels` dim is deliberately at
        // axis 0 so `matches_config` returns true.
        let c = cfg.channels as u64;
        b.add_tensor(
            "encoder.0.conv.weight",
            GgmlType::F32,
            vec![c, c],
            vec![0u8; (c * c * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. DemucsConfig default matches primary-source HT-Demucs 4-stem axes
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_config_default_matches_primary_source_4stem_axes() {
        // Pin the primary-source axes transcribed from the HF card +
        // upstream `demucs/htdemucs.py` + Rouard et al. §3. A rename
        // or axis-value change would land here in the same commit or
        // fail this test.
        let cfg = DemucsConfig::v4stem_default();
        assert_eq!(cfg.audio_channels, 2);
        assert_eq!(cfg.samplerate, 44_100);
        assert_eq!(cfg.sources, 4);
        assert_eq!(cfg.channels, 48);
        assert_eq!(cfg.depth, 4);
        assert_eq!(cfg.nfft, 4096);
        assert_eq!(cfg.lstm_layers, 2);
        assert_eq!(cfg.transformer_layers, 5);
        // HT-Demucs-specific invariant: the two branches share the
        // same depth (matched U-Net encoders / decoders — the "hybrid"
        // step of Rouard et al. §3 relies on parallel-per-stage skip
        // connections). A future release that splits the branch
        // depths would need distinct axes; a silent unification into
        // a single `depth` axis (as here) is a coincidence of the
        // current release, not derivable.
        assert!(cfg.depth >= 1, "primary source depth must be >= 1");
        // Sanity: `Default` matches `v4stem_default` (both must be
        // primary-source-transcribed constants; no silent divergence).
        assert_eq!(DemucsConfig::default(), cfg);
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf full topology chunk-group round-trip (stamped path)
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_from_gguf_round_trips_stamped_chunk_group() {
        let cfg = DemucsConfig::v4stem_default();
        let file = demucs_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::Permissive),
        );
        let d = Demucs::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every stamped axis reads back into
        // the same DemucsConfig value (converter follow-up sub-wave
        // path).
        assert_eq!(*d.config(), cfg);
        // Permissive weight license is the primary-source default
        // per the upstream `LICENSE` file's MIT — the runtime must
        // surface it verbatim from the provenance chunk. The M2-13
        // compliance gate accepts this artifact in commercial mode
        // (T1 tier — permissive, no `--allow-noncommercial` needed).
        assert_eq!(d.weight_license(), LicenseClass::Permissive);
        assert!(d.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf falls back to primary-source constants on absent chunks
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_config_from_gguf_falls_back_to_primary_source_defaults() {
        // The Demucs converter does NOT currently stamp the
        // `vokra.demucs.*` chunk group (only arch / name / category /
        // upstream_hf / provenance). An already-published GGUF must
        // still load — the fallback path reads the primary-source
        // constants transcribed from the HF card + upstream
        // `htdemucs.py` + paper. Mirror of SortformerConfig::from_gguf
        // / PyanNetConfig::from_gguf fallback pattern.
        let cfg = DemucsConfig::v4stem_default();
        let file = demucs_gguf(
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::Permissive),
        );
        let d = Demucs::from_gguf(&file).expect("chunk-free GGUF must bind via fallback");
        // Every axis fell through to its primary-source default —
        // the loader returns the same values as v4stem_default().
        assert_eq!(d.config().audio_channels, DEFAULT_AUDIO_CHANNELS);
        assert_eq!(d.config().samplerate, DEFAULT_SAMPLERATE);
        assert_eq!(d.config().sources, DEFAULT_SOURCES);
        assert_eq!(d.config().channels, DEFAULT_CHANNELS);
        assert_eq!(d.config().depth, DEFAULT_DEPTH);
        assert_eq!(d.config().nfft, DEFAULT_NFFT);
        assert_eq!(d.config().lstm_layers, DEFAULT_LSTM_LAYERS);
        assert_eq!(d.config().transformer_layers, DEFAULT_TRANSFORMER_LAYERS);
    }

    // -----------------------------------------------------------------------
    // 4. from_gguf honors stamped chunks over defaults (converter forward-compat)
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_config_from_gguf_honors_stamped_chunks_over_defaults() {
        // Simulate a future converter sub-wave that starts stamping
        // the chunk group — a fixture with a deliberately non-default
        // value must round-trip through DemucsConfig::from_gguf and
        // override the primary-source default. This is the
        // forward-compat contract that lets the converter add stamps
        // without a runtime code change.
        let mut cfg = DemucsConfig::v4stem_default();
        cfg.channels = 64; // arbitrary non-default (e.g. hypothetical HT-Demucs-XL)
        cfg.depth = 6;
        let file = demucs_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::Permissive),
        );
        let d = Demucs::from_gguf(&file).expect("stamped GGUF must bind");
        assert_eq!(d.config().channels, 64);
        assert_eq!(d.config().depth, 6);
    }

    // -----------------------------------------------------------------------
    // 5. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_from_gguf_rejects_wrong_arch() {
        // A `sepformer` GGUF handed to the Demucs binder by mistake
        // must fail loud with a specific message rather than silently
        // mis-binding (FR-EX-08). Both `sepformer` and `demucs` share
        // the "audio source separator" category but have completely
        // different topologies (SepFormer = dual-path time-domain
        // Transformer; HT-Demucs = hybrid waveform + spectrogram U-Net
        // + cross-domain attention), so silent aliasing would misroute.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "sepformer");
        b.add_u32(GGUF_KEY_CHANNELS, 48);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Demucs::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`sepformer`") && m.contains("`demucs`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("hybrid"),
                    "message should disambiguate HT-Demucs's hybrid topology to help \
                     the reader, got `{m}`"
                );
                assert!(
                    m.contains("htdemucs_multi"),
                    "message should mention the sibling htdemucs_multi arch tag to \
                     help the reader identify the correct binder for the 4/6-source \
                     variants, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6. from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_from_gguf_rejects_missing_arch() {
        // A GGUF that carries no `vokra.model.arch` at all (e.g. a
        // hand-assembled fixture from an unrelated pipeline) must
        // fail loud rather than mis-bind.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "not-demucs");
        // No `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Demucs::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("demucs"),
                    "message must name the demucs binder so a reader knows which \
                     loader complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. Empty tensor manifest fails loud (never binds all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_from_gguf_rejects_empty_tensor_list() {
        // Correct arch + full chunk group but zero tensors — the
        // DemucsWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_AUDIO_CHANNELS, DEFAULT_AUDIO_CHANNELS);
        b.add_u32(GGUF_KEY_SAMPLERATE, DEFAULT_SAMPLERATE);
        b.add_u32(GGUF_KEY_SOURCES, DEFAULT_SOURCES);
        b.add_u32(GGUF_KEY_CHANNELS, DEFAULT_CHANNELS);
        b.add_u32(GGUF_KEY_DEPTH, DEFAULT_DEPTH);
        b.add_u32(GGUF_KEY_NFFT, DEFAULT_NFFT);
        b.add_u32(GGUF_KEY_LSTM_LAYERS, DEFAULT_LSTM_LAYERS);
        b.add_u32(GGUF_KEY_TRANSFORMER_LAYERS, DEFAULT_TRANSFORMER_LAYERS);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Demucs::from_gguf(&file) else {
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
    // 8. separate returns UnsupportedOp with primary-source anchors +
    //    all three deferred-piece names (waveform-branch / spectrogram-
    //    branch / cross-domain)
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_separate_loud_partial_returns_unsupported_op_with_primary_source_urls() {
        let cfg = DemucsConfig::v4stem_default();
        let file = demucs_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::Permissive),
        );
        let d = Demucs::from_gguf(&file).unwrap();
        // 1 second of stereo 44.1 kHz silence (interleaved f32) —
        // legitimate input shape, so the loud-partial gate fires (not
        // some pre-separate validation).
        let pcm = vec![0.0f32; (DEFAULT_SAMPLERATE * DEFAULT_AUDIO_CHANNELS) as usize];
        let Err(err) = d.separate(&pcm) else {
            panic!("separate must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("demucs separate"),
                    "message must call out the demucs separate surface, got `{m}`"
                );
                // Task hint requires "waveform-branch" + "spectrogram-
                // branch" + "cross-domain" substrings so a reader
                // knows all three trunks are pending.
                assert!(
                    m.contains("waveform-branch"),
                    "message must name the waveform-branch deferred piece, got `{m}`"
                );
                assert!(
                    m.contains("spectrogram-branch"),
                    "message must name the spectrogram-branch deferred piece, got `{m}`"
                );
                assert!(
                    m.contains("cross-domain"),
                    "message must name the cross-domain attention deferred piece, got `{m}`"
                );
                // All three primary-source URLs must be cited.
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD),
                    "message must contain the HF card URL substring \
                     (huggingface.co/facebook/demucs), got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_GITHUB_REPO),
                    "message must contain the upstream repo URL substring \
                     (github.com/facebookresearch/demucs), got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_PAPER),
                    "message must contain the paper URL substring \
                     (arxiv.org/abs/2211.08553), got `{m}`"
                );
                // Every config axis must be echoed so the reader
                // can cross-check what topology the follow-up wave
                // targets.
                assert!(m.contains("channels=48"), "channels axis missing: {m}");
                assert!(m.contains("depth=4"), "depth axis missing: {m}");
                assert!(m.contains("nfft=4096"), "nfft axis missing: {m}");
                assert!(m.contains("lstm_layers=2"), "lstm_layers axis missing: {m}");
                assert!(
                    m.contains("transformer_layers=5"),
                    "transformer_layers axis missing: {m}"
                );
                assert!(m.contains("sources=4"), "sources axis missing: {m}");
                assert!(
                    m.contains("audio_channels=2"),
                    "audio_channels axis missing: {m}"
                );
                assert!(
                    m.contains("samplerate=44100"),
                    "samplerate axis missing: {m}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9. DemucsStems surface pin — fields match task-hint spelling +
    //    MUSDB18 4-stem taxonomy
    // -----------------------------------------------------------------------

    #[test]
    fn demucs_stems_surface_pin() {
        // Surface pin: field names must match the task-hint spelling
        // `{drums, bass, vocals, other}` AND the MUSDB18 4-stem
        // taxonomy per HF card + Rouard et al. §4. A rename or
        // field-shape change would land here in the same commit or
        // fail this test.
        let stems = DemucsStems {
            drums: vec![0.1, 0.2, 0.3],
            bass: vec![0.4, 0.5, 0.6],
            vocals: vec![0.7, 0.8, 0.9],
            other: vec![1.0, 1.1, 1.2],
        };
        // Field-access + type-check smoke — the compiler will refuse
        // to compile if any field is renamed or its type shifts.
        let _drums: &Vec<f32> = &stems.drums;
        let _bass: &Vec<f32> = &stems.bass;
        let _vocals: &Vec<f32> = &stems.vocals;
        let _other: &Vec<f32> = &stems.other;
        // Length invariant: each stem is equal-length under a
        // well-formed separator emit.
        assert_eq!(stems.drums.len(), 3);
        assert_eq!(stems.bass.len(), 3);
        assert_eq!(stems.vocals.len(), 3);
        assert_eq!(stems.other.len(), 3);
        // Derives smoke: Debug + Clone + PartialEq — Clone lets
        // downstream code cache stems, PartialEq lets tests compare
        // stems.
        let cloned: DemucsStems = stems.clone();
        assert_eq!(stems, cloned);
        let dbg = format!("{stems:?}");
        assert!(
            dbg.contains("DemucsStems")
                && dbg.contains("drums")
                && dbg.contains("bass")
                && dbg.contains("vocals")
                && dbg.contains("other"),
            "Debug output must render every MUSDB18 stem field spelling, got `{dbg}`"
        );
        // Sortformer's `SpeakerSegment` uses `end_s` — pinned distinct
        // from the pyannote `duration_s` semantic; this same
        // discipline pins DemucsStems to the MUSDB18 4-stem taxonomy
        // and forbids a silent expansion to a 6-source variant
        // (guitar / piano — the sibling `htdemucs_multi` binder is
        // the correct home for those).
        // The presence-of-fields assertion above already exercises
        // this — a rename or expansion would fail to compile.
    }

    // -----------------------------------------------------------------------
    // 10. Default weight license is Permissive (T1 tier commercial-ok)
    // -----------------------------------------------------------------------

    #[test]
    fn default_weight_license_stamps_permissive_t1_tier() {
        // The Demucs converter's DEFAULT_LICENSE_SPDX is `mit` →
        // LicenseClass::Permissive. The runtime must surface this
        // verbatim so the M2-13 compliance gate accepts the artifact
        // in commercial mode without `--allow-noncommercial` (T1
        // tier — permissive, sibling to Whisper / piper-plus /
        // Silero / CAM++ posture).
        let cfg = DemucsConfig::v4stem_default();
        let file = demucs_gguf(
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::Permissive),
        );
        let d = Demucs::from_gguf(&file).expect("bind");
        assert_eq!(
            d.weight_license(),
            LicenseClass::Permissive,
            "the Demucs converter defaults to Permissive per the upstream `LICENSE` \
             file's MIT — the runtime binder must surface it so the M2-13 compliance \
             gate can accept commercial-mode load (T1 tier)"
        );
        // Missing provenance stamp falls back to Unknown (fail-closed
        // at the gate).
        let file_no_license = demucs_gguf(cfg, /*stamp_topology=*/ false, None);
        let d_no_license = Demucs::from_gguf(&file_no_license).expect("bind without license stamp");
        assert_eq!(
            d_no_license.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed)"
        );
    }

    // -----------------------------------------------------------------------
    // 11. Structural pin — arch tag is stable and distinct from sibling
    //     separator arches
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_separator_arches() {
        // Pin the arch string so a rename would land here in the
        // same commit or fail this test. The sibling separator
        // arches MUST NOT collide with ours — each has a
        // fundamentally different topology.
        assert_eq!(ARCH, "demucs");
        assert_eq!(NAME, "demucs-htdemucs");
        // Direct string comparisons against the sibling arch tags to
        // document the "which sibling should NOT be aliased" contract
        // at test time (a future rename of any sibling arch would
        // land here in the same commit or fail this test).
        assert_ne!(
            ARCH, "sepformer",
            "demucs (hybrid music-separation) and sepformer (dual-path time-domain \
             speech-separation) share the separator category but have different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "tiger_separator",
            "demucs (hybrid music-separation) and tiger_separator (time-frequency \
             dialog/effects/music) share the separator category but have different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "mp_senet",
            "demucs (hybrid music-separation) and mp_senet (magnitude-phase parallel \
             speech enhancement) share the audio category but have different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "bs_roformer",
            "demucs (hybrid music-separation) and bs_roformer (STFT-domain band-split) \
             share the separator category but have different topologies — sharing \
             arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "htdemucs_multi",
            "demucs (base 4-stem HT-Demucs) and htdemucs_multi (4/6-source HT-Demucs \
             variants including `htdemucs_ft` + `htdemucs_6s`) share the HT-Demucs \
             architecture but have different terminal stem counts — sharing arch \
             would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "mossformer2_ss_16k",
            "demucs (hybrid music-separation, 44.1 kHz) and mossformer2_ss_16k \
             (FSMN + gated attention 16 kHz cocktail-party speech separation) share \
             the separator category but have different topologies AND sample rates \
             — sharing arch would mis-route (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 12. matches_config soft accessor honestly reflects shape presence
    // -----------------------------------------------------------------------

    #[test]
    fn matches_config_soft_accessor_finds_channels_axis() {
        // The soft accessor should return true when at least one
        // bound tensor has an axis matching `channels`. The fixture
        // encoder tensor's rows/cols are both `channels` so this must
        // pass.
        let cfg = DemucsConfig::v4stem_default();
        let file = demucs_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::Permissive),
        );
        let d = Demucs::from_gguf(&file).unwrap();
        assert!(
            d.weights.matches_config(d.config()),
            "at least one bound tensor must have an axis matching config.channels"
        );
        // Sanity: a stale config (bogus channels) does NOT match the
        // fixture — pins the accessor as a real check (not a stub
        // that always returns true).
        let stale = DemucsConfig {
            channels: 99999,
            ..cfg
        };
        assert!(
            !d.weights.matches_config(&stale),
            "matches_config must return false for a channels axis with no matching dim"
        );
    }
}
