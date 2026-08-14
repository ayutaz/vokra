//! **M2D** — *Masked Modeling Duo* (`nttcslab/m2d`, **license Unknown /
//! fail-closed**): runtime binder for the `m2d` converter arch.
//!
//! Closes a real read-side gap. The converter
//! (`crates/vokra-convert/src/models/m2d.rs`, SSL audio-encoder wave
//! 2026-08-13) stamps `vokra.model.arch = "m2d"` /
//! `vokra.model.name = "m2d-base"` / `vokra.model.category =
//! "audio-embedding"` / `vokra.provenance.upstream_url =
//! "github.com/nttcslab/m2d"` and passes every F32 / F16 / BF16 tensor
//! through verbatim — but a workspace-wide grep proved **nothing read that
//! arch string back**. Weights converted, and then nothing could load them.
//! This module is that consumer.
//!
//! # Primary sources
//!
//! - Reference code / release: <https://github.com/nttcslab/m2d>
//!   (NTT Communication Science Laboratories)
//! - Paper: Niizumi et al., *"Masked Modeling Duo: Learning Representations
//!   by Encouraging Both Networks to Model the Input"*, ICASSP 2023
//!   (<https://arxiv.org/abs/2210.14648>); TASLP 2024 extension covering
//!   sound-event detection and speech.
//! - License: <https://github.com/nttcslab/m2d/blob/master/LICENSE.pdf> —
//!   a **PDF**, which GitHub's license classifier cannot machine-read
//!   (`/repos/nttcslab/m2d/license` answers `spdx_id: NOASSERTION`,
//!   verified 2026-08-13 per the converter docstring).
//!
//! Everything this module asserts about M2D is traceable to those three
//! anchors. Everything it does **not** know is named explicitly below rather
//! than filled in with a plausible guess (CLAUDE.md「ハルシネーション厳禁」).
//!
//! # What M2D is
//!
//! M2D is a self-supervised **general-audio** encoder trained by a *duo* of
//! networks: an `online` network and a `target` network. Per the paper title,
//! the objective encourages **both** networks to model the input — the online
//! branch is asked to predict the target branch's representation of the
//! masked content rather than only reconstructing raw masked patches. The
//! released base variant is in the ~86M-parameter class (~200 MB), and the
//! repository positions it as an embedding backbone that downstream
//! sound-event-detection / audio-tagging / speaker heads are fine-tuned on
//! top of.
//!
//! **It is therefore a feature extractor, not an end-task model.** The
//! converted checkpoint carries an encoder, not a classifier. This binder
//! exposes exactly that: [`M2d::encode`] (a sequence of hidden states) and
//! [`M2d::embed`] (an utterance-level pooled embedding). **No classification
//! head is invented**, because the checkpoint does not contain one — upstream
//! ships task heads separately, as fine-tuning recipes.
//!
//! # Why `m2d` is its own arch tag
//!
//! Sibling SSL audio/music encoders in the converter tree — `beats`
//! (iterative acoustic tokenizer), `eat` (utterance-level Transformer +
//! inverse block masking), `atst` (teacher–student patchout), `dasheng`
//! (universal single-branch MAE), `mert` / `muq` (music embedding) — plus the
//! wav2vec2-lineage neighbourhood (`wav2vec2_ctc`, `wavlm_sv`, `hubert`,
//! `emotion2vec`) all live nearby, and every one of them has a *different*
//! parameter topology. M2D's masked-modeling-**duo** objective leaves two
//! parallel branches in the state dict; a single-branch MAE loader pointed at
//! such a checkpoint does not crash, it silently binds one branch's tensors
//! and produces plausible-but-wrong embeddings. FR-EX-08 forbids exactly that
//! silent misroute, so the arch gate here is strict and its error enumerates
//! the whole neighbourhood.
//!
//! # Loud-partial classification
//!
//! CLAUDE.md 教訓 (a) —「loud-partial は fake-complete より honest」.
//!
//! **Real (this WP)**:
//!
//! - [`M2d::from_gguf`]: strict `vokra.model.arch == "m2d"` verification that
//!   refuses a foreign GGUF loudly, naming **both** the expected and the
//!   actual tag and enumerating the sibling fleet.
//! - [`M2dConfig::from_gguf`]: a **forward-compatible optional**
//!   `vokra.m2d.*` axis group. The current converter stamps **none** of it,
//!   so a real artifact today resolves to
//!   [`M2dConfigSource::ConverterSilent`] — absence is normal and is *not* an
//!   error. But a key that is **present with the wrong dtype** fails loud
//!   rather than being silently ignored, so a future converter revision
//!   cannot half-land an axis group unnoticed.
//! - [`M2dWeights::from_gguf`]: the tensor manifest over the verbatim
//!   upstream `state_dict` names the converter passes through, with a
//!   non-empty gate plus [`M2dWeights::require_tensor`] /
//!   [`M2dWeights::require_tensor_dims`] lookups that name the missing
//!   tensor, or **both** the expected and the actual dims.
//! - License surfacing that fail-closes to [`LicenseClass::Unknown`], plus
//!   [`M2d::requires_research_flag`] so a caller can see the M2-13 gate state
//!   without re-reading the GGUF.
//! - [`M2dHiddenStates`], a checked carrier for the `[num_frames,
//!   hidden_size]` contract [`M2d::encode`] will return once the forward
//!   lands.
//!
//! **Loud-partial (this WP)**: [`M2d::encode`] and [`M2d::embed`] return
//! [`VokraError::UnsupportedOp`] naming five distinct blockers (see
//! [`M2d::encode`]). No fabricated hidden states or embeddings are **ever**
//! emitted (FR-EX-08 — no silent partial output).
//!
//! # Deliberately not transcribed
//!
//! M2D ships **no HuggingFace mirror** as of 2026-08-13 and therefore no
//! `config.json` to transcribe. Consequently this module states **no**
//! hidden size, layer count, head count, patch geometry, mel-bin count,
//! sample rate, or pooling recipe. Those are the `Option`-typed axes of
//! [`M2dConfig`], all `None` today. Copying a sibling's ViT-Base numbers
//! would be fabrication across a different release, so
//! [`M2dConfig::validate_for_forward`] **refuses** while they are unset
//! rather than defaulting.
//!
//! The same applies to branch selection: whether the inference path reads the
//! `online` or the `target` sub-tree is not stated by any machine-readable
//! primary source available to this wave, so
//! [`M2dConfig::inference_branch`] is `Option<M2dBranch>` and stays `None`.
//! Guessing has no loud failure mode — it silently returns the wrong
//! embedding — which is precisely why it is a blocker and not a default.
//!
//! # Licensing (owner-gated, fail-closed)
//!
//! The converter's `DEFAULT_LICENSE_SPDX` is `"unknown"`, which classifies to
//! [`LicenseClass::Unknown`]: [`LicenseClass::requires_research_flag`] is
//! `true` and [`LicenseClass::redistributable`] is `false`, so the M2-13
//! runtime gate refuses to load without a research flag and the publish path
//! refuses outright. Clearing that is an **owner** action — download
//! `LICENSE.pdf`, read it, confirm the SPDX tier from that primary source,
//! then re-convert with `--license <spdx>`. `docs/license-audit.md` §3.1
//! sign-off stays **blank** (owner-only per
//! `[[feedback-license-signoff-primary-source]]` — CC does not sign).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] are mirrors of the converter's constants — the
//! same rule the sibling binders (`wavlm` / `emotion2vec` / `panns` /
//! `redimnet` / `canary_1b_flash`) follow so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! M2D ships as a PyTorch `.pth` pickle from the upstream release; neither
//! this runtime nor the converter ever touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02). The bridge is an offline uv-managed Python 3.12 sidecar
//! (memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`),
//! mirroring the DAC / Kokoro / UTMOSv2 pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/m2d.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model m2d-base`.
///
/// Deliberately distinct from every sibling SSL audio/music-encoder arch tag
/// (`beats` / `eat` / `atst` / `dasheng` / `mert` / `muq`) and from the
/// wav2vec2-lineage neighbourhood (`wav2vec2_ctc` / `wavlm_sv` / `hubert` /
/// `emotion2vec`). Silently sharing a tag would let runtime dispatch bind a
/// dual-branch M2D checkpoint through a single-branch loader — shape-valid,
/// silently wrong (FR-EX-08).
pub const ARCH: &str = "m2d";

/// Expected `vokra.model.name` value written by the converter — the canonical
/// `m2d-base` size point.
///
/// Sibling release identities (the `m2d-eat` sound-event-detection
/// specialisation, speech-specific fine-tunes) are separate publish
/// identities that would arrive as their own `ModelKind` + `NAME`, following
/// the `snac_24khz` / `snac_44khz` precedent.
pub const NAME: &str = "m2d-base";

/// Expected `vokra.model.category` value — general audio embedding, sibling
/// of `dasheng` / `beats` / `eat` / `atst`.
///
/// Consumed by the model-card generator and the zoo-manifest tier gate so an
/// embedding backbone is never advertised as an ASR or TTS release.
pub const CATEGORY: &str = "audio-embedding";

/// Upstream source tree. M2D is **not** hosted on HuggingFace, so the
/// converter stamps `vokra.provenance.upstream_url` rather than
/// `upstream_hf` — the same posture as the `beats` / `eat` / `atst` /
/// `nsnet2` GitHub-only releases.
pub const UPSTREAM_URL: &str = "github.com/nttcslab/m2d";

/// The converter's default SPDX string: `"unknown"`, which
/// [`LicenseClass::from_license_str`] resolves to [`LicenseClass::Unknown`]
/// (fail-closed under M2-13). See the module docstring "Licensing" section
/// for the owner action that clears it.
pub const DEFAULT_LICENSE_SPDX: &str = "unknown";

/// `vokra.model.category` metadata key (not exported by
/// [`vokra_core::gguf::chunks`], so mirrored here from the converter).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` metadata key — the provenance surface this
/// arch uses, since M2D has no HuggingFace mirror.
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// --- Primary-source anchors, echoed in every loud-partial diagnostic -------

/// Primary-source anchor: the upstream reference code / release tree.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/nttcslab/m2d";
/// Primary-source anchor: the ICASSP 2023 paper (Niizumi et al.).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2210.14648";
/// Primary-source anchor: the non-machine-readable upstream license PDF.
pub const PRIMARY_SOURCE_LICENSE_PDF: &str = "github.com/nttcslab/m2d/blob/master/LICENSE.pdf";

// --- Optional, forward-compatible `vokra.m2d.*` axis group ----------------
//
// The converter stamps NONE of these today. They exist so a follow-up wave
// that walks the upstream release can stamp axes WITHOUT a breaking change,
// and so a half-landed axis group fails loud instead of silently reading as
// absent.

/// Optional axis: Transformer hidden width.
pub const GGUF_KEY_HIDDEN_SIZE: &str = "vokra.m2d.hidden_size";
/// Optional axis: number of Transformer encoder blocks.
pub const GGUF_KEY_NUM_HIDDEN_LAYERS: &str = "vokra.m2d.num_hidden_layers";
/// Optional axis: self-attention head count.
pub const GGUF_KEY_NUM_ATTENTION_HEADS: &str = "vokra.m2d.num_attention_heads";
/// Optional axis: patch height in mel bins (spectrogram patch tokenization).
pub const GGUF_KEY_PATCH_HEIGHT: &str = "vokra.m2d.patch_height";
/// Optional axis: patch width in frames (spectrogram patch tokenization).
pub const GGUF_KEY_PATCH_WIDTH: &str = "vokra.m2d.patch_width";
/// Optional axis: mel-filterbank bin count of the front-end.
pub const GGUF_KEY_N_MELS: &str = "vokra.m2d.n_mels";
/// Optional axis: input sample rate in Hz.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.m2d.sample_rate";
/// Optional axis (**string**): which duo branch the inference path reads —
/// `"online"` or `"target"`. See [`M2dBranch`].
pub const GGUF_KEY_INFERENCE_BRANCH: &str = "vokra.m2d.inference_branch";

/// Conventional `state_dict` prefix of the duo's **online** branch.
///
/// Illustrative only — recorded so [`M2dWeights::branch_tensor_count`] has a
/// documented argument, **not** asserted as the verified upstream manifest.
/// No in-repo transcription of M2D's `state_dict` naming exists yet; that is
/// blocker (2) of [`M2d::encode`].
pub const BRANCH_PREFIX_ONLINE: &str = "online.";

/// Conventional `state_dict` prefix of the duo's **target** branch. Same
/// caveat as [`BRANCH_PREFIX_ONLINE`].
pub const BRANCH_PREFIX_TARGET: &str = "target.";

// ---------------------------------------------------------------------------
// M2dBranch — which half of the duo the inference path reads.
// ---------------------------------------------------------------------------

/// Which network of the Masked Modeling **Duo** an inference pass reads.
///
/// M2D trains two networks jointly. Only one of them is the encoder a
/// downstream task consumes, and **this module does not know which** — no
/// machine-readable primary source available to this wave states it (M2D has
/// no HuggingFace mirror, so there is no `config.json`, and the repository's
/// own license is a PDF the classifier cannot read).
///
/// Guessing is uniquely dangerous here: both branches are shape-compatible,
/// so a wrong pick produces a full-rank, finite, plausible-looking embedding
/// that is simply *not the model's output*. There is no loud failure mode to
/// catch it downstream. That is why this rides an `Option` on
/// [`M2dConfig::inference_branch`] and why
/// [`M2dConfig::validate_for_forward`] refuses while it is `None`
/// (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M2dBranch {
    /// The `online` network of the duo.
    Online,
    /// The `target` network of the duo.
    Target,
}

impl M2dBranch {
    /// The wire string written under [`GGUF_KEY_INFERENCE_BRANCH`].
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Target => "target",
        }
    }

    /// The conventional `state_dict` prefix associated with this branch.
    ///
    /// See [`BRANCH_PREFIX_ONLINE`] for the "illustrative, not verified"
    /// caveat.
    #[inline]
    #[must_use]
    pub const fn tensor_prefix(self) -> &'static str {
        match self {
            Self::Online => BRANCH_PREFIX_ONLINE,
            Self::Target => BRANCH_PREFIX_TARGET,
        }
    }

    /// Parses the wire string. Returns `None` for anything other than
    /// `"online"` / `"target"` — the caller turns that into a loud error
    /// naming both accepted values (FR-EX-08, never a silent default).
    #[inline]
    #[must_use]
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "online" => Some(Self::Online),
            "target" => Some(Self::Target),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// M2dConfigSource — how much of the optional axis group an artifact carries.
// ---------------------------------------------------------------------------

/// How much of the optional `vokra.m2d.*` axis group a bound artifact
/// carries.
///
/// Reported so a caller can distinguish "the converter is silent, as
/// expected" from "somebody stamped half a group", without inspecting each
/// axis. Neither state is an error at bind time; both are refused by
/// [`M2dConfig::validate_for_forward`], with different messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M2dConfigSource {
    /// **The expected state today.** No `vokra.m2d.*` key present — the
    /// converter stamps none of them.
    ConverterSilent,
    /// Some, but not all, axes present — a half-landed converter revision.
    PartiallyStamped,
    /// Every axis present, including the branch selector. Only in this state
    /// can [`M2dConfig::validate_for_forward`] succeed.
    FullyStamped,
}

// ---------------------------------------------------------------------------
// M2dConfig — the optional axis group. Every field is `Option`: nothing is
// defaulted, because no primary source states a value (FR-EX-08 — refusing
// beats fabricating).
// ---------------------------------------------------------------------------

/// The optional `vokra.m2d.*` topology axes.
///
/// **Every field is `Option` and every one is `None` for an artifact produced
/// by the current converter.** That is deliberate: M2D ships no HuggingFace
/// mirror and hence no `config.json`, so there is nothing to transcribe, and
/// borrowing a sibling's ViT-Base numbers would be fabrication across a
/// different release (CLAUDE.md「ハルシネーション厳禁」).
///
/// [`from_gguf`](Self::from_gguf) therefore treats **absence as normal** but
/// a **present key of the wrong dtype as a loud error**, so a future
/// converter revision cannot half-land the group unnoticed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct M2dConfig {
    /// Transformer hidden width, if stamped.
    pub hidden_size: Option<u32>,
    /// Transformer encoder block count, if stamped.
    pub num_hidden_layers: Option<u32>,
    /// Self-attention head count, if stamped.
    pub num_attention_heads: Option<u32>,
    /// Spectrogram patch height in mel bins, if stamped.
    pub patch_height: Option<u32>,
    /// Spectrogram patch width in frames, if stamped.
    pub patch_width: Option<u32>,
    /// Mel-filterbank bin count of the front-end, if stamped.
    pub n_mels: Option<u32>,
    /// Input sample rate in Hz, if stamped.
    pub sample_rate: Option<u32>,
    /// Which duo branch the inference path reads, if stamped. See
    /// [`M2dBranch`] for why this can never be defaulted.
    pub inference_branch: Option<M2dBranch>,
}

impl M2dConfig {
    /// Reads the optional `vokra.m2d.*` axis group.
    ///
    /// - Key absent → the field stays `None` (**normal** — the current
    ///   converter stamps nothing).
    /// - Key present with a readable value → bound.
    /// - Key present with the **wrong dtype** → loud [`VokraError::ModelLoad`]
    ///   naming the key, the expected dtype and the actual one. Silently
    ///   ignoring it would let a half-landed converter revision look like a
    ///   silent converter (FR-EX-08).
    /// - [`GGUF_KEY_INFERENCE_BRANCH`] present with a value outside
    ///   `{"online", "target"}` → loud, naming both accepted values.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] for any present-but-unreadable
    ///   `vokra.m2d.*` key.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn opt_u32(gguf: &GgufFile, key: &str) -> Result<Option<u32>> {
            match gguf.get(key) {
                None => Ok(None),
                Some(v) => match v.as_u64() {
                    Some(n) => Ok(Some(n as u32)),
                    None => Err(VokraError::ModelLoad(format!(
                        "m2d: GGUF key `{key}` is present but is not an unsigned \
                         integer (actual GGUF value type: {actual:?}). The \
                         `vokra.m2d.*` axis group is OPTIONAL — the current \
                         converter stamps none of it, and absence is normal — but a \
                         key that IS present must be readable, otherwise a \
                         half-landed converter revision would be indistinguishable \
                         from a silent converter and the runtime would bind \
                         fabricated topology (FR-EX-08). Either drop the key or \
                         stamp it as a u32.",
                        actual = v.value_type()
                    ))),
                },
            }
        }

        let inference_branch = match gguf.get(GGUF_KEY_INFERENCE_BRANCH) {
            None => None,
            Some(v) => {
                let s = v.as_str().ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "m2d: GGUF key `{GGUF_KEY_INFERENCE_BRANCH}` is present but is \
                         not a string (actual GGUF value type: {actual:?}). It selects \
                         which network of the Masked Modeling Duo the inference path \
                         reads and must be exactly `online` or `target` (FR-EX-08 — \
                         never a silent default, because both branches are \
                         shape-compatible and a wrong pick returns a \
                         plausible-but-wrong embedding with no loud failure mode).",
                        actual = v.value_type()
                    ))
                })?;
                Some(M2dBranch::from_wire(s).ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "m2d: GGUF key `{GGUF_KEY_INFERENCE_BRANCH}` is `{s}`, but the \
                         only accepted values are `online` and `target` — the two \
                         networks of the Masked Modeling Duo (Niizumi et al., \
                         {paper}). Refusing rather than defaulting: both branches are \
                         shape-compatible, so a wrong pick returns a \
                         plausible-but-wrong embedding with no loud failure mode \
                         (FR-EX-08).",
                        paper = PRIMARY_SOURCE_PAPER
                    ))
                })?)
            }
        };

        Ok(Self {
            hidden_size: opt_u32(gguf, GGUF_KEY_HIDDEN_SIZE)?,
            num_hidden_layers: opt_u32(gguf, GGUF_KEY_NUM_HIDDEN_LAYERS)?,
            num_attention_heads: opt_u32(gguf, GGUF_KEY_NUM_ATTENTION_HEADS)?,
            patch_height: opt_u32(gguf, GGUF_KEY_PATCH_HEIGHT)?,
            patch_width: opt_u32(gguf, GGUF_KEY_PATCH_WIDTH)?,
            n_mels: opt_u32(gguf, GGUF_KEY_N_MELS)?,
            sample_rate: opt_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            inference_branch,
        })
    }

    /// Names of the axes that are still unstamped, in a stable order.
    ///
    /// Empty exactly when [`Self::source`] is
    /// [`M2dConfigSource::FullyStamped`].
    #[must_use]
    pub fn missing_axes(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.hidden_size.is_none() {
            missing.push(GGUF_KEY_HIDDEN_SIZE);
        }
        if self.num_hidden_layers.is_none() {
            missing.push(GGUF_KEY_NUM_HIDDEN_LAYERS);
        }
        if self.num_attention_heads.is_none() {
            missing.push(GGUF_KEY_NUM_ATTENTION_HEADS);
        }
        if self.patch_height.is_none() {
            missing.push(GGUF_KEY_PATCH_HEIGHT);
        }
        if self.patch_width.is_none() {
            missing.push(GGUF_KEY_PATCH_WIDTH);
        }
        if self.n_mels.is_none() {
            missing.push(GGUF_KEY_N_MELS);
        }
        if self.sample_rate.is_none() {
            missing.push(GGUF_KEY_SAMPLE_RATE);
        }
        if self.inference_branch.is_none() {
            missing.push(GGUF_KEY_INFERENCE_BRANCH);
        }
        missing
    }

    /// How much of the optional axis group this config carries.
    ///
    /// A converter-produced artifact today reports
    /// [`M2dConfigSource::ConverterSilent`].
    #[must_use]
    pub fn source(&self) -> M2dConfigSource {
        let missing = self.missing_axes().len();
        if missing == 0 {
            M2dConfigSource::FullyStamped
        } else if missing == TOTAL_OPTIONAL_AXES {
            M2dConfigSource::ConverterSilent
        } else {
            M2dConfigSource::PartiallyStamped
        }
    }

    /// Refuses unless every axis is stamped.
    ///
    /// The encoder forward cannot be shaped without them, and no primary
    /// source available to this wave supplies a default, so refusing is the
    /// honest behaviour — defaulting would silently bind fabricated topology
    /// (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] naming every unstamped axis and both
    ///   primary sources.
    pub fn validate_for_forward(&self) -> Result<()> {
        let missing = self.missing_axes();
        if missing.is_empty() {
            return Ok(());
        }
        Err(VokraError::UnsupportedOp(format!(
            "m2d config (loud-partial): the optional `vokra.m2d.*` axis group is \
             {state:?} — {n} of {total} axes are unstamped: [{missing}]. The current \
             converter stamps NONE of this group, and M2D ships no HuggingFace mirror \
             as of 2026-08-13, so there is no upstream `config.json` to transcribe \
             them from. This binder refuses to default them: borrowing a sibling SSL \
             encoder's ViT-Base numbers would be fabrication across a different \
             release, and the runtime would then bind fabricated topology with no \
             loud failure mode (FR-EX-08). Primary sources: code {code}, paper \
             {paper}.",
            state = self.source(),
            n = missing.len(),
            total = TOTAL_OPTIONAL_AXES,
            missing = missing.join(", "),
            code = PRIMARY_SOURCE_CODE,
            paper = PRIMARY_SOURCE_PAPER,
        )))
    }
}

/// Number of axes in the optional `vokra.m2d.*` group — used to tell
/// "converter is silent" apart from "half a group landed".
pub const TOTAL_OPTIONAL_AXES: usize = 8;

// ---------------------------------------------------------------------------
// M2dHiddenStates — the shape contract `encode` will honour.
// ---------------------------------------------------------------------------

/// A `[num_frames, hidden_size]` block of encoder hidden states, row-major.
///
/// M2D is a **feature extractor**: its output is a sequence of per-patch /
/// per-frame hidden states, not class logits. This type carries that contract
/// explicitly so the shape is legible at the call site rather than implied by
/// a bare `Vec<f32>`.
#[derive(Debug, Clone, PartialEq)]
pub struct M2dHiddenStates {
    num_frames: usize,
    hidden_size: usize,
    data: Vec<f32>,
}

impl M2dHiddenStates {
    /// Builds a hidden-state block, checking that `data.len() == num_frames *
    /// hidden_size`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when the payload length does not
    ///   match the declared shape, naming both the expected and the actual
    ///   length.
    pub fn new(num_frames: usize, hidden_size: usize, data: Vec<f32>) -> Result<Self> {
        let expected = num_frames.checked_mul(hidden_size).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "m2d hidden states: declared shape [{num_frames}, {hidden_size}] \
                     overflows usize"
            ))
        })?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "m2d hidden states: payload length {actual} does not match the \
                 declared shape [{num_frames}, {hidden_size}] (expected {expected} \
                 elements)",
                actual = data.len()
            )));
        }
        Ok(Self {
            num_frames,
            hidden_size,
            data,
        })
    }

    /// Number of time steps (rows).
    #[inline]
    #[must_use]
    pub const fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Hidden width (columns).
    #[inline]
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The flat row-major payload.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// One row, or `None` when `frame >= num_frames`.
    #[must_use]
    pub fn frame(&self, frame: usize) -> Option<&[f32]> {
        if frame >= self.num_frames {
            return None;
        }
        let start = frame * self.hidden_size;
        self.data.get(start..start + self.hidden_size)
    }
}

// ---------------------------------------------------------------------------
// M2dWeights — the tensor manifest plus loud lookups.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an M2D GGUF, keyed by verbatim upstream
/// `state_dict` name.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF carrying zero tensors is refused rather than silently running
/// an all-zero forward (FR-EX-08) — an ~86M-parameter class encoder always
/// carries hundreds of parameter tensors, so an empty manifest always signals
/// a mis-produced GGUF.
#[derive(Debug)]
pub struct M2dWeights {
    /// Tensor names + GGUF-side dims discovered on disk, in file order.
    tensors: Vec<(String, Vec<usize>)>,
}

impl M2dWeights {
    /// Scans `gguf` for the M2D `state_dict` tensors.
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
            return Err(VokraError::ModelLoad(format!(
                "m2d: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate M2D checkpoint is in the ~86M-parameter \
                 class (arch={ARCH}, name={NAME}) and always carries hundreds of \
                 parameter tensors across the duo's two branches; zero tensors always \
                 signals a mis-produced GGUF. Re-run `vokra-cli convert --model \
                 m2d-base` against an upstream `{UPSTREAM_URL}` release checkpoint."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Every bound tensor name, in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    /// How many bound tensor names start with `prefix`.
    ///
    /// Intended for duo-branch triage (see [`BRANCH_PREFIX_ONLINE`] /
    /// [`BRANCH_PREFIX_TARGET`], both illustrative rather than verified).
    #[must_use]
    pub fn branch_tensor_count(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
    }

    /// Looks up one tensor's dims, failing loud and **naming the tensor** when
    /// it is absent.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the missing tensor, the manifest
    ///   size, and the repro command.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, dims)| dims.as_slice())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "m2d: required tensor `{name}` is absent from the GGUF (the \
                     manifest carries {count} tensors). The converter copies every \
                     float tensor under its VERBATIM upstream `state_dict` name, so an \
                     absent name means either the checkpoint does not carry it or the \
                     expected name is wrong — and nothing in-repo has transcribed \
                     M2D's `state_dict` naming yet (see `M2d::encode` blocker 2). \
                     Refusing rather than substituting a zero tensor (FR-EX-08). \
                     Re-run `vokra-cli convert --model m2d-base` against an upstream \
                     `{UPSTREAM_URL}` release checkpoint.",
                    count = self.tensors.len()
                ))
            })
    }

    /// Looks up one tensor and checks its dims, naming **both** the expected
    /// and the actual shape on mismatch.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the tensor is absent (via
    ///   [`Self::require_tensor`]) or its dims differ from `expected`.
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<&[usize]> {
        let dims = self.require_tensor(name)?;
        if dims != expected {
            return Err(VokraError::ModelLoad(format!(
                "m2d: tensor `{name}` has dims {dims:?}, expected {expected:?}. \
                 Refusing to bind a shape-mismatched tensor — reading it anyway would \
                 produce shape-valid garbage rather than a crash (FR-EX-08)."
            )));
        }
        Ok(dims)
    }
}

// ---------------------------------------------------------------------------
// M2d — the runtime binder handle.
// ---------------------------------------------------------------------------

/// M2D (*Masked Modeling Duo*, `nttcslab/m2d`) runtime binder — a
/// self-supervised **general-audio feature extractor**.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`encode`](Self::encode) for a sequence of hidden states or
/// [`embed`](Self::embed) for an utterance-level pooled embedding. There is
/// deliberately **no classification method**: the converted checkpoint
/// contains an encoder, not a task head — upstream ships those separately as
/// fine-tuning recipes.
///
/// See the module doc for the implementation-status matrix and the FR-EX-08
/// loud-error contract on the deferred forward.
#[derive(Debug)]
pub struct M2d {
    config: M2dConfig,
    weights: M2dWeights,
    weight_license: LicenseClass,
    name: Option<String>,
    category: Option<String>,
    upstream_url: Option<String>,
}

impl M2d {
    /// Binds an M2D GGUF: verifies arch strictly, reads the optional
    /// `vokra.m2d.*` axis group, discovers the tensor manifest, and surfaces
    /// the stamped provenance.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key, so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent, or is
    ///   not `"m2d"` (a sibling SSL-encoder GGUF handed here by mistake fails
    ///   with both tags named and the neighbourhood enumerated).
    /// - [`VokraError::ModelLoad`] when a `vokra.m2d.*` key is present with
    ///   the wrong dtype ([`M2dConfig::from_gguf`]).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`M2dWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch gate FIRST, so a mis-typed model fails with a specific
        //    message instead of a downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "m2d: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model m2d-base`? Note that the \
                     sibling SSL audio/music-encoder arch tags — `beats` (iterative \
                     acoustic tokenizer), `eat` (utterance-level Transformer + inverse \
                     block masking), `atst` (teacher-student patchout), `dasheng` \
                     (universal single-branch MAE), `mert` and `muq` (music embedding) \
                     — plus the wav2vec2-lineage neighbourhood `wav2vec2_ctc` (CTC ASR \
                     head), `wavlm_sv` (XVector speaker head), `hubert` (bare SSL \
                     encoder) and `emotion2vec` (9-way emotion head), all live in the \
                     same neighbourhood but are DIFFERENT topologies. M2D's \
                     masked-modeling-duo objective leaves TWO parallel branches \
                     (`online` and `target`) in the state dict; a single-branch loader \
                     pointed at such a checkpoint does not crash, it silently binds \
                     one branch and returns a plausible-but-wrong embedding. FR-EX-08 \
                     forbids that silent misroute, so the arch tags stay distinct."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "m2d: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native m2d GGUF (was it produced by `vokra-cli convert \
                     --model m2d-base`? expected arch `{ARCH}`)."
                )));
            }
        }

        // 2. Optional axis group. Absent is normal; present-but-unreadable is
        //    loud.
        let config = M2dConfig::from_gguf(file)?;

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = M2dWeights::from_gguf(file)?;

        // 4. Provenance surfacing. Fail-closed to `Unknown` — which is also
        //    the converter's production default for this arch, since upstream
        //    `LICENSE.pdf` is not machine-readable.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        let read_str = |key: &str| -> Option<String> {
            file.get(key).and_then(|v| v.as_str()).map(str::to_owned)
        };

        Ok(Self {
            config,
            weights,
            weight_license,
            name: read_str(chunks::KEY_MODEL_NAME),
            category: read_str(KEY_MODEL_CATEGORY),
            upstream_url: read_str(KEY_PROVENANCE_UPSTREAM_URL),
        })
    }

    /// The optional `vokra.m2d.*` axes as bound (all `None` for a
    /// converter-produced artifact today).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &M2dConfig {
        &self.config
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &M2dWeights {
        &self.weights
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The stamped weight-license class.
    ///
    /// The M2D converter's default is `unknown` → [`LicenseClass::Unknown`],
    /// because upstream's `LICENSE.pdf` is not machine-readable. A GGUF
    /// missing the stamp also reads back as `Unknown` (fail-closed at the
    /// M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Whether the M2-13 compliance gate demands an explicit research flag
    /// before this artifact may be loaded — `true` for the `Unknown` class
    /// this arch defaults to.
    #[inline]
    #[must_use]
    pub fn requires_research_flag(&self) -> bool {
        self.weight_license.requires_research_flag()
    }

    /// The stamped `vokra.model.name`, if present.
    #[inline]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, if present.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_url`, if present. M2D has no
    /// HuggingFace mirror, so provenance rides this key rather than
    /// `upstream_hf`.
    #[inline]
    #[must_use]
    pub fn upstream_url(&self) -> Option<&str> {
        self.upstream_url.as_deref()
    }

    /// Encodes a PCM waveform to a sequence of encoder hidden states.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. Five blockers stand between the
    /// current landing and a real forward:
    ///
    /// 1. **No branch selection.** M2D trains a *duo*; which of `online` /
    ///    `target` the inference path reads is not stated by any
    ///    machine-readable primary source available to this wave, and both
    ///    branches are shape-compatible, so a guess returns a
    ///    plausible-but-wrong embedding with no loud failure mode. Cleared by
    ///    stamping [`GGUF_KEY_INFERENCE_BRANCH`].
    /// 2. **No tensor-name manifest.** The converter copies every float
    ///    tensor under its verbatim upstream `state_dict` name and nothing
    ///    in-repo transcribes M2D's naming, so walking guessed names into
    ///    typed slots would bind shape-valid garbage.
    /// 3. **No topology axes.** The converter stamps none of the optional
    ///    `vokra.m2d.*` group and there is no upstream `config.json` to
    ///    transcribe (M2D ships no HuggingFace mirror as of 2026-08-13) — see
    ///    [`M2dConfig::validate_for_forward`].
    /// 4. **No patch-embedding front-end.** The log-mel → spectrogram-patch
    ///    tokenization needs the patch geometry and mel-bin count from
    ///    blocker 3 before `vokra_ops::mel` / `vokra_ops::waveform_frontend`
    ///    can be pointed at it.
    /// 5. **Missing primitive.** No ViT-style Transformer encoder over 2-D
    ///    spectrogram patch tokens is composed in `vokra-ops`; the encoders
    ///    that exist (`vokra_ops::conformer`, `zipformer`, `ebranchformer`)
    ///    are 1-D ASR encoders with different token geometry.
    ///
    /// **No fabricated hidden states are ever emitted** (FR-EX-08 — no silent
    /// partial output).
    ///
    /// `_pcm` is the raw mono f32 waveform in `[-1, 1]`. Its expected sample
    /// rate is itself blocker 3, so no resampling is attempted here — that
    /// would be a silent assumption.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn encode(&self, _pcm: &[f32]) -> Result<M2dHiddenStates> {
        let _ = _pcm;
        Err(forward_loud_partial(&self.config, "encode", ENCODE_OUTPUT))
    }

    /// Encodes a PCM waveform to a single utterance-level pooled embedding.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] with the same five blockers as
    /// [`encode`](Self::encode), plus a sixth specific to this surface: the
    /// **pooling recipe itself is unresolved**. M2D is published as an
    /// embedding backbone, but exactly how the per-patch hidden states are
    /// reduced to one clip-level vector (which layer(s), mean vs. mean+max,
    /// whether a normalisation follows) is defined by the upstream inference
    /// wrapper, not by anything this wave can read. Applying a plain mean and
    /// calling it "the M2D embedding" would be fabrication.
    ///
    /// **No fabricated embedding is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred pooled-embedding forward.
    pub fn embed(&self, _pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = _pcm;
        Err(forward_loud_partial(&self.config, "embed", EMBED_OUTPUT))
    }
}

/// Output-contract clause used by [`M2d::encode`]'s loud-partial message.
const ENCODE_OUTPUT: &str = "a [num_frames, hidden_size] block of encoder hidden states (M2D is a feature \
     extractor — the checkpoint carries an encoder, not a task head, so no \
     classification output exists to emit)";

/// Output-contract clause used by [`M2d::embed`]'s loud-partial message.
const EMBED_OUTPUT: &str = "an utterance-level pooled embedding, whose POOLING RECIPE is itself unresolved \
     (which layer(s) are pooled, mean vs. mean+max, whether a normalisation follows) \
     — it is defined by the upstream inference wrapper, not by anything this wave can \
     read, so applying a plain mean and calling it the M2D embedding would be \
     fabrication";

/// Builds the loud-partial [`VokraError::UnsupportedOp`] shared by
/// [`M2d::encode`] and [`M2d::embed`].
///
/// Names all five blockers by exact identifier, echoes the current axis-group
/// state, and cites all three primary sources (code + paper + license PDF) so
/// a reader diagnosing the gap has exactly three places to walk — the
/// `wavlm` / `emotion2vec` / `panns` / `canary_1b_flash` loud-partial-message
/// precedent (CLAUDE.md 教訓 (a)).
fn forward_loud_partial(cfg: &M2dConfig, surface: &str, output: &str) -> VokraError {
    let missing = cfg.missing_axes();
    let missing_desc = if missing.is_empty() {
        "none (every axis stamped)".to_owned()
    } else {
        missing.join(", ")
    };
    let branch = match cfg.inference_branch {
        Some(b) => b.as_str(),
        None => "UNSET",
    };
    VokraError::UnsupportedOp(format!(
        "m2d {surface} (loud-partial): the M2D (Masked Modeling Duo) encoder forward \
         is deferred. Target output: {output}. Five blockers must clear first: \
         (1) NO BRANCH SELECTION — M2D trains a duo of an `online` and a `target` \
         network (Niizumi et al., {paper}); which one the inference path reads is not \
         stated by any machine-readable primary source available to this wave, and \
         both branches are shape-compatible, so a guess returns a \
         plausible-but-wrong embedding with NO loud failure mode. Current \
         `{branch_key}` = {branch}. \
         (2) NO TENSOR-NAME MANIFEST — the converter copies every float tensor under \
         its VERBATIM upstream `state_dict` name and nothing in-repo transcribes \
         M2D's naming, so walking guessed names into typed slots would bind \
         shape-valid garbage. \
         (3) NO TOPOLOGY AXES — the optional `vokra.m2d.*` group is {state:?}; \
         unstamped axes: [{missing_desc}]. M2D ships NO HuggingFace mirror as of \
         2026-08-13, so there is no upstream `config.json` to transcribe, and \
         borrowing a sibling SSL encoder's ViT-Base numbers would be fabrication \
         across a different release. \
         (4) NO PATCH-EMBEDDING FRONT-END — the log-mel to spectrogram-patch \
         tokenization needs the patch geometry and mel-bin count from blocker (3) \
         before `vokra_ops::mel` / `vokra_ops::waveform_frontend` can be pointed at \
         it. \
         (5) MISSING PRIMITIVE — no ViT-style Transformer encoder over 2-D \
         spectrogram patch tokens is composed in `vokra-ops`; the encoders that do \
         exist (`vokra_ops::conformer`, `zipformer`, `ebranchformer`) are 1-D ASR \
         encoders with different token geometry. \
         Primary sources: code {code}, paper {paper}, license {license_pdf} (a PDF \
         GitHub's classifier cannot read, which is why this arch defaults to \
         LicenseClass::Unknown and fails closed at the M2-13 gate). \
         Runtime cannot fabricate an output (FR-EX-08 — no silent partial output).",
        state = cfg.source(),
        branch_key = GGUF_KEY_INFERENCE_BRANCH,
        code = PRIMARY_SOURCE_CODE,
        paper = PRIMARY_SOURCE_PAPER,
        license_pdf = PRIMARY_SOURCE_LICENSE_PDF,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the M2D runtime binder.
    //!
    //! # What "round-trip" means here
    //!
    //! On real audio this would be `encode(...)` returning real hidden
    //! states, but the encoder forward is deferred (module doc +
    //! [`M2d::encode`] rustdoc). Fabricating an output would violate
    //! CLAUDE.md 教訓 (a) —「loud-partial は fake-complete より honest」.
    //!
    //! The semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` match the converter, and
    //!    the arch tag is distinct from every sibling SSL-encoder tag.
    //! 2. **Metadata round-trip** — a converter-shaped GGUF binds, and the
    //!    optional axis group reads back as `ConverterSilent`.
    //! 3. **Loud negative space** — missing arch, foreign arch, empty tensor
    //!    manifest, missing tensor, dim mismatch, present-but-wrong-dtype
    //!    axis, and a bogus branch string each fire at their documented
    //!    surface in their documented variant.
    //! 4. **Loud-partial contract** — `encode` / `embed` name the missing
    //!    primitive and every primary source.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a GGUF shaped like the one the M2D converter emits: arch +
    /// name + category + upstream_url + optional license stamp + two
    /// representative duo-branch tensors.
    ///
    /// The tensor names mirror the converter's own test fixtures
    /// (`online.blocks.0.attn.qkv.weight` / `target.blocks.0.attn.qkv.weight`)
    /// and are **illustrative**, not a verified upstream manifest — see
    /// [`BRANCH_PREFIX_ONLINE`].
    fn m2d_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        b.add_tensor(
            "online.blocks.0.attn.qkv.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor online");
        b.add_tensor(
            "target.blocks.0.attn.qkv.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor target");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1 — Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "m2d", "m2d arch tag pin");
        assert_eq!(NAME, "m2d-base", "canonical size-point name pin");
        assert_eq!(CATEGORY, "audio-embedding", "category pin");
        assert_eq!(UPSTREAM_URL, "github.com/nttcslab/m2d", "upstream URL pin");
        assert_eq!(
            DEFAULT_LICENSE_SPDX, "unknown",
            "M2D's upstream LICENSE.pdf is not machine-readable, so the converter \
             default must stay `unknown` (fail-closed under M2-13)"
        );
        // The fail-closed posture must actually be fail-closed.
        let cls = LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX);
        assert_eq!(cls, LicenseClass::Unknown);
        assert!(
            cls.requires_research_flag(),
            "Unknown must demand a research flag (FR-CP-03 fail-closed)"
        );
        assert!(
            !cls.redistributable(),
            "an unclassifiable weight is never republished"
        );
        // Metadata keys.
        assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            KEY_PROVENANCE_UPSTREAM_URL, "vokra.provenance.upstream_url",
            "M2D has no HF mirror, so provenance rides `upstream_url`, not `upstream_hf`"
        );
        // Axis-group width must match the field count `missing_axes` walks.
        assert_eq!(
            M2dConfig::default().missing_axes().len(),
            TOTAL_OPTIONAL_AXES,
            "TOTAL_OPTIONAL_AXES must equal the number of axes missing_axes reports"
        );
    }

    // -----------------------------------------------------------------------
    // 2 — Arch-tag distinctness pin across the SSL-encoder neighbourhood
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_ssl_encoder_arches() {
        for sibling in [
            "beats",
            "eat",
            "atst",
            "dasheng",
            "mert",
            "muq",
            "wav2vec2_ctc",
            "wavlm_sv",
            "hubert",
            "emotion2vec",
        ] {
            assert_ne!(
                ARCH, sibling,
                "m2d (dual-branch masked-modeling duo) and {sibling} are different \
                 topologies — sharing an arch tag would let a single-branch loader \
                 silently bind one branch of an M2D checkpoint (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3 — Metadata round-trip on a converter-shaped GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        let file = m2d_gguf(Some(LicenseClass::Unknown));
        let m = M2d::from_gguf(&file).expect("converter-shaped GGUF must bind");

        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_url(), Some(UPSTREAM_URL));
        assert_eq!(m.tensor_count(), 2);

        // The converter stamps NO `vokra.m2d.*` axis — absence is normal.
        assert_eq!(
            m.config().source(),
            M2dConfigSource::ConverterSilent,
            "a converter-produced artifact carries none of the optional axis group"
        );
        assert_eq!(m.config().hidden_size, None);
        assert_eq!(m.config().inference_branch, None);

        // Fail-closed licensing: Unknown demands the research flag.
        assert_eq!(m.weight_license(), LicenseClass::Unknown);
        assert!(m.requires_research_flag());

        // Duo-branch triage over the illustrative fixture names.
        assert_eq!(m.weights().branch_tensor_count(BRANCH_PREFIX_ONLINE), 1);
        assert_eq!(m.weights().branch_tensor_count(BRANCH_PREFIX_TARGET), 1);
        let names: Vec<&str> = m.weights().tensor_names().collect();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = m2d_gguf(None);
        let m = M2d::from_gguf(&file).expect("valid arch must bind");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an absent provenance stamp must fail closed, never read as permissive"
        );
        assert!(m.requires_research_flag());
    }

    // -----------------------------------------------------------------------
    // 4 — Arch metadata absent → loud ModelLoad
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = M2d::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native m2d GGUF"),
                    "message must name the surface, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model m2d-base"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5 — Foreign arch → loud ModelLoad naming BOTH expected and actual
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // A `dasheng` GGUF (single-branch universal MAE) handed here by
        // mistake must fail loud rather than silently binding one branch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "dasheng");
        b.add_string(chunks::KEY_MODEL_NAME, "dasheng-base");
        b.add_tensor("dasheng.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = M2d::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`dasheng`"),
                    "message must name the ACTUAL arch, got `{m}`"
                );
                assert!(
                    m.contains("`m2d`"),
                    "message must name the EXPECTED arch, got `{m}`"
                );
                for sibling in ["beats", "eat", "atst", "mert", "muq", "wavlm_sv"] {
                    assert!(
                        m.contains(sibling),
                        "message must enumerate sibling `{sibling}`, got `{m}`"
                    );
                }
                assert!(
                    m.contains("online") && m.contains("target"),
                    "message must explain the dual-branch misroute hazard, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6 — Empty tensor manifest → loud (never binds an all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = M2d::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model m2d-base"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7 — Missing tensor → loud error NAMING the tensor; dim mismatch names
    //     both shapes
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = m2d_gguf(None);
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");

        // Present: dims read back as converted.
        let dims = m
            .weights()
            .require_tensor("online.blocks.0.attn.qkv.weight")
            .expect("fixture tensor must be present");
        assert_eq!(dims, &[2_usize, 3]);

        // Absent: loud, naming the tensor.
        let Err(err) = m
            .weights()
            .require_tensor("online.blocks.99.mlp.fc1.weight")
        else {
            panic!("expected ModelLoad for an absent tensor");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("online.blocks.99.mlp.fc1.weight"),
                    "message must NAME the missing tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Dim mismatch: loud, naming BOTH the expected and the actual shape.
        let Err(err) = m
            .weights()
            .require_tensor_dims("online.blocks.0.attn.qkv.weight", &[7, 11])
        else {
            panic!("expected ModelLoad on a dim mismatch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("[2, 3]"),
                    "message must name the ACTUAL dims, got `{msg}`"
                );
                assert!(
                    msg.contains("[7, 11]"),
                    "message must name the EXPECTED dims, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Matching dims succeed.
        m.weights()
            .require_tensor_dims("target.blocks.0.attn.qkv.weight", &[2, 3])
            .expect("matching dims must bind");
    }

    // -----------------------------------------------------------------------
    // 8 — Optional axis group: absent is fine, present-but-wrong-dtype is
    //     loud, bogus branch string is loud
    // -----------------------------------------------------------------------

    #[test]
    fn optional_axis_present_with_wrong_dtype_fails_loud() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // Wrong dtype: a String where a u32 is expected.
        b.add_string(GGUF_KEY_HIDDEN_SIZE, "768");
        b.add_tensor("online.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = M2d::from_gguf(&file) else {
            panic!("expected ModelLoad for a present-but-wrong-dtype axis");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_HIDDEN_SIZE),
                    "message must name the offending key, got `{m}`"
                );
                assert!(
                    m.contains("not an unsigned integer"),
                    "message must name the dtype problem, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn inference_branch_rejects_a_value_outside_the_duo() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_INFERENCE_BRANCH, "student");
        b.add_tensor("online.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = M2d::from_gguf(&file) else {
            panic!("expected ModelLoad for a branch outside {{online, target}}");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("student"),
                    "message must echo the bad value: {m}"
                );
                assert!(
                    m.contains("`online`") && m.contains("`target`"),
                    "message must name BOTH accepted values, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn fully_stamped_axis_group_round_trips_and_validates() {
        // A future converter revision stamping the whole group must bind and
        // pass `validate_for_forward` — the forward-compatibility contract.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, 768);
        b.add_u32(GGUF_KEY_NUM_HIDDEN_LAYERS, 12);
        b.add_u32(GGUF_KEY_NUM_ATTENTION_HEADS, 12);
        b.add_u32(GGUF_KEY_PATCH_HEIGHT, 16);
        b.add_u32(GGUF_KEY_PATCH_WIDTH, 16);
        b.add_u32(GGUF_KEY_N_MELS, 80);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16_000);
        b.add_string(GGUF_KEY_INFERENCE_BRANCH, M2dBranch::Online.as_str());
        b.add_tensor("online.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let m = M2d::from_gguf(&file).expect("fully stamped GGUF must bind");
        let cfg = m.config();
        // NOTE: these values are a synthetic FIXTURE, not a primary-source
        // claim about M2D's real topology — see the module doc's
        // "Deliberately not transcribed" section.
        assert_eq!(cfg.hidden_size, Some(768));
        assert_eq!(cfg.num_hidden_layers, Some(12));
        assert_eq!(cfg.num_attention_heads, Some(12));
        assert_eq!(cfg.patch_height, Some(16));
        assert_eq!(cfg.patch_width, Some(16));
        assert_eq!(cfg.n_mels, Some(80));
        assert_eq!(cfg.sample_rate, Some(16_000));
        assert_eq!(cfg.inference_branch, Some(M2dBranch::Online));
        assert_eq!(cfg.source(), M2dConfigSource::FullyStamped);
        assert!(cfg.missing_axes().is_empty());
        cfg.validate_for_forward()
            .expect("a fully stamped group must validate");
        assert_eq!(
            M2dBranch::Online.tensor_prefix(),
            BRANCH_PREFIX_ONLINE,
            "branch prefix mapping pin"
        );
        assert_eq!(M2dBranch::from_wire("target"), Some(M2dBranch::Target));
        assert_eq!(M2dBranch::from_wire("Online"), None, "wire form is exact");
    }

    #[test]
    fn partially_stamped_axis_group_is_distinguishable_and_refused() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, 768);
        // …and nothing else.
        b.add_tensor("online.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let m = M2d::from_gguf(&file).expect("a half-stamped group still binds");
        assert_eq!(
            m.config().source(),
            M2dConfigSource::PartiallyStamped,
            "half a group must be distinguishable from a silent converter"
        );
        let Err(err) = m.config().validate_for_forward() else {
            panic!("a partial axis group must be refused for forward");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains(GGUF_KEY_INFERENCE_BRANCH),
                    "message must list the unstamped branch axis, got `{msg}`"
                );
                assert!(
                    !msg.contains(GGUF_KEY_HIDDEN_SIZE),
                    "the STAMPED axis must not be listed as missing, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9 — encode / embed loud-partial: name the missing primitive + sources
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partials_naming_every_blocker_and_source() {
        let file = m2d_gguf(Some(LicenseClass::Unknown));
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");

        // 1 s of legitimately shaped mono PCM, so the loud-partial gate is
        // what fires — not some pre-encode length validation.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("m2d encode"), "surface must be named: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // All five blockers, by their exact identifiers.
                assert!(
                    msg.contains("NO BRANCH SELECTION"),
                    "blocker 1 missing: {msg}"
                );
                assert!(
                    msg.contains("NO TENSOR-NAME MANIFEST"),
                    "blocker 2 missing: {msg}"
                );
                assert!(msg.contains("NO TOPOLOGY AXES"), "blocker 3 missing: {msg}");
                assert!(
                    msg.contains("NO PATCH-EMBEDDING FRONT-END"),
                    "blocker 4 missing: {msg}"
                );
                // The MISSING PRIMITIVE must be named explicitly, per the
                // loud-partial contract.
                assert!(
                    msg.contains("MISSING PRIMITIVE"),
                    "blocker 5 missing: {msg}"
                );
                assert!(
                    msg.contains("ViT-style Transformer encoder"),
                    "the missing primitive must be named exactly: {msg}"
                );
                assert!(
                    msg.contains("vokra_ops::conformer"),
                    "message should name the primitives that DO exist so the reader \
                     can see why they do not fit: {msg}"
                );

                // Honest about what the encoder does and does not carry.
                assert!(
                    msg.contains("feature extractor"),
                    "message must state M2D is a feature extractor, not a task \
                     model: {msg}"
                );

                // Current axis-group state echoed.
                assert!(
                    msg.contains("ConverterSilent"),
                    "message must echo the axis-group state: {msg}"
                );
                assert!(
                    msg.contains(GGUF_KEY_INFERENCE_BRANCH) && msg.contains("UNSET"),
                    "message must report the unset branch selector: {msg}"
                );

                // All three primary sources cited.
                for url in [
                    PRIMARY_SOURCE_CODE,
                    PRIMARY_SOURCE_PAPER,
                    PRIMARY_SOURCE_LICENSE_PDF,
                ] {
                    assert!(msg.contains(url), "primary source `{url}` not cited: {msg}");
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn embed_loud_partials_and_calls_out_the_unresolved_pooling() {
        let file = m2d_gguf(None);
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");
        let pcm = vec![0.0_f32; 16_000];

        let Err(err) = m.embed(&pcm) else {
            panic!("embed must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("m2d embed"), "surface must be named: {msg}");
                assert!(
                    msg.contains("POOLING RECIPE"),
                    "embed must call out the unresolved pooling recipe: {msg}"
                );
                assert!(
                    msg.contains("MISSING PRIMITIVE"),
                    "embed shares the encoder blockers: {msg}"
                );
                assert!(
                    msg.contains(PRIMARY_SOURCE_PAPER),
                    "primary source must be cited: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10 — M2dHiddenStates shape contract
    // -----------------------------------------------------------------------

    #[test]
    fn hidden_states_enforce_their_declared_shape() {
        let hs = M2dHiddenStates::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("well-shaped payload must bind");
        assert_eq!(hs.num_frames(), 3);
        assert_eq!(hs.hidden_size(), 2);
        assert_eq!(hs.frame(0), Some(&[1.0, 2.0][..]));
        assert_eq!(hs.frame(2), Some(&[5.0, 6.0][..]));
        assert_eq!(hs.frame(3), None, "out-of-range frame must be None");
        assert_eq!(hs.data().len(), 6);

        let Err(err) = M2dHiddenStates::new(3, 2, vec![1.0, 2.0]) else {
            panic!("a payload/shape mismatch must be refused");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("[3, 2]") && msg.contains('6'),
                    "message must name the declared shape and expected length: {msg}"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }
}
