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
//! anchors, or to the converter's own per-axis transcription of them.
//! Everything it does **not** know is named explicitly below rather than
//! filled in with a plausible guess (CLAUDE.md「ハルシネーション厳禁」).
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
//! parallel branches in the state dict; a single-branch loader pointed at
//! such a checkpoint does not crash, it silently binds one branch's tensors
//! and produces plausible-but-wrong embeddings. FR-EX-08 forbids exactly that
//! silent misroute, so the arch gate here is strict and its error enumerates
//! the whole neighbourhood.
//!
//! # The `vokra.m2d.*` axis group is REQUIRED
//!
//! An earlier revision of this module treated the group as *optional* and
//! forward-compatible, because the converter of the day stamped none of it.
//! **That is no longer true.** The converter now stamps all
//! [`TOTAL_STAMPED_AXES`] keys, every value transcribed from a primary source
//! its constants cite line by line (upstream `examples/portable_m2d.py`
//! `get_backbone()` / `Config` / `get_to_melspec()`, plus arXiv:2210.14648 §3
//! and §4.1).
//!
//! So [`M2dConfig::from_gguf`] treats each key as **mandatory** and fails with
//! a loud [`VokraError::ModelLoad`] naming the absent one — the `wavlm_sv`
//! posture. There is deliberately **no primary-source constant fallback**: the
//! producer stamps these, so a silent default would let a mismatched artifact
//! (a differently-configured release, a half-written GGUF, the 32 kHz M2D
//! variant whose `sample_rate` differs) bind as though it were the canonical
//! one, with no loud failure mode (FR-EX-08).
//!
//! # What the axis group does *not* carry
//!
//! [`vokra_ops::vit::ViTAttrs`] has twelve axes. The stamped group supplies
//! five of them (`embed_dim`, `depth`, `n_heads`, `patch_h`, `patch_w`); the
//! other seven — listed in [`UNSTAMPED_VIT_AXES`] — are carried by no
//! `vokra.m2d.*` key at all. The converter's docstring records `mlp_ratio = 4`
//! and LayerNorm `eps = 1e-6` from upstream `get_backbone()` and then
//! deliberately declines to stamp them, on the grounds that they "land only if
//! and when the binder grows fields for them, in the same commit".
//!
//! This binder therefore does **not** read them, and equally does not assume
//! them: [`M2dConfig::vit_attrs`] takes them from the caller as an explicit
//! [`M2dUnstampedAxes`], mirroring the no-defaults posture of `ViTAttrs`
//! itself. Hard-coding them here would put a constant in the runtime that the
//! artifact cannot contradict — the same silent-mismatch hazard the required
//! reader exists to prevent.
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
//! - [`M2dConfig::from_gguf`]: the **required** eight-key `vokra.m2d.*` reader
//!   described above.
//! - [`M2dConfig::vit_attrs`]: the config → [`vokra_ops::vit::ViTAttrs`]
//!   mapping, validated through `ViTAttrs::validate`.
//! - [`M2dWeights::from_gguf`]: the tensor manifest over the verbatim upstream
//!   `state_dict` names the converter passes through, with a non-empty gate
//!   plus [`M2dWeights::require_tensor`] / [`M2dWeights::require_tensor_dims`]
//!   lookups that name the missing tensor, or **both** the expected and the
//!   actual dims, plus [`M2dWeights::branch_triage`] — an *observation* of how
//!   the bound manifest is prefixed, never an assertion about how it ought to
//!   be.
//! - License surfacing that fail-closes to [`LicenseClass::Unknown`], plus
//!   [`M2d::requires_research_flag`] so a caller can see the M2-13 gate state
//!   without re-reading the GGUF.
//! - [`M2dHiddenStates`], a checked carrier for the `[num_frames,
//!   hidden_size]` contract [`M2d::encode`] will return once the forward
//!   lands.
//!
//! **Loud-partial (this WP)**: [`M2d::encode`] and [`M2d::embed`] return
//! [`VokraError::UnsupportedOp`] naming three remaining blockers (see
//! [`M2d::encode`]), plus a fourth specific to `embed`. No fabricated hidden
//! states or embeddings are **ever** emitted (FR-EX-08 — no silent partial
//! output).
//!
//! **Resolved since the gate was first written** — the message says so
//! explicitly, so nobody re-reports them: the `vokra.m2d.*` group is stamped,
//! branch selection rides [`GGUF_KEY_INFERENCE_BRANCH`], and
//! [`vokra_ops::vit`] now supplies the 2-D patch embedding + pre-norm
//! Transformer encoder this arch needs (the `conformer` / `zipformer` /
//! `ebranchformer` encoders were 1-D ASR encoders with different token
//! geometry, and are not substitutes).
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
//! [`DEFAULT_LICENSE_SPDX`] and every `GGUF_KEY_*` are mirrors of the
//! converter's constants — the same rule the sibling binders (`wavlm` /
//! `emotion2vec` / `panns` / `redimnet` / `canary_qwen`) follow so
//! `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
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
use vokra_ops::vit::{GeluKind, PosEmbedPolicy, ViTAttrs};

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

// --- The REQUIRED `vokra.m2d.*` axis group --------------------------------
//
// Key spellings are byte-identical to the converter's `KEY_M2D_*` constants;
// its `topology_axis_keys_mirror_the_runtime_binder` test pins them from the
// other side, and `stamped_axis_key_spellings_mirror_the_converter` below
// pins them from this one. The two halves only meet if both hold.

/// Required axis: Transformer hidden width. Maps to
/// [`vokra_ops::vit::ViTAttrs::embed_dim`].
pub const GGUF_KEY_HIDDEN_SIZE: &str = "vokra.m2d.hidden_size";
/// Required axis: number of Transformer encoder blocks. Maps to
/// [`vokra_ops::vit::ViTAttrs::depth`].
pub const GGUF_KEY_NUM_HIDDEN_LAYERS: &str = "vokra.m2d.num_hidden_layers";
/// Required axis: self-attention head count. Maps to
/// [`vokra_ops::vit::ViTAttrs::n_heads`].
pub const GGUF_KEY_NUM_ATTENTION_HEADS: &str = "vokra.m2d.num_attention_heads";
/// Required axis: patch height in mel bins. Maps to
/// [`vokra_ops::vit::ViTAttrs::patch_h`].
pub const GGUF_KEY_PATCH_HEIGHT: &str = "vokra.m2d.patch_height";
/// Required axis: patch width in frames. Maps to
/// [`vokra_ops::vit::ViTAttrs::patch_w`].
pub const GGUF_KEY_PATCH_WIDTH: &str = "vokra.m2d.patch_width";
/// Required axis: mel-filterbank bin count of the front-end.
///
/// A front-end axis, **not** a [`vokra_ops::vit::ViTAttrs`] field: it fixes
/// the height of the `[n_mels, n_frames]` plane the encoder consumes.
pub const GGUF_KEY_N_MELS: &str = "vokra.m2d.n_mels";
/// Required axis: input sample rate in Hz. Also a front-end axis rather than
/// a `ViTAttrs` field.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.m2d.sample_rate";
/// Required axis (**string**): which duo branch the inference path reads —
/// `"online"` or `"target"`. See [`M2dBranch`].
pub const GGUF_KEY_INFERENCE_BRANCH: &str = "vokra.m2d.inference_branch";

/// Number of keys in the required `vokra.m2d.*` group.
pub const TOTAL_STAMPED_AXES: usize = 8;

/// Every required `vokra.m2d.*` key, in a stable order.
pub const STAMPED_AXIS_KEYS: [&str; TOTAL_STAMPED_AXES] = [
    GGUF_KEY_HIDDEN_SIZE,
    GGUF_KEY_NUM_HIDDEN_LAYERS,
    GGUF_KEY_NUM_ATTENTION_HEADS,
    GGUF_KEY_PATCH_HEIGHT,
    GGUF_KEY_PATCH_WIDTH,
    GGUF_KEY_N_MELS,
    GGUF_KEY_SAMPLE_RATE,
    GGUF_KEY_INFERENCE_BRANCH,
];

/// The [`vokra_ops::vit::ViTAttrs`] fields that **no** `vokra.m2d.*` key
/// carries, and which [`M2dConfig::vit_attrs`] therefore takes from the
/// caller as an [`M2dUnstampedAxes`].
///
/// Named as `ViTAttrs` field identifiers rather than as GGUF keys, precisely
/// because there are no GGUF keys for them.
pub const UNSTAMPED_VIT_AXES: [&str; 7] = [
    "stride_h",
    "stride_w",
    "n_prepended_tokens",
    "mlp_ratio",
    "layer_norm_eps",
    "gelu",
    "pos_embed_policy",
];

/// Conventional `state_dict` prefix of the duo's **online** branch.
///
/// Illustrative only — recorded so [`M2dWeights::branch_tensor_count`] has a
/// documented argument, **not** asserted as the verified upstream manifest.
/// No in-repo transcription of M2D's `state_dict` naming exists yet; that is
/// blocker (1) of [`M2d::encode`].
pub const BRANCH_PREFIX_ONLINE: &str = "online.";

/// Conventional `state_dict` prefix of the duo's **target** branch. Same
/// caveat as [`BRANCH_PREFIX_ONLINE`].
pub const BRANCH_PREFIX_TARGET: &str = "target.";

// ---------------------------------------------------------------------------
// M2dBranch — which half of the duo the inference path reads.
// ---------------------------------------------------------------------------

/// Which network of the Masked Modeling **Duo** an inference pass reads.
///
/// M2D trains two networks jointly, and only one of them is the encoder a
/// downstream task consumes. Guessing would be uniquely dangerous: both
/// branches are shape-compatible, so a wrong pick produces a full-rank,
/// finite, plausible-looking embedding that is simply *not the model's
/// output*, with no loud failure mode downstream.
///
/// It is not guessed. The converter stamps [`GGUF_KEY_INFERENCE_BRANCH`] from
/// a primary source that states it outright — paper §3 defines "the online
/// encoder f_θ" against "The target network … consists only of momentum
/// encoder f_ξ" and then concludes "After the training, we transfer only the
/// f_θ as a pre-trained model", corroborated operationally by upstream
/// `util/to_encoder_only_weight.py`. This binder reads that stamp and refuses
/// anything outside `{"online", "target"}` (FR-EX-08).
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
// M2dConfig — the REQUIRED axis group. Nothing is defaulted: the converter
// stamps every key, so a silent fallback would let a mismatched artifact bind
// (FR-EX-08).
// ---------------------------------------------------------------------------

/// The `vokra.m2d.*` topology axes, all of them mandatory.
///
/// Each field is stamped by `crates/vokra-convert/src/models/m2d.rs` from a
/// primary source its own constants cite line by line. See the module
/// docstring section "The `vokra.m2d.*` axis group is REQUIRED" for why
/// there is no constant fallback here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M2dConfig {
    /// Transformer hidden width, from [`GGUF_KEY_HIDDEN_SIZE`].
    pub hidden_size: u32,
    /// Transformer encoder block count, from [`GGUF_KEY_NUM_HIDDEN_LAYERS`].
    pub num_hidden_layers: u32,
    /// Self-attention head count, from [`GGUF_KEY_NUM_ATTENTION_HEADS`].
    pub num_attention_heads: u32,
    /// Spectrogram patch height in mel bins, from [`GGUF_KEY_PATCH_HEIGHT`].
    pub patch_height: u32,
    /// Spectrogram patch width in frames, from [`GGUF_KEY_PATCH_WIDTH`].
    pub patch_width: u32,
    /// Mel-filterbank bin count of the front-end, from [`GGUF_KEY_N_MELS`].
    pub n_mels: u32,
    /// Input sample rate in Hz, from [`GGUF_KEY_SAMPLE_RATE`].
    pub sample_rate: u32,
    /// Which duo branch the inference path reads, from
    /// [`GGUF_KEY_INFERENCE_BRANCH`].
    pub inference_branch: M2dBranch,
}

impl M2dConfig {
    /// Reads every required `vokra.m2d.*` key.
    ///
    /// - Key absent → loud [`VokraError::ModelLoad`] **naming that key**.
    /// - Key present with the wrong GGUF value type → loud
    ///   [`VokraError::ModelLoad`] naming the key, what was expected, and the
    ///   actual value type. Distinguished from absence so a half-written
    ///   artifact reports the truth rather than "missing".
    /// - [`GGUF_KEY_INFERENCE_BRANCH`] present with a value outside
    ///   `{"online", "target"}` → loud, naming both accepted values.
    ///
    /// There is no primary-source constant fallback anywhere in this reader.
    /// The producer stamps these axes, so defaulting a missing one would let
    /// a mismatched artifact — a differently-configured release, a truncated
    /// GGUF, the separate 32 kHz M2D identity — bind as the canonical one
    /// with no loud failure mode (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any of the [`TOTAL_STAMPED_AXES`]
    ///   keys is absent, unreadable, or (for the branch selector) outside the
    ///   duo.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            hidden_size: req_u32(gguf, GGUF_KEY_HIDDEN_SIZE)?,
            num_hidden_layers: req_u32(gguf, GGUF_KEY_NUM_HIDDEN_LAYERS)?,
            num_attention_heads: req_u32(gguf, GGUF_KEY_NUM_ATTENTION_HEADS)?,
            patch_height: req_u32(gguf, GGUF_KEY_PATCH_HEIGHT)?,
            patch_width: req_u32(gguf, GGUF_KEY_PATCH_WIDTH)?,
            n_mels: req_u32(gguf, GGUF_KEY_N_MELS)?,
            sample_rate: req_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            inference_branch: req_branch(gguf)?,
        })
    }

    /// Maps the stamped axes onto a [`vokra_ops::vit::ViTAttrs`], taking the
    /// seven axes no `vokra.m2d.*` key carries from `unstamped`.
    ///
    /// Where each `ViTAttrs` field comes from:
    ///
    /// - `embed_dim` ← [`GGUF_KEY_HIDDEN_SIZE`]
    /// - `depth` ← [`GGUF_KEY_NUM_HIDDEN_LAYERS`]
    /// - `n_heads` ← [`GGUF_KEY_NUM_ATTENTION_HEADS`]
    /// - `patch_h` ← [`GGUF_KEY_PATCH_HEIGHT`]
    /// - `patch_w` ← [`GGUF_KEY_PATCH_WIDTH`]
    /// - `stride_h`, `stride_w`, `n_prepended_tokens`, `mlp_ratio`,
    ///   `layer_norm_eps`, `gelu`, `pos_embed_policy` ← the caller, via
    ///   `unstamped` ([`UNSTAMPED_VIT_AXES`])
    ///
    /// [`GGUF_KEY_N_MELS`] and [`GGUF_KEY_SAMPLE_RATE`] are deliberately
    /// absent from that list: they describe the log-mel plane fed *to* the
    /// encoder, not the encoder, and `ViTAttrs` has no field for either.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `ViTAttrs::validate` when the
    ///   combined axis set is inconsistent — a zero axis, a head count that
    ///   does not divide the width, a non-positive `mlp_ratio` or
    ///   `layer_norm_eps`, or an `mlp_ratio` that rounds the MLP hidden width
    ///   to zero.
    pub fn vit_attrs(&self, unstamped: &M2dUnstampedAxes) -> Result<ViTAttrs> {
        let attrs = ViTAttrs {
            // --- stamped by the converter, read above -----------------------
            embed_dim: self.hidden_size as usize,
            depth: self.num_hidden_layers as usize,
            n_heads: self.num_attention_heads as usize,
            patch_h: self.patch_height as usize,
            patch_w: self.patch_width as usize,
            // --- caller-supplied: no `vokra.m2d.*` key carries these --------
            stride_h: unstamped.stride_h,
            stride_w: unstamped.stride_w,
            n_prepended_tokens: unstamped.n_prepended_tokens,
            mlp_ratio: unstamped.mlp_ratio,
            layer_norm_eps: unstamped.layer_norm_eps,
            gelu: unstamped.gelu,
            pos_embed_policy: unstamped.pos_embed_policy,
        };
        attrs.validate()?;
        Ok(attrs)
    }
}

/// Reads one mandatory u32 axis, distinguishing "absent" from "present but
/// not an unsigned integer".
fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    match gguf.get(key) {
        Some(v) => match v.as_u64() {
            Some(n) => Ok(n as u32),
            None => Err(VokraError::ModelLoad(format!(
                "m2d: GGUF key `{key}` is present but is not an unsigned integer \
                 (actual GGUF value type: {actual:?}). Every `vokra.m2d.*` axis is \
                 REQUIRED and the converter stamps it with `add_u32`, so a key of \
                 another type means a hand-edited or half-written artifact. Refusing \
                 rather than ignoring it: a silently skipped axis would be \
                 indistinguishable from a correctly stamped one and the runtime would \
                 shape the encoder from fabricated topology (FR-EX-08).",
                actual = v.value_type()
            ))),
        },
        None => Err(VokraError::ModelLoad(format!(
            "m2d: GGUF is missing required u32 chunk `{key}` — the converter \
             (`vokra-cli convert --model m2d-base`) stamps all {TOTAL_STAMPED_AXES} \
             `vokra.m2d.*` axes, each transcribed from a primary source it cites line \
             by line (upstream `examples/portable_m2d.py` `get_backbone()` / `Config` \
             / `get_to_melspec()`, plus {paper} §3 and §4.1), so a proper conversion \
             carries the whole group. This binder refuses to substitute a \
             primary-source constant for the absent key (FR-EX-08): the producer \
             stamps it, so a default here would let a mismatched artifact — a \
             differently-configured release, a truncated GGUF, or the separate 32 kHz \
             M2D identity whose `{sr_key}` differs — bind as the canonical one with no \
             loud failure mode. Re-run the conversion against an upstream \
             `{UPSTREAM_URL}` release checkpoint.",
            paper = PRIMARY_SOURCE_PAPER,
            sr_key = GGUF_KEY_SAMPLE_RATE,
        ))),
    }
}

/// Reads the mandatory string-valued branch selector.
fn req_branch(gguf: &GgufFile) -> Result<M2dBranch> {
    let value = gguf.get(GGUF_KEY_INFERENCE_BRANCH).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "m2d: GGUF is missing required string chunk `{GGUF_KEY_INFERENCE_BRANCH}` \
             — it selects which network of the Masked Modeling Duo the inference path \
             reads, and the converter stamps it as `online` on the authority of \
             {paper} §3 (\"After the training, we transfer only the f_θ as a \
             pre-trained model\", f_θ being the online encoder). This binder will not \
             default it: both branches are shape-compatible, so a wrong pick returns a \
             plausible-but-wrong embedding with NO loud failure mode (FR-EX-08). \
             Re-run `vokra-cli convert --model m2d-base`.",
            paper = PRIMARY_SOURCE_PAPER
        ))
    })?;
    let wire = value.as_str().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "m2d: GGUF key `{GGUF_KEY_INFERENCE_BRANCH}` is present but is not a \
             string (actual GGUF value type: {actual:?}). It must be exactly `online` \
             or `target` — the converter stamps it with `add_string`, so another type \
             means a hand-edited artifact (FR-EX-08).",
            actual = value.value_type()
        ))
    })?;
    M2dBranch::from_wire(wire).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "m2d: GGUF key `{GGUF_KEY_INFERENCE_BRANCH}` is `{wire}`, but the only \
             accepted values are `online` and `target` — the two networks of the \
             Masked Modeling Duo (Niizumi et al., {paper}). Refusing rather than \
             defaulting: both branches are shape-compatible, so a wrong pick returns a \
             plausible-but-wrong embedding with no loud failure mode (FR-EX-08).",
            paper = PRIMARY_SOURCE_PAPER
        ))
    })
}

// ---------------------------------------------------------------------------
// M2dUnstampedAxes — the ViT axes the artifact does not carry.
// ---------------------------------------------------------------------------

/// The seven [`vokra_ops::vit::ViTAttrs`] axes that no `vokra.m2d.*` key
/// carries, supplied by the caller.
///
/// This type deliberately implements neither `Default` nor any `*_base()`
/// constructor, for the same reason `ViTAttrs` does not: a default here would
/// be a number the artifact cannot contradict, so a release configured
/// differently from the one whoever wrote the default had in mind would bind
/// silently wrong (FR-EX-08).
///
/// # Where a caller gets these
///
/// The converter's module docstring records two of them from upstream
/// `examples/portable_m2d.py` `get_backbone()`, which it transcribes as
/// `LocalViT(in_chans=1, …, embed_dim=768, depth=12, num_heads=12,
/// mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))` — i.e.
/// `mlp_ratio = 4` and `layer_norm_eps = 1e-6` — and then declines to stamp
/// them, noting they "land only if and when the binder grows fields for them,
/// in the same commit". A caller who has read that same line supplies them
/// here explicitly; this binder does not assume them on the caller's behalf.
///
/// The remaining five (`stride_h`, `stride_w`, `n_prepended_tokens`, `gelu`,
/// `pos_embed_policy`) are recorded by **no** in-repo transcription at all and
/// must be read off the upstream release or a real checkpoint. In particular
/// `n_prepended_tokens` is settled by the row count of the positional table
/// relative to the patch grid, which needs the checkpoint — see blocker (1) of
/// [`M2d::encode`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct M2dUnstampedAxes {
    /// Patch stride along the mel-bin axis. Equal to the patch height for a
    /// non-overlapping tiling, smaller for overlapping patches.
    pub stride_h: usize,
    /// Patch stride along the frame axis. Equal to the patch width for a
    /// non-overlapping tiling, smaller for overlapping patches.
    pub stride_w: usize,
    /// How many learned tokens are prepended ahead of the patch tokens
    /// (class / distillation). May be `0`.
    pub n_prepended_tokens: usize,
    /// MLP hidden width as a multiple of the embedding width.
    pub mlp_ratio: f32,
    /// LayerNorm epsilon used by every norm in the encoder.
    pub layer_norm_eps: f32,
    /// Which GELU formulation the MLP uses.
    pub gelu: GeluKind,
    /// What to do when the positional table length does not match the runtime
    /// token count.
    pub pos_embed_policy: PosEmbedPolicy,
}

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
// M2dBranchTriage — an OBSERVATION of how the bound manifest is prefixed.
// ---------------------------------------------------------------------------

/// How the bound tensor manifest is split across the duo's branch prefixes.
///
/// Every field is **counted from the artifact**, never asserted about it. The
/// distinction matters: which shape a real M2D GGUF has is exactly blocker (1)
/// of [`M2d::encode`], and reporting the observed counts turns "the manifest
/// is unverified" into a concrete fact a reader can act on.
///
/// Two upstream artifacts plausibly reach the converter and produce different
/// shapes here — the raw duo checkpoint (both prefixes populated) and the
/// encoder-only export, since upstream `util/to_encoder_only_weight.py`
/// persists `PortableM2D(src).backbone.state_dict()`, a single **unprefixed**
/// encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M2dBranchTriage {
    /// Tensors whose name starts with [`BRANCH_PREFIX_ONLINE`].
    pub online_prefixed: usize,
    /// Tensors whose name starts with [`BRANCH_PREFIX_TARGET`].
    pub target_prefixed: usize,
    /// Tensors carrying neither prefix.
    pub unprefixed: usize,
}

impl M2dBranchTriage {
    /// Total tensors observed — the sum of the three counters.
    #[inline]
    #[must_use]
    pub const fn total(&self) -> usize {
        self.online_prefixed + self.target_prefixed + self.unprefixed
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

    /// Counts how the bound manifest splits across the duo's branch prefixes.
    ///
    /// Purely observational — see [`M2dBranchTriage`].
    #[must_use]
    pub fn branch_triage(&self) -> M2dBranchTriage {
        let online_prefixed = self.branch_tensor_count(BRANCH_PREFIX_ONLINE);
        let target_prefixed = self.branch_tensor_count(BRANCH_PREFIX_TARGET);
        M2dBranchTriage {
            online_prefixed,
            target_prefixed,
            unprefixed: self.tensors.len() - online_prefixed - target_prefixed,
        }
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
                     M2D's `state_dict` naming yet (see `M2d::encode` blocker 1). \
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
    /// Binds an M2D GGUF: verifies arch strictly, reads the required
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
    /// - [`VokraError::ModelLoad`] when any required `vokra.m2d.*` key is
    ///   absent or unreadable ([`M2dConfig::from_gguf`]).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`M2dWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch gate FIRST, so a mis-typed model fails with a specific
        //    message instead of a downstream missing-key error.
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

        // 2. Required axis group. Every key must be present and readable.
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

    /// The required `vokra.m2d.*` axes as bound.
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
    /// Returns [`VokraError::UnsupportedOp`]. Three blockers stand between the
    /// current landing and a real forward:
    ///
    /// 1. **Unverified tensor-name manifest.** The converter copies every
    ///    float tensor under its verbatim upstream `state_dict` name, and
    ///    nothing in-repo transcribes M2D's naming — the names both crates'
    ///    fixtures use are illustrative, not read off a checkpoint. Two
    ///    sub-questions only a real checkpoint settles: whether the released
    ///    artifact keeps the `online.` / `target.` prefixes at all (upstream
    ///    `util/to_encoder_only_weight.py` persists a single **unprefixed**
    ///    `backbone.state_dict()`), and whether attention ships one fused
    ///    `qkv` projection or three separate ones, since
    ///    [`vokra_ops::vit::ViTAttnWeights`] holds `wq` / `wk` / `wv`
    ///    separately and a fused split's concat order is a convention rather
    ///    than a transcription.
    /// 2. **Unstamped ViT axes.** [`vokra_ops::vit::ViTAttrs`] has twelve
    ///    axes and the stamped group supplies five; the other seven
    ///    ([`UNSTAMPED_VIT_AXES`]) reach [`M2dConfig::vit_attrs`] from the
    ///    caller. `encode` has no caller-supplied axes, so it cannot shape the
    ///    encoder from this artifact alone.
    /// 3. **No mel front-end binding.** [`vokra_ops::vit::ViTEncoder::forward`]
    ///    consumes a `[n_mels, n_frames]` log-mel plane, not PCM.
    ///    [`GGUF_KEY_N_MELS`] and [`GGUF_KEY_SAMPLE_RATE`] are stamped, but
    ///    `n_fft` / hop / window / `f_min` / `f_max` and the log-scaling
    ///    convention are not, so `_pcm` cannot be turned into that plane.
    ///
    /// **Resolved, and named as resolved in the message** so nobody
    /// re-reports them: the `vokra.m2d.*` axis group is now stamped in full,
    /// branch selection rides [`GGUF_KEY_INFERENCE_BRANCH`], and the ViT
    /// primitive exists in [`vokra_ops::vit`].
    ///
    /// **No fabricated hidden states are ever emitted** (FR-EX-08 — no silent
    /// partial output).
    ///
    /// `_pcm` is the raw mono f32 waveform in `[-1, 1]`, at the stamped
    /// [`GGUF_KEY_SAMPLE_RATE`]. Nothing here resamples: the stamp is the
    /// artifact's claim about its training rate, not a licence to resample
    /// silently.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn encode(&self, _pcm: &[f32]) -> Result<M2dHiddenStates> {
        Err(forward_loud_partial(
            &self.config,
            self.weights.branch_triage(),
            "encode",
            ENCODE_OUTPUT,
        ))
    }

    /// Encodes a PCM waveform to a single utterance-level pooled embedding.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] with the same three blockers as
    /// [`encode`](Self::encode), plus a fourth specific to this surface: the
    /// **pooling recipe itself is unresolved**. [`vokra_ops::vit::ViTPooling`]
    /// now makes the choice an explicit axis — a prepended token versus the
    /// mean over patch tokens — which sharpens the question rather than
    /// answering it. Which convention M2D was published under, over which
    /// layer, and whether a normalisation follows, is defined by the upstream
    /// inference wrapper and not by anything this wave can read. Applying a
    /// plain mean and calling it "the M2D embedding" would be fabrication.
    ///
    /// **No fabricated embedding is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred pooled-embedding forward.
    pub fn embed(&self, _pcm: &[f32]) -> Result<Vec<f32>> {
        Err(forward_loud_partial(
            &self.config,
            self.weights.branch_triage(),
            "embed",
            EMBED_OUTPUT,
        ))
    }
}

/// Output-contract clause used by [`M2d::encode`]'s loud-partial message.
const ENCODE_OUTPUT: &str = "a [num_frames, hidden_size] block of encoder hidden states (M2D is a feature \
     extractor — the checkpoint carries an encoder, not a task head, so no \
     classification output exists to emit)";

/// Output-contract clause used by [`M2d::embed`]'s loud-partial message.
const EMBED_OUTPUT: &str = "an utterance-level pooled embedding, whose POOLING RECIPE is itself unresolved. \
     `vokra_ops::vit::ViTPooling` now makes the choice an explicit axis — a prepended \
     token versus the mean over patch tokens — which sharpens the question rather than \
     answering it: which convention M2D was published under, over which layer, and \
     whether a normalisation follows, is defined by the upstream inference wrapper and \
     not by anything this wave can read, so applying a plain mean and calling it the \
     M2D embedding would be fabrication";

/// Builds the loud-partial [`VokraError::UnsupportedOp`] shared by
/// [`M2d::encode`] and [`M2d::embed`].
///
/// Names the three remaining blockers by exact identifier, states outright
/// which blockers are **resolved** (so a reader does not re-report them),
/// echoes the axes this artifact actually carries and how its manifest is
/// actually prefixed, and cites all three primary sources — the `wavlm` /
/// `emotion2vec` / `panns` / `canary_qwen` loud-partial-message precedent
/// (CLAUDE.md 教訓 (a)).
fn forward_loud_partial(
    cfg: &M2dConfig,
    triage: M2dBranchTriage,
    surface: &str,
    output: &str,
) -> VokraError {
    let unstamped = UNSTAMPED_VIT_AXES.join(", ");
    VokraError::UnsupportedOp(format!(
        "m2d {surface} (loud-partial): the M2D (Masked Modeling Duo) encoder forward \
         is deferred. Target output: {output}. \
         ALREADY RESOLVED, do not re-report: the converter now stamps the full \
         {total}-key `vokra.m2d.*` axis group and this artifact reads back \
         hidden_size={hidden}, num_hidden_layers={layers}, \
         num_attention_heads={heads}, patch {ph}x{pw} (mel bins x frames), \
         n_mels={n_mels}, sample_rate={sr}, `{branch_key}`={branch}; and \
         `vokra_ops::vit` now supplies the 2-D patch embedding + pre-norm Transformer \
         encoder (`ViTEncoder`) this arch needs, which the 1-D ASR encoders \
         (`vokra_ops::conformer`, `zipformer`, `ebranchformer`) could not stand in \
         for. THREE blockers remain: \
         (1) UNVERIFIED TENSOR-NAME MANIFEST — the converter copies every float tensor \
         under its VERBATIM upstream `state_dict` name and nothing in-repo transcribes \
         M2D's naming, so walking guessed names into typed slots would bind \
         shape-valid garbage. Two sub-questions only a real checkpoint settles: \
         (a) whether the released artifact keeps the `online.` / `target.` duo \
         prefixes at all — upstream `util/to_encoder_only_weight.py` persists \
         `PortableM2D(src).backbone.state_dict()`, i.e. a single UNPREFIXED encoder, so \
         a converted encoder-only export and a converted raw duo checkpoint have \
         different manifests; THIS artifact shows {triage:?} — and (b) whether \
         attention ships one FUSED `qkv` projection or three separate ones, since \
         `vokra_ops::vit::ViTAttnWeights` holds wq/wk/wv separately and a fused \
         tensor's concat order is a convention rather than a transcription. \
         (2) UNSTAMPED ViT AXES — `ViTAttrs` has 12 axes and the stamped group supplies \
         5 (embed_dim, depth, n_heads, patch_h, patch_w); the other 7 — {unstamped} — \
         are carried by NO `vokra.m2d.*` key. The converter's docstring records \
         mlp_ratio=4 and LayerNorm eps=1e-6 from upstream `examples/portable_m2d.py` \
         `get_backbone()` but deliberately does not stamp them, so this binder neither \
         reads nor assumes them: `M2dConfig::vit_attrs` takes them from the caller as \
         an `M2dUnstampedAxes`, and this surface has no caller-supplied axes. \
         (3) NO MEL FRONT-END BINDING — `vokra_ops::vit::ViTEncoder::forward` consumes \
         a [n_mels, n_frames] log-mel plane, not PCM. n_mels and sample_rate are \
         stamped, but n_fft, hop, window, f_min, f_max and the log-scaling convention \
         are not, so the PCM handed to this call cannot be turned into that plane. \
         Nothing here resamples: the stamped rate is the artifact's claim about its \
         training rate, not a licence to resample silently. \
         Primary sources: code {code}, paper {paper}, license {license_pdf} (a PDF \
         GitHub's classifier cannot read, which is why this arch defaults to \
         LicenseClass::Unknown and fails closed at the M2-13 gate). \
         Runtime cannot fabricate an output (FR-EX-08 — no silent partial output).",
        total = TOTAL_STAMPED_AXES,
        hidden = cfg.hidden_size,
        layers = cfg.num_hidden_layers,
        heads = cfg.num_attention_heads,
        ph = cfg.patch_height,
        pw = cfg.patch_width,
        n_mels = cfg.n_mels,
        sr = cfg.sample_rate,
        branch_key = GGUF_KEY_INFERENCE_BRANCH,
        branch = cfg.inference_branch.as_str(),
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
    //!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` and every `GGUF_KEY_*`
    //!    spelling match the converter, and the arch tag is distinct from
    //!    every sibling SSL-encoder tag.
    //! 2. **Required axis group** — a fully stamped GGUF parses into
    //!    `M2dConfig` with each field equal to what was stamped, and dropping
    //!    any single key is a loud `ModelLoad` naming that key.
    //! 3. **ViT mapping** — the config maps onto `ViTAttrs` and validates,
    //!    and an inconsistent axis set is refused.
    //! 4. **Loud negative space** — missing arch, foreign arch, empty tensor
    //!    manifest, missing tensor, dim mismatch, wrong-dtype axis and a
    //!    bogus branch string each fire at their documented surface in their
    //!    documented variant.
    //! 5. **Loud-partial contract** — `encode` / `embed` drop the blockers
    //!    that are now resolved and name the ones that remain.
    //!
    //! # On the fixture axis values
    //!
    //! The numbers below mirror the converter's transcribed constants
    //! (`HIDDEN_SIZE`, … in `crates/vokra-convert/src/models/m2d.rs`;
    //! `vokra-convert`'s `models` module is private, so the file is the
    //! only referent), which carry the
    //! per-axis primary-source citation. Restating them here is a
    //! cross-crate handshake, not an independent claim about M2D.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// The seven u32 axes, with the values the converter stamps.
    const FIXTURE_U32_AXES: [(&str, u32); 7] = [
        (GGUF_KEY_HIDDEN_SIZE, 768),
        (GGUF_KEY_NUM_HIDDEN_LAYERS, 12),
        (GGUF_KEY_NUM_ATTENTION_HEADS, 12),
        (GGUF_KEY_PATCH_HEIGHT, 16),
        (GGUF_KEY_PATCH_WIDTH, 16),
        (GGUF_KEY_N_MELS, 80),
        (GGUF_KEY_SAMPLE_RATE, 16_000),
    ];

    /// Builds a GGUF shaped like the one the M2D converter emits: arch + name
    /// + category + upstream_url + the full required axis group + an optional
    ///   license stamp + two representative duo-branch tensors.
    ///
    /// `omit` drops exactly one axis key, for the missing-key rows. The tensor
    /// names mirror the converter's own test fixtures and are
    /// **illustrative**, not a verified upstream manifest — see
    /// [`BRANCH_PREFIX_ONLINE`].
    fn m2d_gguf_omitting(omit: Option<&str>, weight_license: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        if let Some(cls) = weight_license {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        for (key, value) in FIXTURE_U32_AXES {
            if omit == Some(key) {
                continue;
            }
            b.add_u32(key, value);
        }
        if omit != Some(GGUF_KEY_INFERENCE_BRANCH) {
            b.add_string(GGUF_KEY_INFERENCE_BRANCH, M2dBranch::Online.as_str());
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

    /// The fully stamped fixture.
    fn m2d_gguf(weight_license: Option<LicenseClass>) -> GgufFile {
        m2d_gguf_omitting(None, weight_license)
    }

    /// A ViT axis set for the seven axes no `vokra.m2d.*` key carries.
    ///
    /// These are **test inputs**, not claims about M2D: the point of
    /// [`M2dUnstampedAxes`] is that the caller states them. `mlp_ratio` and
    /// `layer_norm_eps` use the values the converter's docstring transcribes
    /// from upstream `get_backbone()`; the rest are chosen to make a
    /// well-formed encoder and are labelled as such.
    fn unstamped_axes() -> M2dUnstampedAxes {
        M2dUnstampedAxes {
            // Non-overlapping tiling: stride == patch extent.
            stride_h: 16,
            stride_w: 16,
            // Test input — the real count is settled by a checkpoint.
            n_prepended_tokens: 1,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-6,
            gelu: GeluKind::Erf,
            pos_embed_policy: PosEmbedPolicy::RequireExact,
        }
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
    }

    #[test]
    fn stamped_axis_key_spellings_mirror_the_converter() {
        // Byte-for-byte pin against `vokra-convert`'s `KEY_M2D_*` constants.
        // The crates cannot share these (no `vokra-models` -> `vokra-convert`
        // dependency edge), so a rename on either side has to land here or
        // fail this row — otherwise a stamped axis would simply never be read.
        assert_eq!(GGUF_KEY_HIDDEN_SIZE, "vokra.m2d.hidden_size");
        assert_eq!(GGUF_KEY_NUM_HIDDEN_LAYERS, "vokra.m2d.num_hidden_layers");
        assert_eq!(
            GGUF_KEY_NUM_ATTENTION_HEADS,
            "vokra.m2d.num_attention_heads"
        );
        assert_eq!(GGUF_KEY_PATCH_HEIGHT, "vokra.m2d.patch_height");
        assert_eq!(GGUF_KEY_PATCH_WIDTH, "vokra.m2d.patch_width");
        assert_eq!(GGUF_KEY_N_MELS, "vokra.m2d.n_mels");
        assert_eq!(GGUF_KEY_SAMPLE_RATE, "vokra.m2d.sample_rate");
        assert_eq!(GGUF_KEY_INFERENCE_BRANCH, "vokra.m2d.inference_branch");

        let mut keys = STAMPED_AXIS_KEYS.to_vec();
        assert_eq!(
            keys.len(),
            TOTAL_STAMPED_AXES,
            "STAMPED_AXIS_KEYS must hold exactly TOTAL_STAMPED_AXES entries"
        );
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            keys.len(),
            TOTAL_STAMPED_AXES,
            "axis keys must be pairwise distinct"
        );
        for key in STAMPED_AXIS_KEYS {
            assert!(
                key.starts_with("vokra.m2d."),
                "`{key}` must live under the arch's own metadata namespace"
            );
        }
        // The seven caller-supplied axes are named as ViTAttrs fields, not as
        // GGUF keys — precisely because there are no GGUF keys for them.
        for axis in UNSTAMPED_VIT_AXES {
            assert!(
                !axis.starts_with("vokra."),
                "`{axis}` is a ViTAttrs field, not a stamped chunk"
            );
            assert!(
                !STAMPED_AXIS_KEYS.iter().any(|k| k.ends_with(axis)),
                "`{axis}` is listed as unstamped but a stamped key matches it"
            );
        }
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
    // 3 — The required axis group parses, field by field
    // -----------------------------------------------------------------------

    #[test]
    fn stamped_axis_group_parses_into_the_config() {
        let file = m2d_gguf(Some(LicenseClass::Unknown));
        let m = M2d::from_gguf(&file).expect("a fully stamped GGUF must bind");
        let cfg = *m.config();

        // Every field equals the value that was stamped.
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.patch_height, 16);
        assert_eq!(cfg.patch_width, 16);
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(
            cfg.inference_branch,
            M2dBranch::Online,
            "paper §3: after training only f_θ (the online encoder) is transferred"
        );

        // …and the u32 half agrees with the fixture table it was built from,
        // so a fixture edit cannot silently pass by moving both sides.
        for (key, value) in FIXTURE_U32_AXES {
            let read = file
                .get(key)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("{key}: missing or not an unsigned integer"));
            assert_eq!(read, u64::from(value), "{key} round-trip");
        }

        // Branch wire form is exact and case-sensitive.
        assert_eq!(M2dBranch::from_wire("target"), Some(M2dBranch::Target));
        assert_eq!(M2dBranch::from_wire("Online"), None, "wire form is exact");
        assert_eq!(M2dBranch::Online.tensor_prefix(), BRANCH_PREFIX_ONLINE);
        assert_eq!(M2dBranch::Target.tensor_prefix(), BRANCH_PREFIX_TARGET);
    }

    #[test]
    fn every_missing_stamped_key_is_a_loud_model_load_naming_it() {
        for key in STAMPED_AXIS_KEYS {
            let file = m2d_gguf_omitting(Some(key), None);
            let Err(err) = M2d::from_gguf(&file) else {
                panic!("dropping `{key}` must be refused — every axis is REQUIRED");
            };
            match err {
                VokraError::ModelLoad(msg) => {
                    assert!(
                        msg.contains(key),
                        "the message must NAME the absent key `{key}`, got `{msg}`"
                    );
                    assert!(
                        msg.contains("FR-EX-08"),
                        "message must cite FR-EX-08 for `{key}`, got `{msg}`"
                    );
                    assert!(
                        msg.contains("m2d-base"),
                        "message must include the repro command for `{key}`, got `{msg}`"
                    );
                }
                other => panic!("expected VokraError::ModelLoad for `{key}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_missing_axis_is_never_filled_from_a_primary_source_constant() {
        // The sharpest regression this reader exists to prevent: an artifact
        // that omits `sample_rate` must NOT bind as the canonical 16 kHz
        // release. The 32 kHz M2D weight is a separate identity, and a
        // defaulted rate would resample silently (FR-EX-08).
        let file = m2d_gguf_omitting(Some(GGUF_KEY_SAMPLE_RATE), None);
        let Err(err) = M2d::from_gguf(&file) else {
            panic!("a missing sample_rate must be refused, never defaulted to 16 kHz");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(GGUF_KEY_SAMPLE_RATE),
                    "must name the key: {msg}"
                );
                assert!(
                    msg.contains("32 kHz"),
                    "message should explain WHY defaulting is unsafe here: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4 — Config -> ViTAttrs mapping
    // -----------------------------------------------------------------------

    #[test]
    fn config_maps_onto_vit_attrs_that_validate() {
        let file = m2d_gguf(None);
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");
        let unstamped = unstamped_axes();
        let attrs = m
            .config()
            .vit_attrs(&unstamped)
            .expect("the mapped axis set must validate");

        // Whole-struct equality rather than field-by-field: a mapping that
        // forgot to copy an axis, or crossed patch_h with patch_w, would slip
        // past a partial check but not past this one.
        let expected = ViTAttrs {
            // <- vokra.m2d.hidden_size / num_hidden_layers /
            //    num_attention_heads / patch_height / patch_width
            embed_dim: 768,
            depth: 12,
            n_heads: 12,
            patch_h: 16,
            patch_w: 16,
            // <- the caller's M2dUnstampedAxes, copied through unchanged
            stride_h: unstamped.stride_h,
            stride_w: unstamped.stride_w,
            n_prepended_tokens: unstamped.n_prepended_tokens,
            mlp_ratio: unstamped.mlp_ratio,
            layer_norm_eps: unstamped.layer_norm_eps,
            gelu: unstamped.gelu,
            pos_embed_policy: unstamped.pos_embed_policy,
        };
        assert_eq!(attrs, expected, "config -> ViTAttrs mapping");

        // The stamped half must really have come from the GGUF, not from the
        // expectation above: perturb one key and the mapped axis must follow.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        for (key, value) in FIXTURE_U32_AXES {
            // A 384-wide, 6-deep variant: still consistent (384 % 12 == 0).
            let value = match key {
                GGUF_KEY_HIDDEN_SIZE => 384,
                GGUF_KEY_NUM_HIDDEN_LAYERS => 6,
                _ => value,
            };
            b.add_u32(key, value);
        }
        b.add_string(GGUF_KEY_INFERENCE_BRANCH, M2dBranch::Target.as_str());
        b.add_tensor("probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let other = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let m_other = M2d::from_gguf(&other).expect("valid GGUF must bind");
        let attrs_other = m_other
            .config()
            .vit_attrs(&unstamped)
            .expect("384/12 is consistent");
        assert_eq!(attrs_other.embed_dim, 384, "embed_dim tracks the stamp");
        assert_eq!(attrs_other.depth, 6, "depth tracks the stamp");
        assert_eq!(
            m_other.config().inference_branch,
            M2dBranch::Target,
            "the branch selector tracks the stamp too"
        );

        // Independently re-validating must also pass, and the derived widths
        // must be consistent with the mapped axes.
        attrs.validate().expect("ViTAttrs::validate");
        assert_eq!(attrs.head_dim(), 64, "768 / 12");
        assert_eq!(attrs.mlp_dim(), 3072, "768 * 4.0");
    }

    #[test]
    fn vit_attrs_mapping_refuses_an_inconsistent_axis_set() {
        // A head count that does not divide the width must be refused by
        // `ViTAttrs::validate` through the mapping, rather than silently
        // flooring the per-head width.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, 768);
        b.add_u32(GGUF_KEY_NUM_HIDDEN_LAYERS, 12);
        b.add_u32(GGUF_KEY_NUM_ATTENTION_HEADS, 7); // 768 % 7 != 0
        b.add_u32(GGUF_KEY_PATCH_HEIGHT, 16);
        b.add_u32(GGUF_KEY_PATCH_WIDTH, 16);
        b.add_u32(GGUF_KEY_N_MELS, 80);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16_000);
        b.add_string(GGUF_KEY_INFERENCE_BRANCH, M2dBranch::Online.as_str());
        b.add_tensor("online.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let m = M2d::from_gguf(&file).expect("the axis group is complete, so it binds");
        let Err(err) = m.config().vit_attrs(&unstamped_axes()) else {
            panic!("7 heads over a 768-wide model must be refused");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );

        // A zero-rounding mlp_ratio is refused through the same seam.
        let mut degenerate = unstamped_axes();
        degenerate.mlp_ratio = 0.0;
        let file2 = m2d_gguf(None);
        let m2 = M2d::from_gguf(&file2).expect("valid GGUF must bind");
        let Err(err2) = m2.config().vit_attrs(&degenerate) else {
            panic!("a non-positive mlp_ratio must be refused");
        };
        assert!(
            matches!(err2, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err2:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 5 — Metadata round-trip on a converter-shaped GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        let file = m2d_gguf(Some(LicenseClass::Unknown));
        let m = M2d::from_gguf(&file).expect("converter-shaped GGUF must bind");

        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_url(), Some(UPSTREAM_URL));
        assert_eq!(m.tensor_count(), 2);

        // Fail-closed licensing: Unknown demands the research flag.
        assert_eq!(m.weight_license(), LicenseClass::Unknown);
        assert!(m.requires_research_flag());

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
    // 6 — Arch metadata absent / foreign / empty manifest
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

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        // Fully stamped metadata, zero tensors: the axis group passes and the
        // manifest gate is what fires, so this row proves the gate is reached.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        for (key, value) in FIXTURE_U32_AXES {
            b.add_u32(key, value);
        }
        b.add_string(GGUF_KEY_INFERENCE_BRANCH, M2dBranch::Online.as_str());
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
    // 7 — Missing tensor names itself; dim mismatch names both shapes
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

    #[test]
    fn branch_triage_counts_what_the_manifest_actually_shows() {
        let file = m2d_gguf(None);
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");
        let triage = m.weights().branch_triage();
        assert_eq!(triage.online_prefixed, 1);
        assert_eq!(triage.target_prefixed, 1);
        assert_eq!(triage.unprefixed, 0);
        assert_eq!(
            triage.total(),
            m.tensor_count(),
            "the counters must partition"
        );
        assert_eq!(
            m.weights().branch_tensor_count(BRANCH_PREFIX_ONLINE),
            triage.online_prefixed
        );
    }

    // -----------------------------------------------------------------------
    // 8 — Wrong dtype / bogus branch string
    // -----------------------------------------------------------------------

    #[test]
    fn axis_present_with_wrong_dtype_fails_loud() {
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
                    "message must name the dtype problem — and NOT report the key as \
                     missing, which would be a different bug, got `{m}`"
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
        for (key, value) in FIXTURE_U32_AXES {
            b.add_u32(key, value);
        }
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

    // -----------------------------------------------------------------------
    // 9 — Loud-partial: the resolved blockers are GONE, the real ones remain
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partial_drops_resolved_blockers_and_names_what_remains() {
        let file = m2d_gguf(Some(LicenseClass::Unknown));
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");

        // 1 s of legitimately shaped mono PCM, so the loud-partial gate is
        // what fires — not some pre-encode length validation.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        let VokraError::UnsupportedOp(msg) = err else {
            panic!("expected VokraError::UnsupportedOp");
        };

        assert!(msg.contains("m2d encode"), "surface must be named: {msg}");
        assert!(msg.contains("loud-partial"), "posture label: {msg}");

        // --- the blockers that are now RESOLVED must not be claimed ---------
        //
        // This is the point of the row. The converter stamps the whole
        // `vokra.m2d.*` group and `vokra_ops::vit` exists, so a message still
        // asserting either gap would actively mislead the next reader.
        assert!(
            !msg.contains("NO TOPOLOGY AXES"),
            "the axis group IS stamped now — this blocker must be gone: {msg}"
        );
        assert!(
            !msg.contains("stamps NONE"),
            "the converter no longer stamps none of the group: {msg}"
        );
        assert!(
            !msg.contains("MISSING PRIMITIVE"),
            "vokra_ops::vit supplies the ViT encoder now: {msg}"
        );
        assert!(
            !msg.contains("NO BRANCH SELECTION"),
            "the branch selector IS stamped now: {msg}"
        );
        assert!(
            !msg.contains("no ViT-style Transformer encoder"),
            "the primitive exists — that claim must be gone: {msg}"
        );

        // --- and the message must say so positively, not just omit them -----
        assert!(
            msg.contains("ALREADY RESOLVED"),
            "the message should tell the reader what NOT to re-report: {msg}"
        );
        assert!(
            msg.contains("vokra_ops::vit"),
            "the message should name the primitive that now exists: {msg}"
        );
        // The axis values this artifact actually carries are echoed, so the
        // reader can see the group really is populated.
        for fragment in ["hidden_size=768", "num_hidden_layers=12", "n_mels=80"] {
            assert!(msg.contains(fragment), "must echo `{fragment}`: {msg}");
        }
        assert!(
            msg.contains(GGUF_KEY_INFERENCE_BRANCH) && msg.contains("=online"),
            "must report the RESOLVED branch selector, not `UNSET`: {msg}"
        );
        assert!(!msg.contains("UNSET"), "nothing is unset any more: {msg}");

        // --- what actually remains ------------------------------------------
        assert!(
            msg.contains("UNVERIFIED TENSOR-NAME MANIFEST"),
            "the real remaining blocker must be named: {msg}"
        );
        assert!(
            msg.contains("to_encoder_only_weight.py"),
            "the manifest blocker must name the concrete ambiguity a checkpoint \
             settles: {msg}"
        );
        assert!(
            msg.contains("UNSTAMPED ViT AXES") && msg.contains("pos_embed_policy"),
            "the caller-supplied axes must be enumerated: {msg}"
        );
        assert!(
            msg.contains("NO MEL FRONT-END BINDING"),
            "the front-end blocker must be named: {msg}"
        );
        // The observed manifest shape is reported, not asserted.
        assert!(
            msg.contains("online_prefixed: 1"),
            "the message must report the OBSERVED manifest shape: {msg}"
        );

        // Honest about what the encoder does and does not carry.
        assert!(
            msg.contains("feature extractor"),
            "message must state M2D is a feature extractor, not a task model: {msg}"
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

    #[test]
    fn embed_loud_partial_still_calls_out_the_unresolved_pooling() {
        let file = m2d_gguf(None);
        let m = M2d::from_gguf(&file).expect("valid GGUF must bind");
        let pcm = vec![0.0_f32; 16_000];

        let Err(err) = m.embed(&pcm) else {
            panic!("embed must loud-partial");
        };
        let VokraError::UnsupportedOp(msg) = err else {
            panic!("expected VokraError::UnsupportedOp");
        };

        assert!(msg.contains("m2d embed"), "surface must be named: {msg}");
        assert!(
            msg.contains("POOLING RECIPE"),
            "embed must call out the unresolved pooling recipe: {msg}"
        );
        assert!(
            msg.contains("ViTPooling"),
            "the pooling clause should name the axis the primitive now exposes, \
             since that sharpens the question rather than answering it: {msg}"
        );
        assert!(
            msg.contains("UNVERIFIED TENSOR-NAME MANIFEST"),
            "embed shares the encoder blockers: {msg}"
        );
        assert!(
            !msg.contains("MISSING PRIMITIVE"),
            "the ViT primitive exists — embed must not claim otherwise: {msg}"
        );
        assert!(
            msg.contains(PRIMARY_SOURCE_PAPER),
            "primary source must be cited: {msg}"
        );
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
