//! **SpeechBrain spoken-language identification on an ECAPA-TDNN trunk** —
//! runtime binder for the `lang_id_ecapa` converter arch (Wave G 2026-08-15,
//! loud-partial per the `maest` / `emotion2vec` / `atst` / `m2d` / `wavlm` /
//! `panns` / `redimnet` precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/speechbrain_lang_id.rs` has been stamping
//! `vokra.model.arch = "lang_id_ecapa"` since the TIER 1 F wave (2026-07-30),
//! but a workspace-wide grep proved that **nothing read that arch string
//! back** — a converted SpeechBrain lang-ID checkpoint was unloadable. This
//! module is that consumer.
//!
//! # Primary sources
//!
//! Every fact below is transcribed from the converter's own module docstring
//! (`crates/vokra-convert/src/models/speechbrain_lang_id.rs`) and its
//! `ModelKind` entries in `crates/vokra-convert/src/lib.rs`, which together
//! are this repository's primary-source record for the family. Nothing here is
//! re-derived from memory (CLAUDE.md「ハルシネーション厳禁」).
//!
//! - Upstream release (F7, canonical):
//!   <https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa>
//! - Upstream release (F9, sibling):
//!   <https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa>
//! - Paper: Valk & Alumäe 2021, *"VoxLingua107: a Dataset for Spoken Language
//!   Recognition"* — <https://arxiv.org/abs/2011.12998>
//! - Reference code / licence: the SpeechBrain umbrella repository,
//!   <https://github.com/speechbrain/speechbrain> (family licence
//!   `apache-2.0`, recorded by the converter as
//!   [`LicenseClass::Permissive`]).
//!
//! # The two variants share one arch — deliberately
//!
//! The converter stamps a single [`ARCH`] for both upstream releases because
//! the topology **is** shared: an ECAPA-TDNN backbone (SE-Res2Blocks +
//! attentive statistics pooling → embedding) followed by a language
//! classification head. The only variant-carrying axis is the target
//! vocabulary size, which is a shape-derivable head width rather than a
//! topology change. The variant is therefore recovered from
//! `vokra.model.name` via [`LangIdVariant::from_model_name`], not from the
//! arch tag.
//!
//! # The language inventory is read off the artifact, never hardcoded
//!
//! This module contains **no language-taxonomy constant of any kind** — no
//! class count, no ISO-code table (the `maest` precedent, which carries no
//! Discogs taxonomy constant either). That is a consequence of what the
//! converter actually writes, and the boundary is worth stating plainly:
//!
//! - The converter stamps **no language list and no language count**. Its
//!   metadata block is exactly `vokra.model.arch`, `vokra.model.name`,
//!   `vokra.model.category`, the `vokra.provenance.*` group and
//!   `vokra.provenance.upstream_hf`. There is no `vokra.lang_id.*` axis
//!   group.
//! - The language **names are therefore unavailable from the artifact.** A
//!   caller holding only a converted GGUF cannot map a logit index to an ISO
//!   639 code. Recovering the names requires the upstream `hyperparams.yaml`
//!   label encoder, which is outside this GGUF.
//! - The language **count** is the one inventory fact that can be recovered,
//!   and only when the payload makes it unambiguous:
//!   [`LangIdWeights::language_count_from_disk`] reads it off the classifier
//!   head projection's leading dimension. It returns `None` rather than
//!   guessing whenever the layout is ambiguous, and it never falls back to a
//!   constant.
//!
//! The upstream model cards describe F7 as a 107-language identifier and F9 as
//! a CommonLanguage-trained variant of roughly 45 languages. Those figures are
//! *upstream documentation*, not stamped axes, so they appear here only as
//! prose — deliberately not as `pub const`s, because a constant would invite
//! validating a payload against a number this repository cannot verify from
//! the artifact.
//!
//! # Architecture (as recorded by the converter — the forward is loud-partial)
//!
//! ```text
//! PCM (mono f32)
//!   -> filterbank front-end                       ← **loud-partial**
//!        (`vokra_ops::kaldi_fbank` exists in-repo, but its opts
//!         — n_mels, sample rate, frame/hop — are NOT stamped by
//!         this converter, so they cannot be derived from the artifact.)
//!   -> ECAPA-TDNN trunk: SE-Res2Blocks + multi-layer
//!      feature aggregation                        ← **loud-partial**
//!   -> attentive statistics pooling -> embedding  ← **loud-partial**
//!   -> language classifier head                   ← **loud-partial**
//!   -> one logit per language (width read off disk when unambiguous,
//!      see `LangIdEcapa::language_count`; the names are unavailable)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`LangIdEcapa::from_gguf`] with strict `vokra.model.arch ==
//!     "lang_id_ecapa"` validation, refusing every sibling arch loudly by
//!     name (FR-EX-08).
//!   - [`LangIdWeights::from_gguf`] with a zero-tensor refusal **and** a
//!     trunk-presence gate on [`TRUNK_PREFIX`], so a payload missing the
//!     ECAPA backbone fails at load rather than part-way through a forward.
//!   - Variant recovery from `vokra.model.name`
//!     ([`LangIdVariant::from_model_name`]) and upstream-slug surfacing.
//!   - Disk-truthful head reporting: [`LangIdWeights::language_head_tensors`]
//!     and [`LangIdWeights::language_count_from_disk`].
//!   - Weight-licence-class surfacing, fail-closed to
//!     [`LicenseClass::Unknown`] when the stamp is absent.
//!
//! - **Loud-partial (this WP)**: [`LangIdEcapa::identify`] returns
//!   [`VokraError::UnsupportedOp`] naming the deferred primitives (the
//!   filterbank front-end whose axes are unstamped, the SE-Res2Block trunk,
//!   attentive statistics pooling, and the classifier head) and citing all
//!   four primary sources. **No fabricated language logits are ever emitted**
//!   (FR-EX-08 — no silent partial output).
//!
//! # No ECAPA trunk binder exists to reuse
//!
//! `crates/vokra-convert/src/models/ecapa_tdnn.rs` (the
//! `speechbrain/spkrec-ecapa-voxceleb` speaker encoder) is a **converter-side**
//! sibling that shares this trunk and the same `embedding_model.` naming
//! convention. A workspace grep confirms that no runtime binder verifies
//! `vokra.model.arch == "ecapa_tdnn"` — that arch appears in `vokra-models`
//! only inside sibling-distinctness error messages. There is therefore no
//! ECAPA trunk implementation to share today. When one lands, both binders
//! should consume it: the trunk is genuinely common, and only the head differs
//! (192-d speaker embedding vs an N-way language logit vector).
//!
//! # Cross-crate constant duplication
//!
//! The [`ARCH`] / name / category / upstream constants below mirror the
//! converter's — the same rule every sibling BF16 pass-through binder uses so
//! `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layering `vokra-ops → nothing GGUF-aware`, `vokra-core →
//! GGUF reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! SpeechBrain ships PyTorch checkpoints; this runtime **never** touches ONNX
//! or pickle (FR-LD-05 / NFR-DS-02).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/speechbrain_lang_id.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model lang-id-voxlingua107` and
/// `--model lang-id-commonlanguage`.
///
/// Shared by both upstream variants because the topology is shared; the
/// variant is recovered from `vokra.model.name` instead (see
/// [`LangIdVariant::from_model_name`]).
///
/// Deliberately **distinct** from `ecapa_tdnn`, the SpeechBrain
/// speaker-encoder arch that uses the very same ECAPA-TDNN trunk. Backbone
/// identity is not topology identity: `ecapa_tdnn` terminates in a 192-d
/// speaker embedding, this arch terminates in an N-way language classifier.
/// Silently aliasing them would misroute runtime dispatch (FR-EX-08).
pub const ARCH: &str = "lang_id_ecapa";

/// Expected `vokra.model.name` for the F7 (VoxLingua107) variant.
pub const NAME_VOXLINGUA107: &str = "lang-id-voxlingua107-ecapa";

/// Expected `vokra.model.name` for the F9 (CommonLanguage) variant.
pub const NAME_COMMONLANGUAGE: &str = "lang-id-commonlanguage-ecapa";

/// Expected `vokra.model.category` value — language identification is a fixed
/// N-way classifier, so the converter records it as `classification`.
pub const CATEGORY: &str = "classification";

/// Upstream HuggingFace slug for the F7 (VoxLingua107) variant.
pub const UPSTREAM_HF_VOXLINGUA107: &str = "speechbrain/lang-id-voxlingua107-ecapa";

/// Upstream HuggingFace slug for the F9 (CommonLanguage) variant.
pub const UPSTREAM_HF_COMMONLANGUAGE: &str = "speechbrain/lang-id-commonlanguage_ecapa";

/// Metadata key holding the model category (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key holding the upstream HuggingFace slug (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// `state_dict` name prefix under which the ECAPA-TDNN trunk sits.
///
/// **Transcribed, not assumed.** Both SpeechBrain converters in this
/// repository record this prefix in their own fixtures: the lang-ID converter
/// uses [`TRUNK_EXAMPLE_TENSOR`] in both of its round-trip tests, and the
/// sibling `ecapa_tdnn.rs` converter uses the same name plus
/// `embedding_model.a.weight` / `embedding_model.b.weight`. It reflects the
/// SpeechBrain recipe convention of naming the backbone module
/// `embedding_model` in `hyperparams.yaml`.
///
/// The trunk is **required**: an artifact carrying no tensor under this prefix
/// is not a lang-ID checkpoint and is refused at load (FR-EX-08).
pub const TRUNK_PREFIX: &str = "embedding_model.";

/// A representative trunk tensor name, quoted in load-failure diagnostics so a
/// reader can see the exact shape of the name the binder is looking for.
///
/// Transcribed from the converter's own test fixtures (both the lang-ID
/// converter and the sibling `ecapa_tdnn.rs` converter build their safetensors
/// fixtures around this name).
pub const TRUNK_EXAMPLE_TENSOR: &str = "embedding_model.blocks.0.tdnn.conv.weight";

// Primary-source anchors, cited inside the loud-partial error so a reader
// diagnosing the gap has fully specified places to walk.

/// Primary-source anchor: the F7 (VoxLingua107) HuggingFace release.
pub const PRIMARY_SOURCE_HF_VOXLINGUA107: &str =
    "huggingface.co/speechbrain/lang-id-voxlingua107-ecapa";

/// Primary-source anchor: the F9 (CommonLanguage) HuggingFace release.
pub const PRIMARY_SOURCE_HF_COMMONLANGUAGE: &str =
    "huggingface.co/speechbrain/lang-id-commonlanguage_ecapa";

/// Primary-source anchor: Valk & Alumäe 2021 — the VoxLingua107 paper.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2011.12998";

/// Primary-source anchor: the SpeechBrain reference implementation and the
/// family licence (`apache-2.0`).
pub const PRIMARY_SOURCE_CODE: &str = "github.com/speechbrain/speechbrain";

// ---------------------------------------------------------------------------
// LangIdVariant — which upstream release this artifact came from.
// ---------------------------------------------------------------------------

/// Which upstream SpeechBrain lang-ID release an artifact came from.
///
/// Recovered from `vokra.model.name`, **not** from [`ARCH`] — both variants
/// share one arch tag because they share one topology (see the module
/// docstring). The variant is the only stamped datum that distinguishes the
/// two language inventories, since the converter stamps neither a language
/// list nor a language count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangIdVariant {
    /// F7: `speechbrain/lang-id-voxlingua107-ecapa` — the canonical variant,
    /// trained on VoxLingua107.
    VoxLingua107,
    /// F9: `speechbrain/lang-id-commonlanguage_ecapa` — the sibling variant,
    /// trained on the CommonLanguage dataset.
    CommonLanguage,
}

impl LangIdVariant {
    /// Recovers the variant from a stamped `vokra.model.name` value.
    ///
    /// Returns `None` for any other string. That is deliberately **not** an
    /// error: `vokra.model.name` is descriptive metadata while
    /// `vokra.model.arch` is the load gate, and a legitimate re-export may
    /// carry a different name (a third SpeechBrain lang-ID variant, or a
    /// downstream fine-tune, would still be topologically bindable here).
    /// Callers who need the raw stamp regardless get it from
    /// [`LangIdEcapa::model_name`], so nothing is hidden by the `None`.
    #[must_use]
    pub fn from_model_name(name: &str) -> Option<Self> {
        match name {
            NAME_VOXLINGUA107 => Some(Self::VoxLingua107),
            NAME_COMMONLANGUAGE => Some(Self::CommonLanguage),
            _ => None,
        }
    }

    /// The canonical `vokra.model.name` value for this variant.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::VoxLingua107 => NAME_VOXLINGUA107,
            Self::CommonLanguage => NAME_COMMONLANGUAGE,
        }
    }

    /// The upstream HuggingFace slug for this variant.
    #[inline]
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::VoxLingua107 => UPSTREAM_HF_VOXLINGUA107,
            Self::CommonLanguage => UPSTREAM_HF_COMMONLANGUAGE,
        }
    }
}

// ---------------------------------------------------------------------------
// LangIdWeights — the disk manifest plus its load gates.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a SpeechBrain lang-ID GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step.
/// A GGUF that carries zero tensors, or one that carries no ECAPA trunk under
/// [`TRUNK_PREFIX`], is refused rather than silently running a partial forward
/// (FR-EX-08).
///
/// Under the current landing this struct stores the tensor names plus their
/// GGUF-side dims. The follow-up wave that implements the trunk sizes its
/// dequant per its kernel needs; today the manifest backs the load gates and
/// the disk-truthful head reporting.
#[derive(Debug, Clone)]
pub struct LangIdWeights {
    /// Tensors discovered on disk, in file order, as
    /// `(upstream state_dict name, GGUF-side dims)`.
    tensors: Vec<(String, Vec<usize>)>,
}

impl LangIdWeights {
    /// Scans `gguf` for the SpeechBrain lang-ID `state_dict` tensors.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   (refusing to bind an all-zero forward).
    /// - [`VokraError::ModelLoad`] when no tensor sits under
    ///   [`TRUNK_PREFIX`] — the ECAPA-TDNN backbone is not optional, so its
    ///   absence means the artifact is not a lang-ID checkpoint at all.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tensors: Vec<(String, Vec<usize>)> = gguf
            .tensors()
            .iter()
            .map(|info| {
                let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                (info.name.clone(), dims)
            })
            .collect();

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate SpeechBrain lang-ID checkpoint is an \
                 ECAPA-TDNN backbone (SE-Res2Blocks + attentive statistics pooling) plus a \
                 language classifier head (arch={ARCH}), which always converts to many \
                 Conv1D / BatchNorm / Linear tensors, so an empty manifest always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model lang-id-voxlingua107` \
                 (or `--model lang-id-commonlanguage`) against an upstream \
                 `{UPSTREAM_HF_VOXLINGUA107}` / `{UPSTREAM_HF_COMMONLANGUAGE}` safetensors \
                 checkpoint."
            )));
        }

        let trunk = tensors
            .iter()
            .filter(|(n, _)| n.starts_with(TRUNK_PREFIX))
            .count();
        if trunk == 0 {
            let found = tensors.len();
            let sample: Vec<&str> = tensors.iter().take(5).map(|(n, _)| n.as_str()).collect();
            let sample = sample.join(", ");
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: GGUF carries {found} tensor(s) but none under the required \
                 ECAPA-TDNN trunk prefix `{TRUNK_PREFIX}` (expected names shaped like \
                 `{TRUNK_EXAMPLE_TENSOR}`). The backbone is not optional — a lang-ID \
                 checkpoint without it cannot produce an embedding to classify, so binding \
                 would guarantee a part-way failure later instead of a clear one now \
                 (FR-EX-08 — no silent partial load). First tensor names on disk: \
                 [{sample}]. Was this GGUF produced by \
                 `vokra-cli convert --model lang-id-voxlingua107`? Primary source: \
                 {PRIMARY_SOURCE_HF_VOXLINGUA107}"
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

    /// The upstream `state_dict` names discovered on disk, in file order.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// GGUF-side dims of `name`, or `None` when the tensor is absent.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// How many tensors have a name starting with `prefix`.
    ///
    /// A plain string count over what is actually on disk — it asserts nothing
    /// about the upstream naming convention.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
    }

    /// The ECAPA-TDNN trunk tensors on disk (those under [`TRUNK_PREFIX`]), as
    /// `(name, dims)` pairs in file order.
    #[must_use]
    pub fn trunk_tensors(&self) -> Vec<(&str, &[usize])> {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(TRUNK_PREFIX))
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect()
    }

    /// The language-head tensors on disk — every tensor **not** under
    /// [`TRUNK_PREFIX`] — as `(name, dims)` pairs in file order.
    ///
    /// The partition is deliberately defined by exclusion rather than by a
    /// head prefix constant. This repository has transcribed the trunk prefix
    /// from two converter fixtures, but it has **not** transcribed the head's
    /// module name from any primary source, and inventing one would be exactly
    /// the fabrication CLAUDE.md forbids. Excluding the trunk needs no such
    /// guess and still isolates the head on a well-formed artifact.
    ///
    /// Pure disk reporting with no interpretation: an empty result means the
    /// artifact is a bare-encoder export, which is a legitimate artifact — it
    /// simply has no language head to read a width from.
    #[must_use]
    pub fn language_head_tensors(&self) -> Vec<(&str, &[usize])> {
        self.tensors
            .iter()
            .filter(|(n, _)| !n.starts_with(TRUNK_PREFIX))
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect()
    }

    /// The language-set size **read off the artifact**, or `None`.
    ///
    /// Returns `Some(dims[0])` when exactly one tensor outside
    /// [`TRUNK_PREFIX`] is 2-D. A PyTorch `nn.Linear` weight is
    /// `[out_features, in_features]`, and the converter passes the safetensors
    /// shape through verbatim (`t.shape.clone()` straight into
    /// `GgufBuilder::add_tensor`), so that leading dimension *is* the number
    /// of languages in whatever checkpoint the caller converted.
    ///
    /// Returns `None` in every other case — no head on disk, several 2-D
    /// candidates, or a head whose projection is not 2-D (SpeechBrain's
    /// cosine-similarity classifier variant stores a rank-3 parameter, whose
    /// axis order this repository has not transcribed from a primary source).
    /// It never falls back to a constant, and this module holds no language
    /// count to fall back to: reporting `None` is how an ambiguous payload is
    /// distinguished from a known one.
    ///
    /// The language **names** are not recoverable at all — see the module
    /// docstring.
    #[must_use]
    pub fn language_count_from_disk(&self) -> Option<usize> {
        let mut two_d = self
            .language_head_tensors()
            .into_iter()
            .filter(|(_, dims)| dims.len() == 2);
        let first = two_d.next()?;
        if two_d.next().is_some() {
            // More than one 2-D tensor outside the trunk: which one is the
            // language projection is ambiguous, so report nothing rather than
            // pick.
            return None;
        }
        first.1.first().copied()
    }
}

// ---------------------------------------------------------------------------
// LangIdEcapa — the runtime binder handle.
// ---------------------------------------------------------------------------

/// SpeechBrain spoken-language identification on an ECAPA-TDNN trunk
/// (`speechbrain/lang-id-voxlingua107-ecapa` and
/// `speechbrain/lang-id-commonlanguage_ecapa`, `apache-2.0`) — runtime binder.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`identify`](Self::identify) on a mono f32 PCM waveform to obtain one logit
/// per language. See the module doc for the implementation-status matrix and
/// the FR-EX-08 loud-error contract on the deferred trunk / pooling / head
/// composition.
#[derive(Debug)]
pub struct LangIdEcapa {
    weights: LangIdWeights,
    model_name: Option<String>,
    variant: Option<LangIdVariant>,
    upstream_hf: Option<String>,
    weight_license: LicenseClass,
}

impl LangIdEcapa {
    /// Binds a SpeechBrain lang-ID GGUF: validates arch strictly, discovers
    /// the tensor manifest behind its load gates, recovers the upstream
    /// variant, and surfaces the stamped weight-licence class for the
    /// compliance-gate cross-checks.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key, so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not
    ///   [`ARCH`] — the message names both the found and the expected tag.
    /// - [`VokraError::ModelLoad`] from [`LangIdWeights::from_gguf`] when the
    ///   GGUF carries zero tensors or no ECAPA trunk.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here fails
        //    with a specific message instead of a downstream missing-tensor
        //    error.
        verify_arch(file)?;

        // 2. Tensor manifest behind the zero-tensor + trunk-presence gates.
        let weights = LangIdWeights::from_gguf(file)?;

        // 3. Variant recovery from the descriptive name stamp. A name this
        //    module does not recognise is not an error (see
        //    `LangIdVariant::from_model_name`); the raw stamp stays available
        //    through `model_name()`.
        let model_name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let variant = model_name
            .as_deref()
            .and_then(LangIdVariant::from_model_name);

        let upstream_hf = file
            .get(GGUF_KEY_PROVENANCE_UPSTREAM_HF)
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        // 4. Provenance surfacing. The converter stamps `Permissive`
        //    (`apache-2.0` per the SpeechBrain family licence); a GGUF missing
        //    the stamp reads back as `Unknown` (fail-closed default per
        //    `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            weights,
            model_name,
            variant,
            upstream_hf,
            weight_license,
        })
    }

    /// The recovered upstream variant, or `None` when `vokra.model.name` is
    /// absent or carries a value this module does not recognise.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> Option<LangIdVariant> {
        self.variant
    }

    /// The raw `vokra.model.name` stamp, whether or not it mapped to a known
    /// [`LangIdVariant`].
    #[inline]
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_hf` slug, or `None`.
    #[inline]
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }

    /// The stamped weight-licence class from
    /// `vokra.provenance.weight_license`. The converter stamps `Permissive`
    /// (`apache-2.0`); a GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed at the M2-13 compliance gate).
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

    /// The bound weight manifest, for callers that want the disk-truthful
    /// tensor listing behind the accessors below.
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &LangIdWeights {
        &self.weights
    }

    /// Whether the artifact carries any tensor outside [`TRUNK_PREFIX`], i.e.
    /// whether it looks like a full lang-ID export rather than a bare-encoder
    /// one.
    #[must_use]
    pub fn has_language_head(&self) -> bool {
        !self.weights.language_head_tensors().is_empty()
    }

    /// The language-set size **read off this artifact**, or `None`.
    ///
    /// Delegates to [`LangIdWeights::language_count_from_disk`]: the value
    /// comes from the payload, never from a constant — this module holds no
    /// language-taxonomy constant at all. The language **names** are not
    /// recoverable from a converted GGUF (see the module docstring).
    #[inline]
    #[must_use]
    pub fn language_count(&self) -> Option<usize> {
        self.weights.language_count_from_disk()
    }

    /// Identifies the spoken language of a PCM waveform, returning one logit
    /// per language.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The full forward needs four
    /// pieces this artifact cannot supply: the filterbank front-end (whose
    /// axes the converter does not stamp — there is no `vokra.lang_id.*`
    /// group), the ECAPA-TDNN SE-Res2Block trunk, attentive statistics
    /// pooling, and the language classifier head. **No fabricated language
    /// logits are ever emitted** (FR-EX-08 — no silent partial output).
    ///
    /// The `_pcm` argument is treated as a raw mono f32 waveform; a rate or
    /// shape mismatch will be a loud error rather than a resample surprise
    /// when the real forward lands.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred front-end / trunk / pooling / head composition.
    pub fn identify(&self, _pcm: &[f32]) -> Result<Vec<f32>> {
        // Bind explicitly so an unused-variable warning cannot mask a future
        // accidental removal of the parameter (mirror of the emotion2vec /
        // panns loud-partial signature discipline).
        let _ = _pcm;
        Err(identify_loud_partial(self.language_count()))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Strict `vokra.model.arch` verification.
///
/// Refuses a foreign GGUF loudly, naming **both** the found and the expected
/// tag and enumerating the speaker-embedding / audio-classifier neighbourhood,
/// so a reader who handed the wrong artifact over knows immediately which
/// loader they wanted (FR-EX-08 — never a silent misroute).
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced \
             by `vokra-cli convert --model lang-id-voxlingua107` or \
             `--model lang-id-commonlanguage`?). The confusable neighbours: `ecapa_tdnn` \
             (`speechbrain/spkrec-ecapa-voxceleb`) is the closest — it shares this exact \
             ECAPA-TDNN trunk and the same `embedding_model.` naming, but terminates in a \
             192-d speaker embedding rather than an N-way language classifier, so backbone \
             identity is NOT topology identity here; `campplus` (D-TDNN with context-aware \
             masking) and `speaker_3d` (3D-Speaker ERes2Net) and `redimnet` reach the same \
             speaker-embedding surface through different backbones; `wavlm_sv` bolts an \
             XVector speaker head onto a WavLM SSL encoder; and `emotion2vec` is a fixed \
             N-way utterance classifier like this one but over emotion classes on a \
             wav2vec2-lineage raw-waveform stem, not over languages on a filterbank stem. \
             Binding any of them here would walk a foreign topology over a lang-ID payload \
             (FR-EX-08 — no silent partial load). Primary source: \
             {PRIMARY_SOURCE_HF_VOXLINGUA107}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: GGUF is missing `vokra.model.arch` — this is not a Vokra-native \
             lang_id_ecapa GGUF (was it produced by \
             `vokra-cli convert --model lang-id-voxlingua107`?). Refusing to guess the arch \
             from the tensor manifest, which would be especially unsafe here because the \
             sibling `ecapa_tdnn` speaker encoder carries a nearly identical trunk manifest \
             (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF_VOXLINGUA107}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`LangIdEcapa::identify`].
///
/// Names each deferred primitive explicitly, states the front-end blocker in
/// terms of what is *not* stamped, reports the output width honestly (read off
/// disk when unambiguous, otherwise declared unavailable), and cites all four
/// primary sources so a reader diagnosing the gap has fully specified places to
/// walk.
fn identify_loud_partial(language_count: Option<usize>) -> VokraError {
    // Bound to `let`s rather than nested inside the outer `format!` args
    // (clippy: no `format!` inside another `format!`'s arguments).
    let width = match language_count {
        Some(n) => {
            format!("{n} (read off the classifier head projection on disk, not from any constant)")
        }
        None => String::from(
            "unavailable from this artifact (no unambiguous 2-D head projection on disk)",
        ),
    };
    VokraError::UnsupportedOp(format!(
        "lang_id_ecapa identify (loud-partial): the full forward is deferred; four pieces \
         must land before real language logits can be emitted: (1) the filterbank front-end \
         — `vokra_ops::kaldi_fbank` exists in-repo, but this converter stamps NO \
         `vokra.lang_id.*` axis group, so its opts (n_mels, sample rate, frame length, hop) \
         cannot be derived from the artifact and must come from a real-checkpoint dump of \
         the upstream `hyperparams.yaml`; (2) the ECAPA-TDNN trunk (SE-Res2Blocks + \
         multi-layer feature aggregation) — no runtime ECAPA trunk binder exists in this \
         workspace yet, the sibling `ecapa_tdnn` arch is converter-side only; \
         (3) attentive statistics pooling onto the utterance embedding; (4) the language \
         classifier head. Output width = {width}. Language NAMES are not recoverable from a \
         converted GGUF at all: the converter stamps neither a language list nor a language \
         count, so an index cannot be mapped to an ISO 639 code without the upstream label \
         encoder. Primary sources: {PRIMARY_SOURCE_HF_VOXLINGUA107}, \
         {PRIMARY_SOURCE_HF_COMMONLANGUAGE}, paper {PRIMARY_SOURCE_PAPER}, reference code \
         {PRIMARY_SOURCE_CODE}. Runtime cannot fabricate a language logits array (FR-EX-08 \
         — no silent partial output)."
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the SpeechBrain lang-ID runtime binder — contract-constant
    //! pins, metadata round-trip, negative-space round-trip on every load
    //! gate, arch-tag distinctness, and the read-not-guessed language count.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real waveform this would be `identify(...)` returning a language
    //! logit vector, but the front-end / trunk / pooling / head composition is
    //! deferred (see the module doc). Fabricating a classification output
    //! would violate CLAUDE.md 教訓 (a). The round-trip semantics we *can*
    //! honestly test are the contract constants, the metadata and manifest
    //! binding, and the loud-error surfaces.
    //!
    //! # About the head tensor name used in fixtures
    //!
    //! The production code never references a head tensor *name* — the head
    //! partition is "every tensor not under [`TRUNK_PREFIX`]" precisely
    //! because this repository has not transcribed the head's module name from
    //! a primary source. The fixtures below therefore use `classifier.weight`
    //! as an explicit **placeholder**, chosen only to exercise the 2-D width
    //! rule; `head_named` proves the rule is name-agnostic by using a
    //! different string.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a lang-ID GGUF: arch + optional name + category + optional
    /// licence stamp + one trunk tensor + an optional head tensor.
    fn lang_id_gguf(
        model_name: Option<&str>,
        weight_license_class: Option<LicenseClass>,
        head: Option<(&str, Vec<u64>)>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        if let Some(n) = model_name {
            b.add_string(chunks::KEY_MODEL_NAME, n);
        }
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF_VOXLINGUA107);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // The ECAPA trunk tensor, using the name transcribed from the
        // converter's own fixtures.
        b.add_tensor(
            TRUNK_EXAMPLE_TENSOR,
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add trunk tensor");
        if let Some((name, dims)) = head {
            let elems: u64 = dims.iter().product();
            b.add_tensor(name, GgmlType::F32, dims, vec![0u8; elems as usize * 4])
                .expect("add head tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate consistency with the
    //          converter) + the deliberate absence of a taxonomy constant.
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "lang_id_ecapa", "lang-ID arch tag pin");
        assert_eq!(NAME_VOXLINGUA107, "lang-id-voxlingua107-ecapa");
        assert_eq!(NAME_COMMONLANGUAGE, "lang-id-commonlanguage-ecapa");
        assert_eq!(CATEGORY, "classification");
        assert_eq!(
            UPSTREAM_HF_VOXLINGUA107,
            "speechbrain/lang-id-voxlingua107-ecapa"
        );
        assert_eq!(
            UPSTREAM_HF_COMMONLANGUAGE,
            "speechbrain/lang-id-commonlanguage_ecapa"
        );
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_HF,
            "vokra.provenance.upstream_hf"
        );
        // The trunk prefix is transcribed from the converter fixtures, and the
        // example tensor must actually live under it.
        assert_eq!(TRUNK_PREFIX, "embedding_model.");
        assert!(
            TRUNK_EXAMPLE_TENSOR.starts_with(TRUNK_PREFIX),
            "the quoted example tensor must sit under the trunk prefix"
        );
        // Variant round-trip through the name stamp.
        assert_eq!(
            LangIdVariant::from_model_name(NAME_VOXLINGUA107),
            Some(LangIdVariant::VoxLingua107)
        );
        assert_eq!(
            LangIdVariant::from_model_name(NAME_COMMONLANGUAGE),
            Some(LangIdVariant::CommonLanguage)
        );
        assert_eq!(
            LangIdVariant::from_model_name("some-third-party-lang-id"),
            None,
            "an unrecognised name is None, not a panic and not a wrong variant"
        );
        assert_eq!(LangIdVariant::VoxLingua107.name(), NAME_VOXLINGUA107);
        assert_eq!(
            LangIdVariant::CommonLanguage.upstream_hf(),
            UPSTREAM_HF_COMMONLANGUAGE
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — Missing arch fails loud.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor(
            TRUNK_EXAMPLE_TENSOR,
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let Err(err) = LangIdEcapa::from_gguf(&file) else {
            panic!("expected an error when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native lang_id_ecapa GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
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
    // Test 3 — Foreign arch fails loud, naming BOTH expected and actual.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // `ecapa_tdnn` is the maximally confusable sibling: same trunk, same
        // `embedding_model.` naming, different head. Silent aliasing would
        // misroute dispatch (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "ecapa_tdnn");
        b.add_string(chunks::KEY_MODEL_NAME, "spkrec-ecapa-voxceleb");
        b.add_tensor(
            TRUNK_EXAMPLE_TENSOR,
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let Err(err) = LangIdEcapa::from_gguf(&file) else {
            panic!("expected an error when the GGUF carries a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH tags named.
                assert!(
                    m.contains("`ecapa_tdnn`"),
                    "message must name the ACTUAL arch found, got `{m}`"
                );
                assert!(
                    m.contains("`lang_id_ecapa`"),
                    "message must name the EXPECTED arch, got `{m}`"
                );
                // The whole confusable neighbourhood is enumerated so the
                // reader knows which loader they actually wanted.
                for sibling in [
                    "ecapa_tdnn",
                    "campplus",
                    "speaker_3d",
                    "redimnet",
                    "wavlm_sv",
                    "emotion2vec",
                ] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                assert!(
                    m.contains("backbone identity is NOT topology identity"),
                    "message must explain why a shared ECAPA trunk is not enough, got `{m}`"
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
    // Test 4 — A synthetic GGUF with the right tensors binds, and every
    //          metadata surface round-trips.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_and_round_trips_metadata() {
        // Head projection is 2-D `[out_features, in_features]`, so the width
        // read off disk must be its leading dimension — 7 here, an arbitrary
        // fixture value chosen precisely to prove the count comes from the
        // payload rather than from any constant this module might hold.
        let file = lang_id_gguf(
            Some(NAME_VOXLINGUA107),
            Some(LicenseClass::Permissive),
            Some(("classifier.weight", vec![7, 3])),
        );
        let m = LangIdEcapa::from_gguf(&file).expect("a well-formed lang-ID GGUF must bind");

        assert_eq!(
            m.variant(),
            Some(LangIdVariant::VoxLingua107),
            "variant must be recovered from the `vokra.model.name` stamp"
        );
        assert_eq!(m.model_name(), Some(NAME_VOXLINGUA107));
        assert_eq!(m.upstream_hf(), Some(UPSTREAM_HF_VOXLINGUA107));
        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "the converter's apache-2.0 stamp must round-trip"
        );
        assert_eq!(m.tensor_count(), 2, "one trunk tensor plus one head tensor");
        assert!(m.has_language_head());
        assert_eq!(
            m.language_count(),
            Some(7),
            "the language count must be READ off the head projection, not hardcoded"
        );

        // Manifest-level accessors agree with the handle-level ones.
        let w = m.weights();
        assert_eq!(w.count_with_prefix(TRUNK_PREFIX), 1);
        assert_eq!(w.trunk_tensors().len(), 1);
        assert_eq!(w.language_head_tensors().len(), 1);
        assert_eq!(w.tensor_dims(TRUNK_EXAMPLE_TENSOR), Some(&[2usize, 3][..]));
        assert!(w.tensor_names().contains(&TRUNK_EXAMPLE_TENSOR));
    }

    // -----------------------------------------------------------------------
    // Test 5 — The F9 variant binds through the same arch, and the head width
    //          is read from the payload rather than assumed from the variant.
    // -----------------------------------------------------------------------

    #[test]
    fn commonlanguage_variant_binds_with_its_own_width() {
        // A different fixture width (5) on the CommonLanguage variant: if the
        // module carried a taxonomy constant per variant, this would disagree.
        let file = lang_id_gguf(
            Some(NAME_COMMONLANGUAGE),
            Some(LicenseClass::Permissive),
            Some(("classifier.weight", vec![5, 3])),
        );
        let m = LangIdEcapa::from_gguf(&file).expect("F9 must bind through the shared arch");
        assert_eq!(m.variant(), Some(LangIdVariant::CommonLanguage));
        assert_eq!(
            m.language_count(),
            Some(5),
            "the width follows the payload, not the variant"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6 — The head partition is name-agnostic, and an ambiguous or
    //          absent head reports `None` instead of guessing.
    // -----------------------------------------------------------------------

    #[test]
    fn language_count_is_read_or_absent_never_guessed() {
        // (a) A head under a completely different name still counts — the
        //     partition is "not under the trunk prefix", so nothing depends on
        //     a head-prefix string this repository has not transcribed.
        let head_named = lang_id_gguf(
            Some(NAME_VOXLINGUA107),
            None,
            Some(("out.linear.w", vec![11, 3])),
        );
        let m = LangIdEcapa::from_gguf(&head_named).expect("bind");
        assert_eq!(m.language_count(), Some(11));
        // No licence stamp on this fixture: fail-closed to Unknown.
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing licence stamp must fail closed to Unknown"
        );

        // (b) Bare-encoder export — trunk only, no head. Legitimate artifact,
        //     but no width to report.
        let bare = lang_id_gguf(Some(NAME_VOXLINGUA107), None, None);
        let m = LangIdEcapa::from_gguf(&bare).expect("a bare-encoder export still binds");
        assert!(!m.has_language_head());
        assert_eq!(
            m.language_count(),
            None,
            "no head on disk means no width — never a fallback constant"
        );

        // (c) A rank-3 head projection is ambiguous (SpeechBrain's
        //     cosine-similarity classifier variant), so report nothing.
        let rank3 = lang_id_gguf(
            Some(NAME_VOXLINGUA107),
            None,
            Some(("classifier.weight", vec![1, 9, 3])),
        );
        let m = LangIdEcapa::from_gguf(&rank3).expect("bind");
        assert!(m.has_language_head());
        assert_eq!(
            m.language_count(),
            None,
            "an unverified rank-3 axis order must not be interpreted"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7 — Missing tensors fail loud, naming what was missing.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_trunk_naming_it() {
        // Correct arch and a non-empty manifest, but no ECAPA trunk: the
        // backbone is not optional, so this must fail at load with a message
        // naming the prefix and an example tensor.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_VOXLINGUA107);
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![7, 3],
            vec![0u8; 7 * 3 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let Err(err) = LangIdEcapa::from_gguf(&file) else {
            panic!("expected an error when the ECAPA trunk tensors are missing");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(TRUNK_PREFIX),
                    "message must NAME the missing trunk prefix, got `{m}`"
                );
                assert!(
                    m.contains(TRUNK_EXAMPLE_TENSOR),
                    "message must show an example of the tensor name it wanted, got `{m}`"
                );
                assert!(
                    m.contains("classifier.weight"),
                    "message must echo what WAS on disk so the reader can compare, got `{m}`"
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
    // Test 8 — An empty tensor manifest fails loud (never an all-zero
    //          forward — FR-EX-08).
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_VOXLINGUA107);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let Err(err) = LangIdEcapa::from_gguf(&file) else {
            panic!("expected an error when the GGUF carries zero tensors");
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
                    m.contains("vokra-cli convert --model lang-id-voxlingua107"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — `identify` loud-partials, naming every missing primitive and
    //          citing the primary sources.
    // -----------------------------------------------------------------------

    #[test]
    fn identify_loud_partial_names_the_missing_primitives() {
        let file = lang_id_gguf(
            Some(NAME_VOXLINGUA107),
            Some(LicenseClass::Permissive),
            Some(("classifier.weight", vec![7, 3])),
        );
        let m = LangIdEcapa::from_gguf(&file).expect("bind");

        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.identify(&pcm) else {
            panic!("expected identify to loud-partial rather than return fabricated logits");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("lang_id_ecapa identify"),
                    "the surface must be named: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Every deferred primitive named explicitly.
                for primitive in [
                    "kaldi_fbank",
                    "vokra.lang_id.*",
                    "ECAPA-TDNN trunk",
                    "SE-Res2Blocks",
                    "attentive statistics pooling",
                    "language classifier head",
                ] {
                    assert!(
                        msg.contains(primitive),
                        "expected the missing primitive '{primitive}' to be named: {msg}"
                    );
                }

                // The output width is reported from the payload, not invented.
                assert!(
                    msg.contains("Output width = 7"),
                    "the width read off disk must be reported: {msg}"
                );
                assert!(
                    msg.contains("not from any constant"),
                    "the message must say the width was read, not hardcoded: {msg}"
                );

                // The unavailability of the names is stated plainly.
                assert!(
                    msg.contains("Language NAMES are not recoverable"),
                    "the absent taxonomy must be stated plainly: {msg}"
                );

                // All four primary sources cited.
                for url in [
                    PRIMARY_SOURCE_HF_VOXLINGUA107,
                    PRIMARY_SOURCE_HF_COMMONLANGUAGE,
                    PRIMARY_SOURCE_PAPER,
                    PRIMARY_SOURCE_CODE,
                ] {
                    assert!(
                        msg.contains(url),
                        "expected primary source URL '{url}' cited: {msg}"
                    );
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "expected the FR-EX-08 rationale for emitting no fake logits: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10 — The loud-partial stays honest when the width is unknown.
    // -----------------------------------------------------------------------

    #[test]
    fn identify_loud_partial_declares_unknown_width_honestly() {
        // Bare-encoder export: no head, so no width can be read.
        let file = lang_id_gguf(Some(NAME_VOXLINGUA107), None, None);
        let m = LangIdEcapa::from_gguf(&file).expect("bind");
        let Err(err) = m.identify(&[0.0_f32; 16]) else {
            panic!("expected identify to loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("Output width = unavailable from this artifact"),
                    "an unreadable width must be declared unavailable, never guessed: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
