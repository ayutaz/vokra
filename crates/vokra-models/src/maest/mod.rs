//! **MAEST** — "Music Audio Efficient Spectrogram Transformer"
//! (`mtg-upf/discogs-maest-30s-pw-129e`, **cc-by-nc-sa-4.0**) — runtime binder
//! for the `maest` converter arch (Wave C2 2026-08-15, loud-partial per the
//! `atst` / `m2d` / `emotion2vec` / `wavlm` / `panns` / `redimnet` precedent —
//! CLAUDE.md 教訓 (a):「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/maest.rs` has been stamping
//! `vokra.model.arch = "maest"` since the 2026-08-13 SSL audio-encoder wave,
//! but a workspace-wide grep proved that **nothing read that arch string
//! back** — a converted MAEST checkpoint was unloadable. This module is that
//! consumer.
//!
//! # Primary sources
//!
//! Every fact below is transcribed from the converter's own module docstring
//! (`crates/vokra-convert/src/models/maest.rs`) and its [`ModelKind`] entry in
//! `crates/vokra-convert/src/lib.rs`, which together are this repository's
//! primary-source record for MAEST. Nothing here is re-derived from memory.
//!
//! - Upstream release: <https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e>
//! - Paper: Alonso-Jiménez et al. 2023, ISMIR — <https://arxiv.org/abs/2309.16418>
//! - Backbone: the HF `config` records `model_type:
//!   audio-spectrogram-transformer` and `architectures:
//!   ["ASTForAudioClassification"]` (verified via the HF cardData API on
//!   2026-08-13 and recorded by the converter).
//! - Scale: safetensors `parameters.F32: 86,858,128` per the HF API
//!   ([`UPSTREAM_PARAM_COUNT_F32`]).
//! - Licence: HF cardData `license: cc-by-nc-sa-4.0` → the **T4 tier +
//!   ShareAlike cascade**, i.e. [`LicenseClass::NonCommercialShareAlike`].
//!
//! [`ModelKind`]: https://docs.rs/vokra-convert
//!
//! # What MAEST is — and why it is the music-domain member of the SSL fleet
//!
//! MAEST is a self-supervised **music** encoder: an AST (Audio Spectrogram
//! Transformer) backbone — a ViT-style patch-wise Transformer over a log-mel
//! spectrogram — pretrained on the MTG Discogs4All **music-tagger** dataset.
//! The `30s-pw-129e` variant this [`NAME`] tracks is 30-second, patch-wise, 129
//! epochs.
//!
//! Unlike its general-audio siblings (`atst` / `eat` / `m2d` / `dasheng`, all
//! of which ship a bare encoder and no task head), MAEST's upstream
//! `architectures` string is `ASTForAudioClassification` — i.e. the release
//! **does** carry a tagging head over a Discogs label taxonomy. The converter
//! is a verbatim float pass-through, so if that head is in the checkpoint its
//! tensors ride through under their upstream `state_dict` names and land on
//! disk. This binder therefore exposes a tag surface — but see the honesty
//! constraint in "Label taxonomy" below.
//!
//! ```text
//! PCM (mono f32)
//!   -> log-mel spectrogram front-end                    ← **loud-partial**
//!        (axes — sample rate / n_fft / hop / n_mels — are NOT stamped by
//!         the converter and are not transcribed anywhere in-repo; see
//!         blocker (i)).
//!   -> 2-D patch embedding over the mel plane           ← **loud-partial**
//!        (ViT-style patchification; the patch grid is part of blocker (i)).
//!   -> pre-norm Transformer encoder (~87M-param AST)    ← **loud-partial**
//!        (blocker (ii): `vokra-ops` has no ViT-style plain pre-norm
//!         Transformer encoder primitive today — the SHARED gap across the
//!         whole SSL fleet).
//!   -> per-patch hidden states     ── [`Maest::encode`]
//!   -> pooled clip embedding       ── [`Maest::embed`]
//!   -> Discogs tag logits          ── [`Maest::tag`]
//! ```
//!
//! # Label taxonomy — count is READ, names and size are never guessed
//!
//! The converter stamps **no** label list and **no** label count: it is a
//! verbatim F32/F16/BF16 pass-through whose only metadata is `vokra.model.*` +
//! `vokra.provenance.*`. Writing a taxonomy size into this module from memory
//! would be exactly the fabrication CLAUDE.md「ハルシネーション厳禁」forbids,
//! so this module contains **no taxonomy constant at all**.
//!
//! What it does instead is *read* the artifact: [`Maest::label_count`] scans
//! the tensors actually on disk under [`TAG_HEAD_PREFIX`] and, when exactly one
//! of them is 2-D, reports that tensor's first dimension — because a PyTorch
//! `nn.Linear` weight is `[out_features, in_features]`, so `dims[0]` of the
//! head projection *is* the label-set size of whatever checkpoint the caller
//! converted. Any other shape on disk yields `None`, never a fallback number.
//! The label **names** are unrecoverable from the artifact entirely, which is
//! blocker (iv) of the [`Maest::tag`] loud-partial.
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real (this WP)** — everything around the forward:
//!
//! - [`Maest::from_gguf`] with **strict** `vokra.model.arch == "maest"`
//!   verification. A sibling SSL-encoder GGUF handed here by mistake fails with
//!   a message naming **both** tags and enumerating the whole
//!   audio/music-embedding neighbourhood — `ast` most sharply, since MAEST
//!   shares its *backbone* but not its objective, its domain or its taxonomy
//!   (FR-EX-08 — never a silent misroute).
//! - [`MaestWeights::from_gguf`] tensor-manifest binding over the verbatim
//!   upstream `state_dict` names the converter passes through, with a non-empty
//!   gate plus [`MaestWeights::require_tensor`] /
//!   [`MaestWeights::require_tensor_dims`] lookups that name the missing
//!   tensor, or **both** the expected and the actual dims.
//! - Tag-head discovery from disk: [`MaestWeights::tag_head_tensors`] /
//!   [`MaestWeights::label_count_from_disk`], reporting only what the artifact
//!   contains.
//! - Metadata surfacing: [`Maest::name`] / [`Maest::category`] /
//!   [`Maest::upstream_hf`] / [`Maest::model_id`] / [`Maest::source`] read back
//!   the converter's stamps.
//! - Weight-licence + FR-MD-09 attribution surfacing, fail-closing to
//!   [`LicenseClass::Unknown`] when the artifact carries no stamp, and the
//!   compliance-gated [`Maest::from_gguf_with_policy`] / [`Maest::from_path`] /
//!   [`Maest::from_path_with_policy`] entry points.
//!
//! **Loud-partial (this WP)** — [`Maest::encode`], [`Maest::embed`] and
//! [`Maest::tag`] return [`VokraError::UnsupportedOp`] naming these blockers:
//!
//! 1. **No `vokra.maest.*` axis chunk group.** The converter stamps
//!    `vokra.model.*` + `vokra.provenance.*` and nothing else. Embedding width,
//!    depth, head count, patch grid, and every log-mel front-end axis are
//!    therefore **absent from the artifact** and are not transcribed anywhere
//!    in this repository. Fabricating them from "typical AST-base" numbers
//!    would bind shape-valid garbage.
//! 2. **No ViT-style encoder primitive in `vokra-ops`.** The log-mel front-end
//!    genuinely exists (`vokra_ops::mel`, `vokra_ops::fused_logmel`,
//!    `vokra_ops::kaldi_fbank` — verified by listing `crates/vokra-ops/src/`),
//!    but the 2-D patch embedding + plain pre-norm Transformer encoder does
//!    not: `vokra_ops::conformer`, `vokra_ops::ebranchformer` and
//!    `vokra_ops::zipformer` are all conv-augmented **ASR** encoders over a 1-D
//!    frame sequence, not ViT patch encoders over a 2-D mel plane. This is the
//!    **same single gap** the sibling `atst` / `eat` / `m2d` / `dasheng`
//!    binders hit — one shared follow-up, not five unrelated ones.
//! 3. **No verified tensor-name manifest.** The converter copies every float
//!    tensor under its verbatim upstream `state_dict` name, and the only MAEST
//!    names recorded in-repo are the two samples in the converter's own test
//!    module. Walking guessed names into typed slots would bind the wrong
//!    tensors without failing.
//! 4. **No label taxonomy** (tag surface only). Even once the encoder lands,
//!    the artifact carries no label *names*, so logits cannot be mapped to
//!    human-readable Discogs genre / mood / instrument / era tags. Only the
//!    label **count** is recoverable, and only by reading the head tensor's
//!    dims off disk (see "Label taxonomy" above).
//!
//! No fabricated hidden states, embeddings or tag logits are ever emitted
//! (FR-EX-08 — no silent partial output). The scaffold is arranged so a
//! follow-up wave flips the switch by (a) teaching the converter to stamp a
//! `vokra.maest.*` axis group plus the label list, (b) adding the shared ViT
//! patch-encoder primitive to `vokra-ops`, and (c) transcribing the upstream
//! tensor names — with the binder, the manifest lookups and the tests already
//! in place.
//!
//! # Sibling family distinctness (SSL audio/music-embedding neighbourhood)
//!
//! [`ARCH`] = `"maest"` is deliberately distinct from every sibling:
//!
//! - `ast` — the **same AST backbone**, but supervised, fine-tuned on AudioSet,
//!   general-audio, and published under a different licence tier
//!   (`bsd-3-clause`). Backbone identity is not topology identity: the
//!   objective, domain, head and taxonomy all differ, so this is the sharpest
//!   confusable pair in the fleet.
//! - `atst` — BYOL-style teacher-student patchout (general audio);
//! - `beats` — iterative acoustic tokenizer + masked acoustic modelling;
//! - `eat` — utterance-level MAE with efficient inverse block masking;
//! - `dasheng` — universal MAE;
//! - `m2d` — masked modelling **duo** (dual online + target branch);
//! - `mert` — HuBERT-derived masked prediction (music);
//! - `muq` — Mel-RVQ + BEATs teacher (music);
//! - `yamnet` / `panns` — supervised audio-tagging CNNs, not SSL at all;
//! - `clap` — contrastive language-audio pretraining (text tower attached);
//! - `hubert` / `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec` — the wav2vec2
//!   lineage, whose encoders sit on a **raw-waveform 1-D conv stem** rather
//!   than a log-mel patch grid.
//!
//! Sharing an arch tag would let runtime dispatch bind, say, an AudioSet
//! 527-class head or a raw-waveform conv stem over a Discogs music-tagger
//! checkpoint (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's constants** — the
//! same rule every sibling binder (`atst` / `m2d` / `emotion2vec` / `wavlm` /
//! `panns` / `redimnet` / `canary_1b_flash`) follows so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # Licence posture — T4 + ShareAlike, fail-closed
//!
//! The converter stamps `cc-by-nc-sa-4.0` → [`LicenseClass::NonCommercialShareAlike`],
//! whose [`LicenseClass::requires_research_flag`] is `true`. A correctly
//! stamped MAEST artifact is therefore **refused** under
//! [`CompliancePolicy::strict`] and loads only with an explicit research opt-in
//! ([`CompliancePolicy::with_research_license`], `VOKRA_ALLOW_RESEARCH_LICENSE=1`,
//! or [`ComplianceLevel::Research`]) — that refusal is the correct behaviour,
//! not a bug. Three obligations cascade: **NonCommercial** (no commercial use
//! without a separate licence from the MTG group), **ShareAlike** (any
//! downstream distribution stays CC-BY-NC-SA 4.0 —
//! [`LicenseClass::requires_license_preserved`]), and **BY** (attribution —
//! [`LicenseClass::requires_attribution`]).
//!
//! Note that the converter's `stamp_provenance` call writes weight-licence,
//! SPDX, model id and source but **not** `vokra.provenance.attribution`, so
//! [`Maest::attribution`] reads `None` on a converter-produced artifact today
//! even though the BY cascade obliges a downstream to display credit. That is a
//! recorded gap, surfaced rather than papered over.
//!
//! This binder only *surfaces* whatever class the artifact carries;
//! `docs/license-audit.md` §3.1 sign-off stays **blank** (owner-only per memory
//! `[[feedback-license-signoff-primary-source]]` — CC does not sign, and does
//! not treat a converter default as a sign-off).
//!
//! [`ComplianceLevel::Research`]: vokra_core::ComplianceLevel::Research
//!
//! # No ONNX / no pickle (permanent)
//!
//! MAEST ships as single-file safetensors (`model.safetensors`); the upstream
//! repo also carries a legacy `pytorch_model.bin` pickle which Vokra never
//! reads. This runtime **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/maest.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model maest-30s-pw-129e`.
///
/// Distinct from every sibling SSL audio/music-embedding arch tag (`ast` /
/// `atst` / `beats` / `eat` / `dasheng` / `m2d` / `mert` / `muq` / `yamnet` /
/// `panns` / `clap`) and from the wav2vec2 lineage (`hubert` /
/// `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec`). The `ast` pair is the sharpest
/// one: MAEST shares AST's backbone but not its objective (Discogs music-tagger
/// SSL vs supervised AudioSet fine-tuning), its domain, its head or its
/// taxonomy — silently sharing a tag would misroute runtime dispatch (FR-EX-08,
/// see the module docstring "Sibling family distinctness" section).
pub const ARCH: &str = "maest";

/// Expected `vokra.model.name` value written by the converter — the canonical
/// `30s-pw-129e` release variant (30-second, patch-wise, 129 epochs).
///
/// Sibling duration / epoch variants (`5s` / `10s` / `20s`, `30s-pw-73e`, …)
/// are distinct release identities that the converter publishes under their own
/// `NAME` following the `snac_24khz` / `snac_44khz` pattern, so this value is
/// **surfaced, not gated** — see [`Maest::name`].
pub const NAME: &str = "maest-30s-pw-129e";

/// Expected `vokra.model.category` value — `music-embedding`, shared with the
/// sibling music SSL encoders (`mert` / `muq`).
///
/// Deliberately **not** `audio-tagging` (the `yamnet` / `panns` / `ast` /
/// `clap` category): MAEST is trained on the Discogs music-tagger dataset, so
/// its outputs are genre / mood / instrument / era annotations over music
/// rather than the general AudioSet audio-event ontology. Consumed by the
/// model-card generator and the zoo-manifest tier gate.
pub const CATEGORY: &str = "music-embedding";

/// Upstream HuggingFace slug — stamped on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "mtg-upf/discogs-maest-30s-pw-129e";

/// Default SPDX stamped by the converter — the **weight** tier.
///
/// Resolves to [`LicenseClass::NonCommercialShareAlike`] (T4 + ShareAlike
/// cascade). A caller with a different attestation may override at the
/// converter boundary (`--license <spdx>`), which is why this binder *surfaces*
/// rather than *asserts* the class.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

/// Metadata key holding [`CATEGORY`] (not part of `vokra_core::gguf::chunks`,
/// so mirrored here from the converter).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key holding [`UPSTREAM_HF`] (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// F32 parameter count of the upstream release, as reported by the HuggingFace
/// API (`parameters.F32: 86,858,128`) and recorded by the converter's module
/// docstring on 2026-08-13.
///
/// Used only to make the empty-manifest refusal concrete — it is never used to
/// validate a payload, because a sibling duration variant legitimately differs.
pub const UPSTREAM_PARAM_COUNT_F32: usize = 86_858_128;

/// `state_dict` name prefix under which the upstream tagging head is expected
/// to sit.
///
/// **Unverified convention.** The converter records that upstream's HF `config`
/// declares `architectures: ["ASTForAudioClassification"]`, and the HF wrapper
/// of that name places its head under a `classifier.` prefix — but no real
/// MAEST tensor listing is transcribed anywhere in this repository, so this
/// string has not been checked against a released checkpoint. Nothing in this
/// binder *depends* on it: it only shapes what
/// [`MaestWeights::tag_head_tensors`] reports, and an artifact with no matching
/// tensor is **not** rejected (a bare-encoder export is a legitimate artifact).
pub const TAG_HEAD_PREFIX: &str = "classifier.";

// Primary-source anchors, cited inside the loud-partial error so a reader
// diagnosing the gap has fully specified places to walk.

/// Primary-source anchor: the upstream HuggingFace release.
pub const PRIMARY_SOURCE_UPSTREAM_HF: &str = "huggingface.co/mtg-upf/discogs-maest-30s-pw-129e";

/// Primary-source anchor: Alonso-Jiménez et al. 2023 (ISMIR) — the MAEST paper.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2309.16418";

// ---------------------------------------------------------------------------
// MaestWeights — the tensor manifest, with loud lookups
// ---------------------------------------------------------------------------

/// Weight tensors bound from a MAEST GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step.
/// A GGUF that carries zero tensors is rejected with [`VokraError::ModelLoad`]
/// (FR-EX-08 — an ~87M-parameter AST never converts to an empty manifest, so
/// zero tensors always signals a mis-produced artifact, and binding it would
/// silently run an all-zero forward).
///
/// Under the current landing this struct stores the tensor names and their
/// GGUF-side dims. The payload is deliberately not dequantised: the forward is
/// loud-partial (see [`Maest::encode`]), and the follow-up wave sizes its
/// dequant per its kernel needs. [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) are already in place so
/// that wave walks a manifest that fails loudly rather than substituting zeros.
#[derive(Debug, Clone)]
pub struct MaestWeights {
    /// Tensors discovered on disk, in file order, as
    /// `(upstream state_dict name, GGUF-side dims)`.
    tensors: Vec<(String, Vec<usize>)>,
}

impl MaestWeights {
    /// Scans `gguf` for the MAEST `state_dict` tensors.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   (FR-EX-08 — refusing to bind an all-zero forward).
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
                "maest: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate MAEST checkpoint is an AST-backbone Transformer of \
                 roughly {UPSTREAM_PARAM_COUNT_F32} F32 parameters (arch={ARCH}, name={NAME}) \
                 and always converts to hundreds of Linear / LayerNorm tensors, so an empty \
                 manifest always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model maest-30s-pw-129e` against the upstream \
                 `{UPSTREAM_HF}` safetensors release."
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

    /// The tensors on disk whose name starts with [`TAG_HEAD_PREFIX`], as
    /// `(name, dims)` pairs in file order.
    ///
    /// Pure disk reporting with no interpretation: an empty result means either
    /// the artifact is a bare-encoder export or the upstream head sits under a
    /// prefix this repository has not transcribed (see [`TAG_HEAD_PREFIX`]).
    /// Neither case is an error.
    #[must_use]
    pub fn tag_head_tensors(&self) -> Vec<(&str, &[usize])> {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(TAG_HEAD_PREFIX))
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect()
    }

    /// The label-set size **read off the artifact**, or `None`.
    ///
    /// Returns `Some(dims[0])` when exactly one tensor under
    /// [`TAG_HEAD_PREFIX`] is 2-D — a PyTorch `nn.Linear` weight is
    /// `[out_features, in_features]`, and the converter passes the safetensors
    /// shape through verbatim, so that leading dimension *is* the number of
    /// Discogs labels in whatever checkpoint the caller converted.
    ///
    /// Returns `None` in every other case (no head on disk, or an ambiguous
    /// shape layout). It never falls back to a taxonomy constant: this module
    /// deliberately contains none, because the converter stamps no label count
    /// and inventing one would be fabrication
    /// (CLAUDE.md「ハルシネーション厳禁」).
    #[must_use]
    pub fn label_count_from_disk(&self) -> Option<usize> {
        let mut two_d = self
            .tag_head_tensors()
            .into_iter()
            .filter(|(_, dims)| dims.len() == 2);
        let first = two_d.next()?;
        if two_d.next().is_some() {
            // More than one 2-D tensor under the head prefix: which one is the
            // label projection is ambiguous, so report nothing rather than pick.
            return None;
        }
        first.1.first().copied()
    }

    /// Dims of a **required** tensor, failing loudly when it is absent.
    ///
    /// The error names the missing tensor, the manifest size, and up to five
    /// nearby names on disk so a caller diagnosing a prefix mismatch has
    /// something concrete to compare against.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `name` is not present (FR-EX-08 —
    ///   never substitute a zero tensor).
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        if let Some(dims) = self.tensor_dims(name) {
            return Ok(dims);
        }
        let segment = name.split('.').next().unwrap_or(name);
        let mut near: Vec<&str> = self
            .tensors
            .iter()
            .filter(|(n, _)| n.starts_with(segment))
            .map(|(n, _)| n.as_str())
            .take(5)
            .collect();
        if near.is_empty() {
            near = self
                .tensors
                .iter()
                .map(|(n, _)| n.as_str())
                .take(5)
                .collect();
        }
        Err(VokraError::ModelLoad(format!(
            "maest: required tensor `{name}` is absent from the GGUF ({count} tensors \
             present; nearest names on disk: {near:?}). The converter passes upstream \
             `state_dict` names through verbatim, so a mismatch means either the checkpoint \
             was flattened with a different prefix policy (upstream wraps the backbone as \
             `ASTForAudioClassification`, whose body sits under an \
             `audio_spectrogram_transformer.` prefix that a re-export may strip) or the \
             caller is walking a manifest transcribed from a different MAEST release variant \
             (the 5s / 10s / 20s durations and the `30s-pw-73e` checkpoint point are \
             published under their own names). Refusing to substitute a zero tensor \
             (FR-EX-08). Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}",
            count = self.tensors.len(),
        )))
    }

    /// Asserts that a required tensor is present **and** has exactly `expected`
    /// dims.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the tensor is absent (see
    ///   [`Self::require_tensor`]) or when its dims differ — the message names
    ///   **both** the expected and the actual dims (FR-EX-08 — never reshape or
    ///   truncate silently).
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "maest: tensor `{name}` has dims {actual:?} but the caller expects \
                 {expected:?} — refusing to reshape or truncate silently (FR-EX-08). MAEST \
                 stamps no `vokra.maest.*` axis group, so a dims disagreement here means the \
                 walking code's transcribed topology does not match the payload (a different \
                 duration variant, a different epoch checkpoint, or a re-export with a \
                 different label-set size). Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Maest — the runtime binder handle
// ---------------------------------------------------------------------------

/// MAEST (Music Audio Efficient Spectrogram Transformer) self-supervised music
/// encoder with a Discogs tagging head.
///
/// Bind with [`from_gguf`](Self::from_gguf) — or the compliance-gated
/// [`from_gguf_with_policy`](Self::from_gguf_with_policy) /
/// [`from_path`](Self::from_path) / [`from_path_with_policy`](Self::from_path_with_policy)
/// — then call [`encode`](Self::encode) for per-patch hidden states,
/// [`embed`](Self::embed) for the pooled clip embedding, or [`tag`](Self::tag)
/// for Discogs tag logits. All three forwards are **loud-partial** today; see
/// the module docstring for the blockers and the FR-EX-08 contract.
#[derive(Debug, Clone)]
pub struct Maest {
    name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
    model_id: Option<String>,
    source: Option<String>,
    weights: MaestWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl Maest {
    /// Binds a MAEST GGUF: verifies the arch strictly, binds the tensor
    /// manifest, and surfaces the converter's metadata + licence stamps.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the missing
    /// or wrong key, so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// `vokra.model.name` is deliberately **surfaced, not gated**: the duration
    /// and epoch sibling variants share this arch under different names, so a
    /// hard name check would make a legitimate future artifact unloadable. See
    /// [`Self::name`].
    ///
    /// This entry point performs **no licence gate** — use
    /// [`Self::from_gguf_with_policy`] for that. MAEST is
    /// [`LicenseClass::NonCommercialShareAlike`], so the gated route refuses
    /// under [`CompliancePolicy::strict`] by design.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not `"maest"` —
    ///   the message names both the found and the expected tag and enumerates
    ///   the SSL audio/music-embedding neighbourhood.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`MaestWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch first, so a mis-routed artifact reports the arch mismatch
        //    (the actionable fact) instead of a downstream missing-tensor trail.
        verify_arch(file)?;

        // 2. Metadata surfacing. Soft: a converter-produced artifact always
        //    carries these, but they are diagnostics, not load gates.
        let read_str = |key: &str| -> Option<String> {
            file.get(key).and_then(|v| v.as_str()).map(str::to_owned)
        };
        let name = read_str(chunks::KEY_MODEL_NAME);
        let category = read_str(GGUF_KEY_MODEL_CATEGORY);
        let upstream_hf = read_str(GGUF_KEY_PROVENANCE_UPSTREAM_HF);
        let model_id = read_str(chunks::KEY_PROVENANCE_MODEL_ID);
        let source = read_str(chunks::KEY_PROVENANCE_SOURCE);

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = MaestWeights::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps
        //    `NonCommercialShareAlike` (cc-by-nc-sa-4.0); an artifact missing
        //    the stamp reads back as `Unknown` — fail-closed at the M2-13 gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let attribution = read_str(chunks::KEY_PROVENANCE_ATTRIBUTION);

        Ok(Self {
            name,
            category,
            upstream_hf,
            model_id,
            source,
            weights,
            weight_license,
            attribution,
        })
    }

    /// Loads a MAEST GGUF from raw bytes under `policy` (the M2-13
    /// weight-licence gate).
    ///
    /// MAEST ships **CC-BY-NC-SA 4.0** → [`LicenseClass::NonCommercialShareAlike`],
    /// whose [`LicenseClass::requires_research_flag`] is `true`, so a correctly
    /// stamped artifact is **refused** under [`CompliancePolicy::strict`] and
    /// loads only with an explicit research opt-in. An artifact with no
    /// provenance stamp resolves to [`LicenseClass::Unknown`] and is refused for
    /// the same reason — fail-closed, never a silent substitution.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on GGUF parse failure, or on a wrong /
    ///   missing `vokra.model.arch`.
    /// - `VokraError::ResearchLicenseRequired` from the compliance gate when the
    ///   weight class is gated and `policy` grants no research opt-in — the
    ///   expected outcome for MAEST under a strict policy.
    /// - See [`Self::from_gguf`] for the remaining bind errors.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("maest GGUF: {e}")))?;
        // Arch before the compliance gate so a mis-routed artifact reports the
        // arch mismatch rather than a licence verdict about a model the caller
        // never meant to load.
        verify_arch(&file)?;
        check_weight_license(&file, policy)?;
        Self::from_gguf(&file)
    }

    /// Loads a MAEST GGUF from a path under [`CompliancePolicy::strict`].
    ///
    /// Because MAEST is non-commercial, this route **refuses** a correctly
    /// stamped artifact — that is the fail-closed default working as intended.
    /// Callers with a research/evaluation basis should use
    /// [`Self::from_path_with_policy`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a MAEST GGUF from a path under an explicit `policy`.
    ///
    /// The route a research/evaluation consumer takes:
    /// `CompliancePolicy::strict().with_research_license(true)` unlocks the
    /// non-commercial gate and emits the mandatory research-only warning. The
    /// ShareAlike and attribution obligations are not waived by that opt-in —
    /// see [`Self::weight_license`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path_with_policy(
        path: impl AsRef<std::path::Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, policy)
    }

    /// The stamped `vokra.model.name`, if present.
    ///
    /// [`NAME`] (`"maest-30s-pw-129e"`) for the release this module tracks; the
    /// duration / epoch sibling variants share [`ARCH`] under different names,
    /// which is why this is surfaced rather than gated.
    #[inline]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, if present — [`CATEGORY`]
    /// (`"music-embedding"`) for a converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_hf`, if present — [`UPSTREAM_HF`]
    /// for a converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }

    /// The stamped `vokra.provenance.model_id`, if present — the converter
    /// passes [`NAME`] into `stamp_provenance`, and the compliance gate uses
    /// this same key when naming a refused model.
    #[inline]
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// The stamped `vokra.provenance.source`, if present — the converter's
    /// free-text upstream description (release slug, objective, scale, paper).
    #[inline]
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &MaestWeights {
        &self.weights
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Whether the artifact carries any tensor under [`TAG_HEAD_PREFIX`].
    ///
    /// **Diagnostic only** — it gates nothing. `false` is not an error: a
    /// bare-encoder export is legitimate, and so is a head under a prefix this
    /// repository has not transcribed.
    #[inline]
    #[must_use]
    pub fn has_tag_head(&self) -> bool {
        self.weights.count_with_prefix(TAG_HEAD_PREFIX) > 0
    }

    /// The Discogs label-set size **read off this artifact**, or `None`.
    ///
    /// Delegates to [`MaestWeights::label_count_from_disk`]: the value comes
    /// from the head projection's leading dimension on disk, never from a
    /// taxonomy constant (this module contains none — see the module docstring
    /// "Label taxonomy" section).
    #[inline]
    #[must_use]
    pub fn label_count(&self) -> Option<usize> {
        self.weights.label_count_from_disk()
    }

    /// The weight-licence class surfaced from
    /// `vokra.provenance.weight_license`.
    ///
    /// [`LicenseClass::NonCommercialShareAlike`] for a correctly stamped MAEST
    /// artifact (cc-by-nc-sa-4.0), carrying three cascading obligations —
    /// non-commercial use only, ShareAlike on any downstream distribution, and
    /// attribution. [`LicenseClass::Unknown`] when the stamp is absent
    /// (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The FR-MD-09 attribution text stamped under
    /// `vokra.provenance.attribution`, if any.
    ///
    /// CC-BY-NC-SA 4.0 carries a BY cascade, so a consumer shipping
    /// MAEST-derived output must render credit. The converter's
    /// `stamp_provenance` call does **not** currently write this key, so a
    /// converter-produced artifact reads `None` — a recorded gap, surfaced here
    /// rather than papered over with invented wording.
    #[inline]
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }

    /// Encodes a mono `f32` PCM slice into the MAEST encoder's **per-patch
    /// hidden states** (`[n_patches][embed_dim]`).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the three blockers
    /// enumerated in the module docstring: (i) no `vokra.maest.*` axis chunk
    /// group, (ii) no ViT-style patch-embedding + pre-norm Transformer encoder
    /// primitive in `vokra-ops` — the gap shared with the whole SSL fleet, (iii)
    /// no verified tensor-name manifest. Both primary sources are cited so a
    /// reader diagnosing the gap has concrete places to walk. **No fabricated
    /// hidden states are ever emitted** (FR-EX-08 — no silent partial output).
    ///
    /// `pcm` is treated as mono `f32` in `[-1, 1]`. Its sample rate is
    /// deliberately **not** asserted: the required rate is part of blocker (i),
    /// and asserting a guessed rate would be exactly the fabrication this module
    /// refuses.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the deferred
    ///   encoder forward.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind explicitly so a future accidental removal of the parameter is
        // not masked by an unused-variable warning (mirror of the atst / m2d /
        // emotion2vec loud-partial signature discipline).
        let _ = pcm;
        Err(forward_loud_partial(
            "encode",
            "per-patch hidden states",
            false,
        ))
    }

    /// Encodes a mono `f32` PCM slice into the **pooled clip embedding**.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] for the same blockers as
    /// [`Self::encode`] — the pooled embedding is the encoder output reduced
    /// over the patch axis, so it cannot exist before the encoder does. The
    /// **width** of the returned vector is itself unknown, because the embedding
    /// dimension is part of blocker (i). **No fabricated embedding is ever
    /// emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the deferred
    ///   encoder forward.
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(forward_loud_partial(
            "embed",
            "pooled clip embedding",
            false,
        ))
    }

    /// Runs the Discogs tagging head over a mono `f32` PCM slice, returning one
    /// logit per label.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the same three encoder
    /// blockers as [`Self::encode`] **plus** a fourth that is specific to this
    /// surface: the artifact carries no label *names*, so even a working head
    /// could not map logits to human-readable Discogs genre / mood / instrument
    /// / era tags. The label **count** is recoverable — see [`Self::label_count`]
    /// — but it is read off the head tensor's dims, never assumed. **No
    /// fabricated logits are ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the deferred
    ///   encoder forward and the absent label taxonomy.
    pub fn tag(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(forward_loud_partial("tag", "Discogs tag logits", true))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Strict `vokra.model.arch` verification.
///
/// Refuses a foreign GGUF loudly, naming **both** the found and the expected tag
/// and enumerating the SSL audio/music-embedding neighbourhood plus the wav2vec2
/// lineage, so a reader who handed the wrong artifact over knows immediately
/// which loader they wanted (FR-EX-08 — never a silent misroute).
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "maest: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced by \
             `vokra-cli convert --model maest-30s-pw-129e`?). MAEST is an AST-backbone \
             self-supervised MUSIC tagger pretrained on the MTG Discogs4All dataset. The \
             confusable neighbours: `ast` shares the very same Audio Spectrogram Transformer \
             backbone but is SUPERVISED, fine-tuned on AudioSet, general-audio, and carries a \
             different label taxonomy — backbone identity is not topology identity; `atst` \
             (BYOL-style teacher-student patchout), `beats` (iterative acoustic tokenizer + \
             masked acoustic modelling), `eat` (utterance-level MAE with inverse block \
             masking), `dasheng` (universal MAE), `m2d` (masked modelling duo, dual online + \
             target branch), `mert` (HuBERT-derived masked prediction, music), `muq` (Mel-RVQ \
             + BEATs teacher, music) differ in the pre-training objective that shapes the \
             topology; `yamnet` and `panns` are supervised audio-tagging CNNs, not SSL at all; \
             `clap` bolts a text tower on for contrastive pretraining; and the wav2vec2 \
             lineage (`hubert`, `wav2vec2_ctc`, `wavlm_sv`, `emotion2vec`) sits on a \
             raw-waveform 1-D conv stem rather than a log-mel patch grid. Binding any of them \
             here would walk a foreign topology over a MAEST payload (FR-EX-08 — no silent \
             partial load). Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "maest: GGUF is missing `vokra.model.arch` — this is not a Vokra-native maest \
             GGUF (was it produced by `vokra-cli convert --model maest-30s-pw-129e`?). \
             Refusing to guess the arch from the tensor manifest (FR-EX-08). Primary source: \
             {PRIMARY_SOURCE_UPSTREAM_HF}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Maest::encode`] / [`Maest::embed`] / [`Maest::tag`] until the MAEST forward
/// wave lands.
///
/// `surface` is the method name, `output` is what that method would have
/// returned, and `tag_head` adds the taxonomy blocker that only the tag surface
/// hits. The message names the blockers and cites both primary sources so a
/// reader diagnosing the gap has fully specified anchors (`atst` / `m2d` /
/// `emotion2vec` / `wavlm` / `panns` / `redimnet` loud-partial-message
/// precedent — CLAUDE.md 教訓 (a)).
fn forward_loud_partial(surface: &str, output: &str, tag_head: bool) -> VokraError {
    // Bound to a `let` rather than nested inside the outer `format!` args
    // (clippy: no `format!` inside another `format!`'s arguments).
    let taxonomy_blocker = if tag_head {
        "(iv) NO LABEL TAXONOMY — the converter stamps no label list and no label count, so \
         even a working head could not map logits onto human-readable Discogs genre / mood / \
         instrument / era tags; the label COUNT is recoverable via `Maest::label_count` by \
         reading the head projection's leading dimension off disk, but the NAMES are \
         unrecoverable from the artifact and this module deliberately hardcodes no taxonomy \
         size; "
    } else {
        ""
    };

    VokraError::UnsupportedOp(format!(
        "maest {surface} (loud-partial): the MAEST encoder forward is deferred, so no \
         {output} can be produced. These pieces must land first: \
         (i) NO `vokra.maest.*` AXIS CHUNK GROUP — the converter \
         (`vokra-cli convert --model maest-30s-pw-129e`) is a verbatim F32/F16/BF16 \
         pass-through that stamps only `vokra.model.*` + `vokra.provenance.*`, so the \
         embedding width, depth, head count, patch grid AND every log-mel front-end axis \
         (sample rate, n_fft, hop, n_mels) are absent from the artifact and are not \
         transcribed anywhere in-repo; fabricating them from 'typical AST-base' numbers would \
         bind shape-valid garbage; \
         (ii) NO ViT-STYLE ENCODER PRIMITIVE — the log-mel front-end does exist \
         (`vokra_ops::mel`, `vokra_ops::fused_logmel`, `vokra_ops::kaldi_fbank`), but the 2-D \
         patch embedding + plain pre-norm Transformer encoder does not: \
         `vokra_ops::conformer`, `vokra_ops::ebranchformer` and `vokra_ops::zipformer` are \
         conv-augmented ASR encoders over a 1-D frame sequence, not ViT patch encoders over a \
         2-D mel plane. This is the SAME SHARED GAP the sibling `atst` / `eat` / `m2d` / \
         `dasheng` binders hit — one follow-up primitive unblocks the whole SSL fleet, not \
         five unrelated ones; \
         (iii) NO VERIFIED TENSOR-NAME MANIFEST — the converter copies every float tensor \
         under its verbatim upstream `state_dict` name and the only MAEST names recorded \
         in-repo are the two samples in the converter's own test module, so walking guessed \
         names into typed slots would bind shape-valid garbage; \
         {taxonomy_blocker}\
         Primary sources: upstream release {upstream}, paper (Alonso-Jiménez et al., ISMIR \
         2023) {paper}. The runtime cannot fabricate {output} (FR-EX-08 — no silent partial \
         output; CLAUDE.md 教訓 (a) 'loud-partial は fake-complete より honest').",
        upstream = PRIMARY_SOURCE_UPSTREAM_HF,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the MAEST runtime binder — contract-constant pins against the
    //! converter, metadata round-trip, loud negative space on every stated
    //! blocker, arch-tag distinctness, and the read-not-guessed label count.
    //!
    //! # What "round-trip" means here
    //!
    //! On real audio this would be `encode(...)` returning hidden states, but
    //! the MAEST forward is loud-partial (see the module doc). Fabricating an
    //! output would violate CLAUDE.md 教訓 (a)
    //! (「loud-partial は fake-complete より honest」). The round-trips we *can*
    //! honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX` and the two metadata keys match
    //!    the converter exactly, so a converter drift without a binder-side
    //!    follow-through fails here.
    //! 2. **Metadata round-trip** — a synthetic GGUF shaped like the
    //!    converter's output binds, and every stamp reads back.
    //! 3. **Loud negative space** — missing arch, foreign arch, empty manifest,
    //!    missing tensor, wrong dims, and all three forward surfaces fire at
    //!    their documented surface point in their documented variant.
    //! 4. **Arch distinctness pin** — the tag differs from every sibling arch,
    //!    `ast` included.
    //! 5. **Label count is data, not a constant** — two synthetic artifacts
    //!    with different head widths report different counts, which is only
    //!    possible if the value is read rather than hardcoded.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Tensor-name samples shaped like the converter's own test module's
    /// "realistic upstream state-dict name" choices (the HF
    /// `ASTForAudioClassification` wrapper places the body under an
    /// `audio_spectrogram_transformer.` prefix). Unverified against a real
    /// checkpoint — used here only to give the manifest something to hold.
    const SAMPLE_TENSORS: [(&str, [u64; 2]); 3] = [
        (
            "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
            [4, 12],
        ),
        (
            "audio_spectrogram_transformer.encoder.layer.0.output.dense.weight",
            [4, 6],
        ),
        ("audio_spectrogram_transformer.layernorm.weight", [4, 1]),
    ];

    /// Builds a GGUF shaped like `convert_maest_file`'s output: arch + name +
    /// category + upstream HF slug, an optional weight-licence class, an
    /// optional FR-MD-09 attribution stamp, the sample tensor manifest, and —
    /// when `head_labels` is `Some(n)` — a tagging head of `n` labels over a
    /// width-4 hidden, shaped like the HF `ASTForAudioClassification` wrapper
    /// (a 1-D LayerNorm pair plus the 2-D `[n, 4]` projection and its 1-D bias).
    fn maest_builder(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
        head_labels: Option<u64>,
    ) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            "mtg-upf/discogs-maest-30s-pw-129e (test)",
        );
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        }
        if attribution {
            b.add_string(
                chunks::KEY_PROVENANCE_ATTRIBUTION,
                "MAEST (mtg-upf/discogs-maest-30s-pw-129e) weights, licensed CC BY-NC-SA 4.0.",
            );
        }
        if with_tensors {
            for (name, dims) in SAMPLE_TENSORS {
                let elems = dims[0] * dims[1];
                b.add_tensor(
                    name,
                    GgmlType::F32,
                    dims.to_vec(),
                    vec![0u8; (elems * 4) as usize],
                )
                .expect("add_tensor");
            }
        }
        if let Some(labels) = head_labels {
            // A PyTorch `nn.Linear` weight is `[out_features, in_features]`, so
            // the leading dim is the label-set size. The LayerNorm siblings are
            // 1-D and must NOT be mistaken for the projection.
            b.add_tensor(
                "classifier.layernorm.weight",
                GgmlType::F32,
                vec![4],
                vec![0u8; 16],
            )
            .expect("add_tensor");
            b.add_tensor(
                "classifier.dense.weight",
                GgmlType::F32,
                vec![labels, 4],
                vec![0u8; (labels * 4 * 4) as usize],
            )
            .expect("add_tensor");
            b.add_tensor(
                "classifier.dense.bias",
                GgmlType::F32,
                vec![labels],
                vec![0u8; (labels * 4) as usize],
            )
            .expect("add_tensor");
        }
        b
    }

    /// Parses a `maest_builder` result into a `GgufFile`.
    fn maest_gguf(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
        head_labels: Option<u64>,
    ) -> GgufFile {
        let b = maest_builder(weight_license_class, attribution, with_tensors, head_labels);
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        // Mirrors of `crates/vokra-convert/src/models/maest.rs`. A converter
        // drift without a binder-side follow-through lands here in the same
        // commit or fails this test.
        assert_eq!(ARCH, "maest", "arch tag pin");
        assert_eq!(NAME, "maest-30s-pw-129e", "canonical variant name pin");
        assert_eq!(
            CATEGORY, "music-embedding",
            "category pin — MAEST is music-domain, NOT the general `audio-tagging` bucket"
        );
        assert_eq!(
            UPSTREAM_HF, "mtg-upf/discogs-maest-30s-pw-129e",
            "upstream HF slug pin"
        );
        assert_eq!(
            DEFAULT_LICENSE_SPDX, "cc-by-nc-sa-4.0",
            "T4 tier + ShareAlike cascade"
        );
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_HF,
            "vokra.provenance.upstream_hf"
        );
        assert_eq!(
            UPSTREAM_PARAM_COUNT_F32, 86_858_128,
            "HF API `parameters.F32` as recorded by the converter on 2026-08-13"
        );

        // The weight SPDX must resolve to the class the converter stamps, and
        // that class must carry all three obligations.
        assert_eq!(
            LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
            LicenseClass::NonCommercialShareAlike,
            "cc-by-nc-sa-4.0 must classify as NonCommercialShareAlike"
        );
        assert!(
            !LicenseClass::NonCommercialShareAlike.commercial_ok(),
            "NC: commercial use forbidden"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_license_preserved(),
            "SA: share-alike cascade on any downstream distribution"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_attribution(),
            "BY: attribution cascade"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_research_flag(),
            "T4 tier must be gated at the M2-13 compliance gate"
        );
        assert!(
            !LicenseClass::NonCommercialShareAlike.redistributable(),
            "T4 tier must be refused at publish without an explicit opt-in"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Arch-tag distinctness pin
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_encoder_arches() {
        // Every sibling below is a real converter arch tag. Sharing one would
        // let runtime dispatch bind a foreign topology over a MAEST payload
        // (FR-EX-08).
        for sibling in [
            // Same AST backbone, different objective / domain / taxonomy.
            "ast",
            // SSL audio/music-embedding neighbourhood.
            "atst",
            "beats",
            "eat",
            "dasheng",
            "m2d",
            "mert",
            "muq",
            // Supervised tagging CNNs / contrastive text-audio.
            "yamnet",
            "panns",
            "clap",
            // wav2vec2 lineage — raw-waveform 1-D conv stem, not a log-mel
            // patch grid.
            "hubert",
            "wav2vec2_ctc",
            "wavlm_sv",
            "emotion2vec",
        ] {
            assert_ne!(
                ARCH, sibling,
                "maest must not share an arch tag with `{sibling}` — a different objective, \
                 domain or head means a different topology (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Metadata round-trip on a synthetic converter-shaped GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_a_synthetic_converter_shaped_gguf() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            true,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("a converter-shaped GGUF must bind");

        // Metadata surfaces round-trip.
        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_hf(), Some(UPSTREAM_HF));
        assert_eq!(m.model_id(), Some(NAME));
        assert!(m.source().is_some(), "provenance source must surface");

        // Tensor manifest.
        assert_eq!(m.tensor_count(), SAMPLE_TENSORS.len());
        assert_eq!(m.weights().tensor_names().len(), SAMPLE_TENSORS.len());
        // Dims round-trip verbatim and are NOT reversed by the writer/reader
        // pair — pinned with an asymmetric shape so a future reversal is caught.
        assert_eq!(
            m.weights().tensor_dims(
                "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight"
            ),
            Some([4usize, 12].as_slice())
        );

        // No head in this artifact: honest `None`, not a taxonomy fallback.
        assert!(!m.has_tag_head());
        assert_eq!(m.label_count(), None);

        // Licence + FR-MD-09 attribution surfaces.
        assert_eq!(m.weight_license(), LicenseClass::NonCommercialShareAlike);
        let attr = m.attribution().expect("attribution stamp must surface");
        assert!(
            attr.contains("CC BY-NC-SA 4.0"),
            "attribution text must name the licence: {attr}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Missing arch fails loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Maest::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native maest GGUF"),
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
    // 5. Foreign arch fails loud, naming BOTH tags
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // An `ast` GGUF handed to the MAEST binder by mistake — the sharpest
        // confusable in the fleet, because MAEST is literally built on the AST
        // backbone. A silent bind would look plausible right up until the
        // numbers (and the label taxonomy) are wrong: exactly the misroute
        // FR-EX-08 forbids.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "ast");
        b.add_string(chunks::KEY_MODEL_NAME, "ast-finetuned-audioset");
        b.add_tensor("ast.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Maest::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH the actual and the expected tag.
                assert!(
                    m.contains("`ast`"),
                    "message must name the arch actually found, got `{m}`"
                );
                assert!(
                    m.contains("`maest`"),
                    "message must name the expected arch, got `{m}`"
                );
                // The neighbourhood must be enumerated so the reader knows
                // which loader they actually wanted.
                for sibling in [
                    "atst",
                    "beats",
                    "eat",
                    "dasheng",
                    "m2d",
                    "mert",
                    "muq",
                    "yamnet",
                    "panns",
                    "clap",
                    "hubert",
                    "wav2vec2_ctc",
                    "wavlm_sv",
                    "emotion2vec",
                ] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` enumerated in the error: {m}"
                    );
                }
                assert!(
                    m.contains("backbone identity is not topology identity"),
                    "message must explain why sharing AST's backbone is not sharing its \
                     topology, got `{m}`"
                );
                assert!(
                    m.contains("Discogs"),
                    "message should state what makes MAEST distinct, got `{m}`"
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
    // 6. Empty tensor manifest fails loud (never binds an all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        // Correct arch + full metadata but zero tensors.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            false,
            None,
        );
        let Err(err) = Maest::from_gguf(&file) else {
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
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model maest-30s-pw-129e"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. require_tensor names the missing tensor
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");

        let missing = "audio_spectrogram_transformer.encoder.layer.11.output.dense.weight";
        let Err(err) = m.weights().require_tensor(missing) else {
            panic!("expected ModelLoad for a tensor that is not on disk");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(missing),
                    "message must name the missing tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("nearest names on disk"),
                    "message must offer the nearby names, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-zero-substitution clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // A tensor that IS on disk resolves.
        assert_eq!(
            m.weights()
                .require_tensor("audio_spectrogram_transformer.layernorm.weight")
                .expect("present tensor must resolve"),
            [4usize, 1].as_slice()
        );
    }

    // -----------------------------------------------------------------------
    // 8. require_tensor_dims names BOTH expected and actual dims
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_dims_names_expected_and_actual() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");
        let name = "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight";

        // Exact match passes.
        m.weights()
            .require_tensor_dims(name, &[4, 12])
            .expect("matching dims must pass");

        // Mismatch fails loud, naming both sides.
        let Err(err) = m.weights().require_tensor_dims(name, &[4, 36]) else {
            panic!("expected ModelLoad on a dims mismatch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("[4, 12]"),
                    "message must name the ACTUAL dims, got `{msg}`"
                );
                assert!(
                    msg.contains("[4, 36]"),
                    "message must name the EXPECTED dims, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-silent-reshape clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9. encode loud-partials, naming the missing primitive
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partials_naming_the_missing_primitive() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");

        // A legitimately shaped buffer, so the loud-partial gate is what fires
        // (not some pre-encode length validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("maest encode"), "surface must be named: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Blocker (i): the missing axis chunk group.
                assert!(
                    msg.contains("vokra.maest.*"),
                    "must name the absent axis chunk group: {msg}"
                );
                // Blocker (ii): the missing PRIMITIVE, named exactly, plus the
                // primitives that genuinely do exist.
                assert!(
                    msg.contains("patch embedding") && msg.contains("pre-norm Transformer"),
                    "must name the missing ViT patch-encoder primitive: {msg}"
                );
                for present in [
                    "vokra_ops::mel",
                    "vokra_ops::fused_logmel",
                    "vokra_ops::kaldi_fbank",
                ] {
                    assert!(
                        msg.contains(present),
                        "must name the front-end primitive `{present}` that DOES exist: {msg}"
                    );
                }
                for absent in [
                    "vokra_ops::conformer",
                    "vokra_ops::ebranchformer",
                    "vokra_ops::zipformer",
                ] {
                    assert!(
                        msg.contains(absent),
                        "must explain why `{absent}` is not a substitute: {msg}"
                    );
                }
                // The gap must be described as SHARED across the SSL fleet so a
                // follow-up wave sees one primitive, not five models.
                assert!(
                    msg.contains("SAME SHARED GAP"),
                    "must flag the shared SSL-fleet gap: {msg}"
                );
                for fleet_sibling in ["atst", "eat", "m2d", "dasheng"] {
                    assert!(
                        msg.contains(fleet_sibling),
                        "must name fleet sibling `{fleet_sibling}` sharing the gap: {msg}"
                    );
                }
                // Blocker (iii): no verified tensor-name manifest.
                assert!(
                    msg.contains("state_dict"),
                    "must name the unverified tensor-name manifest: {msg}"
                );
                // The taxonomy blocker belongs to `tag`, not to `encode`.
                assert!(
                    !msg.contains("NO LABEL TAXONOMY"),
                    "encode must not claim a taxonomy blocker it does not hit: {msg}"
                );

                // Both primary sources.
                for url in [PRIMARY_SOURCE_UPSTREAM_HF, PRIMARY_SOURCE_PAPER] {
                    assert!(msg.contains(url), "expected primary source `{url}`: {msg}");
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10. embed loud-partials on the same blockers
    // -----------------------------------------------------------------------

    #[test]
    fn embed_loud_partials_on_the_same_blockers() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");

        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.embed(&pcm) else {
            panic!("embed must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("maest embed"), "surface must be named: {msg}");
                assert!(
                    msg.contains("pooled clip embedding"),
                    "must name the output it refuses to fabricate: {msg}"
                );
                assert!(
                    msg.contains("vokra.maest.*"),
                    "must name the absent axis chunk group: {msg}"
                );
                assert!(
                    msg.contains("patch embedding") && msg.contains("pre-norm Transformer"),
                    "must name the missing primitive: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. tag loud-partials on the encoder gap AND the absent taxonomy
    // -----------------------------------------------------------------------

    #[test]
    fn tag_loud_partials_naming_the_primitive_and_the_absent_taxonomy() {
        // Give this artifact a head, so the refusal is unambiguously about the
        // deferred forward rather than about a missing classifier.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            Some(11),
        );
        let m = Maest::from_gguf(&file).expect("bind");
        assert!(m.has_tag_head(), "fixture must carry a head");

        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.tag(&pcm) else {
            panic!("tag must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("maest tag"), "surface must be named: {msg}");
                assert!(
                    msg.contains("Discogs tag logits"),
                    "must name the output it refuses to fabricate: {msg}"
                );
                // The shared encoder primitive gap.
                assert!(
                    msg.contains("patch embedding") && msg.contains("pre-norm Transformer"),
                    "must name the missing primitive: {msg}"
                );
                assert!(
                    msg.contains("SAME SHARED GAP"),
                    "must flag the shared SSL-fleet gap: {msg}"
                );
                // The tag-only fourth blocker.
                assert!(
                    msg.contains("NO LABEL TAXONOMY"),
                    "tag must name the absent label taxonomy: {msg}"
                );
                assert!(
                    msg.contains("Maest::label_count"),
                    "must point at the read-from-disk label count: {msg}"
                );
                assert!(
                    msg.contains("hardcodes no taxonomy size"),
                    "must state that no taxonomy size is hardcoded: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. The label count is READ from the artifact, never a constant
    // -----------------------------------------------------------------------

    #[test]
    fn label_count_is_read_from_disk_never_a_constant() {
        // Two artifacts, two different head widths. A hardcoded taxonomy size
        // could not satisfy both, so this test is what makes "read, never
        // guessed" mechanically true rather than merely documented.
        for labels in [7u64, 13] {
            let file = maest_gguf(
                Some(LicenseClass::NonCommercialShareAlike),
                false,
                true,
                Some(labels),
            );
            let m = Maest::from_gguf(&file).expect("bind");
            assert!(m.has_tag_head(), "head must be discovered for {labels}");
            assert_eq!(
                m.label_count(),
                Some(labels as usize),
                "label count must track the head projection's leading dim on disk"
            );
            // The head tensors are reported verbatim, LayerNorm siblings
            // included — three under the prefix in this fixture.
            assert_eq!(m.weights().tag_head_tensors().len(), 3);
        }
    }

    #[test]
    fn label_count_is_none_without_a_head() {
        // A bare-encoder export is legitimate: no head, no count, no error, and
        // above all no fallback taxonomy number.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");
        assert!(!m.has_tag_head());
        assert!(m.weights().tag_head_tensors().is_empty());
        assert_eq!(
            m.label_count(),
            None,
            "no head on disk must yield None, never a guessed taxonomy size"
        );
    }

    #[test]
    fn label_count_is_none_when_the_head_layout_is_ambiguous() {
        // Two 2-D tensors under the head prefix: which one is the label
        // projection cannot be decided from shape alone, so report nothing.
        let mut b = maest_builder(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            Some(9),
        );
        b.add_tensor(
            "classifier.extra_projection.weight",
            GgmlType::F32,
            vec![5, 4],
            vec![0u8; 80],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let m = Maest::from_gguf(&file).expect("bind");

        assert!(m.has_tag_head());
        assert_eq!(
            m.label_count(),
            None,
            "an ambiguous head layout must report None rather than pick a dim"
        );
    }

    // -----------------------------------------------------------------------
    // 13. Missing licence stamp fails closed to Unknown
    // -----------------------------------------------------------------------

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        // No provenance licence stamp at all: the binder still binds (arch +
        // manifest are the load gates), but the licence surface must fail
        // closed.
        let file = maest_gguf(None, false, true, None);
        let m = Maest::from_gguf(&file).expect("arch + manifest are the load gates");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an absent weight-licence stamp must fail closed to Unknown"
        );
        assert!(m.attribution().is_none(), "no stamp => no attribution");
        assert!(
            LicenseClass::Unknown.requires_research_flag(),
            "Unknown must be gated at the M2-13 compliance gate"
        );
    }

    // -----------------------------------------------------------------------
    // 14. Compliance gate: NonCommercialShareAlike is REFUSED under strict
    // -----------------------------------------------------------------------

    #[test]
    fn compliance_gate_refuses_non_commercial_under_strict_and_allows_research_opt_in() {
        let stamped = maest_builder(
            Some(LicenseClass::NonCommercialShareAlike),
            true,
            true,
            None,
        )
        .to_bytes()
        .expect("serialize");

        // Strict refuses — MAEST is T4, so this refusal is the fail-closed
        // default working as intended, NOT a bug.
        let Err(err) = Maest::from_gguf_with_policy(&stamped, &CompliancePolicy::strict()) else {
            panic!("a cc-by-nc-sa-4.0 artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for a NonCommercialShareAlike weight, got {err:?}"
        );

        // An explicit research opt-in unlocks it (and emits the mandatory
        // research-only warning inside the gate).
        let research = CompliancePolicy::strict().with_research_license(true);
        let m = Maest::from_gguf_with_policy(&stamped, &research)
            .expect("the research opt-in must unlock a T4 weight");
        assert_eq!(m.weight_license(), LicenseClass::NonCommercialShareAlike);

        // An unstamped artifact resolves to Unknown and is refused too —
        // fail-closed, never a silent substitution.
        let unstamped = maest_builder(None, false, true, None)
            .to_bytes()
            .expect("serialize");
        let Err(err) = Maest::from_gguf_with_policy(&unstamped, &CompliancePolicy::strict()) else {
            panic!("an unstamped artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for an Unknown weight class, got {err:?}"
        );

        // The gate must not mask an arch mismatch: a foreign artifact reports
        // the arch, which is the actionable fact.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "mert");
        b.add_tensor("mert.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let foreign = b.to_bytes().expect("serialize");
        let Err(err) = Maest::from_gguf_with_policy(&foreign, &CompliancePolicy::strict()) else {
            panic!("a foreign arch must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("`mert`") && msg.contains("`maest`"),
                "arch mismatch must be reported ahead of any licence verdict: {msg}"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
