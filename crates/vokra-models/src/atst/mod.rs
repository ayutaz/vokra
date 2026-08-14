//! **ATST** — "Audio Teacher-Student Transformer" (`Audio-WestlakeU/audiossl`,
//! code MIT / **weight CC-BY-4.0**) — runtime binder for the `atst` converter
//! arch (Wave C2 2026-08-15, loud-partial per the `emotion2vec` / `wavlm` /
//! `panns` / `redimnet` / `storm` precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/atst.rs` has been stamping
//! `vokra.model.arch = "atst"` since the 2026-08-13 SSL-encoder wave, but a
//! workspace-wide grep proved that **nothing read that arch string back** — a
//! converted ATST checkpoint was unloadable. This module is that consumer.
//!
//! # Primary sources
//!
//! Every fact below is transcribed from the converter's own module docstring
//! (`crates/vokra-convert/src/models/atst.rs`), which is this repository's
//! primary-source record for ATST. Nothing here is re-derived from memory.
//!
//! - Upstream tree: <https://github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst>
//! - Paper (utterance-level, this [`NAME`]): Li et al. 2022, INTERSPEECH,
//!   *"ATST: Audio Representation Learning with Teacher-Student Transformer"*
//!   — <https://arxiv.org/abs/2204.12076>
//! - Paper (frame-level `atst-frame` sibling): Li et al. 2023, TASLP —
//!   <https://arxiv.org/abs/2306.04186>
//! - Licence split (upstream `LICENSE` file text, quoted verbatim by the
//!   converter): *"The pretrained checkpoints hyper-linked in this repo are
//!   licensed under CC BY 4.0. … audiossl is licenced under MIT Licence."*
//!   Vokra stamps the **weight** tier, so [`DEFAULT_LICENSE_SPDX`] is
//!   `cc-by-4.0` → [`LicenseClass::AttributionRequired`].
//!
//! # What ATST is (and what this binder therefore exposes)
//!
//! ATST is a **feature extractor**, not an end-task model: it is a
//! self-supervised audio encoder trained with a BYOL-style EMA
//! teacher + student-patchout objective over a log-mel spectrogram, and it
//! maps audio to a sequence of hidden states (plus, for the utterance-level
//! 2022 variant this [`NAME`] tracks, a pooled utterance embedding).
//!
//! **The checkpoint carries no task head.** Downstream sound-event-detection
//! / audio-tagging / speaker heads are trained separately by consumers and are
//! not part of the ATST release, so this module deliberately exposes no
//! classifier and invents no class-label list.
//!
//! ```text
//! PCM (mono f32)
//!   -> log-mel spectrogram front-end                    ← **loud-partial**
//!        (axes — sample rate / n_fft / hop / n_mels — are NOT stamped by
//!         the converter and are not transcribed anywhere in-repo; see
//!         blocker (i)).
//!   -> 2-D patch embedding over the mel plane           ← **loud-partial**
//!        (ViT-style patchification; the patch grid is part of blocker (i)).
//!   -> pre-norm Transformer encoder (~86M-param base)   ← **loud-partial**
//!        (blocker (ii): `vokra-ops` has no ViT-style plain pre-norm
//!         Transformer encoder primitive today).
//!   -> per-patch hidden states     ── [`Atst::encode`]
//!   -> pooled utterance embedding  ── [`Atst::embed`]
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real (this WP)** — everything around the forward:
//!
//! - [`Atst::from_gguf`] with **strict** `vokra.model.arch == "atst"`
//!   verification. A sibling SSL-encoder GGUF handed here by mistake fails
//!   with a message naming **both** tags and enumerating the whole
//!   audio/music-embedding neighbourhood (FR-EX-08 — never a silent
//!   misroute into a foreign topology).
//! - [`AtstWeights::from_gguf`] tensor-manifest binding over the verbatim
//!   upstream `state_dict` names the converter passes through, with a
//!   non-empty gate plus [`AtstWeights::require_tensor`] /
//!   [`AtstWeights::require_tensor_dims`] lookups that name the missing
//!   tensor, or **both** the expected and the actual dims.
//! - Metadata surfacing: [`Atst::name`] / [`Atst::category`] /
//!   [`Atst::upstream_url`] read back the converter's stamps.
//! - Weight-licence + FR-MD-09 attribution surfacing, fail-closing to
//!   [`LicenseClass::Unknown`] when the artifact carries no stamp, and the
//!   compliance-gated [`Atst::from_gguf_with_policy`] / [`Atst::from_path`]
//!   entry points.
//!
//! **Loud-partial (this WP)** — [`Atst::encode`] and [`Atst::embed`] return
//! [`VokraError::UnsupportedOp`] naming **four** blockers:
//!
//! 1. **No `vokra.atst.*` axis chunk group.** The converter stamps
//!    `vokra.model.*` + `vokra.provenance.*` and nothing else — it is a
//!    verbatim BF16/F16/F32 pass-through with no config transcription step.
//!    Embedding width, depth, head count, patch grid, and every log-mel
//!    front-end axis are therefore **absent from the artifact** and are not
//!    transcribed anywhere in this repository. Upstream keeps them in the
//!    training argparse namespace inside the `.ckpt`, which the no-pickle
//!    rule (FR-LD-05 / NFR-DS-02) keeps out of this runtime. Fabricating
//!    them from "typical ViT-base" numbers would bind shape-valid garbage.
//! 2. **No ViT-style encoder primitive in `vokra-ops`.** The log-mel
//!    front-end genuinely exists (`vokra_ops::mel`, `vokra_ops::fused_logmel`,
//!    `vokra_ops::kaldi_fbank`), but the 2-D patch embedding + plain pre-norm
//!    Transformer encoder does not: `vokra_ops::conformer`,
//!    `vokra_ops::ebranchformer` and `vokra_ops::zipformer` are all
//!    conv-augmented **ASR** encoders over a 1-D frame sequence, not ViT
//!    patch encoders over a 2-D mel plane.
//! 3. **Teacher/student branch selection is unresolved.** A BYOL-style EMA
//!    checkpoint carries **both** branches. Picking the wrong one yields a
//!    shape-valid but numerically different embedding — a silent misroute of
//!    exactly the kind FR-EX-08 forbids — so the branch that upstream's own
//!    inference entry point uses must be read off the upstream tree before a
//!    forward may run. [`AtstBranch`] + [`Atst::branch_tensor_count`] exist
//!    today only as *diagnostics* over what is actually on disk; they gate
//!    nothing.
//! 4. **No verified tensor-name manifest.** The converter copies every float
//!    tensor under its verbatim upstream `state_dict` name, and nothing
//!    in-repo transcribes ATST's naming. Walking guessed names into typed
//!    slots would bind the wrong tensors without failing.
//!
//! No fabricated hidden states or embeddings are ever emitted (FR-EX-08 — no
//! silent partial output). The scaffold is arranged so a follow-up wave flips
//! the switch by (a) teaching the converter to stamp a `vokra.atst.*` axis
//! group, (b) adding a ViT patch-encoder primitive to `vokra-ops`, and
//! (c) transcribing the upstream tensor names — with the binder, the manifest
//! lookups and the tests already in place.
//!
//! # Sibling family distinctness (SSL audio/music-embedding neighbourhood)
//!
//! [`ARCH`] = `"atst"` is deliberately distinct from every sibling. All of
//! these are self-supervised audio encoders that emit hidden states, and all
//! of them differ in the pre-training objective that shapes the topology:
//!
//! - `beats` — iterative acoustic tokenizer + masked acoustic modelling;
//! - `eat` — utterance-level MAE with efficient inverse block masking;
//! - `dasheng` — universal MAE;
//! - `m2d` — masked modelling **duo** (dual online + target branch);
//! - `maest` — AST backbone with a Discogs music-tagger SSL objective;
//! - `mert` — HuBERT-derived masked prediction (music);
//! - `muq` — Mel-RVQ + BEATs teacher (music);
//! - `yamnet` — supervised audio-tagging CNN, not SSL at all;
//! - `hubert` / `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec` — the
//!   wav2vec2 lineage, whose encoders sit on a **raw-waveform 1-D conv
//!   stem** rather than a log-mel patch grid.
//!
//! Sharing an arch tag would let runtime dispatch bind, say, an MAE decoder
//! or a raw-waveform conv stem over a teacher-student log-mel checkpoint
//! (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's constants** —
//! the same rule every sibling binder (`emotion2vec` / `wavlm` / `panns` /
//! `redimnet` / `canary_1b_flash`) follows so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # Licence posture
//!
//! The converter stamps `cc-by-4.0` → [`LicenseClass::AttributionRequired`],
//! which is commercially permitted, so a correctly stamped artifact loads
//! under [`CompliancePolicy::strict`] without a research opt-in — but CC-BY
//! 4.0 obliges a downstream to **display attribution**, so
//! [`Atst::attribution`] surfaces the stamped text rather than burying it.
//! An unstamped artifact resolves to [`LicenseClass::Unknown`] and is refused
//! by the gate (fail-closed, never a silent substitution). This binder only
//! *surfaces* whatever class the artifact carries;
//! `docs/license-audit.md` §3.1 sign-off stays **blank** (owner-only per
//! memory `[[feedback-license-signoff-primary-source]]` — CC does not sign,
//! and does not treat a converter default as a sign-off).
//!
//! # No ONNX / no pickle (permanent)
//!
//! ATST ships upstream as a PyTorch `.ckpt` pickle; this runtime **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The bridge is an offline
//! uv-managed Python 3.12 sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), exactly as the converter docstring specifies.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/atst.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model atst-base`.
///
/// Distinct from every sibling SSL audio/music-embedding arch tag (`beats` /
/// `eat` / `dasheng` / `m2d` / `maest` / `mert` / `muq` / `yamnet`) and from
/// the wav2vec2 lineage (`hubert` / `wav2vec2_ctc` / `wavlm_sv` /
/// `emotion2vec`): ATST's BYOL-style teacher-student patchout objective over
/// a log-mel patch grid is a distinct topology axis from all of them, and
/// silently sharing a tag would misroute runtime dispatch (FR-EX-08 — see the
/// module docstring "Sibling family distinctness" section).
pub const ARCH: &str = "atst";

/// Expected `vokra.model.name` value written by the converter — the canonical
/// `atst-base` size point (the utterance-level INTERSPEECH 2022 release).
///
/// The frame-level TASLP 2023 sibling `atst-frame` is published by the
/// converter under its own `NAME` following the `snac_24khz` / `snac_44khz`
/// pattern, so this value is **surfaced, not gated** — see [`Atst::name`].
pub const NAME: &str = "atst-base";

/// Expected `vokra.model.category` value — `audio-embedding`, shared with the
/// sibling general-audio SSL encoders (`beats` / `eat` / `dasheng` / `m2d`).
/// Consumed by the model-card generator and the zoo-manifest tier gate so a
/// feature extractor is never advertised as an ASR / TTS release.
pub const CATEGORY: &str = "audio-embedding";

/// Upstream source tree. ATST is **not** hosted on HuggingFace, so the
/// converter stamps [`GGUF_KEY_PROVENANCE_UPSTREAM_URL`] rather than
/// `vokra.provenance.upstream_hf` (the sibling `beats` / `eat` / `nsnet2`
/// posture); the model-card generator picks up either.
pub const UPSTREAM_URL: &str =
    "github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst";

/// Default SPDX stamped by the converter — the **weight** tier.
///
/// The upstream `LICENSE` file separates code (`mit`) from pretrained
/// checkpoints (`cc-by-4.0`); `vokra.provenance.weight_license` tracks the
/// weight, so this resolves to [`LicenseClass::AttributionRequired`]. A
/// caller with a different attestation may override at the converter boundary
/// (`--license <spdx>`), which is why this binder *surfaces* rather than
/// *asserts* the class.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-4.0";

/// Metadata key holding [`CATEGORY`] (not part of `vokra_core::gguf::chunks`,
/// so mirrored here from the converter).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key holding [`UPSTREAM_URL`] (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// Primary-source anchors, cited inside the loud-partial error so a reader
// diagnosing the gap has fully specified places to walk.

/// Primary-source anchor: the upstream ATST source tree.
pub const PRIMARY_SOURCE_UPSTREAM: &str = UPSTREAM_URL;
/// Primary-source anchor: Li et al. 2022 (INTERSPEECH) — the utterance-level
/// ATST this [`NAME`] tracks.
pub const PRIMARY_SOURCE_PAPER_2022: &str = "arxiv.org/abs/2204.12076";
/// Primary-source anchor: Li et al. 2023 (TASLP) — the frame-level
/// `atst-frame` extension.
pub const PRIMARY_SOURCE_PAPER_2023: &str = "arxiv.org/abs/2306.04186";

// ---------------------------------------------------------------------------
// AtstBranch — the BYOL duo selector (diagnostic only, gates nothing)
// ---------------------------------------------------------------------------

/// Which branch of the BYOL-style teacher-student duo a caller is asking
/// about.
///
/// ATST is trained with an EMA **teacher** tracking a **student**, so a
/// released checkpoint can carry both sets of weights. Which one upstream's
/// own inference entry point uses is **not** recorded anywhere in this
/// repository, and choosing wrongly produces a shape-valid but numerically
/// different embedding — a silent misroute. That is blocker (iii) of the
/// [`Atst::encode`] loud-partial.
///
/// Consequently this enum is a **diagnostic** only: [`Atst::branch_tensor_count`]
/// counts tensors whose name starts with [`Self::prefix`] so an operator can
/// see what a given artifact actually contains. It gates nothing, and a
/// checkpoint carrying neither prefix is **not** rejected — the prefix
/// convention below is transcribed from the converter's own test module and
/// has not been verified against a real upstream checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtstBranch {
    /// The student branch — the one that receives gradient updates and takes
    /// the patchout augmentation during training.
    Student,
    /// The teacher branch — the exponential-moving-average copy of the
    /// student.
    Teacher,
}

impl AtstBranch {
    /// The `state_dict` name prefix associated with this branch.
    ///
    /// **Unverified convention.** These prefixes are transcribed from the
    /// sample tensor names in the converter's own test module
    /// (`student.encoder.blocks.0.norm1.weight` /
    /// `teacher.encoder.blocks.0.attn.qkv.weight`), which that module labels
    /// "realistic upstream state-dict name" — they have not been checked
    /// against a real ATST checkpoint. Nothing in this binder *depends* on
    /// them; they only shape the counts reported by
    /// [`Atst::branch_tensor_count`].
    #[inline]
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Student => "student.",
            Self::Teacher => "teacher.",
        }
    }

    /// Both branches, in a stable order, for callers that want to report on
    /// each in turn.
    #[inline]
    #[must_use]
    pub const fn all() -> [Self; 2] {
        [Self::Student, Self::Teacher]
    }
}

// ---------------------------------------------------------------------------
// AtstWeights — the tensor manifest, with loud lookups
// ---------------------------------------------------------------------------

/// Weight tensors bound from an ATST GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step.
/// A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an ~86M-parameter Transformer never
/// converts to an empty manifest, so zero tensors always signals a
/// mis-produced artifact, and binding it would silently run an all-zero
/// forward).
///
/// Under the current landing this struct stores the tensor names and their
/// GGUF-side dims. The payload is deliberately not dequantised: the forward is
/// loud-partial (see [`Atst::encode`]), and the follow-up wave sizes its
/// dequant per its kernel needs. [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) are already in place so
/// that wave walks a manifest that fails loudly rather than substituting
/// zeros.
#[derive(Debug, Clone)]
pub struct AtstWeights {
    /// Tensors discovered on disk, in file order, as
    /// `(upstream state_dict name, GGUF-side dims)`.
    tensors: Vec<(String, Vec<usize>)>,
}

impl AtstWeights {
    /// Scans `gguf` for the ATST `state_dict` tensors.
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
                "atst: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate ATST checkpoint is an ~86M-parameter \
                 Transformer (arch={ARCH}, name={NAME}) and always converts to hundreds \
                 of Linear / LayerNorm tensors, so an empty manifest always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model atst-base` against \
                 an upstream `{UPSTREAM_URL}` checkpoint flattened to safetensors by the \
                 offline uv-managed Python 3.12 sidecar."
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
    /// A plain string count over what is actually on disk — it asserts
    /// nothing about the upstream naming convention. See
    /// [`AtstBranch::prefix`] for the (unverified) prefixes this pairs with.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
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
            "atst: required tensor `{name}` is absent from the GGUF ({count} tensors \
             present; nearest names on disk: {near:?}). The converter passes upstream \
             `state_dict` names through verbatim, so a mismatch means either the \
             checkpoint was flattened with a different prefix policy (e.g. a branch \
             prefix such as `{student}` / `{teacher}` was stripped) or the caller is \
             walking a manifest transcribed from a different ATST variant \
             (`atst-frame`, the frame-level TASLP 2023 sibling, is published under its \
             own name). Refusing to substitute a zero tensor (FR-EX-08). Primary \
             source: {UPSTREAM_URL}",
            count = self.tensors.len(),
            student = AtstBranch::Student.prefix(),
            teacher = AtstBranch::Teacher.prefix(),
        )))
    }

    /// Asserts that a required tensor is present **and** has exactly
    /// `expected` dims.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the tensor is absent (see
    ///   [`Self::require_tensor`]) or when its dims differ — the message names
    ///   **both** the expected and the actual dims (FR-EX-08 — never reshape
    ///   or truncate silently).
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "atst: tensor `{name}` has dims {actual:?} but the caller expects \
                 {expected:?} — refusing to reshape or truncate silently (FR-EX-08). \
                 ATST stamps no `vokra.atst.*` axis group, so a dims disagreement here \
                 means the walking code's transcribed topology does not match the \
                 payload (a different size point, or the frame-level `atst-frame` \
                 sibling). Primary source: {UPSTREAM_URL}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Atst — the runtime binder handle
// ---------------------------------------------------------------------------

/// ATST (Audio Teacher-Student Transformer) self-supervised audio encoder.
///
/// Bind with [`from_gguf`](Self::from_gguf) — or the compliance-gated
/// [`from_gguf_with_policy`](Self::from_gguf_with_policy) /
/// [`from_path`](Self::from_path) — then call [`encode`](Self::encode) for
/// per-patch hidden states or [`embed`](Self::embed) for the pooled utterance
/// embedding. Both forwards are **loud-partial** today; see the module
/// docstring for the four blockers and the FR-EX-08 contract.
#[derive(Debug, Clone)]
pub struct Atst {
    name: Option<String>,
    category: Option<String>,
    upstream_url: Option<String>,
    weights: AtstWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl Atst {
    /// Binds an ATST GGUF: verifies the arch strictly, binds the tensor
    /// manifest, and surfaces the converter's metadata + licence stamps.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key, so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// `vokra.model.name` is deliberately **surfaced, not gated**: the
    /// frame-level `atst-frame` sibling shares this arch under a different
    /// name, so a hard name check would make a legitimate future artifact
    /// unloadable. See [`Self::name`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not `"atst"` —
    ///   the message names both the found and the expected tag and enumerates
    ///   the SSL audio/music-embedding neighbourhood.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`AtstWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch first, so a mis-routed artifact reports the arch mismatch
        //    (the actionable fact) instead of a downstream missing-tensor
        //    trail.
        verify_arch(file)?;

        // 2. Metadata surfacing. Soft: a converter-produced artifact always
        //    carries these, but they are diagnostics, not load gates.
        let read_str = |key: &str| -> Option<String> {
            file.get(key).and_then(|v| v.as_str()).map(str::to_owned)
        };
        let name = read_str(chunks::KEY_MODEL_NAME);
        let category = read_str(GGUF_KEY_MODEL_CATEGORY);
        let upstream_url = read_str(GGUF_KEY_PROVENANCE_UPSTREAM_URL);

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = AtstWeights::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps `AttributionRequired`
        //    (cc-by-4.0); an artifact missing the stamp reads back as
        //    `Unknown` — fail-closed at the M2-13 gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let attribution = read_str(chunks::KEY_PROVENANCE_ATTRIBUTION);

        Ok(Self {
            name,
            category,
            upstream_url,
            weights,
            weight_license,
            attribution,
        })
    }

    /// Loads an ATST GGUF from raw bytes under `policy` (the M2-13
    /// weight-licence gate).
    ///
    /// ATST ships **CC-BY 4.0** → [`LicenseClass::AttributionRequired`],
    /// which is commercially permitted, so a correctly stamped artifact passes
    /// under [`CompliancePolicy::strict`] without a research opt-in. An
    /// artifact with no provenance stamp resolves to
    /// [`LicenseClass::Unknown`] and is refused — fail-closed, never a silent
    /// substitution.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on GGUF parse failure, or on a wrong /
    ///   missing `vokra.model.arch`.
    /// - `VokraError::ResearchLicenseRequired` from the compliance gate when
    ///   the weight class is gated and `policy` grants no research opt-in.
    /// - See [`Self::from_gguf`] for the remaining bind errors.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("atst GGUF: {e}")))?;
        // Arch before the compliance gate so a mis-routed artifact reports the
        // arch mismatch rather than a licence verdict about a model the caller
        // never meant to load.
        verify_arch(&file)?;
        check_weight_license(&file, policy)?;
        Self::from_gguf(&file)
    }

    /// Loads an ATST GGUF from a path under [`CompliancePolicy::strict`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }

    /// The stamped `vokra.model.name`, if present.
    ///
    /// [`NAME`] (`"atst-base"`) for the utterance-level 2022 release this
    /// module tracks; the frame-level `atst-frame` sibling shares [`ARCH`]
    /// under a different name, which is why this is surfaced rather than
    /// gated.
    #[inline]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, if present — [`CATEGORY`]
    /// (`"audio-embedding"`) for a converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_url`, if present —
    /// [`UPSTREAM_URL`] for a converter-produced artifact. ATST is not on
    /// HuggingFace, so there is no `upstream_hf` counterpart.
    #[inline]
    #[must_use]
    pub fn upstream_url(&self) -> Option<&str> {
        self.upstream_url.as_deref()
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &AtstWeights {
        &self.weights
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// How many tensors on disk carry `branch`'s name prefix.
    ///
    /// **Diagnostic only** — it gates nothing, and `0` for both branches is
    /// not an error (the prefix convention is unverified; see
    /// [`AtstBranch::prefix`]). It exists so an operator inspecting a
    /// converted BYOL checkpoint can see whether it carries one branch or
    /// both, which is blocker (iii) of the [`Self::encode`] loud-partial.
    #[inline]
    #[must_use]
    pub fn branch_tensor_count(&self, branch: AtstBranch) -> usize {
        self.weights.count_with_prefix(branch.prefix())
    }

    /// The weight-licence class surfaced from
    /// `vokra.provenance.weight_license`.
    ///
    /// [`LicenseClass::AttributionRequired`] for a correctly stamped ATST
    /// artifact (cc-by-4.0); [`LicenseClass::Unknown`] when the stamp is
    /// absent (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The FR-MD-09 attribution text stamped under
    /// `vokra.provenance.attribution`, if any.
    ///
    /// CC-BY 4.0 obliges a downstream to display attribution alongside output
    /// derived from the weights, so this is surfaced rather than buried: a
    /// consumer shipping ATST-derived embeddings must render this string.
    /// `None` means the artifact carries no stamp (for example it was
    /// converted with an explicit `--license` override, which suppresses the
    /// CC-BY wording).
    #[inline]
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }

    /// Encodes a mono `f32` PCM slice into the ATST encoder's **per-patch
    /// hidden states** (`[n_patches][embed_dim]`).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the four blockers
    /// enumerated in the module docstring: (i) no `vokra.atst.*` axis chunk
    /// group, (ii) no ViT-style patch-embedding + pre-norm Transformer
    /// encoder primitive in `vokra-ops`, (iii) unresolved teacher/student
    /// branch selection, (iv) no verified tensor-name manifest. All three
    /// primary sources are cited so a reader diagnosing the gap has exactly
    /// three places to walk. **No fabricated hidden states are ever emitted**
    /// (FR-EX-08 — no silent partial output).
    ///
    /// `pcm` is treated as mono `f32` in `[-1, 1]`. Its sample rate is
    /// deliberately **not** asserted: the required rate is part of blocker
    /// (i), and asserting a guessed rate would be exactly the fabrication
    /// this module refuses.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind explicitly so a future accidental removal of the parameter is
        // not masked by an unused-variable warning (mirror of the
        // emotion2vec / wavlm loud-partial signature discipline).
        let _ = pcm;
        Err(forward_loud_partial("encode", "per-patch hidden states"))
    }

    /// Encodes a mono `f32` PCM slice into the **pooled utterance embedding**
    /// the utterance-level ATST ([`NAME`]) defines.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] for the same four blockers as
    /// [`Self::encode`] — the pooled embedding is the encoder output reduced
    /// over the patch axis, so it cannot exist before the encoder does. The
    /// **width** of the returned vector is itself unknown, because the
    /// embedding dimension is part of blocker (i). **No fabricated embedding
    /// is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(forward_loud_partial("embed", "pooled utterance embedding"))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Strict `vokra.model.arch` verification.
///
/// Refuses a foreign GGUF loudly, naming **both** the found and the expected
/// tag and enumerating the SSL audio/music-embedding neighbourhood plus the
/// wav2vec2 lineage, so a reader who handed the wrong artifact over knows
/// immediately which loader they wanted (FR-EX-08 — never a silent misroute).
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "atst: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced by \
             `vokra-cli convert --model atst-base`?). ATST is a BYOL-style \
             teacher-student patchout SSL encoder over a log-mel patch grid; the sibling \
             SSL audio/music-embedding arches differ in the pre-training objective that \
             shapes the topology — `beats` (iterative acoustic tokenizer + masked \
             acoustic modelling), `eat` (utterance-level MAE with inverse block masking), \
             `dasheng` (universal MAE), `m2d` (masked modelling duo, dual online + target \
             branch), `maest` (AST backbone, Discogs music-tagger objective), `mert` \
             (HuBERT-derived masked prediction), `muq` (Mel-RVQ + BEATs teacher), \
             `yamnet` (supervised audio-tagging CNN, not SSL) — and the wav2vec2 lineage \
             (`hubert`, `wav2vec2_ctc`, `wavlm_sv`, `emotion2vec`) sits on a raw-waveform \
             1-D conv stem rather than a log-mel patch grid. Binding any of them here \
             would walk a foreign topology over an ATST payload (FR-EX-08 — no silent \
             partial load). Primary source: {UPSTREAM_URL}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "atst: GGUF is missing `vokra.model.arch` — this is not a Vokra-native atst \
             GGUF (was it produced by `vokra-cli convert --model atst-base`?). Refusing \
             to guess the arch from the tensor manifest (FR-EX-08). Primary source: \
             {UPSTREAM_URL}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Atst::encode`] / [`Atst::embed`] until the ATST forward wave lands.
///
/// `surface` is the method name, `output` is what that method would have
/// returned. The message names **four** blockers and cites **three** primary
/// sources so a reader diagnosing the gap has fully specified anchors
/// (`emotion2vec` / `wavlm` / `panns` / `redimnet` loud-partial-message
/// precedent — CLAUDE.md 教訓 (a)).
fn forward_loud_partial(surface: &str, output: &str) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "atst {surface} (loud-partial): the ATST encoder forward is deferred, so no \
         {output} can be produced. Four pieces must land first: \
         (i) NO `vokra.atst.*` AXIS CHUNK GROUP — the converter \
         (`vokra-cli convert --model atst-base`) is a verbatim F32/F16/BF16 \
         pass-through that stamps only `vokra.model.*` + `vokra.provenance.*`, so the \
         embedding width, depth, head count, patch grid AND every log-mel front-end axis \
         (sample rate, n_fft, hop, n_mels) are absent from the artifact and are not \
         transcribed anywhere in-repo; upstream keeps them in the training argparse \
         namespace inside the `.ckpt`, which the no-pickle rule (FR-LD-05 / NFR-DS-02) \
         keeps out of this runtime; \
         (ii) NO ViT-STYLE ENCODER PRIMITIVE — the log-mel front-end does exist \
         (`vokra_ops::mel`, `vokra_ops::fused_logmel`, `vokra_ops::kaldi_fbank`), but the \
         2-D patch embedding + plain pre-norm Transformer encoder does not: \
         `vokra_ops::conformer`, `vokra_ops::ebranchformer` and `vokra_ops::zipformer` \
         are conv-augmented ASR encoders over a 1-D frame sequence, not ViT patch \
         encoders over a 2-D mel plane; \
         (iii) TEACHER/STUDENT BRANCH SELECTION IS UNRESOLVED — a BYOL-style EMA \
         checkpoint carries both branches (`{student}` / `{teacher}` prefixes), and \
         picking the wrong one yields a shape-valid but numerically different embedding, \
         so which branch upstream's own inference entry point uses must be read off the \
         upstream tree first; \
         (iv) NO VERIFIED TENSOR-NAME MANIFEST — the converter copies every float tensor \
         under its verbatim upstream `state_dict` name and nothing in-repo transcribes \
         ATST's naming, so walking guessed names into typed slots would bind shape-valid \
         garbage. Primary sources: upstream tree {upstream}, paper (utterance-level, \
         INTERSPEECH 2022) {p2022}, paper (frame-level `atst-frame`, TASLP 2023) {p2023}. \
         The runtime cannot fabricate {output} (FR-EX-08 — no silent partial output; \
         CLAUDE.md 教訓 (a) 'loud-partial は fake-complete より honest').",
        student = AtstBranch::Student.prefix(),
        teacher = AtstBranch::Teacher.prefix(),
        upstream = PRIMARY_SOURCE_UPSTREAM,
        p2022 = PRIMARY_SOURCE_PAPER_2022,
        p2023 = PRIMARY_SOURCE_PAPER_2023,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the ATST runtime binder — contract-constant pins against the
    //! converter, metadata round-trip, loud negative space on every stated
    //! blocker, and arch-tag distinctness.
    //!
    //! # What "round-trip" means here
    //!
    //! On real audio this would be `encode(...)` returning hidden states, but
    //! the ATST forward is loud-partial (four blockers, see the module doc).
    //! Fabricating an output would violate CLAUDE.md 教訓 (a)
    //! (「loud-partial は fake-complete より honest」). The round-trips we
    //! *can* honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` and the two metadata keys
    //!    match the converter exactly, so a converter drift without a
    //!    binder-side follow-through fails here.
    //! 2. **Metadata round-trip** — a synthetic GGUF shaped like the
    //!    converter's output binds, and every stamp reads back.
    //! 3. **Loud negative space** — missing arch, foreign arch, empty
    //!    manifest, missing tensor, wrong dims, and both forward surfaces all
    //!    fire at their documented surface point in their documented variant.
    //! 4. **Arch distinctness pin** — the tag differs from every sibling SSL
    //!    audio/music-embedding arch and from the wav2vec2 lineage.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// A tensor-name sample shaped like the converter's own test module's
    /// "realistic upstream state-dict name" choices. Unverified against a real
    /// checkpoint — used here only to give the manifest something to hold.
    const SAMPLE_TENSORS: [(&str, [u64; 2]); 4] = [
        ("student.encoder.blocks.0.norm1.weight", [4, 1]),
        ("student.encoder.blocks.0.attn.qkv.weight", [4, 12]),
        ("teacher.encoder.blocks.0.norm1.weight", [4, 1]),
        ("teacher.encoder.blocks.0.attn.qkv.weight", [4, 12]),
    ];

    /// Builds a GGUF shaped like `convert_atst_file`'s output: arch + name +
    /// category + upstream URL, an optional weight-licence class, an optional
    /// FR-MD-09 attribution stamp, and the sample tensor manifest.
    fn atst_builder(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
    ) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        }
        if attribution {
            b.add_string(
                chunks::KEY_PROVENANCE_ATTRIBUTION,
                "ATST (Audio-WestlakeU/audiossl) weights, licensed CC BY 4.0.",
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
        b
    }

    /// Parses an `atst_builder` result into a `GgufFile`.
    fn atst_gguf(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
    ) -> GgufFile {
        let b = atst_builder(weight_license_class, attribution, with_tensors);
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        // Mirrors of `crates/vokra-convert/src/models/atst.rs`. A converter
        // drift without a binder-side follow-through lands here in the same
        // commit or fails this test.
        assert_eq!(ARCH, "atst", "arch tag pin");
        assert_eq!(NAME, "atst-base", "canonical size-point name pin");
        assert_eq!(CATEGORY, "audio-embedding", "category pin");
        assert_eq!(
            UPSTREAM_URL, "github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst",
            "upstream source tree pin (ATST is not on HuggingFace)"
        );
        assert_eq!(
            DEFAULT_LICENSE_SPDX, "cc-by-4.0",
            "the WEIGHT tier is cc-by-4.0 even though the CODE is mit"
        );
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_URL,
            "vokra.provenance.upstream_url"
        );
        // The weight SPDX must resolve to the class the converter stamps.
        assert_eq!(
            LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
            LicenseClass::AttributionRequired,
            "cc-by-4.0 must classify as AttributionRequired"
        );
        // ... which is commercially usable but carries a display obligation.
        assert!(LicenseClass::AttributionRequired.commercial_ok());
        assert!(LicenseClass::AttributionRequired.requires_attribution());
        // Branch prefixes are stable.
        assert_eq!(AtstBranch::Student.prefix(), "student.");
        assert_eq!(AtstBranch::Teacher.prefix(), "teacher.");
        assert_eq!(
            AtstBranch::all(),
            [AtstBranch::Student, AtstBranch::Teacher]
        );
    }

    // -----------------------------------------------------------------------
    // 2. Arch-tag distinctness pin
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_ssl_encoder_arches() {
        // Every sibling below is a real converter arch tag. Sharing one would
        // let runtime dispatch bind a foreign topology over an ATST payload
        // (FR-EX-08).
        for sibling in [
            // SSL audio/music-embedding neighbourhood.
            "beats",
            "eat",
            "dasheng",
            "m2d",
            "maest",
            "mert",
            "muq",
            // Supervised audio tagging (not SSL at all).
            "yamnet",
            // wav2vec2 lineage — raw-waveform 1-D conv stem, not a log-mel
            // patch grid.
            "hubert",
            "wav2vec2_ctc",
            "wavlm_sv",
            "emotion2vec",
        ] {
            assert_ne!(
                ARCH, sibling,
                "atst must not share an arch tag with `{sibling}` — different \
                 pre-training objective means a different topology (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Metadata round-trip on a synthetic converter-shaped GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_a_synthetic_converter_shaped_gguf() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), true, true);
        let m = Atst::from_gguf(&file).expect("a converter-shaped GGUF must bind");

        // Metadata surfaces round-trip.
        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_url(), Some(UPSTREAM_URL));

        // Tensor manifest.
        assert_eq!(m.tensor_count(), SAMPLE_TENSORS.len());
        assert_eq!(m.weights().tensor_names().len(), SAMPLE_TENSORS.len());
        assert_eq!(
            m.weights()
                .tensor_dims("student.encoder.blocks.0.attn.qkv.weight"),
            Some([4usize, 12].as_slice())
        );

        // BYOL duo diagnostic: the sample manifest carries both branches.
        assert_eq!(m.branch_tensor_count(AtstBranch::Student), 2);
        assert_eq!(m.branch_tensor_count(AtstBranch::Teacher), 2);

        // Licence + FR-MD-09 attribution surfaces.
        assert_eq!(m.weight_license(), LicenseClass::AttributionRequired);
        let attr = m.attribution().expect("attribution stamp must surface");
        assert!(
            attr.contains("CC BY 4.0"),
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

        let Err(err) = Atst::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native atst GGUF"),
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
        // A `beats` GGUF (iterative acoustic tokenizer + MAM) handed to the
        // ATST binder by mistake. Both are `audio-embedding` SSL encoders, so
        // a silent bind would look plausible right up until the numbers are
        // wrong — exactly the misroute FR-EX-08 forbids.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "beats");
        b.add_string(chunks::KEY_MODEL_NAME, "beats-iter3-plus-as2m");
        b.add_tensor("beats.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Atst::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH the actual and the expected tag.
                assert!(
                    m.contains("`beats`"),
                    "message must name the arch actually found, got `{m}`"
                );
                assert!(
                    m.contains("`atst`"),
                    "message must name the expected arch, got `{m}`"
                );
                // The neighbourhood must be enumerated so the reader knows
                // which loader they actually wanted.
                for sibling in [
                    "eat",
                    "dasheng",
                    "m2d",
                    "maest",
                    "mert",
                    "muq",
                    "yamnet",
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
                    m.contains("teacher-student"),
                    "message should state what makes ATST distinct, got `{m}`"
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
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, false);
        let Err(err) = Atst::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model atst-base"),
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
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        let Err(err) = m
            .weights()
            .require_tensor("student.encoder.blocks.11.mlp.fc2.weight")
        else {
            panic!("expected ModelLoad for a tensor that is not on disk");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("student.encoder.blocks.11.mlp.fc2.weight"),
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
                .require_tensor("teacher.encoder.blocks.0.norm1.weight")
                .expect("present tensor must resolve"),
            [4usize, 1].as_slice()
        );
    }

    // -----------------------------------------------------------------------
    // 8. require_tensor_dims names BOTH expected and actual dims
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_dims_names_expected_and_actual() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        // Exact match passes.
        m.weights()
            .require_tensor_dims("student.encoder.blocks.0.attn.qkv.weight", &[4, 12])
            .expect("matching dims must pass");

        // Mismatch fails loud, naming both sides.
        let Err(err) = m
            .weights()
            .require_tensor_dims("student.encoder.blocks.0.attn.qkv.weight", &[4, 36])
        else {
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
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        // A legitimately shaped buffer, so the loud-partial gate is what
        // fires (not some pre-encode length validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("atst encode"), "surface must be named: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Blocker (i): the missing axis chunk group.
                assert!(
                    msg.contains("vokra.atst.*"),
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
                // Blocker (iii): the BYOL branch ambiguity.
                assert!(
                    msg.contains("student.") && msg.contains("teacher."),
                    "must name both BYOL branch prefixes: {msg}"
                );
                // Blocker (iv): no verified tensor-name manifest.
                assert!(
                    msg.contains("state_dict"),
                    "must name the unverified tensor-name manifest: {msg}"
                );

                // All three primary sources.
                for url in [
                    PRIMARY_SOURCE_UPSTREAM,
                    PRIMARY_SOURCE_PAPER_2022,
                    PRIMARY_SOURCE_PAPER_2023,
                ] {
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
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.embed(&pcm) else {
            panic!("embed must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("atst embed"), "surface must be named: {msg}");
                assert!(
                    msg.contains("pooled utterance embedding"),
                    "must name the output it refuses to fabricate: {msg}"
                );
                assert!(
                    msg.contains("vokra.atst.*"),
                    "must name the absent axis chunk group: {msg}"
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
    // 11. Missing licence stamp fails closed to Unknown
    // -----------------------------------------------------------------------

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        // No provenance stamp at all: the binder still binds (arch + manifest
        // are the load gates), but the licence surface must fail closed.
        let file = atst_gguf(None, false, true);
        let m = Atst::from_gguf(&file).expect("arch + manifest are the load gates");
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
    // 12. Compliance gate: AttributionRequired passes strict, Unknown does not
    // -----------------------------------------------------------------------

    #[test]
    fn compliance_gate_passes_attribution_required_and_refuses_unknown() {
        // cc-by-4.0 -> AttributionRequired is commercially permitted, so a
        // correctly stamped artifact loads under the strict policy.
        let stamped = atst_builder(Some(LicenseClass::AttributionRequired), true, true)
            .to_bytes()
            .expect("serialize");
        let m = Atst::from_gguf_with_policy(&stamped, &CompliancePolicy::strict())
            .expect("AttributionRequired must pass the strict gate");
        assert_eq!(m.weight_license(), LicenseClass::AttributionRequired);

        // An unstamped artifact resolves to Unknown and is refused —
        // fail-closed, never a silent substitution.
        let unstamped = atst_builder(None, false, true)
            .to_bytes()
            .expect("serialize");
        let Err(err) = Atst::from_gguf_with_policy(&unstamped, &CompliancePolicy::strict()) else {
            panic!("an unstamped artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for an Unknown weight class, got {err:?}"
        );

        // The gate must not mask an arch mismatch: a foreign artifact reports
        // the arch, which is the actionable fact.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "dasheng");
        b.add_tensor("dasheng.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let foreign = b.to_bytes().expect("serialize");
        let Err(err) = Atst::from_gguf_with_policy(&foreign, &CompliancePolicy::strict()) else {
            panic!("a foreign arch must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("`dasheng`") && msg.contains("`atst`"),
                "arch mismatch must be reported ahead of any licence verdict: {msg}"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
