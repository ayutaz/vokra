//! **Audio deepfake / spoof-countermeasure detector** — runtime binder for
//! the `deepfake_detection` converter arch (Wave G 2026-08-15, loud-partial
//! per the emotion2vec / wavlm / panns / redimnet precedent — CLAUDE.md
//! 教訓 (a): "loud-partial は fake-complete より honest").
//!
//! # Primary sources
//!
//! - HF release: <https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2>
//!   — recorded by the converter as `UPSTREAM_HF`, and independently in
//!   `docs/license-audit.md` §3.1 row "Deepfake audio detection V2" as
//!   `license: apache-2.0, pipeline: audio-classification` (HF cardData API,
//!   fetched 2026-07-30). Both in-repo records call it a **WavLM-based
//!   binary classifier** (real vs synthetic speech). CLAUDE.md
//!   「ハルシネーション厳禁」— nothing below is asserted beyond those two
//!   records plus the WavLM primary sources named next.
//! - WavLM reference code (the backbone this checkpoint fine-tunes):
//!   <https://github.com/microsoft/UniSpeech/tree/main/WavLM>
//! - WavLM paper: Chen et al. 2021, *"WavLM: Large-Scale Self-Supervised
//!   Pre-Training for Full Stack Speech Processing"*
//!   (<https://arxiv.org/abs/2110.13900>)
//!
//! # A detector emits a score, not a verdict
//!
//! This is the load-bearing design decision of this module, and it is why
//! there is deliberately **no `is_fake() -> bool`** anywhere in the public
//! surface.
//!
//! A spoof-countermeasure model produces a real-valued score. Turning that
//! score into "this recording is fake" requires a threshold, and the right
//! threshold is a *deployment* property, not a *model* property: it depends
//! on the base rate of synthetic audio in the incoming stream, on the
//! relative cost of the two error directions, and on who is accountable for
//! the outcome. Both directions do real harm — a false positive brands a
//! genuine speaker a forger, and a false negative waves a forgery through a
//! control that someone is relying on.
//!
//! A `bool`-returning convenience method would bury that choice inside this
//! crate, where the person accountable for the decision cannot see it, review
//! it, or tune it. So the binder returns [`DeepfakeScore`] — the raw logits
//! and their softmax — and the caller picks the operating point. The one
//! comparison helper we do offer, [`DeepfakeScore::exceeds`], takes the
//! threshold as an explicit argument precisely so it shows up at the call
//! site.
//!
//! This also matches the deployment posture the converter docstring already
//! records and `docs/legal-compliance.md` §Article 50(3) describes: the
//! disclosure decision sits with the deployer, not with the runtime.
//!
//! ## We also refuse to guess *which class is which*
//!
//! There is a second, sharper version of the same hazard, and it is a real
//! gap in the current artifact rather than a stylistic preference.
//!
//! The head is binary, so its output is a 2-vector — but **the GGUF does not
//! record which index means "synthetic"**. The converter
//! (`crates/vokra-convert/src/models/deepfake_detection.rs`) copies float
//! tensors verbatim and stamps `vokra.model.{arch,name,category}` plus the
//! `vokra.provenance.*` group; it does *not* transcribe the upstream
//! `config.json` `id2label` map. Nothing in the GGUF pins the ordering.
//!
//! Guessing it here would be the single most damaging thing this module
//! could do: an inverted detector is strictly worse than an absent one,
//! because it reports confidently and in the wrong direction, and the
//! inversion is invisible at the call site. So [`DeepfakeScore`] indexes
//! purely positionally, and [`DeepfakeDetection::spoof_class_index`] is a
//! *loud* [`VokraError::UnsupportedOp`] naming the missing
//! [`GGUF_KEY_ID2LABEL`] chunk rather than a coin flip. Closing that gap is
//! converter-side work (stamp `id2label`), and the error says so.
//!
//! # Runtime layout (loud-partial)
//!
//! ```text
//! raw waveform (mono f32, [T] @ 16 kHz — WavLM lineage convention)
//!   -> 7-layer 1D conv feature-extractor stem              <- **loud-partial**
//!        (HuBERT/wav2vec2 lineage strided conv stack that
//!         downsamples 16 kHz audio to a ~50 Hz feature grid.)
//!   -> WavLM Transformer encoder                            <- **loud-partial**
//!        (The WavLM-specific primitive that neither wav2vec2
//!         nor HuBERT expose, and that no sibling module in
//!         this tree supplies today: a **gated relative
//!         position bias** plus a **convolutional
//!         position-bias fusion** applied around the attention
//!         softmax. Needs a walk against `WavLM.py`
//!         `TransformerSentenceEncoderLayer::forward`.)
//!   -> mean pooling over the encoder time axis              <- **loud-partial**
//!   -> Linear binary classifier head                        <- **REAL (bound)**
//!        (This module binds and shape-checks it: see
//!         [`DeepfakeDetectionWeights`]. Its output width is
//!         verified to be [`N_CLASSES`] = 2.)
//!   -> 2 raw logits -> [`DeepfakeScore`]                     <- **REAL (softmax)**
//! ```
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   1. [`DeepfakeDetection::from_gguf`] with strict
//!      `vokra.model.arch == "deepfake_detection"` verification. A sibling
//!      audio-classification / SSL-lineage GGUF handed here by mistake fails
//!      with a message naming **both** the expected and the actual arch (see
//!      "Sibling family distinctness").
//!   2. [`DeepfakeDetectionWeights::from_gguf`] — a non-empty manifest gate,
//!      plus real resolution and shape verification of the binary classifier
//!      head against [`CLASSIFIER_WEIGHT_CANDIDATES`], including the
//!      out-features == [`N_CLASSES`] check that makes "binary classifier"
//!      an enforced property rather than a docstring claim.
//!   3. [`DeepfakeScore`] end to end: numerically stable 2-way softmax,
//!      positional accessors, and the explicit-threshold
//!      [`DeepfakeScore::exceeds`] comparison. This is the whole decision
//!      surface, and it is real and unit-tested — only the feature
//!      extraction in front of it is deferred.
//!   4. Weight-license surfacing (fail-closed to [`LicenseClass::Unknown`]
//!      when the stamp is absent).
//!
//! - **Loud-partial (this WP)**: [`DeepfakeDetection::score`] returns
//!   [`VokraError::UnsupportedOp`] naming the WavLM gated-relative-position-
//!   bias encoder as the missing primitive and citing all three primary
//!   sources. **No fabricated detection score is ever emitted** (FR-EX-08 —
//!   no silent partial output). For this model in particular, a fabricated
//!   score is not a cosmetic lie: it is a security control reporting a
//!   number it did not compute.
//!
//! # Sibling family distinctness
//!
//! [`ARCH`] = `"deepfake_detection"` is deliberately distinct from every
//! sibling audio-classification / SSL-lineage arch tag verified present in
//! the converter tree — `wavlm_sv` (WavLM + XVector speaker-verification
//! head, 512-d embedding), `emotion2vec` (9-class emotion head), `clap`
//! (contrastive audio-text embedding), `ast` (Audio Spectrogram Transformer
//! tagger), `hubert` (bare SSL encoder, no fixed head). Several share the
//! WavLM/HuBERT backbone lineage, but every one exposes a different
//! downstream head, so silently aliasing the arch would misroute runtime
//! dispatch onto a loader whose tensor walk expects a different head
//! entirely (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] are mirrors of the converter's constants — the
//! same rule the sibling binders use so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layering
//! `vokra-ops -> nothing GGUF-aware`, `vokra-core -> GGUF reader`,
//! `vokra-models -> GGUF binder`, `vokra-convert -> GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The upstream release is consumed as safetensors; this runtime **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02). Any `.bin`-only variant is
//! bridged offline by a `tools/parity/*_prepare_checkpoint.py` sidecar
//! (uv-managed Python 3.12 per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), never in-process.
//!
//! # License
//!
//! `apache-2.0` ([`LicenseClass::Permissive`]) per the converter stamp and
//! the `docs/license-audit.md` §3.1 row. This module reads the stamp; it
//! does not sign anything off (owner-only per
//! `[[feedback-license-signoff-primary-source]]`).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/deepfake_detection.rs`. See the module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model deepfake-detection`.
///
/// Distinct from every sibling audio-classification / SSL-lineage arch tag
/// verified present in the converter tree (`wavlm_sv`, `emotion2vec`,
/// `clap`, `ast`, `hubert`) — silently sharing an arch would misroute
/// runtime dispatch onto a different-head loader (FR-EX-08; see the module
/// docstring "Sibling family distinctness").
pub const ARCH: &str = "deepfake_detection";

/// Expected `vokra.model.name` value written by the converter — the
/// canonical mirror slug for `MelodyMachine/Deepfake-audio-detection-V2`.
pub const NAME: &str = "deepfake-audio-detection-v2";

/// Expected `vokra.model.category` value. Consumed by the model-card
/// generator and the zoo-manifest tier gate so a classifier is never
/// advertised as an ASR / TTS release.
pub const CATEGORY: &str = "classification";

/// Upstream HuggingFace slug, mirrored from the converter so loud-partial
/// diagnostics can cite it without re-fetching a manifest.
pub const UPSTREAM_HF: &str = "MelodyMachine/Deepfake-audio-detection-V2";

/// SPDX identifier the converter stamps when the caller passes no
/// `--license` override. Cross-checked against `docs/license-audit.md` §3.1
/// (HF cardData API, fetched 2026-07-30).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Metadata key the converter writes for the upstream HF slug.
pub const GGUF_KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Metadata key the converter writes for the model category.
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key that **would** carry the upstream `config.json` `id2label`
/// map — the class-index-to-name mapping that says which of the two logits
/// means "synthetic".
///
/// **The converter does not write this key today.** It is named here as the
/// concrete, addressable gap so that
/// [`DeepfakeDetection::spoof_class_index`] can point at it instead of
/// guessing, and so a follow-up converter-side wave has an exact target.
/// See the module docstring, "We also refuse to guess which class is which".
pub const GGUF_KEY_ID2LABEL: &str = "vokra.deepfake.id2label";

/// Output width of the binary classifier head: real vs synthetic.
///
/// Sourced from the two independent in-repo records that both describe this
/// checkpoint as a **binary** classifier — the converter module docstring
/// and the `docs/license-audit.md` §3.1 row. Enforced (not merely
/// documented) by [`DeepfakeDetectionWeights::from_gguf`], which refuses a
/// head whose out-features disagree.
pub const N_CLASSES: u32 = 2;

/// Candidate tensor names for the binary classifier head's weight matrix,
/// in resolution order.
///
/// The converter copies upstream `state_dict` keys through **verbatim**, so
/// the exact key depends on how the upstream checkpoint was saved rather
/// than on anything Vokra controls. Rather than hard-coding one name as
/// though it were certain, [`DeepfakeDetectionWeights::from_gguf`] tries
/// each in order and — if none is present — fails loudly naming every
/// candidate it looked for, so the reader can compare against a real
/// manifest listing (FR-EX-08: never substitute a zero tensor).
///
/// The order follows the HuggingFace `*ForSequenceClassification`
/// convention (a bare `classifier.weight`, optionally namespaced under the
/// backbone), which is the convention an `audio-classification` pipeline
/// release is published under.
pub const CLASSIFIER_WEIGHT_CANDIDATES: [&str; 4] = [
    "classifier.weight",
    "classifier.dense.weight",
    "model.classifier.weight",
    "wavlm.classifier.weight",
];

/// Primary-source anchor: the upstream HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/MelodyMachine/Deepfake-audio-detection-V2";
/// Primary-source anchor: WavLM reference code (the fine-tuned backbone).
pub const PRIMARY_SOURCE_WAVLM_CODE: &str = "github.com/microsoft/UniSpeech/tree/main/WavLM";
/// Primary-source anchor: the WavLM paper (Chen et al. 2021).
pub const PRIMARY_SOURCE_WAVLM_PAPER: &str = "arxiv.org/abs/2110.13900";

/// Name of the missing primitive that blocks the real forward, quoted
/// verbatim in the loud-partial error so a reader diagnosing the gap has an
/// exact identifier to search for.
pub const MISSING_PRIMITIVE: &str = "WavLM gated relative position bias";

// ---------------------------------------------------------------------------
// DeepfakeScore — the real, tested decision surface.
// ---------------------------------------------------------------------------

/// The detector's output: two raw logits and the arithmetic to read them.
///
/// **This type carries a score, never a verdict.** It exposes no
/// `is_fake()`, and it deliberately does not name its two classes — see the
/// module docstring for both reasons. Indices are positional and correspond
/// to the classifier head's output rows in the order the checkpoint stores
/// them; the mapping from index to meaning is not recoverable from the GGUF
/// today (see [`GGUF_KEY_ID2LABEL`] and
/// [`DeepfakeDetection::spoof_class_index`]).
///
/// The softmax here is real, numerically stable, and unit-tested — it is not
/// part of the loud-partial surface. Only the feature extraction that would
/// *produce* the logits is deferred.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepfakeScore {
    logits: [f32; 2],
}

impl DeepfakeScore {
    /// Wraps two raw classifier logits.
    ///
    /// Public so the follow-up wave that lands the WavLM forward — and any
    /// caller running the head themselves — can construct the same score
    /// object this module already tests.
    #[inline]
    #[must_use]
    pub const fn from_logits(logits: [f32; 2]) -> Self {
        Self { logits }
    }

    /// The raw classifier logits, positionally indexed.
    #[inline]
    #[must_use]
    pub const fn logits(&self) -> [f32; 2] {
        self.logits
    }

    /// Softmax over the two logits, positionally indexed, summing to 1.
    ///
    /// Numerically stable: the maximum logit is subtracted before
    /// exponentiating, so a large-magnitude pair cannot overflow to
    /// `inf / inf = NaN`.
    #[must_use]
    pub fn probabilities(&self) -> [f32; 2] {
        let m = if self.logits[0] > self.logits[1] {
            self.logits[0]
        } else {
            self.logits[1]
        };
        let e0 = (self.logits[0] - m).exp();
        let e1 = (self.logits[1] - m).exp();
        let sum = e0 + e1;
        [e0 / sum, e1 / sum]
    }

    /// Probability of one class by positional index.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `index >= 2`. An out-of-range
    ///   index is a caller bug, and for a detector it is the kind of caller
    ///   bug that silently reads the wrong class, so it fails loudly rather
    ///   than saturating (FR-EX-08).
    pub fn probability_of(&self, index: usize) -> Result<f32> {
        if index >= N_CLASSES as usize {
            return Err(VokraError::InvalidArgument(format!(
                "deepfake_detection: class index {index} is out of range for a \
                 {N_CLASSES}-class head (valid: 0..{last}). Refusing to clamp — a \
                 silently clamped index on a spoof detector reads the wrong class \
                 (FR-EX-08).",
                last = N_CLASSES - 1
            )));
        }
        Ok(self.probabilities()[index])
    }

    /// Whether the probability of the class at `index` exceeds
    /// `threshold`.
    ///
    /// The threshold is an **explicit argument on purpose**: it is a
    /// deployment decision, and passing it here keeps it visible at the call
    /// site where the person accountable for the operating point can see and
    /// review it. This crate does not have, and will not grow, a default.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `index >= 2` (see
    ///   [`Self::probability_of`]).
    pub fn exceeds(&self, index: usize, threshold: f32) -> Result<bool> {
        Ok(self.probability_of(index)? > threshold)
    }
}

// ---------------------------------------------------------------------------
// DeepfakeDetectionWeights — real classifier-head resolution + shape gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a `deepfake_detection` GGUF.
///
/// [`from_gguf`](Self::from_gguf) is a *loud* verification step and performs
/// real work: it rejects an empty manifest, resolves the binary classifier
/// head against [`CLASSIFIER_WEIGHT_CANDIDATES`], and verifies that head's
/// shape — rank 2 with out-features equal to [`N_CLASSES`]. That last check
/// is what turns "binary classifier" from a docstring claim into an enforced
/// property of every artifact this binder accepts.
#[derive(Debug, Clone)]
pub struct DeepfakeDetectionWeights {
    tensors: Vec<(String, Vec<usize>)>,
    classifier_weight: String,
    classifier_bias: Option<String>,
    hidden_size: usize,
}

impl DeepfakeDetectionWeights {
    /// Scans `gguf`, resolves the classifier head, and verifies its shape.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    /// - [`VokraError::ModelLoad`] when no candidate classifier-head weight
    ///   is present — the message names every candidate that was tried.
    /// - [`VokraError::ModelLoad`] when the head is not rank 2, or when its
    ///   out-features differ from [`N_CLASSES`] — the message names both the
    ///   expected and the actual value.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate {UPSTREAM_HF} checkpoint \
                 carries the full WavLM backbone plus a binary classifier head \
                 (arch={ARCH}, name={NAME}); zero tensors always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model \
                 deepfake-detection` against an upstream safetensors checkpoint."
            )));
        }

        // Resolve the classifier head. The converter passes upstream
        // state_dict names through verbatim, so we try the documented
        // candidate set rather than hard-coding one name as certain.
        let (classifier_weight, dims) = CLASSIFIER_WEIGHT_CANDIDATES
            .into_iter()
            .find_map(|cand| {
                tensors
                    .iter()
                    .find(|(n, _)| n.as_str() == cand)
                    .map(|(_, d)| (cand, d.clone()))
            })
            .ok_or_else(|| classifier_head_absent(&tensors))?;

        if dims.len() != 2 {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: classifier head `{classifier_weight}` has rank \
                 {rank} (dims {dims:?}), expected rank 2 — a Linear head is stored as \
                 [out_features, in_features]. Refusing to reshape silently \
                 (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF}",
                rank = dims.len()
            )));
        }

        // The out-features axis is the whole reason this arch exists: a
        // binary real-vs-synthetic head. Enforce it rather than assume it.
        if dims[0] != N_CLASSES as usize {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: classifier head `{classifier_weight}` has \
                 out_features {actual} but this arch is a BINARY detector, expected \
                 {N_CLASSES} (dims {dims:?} = [out_features, in_features]). Both \
                 in-repo records for {UPSTREAM_HF} — the converter docstring and the \
                 `docs/license-audit.md` §3.1 row — describe a binary real-vs-synthetic \
                 classifier, so a wider head means this GGUF is a DIFFERENT detector \
                 variant (for example a multi-class spoof-attack-type classifier) and \
                 its class indices would not mean what a caller of this binder expects. \
                 Refusing to bind (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF}",
                actual = dims[0]
            )));
        }
        let hidden_size = dims[1];

        // Bias is recorded, not required: a Linear head may legitimately be
        // constructed with `bias=False`, and nothing in the two in-repo
        // records pins it either way, so making it a hard gate would be a
        // guess dressed up as a check.
        let bias_name = classifier_weight.replace(".weight", ".bias");
        let has_bias = tensors.iter().any(|(n, _)| n.as_str() == bias_name);
        let classifier_bias = if has_bias { Some(bias_name) } else { None };

        Ok(Self {
            tensors,
            classifier_weight: classifier_weight.to_owned(),
            classifier_bias,
            hidden_size,
        })
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The resolved classifier-head weight tensor name (one of
    /// [`CLASSIFIER_WEIGHT_CANDIDATES`]).
    #[inline]
    #[must_use]
    pub fn classifier_weight_name(&self) -> &str {
        &self.classifier_weight
    }

    /// The classifier-head bias tensor name, when the checkpoint carries
    /// one. `None` for a `bias=False` head.
    #[inline]
    #[must_use]
    pub fn classifier_bias_name(&self) -> Option<&str> {
        self.classifier_bias.as_deref()
    }

    /// The head's in-features axis — the pooled backbone width feeding the
    /// classifier, read from the head's own stamped dims rather than from a
    /// hard-coded topology constant.
    #[inline]
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Dims of a tensor on disk, or `None` when it is absent.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }
}

/// Builds the loud [`VokraError::ModelLoad`] for a GGUF with no resolvable
/// classifier head, naming every candidate that was tried plus a sample of
/// what is actually on disk.
fn classifier_head_absent(tensors: &[(String, Vec<usize>)]) -> VokraError {
    let sample: Vec<&str> = tensors.iter().map(|(n, _)| n.as_str()).take(8).collect();
    VokraError::ModelLoad(format!(
        "deepfake_detection: required classifier-head weight tensor is absent from \
         the GGUF. Looked for, in order: {cands:?}. The GGUF carries {count} tensors; \
         first names on disk: {sample:?}. The converter copies upstream `state_dict` \
         keys through verbatim, so a miss means the checkpoint namespaces its head \
         under a prefix not in the candidate list — add it to \
         `CLASSIFIER_WEIGHT_CANDIDATES` after checking a real manifest listing. \
         Refusing to substitute a zero tensor, which on a spoof detector would mean \
         scoring every input with an untrained head (FR-EX-08). Primary source: \
         {PRIMARY_SOURCE_HF}",
        cands = CLASSIFIER_WEIGHT_CANDIDATES,
        count = tensors.len(),
    ))
}

// ---------------------------------------------------------------------------
// DeepfakeDetection — the runtime binder handle.
// ---------------------------------------------------------------------------

/// Audio deepfake / spoof-countermeasure detector
/// (`MelodyMachine/Deepfake-audio-detection-V2`, apache-2.0) runtime binder.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`score`](Self::score) on a mono f32 PCM waveform (16 kHz, WavLM lineage
/// convention) to obtain a [`DeepfakeScore`].
///
/// **There is no `is_fake()`.** The score is returned; the threshold is the
/// caller's. See the module docstring for why that is a deliberate,
/// load-bearing choice rather than an omission.
#[derive(Debug, Clone)]
pub struct DeepfakeDetection {
    weights: DeepfakeDetectionWeights,
    weight_license: LicenseClass,
    name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
}

impl DeepfakeDetection {
    /// Binds a `deepfake_detection` GGUF: verifies the arch strictly,
    /// resolves and shape-checks the binary classifier head, and surfaces
    /// the stamped weight-license class for the compliance-gate
    /// cross-checks.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key, so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is some other
    ///   arch — the message names **both** the expected and the actual tag.
    /// - [`VokraError::ModelLoad`] from
    ///   [`DeepfakeDetectionWeights::from_gguf`] for an empty manifest, an
    ///   unresolvable classifier head, or a head whose shape is not
    ///   `[N_CLASSES, hidden]`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, so a mis-routed model fails with a specific
        //    message instead of a downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "deepfake_detection: GGUF arch is `{other}`, expected `{ARCH}` (was \
                     this GGUF produced by `vokra-cli convert --model \
                     deepfake-detection`?). Sibling audio-classification / SSL-lineage \
                     arch tags in this tree — `wavlm_sv` (WavLM + XVector \
                     speaker-verification head, 512-d embedding), `emotion2vec` (9-class \
                     emotion head), `clap` (contrastive audio-text embedding), `ast` \
                     (Audio Spectrogram Transformer tagger), `hubert` (bare SSL encoder, \
                     no fixed head) — several share the WavLM/HuBERT backbone lineage but \
                     every one exposes a different downstream head. This arch's binary \
                     real-vs-synthetic head has no analog in any sibling, so silently \
                     aliasing the arch would misroute runtime dispatch and, on a spoof \
                     detector, would surface another model's logits as a detection score \
                     (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "deepfake_detection: GGUF is missing `vokra.model.arch` — this is \
                     not a Vokra-native {ARCH} GGUF (was it produced by `vokra-cli \
                     convert --model deepfake-detection`?). Refusing to guess the arch \
                     of an unlabeled artifact (FR-EX-08)."
                )));
            }
        }

        // 2. Real weight binding: manifest + classifier head + shape gate.
        let weights = DeepfakeDetectionWeights::from_gguf(file)?;

        // 3. Provenance surfacing. The converter stamps `Permissive`
        //    (apache-2.0); a GGUF missing the stamp reads back as `Unknown`
        //    (fail-closed per `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned);
        let category = file
            .get(GGUF_KEY_MODEL_CATEGORY)
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned);
        let upstream_hf = file
            .get(GGUF_KEY_UPSTREAM_HF)
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned);

        Ok(Self {
            weights,
            weight_license,
            name,
            category,
            upstream_hf,
        })
    }

    /// The stamped weight-license class from
    /// `vokra.provenance.weight_license`. The converter stamps
    /// [`LicenseClass::Permissive`] (apache-2.0); a GGUF missing the stamp
    /// reads back as [`LicenseClass::Unknown`] (fail-closed at the M2-13
    /// compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The stamped `vokra.model.name`, when present.
    #[inline]
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, when present.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_hf` slug, when present.
    #[inline]
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }

    /// The bound weights, including the resolved classifier-head names and
    /// the head's in-features width.
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &DeepfakeDetectionWeights {
        &self.weights
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The classifier head's in-features width, read from the head tensor's
    /// own dims.
    #[inline]
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size()
    }

    /// The head's output width ([`N_CLASSES`] = 2), verified at bind time.
    #[inline]
    #[must_use]
    pub const fn num_classes() -> u32 {
        N_CLASSES
    }

    /// Which output index means "synthetic".
    ///
    /// # Loud-partial (this WP) — and a deliberate refusal
    ///
    /// Always returns [`VokraError::UnsupportedOp`]. The GGUF does not carry
    /// the upstream `id2label` map (see [`GGUF_KEY_ID2LABEL`]), so the
    /// mapping from index to meaning is genuinely not recoverable from the
    /// artifact.
    ///
    /// Guessing would be the most damaging thing this module could do: an
    /// inverted spoof detector reports confidently in the wrong direction,
    /// and the inversion is invisible at the call site. So this fails loudly
    /// and names the converter-side fix instead of returning a coin flip.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — always, until the converter stamps
    ///   [`GGUF_KEY_ID2LABEL`].
    pub fn spoof_class_index(&self) -> Result<usize> {
        Err(VokraError::UnsupportedOp(format!(
            "deepfake_detection spoof_class_index (loud-partial): the GGUF does not \
             record which of the {N_CLASSES} output indices means \"synthetic\". The \
             converter copies float tensors verbatim and stamps \
             `vokra.model.{{arch,name,category}}` plus the `vokra.provenance.*` group, \
             but it does NOT transcribe the upstream `config.json` `id2label` map, and \
             no other chunk pins the ordering. The fix is converter-side: stamp \
             `{GGUF_KEY_ID2LABEL}` from the upstream config, then this accessor becomes \
             a lookup. Refusing to guess — an inverted spoof detector reports \
             confidently in the wrong direction and the inversion is invisible at the \
             call site, which is strictly worse than no detector at all (FR-EX-08 — no \
             silent partial output). Read {hf} `config.json` for the authoritative \
             mapping in the meantime.",
            hf = PRIMARY_SOURCE_HF,
        )))
    }

    /// Scores a PCM waveform for synthetic-speech likelihood.
    ///
    /// Returns a [`DeepfakeScore`] — logits and their softmax — **not** a
    /// verdict. The caller chooses the operating point; see the module
    /// docstring for why that boundary sits here.
    ///
    /// `_pcm` is the raw waveform as mono f32 in `[-1, 1]` at 16 kHz (WavLM
    /// lineage convention). A rate or shape mismatch will be a loud error
    /// rather than a resample surprise once the real forward lands.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming
    /// [`MISSING_PRIMITIVE`] — the WavLM gated relative position bias (plus
    /// its convolutional position-bias fusion), which no module in this tree
    /// supplies today and which cannot be synthesized from the binder
    /// scaffold without a walk against the WavLM reference code. The
    /// classifier head in front of it *is* bound and shape-verified, and
    /// [`DeepfakeScore`]'s arithmetic is real — the gap is exactly the
    /// feature extractor.
    ///
    /// **No fabricated detection score is ever emitted.** For this model
    /// that is not a stylistic rule: a fabricated score is a security
    /// control reporting a number it did not compute (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred WavLM conv stem + Transformer encoder + mean pooling.
    pub fn score(&self, _pcm: &[f32]) -> Result<DeepfakeScore> {
        let _ = _pcm;
        Err(score_forward_loud_partial(
            self.weights.classifier_weight_name(),
            self.weights.hidden_size(),
        ))
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`DeepfakeDetection::score`] until the WavLM feature extractor lands.
///
/// Names the missing primitive by exact identifier, states what *is*
/// already real (the bound head and its width), and cites all three primary
/// sources so a reader diagnosing the gap has three places to walk.
fn score_forward_loud_partial(classifier_weight: &str, hidden_size: usize) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "deepfake_detection score (loud-partial): the full forward is deferred. The \
         missing primitive is the `{MISSING_PRIMITIVE}` — WavLM applies a gated \
         relative position bias plus a convolutional position-bias fusion around the \
         attention softmax, which neither wav2vec2 nor HuBERT expose and which no \
         module in this tree supplies today. Three pieces must land before a real \
         score can be emitted: (1) the 7-layer 1D conv feature-extractor stem \
         (HuBERT/wav2vec2-lineage strided stack, 16 kHz -> ~50 Hz feature grid); \
         (2) the WavLM Transformer encoder carrying the `{MISSING_PRIMITIVE}`, per a \
         walk against `WavLM.py` `TransformerSentenceEncoderLayer::forward`; \
         (3) mean pooling over the encoder time axis. What is ALREADY real: the \
         binary classifier head is bound and shape-verified at \
         `{classifier_weight}` with dims [{N_CLASSES}, {hidden_size}], and \
         `DeepfakeScore`'s softmax is implemented and tested — the gap is exactly the \
         feature extractor in front of it. Primary sources: HF release {hf}, WavLM \
         reference code {code}, WavLM paper {paper}. The runtime cannot fabricate a \
         detection score: on a spoof-countermeasure model that would be a security \
         control reporting a number it did not compute (FR-EX-08 — no silent partial \
         output).",
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_WAVLM_CODE,
        paper = PRIMARY_SOURCE_WAVLM_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the `deepfake_detection` runtime binder.
    //!
    //! Two kinds of coverage live here. The **real** surface — classifier
    //! head resolution, the binary-width gate, and every bit of
    //! [`DeepfakeScore`]'s arithmetic — is tested for actual correct
    //! behaviour, because it is actually implemented. The **loud-partial**
    //! surface is tested as negative space: each stated blocker must fire at
    //! its documented surface point, in the documented error variant, with a
    //! message naming the missing primitive.
    //!
    //! Fabricating a detection output to make a "real forward" test would
    //! violate CLAUDE.md 教訓 (a) and, for this model specifically, would be
    //! a test asserting that a security control works when it does not.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// f32 comparison helper for the softmax assertions. The tolerance is
    /// loose enough to absorb a few ulps of accumulated f32 rounding (the
    /// log-odds check divides then takes a log) while still being far tighter
    /// than any of the structural properties under test.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    /// Builds a well-formed `deepfake_detection` GGUF: arch + name +
    /// category + upstream slug + optional license stamp, one representative
    /// WavLM backbone tensor, and a binary classifier head of
    /// `[N_CLASSES, hidden]`.
    fn detector_gguf(weight_license_class: Option<LicenseClass>, hidden: u64) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_UPSTREAM_HF, UPSTREAM_HF);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // A representative backbone tensor, using the same realistic
        // WavLM-derived name the converter's own test module picks.
        b.add_tensor(
            "wavlm.encoder.layers.0.attention.q_proj.weight",
            GgmlType::F32,
            vec![hidden, hidden],
            vec![0u8; (hidden * hidden * 4) as usize],
        )
        .expect("add_tensor backbone");
        // The binary classifier head: [out_features, in_features].
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![u64::from(N_CLASSES), hidden],
            vec![0u8; (u64::from(N_CLASSES) * hidden * 4) as usize],
        )
        .expect("add_tensor head");
        b.add_tensor(
            "classifier.bias",
            GgmlType::F32,
            vec![u64::from(N_CLASSES)],
            vec![0u8; (N_CLASSES * 4) as usize],
        )
        .expect("add_tensor bias");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate consistency with the
    //          converter).
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "deepfake_detection", "arch tag pin");
        assert_eq!(NAME, "deepfake-audio-detection-v2", "canonical name pin");
        assert_eq!(CATEGORY, "classification", "category pin");
        assert_eq!(
            UPSTREAM_HF, "MelodyMachine/Deepfake-audio-detection-V2",
            "upstream HF slug pin (cited in loud-partial diagnostics)"
        );
        assert_eq!(DEFAULT_LICENSE_SPDX, "apache-2.0", "default SPDX pin");
        assert_eq!(GGUF_KEY_UPSTREAM_HF, "vokra.provenance.upstream_hf");
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            N_CLASSES, 2,
            "this arch is a binary real-vs-synthetic detector"
        );
        assert_eq!(DeepfakeDetection::num_classes(), N_CLASSES);
        assert_eq!(
            CLASSIFIER_WEIGHT_CANDIDATES[0], "classifier.weight",
            "the HF *ForSequenceClassification convention resolves first"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — Missing arch fails loud (never binds an unlabeled GGUF).
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![2, 8],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = DeepfakeDetection::from_gguf(&file) else {
            panic!("expected an error when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("deepfake_detection"),
                    "message must name the expected arch, got `{m}`"
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
    // Test 3 — Foreign arch fails loud naming BOTH expected and actual.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both() {
        // `wavlm_sv` shares the WavLM backbone lineage but exposes an
        // XVector speaker-verification head — exactly the confusion that
        // must not resolve silently.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "wavlm_sv");
        b.add_string(chunks::KEY_MODEL_NAME, "wavlm-base-plus-sv");
        b.add_tensor("wavlm.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = DeepfakeDetection::from_gguf(&file) else {
            panic!("expected an error when the GGUF carries a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH tags must appear — the actual and the expected.
                assert!(
                    m.contains("`wavlm_sv`"),
                    "message must name the ACTUAL arch, got `{m}`"
                );
                assert!(
                    m.contains("`deepfake_detection`"),
                    "message must name the EXPECTED arch, got `{m}`"
                );
                // The sibling neighbourhood must be enumerated so the reader
                // has fully specified anchors. Every tag below was verified
                // present in the converter tree.
                for sibling in ["wavlm_sv", "emotion2vec", "clap", "ast", "hubert"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — A synthetic GGUF with the right tensors binds.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_well_formed_detector() {
        let file = detector_gguf(Some(LicenseClass::Permissive), 768);
        let d = DeepfakeDetection::from_gguf(&file).expect("well-formed GGUF must bind");

        assert_eq!(
            d.weight_license(),
            LicenseClass::Permissive,
            "Permissive stamp must round-trip (mirror of the converter's apache-2.0 stamp)"
        );
        assert_eq!(d.model_name(), Some(NAME));
        assert_eq!(d.category(), Some(CATEGORY));
        assert_eq!(d.upstream_hf(), Some(UPSTREAM_HF));
        assert_eq!(d.tensor_count(), 3, "backbone + head weight + head bias");

        // The classifier head resolved, and its in-features axis was read
        // from the tensor's own dims rather than a hard-coded constant.
        assert_eq!(d.weights().classifier_weight_name(), "classifier.weight");
        assert_eq!(d.weights().classifier_bias_name(), Some("classifier.bias"));
        assert_eq!(d.hidden_size(), 768);
        assert_eq!(
            d.weights().tensor_dims("classifier.weight"),
            Some([2_usize, 768].as_slice())
        );

        // A different backbone width must be read through, not assumed.
        let narrow = detector_gguf(None, 256);
        let d2 = DeepfakeDetection::from_gguf(&narrow).expect("narrow GGUF must bind");
        assert_eq!(d2.hidden_size(), 256);
        assert_eq!(
            d2.weight_license(),
            LicenseClass::Unknown,
            "a missing license stamp must fail closed to Unknown"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5 — A missing tensor fails loud, naming it.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_classifier_head_naming_it() {
        // Correct arch, a backbone tensor present, but no classifier head
        // under any candidate name.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_tensor(
            "wavlm.encoder.layers.0.attention.q_proj.weight",
            GgmlType::F32,
            vec![8, 8],
            vec![0u8; 256],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = DeepfakeDetection::from_gguf(&file) else {
            panic!("expected an error when the classifier head is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // Every candidate that was tried must be named.
                for cand in CLASSIFIER_WEIGHT_CANDIDATES {
                    assert!(
                        m.contains(cand),
                        "message must name candidate tensor `{cand}`, got `{m}`"
                    );
                }
                // And it must show what is actually on disk.
                assert!(
                    m.contains("wavlm.encoder.layers.0.attention.q_proj.weight"),
                    "message must sample the names actually present, got `{m}`"
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
    // Test 6 — An empty tensor manifest fails loud.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // No tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = DeepfakeDetection::from_gguf(&file) else {
            panic!("expected an error when the GGUF carries zero tensors");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model deepfake-detection"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — A non-binary head fails loud naming expected AND actual.
    //          "Binary classifier" is enforced, not merely documented.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_non_binary_classifier_head() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // A 7-way head — a different detector variant, e.g. a spoof-attack
        // type classifier, whose class indices would not mean what a caller
        // of this binder expects.
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![7, 768],
            vec![0u8; 7 * 768 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = DeepfakeDetection::from_gguf(&file) else {
            panic!("expected an error when the head is not binary");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("out_features 7"),
                    "message must name the ACTUAL width, got `{m}`"
                );
                assert!(
                    m.contains("expected 2"),
                    "message must name the EXPECTED width, got `{m}`"
                );
                assert!(
                    m.contains("classifier.weight"),
                    "message must name the offending tensor, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — A rank-1 head fails loud rather than being reshaped.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_non_rank2_classifier_head() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_tensor("classifier.weight", GgmlType::F32, vec![2], vec![0u8; 8])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = DeepfakeDetection::from_gguf(&file) else {
            panic!("expected an error when the head is not rank 2");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("rank 1"),
                    "message must name the actual rank, got `{m}`"
                );
                assert!(
                    m.contains("expected rank 2"),
                    "message must name the expected rank, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — The forward loud-partials, and the message names the missing
    //          primitive.
    // -----------------------------------------------------------------------

    #[test]
    fn score_loud_partials_naming_the_missing_primitive() {
        let file = detector_gguf(Some(LicenseClass::Permissive), 768);
        let d = DeepfakeDetection::from_gguf(&file).expect("valid GGUF must bind");

        // 1 s of silence at 16 kHz mono — the WavLM-lineage input convention.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = d.score(&pcm) else {
            panic!("expected score() to loud-partial rather than fabricate a detection score");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("deepfake_detection score"),
                    "the surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // THE missing primitive, by exact identifier.
                assert!(
                    msg.contains(MISSING_PRIMITIVE),
                    "message must name the missing primitive `{MISSING_PRIMITIVE}`: {msg}"
                );
                assert!(
                    msg.contains("conv feature-extractor stem"),
                    "message must name the deferred conv stem: {msg}"
                );
                assert!(
                    msg.contains("mean pooling"),
                    "message must name the deferred pooling step: {msg}"
                );

                // It must also say what IS real, so the reader knows the gap
                // is the feature extractor and not the head.
                assert!(
                    msg.contains("classifier.weight"),
                    "message must report the bound head tensor: {msg}"
                );
                assert!(
                    msg.contains("[2, 768]"),
                    "message must report the verified head dims: {msg}"
                );

                // All three primary sources cited.
                for url in [
                    PRIMARY_SOURCE_HF,
                    PRIMARY_SOURCE_WAVLM_CODE,
                    PRIMARY_SOURCE_WAVLM_PAPER,
                ] {
                    assert!(
                        msg.contains(url),
                        "expected primary source URL '{url}' cited: {msg}"
                    );
                }
                assert!(
                    msg.contains("FR-EX-08"),
                    "expected the FR-EX-08 no-fabrication rationale: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10 — The class-index mapping is refused, not guessed.
    // -----------------------------------------------------------------------

    #[test]
    fn spoof_class_index_refuses_to_guess() {
        let file = detector_gguf(Some(LicenseClass::Permissive), 768);
        let d = DeepfakeDetection::from_gguf(&file).expect("valid GGUF must bind");

        let Err(err) = d.spoof_class_index() else {
            panic!("expected spoof_class_index to refuse rather than guess a class ordering");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains(GGUF_KEY_ID2LABEL),
                    "message must name the missing metadata key `{GGUF_KEY_ID2LABEL}`: {msg}"
                );
                assert!(
                    msg.contains("id2label"),
                    "message must name the upstream config field: {msg}"
                );
                assert!(
                    msg.contains("Refusing to guess"),
                    "message must state the refusal explicitly: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11 — DeepfakeScore softmax is REAL: sums to 1, ordered, and
    //           numerically stable at large magnitudes.
    // -----------------------------------------------------------------------

    #[test]
    fn score_softmax_is_real_and_stable() {
        // Equal logits -> a flat posterior.
        let flat = DeepfakeScore::from_logits([0.0, 0.0]);
        let p = flat.probabilities();
        assert!(
            close(p[0], 0.5) && close(p[1], 0.5),
            "flat posterior: {p:?}"
        );

        // Ordering is preserved, and the two probabilities sum to 1.
        let skewed = DeepfakeScore::from_logits([2.0, -1.0]);
        let q = skewed.probabilities();
        assert!(q[0] > q[1], "the larger logit must carry more mass: {q:?}");
        assert!(
            close(q[0] + q[1], 1.0),
            "probabilities must sum to 1: {q:?}"
        );
        // ln-odds of a 2-way softmax equals the logit difference: 2 - (-1).
        assert!(
            close((q[0] / q[1]).ln(), 3.0),
            "2-way softmax log-odds must equal the logit gap: {q:?}"
        );

        // Numerical stability: naive exp() on these would overflow to inf
        // and yield NaN. The max-subtraction must keep it finite.
        let huge = DeepfakeScore::from_logits([200.0, 100.0]);
        let h = huge.probabilities();
        assert!(
            h[0].is_finite() && h[1].is_finite(),
            "large logits must not overflow: {h:?}"
        );
        assert!(
            close(h[0] + h[1], 1.0),
            "stable softmax must sum to 1: {h:?}"
        );
        assert!(close(h[0], 1.0), "a 100-nat gap saturates to 1: {h:?}");

        // Raw logits round-trip untouched.
        assert_eq!(skewed.logits(), [2.0, -1.0]);
    }

    // -----------------------------------------------------------------------
    // Test 12 — The threshold is the caller's: `exceeds` takes it as an
    //           explicit argument, and out-of-range indices fail loud.
    // -----------------------------------------------------------------------

    #[test]
    fn threshold_is_explicit_and_indices_are_checked() {
        let s = DeepfakeScore::from_logits([2.0, -1.0]);
        let p0 = s.probability_of(0).expect("index 0 is in range");
        assert!(close(p0, s.probabilities()[0]));

        // The same score yields opposite answers under different operating
        // points — which is exactly why this crate does not pick one.
        assert!(
            s.exceeds(0, 0.5).expect("index 0 is in range"),
            "p0 ~= 0.953 must exceed a 0.5 threshold"
        );
        assert!(
            !s.exceeds(0, 0.99).expect("index 0 is in range"),
            "p0 ~= 0.953 must NOT exceed a 0.99 threshold"
        );

        // An out-of-range class index is a caller bug that would silently
        // read the wrong class, so it fails loudly rather than clamping.
        let Err(err) = s.probability_of(2) else {
            panic!("expected an error when the class index is out of range");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(
                    m.contains("class index 2 is out of range"),
                    "message must name the offending index, got `{m}`"
                );
                assert!(
                    m.contains("Refusing to clamp"),
                    "message must state the refusal, got `{m}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        let Err(err) = s.exceeds(9, 0.5) else {
            panic!("expected an error when exceeds() gets an out-of-range index");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
