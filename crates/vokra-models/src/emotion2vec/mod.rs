//! emotion2vec+ Large — 9-class speech-emotion recognition runtime binder
//! (Wave 8 2026-08-14 audit follow-up **RETRY** of Wave 7 silently-lost item,
//! loud-partial per panns / redimnet / wavlm / storm / musicgen / audioldm2
//! precedent — CLAUDE.md 教訓 (a): "loud-partial は fake-complete より honest").
//!
//! # Primary sources
//!
//! - HF release: <https://huggingface.co/emotion2vec/emotion2vec_plus_large>
//!   (`license: mit`, verified 2026-07-25 per the converter docstring —
//!   CLAUDE.md「ハルシネーション厳禁」)
//! - Reference code (Apache-2.0 / MIT — the FunASR umbrella hosts the
//!   reference wrapper): <https://github.com/ddlBoJack/emotion2vec>
//! - Paper: Ma et al. 2024, *"emotion2vec: Self-Supervised Pre-Training for
//!   Speech Emotion Representation"*, ACL 2024
//!   (<https://arxiv.org/abs/2312.15185>)
//!
//! # Architecture (transcribed from primary sources — Ma et al. 2024 §III + FunASR pipeline)
//!
//! ```text
//! PCM (mono f32, 16 kHz)                          ← wav2vec2/HuBERT-lineage input convention
//!   -> log-mel + wav2vec2-style feature-extractor stem  ← **loud-partial**
//!        (torchaudio Kaldi fbank frontend + 1D conv
//!         subsampling; exact axes deferred to real-
//!         checkpoint dump since the converter does not
//!         currently stamp `vokra.emotion2vec.*`).
//!   -> wav2vec2-style SSL Transformer encoder       ← **loud-partial**
//!        (base topology: 12 layers × 768 hidden, per
//!         wav2vec2 base lineage; the emotion2vec+ Large
//!         variant's exact layer count / hidden dim
//!         requires a real-checkpoint tensor-name walk).
//!   -> utterance-level mean pooling over time      ← **loud-partial**
//!        (Ma et al. §III-B: utterance = mean over the
//!         encoder time axis before the classifier head).
//!   -> Linear 9-way emotion classifier head        ← **loud-partial**
//!        (out_features = [`N_CLASSES_EMOTION`] = 9, class labels
//!         [`EMOTION_CLASS_LABELS`] verbatim from the
//!         upstream `label.txt`: Angry / Disgusted / Fearful /
//!         Happy / Neutral / Other / Sad / Surprised / `<unk>`).
//!   -> 9-way logits vector (raw logits — softmax is a
//!      consumer-side concern, mirror of the wavlm_sv /
//!      panns loud-partial output-shape contract)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Emotion2Vec::from_gguf`] with strict `vokra.model.arch ==
//!     "emotion2vec"` validation. The sibling SSL-encoder / wav2vec2-lineage
//!     arch tags (`wav2vec2_ctc` / `wavlm_sv` / `hubert` / `data2vec-audio`)
//!     fail with a specific sibling-mis-route [`VokraError::ModelLoad`]
//!     enumerating the whole wav2vec2-SSL-lineage fleet — silent aliasing
//!     would misroute the runtime dispatch to a family with a different
//!     downstream head (CTC ASR head / XVector speaker head / SSL encoder
//!     without a head / masked-prediction encoder), FR-EX-08.
//!   - [`Emotion2VecWeights::from_gguf`] with a floor of non-empty tensor
//!     count enforced loud (a GGUF that carries zero tensors is refused
//!     rather than silently running an all-zero forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to [`LicenseClass::Unknown`]
//!     when the stamp is absent; a converter-produced GGUF surfaces
//!     [`LicenseClass::Permissive`] since emotion2vec+ ships MIT
//!     end-to-end).
//!
//! - **Loud-partial (this WP)**: [`Emotion2Vec::classify`] returns
//!   [`VokraError::UnsupportedOp`] naming the deferred wav2vec2-style SSL
//!   Transformer encoder + linear 9-way classifier head, echoing all three
//!   primary source URLs (HF release + FunASR reference code + ACL 2024
//!   paper) so a reader diagnosing this gap has exactly three places to
//!   walk. All 9 class labels are echoed verbatim so the reader can
//!   cross-check the output width the follow-up wave targets. **No
//!   fabricated emotion logits are ever emitted** (FR-EX-08 — no silent
//!   partial output).
//!
//! # Sibling family distinctness (wav2vec2-SSL-lineage neighbourhood)
//!
//! [`ARCH`] = `"emotion2vec"` is **deliberately distinct** from every
//! sibling wav2vec2-SSL-lineage arch tag — all four siblings share the
//! wav2vec2 SSL encoder lineage but expose completely different downstream
//! heads:
//!
//! - `wav2vec2_ctc` — Meta wav2vec2 base + CTC ASR head (character-level
//!   phone / letter output, not a fixed 9-class emotion output);
//! - `wavlm_sv` — Microsoft WavLM base + XVector speaker verification head
//!   (192-d / 512-d speaker embedding, not a classification vector);
//! - `hubert` — Meta HuBERT SSL encoder without a fixed downstream head
//!   (raw representation output; downstream head is added at fine-tune
//!   time);
//! - `data2vec-audio` — Meta data2vec self-supervised encoder
//!   (masked-prediction on continuous latents; downstream head added at
//!   fine-tune time).
//!
//! Silently sharing arch would let runtime dispatch mis-route an
//! emotion2vec+ checkpoint onto a CTC / XVector / bare-SSL loader — the
//! tensor-name walks would fail with a downstream missing-tensor error
//! instead of a specific arch-mismatch message. FR-EX-08 forbids the
//! silent shape misroute across wav2vec2-lineage arches.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] /
//! [`UPSTREAM_HF`] — same rule the sibling BF16 pass-through binders
//! (`hifigan` / `snac` / `pyannote` / `beat_this` / `mt3` / `musicgen` /
//! `conv_tasnet` / `sepformer` / `redimnet` / `sortformer_diar_4spk_v1` /
//! `audioldm2` / `audiogen` / `jasco` / `panns`) use so `vokra-models`
//! does not gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`, `vokra-core →
//! GGUF reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! emotion2vec+ Large ships upstream as a safetensors checkpoint driven by
//! a FunASR Python pipeline; this runtime **never** touches ONNX or
//! pickle (FR-LD-05 / NFR-DS-02). A future
//! `tools/parity/emotion2vec_prepare_checkpoint.py` sidecar (uv-managed
//! Python 3.12 per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) will front the converter for any variant
//! that ships as a pickle instead of pure safetensors, mirroring the
//! sibling audio-tagging / MIR bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/emotion2vec.rs`.
// See module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model emotion2vec`.
///
/// Distinct from every sibling wav2vec2-SSL-lineage arch tag —
/// `wav2vec2_ctc` (CTC ASR head), `wavlm_sv` (XVector speaker head),
/// `hubert` (bare SSL encoder), `data2vec-audio` (masked-prediction
/// encoder). Silent aliasing would misroute runtime dispatch to a
/// wrong-head loader (FR-EX-08 boundary — see the module docstring
/// "Sibling family distinctness" section).
pub const ARCH: &str = "emotion2vec";

/// Expected `vokra.model.name` value written by the converter — canonical
/// `emotion2vec/emotion2vec_plus_large` mirror slug (the Large variant of
/// the emotion2vec+ family).
pub const NAME: &str = "emotion2vec-plus-large";

/// Expected `vokra.model.category` value — the first `"emotion"` in the
/// converter tree. Consumed by the model-card generator + zoo manifest
/// tier gate so an emotion classifier is not accidentally advertised as
/// an ASR / TTS release.
pub const CATEGORY: &str = "emotion";

/// Upstream HuggingFace slug (mirror of the converter's
/// [`crate::emotion2vec::UPSTREAM_HF`] equivalent — recorded here so the
/// runtime binder can echo it in loud-partial diagnostics without
/// re-fetching a manifest).
pub const UPSTREAM_HF: &str = "emotion2vec/emotion2vec_plus_large";

/// Number of emotion classes in the emotion2vec+ Large downstream
/// classifier head. Primary source: Ma et al. 2024 §III-B +
/// upstream `label.txt`.
pub const N_CLASSES_EMOTION: u32 = 9;

/// Canonical emotion class labels in the exact upstream order (per the
/// upstream `label.txt` bundled with `emotion2vec/emotion2vec_plus_large`,
/// mirrored in the converter's module docstring for cross-crate
/// consistency).
///
/// The order is **load-bearing**: the classifier head emits a
/// 9-dimensional logit vector where index `i` corresponds to
/// `EMOTION_CLASS_LABELS[i]`. A silent reorder would misroute every
/// downstream consumer's `argmax` interpretation (FR-EX-08 — no silent
/// class permutation).
pub const EMOTION_CLASS_LABELS: [&str; 9] = [
    "Angry",
    "Disgusted",
    "Fearful",
    "Happy",
    "Neutral",
    "Other",
    "Sad",
    "Surprised",
    "<unk>",
];

// Primary-source URL constants — cited in the loud-partial error so a
// reader diagnosing the gap has fully specified anchors.

/// Primary-source anchor for the emotion2vec+ Large HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/emotion2vec/emotion2vec_plus_large";
/// Primary-source anchor for the FunASR reference wrapper.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/ddlBoJack/emotion2vec";
/// Primary-source anchor for the paper (Ma et al. 2024 ACL).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2312.15185";

// ---------------------------------------------------------------------------
// Emotion2VecWeights — non-empty tensor gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an emotion2vec+ Large GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// emotion2vec+ checkpoint; the wav2vec2-style SSL Transformer encoder
/// alone carries hundreds of Linear + LayerNorm + Conv1D parameters, so
/// an empty manifest always signals a mis-produced GGUF).
///
/// Under the current landing this struct stores the tensor names + GGUF-
/// side dims discovered on disk. The follow-up wave sizes its dequant per
/// its kernel needs — today only the count + names are consumed so a
/// future `Emotion2VecWeights::bind_encoder_weights` /
/// `bind_classifier_head_weights` tensor walk can find its inputs without
/// re-parsing the GGUF.
#[derive(Debug)]
pub struct Emotion2VecWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict` name
    /// with their GGUF-side dims. Used by the load-time non-emptiness
    /// gate and by the future follow-up wav2vec2-style SSL encoder +
    /// 9-way classifier head forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl Emotion2VecWeights {
    /// Scans `gguf` for the emotion2vec+ state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF
    /// is never a valid emotion2vec+ Large checkpoint).
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
                "emotion2vec: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate emotion2vec+ Large checkpoint carries \
                 hundreds of wav2vec2-style Linear + LayerNorm + Conv1D parameters \
                 (arch={ARCH}, name={NAME}); zero tensors always signals a mis-produced \
                 GGUF. Re-run `vokra-cli convert --model emotion2vec` against an upstream \
                 `{UPSTREAM_HF}` safetensors checkpoint."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up wav2vec2-style SSL encoder + classifier
    /// head forward wave uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Emotion2Vec — the runtime binder handle.
// ---------------------------------------------------------------------------

/// emotion2vec+ Large (`emotion2vec/emotion2vec_plus_large`, MIT) runtime
/// binder for 9-class speech emotion recognition.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`classify`](Self::classify) on a mono f32 PCM waveform (16 kHz per the
/// wav2vec2/HuBERT-lineage input convention) to obtain a 9-way emotion
/// logits vector. See the module doc for the current implementation-status
/// matrix and the FR-EX-08 loud-error contract on the deferred
/// wav2vec2-style SSL Transformer encoder + linear classifier head
/// composition.
#[derive(Debug)]
pub struct Emotion2Vec {
    // The bound weights are held (real, counted) but the wav2vec2-style
    // SSL encoder + 9-way classifier head composition is a follow-up
    // wave; the field is deliberately `#[allow(dead_code)]` until the
    // composition lands so a reader is not misled by an unused field.
    // Same posture as panns / audioldm2 / musicgen / redimnet / storm /
    // sortformer / pyannote / RMVPE / mt3 / beat_this.
    #[allow(dead_code)]
    weights: Emotion2VecWeights,
    weight_license: LicenseClass,
}

impl Emotion2Vec {
    /// Binds an emotion2vec+ Large GGUF: validates arch, discovers
    /// tensors, and surfaces the stamped weight-license class for the
    /// compliance-gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a distinct
    /// [`VokraError::ModelLoad`] naming the missing / wrong key so a
    /// reader diagnosing a mis-produced GGUF has exactly one place to walk
    /// (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"emotion2vec"` (a sibling wav2vec2-SSL-lineage GGUF handed
    ///   here by mistake — `wav2vec2_ctc` / `wavlm_sv` / `hubert` /
    ///   `data2vec-audio` — fails with a clear message instead of a
    ///   downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`Emotion2VecWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "emotion2vec: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model emotion2vec`? Note \
                     that sibling wav2vec2-SSL-lineage arch tags — `wav2vec2_ctc` \
                     (Meta wav2vec2 base + CTC ASR head, character-level phone/letter \
                     output), `wavlm_sv` (Microsoft WavLM base + XVector speaker \
                     verification head, 192-d/512-d speaker embedding), `hubert` \
                     (Meta HuBERT SSL encoder without a fixed downstream head), \
                     `data2vec-audio` (Meta data2vec self-supervised masked-prediction \
                     encoder) — all live in the same wav2vec2-SSL-lineage \
                     neighbourhood but have completely different downstream heads. \
                     emotion2vec+ Large's fixed 9-way emotion classifier head has no \
                     analog in any sibling — silently aliasing arch would misroute the \
                     runtime dispatch (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "emotion2vec: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native emotion2vec GGUF (was it produced by `vokra-cli \
                     convert --model emotion2vec`?)"
                        .to_owned(),
                ));
            }
        }

        // 2. Load the tensor manifest with the non-emptiness gate.
        let weights = Emotion2VecWeights::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for the compliance-gate cross-checks. The emotion2vec+
        //    converter stamps `Permissive` (MIT — verified 2026-07-25 per
        //    the converter docstring); a GGUF missing the stamp reads
        //    back as `Unknown` (fail-closed default per
        //    feedback-license-signoff-primary-source memory).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            weights,
            weight_license,
        })
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The emotion2vec+
    /// converter stamps `Permissive` (MIT — end-to-end per the
    /// `emotion2vec/emotion2vec_plus_large` model card `license: mit`
    /// verified 2026-07-25). A GGUF missing the stamp reads back as
    /// `Unknown` (fail-closed at the M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up wav2vec2-style SSL encoder + 9-way
    /// classifier head forward wave uses it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The canonical 9-emotion class label list (verbatim upstream order,
    /// see [`EMOTION_CLASS_LABELS`] for the ordering rationale). Returned
    /// as a `&'static` slice so consumers can `enumerate` alongside the
    /// classifier's logit vector once the follow-up wave lands.
    #[inline]
    #[must_use]
    pub const fn class_labels() -> &'static [&'static str; 9] {
        &EMOTION_CLASS_LABELS
    }

    /// The output width of the 9-way emotion classifier head
    /// ([`N_CLASSES_EMOTION`] = 9). Load-bearing const — a rename or drift
    /// must be caught by the test suite.
    #[inline]
    #[must_use]
    pub const fn num_classes() -> u32 {
        N_CLASSES_EMOTION
    }

    /// 9-way emotion classification of a PCM waveform.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the full emotion2vec+ Large
    /// forward requires the deferred wav2vec2-style SSL Transformer
    /// encoder + linear 9-way classifier head, which cannot be
    /// synthesized from the current binder scaffold without a real
    /// tensor-name walk against the upstream `emotion2vec_plus_large`
    /// safetensors manifest.
    ///
    /// The error names all three primary source URLs (HF release +
    /// FunASR reference code + ACL 2024 paper) so a reader diagnosing
    /// this gap has exactly three places to walk. All 9 class labels are
    /// echoed verbatim so the reader can cross-check what output width
    /// the follow-up wave targets. **No fabricated emotion logits are
    /// ever emitted** (FR-EX-08 — no silent partial output).
    ///
    /// The `_pcm` argument is treated as the raw waveform at 16 kHz mono
    /// f32 in `[-1, 1]` (wav2vec2/HuBERT-lineage input convention); shape
    /// / rate mismatch will be a loud error rather than a resample
    /// surprise when the real forward lands.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred wav2vec2-style SSL encoder + 9-way classifier head
    ///   composition.
    pub fn classify(&self, _pcm: &[f32]) -> Result<Vec<f32>> {
        // Bind explicitly so an unused-variable warning cannot mask a
        // future accidental removal of the parameter (mirror of the
        // panns / wavlm_sv loud-partial signature discipline).
        let _ = _pcm;
        Err(classify_forward_loud_partial())
    }
}

/// Construct the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Emotion2Vec::classify`] until the wav2vec2-style SSL encoder +
/// linear 9-way classifier head composition lands.
///
/// Names **three** primary source URLs (HF release + FunASR reference
/// code + ACL 2024 paper) so a reader diagnosing the gap has exactly
/// three places to walk. All 9 class labels are echoed verbatim so the
/// reader can cross-check the output-width the follow-up wave targets.
/// Mirror of the panns / audioldm2 / musicgen / conv_tasnet / redimnet /
/// storm / sortformer / RMVPE / pyannote / wavlm loud-partial-message
/// precedent (CLAUDE.md 教訓 (a)).
fn classify_forward_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "emotion2vec classify (loud-partial): the full forward is deferred; \
         two missing pieces must land before real logits can be emitted: \
         (1) wav2vec2-style SSL Transformer encoder walk — base topology \
         12 layers x 768 hidden per the wav2vec2 lineage, but the \
         emotion2vec+ Large variant's exact layer count / hidden dim \
         requires a real-checkpoint tensor-name walk since the converter \
         does not currently stamp `vokra.emotion2vec.*` axes; \
         (2) linear 9-way classifier head (utterance mean-pool over the \
         encoder time axis -> Linear(hidden, {n_classes}) per Ma et al. \
         2024 section III-B). Output width = {n_classes} emotion classes, \
         labels (verbatim upstream order, load-bearing for argmax \
         interpretation): [{c0}, {c1}, {c2}, {c3}, {c4}, {c5}, {c6}, {c7}, {c8}]. \
         Primary sources: HF release {hf}, FunASR reference code {code}, \
         paper {paper}. Runtime cannot fabricate an emotion logits array \
         (FR-EX-08 no silent partial output).",
        n_classes = N_CLASSES_EMOTION,
        c0 = EMOTION_CLASS_LABELS[0],
        c1 = EMOTION_CLASS_LABELS[1],
        c2 = EMOTION_CLASS_LABELS[2],
        c3 = EMOTION_CLASS_LABELS[3],
        c4 = EMOTION_CLASS_LABELS[4],
        c5 = EMOTION_CLASS_LABELS[5],
        c6 = EMOTION_CLASS_LABELS[6],
        c7 = EMOTION_CLASS_LABELS[7],
        c8 = EMOTION_CLASS_LABELS[8],
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_CODE,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the emotion2vec+ Large runtime binder — contract-constant
    //! pins + metadata round-trip + negative-space round-trip on the
    //! loud-partial gates + arch-tag distinctness pin + 9-class label
    //! ordering pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On a real 16 kHz PCM waveform
    //! this would be `classify(...)` returning a 9-way emotion logits
    //! vector, but the wav2vec2-style SSL encoder + linear classifier
    //! head composition is deferred (see the module doc +
    //! [`Emotion2Vec::classify`] rustdoc). Fabricating a real
    //! classification output would violate CLAUDE.md 教訓 (a)
    //! ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin**: `ARCH` / `NAME` / `CATEGORY` /
    //!    `N_CLASSES_EMOTION` / `UPSTREAM_HF` all match the converter's
    //!    values exactly (cross-crate consistency — a converter drift
    //!    without a binder-side follow-through would land here in the
    //!    same commit or fail the test).
    //! 2. **Class-label ordering pin**: the 9 emotion labels are pinned
    //!    verbatim in exact upstream order so a silent reorder cannot
    //!    misroute `argmax` interpretation.
    //! 3. **Metadata round-trip**: `from_gguf` reads arch + name +
    //!    category + license stamp + tensor manifest with the correct
    //!    surface semantics (Permissive stamp binds, Unknown fallback
    //!    fires when the stamp is absent).
    //! 4. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / empty tensor list / unsupported
    //!    forward surface) fires at its documented surface point, in the
    //!    documented error variant.
    //! 5. **Arch-tag distinctness pin**: the arch string is stable and
    //!    distinct from every sibling wav2vec2-SSL-lineage arch tag.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Helper: builds a legitimate emotion2vec+ GGUF (arch + name +
    /// category + optional weight-license stamp + one representative
    /// wav2vec2-style tensor). The tensor uses a placeholder upstream
    /// name (`encoder.embed_tokens.weight`, mirroring the converter test
    /// module's chosen sample) so the non-emptiness gate is satisfied.
    fn emotion2vec_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative wav2vec2-style tensor so the non-emptiness
        // gate passes. Uses a placeholder name matching the converter
        // test module's chosen sample for cross-file consistency.
        b.add_tensor(
            "encoder.embed_tokens.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate consistency with the
    //          converter)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_and_name_pins_are_stable() {
        assert_eq!(ARCH, "emotion2vec", "emotion2vec arch tag pin");
        assert_eq!(
            NAME, "emotion2vec-plus-large",
            "emotion2vec+ Large canonical name pin"
        );
        assert_eq!(
            CATEGORY, "emotion",
            "emotion2vec is the first `category=emotion` model in the converter tree"
        );
        assert_eq!(
            N_CLASSES_EMOTION, 9,
            "emotion2vec+ 9-class classifier head output width pin"
        );
        assert_eq!(
            UPSTREAM_HF, "emotion2vec/emotion2vec_plus_large",
            "upstream HF slug pin (used in loud-partial diagnostics)"
        );
        // The public accessor must mirror the constant.
        assert_eq!(Emotion2Vec::num_classes(), N_CLASSES_EMOTION);
    }

    // -----------------------------------------------------------------------
    // Test 2 — Emotion class-label ordering pin (a silent reorder would
    //          misroute `argmax` interpretation — FR-EX-08 no silent
    //          class permutation)
    // -----------------------------------------------------------------------

    #[test]
    fn emotion_class_labels_pin_matches_converter_docstring() {
        // Pin every label in exact upstream order (per the converter's
        // module docstring + upstream `label.txt`). A reorder / rename /
        // count-drift would land here in the same commit or fail this
        // test.
        assert_eq!(
            EMOTION_CLASS_LABELS,
            [
                "Angry",
                "Disgusted",
                "Fearful",
                "Happy",
                "Neutral",
                "Other",
                "Sad",
                "Surprised",
                "<unk>"
            ]
        );
        assert_eq!(
            EMOTION_CLASS_LABELS.len(),
            N_CLASSES_EMOTION as usize,
            "class-label array width must equal the classifier head output width"
        );
        // The public accessor must return exactly the same slice.
        assert_eq!(Emotion2Vec::class_labels(), &EMOTION_CLASS_LABELS);
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf metadata round-trip (Permissive stamp bound,
    //          non-empty tensor gate passed)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        // Build a legitimate GGUF with arch + name + category + Permissive
        // license stamp + one tensor. The binder must bind, surface the
        // Permissive license class (matching what the converter stamps for
        // emotion2vec+ Large's MIT license), and report at least one
        // tensor bound.
        let file = emotion2vec_gguf(Some(LicenseClass::Permissive));
        let e = Emotion2Vec::from_gguf(&file).expect("valid GGUF must bind");
        // License-class surface: Permissive per the converter's MIT stamp
        // (emotion2vec+ Large ships MIT end-to-end, verified 2026-07-25).
        assert_eq!(
            e.weight_license(),
            LicenseClass::Permissive,
            "Permissive stamp must round-trip (mirror of what the emotion2vec+ converter emits)"
        );
        assert!(
            e.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4 — from_gguf rejects wrong arch (never silently mis-routes
    //          across the wav2vec2-SSL-lineage neighbourhood)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `wavlm_sv` (WavLM + XVector speaker head) GGUF handed to the
        // emotion2vec binder by mistake must fail loud with a specific
        // message rather than silently mis-binding (FR-EX-08). WavLM's
        // XVector speaker embedding head (192-d/512-d) and emotion2vec+'s
        // 9-class emotion classifier head are completely different
        // downstream heads on top of a related SSL encoder, so silent
        // aliasing would misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "wavlm_sv");
        b.add_string(chunks::KEY_MODEL_NAME, "wavlm-base-plus-sv");
        b.add_tensor("wavlm.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Emotion2Vec::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`wavlm_sv`") && m.contains("`emotion2vec`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The message must enumerate the whole wav2vec2-SSL-
                // lineage sibling fleet so the reader has fully specified
                // anchors.
                for sibling in ["wav2vec2_ctc", "wavlm_sv", "hubert", "data2vec-audio"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                // The message must call out the head-topology divergence
                // (CTC vs XVector vs bare SSL vs emotion classifier).
                assert!(
                    m.contains("9-way emotion classifier"),
                    "message should call out the 9-way emotion classifier head divergence, \
                     got `{m}`"
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
    // Test 5 — from_gguf rejects missing arch (never silently binds an
    //          arch-unlabeled GGUF)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        // No `vokra.model.arch` stamp at all — the binder must refuse
        // with a "not a Vokra-native emotion2vec GGUF" diagnostic so a
        // caller who hands a raw GGUF (without the converter's arch
        // stamp) has a clear signal.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Emotion2Vec::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not a Vokra-native emotion2vec GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — Empty tensor manifest fails loud (never binds all-zero
    //          forward — FR-EX-08)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + name + license class but zero tensors — the
        // Emotion2VecWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Emotion2Vec::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model emotion2vec"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — classify loud-partial (returns UnsupportedOp naming
    //          wav2vec2 encoder + linear classifier head + all 3 primary
    //          sources + all 9 class labels + FR-EX-08 rationale)
    // -----------------------------------------------------------------------

    #[test]
    fn classify_loud_partial_returns_unsupported_op() {
        // A GGUF without the license stamp — the binder falls back to
        // Unknown but still binds (arch + tensor manifest are the load
        // gates, license is a compliance surface not a bind gate).
        let file = emotion2vec_gguf(None);
        let e = Emotion2Vec::from_gguf(&file).expect("valid arch must bind");
        assert_eq!(
            e.weight_license(),
            LicenseClass::Unknown,
            "missing license stamp must fail-closed to Unknown"
        );

        // Legitimate PCM shape: 1 s of silence at 16 kHz mono (the
        // wav2vec2/HuBERT-lineage input convention).
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = e.classify(&pcm) else {
            panic!("classify must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // Names the surface + posture.
                assert!(
                    msg.contains("emotion2vec classify"),
                    "surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Names the two missing pieces by exact identifier.
                assert!(
                    msg.contains("wav2vec2"),
                    "message must name the wav2vec2-style SSL encoder gap, got `{msg}`"
                );
                assert!(
                    msg.contains("linear 9-way classifier head")
                        || msg.contains("linear") && msg.contains("classifier head"),
                    "message must name the linear classifier head gap, got `{msg}`"
                );

                // Cites all three primary source URLs so a reader
                // diagnosing the gap has anchors to walk.
                for url in [PRIMARY_SOURCE_HF, PRIMARY_SOURCE_CODE, PRIMARY_SOURCE_PAPER] {
                    assert!(
                        msg.contains(url),
                        "expected primary source URL '{url}' cited: {msg}"
                    );
                }

                // All 9 class labels echoed verbatim (a silent reorder
                // would misroute `argmax` — FR-EX-08).
                for label in EMOTION_CLASS_LABELS.iter() {
                    assert!(
                        msg.contains(label),
                        "expected emotion class label '{label}' in error: {msg}"
                    );
                }

                // FR-EX-08 rationale cited.
                assert!(
                    msg.contains("FR-EX-08"),
                    "expected FR-EX-08 rationale for no fake logits: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
