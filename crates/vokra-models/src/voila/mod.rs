//! **Voila** (`maitrix-org/Voila`, MIT, 2025) — Maitrix's full-duplex
//! speech-to-speech dialog family runtime binder, loud-partial per
//! `llama_omni2` / `moshi` / `csm` full-duplex S2S precedent and per
//! moonshine / emotion2vec / storm / redimnet / musicgen / audioldm2
//! shape-only-scaffold precedent (Wave 9 2026-08-14 audit follow-up —
//! CLAUDE.md 教訓 (a): "loud-partial は fake-complete より honest").
//!
//! # Primary sources
//!
//! - Reference code: <https://github.com/maitrix-org/Voila> (MIT)
//! - Project page / HF collection: <https://huggingface.co/maitrix-org>
//!   (Voila family — `Voila-base`, `Voila-chat`, `Voila-audio-alpha`,
//!   `Voila-autonomous-preview` — sizes / exact release names may drift as
//!   upstream ships new variants; owner should verify at bind time per
//!   CLAUDE.md 「ハルシネーション厳禁」).
//!
//! # What Voila is (primary source)
//!
//! Voila is a **full-duplex** speech-to-speech dialog family from Maitrix
//! that streams user audio in and model audio out on separate concurrent
//! channels (barge-in / talk-over supported). Upstream reports ~195 ms
//! response latency (per the release blog + repo README — owner should
//! verify against the exact commit shipping in a given checkpoint).
//!
//! Voila lives in the same **full-duplex S2S neighbourhood** as
//! [`crate::moshi`] and [`crate::csm`] (both stream-in / stream-out with
//! interruption) — distinct from the **streaming (half-duplex) S2S**
//! sibling [`crate::llama_omni2`] which processes one utterance at a time
//! without concurrent input / output channels. Silently sharing the
//! `voila` arch tag with any of these siblings would misroute runtime
//! dispatch onto a differently-shaped session manager (FR-EX-08 — no
//! silent op-shape misroute).
//!
//! # Runtime layout (loud-partial, `llama_omni2` / `moshi` / `csm` full-duplex precedent)
//!
//! ```text
//! user PCM stream (mono f32, 16 kHz, arriving concurrently with output)
//!   -> full-duplex session manager                    ← **loud-partial**
//!        (concurrent input / output streams,
//!         barge-in / talk-over handling, ~195 ms
//!         end-to-end latency budget per primary source
//!         release blog — mirror of moshi / csm session
//!         code architecturally, but Voila's session
//!         topology is distinct and requires its own
//!         binding pass)
//!   -> Whisper-lineage speech encoder                 ← **loud-partial**
//!        (raw PCM -> semantic features; exact encoder
//!         backbone / hidden dim / layer count deferred
//!         to real-checkpoint dump since the converter
//!         does not currently stamp `vokra.voila.*`
//!         axes — mirror of the emotion2vec / moonshine
//!         "converter does not yet stamp variant axes"
//!         posture)
//!   -> Voila LLM backbone forward                     ← **loud-partial**
//!        (transformer decoder that consumes speech
//!         features + previous turn context to produce
//!         speech tokens; exact backbone family / depth
//!         / width deferred to real-checkpoint dump —
//!         primary source names the family but per-
//!         release axes shift across Voila-base /
//!         Voila-chat / Voila-audio-alpha /
//!         Voila-autonomous-preview)
//!   -> speech decoder + vocoder head                  ← **loud-partial**
//!        (speech tokens -> output PCM; upstream ships
//!         a neural vocoder integrated with the LLM
//!         backbone — see the sibling `hifigan` /
//!         `bigvgan` / `vocos` binders for the vocoder-
//!         family neighbourhood the follow-up wave will
//!         plug into)
//!   -> model PCM stream (mono f32, 16 kHz, streamed
//!      out concurrently with input arrival)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Voila::from_gguf`] with strict `vokra.model.arch == "voila"`
//!     validation. The sibling full-duplex / S2S arch tags (`moshi` /
//!     `csm` / `llama_omni2`) fail with a specific sibling-mis-route
//!     [`VokraError::ModelLoad`] enumerating the S2S-family
//!     neighbourhood — silent aliasing would misroute the runtime dispatch
//!     to a differently-shaped session manager (FR-EX-08).
//!   - [`VoilaWeights::from_gguf`] with a floor of non-empty tensor
//!     count enforced loud (a GGUF that carries zero tensors is refused
//!     rather than silently running an all-zero forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::Unknown`] when the stamp is absent; a converter-
//!     produced GGUF surfaces [`LicenseClass::Permissive`] since Voila
//!     ships MIT end-to-end).
//!
//! - **Loud-partial (this WP)**: [`Voila::converse`] returns
//!   [`VokraError::UnsupportedOp`] naming **four** exact missing pieces:
//!   (i) full-duplex session manager, (ii) Whisper-lineage speech encoder,
//!   (iii) Voila LLM backbone forward, (iv) speech decoder + vocoder head.
//!   The message cites the primary source URL
//!   (`github.com/maitrix-org/Voila`) so a reader diagnosing this gap has
//!   exactly one anchor to walk. **No fabricated speech output is ever
//!   emitted** (FR-EX-08 — no silent zero-fill / noise stream).
//!
//! # Sibling family distinctness (S2S neighbourhood)
//!
//! [`ARCH`] = `"voila"` is **deliberately distinct** from every sibling
//! S2S arch tag:
//!
//! - `moshi` — Kyutai's full-duplex speech dialog (Mimi codec + inner
//!   monologue), CC-BY 4.0 attribution required;
//! - `csm` — Sesame CSM-1B full-duplex conversational speech model
//!   (Mimi codec + depth transformer), Apache 2.0;
//! - `llama_omni2` — ICTNLP streaming (half-duplex) S2S (Qwen2.5
//!   backbone + Whisper encoder), Apache 2.0.
//!
//! All four live in the S2S family neighbourhood but differ in duplex
//! discipline (full-duplex vs streaming half-duplex), codec / vocoder
//! choice (Mimi vs integrated vocoder), and backbone family (Helium /
//! Sesame / Qwen2.5 / Voila). Silently sharing arch would let runtime
//! dispatch mis-route a Voila checkpoint onto a mismatched session
//! manager — FR-EX-08 forbids the silent shape misroute across S2S
//! sibling arches.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] /
//! [`UPSTREAM_HF`] constants — same rule the sibling BF16 pass-through
//! binders (`hifigan` / `snac` / `pyannote` / `beat_this` / `mt3` /
//! `musicgen` / `conv_tasnet` / `sepformer` / `redimnet` /
//! `sortformer_diar_4spk_v1` / `audioldm2` / `audiogen` / `jasco` /
//! `panns` / `emotion2vec` / `moonshine`) use so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
//! reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Voila ships upstream as a PyTorch checkpoint driven by a Python
//! pipeline; this runtime **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). A future
//! `tools/parity/voila_prepare_checkpoint.py` sidecar (uv-managed
//! Python 3.12 per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) will front the converter for any variant
//! that ships as a pickle instead of pure safetensors.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of the converter side. See the module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model voila`.
///
/// Distinct from every sibling S2S arch tag — `moshi` (Kyutai full-duplex,
/// CC-BY 4.0), `csm` (Sesame full-duplex, Apache 2.0), `llama_omni2`
/// (ICTNLP streaming half-duplex, Apache 2.0). Silent aliasing would
/// misroute runtime dispatch to a differently-shaped session manager
/// (FR-EX-08 boundary — see the module docstring "Sibling family
/// distinctness" section).
pub const ARCH: &str = "voila";

/// Expected `vokra.model.name` value written by the converter — canonical
/// `maitrix-org/Voila` mirror slug. Kept as a single tag for now (variant
/// discrimination — `Voila-base` / `Voila-chat` / `Voila-audio-alpha` /
/// `Voila-autonomous-preview` — is deferred to a follow-up wave since
/// upstream axis publishing is still stabilising; owner should verify at
/// bind time per CLAUDE.md 「ハルシネーション厳禁」).
pub const NAME: &str = "voila";

/// Expected `vokra.model.category` value — the S2S dialog family
/// neighbourhood. Consumed by the model-card generator + zoo manifest
/// tier gate so a full-duplex dialog release is not accidentally
/// advertised as an ASR / TTS release.
pub const CATEGORY: &str = "s2s";

/// Upstream reference-code slug (mirror of the converter's
/// `UPSTREAM_HF` / `UPSTREAM_REPO` equivalent — recorded here so the
/// runtime binder can echo it in loud-partial diagnostics without
/// re-fetching a manifest).
pub const UPSTREAM_HF: &str = "maitrix-org/Voila";

/// Raw PCM sample rate the Voila speech encoder consumes at the
/// front-end. Pinned at 16 kHz per the wav2vec2 / Whisper-lineage input
/// convention shared with every sibling speech-encoder-fronted S2S
/// model in this crate (`moshi` / `csm` / `llama_omni2`); owner should
/// verify at bind time against the exact release manifest since
/// Voila may ship 24 kHz variants in future releases.
pub const VOILA_SAMPLE_RATE: u32 = 16_000;

/// Upstream reported end-to-end response latency per the release
/// blog + repo README (~195 ms). Purely informational — echoed in the
/// loud-partial diagnostic so a reader sees the design target the
/// follow-up wave is chasing. Owner should verify against the exact
/// commit shipping in a given checkpoint (CLAUDE.md 「ハルシネーション厳禁」).
pub const VOILA_TARGET_LATENCY_MS: u32 = 195;

// Primary-source URL constant — cited in the loud-partial error so a
// reader diagnosing the gap has a fully specified anchor.

/// Primary-source anchor for the Voila reference code repo.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/maitrix-org/Voila";

// ---------------------------------------------------------------------------
// VoilaWeights — non-empty tensor gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a Voila GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// Voila checkpoint; the LLM backbone + speech encoder + speech decoder +
/// vocoder head together carry hundreds of Linear + LayerNorm + Conv1D
/// parameters, so an empty manifest always signals a mis-produced GGUF).
///
/// Under the current landing this struct stores the tensor names + GGUF-
/// side dims discovered on disk. The follow-up wave sizes its dequant
/// per its kernel needs — today only the count + names are consumed so a
/// future `VoilaWeights::bind_backbone_weights` /
/// `bind_speech_encoder_weights` / `bind_speech_decoder_weights` tensor
/// walk can find its inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct VoilaWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict` name
    /// with their GGUF-side dims. Used by the load-time non-emptiness
    /// gate and by the future follow-up full-duplex session + speech
    /// encoder + LLM backbone + speech decoder forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl VoilaWeights {
    /// Scans `gguf` for the Voila state_dict tensors. Refuses to bind if
    /// the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is never a
    /// valid Voila checkpoint).
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
                "voila: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate Voila checkpoint carries hundreds \
                 of LLM backbone + speech encoder + speech decoder + vocoder head \
                 Linear + LayerNorm + Conv1D parameters (arch={ARCH}, name={NAME}); \
                 zero tensors always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model voila` against an upstream \
                 `{UPSTREAM_HF}` PyTorch checkpoint."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up full-duplex session + speech encoder +
    /// LLM backbone + speech decoder forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Voila — the runtime binder handle.
// ---------------------------------------------------------------------------

/// Voila (`maitrix-org/Voila`, MIT) runtime binder for full-duplex
/// speech-to-speech dialog.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`converse`](Self::converse) on a mono f32 PCM waveform (16 kHz per
/// the Whisper-lineage input convention) to drive the full-duplex
/// session. See the module doc for the current implementation-status
/// matrix and the FR-EX-08 loud-error contract on the deferred
/// full-duplex session + speech encoder + LLM backbone + speech decoder
/// composition.
#[derive(Debug)]
pub struct Voila {
    // The bound weights are held (real, counted) but the full-duplex
    // session + speech encoder + LLM backbone + speech decoder
    // composition is a follow-up wave; the field is deliberately
    // `#[allow(dead_code)]` until the composition lands so a reader is
    // not misled by an unused field. Same posture as
    // moonshine / emotion2vec / panns / audioldm2 / musicgen / redimnet /
    // storm / sortformer / pyannote / RMVPE / mt3 / beat_this.
    #[allow(dead_code)]
    weights: VoilaWeights,
    weight_license: LicenseClass,
}

impl Voila {
    /// Binds a Voila GGUF: validates arch, discovers tensors, and
    /// surfaces the stamped weight-license class for the compliance-gate
    /// cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong key
    /// so a reader diagnosing a mis-produced GGUF has exactly one place
    /// to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"voila"` (a sibling S2S GGUF handed here by mistake —
    ///   `moshi` / `csm` / `llama_omni2` — fails with a clear message
    ///   instead of a downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`VoilaWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "voila: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model voila`? Note that \
                     sibling S2S arch tags — `moshi` (Kyutai full-duplex, CC-BY 4.0 \
                     attribution required, Mimi codec + inner monologue), `csm` \
                     (Sesame CSM-1B full-duplex, Apache 2.0, Mimi codec + depth \
                     transformer), `llama_omni2` (ICTNLP streaming half-duplex, \
                     Apache 2.0, Qwen2.5 backbone + Whisper encoder) — all live in \
                     the S2S family neighbourhood but differ in duplex discipline, \
                     codec / vocoder choice, and backbone family. Voila's full-duplex \
                     session manager has no shape-analog in any sibling — silently \
                     aliasing arch would misroute the runtime dispatch (FR-EX-08 — \
                     no silent partial load). Primary source: \
                     https://{PRIMARY_SOURCE_CODE}."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "voila: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native voila GGUF (was it produced by `vokra-cli \
                     convert --model voila`?). Primary source: https://{PRIMARY_SOURCE_CODE}."
                )));
            }
        }

        // 2. Load the tensor manifest with the non-emptiness gate.
        let weights = VoilaWeights::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for the compliance-gate cross-checks. The Voila converter
        //    stamps `Permissive` (MIT — end-to-end per the
        //    `maitrix-org/Voila` repo LICENSE); a GGUF missing the stamp
        //    reads back as `Unknown` (fail-closed default per
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

    /// Convenience wrapper for [`Self::from_gguf`] that opens the GGUF
    /// from a filesystem path first.
    ///
    /// # Errors
    ///
    /// Any error surfaced by `GgufFile::open` (IO / parse), or any error
    /// surfaced by [`Self::from_gguf`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_gguf(&file)
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The Voila converter
    /// stamps `Permissive` (MIT — end-to-end per the `maitrix-org/Voila`
    /// repo LICENSE). A GGUF missing the stamp reads back as `Unknown`
    /// (fail-closed at the M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up full-duplex session + speech encoder +
    /// LLM backbone + speech decoder forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Full-duplex speech-to-speech converse: input PCM at
    /// [`VOILA_SAMPLE_RATE`] (16 kHz), output PCM at the same rate.
    ///
    /// This is the primary full-duplex S2S entry point. **Real weights
    /// required** and **real forward not yet bound**: the full-duplex
    /// session manager + Whisper-lineage speech encoder + Voila LLM
    /// backbone forward + speech decoder + vocoder head composition
    /// cannot be synthesized from the current binder scaffold without a
    /// real tensor-name walk against the upstream `maitrix-org/Voila`
    /// PyTorch manifest.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming **four** exact
    /// missing pieces so a follow-up wave can flip the switch without
    /// cross-referencing rustdoc:
    /// (i) full-duplex session manager (concurrent input / output
    ///     streams, barge-in / talk-over handling, ~195 ms target
    ///     latency — mirror of moshi / csm session code architecturally),
    /// (ii) Whisper-lineage speech encoder (raw PCM -> semantic
    ///      features; exact backbone / hidden dim / layer count deferred
    ///      to real-checkpoint dump),
    /// (iii) Voila LLM backbone forward (transformer decoder that
    ///       consumes speech features + previous turn context to produce
    ///       speech tokens),
    /// (iv) speech decoder + vocoder head (speech tokens -> output PCM;
    ///      upstream ships a neural vocoder integrated with the LLM
    ///      backbone).
    ///
    /// The message cites the primary source URL
    /// (`github.com/maitrix-org/Voila`) so a reader diagnosing this gap
    /// has exactly one anchor to walk. **No fabricated speech output is
    /// ever emitted** (FR-EX-08 — no silent zero-fill / noise stream).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `pcm_in` is empty (an
    ///   empty input cannot produce a dialog turn; caller passed the
    ///   wrong buffer). Fires **before** the loud-partial gate so the
    ///   caller sees the actionable error (fix the input), not the
    ///   deeper "primitive missing" error.
    /// - [`VokraError::InvalidArgument`] when any sample is non-finite
    ///   (NaN / +Inf / -Inf) — reject at the boundary (FR-EX-08)
    ///   before the deeper loud-partial gate fires.
    /// - [`VokraError::UnsupportedOp`] on well-shaped input — the
    ///   loud-partial gate documented above.
    pub fn converse(&self, pcm_in: &[f32]) -> Result<Vec<f32>> {
        if pcm_in.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "voila converse: input PCM buffer is empty (expected non-empty \
                 {VOILA_SAMPLE_RATE} Hz mono f32 raw waveform; an empty buffer \
                 cannot produce a dialog turn — FR-EX-08, never a silent \
                 empty-vector return)"
            )));
        }
        for (i, sample) in pcm_in.iter().enumerate() {
            if !sample.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "voila converse: pcm_in[{i}]={sample} is not finite \
                     (NaN / +Inf / -Inf) — reject at the boundary (FR-EX-08)"
                )));
            }
        }
        Err(converse_forward_loud_partial())
    }
}

// ---------------------------------------------------------------------------
// Loud-partial constructor — one per surface point, so an owner (or a
// follow-up CC wave) reading the error message knows exactly where to flip
// the switch. Names the primary source URL so no searching is required.
// ---------------------------------------------------------------------------

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Voila::converse`] until the real forward body lands.
///
/// Names the four specific missing pieces (full-duplex session manager,
/// Whisper-lineage speech encoder, Voila LLM backbone forward, speech
/// decoder + vocoder head) plus the primary source URL a reader would
/// need. Mirror of the llama_omni2 / moonshine / emotion2vec loud-
/// partial-message precedent — one place to walk when the switch gets
/// flipped (CLAUDE.md 教訓 (a)).
fn converse_forward_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "voila converse (loud-partial): the full full-duplex S2S forward is \
         deferred; four missing pieces must land before real dialog audio can \
         be emitted: \
         (i) full-duplex session manager — concurrent input / output streams \
         with barge-in / talk-over handling, ~{VOILA_TARGET_LATENCY_MS} ms \
         end-to-end response latency budget per upstream release blog + repo \
         README (mirror of moshi / csm full-duplex session code \
         architecturally, but Voila's session topology is distinct and \
         requires its own binding pass); \
         (ii) Whisper-lineage speech encoder — raw PCM ({VOILA_SAMPLE_RATE} Hz \
         mono f32) -> semantic features; exact encoder backbone / hidden dim \
         / layer count deferred to real-checkpoint dump since the converter \
         does not currently stamp `vokra.voila.*` axes; \
         (iii) Voila LLM backbone forward — transformer decoder that consumes \
         speech features + previous turn context to produce speech tokens; \
         exact backbone family / depth / width deferred to real-checkpoint \
         dump since per-release axes shift across Voila-base / Voila-chat / \
         Voila-audio-alpha / Voila-autonomous-preview; \
         (iv) speech decoder + vocoder head — speech tokens -> output PCM at \
         {VOILA_SAMPLE_RATE} Hz; upstream ships a neural vocoder integrated \
         with the LLM backbone (see sibling `hifigan` / `bigvgan` / `vocos` \
         binders for the vocoder-family neighbourhood the follow-up wave will \
         plug into). \
         Primary source: https://{PRIMARY_SOURCE_CODE} (MIT, Maitrix 2025). \
         Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
         より honest') — no silent fabricated speech ever emitted \
         (FR-EX-08 — no silent zero-fill / noise stream)."
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Voila runtime binder — contract-constant pins +
    //! metadata round-trip + negative-space round-trip on the loud-partial
    //! gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real 16 kHz PCM waveform this would be `converse(...)`
    //! producing streamed output PCM, but the full-duplex session +
    //! speech encoder + LLM backbone + speech decoder composition is
    //! deferred (see the module doc + [`Voila::converse`] rustdoc).
    //! Fabricating a real S2S output would violate CLAUDE.md 教訓 (a)
    //! ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin**: `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `VOILA_SAMPLE_RATE` / `VOILA_TARGET_LATENCY_MS`
    //!    all match the converter's values exactly (cross-crate
    //!    consistency — a converter drift without a binder-side follow-
    //!    through would land here in the same commit or fail the test).
    //! 2. **Metadata round-trip**: `from_gguf` reads arch + license
    //!    stamp + tensor manifest with the correct surface semantics
    //!    (Permissive stamp binds, Unknown fallback fires when the stamp
    //!    is absent).
    //! 3. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / empty tensor list / empty PCM /
    //!    non-finite PCM / unsupported forward surface) fires at its
    //!    documented surface point, in the documented error variant.
    //! 4. **Arch-tag distinctness pin**: the arch string is stable and
    //!    distinct from every sibling S2S arch tag.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Helper: builds a legitimate Voila GGUF (arch + name + category +
    /// optional weight-license stamp + one representative tensor). The
    /// tensor uses a placeholder upstream name so the non-emptiness gate
    /// is satisfied.
    fn voila_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative tensor so the non-emptiness gate passes.
        // Placeholder name — the follow-up wave will bind against the
        // real upstream tensor-name manifest.
        b.add_tensor(
            "backbone.embed_tokens.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate consistency with the
    //          converter + arch-tag distinctness pin)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_name_category_and_source_pins_are_stable() {
        assert_eq!(ARCH, "voila", "voila arch tag pin");
        assert_eq!(NAME, "voila", "voila canonical name pin");
        assert_eq!(
            CATEGORY, "s2s",
            "voila lives in the S2S family neighbourhood"
        );
        assert_eq!(
            UPSTREAM_HF, "maitrix-org/Voila",
            "upstream slug pin (used in loud-partial diagnostics)"
        );
        assert_eq!(
            PRIMARY_SOURCE_CODE, "github.com/maitrix-org/Voila",
            "primary-source URL pin (cited in every loud-partial message)"
        );
        assert_eq!(
            VOILA_SAMPLE_RATE, 16_000,
            "voila front-end sample rate pin (Whisper-lineage convention)"
        );
        assert_eq!(
            VOILA_TARGET_LATENCY_MS, 195,
            "upstream reported ~195 ms end-to-end response latency pin"
        );
        // Distinct from every sibling S2S arch — silent aliasing would
        // misroute runtime dispatch to a differently-shaped session
        // manager (FR-EX-08).
        for sibling in ["moshi", "csm", "llama_omni2"] {
            assert_ne!(
                ARCH, sibling,
                "voila arch must be distinct from sibling S2S arch `{sibling}`"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — from_gguf metadata round-trip (Permissive stamp bound,
    //          non-empty tensor gate passed)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        // Build a legitimate GGUF with arch + name + category + Permissive
        // license stamp + one tensor. The binder must bind, surface the
        // Permissive license class (matching what the converter stamps for
        // Voila's MIT license), and report at least one tensor bound.
        let file = voila_gguf(Some(LicenseClass::Permissive));
        let v = Voila::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            v.weight_license(),
            LicenseClass::Permissive,
            "Permissive stamp must round-trip (mirror of what the voila converter emits)"
        );
        assert!(
            v.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf defaults weight-license to Unknown when the
    //          stamp is absent (fail-closed default at the compliance
    //          gate — see feedback-license-signoff-primary-source memory)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_defaults_weight_license_to_unknown_when_missing() {
        // A GGUF missing `vokra.provenance.weight_license` reads back as
        // `Unknown` (fail-closed at the M2-13 compliance gate). Never a
        // silent Permissive default.
        let file = voila_gguf(None);
        let v = Voila::from_gguf(&file).expect("missing provenance must still bind");
        assert_eq!(v.weight_license(), LicenseClass::Unknown);
    }

    // -----------------------------------------------------------------------
    // Test 4 — from_gguf rejects wrong arch (never silently mis-routes
    //          across the S2S neighbourhood — FR-EX-08)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `moshi` (Kyutai full-duplex, Mimi codec) GGUF handed to the
        // Voila binder by mistake must fail loud with a specific message
        // rather than silently mis-binding (FR-EX-08). Moshi's Mimi-
        // integrated session code and Voila's session manager are
        // completely different downstream shapes, so silent aliasing
        // would misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "moshi");
        b.add_string(chunks::KEY_MODEL_NAME, "moshiko");
        b.add_tensor("moshi.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Voila::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`moshi`") && m.contains("`voila`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The message must enumerate the whole S2S sibling fleet
                // so the reader has fully specified anchors.
                for sibling in ["moshi", "csm", "llama_omni2"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                // The primary source URL must be cited so the reader can
                // walk the arch family without cross-referencing
                // rustdoc.
                assert!(
                    m.contains("github.com/maitrix-org/Voila"),
                    "message must cite the primary source URL, got `{m}`"
                );
                // The message must cite the FR-EX-08 clause.
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
        // with a "not a Vokra-native voila GGUF" diagnostic so a caller
        // who hands a raw GGUF (without the converter's arch stamp) has
        // a clear signal.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Voila::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not a Vokra-native voila GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("github.com/maitrix-org/Voila"),
                    "message must cite the primary source URL, got `{m}`"
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
        // VoilaWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Voila::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model voila"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — converse empty PCM is InvalidArgument (actionable input
    //          error surfaces before the deeper loud-partial gate)
    // -----------------------------------------------------------------------

    #[test]
    fn converse_empty_pcm_is_invalid_argument() {
        // Empty PCM cannot produce a dialog turn; the caller sees the
        // actionable InvalidArgument (fix the input), not the deeper
        // loud-partial gate (which they can't fix at all).
        let file = voila_gguf(Some(LicenseClass::Permissive));
        let v = Voila::from_gguf(&file).unwrap();
        let Err(err) = v.converse(&[]) else {
            panic!("empty pcm must be rejected");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("empty"),
                    "message must call out the empty PCM, got `{msg}`"
                );
                assert!(
                    msg.contains("16000"),
                    "message must name the expected sample rate, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — converse non-finite PCM is InvalidArgument (boundary
    //          rejection fires before the loud-partial gate — FR-EX-08)
    // -----------------------------------------------------------------------

    #[test]
    fn converse_non_finite_pcm_is_invalid_argument() {
        // A NaN / +Inf / -Inf sample in the input must be caught at the
        // boundary — running the loud-partial (or worse, a real forward)
        // over a poisoned buffer would produce non-diagnosable
        // downstream failures.
        let file = voila_gguf(Some(LicenseClass::Permissive));
        let v = Voila::from_gguf(&file).unwrap();
        let mut pcm = vec![0.0_f32; 16_000];
        pcm[7] = f32::NAN;
        let Err(err) = v.converse(&pcm) else {
            panic!("non-finite pcm must be rejected");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("not finite"),
                    "message must call out the non-finite sample, got `{msg}`"
                );
                assert!(
                    msg.contains("[7]"),
                    "message must name the offending index, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — converse loud-partial (returns UnsupportedOp naming all
    //          four missing pieces + primary source URL + FR-EX-08
    //          rationale + latency budget)
    // -----------------------------------------------------------------------

    #[test]
    fn converse_loud_partial_returns_unsupported_op() {
        // A well-shaped non-empty input fires the loud-partial gate.
        // The gate message must cite the primary source URL and name
        // all four missing pieces so the follow-up wave knows exactly
        // where to look.
        let file = voila_gguf(Some(LicenseClass::Permissive));
        let v = Voila::from_gguf(&file).unwrap();
        // 1 s of silence at 16 kHz — legitimate input shape, so the
        // loud-partial gate is what fires (not the empty-buffer / non-
        // finite guards).
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = v.converse(&pcm) else {
            panic!("non-empty pcm must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // Names the surface + posture.
                assert!(
                    msg.contains("voila converse"),
                    "surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // The four missing pieces must all be named (i, ii, iii, iv).
                assert!(
                    msg.contains("(i)")
                        && msg.contains("(ii)")
                        && msg.contains("(iii)")
                        && msg.contains("(iv)"),
                    "message must enumerate all four missing pieces, got `{msg}`"
                );

                // The four distinct primitives must all be called out.
                assert!(
                    msg.contains("full-duplex session"),
                    "message must name the full-duplex session manager gap, got `{msg}`"
                );
                assert!(
                    msg.contains("speech encoder"),
                    "message must name the Whisper-lineage speech encoder gap, got `{msg}`"
                );
                assert!(
                    msg.contains("LLM backbone"),
                    "message must name the Voila LLM backbone forward gap, got `{msg}`"
                );
                assert!(
                    msg.contains("speech decoder") && msg.contains("vocoder"),
                    "message must name the speech decoder + vocoder head gap, got `{msg}`"
                );

                // The primary source URL must be cited so a follow-up
                // wave has one place to walk.
                assert!(
                    msg.contains("github.com/maitrix-org/Voila"),
                    "message must cite the primary source URL, got `{msg}`"
                );

                // The upstream-reported latency budget must be echoed so
                // the reader sees the design target the follow-up wave
                // is chasing.
                assert!(
                    msg.contains("195 ms"),
                    "message must echo the ~195 ms latency target, got `{msg}`"
                );

                // The sample rate must be echoed so a follow-up wave
                // does not silently swap in a different rate.
                assert!(
                    msg.contains("16000"),
                    "message must echo the 16 kHz sample rate, got `{msg}`"
                );

                // FR-EX-08 rationale cited.
                assert!(
                    msg.contains("FR-EX-08"),
                    "expected FR-EX-08 rationale for no fake output: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
