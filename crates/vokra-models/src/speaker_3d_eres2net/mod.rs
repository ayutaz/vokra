//! **3D-Speaker ERes2Net** (`iic/speech_eres2net_sv_zh-cn_16k-common`,
//! Apache-2.0) — Alibaba DAMO's Enhanced Res2Net speaker-verification
//! encoder (Chen et al. 2023, arXiv:2305.12838 "An Enhanced Res2Net with
//! Local and Global Feature Fusion for Speaker Verification") — runtime
//! binder for the `speaker_3d` converter arch.
//!
//! Wave 9 2026-08-14 audit follow-up. Loud-partial per the emotion2vec /
//! moonshine / panns / redimnet / wavlm / storm precedent — CLAUDE.md
//! 教訓 (a): "loud-partial は fake-complete より honest".
//!
//! # Primary sources
//!
//! - HF release: <https://huggingface.co/iic/speech_eres2net_sv_zh-cn_16k-common>
//!   (`license: apache-2.0`, verified 2026-07-25 per the converter
//!   docstring — CLAUDE.md「ハルシネーション厳禁」)
//! - Reference code (Apache-2.0): <https://github.com/alibaba-damo-academy/3D-Speaker>
//!   (the 3D-Speaker toolkit hosts the ERes2Net reference implementation
//!   under `speakerlab/models/eres2net/`).
//! - Paper: Chen et al. 2023, *"An Enhanced Res2Net with Local and
//!   Global Feature Fusion for Speaker Verification"*
//!   (<https://arxiv.org/abs/2305.12838>) — the ERes2Net architecture
//!   the `speech_eres2net_sv_zh-cn_16k-common` release implements.
//!
//! # Architecture (transcribed from primary sources — Chen et al. 2023 §3 + 3D-Speaker toolkit)
//!
//! ```text
//! PCM (mono f32, 16 kHz)                          ← ERes2Net input convention
//!   -> Kaldi 80-band log-mel fbank + CMN            ← front-end (owner-side)
//!        (fbank front-end lives outside this binder — mirror of the
//!         sibling CAM++ (`crate::speaker::camplus`) contract: the
//!         audio->fbank front-end is a separate work item; this binder
//!         consumes an 80-dim fbank matrix per Kaldi convention).
//!   -> ERes2Net stem: Res2Net conv1 + BN + ReLU     ← **loud-partial**
//!        (upstream `speakerlab/models/eres2net/ERes2Net.py` — the
//!         initial 3×3 Conv2D with stride 1 followed by four
//!         Res2NetBlock stages with local + global feature fusion.
//!         Exact per-stage channel widths / block counts require a
//!         real-checkpoint tensor-name walk against the state_dict).
//!   -> Attentive Statistics Pooling head            ← **loud-partial**
//!        (ASP over the time axis — Chen et al. 2023 §3.3: temporal
//!         attention-weighted mean + std concatenation before the
//!         embedding projection).
//!   -> Linear embedding projection                  ← **loud-partial**
//!        (`Linear(embed_in, [`EMBEDDING_DIM`])` — the standard 192-d
//!         speaker embedding shared with the sibling CAM++ 3D-Speaker
//!         encoder (`crate::speaker::camplus::EMBED_DIM`) so downstream
//!         `spk_proj` / cosine similarity consumers see a compatible
//!         vector width).
//!   -> L2-normalized 192-d speaker embedding vector
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Speaker3dEres2Net::from_gguf`] with strict
//!     `vokra.model.arch == "speaker_3d"` validation. The sibling
//!     speaker-encoder arch tags (`campplus` / `xvector` / `ecapa_tdnn` /
//!     `titanet-large` / `wavlm_sv`) fail with a specific
//!     sibling-mis-route [`VokraError::ModelLoad`] enumerating the whole
//!     speaker-encoder fleet — silent aliasing would misroute the
//!     runtime dispatch to a family with a different downstream head
//!     topology (Res2Net + ASP vs TDNN + statistics pool vs XVector +
//!     stats pool vs SE-Res2Net + attention vs WavLM SSL + XVector),
//!     FR-EX-08.
//!   - [`Speaker3dEres2NetWeights::from_gguf`] with a floor of non-empty
//!     tensor count enforced loud (a GGUF that carries zero tensors is
//!     refused rather than silently running an all-zero forward —
//!     FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::Unknown`] when the stamp is absent; a
//!     converter-produced GGUF surfaces [`LicenseClass::Permissive`]
//!     since 3D-Speaker ERes2Net ships apache-2.0 end-to-end).
//!
//! - **Loud-partial (this WP)**: [`Speaker3dEres2Net::encode`] returns
//!   [`VokraError::UnsupportedOp`] naming the deferred ERes2Net stem +
//!   Res2NetBlock stages + Attentive Statistics Pooling head + linear
//!   embedding projection, echoing all three primary source URLs (HF
//!   release + 3D-Speaker toolkit reference code + arXiv 2305.12838
//!   paper) so a reader diagnosing this gap has exactly three places to
//!   walk. **No fabricated speaker embedding is ever emitted** (FR-EX-08
//!   — no silent partial output).
//!
//! # Sibling family distinctness (speaker-encoder neighbourhood)
//!
//! [`ARCH`] = `"speaker_3d"` is **deliberately distinct** from every
//! sibling speaker-encoder arch tag — all live in the same
//! speaker-verification neighbourhood (fbank -> encoder -> pooling ->
//! embedding) but with markedly different encoder / pooling topologies:
//!
//! - `campplus` — CAM++ (context-aware masking + densely connected TDNN,
//!   `crate::speaker::camplus`, the M0-08 first-class speaker encoder);
//! - `xvector` — Kaldi XVector (5-layer TDNN + statistics pooling);
//! - `ecapa_tdnn` — SpeechBrain ECAPA-TDNN (SE-Res2Net + attentive
//!   statistics pooling);
//! - `titanet-large` — NVIDIA TitaNet-L (ContextNet + attentive
//!   statistics pooling);
//! - `wavlm_sv` — Microsoft WavLM base + XVector speaker head (a
//!   wav2vec2-SSL-lineage encoder feeding an XVector-style head).
//!
//! Silently sharing arch would let runtime dispatch mis-route a 3D-Speaker
//! ERes2Net checkpoint onto a CAM++ / XVector / ECAPA-TDNN / TitaNet /
//! WavLM-SV loader — the tensor-name walks would fail with a downstream
//! missing-tensor error instead of a specific arch-mismatch message.
//! FR-EX-08 forbids the silent shape misroute across sibling speaker
//! arches.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] /
//! [`UPSTREAM_HF`] — same rule the sibling BF16 pass-through binders
//! (`snac` / `wavlm` / `pyannote` / `beat_this` / `mt3` / `musicgen` /
//! `conv_tasnet` / `sepformer` / `redimnet` / `sortformer_diar_4spk_v1` /
//! `audioldm2` / `audiogen` / `jasco` / `panns` / `emotion2vec` /
//! `moonshine`) use so `vokra-models` does not gain a dependency edge
//! onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! 3D-Speaker ERes2Net ships upstream as a safetensors checkpoint driven
//! by a Python pipeline (ModelScope + the 3D-Speaker toolkit); this
//! runtime **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02). A
//! future `tools/parity/speaker_3d_eres2net_prepare_checkpoint.py`
//! sidecar (uv-managed Python 3.12 per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`) will
//! front the converter for any variant that ships as a pickle instead of
//! pure safetensors, mirroring the sibling speaker / audio-tagging /
//! MIR bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/speaker_3d.rs`.
// See module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model speaker-3d-eres2net`.
///
/// Distinct from every sibling speaker-encoder arch tag —
/// `campplus` (CAM++ TDNN), `xvector` (Kaldi TDNN), `ecapa_tdnn`
/// (SpeechBrain SE-Res2Net), `titanet-large` (NVIDIA ContextNet),
/// `wavlm_sv` (Microsoft WavLM base + XVector head). Silent aliasing
/// would misroute runtime dispatch to a wrong-topology loader
/// (FR-EX-08 boundary — see the module docstring "Sibling family
/// distinctness" section).
pub const ARCH: &str = "speaker_3d";

/// Expected `vokra.model.name` value written by the converter — canonical
/// `iic/speech_eres2net_sv_zh-cn_16k-common` mirror slug (the primary
/// released variant of the 3D-Speaker ERes2Net family).
pub const NAME: &str = "speech_eres2net_sv_zh-cn_16k-common";

/// Expected `vokra.model.category` value — shared with the sibling CAM++
/// speaker encoder (`crate::speaker::camplus`). Consumed by the model-card
/// generator + zoo manifest tier gate so a speaker encoder is not
/// accidentally advertised as an ASR / TTS release.
pub const CATEGORY: &str = "speaker";

/// Upstream HuggingFace slug (mirror of the converter's `UPSTREAM_HF`
/// constant — recorded here so the runtime binder can echo it in
/// loud-partial diagnostics without re-fetching a manifest).
pub const UPSTREAM_HF: &str = "iic/speech_eres2net_sv_zh-cn_16k-common";

/// Kaldi fbank band count consumed by the ERes2Net stem (per the
/// upstream ModelScope release card + 3D-Speaker toolkit
/// `speakerlab/process/processor.py` FBank frontend defaults). Kept in
/// sync with the sibling CAM++ encoder's fbank convention
/// (`crate::speaker::camplus`) so both speaker binders consume the same
/// front-end.
///
/// **Load-bearing**: an ERes2Net checkpoint fed a 40-band or 64-band
/// fbank would silently corrupt the first-conv spatial layout — the
/// upstream release manifest fixes this at 80 (owner should re-verify
/// at bind time against the exact checkpoint the follow-up wave binds).
pub const N_MELS_FBANK: u32 = 80;

/// Raw PCM sample rate the Kaldi fbank front-end expects at 16 kHz
/// mono (per the release slug `..._zh-cn_16k-common`). Shared with the
/// sibling speaker encoders (`campplus` / `xvector` / `ecapa_tdnn` /
/// `titanet-large` / `wavlm_sv`) — the entire Vokra speaker-encoder
/// fleet keys on 16 kHz mono input.
pub const SAMPLE_RATE: u32 = 16_000;

/// Speaker embedding vector width surfaced by the ERes2Net encoder head
/// (per the upstream ModelScope release card — 192-d, matching the
/// sibling CAM++ `EMBED_DIM = 192` so both encoders' embeddings are
/// L2-cosine comparable and `spk_proj` consumers can dispatch on either
/// interchangeably).
///
/// **Load-bearing**: a downstream cosine-similarity / zero-shot TTS
/// speaker conditioning path assumes 192 for both CAM++ and ERes2Net; a
/// silent axis change would misroute every consumer. Owner should
/// re-verify at bind time against the real state_dict.
pub const EMBEDDING_DIM: u32 = 192;

// Primary-source URL constants — cited in the loud-partial error so a
// reader diagnosing the gap has fully specified anchors.

/// Primary-source anchor for the 3D-Speaker ERes2Net HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/iic/speech_eres2net_sv_zh-cn_16k-common";
/// Primary-source anchor for the 3D-Speaker toolkit reference code
/// (Alibaba DAMO Academy).
pub const PRIMARY_SOURCE_CODE: &str = "github.com/alibaba-damo-academy/3D-Speaker";
/// Primary-source anchor for the paper (Chen et al. 2023).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2305.12838";

// ---------------------------------------------------------------------------
// Speaker3dEres2NetWeights — non-empty tensor gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a 3D-Speaker ERes2Net GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// ERes2Net checkpoint; the Res2Net stem + four Res2NetBlock stages +
/// ASP head + linear embedding projection alone carry hundreds of
/// Conv2D + BN + Linear parameters, so an empty manifest always signals
/// a mis-produced GGUF).
///
/// Under the current landing this struct stores the tensor names + GGUF-
/// side dims discovered on disk. The follow-up wave sizes its dequant
/// per its kernel needs — today only the count + names are consumed so
/// a future `Speaker3dEres2NetWeights::bind_stem_weights` /
/// `bind_asp_head_weights` / `bind_embedding_projection_weights` tensor
/// walk can find its inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct Speaker3dEres2NetWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict` name
    /// with their GGUF-side dims. Used by the load-time non-emptiness
    /// gate and by the future follow-up ERes2Net stem + ASP head +
    /// embedding projection forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl Speaker3dEres2NetWeights {
    /// Scans `gguf` for the ERes2Net state_dict tensors. Refuses to bind
    /// if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is
    /// never a valid 3D-Speaker ERes2Net checkpoint).
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
                "speaker_3d (ERes2Net): GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate 3D-Speaker ERes2Net checkpoint \
                 carries hundreds of Res2Net Conv2D + BatchNorm + Linear parameters \
                 (arch={ARCH}, name={NAME}); zero tensors always signals a mis-produced \
                 GGUF. Re-run `vokra-cli convert --model speaker-3d-eres2net` against an \
                 upstream `{UPSTREAM_HF}` safetensors checkpoint."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up ERes2Net stem + ASP head + embedding
    /// projection forward wave uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Speaker3dEres2Net — the runtime binder handle.
// ---------------------------------------------------------------------------

/// 3D-Speaker ERes2Net (`iic/speech_eres2net_sv_zh-cn_16k-common`,
/// Apache-2.0) runtime binder for speaker verification / speaker
/// embedding.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`encode`](Self::encode) on an 80-band Kaldi log-mel fbank matrix
/// (per the shared Vokra speaker encoder fbank convention) to obtain a
/// 192-d L2-normalized speaker embedding. See the module doc for the
/// current implementation-status matrix and the FR-EX-08 loud-error
/// contract on the deferred ERes2Net stem + Res2NetBlock stages + ASP
/// head + linear embedding projection.
#[derive(Debug)]
pub struct Speaker3dEres2Net {
    // The bound weights are held (real, counted) but the ERes2Net stem +
    // Res2NetBlock stages + ASP head + linear embedding projection
    // composition is a follow-up wave; the field is deliberately
    // `#[allow(dead_code)]` until the composition lands so a reader is
    // not misled by an unused field. Same posture as panns / audioldm2 /
    // musicgen / redimnet / storm / sortformer / pyannote / RMVPE / mt3 /
    // beat_this / emotion2vec / moonshine.
    #[allow(dead_code)]
    weights: Speaker3dEres2NetWeights,
    weight_license: LicenseClass,
}

impl Speaker3dEres2Net {
    /// Binds a 3D-Speaker ERes2Net GGUF: validates arch, discovers
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
    ///   not `"speaker_3d"` (a sibling speaker-encoder GGUF handed here
    ///   by mistake — `campplus` / `xvector` / `ecapa_tdnn` /
    ///   `titanet-large` / `wavlm_sv` — fails with a clear message
    ///   instead of a downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`Speaker3dEres2NetWeights::from_gguf`] refuses to bind an
    ///   all-zero forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "speaker_3d (ERes2Net): GGUF arch is `{other}`, expected `{ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model \
                     speaker-3d-eres2net`? Note that sibling speaker-encoder arch tags — \
                     `campplus` (CAM++ context-aware masking + densely connected TDNN), \
                     `xvector` (Kaldi 5-layer TDNN + statistics pooling), `ecapa_tdnn` \
                     (SpeechBrain SE-Res2Net + attentive statistics pooling), \
                     `titanet-large` (NVIDIA ContextNet + attentive statistics pooling), \
                     `wavlm_sv` (Microsoft WavLM base + XVector speaker head) — all live \
                     in the same speaker-verification neighbourhood but have completely \
                     different encoder / pooling topologies. ERes2Net's Res2Net stem + \
                     four Res2NetBlock stages with local + global feature fusion + \
                     Attentive Statistics Pooling head has no analog in any sibling — \
                     silently aliasing arch would misroute the runtime dispatch \
                     (FR-EX-08 — no silent partial load). Primary source: \
                     https://github.com/alibaba-damo-academy/3D-Speaker"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "speaker_3d (ERes2Net): GGUF is missing `vokra.model.arch` — this \
                     is not a Vokra-native 3D-Speaker ERes2Net GGUF (was it produced by \
                     `vokra-cli convert --model speaker-3d-eres2net`?). Primary source: \
                     https://github.com/alibaba-damo-academy/3D-Speaker"
                        .to_owned(),
                ));
            }
        }

        // 2. Load the tensor manifest with the non-emptiness gate.
        let weights = Speaker3dEres2NetWeights::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for the compliance-gate cross-checks. The 3D-Speaker
        //    ERes2Net converter stamps `Permissive` (apache-2.0 — verified
        //    2026-07-25 per the converter docstring); a GGUF missing the
        //    stamp reads back as `Unknown` (fail-closed default per
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
    /// `vokra.provenance.weight_license` chunk. The 3D-Speaker ERes2Net
    /// converter stamps `Permissive` (apache-2.0 — end-to-end per the
    /// `iic/speech_eres2net_sv_zh-cn_16k-common` model card
    /// `license: apache-2.0` verified 2026-07-25). A GGUF missing the
    /// stamp reads back as `Unknown` (fail-closed at the M2-13 compliance
    /// gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up ERes2Net stem + ASP head + embedding
    /// projection forward wave uses it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The output width of the ERes2Net linear embedding projection
    /// ([`EMBEDDING_DIM`] = 192, matching the sibling CAM++
    /// `EMBED_DIM`). Load-bearing const — a rename or drift must be
    /// caught by the test suite.
    #[inline]
    #[must_use]
    pub const fn embedding_dim() -> u32 {
        EMBEDDING_DIM
    }

    /// The Kaldi fbank band count expected at the encoder input
    /// ([`N_MELS_FBANK`] = 80). Load-bearing const — a rename or drift
    /// must be caught by the test suite.
    #[inline]
    #[must_use]
    pub const fn n_mels_fbank() -> u32 {
        N_MELS_FBANK
    }

    /// Encodes an 80-band Kaldi log-mel fbank matrix into a 192-d
    /// speaker embedding vector.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the full 3D-Speaker
    /// ERes2Net forward requires the deferred ERes2Net stem + four
    /// Res2NetBlock stages with local + global feature fusion +
    /// Attentive Statistics Pooling head + linear embedding projection,
    /// which cannot be synthesized from the current binder scaffold
    /// without a real tensor-name walk against the upstream
    /// `iic/speech_eres2net_sv_zh-cn_16k-common` safetensors manifest.
    ///
    /// The error names all three primary source URLs (HF release +
    /// 3D-Speaker toolkit reference code + arXiv 2305.12838 paper) so a
    /// reader diagnosing this gap has exactly three places to walk. The
    /// expected embedding width is echoed so the reader can cross-check
    /// what output shape the follow-up wave targets. **No fabricated
    /// speaker embedding is ever emitted** (FR-EX-08 — no silent partial
    /// output).
    ///
    /// The `_fbank` argument is treated as a row-major matrix with
    /// [`N_MELS_FBANK`] = 80 bands per frame (mirror of the sibling
    /// `crate::speaker::camplus` fbank convention); shape / band-count
    /// mismatch will be a loud error rather than a silent shape
    /// surprise when the real forward lands.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `fbank` is empty (an empty
    ///   input cannot produce an embedding; caller passed the wrong
    ///   buffer). Fires **before** the loud-partial gate so the caller
    ///   sees the actionable error (fix the input), not the deeper
    ///   "primitive missing" error.
    /// - [`VokraError::UnsupportedOp`] on non-empty input — the
    ///   loud-partial gate for the deferred ERes2Net stem + ASP head +
    ///   embedding projection composition.
    pub fn encode(&self, fbank: &[f32]) -> Result<Vec<f32>> {
        if fbank.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "speaker_3d (ERes2Net) encode: input fbank buffer is empty \
                 (expected non-empty {N_MELS_FBANK}-band Kaldi log-mel fbank \
                 matrix at {SAMPLE_RATE} Hz mono; an empty buffer cannot \
                 produce a speaker embedding — FR-EX-08, never a silent \
                 zero-vector return)"
            )));
        }
        Err(encode_forward_loud_partial())
    }
}

/// Construct the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Speaker3dEres2Net::encode`] until the ERes2Net stem + four
/// Res2NetBlock stages + Attentive Statistics Pooling head + linear
/// embedding projection composition lands.
///
/// Names **three** primary source URLs (HF release + 3D-Speaker toolkit
/// reference code + arXiv 2305.12838 paper) so a reader diagnosing the
/// gap has exactly three places to walk. The expected embedding width +
/// fbank band count are echoed verbatim so the reader can cross-check
/// the output-shape the follow-up wave targets. Mirror of the emotion2vec
/// / moonshine / panns / audioldm2 / musicgen / conv_tasnet / redimnet /
/// storm / sortformer / RMVPE / pyannote / wavlm loud-partial-message
/// precedent (CLAUDE.md 教訓 (a)).
fn encode_forward_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "speaker_3d (ERes2Net) encode (loud-partial): the full forward is \
         deferred; three missing pieces must land before real embeddings can \
         be emitted: (1) ERes2Net stem walk — initial 3x3 Conv2D + BatchNorm \
         + ReLU followed by four Res2NetBlock stages with local + global \
         feature fusion (upstream reference \
         `speakerlab/models/eres2net/ERes2Net.py` in the 3D-Speaker toolkit; \
         the exact per-stage channel widths / block counts require a \
         real-checkpoint tensor-name walk since the converter does not \
         currently stamp `vokra.speaker_3d.*` axes); \
         (2) Attentive Statistics Pooling head (temporal attention-weighted \
         mean + std concatenation over the time axis, Chen et al. 2023 \
         section 3.3); \
         (3) Linear embedding projection (Linear(embed_in, {embed_dim}) + \
         L2-normalize — the standard 192-d speaker embedding shared with \
         sibling CAM++ so downstream `spk_proj` / cosine similarity \
         consumers see a compatible vector width). \
         Expected input: {n_mels}-band Kaldi log-mel fbank matrix at \
         {sample_rate} Hz mono (shared with sibling CAM++). \
         Expected output: {embed_dim}-d L2-normalized speaker embedding \
         vector. Primary sources: HF release {hf}, 3D-Speaker toolkit \
         reference code {code}, paper {paper}. Runtime cannot fabricate a \
         speaker embedding array (FR-EX-08 no silent partial output).",
        embed_dim = EMBEDDING_DIM,
        n_mels = N_MELS_FBANK,
        sample_rate = SAMPLE_RATE,
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
    //! Tests for the 3D-Speaker ERes2Net runtime binder — contract-constant
    //! pins + metadata round-trip + negative-space round-trip on the
    //! loud-partial gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On a real Kaldi fbank matrix
    //! this would be `encode(...)` returning a 192-d L2-normalized speaker
    //! embedding, but the ERes2Net stem + Res2NetBlock stages + ASP head +
    //! linear embedding projection composition is deferred (see the module
    //! doc + [`Speaker3dEres2Net::encode`] rustdoc). Fabricating a real
    //! embedding output would violate CLAUDE.md 教訓 (a) ("loud-partial
    //! は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin**: `ARCH` / `NAME` / `CATEGORY` /
    //!    `EMBEDDING_DIM` / `N_MELS_FBANK` / `SAMPLE_RATE` / `UPSTREAM_HF`
    //!    all match the converter's values exactly (cross-crate
    //!    consistency — a converter drift without a binder-side
    //!    follow-through would land here in the same commit or fail the
    //!    test).
    //! 2. **Metadata round-trip**: `from_gguf` reads arch + name +
    //!    category + license stamp + tensor manifest with the correct
    //!    surface semantics (Permissive stamp binds, Unknown fallback
    //!    fires when the stamp is absent).
    //! 3. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / empty tensor list / empty fbank /
    //!    unsupported forward surface) fires at its documented surface
    //!    point, in the documented error variant.
    //! 4. **Arch-tag distinctness pin**: the arch string is stable and
    //!    distinct from every sibling speaker-encoder arch tag.
    //! 5. **Loud-partial message pin**: the encode() error message cites
    //!    all three primary source URLs + the expected embedding width +
    //!    the expected fbank band count so a follow-up wave has one place
    //!    to walk.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Helper: builds a legitimate 3D-Speaker ERes2Net GGUF (arch + name
    /// + category + optional weight-license stamp + one representative
    /// ERes2Net-style tensor). The tensor uses a placeholder upstream
    /// name (`stem.conv1.weight`, mirroring the upstream
    /// `speakerlab/models/eres2net/ERes2Net.py` stem convention) so the
    /// non-emptiness gate is satisfied.
    fn eres2net_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative ERes2Net stem tensor so the non-emptiness
        // gate passes. Uses a placeholder name matching the upstream
        // `speakerlab/models/eres2net/ERes2Net.py` stem convention.
        b.add_tensor(
            "stem.conv1.weight",
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
        assert_eq!(ARCH, "speaker_3d", "3D-Speaker ERes2Net arch tag pin");
        assert_eq!(
            NAME, "speech_eres2net_sv_zh-cn_16k-common",
            "3D-Speaker ERes2Net canonical name pin"
        );
        assert_eq!(
            CATEGORY, "speaker",
            "3D-Speaker ERes2Net shares the `speaker` category with sibling CAM++"
        );
        assert_eq!(
            EMBEDDING_DIM, 192,
            "ERes2Net linear embedding projection output width pin (matches CAM++)"
        );
        assert_eq!(
            N_MELS_FBANK, 80,
            "ERes2Net Kaldi fbank band count pin (matches CAM++ / sibling speaker encoders)"
        );
        assert_eq!(
            SAMPLE_RATE, 16_000,
            "3D-Speaker ERes2Net 16 kHz mono sample rate pin"
        );
        assert_eq!(
            UPSTREAM_HF, "iic/speech_eres2net_sv_zh-cn_16k-common",
            "upstream HF slug pin (used in loud-partial diagnostics)"
        );
        // The public accessors must mirror the constants.
        assert_eq!(Speaker3dEres2Net::embedding_dim(), EMBEDDING_DIM);
        assert_eq!(Speaker3dEres2Net::n_mels_fbank(), N_MELS_FBANK);
    }

    // -----------------------------------------------------------------------
    // Test 2 — from_gguf metadata round-trip (Permissive stamp bound,
    //          non-empty tensor gate passed)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        // Build a legitimate GGUF with arch + name + category + Permissive
        // license stamp + one tensor. The binder must bind, surface the
        // Permissive license class (matching what the converter stamps
        // for 3D-Speaker ERes2Net's apache-2.0 license), and report at
        // least one tensor bound.
        let file = eres2net_gguf(Some(LicenseClass::Permissive));
        let e = Speaker3dEres2Net::from_gguf(&file).expect("valid GGUF must bind");
        // License-class surface: Permissive per the converter's
        // apache-2.0 stamp (3D-Speaker ERes2Net ships apache-2.0
        // end-to-end, verified 2026-07-25).
        assert_eq!(
            e.weight_license(),
            LicenseClass::Permissive,
            "Permissive stamp must round-trip (mirror of what the 3D-Speaker ERes2Net \
             converter emits)"
        );
        assert!(
            e.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    #[test]
    fn from_gguf_defaults_weight_license_to_unknown_when_missing() {
        // A GGUF missing `vokra.provenance.weight_license` reads back as
        // `Unknown` (fail-closed at the compliance gate). Never a silent
        // Permissive default.
        let file = eres2net_gguf(None);
        let e = Speaker3dEres2Net::from_gguf(&file).expect("missing provenance must still bind");
        assert_eq!(e.weight_license(), LicenseClass::Unknown);
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf rejects wrong arch (never silently mis-routes
    //          across the speaker-encoder neighbourhood)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `campplus` (CAM++ TDNN) GGUF handed to the ERes2Net binder
        // by mistake must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08). CAM++'s densely connected TDNN
        // and ERes2Net's Res2Net stem + Res2NetBlock stages are
        // completely different encoder topologies, so silent aliasing
        // would misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "campplus");
        b.add_string(chunks::KEY_MODEL_NAME, "campplus_zh_16k");
        b.add_tensor("campplus.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Speaker3dEres2Net::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`campplus`") && m.contains("`speaker_3d`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The message must enumerate the whole speaker-encoder
                // sibling fleet so the reader has fully specified
                // anchors.
                for sibling in [
                    "campplus",
                    "xvector",
                    "ecapa_tdnn",
                    "titanet-large",
                    "wavlm_sv",
                ] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                // The message must call out the ERes2Net-distinguishing
                // topology (Res2Net stem + ASP head).
                assert!(
                    m.contains("Res2Net"),
                    "message should call out the Res2Net stem divergence, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                assert!(
                    m.contains("github.com/alibaba-damo-academy/3D-Speaker"),
                    "message must cite the 3D-Speaker toolkit primary source URL, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — from_gguf rejects missing arch (never silently binds an
    //          arch-unlabeled GGUF)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        // No `vokra.model.arch` stamp at all — the binder must refuse
        // with a "not a Vokra-native 3D-Speaker ERes2Net GGUF"
        // diagnostic so a caller who hands a raw GGUF (without the
        // converter's arch stamp) has a clear signal.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Speaker3dEres2Net::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not a Vokra-native 3D-Speaker ERes2Net GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("github.com/alibaba-damo-academy/3D-Speaker"),
                    "message must cite the 3D-Speaker toolkit primary source URL, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — Empty tensor manifest fails loud (never binds all-zero
    //          forward — FR-EX-08)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + name + license class but zero tensors — the
        // Speaker3dEres2NetWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Speaker3dEres2Net::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model speaker-3d-eres2net"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — encode rejects empty fbank with actionable InvalidArgument
    //          (caller sees the fix-your-input error, not the deeper
    //          loud-partial gate)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_empty_fbank_is_invalid_argument() {
        let file = eres2net_gguf(Some(LicenseClass::Permissive));
        let e = Speaker3dEres2Net::from_gguf(&file).unwrap();
        let Err(err) = e.encode(&[]) else {
            panic!("empty fbank must be rejected");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("empty"),
                    "message must call out the empty fbank, got `{msg}`"
                );
                assert!(
                    msg.contains("80"),
                    "message must name the expected fbank band count (80), got `{msg}`"
                );
                assert!(
                    msg.contains("16000"),
                    "message must name the expected sample rate (16000), got `{msg}`"
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
    // Test 7 — encode loud-partial (returns UnsupportedOp naming ERes2Net
    //          stem + ASP head + linear embedding projection + all 3
    //          primary sources + expected embedding width + FR-EX-08
    //          rationale)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partial_returns_unsupported_op() {
        // A well-shaped non-empty input fires the loud-partial gate.
        // The gate message must cite ALL three primary source URLs and
        // call out the ERes2Net-distinguishing "Res2Net stem + ASP head"
        // trait so the follow-up wave knows exactly where to look.
        let file = eres2net_gguf(Some(LicenseClass::Permissive));
        let e = Speaker3dEres2Net::from_gguf(&file).unwrap();
        // A 1-frame, 80-band fbank matrix — legitimate shape, so the
        // loud-partial gate is what fires (not the empty-buffer guard).
        let fbank = vec![0.0f32; N_MELS_FBANK as usize];
        let Err(err) = e.encode(&fbank) else {
            panic!("non-empty fbank must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // Names the surface + posture.
                assert!(
                    msg.contains("speaker_3d (ERes2Net) encode"),
                    "surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Names the three missing pieces by exact identifier.
                assert!(
                    msg.contains("ERes2Net stem"),
                    "message must name the ERes2Net stem gap, got `{msg}`"
                );
                assert!(
                    msg.contains("Res2NetBlock"),
                    "message must name the Res2NetBlock stages gap, got `{msg}`"
                );
                assert!(
                    msg.contains("Attentive Statistics Pooling"),
                    "message must name the ASP head gap, got `{msg}`"
                );
                assert!(
                    msg.contains("Linear embedding projection"),
                    "message must name the linear embedding projection gap, got `{msg}`"
                );

                // Cites all three primary source URLs so a reader
                // diagnosing the gap has anchors to walk.
                for url in [PRIMARY_SOURCE_HF, PRIMARY_SOURCE_CODE, PRIMARY_SOURCE_PAPER] {
                    assert!(
                        msg.contains(url),
                        "expected primary source URL '{url}' cited: {msg}"
                    );
                }

                // Expected output width echoed (192-d, shared with CAM++).
                assert!(
                    msg.contains("192"),
                    "message must echo the expected embedding width (192), got `{msg}`"
                );
                // Expected input band count echoed (80-band Kaldi fbank).
                assert!(
                    msg.contains("80-band"),
                    "message must echo the expected fbank band count (80-band), got `{msg}`"
                );
                // Expected sample rate echoed (16 kHz).
                assert!(
                    msg.contains("16000"),
                    "message must echo the expected sample rate (16000 Hz), got `{msg}`"
                );

                // FR-EX-08 rationale cited.
                assert!(
                    msg.contains("FR-EX-08"),
                    "expected FR-EX-08 rationale for no fake embedding: {msg}"
                );

                // The message must call out CAM++ as the sibling encoder
                // with a compatible embedding width so a follow-up wave
                // consumer knows the vector-width contract.
                assert!(
                    msg.contains("CAM++"),
                    "message should call out sibling CAM++ (shared 192-d embedding), \
                     got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
