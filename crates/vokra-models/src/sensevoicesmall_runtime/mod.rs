//! **FunAudioLLM SenseVoiceSmall** (`FunAudioLLM/SenseVoiceSmall`,
//! **FunASR MODEL_LICENSE** — fail-closed to
//! [`LicenseClass::Unknown`]) — multi-task speech understanding runtime
//! binder for the `sensevoicesmall` converter arch (Wave 9
//! 2026-08-14 audit follow-up, loud-partial per emotion2vec /
//! moonshine / panns / redimnet / wavlm / storm / musicgen /
//! audioldm2 precedent — CLAUDE.md 教訓 (a): "loud-partial は
//! fake-complete より honest").
//!
//! # Module-name disambiguation
//!
//! The converter-side module is called [`crate::sensevoicesmall`]-**wait**,
//! actually the converter lives in `vokra-convert`
//! (`crates/vokra-convert/src/models/sensevoicesmall.rs`). Historically
//! *runtime* modules and *converter* modules share the same crate-local
//! name, but at this point in Wave 9 the runtime side would clash with
//! a pre-existing `sensevoicesmall` symbol namespace inside
//! `vokra-models` were one to appear, so this runtime module is filed
//! under `sensevoicesmall_runtime` per the task spec to keep the name
//! deliberately unambiguous. The wire-format handshake
//! ([`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`]) is the
//! cross-crate contract, not the Rust module name.
//!
//! # Primary sources
//!
//! - HF release: <https://huggingface.co/FunAudioLLM/SenseVoiceSmall>
//!   (`license: FunASR MODEL_LICENSE`, verified against the upstream
//!   `github.com/modelscope/FunASR/blob/main/MODEL_LICENSE` per the
//!   sibling converter's `DEFAULT_LICENSE_SPDX` — CLAUDE.md
//!   「ハルシネーション厳禁」)
//! - Reference code: <https://github.com/FunAudioLLM/SenseVoice>
//!   (the FunAudioLLM umbrella hosts the reference wrapper; the
//!   underlying FunASR runtime lives at
//!   `github.com/modelscope/FunASR`)
//! - Paper: An et al. 2024, *"SenseVoice: A Multilingual Speech
//!   Recognition and Understanding Model"*
//!   (<https://arxiv.org/abs/2407.04051>) — reports **~15x lower
//!   latency than Whisper-Large** on comparable inputs (paper §V,
//!   TableIII); the runtime binder cannot verify this claim without
//!   the actual forward wired, so the number is echoed in the
//!   loud-partial diagnostic as a follow-up-wave target rather than
//!   asserted here.
//!
//! # Architecture (transcribed from primary sources — An et al. 2024
//! # §III + FunASR upstream `sensevoice/sense_voice_encoder.py`)
//!
//! ```text
//! PCM (mono f32, 16 kHz)                    ← FunASR input convention
//!   -> log-mel + FunASR `am.mvn` global stats normalization  ← **loud-partial**
//!        (kaldi-style fbank; the converter does not currently
//!         stamp `vokra.sensevoicesmall.*` axes since the
//!         upstream release ships the mean/var stats as a
//!         side-car `am.mvn` file rather than baking them
//!         into the checkpoint)
//!   -> SAN-M enhanced Conformer encoder      ← **loud-partial**
//!        (SAN-M = Modified Multi-Head Attention with a
//!         parallel Memory-block Fully-Connected branch,
//!         per An et al. 2024 §III-A + FunASR
//!         `funasr/models/sanm/`; distinct from vanilla
//!         Conformer used by Parakeet / Kotoba-Whisper /
//!         Reazonspeech-NeMo-v2 in that the memory block
//!         is a parallel branch, not a sequential module)
//!   -> shared encoder embedding + four per-task heads  ← **loud-partial**
//!        (i) **ASR head** — multilingual token output, 50
//!            languages (zh / en / ja / yue / ko among the
//!            primary release axes per the paper §IV);
//!        (ii) **LID head** — language identification
//!             classification;
//!        (iii) **SER head** — speech emotion recognition
//!              classification;
//!        (iv) **AED head** — audio event detection
//!             classification.
//!   -> multi-task output tuple (ASR ids + LID label + SER
//!      label + AED label) — no fabricated task outputs
//!      ever emitted (FR-EX-08).
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`SenseVoiceSmall::from_gguf`] with strict
//!     `vokra.model.arch == "sensevoicesmall"` validation. The
//!     sibling ASR arch tags (`whisper` / `distil_whisper` /
//!     `kotoba_whisper` / `moonshine` / `parakeet` / `parakeet_ctc` /
//!     `canary` / `canary_qwen` / `omniasr_ctc` / `kyutai_stt` /
//!     `reazonspeech_nemo_v2`) fail with a specific sibling-mis-route
//!     [`VokraError::ModelLoad`] enumerating the whole ASR-family
//!     fleet — silent aliasing would misroute the runtime dispatch to
//!     a single-task ASR loader missing the LID / SER / AED heads
//!     (FR-EX-08).
//!   - [`SenseVoiceSmallWeights::from_gguf`] with a floor of
//!     non-empty tensor count enforced loud (a GGUF that carries zero
//!     tensors is refused rather than silently running an all-zero
//!     forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::Unknown`] — the SenseVoiceSmall upstream
//!     licence is `FunASR MODEL_LICENSE`, a custom licence that is
//!     NOT one of the SPDX ids
//!     [`vokra_core::LicenseClass::from_class_str`] classifies, so
//!     the fail-closed default is correct until owner sign-off).
//!
//! - **Loud-partial (this WP)**: [`SenseVoiceSmall::transcribe`]
//!   returns [`VokraError::UnsupportedOp`] naming the SAN-M enhanced
//!   Conformer encoder + four per-task heads (ASR + LID + SER + AED)
//!   and echoing all three primary source URLs so a reader
//!   diagnosing this gap has exactly three places to walk. The
//!   distinguishing SAN-M topology (parallel Memory-block Fully-
//!   Connected branch alongside Modified Multi-Head Attention) is
//!   called out verbatim so a follow-up wave landing a vanilla
//!   Conformer here would silently misroute against every
//!   sibling ASR encoder. **No fabricated ASR ids or task labels
//!   are ever emitted** (FR-EX-08 — no silent partial output).
//!
//! # Sibling family distinctness (ASR-family neighbourhood)
//!
//! [`ARCH`] = `"sensevoicesmall"` is **deliberately distinct** from
//! every sibling ASR arch tag — all live in the "16 kHz PCM in,
//! text out" family but SenseVoice is the *only* one on this arm
//! shipping four per-task heads:
//!
//! - `whisper` / `distil_whisper` / `kotoba_whisper` — Whisper-family
//!   transformer encoder-decoder with mel front-end + single ASR
//!   head (single-task);
//! - `moonshine` — raw-audio Conv1D stem + RoPE + SwiGLU
//!   transformer encoder-decoder + single ASR head (single-task,
//!   no mel front-end);
//! - `parakeet` / `parakeet_ctc` — NVIDIA NeMo FastConformer / Parakeet
//!   encoder + CTC / TDT head (single-task);
//! - `canary` / `canary_qwen` — NVIDIA Canary FastConformer +
//!   Transformer decoder (single-task multilingual ASR); the
//!   Qwen-decoder sibling adds a Voxtral-style LLM decoder but
//!   still only ASR;
//! - `omniasr_ctc` — CTC-only single-task ASR;
//! - `kyutai_stt` — Kyutai streaming STT (single-task);
//! - `reazonspeech_nemo_v2` — NeMo Conformer + CTC / Transducer
//!   (single-task Japanese ASR).
//!
//! Silently sharing arch would let runtime dispatch mis-route a
//! SenseVoiceSmall checkpoint onto a single-task-head loader — the
//! tensor-name walks would fail with a downstream missing-tensor
//! error instead of a specific arch-mismatch message. FR-EX-08
//! forbids the silent shape misroute across ASR-family arches.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] /
//! [`UPSTREAM_HF`] — same rule the sibling BF16 pass-through binders
//! (`hifigan` / `snac` / `pyannote` / `beat_this` / `mt3` /
//! `musicgen` / `conv_tasnet` / `sepformer` / `redimnet` /
//! `sortformer_diar_4spk_v1` / `audioldm2` / `audiogen` / `jasco` /
//! `panns` / `emotion2vec` / `moonshine`) use so `vokra-models` does
//! not gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! SenseVoiceSmall ships upstream as a torch pickle (`model.pt`)
//! alongside auxiliary text assets; the converter side pre-flattens
//! the pickle to safetensors offline via the existing
//! `tools/parity/nemo_pt_to_safetensors.py --allow-strip-any`
//! bridge (mirrored across Canary / Parakeet / Reazonspeech-NeMo-v2)
//! — pickles never enter the runtime (FR-LD-05 / NFR-DS-02).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/sensevoicesmall.rs`. See the module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model sensevoicesmall`.
///
/// Distinct from every sibling ASR arch tag — `whisper` /
/// `distil_whisper` / `kotoba_whisper` (single-task mel-front-end
/// encoder-decoder), `moonshine` (single-task raw-audio Conv1D
/// encoder-decoder), `parakeet` / `parakeet_ctc` (single-task
/// NeMo FastConformer + CTC/TDT), `canary` / `canary_qwen`
/// (single-task NeMo FastConformer + transformer/Qwen decoder),
/// `omniasr_ctc` (single-task CTC-only), `kyutai_stt`
/// (single-task streaming), `reazonspeech_nemo_v2` (single-task
/// Japanese NeMo Conformer). Silent aliasing would misroute
/// runtime dispatch to a single-task loader missing the LID /
/// SER / AED heads (FR-EX-08 boundary — see the module docstring
/// "Sibling family distinctness" section).
pub const ARCH: &str = "sensevoicesmall";

/// Expected `vokra.model.name` value written by the converter —
/// canonical `FunAudioLLM/SenseVoiceSmall` mirror slug's short
/// name.
pub const NAME: &str = "sensevoicesmall";

/// Expected `vokra.model.category` value — collapses the audit
/// ticket's `asr-multitask-zh` label to the shorter `asr` variant
/// so runtime dispatch and model-card grouping stay uniform with
/// the sibling `asr` family. The multi-task / language axis rides
/// in the model name and per-task heads rather than multiplying
/// category labels.
pub const CATEGORY: &str = "asr";

/// Upstream HuggingFace slug (mirror of the converter's
/// [`crate::sensevoicesmall_runtime::UPSTREAM_HF`] equivalent —
/// recorded here so the runtime binder can echo it in
/// loud-partial diagnostics without re-fetching a manifest).
pub const UPSTREAM_HF: &str = "FunAudioLLM/SenseVoiceSmall";

/// Number of languages officially supported by the SenseVoice
/// family per An et al. 2024 §IV (SenseVoice paper, TableI). The
/// primary release specifically enumerates zh / en / ja / yue /
/// ko among the top-tier axes; the paper claims 50-language
/// coverage overall. Echoed verbatim in the loud-partial
/// diagnostic so a follow-up wave has an exact target for the
/// LID head's output width.
pub const N_LANGUAGES: u32 = 50;

/// Approximate speedup vs Whisper-Large reported by An et al.
/// 2024 §V (SenseVoice paper, TableIII). Echoed verbatim in the
/// loud-partial diagnostic as a follow-up-wave target rather
/// than a runtime-asserted number — the binder cannot verify
/// this claim without the actual forward wired, so it is
/// documented but not enforced (CLAUDE.md
/// 「ハルシネーション厳禁」).
pub const APPROX_SPEEDUP_VS_WHISPER: u32 = 15;

/// The four per-task head names in a fixed order (matches the
/// tuple ordering An et al. 2024 §III-B uses when describing
/// the multi-task output). The order is **load-bearing** for
/// any downstream consumer indexing into the multi-task output
/// tuple.
pub const PER_TASK_HEADS: [&str; 4] = [
    "ASR", "LID", // Language IDentification
    "SER", // Speech Emotion Recognition
    "AED", // Audio Event Detection
];

// Primary-source URL constants — cited in the loud-partial error
// so a reader diagnosing the gap has fully specified anchors.

/// Primary-source anchor for the SenseVoiceSmall HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/FunAudioLLM/SenseVoiceSmall";
/// Primary-source anchor for the FunAudioLLM reference code
/// (the umbrella that hosts SenseVoice's reference wrapper; the
/// underlying FunASR runtime lives at
/// `github.com/modelscope/FunASR`).
pub const PRIMARY_SOURCE_CODE: &str = "github.com/FunAudioLLM/SenseVoice";
/// Primary-source anchor for the paper (An et al. 2024).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2407.04051";

// ---------------------------------------------------------------------------
// SenseVoiceSmallWeights — non-empty tensor gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a SenseVoiceSmall GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is
/// never a valid SenseVoiceSmall checkpoint; the SAN-M enhanced
/// Conformer encoder alone carries hundreds of Linear + LayerNorm +
/// Conv1D + parallel-Memory-block parameters, so an empty manifest
/// always signals a mis-produced GGUF).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future
/// `SenseVoiceSmallWeights::bind_encoder_weights` /
/// `bind_task_head_weights` tensor walk can find its inputs without
/// re-parsing the GGUF.
#[derive(Debug)]
pub struct SenseVoiceSmallWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up SAN-M
    /// enhanced Conformer encoder + four-per-task-head forward
    /// wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl SenseVoiceSmallWeights {
    /// Scans `gguf` for the SenseVoiceSmall state_dict tensors.
    /// Refuses to bind if the GGUF carries zero tensors
    /// (FR-EX-08 — an empty GGUF is never a valid
    /// SenseVoiceSmall checkpoint).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero
    ///   tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "sensevoicesmall: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate SenseVoiceSmall checkpoint \
                 carries hundreds of SAN-M enhanced Conformer Linear + LayerNorm + \
                 Conv1D + parallel-Memory-block parameters plus four per-task heads \
                 (arch={ARCH}, name={NAME}); zero tensors always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model \
                 sensevoicesmall` against an upstream `{UPSTREAM_HF}` \
                 safetensors checkpoint (pre-flatten the upstream `model.pt` torch \
                 pickle via `tools/parity/nemo_pt_to_safetensors.py \
                 --allow-strip-any` first — the same generic pickle-to-safetensors \
                 bridge Canary / Parakeet / ReazonSpeech-NeMo-v2 reuse; pickles \
                 never enter the runtime, FR-LD-05)."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up SAN-M enhanced Conformer encoder +
    /// four-per-task-head forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// SenseVoiceSmall — the runtime binder handle.
// ---------------------------------------------------------------------------

/// SenseVoiceSmall (`FunAudioLLM/SenseVoiceSmall`, FunASR
/// MODEL_LICENSE) runtime binder for multi-task speech
/// understanding (multilingual ASR + LID + SER + AED).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`transcribe`](Self::transcribe) on a mono f32 PCM waveform
/// (16 kHz per the FunASR input convention) to obtain a
/// multi-task output tuple. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error
/// contract on the deferred SAN-M enhanced Conformer encoder +
/// four-per-task-head composition.
#[derive(Debug)]
pub struct SenseVoiceSmall {
    // The bound weights are held (real, counted) but the SAN-M
    // enhanced Conformer encoder + four-per-task-head composition
    // is a follow-up wave; the field is deliberately
    // `#[allow(dead_code)]` until the composition lands so a
    // reader is not misled by an unused field. Same posture as
    // emotion2vec / panns / audioldm2 / musicgen / redimnet /
    // storm / sortformer / pyannote / RMVPE / mt3 / beat_this.
    #[allow(dead_code)]
    weights: SenseVoiceSmallWeights,
    weight_license: LicenseClass,
}

impl SenseVoiceSmall {
    /// Binds a SenseVoiceSmall GGUF: validates arch, discovers
    /// tensors, and surfaces the stamped weight-license class for
    /// the compliance-gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing /
    /// wrong key so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent
    /// partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is
    ///   absent or not `"sensevoicesmall"` (a sibling ASR-family
    ///   GGUF handed here by mistake — `whisper` /
    ///   `distil_whisper` / `kotoba_whisper` / `moonshine` /
    ///   `parakeet` / `parakeet_ctc` / `canary` / `canary_qwen` /
    ///   `omniasr_ctc` / `kyutai_stt` / `reazonspeech_nemo_v2`
    ///   — fails with a clear message instead of a downstream
    ///   missing-tensor error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero
    ///   tensors ([`SenseVoiceSmallWeights::from_gguf`] refuses
    ///   to bind an all-zero forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "sensevoicesmall: GGUF arch is `{other}`, expected `{ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model \
                     sensevoicesmall`? Note that sibling ASR-family arch tags — \
                     `whisper` / `distil_whisper` / `kotoba_whisper` (single-task \
                     mel-front-end encoder-decoder), `moonshine` (single-task \
                     raw-audio Conv1D encoder-decoder, no mel), `parakeet` / \
                     `parakeet_ctc` (single-task NeMo FastConformer + CTC/TDT), \
                     `canary` / `canary_qwen` (single-task NeMo FastConformer + \
                     transformer/Qwen decoder), `omniasr_ctc` (single-task \
                     CTC-only), `kyutai_stt` (single-task streaming), \
                     `reazonspeech_nemo_v2` (single-task Japanese NeMo Conformer) \
                     — all live in the ASR-family neighbourhood but every single \
                     one is single-task. SenseVoiceSmall's four per-task heads \
                     (ASR + LID + SER + AED) have no analog in any sibling — \
                     silently aliasing arch would misroute the runtime dispatch \
                     to a single-task loader missing the LID / SER / AED heads \
                     (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "sensevoicesmall: GGUF is missing `vokra.model.arch` — \
                     this is not a Vokra-native sensevoicesmall GGUF (was it \
                     produced by `vokra-cli convert --model sensevoicesmall`?)"
                        .to_owned(),
                ));
            }
        }

        // 2. Load the tensor manifest with the non-emptiness gate.
        let weights = SenseVoiceSmallWeights::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license
        //    class for the compliance-gate cross-checks. The
        //    SenseVoiceSmall converter emits `FunASR_MODEL_LICENSE`
        //    which is NOT one of the SPDX ids
        //    [`LicenseClass::from_class_str`] classifies, so the
        //    fail-closed default is `Unknown` — the correct default
        //    per feedback-license-signoff-primary-source memory
        //    until owner sign-off adds the id to the matcher or
        //    an explicit CLI override provides a canonical SPDX id
        //    after review.
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
    /// `vokra.provenance.weight_license` chunk. The SenseVoiceSmall
    /// converter's default (`FunASR_MODEL_LICENSE`) is NOT one of
    /// the SPDX ids [`LicenseClass::from_class_str`] classifies —
    /// the classifier returns [`LicenseClass::Unknown`] on this
    /// string (fail-closed default per
    /// `[[feedback-license-signoff-primary-source]]`).
    /// A GGUF missing the stamp reads back as `Unknown`
    /// (fail-closed at the M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up SAN-M enhanced Conformer encoder +
    /// four-per-task-head forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The four per-task head names in a fixed order (see
    /// [`PER_TASK_HEADS`] for the ordering rationale). Returned
    /// as a `&'static` slice so consumers can `enumerate` over
    /// the tuple axis once the follow-up wave lands.
    #[inline]
    #[must_use]
    pub const fn per_task_heads() -> &'static [&'static str; 4] {
        &PER_TASK_HEADS
    }

    /// The declared language coverage of the SenseVoice family
    /// ([`N_LANGUAGES`] = 50 per An et al. 2024 §IV / TableI).
    /// Load-bearing const — a rename or drift must be caught by
    /// the test suite.
    #[inline]
    #[must_use]
    pub const fn num_languages() -> u32 {
        N_LANGUAGES
    }

    /// Multi-task transcription (ASR + LID + SER + AED) of a PCM
    /// waveform.
    ///
    /// Returns a `Vec<u32>` today for shape-parity with the
    /// sibling ASR runtime binders — the follow-up wave will
    /// widen this to a proper `MultiTaskOutput` struct carrying
    /// ASR token ids + LID label + SER label + AED label, but
    /// the wider return type would prematurely lock in a shape
    /// that a real-checkpoint tensor-name walk might refine.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the full
    /// SenseVoiceSmall forward requires the deferred SAN-M
    /// enhanced Conformer encoder + four-per-task-head
    /// composition, which cannot be synthesized from the
    /// current binder scaffold without a real tensor-name walk
    /// against the upstream `FunAudioLLM/SenseVoiceSmall`
    /// checkpoint.
    ///
    /// The error names all three primary source URLs (HF
    /// release + FunAudioLLM reference code + An et al. 2024
    /// paper) so a reader diagnosing this gap has exactly three
    /// places to walk. All four per-task head names are echoed
    /// verbatim (in load-bearing order — see
    /// [`PER_TASK_HEADS`]) so the reader can cross-check the
    /// output-shape the follow-up wave targets. **No fabricated
    /// ASR ids or task labels are ever emitted** (FR-EX-08 —
    /// no silent partial output).
    ///
    /// The `_pcm` argument is treated as the raw waveform at
    /// 16 kHz mono f32 in `[-1, 1]` (FunASR input convention);
    /// shape / rate mismatch will be a loud error rather than a
    /// resample surprise when the real forward lands.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate
    ///   for the deferred SAN-M enhanced Conformer encoder +
    ///   four-per-task-head composition.
    pub fn transcribe(&self, _pcm: &[f32]) -> Result<Vec<u32>> {
        // Bind explicitly so an unused-variable warning cannot
        // mask a future accidental removal of the parameter
        // (mirror of the panns / wavlm_sv / emotion2vec
        // loud-partial signature discipline).
        let _ = _pcm;
        Err(transcribe_forward_loud_partial())
    }
}

/// Construct the loud-partial [`VokraError::UnsupportedOp`]
/// returned by [`SenseVoiceSmall::transcribe`] until the SAN-M
/// enhanced Conformer encoder + four-per-task-head composition
/// lands.
///
/// Names **three** primary source URLs (HF release +
/// FunAudioLLM reference code + An et al. 2024 paper) so a
/// reader diagnosing the gap has exactly three places to walk.
/// All four per-task head names are echoed verbatim (in
/// load-bearing order) so the reader can cross-check the
/// output-shape the follow-up wave targets. Mirror of the
/// emotion2vec / panns / audioldm2 / musicgen / conv_tasnet /
/// redimnet / storm / sortformer / RMVPE / pyannote / wavlm /
/// moonshine loud-partial-message precedent (CLAUDE.md
/// 教訓 (a)).
fn transcribe_forward_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "sensevoicesmall transcribe (loud-partial): the full forward is deferred; \
         two missing pieces must land before real multi-task output can be \
         emitted: (1) SAN-M enhanced Conformer encoder walk — SAN-M = Modified \
         Multi-Head Attention with a parallel Memory-block Fully-Connected \
         branch per An et al. 2024 section III-A + FunASR \
         `funasr/models/sanm/`; distinct from vanilla Conformer used by \
         Parakeet / Kotoba-Whisper / Reazonspeech-NeMo-v2 in that the memory \
         block is a parallel branch, NOT a sequential module. A follow-up \
         wave landing a vanilla Conformer here would silently misroute \
         against every sibling ASR encoder (FR-EX-08); \
         (2) four per-task heads on top of the shared encoder embedding, in \
         load-bearing tuple order: [{h0}, {h1}, {h2}, {h3}] where \
         {h0} = multilingual ASR token output ({n_lang}-language coverage per \
         An et al. 2024 section IV / TableI), \
         {h1} = language identification classifier, \
         {h2} = speech emotion recognition classifier, \
         {h3} = audio event detection classifier. \
         Paper reports approximately {speedup}x lower latency than \
         Whisper-Large (An et al. 2024 section V / TableIII) — this claim \
         cannot be verified without the actual forward wired, so it is a \
         follow-up-wave target rather than a runtime-asserted number. \
         Primary sources: HF release {hf}, FunAudioLLM reference code {code}, \
         paper {paper}. Runtime cannot fabricate ASR ids or task labels \
         (FR-EX-08 no silent partial output).",
        h0 = PER_TASK_HEADS[0],
        h1 = PER_TASK_HEADS[1],
        h2 = PER_TASK_HEADS[2],
        h3 = PER_TASK_HEADS[3],
        n_lang = N_LANGUAGES,
        speedup = APPROX_SPEEDUP_VS_WHISPER,
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
    //! Tests for the SenseVoiceSmall runtime binder — contract-constant
    //! pins + metadata round-trip + negative-space round-trip on the
    //! loud-partial gates + arch-tag distinctness pin + per-task-head
    //! ordering pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On a real 16 kHz PCM
    //! waveform this would be `transcribe(...)` returning a
    //! multi-task output tuple, but the SAN-M enhanced Conformer
    //! encoder + four-per-task-head composition is deferred (see
    //! the module doc + [`SenseVoiceSmall::transcribe`] rustdoc).
    //! Fabricating a real classification output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
    //! honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin**: `ARCH` / `NAME` / `CATEGORY` /
    //!    `N_LANGUAGES` / `PER_TASK_HEADS` / `UPSTREAM_HF` all
    //!    match the converter's values exactly (cross-crate
    //!    consistency — a converter drift without a binder-side
    //!    follow-through would land here in the same commit or
    //!    fail the test).
    //! 2. **Per-task-head ordering pin**: the 4 head names are
    //!    pinned verbatim in exact upstream tuple order so a
    //!    silent reorder cannot misroute a downstream consumer's
    //!    tuple index interpretation.
    //! 3. **Metadata round-trip**: `from_gguf` reads arch + name +
    //!    category + license stamp + tensor manifest with the
    //!    correct surface semantics (Unknown fallback fires when
    //!    the stamp is absent).
    //! 4. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / empty tensor list /
    //!    unsupported forward surface) fires at its documented
    //!    surface point, in the documented error variant.
    //! 5. **Arch-tag distinctness pin**: the arch string is stable
    //!    and distinct from every sibling ASR-family arch tag.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Helper: builds a legitimate SenseVoiceSmall GGUF (arch +
    /// name + category + optional weight-license stamp + one
    /// representative SAN-M-style tensor). The tensor uses a
    /// placeholder upstream name (`encoder.san_m.linear.weight`)
    /// so the non-emptiness gate is satisfied.
    fn sensevoicesmall_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative SAN-M-style tensor so the non-emptiness
        // gate passes. Uses a placeholder name matching what a real
        // upstream SenseVoiceSmall checkpoint would carry.
        b.add_tensor(
            "encoder.san_m.linear.weight",
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
        assert_eq!(ARCH, "sensevoicesmall", "sensevoicesmall arch tag pin");
        assert_eq!(
            NAME, "sensevoicesmall",
            "sensevoicesmall canonical name pin"
        );
        assert_eq!(
            CATEGORY, "asr",
            "sensevoicesmall collapses to the shared `asr` category (multi-task \
             axis rides in name + per-task heads, not category)"
        );
        assert_eq!(
            N_LANGUAGES, 50,
            "SenseVoice paper §IV language coverage pin"
        );
        assert_eq!(
            APPROX_SPEEDUP_VS_WHISPER, 15,
            "SenseVoice paper §V speedup-vs-Whisper-Large pin"
        );
        assert_eq!(
            UPSTREAM_HF, "FunAudioLLM/SenseVoiceSmall",
            "upstream HF slug pin (used in loud-partial diagnostics)"
        );
        // The public accessor must mirror the constant.
        assert_eq!(SenseVoiceSmall::num_languages(), N_LANGUAGES);
    }

    // -----------------------------------------------------------------------
    // Test 2 — Per-task head ordering pin (a silent reorder would misroute
    //          the multi-task output tuple index — FR-EX-08 no silent
    //          class permutation)
    // -----------------------------------------------------------------------

    #[test]
    fn per_task_head_ordering_pin() {
        // Pin every head in exact upstream tuple order (per the paper
        // §III-B multi-task output ordering). A reorder / rename /
        // count-drift would land here in the same commit or fail this
        // test.
        assert_eq!(PER_TASK_HEADS, ["ASR", "LID", "SER", "AED"]);
        assert_eq!(
            PER_TASK_HEADS.len(),
            4,
            "SenseVoiceSmall has exactly 4 per-task heads"
        );
        // The public accessor must return exactly the same slice.
        assert_eq!(SenseVoiceSmall::per_task_heads(), &PER_TASK_HEADS);
        // Distinct-arch guarantee: the head list is not empty (a
        // 0-head SenseVoice would be a single-task model, defeating
        // the whole point of the arch distinctness discipline).
        assert!(!PER_TASK_HEADS.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf metadata round-trip (Unknown fallback for the
    //          fail-closed `FunASR_MODEL_LICENSE` default — the classifier
    //          does not recognize the custom licence SPDX)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip_unknown_license_fail_closed() {
        // Build a legitimate GGUF with arch + name + category but NO
        // license stamp — the binder must bind and surface the
        // Unknown fallback (fail-closed default). This is the
        // *expected* behaviour for SenseVoiceSmall since the upstream
        // licence is FunASR MODEL_LICENSE (custom, not in the SPDX
        // matcher).
        let file = sensevoicesmall_gguf(None);
        let sv = SenseVoiceSmall::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            sv.weight_license(),
            LicenseClass::Unknown,
            "missing license stamp must fail-closed to Unknown (correct default \
             for SenseVoiceSmall's FunASR MODEL_LICENSE per \
             [[feedback-license-signoff-primary-source]])"
        );
        assert!(
            sv.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4 — from_gguf rejects wrong arch (never silently mis-routes
    //          across the ASR-family neighbourhood — moonshine hint fleet)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `whisper` GGUF handed to the sensevoicesmall binder by
        // mistake must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08). Whisper's single ASR head
        // and SenseVoiceSmall's four per-task heads (ASR + LID + SER
        // + AED) are completely different downstream compositions
        // on top of related encoders, so silent aliasing would
        // misroute the runtime dispatch to a single-task loader
        // missing the LID / SER / AED heads.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        b.add_string(chunks::KEY_MODEL_NAME, "whisper-base");
        b.add_tensor("whisper.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SenseVoiceSmall::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`whisper`") && m.contains("`sensevoicesmall`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The message must enumerate a load-bearing subset of
                // the ASR-family sibling fleet so the reader has
                // fully specified anchors. Every one of these must
                // appear in the message.
                for sibling in [
                    "whisper",
                    "distil_whisper",
                    "kotoba_whisper",
                    "moonshine",
                    "parakeet",
                    "canary",
                    "reazonspeech_nemo_v2",
                ] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                // The message must call out the head-composition
                // divergence (four per-task heads vs single-task
                // siblings).
                assert!(
                    m.contains("four per-task heads")
                        || (m.contains("LID") && m.contains("SER") && m.contains("AED")),
                    "message should call out the four per-task heads divergence, \
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
        // with a "not a Vokra-native sensevoicesmall GGUF" diagnostic
        // so a caller who hands a raw GGUF (without the converter's
        // arch stamp) has a clear signal.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SenseVoiceSmall::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not a Vokra-native sensevoicesmall GGUF"),
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
        // Correct arch + name but zero tensors — the
        // SenseVoiceSmallWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SenseVoiceSmall::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model sensevoicesmall"),
                    "message must include the repro command, got `{m}`"
                );
                assert!(
                    m.contains("nemo_pt_to_safetensors.py"),
                    "message must cite the pickle-to-safetensors bridge (no pickle \
                     in runtime — FR-LD-05), got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — transcribe loud-partial (returns UnsupportedOp naming
    //          SAN-M encoder + four per-task heads + all 3 primary
    //          sources + all 4 head names + FR-EX-08 rationale)
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_loud_partial_returns_unsupported_op() {
        let file = sensevoicesmall_gguf(None);
        let sv = SenseVoiceSmall::from_gguf(&file).expect("valid arch must bind");

        // Legitimate PCM shape: 1 s of silence at 16 kHz mono (the
        // FunASR input convention).
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = sv.transcribe(&pcm) else {
            panic!("transcribe must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // Names the surface + posture.
                assert!(
                    msg.contains("sensevoicesmall transcribe"),
                    "surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Names the two missing pieces by exact identifier.
                assert!(
                    msg.contains("SAN-M"),
                    "message must name the SAN-M enhanced Conformer encoder \
                     gap, got `{msg}`"
                );
                assert!(
                    msg.contains("four per-task heads"),
                    "message must name the four per-task heads gap, got `{msg}`"
                );

                // Distinguishing-topology anchor: the parallel Memory-
                // block Fully-Connected branch is what separates SAN-M
                // from vanilla Conformer. A follow-up wave landing a
                // vanilla Conformer here would silently misroute.
                assert!(
                    msg.contains("parallel"),
                    "message must call out the parallel Memory-block branch \
                     (SAN-M vs vanilla Conformer distinguishing trait), got \
                     `{msg}`"
                );

                // Cites all three primary source URLs so a reader
                // diagnosing the gap has anchors to walk.
                for url in [PRIMARY_SOURCE_HF, PRIMARY_SOURCE_CODE, PRIMARY_SOURCE_PAPER] {
                    assert!(
                        msg.contains(url),
                        "expected primary source URL '{url}' cited: {msg}"
                    );
                }

                // All 4 per-task head names echoed verbatim (a silent
                // reorder would misroute the multi-task tuple index —
                // FR-EX-08).
                for head in PER_TASK_HEADS.iter() {
                    assert!(
                        msg.contains(head),
                        "expected per-task head '{head}' in error: {msg}"
                    );
                }

                // Language coverage + speedup claim echoed as
                // follow-up-wave targets (not runtime-asserted).
                assert!(
                    msg.contains("50"),
                    "message must echo the 50-language coverage, got `{msg}`"
                );
                assert!(
                    msg.contains("15"),
                    "message must echo the ~15x speedup-vs-Whisper-Large claim, \
                     got `{msg}`"
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

    // -----------------------------------------------------------------------
    // Test 8 — Arch tag distinctness pin (a rename to any sibling ASR arch
    //          would misroute runtime dispatch)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_asr_family_siblings() {
        // Every sibling ASR arch tag must be distinct from
        // sensevoicesmall's ARCH — this is the wire-format handshake
        // that prevents silent dispatch mis-routes.
        for sibling in [
            "whisper",
            "distil_whisper",
            "kotoba_whisper",
            "moonshine",
            "parakeet",
            "parakeet_ctc",
            "canary",
            "canary_qwen",
            "omniasr_ctc",
            "kyutai_stt",
            "reazonspeech_nemo_v2",
        ] {
            assert_ne!(
                ARCH, sibling,
                "sensevoicesmall ARCH must be distinct from sibling '{sibling}' \
                 — silent aliasing would misroute runtime dispatch (FR-EX-08)"
            );
        }
    }
}
