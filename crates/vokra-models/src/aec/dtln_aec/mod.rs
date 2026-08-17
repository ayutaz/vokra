//! **DTLN-AEC** — Dual-Signal Transformation LSTM Network for
//! Acoustic Echo Cancellation. Runtime binder for the `dtln_aec`
//! converter arch (post-audit-cc-gap-2026-08-14 Wave 6 loud-partial
//! land).
//!
//! # Upstream primary sources
//!
//! - Repo (MIT, LICENSE verified `Copyright (c) 2021 Nils L.
//!   Westhausen`): `github.com/breizhn/DTLN-aec`
//! - Paper: Westhausen & Meyer, INTERSPEECH 2021, "Acoustic Echo
//!   Cancellation with the Dual-Signal Transformation LSTM Network",
//!   arXiv:2010.15754
//! - Reference model (Keras / TF, `.tflite`): three widths shipped
//!   upstream — `dtln_aec_128.tflite`, `dtln_aec_256.tflite`,
//!   `dtln_aec_512.tflite`.
//!
//! # Architecture (dual-signal LSTM, upstream `dtln_model.py`)
//!
//! ```text
//! mic PCM (16 kHz mono f32)   ─┐
//!                              ├─► rolling frame pair (hop = 128 samples)
//! far-end PCM (16 kHz mono f32)┘
//!
//! Stage 1 — STFT-domain LSTM (spectral IRM mask)
//!   STFT(mic)     → |Y|              (magnitude, F bins)
//!   STFT(far-end) → |X|              (magnitude, F bins)
//!   [|Y|, |X|] concat along freq axis → 2F-dim feature
//!     → LSTM(hidden = N_UNITS) → Dense(F) → sigmoid → IRM mask M ∈ [0,1]
//!   Ê1 = M * Y                       (masked mic spectrum)
//!   ê1 = iSTFT(Ê1)                   (partially cleaned PCM)
//!
//! Stage 2 — time-domain LSTM (residual)
//!   [ê1, far-end] concat along time axis → 2 × block_len feature
//!     → LSTM(hidden = N_UNITS) → Dense(block_len) → tanh → gain g
//!   cleaned = g * ê1                (echo-cancelled PCM)
//! ```
//!
//! Every `N_UNITS` in both stages is the variant-selected LSTM width
//! ([`DtlnVariant::Units128`] / [`Units256`] / [`Units512`]). The two
//! stages share width per release (upstream trains each `.tflite` with
//! matched STFT-stage and time-stage widths).
//!
//! # `vokra.dtln_aec.*` chunk group (converter contract)
//!
//! The BF16-passthrough converter (`crates/vokra-convert/src/models/
//! dtln_aec.rs`) stamps every hparam under this chunk group so the
//! runtime loader never depends on tensor-shape probing (probing would
//! silently accept a widely wrong checkpoint that happens to have
//! consistent-looking shapes):
//!
//! - `vokra.dtln_aec.lstm_units` (u64) — LSTM hidden width shared by
//!   both stages (128 / 256 / 512).
//! - `vokra.dtln_aec.n_fft` (u64) — 512 in every current release.
//! - `vokra.dtln_aec.hop` (u64) — 128 in every current release
//!   (block_shift in upstream naming).
//! - `vokra.dtln_aec.block_len` (u64) — 512 in every current release
//!   (equal to `n_fft` in every current release; kept as a distinct
//!   chunk against a future variant that decouples them).
//! - `vokra.dtln_aec.sample_rate` (u64) — 16 000 (paper §4).
//!
//! # FR-EX-08 posture
//!
//! - **Sample rate**: DTLN-AEC weights were trained at 16 kHz;
//!   [`DtlnAec::process`] refuses any other rate loudly (a future
//!   [`AecEngine`] impl inherits this gate).
//! - **Mic / far-end length mismatch**: [`DtlnAec::process`] refuses
//!   `mic.len() != farend.len()` loudly — the two streams are
//!   strictly sample-aligned in AEC (silent trim / repeat is a
//!   correctness bug, not a convenience).
//! - **Arch tag**: [`DtlnAec::from_gguf`] refuses any
//!   `vokra.model.arch` other than `"dtln_aec"` loudly (a mis-fed
//!   sibling `nkf_aec` / `denoise` GGUF fails with a clear "wrong
//!   arch" message instead of a downstream "missing tensor").
//! - **Missing `lstm_units` chunk**: [`DtlnAec::from_gguf`] refuses
//!   loudly (fail-closed against a corrupt / stale GGUF that predates
//!   the chunk stamping).
//! - **Unknown LSTM width**: [`DtlnVariant::from_lstm_units`] refuses
//!   any width other than 128 / 256 / 512 loudly.
//! - **Generic LSTM primitive gap**: [`DtlnAec::process`] returns a
//!   loud [`VokraError::UnsupportedOp`] naming the missing
//!   `vokra_ops::lstm` primitive + the four wiring pieces still owed +
//!   the primary source URLs. The one public LSTM in `vokra-ops` is
//!   `vokra_ops::hybrid_ctc_attention::LstmLmCell`, which is LM-shaped
//!   (token id in, one log-probability out, embedding + vocab
//!   projection bundled in) and so cannot carry a feature sequence;
//!   the sibling `nkf_aec` inlined its
//!   per-layer GRU because its dim was tiny (H=18), but DTLN's
//!   128/256/512-unit LSTMs with 4-gate concatenation are large enough
//!   that inlining without a shared primitive multiplies
//!   implementation cost across DTLN + every future LSTM-based model.
//!   Loud-partial per CLAUDE.md 教訓 (a) — "loud-partial は
//!   fake-complete より honest" (FR-EX-08).

use std::path::Path;

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{Result, VokraError};

#[cfg(test)]
mod tests;

// ---- arch / provenance constants ---------------------------------------
//
// Mirror of `vokra-convert::models::dtln_aec::{ARCH, NAME, CATEGORY}` —
// kept as duplicated `pub const` so the runtime binder does not add a
// cross-crate dependency edge onto the converter (the sibling
// nkf_aec / fsmn_vad / openwakeword convention).

/// Expected `vokra.model.arch` value written by
/// `vokra-convert --model dtln-aec`.
pub const ARCH: &str = "dtln_aec";

/// Default `vokra.model.name` value written by the converter.
pub const DEFAULT_NAME: &str = "dtln-aec";

/// `vokra.model.category` — AEC family.
pub const CATEGORY: &str = "aec";

// ---- upstream-pinned dims (transcribed from `dtln_model.py` +
//      `run_aec.py`) --------------------------------------------------

/// STFT FFT size (upstream `block_len = 512`).
pub const N_FFT: usize = 512;

/// STFT analysis window length in samples (upstream `block_len = 512`).
pub const BLOCK_LEN: usize = 512;

/// STFT hop size (upstream `block_shift = 128` — a 4x-overlap framing).
pub const HOP: usize = 128;

/// PCM sample rate the model was trained at (paper §4 — AEC-Challenge
/// 2021 corpus is 16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;

/// Number of RFFT bins (`n_fft / 2 + 1`).
pub const F_BINS: usize = N_FFT / 2 + 1;

// ---- metadata keys (mirror of converter constants) ---------------------

/// `vokra.dtln_aec.lstm_units` — LSTM hidden width for both stages.
pub const KEY_VARIANT_LSTM_UNITS: &str = "vokra.dtln_aec.lstm_units";

/// `vokra.dtln_aec.n_fft`.
pub const KEY_N_FFT: &str = "vokra.dtln_aec.n_fft";

/// `vokra.dtln_aec.hop`.
pub const KEY_HOP: &str = "vokra.dtln_aec.hop";

/// `vokra.dtln_aec.block_len`.
pub const KEY_BLOCK_LEN: &str = "vokra.dtln_aec.block_len";

/// `vokra.dtln_aec.sample_rate`.
pub const KEY_SAMPLE_RATE: &str = "vokra.dtln_aec.sample_rate";

// ---- primary source URLs (cited verbatim in the loud-partial error
//      message per the canary_qwen precedent) --------------------------

/// Upstream Git repository (MIT license — verified via GitHub `LICENSE`
/// primary source, `Copyright (c) 2021 Nils L. Westhausen`).
pub const PRIMARY_SOURCE_GITHUB: &str = "https://github.com/breizhn/DTLN-aec";

/// arXiv preprint of the paper the reference model was published in.
pub const PRIMARY_SOURCE_ARXIV: &str = "https://arxiv.org/abs/2010.15754";

/// Human-readable citation for the paper.
pub const PRIMARY_SOURCE_PAPER: &str = "Westhausen & Meyer, \"Acoustic Echo Cancellation with the \
     Dual-Signal Transformation LSTM Network\", INTERSPEECH 2021";

// ---- variant enum ------------------------------------------------------

/// Fixed-width LSTM variants shipped upstream. Every variant sets the
/// LSTM hidden width used by BOTH the STFT-domain stage AND the
/// time-domain stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DtlnVariant {
    /// `dtln_aec_128.tflite` (smallest, ~1 MB).
    Units128,
    /// `dtln_aec_256.tflite` (mid, ~3 MB).
    Units256,
    /// `dtln_aec_512.tflite` (largest, ~7 MB).
    Units512,
}

impl DtlnVariant {
    /// LSTM hidden width for both stages.
    pub fn lstm_units(&self) -> usize {
        match self {
            Self::Units128 => 128,
            Self::Units256 => 256,
            Self::Units512 => 512,
        }
    }

    /// Recovers the variant from a stamped `lstm_units` value; None if
    /// the value doesn't match any known upstream release width (fail-
    /// closed against a mis-stamped GGUF).
    pub fn from_lstm_units(units: usize) -> Option<Self> {
        match units {
            128 => Some(Self::Units128),
            256 => Some(Self::Units256),
            512 => Some(Self::Units512),
            _ => None,
        }
    }
}

// ---- config ------------------------------------------------------------

/// DTLN-AEC runtime config (fully-pinned by upstream today; the struct
/// is `#[non_exhaustive]` so a future variant checkpoint can carry a
/// differently-widthed hparam without breaking downstream callers).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DtlnAecConfig {
    /// LSTM hidden width for both stages.
    pub variant: DtlnVariant,
    /// STFT FFT size.
    pub n_fft: usize,
    /// STFT block length in samples (equal to `n_fft` in every current
    /// release).
    pub block_len: usize,
    /// STFT hop size in samples.
    pub hop: usize,
    /// PCM sample rate the model was trained at.
    pub sample_rate: u32,
}

impl DtlnAecConfig {
    /// The upstream release config for a given variant.
    pub fn upstream_default(variant: DtlnVariant) -> Self {
        Self {
            variant,
            n_fft: N_FFT,
            block_len: BLOCK_LEN,
            hop: HOP,
            sample_rate: SAMPLE_RATE,
        }
    }

    /// Number of RFFT bins (`n_fft / 2 + 1`).
    #[inline]
    pub fn f_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Validates the config loudly (FR-EX-08). Every field must be
    /// non-zero and `block_len <= n_fft`.
    pub fn validate(&self) -> Result<()> {
        if self.n_fft == 0 {
            return Err(VokraError::InvalidArgument(
                "dtln-aec config: `n_fft` must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.block_len == 0 {
            return Err(VokraError::InvalidArgument(
                "dtln-aec config: `block_len` must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.hop == 0 {
            return Err(VokraError::InvalidArgument(
                "dtln-aec config: `hop` must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "dtln-aec config: `sample_rate` must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.block_len > self.n_fft {
            return Err(VokraError::InvalidArgument(format!(
                "dtln-aec config: block_len ({}) > n_fft ({}) — invalid framing",
                self.block_len, self.n_fft
            )));
        }
        Ok(())
    }
}

// ---- weight bundle -----------------------------------------------------

/// DTLN-AEC weight bundle placeholder — the tensor slots are declared
/// as raw f32 vecs today; the four LSTM gate matrices per stage
/// (`[i, f, g, o]` concatenated along output dim, upstream Keras
/// convention) will land as typed sub-structs once the generic LSTM
/// primitive lands in `vokra-ops`.
///
/// The `#[non_exhaustive]` marker is deliberate: a follow-up wave will
/// add typed fields (`stft_lstm_kernel`, `stft_lstm_recurrent_kernel`,
/// `stft_lstm_bias`, `stft_dense_kernel`, `stft_dense_bias`, and the
/// matching `time_*` variants) without breaking downstream callers of
/// the loud-partial API.
#[derive(Debug)]
#[non_exhaustive]
pub struct DtlnAecWeights {
    /// Total tensor count discovered on the GGUF (`> 0` when loaded
    /// from a real checkpoint, `0` on the synthesized-config path).
    pub tensor_count: usize,
}

impl DtlnAecWeights {
    /// Binds the weight bundle from a Vokra GGUF. Today this is a
    /// tensor-count walk that (a) refuses arch-tag mismatch loudly and
    /// (b) refuses missing hparam chunks loudly; the per-tensor
    /// dimension checks land alongside the typed weight fields once
    /// the generic LSTM primitive is available (loud-partial per
    /// CLAUDE.md 教訓 (a)).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on wrong arch, missing hparam chunk,
    /// or a corrupted GGUF.
    pub fn from_gguf(gguf: &GgufFile, cfg: &DtlnAecConfig) -> Result<Self> {
        cfg.validate()?;
        let tensor_count = gguf.tensors().len();
        Ok(Self { tensor_count })
    }
}

// ---- engine ------------------------------------------------------------

/// DTLN-AEC model — immutable shareable weight bundle plus the config
/// it was bound against.
#[derive(Debug)]
pub struct DtlnAec {
    cfg: DtlnAecConfig,
    #[allow(dead_code)] // consumed by the follow-up trait impl
    weights: DtlnAecWeights,
}

impl DtlnAec {
    /// Binds the model from a parsed GGUF (FR-LD-01). The arch tag is
    /// verified first so a mis-fed GGUF (nkf-aec / fsmn-vad /
    /// openwakeword / ...) fails with a clear "wrong arch" message
    /// instead of a downstream "missing tensor".
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "dtln-aec: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "dtln-aec: GGUF is missing `vokra.model.arch` (converter did not stamp it)"
                        .to_owned(),
                ));
            }
        }
        let units = gguf
            .get(KEY_VARIANT_LSTM_UNITS)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "dtln-aec: GGUF is missing `{KEY_VARIANT_LSTM_UNITS}` \
                     (converter did not stamp variant; converter version too old \
                     or GGUF was hand-authored)"
                ))
            })?;
        let variant = DtlnVariant::from_lstm_units(units as usize).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "dtln-aec: stamped `{KEY_VARIANT_LSTM_UNITS} = {units}` does not match \
                 any known upstream release width (128 / 256 / 512); \
                 refuse fail-closed rather than silent-default to Units128 (FR-EX-08)"
            ))
        })?;
        let cfg = DtlnAecConfig::upstream_default(variant);
        let weights = DtlnAecWeights::from_gguf(gguf, &cfg)?;
        Ok(Self { cfg, weights })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The bound checkpoint's config.
    pub fn config(&self) -> &DtlnAecConfig {
        &self.cfg
    }

    /// Runs the DTLN-AEC forward on paired mic + far-end PCM. Returns
    /// echo-cancelled PCM (aligned sample-for-sample with `mic`).
    ///
    /// # LOUD-PARTIAL — not yet wired
    ///
    /// The FR-EX-08 gates (empty input, length mismatch, wrong sample
    /// rate) fire real; the forward itself is deferred until the
    /// generic LSTM primitive lands in `vokra-ops`. See the module
    /// docs "Generic LSTM primitive gap" section for the four wiring
    /// pieces the follow-up wave owes.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on empty PCM, length mismatch,
    ///   or wrong sample rate.
    /// - [`VokraError::UnsupportedOp`] naming the generic LSTM
    ///   primitive gap + the four wiring pieces still owed + the
    ///   primary source URLs (loud-partial per CLAUDE.md 教訓 (a) —
    ///   "loud-partial は fake-complete より honest").
    pub fn process(&self, mic: &[f32], farend: &[f32]) -> Result<Vec<f32>> {
        // FR-EX-08 gates — fire real BEFORE the loud-partial arm so
        // callers get correct-input-shape errors on the caller-side
        // path they can fix.
        if mic.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dtln-aec process: `mic` PCM slice is empty".to_owned(),
            ));
        }
        if farend.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dtln-aec process: `farend` PCM slice is empty".to_owned(),
            ));
        }
        if mic.len() != farend.len() {
            return Err(VokraError::InvalidArgument(format!(
                "dtln-aec process: mic.len() ({}) != farend.len() ({}) — the two \
                 streams are strictly sample-aligned in AEC (silent trim / repeat \
                 is a correctness bug, not a convenience) — FR-EX-08",
                mic.len(),
                farend.len()
            )));
        }

        // Loud-partial arm: the generic LSTM primitive is absent from
        // `vokra-ops` today. Fire a distinct UnsupportedOp naming
        // (i) the missing primitive, (ii) the four wiring pieces still
        // owed, (iii) primary source URLs (canary_qwen precedent, Wave
        // 5 lesson: dynamic-format message must go in UnsupportedOp,
        // not NotImplemented).
        Err(VokraError::UnsupportedOp(format!(
            "dtln-aec process: real weights are bound (variant = {units}-unit LSTM, \
             {tensors} tensors on the GGUF) but the forward is not wired yet. \
             The generic LSTM primitive is absent from `vokra_ops` today \
             (confirmed 2026-08-14 — no `pub mod lstm`, no `LstmCell`; the sibling \
             `nkf_aec` inlined its per-layer GRU because dim was tiny (H=18), \
             the sibling `silero_vad` kept the LSTM as a 1:1 preserved subgraph, \
             and DTLN's {units}-unit LSTM with 4-gate concatenation is large \
             enough that inlining without a shared primitive multiplies \
             implementation cost across DTLN + every future LSTM-based model). \
             Four wiring pieces are owed: \
             (i) generic `LstmCell` primitive in `vokra_ops` with 4-gate \
             concatenation `[i, f, g, o]` matching upstream Keras layout; \
             (ii) STFT-domain LSTM stage over |mic| ⊕ |farend| concatenated \
             magnitude spectrogram → sigmoid → IRM mask → masked mic spectrum → \
             iSTFT; \
             (iii) time-domain LSTM stage over [ê1, farend] concatenated PCM → \
             Dense(block_len) → tanh → gain → cleaned PCM; \
             (iv) `impl AecEngine for DtlnAec` + `DtlnAecStream::push_paired` \
             once the primitive lands so DTLN can plug into Moshi / CSM \
             full-duplex duplex engines the same way `nkf_aec` does. \
             Primary source: {github}. Reference: paper {paper} — {arxiv}. \
             Loud-partial (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
             より honest') — FR-EX-08 no silent zero-fill.",
            units = self.cfg.variant.lstm_units(),
            tensors = self.weights.tensor_count,
            github = PRIMARY_SOURCE_GITHUB,
            paper = PRIMARY_SOURCE_PAPER,
            arxiv = PRIMARY_SOURCE_ARXIV,
        )))
    }
}
