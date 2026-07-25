//! Robust Model for Vocal Pitch Estimation (RMVPE).
//!
//! # Primary source
//!
//! - Upstream reference: <https://github.com/Dream-High/RMVPE>
//! - License: **apache-2.0** (permissive — no runtime-side attribution
//!   obligation, unlike the CC-BY-4.0 codec / ASR weights).
//!
//! RMVPE is a small CNN-based polyphonic vocal pitch estimator that also
//! predicts a voiced / unvoiced (V/UV) flag per frame. It is the pitch
//! front-end **required by RVC v2** and is commonly reused by other
//! singing-voice / voice-conversion (GPT-SoVITS, etc.) pipelines.
//!
//! # Scope of this skeleton (SoTA plan Phase 1 op scaffold, 2026-07-25)
//!
//! This is an **honest UNIMPLEMENTED skeleton**. It fixes:
//!
//! 1. The load / extract API shape ([`RMVPE::from_gguf`] /
//!    [`RMVPE::extract`]).
//! 2. The per-hop frame-count contract: `extract(&pcm, sr).len() == pcm.len() / hop`.
//! 3. The GGUF metadata namespace: `vokra.f0.rmvpe.hop` (u32, default 160),
//!    `vokra.f0.rmvpe.fmin` (f32, default 50.0),
//!    `vokra.f0.rmvpe.fmax` (f32, default 1100.0).
//!
//! It does **not** run any inference — the real CNN forward, mel front-end,
//! Viterbi post-decoding and V/UV threshold are a follow-up WP (T29-equivalent
//! in SoTA plan Phase 1). Skeleton output is `hz = 0.0`, `voiced = false`,
//! `confidence = 0.0` for every frame — bit-exactly zero so a downstream
//! silent-vs-implemented divergence is impossible to miss.
//!
//! # Design note — `LoadError`
//!
//! The task spec calls the load-failure return `LoadError`. Vokra's public API
//! (FR-API-02) exposes exactly **one** error type — [`vokra_core::VokraError`]
//! — with a dedicated [`ModelLoad`](vokra_core::VokraError::ModelLoad) variant
//! and an [`Io`](vokra_core::VokraError::Io) variant fed by
//! `From<std::io::Error>` (a missing file becomes `Io(NotFound)`). We map
//! `LoadError → VokraError::{Io, ModelLoad}` (the load-failure classes) rather
//! than adding a per-op error type — introducing a bespoke `LoadError` here
//! would fork the error surface every existing consumer already switches on.

use std::path::Path;

use vokra_core::Result;
use vokra_core::gguf::GgufFile;

use super::F0Frame;

/// GGUF metadata key: analysis hop in samples (u32).
pub const GGUF_KEY_HOP: &str = "vokra.f0.rmvpe.hop";
/// GGUF metadata key: minimum tracked F0 in Hz (f32).
pub const GGUF_KEY_FMIN: &str = "vokra.f0.rmvpe.fmin";
/// GGUF metadata key: maximum tracked F0 in Hz (f32).
pub const GGUF_KEY_FMAX: &str = "vokra.f0.rmvpe.fmax";

/// Default analysis hop in samples (matches the upstream RMVPE default at
/// 16 kHz PCM in — 10 ms).
pub const DEFAULT_HOP: u32 = 160;
/// Default lower pitch bound in Hz (below typical adult male F0 floor).
pub const DEFAULT_FMIN: f32 = 50.0;
/// Default upper pitch bound in Hz (above typical soprano F0 ceiling).
pub const DEFAULT_FMAX: f32 = 1100.0;

/// Robust Model for Vocal Pitch Estimation (RMVPE) — the pitch front-end
/// required by RVC v2 (<https://github.com/Dream-High/RMVPE>, apache-2.0).
///
/// SKELETON: this only carries the GGUF-declared hop / fmin / fmax and
/// enforces the per-hop frame-count contract; the real CNN forward is a
/// follow-up WP.
#[derive(Debug)]
pub struct RMVPE {
    hop: u32,
    #[allow(dead_code)] // wired by the follow-up WP that lands the real CNN forward.
    fmin: f32,
    #[allow(dead_code)] // wired by the follow-up WP that lands the real CNN forward.
    fmax: f32,
}

impl RMVPE {
    /// Loads an RMVPE model from a GGUF at `path`.
    ///
    /// The GGUF may declare any subset of the metadata keys — absent keys use
    /// [`DEFAULT_HOP`] / [`DEFAULT_FMIN`] / [`DEFAULT_FMAX`]. Returns a
    /// [`VokraError::Io`](vokra_core::VokraError::Io) if the path is missing
    /// / unreadable and a
    /// [`VokraError::ModelLoad`](vokra_core::VokraError::ModelLoad) if the
    /// file exists but is not a valid GGUF (see the design note in the module
    /// docs for the `LoadError → VokraError::{Io, ModelLoad}` mapping).
    ///
    /// SKELETON: real weight tensors are not consumed yet — this only reads
    /// the header + metadata to lock the load-failure surface.
    pub fn from_gguf(path: &Path) -> Result<Self> {
        // `GgufFile::open` maps a missing file to `GgufError::Io` (→
        // `VokraError::Io` via the boundary `From` impl) and a malformed file
        // to `GgufError::{BadMagic, ...}` (→ `VokraError::ModelLoad`).
        let gguf = GgufFile::open(path)?;

        // `as_u64` widens U8/U16/U32/U64; the converter is expected to emit
        // U32 for `hop` but the widening keeps us tolerant of larger widths.
        let hop = gguf
            .get(GGUF_KEY_HOP)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(DEFAULT_HOP);
        // `as_f64` accepts F32 and F64; narrowing back to f32 is lossy for
        // F64-declared values but the pitch bounds are always well inside
        // f32 range so the narrow is loud-free.
        let fmin = gguf
            .get(GGUF_KEY_FMIN)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(DEFAULT_FMIN);
        let fmax = gguf
            .get(GGUF_KEY_FMAX)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(DEFAULT_FMAX);

        Ok(Self { hop, fmin, fmax })
    }

    /// Extracts a per-hop F0 track from `pcm`.
    ///
    /// SKELETON: real rmvpe CNN inference is a follow-up WP; this only
    /// guarantees the frame-count contract.
    pub fn extract(&self, pcm: &[f32], sample_rate: u32) -> Vec<F0Frame> {
        let hop = self.hop as usize;
        // Guard against a zero hop from a malformed GGUF (defaults are
        // non-zero — this only fires if the follow-up WP or a hand-forged
        // GGUF writes 0). Producing an empty track is loud-free: downstream
        // consumers see zero frames rather than a divide-by-zero panic.
        if hop == 0 {
            return Vec::new();
        }
        let n = pcm.len() / hop;
        let sr = sample_rate.max(1) as f32; // avoid /0 on caller mistakes
        (0..n)
            .map(|i| F0Frame {
                time_sec: (i * hop) as f32 / sr,
                hz: 0.0,
                voiced: false,
                confidence: 0.0,
            })
            .collect()
    }

    /// Test-only constructor: builds an RMVPE with the default hop / fmin /
    /// fmax without touching a GGUF, so the frame-count-contract test can run
    /// on the skeleton before real GGUF loading lands.
    #[cfg(test)]
    fn with_defaults() -> Self {
        Self {
            hop: DEFAULT_HOP,
            fmin: DEFAULT_FMIN,
            fmax: DEFAULT_FMAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A GGUF path that cannot possibly exist on any developer or CI host.
    fn nonexistent_gguf_path() -> PathBuf {
        // Absolute path under a leading dot-directory unique to this test — the
        // parent does not exist, so both fs::metadata and fs::read map to
        // ErrorKind::NotFound → VokraError::Io.
        PathBuf::from("/nonexistent/.vokra-rmvpe-red-fixture/does-not-exist.gguf")
    }

    /// STEP 1 (RED): `from_gguf` on a missing path must surface a load
    /// failure. Under the skeleton this test fails because `unimplemented!()`
    /// panics; STEP 2 (GREEN) satisfies it by returning
    /// `VokraError::{Io, ModelLoad}` from `GgufFile::open`.
    #[test]
    fn rmvpe_load_stub_reports_load_error() {
        use vokra_core::VokraError;

        let path = nonexistent_gguf_path();
        let err = RMVPE::from_gguf(&path).expect_err("missing GGUF must not load");
        assert!(
            matches!(err, VokraError::Io(_) | VokraError::ModelLoad(_)),
            "expected VokraError::Io or ModelLoad for a missing path, got {err:?}"
        );
    }

    /// STEP 1 (RED): the frame-count contract is
    /// `extract(&pcm, sr).len() == pcm.len() / hop` with `hop = 160`. Under
    /// the skeleton this test fails because `unimplemented!()` panics; STEP 2
    /// (GREEN) satisfies it by returning a `Vec<F0Frame>` of the right shape.
    #[test]
    fn rmvpe_extract_frame_count_matches_hop() {
        let m = RMVPE::with_defaults();
        // A non-trivial PCM length that is not a multiple of the hop so the
        // integer-truncation semantics of `pcm.len() / hop` are exercised
        // (16 000 samples = 1 s at 16 kHz + 33 extra samples).
        let pcm = vec![0.0f32; 16_033];
        let hop = DEFAULT_HOP as usize;
        let frames = m.extract(&pcm, 16_000);
        assert_eq!(frames.len(), pcm.len() / hop);
    }
}
