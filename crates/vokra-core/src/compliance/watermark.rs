//! Watermark configuration — **config surface only** (M2-13-T10/T12).
//!
//! # 2026-07-04 client drop: embedding is deferred, not faked
//!
//! FR-CP-01 (AudioSeal) / FR-CP-02 (C2PA) watermark **embedding** (implementation
//! WP M1-07) was dropped by the client. This module keeps the *design intent* —
//! the default-ON flags of `docs/legal-compliance.md` §8 — as a settable config,
//! but there is **no embedding backend wired**. [`WatermarkConfig::backend_status`]
//! returns [`WatermarkBackendStatus::Deferred`] so no downstream can honestly
//! claim a watermark was embedded. This is deliberate: silently pretending to
//! watermark would be worse for compliance than an explicit "not implemented".
//!
//! Consequently EU AI Act Article 50 (NFR-LG-01) and California SB 942
//! (NFR-LG-02) marking obligations are **currently unmet**; the re-entry point
//! when embedding returns is [`WatermarkConfig::backend_status`] flipping to a
//! future `Active`. The legal sufficiency of any disclosure text is the client's
//! call (FR-MD-13 / X-03), not decided here.

/// Whether a real watermark-embedding backend is available.
///
/// Non-exhaustive so a future `Active` (or per-scheme) status can be added
/// without breaking downstream matches when embedding is re-implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WatermarkBackendStatus {
    /// No embedding backend is wired (2026-07-04 client drop). The config flags
    /// are honoured as *settings*, but output audio is not watermarked.
    Deferred,
}

/// Watermark scheme toggles (`docs/legal-compliance.md` §8).
///
/// **The defaults preserve the FR-CP-01/02 design intent** (AudioSeal + C2PA
/// on), but see the module docs: embedding is deferred, so these are settings
/// awaiting a backend, not a promise that audio is marked. Toggling
/// [`audioseal`](Self::audioseal) off is the opt-out path (FR-CP-01 "opt-out
/// permitted, but a warning is shown"); the warning hook lives on the loader /
/// synthesis path, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WatermarkConfig {
    /// Meta AudioSeal (MIT) — the recommended default (design intent: `true`).
    pub audioseal: bool,
    /// C2PA content-provenance manifest (Apache-2.0) — design intent `true`.
    pub c2pa: bool,
    /// Google SynthID — requires a Google agreement, so default `false`.
    pub synthid: bool,
    /// SilentCipher OSS alternative (doc: v1.0+) — design intent `true`.
    pub silent_cipher: bool,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        // Design-intent defaults transcribed from docs/legal-compliance.md §8.
        // (Embedding is deferred — see the module docs; these are settings.)
        Self {
            audioseal: true,
            c2pa: true,
            synthid: false,
            silent_cipher: true,
        }
    }
}

impl WatermarkConfig {
    /// The status of the embedding backend. Always
    /// [`WatermarkBackendStatus::Deferred`] in M2-13 (2026-07-04 client drop):
    /// callers must consult this before claiming any output is watermarked.
    pub fn backend_status(&self) -> WatermarkBackendStatus {
        WatermarkBackendStatus::Deferred
    }

    /// Whether any watermark scheme is requested on. Used by the loader /
    /// synthesis path to decide whether to emit the "backend deferred" notice
    /// (T12) exactly once, rather than silently doing nothing.
    pub fn any_enabled(&self) -> bool {
        self.audioseal || self.c2pa || self.synthid || self.silent_cipher
    }

    /// Whether the AudioSeal default was opted out of (FR-CP-01). The synthesis
    /// path uses this to fire the opt-out warning hook (implemented where audio
    /// is produced, not here).
    pub fn audioseal_opted_out(&self) -> bool {
        !self.audioseal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_hold_design_intent() {
        let w = WatermarkConfig::default();
        assert!(w.audioseal, "FR-CP-01 default ON");
        assert!(w.c2pa, "FR-CP-02 default ON");
        assert!(!w.synthid, "SynthID needs a Google agreement");
        assert!(w.silent_cipher);
        assert!(w.any_enabled());
        assert!(!w.audioseal_opted_out());
    }

    #[test]
    fn backend_is_always_deferred_in_m2() {
        // The honesty guarantee (T12): no config can report an active backend.
        let w = WatermarkConfig::default();
        assert_eq!(w.backend_status(), WatermarkBackendStatus::Deferred);
        let opted_out = WatermarkConfig {
            audioseal: false,
            ..Default::default()
        };
        assert_eq!(opted_out.backend_status(), WatermarkBackendStatus::Deferred);
        assert!(opted_out.audioseal_opted_out());
    }
}
