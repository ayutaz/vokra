//! # vokra-kws-micro
//!
//! microWakeWord-style keyword-spotting (KWS) forward core as a `#![no_std]`
//! (+ `alloc`) subset, sister crate of [`vokra-vad-micro`] and following the
//! same M5-03 案1 topology: the numeric forward is lifted out of the std-heavy
//! `vokra-models` so it cross-compiles for bare-metal **Cortex-M55**
//! (`thumbv8m-none`, IoT Tier 3 / NFR-PT-03). The std wrapper in
//! `vokra-models` will (in a follow-up WP) depend on this crate and re-export
//! its public surface, keeping one forward shared bit-identically between the
//! std and no_std builds.
//!
//! ## SCAFFOLD ONLY (honest UNIMPLEMENTED)
//!
//! This is a **skeleton**. [`KwsMicro::detect`] returns [`KwsEvent::Idle`] for
//! every input. It is **not** a fake wake detector; it is an intentionally
//! inert stub so that downstream crates can wire the type surface (registering
//! keywords, feeding audio frames, matching on the event) before the real
//! model lands. Real inference — a TFLite-Micro forward pass on the
//! microWakeWord model (Apache 2.0,
//! <https://github.com/kahrendt/microWakeWord>) — is a follow-up WP.
//!
//! ## Design red lines (inherited from `vokra-vad-micro`)
//!
//! - **Zero external deps (NFR-DS-02)**: only `vokra-core`. When the real
//!   forward lands, transcendentals will come from a shared scalar module
//!   (mirroring `vokra_vad_micro::scalar`) — **no `libm`** (deny.toml bans it).
//! - **No `unsafe` (NFR-RL-07)**: workspace lint `unsafe_code = "deny"`.
//! - **1:1 preservation (FR-LD-06 / FR-OP-50 / NFR-QL-05)**: microWakeWord
//!   will be a dedicated subgraph, not lowered to generic audio-dialect ops
//!   (same rule that keeps Silero VAD as its own subgraph).
//!
//! [`vokra-vad-micro`]: https://docs.rs/vokra-vad-micro

// Sister-crate pattern (matches `vokra-vad-micro`): become `#![no_std]` when
// the `std` feature is off (thumbv8m cross-build). Under the default feature
// set this attribute is inert, and the crate compiles as a normal std library
// so `cargo test` sees the standard test harness (NFR-PT-01 cross-build
// non-interference).
#![cfg_attr(not(feature = "std"), no_std)]

// The forward is alloc-dependent (owned `Vec` of keyword definitions,
// `String` names). `extern crate alloc` links it in both modes; under `std`
// it is already present, so this is harmless there.
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// A registered keyword the detector should listen for.
///
/// Held as an owned `String` (via [`alloc`]) so the caller does not need a
/// `'static` lifetime — real deployments load keyword names from GGUF metadata
/// or a runtime configuration blob, neither of which yields `&'static str`.
#[derive(Debug, Clone)]
pub struct KeywordDef {
    /// Stable numeric id used by [`KwsEvent::Wake`] to identify the keyword
    /// without allocating on the emit path. Assigned by the caller (usually
    /// the ordinal index into the loaded model's keyword table).
    pub id: u8,
    /// Human-readable keyword name (e.g. `"hey_vokra"`), for logs and demos.
    /// Not used by the numeric forward.
    pub name: String,
    /// Confidence threshold in `[0.0, 1.0]`. The real forward will only emit
    /// [`KwsEvent::Wake`] when the model's softmax score for this keyword is
    /// `>= threshold`. Currently unused (scaffold).
    pub threshold: f32,
}

impl KeywordDef {
    /// Constructs a [`KeywordDef`]. `name` accepts any `Into<String>` so both
    /// `&str` and owned `String` work.
    pub fn new(id: u8, name: impl Into<String>, threshold: f32) -> Self {
        Self {
            id,
            name: name.into(),
            threshold,
        }
    }
}

/// One frame's KWS decision.
///
/// The forward returns exactly one event per frame; consumers pattern-match
/// on it in their audio loop. Idle vs. Wake is a per-frame decision — burst
/// suppression / debouncing is the caller's responsibility (not part of this
/// crate).
#[derive(Debug, Clone, PartialEq)]
pub enum KwsEvent {
    /// No registered keyword scored above its threshold on this frame.
    Idle,
    /// A registered keyword was detected on this frame. `keyword_id` matches
    /// the [`KeywordDef::id`] of the winning keyword; `score` is the softmax
    /// probability the model assigned.
    Wake {
        /// The [`KeywordDef::id`] of the detected keyword.
        keyword_id: u8,
        /// Model softmax probability in `[0.0, 1.0]`.
        score: f32,
    },
}

/// microWakeWord-style KWS detector (SCAFFOLD).
///
/// See the crate-level docs for the honest UNIMPLEMENTED contract:
/// [`Self::detect`] currently returns [`KwsEvent::Idle`] for every input.
#[derive(Debug, Default)]
pub struct KwsMicro {
    /// Registered keywords, in the order [`Self::add_keyword`] received them.
    /// Public so tests and callers can introspect without going through a
    /// `keywords()` getter; the field is owned (`Vec<KeywordDef>`), so it
    /// cannot be mutated except through the `&mut self` methods on this type.
    pub keywords: Vec<KeywordDef>,
}

impl KwsMicro {
    /// Constructs an empty detector with no keywords registered.
    pub fn new() -> Self {
        Self {
            keywords: Vec::new(),
        }
    }

    /// Registers a keyword. Keywords are stored in insertion order; the real
    /// forward will iterate over them to find the highest-scoring match.
    pub fn add_keyword(&mut self, def: KeywordDef) {
        self.keywords.push(def);
    }

    /// Runs the detector on one audio frame and returns the resulting event.
    ///
    /// SKELETON: real inference (TFLite-Micro forward pass on the
    /// microWakeWord model) is a follow-up WP; the current impl unconditionally
    /// returns [`KwsEvent::Idle`] for all inputs. `frame` is expected to be
    /// 16-bit signed PCM at the model's sample rate (typically 16 kHz mono)
    /// once real inference lands; the current impl ignores the frame contents
    /// entirely.
    pub fn detect(&mut self, frame: &[i16]) -> KwsEvent {
        // Honest UNIMPLEMENTED skeleton (SCAFFOLD, NOT a fake wake detector).
        // Returning `Idle` for every frame is the only truthful behaviour a
        // skeleton can offer — a probabilistic guess would be dishonest, and a
        // `panic!` would break downstream integrators wiring the type surface.
        // Real inference (TFLite-Micro forward pass on the microWakeWord model,
        // Apache 2.0, https://github.com/kahrendt/microWakeWord) is a
        // follow-up WP; when it lands, `frame` will be consumed instead of
        // being silently discarded.
        let _ = frame;
        KwsEvent::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kws_new_creates_empty_detector() {
        let d = KwsMicro::new();
        assert_eq!(d.keywords.len(), 0);
    }

    #[test]
    fn kws_detect_zero_frame_returns_idle() {
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(0, "hey_vokra", 0.5));
        let ev = d.detect(&[0i16; 512]);
        assert_eq!(ev, KwsEvent::Idle);
    }

    #[test]
    fn kws_add_keyword_persists() {
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(0, "one", 0.5));
        d.add_keyword(KeywordDef::new(1, "two", 0.6));
        assert_eq!(d.keywords.len(), 2);
    }
}
