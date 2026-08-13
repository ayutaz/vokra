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

#[cfg(not(feature = "std"))]
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use vokra_core::{Result, VokraError};

// M5-03b Phase 1: the log-mel front-end (features) plus the self-contained
// `log10` it needs. Both modules are `#![no_std]`-clean (only `core` +
// `alloc` + crate-local `scalar`). Consumed by [`KwsMicro::detect`] via
// [`AttachedChain::extractor`] when a real forward chain is attached
// (Phase 3 — see [`interpreter`]).
// See [ADR M5-03b](../../docs/adr/M5-03b-kws-micro-no-std.md).
pub mod features;
// M5-03b Phase 3: the INT8 forward-chain interpreter that walks a fixed
// [`interpreter::ChainConfig`] (Conv2d / DwConv2d / FC / Sigmoid / Softmax)
// over the pre-quantised weights loaded via [`model`]. Consumed by
// [`KwsMicro::detect`] via [`AttachedChain::chain`].
pub mod interpreter;
// M5-03b Phase 2: the scalar INT8 op kernels (`conv2d_int8`,
// `depthwise_conv2d_int8`, `fully_connected_int8`, `sigmoid_int8`,
// `softmax_int8`) the microWakeWord MC-MobileNet forward drives. Sibling to
// `vokra_vad_micro::math` — a `#![no_std]`-clean numeric core with no
// `unsafe`, no SIMD intrinsics, and no `libm`. The Cortex-M55 Helium (MVE)
// fast path is deferred per M5-03 ADR. Consumed by [`interpreter`] via
// each [`interpreter::LayerSpec`] variant.
pub mod kernels;
// M5-03b Phase 2: the runtime *loader* for microWakeWord GGUFs produced by
// `tools/parity/microwakeword/prepare_checkpoint.py`. Reads the
// `vokra.kws.*` metadata contract + every dense F32 tensor via
// `vokra_core::gguf::GgufFile` (no_std-clean under `default-features =
// false`). Callers construct an [`interpreter::ChainConfig`] from a
// [`model::Model`] (per-layer weight binding is model-specific; see the
// [`model`] module docs for the honest-boundary contract on the
// quantisation-params-on-emit follow-up).
pub mod model;
mod scalar;

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

/// Everything [`KwsMicro`] needs to run a real detection pass: an INT8
/// forward chain, a log-mel feature extractor, and the FFI quantisation
/// params that bridge them. Attached via [`KwsMicro::set_chain`]; when
/// unattached [`KwsMicro::detect`] returns [`KwsEvent::Idle`] (honest
/// SCAFFOLD default).
///
/// Private because the fields have interdependent invariants (the extractor's
/// feature vector length must equal the chain's `input_size`) — callers go
/// through [`KwsMicro::set_chain`], which enforces them.
struct AttachedChain {
    /// The INT8 forward chain. `chain.input_size()` must equal
    /// [`features::N_MELS`] (validated in [`KwsMicro::set_chain`]).
    chain: interpreter::ChainConfig,
    /// Log-mel feature extractor (~44 KB of precomputed tables); constructed
    /// eagerly on `set_chain` so the hot path never pays the setup cost.
    extractor: features::FeatureExtractor,
    /// Feature-vector quantisation scale (`features_f32` → `features_i8`).
    /// Must match the chain's first-layer `input_scale` (up to caller
    /// discipline — no runtime check because the chain kernels themselves
    /// don't expose it as a struct field for cross-layer validation).
    feature_scale: f32,
    /// Feature-vector quantisation zero-point.
    feature_zero_point: i8,
    /// Final softmax/sigmoid output scale (used to dequantise output
    /// probabilities into the `[0, 1]` domain the threshold lives in).
    /// Must match the final layer's `output_scale`.
    output_scale: f32,
    /// Final layer output zero-point (must match the final layer's
    /// `output_zero_point`).
    output_zero_point: i8,
}

/// microWakeWord-style KWS detector.
///
/// # Two modes
///
/// * **SCAFFOLD (default)** — no chain attached. [`Self::detect`] returns
///   [`KwsEvent::Idle`] for every frame. This is the honest UNIMPLEMENTED
///   default before a real hey_jarvis chain is loaded (see the crate-level
///   docs for the honest-boundary contract).
/// * **REAL (after [`Self::set_chain`])** — a validated
///   [`interpreter::ChainConfig`] plus a [`features::FeatureExtractor`] are
///   attached; [`Self::detect`] runs the full log-mel → INT8 forward → argmax
///   → threshold pipeline and returns [`KwsEvent::Wake`] when any registered
///   keyword's dequantised probability crosses its
///   [`KeywordDef::threshold`].
#[derive(Debug, Default)]
pub struct KwsMicro {
    /// Registered keywords, in the order [`Self::add_keyword`] received them.
    /// Public so tests and callers can introspect without going through a
    /// `keywords()` getter; the field is owned (`Vec<KeywordDef>`), so it
    /// cannot be mutated except through the `&mut self` methods on this type.
    pub keywords: Vec<KeywordDef>,
    /// Optional inference state. See [`AttachedChain`] and [`Self::detect`]'s
    /// mode contract.
    chain: Option<AttachedChain>,
}

// `AttachedChain` needs a manual `Debug` impl (the fields include primitives
// + `Debug`-derived types, so a straight-forward derive would work, but
// `KwsMicro` derives `Debug` and `AttachedChain` must implement it for that
// to compile). The derive on `AttachedChain` itself has no runtime cost.
impl core::fmt::Debug for AttachedChain {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AttachedChain")
            .field("chain_layers", &self.chain.layer_count())
            .field("chain_input_size", &self.chain.input_size())
            .field("chain_output_size", &self.chain.output_size())
            .field("feature_scale", &self.feature_scale)
            .field("feature_zero_point", &self.feature_zero_point)
            .field("output_scale", &self.output_scale)
            .field("output_zero_point", &self.output_zero_point)
            .finish()
    }
}

impl KwsMicro {
    /// Constructs an empty detector with no keywords and no chain attached
    /// (SCAFFOLD mode — [`Self::detect`] returns [`KwsEvent::Idle`]).
    pub fn new() -> Self {
        Self {
            keywords: Vec::new(),
            chain: None,
        }
    }

    /// Registers a keyword. Keywords are stored in insertion order; when a
    /// chain is attached [`Self::detect`] iterates them and picks the
    /// highest-scoring one above its [`KeywordDef::threshold`].
    ///
    /// [`KeywordDef::id`] indexes into the chain's output vector — if a
    /// keyword's `id` is `>= chain.output_size()` it is silently skipped
    /// during detection (documented on [`KeywordDef::id`]; a caller-side
    /// mistake, not a runtime error).
    pub fn add_keyword(&mut self, def: KeywordDef) {
        self.keywords.push(def);
    }

    /// Attaches a real inference chain, promoting the detector out of
    /// SCAFFOLD mode.
    ///
    /// `feature_scale` / `feature_zero_point` control the front-end
    /// quantisation ([`features_f32`](features::FeatureExtractor::compute_frame_f32)
    /// → INT8 features); they must match the first layer's `input_scale` /
    /// `input_zero_point` in the caller's chain construction.
    ///
    /// `output_scale` / `output_zero_point` mirror the *final* layer's output
    /// quantisation params (typically the standard TFLite softmax convention
    /// `1/256, -128`); they are used to dequantise chain outputs back into
    /// the `[0, 1]` probability space so [`KeywordDef::threshold`] can be
    /// compared directly.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `chain.input_size()` does
    /// not equal [`features::N_MELS`] — the extractor produces exactly
    /// [`features::N_MELS`] features per frame, and a size mismatch would
    /// crash the chain on first call. Fail-closed (FR-EX-08).
    pub fn set_chain(
        &mut self,
        chain: interpreter::ChainConfig,
        feature_scale: f32,
        feature_zero_point: i8,
        output_scale: f32,
        output_zero_point: i8,
    ) -> Result<()> {
        if chain.input_size() != features::N_MELS {
            return Err(VokraError::InvalidArgument(format!(
                "set_chain: chain input size {} != features::N_MELS {}",
                chain.input_size(),
                features::N_MELS,
            )));
        }
        self.chain = Some(AttachedChain {
            chain,
            extractor: features::FeatureExtractor::new(),
            feature_scale,
            feature_zero_point,
            output_scale,
            output_zero_point,
        });
        Ok(())
    }

    /// Reports whether a real chain is attached (i.e. we are in REAL mode
    /// rather than SCAFFOLD mode). Useful for tests and integration diagnostics.
    pub fn has_chain(&self) -> bool {
        self.chain.is_some()
    }

    /// Runs the detector on one audio frame and returns the resulting event.
    ///
    /// # SCAFFOLD mode (no chain attached)
    ///
    /// Returns `Ok(KwsEvent::Idle)` unconditionally, ignoring `frame`
    /// contents. This is the honest UNIMPLEMENTED default before a real
    /// chain is loaded (via [`Self::set_chain`]). A probabilistic guess
    /// would be dishonest; a `panic!` would break integrators wiring the
    /// type surface.
    ///
    /// # REAL mode (chain attached)
    ///
    /// Runs the full pipeline:
    ///
    /// 1. Length-check `frame` against [`features::WINDOW_SAMPLES`]
    ///    (fail-closed [`VokraError::InvalidArgument`] on mismatch —
    ///    FR-EX-08).
    /// 2. Log-mel feature extraction via [`features::FeatureExtractor::compute_frame_f32`]
    ///    → [`features::N_MELS`] f32 features.
    /// 3. Quantise features to INT8 with the attached `feature_scale` /
    ///    `feature_zero_point`.
    /// 4. Run the [`interpreter::ChainConfig`] forward (INT8 activations,
    ///    ping-pong scratch, no heap alloc on hot per-layer path).
    /// 5. For each registered keyword: if `keyword.id < chain.output_size()`,
    ///    dequantise `output[keyword.id]` with the attached `output_scale`
    ///    / `output_zero_point`, clamp to `[0, 1]`, and check against
    ///    [`KeywordDef::threshold`]. The highest-scoring keyword above its
    ///    threshold wins; ties resolve to insertion order.
    /// 6. Emit [`KwsEvent::Wake`] with the winner (or [`KwsEvent::Idle`]
    ///    if no keyword crossed its threshold).
    ///
    /// # Honest boundary
    ///
    /// Real hey_jarvis accuracy verification requires (a) a real MC-MobileNet
    /// chain constructed from a GGUF the sidecar produced, and (b) a canned
    /// "hey jarvis" audio fixture. Both are owner-triggered — the SCAFFOLD
    /// default (no chain) is the safe posture until they land. See the
    /// crate-level docs for the full honest-boundary contract.
    pub fn detect(&mut self, frame: &[i16]) -> Result<KwsEvent> {
        // SCAFFOLD mode: no chain attached → return Idle without touching
        // `frame`. This is the honest UNIMPLEMENTED behaviour before a real
        // hey_jarvis chain is loaded.
        let Some(state) = self.chain.as_mut() else {
            let _ = frame;
            return Ok(KwsEvent::Idle);
        };

        // REAL mode: length-check first (fail-closed — silent zero-pad or
        // truncate would silently misclassify).
        if frame.len() != features::WINDOW_SAMPLES {
            return Err(VokraError::InvalidArgument(format!(
                "detect: frame len {} != expected {} (WINDOW_SAMPLES @ {} Hz)",
                frame.len(),
                features::WINDOW_SAMPLES,
                features::SAMPLE_RATE,
            )));
        }

        // Cache the AttachedChain's scalar params before the chain borrow so
        // the subsequent output-borrow doesn't conflict with reading them.
        let feature_scale = state.feature_scale;
        let feature_zp = state.feature_zero_point;
        let output_scale = state.output_scale;
        let output_zp = state.output_zero_point;

        // (2) Log-mel feature extraction (F32).
        let features_f32 = state.extractor.compute_frame_f32(frame);
        // (3) Quantise features to INT8.
        let features_i8 = features::quantize_int8(&features_f32, feature_scale, feature_zp as i32);
        // (4) Run the INT8 forward chain. `output_i8` is a borrow into the
        // chain's internal ping-pong buffer — valid until the next `run` call.
        let output_i8 = state.chain.run(&features_i8)?;
        let output_len = output_i8.len();

        // (5) Score every registered keyword against the chain's output.
        // Skip out-of-range keywords silently (documented on `KeywordDef::id`):
        // this is caller-side misconfiguration, not a runtime error.
        // `self.keywords` and `self.chain` are disjoint fields of the outer
        // `self`, so borrowing keywords immutably while holding a mutable
        // borrow on chain (via `state`) is allowed by the disjoint-field
        // borrow-check rule.
        let mut best: Option<(u8, f32)> = None;
        for kw in &self.keywords {
            let id = kw.id as usize;
            if id >= output_len {
                continue;
            }
            let q = (output_i8[id] as i32) - (output_zp as i32);
            // Clamp to [0, 1] — dequantised softmax output should be in this
            // range by construction, but numerical drift from the requantise
            // path can push it slightly outside; `KeywordDef::threshold` lives
            // in [0, 1], so we clamp before comparing.
            let prob = ((q as f32) * output_scale).clamp(0.0, 1.0);
            if prob >= kw.threshold {
                match best {
                    None => best = Some((kw.id, prob)),
                    Some((_, best_score)) if prob > best_score => best = Some((kw.id, prob)),
                    _ => {}
                }
            }
        }

        // (6) Emit the winning event.
        match best {
            Some((id, score)) => Ok(KwsEvent::Wake {
                keyword_id: id,
                score,
            }),
            None => Ok(KwsEvent::Idle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use interpreter::{ChainConfig, LayerSpec};

    #[test]
    fn kws_new_creates_empty_detector_without_chain() {
        let d = KwsMicro::new();
        assert_eq!(d.keywords.len(), 0);
        assert!(!d.has_chain(), "fresh detector must start in SCAFFOLD mode");
    }

    #[test]
    fn kws_detect_scaffold_mode_returns_idle_for_any_frame() {
        // No chain attached → SCAFFOLD mode: every frame is Idle, regardless
        // of content or length (SCAFFOLD ignores `frame` entirely, so this
        // holds even for frames the REAL path would reject).
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(0, "hey_vokra", 0.5));
        for frame_len in [0usize, 128, 512, 1024] {
            let frame = alloc::vec![0i16; frame_len];
            let ev = d.detect(&frame).unwrap();
            assert_eq!(
                ev,
                KwsEvent::Idle,
                "SCAFFOLD mode must return Idle for frame_len={frame_len}"
            );
        }
    }

    #[test]
    fn kws_add_keyword_persists() {
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(0, "one", 0.5));
        d.add_keyword(KeywordDef::new(1, "two", 0.6));
        assert_eq!(d.keywords.len(), 2);
    }

    // ---- REAL-mode helpers ---------------------------------------------

    /// Builds a trivial "always fires class 0" chain:
    /// - `FullyConnected(N_MELS → 1)` with `weight = 0` and a strong positive
    ///   bias → post-requantise output pins at `+127`.
    /// - `Sigmoid(1)` on that → dequantised probability ≈ 0.996.
    ///
    /// Feature quantisation params are the same the sigmoid consumes
    /// (`(1.0, 0)` → `(1/256, -128)`) so `detect`'s output dequantisation
    /// matches the sigmoid's stated output params.
    fn always_wake_chain() -> ChainConfig {
        let fc = LayerSpec::FullyConnected {
            weight_i8: alloc::vec![0i8; features::N_MELS],
            bias_i32: alloc::vec![1000i32],
            in_dim: features::N_MELS,
            out_dim: 1,
            input_zero_point: 0,
            output_zero_point: 0,
            // acc = 0 + 1000; requantise 1000 · 0.1 = 100 → clamps well
            // below 127. Actually 100 is fine, no clamp: sigmoid input = 100.
            output_scale: 0.1,
        };
        let sigmoid = LayerSpec::Sigmoid {
            size: 1,
            input_scale: 1.0,
            input_zero_point: 0,
            output_scale: 1.0 / 256.0,
            output_zero_point: -128,
        };
        ChainConfig::new(alloc::vec![fc, sigmoid]).unwrap()
    }

    #[test]
    fn set_chain_rejects_wrong_input_size() {
        // Chain input size 3 ≠ N_MELS (40) — must fail-closed.
        let bad_chain = ChainConfig::new(alloc::vec![LayerSpec::FullyConnected {
            weight_i8: alloc::vec![0i8; 3 * 2],
            bias_i32: alloc::vec![0i32; 2],
            in_dim: 3,
            out_dim: 2,
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        }])
        .unwrap();
        let mut d = KwsMicro::new();
        let err = d
            .set_chain(bad_chain, 1.0, 0, 1.0 / 256.0, -128)
            .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(
            !d.has_chain(),
            "failed set_chain must not leave a partial attach"
        );
    }

    #[test]
    fn detect_real_mode_rejects_wrong_frame_length() {
        let mut d = KwsMicro::new();
        d.set_chain(always_wake_chain(), 1.0, 0, 1.0 / 256.0, -128)
            .unwrap();
        assert!(
            d.has_chain(),
            "chain must be attached before REAL-mode test"
        );
        // Wrong length: WINDOW_SAMPLES is 512, feed 100.
        let err = d.detect(&[0i16; 100]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn detect_real_mode_fires_wake_when_threshold_crossed() {
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(0, "always_fires", 0.5));
        d.set_chain(always_wake_chain(), 1.0, 0, 1.0 / 256.0, -128)
            .unwrap();
        // Silence frame — the always-wake chain ignores features (weight = 0)
        // and emits ~1.0 probability regardless.
        let ev = d.detect(&[0i16; features::WINDOW_SAMPLES]).unwrap();
        match ev {
            KwsEvent::Wake { keyword_id, score } => {
                assert_eq!(keyword_id, 0);
                assert!(
                    score >= 0.5,
                    "score {score} must be at or above threshold 0.5"
                );
                assert!(
                    score <= 1.0,
                    "score {score} must be clamped to the [0,1] probability space"
                );
            }
            other => panic!("expected Wake, got {other:?}"),
        }
    }

    #[test]
    fn detect_real_mode_stays_idle_when_no_keyword_crosses_threshold() {
        // Same always-fires chain, but the sole keyword's threshold is set
        // above the chain's max output (0.996) so no Wake should fire.
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(0, "always_high_threshold", 0.9999));
        d.set_chain(always_wake_chain(), 1.0, 0, 1.0 / 256.0, -128)
            .unwrap();
        let ev = d.detect(&[0i16; features::WINDOW_SAMPLES]).unwrap();
        assert_eq!(ev, KwsEvent::Idle);
    }

    #[test]
    fn detect_real_mode_skips_out_of_range_keyword_ids() {
        // Keyword with id=5 but chain output size = 1 → silently skipped
        // (documented on `KeywordDef::id`; caller-side mistake, not a runtime
        // error). With no in-range keyword, event is Idle.
        let mut d = KwsMicro::new();
        d.add_keyword(KeywordDef::new(5, "out_of_range", 0.5));
        d.set_chain(always_wake_chain(), 1.0, 0, 1.0 / 256.0, -128)
            .unwrap();
        let ev = d.detect(&[0i16; features::WINDOW_SAMPLES]).unwrap();
        assert_eq!(ev, KwsEvent::Idle);
    }
}
