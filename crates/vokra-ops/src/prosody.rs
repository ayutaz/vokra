//! `prosody_control` unified API (M3-17; FR-OP-74).
//!
//! [`ProsodyControl`] is the unified prosody-control message. The consuming
//! WPs (M3-09 CosyVoice2 instruction control in v0.9; v2.0+ StyleTTS style
//! token / EmotiVoice emotion label) will implement [`ApplyProsody`] on
//! their respective model handles to fold pitch/speed/pause into whatever
//! model-native representation they need (a text instruction string for
//! CosyVoice2, a style token for StyleTTS, etc.).
//!
//! This WP lands **API surface + trait only**. The v0.9 scope is CosyVoice2,
//! and M3-09 has not landed yet, so no adapter is wired up here — that is
//! deliberate (milestones.md §7.2 M3-17 depends on M3-09).
//!
//! # v0.9 scope (milestones.md §7.2)
//!
//! - Only CosyVoice2 (M3-09) will implement [`ApplyProsody`] in v0.9.
//! - StyleTTS / EmotiVoice adapters are v2.0+; front-loading them here is
//!   forbidden by the milestone plan.
//! - piper-plus native / Kokoro / Whisper carry no `instruction`-style
//!   prosody control today. Their TTS entry points must **explicitly**
//!   reject a non-identity [`ProsodyControl`] rather than silently ignore
//!   it (FR-EX-08 — no silent fallback). This trait is not wired to those
//!   models; the rejection contract is caller-side.
//!
//! # Identity default (passthrough)
//!
//! [`ProsodyControl::default`] returns an all-`None` instance:
//! [`ProsodyControl::is_identity`] returns `true`. Adapters MUST observe
//! this state as "no prosody control requested" and preserve their model's
//! default synthesis behaviour bit-for-bit.
//!
//! # Field semantics
//!
//! Concrete units / valid ranges are pinned by the adapter (M3-09 for
//! CosyVoice2). This crate documents each field's *shape*; the adapter
//! decides whether e.g. `pitch_shift` is in semitones or Hz, and what the
//! useful `speed_scale` range is. Inventing units at this layer is
//! forbidden (CLAUDE.md hallucination ban).

use vokra_core::{Result, VokraError};

/// Unified prosody-control message (FR-OP-74).
///
/// A default-constructed value carries no prosody control — every field is
/// `None`, which adapters interpret as passthrough
/// ([`ProsodyControl::is_identity`] returns `true`). Callers set only the
/// fields they want to override; the exact units and valid ranges are the
/// adapter's job (M3-09 for CosyVoice2 in v0.9).
///
/// The struct is `#[non_exhaustive]` so v2.0+ can add axes (emotion tokens,
/// style tokens, …) without breaking downstream destructuring matches.
///
/// # Example
///
/// ```
/// use vokra_ops::ProsodyControl;
///
/// // Identity = passthrough.
/// assert!(ProsodyControl::default().is_identity());
///
/// // Override one axis at a time via chainable setters.
/// let ctrl = ProsodyControl::default()
///     .with_speed_scale(1.25)
///     .with_pause_ms(300);
/// assert_eq!(ctrl.speed_scale, Some(1.25));
/// assert_eq!(ctrl.pause_ms, Some(300));
/// assert!(ctrl.pitch_shift.is_none());
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProsodyControl {
    /// Pitch shift. Units are the adapter's contract (semitones vs. Hz,
    /// absolute vs. relative); the CosyVoice2 adapter in M3-09 fixes the
    /// interpretation for v0.9.
    pub pitch_shift: Option<f32>,
    /// Speed multiplier (1.0 = adapter's neutral speed). The useful range
    /// is the adapter's contract (M3-09).
    pub speed_scale: Option<f32>,
    /// Pause duration in milliseconds inserted at an adapter-chosen point
    /// (M3-09 decides whether this is a leading pause, a mid-utterance
    /// break, or an inter-sentence gap).
    pub pause_ms: Option<u32>,
    /// Free-form instruction text. CosyVoice2 accepts natural-language
    /// instructions here (M3-09); adapters that don't consume text
    /// instructions should reject a non-`None` value explicitly (FR-EX-08).
    pub instruction: Option<String>,
}

impl ProsodyControl {
    /// Return an identity (all-`None`) control — the passthrough default.
    ///
    /// Alias for [`ProsodyControl::default`], kept so call sites read as
    /// "identity" rather than "default" where that intent matters.
    pub fn identity() -> Self {
        Self::default()
    }

    /// `true` iff every field is `None`, i.e. the control is a passthrough.
    ///
    /// A `#[non_exhaustive]` struct grows fields over time (v2.0+ emotion /
    /// style axes); each new field must be checked here so that "identity"
    /// remains "every axis untouched" (compile-time reminder: adding a
    /// field without extending this check makes the identity contract lie
    /// silently, so treat this method as the authoritative predicate).
    pub fn is_identity(&self) -> bool {
        let Self {
            pitch_shift,
            speed_scale,
            pause_ms,
            instruction,
        } = self;
        pitch_shift.is_none()
            && speed_scale.is_none()
            && pause_ms.is_none()
            && instruction.is_none()
    }

    /// Validate the *structural* invariants of the control.
    ///
    /// Rejects non-finite `pitch_shift` (NaN / ±∞) and
    /// non-finite-or-non-positive `speed_scale` (NaN / ±∞ / ≤ 0). The valid
    /// *range* (e.g. `0.5 ≤ speed_scale ≤ 2.0`) is the adapter's contract
    /// (M3-09 pins it for CosyVoice2); this method rejects only values
    /// that are structurally impossible for any adapter to interpret.
    ///
    /// Callers must run `validate()` before passing a control to an
    /// [`ApplyProsody`] adapter — the trait's `apply` signature has no
    /// return channel (see its rustdoc), so validation happens up-front.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of:
    /// - `pitch_shift` is NaN or infinite;
    /// - `speed_scale` is NaN, infinite, zero, or negative.
    ///
    /// (`pause_ms` is a `u32`, so its domain is structurally constrained.
    /// `instruction` is a free-form string, so nothing is validated here —
    /// the adapter decides what to accept.)
    pub fn validate(&self) -> Result<()> {
        if let Some(p) = self.pitch_shift {
            if !p.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "ProsodyControl.pitch_shift must be finite (got {p})"
                )));
            }
        }
        if let Some(s) = self.speed_scale {
            if !s.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "ProsodyControl.speed_scale must be finite (got {s})"
                )));
            }
            if s <= 0.0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ProsodyControl.speed_scale must be positive (got {s})"
                )));
            }
        }
        Ok(())
    }

    /// Return `self` with `pitch_shift` set to `Some(value)`. Chainable.
    #[must_use = "with_pitch_shift returns a new ProsodyControl by value"]
    pub fn with_pitch_shift(mut self, value: f32) -> Self {
        self.pitch_shift = Some(value);
        self
    }

    /// Return `self` with `speed_scale` set to `Some(value)`. Chainable.
    #[must_use = "with_speed_scale returns a new ProsodyControl by value"]
    pub fn with_speed_scale(mut self, value: f32) -> Self {
        self.speed_scale = Some(value);
        self
    }

    /// Return `self` with `pause_ms` set to `Some(value)`. Chainable.
    #[must_use = "with_pause_ms returns a new ProsodyControl by value"]
    pub fn with_pause_ms(mut self, value: u32) -> Self {
        self.pause_ms = Some(value);
        self
    }

    /// Return `self` with `instruction` set to `Some(value.into())`.
    /// Chainable.
    #[must_use = "with_instruction returns a new ProsodyControl by value"]
    pub fn with_instruction(mut self, value: impl Into<String>) -> Self {
        self.instruction = Some(value.into());
        self
    }
}

/// Trait implemented by prosody-aware TTS models to consume a
/// [`ProsodyControl`] (FR-OP-74).
///
/// `apply` operates in place on `ctx`. A typical adapter (e.g. CosyVoice2 in
/// M3-09) reads `pitch_shift` / `speed_scale` / `pause_ms` and folds them
/// into `ctx.instruction` as a model-native instruction string; a
/// style-token adapter (v2.0+ StyleTTS) would write to a future `style`
/// axis on the same struct.
///
/// # Contract
///
/// - **Identity is passthrough.** If `ctx.is_identity()` on entry, adapters
///   should leave `ctx` untouched (this is what makes an identity default a
///   real passthrough — no adapter turns "no request" into a request).
/// - **Idempotent given a fixed `self`.** Calling `apply` twice on the same
///   `ctx` with the same `self` must yield the same final `ctx`.
/// - **No silent fallback (FR-EX-08).** If `self` cannot honour a field
///   (e.g. `speed_scale` is out of the adapter's supported range),
///   implementations must not silently clamp or drop the axis. Because
///   `apply` has no `Result` return channel, callers are expected to have
///   run [`ProsodyControl::validate`] first for structural invariants; any
///   adapter-specific range checks must be surfaced up-front by the same
///   caller (e.g. via the model's TTS entry-point signature), not silently
///   inside `apply`.
///
/// # v0.9 wiring
///
/// The only concrete implementer in v0.9 is **CosyVoice2** (M3-09).
/// piper-plus native / Kokoro / Whisper deliberately do *not* implement
/// this trait; their TTS entry-points reject a non-identity
/// [`ProsodyControl`] explicitly at the caller boundary (FR-EX-08).
pub trait ApplyProsody {
    /// Fold `self`'s model-specific interpretation of the prosody axes
    /// into `ctx`. See the trait-level contract for behavioural
    /// requirements.
    fn apply(&self, ctx: &mut ProsodyControl);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Default identity -----------------------------------------------

    #[test]
    fn default_is_identity_passthrough() {
        // The passthrough contract: default() = every axis None, and
        // is_identity() must agree with that observation.
        let ctrl = ProsodyControl::default();
        assert!(ctrl.pitch_shift.is_none());
        assert!(ctrl.speed_scale.is_none());
        assert!(ctrl.pause_ms.is_none());
        assert!(ctrl.instruction.is_none());
        assert!(ctrl.is_identity());
    }

    #[test]
    fn identity_constructor_matches_default() {
        // identity() is a documentation-facing alias for default(); the
        // two must be observationally identical.
        assert_eq!(ProsodyControl::identity(), ProsodyControl::default());
    }

    // ---- Individual field override --------------------------------------

    #[test]
    fn pitch_shift_override_leaves_other_axes_none() {
        let ctrl = ProsodyControl::default().with_pitch_shift(2.0);
        assert_eq!(ctrl.pitch_shift, Some(2.0));
        assert!(ctrl.speed_scale.is_none());
        assert!(ctrl.pause_ms.is_none());
        assert!(ctrl.instruction.is_none());
        assert!(!ctrl.is_identity());
    }

    #[test]
    fn speed_scale_override_leaves_other_axes_none() {
        let ctrl = ProsodyControl::default().with_speed_scale(1.5);
        assert_eq!(ctrl.speed_scale, Some(1.5));
        assert!(ctrl.pitch_shift.is_none());
        assert!(ctrl.pause_ms.is_none());
        assert!(ctrl.instruction.is_none());
        assert!(!ctrl.is_identity());
    }

    #[test]
    fn pause_ms_override_leaves_other_axes_none() {
        let ctrl = ProsodyControl::default().with_pause_ms(250);
        assert_eq!(ctrl.pause_ms, Some(250));
        assert!(ctrl.pitch_shift.is_none());
        assert!(ctrl.speed_scale.is_none());
        assert!(ctrl.instruction.is_none());
        assert!(!ctrl.is_identity());
    }

    #[test]
    fn instruction_override_leaves_other_axes_none() {
        let ctrl = ProsodyControl::default().with_instruction("speak calmly");
        assert_eq!(ctrl.instruction.as_deref(), Some("speak calmly"));
        assert!(ctrl.pitch_shift.is_none());
        assert!(ctrl.speed_scale.is_none());
        assert!(ctrl.pause_ms.is_none());
        assert!(!ctrl.is_identity());
    }

    // ---- Chained overrides ----------------------------------------------

    #[test]
    fn chained_overrides_set_every_axis() {
        // Chained builder: every field takes its assigned value; the
        // control is no longer identity.
        let ctrl = ProsodyControl::default()
            .with_pitch_shift(-1.0)
            .with_speed_scale(0.75)
            .with_pause_ms(400)
            .with_instruction("emphasise the second word");
        assert_eq!(ctrl.pitch_shift, Some(-1.0));
        assert_eq!(ctrl.speed_scale, Some(0.75));
        assert_eq!(ctrl.pause_ms, Some(400));
        assert_eq!(
            ctrl.instruction.as_deref(),
            Some("emphasise the second word")
        );
        assert!(!ctrl.is_identity());
    }

    #[test]
    fn later_override_wins_on_same_axis() {
        // Chaining is last-write-wins per axis (no accidental merging).
        let ctrl = ProsodyControl::default()
            .with_speed_scale(1.25)
            .with_speed_scale(1.5);
        assert_eq!(ctrl.speed_scale, Some(1.5));
    }

    // ---- Validation -----------------------------------------------------

    #[test]
    fn validate_accepts_identity_and_reasonable_values() {
        assert!(ProsodyControl::default().validate().is_ok());
        let ctrl = ProsodyControl::default()
            .with_pitch_shift(0.5)
            .with_speed_scale(1.5)
            .with_pause_ms(200)
            .with_instruction("neutral tone");
        assert!(ctrl.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_finite_pitch() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let ctrl = ProsodyControl::default().with_pitch_shift(bad);
            assert!(
                matches!(ctrl.validate(), Err(VokraError::InvalidArgument(_))),
                "pitch_shift {bad} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_non_finite_speed() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let ctrl = ProsodyControl::default().with_speed_scale(bad);
            assert!(
                matches!(ctrl.validate(), Err(VokraError::InvalidArgument(_))),
                "speed_scale {bad} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_non_positive_speed() {
        for bad in [0.0f32, -0.5, -1.0] {
            let ctrl = ProsodyControl::default().with_speed_scale(bad);
            assert!(
                matches!(ctrl.validate(), Err(VokraError::InvalidArgument(_))),
                "speed_scale {bad} must be rejected"
            );
        }
    }

    // ---- ApplyProsody trait plumbing ------------------------------------

    /// Internal mock that mimics what the future CosyVoice2 adapter will do:
    /// fold the numeric axes into an `instruction` string. Used to prove
    /// the trait wiring works today, without needing M3-09 to have landed.
    struct MockAdapter;

    impl ApplyProsody for MockAdapter {
        fn apply(&self, ctx: &mut ProsodyControl) {
            // Identity passthrough contract: leave an identity control
            // untouched.
            if ctx.is_identity() {
                return;
            }
            let mut parts: Vec<String> = Vec::new();
            if let Some(p) = ctx.pitch_shift {
                parts.push(format!("pitch={p}"));
            }
            if let Some(s) = ctx.speed_scale {
                parts.push(format!("speed={s}"));
            }
            if let Some(ms) = ctx.pause_ms {
                parts.push(format!("pause_ms={ms}"));
            }
            if let Some(existing) = ctx.instruction.as_ref() {
                parts.push(existing.clone());
            }
            ctx.instruction = Some(parts.join("; "));
        }
    }

    #[test]
    fn apply_prosody_can_be_implemented_and_leaves_identity_untouched() {
        // The core trait plumbing test: someone (CosyVoice2 in M3-09) can
        // implement ApplyProsody, and the identity contract is honoured.
        let mut ctrl = ProsodyControl::default();
        MockAdapter.apply(&mut ctrl);
        assert!(
            ctrl.is_identity(),
            "identity input must remain identity: {ctrl:?}"
        );
    }

    #[test]
    fn apply_prosody_mutates_in_place_when_non_identity() {
        // Non-identity input: apply() may rewrite ctx (here, fold into
        // instruction). The exact folding is model-specific — this test
        // only verifies that the trait signature really is &mut and that
        // an implementation can observe every axis.
        let mut ctrl = ProsodyControl::default()
            .with_pitch_shift(1.0)
            .with_speed_scale(1.5);
        MockAdapter.apply(&mut ctrl);
        let s = ctrl.instruction.as_deref().unwrap_or_default();
        assert!(s.contains("pitch=1"), "instruction={s}");
        assert!(s.contains("speed=1.5"), "instruction={s}");
    }

    #[test]
    fn apply_prosody_is_idempotent_on_identity() {
        // Identity + identical `self` on repeated calls: control must
        // stay identity (contract per trait rustdoc).
        let mut ctrl = ProsodyControl::default();
        MockAdapter.apply(&mut ctrl);
        MockAdapter.apply(&mut ctrl);
        assert!(ctrl.is_identity());
    }
}
