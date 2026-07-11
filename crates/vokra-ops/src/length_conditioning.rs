//! F5-TTS / CosyVoice2 型 length conditioning for Flow Matching full-length
//! generation (M3-08; FR-OP-71).
//!
//! Produces the **target frame count** a Flow Matching sampler needs to
//! generate a full utterance in one shot. Two estimation modes are exposed
//! (both round to the nearest frame, ties-to-even via `f32::round`):
//!
//! - **Mode A — `UserSpecified`**: the caller supplies a duration in
//!   [`DurationUnit::Seconds`] or [`DurationUnit::Frames`]. Seconds are
//!   converted to frames via the attrs' `sample_rate` and `hop_length`
//!   (the frontend_spec-derived values from M0-04 / M1-03):
//!   `frames = round(seconds · sample_rate / hop_length)`.
//! - **Mode B — `RefLinear`**: a reference utterance length in frames plus a
//!   text-length ratio yields `target = round(ref_speech_frames · text_ratio)`.
//!   The linear form matches the simplest F5-TTS / CosyVoice2 formulation
//!   (ADR 0010 §D4). Any tighter coupling with a specific model (e.g. the
//!   Flow Matching sampler's chunk-aware CFM path in CosyVoice2) is deferred
//!   to the consumer WP M3-09, which is where a *reference-matching* parity
//!   test lives — this WP is op-only.
//!
//! # Distinct op — NOT `duration_expander`
//!
//! `length_conditioning` (this op) is Flow Matching 全長条件付け: given a
//! single caller/reference-derived length, it produces **one** frame count
//! for the entire utterance. `duration_expander` (FR-OP-70 想定,
//! FastSpeech2 型) predicts **per-phoneme** durations and expands the
//! phoneme sequence to a frame sequence. Both the IR variants
//! ([`vokra_core::OpKind::LengthConditioning`] vs the reserved
//! `duration_expander` placeholder) and the attribute types are separate,
//! so mixing them is a
//! **compile-time** error rather than a runtime one. See ADR 0010 §D2 / §D6
//! and the IR-distinction test in `tests/length_conditioning_ir_distinction.rs`.
//!
//! `mas` (FR-OP-72, Monotonic Alignment Search — the VITS-family
//! phoneme-frame aligner used inside piper-plus) is a different alignment
//! layer altogether and is out of this WP's scope (ADR 0010 §D3).
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Invalid inputs — negatives, non-finite floats, zero `sample_rate` /
//! `hop_length` when converting seconds, zero `ref_speech_frames`, a
//! non-finite `text_ratio`, or a computed target that overflows `u32` —
//! all raise [`VokraError::InvalidArgument`] rather than silently clamping,
//! defaulting, or producing a bogus length.

use vokra_core::ir::graph::{DurationUnit, LengthConditioningAttrs, LengthConditioningMode};
use vokra_core::{Result, VokraError};

/// Evaluates the [`OpKind::LengthConditioning`] op and returns the target
/// frame count for Flow Matching full-length generation (FR-OP-71).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - a negative or non-finite `duration` (mode A);
/// - a zero `sample_rate` or `hop_length` when `unit == Seconds` (mode A);
/// - a zero `ref_speech_frames` or a non-finite / negative `text_ratio`
///   (mode B);
/// - a computed frame count that overflows `u32` (either mode).
///
/// The op never silently rounds a bad input to zero (FR-EX-08).
///
/// [`OpKind::LengthConditioning`]: vokra_core::OpKind::LengthConditioning
pub fn length_conditioning(attrs: &LengthConditioningAttrs) -> Result<u32> {
    let frames = match attrs.mode {
        LengthConditioningMode::UserSpecified { duration } => {
            eval_user_specified(duration, attrs.unit, attrs.sample_rate, attrs.hop_length)?
        }
        LengthConditioningMode::RefLinear {
            ref_speech_frames,
            text_ratio,
        } => eval_ref_linear(ref_speech_frames, text_ratio)?,
    };
    Ok(frames)
}

/// Mode A: caller-supplied duration.
///
/// For `Seconds`, converts via `frames = round(seconds · sample_rate /
/// hop_length)` using the frontend_spec-derived `sample_rate` and
/// `hop_length` (M0-04 / M1-03). For `Frames`, uses the value verbatim.
fn eval_user_specified(
    duration: f32,
    unit: DurationUnit,
    sample_rate: u32,
    hop_length: u32,
) -> Result<u32> {
    if !duration.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "length_conditioning: UserSpecified duration must be finite (got {duration})"
        )));
    }
    if duration < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "length_conditioning: UserSpecified duration must be non-negative (got {duration})"
        )));
    }
    let frames_f32 = match unit {
        DurationUnit::Frames => duration,
        DurationUnit::Seconds => {
            if sample_rate == 0 || hop_length == 0 {
                return Err(VokraError::InvalidArgument(
                    "length_conditioning: unit=Seconds requires non-zero sample_rate and \
                     hop_length"
                        .to_owned(),
                ));
            }
            // f64 intermediate keeps sample_rate * duration from cascading a
            // second rounding step (u32 -> f32 loses precision for
            // sample_rate ≥ 16_777_217, which is outside any speech target but
            // still cheap to avoid).
            let sr = f64::from(sample_rate);
            let hop = f64::from(hop_length);
            (f64::from(duration) * sr / hop) as f32
        }
    };
    round_to_u32(frames_f32, "UserSpecified")
}

/// Mode B: linear estimation `target = round(ref_speech_frames · text_ratio)`.
///
/// The simplest F5-TTS / CosyVoice2 formulation (ADR 0010 §D4). A tighter
/// coupling (e.g. per-chunk correction) is a consumer-WP concern (M3-09).
fn eval_ref_linear(ref_speech_frames: u32, text_ratio: f32) -> Result<u32> {
    if ref_speech_frames == 0 {
        return Err(VokraError::InvalidArgument(
            "length_conditioning: RefLinear ref_speech_frames must be non-zero".to_owned(),
        ));
    }
    if !text_ratio.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "length_conditioning: RefLinear text_ratio must be finite (got {text_ratio})"
        )));
    }
    if text_ratio < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "length_conditioning: RefLinear text_ratio must be non-negative (got {text_ratio})"
        )));
    }
    // f64 intermediate: u32 · f32 can lose precision past 2^24; scaling in
    // f64 leaves the product exact for every u32 · finite-f32 pair.
    let frames_f32 = (f64::from(ref_speech_frames) * f64::from(text_ratio)) as f32;
    round_to_u32(frames_f32, "RefLinear")
}

/// Rounds a non-negative finite `f32` to `u32`, returning an explicit error
/// if the value overflows the `u32` range (FR-EX-08 — no silent clamp).
fn round_to_u32(value: f32, ctx: &str) -> Result<u32> {
    if !value.is_finite() || value < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "length_conditioning: {ctx} produced non-representable frame count {value}"
        )));
    }
    let rounded = value.round();
    // `u32::MAX as f32` rounds to 2^32, so guard against >= 2^32 explicitly.
    if rounded >= f32::from_bits(0x4F800000) {
        // 2^32 as f32
        return Err(VokraError::InvalidArgument(format!(
            "length_conditioning: {ctx} produced a frame count {rounded} that overflows u32"
        )));
    }
    Ok(rounded as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::OpKind;

    // ---- Mode A: UserSpecified ------------------------------------------

    #[test]
    fn user_specified_frames_returns_rounded_value() {
        // Explicit frame input: no unit conversion, just round-to-nearest.
        let attrs = LengthConditioningAttrs::user_specified_frames(200.0);
        assert_eq!(length_conditioning(&attrs).unwrap(), 200);

        // Non-integer input rounds to nearest (200.5 -> 200 by ties-to-even,
        // 200.7 -> 201).
        let attrs = LengthConditioningAttrs::user_specified_frames(200.7);
        assert_eq!(length_conditioning(&attrs).unwrap(), 201);
    }

    #[test]
    fn user_specified_seconds_converts_via_sample_rate_and_hop() {
        // 2.0 s at 22_050 Hz with hop = 256 → 2.0 · 22_050 / 256 = 172.265625,
        // round → 172. The stable arithmetic is what makes it safe to bake into
        // a graph attribute (M1-03 frontend_spec).
        let attrs = LengthConditioningAttrs::user_specified_seconds(2.0, 22_050, 256);
        assert_eq!(length_conditioning(&attrs).unwrap(), 172);

        // 1.0 s at 16_000 Hz with hop = 160 (Whisper front-end) → 100 frames.
        let attrs = LengthConditioningAttrs::user_specified_seconds(1.0, 16_000, 160);
        assert_eq!(length_conditioning(&attrs).unwrap(), 100);

        // Exactly-zero duration is a valid degenerate case (empty utterance).
        let attrs = LengthConditioningAttrs::user_specified_seconds(0.0, 16_000, 160);
        assert_eq!(length_conditioning(&attrs).unwrap(), 0);
    }

    #[test]
    fn user_specified_rejects_negative_and_non_finite() {
        let neg = LengthConditioningAttrs::user_specified_frames(-1.0);
        assert!(matches!(
            length_conditioning(&neg),
            Err(VokraError::InvalidArgument(_))
        ));
        let nan = LengthConditioningAttrs::user_specified_frames(f32::NAN);
        assert!(matches!(
            length_conditioning(&nan),
            Err(VokraError::InvalidArgument(_))
        ));
        let inf = LengthConditioningAttrs::user_specified_frames(f32::INFINITY);
        assert!(matches!(
            length_conditioning(&inf),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn user_specified_seconds_rejects_zero_sample_rate_or_hop() {
        // Zero sample_rate: seconds cannot be converted (FR-EX-08 = no silent
        // default to 16 kHz — the graph attribute must supply the front-end).
        let bad_sr = LengthConditioningAttrs {
            mode: LengthConditioningMode::UserSpecified { duration: 1.0 },
            unit: DurationUnit::Seconds,
            sample_rate: 0,
            hop_length: 160,
        };
        assert!(matches!(
            length_conditioning(&bad_sr),
            Err(VokraError::InvalidArgument(_))
        ));

        let bad_hop = LengthConditioningAttrs {
            mode: LengthConditioningMode::UserSpecified { duration: 1.0 },
            unit: DurationUnit::Seconds,
            sample_rate: 16_000,
            hop_length: 0,
        };
        assert!(matches!(
            length_conditioning(&bad_hop),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- Mode B: RefLinear -----------------------------------------------

    #[test]
    fn ref_linear_scales_reference_length_by_text_ratio() {
        // ref_len=100 frames, target text is 2x → 200 frames (the concrete
        // shape spec called out in the WP directive: len(ref)=100 · 2.0 = 200).
        let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
        assert_eq!(length_conditioning(&attrs).unwrap(), 200);

        // Sub-1 ratio: 100 · 0.5 = 50.
        let attrs = LengthConditioningAttrs::ref_linear(100, 0.5);
        assert_eq!(length_conditioning(&attrs).unwrap(), 50);

        // Rounding of a non-integer product: 33 · 1.5 = 49.5 → 50 (round even).
        let attrs = LengthConditioningAttrs::ref_linear(33, 1.5);
        assert_eq!(length_conditioning(&attrs).unwrap(), 50);
    }

    #[test]
    fn ref_linear_identity_ratio_preserves_reference_length() {
        // ratio = 1.0 recovers the reference length exactly; a useful invariant
        // for "regenerate the same utterance under Flow Matching" test paths.
        let attrs = LengthConditioningAttrs::ref_linear(157, 1.0);
        assert_eq!(length_conditioning(&attrs).unwrap(), 157);
    }

    #[test]
    fn ref_linear_zero_ratio_yields_zero_frames() {
        // Degenerate but well-defined: text_ratio=0 collapses to a zero-frame
        // target (empty output). Distinct from an error case.
        let attrs = LengthConditioningAttrs::ref_linear(100, 0.0);
        assert_eq!(length_conditioning(&attrs).unwrap(), 0);
    }

    #[test]
    fn ref_linear_rejects_zero_reference_and_bad_ratio() {
        // Zero reference: F5-TTS / CosyVoice2 need *some* utterance length
        // signal; a zero reference has no linear estimate.
        let zero_ref = LengthConditioningAttrs::ref_linear(0, 1.0);
        assert!(matches!(
            length_conditioning(&zero_ref),
            Err(VokraError::InvalidArgument(_))
        ));

        // NaN / Infinity / negative ratio → explicit error.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5] {
            let attrs = LengthConditioningAttrs::ref_linear(100, bad);
            assert!(
                matches!(
                    length_conditioning(&attrs),
                    Err(VokraError::InvalidArgument(_))
                ),
                "ratio {bad} must error"
            );
        }
    }

    #[test]
    fn ref_linear_overflow_is_explicit() {
        // 2^32-ish target: ref_speech_frames · text_ratio round up past u32::MAX
        // must not silently wrap. This is the "no silent clamp" contract.
        let attrs = LengthConditioningAttrs::ref_linear(u32::MAX, 2.0);
        assert!(matches!(
            length_conditioning(&attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- IR variant plumbing --------------------------------------------

    #[test]
    fn opkind_length_conditioning_carries_attrs_and_is_debug() {
        // The op enum wraps our attrs and is Debug-printable, i.e. it plugs
        // into `AudioGraph` node listings uniformly with the other variants.
        let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
        let op = OpKind::LengthConditioning(attrs);
        let s = format!("{op:?}");
        assert!(s.contains("LengthConditioning"), "debug repr: {s}");
    }
}
