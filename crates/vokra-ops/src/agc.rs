//! `agc` — automatic gain control (M4-20 (c), FR-OP-62): an envelope-following
//! digital AGC for the WebRTC audio-processing role (CLAUDE.md「agc — WebRTC
//! audio processing pipeline 準拠」).
//!
//! # Runtime function, not an `OpKind` variant (ADR M4-20 §D-5)
//!
//! `agc` carries per-frame gain / envelope state, so — like `aec` and `hpf` —
//! it is a first-class API function, not an `OpKind` enum variant. A graph-side
//! call falls into the `dispatch.rs` `UnsupportedOp` default (FR-EX-08).
//!
//! # Algorithm (digital-AGC gain law — tunables, not invented constants)
//!
//! A peak-envelope follower (fast attack, slow release) drives a gain that
//! pulls the signal level toward `target_level`, capped at `max_gain`, with a
//! hard output limiter at `limiter_ceiling`. WebRTC's own AGC uses specific
//! internal operating constants; those are **not** transcribed here — the gain
//! law is the documented digital-AGC form and every constant is an
//! [`AgcAttrs`] field the caller sets (ADR M4-20 §D-5). The mathematical
//! oracles the parity test asserts (T13): a stationary tone converges so the
//! output level ≈ `target_level` (when reachable within `max_gain`), and the
//! output never exceeds `limiter_ceiling`.

use vokra_core::{Result, VokraError};

/// Attributes for the [`agc`] automatic gain control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgcAttrs {
    /// Target output level in `(0, 1]` (peak amplitude the AGC aims for).
    pub target_level: f32,
    /// Maximum linear gain the AGC may apply (`>= 1`; e.g. `20.0` ≈ 26 dB).
    pub max_gain: f32,
    /// Per-sample envelope **attack** coefficient in `(0, 1]` (fraction of the
    /// gap closed when the input rises above the envelope; larger = faster).
    pub attack: f32,
    /// Per-sample envelope **release** coefficient in `(0, 1]` (fraction closed
    /// when the input falls below the envelope; smaller = slower decay).
    pub release: f32,
    /// Per-sample gain-smoothing coefficient in `(0, 1]` (anti-zipper).
    pub gain_smooth: f32,
    /// Hard output ceiling (`(0, 1]`) applied after the gain (the limiter).
    pub limiter_ceiling: f32,
}

impl AgcAttrs {
    /// A sensible speech default: target 0.25, max gain 20, fast attack /
    /// slow release / gentle gain smoothing, limiter at 0.99. The time
    /// constants are documented tunables (not WebRTC-exact, ADR M4-20 §D-5).
    pub fn speech_default() -> Self {
        Self {
            target_level: 0.25,
            max_gain: 20.0,
            attack: 0.2,
            release: 0.002,
            gain_smooth: 0.02,
            limiter_ceiling: 0.99,
        }
    }
}

/// Applies automatic gain control to `input`, returning the leveled signal
/// (same length). Zero initial envelope; unity initial gain.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for out-of-range attributes (a coefficient
/// outside `(0, 1]`, `target_level` / `limiter_ceiling` outside `(0, 1]`,
/// `max_gain < 1`) or a non-finite input sample (FR-EX-08).
pub fn agc(input: &[f32], attrs: &AgcAttrs) -> Result<Vec<f32>> {
    let unit = |v: f32| v > 0.0 && v <= 1.0;
    if !unit(attrs.target_level) || !unit(attrs.limiter_ceiling) {
        return Err(VokraError::InvalidArgument(
            "agc: target_level and limiter_ceiling must be in (0, 1]".into(),
        ));
    }
    if attrs.max_gain < 1.0 || attrs.max_gain.is_nan() {
        return Err(VokraError::InvalidArgument(
            "agc: max_gain must be >= 1".into(),
        ));
    }
    if !unit(attrs.attack) || !unit(attrs.release) || !unit(attrs.gain_smooth) {
        return Err(VokraError::InvalidArgument(
            "agc: attack / release / gain_smooth must be in (0, 1]".into(),
        ));
    }
    if input.iter().any(|s| !s.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "agc: input has a non-finite sample".into(),
        ));
    }

    const EPS: f32 = 1e-6;
    let mut out = Vec::with_capacity(input.len());
    let mut env = 0.0f32; // peak envelope of |x|
    let mut gain = 1.0f32; // smoothed applied gain
    for &s in input {
        let mag = s.abs();
        // Peak envelope: fast attack up, slow release down.
        if mag > env {
            env += attrs.attack * (mag - env);
        } else {
            env += attrs.release * (mag - env);
        }
        // Desired gain to reach the target, capped at max_gain.
        let desired = (attrs.target_level / env.max(EPS)).clamp(0.0, attrs.max_gain);
        gain += attrs.gain_smooth * (desired - gain);
        let y = (s * gain).clamp(-attrs.limiter_ceiling, attrs.limiter_ceiling);
        out.push(y);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, fs: f32, amp: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
            .collect()
    }

    fn tail_peak(x: &[f32], tail: usize) -> f32 {
        x[x.len() - tail..]
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    #[test]
    fn quiet_tone_is_boosted_toward_target() {
        // A quiet 300 Hz tone (amp 0.05) with target 0.25 and max_gain 20
        // (reachable: needs gain 5) converges so the output peak ≈ target.
        let fs = 16000.0;
        let x = tone(300.0, fs, 0.05, 48000);
        let out = agc(&x, &AgcAttrs::speech_default()).unwrap();
        let peak = tail_peak(&out, 8000);
        assert!(
            (0.20..=0.30).contains(&peak),
            "quiet tone should be boosted to ~target 0.25, got {peak}"
        );
    }

    #[test]
    fn loud_tone_is_attenuated_toward_target() {
        // A loud tone (amp 0.8) with target 0.25 → gain < 1 → output ≈ target.
        let fs = 16000.0;
        let x = tone(300.0, fs, 0.8, 48000);
        let out = agc(&x, &AgcAttrs::speech_default()).unwrap();
        let peak = tail_peak(&out, 8000);
        assert!(
            (0.20..=0.30).contains(&peak),
            "loud tone should be pulled down to ~target 0.25, got {peak}"
        );
    }

    #[test]
    fn output_never_exceeds_limiter_ceiling() {
        // A signal whose target gain would push it past 1.0 is clamped by the
        // limiter: no output sample exceeds the ceiling.
        let mut attrs = AgcAttrs::speech_default();
        attrs.target_level = 1.0;
        attrs.limiter_ceiling = 0.5; // force clipping to the ceiling
        let x = tone(300.0, 16000.0, 0.9, 16000);
        let out = agc(&x, &attrs).unwrap();
        let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak <= 0.5 + 1e-6, "limiter must cap at 0.5, got {peak}");
    }

    #[test]
    fn max_gain_caps_the_boost() {
        // A near-silent input (amp 0.001) with target 0.25 would need gain 250,
        // but max_gain 20 caps it: output peak ≈ 0.001 * 20 = 0.02, far below
        // target. Proves the cap fires (never a silent unbounded gain).
        let x = tone(300.0, 16000.0, 0.001, 48000);
        let out = agc(&x, &AgcAttrs::speech_default()).unwrap();
        let peak = tail_peak(&out, 8000);
        assert!(peak < 0.05, "max_gain must cap the boost, got {peak}");
    }

    #[test]
    fn rejects_bad_attrs_and_nonfinite() {
        let base = AgcAttrs::speech_default();
        assert!(
            agc(
                &[0.0],
                &AgcAttrs {
                    target_level: 0.0,
                    ..base
                }
            )
            .is_err()
        );
        assert!(
            agc(
                &[0.0],
                &AgcAttrs {
                    max_gain: 0.5,
                    ..base
                }
            )
            .is_err()
        );
        assert!(
            agc(
                &[0.0],
                &AgcAttrs {
                    attack: 1.5,
                    ..base
                }
            )
            .is_err()
        );
        assert!(agc(&[f32::INFINITY], &base).is_err());
    }
}
