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

/// Validates the [`AgcAttrs`] ranges (shared by [`agc`] and [`AgcState::new`]).
fn validate_agc_attrs(attrs: &AgcAttrs) -> Result<()> {
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
    Ok(())
}

/// Applies automatic gain control to `input`, returning the leveled signal
/// (same length). Zero initial envelope; unity initial gain.
///
/// This is the whole-buffer convenience wrapper over [`AgcState`]: it is
/// exactly `AgcState::new(attrs)?.process_chunk(input)`, so one call is
/// bit-identical to feeding the same samples through the streaming state in
/// pieces (the `streaming_chunks_match_whole_buffer` test pins this).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for out-of-range attributes (a coefficient
/// outside `(0, 1]`, `target_level` / `limiter_ceiling` outside `(0, 1]`,
/// `max_gain < 1`) or a non-finite input sample (FR-EX-08).
pub fn agc(input: &[f32], attrs: &AgcAttrs) -> Result<Vec<f32>> {
    AgcState::new(attrs)?.process_chunk(input)
}

/// Streaming AGC state: carries the peak envelope and the smoothed gain across
/// [`process_chunk`](AgcState::process_chunk) calls, so a call-center / Discord
/// streaming caller can feed successive chunks and get output bit-identical to
/// running [`agc`] over the concatenated buffer.
///
/// Like [`Aec`](crate::Aec) and [`HpfState`](crate::HpfState), this is a
/// first-class API type rather than an `OpKind` variant (ADR M4-20 §D-5): it
/// owns live per-frame state, so a graph-side call has no home and falls into
/// the `dispatch.rs` `UnsupportedOp` default (FR-EX-08).
#[derive(Debug, Clone)]
pub struct AgcState {
    attrs: AgcAttrs,
    /// Peak envelope of `|x|` (fast-attack / slow-release follower).
    env: f32,
    /// Smoothed applied gain.
    gain: f32,
}

impl AgcState {
    /// Builds a fresh AGC state (zero envelope, unity gain).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for out-of-range attributes (same
    /// ranges as [`agc`]).
    pub fn new(attrs: &AgcAttrs) -> Result<Self> {
        validate_agc_attrs(attrs)?;
        Ok(Self {
            attrs: *attrs,
            env: 0.0,
            gain: 1.0,
        })
    }

    /// The attributes this state was built with.
    #[inline]
    #[must_use]
    pub fn attrs(&self) -> &AgcAttrs {
        &self.attrs
    }

    /// Rewinds to the as-new state (zero envelope, unity gain) — the barge-in
    /// reset companion, mirroring [`Aec::reset`](crate::Aec::reset).
    pub fn reset(&mut self) {
        self.env = 0.0;
        self.gain = 1.0;
    }

    /// Processes one `chunk`, advancing the carried envelope / gain, and
    /// returns the leveled samples (same length as `chunk`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a non-finite input sample (FR-EX-08
    /// — never a silent clamp).
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<f32>> {
        if chunk.iter().any(|s| !s.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "agc: input has a non-finite sample".into(),
            ));
        }
        const EPS: f32 = 1e-6;
        let mut out = Vec::with_capacity(chunk.len());
        for &s in chunk {
            let mag = s.abs();
            // Peak envelope: fast attack up, slow release down.
            if mag > self.env {
                self.env += self.attrs.attack * (mag - self.env);
            } else {
                self.env += self.attrs.release * (mag - self.env);
            }
            // Desired gain to reach the target, capped at max_gain.
            let desired =
                (self.attrs.target_level / self.env.max(EPS)).clamp(0.0, self.attrs.max_gain);
            self.gain += self.attrs.gain_smooth * (desired - self.gain);
            let y = (s * self.gain).clamp(-self.attrs.limiter_ceiling, self.attrs.limiter_ceiling);
            out.push(y);
        }
        Ok(out)
    }
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

    // ---- Streaming state (AgcState) --------------------------------------

    #[test]
    fn streaming_chunks_match_whole_buffer() {
        // Chunked processing through `AgcState` must be bit-identical to the
        // one-shot stateless `agc()` over the same signal — the state carries
        // the peak envelope + smoothed gain across `process_chunk` calls. A
        // level-stepped signal keeps the envelope/gain mid-evolution at every
        // chunk boundary (a static tone would mask a state-carry bug).
        let attrs = AgcAttrs::speech_default();
        let mut x = tone(300.0, 16000.0, 0.05, 4000);
        x.extend(tone(300.0, 16000.0, 0.6, 4000));
        x.extend(tone(300.0, 16000.0, 0.2, 4000));

        let whole = agc(&x, &attrs).unwrap();

        let mut state = AgcState::new(&attrs).unwrap();
        let mut streamed = Vec::new();
        // Uneven chunk length, unaligned to the tone period or the level
        // steps, so the carry is exercised at arbitrary phases.
        for chunk in x.chunks(777) {
            streamed.extend(state.process_chunk(chunk).unwrap());
        }
        assert_eq!(
            streamed, whole,
            "chunked AGC must be bit-identical to whole-buffer"
        );
    }

    #[test]
    fn reset_restores_fresh_state() {
        // A second pass without reset differs (carried gain/env); after
        // `reset` the state is as-new and reproduces the first pass exactly.
        let attrs = AgcAttrs::speech_default();
        let x = tone(300.0, 16000.0, 0.4, 2000);

        let mut state = AgcState::new(&attrs).unwrap();
        let first = state.process_chunk(&x).unwrap();
        let second = state.process_chunk(&x).unwrap();
        assert_ne!(first, second, "carried state must affect the 2nd pass");
        state.reset();
        let third = state.process_chunk(&x).unwrap();
        assert_eq!(first, third, "reset must reproduce a fresh AgcState");
    }

    #[test]
    fn state_new_rejects_bad_attrs_and_process_rejects_nonfinite() {
        let base = AgcAttrs::speech_default();
        assert!(
            AgcState::new(&AgcAttrs {
                target_level: 0.0,
                ..base
            })
            .is_err()
        );
        assert!(
            AgcState::new(&AgcAttrs {
                max_gain: 0.5,
                ..base
            })
            .is_err()
        );
        let mut state = AgcState::new(&base).unwrap();
        assert!(state.process_chunk(&[f32::INFINITY]).is_err());
        assert_eq!(state.attrs().target_level, base.target_level);
    }
}
