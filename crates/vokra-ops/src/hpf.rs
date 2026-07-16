//! `hpf` — high-pass filter (M4-20 (c), FR-OP-62): a 2nd-order high-pass
//! biquad for the WebRTC audio-processing pre-filter role (CLAUDE.md「hpf
//! (30-80 Hz)」).
//!
//! # Runtime function, not an `OpKind` variant (ADR M4-20 §D-5)
//!
//! Like `aec` (that module), `hpf` carries biquad state and is exposed as a
//! first-class API function, **not** an `OpKind` enum variant / `dispatch.rs`
//! arm — a graph-side call falls into the existing `UnsupportedOp` default
//! (FR-EX-08). `HpfAttrs` is defined here (not in `vokra-core`) for the same
//! reason: it is not embedded in any `OpKind`.
//!
//! # Design (precise published spec — no invented constants)
//!
//! A 2nd-order high-pass **biquad** with the Robert Bristow-Johnson (RBJ)
//! audio-EQ cookbook transfer function, Direct-Form-I. Default resonance
//! `Q = 1/√2` (Butterworth, maximally flat passband). WebRTC's own high-pass
//! is a fixed internal biquad cascade; the RBJ design is the documented general
//! spec anchor for FR-OP-62 and its coefficients are computed from the cutoff /
//! sample-rate, not transcribed magic numbers (ADR M4-20 §D-5).
//!
//! The DC gain of a high-pass biquad is exactly 0 (`b0 + b1 + b2 = 0`), so a
//! constant (DC) input decays to zero — the mathematical oracle the parity test
//! asserts (T13).

use vokra_core::{Result, VokraError};

/// Attributes for the [`hpf`] high-pass filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HpfAttrs {
    /// Cutoff frequency in Hz (−3 dB point). FR-OP-62 targets the 30–80 Hz
    /// band; any value in `(0, sample_rate/2)` is accepted.
    pub cutoff_hz: f32,
    /// Input sample rate in Hz (must match `vokra.frontend.*`, M0-04).
    pub sample_rate: u32,
    /// Biquad resonance `Q`. Default `1/√2` (Butterworth). Higher `Q` sharpens
    /// the corner (with passband ripple).
    pub q: f32,
}

impl HpfAttrs {
    /// A Butterworth (`Q = 1/√2`) high-pass at `cutoff_hz` for `sample_rate`.
    pub fn butterworth(cutoff_hz: f32, sample_rate: u32) -> Self {
        Self {
            cutoff_hz,
            sample_rate,
            q: std::f32::consts::FRAC_1_SQRT_2,
        }
    }
}

/// Applies a 2nd-order high-pass biquad to `input`, returning the filtered
/// signal (same length). Zero initial state (a whole-signal transform).
///
/// This is the whole-buffer convenience wrapper over [`HpfState`]: it is
/// exactly `HpfState::new(attrs)?.process_chunk(input)`, so one call is
/// bit-identical to streaming the same samples through the state in pieces
/// (the `streaming_chunks_match_whole_buffer` test pins this).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for a zero sample rate, a cutoff outside
/// `(0, sample_rate/2)`, a non-positive `Q`, or a non-finite input sample
/// (FR-EX-08 — never a silent clamp).
pub fn hpf(input: &[f32], attrs: &HpfAttrs) -> Result<Vec<f32>> {
    HpfState::new(attrs)?.process_chunk(input)
}

/// Streaming high-pass state: carries the Direct-Form-I biquad memory
/// (`x1/x2/y1/y2`) across [`process_chunk`](HpfState::process_chunk) calls, so
/// a call-center / Discord streaming caller can feed successive chunks and get
/// output bit-identical to running [`hpf`] over the concatenated buffer. The
/// RBJ coefficients are computed once at construction.
///
/// Like [`Aec`](crate::Aec) and [`AgcState`](crate::AgcState), this is a
/// first-class API type rather than an `OpKind` variant (ADR M4-20 §D-5): it
/// owns live filter state, so a graph-side call falls into the `dispatch.rs`
/// `UnsupportedOp` default (FR-EX-08).
#[derive(Debug, Clone)]
pub struct HpfState {
    attrs: HpfAttrs,
    // Normalized RBJ high-pass biquad coefficients (already a0-divided).
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // Direct-Form-I memory: previous two inputs / outputs.
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl HpfState {
    /// Builds a fresh high-pass state, computing the RBJ coefficients from
    /// `attrs` and zeroing the biquad memory.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for a zero sample rate, a cutoff
    /// outside `(0, sample_rate/2)`, or a non-positive `Q`.
    pub fn new(attrs: &HpfAttrs) -> Result<Self> {
        if attrs.sample_rate == 0 {
            return Err(VokraError::InvalidArgument("hpf: sample_rate is 0".into()));
        }
        let nyquist = attrs.sample_rate as f32 * 0.5;
        if !(attrs.cutoff_hz > 0.0 && attrs.cutoff_hz < nyquist) {
            return Err(VokraError::InvalidArgument(format!(
                "hpf: cutoff_hz {} must be in (0, {nyquist})",
                attrs.cutoff_hz
            )));
        }
        if attrs.q <= 0.0 || attrs.q.is_nan() {
            return Err(VokraError::InvalidArgument(format!(
                "hpf: Q {} must be > 0",
                attrs.q
            )));
        }

        // RBJ high-pass biquad coefficients (f64 for coefficient precision).
        let w0 = 2.0 * std::f64::consts::PI * attrs.cutoff_hz as f64 / attrs.sample_rate as f64;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * attrs.q as f64);
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        // Normalize by a0.
        let (b0, b1, b2, a1, a2) = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);

        Ok(Self {
            attrs: *attrs,
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        })
    }

    /// The attributes this state was built with.
    #[inline]
    #[must_use]
    pub fn attrs(&self) -> &HpfAttrs {
        &self.attrs
    }

    /// Rewinds the biquad memory to zero (as a fresh filter), keeping the
    /// coefficients — the barge-in reset companion, mirroring
    /// [`Aec::reset`](crate::Aec::reset).
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Filters one `chunk`, advancing the carried biquad memory, and returns
    /// the filtered samples (same length as `chunk`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a non-finite input sample (FR-EX-08
    /// — never a silent clamp).
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<f32>> {
        if chunk.iter().any(|s| !s.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "hpf: input has a non-finite sample".into(),
            ));
        }
        // Direct Form I, f64 accumulation.
        let mut out = Vec::with_capacity(chunk.len());
        for &s in chunk {
            let x0 = s as f64;
            let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x0;
            self.y2 = self.y1;
            self.y1 = y0;
            out.push(y0 as f32);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_input_decays_to_zero() {
        // A constant (DC) signal has zero high-pass DC gain: the steady-state
        // output must decay to ~0 (the biquad `b0+b1+b2 = 0` oracle).
        let attrs = HpfAttrs::butterworth(50.0, 16000);
        let out = hpf(&[1.0f32; 4096], &attrs).unwrap();
        let tail = &out[out.len() - 256..];
        let peak = tail.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 1e-3, "DC steady-state must decay to ~0, got {peak}");
    }

    #[test]
    fn high_frequency_tone_passes_largely_unattenuated() {
        // A tone well ABOVE the cutoff (2 kHz vs 50 Hz corner) passes with
        // near-unity gain: steady-state amplitude ≈ input amplitude.
        let fs = 16000u32;
        let f = 2000.0f32;
        let n = 8000usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / fs as f32).sin())
            .collect();
        let out = hpf(&x, &HpfAttrs::butterworth(50.0, fs)).unwrap();
        let tail = &out[out.len() - 2000..];
        let peak = tail.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(
            (0.9..=1.05).contains(&peak),
            "high tone should pass ~unattenuated, peak {peak}"
        );
    }

    #[test]
    fn low_frequency_tone_is_attenuated() {
        // A tone well BELOW the cutoff (10 Hz vs 80 Hz corner) is strongly
        // attenuated relative to its unit input amplitude.
        let fs = 16000u32;
        let f = 10.0f32;
        let n = 32000usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / fs as f32).sin())
            .collect();
        let out = hpf(&x, &HpfAttrs::butterworth(80.0, fs)).unwrap();
        let tail = &out[out.len() - 8000..];
        let peak = tail.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 0.3, "low tone should be attenuated, peak {peak}");
    }

    #[test]
    fn rejects_bad_attrs_and_nonfinite() {
        assert!(hpf(&[0.0], &HpfAttrs::butterworth(50.0, 0)).is_err());
        // cutoff above nyquist.
        assert!(hpf(&[0.0], &HpfAttrs::butterworth(9000.0, 16000)).is_err());
        // non-positive Q.
        assert!(
            hpf(
                &[0.0],
                &HpfAttrs {
                    cutoff_hz: 50.0,
                    sample_rate: 16000,
                    q: 0.0
                }
            )
            .is_err()
        );
        // non-finite input.
        assert!(hpf(&[f32::NAN], &HpfAttrs::butterworth(50.0, 16000)).is_err());
    }

    // ---- Streaming state (HpfState) --------------------------------------

    #[test]
    fn streaming_chunks_match_whole_buffer() {
        // Chunked biquad filtering through `HpfState` must be bit-identical to
        // the one-shot `hpf()`: the state carries the Direct-Form-I memory
        // (x1/x2/y1/y2) across `process_chunk` calls, and the coefficients are
        // computed once at construction. A mix of a sub-cutoff and a
        // supra-cutoff tone keeps real energy in the recurrence at every chunk
        // boundary.
        let fs = 16000u32;
        let attrs = HpfAttrs::butterworth(80.0, fs);
        let n = 10000usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.5 * (2.0 * std::f32::consts::PI * 30.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 1500.0 * t).sin()
            })
            .collect();

        let whole = hpf(&x, &attrs).unwrap();

        let mut state = HpfState::new(&attrs).unwrap();
        let mut streamed = Vec::new();
        // Unaligned chunk length so the biquad memory carry is exercised at
        // arbitrary phases.
        for chunk in x.chunks(513) {
            streamed.extend(state.process_chunk(chunk).unwrap());
        }
        assert_eq!(
            streamed, whole,
            "chunked HPF must be bit-identical to whole-buffer"
        );
    }

    #[test]
    fn reset_restores_fresh_state() {
        let attrs = HpfAttrs::butterworth(80.0, 16000);
        let x: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();
        let mut state = HpfState::new(&attrs).unwrap();
        let first = state.process_chunk(&x).unwrap();
        state.reset();
        let again = state.process_chunk(&x).unwrap();
        assert_eq!(first, again, "reset must restore the zero-memory state");
    }

    #[test]
    fn state_new_rejects_bad_attrs_and_process_rejects_nonfinite() {
        assert!(HpfState::new(&HpfAttrs::butterworth(50.0, 0)).is_err());
        assert!(HpfState::new(&HpfAttrs::butterworth(9000.0, 16000)).is_err());
        let attrs = HpfAttrs::butterworth(50.0, 16000);
        let mut state = HpfState::new(&attrs).unwrap();
        assert!(state.process_chunk(&[f32::NAN]).is_err());
        assert_eq!(state.attrs().cutoff_hz, attrs.cutoff_hz);
    }
}
