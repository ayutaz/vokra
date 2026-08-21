//! Arbitrary-rate audio resampling (M1-06a; FR-OP-64 `resample`).
//!
//! Converts a real PCM stream from `in_rate` to `out_rate` with a
//! **Kaiser-windowed-sinc interpolation kernel**. This is the standard
//! band-limited-interpolation / polyphase resampler of textbook DSP
//! (Oppenheim & Schafer §4.6; Kaiser 1974) — a *from-scratch re-derivation*,
//! not a port. It copies no code from **soxr** (LGPL) or **rubberband** (GPL);
//! those are neither vendored nor referenced (NFR-LC-03/04, CLAUDE.md red
//! line). There is no external crate: the [`sinc`](crate::window) and Kaiser
//! window (`I0` Bessel series) are first-party in [`crate::window`].
//!
//! # Algorithm
//!
//! Output sample `j` sits at input-sample position `c = j · in_rate/out_rate`.
//! Its value is a weighted sum of the input samples within a finite support
//! around `c`:
//!
//! ```text
//!   y[j] = ( Σ_i x[i] · h(c − i) ) / ( Σ_i h(c − i) )
//!   h(τ) = cutoff · sinc(cutoff · τ) · kaiser(τ / R)      for |τ| < R,  else 0
//! ```
//!
//! - `cutoff = min(1, out_rate/in_rate)` places the low-pass corner at the
//!   lower of the two Nyquist frequencies — an anti-imaging filter when
//!   upsampling, an anti-aliasing filter when downsampling.
//! - `R = half / cutoff` is the support half-width in input samples, where
//!   `half` is the number of sinc zero-crossings retained per side (a quality
//!   knob).
//! - The **per-output normalization** (dividing by the tap sum) pins the DC /
//!   passband gain to exactly `1.0` for every fractional phase, so a constant
//!   input maps to itself and a linear ramp is preserved on the interior.
//!
//! Equal rates short-circuit to a bit-exact copy. [`StreamingResampler`] adds
//! a fixed-size history ring and an absolute output-phase counter. It delays
//! each centered-sinc output until the right-hand support is available, then
//! treats `process_into(&[], out)` as the explicit end-of-stream flush. That
//! preserves the one-shot arithmetic exactly across arbitrary chunk borders.

use vokra_core::{Result, VokraError};

use crate::window::{bessel_i0, sinc};

/// The default resampling quality used by the frontend chain
/// ([`crate::preprocess::apply_frontend`]).
///
/// `frontend_spec` stores no resampler quality, so the chain fixes a strong
/// default here (see [`quality_params`]).
pub const DEFAULT_QUALITY: u8 = 5;

/// Resamples `input` from `in_rate` Hz to `out_rate` Hz.
///
/// `quality` selects the filter's zero-crossing count and Kaiser β (see
/// [`quality_params`]); higher is sharper and slower. Equal rates return a
/// bit-exact copy of `input`. The output length is
/// `round(input.len() · out_rate / in_rate)`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if `in_rate` or `out_rate` is zero.
///
/// # Examples
///
/// ```
/// let x = vec![0.0f32, 1.0, 0.0, -1.0];
/// // 1:1 resampling is the exact identity.
/// let y = vokra_ops::resample(&x, 16_000, 16_000, 5).unwrap();
/// assert_eq!(x, y);
/// ```
pub fn resample(input: &[f32], in_rate: u32, out_rate: u32, quality: u8) -> Result<Vec<f32>> {
    if in_rate == 0 || out_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "resample: in_rate and out_rate must be non-zero".to_owned(),
        ));
    }
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    Ok(SincResampler::new(in_rate, out_rate, quality).resample(input))
}

/// Maps a `quality` byte to `(half, beta)`: the sinc half-length in
/// zero-crossings and the Kaiser shape parameter β.
///
/// This table is **Vokra's own** internally-consistent design (we reimplement,
/// so it need not match speexdsp/soxr): larger β lowers the stopband floor,
/// larger `half` narrows the transition band. The realized passband ripple and
/// stopband attenuation are what the resampler tests assert against, rather
/// than any borrowed magic numbers. Values `>= 10` saturate to the top row.
pub fn quality_params(quality: u8) -> (usize, f64) {
    match quality {
        0 => (4, 6.0),
        1 => (6, 6.5),
        2 => (8, 7.0),
        3 => (10, 8.0),
        4 => (12, 8.5),
        5 => (16, 9.0),
        6 => (20, 10.0),
        7 => (24, 11.0),
        8 => (32, 12.0),
        9 => (48, 13.0),
        _ => (64, 14.0),
    }
}

/// A configured windowed-sinc resampler.
///
/// Holds the precomputed kernel geometry for one `(in_rate, out_rate, quality)`
/// triple. Both the one-shot path and [`StreamingResampler`] share this exact
/// kernel so chunking cannot alter tap order or floating-point arithmetic.
struct SincResampler {
    /// Output-to-input rate ratio (`out_rate / in_rate`).
    ratio: f64,
    /// Input samples advanced per output sample (`in_rate / out_rate`).
    step: f64,
    /// Low-pass cutoff as a fraction of the input Nyquist, `min(1, ratio)`.
    cutoff: f64,
    /// Kernel support half-width in input samples, `half / cutoff`.
    radius: f64,
    /// Kaiser shape parameter β.
    beta: f64,
    /// `I0(β)`, precomputed to normalize the Kaiser envelope.
    i0_beta: f64,
}

impl SincResampler {
    fn new(in_rate: u32, out_rate: u32, quality: u8) -> Self {
        let (half, beta) = quality_params(quality);
        let ratio = f64::from(out_rate) / f64::from(in_rate);
        let cutoff = ratio.min(1.0);
        Self {
            ratio,
            step: f64::from(in_rate) / f64::from(out_rate),
            cutoff,
            radius: half as f64 / cutoff,
            beta,
            i0_beta: bessel_i0(beta),
        }
    }

    /// The interpolation kernel `h(τ)` for an input-sample offset `τ`.
    ///
    /// Zero outside the open support `(−radius, radius)`; the endpoints are
    /// excluded symmetrically so the retained tap set stays symmetric about the
    /// output position (which is what preserves DC and linear signals).
    fn kernel(&self, tau: f64) -> f64 {
        let r = tau / self.radius;
        if r.abs() >= 1.0 {
            return 0.0;
        }
        let env = bessel_i0(self.beta * (1.0 - r * r).sqrt()) / self.i0_beta;
        self.cutoff * sinc(self.cutoff * tau) * env
    }

    fn resample(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let out_len = (n as f64 * self.ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for j in 0..out_len {
            let center = j as f64 * self.step;
            let lo = (center - self.radius).ceil() as isize;
            let hi = (center + self.radius).floor() as isize;
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in lo..=hi {
                let h = self.kernel(center - i as f64);
                den += h;
                if i >= 0 && (i as usize) < n {
                    num += f64::from(input[i as usize]) * h;
                }
            }
            let y = if den != 0.0 { num / den } else { 0.0 };
            out.push(y as f32);
        }
        out
    }
}

/// Stateful, allocation-free-after-construction streaming resampler.
///
/// The centered sinc kernel needs future input. [`process_into`](Self::process_into)
/// therefore emits only samples whose full right-hand support has arrived.
/// Call it once with an empty input slice after the final chunk to flush the
/// zero-padded tail and reproduce [`resample`] exactly. After a flush, more
/// non-empty input is rejected until [`reset`](Self::reset) makes the stream
/// reusable.
///
/// `out` must be large enough for every output made available by the supplied
/// chunk (or by the final flush). A too-small buffer returns
/// [`VokraError::InvalidArgument`] without consuming input or advancing phase.
/// This all-or-nothing contract is necessary because the API returns an output
/// count, not a separate consumed-input count.
pub struct StreamingResampler {
    sinc: SincResampler,
    /// Fixed-size ring of the most recent input samples. A newly-ready output
    /// spans at most `ceil(2 * radius) + 1` samples.
    history: Vec<f32>,
    delay_samples: usize,
    total_input: usize,
    next_output: usize,
    flushed: bool,
}

impl StreamingResampler {
    /// Creates a streaming resampler for one fixed rate/quality triple.
    ///
    /// All storage is allocated here. Successful calls to
    /// [`process_into`](Self::process_into) allocate nothing.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if either rate is zero.
    pub fn new(in_rate: u32, out_rate: u32, quality: u8) -> Result<Self> {
        if in_rate == 0 || out_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "StreamingResampler::new: in_rate and out_rate must be non-zero".to_owned(),
            ));
        }
        let (half, _) = quality_params(quality);
        let sinc = SincResampler::new(in_rate, out_rate, quality);
        let history_len = if in_rate == out_rate {
            1
        } else {
            (2.0 * sinc.radius).ceil() as usize + 3
        };
        let delay_samples = if in_rate == out_rate {
            0
        } else if out_rate < in_rate {
            // radius = half / (out_rate / in_rate), so converting the
            // look-ahead back to output-rate samples cancels the ratio.
            half
        } else {
            // Upsampling uses radius = half. Compute ceil(half * out / in)
            // with integers so an exact boundary cannot round up spuriously.
            let numerator = half as u64 * u64::from(out_rate);
            let denominator = u64::from(in_rate);
            numerator.div_ceil(denominator) as usize
        };
        Ok(Self {
            sinc,
            history: vec![0.0; history_len],
            delay_samples,
            total_input: 0,
            next_output: 0,
            flushed: false,
        })
    }

    /// Processes one input chunk into a caller-owned output buffer.
    ///
    /// A non-empty `input` appends to the current stream and returns the number
    /// of newly available output samples written to the start of `out`. An
    /// empty `input` is the explicit end-of-stream flush: it emits the
    /// centered kernel's zero-padded tail up to the same
    /// `round(total_input * out_rate / in_rate)` length as [`resample`].
    ///
    /// The successful path is zero-allocation. Output is bit-identical to the
    /// one-shot implementation because it uses the same absolute output index,
    /// tap order, f64 accumulation, normalization and f32 cast.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `out` is too small, the cumulative
    /// input length exceeds the implementation's `isize` indexing range, or a
    /// non-empty chunk is supplied after flush without an intervening
    /// [`reset`](Self::reset). Errors leave the stream state unchanged.
    // ZERO-ALLOC-BEGIN (#50: steady-state streaming sample-rate conversion)
    pub fn process_into(&mut self, input: &[f32], out: &mut [f32]) -> Result<usize> {
        if input.is_empty() {
            return self.flush_into(out);
        }
        if self.flushed {
            return Err(VokraError::InvalidArgument(
                "StreamingResampler::process_into: non-empty input after end-of-stream flush; \
                 call reset() before starting another stream"
                    .to_owned(),
            ));
        }

        let new_total = self.total_input.checked_add(input.len()).ok_or_else(|| {
            VokraError::InvalidArgument(
                "StreamingResampler::process_into: cumulative input length overflows usize"
                    .to_owned(),
            )
        })?;
        if new_total > isize::MAX as usize {
            return Err(VokraError::InvalidArgument(format!(
                "StreamingResampler::process_into: cumulative input length {new_total} exceeds \
                 isize::MAX"
            )));
        }

        if self.sinc.ratio == 1.0 {
            if out.len() < input.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "StreamingResampler::process_into: out.len() {} < required {} for equal-rate \
                     copy",
                    out.len(),
                    input.len(),
                )));
            }
            let new_next_output = self.next_output.checked_add(input.len()).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "StreamingResampler::process_into: output index overflows usize".to_owned(),
                )
            })?;
            out[..input.len()].copy_from_slice(input);
            self.total_input = new_total;
            self.next_output = new_next_output;
            return Ok(input.len());
        }

        let ready_end = self.ready_output_end(new_total);
        let required = ready_end - self.next_output;
        if out.len() < required {
            return Err(VokraError::InvalidArgument(format!(
                "StreamingResampler::process_into: out.len() {} < {required} newly available \
                 samples (state unchanged)",
                out.len(),
            )));
        }

        let mut written = 0usize;
        for &sample in input {
            let slot = self.total_input % self.history.len();
            self.history[slot] = sample;
            self.total_input += 1;
            while self.output_is_ready(self.next_output, self.total_input) {
                out[written] = self.interpolate(self.next_output);
                written += 1;
                self.next_output += 1;
            }
        }
        debug_assert_eq!(written, required);
        Ok(written)
    }
    // ZERO-ALLOC-END

    /// Algorithmic look-ahead latency in **output-rate samples**.
    ///
    /// This is `ceil(radius * out_rate / in_rate)`, the centered kernel's
    /// right-hand support converted from input samples to output samples.
    /// Equal-rate copy mode reports zero. At quality 5 the NanoCodec device
    /// routes report 35 samples (22.05→48 kHz), 18 (→24 kHz), and 16 (→16
    /// kHz), all far below one 80 ms / 12.5 fps codec frame.
    #[must_use]
    pub fn delay_samples(&self) -> usize {
        self.delay_samples
    }

    /// Clears history and phase so this allocation can process a new stream.
    pub fn reset(&mut self) {
        self.total_input = 0;
        self.next_output = 0;
        self.flushed = false;
    }

    fn flush_into(&mut self, out: &mut [f32]) -> Result<usize> {
        if self.total_input == 0 || self.flushed {
            return Ok(0);
        }
        let target = (self.total_input as f64 * self.sinc.ratio).round() as usize;
        debug_assert!(self.next_output <= target);
        let required = target - self.next_output;
        if out.len() < required {
            return Err(VokraError::InvalidArgument(format!(
                "StreamingResampler::process_into flush: out.len() {} < {required} tail \
                 samples (state unchanged)",
                out.len(),
            )));
        }
        for dst in &mut out[..required] {
            *dst = self.interpolate(self.next_output);
            self.next_output += 1;
        }
        self.flushed = true;
        Ok(required)
    }

    fn ready_output_end(&self, input_len: usize) -> usize {
        let mut output_index = self.next_output;
        while self.output_is_ready(output_index, input_len) {
            output_index += 1;
        }
        output_index
    }

    fn output_is_ready(&self, output_index: usize, input_len: usize) -> bool {
        let center = output_index as f64 * self.sinc.step;
        let hi = (center + self.sinc.radius).floor() as usize;
        hi < input_len
    }

    fn interpolate(&self, output_index: usize) -> f32 {
        let center = output_index as f64 * self.sinc.step;
        let lo = (center - self.sinc.radius).ceil() as isize;
        let hi = (center + self.sinc.radius).floor() as isize;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in lo..=hi {
            let h = self.sinc.kernel(center - i as f64);
            den += h;
            if i >= 0 && (i as usize) < self.total_input {
                let absolute = i as usize;
                debug_assert!(self.total_input - absolute <= self.history.len());
                num += f64::from(self.history[absolute % self.history.len()]) * h;
            }
        }
        if den != 0.0 { (num / den) as f32 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::stft;
    use vokra_core::ir::graph::StftAttrs;

    const TAU: f64 = std::f64::consts::TAU;

    fn sine(freq: f64, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|t| (TAU * freq * t as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// Dominant STFT bin of an interior frame, in Hz.
    fn dominant_freq(signal: &[f32], rate: u32, n_fft: usize) -> f64 {
        let attrs = StftAttrs::new(n_fft, n_fft / 4);
        let spec = stft(signal, &attrs).unwrap();
        let f = spec.frames / 2;
        let base = f * spec.bins;
        let (argmax, _) = (0..spec.bins)
            .map(|b| spec.re[base + b].hypot(spec.im[base + b]))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        argmax as f64 * f64::from(rate) / n_fft as f64
    }

    #[test]
    fn equal_rate_is_bit_exact_identity() {
        let x: Vec<f32> = (0..500).map(|i| (i as f32 * 0.013).sin()).collect();
        let y = resample(&x, 44_100, 44_100, 5).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn zero_rate_is_rejected() {
        assert!(matches!(
            resample(&[0.0; 4], 0, 16_000, 5),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            resample(&[0.0; 4], 16_000, 0, 5),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(resample(&[], 16_000, 8_000, 5).unwrap().is_empty());
    }

    #[test]
    fn output_length_follows_the_rounding_formula() {
        // Our defined length contract: round(in_len * out/in).
        let cases = [
            (1000usize, 16_000u32, 8_000u32, 500usize),
            (1000, 8_000, 16_000, 2000),
            (1000, 16_000, 24_000, 1500),
            (999, 3, 2, 666),            // round(999*2/3) = 666
            (1000, 44_100, 16_000, 363), // round(1000*16000/44100) = 363
        ];
        for (n, fin, fout, want) in cases {
            let y = resample(&vec![0.0f32; n], fin, fout, 5).unwrap();
            assert_eq!(y.len(), want, "{n} @ {fin}->{fout}");
        }
    }

    #[test]
    fn determinism_bit_identical() {
        let x = sine(220.0, 16_000, 4096);
        let a = resample(&x, 16_000, 22_050, 6).unwrap();
        let b = resample(&x, 16_000, 22_050, 6).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn constant_passband_gain_is_unity() {
        // A DC input reproduces itself on the interior (per-phase tap-sum
        // normalization pins the passband gain to 1.0).
        let x = vec![0.75f32; 4000];
        for &(fin, fout) in &[(16_000u32, 8_000u32), (8_000, 16_000), (16_000, 24_000)] {
            let y = resample(&x, fin, fout, 5).unwrap();
            let lo = y.len() / 4;
            let hi = 3 * y.len() / 4;
            for &v in &y[lo..hi] {
                assert!((v - 0.75).abs() < 1e-4, "{fin}->{fout}: {v}");
            }
        }
    }

    #[test]
    fn linear_ramp_is_preserved_on_the_interior() {
        // For integer up/down ratios the retained tap set is symmetric about
        // the output position, so a linear signal maps exactly (to f32) to the
        // ramp sampled at the new rate — a clean analytic oracle. Cases chosen
        // so every output phase is integer or half-integer (symmetric taps):
        // 1:2 up, 2:1 down, 3:1 down.
        let n = 600;
        let x: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
        for &(fin, fout) in &[(1u32, 2u32), (2, 1), (3, 1)] {
            let y = resample(&x, fin, fout, 5).unwrap();
            let step = f64::from(fin) / f64::from(fout);
            let radius_in = 16.0 / f64::from(fout.min(fin)) * f64::from(fin); // half/cutoff
            let guard = (radius_in / step).ceil() as usize + 2;
            let end = y.len().saturating_sub(guard);
            for (j, &yj) in y.iter().enumerate().take(end).skip(guard) {
                let want = 0.01_f64 * (j as f64 * step);
                assert!(
                    (f64::from(yj) - want).abs() < 2e-3,
                    "{fin}->{fout} j={j}: {yj} vs {want}"
                );
            }
        }
    }

    #[test]
    fn upsample_preserves_a_tone_frequency() {
        // 1 kHz sine, 16 kHz -> 48 kHz: the dominant bin stays at ~1 kHz.
        let x = sine(1000.0, 16_000, 4000);
        let y = resample(&x, 16_000, 48_000, 6).unwrap();
        let f = dominant_freq(&y, 48_000, 2048);
        assert!((f - 1000.0).abs() < 48_000.0 / 2048.0, "dominant {f} Hz");
    }

    #[test]
    fn downsample_passband_tone_survives_with_unit_gain() {
        // 500 Hz is well below the 1 kHz post-downsample Nyquist: it passes at
        // ~unit amplitude and keeps its frequency.
        let x = sine(500.0, 8_000, 8000);
        let y = resample(&x, 8_000, 2_000, 6).unwrap();
        let interior = &y[200..y.len() - 200];
        let ratio = rms(interior) / rms(&x);
        assert!((ratio - 1.0).abs() < 0.05, "passband gain {ratio}");
        let f = dominant_freq(interior, 2_000, 512);
        assert!((f - 500.0).abs() < 2_000.0 / 512.0, "dominant {f} Hz");
    }

    #[test]
    fn downsample_rejects_out_of_band_tone() {
        // A 3 kHz tone is far above the 1 kHz post-downsample Nyquist and deep
        // in the designed stopband: it is attenuated by >30 dB (the anti-alias
        // oracle the Kaiser-sinc filter is built for).
        let x = sine(3000.0, 8_000, 8000);
        let y = resample(&x, 8_000, 2_000, 6).unwrap();
        let interior = &y[200..y.len() - 200];
        let atten_db = 20.0 * (rms(&x) / rms(interior).max(1e-12)).log10();
        assert!(atten_db > 30.0, "only {atten_db} dB of attenuation");
    }

    #[test]
    fn up_then_down_roundtrip_matches_interior() {
        // A band-limited signal survives R -> 2R -> R and R -> 3R/2 -> R on the
        // interior (edges carry the filter transient and are excluded).
        let n = 4000;
        let x: Vec<f32> = (0..n)
            .map(|t| {
                let s = TAU * t as f64 / 8_000.0;
                (0.6 * (300.0 * s).sin() + 0.3 * (700.0 * s).sin()) as f32
            })
            .collect();
        for &mid in &[16_000u32, 12_000u32] {
            let up = resample(&x, 8_000, mid, 7).unwrap();
            let back = resample(&up, mid, 8_000, 7).unwrap();
            assert_eq!(back.len(), n);
            let mut max = 0.0f32;
            for i in 300..n - 300 {
                max = max.max((x[i] - back[i]).abs());
            }
            assert!(max < 2e-2, "roundtrip via {mid}: max err {max}");
        }
    }

    fn stream_in_irregular_chunks(input: &[f32], out_rate: u32) -> Vec<f32> {
        let mut stream = StreamingResampler::new(22_050, out_rate, 5).unwrap();
        let chunk_sizes = [1usize, 17, 160, 7, 441, 89, 1024, 3, 256];
        let mut output = Vec::new();
        let mut cursor = 0usize;
        let mut chunk_index = 0usize;
        while cursor < input.len() {
            let take = chunk_sizes[chunk_index % chunk_sizes.len()].min(input.len() - cursor);
            let mut scratch = vec![0.0f32; take * 3 + 256];
            let written = stream
                .process_into(&input[cursor..cursor + take], &mut scratch)
                .unwrap();
            output.extend_from_slice(&scratch[..written]);
            cursor += take;
            chunk_index += 1;
        }
        let mut tail = vec![0.0f32; stream.delay_samples() + 256];
        let written = stream.process_into(&[], &mut tail).unwrap();
        output.extend_from_slice(&tail[..written]);
        output
    }

    #[test]
    fn streaming_chunks_are_bit_identical_to_one_shot_for_nanocodec_rates() {
        let input: Vec<f32> = (0..5000)
            .map(|i| {
                let t = i as f64 / 22_050.0;
                (0.6 * (TAU * 237.0 * t).sin() + 0.25 * (TAU * 1703.0 * t).sin()) as f32
            })
            .collect();
        for out_rate in [48_000u32, 24_000, 16_000] {
            let want = resample(&input, 22_050, out_rate, 5).unwrap();
            let got = stream_in_irregular_chunks(&input, out_rate);
            assert_eq!(
                got, want,
                "chunk boundaries must not change samples for 22050 -> {out_rate}",
            );
        }
    }

    #[test]
    fn streaming_delay_is_reported_in_output_samples_and_below_one_codec_frame() {
        for (out_rate, expected_delay) in [(48_000u32, 35usize), (24_000, 18), (16_000, 16)] {
            let stream = StreamingResampler::new(22_050, out_rate, 5).unwrap();
            let one_codec_frame = out_rate as usize * 2 / 25; // 12.5 fps = 80 ms.
            assert_eq!(stream.delay_samples(), expected_delay);
            assert!(
                stream.delay_samples() <= one_codec_frame,
                "22050 -> {out_rate}: delay {} > one 80ms frame {one_codec_frame}",
                stream.delay_samples(),
            );
        }
        assert_eq!(
            StreamingResampler::new(22_050, 22_050, 5)
                .unwrap()
                .delay_samples(),
            0
        );
    }

    #[test]
    fn streaming_reset_reproduces_a_fresh_stream() {
        let input = sine(997.0, 22_050, 2048);
        let mut stream = StreamingResampler::new(22_050, 48_000, 5).unwrap();
        let mut first = vec![0.0f32; 5000];
        let n_first = stream.process_into(&input, &mut first).unwrap();
        let n_tail = stream.process_into(&[], &mut first[n_first..]).unwrap();
        let first = first[..n_first + n_tail].to_vec();

        stream.reset();
        let mut second = vec![0.0f32; 5000];
        let n_second = stream.process_into(&input, &mut second).unwrap();
        let n_tail = stream.process_into(&[], &mut second[n_second..]).unwrap();
        assert_eq!(first, second[..n_second + n_tail]);
    }

    #[test]
    fn streaming_buffer_errors_do_not_advance_state() {
        let input = sine(440.0, 22_050, 256);
        let mut stream = StreamingResampler::new(22_050, 48_000, 5).unwrap();
        assert!(matches!(
            stream.process_into(&input, &mut []),
            Err(VokraError::InvalidArgument(_)),
        ));

        let mut got = vec![0.0f32; 1024];
        let n = stream.process_into(&input, &mut got).unwrap();
        let mut fresh = StreamingResampler::new(22_050, 48_000, 5).unwrap();
        let mut want = vec![0.0f32; 1024];
        let want_n = fresh.process_into(&input, &mut want).unwrap();
        assert_eq!(n, want_n);
        assert_eq!(got[..n], want[..want_n]);

        let mut too_short_tail = [];
        assert!(matches!(
            stream.process_into(&[], &mut too_short_tail),
            Err(VokraError::InvalidArgument(_)),
        ));
        let mut got_tail = [0.0f32; 256];
        let got_tail_n = stream.process_into(&[], &mut got_tail).unwrap();
        let mut want_tail = [0.0f32; 256];
        let want_tail_n = fresh.process_into(&[], &mut want_tail).unwrap();
        assert_eq!(got_tail_n, want_tail_n);
        assert_eq!(got_tail[..got_tail_n], want_tail[..want_tail_n]);
    }

    #[test]
    fn streaming_equal_rate_is_bit_exact_and_rejects_short_output() {
        let input = sine(440.0, 22_050, 256);
        let mut stream = StreamingResampler::new(22_050, 22_050, 5).unwrap();
        assert!(matches!(
            stream.process_into(&input, &mut [0.0; 255]),
            Err(VokraError::InvalidArgument(_)),
        ));
        let mut output = [0.0f32; 256];
        let written = stream.process_into(&input, &mut output).unwrap();
        assert_eq!(written, input.len());
        assert_eq!(output.as_slice(), input.as_slice());
        assert_eq!(stream.process_into(&[], &mut output).unwrap(), 0);
    }

    #[test]
    fn streaming_zero_rate_is_rejected() {
        assert!(matches!(
            StreamingResampler::new(0, 16_000, 5),
            Err(VokraError::InvalidArgument(_)),
        ));
        assert!(matches!(
            StreamingResampler::new(16_000, 0, 5),
            Err(VokraError::InvalidArgument(_)),
        ));
    }

    #[test]
    fn streaming_rejects_more_input_after_flush_until_reset() {
        let mut stream = StreamingResampler::new(22_050, 16_000, 5).unwrap();
        let mut out = [0.0f32; 256];
        stream.process_into(&[0.25; 128], &mut out).unwrap();
        stream.process_into(&[], &mut out).unwrap();
        assert!(matches!(
            stream.process_into(&[0.5], &mut out),
            Err(VokraError::InvalidArgument(_)),
        ));
        stream.reset();
        assert!(stream.process_into(&[0.5], &mut out).is_ok());
    }

    #[test]
    fn quality_table_is_monotonic_and_saturates() {
        let mut prev = (0usize, 0.0f64);
        for q in 0u8..=9 {
            let (half, beta) = quality_params(q);
            assert!(half >= prev.0 && beta >= prev.1, "q={q} not monotonic");
            prev = (half, beta);
        }
        // Values past the table saturate to the top row.
        assert_eq!(quality_params(10), quality_params(255));
    }
}
