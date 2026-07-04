//! Streaming inverse STFT via chunked weighted overlap-add (M2-05; FR-OP-02,
//! FR-ST-05).
//!
//! The chunked, tail-buffering counterpart of [`crate::istft`]. Where the batch
//! op accumulates *every* frame into one contiguous overlap-add buffer and then
//! trims/normalizes, the streaming op ingests frames a chunk at a time,
//! **flushes every sample whose overlap-add is complete**, and carries the
//! not-yet-final overlap tail (plus the running window-sum-of-squares tail) into
//! the next chunk. The design contract is exact: for the same spectrogram and
//! attributes, the concatenation of every [`IstftStreamingState::push`] output
//! followed by [`IstftStreamingState::finish`] is **bit-for-bit** equal to a
//! single [`crate::istft`] call — regardless of how the frames are split into
//! chunks (per-chunk / per-frame invariance). That is the essence of FR-OP-02's
//! "per-layer state carry-over": the tail state is the only thing that crosses a
//! chunk boundary, and it crosses losslessly.
//!
//! # Why bit-exact, and where the essence lives
//!
//! STFT ≠ FFT (CLAUDE.md pitfall): the hard part of a *streaming* iSTFT is not
//! the transform but keeping the **overlap-add and window-compensation
//! continuous across chunk boundaries**. A sample at absolute index `i` receives
//! contributions from every frame `f` with `f·hop ≤ i < f·hop + n_fft`, added in
//! increasing `f`. This op:
//!
//! - processes frames in order and adds each frame's windowed contribution in
//!   the identical order the batch op uses (so the floating-point running sums
//!   are the same bits), reusing the batch op's exact per-frame inverse
//!   ([`crate::istft::FrameInverter`]), synthesis window
//!   ([`crate::istft::build_synth_window`]) and NOLA guard
//!   ([`crate::istft::NOLA_EPS`]);
//! - only *emits* a sample once **no future frame can touch it** (`i <
//!   frames_done · hop`) and it is **outside the final `center` tail-trim zone**
//!   (held back by `n_fft/2`), so a flushed sample already has its final,
//!   fully-accumulated value.
//!
//! # FR-ST-05 (state hiding)
//!
//! All carry-over state — the overlap-add tail, the window-sum-of-squares tail
//! and the position counters — is private to [`IstftStreamingState`]. A caller
//! only ever pushes spectral chunks and receives waveform samples; it never
//! names a tensor or manages a buffer (the same discipline the M0-05 `VadStream`
//! applies to its hidden LSTM `h`/`c`). See the module-level note in
//! `vokra_core::stream` on stream-handle ownership: a future audio-carrying
//! stream transport embeds this state verbatim (its `reset` already reproduces
//! the first run bit-for-bit, the stream-handle `reset` contract).
//!
//! GPU FFT lowering (cuFFT / MPS FFT, the GPU side of FR-OP-05) is out of scope
//! here — the overlap-add / tail carry-over is host control flow and the inner
//! `c2r` reuses the M0-04 pocketfft (BSD-3) Rust port; no FFTW3 / soxr / GPL.

use vokra_core::ir::graph::IstftStreamingAttrs;
use vokra_core::{Result, VokraError};

use crate::Spectrogram;
use crate::fft::{FftPlan, RealFftPlan};
use crate::istft::{FrameInverter, NOLA_EPS, build_synth_window, frame_unscale};

/// Streaming inverse-STFT state (FR-OP-02 / FR-ST-05).
///
/// Drive it with [`push`](Self::push) (feed a chunk of spectral frames, receive
/// the samples finalized by that chunk) and [`finish`](Self::finish) (flush the
/// final tail once no more frames will arrive). [`reset`](Self::reset) returns it
/// to the initial state so a fresh run reproduces the first bit-for-bit.
///
/// Every field is private: the carry-over tail is never exposed as a tensor
/// (FR-ST-05). Construct it with [`IstftStreamingState::new`].
pub struct IstftStreamingState {
    // ---- immutable configuration (mirrors the batch `istft` parameters) ----
    n: usize,
    hop: usize,
    /// Expected bins per input frame: `n/2+1` (real) or `n` (complex).
    bins: usize,
    /// Samples trimmed off the front of the reconstruction (`n/2` when
    /// `center`, else 0) — the first output sample is recon index `head_trim`.
    head_trim: usize,
    /// Samples trimmed off the tail of the reconstruction (`n/2` when `center`,
    /// else 0) — held back during streaming since the final length is unknown
    /// until `finish`.
    tail_trim: usize,
    /// Optional target output length (batch `istft` `length` override).
    length: Option<usize>,
    /// Synthesis window, length `n` (identical to the batch op's).
    synth_window: Vec<f32>,
    /// Precomputed `synth_window[i]²` — the per-frame window-sum-of-squares
    /// increment (the exact product the batch op adds each frame).
    wsq: Vec<f32>,
    /// Forward-normalization undo factor (see [`frame_unscale`]).
    unscale: f32,
    /// Real (RFFT) inverse plan for the `real_input` path.
    real_plan: Option<RealFftPlan>,
    /// Full complex inverse plan for the non-`real_input` path.
    complex_plan: Option<FftPlan>,

    // ---- running carry-over state (the hidden tail) ------------------------
    /// Overlap-add accumulator, absolute recon indices `[buf_start, buf_start +
    /// acc.len())`.
    acc: Vec<f32>,
    /// Window-sum-of-squares accumulator, same index span as `acc`.
    wss: Vec<f32>,
    /// Absolute recon index of `acc[0]` (everything below is consumed).
    buf_start: usize,
    /// Number of frames ingested so far.
    frames_done: usize,
    /// Next absolute recon index to emit (starts at `head_trim`).
    emit_ptr: usize,
    /// Output samples emitted so far (post-trim, post-`length`-clamp).
    out_emitted: usize,
    /// Set once [`finish`](Self::finish) has flushed the final tail.
    finished: bool,
}

impl IstftStreamingState {
    /// Builds streaming state for `attrs`.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] on a zero `n_fft` / `hop_length`,
    /// an out-of-range `win_length`, or a `tail_len` smaller than the
    /// inter-frame overlap `n_fft − hop_length` (FR-OP-02: the tail buffer must
    /// be able to hold the overlap it carries across chunks).
    pub fn new(attrs: &IstftStreamingAttrs) -> Result<Self> {
        let ia = &attrs.istft;
        if ia.n_fft == 0 || ia.hop_length == 0 {
            return Err(VokraError::InvalidArgument(
                "istft_streaming: n_fft and hop_length must be non-zero".to_owned(),
            ));
        }
        if ia.win_length == 0 || ia.win_length > ia.n_fft {
            return Err(VokraError::InvalidArgument(
                "istft_streaming: win_length must be in 1..=n_fft".to_owned(),
            ));
        }
        let n = ia.n_fft;
        let hop = ia.hop_length;
        let overlap = n.saturating_sub(hop);
        if attrs.tail_len < overlap {
            return Err(VokraError::InvalidArgument(format!(
                "istft_streaming: tail_len {} is below the inter-frame overlap n_fft-hop_length \
                 = {overlap}; the overlap cannot be carried across chunks",
                attrs.tail_len
            )));
        }

        let synth_window = build_synth_window(ia);
        let wsq: Vec<f32> = synth_window.iter().map(|w| w * w).collect();
        let unscale = frame_unscale(ia.normalization, n);
        let real_plan = ia.real_input.then(|| RealFftPlan::new(n));
        let complex_plan = (!ia.real_input).then(|| FftPlan::new(n));
        let bins = if ia.real_input { n / 2 + 1 } else { n };
        let trim = if ia.center { n / 2 } else { 0 };

        // Steady-state buffer length is bounded by max(n-hop, tail_trim) + hop;
        // reserve against `tail_len` (which the caller may set above the
        // minimum) plus a frame so per-chunk pushes never reallocate the tail
        // buffers (FR-EX-05, hot-path allocation stability).
        let cap = attrs.tail_len.max(n).saturating_add(hop).saturating_add(1);

        Ok(Self {
            n,
            hop,
            bins,
            head_trim: trim,
            tail_trim: trim,
            length: ia.length,
            synth_window,
            wsq,
            unscale,
            real_plan,
            complex_plan,
            acc: Vec::with_capacity(cap),
            wss: Vec::with_capacity(cap),
            buf_start: 0,
            frames_done: 0,
            emit_ptr: trim,
            out_emitted: 0,
            finished: false,
        })
    }

    /// The number of frequency bins each pushed frame must carry
    /// (`n_fft/2+1` for the real path, `n_fft` for the complex path).
    pub fn expected_bins(&self) -> usize {
        self.bins
    }

    /// Total output samples emitted so far across all `push`/`finish` calls.
    pub fn emitted(&self) -> usize {
        self.out_emitted
    }

    /// Feeds one chunk of spectral frames and returns the waveform samples the
    /// chunk *finalized* (an empty vector when the chunk only advanced the
    /// overlap tail). Frames are `[frames, bins]` in `chunk` and consumed in
    /// order; the not-yet-final overlap is carried into the next call.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if [`finish`](Self::finish) has already
    /// run, if `chunk.bins` differs from [`expected_bins`](Self::expected_bins),
    /// or if `chunk`'s `re`/`im` length is not `frames · bins`.
    pub fn push(&mut self, chunk: &Spectrogram) -> Result<Vec<f32>> {
        if self.finished {
            return Err(VokraError::InvalidArgument(
                "istft_streaming: push after finish; call reset to start a new stream".to_owned(),
            ));
        }
        if chunk.bins != self.bins {
            return Err(VokraError::InvalidArgument(format!(
                "istft_streaming: chunk has {} bins, expected {}",
                chunk.bins, self.bins
            )));
        }
        let expected = chunk.frames.saturating_mul(chunk.bins);
        if chunk.re.len() != expected || chunk.im.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "istft_streaming: chunk re/im length must be frames·bins = {expected}"
            )));
        }
        let mut out = Vec::new();
        for f_local in 0..chunk.frames {
            self.accumulate_frame(&chunk.re, &chunk.im, f_local * self.bins);
            let boundary = self.push_boundary();
            self.emit_range(boundary, &mut out);
        }
        Ok(out)
    }

    /// Flushes the final overlap tail once no more frames will arrive, returning
    /// the remaining samples (with the `center` tail trim applied and the
    /// `length` override honored, so the total emitted length matches the batch
    /// `istft` output exactly). Idempotent: a second call returns an empty
    /// vector. Use [`reset`](Self::reset) to start another stream.
    pub fn finish(&mut self) -> Vec<f32> {
        if self.finished {
            return Vec::new();
        }
        let mut out = Vec::new();
        if self.frames_done > 0 {
            // No more frames: every buffered sample is final; emit up to the
            // center tail-trim boundary.
            let buffer_end = self.buf_start + self.acc.len();
            let boundary = buffer_end.saturating_sub(self.tail_trim);
            self.emit_range(boundary, &mut out);
        }
        // `length` override: zero-pad the tail so the total equals the target
        // (mirrors the batch op's `out.resize(len, 0.0)` when len > natural).
        if let Some(len) = self.length {
            if self.out_emitted < len {
                out.resize(out.len() + (len - self.out_emitted), 0.0);
                self.out_emitted = len;
            }
        }
        self.finished = true;
        out
    }

    /// Clears all carry-over state, returning the stepper to its initial state
    /// so a fresh run reproduces the first run bit-for-bit (the stream-handle
    /// `reset` contract, FR-ST-05). Retains the buffer capacity.
    pub fn reset(&mut self) {
        self.acc.clear();
        self.wss.clear();
        self.buf_start = 0;
        self.frames_done = 0;
        self.emit_ptr = self.head_trim;
        self.out_emitted = 0;
        self.finished = false;
    }

    /// Inverts one frame (bins at `re[base]` / `im[base]`) into its length-`n`
    /// time-domain frame, using the exact shared inverse of the batch op.
    fn invert_frame_at(&self, re: &[f32], im: &[f32], base: usize) -> Vec<f32> {
        let inverter = FrameInverter {
            n: self.n,
            unscale: self.unscale,
            real_plan: self.real_plan.as_ref(),
            complex_plan: self.complex_plan.as_ref(),
        };
        inverter.invert(re, im, base, self.bins)
    }

    /// Overlap-adds one frame's windowed contribution into the tail buffers at
    /// its absolute position, zero-extending the buffers (leaving zeros in any
    /// hop-gap when `hop > n_fft`) as needed.
    fn accumulate_frame(&mut self, re: &[f32], im: &[f32], base: usize) {
        let frame_time = self.invert_frame_at(re, im, base);
        let start = self.frames_done * self.hop;
        let needed_end = start + self.n;
        let cur_end = self.buf_start + self.acc.len();
        if needed_end > cur_end {
            let add = needed_end - cur_end;
            self.acc.resize(self.acc.len() + add, 0.0);
            self.wss.resize(self.wss.len() + add, 0.0);
        }
        // start >= buf_start always (a future frame never precedes the consumed
        // front — its start `frames_done·hop` is >= the last emit boundary).
        let off = start - self.buf_start;
        // Identical add order and operands to the batch op ⇒ identical bits
        // (`frame_time.len() == n`, so `i` also indexes the window / wss buffers).
        for (i, &ft) in frame_time.iter().enumerate() {
            self.acc[off + i] += ft * self.synth_window[i];
            self.wss[off + i] += self.wsq[i];
        }
        self.frames_done += 1;
    }

    /// The absolute recon index up to which a *mid-stream* flush is safe:
    /// finalized by overlap (`< frames_done·hop`, no future frame touches it)
    /// and outside the yet-unknown final tail-trim zone (held back by
    /// `tail_trim` from the current buffer end).
    fn push_boundary(&self) -> usize {
        let buffer_end = self.buf_start + self.acc.len();
        let ready = self.frames_done * self.hop;
        ready.min(buffer_end.saturating_sub(self.tail_trim))
    }

    /// Emits recon samples `[emit_ptr, boundary)` (NOLA-guarded division,
    /// `length` clamp), then drops the consumed front of the tail buffers.
    fn emit_range(&mut self, boundary: usize, out: &mut Vec<f32>) {
        while self.emit_ptr < boundary {
            if let Some(len) = self.length {
                if self.out_emitted >= len {
                    break;
                }
            }
            let k = self.emit_ptr - self.buf_start;
            let a = self.acc[k];
            let w = self.wss[k];
            // Matches the batch op: divide only where the window overlap is
            // non-negligible, else leave the raw sum (which is ~0 there).
            out.push(if w > NOLA_EPS { a / w } else { a });
            self.out_emitted += 1;
            self.emit_ptr += 1;
        }
        // If the `length` clamp cut the loop short, the remaining
        // `[emit_ptr, boundary)` are samples the batch op truncates away — skip
        // past them so the pointers stay aligned and the tail buffer still
        // drains. (No effect when `boundary <= emit_ptr`, e.g. still inside the
        // head trim.)
        if self.length.is_some() && self.emit_ptr < boundary {
            self.emit_ptr = boundary;
        }
        if boundary > self.buf_start {
            let drop = boundary - self.buf_start;
            self.acc.drain(0..drop);
            self.wss.drain(0..drop);
            self.buf_start = boundary;
        }
    }
}

/// One-shot streaming reconstruction: `new` → `push(all frames)` → `finish`.
///
/// Equivalent to [`crate::istft`] with `attrs.istft` — the degenerate
/// single-chunk case of the streaming op, used by the IR dispatch (a graph node
/// evaluates the whole spectrogram at once, its tail state never a graph tensor)
/// and as a parity oracle.
///
/// # Errors
///
/// Propagates [`IstftStreamingState::new`] / [`IstftStreamingState::push`]
/// errors (bad sizes, insufficient `tail_len`, bin-count mismatch).
pub fn istft_streaming_oneshot(
    spectrogram: &Spectrogram,
    attrs: &IstftStreamingAttrs,
) -> Result<Vec<f32>> {
    let mut state = IstftStreamingState::new(attrs)?;
    let mut out = state.push(spectrogram)?;
    out.extend(state.finish());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::istft::istft;
    use crate::stft::stft;
    use vokra_core::ir::graph::{IstftAttrs, Normalization, StftAttrs, Window, WindowSymmetry};

    /// Builds a deterministic test spectrogram by analysing a synthetic signal
    /// (an internal oracle — no external fixtures / PyTorch needed).
    fn spec_for(
        n_fft: usize,
        hop: usize,
        real_input: bool,
        window: Window,
        center: bool,
    ) -> Spectrogram {
        let signal: Vec<f32> = (0..4000)
            .map(|t| {
                let t = t as f32;
                (t * 0.02).sin() + 0.3 * (t * 0.113).cos() + 0.05 * (t * 0.31).sin()
            })
            .collect();
        let mut sa = StftAttrs::new(n_fft, hop);
        sa.real_input = real_input;
        sa.window = window;
        sa.center = center;
        stft(&signal, &sa).unwrap()
    }

    /// The batch iSTFT this op must reproduce bit-for-bit.
    fn batch(spec: &Spectrogram, ia: &IstftAttrs) -> Vec<f32> {
        istft(spec, ia).unwrap()
    }

    /// Streams `spec` through the given chunk sizes (in frames) and concatenates
    /// push outputs + the final flush.
    fn stream_chunks(
        spec: &Spectrogram,
        attrs: &IstftStreamingAttrs,
        chunk_sizes: &[usize],
    ) -> Vec<f32> {
        let mut state = IstftStreamingState::new(attrs).unwrap();
        let mut out = Vec::new();
        let mut f = 0usize;
        let mut ci = 0usize;
        while f < spec.frames {
            let take = if chunk_sizes.is_empty() {
                spec.frames - f
            } else {
                chunk_sizes[ci % chunk_sizes.len()]
                    .max(1)
                    .min(spec.frames - f)
            };
            ci += 1;
            let sub = slice_frames(spec, f, take);
            out.extend(state.push(&sub).unwrap());
            f += take;
        }
        out.extend(state.finish());
        out
    }

    /// Extracts `count` frames starting at `start` as a standalone spectrogram.
    fn slice_frames(spec: &Spectrogram, start: usize, count: usize) -> Spectrogram {
        let b = spec.bins;
        Spectrogram {
            frames: count,
            bins: b,
            re: spec.re[start * b..(start + count) * b].to_vec(),
            im: spec.im[start * b..(start + count) * b].to_vec(),
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn oneshot_matches_batch_bit_exact() {
        // A single push covering all frames is the streaming op's degenerate
        // case and must equal the batch op exactly (identical bits).
        let spec = spec_for(512, 128, true, Window::Hann, true);
        let mut ia = IstftAttrs::new(512, 128);
        ia.length = Some(4000);
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());
        let streamed = istft_streaming_oneshot(&spec, &attrs).unwrap();
        let expect = batch(&spec, &ia);
        assert_eq!(
            streamed, expect,
            "one-shot streaming must equal batch bit-for-bit"
        );
    }

    #[test]
    fn streaming_equals_batch_across_chunk_splits() {
        // THE WP oracle: however the frames are split into chunks, the streamed
        // reconstruction equals the batch istft (bit-for-bit — center trim and
        // tail flush included).
        let spec = spec_for(512, 128, true, Window::Hann, true);
        let ia = IstftAttrs::new(512, 128);
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());
        let expect = batch(&spec, &ia);

        for pattern in [
            &[1usize][..],     // one frame at a time
            &[2, 3, 5, 7][..], // uneven
            &[4][..],          // fixed
            &[13, 1, 8][..],   // irregular
            &[][..],           // all frames in one chunk
        ] {
            let streamed = stream_chunks(&spec, &attrs, pattern);
            let d = max_abs_diff(&streamed, &expect);
            assert_eq!(
                d, 0.0,
                "chunk pattern {pattern:?}: max|Δ| = {d} (must be exactly 0)"
            );
        }
    }

    #[test]
    fn chunk_splitting_is_invariant() {
        // Two different chunkings of the same spectrogram yield identical output.
        let spec = spec_for(400, 160, true, Window::Hamming, true);
        let attrs = IstftStreamingAttrs::new(400, 160);
        let a = stream_chunks(&spec, &attrs, &[1]);
        let b = stream_chunks(&spec, &attrs, &[7, 2, 11]);
        let c = stream_chunks(&spec, &attrs, &[]);
        assert_eq!(a, b, "per-frame vs irregular chunking diverged");
        assert_eq!(a, c, "per-frame vs single-chunk diverged");
    }

    #[test]
    fn matches_batch_over_windows_center_realinput_norm() {
        // Sweep the attribute space the batch op covers; streaming must track it
        // exactly in every combination.
        let windows = [
            Window::Hann,
            Window::Hamming,
            Window::BlackmanHarris,
            Window::Kaiser { beta: 8.0 },
        ];
        for &window in &windows {
            for &center in &[true, false] {
                for &real_input in &[true, false] {
                    for &norm in &[
                        Normalization::Backward,
                        Normalization::Forward,
                        Normalization::Ortho,
                    ] {
                        let spec = spec_for(256, 64, real_input, window, center);
                        let mut ia = IstftAttrs::new(256, 64);
                        ia.window = window;
                        ia.center = center;
                        ia.real_input = real_input;
                        ia.normalization = norm;
                        let attrs = IstftStreamingAttrs::from_istft(ia.clone());
                        let expect = batch(&spec, &ia);
                        let streamed = stream_chunks(&spec, &attrs, &[3, 1, 5]);
                        let d = max_abs_diff(&streamed, &expect);
                        assert_eq!(
                            d, 0.0,
                            "window={window:?} center={center} real={real_input} norm={norm:?}: max|Δ|={d}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn length_override_shorter_and_longer_match_batch() {
        let spec = spec_for(512, 128, true, Window::Hann, true);
        let natural = {
            let ia = IstftAttrs::new(512, 128);
            batch(&spec, &ia).len()
        };
        for len in [natural / 2, natural, natural + 500] {
            let mut ia = IstftAttrs::new(512, 128);
            ia.length = Some(len);
            let attrs = IstftStreamingAttrs::from_istft(ia.clone());
            let expect = batch(&spec, &ia);
            assert_eq!(expect.len(), len);
            let streamed = stream_chunks(&spec, &attrs, &[3, 1, 6]);
            assert_eq!(streamed.len(), len, "length={len} output length");
            let d = max_abs_diff(&streamed, &expect);
            assert_eq!(d, 0.0, "length={len}: max|Δ|={d}");
        }
    }

    #[test]
    fn win_length_shorter_than_n_fft_matches_batch() {
        let mut sa = StftAttrs::new(512, 128);
        sa.win_length = 400;
        let signal: Vec<f32> = (0..4000).map(|t| (t as f32 * 0.03).sin()).collect();
        let spec = stft(&signal, &sa).unwrap();
        let mut ia = IstftAttrs::new(512, 128);
        ia.win_length = 400;
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());
        let expect = batch(&spec, &ia);
        let streamed = stream_chunks(&spec, &attrs, &[2, 5, 1]);
        assert_eq!(max_abs_diff(&streamed, &expect), 0.0);
    }

    #[test]
    fn nola_violating_hop_matches_batch_and_stays_finite() {
        // hop == n_fft: no overlap, periodic Hann w[0]=0 zeros the frame
        // boundaries. tail_len defaults to 0 (n-hop). Streaming must reproduce
        // the batch op's zeroed boundaries exactly and never emit NaN/Inf.
        let n = 256;
        let mut sa = StftAttrs::new(n, n);
        sa.center = false;
        let signal: Vec<f32> = (0..4 * n).map(|t| (t as f32 * 0.021).sin()).collect();
        let spec = stft(&signal, &sa).unwrap();
        let mut ia = IstftAttrs::new(n, n);
        ia.center = false;
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());
        assert_eq!(attrs.tail_len, 0, "hop==n_fft ⇒ zero overlap tail");
        let expect = batch(&spec, &ia);
        let streamed = stream_chunks(&spec, &attrs, &[1, 3]);
        assert_eq!(max_abs_diff(&streamed, &expect), 0.0);
        assert!(streamed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hop_larger_than_n_fft_leaves_gaps_like_batch() {
        // hop > n_fft ⇒ frames don't tile: batch leaves zero gaps (wss==0).
        // Streaming must zero-fill the same gaps.
        let n = 128;
        let hop = 200;
        let mut sa = StftAttrs::new(n, hop);
        sa.center = false;
        let signal: Vec<f32> = (0..2000).map(|t| (t as f32 * 0.05).sin()).collect();
        let spec = stft(&signal, &sa).unwrap();
        let mut ia = IstftAttrs::new(n, hop);
        ia.center = false;
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());
        let expect = batch(&spec, &ia);
        let streamed = stream_chunks(&spec, &attrs, &[1, 2]);
        assert_eq!(max_abs_diff(&streamed, &expect), 0.0);
    }

    #[test]
    fn reset_reproduces_first_run() {
        // The stream-handle reset contract (FR-ST-05): a fresh run after reset
        // reproduces the first run bit-for-bit.
        let spec = spec_for(256, 64, true, Window::Hann, true);
        let attrs = IstftStreamingAttrs::new(256, 64);
        let mut state = IstftStreamingState::new(&attrs).unwrap();
        let first = {
            let mut o = state.push(&spec).unwrap();
            o.extend(state.finish());
            o
        };
        state.reset();
        let second = {
            let mut o = state.push(&spec).unwrap();
            o.extend(state.finish());
            o
        };
        assert_eq!(first, second, "reset must rewind to a bit-identical run");
    }

    #[test]
    fn empty_and_single_frame_chunks_are_safe() {
        let spec = spec_for(256, 64, true, Window::Hann, true);
        let attrs = IstftStreamingAttrs::new(256, 64);
        let ia = IstftAttrs::new(256, 64);
        let expect = batch(&spec, &ia);

        let mut state = IstftStreamingState::new(&attrs).unwrap();
        let mut out = Vec::new();
        // An empty chunk (0 frames) must not panic and must emit nothing.
        let empty = Spectrogram {
            frames: 0,
            bins: attrs.istft.n_fft / 2 + 1,
            re: vec![],
            im: vec![],
        };
        assert!(state.push(&empty).unwrap().is_empty());
        // Interleave empty chunks between single frames.
        for f in 0..spec.frames {
            out.extend(state.push(&slice_frames(&spec, f, 1)).unwrap());
            assert!(state.push(&empty).unwrap().is_empty());
        }
        out.extend(state.finish());
        assert_eq!(max_abs_diff(&out, &expect), 0.0);
    }

    #[test]
    fn tail_len_below_overlap_is_rejected_and_at_or_above_is_ok() {
        // FR-OP-02: tail_len must hold the inter-frame overlap n_fft-hop_length.
        let n = 512;
        let hop = 128;
        let overlap = n - hop; // 384
        let ia = IstftAttrs::new(n, hop);

        // Below the overlap ⇒ explicit error (never a silent bad reconstruction).
        let bad = IstftStreamingAttrs {
            istft: ia.clone(),
            tail_len: overlap - 1,
        };
        assert!(matches!(
            IstftStreamingState::new(&bad),
            Err(VokraError::InvalidArgument(_))
        ));

        // Exactly the overlap (the default) and a larger tail both work and stay
        // bit-exact vs batch.
        let spec = spec_for(n, hop, true, Window::Hann, true);
        let expect = batch(&spec, &ia);
        for tail_len in [overlap, overlap + 200, n] {
            let attrs = IstftStreamingAttrs {
                istft: ia.clone(),
                tail_len,
            };
            let streamed = stream_chunks(&spec, &attrs, &[3, 1]);
            assert_eq!(
                max_abs_diff(&streamed, &expect),
                0.0,
                "tail_len={tail_len} must stay bit-exact"
            );
        }
    }

    #[test]
    fn push_after_finish_errors_until_reset() {
        let spec = spec_for(256, 64, true, Window::Hann, true);
        let attrs = IstftStreamingAttrs::new(256, 64);
        let mut state = IstftStreamingState::new(&attrs).unwrap();
        let _ = state.push(&spec).unwrap();
        let _ = state.finish();
        assert!(matches!(
            state.push(&spec),
            Err(VokraError::InvalidArgument(_))
        ));
        // A second finish is a harmless no-op.
        assert!(state.finish().is_empty());
        // reset re-enables pushing.
        state.reset();
        assert!(state.push(&spec).is_ok());
    }

    #[test]
    fn buffer_capacity_is_stable_across_chunks() {
        // FR-EX-05 (hot-path allocation stability): the tail buffers are
        // reserved once in `new` and never reallocate in steady state. This is
        // the authoritative zero-alloc proof for the streaming state buffers
        // (the per-frame inverse FFT allocates exactly as the batch op does).
        let spec = spec_for(1024, 256, true, Window::Hann, true);
        let attrs = IstftStreamingAttrs::new(1024, 256);
        let mut state = IstftStreamingState::new(&attrs).unwrap();

        // Warm up past the first buffer growth.
        let mut f = 0;
        let warm = 3.min(spec.frames);
        while f < warm {
            let _ = state.push(&slice_frames(&spec, f, 1)).unwrap();
            f += 1;
        }
        let (acc_cap, wss_cap) = (state.acc.capacity(), state.wss.capacity());

        // Push the remaining frames one at a time; capacity must not grow.
        while f < spec.frames {
            let _ = state.push(&slice_frames(&spec, f, 1)).unwrap();
            assert_eq!(state.acc.capacity(), acc_cap, "acc buffer reallocated");
            assert_eq!(state.wss.capacity(), wss_cap, "wss buffer reallocated");
            f += 1;
        }
        let _ = state.finish();
        assert_eq!(state.acc.capacity(), acc_cap);
    }

    #[test]
    fn rejects_wrong_bin_count() {
        let attrs = IstftStreamingAttrs::new(256, 64); // expects 129 bins (real)
        let mut state = IstftStreamingState::new(&attrs).unwrap();
        let wrong = Spectrogram {
            frames: 1,
            bins: 99,
            re: vec![0.0; 99],
            im: vec![0.0; 99],
        };
        assert!(matches!(
            state.push(&wrong),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn hidden_state_api_needs_no_tensor_names() {
        // FR-ST-05 in practice: a full stream driven only by push/finish/reset,
        // touching no internal field or tensor name, still matches batch. Uses
        // the symmetric-window analysis path for extra coverage.
        let mut sa = StftAttrs::new(320, 80);
        sa.window_symmetry = WindowSymmetry::Symmetric;
        let signal: Vec<f32> = (0..3000).map(|t| (t as f32 * 0.04).cos()).collect();
        let spec = stft(&signal, &sa).unwrap();
        let mut ia = IstftAttrs::new(320, 80);
        ia.window_symmetry = WindowSymmetry::Symmetric;
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());

        let mut state = IstftStreamingState::new(&attrs).unwrap();
        assert_eq!(state.expected_bins(), 161);
        let mut out = Vec::new();
        for f in 0..spec.frames {
            out.extend(state.push(&slice_frames(&spec, f, 1)).unwrap());
        }
        out.extend(state.finish());
        assert_eq!(state.emitted(), out.len());
        assert_eq!(max_abs_diff(&out, &batch(&spec, &ia)), 0.0);
    }
}
