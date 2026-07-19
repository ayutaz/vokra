//! `VadStream` — the handle that hides all recurrent state (M0-05-T08).
//!
//! Implements [`vokra_core::engines::VadStreamHandle`]. Per research 03 §3.1 and
//! FR-LD-06, the caller only pushes PCM and reads probabilities; the LSTM
//! `h`/`c`, the rolling audio context, the fixed-frame buffering and the
//! pseudo-STFT are all private here — none is exposed as a public field. A
//! stream is single-rate: the sample rate is fixed by the first `push_pcm` and
//! a later change is rejected.
//!
//! Framing matches the **official** upstream usage (`utils_vad.py
//! OnnxWrapper`): non-overlapping fixed frames of 512 samples @ 16 kHz / 256
//! @ 8 kHz, each prefixed with a rolling [`SampleRate::context_len`]-sample
//! audio context — the previous frame's tail, zeros before the first frame —
//! so the subgraph sees 576 / 288 samples per step; the LSTM state is carried
//! across frames and a trailing partial frame is buffered until the next
//! push. The context resets together with the state ([`VadStreamHandle::reset`]),
//! as upstream `reset_states` does. Without the context the probabilities
//! collapse on real speech — max prob 0.0037 on `jfk-30s.wav` vs 0.9999 with
//! it (2026-07-16 real-weight eval P1) — which is why [`ContextMode::Raw`]
//! (the bare 1:1 graph interface, no audio context) exists only for parity
//! against the bare-frame fixtures, not as a public mode.
//!
//! `stream.poll(events)` + lock-free ring buffer (FR-ST-02) and generalised
//! streaming state (FR-ST-05) are M1; M0 provides this minimal synchronous
//! handle only (see the ticket scope note).

use std::sync::Arc;

use vokra_core::engines::VadStreamHandle;
use vokra_core::{Result, VokraError};

use super::lstm::LstmState;
use super::weights::SileroWeights;
use super::{SampleRate, run_frame};

/// How PCM frames are presented to the subgraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContextMode {
    /// Official wrapper semantics (the default): prepend the rolling
    /// [`SampleRate::context_len`]-sample context to every frame.
    Official,
    /// Raw 1:1 ONNX graph interface: bare fixed frames, no audio context.
    /// Parity use only (`SileroVadV5::open_raw_stream`, test-gated) — never
    /// constructed in production builds, hence dead code outside tests.
    #[cfg_attr(not(test), allow(dead_code))]
    Raw,
}

/// A stateful, single-rate VAD stream over a shared model.
pub(super) struct VadStream {
    weights: Arc<SileroWeights>,
    mode: ContextMode,
    /// Fixed once the first sample arrives; `None` until then.
    rate: Option<SampleRate>,
    state: LstmState,
    /// Official-mode rolling context: the previous frame's last
    /// [`SampleRate::context_len`] samples. Empty = fresh (zero context);
    /// sized lazily once the rate is known. Always empty in raw mode.
    context: Vec<f32>,
    /// Samples not yet forming a complete frame.
    pending: Vec<f32>,
}

impl VadStream {
    /// Creates a fresh official-mode stream (zeroed state, zero context) over
    /// the model's weights.
    pub(super) fn new(weights: Arc<SileroWeights>) -> Self {
        Self::with_mode(weights, ContextMode::Official)
    }

    /// Creates a fresh raw-interface stream (parity use, test-gated).
    #[cfg(test)]
    pub(super) fn new_raw(weights: Arc<SileroWeights>) -> Self {
        Self::with_mode(weights, ContextMode::Raw)
    }

    fn with_mode(weights: Arc<SileroWeights>, mode: ContextMode) -> Self {
        Self {
            weights,
            mode,
            rate: None,
            state: LstmState::zeros(),
            context: Vec::new(),
            pending: Vec::new(),
        }
    }
}

impl VadStreamHandle for VadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let rate = SampleRate::from_hz(sample_rate)?;
        match self.rate {
            None => {
                // The model must actually carry this rate's weights.
                if self.weights.rate(rate).is_none() {
                    return Err(VokraError::InvalidArgument(format!(
                        "model has no weights for {} Hz (this GGUF carries only the other rate)",
                        rate.hz()
                    )));
                }
                self.rate = Some(rate);
            }
            Some(fixed) if fixed != rate => {
                return Err(VokraError::InvalidArgument(format!(
                    "stream is fixed at {} Hz; cannot switch to {} Hz mid-stream (open a new stream)",
                    fixed.hz(),
                    rate.hz()
                )));
            }
            Some(_) => {}
        }

        let w = self
            .weights
            .rate(rate)
            .expect("presence checked when the rate was fixed");
        let frame_len = rate.frame_len();
        let ctx_len = rate.context_len();
        self.pending.extend_from_slice(pcm);

        let mut probs = Vec::new();
        let mut consumed = 0;
        while self.pending.len() - consumed >= frame_len {
            let frame = &self.pending[consumed..consumed + frame_len];
            let prob = match self.mode {
                ContextMode::Raw => run_frame(rate, w, frame, &mut self.state),
                ContextMode::Official => {
                    if self.context.is_empty() {
                        // Fresh stream (or just reset): the official wrapper
                        // starts from a zero context.
                        self.context.resize(ctx_len, 0.0);
                    }
                    let mut buf = Vec::with_capacity(ctx_len + frame_len);
                    buf.extend_from_slice(&self.context);
                    buf.extend_from_slice(frame);
                    let p = run_frame(rate, w, &buf, &mut self.state);
                    // Carry the concatenated input's tail into the next step
                    // (== the frame's last ctx_len samples, as frame_len > ctx_len),
                    // exactly as the upstream wrapper keeps `x[..., -context_size:]`.
                    self.context.copy_from_slice(&buf[buf.len() - ctx_len..]);
                    p
                }
            };
            probs.push(prob);
            consumed += frame_len;
        }
        if consumed > 0 {
            self.pending.drain(0..consumed);
        }
        Ok(probs)
    }

    fn reset(&mut self) {
        self.state = LstmState::zeros();
        self.pending.clear();
        // Official mode: back to the zero context (upstream `reset_states`
        // clears `_context` together with the LSTM state).
        self.context.clear();
        self.rate = None;
    }
}

#[cfg(test)]
mod tests {
    use vokra_core::engines::{VadEngine, VadStreamHandle};
    use vokra_core::rng::Xorshift64Star;

    use super::super::lstm::LstmState;
    use crate::silero_vad::wav::read_wav_f32;
    use crate::silero_vad::{SampleRate, SileroVadV5, parity_dir, run_frame, test_gguf_path};

    fn stream() -> Box<dyn VadStreamHandle + Send> {
        SileroVadV5::open(test_gguf_path())
            .expect("load fixture gguf")
            .open_stream()
    }

    /// Official-context oracle for one step: runs `[context ++ frame]` through
    /// the subgraph, advancing `state`, and returns (probability, next context
    /// = the concatenated input's last `context_len` samples). This is the
    /// upstream `utils_vad.py OnnxWrapper` step, spelled out manually so the
    /// stream's private rolling-context bookkeeping has an in-module oracle.
    fn official_step(
        model: &SileroVadV5,
        rate: SampleRate,
        context: &[f32],
        frame: &[f32],
        state: &mut LstmState,
    ) -> (f32, Vec<f32>) {
        assert_eq!(context.len(), rate.context_len());
        assert_eq!(frame.len(), rate.frame_len());
        let w = model.weights.rate(rate).expect("rate present");
        let mut buf = Vec::with_capacity(context.len() + frame.len());
        buf.extend_from_slice(context);
        buf.extend_from_slice(frame);
        let prob = run_frame(rate, w, &buf, state);
        let next = buf[buf.len() - rate.context_len()..].to_vec();
        (prob, next)
    }

    /// Streamed probabilities must equal the manual official-context sequence:
    /// zero context on the first frame, then each frame prepends the previous
    /// frame's tail, LSTM state carried throughout — bit-identically.
    fn assert_official_context_sequence(wav_name: &str, rate: SampleRate) {
        let model = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        let wav = read_wav_f32(parity_dir().join(wav_name)).expect("read fixture wav");
        assert_eq!(wav.sample_rate, rate.hz(), "fixture rate");
        let frame_len = rate.frame_len();
        let pcm = &wav.samples[..frame_len * 3];

        let mut state = LstmState::zeros();
        let mut context = vec![0.0f32; rate.context_len()];
        let mut want = Vec::new();
        for frame in pcm.chunks_exact(frame_len) {
            let (prob, next) = official_step(&model, rate, &context, frame, &mut state);
            want.push(prob);
            context = next;
        }

        let mut stream = model.open_stream();
        let got = stream.push_pcm(pcm, rate.hz()).unwrap();
        assert_eq!(got, want, "{} Hz official context sequence", rate.hz());
    }

    #[test]
    fn official_context_sequence_16k() {
        assert_official_context_sequence("test_16k.wav", SampleRate::Hz16000);
    }

    #[test]
    fn official_context_sequence_8k() {
        assert_official_context_sequence("test_8k.wav", SampleRate::Hz8000);
    }

    /// First frame of a fresh stream sees a **zero** context (not the raw bare
    /// frame): its probability must differ from the raw interface on real
    /// fixture content and must equal the zero-context oracle.
    #[test]
    fn first_frame_uses_zero_context() {
        let model = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");
        let rate = SampleRate::Hz16000;
        // A frame with non-trivial content (the fixture's noise-burst region)
        // so that zero-context and raw probabilities are distinguishable.
        let start = rate.frame_len() * 13;
        let frame = &wav.samples[start..start + rate.frame_len()];

        let mut state = LstmState::zeros();
        let ctx = vec![0.0f32; rate.context_len()];
        let (want, _) = official_step(&model, rate, &ctx, frame, &mut state);

        let mut stream = model.open_stream();
        let got = stream.push_pcm(frame, rate.hz()).unwrap();
        assert_eq!(got, vec![want]);

        let w = model.weights.rate(rate).unwrap();
        let raw = run_frame(rate, w, frame, &mut LstmState::zeros());
        assert_ne!(
            got[0], raw,
            "official first frame must not be the bare raw frame"
        );
    }

    /// `reset` must clear the rolling context along with the LSTM state: a
    /// replay after reset reproduces the first run bit-identically even though
    /// the pre-reset context tail was non-zero.
    #[test]
    fn reset_clears_rolling_context() {
        let model = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");
        let pcm = &wav.samples[..512 * 2];

        let mut stream = model.open_stream();
        let first = stream.push_pcm(pcm, 16_000).unwrap();
        stream.reset();
        let replay = stream.push_pcm(pcm, 16_000).unwrap();
        assert_eq!(first, replay);
    }

    /// Property: with LSTM state carried across non-overlapping fixed frames,
    /// the per-frame probabilities must not depend on how the *same* PCM is
    /// split across `push_pcm` calls. A whole-utterance push and the identical
    /// samples re-split into irregular chunks must yield bit-identical output.
    /// Oracle is internal self-consistency; no external reference.
    fn assert_chunk_invariant(wav_name: &str, hz: u32) {
        let model = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        let wav = read_wav_f32(parity_dir().join(wav_name)).expect("read fixture wav");
        assert_eq!(wav.sample_rate, hz, "fixture rate");

        // (a) one whole-utterance push.
        let mut whole = model.open_stream();
        let probs_whole = whole.push_pcm(&wav.samples, hz).unwrap();
        assert!(!probs_whole.is_empty(), "fixture yields at least one frame");

        // (b) the same samples re-split into irregular chunks (sizes 1..=700,
        // straddling the 256/512-sample frame boundary and single-sample pushes).
        let mut split = model.open_stream();
        let mut probs_split = Vec::new();
        let mut rng = Xorshift64Star::new(0x5EED_51E1);
        let mut i = 0;
        while i < wav.samples.len() {
            let remaining = wav.samples.len() - i;
            let len = (1 + (rng.next_u64() % 700) as usize).min(remaining);
            probs_split.extend(split.push_pcm(&wav.samples[i..i + len], hz).unwrap());
            i += len;
        }

        assert_eq!(
            probs_whole, probs_split,
            "{hz} Hz: framing must be chunk-invariant"
        );
    }

    #[test]
    fn rejects_unsupported_sample_rate() {
        let mut s = stream();
        assert!(s.push_pcm(&[0.0; 512], 44100).is_err());
    }

    #[test]
    fn buffers_partial_frames_until_complete() {
        let mut s = stream();
        // 300 samples < 512: no frame yet.
        assert!(s.push_pcm(&vec![0.0; 300], 16000).unwrap().is_empty());
        // +300 = 600 >= 512: exactly one frame emitted, 88 buffered.
        assert_eq!(s.push_pcm(&vec![0.0; 300], 16000).unwrap().len(), 1);
    }

    #[test]
    fn rejects_rate_switch_mid_stream() {
        let mut s = stream();
        s.push_pcm(&[0.0; 512], 16000).unwrap();
        assert!(s.push_pcm(&[0.0; 256], 8000).is_err());
    }

    #[test]
    fn reset_reproduces_first_run() {
        let mut s = stream();
        let a = s.push_pcm(&vec![0.1; 512 * 3], 16000).unwrap();
        s.reset();
        let b = s.push_pcm(&vec![0.1; 512 * 3], 16000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn push_pcm_is_chunk_invariant_16k() {
        assert_chunk_invariant("test_16k.wav", 16000);
    }

    #[test]
    fn push_pcm_is_chunk_invariant_8k() {
        assert_chunk_invariant("test_8k.wav", 8000);
    }
}
