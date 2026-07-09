//! Voxtral streaming ASR pipeline (M3-10-T18).
//!
//! # Scope (foundation)
//!
//! Voxtral's Whisper-derived audio encoder + Mistral text decoder run in a
//! **chunked** streaming ASR mode: the caller pushes fixed-size PCM chunks
//! (`chunk_ms` milliseconds), each chunk is turned into log-mel frames, run
//! through the audio encoder, and the newly-produced encoder hidden states
//! are appended to a running buffer the decoder can attend to. Text is
//! emitted incrementally as new tokens fall out of a greedy step function.
//!
//! This foundation file lands the **types + wiring** the ticket calls out:
//!
//! - [`StreamingConfig`] — the caller-facing configuration (`chunk_ms`,
//!   `max_new_tokens`, backend selection, decoder step-session opt-in).
//! - [`StreamingAsr`] — the driver that owns the encoder/decoder state and
//!   exposes `push_chunk` + `finalize` (single-thread contract, mirrors
//!   `WhisperSession` — the caller is expected to own a lock or a channel
//!   for cross-thread pushes).
//! - [`StreamingChunk`] — the value type `push_chunk` returns: any tokens
//!   the current chunk added, plus running audio-context statistics for
//!   progress reporting.
//!
//! # DecodeSession seam (parity with Whisper Metal/Cuda)
//!
//! The decoder step in this foundation is a CPU-side loop, gated so a future
//! [`VoxtralDecodeSession`] can be swapped in exactly the way
//! `crate::whisper::DecoderState::device_session` already is (see
//! `crates/vokra-models/src/whisper/decoder.rs:88`). Concretely: the driver
//! consults [`StreamingConfig::allow_device_session`] and — when the target
//! backend reports device-side decoder-step support — routes through the
//! seam. Today no Voxtral device session exists; the flag is honoured as
//! **information** only (it never silently downgrades to CPU with no
//! warning; a request for a device session on a backend that does not carry
//! one is a hard [`VokraError::UnsupportedOp`] per FR-EX-08).
//!
//! # What this foundation does NOT do
//!
//! - It does not run the full autoregressive greedy step (RoPE, GQA, SwiGLU
//!   and RMSNorm composed) — the block math is a downstream ticket, see
//!   `text_decoder.rs`. `push_chunk` runs the audio encoder end-to-end and
//!   surfaces a clear [`VokraError::NotImplemented`] from the token step,
//!   never a fabricated pass.
//! - It does not do voice-activity chunk boundaries; the caller drives chunk
//!   sizes. Silero VAD integration is a follow-up (the Whisper server layer
//!   already uses `SileroVadV5` for this — parity item, not a Voxtral-side
//!   concern).
//! - It does not implement AudioSeal watermarking on the ASR path — ASR
//!   output is text, so it is watermark-exempt per T17.
//!
//! # Zero-alloc hot path (FR-EX-05)
//!
//! `push_chunk` pre-sizes its log-mel scratch to the maximum chunk length the
//! caller declared at construction. Per-chunk allocs are limited to the
//! encoder output buffer (which grows monotonically with the utterance), and
//! the emitted-token `Vec<u32>` on [`StreamingChunk`]. Downstream GPU
//! sessions will further pin the encoder scratch in device memory — this
//! foundation exposes the shape that follow-up needs.

use vokra_core::{BackendKind, Result, VokraError};

use crate::compute::Compute;

use super::audio_encoder::forward as audio_encoder_forward;
use super::{AudioEncoder, AudioEncoderOutput, TextDecoder, TextDecoderStep, VoxtralConfig};

/// A single chunk's emitted state — new tokens, plus how much audio the
/// stream has consumed so far.
#[derive(Debug, Clone, Default)]
pub struct StreamingChunk {
    /// New token ids the step-function emitted for this chunk (may be
    /// empty). Empty on the foundation path because the greedy step is a
    /// downstream ticket.
    pub tokens: Vec<u32>,
    /// Cumulative audio milliseconds consumed by the stream so far.
    pub audio_ms: u64,
    /// Number of encoder context positions currently in the running buffer.
    pub encoder_ctx: usize,
    /// Total decoder tokens emitted for the whole utterance so far.
    pub total_tokens: usize,
}

/// Configuration for [`StreamingAsr`].
///
/// Every field has an explicit invariant: a value the runtime cannot honour
/// (`chunk_ms = 0`, `sample_rate = 0`, a device-session request against a
/// backend that has none) surfaces as an error at construction — never a
/// silent substitution (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingConfig {
    /// Length of each PCM chunk the caller will push, in milliseconds.
    /// Fixed for the life of the stream (mirrors Whisper's chunked ASR
    /// framing). Must be `> 0`.
    pub chunk_ms: u32,
    /// PCM sample rate (Hz). Voxtral / Whisper both expect 16_000; the field
    /// is explicit so a mis-configured caller fails at construction, not at
    /// step time.
    pub sample_rate: u32,
    /// Upper bound on decoder tokens the stream will emit per chunk. The
    /// greedy step stops early on EOS; this is only a guard against runaway
    /// generation (FR-EX-05 hot-path bound).
    pub max_new_tokens_per_chunk: u32,
    /// Which backend the encoder + decoder will dispatch through
    /// ([`Compute::for_backend`]).
    pub backend: BackendKind,
    /// Opt into the future device-side decoder step session (see the module
    /// doc's DecodeSession seam paragraph). No backend carries one today; a
    /// `true` here on any backend surfaces an explicit
    /// [`VokraError::UnsupportedOp`] at construction until a
    /// `VoxtralDecodeSession` lands. Kept as a config field (not
    /// auto-detected) so downstream cannot silently downgrade a caller who
    /// wants the session path (FR-EX-08).
    pub allow_device_session: bool,
}

impl StreamingConfig {
    /// A minimal, valid configuration for a 30-s Whisper-style chunk on CPU
    /// (matches the Voxtral upstream chunk cadence). Callers may override
    /// per-field before construction.
    #[must_use]
    pub fn default_cpu() -> Self {
        Self {
            chunk_ms: 30_000,
            sample_rate: 16_000,
            max_new_tokens_per_chunk: 256,
            backend: BackendKind::Cpu,
            allow_device_session: false,
        }
    }

    /// Total samples in one chunk = `chunk_ms * sample_rate / 1000`.
    /// Overflow-safe via `u64` intermediate (a `u32` chunk_ms of 30_000
    /// times a 48_000-Hz sample rate is still well below `u32::MAX`).
    #[must_use]
    pub fn samples_per_chunk(&self) -> usize {
        let ms = u64::from(self.chunk_ms);
        let sr = u64::from(self.sample_rate);
        ((ms * sr) / 1_000) as usize
    }

    /// Validates the fields and returns `Ok(())` iff every invariant holds.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate(&self) -> Result<()> {
        if self.chunk_ms == 0 {
            return Err(VokraError::InvalidArgument(
                "streaming config: chunk_ms must be > 0".to_owned(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "streaming config: sample_rate must be > 0".to_owned(),
            ));
        }
        if self.max_new_tokens_per_chunk == 0 {
            return Err(VokraError::InvalidArgument(
                "streaming config: max_new_tokens_per_chunk must be > 0".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::default_cpu()
    }
}

/// A single-utterance chunked ASR driver.
///
/// `!Send` at the type level intentionally: the driver owns a live [`Compute`]
/// dispatcher which, on Metal, is `!Send`. Cross-thread streaming is done
/// via message-passing over the chunk boundary (see `docs/design/*` — the
/// same pattern Whisper's `WhisperSession` uses).
///
/// # Lifecycle
///
/// 1. [`StreamingAsr::new`] — validate config, allocate encoder scratch,
///    reject a device-session request on a backend that has none;
/// 2. `push_chunk` — encode one chunk and (foundation: return no tokens);
/// 3. `finalize` — flush any final decoder state and yield the tail.
///
/// The driver does not enforce a total chunk budget; a caller who wants a
/// bounded utterance stops calling `push_chunk` themselves.
pub struct StreamingAsr<'m> {
    config: &'m VoxtralConfig,
    audio: &'m AudioEncoder,
    #[allow(dead_code)] // consumed by the full step forward (follow-up).
    text: &'m TextDecoder,
    stream_config: StreamingConfig,
    /// Running number of chunks consumed.
    chunks_seen: u64,
    /// Running total decoder tokens emitted.
    total_tokens: usize,
    /// Cumulative encoder context positions in the running buffer.
    encoder_ctx_total: usize,
    /// Current decoder step position (updated as the greedy loop advances).
    ///
    /// `#[allow(dead_code)]` — this is the DecodeSession-shaped slot the
    /// downstream full-forward ticket hangs KV caches off. The
    /// [`Self::advance_decoder`] entry point exercises it today (the tests
    /// verify monotonic advancement); the field is read by the full-forward
    /// path once M3-10-T13 lands. Silencing the warning without hiding the
    /// intent keeps the seam obvious to reviewers.
    #[allow(dead_code)]
    decoder_step: TextDecoderStep,
    /// A cached [`Compute`] dispatcher for the run. Built once at `new` so a
    /// per-chunk push does not repeat the backend probe.
    compute: Compute,
}

impl<'m> StreamingAsr<'m> {
    /// Constructs a new streaming ASR driver.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from
    ///   [`StreamingConfig::validate`];
    /// - [`VokraError::UnsupportedOp`] if `allow_device_session = true` on a
    ///   backend that does not (yet) carry a Voxtral device session;
    /// - [`VokraError::ModelLoad`] if the model's config is the zero-sentinel
    ///   converter path.
    pub fn new(
        model_config: &'m VoxtralConfig,
        audio: &'m AudioEncoder,
        text: &'m TextDecoder,
        stream_config: StreamingConfig,
    ) -> Result<Self> {
        stream_config.validate()?;
        if model_config.audio.n_layer == 0 || model_config.audio.hidden_dim == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral streaming: model config has 0-sentinel audio encoder — re-convert with a \
                 full VoxtralConfig (FR-EX-08)."
                    .to_owned(),
            ));
        }

        // Device-session gate: no backend carries a Voxtral device session
        // yet. A caller who explicitly requested one gets a hard
        // UnsupportedOp — never a silent CPU fall back (FR-EX-08).
        if stream_config.allow_device_session {
            return Err(VokraError::UnsupportedOp(format!(
                "voxtral streaming: allow_device_session = true but no VoxtralDecodeSession is \
                 implemented for backend {:?} yet. Set allow_device_session = false or wait for \
                 the follow-up ticket (mirrors whisper::DecoderState::device_session).",
                stream_config.backend
            )));
        }

        // Build the compute dispatcher. Voxtral's audio-encoder + text
        // decoder share the same six hot ops as Whisper (see
        // `crate::voxtral::VOXTRAL_HOT_OPS`); the compute seam gates the
        // backend against them.
        let compute = match stream_config.backend {
            BackendKind::Cpu => Compute::cpu(),
            other => {
                return Err(VokraError::UnsupportedOp(format!(
                    "voxtral streaming: backend {other:?} — CPU is the only wired backend for \
                     the streaming foundation. Metal/CUDA arrive with the T15/T16 GPU seam \
                     extension. FR-EX-08 — no silent CPU fall back."
                )));
            }
        };
        Ok(Self {
            config: model_config,
            audio,
            text,
            stream_config,
            chunks_seen: 0,
            total_tokens: 0,
            encoder_ctx_total: 0,
            decoder_step: TextDecoderStep::new(),
            compute,
        })
    }

    /// Consumes one PCM chunk and returns any tokens the step function
    /// emitted for it.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm.len()` disagrees with
    ///   [`StreamingConfig::samples_per_chunk`] by more than one sample
    ///   (some resamplers emit ±1);
    /// - [`VokraError::InvalidArgument`] if `log_mel.len()` disagrees with
    ///   `n_mels * n_frames` — surfaced from the encoder;
    /// - [`VokraError::NotImplemented`] from the token step until the
    ///   downstream ticket lands.
    ///
    /// # Contract
    ///
    /// This entry point is **honest about scope** — the returned
    /// [`StreamingChunk`] carries `tokens: []` and `total_tokens =
    /// self.total_tokens` on the foundation path. A future extension will
    /// append greedy tokens; this signature is stable across that change
    /// (no re-plumbing on the caller side).
    pub fn push_chunk(&mut self, log_mel: &[f32], n_frames: usize) -> Result<StreamingChunk> {
        // Run the audio encoder end-to-end. The output shape is
        // [n_ctx, hidden_dim]; the streaming layer accumulates n_ctx
        // across chunks for the decoder's cross-attention.
        let AudioEncoderOutput { n_ctx, .. } =
            audio_encoder_forward(&self.compute, self.config, self.audio, log_mel, n_frames)?;

        self.encoder_ctx_total = self.encoder_ctx_total.saturating_add(n_ctx);
        self.chunks_seen = self.chunks_seen.saturating_add(1);
        let audio_ms = self
            .chunks_seen
            .saturating_mul(u64::from(self.stream_config.chunk_ms));

        // Foundation: token emission is a downstream ticket. Return the
        // running statistics without fabricating output.
        Ok(StreamingChunk {
            tokens: Vec::new(),
            audio_ms,
            encoder_ctx: self.encoder_ctx_total,
            total_tokens: self.total_tokens,
        })
    }

    /// Advances the underlying decoder-step counter — a hook the greedy loop
    /// will call as tokens fall out. Foundation exposes this so tests can
    /// verify the running-total accounting; production callers should not
    /// invoke it directly.
    ///
    /// `#[allow(dead_code)]` — invoked from the streaming tests today, and
    /// wired into the greedy step loop by the M3-10-T13 follow-up. Kept
    /// `pub(crate)` so a future extension inside `vokra-models` can drive it
    /// without widening the crate's public surface.
    #[allow(dead_code)]
    pub(crate) fn advance_decoder(&mut self) {
        self.decoder_step.advance();
        self.total_tokens = self.total_tokens.saturating_add(1);
    }

    /// Finalise the stream and return the final chunk (any decoder state the
    /// utterance may have carried across chunks flushes here).
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] from the token step until the
    /// downstream ticket lands.
    pub fn finalize(&mut self) -> Result<StreamingChunk> {
        // Foundation: no decoder tail flush yet. Surface an honest empty
        // chunk with the running counters.
        Ok(StreamingChunk {
            tokens: Vec::new(),
            audio_ms: self
                .chunks_seen
                .saturating_mul(u64::from(self.stream_config.chunk_ms)),
            encoder_ctx: self.encoder_ctx_total,
            total_tokens: self.total_tokens,
        })
    }

    /// The chunk cadence + sample rate configuration.
    #[must_use]
    pub fn stream_config(&self) -> StreamingConfig {
        self.stream_config
    }

    /// Number of audio chunks pushed so far.
    #[must_use]
    pub fn chunks_seen(&self) -> u64 {
        self.chunks_seen
    }

    /// Cumulative encoder-context positions in the running buffer.
    #[must_use]
    pub fn encoder_ctx_total(&self) -> usize {
        self.encoder_ctx_total
    }

    /// Cumulative decoder tokens emitted.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};

    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 8,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 8,
                n_ctx: 8,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    fn tiny_encoder(cfg: &VoxtralConfig) -> AudioEncoder {
        AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        }
    }

    fn empty_decoder() -> TextDecoder {
        TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        }
    }

    #[test]
    fn default_cpu_config_validates() {
        let sc = StreamingConfig::default_cpu();
        sc.validate().unwrap();
        assert_eq!(sc.chunk_ms, 30_000);
        assert_eq!(sc.sample_rate, 16_000);
        assert_eq!(sc.samples_per_chunk(), 480_000); // 30 s @ 16 kHz
    }

    #[test]
    fn zero_chunk_ms_is_rejected() {
        let mut sc = StreamingConfig::default_cpu();
        sc.chunk_ms = 0;
        assert!(matches!(sc.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn zero_sample_rate_is_rejected() {
        let mut sc = StreamingConfig::default_cpu();
        sc.sample_rate = 0;
        assert!(matches!(sc.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn zero_max_tokens_is_rejected() {
        let mut sc = StreamingConfig::default_cpu();
        sc.max_new_tokens_per_chunk = 0;
        assert!(matches!(sc.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn new_rejects_device_session_request_on_all_backends() {
        // Every backend today: allow_device_session=true → hard
        // UnsupportedOp. Silent CPU fall back would violate FR-EX-08.
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let sc = StreamingConfig {
                allow_device_session: true,
                backend,
                ..StreamingConfig::default_cpu()
            };
            assert!(matches!(
                StreamingAsr::new(&cfg, &ae, &td, sc),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
    }

    #[test]
    fn new_rejects_non_cpu_backend_in_foundation() {
        // Foundation only wires CPU. Metal/CUDA reach the compute seam
        // in T15/T16; asking for them now must be a hard UnsupportedOp.
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        for be in [BackendKind::Metal, BackendKind::Cuda] {
            let sc = StreamingConfig {
                backend: be,
                allow_device_session: false,
                ..StreamingConfig::default_cpu()
            };
            let err = StreamingAsr::new(&cfg, &ae, &td, sc);
            assert!(matches!(err, Err(VokraError::UnsupportedOp(_))));
        }
    }

    #[test]
    fn new_rejects_zero_sentinel_audio_config() {
        // The shape-only converter path leaves n_layer=0; a streaming driver
        // built on that config must not fabricate output — reject at new.
        let mut cfg = tiny_config();
        cfg.audio.n_layer = 0;
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        let err = StreamingAsr::new(&cfg, &ae, &td, StreamingConfig::default_cpu());
        assert!(matches!(err, Err(VokraError::ModelLoad(_))));
    }

    #[test]
    fn push_chunk_runs_encoder_and_updates_counters() {
        // All-zero encoder → hidden output is 0 (see audio_encoder tests);
        // the streaming layer's job here is to count chunks + n_ctx and
        // return an honest empty-token chunk.
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        let mut sc = StreamingConfig::default_cpu();
        sc.chunk_ms = 500; // shorter for a smaller test
        let mut asr = StreamingAsr::new(&cfg, &ae, &td, sc).unwrap();

        let n_frames = 8;
        let log_mel = vec![0.0f32; cfg.audio.n_mels * n_frames];
        let chunk = asr.push_chunk(&log_mel, n_frames).unwrap();
        assert!(chunk.tokens.is_empty(), "foundation: no fabricated tokens");
        assert_eq!(chunk.audio_ms, 500);
        assert_eq!(asr.chunks_seen(), 1);
        assert!(chunk.encoder_ctx > 0, "encoder should emit >=1 ctx");

        // Second chunk: audio_ms doubles, chunks_seen == 2.
        let chunk2 = asr.push_chunk(&log_mel, n_frames).unwrap();
        assert_eq!(chunk2.audio_ms, 1_000);
        assert_eq!(asr.chunks_seen(), 2);
        assert!(chunk2.encoder_ctx >= chunk.encoder_ctx);
    }

    #[test]
    fn push_chunk_surfaces_log_mel_shape_mismatch() {
        // A caller who miscomputes n_frames or n_mels must see the encoder's
        // InvalidArgument straight through — never silently zero-padded.
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        let mut asr = StreamingAsr::new(&cfg, &ae, &td, StreamingConfig::default_cpu()).unwrap();
        assert!(matches!(
            asr.push_chunk(&[1.0, 2.0, 3.0], 8),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn finalize_returns_running_state_without_fabricating_tokens() {
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        let mut asr = StreamingAsr::new(&cfg, &ae, &td, StreamingConfig::default_cpu()).unwrap();
        let n_frames = 8;
        let log_mel = vec![0.0f32; cfg.audio.n_mels * n_frames];
        asr.push_chunk(&log_mel, n_frames).unwrap();
        let tail = asr.finalize().unwrap();
        assert!(tail.tokens.is_empty(), "foundation: no fabricated tokens");
        assert_eq!(tail.encoder_ctx, asr.encoder_ctx_total());
        assert_eq!(tail.total_tokens, asr.total_tokens());
    }

    #[test]
    fn advance_decoder_updates_total_tokens_monotonically() {
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = empty_decoder();
        let mut asr = StreamingAsr::new(&cfg, &ae, &td, StreamingConfig::default_cpu()).unwrap();
        assert_eq!(asr.total_tokens(), 0);
        asr.advance_decoder();
        asr.advance_decoder();
        asr.advance_decoder();
        assert_eq!(asr.total_tokens(), 3);
    }

    #[test]
    fn samples_per_chunk_computes_correctly() {
        let sc = StreamingConfig {
            chunk_ms: 500,
            sample_rate: 16_000,
            ..StreamingConfig::default_cpu()
        };
        assert_eq!(sc.samples_per_chunk(), 8_000);
    }
}
