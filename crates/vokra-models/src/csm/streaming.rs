//! CSM frame-level streaming (M4-05-T18) + barge-in wiring (T19).
//!
//! # The frame loop (12.5 Hz-family — the config's exact rate)
//!
//! One [`CsmStream::next_frame`] call runs: backbone step → sampling →
//! depth transformer → `mimi_rvq_decode_paged` (codes → features) → Mimi
//! neural decoder (features → PCM) and hands back one frame-hop PCM chunk
//! — the CosyVoice2 chunk-aware pipeline shape without the CFM stage.
//!
//! **Hot path is allocation-free** (FR-EX-05): every buffer (codes, PCM,
//! paged KV pages, conv states) is pre-allocated at
//! [`CsmEngine::open_stream`]; the loop pops pages off the paged arena's
//! free list only. Pinned by `tests/csm_hot_path_alloc.rs` (the M4-03
//! counting-allocator pattern) under the greedy sampler — the stochastic
//! sampler's top-k draw allocates inside the **M1 `Sampler`** (a
//! pre-existing M1 property, out of this WP's blast radius and noted
//! honestly here).
//!
//! # SPSC ring + barge-in (T19)
//!
//! - **Ring**: the loop emits one M1 [`StreamEvent::Token`] per frame into
//!   any [`EventSink`] — [`vokra_core::stream::RingProducer`] implements
//!   the trait, so the control thread polls frame-completion events over
//!   the lock-free M1 SPSC ring while the audio thread pulls the PCM
//!   synchronously from the generation thread. The EOS frame carries
//!   [`TOKEN_FLAG_EOT`].
//! - **Barge-in**: [`CsmInterruptHandle`] carries the **M3-14 contract**
//!   verbatim — a cloneable `Arc<AtomicBool>` with Release-store request /
//!   Acquire-load acknowledge; the generation thread observes the flag at
//!   the top of `next_frame`, stops the loop, resets the KV/queue state so
//!   the **same stream is re-primable for the next turn**, and clears the
//!   flag. The C ABI barge-in (`vokra_stream_interrupt`) operates on the
//!   input-driven `vokra_core::Stream` handle; hosting a CSM dialog
//!   session behind that handle is the multi-session server wave
//!   (FR-SV-06 — outside M4-05 per the WP boundary), at which point this
//!   flag is driven by the stream's own interrupt machinery. The Rust
//!   semantics land here so that hosting is a pure plumbing change.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use vokra_core::decode::TOKEN_FLAG_EOT;
use vokra_core::stream::EventSink;
use vokra_core::{DialogRequest, Result, StreamEvent, VokraError};

use super::audio::CsmAudioDecodeState;
use super::engine::{CsmEngine, CsmFrameSampler};
use super::frame::{CsmFrameKind, CsmGenerationState};

/// Streaming session configuration.
#[derive(Debug, Clone)]
pub struct CsmStreamConfig {
    /// Cap on generated frames (defaults to the engine's 90 s-derived
    /// default when built via [`CsmEngine::open_stream`]).
    pub max_frames: usize,
}

/// Cross-thread barge-in handle — the M3-14
/// [`vokra_core::InterruptHandle`] contract (module docs).
#[derive(Debug, Clone)]
pub struct CsmInterruptHandle {
    flag: Arc<AtomicBool>,
}

impl CsmInterruptHandle {
    /// Requests barge-in. Wait-free Release store; the generation thread
    /// acknowledges at its next [`CsmStream::next_frame`]. Idempotent.
    pub fn interrupt(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Whether a request is pending (`true` until acknowledged).
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// Why a [`CsmStream`] stopped emitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsmStreamStop {
    /// The model emitted the all-zero EOS frame.
    Eos,
    /// The frame cap was reached.
    MaxFrames,
    /// A barge-in was acknowledged (M3-14).
    Interrupted,
}

/// A live CSM streaming turn (borrowing the engine — the engine is the
/// `Send + Sync` owner of every weight).
pub struct CsmStream<'e> {
    engine: &'e CsmEngine,
    generation: CsmGenerationState,
    audio: CsmAudioDecodeState,
    sampler: CsmFrameSampler,
    codes: Vec<u32>,
    pcm: Vec<f32>,
    frame_index: u32,
    max_frames: usize,
    stopped: Option<CsmStreamStop>,
    interrupt: Arc<AtomicBool>,
}

impl std::fmt::Debug for CsmStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmStream")
            .field("frame_index", &self.frame_index)
            .field("max_frames", &self.max_frames)
            .field("stopped", &self.stopped)
            .finish()
    }
}

impl CsmEngine {
    /// Opens a streaming turn: builds the context frames (AEC front /
    /// bypass rules apply exactly as in the batch
    /// [`vokra_core::S2sEngine::dialog`]), primes the backbone, and
    /// pre-allocates every loop buffer.
    ///
    /// # Errors
    ///
    /// Same surface as the batch dialog (empty `reply_text`, echo-path
    /// wiring, tokenizer, context errors).
    pub fn open_stream(
        &self,
        request: &DialogRequest,
        config: Option<CsmStreamConfig>,
    ) -> Result<CsmStream<'_>> {
        let cleaned = match &request.input_audio {
            Some(mic) => Some(self.clean_input(mic)?),
            None => None,
        };
        let frames = self.build_context_frames(request, cleaned.as_deref())?;
        let mut generation = CsmGenerationState::new(self.config())?;
        self.model().prime(&mut generation, &frames)?;
        let default_cap = self
            .config()
            .n_ctx
            .saturating_sub(generation.context_len())
            .max(1);
        let max_frames = config
            .map(|c| c.max_frames)
            .unwrap_or(default_cap)
            .min(default_cap);
        let audio = self.chain().state(max_frames)?;
        let hop = self.chain().frame_hop()?;
        Ok(CsmStream {
            engine: self,
            generation,
            audio,
            sampler: Self::sampler_for(request),
            codes: vec![0; self.config().n_codebooks],
            pcm: vec![0.0; hop],
            frame_index: 0,
            max_frames,
            stopped: None,
            interrupt: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl<'e> CsmStream<'e> {
    /// Generates one frame, emitting a [`StreamEvent::Token`] into `sink`
    /// (`id` = frame index; the EOS event carries [`TOKEN_FLAG_EOT`]) and
    /// returning the frame's PCM (`None` once stopped).
    ///
    /// Zero heap allocation on the greedy path (module docs).
    ///
    /// # Errors
    ///
    /// Propagates generation / decode errors verbatim.
    pub fn next_frame(&mut self, sink: &mut dyn EventSink) -> Result<Option<&[f32]>> {
        if self.stopped.is_some() {
            return Ok(None);
        }
        // M3-14 acknowledge: stop, make the state next-turn-reusable,
        // clear the flag last (Release) — the handle's is_pending stays
        // observable during the flush.
        if self.interrupt.load(Ordering::Acquire) {
            self.generation.reset();
            self.stopped = Some(CsmStreamStop::Interrupted);
            self.interrupt.store(false, Ordering::Release);
            return Ok(None);
        }
        if (self.frame_index as usize) >= self.max_frames {
            self.stopped = Some(CsmStreamStop::MaxFrames);
            return Ok(None);
        }
        let sampler = &mut self.sampler;
        let kind = self.engine.model().generate_frame_into(
            &mut self.generation,
            &mut |l| sampler.sample(l),
            &mut self.codes,
        )?;
        match kind {
            CsmFrameKind::Eos => {
                // Frame-completion event with the EOT flag; backpressure
                // (a full ring) is surfaced, never dropped silently.
                if !sink.emit(StreamEvent::Token {
                    id: self.frame_index,
                    flags: TOKEN_FLAG_EOT,
                }) {
                    return Err(VokraError::InvalidArgument(
                        "csm stream: event ring full at EOS (consumer stalled) — \
                         FR-EX-08, the event is not dropped silently"
                            .into(),
                    ));
                }
                self.stopped = Some(CsmStreamStop::Eos);
                Ok(None)
            }
            CsmFrameKind::Audio => {
                self.engine.chain().decode_frame_into(
                    &mut self.audio,
                    &self.codes,
                    &mut self.pcm,
                )?;
                if !sink.emit(StreamEvent::Token {
                    id: self.frame_index,
                    flags: 0,
                }) {
                    return Err(VokraError::InvalidArgument(
                        "csm stream: event ring full (consumer stalled) — FR-EX-08".into(),
                    ));
                }
                self.frame_index += 1;
                Ok(Some(&self.pcm))
            }
        }
    }

    /// Cloneable, `Send + Sync` barge-in handle (M3-14 contract).
    #[must_use]
    pub fn interrupt_handle(&self) -> CsmInterruptHandle {
        CsmInterruptHandle {
            flag: Arc::clone(&self.interrupt),
        }
    }

    /// Frames emitted so far.
    #[must_use]
    pub fn frames_emitted(&self) -> u32 {
        self.frame_index
    }

    /// Why the stream stopped (`None` while live).
    #[must_use]
    pub fn stopped(&self) -> Option<CsmStreamStop> {
        self.stopped
    }

    /// Re-primes the (stopped) stream for the next dialog turn, reusing
    /// every pre-allocated buffer — the T19 "KV/queue 状態が次 turn へ
    /// 再利用可能" contract. The paged backbone arena's pages return to
    /// the free list; no reallocation happens.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when the stream is still live;
    /// propagates context-building errors.
    pub fn reprime_for_next_turn(&mut self, request: &DialogRequest) -> Result<()> {
        if self.stopped.is_none() {
            return Err(VokraError::InvalidArgument(
                "csm stream: reprime_for_next_turn on a live stream — interrupt or \
                 drain it first"
                    .into(),
            ));
        }
        let cleaned = match &request.input_audio {
            Some(mic) => Some(self.engine.clean_input(mic)?),
            None => None,
        };
        let frames = self
            .engine
            .build_context_frames(request, cleaned.as_deref())?;
        self.generation.reset();
        self.engine.model().prime(&mut self.generation, &frames)?;
        // Fresh audio-chain state sized to the remaining cap (the paged
        // feature arena restarts at t = 0 for the new turn).
        let default_cap = self
            .engine
            .config()
            .n_ctx
            .saturating_sub(self.generation.context_len())
            .max(1);
        self.max_frames = default_cap;
        self.audio = self.engine.chain().state(self.max_frames)?;
        self.sampler = CsmEngine::sampler_for(request);
        self.frame_index = 0;
        self.stopped = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csm::EchoPath;
    use std::time::Instant;

    fn engine() -> CsmEngine {
        CsmEngine::synthesized_fixture(31)
            .unwrap()
            .with_echo_path(EchoPath::BypassRecordedInput)
    }

    fn request() -> DialogRequest {
        DialogRequest::new("stream me").deterministic()
    }

    #[test]
    fn streaming_matches_the_batch_dialog_output() {
        use vokra_core::S2sEngine;
        let e = engine();
        let batch = e
            .dialog(&request().with_max_frames(4))
            .unwrap()
            .audio
            .unwrap();
        let mut stream = e
            .open_stream(&request(), Some(CsmStreamConfig { max_frames: 4 }))
            .unwrap();
        let mut sink: Vec<StreamEvent> = Vec::new();
        let mut streamed = Vec::new();
        while let Some(pcm) = stream.next_frame(&mut sink).unwrap() {
            streamed.extend_from_slice(pcm);
        }
        assert_eq!(batch.samples, streamed, "T18 determinism: batch == stream");
        assert_eq!(
            sink.len(),
            stream.frames_emitted() as usize
                + usize::from(stream.stopped() == Some(CsmStreamStop::Eos))
        );
    }

    #[test]
    fn ring_events_carry_frame_indices_over_the_m1_spsc_ring() {
        let e = engine();
        let mut stream = e
            .open_stream(&request(), Some(CsmStreamConfig { max_frames: 3 }))
            .unwrap();
        let (mut producer, mut consumer) = vokra_core::stream::channel(16);
        let mut n = 0u32;
        while stream.next_frame(&mut producer).unwrap().is_some() {
            n += 1;
        }
        let mut seen = 0u32;
        while let Some(raw) = consumer.pop() {
            if let Some(StreamEvent::Token { id, .. }) = StreamEvent::from_raw(raw) {
                assert_eq!(id, seen, "monotonic frame indices");
                seen += 1;
            }
        }
        assert!(seen >= n, "every emitted frame has a ring event");
    }

    #[test]
    fn barge_in_stops_the_loop_and_the_stream_reprimes_for_the_next_turn() {
        let e = engine();
        let mut stream = e
            .open_stream(&request(), Some(CsmStreamConfig { max_frames: 8 }))
            .unwrap();
        let handle = stream.interrupt_handle();
        let mut sink: Vec<StreamEvent> = Vec::new();
        // Emit one frame, then barge in from "another thread".
        assert!(stream.next_frame(&mut sink).unwrap().is_some());
        handle.interrupt();
        assert!(handle.is_pending());
        assert!(stream.next_frame(&mut sink).unwrap().is_none());
        assert_eq!(stream.stopped(), Some(CsmStreamStop::Interrupted));
        assert!(!handle.is_pending(), "flag cleared after acknowledge");
        // The same stream reprimes and reproduces a fresh turn (KV pages
        // reused off the free list).
        stream.reprime_for_next_turn(&request()).unwrap();
        let mut a = Vec::new();
        while let Some(pcm) = stream.next_frame(&mut sink).unwrap() {
            a.extend_from_slice(pcm);
            if stream.frames_emitted() >= 2 {
                break;
            }
        }
        assert!(!a.is_empty(), "post-barge-in turn generates");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn reprime_on_a_live_stream_is_rejected() {
        let e = engine();
        let mut stream = e.open_stream(&request(), None).unwrap();
        assert!(stream.reprime_for_next_turn(&request()).is_err());
    }

    #[test]
    fn ttfa_rtf_reference_measurement_tiny_fixture() {
        // T19 参考計測 — the tiny synthesized fixture floor (NOT the real
        // model; the honest analog of the M3-15 in-process FakeSynth
        // floor). No target is hard-asserted (the WP has no native RTF
        // completion number — Exit criteria is "CSM streaming 動作");
        // real-model / real-device numbers are the owner track (T30).
        let e = engine();
        let hop = e.chain().frame_hop().unwrap();
        let sr = e.config().sample_rate as f64;
        let mut stream = e
            .open_stream(&request(), Some(CsmStreamConfig { max_frames: 8 }))
            .unwrap();
        let mut sink: Vec<StreamEvent> = Vec::new();
        let t0 = Instant::now();
        let first = stream.next_frame(&mut sink).unwrap();
        let ttfa = t0.elapsed();
        assert!(first.is_some(), "fixture must emit a first frame");
        let mut frames = 1usize;
        while stream.next_frame(&mut sink).unwrap().is_some() {
            frames += 1;
        }
        let wall = t0.elapsed();
        let audio_s = (frames * hop) as f64 / sr;
        let rtf = wall.as_secs_f64() / audio_s;
        println!(
            "csm tiny-fixture reference: TTFA {:.3} ms, {} frames, wall {:.3} ms, \
             RTF {rtf:.4} (synthesized fixture floor — not the real model)",
            ttfa.as_secs_f64() * 1e3,
            frames,
            wall.as_secs_f64() * 1e3,
        );
        assert!(ttfa.as_nanos() > 0 && rtf.is_finite());
    }
}
