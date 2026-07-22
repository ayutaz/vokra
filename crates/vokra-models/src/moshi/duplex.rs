//! Moshi full-duplex session — mic → AEC → Mimi encode → frame step →
//! Mimi decode → speaker, continuously (M4-06-T16 API + T17 AEC wiring +
//! T18 barge-in + T21 AEC-required posture).
//!
//! # Pipeline (ADR M4-06 §D4)
//!
//! ```text
//! push_mic_frame(pcm[hop]):
//!   mic ── AecFront::process_mic(mic, mic_pos, reader) ──> cleaned
//!       └ (explicit opt-out only) passthrough, warning recorded
//!   cleaned ── MimiEncoder (streaming state) ──> user codes
//!   codes ── MoshiModel::step (delay ring + backbone + depformer)
//!         ──> Option<(text, own codes)>
//!   own codes ── RVQ features + Mimi neural decoder ──> model PCM [hop]
//!   model PCM ──> out queue
//! pull_model_frame():
//!   out queue pop ──> AecRefWriter::push(pcm, play_pos + offset) ──> Some(pcm)
//! ```
//!
//! Wall-clock-free: `mic_pos` / `play_pos` are sample counters derived
//! from the session frame index; real playback latency is compensated
//! with [`DuplexSessionConfig::playback_offset_samples`] (owner-tunable
//! at the T30 real-hardware acceptance).
//!
//! # AEC is the default, bypass is explicit (FR-EX-08 / FR-OP-60)
//!
//! **AEC 無しの Moshi/CSM は自己エコーで即崩壊** (CLAUDE.md レビュアー C
//! 指摘 #3): a duplex model that hears its own playback re-encodes its own
//! voice as the user and spirals. A session opens with the canceller
//! required; [`DuplexSessionConfig::with_aec_disabled_explicitly`] is the
//! only bypass and leaves a loud warning on
//! [`MoshiDuplexSession::warnings`] (plus stderr) — there is no silent
//! skip path.
//!
//! # Barge-in (T18 — M3-14 semantics)
//!
//! [`DuplexInterruptHandle::interrupt`] is acknowledged at the next
//! push/pull boundary: pending model frames flush, the generation stack
//! (delay ring / paged KV / depformer / Mimi streaming states / inner
//! monologue) resets, the flag clears last (Release) — mic intake
//! continues. The AEC front and its reference clock are **not** reset
//! (the physical playback clock keeps running — `csm::aec_front` reset
//! rationale). With identical *cleaned* input, post-reset generation is
//! bit-identical to a fresh session (pinned by test; with the canceller
//! enabled its adaptive state intentionally survives, so equality is
//! scoped to the generation stack).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use vokra_core::{
    DuplexInterruptHandle, DuplexPushReport, DuplexSessionConfig, Result, S2sDuplexHandle,
    VokraError,
};

use super::engine::MoshiEngine;
use super::frame::{MoshiFrameOut, MoshiGenerationState, MoshiSamplerPair};
use super::tokenizer::decode_monologue;
use crate::csm::aec_front::AecFront;
use crate::csm::audio::{CsmAudioDecodeState, OutputLimiter};
use crate::mimi::MimiEncoderState;
use vokra_core::stream::AecRefWriter;

/// The live AEC pair (front + the playback-side writer) with its clocks.
struct AecSession {
    front: AecFront,
    writer: AecRefWriter,
}

/// A live Moshi full-duplex session (module docs). Obtained through
/// [`MoshiEngine::open_duplex_session`] (owning, `E = Arc<MoshiEngine>` —
/// the [`vokra_core::S2s::duplex`] facade / C ABI path) or borrowed
/// internally by the batch [`vokra_core::S2sEngine::dialog`] face
/// (`E = &MoshiEngine`) — one pipeline body, two ownership shapes.
pub struct MoshiDuplexSession<E: std::borrow::Borrow<MoshiEngine> = Arc<MoshiEngine>> {
    engine: E,
    gen_state: MoshiGenerationState,
    samplers: MoshiSamplerPair,
    enc_state: MimiEncoderState,
    audio_state: CsmAudioDecodeState,
    aec: Option<AecSession>,
    /// Config snapshot (sampler rebuild on barge-in reset).
    deterministic: bool,
    seed: u64,
    playback_offset_samples: u64,
    /// Sample clocks (frame index × hop).
    mic_pos: u64,
    play_pos: u64,
    /// Pending model frames (pull side). Unbounded by design: growth is
    /// bounded by the caller's push/pull imbalance, and dropping frames
    /// silently is forbidden (FR-EX-08).
    out_queue: VecDeque<Vec<f32>>,
    /// Inner-monologue token stream (undelayed).
    monologue: Vec<u32>,
    interrupt: Arc<AtomicBool>,
    /// Opt-in output limiter applied to each decoded model frame before it is
    /// queued (M4-RESIDUAL-B (C)). Default [`OutputLimiter::None`] = the current
    /// behaviour, bit-for-bit (FR-EX-08 — no silent change).
    limiter: OutputLimiter,
    warnings: Vec<String>,
    /// run_inference.py first-frame convention: the first real codes are
    /// stepped twice (ADR M4-06 §D2).
    first_frame_done: bool,
    // Pre-allocated scratch.
    user_codes: Vec<u32>,
    frame_out: MoshiFrameOut,
    cleaned: Vec<f32>,
    frame_pcm: Vec<f32>,
    hop: usize,
}

impl<E: std::borrow::Borrow<MoshiEngine>> std::fmt::Debug for MoshiDuplexSession<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiDuplexSession")
            .field("steps", &self.gen_state.steps())
            .field("aec_wired", &self.aec.is_some())
            .field("pending_frames", &self.out_queue.len())
            .field("warnings", &self.warnings.len())
            .finish()
    }
}

impl<E: std::borrow::Borrow<MoshiEngine>> MoshiDuplexSession<E> {
    /// The engine behind this session.
    fn eng(&self) -> &MoshiEngine {
        self.engine.borrow()
    }

    /// Assembles a session (called by
    /// [`MoshiEngine::open_duplex_session`], which owns the AEC posture
    /// checks — T21).
    pub(super) fn new(
        engine: E,
        config: &DuplexSessionConfig,
        aec: Option<(AecFront, AecRefWriter)>,
        warnings: Vec<String>,
    ) -> Result<Self> {
        let cfg = engine.borrow().config().clone();
        let hop = engine.borrow().encoder().frame_hop()?;
        if let Some((front, _)) = &aec {
            let fs = front.frame_size();
            if fs == 0 || hop % fs != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi duplex: Mimi frame hop {hop} is not a whole multiple of \
                     the AEC frame size {fs} — pick an AEC frame size dividing the \
                     hop (FR-EX-08, no partial-frame cancellation)"
                )));
            }
        }
        let samplers = if config.deterministic {
            MoshiSamplerPair::greedy()
        } else {
            MoshiSamplerPair::stochastic(config.seed)
        };
        Ok(Self {
            gen_state: MoshiGenerationState::new(&cfg)?,
            samplers,
            enc_state: engine.borrow().encoder().state(1)?,
            audio_state: engine.borrow().chain().state(cfg.max_ctx)?,
            aec: aec.map(|(front, writer)| AecSession { front, writer }),
            deterministic: config.deterministic,
            seed: config.seed,
            playback_offset_samples: config.playback_offset_samples,
            mic_pos: 0,
            play_pos: 0,
            out_queue: VecDeque::new(),
            monologue: Vec::new(),
            interrupt: Arc::new(AtomicBool::new(false)),
            limiter: OutputLimiter::None,
            warnings,
            first_frame_done: false,
            user_codes: vec![0; cfg.n_user_streams()],
            frame_out: MoshiFrameOut::new(&cfg),
            cleaned: vec![0.0; hop],
            frame_pcm: vec![0.0; hop],
            hop,
            engine,
        })
    }

    /// The inner-monologue token ids accumulated so far (undelayed
    /// stream — the raw side of [`S2sDuplexHandle::monologue_text`]).
    #[must_use]
    pub fn monologue_tokens(&self) -> &[u32] {
        &self.monologue
    }

    /// Frames currently waiting to be pulled.
    #[must_use]
    pub fn pending_frames(&self) -> usize {
        self.out_queue.len()
    }

    /// Sets the opt-in output limiter (M4-RESIDUAL-B (C)).
    ///
    /// Applied to each decoded model frame at the decode→queue boundary in
    /// [`S2sDuplexHandle::push_mic_frame`]. The default is [`OutputLimiter::None`]
    /// (bit-identical to the pre-limiter path); [`OutputLimiter::Clamp`] hard-
    /// limits to `[-1, 1]` and [`OutputLimiter::PeakNormalize`] frame-locally
    /// attenuates over-1.0 frames — either guards a PCM16 consumer against the
    /// clip the synthesized-Mimi bridge can produce (peak > 1.0). Builder form so
    /// it composes with [`MoshiEngine::open_duplex_session`] one-liners; the
    /// mutable [`Self::set_output_limiter`] covers a live session.
    #[must_use]
    pub fn with_output_limiter(mut self, limiter: OutputLimiter) -> Self {
        self.limiter = limiter;
        self
    }

    /// The active output limiter.
    #[must_use]
    pub fn output_limiter(&self) -> OutputLimiter {
        self.limiter
    }

    /// Sets the output limiter on a live session ([`Self::with_output_limiter`]
    /// docs). Barge-in reset does not clear it — the limiter is a caller policy,
    /// not generation state.
    pub fn set_output_limiter(&mut self, limiter: OutputLimiter) {
        self.limiter = limiter;
    }

    /// M3-14 acknowledge: flush pending output, reset the generation
    /// stack (ring / paged KV / depformer / Mimi streaming states /
    /// monologue / samplers), clear the flag **last** (Release). The AEC
    /// front and both sample clocks survive (module docs).
    fn acknowledge_interrupt(&mut self) -> Result<()> {
        if !self.interrupt.load(Ordering::Acquire) {
            return Ok(());
        }
        self.out_queue.clear();
        self.gen_state.reset();
        // Mimi streaming states are rebuilt (no in-place reset surface on
        // the shared module — an allocation here is fine, barge-in is not
        // the frame hot path).
        let eng = self.engine.borrow();
        self.enc_state = eng.encoder().state(1)?;
        self.audio_state = eng.chain().state(eng.config().max_ctx)?;
        self.monologue.clear();
        self.first_frame_done = false;
        self.samplers = if self.deterministic {
            MoshiSamplerPair::greedy()
        } else {
            MoshiSamplerPair::stochastic(self.seed)
        };
        self.interrupt.store(false, Ordering::Release);
        Ok(())
    }
}

fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    let sum: f64 = pcm.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    ((sum / pcm.len() as f64) as f32).sqrt()
}

impl<E: std::borrow::Borrow<MoshiEngine>> S2sDuplexHandle for MoshiDuplexSession<E> {
    fn push_mic_frame(&mut self, pcm: &[f32]) -> Result<DuplexPushReport> {
        self.acknowledge_interrupt()?;
        if pcm.len() != self.hop {
            return Err(VokraError::InvalidArgument(format!(
                "moshi duplex: push_mic_frame wants exactly one frame of {} samples, \
                 got {} (buffer whole frames — FR-EX-08)",
                self.hop,
                pcm.len()
            )));
        }
        let raw_rms = rms(pcm);
        let aec_applied = self.aec.is_some();
        if let Some(aec) = &mut self.aec {
            let fs = aec.front.frame_size();
            for (i, chunk) in pcm.chunks_exact(fs).enumerate() {
                let pos = self.mic_pos + (i * fs) as u64;
                aec.front
                    .process_mic(chunk, pos, &mut self.cleaned[i * fs..(i + 1) * fs])?;
            }
        } else {
            self.cleaned.copy_from_slice(pcm);
        }
        self.mic_pos += self.hop as u64;
        let cleaned_rms = rms(&self.cleaned);

        // Mimi encode (streaming state carries the causal tails).
        let eng = self.engine.borrow();
        eng.encoder()
            .encode_into(&mut self.enc_state, &self.cleaned, &mut self.user_codes)?;

        // First-frame convention (module docs): step the first real codes
        // twice so they are seen behind the initial-token substitution.
        if !self.first_frame_done {
            let warmup = eng.model().step_into(
                &mut self.gen_state,
                &self.user_codes,
                &mut self.samplers,
                &mut self.frame_out,
            )?;
            debug_assert!(!warmup, "the double-step's first half is inside warmup");
            self.first_frame_done = true;
        }
        let emitted = eng.model().step_into(
            &mut self.gen_state,
            &self.user_codes,
            &mut self.samplers,
            &mut self.frame_out,
        )?;
        if emitted {
            self.monologue.push(self.frame_out.text);
            eng.chain().decode_frame_into(
                &mut self.audio_state,
                &self.frame_out.audio,
                &mut self.frame_pcm,
            )?;
            // Opt-in output limiter (default None = no-op = bit-identical, so
            // the shipping path is unchanged — FR-EX-08).
            self.limiter.apply(&mut self.frame_pcm);
            self.out_queue.push_back(self.frame_pcm.clone());
        }
        Ok(DuplexPushReport {
            step_emitted: emitted,
            aec_applied,
            raw_rms,
            cleaned_rms,
        })
    }

    fn pull_model_frame(&mut self) -> Result<Option<Vec<f32>>> {
        self.acknowledge_interrupt()?;
        let Some(pcm) = self.out_queue.pop_front() else {
            return Ok(None);
        };
        // Pulling is the playback hand-off: stamp the far-end reference
        // now (monotone sample tags; the owner-tunable offset compensates
        // real playback latency — T17).
        if let Some(aec) = &mut self.aec {
            aec.writer
                .push(&pcm, self.play_pos + self.playback_offset_samples)?;
        }
        self.play_pos += pcm.len() as u64;
        Ok(Some(pcm))
    }

    fn monologue_text(&self) -> Result<String> {
        decode_monologue(self.eng().tokenizer(), self.eng().config(), &self.monologue)
    }

    fn interrupt_handle(&self) -> DuplexInterruptHandle {
        DuplexInterruptHandle::new(Arc::clone(&self.interrupt))
    }

    fn warnings(&self) -> &[String] {
        &self.warnings
    }

    fn frame_hop(&self) -> usize {
        self.hop
    }

    fn sample_rate(&self) -> u32 {
        self.eng().mimi_config().sample_rate
    }
}
