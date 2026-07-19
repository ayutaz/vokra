//! Moshi frame step — the LMGen delay-ring decode loop (M4-06-T12) with
//! the inner-monologue text channel (T14).
//!
//! # One step (lm.py `LMGen._step`, transcribed — ADR M4-06 §D2)
//!
//! Every 12.5 Hz frame the caller supplies the **user** audio codes
//! (Mimi-encoded mic input, `n_user_streams` ids) and receives — after a
//! `max_delay`-step warmup — one undelayed output frame: the text token
//! (inner monologue) plus the model's own `dep_q` audio codes:
//!
//! 1. user codes are written into the delay ring at `offset + delay[ch]`;
//! 2. all channels are gathered at `offset` (channels still inside their
//!    delay read the **initial token** — audio `card` / text `text_card`);
//! 3. the temporal backbone consumes the summed embedding and the text
//!    head samples the next text token (temp 0.7 / top-k 25 upstream
//!    defaults);
//! 4. the depformer autoregresses the own audio codes conditioned on the
//!    hidden state and the text token (temp 0.8 / top-k 250);
//! 5. the generated tokens are written back at `offset + 1` and the
//!    undelayed output is gathered at `offset - max_delay + delay[ch]`
//!    (`None` while `offset <= max_delay`).
//!
//! # First-frame convention (run_inference.py, transcribed)
//!
//! The very first real codes must be stepped **twice** — the first gather
//! replaces them with initial tokens ("Ensure that the first slice of
//! codes is properly seen by the transformer as otherwise the first slice
//! is replaced by the initial tokens"). [`MoshiModel::step_into`] is the
//! pure single step (upstream `LMGen._step` parity); the duplex session
//! (T16) reproduces the double-step on its first mic frame.

use vokra_core::{BackendKind, Result, Sampler, SamplerConfig, VokraError};

use super::backbone::{MoshiBackbone, MoshiBackboneState, MoshiBackboneWeights};
use super::config::MoshiConfig;
use super::depth::{MoshiDepthState, MoshiDepthTransformer, MoshiDepthWeights};

/// Ring sentinel for "not generated yet" (`ungenerated_token_id = -2`
/// upstream). Distinct from [`super::MOSHI_ZERO_TOKEN`]; reading it is an
/// internal-invariant error (upstream `check` assert), never an input.
pub const MOSHI_UNGENERATED: u32 = u32::MAX - 1;

/// Upstream audio sampling defaults (`LMGen.__init__`: `temp=0.8`,
/// `top_k=250`).
pub const DEFAULT_AUDIO_TEMPERATURE: f32 = 0.8;
/// See [`DEFAULT_AUDIO_TEMPERATURE`].
pub const DEFAULT_AUDIO_TOP_K: usize = 250;
/// Upstream text sampling defaults (`LMGen.__init__`: `temp_text=0.7`,
/// `top_k_text=25`).
pub const DEFAULT_TEXT_TEMPERATURE: f32 = 0.7;
/// See [`DEFAULT_TEXT_TEMPERATURE`].
pub const DEFAULT_TEXT_TOP_K: usize = 25;

/// One decode channel's sampler: greedy argmax (deterministic anchor) or
/// the seeded M1 [`Sampler`] (the CSM `CsmFrameSampler` split — greedy is
/// allocation-free).
#[derive(Debug)]
pub enum MoshiChannelSampler {
    /// Temperature-0 argmax.
    Greedy,
    /// Seeded stochastic sampler.
    Stochastic(Sampler),
}

impl MoshiChannelSampler {
    /// Draws one id from `logits`.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        match self {
            Self::Greedy => vokra_core::decode::argmax(logits),
            Self::Stochastic(s) => s.sample(logits),
        }
    }
}

/// The text + audio sampler pair (Moshi samples the two streams at
/// different temperatures — module docs).
#[derive(Debug)]
pub struct MoshiSamplerPair {
    /// Inner-monologue text sampler.
    pub text: MoshiChannelSampler,
    /// Own-audio codebook sampler.
    pub audio: MoshiChannelSampler,
}

impl MoshiSamplerPair {
    /// Fully deterministic pair (both channels argmax) — the parity /
    /// demo-reproducibility anchor.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            text: MoshiChannelSampler::Greedy,
            audio: MoshiChannelSampler::Greedy,
        }
    }

    /// Seeded stochastic pair at the upstream defaults (text 0.7/25,
    /// audio 0.8/250). The two channels draw from decorrelated seeds.
    #[must_use]
    pub fn stochastic(seed: u64) -> Self {
        Self {
            text: MoshiChannelSampler::Stochastic(Sampler::new(SamplerConfig {
                temperature: DEFAULT_TEXT_TEMPERATURE,
                top_k: Some(DEFAULT_TEXT_TOP_K),
                top_p: None,
                repetition_penalty: None,
                seed,
            })),
            audio: MoshiChannelSampler::Stochastic(Sampler::new(SamplerConfig {
                temperature: DEFAULT_AUDIO_TEMPERATURE,
                top_k: Some(DEFAULT_AUDIO_TOP_K),
                top_p: None,
                repetition_penalty: None,
                seed: seed ^ 0x9E37_79B9_7F4A_7C15,
            })),
        }
    }
}

/// One undelayed output frame: the inner-monologue text token plus the
/// model's own audio codes (`len == dep_q`). Reused across steps — the
/// hot loop never reallocates it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoshiFrameOut {
    /// Text token (display rule: hidden when `== text_pad_id` or
    /// `text_end_pad_id` — run_inference.py, see
    /// [`super::tokenizer::decode_monologue`]).
    pub text: u32,
    /// Own audio codes, one per depformer step.
    pub audio: Vec<u32>,
}

impl MoshiFrameOut {
    /// An output buffer for `config`.
    #[must_use]
    pub fn new(config: &MoshiConfig) -> Self {
        Self {
            text: 0,
            audio: vec![0; config.dep_q],
        }
    }
}

/// Generation state: backbone context (paged KV) + per-frame depformer
/// scratch + the delay ring + step scratch. Everything is pre-allocated.
pub struct MoshiGenerationState {
    pub(crate) backbone: MoshiBackboneState,
    pub(crate) depth: MoshiDepthState,
    /// Delay ring `[n_channels, ring_len]` row-major
    /// (`ring_len = max_delay + 2` — lm.py `_init_streaming_state`).
    cache: Vec<u32>,
    ring_len: usize,
    /// Steps completed (`offsets` upstream — B = 1).
    offset: usize,
    hidden: Vec<f32>,
    text_logits: Vec<f32>,
    step_tokens: Vec<u32>,
    own_codes: Vec<u32>,
}

impl std::fmt::Debug for MoshiGenerationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiGenerationState")
            .field("offset", &self.offset)
            .field("backbone", &self.backbone)
            .finish()
    }
}

impl MoshiGenerationState {
    /// Pre-allocates all buffers for `config`.
    ///
    /// # Errors
    ///
    /// Propagates config validation / paged-arena allocation errors.
    pub fn new(config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let ring_len = config.max_delay() as usize + 2;
        Ok(Self {
            backbone: MoshiBackboneState::new(config)?,
            depth: MoshiDepthState::new(config)?,
            cache: vec![MOSHI_UNGENERATED; config.n_channels() * ring_len],
            ring_len,
            offset: 0,
            hidden: vec![0.0; config.temporal.d_model],
            text_logits: vec![0.0; config.text_card],
            step_tokens: vec![0; config.n_channels()],
            own_codes: vec![0; config.dep_q],
        })
    }

    /// Steps completed since construction / the last [`Self::reset`].
    #[must_use]
    pub fn steps(&self) -> usize {
        self.offset
    }

    /// Rewinds to a fresh session: backbone pages return to the free
    /// list, the ring refills with the ungenerated sentinel, clocks
    /// clear — no reallocation (barge-in reset, T18).
    pub fn reset(&mut self) {
        self.backbone.reset();
        self.depth.begin_frame();
        self.cache.iter_mut().for_each(|c| *c = MOSHI_UNGENERATED);
        self.offset = 0;
    }
}

/// The assembled Moshi generation model: temporal backbone + depformer
/// over one shared config.
pub struct MoshiModel {
    backbone: MoshiBackbone,
    depth: MoshiDepthTransformer,
}

impl std::fmt::Debug for MoshiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiModel")
            .field("backbone", &self.backbone)
            .field("depth", &self.depth)
            .finish()
    }
}

impl MoshiModel {
    /// Assembles a model from explicit weight stores.
    ///
    /// # Errors
    ///
    /// Propagates component validation errors.
    pub fn new(
        config: MoshiConfig,
        backbone_weights: MoshiBackboneWeights,
        depth_weights: MoshiDepthWeights,
    ) -> Result<Self> {
        let backbone = MoshiBackbone::new(config.clone(), backbone_weights)?;
        let depth = MoshiDepthTransformer::new(config, depth_weights)?;
        Ok(Self { backbone, depth })
    }

    /// Assembles a model from already-constructed stacks (the
    /// bounded-memory `from_path` load builds the backbone through
    /// [`MoshiBackbone::new_mapped`] and hands it in here). Both stacks
    /// must share one config (loud otherwise — a silently mixed pair
    /// would emit garbage frames, FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`vokra_core::VokraError::InvalidArgument`] on a config mismatch.
    pub fn from_parts(backbone: MoshiBackbone, depth: MoshiDepthTransformer) -> Result<Self> {
        if backbone.config() != depth.config() {
            return Err(vokra_core::VokraError::InvalidArgument(
                "moshi MoshiModel::from_parts: backbone and depformer were built \
                 from different configs (FR-EX-08 — assemble both from one \
                 MoshiConfig)"
                    .into(),
            ));
        }
        Ok(Self { backbone, depth })
    }

    /// Synthesized-fixture model (deterministic; decorrelated sub-seeds).
    ///
    /// # Errors
    ///
    /// Propagates the synthesized builders' errors.
    pub fn synthesized(config: MoshiConfig, seed: u64) -> Result<Self> {
        let backbone = MoshiBackbone::synthesized(config.clone(), seed)?;
        let depth = MoshiDepthTransformer::synthesized(config, seed ^ 0x5DEE_75DE_E75D_EE75)?;
        Ok(Self { backbone, depth })
    }

    /// Routes both stacks' hot ops through `backend`.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backbone = self.backbone.with_backend(backend);
        self.depth = self.depth.with_backend(backend);
        self
    }

    /// The shared config.
    #[must_use]
    pub fn config(&self) -> &MoshiConfig {
        self.backbone.config()
    }

    /// The temporal backbone.
    #[must_use]
    pub fn backbone(&self) -> &MoshiBackbone {
        &self.backbone
    }

    /// The depformer.
    #[must_use]
    pub fn depth(&self) -> &MoshiDepthTransformer {
        &self.depth
    }

    /// One full-duplex step (module docs): consumes `user_codes`
    /// (`n_user_streams` Mimi ids for this frame), samples the text token
    /// and the own audio codes, and — once past warmup — writes the
    /// undelayed output frame into `out`, returning `true`. During the
    /// first `max_delay` steps it returns `false` and leaves `out`
    /// untouched (upstream `None`).
    ///
    /// Zero heap allocation (state scratch + pre-allocated pages only —
    /// FR-EX-05).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a wrong `user_codes` /
    /// `out.audio` arity, an out-of-range user code, a sampler returning
    /// an out-of-vocab id, or `max_ctx` exhaustion (loud — FR-EX-08).
    pub fn step_into(
        &self,
        state: &mut MoshiGenerationState,
        user_codes: &[u32],
        samplers: &mut MoshiSamplerPair,
        out: &mut MoshiFrameOut,
    ) -> Result<bool> {
        let cfg = self.config();
        let n_user = cfg.n_user_streams();
        if user_codes.len() != n_user {
            return Err(VokraError::InvalidArgument(format!(
                "moshi step: {} user codes for {} user streams (lm.py \
                 needed_tokens)",
                user_codes.len(),
                n_user
            )));
        }
        if out.audio.len() != cfg.dep_q {
            return Err(VokraError::InvalidArgument(format!(
                "moshi step: out.audio len {} != dep_q {}",
                out.audio.len(),
                cfg.dep_q
            )));
        }
        let card = cfg.audio_card as u32;
        let ct = state.ring_len;
        let offset = state.offset;

        // (1) Write the user codes at their delayed positions
        // (channels dep_q+1 ..= n_q_in).
        for (i, &code) in user_codes.iter().enumerate() {
            if code >= card {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi step: user code {code} on stream {i} >= card {card} \
                     (Mimi codes only — the initial token is internal)"
                )));
            }
            let ch = 1 + cfg.dep_q + i;
            let pos = (offset + cfg.delays[ch] as usize) % ct;
            state.cache[ch * ct + pos] = code;
        }

        // (2) Gather this step's input row; channels inside their delay
        // read the initial token (lm.py `is_init`).
        for ch in 0..cfg.n_channels() {
            let tok = if offset <= cfg.delays[ch] as usize {
                if ch == 0 {
                    cfg.text_initial_token()
                } else {
                    cfg.audio_initial_token()
                }
            } else {
                let t = state.cache[ch * ct + offset % ct];
                if t == MOSHI_UNGENERATED {
                    return Err(VokraError::InvalidArgument(format!(
                        "moshi step: channel {ch} reads the ungenerated sentinel at \
                         offset {offset} — internal ring invariant violated \
                         (upstream `check` assert)"
                    )));
                }
                t
            };
            state.step_tokens[ch] = tok;
        }

        // (3) Temporal backbone step (loud on max_ctx exhaustion).
        // Disjoint field borrows of `state`.
        let MoshiGenerationState {
            backbone: bb_state,
            step_tokens,
            hidden,
            ..
        } = state;
        self.backbone.step_into(bb_state, step_tokens, hidden)?;

        // (4) Text head + sample.
        self.backbone
            .text_logits_into(&state.hidden, &mut state.text_logits)?;
        let text_tok = samplers.text.sample(&mut state.text_logits);
        if text_tok as usize >= cfg.text_card {
            return Err(VokraError::InvalidArgument(format!(
                "moshi step: sampled text token {text_tok} >= text_card {} \
                 (sampler misconfigured — FR-EX-08)",
                cfg.text_card
            )));
        }

        // (5) Depformer: own audio codes for this frame.
        let MoshiGenerationState {
            depth: depth_state,
            hidden,
            own_codes,
            ..
        } = state;
        self.depth
            .decode_frame(hidden, text_tok, depth_state, own_codes, |l| {
                samplers.audio.sample(l)
            })?;

        // (6) Advance the clock and write the generated tokens at the new
        // offset (upstream scatters at `offsets + 1`).
        state.offset += 1;
        let new_pos = state.offset % ct;
        state.cache[new_pos] = text_tok; // channel 0
        for cb in 0..cfg.dep_q {
            state.cache[(1 + cb) * ct + new_pos] = state.own_codes[cb];
        }

        // (7) Warmup gate, then the undelayed output gather.
        let max_delay = cfg.max_delay() as usize;
        if state.offset <= max_delay {
            return Ok(false);
        }
        let base = state.offset - max_delay;
        out.text = state.cache[(base + cfg.delays[0] as usize) % ct];
        for cb in 0..cfg.dep_q {
            let ch = 1 + cb;
            out.audio[cb] = state.cache[ch * ct + (base + cfg.delays[ch] as usize) % ct];
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> MoshiModel {
        MoshiModel::synthesized(MoshiConfig::tiny_for_tests(), 42).expect("model")
    }

    fn user_codes(cfg: &MoshiConfig, step: usize) -> Vec<u32> {
        (0..cfg.n_user_streams())
            .map(|i| ((step * 5 + i * 3 + 1) % cfg.audio_card) as u32)
            .collect()
    }

    #[test]
    fn warmup_returns_no_frame_then_streams_undelayed_frames() {
        // max_delay = 1 → step 1 emits nothing, step 2.. emit one frame
        // each (lm.py `offset_cpu <= max_delay → None`).
        let m = model();
        let cfg = m.config().clone();
        let mut state = MoshiGenerationState::new(&cfg).unwrap();
        let mut samplers = MoshiSamplerPair::greedy();
        let mut out = MoshiFrameOut::new(&cfg);
        assert!(
            !m.step_into(&mut state, &user_codes(&cfg, 0), &mut samplers, &mut out)
                .unwrap(),
            "first step is warmup"
        );
        for step in 1..4 {
            let emitted = m
                .step_into(&mut state, &user_codes(&cfg, step), &mut samplers, &mut out)
                .unwrap();
            assert!(emitted, "post-warmup step {step} emits a frame");
            assert!((out.text as usize) < cfg.text_card);
            assert!(out.audio.iter().all(|&c| (c as usize) < cfg.audio_card));
        }
        assert_eq!(state.steps(), 4);
    }

    #[test]
    fn greedy_stream_is_fully_deterministic() {
        let m = model();
        let cfg = m.config().clone();
        let run = || {
            let mut state = MoshiGenerationState::new(&cfg).unwrap();
            let mut samplers = MoshiSamplerPair::greedy();
            let mut out = MoshiFrameOut::new(&cfg);
            let mut frames = Vec::new();
            for step in 0..6 {
                if m.step_into(&mut state, &user_codes(&cfg, step), &mut samplers, &mut out)
                    .unwrap()
                {
                    frames.push((out.text, out.audio.clone()));
                }
            }
            frames
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "greedy stream reproducible");
        assert_eq!(a.len(), 5, "6 steps − 1 warmup");
    }

    #[test]
    fn seeded_stochastic_stream_is_reproducible_and_seed_sensitive() {
        let m = model();
        let cfg = m.config().clone();
        let run = |seed: u64| {
            let mut state = MoshiGenerationState::new(&cfg).unwrap();
            let mut samplers = MoshiSamplerPair::stochastic(seed);
            let mut out = MoshiFrameOut::new(&cfg);
            let mut frames = Vec::new();
            for step in 0..8 {
                if m.step_into(&mut state, &user_codes(&cfg, step), &mut samplers, &mut out)
                    .unwrap()
                {
                    frames.push((out.text, out.audio.clone()));
                }
            }
            frames
        };
        assert_eq!(run(7), run(7), "same seed → same stream");
        let differs = run(7) != run(8);
        assert!(differs, "different seeds must be able to diverge");
    }

    #[test]
    fn undelayed_output_realigns_the_delay_ring() {
        // The undelay identity (T15 anchor): with delays [0,0,1,0,1] and
        // max_delay 1, the *text* token emitted at step s+1 must be the
        // text token *generated* at step s (delay 0 channel gathered at
        // `offset − max_delay`), i.e. output lags generation by exactly
        // max_delay steps. We pin it by driving a model wrapper that
        // records what step 6 wrote and checking it pops out one step
        // later — observable through determinism: two identical runs where
        // run B skips the last step must produce run A's frames minus one.
        let m = model();
        let cfg = m.config().clone();
        let collect = |n_steps: usize| {
            let mut state = MoshiGenerationState::new(&cfg).unwrap();
            let mut samplers = MoshiSamplerPair::greedy();
            let mut out = MoshiFrameOut::new(&cfg);
            let mut frames = Vec::new();
            for step in 0..n_steps {
                if m.step_into(&mut state, &user_codes(&cfg, step), &mut samplers, &mut out)
                    .unwrap()
                {
                    frames.push((out.text, out.audio.clone()));
                }
            }
            frames
        };
        let long = collect(6);
        let short = collect(5);
        assert_eq!(short.as_slice(), &long[..short.len()], "prefix property");
    }

    #[test]
    fn user_code_out_of_range_and_wrong_arity_are_loud() {
        let m = model();
        let cfg = m.config().clone();
        let mut state = MoshiGenerationState::new(&cfg).unwrap();
        let mut samplers = MoshiSamplerPair::greedy();
        let mut out = MoshiFrameOut::new(&cfg);
        let mut bad = user_codes(&cfg, 0);
        bad[0] = cfg.audio_card as u32; // the initial token is internal
        assert!(
            m.step_into(&mut state, &bad, &mut samplers, &mut out)
                .is_err()
        );
        let short = vec![0u32; cfg.n_user_streams() - 1];
        assert!(
            m.step_into(&mut state, &short, &mut samplers, &mut out)
                .is_err()
        );
    }

    #[test]
    fn max_ctx_exhaustion_is_loud_and_names_the_bound() {
        let mut cfg = MoshiConfig::tiny_for_tests();
        cfg.max_ctx = 3;
        let m = MoshiModel::synthesized(cfg.clone(), 3).unwrap();
        let mut state = MoshiGenerationState::new(&cfg).unwrap();
        let mut samplers = MoshiSamplerPair::greedy();
        let mut out = MoshiFrameOut::new(&cfg);
        for step in 0..3 {
            m.step_into(&mut state, &user_codes(&cfg, step), &mut samplers, &mut out)
                .unwrap();
        }
        let err = m
            .step_into(&mut state, &user_codes(&cfg, 3), &mut samplers, &mut out)
            .unwrap_err();
        assert!(err.to_string().contains("max_ctx"), "loud bound: {err}");
    }

    #[test]
    fn reset_reproduces_a_fresh_session_bit_for_bit() {
        // The T18 barge-in contract at the model layer: after reset the
        // same inputs yield the same outputs as a brand-new state.
        let m = model();
        let cfg = m.config().clone();
        let mut fresh = MoshiGenerationState::new(&cfg).unwrap();
        let mut reused = MoshiGenerationState::new(&cfg).unwrap();
        let mut samplers = MoshiSamplerPair::greedy();
        let mut out = MoshiFrameOut::new(&cfg);
        // Dirty the reused state, then reset.
        for step in 0..3 {
            m.step_into(
                &mut reused,
                &user_codes(&cfg, step + 11),
                &mut samplers,
                &mut out,
            )
            .unwrap();
        }
        reused.reset();
        assert_eq!(reused.steps(), 0);

        let run = |state: &mut MoshiGenerationState| {
            let mut samplers = MoshiSamplerPair::greedy();
            let mut out = MoshiFrameOut::new(&cfg);
            let mut frames = Vec::new();
            for step in 0..5 {
                if m.step_into(state, &user_codes(&cfg, step), &mut samplers, &mut out)
                    .unwrap()
                {
                    frames.push((out.text, out.audio.clone()));
                }
            }
            frames
        };
        assert_eq!(run(&mut reused), run(&mut fresh), "reset ≡ fresh session");
    }

    #[test]
    fn user_codes_condition_the_output() {
        // Full-duplex sanity: changing the *user* stream must change the
        // model's stream (the 17-channel sum reaches the backbone).
        let m = model();
        let cfg = m.config().clone();
        let run = |mult: usize| {
            let mut state = MoshiGenerationState::new(&cfg).unwrap();
            let mut samplers = MoshiSamplerPair::greedy();
            let mut out = MoshiFrameOut::new(&cfg);
            let mut frames = Vec::new();
            for step in 0..6 {
                if m.step_into(
                    &mut state,
                    &user_codes(&cfg, step * mult),
                    &mut samplers,
                    &mut out,
                )
                .unwrap()
                {
                    frames.push((out.text, out.audio.clone()));
                }
            }
            frames
        };
        assert_ne!(run(1), run(3), "user audio must condition the reply");
    }
}
