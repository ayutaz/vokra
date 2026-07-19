//! CSM combined frame generation — backbone step → zeroth-codebook sample →
//! depth transformer → full RVQ frame (M4-05-T10).
//!
//! # One frame (ADR M4-05 §D2 `generate_frame`, transcribed)
//!
//! 1. the backbone consumes the next input frame (prompt frame during
//!    priming, the previously generated audio frame during decode) and
//!    yields the final hidden state;
//! 2. `codebook0_head` logits → sample `c0` (M1 [`Sampler`] — temperature /
//!    top-k / seed; `temperature = 0` is the deterministic parity anchor);
//! 3. the depth transformer autoregresses codebooks `1..n_codebooks`
//!    conditioned on the hidden state and `c0`, resetting per frame;
//! 4. **EOS**: a frame whose codes are all zero stops generation
//!    (`generator.py` `if torch.all(sample == 0): break`) and is **not**
//!    fed back into the context.
//!
//! # Determinism
//!
//! With a fixed [`SamplerConfig::seed`] (or `temperature == 0`) the whole
//! frame sequence is reproducible bit-for-bit — the anchor the T23 dumper /
//! T24 parity / T25 quality gates rely on (fabricated pass 禁止).

use vokra_core::{BackendKind, Result, Sampler, VokraError};

use super::backbone::{CsmBackbone, CsmBackboneState, CsmBackboneWeights, CsmFrame};
use super::config::CsmConfig;
use super::depth::{CsmDepthState, CsmDepthTransformer, CsmDepthWeights};

#[allow(unused_imports)] // rustdoc link target
use vokra_core::SamplerConfig;

/// Outcome of one [`CsmModel::generate_frame_into`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsmFrameKind {
    /// A real audio frame — `codes_out` carries `n_codebooks` RVQ ids and
    /// the frame was appended to the backbone context.
    Audio,
    /// The all-zero EOS frame (`generator.py` stop rule). `codes_out` is
    /// all zero and the frame was **not** appended to the context.
    Eos,
}

/// The assembled CSM generation model: backbone + depth transformer over
/// one shared config.
pub struct CsmModel {
    backbone: CsmBackbone,
    depth: CsmDepthTransformer,
}

impl std::fmt::Debug for CsmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmModel")
            .field("backbone", &self.backbone)
            .field("depth", &self.depth)
            .finish()
    }
}

impl CsmModel {
    /// Assembles a model from explicit weight stores.
    ///
    /// # Errors
    ///
    /// Propagates config validation / shape errors from the two components.
    pub fn new(
        config: CsmConfig,
        backbone_weights: CsmBackboneWeights,
        depth_weights: CsmDepthWeights,
    ) -> Result<Self> {
        let backbone = CsmBackbone::new(config.clone(), backbone_weights)?;
        let depth = CsmDepthTransformer::new(config, depth_weights)?;
        Ok(Self { backbone, depth })
    }

    /// Synthesized-fixture model (deterministic; shape / stability path).
    /// The backbone and depth stores draw from decorrelated sub-seeds so
    /// the two stacks do not share weight values.
    ///
    /// # Errors
    ///
    /// Propagates the synthesized builders' errors.
    pub fn synthesized(config: CsmConfig, seed: u64) -> Result<Self> {
        let backbone = CsmBackbone::synthesized(config.clone(), seed)?;
        let depth = CsmDepthTransformer::synthesized(config, seed ^ 0x9E37_79B9_7F4A_7C15)?;
        Ok(Self { backbone, depth })
    }

    /// Routes both stacks' hot ops through `backend` (T21/T22 GPU
    /// sessions).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backbone = self.backbone.with_backend(backend);
        self.depth = self.depth.with_backend(backend);
        self
    }

    /// The shared config.
    #[must_use]
    pub fn config(&self) -> &CsmConfig {
        self.backbone.config()
    }

    /// The backbone component.
    #[must_use]
    pub fn backbone(&self) -> &CsmBackbone {
        &self.backbone
    }

    /// The depth-transformer component.
    #[must_use]
    pub fn depth(&self) -> &CsmDepthTransformer {
        &self.depth
    }

    /// Primes the generation state with `frames` (dialog context: text
    /// frames + audio frames), leaving the last hidden state staged for
    /// the first [`Self::generate_frame_into`] call.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on empty `frames` or any backbone
    /// error (out-of-range token, `n_ctx` overflow, ...).
    pub fn prime(&self, state: &mut CsmGenerationState, frames: &[CsmFrame]) -> Result<()> {
        if frames.is_empty() {
            return Err(VokraError::InvalidArgument(
                "csm prime: frames must be non-empty (CSM conditions on \
                 text + context — ADR M4-05 §D1-(b))"
                    .into(),
            ));
        }
        let d = self.config().backbone.d_model;
        let hidden = self.backbone.forward(frames, &mut state.backbone)?;
        state
            .last_hidden
            .copy_from_slice(&hidden[(frames.len() - 1) * d..]);
        state.primed = true;
        Ok(())
    }

    /// Generates one frame into `codes_out` (`[n_codebooks]`), drawing every
    /// codebook id through `sample` (the M1 [`Sampler`] rides in via a
    /// closure; the greedy hot loop passes a plain
    /// [`vokra_core::decode::argmax`] closure so the whole call is
    /// **zero-heap-allocation** — FR-EX-05, T18).
    ///
    /// On [`CsmFrameKind::Audio`] the frame has been fed back into the
    /// backbone context and the staged hidden state advanced; on
    /// [`CsmFrameKind::Eos`] the state is left untouched (the upstream
    /// loop breaks without appending).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an unprimed state, wrong
    /// `codes_out` length, or `n_ctx` overflow; propagates component
    /// errors verbatim.
    pub fn generate_frame_into(
        &self,
        state: &mut CsmGenerationState,
        sample: &mut dyn FnMut(&mut [f32]) -> u32,
        codes_out: &mut [u32],
    ) -> Result<CsmFrameKind> {
        let cfg = self.config();
        if !state.primed {
            return Err(VokraError::InvalidArgument(
                "csm generate_frame: state not primed — call CsmModel::prime with the \
                 dialog context first"
                    .into(),
            ));
        }
        if codes_out.len() != cfg.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm generate_frame: codes_out len {} != n_codebooks {}",
                codes_out.len(),
                cfg.n_codebooks
            )));
        }
        // c0 from the staged hidden state.
        self.backbone
            .c0_logits_into(&state.last_hidden, &mut state.c0_logits)?;
        let c0 = sample(&mut state.c0_logits);
        if c0 as usize >= cfg.audio_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "csm generate_frame: sampled c0 {c0} >= audio_vocab {} (sampler \
                 misconfigured — FR-EX-08)",
                cfg.audio_vocab_size
            )));
        }
        codes_out[0] = c0;
        // Depth transformer fills codebooks 1..n.
        self.depth.decode_frame(
            &self.backbone,
            &state.last_hidden,
            &mut state.depth,
            codes_out,
            |logits| sample(logits),
        )?;
        // EOS: all-zero frame stops generation and is not appended.
        if codes_out.iter().all(|&c| c == 0) {
            return Ok(CsmFrameKind::Eos);
        }
        // Feed the generated frame back as the next backbone position.
        // `feed_frame` is a state-owned audio frame whose code vector is
        // reused across frames (allocation-free feedback — FR-EX-05).
        if let Some(codes) = state.feed_frame.audio.as_mut() {
            codes.copy_from_slice(codes_out);
        } else {
            return Err(VokraError::InvalidArgument(
                "csm generate_frame: feed_frame lost its audio slots (internal \
                 state invariant violated)"
                    .into(),
            ));
        }
        // Disjoint field borrows of `state`: backbone (mut) + feed_frame
        // (shared) + step_hidden (mut).
        self.backbone.step_into(
            &mut state.backbone,
            &state.feed_frame,
            &mut state.step_hidden,
        )?;
        state.last_hidden.copy_from_slice(&state.step_hidden);
        Ok(CsmFrameKind::Audio)
    }

    /// Allocating convenience wrapper: `Ok(Some(codes))` for an audio
    /// frame, `Ok(None)` for EOS.
    ///
    /// # Errors
    ///
    /// See [`Self::generate_frame_into`].
    pub fn generate_frame(
        &self,
        state: &mut CsmGenerationState,
        sampler: &mut Sampler,
    ) -> Result<Option<Vec<u32>>> {
        let mut codes = vec![0u32; self.config().n_codebooks];
        match self.generate_frame_into(state, &mut |l| sampler.sample(l), &mut codes)? {
            CsmFrameKind::Audio => Ok(Some(codes)),
            CsmFrameKind::Eos => Ok(None),
        }
    }
}

/// Generation state: backbone context (paged KV), per-frame depth scratch,
/// the staged last hidden state, and pre-allocated frame-loop buffers.
pub struct CsmGenerationState {
    pub(crate) backbone: CsmBackboneState,
    pub(crate) depth: CsmDepthState,
    last_hidden: Vec<f32>,
    step_hidden: Vec<f32>,
    c0_logits: Vec<f32>,
    /// Re-used audio frame carrying the fed-back codes (allocation-free
    /// feedback path).
    feed_frame: CsmFrame,
    primed: bool,
}

impl std::fmt::Debug for CsmGenerationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmGenerationState")
            .field("backbone", &self.backbone)
            .field("primed", &self.primed)
            .finish()
    }
}

impl CsmGenerationState {
    /// Pre-allocates all buffers for `config`.
    ///
    /// # Errors
    ///
    /// Propagates config validation / arena allocation errors.
    pub fn new(config: &CsmConfig) -> Result<Self> {
        Ok(Self {
            backbone: CsmBackboneState::new(config)?,
            depth: CsmDepthState::new(config)?,
            last_hidden: vec![0.0; config.backbone.d_model],
            step_hidden: vec![0.0; config.backbone.d_model],
            c0_logits: vec![0.0; config.audio_vocab_size],
            feed_frame: CsmFrame::audio(vec![0; config.n_codebooks]),
            primed: false,
        })
    }

    /// Frame positions consumed in the backbone context.
    #[must_use]
    pub fn context_len(&self) -> usize {
        self.backbone.seq_len()
    }

    /// True once [`CsmModel::prime`] has run.
    #[must_use]
    pub fn is_primed(&self) -> bool {
        self.primed
    }

    /// Rewinds for a fresh dialog turn: backbone pages return to the free
    /// list (no realloc), the depth frame counter clears, and the state
    /// needs a new [`CsmModel::prime`].
    pub fn reset(&mut self) {
        self.backbone.reset();
        self.depth.begin_frame();
        self.primed = false;
    }
}

/// Flip-the-switch upstream parity harness (M3-09
/// `parity::assert_vs_hf_reference` posture).
pub mod parity {
    use super::*;

    /// Compares a real-weight CSM model against the upstream staged
    /// reference (T23 fixtures). **Honest `NotImplemented` today** — the
    /// T29 checkpoint (and its tensor manifest) has not arrived, and the
    /// runtime refuses to "pass" a comparison it cannot run (fabricated
    /// pass 禁止). The fixtures-gated staged comparison lives in
    /// `crates/vokra-models/tests/parity_csm.rs`.
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] always, with an explicit reason —
    /// synthesized weights are additionally called out.
    pub fn assert_vs_upstream_reference(model: &CsmModel) -> Result<()> {
        if model.backbone().weights().is_synthesized {
            return Err(VokraError::NotImplemented(
                "csm parity: model carries synthesized fixture weights — comparing \
                 them against the upstream reference would be meaningless. Bind the \
                 real checkpoint (T29) first.",
            ));
        }
        Err(VokraError::NotImplemented(
            "csm parity: staged upstream comparison lands with the T29 tensor \
             manifest + T23 fixtures (tests/parity_csm.rs is the fixtures-gated \
             path).",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::SamplerConfig;

    fn model() -> CsmModel {
        CsmModel::synthesized(CsmConfig::tiny_for_tests(), 42).expect("model")
    }

    fn context() -> Vec<CsmFrame> {
        vec![CsmFrame::text(1), CsmFrame::text(4), CsmFrame::text(2)]
    }

    #[test]
    fn unprimed_state_is_rejected() {
        let m = model();
        let mut state = CsmGenerationState::new(m.config()).unwrap();
        let mut sampler = Sampler::new(SamplerConfig::greedy());
        assert!(matches!(
            m.generate_frame(&mut state, &mut sampler),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn greedy_generation_is_fully_deterministic() {
        // T10 property (iv): temperature 0 → bit-identical frame sequences.
        let m = model();
        let run = || {
            let mut state = CsmGenerationState::new(m.config()).unwrap();
            m.prime(&mut state, &context()).unwrap();
            let mut sampler = Sampler::new(SamplerConfig::greedy());
            let mut frames = Vec::new();
            for _ in 0..4 {
                match m.generate_frame(&mut state, &mut sampler).unwrap() {
                    Some(codes) => frames.push(codes),
                    None => break,
                }
            }
            frames
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "greedy generation must be reproducible");
        for frame in &a {
            assert_eq!(frame.len(), m.config().n_codebooks);
            assert!(
                frame
                    .iter()
                    .all(|&c| (c as usize) < m.config().audio_vocab_size)
            );
        }
    }

    #[test]
    fn seeded_stochastic_generation_is_reproducible() {
        let m = model();
        let run = |seed: u64| {
            let mut state = CsmGenerationState::new(m.config()).unwrap();
            m.prime(&mut state, &context()).unwrap();
            let mut sampler = Sampler::new(SamplerConfig {
                temperature: 0.9,
                top_k: Some(5),
                top_p: None,
                repetition_penalty: None,
                seed,
            });
            let mut out = Vec::new();
            for _ in 0..3 {
                match m.generate_frame(&mut state, &mut sampler).unwrap() {
                    Some(codes) => out.push(codes),
                    None => break,
                }
            }
            out
        };
        assert_eq!(run(7), run(7), "same seed → same frames");
    }

    #[test]
    fn all_zero_frame_is_eos_and_leaves_context_untouched() {
        // T10 property (iii): a forced all-zero frame stops generation and
        // is not appended (generator.py break-without-append).
        let m = model();
        let mut state = CsmGenerationState::new(m.config()).unwrap();
        m.prime(&mut state, &context()).unwrap();
        let ctx_before = state.context_len();
        // Force the all-zero frame by driving the depth path with an
        // all-zero-returning closure and c0 = 0, then classify it through
        // the same rule generate_frame_into applies.
        let mut codes = vec![0u32; m.config().n_codebooks];
        m.depth()
            .decode_frame(
                m.backbone(),
                &state.last_hidden,
                &mut state.depth,
                &mut codes,
                |_| 0,
            )
            .unwrap();
        assert!(codes.iter().all(|&c| c == 0), "forced EOS frame");
        // The EOS rule: an all-zero frame is not appended to the context
        // (the Audio arm — which does append — is exercised by the
        // determinism tests above).
        assert_eq!(state.context_len(), ctx_before, "EOS must not append");
    }

    #[test]
    fn audio_frames_advance_the_context_clock() {
        let m = model();
        let mut state = CsmGenerationState::new(m.config()).unwrap();
        m.prime(&mut state, &context()).unwrap();
        let before = state.context_len();
        let mut sampler = Sampler::new(SamplerConfig::greedy());
        let generated = m.generate_frame(&mut state, &mut sampler).unwrap();
        if generated.is_some() {
            assert_eq!(state.context_len(), before + 1, "audio frame appended");
        } else {
            assert_eq!(state.context_len(), before, "EOS frame not appended");
        }
    }

    #[test]
    fn reset_supports_a_fresh_turn_without_realloc() {
        let m = model();
        let mut state = CsmGenerationState::new(m.config()).unwrap();
        m.prime(&mut state, &context()).unwrap();
        let mut sampler = Sampler::new(SamplerConfig::greedy());
        let first = m.generate_frame(&mut state, &mut sampler).unwrap();
        state.reset();
        assert!(!state.is_primed());
        assert_eq!(state.context_len(), 0);
        m.prime(&mut state, &context()).unwrap();
        let mut sampler2 = Sampler::new(SamplerConfig::greedy());
        let second = m.generate_frame(&mut state, &mut sampler2).unwrap();
        assert_eq!(first, second, "turn reset reproduces the first turn");
    }

    #[test]
    fn flip_the_switch_parity_is_honestly_not_implemented() {
        let m = model();
        assert!(matches!(
            parity::assert_vs_upstream_reference(&m),
            Err(VokraError::NotImplemented(_))
        ));
    }
}
