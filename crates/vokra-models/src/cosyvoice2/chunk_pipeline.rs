//! CosyVoice2 chunk-aware streaming pipeline (M3-09-T12 / T14 partial).
//!
//! Composes the three M3 building blocks the CosyVoice2 native model needs
//! for real-time streaming synthesis:
//!
//! 1. [`vokra_ops::length_conditioning`] (M3-08) — resolves the caller's
//!    duration hint (mode A: user-specified seconds/frames or mode B:
//!    linear estimate from a reference utterance) to a `target_frames`
//!    count. Flow Matching CFM works in one shot over the full utterance,
//!    so the target length is set outside the sampler graph (FR-EX-10 --
//!    the sampler is a runtime function, not a graph node).
//! 2. [`ChunkAwareCfm::run_chunks`] (M3-09-T10/T11) — iterates the flow
//!    sampler across `n_chunks` boundaries, feeding a caller-supplied
//!    velocity closure the previous chunk's terminal state as
//!    [`ChunkContinuation`] context. The closure is what a real
//!    CosyVoice2 LLM backbone (T07/T08) will fill in; today the pipeline
//!    is exercised through an injected closure for internal-oracle tests.
//! 3. [`MimiBridge::decode_chunk`] (M3-09-T13) — decodes each chunk's
//!    output codes (`[chunk_size, n_codebooks]`) into a `[chunk_size,
//!    d_model]` feature buffer via the M3-06 [`vokra_ops::MimiDecoder`].
//!    An identity fixture decoder is the internal-oracle path today; the
//!    T13 follow-on binds a real Mimi codebook tensor slice.
//!
//! # What this session lands
//!
//! [`ChunkAwareStreamingPipeline`] — the composition layer. It takes:
//!
//! - a [`CosyVoice2Config`] (chunk_size / chunk_hop / mimi.* / streaming.*
//!   read at load time),
//! - a [`ChunkAwareCfm`] instance (Flow Matching driver),
//! - a [`MimiBridge`] instance (either identity-decoder or real T13),
//! - a [`LengthConditioningInput`] (mode A or B).
//!
//! and produces a `Vec<PipelineChunk>` — one per chunk boundary — carrying
//! the terminal Flow state, the codes rendered from that state via the
//! caller-supplied code closure, and the Mimi feature buffer.
//!
//! # Not yet in this session
//!
//! - **Real LLM velocity closure** (T07/T08) — the pipeline takes an
//!   injected closure so internal-oracle tests can validate the plumbing.
//! - **Real Mimi checkpoint decoder** (T13 real-checkpoint) — identity
//!   fixture only.
//! - **Real code sampling** (T10 detailed CFM head) — the pipeline exposes
//!   a code closure that maps a terminal state to `[chunk_size,
//!   n_codebooks]` `u32` indices; the internal-oracle tests use a
//!   deterministic hash so chunk boundaries are testable.
//! - **`vokra-eval` MEL/UTMOS gate** (T23) — depends on real audio, which
//!   depends on the LLM velocity closure.
//! - **HTTP / gRPC streaming API surface** (T15/T28) — the SPSC ring
//!   plumbing is out-of-scope for this session; the pipeline returns a
//!   full `Vec<PipelineChunk>` which the streaming layer wraps.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Every stage validates its shape/attr contract on entry:
//!
//! - `length_conditioning` rejects a negative / non-finite duration and
//!   zero sample_rate / hop_length when converting seconds;
//! - `run_chunks` rejects `n_chunks == 0` and a closure that returns a
//!   different-shaped state;
//! - `MimiBridge::decode_chunk` rejects a codes buffer whose length does
//!   not match `time · n_codebooks`.
//!
//! The pipeline additionally rejects `chunk_size == 0`, `chunk_hop == 0`,
//! and a mismatched code closure output.

use vokra_core::ir::graph::LengthConditioningAttrs;
use vokra_core::{Result, VokraError};
use vokra_ops::{FlowSamplerState, ForwardPass, length_conditioning};

use super::config::CosyVoice2Config;
use super::flow_matching::{ChunkAwareCfm, ChunkContinuation};
use super::mimi_bridge::MimiBridge;

/// One chunk's output of the streaming pipeline.
#[derive(Debug, Clone)]
pub struct PipelineChunk {
    /// Zero-based chunk index (`0..n_chunks`).
    pub chunk_index: usize,
    /// Terminal state of Flow Matching for this chunk (`t = 1.0`).
    pub terminal_state: FlowSamplerState,
    /// Codes rendered from `terminal_state` by the pipeline's `code_fn`
    /// closure, shape `[chunk_frames, n_codebooks]` row-major.
    ///
    /// The last chunk's `chunk_frames` may be shorter than
    /// `config.streaming_chunk_size` when the target frame count is not
    /// a multiple of the chunk size.
    pub codes: Vec<u32>,
    /// Number of frames in this chunk's codes (`chunk_frames`).
    pub chunk_frames: usize,
    /// Mimi feature buffer for this chunk, shape
    /// `[chunk_frames, d_model]` row-major.
    pub features: Vec<f32>,
}

/// Streaming pipeline result: one `PipelineChunk` per chunk boundary plus
/// the resolved target frame count.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Resolved target frame count from `length_conditioning`.
    pub target_frames: u32,
    /// One `PipelineChunk` per chunk boundary (`chunks.len() == n_chunks`).
    pub chunks: Vec<PipelineChunk>,
}

/// The composition layer: length_conditioning → flow_matching → mimi_bridge.
///
/// Owns the config + driver + bridge; every synthesis invocation takes an
/// injected velocity closure (M3-09-T07/T08 real path) and code closure
/// (M3-09-T10 detailed CFM head). Internal-oracle tests use simple
/// deterministic closures — no real safetensors checkpoint is invoked.
#[derive(Debug)]
pub struct ChunkAwareStreamingPipeline<'a> {
    config: &'a CosyVoice2Config,
    cfm: &'a ChunkAwareCfm,
    bridge: &'a MimiBridge,
}

impl<'a> ChunkAwareStreamingPipeline<'a> {
    /// Builds the pipeline wrapper. Every field is a borrow so callers can
    /// keep the underlying driver + bridge instances live across
    /// invocations (the T14 streaming session pattern).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.streaming_chunk_size ==
    /// 0` or `config.streaming_chunk_hop == 0`. Bridge shape sanity is
    /// checked by comparing `bridge.attrs()` against `config`'s
    /// `mimi_*` fields (mismatch is a loud error, FR-EX-08).
    pub fn new(
        config: &'a CosyVoice2Config,
        cfm: &'a ChunkAwareCfm,
        bridge: &'a MimiBridge,
    ) -> Result<Self> {
        if config.streaming_chunk_size == 0 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 chunk pipeline: streaming_chunk_size must be non-zero \
                 (FR-EX-08 — no silent skip)"
                    .to_owned(),
            ));
        }
        if config.streaming_chunk_hop == 0 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 chunk pipeline: streaming_chunk_hop must be non-zero \
                 (FR-EX-08 — no silent skip)"
                    .to_owned(),
            ));
        }
        let bridge_attrs = bridge.attrs();
        if bridge_attrs.n_codebooks != config.mimi_n_codebooks as usize
            || bridge_attrs.codebook_size != config.mimi_codebook_size as usize
            || bridge_attrs.d_model != config.mimi_d_model as usize
        {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 chunk pipeline: bridge attrs {bridge_attrs:?} disagree with \
                 config (n_codebooks={}, codebook_size={}, d_model={})",
                config.mimi_n_codebooks, config.mimi_codebook_size, config.mimi_d_model
            )));
        }
        Ok(Self {
            config,
            cfm,
            bridge,
        })
    }

    /// The number of chunks needed to cover `target_frames` frames at
    /// `chunk_size` boundaries — `ceil(target_frames / chunk_size)`.
    ///
    /// A `target_frames = 0` produces `n_chunks = 0` (an empty synthesis
    /// is a valid degenerate case; the caller decides whether to reject
    /// it downstream).
    #[must_use]
    pub fn n_chunks_for(&self, target_frames: u32) -> usize {
        let cs = self.config.streaming_chunk_size as usize;
        let tf = target_frames as usize;
        tf.div_ceil(cs)
    }

    /// Runs the full pipeline: resolves target frames via `length_input`,
    /// iterates the Flow Matching driver over the resulting chunks with
    /// `velocity_fn`, renders codes via `code_fn`, and Mimi-decodes each
    /// chunk into features.
    ///
    /// # Arguments
    ///
    /// - `length_input` — the M3-08 length_conditioning input (mode A / B).
    /// - `initial_state` — Flow Matching starting state for the first
    ///   chunk. Shape is preserved across all chunks (FR-EX-08 — a
    ///   closure that reshapes triggers an explicit error).
    /// - `velocity_fn` — the caller-supplied velocity closure (real LLM
    ///   forward closure lands with T07/T08; internal-oracle tests use
    ///   identity/decay/…).
    /// - `code_fn` — the caller-supplied "state → codes" mapper. Given
    ///   the chunk's terminal state + chunk_frames + `n_codebooks`,
    ///   returns `[chunk_frames · n_codebooks]` `u32` indices. Real
    ///   CosyVoice2 samples codes from the LLM output; internal-oracle
    ///   tests use a deterministic hash so chunk boundaries are testable.
    ///
    /// # Errors
    ///
    /// Propagates every downstream error verbatim
    /// ([`length_conditioning`], [`ChunkAwareCfm::run_chunks`],
    /// [`MimiBridge::decode_chunk`]) and additionally:
    /// - `VokraError::InvalidArgument` if `code_fn` returns a wrongly-shaped
    ///   codes vector for any chunk (FR-EX-08).
    pub fn synthesize<V, C>(
        &self,
        length_input: LengthConditioningAttrs,
        initial_state: &FlowSamplerState,
        velocity_fn: V,
        mut code_fn: C,
    ) -> Result<PipelineOutput>
    where
        V: FnMut(
            &FlowSamplerState,
            f32,
            ForwardPass,
            &ChunkContinuation<'_>,
        ) -> Result<FlowSamplerState>,
        C: FnMut(&FlowSamplerState, usize, usize) -> Result<Vec<u32>>,
    {
        let target_frames = length_conditioning(&length_input)?;
        let n_chunks = self.n_chunks_for(target_frames);
        if n_chunks == 0 {
            // Degenerate but well-defined: zero-frame target ↔ zero chunks.
            // We return an empty PipelineOutput rather than error out —
            // the M3-08 op itself accepts zero-frame targets so we mirror
            // that contract here.
            return Ok(PipelineOutput {
                target_frames,
                chunks: Vec::new(),
            });
        }

        // Run the Flow Matching driver across all chunks.
        let terminals = self.cfm.run_chunks(initial_state, n_chunks, velocity_fn)?;
        debug_assert_eq!(terminals.len(), n_chunks);

        // Render codes + Mimi features per chunk, honouring the "last
        // chunk may be shorter" invariant.
        let chunk_size = self.config.streaming_chunk_size as usize;
        let n_codebooks = self.config.mimi_n_codebooks as usize;
        let target_usize = target_frames as usize;
        let mut chunks: Vec<PipelineChunk> = Vec::with_capacity(n_chunks);
        for (chunk_index, terminal_state) in terminals.into_iter().enumerate() {
            let chunk_start = chunk_index * chunk_size;
            let chunk_end = target_usize.min(chunk_start + chunk_size);
            let chunk_frames = chunk_end - chunk_start;
            let codes = code_fn(&terminal_state, chunk_frames, n_codebooks)?;
            let expected_codes = chunk_frames * n_codebooks;
            if codes.len() != expected_codes {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 chunk pipeline: chunk {chunk_index} code_fn returned {} \
                     codes, expected chunk_frames·n_codebooks = {expected_codes}",
                    codes.len()
                )));
            }
            let features = self.bridge.decode_chunk(&codes, chunk_frames)?;
            chunks.push(PipelineChunk {
                chunk_index,
                terminal_state,
                codes,
                chunk_frames,
                features,
            });
        }
        Ok(PipelineOutput {
            target_frames,
            chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn stub_config(
        chunk_size: u32,
        chunk_hop: u32,
        n_codebooks: u32,
        codebook_size: u32,
        d_model: u32,
        nfe: u32,
    ) -> CosyVoice2Config {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(super::super::config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(super::super::config::KEY_VOCAB_SIZE, 32);
        b.add_u32(super::super::config::KEY_HIDDEN_DIM, 16);
        b.add_u32(super::super::config::KEY_N_LAYER, 2);
        b.add_u32(super::super::config::KEY_N_HEAD, 2);
        b.add_u32(super::super::config::KEY_FFN_DIM, 32);
        b.add_u32(super::super::config::KEY_FLOW_NFE, nfe);
        b.add_string(super::super::config::KEY_FLOW_SCHEDULE, "linear");
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, n_codebooks);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, codebook_size);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, d_model);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, chunk_size);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, chunk_hop);
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        CosyVoice2Config::from_gguf(&file).expect("read")
    }

    /// Zero velocity → each chunk's terminal is the chunk's initial state
    /// (Euler on `dx/dt = 0` is identity), giving a predictable driver
    /// trajectory to reason about.
    fn zero_velocity(
        s: &FlowSamplerState,
        _t: f32,
        _p: ForwardPass,
        _cont: &ChunkContinuation<'_>,
    ) -> Result<FlowSamplerState> {
        Ok(FlowSamplerState {
            shape: s.shape.clone(),
            data: vec![0.0; s.data.len()],
        })
    }

    /// Deterministic code closure: returns codes = 1 everywhere so the
    /// identity Mimi decoder produces a predictable feature buffer
    /// (col 1 = n_codebooks, else 0 — see MimiDecoder::identity).
    fn constant_ones_codes(
        _s: &FlowSamplerState,
        chunk_frames: usize,
        n_codebooks: usize,
    ) -> Result<Vec<u32>> {
        Ok(vec![1u32; chunk_frames * n_codebooks])
    }

    // ---- Constructor validation ----------------------------------------

    #[test]
    fn constructor_rejects_zero_chunk_size() {
        let cfg = stub_config(0, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let err = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge)
            .expect_err("chunk_size=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn constructor_rejects_zero_chunk_hop() {
        let cfg = stub_config(4, 0, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let err = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge)
            .expect_err("chunk_hop=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- n_chunks_for -----------------------------------------------------

    #[test]
    fn n_chunks_for_ceiling_division() {
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        // chunk_size = 4
        assert_eq!(p.n_chunks_for(0), 0);
        assert_eq!(p.n_chunks_for(1), 1);
        assert_eq!(p.n_chunks_for(4), 1);
        assert_eq!(p.n_chunks_for(5), 2);
        assert_eq!(p.n_chunks_for(12), 3);
    }

    // ---- synthesize: end-to-end oracle ----------------------------------

    #[test]
    fn synthesize_full_pipeline_produces_expected_chunk_count() {
        // target = 10 frames at chunk_size=4 → 3 chunks: [4, 4, 2].
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(10.0);
        let x0 = FlowSamplerState::new(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .expect("synthesize");
        assert_eq!(out.target_frames, 10, "target_frames from length_input");
        assert_eq!(out.chunks.len(), 3, "n_chunks");
        // Last chunk shorter than chunk_size.
        assert_eq!(out.chunks[0].chunk_frames, 4);
        assert_eq!(out.chunks[1].chunk_frames, 4);
        assert_eq!(out.chunks[2].chunk_frames, 2, "last chunk = remainder");
    }

    #[test]
    fn synthesize_features_shape_is_chunk_frames_by_d_model() {
        // Every chunk's features must be shape [chunk_frames, d_model]
        // row-major (the Mimi output shape contract).
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(10.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .unwrap();
        let d_model = cfg.mimi_d_model as usize;
        for c in &out.chunks {
            assert_eq!(
                c.features.len(),
                c.chunk_frames * d_model,
                "chunk {} features [chunk_frames={}, d_model={}]",
                c.chunk_index,
                c.chunk_frames,
                d_model
            );
        }
    }

    #[test]
    fn synthesize_features_match_identity_decoder_invariant() {
        // Identity decoder: every codebook row i = one-hot at col (i mod d_model),
        // so codes=1 → each timestep sums n_codebooks one-hots at col 1.
        // Expected feature row = [0, n_codebooks, 0, 0, ...].
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(4.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .unwrap();
        assert_eq!(out.chunks.len(), 1);
        let d_model = cfg.mimi_d_model as usize;
        let n_cb = cfg.mimi_n_codebooks as f32;
        for t in 0..4 {
            let base = t * d_model;
            // col 1 == n_codebooks, others == 0.
            assert_eq!(
                out.chunks[0].features[base + 1],
                n_cb,
                "t={t} col=1 must sum to n_codebooks"
            );
            for d in 0..d_model {
                if d == 1 {
                    continue;
                }
                assert_eq!(
                    out.chunks[0].features[base + d],
                    0.0,
                    "t={t} col={d} must be 0"
                );
            }
        }
    }

    #[test]
    fn synthesize_zero_target_frames_returns_empty_chunks() {
        // target = 0 → n_chunks = 0 → empty PipelineOutput. The M3-08 op
        // accepts a zero-frame target as a valid degenerate case; the
        // pipeline mirrors that contract.
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(0.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .unwrap();
        assert_eq!(out.target_frames, 0);
        assert_eq!(out.chunks.len(), 0);
    }

    #[test]
    fn synthesize_ref_linear_mode_uses_length_conditioning_op() {
        // Mode B: ref_speech_frames=8, text_ratio=1.5 → target = 12 frames
        // → n_chunks = 3 at chunk_size=4.
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::ref_linear(8, 1.5);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .unwrap();
        assert_eq!(out.target_frames, 12, "ref_linear: 8 · 1.5 = 12");
        assert_eq!(out.chunks.len(), 3);
    }

    #[test]
    fn synthesize_code_closure_wrong_length_fails_loudly() {
        // A code closure that returns the wrong number of codes must not
        // be silently truncated / padded (FR-EX-08).
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(4.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let err = p
            .synthesize(
                length_input,
                &x0,
                zero_velocity,
                |_s, _chunk_frames, _n_cb| {
                    // Wrong length: return a single code no matter the ask.
                    Ok(vec![0u32])
                },
            )
            .expect_err("bad code_fn length must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn synthesize_propagates_velocity_closure_error() {
        // A closure that returns an error must not be swallowed (FR-EX-08).
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(4.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let err = p
            .synthesize(
                length_input,
                &x0,
                |_s, _t, _p, _c| {
                    Err(VokraError::NotImplemented(
                        "test: velocity closure refuses to run",
                    ))
                },
                constant_ones_codes,
            )
            .expect_err("velocity error must propagate");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn synthesize_bridge_without_decoder_returns_not_implemented() {
        // The pipeline itself does not require a decoder — the error
        // surfaces from MimiBridge::decode_chunk when it's actually called.
        // FR-EX-08: a bridge without a bound decoder returns
        // NotImplemented rather than a silent zero-fill feature buffer.
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::from_config(&cfg).expect("build"); // no decoder
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(4.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let err = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .expect_err("no bound decoder must fail");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn synthesize_length_conditioning_rejects_bad_input() {
        // FR-EX-08: a negative duration must not be silently clamped to
        // zero. `length_conditioning` catches it up front and the
        // pipeline propagates.
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(-1.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let err = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .expect_err("negative duration must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn synthesize_chunk_indices_are_monotone_zero_indexed() {
        // The PipelineChunk.chunk_index emitted by the pipeline must be
        // 0, 1, 2, ... — mirrors the ChunkContinuation.chunk_index the
        // driver hands the velocity closure.
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(10.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .unwrap();
        let indices: Vec<usize> = out.chunks.iter().map(|c| c.chunk_index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn synthesize_features_range_finite_and_non_negative_for_identity_decoder() {
        // The identity decoder produces features in {0.0, n_codebooks}.
        // Both are finite; nothing NaN / infinity should ever leak (the
        // "no silent NaN" invariant every audio op preserves).
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&cfg).unwrap();
        let p = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge).unwrap();
        let length_input = LengthConditioningAttrs::user_specified_frames(8.0);
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = p
            .synthesize(length_input, &x0, zero_velocity, constant_ones_codes)
            .unwrap();
        for c in &out.chunks {
            for &v in &c.features {
                assert!(
                    v.is_finite(),
                    "chunk {} feature must be finite",
                    c.chunk_index
                );
                assert!(v >= 0.0, "identity decoder features are non-negative");
            }
        }
    }

    #[test]
    fn bridge_config_mismatch_fails_at_constructor() {
        // A bridge built from a *different* config's mimi_* shape must
        // not silently accept the pipeline's config (FR-EX-08). Build
        // two configs that disagree on d_model and confirm the pipeline
        // constructor fails loudly.
        let cfg = stub_config(4, 4, 2, 8, 4, 2);
        let other_cfg = stub_config(4, 4, 2, 8, 8, 2); // different d_model
        let cfm = ChunkAwareCfm::new(cfg.clone()).unwrap();
        let bridge = MimiBridge::with_identity_decoder(&other_cfg).unwrap();
        let err = ChunkAwareStreamingPipeline::new(&cfg, &cfm, &bridge)
            .expect_err("mismatched bridge must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
