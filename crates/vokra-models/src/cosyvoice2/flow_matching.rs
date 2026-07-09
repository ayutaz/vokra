//! CosyVoice2 Flow Matching CFM (Conditional Flow Matching) — chunk-aware
//! driver (M3-09-T10 / T11 / T14 partial).
//!
//! # Runtime function seam (FR-EX-10)
//!
//! The Flow Matching sampler is [`vokra_ops::flow_sample`] — a **runtime
//! function**, NOT an `OpKind` variant (M3-05 red-line). Every axis
//! (`cfg_mode` / `cfg_scale` / `nfe` / `schedule` / `solver`) is
//! runtime-selectable per invocation; nothing about a sampler choice is
//! baked into the model graph or the GGUF.
//!
//! [`ChunkAwareCfm`] owns the CosyVoice2-side plumbing:
//!
//! - schedule tag ↔ [`vokra_ops::Schedule`] mapping (T04 metadata read at
//!   load time; per-invocation override reserved for CLI `--flow-schedule`);
//! - chunk-aware velocity closure — one chunk's velocity is estimated by the
//!   LLM backbone + Flow Matching sub-network; between-chunk continuity is
//!   carried by [`ChunkAwareCfm::run_chunks`] via a
//!   [`ChunkContinuation`] handle passed to the caller-supplied velocity
//!   closure. That handle is what the T14/T16 streaming pipeline binds a
//!   paged KV cache (M3-03) to when the real LLM backbone lands (T08).
//!
//! # Promotion from scaffold → runnable driver (M3-09 this session)
//!
//! Two entry points now exist:
//!
//! - [`ChunkAwareCfm::step`] — single-chunk sampler (existing scaffold path;
//!   returns [`VokraError::NotImplemented`] when no velocity closure is
//!   supplied so no silent zero-fill can leak, FR-EX-08).
//! - [`ChunkAwareCfm::run_chunks`] — **new**. Iterates over
//!   `n_chunks` boundaries derived from `chunk_size` / `chunk_hop` and
//!   invokes the caller's velocity closure per chunk with a
//!   [`ChunkContinuation`] snapshot (prev chunk's terminal state + a
//!   `chunk_index`). This is the driver the T14 streaming pipeline binds a
//!   real LLM velocity closure to when T07/T08 land; today's internal-
//!   oracle tests use an injected velocity closure to exercise the
//!   plumbing without inventing upstream tensor names.
//!
//! No parity claim is made against the real CosyVoice2 checkpoint — that is
//! the follow-up owner ticket (T21 fixture generation + T22 parity CI).

use vokra_core::{Result, VokraError};
use vokra_ops::{
    CfgMode, CfgScaleProfile, FlowSamplerConfig, FlowSamplerState, ForwardPass, OdeSolver,
    Schedule, flow_sample,
};

use super::config::CosyVoice2Config;

/// Runtime-selectable Flow Matching sampler parameters (FR-EX-10).
///
/// Constructed from the config-derived defaults (T04) plus any caller
/// override (CLI / config file). Every field is runtime-mutable **without
/// re-converting the model** — the M3-05 red-line the sampler enforces.
#[derive(Debug, Clone)]
pub struct FlowMatchingRuntimeParams {
    /// Classifier-free guidance mode.
    pub cfg_mode: CfgMode,
    /// CFG scale profile (constant / dynamic).
    pub cfg_scale: CfgScaleProfile,
    /// Number of function evaluations per chunk.
    pub nfe: usize,
    /// Timestep schedule.
    pub schedule: Schedule,
    /// ODE solver.
    pub solver: OdeSolver,
}

impl FlowMatchingRuntimeParams {
    /// Derives runtime params from a CosyVoice2 config's default axes.
    ///
    /// Currently maps the schedule tag (`vokra.cosyvoice2.flow.schedule`)
    /// via [`Self::schedule_from_tag`]; every other axis takes the
    /// `euler_defaults` starting point from M3-05. A caller who wants
    /// e.g. `CfgMode::SplitBatch + cfg_scale=3.0` overrides after
    /// [`Self::from_config`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the config carries an unknown
    /// schedule tag (FR-EX-08 — no silent fallback to linear).
    pub fn from_config(config: &CosyVoice2Config) -> Result<Self> {
        let schedule = Self::schedule_from_tag(&config.flow_schedule_tag)?;
        Ok(Self {
            cfg_mode: CfgMode::None,
            cfg_scale: CfgScaleProfile::Constant(1.0),
            nfe: config.flow_nfe as usize,
            schedule,
            solver: OdeSolver::Euler,
        })
    }

    /// Maps the `vokra.cosyvoice2.flow.schedule` string tag to a
    /// [`vokra_ops::Schedule`] variant. Rejects unknown tags loudly.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any tag other than
    /// `"linear"` / `"sway"` / `"epss"`.
    pub fn schedule_from_tag(tag: &str) -> Result<Schedule> {
        match tag {
            "linear" => Ok(Schedule::Linear),
            "sway" => Ok(Schedule::Sway),
            // `epss` maps to Schedule::EpsS (M3-05 documents this variant
            // as a stub pending M3-09's real schedule spec). Wiring the
            // tag is stable even while the schedule itself is under
            // revision.
            "epss" => Ok(Schedule::EpsS),
            other => Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 flow schedule tag `{other}` is not one of \
                 `linear` / `sway` / `epss`"
            ))),
        }
    }

    /// Assembles the [`FlowSamplerConfig`] that
    /// [`vokra_ops::flow_sample`] consumes.
    #[must_use]
    pub fn sampler_config(&self) -> FlowSamplerConfig {
        FlowSamplerConfig {
            cfg_mode: self.cfg_mode,
            cfg_scale: self.cfg_scale.clone(),
            nfe: self.nfe,
            schedule: self.schedule,
            solver: self.solver,
        }
    }
}

/// Chunk-aware Conditional Flow Matching driver — promoted from scaffold to
/// runnable driver in this session (M3-09-T10 / T11 / T14 partial).
///
/// The concrete velocity closure (LLM backbone + Flow Matching sub-network,
/// with the KV cache continued across chunk boundaries via M3-03 paged
/// storage) is the T07/T08 numeric path — this session does not invent
/// upstream tensor names. What the driver *does* own now:
///
/// - [`Self::step`] — one-chunk sampler, unchanged from the earlier scaffold
///   (returns [`VokraError::NotImplemented`] on the built-in velocity
///   closure so no silent output leaks, FR-EX-08).
/// - [`Self::run_chunks`] — **new**. Iterates over `n_chunks` boundaries
///   derived from `chunk_size` / `chunk_hop` and invokes a
///   **caller-supplied** velocity closure with a
///   [`ChunkContinuation`] snapshot (previous chunk's terminal state +
///   `chunk_index`). Between-chunk causality is carried by that snapshot
///   plus the sampler's own step-by-step state; the KV cache continuation
///   that a real LLM backbone would use is orthogonal to the sampler and
///   lives one layer above (T14 / T16).
///
/// The **injected-velocity form** is the internal-oracle testable path
/// today: with an identity or decay closure the driver produces a
/// deterministic trajectory whose chunk boundaries and length can be
/// verified without touching a real safetensors checkpoint (T09 smoke →
/// T21/T22 parity is the honest owner-side gate).
#[derive(Debug)]
pub struct ChunkAwareCfm {
    /// Copy of the config for `chunk_size` / `chunk_hop`.
    config: CosyVoice2Config,
    /// Resolved sampler axes (constructed once at engine build time; the
    /// per-invocation override is the CLI's job).
    params: FlowMatchingRuntimeParams,
}

/// Snapshot the driver hands the caller-supplied velocity closure at every
/// step of every chunk in [`ChunkAwareCfm::run_chunks`].
///
/// The snapshot is the **only** state carried across chunks by the sampler
/// itself; a real LLM velocity closure will additionally consult its own
/// paged KV cache (M3-03) — but that cache is opaque to the driver and
/// belongs to the caller's closure state. Keeping the driver ignorant of
/// the closure's inner state is what makes the injected-closure test path
/// deterministic (no hidden global mutation).
///
/// The closure receives this snapshot **by reference** so the driver can
/// mutate `chunk_index` between chunks without allocating a fresh struct
/// on every step.
#[derive(Debug, Clone)]
pub struct ChunkContinuation<'a> {
    /// Zero-based chunk index (`0..n_chunks`).
    pub chunk_index: usize,
    /// Total chunk count for the current [`ChunkAwareCfm::run_chunks`] call.
    pub n_chunks: usize,
    /// Terminal state (`t = 1.0`) of the previous chunk, or `None` for the
    /// first chunk.
    pub prev_terminal: Option<&'a FlowSamplerState>,
}

impl ChunkAwareCfm {
    /// Builds a chunk-aware CFM driver bound to `config`.
    ///
    /// # Errors
    ///
    /// Propagates [`FlowMatchingRuntimeParams::from_config`] validation
    /// errors (unknown schedule tag, etc.).
    pub fn new(config: CosyVoice2Config) -> Result<Self> {
        let params = FlowMatchingRuntimeParams::from_config(&config)?;
        Ok(Self { config, params })
    }

    /// The resolved runtime params (schedule / nfe / cfg_mode / …).
    #[must_use]
    pub fn params(&self) -> &FlowMatchingRuntimeParams {
        &self.params
    }

    /// The chunk-aware config surface (`streaming_chunk_size` and
    /// `streaming_chunk_hop` from the CosyVoice2 GGUF).
    #[must_use]
    pub fn config(&self) -> &CosyVoice2Config {
        &self.config
    }

    /// Runs one Flow Matching chunk from `initial_state` to the terminal
    /// timestep using [`vokra_ops::flow_sample`].
    ///
    /// This is the built-in NotImplemented path (FR-EX-08 — no silent
    /// zero-fill). Callers with an injected velocity closure use
    /// [`Self::step_with_velocity`] or [`Self::run_chunks`] instead.
    ///
    /// # Errors
    ///
    /// - [`VokraError::NotImplemented`] until T07/T08 wire the real LLM
    ///   velocity closure.
    /// - Propagates [`vokra_ops::flow_sample`] configuration errors
    ///   (invalid `nfe`, mismatched shapes, non-finite `cfg_scale`).
    pub fn step(&self, initial_state: &FlowSamplerState) -> Result<FlowSamplerState> {
        let cfg = self.params.sampler_config();
        // The velocity closure returns NotImplemented; the sampler
        // propagates that error on step 0 so the caller sees a loud
        // failure identical to the top-level engine's synthesize()
        // error surface (FR-EX-08).
        //
        // We still *invoke* flow_sample so the config-validation path
        // is exercised — a caller with e.g. `nfe = 0` today hits the
        // M3-05 validate_config error rather than the placeholder body
        // below.
        flow_sample(initial_state, &cfg, |_state, _t, _pass| {
            Err(VokraError::NotImplemented(
                "CosyVoice2 Flow Matching velocity closure is not implemented in this \
                 scaffold; T07 embedding / T08 LLM backbone / T10 CFM module wire the \
                 real velocity path against the upstream safetensors manifest",
            ))
        })
    }

    /// Runs one Flow Matching chunk with a caller-supplied velocity closure.
    ///
    /// This is the injected-velocity form the internal-oracle tests use —
    /// it exercises the sampler plumbing end-to-end without inventing
    /// upstream tensor names. A real LLM backbone (T07/T08) will land the
    /// closure body that consumes a paged KV cache + text conditioning +
    /// mel prior; today the tests use identity / decay velocities to
    /// verify chunk boundaries and length preservation.
    ///
    /// # Errors
    ///
    /// Propagates every [`vokra_ops::flow_sample`] validation error and
    /// any error returned by the caller's closure verbatim.
    pub fn step_with_velocity<F>(
        &self,
        initial_state: &FlowSamplerState,
        mut velocity: F,
    ) -> Result<FlowSamplerState>
    where
        F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
    {
        let cfg = self.params.sampler_config();
        flow_sample(initial_state, &cfg, |state, t, pass| {
            velocity(state, t, pass)
        })
    }

    /// Runs the full chunk-aware Flow Matching pipeline over `n_chunks`
    /// boundaries starting from `initial_state`.
    ///
    /// For each chunk `i ∈ 0..n_chunks`:
    ///
    /// 1. builds a [`ChunkContinuation`] snapshot (previous terminal =
    ///    `Some(&prev_state)` for `i > 0`, `None` for the first chunk);
    /// 2. runs [`vokra_ops::flow_sample`] with the caller's velocity closure
    ///    (which additionally receives `&ChunkContinuation` so the closure
    ///    can key its state on `chunk_index` — e.g. advance a KV cache
    ///    write cursor by `chunk_size` frames);
    /// 3. records the terminal state and feeds it into the next chunk as
    ///    the initial state (identity carry-over — models that need a
    ///    fresh noise re-init override this by ignoring `prev_terminal`
    ///    inside the closure).
    ///
    /// **Chunk boundary semantics** — the driver's job is only to iterate
    /// and carry state; nothing about `chunk_size` / `chunk_hop` (frame
    /// counts) enters the sampler math. Those live one layer above in the
    /// pipeline that translates a target frame count into chunk boundaries
    /// (M3-09 chunk_pipeline module).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `n_chunks == 0`; propagates
    /// every [`vokra_ops::flow_sample`] validation error and any closure
    /// error verbatim.
    pub fn run_chunks<F>(
        &self,
        initial_state: &FlowSamplerState,
        n_chunks: usize,
        mut velocity: F,
    ) -> Result<Vec<FlowSamplerState>>
    where
        F: FnMut(
            &FlowSamplerState,
            f32,
            ForwardPass,
            &ChunkContinuation<'_>,
        ) -> Result<FlowSamplerState>,
    {
        if n_chunks == 0 {
            return Err(VokraError::InvalidArgument(
                "ChunkAwareCfm::run_chunks: n_chunks must be non-zero (FR-EX-08 — no \
                 silent skip)"
                    .to_owned(),
            ));
        }
        let cfg = self.params.sampler_config();
        let mut chunks: Vec<FlowSamplerState> = Vec::with_capacity(n_chunks);
        // Local snapshot state: we cannot store the previous terminal
        // inside `ChunkContinuation` and pass it through `flow_sample`'s
        // closure because that closure captures the continuation by ref
        // through this outer scope — a straightforward pattern.
        let mut x = initial_state.clone();
        for chunk_index in 0..n_chunks {
            let prev_terminal: Option<&FlowSamplerState> = if chunk_index == 0 {
                None
            } else {
                chunks.last()
            };
            let cont = ChunkContinuation {
                chunk_index,
                n_chunks,
                prev_terminal,
            };
            let terminal = flow_sample(&x, &cfg, |state, t, pass| velocity(state, t, pass, &cont))?;
            if terminal.shape != x.shape {
                return Err(VokraError::InvalidArgument(format!(
                    "ChunkAwareCfm::run_chunks: chunk {chunk_index} terminal shape \
                     {:?} != chunk initial shape {:?} (FR-EX-08 — no silent reshape)",
                    terminal.shape, x.shape
                )));
            }
            x = terminal.clone();
            chunks.push(terminal);
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn stub_config_with_schedule(tag: &str, nfe: u32) -> CosyVoice2Config {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(super::super::config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(super::super::config::KEY_VOCAB_SIZE, 32);
        b.add_u32(super::super::config::KEY_HIDDEN_DIM, 16);
        b.add_u32(super::super::config::KEY_N_LAYER, 2);
        b.add_u32(super::super::config::KEY_N_HEAD, 2);
        b.add_u32(super::super::config::KEY_FFN_DIM, 32);
        b.add_u32(super::super::config::KEY_FLOW_NFE, nfe);
        b.add_string(super::super::config::KEY_FLOW_SCHEDULE, tag);
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, 4);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, 16);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, 8);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, 4);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, 4);
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        CosyVoice2Config::from_gguf(&file).expect("read")
    }

    #[test]
    fn schedule_tag_maps_to_ops_variant() {
        // Every tag we accept must round-trip to a Schedule variant that
        // vokra_ops::flow_sample understands.
        assert!(matches!(
            FlowMatchingRuntimeParams::schedule_from_tag("linear").unwrap(),
            Schedule::Linear
        ));
        assert!(matches!(
            FlowMatchingRuntimeParams::schedule_from_tag("sway").unwrap(),
            Schedule::Sway
        ));
        assert!(matches!(
            FlowMatchingRuntimeParams::schedule_from_tag("epss").unwrap(),
            Schedule::EpsS
        ));
    }

    #[test]
    fn unknown_schedule_tag_fails_loudly() {
        // FR-EX-08: no silent fallback to Schedule::Linear.
        let err = FlowMatchingRuntimeParams::schedule_from_tag("cosine")
            .expect_err("unknown tag must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn cfm_step_returns_not_implemented_from_velocity_closure() {
        // A caller who runs the sampler today gets a NotImplemented error
        // out of the velocity closure — never a silent zero-fill.
        let cfg = stub_config_with_schedule("linear", 4);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![2], vec![1.0, 0.5]).unwrap();
        let err = cfm
            .step(&x0)
            .expect_err("scaffold must not produce samples");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn cfm_step_still_validates_sampler_config() {
        // The sampler plumbing runs its own validation before the
        // velocity closure — a `nfe = 0` config fails there (M3-05
        // validate_config), before the scaffold NotImplemented would
        // fire. This is what makes the FR-EX-10 "runtime-selectable"
        // contract testable today.
        let cfg = stub_config_with_schedule("linear", 0);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![2], vec![1.0, 0.5]).unwrap();
        let err = cfm.step(&x0).expect_err("nfe=0 must fail up front");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- run_chunks: injected velocity closure --------------------------

    /// A zero-velocity closure: v(x, t) = 0. With Euler this leaves the
    /// state unchanged over any number of steps, giving an internal oracle
    /// for `run_chunks` iteration + carry-over without a real model.
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

    #[test]
    fn run_chunks_zero_velocity_preserves_shape_across_all_chunks() {
        // Shape and dtype invariance across chunk boundaries — the
        // essential "does the driver not silently drop or reshape a
        // chunk" check (FR-EX-08 + doc invariant).
        let cfg = stub_config_with_schedule("linear", 4);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![3], vec![1.0, -0.5, 0.25]).unwrap();
        let n_chunks = 5;
        let chunks = cfm
            .run_chunks(&x0, n_chunks, |state, _t, _p, _cont| {
                Ok(FlowSamplerState {
                    shape: state.shape.clone(),
                    data: vec![0.0; state.data.len()],
                })
            })
            .expect("run_chunks succeeds with zero velocity");
        assert_eq!(chunks.len(), n_chunks, "one terminal state per chunk");
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.shape, vec![3], "chunk {i} preserves shape");
            assert!(
                c.data.iter().all(|v| v.is_finite()),
                "chunk {i} has finite values (no NaN leak)"
            );
        }
    }

    #[test]
    fn run_chunks_reports_chunk_index_monotonically() {
        // The `chunk_index` carried by ChunkContinuation is what a real
        // KV cache write cursor would consult — verify the driver emits
        // 0, 1, 2, ... in order and equals `n_chunks - 1` on the last
        // step.
        let cfg = stub_config_with_schedule("linear", 2);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![1], vec![0.5]).unwrap();
        let n_chunks = 4;
        let mut seen_indices: Vec<usize> = Vec::new();
        // The closure records `cont.chunk_index` on every step; the
        // driver must never invoke the closure with an index out of
        // `0..n_chunks`.
        cfm.run_chunks(&x0, n_chunks, |state, _t, _p, cont| {
            assert_eq!(cont.n_chunks, n_chunks, "n_chunks passthrough");
            assert!(
                cont.chunk_index < cont.n_chunks,
                "chunk_index {} < n_chunks {}",
                cont.chunk_index,
                cont.n_chunks
            );
            seen_indices.push(cont.chunk_index);
            Ok(FlowSamplerState {
                shape: state.shape.clone(),
                data: vec![0.0; state.data.len()],
            })
        })
        .unwrap();
        // Each chunk runs `nfe = 2` sampler steps; the closure is called
        // once per step (CfgMode::None, one forward per step). Extract
        // the *distinct* chunk indices in order.
        let mut distinct: Vec<usize> = Vec::new();
        for &i in &seen_indices {
            if distinct.last() != Some(&i) {
                distinct.push(i);
            }
        }
        assert_eq!(distinct, vec![0, 1, 2, 3], "monotonically increasing");
    }

    #[test]
    fn run_chunks_first_chunk_prev_terminal_is_none() {
        // The first chunk sees no previous terminal (that is the invariant
        // the causal-mask consumer relies on — the very first frame has no
        // history).
        let cfg = stub_config_with_schedule("linear", 1);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let mut prev_terminal_on_first_chunk_was_none = false;
        cfm.run_chunks(&x0, 1, |state, _t, _p, cont| {
            if cont.chunk_index == 0 {
                prev_terminal_on_first_chunk_was_none = cont.prev_terminal.is_none();
            }
            Ok(FlowSamplerState {
                shape: state.shape.clone(),
                data: vec![0.0; state.data.len()],
            })
        })
        .unwrap();
        assert!(
            prev_terminal_on_first_chunk_was_none,
            "first chunk must see prev_terminal == None"
        );
    }

    #[test]
    fn run_chunks_subsequent_chunks_receive_prev_terminal() {
        // Chunks after the first must see the previous chunk's terminal
        // state — this is what a causal LLM would advance its KV cache
        // cursor by. We check by emitting a distinct velocity per chunk
        // and confirming the driver's carry-over is the previous
        // chunk's terminal.
        let cfg = stub_config_with_schedule("linear", 1); // 1 step per chunk
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        // Constant additive velocity: v = 1.0 → after one step of dt=1
        // (nfe=1, linear), the terminal is x_0 + 1.0.
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let n_chunks = 3;
        let mut prev_terminal_shapes: Vec<Option<Vec<usize>>> = Vec::new();
        let chunks = cfm
            .run_chunks(&x0, n_chunks, |state, _t, _p, cont| {
                if cont.chunk_index > 0 {
                    prev_terminal_shapes.push(cont.prev_terminal.map(|s| s.shape.clone()));
                }
                Ok(FlowSamplerState {
                    shape: state.shape.clone(),
                    data: vec![1.0; state.data.len()],
                })
            })
            .expect("run");
        // Each subsequent chunk sees a prev_terminal of matching shape.
        assert_eq!(
            prev_terminal_shapes,
            vec![Some(vec![1]), Some(vec![1])],
            "shape carry-over across chunks"
        );
        // Terminal state of chunk i is x_i + 1.0; the driver feeds
        // that into chunk i+1 — so chunk 0's terminal = 1.0, chunk 1's =
        // 2.0, chunk 2's = 3.0.
        assert_eq!(chunks[0].data, vec![1.0]);
        assert_eq!(chunks[1].data, vec![2.0]);
        assert_eq!(chunks[2].data, vec![3.0]);
    }

    #[test]
    fn run_chunks_zero_n_chunks_fails_loudly() {
        // FR-EX-08: no silent skip. n_chunks = 0 must be an explicit error.
        let cfg = stub_config_with_schedule("linear", 4);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let err = cfm
            .run_chunks(&x0, 0, zero_velocity)
            .expect_err("n_chunks=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn run_chunks_shape_mismatch_from_closure_is_caught_at_driver_level() {
        // The flow_sampler itself catches per-step shape mismatch; the
        // driver additionally catches "terminal shape != chunk initial
        // shape" (which is only reachable via a non-shape-preserving
        // closure — a stress path for FR-EX-08).
        //
        // In practice `flow_sample` errors first because its
        // check_same_shape assertion fires on the first step; that is
        // the desired outcome (loud, early). This test documents the
        // driver-level guard exists in case a future flow_sampler
        // change ever relaxes the per-step check.
        let cfg = stub_config_with_schedule("linear", 1);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![2], vec![0.0, 0.0]).unwrap();
        let err = cfm
            .run_chunks(&x0, 2, |_state, _t, _p, _cont| {
                Ok(FlowSamplerState {
                    shape: vec![3],
                    data: vec![0.0; 3],
                })
            })
            .expect_err("bad shape must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn run_chunks_propagates_closure_errors_verbatim() {
        // A closure that returns an error must not be swallowed by the
        // driver (FR-EX-08 — every failure surfaces).
        let cfg = stub_config_with_schedule("linear", 1);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let err = cfm
            .run_chunks(&x0, 2, |_state, _t, _p, _cont| {
                Err(VokraError::NotImplemented("test: closure refuses to run"))
            })
            .expect_err("closure error must propagate");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn step_with_velocity_matches_run_chunks_of_one() {
        // A single-chunk run is bit-identical to step_with_velocity —
        // consistency check that run_chunks(_, 1, _) reduces to the
        // one-shot form.
        let cfg = stub_config_with_schedule("linear", 3);
        let cfm = ChunkAwareCfm::new(cfg).expect("build");
        let x0 = FlowSamplerState::new(vec![2], vec![1.0, -0.5]).unwrap();
        let single = cfm
            .step_with_velocity(&x0, |s, _t, _p| {
                Ok(FlowSamplerState {
                    shape: s.shape.clone(),
                    data: s.data.iter().map(|v| -v).collect(),
                })
            })
            .expect("step_with_velocity");
        let chunks = cfm
            .run_chunks(&x0, 1, |s, _t, _p, _cont| {
                Ok(FlowSamplerState {
                    shape: s.shape.clone(),
                    data: s.data.iter().map(|v| -v).collect(),
                })
            })
            .expect("run_chunks");
        assert_eq!(chunks.len(), 1);
        // Bit-identical trajectories.
        for (a, b) in chunks[0].data.iter().zip(single.data.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "run_chunks(_,1,_) trajectory {a} != step_with_velocity {b}"
            );
        }
    }
}
