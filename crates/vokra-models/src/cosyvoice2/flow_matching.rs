//! CosyVoice2 Flow Matching CFM (Conditional Flow Matching) — chunk-aware
//! caller stub (M3-09-T10 / T11).
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
//!   LLM backbone + Flow Matching sub-network; between-chunk KV cache
//!   continuation is the T14 streaming pipeline's job (M3-03 paged KV cache).
//!
//! The concrete velocity closure lands with T10/T11; today [`ChunkAwareCfm::step`]
//! returns [`VokraError::NotImplemented`] with a clear next-step message so no
//! silent zero-fill fallback (FR-EX-08) leaks into the pipeline.

use vokra_core::{Result, VokraError};
use vokra_ops::{
    CfgMode, CfgScaleProfile, FlowSamplerConfig, FlowSamplerState, OdeSolver, Schedule, flow_sample,
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

/// Chunk-aware Conditional Flow Matching driver — scaffold.
///
/// The concrete velocity closure (LLM backbone + Flow Matching sub-network,
/// with the KV cache continued across chunk boundaries via M3-03 paged
/// storage) is the T10/T11 numeric path. This scaffold owns the sampler
/// wiring so a caller can build a driver today, and [`Self::step`] returns
/// [`VokraError::NotImplemented`] with a clear next-step message rather
/// than a silent noise-shaped output (FR-EX-08).
pub struct ChunkAwareCfm {
    /// Copy of the config for `chunk_size` / `chunk_hop` (M3-09-T14 will
    /// consume these when the streaming pipeline lands).
    #[allow(dead_code)] // consumed by T10-T14 numeric path
    config: CosyVoice2Config,
    /// Resolved sampler axes (constructed once at engine build time; the
    /// per-invocation override is the CLI's job).
    params: FlowMatchingRuntimeParams,
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

    /// Runs one Flow Matching chunk from `initial_state` to the terminal
    /// timestep using [`vokra_ops::flow_sample`].
    ///
    /// This scaffold returns [`VokraError::NotImplemented`] on the
    /// velocity closure — the sampler *plumbing* is exercised (invalid
    /// configs still fail via the underlying `flow_sample` validation),
    /// but the LLM backbone forward that computes velocity does not yet
    /// exist. The T10/T11 follow-on session replaces the closure body.
    ///
    /// # Errors
    ///
    /// - [`VokraError::NotImplemented`] until T10/T11 wire the velocity
    ///   closure.
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
                 scaffold; T10 CFM module + T11 flow_sampler wiring land the numeric \
                 path against the upstream safetensors manifest",
            ))
        })
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
}
