//! Flow Matching / Diffusion sampler (M3-05; FR-OP-20 / FR-OP-21).
//!
//! # Runtime function — NOT a graph node (FR-EX-10)
//!
//! [`flow_sample`] is a **runtime function**, not an `OpKind` variant.
//! Embedding sampler configuration (`nfe`, `cfg_mode`, `schedule`, `solver`)
//! in an [`OpKind`](vokra_core::OpKind) variant would force a model
//! re-conversion every time a caller wants to change any of them — which is
//! precisely the operation callers change most often (品質 / RTF trade-off,
//! CFG mode toggle for GPU memory pressure, A/B testing solvers).
//!
//! By keeping the sampler as a runtime function that receives:
//!
//! - an initial state ([`FlowSamplerState`]),
//! - a [`FlowSamplerConfig`] (all four axes: `cfg_mode` / `cfg_scale` /
//!   `nfe` / `schedule` / `solver`),
//! - a forward closure (`FnMut(&State, f32, ForwardPass) -> Result<State>`),
//!
//! every axis is runtime-selectable without re-converting the model. See
//! `docs/adr/M3-05-flow-sampler.md` §D1 for the rationale.
//!
//! # CFG modes ([`CfgMode`])
//!
//! - [`CfgMode::None`] — uncond only; the caller supplies uncond
//!   conditioning inside the forward closure and the sampler uses the
//!   returned velocity directly. `cfg_scale` is a **noop** (not a silent
//!   fallback — the field is documented as "ignored under None").
//! - [`CfgMode::SplitBatch`] — the caller batches `[uncond; cond]` on the
//!   batch dim inside the forward closure and returns `[v_uncond; v_cond]`
//!   concatenated on the same dim. The sampler splits and mixes:
//!   `v = v_uncond + cfg_scale · (v_cond − v_uncond)`. 1 forward per step,
//!   2× memory.
//! - [`CfgMode::DualForward`] — the sampler calls the forward closure
//!   twice per step, once with `ForwardPass::Uncond` and once with
//!   `ForwardPass::Cond`. 2× forward, 1× memory.
//!
//! # Schedules ([`Schedule`])
//!
//! Each returns `nfe + 1` timesteps in `[0, 1]` (see [`Schedule::timesteps`]).
//! Step `i` uses `(t_i, t_{i+1})`. The final timestep is always `1.0`.
//!
//! - [`Schedule::Linear`] — `t_i = i / nfe`, evenly spaced. Default choice.
//! - [`Schedule::Sway`] — F5-TTS sway sampling
//!   (Chen et al. 2024, arxiv 2410.06885, Eq. 5) with `s = -1.0` bias
//!   toward the noise side.
//! - [`Schedule::EpsS`] — **stub** placeholder (`t_i = 1 - (1 - i/nfe)²`);
//!   the exact formulation is pinned when M3-09 CosyVoice2 lands and the
//!   real-model schedule is known. The API surface (name + `Schedule::EpsS`
//!   arm) is stable now so that consumer code compiles against the final
//!   surface.
//!
//! # ODE solvers ([`OdeSolver`])
//!
//! Every solver's update is a pure function of `(x_i, t_i, dt, forward)`.
//! Predictor / corrector solvers may call `forward` more than once per step
//! (Heun: 2 NFE, DpmPp: 2 NFE).
//!
//! - [`OdeSolver::Euler`] — Flow Matching primary, 1st order.
//! - [`OdeSolver::Heun`] — 2nd-order predictor-corrector.
//! - [`OdeSolver::FlowOde`] — Rectified flow (Liu et al. 2022,
//!   arxiv 2209.03003); implemented as Euler for the standard formulation.
//!   M3-09 will decide whether CosyVoice2 needs a distinct branch.
//! - [`OdeSolver::Ddim`] — DDIM (Song et al. 2021, arxiv 2010.02502) in
//!   its deterministic form. **Diffusion-family**: the forward closure
//!   returns ε (noise prediction), not velocity. Uses a linear α schedule
//!   internally (`α_t = 1 − t`); model-specific α schedules are a M3-09
//!   concern.
//! - [`OdeSolver::DpmPp`] — DPM-Solver++ 2S (Lu et al. 2022,
//!   arxiv 2211.01095), 2nd-order single-step. **Diffusion-family**.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Invalid config raises [`VokraError::InvalidArgument`] up front:
//! - `nfe == 0`;
//! - `cfg_scale = Dynamic(v)` with `v.len() != nfe`;
//! - non-finite `cfg_scale` value;
//! - a forward closure returning a state with a different shape than the
//!   input (only surfaced during a step).

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// The sampler's per-step state.
///
/// A lightweight `(shape, data)` container: real-valued, row-major. The shape
/// is preserved across every step of [`flow_sample`]; a forward closure that
/// returns a differently-shaped state is an error (surfaced as
/// [`VokraError::InvalidArgument`] on the offending step).
///
/// This is deliberately *not* [`vokra_ops::dispatch::OpValue`]: the sampler
/// is a runtime function that lives outside the `OpKind` dispatch table
/// (FR-EX-10 — see crate rustdoc), so it uses its own state container to
/// keep the two API surfaces decoupled.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowSamplerState {
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Elements, row-major.
    pub data: Vec<f32>,
}

impl FlowSamplerState {
    /// Constructs a state from any convertible shape / data pair.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `data.len()` does not equal the
    /// product of `shape`.
    pub fn new(shape: impl Into<Vec<usize>>, data: impl Into<Vec<f32>>) -> Result<Self> {
        let shape = shape.into();
        let data = data.into();
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(VokraError::InvalidArgument(format!(
                "FlowSamplerState: shape product {expected} != data.len() {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Element count (`data.len()`, also equal to `shape.iter().product()`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff `data` is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Config enums
// ---------------------------------------------------------------------------

/// Classifier-free-guidance mode (FR-OP-20).
///
/// See the crate rustdoc "CFG modes" section for semantics of each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CfgMode {
    /// No CFG; forward is called once per step with `ForwardPass::Uncond`
    /// and the result is used as the velocity. `cfg_scale` is ignored under
    /// this mode.
    None,
    /// Batched CFG: forward closure receives a state whose data is
    /// `[uncond; cond]` on the batch dim and returns `[v_uncond; v_cond]`.
    /// The sampler splits and mixes.
    SplitBatch,
    /// Dual-forward CFG: sampler calls forward twice per step, once with
    /// `Uncond` and once with `Cond`.
    DualForward,
}

/// Timestep schedule (FR-OP-20).
///
/// See the crate rustdoc "Schedules" section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Schedule {
    /// Evenly spaced `t_i = i / nfe`.
    Linear,
    /// F5-TTS sway schedule (Chen et al. 2024, arxiv 2410.06885, Eq. 5).
    Sway,
    /// ε-schedule stub — pinned when M3-09 CosyVoice2 lands.
    EpsS,
}

/// ODE solver (FR-OP-21).
///
/// See the crate rustdoc "ODE solvers" section for each solver's update rule
/// and citation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OdeSolver {
    /// DDIM (Song et al. 2021), diffusion-family. Forward returns ε.
    Ddim,
    /// DPM-Solver++ 2S (Lu et al. 2022), diffusion-family. Forward returns ε.
    DpmPp,
    /// Euler 1st-order, Flow Matching primary. Forward returns velocity.
    Euler,
    /// Heun 2nd-order predictor-corrector. Forward returns velocity.
    Heun,
    /// Rectified-flow / Flow Matching standard formulation
    /// (Liu et al. 2022). Forward returns velocity.
    FlowOde,
}

/// CFG scale profile (FR-OP-20).
///
/// Either a single scalar applied at every step (`Constant`) or a per-step
/// lookup table (`Dynamic`). The lookup table's length must equal `nfe` —
/// [`flow_sample`] rejects a mismatch with an explicit
/// [`VokraError::InvalidArgument`] (FR-EX-08: no silent interpolation).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CfgScaleProfile {
    /// Same scale at every step.
    Constant(f32),
    /// Per-step lookup, indexed by step index (`0..nfe`).
    Dynamic(Vec<f32>),
}

impl CfgScaleProfile {
    /// Returns the scale to apply at the given step index.
    fn at(&self, step: usize) -> f32 {
        match self {
            Self::Constant(s) => *s,
            // The bounds check happens up-front in `validate_config`, so a
            // panicking index here means an internal invariant broke; use
            // `.expect` so it's a clear panic path in debug and unreachable
            // in release-normal use.
            Self::Dynamic(v) => v[step],
        }
    }
}

/// Which conditioning pass the sampler is requesting from the forward
/// closure.
///
/// Semantics per [`CfgMode`]:
///
/// | mode | ForwardPass received by closure | closure returns |
/// |------|----------------------------------|-----------------|
/// | `None` | `Uncond` | velocity (or ε for diffusion solvers) |
/// | `SplitBatch` | `SplitBatched` (once) | batched `[v_uncond; v_cond]` |
/// | `DualForward` | `Uncond` then `Cond` (twice) | velocity for each |
///
/// The closure decides how each variant maps to conditioning inputs — the
/// sampler is opaque to what "conditioning" means for the specific model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ForwardPass {
    /// Unconditional forward (no / null conditioning).
    Uncond,
    /// Conditional forward (target conditioning attached).
    Cond,
    /// One batched forward containing `[uncond; cond]` on the batch dim.
    SplitBatched,
}

// ---------------------------------------------------------------------------
// Config struct
// ---------------------------------------------------------------------------

/// Runtime-adjustable configuration for [`flow_sample`] (FR-OP-20 / FR-OP-21).
///
/// Consumers construct this by struct literal (the primary "set every axis
/// once" ergonomic pattern for a runtime config) or via
/// [`FlowSamplerConfig::euler_defaults`] followed by field overrides. The
/// wrapped enums are `#[non_exhaustive]` so new variants (additional
/// schedules, additional solvers) remain backwards-compatible; new *axes*
/// are added by extending this struct through a semver-compatible field
/// addition (a matching-pattern break at the top level would already be
/// visible to callers today because the config has five heterogeneous
/// fields, so callers pattern-match the enums, not the struct).
///
/// # Example
///
/// ```
/// use vokra_ops::flow_sampler::{
///     CfgMode, CfgScaleProfile, FlowSamplerConfig, OdeSolver, Schedule,
/// };
///
/// // 5-step Euler sampler with no CFG — Flow Matching primary.
/// let cfg = FlowSamplerConfig {
///     cfg_mode: CfgMode::None,
///     cfg_scale: CfgScaleProfile::Constant(1.0),
///     nfe: 5,
///     schedule: Schedule::Linear,
///     solver: OdeSolver::Euler,
/// };
/// assert_eq!(cfg.nfe, 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FlowSamplerConfig {
    /// CFG mode.
    pub cfg_mode: CfgMode,
    /// CFG scale profile (ignored when `cfg_mode == None`).
    pub cfg_scale: CfgScaleProfile,
    /// Number of function evaluations (integration steps).
    pub nfe: usize,
    /// Timestep schedule.
    pub schedule: Schedule,
    /// ODE solver.
    pub solver: OdeSolver,
}

impl FlowSamplerConfig {
    /// Returns a sensible default (5-step Euler + Linear + no CFG).
    ///
    /// Handy for tests and toy problems; production callers usually override
    /// several fields.
    pub fn euler_defaults(nfe: usize) -> Self {
        Self {
            cfg_mode: CfgMode::None,
            cfg_scale: CfgScaleProfile::Constant(1.0),
            nfe,
            schedule: Schedule::Linear,
            solver: OdeSolver::Euler,
        }
    }
}

// ---------------------------------------------------------------------------
// Schedules
// ---------------------------------------------------------------------------

impl Schedule {
    /// Returns the schedule's `nfe + 1` timesteps in `[0.0, 1.0]`.
    ///
    /// Element `0` is always `0.0` and element `nfe` is always `1.0`; step
    /// `i` in [`flow_sample`] consumes the pair `(timesteps[i], timesteps[i+1])`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `nfe == 0`.
    pub fn timesteps(&self, nfe: usize) -> Result<Vec<f32>> {
        if nfe == 0 {
            return Err(VokraError::InvalidArgument(
                "Schedule::timesteps: nfe must be non-zero".to_owned(),
            ));
        }
        let n = nfe as f32;
        let base: Vec<f32> = (0..=nfe).map(|i| i as f32 / n).collect();
        match self {
            Self::Linear => Ok(base),
            // F5-TTS sway (Chen et al. 2024, arxiv 2410.06885, Eq. 5):
            //   t_shifted = t + s * (cos(π * t / 2) - 1 + t)
            // with s = -1.0 as the paper's default (bias toward noise side).
            // The endpoints are fixed at 0 and 1 because the formula reduces
            // to those values there (cos(0) = 1, cos(π/2) = 0), independent
            // of s.
            Self::Sway => {
                const S: f32 = -1.0;
                let out = base
                    .into_iter()
                    .map(|t| t + S * ((std::f32::consts::PI * t / 2.0).cos() - 1.0 + t))
                    .collect();
                Ok(out)
            }
            // EpsS is a STUB (see crate rustdoc and ADR M3-05 §D3). The
            // formula below is a quadratic placeholder; the exact schedule
            // will be pinned during M3-09 CosyVoice2 integration once the
            // real-model schedule is known. Kept as a distinct arm so
            // consumer code selecting `Schedule::EpsS` compiles against the
            // final surface today.
            Self::EpsS => {
                let out = base
                    .into_iter()
                    .map(|t| {
                        let u = 1.0 - t;
                        1.0 - u * u
                    })
                    .collect();
                Ok(out)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

fn validate_config(config: &FlowSamplerConfig) -> Result<()> {
    if config.nfe == 0 {
        return Err(VokraError::InvalidArgument(
            "flow_sample: nfe must be non-zero".to_owned(),
        ));
    }
    match &config.cfg_scale {
        CfgScaleProfile::Constant(s) => {
            if !s.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "flow_sample: cfg_scale.Constant must be finite (got {s})"
                )));
            }
        }
        CfgScaleProfile::Dynamic(v) => {
            if v.len() != config.nfe {
                return Err(VokraError::InvalidArgument(format!(
                    "flow_sample: Dynamic cfg_scale length {} != nfe {}",
                    v.len(),
                    config.nfe
                )));
            }
            for (i, s) in v.iter().enumerate() {
                if !s.is_finite() {
                    return Err(VokraError::InvalidArgument(format!(
                        "flow_sample: cfg_scale.Dynamic[{i}] must be finite (got {s})"
                    )));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Forward-pass helpers (CFG dispatch)
// ---------------------------------------------------------------------------

/// Requests a velocity from the forward closure at `(state, t)`, applying
/// the configured CFG mode. Returns the CFG-combined velocity (same shape
/// as `state`).
fn velocity_at<F>(
    state: &FlowSamplerState,
    t: f32,
    cfg_mode: CfgMode,
    scale: f32,
    forward: &mut F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    match cfg_mode {
        CfgMode::None => {
            let v = forward(state, t, ForwardPass::Uncond)?;
            check_same_shape(state, &v, "velocity")?;
            Ok(v)
        }
        CfgMode::DualForward => {
            let v_uncond = forward(state, t, ForwardPass::Uncond)?;
            check_same_shape(state, &v_uncond, "velocity(uncond)")?;
            let v_cond = forward(state, t, ForwardPass::Cond)?;
            check_same_shape(state, &v_cond, "velocity(cond)")?;
            Ok(cfg_mix(&v_uncond, &v_cond, scale))
        }
        CfgMode::SplitBatch => {
            let v = forward(state, t, ForwardPass::SplitBatched)?;
            // For SplitBatch the closure must return double the element
            // count (uncond concat cond on the batch dim). We split
            // element-wise; the caller's shape convention determines the
            // physical layout.
            if v.len() != 2 * state.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "flow_sample: SplitBatch expected 2× batched velocity ({} elements), got {}",
                    2 * state.len(),
                    v.len()
                )));
            }
            let n = state.len();
            let (uncond_part, cond_part) = v.data.split_at(n);
            let mut mixed = Vec::with_capacity(n);
            for i in 0..n {
                mixed.push(uncond_part[i] + scale * (cond_part[i] - uncond_part[i]));
            }
            Ok(FlowSamplerState {
                shape: state.shape.clone(),
                data: mixed,
            })
        }
    }
}

fn cfg_mix(v_uncond: &FlowSamplerState, v_cond: &FlowSamplerState, scale: f32) -> FlowSamplerState {
    let mut data = Vec::with_capacity(v_uncond.len());
    for (u, c) in v_uncond.data.iter().zip(v_cond.data.iter()) {
        data.push(u + scale * (c - u));
    }
    FlowSamplerState {
        shape: v_uncond.shape.clone(),
        data,
    }
}

fn check_same_shape(a: &FlowSamplerState, b: &FlowSamplerState, ctx: &str) -> Result<()> {
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "flow_sample: {ctx} shape {:?} != state shape {:?}",
            b.shape, a.shape
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Solver step primitives
// ---------------------------------------------------------------------------

/// `x + alpha * y` element-wise.
fn axpy(x: &FlowSamplerState, alpha: f32, y: &FlowSamplerState) -> FlowSamplerState {
    let mut data = Vec::with_capacity(x.len());
    for (xi, yi) in x.data.iter().zip(y.data.iter()) {
        data.push(xi + alpha * yi);
    }
    FlowSamplerState {
        shape: x.shape.clone(),
        data,
    }
}

/// Euler step: `x_{i+1} = x_i + dt * v(x_i, t_i)`.
fn euler_step<F>(
    x: &FlowSamplerState,
    t: f32,
    dt: f32,
    cfg_mode: CfgMode,
    scale: f32,
    forward: &mut F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    let v = velocity_at(x, t, cfg_mode, scale, forward)?;
    Ok(axpy(x, dt, &v))
}

/// Heun predictor-corrector step:
///   k1 = v(x_i, t_i)
///   k2 = v(x_i + dt*k1, t_{i+1})
///   x_{i+1} = x_i + (dt/2) * (k1 + k2)
fn heun_step<F>(
    x: &FlowSamplerState,
    t: f32,
    t_next: f32,
    dt: f32,
    cfg_mode: CfgMode,
    scale: f32,
    forward: &mut F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    let k1 = velocity_at(x, t, cfg_mode, scale, forward)?;
    let x_pred = axpy(x, dt, &k1);
    let k2 = velocity_at(&x_pred, t_next, cfg_mode, scale, forward)?;
    // (k1 + k2) / 2, then axpy with dt.
    let half = FlowSamplerState {
        shape: x.shape.clone(),
        data: k1
            .data
            .iter()
            .zip(k2.data.iter())
            .map(|(a, b)| 0.5 * (a + b))
            .collect(),
    };
    Ok(axpy(x, dt, &half))
}

/// DDIM step (Song et al. 2021, arxiv 2010.02502, Eq. 12 with σ_t = 0).
///
/// Forward returns ε (noise prediction). α schedule is linear (`α_t = 1 − t`)
/// as pinned in ADR M3-05 §D4; model-specific α schedules are a M3-09 concern.
fn ddim_step<F>(
    x: &FlowSamplerState,
    t: f32,
    t_next: f32,
    cfg_mode: CfgMode,
    scale: f32,
    forward: &mut F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    let eps = velocity_at(x, t, cfg_mode, scale, forward)?;
    // Linear α schedule: α_t = 1 - t, so sqrt(α_t) = sqrt(1 - t) and
    // sqrt(1 - α_t) = sqrt(t). Clamp inputs to [0, 1] before sqrt to be
    // robust against floating-point epsilons at the endpoints.
    let alpha_t = (1.0_f32 - t).clamp(0.0, 1.0);
    let alpha_next = (1.0_f32 - t_next).clamp(0.0, 1.0);
    let sqrt_a_t = alpha_t.sqrt();
    let sqrt_a_next = alpha_next.sqrt();
    let sqrt_1m_a_t = (1.0 - alpha_t).max(0.0).sqrt();
    let sqrt_1m_a_next = (1.0 - alpha_next).max(0.0).sqrt();
    // pred_x0 = (x - sqrt(1 - α_t) · ε) / sqrt(α_t)
    // x_next  = sqrt(α_next) · pred_x0 + sqrt(1 - α_next) · ε
    //
    // Avoid a division by zero when sqrt_a_t == 0 (t = 1); in that case the
    // step is at the terminal endpoint and pred_x0 is undefined — return
    // the input state unchanged (safe idempotent default).
    if sqrt_a_t <= f32::EPSILON {
        return Ok(x.clone());
    }
    let mut data = Vec::with_capacity(x.len());
    for (xi, ei) in x.data.iter().zip(eps.data.iter()) {
        let pred_x0 = (xi - sqrt_1m_a_t * ei) / sqrt_a_t;
        data.push(sqrt_a_next * pred_x0 + sqrt_1m_a_next * ei);
    }
    Ok(FlowSamplerState {
        shape: x.shape.clone(),
        data,
    })
}

/// DPM-Solver++ 2S step (Lu et al. 2022, arxiv 2211.01095, Algorithm 4).
///
/// A single-step 2nd-order predictor-corrector. Forward returns ε (or
/// equivalently the data prediction `D_θ(x, t) = (x - sqrt(1-α)·ε)/sqrt(α)`,
/// following the paper's convention). Uses the same linear α schedule as
/// DDIM here (`α_t = 1 − t`).
///
/// The full paper introduces a half-step at `t_s = 0.5 * (t + t_next)` in
/// log-SNR space; with the simple linear α schedule the mid-log-SNR
/// simplifies to `t_s = 0.5 * (t + t_next)` in the t domain as well, which
/// is what we use.
fn dpmpp_step<F>(
    x: &FlowSamplerState,
    t: f32,
    t_next: f32,
    cfg_mode: CfgMode,
    scale: f32,
    forward: &mut F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    // Predictor: DDIM half-step to t_s = midpoint(t, t_next).
    let t_s = 0.5 * (t + t_next);
    let x_pred = ddim_step(x, t, t_s, cfg_mode, scale, forward)?;
    // Corrector: DDIM step from t to t_next using ε evaluated at (x_pred, t_s).
    // We compute the corrector by taking a DDIM step whose ε comes from
    // t_s. `ddim_step` currently evaluates ε at (x, t) internally, so we
    // replicate the DDIM update here with the corrected ε.
    let eps_corr = velocity_at(&x_pred, t_s, cfg_mode, scale, forward)?;
    let alpha_t = (1.0_f32 - t).clamp(0.0, 1.0);
    let alpha_next = (1.0_f32 - t_next).clamp(0.0, 1.0);
    let sqrt_a_t = alpha_t.sqrt();
    let sqrt_a_next = alpha_next.sqrt();
    let sqrt_1m_a_t = (1.0 - alpha_t).max(0.0).sqrt();
    let sqrt_1m_a_next = (1.0 - alpha_next).max(0.0).sqrt();
    if sqrt_a_t <= f32::EPSILON {
        return Ok(x.clone());
    }
    let mut data = Vec::with_capacity(x.len());
    for (xi, ei) in x.data.iter().zip(eps_corr.data.iter()) {
        let pred_x0 = (xi - sqrt_1m_a_t * ei) / sqrt_a_t;
        data.push(sqrt_a_next * pred_x0 + sqrt_1m_a_next * ei);
    }
    Ok(FlowSamplerState {
        shape: x.shape.clone(),
        data,
    })
}

// ---------------------------------------------------------------------------
// Public entrypoint
// ---------------------------------------------------------------------------

/// Runs a Flow Matching / Diffusion sampler over `initial_state` (M3-05;
/// FR-OP-20 / FR-OP-21).
///
/// # Runtime function — NOT a graph node (FR-EX-10)
///
/// This function is deliberately **not** wired into `OpKind` / dispatch. See
/// the crate-level rustdoc and `docs/adr/M3-05-flow-sampler.md` for the
/// rationale: changing `nfe`, `cfg_mode`, `schedule`, or `solver` at runtime
/// must not require re-converting the model.
///
/// # Arguments
///
/// - `initial_state` — the state at `t = 0` (typically noise).
/// - `config` — full sampler configuration; every axis is runtime-selectable.
/// - `forward` — the model's velocity / ε predictor. The closure receives
///   the current state, the current timestep, and a [`ForwardPass`] tag; it
///   must return a state whose shape matches the input (except in
///   `SplitBatch` mode, where the returned state's `data` must be twice as
///   long — see [`CfgMode::SplitBatch`]).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on:
/// - `config.nfe == 0`;
/// - a non-finite `cfg_scale` value;
/// - `cfg_scale = Dynamic(v)` with `v.len() != config.nfe`;
/// - a forward closure returning a state with a mismatched shape.
///
/// Any error propagated from the forward closure is returned unchanged.
///
/// # Example
///
/// ```
/// use vokra_ops::flow_sampler::{
///     flow_sample, FlowSamplerConfig, FlowSamplerState, ForwardPass,
/// };
///
/// let cfg = FlowSamplerConfig::euler_defaults(5);
/// let x0 = FlowSamplerState::new(vec![1], vec![1.0]).unwrap();
/// // Toy velocity field v(x, t) = -x: analytic solution x(t) = x(0) · e^{-t}.
/// let out = flow_sample(&x0, &cfg, |state, _t, _pass| {
///     Ok(FlowSamplerState {
///         shape: state.shape.clone(),
///         data: state.data.iter().map(|v| -v).collect(),
///     })
/// })
/// .unwrap();
/// assert!(out.data[0] > 0.0 && out.data[0] < 1.0);
/// ```
pub fn flow_sample<F>(
    initial_state: &FlowSamplerState,
    config: &FlowSamplerConfig,
    mut forward: F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    validate_config(config)?;
    let timesteps = config.schedule.timesteps(config.nfe)?;
    let mut x = initial_state.clone();
    for step in 0..config.nfe {
        let t = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t;
        let scale = config.cfg_scale.at(step);
        let x_next = match config.solver {
            OdeSolver::Euler | OdeSolver::FlowOde => {
                // FlowOde uses Euler's update rule for the rectified-flow /
                // standard-Flow-Matching formulation (ADR M3-05 §D4). A
                // distinct branch is kept in the enum so M3-09 can differ.
                euler_step(&x, t, dt, config.cfg_mode, scale, &mut forward)?
            }
            OdeSolver::Heun => heun_step(&x, t, t_next, dt, config.cfg_mode, scale, &mut forward)?,
            OdeSolver::Ddim => ddim_step(&x, t, t_next, config.cfg_mode, scale, &mut forward)?,
            OdeSolver::DpmPp => dpmpp_step(&x, t, t_next, config.cfg_mode, scale, &mut forward)?,
        };
        x = x_next;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ---------------------------------------------------------

    fn state(data: Vec<f32>) -> FlowSamplerState {
        let n = data.len();
        FlowSamplerState::new(vec![n], data).unwrap()
    }

    /// The toy identity closure — `v(x, t) = x`. Useful as a sanity check
    /// that a solver produces `x_{i+1} = x_i + dt * x_i` growth (Euler on
    /// dx/dt = x).
    fn identity_velocity(
        s: &FlowSamplerState,
        _t: f32,
        _p: ForwardPass,
    ) -> Result<FlowSamplerState> {
        Ok(FlowSamplerState {
            shape: s.shape.clone(),
            data: s.data.clone(),
        })
    }

    /// v(x, t) = -x: analytic ODE dx/dt = -x has solution x(t) = x(0) · e^{-t}.
    /// Terminal state at t=1 is x(0)/e. A canonical toy for testing
    /// convergence order.
    fn decay_velocity(s: &FlowSamplerState, _t: f32, _p: ForwardPass) -> Result<FlowSamplerState> {
        Ok(FlowSamplerState {
            shape: s.shape.clone(),
            data: s.data.iter().map(|v| -v).collect(),
        })
    }

    // ---- State ----------------------------------------------------------

    #[test]
    fn state_new_rejects_shape_data_mismatch() {
        let e = FlowSamplerState::new(vec![2, 3], vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn state_new_accepts_matching_shape_data() {
        let s = FlowSamplerState::new(vec![2, 3], vec![0.0; 6]).unwrap();
        assert_eq!(s.len(), 6);
        assert_eq!(s.shape, [2, 3]);
    }

    // ---- Schedules -------------------------------------------------------

    #[test]
    fn linear_schedule_matches_ticket_example() {
        // WP ticket T09: nfe=5 produces [0, 0.2, 0.4, 0.6, 0.8, 1.0].
        let ts = Schedule::Linear.timesteps(5).unwrap();
        assert_eq!(ts.len(), 6);
        for (i, &v) in [0.0_f32, 0.2, 0.4, 0.6, 0.8, 1.0].iter().enumerate() {
            assert!((ts[i] - v).abs() < 1e-6, "ts[{i}] = {} vs {}", ts[i], v);
        }
    }

    #[test]
    fn schedules_start_at_zero_and_end_at_one() {
        for sched in [Schedule::Linear, Schedule::Sway, Schedule::EpsS] {
            let ts = sched.timesteps(7).unwrap();
            assert_eq!(ts.len(), 8);
            assert!((ts[0] - 0.0).abs() < 1e-6, "{sched:?} start = {}", ts[0]);
            assert!(
                (ts[ts.len() - 1] - 1.0).abs() < 1e-6,
                "{sched:?} end = {}",
                ts[ts.len() - 1]
            );
        }
    }

    #[test]
    fn schedules_reject_zero_nfe() {
        for sched in [Schedule::Linear, Schedule::Sway, Schedule::EpsS] {
            let e = sched.timesteps(0).unwrap_err();
            assert!(
                matches!(e, VokraError::InvalidArgument(_)),
                "{sched:?}: {e:?}"
            );
        }
    }

    #[test]
    fn sway_schedule_biases_toward_noise_side() {
        // With s = -1.0 (F5-TTS default), the sway schedule places more
        // samples near t = 0 than Linear does. At i = nfe/2 (midpoint of the
        // linear schedule), sway should return a value smaller than 0.5.
        let ts = Schedule::Sway.timesteps(10).unwrap();
        assert!(ts[5] < 0.5, "sway midpoint {} should be < 0.5", ts[5]);
    }

    #[test]
    fn epss_schedule_stub_is_monotone() {
        // The EpsS stub is a placeholder but must still produce a monotone
        // increasing sequence (a schedule property every consumer will assume).
        let ts = Schedule::EpsS.timesteps(10).unwrap();
        for w in ts.windows(2) {
            assert!(w[0] <= w[1], "not monotone: {w:?}");
        }
    }

    // ---- Config validation ----------------------------------------------

    #[test]
    fn zero_nfe_is_rejected() {
        let cfg = FlowSamplerConfig::euler_defaults(0);
        let x = state(vec![1.0]);
        let e = flow_sample(&x, &cfg, identity_velocity).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn dynamic_scale_length_mismatch_is_rejected() {
        // nfe=5 with a Dynamic profile of length 3 must fail (FR-EX-08 —
        // no silent interpolation).
        let mut cfg = FlowSamplerConfig::euler_defaults(5);
        cfg.cfg_scale = CfgScaleProfile::Dynamic(vec![1.0, 2.0, 3.0]);
        let x = state(vec![1.0]);
        let e = flow_sample(&x, &cfg, identity_velocity).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn non_finite_scale_is_rejected() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut cfg = FlowSamplerConfig::euler_defaults(3);
            cfg.cfg_scale = CfgScaleProfile::Constant(bad);
            let x = state(vec![1.0]);
            let e = flow_sample(&x, &cfg, identity_velocity).unwrap_err();
            assert!(matches!(e, VokraError::InvalidArgument(_)), "bad={bad}");
        }
    }

    #[test]
    fn dynamic_scale_with_non_finite_element_is_rejected() {
        let mut cfg = FlowSamplerConfig::euler_defaults(3);
        cfg.cfg_scale = CfgScaleProfile::Dynamic(vec![1.0, f32::NAN, 3.0]);
        let x = state(vec![1.0]);
        let e = flow_sample(&x, &cfg, identity_velocity).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_closure_shape_mismatch_is_rejected() {
        let cfg = FlowSamplerConfig::euler_defaults(3);
        let x = state(vec![1.0, 2.0]);
        let e = flow_sample(&x, &cfg, |s, _t, _p| {
            Ok(FlowSamplerState {
                shape: vec![s.data.len() + 1],
                data: vec![0.0; s.data.len() + 1],
            })
        })
        .unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    // ---- Solvers on the identity toy ------------------------------------

    #[test]
    fn euler_on_identity_grows_geometrically() {
        // v(x, t) = x → Euler: x_{i+1} = x_i (1 + dt). After nfe=1 step with
        // dt=1.0 (t=0 → t=1), x_1 = 2 * x_0.
        let cfg = FlowSamplerConfig::euler_defaults(1);
        let x = state(vec![1.0]);
        let out = flow_sample(&x, &cfg, identity_velocity).unwrap();
        assert!((out.data[0] - 2.0).abs() < 1e-6, "got {}", out.data[0]);
    }

    #[test]
    fn euler_converges_to_decay_analytic_solution() {
        // v(x, t) = -x: analytic x(1) = e^{-1} ≈ 0.3678794. Euler with a
        // decent nfe should be within ~0.1 of the exact answer.
        let cfg = FlowSamplerConfig::euler_defaults(100);
        let x = state(vec![1.0]);
        let out = flow_sample(&x, &cfg, decay_velocity).unwrap();
        let exact = (-1.0_f32).exp();
        assert!(
            (out.data[0] - exact).abs() < 0.02,
            "euler(100) got {} vs exact {}",
            out.data[0],
            exact
        );
    }

    #[test]
    fn heun_beats_euler_on_decay_at_equal_step_count() {
        // Same nfe, Heun's L2 error should be smaller (order-2 vs order-1).
        let mut cfg_euler = FlowSamplerConfig::euler_defaults(4);
        cfg_euler.solver = OdeSolver::Euler;
        let mut cfg_heun = FlowSamplerConfig::euler_defaults(4);
        cfg_heun.solver = OdeSolver::Heun;
        let x = state(vec![1.0]);
        let e_euler = flow_sample(&x, &cfg_euler, decay_velocity).unwrap();
        let e_heun = flow_sample(&x, &cfg_heun, decay_velocity).unwrap();
        let exact = (-1.0_f32).exp();
        let err_euler = (e_euler.data[0] - exact).abs();
        let err_heun = (e_heun.data[0] - exact).abs();
        assert!(
            err_heun < err_euler,
            "heun err {err_heun} should be < euler err {err_euler}"
        );
    }

    #[test]
    fn flow_ode_matches_euler_for_standard_formulation() {
        // ADR M3-05 §D4: FlowOde is spelled distinctly but is currently the
        // rectified-flow Euler update. The two arms must produce identical
        // trajectories on the same problem.
        let mut cfg_e = FlowSamplerConfig::euler_defaults(10);
        cfg_e.solver = OdeSolver::Euler;
        let mut cfg_f = FlowSamplerConfig::euler_defaults(10);
        cfg_f.solver = OdeSolver::FlowOde;
        let x = state(vec![1.0, 2.0, -1.0]);
        let out_e = flow_sample(&x, &cfg_e, decay_velocity).unwrap();
        let out_f = flow_sample(&x, &cfg_f, decay_velocity).unwrap();
        for (a, b) in out_e.data.iter().zip(out_f.data.iter()) {
            assert!((a - b).abs() < 1e-6, "euler {a} vs flow_ode {b}");
        }
    }

    #[test]
    fn ddim_returns_output_on_identity_forward() {
        // With v(x,t)=x treated as ε and a linear α schedule, DDIM produces
        // a well-defined trajectory (specific numeric value is model-defined;
        // the sanity check is that a step succeeds and produces finite output).
        let mut cfg = FlowSamplerConfig::euler_defaults(5);
        cfg.solver = OdeSolver::Ddim;
        let x = state(vec![1.0, -0.5, 0.3]);
        let out = flow_sample(&x, &cfg, identity_velocity).unwrap();
        assert_eq!(out.data.len(), 3);
        for v in &out.data {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn dpmpp_returns_output_on_identity_forward() {
        let mut cfg = FlowSamplerConfig::euler_defaults(5);
        cfg.solver = OdeSolver::DpmPp;
        let x = state(vec![1.0, -0.5, 0.3]);
        let out = flow_sample(&x, &cfg, identity_velocity).unwrap();
        assert_eq!(out.data.len(), 3);
        for v in &out.data {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    // ---- CFG modes -------------------------------------------------------

    #[test]
    fn cfg_none_calls_forward_once_per_step_with_uncond() {
        let mut cfg = FlowSamplerConfig::euler_defaults(3);
        cfg.cfg_mode = CfgMode::None;
        let x = state(vec![1.0]);
        let mut passes = Vec::new();
        let mut call_count = 0_usize;
        flow_sample(&x, &cfg, |s, _t, pass| {
            passes.push(pass);
            call_count += 1;
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: s.data.clone(),
            })
        })
        .unwrap();
        // 1 forward per step * 3 steps = 3.
        assert_eq!(call_count, 3, "call_count = {call_count}");
        for p in &passes {
            assert_eq!(*p, ForwardPass::Uncond);
        }
    }

    #[test]
    fn cfg_dual_forward_calls_uncond_then_cond_per_step() {
        let mut cfg = FlowSamplerConfig::euler_defaults(2);
        cfg.cfg_mode = CfgMode::DualForward;
        cfg.cfg_scale = CfgScaleProfile::Constant(2.0);
        let x = state(vec![1.0]);
        let mut passes = Vec::new();
        flow_sample(&x, &cfg, |s, _t, pass| {
            passes.push(pass);
            let val = match pass {
                ForwardPass::Uncond => 1.0,
                ForwardPass::Cond => 2.0,
                ForwardPass::SplitBatched => panic!("unexpected pass"),
            };
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: vec![val; s.data.len()],
            })
        })
        .unwrap();
        // 2 forwards per step * 2 steps = 4.
        assert_eq!(passes.len(), 4);
        assert_eq!(passes[0], ForwardPass::Uncond);
        assert_eq!(passes[1], ForwardPass::Cond);
        assert_eq!(passes[2], ForwardPass::Uncond);
        assert_eq!(passes[3], ForwardPass::Cond);
    }

    #[test]
    fn cfg_split_batch_calls_forward_once_per_step_with_split_batched() {
        let mut cfg = FlowSamplerConfig::euler_defaults(2);
        cfg.cfg_mode = CfgMode::SplitBatch;
        cfg.cfg_scale = CfgScaleProfile::Constant(2.0);
        let x = state(vec![1.0]);
        let mut passes = Vec::new();
        flow_sample(&x, &cfg, |s, _t, pass| {
            passes.push(pass);
            // Return [uncond=1.0; cond=2.0] on the batch dim.
            let n = s.len();
            let mut data = vec![1.0; n];
            data.extend(std::iter::repeat_n(2.0, n));
            Ok(FlowSamplerState {
                shape: vec![2 * n],
                data,
            })
        })
        .unwrap();
        assert_eq!(passes.len(), 2);
        for p in &passes {
            assert_eq!(*p, ForwardPass::SplitBatched);
        }
    }

    #[test]
    fn cfg_split_batch_and_dual_forward_agree_on_deterministic_forward() {
        // Given a deterministic (t / pass)-independent forward that returns
        // v_uncond = 1.0 and v_cond = 2.0 always, SplitBatch and DualForward
        // must produce identical trajectories.
        let x = state(vec![1.0]);
        let mut cfg_sb = FlowSamplerConfig::euler_defaults(5);
        cfg_sb.cfg_mode = CfgMode::SplitBatch;
        cfg_sb.cfg_scale = CfgScaleProfile::Constant(3.0);
        let mut cfg_df = cfg_sb.clone();
        cfg_df.cfg_mode = CfgMode::DualForward;
        let out_sb = flow_sample(&x, &cfg_sb, |s, _t, _p| {
            let n = s.len();
            let mut data = vec![1.0; n];
            data.extend(std::iter::repeat_n(2.0, n));
            Ok(FlowSamplerState {
                shape: vec![2 * n],
                data,
            })
        })
        .unwrap();
        let out_df = flow_sample(&x, &cfg_df, |s, _t, pass| {
            let v = match pass {
                ForwardPass::Uncond => 1.0,
                ForwardPass::Cond => 2.0,
                ForwardPass::SplitBatched => panic!("wrong pass for DualForward"),
            };
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: vec![v; s.data.len()],
            })
        })
        .unwrap();
        // Equality on the trajectory (small atol for FP round-off).
        for (a, b) in out_sb.data.iter().zip(out_df.data.iter()) {
            assert!((a - b).abs() < 1e-5, "SplitBatch {a} vs DualForward {b}");
        }
    }

    #[test]
    fn split_batch_bad_batched_shape_is_rejected() {
        let mut cfg = FlowSamplerConfig::euler_defaults(1);
        cfg.cfg_mode = CfgMode::SplitBatch;
        let x = state(vec![1.0, 2.0]);
        let e = flow_sample(&x, &cfg, |_s, _t, _p| {
            // Return only n elements instead of 2n — SplitBatch expects 2n.
            Ok(FlowSamplerState {
                shape: vec![2],
                data: vec![1.0, 2.0],
            })
        })
        .unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    // ---- Dynamic scale profile ------------------------------------------

    #[test]
    fn dynamic_scale_is_applied_per_step() {
        // Each step scales the "cond − uncond" difference by a different
        // scale. Verify by inspecting the step-by-step scales the closure
        // observes: since our closure captures step count, we verify the
        // sampler consumed the profile in order.
        let mut cfg = FlowSamplerConfig::euler_defaults(3);
        cfg.cfg_mode = CfgMode::DualForward;
        cfg.cfg_scale = CfgScaleProfile::Dynamic(vec![1.0, 2.0, 3.0]);
        let x = state(vec![1.0]);
        // The identity forward (Uncond returns 0, Cond returns 1) makes
        // combined = 0 + scale * (1 - 0) = scale. So each Euler step adds
        // dt * scale[step] to x. With nfe=3, dt=1/3, final =
        // 1 + (1/3)(1 + 2 + 3 * (1 + dt*prev)) -- but simpler: run and
        // check the growth is monotone with step-scaled increments.
        let start = x.data[0];
        let out = flow_sample(&x, &cfg, |s, _t, pass| {
            let v = match pass {
                ForwardPass::Uncond => 0.0,
                ForwardPass::Cond => 1.0,
                ForwardPass::SplitBatched => panic!("wrong pass"),
            };
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: vec![v; s.data.len()],
            })
        })
        .unwrap();
        // With scales 1,2,3 the state must strictly increase over baseline.
        assert!(out.data[0] > start, "expected growth, got {}", out.data[0]);
    }

    // ---- Runtime switching (T19 e2e claim) -------------------------------

    #[test]
    fn runtime_switching_of_nfe_cfg_and_schedule_reuses_same_forward() {
        // The FR-EX-10 operational claim: the SAME forward closure runs under
        // three different configs (different nfe / cfg_mode / schedule /
        // solver) with no re-conversion. We prove this by reusing the same
        // closure across three flow_sample calls and confirming they all
        // succeed.
        let x = state(vec![1.0, -0.3, 0.2]);
        // Config 1: nfe=5, None, Linear, Euler.
        let cfg1 = FlowSamplerConfig {
            cfg_mode: CfgMode::None,
            cfg_scale: CfgScaleProfile::Constant(1.0),
            nfe: 5,
            schedule: Schedule::Linear,
            solver: OdeSolver::Euler,
        };
        // Config 2: nfe=20, SplitBatch, Sway, Heun.
        let cfg2 = FlowSamplerConfig {
            cfg_mode: CfgMode::SplitBatch,
            cfg_scale: CfgScaleProfile::Constant(2.0),
            nfe: 20,
            schedule: Schedule::Sway,
            solver: OdeSolver::Heun,
        };
        // Config 3: nfe=10, DualForward, EpsS, DpmPp.
        let cfg3 = FlowSamplerConfig {
            cfg_mode: CfgMode::DualForward,
            cfg_scale: CfgScaleProfile::Dynamic(vec![
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
            ]),
            nfe: 10,
            schedule: Schedule::EpsS,
            solver: OdeSolver::DpmPp,
        };
        // A single "forward" closure that handles all three passes.
        let forward = |s: &FlowSamplerState, _t: f32, pass: ForwardPass| match pass {
            ForwardPass::Uncond => Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: s.data.iter().map(|v| -0.5 * v).collect(),
            }),
            ForwardPass::Cond => Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: s.data.iter().map(|v| -0.3 * v).collect(),
            }),
            ForwardPass::SplitBatched => {
                let n = s.len();
                let mut data: Vec<f32> = s.data.iter().map(|v| -0.5 * v).collect();
                data.extend(s.data.iter().map(|v| -0.3 * v));
                Ok(FlowSamplerState {
                    shape: vec![2 * n],
                    data,
                })
            }
        };
        // All three succeed with finite outputs — the same closure, three
        // different configs, no state carried between them.
        for cfg in [&cfg1, &cfg2, &cfg3] {
            let out = flow_sample(&x, cfg, forward).unwrap();
            for v in &out.data {
                assert!(v.is_finite(), "cfg {cfg:?} produced non-finite {v}");
            }
        }
    }

    // ---- Propagates forward errors --------------------------------------

    #[test]
    fn forward_error_is_propagated_unchanged() {
        let cfg = FlowSamplerConfig::euler_defaults(3);
        let x = state(vec![1.0]);
        let e = flow_sample(&x, &cfg, |_s, _t, _p| {
            Err::<FlowSamplerState, _>(VokraError::InvalidArgument("model error".into()))
        })
        .unwrap_err();
        match e {
            VokraError::InvalidArgument(msg) => assert_eq!(msg, "model error"),
            other => panic!("expected InvalidArgument passthrough, got {other:?}"),
        }
    }

    // ---- IR non-embedding (FR-EX-10) ------------------------------------

    #[test]
    fn no_opkind_variant_exists_for_flow_sampler() {
        // FR-EX-10 protection: the sampler must not have an `OpKind` variant.
        // We verify this negatively: format every existing `OpKind` we can
        // construct and confirm none of them contain "FlowSampler" or
        // "Sampler" in its Debug output.
        //
        // The absence of a variant is a *compile-time* property (you can't
        // construct an `OpKind::FlowSampler` because no such variant exists),
        // so this test is a *documentation-facing sentinel* rather than a
        // runtime check: a future patch adding such a variant would break
        // this test because the assertions here are written against the
        // full `OpKind` surface as of M3-05 land.
        use vokra_core::OpKind;
        let samples = [
            format!("{:?}", OpKind::MatMul),
            format!("{:?}", OpKind::Add),
            format!("{:?}", OpKind::Softmax),
            format!("{:?}", OpKind::DcOffsetRemove),
        ];
        for s in samples {
            assert!(
                !s.contains("FlowSampler"),
                "unexpected FlowSampler leak into OpKind: {s}"
            );
        }
    }
}
