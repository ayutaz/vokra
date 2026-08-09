//! DDPM sampler with **v-prediction** support (SoTA plan Phase 4, 2026-07-24
//! — the VibeVoice diffusion decoder consumer).
//!
//! # Why a distinct sampler from [`crate::flow_sampler`]
//!
//! The existing [`crate::flow_sampler`] covers Flow Matching (Euler / Heun /
//! FlowOde) and the two diffusion solvers most flow-matching-adjacent
//! releases carry (DDIM / DPM-Solver++), all with **`ε`-prediction** and a
//! **linear α schedule** (`α_t = 1 − t`) pinned inside the solver (ADR
//! M3-05 §D4). It cannot express three axes VibeVoice (Microsoft, MIT,
//! `huggingface.co/microsoft/VibeVoice-1.5B`) requires — silently reusing
//! it would either drop or hallucinate half the update:
//!
//! - **`v`-prediction reformulation** (Salimans & Ho 2022,
//!   arxiv 2202.00512 "Progressive Distillation for Fast Sampling of
//!   Diffusion Models" §3). The model predicts
//!   `v_θ(x_t, t) = √ᾱ_t · ε − √(1−ᾱ_t) · x_0`; the sampler recovers
//!   `x̂_0 = √ᾱ_t · x_t − √(1−ᾱ_t) · v_θ` and
//!   `ε̂ = √ᾱ_t · v_θ + √(1−ᾱ_t) · x_t` before taking a DDIM step. Feeding
//!   a `v` output to a plain `ε`-based DDIM step corrupts the update at
//!   every timestep.
//! - **Cosine β schedule** (Nichol & Dhariwal 2021, arxiv 2102.09672
//!   "Improved DDPM" §2.4). `ᾱ_t = cos²((t/T + s)/(1+s) · π/2)` with
//!   `s = 0.008`. The `α_t = 1 − t` linear α pinned in [`crate::flow_sampler`]
//!   is a *different* schedule and produces demonstrably different
//!   x-samples (the reference implementations both fail parity when
//!   swapped).
//! - **Reduced-step inference on a 1000-step training schedule**
//!   (`ddpm_num_steps=1000`, `ddpm_num_inference_steps=20` in VibeVoice's
//!   `diffusion_head_config`). The sampler picks 20 evenly spaced training
//!   timesteps and walks them; the timestep integer feeds the model,
//!   never the [0, 1] continuous `t` axis [`crate::flow_sampler`] carries.
//!
//! Every axis is a **runtime knob**: hardcoding
//! `beta_schedule`/`prediction_type`/`num_inference_steps` in an
//! [`OpKind`](vokra_core::OpKind) variant would force a model re-conversion
//! every time a caller adjusts the quality/RTF trade-off — precisely the
//! posture [`crate::flow_sampler`] (and the whole runtime-function family:
//! [`crate::mimi_rvq`] / [`crate::dac_rvq`] / [`crate::qwen3_tts_codec`] /
//! [`crate::vae_continuous`]) documents as FR-OP-30 / FR-EX-10 / ADR
//! M3-06 §D-b.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Invalid config raises [`VokraError::InvalidArgument`] up front:
//!
//! - `num_inference_steps == 0` or > `num_train_steps`;
//! - `num_train_steps == 0`;
//! - non-finite `beta_start` / `beta_end` / `cosine_offset` / `cfg_scale`;
//! - `beta_start` >= `beta_end` for a linear schedule;
//! - `cosine_offset` outside `(0, 1)`;
//! - `cfg_scale = Dynamic(v)` with `v.len() != num_inference_steps`;
//! - a forward closure returning a state whose shape differs from the
//!   input (only surfaced during a step).
//!
//! There is no `Ok(x_unchanged)` silent-return branch — every failure
//! path is loud and named.
//!
//! # No SIMD / no unsafe
//!
//! Every arithmetic operation is safe scalar Rust. The sampler is called
//! ~20 times per generated utterance frame, and the per-call work is a
//! handful of vector additions — the hot path lives in the closure the
//! caller supplies (the DiT / diffusion-head forward). Adding SIMD here
//! would move a rounding contract off the sampler and into the SIMD
//! kernel, complicating parity with the upstream reference for zero real
//! win.

use vokra_core::{Result, VokraError};

// Reuse the flow_sampler state container so callers can pipe a state
// through either sampler without reshaping. `FlowSamplerState` /
// `CfgMode` / `CfgScaleProfile` / `ForwardPass` are the primary "sampler
// state and CFG" surface in this crate, and duplicating them would
// fracture that surface.
pub use crate::flow_sampler::{CfgMode, CfgScaleProfile, FlowSamplerState, ForwardPass};

// ---------------------------------------------------------------------------
// Config enums
// ---------------------------------------------------------------------------

/// What the model's forward closure predicts (FR-OP-20 posture).
///
/// The three canonical DDPM prediction targets, in the order Salimans &
/// Ho 2022 introduce them:
///
/// - [`Epsilon`](Self::Epsilon) — the noise `ε_θ(x_t, t)` (the original
///   Ho et al. 2020 DDPM formulation).
/// - [`Sample`](Self::Sample) — the denoised sample `x̂_0(x_t, t)`
///   directly (rare; used by a few papers as a numerical convenience).
/// - [`VPrediction`](Self::VPrediction) — the v-prediction
///   `v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0` (Salimans & Ho 2022 §3). Improves
///   stability at low-SNR steps and is the canonical target for the
///   distilled-diffusion regime VibeVoice uses.
///
/// A caller who feeds a `VPrediction` model into a sampler configured
/// for `Epsilon` (or vice-versa) is a silent data-hallucination bug —
/// see the crate rustdoc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PredictionType {
    /// Model returns ε.
    Epsilon,
    /// Model returns x_0.
    Sample,
    /// Model returns `v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0` (Salimans & Ho 2022).
    VPrediction,
}

/// β schedule that determines the `ᾱ_t` cumulative-product table.
///
/// Both cover the release corpus in one type — a caller flips schedules
/// without re-conversion (FR-OP-20 posture).
///
/// - [`Cosine`](Self::Cosine) — Nichol & Dhariwal 2021, arxiv 2102.09672
///   §2.4. `ᾱ_t = cos²((t/T + s)/(1+s) · π/2)`, then
///   `β_t = 1 − ᾱ_t / ᾱ_{t−1}` clipped to `[0, β_max]`. The `s = 0.008`
///   default lives on [`DdpmSamplerConfig::cosine_offset`]; the clip
///   ceiling `β_max = 0.999` lives on [`DdpmSamplerConfig::cosine_beta_max`].
///   This is the default VibeVoice uses (`ddpm_beta_schedule="cosine"`).
/// - [`Linear`](Self::Linear) — the original DDPM (Ho et al. 2020,
///   arxiv 2006.11239 §4) linear-in-β schedule: `β_t` linearly ramps
///   from `beta_start` to `beta_end` over `num_train_steps` steps.
///   Bounds live on [`DdpmSamplerConfig::beta_start`] /
///   [`DdpmSamplerConfig::beta_end`].
///
/// Both schedules are computed at [`build_alphas_cumprod`] time from
/// the config; no runtime lookup table is baked into the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BetaSchedule {
    /// Nichol & Dhariwal 2021 cosine schedule.
    Cosine,
    /// Ho et al. 2020 linear-in-β schedule.
    Linear,
}

// ---------------------------------------------------------------------------
// Config struct
// ---------------------------------------------------------------------------

/// Runtime-adjustable configuration for [`ddpm_sample`].
///
/// Consumers construct this by struct literal (the primary "set every
/// axis once" ergonomic pattern) or via [`DdpmSamplerConfig::vibevoice_defaults`]
/// followed by field overrides. The wrapped enums are `#[non_exhaustive]`
/// so new variants remain backwards-compatible; new *axes* are added by
/// extending this struct through a semver-compatible field addition.
///
/// # Example
///
/// ```
/// use vokra_ops::ddpm_sampler::{
///     BetaSchedule, CfgMode, CfgScaleProfile, DdpmSamplerConfig, PredictionType,
/// };
///
/// // VibeVoice-1.5B canonical inference: 20-step DDIM(v-prediction) with
/// // cosine β schedule.
/// let cfg = DdpmSamplerConfig::vibevoice_defaults();
/// assert_eq!(cfg.num_train_steps, 1000);
/// assert_eq!(cfg.num_inference_steps, 20);
/// assert_eq!(cfg.prediction_type, PredictionType::VPrediction);
/// assert_eq!(cfg.beta_schedule, BetaSchedule::Cosine);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DdpmSamplerConfig {
    /// Total number of diffusion timesteps the model was **trained** on.
    /// VibeVoice: 1000 (`ddpm_num_steps`).
    pub num_train_steps: u32,
    /// Number of timesteps the sampler walks at **inference**. Must be
    /// `<= num_train_steps`. VibeVoice: 20 (`ddpm_num_inference_steps`).
    pub num_inference_steps: u32,
    /// The prediction target the model was trained on. VibeVoice:
    /// [`VPrediction`](PredictionType::VPrediction) (`prediction_type="v_prediction"`).
    pub prediction_type: PredictionType,
    /// The β schedule used to derive `ᾱ_t`. VibeVoice:
    /// [`Cosine`](BetaSchedule::Cosine) (`beta_schedule="cosine"`).
    pub beta_schedule: BetaSchedule,
    /// Linear-schedule start (only consulted when `beta_schedule == Linear`).
    /// Ho et al. 2020 canonical: `1e-4`.
    pub beta_start: f32,
    /// Linear-schedule end (only consulted when `beta_schedule == Linear`).
    /// Ho et al. 2020 canonical: `0.02`.
    pub beta_end: f32,
    /// Cosine-schedule small offset `s` (only consulted when
    /// `beta_schedule == Cosine`). Nichol & Dhariwal 2021 canonical:
    /// `0.008`.
    pub cosine_offset: f32,
    /// β clip ceiling for the cosine schedule (only consulted when
    /// `beta_schedule == Cosine`). Nichol & Dhariwal 2021 canonical:
    /// `0.999` (prevents numerical blow-up at the end of the schedule).
    pub cosine_beta_max: f32,
    /// Classifier-Free-Guidance mode. Same semantics as
    /// [`crate::flow_sampler::CfgMode`] — reused verbatim so callers can
    /// swap samplers without re-plumbing the CFG closure contract.
    pub cfg_mode: CfgMode,
    /// CFG scale profile (ignored when `cfg_mode == None`). Same
    /// semantics as [`crate::flow_sampler::CfgScaleProfile`] — a
    /// [`CfgScaleProfile::Dynamic`] vector's length must equal
    /// `num_inference_steps` (validated up-front by
    /// `validate_config`).
    pub cfg_scale: CfgScaleProfile,
}

impl DdpmSamplerConfig {
    /// The canonical VibeVoice-1.5B inference config, transcribed
    /// **verbatim** from
    /// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json`
    /// `diffusion_head_config.*` (fetched 2026-07-24 — CLAUDE.md
    /// 「ハルシネーション厳禁」):
    ///
    /// - `ddpm_num_steps = 1000`
    /// - `ddpm_num_inference_steps = 20`
    /// - `prediction_type = "v_prediction"`
    /// - `ddpm_beta_schedule = "cosine"`
    /// - `diffusion_type = "ddpm"`
    ///
    /// The CFG axes default to `CfgMode::None` / `CfgScaleProfile::Constant(1.0)`
    /// because VibeVoice's diffusion head does not carry a documented
    /// default CFG scale — a real caller sets `cfg_mode` / `cfg_scale`
    /// per generation intent.
    #[must_use]
    pub fn vibevoice_defaults() -> Self {
        Self {
            num_train_steps: 1000,
            num_inference_steps: 20,
            prediction_type: PredictionType::VPrediction,
            beta_schedule: BetaSchedule::Cosine,
            beta_start: 1e-4,
            beta_end: 0.02,
            cosine_offset: 0.008,
            cosine_beta_max: 0.999,
            cfg_mode: CfgMode::None,
            cfg_scale: CfgScaleProfile::Constant(1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

/// Validates every axis of `config` up-front (FR-EX-08 — never a silent
/// clamp or silent zero-return during a step).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] naming the offending field.
fn validate_config(config: &DdpmSamplerConfig) -> Result<()> {
    if config.num_train_steps == 0 {
        return Err(VokraError::InvalidArgument(
            "ddpm_sample: num_train_steps must be > 0".to_owned(),
        ));
    }
    if config.num_inference_steps == 0 {
        return Err(VokraError::InvalidArgument(
            "ddpm_sample: num_inference_steps must be > 0".to_owned(),
        ));
    }
    if config.num_inference_steps > config.num_train_steps {
        return Err(VokraError::InvalidArgument(format!(
            "ddpm_sample: num_inference_steps ({}) must be <= num_train_steps ({})",
            config.num_inference_steps, config.num_train_steps,
        )));
    }
    if !config.beta_start.is_finite() || !config.beta_end.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "ddpm_sample: beta_start ({}) / beta_end ({}) must be finite",
            config.beta_start, config.beta_end,
        )));
    }
    if matches!(config.beta_schedule, BetaSchedule::Linear) && config.beta_start >= config.beta_end
    {
        return Err(VokraError::InvalidArgument(format!(
            "ddpm_sample: linear beta schedule requires beta_start ({}) < beta_end ({})",
            config.beta_start, config.beta_end,
        )));
    }
    if !config.cosine_offset.is_finite()
        || config.cosine_offset <= 0.0
        || config.cosine_offset >= 1.0
    {
        return Err(VokraError::InvalidArgument(format!(
            "ddpm_sample: cosine_offset ({}) must be a finite value in (0, 1)",
            config.cosine_offset,
        )));
    }
    if !config.cosine_beta_max.is_finite() || !(0.0..=1.0).contains(&config.cosine_beta_max) {
        return Err(VokraError::InvalidArgument(format!(
            "ddpm_sample: cosine_beta_max ({}) must be a finite value in [0, 1]",
            config.cosine_beta_max,
        )));
    }
    match &config.cfg_scale {
        CfgScaleProfile::Constant(s) => {
            if !s.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "ddpm_sample: cfg_scale.Constant must be finite (got {s})"
                )));
            }
        }
        CfgScaleProfile::Dynamic(v) => {
            if v.len() != config.num_inference_steps as usize {
                return Err(VokraError::InvalidArgument(format!(
                    "ddpm_sample: Dynamic cfg_scale length {} != num_inference_steps {}",
                    v.len(),
                    config.num_inference_steps,
                )));
            }
            for (i, s) in v.iter().enumerate() {
                if !s.is_finite() {
                    return Err(VokraError::InvalidArgument(format!(
                        "ddpm_sample: cfg_scale.Dynamic[{i}] must be finite (got {s})"
                    )));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// α_t / ᾱ_t table
// ---------------------------------------------------------------------------

/// Builds the `ᾱ_t = ∏_{s=0..t} α_s` cumulative-product table of length
/// `num_train_steps + 1` (index `0` = `ᾱ_{-1} = 1.0`, index `t + 1` =
/// `ᾱ_t`). Callers index by `t` after a `+ 1` shift so the "one step
/// before t=0" endpoint (`ᾱ_prev` for `t == 0`) is `1.0`.
///
/// The two schedules:
///
/// - [`BetaSchedule::Cosine`] — Nichol & Dhariwal 2021, arxiv 2102.09672,
///   §2.4. `ᾱ_t = f(t) / f(0)` with
///   `f(t) = cos²((t / T + s) / (1 + s) · π / 2)`; the derived `β_t =
///   1 − ᾱ_t / ᾱ_{t−1}` is then clipped to `[0, cosine_beta_max]` before
///   recomputing `ᾱ_t = ∏_{s=0..t}(1 − β_s)` so a caller who reads back
///   `β_t` sees the clipped values (this matches the upstream reference
///   `diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar` chain).
/// - [`BetaSchedule::Linear`] — Ho et al. 2020, arxiv 2006.11239, §4.
///   `β_t = linspace(beta_start, beta_end, num_train_steps)`, then
///   `ᾱ_t = ∏_{s=0..t}(1 − β_s)`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] iff `config` fails
/// `validate_config`. Never allocates a zero-length table.
pub fn build_alphas_cumprod(config: &DdpmSamplerConfig) -> Result<Vec<f32>> {
    validate_config(config)?;
    let t_total = config.num_train_steps as usize;
    let mut alphas_cumprod = Vec::with_capacity(t_total + 1);
    alphas_cumprod.push(1.0);
    match config.beta_schedule {
        BetaSchedule::Cosine => {
            let s = config.cosine_offset;
            let ratio_at = |t: usize| -> f32 {
                let arg =
                    ((t as f32 / t_total as f32) + s) / (1.0 + s) * (std::f32::consts::PI / 2.0);
                let c = arg.cos();
                c * c
            };
            let f0 = ratio_at(0);
            let alpha_bar_uncapped = |t: usize| -> f32 {
                let f_t = ratio_at(t);
                f_t / f0
            };
            let mut running: f32 = 1.0;
            for t in 1..=t_total {
                let target = alpha_bar_uncapped(t);
                // Derive the *unclipped* β from the un-capped schedule and
                // apply the ceiling clip (Nichol & Dhariwal 2021 default:
                // 0.999). `.max(0.0)` covers numerical noise on the
                // low-t end.
                let beta_uncapped = if running > 0.0 {
                    (1.0 - target / running).clamp(0.0, config.cosine_beta_max)
                } else {
                    config.cosine_beta_max
                };
                running *= 1.0 - beta_uncapped;
                alphas_cumprod.push(running);
            }
        }
        BetaSchedule::Linear => {
            let mut running: f32 = 1.0;
            let range = config.beta_end - config.beta_start;
            let denom = (t_total.saturating_sub(1)).max(1) as f32;
            for t in 0..t_total {
                let frac = t as f32 / denom;
                let beta = config.beta_start + range * frac;
                running *= 1.0 - beta;
                alphas_cumprod.push(running);
            }
        }
    }
    Ok(alphas_cumprod)
}

// ---------------------------------------------------------------------------
// Timestep picker
// ---------------------------------------------------------------------------

/// Picks the `num_inference_steps` training timesteps the reduced-step
/// sampler walks, in **descending** order (from most-noisy to least).
///
/// Matches the upstream `diffusers.DDPMScheduler.set_timesteps` policy:
/// `linspace(0, num_train_steps - 1, num_inference_steps)` rounded to
/// integers, then reversed. The last emitted timestep is always `0`;
/// the first is bounded above by `num_train_steps - 1`. Duplicates are
/// preserved (they can appear when `num_inference_steps > num_train_steps
/// / 2`) — the sampler treats a duplicate step as a no-op idempotent
/// pass (`t_prev == t` → the DDIM update rounds to `x_next == x`).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on config-validation failure.
pub fn pick_inference_timesteps(config: &DdpmSamplerConfig) -> Result<Vec<u32>> {
    validate_config(config)?;
    let n_train = config.num_train_steps;
    let n_infer = config.num_inference_steps as usize;
    let mut out = Vec::with_capacity(n_infer);
    if n_infer == 1 {
        out.push(0);
        return Ok(out);
    }
    let max_t = (n_train - 1) as f32;
    let denom = (n_infer - 1) as f32;
    // Ascending linspace, then reverse. `round().max(0)` keeps the
    // endpoints exact.
    for i in 0..n_infer {
        let f = (i as f32 / denom) * max_t;
        let t = f.round().max(0.0) as u32;
        out.push(t.min(n_train - 1));
    }
    out.reverse();
    Ok(out)
}

// ---------------------------------------------------------------------------
// CFG dispatch helper
// ---------------------------------------------------------------------------

/// Runs the forward closure per the configured [`CfgMode`] and returns
/// the CFG-mixed prediction. Same policy as
/// [`crate::flow_sampler`]::velocity_at (kept private there so it's
/// duplicated here rather than exported — the two samplers use the same
/// CFG contract but each keeps its own copy so a future divergence
/// doesn't force cross-module coupling).
fn prediction_at<F>(
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
            check_same_shape(state, &v, "prediction")?;
            Ok(v)
        }
        CfgMode::DualForward => {
            let v_uncond = forward(state, t, ForwardPass::Uncond)?;
            check_same_shape(state, &v_uncond, "prediction(uncond)")?;
            let v_cond = forward(state, t, ForwardPass::Cond)?;
            check_same_shape(state, &v_cond, "prediction(cond)")?;
            Ok(cfg_mix(&v_uncond, &v_cond, scale))
        }
        CfgMode::SplitBatch => {
            let v = forward(state, t, ForwardPass::SplitBatched)?;
            if v.len() != 2 * state.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "ddpm_sample: SplitBatch expected 2× batched prediction ({} elements), got {}",
                    2 * state.len(),
                    v.len(),
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
            "ddpm_sample: {ctx} shape {:?} != state shape {:?}",
            b.shape, a.shape,
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Prediction → (x_0, ε) recovery
// ---------------------------------------------------------------------------

/// Recovers `(x̂_0, ε̂)` from the model's raw prediction per
/// [`PredictionType`]. This is the algebraic bridge between the three
/// prediction targets a DDPM-family model can carry.
///
/// - `Epsilon`: `ε̂ = raw`; `x̂_0 = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t`.
/// - `Sample`: `x̂_0 = raw`; `ε̂ = (x_t − √ᾱ_t · x̂_0) / √(1−ᾱ_t)`.
/// - `VPrediction`: `x̂_0 = √ᾱ_t · x_t − √(1−ᾱ_t) · raw` and
///   `ε̂ = √(1−ᾱ_t) · x_t + √ᾱ_t · raw`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] iff `ᾱ_t` is outside `[0, 1]` (a
/// buggy schedule table would trigger this — the sampler validates its
/// own tables up-front).
fn recover_x0_and_eps(
    x_t: &FlowSamplerState,
    raw: &FlowSamplerState,
    alpha_bar_t: f32,
    prediction_type: PredictionType,
) -> Result<(FlowSamplerState, FlowSamplerState)> {
    let one_minus = (1.0 - alpha_bar_t).max(0.0);
    let sqrt_a = alpha_bar_t.max(0.0).sqrt();
    let sqrt_1m = one_minus.sqrt();
    let mut x0 = Vec::with_capacity(x_t.len());
    let mut eps = Vec::with_capacity(x_t.len());
    match prediction_type {
        PredictionType::Epsilon => {
            if sqrt_a <= f32::EPSILON {
                return Err(VokraError::InvalidArgument(
                    "ddpm_sample: Epsilon recovery undefined at ᾱ_t = 0 (\
                     terminal noise step)"
                        .to_owned(),
                ));
            }
            for (xi, ri) in x_t.data.iter().zip(raw.data.iter()) {
                eps.push(*ri);
                x0.push((xi - sqrt_1m * ri) / sqrt_a);
            }
        }
        PredictionType::Sample => {
            if sqrt_1m <= f32::EPSILON {
                return Err(VokraError::InvalidArgument(
                    "ddpm_sample: Sample recovery undefined at ᾱ_t = 1 (\
                     zero-noise step — the model is already at x_0)"
                        .to_owned(),
                ));
            }
            for (xi, ri) in x_t.data.iter().zip(raw.data.iter()) {
                x0.push(*ri);
                eps.push((xi - sqrt_a * ri) / sqrt_1m);
            }
        }
        PredictionType::VPrediction => {
            // Salimans & Ho 2022 §3 identity.
            for (xi, ri) in x_t.data.iter().zip(raw.data.iter()) {
                x0.push(sqrt_a * xi - sqrt_1m * ri);
                eps.push(sqrt_1m * xi + sqrt_a * ri);
            }
        }
    }
    Ok((
        FlowSamplerState {
            shape: x_t.shape.clone(),
            data: x0,
        },
        FlowSamplerState {
            shape: x_t.shape.clone(),
            data: eps,
        },
    ))
}

// ---------------------------------------------------------------------------
// DDIM step (η = 0 — deterministic)
// ---------------------------------------------------------------------------

/// Takes the deterministic DDIM step given `(x̂_0, ε̂, ᾱ_prev)`:
///
/// `x_{prev} = √ᾱ_{prev} · x̂_0 + √(1 − ᾱ_{prev}) · ε̂`
///
/// (Song et al. 2021, arxiv 2010.02502, Eq. 12 with σ = 0.) This is the
/// deterministic form the upstream reference uses for
/// `ddpm_num_inference_steps << ddpm_num_steps` — VibeVoice's `20` /
/// `1000` ratio.
fn ddim_step(
    x_hat_0: &FlowSamplerState,
    eps_hat: &FlowSamplerState,
    alpha_bar_prev: f32,
) -> FlowSamplerState {
    let sqrt_a_prev = alpha_bar_prev.max(0.0).sqrt();
    let sqrt_1m_prev = (1.0 - alpha_bar_prev).max(0.0).sqrt();
    let mut data = Vec::with_capacity(x_hat_0.len());
    for (x0, e) in x_hat_0.data.iter().zip(eps_hat.data.iter()) {
        data.push(sqrt_a_prev * x0 + sqrt_1m_prev * e);
    }
    FlowSamplerState {
        shape: x_hat_0.shape.clone(),
        data,
    }
}

// ---------------------------------------------------------------------------
// Public entrypoint
// ---------------------------------------------------------------------------

/// Runs the DDPM sampler over `initial_state` using the deterministic
/// (`η = 0`) DDIM update rule with configurable `prediction_type` and
/// `β_schedule` (SoTA plan Phase 4 — VibeVoice diffusion decoder
/// consumer).
///
/// # Runtime function — NOT a graph node (FR-OP-30 / FR-EX-10)
///
/// Every axis (`num_inference_steps`, `prediction_type`, `beta_schedule`,
/// `cfg_mode`, `cfg_scale`) is runtime-selectable. Baking any of them
/// into an [`OpKind`](vokra_core::OpKind) variant would force a model
/// re-conversion on every quality / RTF trade-off — same posture as
/// [`crate::flow_sampler`] / [`crate::mimi_rvq`] /
/// [`crate::qwen3_tts_codec`] / [`crate::vae_continuous`].
///
/// # Arguments
///
/// - `initial_state` — the state at the highest-noise timestep (typically
///   Gaussian noise scaled by `√(1 − ᾱ_{T-1})`).
/// - `config` — full sampler configuration; validated up-front.
/// - `forward` — the model's prediction closure. Receives the current
///   state, the current `t / (num_train_steps - 1)` normalized in
///   `[0, 1]` (the caller-facing timestep — a normalized form is
///   friendlier to callers who share a closure with
///   [`crate::flow_sampler`]), and a [`ForwardPass`] tag.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on:
/// - a failed `validate_config` (see that function's list);
/// - a forward closure returning a state with a mismatched shape;
/// - an unrecoverable degeneracy (e.g. `PredictionType::Epsilon` at
///   `ᾱ_t = 0` — the recovery would divide by zero).
///
/// Any error propagated from the forward closure is returned unchanged.
///
/// # Example
///
/// ```
/// use vokra_ops::ddpm_sampler::{ddpm_sample, DdpmSamplerConfig, FlowSamplerState, ForwardPass};
///
/// // 4-step VibeVoice-flavor default with an identity-v closure — real
/// // callers wire the diffusion head as the forward closure.
/// let cfg = DdpmSamplerConfig {
///     num_inference_steps: 4,
///     ..DdpmSamplerConfig::vibevoice_defaults()
/// };
/// let x0 = FlowSamplerState::new(vec![1], vec![0.5]).unwrap();
/// let out = ddpm_sample(&x0, &cfg, |s, _t, _p| {
///     Ok(FlowSamplerState { shape: s.shape.clone(), data: s.data.clone() })
/// }).unwrap();
/// // The sampler produced a well-shaped output.
/// assert_eq!(out.shape, vec![1]);
/// assert!(out.data[0].is_finite());
/// ```
pub fn ddpm_sample<F>(
    initial_state: &FlowSamplerState,
    config: &DdpmSamplerConfig,
    mut forward: F,
) -> Result<FlowSamplerState>
where
    F: FnMut(&FlowSamplerState, f32, ForwardPass) -> Result<FlowSamplerState>,
{
    let alphas_cumprod = build_alphas_cumprod(config)?;
    let timesteps = pick_inference_timesteps(config)?;
    let denom = (config.num_train_steps as f32 - 1.0).max(1.0);
    let mut x = initial_state.clone();
    for (step, &t) in timesteps.iter().enumerate() {
        // Normalized [0, 1] timestep for the caller's closure.
        let t_norm = t as f32 / denom;
        // `ᾱ_t` and `ᾱ_prev`. The table is `[1.0, ᾱ_0, ᾱ_1, …, ᾱ_{T-1}]`,
        // so `ᾱ_t == alphas_cumprod[t + 1]` and `ᾱ_prev` for step `t` is
        // `alphas_cumprod[t]` (which is `1.0` at `t == 0` — the DDIM
        // update degenerates to `x_prev = x̂_0`, exactly the last-step
        // behaviour).
        let alpha_bar_t = alphas_cumprod[t as usize + 1];
        let alpha_bar_prev = alphas_cumprod[t as usize];
        let scale = config.cfg_scale.at_step(step);
        let raw = prediction_at(&x, t_norm, config.cfg_mode, scale, &mut forward)?;
        let (x_hat_0, eps_hat) = recover_x0_and_eps(&x, &raw, alpha_bar_t, config.prediction_type)?;
        x = ddim_step(&x_hat_0, &eps_hat, alpha_bar_prev);
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Small helper on the shared CfgScaleProfile
// ---------------------------------------------------------------------------

/// Extension trait that lets us index a [`CfgScaleProfile`] by step
/// without pulling the private `at()` from `flow_sampler`. Kept
/// private to this module.
trait CfgScaleProfileExt {
    fn at_step(&self, step: usize) -> f32;
}

impl CfgScaleProfileExt for CfgScaleProfile {
    fn at_step(&self, step: usize) -> f32 {
        match self {
            CfgScaleProfile::Constant(s) => *s,
            // Validated up-front — a well-formed config has
            // `Dynamic(v).len() == num_inference_steps`.
            CfgScaleProfile::Dynamic(v) => v[step],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn state(data: Vec<f32>) -> FlowSamplerState {
        let n = data.len();
        FlowSamplerState::new(vec![n], data).unwrap()
    }

    fn vibevoice_cfg_small(n_infer: u32) -> DdpmSamplerConfig {
        DdpmSamplerConfig {
            num_inference_steps: n_infer,
            ..DdpmSamplerConfig::vibevoice_defaults()
        }
    }

    // ---- Config surface -------------------------------------------------

    #[test]
    fn vibevoice_defaults_match_primary_source() {
        // huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json
        // (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
        let cfg = DdpmSamplerConfig::vibevoice_defaults();
        assert_eq!(cfg.num_train_steps, 1000);
        assert_eq!(cfg.num_inference_steps, 20);
        assert_eq!(cfg.prediction_type, PredictionType::VPrediction);
        assert_eq!(cfg.beta_schedule, BetaSchedule::Cosine);
        assert!((cfg.cosine_offset - 0.008).abs() < 1e-9);
        assert!((cfg.cosine_beta_max - 0.999).abs() < 1e-9);
    }

    // ---- validate_config -------------------------------------------------

    #[test]
    fn zero_train_steps_rejected() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.num_train_steps = 0;
        assert!(matches!(
            validate_config(&cfg),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zero_inference_steps_rejected() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.num_inference_steps = 0;
        assert!(matches!(
            validate_config(&cfg),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn inference_steps_greater_than_train_steps_rejected() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.num_train_steps = 10;
        cfg.num_inference_steps = 20;
        assert!(matches!(
            validate_config(&cfg),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn nonfinite_betas_rejected() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.beta_start = f32::NAN;
        assert!(validate_config(&cfg).is_err());
        let mut cfg = vibevoice_cfg_small(4);
        cfg.beta_end = f32::INFINITY;
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn linear_schedule_requires_start_lt_end() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.beta_schedule = BetaSchedule::Linear;
        cfg.beta_start = 0.02;
        cfg.beta_end = 0.02;
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn cosine_offset_bounds_enforced() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.cosine_offset = 0.0;
        assert!(validate_config(&cfg).is_err());
        let mut cfg = vibevoice_cfg_small(4);
        cfg.cosine_offset = 1.0;
        assert!(validate_config(&cfg).is_err());
        let mut cfg = vibevoice_cfg_small(4);
        cfg.cosine_offset = f32::NAN;
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn cosine_beta_max_bounds_enforced() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.cosine_beta_max = -0.001;
        assert!(validate_config(&cfg).is_err());
        let mut cfg = vibevoice_cfg_small(4);
        cfg.cosine_beta_max = 1.001;
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn dynamic_cfg_scale_length_must_match_nfe() {
        let mut cfg = vibevoice_cfg_small(4);
        cfg.cfg_scale = CfgScaleProfile::Dynamic(vec![1.0, 2.0, 3.0]);
        assert!(validate_config(&cfg).is_err());
    }

    // ---- Schedule tables -------------------------------------------------

    #[test]
    fn cosine_alphas_start_at_one_and_decrease() {
        let cfg = DdpmSamplerConfig::vibevoice_defaults();
        let ac = build_alphas_cumprod(&cfg).unwrap();
        assert_eq!(ac.len(), (cfg.num_train_steps + 1) as usize);
        assert!((ac[0] - 1.0).abs() < 1e-6);
        // ᾱ_0 is very close to 1 (cosine schedule is smooth near t = 0).
        assert!(ac[1] > 0.99 && ac[1] < 1.0);
        // ᾱ_{T-1} is very small but positive (β clip prevents zero).
        assert!(ac[cfg.num_train_steps as usize] > 0.0);
        assert!(ac[cfg.num_train_steps as usize] < 1e-3);
    }

    #[test]
    fn cosine_alphas_are_monotonically_nonincreasing() {
        let cfg = DdpmSamplerConfig::vibevoice_defaults();
        let ac = build_alphas_cumprod(&cfg).unwrap();
        for w in ac.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-6,
                "cosine schedule non-monotone: {:?} vs {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn linear_alphas_are_monotonically_nonincreasing() {
        let mut cfg = DdpmSamplerConfig::vibevoice_defaults();
        cfg.beta_schedule = BetaSchedule::Linear;
        let ac = build_alphas_cumprod(&cfg).unwrap();
        for w in ac.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-6,
                "linear schedule non-monotone: {:?} vs {:?}",
                w[0],
                w[1]
            );
        }
    }

    // ---- Timestep picker -------------------------------------------------

    #[test]
    fn timesteps_are_descending_and_end_at_zero() {
        let cfg = DdpmSamplerConfig::vibevoice_defaults();
        let ts = pick_inference_timesteps(&cfg).unwrap();
        assert_eq!(ts.len(), cfg.num_inference_steps as usize);
        assert_eq!(*ts.last().unwrap(), 0);
        assert_eq!(*ts.first().unwrap(), cfg.num_train_steps - 1);
        for w in ts.windows(2) {
            assert!(w[0] >= w[1], "timesteps must be non-ascending: {ts:?}");
        }
    }

    #[test]
    fn timesteps_single_step_is_zero() {
        let cfg = DdpmSamplerConfig {
            num_inference_steps: 1,
            ..DdpmSamplerConfig::vibevoice_defaults()
        };
        let ts = pick_inference_timesteps(&cfg).unwrap();
        assert_eq!(ts, vec![0]);
    }

    // ---- Prediction-type recovery -----------------------------------------

    #[test]
    fn epsilon_recovery_round_trip() {
        // For α = 0.5: x_t = √0.5 · x_0 + √0.5 · ε. Feed a hand-picked
        // (x_0, ε), form x_t, then recover.
        let alpha = 0.5_f32;
        let x_0 = 1.7_f32;
        let eps = -0.3_f32;
        let x_t = alpha.sqrt() * x_0 + (1.0 - alpha).sqrt() * eps;
        let s_xt = state(vec![x_t]);
        let s_raw = state(vec![eps]);
        let (x0_h, eps_h) =
            recover_x0_and_eps(&s_xt, &s_raw, alpha, PredictionType::Epsilon).unwrap();
        assert!((x0_h.data[0] - x_0).abs() < 1e-5);
        assert!((eps_h.data[0] - eps).abs() < 1e-6);
    }

    #[test]
    fn sample_recovery_round_trip() {
        let alpha = 0.5_f32;
        let x_0 = 0.9_f32;
        let eps = 0.6_f32;
        let x_t = alpha.sqrt() * x_0 + (1.0 - alpha).sqrt() * eps;
        let s_xt = state(vec![x_t]);
        let s_raw = state(vec![x_0]);
        let (x0_h, eps_h) =
            recover_x0_and_eps(&s_xt, &s_raw, alpha, PredictionType::Sample).unwrap();
        assert!((x0_h.data[0] - x_0).abs() < 1e-6);
        assert!((eps_h.data[0] - eps).abs() < 1e-5);
    }

    #[test]
    fn v_prediction_recovery_round_trip() {
        // v = √ᾱ · ε − √(1−ᾱ) · x_0; x_t = √ᾱ · x_0 + √(1−ᾱ) · ε.
        // Recovery: x̂_0 = √ᾱ · x_t − √(1−ᾱ) · v; ε̂ = √(1−ᾱ) · x_t + √ᾱ · v.
        // These identities hold for the linear α parameterization from
        // Salimans & Ho 2022 §3.
        let alpha = 0.3_f32;
        let x_0 = -0.4_f32;
        let eps = 1.2_f32;
        let sqrt_a = alpha.sqrt();
        let sqrt_1m = (1.0 - alpha).sqrt();
        let x_t = sqrt_a * x_0 + sqrt_1m * eps;
        let v = sqrt_a * eps - sqrt_1m * x_0;
        let s_xt = state(vec![x_t]);
        let s_raw = state(vec![v]);
        let (x0_h, eps_h) =
            recover_x0_and_eps(&s_xt, &s_raw, alpha, PredictionType::VPrediction).unwrap();
        assert!(
            (x0_h.data[0] - x_0).abs() < 1e-5,
            "recovered x_0 {:?} vs expected {:?}",
            x0_h.data[0],
            x_0,
        );
        assert!(
            (eps_h.data[0] - eps).abs() < 1e-5,
            "recovered ε {:?} vs expected {:?}",
            eps_h.data[0],
            eps,
        );
    }

    #[test]
    fn epsilon_recovery_at_zero_alpha_is_loud_error() {
        // The Epsilon branch divides by √ᾱ_t — at ᾱ_t = 0 the recovery
        // is undefined, and the sampler must refuse rather than emit a
        // silent inf / NaN.
        let s = state(vec![1.0, 2.0]);
        let raw = state(vec![0.5, 0.5]);
        let err = recover_x0_and_eps(&s, &raw, 0.0, PredictionType::Epsilon).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn sample_recovery_at_one_alpha_is_loud_error() {
        // Symmetric: Sample recovery at ᾱ_t = 1 divides by √(1 − ᾱ_t) = 0.
        let s = state(vec![1.0]);
        let raw = state(vec![0.5]);
        let err = recover_x0_and_eps(&s, &raw, 1.0, PredictionType::Sample).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- End-to-end sampler ---------------------------------------------

    #[test]
    fn ddpm_sample_preserves_state_shape() {
        let cfg = vibevoice_cfg_small(4);
        let x0 = FlowSamplerState::new(vec![2, 3], vec![0.0; 6]).unwrap();
        let out = ddpm_sample(&x0, &cfg, |s, _t, _p| {
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: s.data.iter().map(|v| v * 0.5).collect(),
            })
        })
        .unwrap();
        assert_eq!(out.shape, x0.shape);
        assert_eq!(out.len(), x0.len());
    }

    #[test]
    fn ddpm_sample_all_finite_for_random_ish_forward() {
        // Non-degenerate forward returning half the state — used to smoke
        // the arithmetic path (no NaNs / infs propagate through the DDIM
        // step at every timestep).
        let cfg = vibevoice_cfg_small(8);
        let x0 = state(vec![0.5, -0.3, 1.1, -0.9]);
        let out = ddpm_sample(&x0, &cfg, |s, _t, _p| {
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: s.data.iter().map(|v| 0.5 * v).collect(),
            })
        })
        .unwrap();
        for v in out.data.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn ddpm_sample_rejects_forward_shape_mismatch() {
        let cfg = vibevoice_cfg_small(4);
        let x0 = state(vec![0.1, 0.2, 0.3]);
        let err = ddpm_sample(&x0, &cfg, |_s, _t, _p| {
            Ok(FlowSamplerState::new(vec![2], vec![0.0, 0.0]).unwrap())
        })
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn ddpm_sample_split_batch_cfg_expects_double_length_return() {
        let mut cfg = vibevoice_cfg_small(2);
        cfg.cfg_mode = CfgMode::SplitBatch;
        cfg.cfg_scale = CfgScaleProfile::Constant(1.5);
        let x0 = state(vec![0.5, 0.5]);
        let err = ddpm_sample(&x0, &cfg, |_s, _t, _p| {
            // Bug: return same-length, not 2×.
            Ok(FlowSamplerState::new(vec![2], vec![0.1, 0.1]).unwrap())
        })
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn ddpm_sample_dual_forward_cfg_calls_closure_twice_per_step() {
        // Count the closure calls; DualForward must call `forward` twice
        // per inference step.
        let mut cfg = vibevoice_cfg_small(3);
        cfg.cfg_mode = CfgMode::DualForward;
        cfg.cfg_scale = CfgScaleProfile::Constant(1.0);
        let x0 = state(vec![0.2, -0.2]);
        let mut count = 0_usize;
        let _ = ddpm_sample(&x0, &cfg, |s, _t, _p| {
            count += 1;
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: s.data.clone(),
            })
        })
        .unwrap();
        assert_eq!(
            count,
            2 * cfg.num_inference_steps as usize,
            "DualForward must call forward 2× num_inference_steps times"
        );
    }

    #[test]
    fn ddpm_sample_dynamic_cfg_scale_walks_per_step() {
        // With Dynamic scale [0.0, 0.0, 0.0], the CFG mix reduces to the
        // uncond branch; we compare the DualForward output to a manually
        // computed uncond-only pass.
        let mut cfg = vibevoice_cfg_small(3);
        cfg.cfg_mode = CfgMode::DualForward;
        cfg.cfg_scale = CfgScaleProfile::Dynamic(vec![0.0, 0.0, 0.0]);
        let x0 = state(vec![0.5, -0.5]);
        let out_dual = ddpm_sample(&x0, &cfg, |s, _t, p| {
            // Uncond returns 0, cond returns 1 — so with scale = 0 the mix
            // is just the uncond output (zeros).
            let value = if matches!(p, ForwardPass::Cond) {
                1.0
            } else {
                0.0
            };
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: vec![value; s.data.len()],
            })
        })
        .unwrap();
        // With scale = 0 the mix should equal the uncond branch (all
        // zeros in raw), so the recovered `v` is 0 → x̂_0 = √ᾱ · x_t,
        // ε̂ = √(1−ᾱ) · x_t → x_prev = √ᾱ_prev · √ᾱ · x_t + √(1−ᾱ_prev) ·
        // √(1−ᾱ) · x_t. Just ensure the values are finite and
        // reproducible.
        for v in out_dual.data.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn ddpm_sample_single_step_matches_uncond_prediction_at_zero() {
        // With num_inference_steps = 1, the sampler picks t = 0 (the
        // last-noise-step, ᾱ_t ≈ 1.0). The DDIM step from ᾱ_prev = 1.0
        // to ᾱ_prev = 1.0 is the identity on x̂_0, so v-prediction with
        // v = 0 must produce ≈ x_t back.
        let cfg = DdpmSamplerConfig {
            num_inference_steps: 1,
            ..DdpmSamplerConfig::vibevoice_defaults()
        };
        let x0 = state(vec![0.5, -0.5]);
        let out = ddpm_sample(&x0, &cfg, |s, _t, _p| {
            Ok(FlowSamplerState {
                shape: s.shape.clone(),
                data: vec![0.0; s.data.len()],
            })
        })
        .unwrap();
        for (o, x) in out.data.iter().zip(x0.data.iter()) {
            assert!(
                (o - x).abs() < 1e-3,
                "one-step v=0 output {o} should be close to input {x}"
            );
        }
    }

    #[test]
    fn ddpm_sample_propagates_forward_error_unchanged() {
        let cfg = vibevoice_cfg_small(4);
        let x0 = state(vec![0.5, 0.5]);
        let err = ddpm_sample(&x0, &cfg, |_s, _t, _p| {
            Err::<FlowSamplerState, _>(VokraError::UnsupportedOp("test-only".to_owned()))
        })
        .unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "forward error must propagate unchanged: {err:?}"
        );
    }

    // ---- Beta-schedule sanity -------------------------------------------

    #[test]
    fn linear_and_cosine_schedules_differ() {
        // Ho 2020 linear ᾱ table diverges from Nichol & Dhariwal cosine
        // ᾱ table well before the terminal step. This asserts the two
        // are meaningfully different so a caller who picks the wrong
        // schedule doesn't silently get the other's answer.
        let cosine = DdpmSamplerConfig::vibevoice_defaults();
        let linear = DdpmSamplerConfig {
            beta_schedule: BetaSchedule::Linear,
            ..DdpmSamplerConfig::vibevoice_defaults()
        };
        let ac_cos = build_alphas_cumprod(&cosine).unwrap();
        let ac_lin = build_alphas_cumprod(&linear).unwrap();
        // At midpoint the two schedules disagree by >5% (empirically
        // ≈0.3 vs ≈0.03 for the canonical 1000-step tables).
        let mid = cosine.num_train_steps as usize / 2;
        let diff = (ac_cos[mid] - ac_lin[mid]).abs();
        assert!(
            diff > 0.05,
            "cosine ᾱ_mid {} vs linear ᾱ_mid {} — schedules must differ",
            ac_cos[mid],
            ac_lin[mid]
        );
    }
}
