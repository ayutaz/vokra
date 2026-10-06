//! Deterministic classifier-free guidance sampling for VibeVoice Realtime.
//!
//! This module is intentionally only the diffusion sampling seam.  It does
//! not implement the Qwen streaming language models, acoustic tokenizer, or
//! PCM synthesis.  The CFG loop follows the fixed Microsoft VibeVoice source
//! at `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`,
//! `modeling_vibevoice_streaming_inference.py::sample_speech_tokens` (the
//! `condition/negative` pair is evaluated at the same current latent before
//! guidance is applied).  The scheduler contract is the fixed upstream
//! `vibevoice/schedule/dpm_solver.py` blob
//! `b392a480faef86a9e3518fc2c44815ff4dd17171`: 1000 training steps, cosine
//! betas, v-prediction, DPM-Solver++ midpoint, zero final sigma, and 20
//! inference steps.  The existing native scheduler is reused only for that
//! matching contract; this module does not claim independent numerical parity
//! with the Python implementation.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use super::diffusion::VibeVoiceStreamingDiffusionHead;
use crate::vibevoice::VibeVoiceDpmSolverMultistep;

/// Width of one Realtime acoustic latent.
pub const VIBEVOICE_REALTIME_LATENT_WIDTH: usize = 64;
/// Width of one Realtime diffusion condition.
pub const VIBEVOICE_REALTIME_CONDITION_WIDTH: usize = 896;
/// Authenticated scheduler training-step count.
pub const VIBEVOICE_REALTIME_TRAIN_STEPS: usize = 1_000;
/// Authenticated scheduler inference-step count.
pub const VIBEVOICE_REALTIME_INFERENCE_STEPS: usize = 20;

/// Runs the bounded Realtime diffusion CFG loop on an authenticated head.
///
/// `positive_condition` and `negative_condition` are the two 896-wide hidden
/// states supplied by the staged streaming runtime.  `initial_noise` is a
/// caller-owned 64-wide draw; no random generator is hidden in this API.
/// Every step evaluates both conditions against the same current latent, then
/// applies `uncond + guidance_scale * (cond - uncond)` before the scheduler
/// advances it.
///
/// The learned prediction head runs on its selected backend.  The DPM
/// scheduler is an explicit host-control stage over the 64-wide latent; it
/// does not bind learned weights and never migrates the head or its tensors to
/// CPU.  Realtime currently composes this path only with CPU or Metal heads;
/// other backends fail closed rather than becoming an implicit fallback.
pub fn sample_vibevoice_realtime_cfg(
    head: &VibeVoiceStreamingDiffusionHead,
    positive_condition: &[f32],
    negative_condition: &[f32],
    initial_noise: &[f32],
    guidance_scale: f32,
) -> Result<Vec<f32>> {
    ensure_realtime_head_backend(head.backend())?;
    sample_with_predictor(
        positive_condition,
        negative_condition,
        initial_noise,
        guidance_scale,
        |sample, condition, timestep| head.forward(sample, condition, timestep),
    )
}

/// Diagnostic-only variant of [`sample_vibevoice_realtime_cfg`]. The callback
/// receives the two native prediction arrays before CFG combination at every
/// scheduler step; it does not alter either array or the scheduler state.
pub(crate) fn sample_vibevoice_realtime_cfg_with_observer<F>(
    head: &VibeVoiceStreamingDiffusionHead,
    positive_condition: &[f32],
    negative_condition: &[f32],
    initial_noise: &[f32],
    guidance_scale: f32,
    observe: F,
) -> Result<Vec<f32>>
where
    F: FnMut(usize, usize, &[f32], &[f32]) -> Result<()>,
{
    ensure_realtime_head_backend(head.backend())?;
    sample_with_predictor_observed(
        positive_condition,
        negative_condition,
        initial_noise,
        guidance_scale,
        |sample, condition, timestep| head.forward(sample, condition, timestep),
        observe,
    )
}

/// Runs the production scheduler/CFG loop with a caller-provided prediction
/// seam.  The seam is private so model-free tests can exercise the exact loop
/// without constructing a checkpoint-backed head; it is not an alternate
/// public model API or an official numerical reference.
fn sample_with_predictor<P>(
    positive_condition: &[f32],
    negative_condition: &[f32],
    initial_noise: &[f32],
    guidance_scale: f32,
    mut predict: P,
) -> Result<Vec<f32>>
where
    P: FnMut(&[f32], &[f32], f32) -> Result<Vec<f32>>,
{
    sample_with_predictor_optional(
        positive_condition,
        negative_condition,
        initial_noise,
        guidance_scale,
        &mut predict,
        None,
    )
}

fn sample_with_predictor_observed<P, F>(
    positive_condition: &[f32],
    negative_condition: &[f32],
    initial_noise: &[f32],
    guidance_scale: f32,
    mut predict: P,
    mut observe: F,
) -> Result<Vec<f32>>
where
    P: FnMut(&[f32], &[f32], f32) -> Result<Vec<f32>>,
    F: FnMut(usize, usize, &[f32], &[f32]) -> Result<()>,
{
    sample_with_predictor_optional(
        positive_condition,
        negative_condition,
        initial_noise,
        guidance_scale,
        &mut predict,
        Some(&mut observe),
    )
}

fn sample_with_predictor_optional<P>(
    positive_condition: &[f32],
    negative_condition: &[f32],
    initial_noise: &[f32],
    guidance_scale: f32,
    mut predict: P,
    mut observe: Option<&mut dyn FnMut(usize, usize, &[f32], &[f32]) -> Result<()>>,
) -> Result<Vec<f32>>
where
    P: FnMut(&[f32], &[f32], f32) -> Result<Vec<f32>>,
{
    validate_inputs(
        positive_condition,
        negative_condition,
        initial_noise,
        guidance_scale,
    )?;
    let mut scheduler = VibeVoiceDpmSolverMultistep::new(
        VIBEVOICE_REALTIME_TRAIN_STEPS,
        VIBEVOICE_REALTIME_INFERENCE_STEPS,
    )?;
    scheduler.reset();
    let mut sample = initial_noise.to_vec();
    for (diffusion_step, timestep) in scheduler.timesteps().to_vec().into_iter().enumerate() {
        // Both predictions deliberately receive the same unchanged sample.
        // Do not mutate or replace it between branches: that would alter the
        // upstream CFG contract and make the two predictions asymmetric.
        let conditional = predict(&sample, positive_condition, timestep as f32)?;
        let unconditional = predict(&sample, negative_condition, timestep as f32)?;
        let guided = combine_cfg(&conditional, &unconditional, guidance_scale)?;
        if let Some(observe) = observe.as_deref_mut() {
            observe(diffusion_step, timestep, &conditional, &unconditional)?;
        }
        sample = scheduler.step(&guided, timestep, &sample)?.sample;
    }
    Ok(sample)
}

fn ensure_realtime_head_backend(backend: BackendKind) -> Result<()> {
    match backend {
        BackendKind::Cpu | BackendKind::Metal => Ok(()),
        _ => Err(VokraError::UnsupportedOp(format!(
            "vibevoice realtime CFG head backend {backend:?} is not covered by the composite; no CPU fallback"
        ))),
    }
}

fn validate_inputs(
    positive_condition: &[f32],
    negative_condition: &[f32],
    initial_noise: &[f32],
    guidance_scale: f32,
) -> Result<()> {
    if positive_condition.len() != VIBEVOICE_REALTIME_CONDITION_WIDTH
        || negative_condition.len() != VIBEVOICE_REALTIME_CONDITION_WIDTH
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG conditions must each have width 896".to_owned(),
        ));
    }
    if initial_noise.len() != VIBEVOICE_REALTIME_LATENT_WIDTH {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG initial noise must have width 64".to_owned(),
        ));
    }
    if !guidance_scale.is_finite() {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG scale must be finite".to_owned(),
        ));
    }
    if positive_condition
        .iter()
        .chain(negative_condition)
        .chain(initial_noise)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG inputs must be finite".to_owned(),
        ));
    }
    Ok(())
}

fn combine_cfg(
    conditional: &[f32],
    unconditional: &[f32],
    guidance_scale: f32,
) -> Result<Vec<f32>> {
    if conditional.len() != VIBEVOICE_REALTIME_LATENT_WIDTH
        || unconditional.len() != VIBEVOICE_REALTIME_LATENT_WIDTH
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG predictions must each have width 64".to_owned(),
        ));
    }
    if !guidance_scale.is_finite()
        || conditional
            .iter()
            .chain(unconditional)
            .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG predictions and scale must be finite".to_owned(),
        ));
    }
    let guided: Vec<f32> = conditional
        .iter()
        .zip(unconditional)
        .map(|(&cond, &uncond)| uncond + guidance_scale * (cond - uncond))
        .collect();
    if guided.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime CFG prediction became non-finite".to_owned(),
        ));
    }
    Ok(guided)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_uses_conditional_minus_unconditional_in_upstream_order() {
        let mut conditional = [0.0_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let mut unconditional = [0.0_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        conditional[0] = 4.0;
        conditional[1] = -2.0;
        unconditional[0] = 1.0;
        unconditional[1] = 3.0;
        let guided = combine_cfg(&conditional, &unconditional, 2.0).unwrap();
        assert_eq!(&guided[..2], &[7.0, -7.0]);
        assert!(guided[2..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn production_loop_uses_same_latent_for_both_branches_and_twenty_steps() {
        let mut positive = vec![0.0_f32; VIBEVOICE_REALTIME_CONDITION_WIDTH];
        let mut negative = vec![0.0_f32; VIBEVOICE_REALTIME_CONDITION_WIDTH];
        positive[0] = 2.0;
        negative[0] = 1.0;
        let initial = vec![0.125_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let mut calls: Vec<(Vec<f32>, bool, f32)> = Vec::new();
        let output = sample_with_predictor(
            &positive,
            &negative,
            &initial,
            3.0,
            |sample, condition, timestep| {
                let is_positive = condition[0] == 2.0;
                calls.push((sample.to_vec(), is_positive, timestep));
                Ok(vec![
                    if is_positive { 2.0 } else { 1.0 };
                    VIBEVOICE_REALTIME_LATENT_WIDTH
                ])
            },
        )
        .unwrap();
        assert_eq!(calls.len(), VIBEVOICE_REALTIME_INFERENCE_STEPS * 2);
        for pair in calls.chunks_exact(2) {
            assert!(pair[0].1, "positive branch must run first");
            assert!(!pair[1].1, "negative branch must run second");
            assert_eq!(pair[0].0, pair[1].0, "branches must share current latent");
            assert_eq!(pair[0].2, pair[1].2, "branches must share timestep");
        }

        // The synthetic predictor returns cond=2 and uncond=1.  With scale
        // 3, the production CFG expression must therefore pass 4 to every
        // scheduler step.  This checks the full loop, not official parity.
        let mut expected_scheduler = VibeVoiceDpmSolverMultistep::new(
            VIBEVOICE_REALTIME_TRAIN_STEPS,
            VIBEVOICE_REALTIME_INFERENCE_STEPS,
        )
        .unwrap();
        let expected_timesteps = expected_scheduler.timesteps().to_vec();
        let prediction = vec![4.0_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let mut expected = initial;
        for timestep in expected_timesteps {
            expected = expected_scheduler
                .step(&prediction, timestep, &expected)
                .unwrap()
                .sample;
        }
        assert_eq!(output, expected);
    }

    #[test]
    fn observed_loop_is_borrowed_ordered_and_error_stops_later_events() {
        let positive = vec![0.0_f32; VIBEVOICE_REALTIME_CONDITION_WIDTH];
        let negative = vec![1.0_f32; VIBEVOICE_REALTIME_CONDITION_WIDTH];
        let initial = vec![0.125_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let baseline = sample_with_predictor(
            &positive,
            &negative,
            &initial,
            2.0,
            |sample, condition, timestep| {
                Ok(vec![
                    condition[0] + sample[0] * 0.25 + timestep * 0.001;
                    VIBEVOICE_REALTIME_LATENT_WIDTH
                ])
            },
        )
        .unwrap();
        let mut observations = Vec::new();
        let mut prediction_calls = Vec::new();
        let output = sample_with_predictor_observed(
            &positive,
            &negative,
            &initial,
            2.0,
            |sample, condition, timestep| {
                prediction_calls.push((sample[0], condition[0], timestep));
                Ok(vec![
                    condition[0] + sample[0] * 0.25 + timestep * 0.001;
                    VIBEVOICE_REALTIME_LATENT_WIDTH
                ])
            },
            |diffusion_step, timestep, conditional, unconditional| {
                observations.push((diffusion_step, timestep, conditional[0], unconditional[0]));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(output, baseline, "observation must not alter native output");
        assert_eq!(observations.len(), VIBEVOICE_REALTIME_INFERENCE_STEPS);
        assert_eq!(
            prediction_calls.len(),
            VIBEVOICE_REALTIME_INFERENCE_STEPS * 2
        );
        for pair in prediction_calls.chunks_exact(2) {
            assert_eq!(pair[0].0, pair[1].0, "CFG branches share current sample");
            assert_eq!(pair[0].2, pair[1].2, "CFG branches share timestep");
        }
        assert!(
            observations
                .iter()
                .all(|(_, _, conditional, unconditional)| conditional != unconditional),
            "observed branches must carry distinct deterministic predictions"
        );
        assert!(
            observations
                .windows(2)
                .all(|pair| pair[0].0 + 1 == pair[1].0)
        );

        let mut callback_count = 0;
        let error = sample_with_predictor_observed(
            &positive,
            &negative,
            &initial,
            2.0,
            |_sample, condition, _timestep| Ok(vec![condition[0]; VIBEVOICE_REALTIME_LATENT_WIDTH]),
            |diffusion_step, _timestep, _conditional, _unconditional| {
                callback_count += 1;
                if diffusion_step == 3 {
                    Err(VokraError::ModelLoad(
                        "synthetic observer failure".to_owned(),
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert!(error.is_err());
        assert_eq!(callback_count, 4, "observer failure must stop later steps");

        let mut malformed_callbacks = 0;
        let malformed = sample_with_predictor_observed(
            &positive,
            &negative,
            &initial,
            2.0,
            |_sample, _condition, _timestep| Ok(vec![0.0; VIBEVOICE_REALTIME_LATENT_WIDTH - 1]),
            |_diffusion_step, _timestep, _conditional, _unconditional| {
                malformed_callbacks += 1;
                Ok(())
            },
        );
        assert!(malformed.is_err());
        assert_eq!(
            malformed_callbacks, 0,
            "malformed prediction emits no observation"
        );
    }

    #[test]
    fn scheduler_is_reset_and_runs_exactly_twenty_steps() {
        let mut scheduler = VibeVoiceDpmSolverMultistep::new(
            VIBEVOICE_REALTIME_TRAIN_STEPS,
            VIBEVOICE_REALTIME_INFERENCE_STEPS,
        )
        .unwrap();
        assert_eq!(
            scheduler.timesteps().len(),
            VIBEVOICE_REALTIME_INFERENCE_STEPS
        );
        assert_eq!(scheduler.timesteps().first(), Some(&999));
        assert_eq!(scheduler.timesteps().last(), Some(&50));
        let initial = vec![0.125_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let prediction = vec![0.0_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let mut sample = initial.clone();
        for timestep in scheduler.timesteps().to_vec() {
            sample = scheduler
                .step(&prediction, timestep, &sample)
                .unwrap()
                .sample;
        }
        assert!(sample.iter().all(|value| value.is_finite()));
        scheduler.reset();
        let reset_timestep = scheduler.timesteps()[0];
        let replay = scheduler
            .step(&prediction, reset_timestep, &initial)
            .unwrap();
        let mut fresh = VibeVoiceDpmSolverMultistep::new(
            VIBEVOICE_REALTIME_TRAIN_STEPS,
            VIBEVOICE_REALTIME_INFERENCE_STEPS,
        )
        .unwrap();
        let fresh_timestep = fresh.timesteps()[0];
        let expected = fresh.step(&prediction, fresh_timestep, &initial).unwrap();
        assert_eq!(replay, expected);
    }

    #[test]
    fn invalid_inputs_are_rejected_without_model_access() {
        let condition = vec![0.0_f32; VIBEVOICE_REALTIME_CONDITION_WIDTH];
        let noise = vec![0.0_f32; VIBEVOICE_REALTIME_LATENT_WIDTH];
        assert!(validate_inputs(&condition[..895], &condition, &noise, 1.0).is_err());
        assert!(validate_inputs(&condition, &condition, &noise[..63], 1.0).is_err());
        assert!(validate_inputs(&condition, &condition, &noise, f32::NAN).is_err());
        let mut nonfinite = condition.clone();
        nonfinite[0] = f32::INFINITY;
        assert!(validate_inputs(&nonfinite, &condition, &noise, 1.0).is_err());
    }

    #[test]
    fn metal_head_keeps_learned_ops_on_selected_backend() {
        assert!(ensure_realtime_head_backend(BackendKind::Metal).is_ok());
    }

    #[test]
    fn uncovered_head_backend_is_rejected_without_fallback() {
        let error = ensure_realtime_head_backend(BackendKind::Cuda).unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("no CPU fallback"));
    }
}
