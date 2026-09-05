//! OUVE-SDE primitives used by score-based speech enhancement.
//!
//! This module is deliberately a small, tensor-layout agnostic numerical
//! seam.  SGMSE's complex STFT tensor is represented as a flat `f32` slice;
//! the model owns batching and channel layout while these functions own the
//! scalar SDE coefficients and element-wise updates.  The equations mirror
//! the pinned `OUVESDE`, `ReverseDiffusionPredictor`, and
//! `AnnealedLangevinDynamics` implementations in `sp-uhh/sgmse`.
//! Source pin: <https://github.com/sp-uhh/sgmse/tree/1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e>.
//!
//! No random numbers are generated here.  Callers provide the noise buffer,
//! which keeps seeded reference comparisons independent of the runtime RNG.

use vokra_core::{Result, VokraError};

/// Parameters of the Ornstein-Uhlenbeck variance-exploding SDE.
///
/// The upstream process is
/// `dx = theta * (y - x) dt + sigma_min * (sigma_max / sigma_min)^t
///       * sqrt(2 * log(sigma_max / sigma_min)) dW`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OuvEConfig {
    /// OU stiffness (`theta`).
    pub theta: f32,
    /// Diffusion at `t = 0`.
    pub sigma_min: f32,
    /// Diffusion envelope at `t = 1` before the OU normalization factor.
    pub sigma_max: f32,
}

impl OuvEConfig {
    /// Constructs and validates an OUVE configuration.
    pub fn new(theta: f32, sigma_min: f32, sigma_max: f32) -> Result<Self> {
        let config = Self {
            theta,
            sigma_min,
            sigma_max,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<()> {
        if !self.theta.is_finite()
            || !self.sigma_min.is_finite()
            || !self.sigma_max.is_finite()
            || self.theta < 0.0
            || self.sigma_min <= 0.0
            || self.sigma_max <= self.sigma_min
        {
            return Err(VokraError::InvalidArgument(
                "OUVE config requires finite theta >= 0 and 0 < sigma_min < sigma_max".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_time(self, t: f32) -> Result<()> {
        self.validate()?;
        if !t.is_finite() || !(0.0..=1.0).contains(&t) {
            return Err(VokraError::InvalidArgument(
                "OUVE time must be finite and in [0, 1]".to_owned(),
            ));
        }
        Ok(())
    }

    fn log_ratio(self) -> Result<f32> {
        let ratio = self.sigma_max / self.sigma_min;
        if !ratio.is_finite() || ratio <= 1.0 {
            return Err(VokraError::InvalidArgument(
                "OUVE sigma_max/sigma_min ratio is not finite and greater than one".to_owned(),
            ));
        }
        let log_ratio = ratio.ln();
        if !log_ratio.is_finite() || log_ratio <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "OUVE log(sigma_max/sigma_min) is not finite and positive".to_owned(),
            ));
        }
        Ok(log_ratio)
    }

    /// The OUVE diffusion coefficient `g(t)`.
    pub fn diffusion(self, t: f32) -> Result<f32> {
        self.validate_time(t)?;
        let log_ratio = self.log_ratio()?;
        let exponent = log_ratio * t;
        let sqrt_arg = 2.0 * log_ratio;
        if !exponent.is_finite() || !sqrt_arg.is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE diffusion intermediate is not finite".to_owned(),
            ));
        }
        let diffusion = self.sigma_min * exponent.exp() * sqrt_arg.sqrt();
        if !diffusion.is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE diffusion coefficient is not finite".to_owned(),
            ));
        }
        Ok(diffusion)
    }

    /// The closed-form marginal standard deviation `std(t)`.
    pub fn std(self, t: f32) -> Result<f32> {
        self.validate_time(t)?;
        let log_ratio = self.log_ratio()?;
        let denominator = self.theta + log_ratio;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "OUVE marginal variance denominator is not finite and positive".to_owned(),
            ));
        }
        let decay = -2.0 * self.theta * t;
        let growth = 2.0 * denominator * t;
        if !decay.is_finite() || !growth.is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE marginal variance exponent is not finite".to_owned(),
            ));
        }
        let numerator =
            self.sigma_min * self.sigma_min * decay.exp() * (growth.exp() - 1.0) * log_ratio;
        let variance = numerator / denominator;
        if !variance.is_finite() || variance < 0.0 {
            return Err(VokraError::InvalidArgument(
                "OUVE marginal variance is not finite".to_owned(),
            ));
        }
        Ok(variance.sqrt())
    }

    /// Conditional OU mean `alpha(t) * x0 + (1 - alpha(t)) * y`.
    pub fn mean(self, x0: f32, y: f32, t: f32) -> Result<f32> {
        self.validate_time(t)?;
        if !x0.is_finite() || !y.is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE mean inputs must be finite".to_owned(),
            ));
        }
        let exponent = -self.theta * t;
        if !exponent.is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE mean exponent is not finite".to_owned(),
            ));
        }
        let alpha = exponent.exp();
        let value = alpha * x0 + (1.0 - alpha) * y;
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE mean output is not finite".to_owned(),
            ));
        }
        Ok(value)
    }
}

fn validate_buffers(x: &[f32], y: &[f32], score: &[f32], noise: &[f32], out: &[f32]) -> Result<()> {
    let expected = x.len();
    if y.len() != expected
        || score.len() != expected
        || noise.len() != expected
        || out.len() != expected
    {
        return Err(VokraError::InvalidArgument(format!(
            "OUVE buffers must have equal lengths: x={}, y={}, score={}, noise={}, out={}",
            x.len(),
            y.len(),
            score.len(),
            noise.len(),
            out.len()
        )));
    }
    if x.iter()
        .chain(y)
        .chain(score)
        .chain(noise)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "OUVE buffers must contain finite values".to_owned(),
        ));
    }
    Ok(())
}

fn validate_unary_buffers(x: &[f32], score: &[f32], noise: &[f32], out: &[f32]) -> Result<()> {
    let expected = x.len();
    if score.len() != expected || noise.len() != expected || out.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "OUVE buffers must have equal lengths: x={}, score={}, noise={}, out={}",
            x.len(),
            score.len(),
            noise.len(),
            out.len()
        )));
    }
    if x.iter()
        .chain(score)
        .chain(noise)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "OUVE buffers must contain finite values".to_owned(),
        ));
    }
    Ok(())
}

/// One reverse-diffusion predictor update.
///
/// `step` is positive even though the process runs from `t=1` toward `t=0`.
/// This matches the upstream discretization convention: the forward drift is
/// multiplied by `step`, then subtracted from the current state.  `out_mean`
/// is the deterministic predictor state; `out` adds caller-provided noise.
// This public buffer-oriented seam intentionally exposes all numerical inputs
// and output buffers; keep the lint allowance local rather than changing the
// established API shape.
#[allow(clippy::too_many_arguments)]
pub fn reverse_diffusion_step(
    config: OuvEConfig,
    x: &[f32],
    y: &[f32],
    score: &[f32],
    t: f32,
    step: f32,
    noise: &[f32],
    probability_flow: bool,
    out: &mut [f32],
    out_mean: &mut [f32],
) -> Result<()> {
    config.validate_time(t)?;
    if !step.is_finite() || step <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "OUVE reverse step must be finite and positive".to_owned(),
        ));
    }
    if out_mean.len() != x.len() {
        return Err(VokraError::InvalidArgument(
            "OUVE deterministic output length does not match input".to_owned(),
        ));
    }
    validate_buffers(x, y, score, noise, out)?;
    let diffusion = config.diffusion(t)?;
    let score_scale = diffusion * diffusion * if probability_flow { 0.5 } else { 1.0 };
    let noise_scale = if probability_flow {
        0.0
    } else {
        diffusion * step.sqrt()
    };
    if !score_scale.is_finite() || !noise_scale.is_finite() {
        return Err(VokraError::InvalidArgument(
            "OUVE reverse score/noise scale is not finite".to_owned(),
        ));
    }
    for index in 0..x.len() {
        let forward_drift = config.theta * (y[index] - x[index]);
        let reverse_increment = (forward_drift - score_scale * score[index]) * step;
        out_mean[index] = x[index] - reverse_increment;
        out[index] = out_mean[index] + noise_scale * noise[index];
        if !out_mean[index].is_finite() || !out[index].is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE reverse update produced a non-finite value".to_owned(),
            ));
        }
    }
    Ok(())
}

/// One annealed Langevin corrector update.
///
/// The source implementation uses `2 * (snr * std(t))²` as the step size and
/// adds `sqrt(2 * step_size) * noise` after the deterministic score update.
// This public buffer-oriented seam intentionally exposes all numerical inputs
// and output buffers; keep the lint allowance local rather than changing the
// established API shape.
#[allow(clippy::too_many_arguments)]
pub fn annealed_langevin_step(
    config: OuvEConfig,
    x: &[f32],
    score: &[f32],
    t: f32,
    snr: f32,
    noise: &[f32],
    out: &mut [f32],
    out_mean: &mut [f32],
) -> Result<()> {
    config.validate_time(t)?;
    if !snr.is_finite() || snr < 0.0 {
        return Err(VokraError::InvalidArgument(
            "OUVE annealed Langevin SNR must be finite and non-negative".to_owned(),
        ));
    }
    if out_mean.len() != x.len() {
        return Err(VokraError::InvalidArgument(
            "OUVE corrector deterministic output length does not match input".to_owned(),
        ));
    }
    validate_unary_buffers(x, score, noise, out)?;
    let std = config.std(t)?;
    let scaled_std = snr * std;
    if !scaled_std.is_finite() {
        return Err(VokraError::InvalidArgument(
            "OUVE Langevin score coefficient is not finite".to_owned(),
        ));
    }
    let step_size = 2.0 * scaled_std.powi(2);
    let noise_variance = 2.0 * step_size;
    if !noise_variance.is_finite() || noise_variance < 0.0 {
        return Err(VokraError::InvalidArgument(
            "OUVE Langevin noise coefficient is not finite".to_owned(),
        ));
    }
    let noise_scale = noise_variance.sqrt();
    if !step_size.is_finite() || !noise_scale.is_finite() {
        return Err(VokraError::InvalidArgument(
            "OUVE Langevin step is not finite".to_owned(),
        ));
    }
    for index in 0..x.len() {
        out_mean[index] = x[index] + step_size * score[index];
        out[index] = out_mean[index] + noise_scale * noise[index];
        if !out_mean[index].is_finite() || !out[index].is_finite() {
            return Err(VokraError::InvalidArgument(
                "OUVE Langevin update produced a non-finite value".to_owned(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: OuvEConfig = OuvEConfig {
        theta: 1.5,
        sigma_min: 0.05,
        sigma_max: 0.5,
    };

    #[test]
    fn pinned_ouve_coefficients_are_finite_and_monotonic() {
        let start = CONFIG.diffusion(0.0).unwrap();
        let end = CONFIG.diffusion(1.0).unwrap();
        assert!(start.is_finite() && end.is_finite() && end > start);
        assert_eq!(CONFIG.std(0.0).unwrap(), 0.0);
        assert!(CONFIG.std(1.0).unwrap() > CONFIG.std(0.0).unwrap());
        assert_eq!(CONFIG.mean(2.0, -1.0, 0.0).unwrap(), 2.0);
    }

    #[test]
    fn reverse_step_matches_independent_scalar_oracle() {
        let x = [0.25, -0.5];
        let y = [0.0, 0.5];
        let score = [0.2, -0.4];
        let noise = [0.0, 0.0];
        let mut out = [f32::NAN; 2];
        let mut mean = [f32::NAN; 2];
        reverse_diffusion_step(
            CONFIG, &x, &y, &score, 0.4, 0.1, &noise, false, &mut out, &mut mean,
        )
        .unwrap();
        let g = CONFIG.diffusion(0.4).unwrap();
        for index in 0..2 {
            let drift = CONFIG.theta * (y[index] - x[index]);
            let expected = x[index] - (drift - g * g * score[index]) * 0.1;
            assert!((mean[index] - expected).abs() < 1e-6);
            assert_eq!(out[index], mean[index]);
        }
    }

    #[test]
    fn annealed_langevin_matches_independent_scalar_oracle() {
        let x = [0.25, -0.5];
        let score = [0.2, -0.4];
        let noise = [0.0, 0.0];
        let mut out = [f32::NAN; 2];
        let mut mean = [f32::NAN; 2];
        annealed_langevin_step(CONFIG, &x, &score, 0.4, 0.5, &noise, &mut out, &mut mean).unwrap();
        let step = 2.0 * (0.5 * CONFIG.std(0.4).unwrap()).powi(2);
        for index in 0..2 {
            assert!((mean[index] - (x[index] + step * score[index])).abs() < 1e-6);
            assert_eq!(out[index], mean[index]);
        }
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(OuvEConfig::new(1.5, 0.5, 0.05).is_err());
        assert!(CONFIG.diffusion(-0.1).is_err());
        let ratio_overflow = OuvEConfig::new(0.0, f32::MIN_POSITIVE, f32::MAX).unwrap();
        assert!(ratio_overflow.diffusion(1.0).is_err());
        let large_theta = OuvEConfig::new(f32::MAX, 0.05, 0.5).unwrap();
        assert!(large_theta.std(1.0).is_err());
        // exp(-theta * t) underflows to zero here; the pinned OU mean has the
        // finite limiting value y, while non-finite outputs remain rejected.
        assert_eq!(
            large_theta.mean(f32::MAX, -f32::MAX, 1.0).unwrap(),
            -f32::MAX
        );
        let mut out = [0.0; 1];
        let mut mean = [0.0; 1];
        assert!(
            reverse_diffusion_step(
                CONFIG,
                &[0.0],
                &[0.0],
                &[0.0],
                0.5,
                0.0,
                &[0.0],
                false,
                &mut out,
                &mut mean
            )
            .is_err()
        );
        assert!(
            reverse_diffusion_step(
                CONFIG,
                &[f32::MAX],
                &[-f32::MAX],
                &[f32::MAX],
                0.5,
                0.1,
                &[f32::MAX],
                false,
                &mut out,
                &mut mean,
            )
            .is_err()
        );
        assert!(
            annealed_langevin_step(
                CONFIG,
                &[0.0],
                &[0.0],
                0.5,
                f32::NAN,
                &[0.0],
                &mut out,
                &mut mean
            )
            .is_err()
        );
        assert!(
            annealed_langevin_step(
                CONFIG,
                &[0.0],
                &[f32::MAX],
                1.0,
                f32::MAX,
                &[f32::MAX],
                &mut out,
                &mut mean,
            )
            .is_err()
        );
    }
}
