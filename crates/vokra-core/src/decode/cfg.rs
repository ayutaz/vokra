//! Classifier-free guidance (CFG) combiner for flow / diffusion sampling
//! (FR-EX-10, the `flow_sampler` `cfg_mode` attribute).
//!
//! CFG steers a conditional generative model by extrapolating away from its
//! unconditional prediction: given a conditional output `cond` and an
//! unconditional output `uncond` over the same tensor, the guided output is
//!
//! ```text
//! out = uncond + scale * (cond - uncond)
//! ```
//!
//! `scale == 0` collapses to the unconditional prediction, `scale == 1` to the
//! conditional one, and `scale > 1` (the usual regime) over-emphasizes the
//! conditioning. The combiner is a pure, model-independent host function; the
//! model decides how it obtains `cond` / `uncond` (see [`CfgMode`]).
//!
//! # Scope
//!
//! This lands the combiner and mode enum ahead of the first in-tree consumer so
//! the FR-EX-10 surface is fixed; the `flow_sampler` op (CosyVoice2 / F5-TTS,
//! v1.0) wires them in later.

use crate::error::{Result, VokraError};

/// How the two guidance forwards (`cond` / `uncond`) are produced — the
/// `flow_sampler` `cfg_mode` attribute (FR-EX-10, CLAUDE.md audio dialect).
///
/// The enum only *names* the strategy; [`apply_cfg`] does the identical
/// arithmetic regardless of how the two inputs were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CfgMode {
    /// No guidance: a single conditional forward is used verbatim (`scale`
    /// ignored). The default.
    #[default]
    None,
    /// One batched forward carrying both the conditional and unconditional
    /// rows, split apart before combining (one kernel launch, 2× batch).
    SplitBatch,
    /// Two separate forwards, one conditional and one unconditional (2× launches,
    /// 1× batch each) — the memory-lean alternative to [`CfgMode::SplitBatch`].
    DualForward,
}

/// Writes the classifier-free-guidance combination of `cond` and `uncond` into
/// `out`: `out[i] = uncond[i] + scale * (cond[i] - uncond[i])`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if the three slices do not all have the same
/// length.
pub fn apply_cfg(cond: &[f32], uncond: &[f32], scale: f32, out: &mut [f32]) -> Result<()> {
    if cond.len() != uncond.len() || cond.len() != out.len() {
        return Err(VokraError::InvalidArgument(format!(
            "apply_cfg: length mismatch (cond {}, uncond {}, out {})",
            cond.len(),
            uncond.len(),
            out.len()
        )));
    }
    for (o, (&c, &u)) in out.iter_mut().zip(cond.iter().zip(uncond)) {
        *o = u + scale * (c - u);
    }
    Ok(())
}

/// In-place variant of [`apply_cfg`]: `cond` is overwritten with the guided
/// result `uncond[i] + scale * (cond[i] - uncond[i])`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `cond` and `uncond` differ in length.
pub fn apply_cfg_inplace(cond: &mut [f32], uncond: &[f32], scale: f32) -> Result<()> {
    if cond.len() != uncond.len() {
        return Err(VokraError::InvalidArgument(format!(
            "apply_cfg_inplace: length mismatch (cond {}, uncond {})",
            cond.len(),
            uncond.len()
        )));
    }
    for (c, &u) in cond.iter_mut().zip(uncond) {
        *c = u + scale * (*c - u);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_zero_is_unconditional() {
        let cond = [10.0, -3.0, 7.0];
        let uncond = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        apply_cfg(&cond, &uncond, 0.0, &mut out).unwrap();
        assert_eq!(out, uncond);
    }

    #[test]
    fn scale_one_is_conditional() {
        let cond = [10.0, -3.0, 7.0];
        let uncond = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        apply_cfg(&cond, &uncond, 1.0, &mut out).unwrap();
        assert_eq!(out, cond);
    }

    #[test]
    fn matches_hand_computed_linearity() {
        // out = uncond + scale*(cond - uncond), element by element.
        let cond = [4.0, 0.0, -2.0];
        let uncond = [1.0, 1.0, 1.0];
        let scale = 3.0;
        let mut out = [0.0; 3];
        apply_cfg(&cond, &uncond, scale, &mut out).unwrap();
        // idx0: 1 + 3*(4-1) = 10 ; idx1: 1 + 3*(0-1) = -2 ; idx2: 1 + 3*(-2-1) = -8
        assert_eq!(out, [10.0, -2.0, -8.0]);
    }

    #[test]
    fn inplace_agrees_with_out_of_place() {
        let uncond = [1.0, 2.0, 3.0, 4.0];
        let base = [4.0, 3.0, 2.0, 1.0];
        let scale = 2.5;
        let mut out = [0.0; 4];
        apply_cfg(&base, &uncond, scale, &mut out).unwrap();
        let mut cond = base;
        apply_cfg_inplace(&mut cond, &uncond, scale).unwrap();
        assert_eq!(cond, out);
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let mut out = [0.0; 3];
        assert!(matches!(
            apply_cfg(&[1.0, 2.0], &[1.0, 2.0, 3.0], 1.0, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            apply_cfg(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1.0, &mut [0.0; 2]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            apply_cfg_inplace(&mut [1.0, 2.0], &[1.0], 1.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn cfg_mode_default_is_none() {
        assert_eq!(CfgMode::default(), CfgMode::None);
    }
}
