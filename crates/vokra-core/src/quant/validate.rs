//! `validate_policy_against_model` (T09) — reject INT8 for fp16-required ops.
//!
//! Given a [`QuantPolicy`], the list of op-kind identifiers a model uses, and
//! a [`MinDtypeRegistry`], reject any policy that would drive a registered op
//! below its minimum activation dtype **before** hitting a backend
//! (FR-EX-08: no silent CPU fallback, generalized here to "no silent widen").
//!
//! # Algorithm
//!
//! For each `op_name` in `ops_in_use`:
//!
//! 1. Look the op up in the registry. Unregistered ops are unconstrained —
//!    they accept whatever scheme the policy resolves to (matches the
//!    `MinDtypeRegistry::lookup(...) -> None` contract).
//! 2. When the op is registered, walk every scheme the policy could resolve
//!    to for this op — the default plus every rule's scheme — since we don't
//!    carry per-op tensor names in [`ops_in_use`], the safe superset over the
//!    policy's scheme table is what T09 checks. If **any** of those schemes
//!    reports an [`ActivationDtype`](crate::quant::scheme::ActivationDtype)
//!    that fails [`MinDtype::is_satisfied_by`], raise
//!    [`VokraError::MinDtypeViolation`].
//! 3. The HiFi-GAN opt-in gate ([`DowngradePolicy::HifiganOptIn`], T10)
//!    suppresses the error when *both* `policy.hifigan_int8_opt_in() == true`
//!    *and* a calibration reference is attached — the calibration presence is
//!    guaranteed by construction via
//!    [`QuantPolicy::with_hifigan_int8_opt_in`], and [`Self::validate_self`]
//!    catches manufactured corruption. Vocos / BigVGAN
//!    ([`DowngradePolicy::Forbidden`]) are rejected regardless of any flag.
//!
//! Enforcement runs *pre-kernel* — no backend is invoked before this pass, so
//! a policy that resolves to an INT8 activation on a fp16-required op fails
//! deterministically at converter / session-ctor time.
//!
//! [`QuantPolicy`]: crate::quant::QuantPolicy
//! [`MinDtypeRegistry`]: crate::quant::MinDtypeRegistry
//! [`MinDtype::is_satisfied_by`]: crate::quant::MinDtype::is_satisfied_by
//! [`DowngradePolicy::Forbidden`]: crate::quant::DowngradePolicy::Forbidden
//! [`DowngradePolicy::HifiganOptIn`]: crate::quant::DowngradePolicy::HifiganOptIn
//! [`QuantPolicy::with_hifigan_int8_opt_in`]: crate::quant::QuantPolicy::with_hifigan_int8_opt_in
//! [`VokraError::MinDtypeViolation`]: crate::VokraError::MinDtypeViolation

use crate::error::{Result, VokraError};
use crate::quant::policy::QuantPolicy;
use crate::quant::registry::{DowngradePolicy, MinDtype, MinDtypeRegistry};

/// Canonical string form of a [`MinDtype`], used in the audit error message.
///
/// Kept local — [`MinDtype`] is `#[non_exhaustive]` and the alias vocabulary is
/// stable for M2-08. If a new tier is added in a follow-up, extend the match
/// arm and update the doc.
fn min_dtype_str(m: MinDtype) -> &'static str {
    match m {
        MinDtype::Fp16 => "fp16",
        MinDtype::Fp32 => "fp32",
    }
}

/// Validate a [`QuantPolicy`] against the op-kind identifiers a model uses.
///
/// See the module doc for the full algorithm. Returns `Ok(())` when every
/// registered op in `ops_in_use` can be satisfied by *every* scheme the policy
/// could resolve to (default + rules), or the HiFi-GAN opt-in gate suppresses
/// the mismatch.
///
/// # Errors
///
/// [`VokraError::MinDtypeViolation`] when a registered op's activation dtype
/// minimum is violated and no downgrade path applies. The error carries the
/// FR-* audit reference from the registry so the operator knows which
/// requirement fired (e.g. `FR-OP-12` for Vocos).
pub fn validate_policy_against_model(
    policy: &QuantPolicy,
    ops_in_use: &[&str],
    registry: &MinDtypeRegistry,
) -> Result<()> {
    // Catch a manufactured `opt_in=true` + `calibration=None` state before we
    // rely on the opt-in gate below. Construction-time invariants normally
    // make this unreachable, but validate_self() is the belt-and-braces check.
    policy.validate_self()?;

    for op_name in ops_in_use {
        let Some(entry) = registry.lookup(op_name) else {
            // Unregistered ops are unconstrained — the registry docs the
            // `None` contract explicitly.
            continue;
        };

        // Walk every scheme the policy could resolve to for tensors of this
        // op. Since [`ops_in_use`] carries op-kind identifiers rather than
        // tensor names, the safe superset is the union of `policy.default`
        // and every rule's scheme.
        let candidate_schemes =
            std::iter::once(policy.default_scheme()).chain(policy.rules().iter().map(|r| r.scheme));

        for scheme in candidate_schemes {
            if entry
                .min_activation
                .is_satisfied_by(scheme.activation_dtype())
            {
                continue;
            }

            // Below the minimum. Only the HiFi-GAN opt-in path can suppress
            // this — Forbidden entries reject unconditionally.
            let suppressed = matches!(entry.downgrade, DowngradePolicy::HifiganOptIn)
                && policy.hifigan_int8_opt_in()
                && policy.hifigan_int8_calibration().is_some();

            if suppressed {
                // The T12 eval-verify gate (NFR-QL-02 5% MEL-loss check) is a
                // separate deployment-time concern wired in `vokra-cli` /
                // session ctor; T09's job is to accept the policy shape here.
                continue;
            }

            return Err(VokraError::MinDtypeViolation {
                op: (*op_name).to_owned(),
                requested_scheme: scheme.as_str().to_owned(),
                min_required: min_dtype_str(entry.min_activation).to_owned(),
                fr_ref: entry.fr_ref.to_owned(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::policy::{CalibrationRef, LayerPattern};
    use crate::quant::registry::{
        BIGVGAN_GENERATOR_OP, HIFIGAN_GENERATOR_OP, SNAKE_ACTIVATION_OP, VOCOS_HEAD_OP,
    };
    use crate::quant::scheme::QuantScheme;

    // ----- Forbidden downgrade (Vocos / BigVGAN) -----------------------------

    #[test]
    fn vocos_head_rejects_w8a8_default() {
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
        let reg = MinDtypeRegistry::builtin();
        let err = validate_policy_against_model(&policy, &[VOCOS_HEAD_OP], &reg)
            .expect_err("vocos_head + w8a8 must reject");
        match err {
            VokraError::MinDtypeViolation {
                op,
                requested_scheme,
                min_required,
                fr_ref,
            } => {
                assert_eq!(op, "vocos_head");
                assert_eq!(requested_scheme, "w8a8");
                assert_eq!(min_required, "fp16");
                assert_eq!(fr_ref, "FR-OP-12");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn vocos_head_rejects_even_with_hifigan_opt_in() {
        // Forbidden downgrade — the HiFi-GAN opt-in flag is orthogonal.
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8)
            .with_hifigan_int8_opt_in(CalibrationRef::new("cal-vocos"));
        let reg = MinDtypeRegistry::builtin();
        assert!(matches!(
            validate_policy_against_model(&policy, &[VOCOS_HEAD_OP], &reg),
            Err(VokraError::MinDtypeViolation { .. })
        ));
    }

    #[test]
    fn bigvgan_rejects_w8a8_rule() {
        // W8A8 hides in a rule rather than the default — the walk still
        // catches it.
        let policy = QuantPolicy::new(QuantScheme::Fp16).with_rule(
            LayerPattern::Prefix("generator.".to_owned()),
            QuantScheme::W8A8Int8,
        );
        let reg = MinDtypeRegistry::builtin();
        let err = validate_policy_against_model(&policy, &[BIGVGAN_GENERATOR_OP], &reg)
            .expect_err("bigvgan + w8a8 rule must reject");
        match err {
            VokraError::MinDtypeViolation {
                op,
                min_required,
                fr_ref,
                ..
            } => {
                assert_eq!(op, "bigvgan_generator");
                assert_eq!(min_required, "fp16");
                assert_eq!(fr_ref, "FR-OP-11");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ----- HiFi-GAN opt-in path ---------------------------------------------

    #[test]
    fn hifigan_rejects_w8a8_without_opt_in() {
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
        let reg = MinDtypeRegistry::builtin();
        let err = validate_policy_against_model(&policy, &[HIFIGAN_GENERATOR_OP], &reg)
            .expect_err("hifigan + w8a8 + no opt-in must reject");
        match err {
            VokraError::MinDtypeViolation { op, fr_ref, .. } => {
                assert_eq!(op, "hifigan_generator");
                assert_eq!(fr_ref, "FR-OP-10");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn hifigan_accepts_w8a8_with_opt_in_and_calibration() {
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8)
            .with_hifigan_int8_opt_in(CalibrationRef::new("hifigan-cal-v1"));
        let reg = MinDtypeRegistry::builtin();
        validate_policy_against_model(&policy, &[HIFIGAN_GENERATOR_OP], &reg)
            .expect("hifigan + w8a8 + opt-in + calibration must pass validate");
    }

    // ----- Snake activation (FP32 minimum) ----------------------------------

    #[test]
    fn snake_activation_rejects_fp16_default() {
        // Snake's minimum is Fp32, so even Fp16 is below the bar.
        let policy = QuantPolicy::new(QuantScheme::Fp16);
        let reg = MinDtypeRegistry::builtin();
        let err = validate_policy_against_model(&policy, &[SNAKE_ACTIVATION_OP], &reg)
            .expect_err("snake_activation + fp16 must reject");
        match err {
            VokraError::MinDtypeViolation {
                op,
                min_required,
                fr_ref,
                ..
            } => {
                assert_eq!(op, "snake_activation");
                assert_eq!(min_required, "fp32");
                assert_eq!(fr_ref, "FR-OP-13");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn snake_activation_accepts_fp32() {
        let policy = QuantPolicy::new(QuantScheme::Fp32);
        let reg = MinDtypeRegistry::builtin();
        validate_policy_against_model(&policy, &[SNAKE_ACTIVATION_OP], &reg)
            .expect("snake_activation + fp32 must pass");
    }

    #[test]
    fn snake_activation_rejects_w8a8_even_with_hifigan_opt_in() {
        // Snake's downgrade is Forbidden — HiFi-GAN opt-in cannot suppress it.
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8)
            .with_hifigan_int8_opt_in(CalibrationRef::new("cal"));
        let reg = MinDtypeRegistry::builtin();
        assert!(matches!(
            validate_policy_against_model(&policy, &[SNAKE_ACTIVATION_OP], &reg),
            Err(VokraError::MinDtypeViolation { .. })
        ));
    }

    // ----- Unregistered ops are unconstrained -------------------------------

    #[test]
    fn unregistered_op_accepts_any_scheme() {
        // "matmul" is not in the built-in registry — anything the policy
        // resolves to is accepted (the registry's `None` contract).
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
        let reg = MinDtypeRegistry::builtin();
        validate_policy_against_model(&policy, &["matmul", "add", "softmax"], &reg)
            .expect("unregistered ops accept any scheme");
    }

    #[test]
    fn empty_ops_in_use_accepts_any_policy() {
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
        let reg = MinDtypeRegistry::builtin();
        validate_policy_against_model(&policy, &[], &reg).expect("no ops → nothing to constrain");
    }

    // ----- Valid combinations pass ------------------------------------------

    #[test]
    fn vocos_head_accepts_fp16() {
        let policy = QuantPolicy::new(QuantScheme::Fp16);
        let reg = MinDtypeRegistry::builtin();
        validate_policy_against_model(&policy, &[VOCOS_HEAD_OP], &reg)
            .expect("vocos_head + fp16 must pass");
    }

    #[test]
    fn vocos_head_accepts_fp32() {
        let policy = QuantPolicy::new(QuantScheme::Fp32);
        let reg = MinDtypeRegistry::builtin();
        validate_policy_against_model(&policy, &[VOCOS_HEAD_OP], &reg)
            .expect("vocos_head + fp32 must pass");
    }

    #[test]
    fn vocos_head_accepts_w4a16_variants() {
        // W4A16 tiers report F16 activation → satisfies Fp16 minimum.
        for scheme in [
            QuantScheme::W4A16Q4K,
            QuantScheme::W4A16Q5K,
            QuantScheme::W4A16Q6K,
        ] {
            let policy = QuantPolicy::new(scheme);
            let reg = MinDtypeRegistry::builtin();
            validate_policy_against_model(&policy, &[VOCOS_HEAD_OP], &reg).unwrap_or_else(|e| {
                panic!("vocos_head + {} must pass, got {e:?}", scheme.as_str())
            });
        }
    }

    // ----- Multi-op walk ----------------------------------------------------

    #[test]
    fn multiple_registered_ops_all_checked() {
        // Policy is fp16 default but has a rule that would put a tensor in
        // W8A8 — vocos_head must reject, hifigan without opt-in must reject.
        let policy = QuantPolicy::new(QuantScheme::Fp16).with_rule(
            LayerPattern::Prefix("gen.".to_owned()),
            QuantScheme::W8A8Int8,
        );
        let reg = MinDtypeRegistry::builtin();
        let err =
            validate_policy_against_model(&policy, &[VOCOS_HEAD_OP, HIFIGAN_GENERATOR_OP], &reg)
                .expect_err("W8A8 rule triggers a violation on the first registered op");
        assert!(matches!(err, VokraError::MinDtypeViolation { .. }));
    }
}
