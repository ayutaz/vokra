//! Policy resolution (T04) plus the two shipped presets.
//!
//! [`resolve`] maps a tensor name to a [`QuantScheme`] using the following
//! priority, with fall-through to `policy.default`:
//!
//! 1. **Exact** matches beat everything else.
//! 2. Then **Prefix**, then **Suffix**, then **Glob**.
//! 3. Ties within the same priority tier are broken by rule order (earlier
//!    wins), so callers can layer more-specific patterns first.
//!
//! Resolution never returns `Result` — every tensor gets *a* scheme.
//! Applicability (whether the backend supports the resolved scheme's
//! activation dtype, whether the op tolerates it, etc.) is a separate
//! validate-time concern (T07 runtime activation gate + T09 op registry).

use crate::quant::policy::{LayerPattern, QuantPolicy};
use crate::quant::scheme::QuantScheme;

/// Resolve `tensor_name` to a scheme using `policy`'s ordered rule list.
///
/// See the module doc for the priority rules. Falls through to
/// `policy.default_scheme()` when nothing matches.
pub fn resolve(policy: &QuantPolicy, tensor_name: &str) -> QuantScheme {
    let mut best: Option<(u8, usize, QuantScheme)> = None;
    for (idx, rule) in policy.rules().iter().enumerate() {
        if !rule.pattern.matches(tensor_name) {
            continue;
        }
        let prio = rule.pattern.priority();
        let candidate = (prio, idx, rule.scheme);
        match &best {
            None => best = Some(candidate),
            Some((best_prio, best_idx, _)) => {
                // Lower priority number == more specific. On ties, earlier
                // rule index wins (first-match-wins within a tier).
                if prio < *best_prio || (prio == *best_prio && idx < *best_idx) {
                    best = Some(candidate);
                }
            }
        }
    }
    best.map(|(_, _, s)| s).unwrap_or(policy.default_scheme())
}

// -- Presets ---------------------------------------------------------------

/// All-fp16 preset — the vocoder-safe default.
///
/// Vocos / BigVGAN / HiFi-GAN all require FP16 minimum (registry
/// `DowngradePolicy::Forbidden` / `HifiganOptIn`, FR-OP-10/11/12), and
/// current Vokra kernels are FP32 with FP16 activation as metadata-only.
/// Rules are empty — the default scheme flows through unconditionally.
pub fn default_vocoder_safe() -> QuantPolicy {
    QuantPolicy::new(QuantScheme::Fp16)
}

/// Whisper-oriented preset — Q4_K weight-only quantization with the two
/// classic exceptions promoted back to FP32:
///
/// - `Suffix(".bias")` → biases stay FP32 (rank-1, K-quant inapplicable).
/// - `Suffix(".weight_norm")` → weight-norm scalars stay FP32.
///
/// Ordering is authoritative for the ties handled by [`resolve`]: both
/// exceptions share the `Suffix` priority tier, and the order they're
/// declared is the order they're tried.
pub fn whisper_q4_k() -> QuantPolicy {
    QuantPolicy::new(QuantScheme::W4A16Q4K)
        .with_rule(LayerPattern::Suffix(".bias".to_owned()), QuantScheme::Fp32)
        .with_rule(
            LayerPattern::Suffix(".weight_norm".to_owned()),
            QuantScheme::Fp32,
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::policy::{CalibrationRef, LayerPattern};

    #[test]
    fn resolve_falls_through_to_default_when_no_rules() {
        let p = QuantPolicy::new(QuantScheme::W4A16Q5K);
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::W4A16Q5K
        );
    }

    #[test]
    fn resolve_exact_beats_prefix_beats_suffix_beats_glob() {
        // All four rules match "encoder.blocks.0.mlp.0.weight" — Exact wins.
        let p = QuantPolicy::new(QuantScheme::Fp32)
            .with_rule(
                LayerPattern::Glob("encoder.*.weight".to_owned()),
                QuantScheme::W4A16Q6K,
            )
            .with_rule(
                LayerPattern::Suffix(".weight".to_owned()),
                QuantScheme::W4A16Q5K,
            )
            .with_rule(
                LayerPattern::Prefix("encoder.".to_owned()),
                QuantScheme::Fp16,
            )
            .with_rule(
                LayerPattern::Exact("encoder.blocks.0.mlp.0.weight".to_owned()),
                QuantScheme::W4A16Q4K,
            );
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::W4A16Q4K
        );

        // Drop Exact → Prefix wins over Suffix and Glob.
        let p = QuantPolicy::new(QuantScheme::Fp32)
            .with_rule(
                LayerPattern::Glob("encoder.*.weight".to_owned()),
                QuantScheme::W4A16Q6K,
            )
            .with_rule(
                LayerPattern::Suffix(".weight".to_owned()),
                QuantScheme::W4A16Q5K,
            )
            .with_rule(
                LayerPattern::Prefix("encoder.".to_owned()),
                QuantScheme::Fp16,
            );
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::Fp16
        );

        // Drop Prefix → Suffix wins over Glob.
        let p = QuantPolicy::new(QuantScheme::Fp32)
            .with_rule(
                LayerPattern::Glob("encoder.*.weight".to_owned()),
                QuantScheme::W4A16Q6K,
            )
            .with_rule(
                LayerPattern::Suffix(".weight".to_owned()),
                QuantScheme::W4A16Q5K,
            );
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::W4A16Q5K
        );

        // Just Glob → Glob wins.
        let p = QuantPolicy::new(QuantScheme::Fp32).with_rule(
            LayerPattern::Glob("encoder.*.weight".to_owned()),
            QuantScheme::W4A16Q6K,
        );
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::W4A16Q6K
        );
    }

    #[test]
    fn resolve_within_tier_first_wins() {
        // Two Suffix rules both matching — first declared wins.
        let p = QuantPolicy::new(QuantScheme::Fp32)
            .with_rule(
                LayerPattern::Suffix(".weight".to_owned()),
                QuantScheme::Fp16,
            )
            .with_rule(
                LayerPattern::Suffix(".weight".to_owned()),
                QuantScheme::W4A16Q4K,
            );
        assert_eq!(resolve(&p, "foo.weight"), QuantScheme::Fp16);
    }

    #[test]
    fn preset_default_vocoder_safe_is_all_fp16() {
        let p = default_vocoder_safe();
        assert_eq!(p.default_scheme(), QuantScheme::Fp16);
        assert!(p.rules().is_empty());
        assert_eq!(
            resolve(&p, "generator.upsample.0.weight"),
            QuantScheme::Fp16
        );
        assert_eq!(resolve(&p, "generator.upsample.0.bias"), QuantScheme::Fp16);
    }

    #[test]
    fn preset_whisper_q4_k_keeps_bias_and_weight_norm_in_fp32() {
        let p = whisper_q4_k();
        assert_eq!(p.default_scheme(), QuantScheme::W4A16Q4K);
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::W4A16Q4K
        );
        // Biases: Suffix `.bias` → Fp32.
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.bias"),
            QuantScheme::Fp32
        );
        assert_eq!(resolve(&p, "encoder.ln_post.bias"), QuantScheme::Fp32);
        // Weight-norm scalars: Suffix `.weight_norm` → Fp32.
        assert_eq!(
            resolve(&p, "decoder.blocks.3.attn.qkv.weight_norm"),
            QuantScheme::Fp32
        );
    }

    #[test]
    fn hifigan_opt_in_preserves_resolve_semantics() {
        // T10 wiring: the opt-in flag is orthogonal to resolve — it only
        // matters at validate time. Confirm the scheme returned is unaffected.
        let base = whisper_q4_k();
        let with_opt = whisper_q4_k().with_hifigan_int8_opt_in(CalibrationRef::new("cal-1"));
        assert_eq!(
            resolve(&base, "encoder.blocks.0.mlp.0.weight"),
            resolve(&with_opt, "encoder.blocks.0.mlp.0.weight")
        );
    }
}
