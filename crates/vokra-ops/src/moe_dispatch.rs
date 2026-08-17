//! Mixture-of-Experts (MoE) token→expert dispatch primitive
//! (SoTA plan Wave C blocker for `qwen3-omni-30b-a3b-moe` and
//! `zonos2-8b-moe`; see
//! `docs/tickets/coverage-audit-2026-08-03/IMPL-PLAN.md` §2.3).
//!
//! This module hosts the **routing plan** step alone — the top-k expert
//! selection with a capacity-factor cap. It deliberately does not touch
//! per-expert weights: those consume the plan through
//! [`crate::moe_expert_gemm::moe_expert_gemm`]. Splitting the two keeps
//! the routing decision independently unit-testable and lets a future
//! GPU / SIMD backend replace the per-expert GEMM without redoing the
//! routing math.
//!
//! # Runtime function, not an `OpKind` variant
//!
//! MoE dispatch has the same posture as [`crate::flow_sampler::flow_sample`],
//! [`crate::mimi_rvq::mimi_rvq_decode`], and
//! [`crate::openwakeword::openwakeword_classifier_forward`] — see ADR
//! M3-06 §D-b and FR-EX-10. The heterogeneous inputs (router logits,
//! per-expert weight bundles, dispatch plans) do not fit the `OpValue`
//! dispatch surface without hiding shape errors behind a runtime tag, so
//! this crate exposes it as a plain function.
//!
//! # FR-EX-08 loud-fail contract
//!
//! Every shape / attribute mismatch is a hard [`vokra_core::VokraError::InvalidArgument`]
//! naming the offending dimension. No silent zero-padding, no silent
//! `top_k` clamping, no silent capacity-drop that is not reported in
//! [`MoeDispatchPlan::drop_rate`].
//!
//! # References
//!
//! - Fedus, Zoph, Shazeer 2022 "Switch Transformers" — the capacity
//!   factor formulation this module implements.
//! - Shazeer et al. 2017 "Outrageously Large Neural Networks" — the
//!   original top-k gating + noise formulation (noise is intentionally
//!   omitted here, per the Switch / Mixtral / DeepSeek-MoE lineage that
//!   the target Vokra models use).
//! - Jiang et al. 2024 "Mixtral of Experts" — the `renormalize_gates`
//!   default (Mixtral re-normalizes the top-k gate weights to sum to 1
//!   after softmax; this is the released Mixtral behaviour and the
//!   default here for backward compatibility with permissive-license
//!   MoE checkpoints).

use vokra_core::{Result, VokraError};

/// Routing attributes for [`moe_dispatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct MoeDispatchAttrs {
    /// Number of experts in the layer (≥ 1). Must equal the last
    /// dimension of the router logit matrix.
    pub num_experts: usize,
    /// Number of experts each token dispatches to (`1 ≤ top_k ≤ num_experts`).
    pub top_k: usize,
    /// Capacity factor `f` (Switch Transformers §2.2). The per-expert
    /// capacity is `ceil((num_tokens * top_k / num_experts) * f)`.
    /// `f = 1.0` gives the balanced-load capacity; `f > 1.0` gives
    /// slack; `f < 1.0` is aggressive dropping. Must be strictly
    /// positive and finite.
    pub capacity_factor: f32,
    /// If `true`, over-capacity assignments are dropped and reflected in
    /// [`MoeDispatchPlan::drop_rate`] (Switch / Mixtral behaviour).
    /// If `false`, capacity is soft (all assignments kept, the caller
    /// is expected to handle overflow) — this exists for evaluation
    /// harnesses; production inference always sets this `true`.
    pub drop_tokens: bool,
    /// If `true`, the top-k gate weights per token are re-normalized so
    /// they sum to 1 after the softmax (Mixtral default). If `false`,
    /// the raw softmax probabilities are kept (Switch's k=1 case is
    /// unaffected either way because a single value renormalises to
    /// itself).
    pub renormalize_gates: bool,
}

impl MoeDispatchAttrs {
    /// Validates the attributes loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any out-of-range value.
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(VokraError::InvalidArgument(
                "moe_dispatch: num_experts must be ≥ 1 (got 0)".to_owned(),
            ));
        }
        if self.top_k == 0 {
            return Err(VokraError::InvalidArgument(
                "moe_dispatch: top_k must be ≥ 1 (got 0)".to_owned(),
            ));
        }
        if self.top_k > self.num_experts {
            return Err(VokraError::InvalidArgument(format!(
                "moe_dispatch: top_k ({}) must be ≤ num_experts ({})",
                self.top_k, self.num_experts
            )));
        }
        if !self.capacity_factor.is_finite() || self.capacity_factor <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "moe_dispatch: capacity_factor must be finite and > 0 (got {})",
                self.capacity_factor
            )));
        }
        Ok(())
    }

    /// Per-expert capacity for `num_tokens` under this attribute bundle
    /// (Switch Transformers §2.2 formula, `ceil((N * k / E) * f)`).
    ///
    /// The minimum capacity is 1 (a strictly-positive capacity factor
    /// must never round the load to 0 experts, else a valid single-token
    /// dispatch would be silently dropped — FR-EX-08).
    pub fn capacity_for(&self, num_tokens: usize) -> usize {
        if num_tokens == 0 {
            return 0;
        }
        let ideal = (num_tokens as f64) * (self.top_k as f64) / (self.num_experts as f64);
        let scaled = ideal * (self.capacity_factor as f64);
        (scaled.ceil() as usize).max(1)
    }
}

/// One `(token, gate_weight)` pair assigned to an expert. The order in
/// which pairs appear in [`MoeDispatchPlan::expert_assignments`] is the
/// **dispatch order** that [`crate::moe_expert_gemm::moe_expert_gemm`]
/// uses to build the per-expert input packing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeAssignment {
    /// Index of the token in the original `[num_tokens, hidden]` input.
    pub token_idx: usize,
    /// The gate weight the expert should multiply its GEMM output by
    /// before combining back into the token's output vector.
    pub gate_weight: f32,
}

/// The routing plan produced by [`moe_dispatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct MoeDispatchPlan {
    /// Number of tokens the plan was built for (mirrors the input).
    pub num_tokens: usize,
    /// Number of experts (mirrors the attributes).
    pub num_experts: usize,
    /// Top-k value the plan was built with (mirrors the attributes).
    pub top_k: usize,
    /// Per-expert capacity (see [`MoeDispatchAttrs::capacity_for`]).
    pub capacity: usize,
    /// For each expert `e`, the list of tokens assigned to it (in the
    /// order they arrived — deterministic under the top-k tie break
    /// `(descending gate, ascending expert index, ascending arrival)`).
    /// Length is exactly `num_experts`.
    pub expert_assignments: Vec<Vec<MoeAssignment>>,
    /// Fraction of `(token, k-slot)` pairs that were dropped because the
    /// per-expert capacity was already full (only nonzero when
    /// `drop_tokens = true`). `0.0` means every top-k assignment
    /// landed on its intended expert.
    pub drop_rate: f32,
}

/// Runs the softmax + top-k + capacity-factor routing.
///
/// `router_logits` is a `[num_tokens, num_experts]` row-major matrix of
/// per-token per-expert logits (raw, pre-softmax). This is the standard
/// output of the gate `Linear(hidden → num_experts)` layer that every
/// released Switch / Mixtral / DeepSeek-MoE / Qwen3-MoE / Zonos2-MoE
/// checkpoint carries.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if
///   [`MoeDispatchAttrs::validate`] rejects the attrs.
/// - [`VokraError::InvalidArgument`] if `router_logits.len()` does not
///   equal `num_tokens * num_experts`.
/// - [`VokraError::InvalidArgument`] if any logit is NaN (a silent
///   NaN → 0-gate substitution would hide numeric corruption from an
///   upstream layer — FR-EX-08).
pub fn moe_dispatch(
    router_logits: &[f32],
    num_tokens: usize,
    attrs: &MoeDispatchAttrs,
) -> Result<MoeDispatchPlan> {
    attrs.validate()?;
    let expected = num_tokens.saturating_mul(attrs.num_experts);
    if router_logits.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "moe_dispatch: router_logits has {} elements, expected {} \
             (num_tokens={} * num_experts={})",
            router_logits.len(),
            expected,
            num_tokens,
            attrs.num_experts
        )));
    }

    let capacity = attrs.capacity_for(num_tokens);
    let mut expert_assignments: Vec<Vec<MoeAssignment>> =
        (0..attrs.num_experts).map(|_| Vec::new()).collect();
    let mut attempted_slots: usize = 0;
    let mut dropped_slots: usize = 0;

    // Reusable per-token softmax + top-k buffers to avoid a Vec allocation
    // per token on the hot path.
    let mut probs = vec![0.0f32; attrs.num_experts];
    let mut topk_idx = vec![0usize; attrs.top_k];
    let mut topk_p = vec![0.0f32; attrs.top_k];

    for t in 0..num_tokens {
        let row = &router_logits[t * attrs.num_experts..(t + 1) * attrs.num_experts];
        // Reject NaN loudly (FR-EX-08 — a silent NaN → 0-gate would
        // corrupt downstream GEMM without any diagnostic).
        for (e, &lg) in row.iter().enumerate() {
            if lg.is_nan() {
                return Err(VokraError::InvalidArgument(format!(
                    "moe_dispatch: router_logits[token={t}, expert={e}] is NaN"
                )));
            }
        }

        // Numerically-stable softmax over the row.
        let mut max_lg = f32::NEG_INFINITY;
        for &lg in row.iter() {
            if lg > max_lg {
                max_lg = lg;
            }
        }
        let mut sum = 0.0f32;
        for (e, &lg) in row.iter().enumerate() {
            let p = (lg - max_lg).exp();
            probs[e] = p;
            sum += p;
        }
        // `sum` is strictly positive: every term is exp(finite - max) with
        // at least the argmax term = exp(0) = 1.
        let inv = 1.0f32 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }

        // Top-k selection with a deterministic tie break:
        // `(descending probability, ascending expert index)`.
        // Implemented as a small in-place insertion pass — `top_k` is
        // O(few) for every consumer of this crate, so a full sort is
        // wasted work.
        // We seed with the sentinel `-inf` so a genuine `probs[e] = 0.0`
        // still overrides the sentinel.
        for slot in 0..attrs.top_k {
            topk_p[slot] = f32::NEG_INFINITY;
            topk_idx[slot] = usize::MAX;
        }
        for (e, &p) in probs.iter().enumerate() {
            // Try to insert (p, e) into the top-k in descending order.
            let mut insert_at = attrs.top_k;
            for slot in 0..attrs.top_k {
                let (sp, se) = (topk_p[slot], topk_idx[slot]);
                // Descending by probability; ascending by expert index
                // on ties (deterministic so parity tests reproduce byte-
                // exactly across runs).
                if p > sp || (p == sp && e < se) {
                    insert_at = slot;
                    break;
                }
            }
            if insert_at < attrs.top_k {
                // Shift down and drop the tail.
                let mut cur_p = p;
                let mut cur_e = e;
                for slot in insert_at..attrs.top_k {
                    std::mem::swap(&mut topk_p[slot], &mut cur_p);
                    std::mem::swap(&mut topk_idx[slot], &mut cur_e);
                }
            }
        }

        // Optional gate renormalisation (Mixtral default).
        if attrs.renormalize_gates && attrs.top_k > 1 {
            let s: f32 = topk_p.iter().sum();
            if s > 0.0 {
                let inv = 1.0 / s;
                for p in topk_p.iter_mut() {
                    *p *= inv;
                }
            }
        }

        // Emit the k assignments in `(most-preferred expert first)`
        // order — this is the dispatch order downstream reduction
        // relies on.
        for slot in 0..attrs.top_k {
            let e = topk_idx[slot];
            debug_assert!(e < attrs.num_experts, "top-k emitted a sentinel index");
            let gate = topk_p[slot];
            attempted_slots += 1;
            if attrs.drop_tokens && expert_assignments[e].len() >= capacity {
                dropped_slots += 1;
                continue;
            }
            expert_assignments[e].push(MoeAssignment {
                token_idx: t,
                gate_weight: gate,
            });
        }
    }

    let drop_rate = if attempted_slots == 0 {
        0.0
    } else {
        (dropped_slots as f32) / (attempted_slots as f32)
    };

    Ok(MoeDispatchPlan {
        num_tokens,
        num_experts: attrs.num_experts,
        top_k: attrs.top_k,
        capacity,
        expert_assignments,
        drop_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_1of4() -> MoeDispatchAttrs {
        MoeDispatchAttrs {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 2.0,
            drop_tokens: true,
            renormalize_gates: true,
        }
    }

    #[test]
    fn top1_argmax_routing_sends_each_token_to_its_argmax_expert() {
        // 3 tokens, 4 experts. Argmax per row is (0, 2, 3).
        #[rustfmt::skip]
        let logits = vec![
            10.0,  1.0,  1.0,  1.0, // t0 → expert 0
             1.0,  1.0, 10.0,  1.0, // t1 → expert 2
             1.0,  1.0,  1.0, 10.0, // t2 → expert 3
        ];
        let attrs = attrs_1of4();
        let plan = moe_dispatch(&logits, 3, &attrs).unwrap();
        assert_eq!(plan.num_tokens, 3);
        assert_eq!(plan.num_experts, 4);
        assert_eq!(plan.top_k, 1);
        assert_eq!(plan.drop_rate, 0.0);

        assert_eq!(plan.expert_assignments[0].len(), 1);
        assert_eq!(plan.expert_assignments[0][0].token_idx, 0);
        assert_eq!(plan.expert_assignments[1].len(), 0);
        assert_eq!(plan.expert_assignments[2].len(), 1);
        assert_eq!(plan.expert_assignments[2][0].token_idx, 1);
        assert_eq!(plan.expert_assignments[3].len(), 1);
        assert_eq!(plan.expert_assignments[3][0].token_idx, 2);
    }

    #[test]
    fn top1_gate_weight_is_softmax_probability_of_argmax() {
        // 1 token, 3 experts. logits = [2, 0, 0] → probs proportional
        // to (e^2, 1, 1). Argmax is expert 0. Its softmax probability
        // is e^2 / (e^2 + 2) ≈ 0.7869 (top_k=1, renormalise trivially).
        let logits = vec![2.0f32, 0.0, 0.0];
        let attrs = MoeDispatchAttrs {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 4.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&logits, 1, &attrs).unwrap();
        let expected = (2.0f32).exp() / ((2.0f32).exp() + 2.0);
        let got = plan.expert_assignments[0][0].gate_weight;
        assert!(
            (got - expected).abs() < 1e-6,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn top2_mixtral_renormalises_gate_weights_to_sum_to_one() {
        // 1 token, 4 experts. logits = [3, 2, 1, 0] → top-2 = experts
        // (0, 1). Mixtral re-normalises the two-way gate so their
        // weights sum to 1.
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let attrs = MoeDispatchAttrs {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 4.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&logits, 1, &attrs).unwrap();
        // Expert 0 and expert 1 each get one assignment.
        assert_eq!(plan.expert_assignments[0].len(), 1);
        assert_eq!(plan.expert_assignments[1].len(), 1);
        // Their gate weights should sum to 1 (renormalise = true).
        let g0 = plan.expert_assignments[0][0].gate_weight;
        let g1 = plan.expert_assignments[1][0].gate_weight;
        assert!(
            (g0 + g1 - 1.0).abs() < 1e-6,
            "renormalised gates should sum to 1, got {g0} + {g1} = {}",
            g0 + g1
        );
        // And expert 0 (higher logit) must have the larger weight.
        assert!(g0 > g1, "expected g0 > g1, got g0={g0}, g1={g1}");
    }

    #[test]
    fn top2_no_renormalise_leaves_softmax_probabilities() {
        // Same as above but renormalise = false → gate weights are the
        // raw softmax probabilities of the top-2 experts.
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let attrs = MoeDispatchAttrs {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 4.0,
            drop_tokens: true,
            renormalize_gates: false,
        };
        let plan = moe_dispatch(&logits, 1, &attrs).unwrap();
        // Softmax denominator: e^3 + e^2 + e^1 + e^0.
        let denom: f32 = [3.0f32, 2.0, 1.0, 0.0].iter().map(|x| x.exp()).sum();
        let expected0 = 3.0f32.exp() / denom;
        let expected1 = 2.0f32.exp() / denom;
        let g0 = plan.expert_assignments[0][0].gate_weight;
        let g1 = plan.expert_assignments[1][0].gate_weight;
        assert!((g0 - expected0).abs() < 1e-6, "g0: {g0} vs {expected0}");
        assert!((g1 - expected1).abs() < 1e-6, "g1: {g1} vs {expected1}");
    }

    #[test]
    fn capacity_factor_bounds_per_expert_load_switch_style() {
        // 8 tokens, 2 experts, top_k=1, capacity_factor=1.0 → ideal
        // capacity = ceil(8 * 1 / 2) = 4. All 8 tokens want expert 0
        // (logits favour 0 always). Expected: 4 land, 4 are dropped.
        let mut logits = Vec::with_capacity(16);
        for _ in 0..8 {
            logits.extend_from_slice(&[5.0f32, 0.0]);
        }
        let attrs = MoeDispatchAttrs {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 1.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&logits, 8, &attrs).unwrap();
        assert_eq!(plan.capacity, 4);
        assert_eq!(plan.expert_assignments[0].len(), 4);
        assert_eq!(plan.expert_assignments[1].len(), 0);
        // 4 of 8 top-1 slots were dropped.
        let expected_rate = 4.0f32 / 8.0;
        assert!(
            (plan.drop_rate - expected_rate).abs() < 1e-6,
            "expected drop_rate = {expected_rate}, got {}",
            plan.drop_rate
        );
    }

    #[test]
    fn soft_capacity_keeps_every_assignment() {
        // Same over-loaded routing but `drop_tokens = false`: every
        // top-1 slot is retained and `drop_rate` stays 0.
        let mut logits = Vec::with_capacity(16);
        for _ in 0..8 {
            logits.extend_from_slice(&[5.0f32, 0.0]);
        }
        let attrs = MoeDispatchAttrs {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 1.0,
            drop_tokens: false,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&logits, 8, &attrs).unwrap();
        assert_eq!(plan.expert_assignments[0].len(), 8);
        assert_eq!(plan.expert_assignments[1].len(), 0);
        assert_eq!(plan.drop_rate, 0.0);
    }

    #[test]
    fn capacity_for_zero_tokens_is_zero() {
        // Empty dispatch (0 tokens) → capacity 0 (there is nothing to
        // route, so a per-expert budget makes no sense). This is a
        // property of the formula alone; the actual `moe_dispatch`
        // call with num_tokens=0 also exercises the "no-op loop"
        // code path.
        let attrs = attrs_1of4();
        assert_eq!(attrs.capacity_for(0), 0);
    }

    #[test]
    fn capacity_for_never_rounds_down_to_zero() {
        // 1 token, 8 experts, top_k=1, capacity_factor=0.1 → ideal
        // = 0.0125, ceiling = 1. Silent 0-capacity would drop the
        // sole token — the min-1 clamp exists exactly to catch that
        // (FR-EX-08).
        let attrs = MoeDispatchAttrs {
            num_experts: 8,
            top_k: 1,
            capacity_factor: 0.1,
            drop_tokens: true,
            renormalize_gates: true,
        };
        assert!(attrs.capacity_for(1) >= 1);
    }

    #[test]
    fn tie_break_is_deterministic_by_ascending_expert_index() {
        // 1 token, 4 experts, all logits equal → after softmax all
        // probabilities equal 0.25. The tie-break is `ascending expert
        // index`, so top-2 must be (expert 0, expert 1).
        let logits = vec![1.0f32, 1.0, 1.0, 1.0];
        let attrs = MoeDispatchAttrs {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 4.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&logits, 1, &attrs).unwrap();
        assert_eq!(plan.expert_assignments[0].len(), 1);
        assert_eq!(plan.expert_assignments[1].len(), 1);
        assert_eq!(plan.expert_assignments[2].len(), 0);
        assert_eq!(plan.expert_assignments[3].len(), 0);
    }

    #[test]
    fn nan_logit_is_rejected_loudly() {
        // A NaN anywhere in the row must fail — silent NaN → 0-gate
        // would corrupt the downstream GEMM without any diagnostic.
        let logits = vec![1.0f32, f32::NAN, 3.0, 0.0];
        let attrs = attrs_1of4();
        let err = moe_dispatch(&logits, 1, &attrs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NaN"), "expected NaN diagnostic, got {msg}");
    }

    #[test]
    fn wrong_router_logits_length_is_rejected_loudly() {
        // Shape mismatch must be a hard error with dimensions in the
        // message (FR-EX-08 policy the rest of the crate follows).
        let logits = vec![1.0f32; 7]; // not 3 * 4 = 12
        let attrs = attrs_1of4();
        let err = moe_dispatch(&logits, 3, &attrs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains('7'), "expected offending length in msg: {msg}");
        assert!(
            msg.contains("num_tokens=3"),
            "expected num_tokens in msg: {msg}"
        );
    }

    #[test]
    fn top_k_greater_than_num_experts_is_rejected() {
        let attrs = MoeDispatchAttrs {
            num_experts: 2,
            top_k: 3,
            capacity_factor: 1.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let logits = vec![1.0f32, 2.0];
        let err = moe_dispatch(&logits, 1, &attrs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("top_k"), "expected top_k diagnostic: {msg}");
    }

    #[test]
    fn zero_num_experts_is_rejected() {
        let attrs = MoeDispatchAttrs {
            num_experts: 0,
            top_k: 1,
            capacity_factor: 1.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let err = moe_dispatch(&[], 0, &attrs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("num_experts"),
            "expected num_experts diagnostic: {msg}"
        );
    }

    #[test]
    fn zero_capacity_factor_is_rejected() {
        let attrs = MoeDispatchAttrs {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 0.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let logits = vec![1.0f32; 4];
        let err = moe_dispatch(&logits, 1, &attrs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("capacity_factor"),
            "expected capacity_factor diagnostic: {msg}"
        );
    }

    #[test]
    fn negative_capacity_factor_is_rejected() {
        let attrs = MoeDispatchAttrs {
            num_experts: 4,
            top_k: 1,
            capacity_factor: -1.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let logits = vec![1.0f32; 4];
        let err = moe_dispatch(&logits, 1, &attrs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("capacity_factor"),
            "expected capacity_factor diagnostic: {msg}"
        );
    }

    #[test]
    fn empty_dispatch_returns_empty_plan() {
        // 0 tokens is a valid corner case (e.g. a decoder step with
        // zero new tokens). The plan should carry num_tokens = 0 and
        // one empty vec per expert.
        let attrs = attrs_1of4();
        let plan = moe_dispatch(&[], 0, &attrs).unwrap();
        assert_eq!(plan.num_tokens, 0);
        assert_eq!(plan.expert_assignments.len(), 4);
        assert!(plan.expert_assignments.iter().all(|v| v.is_empty()));
        assert_eq!(plan.drop_rate, 0.0);
    }

    #[test]
    fn dispatch_order_matches_arrival_order_across_tokens() {
        // 4 tokens, all wanting expert 0, capacity 4 → each land in
        // arrival order (t0, t1, t2, t3). This is the invariant the
        // per-expert GEMM packing relies on.
        let mut logits = Vec::with_capacity(16);
        for _ in 0..4 {
            logits.extend_from_slice(&[5.0f32, 0.0, 0.0, 0.0]);
        }
        let attrs = MoeDispatchAttrs {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 4.0, // ceil(4 * 1 / 4 * 4) = 4, enough for all
            drop_tokens: true,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&logits, 4, &attrs).unwrap();
        assert_eq!(plan.expert_assignments[0].len(), 4);
        for (i, a) in plan.expert_assignments[0].iter().enumerate() {
            assert_eq!(a.token_idx, i);
        }
    }
}
