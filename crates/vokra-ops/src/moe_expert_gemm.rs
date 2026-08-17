//! Mixture-of-Experts (MoE) per-expert GEMM primitive (SoTA plan
//! Wave C blocker for `qwen3-omni-30b-a3b-moe` and `zonos2-8b-moe` —
//! see `docs/tickets/coverage-audit-2026-08-03/IMPL-PLAN.md` §2.3).
//!
//! Consumes an [`MoeDispatchPlan`] produced by
//! [`crate::moe_dispatch::moe_dispatch`] and a per-expert weight
//! bundle, and folds each expert's contribution back into the per-token
//! output. Splitting `moe_dispatch` from this GEMM keeps the routing
//! decision independently testable and lets a later SIMD / GPU / Metal
//! kernel replace this loop without redoing the softmax + top-k math.
//!
//! # Runtime function, not an `OpKind` variant
//!
//! Same posture as [`crate::moe_dispatch`] and
//! [`crate::openwakeword::openwakeword_classifier_forward`]: the
//! per-expert weight bundle (a `Vec<Vec<f32>>` of variable outer length)
//! does not fit the flat-tensor `OpValue` dispatch surface. See ADR
//! M3-06 §D-b.
//!
//! # Per-expert loop, not a batched sparse GEMM
//!
//! The initial implementation is a **per-expert inner loop**
//! (Switch-Transformer reference style), not a batched sparse-GEMM
//! kernel. This is the "SIMD 化 optional、初版は per-expert loop で可"
//! guidance from `IMPL-PLAN.md §2.3`. Every accumulator uses a `f64`
//! reduction to protect against the `f32` cancellation drift that hits
//! large `in_dim` (`Qwen3-Omni-A3B` inner MoE = 5120-d), a habit
//! established by the Kokoro T17-fixup #1 audit
//! (`crates/vokra-models/src/kokoro/decoder.rs`).
//!
//! # FR-EX-08 loud-fail contract
//!
//! Every shape mismatch — wrong per-expert weight length, wrong bias
//! width, wrong input length, dispatch plan referencing a token index
//! past the input's `num_tokens`, or an expert index past the weight
//! bundle's `num_experts` — is a hard
//! [`vokra_core::VokraError::InvalidArgument`] naming the offending
//! dimension. No silent zero-pad, no silent truncation.

use crate::moe_dispatch::MoeDispatchPlan;
use vokra_core::{Result, VokraError};

/// Per-expert weight bundle for [`moe_expert_gemm`].
///
/// The outer `Vec` length is the expert count. Every entry is a flat
/// row-major `[out_dim, in_dim]` weight matrix; if `biases` is
/// `Some(_)`, every entry there is a length-`out_dim` bias vector for
/// the corresponding expert. Vokra's binder policy is one flat `Vec`
/// per tensor (mirror of [`crate::dac_rvq::DacOutProj`] and
/// [`crate::hifigan::HifiGanWeights`]).
#[derive(Debug, Clone, PartialEq)]
pub struct MoeExpertWeights {
    /// Input width (columns of every expert weight).
    pub in_dim: usize,
    /// Output width (rows of every expert weight).
    pub out_dim: usize,
    /// One `[out_dim, in_dim]` row-major matrix per expert. The outer
    /// length is `num_experts` (must match the dispatch plan's
    /// `num_experts`).
    pub experts: Vec<Vec<f32>>,
    /// Optional per-expert bias `[out_dim]`. `None` means every expert
    /// has bias = 0. `Some(v)` requires `v.len() == experts.len()` and
    /// every inner slice has length `out_dim`.
    pub biases: Option<Vec<Vec<f32>>>,
}

impl MoeExpertWeights {
    /// Validates the shape contract loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any dimension mismatch.
    pub fn validate(&self) -> Result<()> {
        if self.experts.is_empty() {
            return Err(VokraError::InvalidArgument(
                "moe_expert_gemm: experts must be non-empty (got 0 experts)".to_owned(),
            ));
        }
        if self.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "moe_expert_gemm: in_dim must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.out_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "moe_expert_gemm: out_dim must be > 0 (got 0)".to_owned(),
            ));
        }
        let expected_len = self.out_dim.saturating_mul(self.in_dim);
        for (e, weight) in self.experts.iter().enumerate() {
            if weight.len() != expected_len {
                return Err(VokraError::InvalidArgument(format!(
                    "moe_expert_gemm: expert[{e}] weight has {} elements, expected {} \
                     (out_dim={} * in_dim={})",
                    weight.len(),
                    expected_len,
                    self.out_dim,
                    self.in_dim
                )));
            }
        }
        if let Some(biases) = &self.biases {
            if biases.len() != self.experts.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "moe_expert_gemm: biases outer length {} does not match experts \
                     length {}",
                    biases.len(),
                    self.experts.len()
                )));
            }
            for (e, bias) in biases.iter().enumerate() {
                if bias.len() != self.out_dim {
                    return Err(VokraError::InvalidArgument(format!(
                        "moe_expert_gemm: bias[{e}] has {} elements, expected \
                         out_dim={}",
                        bias.len(),
                        self.out_dim
                    )));
                }
            }
        }
        Ok(())
    }

    /// Number of experts the bundle carries (`experts.len()`).
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }
}

/// Runs the per-expert GEMM and reduces the top-k contributions back
/// into a per-token output.
///
/// For every token `t`, the returned `[num_tokens, out_dim]` row is:
///
/// ```text
/// out[t] = Σ_{(t, gate) ∈ expert_assignments[e]}  gate * (W_e @ input[t] + b_e)
///          over all experts e
/// ```
///
/// Tokens that were dropped at the [`crate::moe_dispatch`] step (their
/// slot never made it into any `expert_assignments[e]`) contribute
/// zero — this is the standard Switch / Mixtral behaviour and is why
/// the [`MoeDispatchPlan::drop_rate`] must be surfaced honestly by the
/// caller.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if
///   [`MoeExpertWeights::validate`] rejects the bundle.
/// - [`VokraError::InvalidArgument`] if `weights.num_experts()` does
///   not match `plan.num_experts`.
/// - [`VokraError::InvalidArgument`] if `input.len()` is not
///   `plan.num_tokens * weights.in_dim`.
/// - [`VokraError::InvalidArgument`] if any assignment's `token_idx`
///   is `>= plan.num_tokens` (the dispatch plan was corrupted).
pub fn moe_expert_gemm(
    input: &[f32],
    plan: &MoeDispatchPlan,
    weights: &MoeExpertWeights,
) -> Result<Vec<f32>> {
    weights.validate()?;

    if weights.num_experts() != plan.num_experts {
        return Err(VokraError::InvalidArgument(format!(
            "moe_expert_gemm: weights carry {} experts, plan built for {}",
            weights.num_experts(),
            plan.num_experts
        )));
    }
    if plan.expert_assignments.len() != plan.num_experts {
        return Err(VokraError::InvalidArgument(format!(
            "moe_expert_gemm: dispatch plan corrupt — expert_assignments has {} \
             entries, plan.num_experts is {}",
            plan.expert_assignments.len(),
            plan.num_experts
        )));
    }

    let expected_input = plan.num_tokens.saturating_mul(weights.in_dim);
    if input.len() != expected_input {
        return Err(VokraError::InvalidArgument(format!(
            "moe_expert_gemm: input has {} elements, expected {} \
             (num_tokens={} * in_dim={})",
            input.len(),
            expected_input,
            plan.num_tokens,
            weights.in_dim
        )));
    }

    let out_dim = weights.out_dim;
    let in_dim = weights.in_dim;

    let mut output = vec![0.0f32; plan.num_tokens.saturating_mul(out_dim)];

    // f64 partial-sums per output row on the hot path, folded into the
    // final f32 buffer at the end of each expert loop. The Kokoro
    // T17-fixup #1 audit established that f64 accumulation over the
    // matmul row is the correct precision choice for reference parity
    // (crates/vokra-models/src/kokoro/decoder.rs).
    let mut row_acc = vec![0.0f64; out_dim];

    for (e, assignments) in plan.expert_assignments.iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }
        let weight = &weights.experts[e];
        let bias: Option<&[f32]> = weights.biases.as_ref().map(|b| b[e].as_slice());

        for assignment in assignments {
            let t = assignment.token_idx;
            if t >= plan.num_tokens {
                return Err(VokraError::InvalidArgument(format!(
                    "moe_expert_gemm: dispatch plan corrupt — expert {e} has an \
                     assignment for token {t}, which is out of range (num_tokens={})",
                    plan.num_tokens
                )));
            }
            let x = &input[t * in_dim..(t + 1) * in_dim];
            let gate = assignment.gate_weight;

            // Reset the per-row accumulator (seeded with the bias if
            // present, else 0).
            match bias {
                Some(b) => {
                    for (acc, &bv) in row_acc.iter_mut().zip(b.iter()) {
                        *acc = bv as f64;
                    }
                }
                None => {
                    for acc in row_acc.iter_mut() {
                        *acc = 0.0;
                    }
                }
            }

            // Row of the weight matrix `o` × input dot product.
            for o in 0..out_dim {
                let w_row = &weight[o * in_dim..(o + 1) * in_dim];
                let mut s = 0.0f64;
                for (wv, xv) in w_row.iter().zip(x.iter()) {
                    s += (*wv as f64) * (*xv as f64);
                }
                row_acc[o] += s;
            }

            // Fold gate * accumulator into the output row.
            let out_row = &mut output[t * out_dim..(t + 1) * out_dim];
            for (out_cell, &acc) in out_row.iter_mut().zip(row_acc.iter()) {
                *out_cell += (acc as f32) * gate;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe_dispatch::{MoeAssignment, MoeDispatchPlan};

    /// Builds a plan by hand (bypassing `moe_dispatch`) so the tests
    /// exercise `moe_expert_gemm` in isolation from routing.
    fn build_plan(
        num_tokens: usize,
        num_experts: usize,
        top_k: usize,
        assignments: Vec<Vec<(usize, f32)>>,
    ) -> MoeDispatchPlan {
        assert_eq!(assignments.len(), num_experts);
        MoeDispatchPlan {
            num_tokens,
            num_experts,
            top_k,
            capacity: num_tokens,
            expert_assignments: assignments
                .into_iter()
                .map(|per_expert| {
                    per_expert
                        .into_iter()
                        .map(|(token_idx, gate_weight)| MoeAssignment {
                            token_idx,
                            gate_weight,
                        })
                        .collect()
                })
                .collect(),
            drop_rate: 0.0,
        }
    }

    #[test]
    fn shape_roundtrip_num_tokens_x_out_dim_row_major() {
        // 3 tokens, in_dim=4, out_dim=5. Every expert has an identity-
        // ish weight and 1 assignment. Output length must be 15
        // (num_tokens * out_dim).
        let num_tokens = 3;
        let in_dim = 4;
        let out_dim = 5;
        let num_experts = 3;
        let weights = MoeExpertWeights {
            in_dim,
            out_dim,
            experts: (0..num_experts)
                .map(|_| vec![0.0; out_dim * in_dim])
                .collect(),
            biases: None,
        };
        // t0 → expert 0, t1 → expert 1, t2 → expert 2, gate 1.0 each.
        let plan = build_plan(
            num_tokens,
            num_experts,
            1,
            vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
        );
        let input = vec![0.0f32; num_tokens * in_dim];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert_eq!(out.len(), num_tokens * out_dim);
    }

    #[test]
    fn single_expert_gate1_computes_gemm_row_by_row() {
        // 1 token, in_dim=2, out_dim=3, 1 expert.
        // W = [[1, 2], [3, 4], [5, 6]] (row-major [3, 2]).
        // x = [10, 20].
        // W @ x = [1*10 + 2*20, 3*10 + 4*20, 5*10 + 6*20] = [50, 110, 170].
        // Gate = 1.0, no bias → output row = [50, 110, 170].
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 3,
            experts: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            biases: None,
        };
        let plan = build_plan(1, 1, 1, vec![vec![(0, 1.0)]]);
        let input = vec![10.0f32, 20.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert_eq!(out, vec![50.0, 110.0, 170.0]);
    }

    #[test]
    fn gate_weight_scales_output() {
        // Same setup but gate = 0.5 → every output should halve.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 3,
            experts: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            biases: None,
        };
        let plan = build_plan(1, 1, 1, vec![vec![(0, 0.5)]]);
        let input = vec![10.0f32, 20.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert_eq!(out, vec![25.0, 55.0, 85.0]);
    }

    #[test]
    fn bias_is_added_when_present() {
        // Same W as above but with bias = [100, 200, 300], gate = 1.
        // Row 0: 50 + 100 = 150; row 1: 110 + 200 = 310; row 2: 170 + 300 = 470.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 3,
            experts: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            biases: Some(vec![vec![100.0, 200.0, 300.0]]),
        };
        let plan = build_plan(1, 1, 1, vec![vec![(0, 1.0)]]);
        let input = vec![10.0f32, 20.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert_eq!(out, vec![150.0, 310.0, 470.0]);
    }

    #[test]
    fn top2_expert_contributions_are_summed_with_their_gates() {
        // 1 token, in_dim=2, out_dim=2, 2 experts.
        // Expert 0: W = I₂ (identity) → contribution = 0.6 * [10, 20] = [6, 12].
        // Expert 1: W = 2*I₂ → contribution = 0.4 * [20, 40] = [8, 16].
        // Sum: [14, 28].
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![
                vec![1.0, 0.0, 0.0, 1.0], // I₂
                vec![2.0, 0.0, 0.0, 2.0], // 2*I₂
            ],
            biases: None,
        };
        let plan = build_plan(1, 2, 2, vec![vec![(0, 0.6)], vec![(0, 0.4)]]);
        let input = vec![10.0f32, 20.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert!(
            (out[0] - 14.0).abs() < 1e-6 && (out[1] - 28.0).abs() < 1e-6,
            "expected [14, 28], got {:?}",
            out
        );
    }

    #[test]
    fn dropped_token_contributes_zero() {
        // 2 tokens. Token 0 is assigned to expert 0 with gate 1.0.
        // Token 1 has no assignment (was dropped at dispatch) → its
        // output row must be all zeros.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![1.0, 0.0, 0.0, 1.0]], // I₂
            biases: None,
        };
        let plan = build_plan(2, 1, 1, vec![vec![(0, 1.0)]]);
        let input = vec![10.0f32, 20.0, 30.0, 40.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 0.0, 0.0]);
    }

    #[test]
    fn multi_token_multi_expert_bookkeeping_is_per_token() {
        // 2 tokens, 2 experts, top_k=1.
        // Token 0 → expert 0 (gate 1). Token 1 → expert 1 (gate 1).
        // Expert 0: 2*I₂; expert 1: 3*I₂.
        // Output: row 0 = 2 * [1, 2] = [2, 4]; row 1 = 3 * [10, 20] = [30, 60].
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![2.0, 0.0, 0.0, 2.0], vec![3.0, 0.0, 0.0, 3.0]],
            biases: None,
        };
        let plan = build_plan(2, 2, 1, vec![vec![(0, 1.0)], vec![(1, 1.0)]]);
        let input = vec![1.0f32, 2.0, 10.0, 20.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();
        assert_eq!(out, vec![2.0, 4.0, 30.0, 60.0]);
    }

    #[test]
    fn wrong_input_length_is_rejected_loudly() {
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![1.0, 0.0, 0.0, 1.0]],
            biases: None,
        };
        let plan = build_plan(3, 1, 1, vec![vec![]]);
        // Expected 3 * 2 = 6 f32, got 5.
        let input = vec![0.0f32; 5];
        let err = moe_expert_gemm(&input, &plan, &weights).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("input"), "expected 'input' in msg: {msg}");
        assert!(msg.contains("num_tokens=3"), "expected shape in msg: {msg}");
    }

    #[test]
    fn wrong_expert_count_is_rejected_loudly() {
        // Plan built for 2 experts, weights supply 1.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![1.0, 0.0, 0.0, 1.0]],
            biases: None,
        };
        let plan = build_plan(1, 2, 1, vec![vec![(0, 1.0)], vec![]]);
        let input = vec![1.0f32, 2.0];
        let err = moe_expert_gemm(&input, &plan, &weights).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains('1') && msg.contains('2'),
            "expected 1 and 2 in msg: {msg}"
        );
    }

    #[test]
    fn wrong_expert_weight_length_is_rejected_loudly() {
        // out_dim=2, in_dim=3 → expect 6 elements per expert; provide 5.
        let weights = MoeExpertWeights {
            in_dim: 3,
            out_dim: 2,
            experts: vec![vec![0.0; 5]],
            biases: None,
        };
        let plan = build_plan(0, 1, 1, vec![vec![]]);
        let input: Vec<f32> = vec![];
        let err = moe_expert_gemm(&input, &plan, &weights).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expert[0]"),
            "expected 'expert[0]' in msg: {msg}"
        );
        assert!(msg.contains("out_dim=2"), "expected out_dim in msg: {msg}");
    }

    #[test]
    fn wrong_bias_width_is_rejected_loudly() {
        // out_dim=2, in_dim=2 → expect bias length 2 per expert;
        // provide 3.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![0.0; 4]],
            biases: Some(vec![vec![0.0; 3]]),
        };
        let plan = build_plan(0, 1, 1, vec![vec![]]);
        let input: Vec<f32> = vec![];
        let err = moe_expert_gemm(&input, &plan, &weights).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bias[0]"), "expected 'bias[0]' in msg: {msg}");
    }

    #[test]
    fn dispatch_plan_corrupt_token_index_is_rejected_loudly() {
        // Plan claims a token index 5 for a plan built with num_tokens=2.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![1.0, 0.0, 0.0, 1.0]],
            biases: None,
        };
        let plan = build_plan(2, 1, 1, vec![vec![(5, 1.0)]]);
        let input = vec![0.0f32; 2 * 2];
        let err = moe_expert_gemm(&input, &plan, &weights).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("token 5"), "expected 'token 5' in msg: {msg}");
        assert!(
            msg.contains("num_tokens=2"),
            "expected 'num_tokens=2' in msg: {msg}"
        );
    }

    #[test]
    fn zero_tokens_returns_empty_output() {
        // A dispatch step with 0 tokens is a no-op — the output should
        // be empty and no error should fire.
        let weights = MoeExpertWeights {
            in_dim: 4,
            out_dim: 8,
            experts: vec![vec![0.0; 4 * 8]],
            biases: None,
        };
        let plan = build_plan(0, 1, 1, vec![vec![]]);
        let out = moe_expert_gemm(&[], &plan, &weights).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_experts_bundle_is_rejected() {
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![],
            biases: None,
        };
        let err = weights.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("experts") && msg.contains("non-empty"),
            "expected non-empty diagnostic: {msg}"
        );
    }

    #[test]
    fn end_to_end_with_moe_dispatch_produces_expected_output() {
        // Integration: run the real `moe_dispatch` and then feed the
        // resulting plan into `moe_expert_gemm`. This is the routing-
        // correctness cross-check the spec calls for.
        //
        // 2 tokens, 2 experts, top_k=1, in_dim=out_dim=2.
        // Token 0 logits [10, 0] → expert 0. Token 1 logits [0, 10] → expert 1.
        // Expert 0: W = 3*I₂. Expert 1: W = 5*I₂. No bias.
        // Softmax(10, 0) ≈ 0.99995 → gate ≈ 1.0 (top_k=1 renormalises to 1
        // automatically since it's a single value).
        // Expected: out[0] ≈ 1.0 * 3 * input[0] = [3, 6];
        //           out[1] ≈ 1.0 * 5 * input[1] = [50, 100].
        use crate::moe_dispatch::{MoeDispatchAttrs, moe_dispatch};
        let router_logits = vec![10.0f32, 0.0, 0.0, 10.0];
        let attrs = MoeDispatchAttrs {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 4.0,
            drop_tokens: true,
            renormalize_gates: true,
        };
        let plan = moe_dispatch(&router_logits, 2, &attrs).unwrap();
        // top_k=1 → the single gate weight for a top-1 pick is the raw
        // softmax probability of the argmax expert; renormalise=true on
        // a single value is a no-op (1 / 1 = 1) only when the caller has
        // explicitly normalised, but here it stays at the softmax
        // probability. We assert on the numerical output within a
        // tolerance that admits that ≈0.99995 factor.
        let weights = MoeExpertWeights {
            in_dim: 2,
            out_dim: 2,
            experts: vec![vec![3.0, 0.0, 0.0, 3.0], vec![5.0, 0.0, 0.0, 5.0]],
            biases: None,
        };
        let input = vec![1.0f32, 2.0, 10.0, 20.0];
        let out = moe_expert_gemm(&input, &plan, &weights).unwrap();

        // Softmax(10, 0) = e^10 / (e^10 + 1) ≈ 0.9999546.
        let gate0 = 10.0f32.exp() / (10.0f32.exp() + 1.0);
        // The k=1 code path uses renormalise on a single value, which
        // is `g / g = 1.0` — so top_k=1 renormalised = 1.0 exactly.
        // (See `moe_dispatch::top2_mixtral_renormalises_...` for the
        // k>1 assertion; the k=1 branch skips the renormalise.)
        assert!(gate0 > 0.9999);
        // top_k=1 branch: no renormalisation (the `attrs.top_k > 1`
        // guard in moe_dispatch skips it), so the emitted gate is
        // exactly the softmax probability.
        let expected_r0 = gate0 * 3.0 * 1.0;
        let expected_r1 = gate0 * 3.0 * 2.0;
        let expected_r2 = gate0 * 5.0 * 10.0;
        let expected_r3 = gate0 * 5.0 * 20.0;
        assert!((out[0] - expected_r0).abs() < 1e-4);
        assert!((out[1] - expected_r1).abs() < 1e-4);
        assert!((out[2] - expected_r2).abs() < 1e-3);
        assert!((out[3] - expected_r3).abs() < 1e-3);
    }
}
