//! Log-mel fusion pattern (M2-04-T04).
//!
//! # What is matched
//!
//! The pattern anchors on an [`OpKind::Stft`] node and greedily walks the
//! chain
//!
//! ```text
//! Stft(stft_attrs) → (mul-squared-magnitude proxy)
//!                  → MelFilterbank(mel_attrs)
//!                  → (log proxy)
//! ```
//!
//! and rewrites it into a single
//! [`OpKind::Fused(FusedOp::LogMel { stft, mel })`](crate::ir::FusedOp::LogMel)
//! node. The four consumed nodes collapse to one → the total node count
//! decreases by three per match.
//!
//! # Op proxies (no Power / Log op in the IR today)
//!
//! Vokra's IR does not (yet) have an explicit `Power` / `Log10` op —
//! the log-mel front-end lives entirely inside
//! `crates/vokra-models/src/whisper/mel.rs::log_mel`, which calls the
//! scalar Rust helpers `Spectrogram::power()` and `.log10()` on the
//! materialised intermediates. **No `AudioGraph` in-tree currently
//! contains this chain**: the pattern is therefore exercised only by
//! synthetic unit-test graphs. The imperative-frontend routing that
//! actually drives log-mel through the fused kernel today is
//! M2-04-T08 (`vokra-models` toggle), out of scope here.
//!
//! The synthetic proxy chosen for M2-04-T04 is [`OpKind::Mul`] — the
//! only element-wise op in `OpKind` today can stand in for both the
//! squared-magnitude reduction (`real² + imag²` collapses to a single
//! element-wise op in an IR that carries no split-complex tensors) and
//! the log slot (an element-wise unary transform; the real `log10` is
//! applied inside the fused kernel body — see M2-04-T05, out of this
//! file's scope). The pattern's log-slot check therefore also accepts
//! [`OpKind::Mul`] until a first-class `Log10` op lands (post-M2-07);
//! at that point `try_match` swaps the proxy variant without touching
//! the rewrite shape or [`FusedOp`] payload.
//!
//! # Single-consumer edges only
//!
//! Intermediate tensors on the fused chain (STFT output, power output,
//! mel output) must have **exactly one downstream reader**. A shared
//! intermediate blocks the fusion: another consumer would be left
//! dangling after the four producing nodes are removed. `try_match`
//! scans every node's inputs to enforce this locally.
//!
//! # FR-EX-08 de-fusion
//!
//! This file only *produces* rewrites; the fusion driver (`fuse`,
//! landing separately) is responsible for querying backend support and
//! discarding rewrites the target cannot execute. A [`FusedOp`] handed
//! to an unsupported backend surfaces as
//! [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp),
//! never a silent cross-backend fallback (parent module ADR,
//! FR-EX-08).

use crate::ir::fusion::{FusedOp, FusionPattern, FusionRewrite};
use crate::ir::graph::{AudioGraph, Node, OpKind};
use crate::ir::tensor::TensorId;

/// The Whisper log-mel front-end fusion pattern
/// (`Stft → power-proxy → MelFilterbank → log-proxy`).
///
/// See the module documentation for the proxy-op rationale and the
/// single-consumer-edge rule.
#[derive(Debug, Default, Clone, Copy)]
pub struct LogMelPattern;

impl LogMelPattern {
    /// Construct a fresh instance (patterns are stateless).
    pub(crate) fn new() -> Self {
        Self
    }
}

impl FusionPattern for LogMelPattern {
    fn name(&self) -> &'static str {
        "logmel"
    }

    fn fused_variant_probe(&self) -> FusedOp {
        // The driver only inspects the variant tag (via `Backend::supports`),
        // so a default attribute payload is enough.
        FusedOp::LogMel {
            stft: crate::ir::graph::StftAttrs::new(400, 160),
            mel: crate::ir::graph::MelAttrs::new(16_000, 400, 80),
        }
    }

    fn try_match(&self, graph: &AudioGraph, root_node: usize) -> Option<FusionRewrite> {
        let nodes = graph.nodes();
        let stft_node = nodes.get(root_node)?;
        let stft_attrs = match stft_node.op() {
            OpKind::Stft(a) => a.clone(),
            _ => return None,
        };
        // The STFT op is expected to produce a single output tensor
        // consumed by the power proxy (Mul).
        let stft_out = single_output(stft_node)?;

        // The power proxy node: only downstream consumer of the STFT
        // output.
        let (power_idx, power_node) = unique_consumer(nodes, stft_out)?;
        if !is_elementwise_proxy(power_node.op()) {
            return None;
        }
        let power_out = single_output(power_node)?;

        // The mel filter-bank node: only downstream consumer of the
        // power output.
        let (mel_idx, mel_node) = unique_consumer(nodes, power_out)?;
        let mel_attrs = match mel_node.op() {
            OpKind::MelFilterbank(a) => a.clone(),
            _ => return None,
        };
        let mel_out = single_output(mel_node)?;

        // The log proxy node: only downstream consumer of the mel
        // output. Its output tensor id becomes the fused node's output
        // — the fused node presents the same log-mel tensor id to
        // downstream code as the un-fused chain did.
        let (log_idx, log_node) = unique_consumer(nodes, mel_out)?;
        if !is_elementwise_proxy(log_node.op()) {
            return None;
        }
        let log_out = single_output(log_node)?;

        // The four indices must be distinct (a node cannot appear twice
        // in a topological chain), but assert it defensively so a
        // pathological input graph does not silently emit a rewrite
        // that removes fewer nodes than expected.
        debug_assert!(
            root_node != power_idx
                && root_node != mel_idx
                && root_node != log_idx
                && power_idx != mel_idx
                && power_idx != log_idx
                && mel_idx != log_idx,
            "logmel pattern selected overlapping node indices — graph invariants broken"
        );

        // Collapse the four nodes into a single fused op. The fused
        // node's input is the STFT input (the raw PCM tensor), its
        // output is the log-proxy output (the log-mel tensor).
        let stft_in = stft_node.inputs().first().copied()?;
        let fused = Node {
            op: OpKind::Fused(FusedOp::LogMel {
                stft: stft_attrs,
                mel: mel_attrs,
            }),
            inputs: vec![stft_in],
            outputs: vec![log_out],
        };

        // Self-remap the pattern's output tensor. The fused node writes to
        // the same [`TensorId`] the un-fused chain's last node did (`log_out`),
        // so downstream readers — including graph-output declarations — are
        // preserved without any actual redirection. The [`fuse`] driver's
        // single-consumer edge check inspects `tensor_remap` to know which
        // "removed-node outputs" are legitimately observed externally; a
        // self-remap tells it "yes, this tensor survives, don't count it as an
        // orphaned intermediate."
        let mut tensor_remap = std::collections::HashMap::new();
        tensor_remap.insert(log_out, log_out);

        Some(FusionRewrite {
            removed_nodes: vec![root_node, power_idx, mel_idx, log_idx],
            inserted_node: fused,
            tensor_remap,
        })
    }
}

// ---------------------------------------------------------------------------
// Small graph-inspection helpers. These are file-private on purpose:
// they encode the local invariants of this specific pattern (each
// intermediate has exactly one output tensor, that tensor has exactly
// one downstream reader). Broader graph-topology utilities belong on
// `AudioGraph` itself and land with the general `fuse` driver.
// ---------------------------------------------------------------------------

/// Returns the sole output [`TensorId`] of `node`, or `None` if the
/// node has zero or more than one output.
fn single_output(node: &Node) -> Option<TensorId> {
    match node.outputs() {
        [t] => Some(*t),
        _ => None,
    }
}

/// Element-wise proxy op accepted in the power / log slot. Until the
/// IR grows first-class `Power` / `Log10` ops (post-M2-07), the
/// synthetic-graph tests wire [`OpKind::Mul`] in the power slot and
/// [`OpKind::Softmax`] in the log slot; this helper centralises the
/// accept list so the future promotion is a single-line change to
/// `matches!(op, OpKind::Power | OpKind::Log10)`.
fn is_elementwise_proxy(op: &OpKind) -> bool {
    matches!(op, OpKind::Mul | OpKind::Softmax)
}

/// Returns the sole downstream consumer of `tensor` in `nodes` as
/// `(index, &Node)`, or `None` if the tensor has zero or more than one
/// consumer. This is the single-consumer-edge check documented in the
/// module header.
fn unique_consumer<'a>(nodes: &'a [Node], tensor: TensorId) -> Option<(usize, &'a Node)> {
    let mut hit: Option<(usize, &'a Node)> = None;
    for (i, node) in nodes.iter().enumerate() {
        if node.inputs().contains(&tensor) {
            if hit.is_some() {
                // Second consumer → not a single-consumer edge.
                return None;
            }
            hit = Some((i, node));
        }
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{GraphBuilder, MelAttrs, StftAttrs};
    use crate::ir::tensor::{DType, TensorDesc};

    fn t(name: &str) -> TensorDesc {
        TensorDesc::new(name, DType::F32, [1])
    }

    fn build_chain_graph() -> AudioGraph {
        // Layout: pcm → Stft → spec → Mul(=power) → power → MelFilterbank
        //  → mel → Mul(=log) → logmel.
        //
        // Both proxy slots use `Mul` — the only element-wise op in
        // `OpKind` today. See the module docs on op proxies. Each
        // intermediate has exactly one downstream reader, which is the
        // invariant the LogMelPattern requires.
        let mut b = GraphBuilder::new();
        let pcm = b.add_tensor(t("pcm"));
        let spec = b.add_tensor(t("spec"));
        let power = b.add_tensor(t("power"));
        let mel = b.add_tensor(t("mel"));
        let logmel = b.add_tensor(t("logmel"));

        b.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[pcm], &[spec]);
        // Mul stands in for the squared-magnitude reduction; the second
        // input is the same tensor so the node is structurally valid
        // (both refs land in `spec`).
        b.add_node(OpKind::Mul, &[spec, spec], &[power]);
        b.add_node(
            OpKind::MelFilterbank(MelAttrs::new(16_000, 400, 80)),
            &[power],
            &[mel],
        );
        // Mul as log proxy — the real log10 is applied inside the
        // fused kernel body (M2-04-T05, out of scope here).
        b.add_node(OpKind::Mul, &[mel, mel], &[logmel]);

        b.mark_input(pcm);
        b.mark_output(logmel);
        b.finish().expect("chain graph builds")
    }

    #[test]
    fn synthetic_stft_power_mel_log_chain_fuses_to_single_node() {
        let mut graph = build_chain_graph();
        let before = graph.nodes().len();
        assert_eq!(before, 4, "chain should have 4 nodes pre-fusion");

        let rewrite = LogMelPattern::new()
            .try_match(&graph, 0)
            .expect("chain matches the LogMel pattern at node 0");

        // Every node in the four-op chain is removed.
        assert_eq!(rewrite.removed_nodes.len(), 4);
        // The replacement is a single FusedOp::LogMel node.
        assert!(matches!(
            rewrite.inserted_node.op(),
            OpKind::Fused(FusedOp::LogMel { .. })
        ));

        // Apply the rewrite and confirm the node count drops by three.
        graph.rewrite_with(vec![rewrite]);
        let after = graph.nodes().len();
        assert_eq!(after, before - 3, "node count should decrease by three");
        assert_eq!(after, 1);

        // The remaining node is the fused LogMel node with both attribute
        // payloads preserved.
        match graph.nodes()[0].op() {
            OpKind::Fused(FusedOp::LogMel { stft, mel }) => {
                assert_eq!(stft.n_fft, 400);
                assert_eq!(stft.hop_length, 160);
                assert_eq!(mel.sample_rate, 16_000);
                assert_eq!(mel.n_fft, 400);
                assert_eq!(mel.n_mels, 80);
            }
            other => panic!("expected FusedOp::LogMel, got {other:?}"),
        }
    }

    #[test]
    fn non_stft_root_does_not_match() {
        // Root anchored at the power-proxy (Mul) node — pattern must
        // decline because only Stft can anchor the chain.
        let graph = build_chain_graph();
        assert!(LogMelPattern::new().try_match(&graph, 1).is_none());
    }

    #[test]
    fn multi_consumer_intermediate_blocks_fusion() {
        // Build the log-mel chain, then attach a *second* consumer to
        // the power tensor: the single-consumer-edge rule must reject
        // the match.
        let mut b = GraphBuilder::new();
        let pcm = b.add_tensor(t("pcm"));
        let spec = b.add_tensor(t("spec"));
        let power = b.add_tensor(t("power"));
        let mel = b.add_tensor(t("mel"));
        let logmel = b.add_tensor(t("logmel"));
        let sink = b.add_tensor(t("sink"));

        b.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[pcm], &[spec]);
        b.add_node(OpKind::Mul, &[spec, spec], &[power]);
        b.add_node(
            OpKind::MelFilterbank(MelAttrs::new(16_000, 400, 80)),
            &[power],
            &[mel],
        );
        b.add_node(OpKind::Mul, &[mel, mel], &[logmel]);
        // Second consumer of `power` — blocks fusion.
        b.add_node(OpKind::Add, &[power, power], &[sink]);

        b.mark_input(pcm);
        b.mark_output(logmel);
        b.mark_output(sink);
        let graph = b.finish().expect("multi-consumer graph builds");

        assert!(
            LogMelPattern::new().try_match(&graph, 0).is_none(),
            "shared intermediate must block the fusion"
        );
    }

    #[test]
    fn fused_variant_probe_is_log_mel() {
        // The FR-EX-08 de-fusion probe must reflect the pattern's
        // target variant so `Backend::supports` can answer correctly.
        assert!(matches!(
            LogMelPattern::new().fused_variant_probe(),
            FusedOp::LogMel { .. },
        ));
    }
}
