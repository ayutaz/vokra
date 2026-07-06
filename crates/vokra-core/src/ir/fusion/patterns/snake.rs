//! Snake / BigVGAN AMP fusion pattern scaffold (M2-04-T09 + T10).
//!
//! # Why this file is (almost entirely) `#[cfg(feature = "fusion-snake-stub")]`
//!
//! Snake activation (FR-OP-13 `snake_activation`), the BigVGAN generator
//! (FR-OP-11 `bigvgan_generator`), and anti-aliased upsampling
//! (FR-OP-14 `anti_aliased_upsample`) are **model-synced** deliverables
//! per SRS §2.3: the kernel bodies land with the first consumer model,
//! not ahead of it. M2 has **no** Snake consumer in tree —
//!
//! - Kokoro-82M (M2-07) uses the iSTFTNet head (StyleTTS 2 derivative);
//! - piper-plus uses LeakyReLU + MRF;
//! - Whisper / CAM++ / Silero VAD never touch Snake.
//!
//! Adding orphan `OpKind::Conv1d` / `OpKind::Snake` / `OpKind::Upsample`
//! variants to the IR before a real caller exists would (a) pollute
//! `OpKind` with unimplementable variants, (b) force every backend's
//! `Backend::supports` / `eval_op` switch to grow arms that only ever
//! return `UnsupportedOp`, (c) create a documentation-tested surface
//! with no reference implementation. Instead this module ships
//! **pattern shapes only**, gated on the `fusion-snake-stub` cargo
//! feature (default = OFF, opt-in for this file's own unit tests).
//!
//! When the consumer model arrives (FR-OP-11/13/14 kernel bodies +
//! e2e parity, tracked as an M2-04-T13 followup issue), the gate is
//! removed, real op variants land in [`OpKind`](crate::ir::graph::OpKind),
//! and the `FusedOp` enum grows `ConvSnake` / `UpSnakeResidual`
//! variants co-delivered with their backends.
//!
//! # What this file registers today
//!
//! Two patterns:
//!
//! - [`Conv1dSnakePattern`] — `Conv1d → Snake` (T09). A single
//!   activation fused after a 1-D convolution; the BigVGAN generator
//!   uses this hundreds of times per forward pass.
//! - [`UpsampleSnakeResidualPattern`] — `Upsample → Snake → Add(residual)`
//!   (T10). The BigVGAN Anti-Aliased Multi-Periodicity block: an
//!   anti-aliased upsample immediately followed by a Snake activation
//!   and a residual add.
//!
//! Neither pattern is registered in
//! [`super::super::pattern_registry`](crate::ir::fusion) because that
//! would drag the `fusion-snake-stub` cfg into the driver's build graph;
//! registration lands in the same PR that promotes Snake / Conv1d /
//! Upsample to real ops in [`OpKind`](crate::ir::graph::OpKind).
//!
//! # Invariants preserved across fusion (recorded here so the followup
//! # kernel authors do not silently drop them)
//!
//! 1. **Snake `internal_precision` default = FP32** (FR-OP-13). Fusion
//!    must NOT downgrade the Snake pre-activation accumulator to BF16
//!    — BF16 mantissa (7 explicit + 1 implicit bit) is insufficient for
//!    Snake's `x + (1/α)·sin²(α·x)` non-linearity, especially in the
//!    small-α regime where `sin²(α·x) ≈ (α·x)²` accumulates. The
//!    `internal_precision` attribute is preserved verbatim into the
//!    fused op — see `FusedOp::ConvSnake { …, internal_precision }`
//!    when it lands.
//!
//! 2. **BigVGAN kernel body pending scratch reimplementation.** The
//!    upstream NVIDIA BigVGAN reference is NVIDIA Source Code License-NC
//!    (non-commercial) — see `docs/license-audit.md` redline. Vokra's
//!    BigVGAN kernel MUST be written from the published paper, not
//!    ported from the NVIDIA implementation. This gate lands in the
//!    same followup that removes the `fusion-snake-stub` cfg.
//!
//! 3. **Vocos / BigVGAN minimum dtype = fp16** (FR-QT-03 / FR-OP-11 /
//!    FR-OP-12). Snake activation, anti-aliased upsample, and their
//!    fused variants MUST refuse INT8 execution — Vocos / BigVGAN
//!    exhibit INT8 崩壊 (mel-loss / UTMOS regressions). The fused
//!    op's dtype validator will return
//!    [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp)
//!    if handed an INT8 activation tensor. (Enforcement lands with
//!    the kernel body.)
//!
//! 4. **FR-EX-08 de-fusion still applies.** Same rule as
//!    [`LogMelPattern`](super::logmel::LogMelPattern): if the target
//!    backend does not implement the fused variant, the pass leaves
//!    the base ops in place — Snake / Conv1d / Upsample base ops
//!    must have full CPU coverage before Snake fusion is registered
//!    in [`super::super::pattern_registry`](crate::ir::fusion).

// Everything below this line is gated on the opt-in cargo feature so the
// default build does not carry references to op variants that do not
// exist in [`OpKind`](crate::ir::graph::OpKind) yet. The gate is removed
// in the follow-up PR that lands the FR-OP-11/13/14 kernel bodies +
// their consumer model (BigVGAN / any Snake-using vocoder).

#[cfg(all(test, feature = "fusion-snake-stub"))]
mod stub {
    //! Stub-op enum and pattern matchers for M2-04-T09 / T10.
    //!
    //! This module is compiled ONLY under
    //! `--features fusion-snake-stub` in `#[cfg(test)]` builds and
    //! exists solely to unit-test the pattern-matcher plumbing ahead of
    //! the consumer model landing. It defines a private `StubOpKind`
    //! enum standing in for the (future) real
    //! [`OpKind::Conv1d`](crate::ir::graph::OpKind) /
    //! [`OpKind::Snake`](crate::ir::graph::OpKind) /
    //! [`OpKind::Upsample`](crate::ir::graph::OpKind) variants.
    //! Nothing in `StubOpKind` is exposed outside this module — a
    //! non-test build with the feature enabled compiles the module out
    //! entirely, so the FR-EX-08 explicit-error contract on the real
    //! `OpKind` surface is never bypassed by the scaffold.

    /// Synthetic op tag standing in for the (future) real IR variants.
    /// Named `StubOpKind` — not `OpKind` — so it can never be confused
    /// with the crate-level [`OpKind`](crate::ir::graph::OpKind) enum
    /// in a diff review.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum StubOpKind {
        /// Stub for FR-OP not-yet-assigned `Conv1d` (1-D convolution).
        Conv1d,
        /// Stub for FR-OP-13 `snake_activation`.
        Snake,
        /// Stub for FR-OP-14 `anti_aliased_upsample`.
        Upsample,
        /// Stub for the residual-add tail of the AMP block. Uses the
        /// real semantic name of the (future) IR op.
        Add,
    }

    /// A synthetic node in a stub graph. Deliberately minimal — this
    /// exercises the matcher's edge-walking logic, not the full
    /// [`Node`](crate::ir::graph::Node) surface.
    #[derive(Debug, Clone)]
    pub(super) struct StubNode {
        pub op: StubOpKind,
        pub inputs: Vec<usize>,
        pub outputs: Vec<usize>,
    }

    /// A synthetic graph over [`StubNode`]s. The matcher walks
    /// `nodes` in insertion order and consults `consumer_count` (populated
    /// by [`StubGraph::finish`]) for the single-consumer-edge check.
    #[derive(Debug, Default)]
    pub(super) struct StubGraph {
        pub nodes: Vec<StubNode>,
        pub consumer_count: Vec<usize>,
    }

    impl StubGraph {
        pub(super) fn new() -> Self {
            Self::default()
        }

        pub(super) fn add_node(
            &mut self,
            op: StubOpKind,
            inputs: &[usize],
            outputs: &[usize],
        ) -> usize {
            let idx = self.nodes.len();
            self.nodes.push(StubNode {
                op,
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
            });
            idx
        }

        /// Populates `consumer_count`. The synthetic graph has no
        /// concept of "graph outputs" so this counts only node-input
        /// reads — sufficient for the pattern-matcher unit tests.
        pub(super) fn finish(mut self, n_tensors: usize) -> Self {
            self.consumer_count = vec![0; n_tensors];
            for node in &self.nodes {
                for &t in &node.inputs {
                    if t < self.consumer_count.len() {
                        self.consumer_count[t] += 1;
                    }
                }
            }
            self
        }

        pub(super) fn unique_consumer(&self, tensor: usize) -> Option<usize> {
            if tensor >= self.consumer_count.len() || self.consumer_count[tensor] != 1 {
                return None;
            }
            self.nodes.iter().position(|n| n.inputs.contains(&tensor))
        }
    }

    /// Match `Conv1d → Snake` anchored at `root_node`. Returns the
    /// indices of the two nodes on success (T09).
    pub(super) fn try_match_conv1d_snake(
        graph: &StubGraph,
        root_node: usize,
    ) -> Option<(usize, usize)> {
        let conv = graph.nodes.get(root_node)?;
        if conv.op != StubOpKind::Conv1d {
            return None;
        }
        let conv_out = *conv.outputs.first()?;
        let snake_idx = graph.unique_consumer(conv_out)?;
        let snake = graph.nodes.get(snake_idx)?;
        if snake.op != StubOpKind::Snake {
            return None;
        }
        debug_assert!(
            root_node != snake_idx,
            "conv1d+snake pattern selected overlapping indices — graph invariants broken"
        );
        Some((root_node, snake_idx))
    }

    /// Match `Upsample → Snake → Add(residual)` anchored at `root_node`.
    /// The `Add` node must have exactly two inputs: the Snake output and
    /// the residual tensor (which came from *outside* the fused chain,
    /// hence its `consumer_count` is unrestricted). Returns the three
    /// node indices on success (T10).
    pub(super) fn try_match_upsample_snake_residual(
        graph: &StubGraph,
        root_node: usize,
    ) -> Option<(usize, usize, usize)> {
        let up = graph.nodes.get(root_node)?;
        if up.op != StubOpKind::Upsample {
            return None;
        }
        let up_out = *up.outputs.first()?;
        let snake_idx = graph.unique_consumer(up_out)?;
        let snake = graph.nodes.get(snake_idx)?;
        if snake.op != StubOpKind::Snake {
            return None;
        }
        let snake_out = *snake.outputs.first()?;
        let add_idx = graph.unique_consumer(snake_out)?;
        let add = graph.nodes.get(add_idx)?;
        if add.op != StubOpKind::Add {
            return None;
        }
        // Add(residual) must have exactly 2 inputs — one from Snake,
        // one from outside the fused chain. If the residual leg is
        // missing (single-input Add) or extra legs exist (>2 inputs)
        // the AMP shape does not match.
        if add.inputs.len() != 2 || !add.inputs.contains(&snake_out) {
            return None;
        }
        debug_assert!(
            root_node != snake_idx && root_node != add_idx && snake_idx != add_idx,
            "upsample+snake+add pattern selected overlapping indices — graph invariants broken"
        );
        Some((root_node, snake_idx, add_idx))
    }
}

// ---------------------------------------------------------------------------
// Public marker types — always present, even when the stub feature is off.
// These give the followup PR a stable place to hang the real `impl
// FusionPattern` blocks when Snake / Conv1d / Upsample land in `OpKind`
// (co-delivered with the consumer model). Until then they carry no
// executable code.
// ---------------------------------------------------------------------------

/// Fusion pattern marker: `Conv1d → Snake` (M2-04-T09).
///
/// Registration in [`super::super::pattern_registry`](crate::ir::fusion)
/// is intentionally deferred: `OpKind` has no `Conv1d` / `Snake` variants
/// today. The active matcher for this pattern lives behind the
/// `fusion-snake-stub` cargo feature (default OFF) so the default build
/// does not depend on placeholder ops. See the module documentation for
/// the FR-OP-13 / FR-OP-11 / FR-OP-14 co-delivery rationale and the four
/// invariants preserved through fusion.
#[derive(Debug, Default, Clone, Copy)]
pub struct Conv1dSnakePattern;

impl Conv1dSnakePattern {
    /// Construct a fresh instance (the pattern is stateless).
    ///
    /// Marked `#[allow(dead_code)]` because the `impl FusionPattern for
    /// Conv1dSnakePattern` block is deferred to the follow-up PR that
    /// promotes Conv1d / Snake to real ops in
    /// [`OpKind`](crate::ir::graph::OpKind).
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self
    }
}

/// Fusion pattern marker: `Upsample → Snake → Add(residual)` (M2-04-T10).
///
/// This is the BigVGAN Anti-Aliased Multi-Periodicity (AMP) block. Kernel
/// body co-delivered with FR-OP-11 (`bigvgan_generator`) / FR-OP-14
/// (`anti_aliased_upsample`) — see the module documentation for the
/// NVIDIA Source Code License-NC redline (scratch reimplementation
/// obligation, `docs/license-audit.md`).
#[derive(Debug, Default, Clone, Copy)]
pub struct UpsampleSnakeResidualPattern;

impl UpsampleSnakeResidualPattern {
    /// Construct a fresh instance (the pattern is stateless).
    ///
    /// Marked `#[allow(dead_code)]` because the
    /// `impl FusionPattern for UpsampleSnakeResidualPattern` block is
    /// deferred to the follow-up PR that promotes Upsample / Snake to
    /// real ops in [`OpKind`](crate::ir::graph::OpKind).
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// Unit tests — matcher plumbing exercised against synthetic graphs.
// Gated on the same `fusion-snake-stub` feature as the stub op-kind
// enum. Default `cargo test -p vokra-core` compiles the tests out and
// stays green with no Snake ops in tree.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "fusion-snake-stub"))]
mod tests {
    use super::stub::{
        StubGraph, StubOpKind, try_match_conv1d_snake, try_match_upsample_snake_residual,
    };

    // ------------------------------------------------------------------
    // T09: Conv1dSnakePattern
    // ------------------------------------------------------------------

    #[test]
    fn conv1d_snake_chain_matches() {
        // Layout: t0 → Conv1d → t1 → Snake → t2.
        let mut g = StubGraph::new();
        let conv = g.add_node(StubOpKind::Conv1d, &[0], &[1]);
        let snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let g = g.finish(3);

        let (c, s) =
            try_match_conv1d_snake(&g, conv).expect("conv1d+snake chain must match at Conv1d root");
        assert_eq!(c, conv);
        assert_eq!(s, snake);
    }

    #[test]
    fn conv1d_snake_non_conv1d_root_does_not_match() {
        let mut g = StubGraph::new();
        let _conv = g.add_node(StubOpKind::Conv1d, &[0], &[1]);
        let snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let g = g.finish(3);
        // Root anchored at Snake — only Conv1d can anchor.
        assert!(try_match_conv1d_snake(&g, snake).is_none());
    }

    #[test]
    fn conv1d_snake_multi_consumer_conv1d_blocks_fusion() {
        // Conv1d output feeds Snake AND a second sink — the
        // single-consumer edge rule must reject.
        let mut g = StubGraph::new();
        let conv = g.add_node(StubOpKind::Conv1d, &[0], &[1]);
        let _snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        // Second consumer of t1 (Conv1d output).
        let _sink = g.add_node(StubOpKind::Add, &[1, 3], &[4]);
        let g = g.finish(5);

        assert!(
            try_match_conv1d_snake(&g, conv).is_none(),
            "shared Conv1d output must block the fusion"
        );
    }

    #[test]
    fn conv1d_snake_wrong_consumer_op_does_not_match() {
        // Conv1d → Add (not Snake) — pattern must decline.
        let mut g = StubGraph::new();
        let conv = g.add_node(StubOpKind::Conv1d, &[0], &[1]);
        let _add = g.add_node(StubOpKind::Add, &[1, 2], &[3]);
        let g = g.finish(4);
        assert!(try_match_conv1d_snake(&g, conv).is_none());
    }

    // ------------------------------------------------------------------
    // T10: UpsampleSnakeResidualPattern
    // ------------------------------------------------------------------

    #[test]
    fn upsample_snake_residual_chain_matches() {
        // Layout: t0 → Upsample → t1 → Snake → t2, t3 → Add(t2, t3) → t4.
        // (t3 is the residual — comes from outside the fused chain.)
        let mut g = StubGraph::new();
        let up = g.add_node(StubOpKind::Upsample, &[0], &[1]);
        let snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let add = g.add_node(StubOpKind::Add, &[2, 3], &[4]);
        let g = g.finish(5);

        let (u, s, a) = try_match_upsample_snake_residual(&g, up)
            .expect("upsample+snake+add chain must match at Upsample root");
        assert_eq!(u, up);
        assert_eq!(s, snake);
        assert_eq!(a, add);
    }

    #[test]
    fn upsample_snake_residual_non_upsample_root_does_not_match() {
        let mut g = StubGraph::new();
        let _up = g.add_node(StubOpKind::Upsample, &[0], &[1]);
        let snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let _add = g.add_node(StubOpKind::Add, &[2, 3], &[4]);
        let g = g.finish(5);
        // Root anchored at Snake — only Upsample can anchor.
        assert!(try_match_upsample_snake_residual(&g, snake).is_none());
    }

    #[test]
    fn upsample_snake_residual_missing_residual_leg_does_not_match() {
        // Add with a single input (self-loop `[t2]`) — the AMP shape
        // requires a two-input Add carrying an external residual.
        let mut g = StubGraph::new();
        let up = g.add_node(StubOpKind::Upsample, &[0], &[1]);
        let _snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let _add = g.add_node(StubOpKind::Add, &[2], &[3]);
        let g = g.finish(4);
        assert!(try_match_upsample_snake_residual(&g, up).is_none());
    }

    #[test]
    fn upsample_snake_residual_multi_consumer_snake_blocks_fusion() {
        // Snake output feeds Add AND a second sink — reject.
        let mut g = StubGraph::new();
        let up = g.add_node(StubOpKind::Upsample, &[0], &[1]);
        let _snake = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let _add = g.add_node(StubOpKind::Add, &[2, 3], &[4]);
        // Second consumer of t2 (Snake output).
        let _sink = g.add_node(StubOpKind::Add, &[2, 5], &[6]);
        let g = g.finish(7);

        assert!(
            try_match_upsample_snake_residual(&g, up).is_none(),
            "shared Snake output must block the fusion"
        );
    }

    #[test]
    fn upsample_snake_residual_wrong_tail_op_does_not_match() {
        // Upsample → Snake → Snake (wrong tail) — reject.
        let mut g = StubGraph::new();
        let up = g.add_node(StubOpKind::Upsample, &[0], &[1]);
        let _snake1 = g.add_node(StubOpKind::Snake, &[1], &[2]);
        let _snake2 = g.add_node(StubOpKind::Snake, &[2], &[3]);
        let g = g.finish(4);
        assert!(try_match_upsample_snake_residual(&g, up).is_none());
    }
}
