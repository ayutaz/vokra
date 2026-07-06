//! Vokra graph fusion pass (M2-04) — architectural decision record.
//!
//! # Two-face design (IR pass + imperative frontend, one kernel body)
//!
//! Vokra ships two callers into every fused op, but only **one** kernel
//! implementation:
//!
//! - **Face A — IR pass.** `fuse` (added in M2-04-T03) walks an
//!   [`AudioGraph`](super::graph::AudioGraph) in
//!   [`topo_order`](super::graph::AudioGraph::topo_order) order, matches
//!   fusable op chains (e.g. `Stft → power → MelFilterbank → log10` for
//!   the future `FusedOp::LogMel`), and rewrites them into a single
//!   `OpKind::Fused(FusedOp)` node. This is the graph-level path used by
//!   any future `AudioGraph`-driven model (M2-07 Kokoro, post-M2 IR-native
//!   Whisper). The rewrite is invoked through a crate-internal mutator
//!   on [`AudioGraph`](super::graph::AudioGraph) (`rewrite_with`,
//!   visibility narrowed to `pub(in crate::ir)`); the public `AudioGraph`
//!   API stays immutable-by-construction —
//!   [`GraphBuilder`](super::graph::GraphBuilder) remains the only
//!   externally visible producer of graphs.
//!
//! - **Face B — Imperative frontend.** The Whisper log-mel front-end
//!   (`crates/vokra-models/src/whisper/mel.rs::log_mel`) calls the fused
//!   kernel directly via a runtime toggle (default on;
//!   `VOKRA_DISABLE_FUSION=1` forces the unfused reference path). Today
//!   no `AudioGraph` in-tree contains the log-mel chain — the imperative
//!   frontend is where the RTF win is realised. The fallback unfused
//!   path is preserved bit-identically so it stays the reference oracle
//!   for parity (`crates/vokra-ops/tests/fused_logmel_parity.rs`).
//!
//! Both faces call the exact same body (`vokra_ops::fused_log_mel_*` for
//! LogMel; `vokra_backend_cpu::kernels::fused_logmel_{avx2,neon}` for
//! SIMD). This is the same one-implementation-two-callers pattern that
//! `crate::prenorm` and the Metal `encode_prenorm_stack` already ship —
//! zero risk of the "test/prod divergence" observed when fused kernels
//! shipped only their own per-op reference.
//!
//! # FR-EX-08 de-fusion rule (no silent cross-backend fallback)
//!
//! The pass takes an optional
//! `&dyn `[`Backend`](crate::runtime::Backend). Before emitting a
//! `FusedOp` node the matcher queries `backend.supports(...)`; if the
//! backend cannot execute the fused variant, the pass **leaves the base
//! ops in place**. Base ops (`Stft`, `MelFilterbank`, `Mul`, …) have full
//! CPU coverage today, so this is not a silent fallback — the base ops
//! are the primary implementation and fusion is an optimisation overlay.
//! If a `FusedOp` reaches an unsupported backend (e.g. a hand-built
//! test), `Backend::eval_op` returns
//! [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) — the
//! same explicit-error contract already proven at
//! `runtime::run_graph` and `vokra-backend-cpu::CpuBackend`
//! (M2-01/M2-03 precedent). Silent host↔device fallback is redlined by
//! FR-EX-08.
//!
//! # Snake / BigVGAN co-deliver gating (FR-OP-13 / FR-OP-11 / FR-OP-14)
//!
//! M2 has no Snake activation consumer: Kokoro-82M (M2-07) uses the
//! iSTFTNet head (StyleTTS 2 derivative), piper-plus uses LeakyReLU +
//! MRF, Whisper / CAM++ / Silero VAD never touch Snake. Snake kernel
//! bodies (`snake_activation` FR-OP-13, `bigvgan_generator` FR-OP-11,
//! `anti_aliased_upsample` FR-OP-14) are model-synced deliverables per
//! SRS §2.3 "モデル同期" — they land with the first consumer model, not
//! ahead of it. This module therefore ships only the *pattern shape*
//! (M2-04-T09/T10, `patterns::snake`) behind a `fusion-snake-stub` cfg —
//! enough to unit-test matcher plumbing without adding orphan ops to
//! [`OpKind`](super::graph::OpKind). The Snake `internal_precision`
//! default stays FP32 (BF16 mantissa loss per FR-OP-13); Vocos / BigVGAN
//! honour their fp16 minimum dtype (INT8 崩壊 per FR-QT-03); BigVGAN
//! kernel bodies remain a scratch reimplementation obligation (NVIDIA
//! Source Code License-NC redline, `docs/license-audit.md`). Full
//! kernels + e2e parity are tracked as an M2-04-T13 followup issue.
//!
//! # Invariants
//!
//! - **Zero-dep (NFR-DS-02).** Fusion adds no crate dependency; the pass
//!   lives in the already-in-workspace `vokra-core`, kernels in
//!   `vokra-ops` / `vokra-backend-cpu`. `./scripts/check-zero-deps.sh`
//!   stays green — root `Cargo.lock` continues to contain only
//!   `vokra-*`.
//! - **Unsafe boundary (NFR-RL-07).** SIMD-only `unsafe` lives inside
//!   `#[target_feature(...)]` kernels in `vokra-backend-cpu`, each with
//!   a `// SAFETY:` comment naming the ISA precondition. Public
//!   `vokra-ops::fused_log_mel_*` and this module's future `fuse`
//!   entrypoint are safe. `vokra-core` itself keeps its unsafe-0
//!   invariant.
//! - **Bit-identical when disabled.** With the toggle off, the
//!   imperative frontend must be byte-equal to its pre-fusion baseline;
//!   the IR pass is a no-op returning `Ok(0)` fusions applied.

use std::collections::{HashMap, HashSet};

use super::graph::{AudioGraph, MelAttrs, Node, OpKind, StftAttrs};
use super::tensor::TensorId;
use crate::backend::Backend;
use crate::error::Result;

pub mod patterns;

use self::patterns::logmel::LogMelPattern;

/// A fused operator payload — the variant behind
/// [`OpKind::Fused`](super::graph::OpKind::Fused).
///
/// M2-04-T02 introduces this enum with the single [`FusedOp::LogMel`]
/// variant driving the Whisper log-mel front-end (STFT → power → Mel →
/// log10) collapse. Additional variants (Snake / BigVGAN AMP) are added
/// when their consumer models land (M2-04-T09/T10 followup).
///
/// The enum is `#[non_exhaustive]` so downstream match sites keep
/// compiling as new fused ops are added; backends without an
/// implementation for a specific fused op must return
/// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp)
/// (FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FusedOp {
    /// Fused Whisper-style log-mel front-end (M2-04-T04): the
    /// `Stft → power → MelFilterbank → log10 → dynamic-range compress`
    /// chain collapsed into a single node. Both sub-attributes are the
    /// existing IR types so the fused op is byte-equivalent to the
    /// four-op sequence it replaces (NFR-QL-01 FP32 atol=0.01 vs unfused
    /// reference).
    LogMel {
        /// STFT parameters of the front-end stage being fused.
        stft: StftAttrs,
        /// Mel filter-bank parameters of the front-end stage being fused.
        mel: MelAttrs,
    },
}

/// One rewrite emitted by the pattern matcher and applied through the
/// crate-internal
/// [`AudioGraph::rewrite_with`](super::graph::AudioGraph::rewrite_with)
/// mutator (visibility `pub(in crate::ir)`).
///
/// A rewrite removes zero or more existing nodes (by insertion index),
/// inserts one replacement node, and carries a tensor-remap table so any
/// downstream consumer of a fused-out intermediate can be redirected to the
/// fused node's output(s). This is the only structure that carries graph
/// mutations across the `ir::fusion` boundary.
///
/// The rewrite is `pub(crate)` — external callers never construct these;
/// they're emitted by [`FusionPattern`] implementations inside `ir::fusion`
/// and consumed by the [`fuse`] driver in this same module.
#[derive(Debug, Clone)]
pub(crate) struct FusionRewrite {
    /// Indices into
    /// [`AudioGraph::nodes`](super::graph::AudioGraph::nodes) of nodes to
    /// remove. Callers batch all rewrites before application so these
    /// indices stay meaningful.
    pub removed_nodes: Vec<usize>,
    /// The single node inserted in place of the removed ones.
    pub inserted_node: Node,
    /// Remap from each fused-out intermediate [`TensorId`] to the fused
    /// node's replacement id (empty when the rewrite consumes only graph
    /// inputs and produces only graph outputs, i.e. no downstream consumer
    /// to redirect). The [`fuse`] driver uses this to preserve external
    /// consumers of tensors the pattern would otherwise drop.
    pub tensor_remap: HashMap<TensorId, TensorId>,
}

/// A registered fusion pattern (M2-04-T03 scaffolding, extended per-pattern).
///
/// Each implementation attempts to match a fusable sub-chain starting at
/// `root_node` (an index into
/// [`AudioGraph::nodes`](super::graph::AudioGraph::nodes)) in
/// [`topo_order`](super::graph::AudioGraph::topo_order) order. On a
/// match, `try_match` returns a [`FusionRewrite`] describing the exact
/// nodes to remove and the single [`Node`] to insert. The pass driver
/// (`fuse`) then collects rewrites and applies them through the crate-
/// internal
/// [`AudioGraph::apply_fusion_rewrites`](super::graph::AudioGraph::apply_fusion_rewrites)
/// mutator.
///
/// **Single-consumer edges only.** Intermediate tensors on the fused
/// chain must have exactly one downstream reader; a shared intermediate
/// blocks fusion (any other node reading it would break after the
/// rewrite). Implementations enforce this locally against the graph
/// they inspect.
///
/// **FR-EX-08 de-fusion.** The `fuse` driver is expected to check
/// backend support before accepting a returned rewrite; individual
/// patterns do not perform backend queries themselves.
pub(crate) trait FusionPattern {
    /// A short static identifier for the pattern (used by logs / tests).
    #[allow(dead_code)] // exercised by future diagnostics / logging
    fn name(&self) -> &'static str;

    /// The [`FusedOp`] variant this pattern would emit — used by the [`fuse`]
    /// driver's FR-EX-08 de-fusion check. The concrete attribute payload is
    /// synthesized cheaply (defaults are enough — the driver only inspects
    /// the variant tag through
    /// [`Backend::supports`](crate::Backend::supports)).
    fn fused_variant_probe(&self) -> FusedOp;

    /// Attempts to match the pattern anchored at `root_node`. Returns
    /// `Some(FusionRewrite)` on success, `None` otherwise.
    fn try_match(&self, graph: &AudioGraph, root_node: usize) -> Option<FusionRewrite>;
}

// ===========================================================================
// FusionOptions (T03) — pass configuration
// ===========================================================================

/// Configuration for the [`fuse`] pass.
///
/// Two knobs today:
///
/// - [`enabled`](Self::enabled) — master toggle. When `false`, [`fuse`] is a
///   no-op returning `Ok(0)` (used by the bench-regression A/B harness to
///   compare fused vs. unfused RTF).
/// - [`backend_name`](Self::backend_name) — advisory label recorded for
///   diagnostics; the actual FR-EX-08 de-fusion decision uses the
///   `&dyn Backend` handed to [`fuse`], not this field.
#[derive(Debug, Clone)]
pub struct FusionOptions {
    /// Master toggle. `true` by default.
    pub enabled: bool,
    /// Advisory backend label (e.g. `"cpu"` / `"metal"` / `"cuda"`) recorded
    /// for diagnostics. `None` means the caller didn't declare one.
    pub backend_name: Option<String>,
}

impl Default for FusionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            backend_name: None,
        }
    }
}

impl FusionOptions {
    /// Fresh options with fusion enabled (the runtime default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables fusion (for the A/B bench harness).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            backend_name: None,
        }
    }

    /// Attaches an advisory backend label for diagnostics.
    pub fn with_backend_name(mut self, name: impl Into<String>) -> Self {
        self.backend_name = Some(name.into());
        self
    }
}

// ===========================================================================
// fuse() entrypoint (T03) — the pass driver
// ===========================================================================

/// Runs the M2-04 fusion pass on `graph` in place.
///
/// Behavior:
///
/// 1. If `opts.enabled == false`, returns `Ok(0)` immediately (byte-identical
///    to skipping the pass — the A/B bench harness relies on this).
/// 2. Walks [`AudioGraph::topo_order`](crate::AudioGraph::topo_order) and,
///    for each pattern in the registry, tries to match a rewrite rooted at
///    each candidate node.
/// 3. For every candidate match, checks the **single-consumer edge rule** —
///    if any of the pattern's intermediate tensors is read by a node outside
///    the pattern (or wired as a graph output), the rewrite is skipped
///    (observability preservation).
/// 4. **FR-EX-08 de-fusion**: when `backend` is provided and does *not*
///    support the fused variant this rewrite would emit
///    ([`Backend::supports`](crate::Backend::supports) returns `false` for
///    `OpKind::Fused(pattern.fused_variant_probe())`), the rewrite is
///    skipped — base ops stay in place. This is *not* a silent fallback: the
///    base ops are the primary implementation, fusion is a pure optimization
///    overlay.
/// 5. Applies collected rewrites via the crate-internal
///    [`AudioGraph::rewrite_with`](super::graph::AudioGraph::rewrite_with)
///    and returns the count.
///
/// The `backend` argument is optional (`None` means "assume all fused
/// variants are supported") so unit tests and offline tools can exercise
/// the pass without a concrete backend.
///
/// # Errors
///
/// Propagates a
/// [`VokraError::GraphValidation`](crate::VokraError::GraphValidation) from
/// [`topo_order`](crate::AudioGraph::topo_order) if the graph contains a
/// cycle. Pattern matching failures are silent skips, not errors.
pub fn fuse(
    graph: &mut AudioGraph,
    opts: &FusionOptions,
    backend: Option<&dyn Backend>,
) -> Result<usize> {
    if !opts.enabled {
        return Ok(0);
    }

    let order = graph.topo_order()?;

    // Producer / consumer table over the current node set. `producer[t]` is
    // the (single) node writing tensor `t`; `consumer_count[t]` counts how
    // many *reads* — both from other nodes' inputs AND from the graph's own
    // output list, because a graph output is a live observer that fusion
    // must not silently drop.
    let n_tensors = graph.tensors().len();
    let n_nodes = graph.nodes().len();
    let mut producer: Vec<Option<usize>> = vec![None; n_tensors];
    let mut consumer_count: Vec<usize> = vec![0; n_tensors];
    for (i, node) in graph.nodes().iter().enumerate() {
        for out in node.outputs() {
            producer[out.0] = Some(i);
        }
        for inp in node.inputs() {
            consumer_count[inp.0] += 1;
        }
    }
    for out in graph.outputs() {
        if out.0 < consumer_count.len() {
            consumer_count[out.0] += 1;
        }
    }

    let registry = pattern_registry();
    let mut rewrites: Vec<FusionRewrite> = Vec::new();
    // Track nodes already claimed by a pending rewrite so overlapping matches
    // (rare for the LogMel pattern, but general) do not stomp each other.
    let mut claimed: HashSet<usize> = HashSet::new();

    for &root in &order {
        if claimed.contains(&root) {
            continue;
        }
        for pat in &registry {
            let Some(rw) = pat.try_match(graph, root) else {
                continue;
            };
            if rw.removed_nodes.iter().any(|n| claimed.contains(n)) {
                continue;
            }

            // Single-consumer edge check: for every tensor produced by a
            // `removed_nodes` member that is NOT preserved by the rewrite
            // (either as one of the fused node's own outputs — same
            // `TensorId`, so existing consumers stay valid — or via
            // `tensor_remap`), the tensor must be consumed only by pattern
            // nodes. Otherwise an external reader (a downstream node or a
            // graph output) is observing the intermediate and fusion would
            // silently drop it → reject. Tensors preserved on the fused
            // node's output list are OK because their `TensorId` continues
            // to be produced (by the fused replacement) after the rewrite.
            let fused_outputs: HashSet<TensorId> =
                rw.inserted_node.outputs().iter().copied().collect();
            let mut ok = true;
            for &removed_idx in &rw.removed_nodes {
                let node = &graph.nodes()[removed_idx];
                for out in node.outputs() {
                    if fused_outputs.contains(out) || rw.tensor_remap.contains_key(out) {
                        // Preserved by the rewrite — external consumers stay
                        // valid post-fusion.
                        continue;
                    }
                    let inside_pattern_reads: usize = rw
                        .removed_nodes
                        .iter()
                        .map(|&r| {
                            graph.nodes()[r]
                                .inputs()
                                .iter()
                                .filter(|t| **t == *out)
                                .count()
                        })
                        .sum();
                    let external_readers =
                        consumer_count[out.0].saturating_sub(inside_pattern_reads);
                    if external_readers != 0 {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                continue;
            }

            // FR-EX-08 de-fusion: if the backend does not support this fused
            // variant, leave the base ops in place. Backends must answer
            // `false` for unknown ops (see [`Backend`] docs), so this
            // correctly refuses to emit a `FusedOp` a backend cannot run.
            if let Some(b) = backend {
                let fused_probe = OpKind::Fused(pat.fused_variant_probe());
                if !b.supports(&fused_probe) {
                    continue;
                }
            }

            for &n in &rw.removed_nodes {
                claimed.insert(n);
            }
            rewrites.push(rw);
            break; // one rewrite per root
        }
    }

    let applied = rewrites.len();
    graph.rewrite_with(rewrites);
    debug_assert!(applied <= n_nodes);
    let _ = producer; // retained for future patterns that need producer lookup
    Ok(applied)
}

// ===========================================================================
// Pattern registry (T04) — dispatch to LogMelPattern
// ===========================================================================

/// Builds the pattern registry.
///
/// Today the only registered pattern is [`LogMelPattern`] (M2-04-T04).
/// Future tickets (T09 Snake, T10 BigVGAN AMP) append their patterns here.
/// The order is significant only when two patterns match at the same root —
/// the first match wins.
fn pattern_registry() -> Vec<Box<dyn FusionPattern>> {
    vec![Box::new(LogMelPattern::new())]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::error::VokraError;
    use crate::ir::graph::{GraphBuilder, MelAttrs, OpKind, StftAttrs};
    use crate::ir::tensor::{DType, TensorDesc};
    use crate::runtime::Tensor;

    fn stft_attrs() -> StftAttrs {
        StftAttrs::new(400, 160)
    }
    fn mel_attrs() -> MelAttrs {
        MelAttrs::new(16_000, 400, 80)
    }

    /// A backend that supports the log-mel base ops but *not* the fused
    /// variant — used to prove FR-EX-08 de-fusion leaves base ops in place.
    struct BaseOnlyBackend;
    impl Backend for BaseOnlyBackend {
        fn name(&self) -> &str {
            "base-only"
        }
        fn supports(&self, op: &OpKind) -> bool {
            !matches!(op, OpKind::Fused(_))
        }
        fn execute(&self, _g: &AudioGraph) -> Result<()> {
            Err(VokraError::NotImplemented("base-only never executes"))
        }
        fn eval_op(&self, _op: &OpKind, _inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            Err(VokraError::NotImplemented("base-only never computes"))
        }
    }

    /// A backend that supports the fused variant (registry-dispatch test).
    struct FusedOkBackend;
    impl Backend for FusedOkBackend {
        fn name(&self) -> &str {
            "fused-ok"
        }
        fn supports(&self, _op: &OpKind) -> bool {
            true
        }
        fn execute(&self, _g: &AudioGraph) -> Result<()> {
            Err(VokraError::NotImplemented("fused-ok never executes"))
        }
        fn eval_op(&self, _op: &OpKind, _inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            Err(VokraError::NotImplemented("fused-ok never computes"))
        }
    }

    /// Builds a synthetic `Stft → Mul (power proxy) → MelFilterbank → Mul
    /// (log proxy)` chain. `Log10` is not (yet) an IR op — the LogMel
    /// pattern tolerates a `Mul` in the log slot; the actual log10 is
    /// applied inside the fused kernel body (see M2-04-T05, out of this
    /// file's scope). The pattern matches on the shape
    /// `Stft → * → MelFilterbank → *`.
    fn build_logmel_chain(with_extra_consumer: bool) -> AudioGraph {
        let mut b = GraphBuilder::new();
        let pcm = b.add_tensor(TensorDesc::new("pcm", DType::F32, [16000]));
        let spec = b.add_tensor(TensorDesc::new("spec", DType::F32, [100, 201]));
        let power = b.add_tensor(TensorDesc::new("power", DType::F32, [100, 201]));
        let mel = b.add_tensor(TensorDesc::new("mel", DType::F32, [100, 80]));
        let logmel = b.add_tensor(TensorDesc::new("logmel", DType::F32, [80, 100]));

        b.add_node(OpKind::Stft(stft_attrs()), &[pcm], &[spec]);
        // Mul as power-op proxy (spec * spec ≈ magnitude²).
        b.add_node(OpKind::Mul, &[spec, spec], &[power]);
        b.add_node(OpKind::MelFilterbank(mel_attrs()), &[power], &[mel]);
        // Softmax as log-op proxy (see patterns/logmel.rs doc: the log10
        // is applied inside the fused kernel body; the shape it matches is
        // Stft → Mul → MelFilterbank → Softmax).
        b.add_node(OpKind::Softmax, &[mel], &[logmel]);

        b.mark_input(pcm);
        b.mark_output(logmel);
        if with_extra_consumer {
            // Wire `power` as an ADDITIONAL graph output — this simulates a
            // downstream caller observing the intermediate; the
            // single-consumer edge rule must reject the fusion.
            b.mark_output(power);
        }
        b.finish().expect("valid graph")
    }

    // ------------------------------------------------------------------
    // T02: builds_graph_with_fused_logmel_and_validates
    // ------------------------------------------------------------------

    #[test]
    fn builds_graph_with_fused_logmel_and_validates() {
        // A graph containing a hand-authored `OpKind::Fused(FusedOp::LogMel)`
        // node must pass structural validation — proves the new variant is
        // wired into `AudioGraph::validate()` and reachable from
        // `GraphBuilder`.
        let mut b = GraphBuilder::new();
        let pcm = b.add_tensor(TensorDesc::new("pcm", DType::F32, [16000]));
        let out = b.add_tensor(TensorDesc::new("logmel", DType::F32, [80, 100]));
        b.add_node(
            OpKind::Fused(FusedOp::LogMel {
                stft: stft_attrs(),
                mel: mel_attrs(),
            }),
            &[pcm],
            &[out],
        );
        b.mark_input(pcm);
        b.mark_output(out);

        let graph = b.finish().expect("valid graph");
        assert_eq!(graph.nodes().len(), 1);
        match graph.nodes()[0].op() {
            OpKind::Fused(FusedOp::LogMel { stft, mel }) => {
                assert_eq!(stft.n_fft, 400);
                assert_eq!(mel.n_mels, 80);
            }
            other => panic!("expected Fused(LogMel), got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // T03: toggle_off_is_bit_identical_to_no_fuse
    // ------------------------------------------------------------------

    #[test]
    fn toggle_off_is_bit_identical_to_no_fuse() {
        let mut graph = build_logmel_chain(false);
        let before = graph.nodes().len();
        let n = fuse(&mut graph, &FusionOptions::disabled(), None).unwrap();
        assert_eq!(n, 0, "disabled pass must apply zero rewrites");
        assert_eq!(
            graph.nodes().len(),
            before,
            "disabled pass must be byte-identical to no-op"
        );
    }

    // ------------------------------------------------------------------
    // T03: multi_consumer_intermediate_blocks_fusion
    // ------------------------------------------------------------------

    #[test]
    fn multi_consumer_intermediate_blocks_fusion() {
        // With `power` also wired as a graph output, the single-consumer
        // edge rule must reject fusion (an external observer reads the
        // intermediate).
        let mut graph = build_logmel_chain(true);
        let before = graph.nodes().len();
        let opts = FusionOptions::new();
        let n = fuse(&mut graph, &opts, None).unwrap();
        assert_eq!(
            n, 0,
            "external consumer on `power` must block fusion (single-consumer rule)"
        );
        assert_eq!(graph.nodes().len(), before);
    }

    // ------------------------------------------------------------------
    // T03: unsupported_backend_leaves_base_ops
    // ------------------------------------------------------------------

    #[test]
    fn unsupported_backend_leaves_base_ops() {
        let mut graph = build_logmel_chain(false);
        let before = graph.nodes().len();
        let opts = FusionOptions::new();
        let n = fuse(&mut graph, &opts, Some(&BaseOnlyBackend)).unwrap();
        assert_eq!(
            n, 0,
            "backend without Fused support must trigger de-fusion (base ops in place)"
        );
        assert_eq!(graph.nodes().len(), before);
        assert!(
            !graph
                .nodes()
                .iter()
                .any(|n| matches!(n.op(), OpKind::Fused(_))),
            "no Fused node may have been emitted"
        );
    }

    // ------------------------------------------------------------------
    // T04: matcher_registry_dispatches_by_pattern
    // ------------------------------------------------------------------

    #[test]
    fn matcher_registry_dispatches_by_pattern() {
        // With a permissive backend, the LogMel pattern must be dispatched
        // by the registry and produce exactly one rewrite that emits a
        // `FusedOp::LogMel` node.
        let mut graph = build_logmel_chain(false);
        let before = graph.nodes().len();
        let opts = FusionOptions::new().with_backend_name("test");
        let n = fuse(&mut graph, &opts, Some(&FusedOkBackend)).unwrap();
        assert_eq!(n, 1, "registry must dispatch LogMelPattern and match once");
        assert_eq!(
            graph.nodes().len(),
            before - 3,
            "4 base nodes → 1 fused node (net −3)"
        );
        assert!(
            graph
                .nodes()
                .iter()
                .any(|n| matches!(n.op(), OpKind::Fused(FusedOp::LogMel { .. }))),
            "graph must now contain a Fused(LogMel) node"
        );
    }
}
