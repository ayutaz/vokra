//! Data-carrying graph evaluator (Phase 1 of the GPU execution architecture).
//!
//! The [`AudioGraph`](crate::AudioGraph) IR is a *descriptor*: its tensors
//! carry shapes but no data, and its [`Backend`](crate::Backend) `execute`
//! entry point only validates op coverage. This module adds the piece that was
//! deferred there — an engine that actually runs a graph, threading real
//! [`Tensor`] values from node to node.
//!
//! # Contract (permanent constraints)
//!
//! - **One graph = one backend, no silent fallback** (FR-EX-08). Before
//!   evaluating anything, [`run_graph`] checks that the backend supports
//!   *every* op in the graph; a single unsupported op is an explicit
//!   [`VokraError::UnsupportedOp`]. There is no per-op CPU fallback and no
//!   ONNX-Runtime-style execution-provider graph partitioning.
//! - **Deterministic schedule.** Nodes run in the topological order returned by
//!   [`AudioGraph::topo_order`](crate::AudioGraph::topo_order) (Kahn's
//!   algorithm, index-stable for independent nodes), so a graph evaluates the
//!   same way every run.
//! - **Validation lives in the engine.** A backend's
//!   [`eval_op`](crate::Backend::eval_op) only computes; [`run_graph`] checks
//!   its output arity and shapes against the declared
//!   [`TensorDesc`](crate::TensorDesc)s.
//!
//! The module is named `runtime` (not `engine`) to avoid confusion with
//! [`engines`](crate::engines), which holds the task-level `AsrEngine` /
//! `TtsEngine` / `VadEngine` traits.

pub mod tensor;

pub use tensor::Tensor;

use crate::backend::Backend;
use crate::error::{Result, VokraError};
use crate::ir::{AudioGraph, Dim, OpKind, TensorDesc, TensorId};

/// Evaluates `graph` on `backend`, returning the values of its declared output
/// tensors in declaration order.
///
/// `inputs` supplies a value for every *leaf* tensor of the graph — the graph
/// inputs plus any constants / weights that no node produces. Intermediate
/// tensors are computed and held internally.
///
/// # Errors
///
/// - [`VokraError::UnsupportedOp`] if the backend does not support some op in
///   the graph (checked up front, before any evaluation — FR-EX-08).
/// - [`VokraError::GraphValidation`] if the graph contains a cycle, or a node
///   reads a tensor that is neither supplied in `inputs` nor produced by an
///   earlier node, or a declared output was never produced.
/// - [`VokraError::InvalidArgument`] if an `inputs` entry names an
///   out-of-range tensor id, or a node's `eval_op` returns the wrong number of
///   outputs or an output whose shape contradicts its declared descriptor.
/// - whatever a backend's [`eval_op`](crate::Backend::eval_op) itself returns.
///
/// # Examples
///
/// ```
/// use vokra_core::{
///     backend::Backend, run_graph, DType, GraphBuilder, OpKind, Result, Tensor,
///     TensorDesc, VokraError,
/// };
///
/// // A tiny backend that only knows element-wise Add.
/// struct AddBackend;
/// impl Backend for AddBackend {
///     fn name(&self) -> &str { "add-only" }
///     fn supports(&self, op: &OpKind) -> bool { matches!(op, OpKind::Add) }
///     fn execute(&self, _g: &vokra_core::AudioGraph) -> Result<()> {
///         Err(VokraError::NotImplemented("coverage stub"))
///     }
///     fn eval_op(&self, _op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
///         let a = inputs[0].as_f32()?;
///         let b = inputs[1].as_f32()?;
///         let out = a.iter().zip(b).map(|(x, y)| x + y).collect();
///         Ok(vec![Tensor::host_f32(inputs[0].shape.clone(), out)?])
///     }
/// }
///
/// let mut g = GraphBuilder::new();
/// let a = g.add_tensor(TensorDesc::new("a", DType::F32, [3]));
/// let b = g.add_tensor(TensorDesc::new("b", DType::F32, [3]));
/// let y = g.add_tensor(TensorDesc::new("y", DType::F32, [3]));
/// g.add_node(OpKind::Add, &[a, b], &[y]);
/// g.mark_output(y);
/// let graph = g.finish()?;
///
/// let outs = run_graph(
///     &AddBackend,
///     &graph,
///     &[
///         (a, Tensor::host_f32(vec![3], vec![1.0, 2.0, 3.0])?),
///         (b, Tensor::host_f32(vec![3], vec![10.0, 20.0, 30.0])?),
///     ],
/// )?;
/// assert_eq!(outs[0].as_f32()?, &[11.0, 22.0, 33.0]);
/// # Ok::<(), VokraError>(())
/// ```
pub fn run_graph(
    backend: &dyn Backend,
    graph: &AudioGraph,
    inputs: &[(TensorId, Tensor)],
) -> Result<Vec<Tensor>> {
    // 1) Whole-graph coverage precheck (FR-EX-08): a single unsupported op is
    //    an explicit error, before any evaluation. No per-op fallback.
    for node in graph.nodes() {
        if !backend.supports(node.op()) {
            return Err(VokraError::UnsupportedOp(format!(
                "{} backend has no kernel for {:?}",
                backend.name(),
                node.op()
            )));
        }
    }

    // 2) Slot table keyed by tensor-table position; seed the supplied leaves.
    let n_tensors = graph.tensors().len();
    let mut env: Vec<Option<Tensor>> = vec![None; n_tensors];
    for (id, value) in inputs {
        let slot = env.get_mut(id.index()).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "run_graph input references tensor id {} but the graph has only {} tensors",
                id.index(),
                n_tensors
            ))
        })?;
        *slot = Some(value.clone());
    }

    // 3) Evaluate node-by-node in topological order (also rejects cycles).
    for ni in graph.topo_order()? {
        let node = &graph.nodes()[ni];

        // Gather resolved inputs; a missing value is a structural error.
        let mut ins: Vec<&Tensor> = Vec::with_capacity(node.inputs().len());
        for id in node.inputs() {
            let value = env
                .get(id.index())
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    VokraError::GraphValidation(format!(
                        "node #{ni} ({:?}) reads tensor id {} which is neither a supplied input \
                     nor produced by an earlier node",
                        node.op(),
                        id.index()
                    ))
                })?;
            ins.push(value);
        }

        let outs = backend.eval_op(node.op(), &ins)?;

        // 4) Validate arity + shape against the declared descriptors, then bind.
        if outs.len() != node.outputs().len() {
            return Err(VokraError::InvalidArgument(format!(
                "node #{ni} ({:?}) produced {} output(s) but the graph declares {}",
                node.op(),
                outs.len(),
                node.outputs().len()
            )));
        }
        for (id, value) in node.outputs().iter().zip(outs) {
            if let Some(desc) = graph.tensor(*id) {
                check_output_shape(desc, &value, ni, node.op())?;
            }
            env[id.index()] = Some(value);
        }
    }

    // 5) Collect declared outputs in order.
    graph
        .outputs()
        .iter()
        .map(|id| {
            env.get(id.index()).cloned().flatten().ok_or_else(|| {
                VokraError::GraphValidation(format!(
                    "graph output tensor id {} was never produced",
                    id.index()
                ))
            })
        })
        .collect()
}

/// Checks a produced value's shape against its declared descriptor: the rank
/// must match and every statically-fixed axis must equal the produced extent
/// ([`Dim::Dynamic`] axes accept any extent).
fn check_output_shape(desc: &TensorDesc, value: &Tensor, ni: usize, op: &OpKind) -> Result<()> {
    if desc.shape.len() != value.shape.len() {
        return Err(VokraError::InvalidArgument(format!(
            "node #{ni} ({op:?}) output `{}`: produced rank {} does not match declared rank {}",
            desc.name,
            value.shape.len(),
            desc.shape.len()
        )));
    }
    for (axis, (declared, &got)) in desc.shape.iter().zip(&value.shape).enumerate() {
        if let Dim::Fixed(want) = declared {
            if *want != got {
                return Err(VokraError::InvalidArgument(format!(
                    "node #{ni} ({op:?}) output `{}` axis {axis}: produced extent {got} does \
                     not match declared {want}",
                    desc.name
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::StftAttrs;
    use crate::{AudioGraph, DType, GraphBuilder, TensorDesc};

    /// Test backend covering element-wise `Add` / `Mul` (both 2-input, shape
    /// preserving). Enough to exercise the engine mechanics purely in core,
    /// without depending on `vokra-backend-cpu`.
    struct ElementwiseBackend;

    impl Backend for ElementwiseBackend {
        fn name(&self) -> &str {
            "elementwise-test"
        }
        fn supports(&self, op: &OpKind) -> bool {
            matches!(op, OpKind::Add | OpKind::Mul)
        }
        fn execute(&self, _graph: &AudioGraph) -> Result<()> {
            Err(VokraError::NotImplemented("coverage stub"))
        }
        fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            let a = inputs[0].as_f32()?;
            let b = inputs[1].as_f32()?;
            let out: Vec<f32> = match op {
                OpKind::Add => a.iter().zip(b).map(|(x, y)| x + y).collect(),
                OpKind::Mul => a.iter().zip(b).map(|(x, y)| x * y).collect(),
                other => {
                    return Err(VokraError::UnsupportedOp(format!("{other:?}")));
                }
            };
            Ok(vec![Tensor::host_f32(inputs[0].shape.clone(), out)?])
        }
    }

    fn t3(data: [f32; 3]) -> Tensor {
        Tensor::host_f32(vec![3], data.to_vec()).unwrap()
    }

    #[test]
    fn evaluates_multi_node_graph_in_dependency_order() {
        // (a + b) then result * c. Nodes are inserted with the MUL first so the
        // engine cannot rely on insertion order — it must schedule ADD first.
        let mut g = GraphBuilder::new();
        let a = g.add_tensor(TensorDesc::new("a", DType::F32, [3]));
        let b = g.add_tensor(TensorDesc::new("b", DType::F32, [3]));
        let c = g.add_tensor(TensorDesc::new("c", DType::F32, [3]));
        let sum = g.add_tensor(TensorDesc::new("sum", DType::F32, [3]));
        let out = g.add_tensor(TensorDesc::new("out", DType::F32, [3]));
        g.add_node(OpKind::Mul, &[sum, c], &[out]); // inserted first, must run second
        g.add_node(OpKind::Add, &[a, b], &[sum]); // inserted second, must run first
        g.mark_output(out);
        let graph = g.finish().unwrap();

        let outs = run_graph(
            &ElementwiseBackend,
            &graph,
            &[
                (a, t3([1.0, 2.0, 3.0])),
                (b, t3([10.0, 20.0, 30.0])),
                (c, t3([2.0, 2.0, 2.0])),
            ],
        )
        .unwrap();
        assert_eq!(outs.len(), 1);
        // (a+b)*c = [22, 44, 66].
        assert_eq!(outs[0].as_f32().unwrap(), &[22.0, 44.0, 66.0]);
    }

    #[test]
    fn unsupported_op_is_explicit_before_any_eval() {
        // Softmax is not covered by ElementwiseBackend → UnsupportedOp (V5).
        let mut g = GraphBuilder::new();
        let x = g.add_tensor(TensorDesc::new("x", DType::F32, [3]));
        let y = g.add_tensor(TensorDesc::new("y", DType::F32, [3]));
        g.add_node(OpKind::Softmax, &[x], &[y]);
        g.mark_output(y);
        let graph = g.finish().unwrap();

        let err = run_graph(&ElementwiseBackend, &graph, &[(x, t3([1.0, 2.0, 3.0]))]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    #[test]
    fn missing_leaf_input_is_reported() {
        // `b` is a leaf but is not supplied → GraphValidation at gather time.
        let mut g = GraphBuilder::new();
        let a = g.add_tensor(TensorDesc::new("a", DType::F32, [3]));
        let b = g.add_tensor(TensorDesc::new("b", DType::F32, [3]));
        let y = g.add_tensor(TensorDesc::new("y", DType::F32, [3]));
        g.add_node(OpKind::Add, &[a, b], &[y]);
        g.mark_output(y);
        let graph = g.finish().unwrap();

        let err = run_graph(&ElementwiseBackend, &graph, &[(a, t3([1.0, 2.0, 3.0]))]).unwrap_err();
        assert!(matches!(err, VokraError::GraphValidation(_)));
    }

    #[test]
    fn out_of_range_input_id_is_rejected() {
        let mut g = GraphBuilder::new();
        let a = g.add_tensor(TensorDesc::new("a", DType::F32, [3]));
        let b = g.add_tensor(TensorDesc::new("b", DType::F32, [3]));
        let y = g.add_tensor(TensorDesc::new("y", DType::F32, [3]));
        g.add_node(OpKind::Add, &[a, b], &[y]);
        g.mark_output(y);
        let graph = g.finish().unwrap();

        // TensorId(99) is out of range for this 3-tensor graph (the tuple field
        // is crate-visible, so a bogus id can be fabricated here).
        let err = run_graph(
            &ElementwiseBackend,
            &graph,
            &[(TensorId(99), t3([0.0, 0.0, 0.0]))],
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn output_shape_contradiction_is_rejected() {
        // A backend that returns a mis-shaped output; the engine catches it
        // against the declared [3] descriptor.
        struct BadShapeBackend;
        impl Backend for BadShapeBackend {
            fn name(&self) -> &str {
                "bad-shape"
            }
            fn supports(&self, op: &OpKind) -> bool {
                matches!(op, OpKind::Add)
            }
            fn execute(&self, _g: &AudioGraph) -> Result<()> {
                Err(VokraError::NotImplemented("stub"))
            }
            fn eval_op(&self, _op: &OpKind, _inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
                // Declared output is [3]; return [1] instead.
                Ok(vec![Tensor::host_f32(vec![1], vec![0.0]).unwrap()])
            }
        }

        let mut g = GraphBuilder::new();
        let a = g.add_tensor(TensorDesc::new("a", DType::F32, [3]));
        let b = g.add_tensor(TensorDesc::new("b", DType::F32, [3]));
        let y = g.add_tensor(TensorDesc::new("y", DType::F32, [3]));
        g.add_node(OpKind::Add, &[a, b], &[y]);
        g.mark_output(y);
        let graph = g.finish().unwrap();

        let err = run_graph(
            &BadShapeBackend,
            &graph,
            &[(a, t3([0.0, 0.0, 0.0])), (b, t3([0.0, 0.0, 0.0]))],
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn front_end_op_stft_is_unsupported_here() {
        // The engine surfaces an uncovered front-end op (Stft) as an explicit
        // error regardless of the backend's kernels (V5, generic).
        let mut g = GraphBuilder::new();
        let x = g.add_tensor(TensorDesc::new("x", DType::F32, [400]));
        let y = g.add_tensor(TensorDesc::new("y", DType::F32, [3, 201]));
        g.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[x], &[y]);
        g.mark_input(x);
        g.mark_output(y);
        let graph = g.finish().unwrap();

        let err = run_graph(
            &ElementwiseBackend,
            &graph,
            &[(x, Tensor::zeros_f32(vec![400]))],
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
