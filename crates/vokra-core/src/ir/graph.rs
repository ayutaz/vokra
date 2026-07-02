//! Flat op enum and the audio graph descriptor (M0-02-T08/T09).
//!
//! FR-EX-01: "MVP は ggml 型 flat op enum" — the MVP IR is a ggml-style flat
//! op enum; an MLIR audio dialect + StableHLO is re-evaluated in v1.5+.
//!
//! # Design red line (permanent)
//!
//! **ONNX graphs are never loaded at runtime** (FR-LD-05, SRS §5-(2)): ONNX
//! models are handled exclusively by the offline conversion tool. This IR is
//! Vokra's own definition and depends on none of protobuf / abseil / onnx
//! (NFR-DS-02); the `deny.toml` bans list enforces this at the dependency
//! level.

use std::collections::HashSet;

use crate::error::{Result, VokraError};

use super::tensor::{TensorDesc, TensorId};

/// Operation kind — the ggml-style *flat op enum* of the Vokra IR (FR-EX-01).
///
/// M0-02 carries only minimal **placeholder** variants so the graph plumbing
/// can be exercised. Op families are added by their owning work packages —
/// do not add them here ahead of schedule:
///
/// - speech front-end ops (`stft` / `istft` / `mel_filterbank` / `mfcc` /
///   `dct`, FR-OP-01/03) and their attribute definitions: **M0-04**
/// - LSTM family for the Silero VAD subgraph: **M0-05**
/// - attention / decoder family for Whisper: **M0-06**
///
/// The enum is `#[non_exhaustive]` so those additions do not break
/// downstream matches; backends must treat unknown ops as unsupported
/// (explicit error, FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OpKind {
    /// Matrix multiplication (placeholder).
    MatMul,
    /// Element-wise addition (placeholder).
    Add,
    /// Element-wise multiplication (placeholder).
    Mul,
    /// Softmax over the innermost dimension (placeholder).
    Softmax,
}

/// One node of an [`AudioGraph`]: an op together with its tensor
/// inputs / outputs (referenced by [`TensorId`]).
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) op: OpKind,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
}

impl Node {
    /// The operation this node performs.
    pub fn op(&self) -> &OpKind {
        &self.op
    }

    /// Tensors read by this node.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Tensors written by this node.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }
}

/// The *audio graph descriptor* — Vokra's own IR container (FR-EX-01).
///
/// A graph owns a tensor table ([`TensorDesc`]) plus a flat list of
/// [`Node`]s, and declares which tensors are graph inputs / outputs.
/// Construct it with [`GraphBuilder`]; [`AudioGraph::validate`] checks
/// structural consistency.
#[derive(Debug, Clone)]
pub struct AudioGraph {
    pub(crate) tensors: Vec<TensorDesc>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
}

impl AudioGraph {
    /// Tensor table of the graph.
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    /// Descriptor for `id`, or `None` if `id` is out of range.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors.get(id.0)
    }

    /// Nodes in insertion order (M0: no scheduling / topological pass yet).
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Tensors declared as graph inputs.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Tensors declared as graph outputs.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    /// Validates structural consistency of the graph (M0-02-T09).
    ///
    /// Checks performed:
    ///
    /// - every [`TensorId`] referenced by nodes and by the graph
    ///   input / output lists is in range (no dangling ids),
    /// - tensor names are unique,
    /// - every tensor is produced by at most one node (single-producer
    ///   consistency of node outputs).
    ///
    /// Violations are reported as [`VokraError::GraphValidation`].
    pub fn validate(&self) -> Result<()> {
        let len = self.tensors.len();

        let mut names: HashSet<&str> = HashSet::with_capacity(len);
        for desc in &self.tensors {
            if !names.insert(desc.name.as_str()) {
                return Err(VokraError::GraphValidation(format!(
                    "duplicate tensor name `{}`",
                    desc.name
                )));
            }
        }

        for (i, node) in self.nodes.iter().enumerate() {
            for id in &node.inputs {
                check_id(*id, len, &format!("node #{i} ({:?}) input", node.op))?;
            }
            for id in &node.outputs {
                check_id(*id, len, &format!("node #{i} ({:?}) output", node.op))?;
            }
        }

        let mut producer: Vec<Option<usize>> = vec![None; len];
        for (i, node) in self.nodes.iter().enumerate() {
            for id in &node.outputs {
                if let Some(prev) = producer[id.0] {
                    return Err(VokraError::GraphValidation(format!(
                        "tensor `{}` is produced by both node #{prev} and node #{i}",
                        self.tensors[id.0].name
                    )));
                }
                producer[id.0] = Some(i);
            }
        }

        for id in &self.inputs {
            check_id(*id, len, "graph input")?;
        }
        for id in &self.outputs {
            check_id(*id, len, "graph output")?;
        }

        Ok(())
    }
}

fn check_id(id: TensorId, len: usize, what: &str) -> Result<()> {
    if id.0 >= len {
        return Err(VokraError::GraphValidation(format!(
            "{what} references tensor id {} but the graph has only {len} tensors",
            id.0
        )));
    }
    Ok(())
}

/// Incremental builder for [`AudioGraph`] (M0-02-T09).
///
/// [`GraphBuilder::finish`] runs [`AudioGraph::validate`] so an invalid
/// graph is rejected at construction time.
///
/// # Examples
///
/// ```
/// use vokra_core::{DType, GraphBuilder, OpKind, TensorDesc};
///
/// let mut builder = GraphBuilder::new();
/// let x = builder.add_tensor(TensorDesc::new("x", DType::F32, [2, 4]));
/// let w = builder.add_tensor(TensorDesc::new("w", DType::F32, [4, 8]));
/// let y = builder.add_tensor(TensorDesc::new("y", DType::F32, [2, 8]));
/// builder.add_node(OpKind::MatMul, &[x, w], &[y]);
/// builder.mark_input(x);
/// builder.mark_output(y);
///
/// let graph = builder.finish().expect("graph is structurally valid");
/// assert_eq!(graph.nodes().len(), 1);
/// assert_eq!(graph.tensor(y).unwrap().name, "y");
/// ```
#[derive(Debug, Default)]
pub struct GraphBuilder {
    tensors: Vec<TensorDesc>,
    nodes: Vec<Node>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
}

impl GraphBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tensor and returns its id within the graph being built.
    pub fn add_tensor(&mut self, desc: TensorDesc) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(desc);
        id
    }

    /// Appends a node executing `op` over the given tensors.
    pub fn add_node(&mut self, op: OpKind, inputs: &[TensorId], outputs: &[TensorId]) {
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        });
    }

    /// Declares `id` as a graph input.
    pub fn mark_input(&mut self, id: TensorId) {
        self.inputs.push(id);
    }

    /// Declares `id` as a graph output.
    pub fn mark_output(&mut self, id: TensorId) {
        self.outputs.push(id);
    }

    /// Finalizes the graph, running [`AudioGraph::validate`].
    pub fn finish(self) -> Result<AudioGraph> {
        let graph = AudioGraph {
            tensors: self.tensors,
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
        };
        graph.validate()?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::DType;

    fn desc(name: &str) -> TensorDesc {
        TensorDesc::new(name, DType::F32, [2, 2])
    }

    fn assert_graph_validation_err(result: Result<AudioGraph>, needle: &str) {
        match result {
            Err(VokraError::GraphValidation(msg)) => {
                assert!(
                    msg.contains(needle),
                    "message `{msg}` should contain `{needle}`"
                );
            }
            other => panic!("expected GraphValidation error, got {other:?}"),
        }
    }

    #[test]
    fn small_graph_builds_and_validates() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let w = b.add_tensor(desc("w"));
        let h = b.add_tensor(desc("h"));
        let bias = b.add_tensor(desc("bias"));
        let y = b.add_tensor(desc("y"));
        b.add_node(OpKind::MatMul, &[x, w], &[h]);
        b.add_node(OpKind::Add, &[h, bias], &[y]);
        b.mark_input(x);
        b.mark_output(y);

        let graph = b.finish().expect("valid graph");
        assert_eq!(graph.tensors().len(), 5);
        assert_eq!(graph.nodes().len(), 2);
        assert_eq!(graph.inputs(), &[x]);
        assert_eq!(graph.outputs(), &[y]);
        assert_eq!(graph.nodes()[1].op(), &OpKind::Add);
        assert_eq!(graph.nodes()[0].inputs(), &[x, w]);
        assert_eq!(graph.nodes()[0].outputs(), &[h]);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn dangling_node_input_is_rejected() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let y = b.add_tensor(desc("y"));
        // TensorId(42) does not exist in this graph (ids are crate-internal,
        // so a dangling id can only be fabricated here in unit tests or by
        // mixing ids across builders).
        b.add_node(OpKind::Add, &[x, TensorId(42)], &[y]);
        b.mark_output(y);
        assert_graph_validation_err(b.finish(), "tensor id 42");
    }

    #[test]
    fn duplicate_tensor_name_is_rejected() {
        let mut b = GraphBuilder::new();
        let a = b.add_tensor(desc("same"));
        let _dup = b.add_tensor(desc("same"));
        b.mark_output(a);
        assert_graph_validation_err(b.finish(), "duplicate tensor name `same`");
    }

    #[test]
    fn undefined_graph_output_is_rejected() {
        let mut b = GraphBuilder::new();
        let _x = b.add_tensor(desc("x"));
        b.mark_output(TensorId(9));
        assert_graph_validation_err(b.finish(), "graph output");
    }

    #[test]
    fn double_producer_is_rejected() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let y = b.add_tensor(desc("y"));
        b.add_node(OpKind::Softmax, &[x], &[y]);
        b.add_node(OpKind::Mul, &[x, x], &[y]);
        b.mark_output(y);
        assert_graph_validation_err(b.finish(), "produced by both node #0 and node #1");
    }
}
