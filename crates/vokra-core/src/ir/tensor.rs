//! IR base types: element data type and tensor descriptor (M0-02-T07).
//!
//! These are the foundation of Vokra's own IR, the *audio graph descriptor*
//! (FR-EX-01; CLAUDE.md "IR: 独自 IR (audio graph descriptor)"). The IR is
//! defined from scratch and carries no protobuf / abseil / onnx dependency
//! (NFR-DS-02).

/// Element data type of a tensor.
///
/// The enum is `#[non_exhaustive]` to keep room for later additions:
/// `complex64` (FR-EX-09) is a **v0.1 MVP requirement and intentionally not
/// part of M0**; quantized types follow their own roadmap items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float (descriptor only in M0; no host storage type yet).
    F16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// Boolean, stored as one byte per element.
    Bool,
}

impl DType {
    /// Size in bytes of a single element of this type.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }
}

/// Identifier of a tensor inside one [`AudioGraph`](super::AudioGraph).
///
/// This is an index into the graph's tensor table; it is only meaningful for
/// the graph (or [`GraphBuilder`](super::GraphBuilder)) that issued it.
/// [`AudioGraph::validate`](super::AudioGraph::validate) rejects out-of-range
/// ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) usize);

impl TensorId {
    /// Position of the tensor in the graph's tensor table.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Static description of a tensor: name, element type and shape.
///
/// M0 keeps shapes fully static (`Vec<usize>`); symbolic / streaming
/// dimensions are a later design step tied to the streaming requirements.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDesc {
    /// Human-readable name, unique within a graph
    /// (uniqueness is enforced by
    /// [`AudioGraph::validate`](super::AudioGraph::validate)).
    pub name: String,
    /// Element data type.
    pub dtype: DType,
    /// Dimensions, outermost first. An empty shape denotes a scalar.
    pub shape: Vec<usize>,
}

impl TensorDesc {
    /// Creates a new tensor descriptor.
    pub fn new(name: impl Into<String>, dtype: DType, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
        }
    }

    /// Total number of elements (product of all dimensions).
    ///
    /// Returns 1 for a scalar (empty shape) and `None` if the product
    /// overflows `usize`.
    pub fn num_elements(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }

    /// Total size in bytes (`num_elements * dtype size`).
    ///
    /// Returns `None` if the computation overflows `usize`.
    pub fn byte_size(&self) -> Option<usize> {
        self.num_elements()?.checked_mul(self.dtype.size_in_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::I32.size_in_bytes(), 4);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        assert_eq!(DType::U8.size_in_bytes(), 1);
        assert_eq!(DType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn num_elements_and_byte_size() {
        let t = TensorDesc::new("mel", DType::F32, [2, 3, 4]);
        assert_eq!(t.num_elements(), Some(24));
        assert_eq!(t.byte_size(), Some(96));

        let scalar = TensorDesc::new("gain", DType::F16, []);
        assert_eq!(scalar.num_elements(), Some(1));
        assert_eq!(scalar.byte_size(), Some(2));

        let zero = TensorDesc::new("empty", DType::I64, [0, 8]);
        assert_eq!(zero.num_elements(), Some(0));
        assert_eq!(zero.byte_size(), Some(0));
    }

    #[test]
    fn overflow_is_detected() {
        let t = TensorDesc::new("huge", DType::U8, [usize::MAX, 2]);
        assert_eq!(t.num_elements(), None);
        assert_eq!(t.byte_size(), None);

        let t2 = TensorDesc::new("huge2", DType::I64, [usize::MAX]);
        assert_eq!(t2.num_elements(), Some(usize::MAX));
        assert_eq!(t2.byte_size(), None);
    }

    #[test]
    fn tensor_id_index_roundtrip() {
        assert_eq!(TensorId(3).index(), 3);
    }
}
