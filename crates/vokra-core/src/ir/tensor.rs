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

/// One axis extent of a [`TensorDesc`] shape: either statically known or
/// symbolic (variable-length).
///
/// Symbolic dims are how the IR represents **variable-length I/O** — a
/// streaming audio input, a decode whose token count is not known ahead of
/// time, and so on (M1-04 sub-part 2). A [`Dim::Dynamic`] axis has no fixed
/// extent, so any element-count arithmetic over it is undefined and reported as
/// `None` (see [`TensorDesc::num_elements`]);
/// [`AudioGraph::validate`](super::AudioGraph::validate) skips element-count
/// checks on such axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A statically known extent.
    Fixed(usize),
    /// A symbolic, variable-length extent (unknown until run time).
    Dynamic,
}

impl Dim {
    /// The static extent, or `None` for [`Dim::Dynamic`].
    pub const fn size(self) -> Option<usize> {
        match self {
            Self::Fixed(n) => Some(n),
            Self::Dynamic => None,
        }
    }

    /// Whether this axis is symbolic (variable-length).
    pub const fn is_dynamic(self) -> bool {
        matches!(self, Self::Dynamic)
    }
}

impl From<usize> for Dim {
    /// A plain extent is a [`Dim::Fixed`] — this is what lets
    /// [`TensorDesc::new`] keep accepting ordinary `usize` shapes unchanged.
    fn from(n: usize) -> Self {
        Self::Fixed(n)
    }
}

/// Static description of a tensor: name, element type and shape.
///
/// Each axis is a [`Dim`], so a shape may mix statically known extents with
/// symbolic [`Dim::Dynamic`] axes for variable-length I/O (M1-04 sub-part 2).
/// [`TensorDesc::new`] builds an all-[`Fixed`](Dim::Fixed) shape from a plain
/// `usize` shape (so existing call sites are unchanged); [`from_dims`](Self::from_dims)
/// takes an explicit [`Dim`] shape that may contain dynamic axes.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDesc {
    /// Human-readable name, unique within a graph
    /// (uniqueness is enforced by
    /// [`AudioGraph::validate`](super::AudioGraph::validate)).
    pub name: String,
    /// Element data type.
    pub dtype: DType,
    /// Dimensions, outermost first. An empty shape denotes a scalar; a
    /// [`Dim::Dynamic`] axis marks a variable-length extent.
    pub shape: Vec<Dim>,
}

impl TensorDesc {
    /// Creates a tensor descriptor with an all-[`Fixed`](Dim::Fixed) shape.
    ///
    /// The `shape` is any plain `usize` shape (`[2, 4]`, `vec![80]`, the scalar
    /// `[]`, …); every extent becomes a [`Dim::Fixed`]. Keeping this `usize`
    /// signature is what lets every existing call site compile unchanged. Use
    /// [`from_dims`](Self::from_dims) to build a shape with symbolic axes.
    pub fn new(name: impl Into<String>, dtype: DType, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into().into_iter().map(Dim::Fixed).collect(),
        }
    }

    /// Creates a tensor descriptor from an explicit [`Dim`] shape, which may
    /// contain [`Dim::Dynamic`] axes (variable-length I/O, M1-04 sub-part 2).
    ///
    /// `[Dim::Dynamic, Dim::Fixed(80)]` and `vec![Dim::Fixed(2), Dim::Dynamic]`
    /// are both accepted (via the standard array / `Vec` → `Vec<Dim>`
    /// conversions).
    pub fn from_dims(name: impl Into<String>, dtype: DType, shape: impl Into<Vec<Dim>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
        }
    }

    /// Total number of elements (product of all axis extents).
    ///
    /// Returns `Some(1)` for a scalar (empty shape). Returns `None` if any axis
    /// is [`Dim::Dynamic`] (a variable-length extent has no fixed element
    /// count) **or** if the product of the fixed extents overflows `usize`.
    pub fn num_elements(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, d| d.size().and_then(|n| acc.checked_mul(n)))
    }

    /// Total size in bytes (`num_elements * dtype size`).
    ///
    /// Returns `None` when [`num_elements`](Self::num_elements) is `None` (a
    /// dynamic axis or an overflowing fixed product), or the byte product
    /// itself overflows `usize`.
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

    #[test]
    fn dim_helpers() {
        assert_eq!(Dim::from(5usize), Dim::Fixed(5));
        assert_eq!(Dim::Fixed(3).size(), Some(3));
        assert_eq!(Dim::Dynamic.size(), None);
        assert!(Dim::Dynamic.is_dynamic());
        assert!(!Dim::Fixed(0).is_dynamic());
    }

    #[test]
    fn dynamic_dim_has_no_element_count() {
        // A variable-length axis makes both counts undefined (`None`), whatever
        // its position in the shape.
        let t = TensorDesc::from_dims("audio", DType::F32, [Dim::Dynamic, Dim::Fixed(80)]);
        assert_eq!(t.num_elements(), None);
        assert_eq!(t.byte_size(), None);

        let sig = TensorDesc::from_dims("pcm", DType::F32, [Dim::Dynamic]);
        assert_eq!(sig.num_elements(), None);
        assert_eq!(sig.byte_size(), None);
    }

    #[test]
    fn from_dims_all_fixed_matches_new() {
        // `from_dims` with only fixed extents reproduces `new`'s all-Fixed shape
        // and its element / byte counts exactly (the existing
        // `num_elements_and_byte_size` numbers).
        let a = TensorDesc::new("mel", DType::F32, [2, 3, 4]);
        let b = TensorDesc::from_dims(
            "mel",
            DType::F32,
            [Dim::Fixed(2), Dim::Fixed(3), Dim::Fixed(4)],
        );
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.num_elements(), Some(24));
        assert_eq!(b.num_elements(), Some(24));
        assert_eq!(a.byte_size(), b.byte_size());
    }
}
