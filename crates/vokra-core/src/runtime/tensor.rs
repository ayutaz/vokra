//! Runtime tensor with real element storage (Phase 1 of the graph engine).
//!
//! [`TensorDesc`](crate::TensorDesc) is a *descriptor* — a name, dtype and
//! shape with no data (it is what the [`AudioGraph`](crate::AudioGraph) IR is
//! built from). The graph evaluator ([`run_graph`](crate::run_graph)) needs a
//! value type that actually *carries elements* between nodes; that is
//! [`Tensor`].
//!
//! # Host-f32-first, device-ready
//!
//! The MVP canonical form is host-resident row-major `f32`
//! ([`Storage::Host`]). The [`Storage`] enum keeps room to add on-device
//! residency later (a `Device` variant holding an opaque handle) **without
//! breaking this type's public API or the [`Backend`](crate::Backend) trait**:
//! [`as_f32`](Tensor::as_f32) is already fallible so a future device tensor can
//! copy back to the host there, and [`numel`](Tensor::numel) is derived from
//! the shape, so it is storage-independent.

use crate::error::{Result, VokraError};
use crate::ir::DType;

/// Element count of a shape (product of its axis extents), or `None` on
/// `usize` overflow. A scalar (empty shape) has one element.
fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Backing store for a [`Tensor`].
///
/// MVP has only host-resident storage; the commented `Device` arm is the
/// Phase 5 extension point (an opaque device handle) and is why
/// [`Tensor::as_f32`] returns a [`Result`].
#[derive(Debug, Clone, PartialEq)]
enum Storage {
    /// Canonical MVP form: host-resident, row-major `f32` elements.
    Host(Vec<f32>),
    // Device(Arc<dyn DeviceTensor>),  // Phase 5: on-device residency (D2H in as_f32).
}

/// A runtime value flowing between [`AudioGraph`](crate::AudioGraph) nodes:
/// an element buffer plus its dtype and shape.
///
/// Distinct from [`TensorDesc`](crate::TensorDesc), which describes a graph
/// tensor slot but holds no data. Construct one with
/// [`host_f32`](Self::host_f32) or [`zeros_f32`](Self::zeros_f32).
///
/// The `shape` is `Vec<usize>` (concrete, fully-resolved extents) rather than
/// the descriptor's [`Dim`](crate::Dim) shape: by the time a value exists its
/// variable-length axes have been resolved to real sizes.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Element data type. The MVP evaluator produces [`DType::F32`] values.
    pub dtype: DType,
    /// Concrete shape, outermost axis first. An empty shape denotes a scalar.
    /// The product of the extents equals the element count of the backing
    /// storage (upheld by the constructors).
    pub shape: Vec<usize>,
    /// Backing element store (host-resident in the MVP).
    storage: Storage,
}

impl Tensor {
    /// Builds a host-resident `f32` tensor from an explicit `shape` and `data`.
    ///
    /// The element count of `shape` must equal `data.len()`, otherwise this is
    /// an explicit [`VokraError::InvalidArgument`] (a shape whose product
    /// overflows `usize` can never match a real buffer length and is rejected
    /// the same way).
    pub fn host_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected = checked_numel(&shape).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "Tensor::host_f32: shape {shape:?} element count overflows usize"
            ))
        })?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "Tensor::host_f32: data length {} does not match shape {:?} (= {} elements)",
                data.len(),
                shape,
                expected
            )));
        }
        Ok(Self {
            dtype: DType::F32,
            shape,
            storage: Storage::Host(data),
        })
    }

    /// Builds a host-resident `f32` tensor of the given `shape` filled with
    /// zeros.
    ///
    /// The shape's element count must fit `usize` (a shape that large cannot be
    /// allocated in any case).
    pub fn zeros_f32(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            dtype: DType::F32,
            shape,
            storage: Storage::Host(vec![0.0f32; n]),
        }
    }

    /// Borrows the elements as an `f32` slice.
    ///
    /// Fallible so a future on-device tensor can perform a device-to-host copy
    /// here; a host tensor always succeeds. (No non-`f32` storage exists in the
    /// MVP, so this never errors today.)
    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.storage {
            Storage::Host(data) => Ok(data),
        }
    }

    /// Number of elements (product of the shape extents; `1` for a scalar).
    ///
    /// Derived from the shape, so it is independent of where the storage lives.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_f32_roundtrips_shape_and_data() {
        let t = Tensor::host_f32(vec![2, 3], (0..6).map(|v| v as f32).collect()).unwrap();
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_f32().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn scalar_shape_has_one_element() {
        // Empty shape = scalar: product over no axes is 1.
        let t = Tensor::host_f32(vec![], vec![42.0]).unwrap();
        assert_eq!(t.numel(), 1);
        assert_eq!(t.as_f32().unwrap(), &[42.0]);
    }

    #[test]
    fn zeros_f32_is_all_zero_and_right_size() {
        let t = Tensor::zeros_f32(vec![4]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.as_f32().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn host_f32_rejects_length_shape_mismatch() {
        // 2*3 = 6 elements expected, 5 supplied → explicit InvalidArgument.
        let err = Tensor::host_f32(vec![2, 3], vec![0.0; 5]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn host_f32_rejects_overflowing_shape() {
        // A fully-static shape whose product overflows usize can never match a
        // real buffer length → rejected (not a silent wrap).
        let err = Tensor::host_f32(vec![usize::MAX, 2], vec![0.0; 4]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn zero_extent_axis_is_empty() {
        let t = Tensor::host_f32(vec![0, 8], vec![]).unwrap();
        assert_eq!(t.numel(), 0);
        assert!(t.as_f32().unwrap().is_empty());
    }
}
