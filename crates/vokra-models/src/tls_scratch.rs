//! Grow-only thread-local `f32` scratch for model-side conv glue
//! (M5-14 Wave-2 T16/T17).
//!
//! Wave-0 attributed a large share of the piper / CAM++ / Mimi wall to
//! **fresh per-call im2col buffers** (`vec![0.0; k * out_len]` — up to
//! ~50 MB per CAM++ conv2d call): the allocator round-trip plus the kernel
//! zeroing of fresh pages dominates the actual gather. These helpers keep
//! one grow-only buffer per thread (the whisper.cpp reused-arena posture,
//! FR-EX-05) so steady-state conv calls allocate nothing.
//!
//! # Contract
//!
//! The slice handed to `f` is **uninitialised** (stale bytes from earlier
//! calls). Callers must write every element they later read — the conv
//! im2col fills do exactly that by construction (interior copy + explicit
//! zero-fill of the padding ranges), which is also what makes them
//! bit-identical to the old `vec![0.0; ..]` + partial-overwrite pattern.
//!
//! # Reentrancy / threading
//!
//! Buffers are thread-local, so concurrent model calls on different
//! threads never share scratch. A *reentrant* use on one thread (an `f`
//! that itself calls back into a scratch user) would hit the `RefCell`
//! double-borrow; instead of panicking, the helpers fall back to a fresh
//! one-shot allocation — correctness never depends on the reuse.

use std::cell::RefCell;

thread_local! {
    /// Primary conv scratch (the im2col `col` matrix).
    static SCRATCH_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Secondary conv scratch (per-group GEMM output, transposes, …).
    static SCRATCH_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with a `len`-element scratch slice from this thread's grow-only
/// primary buffer (fresh heap allocation only on growth or reentrancy).
pub(crate) fn with_col_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    SCRATCH_A.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < len {
                buf.resize(len, 0.0);
            }
            f(&mut buf[..len])
        }
        // Reentrant use: fall back to a one-shot allocation.
        Err(_) => f(&mut vec![0.0f32; len]),
    })
}

/// Like [`with_col_scratch`] but hands out **two** independent slices (the
/// im2col `col` plus a per-group GEMM output), each from its own
/// thread-local buffer.
pub(crate) fn with_col_scratch2<R>(
    len_a: usize,
    len_b: usize,
    f: impl FnOnce(&mut [f32], &mut [f32]) -> R,
) -> R {
    SCRATCH_A.with(|cell_a| match cell_a.try_borrow_mut() {
        Ok(mut a) => {
            if a.len() < len_a {
                a.resize(len_a, 0.0);
            }
            SCRATCH_B.with(|cell_b| match cell_b.try_borrow_mut() {
                Ok(mut b) => {
                    if b.len() < len_b {
                        b.resize(len_b, 0.0);
                    }
                    f(&mut a[..len_a], &mut b[..len_b])
                }
                Err(_) => f(&mut a[..len_a], &mut vec![0.0f32; len_b]),
            })
        }
        Err(_) => f(&mut vec![0.0f32; len_a], &mut vec![0.0f32; len_b]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_grows_and_is_reused() {
        let cap0 = with_col_scratch(16, |s| {
            assert_eq!(s.len(), 16);
            s.fill(1.0);
            s.len()
        });
        assert_eq!(cap0, 16);
        // A larger request grows; a smaller one hands back a prefix (stale
        // contents allowed by the contract).
        with_col_scratch(64, |s| assert_eq!(s.len(), 64));
        with_col_scratch(8, |s| assert_eq!(s.len(), 8));
    }

    #[test]
    fn two_slices_are_disjoint() {
        with_col_scratch2(8, 8, |a, b| {
            a.fill(1.0);
            b.fill(2.0);
            assert!(a.iter().all(|&v| v == 1.0));
            assert!(b.iter().all(|&v| v == 2.0));
        });
    }

    #[test]
    fn reentrant_use_falls_back_to_fresh_alloc() {
        with_col_scratch(4, |outer| {
            outer.fill(3.0);
            // Inner borrow of the SAME thread-local must not panic.
            with_col_scratch(4, |inner| {
                inner.fill(7.0);
                assert!(inner.iter().all(|&v| v == 7.0));
            });
            assert!(outer.iter().all(|&v| v == 3.0), "outer slice untouched");
        });
    }
}
