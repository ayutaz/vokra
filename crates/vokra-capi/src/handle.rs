//! Opaque handles and the `Box` raw-pointer lifecycle (M0-09-T04).
//!
//! The two FR-API-01 handles — `vokra_session_t` and `vokra_stream_t` —
//! wrap Rust types that have no C layout (`Arc`, `Box<dyn ...>`, `VecDeque`),
//! so cbindgen emits them as **opaque** forward declarations: C code only ever
//! holds a pointer, obtained from a `create`/`open` call and released by the
//! matching `destroy` (ADR-0003 §1). Handles cross the boundary as
//! `Box::into_raw` pointers and come back via `Box::from_raw`.

use std::collections::VecDeque;

use vokra_core::Session;
use vokra_core::engines::VadStreamHandle;

/// Opaque session handle: one loaded model bound to one backend, with the
/// matching native engine injected (ASR / TTS / VAD — see
/// `crate::session::vokra_session_create_from_file`).
///
/// Created by `vokra_session_create_from_file`, released by
/// `vokra_session_destroy`. Opaque to C.
//
// C-style name so cbindgen emits `vokra_session_t` verbatim (see error.rs).
#[allow(non_camel_case_types)]
pub struct vokra_session_t {
    pub(crate) session: Session,
}

/// Opaque VAD stream handle: a stateful Silero VAD stream plus a FIFO of speech
/// probabilities computed by `vokra_stream_push_pcm` and drained by
/// `vokra_stream_poll`.
///
/// Created by `vokra_stream_open`, released by `vokra_stream_destroy`. Opaque
/// to C. All recurrent state (LSTM `h`/`c`, framing) is hidden inside
/// `handle` (FR-LD-06).
#[allow(non_camel_case_types)]
pub struct vokra_stream_t {
    /// The native VAD stream; hides the recurrent state.
    pub(crate) handle: Box<dyn VadStreamHandle + Send>,
    /// Sample rate fixed at open and passed to every `push_pcm`.
    pub(crate) sample_rate: u32,
    /// Frame probabilities awaiting `poll` (front = oldest).
    pub(crate) pending: VecDeque<f32>,
}

/// Moves `value` onto the heap and hands ownership to C as a raw pointer.
pub(crate) fn into_raw<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

/// Reconstructs and drops a handle produced by `into_raw`, freeing it exactly
/// once. `NULL` is a no-op (the destroy contract, ADR-0003 §3-a).
///
/// # Safety
///
/// `ptr` must be `NULL`, or a pointer returned by `into_raw::<T>` that has not
/// already been freed. Passing a dangling or foreign pointer, or freeing twice,
/// is undefined behaviour.
pub(crate) unsafe fn drop_raw<T>(ptr: *mut T) {
    if !ptr.is_null() {
        // SAFETY: per the contract `ptr` is a live `Box<T>` from `into_raw`;
        // reconstructing the Box and dropping it frees the allocation once.
        drop(unsafe { Box::from_raw(ptr) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_raw_then_drop_raw_roundtrips() {
        let ptr = into_raw(1234u32);
        assert!(!ptr.is_null());
        // SAFETY: `ptr` came from `into_raw` above and has not been freed.
        unsafe { drop_raw(ptr) };
    }

    #[test]
    fn drop_raw_null_is_a_noop() {
        // SAFETY: NULL is an explicit no-op in the contract.
        unsafe { drop_raw::<u32>(std::ptr::null_mut()) };
    }

    #[test]
    fn drop_raw_runs_destructor_once() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let ptr = into_raw(Guard(Arc::clone(&drops)));
        // SAFETY: live pointer from `into_raw`, freed exactly once here.
        unsafe { drop_raw(ptr) };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
