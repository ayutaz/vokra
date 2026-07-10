//! Vokra VAD streaming API exposed to Godot (T08). Silero VAD v5 is driven
//! through a single-producer / single-consumer streaming state machine
//! (`vokra_stream_*`, M1-08). Godot GDScript owns the push side (mic PCM
//! chunks) and the poll side (drained speech probabilities or typed events).
//!
//! GDScript surface (proposal, finalised in T05/T09):
//!
//! ```gdscript
//! # Open a Silero VAD stream at 16000 Hz.
//! var stream: VokraStream = session.vad_open_stream(16000)
//! stream.push_pcm(chunk: PackedFloat32Array)
//! var probs: PackedFloat32Array = stream.poll(capacity: int)
//! stream.interrupt()  # M3-14 barge-in
//! # ...
//! stream.free()
//! ```

use core::ptr;

use crate::error::{VokraError, check};
use crate::ffi::capi::{
    VokraEvent, VokraStatus, VokraStream as CVokraStream, vokra_stream_destroy,
    vokra_stream_interrupt, vokra_stream_open, vokra_stream_poll, vokra_stream_poll_events,
    vokra_stream_push_pcm,
};
use crate::session::VokraSession;

/// RAII wrapper over `vokra_stream_t`. See the module docs for lifecycle.
pub struct VokraStream {
    handle: *mut CVokraStream,
}

// SAFETY: `vokra_stream_t` is `Send + Sync` by C ABI docs (see
// `crates/vokra-capi/src/stream.rs`). ADR-00xx §3 pins single-thread ownership.
unsafe impl Send for VokraStream {}
// SAFETY: Same rationale as `Send`; the underlying ring is SPSC per ADR-0003.
unsafe impl Sync for VokraStream {}

impl VokraStream {
    /// Open a VAD stream over a Silero session at the given sample rate.
    pub fn open(session: &VokraSession, sample_rate: i32) -> Result<Self, VokraError> {
        let mut out: *mut CVokraStream = ptr::null_mut();
        // SAFETY: `session.as_raw()` is a live non-null handle; `&mut out`
        // is a writable slot. ABI writes it only on VOKRA_OK.
        let status = unsafe { vokra_stream_open(session.as_raw(), sample_rate, &mut out) };
        check(status)?;
        if out.is_null() {
            return Err(VokraError {
                status: VokraStatus::Other,
                message: String::from("vokra_stream_open returned VOKRA_OK with a NULL stream"),
            });
        }
        Ok(Self { handle: out })
    }

    /// Push mono f32 PCM into the stream. Each completed VAD frame produces
    /// one event on the ring; drain them with [`poll`] or [`poll_events`].
    pub fn push_pcm(&mut self, pcm: &[f32]) -> Result<(), VokraError> {
        let (ptr_, n) = if pcm.is_empty() {
            (ptr::null::<f32>(), 0usize)
        } else {
            (pcm.as_ptr(), pcm.len())
        };
        // SAFETY: `self.handle` is a live stream (Drop is the only path to
        // NULL). `ptr_` is either NULL for empty or a valid pointer to `n`
        // f32 samples for this call.
        let status = unsafe { vokra_stream_push_pcm(self.handle, ptr_, n) };
        check(status)
    }

    /// Drain up to `capacity` speech probabilities into a fresh `Vec`.
    /// Non-blocking; returns an empty `Vec` when nothing is pending.
    pub fn poll(&mut self, capacity: usize) -> Result<Vec<f32>, VokraError> {
        let mut buf: Vec<f32> = Vec::with_capacity(capacity);
        let mut count: usize = 0;
        // SAFETY: `buf` has `capacity` writable f32 slots; the ABI writes at
        // most `capacity` values. We set the Vec length AFTER the ABI
        // returns the actual count.
        let status = unsafe {
            vokra_stream_poll(
                self.handle,
                if capacity == 0 {
                    ptr::null_mut()
                } else {
                    buf.as_mut_ptr()
                },
                capacity,
                &mut count,
            )
        };
        check(status)?;
        // SAFETY: the ABI wrote `count <= capacity` valid f32 values into
        // `buf`'s allocation; set_len is the standard MaybeUninit-fill idiom.
        unsafe {
            buf.set_len(count);
        }
        Ok(buf)
    }

    /// Drain up to `capacity` typed events (M1-08) into a fresh `Vec`.
    pub fn poll_events(&mut self, capacity: usize) -> Result<Vec<VokraEvent>, VokraError> {
        let mut buf: Vec<VokraEvent> = Vec::with_capacity(capacity);
        let mut count: usize = 0;
        // SAFETY: `buf` has `capacity` writable `VokraEvent` slots (POD, 12
        // bytes each); the ABI writes at most `capacity` events.
        let status = unsafe {
            vokra_stream_poll_events(
                self.handle,
                if capacity == 0 {
                    ptr::null_mut()
                } else {
                    buf.as_mut_ptr()
                },
                capacity,
                &mut count,
            )
        };
        check(status)?;
        // SAFETY: ABI wrote `count <= capacity` valid events.
        unsafe {
            buf.set_len(count);
        }
        Ok(buf)
    }

    /// Barge-in: flush chunk output, drain the ring, reset hidden state
    /// (M3-14 / FR-ST-03).
    pub fn interrupt(&mut self) -> Result<(), VokraError> {
        // SAFETY: `self.handle` is a live stream.
        let status = unsafe { vokra_stream_interrupt(self.handle) };
        check(status)
    }

    /// Test-only helper: construct a stream with a NULL C handle so sibling
    /// unit tests (e.g. `stream_push_pcm` trampoline dispatch) can exercise
    /// the backend-error branch without a real Silero VAD GGUF fixture. Any
    /// C ABI call on the resulting stream returns a non-OK status because
    /// `vokra-capi`'s `ffi_guard::required_mut` rejects NULL handles — see
    /// `crates/vokra-capi/src/stream.rs`. `Drop` is a no-op for NULL
    /// (see [`Drop::drop`]), so this is leak-free.
    ///
    /// Kept `pub(crate)` and gated on `#[cfg(test)]` to prevent accidental
    /// use from application code; the RAII invariant of the type
    /// (exactly-one-refcount-per-value) does NOT apply to a NULL handle
    /// because there is nothing to refcount.
    #[cfg(test)]
    pub(crate) fn null_for_tests() -> Self {
        Self {
            handle: ptr::null_mut(),
        }
    }
}

impl Drop for VokraStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` came from `vokra_stream_open` and has
            // not been destroyed (Drop only runs once).
            unsafe { vokra_stream_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    // Cross-thread type-hygiene assertions are the only tests we can run
    // without a real Silero GGUF. Full behaviour is exercised in Godot at T19.
    #[allow(dead_code)]
    fn stream_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<super::VokraStream>();
        assert_sync::<super::VokraStream>();
    }
}
