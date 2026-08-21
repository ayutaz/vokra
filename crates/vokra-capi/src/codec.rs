//! Generic streaming codec-decoder C ABI (#48): token indices in, mono PCM
//! out, without forcing the caller's acoustic model into Vokra.
//!
//! # Ownership and threads
//!
//! `vokra_codec_decoder_t` is an opaque, stateful, **single-owner-thread**
//! handle, matching `vokra_s2s_duplex_t`: it may be moved to another thread,
//! but the same live handle must never be pushed, pulled, reset, or destroyed
//! concurrently. Each `vokra_codec_decoder_open` call creates independent
//! causal state and retains a clone of its source session, so the caller may
//! destroy the original `vokra_session_t` immediately after open.
//!
//! The immutable model weights can therefore be shared through multiple
//! independently opened decoder handles without sharing mutable decode state.
//! There is no worker thread and no callback: callers drive progress solely
//! through push/pull, preserving the Unity WebGL thread-free contract checked
//! by `scripts/check-capi-thread-free.sh`.
//!
//! # Shape and backpressure
//!
//! `n_codebooks` comes from the loaded checkpoint and is also a call-time
//! argument to `push_codes`; it is deliberately not a header constant. One
//! successful push accepts exactly one complete code frame. The caller must
//! pull its PCM before pushing the next frame; violations fail loudly rather
//! than overwriting pending audio.

use vokra_core::{CodecDecoderHandle, Session, VokraError};

use crate::error::{self, vokra_status_t};
use crate::ffi_guard;
use crate::handle::{self, vokra_session_t};

/// Opaque single-owner streaming codec decoder (module docs).
#[allow(non_camel_case_types)]
pub struct vokra_codec_decoder_t {
    /// Stateful decoder; drops before the retained model session below.
    decoder: Box<dyn CodecDecoderHandle + Send>,
    /// Keeps model weights alive independently of the source C handle.
    _session: Session,
}

/// Opens a fresh streaming codec decoder for `session`.
///
/// Returns `NULL` and records detail in `vokra_last_error()` when the loaded
/// model does not expose a complete streaming token-to-PCM decoder. Currently
/// standalone Mimi opts in; partial SNAC support remains an explicit error
/// until its terminal PCM decoder exists.
///
/// # Safety
///
/// `session` must be `NULL` or a live `vokra_session_t`. The returned handle,
/// when non-NULL, must be destroyed exactly once with
/// [`vokra_codec_decoder_destroy`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_open(
    session: *const vokra_session_t,
) -> *mut vokra_codec_decoder_t {
    ffi_guard::guard_ptr(|| {
        // SAFETY: NULL is rejected; a non-NULL pointer is live per contract.
        let s = match unsafe { ffi_guard::required_ref(session, "session") } {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        match s.session.open_codec_decoder() {
            Ok(decoder) => handle::into_raw(vokra_codec_decoder_t {
                decoder,
                _session: s.session.clone(),
            }),
            Err(err) => {
                error::fail(&err);
                std::ptr::null_mut()
            }
        }
    })
}

/// Returns PCM samples emitted per complete code frame, or `-1` on error.
///
/// # Safety
///
/// `decoder` must be a live handle or `NULL` (reported as `-1`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_frame_hop(
    decoder: *const vokra_codec_decoder_t,
) -> i32 {
    direct_i32_property(decoder, "decoder", |d| d.decoder.frame_hop())
}

/// Returns the decoder PCM sample rate in Hz, or `-1` on error.
///
/// # Safety
///
/// `decoder` must be a live handle or `NULL` (reported as `-1`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_sample_rate(
    decoder: *const vokra_codec_decoder_t,
) -> i32 {
    direct_i32_property(decoder, "decoder", |d| d.decoder.sample_rate() as usize)
}

/// Returns the checkpoint's codebook count, or `-1` on error. This value is
/// informational; callers must still pass the count to every push so shape
/// mismatches remain observable at the ABI boundary.
///
/// # Safety
///
/// `decoder` must be a live handle or `NULL` (reported as `-1`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_n_codebooks(
    decoder: *const vokra_codec_decoder_t,
) -> i32 {
    direct_i32_property(decoder, "decoder", |d| d.decoder.n_codebooks())
}

fn direct_i32_property(
    decoder: *const vokra_codec_decoder_t,
    name: &str,
    property: impl FnOnce(&vokra_codec_decoder_t) -> usize,
) -> i32 {
    let mut value = -1;
    let status = ffi_guard::guard(|| {
        // SAFETY: the caller-facing wrappers carry the live-or-NULL contract.
        let d = unsafe { ffi_guard::required_ref(decoder, name)? };
        value = i32::try_from(property(d)).map_err(|_| {
            error::fail(&VokraError::InvalidArgument(
                "codec decoder property exceeds INT32_MAX".into(),
            ))
        })?;
        Ok(())
    });
    if status == vokra_status_t::VOKRA_OK {
        value
    } else {
        -1
    }
}

/// Pushes one complete code frame.
///
/// `n_codebooks` is a required call-time shape and must exactly match
/// `vokra_codec_decoder_n_codebooks(decoder)`. On success,
/// `*out_frames_emitted` is `1`; pull the corresponding PCM before the next
/// push. The warmed successful push/pull path performs no heap allocation.
///
/// # Safety
///
/// `decoder` must be a live handle owned by the calling thread; `codes` must
/// point to `n_codebooks` readable `uint32_t` values; `out_frames_emitted`
/// must be valid and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_push_codes(
    decoder: *mut vokra_codec_decoder_t,
    codes: *const u32,
    n_codebooks: usize,
    out_frames_emitted: *mut i32,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: NULL-checked unique borrow per the caller contract.
        let d = unsafe { ffi_guard::required_mut(decoder, "decoder")? };
        ffi_guard::require_out_ptr(out_frames_emitted, "out_frames_emitted")?;
        if n_codebooks != d.decoder.n_codebooks() {
            return Err(error::fail_invalid(&format!(
                "n_codebooks {n_codebooks} != checkpoint {}",
                d.decoder.n_codebooks()
            )));
        }
        // SAFETY: non-zero checkpoint width plus caller-readable buffer.
        let codes = unsafe { ffi_guard::required_slice(codes, n_codebooks, "codes")? };
        let emitted = d.decoder.push_codes(codes).map_err(|e| error::fail(&e))?;
        let emitted = i32::try_from(emitted).map_err(|_| {
            error::fail(&VokraError::InvalidArgument(
                "codec emitted frame count exceeds INT32_MAX".into(),
            ))
        })?;
        // SAFETY: out pointer was checked and is writable per contract.
        unsafe { *out_frames_emitted = emitted };
        Ok(())
    })
}

/// Pulls one pending PCM frame into `out`.
///
/// `capacity` must be at least `vokra_codec_decoder_frame_hop(decoder)` when
/// a frame is pending. `*out_len == 0` means there was nothing to pull.
///
/// # Safety
///
/// `decoder` must be a live handle owned by the calling thread; when
/// `capacity > 0`, `out` must point to that many writable floats; `out_len`
/// must be valid and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_pull_pcm(
    decoder: *mut vokra_codec_decoder_t,
    out: *mut f32,
    capacity: usize,
    out_len: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: NULL-checked unique borrow per the caller contract.
        let d = unsafe { ffi_guard::required_mut(decoder, "decoder")? };
        ffi_guard::require_out_ptr(out_len, "out_len")?;
        let output: &mut [f32] = if capacity == 0 {
            &mut []
        } else {
            if out.is_null() {
                return Err(error::fail_invalid(
                    "out must not be NULL when capacity is non-zero",
                ));
            }
            // SAFETY: caller guarantees `capacity` writable floats.
            unsafe { std::slice::from_raw_parts_mut(out, capacity) }
        };
        let written = d.decoder.pull_pcm(output).map_err(|e| error::fail(&e))?;
        // SAFETY: out pointer was checked and is writable per contract.
        unsafe { *out_len = written };
        Ok(())
    })
}

/// Resets the causal decoder to its as-new state and discards pending PCM.
/// Errors (including `NULL`) are recorded in `vokra_last_error()`; this exact
/// pre-freeze API is void, matching the issue contract.
///
/// # Safety
///
/// `decoder` must be a live handle owned by the calling thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_reset(decoder: *mut vokra_codec_decoder_t) {
    let _ = ffi_guard::guard(|| {
        // SAFETY: NULL-checked unique borrow per the caller contract.
        let d = unsafe { ffi_guard::required_mut(decoder, "decoder")? };
        d.decoder.reset().map_err(|e| error::fail(&e))?;
        Ok(())
    });
}

/// Destroys a decoder handle. `NULL` is a no-op; double-free or concurrent
/// use/destroy is undefined behaviour.
///
/// # Safety
///
/// `decoder` must be `NULL` or a live handle returned by
/// [`vokra_codec_decoder_open`] and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_codec_decoder_destroy(decoder: *mut vokra_codec_decoder_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: NULL or a unique live Box from open, per contract.
        unsafe { handle::drop_raw(decoder) };
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vokra_core::gguf::{GgufBuilder, GgufFile};
    use vokra_core::{CodecDecoderEngine, Result};

    use super::*;

    struct FakeEngine;
    struct FakeDecoder {
        pending: bool,
        value: f32,
    }

    impl CodecDecoderEngine for FakeEngine {
        fn open_decoder(&self) -> Result<Box<dyn CodecDecoderHandle + Send>> {
            Ok(Box::new(FakeDecoder {
                pending: false,
                value: 0.0,
            }))
        }
    }

    impl CodecDecoderHandle for FakeDecoder {
        fn frame_hop(&self) -> usize {
            4
        }
        fn sample_rate(&self) -> u32 {
            22_050
        }
        fn n_codebooks(&self) -> usize {
            3
        }
        fn push_codes(&mut self, codes: &[u32]) -> Result<usize> {
            self.value = codes.iter().sum::<u32>() as f32;
            self.pending = true;
            Ok(1)
        }
        fn pull_pcm(&mut self, out: &mut [f32]) -> Result<usize> {
            if !self.pending {
                return Ok(0);
            }
            if out.len() < 4 {
                return Err(VokraError::InvalidArgument("short output".into()));
            }
            out[..4].copy_from_slice(&[self.value, 1.0, -1.0, -self.value]);
            self.pending = false;
            Ok(4)
        }
        fn reset(&mut self) -> Result<()> {
            self.pending = false;
            self.value = 0.0;
            Ok(())
        }
    }

    fn session_handle(with_engine: bool) -> *mut vokra_session_t {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "test");
        let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let mut session = Session::from_gguf(gguf).build().unwrap();
        if with_engine {
            session = session.with_codec_decoder_engine(Arc::new(FakeEngine));
        }
        handle::into_raw(vokra_session_t { session })
    }

    #[test]
    fn c_surface_roundtrip_and_call_time_codebook_shape() {
        let session = session_handle(true);
        // SAFETY: live session handle.
        let decoder = unsafe { vokra_codec_decoder_open(session) };
        assert!(!decoder.is_null());
        // SAFETY: live decoder handle.
        unsafe {
            assert_eq!(vokra_codec_decoder_frame_hop(decoder), 4);
            assert_eq!(vokra_codec_decoder_sample_rate(decoder), 22_050);
            assert_eq!(vokra_codec_decoder_n_codebooks(decoder), 3);
        }

        let codes = [2u32, 3, 4];
        let mut emitted = -7;
        // SAFETY: live handle, readable codes, writable output.
        let bad =
            unsafe { vokra_codec_decoder_push_codes(decoder, codes.as_ptr(), 2, &mut emitted) };
        assert_eq!(bad, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert_eq!(emitted, -7, "out-param is untouched on error");

        // SAFETY: live handle and exact checkpoint shape.
        let ok = unsafe {
            vokra_codec_decoder_push_codes(decoder, codes.as_ptr(), codes.len(), &mut emitted)
        };
        assert_eq!(ok, vokra_status_t::VOKRA_OK);
        assert_eq!(emitted, 1);

        let mut pcm = [0.0f32; 4];
        let mut written = usize::MAX;
        // SAFETY: live handle and writable output buffer.
        let ok = unsafe {
            vokra_codec_decoder_pull_pcm(decoder, pcm.as_mut_ptr(), pcm.len(), &mut written)
        };
        assert_eq!(ok, vokra_status_t::VOKRA_OK);
        assert_eq!(written, 4);
        assert_eq!(pcm, [9.0, 1.0, -1.0, -9.0]);

        // SAFETY: live handles, each destroyed exactly once.
        unsafe {
            vokra_codec_decoder_reset(decoder);
            vokra_codec_decoder_destroy(decoder);
            handle::drop_raw(session);
        }
    }

    #[test]
    fn open_fails_loudly_without_codec_engine_and_null_queries_are_minus_one() {
        let session = session_handle(false);
        // SAFETY: live session but intentionally no codec engine.
        assert!(unsafe { vokra_codec_decoder_open(session) }.is_null());
        // SAFETY: NULL is the documented error branch.
        assert_eq!(
            unsafe { vokra_codec_decoder_n_codebooks(std::ptr::null()) },
            -1
        );
        // SAFETY: session is live and freed once.
        unsafe { handle::drop_raw(session) };
    }
}
