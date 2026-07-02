//! VAD streaming: open / push PCM / poll probabilities / destroy (M0-09-T09).
//!
//! The M0 streaming route is Silero VAD (M0-05). [`vokra_stream_open`] wraps the
//! session's `open_vad_stream()` handle; [`vokra_stream_push_pcm`] feeds PCM and
//! buffers the resulting per-frame speech probabilities; [`vokra_stream_poll`]
//! drains them non-blockingly. This is the minimal synchronous push/poll of
//! FR-API-01 — the `step_chunk`/`step_frame` API, the lock-free ring buffer
//! (FR-ST-01/02) and atomic ref counting (FR-API-03) are M1 (ADR-0003, M0
//! scope). All recurrent state stays hidden inside the handle (FR-LD-06).

use std::collections::VecDeque;

use crate::error::vokra_status_t;
use crate::handle::{self, vokra_session_t, vokra_stream_t};
use crate::{error, ffi_guard};

/// Opens a VAD stream over a session (FR-API-01, proposed lifecycle helper).
///
/// # Parameters
///
/// - `session`: a session created from a Silero VAD GGUF.
/// - `sample_rate`: the stream sample rate in Hz (Silero accepts 8000 or
///   16000; other rates are rejected by the first `vokra_stream_push_pcm`).
/// - `out_stream`: on `VOKRA_OK`, receives a stream handle freed with
///   `vokra_stream_destroy`.
///
/// Returns `VOKRA_ERROR_NOT_IMPLEMENTED` if the session's model is not a VAD
/// model (no VAD engine was injected).
///
/// # Safety
///
/// `session` must be a valid session handle and `out_stream` a writable
/// `vokra_stream_t*` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_stream_open(
    session: *const vokra_session_t,
    sample_rate: i32,
    out_stream: *mut *mut vokra_stream_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `session` validated (NULL rejected) by `required_ref`.
        let handle = unsafe { ffi_guard::required_ref(session, "session")? };
        ffi_guard::require_out_ptr(out_stream, "out_stream")?;
        if sample_rate <= 0 {
            return Err(error::fail_invalid("`sample_rate` must be positive"));
        }
        // Errors with NotImplemented if the session has no VAD engine — the
        // model/task mismatch signal (ADR-0003 §2). The specific 8k/16k check
        // is enforced by the handle on the first push (kept out of the C layer
        // so it stays model-agnostic).
        let vad = handle
            .session
            .open_vad_stream()
            .map_err(|e| error::fail(&e))?;
        let stream = vokra_stream_t {
            handle: vad,
            sample_rate: sample_rate as u32,
            pending: VecDeque::new(),
        };
        let boxed = handle::into_raw(stream);
        // SAFETY: `out_stream` is non-null (checked).
        unsafe { *out_stream = boxed };
        Ok(())
    })
}

/// Pushes mono `f32` PCM into a VAD stream (FR-API-01).
///
/// The stream buffers each completed frame's speech probability internally;
/// retrieve them with `vokra_stream_poll`. A trailing partial frame is held
/// until the next push.
///
/// # Safety
///
/// `stream` must be a valid stream handle and `pcm` valid for `num_samples`
/// reads (or `NULL` when `num_samples == 0`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_stream_push_pcm(
    stream: *mut vokra_stream_t,
    pcm: *const f32,
    num_samples: usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `stream` validated (NULL rejected) by `required_mut`.
        let s = unsafe { ffi_guard::required_mut(stream, "stream")? };
        // SAFETY: `pcm`/`num_samples` validated by `required_slice`.
        let samples = unsafe { ffi_guard::required_slice(pcm, num_samples, "pcm")? };
        let rate = s.sample_rate;
        let probs = s
            .handle
            .push_pcm(samples, rate)
            .map_err(|e| error::fail(&e))?;
        s.pending.extend(probs);
        Ok(())
    })
}

/// Drains up to `capacity` buffered speech probabilities into `out_probs`
/// (FR-API-01). Non-blocking: writes `*out_count` = number written (0 if none
/// pending). `out_probs` may be `NULL` only when `capacity == 0`.
///
/// # Safety
///
/// `stream` must be a valid stream handle, `out_probs` valid for `capacity`
/// writes (or `NULL` when `capacity == 0`), and `out_count` a writable
/// `size_t` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_stream_poll(
    stream: *mut vokra_stream_t,
    out_probs: *mut f32,
    capacity: usize,
    out_count: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `stream` validated (NULL rejected) by `required_mut`.
        let s = unsafe { ffi_guard::required_mut(stream, "stream")? };
        ffi_guard::require_out_ptr(out_count, "out_count")?;
        let n = capacity.min(s.pending.len());
        if n > 0 {
            ffi_guard::require_out_ptr(out_probs, "out_probs")?;
            // SAFETY: `out_probs` is non-null (checked) and valid for `capacity`
            // writes per the contract; we write only `n <= capacity` elements.
            let dst = unsafe { std::slice::from_raw_parts_mut(out_probs, n) };
            for slot in dst.iter_mut() {
                *slot = s
                    .pending
                    .pop_front()
                    .expect("n was clamped to pending.len()");
            }
        }
        // SAFETY: `out_count` is non-null (checked above).
        unsafe { *out_count = n };
        Ok(())
    })
}

/// Frees a stream handle from `vokra_stream_open`. `NULL` is a no-op.
///
/// # Safety
///
/// `stream` must be `NULL` or a handle from `vokra_stream_open` not already
/// destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_stream_destroy(stream: *mut vokra_stream_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: `stream` is NULL or a live handle from `into_raw`.
        unsafe { handle::drop_raw(stream) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{vokra_session_create_from_file, vokra_session_destroy};
    use std::ffi::CString;
    use std::path::PathBuf;

    use vokra_core::engines::VadEngine;
    use vokra_models::silero_vad::SileroVadV5;
    use vokra_models::silero_vad::wav::read_wav_f32;

    fn parity_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/silero_vad")
    }

    /// Creates a live session over the committed Silero VAD fixture GGUF (the
    /// same asset `c_abi_stream_matches_direct_rust_api` loads). Caller frees it
    /// with `vokra_session_destroy`.
    fn create_silero_session() -> *mut vokra_session_t {
        let gguf = parity_dir().join("silero-vad-v5.gguf");
        let cpath = CString::new(gguf.to_str().unwrap()).unwrap();
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: valid C path and out-pointer.
        let st = unsafe { vokra_session_create_from_file(cpath.as_ptr(), &mut session) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        session
    }

    #[test]
    fn push_pcm_rejects_null_stream() {
        // SAFETY: NULL stream is the rejected branch; len 0 skips pcm deref.
        let status = unsafe { vokra_stream_push_pcm(std::ptr::null_mut(), std::ptr::null(), 0) };
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn stream_open_rejects_null_session() {
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: NULL session is the rejected branch; out_stream is a writable slot.
        let st = unsafe { vokra_stream_open(std::ptr::null(), 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(stream.is_null(), "out_stream stays NULL on the reject path");
    }

    #[test]
    fn stream_open_rejects_nonpositive_sample_rate() {
        // The `sample_rate <= 0` guard runs before `open_vad_stream`, so it must
        // reject a bogus rate even on a valid VAD session.
        let session = create_silero_session();
        for rate in [0, -1] {
            let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
            // SAFETY: valid session/out-pointer; `rate <= 0` is the rejected branch.
            let st = unsafe { vokra_stream_open(session, rate, &mut stream) };
            assert_eq!(
                st,
                vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
                "sample_rate {rate} must be rejected"
            );
            assert!(stream.is_null(), "no handle is written on the reject path");
        }
        // SAFETY: session from `create_silero_session`, freed once.
        unsafe { vokra_session_destroy(session) };
    }

    #[test]
    fn stream_poll_rejects_null_stream() {
        let mut buf = [0.0f32; 4];
        let mut count: usize = 7;
        // SAFETY: NULL stream is the rejected branch; buf/count are writable.
        let st = unsafe {
            vokra_stream_poll(
                std::ptr::null_mut(),
                buf.as_mut_ptr(),
                buf.len(),
                &mut count,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert_eq!(count, 7, "out_count is untouched on the reject path");
    }

    #[test]
    fn stream_poll_rejects_null_out_count() {
        let session = create_silero_session();
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: valid session handle and out-pointer.
        let st = unsafe { vokra_stream_open(session, 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        let mut buf = [0.0f32; 4];
        // SAFETY: valid stream/buffer; NULL out_count is the rejected branch.
        let st =
            unsafe { vokra_stream_poll(stream, buf.as_mut_ptr(), buf.len(), std::ptr::null_mut()) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // SAFETY: handles from create/open, each freed once.
        unsafe {
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
        }
    }

    /// The C ABI push/poll must reproduce the M0-05 Rust API exactly (T09).
    #[test]
    fn c_abi_stream_matches_direct_rust_api() {
        let gguf = parity_dir().join("silero-vad-v5.gguf");
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");
        assert_eq!(wav.sample_rate, 16_000);

        // Reference: the M0-05 Rust API, direct.
        let model = SileroVadV5::open(&gguf).expect("load silero");
        let mut ref_stream = model.open_stream();
        let reference = ref_stream
            .push_pcm(&wav.samples, wav.sample_rate)
            .expect("reference push");
        assert!(!reference.is_empty(), "fixture should yield frames");

        // C ABI: create session -> open stream -> push -> poll.
        let cpath = CString::new(gguf.to_str().unwrap()).unwrap();
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: valid C path and out-pointer.
        let st = unsafe { vokra_session_create_from_file(cpath.as_ptr(), &mut session) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: valid session handle and out-pointer.
        let st = unsafe { vokra_stream_open(session, 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        // SAFETY: valid stream; pcm valid for its length.
        let st = unsafe { vokra_stream_push_pcm(stream, wav.samples.as_ptr(), wav.samples.len()) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        let mut buf = vec![0.0f32; reference.len() + 8];
        let mut count: usize = 0;
        // SAFETY: `buf` is valid for `buf.len()` writes; count is writable.
        let st = unsafe { vokra_stream_poll(stream, buf.as_mut_ptr(), buf.len(), &mut count) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        assert_eq!(count, reference.len(), "frame count matches the Rust API");
        assert_eq!(
            &buf[..count],
            &reference[..],
            "probabilities match bit-for-bit"
        );

        // A second poll with nothing pending returns zero.
        let mut count2: usize = 123;
        // SAFETY: valid stream / buffer / count.
        let st = unsafe { vokra_stream_poll(stream, buf.as_mut_ptr(), buf.len(), &mut count2) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert_eq!(count2, 0);

        // SAFETY: handles from the create/open calls, each freed once.
        unsafe {
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
        }
    }
}
