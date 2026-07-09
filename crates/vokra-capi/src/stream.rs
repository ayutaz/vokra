//! Streaming C ABI: open / push PCM / poll (probabilities or typed events) /
//! interrupt / destroy (M0-09-T09, generalized for M1-08, extended for M3-14).
//!
//! `vokra_stream_open` builds a native stepping [`Stream`](vokra_core::Stream)
//! over the session's VAD engine (Silero VAD = M0-05); `vokra_stream_push_pcm`
//! feeds PCM into the stepper, which emits events into a lock-free SPSC ring
//! (FR-ST-02). `vokra_stream_poll` drains VAD speech probabilities (the f32 fast
//! path) and `vokra_stream_poll_events` drains typed [`vokra_event_t`]s. Both
//! polls are non-blocking. `vokra_stream_interrupt` (M3-14 / FR-ST-03) is the
//! synchronous barge-in: it drains the ring, resets the stepper, and clears
//! the barge-in flag so the next push starts clean. All recurrent state stays
//! hidden inside the stepper (FR-LD-06 / FR-ST-05).

use vokra_core::StreamEvent;

use crate::error::vokra_status_t;
use crate::handle::{self, vokra_session_t, vokra_stream_t};
use crate::{error, ffi_guard};

/// Kind tag of a [`vokra_event_t`] (M1-08). The numeric values are part of the
/// (M0-unstable) ABI.
//
// C-style names so cbindgen emits them verbatim.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum vokra_event_kind_t {
    /// Unknown / reserved event kind (forward-compat placeholder).
    VOKRA_EVENT_UNKNOWN = 0,
    /// A VAD speech-probability event: `a` = frame index, `b` = probability.
    VOKRA_EVENT_SPEECH_PROB = 1,
    /// An ASR token event: `a` = token id, `b` = reserved (`0`).
    VOKRA_EVENT_TOKEN = 2,
}

/// A generalized streaming event drained by `vokra_stream_poll_events` (M1-08).
///
/// A fixed 12-byte POD (`kind` + `a` + `b`); the meaning of `a` / `b` depends on
/// `kind` (see [`vokra_event_kind_t`]). This layout is a new (M0-unstable) ABI
/// surface, pinned by the numeric-layout test.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct vokra_event_t {
    /// Discriminates how `a` / `b` are interpreted.
    pub kind: vokra_event_kind_t,
    /// Primary integer field (frame index for VAD, token id for ASR).
    pub a: u32,
    /// Secondary float field (probability for VAD; reserved `0` for ASR).
    pub b: f32,
}

/// Maps a native [`StreamEvent`] to its C ABI representation.
fn to_c_event(ev: StreamEvent) -> vokra_event_t {
    match ev {
        StreamEvent::SpeechProb { frame_index, prob } => vokra_event_t {
            kind: vokra_event_kind_t::VOKRA_EVENT_SPEECH_PROB,
            a: frame_index,
            b: prob,
        },
        StreamEvent::Token { id, .. } => vokra_event_t {
            kind: vokra_event_kind_t::VOKRA_EVENT_TOKEN,
            a: id,
            b: 0.0,
        },
        // StreamEvent is #[non_exhaustive]; a future kind surfaces as UNKNOWN.
        _ => vokra_event_t {
            kind: vokra_event_kind_t::VOKRA_EVENT_UNKNOWN,
            a: 0,
            b: 0.0,
        },
    }
}

/// Opens a VAD stream over a session (FR-API-01 / FR-ST-02).
///
/// # Parameters
///
/// - `session`: a session created from a Silero VAD GGUF.
/// - `sample_rate`: the stream sample rate in Hz (Silero accepts 8000 or
///   16000; other rates are rejected by the first `vokra_stream_push_pcm`).
/// - `out_stream`: on `VOKRA_OK`, receives a stream handle freed with
///   `vokra_stream_destroy`. The stream retains the session, so it keeps the
///   model alive even after `vokra_session_destroy`.
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
        // NotImplemented if the session has no VAD engine (model/task mismatch,
        // ADR-0003 §2). The 8k/16k check is enforced by the stepper on push.
        let stream = handle
            .session
            .open_vad_stepper(sample_rate as u32)
            .map_err(|e| error::fail(&e))?;
        let boxed = handle::into_raw(vokra_stream_t {
            stream,
            _session: handle.session.clone(),
        });
        // SAFETY: `out_stream` is non-null (checked).
        unsafe { *out_stream = boxed };
        Ok(())
    })
}

/// Pushes mono `f32` PCM into a VAD stream (FR-API-01). Each completed frame's
/// event is enqueued on the stream's ring; drain them with `vokra_stream_poll`
/// or `vokra_stream_poll_events`. A trailing partial frame is held internally.
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
        // `push` returns the number of events enqueued (backpressure signal); the
        // synchronous C caller polls between/after pushes, so we drive it and
        // surface only stepper errors here.
        s.stream.push(samples).map_err(|e| error::fail(&e))?;
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
        let mut n = 0usize;
        if capacity > 0 {
            ffi_guard::require_out_ptr(out_probs, "out_probs")?;
            // SAFETY: `out_probs` is non-null (checked) and valid for `capacity`
            // writes per the contract; we write only indices `< capacity`.
            let dst = unsafe { std::slice::from_raw_parts_mut(out_probs, capacity) };
            while n < capacity {
                match s.stream.poll_one() {
                    Some(StreamEvent::SpeechProb { prob, .. }) => {
                        dst[n] = prob;
                        n += 1;
                    }
                    // A non-VAD event on a VAD stream never occurs; drop it
                    // defensively without consuming an output slot.
                    Some(_) => {}
                    None => break,
                }
            }
        }
        // SAFETY: `out_count` is non-null (checked above).
        unsafe { *out_count = n };
        Ok(())
    })
}

/// Drains up to `capacity` typed events into `out_events` (M1-08). Non-blocking:
/// writes `*out_count` = number written. `out_events` may be `NULL` only when
/// `capacity == 0`. This is the generalized poll; `vokra_stream_poll` is the
/// f32-probability fast path over the same ring.
///
/// # Safety
///
/// `stream` must be a valid stream handle, `out_events` valid for `capacity`
/// writes (or `NULL` when `capacity == 0`), and `out_count` a writable
/// `size_t` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_stream_poll_events(
    stream: *mut vokra_stream_t,
    out_events: *mut vokra_event_t,
    capacity: usize,
    out_count: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `stream` validated (NULL rejected) by `required_mut`.
        let s = unsafe { ffi_guard::required_mut(stream, "stream")? };
        ffi_guard::require_out_ptr(out_count, "out_count")?;
        let mut n = 0usize;
        if capacity > 0 {
            ffi_guard::require_out_ptr(out_events, "out_events")?;
            // SAFETY: `out_events` is non-null (checked) and valid for `capacity`
            // writes per the contract; we write only indices `< capacity`.
            let dst = unsafe { std::slice::from_raw_parts_mut(out_events, capacity) };
            while n < capacity {
                match s.stream.poll_one() {
                    Some(ev) => {
                        dst[n] = to_c_event(ev);
                        n += 1;
                    }
                    None => break,
                }
            }
        }
        // SAFETY: `out_count` is non-null (checked above).
        unsafe { *out_count = n };
        Ok(())
    })
}

/// Barge-in: flushes the current chunk output, drains the stream's ring
/// (when the consumer half is still on this stream), resets the stepper's
/// hidden state, and clears the barge-in flag — all synchronously, so the
/// next `vokra_stream_push_pcm` is accepted in a clean state (M3-14 /
/// FR-ST-03).
///
/// This is the C-ABI counterpart of [`Stream::interrupt`](vokra_core::Stream::interrupt).
/// A bare (no-stepper) stream is a documented no-op that still returns
/// `VOKRA_OK`.
///
/// # Thread safety
///
/// `vokra_stream_interrupt` takes exclusive access to the stream handle for
/// the duration of the call (mirroring the `&mut self` receiver on the Rust
/// API). A C caller that shares one `vokra_stream_t*` across threads MUST
/// serialise access — the underlying barge-in flag is lock-free, but the
/// handler mutates the ring and stepper state. For cross-thread barge-in
/// without ownership of the stream, hold the stream on one thread and drive
/// interrupts through a Rust-side [`InterruptHandle`](vokra_core::InterruptHandle);
/// a dedicated C ABI for the cross-thread handle is a follow-on (M3-16 v0.9
/// ABI changelog).
///
/// # Safety
///
/// `stream` must be a valid stream handle from `vokra_stream_open` and must
/// not be aliased on another thread for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_stream_interrupt(stream: *mut vokra_stream_t) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `stream` validated (NULL rejected) by `required_mut`.
        let s = unsafe { ffi_guard::required_mut(stream, "stream")? };
        s.stream.interrupt().map_err(|e| error::fail(&e))?;
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
    use crate::session::{
        vokra_session_create_from_file, vokra_session_destroy, vokra_session_retain,
    };
    use std::ffi::CString;
    use std::mem::{align_of, offset_of, size_of};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::Barrier;

    use vokra_core::Session;
    use vokra_core::engines::VadEngine;
    use vokra_models::silero_vad::SileroVadV5;
    use vokra_models::silero_vad::wav::read_wav_f32;

    fn parity_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/silero_vad")
    }

    /// Creates a live session over the committed Silero VAD fixture GGUF. Caller
    /// frees it with `vokra_session_destroy`.
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
    fn event_layout_pins_abi() {
        // vokra_event_t is a new (M0-unstable) ABI surface: pin its size, field
        // offsets and the kind discriminants so a reorder fails loudly.
        assert_eq!(size_of::<vokra_event_t>(), 12);
        assert_eq!(align_of::<vokra_event_t>(), 4);
        assert_eq!(offset_of!(vokra_event_t, kind), 0);
        assert_eq!(offset_of!(vokra_event_t, a), 4);
        assert_eq!(offset_of!(vokra_event_t, b), 8);
        assert_eq!(vokra_event_kind_t::VOKRA_EVENT_UNKNOWN as u32, 0);
        assert_eq!(vokra_event_kind_t::VOKRA_EVENT_SPEECH_PROB as u32, 1);
        assert_eq!(vokra_event_kind_t::VOKRA_EVENT_TOKEN as u32, 2);
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
        // SAFETY: NULL session is the rejected branch; out_stream is writable.
        let st = unsafe { vokra_stream_open(std::ptr::null(), 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(stream.is_null(), "out_stream stays NULL on the reject path");
    }

    #[test]
    fn stream_open_rejects_nonpositive_sample_rate() {
        let session = create_silero_session();
        for rate in [0, -1] {
            let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
            // SAFETY: valid session/out-pointer; `rate <= 0` is the reject branch.
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

    /// The reference VAD frame probabilities for the committed 16k fixture, via
    /// the direct M0-05 Rust API.
    fn reference_probs() -> Vec<f32> {
        let gguf = parity_dir().join("silero-vad-v5.gguf");
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");
        assert_eq!(wav.sample_rate, 16_000);
        let model = SileroVadV5::open(&gguf).expect("load silero");
        let mut ref_stream = model.open_stream();
        let probs = ref_stream
            .push_pcm(&wav.samples, wav.sample_rate)
            .expect("reference push");
        assert!(!probs.is_empty(), "fixture should yield frames");
        probs
    }

    /// The C ABI push/poll must reproduce the M0-05 Rust API exactly (T09).
    #[test]
    fn c_abi_stream_matches_direct_rust_api() {
        let reference = reference_probs();
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");

        let session = create_silero_session();
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

    /// `vokra_stream_poll_events` yields the same probabilities as the f32 poll,
    /// now as typed `VOKRA_EVENT_SPEECH_PROB` events with monotonic frame ids.
    #[test]
    fn poll_events_matches_the_f32_poll() {
        let reference = reference_probs();
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");

        let session = create_silero_session();
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: valid session/out-pointer.
        let st = unsafe { vokra_stream_open(session, 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        // SAFETY: valid stream; pcm valid for its length.
        let st = unsafe { vokra_stream_push_pcm(stream, wav.samples.as_ptr(), wav.samples.len()) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        let mut events = vec![
            vokra_event_t {
                kind: vokra_event_kind_t::VOKRA_EVENT_UNKNOWN,
                a: 0,
                b: 0.0,
            };
            reference.len() + 8
        ];
        let mut count: usize = 0;
        // SAFETY: `events` valid for its length; count writable.
        let st = unsafe {
            vokra_stream_poll_events(stream, events.as_mut_ptr(), events.len(), &mut count)
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert_eq!(count, reference.len());
        for (i, ev) in events[..count].iter().enumerate() {
            assert_eq!(ev.kind, vokra_event_kind_t::VOKRA_EVENT_SPEECH_PROB);
            assert_eq!(ev.a as usize, i, "monotonic frame index");
            assert_eq!(ev.b.to_bits(), reference[i].to_bits(), "prob bit-for-bit");
        }

        // SAFETY: handles freed once each.
        unsafe {
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
        }
    }

    /// `vokra_session_retain` is the atomic ref count: the model outlives a
    /// destroy of the original handle as long as a retained handle is alive.
    #[test]
    fn retain_keeps_model_alive_after_original_destroy() {
        let session = create_silero_session();
        let mut retained: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: valid session; out-pointer writable.
        let st = unsafe { vokra_session_retain(session, &mut retained) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(!retained.is_null());

        // Drop the ORIGINAL; the model must stay alive for the retained handle.
        // SAFETY: original handle, freed once.
        unsafe { vokra_session_destroy(session) };

        // Streaming through the retained handle still works.
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: retained is a valid handle; out-pointer writable.
        let st = unsafe { vokra_stream_open(retained, 16_000, &mut stream) };
        assert_eq!(
            st,
            vokra_status_t::VOKRA_OK,
            "model alive via retained handle"
        );

        // SAFETY: handles freed once each.
        unsafe {
            vokra_stream_destroy(stream);
            vokra_session_destroy(retained);
        }
    }

    /// A live stream keeps the model alive after `vokra_session_destroy` (the
    /// stream retains its own session clone).
    #[test]
    fn stream_outlives_session_destroy() {
        let reference = reference_probs();
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");

        let session = create_silero_session();
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: valid session/out-pointer.
        let st = unsafe { vokra_stream_open(session, 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        // Destroy the session while the stream is still open.
        // SAFETY: session handle, freed once.
        unsafe { vokra_session_destroy(session) };

        // Push/poll on the now-orphaned stream still works and matches.
        // SAFETY: valid stream; pcm valid for its length.
        let st = unsafe { vokra_stream_push_pcm(stream, wav.samples.as_ptr(), wav.samples.len()) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        let mut buf = vec![0.0f32; reference.len() + 8];
        let mut count: usize = 0;
        // SAFETY: buf valid for its length; count writable.
        let st = unsafe { vokra_stream_poll(stream, buf.as_mut_ptr(), buf.len(), &mut count) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert_eq!(count, reference.len());
        assert_eq!(&buf[..count], &reference[..]);

        // SAFETY: stream handle, freed once.
        unsafe { vokra_stream_destroy(stream) };
    }

    /// M3-14 T08 — the C ABI `vokra_stream_interrupt` behaves like
    /// `Stream::interrupt`: after a prime push + interrupt, the next poll is
    /// empty (ring drained) AND the frame-index counter restarts (stepper
    /// reset).
    #[test]
    fn stream_interrupt_drains_ring_and_resets_stepper() {
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");

        let session = create_silero_session();
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: valid session/out-pointer.
        let st = unsafe { vokra_stream_open(session, 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        // Prime the ring: push enough PCM to produce at least a few frames but
        // do NOT poll — the frames sit on the ring until the interrupt drains
        // them.
        let prime_len = wav.samples.len().min(4096);
        // SAFETY: valid stream; pcm slice valid for `prime_len`.
        let st = unsafe { vokra_stream_push_pcm(stream, wav.samples.as_ptr(), prime_len) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        // Interrupt.
        // SAFETY: valid stream handle.
        let st = unsafe { vokra_stream_interrupt(stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK, "interrupt returns VOKRA_OK");

        // Ring is empty right after the interrupt.
        let mut probs = [0.0f32; 32];
        let mut count: usize = 123;
        // SAFETY: valid stream/buf/count.
        let st = unsafe { vokra_stream_poll(stream, probs.as_mut_ptr(), probs.len(), &mut count) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert_eq!(count, 0, "ring drained by interrupt");

        // A subsequent push processes new PCM cleanly and yields events again.
        // SAFETY: valid stream/pcm.
        let st = unsafe { vokra_stream_push_pcm(stream, wav.samples.as_ptr(), prime_len) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        let mut probs2 = vec![0.0f32; wav.samples.len()];
        let mut count2: usize = 0;
        // SAFETY: valid stream/buf/count.
        let st =
            unsafe { vokra_stream_poll(stream, probs2.as_mut_ptr(), probs2.len(), &mut count2) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(
            count2 > 0,
            "post-interrupt push still produces frames (stepper is functional)"
        );

        // SAFETY: handles freed exactly once.
        unsafe {
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
        }
    }

    /// M3-14 T08 — NULL stream is rejected by the argument guard, mirroring
    /// the other C-ABI stream calls.
    #[test]
    fn stream_interrupt_rejects_null_stream() {
        // SAFETY: NULL is the rejected branch; no deref happens.
        let st = unsafe { vokra_stream_interrupt(std::ptr::null_mut()) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    /// M3-14 T08 — interrupt on a live stream that had no pending events is a
    /// benign no-op that still returns `VOKRA_OK` (matches the Rust API
    /// contract: `Stream::interrupt` is a documented no-op on empty state).
    #[test]
    fn stream_interrupt_on_empty_stream_is_ok() {
        let session = create_silero_session();
        let mut stream: *mut vokra_stream_t = std::ptr::null_mut();
        // SAFETY: valid session/out-pointer.
        let st = unsafe { vokra_stream_open(session, 16_000, &mut stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        // Interrupt with nothing pending.
        // SAFETY: valid handle.
        let st = unsafe { vokra_stream_interrupt(stream) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);

        // SAFETY: handles freed exactly once.
        unsafe {
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
        }
    }

    /// Retain N times then destroy every handle: no panic / double-free (the
    /// `Guard`-style exactly-once free is proven in `handle.rs`; here we assert
    /// the refcount survives many retains and releases cleanly).
    #[test]
    fn retain_then_destroy_all_is_clean() {
        let session = create_silero_session();
        let mut handles = vec![session];
        for _ in 0..8 {
            let mut extra: *mut vokra_session_t = std::ptr::null_mut();
            // SAFETY: `session` (handles[0]) valid; out-pointer writable.
            let st = unsafe { vokra_session_retain(handles[0], &mut extra) };
            assert_eq!(st, vokra_status_t::VOKRA_OK);
            handles.push(extra);
        }
        // Destroy in reverse; only the last release frees the model.
        for h in handles.into_iter().rev() {
            // SAFETY: each handle came from create/retain, freed exactly once.
            unsafe { vokra_session_destroy(h) };
        }
    }

    /// The FR-ST-02 cross-thread split over the REAL Silero model: the producer
    /// `Stream` pushes the fixture in irregular chunks on a spawned thread while
    /// the main thread polls an `EventPoller`; the drained probabilities must
    /// equal the single-threaded reference bit-for-bit (deterministic — the
    /// threads only move data; chunk-invariance makes the arithmetic identical).
    /// Exercised at the Rust API level (no unsafe C-handle aliasing across
    /// threads).
    #[test]
    fn cross_thread_push_poll_over_real_model() {
        let reference = reference_probs();
        let gguf = parity_dir().join("silero-vad-v5.gguf");
        let wav = read_wav_f32(parity_dir().join("test_16k.wav")).expect("read fixture wav");

        // Build a VAD session directly and open a stepping stream.
        let base = Session::from_file(&gguf).build().expect("session builds");
        let vad = SileroVadV5::from_gguf(base.gguf()).expect("silero from gguf");
        let session = base.with_vad_engine(Arc::new(vad));
        let mut stream = session.open_vad_stepper(16_000).expect("vad stepper");
        let mut poller = stream.take_poller().expect("poller");

        let samples = wav.samples;
        let total = samples.len();
        let barrier = Arc::new(Barrier::new(2));
        let b2 = Arc::clone(&barrier);

        let producer = std::thread::spawn(move || {
            b2.wait();
            let chunks = [128usize, 500, 37, 4096, 1000];
            let mut off = 0;
            let mut ci = 0;
            while off < total {
                let take = chunks[ci % chunks.len()].min(total - off);
                ci += 1;
                stream.push(&samples[off..off + take]).expect("push");
                off += take;
            }
            stream // keep alive until join
        });

        barrier.wait();
        let mut got: Vec<f32> = Vec::with_capacity(reference.len());
        let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 32];
        // The producer emits exactly `reference.len()` frames; poll until we have
        // them all (the last chunk's frames may arrive after the join point, so
        // keep the producer alive via the returned Stream until we finish).
        let _stream = loop {
            let m = poller.poll(&mut buf);
            for e in &buf[..m] {
                if let StreamEvent::SpeechProb { prob, .. } = e {
                    got.push(*prob);
                }
            }
            if got.len() >= reference.len() {
                break producer.join().expect("producer joins");
            }
        };

        assert_eq!(got.len(), reference.len());
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(g.to_bits(), r.to_bits(), "frame {i} prob matches reference");
        }
    }
}
