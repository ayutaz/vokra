//! `vokra_aec_*` — acoustic echo cancellation C ABI (M4-03, FR-OP-60).
//!
//! Wraps `vokra_ops::Aec` (the SpeexDSP MDF/AUMDF float-build port) plus the
//! sample-clock far-end reference queue of `vokra_core::stream::aec_ref`.
//!
//! # Handle model (ADR M4-03 §D-(j): writer split off from day one)
//!
//! [`vokra_aec_create`] hands out **two** owning handles:
//!
//! - `vokra_aec_ref_writer_t` — the far-end (playback) side. Call
//!   `vokra_aec_ref_push` from the **playback callback thread** with each
//!   chunk the app sends to the speaker, tagged with its absolute playback
//!   sample position.
//! - `vokra_aec_t` — the canceller + the queue's reader half. Call
//!   `vokra_aec_process` / `vokra_aec_reset` from the **inference thread**.
//!
//! The two handles may be used **concurrently from different threads** (the
//! queue between them is a lock-free SPSC ring — that is the point of the
//! split; the M3-14 `vokra_stream_interrupt` follow-on taught us to design
//! the cross-thread surface up front). What stays forbidden is concurrent
//! use of the *same* handle from two threads, and any use after its destroy.
//! Destroy order between the two handles is free; each `*_destroy` is
//! `NULL`-tolerant and final (ADR-0003 §3-a).
//!
//! Everything else follows the house C ABI rules: opaque handles,
//! `vokra_status_t` returns + thread-local `vokra_last_error()`, panics
//! never cross the boundary ([`crate::ffi_guard`]), no locale-sensitive
//! number parsing.

use std::ffi::c_float;

use vokra_core::stream::{AecRefReader, AecRefWriter, aec_ref_queue};
use vokra_ops::{Aec, AecAttrs, AecStatus};

use crate::error::{fail, fail_invalid, vokra_status_t};
use crate::ffi_guard::{guard, guard_void, require_out_ptr, required_mut, required_ref};
use crate::handle::{drop_raw, into_raw};

/// Construction parameters for [`vokra_aec_create`].
///
/// `sample_rate` / `frame_size` / `filter_length` follow the SpeexDSP
/// guidance (speex_echo.h: a frame of 10-20 ms; a tail of 100-500 ms).
/// `frame_size` must be even. `ref_queue_capacity_samples` sizes the far-end
/// queue; `0` selects the documented default of `8 * filter_length` samples
/// (rounded up to a power of two).
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct vokra_aec_config_t {
    /// Sample rate of mic and far-end PCM (must match on both sides).
    pub sample_rate: u32,
    /// Samples per `vokra_aec_process` call (even, > 0).
    pub frame_size: usize,
    /// Echo tail length in samples (>= frame_size).
    pub filter_length: usize,
    /// Far-end queue capacity in samples; 0 = default (8 * filter_length).
    pub ref_queue_capacity_samples: usize,
}

/// Per-frame outcome reported by [`vokra_aec_process`] (mirrors
/// `vokra_ops::AecStatus`; FR-EX-08 — degraded modes are visible, never
/// silent).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum vokra_aec_status_t {
    /// Far-end window fully covered; the canceller ran normally.
    VOKRA_AEC_CANCELLED = 0,
    /// Nothing is playing and the far-end history is silent: the mic frame
    /// was copied through bit-exactly (state frozen).
    VOKRA_AEC_PASS_THROUGH = 1,
    /// Part (or all) of the far-end window had no data and was zero-filled;
    /// the live echo tail keeps being cancelled. The missing sample count is
    /// reported through `out_missing`.
    VOKRA_AEC_PARTIAL_REFERENCE = 2,
    /// The divergence guard fired and the canceller reset itself this frame.
    VOKRA_AEC_RESET = 3,
}

/// Opaque AEC handle: the canceller state plus the reader half of the
/// far-end queue. Owned by the inference thread. Opaque to C.
#[allow(non_camel_case_types)]
pub struct vokra_aec_t {
    aec: Aec,
    reader: AecRefReader,
}

/// Opaque far-end writer handle: the producer half of the far-end queue.
/// Owned by the playback-callback thread. Opaque to C.
#[allow(non_camel_case_types)]
pub struct vokra_aec_ref_writer_t {
    writer: AecRefWriter,
}

/// Creates an echo canceller and its far-end reference writer.
///
/// On success writes one handle through each out-pointer; on failure both
/// out-pointers are left untouched. The two handles have independent
/// lifetimes (destroy each with its own `*_destroy`, in either order).
///
/// # Safety
///
/// `config` must be `NULL` or point to a valid [`vokra_aec_config_t`];
/// `out_aec` / `out_writer` must be valid non-`NULL` out-pointers. `NULL`
/// `config` is rejected as `VOKRA_ERROR_INVALID_ARGUMENT` (never
/// dereferenced).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_aec_create(
    config: *const vokra_aec_config_t,
    out_aec: *mut *mut vokra_aec_t,
    out_writer: *mut *mut vokra_aec_ref_writer_t,
) -> vokra_status_t {
    guard(|| {
        // SAFETY: NULL-checked borrow per the function contract.
        let config = unsafe { required_ref(config, "config") }?;
        require_out_ptr(out_aec, "out_aec")?;
        require_out_ptr(out_writer, "out_writer")?;

        let attrs = AecAttrs {
            sample_rate: config.sample_rate,
            frame_size: config.frame_size,
            filter_length: config.filter_length,
        };
        let aec = Aec::new(&attrs).map_err(|e| fail(&e))?;
        let capacity = if config.ref_queue_capacity_samples == 0 {
            8 * config.filter_length
        } else {
            config.ref_queue_capacity_samples
        };
        let (writer, reader) = aec_ref_queue(capacity, config.sample_rate).map_err(|e| fail(&e))?;

        // SAFETY: both out-pointers were verified non-NULL above; writing a
        // fresh heap pointer through them is the documented contract.
        unsafe {
            *out_aec = into_raw(vokra_aec_t { aec, reader });
            *out_writer = into_raw(vokra_aec_ref_writer_t { writer });
        }
        Ok(())
    })
}

/// Pushes a far-end (playback) chunk whose first sample plays at absolute
/// sample position `playback_pos`, writing the number of samples accepted to
/// `out_accepted`.
///
/// Reject-on-full backpressure: when the queue cannot take the whole chunk,
/// only the fitting prefix is accepted (`*out_accepted < num_samples`,
/// possibly 0); retry the remainder at `playback_pos + accepted` after the
/// inference thread has consumed a frame. `out_accepted` is mandatory so a
/// partial accept is never silent (FR-EX-08).
///
/// Backward / overlapping `playback_pos` tags are an explicit
/// `VOKRA_ERROR_INVALID_ARGUMENT` (time tags are monotonic; a forward gap is
/// legal and reads back as silence).
///
/// # Safety
///
/// `writer` must be a live handle from [`vokra_aec_create`], used by one
/// thread at a time (the playback callback thread). `pcm` must be `NULL`
/// only when `num_samples == 0`, otherwise valid for `num_samples` reads.
/// `out_accepted` must be a valid non-`NULL` out-pointer. May run
/// concurrently with `vokra_aec_process` on the paired `vokra_aec_t`
/// (SPSC queue), but not with itself on the same handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_aec_ref_push(
    writer: *mut vokra_aec_ref_writer_t,
    pcm: *const c_float,
    num_samples: usize,
    playback_pos: u64,
    out_accepted: *mut usize,
) -> vokra_status_t {
    guard(|| {
        // SAFETY: NULL-checked unique borrow per the function contract.
        let writer = unsafe { required_mut(writer, "writer") }?;
        require_out_ptr(out_accepted, "out_accepted")?;
        // SAFETY: `pcm` is valid for `num_samples` reads per the contract;
        // NULL is accepted only for the zero-length case.
        let pcm = unsafe { crate::ffi_guard::required_slice(pcm, num_samples, "pcm") }?;
        let accepted = writer
            .writer
            .push(pcm, playback_pos)
            .map_err(|e| fail(&e))?;
        // SAFETY: `out_accepted` verified non-NULL above.
        unsafe {
            *out_accepted = accepted;
        }
        Ok(())
    })
}

/// Cancels the echo of one mic frame (`num_samples == frame_size`) whose
/// first sample was captured at absolute sample position `mic_pos` on the
/// same clock the far-end pushes use. Writes `frame_size` cancelled samples
/// to `out` and the per-frame outcome ([`vokra_aec_status_t`]) to
/// `out_status`; when the outcome is `VOKRA_AEC_PARTIAL_REFERENCE`, the
/// number of zero-filled far-end samples is written to `out_missing`
/// (optional pointer — pass `NULL` if not needed; it is set to 0 for the
/// other outcomes).
///
/// # Safety
///
/// `aec` must be a live handle from [`vokra_aec_create`], used by one
/// thread at a time (the inference thread). `mic` must be valid for
/// `num_samples` reads and `out` for `num_samples` writes; `out_status`
/// must be a valid non-`NULL` out-pointer; `out_missing` may be `NULL`.
/// May run concurrently with `vokra_aec_ref_push` on the paired writer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_aec_process(
    aec: *mut vokra_aec_t,
    mic: *const c_float,
    mic_pos: u64,
    out: *mut c_float,
    num_samples: usize,
    out_status: *mut vokra_aec_status_t,
    out_missing: *mut usize,
) -> vokra_status_t {
    guard(|| {
        // SAFETY: NULL-checked unique borrow per the function contract.
        let handle = unsafe { required_mut(aec, "aec") }?;
        require_out_ptr(out_status, "out_status")?;
        // SAFETY: `mic` is valid for `num_samples` reads per the contract.
        let mic = unsafe { crate::ffi_guard::required_slice(mic, num_samples, "mic") }?;
        if out.is_null() {
            return Err(fail_invalid("argument `out` must not be NULL"));
        }
        if num_samples == 0 {
            return Err(fail_invalid(
                "argument `num_samples` must equal the configured frame_size (got 0)",
            ));
        }
        // SAFETY: `out` is non-NULL and valid for `num_samples` writes per
        // the contract.
        let out = unsafe { std::slice::from_raw_parts_mut(out, num_samples) };

        let status = handle
            .aec
            .process(mic, mic_pos, &mut handle.reader, out)
            .map_err(|e| fail(&e))?;
        let (code, missing) = match status {
            AecStatus::Cancelled => (vokra_aec_status_t::VOKRA_AEC_CANCELLED, 0),
            AecStatus::PassThrough => (vokra_aec_status_t::VOKRA_AEC_PASS_THROUGH, 0),
            AecStatus::PartialReference { missing } => {
                (vokra_aec_status_t::VOKRA_AEC_PARTIAL_REFERENCE, missing)
            }
            AecStatus::Reset => (vokra_aec_status_t::VOKRA_AEC_RESET, 0),
        };
        // SAFETY: `out_status` verified non-NULL above.
        unsafe {
            *out_status = code;
        }
        if !out_missing.is_null() {
            // SAFETY: non-NULL `out_missing` is a valid out-pointer per the
            // contract.
            unsafe {
                *out_missing = missing;
            }
        }
        Ok(())
    })
}

/// Resets the canceller to its as-new state (bit-exact with a fresh
/// [`vokra_aec_create`] of the same config). Pair it with barge-in
/// (`vokra_stream_interrupt`) in full-duplex loops.
///
/// # Safety
///
/// `aec` must be a live handle, used by one thread at a time.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_aec_reset(aec: *mut vokra_aec_t) -> vokra_status_t {
    guard(|| {
        // SAFETY: NULL-checked unique borrow per the function contract.
        let handle = unsafe { required_mut(aec, "aec") }?;
        handle.aec.reset();
        Ok(())
    })
}

/// Destroys an AEC handle. `NULL` is a no-op. The paired writer handle stays
/// valid (its pushes simply pile up unread) and must be destroyed with
/// [`vokra_aec_ref_writer_destroy`].
///
/// # Safety
///
/// `aec` must be `NULL` or a live handle from [`vokra_aec_create`] not used
/// after this call (and not concurrently with it).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_aec_destroy(aec: *mut vokra_aec_t) {
    guard_void(|| {
        // SAFETY: per the contract `aec` is NULL or a live Box from create.
        unsafe { drop_raw(aec) }
    });
}

/// Destroys a far-end writer handle. `NULL` is a no-op. The paired AEC
/// handle stays valid (its windows read as silence once the queue drains —
/// the pass-through semantics take over).
///
/// # Safety
///
/// `writer` must be `NULL` or a live handle from [`vokra_aec_create`] not
/// used after this call (and not concurrently with it).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_aec_ref_writer_destroy(writer: *mut vokra_aec_ref_writer_t) {
    guard_void(|| {
        // SAFETY: per the contract `writer` is NULL or a live Box from create.
        unsafe { drop_raw(writer) }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::vokra_last_error;
    use std::ffi::CStr;

    fn config() -> vokra_aec_config_t {
        vokra_aec_config_t {
            sample_rate: 16_000,
            frame_size: 64,
            filter_length: 256,
            ref_queue_capacity_samples: 0, // default = 8 * filter_length
        }
    }

    fn create(cfg: &vokra_aec_config_t) -> (*mut vokra_aec_t, *mut vokra_aec_ref_writer_t) {
        let mut aec: *mut vokra_aec_t = std::ptr::null_mut();
        let mut writer: *mut vokra_aec_ref_writer_t = std::ptr::null_mut();
        // SAFETY: valid config reference and out-pointers.
        let st = unsafe { vokra_aec_create(cfg, &mut aec, &mut writer) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(!aec.is_null());
        assert!(!writer.is_null());
        (aec, writer)
    }

    fn last_error_contains(needle: &str) {
        let ptr = vokra_last_error();
        assert!(!ptr.is_null(), "an error message must be recorded");
        // SAFETY: vokra_last_error returns a live thread-local C string.
        let msg = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert!(
            msg.contains(needle),
            "last error {msg:?} must mention {needle:?}"
        );
    }

    /// T13: the full create → push → process → destroy round trip, with the
    /// echo actually shrinking (an e2e sanity of the C surface, not a
    /// numerical gate — those live in parity_aec.rs).
    #[test]
    fn round_trip_cancels_echo() {
        let cfg = config();
        let n = cfg.frame_size;
        let (aec, writer) = create(&cfg);

        // Deterministic far-end noise; near-end = simple 2-tap echo.
        let frames = 200usize;
        let mut far = vec![0.0f32; frames * n];
        let mut state = 0x1234_5678_9abc_def0u64;
        for v in far.iter_mut() {
            // SplitMix64 step (test-local).
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let r = (z ^ (z >> 31)) as f64 / u64::MAX as f64;
            *v = (r as f32 - 0.5) * 0.5;
        }
        let mut near = vec![0.0f32; frames * n];
        for i in 0..near.len() {
            near[i] = 0.5 * far[i.saturating_sub(2)] - 0.25 * far[i.saturating_sub(9)];
        }

        let mut out = vec![0.0f32; n];
        let mut early = 0.0f64;
        let mut late = 0.0f64;
        for f in 0..frames {
            let pos = (f * n) as u64;
            let mut accepted = 0usize;
            // SAFETY: live writer handle + valid slice + out-pointer.
            let st =
                unsafe { vokra_aec_ref_push(writer, far[f * n..].as_ptr(), n, pos, &mut accepted) };
            assert_eq!(st, vokra_status_t::VOKRA_OK);
            assert_eq!(accepted, n);

            let mut status = vokra_aec_status_t::VOKRA_AEC_RESET;
            let mut missing = usize::MAX;
            // SAFETY: live aec handle + valid mic/out slices + out-pointers.
            let st = unsafe {
                vokra_aec_process(
                    aec,
                    near[f * n..].as_ptr(),
                    pos,
                    out.as_mut_ptr(),
                    n,
                    &mut status,
                    &mut missing,
                )
            };
            assert_eq!(st, vokra_status_t::VOKRA_OK);
            assert_eq!(status, vokra_aec_status_t::VOKRA_AEC_CANCELLED);
            assert_eq!(missing, 0);

            let e: f64 = out.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            if (10..40).contains(&f) {
                early += e;
            }
            if (160..200).contains(&f) {
                late += e;
            }
        }
        assert!(
            late < 0.5 * early,
            "echo must shrink through the C surface: early {early:e} late {late:e}"
        );

        // SAFETY: handles are live and not used after these calls.
        unsafe {
            vokra_aec_destroy(aec);
            vokra_aec_ref_writer_destroy(writer);
        }
    }

    #[test]
    fn null_arguments_are_invalid_not_crashes() {
        let cfg = config();
        let n = cfg.frame_size;
        let (aec, writer) = create(&cfg);
        let mic = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let mut status = vokra_aec_status_t::VOKRA_AEC_CANCELLED;
        let mut accepted = 0usize;

        // NULL config / out-pointers on create.
        let mut a: *mut vokra_aec_t = std::ptr::null_mut();
        let mut w: *mut vokra_aec_ref_writer_t = std::ptr::null_mut();
        // SAFETY: NULL config is the rejected branch (never dereferenced).
        let st = unsafe { vokra_aec_create(std::ptr::null(), &mut a, &mut w) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        last_error_contains("config");

        // NULL writer handle.
        // SAFETY: NULL handle is the rejected branch.
        let st =
            unsafe { vokra_aec_ref_push(std::ptr::null_mut(), mic.as_ptr(), n, 0, &mut accepted) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // NULL pcm with non-zero length.
        // SAFETY: NULL pcm with len > 0 is the rejected branch.
        let st = unsafe { vokra_aec_ref_push(writer, std::ptr::null(), n, 0, &mut accepted) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // NULL out_accepted (mandatory: partial accepts must be visible).
        // SAFETY: NULL out-pointer is the rejected branch.
        let st = unsafe { vokra_aec_ref_push(writer, mic.as_ptr(), n, 0, std::ptr::null_mut()) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        last_error_contains("out_accepted");

        // NULL aec handle / NULL out / NULL out_status on process.
        // SAFETY: rejected branches; no deref of the NULLs.
        let st = unsafe {
            vokra_aec_process(
                std::ptr::null_mut(),
                mic.as_ptr(),
                0,
                out.as_mut_ptr(),
                n,
                &mut status,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        // SAFETY: as above.
        let st = unsafe {
            vokra_aec_process(
                aec,
                mic.as_ptr(),
                0,
                std::ptr::null_mut(),
                n,
                &mut status,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        last_error_contains("out");
        // SAFETY: as above.
        let st = unsafe {
            vokra_aec_process(
                aec,
                mic.as_ptr(),
                0,
                out.as_mut_ptr(),
                n,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        last_error_contains("out_status");

        // NULL reset handle.
        // SAFETY: rejected branch.
        let st = unsafe { vokra_aec_reset(std::ptr::null_mut()) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // SAFETY: live handles, destroyed exactly once.
        unsafe {
            vokra_aec_destroy(aec);
            vokra_aec_ref_writer_destroy(writer);
        }
    }

    #[test]
    fn invalid_config_and_lengths_are_explicit_errors() {
        // Zero rate.
        let mut bad = config();
        bad.sample_rate = 0;
        let mut a: *mut vokra_aec_t = std::ptr::null_mut();
        let mut w: *mut vokra_aec_ref_writer_t = std::ptr::null_mut();
        // SAFETY: valid config reference and out-pointers.
        let st = unsafe { vokra_aec_create(&bad, &mut a, &mut w) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(
            a.is_null() && w.is_null(),
            "failure leaves out-params untouched"
        );

        // Odd frame size.
        let mut bad = config();
        bad.frame_size = 63;
        // SAFETY: as above.
        let st = unsafe { vokra_aec_create(&bad, &mut a, &mut w) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // Wrong frame length on process (num_samples != frame_size).
        let cfg = config();
        let (aec, writer) = create(&cfg);
        let mic = vec![0.0f32; cfg.frame_size / 2];
        let mut out = vec![0.0f32; cfg.frame_size / 2];
        let mut status = vokra_aec_status_t::VOKRA_AEC_CANCELLED;
        // SAFETY: live handle; the length mismatch is the rejected branch.
        let st = unsafe {
            vokra_aec_process(
                aec,
                mic.as_ptr(),
                0,
                out.as_mut_ptr(),
                mic.len(),
                &mut status,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // Zero num_samples.
        // SAFETY: zero-length branch, no deref.
        let st = unsafe {
            vokra_aec_process(
                aec,
                mic.as_ptr(),
                0,
                out.as_mut_ptr(),
                0,
                &mut status,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // Backward playback_pos tag.
        let pcm = vec![0.0f32; cfg.frame_size];
        let mut accepted = 0usize;
        // SAFETY: live writer + valid slice.
        let st =
            unsafe { vokra_aec_ref_push(writer, pcm.as_ptr(), pcm.len(), 1000, &mut accepted) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        // SAFETY: as above; the backward tag is the rejected branch.
        let st = unsafe { vokra_aec_ref_push(writer, pcm.as_ptr(), pcm.len(), 500, &mut accepted) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // SAFETY: live handles, destroyed exactly once.
        unsafe {
            vokra_aec_destroy(aec);
            vokra_aec_ref_writer_destroy(writer);
        }
    }

    #[test]
    fn destroy_null_is_a_noop() {
        // SAFETY: NULL destroy is the documented no-op.
        unsafe {
            vokra_aec_destroy(std::ptr::null_mut());
            vokra_aec_ref_writer_destroy(std::ptr::null_mut());
        }
    }

    /// T13: writer and aec handles run concurrently from two threads (the
    /// designed use: playback callback pushes, inference thread processes).
    #[test]
    fn concurrent_push_and_process_from_two_threads() {
        let cfg = config();
        let n = cfg.frame_size;
        let (aec, writer) = create(&cfg);
        let frames = 400usize;

        // Raw pointers are not Send; wrap them for the test threads (each
        // handle is used by exactly one thread — the documented contract).
        struct SendPtr<T>(*mut T);
        // SAFETY: test-scoped wrapper; each pointer crosses to exactly one
        // thread and is used only there.
        unsafe impl<T> Send for SendPtr<T> {}

        let w = SendPtr(writer);
        let pusher = std::thread::spawn(move || {
            let w = w;
            let chunk = [0.25f32; 32];
            let mut pos = 0u64;
            for _ in 0..frames * 2 {
                let mut accepted = 0usize;
                // SAFETY: this thread is the sole user of the writer handle.
                let st = unsafe {
                    vokra_aec_ref_push(w.0, chunk.as_ptr(), chunk.len(), pos, &mut accepted)
                };
                assert_eq!(st, vokra_status_t::VOKRA_OK);
                pos += accepted as u64;
                if accepted < chunk.len() {
                    std::thread::yield_now();
                }
            }
            w
        });

        let a = SendPtr(aec);
        let processor = std::thread::spawn(move || {
            let a = a;
            let mic = vec![0.1f32; n];
            let mut out = vec![0.0f32; n];
            let mut status = vokra_aec_status_t::VOKRA_AEC_CANCELLED;
            for f in 0..frames {
                // SAFETY: this thread is the sole user of the aec handle.
                let st = unsafe {
                    vokra_aec_process(
                        a.0,
                        mic.as_ptr(),
                        (f * n) as u64,
                        out.as_mut_ptr(),
                        n,
                        &mut status,
                        std::ptr::null_mut(),
                    )
                };
                assert_eq!(st, vokra_status_t::VOKRA_OK);
            }
            a
        });

        let w = pusher.join().expect("pusher joins");
        let a = processor.join().expect("processor joins");
        // SAFETY: both threads are done; destroy exactly once.
        unsafe {
            vokra_aec_destroy(a.0);
            vokra_aec_ref_writer_destroy(w.0);
        }
    }

    #[test]
    fn reset_returns_ok_on_live_handle() {
        let cfg = config();
        let (aec, writer) = create(&cfg);
        // SAFETY: live handle.
        let st = unsafe { vokra_aec_reset(aec) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        // SAFETY: live handles, destroyed exactly once.
        unsafe {
            vokra_aec_destroy(aec);
            vokra_aec_ref_writer_destroy(writer);
        }
    }
}
