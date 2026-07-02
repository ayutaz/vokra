//! TTS synthesis and PCM buffer ownership (M0-09-T08).
//!
//! [`vokra_tts_synthesize`] delegates to the injected [`TtsEngine`] via
//! `session.tts().synthesize()` — the **piper-plus native** MB-iSTFT-VITS2 path
//! (M0-07); the end-to-end route contains no onnxruntime (FR-LD-05, ADR-0002).
//! The returned PCM is a Vokra-owned buffer freed with [`vokra_audio_free`].
//! Calling it on a non-TTS model returns `VOKRA_ERROR_NOT_IMPLEMENTED`.

use std::ffi::c_char;

use crate::error::vokra_status_t;
use crate::handle::vokra_session_t;
use crate::{error, ffi_guard};

/// Synthesizes speech PCM from UTF-8 text using the session's TTS engine
/// (FR-API-01, piper-plus native path).
///
/// # Parameters
///
/// - `session`: a session created from a piper-plus voice GGUF.
/// - `text_utf8`: NUL-terminated UTF-8 text to synthesize.
/// - `out_pcm`: on `VOKRA_OK`, receives a Vokra-owned mono `f32` buffer (values
///   in `[-1, 1]`) to be freed with `vokra_audio_free(*out_pcm, *out_num_samples)`.
/// - `out_num_samples`: receives the sample count of `*out_pcm`.
/// - `out_sample_rate`: receives the output sample rate in Hz.
///
/// All three out-params are written only on `VOKRA_OK`.
///
/// # Safety
///
/// `session` must be a valid session handle, `text_utf8` a valid C string, and
/// the three out-pointers writable locations of the matching type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_tts_synthesize(
    session: *const vokra_session_t,
    text_utf8: *const c_char,
    out_pcm: *mut *mut f32,
    out_num_samples: *mut usize,
    out_sample_rate: *mut i32,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `session` validated (NULL rejected) by `required_ref`.
        let handle = unsafe { ffi_guard::required_ref(session, "session")? };
        // SAFETY: `text_utf8` validated (NULL / non-UTF-8 rejected).
        let text = unsafe { ffi_guard::required_str(text_utf8, "text_utf8")? };
        ffi_guard::require_out_ptr(out_pcm, "out_pcm")?;
        ffi_guard::require_out_ptr(out_num_samples, "out_num_samples")?;
        ffi_guard::require_out_ptr(out_sample_rate, "out_sample_rate")?;

        let audio = handle
            .session
            .tts()
            .synthesize(text)
            .map_err(|e| error::fail(&e))?;
        let sample_rate = i32::try_from(audio.sample_rate)
            .map_err(|_| error::fail_invalid("model sample rate overflows int32_t"))?;

        // Hand the samples to C as an exact-length `Box<[f32]>` so freeing needs
        // only (ptr, len): `into_boxed_slice` makes capacity == length, and
        // `vokra_audio_free` reconstructs the box from that length.
        let num_samples = audio.samples.len();
        let data_ptr = Box::into_raw(audio.samples.into_boxed_slice()).cast::<f32>();

        // SAFETY: all three out-pointers are non-null (checked above).
        unsafe {
            *out_pcm = data_ptr;
            *out_num_samples = num_samples;
            *out_sample_rate = sample_rate;
        }
        Ok(())
    })
}

/// Frees a PCM buffer returned by `vokra_tts_synthesize`. Pass the pointer and
/// the sample count from the same call. `NULL` is a no-op; double-free is
/// undefined behaviour.
///
/// # Safety
///
/// `pcm` must be `NULL`, or the `*out_pcm` from a `vokra_tts_synthesize` call
/// with the matching `num_samples`, not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_audio_free(pcm: *mut f32, num_samples: usize) {
    ffi_guard::guard_void(|| {
        if !pcm.is_null() {
            let slice = std::ptr::slice_from_raw_parts_mut(pcm, num_samples);
            // SAFETY: `pcm`/`num_samples` describe the `Box<[f32]>` created in
            // `vokra_tts_synthesize`; reconstruct and drop it exactly once.
            drop(unsafe { Box::from_raw(slice) });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_rejects_null_session() {
        let mut pcm: *mut f32 = std::ptr::null_mut();
        let mut n: usize = 0;
        let mut sr: i32 = 0;
        let text = std::ffi::CString::new("hi").unwrap();
        // SAFETY: NULL session is the rejected branch; text is a live C string.
        let status = unsafe {
            vokra_tts_synthesize(std::ptr::null(), text.as_ptr(), &mut pcm, &mut n, &mut sr)
        };
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(pcm.is_null());
    }

    #[test]
    fn audio_free_null_is_noop() {
        // SAFETY: NULL is an explicit no-op.
        unsafe { vokra_audio_free(std::ptr::null_mut(), 0) };
    }

    #[test]
    fn audio_free_reclaims_box() {
        // Round-trip the exact allocation path used by vokra_tts_synthesize.
        let samples = vec![0.1f32, 0.2, 0.3, 0.4];
        let n = samples.len();
        let ptr = Box::into_raw(samples.into_boxed_slice()).cast::<f32>();
        // SAFETY: `ptr`/`n` describe a live Box<[f32]>, freed exactly once.
        unsafe { vokra_audio_free(ptr, n) };
    }
}
