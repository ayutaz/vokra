//! Streaming continuous speech-encoder features across the C ABI (#49).
//!
//! The first native family is Moshi's causal Mimi input encoder. It emits the
//! bottleneck-transformer hidden grid at 25 Hz, before token-rate resampling
//! and RVQ, so downstream models receive continuous rather than quantized
//! representations. The engine trait keeps this ABI model-independent.

use crate::error::vokra_status_t;
use crate::handle::{self, vokra_feat_t, vokra_session_t};
use crate::{error, ffi_guard};

/// Opens a continuous speech-feature stream for `session`.
///
/// The returned handle retains the session and must be released with
/// [`vokra_feat_destroy`]. Returns `NULL` with detail in `vokra_last_error()`
/// when the model family has no feature engine or construction fails.
///
/// # Safety
///
/// `session` must be a live session handle or `NULL` (the rejected branch).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_open(session: *const vokra_session_t) -> *mut vokra_feat_t {
    ffi_guard::guard_ptr(|| {
        // SAFETY: validated by `required_ref`; NULL becomes a recorded error.
        let session = match unsafe { ffi_guard::required_ref(session, "session") } {
            Ok(session) => session,
            Err(_) => return std::ptr::null_mut(),
        };
        let stream = match session.session.open_speech_feature_stream() {
            Ok(stream) => stream,
            Err(err) => {
                error::fail(&err);
                return std::ptr::null_mut();
            }
        };
        handle::into_raw(vokra_feat_t {
            stream,
            _session: session.session.clone(),
        })
    })
}

/// Returns the encoder's native frame rate in milli-Hertz (25 Hz = 25,000).
/// Returns `-1` on a NULL handle or internal failure.
///
/// # Safety
///
/// `feat` must be a live feature handle or `NULL` (the rejected branch).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_frame_rate_mhz(feat: *const vokra_feat_t) -> i32 {
    ffi_guard::guard_i32(|| {
        // SAFETY: validated by `required_ref`.
        let feat = match unsafe { ffi_guard::required_ref(feat, "feat") } {
            Ok(feat) => feat,
            Err(_) => return -1,
        };
        match i32::try_from(feat.stream.frame_rate_millihz()) {
            Ok(value) => value,
            Err(_) => {
                error::fail_invalid("feature frame rate does not fit int32_t");
                -1
            }
        }
    })
}

/// Returns the number of `float` values in one feature frame, or `-1` on
/// failure.
///
/// # Safety
///
/// `feat` must be a live feature handle or `NULL` (the rejected branch).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_dim(feat: *const vokra_feat_t) -> i32 {
    ffi_guard::guard_i32(|| {
        // SAFETY: validated by `required_ref`.
        let feat = match unsafe { ffi_guard::required_ref(feat, "feat") } {
            Ok(feat) => feat,
            Err(_) => return -1,
        };
        match i32::try_from(feat.stream.feature_dim()) {
            Ok(value) => value,
            Err(_) => {
                error::fail_invalid("feature dimension does not fit int32_t");
                -1
            }
        }
    })
}

/// Appends arbitrary-length mono PCM at the model's native sample rate.
///
/// A trailing partial encoder frame is retained. The handle owns a bounded
/// pending-feature queue; if the supplied PCM would overflow it, this returns
/// `VOKRA_ERROR_INVALID_ARGUMENT` without consuming input. Successful calls
/// allocate nothing after stream construction/warmup.
///
/// # Safety
///
/// `feat` must be live and uniquely accessed for the call. `pcm` must point to
/// `n` readable floats, or may be `NULL` only when `n == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_push_pcm(
    feat: *mut vokra_feat_t,
    pcm: *const f32,
    n: usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: raw arguments are checked by the shared boundary helpers.
        let feat = unsafe { ffi_guard::required_mut(feat, "feat")? };
        // SAFETY: NULL is accepted only for the zero-length slice.
        let pcm = unsafe { ffi_guard::required_slice(pcm, n, "pcm")? };
        feat.stream.push_pcm(pcm).map_err(|err| error::fail(&err))?;
        Ok(())
    })
}

/// Pulls whole feature frames into `out` without blocking.
///
/// `cap` is a capacity in **floats**, not frames. Up to
/// `floor(cap / vokra_feat_dim(feat))` rows are written. `out_frames` receives
/// the row count, and `out_start_sample` receives the exact source-PCM sample
/// index of the first row (`-1` when no row was pending). A non-zero `cap`
/// smaller than one row is rejected without consuming queued output.
///
/// # Safety
///
/// `feat` must be live and uniquely accessed. `out` must point to `cap`
/// writable floats, or may be `NULL` only when `cap == 0`; both scalar output
/// pointers must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_pull(
    feat: *mut vokra_feat_t,
    out: *mut f32,
    cap: usize,
    out_frames: *mut usize,
    out_start_sample: *mut i64,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // Validate every pointer before calling the state-mutating pull.
        // SAFETY: validated by `required_mut`.
        let feat = unsafe { ffi_guard::required_mut(feat, "feat")? };
        ffi_guard::require_out_ptr(out_frames, "out_frames")?;
        ffi_guard::require_out_ptr(out_start_sample, "out_start_sample")?;
        let dst = if cap == 0 {
            &mut []
        } else {
            ffi_guard::require_out_ptr(out, "out")?;
            // SAFETY: non-null above; caller guarantees `cap` writable floats.
            unsafe { std::slice::from_raw_parts_mut(out, cap) }
        };
        let (frames, start_sample) = feat
            .stream
            .pull_into(dst)
            .map_err(|err| error::fail(&err))?;
        // SAFETY: both out-pointers are non-null and writable by contract.
        unsafe {
            *out_frames = frames;
            *out_start_sample = if frames == 0 { -1 } else { start_sample };
        }
        Ok(())
    })
}

/// Discards PCM tail, queued frames, timestamps and recurrent state while
/// retaining all allocations.
///
/// # Safety
///
/// `feat` must be a live, uniquely accessed feature handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_reset(feat: *mut vokra_feat_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: NULL is rejected and recorded; no dereference on failure.
        if let Ok(feat) = unsafe { ffi_guard::required_mut(feat, "feat") } {
            feat.stream.reset();
        }
    });
}

/// Frees a feature handle. `NULL` is a no-op; double-free is undefined.
///
/// # Safety
///
/// `feat` must be `NULL` or a live pointer returned by [`vokra_feat_open`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_feat_destroy(feat: *mut vokra_feat_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: the destroy contract matches `handle::drop_raw`.
        unsafe { handle::drop_raw(feat) };
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vokra_core::Session;
    use vokra_core::engines::SpeechFeatureEngine;
    use vokra_core::gguf::{GgufBuilder, GgufFile};
    use vokra_models::moshi::MoshiEngine;

    use super::*;

    fn feature_session(seed: u64) -> (vokra_session_t, Arc<MoshiEngine>) {
        let gguf = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let engine = Arc::new(MoshiEngine::synthesized_fixture(seed).unwrap());
        let session = Session::from_gguf(gguf)
            .build()
            .unwrap()
            .with_speech_feature_engine(engine.clone());
        (vokra_session_t { session }, engine)
    }

    fn pcm(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.019).sin() * 0.3).collect()
    }

    #[test]
    fn c_chunks_match_the_same_native_engine_whole_buffer_and_timestamp() {
        let (session, engine) = feature_session(149);
        let mut reference = engine.open_feature_stream().unwrap();
        let token_hop = reference.feature_frame_hop() * 2;
        let input = pcm(token_hop * 3);
        reference.push_pcm(&input).unwrap();
        let dim = reference.feature_dim();
        let frame_rate_millihz = reference.frame_rate_millihz();
        let mut want = vec![0.0f32; dim * 6];
        assert_eq!(reference.pull_into(&mut want).unwrap(), (6, 0));

        // SAFETY: stack session stays live until the feature handle is freed.
        let feat = unsafe { vokra_feat_open(&session) };
        assert!(!feat.is_null());
        assert_eq!(
            // SAFETY: `feat` is a non-null live handle created above.
            unsafe { vokra_feat_frame_rate_mhz(feat) },
            frame_rate_millihz as i32,
        );
        // SAFETY: `feat` is a non-null live handle created above.
        assert_eq!(unsafe { vokra_feat_dim(feat) }, dim as i32);
        let mut cursor = 0usize;
        for chunk in [1usize, 37, token_hop + 5, 211, input.len()] {
            if cursor == input.len() {
                break;
            }
            let take = chunk.min(input.len() - cursor);
            // SAFETY: `feat` is live and the PCM pointer covers exactly `take` samples.
            let status =
                unsafe { vokra_feat_push_pcm(feat, input[cursor..cursor + take].as_ptr(), take) };
            assert_eq!(status, vokra_status_t::VOKRA_OK);
            cursor += take;
        }
        let mut got = vec![0.0f32; want.len()];
        let mut frames = usize::MAX;
        let mut start = -2i64;
        // SAFETY: `feat` is live, `got` has the advertised capacity, and both scalar
        // output pointers remain valid for the call.
        let status =
            unsafe { vokra_feat_pull(feat, got.as_mut_ptr(), got.len(), &mut frames, &mut start) };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        assert_eq!((frames, start), (6, 0));
        assert_eq!(got, want);
        // SAFETY: live handle, freed exactly once.
        unsafe { vokra_feat_destroy(feat) };
    }

    #[test]
    fn pull_empty_uses_minus_one_timestamp_and_reset_restarts_zero() {
        let (session, _) = feature_session(150);
        // SAFETY: valid session.
        let feat = unsafe { vokra_feat_open(&session) };
        let mut frames = 99usize;
        let mut start = 99i64;
        assert_eq!(
            // SAFETY: `feat` and scalar outputs are live; a null output is valid at
            // zero capacity.
            unsafe { vokra_feat_pull(feat, std::ptr::null_mut(), 0, &mut frames, &mut start) },
            vokra_status_t::VOKRA_OK,
        );
        assert_eq!((frames, start), (0, -1));
        // SAFETY: live handle.
        unsafe { vokra_feat_reset(feat) };
        assert_eq!(
            // SAFETY: same live handle and valid scalar outputs as the first pull.
            unsafe { vokra_feat_pull(feat, std::ptr::null_mut(), 0, &mut frames, &mut start) },
            vokra_status_t::VOKRA_OK,
        );
        assert_eq!((frames, start), (0, -1));
        // SAFETY: live handle, freed once.
        unsafe { vokra_feat_destroy(feat) };
    }

    #[test]
    fn null_and_model_mismatch_fail_closed() {
        // SAFETY: this deliberately passes null to verify the FFI boundary rejects it.
        assert!(unsafe { vokra_feat_open(std::ptr::null()) }.is_null());
        // SAFETY: this deliberately passes null to verify the query fails closed.
        assert_eq!(unsafe { vokra_feat_dim(std::ptr::null()) }, -1);
        assert_eq!(
            // SAFETY: null arguments are intentional inputs to the validation path.
            unsafe { vokra_feat_push_pcm(std::ptr::null_mut(), std::ptr::null(), 0) },
            vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
        );

        let gguf = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let bare = vokra_session_t {
            session: Session::from_gguf(gguf).build().unwrap(),
        };
        // SAFETY: `bare` is live; the test verifies that a session without a feature
        // engine is rejected without dereferencing an invalid pointer.
        assert!(unsafe { vokra_feat_open(&bare) }.is_null());
        assert!(!error::vokra_last_error().is_null());
    }

    #[test]
    fn pull_argument_error_leaves_scalar_outputs_untouched() {
        let (session, _) = feature_session(151);
        // SAFETY: valid session.
        let feat = unsafe { vokra_feat_open(&session) };
        let mut frames = 77usize;
        let mut start = 88i64;
        assert_eq!(
            // SAFETY: `feat` and scalar outputs are live. The null buffer with nonzero
            // capacity is intentional and must be rejected before dereference.
            unsafe { vokra_feat_pull(feat, std::ptr::null_mut(), 1, &mut frames, &mut start,) },
            vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
        );
        assert_eq!((frames, start), (77, 88));
        // SAFETY: live handle, freed once.
        unsafe { vokra_feat_destroy(feat) };
    }
}
