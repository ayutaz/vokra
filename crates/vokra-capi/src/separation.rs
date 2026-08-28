//! Offline source separation and enhancement through the C ABI.

use crate::error::vokra_status_t;
use crate::handle::vokra_session_t;
use crate::{error, ffi_guard};

/// Separates or enhances a complete mono waveform with the session's model.
///
/// The returned allocation is stream-major `[stream][sample]`. Every stream
/// has `out_num_samples_per_stream` samples. Free it with
/// `vokra_audio_free(out_pcm, out_num_streams * out_num_samples_per_stream)`.
/// All output pointers are written only on `VOKRA_OK`.
///
/// # Safety
///
/// `session` must be a live session handle, `pcm` must point at
/// `num_samples` initialized `f32` values, and all output pointers must be
/// writable locations of their declared type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_separate(
    session: *const vokra_session_t,
    pcm: *const f32,
    num_samples: usize,
    sample_rate: i32,
    out_pcm: *mut *mut f32,
    out_num_streams: *mut usize,
    out_num_samples_per_stream: *mut usize,
    out_sample_rate: *mut i32,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: validated by `required_ref`; callers own handle lifetime.
        let handle = unsafe { ffi_guard::required_ref(session, "session")? };
        if pcm.is_null() {
            return Err(error::fail_invalid("pcm must not be NULL"));
        }
        if num_samples == 0 {
            return Err(error::fail_invalid("num_samples must be non-zero"));
        }
        ffi_guard::require_out_ptr(out_pcm, "out_pcm")?;
        ffi_guard::require_out_ptr(out_num_streams, "out_num_streams")?;
        ffi_guard::require_out_ptr(out_num_samples_per_stream, "out_num_samples_per_stream")?;
        ffi_guard::require_out_ptr(out_sample_rate, "out_sample_rate")?;

        let expected_rate = handle
            .session
            .separation_sample_rate()
            .map_err(|error| error::fail(&error))?;
        let actual_rate = u32::try_from(sample_rate)
            .map_err(|_| error::fail_invalid("sample_rate must be positive"))?;
        if actual_rate != expected_rate {
            return Err(error::fail_invalid(&format!(
                "separation model expects {expected_rate} Hz PCM, got {actual_rate} Hz; resample explicitly"
            )));
        }

        // SAFETY: `pcm` is non-null and the caller promises `num_samples`
        // initialized values for this call.
        let input = unsafe { std::slice::from_raw_parts(pcm, num_samples) };
        let outputs = handle
            .session
            .separate_audio(input)
            .map_err(|error| error::fail(&error))?;
        let expected_streams = handle
            .session
            .separation_output_streams()
            .map_err(|error| error::fail(&error))?;
        if outputs.len() != expected_streams || outputs.is_empty() {
            return Err(error::fail_invalid(&format!(
                "separation engine returned {} streams, expected {expected_streams}",
                outputs.len()
            )));
        }
        let samples_per_stream = outputs[0].len();
        if outputs
            .iter()
            .any(|stream| stream.len() != samples_per_stream)
        {
            return Err(error::fail_invalid(
                "separation engine returned streams with different sample counts",
            ));
        }
        let total_samples = outputs
            .len()
            .checked_mul(samples_per_stream)
            .ok_or_else(|| error::fail_invalid("separation output size overflow"))?;
        let mut flattened = Vec::with_capacity(total_samples);
        for stream in outputs {
            flattened.extend_from_slice(&stream);
        }
        let output_rate = i32::try_from(expected_rate)
            .map_err(|_| error::fail_invalid("model sample rate overflows int32_t"))?;
        let stream_count = expected_streams;
        let data_ptr = Box::into_raw(flattened.into_boxed_slice()).cast::<f32>();

        // SAFETY: every output pointer was checked above and the allocation is
        // now owned by the C caller until `vokra_audio_free`.
        unsafe {
            *out_pcm = data_ptr;
            *out_num_streams = stream_count;
            *out_num_samples_per_stream = samples_per_stream;
            *out_sample_rate = output_rate;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vokra_core::engines::SeparationEngine;
    use vokra_core::{BackendKind, Result, Session};

    use super::*;
    use crate::handle;
    use crate::session::{vokra_session_create_from_file, vokra_session_destroy};
    use crate::tts::vokra_audio_free;

    const SEPFORMER_PCM: &[u8] =
        include_bytes!("../../vokra-models/tests/fixtures/sepformer/pcm.f32.bin");
    const SEPFORMER_OUTPUT: &[u8] =
        include_bytes!("../../vokra-models/tests/fixtures/sepformer/separated.f32.bin");
    const CONV_TASNET_PCM: &[u8] =
        include_bytes!("../../vokra-models/tests/fixtures/conv_tasnet/pcm.f32.bin");
    const CONV_TASNET_OUTPUT: &[u8] =
        include_bytes!("../../vokra-models/tests/fixtures/conv_tasnet/separated.f32.bin");

    fn fixture_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    struct FakeSeparator;

    impl SeparationEngine for FakeSeparator {
        fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![
                pcm.to_vec(),
                pcm.iter().map(|value| -*value).collect(),
            ])
        }

        fn sample_rate(&self) -> u32 {
            8_000
        }

        fn output_streams(&self) -> usize {
            2
        }

        fn backend(&self) -> BackendKind {
            BackendKind::Cpu
        }
    }

    fn fake_session() -> *mut vokra_session_t {
        let mut builder = vokra_core::gguf::GgufBuilder::new();
        builder.add_string("vokra.model.arch", "fake-separator");
        let file = vokra_core::gguf::GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let session = Session::from_gguf(file)
            .build()
            .unwrap()
            .with_separation_engine(Arc::new(FakeSeparator));
        handle::into_raw(vokra_session_t { session })
    }

    #[test]
    fn separation_returns_stream_major_audio() {
        let session = fake_session();
        let input = [0.25f32, -0.5, 0.75];
        let mut output = std::ptr::null_mut();
        let mut streams = 0;
        let mut samples = 0;
        let mut rate = 0;
        // SAFETY: all input and output pointers remain live for the call.
        let status = unsafe {
            vokra_separate(
                session,
                input.as_ptr(),
                input.len(),
                8_000,
                &mut output,
                &mut streams,
                &mut samples,
                &mut rate,
            )
        };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        assert_eq!((streams, samples, rate), (2, 3, 8_000));
        // SAFETY: successful call returned exactly streams*samples values.
        let actual = unsafe { std::slice::from_raw_parts(output, streams * samples) };
        assert_eq!(actual, &[0.25, -0.5, 0.75, -0.25, 0.5, -0.75]);
        // SAFETY: matching allocation length and live session handle.
        unsafe {
            vokra_audio_free(output, streams * samples);
            handle::drop_raw(session);
        }
    }

    #[test]
    fn separation_rejects_wrong_rate_without_writing_outputs() {
        let session = fake_session();
        let input = [0.0f32; 16];
        let mut output = std::ptr::null_mut();
        let mut streams = 0;
        let mut samples = 0;
        let mut rate = 0;
        // SAFETY: all pointers are live; rate mismatch is the tested branch.
        let status = unsafe {
            vokra_separate(
                session,
                input.as_ptr(),
                input.len(),
                16_000,
                &mut output,
                &mut streams,
                &mut samples,
                &mut rate,
            )
        };
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(output.is_null());
        assert_eq!((streams, samples, rate), (0, 0, 0));
        // SAFETY: live handle, freed exactly once.
        unsafe { handle::drop_raw(session) };
    }

    #[test]
    fn public_sepformer_runs_through_session_and_c_abi() {
        let Some(path) = std::env::var_os("VOKRA_SEPFORMER_GGUF") else {
            eprintln!(
                "[vokra-capi separation] SKIP: set VOKRA_SEPFORMER_GGUF to the public WHAM16k GGUF"
            );
            return;
        };
        let path = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut session = std::ptr::null_mut();
        // SAFETY: C string and output slot stay live for the call.
        let status = unsafe { vokra_session_create_from_file(path.as_ptr(), &mut session) };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        assert!(!session.is_null());

        let input = fixture_f32(SEPFORMER_PCM);
        let expected = fixture_f32(SEPFORMER_OUTPUT);
        let mut output = std::ptr::null_mut();
        let mut streams = 0;
        let mut samples = 0;
        let mut rate = 0;
        // SAFETY: live session/input and writable output slots.
        let status = unsafe {
            vokra_separate(
                session,
                input.as_ptr(),
                input.len(),
                16_000,
                &mut output,
                &mut streams,
                &mut samples,
                &mut rate,
            )
        };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        assert_eq!((streams, samples, rate), (1, expected.len(), 16_000));
        // SAFETY: successful call returned exactly streams*samples values.
        let actual = unsafe { std::slice::from_raw_parts(output, streams * samples) };
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .sum::<f32>()
            / actual.len() as f32;
        assert!(max_abs <= 0.01, "C ABI SepFormer max_abs={max_abs:.9e}");
        assert!(mean_abs <= 0.001, "C ABI SepFormer mean_abs={mean_abs:.9e}");
        // SAFETY: matching allocation length and live session handle.
        unsafe {
            vokra_audio_free(output, streams * samples);
            vokra_session_destroy(session);
        }
    }

    #[test]
    fn corrected_conv_tasnet_runs_through_session_and_c_abi() {
        let Some(path) = std::env::var_os("VOKRA_CONV_TASNET_GGUF") else {
            eprintln!(
                "[vokra-capi separation] SKIP: set VOKRA_CONV_TASNET_GGUF to a GGUF converted from the pinned official checkpoint"
            );
            return;
        };
        let path = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut session = std::ptr::null_mut();
        // SAFETY: C string and output slot stay live for the call.
        let status = unsafe { vokra_session_create_from_file(path.as_ptr(), &mut session) };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        assert!(!session.is_null());

        let input = fixture_f32(CONV_TASNET_PCM);
        let expected = fixture_f32(CONV_TASNET_OUTPUT);
        let mut output = std::ptr::null_mut();
        let mut streams = 0;
        let mut samples = 0;
        let mut rate = 0;
        // SAFETY: live session/input and writable output slots.
        let status = unsafe {
            vokra_separate(
                session,
                input.as_ptr(),
                input.len(),
                16_000,
                &mut output,
                &mut streams,
                &mut samples,
                &mut rate,
            )
        };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        assert_eq!((streams, samples, rate), (1, expected.len(), 16_000));
        // SAFETY: successful call returned exactly streams*samples values.
        let actual = unsafe { std::slice::from_raw_parts(output, streams * samples) };
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let relative_l1 = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .sum::<f32>()
            / expected.iter().map(|value| value.abs()).sum::<f32>();
        assert!(max_abs <= 0.40, "C ABI Conv-TasNet max_abs={max_abs:.9e}");
        assert!(
            relative_l1 <= 0.001,
            "C ABI Conv-TasNet relative_l1={relative_l1:.9e}"
        );
        // SAFETY: matching allocation length and live session handle.
        unsafe {
            vokra_audio_free(output, streams * samples);
            vokra_session_destroy(session);
        }
    }
}
