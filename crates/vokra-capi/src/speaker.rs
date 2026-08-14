//! Speaker embedding and verification (`speaker_encode` / `speaker_verify`,
//! FR-OP-80 / FR-OP-81; design
//! `docs/superpowers/specs/2026-08-14-c-abi-backend-speaker-design.md` §3.3).
//!
//! # Why this is in core, not in the voice-clone repository
//!
//! CLAUDE.md design note 8 splits voice *cloning* (RVC / GPT-SoVITS) into the
//! separate `vokra-voiceclone-experimental` repository because the Tennessee
//! ELVIS Act judges a tool by its "primary purpose or effect". Speaker
//! **embedding** stays in core and is exposed here (design D3): it returns a
//! feature vector and synthesizes no audio, and every modern zero-shot TTS
//! needs it as an input.
//!
//! # PCM in, not filterbank in
//!
//! `SpeakerEncoder::embed` consumes a Kaldi filterbank. No host binding
//! (C# / GDScript / Swift / Kotlin) can reasonably compute one, so
//! [`vokra_speaker_embed`] takes the waveform and runs the model's own
//! front-end — the same choice `vokra_asr_transcribe` already makes. The rate
//! is checked, never silently converted (FR-EX-08).
//!
//! # Caller-owned output buffer
//!
//! The embedding is written into the caller's array rather than returned as a
//! Vokra-allocated block (`vokra_string_free` style): Unity's C# marshalling
//! handles a pinned `float[]` far better than a pointer it must remember to
//! free. CAM++ is 192-d today, but ECAPA-TDNN / TitaNet / WeSpeaker differ, so
//! the size is discovered rather than assumed — see [`vokra_speaker_embed`] for
//! the two-call idiom.

use vokra_models::speaker::speaker_verify;

use crate::error::{self, fail_invalid, vokra_status_t};
use crate::ffi_guard;
use crate::handle::vokra_session_t;

/// Computes the speaker embedding of one mono reference utterance.
///
/// The session must have been created from a speaker-encoder model (GGUF arch
/// `campplus`); any other model reports `VOKRA_ERROR_NOT_IMPLEMENTED`, the same
/// task-mismatch posture as `vokra_asr_transcribe` on a TTS voice.
///
/// # Parameters
///
/// - `session`: a session holding a speaker-encoder model.
/// - `pcm` / `num_samples`: mono `f32` samples in `[-1, 1]`. The clip must
///   cover at least one analysis frame (25 ms at 16 kHz).
/// - `sample_rate`: sample rate of `pcm` in Hz. Must equal the rate the model's
///   front-end was trained at (16000 for CAM++); a mismatch is rejected instead
///   of resampled, because a silent resample would change the embedding without
///   telling the caller (FR-EX-08).
/// - `out_embedding` / `out_capacity`: caller-owned destination array and its
///   length **in floats**. May be `NULL` / `0` to query the size only.
/// - `out_written`: receives the embedding dimension — on success the number of
///   floats written, and on a too-small buffer the number of floats required.
///
/// # Returns
///
/// `VOKRA_OK` when the embedding was written, or
/// `VOKRA_ERROR_INVALID_ARGUMENT` when `out_capacity` is too small — in which
/// case `*out_written` still holds the required size, so the two-call idiom is:
///
/// ```c
/// size_t n = 0;
/// vokra_speaker_embed(s, pcm, len, 16000, NULL, 0, &n);   /* INVALID_ARGUMENT, n = 192 */
/// float *emb = malloc(n * sizeof(float));
/// vokra_speaker_embed(s, pcm, len, 16000, emb, n, &n);    /* VOKRA_OK */
/// ```
///
/// Note that this differs from `vokra_s2s_text` / `vokra_model_attribution`,
/// which report a short buffer as `VOKRA_OK`. Those return optional text a
/// caller may legitimately skip; a truncated embedding is never useful, so it
/// is an error here. `*out_written` is the one output written on that error
/// path — every other failure leaves all outputs untouched.
///
/// # Safety
///
/// `session` must be a live session handle; `pcm` must point at `num_samples`
/// valid floats (or be `NULL` when `num_samples` is 0); `out_embedding` must be
/// `NULL` or point at `out_capacity` writable floats; `out_written` must be a
/// valid, writable `size_t` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_speaker_embed(
    session: *const vokra_session_t,
    pcm: *const f32,
    num_samples: usize,
    sample_rate: i32,
    out_embedding: *mut f32,
    out_capacity: usize,
    out_written: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `session` is validated (NULL rejected) by `required_ref`.
        let handle = unsafe { ffi_guard::required_ref(session, "session") }?;
        ffi_guard::require_out_ptr(out_written, "out_written")?;
        if out_capacity > 0 {
            ffi_guard::require_out_ptr(out_embedding, "out_embedding")?;
        }
        // SAFETY: `pcm` is valid for `num_samples` reads per the contract; a
        // zero length never dereferences it.
        let samples = unsafe { ffi_guard::required_slice(pcm, num_samples, "pcm") }?;
        if sample_rate <= 0 {
            return Err(fail_invalid(&format!(
                "argument `sample_rate` must be positive, got {sample_rate}"
            )));
        }
        let rate = u32::try_from(sample_rate).map_err(|_| {
            fail_invalid(&format!(
                "argument `sample_rate` = {sample_rate} does not fit in a u32"
            ))
        })?;

        let embedding = handle
            .session
            .speaker()
            .embed(samples, rate)
            .map_err(|e| error::fail(&e))?;

        // The required size is reported on both the success and the
        // buffer-too-small path, which is what makes the two-call idiom work.
        // SAFETY: `out_written` is non-null (checked) and points at a writable
        // `size_t` per the contract.
        unsafe { *out_written = embedding.len() };

        if out_capacity < embedding.len() {
            return Err(fail_invalid(&format!(
                "argument `out_capacity` = {out_capacity} is too small for a {}-dimensional \
                 speaker embedding; `*out_written` now holds the required length",
                embedding.len()
            )));
        }
        // The empty case is skipped rather than copied: `copy_nonoverlapping`
        // requires a non-null, aligned destination even for a zero count, and
        // `out_embedding` is only NULL-checked above when `out_capacity > 0`.
        // No in-tree engine returns an empty embedding (CAM++ is 192-d, and it
        // is the only `SpeakerEngine`), so this guard is unreachable today —
        // it is here so the `unsafe` block below rests on conditions this
        // function actually checks, not on a property of the current engine
        // set (2026-08-14 C ABI review).
        if !embedding.is_empty() {
            // SAFETY: `out_embedding` is non-null — `out_capacity >=
            // embedding.len()` was just checked and `embedding.len() > 0`
            // here, so the `out_capacity > 0` NULL check above ran — and it is
            // valid for `out_capacity >= embedding.len()` writes; the source is
            // an owned, non-overlapping Vec.
            unsafe {
                std::ptr::copy_nonoverlapping(embedding.as_ptr(), out_embedding, embedding.len());
            }
        }
        Ok(())
    })
}

/// Compares two speaker embeddings (FR-OP-81).
///
/// Takes no session: it is arithmetic on two vectors, so embeddings can be
/// stored (in a database, a save file) and matched later without a model
/// loaded. The inputs need not be L2-normalized — `vokra_speaker_embed`
/// output goes straight in.
///
/// # Parameters
///
/// - `a` / `a_len`, `b` / `b_len`: the two embeddings. They must be the same
///   non-zero length, and neither may be all zeros (a zero vector has no
///   direction).
/// - `threshold`: the accept/reject operating point, used **only** when
///   `out_same_speaker` is non-`NULL`. It must be finite.
/// - `out_similarity`: receives the cosine similarity in `[-1, 1]` (1 = same
///   direction). Required.
/// - `out_same_speaker`: optional. When non-`NULL`, receives
///   `similarity >= threshold`. Pass `NULL` to get the similarity only —
///   Vokra deliberately does not ship a default threshold (ADR M4-20 §D-4);
///   take the operating point from your model's published EER.
///
/// # Returns
///
/// `VOKRA_OK`, or `VOKRA_ERROR_INVALID_ARGUMENT` for a NULL required pointer,
/// mismatched or empty lengths, a zero-norm embedding, or a non-finite
/// `threshold` when a decision was requested.
///
/// # Safety
///
/// `a` / `b` must point at `a_len` / `b_len` valid floats; `out_similarity`
/// must be a valid, writable `float` location; `out_same_speaker` must be
/// `NULL` or a valid, writable `bool` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_speaker_verify(
    a: *const f32,
    a_len: usize,
    b: *const f32,
    b_len: usize,
    threshold: f32,
    out_similarity: *mut f32,
    out_same_speaker: *mut bool,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        ffi_guard::require_out_ptr(out_similarity, "out_similarity")?;
        // SAFETY: `a` / `b` are valid for `a_len` / `b_len` reads per the
        // contract; a zero length never dereferences them.
        let lhs = unsafe { ffi_guard::required_slice(a, a_len, "a") }?;
        // SAFETY: as above for `b`.
        let rhs = unsafe { ffi_guard::required_slice(b, b_len, "b") }?;

        // A threshold is only meaningful when a decision was asked for; only
        // then is it validated, so `NULL`-decision callers may pass anything.
        let wants_decision = !out_same_speaker.is_null();
        if wants_decision && !threshold.is_finite() {
            return Err(fail_invalid(&format!(
                "argument `threshold` must be finite when `out_same_speaker` is requested, \
                 got {threshold}"
            )));
        }
        let threshold = wants_decision.then_some(threshold);

        let result = speaker_verify(lhs, rhs, threshold).map_err(|e| error::fail(&e))?;

        // SAFETY: `out_similarity` is non-null (checked) and points at a
        // writable float per the contract.
        unsafe { *out_similarity = result.similarity };
        if let Some(accepted) = result.accepted {
            // SAFETY: reached only when `out_same_speaker` is non-null (that is
            // exactly what put a `Some` in `threshold`), and it points at a
            // writable bool per the contract.
            unsafe { *out_same_speaker = accepted };
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    /// Two embeddings that are similar but not identical, plus their exact
    /// cosine similarity computed the way `vokra-models` computes it — the
    /// oracle for the checks below.
    fn pair() -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..192).map(|i| ((i % 7) as f32) - 3.0).collect();
        let b: Vec<f32> = (0..192).map(|i| ((i % 5) as f32) - 2.0).collect();
        (a, b)
    }

    #[test]
    fn verify_of_an_embedding_with_itself_is_one() {
        let (a, _) = pair();
        let mut similarity = 0.0f32;
        let mut same = false;
        // SAFETY: live slices, writable out-slots.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                a.as_ptr(),
                a.len(),
                0.99,
                &mut similarity,
                &mut same,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(
            (similarity - 1.0).abs() < 1e-5,
            "self-similarity is 1.0, got {similarity}"
        );
        assert!(same, "an embedding must match itself at threshold 0.99");
    }

    #[test]
    fn verify_matches_the_rust_oracle_and_honors_the_threshold() {
        let (a, b) = pair();
        let expected = vokra_models::speaker::cosine_similarity(&a, &b).expect("cosine");
        // Guard against a degenerate fixture: a cross pair at similarity 1.0
        // would make the threshold assertions below vacuous.
        assert!(expected < 0.99, "fixture pair is too similar: {expected}");

        let mut similarity = 0.0f32;
        let mut same = true;
        // SAFETY: live slices, writable out-slots.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                0.99,
                &mut similarity,
                &mut same,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert_eq!(
            similarity.to_bits(),
            expected.to_bits(),
            "the C ABI must return the same float `speaker_verify` computes"
        );
        assert!(!same, "different speakers must be rejected at 0.99");

        // A threshold below the similarity accepts.
        // SAFETY: live slices, writable out-slots.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                expected - 0.01,
                &mut similarity,
                &mut same,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(same, "a threshold under the similarity must accept");
    }

    /// A NULL `out_same_speaker` is the "similarity only" mode: it must not be
    /// written, and the threshold must be ignored (Vokra ships no default
    /// operating point — ADR M4-20 §D-4).
    #[test]
    fn verify_without_a_decision_slot_ignores_the_threshold() {
        let (a, b) = pair();
        let mut similarity = 0.0f32;
        // SAFETY: live slices; NULL decision slot is the documented
        // similarity-only mode.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                f32::NAN,
                &mut similarity,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            st,
            vokra_status_t::VOKRA_OK,
            "a NaN threshold is irrelevant when no decision was requested"
        );
        assert!(similarity.is_finite());
    }

    #[test]
    fn verify_rejects_a_non_finite_threshold_when_a_decision_is_requested() {
        let (a, b) = pair();
        let mut similarity = 0.0f32;
        let mut same = false;
        // SAFETY: live slices, writable out-slots.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                f32::NAN,
                &mut similarity,
                &mut same,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn verify_rejects_null_length_mismatch_and_zero_norm() {
        let (a, b) = pair();
        let mut similarity = 0.0f32;

        // NULL out_similarity.
        // SAFETY: NULL out-slot is the rejected branch.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                0.5,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // NULL embedding with a non-zero length.
        // SAFETY: NULL with len > 0 is the rejected branch; no deref happens.
        let st = unsafe {
            vokra_speaker_verify(
                ptr::null(),
                4,
                b.as_ptr(),
                b.len(),
                0.5,
                &mut similarity,
                ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // Length mismatch.
        // SAFETY: live slices, writable out-slot.
        let st = unsafe {
            vokra_speaker_verify(
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len() - 1,
                0.5,
                &mut similarity,
                ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);

        // Zero-norm embedding has no direction.
        let zeros = vec![0.0f32; a.len()];
        // SAFETY: live slices, writable out-slot.
        let st = unsafe {
            vokra_speaker_verify(
                zeros.as_ptr(),
                zeros.len(),
                a.as_ptr(),
                a.len(),
                0.5,
                &mut similarity,
                ptr::null_mut(),
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    /// A deterministic, non-degenerate 1 s mono signal at 16 kHz.
    ///
    /// Two incommensurate sinusoids plus a slow envelope: CAM++'s front-end
    /// subtracts the cepstral mean, so a constant (or silent) clip would
    /// collapse to a zero-norm embedding and make the comparisons below
    /// vacuous.
    fn reference_pcm() -> Vec<f32> {
        (0..16_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 1.7 * t).sin();
                env * (0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1337.0 * t).sin())
            })
            .collect()
    }

    /// End-to-end over a **real** CAM++ GGUF: the C ABI's PCM → embedding path
    /// must reproduce the Rust `kaldi_fbank` + `SpeakerEncoder::embed` chain
    /// exactly, and the two-call sizing idiom must work.
    ///
    /// The oracle is the Rust API, not a re-derivation of it: the Rust
    /// fbank→embedding forward is itself pinned against onnxruntime by
    /// `vokra-models`' `speaker::parity` (atol 0.01), and the PCM→fbank
    /// front-end against a torchaudio reference by `vokra-ops`'
    /// `tests/kaldi_fbank_parity.rs` (atol 2e-4). What is unverified until
    /// here is the *wiring* — that the C entry runs that chain and no other —
    /// so the comparison is bit-exact rather than toleranced.
    ///
    /// Gated on `VOKRA_CAMPLUS_GGUF` (the 27 MB model is not committed) and
    /// skips cleanly when unset, the same convention as
    /// `vokra-models`' `speaker::parity` and the CLI's speaker e2e:
    ///
    /// ```text
    /// VOKRA_CAMPLUS_GGUF=campplus.gguf cargo test -p vokra-capi speaker
    /// ```
    #[test]
    fn embed_over_a_real_campplus_gguf_matches_the_rust_chain() {
        let Ok(model) = std::env::var("VOKRA_CAMPLUS_GGUF") else {
            eprintln!("skipping speaker C ABI e2e: set VOKRA_CAMPLUS_GGUF to run");
            return;
        };
        let cpath = std::ffi::CString::new(model.clone()).expect("path has no interior NUL");
        let mut session: *mut vokra_session_t = ptr::null_mut();
        // SAFETY: valid C path; NULL options selects the documented defaults;
        // `session` is a writable out-slot.
        let st = unsafe {
            crate::session::vokra_session_create_from_file_with_options(
                cpath.as_ptr(),
                ptr::null(),
                &mut session,
            )
        };
        assert_eq!(
            st,
            vokra_status_t::VOKRA_OK,
            "loading the CAM++ GGUF failed: {:?}",
            crate::error::vokra_last_error()
        );
        assert!(!session.is_null());

        let pcm = reference_pcm();

        // (1) Sizing call: no buffer at all still reports the dimension.
        let mut needed = 0usize;
        // SAFETY: live session; NULL/0 output is the documented sizing form.
        let st = unsafe {
            vokra_speaker_embed(
                session,
                pcm.as_ptr(),
                pcm.len(),
                16_000,
                ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        assert_eq!(
            st,
            vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
            "a zero-capacity call reports the size and refuses to write"
        );
        assert_eq!(needed, 192, "CAM++ embeddings are 192-d");

        // (2) A buffer one short is still refused, and still reports the size.
        let mut short = vec![0.0f32; needed - 1];
        let mut written = 0usize;
        // SAFETY: live session; `short` is valid for `short.len()` writes.
        let st = unsafe {
            vokra_speaker_embed(
                session,
                pcm.as_ptr(),
                pcm.len(),
                16_000,
                short.as_mut_ptr(),
                short.len(),
                &mut written,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert_eq!(written, needed);
        assert!(
            short.iter().all(|&v| v == 0.0),
            "a refused call must not partially fill the caller's buffer"
        );

        // (3) The real call.
        let mut embedding = vec![0.0f32; needed];
        // SAFETY: live session; `embedding` is valid for `needed` writes.
        let st = unsafe {
            vokra_speaker_embed(
                session,
                pcm.as_ptr(),
                pcm.len(),
                16_000,
                embedding.as_mut_ptr(),
                embedding.len(),
                &mut written,
            )
        };
        assert_eq!(
            st,
            vokra_status_t::VOKRA_OK,
            "embed failed: {:?}",
            crate::error::vokra_last_error()
        );
        assert_eq!(written, needed);
        assert!(
            embedding.iter().any(|&v| v != 0.0),
            "the embedding is all zeros — the oracle below would be vacuous"
        );

        // (4) Oracle: the same chain driven through the Rust API.
        let encoder =
            vokra_models::speaker::SpeakerEncoder::from_path(&model).expect("bind CAM++ encoder");
        let opts = vokra_ops::KaldiFbankOpts::camplus();
        let (fbank, frames) = vokra_ops::kaldi_fbank(&pcm, &opts).expect("fbank");
        let expected = encoder.embed(&fbank, frames).expect("rust embed");
        for (i, (got, want)) in embedding.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "dimension {i} differs ({got} vs {want}) — the C entry is not running the \
                 Rust chain"
            );
        }

        // (5) An embedding always matches itself.
        let mut similarity = 0.0f32;
        let mut same = false;
        // SAFETY: live slice, writable out-slots.
        let st = unsafe {
            vokra_speaker_verify(
                embedding.as_ptr(),
                embedding.len(),
                embedding.as_ptr(),
                embedding.len(),
                0.99,
                &mut similarity,
                &mut same,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(
            (similarity - 1.0).abs() < 1e-5,
            "self-similarity {similarity}"
        );
        assert!(same);

        // (6) A rate the front-end was not trained at is refused, never
        //     resampled behind the caller's back (FR-EX-08).
        // SAFETY: live session and buffers.
        let st = unsafe {
            vokra_speaker_embed(
                session,
                pcm.as_ptr(),
                pcm.len(),
                22_050,
                embedding.as_mut_ptr(),
                embedding.len(),
                &mut written,
            )
        };
        assert_eq!(
            st,
            vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
            "a 22.05 kHz clip must be rejected, not silently resampled"
        );

        // SAFETY: freshly created handle, destroyed exactly once.
        unsafe { crate::session::vokra_session_destroy(session) };
    }

    #[test]
    fn embed_rejects_null_session_and_null_out_written() {
        let pcm = vec![0.0f32; 16_000];
        let mut written = 0usize;
        // SAFETY: NULL session is the rejected branch.
        let st = unsafe {
            vokra_speaker_embed(
                ptr::null(),
                pcm.as_ptr(),
                pcm.len(),
                16_000,
                ptr::null_mut(),
                0,
                &mut written,
            )
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert_eq!(written, 0, "out_written untouched on the reject path");
    }
}
