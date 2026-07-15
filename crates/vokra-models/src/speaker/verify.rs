//! Speaker verification (M4-20 (b), FR-OP-81): cosine similarity between two
//! speaker embeddings, with an optional accept/reject threshold.
//!
//! # L2 normalization is done here (ADR M4-20 §D-4)
//!
//! CAM++ [`SpeakerEncoder::embed`](super::camplus::SpeakerEncoder::embed)
//! returns a **NOT L2-normalized** `[f32; 192]` (`camplus.rs:21`/`:102`), so
//! [`cosine_similarity`] computes `dot(a, b) / (‖a‖·‖b‖)` explicitly and never
//! assumes unit norm. Accumulation is in `f64` for numerical stability; the
//! result is clamped to `[-1, 1]`.
//!
//! # Threshold has no invented default
//!
//! [`speaker_verify`] takes `threshold: Option<f32>`. `None` returns the
//! similarity only; `Some(t)` also returns `accepted = similarity >= t`. The
//! acceptance operating point (EER threshold) is a caller / upstream-reference
//! concern (CAM++ / 3D-Speaker), **not** invented here (ADR M4-20 §D-4).
//!
//! # Generic over embedding length
//!
//! The functions take equal-length `&[f32]` slices, not a fixed 192, so a
//! future ECAPA-TDNN / WeSpeaker embedding (M5) reuses them. A length mismatch,
//! an empty input, or a zero-norm vector is an explicit
//! [`VokraError::InvalidArgument`] (FR-EX-08 spirit, NFR-RL-07).

use vokra_core::{Result, VokraError};

/// The result of a [`speaker_verify`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerVerifyResult {
    /// Cosine similarity in `[-1, 1]` (1 = identical direction).
    pub similarity: f32,
    /// `Some(similarity >= threshold)` when a threshold was supplied, else
    /// `None` (similarity-only mode).
    pub accepted: Option<bool>,
}

/// Cosine similarity `dot(a, b) / (‖a‖·‖b‖)` of two equal-length embeddings,
/// clamped to `[-1, 1]`. Neither input is assumed L2-normalized (ADR M4-20
/// §D-4).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when the slices differ in length, are empty,
/// or either has zero norm (an all-zero embedding has no direction).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(VokraError::InvalidArgument(format!(
            "cosine_similarity: length mismatch {} != {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(VokraError::InvalidArgument(
            "cosine_similarity: empty embeddings".into(),
        ));
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "cosine_similarity: zero-norm embedding has no direction".into(),
        ));
    }
    // Clamp against tiny floating-point overshoot outside [-1, 1].
    Ok(((dot / denom) as f32).clamp(-1.0, 1.0))
}

/// Speaker verification (FR-OP-81): the cosine similarity of two embeddings,
/// plus an optional accept/reject decision.
///
/// Pass the two [`SpeakerEncoder::embed`](super::camplus::SpeakerEncoder::embed)
/// outputs directly (they are raw / non-normalized —
/// [`cosine_similarity`] normalizes). `threshold`:
///
/// - `None` → `accepted = None` (similarity only);
/// - `Some(t)` → `accepted = Some(similarity >= t)`.
///
/// The default acceptance threshold is intentionally **not** invented (ADR
/// M4-20 §D-4): the caller supplies the operating point from the trigger
/// model's published EER, or omits it.
///
/// # Errors
///
/// Propagates [`cosine_similarity`]'s errors (length mismatch / empty /
/// zero-norm).
pub fn speaker_verify(a: &[f32], b: &[f32], threshold: Option<f32>) -> Result<SpeakerVerifyResult> {
    let similarity = cosine_similarity(a, b)?;
    Ok(SpeakerVerifyResult {
        similarity,
        accepted: threshold.map(|t| similarity >= t),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_are_similarity_one() {
        let v = [1.0f32, 2.0, -3.0, 0.5];
        let s = cosine_similarity(&v, &v).unwrap();
        assert!((s - 1.0).abs() < 1e-6, "identical → 1.0, got {s}");
    }

    #[test]
    fn parallel_but_unnormalized_is_one() {
        // b = 3 * a: same direction, different magnitude → cosine 1 (proves the
        // self-normalization, ADR M4-20 §D-4 — CAM++ embeddings are not unit).
        let a = [1.0f32, 2.0, 2.0];
        let b = [3.0f32, 6.0, 6.0];
        let s = cosine_similarity(&a, &b).unwrap();
        assert!((s - 1.0).abs() < 1e-6, "parallel → 1.0, got {s}");
    }

    #[test]
    fn orthogonal_vectors_are_zero() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        let s = cosine_similarity(&a, &b).unwrap();
        assert!(s.abs() < 1e-6, "orthogonal → 0.0, got {s}");
    }

    #[test]
    fn opposite_vectors_are_minus_one() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [-1.0f32, -2.0, -3.0];
        let s = cosine_similarity(&a, &b).unwrap();
        assert!((s + 1.0).abs() < 1e-6, "opposite → -1.0, got {s}");
    }

    #[test]
    fn length_mismatch_is_explicit_error() {
        assert!(matches!(
            cosine_similarity(&[1.0, 2.0], &[1.0]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn empty_is_explicit_error() {
        assert!(matches!(
            cosine_similarity(&[], &[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zero_norm_is_explicit_error() {
        assert!(matches!(
            cosine_similarity(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn verify_without_threshold_is_similarity_only() {
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 1.0];
        let r = speaker_verify(&a, &b, None).unwrap();
        assert!(r.accepted.is_none());
        // cos(45°) = 1/sqrt(2) ≈ 0.7071.
        assert!((r.similarity - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn verify_threshold_accept_and_reject_branches() {
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 1.0]; // similarity ≈ 0.7071
        // Below the similarity → accepted.
        let accept = speaker_verify(&a, &b, Some(0.5)).unwrap();
        assert_eq!(accept.accepted, Some(true));
        // Above the similarity → rejected.
        let reject = speaker_verify(&a, &b, Some(0.9)).unwrap();
        assert_eq!(reject.accepted, Some(false));
        // Exactly equal → accepted (>= is inclusive).
        let eq = speaker_verify(&a, &a, Some(1.0)).unwrap();
        assert_eq!(eq.accepted, Some(true));
    }

    #[test]
    fn verify_propagates_input_errors() {
        assert!(matches!(
            speaker_verify(&[1.0], &[1.0, 2.0], Some(0.5)),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
