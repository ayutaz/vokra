//! `SpeakerEmbedding` lookup tests (Task 18) + `ExternalSpeakerProjection`
//! forward tests (Blocker 3).
//!
//! `lookup` returns `Result<&[f32], VokraError>` rather than a bare
//! slice: an out-of-range `speaker_id` is a caller error that must
//! surface loudly (FR-EX-08), not panic or silently clamp.
//!
//! Blocker 3 additions ([`ExternalSpeakerProjection`]) implement the
//! `Linear(d_speaker → d_model)` projection the real SBV2 ckpt's
//! `enc_p.encoder.spk_emb_linear.{weight,bias}` performs on an external
//! zero-shot speaker embedding — the base ckpt has **no per-speaker
//! embedding table** (`emb_g`), so the projection is the sole speaker
//! conditioning path. The tests here pin the same "linear + bias, no
//! silent reshape" contract for the projection that
//! [`SpeakerEmbedding::lookup`] already pins for the lookup path.

use vokra_models::sbv2::{ExternalSpeakerProjection, SpeakerEmbedding};

/// Valid speaker_id returns a [d_speaker] slice with the expected content.
#[test]
fn valid_speaker_id_returns_expected_slice() {
    let n_speakers = 3;
    let d_speaker = 4;
    // Populate distinct values per speaker so slice identity is verifiable.
    let mut table = Vec::with_capacity(n_speakers * d_speaker);
    for s in 0..n_speakers {
        for d in 0..d_speaker {
            table.push((s * 10 + d) as f32);
        }
    }
    let emb = SpeakerEmbedding::from_table(table, n_speakers, d_speaker);
    let slice = emb.lookup(1).expect("valid id must return Ok");
    assert_eq!(
        slice,
        &[10.0, 11.0, 12.0, 13.0],
        "speaker 1 slice must match"
    );
    assert_eq!(slice.len(), d_speaker);
}

/// Out-of-range speaker_id returns Err (FR-EX-08 loud error, not silent panic).
#[test]
fn out_of_range_speaker_id_returns_err() {
    let emb = SpeakerEmbedding::from_table(vec![0.0; 6], 3, 2);
    let result = emb.lookup(3); // n_speakers == 3, valid ids are 0..=2
    assert!(
        result.is_err(),
        "speaker_id 3 must return Err, not slice or panic"
    );
    // Verify the error message mentions the offending id and the bound.
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains('3'),
        "error message should mention the offending id: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Blocker 3: ExternalSpeakerProjection — Linear(d_in → d_out) + bias.
// ---------------------------------------------------------------------------

/// Zero-input to a zero-weight projection returns only the bias (the
/// projection's Linear layer degenerates to `y = b`). Pins the identity
/// case: the projection is a plain `y = W x + b`, no hidden non-linearity.
#[test]
fn external_projection_zero_input_returns_only_bias() {
    let d_in = 4;
    let d_out = 3;
    let weight = vec![0.0_f32; d_out * d_in];
    let bias = vec![1.0_f32, 2.0_f32, 3.0_f32];
    let proj = ExternalSpeakerProjection::from_weights(weight, bias.clone(), d_in, d_out);
    let input = vec![0.0_f32; d_in];
    let out = proj
        .forward(&input)
        .expect("in-range zero input must return Ok");
    assert_eq!(out.len(), d_out);
    assert_eq!(out, bias, "zero input × zero weight must yield only bias");
}

/// Non-trivial forward reproduces the manual `y[o] = Σ_i w[o, i] · x[i] + b[o]`
/// row-major linear formula — this is the same convention
/// [`vokra_models::sbv2::BertBridge::from_conv`] and every SBV2 module uses,
/// so a caller can verify the projection numerically against a hand-computed
/// reference.
#[test]
fn external_projection_forward_matches_manual_linear() {
    let d_in = 3;
    let d_out = 2;
    // Row-major `[d_out, d_in]`:
    //   out[0] = 1*x[0] + 2*x[1] + 3*x[2] + b[0]
    //   out[1] = 4*x[0] + 5*x[1] + 6*x[2] + b[1]
    let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = vec![0.5, -0.25];
    let proj = ExternalSpeakerProjection::from_weights(weight, bias, d_in, d_out);
    let input = vec![1.0, 2.0, -1.0];
    let out = proj.forward(&input).expect("forward must succeed");
    // Expected: [1*1 + 2*2 + 3*-1 + 0.5, 4*1 + 5*2 + 6*-1 - 0.25]
    //         = [1 + 4 - 3 + 0.5, 4 + 10 - 6 - 0.25]
    //         = [2.5, 7.75]
    assert_eq!(out.len(), d_out);
    assert!(
        (out[0] - 2.5).abs() < 1e-6,
        "out[0] should be 2.5, got {}",
        out[0]
    );
    assert!(
        (out[1] - 7.75).abs() < 1e-6,
        "out[1] should be 7.75, got {}",
        out[1]
    );
}

/// A wrong-length input surfaces loudly (FR-EX-08 — never silently
/// reshaped, never zero-padded). Verifies the returned error names
/// both the actual and expected lengths so a caller sees the discrepancy.
#[test]
fn external_projection_wrong_length_input_is_invalid_argument() {
    let d_in = 512;
    let d_out = 192;
    let weight = vec![0.0_f32; d_out * d_in];
    let bias = vec![0.0_f32; d_out];
    let proj = ExternalSpeakerProjection::from_weights(weight, bias, d_in, d_out);
    // 511 ≠ 512 → InvalidArgument (not a silent zero-pad).
    let too_short = vec![0.0_f32; 511];
    let result = proj.forward(&too_short);
    assert!(
        result.is_err(),
        "wrong-length input must return Err, not Vec"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("511"),
        "error message should name the actual length (511), got: {msg}"
    );
    assert!(
        msg.contains("512"),
        "error message should name the expected length (512), got: {msg}"
    );
}

/// Accessors expose the projection's I/O dims so a caller can validate
/// shapes before calling `forward` (avoids the InvalidArgument round-trip
/// when the caller has the input size already).
#[test]
fn external_projection_dim_accessors_return_construction_values() {
    let d_in = 512;
    let d_out = 192;
    let proj = ExternalSpeakerProjection::from_weights(
        vec![0.0_f32; d_out * d_in],
        vec![0.0_f32; d_out],
        d_in,
        d_out,
    );
    assert_eq!(proj.d_in(), d_in);
    assert_eq!(proj.d_out(), d_out);
}
