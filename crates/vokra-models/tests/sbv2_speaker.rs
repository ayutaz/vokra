//! `SpeakerEmbedding` lookup tests (Task 18).
//!
//! `lookup` returns `Result<&[f32], VokraError>` rather than a bare
//! slice: an out-of-range `speaker_id` is a caller error that must
//! surface loudly (FR-EX-08), not panic or silently clamp.

use vokra_models::sbv2::SpeakerEmbedding;

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
