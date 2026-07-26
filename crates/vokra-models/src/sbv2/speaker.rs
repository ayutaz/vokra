//! SBV2 speaker-embedding lookup: `speaker_id -> [d_speaker]` table row.
//! (Clean-room comment: see `mod.rs`.)

use vokra_core::{Result, VokraError};

/// Per-speaker embedding table: a flat, row-major `[n_speakers,
/// d_speaker]` buffer, indexed by `speaker_id` to a `[d_speaker]` slice
/// (see [`lookup`](SpeakerEmbedding::lookup)).
pub struct SpeakerEmbedding {
    /// Row-major `[n_speakers, d_speaker]` embedding table
    /// (`table[id * d_speaker .. (id + 1) * d_speaker]` addresses speaker
    /// `id`'s embedding row).
    table: Vec<f32>,
    /// Speaker count (`table.len() == n_speakers * d_speaker`).
    n_speakers: usize,
    /// Per-speaker embedding dimensionality.
    d_speaker: usize,
}

impl SpeakerEmbedding {
    /// Builds a speaker-embedding table from a pre-trained row-major
    /// `[n_speakers, d_speaker]` buffer.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs for why this crate uses `debug_assert!` rather than
    /// `Result` for constructor shape checks) if `table.len() !=
    /// n_speakers * d_speaker`.
    pub fn from_table(table: Vec<f32>, n_speakers: usize, d_speaker: usize) -> Self {
        debug_assert_eq!(
            table.len(),
            n_speakers * d_speaker,
            "table must be [n_speakers, d_speaker]"
        );
        Self {
            table,
            n_speakers,
            d_speaker,
        }
    }

    /// Looks up `speaker_id`'s `[d_speaker]` embedding row.
    ///
    /// Unlike this module's constructor shape check (`debug_assert!` in
    /// [`from_table`](Self::from_table) — a hot/setup-time invariant, see
    /// its panic docs), `speaker_id` here is externally sourced (a request
    /// field, a CLI argument, ...), so a caller can reach this with an
    /// out-of-range id in a release build. Per FR-EX-08 (no silent
    /// fallback, no silent panic) this returns an `Err` instead of
    /// asserting or panicking.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `speaker_id` (as
    /// `usize`) is `>= n_speakers`, naming the offending id and the valid
    /// bound. [`VokraError::InvalidArgument`] is chosen over
    /// [`VokraError::ModelLoad`] because this is a caller-argument
    /// validation at call time, not a model file parse/load failure — see
    /// `crates/vokra-core/src/error.rs:40-41`.
    pub fn lookup(&self, speaker_id: u32) -> Result<&[f32]> {
        let id = speaker_id as usize;
        if id >= self.n_speakers {
            return Err(VokraError::InvalidArgument(format!(
                "SpeakerEmbedding::lookup: speaker_id {} out of range — table has {} \
                 speaker(s), valid ids are 0..{}",
                speaker_id, self.n_speakers, self.n_speakers,
            )));
        }
        let start = id * self.d_speaker;
        Ok(&self.table[start..start + self.d_speaker])
    }
}
