//! SBV2 speaker conditioning: two co-existing paths.
//!
//! - [`SpeakerEmbedding`]: `speaker_id -> [d_speaker]` table row. Used by
//!   `SbV2Model::synthetic_for_test` and any legacy converter that emits a
//!   `sbv2.speaker.table` tensor.
//! - [`ExternalSpeakerProjection`] (Blocker 3): the real SBV2 v2 base
//!   checkpoint has **no** per-speaker embedding table; instead the
//!   caller supplies an external `[d_speaker=512]` zero-shot embedding
//!   which `enc_p.encoder.spk_emb_linear.{weight,bias}` projects to
//!   `[d_model=192]` via a bias-full `Linear(d_speaker → d_model)`. This
//!   struct implements that projection as a plain
//!   `y[o] = Σ_i w[o, i] · x[i] + b[o]` row-major linear map, matching
//!   the convention every other SBV2 module in this crate uses (see e.g.
//!   `text_encoder::BertBridge::from_conv`).
//!
//! Both paths co-exist: [`SbV2Model`](super::SbV2Model) holds a
//! `SpeakerEmbedding` (always populated so existing synthetic tests keep
//! working) plus an `Option<ExternalSpeakerProjection>` (`Some` when a
//! real-ckpt loader binds the `spk_emb_linear` weights). Synthesize
//! dispatches on `(request.speaker_embedding, model.speaker_projection)`
//! — see `SbV2Model::synthesize`'s step 5 for the exact table.
//!
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

/// External zero-shot speaker-embedding projection (Blocker 3): a
/// bias-full linear map `Linear(d_in → d_out)` — for the real SBV2 v2
/// base ckpt, `d_in = d_speaker = 512` (the external caller-supplied
/// embedding width) and `d_out = d_model = 192` (the text-encoder hidden
/// width the projection's output broadcast-adds into). Matches the real
/// ckpt's `enc_p.encoder.spk_emb_linear.{weight,bias}` tensors verbatim
/// (row-major `[d_out, d_in]` weight + `[d_out]` bias), so a converter
/// that renames these two tensors into the `sbv2.text_encoder.*` chunk
/// can bind them here 1:1.
///
/// Kept as its own type — rather than folded into [`SpeakerEmbedding`]
/// — because the two paths have distinct shapes (lookup vs. project) and
/// distinct semantic contracts: the lookup path indexes a
/// caller-supplied discrete speaker id into a pre-trained table,
/// whereas this projection consumes a caller-supplied continuous
/// embedding via `y = W · x + b`. See this module's doc for the full
/// dispatch table.
pub struct ExternalSpeakerProjection {
    /// Row-major `[d_out, d_in]` projection weight
    /// (`y[o] = Σ_i weight[o, i] · x[i] + bias[o]` — same convention as
    /// [`super::text_encoder::BertBridge::from_conv`]).
    weight: Vec<f32>,
    /// `[d_out]` per-output-channel bias.
    bias: Vec<f32>,
    /// Input width — the caller-supplied external embedding length. Real
    /// SBV2 v2 base ckpt: 512.
    d_in: usize,
    /// Output width — the text-encoder hidden width the projection's
    /// output broadcast-adds into. Real SBV2 v2 base ckpt: 192 (`d_model`).
    d_out: usize,
}

impl ExternalSpeakerProjection {
    /// Builds a projection from pre-trained `[d_out, d_in]` weight and
    /// `[d_out]` bias tensors.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — same
    /// setup-time convention as
    /// [`super::style::StyleVectorInjector::from_projections`]'s panic
    /// docs) if `weight.len() != d_out * d_in` or `bias.len() != d_out`.
    pub fn from_weights(weight: Vec<f32>, bias: Vec<f32>, d_in: usize, d_out: usize) -> Self {
        debug_assert_eq!(weight.len(), d_out * d_in, "weight must be [d_out, d_in]");
        debug_assert_eq!(bias.len(), d_out, "bias must be [d_out]");
        Self {
            weight,
            bias,
            d_in,
            d_out,
        }
    }

    /// The projection's input width — the length every `forward` call's
    /// `embedding` slice must have. Real SBV2 v2 base ckpt: 512.
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// The projection's output width — the length of every `forward`
    /// call's returned `Vec`. Real SBV2 v2 base ckpt: 192 (`d_model`).
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Projects a `[d_in]` external embedding into a `[d_out]`
    /// broadcast-add contribution: `out[o] = Σ_i weight[o, i] ·
    /// embedding[i] + bias[o]`.
    ///
    /// Unlike [`SpeakerEmbedding::from_table`]'s constructor shape check
    /// (`debug_assert!`, setup-time — see its panic docs), the
    /// `embedding` slice here is caller-supplied at call time
    /// (a request field, a CLI flag, ...), so a wrong-length input can
    /// reach this method in a release build. Per FR-EX-08 (no silent
    /// fallback, no silent reshape) this returns an `Err` instead of
    /// panicking or truncating / zero-padding.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `embedding.len() !=
    /// self.d_in`, naming both the actual and expected lengths.
    /// [`VokraError::InvalidArgument`] is chosen over
    /// [`VokraError::ModelLoad`] because this is a caller-argument
    /// validation at call time, not a model file parse/load failure —
    /// see `crates/vokra-core/src/error.rs:40-41`.
    pub fn forward(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if embedding.len() != self.d_in {
            return Err(VokraError::InvalidArgument(format!(
                "ExternalSpeakerProjection::forward: embedding length {} does not match \
                 d_in {} — the projection expects a fixed-width external speaker embedding \
                 (FR-EX-08: no silent zero-pad/truncate)",
                embedding.len(),
                self.d_in,
            )));
        }
        // Plain row-major linear map + bias. Not a hot path (one call
        // per synthesize, projecting 512 → 192 = ~98k FMAs), so an
        // explicit nested loop is more readable than a SIMD dispatch;
        // matches [`super::text_encoder::BertBridge::forward`]'s
        // linear-projection subroutine `linear_rows_biased`.
        let mut out = vec![0.0_f32; self.d_out];
        for ((o_slot, row), &b) in out
            .iter_mut()
            .zip(self.weight.chunks_exact(self.d_in))
            .zip(self.bias.iter())
        {
            let mut acc = b;
            for (w, x) in row.iter().zip(embedding.iter()) {
                acc += w * x;
            }
            *o_slot = acc;
        }
        Ok(out)
    }
}
