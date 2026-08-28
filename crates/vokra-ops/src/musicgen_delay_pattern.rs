//! MusicGen / AudioGen multi-codebook delay-pattern scheduling.
//!
//! The autoregressive decoder predicts several residual-codebook streams in
//! parallel.  Codebook `c` is shifted by `c` positions (or by `c / 2` for
//! interleaved stereo codebooks), with beginning/end padding around the
//! prediction window.  This module owns that deterministic host-side token
//! layout so MusicGen and AudioGen cannot grow divergent copies.
//!
//! This is not a learned tensor kernel and therefore is not a CPU fallback:
//! attention, projection and sampling math remains on the backend selected by
//! the consuming model.  The layout follows the Apache-2.0 Transformers
//! `MusicgenForCausalLM.build_delay_pattern_mask` and
//! `apply_delay_pattern_mask` implementation.  Invalid dimensions fail
//! explicitly instead of relying on a tensor slice assignment to panic.

use vokra_core::{Result, VokraError};

/// Sentinel stored in [`MusicGenDelayPattern::pattern`] where a generated
/// token must be preserved rather than replaced by a prompt/padding token.
pub const MUSICGEN_PREDICT_TOKEN: i64 = -1;

/// Explicit geometry for the MusicGen-family delay pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicGenDelayPatternAttrs {
    /// Number of independently generated examples.
    pub batch_size: usize,
    /// Total codebook rows per example. Stereo rows are interleaved
    /// `[left_0, right_0, left_1, right_1, ...]`.
    pub num_codebooks: usize,
    /// Current prompt width in each codebook row, including the decoder start
    /// token used by the upstream generation API.
    pub prompt_len: usize,
    /// Full delayed sequence width, including beginning/end padding.
    pub max_length: usize,
    /// One for mono or two for interleaved stereo generation.
    pub audio_channels: usize,
    /// Decoder padding/BOS token. For released MusicGen checkpoints this is
    /// one past the learned codebook vocabulary.
    pub pad_token_id: u32,
}

impl MusicGenDelayPatternAttrs {
    fn rows(self) -> Result<usize> {
        self.batch_size
            .checked_mul(self.num_codebooks)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "musicgen delay pattern batch_size * num_codebooks overflows usize".to_owned(),
                )
            })
    }

    fn channel_codebooks(self) -> Result<usize> {
        if self.batch_size == 0
            || self.num_codebooks == 0
            || self.prompt_len == 0
            || self.max_length == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "musicgen delay pattern axes must be non-zero, got batch_size={}, num_codebooks={}, prompt_len={}, max_length={}",
                self.batch_size, self.num_codebooks, self.prompt_len, self.max_length
            )));
        }
        if self.audio_channels != 1 && self.audio_channels != 2 {
            return Err(VokraError::InvalidArgument(format!(
                "musicgen delay pattern audio_channels must be 1 or 2, got {}",
                self.audio_channels
            )));
        }
        if self.audio_channels == 2 && self.num_codebooks % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "musicgen stereo delay pattern requires an even num_codebooks, got {}",
                self.num_codebooks
            )));
        }
        if self.prompt_len > self.max_length {
            return Err(VokraError::InvalidArgument(format!(
                "musicgen delay pattern prompt_len {} exceeds max_length {}",
                self.prompt_len, self.max_length
            )));
        }
        Ok(self.num_codebooks / self.audio_channels)
    }
}

/// A full delayed mask plus the prefix fed to the first decoder step.
///
/// Both buffers are batch-major, then codebook-major. `prefix` has logical
/// shape `[batch_size * num_codebooks, prefix_len]`; `pattern` has shape
/// `[batch_size * num_codebooks, max_length]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicGenDelayPattern {
    /// Prompt/padding prefix supplied before autoregressive prediction starts.
    pub prefix: Vec<u32>,
    /// Full prompt/padding/prediction mask. Non-negative entries override a
    /// generated token; [`MUSICGEN_PREDICT_TOKEN`] preserves it.
    pub pattern: Vec<i64>,
    /// Number of batch/codebook rows in both buffers.
    pub rows: usize,
    /// Logical width of each row in [`Self::prefix`].
    pub prefix_len: usize,
    /// Logical width of each row in [`Self::pattern`].
    pub max_length: usize,
}

/// Construct the exact multi-codebook delay pattern used by MusicGen and
/// AudioGen generation.
///
/// `input_ids` has logical shape
/// `[batch_size * num_codebooks, prompt_len]`, batch-major and then
/// codebook-major. When `max_length` is too short to contain one complete
/// staggered window, this preserves the official short-sequence behaviour:
/// the original prompt is returned unchanged and the whole mask remains
/// predictive.
pub fn build_musicgen_delay_pattern(
    input_ids: &[u32],
    attrs: MusicGenDelayPatternAttrs,
) -> Result<MusicGenDelayPattern> {
    let channel_codebooks = attrs.channel_codebooks()?;
    let rows = attrs.rows()?;
    let expected_input = rows.checked_mul(attrs.prompt_len).ok_or_else(|| {
        VokraError::InvalidArgument("musicgen delay pattern input shape overflows usize".to_owned())
    })?;
    if input_ids.len() != expected_input {
        return Err(VokraError::InvalidArgument(format!(
            "musicgen delay pattern input_ids.len() {} != batch_size * num_codebooks * prompt_len {}",
            input_ids.len(),
            expected_input
        )));
    }
    let pattern_len = rows.checked_mul(attrs.max_length).ok_or_else(|| {
        VokraError::InvalidArgument(
            "musicgen delay pattern output shape overflows usize".to_owned(),
        )
    })?;
    let minimum_staggered_length = channel_codebooks
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "musicgen delay pattern minimum staggered length overflows usize".to_owned(),
            )
        })?;

    if attrs.max_length < minimum_staggered_length {
        return Ok(MusicGenDelayPattern {
            prefix: input_ids.to_vec(),
            pattern: vec![MUSICGEN_PREDICT_TOKEN; pattern_len],
            rows,
            prefix_len: attrs.prompt_len,
            max_length: attrs.max_length,
        });
    }

    let largest_delay = channel_codebooks - 1;
    let shifted_prompt_end = attrs.prompt_len.checked_add(largest_delay).ok_or_else(|| {
        VokraError::InvalidArgument(
            "musicgen delay pattern shifted prompt length overflows usize".to_owned(),
        )
    })?;
    if shifted_prompt_end > attrs.max_length {
        return Err(VokraError::InvalidArgument(format!(
            "musicgen delay pattern shifted prompt needs {} positions but max_length is {}",
            shifted_prompt_end, attrs.max_length
        )));
    }

    let mut pattern = vec![MUSICGEN_PREDICT_TOKEN; pattern_len];
    let eos_base = attrs.max_length - channel_codebooks + 1;
    for batch in 0..attrs.batch_size {
        for codebook in 0..attrs.num_codebooks {
            let row = batch * attrs.num_codebooks + codebook;
            let delay = if attrs.audio_channels == 2 {
                codebook / 2
            } else {
                codebook
            };
            let input_base = row * attrs.prompt_len;
            let output_base = row * attrs.max_length;
            for position in 0..attrs.prompt_len {
                pattern[output_base + delay + position] =
                    i64::from(input_ids[input_base + position]);
            }

            let eos_start = eos_base + delay;
            for position in 0..attrs.max_length {
                if position <= delay || position >= eos_start {
                    pattern[output_base + position] = i64::from(attrs.pad_token_id);
                }
            }
        }
    }

    // Transformers starts generation at the earliest predictive position in
    // codebook zero across the batch. If none exists, it returns `seq_len`.
    let mut prefix_len = None;
    for batch in 0..attrs.batch_size {
        let first_codebook_base = batch * attrs.num_codebooks * attrs.max_length;
        if let Some(position) = pattern[first_codebook_base..first_codebook_base + attrs.max_length]
            .iter()
            .position(|&token| token == MUSICGEN_PREDICT_TOKEN)
        {
            prefix_len = Some(prefix_len.map_or(position, |current: usize| current.min(position)));
        }
    }
    let prefix_len = prefix_len.unwrap_or(attrs.prompt_len);
    let prefix_capacity = rows.checked_mul(prefix_len).ok_or_else(|| {
        VokraError::InvalidArgument(
            "musicgen delay pattern prefix shape overflows usize".to_owned(),
        )
    })?;
    let mut prefix = Vec::with_capacity(prefix_capacity);
    for row in 0..rows {
        let base = row * attrs.max_length;
        for &token in &pattern[base..base + prefix_len] {
            let token = u32::try_from(token).map_err(|_| {
                VokraError::InvalidArgument(format!(
                    "musicgen delay pattern prefix unexpectedly contains predictive sentinel at row {row}"
                ))
            })?;
            prefix.push(token);
        }
    }

    Ok(MusicGenDelayPattern {
        prefix,
        pattern,
        rows,
        prefix_len,
        max_length: attrs.max_length,
    })
}

/// Apply a previously built delay pattern to generated token rows in place.
///
/// `input_ids` has shape `[rows, seq_len]`. Prompt/padding entries from the
/// mask overwrite the corresponding generated value; predictive sentinel
/// entries leave it untouched. The full mask is validated so a malformed
/// future position cannot remain latent until a later generation step.
pub fn apply_musicgen_delay_pattern(
    input_ids: &mut [u32],
    rows: usize,
    seq_len: usize,
    pattern: &[i64],
    max_length: usize,
) -> Result<()> {
    if rows == 0 || seq_len == 0 || max_length == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "musicgen delay pattern apply axes must be non-zero, got rows={rows}, seq_len={seq_len}, max_length={max_length}"
        )));
    }
    if seq_len > max_length {
        return Err(VokraError::InvalidArgument(format!(
            "musicgen delay pattern apply seq_len {seq_len} exceeds max_length {max_length}"
        )));
    }
    let expected_input = rows.checked_mul(seq_len).ok_or_else(|| {
        VokraError::InvalidArgument(
            "musicgen delay pattern apply input shape overflows usize".to_owned(),
        )
    })?;
    let expected_pattern = rows.checked_mul(max_length).ok_or_else(|| {
        VokraError::InvalidArgument(
            "musicgen delay pattern apply mask shape overflows usize".to_owned(),
        )
    })?;
    if input_ids.len() != expected_input || pattern.len() != expected_pattern {
        return Err(VokraError::InvalidArgument(format!(
            "musicgen delay pattern apply shape mismatch: input_ids.len()={} (expected {expected_input}), pattern.len()={} (expected {expected_pattern})",
            input_ids.len(),
            pattern.len()
        )));
    }
    for (index, &token) in pattern.iter().enumerate() {
        if token != MUSICGEN_PREDICT_TOKEN && u32::try_from(token).is_err() {
            return Err(VokraError::InvalidArgument(format!(
                "musicgen delay pattern contains invalid token {token} at flat index {index}"
            )));
        }
    }
    for row in 0..rows {
        for position in 0..seq_len {
            let mask_token = pattern[row * max_length + position];
            if mask_token != MUSICGEN_PREDICT_TOKEN {
                input_ids[row * seq_len + position] = mask_token as u32;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: u32 = 2_048;

    fn mono_attrs(prompt_len: usize, max_length: usize) -> MusicGenDelayPatternAttrs {
        MusicGenDelayPatternAttrs {
            batch_size: 1,
            num_codebooks: 4,
            prompt_len,
            max_length,
            audio_channels: 1,
            pad_token_id: PAD,
        }
    }

    #[test]
    fn no_prompt_layout_matches_transformers_documented_example() {
        let got = build_musicgen_delay_pattern(&[PAD; 4], mono_attrs(1, 8)).unwrap();
        let p = i64::from(PAD);
        let x = MUSICGEN_PREDICT_TOKEN;
        assert_eq!(got.rows, 4);
        assert_eq!(got.prefix_len, 1);
        assert_eq!(got.prefix, vec![PAD; 4]);
        assert_eq!(
            got.pattern,
            vec![
                p, x, x, x, x, p, p, p, // codebook 0
                p, p, x, x, x, x, p, p, // codebook 1
                p, p, p, x, x, x, x, p, // codebook 2
                p, p, p, p, x, x, x, x, // codebook 3
            ]
        );
    }

    #[test]
    fn prompted_layout_matches_transformers_documented_example() {
        let input = [PAD, 10, 11, PAD, 12, 13, PAD, 14, 15, PAD, 16, 17];
        let got = build_musicgen_delay_pattern(&input, mono_attrs(3, 8)).unwrap();
        let p = i64::from(PAD);
        let x = MUSICGEN_PREDICT_TOKEN;
        assert_eq!(got.prefix_len, 3);
        assert_eq!(
            got.prefix,
            vec![PAD, 10, 11, PAD, PAD, 12, PAD, PAD, PAD, PAD, PAD, PAD]
        );
        assert_eq!(
            got.pattern,
            vec![
                p, 10, 11, x, x, p, p, p, // codebook 0
                p, p, 12, 13, x, x, p, p, // codebook 1
                p, p, p, 14, 15, x, x, p, // codebook 2
                p, p, p, p, 16, 17, x, x, // codebook 3
            ]
        );
    }

    #[test]
    fn apply_overwrites_only_prompt_and_padding_positions() {
        let input = [PAD, 10, 11, PAD, 12, 13, PAD, 14, 15, PAD, 16, 17];
        let built = build_musicgen_delay_pattern(&input, mono_attrs(3, 8)).unwrap();
        let mut generated = vec![99; 4 * 5];
        apply_musicgen_delay_pattern(&mut generated, 4, 5, &built.pattern, 8).unwrap();
        assert_eq!(
            generated,
            vec![
                PAD, 10, 11, 99, 99, // codebook 0
                PAD, PAD, 12, 13, 99, // codebook 1
                PAD, PAD, PAD, 14, 15, // codebook 2
                PAD, PAD, PAD, PAD, 16, // codebook 3
            ]
        );
    }

    #[test]
    fn stereo_interleaves_rows_with_the_same_channel_delay() {
        let attrs = MusicGenDelayPatternAttrs {
            batch_size: 1,
            num_codebooks: 4,
            prompt_len: 1,
            max_length: 5,
            audio_channels: 2,
            pad_token_id: PAD,
        };
        let got = build_musicgen_delay_pattern(&[PAD; 4], attrs).unwrap();
        let p = i64::from(PAD);
        let x = MUSICGEN_PREDICT_TOKEN;
        assert_eq!(
            got.pattern,
            vec![
                p, x, x, x, p, // left codebook 0
                p, x, x, x, p, // right codebook 0
                p, p, x, x, x, // left codebook 1
                p, p, x, x, x, // right codebook 1
            ]
        );
    }

    #[test]
    fn short_sequence_preserves_official_bypass_behaviour() {
        let input = [PAD, PAD, PAD, PAD];
        let got = build_musicgen_delay_pattern(&input, mono_attrs(1, 6)).unwrap();
        assert_eq!(got.prefix, input);
        assert_eq!(got.prefix_len, 1);
        assert_eq!(got.pattern, vec![MUSICGEN_PREDICT_TOKEN; 4 * 6]);
    }

    #[test]
    fn malformed_geometry_and_shapes_fail_loudly() {
        let mut attrs = mono_attrs(1, 8);
        attrs.batch_size = 0;
        assert!(matches!(
            build_musicgen_delay_pattern(&[], attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut attrs = mono_attrs(1, 8);
        attrs.audio_channels = 3;
        assert!(matches!(
            build_musicgen_delay_pattern(&[PAD; 4], attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut attrs = mono_attrs(1, 8);
        attrs.audio_channels = 2;
        attrs.num_codebooks = 3;
        assert!(matches!(
            build_musicgen_delay_pattern(&[PAD; 3], attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        assert!(matches!(
            build_musicgen_delay_pattern(&[PAD; 3], mono_attrs(1, 8)),
            Err(VokraError::InvalidArgument(_))
        ));

        let attrs = MusicGenDelayPatternAttrs {
            batch_size: 1,
            num_codebooks: 2,
            prompt_len: 3,
            max_length: 3,
            audio_channels: 1,
            pad_token_id: PAD,
        };
        assert!(matches!(
            build_musicgen_delay_pattern(&[PAD; 6], attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut generated = [0; 2];
        assert!(matches!(
            apply_musicgen_delay_pattern(&mut generated, 1, 2, &[-2, -1], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            apply_musicgen_delay_pattern(&mut generated, 1, 2, &[-1], 2),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
