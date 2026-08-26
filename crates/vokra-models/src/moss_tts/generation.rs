use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::transformer;
use super::weights::{AUDIO_VOCAB_SIZE, HIDDEN_DIM, NUM_CODEBOOKS, NanoWeights, TEXT_VOCAB_SIZE};
use super::{
    AUDIO_ASSISTANT_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, MAX_POSITION_EMBEDDINGS,
};

const LABEL: &str = "moss_tts/nano";
const ROW_WIDTH: usize = NUM_CODEBOOKS + 1;

/// Frame-major codes produced by the 16 autoregressive local heads.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossTtsGeneratedCodes {
    /// `[frames, 16]` frame-major code matrix.
    pub codes: Vec<u32>,
    /// Number of generated codec frames.
    pub frames: usize,
    /// Always 16 for Nano.
    pub num_codebooks: usize,
}

pub(super) fn generate_codes(
    compute: &Compute,
    weights: &NanoWeights,
    prompt_rows: &[u32],
    max_new_frames: usize,
) -> Result<MossTtsGeneratedCodes> {
    validate_prompt_rows(prompt_rows)?;
    let prompt_len = prompt_rows.len() / ROW_WIDTH;
    let maximum_len = prompt_len.checked_add(max_new_frames).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: prompt length + max_new_frames overflows usize"
        ))
    })?;
    if maximum_len > MAX_POSITION_EMBEDDINGS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt {prompt_len} + max_new_frames {max_new_frames} exceeds the official {MAX_POSITION_EMBEDDINGS}-position context"
        )));
    }

    let mut current_rows = prompt_rows.to_vec();
    let mut codes = Vec::with_capacity(max_new_frames.saturating_mul(NUM_CODEBOOKS));
    let mut frames = 0usize;
    for _ in 0..max_new_frames {
        let rows = current_rows.len() / ROW_WIDTH;
        let global_inputs = build_global_embeddings(weights, &current_rows)?;
        let global_hidden = transformer::forward(compute, &global_inputs, rows, &weights.global)?;
        let global_last = last_hidden(&global_hidden, rows)?;

        let mut local_inputs = global_last.to_vec();
        let local_hidden = transformer::forward(compute, &local_inputs, 1, &weights.local)?;
        let text_logits = weights
            .text_head
            .forward(compute, last_hidden(&local_hidden, 1)?, 1)?;
        let continue_logit = finite_logit(
            &text_logits,
            AUDIO_ASSISTANT_SLOT_TOKEN_ID as usize,
            "assistant-slot text logit",
        )?;
        let stop_logit = finite_logit(
            &text_logits,
            AUDIO_END_TOKEN_ID as usize,
            "audio-end text logit",
        )?;
        // Upstream candidate order is [assistant_slot, audio_end], so an
        // exact tie continues (`argmax` selects the first candidate).
        if continue_logit < stop_logit {
            break;
        }

        append_embedding(
            &mut local_inputs,
            &weights.text_embedding,
            AUDIO_ASSISTANT_SLOT_TOKEN_ID as usize,
            TEXT_VOCAB_SIZE,
            "assistant-slot text embedding",
        )?;
        let mut frame = [0u32; NUM_CODEBOOKS];
        for (codebook, frame_token) in frame.iter_mut().enumerate() {
            let local_rows = local_inputs.len() / HIDDEN_DIM;
            let local_hidden =
                transformer::forward(compute, &local_inputs, local_rows, &weights.local)?;
            let logits = weights.audio_heads[codebook].forward(
                compute,
                last_hidden(&local_hidden, local_rows)?,
                1,
            )?;
            let token = finite_argmax(&logits, &format!("audio logits[{codebook}]"))?;
            *frame_token = token as u32;
            append_embedding(
                &mut local_inputs,
                &weights.audio_embeddings[codebook],
                token,
                AUDIO_VOCAB_SIZE,
                &format!("audio embedding[{codebook}]"),
            )?;
        }

        codes.extend(frame);
        current_rows.push(AUDIO_ASSISTANT_SLOT_TOKEN_ID);
        current_rows.extend(frame);
        frames += 1;
    }
    Ok(MossTtsGeneratedCodes {
        codes,
        frames,
        num_codebooks: NUM_CODEBOOKS,
    })
}

fn validate_prompt_rows(prompt_rows: &[u32]) -> Result<()> {
    if prompt_rows.is_empty() || !prompt_rows.len().is_multiple_of(ROW_WIDTH) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt must be a non-empty frame-major [rows, {ROW_WIDTH}] u32 matrix; got {} values",
            prompt_rows.len()
        )));
    }
    let rows = prompt_rows.len() / ROW_WIDTH;
    if rows > MAX_POSITION_EMBEDDINGS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt has {rows} rows, exceeding {MAX_POSITION_EMBEDDINGS}"
        )));
    }
    for row in 0..rows {
        let text = prompt_rows[row * ROW_WIDTH] as usize;
        if text >= TEXT_VOCAB_SIZE {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt[{row},0]={text} is outside text vocabulary 0..{TEXT_VOCAB_SIZE}"
            )));
        }
        for codebook in 0..NUM_CODEBOOKS {
            let token = prompt_rows[row * ROW_WIDTH + 1 + codebook];
            if token != AUDIO_PAD_TOKEN_ID && token as usize >= AUDIO_VOCAB_SIZE {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: prompt[{row},{}]={token} is neither audio pad {AUDIO_PAD_TOKEN_ID} nor code 0..{AUDIO_VOCAB_SIZE}",
                    codebook + 1
                )));
            }
        }
    }
    Ok(())
}

fn build_global_embeddings(weights: &NanoWeights, prompt_rows: &[u32]) -> Result<Vec<f32>> {
    let rows = prompt_rows.len() / ROW_WIDTH;
    let mut output = Vec::with_capacity(rows * HIDDEN_DIM);
    for row in 0..rows {
        let text = prompt_rows[row * ROW_WIDTH] as usize;
        output.extend_from_slice(embedding_row(
            &weights.text_embedding,
            text,
            TEXT_VOCAB_SIZE,
            "text embedding",
        )?);
        let target = &mut output[row * HIDDEN_DIM..(row + 1) * HIDDEN_DIM];
        for codebook in 0..NUM_CODEBOOKS {
            let token = prompt_rows[row * ROW_WIDTH + 1 + codebook];
            if token == AUDIO_PAD_TOKEN_ID {
                continue;
            }
            let audio = embedding_row(
                &weights.audio_embeddings[codebook],
                token as usize,
                AUDIO_VOCAB_SIZE,
                "audio embedding",
            )?;
            for (value, addition) in target.iter_mut().zip(audio) {
                *value += addition;
            }
        }
    }
    Ok(output)
}

fn append_embedding(
    output: &mut Vec<f32>,
    table: &[f32],
    token: usize,
    vocabulary: usize,
    label: &str,
) -> Result<()> {
    output.extend_from_slice(embedding_row(table, token, vocabulary, label)?);
    Ok(())
}

fn embedding_row<'a>(
    table: &'a [f32],
    token: usize,
    vocabulary: usize,
    label: &str,
) -> Result<&'a [f32]> {
    if table.len() != vocabulary * HIDDEN_DIM || token >= vocabulary {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} shape/token mismatch: table={}, token={token}, vocabulary={vocabulary}, hidden={HIDDEN_DIM}",
            table.len()
        )));
    }
    Ok(&table[token * HIDDEN_DIM..(token + 1) * HIDDEN_DIM])
}

fn last_hidden(values: &[f32], rows: usize) -> Result<&[f32]> {
    if rows == 0 || values.len() != rows * HIDDEN_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: hidden state has {} values, expected [{rows}, {HIDDEN_DIM}]",
            values.len()
        )));
    }
    Ok(&values[(rows - 1) * HIDDEN_DIM..rows * HIDDEN_DIM])
}

fn finite_logit(logits: &[f32], index: usize, label: &str) -> Result<f32> {
    let value = logits.get(index).copied().ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: {label} index {index} is outside {} logits",
            logits.len()
        ))
    })?;
    if !value.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} is non-finite: {value}"
        )));
    }
    Ok(value)
}

fn finite_argmax(values: &[f32], label: &str) -> Result<usize> {
    let (&first, tail) = values
        .split_first()
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: {label} is empty")))?;
    if !first.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label}[0] is non-finite: {first}"
        )));
    }
    let mut best_index = 0usize;
    let mut best_value = first;
    for (offset, &value) in tail.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: {label}[{}] is non-finite: {value}",
                offset + 1
            )));
        }
        if value > best_value {
            best_index = offset + 1;
            best_value = value;
        }
    }
    Ok(best_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_keeps_first_index_on_ties() {
        assert_eq!(finite_argmax(&[2.0, 2.0, 1.0], "test").unwrap(), 0);
    }

    #[test]
    fn prompt_rejects_foreign_audio_sentinels() {
        let mut row = [AUDIO_PAD_TOKEN_ID; ROW_WIDTH];
        row[0] = 4;
        row[3] = AUDIO_PAD_TOKEN_ID + 1;
        assert!(validate_prompt_rows(&row).is_err());
    }
}
