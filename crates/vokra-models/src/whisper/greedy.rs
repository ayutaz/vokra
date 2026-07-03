//! Greedy decode loop for Whisper.
//!
//! Plain argmax decoding: build the forced special-token prefix
//! (`<|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>`, read from
//! `vokra.whisper.decoder_start_ids`), run the decoder over the prefix, then
//! repeatedly take the arg-max token and feed it back until the end-of-
//! transcript token or a length cap.
//!
//! This is the pre-search baseline and the end-to-end parity path (M0-06-T17,
//! T18). It deliberately applies **no logit suppression** (Whisper's
//! `suppress_tokens` / `begin_suppress_tokens` are a generation-config concern,
//! not part of the model forward), so the reference dump uses the identical
//! plain-argmax loop and the two agree token-for-token.

use vokra_core::decode::argmax;
use vokra_core::{Result, VokraError};

use super::decoder::DecoderState;

/// Default cap on generated tokens when the caller passes `None` — Whisper's
/// per-window text budget is `n_text_ctx / 2 = 224`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 224;

/// Greedily decodes starting from `start_ids`, returning the generated token
/// ids (the forced prefix is **not** included). Generation stops at `eot`
/// (which **is** included in the result, matching HF) or after `max_new`
/// tokens.
///
/// # Errors
///
/// Propagates decoder errors (out-of-range token, `n_text_ctx` overflow).
pub fn greedy_decode(
    state: &mut DecoderState,
    start_ids: &[u32],
    eot: u32,
    max_new: usize,
) -> Result<Vec<u32>> {
    if start_ids.is_empty() {
        return Err(VokraError::InvalidArgument(
            "greedy_decode: start_ids must not be empty".into(),
        ));
    }
    state.reset();
    // Alloc-free hot loop: `step_into` leaves the logits in the decoder's reused
    // scratch and `last_logits_row` borrows them, so no per-token logits `Vec`
    // is allocated (only `generated` grows). Same argmax on the same logits as
    // the former `step_last`, so the produced token sequence is unchanged.
    state.step_into(start_ids)?;
    let mut generated = Vec::new();
    for _ in 0..max_new {
        let next = argmax(state.last_logits_row());
        generated.push(next);
        if next == eot {
            break;
        }
        state.step_into(&[next])?;
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::decoder::test_support::{tiny_encoder, tiny_model};

    #[test]
    fn argmax_picks_first_max_on_ties() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn empty_start_ids_is_rejected() {
        let model = tiny_model(1);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();
        // Documented empty-prefix guard (greedy.rs line 37).
        let err = greedy_decode(&mut st, &[], /*eot*/ 999, 8).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn respects_max_new_and_is_deterministic() {
        let model = tiny_model(1);
        let n_vocab = model.config().n_vocab;
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();

        // eot is outside the vocab (argmax is always < n_vocab), so it can never
        // be produced: the loop runs to the max_new cap and yields exactly that
        // many tokens.
        let eot = n_vocab as u32 + 100;
        let run1 = greedy_decode(&mut st, &[1], eot, 3).unwrap();
        assert_eq!(run1.len(), 3, "should hit the max_new cap");
        assert!(
            run1.iter().all(|&t| (t as usize) < n_vocab),
            "every generated id is a valid in-vocab argmax: {run1:?}"
        );

        // greedy_decode resets the state internally, so a second call reproduces
        // the first bit-for-bit.
        let run2 = greedy_decode(&mut st, &[1], eot, 3).unwrap();
        assert_eq!(run1, run2);
    }
}
