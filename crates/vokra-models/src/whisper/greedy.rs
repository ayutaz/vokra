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
    state: &mut DecoderState<'_>,
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
    let mut logits = state.step_last(start_ids)?;
    let mut generated = Vec::new();
    for _ in 0..max_new {
        let next = argmax(&logits);
        generated.push(next);
        if next == eot {
            break;
        }
        logits = state.step_last(&[next])?;
    }
    Ok(generated)
}

/// Index of the maximum element (first on ties), i.e. the greedy token.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::decoder::test_support::{tiny_cfg, tiny_encoder, tiny_weights};

    #[test]
    fn argmax_picks_first_max_on_ties() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn empty_start_ids_is_rejected() {
        let cfg = tiny_cfg(1);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();
        // Documented empty-prefix guard (greedy.rs line 37).
        let err = greedy_decode(&mut st, &[], /*eot*/ 999, 8).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn respects_max_new_and_is_deterministic() {
        let cfg = tiny_cfg(1);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();

        // eot is outside the vocab (argmax is always < n_vocab), so it can never
        // be produced: the loop runs to the max_new cap and yields exactly that
        // many tokens.
        let eot = cfg.n_vocab as u32 + 100;
        let run1 = greedy_decode(&mut st, &[1], eot, 3).unwrap();
        assert_eq!(run1.len(), 3, "should hit the max_new cap");
        assert!(
            run1.iter().all(|&t| (t as usize) < cfg.n_vocab),
            "every generated id is a valid in-vocab argmax: {run1:?}"
        );

        // greedy_decode resets the state internally, so a second call reproduces
        // the first bit-for-bit.
        let run2 = greedy_decode(&mut st, &[1], eot, 3).unwrap();
        assert_eq!(run1, run2);
    }
}
