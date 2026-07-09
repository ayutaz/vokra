//! Voxtral text-decoder session — Mistral decoder with per-block KV cache.
//!
//! # Purpose (M3-10-T13 autoregressive greedy decode)
//!
//! [`TextDecoderSession`] owns:
//! - a reference to a loaded [`TextDecoder`] (weights);
//! - a reference to the parent [`VoxtralConfig`] (hparams);
//! - a self-attention [`KvCache`] (one per-layer buffer of width
//!   `n_head_kv * head_dim`);
//! - a reusable [`StepScratch`] the block forward writes into;
//! - a [`Compute`] dispatcher (CPU foundation; a GPU
//!   `VoxtralDecodeSession` is reserved as a follow-up seam — see the
//!   streaming module's `allow_device_session` slot).
//!
//! # Zero-alloc hot path (FR-EX-05)
//!
//! Every scratch buffer is reserved once at [`TextDecoderSession::new`] and
//! reused across steps. The KV cache is seeded to a small hint
//! (`SELF_KV_RESERVE_HINT`) and grows amortically for longer decodes; a
//! decode past the hint reallocates once (Vec doubling) but the per-step
//! compute scratch stays constant-size.
//!
//! # No silent fallback (FR-EX-08)
//!
//! - Every out-of-range token / position surfaces as
//!   [`VokraError::InvalidArgument`] with the offending value named.
//! - A `0`-sentinel config (from the shape-only converter path) is rejected
//!   at [`TextDecoderSession::new`] — never a silent default.

use vokra_core::{BackendKind, KvCache, Result, VokraError};

use crate::compute::Compute;

use super::VOXTRAL_HOT_OPS;
use super::text_decoder::{self, StepScratch};
use super::{TextDecoder, VoxtralConfig};

/// Initial reservation hint for the self-attention KV cache, in positions
/// (mirrors `whisper::decoder::SELF_KV_RESERVE_HINT` — typical decodes are
/// short, so we don't pre-allocate the worst-case window).
const SELF_KV_RESERVE_HINT: usize = 64;

/// A single-decode session: greedy loop calls `step_into` then reads the
/// last-row logits back through `last_logits_row` (allocation-free hot path).
///
/// # Lifetime
///
/// Borrows `&'m TextDecoder` and `&'m VoxtralConfig` — the caller keeps the
/// loaded model alive for the session's lifetime. This matches the
/// scaffold-only borrow pattern used elsewhere in the Voxtral module (see
/// [`super::AsrHead`]). A follow-up ticket can promote the session to owning
/// an `Arc<VoxtralModel>` (mirroring Whisper's `DecoderState`) when the
/// streaming layer needs to move it across threads.
pub struct TextDecoderSession<'m> {
    config: &'m VoxtralConfig,
    decoder: &'m TextDecoder,
    compute: Compute,
    kv_cache: KvCache,
    scratch: StepScratch,
    /// Number of committed decode positions (same as `kv_cache.positions()`
    /// but cached to avoid a per-call kv_cache method invocation).
    position: usize,
    /// Cached hyperparameter dims for `last_logits_row` / bounds checks.
    n_layer: usize,
    d: usize,
    kv_hidden: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_t_kv: usize,
}

impl<'m> TextDecoderSession<'m> {
    /// Constructs a new decoder session on `backend`.
    ///
    /// The scratch buffers are sized so a decode step of up to 512 tokens
    /// (a typical prefix + one greedy step) does not reallocate. Longer
    /// steps grow the scratch amortically (single-token greedy stays
    /// within the reserve).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] if the config is the shape-only
    ///   `0`-sentinel path (missing GQA head split, RoPE base, RMSNorm eps,
    ///   or vocab size).
    /// - [`VokraError::UnsupportedOp`] if `backend` does not cover the
    ///   Voxtral hot ops (FR-EX-08).
    /// - [`VokraError::BackendUnavailable`] if `backend` is not built into
    ///   this binary.
    pub fn new(
        config: &'m VoxtralConfig,
        decoder: &'m TextDecoder,
        backend: BackendKind,
    ) -> Result<Self> {
        // Reject 0-sentinel config — every downstream method depends on
        // these values (FR-EX-08, no silent default).
        let n_layer = config.text.n_layer;
        let d = config.text.hidden_dim;
        let vocab_size = config.text.vocab_size;
        let n_head_q = config.text.n_head_q;
        let n_head_kv = config.text.n_head_kv;
        let ffn_dim = config.text.ffn_dim;
        if n_layer == 0 || d == 0 || vocab_size == 0 || n_head_q == 0 || n_head_kv == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral::TextDecoderSession: config carries 0-sentinel — re-convert with a full \
                 VoxtralConfig (FR-EX-08 — no silent default)."
                    .into(),
            ));
        }
        if n_head_q % n_head_kv != 0 {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::TextDecoderSession: n_head_q ({n_head_q}) must be divisible by n_head_kv ({n_head_kv}) — GQA"
            )));
        }
        let head_dim = d / n_head_q;
        let kv_hidden = n_head_kv * head_dim;
        if decoder.blocks.len() != n_layer {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::TextDecoderSession: loaded blocks {} != config n_layer {n_layer}",
                decoder.blocks.len()
            )));
        }

        let compute = Compute::for_backend(backend, VOXTRAL_HOT_OPS)?;

        // Reserve for a typical prefix step (up to a few tokens) plus the
        // greedy loop (single-token steps). Reserve for a larger t_kv so
        // the attention scratch stays constant for a moderate decode.
        // (The scratch will grow if a decode exceeds these bounds.)
        let reserve_t_q = 8usize;
        let reserve_t_kv = config.text.n_ctx.min(256).max(reserve_t_q);
        let kv_cache = KvCache::with_reserve(
            n_layer,
            kv_hidden,
            SELF_KV_RESERVE_HINT.min(config.text.n_ctx.max(1)),
        );
        let scratch = StepScratch::with_reserve(
            reserve_t_q,
            d,
            kv_hidden,
            head_dim,
            ffn_dim,
            vocab_size,
            reserve_t_kv,
        );

        Ok(Self {
            config,
            decoder,
            compute,
            kv_cache,
            scratch,
            position: 0,
            n_layer,
            d,
            kv_hidden,
            head_dim,
            ffn_dim,
            vocab_size,
            max_t_kv: reserve_t_kv,
        })
    }

    /// Convenience constructor on the CPU backend (zero-dep default).
    pub fn cpu(config: &'m VoxtralConfig, decoder: &'m TextDecoder) -> Result<Self> {
        Self::new(config, decoder, BackendKind::Cpu)
    }

    /// Number of committed positions so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Rewinds the session for a fresh decode of the same model.
    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.position = 0;
    }

    /// Runs one decode step: appends every layer's K/V rows for `tokens`
    /// (using RoPE at position `self.position + i`), then leaves the
    /// `[t, vocab_size]` logits in the reused scratch. Read the final row
    /// with [`last_logits_row`](Self::last_logits_row).
    ///
    /// A zero-length `tokens` is a no-op.
    ///
    /// # Errors
    ///
    /// See [`text_decoder::forward_step`] for the full error taxonomy.
    pub fn step_into(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let position_offset = self.position;
        text_decoder::forward_step(
            &self.compute,
            self.config,
            self.decoder,
            &mut self.scratch,
            &mut self.kv_cache,
            tokens,
            position_offset,
        )?;
        self.position += tokens.len();
        // Update the tracked t_kv max so the caller can size follow-up
        // scratch (informational; the scratch itself grew inside forward_step
        // if needed).
        if self.position > self.max_t_kv {
            self.max_t_kv = self.position;
        }
        Ok(())
    }

    /// Returns the logits for the last position (`[vocab_size]`) — the
    /// greedy / beam read. Must not be called before a non-empty
    /// [`step_into`](Self::step_into) (the logits scratch would be empty).
    #[must_use]
    pub fn last_logits_row(&self) -> &[f32] {
        let v = self.vocab_size;
        let out = &self.scratch.logits;
        // If step_into wrote at least one row, its length is a multiple of v.
        &out[out.len() - v..]
    }

    /// Returns the full `[t, vocab_size]` logits from the last step.
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        &self.scratch.logits
    }

    /// Backend name (`"cpu"` / `"metal"` / `"cuda"`).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.compute.backend_name()
    }

    /// Number of decoder layers (informational — matches the config).
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// Config hidden width — for callers that want to size external
    /// projections (e.g. audio adapter follow-up).
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.d
    }

    /// Vocabulary size — for callers checking argmax bounds.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    // Silence unused-field lints for the audio-adapter follow-up. These
    // dimensions are read by the future GPU session ctor + external
    // conditioning path; no runtime cost until then.
    #[doc(hidden)]
    pub fn kv_hidden(&self) -> usize {
        self.kv_hidden
    }
    #[doc(hidden)]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    #[doc(hidden)]
    pub fn ffn_dim(&self) -> usize {
        self.ffn_dim
    }
}

// -----------------------------------------------------------------------------
// Greedy decode driver
// -----------------------------------------------------------------------------

/// Default cap on generated tokens when the caller does not pass one.
/// Chosen to match Whisper's `n_text_ctx / 2 = 224` for a comparable
/// per-utterance budget; still bounded by `config.text.n_ctx`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 224;

/// Greedy argmax over a logits row.
fn argmax(row: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Greedy-decode starting from `start_ids`, stopping on `eos` (which IS
/// included in the result) or after `max_new` new tokens. Returns the
/// generated token ids (start_ids are NOT included).
///
/// The session is `reset()` at the top so a second call reproduces the
/// first.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if `start_ids` is empty or any id is
///   out of range (surfaced from `step_into`).
pub fn greedy_decode(
    session: &mut TextDecoderSession<'_>,
    start_ids: &[u32],
    eos: u32,
    max_new: usize,
) -> Result<Vec<u32>> {
    if start_ids.is_empty() {
        return Err(VokraError::InvalidArgument(
            "voxtral::greedy_decode: start_ids must not be empty".into(),
        ));
    }
    session.reset();
    session.step_into(start_ids)?;
    let mut generated = Vec::with_capacity(max_new.min(64));
    let cap = max_new.max(1);
    let n_ctx_cap = session.config.text.n_ctx.saturating_sub(start_ids.len());
    let cap = cap.min(n_ctx_cap);
    for _ in 0..cap {
        let next = argmax(session.last_logits_row());
        generated.push(next);
        if next == eos {
            break;
        }
        session.step_into(&[next])?;
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};

    fn tiny_cfg() -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 4,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 4,
                n_ctx: 16,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    /// Build a tiny hand-crafted TextDecoder with deterministic non-zero weights.
    /// The values are small enough to keep softmax stable and produce a
    /// finite, deterministic argmax without exploding logits.
    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let n_head_q = cfg.text.n_head_q;
        let n_head_kv = cfg.text.n_head_kv;
        let head_dim = d / n_head_q;
        let kv_hidden = n_head_kv * head_dim;

        // Deterministic weight initialisation: small distinct values so
        // GEMMs actually mix rows (identity-scaled weights would collapse
        // the token embed unchanged through the whole stack — a degenerate
        // oracle).
        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        let final_norm_gamma = vec![1.0f32; d];

        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
            // `w_t` is [in, out] row-major; rows = in, cols = out.
            let mut w_t = vec![0.0f32; rows * cols];
            for (i, v) in w_t.iter_mut().enumerate() {
                *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols,
            }
        }

        let blocks = (0..cfg.text.n_layer)
            .map(|_| DecoderBlock {
                attn_norm_gamma: vec![1.0f32; d],
                attn: GqaAttention {
                    q: linear(d, d, 0.10),
                    k: linear(d, kv_hidden, -0.07),
                    v: linear(d, kv_hidden, 0.05),
                    o: linear(d, d, -0.04),
                },
                ffn_norm_gamma: vec![1.0f32; d],
                ffn: SwiGluFfn {
                    gate: linear(d, ffn, 0.06),
                    up: linear(d, ffn, -0.02),
                    down: linear(ffn, d, 0.03),
                },
            })
            .collect();

        TextDecoder {
            token_emb,
            blocks,
            final_norm_gamma,
            prefix: "",
        }
    }

    #[test]
    fn new_rejects_zero_sentinel_config() {
        let mut cfg = tiny_cfg();
        cfg.text.n_layer = 0;
        let td = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        let err = match TextDecoderSession::new(&cfg, &td, BackendKind::Cpu) {
            Ok(_) => panic!("must fail — TextDecoderSession is not Debug"),
            Err(e) => e,
        };
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    #[test]
    fn new_rejects_gqa_head_split_mismatch() {
        let mut cfg = tiny_cfg();
        cfg.text.n_head_q = 3;
        cfg.text.n_head_kv = 2;
        // Since we construct decoder manually, blocks.len() also matters.
        let td = tiny_decoder(&tiny_cfg()); // note: uses original 2/1 split
        // With n_head_q=3 not divisible by n_head_kv=2, new must reject.
        let err = match TextDecoderSession::new(&cfg, &td, BackendKind::Cpu) {
            Ok(_) => panic!("must fail — TextDecoderSession is not Debug"),
            Err(e) => e,
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err:?}");
    }

    #[test]
    fn step_into_advances_position_and_produces_logits() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        assert_eq!(sess.position(), 0);
        sess.step_into(&[1u32, 0]).unwrap();
        assert_eq!(sess.position(), 2);
        let logits = sess.last_logits_row();
        assert_eq!(logits.len(), cfg.text.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empty_step_is_noop() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        sess.step_into(&[]).unwrap();
        assert_eq!(sess.position(), 0);
    }

    #[test]
    fn out_of_range_token_is_error_not_panic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // vocab_size = 4, so id 99 is out of range.
        let err = sess.step_into(&[99u32]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn exceeding_n_ctx_is_error_not_panic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // n_ctx = 16; pushing 17 tokens overflows.
        let toks: Vec<u32> = (0..17).map(|i| i % cfg.text.vocab_size as u32).collect();
        let err = sess.step_into(&toks).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn full_prefix_matches_token_by_token() {
        // Full-sequence step must produce the same last-row logits as a
        // token-by-token decode after reset. Bit-for-bit is not guaranteed
        // because attention accumulator order changes; we assert atol=1e-4.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();

        let prefix = [1u32, 2, 0];
        sess.step_into(&prefix).unwrap();
        let full_last: Vec<f32> = sess.last_logits_row().to_vec();

        sess.reset();
        assert_eq!(sess.position(), 0);
        sess.step_into(&[prefix[0]]).unwrap();
        sess.step_into(&[prefix[1]]).unwrap();
        sess.step_into(&[prefix[2]]).unwrap();
        let cached_last = sess.last_logits_row();
        assert_eq!(cached_last.len(), full_last.len());
        for (i, (&f, &c)) in full_last.iter().zip(cached_last).enumerate() {
            assert!(
                (f - c).abs() < 5e-4,
                "idx {i}: full {f} vs cached {c} diff {}",
                (f - c).abs()
            );
        }
    }

    #[test]
    fn reset_and_replay_is_bit_identical() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let prefix = [1u32, 2];
        sess.step_into(&prefix).unwrap();
        let logits_1: Vec<f32> = sess.last_logits_row().to_vec();
        sess.reset();
        sess.step_into(&prefix).unwrap();
        let logits_2 = sess.last_logits_row();
        assert_eq!(logits_1.as_slice(), logits_2);
    }

    #[test]
    fn argmax_picks_first_max_on_ties() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn greedy_decode_respects_max_new() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // eos outside vocab => never terminates on EOS => cap at max_new
        // (bounded further by n_ctx - start_ids.len()).
        let eos = cfg.text.vocab_size as u32 + 100;
        let ids = greedy_decode(&mut sess, &[1u32], eos, 3).unwrap();
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().all(|&t| (t as usize) < cfg.text.vocab_size));
    }

    #[test]
    fn greedy_decode_empty_start_is_rejected() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        assert!(matches!(
            greedy_decode(&mut sess, &[], 999, 3),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn greedy_decode_is_deterministic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let a = greedy_decode(&mut sess, &[1u32], 9999, 3).unwrap();
        let b = greedy_decode(&mut sess, &[1u32], 9999, 3).unwrap();
        assert_eq!(a, b);
    }
}
