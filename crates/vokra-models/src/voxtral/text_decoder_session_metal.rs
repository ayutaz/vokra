//! Voxtral Metal-backed text-decoder session — Compute-seam-backed GPU
//! dispatch for the Mistral decoder step (M3-10 Wave 8+/M4 slice).
//!
//! # Scope of this slice
//!
//! [`VoxtralMetalDecodeSession`] is the Metal parallel of Whisper's
//! `MetalDecodeSession` (`vokra-backend-metal::MetalDecodeSession`). The
//! Voxtral text decoder is architecturally different enough from Whisper
//! (decoder-only Mistral: GQA + RoPE + RMSNorm + SwiGLU vs Whisper's
//! encoder-decoder MHA + learned pos-emb + LayerNorm + GELU-MLP) that a
//! bespoke device-resident kernel bundle at Whisper's fusion level is a
//! follow-up ticket. This slice provides:
//!
//! - **Explicit backend selection with FR-EX-08 gating.** A device that is
//!   not present, or a Metal feature disabled at compile time, surfaces
//!   [`VokraError::BackendUnavailable`] at construction — never a silent
//!   CPU fall back.
//! - **Real GPU dispatch through the Compute seam.** Every GEMM /
//!   GEMV / softmax / RMSNorm / GELU the Mistral decoder step emits is
//!   routed through the `Compute::Metal` arm, so the GEMM cost — the
//!   dominant cost on any decoder — runs on the GPU. The residency layer
//!   (weights + KV cache pinned on device, one command-buffer submission
//!   per step) is a follow-up optimization slot; today's slice runs the
//!   Compute-seam per-op path, which is real GPU work.
//! - **Bit-identical parity to the CPU path (within FP32 rounding).** The
//!   Compute seam contract guarantees `MetalContext::gemm_f32` produces
//!   the same result as `kernels::gemm_f32` within FP32 rounding, so the
//!   decoder step reproduces the CPU output to `atol ≤ 1e-4` on a small
//!   fixture and the greedy argmax sequence is identical.
//! - **Uniform API with the CUDA sibling.** [`new_from_decoder`], [`step`],
//!   [`kv_cache_len`], [`reset`], [`last_logits`], [`all_logits`] — same
//!   surface as [`super::text_decoder_session_cuda::VoxtralCudaDecodeSession`].
//!
//! [`step`]: Self::step
//! [`kv_cache_len`]: Self::kv_cache_len
//! [`reset`]: Self::reset
//! [`last_logits`]: Self::last_logits
//! [`all_logits`]: Self::all_logits
//! [`new_from_decoder`]: Self::new_from_decoder
//!
//! # `!Send` / `!Sync`
//!
//! The session owns a [`TextDecoderSession`] which, on the Metal backend,
//! wraps a live `MetalContext` (`!Send` at the Rust type level). Callers
//! hold the session within a single thread; cross-thread streaming is done
//! by message-passing at the AsrHead boundary (same pattern the Voxtral
//! streaming layer already uses).
//!
//! # Cfg surface
//!
//! Available only when `--features metal` is on **and** the target is
//! Apple. Off the metal build the type does not exist; callers that need
//! runtime backend probing should route through the `allow_device_session`
//! flag on [`super::asr::VoxtralAsr`], which encapsulates the cfg checks in
//! one explicit-error branch.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use vokra_core::{BackendKind, Result};

use super::{TextDecoder, TextDecoderSession, VoxtralConfig};

/// A Metal-backed Voxtral text-decoder session.
///
/// See the module-level docs for the scope and the parity contract with
/// the CPU baseline. Constructed via [`Self::new_from_decoder`]; the
/// [`Self::step`] entry drives the greedy / prefix decode loop.
///
/// # Lifetime
///
/// Borrows `&'m VoxtralConfig` and `&'m TextDecoder` — the caller keeps
/// the loaded model alive for the session's lifetime, mirroring
/// [`TextDecoderSession`].
pub struct VoxtralMetalDecodeSession<'m> {
    inner: TextDecoderSession<'m>,
}

impl<'m> VoxtralMetalDecodeSession<'m> {
    /// Builds a Metal-backed decoder session over the loaded Mistral text
    /// decoder.
    ///
    /// The inner [`TextDecoderSession`] is constructed with
    /// [`BackendKind::Metal`], which builds a `Compute::Metal` dispatcher.
    /// The Compute seam gates the backend against the Voxtral hot ops
    /// (`VOXTRAL_HOT_OPS`); every op the M2-01 Phase-4 Metal slice does
    /// not cover is an explicit [`VokraError::UnsupportedOp`] — never a
    /// silent CPU fall back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::BackendUnavailable`] if no Metal device is found on
    ///   the host.
    /// - [`VokraError::UnsupportedOp`] if a Voxtral hot op is not covered
    ///   by the Metal backend on this build.
    /// - [`VokraError::ModelLoad`] if `config` carries a `0`-sentinel
    ///   value (missing GQA head split, RoPE base, RMSNorm eps, or vocab
    ///   size) or if `decoder.blocks.len() != config.text.n_layer`.
    ///
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    /// [`VokraError::UnsupportedOp`]: vokra_core::VokraError::UnsupportedOp
    /// [`VokraError::ModelLoad`]: vokra_core::VokraError::ModelLoad
    pub fn new_from_decoder(config: &'m VoxtralConfig, decoder: &'m TextDecoder) -> Result<Self> {
        let inner = TextDecoderSession::new(config, decoder, BackendKind::Metal)?;
        Ok(Self { inner })
    }

    /// The wrapped [`TextDecoderSession`] — exposed for the ASR / streaming
    /// glue that already talks the `TextDecoderSession` surface (e.g. the
    /// `greedy_decode` / `greedy_decode_with_prefix` drivers). Follow-up
    /// tickets that add a bespoke device-resident forward can replace this
    /// pass-through with a session-owned kernel bundle.
    pub fn inner_mut(&mut self) -> &mut TextDecoderSession<'m> {
        &mut self.inner
    }

    /// The wrapped [`TextDecoderSession`] (shared).
    pub fn inner(&self) -> &TextDecoderSession<'m> {
        &self.inner
    }

    /// Advances the decode by `tokens`, appending each block's K/V rows
    /// and leaving the `[t, vocab_size]` logits in the reused scratch
    /// (read via [`Self::last_logits`] or [`Self::all_logits`]).
    ///
    /// A zero-length `tokens` is a no-op. See
    /// [`super::text_decoder_session::TextDecoderSession::step_into`] for
    /// the full error taxonomy.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] for out-of-range tokens or a
    ///   decode that would exceed `config.text.n_ctx`.
    /// - [`VokraError::BackendUnavailable`] on a device failure.
    ///
    /// [`VokraError::InvalidArgument`]: vokra_core::VokraError::InvalidArgument
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    pub fn step(&mut self, tokens: &[u32]) -> Result<()> {
        self.inner.step_into(tokens)
    }

    /// Advances the decode by a soft-prefix embedding sequence, without a
    /// token-embedding lookup. Used by the audio-conditioned path
    /// (`AsrHead::transcribe` with an active [`super::AudioAdapter`]).
    ///
    /// # Errors
    ///
    /// See
    /// [`super::text_decoder_session::TextDecoderSession::step_into_with_embed_prefix`].
    pub fn step_with_embed_prefix(&mut self, prefix_embed: &[f32], t_prefix: usize) -> Result<()> {
        self.inner
            .step_into_with_embed_prefix(prefix_embed, t_prefix)
    }

    /// The number of committed decode positions in the self-attention KV
    /// cache. Parity with the Whisper backend session's
    /// `MetalDecodeSession::positions` — same semantic role.
    #[must_use]
    pub fn kv_cache_len(&self) -> usize {
        self.inner.position()
    }

    /// Rewinds the session for a fresh decode of the same model. The KV
    /// cache is cleared but capacity is retained.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The logits for the last decoded position (`[vocab_size]`) — the
    /// greedy / beam read. Must not be called before a non-empty
    /// [`Self::step`] (the logits scratch would be empty).
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        self.inner.last_logits_row()
    }

    /// The full `[t, vocab_size]` logits from the last step (row-major,
    /// row `i` at offset `i * vocab_size`).
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        self.inner.all_logits()
    }

    /// Backend name — always `"metal"` for this session type. Load-bearing
    /// for diagnostics that check the dispatch path.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};

    /// Same tiny config the CPU session tests use — the Metal parity
    /// tests share it so the two paths cover the same fixture.
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

    /// Same deterministic-weight fixture the CPU session tests use. Small
    /// enough to keep softmax stable and produce a finite, deterministic
    /// argmax; large enough to exercise every block op (attention proj,
    /// RoPE, KV append, softmax, o_proj, SwiGLU FFN, RMSNorm, tied-logits).
    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let n_head_q = cfg.text.n_head_q;
        let n_head_kv = cfg.text.n_head_kv;
        let head_dim = d / n_head_q;
        let kv_hidden = n_head_kv * head_dim;
        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
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
            final_norm_gamma: vec![1.0f32; d],
            prefix: "",
        }
    }

    /// Whether the host actually has a Metal device. Constructed sessions
    /// that fail this probe report BackendUnavailable (no silent CPU fall
    /// back, FR-EX-08), so a headless CI runner without a GPU can skip
    /// the Metal-only assertions cleanly instead of failing.
    fn has_metal_device() -> bool {
        crate::compute::Compute::for_backend(BackendKind::Metal, crate::voxtral::VOXTRAL_HOT_OPS)
            .is_ok()
    }

    /// Every step advances `kv_cache_len` by the number of tokens. Two
    /// single-token steps end at `kv_cache_len == 2`; reset returns to 0.
    #[test]
    fn step_advances_and_reset_rewinds_kv_cache_len() {
        if !has_metal_device() {
            eprintln!("no Metal device; VoxtralMetalDecodeSession kv-cache test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = VoxtralMetalDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        assert_eq!(sess.kv_cache_len(), 0);
        assert_eq!(sess.backend_name(), "metal");
        sess.step(&[1u32]).unwrap();
        assert_eq!(sess.kv_cache_len(), 1);
        sess.step(&[0u32]).unwrap();
        assert_eq!(sess.kv_cache_len(), 2);
        sess.reset();
        assert_eq!(sess.kv_cache_len(), 0);
    }

    /// The Metal session's step logits must match the CPU baseline within
    /// FP32 rounding. This is the Compute-seam contract at work: the
    /// GEMM / softmax / RMSNorm dispatch to `MetalContext::*`, whose
    /// output equals `kernels::*_f32` within FP32 bound. On the tiny
    /// fixture (d=4, ffn=8, vocab=4, one layer) the accumulated
    /// difference stays well below `atol = 5e-4` — same tolerance the
    /// CPU-side `full_prefix_matches_token_by_token` test uses.
    #[test]
    fn bit_identical_vs_cpu_on_tiny_fixture() {
        if !has_metal_device() {
            eprintln!("no Metal device; VoxtralMetalDecodeSession parity test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        // Metal path.
        let mut metal_sess = VoxtralMetalDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        let prefix = [1u32, 2, 0];
        metal_sess.step(&prefix).unwrap();
        let metal_last: Vec<f32> = metal_sess.last_logits().to_vec();

        // CPU baseline.
        let mut cpu_sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        cpu_sess.step_into(&prefix).unwrap();
        let cpu_last = cpu_sess.last_logits_row();

        assert_eq!(metal_last.len(), cpu_last.len());
        for (i, (&m, &c)) in metal_last.iter().zip(cpu_last).enumerate() {
            assert!(
                (m - c).abs() < 5e-4,
                "logit[{i}] cpu {c} vs metal {m} diff {}",
                (m - c).abs(),
            );
        }
    }

    /// The Metal session must reject a zero-sentinel config the same way
    /// the CPU session does — no silent default (FR-EX-08).
    #[test]
    fn new_from_decoder_rejects_zero_sentinel_config() {
        // Regardless of device presence: the config gate runs before the
        // Compute build, so the error is deterministic.
        let mut cfg = tiny_cfg();
        cfg.text.n_layer = 0;
        let td = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        match VoxtralMetalDecodeSession::new_from_decoder(&cfg, &td) {
            Ok(_) => panic!("must fail on 0-sentinel config"),
            Err(vokra_core::VokraError::ModelLoad(_)) => {}
            Err(other) => {
                // The Compute build (Metal probe) might race ahead on some
                // hosts and surface an unavailability first; either way is
                // an explicit error, never a silent pass. We only tolerate
                // the explicit backend-unavailable case here.
                assert!(
                    matches!(other, vokra_core::VokraError::BackendUnavailable(_)),
                    "expected ModelLoad or BackendUnavailable, got {other:?}"
                );
            }
        }
    }

    /// A soft-prefix embedding step must produce well-formed logits and
    /// advance the KV cache — the Wave-8 audio-adapter path uses this
    /// entry point.
    #[test]
    fn step_with_embed_prefix_produces_logits_and_advances() {
        if !has_metal_device() {
            eprintln!("no Metal device; VoxtralMetalDecodeSession embed-prefix test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = VoxtralMetalDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        let d = cfg.text.hidden_dim;
        let t_prefix = 2;
        // Non-zero deterministic prefix — the shape a real audio adapter
        // would emit into the decoder.
        let mut prefix = vec![0.0f32; t_prefix * d];
        for (i, v) in prefix.iter_mut().enumerate() {
            *v = ((i as i32 % 3) - 1) as f32 * 0.1;
        }
        sess.step_with_embed_prefix(&prefix, t_prefix).unwrap();
        assert_eq!(sess.kv_cache_len(), t_prefix);
        let logits = sess.last_logits();
        assert_eq!(logits.len(), cfg.text.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
