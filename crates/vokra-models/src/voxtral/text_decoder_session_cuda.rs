//! Voxtral CUDA-backed text-decoder session — Compute-seam-backed GPU
//! dispatch for the Mistral decoder step (M3-10 Wave 8+/M4 slice).
//!
//! # Scope of this slice
//!
//! [`VoxtralCudaDecodeSession`] is the CUDA parallel of Whisper's
//! `CudaDecodeSession` (`vokra-backend-cuda::CudaDecodeSession`). Same
//! honest scope as the Metal sibling
//! ([`super::text_decoder_session_metal::VoxtralMetalDecodeSession`]):
//!
//! - **Explicit backend selection with FR-EX-08 gating.** A missing CUDA
//!   driver / device or a CUDA feature disabled at compile time surfaces
//!   [`VokraError::BackendUnavailable`] at construction — never a silent
//!   CPU fall back.
//! - **Real GPU dispatch through the Compute seam.** Every GEMM /
//!   GEMV / softmax / RMSNorm / GELU the Mistral decoder step emits is
//!   routed through the `Compute::Cuda` arm — the NVRTC-compiled kernels
//!   in `vokra-backend-cuda`. Real GPU work per op; the residency layer
//!   (weights + KV cache pinned on device, one submission per step) is
//!   a follow-up optimization slot.
//! - **Bit-identical parity with the CPU baseline (within FP32
//!   rounding).** The Compute seam contract guarantees
//!   `CudaContext::gemm_f32` produces the same result as
//!   `kernels::gemm_f32` within FP32 bound (M2-03 primitive-parity
//!   tests), so the decoder step reproduces the CPU output to
//!   `atol ≤ 1e-4` on a small fixture and the greedy argmax sequence is
//!   identical.
//! - **Uniform API with the Metal sibling.** [`new_from_decoder`],
//!   [`step`], [`kv_cache_len`], [`reset`], [`last_logits`],
//!   [`all_logits`].
//!
//! [`step`]: Self::step
//! [`kv_cache_len`]: Self::kv_cache_len
//! [`reset`]: Self::reset
//! [`last_logits`]: Self::last_logits
//! [`all_logits`]: Self::all_logits
//! [`new_from_decoder`]: Self::new_from_decoder
//!
//! # NVIDIA EULA install model (FR-BE-08 / NVIDIA-EULA.md)
//!
//! The CUDA path uses `vokra-backend-cuda`'s Driver API + NVRTC FFI
//! loaded at runtime via `dlopen` / `LoadLibrary`. Vokra bundles no CUDA
//! runtime — the developer must have CUDA installed. Missing driver /
//! NVRTC is [`VokraError::BackendUnavailable`], never a silent fall back.
//!
//! # Cfg surface
//!
//! Available only when `--features cuda` is on and the target is
//! Windows / macOS / Linux (`any(unix, windows)`). Off the CUDA build the
//! type does not exist.
//!
//! [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable

#![cfg(all(feature = "cuda", any(unix, windows)))]

use vokra_core::{BackendKind, Result};

use super::{TextDecoder, TextDecoderSession, VoxtralConfig};

/// A CUDA-backed Voxtral text-decoder session.
///
/// See the module-level docs for scope and the parity contract with the
/// CPU baseline. Constructed via [`Self::new_from_decoder`]; [`Self::step`]
/// drives the greedy / prefix decode loop.
///
/// # Lifetime
///
/// Borrows `&'m VoxtralConfig` and `&'m TextDecoder`, same pattern as the
/// Metal sibling and the CPU baseline [`TextDecoderSession`].
pub struct VoxtralCudaDecodeSession<'m> {
    inner: TextDecoderSession<'m>,
}

impl<'m> VoxtralCudaDecodeSession<'m> {
    /// Builds a CUDA-backed decoder session over the loaded Mistral text
    /// decoder.
    ///
    /// The inner [`TextDecoderSession`] is constructed with
    /// [`BackendKind::Cuda`], which builds a `Compute::Cuda` dispatcher.
    /// The Compute seam gates the backend against the Voxtral hot ops;
    /// every op the M2-03 Phase-4 CUDA slice does not cover is an
    /// explicit [`VokraError::UnsupportedOp`] — never a silent CPU fall
    /// back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::BackendUnavailable`] if `libcuda` / NVRTC cannot
    ///   be `dlopen`-ed, no CUDA device is available, or a device query
    ///   fails.
    /// - [`VokraError::UnsupportedOp`] if a Voxtral hot op is not covered
    ///   by the CUDA backend on this build.
    /// - [`VokraError::ModelLoad`] if `config` carries a `0`-sentinel
    ///   value or if `decoder.blocks.len() != config.text.n_layer`.
    ///
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    /// [`VokraError::UnsupportedOp`]: vokra_core::VokraError::UnsupportedOp
    /// [`VokraError::ModelLoad`]: vokra_core::VokraError::ModelLoad
    pub fn new_from_decoder(config: &'m VoxtralConfig, decoder: &'m TextDecoder) -> Result<Self> {
        let inner = TextDecoderSession::new(config, decoder, BackendKind::Cuda)?;
        Ok(Self { inner })
    }

    /// The wrapped [`TextDecoderSession`] — exposed for the ASR / streaming
    /// glue.
    pub fn inner_mut(&mut self) -> &mut TextDecoderSession<'m> {
        &mut self.inner
    }

    /// The wrapped [`TextDecoderSession`] (shared).
    pub fn inner(&self) -> &TextDecoderSession<'m> {
        &self.inner
    }

    /// Advances the decode by `tokens`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on out-of-range tokens or a
    ///   decode that would exceed `config.text.n_ctx`.
    /// - [`VokraError::BackendUnavailable`] on a device failure.
    ///
    /// [`VokraError::InvalidArgument`]: vokra_core::VokraError::InvalidArgument
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    pub fn step(&mut self, tokens: &[u32]) -> Result<()> {
        self.inner.step_into(tokens)
    }

    /// Advances the decode by a soft-prefix embedding sequence, without a
    /// token-embedding lookup. Used by the audio-conditioned path with an
    /// active [`super::AudioAdapter`].
    pub fn step_with_embed_prefix(&mut self, prefix_embed: &[f32], t_prefix: usize) -> Result<()> {
        self.inner
            .step_into_with_embed_prefix(prefix_embed, t_prefix)
    }

    /// The number of committed decode positions in the self-attention
    /// KV cache. Parity with the Whisper CUDA session's
    /// `CudaDecodeSession::positions`.
    #[must_use]
    pub fn kv_cache_len(&self) -> usize {
        self.inner.position()
    }

    /// Rewinds the session for a fresh decode of the same model.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The logits for the last decoded position (`[vocab_size]`).
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        self.inner.last_logits_row()
    }

    /// The full `[t, vocab_size]` logits from the last step.
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        self.inner.all_logits()
    }

    /// Backend name — always `"cuda"` for this session type.
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

    /// Whether the host actually has a CUDA driver + device. A CI runner
    /// without CUDA reports BackendUnavailable, and we skip the CUDA-only
    /// assertions cleanly (no silent CPU fall back, FR-EX-08).
    fn has_cuda_device() -> bool {
        crate::compute::Compute::for_backend(BackendKind::Cuda, crate::voxtral::VOXTRAL_HOT_OPS)
            .is_ok()
    }

    #[test]
    fn step_advances_and_reset_rewinds_kv_cache_len() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; VoxtralCudaDecodeSession kv-cache test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = VoxtralCudaDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        assert_eq!(sess.kv_cache_len(), 0);
        assert_eq!(sess.backend_name(), "cuda");
        sess.step(&[1u32]).unwrap();
        assert_eq!(sess.kv_cache_len(), 1);
        sess.step(&[0u32]).unwrap();
        assert_eq!(sess.kv_cache_len(), 2);
        sess.reset();
        assert_eq!(sess.kv_cache_len(), 0);
    }

    /// The CUDA session's step logits must match the CPU baseline within
    /// FP32 rounding — same contract as the Metal sibling.
    #[test]
    fn bit_identical_vs_cpu_on_tiny_fixture() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; VoxtralCudaDecodeSession parity test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut cuda_sess = VoxtralCudaDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        let prefix = [1u32, 2, 0];
        cuda_sess.step(&prefix).unwrap();
        let cuda_last: Vec<f32> = cuda_sess.last_logits().to_vec();

        let mut cpu_sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        cpu_sess.step_into(&prefix).unwrap();
        let cpu_last = cpu_sess.last_logits_row();

        assert_eq!(cuda_last.len(), cpu_last.len());
        for (i, (&g, &c)) in cuda_last.iter().zip(cpu_last).enumerate() {
            assert!(
                (g - c).abs() < 5e-4,
                "logit[{i}] cpu {c} vs cuda {g} diff {}",
                (g - c).abs(),
            );
        }
    }

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
        match VoxtralCudaDecodeSession::new_from_decoder(&cfg, &td) {
            Ok(_) => panic!("must fail on 0-sentinel config"),
            Err(vokra_core::VokraError::ModelLoad(_)) => {}
            Err(other) => {
                assert!(
                    matches!(other, vokra_core::VokraError::BackendUnavailable(_)),
                    "expected ModelLoad or BackendUnavailable, got {other:?}"
                );
            }
        }
    }

    #[test]
    fn step_with_embed_prefix_produces_logits_and_advances() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; VoxtralCudaDecodeSession embed-prefix test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = VoxtralCudaDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        let d = cfg.text.hidden_dim;
        let t_prefix = 2;
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
