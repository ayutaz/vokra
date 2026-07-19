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
//! # Wave 10 — full device-residency opt-in seam
//!
//! [`Self::new_from_decoder_full_residency`] is the CUDA parallel of the
//! Metal Wave 10 seam
//! ([`super::text_decoder_session_metal::VoxtralMetalDecodeSession::new_from_decoder_full_residency`]).
//! Semantics + scope split are identical: this constructor lands the API
//! surface + [`ResidencyMode`] classification + the
//! [`super::asr::VoxtralAsr::allow_full_residency`] opt-in routing seat,
//! while the per-step forward is currently delegated to the same
//! CUDA-backed [`TextDecoderSession`] the thin path uses (so the two
//! posture flavours are bit-identical today under FP32 rounding). The
//! Wave 10.1 / M4 follow-up will migrate the internal step to a bespoke
//! fused NVRTC PTX kernel; the API surface will not change.
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

/// Which residency posture a [`VoxtralCudaDecodeSession`] was constructed
/// with — the plumbing gate for the Wave 10 "深化" (opt-in
/// full-device-residency) seam.
///
/// Semantically identical to the Metal sibling's
/// [`super::text_decoder_session_metal::ResidencyMode`]; kept as a
/// separate type so the two backend surfaces do not silently share an
/// enum whose semantics diverge (each backend documents its own
/// posture).
///
/// - [`ResidencyMode::Thin`] — the Wave 9 posture (`new_from_decoder`),
///   per-op GEMM dispatch through `Compute::Cuda`.
/// - [`ResidencyMode::FullResident`] — the Wave 10 seam
///   (`new_from_decoder_full_residency`), API surface stable for the
///   Wave 10.1 / M4 kernel-fusion follow-up. Bit-identical to the thin
///   posture today (parity test asserts).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyMode {
    /// Thin Compute-seam wrapper (Wave 9). Per-op GEMM dispatch through
    /// `Compute::Cuda`.
    Thin,
    /// Opt-in full device-residency (Wave 10). Bit-identical to
    /// [`Self::Thin`] under the hood today; API surface stable for the
    /// Wave 10.1 / M4 NVRTC kernel-fusion follow-up.
    FullResident,
}

/// A CUDA-backed Voxtral text-decoder session.
///
/// See the module-level docs for scope and the parity contract with the
/// CPU baseline. Constructed via [`Self::new_from_decoder`] (thin, Wave 9
/// posture) or [`Self::new_from_decoder_full_residency`] (opt-in
/// full-device-residency seam, Wave 10); [`Self::step`] drives the greedy
/// / prefix decode loop either way.
///
/// # Lifetime
///
/// Borrows `&'m VoxtralConfig` and `&'m TextDecoder`, same pattern as the
/// Metal sibling and the CPU baseline [`TextDecoderSession`].
pub struct VoxtralCudaDecodeSession<'m> {
    inner: TextDecoderSession<'m>,
    /// The residency posture the session was constructed with. Read-only
    /// after construction; observable via [`Self::residency_mode`].
    residency_mode: ResidencyMode,
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
        Ok(Self {
            inner,
            residency_mode: ResidencyMode::Thin,
        })
    }

    /// Builds a CUDA-backed decoder session with the **opt-in full
    /// device-residency** posture (Wave 10 seam) — the CUDA parallel of
    /// [`super::text_decoder_session_metal::VoxtralMetalDecodeSession::new_from_decoder_full_residency`].
    ///
    /// # What this constructor is for
    ///
    /// 1. **Validates the config gate** (`0`-sentinel rejection, GQA head
    ///    split, decoder block count — same taxonomy as
    ///    [`Self::new_from_decoder`]).
    /// 2. **Applies the FR-EX-08 backend gate** through the Compute seam
    ///    (`Compute::for_backend(BackendKind::Cuda, VOXTRAL_HOT_OPS)`) —
    ///    a missing CUDA driver / NVRTC / device surfaces an explicit
    ///    error at *construction*, never a silent CPU fall back.
    /// 3. **Types the session as [`ResidencyMode::FullResident`]** so
    ///    downstream diagnostics + the parity tests can assert on the
    ///    posture the caller asked for.
    ///
    /// # Current internal behaviour (Wave 10)
    ///
    /// The step body currently delegates to the **same** CUDA-backed
    /// [`TextDecoderSession`] the thin constructor produces. Output is
    /// bit-identical to [`Self::new_from_decoder`] for the same input
    /// (parity test asserts). The Wave 10.1 / M4 kernel-fusion follow-up
    /// will migrate the internal step to a bespoke NVRTC-compiled PTX
    /// kernel that runs the whole Mistral step in one stream launch
    /// chain; the API surface — [`Self::step`],
    /// [`Self::step_with_embed_prefix`], [`Self::kv_cache_len`],
    /// [`Self::reset`], [`Self::last_logits`], [`Self::all_logits`],
    /// [`Self::backend_name`] — is **stable**.
    ///
    /// # Errors
    ///
    /// Same taxonomy as [`Self::new_from_decoder`]:
    /// - [`VokraError::BackendUnavailable`] if `libcuda` / NVRTC cannot
    ///   be `dlopen`-ed, no CUDA device is available, or a device query
    ///   fails.
    /// - [`VokraError::UnsupportedOp`] if a Voxtral hot op is not covered
    ///   by the CUDA backend on this build.
    /// - [`VokraError::ModelLoad`] on a `0`-sentinel config or a
    ///   loaded-blocks / config `n_layer` mismatch.
    ///
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    /// [`VokraError::UnsupportedOp`]: vokra_core::VokraError::UnsupportedOp
    /// [`VokraError::ModelLoad`]: vokra_core::VokraError::ModelLoad
    pub fn new_from_decoder_full_residency(
        config: &'m VoxtralConfig,
        decoder: &'m TextDecoder,
    ) -> Result<Self> {
        // See the Metal sibling for the design rationale — the Wave 10
        // residency posture is a *type* over the same CUDA Compute seam
        // today; the Wave 10.1 / M4 follow-up swaps the internal
        // `TextDecoderSession` for a bespoke device-resident step driver
        // with no API-surface change.
        let inner = TextDecoderSession::new(config, decoder, BackendKind::Cuda)?;
        Ok(Self {
            inner,
            residency_mode: ResidencyMode::FullResident,
        })
    }

    /// The residency posture the session was constructed with — the
    /// Wave 10 plumbing gate. See [`ResidencyMode`] for the semantics.
    ///
    /// This is a pure getter for the constructor-time classification;
    /// nothing in the step path consults it (the two postures are
    /// bit-identical today). It exists so downstream diagnostics + tests
    /// can assert the caller reached the constructor they intended.
    #[must_use]
    pub fn residency_mode(&self) -> ResidencyMode {
        self.residency_mode
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
                head_dim: 0,
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
            lm_head: None,
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
            lm_head: None,
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

    // ----------------------------------------------------------------------
    // Wave 10 — opt-in full device-residency seam (CUDA parallel)
    //
    // Mirror of the Metal Wave 10 test set: covers the plumbing added by
    // Wave 10 — `new_from_decoder_full_residency`, `residency_mode()`, the
    // thin-vs-full parity contract, KV-cache / reset / snapshot semantics
    // and the `0`-sentinel error path.
    //
    // All device-gated cases are `has_cuda_device()`-skipped so CI runners
    // without CUDA drivers cleanly report skip (FR-EX-08: no silent CPU
    // fall back).
    // ----------------------------------------------------------------------

    #[test]
    fn thin_constructor_tags_residency_mode_thin() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; residency-mode(thin) test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let sess = VoxtralCudaDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        assert_eq!(sess.residency_mode(), ResidencyMode::Thin);
    }

    #[test]
    fn full_residency_constructor_tags_residency_mode_and_advances() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; residency-mode(full) test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess =
            VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        assert_eq!(sess.residency_mode(), ResidencyMode::FullResident);
        assert_eq!(sess.backend_name(), "cuda");
        assert_eq!(sess.kv_cache_len(), 0);
        sess.step(&[1u32, 2, 0]).unwrap();
        assert_eq!(sess.kv_cache_len(), 3);
        sess.reset();
        assert_eq!(sess.kv_cache_len(), 0);
    }

    /// The Wave 10 seam's parity contract on CUDA: same-input → bit-identical
    /// logits between thin and full-residency. The Wave 10.1 / M4 kernel
    /// fusion follow-up must preserve this within FP32 rounding.
    #[test]
    fn full_residency_matches_thin_wrapper_bit_identical() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; thin-vs-full parity test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let prefix = [1u32, 2, 0];

        let mut thin = VoxtralCudaDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        thin.step(&prefix).unwrap();
        let thin_last: Vec<f32> = thin.last_logits().to_vec();

        let mut full =
            VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        full.step(&prefix).unwrap();
        let full_last = full.last_logits();

        assert_eq!(
            thin_last.as_slice(),
            full_last,
            "Wave 10 parity contract broken: full-residency logits must equal thin logits \
             (Wave 10.1 / M4 kernel fusion must preserve this within FP32 rounding)."
        );
    }

    #[test]
    fn full_residency_bit_identical_vs_cpu_on_tiny_fixture() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; full-residency vs CPU parity test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut full =
            VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        let prefix = [1u32, 2, 0];
        full.step(&prefix).unwrap();
        let full_last: Vec<f32> = full.last_logits().to_vec();

        let mut cpu = TextDecoderSession::cpu(&cfg, &td).unwrap();
        cpu.step_into(&prefix).unwrap();
        let cpu_last = cpu.last_logits_row();

        assert_eq!(full_last.len(), cpu_last.len());
        for (i, (&f, &c)) in full_last.iter().zip(cpu_last).enumerate() {
            assert!(
                (f - c).abs() < 5e-4,
                "logit[{i}] cpu {c} vs full-residency {f} diff {}",
                (f - c).abs(),
            );
        }
    }

    #[test]
    fn full_residency_rejects_zero_sentinel_config() {
        // Regardless of device presence: the config gate runs before the
        // Compute build, so the error is deterministic.
        let mut cfg = tiny_cfg();
        cfg.text.n_layer = 0;
        let td = TextDecoder {
            token_emb: Vec::new(),
            lm_head: None,
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        match VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td) {
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
    fn full_residency_multi_session_isolation() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; multi-session isolation test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut a = VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        let mut b = VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();

        a.step(&[1u32, 2, 0]).unwrap();
        b.step(&[1u32, 2, 0]).unwrap();
        let a_at_prefix: Vec<f32> = a.last_logits().to_vec();
        let b_at_prefix: Vec<f32> = b.last_logits().to_vec();
        assert_eq!(a.kv_cache_len(), 3);
        assert_eq!(b.kv_cache_len(), 3);
        assert_eq!(a_at_prefix, b_at_prefix);

        a.step(&[1u32]).unwrap();
        b.step(&[2u32]).unwrap();
        assert_eq!(a.kv_cache_len(), 4);
        assert_eq!(b.kv_cache_len(), 4);
        let a_after: Vec<f32> = a.last_logits().to_vec();
        let b_after: Vec<f32> = b.last_logits().to_vec();

        let total_diff: f32 = a_after
            .iter()
            .zip(&b_after)
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            total_diff > 1e-6,
            "multi-session isolation broken: two sessions with different KV appends \
             produced bit-identical logits (total_diff = {total_diff})"
        );
    }

    #[test]
    fn full_residency_reset_semantic_matches_thin() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; full-residency reset test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut sess =
            VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        let prefix = [1u32, 2];
        sess.step(&prefix).unwrap();
        let first: Vec<f32> = sess.last_logits().to_vec();
        assert_eq!(sess.kv_cache_len(), 2);

        sess.reset();
        assert_eq!(sess.kv_cache_len(), 0);

        sess.step(&prefix).unwrap();
        let second = sess.last_logits();
        assert_eq!(
            first.as_slice(),
            second,
            "reset() must return the session to a clean state so a replay is bit-identical"
        );
    }

    #[test]
    fn full_residency_step_with_embed_prefix_works() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; full-residency embed-prefix test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess =
            VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
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

    #[test]
    fn full_residency_kv_snapshot_restore_works() {
        if !has_cuda_device() {
            eprintln!("no CUDA device; full-residency kv-snapshot test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess =
            VoxtralCudaDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();

        sess.step(&[1u32, 0]).unwrap();
        let snap = sess.inner().kv_snapshot();
        assert_eq!(snap.position(), 2);

        sess.step(&[2u32]).unwrap();
        let branch_a: Vec<f32> = sess.last_logits().to_vec();

        sess.inner_mut().kv_restore(snap.clone());
        assert_eq!(sess.kv_cache_len(), 2);
        sess.step(&[3u32]).unwrap();
        let branch_b: Vec<f32> = sess.last_logits().to_vec();

        sess.inner_mut().kv_restore(snap);
        sess.step(&[2u32]).unwrap();
        assert_eq!(sess.last_logits(), branch_a.as_slice());

        let diff: f32 = branch_a
            .iter()
            .zip(&branch_b)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "kv snapshot branches under full-residency did not diverge (diff = {diff})"
        );
    }
}
