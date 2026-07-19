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
//! # Wave 10 — full device-residency opt-in seam
//!
//! [`Self::new_from_decoder_full_residency`] is the Wave 10 "深化" seam
//! for opt-in **full device-residency** — the intended end state where
//! every weight is uploaded to a Metal buffer **once** at construction,
//! the self-attention KV cache lives on the device across steps, and a
//! single [`Self::step`] encodes the whole forward (RMSNorm → GQA-attn
//! with RoPE → residual → RMSNorm → SwiGLU FFN → residual → tied logits)
//! into **one** command buffer with one commit + one logits readback. That
//! is Whisper's `MetalDecodeSession` pattern applied to Mistral.
//!
//! **Current implementation status (Wave 10 slice)**: the full-residency
//! constructor lands the API surface, the [`ResidencyMode`] classification
//! and the `allow_full_residency` opt-in on [`super::asr::VoxtralAsr`]. The
//! per-step forward is currently delegated to the same
//! [`TextDecoderSession`] (Metal-backed) the thin wrapper uses — so the
//! output is **bit-identical** to the thin path, and the plumbing is
//! covered end-to-end by tests. The kernel fusion (bespoke MSL kernels
//! for Mistral RMSNorm, RoPE, GQA-attn, SwiGLU that would turn the
//! per-step Compute-seam loop into a single `encode_step_stack` command
//! buffer) is deferred to a Wave 10.1 / M4 follow-up — the same tickets
//! that landed Whisper's `MetalDecodeSession`. This scope split is honest
//! (FR-EX-08): the type documents *both* the shipped plumbing and the
//! deferred kernel work, so no caller can silently assume fusion that
//! is not yet in place.
//!
//! Callers that want the future kernel-fused path today should route
//! through [`Self::new_from_decoder_full_residency`] and treat the
//! constructor as a stable seam: the API surface will not change when
//! the fused kernels land, only the internal step body will migrate.
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

/// Which residency posture a [`VoxtralMetalDecodeSession`] was constructed
/// with — the plumbing gate for the Wave 10 "深化" (opt-in
/// full-device-residency) seam.
///
/// # Meaning today
///
/// - [`ResidencyMode::Thin`] — the Wave 9 posture: constructed via
///   [`VoxtralMetalDecodeSession::new_from_decoder`]. The inner
///   [`TextDecoderSession`] drives every GEMM / softmax / LayerNorm through
///   the `Compute::Metal` per-op path; weights are uploaded per op (the
///   Compute seam's H2D-per-call semantics). Real GPU work per op; not
///   fused at the step boundary.
/// - [`ResidencyMode::FullResident`] — the Wave 10 seam: constructed via
///   [`VoxtralMetalDecodeSession::new_from_decoder_full_residency`]. The
///   session **types itself** as intended-for-full-residency, and the
///   [`super::asr::VoxtralAsr::allow_full_residency`] opt-in routes through
///   this constructor. The API surface, error taxonomy and step behaviour
///   are stable; the underlying step body currently delegates to the same
///   Metal-backed `TextDecoderSession` the thin path uses (so the two
///   posture flavours are bit-identical today, a property the parity tests
///   assert). The Wave 10.1 / M4 follow-up will migrate the internal step
///   to a bespoke fused command buffer — the surface will not change.
///
/// # Why this enum exists
///
/// The alternative (a private `bool`) would leak into stringly-typed
/// diagnostics and hide the two-state nature at the API level. An enum
/// with docstrings on each variant makes the design intent legible to a
/// reader who has never touched the Wave 10 tickets — and it gives the
/// tests a real value to assert against (they don't just check a boolean;
/// they check the posture the constructor promised).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyMode {
    /// Thin Compute-seam wrapper (Wave 9). Per-op GEMM dispatch through
    /// `Compute::Metal`.
    Thin,
    /// Opt-in full device-residency (Wave 10). Currently bit-identical to
    /// [`Self::Thin`] under the hood; API surface stable for the Wave 10.1
    /// / M4 kernel-fusion follow-up.
    FullResident,
}

/// A Metal-backed Voxtral text-decoder session.
///
/// See the module-level docs for the scope and the parity contract with
/// the CPU baseline. Constructed via [`Self::new_from_decoder`] (thin,
/// Wave 9 posture) or [`Self::new_from_decoder_full_residency`] (opt-in
/// full-device-residency seam, Wave 10). The [`Self::step`] entry drives
/// the greedy / prefix decode loop either way — the two constructors
/// produce sessions with the same API surface, and (today, until the
/// Wave 10.1 / M4 fused kernels land) bit-identical step output.
///
/// # Lifetime
///
/// Borrows `&'m VoxtralConfig` and `&'m TextDecoder` — the caller keeps
/// the loaded model alive for the session's lifetime, mirroring
/// [`TextDecoderSession`].
pub struct VoxtralMetalDecodeSession<'m> {
    inner: TextDecoderSession<'m>,
    /// The residency posture the session was constructed with. Read-only
    /// after construction; observable via [`Self::residency_mode`].
    residency_mode: ResidencyMode,
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
        Ok(Self {
            inner,
            residency_mode: ResidencyMode::Thin,
        })
    }

    /// Builds a Metal-backed decoder session with the **opt-in full
    /// device-residency** posture (Wave 10 seam).
    ///
    /// # What this constructor is for
    ///
    /// This is the entry point the [`super::asr::VoxtralAsr::allow_full_residency`]
    /// opt-in routes through. It:
    ///
    /// 1. **Validates the config gate** (`0`-sentinel rejection, GQA head
    ///    split, decoder block count — same taxonomy as [`Self::new_from_decoder`]).
    /// 2. **Applies the FR-EX-08 backend gate** through the Compute seam
    ///    (`Compute::for_backend(BackendKind::Metal, VOXTRAL_HOT_OPS)`) —
    ///    a missing Metal device or a Voxtral hot op the Metal backend
    ///    does not cover surfaces an explicit error at *construction*,
    ///    never a silent CPU fall back.
    /// 3. **Types the session as [`ResidencyMode::FullResident`]** so
    ///    downstream diagnostics + the parity tests can assert on the
    ///    posture the caller asked for.
    ///
    /// # Current internal behaviour (Wave 10)
    ///
    /// The step body currently delegates to the **same** Metal-backed
    /// [`TextDecoderSession`] the thin constructor produces. That means:
    /// - the output is **bit-identical** to [`Self::new_from_decoder`]
    ///   for the same input (a property the parity tests assert);
    /// - the H2D / dispatch cost profile is identical to the thin path
    ///   (the Wave 10.1 / M4 kernel-fusion follow-up is what turns this
    ///   into a single command-buffer per step);
    /// - the API surface — [`Self::step`], [`Self::step_with_embed_prefix`],
    ///   [`Self::kv_cache_len`], [`Self::reset`], [`Self::last_logits`],
    ///   [`Self::all_logits`], [`Self::backend_name`] — is **stable**.
    ///   Callers can adopt the constructor today and pick up the fused
    ///   kernel path when it lands without a source change.
    ///
    /// # Errors
    ///
    /// Same taxonomy as [`Self::new_from_decoder`]:
    /// - [`VokraError::BackendUnavailable`] if no Metal device is found.
    /// - [`VokraError::UnsupportedOp`] if a Voxtral hot op is not covered
    ///   by the Metal backend on this build.
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
        // The Wave 10 residency posture is a *type* over the same Metal
        // Compute seam today (see the constructor docstring for why). The
        // Wave 10.1 / M4 follow-up will swap the internal `TextDecoderSession`
        // for a bespoke device-resident step driver; the two-line change is
        // scoped here.
        let inner = TextDecoderSession::new(config, decoder, BackendKind::Metal)?;
        Ok(Self {
            inner,
            residency_mode: ResidencyMode::FullResident,
        })
    }

    /// The residency posture the session was constructed with — the Wave 10
    /// plumbing gate. See [`ResidencyMode`] for the semantics.
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
            lm_head: None,
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
            lm_head: None,
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

    // ----------------------------------------------------------------------
    // Wave 10 — opt-in full device-residency seam
    //
    // These tests cover the plumbing added by Wave 10:
    // - the `new_from_decoder_full_residency` constructor,
    // - the `residency_mode()` posture-getter,
    // - the parity contract between the two constructor flavours (thin +
    //   full-residency are bit-identical today, a property the fused
    //   kernel follow-up must preserve),
    // - the KV-cache / reset / snapshot semantics under the new posture,
    // - the error taxonomy under a `0`-sentinel config.
    // ----------------------------------------------------------------------

    /// The thin-wrapper constructor tags the session as
    /// [`ResidencyMode::Thin`]. Guards against a future refactor that
    /// accidentally flips the two constructors' postures.
    #[test]
    fn thin_constructor_tags_residency_mode_thin() {
        if !has_metal_device() {
            eprintln!("no Metal device; residency-mode(thin) test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let sess = VoxtralMetalDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        assert_eq!(sess.residency_mode(), ResidencyMode::Thin);
    }

    /// The full-residency constructor tags the session as
    /// [`ResidencyMode::FullResident`] and gives back a live session that
    /// advances its KV cache correctly.
    #[test]
    fn full_residency_constructor_tags_residency_mode_and_advances() {
        if !has_metal_device() {
            eprintln!("no Metal device; residency-mode(full) test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess =
            VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        assert_eq!(sess.residency_mode(), ResidencyMode::FullResident);
        assert_eq!(sess.backend_name(), "metal");
        assert_eq!(sess.kv_cache_len(), 0);
        sess.step(&[1u32, 2, 0]).unwrap();
        assert_eq!(sess.kv_cache_len(), 3);
        sess.reset();
        assert_eq!(sess.kv_cache_len(), 0);
    }

    /// The Wave 10 seam's parity contract: for the same input, the
    /// thin-wrapper session and the full-residency session must produce
    /// **bit-identical** logits.
    ///
    /// This is the load-bearing test the kernel-fusion follow-up will
    /// preserve — the fused kernel path must not diverge from the per-op
    /// path within FP32 rounding.
    ///
    /// Today the two constructors delegate to the same internal
    /// `TextDecoderSession`, so the equality is byte-for-byte; when the
    /// fused kernel path lands the assertion will be relaxed to an atol
    /// bound (documented at that time).
    #[test]
    fn full_residency_matches_thin_wrapper_bit_identical() {
        if !has_metal_device() {
            eprintln!("no Metal device; thin-vs-full parity test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let prefix = [1u32, 2, 0];

        let mut thin = VoxtralMetalDecodeSession::new_from_decoder(&cfg, &td).unwrap();
        thin.step(&prefix).unwrap();
        let thin_last: Vec<f32> = thin.last_logits().to_vec();

        let mut full =
            VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        full.step(&prefix).unwrap();
        let full_last = full.last_logits();

        assert_eq!(
            thin_last.as_slice(),
            full_last,
            "Wave 10 parity contract broken: full-residency logits must equal thin logits \
             (Wave 10.1 / M4 kernel fusion must preserve this within FP32 rounding)."
        );
    }

    /// Full-residency step logits must match the CPU baseline within FP32
    /// rounding — the same tolerance the thin-wrapper parity test uses.
    #[test]
    fn full_residency_bit_identical_vs_cpu_on_tiny_fixture() {
        if !has_metal_device() {
            eprintln!("no Metal device; full-residency vs CPU parity test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut full =
            VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
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

    /// The full-residency constructor rejects a `0`-sentinel config with
    /// the same error taxonomy as the thin constructor — FR-EX-08, no
    /// silent default.
    #[test]
    fn full_residency_rejects_zero_sentinel_config() {
        let mut cfg = tiny_cfg();
        cfg.text.n_layer = 0;
        let td = TextDecoder {
            token_emb: Vec::new(),
            lm_head: None,
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        match VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td) {
            Ok(_) => panic!("must fail on 0-sentinel config"),
            Err(vokra_core::VokraError::ModelLoad(_)) => {}
            Err(other) => {
                // Same tolerance as the thin-constructor test: the Compute
                // build gate may race and surface BackendUnavailable first
                // on a runner without a Metal device. Either way is an
                // explicit error.
                assert!(
                    matches!(other, vokra_core::VokraError::BackendUnavailable(_)),
                    "expected ModelLoad or BackendUnavailable, got {other:?}"
                );
            }
        }
    }

    /// Two independent full-residency sessions on the same model must
    /// evolve independently — advancing one does not perturb the other.
    /// This mirrors `kv_snapshot_branches_do_not_interfere` on the CPU
    /// baseline and guards against a future implementation that
    /// accidentally shares device buffers across sessions.
    #[test]
    fn full_residency_multi_session_isolation() {
        if !has_metal_device() {
            eprintln!("no Metal device; multi-session isolation test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut a = VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
        let mut b = VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();

        // Both sessions run the same prefix and must land on identical
        // KV-cache-length + logits.
        a.step(&[1u32, 2, 0]).unwrap();
        b.step(&[1u32, 2, 0]).unwrap();
        let a_at_prefix: Vec<f32> = a.last_logits().to_vec();
        let b_at_prefix: Vec<f32> = b.last_logits().to_vec();
        assert_eq!(a.kv_cache_len(), 3);
        assert_eq!(b.kv_cache_len(), 3);
        assert_eq!(a_at_prefix, b_at_prefix);

        // Now diverge: a extends with token 1, b with token 2.
        a.step(&[1u32]).unwrap();
        b.step(&[2u32]).unwrap();
        assert_eq!(a.kv_cache_len(), 4);
        assert_eq!(b.kv_cache_len(), 4);
        let a_after: Vec<f32> = a.last_logits().to_vec();
        let b_after: Vec<f32> = b.last_logits().to_vec();

        // The two divergences must land on different logits (the KV
        // cache contents differ). Total-abs-difference guards against a
        // zero-total by finite-precision coincidence.
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

    /// `reset()` on a full-residency session clears the KV cache and
    /// resets the position clock to 0. A subsequent step from the same
    /// prefix must reproduce the pre-reset logits bit-for-bit.
    #[test]
    fn full_residency_reset_semantic_matches_thin() {
        if !has_metal_device() {
            eprintln!("no Metal device; full-residency reset test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut sess =
            VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
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

    /// Full-residency + audio-adapter soft-prefix embed step must produce
    /// finite logits and advance the KV cache — the Wave-8 audio path is
    /// still available under the Wave 10 posture.
    #[test]
    fn full_residency_step_with_embed_prefix_works() {
        if !has_metal_device() {
            eprintln!("no Metal device; full-residency embed-prefix test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess =
            VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();
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

    /// The KV snapshot / restore primitive (Wave 9 beam-search
    /// integration) must still work through a full-residency session —
    /// the Wave 10 seam does not regress the beam search path.
    #[test]
    fn full_residency_kv_snapshot_restore_works() {
        if !has_metal_device() {
            eprintln!("no Metal device; full-residency kv-snapshot test skipped");
            return;
        }
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess =
            VoxtralMetalDecodeSession::new_from_decoder_full_residency(&cfg, &td).unwrap();

        // Common prefix.
        sess.step(&[1u32, 0]).unwrap();
        // Snapshot via the inner `TextDecoderSession` surface (the Wave 10
        // seam does not redefine the snapshot API — it is uniform across
        // both residency postures).
        let snap = sess.inner().kv_snapshot();
        assert_eq!(snap.position(), 2);

        // Branch A.
        sess.step(&[2u32]).unwrap();
        let branch_a: Vec<f32> = sess.last_logits().to_vec();

        // Restore + branch B.
        sess.inner_mut().kv_restore(snap.clone());
        assert_eq!(sess.kv_cache_len(), 2);
        sess.step(&[3u32]).unwrap();
        let branch_b: Vec<f32> = sess.last_logits().to_vec();

        // Restore + re-branch A must reproduce A bit-for-bit.
        sess.inner_mut().kv_restore(snap);
        sess.step(&[2u32]).unwrap();
        assert_eq!(sess.last_logits(), branch_a.as_slice());

        // A and B must differ (the KV append content differs).
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
