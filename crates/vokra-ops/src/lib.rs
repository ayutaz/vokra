//! # vokra-ops
//!
//! Speech-specialized operators for the Vokra runtime (SRS §1.3:
//! "音声オペレータ" — the audio operators crate).
//!
//! Operator implementations land with their owning work packages:
//!
//! - **M0-04** (this WP, landed): `stft` / `istft` / `mel_filterbank` /
//!   `mfcc` / `dct` with explicit attributes (window / hop / n_fft / pad /
//!   normalization / causal / `real_input` RFFT — FR-OP-01/03) and the CPU FFT
//!   lowering (a from-scratch Rust reimplementation of the pocketfft algorithm,
//!   BSD-3 — FR-OP-05). See [`fft`], [`window`], [`stft`], [`istft`], [`mel`],
//!   [`dct`], [`mfcc`] and the [`dispatch`] bridge to the IR;
//! - **M0-05**: LSTM family needed by the Silero VAD subgraph;
//! - **M0-06**: attention / decoder family needed by Whisper;
//! - **M1-06** (landed): front-end preprocessing — [`resample`] (a native
//!   Kaiser-windowed-sinc converter, GPL-free by construction) and the
//!   `frontend_spec`-driven [`dc_offset_remove`] / [`pre_emphasis`] chain
//!   ([`apply_frontend`]);
//! - **M1-03** (landed): the [`frontend`] `frontend_spec` → `StftAttrs` /
//!   `MelAttrs` translation ([`stft_attrs_from_spec`] / [`mel_attrs_from_spec`])
//!   — the librosa/torchaudio/TF compat layer that makes the log-mel front-end
//!   data-driven; the bit-exact *inspection* of the chunk lives in `vokra-core`;
//! - **M0-08** (landed): the Kaldi fbank front-end the CAM++ speaker encoder
//!   needs — the [`window::povey`] window, the [`mel`] Kaldi mel-domain ramp
//!   (`MelInterp::Mel`), and [`kaldi_fbank`] (snip-edges framing, per-frame
//!   DC/pre-emphasis, power spectrum, log, CMN);
//! - later WPs: vocoder chains, flow-matching samplers, codec decode, and
//!   the rest of the audio dialect (CLAUDE.md "音声特化オペレータ").
//!
//! The corresponding [`vokra_core::OpKind`] variants for the M0-04 ops are
//! defined in `vokra-core` (the attribute types embedded in those variants
//! live there because the crate dependency edge runs `vokra-ops → vokra-core`);
//! remaining families are added by their own WPs.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! `unsafe` + SIMD intrinsics are *permitted inside operator
//! implementations* for RTF, which is why this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` below. Public APIs must stay safe,
//! and every `unsafe` block requires a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks` at the workspace level).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03). The M0-04 ops are
// written in safe Rust; the opt-out is kept for the SIMD kernels of later WPs.
#![allow(unsafe_code)]

// ---- M4-03 aec (FR-OP-60, runtime function — not an OpKind variant) -----
// SpeexDSP MDF/AUMDF float-build port; the time-tagged far-end queue lives
// in vokra-core::stream::aec_ref (crate edge runs ops → core). New module +
// re-export kept as one localized patch block (M3-05/M3-06 pattern) so
// parallel M4 waves rebase cleanly.
pub mod aec;
// -------------------------------------------------------------------------
pub mod attrs;
// ---- M4-04 dac_rvq codec decode (RVQ family, FR-OP-30) ------------------
// DAC's factorized (low-dim codebook + per-quantizer out_proj) residual VQ
// decode. Shapes verified from the upstream descript-audio-codec (MIT)
// implementation + the 24 kHz checkpoint metadata (ADR M4-04 §T02). Paged
// variant primary block size = 4 (75-86 Hz released variants).
pub mod dac_rvq;
// -------------------------------------------------------------------------
pub mod dct;
// ---- M4-04 encodec_rvq (engine op only — FR-OP-32 permanent weight
// exclusion; parity uses synthetic codebooks, never pretrained weights) ----
pub mod encodec_rvq;
// -------------------------------------------------------------------------
pub mod dispatch;
pub mod fft;
// ---- M4-16 FSQ codec family (FR-OP-31, runtime functions — not OpKind
// variants). Single-stage subgraph, deliberately separate from the RVQ
// family (FR-OP-30: mimi_rvq / dac_rvq / encodec_rvq): no cross-codebook
// residual sum, no paged variant, no cross-family adapter. Localized patch
// block (M3-05/M3-06 pattern) for clean parallel-wave rebases.
pub mod fsq_codec;
// -------------------------------------------------------------------------
// ---- M3-05 flow_sampler / ODE solvers (runtime function, FR-EX-10) -----
// New module + re-export block, kept as a single localized patch so Wave 3
// (M3-06 / M3-07) has a clean rebase target. The op-only re-export follows
// the M3-08 length_conditioning and M3-17 prosody pattern.
pub mod flow_sampler;
// -----------------------------------------------------------------------
pub mod frontend;
pub mod fused_logmel;
// ---- M3-07 hifigan_generator (vocoder chain, FR-OP-10) ------------------
// New module + re-export block. INT8 is an opt-in path (per-channel
// calibration + NFR-QL-02 5% spectral check required); FR-EX-08 is preserved
// at the runtime function (`VokraError::HifiganInt8VerifyMissing` when the
// gate is un-satisfied, `VokraError::UnsupportedOp` while the INT8 kernel
// stays deferred to the M3-09 consumer WP). ADR-equivalent rationale lives in
// the module-level docstring.
pub mod hifigan;
// -------------------------------------------------------------------------
pub mod istft;
pub mod istft_streaming;
pub mod kaldi_fbank;
pub mod length_conditioning;
pub mod mel;
pub mod mfcc;
// ---- M3-06 mimi_rvq codec decode (RVQ family, FR-OP-30) -----------------
// New module + re-export block. Wave 3 (M3-07) will touch the same file, so
// this block is kept localised for a clean rebase target. Mimi is CC-BY 4.0
// (attribution recorded in NOTICE / docs/license-audit.md — ADR M3-06 §D3);
// EnCodec weights (CC-BY-NC 4.0) are permanently excluded from the official
// model zoo (FR-OP-32 — enforced by the M2-13 compliance gate and the
// `scripts/compliance/check-encodec-exclusion.sh` release-side script).
pub mod mimi_rvq;
// -------------------------------------------------------------------------
pub mod preprocess;
pub mod prosody;
pub mod resample;
pub mod stft;
pub mod window;

// ---- M4-03 aec re-exports ------------------------------------------------
pub use aec::{Aec, AecAttrs, AecStatus};
// ---------------------------------------------------------------------------
// ---- M4-04 dac_rvq re-exports --------------------------------------------
pub use dac_rvq::{
    DacOutProj, DacRvqAttrs, dac_paged_dims, dac_rvq_decode, dac_rvq_decode_paged,
    dac_rvq_read_summed,
};
// ---------------------------------------------------------------------------
pub use dct::dct;
// ---- M4-04 encodec_rvq re-exports -----------------------------------------
pub use encodec_rvq::{EncodecRvqAttrs, encodec_rvq_decode};
// ---------------------------------------------------------------------------
pub use dispatch::{OpValue, dispatch};
// ---- M3-05 flow_sampler re-exports --------------------------------------
pub use flow_sampler::{
    CfgMode, CfgScaleProfile, FlowSamplerConfig, FlowSamplerState, ForwardPass, OdeSolver,
    Schedule, flow_sample,
};
// -------------------------------------------------------------------------
// ---- M4-16 fsq_codec re-exports ------------------------------------------
pub use fsq_codec::{
    FsqOutProj, WavTokenizerVqAttrs, Xcodec2FsqAttrs, fsq_index_to_grid_codes,
    wavtokenizer_vq_decode, xcodec2_fsq_decode,
};
// ---------------------------------------------------------------------------
pub use frontend::{mel_attrs_from_spec, stft_attrs_from_spec};
pub use fused_logmel::fused_log_mel_scalar;
// ---- M3-07 hifigan_generator re-exports ---------------------------------
pub use hifigan::{
    CalibrationStrategy, CalibrationTable, HifiGanCalibrator, HifiGanConfig, HifiGanPrecision,
    HifiGanSpectralChecker, HifiGanWeights, MrfBranchWeights, ResBlockLayer,
    SPECTRAL_CHECK_THRESHOLD, SpectralCheckResult, UpsampleStageWeights, hifigan_generator,
};
// -------------------------------------------------------------------------
pub use istft::istft;
pub use istft_streaming::{IstftStreamingState, istft_streaming_oneshot};
pub use kaldi_fbank::{KaldiFbankOpts, kaldi_fbank};
pub use length_conditioning::length_conditioning;
pub use mel::mel_filterbank;
pub use mfcc::mfcc;
// ---- M3-06 mimi_rvq re-exports ------------------------------------------
pub use mimi_rvq::{
    CodebookTable, MimiDecoder, MimiRvqAttrs, codebook_lookup, mimi_paged_dims, mimi_rvq_decode,
    mimi_rvq_decode_paged, mimi_rvq_read_summed, mimi_rvq_read_summed_range,
};
// -------------------------------------------------------------------------
pub use preprocess::{apply_frontend, dc_offset_remove, pre_emphasis};
pub use prosody::{ApplyProsody, ProsodyControl};
pub use resample::resample;
pub use stft::{Spectrogram, stft};
pub use vokra_core::Complex32;

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_vokra_core() {
        // Smoke test for the crate wiring (M0-02-T02): vokra-ops builds on
        // the vokra-core IR types.
        let dtype = vokra_core::DType::F32;
        assert_eq!(dtype.size_in_bytes(), 4);
    }
}
