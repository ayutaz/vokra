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

pub mod attrs;
pub mod dct;
pub mod dispatch;
pub mod fft;
pub mod frontend;
pub mod fused_logmel;
pub mod istft;
pub mod istft_streaming;
pub mod kaldi_fbank;
pub mod mel;
pub mod mfcc;
pub mod preprocess;
pub mod resample;
pub mod stft;
pub mod window;

pub use dct::dct;
pub use dispatch::{OpValue, dispatch};
pub use frontend::{mel_attrs_from_spec, stft_attrs_from_spec};
pub use fused_logmel::fused_log_mel_scalar;
pub use istft::istft;
pub use istft_streaming::{IstftStreamingState, istft_streaming_oneshot};
pub use kaldi_fbank::{KaldiFbankOpts, kaldi_fbank};
pub use mel::mel_filterbank;
pub use mfcc::mfcc;
pub use preprocess::{apply_frontend, dc_offset_remove, pre_emphasis};
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
