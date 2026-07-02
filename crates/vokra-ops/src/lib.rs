//! # vokra-ops
//!
//! Speech-specialized operators for the Vokra runtime (SRS §1.3:
//! "音声オペレータ" — the audio operators crate).
//!
//! M0-02 ships only the crate skeleton. Operator implementations land with
//! their owning work packages:
//!
//! - **M0-04**: `stft` / `istft` / `mel_filterbank` / `mfcc` / `dct` with
//!   explicit attributes (window / hop / n_fft / pad / normalization /
//!   causal / `real_input` RFFT — FR-OP-01/03) and the CPU FFT lowering
//!   (pocketfft, BSD-3, ported to Rust — FR-OP-05);
//! - **M0-05**: LSTM family needed by the Silero VAD subgraph;
//! - **M0-06**: attention / decoder family needed by Whisper;
//! - later WPs: vocoder chains, flow-matching samplers, codec decode, and
//!   the rest of the audio dialect (CLAUDE.md "音声特化オペレータ").
//!
//! The corresponding `OpKind` variants are added in `vokra-core` by those
//! same WPs.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! `unsafe` + SIMD intrinsics are *permitted inside operator
//! implementations* for RTF, which is why this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` below. Public APIs must stay safe,
//! and every `unsafe` block requires a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks` at the workspace level).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03).
#![allow(unsafe_code)]

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
