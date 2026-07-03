//! `frontend_spec` → op-attribute translation (M1-03; FR-OP-03 / NFR-QL-03).
//!
//! Maps a [`FrontendSpec`] (the `vokra.frontend.*` GGUF chunk) onto the
//! [`StftAttrs`] / [`MelAttrs`] the M0-04 front-end ops consume. This is the
//! **librosa / torchaudio / TF compatibility layer**: instead of assuming
//! Whisper's Slaney defaults, the runtime honours the spec's own
//! `window_type` / `pad_mode` / `mel_norm` / `htk_mode` / `fmin` / `fmax`, so a
//! model configured to a different reference front-end builds the matching
//! filterbank and window (reviewer C note #2, frontend bit-exactness).
//!
//! The five STFT knobs that the 13-field `frontend_spec` does **not** carry —
//! `window_symmetry`, `center`, `normalization`, `causal`, `real_input` — are
//! fixed here to the standard analysis-STFT conventions ([`StftAttrs::new`]:
//! periodic window, `center = true`, backward norm, non-causal, real RFFT),
//! matching `torch.stft` / librosa defaults.
//!
//! Unknown enum strings, and any `window_type` the spec cannot fully describe,
//! are rejected with [`VokraError::InvalidArgument`] rather than silently
//! defaulted.

use vokra_core::ir::graph::{
    MelAttrs, MelNorm, MelScale, Normalization, PadMode, StftAttrs, Window, WindowSymmetry,
};
use vokra_core::{FrontendSpec, Result, VokraError};

/// Builds [`StftAttrs`] from a [`FrontendSpec`].
///
/// `n_fft` / `hop` / `win_length` are taken verbatim; `window_type` and
/// `pad_mode` are parsed into their enums; the remaining STFT conventions
/// (periodic / centered / backward / non-causal / real) are the standard
/// analysis defaults (see the module docs).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `window_type` or `pad_mode` is unknown, or
/// if `window_type` is `"kaiser"` (not representable — see [`window_from_str`]).
pub fn stft_attrs_from_spec(spec: &FrontendSpec) -> Result<StftAttrs> {
    Ok(StftAttrs {
        n_fft: spec.n_fft as usize,
        hop_length: spec.hop as usize,
        win_length: spec.win_length as usize,
        window: window_from_str(&spec.window_type)?,
        window_symmetry: WindowSymmetry::Periodic,
        center: true,
        pad_mode: pad_mode_from_str(&spec.pad_mode)?,
        normalization: Normalization::Backward,
        causal: false,
        real_input: true,
    })
}

/// Builds [`MelAttrs`] from a [`FrontendSpec`].
///
/// `sample_rate` / `n_fft` / `n_mels` / `fmin` are taken verbatim; `fmax` is
/// carried explicitly (`Some(spec.fmax)`); the warp is [`MelScale::Htk`] when
/// `htk_mode` is set, else [`MelScale::Slaney`]; `mel_norm` selects the
/// normalization.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `mel_norm` is unknown.
pub fn mel_attrs_from_spec(spec: &FrontendSpec) -> Result<MelAttrs> {
    Ok(MelAttrs {
        sample_rate: spec.sample_rate,
        n_fft: spec.n_fft as usize,
        n_mels: spec.n_mels as usize,
        fmin: spec.fmin,
        fmax: Some(spec.fmax),
        scale: if spec.htk_mode {
            MelScale::Htk
        } else {
            MelScale::Slaney
        },
        norm: mel_norm_from_str(&spec.mel_norm)?,
    })
}

/// Parses a `frontend_spec` `window_type` string into a [`Window`].
///
/// `"kaiser"` is deliberately rejected: the [`Window::Kaiser`] variant needs a
/// shape parameter `beta`, and the 13-field `frontend_spec` carries no such key,
/// so it cannot be reconstructed without inventing a value (CLAUDE.md: no
/// fabricated numbers). Whisper — the only shipped front-end — uses `"hann"`.
fn window_from_str(s: &str) -> Result<Window> {
    match s {
        "hann" => Ok(Window::Hann),
        "hamming" => Ok(Window::Hamming),
        "blackman_harris" | "blackmanharris" => Ok(Window::BlackmanHarris),
        "kaiser" => Err(VokraError::InvalidArgument(
            "frontend_spec window_type=\"kaiser\" is not representable: the \
             chunk carries no Kaiser beta"
                .to_owned(),
        )),
        other => Err(VokraError::InvalidArgument(format!(
            "unknown frontend_spec window_type `{other}`"
        ))),
    }
}

/// Parses a `frontend_spec` `mel_norm` string into a [`MelNorm`].
fn mel_norm_from_str(s: &str) -> Result<MelNorm> {
    match s {
        "slaney" => Ok(MelNorm::Slaney),
        "none" => Ok(MelNorm::None),
        other => Err(VokraError::InvalidArgument(format!(
            "unknown frontend_spec mel_norm `{other}`"
        ))),
    }
}

/// Parses a `frontend_spec` `pad_mode` string into a [`PadMode`].
fn pad_mode_from_str(s: &str) -> Result<PadMode> {
    match s {
        "reflect" => Ok(PadMode::Reflect),
        "constant" => Ok(PadMode::Constant),
        "edge" => Ok(PadMode::Edge),
        other => Err(VokraError::InvalidArgument(format!(
            "unknown frontend_spec pad_mode `{other}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::MelFilterbank;

    /// A Whisper-shaped spec (the values the converter writes for whisper-base).
    fn whisper_spec() -> FrontendSpec {
        FrontendSpec {
            n_fft: 400,
            hop: 160,
            win_length: 400,
            window_type: "hann".to_owned(),
            mel_norm: "slaney".to_owned(),
            htk_mode: false,
            fmin: 0.0,
            fmax: 8000.0,
            n_mels: 80,
            pad_mode: "reflect".to_owned(),
            dc_offset_removal: false,
            pre_emphasis: 0.0,
            sample_rate: 16_000,
        }
    }

    #[test]
    fn whisper_spec_maps_to_the_hard_coded_stft_attrs() {
        // The translation of the Whisper spec must equal the librosa/torch
        // default `StftAttrs::new(400, 160)` the M0-06 front-end used directly,
        // so making `log_mel` data-driven changes nothing numerically.
        let got = stft_attrs_from_spec(&whisper_spec()).unwrap();
        assert_eq!(got, StftAttrs::new(400, 160));
    }

    #[test]
    fn whisper_spec_builds_a_bit_identical_mel_filterbank() {
        // `mel_attrs_from_spec` carries `fmax = Some(8000.0)` where the M0-06
        // path used `MelAttrs::new(..)`'s `None` (⇒ Nyquist = 8000.0). The two
        // produce byte-identical filter weights, so log-mel output is unchanged.
        let from_spec = MelFilterbank::new(&mel_attrs_from_spec(&whisper_spec()).unwrap());
        let hard_coded = MelFilterbank::new(&MelAttrs::new(16_000, 400, 80));
        assert_eq!(from_spec.n_mels, hard_coded.n_mels);
        assert_eq!(from_spec.n_freqs, hard_coded.n_freqs);
        assert_eq!(from_spec.weights, hard_coded.weights);
    }

    #[test]
    fn window_strings_map_to_the_right_enum() {
        assert_eq!(window_from_str("hann").unwrap(), Window::Hann);
        assert_eq!(window_from_str("hamming").unwrap(), Window::Hamming);
        assert_eq!(
            window_from_str("blackman_harris").unwrap(),
            Window::BlackmanHarris
        );
        assert_eq!(
            window_from_str("blackmanharris").unwrap(),
            Window::BlackmanHarris
        );
        // kaiser needs a beta the chunk cannot carry → rejected, not defaulted.
        assert!(matches!(
            window_from_str("kaiser"),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            window_from_str("triangular"),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn mel_norm_and_scale_track_the_spec() {
        assert_eq!(mel_norm_from_str("slaney").unwrap(), MelNorm::Slaney);
        assert_eq!(mel_norm_from_str("none").unwrap(), MelNorm::None);
        assert!(matches!(
            mel_norm_from_str("l2"),
            Err(VokraError::InvalidArgument(_))
        ));

        // htk_mode drives MelScale independently of the norm string.
        let mut spec = whisper_spec();
        assert_eq!(mel_attrs_from_spec(&spec).unwrap().scale, MelScale::Slaney);
        spec.htk_mode = true;
        assert_eq!(mel_attrs_from_spec(&spec).unwrap().scale, MelScale::Htk);
    }

    #[test]
    fn pad_mode_strings_map_and_reject_unknown() {
        assert_eq!(pad_mode_from_str("reflect").unwrap(), PadMode::Reflect);
        assert_eq!(pad_mode_from_str("constant").unwrap(), PadMode::Constant);
        assert_eq!(pad_mode_from_str("edge").unwrap(), PadMode::Edge);
        assert!(matches!(
            pad_mode_from_str("wrap"),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn unknown_enum_strings_propagate_through_the_attr_builders() {
        let mut spec = whisper_spec();
        spec.window_type = "bogus".to_owned();
        assert!(stft_attrs_from_spec(&spec).is_err());

        let mut spec = whisper_spec();
        spec.pad_mode = "bogus".to_owned();
        assert!(stft_attrs_from_spec(&spec).is_err());

        let mut spec = whisper_spec();
        spec.mel_norm = "bogus".to_owned();
        assert!(mel_attrs_from_spec(&spec).is_err());
    }
}
