//! IR dispatch for the M0-04 front-end ops (M0-04-T17).
//!
//! Bridges the `vokra-core` flat op enum ([`vokra_core::OpKind`]) to the
//! `vokra-ops` implementations. M0 has no graph executor yet (the
//! [`AudioGraph`](vokra_core::AudioGraph) carries structure, not buffers), so
//! this is the op-level evaluation the executor will call once it lands: given
//! an `OpKind` (a graph node's op, i.e. the "graph path") and its input
//! [`OpValue`]s, it produces the outputs — identical to calling the op function
//! directly (verified in the tests).
//!
//! Ops outside the M0-04 set return [`VokraError::UnsupportedOp`] rather than
//! silently doing nothing (FR-EX-08: no silent fallback).

use vokra_core::ir::graph::{DctAttrs, IstftAttrs, MelAttrs, MfccAttrs, StftAttrs};
use vokra_core::{OpKind, Result, VokraError};

use crate::dct::dct;
use crate::istft::istft;
use crate::mel::MelFilterbank;
use crate::mfcc::mfcc;
use crate::stft::{Spectrogram, stft};

/// A runtime tensor value crossing an op boundary: real, or split-complex.
///
/// Shapes are row-major. This is the M0 stand-in for typed IR tensor storage;
/// complex values stay split into `re`/`im` because public `complex64` on the
/// IR is FR-EX-09 (out of M0 scope).
#[derive(Debug, Clone, PartialEq)]
pub enum OpValue {
    /// A real tensor.
    Real {
        /// Row-major dimensions.
        shape: Vec<usize>,
        /// Elements, row-major.
        data: Vec<f32>,
    },
    /// A split-complex tensor (`re`/`im` share `shape`).
    Complex {
        /// Row-major dimensions.
        shape: Vec<usize>,
        /// Real parts, row-major.
        re: Vec<f32>,
        /// Imaginary parts, row-major.
        im: Vec<f32>,
    },
}

impl OpValue {
    /// Builds a [`OpValue::Real`].
    pub fn real(shape: impl Into<Vec<usize>>, data: impl Into<Vec<f32>>) -> Self {
        Self::Real {
            shape: shape.into(),
            data: data.into(),
        }
    }

    /// Returns the shape and data of a [`OpValue::Real`], or `None`.
    pub fn as_real(&self) -> Option<(&[usize], &[f32])> {
        match self {
            Self::Real { shape, data } => Some((shape, data)),
            Self::Complex { .. } => None,
        }
    }

    /// Returns the shape and split parts of a [`OpValue::Complex`], or `None`.
    pub fn as_complex(&self) -> Option<(&[usize], &[f32], &[f32])> {
        match self {
            Self::Complex { shape, re, im } => Some((shape, re, im)),
            Self::Real { .. } => None,
        }
    }
}

/// Evaluates an M0-04 front-end op over `inputs`.
///
/// # Errors
///
/// Returns [`VokraError::UnsupportedOp`] for ops outside the M0-04 set and
/// [`VokraError::InvalidArgument`] for arity / kind / shape mismatches (plus any
/// error from the underlying op).
pub fn dispatch(op: &OpKind, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    match op {
        OpKind::Stft(attrs) => run_stft(attrs, inputs),
        OpKind::Istft(attrs) => run_istft(attrs, inputs),
        OpKind::MelFilterbank(attrs) => run_mel(attrs, inputs),
        OpKind::Mfcc(attrs) => run_mfcc(attrs, inputs),
        OpKind::Dct(attrs) => run_dct(attrs, inputs),
        other => Err(VokraError::UnsupportedOp(format!(
            "{other:?} is not an M0-04 front-end op"
        ))),
    }
}

fn expect_arity(inputs: &[OpValue], n: usize, op: &str) -> Result<()> {
    if inputs.len() != n {
        return Err(VokraError::InvalidArgument(format!(
            "{op}: expected {n} input(s), got {}",
            inputs.len()
        )));
    }
    Ok(())
}

fn expect_real<'a>(v: &'a OpValue, op: &str) -> Result<(&'a [usize], &'a [f32])> {
    v.as_real()
        .ok_or_else(|| VokraError::InvalidArgument(format!("{op}: expected a real input")))
}

fn run_stft(attrs: &StftAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "stft")?;
    let (_, signal) = expect_real(&inputs[0], "stft")?;
    let spec = stft(signal, attrs)?;
    Ok(vec![OpValue::Complex {
        shape: vec![spec.frames, spec.bins],
        re: spec.re,
        im: spec.im,
    }])
}

fn run_istft(attrs: &IstftAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "istft")?;
    let (shape, re, im) = inputs[0]
        .as_complex()
        .ok_or_else(|| VokraError::InvalidArgument("istft: expected a complex input".to_owned()))?;
    if shape.len() != 2 {
        return Err(VokraError::InvalidArgument(
            "istft: input must be [frames, bins]".to_owned(),
        ));
    }
    let spec = Spectrogram {
        frames: shape[0],
        bins: shape[1],
        re: re.to_vec(),
        im: im.to_vec(),
    };
    let signal = istft(&spec, attrs)?;
    Ok(vec![OpValue::real(vec![signal.len()], signal)])
}

/// Splits a `[frames, cols]` real input into `(frames, cols, data)`.
fn as_matrix<'a>(v: &'a OpValue, op: &str) -> Result<(usize, usize, &'a [f32])> {
    let (shape, data) = expect_real(v, op)?;
    if shape.len() != 2 {
        return Err(VokraError::InvalidArgument(format!(
            "{op}: input must be 2-D [frames, cols]"
        )));
    }
    Ok((shape[0], shape[1], data))
}

fn run_mel(attrs: &MelAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "mel_filterbank")?;
    let (frames, n_freqs, power) = as_matrix(&inputs[0], "mel_filterbank")?;
    let fb = MelFilterbank::new(attrs);
    if n_freqs != fb.n_freqs {
        return Err(VokraError::InvalidArgument(format!(
            "mel_filterbank: input has {n_freqs} freq bins, expected {}",
            fb.n_freqs
        )));
    }
    let mel = fb.apply(power, frames);
    Ok(vec![OpValue::real(vec![frames, fb.n_mels], mel)])
}

fn run_mfcc(attrs: &MfccAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "mfcc")?;
    let (frames, n_freqs, power) = as_matrix(&inputs[0], "mfcc")?;
    let expected = attrs.mel.n_fft / 2 + 1;
    if n_freqs != expected {
        return Err(VokraError::InvalidArgument(format!(
            "mfcc: input has {n_freqs} freq bins, expected {expected}"
        )));
    }
    let out = mfcc(power, frames, attrs);
    Ok(vec![OpValue::real(vec![frames, attrs.n_mfcc], out)])
}

fn run_dct(attrs: &DctAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "dct")?;
    let (rows, n, data) = as_matrix(&inputs[0], "dct")?;
    if let Some(n_out) = attrs.n_out {
        if n_out > n {
            return Err(VokraError::InvalidArgument(
                "dct: n_out exceeds transform length".to_owned(),
            ));
        }
    }
    let out = dct(data, rows, n, attrs);
    let n_out = attrs.n_out.unwrap_or(n);
    Ok(vec![OpValue::real(vec![rows, n_out], out)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::ir::graph::{MelAttrs, MfccAttrs, StftAttrs};

    #[test]
    fn stft_dispatch_matches_direct_call() {
        let signal: Vec<f32> = (0..2048).map(|t| (t as f32 * 0.05).sin()).collect();
        let attrs = StftAttrs::new(256, 128);
        let direct = stft(&signal, &attrs).unwrap();

        let out = dispatch(
            &OpKind::Stft(attrs),
            &[OpValue::real(vec![signal.len()], signal.clone())],
        )
        .unwrap();
        let (shape, re, im) = out[0].as_complex().unwrap();
        assert_eq!(shape, &[direct.frames, direct.bins]);
        assert_eq!(re, direct.re.as_slice());
        assert_eq!(im, direct.im.as_slice());
    }

    #[test]
    fn dct_dispatch_matches_direct_call() {
        let n = 8;
        let rows = 3;
        let data: Vec<f32> = (0..rows * n).map(|i| i as f32 * 0.25).collect();
        let attrs = DctAttrs::new();
        let direct = dct(&data, rows, n, &attrs);
        let out = dispatch(&OpKind::Dct(attrs), &[OpValue::real(vec![rows, n], data)]).unwrap();
        let (shape, got) = out[0].as_real().unwrap();
        assert_eq!(shape, &[rows, n]);
        assert_eq!(got, direct.as_slice());
    }

    #[test]
    fn mel_and_mfcc_dispatch_match_direct_calls() {
        let frames = 4;
        let mel_attrs = MelAttrs::new(16000, 400, 40);
        let n_freqs = 400 / 2 + 1;
        let power: Vec<f32> = (0..frames * n_freqs)
            .map(|i| (i % 5) as f32 * 0.1)
            .collect();

        let fb_direct = MelFilterbank::new(&mel_attrs).apply(&power, frames);
        let mel_out = dispatch(
            &OpKind::MelFilterbank(mel_attrs.clone()),
            &[OpValue::real(vec![frames, n_freqs], power.clone())],
        )
        .unwrap();
        assert_eq!(mel_out[0].as_real().unwrap().1, fb_direct.as_slice());

        let mfcc_attrs = MfccAttrs::new(mel_attrs, 13);
        let mfcc_direct = mfcc(&power, frames, &mfcc_attrs);
        let mfcc_out = dispatch(
            &OpKind::Mfcc(mfcc_attrs),
            &[OpValue::real(vec![frames, n_freqs], power)],
        )
        .unwrap();
        assert_eq!(mfcc_out[0].as_real().unwrap().1, mfcc_direct.as_slice());
    }

    #[test]
    fn stft_istft_dispatch_roundtrip() {
        let signal: Vec<f32> = (0..4096).map(|t| (t as f32 * 0.03).sin()).collect();
        let sa = StftAttrs::new(512, 128);
        let spec = dispatch(
            &OpKind::Stft(sa),
            &[OpValue::real(vec![signal.len()], signal.clone())],
        )
        .unwrap();

        let mut ia = IstftAttrs::new(512, 128);
        ia.length = Some(signal.len());
        let recon = dispatch(&OpKind::Istft(ia), &spec).unwrap();
        let (_, y) = recon[0].as_real().unwrap();
        let mut max = 0.0f32;
        for i in 512..signal.len() - 512 {
            max = max.max((signal[i] - y[i]).abs());
        }
        assert!(max < 1e-2, "roundtrip error {max}");
    }

    #[test]
    fn non_m0_04_op_is_unsupported() {
        let err = dispatch(&OpKind::MatMul, &[]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
