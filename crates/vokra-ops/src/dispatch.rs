//! IR dispatch for the front-end / preprocessing ops (M0-04-T17; M1-03).
//!
//! Bridges the `vokra-core` flat op enum ([`vokra_core::OpKind`]) to the
//! `vokra-ops` implementations. M0 has no graph executor yet (the
//! [`AudioGraph`](vokra_core::AudioGraph) carries structure, not buffers), so
//! this is the op-level evaluation the executor will call once it lands: given
//! an `OpKind` (a graph node's op, i.e. the "graph path") and its input
//! [`OpValue`]s, it produces the outputs — identical to calling the op function
//! directly (verified in the tests). It covers the M0-04 front-end ops and the
//! M1-06 amplitude-preprocessing ops (`resample` / `dc_offset_remove` /
//! `pre_emphasis`, wrapped as [`OpKind`] variants in M1-03).
//!
//! Ops outside that set return [`VokraError::UnsupportedOp`] rather than
//! silently doing nothing (FR-EX-08: no silent fallback).

use vokra_core::ir::graph::{
    DctAttrs, IstftAttrs, IstftStreamingAttrs, LengthConditioningAttrs, MelAttrs, MfccAttrs,
    PreEmphasisAttrs, ResampleAttrs, StftAttrs,
};
use vokra_core::{OpKind, Result, VokraError};

use crate::dct::dct;
use crate::istft::istft;
use crate::istft_streaming::istft_streaming_oneshot;
use crate::length_conditioning::length_conditioning;
use crate::mel::MelFilterbank;
use crate::mfcc::mfcc;
use crate::preprocess::{dc_offset_remove, pre_emphasis};
use crate::resample::resample;
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

/// Evaluates a front-end / preprocessing op over `inputs`.
///
/// Covers the M0-04 front-end ops (`stft` / `istft` / `mel_filterbank` /
/// `mfcc` / `dct`) and the M1-06 amplitude-preprocessing ops (`resample` /
/// `dc_offset_remove` / `pre_emphasis`).
///
/// # Errors
///
/// Returns [`VokraError::UnsupportedOp`] for ops outside that set and
/// [`VokraError::InvalidArgument`] for arity / kind / shape mismatches (plus any
/// error from the underlying op).
pub fn dispatch(op: &OpKind, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    match op {
        OpKind::Stft(attrs) => run_stft(attrs, inputs),
        OpKind::Istft(attrs) => run_istft(attrs, inputs),
        OpKind::IstftStreaming(attrs) => run_istft_streaming(attrs, inputs),
        OpKind::MelFilterbank(attrs) => run_mel(attrs, inputs),
        OpKind::Mfcc(attrs) => run_mfcc(attrs, inputs),
        OpKind::Dct(attrs) => run_dct(attrs, inputs),
        OpKind::Resample(attrs) => run_resample(attrs, inputs),
        OpKind::DcOffsetRemove => run_dc_offset_remove(inputs),
        OpKind::PreEmphasis(attrs) => run_pre_emphasis(attrs, inputs),
        OpKind::LengthConditioning(attrs) => run_length_conditioning(attrs, inputs),
        other => Err(VokraError::UnsupportedOp(format!(
            "{other:?} is not a front-end / preprocessing op"
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

/// Streaming iSTFT as a graph node (FR-OP-02): a one-shot evaluation over the
/// whole `[frames, bins]` chunk. The overlap tail is created and consumed inside
/// this call, so it is never a graph tensor (FR-ST-05); the result equals a
/// batch [`istft`] with the same parameters.
fn run_istft_streaming(attrs: &IstftStreamingAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "istft_streaming")?;
    let (shape, re, im) = inputs[0].as_complex().ok_or_else(|| {
        VokraError::InvalidArgument("istft_streaming: expected a complex input".to_owned())
    })?;
    if shape.len() != 2 {
        return Err(VokraError::InvalidArgument(
            "istft_streaming: input must be [frames, bins]".to_owned(),
        ));
    }
    let spec = Spectrogram {
        frames: shape[0],
        bins: shape[1],
        re: re.to_vec(),
        im: im.to_vec(),
    };
    let signal = istft_streaming_oneshot(&spec, attrs)?;
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

fn run_resample(attrs: &ResampleAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "resample")?;
    let (_, signal) = expect_real(&inputs[0], "resample")?;
    let out = resample(signal, attrs.in_rate, attrs.out_rate, attrs.quality)?;
    Ok(vec![OpValue::real(vec![out.len()], out)])
}

fn run_dc_offset_remove(inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "dc_offset_remove")?;
    let (_, signal) = expect_real(&inputs[0], "dc_offset_remove")?;
    let out = dc_offset_remove(signal);
    Ok(vec![OpValue::real(vec![out.len()], out)])
}

fn run_pre_emphasis(attrs: &PreEmphasisAttrs, inputs: &[OpValue]) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 1, "pre_emphasis")?;
    let (_, signal) = expect_real(&inputs[0], "pre_emphasis")?;
    let out = pre_emphasis(signal, attrs.coeff);
    Ok(vec![OpValue::real(vec![out.len()], out)])
}

/// `length_conditioning` graph node (FR-OP-71, M3-08).
///
/// The op takes **no runtime inputs** — the target duration lives in the
/// attrs, and mode B's `ref_speech_frames` is a caller-supplied scalar
/// baked into the graph the same way the frontend_spec-driven `sample_rate`
/// / `hop_length` are (FR-EX-10 精神 — sampler-adjacent metadata is not a
/// graph tensor). Output is the target frame count encoded as a length-1
/// real tensor with the `u32` cast to `f32`; downstream consumers (M3-09
/// CosyVoice2's Flow Matching sampler) turn it back into an integer.
fn run_length_conditioning(
    attrs: &LengthConditioningAttrs,
    inputs: &[OpValue],
) -> Result<Vec<OpValue>> {
    expect_arity(inputs, 0, "length_conditioning")?;
    let frames = length_conditioning(attrs)?;
    // f32 exactly represents every u32 the round_to_u32 gate lets through
    // (values < 2^24 are exact; the gate rejects the >= 2^32 range and
    // 2^24 .. 2^32 rounds to the same u32 that composes with u32::from(f as u32)
    // downstream). This is the safe encoding — a scalar tensor is what
    // graph consumers see today (OpValue has no integer variant in M0).
    Ok(vec![OpValue::real(vec![1], vec![frames as f32])])
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
    fn istft_streaming_dispatch_equals_batch_istft() {
        // The IstftStreaming graph node (one-shot over the whole chunk) must
        // equal both a direct istft_streaming_oneshot call and the batch istft
        // (bit-for-bit) — the "graph path == direct call" contract (M2-05-T10),
        // with the tail state never appearing as a graph tensor.
        use vokra_core::ir::graph::IstftStreamingAttrs;

        let signal: Vec<f32> = (0..4096).map(|t| (t as f32 * 0.03).sin()).collect();
        let sa = StftAttrs::new(512, 128);
        let spec = dispatch(
            &OpKind::Stft(sa),
            &[OpValue::real(vec![signal.len()], signal.clone())],
        )
        .unwrap();

        let mut ia = IstftAttrs::new(512, 128);
        ia.length = Some(signal.len());
        let attrs = IstftStreamingAttrs::from_istft(ia.clone());

        let via_graph = dispatch(&OpKind::IstftStreaming(attrs), &spec).unwrap();
        let (_, streamed) = via_graph[0].as_real().unwrap();

        let (shape, re, im) = spec[0].as_complex().unwrap();
        let recon = Spectrogram {
            frames: shape[0],
            bins: shape[1],
            re: re.to_vec(),
            im: im.to_vec(),
        };
        let batch = istft(&recon, &ia).unwrap();
        assert_eq!(
            streamed,
            batch.as_slice(),
            "graph-path streaming must equal batch"
        );
    }

    #[test]
    fn istft_streaming_dispatch_rejects_real_and_rank() {
        use vokra_core::ir::graph::IstftStreamingAttrs;
        let attrs = IstftStreamingAttrs::new(256, 128);
        // Wants complex, given real.
        let e = dispatch(
            &OpKind::IstftStreaming(attrs.clone()),
            &[OpValue::real(vec![4], vec![0.0; 4])],
        )
        .unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)), "real: {e:?}");
        // Complex input must be 2-D.
        let one_d = OpValue::Complex {
            shape: vec![4],
            re: vec![0.0; 4],
            im: vec![0.0; 4],
        };
        let e = dispatch(&OpKind::IstftStreaming(attrs), &[one_d]).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)), "rank: {e:?}");
    }

    #[test]
    fn non_m0_04_op_is_unsupported() {
        let err = dispatch(&OpKind::MatMul, &[]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    #[test]
    fn preprocess_ops_dispatch_match_direct_calls() {
        use vokra_core::ir::graph::{PreEmphasisAttrs, ResampleAttrs};

        let sig: Vec<f32> = (0..128).map(|i| 0.3 + (i as f32 * 0.05).sin()).collect();

        // dc_offset_remove: no attrs.
        let out = dispatch(
            &OpKind::DcOffsetRemove,
            &[OpValue::real(vec![sig.len()], sig.clone())],
        )
        .unwrap();
        assert_eq!(
            out[0].as_real().unwrap().1,
            crate::dc_offset_remove(&sig).as_slice()
        );

        // pre_emphasis: coeff attr.
        let out = dispatch(
            &OpKind::PreEmphasis(PreEmphasisAttrs { coeff: 0.97 }),
            &[OpValue::real(vec![sig.len()], sig.clone())],
        )
        .unwrap();
        assert_eq!(
            out[0].as_real().unwrap().1,
            crate::pre_emphasis(&sig, 0.97).as_slice()
        );

        // resample: rate-change attrs.
        let attrs = ResampleAttrs {
            in_rate: 16_000,
            out_rate: 8_000,
            quality: crate::resample::DEFAULT_QUALITY,
        };
        let out = dispatch(
            &OpKind::Resample(attrs),
            &[OpValue::real(vec![sig.len()], sig.clone())],
        )
        .unwrap();
        let want = crate::resample(&sig, 16_000, 8_000, crate::resample::DEFAULT_QUALITY).unwrap();
        assert_eq!(out[0].as_real().unwrap().1, want.as_slice());
    }

    #[test]
    fn preprocess_ops_reject_wrong_arity_and_kind() {
        use vokra_core::ir::graph::PreEmphasisAttrs;

        // Arity: dc_offset_remove needs exactly one input.
        assert!(matches!(
            dispatch(&OpKind::DcOffsetRemove, &[]),
            Err(VokraError::InvalidArgument(_))
        ));
        // Kind: a complex input where a real signal is required.
        let cplx = OpValue::Complex {
            shape: vec![1, 2],
            re: vec![0.0; 2],
            im: vec![0.0; 2],
        };
        assert!(matches!(
            dispatch(
                &OpKind::PreEmphasis(PreEmphasisAttrs { coeff: 0.5 }),
                &[cplx]
            ),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn dispatch_rejects_arity_kind_shape_and_bounds() {
        use vokra_core::ir::graph::{DctAttrs, IstftAttrs, Normalization};

        // Arity: stft expects exactly one input, given zero.
        let e = dispatch(&OpKind::Stft(StftAttrs::new(256, 128)), &[]).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)), "arity: {e:?}");

        // Wrong value-kind: stft wants a real input, given complex.
        let cplx = OpValue::Complex {
            shape: vec![1, 3],
            re: vec![0.0; 3],
            im: vec![0.0; 3],
        };
        let e = dispatch(&OpKind::Stft(StftAttrs::new(256, 128)), &[cplx]).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)), "kind: {e:?}");

        // istft wants a complex input, given real.
        let e = dispatch(
            &OpKind::Istft(IstftAttrs::new(256, 128)),
            &[OpValue::real(vec![4], vec![0.0; 4])],
        )
        .unwrap_err();
        assert!(
            matches!(e, VokraError::InvalidArgument(_)),
            "istft real: {e:?}"
        );

        // istft complex input must be 2-D [frames, bins]; give it 1-D.
        let one_d = OpValue::Complex {
            shape: vec![4],
            re: vec![0.0; 4],
            im: vec![0.0; 4],
        };
        let e = dispatch(&OpKind::Istft(IstftAttrs::new(256, 128)), &[one_d]).unwrap_err();
        assert!(
            matches!(e, VokraError::InvalidArgument(_)),
            "istft rank: {e:?}"
        );

        // mel_filterbank: freq-bin count must equal n_fft/2+1 (= 201 here).
        let mel = MelAttrs::new(16000, 400, 40);
        let wrong = OpValue::real(vec![2, 100], vec![0.0; 200]);
        let e = dispatch(&OpKind::MelFilterbank(mel), &[wrong]).unwrap_err();
        assert!(
            matches!(e, VokraError::InvalidArgument(_)),
            "mel freqs: {e:?}"
        );

        // dct: n_out must not exceed the transform length n.
        let attrs = DctAttrs {
            n_out: Some(5),
            normalization: Normalization::Ortho,
        };
        let e = dispatch(
            &OpKind::Dct(attrs),
            &[OpValue::real(vec![2, 4], vec![0.0; 8])],
        )
        .unwrap_err();
        assert!(
            matches!(e, VokraError::InvalidArgument(_)),
            "dct n_out: {e:?}"
        );
    }
}
