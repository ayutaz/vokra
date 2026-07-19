//! GGUF weight binding for the Silero VAD v5 subgraph (M0-05-T03).
//!
//! # Layer ↔ GGUF tensor map (source: upstream `silero_vad.onnx`, inspected)
//!
//! Silero v5 is really **two** networks with identical topology but
//! independently trained weights — one per sample rate — selected at the top of
//! the ONNX graph by `If(sr == 16000)`. Every one of the 15 weight tensors
//! **differs in value** between the two branches (not just the two rate-shaped
//! ones); see the crate SPEC. This module therefore binds a *per-rate* weight
//! set. The tensor names below are the upstream PyTorch parameter names, carried
//! through conversion:
//!
//! | subgraph stage                     | GGUF tensor (per rate)             | shape 16 k / 8 k |
//! |------------------------------------|------------------------------------|------------------|
//! | pseudo-STFT `Conv1d`               | `stft.forward_basis_buffer`        | `[258,1,256]` / `[130,1,128]` |
//! | encoder 0 `Conv1d`+ReLU            | `encoder.0.reparam_conv.{weight,bias}` | `[128,129,3]`,`[128]` / `[128,65,3]`,`[128]` |
//! | encoder 1 `Conv1d`(s2)+ReLU        | `encoder.1.reparam_conv.{weight,bias}` | `[64,128,3]`,`[64]` |
//! | encoder 2 `Conv1d`(s2)+ReLU        | `encoder.2.reparam_conv.{weight,bias}` | `[64,64,3]`,`[64]` |
//! | encoder 3 `Conv1d`+ReLU            | `encoder.3.reparam_conv.{weight,bias}` | `[128,64,3]`,`[128]` |
//! | LSTM(128,128) cell                 | `decoder.rnn.{weight_ih,weight_hh,bias_ih,bias_hh}` | `[512,128]`×2,`[512]`×2 |
//! | head `Conv1d`(k=1)+Sigmoid         | `decoder.decoder.2.{weight,bias}`  | `[1,128,1]`,`[1]` |
//!
//! # GGUF naming schemes accepted
//!
//! * **corrected (both rates)** — tensors are prefixed `sr8k.` / `sr16k.`; this
//!   is what the fixture `tests/parity/silero_vad/silero-vad-v5.gguf` uses and
//!   what a fixed `vokra-convert` should emit (see the crate SPEC "known
//!   conversion gap");
//! * **legacy (single rate)** — bare parameter names, as the current
//!   `vokra-convert` emits: its branch de-dup keeps only one rate, inferred here
//!   from the `stft.forward_basis_buffer` kernel length.

use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

use super::SampleRate;

/// A bound `Conv1d` weight (`[c_out, c_in, k]` row-major) with its bias.
pub(super) struct Conv1dW {
    pub(super) weight: Vec<f32>,
    /// The same weight transposed to `[c_in·k, c_out]` (tap-major, output
    /// channel fastest), built once at load (M5-14 Wave-2 T21). The conv hot
    /// loop iterates taps outer / output channels inner over contiguous
    /// `weight_t` rows — the auto-vectorizable, **bit-identical** formulation
    /// of the original per-channel scalar reduction (`math::conv1d_wt`).
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) c_out: usize,
    pub(super) c_in: usize,
    pub(super) k: usize,
}

impl Conv1dW {
    /// `[c_out, c_in, k]` → `[c_in·k, c_out]` (see `weight_t`).
    fn transpose(weight: &[f32], c_out: usize, c_in: usize, k: usize) -> Vec<f32> {
        let taps = c_in * k;
        let mut t = vec![0.0f32; taps * c_out];
        for co in 0..c_out {
            for tap in 0..taps {
                t[tap * c_out + co] = weight[co * taps + tap];
            }
        }
        t
    }
}

/// The full weight set for one sample rate (one ONNX `If` branch).
pub(super) struct RateWeights {
    /// Pseudo-STFT basis as a `Conv1d(1, 2*bins, k)` (no bias).
    pub(super) stft: Conv1dW,
    /// Encoder stack: conv0..conv3 (strides 1,2,2,1; each followed by ReLU).
    pub(super) encoder: [Conv1dW; 4],
    /// LSTM input weights `[4*128, 128]` (PyTorch `ifgo` gate order).
    pub(super) lstm_wih: Vec<f32>,
    /// LSTM recurrent weights `[4*128, 128]`.
    pub(super) lstm_whh: Vec<f32>,
    /// LSTM input bias `[4*128]`.
    pub(super) lstm_bih: Vec<f32>,
    /// LSTM recurrent bias `[4*128]`.
    pub(super) lstm_bhh: Vec<f32>,
    /// Output head `Conv1d(128, 1, k=1)` before the sigmoid.
    pub(super) head: Conv1dW,
}

/// LSTM hidden width (Silero v5).
pub(super) const HIDDEN: usize = 128;

/// Weights for whichever sample rate(s) the GGUF carries.
pub(super) struct SileroWeights {
    pub(super) r8k: Option<RateWeights>,
    pub(super) r16k: Option<RateWeights>,
}

impl SileroWeights {
    /// Binds Silero VAD v5 weights from a parsed GGUF (FR-LD-01).
    ///
    /// Accepts the corrected both-rate naming (`sr8k.` / `sr16k.` prefixes) and
    /// falls back to the legacy single-rate bare naming. Missing tensors, wrong
    /// shapes or non-`F32` dtypes are reported as [`VokraError::ModelLoad`].
    pub(super) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut r8k = None;
        let mut r16k = None;

        // Corrected scheme: rate-prefixed tensors.
        for rate in [SampleRate::Hz8000, SampleRate::Hz16000] {
            let key = format!("{}.stft.forward_basis_buffer", rate.gguf_prefix());
            if gguf.tensor_info(&key).is_some() {
                let w = load_rate(gguf, rate, Some(rate.gguf_prefix()))?;
                match rate {
                    SampleRate::Hz8000 => r8k = Some(w),
                    SampleRate::Hz16000 => r16k = Some(w),
                }
            }
        }

        // Legacy scheme: bare names; infer the single rate from the stft kernel.
        if r8k.is_none() && r16k.is_none() {
            if let Some(info) = gguf.tensor_info("stft.forward_basis_buffer") {
                let k = *info.dimensions.last().ok_or_else(|| {
                    VokraError::ModelLoad("stft.forward_basis_buffer has no dims".into())
                })?;
                let rate = match k {
                    256 => SampleRate::Hz16000,
                    128 => SampleRate::Hz8000,
                    other => {
                        return Err(VokraError::ModelLoad(format!(
                            "stft.forward_basis_buffer kernel {other} matches no sample rate"
                        )));
                    }
                };
                let w = load_rate(gguf, rate, None)?;
                match rate {
                    SampleRate::Hz8000 => r8k = Some(w),
                    SampleRate::Hz16000 => r16k = Some(w),
                }
            }
        }

        if r8k.is_none() && r16k.is_none() {
            return Err(VokraError::ModelLoad(
                "GGUF carries no Silero VAD weights (no stft.forward_basis_buffer)".into(),
            ));
        }
        Ok(Self { r8k, r16k })
    }

    /// Returns the weight set for `rate`, or `None` if the GGUF lacks it.
    pub(super) fn rate(&self, rate: SampleRate) -> Option<&RateWeights> {
        match rate {
            SampleRate::Hz8000 => self.r8k.as_ref(),
            SampleRate::Hz16000 => self.r16k.as_ref(),
        }
    }
}

/// Binds every tensor for one rate, validating each shape against the rate.
fn load_rate(gguf: &GgufFile, rate: SampleRate, prefix: Option<&str>) -> Result<RateWeights> {
    let name = |p: &str| match prefix {
        Some(pre) => format!("{pre}.{p}"),
        None => p.to_owned(),
    };
    let bins = rate.bins(); // 65 or 129
    let k = rate.n_fft(); // 128 or 256
    let stft = conv(gguf, &name("stft.forward_basis_buffer"), 2 * bins, 1, k)?;
    let encoder = [
        conv(gguf, &name("encoder.0.reparam_conv.weight"), 128, bins, 3)
            .and_then(|w| with_bias(gguf, w, &name("encoder.0.reparam_conv.bias")))?,
        conv(gguf, &name("encoder.1.reparam_conv.weight"), 64, 128, 3)
            .and_then(|w| with_bias(gguf, w, &name("encoder.1.reparam_conv.bias")))?,
        conv(gguf, &name("encoder.2.reparam_conv.weight"), 64, 64, 3)
            .and_then(|w| with_bias(gguf, w, &name("encoder.2.reparam_conv.bias")))?,
        conv(gguf, &name("encoder.3.reparam_conv.weight"), 128, 64, 3)
            .and_then(|w| with_bias(gguf, w, &name("encoder.3.reparam_conv.bias")))?,
    ];
    let lstm_wih = vec1d(gguf, &name("decoder.rnn.weight_ih"), &[4 * HIDDEN, HIDDEN])?;
    let lstm_whh = vec1d(gguf, &name("decoder.rnn.weight_hh"), &[4 * HIDDEN, HIDDEN])?;
    let lstm_bih = vec1d(gguf, &name("decoder.rnn.bias_ih"), &[4 * HIDDEN])?;
    let lstm_bhh = vec1d(gguf, &name("decoder.rnn.bias_hh"), &[4 * HIDDEN])?;
    let head = conv(gguf, &name("decoder.decoder.2.weight"), 1, HIDDEN, 1)
        .and_then(|w| with_bias(gguf, w, &name("decoder.decoder.2.bias")))?;

    Ok(RateWeights {
        stft,
        encoder,
        lstm_wih,
        lstm_whh,
        lstm_bih,
        lstm_bhh,
        head,
    })
}

/// Loads a `Conv1d` weight (no bias yet) validating its `[c_out, c_in, k]` shape.
fn conv(gguf: &GgufFile, name: &str, c_out: usize, c_in: usize, k: usize) -> Result<Conv1dW> {
    let weight = vec1d(gguf, name, &[c_out, c_in, k])?;
    let weight_t = Conv1dW::transpose(&weight, c_out, c_in, k);
    Ok(Conv1dW {
        weight,
        weight_t,
        bias: Vec::new(),
        c_out,
        c_in,
        k,
    })
}

/// Attaches the bias `[c_out]` to a previously loaded conv weight.
fn with_bias(gguf: &GgufFile, mut w: Conv1dW, bias_name: &str) -> Result<Conv1dW> {
    w.bias = vec1d(gguf, bias_name, &[w.c_out])?;
    Ok(w)
}

#[cfg(test)]
impl RateWeights {
    /// All-zero weights of the correct per-rate shapes, for shape/plumbing tests
    /// that must not depend on the committed GGUF fixture.
    pub(super) fn zeros_for_test(rate: SampleRate) -> Self {
        let bins = rate.bins();
        let conv = |c_out: usize, c_in: usize, k: usize| Conv1dW {
            weight: vec![0.0; c_out * c_in * k],
            weight_t: vec![0.0; c_out * c_in * k],
            bias: vec![0.0; c_out],
            c_out,
            c_in,
            k,
        };
        Self {
            stft: Conv1dW {
                weight: vec![0.0; 2 * bins * rate.n_fft()],
                weight_t: vec![0.0; 2 * bins * rate.n_fft()],
                bias: Vec::new(),
                c_out: 2 * bins,
                c_in: 1,
                k: rate.n_fft(),
            },
            encoder: [
                conv(128, bins, 3),
                conv(64, 128, 3),
                conv(64, 64, 3),
                conv(128, 64, 3),
            ],
            lstm_wih: vec![0.0; 4 * HIDDEN * HIDDEN],
            lstm_whh: vec![0.0; 4 * HIDDEN * HIDDEN],
            lstm_bih: vec![0.0; 4 * HIDDEN],
            lstm_bhh: vec![0.0; 4 * HIDDEN],
            head: conv(1, HIDDEN, 1),
        }
    }
}

/// Reads a tensor's payload as `Vec<f32>`, checking presence, dtype and shape.
fn vec1d(gguf: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("missing tensor `{name}`")))?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "tensor `{name}` has dtype {:?}, expected F32",
            info.dtype
        )));
    }
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != expected {
        return Err(VokraError::ModelLoad(format!(
            "tensor `{name}` shape {got:?}, expected {expected:?}"
        )));
    }
    let bytes = gguf
        .tensor_data(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("no data for tensor `{name}`")))?;
    if bytes.len() % 4 != 0 {
        return Err(VokraError::ModelLoad(format!(
            "tensor `{name}` byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::engines::VadEngine;
    use vokra_core::gguf::GgufBuilder;

    use crate::silero_vad::SileroVadV5;

    /// Queues an all-zero `F32` tensor of the given logical shape.
    fn add_zeros(b: &mut GgufBuilder, name: &str, dims: &[u64]) {
        let n: u64 = dims.iter().product();
        b.add_tensor(
            name,
            GgmlType::F32,
            dims.to_vec(),
            vec![0u8; n as usize * 4],
        )
        .expect("add zero tensor");
    }

    /// Adds the correctly-shaped 8 kHz `stft.forward_basis_buffer` (`[130,1,128]`)
    /// under `prefix`, so `from_gguf` enters `load_rate` on the 8 kHz branch
    /// (2*bins = 130, kernel = 128 -> 8 kHz).
    fn add_stft_8k(b: &mut GgufBuilder, prefix: &str) {
        add_zeros(
            b,
            &format!("{prefix}stft.forward_basis_buffer"),
            &[130, 1, 128],
        );
    }

    /// Adds all 15 correctly-shaped 8 kHz weight tensors (all zeros) under
    /// `prefix` (mirrors `RateWeights::zeros_for_test` shapes; bins = 65).
    fn add_all_8k(b: &mut GgufBuilder, prefix: &str) {
        add_stft_8k(b, prefix);
        for (suffix, dims) in [
            ("encoder.0.reparam_conv.weight", &[128, 65, 3][..]),
            ("encoder.0.reparam_conv.bias", &[128][..]),
            ("encoder.1.reparam_conv.weight", &[64, 128, 3][..]),
            ("encoder.1.reparam_conv.bias", &[64][..]),
            ("encoder.2.reparam_conv.weight", &[64, 64, 3][..]),
            ("encoder.2.reparam_conv.bias", &[64][..]),
            ("encoder.3.reparam_conv.weight", &[128, 64, 3][..]),
            ("encoder.3.reparam_conv.bias", &[128][..]),
            ("decoder.rnn.weight_ih", &[512, 128][..]),
            ("decoder.rnn.weight_hh", &[512, 128][..]),
            ("decoder.rnn.bias_ih", &[512][..]),
            ("decoder.rnn.bias_hh", &[512][..]),
            ("decoder.decoder.2.weight", &[1, 128, 1][..]),
            ("decoder.decoder.2.bias", &[1][..]),
        ] {
            let dims: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
            add_zeros(b, &format!("{prefix}{suffix}"), &dims);
        }
    }

    fn to_gguf(b: &GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().expect("serialize gguf")).expect("parse gguf")
    }

    // ---- weights-validation-error-paths (per-tensor presence/shape/dtype) ----

    #[test]
    fn from_gguf_rejects_missing_encoder_tensor() {
        // stft present (enters load_rate) but encoder.0 weight absent.
        let mut b = GgufBuilder::new();
        add_stft_8k(&mut b, "sr8k.");
        assert!(matches!(
            SileroVadV5::from_gguf(&to_gguf(&b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn from_gguf_rejects_wrong_encoder_shape() {
        // Correct 8 kHz encoder.0 shape is [128,65,3]; [128,64,3] is wrong.
        let mut b = GgufBuilder::new();
        add_stft_8k(&mut b, "sr8k.");
        add_zeros(&mut b, "sr8k.encoder.0.reparam_conv.weight", &[128, 64, 3]);
        assert!(matches!(
            SileroVadV5::from_gguf(&to_gguf(&b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn from_gguf_rejects_non_f32_dtype() {
        // Correct shape [128,65,3] but F16 dtype -> dtype check fires first.
        let mut b = GgufBuilder::new();
        add_stft_8k(&mut b, "sr8k.");
        let n: usize = 128 * 65 * 3;
        b.add_tensor(
            "sr8k.encoder.0.reparam_conv.weight",
            GgmlType::F16,
            vec![128, 65, 3],
            vec![0u8; n * 2],
        )
        .expect("add f16 tensor");
        assert!(matches!(
            SileroVadV5::from_gguf(&to_gguf(&b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    // ---- weights-legacy-single-rate (bare names + kernel->rate inference) ----

    #[test]
    fn loads_legacy_bare_name_single_rate_8k() {
        // Bare (un-prefixed) names -> legacy scheme; kernel 128 -> 8 kHz only.
        let mut b = GgufBuilder::new();
        add_all_8k(&mut b, "");
        let m = SileroVadV5::from_gguf(&to_gguf(&b)).expect("legacy 8 kHz model loads");

        assert!(m.supports(SampleRate::Hz8000));
        assert!(!m.supports(SampleRate::Hz16000));

        // The absent rate is rejected at both entry points.
        assert!(matches!(
            m.forward_chunk(SampleRate::Hz16000, &[0.0; 512]),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut s16 = m.open_stream();
        assert!(s16.push_pcm(&[0.0; 512], 16000).is_err());

        // The present rate is usable: a 256-sample frame yields one probability.
        let mut s8 = m.open_stream();
        assert_eq!(s8.push_pcm(&[0.0; 256], 8000).unwrap().len(), 1);
    }

    #[test]
    fn legacy_unknown_stft_kernel_is_rejected() {
        // A bare stft buffer whose kernel length matches neither rate.
        let mut b = GgufBuilder::new();
        add_zeros(&mut b, "stft.forward_basis_buffer", &[1, 1, 200]);
        let r = SileroVadV5::from_gguf(&to_gguf(&b));
        assert!(
            matches!(&r, Err(VokraError::ModelLoad(m)) if m.contains("matches no sample rate")),
            "expected `matches no sample rate` ModelLoad"
        );
    }
}
