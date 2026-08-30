//! Native GigaAM v3 RNNT route.
//!
//! The graph and decoding loop are transcribed from the authenticated
//! `ai-sage/GigaAM-v3` remote code at revision
//! `ec1dc1f01d0d627ab2c0d3acc1e235702300d95e`. Binding remains fail-closed
//! against any prepared artifact other than the independently reviewed digest.

use crate::compute::{Compute, HotOp};
use std::collections::BTreeSet;
use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::{Result, VokraError};
use vokra_ops::conformer::{
    ConformerCompute, ConformerConfig, ConformerConvWeights, ConformerEncoder,
    ConformerLayerWeights, ConformerSubsampleWeights, ConformerWeights, ConvSubsampleKind,
    FeedForwardWeights, MhaWeights, PositionEncoding,
};

/// Complete learned operation set for the GigaAM v3 RNNT graph.
pub const GIGAAM_V3_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Silu,
    HotOp::Relu,
    HotOp::Tanh,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

/// GGUF architecture marker for the v3 RNNT route.
pub const ARCH: &str = "sber_gigaam_v3";
/// GGUF model name marker for the v3 RNNT route.
pub const NAME: &str = "gigaam-v3";
/// Fixed HF model revision.
pub const HF_REVISION: &str = "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e";
/// Fixed upstream source revision.
pub const SOURCE_REVISION: &str = "7447938d791c4f3e643386ee22c33777004293a5";
/// Fixed checkpoint SHA-256.
pub const CHECKPOINT_SHA256: &str =
    "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a";
/// Fixed HF remote-code source SHA-256.
pub const MODELING_SHA256: &str =
    "269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86";
/// Fixed config SHA-256.
pub const CONFIG_SHA256: &str = "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52";
/// Fixed SentencePiece tokenizer SHA-256.
pub const TOKENIZER_SHA256: &str =
    "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a";
/// Number of checkpoint tensors.
pub const TENSOR_COUNT: usize = 561;
/// SentencePiece vocabulary size authenticated by the inspection manifest.
pub const VOCAB_SIZE: usize = 1024;
/// RNNT blank class (`num_classes - 1` in the fixed remote code).
pub const BLANK_ID: usize = 1024;
/// RNNT output class count.
pub const NUM_CLASSES: usize = 1025;
/// Official greedy decoder symbol bound per encoder frame.
pub const MAX_SYMBOLS_PER_STEP: usize = 10;
/// Required audio sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Filled only after independent VAST review of the prepared artifact.
pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> =
    Some("cee04765f031d6ee5088849ecb0e5c1db4e58ca28a345ce4d049015cd683a64e");

type LstmState = (Vec<f32>, Vec<f32>);

/// Diagnostic output from one native greedy RNNT pass.
///
/// `log_mel` is row-major `[mel_frames, 64]`, `encoded` is row-major
/// `[encoded_frames, 768]`, and `rnnt_logits` is row-major
/// `[decisions, 1025]` containing the official joint's stable log-softmax
/// output. Decision metadata records the frame and symbol-at-frame for each
/// row. This is an observation surface for independent parity; it does not
/// enable text decoding or weaken the prepared-artifact gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GigaamV3Trace {
    /// Frontend log-mel values, shaped `[mel_frames, 64]`.
    pub log_mel: Vec<f32>,
    /// Number of frontend frames.
    pub mel_frames: usize,
    /// Encoder output values, shaped `[encoded_frames, 768]`.
    pub encoded: Vec<f32>,
    /// Number of valid encoder frames.
    pub encoded_frames: usize,
    /// Joint log-softmax rows, shaped `[decisions, 1025]`.
    pub rnnt_logits: Vec<f32>,
    /// Encoder frame for every joint row.
    pub decision_frames: Vec<usize>,
    /// Symbol index within the encoder frame for every joint row.
    pub decision_symbols: Vec<usize>,
    /// Argmax class for every joint row.
    pub decision_argmax: Vec<u32>,
    /// Emitted non-blank token IDs in greedy order.
    pub token_ids: Vec<u32>,
}

#[cfg(test)]
fn row_log_softmax(row: &mut [f32]) -> Result<usize> {
    let mut max = f32::NEG_INFINITY;
    let mut argmax = 0;
    for (index, &value) in row.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::ModelLoad(
                "GigaAM v3 joint produced non-finite logits".into(),
            ));
        }
        if value > max {
            max = value;
            argmax = index;
        }
    }
    let sum = row.iter().map(|value| (*value - max).exp()).sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(VokraError::ModelLoad(
            "GigaAM v3 joint log-softmax normalization failed".into(),
        ));
    }
    let log_sum = max + sum.ln();
    for value in row {
        *value -= log_sum;
    }
    Ok(argmax)
}

#[cfg(test)]
fn greedy_decode_factored(
    frames: &[Vec<Vec<f32>>],
    max_symbols_per_step: usize,
) -> Result<(Vec<u32>, Vec<usize>)> {
    if max_symbols_per_step == 0 {
        return Err(VokraError::InvalidArgument(
            "GigaAM v3 max_symbols_per_step must be nonzero".into(),
        ));
    }
    let mut tokens = Vec::new();
    let mut committed = Vec::new();
    for frame in frames {
        let mut symbols = 0;
        for row in frame.iter().take(max_symbols_per_step) {
            let mut row = row.clone();
            let argmax = row_log_softmax(&mut row)?;
            if argmax == BLANK_ID {
                break;
            }
            tokens.push(argmax as u32);
            committed.push(argmax);
            symbols += 1;
            if symbols == max_symbols_per_step {
                break;
            }
        }
    }
    Ok((tokens, committed))
}

fn expected_manifest() -> Vec<(String, Vec<u64>, GgmlType)> {
    let mut out = vec![
        (
            "model.preprocessor.featurizer.0.spectrogram.window".into(),
            vec![320],
            GgmlType::F32,
        ),
        (
            "model.preprocessor.featurizer.0.mel_scale.fb".into(),
            vec![161, 64],
            GgmlType::F32,
        ),
        (
            "model.encoder.pre_encode.conv.0.weight".into(),
            vec![768, 64, 5],
            GgmlType::F16,
        ),
        (
            "model.encoder.pre_encode.conv.0.bias".into(),
            vec![768],
            GgmlType::F16,
        ),
        (
            "model.encoder.pre_encode.conv.2.weight".into(),
            vec![768, 768, 5],
            GgmlType::F16,
        ),
        (
            "model.encoder.pre_encode.conv.2.bias".into(),
            vec![768],
            GgmlType::F16,
        ),
    ];
    for layer in 0..16 {
        let p = format!("model.encoder.layers.{layer}");
        for n in [
            "norm_feed_forward1",
            "norm_conv",
            "norm_self_att",
            "norm_feed_forward2",
            "norm_out",
        ] {
            out.push((format!("{p}.{n}.weight"), vec![768], GgmlType::F16));
            out.push((format!("{p}.{n}.bias"), vec![768], GgmlType::F16));
        }
        for branch in ["feed_forward1", "feed_forward2"] {
            out.push((
                format!("{p}.{branch}.linear1.weight"),
                vec![3072, 768],
                GgmlType::F16,
            ));
            out.push((
                format!("{p}.{branch}.linear1.bias"),
                vec![3072],
                GgmlType::F16,
            ));
            out.push((
                format!("{p}.{branch}.linear2.weight"),
                vec![768, 3072],
                GgmlType::F16,
            ));
            out.push((
                format!("{p}.{branch}.linear2.bias"),
                vec![768],
                GgmlType::F16,
            ));
        }
        for (n, shape) in [
            ("pointwise_conv1.weight", vec![1536, 768, 1]),
            ("pointwise_conv1.bias", vec![1536]),
            ("depthwise_conv.weight", vec![768, 1, 5]),
            ("depthwise_conv.bias", vec![768]),
            ("batch_norm.weight", vec![768]),
            ("batch_norm.bias", vec![768]),
            ("pointwise_conv2.weight", vec![768, 768, 1]),
            ("pointwise_conv2.bias", vec![768]),
        ] {
            out.push((format!("{p}.conv.{n}"), shape, GgmlType::F16));
        }
        for n in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            out.push((
                format!("{p}.self_attn.{n}.weight"),
                vec![768, 768],
                GgmlType::F16,
            ));
            out.push((format!("{p}.self_attn.{n}.bias"), vec![768], GgmlType::F16));
        }
    }
    out.extend([
        (
            "model.head.decoder.embed.weight".into(),
            vec![1025, 320],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.weight_ih_l0".into(),
            vec![1280, 320],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.weight_hh_l0".into(),
            vec![1280, 320],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.bias_ih_l0".into(),
            vec![1280],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.bias_hh_l0".into(),
            vec![1280],
            GgmlType::F32,
        ),
        (
            "model.head.joint.pred.weight".into(),
            vec![320, 320],
            GgmlType::F32,
        ),
        (
            "model.head.joint.pred.bias".into(),
            vec![320],
            GgmlType::F32,
        ),
        (
            "model.head.joint.enc.weight".into(),
            vec![320, 768],
            GgmlType::F32,
        ),
        ("model.head.joint.enc.bias".into(), vec![320], GgmlType::F32),
        (
            "model.head.joint.joint_net.1.weight".into(),
            vec![1025, 320],
            GgmlType::F32,
        ),
        (
            "model.head.joint.joint_net.1.bias".into(),
            vec![1025],
            GgmlType::F32,
        ),
    ]);
    out
}

fn tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>> {
    file.tensor_f32(name)
        .map_err(|e| VokraError::ModelLoad(format!("GigaAM v3 tensor `{name}`: {e}")))
}

fn require_str(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    if file.get(key).and_then(|v| v.as_str()) != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "GigaAM v3 metadata `{key}` mismatch"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    if file.get(key).and_then(|v| v.as_u64()) != Some(expected as u64) {
        return Err(VokraError::ModelLoad(format!(
            "GigaAM v3 metadata `{key}` must be {expected}"
        )));
    }
    Ok(())
}

fn bind_ff(file: &GgufFile, prefix: &str) -> Result<FeedForwardWeights> {
    Ok(FeedForwardWeights {
        w1: tensor(file, &format!("{prefix}.linear1.weight"))?,
        b1: tensor(file, &format!("{prefix}.linear1.bias"))?,
        w2: tensor(file, &format!("{prefix}.linear2.weight"))?,
        b2: tensor(file, &format!("{prefix}.linear2.bias"))?,
    })
}

fn bind_norm(file: &GgufFile, prefix: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        tensor(file, &format!("{prefix}.weight"))?,
        tensor(file, &format!("{prefix}.bias"))?,
    ))
}

fn bind_layer(file: &GgufFile, index: usize) -> Result<ConformerLayerWeights> {
    let p = format!("model.encoder.layers.{index}");
    let (ln1_gamma, ln1_beta) = bind_norm(file, &format!("{p}.norm_feed_forward1"))?;
    let (ln2_gamma, ln2_beta) = bind_norm(file, &format!("{p}.norm_self_att"))?;
    let (ln3_gamma, ln3_beta) = bind_norm(file, &format!("{p}.norm_conv"))?;
    let (ln4_gamma, ln4_beta) = bind_norm(file, &format!("{p}.norm_feed_forward2"))?;
    let (ln_out_gamma, ln_out_beta) = bind_norm(file, &format!("{p}.norm_out"))?;
    Ok(ConformerLayerWeights {
        ln1_gamma,
        ln1_beta,
        ff1: bind_ff(file, &format!("{p}.feed_forward1"))?,
        ln2_gamma,
        ln2_beta,
        mha: MhaWeights {
            wq: tensor(file, &format!("{p}.self_attn.linear_q.weight"))?,
            bq: tensor(file, &format!("{p}.self_attn.linear_q.bias"))?,
            wk: tensor(file, &format!("{p}.self_attn.linear_k.weight"))?,
            bk: tensor(file, &format!("{p}.self_attn.linear_k.bias"))?,
            wv: tensor(file, &format!("{p}.self_attn.linear_v.weight"))?,
            bv: tensor(file, &format!("{p}.self_attn.linear_v.bias"))?,
            wo: tensor(file, &format!("{p}.self_attn.linear_out.weight"))?,
            bo: tensor(file, &format!("{p}.self_attn.linear_out.bias"))?,
        },
        ln3_gamma,
        ln3_beta,
        conv: ConformerConvWeights {
            pointwise1_w: tensor(file, &format!("{p}.conv.pointwise_conv1.weight"))?,
            pointwise1_b: tensor(file, &format!("{p}.conv.pointwise_conv1.bias"))?,
            depthwise_w: tensor(file, &format!("{p}.conv.depthwise_conv.weight"))?,
            depthwise_b: tensor(file, &format!("{p}.conv.depthwise_conv.bias"))?,
            norm_gamma: tensor(file, &format!("{p}.conv.batch_norm.weight"))?,
            norm_beta: tensor(file, &format!("{p}.conv.batch_norm.bias"))?,
            pointwise2_w: tensor(file, &format!("{p}.conv.pointwise_conv2.weight"))?,
            pointwise2_b: tensor(file, &format!("{p}.conv.pointwise_conv2.bias"))?,
        },
        ln4_gamma,
        ln4_beta,
        ff2: bind_ff(file, &format!("{p}.feed_forward2"))?,
        ln_out_gamma,
        ln_out_beta,
    })
}

/// Bound native GigaAM v3 RNNT model.
#[derive(Debug)]
pub struct GigaamV3 {
    encoder: ConformerEncoder,
    window: Vec<f32>,
    mel_filter: Vec<f32>,
    embed: Vec<f32>,
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
    pred_w: Vec<f32>,
    pred_b: Vec<f32>,
    enc_w: Vec<f32>,
    enc_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    backend: BackendKind,
}

impl GigaamV3 {
    /// Bind a strict GGUF. The prepared SHA gate intentionally rejects until
    /// VAST independently authenticates the prepared artifact.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_str(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_str(file, chunks::KEY_MODEL_NAME, NAME)?;
        let Some(prepared_sha) = AUTHENTICATED_PREPARED_SHA256 else {
            return Err(VokraError::ModelLoad(
                "GigaAM v3 prepared SHA-256 is not independently authenticated".into(),
            ));
        };
        for (k, v) in [
            ("sample_rate", 16000),
            ("n_mels", 64),
            ("n_fft", 320),
            ("hop_length", 160),
            ("win_length", 320),
            ("n_layers", 16),
            ("d_model", 768),
            ("n_heads", 16),
            ("ffn_dim", 3072),
            ("conv_kernel_size", 5),
            ("subsampling_kernel_size", 5),
            ("subsampling_stride", 2),
            ("subsampling_padding", 2),
            ("pred_hidden", 320),
            ("pred_rnn_layers", 1),
            ("joint_hidden", 320),
            ("vocab_size", 1025),
            ("blank_id", 1024),
        ] {
            require_u32(file, &format!("vokra.gigaam_v3.{k}"), v)?;
        }
        for (k, v) in [
            ("preprocessor_center", "false"),
            ("mel_scale", "htk"),
            ("mel_norm", "None"),
            ("power", "2"),
        ] {
            require_str(file, &format!("vokra.gigaam_v3.{k}"), v)?;
        }
        for (k, v) in [
            ("model_class", "rnnt"),
            ("model_name", "v3_e2e_rnnt"),
            ("topology", "RNNT"),
            ("revision", HF_REVISION),
            ("source_revision", SOURCE_REVISION),
            ("config_sha256", CONFIG_SHA256),
            ("checkpoint_sha256", CHECKPOINT_SHA256),
            ("modeling_sha256", MODELING_SHA256),
            ("tokenizer_sha256", TOKENIZER_SHA256),
            ("prepared_sha256", prepared_sha),
        ] {
            require_str(file, &format!("vokra.gigaam_v3.{k}"), v)?;
        }
        let expected = expected_manifest();
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "GigaAM v3 tensor count must be {TENSOR_COUNT}"
            )));
        }
        let expected_names: BTreeSet<&str> =
            expected.iter().map(|(name, _, _)| name.as_str()).collect();
        let actual_names: BTreeSet<&str> = file
            .tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        if expected_names != actual_names || actual_names.len() != file.tensors().len() {
            return Err(VokraError::ModelLoad(
                "GigaAM v3 tensor name set/duplicate mismatch".into(),
            ));
        }
        for (name, shape, dtype) in &expected {
            let info = file.tensor_info(name).ok_or_else(|| {
                VokraError::ModelLoad(format!("GigaAM v3 tensor `{name}` missing"))
            })?;
            if info.dtype != *dtype || info.dimensions.as_slice() != shape.as_slice() {
                return Err(VokraError::ModelLoad(format!(
                    "GigaAM v3 tensor `{name}` dtype/shape mismatch"
                )));
            }
        }
        let sub = ConformerSubsampleWeights {
            linear_w: Vec::new(),
            linear_b: Vec::new(),
            norm_gamma: None,
            norm_beta: None,
            conv1_w: Some(tensor(file, "model.encoder.pre_encode.conv.0.weight")?),
            conv1_b: Some(tensor(file, "model.encoder.pre_encode.conv.0.bias")?),
            conv2_w: Some(tensor(file, "model.encoder.pre_encode.conv.2.weight")?),
            conv2_b: Some(tensor(file, "model.encoder.pre_encode.conv.2.bias")?),
        };
        let config = ConformerConfig {
            in_dim: 64,
            d_model: 768,
            n_heads: 16,
            ffn_dim: 3072,
            n_layers: 16,
            kernel_size: 5,
            subsample_type: ConvSubsampleKind::Conv1d {
                kernel: 5,
                stride: 2,
                padding: 2,
            },
            position_encoding: PositionEncoding::GigaamRope {
                theta: 5000.0,
                max_len: 5000,
            },
        };
        let encoder = ConformerEncoder::new(
            config,
            ConformerWeights {
                subsample: sub,
                layers: (0..16)
                    .map(|i| bind_layer(file, i))
                    .collect::<Result<Vec<_>>>()?,
            },
        )?;
        Ok(Self {
            encoder,
            window: tensor(file, "model.preprocessor.featurizer.0.spectrogram.window")?,
            mel_filter: tensor(file, "model.preprocessor.featurizer.0.mel_scale.fb")?,
            embed: tensor(file, "model.head.decoder.embed.weight")?,
            weight_ih: tensor(file, "model.head.decoder.lstm.weight_ih_l0")?,
            weight_hh: tensor(file, "model.head.decoder.lstm.weight_hh_l0")?,
            bias_ih: tensor(file, "model.head.decoder.lstm.bias_ih_l0")?,
            bias_hh: tensor(file, "model.head.decoder.lstm.bias_hh_l0")?,
            pred_w: tensor(file, "model.head.joint.pred.weight")?,
            pred_b: tensor(file, "model.head.joint.pred.bias")?,
            enc_w: tensor(file, "model.head.joint.enc.weight")?,
            enc_b: tensor(file, "model.head.joint.enc.bias")?,
            out_w: tensor(file, "model.head.joint.joint_net.1.weight")?,
            out_b: tensor(file, "model.head.joint.joint_net.1.bias")?,
            backend: BackendKind::Cpu,
        })
    }

    /// Select only a backend that implements the complete learned graph.
    pub fn with_backend(mut self, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, GIGAAM_V3_HOT_OPS)?;
        self.backend = backend;
        Ok(self)
    }

    /// Return the selected backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Decode PCM to RNNT token IDs using the authenticated greedy algorithm.
    pub fn transcribe_token_ids(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        Ok(self.trace_pcm(pcm)?.token_ids)
    }

    /// Return frontend, encoder, and per-decision greedy RNNT diagnostics.
    pub fn trace_pcm(&self, pcm: &[f32]) -> Result<GigaamV3Trace> {
        let compute = Compute::for_backend(self.backend, GIGAAM_V3_HOT_OPS)?;
        let (mel, frames) = self.log_mel_from_pcm(pcm)?;
        let (hidden, encoded_frames) = self
            .encoder
            .forward_with_compute(&mel, frames, None, &compute)?;
        if hidden.len() != encoded_frames * 768 {
            return Err(VokraError::ModelLoad(
                "GigaAM v3 encoder output shape mismatch".into(),
            ));
        }
        let mut rnnt_logits = Vec::new();
        let mut decision_frames = Vec::new();
        let mut decision_symbols = Vec::new();
        let mut decision_argmax = Vec::new();
        let mut output = Vec::new();
        let mut state: Option<LstmState> = None;
        let mut last_label: Option<usize> = None;
        for t in 0..encoded_frames {
            let f = &hidden[t * 768..(t + 1) * 768];
            let mut symbols = 0;
            loop {
                if symbols >= MAX_SYMBOLS_PER_STEP {
                    break;
                }
                let (g, next_state) = self.predict(last_label, state.as_ref(), &compute)?;
                let mut row = self.joint_logits(f, &g, &compute)?;
                let mut log_probs = vec![0.0; row.len()];
                compute.log_softmax(&row, &mut log_probs, 1, NUM_CLASSES)?;
                let k = log_probs
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(index, _)| index)
                    .ok_or_else(|| VokraError::ModelLoad("GigaAM v3 empty joint row".into()))?;
                row.copy_from_slice(&log_probs);
                rnnt_logits.extend_from_slice(&row);
                decision_frames.push(t);
                decision_symbols.push(symbols);
                decision_argmax.push(k as u32);
                if k == BLANK_ID {
                    break;
                }
                output.push(k as u32);
                state = Some(next_state);
                last_label = Some(k);
                symbols += 1;
            }
        }
        Ok(GigaamV3Trace {
            log_mel: mel,
            mel_frames: frames,
            encoded: hidden,
            encoded_frames,
            rnnt_logits,
            decision_frames,
            decision_symbols,
            decision_argmax,
            token_ids: output,
        })
    }

    /// Text transcription is intentionally unsupported until the exact
    /// SentencePiece runtime is available in the dependency-free runtime.
    pub fn transcribe(&self, _pcm: &[f32]) -> Result<String> {
        Err(VokraError::UnsupportedOp(
            "GigaAM v3 SentencePiece text decode is not implemented; use transcribe_token_ids"
                .into(),
        ))
    }

    fn predict(
        &self,
        label: Option<usize>,
        state: Option<&LstmState>,
        compute: &impl ConformerCompute,
    ) -> Result<(Vec<f32>, LstmState)> {
        let mut input = vec![0.0; 320];
        if let Some(label) = label {
            let start = label.checked_mul(320).ok_or_else(|| {
                VokraError::ModelLoad("GigaAM v3 embedding offset overflow".into())
            })?;
            input.copy_from_slice(self.embed.get(start..start + 320).ok_or_else(|| {
                VokraError::ModelLoad("GigaAM v3 label is outside embedding".into())
            })?);
        }
        let (prev_h, prev_c) = state.map_or_else(
            || (vec![0.0; 320], vec![0.0; 320]),
            |(h, c)| (h.clone(), c.clone()),
        );
        let mut input_gates = vec![0.0; 1280];
        let mut recurrent_gates = vec![0.0; 1280];
        compute.linear_row(
            &input,
            1,
            320,
            &self.weight_ih,
            &self.bias_ih,
            1280,
            &mut input_gates,
        )?;
        compute.linear_row(
            &prev_h,
            1,
            320,
            &self.weight_hh,
            &self.bias_hh,
            1280,
            &mut recurrent_gates,
        )?;
        let mut gates = vec![0.0; 1280];
        for ((gate, input_gate), recurrent_gate) in
            gates.iter_mut().zip(input_gates).zip(recurrent_gates)
        {
            *gate = input_gate + recurrent_gate;
        }
        let mut h = vec![0.0; 320];
        let mut c = vec![0.0; 320];
        let mut input_gate_values = vec![0.0; 320];
        let mut forget_gate_values = vec![0.0; 320];
        let mut output_gate_values = vec![0.0; 320];
        compute.sigmoid(&gates[..320], &mut input_gate_values)?;
        compute.sigmoid(&gates[320..640], &mut forget_gate_values)?;
        compute.sigmoid(&gates[960..], &mut output_gate_values)?;
        let mut cell_gates = vec![0.0; 320];
        compute.tanh(&gates[640..960], &mut cell_gates)?;
        let mut cell_states = vec![0.0; 320];
        for (i, (h_value, c_value)) in h.iter_mut().zip(c.iter_mut()).enumerate() {
            let input_gate = input_gate_values[i];
            let forget_gate = forget_gate_values[i];
            let cell_gate = cell_gates[i];
            let output_gate = output_gate_values[i];
            *c_value = forget_gate * prev_c[i] + input_gate * cell_gate;
            cell_states[i] = *c_value;
            *h_value = output_gate;
        }
        let mut state_tanh = vec![0.0; 320];
        compute.tanh(&cell_states, &mut state_tanh)?;
        for (i, h_value) in h.iter_mut().enumerate() {
            *h_value *= state_tanh[i];
        }
        Ok((h.clone(), (h, c)))
    }

    fn joint_logits(
        &self,
        enc: &[f32],
        pred: &[f32],
        compute: &impl ConformerCompute,
    ) -> Result<Vec<f32>> {
        let mut hidden = vec![0.0; 320];
        let mut enc_hidden = vec![0.0; 320];
        let mut pred_hidden = vec![0.0; 320];
        compute.linear_row(enc, 1, 768, &self.enc_w, &self.enc_b, 320, &mut enc_hidden)?;
        compute.linear_row(
            pred,
            1,
            320,
            &self.pred_w,
            &self.pred_b,
            320,
            &mut pred_hidden,
        )?;
        for ((hidden_value, enc_value), pred_value) in
            hidden.iter_mut().zip(enc_hidden).zip(pred_hidden)
        {
            *hidden_value = enc_value + pred_value;
        }
        let pre_activation = hidden.clone();
        compute.relu(&pre_activation, &mut hidden)?;
        let mut logits = vec![0.0; NUM_CLASSES];
        compute.linear_row(
            &hidden,
            1,
            320,
            &self.out_w,
            &self.out_b,
            NUM_CLASSES,
            &mut logits,
        )?;
        Ok(logits)
    }

    fn log_mel_from_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        if pcm.len() < 320 {
            return Err(VokraError::InvalidArgument(
                "GigaAM v3 audio must contain at least one frame".into(),
            ));
        }
        if !pcm.iter().all(|v| v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "GigaAM v3 PCM contains non-finite values".into(),
            ));
        }
        let frames = (pcm.len() - 320) / 160 + 1;
        let mut out = vec![0.0; frames * 64];
        let mut power = vec![0.0; 161];
        for frame in 0..frames {
            for (freq, power_value) in power.iter_mut().enumerate() {
                let mut real = 0.0;
                let mut imag = 0.0;
                for sample in 0..320 {
                    let angle = 2.0 * core::f32::consts::PI * freq as f32 * sample as f32 / 320.0;
                    let value = pcm[frame * 160 + sample] * self.window[sample];
                    real += value * angle.cos();
                    imag -= value * angle.sin();
                }
                *power_value = real * real + imag * imag;
            }
            for mel in 0..64 {
                let mut value = 0.0;
                for (freq, &power_value) in power.iter().enumerate() {
                    value += power_value * self.mel_filter[freq * 64 + mel];
                }
                out[frame * 64 + mel] = value.clamp(1e-9, 1e9).ln();
            }
        }
        Ok((out, frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_registry_covers_every_learned_v3_operation() {
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::Gemm));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::Softmax));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::LayerNorm));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::Silu));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::Relu));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::Tanh));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::Conv1d));
        assert!(GIGAAM_V3_HOT_OPS.contains(&HotOp::GroupedConv1d));
    }

    fn row(class: usize, value: f32) -> Vec<f32> {
        let mut row = vec![0.0; NUM_CLASSES];
        row[class] = value;
        row
    }

    #[test]
    fn fixed_rnnt_contract_is_not_ctc() {
        assert_eq!(expected_manifest().len(), TENSOR_COUNT);
        assert_eq!(BLANK_ID, NUM_CLASSES - 1);
        assert_eq!(MAX_SYMBOLS_PER_STEP, 10);
        assert_eq!(VOCAB_SIZE + 1, NUM_CLASSES);
        assert_eq!(
            AUTHENTICATED_PREPARED_SHA256,
            Some("cee04765f031d6ee5088849ecb0e5c1db4e58ca28a345ce4d049015cd683a64e")
        );
    }

    #[test]
    fn factored_greedy_blank_advances_and_commits_nonblank_only() {
        let frames = vec![vec![row(7, 2.0), row(BLANK_ID, 3.0)], vec![row(9, 3.0)]];
        let (tokens, committed) = greedy_decode_factored(&frames, MAX_SYMBOLS_PER_STEP).unwrap();
        assert_eq!(tokens, vec![7, 9]);
        assert_eq!(committed, vec![7, 9]);
    }

    #[test]
    fn factored_greedy_caps_each_frame_at_ten_symbols() {
        let frames = vec![vec![row(3, 2.0); MAX_SYMBOLS_PER_STEP + 1]];
        let (tokens, committed) = greedy_decode_factored(&frames, MAX_SYMBOLS_PER_STEP).unwrap();
        assert_eq!(tokens.len(), MAX_SYMBOLS_PER_STEP);
        assert_eq!(committed.len(), MAX_SYMBOLS_PER_STEP);
    }

    #[test]
    fn factored_greedy_ties_choose_first_and_reject_nonfinite() {
        let ties = vec![vec![vec![0.0; NUM_CLASSES]]];
        let (tokens, _) = greedy_decode_factored(&ties, MAX_SYMBOLS_PER_STEP).unwrap();
        assert_eq!(tokens, vec![0]);
        let mut bad = row(1, 1.0);
        bad[2] = f32::NAN;
        assert!(greedy_decode_factored(&[vec![bad]], MAX_SYMBOLS_PER_STEP).is_err());
    }
}
