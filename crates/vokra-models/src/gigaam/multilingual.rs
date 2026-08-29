//! Native GigaAM Multilingual CTC model binding.
//!
//! This module binds only the converter's authenticated 552-tensor GGUF
//! contract. It intentionally exposes CPU execution today; requesting Metal
//! returns `UnsupportedOp` until the complete learned op set is dispatched
//! through `Compute`.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{Result, VokraError};
use vokra_ops::conformer::{
    ConformerConfig, ConformerConvWeights, ConformerEncoder, ConformerLayerWeights,
    ConformerSubsampleWeights, ConformerWeights, ConvSubsampleKind, FeedForwardWeights, MhaWeights,
    PositionEncoding,
};
use vokra_ops::ctc_decode_greedy;

/// GGUF architecture marker accepted by the Multilingual binder.
pub const ARCH: &str = "gigaam_multilingual";
/// Stable model name marker required by the Multilingual binder.
pub const NAME: &str = "sber-gigaam-multilingual";
/// Number of output classes, including the CTC blank class.
pub const VOCAB_SIZE: usize = 71;
/// CTC blank class index; the 70 vocabulary symbols occupy indices `0..70`.
pub const BLANK_ID: usize = 70;
/// Required input PCM sample rate in hertz.
pub const SAMPLE_RATE: u32 = 16_000;
/// Fixed upstream source revision used to authenticate the model topology.
pub const SOURCE_REVISION: &str = "7447938d791c4f3e643386ee22c33777004293a5";
/// Fixed Hugging Face model revision used by the authenticated checkpoint.
pub const HF_REVISION: &str = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8";
/// SHA-256 of the fixed multilingual `config.json` at [`HF_REVISION`].
pub const CONFIG_SHA256: &str = "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653";
/// SHA-256 of the authenticated upstream PyTorch checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728";
/// Filled only after an independent VAST digest review of the prepared
/// safetensors artifact. A self-declared sidecar digest is insufficient.
pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = None;
/// The fixed HF config explicitly overrides the upstream preprocessor
/// default: `center=false` (config SHA above), not the library default.
pub const PREPROCESSOR_CENTER: bool = false;

/// Exact 70-symbol CTC vocabulary from the authenticated multilingual card.
pub const VOCABULARY: &[char] = &[
    ' ', '\'', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь',
    'э', 'ю', 'я', 'ё', 'і', 'ғ', 'қ', 'ң', 'ү', 'ұ', 'һ', 'ә', 'ө',
];

/// Native intermediate values used by the VAST-only real-weight parity test.
///
/// The vectors contain only the valid encoded prefix. This diagnostic surface
/// is separate from ordinary transcription and does not authorize an
/// unauthenticated checkpoint.
#[derive(Debug)]
pub struct GigaamMultilingualTrace {
    /// Encoder output in row-major `[encoded_frames, 768]` order.
    pub encoded: Vec<f32>,
    /// Number of valid encoder frames represented by [`Self::encoded`].
    pub encoded_frames: usize,
    /// CTC head log-probabilities in row-major `[encoded_frames, 71]` order.
    pub logits: Vec<f32>,
    /// Per-frame greedy argmax IDs before CTC adjacent-repeat collapsing and
    /// blank removal.
    pub raw_argmax: Vec<u32>,
    /// Greedy CTC IDs after adjacent-repeat collapsing and blank removal.
    pub token_ids: Vec<u32>,
}

fn tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>> {
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("GigaAM tensor `{name}`: {error}")))
}

fn meta_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let got = file.get(key).and_then(|value| value.as_u64());
    if got != Some(expected as u64) {
        return Err(VokraError::ModelLoad(format!(
            "GigaAM metadata `{key}` must be {expected}, found {got:?}"
        )));
    }
    Ok(())
}

fn meta_str(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    if file.get(key).and_then(|value| value.as_str()) != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "GigaAM metadata `{key}` must be {expected:?}"
        )));
    }
    Ok(())
}

fn expected_names() -> Vec<String> {
    let mut names = vec![
        "model.preprocessor.featurizer.0.spectrogram.window".into(),
        "model.preprocessor.featurizer.0.mel_scale.fb".into(),
        "model.encoder.pre_encode.conv.0.weight".into(),
        "model.encoder.pre_encode.conv.0.bias".into(),
        "model.encoder.pre_encode.conv.2.weight".into(),
        "model.encoder.pre_encode.conv.2.bias".into(),
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
            names.extend([format!("{p}.{n}.weight"), format!("{p}.{n}.bias")]);
        }
        for branch in ["feed_forward1", "feed_forward2"] {
            names.extend([
                format!("{p}.{branch}.linear1.weight"),
                format!("{p}.{branch}.linear1.bias"),
                format!("{p}.{branch}.linear2.weight"),
                format!("{p}.{branch}.linear2.bias"),
            ]);
        }
        names.extend([
            format!("{p}.conv.pointwise_conv1.weight"),
            format!("{p}.conv.pointwise_conv1.bias"),
            format!("{p}.conv.depthwise_conv.weight"),
            format!("{p}.conv.depthwise_conv.bias"),
            format!("{p}.conv.batch_norm.weight"),
            format!("{p}.conv.batch_norm.bias"),
            format!("{p}.conv.pointwise_conv2.weight"),
            format!("{p}.conv.pointwise_conv2.bias"),
        ]);
        for n in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            names.extend([
                format!("{p}.self_attn.{n}.weight"),
                format!("{p}.self_attn.{n}.bias"),
            ]);
        }
    }
    names.extend([
        "model.head.decoder_layers.0.weight".into(),
        "model.head.decoder_layers.0.bias".into(),
    ]);
    names
}

fn expected_shape(name: &str) -> &'static [u64] {
    if name.ends_with("spectrogram.window") {
        &[320]
    } else if name.ends_with("mel_scale.fb") {
        &[161, 64]
    } else if name.ends_with("pre_encode.conv.0.weight") {
        &[768, 64, 5]
    } else if name.ends_with("pre_encode.conv.2.weight") {
        &[768, 768, 5]
    } else if name.ends_with("pre_encode.conv.0.bias") || name.ends_with("pre_encode.conv.2.bias") {
        &[768]
    } else if name.ends_with("decoder_layers.0.weight") {
        &[71, 768, 1]
    } else if name.ends_with("decoder_layers.0.bias") {
        &[71]
    } else if name.contains("linear1.weight") {
        &[3072, 768]
    } else if name.contains("linear1.bias") {
        &[3072]
    } else if name.contains("linear2.weight") {
        &[768, 3072]
    } else if name.contains("linear2.bias") {
        &[768]
    } else if name.contains("pointwise_conv1.weight") {
        &[1536, 768, 1]
    } else if name.contains("pointwise_conv1.bias") {
        &[1536]
    } else if name.contains("depthwise_conv.weight") {
        &[768, 1, 5]
    } else if name.contains("pointwise_conv2.weight") {
        &[768, 768, 1]
    } else if name.ends_with(".weight") && name.contains("self_attn.linear") {
        &[768, 768]
    } else if name.ends_with(".weight") || name.ends_with(".bias") {
        &[768]
    } else {
        &[]
    }
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

fn log_softmax_rows(values: &mut [f32], classes: usize) -> Result<()> {
    if classes == 0 || values.len() % classes != 0 {
        return Err(VokraError::InvalidArgument(
            "GigaAM CTC logits shape is not divisible by the class count".into(),
        ));
    }
    for row in values.chunks_exact_mut(classes) {
        let max = row
            .iter()
            .copied()
            .max_by(f32::total_cmp)
            .ok_or_else(|| VokraError::ModelLoad("GigaAM empty CTC row".into()))?;
        let sum = row.iter().map(|value| (*value - max).exp()).sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(VokraError::ModelLoad(
                "GigaAM CTC log-softmax normalization is non-finite".into(),
            ));
        }
        let log_norm = max + sum.ln();
        for value in row {
            *value -= log_norm;
        }
    }
    Ok(())
}

fn bind_layer(file: &GgufFile, index: usize) -> Result<ConformerLayerWeights> {
    let p = format!("model.encoder.layers.{index}");
    let (ln1_gamma, ln1_beta) = bind_norm(file, &format!("{p}.norm_feed_forward1"))?;
    let (ln3_gamma, ln3_beta) = bind_norm(file, &format!("{p}.norm_conv"))?;
    let (ln2_gamma, ln2_beta) = bind_norm(file, &format!("{p}.norm_self_att"))?;
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

fn frame_count(samples: usize) -> Result<usize> {
    if samples < 320 {
        return Err(VokraError::InvalidArgument(
            "GigaAM audio must contain at least one 320-sample frame".into(),
        ));
    }
    samples
        .checked_sub(320)
        .and_then(|value| value.checked_div(160))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| VokraError::InvalidArgument("GigaAM frame count overflows usize".into()))
}

/// Authenticated GigaAM Multilingual CTC model with a native CPU route.
///
/// Construction accepts only a GGUF file carrying the fixed metadata and
/// complete 552-tensor manifest. The prepared-artifact digest gate remains
/// fail-closed until an independently reviewed VAST digest is recorded.
#[derive(Debug)]
pub struct GigaamMultilingual {
    encoder: ConformerEncoder,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    window: Vec<f32>,
    mel_filter: Vec<f32>,
    backend: BackendKind,
}

impl GigaamMultilingual {
    /// Bind the authenticated Multilingual CTC model from a GGUF file.
    ///
    /// This validates the architecture, fixed revisions and configuration,
    /// prepared-artifact identity, provenance metadata, and all 552 tensor
    /// names, shapes, and dtypes before constructing the native encoder.
    /// Binding returns an error while the independent prepared-artifact digest
    /// is unavailable.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        meta_str(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        meta_str(file, chunks::KEY_MODEL_NAME, NAME)?;
        let Some(expected_prepared_sha256) = AUTHENTICATED_PREPARED_SHA256 else {
            return Err(VokraError::ModelLoad(
                "GigaAM prepared safetensors digest is not independently authenticated; runtime bind is disabled until VAST evidence is reviewed".into(),
            ));
        };
        for (key, value) in [
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
            ("vocab_size", 71),
            ("blank_id", 70),
        ] {
            meta_u32(file, &format!("vokra.gigaam_multilingual.{key}"), value)?;
        }
        for (key, value) in [
            ("model_class", "ctc"),
            ("model_name", "multilingual_ctc"),
            ("topology", "CTC"),
        ] {
            meta_str(file, &format!("vokra.gigaam_multilingual.{key}"), value)?;
        }
        meta_str(file, "vokra.gigaam_multilingual.revision", HF_REVISION)?;
        meta_str(
            file,
            "vokra.gigaam_multilingual.source_revision",
            SOURCE_REVISION,
        )?;
        meta_str(
            file,
            "vokra.gigaam_multilingual.config_sha256",
            CONFIG_SHA256,
        )?;
        meta_str(
            file,
            "vokra.gigaam_multilingual.checkpoint_sha256",
            CHECKPOINT_SHA256,
        )?;
        meta_str(
            file,
            "vokra.gigaam_multilingual.prepared_sha256",
            expected_prepared_sha256,
        )?;
        meta_str(file, "vokra.provenance.weight_license", "permissive")?;
        meta_str(file, "vokra.provenance.license", "MIT")?;
        meta_str(
            file,
            "vokra.provenance.model_id",
            "ai-sage/GigaAM-Multilingual",
        )?;
        meta_str(
            file,
            "vokra.provenance.source",
            "https://huggingface.co/ai-sage/GigaAM-Multilingual",
        )?;
        let expected = expected_names();
        let actual: std::collections::BTreeSet<&str> = file
            .tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        let wanted: std::collections::BTreeSet<&str> =
            expected.iter().map(String::as_str).collect();
        if actual != wanted {
            return Err(VokraError::ModelLoad(format!(
                "GigaAM tensor manifest mismatch: expected {}, found {}",
                wanted.len(),
                actual.len()
            )));
        }
        if file.tensors().len() != 552 {
            return Err(VokraError::ModelLoad(format!(
                "GigaAM tensor count must be 552, found {}",
                file.tensors().len()
            )));
        }
        for name in &expected {
            let info = file
                .tensor_info(name)
                .ok_or_else(|| VokraError::ModelLoad(format!("GigaAM tensor `{name}` missing")))?;
            let shape = expected_shape(name);
            if info.dtype != vokra_core::gguf::GgmlType::F32 || info.dimensions.as_slice() != shape
            {
                return Err(VokraError::ModelLoad(format!(
                    "GigaAM tensor `{name}` must be F32 {shape:?}, found {:?} {:?}",
                    info.dtype, info.dimensions
                )));
            }
        }
        let window = tensor(file, "model.preprocessor.featurizer.0.spectrogram.window")?;
        let mel_filter = tensor(file, "model.preprocessor.featurizer.0.mel_scale.fb")?;
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
                    .map(|index| bind_layer(file, index))
                    .collect::<Result<Vec<_>>>()?,
            },
        )?;
        Ok(Self {
            encoder,
            head_w: tensor(file, "model.head.decoder_layers.0.weight")?,
            head_b: tensor(file, "model.head.decoder_layers.0.bias")?,
            window,
            mel_filter,
            backend: BackendKind::Cpu,
        })
    }

    /// Select the backend for the complete learned operation graph.
    ///
    /// Only [`BackendKind::Cpu`] is currently implemented. Other backends
    /// return [`VokraError::UnsupportedOp`] rather than silently falling back
    /// to scalar CPU execution.
    pub fn with_backend(mut self, backend: BackendKind) -> Result<Self> {
        if backend != BackendKind::Cpu {
            return Err(VokraError::UnsupportedOp("gigaam multilingual Metal/CUDA backend is not wired for the complete learned frontend and encoder op set".into()));
        }
        self.backend = backend;
        Ok(self)
    }

    /// Backend actually selected for every learned operation.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Run the diagnostic native route on a prepared `[frames, 64]` log-mel
    /// input and expose encoder, logits, and greedy token IDs for VAST parity.
    ///
    /// This method is intended for the ignored real-weight parity test. It
    /// uses the same strict binder and CPU-only backend as [`Self::transcribe`].
    pub fn parity_trace_log_mel(
        &self,
        mel: &[f32],
        frames: usize,
    ) -> Result<GigaamMultilingualTrace> {
        self.parity_trace_log_mel_impl(mel, frames, None)
    }

    /// Run the diagnostic native route on mono 16 kHz PCM for VAST parity.
    ///
    /// The returned intermediate values use the authenticated fixed
    /// `center=false`, 320-point, 160-hop frontend.
    pub fn parity_trace_pcm(&self, pcm: &[f32]) -> Result<GigaamMultilingualTrace> {
        let (mel, frames) = self.log_mel_from_pcm(pcm)?;
        self.parity_trace_log_mel(&mel, frames)
    }

    /// Run the native CTC route on a prepared `[frames, 64]` log-mel input.
    ///
    /// Greedy CTC decoding collapses adjacent repeats, removes blank IDs, and
    /// joins the remaining IDs using the authenticated 70-symbol vocabulary.
    pub fn transcribe_log_mel(&self, mel: &[f32], frames: usize) -> Result<String> {
        self.transcribe_log_mel_impl(mel, frames, None)
    }

    /// Run CTC on a padded log-mel buffer while limiting decoding to the
    /// sample's valid post-stem length. This is the no-tail-leak path for
    /// batched short samples.
    ///
    /// The buffer must contain `frames * 64` finite values, and
    /// `valid_frames` must not exceed `frames`.
    pub fn transcribe_log_mel_with_valid_frames(
        &self,
        mel: &[f32],
        frames: usize,
        valid_frames: usize,
    ) -> Result<String> {
        self.transcribe_log_mel_impl(mel, frames, Some(valid_frames))
    }

    fn transcribe_log_mel_impl(
        &self,
        mel: &[f32],
        frames: usize,
        valid_frames: Option<usize>,
    ) -> Result<String> {
        let trace = self.parity_trace_log_mel_impl(mel, frames, valid_frames)?;
        let mut text = String::new();
        for id in trace.token_ids {
            let symbol = VOCABULARY.get(id as usize).ok_or_else(|| {
                VokraError::ModelLoad("GigaAM decoded vocabulary id out of range".into())
            })?;
            text.push(*symbol);
        }
        Ok(text)
    }

    fn parity_trace_log_mel_impl(
        &self,
        mel: &[f32],
        frames: usize,
        valid_frames: Option<usize>,
    ) -> Result<GigaamMultilingualTrace> {
        if self.backend != BackendKind::Cpu {
            return Err(VokraError::UnsupportedOp(
                "gigaam multilingual backend is not available".into(),
            ));
        }
        if !mel.iter().all(|value| value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "GigaAM log-mel input contains non-finite values".into(),
            ));
        }
        let (hidden, time, valid_time) = if let Some(valid_frames) = valid_frames {
            let (hidden, time, valid_time) =
                self.encoder
                    .forward_with_valid_frames(mel, frames, valid_frames)?;
            (hidden, time, valid_time)
        } else {
            let (hidden, time) = self.encoder.forward(mel, frames)?;
            (hidden, time, time)
        };
        if valid_time == 0 || valid_time > time {
            return Err(VokraError::InvalidArgument(
                "GigaAM valid CTC length is outside encoded bounds".into(),
            ));
        }
        let encoded_len = valid_time.checked_mul(768).ok_or_else(|| {
            VokraError::InvalidArgument("GigaAM encoded shape overflows usize".into())
        })?;
        if hidden.len() < encoded_len {
            return Err(VokraError::InvalidArgument(
                "GigaAM encoder returned fewer frames than its valid length".into(),
            ));
        }
        let mut hidden = hidden;
        hidden.truncate(encoded_len);
        let logits_len = valid_time.checked_mul(VOCAB_SIZE).ok_or_else(|| {
            VokraError::InvalidArgument("GigaAM CTC logits shape overflows usize".into())
        })?;
        let mut logits = vec![0.0; logits_len];
        for t in 0..valid_time {
            for class in 0..VOCAB_SIZE {
                let mut value = self.head_b[class];
                for d in 0..768 {
                    value += self.head_w[(class * 768 + d) * 1] * hidden[t * 768 + d];
                }
                logits[t * VOCAB_SIZE + class] = value;
            }
        }
        log_softmax_rows(&mut logits, VOCAB_SIZE)?;
        let raw_argmax = logits
            .chunks_exact(VOCAB_SIZE)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(index, _)| index as u32)
                    .ok_or_else(|| VokraError::ModelLoad("GigaAM empty CTC row".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids = ctc_decode_greedy(&logits, valid_time, VOCAB_SIZE, BLANK_ID)?
            .into_iter()
            .map(|id| id as u32)
            .collect();
        Ok(GigaamMultilingualTrace {
            encoded: hidden,
            encoded_frames: valid_time,
            logits,
            raw_argmax,
            token_ids,
        })
    }

    /// Extract the authenticated 320-point log-mel frontend from mono 16 kHz
    /// PCM, then run the native CTC route. This mirrors
    /// `MelSpectrogram(power=2, center=false)` and uses the checkpoint's
    /// learned window and frequency×mel filter matrix.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        let (mel, frames) = self.log_mel_from_pcm(pcm)?;
        self.transcribe_log_mel(&mel, frames)
    }

    fn log_mel_from_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        if self.backend != BackendKind::Cpu {
            return Err(VokraError::UnsupportedOp(
                "gigaam multilingual backend is not available".into(),
            ));
        }
        if !pcm.iter().all(|value| value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "GigaAM PCM input contains non-finite values".into(),
            ));
        }
        let frames = frame_count(pcm.len())?;
        let mel_len = frames.checked_mul(64).ok_or_else(|| {
            VokraError::InvalidArgument("GigaAM mel shape overflows usize".into())
        })?;
        let mut mel = vec![0.0f32; mel_len];
        let mut power = vec![0.0f32; 161];
        for frame in 0..frames {
            let start = frame * 160;
            for freq in 0..=160 {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for sample in 0..320 {
                    let angle = 2.0 * core::f32::consts::PI * freq as f32 * sample as f32 / 320.0;
                    let value = pcm[start + sample] * self.window[sample];
                    real += value * angle.cos();
                    imag -= value * angle.sin();
                }
                power[freq] = real * real + imag * imag;
            }
            for channel in 0..64 {
                let mut value = 0.0f32;
                for freq in 0..161 {
                    value += power[freq] * self.mel_filter[freq * 64 + channel];
                }
                mel[frame * 64 + channel] = value.clamp(1e-9, 1e9).ln();
            }
        }
        Ok((mel, frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticated_ctc_contract_is_fixed() {
        assert_eq!(VOCABULARY.len(), 70);
        assert_eq!(BLANK_ID, VOCABULARY.len());
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(SOURCE_REVISION, "7447938d791c4f3e643386ee22c33777004293a5");
        assert_eq!(HF_REVISION, "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8");
        assert_eq!(
            CONFIG_SHA256,
            "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653"
        );
        assert!(!PREPROCESSOR_CENTER);
    }

    #[test]
    fn strict_manifest_generator_has_exact_tensor_count() {
        assert_eq!(expected_names().len(), 552);
        assert_eq!(
            expected_shape("model.encoder.layers.0.feed_forward1.linear1.bias"),
            &[3072]
        );
        assert_eq!(
            expected_shape("model.encoder.layers.0.conv.pointwise_conv1.bias"),
            &[1536]
        );
        assert!(
            !expected_names()
                .iter()
                .any(|name| { name.ends_with("running_mean") || name.ends_with("running_var") })
        );
    }

    #[test]
    fn fixed_center_false_frame_count_is_floor_formula() {
        assert_eq!(frame_count(320).unwrap(), 1);
        assert_eq!(frame_count(480).unwrap(), 2);
        assert_eq!(frame_count(481).unwrap(), 2);
        assert!(frame_count(319).is_err());
    }

    #[test]
    fn ctc_log_softmax_rows_are_normalized() {
        let mut rows = vec![0.0f32, 1.0, 2.0, -3.0, 4.0, 0.5];
        log_softmax_rows(&mut rows, 3).unwrap();
        for row in rows.chunks_exact(3) {
            let sum = row.iter().map(|value| value.exp()).sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-6, "row exp sum={sum}");
        }
    }
}
