//! MeloTTS phoneme-feature encoder shared by all five language releases.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::sbv2::text_encoder::{LayerNorm, PositionWiseFFN, RelPositionMHA, SbV2TransformerBlock};
use crate::strict_checkpoint::load_tensor;

use super::{
    FILTER_CHANNELS, GIN_CHANNELS, HIDDEN_CHANNELS, INTER_CHANNELS, LABEL, MeloConfig, MeloVariant,
    N_HEADS, N_LAYERS, N_SPEAKERS_CAPACITY,
};

const BERT_DIMENSION: usize = 1_024;
const JA_BERT_DIMENSION: usize = 768;
const WINDOW_SIZE: usize = 4;
const FFN_KERNEL: usize = 3;
const CONDITION_LAYER: usize = 2;

/// Backend operations required by the MeloTTS text encoder.
pub const MELOTTS_TEXT_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm, HotOp::Conv1d];

/// Already-tokenized, language-aware features consumed by MeloTTS.
///
/// Every sequence has the same length. `bert` and `ja_bert` are
/// position-major matrices with widths 1,024 and 768 respectively. This
/// low-level surface is deliberately separate from raw-text G2P/tokenizer/BERT
/// sidecars, whose availability differs by language.
#[derive(Debug, Clone, Copy)]
pub struct MeloTextFeatures<'a> {
    /// MeloTTS symbol IDs, one per output text position.
    pub phoneme_ids: &'a [u32],
    /// Tone IDs, one per output text position.
    pub tones: &'a [u32],
    /// Language IDs, one per output text position.
    pub language_ids: &'a [u32],
    /// English/Chinese BERT features, `[sequence_len, 1024]` row-major.
    pub bert: &'a [f32],
    /// Japanese/Korean BERT features, `[sequence_len, 768]` row-major.
    pub ja_bert: &'a [f32],
    /// Official `spk2id` entry used for global conditioning.
    pub speaker_id: u32,
}

/// Native output of the MeloTTS text encoder.
#[derive(Debug, Clone)]
pub struct MeloTextOutput {
    /// Transformer hidden state, `[sequence_len, 192]` row-major.
    pub hidden: Vec<f32>,
    /// Prior mean, `[sequence_len, 192]` row-major.
    pub mean: Vec<f32>,
    /// Prior log-scale, `[sequence_len, 192]` row-major.
    pub log_scale: Vec<f32>,
    /// Speaker conditioning vector, `[256]`.
    pub speaker_conditioning: Vec<f32>,
    /// Number of encoded text positions.
    pub sequence_len: usize,
}

#[derive(Debug, Clone)]
struct Affine {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Affine {
    fn load(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.weight"),
                &[output, input, 1],
            )?,
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
            input,
            output,
        })
    }

    fn forward(&self, compute: &Compute, rows: &[f32]) -> Result<Vec<f32>> {
        if rows.is_empty() || rows.len() % self.input != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "melotts affine: input length {} is not a non-zero multiple of {}",
                rows.len(),
                self.input
            )));
        }
        let count = rows.len() / self.input;
        let weight = transpose_matrix(&self.weight, self.output, self.input);
        let mut output = vec![0.0; count * self.output];
        compute.gemm_f32(
            count,
            self.output,
            self.input,
            rows,
            &weight,
            Some(&self.bias),
            &mut output,
        )?;
        Ok(output)
    }
}

/// Loaded MeloTTS text encoder and speaker table.
pub struct MeloTextEncoder {
    config: MeloConfig,
    phoneme_embedding: Vec<f32>,
    tone_embedding: Vec<f32>,
    language_embedding: Vec<f32>,
    speaker_embedding: Vec<f32>,
    bert_projection: Affine,
    ja_bert_projection: Affine,
    speaker_projection: Affine,
    transformer: Vec<SbV2TransformerBlock>,
    stats_projection: Affine,
}

impl MeloTextEncoder {
    pub(super) fn from_gguf(file: &GgufFile, config: MeloConfig) -> Result<Self> {
        let hidden = HIDDEN_CHANNELS as usize;
        let head = hidden / N_HEADS as usize;
        let mut transformer = Vec::with_capacity(N_LAYERS as usize);
        for layer in 0..N_LAYERS as usize {
            let attention_prefix = format!("enc_p.encoder.attn_layers.{layer}");
            let q_weight = tensor(
                file,
                &format!("{attention_prefix}.conv_q.weight"),
                &[hidden, hidden, 1],
            )?;
            let q_bias = tensor(file, &format!("{attention_prefix}.conv_q.bias"), &[hidden])?;
            let k_weight = tensor(
                file,
                &format!("{attention_prefix}.conv_k.weight"),
                &[hidden, hidden, 1],
            )?;
            let k_bias = tensor(file, &format!("{attention_prefix}.conv_k.bias"), &[hidden])?;
            let v_weight = tensor(
                file,
                &format!("{attention_prefix}.conv_v.weight"),
                &[hidden, hidden, 1],
            )?;
            let v_bias = tensor(file, &format!("{attention_prefix}.conv_v.bias"), &[hidden])?;
            let o_weight = tensor(
                file,
                &format!("{attention_prefix}.conv_o.weight"),
                &[hidden, hidden, 1],
            )?;
            let o_bias = tensor(file, &format!("{attention_prefix}.conv_o.bias"), &[hidden])?;
            let relative_key = tensor(
                file,
                &format!("{attention_prefix}.emb_rel_k"),
                &[1, 2 * WINDOW_SIZE + 1, head],
            )?;
            let relative_value = tensor(
                file,
                &format!("{attention_prefix}.emb_rel_v"),
                &[1, 2 * WINDOW_SIZE + 1, head],
            )?;
            let attention = RelPositionMHA::new(
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                o_weight,
                o_bias,
                relative_key,
                relative_value,
                N_HEADS as usize,
                head,
                WINDOW_SIZE,
            );

            let ffn_prefix = format!("enc_p.encoder.ffn_layers.{layer}");
            let ffn = PositionWiseFFN::new(
                tensor(
                    file,
                    &format!("{ffn_prefix}.conv_1.weight"),
                    &[FILTER_CHANNELS as usize, hidden, FFN_KERNEL],
                )?,
                tensor(
                    file,
                    &format!("{ffn_prefix}.conv_1.bias"),
                    &[FILTER_CHANNELS as usize],
                )?,
                tensor(
                    file,
                    &format!("{ffn_prefix}.conv_2.weight"),
                    &[hidden, FILTER_CHANNELS as usize, FFN_KERNEL],
                )?,
                tensor(file, &format!("{ffn_prefix}.conv_2.bias"), &[hidden])?,
                hidden,
                FILTER_CHANNELS as usize,
                FFN_KERNEL,
            );
            let norm1 = load_norm(file, "enc_p.encoder.norm_layers_1", layer, hidden)?;
            let norm2 = load_norm(file, "enc_p.encoder.norm_layers_2", layer, hidden)?;
            transformer.push(SbV2TransformerBlock::new(
                attention, norm1, ffn, norm2, hidden,
            ));
        }

        Ok(Self {
            config,
            phoneme_embedding: tensor(
                file,
                "enc_p.emb.weight",
                &[config.n_symbols as usize, hidden],
            )?,
            tone_embedding: tensor(
                file,
                "enc_p.tone_emb.weight",
                &[config.num_tones as usize, hidden],
            )?,
            language_embedding: tensor(
                file,
                "enc_p.language_emb.weight",
                &[config.num_languages as usize, hidden],
            )?,
            speaker_embedding: tensor(
                file,
                "emb_g.weight",
                &[N_SPEAKERS_CAPACITY as usize, GIN_CHANNELS as usize],
            )?,
            bert_projection: Affine::load(file, "enc_p.bert_proj", BERT_DIMENSION, hidden)?,
            ja_bert_projection: Affine::load(
                file,
                "enc_p.ja_bert_proj",
                JA_BERT_DIMENSION,
                hidden,
            )?,
            speaker_projection: load_linear(
                file,
                "enc_p.encoder.spk_emb_linear",
                GIN_CHANNELS as usize,
                hidden,
            )?,
            transformer,
            stats_projection: Affine::load(
                file,
                "enc_p.proj",
                hidden,
                2 * INTER_CHANNELS as usize,
            )?,
        })
    }

    /// Runs the official feature encoder on one explicitly selected backend.
    ///
    /// A backend is admitted only if it covers the complete text hot-op set.
    /// There is no per-op CPU fallback.
    pub fn encode(
        &self,
        features: MeloTextFeatures<'_>,
        backend: BackendKind,
    ) -> Result<MeloTextOutput> {
        validate_features(self.config, features)?;
        let compute = Compute::for_backend(backend, MELOTTS_TEXT_HOT_OPS)?;
        self.encode_with_compute(features, &compute)
    }

    fn encode_with_compute(
        &self,
        features: MeloTextFeatures<'_>,
        compute: &Compute,
    ) -> Result<MeloTextOutput> {
        let sequence_len = features.phoneme_ids.len();
        let hidden_width = HIDDEN_CHANNELS as usize;
        let bert = self.bert_projection.forward(compute, features.bert)?;
        let ja_bert = self.ja_bert_projection.forward(compute, features.ja_bert)?;
        let mut hidden = vec![0.0; sequence_len * hidden_width];
        let scale = vokra_math::sqrt(HIDDEN_CHANNELS as f32);
        for position in 0..sequence_len {
            let phoneme = features.phoneme_ids[position] as usize;
            let tone = features.tones[position] as usize;
            let language = features.language_ids[position] as usize;
            for channel in 0..hidden_width {
                hidden[position * hidden_width + channel] = (self.phoneme_embedding
                    [phoneme * hidden_width + channel]
                    + self.tone_embedding[tone * hidden_width + channel]
                    + self.language_embedding[language * hidden_width + channel]
                    + bert[position * hidden_width + channel]
                    + ja_bert[position * hidden_width + channel])
                    * scale;
            }
        }

        let speaker_offset = features.speaker_id as usize * GIN_CHANNELS as usize;
        let speaker_conditioning =
            self.speaker_embedding[speaker_offset..speaker_offset + GIN_CHANNELS as usize].to_vec();
        let projected_speaker = self
            .speaker_projection
            .forward(compute, &speaker_conditioning)?;
        for (layer, block) in self.transformer.iter().enumerate() {
            if layer == CONDITION_LAYER {
                for row in hidden.chunks_exact_mut(hidden_width) {
                    for (value, conditioning) in row.iter_mut().zip(&projected_speaker) {
                        *value += conditioning;
                    }
                }
            }
            block.forward_with_compute(compute, &mut hidden, sequence_len)?;
        }

        let stats = self.stats_projection.forward(compute, &hidden)?;
        let latent = INTER_CHANNELS as usize;
        let mut mean = vec![0.0; sequence_len * latent];
        let mut log_scale = vec![0.0; sequence_len * latent];
        for position in 0..sequence_len {
            let source = &stats[position * 2 * latent..(position + 1) * 2 * latent];
            mean[position * latent..(position + 1) * latent].copy_from_slice(&source[..latent]);
            log_scale[position * latent..(position + 1) * latent]
                .copy_from_slice(&source[latent..]);
        }
        Ok(MeloTextOutput {
            hidden,
            mean,
            log_scale,
            speaker_conditioning,
            sequence_len,
        })
    }

    /// Returns the release configuration used to validate feature IDs.
    #[must_use]
    pub const fn config(&self) -> MeloConfig {
        self.config
    }
}

fn validate_features(config: MeloConfig, features: MeloTextFeatures<'_>) -> Result<()> {
    let sequence_len = features.phoneme_ids.len();
    if sequence_len == 0 {
        return Err(VokraError::InvalidArgument(
            "melotts text encoder: phoneme sequence is empty".to_owned(),
        ));
    }
    for (label, actual) in [
        ("tones", features.tones.len()),
        ("language_ids", features.language_ids.len()),
    ] {
        if actual != sequence_len {
            return Err(VokraError::InvalidArgument(format!(
                "melotts text encoder: {label} length {actual}, expected {sequence_len}"
            )));
        }
    }
    if features.bert.len() != sequence_len * BERT_DIMENSION {
        return Err(VokraError::InvalidArgument(format!(
            "melotts text encoder: bert length {}, expected {}",
            features.bert.len(),
            sequence_len * BERT_DIMENSION
        )));
    }
    if features.ja_bert.len() != sequence_len * JA_BERT_DIMENSION {
        return Err(VokraError::InvalidArgument(format!(
            "melotts text encoder: ja_bert length {}, expected {}",
            features.ja_bert.len(),
            sequence_len * JA_BERT_DIMENSION
        )));
    }
    validate_ids("phoneme_ids", features.phoneme_ids, config.n_symbols)?;
    validate_ids("tones", features.tones, config.num_tones)?;
    validate_ids("language_ids", features.language_ids, config.num_languages)?;
    if !speaker_is_active(config.variant, features.speaker_id) {
        return Err(VokraError::InvalidArgument(format!(
            "melotts text encoder: speaker_id {} is not registered by {:?}",
            features.speaker_id, config.variant
        )));
    }
    Ok(())
}

fn validate_ids(label: &str, ids: &[u32], upper: u32) -> Result<()> {
    if let Some((position, id)) = ids.iter().enumerate().find(|(_, id)| **id >= upper) {
        return Err(VokraError::InvalidArgument(format!(
            "melotts text encoder: {label}[{position}]={id} is outside 0..{upper}"
        )));
    }
    Ok(())
}

const fn speaker_is_active(variant: MeloVariant, speaker_id: u32) -> bool {
    match variant {
        MeloVariant::English => speaker_id < 5,
        MeloVariant::Chinese => speaker_id == 1,
        MeloVariant::Korean | MeloVariant::Spanish | MeloVariant::Japanese => speaker_id == 0,
    }
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    load_tensor(file, LABEL, name, expected)
}

fn load_norm(file: &GgufFile, prefix: &str, layer: usize, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        tensor(file, &format!("{prefix}.{layer}.gamma"), &[width])?,
        tensor(file, &format!("{prefix}.{layer}.beta"), &[width])?,
        width,
    ))
}

fn load_linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Affine> {
    Ok(Affine {
        weight: tensor(file, &format!("{prefix}.weight"), &[output, input])?,
        bias: tensor(file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
    })
}

fn transpose_matrix(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * columns);
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for column in 0..columns {
            output[column * rows + row] = input[row * columns + column];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_speaker_ids_are_fail_closed() {
        assert!(speaker_is_active(MeloVariant::English, 4));
        assert!(!speaker_is_active(MeloVariant::English, 5));
        assert!(speaker_is_active(MeloVariant::Chinese, 1));
        assert!(!speaker_is_active(MeloVariant::Chinese, 0));
        assert!(speaker_is_active(MeloVariant::Japanese, 0));
        assert!(!speaker_is_active(MeloVariant::Japanese, 1));
    }

    #[test]
    fn transpose_matrix_handles_rectangular_weights() {
        assert_eq!(
            transpose_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
