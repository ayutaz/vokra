use std::sync::Arc;

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::require_tensor_shape;

use super::{BarkConfig, CODEBOOK_DIM, CODEBOOK_SIZE};

const LABEL: &str = "bark";

/// Mapping-owning weight handle. Tensor slices are borrowed on demand so the
/// 1.67/4.47 GB checkpoints are never copied into a second resident store.
#[derive(Debug, Clone)]
pub(super) struct BarkMappedWeights {
    file: Arc<GgufFile>,
}

impl BarkMappedWeights {
    pub(super) fn bind(file: Arc<GgufFile>, config: &BarkConfig) -> Result<Self> {
        validate_language_models(&file, config)?;
        validate_codec_decoder(&file)?;

        // Validate the zero-copy contract for every authenticated tensor,
        // including training-side encoder/codebook statistics not read by the
        // released synthesis route. Creating a slice does not fault tensor
        // pages or allocate a resident copy.
        for tensor in file.tensors() {
            vokra_mmap::tensor_f32_view(&file, &tensor.name).map_err(|error| {
                VokraError::ModelLoad(format!(
                    "{LABEL}: zero-copy view for `{}` failed: {error}",
                    tensor.name
                ))
            })?;
        }
        Ok(Self { file })
    }

    pub(super) fn tensor<'a>(&'a self, name: &str, shape: &[usize]) -> Result<&'a [f32]> {
        require_tensor_shape(&self.file, LABEL, name, shape)?;
        vokra_mmap::tensor_f32_view(&self.file, name)
    }
}

fn validate_language_models(file: &GgufFile, config: &BarkConfig) -> Result<()> {
    validate_causal_stage(file, "semantic", 129_600, 10_048, config)?;
    validate_causal_stage(file, "coarse_acoustics", 12_096, 12_096, config)?;
    validate_fine_stage(file, config)
}

fn validate_causal_stage(
    file: &GgufFile,
    prefix: &str,
    input_vocab: usize,
    output_vocab: usize,
    config: &BarkConfig,
) -> Result<()> {
    let hidden = config.hidden_size;
    bind(
        file,
        &format!("{prefix}.input_embeds_layer.weight"),
        &[input_vocab, hidden],
    )?;
    bind(
        file,
        &format!("{prefix}.position_embeds_layer.weight"),
        &[config.block_size, hidden],
    )?;
    bind(file, &format!("{prefix}.layernorm_final.weight"), &[hidden])?;
    bind(
        file,
        &format!("{prefix}.lm_head.weight"),
        &[output_vocab, hidden],
    )?;
    for layer in 0..config.num_layers_per_stage {
        let base = format!("{prefix}.layers.{layer}");
        bind(
            file,
            &format!("{base}.attn.att_proj.weight"),
            &[3 * hidden, hidden],
        )?;
        bind(
            file,
            &format!("{base}.attn.out_proj.weight"),
            &[hidden, hidden],
        )?;
        bind(file, &format!("{base}.layernorm_1.weight"), &[hidden])?;
        bind(file, &format!("{base}.layernorm_2.weight"), &[hidden])?;
        bind(
            file,
            &format!("{base}.mlp.in_proj.weight"),
            &[4 * hidden, hidden],
        )?;
        bind(
            file,
            &format!("{base}.mlp.out_proj.weight"),
            &[hidden, 4 * hidden],
        )?;
    }
    Ok(())
}

fn validate_fine_stage(file: &GgufFile, config: &BarkConfig) -> Result<()> {
    let prefix = "fine_acoustics";
    let hidden = config.hidden_size;
    for codebook in 0..8 {
        bind(
            file,
            &format!("{prefix}.input_embeds_layers.{codebook}.weight"),
            &[1_056, hidden],
        )?;
    }
    bind(
        file,
        &format!("{prefix}.position_embeds_layer.weight"),
        &[config.block_size, hidden],
    )?;
    bind(file, &format!("{prefix}.layernorm_final.weight"), &[hidden])?;
    bind(file, &format!("{prefix}.layernorm_final.bias"), &[hidden])?;
    for head in 0..7 {
        bind(
            file,
            &format!("{prefix}.lm_heads.{head}.weight"),
            &[1_056, hidden],
        )?;
    }
    for layer in 0..config.num_layers_per_stage {
        let base = format!("{prefix}.layers.{layer}");
        bind(
            file,
            &format!("{base}.attn.att_proj.weight"),
            &[3 * hidden, hidden],
        )?;
        bind(
            file,
            &format!("{base}.attn.out_proj.weight"),
            &[hidden, hidden],
        )?;
        for layer_norm in ["layernorm_1", "layernorm_2"] {
            bind(file, &format!("{base}.{layer_norm}.weight"), &[hidden])?;
            bind(file, &format!("{base}.{layer_norm}.bias"), &[hidden])?;
        }
        bind(
            file,
            &format!("{base}.mlp.in_proj.weight"),
            &[4 * hidden, hidden],
        )?;
        bind(
            file,
            &format!("{base}.mlp.out_proj.weight"),
            &[hidden, 4 * hidden],
        )?;
    }
    Ok(())
}

fn validate_codec_decoder(file: &GgufFile) -> Result<()> {
    bind_weight_norm_conv(
        file,
        "codec_model.decoder.layers.0.conv",
        &[512, 128, 7],
        512,
    )?;

    for layer in 0..2 {
        for family in ["weight_ih", "weight_hh"] {
            bind(
                file,
                &format!("codec_model.decoder.layers.1.lstm.{family}_l{layer}"),
                &[2_048, 512],
            )?;
        }
        for family in ["bias_ih", "bias_hh"] {
            bind(
                file,
                &format!("codec_model.decoder.layers.1.lstm.{family}_l{layer}"),
                &[2_048],
            )?;
        }
    }

    for (layer, input, output, kernel) in [
        (3usize, 512usize, 256usize, 16usize),
        (6, 256, 128, 10),
        (9, 128, 64, 8),
        (12, 64, 32, 4),
    ] {
        bind_weight_norm_conv_transpose(
            file,
            &format!("codec_model.decoder.layers.{layer}.conv"),
            input,
            output,
            kernel,
        )?;
    }

    for (layer, channels) in [(4usize, 256usize), (7, 128), (10, 64), (13, 32)] {
        let hidden = channels / 2;
        let base = format!("codec_model.decoder.layers.{layer}");
        bind_weight_norm_conv(
            file,
            &format!("{base}.block.1.conv"),
            &[hidden, channels, 3],
            hidden,
        )?;
        bind_weight_norm_conv(
            file,
            &format!("{base}.block.3.conv"),
            &[channels, hidden, 1],
            channels,
        )?;
        bind_weight_norm_conv(
            file,
            &format!("{base}.shortcut.conv"),
            &[channels, channels, 1],
            channels,
        )?;
    }
    bind_weight_norm_conv(file, "codec_model.decoder.layers.15.conv", &[1, 32, 7], 1)?;

    // The checkpoint publishes 32 residual codebooks. Bark synthesis uses the
    // first eight, but every table remains authenticated by the complete
    // manifest and shape-checked here so a same-count rearrangement cannot be
    // mistaken for the public codec.
    for codebook in 0..32 {
        let base = format!("codec_model.quantizer.layers.{codebook}.codebook");
        bind(file, &format!("{base}.cluster_size"), &[CODEBOOK_SIZE])?;
        bind(
            file,
            &format!("{base}.embed"),
            &[CODEBOOK_SIZE, CODEBOOK_DIM],
        )?;
        bind(
            file,
            &format!("{base}.embed_avg"),
            &[CODEBOOK_SIZE, CODEBOOK_DIM],
        )?;
        bind(file, &format!("{base}.inited"), &[1])?;
    }
    Ok(())
}

fn bind_weight_norm_conv(
    file: &GgufFile,
    prefix: &str,
    weight_shape: &[usize],
    output: usize,
) -> Result<()> {
    bind(file, &format!("{prefix}.weight_v"), weight_shape)?;
    bind(file, &format!("{prefix}.weight_g"), &[output, 1, 1])?;
    bind(file, &format!("{prefix}.bias"), &[output])
}

fn bind_weight_norm_conv_transpose(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) -> Result<()> {
    bind(
        file,
        &format!("{prefix}.weight_v"),
        &[input, output, kernel],
    )?;
    bind(file, &format!("{prefix}.weight_g"), &[input, 1, 1])?;
    bind(file, &format!("{prefix}.bias"), &[output])
}

fn bind(file: &GgufFile, name: &str, shape: &[usize]) -> Result<()> {
    require_tensor_shape(file, LABEL, name, shape)
}
