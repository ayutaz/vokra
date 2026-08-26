//! Embedded 44.1 kHz DAC adapters for the two authenticated Parler releases.
//!
//! Mini v1 preserves `parler-tts/dac_44khZ_8kbps`'s weight-normalized
//! `audio_encoder.model.*` tensors. Mini Multilingual v1.1 embeds the newer
//! plain-convolution `DacModel` under `audio_encoder.*`. Both implement the
//! same factorized 9 × (1024 × 8 → 1024) RVQ and the same 512× SEANet decoder;
//! only the checkpoint parameterization/names differ.

use vokra_core::gguf::GgufFile;
use vokra_core::{BackendKind, Result, VokraError};
use vokra_ops::{CodebookTable, DacOutProj, DacRvqAttrs};

use crate::compute::Compute;
use crate::dac::{DAC_HOT_OPS, DacDecoder};

use super::{
    CODEBOOK_SIZE, NUM_CODEBOOKS, PARLER_HOT_OPS, ParlerGeneratedCodes, ParlerGenerationConfig,
    ParlerModel, ParlerVariant, SAMPLE_RATE,
};

const CODEBOOK_DIM: usize = 8;
const LATENT_DIM: usize = 1_024;
const HOP_LENGTH: usize = 512;

/// End-to-end mono waveform result.
#[derive(Debug, Clone, PartialEq)]
pub struct ParlerSynthesis {
    /// 44.1 kHz mono PCM, normalized to the embedded DAC's float output.
    pub samples: Vec<f32>,
    /// Always 44,100 for the pinned releases.
    pub sample_rate: u32,
    /// Number of valid DAC frames decoded.
    pub frames: usize,
}

#[derive(Debug, Clone)]
pub(super) struct EmbeddedDac {
    tables: Vec<CodebookTable>,
    out_projs: Vec<DacOutProj>,
    attrs: DacRvqAttrs,
    decoder: DacDecoder,
    backend: BackendKind,
}

impl EmbeddedDac {
    pub(super) fn bind(
        file: &GgufFile,
        variant: ParlerVariant,
        backend: BackendKind,
    ) -> Result<Self> {
        let _ = Compute::for_backend(backend, DAC_HOT_OPS)?;
        let mut tables = Vec::with_capacity(NUM_CODEBOOKS);
        let mut out_projs = Vec::with_capacity(NUM_CODEBOOKS);
        for codebook in 0..NUM_CODEBOOKS {
            let base = match variant {
                ParlerVariant::MiniV1English => {
                    format!("audio_encoder.model.quantizer.quantizers.{codebook}")
                }
                ParlerVariant::MiniMultilingualV11 => {
                    format!("audio_encoder.quantizer.quantizers.{codebook}")
                }
            };
            tables.push(CodebookTable::new(
                CODEBOOK_SIZE,
                CODEBOOK_DIM,
                tensor(
                    file,
                    &format!("{base}.codebook.weight"),
                    &[CODEBOOK_SIZE, CODEBOOK_DIM],
                )?,
            )?);
            let bias = tensor(file, &format!("{base}.out_proj.bias"), &[LATENT_DIM])?;
            let weight = match variant {
                ParlerVariant::MiniV1English => {
                    let g = tensor(
                        file,
                        &format!("{base}.out_proj.weight_g"),
                        &[LATENT_DIM, 1, 1],
                    )?;
                    let v = tensor(
                        file,
                        &format!("{base}.out_proj.weight_v"),
                        &[LATENT_DIM, CODEBOOK_DIM, 1],
                    )?;
                    fold_weight_norm(&v, &g, LATENT_DIM, CODEBOOK_DIM)?
                }
                ParlerVariant::MiniMultilingualV11 => tensor(
                    file,
                    &format!("{base}.out_proj.weight"),
                    &[LATENT_DIM, CODEBOOK_DIM, 1],
                )?,
            };
            out_projs.push(DacOutProj::new(LATENT_DIM, CODEBOOK_DIM, weight, bias)?);
        }
        let decoder = match variant {
            ParlerVariant::MiniV1English => {
                DacDecoder::load_prefixed_weight_norm_44khz(file, "audio_encoder.model.")?
            }
            ParlerVariant::MiniMultilingualV11 => {
                DacDecoder::load_plain_transformers_44khz(file, "audio_encoder.decoder")?
            }
        };
        Ok(Self {
            tables,
            out_projs,
            attrs: DacRvqAttrs {
                n_codebooks: NUM_CODEBOOKS,
                codebook_size: CODEBOOK_SIZE,
                codebook_dim: CODEBOOK_DIM,
                d_model: LATENT_DIM,
            },
            decoder,
            backend,
        })
    }

    pub(super) fn decode(&self, codes: &ParlerGeneratedCodes) -> Result<Vec<f32>> {
        if codes.frames() == 0 {
            return Err(VokraError::InvalidArgument(
                "parler DAC decode requires at least one valid code frame".to_owned(),
            ));
        }
        let compute = Compute::for_backend(self.backend, PARLER_HOT_OPS)?;
        let time_major = compute.dac_rvq_f32(
            codes.as_frame_major(),
            codes.frames(),
            &self.tables,
            &self.out_projs,
            &self.attrs,
        )?;
        let mut channel_major = vec![0.0; time_major.len()];
        for frame in 0..codes.frames() {
            for channel in 0..LATENT_DIM {
                channel_major[channel * codes.frames() + frame] =
                    time_major[frame * LATENT_DIM + channel];
            }
        }
        let pcm = self
            .decoder
            .forward_with_compute(&channel_major, &compute)?;
        let expected = codes.frames().checked_mul(HOP_LENGTH).ok_or_else(|| {
            VokraError::InvalidArgument("parler DAC output extent overflows usize".to_owned())
        })?;
        if pcm.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "parler DAC emitted {} samples for {} frames, expected {expected}",
                pcm.len(),
                codes.frames()
            )));
        }
        Ok(pcm)
    }
}

impl ParlerModel {
    /// Decodes previously generated valid frame-major codes through the
    /// variant-matched embedded DAC.
    pub fn decode_codes(&self, codes: &ParlerGeneratedCodes) -> Result<ParlerSynthesis> {
        let frames = codes.frames();
        Ok(ParlerSynthesis {
            samples: self.codec().decode(codes)?,
            sample_rate: SAMPLE_RATE,
            frames,
        })
    }

    /// Runs FLAN-T5, Parler delayed generation, and the embedded DAC.
    pub fn synthesize(
        &self,
        description_token_ids: &[u32],
        description_mask: Option<&[bool]>,
        prompt_token_ids: &[u32],
        generation: &ParlerGenerationConfig,
    ) -> Result<ParlerSynthesis> {
        let codes = self.generate_codes(
            description_token_ids,
            description_mask,
            prompt_token_ids,
            generation,
        )?;
        self.decode_codes(&codes)
    }
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("parler DAC: required tensor `{name}` is missing"))
    })?;
    let actual: Vec<usize> = info.dimensions.iter().map(|&axis| axis as usize).collect();
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "parler DAC: tensor `{name}` shape {actual:?}, expected {expected:?}"
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "parler DAC: tensor `{name}` could not decode to f32: {error}"
        ))
    })
}

fn fold_weight_norm(v: &[f32], g: &[f32], rows: usize, row_width: usize) -> Result<Vec<f32>> {
    if v.len() != rows.saturating_mul(row_width) || g.len() != rows {
        return Err(VokraError::ModelLoad(format!(
            "parler DAC: weight-norm buffers are v={} g={}, expected {} and {rows}",
            v.len(),
            g.len(),
            rows.saturating_mul(row_width)
        )));
    }
    let mut weight = vec![0.0; v.len()];
    for row in 0..rows {
        let source = &v[row * row_width..(row + 1) * row_width];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "parler DAC: invalid weight-norm row {row}: norm={norm}, g={}",
                g[row]
            )));
        }
        let scale = g[row] / norm;
        for (destination, source) in weight[row * row_width..(row + 1) * row_width]
            .iter_mut()
            .zip(source)
        {
            *destination = *source * scale;
        }
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_dac_geometry_is_the_released_44khz_contract() {
        let attrs = DacRvqAttrs {
            n_codebooks: NUM_CODEBOOKS,
            codebook_size: CODEBOOK_SIZE,
            codebook_dim: CODEBOOK_DIM,
            d_model: LATENT_DIM,
        };
        assert_eq!(attrs.n_codebooks, 9);
        assert_eq!(attrs.codebook_size, 1_024);
        assert_eq!(attrs.codebook_dim, 8);
        assert_eq!(attrs.d_model, 1_024);
        assert_eq!(HOP_LENGTH, 512);
    }

    #[test]
    fn weight_norm_fold_matches_rowwise_torch_parameterization() {
        let folded = fold_weight_norm(&[3.0, 4.0, 0.0, 2.0], &[2.0, 3.0], 2, 2).unwrap();
        assert_eq!(folded, [1.2, 1.6, 0.0, 3.0]);
    }
}
