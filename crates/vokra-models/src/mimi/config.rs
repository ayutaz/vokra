//! Mimi neural-chain configuration resolved from the `vokra.mimi.*` GGUF
//! chunk group (M4-05-T11/T31 design fix; ADR M4-05 §D2/§D9).
//!
//! Every numeric comes from the GGUF — the converter writes the upstream
//! `kyutai-labs/moshi` `loaders.py` constants (`_seanet_kwargs` /
//! `_quantizer_kwargs` / `_transformer_kwargs`, transcribed in the ADR);
//! the runtime never hard-codes them (FR-LD-02 / FR-MD-02). `ratios` is
//! encoded as a count + indexed keys (`vokra.mimi.seanet.n_ratios` +
//! `vokra.mimi.seanet.ratio.{i}`) — the `vokra.quant.rule.*` precedent —
//! to keep the reader free of GGUF-array plumbing.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.mimi.sample_rate";
pub(crate) const KEY_FRAME_RATE_MHZ: &str = "vokra.mimi.frame_rate_mhz";

pub(crate) const KEY_SEANET_DIMENSION: &str = "vokra.mimi.seanet.dimension";
pub(crate) const KEY_SEANET_N_FILTERS: &str = "vokra.mimi.seanet.n_filters";
pub(crate) const KEY_SEANET_N_RESIDUAL_LAYERS: &str = "vokra.mimi.seanet.n_residual_layers";
pub(crate) const KEY_SEANET_KERNEL_SIZE: &str = "vokra.mimi.seanet.kernel_size";
pub(crate) const KEY_SEANET_RESIDUAL_KERNEL_SIZE: &str = "vokra.mimi.seanet.residual_kernel_size";
pub(crate) const KEY_SEANET_LAST_KERNEL_SIZE: &str = "vokra.mimi.seanet.last_kernel_size";
pub(crate) const KEY_SEANET_COMPRESS: &str = "vokra.mimi.seanet.compress";
pub(crate) const KEY_SEANET_DILATION_BASE: &str = "vokra.mimi.seanet.dilation_base";
pub(crate) const KEY_SEANET_N_RATIOS: &str = "vokra.mimi.seanet.n_ratios";
pub(crate) const PREFIX_SEANET_RATIO: &str = "vokra.mimi.seanet.ratio.";

pub(crate) const KEY_QUANTIZER_DIMENSION: &str = "vokra.mimi.quantizer.dimension";
pub(crate) const KEY_QUANTIZER_N_Q: &str = "vokra.mimi.quantizer.n_q";
pub(crate) const KEY_QUANTIZER_BINS: &str = "vokra.mimi.quantizer.bins";
pub(crate) const KEY_QUANTIZER_INPUT_DIMENSION: &str = "vokra.mimi.quantizer.input_dimension";
pub(crate) const KEY_QUANTIZER_OUTPUT_DIMENSION: &str = "vokra.mimi.quantizer.output_dimension";

pub(crate) const KEY_TRANSFORMER_D_MODEL: &str = "vokra.mimi.transformer.d_model";
pub(crate) const KEY_TRANSFORMER_N_HEAD: &str = "vokra.mimi.transformer.n_head";
pub(crate) const KEY_TRANSFORMER_N_LAYER: &str = "vokra.mimi.transformer.n_layer";
pub(crate) const KEY_TRANSFORMER_FF_DIM: &str = "vokra.mimi.transformer.ff_dim";
pub(crate) const KEY_TRANSFORMER_CONTEXT: &str = "vokra.mimi.transformer.context";
pub(crate) const KEY_TRANSFORMER_MAX_PERIOD: &str = "vokra.mimi.transformer.max_period";
pub(crate) const KEY_TRANSFORMER_LAYER_SCALE: &str = "vokra.mimi.transformer.layer_scale";

/// SEANet stack hparams (`loaders.py` `_seanet_kwargs` — ADR §D2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimiSeanetConfig {
    /// Latent width the encoder ends at / decoder starts from (512).
    pub dimension: usize,
    /// Base channel count (64); stage `s` runs at `n_filters * 2^s`.
    pub n_filters: usize,
    /// Residual blocks per stage (1).
    pub n_residual_layers: usize,
    /// Init conv kernel (7).
    pub kernel_size: usize,
    /// Residual-unit first conv kernel (3; the second conv is k=1).
    pub residual_kernel_size: usize,
    /// Final conv kernel (3).
    pub last_kernel_size: usize,
    /// Residual hidden divisor (`hidden = dim / compress`, 2).
    pub compress: usize,
    /// Residual dilation growth base (2; layer `j` dilates by `base^j`).
    pub dilation_base: usize,
    /// Up/down-sampling ratios in **decoder (upsampling) order**
    /// ([8, 6, 5, 4]); the encoder consumes them reversed (seanet.py).
    pub ratios: Vec<usize>,
}

impl MimiSeanetConfig {
    /// Total SEANet hop: `prod(ratios)` (960 → 25 Hz at 24 kHz).
    #[must_use]
    pub fn hop(&self) -> usize {
        self.ratios.iter().product()
    }
}

/// Bottleneck transformer hparams (`loaders.py` `_transformer_kwargs`).
#[derive(Debug, Clone, PartialEq)]
pub struct MimiTransformerConfig {
    /// Width (512).
    pub d_model: usize,
    /// Attention heads (8, MHA — no GQA split upstream).
    pub n_head: usize,
    /// Layers (8).
    pub n_layer: usize,
    /// Feed-forward width (2048; `gating="none"` = plain GELU MLP).
    pub ff_dim: usize,
    /// Causal attention context window (250 frames).
    pub context: usize,
    /// RoPE θ base (`max_period` = 10 000).
    pub max_period: usize,
    /// LayerScale initial value (0.01).
    pub layer_scale: f32,
}

/// Quantizer shape (`loaders.py` `_quantizer_kwargs`). The codebook tables
/// themselves are `vokra_ops::mimi_rvq::CodebookTable` — shared with the
/// decode op, never duplicated (ADR §D1-(c)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimiQuantizerConfig {
    /// Codebook entry width (256).
    pub dimension: usize,
    /// Codebook count (32).
    pub n_q: usize,
    /// Entries per codebook (2048).
    pub bins: usize,
    /// Latent width entering the quantizer (512 — projected to
    /// `dimension`).
    pub input_dimension: usize,
    /// Latent width leaving the quantizer (512).
    pub output_dimension: usize,
}

/// The resolved Mimi neural-chain hparams.
#[derive(Debug, Clone, PartialEq)]
pub struct MimiNeuralConfig {
    /// PCM rate (24 000 Hz).
    pub sample_rate: u32,
    /// Token frame rate in milli-Hz (12 500 = 12.5 Hz).
    pub frame_rate_mhz: u32,
    /// SEANet stack.
    pub seanet: MimiSeanetConfig,
    /// Bottleneck transformer (encoder and decoder each run one).
    pub transformer: MimiTransformerConfig,
    /// RVQ quantizer shape.
    pub quantizer: MimiQuantizerConfig,
}

impl MimiNeuralConfig {
    /// Reads the chunk group. Missing numerics read `0` (loud at
    /// [`Self::validate`]); wrong types are loud immediately (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a wrong-typed key or a ratio
    /// index hole.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_ratios = read_u32_or_zero(file, KEY_SEANET_N_RATIOS)? as usize;
        let mut ratios = Vec::with_capacity(n_ratios);
        for i in 0..n_ratios {
            let key = format!("{PREFIX_SEANET_RATIO}{i}");
            match file.get(&key) {
                Some(GgufMetadataValue::U32(v)) => ratios.push(*v as usize),
                None => {
                    return Err(VokraError::InvalidArgument(format!(
                        "mimi config: `{key}` missing but n_ratios = {n_ratios} \
                         (indexed-key hole)"
                    )));
                }
                Some(other) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "mimi config: `{key}` is not a UINT32 (got {:?})",
                        other.value_type()
                    )));
                }
            }
        }
        Ok(Self {
            sample_rate: read_u32_or_zero(file, KEY_SAMPLE_RATE)?,
            frame_rate_mhz: read_u32_or_zero(file, KEY_FRAME_RATE_MHZ)?,
            seanet: MimiSeanetConfig {
                dimension: read_u32_or_zero(file, KEY_SEANET_DIMENSION)? as usize,
                n_filters: read_u32_or_zero(file, KEY_SEANET_N_FILTERS)? as usize,
                n_residual_layers: read_u32_or_zero(file, KEY_SEANET_N_RESIDUAL_LAYERS)? as usize,
                kernel_size: read_u32_or_zero(file, KEY_SEANET_KERNEL_SIZE)? as usize,
                residual_kernel_size: read_u32_or_zero(file, KEY_SEANET_RESIDUAL_KERNEL_SIZE)?
                    as usize,
                last_kernel_size: read_u32_or_zero(file, KEY_SEANET_LAST_KERNEL_SIZE)? as usize,
                compress: read_u32_or_zero(file, KEY_SEANET_COMPRESS)? as usize,
                dilation_base: read_u32_or_zero(file, KEY_SEANET_DILATION_BASE)? as usize,
                ratios,
            },
            transformer: MimiTransformerConfig {
                d_model: read_u32_or_zero(file, KEY_TRANSFORMER_D_MODEL)? as usize,
                n_head: read_u32_or_zero(file, KEY_TRANSFORMER_N_HEAD)? as usize,
                n_layer: read_u32_or_zero(file, KEY_TRANSFORMER_N_LAYER)? as usize,
                ff_dim: read_u32_or_zero(file, KEY_TRANSFORMER_FF_DIM)? as usize,
                context: read_u32_or_zero(file, KEY_TRANSFORMER_CONTEXT)? as usize,
                max_period: read_u32_or_zero(file, KEY_TRANSFORMER_MAX_PERIOD)? as usize,
                layer_scale: read_f32_or(file, KEY_TRANSFORMER_LAYER_SCALE, 0.0)?,
            },
            quantizer: MimiQuantizerConfig {
                dimension: read_u32_or_zero(file, KEY_QUANTIZER_DIMENSION)? as usize,
                n_q: read_u32_or_zero(file, KEY_QUANTIZER_N_Q)? as usize,
                bins: read_u32_or_zero(file, KEY_QUANTIZER_BINS)? as usize,
                input_dimension: read_u32_or_zero(file, KEY_QUANTIZER_INPUT_DIMENSION)? as usize,
                output_dimension: read_u32_or_zero(file, KEY_QUANTIZER_OUTPUT_DIMENSION)? as usize,
            },
        })
    }

    /// Rejects `0`-placeholder / inconsistent shapes before any weights
    /// bind (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate(&self) -> Result<()> {
        let s = &self.seanet;
        if s.dimension == 0
            || s.n_filters == 0
            || s.kernel_size == 0
            || s.residual_kernel_size == 0
            || s.last_kernel_size == 0
            || s.compress == 0
            || s.dilation_base == 0
            || s.ratios.is_empty()
            || s.ratios.contains(&0)
        {
            return Err(VokraError::InvalidArgument(format!(
                "mimi config: seanet carries a 0-placeholder ({s:?})"
            )));
        }
        let t = &self.transformer;
        if t.n_layer > 0 {
            if t.d_model == 0 || t.n_head == 0 || t.ff_dim == 0 || t.max_period == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "mimi config: transformer carries a 0-placeholder ({t:?})"
                )));
            }
            if t.d_model % t.n_head != 0 || (t.d_model / t.n_head) % 2 != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "mimi config: transformer d_model {} must split into even-width \
                     heads (n_head {})",
                    t.d_model, t.n_head
                )));
            }
            if t.d_model != s.dimension {
                return Err(VokraError::InvalidArgument(format!(
                    "mimi config: transformer d_model {} != seanet dimension {} — the \
                     upstream bottleneck runs at the SEANet latent width",
                    t.d_model, s.dimension
                )));
            }
        }
        let q = &self.quantizer;
        if q.dimension == 0 || q.n_q == 0 || q.bins == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi config: quantizer carries a 0-placeholder ({q:?})"
            )));
        }
        if q.input_dimension != s.dimension || q.output_dimension != s.dimension {
            return Err(VokraError::InvalidArgument(format!(
                "mimi config: quantizer input/output dims ({}, {}) must equal the \
                 seanet dimension {} (loaders.py wiring)",
                q.input_dimension, q.output_dimension, s.dimension
            )));
        }
        self.frame_downsample_stride()?;
        Ok(())
    }

    /// SEANet hop in samples (`prod(ratios)`).
    #[must_use]
    pub fn seanet_hop(&self) -> usize {
        self.seanet.hop()
    }

    /// Samples per token frame (`sample_rate / frame_rate` — 1920 for
    /// 24 kHz / 12.5 Hz).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on zero / non-exact rates.
    pub fn frame_hop_samples(&self) -> Result<usize> {
        if self.sample_rate == 0 || self.frame_rate_mhz == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi config: sample_rate={} / frame_rate_mhz={} — both must be > 0",
                self.sample_rate, self.frame_rate_mhz
            )));
        }
        let num = self.sample_rate as u64 * 1000;
        let den = self.frame_rate_mhz as u64;
        if num % den != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi config: sample_rate {} not an integer multiple of the frame \
                 rate ({} mHz)",
                self.sample_rate, self.frame_rate_mhz
            )));
        }
        Ok((num / den) as usize)
    }

    /// Stride of the conv resample between the SEANet rate and the token
    /// frame rate (`get_mimi` encoder_frame_rate → frame_rate; 25 Hz →
    /// 12.5 Hz = 2). Must divide exactly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on zero / non-exact rate pairs.
    pub fn frame_downsample_stride(&self) -> Result<usize> {
        let frame_hop = self.frame_hop_samples()?;
        let seanet_hop = self.seanet_hop();
        if seanet_hop == 0 || frame_hop % seanet_hop != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi config: frame hop {frame_hop} not an integer multiple of the \
                 seanet hop {seanet_hop} — the conv resample stride would not be exact"
            )));
        }
        Ok(frame_hop / seanet_hop)
    }

    /// A miniature config for synthesized-weight tests: same shape
    /// relationships as the real Mimi (exact hops, transformer at the
    /// latent width) at toy dims. hop = 2·2 = 4; frame hop 8 → stride 2.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate: 16_000,
            frame_rate_mhz: 2_000_000, // 2 kHz frame rate → hop 8 samples
            seanet: MimiSeanetConfig {
                dimension: 8,
                n_filters: 2,
                n_residual_layers: 1,
                kernel_size: 5,
                residual_kernel_size: 3,
                last_kernel_size: 3,
                compress: 2,
                dilation_base: 2,
                ratios: vec![2, 2],
            },
            transformer: MimiTransformerConfig {
                d_model: 8,
                n_head: 2,
                n_layer: 1,
                ff_dim: 16,
                context: 16,
                max_period: 10_000,
                layer_scale: 0.01,
            },
            quantizer: MimiQuantizerConfig {
                dimension: 4,
                n_q: 3,
                bins: 8,
                input_dimension: 8,
                output_dimension: 8,
            },
        }
    }
}

fn read_u32_or_zero(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v),
        None => Ok(0),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "mimi config: `{key}` is not a UINT32 (got {:?})",
            other.value_type()
        ))),
    }
}

fn read_f32_or(file: &GgufFile, key: &str, default: f32) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        None => Ok(default),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "mimi config: `{key}` is not a FLOAT32 (got {:?})",
            other.value_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    #[test]
    fn tiny_config_is_self_consistent() {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        cfg.validate().expect("tiny config validates");
        assert_eq!(cfg.seanet_hop(), 4);
        assert_eq!(cfg.frame_hop_samples().unwrap(), 8);
        assert_eq!(cfg.frame_downsample_stride().unwrap(), 2);
    }

    #[test]
    fn gguf_round_trip_including_indexed_ratios() {
        let t = MimiNeuralConfig::tiny_for_tests();
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, t.sample_rate);
        b.add_u32(KEY_FRAME_RATE_MHZ, t.frame_rate_mhz);
        b.add_u32(KEY_SEANET_DIMENSION, t.seanet.dimension as u32);
        b.add_u32(KEY_SEANET_N_FILTERS, t.seanet.n_filters as u32);
        b.add_u32(
            KEY_SEANET_N_RESIDUAL_LAYERS,
            t.seanet.n_residual_layers as u32,
        );
        b.add_u32(KEY_SEANET_KERNEL_SIZE, t.seanet.kernel_size as u32);
        b.add_u32(
            KEY_SEANET_RESIDUAL_KERNEL_SIZE,
            t.seanet.residual_kernel_size as u32,
        );
        b.add_u32(
            KEY_SEANET_LAST_KERNEL_SIZE,
            t.seanet.last_kernel_size as u32,
        );
        b.add_u32(KEY_SEANET_COMPRESS, t.seanet.compress as u32);
        b.add_u32(KEY_SEANET_DILATION_BASE, t.seanet.dilation_base as u32);
        b.add_u32(KEY_SEANET_N_RATIOS, t.seanet.ratios.len() as u32);
        for (i, r) in t.seanet.ratios.iter().enumerate() {
            b.add_u32(&format!("{PREFIX_SEANET_RATIO}{i}"), *r as u32);
        }
        b.add_u32(KEY_TRANSFORMER_D_MODEL, t.transformer.d_model as u32);
        b.add_u32(KEY_TRANSFORMER_N_HEAD, t.transformer.n_head as u32);
        b.add_u32(KEY_TRANSFORMER_N_LAYER, t.transformer.n_layer as u32);
        b.add_u32(KEY_TRANSFORMER_FF_DIM, t.transformer.ff_dim as u32);
        b.add_u32(KEY_TRANSFORMER_CONTEXT, t.transformer.context as u32);
        b.add_u32(KEY_TRANSFORMER_MAX_PERIOD, t.transformer.max_period as u32);
        b.add_f32(KEY_TRANSFORMER_LAYER_SCALE, t.transformer.layer_scale);
        b.add_u32(KEY_QUANTIZER_DIMENSION, t.quantizer.dimension as u32);
        b.add_u32(KEY_QUANTIZER_N_Q, t.quantizer.n_q as u32);
        b.add_u32(KEY_QUANTIZER_BINS, t.quantizer.bins as u32);
        b.add_u32(
            KEY_QUANTIZER_INPUT_DIMENSION,
            t.quantizer.input_dimension as u32,
        );
        b.add_u32(
            KEY_QUANTIZER_OUTPUT_DIMENSION,
            t.quantizer.output_dimension as u32,
        );
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let cfg = MimiNeuralConfig::from_gguf(&file).expect("from_gguf");
        assert_eq!(cfg, t);
    }

    #[test]
    fn ratio_index_hole_is_loud() {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SEANET_N_RATIOS, 2);
        b.add_u32(&format!("{PREFIX_SEANET_RATIO}0"), 2);
        // ratio.1 missing.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            MimiNeuralConfig::from_gguf(&file),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zero_placeholders_fail_validate() {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SEANET_N_RATIOS, 0);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let cfg = MimiNeuralConfig::from_gguf(&file).expect("from_gguf");
        assert!(matches!(
            cfg.validate(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transformer_width_must_match_seanet_dimension() {
        let mut cfg = MimiNeuralConfig::tiny_for_tests();
        cfg.transformer.d_model = 16;
        cfg.transformer.n_head = 4;
        assert!(matches!(
            cfg.validate(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn non_exact_resample_stride_is_rejected() {
        let mut cfg = MimiNeuralConfig::tiny_for_tests();
        cfg.seanet.ratios = vec![3]; // hop 3; frame hop 8 → 8 % 3 != 0
        assert!(matches!(
            cfg.validate(),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
