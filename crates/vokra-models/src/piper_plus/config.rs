//! piper-plus (MB-iSTFT-VITS2 medium) hyper-parameters (M0-07-T11).
//!
//! Runtime parameters (sample rate, noise/length scales, phoneme/language
//! tables, iSTFT/PQMF sizes) are read from the `vokra.piper.*` GGUF metadata
//! the converter wrote (M0-07-T06/T07) — never hard-coded. The fixed
//! architecture constants of the *medium* config (hidden size, head/layer
//! counts, flow/decoder structure) are recorded here with their sources and are
//! cross-checked against the loaded tensor shapes in
//! [`super::weights`](super::weights), so a mismatched voice fails loudly rather
//! than silently misreads weights.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

// --- fixed medium-config architecture constants (piper-plus vits/*.py) -------

/// Text-encoder / flow hidden size (`enc_p.emb.weight` is `[n_vocab, 192]`).
pub(crate) const HIDDEN: usize = 192;
/// Attention heads (`window_size` relative attention, `k_channels = 192/2`).
pub(crate) const N_HEADS: usize = 2;
/// Per-head channel count (`HIDDEN / N_HEADS`).
pub(crate) const K_CHANNELS: usize = HIDDEN / N_HEADS;
/// Encoder layers (`enc_p.encoder.attn_layers.0..5`).
pub(crate) const N_LAYERS: usize = 6;
/// FFN inner size (`ffn_layers.*.conv_1.weight` is `[768, 192, 3]`).
pub(crate) const FFN_CHANNELS: usize = 768;
/// FFN conv kernel (same-padding).
pub(crate) const FFN_KERNEL: usize = 3;
/// Relative-attention window (`emb_rel_k` is `[1, 2·4+1, 96]`).
pub(crate) const WINDOW_SIZE: usize = 4;
/// Global conditioning width (`emb_lang.weight` is `[n_lang, 512]`).
pub(crate) const GIN: usize = 512;

/// Flow coupling layers (`flow.flows.{0,2,4,6}`; the odds are `Flip`).
pub(crate) const FLOW_N_FLOWS: usize = 4;
/// Flow WN dilated-conv layers per coupling block.
pub(crate) const FLOW_WN_LAYERS: usize = 4;
/// Flow WN conv kernel.
pub(crate) const FLOW_WN_KERNEL: usize = 5;
/// Flow WN dilation (this voice exports every layer at dilation 1).
pub(crate) const FLOW_WN_DILATION: usize = 1;

/// Stochastic-duration-predictor hidden size (`dp.pre.weight` is
/// `[208, 208, 1]`; 208 = `HIDDEN` + `PROSODY_DIM`).
pub(crate) const DP_FILTER: usize = 208;
/// Prosody projection width (`prosody_proj` maps 3 → 16, concatenated onto the
/// encoder output for the duration predictor).
pub(crate) const PROSODY_DIM: usize = 16;
/// SDP DDSConv layers (per `convs` block).
pub(crate) const DP_CONV_LAYERS: usize = 3;
/// SDP DDSConv / ConvFlow kernel.
pub(crate) const DP_KERNEL: usize = 3;
/// Rational-quadratic-spline bins (`ConvFlow`).
pub(crate) const RQS_NUM_BINS: usize = 10;
/// Rational-quadratic-spline tail bound.
pub(crate) const RQS_TAIL_BOUND: f32 = 5.0;

/// Decoder pre-conv output width (`dec.conv_pre.weight` is `[256, 192, 7]`).
pub(crate) const DEC_INITIAL: usize = 256;
/// Decoder upsample kernel / stride / pad.
pub(crate) const DEC_UP_KERNEL: usize = 16;
pub(crate) const DEC_UP_STRIDE: usize = 4;
pub(crate) const DEC_UP_PAD: usize = 6;
/// ResBlock2 kernels (one MRF branch each).
pub(crate) const RESBLOCK_KERNELS: [usize; 3] = [3, 5, 7];
/// ResBlock2 dilation pairs, per kernel branch.
pub(crate) const RESBLOCK_DILATIONS: [[usize; 2]; 3] = [[1, 2], [2, 6], [3, 12]];
/// LeakyReLU slope used throughout the decoder (`mb_istft.py` `LRELU_SLOPE`).
pub(crate) const LRELU_SLOPE: f32 = 0.1;
/// PQMF filter taps and design (extracted as buffers, kept for reference).
pub(crate) const PQMF_TAPS: usize = 62;

/// LayerNorm epsilon (`nn.LayerNorm` default; VITS inherits it).
pub(crate) const LAYER_NORM_EPS: f32 = 1e-5;

/// Resolved runtime configuration read from the voice GGUF metadata.
#[derive(Debug, Clone)]
pub struct PiperConfig {
    /// Output PCM sample rate, Hz.
    pub sample_rate: u32,
    /// Phoneme embedding table size (`enc_p.emb.weight` rows).
    pub num_symbols: usize,
    /// Language embedding table size.
    pub num_languages: usize,
    /// Default z_p noise scale (0 = deterministic).
    pub noise_scale: f32,
    /// Default duration length scale.
    pub length_scale: f32,
    /// Default stochastic-duration noise scale (0 = deterministic).
    pub noise_w: f32,
    /// Decoder iSTFT FFT size.
    pub istft_n_fft: usize,
    /// Decoder iSTFT hop length.
    pub istft_hop: usize,
    /// PQMF sub-band count.
    pub pqmf_subbands: usize,
    /// Phoneme symbol per id (`vokra.piper.phoneme_symbols`), index = id.
    pub phoneme_symbols: Vec<String>,
    /// Language code per id (`vokra.piper.language_codes`), index = id.
    pub language_codes: Vec<String>,
}

impl PiperConfig {
    /// Reads the configuration from a loaded voice GGUF.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if a required `vokra.piper.*` key
    /// is missing or has the wrong type.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            sample_rate: u32(file, "vokra.piper.sample_rate")?,
            num_symbols: u32(file, "vokra.piper.num_symbols")? as usize,
            num_languages: u32(file, "vokra.piper.num_languages")? as usize,
            noise_scale: f32v(file, "vokra.piper.noise_scale")?,
            length_scale: f32v(file, "vokra.piper.length_scale")?,
            noise_w: f32v(file, "vokra.piper.noise_w")?,
            istft_n_fft: u32(file, "vokra.piper.istft.n_fft")? as usize,
            istft_hop: u32(file, "vokra.piper.istft.hop")? as usize,
            pqmf_subbands: u32(file, "vokra.piper.pqmf.subbands")? as usize,
            phoneme_symbols: string_array(file, "vokra.piper.phoneme_symbols")?,
            language_codes: string_array(file, "vokra.piper.language_codes")?,
        })
    }

    /// Language id for a code (`"ja"`, `"en"`, …), or `None` if absent.
    pub fn language_id(&self, code: &str) -> Option<i64> {
        self.language_codes
            .iter()
            .position(|c| c == code)
            .map(|i| i as i64)
    }

    /// Total decoder upsample factor (samples per encoder frame): the two
    /// stride-4 transposed convs × the iSTFT hop × the PQMF sub-bands = 256.
    pub fn samples_per_frame(&self) -> usize {
        DEC_UP_STRIDE * DEC_UP_STRIDE * self.istft_hop * self.pqmf_subbands
    }
}

fn get<'a>(file: &'a GgufFile, key: &str) -> Result<&'a GgufMetadataValue> {
    file.get(key)
        .ok_or_else(|| VokraError::InvalidArgument(format!("piper voice GGUF missing `{key}`")))
}

fn u32(file: &GgufFile, key: &str) -> Result<u32> {
    match get(file, key)? {
        GgufMetadataValue::U32(v) => Ok(*v),
        _ => Err(VokraError::InvalidArgument(format!(
            "`{key}` is not a UINT32"
        ))),
    }
}

fn f32v(file: &GgufFile, key: &str) -> Result<f32> {
    match get(file, key)? {
        GgufMetadataValue::F32(v) => Ok(*v),
        _ => Err(VokraError::InvalidArgument(format!(
            "`{key}` is not a FLOAT32"
        ))),
    }
}

fn string_array(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    let arr = get(file, key)?
        .as_array()
        .ok_or_else(|| VokraError::InvalidArgument(format!("`{key}` is not an array")))?;
    arr.values
        .iter()
        .map(|v| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                VokraError::InvalidArgument(format!("`{key}` has a non-string element"))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// A builder carrying all 11 `vokra.piper.*` keys, each numeric field a
    /// distinct value so a field-swap regression is caught.
    fn valid_builder() -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.piper.sample_rate", 22050);
        b.add_u32("vokra.piper.num_symbols", 256);
        b.add_u32("vokra.piper.num_languages", 2);
        b.add_u32("vokra.piper.istft.n_fft", 16);
        b.add_u32("vokra.piper.istft.hop", 5);
        b.add_u32("vokra.piper.pqmf.subbands", 3);
        b.add_f32("vokra.piper.noise_scale", 0.667);
        b.add_f32("vokra.piper.length_scale", 1.1);
        b.add_f32("vokra.piper.noise_w", 0.8);
        b.add_metadata(
            "vokra.piper.phoneme_symbols",
            str_array(&["_", "^", "$", "a"]),
        );
        b.add_metadata("vokra.piper.language_codes", str_array(&["ja", "en"]));
        b
    }

    fn config_of(b: &GgufBuilder) -> Result<PiperConfig> {
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        PiperConfig::from_gguf(&file)
    }

    #[test]
    fn parses_every_field_and_derives_lookups() {
        let cfg = config_of(&valid_builder()).expect("valid config");
        assert_eq!(cfg.sample_rate, 22050);
        assert_eq!(cfg.num_symbols, 256);
        assert_eq!(cfg.num_languages, 2);
        assert_eq!(cfg.istft_n_fft, 16);
        assert_eq!(cfg.istft_hop, 5);
        assert_eq!(cfg.pqmf_subbands, 3);
        assert_eq!(cfg.noise_scale, 0.667);
        assert_eq!(cfg.length_scale, 1.1);
        assert_eq!(cfg.noise_w, 0.8);
        assert_eq!(cfg.phoneme_symbols, ["_", "^", "$", "a"]);
        assert_eq!(cfg.language_codes, ["ja", "en"]);
        // Language id = position in the code table; absent code = None.
        assert_eq!(cfg.language_id("ja"), Some(0));
        assert_eq!(cfg.language_id("en"), Some(1));
        assert_eq!(cfg.language_id("zz"), None);
        // samples_per_frame = DEC_UP_STRIDE^2 · hop · subbands = 16 · 5 · 3.
        assert_eq!(cfg.samples_per_frame(), 4 * 4 * 5 * 3);
    }

    #[test]
    fn missing_key_fails_with_missing_message() {
        // Everything except sample_rate.
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.piper.num_symbols", 256);
        b.add_u32("vokra.piper.num_languages", 2);
        b.add_u32("vokra.piper.istft.n_fft", 16);
        b.add_u32("vokra.piper.istft.hop", 5);
        b.add_u32("vokra.piper.pqmf.subbands", 3);
        b.add_f32("vokra.piper.noise_scale", 0.667);
        b.add_f32("vokra.piper.length_scale", 1.1);
        b.add_f32("vokra.piper.noise_w", 0.8);
        b.add_metadata("vokra.piper.phoneme_symbols", str_array(&["_"]));
        b.add_metadata("vokra.piper.language_codes", str_array(&["ja"]));
        match config_of(&b) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("missing"), "message was: {msg}");
            }
            other => panic!("expected InvalidArgument(missing), got {other:?}"),
        }
    }

    #[test]
    fn wrong_type_for_scalar_key_is_rejected() {
        // sample_rate written as FLOAT32 instead of UINT32 (add_metadata
        // overwrites the earlier u32 in place).
        let mut b = valid_builder();
        b.add_f32("vokra.piper.sample_rate", 22050.0);
        assert!(matches!(config_of(&b), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn non_array_for_table_key_is_rejected() {
        // phoneme_symbols written as a scalar u32 instead of a string array.
        let mut b = valid_builder();
        b.add_u32("vokra.piper.phoneme_symbols", 3);
        assert!(matches!(config_of(&b), Err(VokraError::InvalidArgument(_))));
    }
}
