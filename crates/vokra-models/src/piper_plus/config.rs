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

use super::weights::TensorStore;

// --- fixed medium-config architecture constants (piper-plus vits/*.py) -------

/// Text-encoder / flow hidden size (`enc_p.emb.weight` is `[n_vocab, 192]`).
pub(crate) const HIDDEN: usize = 192;
/// Attention heads (`window_size` relative attention, `k_channels = hidden/2`);
/// the per-head channel count is derived ([`Dims::k_channels`]).
pub(crate) const N_HEADS: usize = 2;
/// FFN conv kernel (same-padding). The FFN inner *width* is shape-derived
/// ([`Dims::ffn`]); the kernel is fixed.
pub(crate) const FFN_KERNEL: usize = 3;
/// Relative-attention window (`emb_rel_k` is `[1, 2·4+1, 96]`).
pub(crate) const WINDOW_SIZE: usize = 4;
/// Global conditioning width (`emb_lang.weight` is `[n_lang, 512]`).
pub(crate) const GIN: usize = 512;

/// Flow WN conv kernel (`flow.flows.*.enc.in_layers.*` weight dim 2). The
/// coupling count, WN layer count and dilation base are shape-/architecture-
/// derived ([`Dims`]); the kernel is fixed across configs.
pub(crate) const FLOW_WN_KERNEL: usize = 5;

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
/// Decoder upsample transposed-conv stride, uniform across stages. The stride is
/// a `ConvTranspose1d` attribute that leaves no shape trace in the GGUF, so it is
/// not per-stage derivable (the converter baking non-uniform strides is an open
/// item, M4-RESIDUAL-B (A) open question 3). The per-stage **kernel** is now
/// shape-derived ([`Dims::dec_up_kernel`]) and the per-stage **pad** follows the
/// `(kernel − stride)/2` same-padding convention (`decoder.rs`); the shipping
/// css10 / v7 voices are the canonical kernel-16 / stride-4 / pad-6 geometry.
pub(crate) const DEC_UP_STRIDE: usize = 4;
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

/// Language id whose prosody features (A1/A2/A3) are honoured; every other
/// language gates them to zero (`Equal(lid, 0)` in the v7 graph). Consumed by
/// the duration-predictor prosody feed ([`super::ProsodyProj::channels`]);
/// declared here as the single baked source of truth.
pub(crate) const PROSODY_LANG_ID: i64 = 0;

/// Shape-derived model dimensions, read from the loaded voice's tensor shapes
/// rather than assumed — so a single-speaker medium voice and the zero-shot v7
/// voice (which adds `spk_proj` / FiLM / prosody) both resolve correctly. The
/// *fixed* architecture constants above (attention window/heads, ResBlock
/// kernels/dilations, upsample geometry, spline bins, LeakyReLU slope) are
/// identical across those configs and stay as consts.
#[derive(Debug, Clone)]
pub(crate) struct Dims {
    /// Global conditioning width `g` (`emb_lang.weight` dim 1).
    pub gin: usize,
    /// External speaker-embedding width (`spk_proj.0.weight` dim 1).
    pub spk_emb_dim: usize,
    /// Text-encoder / flow hidden size (`enc_p.emb.weight` dim 1).
    pub hidden: usize,
    /// Encoder transformer layers (`enc_p.encoder.attn_layers.*` count).
    pub n_enc_layers: usize,
    /// Encoder FFN inner width (`enc_p.encoder.ffn_layers.0.conv_1.weight` dim 0).
    pub ffn: usize,
    /// Duration-predictor input width (`dp.pre.weight` dim 0 = `hidden` + prosody).
    pub dp_filter: usize,
    /// Prosody feature width fed to the projection (`prosody_proj.weight` dim 0).
    pub prosody_in: usize,
    /// Prosody projection output width (`prosody_proj.weight` dim 1).
    pub prosody_out: usize,
    /// Decoder pre-conv output width (`dec.conv_pre.weight` dim 0).
    pub dec_initial: usize,
    /// Per-upsample output channel counts (`dec.ups.{i}.weight` dim 1).
    pub dec_up_out: Vec<usize>,
    /// Per-upsample transposed-conv kernel widths (`dec.ups.{i}.weight` dim 2).
    /// Shape-derived so a voice whose stages differ in kernel (the general
    /// MB-iSTFT geometry) loads without the former `DEC_UP_KERNEL` hard-assert;
    /// the shipping css10 / v7 voices are uniform 16 and reduce to the old const.
    pub dec_up_kernel: Vec<usize>,
    /// Upsample stage count (`dec_up_out.len()`).
    pub n_ups: usize,
    /// FiLM stage target channels `[dec_initial, dec_up_out...]` (conditioning
    /// is applied after `conv_pre` and after each upsample+MRF stage; also the
    /// per-upsample input widths).
    pub dec_channels: Vec<usize>,
    /// Flow coupling-layer count (`flow.flows.{0,2,4,...}` with a WN conditioner).
    pub flow_n_flows: usize,
    /// Flow WN dilated-conv layers per coupling (`flow.flows.0.enc.in_layers.*`).
    pub flow_wn_layers: usize,
    /// Flow WN dilation base: layer `i` uses `dilation = flow_wn_dilation_rate^i`.
    /// A Conv attribute (absent from the GGUF, so not shape-derivable); inferred
    /// from the architecture discriminator — the zero-shot v7 (`film`) flow uses
    /// `dilation_rate = 2` (dilations 1,2,4,8, the standard VITS WN), the legacy
    /// single-speaker additive flow uses `1` (every layer dilation 1). Verified
    /// against the v7 `flow_z` fixture (`parity_v7`).
    pub flow_wn_dilation_rate: usize,
    /// Decoder conditioning is multi-stage gated FiLM (`dec.cond_layers.*`
    /// present) rather than the single additive `x + cond(g)`.
    pub film: bool,
}

impl Dims {
    /// Derives the model dimensions from the loaded voice's tensor shapes.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if a shape-defining tensor is
    /// missing/degenerate or the shapes are internally inconsistent (a
    /// malformed voice fails loudly here rather than mid-forward).
    pub(crate) fn derive(store: &TensorStore) -> Result<Self> {
        let gin = axis(store, "emb_lang.weight", 1)?;
        let spk_emb_dim = axis(store, "spk_proj.0.weight", 1)?;
        let hidden = axis(store, "enc_p.emb.weight", 1)?;
        let ffn = axis(store, "enc_p.encoder.ffn_layers.0.conv_1.weight", 0)?;
        let dp_filter = axis(store, "dp.pre.weight", 0)?;
        let prosody_in = axis(store, "prosody_proj.weight", 0)?;
        let prosody_out = axis(store, "prosody_proj.weight", 1)?;
        let dec_initial = axis(store, "dec.conv_pre.weight", 0)?;

        let mut n_enc_layers = 0;
        while store
            .shape(&format!(
                "enc_p.encoder.attn_layers.{n_enc_layers}.conv_q.weight"
            ))
            .is_ok()
        {
            n_enc_layers += 1;
        }

        // Per-stage upsample geometry: out-channels (dim 1) and kernel (dim 2)
        // are both shape-derived; the stride is a ConvTranspose attribute that
        // leaves no shape trace (uniform `DEC_UP_STRIDE` today — see the module
        // note on `flow_wn_dilation_rate` for the same GGUF limitation).
        let mut dec_up_out = Vec::new();
        let mut dec_up_kernel = Vec::new();
        loop {
            let i = dec_up_out.len();
            let Ok(shape) = store.shape(&format!("dec.ups.{i}.weight")) else {
                break;
            };
            dec_up_out.push(*shape.get(1).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "piper voice: dec.ups.{i} weight shape {shape:?} lacks an out-channel axis"
                ))
            })?);
            dec_up_kernel.push(*shape.get(2).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "piper voice: dec.ups.{i} weight shape {shape:?} lacks a kernel axis"
                ))
            })?);
        }
        let n_ups = dec_up_out.len();
        let mut dec_channels = Vec::with_capacity(n_ups + 1);
        dec_channels.push(dec_initial);
        dec_channels.extend_from_slice(&dec_up_out);

        // Coupling blocks live at even flow indices (odds are `Flip`).
        let mut flow_n_flows = 0;
        while store
            .shape(&format!(
                "flow.flows.{}.enc.cond_layer.weight",
                2 * flow_n_flows
            ))
            .is_ok()
        {
            flow_n_flows += 1;
        }

        // WN dilated-conv layers per coupling block (first coupling is `flows.0`).
        let mut flow_wn_layers = 0;
        while store
            .shape(&format!(
                "flow.flows.0.enc.in_layers.{flow_wn_layers}.weight"
            ))
            .is_ok()
        {
            flow_wn_layers += 1;
        }

        let film = store.shape("dec.cond_layers.0.weight").is_ok();

        // The flow WN dilation base is a Conv attribute — not stored in the GGUF,
        // so not shape-derivable. It travels with the architecture: the zero-shot
        // v7 (FiLM) flow uses dilation_rate = 2 (per-layer dilations 1,2,4,8, the
        // standard VITS WN); the legacy single-speaker additive flow used 1.
        let flow_wn_dilation_rate = if film { 2 } else { 1 };

        // Internal-consistency checks (fail loudly on a malformed voice).
        if axis(store, "enc_p.cond_layer.weight", 1)? != gin {
            return Err(VokraError::InvalidArgument(
                "piper voice: enc_p.cond_layer conditioning width disagrees with emb_lang".into(),
            ));
        }
        if hidden % N_HEADS != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "piper voice: hidden {hidden} not divisible by {N_HEADS} attention heads"
            )));
        }
        if n_enc_layers == 0 || n_ups == 0 || flow_n_flows == 0 || flow_wn_layers == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "piper voice: degenerate structure (enc_layers={n_enc_layers}, ups={n_ups}, flows={flow_n_flows}, wn_layers={flow_wn_layers})"
            )));
        }
        // The duration-predictor input concatenates the encoder output with the
        // prosody channels, so its width must equal `hidden + prosody_out`; for
        // the medium config (identical for the single-speaker and zero-shot v7
        // voices) that is the documented `DP_FILTER = HIDDEN + PROSODY_DIM`. A
        // mismatch is a wrongly-shaped or non-medium voice — fail loudly here
        // rather than mid-forward.
        if dp_filter != hidden + prosody_out {
            return Err(VokraError::InvalidArgument(format!(
                "piper voice: dp.pre width {dp_filter} != encoder hidden {hidden} + prosody {prosody_out}"
            )));
        }
        if dp_filter != DP_FILTER || prosody_out != PROSODY_DIM {
            return Err(VokraError::InvalidArgument(format!(
                "piper voice: non-medium duration/prosody dims (dp_filter={dp_filter}, prosody_out={prosody_out}); expected {DP_FILTER} / {PROSODY_DIM}"
            )));
        }

        Ok(Self {
            gin,
            spk_emb_dim,
            hidden,
            n_enc_layers,
            ffn,
            dp_filter,
            prosody_in,
            prosody_out,
            dec_initial,
            dec_up_out,
            dec_up_kernel,
            n_ups,
            dec_channels,
            flow_n_flows,
            flow_wn_layers,
            flow_wn_dilation_rate,
            film,
        })
    }

    /// Per-head channel count (`hidden / N_HEADS`).
    pub(crate) fn k_channels(&self) -> usize {
        self.hidden / N_HEADS
    }
}

/// Reads dimension `i` of a tensor's stored shape, or a loud error.
fn axis(store: &TensorStore, name: &str, i: usize) -> Result<usize> {
    let shape = store.shape(name)?;
    shape.get(i).copied().ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "piper voice: tensor `{name}` shape {shape:?} has no axis {i}"
        ))
    })
}

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

    /// Total decoder upsample factor (samples per encoder frame): the product
    /// of the per-stage transposed-conv strides × the iSTFT hop × the PQMF
    /// sub-bands. For the shipping css10 / v7 voices `up_strides` is the two
    /// uniform stride-4 stages, giving `4·4·hop·subbands = 256`.
    ///
    /// `up_strides` is per-upsample-stage; the stride is a `ConvTranspose1d`
    /// attribute with no shape trace in the GGUF, so the caller supplies it
    /// (currently `[DEC_UP_STRIDE; n_ups]` — the decoder does not derive
    /// non-uniform strides until the converter bakes them, an open item). This
    /// signature was generalized from the former 2-stage constant form in
    /// M4-RESIDUAL-B (A); **there is no production caller** — `synthesize_phonemes`
    /// derives its frame counts from the flow output length, not this — so the
    /// change is an API-consistency fix, verified by the unit test below.
    pub fn samples_per_frame(&self, up_strides: &[usize]) -> usize {
        up_strides.iter().product::<usize>() * self.istft_hop * self.pqmf_subbands
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
        // samples_per_frame = Π(up_strides) · hop · subbands. Two uniform
        // stride-4 stages (the shipping css10 / v7 geometry) = 16 · 5 · 3,
        // reducing to the former 2-stage constant form.
        assert_eq!(
            cfg.samples_per_frame(&[DEC_UP_STRIDE, DEC_UP_STRIDE]),
            4 * 4 * 5 * 3
        );
        // A non-uniform 3-stage geometry (kernel/stride vary per stage) takes
        // the product: 4·4·2 · hop · subbands.
        assert_eq!(cfg.samples_per_frame(&[4, 4, 2]), 4 * 4 * 2 * 5 * 3);
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
