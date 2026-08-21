//! Canonical Charsiu English 10 ms forced-aligner checkpoint → GGUF.
//!
//! This converter intentionally accepts one topology only:
//! `charsiu/en_w2v2_fc_10ms` at HF revision
//! `e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f`. The source checkpoint is
//! `pytorch_model.bin` (SHA-256
//! `6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1`);
//! callers first flatten it with `tools/parity/nemo_pt_to_safetensors.py` so
//! torch pickle never enters the runtime.
//!
//! The converter verifies the complete 213-tensor manifest and every shape.
//! The training-only `wav2vec2.masked_spec_embed` is consumed but not emitted.
//! The positional convolution's `weight_norm(..., dim=2)` pair is folded into
//! `charsiu.pos_conv.weight`; all remaining tensors retain their upstream HF
//! names. A missing, extra, or reshaped tensor is a hard error.

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "charsiu";
const NAME: &str = "charsiu/en_w2v2_fc_10ms";
const CATEGORY: &str = "alignment";
const REVISION: &str = "e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f";
const CHECKPOINT_SHA256: &str = "6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1";

const HIDDEN: usize = 768;
const FFN: usize = 3072;
const N_LAYER: usize = 12;
const N_HEAD: usize = 12;
const VOCAB_SIZE: usize = 42;
const POS_KERNEL: usize = 128;
const POS_GROUPS: usize = 16;

const VOCAB: [&str; VOCAB_SIZE] = [
    "[SIL]", "NG", "F", "M", "AE", "R", "UW", "N", "IY", "AW", "V", "UH", "OW", "AA", "ER", "HH",
    "Z", "K", "CH", "W", "EY", "ZH", "T", "EH", "Y", "AH", "B", "P", "TH", "DH", "AO", "G", "L",
    "JH", "OY", "SH", "D", "AY", "S", "IH", "[UNK]", "[PAD]",
];

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct CharsiuReport {
    pub(crate) emitted: usize,
    pub(crate) consumed: usize,
}

pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CharsiuReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let mut consumed = vec![false; st.tensors().len()];
    let mut b = GgufBuilder::new();
    let mut report = CharsiuReport::default();

    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string("vokra.model.category", CATEGORY);
    b.add_string("vokra.charsiu.revision", REVISION);
    b.add_string("vokra.charsiu.checkpoint_sha256", CHECKPOINT_SHA256);
    b.add_u32("vokra.charsiu.hidden_size", HIDDEN as u32);
    b.add_u32("vokra.charsiu.ffn_dim", FFN as u32);
    b.add_u32("vokra.charsiu.n_layer", N_LAYER as u32);
    b.add_u32("vokra.charsiu.n_head", N_HEAD as u32);
    b.add_u32("vokra.charsiu.vocab_size", VOCAB_SIZE as u32);
    b.add_u32("vokra.charsiu.silence_id", 0);
    b.add_u32("vokra.charsiu.pad_id", 41);
    b.add_u32("vokra.charsiu.sample_rate", 16_000);
    b.add_f32("vokra.charsiu.frame_shift_sec", 0.01);
    b.add_f32("vokra.charsiu.layer_norm_eps", 1e-5);
    b.add_u32("vokra.charsiu.pos_conv_kernel", POS_KERNEL as u32);
    b.add_u32("vokra.charsiu.pos_conv_groups", POS_GROUPS as u32);
    b.add_u32("vokra.charsiu.silence_threshold", 4);
    b.add_metadata(
        "vokra.charsiu.vocab",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: VOCAB
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        }),
    );
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some("charsiu"),
        Some("charsiu/en_w2v2_fc_10ms"),
    );

    let mut take = |name: &str, dims: &[usize]| -> Result<Vec<f32>, ConvertError> {
        let (idx, info) = st
            .tensors()
            .iter()
            .enumerate()
            .find(|(_, t)| t.name == name)
            .ok_or_else(|| ConvertError::Parse(format!("charsiu: missing tensor `{name}`")))?;
        if consumed[idx] {
            return Err(ConvertError::Parse(format!(
                "charsiu: tensor `{name}` was consumed twice"
            )));
        }
        let want: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        if info.shape != want {
            return Err(ConvertError::Parse(format!(
                "charsiu: tensor `{name}` has shape {:?}, expected {want:?}",
                info.shape
            )));
        }
        if info.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "charsiu: canonical prepared tensor `{name}` must be F32, got {:?}",
                info.dtype
            )));
        }
        consumed[idx] = true;
        st.tensor_f32(name)
            .map_err(|e| ConvertError::Parse(format!("charsiu: reading `{name}`: {e}")))
    };

    let mut emit = |b: &mut GgufBuilder,
                    name: &str,
                    dims: &[usize],
                    data: &[f32]|
     -> Result<(), ConvertError> {
        b.add_tensor(
            name,
            GgmlType::F32,
            dims.iter().map(|&d| d as u64).collect(),
            data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )?;
        report.emitted += 1;
        Ok(())
    };

    let stem_kernels = [10usize, 3, 3, 3, 3, 2, 2];
    let mut in_ch = 1usize;
    for (i, &kernel) in stem_kernels.iter().enumerate() {
        let name = format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.weight");
        let data = take(&name, &[512, in_ch, kernel])?;
        emit(&mut b, &name, &[512, in_ch, kernel], &data)?;
        if i == 0 {
            for suffix in ["weight", "bias"] {
                let name = format!("wav2vec2.feature_extractor.conv_layers.0.layer_norm.{suffix}");
                let data = take(&name, &[512])?;
                emit(&mut b, &name, &[512], &data)?;
            }
        }
        in_ch = 512;
    }

    for (name, dims) in [
        ("wav2vec2.feature_projection.layer_norm.weight", vec![512]),
        ("wav2vec2.feature_projection.layer_norm.bias", vec![512]),
        (
            "wav2vec2.feature_projection.projection.weight",
            vec![HIDDEN, 512],
        ),
        ("wav2vec2.feature_projection.projection.bias", vec![HIDDEN]),
    ] {
        let data = take(name, &dims)?;
        emit(&mut b, name, &dims, &data)?;
    }

    let g_name = "wav2vec2.encoder.pos_conv_embed.conv.weight_g";
    let v_name = "wav2vec2.encoder.pos_conv_embed.conv.weight_v";
    let g = take(g_name, &[1, 1, POS_KERNEL])?;
    let v = take(v_name, &[HIDDEN, HIDDEN / POS_GROUPS, POS_KERNEL])?;
    let folded = fold_weight_norm_dim2(&g, &v, HIDDEN, HIDDEN / POS_GROUPS, POS_KERNEL)?;
    emit(
        &mut b,
        "charsiu.pos_conv.weight",
        &[HIDDEN, HIDDEN / POS_GROUPS, POS_KERNEL],
        &folded,
    )?;
    let pos_bias_name = "wav2vec2.encoder.pos_conv_embed.conv.bias";
    let pos_bias = take(pos_bias_name, &[HIDDEN])?;
    emit(&mut b, "charsiu.pos_conv.bias", &[HIDDEN], &pos_bias)?;

    for suffix in ["weight", "bias"] {
        let name = format!("wav2vec2.encoder.layer_norm.{suffix}");
        let data = take(&name, &[HIDDEN])?;
        emit(&mut b, &name, &[HIDDEN], &data)?;
    }

    for i in 0..N_LAYER {
        let p = format!("wav2vec2.encoder.layers.{i}");
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            for suffix in ["weight", "bias"] {
                let name = format!("{p}.attention.{projection}.{suffix}");
                let dims = if suffix == "weight" {
                    vec![HIDDEN, HIDDEN]
                } else {
                    vec![HIDDEN]
                };
                let data = take(&name, &dims)?;
                emit(&mut b, &name, &dims, &data)?;
            }
        }
        for norm in ["layer_norm", "final_layer_norm"] {
            for suffix in ["weight", "bias"] {
                let name = format!("{p}.{norm}.{suffix}");
                let data = take(&name, &[HIDDEN])?;
                emit(&mut b, &name, &[HIDDEN], &data)?;
            }
        }
        for (dense, dims) in [
            ("intermediate_dense", [FFN, HIDDEN]),
            ("output_dense", [HIDDEN, FFN]),
        ] {
            let w = format!("{p}.feed_forward.{dense}.weight");
            let data = take(&w, &dims)?;
            emit(&mut b, &w, &dims, &data)?;
            let bias_dims = [dims[0]];
            let bias = format!("{p}.feed_forward.{dense}.bias");
            let data = take(&bias, &bias_dims)?;
            emit(&mut b, &bias, &bias_dims, &data)?;
        }
    }

    for (name, dims) in [
        ("lm_head.weight", vec![VOCAB_SIZE, HIDDEN]),
        ("lm_head.bias", vec![VOCAB_SIZE]),
    ] {
        let data = take(name, &dims)?;
        emit(&mut b, name, &dims, &data)?;
    }

    // Training-time masking vector. It is part of the canonical manifest but
    // Wav2Vec2Model never reads it in eval mode when mask_time_indices is
    // absent, so consume it explicitly without shipping dead runtime weight.
    let _ = take("wav2vec2.masked_spec_embed", &[HIDDEN])?;

    let leftovers: Vec<&str> = consumed
        .iter()
        .enumerate()
        .filter(|&(_, used)| !*used)
        .map(|(i, _)| st.tensors()[i].name.as_str())
        .collect();
    if !leftovers.is_empty() {
        return Err(ConvertError::Parse(format!(
            "charsiu: {} unrecognized upstream tensor(s); refusing a partial conversion: {:?}",
            leftovers.len(),
            &leftovers[..leftovers.len().min(8)]
        )));
    }
    report.consumed = consumed.len();
    Ok((b, report))
}

fn fold_weight_norm_dim2(
    g: &[f32],
    v: &[f32],
    out: usize,
    in_per: usize,
    kernel: usize,
) -> Result<Vec<f32>, ConvertError> {
    let mut weight = vec![0.0f32; out * in_per * kernel];
    for k in 0..kernel {
        let mut squared = 0.0f64;
        for o in 0..out {
            for i in 0..in_per {
                let value = f64::from(v[(o * in_per + i) * kernel + k]);
                squared += value * value;
            }
        }
        let norm = squared.sqrt();
        if norm == 0.0 {
            return Err(ConvertError::Parse(format!(
                "charsiu: positional weight_v tap {k} has zero norm"
            )));
        }
        let scale = (f64::from(g[k]) / norm) as f32;
        for o in 0..out {
            for i in 0..in_per {
                let idx = (o * in_per + i) * kernel + k;
                weight[idx] = v[idx] * scale;
            }
        }
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_ids_match_the_official_release() {
        assert_eq!(VOCAB.len(), 42);
        assert_eq!(VOCAB[0], "[SIL]");
        assert_eq!(VOCAB[40], "[UNK]");
        assert_eq!(VOCAB[41], "[PAD]");
    }

    #[test]
    fn weight_norm_fold_matches_definition() {
        let g = [2.0, 3.0];
        let v = [3.0, 0.0, 4.0, 4.0]; // [out=2, in=1, k=2]
        let w = fold_weight_norm_dim2(&g, &v, 2, 1, 2).unwrap();
        assert!((w[0] - 1.2).abs() < 1e-6);
        assert_eq!(w[1], 0.0);
        assert!((w[2] - 1.6).abs() < 1e-6);
        assert!((w[3] - 3.0).abs() < 1e-6);
    }
}
