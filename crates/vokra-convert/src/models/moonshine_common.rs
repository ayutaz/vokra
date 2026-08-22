//! Shared strict manifest and metadata writer for the two official Moonshine releases.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub(crate) const ARCH: &str = "moonshine";
pub(crate) const CATEGORY: &str = "asr";
pub(crate) const DEFAULT_LICENSE_SPDX: &str = "mit";
pub(crate) const TOKENIZER_KEY: &str = "vokra.tokenizer.model";

#[derive(Debug, Clone, Copy)]
pub(crate) struct VariantSpec {
    pub name: &'static str,
    pub upstream_hf: &'static str,
    pub revision: &'static str,
    pub checkpoint_sha256: &'static str,
    pub tokenizer_sha256: &'static str,
    pub hidden: usize,
    pub intermediate: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub partial_rotary_factor: f32,
}

impl VariantSpec {
    pub(crate) const fn tensor_count(self) -> usize {
        10 + self.encoder_layers * 10 + self.decoder_layers * 15
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Report {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
    pub tokenizer_embedded: bool,
}

fn expected_manifest(spec: VariantSpec) -> Vec<(String, Vec<u64>)> {
    let d = spec.hidden as u64;
    let ff = spec.intermediate as u64;
    let mut out = vec![
        ("model.decoder.embed_tokens.weight".into(), vec![32_768, d]),
        ("model.decoder.norm.weight".into(), vec![d]),
        ("model.encoder.conv1.weight".into(), vec![d, 1, 127]),
        ("model.encoder.conv2.bias".into(), vec![2 * d]),
        ("model.encoder.conv2.weight".into(), vec![2 * d, d, 7]),
        ("model.encoder.conv3.bias".into(), vec![d]),
        ("model.encoder.conv3.weight".into(), vec![d, 2 * d, 3]),
        ("model.encoder.groupnorm.bias".into(), vec![d]),
        ("model.encoder.groupnorm.weight".into(), vec![d]),
        ("model.encoder.layer_norm.weight".into(), vec![d]),
    ];
    for layer in 0..spec.encoder_layers {
        let p = format!("model.encoder.layers.{layer}");
        out.extend([
            (format!("{p}.input_layernorm.weight"), vec![d]),
            (format!("{p}.post_attention_layernorm.weight"), vec![d]),
            (format!("{p}.self_attn.q_proj.weight"), vec![d, d]),
            (format!("{p}.self_attn.k_proj.weight"), vec![d, d]),
            (format!("{p}.self_attn.v_proj.weight"), vec![d, d]),
            (format!("{p}.self_attn.o_proj.weight"), vec![d, d]),
            (format!("{p}.mlp.fc1.weight"), vec![ff, d]),
            (format!("{p}.mlp.fc1.bias"), vec![ff]),
            (format!("{p}.mlp.fc2.weight"), vec![d, ff]),
            (format!("{p}.mlp.fc2.bias"), vec![d]),
        ]);
    }
    for layer in 0..spec.decoder_layers {
        let p = format!("model.decoder.layers.{layer}");
        out.extend([
            (format!("{p}.input_layernorm.weight"), vec![d]),
            (format!("{p}.post_attention_layernorm.weight"), vec![d]),
            (format!("{p}.final_layernorm.weight"), vec![d]),
            (format!("{p}.self_attn.q_proj.weight"), vec![d, d]),
            (format!("{p}.self_attn.k_proj.weight"), vec![d, d]),
            (format!("{p}.self_attn.v_proj.weight"), vec![d, d]),
            (format!("{p}.self_attn.o_proj.weight"), vec![d, d]),
            (format!("{p}.encoder_attn.q_proj.weight"), vec![d, d]),
            (format!("{p}.encoder_attn.k_proj.weight"), vec![d, d]),
            (format!("{p}.encoder_attn.v_proj.weight"), vec![d, d]),
            (format!("{p}.encoder_attn.o_proj.weight"), vec![d, d]),
            (format!("{p}.mlp.fc1.weight"), vec![2 * ff, d]),
            (format!("{p}.mlp.fc1.bias"), vec![2 * ff]),
            (format!("{p}.mlp.fc2.weight"), vec![d, ff]),
            (format!("{p}.mlp.fc2.bias"), vec![d]),
        ]);
    }
    out
}

fn validate_tokenizer(bytes: &[u8]) -> Result<(), ConvertError> {
    let root = vokra_core::json::parse(bytes)
        .map_err(|error| ConvertError::Parse(format!("moonshine tokenizer.json: {error}")))?;
    let model = root
        .get("model")
        .ok_or_else(|| ConvertError::Parse("moonshine tokenizer.json: missing `model`".into()))?;
    if model.get("type").and_then(|value| value.as_str()) != Some("BPE") {
        return Err(ConvertError::Parse(
            "moonshine tokenizer.json: `model.type` must be `BPE`".into(),
        ));
    }
    let vocab = model
        .get("vocab")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            ConvertError::Parse("moonshine tokenizer.json: missing object `model.vocab`".into())
        })?;
    if vocab.len() != 32_000 {
        return Err(ConvertError::Parse(format!(
            "moonshine tokenizer.json: model vocab has {} entries, expected 32000",
            vocab.len()
        )));
    }
    let mut pieces = vec![None; 32_768];
    for (text, id) in vocab {
        let id = id
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value < pieces.len())
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "moonshine tokenizer.json: invalid id for model piece {text:?}"
                ))
            })?;
        if pieces[id].replace(text.as_str()).is_some() {
            return Err(ConvertError::Parse(format!(
                "moonshine tokenizer.json: duplicate model-vocab id {id}"
            )));
        }
    }
    let added = root
        .get("added_tokens")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            ConvertError::Parse("moonshine tokenizer.json: missing `added_tokens` array".into())
        })?;
    if added.len() != 771 {
        return Err(ConvertError::Parse(format!(
            "moonshine tokenizer.json: added_tokens has {} entries, expected 771",
            added.len()
        )));
    }
    let mut seen_added = vec![false; pieces.len()];
    for token in added {
        let id = token
            .get("id")
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value < pieces.len())
            .ok_or_else(|| {
                ConvertError::Parse(
                    "moonshine tokenizer.json: added token has an invalid id".into(),
                )
            })?;
        if std::mem::replace(&mut seen_added[id], true) {
            return Err(ConvertError::Parse(format!(
                "moonshine tokenizer.json: duplicate added-token id {id}"
            )));
        }
        let text = token
            .get("content")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                ConvertError::Parse(
                    "moonshine tokenizer.json: added token is missing `content`".into(),
                )
            })?;
        if token.get("special") != Some(&vokra_core::json::JsonValue::Bool(true)) {
            return Err(ConvertError::Parse(format!(
                "moonshine tokenizer.json: added token id {id} is not special"
            )));
        }
        match pieces[id] {
            Some(existing) if existing != text => {
                return Err(ConvertError::Parse(format!(
                    "moonshine tokenizer.json: added token id {id} conflicts with model-vocab piece"
                )));
            }
            _ => pieces[id] = Some(text),
        }
    }
    if pieces.iter().any(Option::is_none) {
        return Err(ConvertError::Parse(
            "moonshine tokenizer.json does not cover every id in 0..32768".into(),
        ));
    }
    Ok(())
}

pub(crate) fn convert(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
    license: Option<&str>,
    spec: VariantSpec,
) -> Result<Report, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let expected = expected_manifest(spec);
    if st.tensors().len() != spec.tensor_count() {
        return Err(ConvertError::Parse(format!(
            "{}: source has {} tensors, expected exactly {} at revision {}",
            spec.name,
            st.tensors().len(),
            spec.tensor_count(),
            spec.revision
        )));
    }
    for (name, shape) in &expected {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == *name)
            .ok_or_else(|| {
                ConvertError::Parse(format!("{}: missing tensor `{name}`", spec.name))
            })?;
        if tensor.shape != *shape {
            return Err(ConvertError::Parse(format!(
                "{}: tensor `{name}` has shape {:?}, expected {shape:?}",
                spec.name, tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "{}: tensor `{name}` is {:?}, expected F32 from the pinned official checkpoint",
                spec.name, tensor.dtype
            )));
        }
    }
    for tensor in st.tensors() {
        if !expected.iter().any(|(name, _)| *name == tensor.name) {
            return Err(ConvertError::Parse(format!(
                "{}: unexpected tensor `{}` at pinned revision {}",
                spec.name, tensor.name, spec.revision
            )));
        }
    }

    let tokenizer_bytes = tokenizer.map(std::fs::read).transpose()?;
    if let Some(bytes) = tokenizer_bytes.as_deref() {
        validate_tokenizer(bytes)?;
    }

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, spec.name);
    builder.add_string("vokra.model.category", CATEGORY);
    builder.add_string("vokra.provenance.upstream_hf", spec.upstream_hf);
    builder.add_string("vokra.moonshine.revision", spec.revision);
    builder.add_string("vokra.moonshine.checkpoint_sha256", spec.checkpoint_sha256);
    builder.add_string("vokra.moonshine.tokenizer_sha256", spec.tokenizer_sha256);
    builder.add_u32("vokra.moonshine.sample_rate", 16_000);
    builder.add_u32("vokra.moonshine.hidden_size", spec.hidden as u32);
    builder.add_u32(
        "vokra.moonshine.intermediate_size",
        spec.intermediate as u32,
    );
    builder.add_u32("vokra.moonshine.encoder_layers", spec.encoder_layers as u32);
    builder.add_u32("vokra.moonshine.decoder_layers", spec.decoder_layers as u32);
    builder.add_u32("vokra.moonshine.encoder_heads", 8);
    builder.add_u32("vokra.moonshine.decoder_heads", 8);
    builder.add_u32("vokra.moonshine.vocab_size", 32_768);
    builder.add_u32("vokra.moonshine.max_positions", 194);
    builder.add_u32("vokra.moonshine.decoder_start_token_id", 1);
    builder.add_u32("vokra.moonshine.eos_token_id", 2);
    builder.add_f32("vokra.moonshine.rope_theta", 10_000.0);
    builder.add_f32(
        "vokra.moonshine.partial_rotary_factor",
        spec.partial_rotary_factor,
    );
    builder.add_string("vokra.moonshine.encoder_activation", "gelu");
    builder.add_string("vokra.moonshine.decoder_activation", "silu-swiglu");
    let (spdx, class) = match license {
        Some(value) if !value.is_empty() => {
            (value.to_owned(), LicenseClass::from_license_str(value))
        }
        _ => (DEFAULT_LICENSE_SPDX.into(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut builder,
        class,
        &spdx,
        Some(spec.name),
        Some(spec.upstream_hf),
    );
    if let Some(bytes) = tokenizer_bytes.as_deref() {
        builder.add_metadata(
            TOKENIZER_KEY,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: bytes.iter().copied().map(GgufMetadataValue::U8).collect(),
            }),
        );
    }
    for (name, shape) in &expected {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == *name)
            .expect("strict manifest validated above");
        builder.add_tensor(
            name,
            GgmlType::F32,
            shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(Report {
        read: expected.len(),
        written: expected.len(),
        skipped_non_float: 0,
        bf16_passthrough: 0,
        tokenizer_embedded: tokenizer_bytes.is_some(),
    })
}
