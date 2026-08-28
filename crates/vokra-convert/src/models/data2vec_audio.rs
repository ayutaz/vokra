//! Dedicated Data2Vec Audio Base 960h safetensors-to-GGUF converter.
//!
//! Data2Vec Audio is not a Wav2Vec2 alias: the released checkpoint uses
//! `data2vec_audio.*` names, LayerNorm on all seven waveform convolutions,
//! and five kernel-19 positional-convolution layers. This converter fixes
//! the legacy public stamp while preserving every float tensor verbatim.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "data2vec_audio";
pub const NAME: &str = "data2vec-audio-base-960h";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "facebook/data2vec-audio-base-960h";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const SOURCE: &str =
    "facebook/data2vec-audio-base-960h@32331f3123e703528918aa688a9a38232d58c872 (apache-2.0)";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM: &str = "vokra.provenance.upstream_hf";
const PREFIX: &str = "vokra.data2vec_audio";
const CONV_DIM: [u32; 7] = [512; 7];
const CONV_KERNEL: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Data2VecAudioReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_data2vec_audio_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Data2VecAudioReport, ConvertError> {
    let st = SafetensorsFile::parse(std::fs::read(input)?)?;
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_CATEGORY, CATEGORY);
    let (spdx, class) = match license {
        Some(value) if !value.is_empty() => {
            (value.to_owned(), LicenseClass::from_license_str(value))
        }
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut builder, class, &spdx, Some(NAME), Some(SOURCE));
    builder.add_string(KEY_UPSTREAM, UPSTREAM_HF);

    builder.add_u32(&format!("{PREFIX}.hidden_size"), 768);
    builder.add_u32(&format!("{PREFIX}.n_layer"), 12);
    builder.add_u32(&format!("{PREFIX}.n_head"), 12);
    builder.add_u32(&format!("{PREFIX}.intermediate_size"), 3072);
    builder.add_u32(&format!("{PREFIX}.vocab_size"), 32);
    builder.add_f32(&format!("{PREFIX}.layer_norm_eps"), 1e-5);
    builder.add_u32(&format!("{PREFIX}.num_conv_pos_embeddings"), 5);
    builder.add_u32(&format!("{PREFIX}.conv_pos_kernel_size"), 19);
    builder.add_u32(&format!("{PREFIX}.num_conv_pos_embedding_groups"), 16);
    builder.add_u32(&format!("{PREFIX}.num_feat_extract_layers"), 7);
    builder.add_bool(&format!("{PREFIX}.conv_bias"), false);
    write_u32_array(&mut builder, &format!("{PREFIX}.conv_dim"), &CONV_DIM);
    write_u32_array(&mut builder, &format!("{PREFIX}.conv_kernel"), &CONV_KERNEL);
    write_u32_array(&mut builder, &format!("{PREFIX}.conv_stride"), &CONV_STRIDE);

    let mut report = Data2VecAudioReport::default();
    for tensor in st.tensors() {
        report.read += 1;
        match tensor.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                builder
                    .add_tensor(
                        &tensor.name,
                        tensor.dtype,
                        tensor.shape.clone(),
                        st.tensor_bytes(tensor).to_vec(),
                    )
                    .map_err(|error| ConvertError::Gguf(error.to_string()))?;
                report.written += 1;
                report.bf16_passthrough += usize::from(tensor.dtype == GgmlType::BF16);
            }
            _ => report.skipped_non_float += 1,
        }
    }
    std::fs::write(
        output,
        builder
            .to_bytes()
            .map_err(|error| ConvertError::Gguf(error.to_string()))?,
    )?;
    Ok(report)
}

fn write_u32_array(builder: &mut GgufBuilder, key: &str, values: &[u32]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().copied().map(GgufMetadataValue::U32).collect(),
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    #[test]
    fn stamps_the_distinct_data2vec_contract() {
        let header = r#"{"data2vec_audio.masked_spec_embed":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut source = Vec::new();
        source.extend_from_slice(&(header.len() as u64).to_le_bytes());
        source.extend_from_slice(header.as_bytes());
        source.extend_from_slice(&1.0f32.to_le_bytes());
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let tag = format!(
            "{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let input = std::env::temp_dir().join(format!("data2vec-{tag}.safetensors"));
        let output = std::env::temp_dir().join(format!("data2vec-{tag}.gguf"));
        std::fs::write(&input, source).unwrap();
        let report = convert_data2vec_audio_file(&input, &output, None).unwrap();
        assert_eq!((report.read, report.written), (1, 1));
        let file = GgufFile::open(&output).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get("vokra.data2vec_audio.num_conv_pos_embeddings")
                .and_then(|value| value.as_u64()),
            Some(5)
        );
        assert_eq!(
            file.get("vokra.data2vec_audio.conv_pos_kernel_size")
                .and_then(|value| value.as_u64()),
            Some(19)
        );
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }
}
