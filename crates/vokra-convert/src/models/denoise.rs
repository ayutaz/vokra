//! DeepFilterNet3 `denoise` → `vokra.denoise.*` GGUF conversion (M4-20
//! T12/T17).
//!
//! The GGUF schema (config keys + upstream-named F32 tensors) and the
//! runtime loader live in `vokra-ops::denoise`; this module is the
//! `vokra-convert` offline entry.
//!
//! # Input
//!
//! A safetensors flatten of the released checkpoint, produced by
//! `tools/parity/dfn3_prepare_checkpoint.py` (verbatim state-dict keys; the
//! I64 `*.num_batches_tracked` BatchNorm training counters are dropped
//! there with an explicit report). This module then:
//!
//! * validates every manifest tensor ([`vokra_ops::denoise_tensor_manifest`])
//!   is present with the EXACT declared shape (dims, not just element
//!   count) — a missing or mis-shaped tensor is a hard error;
//! * skips exactly the documented dead tensors
//!   ([`vokra_ops::denoise_skipped_checkpoint_tensors`]: `erb_fb`,
//!   `df_dec.df_fc_a.*` — never read by the DFN3 inference graph) and hard
//!   errors on any *other* unconsumed tensor (checkpoint layout drift);
//! * writes the `vokra.denoise.*` config metadata + tensors and re-parses
//!   its own output through [`DenoiseModel::from_gguf`] as a final
//!   loadability gate (nothing is emitted that the runtime cannot bind).

use std::path::Path;

use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};
use vokra_core::safetensors::SafetensorsFile;
use vokra_core::{Result, VokraError};
use vokra_ops::denoise::DeepFilterNetConfig;
use vokra_ops::{
    DenoiseModel, denoise_skipped_checkpoint_tensors, denoise_synthesized_tensors,
    denoise_tensor_manifest,
};

/// Builder-producing core of the real conversion (shared by
/// [`convert_denoise_bytes`] and the umbrella `convert_file` arm). Returns
/// the populated builder + the number of tensors written; the emitted GGUF
/// is already loadability-checked through [`DenoiseModel::from_gguf`].
pub(crate) fn convert_builder(data: Vec<u8>) -> Result<(GgufBuilder, usize)> {
    let st = SafetensorsFile::parse(data)?;
    let cfg = DeepFilterNetConfig::deep_filter_net3();
    let manifest = denoise_tensor_manifest(&cfg);

    let mut b = GgufBuilder::new();
    cfg.write_gguf_metadata(&mut b);
    for spec in &manifest {
        let info = st.tensor_info(&spec.name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "denoise convert: checkpoint is missing tensor `{}`",
                spec.name
            ))
        })?;
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        if shape != spec.shape {
            return Err(VokraError::ModelLoad(format!(
                "denoise convert: tensor `{}` has shape {shape:?}, expected {:?}",
                spec.name, spec.shape
            )));
        }
        let data = st.tensor_f32(&spec.name)?;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut dims: Vec<u64> = spec.shape.iter().map(|&d| d as u64).collect();
        dims.reverse(); // GGUF stores dims innermost-first
        b.add_tensor(&spec.name, GgmlType::F32, dims, bytes)
            .map_err(|e| VokraError::ModelLoad(format!("denoise convert: {e}")))?;
    }

    // Completeness sweep: every checkpoint tensor must be either consumed
    // (manifest) or on the documented dead-tensor skip list.
    let skip = denoise_skipped_checkpoint_tensors();
    for info in st.tensors() {
        let name = info.name.as_str();
        let consumed = manifest.iter().any(|s| s.name == name);
        if !consumed && !skip.contains(&name) {
            return Err(VokraError::ModelLoad(format!(
                "denoise convert: unexpected checkpoint tensor `{name}` (layout drift?)"
            )));
        }
    }

    // Loadability gate: the emitted GGUF must bind through the runtime
    // loader (which re-validates every tensor + the config).
    let bytes = b
        .to_bytes()
        .map_err(|e| VokraError::ModelLoad(format!("denoise convert: {e}")))?;
    let gguf = GgufFile::parse(bytes)
        .map_err(|e| VokraError::ModelLoad(format!("denoise convert self-check: {e}")))?;
    DenoiseModel::from_gguf(&gguf)?;
    Ok((b, manifest.len()))
}

/// Converts a prepared DeepFilterNet3 safetensors byte buffer into
/// `vokra.denoise.*` GGUF bytes (published DFN3 hyper-parameters).
///
/// # Errors
///
/// [`VokraError::ModelLoad`] for a missing / mis-shaped / unknown tensor or
/// an unparsable input.
pub fn convert_denoise_bytes(data: Vec<u8>) -> Result<Vec<u8>> {
    let (b, _) = convert_builder(data)?;
    b.to_bytes()
        .map_err(|e| VokraError::ModelLoad(format!("denoise convert: {e}")))
}

/// File-path wrapper over [`convert_denoise_bytes`]
/// (`vokra-cli convert --model denoise`).
///
/// # Errors
///
/// Propagates I/O and [`convert_denoise_bytes`] errors.
pub fn convert_denoise_file(input: &Path, output: &Path) -> Result<()> {
    let data = std::fs::read(input).map_err(VokraError::Io)?;
    let bytes = convert_denoise_bytes(data)?;
    std::fs::write(output, bytes).map_err(VokraError::Io)?;
    Ok(())
}

/// Builds a **synthetic** real-topology DFN3 model for `cfg` and writes its
/// `vokra.denoise.*` GGUF (offline round-trip / demo path — NOT a trained
/// network; real weights come from the released checkpoint via
/// [`convert_denoise_file`]).
///
/// # Errors
///
/// Propagates GGUF write / loadability errors.
pub fn convert_denoise_synthetic(cfg: DeepFilterNetConfig, seed: u64) -> Result<Vec<u8>> {
    let mut b = GgufBuilder::new();
    cfg.write_gguf_metadata(&mut b);
    for (spec, data) in denoise_synthesized_tensors(&cfg, seed) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut dims: Vec<u64> = spec.shape.iter().map(|&d| d as u64).collect();
        dims.reverse();
        b.add_tensor(&spec.name, GgmlType::F32, dims, bytes)
            .map_err(|e| VokraError::ModelLoad(format!("denoise convert: {e}")))?;
    }
    let bytes = b
        .to_bytes()
        .map_err(|e| VokraError::ModelLoad(format!("denoise convert: {e}")))?;
    let gguf = GgufFile::parse(bytes.clone())
        .map_err(|e| VokraError::ModelLoad(format!("denoise convert self-check: {e}")))?;
    DenoiseModel::from_gguf(&gguf)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> DeepFilterNetConfig {
        DeepFilterNetConfig {
            n_fft: 64,
            hop: 32,
            sample_rate: 16000,
            n_erb: 8,
            df_bins: 12,
            df_order: 3,
            min_nb_erb_freqs: 1,
            conv_lookahead: 1,
            df_lookahead: 1,
            conv_ch: 8,
            emb_hidden: 16,
            df_hidden: 16,
            enc_linear_groups: 4,
            linear_groups: 4,
            df_gru_linear_groups: 2,
            emb_num_layers: 3,
            df_num_layers: 2,
            lsnr_min: -15.0,
            lsnr_max: 35.0,
            norm_alpha: 0.99,
        }
    }

    #[test]
    fn synthetic_conversion_round_trips_through_gguf() {
        // Offline path: config → synthetic real-topology model → GGUF bytes
        // → parse → bind → enhance runs (real-checkpoint numeric parity is
        // the env-gated suite in vokra-ops).
        let cfg = small_cfg();
        let bytes = convert_denoise_synthetic(cfg, 11).unwrap();
        let gguf = GgufFile::parse(bytes).unwrap();
        let model = DenoiseModel::from_gguf(&gguf).unwrap();
        assert_eq!(model.config(), &cfg);
        let noisy: Vec<f32> = (0..1024).map(|i| 0.1 * (i as f32 * 0.07).sin()).collect();
        let enhanced = model.enhance(&noisy).unwrap();
        assert_eq!(enhanced.len(), noisy.len());
    }

    /// Hand-rolled minimal safetensors buffer (header-len + JSON + data).
    fn safetensors_from(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut blob: Vec<u8> = Vec::new();
        for (i, (name, shape, data)) in tensors.iter().enumerate() {
            let start = blob.len();
            for v in data {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            let end = blob.len();
            let dims = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{dims}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&blob);
        out
    }

    #[test]
    fn real_conversion_rejects_missing_and_unknown_tensors() {
        // A safetensors with only one (valid-shaped) tensor → missing-tensor
        // hard error naming a manifest entry.
        let st = safetensors_from(&[(
            "enc.erb_conv0.1.weight",
            vec![64, 1, 3, 3],
            vec![0.0; 64 * 9],
        )]);
        let err = convert_denoise_bytes(st).unwrap_err();
        assert!(err.to_string().contains("missing tensor"), "{err}");

        // A wrong-shaped manifest tensor → shape hard error.
        let st = safetensors_from(&[("enc.erb_conv0.1.weight", vec![64, 9], vec![0.0; 64 * 9])]);
        let err = convert_denoise_bytes(st).unwrap_err();
        assert!(err.to_string().contains("shape"), "{err}");
    }
}
