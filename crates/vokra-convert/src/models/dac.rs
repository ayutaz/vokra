//! DAC (Descript Audio Codec) checkpoint → GGUF conversion (M4-04 T11).
//!
//! # Input
//!
//! A **prepared** safetensors checkpoint + a JSON config side-car, both
//! produced offline by `tools/parity/dac_prepare_checkpoint.py` from the
//! upstream `.pth` release (torch-pickle never enters the converter —
//! zero-dep, NFR-DS-02; `kokoro_prepare_checkpoint.py` precedent):
//!
//! - safetensors: the flattened upstream `state_dict` (moshi-style naming
//!   `quantizer.quantizers.{i}.codebook.weight` /
//!   `.out_proj.{weight_g,weight_v,bias}` / `.in_proj.*` + encoder/decoder
//!   chain);
//! - config JSON: the checkpoint's `metadata.kwargs`-derived shape facts —
//!   `{"n_codebooks", "codebook_size", "codebook_dim", "d_model",
//!   "sample_rate", "hop_length"}` (all required; the zoo-primary 24 kHz
//!   variant is 32 / 1024 / 8 / 1024 / 24000 / 320, verified in ADR M4-04
//!   §T02).
//!
//! # What is written
//!
//! 1. **Every upstream tensor pass-through** (decoder chain included — ADR
//!    M4-04 §D-f forward-compat for the future feature→PCM consumer WP).
//! 2. **Derived decode-ready tensors** per quantizer `i` (the runtime binder
//!    consumes these; upstream names stay untouched next to them):
//!    - `vokra.dac.quantizer.{i}.codebook` — `[codebook_size, codebook_dim]`
//!      (verbatim copy of `codebook.weight` under the stable runtime name);
//!    - `vokra.dac.quantizer.{i}.out_proj_weight` — `[d_model, codebook_dim]`
//!      with the **weight norm folded offline**:
//!      `W[o,:] = g[o] * v[o,:] / ||v[o,:]||₂` (torch `weight_norm`,
//!      `dim=0` — norm over all dims except the output channel);
//!    - `vokra.dac.quantizer.{i}.out_proj_bias` — `[d_model]` (verbatim; DAC's
//!      `WNConv1d` keeps `nn.Conv1d`'s default `bias=True`).
//! 3. `vokra.dac.{n_codebooks,codebook_size,codebook_dim,d_model,sample_rate,
//!    hop_length}` metadata (config-driven, cross-checked against tensor
//!    shapes — a mismatch is an explicit error, FR-EX-08).
//! 4. `vokra.provenance.*`: `model_id = "dac"` → `Permissive` (MIT).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::json::{self, JsonValue};
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for standalone DAC codec GGUFs.
pub(crate) const ARCH: &str = "dac";
/// `vokra.model.name` value.
const NAME: &str = "DAC (Descript Audio Codec)";

const KEY_N_CODEBOOKS: &str = "vokra.dac.n_codebooks";
const KEY_CODEBOOK_SIZE: &str = "vokra.dac.codebook_size";
const KEY_CODEBOOK_DIM: &str = "vokra.dac.codebook_dim";
const KEY_D_MODEL: &str = "vokra.dac.d_model";
const KEY_SAMPLE_RATE: &str = "vokra.dac.sample_rate";
const KEY_HOP_LENGTH: &str = "vokra.dac.hop_length";

/// Parsed DAC config side-car (all six fields required — the prepare script
/// emits them from the checkpoint's own metadata, nothing invented).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DacConfig {
    pub(crate) n_codebooks: usize,
    pub(crate) codebook_size: usize,
    pub(crate) codebook_dim: usize,
    pub(crate) d_model: usize,
    pub(crate) sample_rate: u32,
    pub(crate) hop_length: u32,
}

impl DacConfig {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
        let req = |key: &str| -> Result<u64, ConvertError> {
            root.get(key).and_then(JsonValue::as_u64).ok_or_else(|| {
                ConvertError::Parse(format!(
                    "dac config: required non-negative integer field `{key}` missing (the \
                         dac_prepare_checkpoint.py side-car emits it from the checkpoint's \
                         metadata.kwargs)"
                ))
            })
        };
        let cfg = Self {
            n_codebooks: req("n_codebooks")? as usize,
            codebook_size: req("codebook_size")? as usize,
            codebook_dim: req("codebook_dim")? as usize,
            d_model: req("d_model")? as usize,
            sample_rate: req("sample_rate")? as u32,
            hop_length: req("hop_length")? as u32,
        };
        if cfg.n_codebooks == 0
            || cfg.codebook_size == 0
            || cfg.codebook_dim == 0
            || cfg.d_model == 0
            || cfg.sample_rate == 0
            || cfg.hop_length == 0
        {
            return Err(ConvertError::Parse(
                "dac config: every field must be > 0".to_owned(),
            ));
        }
        Ok(cfg)
    }
}

/// Conversion report.
#[derive(Debug, Default)]
pub(crate) struct DacReport {
    /// Upstream tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32/F16 tensors skipped (defensive).
    pub(crate) skipped_non_float: usize,
}

/// Converts a prepared DAC safetensors buffer + config side-car into a
/// populated GGUF builder.
pub(crate) fn convert(
    bytes: Vec<u8>,
    config: &DacConfig,
) -> Result<(GgufBuilder, DacReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let find = |name: &str| st.tensors().iter().find(|t| t.name == name);

    let f32_tensor = |name: &str| -> Result<(Vec<u64>, Vec<f32>), ConvertError> {
        let t = find(name).ok_or_else(|| {
            ConvertError::Parse(format!(
                "dac: required tensor `{name}` not found — not a \
                                         prepared DAC checkpoint (run \
                                         tools/parity/dac_prepare_checkpoint.py first)"
            ))
        })?;
        if t.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "dac: tensor `{name}` must be F32, got {:?}",
                t.dtype
            )));
        }
        let vals = st
            .tensor_bytes(t)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((t.shape.clone(), vals))
    };

    // ---- Assemble ------------------------------------------------------------
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_u32(KEY_N_CODEBOOKS, config.n_codebooks as u32);
    b.add_u32(KEY_CODEBOOK_SIZE, config.codebook_size as u32);
    b.add_u32(KEY_CODEBOOK_DIM, config.codebook_dim as u32);
    b.add_u32(KEY_D_MODEL, config.d_model as u32);
    b.add_u32(KEY_SAMPLE_RATE, config.sample_rate);
    b.add_u32(KEY_HOP_LENGTH, config.hop_length);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some("dac"),
        Some(
            "descriptinc/descript-audio-codec GitHub release (prepared via \
              dac_prepare_checkpoint.py)",
        ),
    );

    let mut report = DacReport::default();
    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
            }
            _ => report.skipped_non_float += 1,
        }
    }

    // ---- Derived per-quantizer decode tensors --------------------------------
    for i in 0..config.n_codebooks {
        let prefix = format!("quantizer.quantizers.{i}");

        let (cb_shape, cb_vals) = f32_tensor(&format!("{prefix}.codebook.weight"))?;
        if cb_shape != vec![config.codebook_size as u64, config.codebook_dim as u64] {
            return Err(ConvertError::Parse(format!(
                "dac: {prefix}.codebook.weight shape {cb_shape:?} != config \
                 [{}, {}]",
                config.codebook_size, config.codebook_dim
            )));
        }

        let (g_shape, g_vals) = f32_tensor(&format!("{prefix}.out_proj.weight_g"))?;
        let (v_shape, v_vals) = f32_tensor(&format!("{prefix}.out_proj.weight_v"))?;
        let (bias_shape, bias_vals) = f32_tensor(&format!("{prefix}.out_proj.bias"))?;
        if g_shape != vec![config.d_model as u64, 1, 1] {
            return Err(ConvertError::Parse(format!(
                "dac: {prefix}.out_proj.weight_g shape {g_shape:?} != [{}, 1, 1]",
                config.d_model
            )));
        }
        if v_shape != vec![config.d_model as u64, config.codebook_dim as u64, 1] {
            return Err(ConvertError::Parse(format!(
                "dac: {prefix}.out_proj.weight_v shape {v_shape:?} != [{}, {}, 1]",
                config.d_model, config.codebook_dim
            )));
        }
        if bias_shape != vec![config.d_model as u64] {
            return Err(ConvertError::Parse(format!(
                "dac: {prefix}.out_proj.bias shape {bias_shape:?} != [{}]",
                config.d_model
            )));
        }

        // Weight-norm fold: W[o,:] = g[o] * v[o,:] / ||v[o,:]||₂ (torch
        // weight_norm dim=0 — norm over the non-output dims; kernel_size is
        // 1 so the row is exactly codebook_dim long). FP32 accumulate.
        let dim = config.codebook_dim;
        let mut folded = vec![0.0_f32; config.d_model * dim];
        for o in 0..config.d_model {
            let v_row = &v_vals[o * dim..(o + 1) * dim];
            let norm = v_row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                return Err(ConvertError::Parse(format!(
                    "dac: {prefix}.out_proj.weight_v row {o} has zero norm — cannot fold \
                     weight_norm (corrupt checkpoint?)"
                )));
            }
            let scale = g_vals[o] / norm;
            for c in 0..dim {
                folded[o * dim + c] = v_row[c] * scale;
            }
        }

        let to_bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        b.add_tensor(
            &format!("vokra.dac.quantizer.{i}.codebook"),
            GgmlType::F32,
            vec![config.codebook_size as u64, config.codebook_dim as u64],
            to_bytes(&cb_vals),
        )?;
        b.add_tensor(
            &format!("vokra.dac.quantizer.{i}.out_proj_weight"),
            GgmlType::F32,
            vec![config.d_model as u64, config.codebook_dim as u64],
            to_bytes(&folded),
        )?;
        b.add_tensor(
            &format!("vokra.dac.quantizer.{i}.out_proj_bias"),
            GgmlType::F32,
            vec![config.d_model as u64],
            to_bytes(&bias_vals),
        )?;
    }

    Ok((b, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn build_safetensors(entries: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut data = Vec::<u8>::new();
        for (i, (name, shape, vals)) in entries.iter().enumerate() {
            let start = data.len();
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            if i > 0 {
                header.push(',');
            }
            let dims = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{start},{end}]}}"#
            ));
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    /// 2 quantizers, codebook_size 3, codebook_dim 2, d_model 4.
    fn tiny_config() -> DacConfig {
        DacConfig {
            n_codebooks: 2,
            codebook_size: 3,
            codebook_dim: 2,
            d_model: 4,
            sample_rate: 24000,
            hop_length: 320,
        }
    }

    fn synthetic_dac(cfg: &DacConfig) -> Vec<u8> {
        let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for i in 0..cfg.n_codebooks {
            let prefix = format!("quantizer.quantizers.{i}");
            let cb: Vec<f32> = (0..cfg.codebook_size * cfg.codebook_dim)
                .map(|k| k as f32 + i as f32 * 10.0)
                .collect();
            entries.push((
                format!("{prefix}.codebook.weight"),
                vec![cfg.codebook_size, cfg.codebook_dim],
                cb,
            ));
            // weight_v rows: [3, 4] scaled per row (norm 5 * row_scale);
            // weight_g row o: o+1  →  folded W[o,:] = (o+1) * [3,4]/5.
            let mut v = Vec::new();
            for _o in 0..cfg.d_model {
                v.extend_from_slice(&[3.0, 4.0]);
            }
            entries.push((
                format!("{prefix}.out_proj.weight_v"),
                vec![cfg.d_model, cfg.codebook_dim, 1],
                v,
            ));
            let g: Vec<f32> = (0..cfg.d_model).map(|o| (o + 1) as f32).collect();
            entries.push((
                format!("{prefix}.out_proj.weight_g"),
                vec![cfg.d_model, 1, 1],
                g,
            ));
            let bias: Vec<f32> = (0..cfg.d_model).map(|o| o as f32 * 0.25).collect();
            entries.push((format!("{prefix}.out_proj.bias"), vec![cfg.d_model], bias));
            // in_proj tensors pass through untouched.
            entries.push((
                format!("{prefix}.in_proj.weight_v"),
                vec![cfg.codebook_dim, cfg.d_model, 1],
                vec![0.0; cfg.codebook_dim * cfg.d_model],
            ));
        }
        // Decoder-chain stand-in.
        entries.push(("decoder.model.1.weight".into(), vec![2], vec![1.0, 2.0]));
        build_safetensors(&entries)
    }

    #[test]
    fn config_parse_requires_all_fields() {
        let ok = br#"{"n_codebooks":32,"codebook_size":1024,"codebook_dim":8,"d_model":1024,"sample_rate":24000,"hop_length":320}"#;
        let cfg = DacConfig::parse(ok).expect("parse");
        assert_eq!(cfg.n_codebooks, 32);
        assert_eq!(cfg.hop_length, 320);

        let missing = br#"{"n_codebooks":32}"#;
        assert!(DacConfig::parse(missing).is_err());
        let zero = br#"{"n_codebooks":0,"codebook_size":1024,"codebook_dim":8,"d_model":1024,"sample_rate":24000,"hop_length":320}"#;
        assert!(DacConfig::parse(zero).is_err());
    }

    #[test]
    fn convert_folds_weight_norm_and_emits_metadata() {
        let cfg = tiny_config();
        let (b, report) = convert(synthetic_dac(&cfg), &cfg).expect("convert");
        assert_eq!(report.skipped_non_float, 0);

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        assert!(matches!(
            file.get(KEY_N_CODEBOOKS),
            Some(GgufMetadataValue::U32(2))
        ));
        assert!(matches!(
            file.get(KEY_CODEBOOK_DIM),
            Some(GgufMetadataValue::U32(2))
        ));
        assert!(matches!(
            file.get(KEY_SAMPLE_RATE),
            Some(GgufMetadataValue::U32(24000))
        ));

        // Folded weight: v row [3,4] (norm 5), g[o] = o+1 →
        // W[o,:] = (o+1)/5 * [3,4]. Hand-check o=0 and o=3 of quantizer 0.
        let raw = file
            .tensor_data("vokra.dac.quantizer.0.out_proj_weight")
            .expect("folded weight");
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(&vals[0..2], &[3.0 / 5.0, 4.0 / 5.0]);
        assert_eq!(&vals[6..8], &[4.0 * 3.0 / 5.0, 4.0 * 4.0 / 5.0]);

        // Stable runtime names exist for both quantizers; raw names remain.
        for i in 0..2 {
            for suffix in ["codebook", "out_proj_weight", "out_proj_bias"] {
                assert!(
                    file.tensor_info(&format!("vokra.dac.quantizer.{i}.{suffix}"))
                        .is_some(),
                    "missing derived tensor {i}/{suffix}"
                );
            }
            assert!(
                file.tensor_info(&format!("quantizer.quantizers.{i}.out_proj.weight_v"))
                    .is_some()
            );
        }
        assert!(file.tensor_info("decoder.model.1.weight").is_some());

        // Provenance: permissive MIT with model_id "dac".
        assert!(matches!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID),
            Some(GgufMetadataValue::String(s)) if s == "dac"
        ));
    }

    #[test]
    fn convert_rejects_shape_mismatch_and_missing_tensors() {
        let cfg = tiny_config();
        // Config says 3 quantizers but the checkpoint has 2 → missing tensor.
        let wrong = DacConfig {
            n_codebooks: 3,
            ..cfg
        };
        let err = convert(synthetic_dac(&cfg), &wrong).expect_err("must reject");
        assert!(err.to_string().contains("not found"));

        // Config codebook_dim mismatch → explicit shape error.
        let wrong_dim = DacConfig {
            codebook_dim: 5,
            ..cfg
        };
        let err = convert(synthetic_dac(&cfg), &wrong_dim).expect_err("must reject");
        assert!(err.to_string().contains("shape"));
    }
}
