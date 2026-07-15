//! Standalone Mimi (Kyutai) codec checkpoint → GGUF conversion (M4-04 T10).
//!
//! # Input format (one accepted naming — FR-EX-08 on anything else)
//!
//! The converter accepts the **moshi-native** safetensors naming
//! (`kyutai/moshiko-pytorch-bf16` `tokenizer-e351c8d8-checkpoint125.safetensors`,
//! pinned in ADR M4-04 §D-k — this is the format the M4-05 (Sesame CSM) /
//! M4-06 (Moshi) consumers feed):
//!
//! ```text
//!   quantizer.rvq_first.output_proj.weight              [d_model, dim, 1]
//!   quantizer.rvq_first.vq.layers.0._codebook.embedding_sum  [codebook_size, dim]
//!   quantizer.rvq_first.vq.layers.0._codebook.cluster_usage  [codebook_size]
//!   quantizer.rvq_rest.output_proj.weight               [d_model, dim, 1]
//!   quantizer.rvq_rest.vq.layers.{k}._codebook.…        k = 0..n_acoustic
//! ```
//!
//! The transformers-format `kyutai/mimi` repo (renamed tensors,
//! `quantizer.semantic_residual_vector_quantizer.…`) is **not** accepted —
//! the error message names the expected repo/file instead of guessing.
//!
//! # What is written
//!
//! 1. **Every upstream tensor pass-through** (encoder / decoder chain
//!    included) so M4-05/06 consume the same GGUF without re-running the
//!    converter (ADR M4-04 §D-f).
//! 2. **One derived tensor** `vokra.mimi.codebook_tables` — f32
//!    `[n_codebooks, codebook_size, d_model]` **effective (pre-projected)**
//!    tables in codebook order (semantic first, then acoustic):
//!
//!    ```text
//!      embedding[cb]  = embedding_sum[cb] / clamp(cluster_usage[cb], 1e-5)
//!      table[cb][i,:] = W_split(cb) @ embedding[cb][i,:]
//!    ```
//!
//!    where `W_split` is the split's `output_proj` (a **bias-free** 1×1
//!    conv — verified from the checkpoint: no bias tensors exist — so
//!    project-then-sum ≡ sum-then-project up to FP32 reassociation; the
//!    runtime decode is a plain gather + FP32 fold over these tables).
//!    The `clamp(min=1e-5)` mirrors moshi's `EuclideanCodebook.embedding`
//!    property (`epsilon = 1e-5` — kyutai-labs/moshi core_vq.py, ADR M4-04).
//! 3. `vokra.mimi.{n_codebooks,codebook_size,d_model}` metadata read from
//!    the checkpoint shapes (never hard-coded — the physical checkpoint has
//!    1 semantic + 31 acoustic = 32 codebooks; consumers slice the prefix
//!    they need, e.g. Moshi's LM uses the first 8).
//! 4. `vokra.provenance.*`: `model_id = "mimi"` → `AttributionRequired`
//!    (CC-BY 4.0; the M2-13 gate admits attribution-class weights without a
//!    research flag, and the NOTICE §5 clause discharges the attribution).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for standalone Mimi codec GGUFs.
pub(crate) const ARCH: &str = "mimi";
/// `vokra.model.name` value.
const NAME: &str = "Mimi (Kyutai) neural audio codec";

const KEY_N_CODEBOOKS: &str = "vokra.mimi.n_codebooks";
const KEY_CODEBOOK_SIZE: &str = "vokra.mimi.codebook_size";
const KEY_D_MODEL: &str = "vokra.mimi.d_model";

/// Name of the derived effective-codebook-tables tensor (ADR M4-04 §D-f).
pub const DERIVED_TABLES_TENSOR: &str = "vokra.mimi.codebook_tables";

/// moshi `EuclideanCodebook` epsilon (core_vq.py `epsilon: float = 1e-5`).
const CLUSTER_USAGE_EPSILON: f32 = 1e-5;

/// Conversion report.
#[derive(Debug, Default)]
pub(crate) struct MimiReport {
    /// Upstream tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32/F16 tensors skipped (defensive; the checkpoint is all-F32).
    pub(crate) skipped_non_float: usize,
    /// Codebook count derived from the checkpoint (semantic + acoustic).
    pub(crate) n_codebooks: usize,
    /// Entries per codebook.
    pub(crate) codebook_size: usize,
    /// Output feature width (output_proj rows).
    pub(crate) d_model: usize,
}

/// Converts a moshi-native Mimi safetensors buffer into a populated GGUF
/// builder (all tensors pass-through + derived tables + metadata).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, MimiReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    // ---- Locate the quantizer tensors (one accepted naming) ---------------
    let find = |name: &str| st.tensors().iter().find(|t| t.name == name);
    let semantic_sum = "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
    if find(semantic_sum).is_none() {
        // Give the transformers-format user a precise redirect instead of a
        // generic "missing tensor" (FR-EX-08: explicit, actionable error).
        let looks_transformers = st.tensors().iter().any(|t| {
            t.name
                .starts_with("quantizer.semantic_residual_vector_quantizer")
        });
        return Err(ConvertError::Parse(if looks_transformers {
            "mimi: this looks like the transformers-format `kyutai/mimi` checkpoint \
             (quantizer.semantic_residual_vector_quantizer.*). The Vokra converter accepts the \
             moshi-native naming only (kyutai/moshiko-pytorch-bf16 \
             tokenizer-e351c8d8-checkpoint125.safetensors, `quantizer.rvq_first.*`) — see ADR \
             M4-04 §D-k."
                .to_owned()
        } else {
            format!(
                "mimi: required tensor `{semantic_sum}` not found — not a moshi-native Mimi \
                     checkpoint"
            )
        }));
    }

    // Split geometry: 1 semantic layer + contiguous acoustic layers 0..n.
    let mut n_acoustic = 0usize;
    while find(&format!(
        "quantizer.rvq_rest.vq.layers.{n_acoustic}._codebook.embedding_sum"
    ))
    .is_some()
    {
        n_acoustic += 1;
    }
    if n_acoustic == 0 {
        return Err(ConvertError::Parse(
            "mimi: no acoustic quantizer layers (quantizer.rvq_rest.vq.layers.0.*) found"
                .to_owned(),
        ));
    }
    // The semantic split must have exactly one layer (moshi SplitRVQ with
    // n_q_semantic = 1); a second semantic layer means an unknown variant.
    if find("quantizer.rvq_first.vq.layers.1._codebook.embedding_sum").is_some() {
        return Err(ConvertError::Parse(
            "mimi: more than one semantic quantizer layer — unknown Mimi variant (expected \
             n_q_semantic = 1)"
                .to_owned(),
        ));
    }

    // ---- Read quantizer tensors as f32 -------------------------------------
    let f32_tensor = |name: &str| -> Result<(Vec<u64>, Vec<f32>), ConvertError> {
        let t = find(name).ok_or_else(|| {
            ConvertError::Parse(format!("mimi: required tensor `{name}` not found"))
        })?;
        if t.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "mimi: tensor `{name}` must be F32 (checkpoint quantizers are F32), got {:?}",
                t.dtype
            )));
        }
        let raw = st.tensor_bytes(t);
        let vals = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((t.shape.clone(), vals))
    };

    let (first_proj_shape, first_proj) = f32_tensor("quantizer.rvq_first.output_proj.weight")?;
    let (rest_proj_shape, rest_proj) = f32_tensor("quantizer.rvq_rest.output_proj.weight")?;
    if first_proj_shape.len() != 3 || first_proj_shape[2] != 1 {
        return Err(ConvertError::Parse(format!(
            "mimi: rvq_first.output_proj.weight must be [d_model, dim, 1], got {first_proj_shape:?}"
        )));
    }
    if rest_proj_shape != first_proj_shape {
        return Err(ConvertError::Parse(format!(
            "mimi: output_proj shapes differ between splits ({first_proj_shape:?} vs \
             {rest_proj_shape:?})"
        )));
    }
    let d_model = first_proj_shape[0] as usize;
    let dim = first_proj_shape[1] as usize;

    // ---- Derive effective tables (semantic first, then acoustic) ----------
    let mut codebook_size = 0usize;
    let n_codebooks = 1 + n_acoustic;
    let mut tables = Vec::<f32>::new();

    for cb in 0..n_codebooks {
        let (split, layer, proj) = if cb == 0 {
            ("rvq_first", 0, &first_proj)
        } else {
            ("rvq_rest", cb - 1, &rest_proj)
        };
        let base = format!("quantizer.{split}.vq.layers.{layer}._codebook");
        let (sum_shape, sum) = f32_tensor(&format!("{base}.embedding_sum"))?;
        let (usage_shape, usage) = f32_tensor(&format!("{base}.cluster_usage"))?;
        if sum_shape.len() != 2 || sum_shape[1] != dim as u64 {
            return Err(ConvertError::Parse(format!(
                "mimi: {base}.embedding_sum must be [codebook_size, {dim}], got {sum_shape:?}"
            )));
        }
        if usage_shape != vec![sum_shape[0]] {
            return Err(ConvertError::Parse(format!(
                "mimi: {base}.cluster_usage must be [{}], got {usage_shape:?}",
                sum_shape[0]
            )));
        }
        if cb == 0 {
            codebook_size = sum_shape[0] as usize;
        } else if sum_shape[0] != codebook_size as u64 {
            return Err(ConvertError::Parse(format!(
                "mimi: {base} codebook_size {} != first codebook's {codebook_size}",
                sum_shape[0]
            )));
        }

        // embedding = embedding_sum / clamp(cluster_usage, 1e-5); then the
        // bias-free 1x1 conv projection per row (FP32 throughout).
        for i in 0..codebook_size {
            let denom = usage[i].max(CLUSTER_USAGE_EPSILON);
            let row = &sum[i * dim..(i + 1) * dim];
            for o in 0..d_model {
                let w_row = &proj[o * dim..(o + 1) * dim];
                let mut acc = 0.0_f32;
                for c in 0..dim {
                    acc += w_row[c] * (row[c] / denom);
                }
                tables.push(acc);
            }
        }
    }

    // ---- Assemble the GGUF --------------------------------------------------
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_u32(KEY_N_CODEBOOKS, n_codebooks as u32);
    b.add_u32(KEY_CODEBOOK_SIZE, codebook_size as u32);
    b.add_u32(KEY_D_MODEL, d_model as u32);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("mimi"),
        Some("kyutai/moshiko-pytorch-bf16 tokenizer-e351c8d8-checkpoint125.safetensors"),
    );

    let mut report = MimiReport {
        n_codebooks,
        codebook_size,
        d_model,
        ..MimiReport::default()
    };

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

    let table_bytes: Vec<u8> = tables.iter().flat_map(|f| f.to_le_bytes()).collect();
    b.add_tensor(
        DERIVED_TABLES_TENSOR,
        GgmlType::F32,
        vec![n_codebooks as u64, codebook_size as u64, d_model as u64],
        table_bytes,
    )?;

    Ok((b, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    /// Builds a synthetic moshi-native Mimi checkpoint: 1 semantic + 2
    /// acoustic codebooks, codebook_size = 4, dim = 2, d_model = 3.
    fn synthetic_mimi() -> Vec<u8> {
        let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        // Projections: rvq_first W[o,c] = (o+1) * 0.5 + c; rvq_rest W = -W.
        let first_proj: Vec<f32> = (0..3)
            .flat_map(|o| (0..2).map(move |c| (o + 1) as f32 * 0.5 + c as f32))
            .collect();
        let rest_proj: Vec<f32> = first_proj.iter().map(|x| -x).collect();
        entries.push((
            "quantizer.rvq_first.output_proj.weight".into(),
            vec![3, 2, 1],
            first_proj,
        ));
        entries.push((
            "quantizer.rvq_rest.output_proj.weight".into(),
            vec![3, 2, 1],
            rest_proj,
        ));
        // input_proj tensors are passed through but unused by the derivation.
        entries.push((
            "quantizer.rvq_first.input_proj.weight".into(),
            vec![2, 3, 1],
            vec![0.0; 6],
        ));
        // Codebooks: layer sums are ramps; usages include one below-epsilon
        // entry (0.0) to exercise the clamp path.
        for (split, layer, salt) in [
            ("rvq_first", 0usize, 1.0f32),
            ("rvq_rest", 0, 2.0),
            ("rvq_rest", 1, 3.0),
        ] {
            let base = format!("quantizer.{split}.vq.layers.{layer}._codebook");
            let sum: Vec<f32> = (0..4 * 2).map(|i| i as f32 * salt).collect();
            let usage: Vec<f32> = vec![1.0, 2.0, 0.0, 4.0]; // 0.0 → clamped to 1e-5
            entries.push((format!("{base}.embedding_sum"), vec![4, 2], sum));
            entries.push((format!("{base}.cluster_usage"), vec![4], usage));
            entries.push((format!("{base}._initialized"), vec![1], vec![1.0]));
        }
        // A non-quantizer tensor (decoder chain stand-in) — must pass through.
        entries.push((
            "decoder.model.0.conv.weight".into(),
            vec![2],
            vec![7.0, 8.0],
        ));

        build_safetensors(&entries)
    }

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

    #[test]
    #[allow(clippy::identity_op)] // keep the (cb * rows + i) * d_model formula shape visible
    fn convert_derives_tables_and_metadata_from_checkpoint_shapes() {
        let (b, report) = convert(synthetic_mimi()).expect("convert");
        assert_eq!(report.n_codebooks, 3);
        assert_eq!(report.codebook_size, 4);
        assert_eq!(report.d_model, 3);
        assert_eq!(report.skipped_non_float, 0);

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        assert!(matches!(
            file.get(KEY_N_CODEBOOKS),
            Some(GgufMetadataValue::U32(3))
        ));
        assert!(matches!(
            file.get(KEY_CODEBOOK_SIZE),
            Some(GgufMetadataValue::U32(4))
        ));
        assert!(matches!(
            file.get(KEY_D_MODEL),
            Some(GgufMetadataValue::U32(3))
        ));

        // Derived tensor shape + hand-computed spot values.
        let info = file
            .tensor_info(DERIVED_TABLES_TENSOR)
            .expect("derived tensor");
        assert_eq!(info.dimensions, vec![3, 4, 3]);
        let raw = file.tensor_data(DERIVED_TABLES_TENSOR).unwrap();
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // cb=0 (semantic, salt=1): entry i=1 → sum row [2, 3], usage 2.0 →
        // emb [1.0, 1.5]; W rows: o0 [0.5, 1.5], o1 [1.0, 2.0], o2 [1.5, 2.5]
        // → table[0][1,:] = [0.5*1 + 1.5*1.5, 1.0*1 + 2.0*1.5, 1.5*1 + 2.5*1.5]
        //                 = [2.75, 4.0, 5.25]
        let base = (0 * 4 + 1) * 3;
        assert_eq!(&vals[base..base + 3], &[2.75, 4.0, 5.25]);
        // cb=0 entry i=2 exercises the clamp: usage 0.0 → denom 1e-5,
        // sum row [4, 5] → emb [4e5, 5e5]; o0: 0.5*4e5 + 1.5*5e5 = 9.5e5.
        let base = (0 * 4 + 2) * 3;
        assert_eq!(vals[base], 0.5 * (4.0 / 1e-5) + 1.5 * (5.0 / 1e-5));
        // cb=1 (acoustic 0, salt=2, negated proj): entry i=1 → sum [4, 6],
        // usage 2 → emb [2, 3]; o0: -(0.5*2 + 1.5*3) = -5.5.
        let base = (1 * 4 + 1) * 3;
        assert_eq!(vals[base], -5.5);

        // Pass-through: the decoder-chain stand-in and the raw quantizer
        // tensors are all present (full pass-through, ADR M4-04 §D-f).
        assert!(file.tensor_info("decoder.model.0.conv.weight").is_some());
        assert!(
            file.tensor_info("quantizer.rvq_rest.vq.layers.1._codebook.embedding_sum")
                .is_some()
        );

        // Provenance: attribution class (CC-BY 4.0) with model_id "mimi".
        assert!(matches!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID),
            Some(GgufMetadataValue::String(s)) if s == "mimi"
        ));
    }

    #[test]
    fn convert_rejects_transformers_format_with_redirect() {
        let entries = vec![(
            "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum".to_string(),
            vec![4usize, 2],
            vec![0.0f32; 8],
        )];
        let err = convert(build_safetensors(&entries)).expect_err("must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("moshi-native") && msg.contains("kyutai/mimi"),
            "error must redirect to the accepted format, got: {msg}"
        );
    }

    #[test]
    fn convert_rejects_checkpoint_without_acoustic_layers() {
        let entries = vec![
            (
                "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum".to_string(),
                vec![4usize, 2],
                vec![0.0f32; 8],
            ),
            (
                "quantizer.rvq_first.vq.layers.0._codebook.cluster_usage".to_string(),
                vec![4usize],
                vec![1.0f32; 4],
            ),
            (
                "quantizer.rvq_first.output_proj.weight".to_string(),
                vec![3usize, 2, 1],
                vec![0.0f32; 6],
            ),
        ];
        let err = convert(build_safetensors(&entries)).expect_err("must reject");
        assert!(err.to_string().contains("no acoustic quantizer layers"));
    }
}
