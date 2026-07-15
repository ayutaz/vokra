//! Standalone codec GGUF binders (M4-04 T10/T11) — dumb, validation-heavy
//! bridges from a codec GGUF to the `vokra-ops` RVQ decode inputs.
//!
//! The converter is the offline-math home (weight-norm folding, Mimi
//! `embedding_sum / clamp(cluster_usage)` + pre-projection — ADR M4-04 §D-f);
//! this module only **binds** the derived tensors:
//!
//! - [`MimiCodecGguf::from_gguf`] — `vokra.mimi.*` metadata +
//!   `vokra.mimi.codebook_tables` → `Vec<CodebookTable>` + [`MimiRvqAttrs`];
//! - [`DacCodecGguf::from_gguf`] — `vokra.dac.*` metadata +
//!   `vokra.dac.quantizer.{i}.*` → low-dim tables + [`DacOutProj`]s +
//!   [`DacRvqAttrs`].
//!
//! Living in `vokra-models` keeps the dependency direction intact:
//! `vokra-ops` never learns about GGUF (its ops take plain slices), and the
//! GGUF reader lives in `vokra-core` (ADR M4-04 §D-f; the same reasoning as
//! M3-06's "keep the helper in vokra-ops so the crate edge does not
//! reverse", now one level up).
//!
//! Every missing key / tensor / dtype / shape mismatch is an explicit
//! [`VokraError::ModelLoad`] (FR-EX-08 — a codec GGUF that half-loads would
//! corrupt the feature stream plausibly).
//!
//! Note: this binder does **not** run the M2-13 weight-license gate itself —
//! callers loading untrusted GGUFs go through the usual
//! `vokra_core::check_weight_license` path first (Mimi is
//! `AttributionRequired` = admitted with attribution; DAC is `Permissive`).

use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};
use vokra_ops::{CodebookTable, DacOutProj, DacRvqAttrs, MimiRvqAttrs};

/// Reads a `u32` metadata key or fails loudly.
fn get_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(v) => v.as_u64().map(|x| x as u32).ok_or_else(|| {
            VokraError::ModelLoad(format!("codec GGUF: metadata `{key}` is not an integer"))
        }),
        None => Err(VokraError::ModelLoad(format!(
            "codec GGUF: required metadata `{key}` missing (was this GGUF produced by \
             `vokra-cli convert --model mimi|dac`?)"
        ))),
    }
}

/// Reads an F32 tensor's raw data + dimensions or fails loudly.
fn f32_tensor<'a>(file: &'a GgufFile, name: &str) -> Result<(Vec<u64>, &'a [u8])> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("codec GGUF: required tensor `{name}` missing"))
    })?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "codec GGUF: tensor `{name}` must be F32, got {:?}",
            info.dtype
        )));
    }
    let data = file
        .tensor_data(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("codec GGUF: tensor `{name}` has no data")))?;
    Ok((info.dimensions.clone(), data))
}

fn le_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Mimi
// ---------------------------------------------------------------------------

/// A standalone Mimi codec GGUF bound to its RVQ decode inputs.
#[derive(Debug, Clone)]
pub struct MimiCodecGguf {
    /// Shape attributes (from `vokra.mimi.*` metadata).
    pub attrs: MimiRvqAttrs,
    /// One effective (pre-projected) table per codebook, semantic first.
    pub tables: Vec<CodebookTable>,
}

impl MimiCodecGguf {
    /// Binds `vokra.mimi.*` + the derived `vokra.mimi.codebook_tables`
    /// tensor. Zero math — the converter already derived everything.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_codebooks = get_u32(file, "vokra.mimi.n_codebooks")? as usize;
        let codebook_size = get_u32(file, "vokra.mimi.codebook_size")? as usize;
        let d_model = get_u32(file, "vokra.mimi.d_model")? as usize;
        let attrs = MimiRvqAttrs {
            n_codebooks,
            codebook_size,
            d_model,
        };
        if n_codebooks == 0 || codebook_size == 0 || d_model == 0 {
            return Err(VokraError::ModelLoad(
                "mimi codec GGUF: vokra.mimi.* metadata has a zero axis".to_owned(),
            ));
        }

        let (dims, raw) = f32_tensor(file, "vokra.mimi.codebook_tables")?;
        let want = vec![n_codebooks as u64, codebook_size as u64, d_model as u64];
        if dims != want {
            return Err(VokraError::ModelLoad(format!(
                "mimi codec GGUF: vokra.mimi.codebook_tables dims {dims:?} != metadata {want:?}"
            )));
        }
        let vals = le_f32s(raw);
        let per_table = codebook_size * d_model;
        if vals.len() != n_codebooks * per_table {
            return Err(VokraError::ModelLoad(format!(
                "mimi codec GGUF: codebook_tables has {} f32s, expected {}",
                vals.len(),
                n_codebooks * per_table
            )));
        }
        let mut tables = Vec::with_capacity(n_codebooks);
        for cb in 0..n_codebooks {
            let slice = vals[cb * per_table..(cb + 1) * per_table].to_vec();
            tables.push(CodebookTable::new(codebook_size, d_model, slice)?);
        }
        Ok(Self { attrs, tables })
    }
}

// ---------------------------------------------------------------------------
// DAC
// ---------------------------------------------------------------------------

/// A standalone DAC codec GGUF bound to its factorized RVQ decode inputs.
#[derive(Debug, Clone)]
pub struct DacCodecGguf {
    /// Shape attributes (from `vokra.dac.*` metadata).
    pub attrs: DacRvqAttrs,
    /// One low-dim codebook per quantizer (`[codebook_size, codebook_dim]`).
    pub tables: Vec<CodebookTable>,
    /// One weight-norm-folded output projection per quantizer.
    pub out_projs: Vec<DacOutProj>,
    /// Model sample rate (`vokra.dac.sample_rate`).
    pub sample_rate: u32,
    /// Encoder hop length (`vokra.dac.hop_length`) — frame rate =
    /// `sample_rate / hop_length` (24 kHz variant: 24000/320 = 75 Hz).
    pub hop_length: u32,
}

impl DacCodecGguf {
    /// Binds `vokra.dac.*` + the derived per-quantizer decode tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_codebooks = get_u32(file, "vokra.dac.n_codebooks")? as usize;
        let codebook_size = get_u32(file, "vokra.dac.codebook_size")? as usize;
        let codebook_dim = get_u32(file, "vokra.dac.codebook_dim")? as usize;
        let d_model = get_u32(file, "vokra.dac.d_model")? as usize;
        let sample_rate = get_u32(file, "vokra.dac.sample_rate")?;
        let hop_length = get_u32(file, "vokra.dac.hop_length")?;
        let attrs = DacRvqAttrs {
            n_codebooks,
            codebook_size,
            codebook_dim,
            d_model,
        };
        if n_codebooks == 0 || codebook_size == 0 || codebook_dim == 0 || d_model == 0 {
            return Err(VokraError::ModelLoad(
                "dac codec GGUF: vokra.dac.* metadata has a zero axis".to_owned(),
            ));
        }

        let mut tables = Vec::with_capacity(n_codebooks);
        let mut out_projs = Vec::with_capacity(n_codebooks);
        for i in 0..n_codebooks {
            let (cb_dims, cb_raw) = f32_tensor(file, &format!("vokra.dac.quantizer.{i}.codebook"))?;
            if cb_dims != vec![codebook_size as u64, codebook_dim as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "dac codec GGUF: quantizer {i} codebook dims {cb_dims:?} != \
                     [{codebook_size}, {codebook_dim}]"
                )));
            }
            tables.push(CodebookTable::new(
                codebook_size,
                codebook_dim,
                le_f32s(cb_raw),
            )?);

            let (w_dims, w_raw) =
                f32_tensor(file, &format!("vokra.dac.quantizer.{i}.out_proj_weight"))?;
            if w_dims != vec![d_model as u64, codebook_dim as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "dac codec GGUF: quantizer {i} out_proj_weight dims {w_dims:?} != \
                     [{d_model}, {codebook_dim}]"
                )));
            }
            let (b_dims, b_raw) =
                f32_tensor(file, &format!("vokra.dac.quantizer.{i}.out_proj_bias"))?;
            if b_dims != vec![d_model as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "dac codec GGUF: quantizer {i} out_proj_bias dims {b_dims:?} != [{d_model}]"
                )));
            }
            out_projs.push(DacOutProj::new(
                d_model,
                codebook_dim,
                le_f32s(w_raw),
                le_f32s(b_raw),
            )?);
        }
        Ok(Self {
            attrs,
            tables,
            out_projs,
            sample_rate,
            hop_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    /// A hand-assembled Mimi codec GGUF (bypassing the converter — the
    /// converter e2e lives in tests/codec_gguf_roundtrip.rs).
    fn mimi_gguf(n_cb: u32, cb_size: u32, d_model: u32) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.mimi.n_codebooks", n_cb);
        b.add_u32("vokra.mimi.codebook_size", cb_size);
        b.add_u32("vokra.mimi.d_model", d_model);
        let n = (n_cb * cb_size * d_model) as usize;
        let vals: Vec<u8> = (0..n)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        b.add_tensor(
            "vokra.mimi.codebook_tables",
            GgmlType::F32,
            vec![n_cb as u64, cb_size as u64, d_model as u64],
            vals,
        )
        .unwrap();
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn mimi_binder_splits_tables_per_codebook() {
        let file = mimi_gguf(2, 3, 4);
        let codec = MimiCodecGguf::from_gguf(&file).expect("bind");
        assert_eq!(codec.attrs.n_codebooks, 2);
        assert_eq!(codec.tables.len(), 2);
        assert_eq!(codec.tables[0].codebook_size, 3);
        assert_eq!(codec.tables[0].d_model, 4);
        // Table 1's first element continues the ramp where table 0 ended.
        assert_eq!(codec.tables[1].data[0], (3 * 4) as f32 * 0.5);
    }

    #[test]
    fn mimi_binder_rejects_missing_or_mismatched_pieces() {
        // Missing metadata key.
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.mimi.n_codebooks", 2);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            MimiCodecGguf::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));

        // Dims mismatch between metadata and tensor.
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.mimi.n_codebooks", 2);
        b.add_u32("vokra.mimi.codebook_size", 3);
        b.add_u32("vokra.mimi.d_model", 4);
        b.add_tensor(
            "vokra.mimi.codebook_tables",
            GgmlType::F32,
            vec![1, 3, 4],
            vec![0u8; 3 * 4 * 4],
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            MimiCodecGguf::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn dac_binder_rejects_missing_quantizer_tensor() {
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.dac.n_codebooks", 1);
        b.add_u32("vokra.dac.codebook_size", 2);
        b.add_u32("vokra.dac.codebook_dim", 2);
        b.add_u32("vokra.dac.d_model", 3);
        b.add_u32("vokra.dac.sample_rate", 24000);
        b.add_u32("vokra.dac.hop_length", 320);
        // No quantizer tensors at all.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = DacCodecGguf::from_gguf(&file).expect_err("must fail");
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }
}
