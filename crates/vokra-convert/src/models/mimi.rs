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
//! 5. **Neural-chain adapter (T29)** — when the checkpoint carries the
//!    SEANet chain (`encoder.model.0.conv.conv.weight` present):
//!    - the `vokra.mimi.*` config chunk group
//!      (`vokra-models::mimi::config`), shape-derived where observable
//!      (dimension / n_filters / kernels / ratios / compress /
//!      n_residual_layers / transformer d·ff·n_layer / quantizer shape) and
//!      transcribed from `loaders.py` `_mimi_config` where not
//!      (sample_rate, frame_rate, dilation_base, n_head, context,
//!      max_period, layer_scale — the same constants the Moshi / CSM
//!      converters stamp);
//!    - derived **structural tensors** `mimi.enc.*` / `mimi.dec.*` in the
//!      exact layout the runtime binders consume: convs verbatim
//!      (`[out, in, k]` / conv-transpose `[in, out, k]` — the checkpoint
//!      stores plain **fused** weights; loaders.py: "weights are
//!      pre-processed for inference", `norm: "none"`, so no `weight_norm`
//!      fusion is needed), transformer linears **transposed** to the
//!      runtime `w_t = [in, out]` GEMM layout (fused `in_proj_weight`
//!      split into q/k/v), the channel-wise `upsample` conv-transpose
//!      (`[dim, 1, k]`, upstream `upsample_channel_wise_bug=True`)
//!      **zero-expanded** to the dense `[dim, dim, k]` equivalent, raw
//!      (un-projected) per-codebook embeddings `mimi.enc.cb{i}` at the
//!      quantizer width, and both split input projections
//!      (`mimi.enc.input_proj` = `rvq_first`, `mimi.enc.input_proj_rest`
//!      = `rvq_rest`).
//!
//!    A checkpoint **without** the chain (quantizer-only synthetic
//!    fixtures) converts as before; a checkpoint **with** the chain that
//!    deviates from the moshi-native geometry is a loud error naming the
//!    offending tensor (FR-EX-08 — never a silent partial mapping).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` value for standalone Mimi codec GGUFs.
pub(crate) const ARCH: &str = "mimi";
/// `vokra.model.name` value.
const NAME: &str = "Mimi (Kyutai) neural audio codec";

const KEY_N_CODEBOOKS: &str = "vokra.mimi.n_codebooks";
const KEY_CODEBOOK_SIZE: &str = "vokra.mimi.codebook_size";
const KEY_D_MODEL: &str = "vokra.mimi.d_model";

// --- vokra.mimi.* config keys (duplicated from
// vokra-models/src/mimi/config.rs per the cross-crate pattern — the same
// block the Moshi / CSM converters carry) -----------------------------------
const KEY_MIMI_SAMPLE_RATE: &str = "vokra.mimi.sample_rate";
const KEY_MIMI_FRAME_RATE_MHZ: &str = "vokra.mimi.frame_rate_mhz";
const KEY_MIMI_SEANET_DIMENSION: &str = "vokra.mimi.seanet.dimension";
const KEY_MIMI_SEANET_N_FILTERS: &str = "vokra.mimi.seanet.n_filters";
const KEY_MIMI_SEANET_N_RESIDUAL_LAYERS: &str = "vokra.mimi.seanet.n_residual_layers";
const KEY_MIMI_SEANET_KERNEL_SIZE: &str = "vokra.mimi.seanet.kernel_size";
const KEY_MIMI_SEANET_RESIDUAL_KERNEL_SIZE: &str = "vokra.mimi.seanet.residual_kernel_size";
const KEY_MIMI_SEANET_LAST_KERNEL_SIZE: &str = "vokra.mimi.seanet.last_kernel_size";
const KEY_MIMI_SEANET_COMPRESS: &str = "vokra.mimi.seanet.compress";
const KEY_MIMI_SEANET_DILATION_BASE: &str = "vokra.mimi.seanet.dilation_base";
const KEY_MIMI_SEANET_N_RATIOS: &str = "vokra.mimi.seanet.n_ratios";
const PREFIX_MIMI_SEANET_RATIO: &str = "vokra.mimi.seanet.ratio.";
const KEY_MIMI_QUANTIZER_DIMENSION: &str = "vokra.mimi.quantizer.dimension";
const KEY_MIMI_QUANTIZER_N_Q: &str = "vokra.mimi.quantizer.n_q";
const KEY_MIMI_QUANTIZER_BINS: &str = "vokra.mimi.quantizer.bins";
const KEY_MIMI_QUANTIZER_INPUT_DIMENSION: &str = "vokra.mimi.quantizer.input_dimension";
const KEY_MIMI_QUANTIZER_OUTPUT_DIMENSION: &str = "vokra.mimi.quantizer.output_dimension";
const KEY_MIMI_TRANSFORMER_D_MODEL: &str = "vokra.mimi.transformer.d_model";
const KEY_MIMI_TRANSFORMER_N_HEAD: &str = "vokra.mimi.transformer.n_head";
const KEY_MIMI_TRANSFORMER_N_LAYER: &str = "vokra.mimi.transformer.n_layer";
const KEY_MIMI_TRANSFORMER_FF_DIM: &str = "vokra.mimi.transformer.ff_dim";
const KEY_MIMI_TRANSFORMER_CONTEXT: &str = "vokra.mimi.transformer.context";
const KEY_MIMI_TRANSFORMER_MAX_PERIOD: &str = "vokra.mimi.transformer.max_period";
const KEY_MIMI_TRANSFORMER_LAYER_SCALE: &str = "vokra.mimi.transformer.layer_scale";

// --- `loaders.py` `_mimi_config` constants for the hparams a checkpoint's
// tensor shapes cannot express (transcribed verbatim; the shape-observable
// values are derived from the checkpoint and cross-checked instead) ---------
const MIMI_SAMPLE_RATE: u32 = 24_000;
const MIMI_FRAME_RATE_MHZ: u32 = 12_500;
/// `_seanet_kwargs["dilation_base"]` — dilation leaves no shape trace.
const MIMI_SEANET_DILATION_BASE: u32 = 2;
/// `_transformer_kwargs["num_heads"]` — the head split leaves no shape
/// trace in the fused `in_proj_weight`.
const MIMI_TRANSFORMER_N_HEAD: u32 = 8;
const MIMI_TRANSFORMER_CONTEXT: u32 = 250;
const MIMI_TRANSFORMER_MAX_PERIOD: u32 = 10_000;
const MIMI_TRANSFORMER_LAYER_SCALE: f32 = 0.01;

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
    /// Structural `mimi.enc.*` / `mimi.dec.*` tensors written by the
    /// neural-chain adapter (`0` when the checkpoint carries no chain —
    /// quantizer-only synthetic fixtures).
    pub(crate) structural_written: usize,
}

/// Finds a tensor by exact name.
fn find<'a>(st: &'a SafetensorsFile, name: &str) -> Option<&'a SafeTensorInfo> {
    st.tensors().iter().find(|t| t.name == name)
}

/// Reads a named tensor as f32, requiring the F32 dtype (the moshi-native
/// checkpoint is all-F32).
fn f32_tensor(st: &SafetensorsFile, name: &str) -> Result<(Vec<u64>, Vec<f32>), ConvertError> {
    let t = find(st, name)
        .ok_or_else(|| ConvertError::Parse(format!("mimi: required tensor `{name}` not found")))?;
    if t.dtype != GgmlType::F32 {
        return Err(ConvertError::Parse(format!(
            "mimi: tensor `{name}` must be F32 (checkpoint tensors are F32), got {:?}",
            t.dtype
        )));
    }
    let raw = st.tensor_bytes(t);
    let vals = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((t.shape.clone(), vals))
}

/// Converts a moshi-native Mimi safetensors buffer into a populated GGUF
/// builder (all tensors pass-through + derived tables + metadata).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, MimiReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    // ---- Locate the quantizer tensors (one accepted naming) ---------------
    let semantic_sum = "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
    if find(&st, semantic_sum).is_none() {
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
    while find(
        &st,
        &format!("quantizer.rvq_rest.vq.layers.{n_acoustic}._codebook.embedding_sum"),
    )
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
    if find(
        &st,
        "quantizer.rvq_first.vq.layers.1._codebook.embedding_sum",
    )
    .is_some()
    {
        return Err(ConvertError::Parse(
            "mimi: more than one semantic quantizer layer — unknown Mimi variant (expected \
             n_q_semantic = 1)"
                .to_owned(),
        ));
    }

    // ---- Read quantizer tensors as f32 -------------------------------------
    let (first_proj_shape, first_proj) = f32_tensor(&st, "quantizer.rvq_first.output_proj.weight")?;
    let (rest_proj_shape, rest_proj) = f32_tensor(&st, "quantizer.rvq_rest.output_proj.weight")?;
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
        let (sum_shape, sum) = f32_tensor(&st, &format!("{base}.embedding_sum"))?;
        let (usage_shape, usage) = f32_tensor(&st, &format!("{base}.cluster_usage"))?;
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

    // ---- Neural-chain adapter (T29 — module docs §5) ------------------------
    // Presence-driven: the physical checkpoint always carries the chain;
    // quantizer-only synthetic fixtures do not. A *present* chain that then
    // fails any geometry check is a loud error (FR-EX-08).
    if find(&st, "encoder.model.0.conv.conv.weight").is_some() {
        report.structural_written = map_neural_chain(&st, &mut b, n_codebooks, codebook_size)?;
    }

    Ok((b, report))
}

// ============================================================================
// Neural-chain adapter (T29): moshi-native names → Vokra structural naming
// ============================================================================

fn parse_err(msg: String) -> ConvertError {
    ConvertError::Parse(msg)
}

/// `{prefix}.model.{i}` indices whose module carries `tail` (sorted).
fn model_indices(st: &SafetensorsFile, prefix: &str, tail: &str) -> Vec<usize> {
    let head = format!("{prefix}.model.");
    let mut idx: Vec<usize> = st
        .tensors()
        .iter()
        .filter_map(|t| {
            let rest = t.name.strip_prefix(&head)?;
            let (i, t_tail) = rest.split_once('.')?;
            (t_tail == tail).then(|| i.parse::<usize>().ok()).flatten()
        })
        .collect();
    idx.sort_unstable();
    idx.dedup();
    idx
}

/// Reads a 3-D shape as `(a, b, c)`.
fn shape3(st: &SafetensorsFile, name: &str) -> Result<(usize, usize, usize), ConvertError> {
    let t = find(st, name)
        .ok_or_else(|| parse_err(format!("mimi: required tensor `{name}` not found")))?;
    if t.shape.len() != 3 {
        return Err(parse_err(format!(
            "mimi: tensor `{name}` must be 3-D, got {:?}",
            t.shape
        )));
    }
    Ok((
        t.shape[0] as usize,
        t.shape[1] as usize,
        t.shape[2] as usize,
    ))
}

/// Verbatim F32 copy under a new name (shape preserved).
fn copy_tensor(
    st: &SafetensorsFile,
    b: &mut GgufBuilder,
    src: &str,
    dst: &str,
) -> Result<(), ConvertError> {
    let t = find(st, src)
        .ok_or_else(|| parse_err(format!("mimi: required tensor `{src}` not found")))?;
    if t.dtype != GgmlType::F32 {
        return Err(parse_err(format!(
            "mimi: tensor `{src}` must be F32, got {:?}",
            t.dtype
        )));
    }
    b.add_tensor(
        dst,
        GgmlType::F32,
        t.shape.clone(),
        st.tensor_bytes(t).to_vec(),
    )?;
    Ok(())
}

/// Copies `{src_base}.weight` (+ `.bias` when `expect_bias`) to
/// `{dst_base}.weight` / `.bias`, enforcing the bias presence the runtime
/// geometry assumes (a mismatch is an unknown variant — loud, FR-EX-08).
fn copy_conv(
    st: &SafetensorsFile,
    b: &mut GgufBuilder,
    src_base: &str,
    dst_base: &str,
    expect_bias: bool,
    written: &mut usize,
) -> Result<(), ConvertError> {
    copy_tensor(
        st,
        b,
        &format!("{src_base}.weight"),
        &format!("{dst_base}.weight"),
    )?;
    *written += 1;
    let bias_src = format!("{src_base}.bias");
    let has_bias = find(st, &bias_src).is_some();
    if has_bias != expect_bias {
        return Err(parse_err(format!(
            "mimi: `{bias_src}` presence ({has_bias}) does not match the moshi-native geometry \
             (expected {expect_bias}) — unknown Mimi variant"
        )));
    }
    if expect_bias {
        copy_tensor(st, b, &bias_src, &format!("{dst_base}.bias"))?;
        *written += 1;
    }
    Ok(())
}

/// `[out, in]` row-major → `[in, out]` row-major (runtime `w_t` GEMM
/// layout for the bias-less transformer linears).
fn transpose_oi(w: &[f32], out: usize, inn: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; w.len()];
    for o in 0..out {
        for c in 0..inn {
            t[c * out + o] = w[o * inn + c];
        }
    }
    t
}

/// Adds a plain f32 tensor.
fn add_f32_tensor(
    b: &mut GgufBuilder,
    name: &str,
    dims: Vec<u64>,
    vals: &[f32],
) -> Result<(), ConvertError> {
    let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    b.add_tensor(name, GgmlType::F32, dims, bytes)?;
    Ok(())
}

/// The moshi-native → structural adapter (module docs §5). Returns the
/// number of structural tensors written; every deviation from the
/// moshi-native geometry is a loud [`ConvertError::Parse`].
#[allow(clippy::too_many_lines)] // one linear transcription pass, kept in source order
fn map_neural_chain(
    st: &SafetensorsFile,
    b: &mut GgufBuilder,
    n_codebooks: usize,
    bins: usize,
) -> Result<usize, ConvertError> {
    let mut written = 0usize;

    // ---- Encoder SEANet walk (seanet.py encoder order) ---------------------
    let enc_convs = model_indices(st, "encoder", "conv.conv.weight");
    let enc_blocks = model_indices(st, "encoder", "block.1.conv.conv.weight");
    if enc_convs.len() < 3 {
        return Err(parse_err(format!(
            "mimi: encoder needs init + >=1 downsample + final plain convs, found model \
             indices {enc_convs:?}"
        )));
    }
    let enc_init = enc_convs[0];
    let enc_final = *enc_convs.last().expect("len >= 3");
    let enc_downs = &enc_convs[1..enc_convs.len() - 1];

    let init_name = format!("encoder.model.{enc_init}.conv.conv.weight");
    let (n_filters, init_in, kernel_size) = shape3(st, &init_name)?;
    if init_in != 1 {
        return Err(parse_err(format!(
            "mimi: `{init_name}` in_ch must be 1 (mono PCM), got {init_in}"
        )));
    }
    copy_conv(
        st,
        b,
        &format!("encoder.model.{enc_init}.conv.conv"),
        "mimi.enc.init",
        true,
        &mut written,
    )?;

    let mut ch = n_filters;
    let mut enc_ratios: Vec<usize> = Vec::new(); // encoder (reversed) order
    let mut n_res: Option<usize> = None;
    let mut compress: Option<usize> = None;
    let mut res_kernel: Option<usize> = None;
    let mut prev_bound = enc_init;
    for (s, &down_i) in enc_downs.iter().enumerate() {
        let stage_blocks: Vec<usize> = enc_blocks
            .iter()
            .copied()
            .filter(|&i| i > prev_bound && i < down_i)
            .collect();
        if stage_blocks.is_empty() {
            return Err(parse_err(format!(
                "mimi: encoder stage {s} has no residual block between model.{prev_bound} \
                 and model.{down_i}"
            )));
        }
        match n_res {
            None => n_res = Some(stage_blocks.len()),
            Some(n) if n == stage_blocks.len() => {}
            Some(n) => {
                return Err(parse_err(format!(
                    "mimi: encoder stage {s} has {} residual blocks, stage 0 had {n} — \
                     non-uniform n_residual_layers",
                    stage_blocks.len()
                )));
            }
        }
        for (j, &bi) in stage_blocks.iter().enumerate() {
            let c1_name = format!("encoder.model.{bi}.block.1.conv.conv.weight");
            let (hidden, c1_in, k_res) = shape3(st, &c1_name)?;
            if c1_in != ch {
                return Err(parse_err(format!(
                    "mimi: `{c1_name}` in_ch {c1_in} != stage width {ch}"
                )));
            }
            if hidden == 0 || ch % hidden != 0 {
                return Err(parse_err(format!(
                    "mimi: `{c1_name}` hidden {hidden} does not divide stage width {ch} \
                     (compress must be integer)"
                )));
            }
            let this_compress = ch / hidden;
            match compress {
                None => compress = Some(this_compress),
                Some(c) if c == this_compress => {}
                Some(c) => {
                    return Err(parse_err(format!(
                        "mimi: `{c1_name}` implies compress {this_compress}, earlier blocks \
                         implied {c}"
                    )));
                }
            }
            match res_kernel {
                None => res_kernel = Some(k_res),
                Some(k) if k == k_res => {}
                Some(k) => {
                    return Err(parse_err(format!(
                        "mimi: `{c1_name}` kernel {k_res} != earlier residual kernel {k}"
                    )));
                }
            }
            let c2_name = format!("encoder.model.{bi}.block.3.conv.conv.weight");
            let (c2_out, c2_in, c2_k) = shape3(st, &c2_name)?;
            if c2_out != ch || c2_in != hidden || c2_k != 1 {
                return Err(parse_err(format!(
                    "mimi: `{c2_name}` must be [{ch}, {hidden}, 1], got [{c2_out}, {c2_in}, \
                     {c2_k}]"
                )));
            }
            copy_conv(
                st,
                b,
                &format!("encoder.model.{bi}.block.1.conv.conv"),
                &format!("mimi.enc.s{s}.b{j}.c1"),
                true,
                &mut written,
            )?;
            copy_conv(
                st,
                b,
                &format!("encoder.model.{bi}.block.3.conv.conv"),
                &format!("mimi.enc.s{s}.b{j}.c2"),
                true,
                &mut written,
            )?;
        }
        let down_name = format!("encoder.model.{down_i}.conv.conv.weight");
        let (down_out, down_in, down_k) = shape3(st, &down_name)?;
        if down_in != ch || down_out != ch * 2 {
            return Err(parse_err(format!(
                "mimi: `{down_name}` must double the channels [{}, {ch}, 2*ratio], got \
                 [{down_out}, {down_in}, {down_k}]",
                ch * 2
            )));
        }
        if down_k % 2 != 0 || down_k == 0 {
            return Err(parse_err(format!(
                "mimi: `{down_name}` kernel {down_k} is not 2*ratio"
            )));
        }
        enc_ratios.push(down_k / 2);
        copy_conv(
            st,
            b,
            &format!("encoder.model.{down_i}.conv.conv"),
            &format!("mimi.enc.s{s}.down"),
            true,
            &mut written,
        )?;
        ch *= 2;
        prev_bound = down_i;
    }
    if enc_blocks.iter().any(|&i| i > prev_bound) {
        return Err(parse_err(format!(
            "mimi: encoder has residual blocks after the last downsample (model.{prev_bound}) \
             — unknown variant"
        )));
    }
    let final_name = format!("encoder.model.{enc_final}.conv.conv.weight");
    let (dimension, final_in, last_kernel_size) = shape3(st, &final_name)?;
    if final_in != ch {
        return Err(parse_err(format!(
            "mimi: `{final_name}` in_ch {final_in} != channel walk {ch}"
        )));
    }
    copy_conv(
        st,
        b,
        &format!("encoder.model.{enc_final}.conv.conv"),
        "mimi.enc.final",
        true,
        &mut written,
    )?;
    let n_res = n_res.expect(">=1 stage walked");
    let compress = compress.expect(">=1 block walked");
    let res_kernel = res_kernel.expect(">=1 block walked");

    // ---- Decoder SEANet walk (ratios as given — coarsest first) ------------
    let dec_convs = model_indices(st, "decoder", "conv.conv.weight");
    let dec_ups = model_indices(st, "decoder", "convtr.convtr.weight");
    let dec_blocks = model_indices(st, "decoder", "block.1.conv.conv.weight");
    if dec_convs.len() != 2 {
        return Err(parse_err(format!(
            "mimi: decoder must have exactly init + final plain convs, found model indices \
             {dec_convs:?}"
        )));
    }
    if dec_ups.len() != enc_ratios.len() {
        return Err(parse_err(format!(
            "mimi: decoder has {} upsample stages, encoder has {} downsample stages",
            dec_ups.len(),
            enc_ratios.len()
        )));
    }
    let dec_init = dec_convs[0];
    let dec_final = dec_convs[1];
    let dec_init_name = format!("decoder.model.{dec_init}.conv.conv.weight");
    let (dec_ch0, dec_init_in, dec_init_k) = shape3(st, &dec_init_name)?;
    let expect_ch0 = n_filters << enc_ratios.len();
    if dec_init_in != dimension || dec_ch0 != expect_ch0 || dec_init_k != kernel_size {
        return Err(parse_err(format!(
            "mimi: `{dec_init_name}` must be [{expect_ch0}, {dimension}, {kernel_size}], got \
             [{dec_ch0}, {dec_init_in}, {dec_init_k}]"
        )));
    }
    copy_conv(
        st,
        b,
        &format!("decoder.model.{dec_init}.conv.conv"),
        "mimi.dec.init",
        true,
        &mut written,
    )?;

    let mut ch = dec_ch0;
    let mut dec_ratios: Vec<usize> = Vec::new();
    for (s, &up_i) in dec_ups.iter().enumerate() {
        let up_name = format!("decoder.model.{up_i}.convtr.convtr.weight");
        // PyTorch ConvTranspose1d weight layout: [in_ch, out_ch, k].
        let (up_in, up_out, up_k) = shape3(st, &up_name)?;
        if up_in != ch || up_out != ch / 2 {
            return Err(parse_err(format!(
                "mimi: `{up_name}` must halve the channels [{ch}, {}, 2*ratio], got \
                 [{up_in}, {up_out}, {up_k}]",
                ch / 2
            )));
        }
        if up_k % 2 != 0 || up_k == 0 {
            return Err(parse_err(format!(
                "mimi: `{up_name}` kernel {up_k} is not 2*ratio"
            )));
        }
        dec_ratios.push(up_k / 2);
        copy_conv(
            st,
            b,
            &format!("decoder.model.{up_i}.convtr.convtr"),
            &format!("mimi.dec.s{s}.up"),
            true,
            &mut written,
        )?;
        ch = up_out;
        let next_bound = dec_ups.get(s + 1).copied().unwrap_or(dec_final);
        let stage_blocks: Vec<usize> = dec_blocks
            .iter()
            .copied()
            .filter(|&i| i > up_i && i < next_bound)
            .collect();
        if stage_blocks.len() != n_res {
            return Err(parse_err(format!(
                "mimi: decoder stage {s} has {} residual blocks, encoder walk fixed \
                 n_residual_layers = {n_res}",
                stage_blocks.len()
            )));
        }
        for (j, &bi) in stage_blocks.iter().enumerate() {
            let c1_name = format!("decoder.model.{bi}.block.1.conv.conv.weight");
            let (hidden, c1_in, k_res) = shape3(st, &c1_name)?;
            if c1_in != ch || k_res != res_kernel || hidden == 0 || ch / hidden != compress {
                return Err(parse_err(format!(
                    "mimi: `{c1_name}` must be [{}, {ch}, {res_kernel}], got [{hidden}, \
                     {c1_in}, {k_res}]",
                    ch / compress
                )));
            }
            let c2_name = format!("decoder.model.{bi}.block.3.conv.conv.weight");
            let (c2_out, c2_in, c2_k) = shape3(st, &c2_name)?;
            if c2_out != ch || c2_in != hidden || c2_k != 1 {
                return Err(parse_err(format!(
                    "mimi: `{c2_name}` must be [{ch}, {hidden}, 1], got [{c2_out}, {c2_in}, \
                     {c2_k}]"
                )));
            }
            copy_conv(
                st,
                b,
                &format!("decoder.model.{bi}.block.1.conv.conv"),
                &format!("mimi.dec.s{s}.b{j}.c1"),
                true,
                &mut written,
            )?;
            copy_conv(
                st,
                b,
                &format!("decoder.model.{bi}.block.3.conv.conv"),
                &format!("mimi.dec.s{s}.b{j}.c2"),
                true,
                &mut written,
            )?;
        }
    }
    let dec_final_name = format!("decoder.model.{dec_final}.conv.conv.weight");
    let (dec_out, dec_final_in, dec_final_k) = shape3(st, &dec_final_name)?;
    if dec_out != 1 || dec_final_in != ch || dec_final_k != last_kernel_size {
        return Err(parse_err(format!(
            "mimi: `{dec_final_name}` must be [1, {ch}, {last_kernel_size}], got [{dec_out}, \
             {dec_final_in}, {dec_final_k}]"
        )));
    }
    copy_conv(
        st,
        b,
        &format!("decoder.model.{dec_final}.conv.conv"),
        "mimi.dec.final",
        true,
        &mut written,
    )?;

    // Encoder consumes the (decoder-order) ratios reversed (seanet.py):
    // the walks must agree.
    let mut enc_rev = enc_ratios.clone();
    enc_rev.reverse();
    if enc_rev != dec_ratios {
        return Err(parse_err(format!(
            "mimi: encoder ratios (reversed) {enc_rev:?} != decoder ratios {dec_ratios:?}"
        )));
    }

    // ---- Frame resample pair (resample.py) ----------------------------------
    let down_name = "downsample.conv.conv.conv.weight";
    let (rd_out, rd_in, rd_k) = shape3(st, down_name)?;
    if rd_out != dimension || rd_in != dimension || rd_k % 2 != 0 || rd_k == 0 {
        return Err(parse_err(format!(
            "mimi: `{down_name}` must be a dense [{dimension}, {dimension}, 2*stride] conv, \
             got [{rd_out}, {rd_in}, {rd_k}]"
        )));
    }
    let stride = rd_k / 2;
    if find(st, "downsample.conv.conv.conv.bias").is_some() {
        return Err(parse_err(
            "mimi: `downsample.conv.conv.conv.bias` exists — ConvDownsample1d is bias-less \
             (resample.py); unknown variant"
                .to_owned(),
        ));
    }
    // Frame-rate arithmetic must close: frame_hop = seanet_hop * stride.
    let frame_hop = (MIMI_SAMPLE_RATE as u64 * 1000 / MIMI_FRAME_RATE_MHZ as u64) as usize;
    let seanet_hop: usize = dec_ratios.iter().product();
    if seanet_hop == 0 || frame_hop % seanet_hop != 0 || frame_hop / seanet_hop != stride {
        return Err(parse_err(format!(
            "mimi: resample stride {stride} does not close the rate arithmetic (frame hop \
             {frame_hop}, seanet hop {seanet_hop})"
        )));
    }
    copy_tensor(st, b, down_name, "mimi.enc.frame_down.weight")?;
    written += 1;

    let up_name = "upsample.convtr.convtr.convtr.weight";
    let (ru_in, ru_out, ru_k) = shape3(st, up_name)?;
    if find(st, "upsample.convtr.convtr.convtr.bias").is_some() {
        return Err(parse_err(
            "mimi: `upsample.convtr.convtr.convtr.bias` exists — ConvTrUpsample1d is \
             bias-less (resample.py); unknown variant"
                .to_owned(),
        ));
    }
    if ru_k != rd_k || ru_in != dimension {
        return Err(parse_err(format!(
            "mimi: `{up_name}` [{ru_in}, {ru_out}, {ru_k}] does not mirror the downsample \
             [{rd_out}, {rd_in}, {rd_k}]"
        )));
    }
    if ru_out == 1 {
        // Channel-wise conv-transpose (groups = dimension — the upstream
        // `upsample_channel_wise_bug=True` shape). Zero-expand to the
        // dense `[in, out, k]` equivalent the runtime consumes: output
        // channel o receives input channel i's kernel iff i == o. Exact.
        let (_, dw) = f32_tensor(st, up_name)?;
        let mut dense = vec![0.0f32; dimension * dimension * ru_k];
        for i in 0..dimension {
            for kk in 0..ru_k {
                dense[i * dimension * ru_k + i * ru_k + kk] = dw[i * ru_k + kk];
            }
        }
        add_f32_tensor(
            b,
            "mimi.dec.frame_up.weight",
            vec![dimension as u64, dimension as u64, ru_k as u64],
            &dense,
        )?;
        written += 1;
    } else if ru_out == dimension {
        copy_tensor(st, b, up_name, "mimi.dec.frame_up.weight")?;
        written += 1;
    } else {
        return Err(parse_err(format!(
            "mimi: `{up_name}` out_ch {ru_out} is neither 1 (channel-wise) nor {dimension} \
             (dense)"
        )));
    }

    // ---- Bottleneck transformers (transformer.py, bias-less linears) -------
    let mut tf_shape: Option<(usize, usize, usize)> = None; // (d, ff, n_layer)
    for (src_prefix, dst_prefix) in [
        ("encoder_transformer", "mimi.enc.tf"),
        ("decoder_transformer", "mimi.dec.tf"),
    ] {
        // ProjectedTransformer must be Identity (d_model == dimension).
        for proj in ["input_proj.weight", "output_projs.0.weight"] {
            let name = format!("{src_prefix}.{proj}");
            if find(st, &name).is_some() {
                return Err(parse_err(format!(
                    "mimi: `{name}` exists — ProjectedTransformer is Identity when d_model \
                     == seanet dimension; unknown variant"
                )));
            }
        }
        let mut l = 0usize;
        let (mut d_model, mut ff_dim) = (0usize, 0usize);
        loop {
            let base = format!("{src_prefix}.transformer.layers.{l}");
            let norm1 = format!("{base}.norm1.weight");
            let Some(norm1_t) = find(st, &norm1) else {
                break;
            };
            if norm1_t.shape.len() != 1 {
                return Err(parse_err(format!(
                    "mimi: `{norm1}` must be 1-D, got {:?}",
                    norm1_t.shape
                )));
            }
            let d = norm1_t.shape[0] as usize;
            if l == 0 {
                d_model = d;
            } else if d != d_model {
                return Err(parse_err(format!(
                    "mimi: `{norm1}` width {d} != layer-0 width {d_model}"
                )));
            }
            // The runtime models bias-less linears — a bias tensor means an
            // unknown variant (loud, never silently dropped).
            for forbidden in [
                format!("{base}.self_attn.in_proj_bias"),
                format!("{base}.self_attn.out_proj.bias"),
                format!("{base}.linear1.bias"),
                format!("{base}.linear2.bias"),
            ] {
                if find(st, &forbidden).is_some() {
                    return Err(parse_err(format!(
                        "mimi: `{forbidden}` exists — the moshi-native Mimi transformer \
                         linears are bias-less; unknown variant"
                    )));
                }
            }
            let dst = format!("{dst_prefix}{l}");

            let in_proj_name = format!("{base}.self_attn.in_proj_weight");
            let (ip_shape, ip) = f32_tensor(st, &in_proj_name)?;
            if ip_shape != vec![3 * d as u64, d as u64] {
                return Err(parse_err(format!(
                    "mimi: `{in_proj_name}` must be [{}, {d}] (fused q/k/v), got {ip_shape:?}",
                    3 * d
                )));
            }
            for (part, name) in ["q", "k", "v"].iter().enumerate() {
                let w = &ip[part * d * d..(part + 1) * d * d];
                add_f32_tensor(
                    b,
                    &format!("{dst}.{name}"),
                    vec![d as u64, d as u64],
                    &transpose_oi(w, d, d),
                )?;
                written += 1;
            }

            let out_proj_name = format!("{base}.self_attn.out_proj.weight");
            let (op_shape, op) = f32_tensor(st, &out_proj_name)?;
            if op_shape != vec![d as u64, d as u64] {
                return Err(parse_err(format!(
                    "mimi: `{out_proj_name}` must be [{d}, {d}], got {op_shape:?}"
                )));
            }
            add_f32_tensor(
                b,
                &format!("{dst}.o"),
                vec![d as u64, d as u64],
                &transpose_oi(&op, d, d),
            )?;
            written += 1;

            let l1_name = format!("{base}.linear1.weight");
            let (l1_shape, l1) = f32_tensor(st, &l1_name)?;
            if l1_shape.len() != 2 || l1_shape[1] != d as u64 {
                return Err(parse_err(format!(
                    "mimi: `{l1_name}` must be [ff, {d}], got {l1_shape:?}"
                )));
            }
            let ff = l1_shape[0] as usize;
            if l == 0 {
                ff_dim = ff;
            } else if ff != ff_dim {
                return Err(parse_err(format!(
                    "mimi: `{l1_name}` ff {ff} != layer-0 ff {ff_dim}"
                )));
            }
            add_f32_tensor(
                b,
                &format!("{dst}.fc1"),
                vec![d as u64, ff as u64],
                &transpose_oi(&l1, ff, d),
            )?;
            written += 1;

            let l2_name = format!("{base}.linear2.weight");
            let (l2_shape, l2) = f32_tensor(st, &l2_name)?;
            if l2_shape != vec![d as u64, ff as u64] {
                return Err(parse_err(format!(
                    "mimi: `{l2_name}` must be [{d}, {ff}], got {l2_shape:?}"
                )));
            }
            add_f32_tensor(
                b,
                &format!("{dst}.fc2"),
                vec![ff as u64, d as u64],
                &transpose_oi(&l2, d, ff),
            )?;
            written += 1;

            for (src_t, dst_t) in [
                ("norm1.weight", "ln1_gamma"),
                ("norm1.bias", "ln1_beta"),
                ("norm2.weight", "ln2_gamma"),
                ("norm2.bias", "ln2_beta"),
                ("layer_scale_1.scale", "ls1"),
                ("layer_scale_2.scale", "ls2"),
            ] {
                copy_tensor(st, b, &format!("{base}.{src_t}"), &format!("{dst}.{dst_t}"))?;
                written += 1;
            }
            l += 1;
        }
        if l == 0 {
            return Err(parse_err(format!(
                "mimi: `{src_prefix}` has no layers (`…transformer.layers.0.norm1.weight` \
                 not found)"
            )));
        }
        if d_model != dimension {
            return Err(parse_err(format!(
                "mimi: `{src_prefix}` d_model {d_model} != seanet dimension {dimension} — \
                 the bottleneck runs at the latent width"
            )));
        }
        match tf_shape {
            None => tf_shape = Some((d_model, ff_dim, l)),
            Some(prev) if prev == (d_model, ff_dim, l) => {}
            Some(prev) => {
                return Err(parse_err(format!(
                    "mimi: encoder/decoder transformers disagree ({prev:?} vs \
                     ({d_model}, {ff_dim}, {l}))"
                )));
            }
        }
    }
    let (d_model, ff_dim, n_layer) = tf_shape.expect("two sides walked");
    let n_head = MIMI_TRANSFORMER_N_HEAD as usize;
    if d_model % n_head != 0 || (d_model / n_head) % 2 != 0 {
        return Err(parse_err(format!(
            "mimi: d_model {d_model} does not split into {n_head} even-width heads \
             (loaders.py num_heads)"
        )));
    }

    // ---- Quantizer: raw (encode-side) codebooks + split input projs --------
    let ip_first_name = "quantizer.rvq_first.input_proj.weight";
    let (ipf_shape, _) = f32_tensor(st, ip_first_name)?;
    if ipf_shape.len() != 3 || ipf_shape[1] != dimension as u64 || ipf_shape[2] != 1 {
        return Err(parse_err(format!(
            "mimi: `{ip_first_name}` must be [q_dim, {dimension}, 1], got {ipf_shape:?}"
        )));
    }
    let q_dim = ipf_shape[0] as usize;
    let ip_rest_name = "quantizer.rvq_rest.input_proj.weight";
    let (ipr_shape, _) = f32_tensor(st, ip_rest_name)?;
    if ipr_shape != ipf_shape {
        return Err(parse_err(format!(
            "mimi: `{ip_rest_name}` shape {ipr_shape:?} != rvq_first's {ipf_shape:?}"
        )));
    }
    copy_tensor(st, b, ip_first_name, "mimi.enc.input_proj")?;
    written += 1;
    copy_tensor(st, b, ip_rest_name, "mimi.enc.input_proj_rest")?;
    written += 1;

    for cb in 0..n_codebooks {
        let (split, layer) = if cb == 0 {
            ("rvq_first", 0)
        } else {
            ("rvq_rest", cb - 1)
        };
        let base = format!("quantizer.{split}.vq.layers.{layer}._codebook");
        let (sum_shape, sum) = f32_tensor(st, &format!("{base}.embedding_sum"))?;
        let (_, usage) = f32_tensor(st, &format!("{base}.cluster_usage"))?;
        if sum_shape != vec![bins as u64, q_dim as u64] {
            return Err(parse_err(format!(
                "mimi: `{base}.embedding_sum` must be [{bins}, {q_dim}] (raw quantizer \
                 width), got {sum_shape:?}"
            )));
        }
        // Raw embedding = embedding_sum / clamp(cluster_usage, 1e-5)
        // (core_vq.py `EuclideanCodebook.embedding`) — the encode-side
        // nearest-neighbour table at the quantizer width (the decode side
        // uses the pre-projected DERIVED_TABLES_TENSOR instead).
        let mut emb = vec![0.0f32; bins * q_dim];
        for i in 0..bins {
            let denom = usage[i].max(CLUSTER_USAGE_EPSILON);
            for c in 0..q_dim {
                emb[i * q_dim + c] = sum[i * q_dim + c] / denom;
            }
        }
        add_f32_tensor(
            b,
            &format!("mimi.enc.cb{cb}"),
            vec![bins as u64, q_dim as u64],
            &emb,
        )?;
        written += 1;
    }

    // ---- vokra.mimi.* config chunk group ------------------------------------
    // Shape-derived values from the walks above; loaders.py constants for
    // the rest (module docs §5).
    b.add_u32(KEY_MIMI_SAMPLE_RATE, MIMI_SAMPLE_RATE);
    b.add_u32(KEY_MIMI_FRAME_RATE_MHZ, MIMI_FRAME_RATE_MHZ);
    b.add_u32(KEY_MIMI_SEANET_DIMENSION, dimension as u32);
    b.add_u32(KEY_MIMI_SEANET_N_FILTERS, n_filters as u32);
    b.add_u32(KEY_MIMI_SEANET_N_RESIDUAL_LAYERS, n_res as u32);
    b.add_u32(KEY_MIMI_SEANET_KERNEL_SIZE, kernel_size as u32);
    b.add_u32(KEY_MIMI_SEANET_RESIDUAL_KERNEL_SIZE, res_kernel as u32);
    b.add_u32(KEY_MIMI_SEANET_LAST_KERNEL_SIZE, last_kernel_size as u32);
    b.add_u32(KEY_MIMI_SEANET_COMPRESS, compress as u32);
    b.add_u32(KEY_MIMI_SEANET_DILATION_BASE, MIMI_SEANET_DILATION_BASE);
    b.add_u32(KEY_MIMI_SEANET_N_RATIOS, dec_ratios.len() as u32);
    for (i, r) in dec_ratios.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_MIMI_SEANET_RATIO}{i}"), *r as u32);
    }
    b.add_u32(KEY_MIMI_TRANSFORMER_D_MODEL, d_model as u32);
    b.add_u32(KEY_MIMI_TRANSFORMER_N_HEAD, MIMI_TRANSFORMER_N_HEAD);
    b.add_u32(KEY_MIMI_TRANSFORMER_N_LAYER, n_layer as u32);
    b.add_u32(KEY_MIMI_TRANSFORMER_FF_DIM, ff_dim as u32);
    b.add_u32(KEY_MIMI_TRANSFORMER_CONTEXT, MIMI_TRANSFORMER_CONTEXT);
    b.add_u32(KEY_MIMI_TRANSFORMER_MAX_PERIOD, MIMI_TRANSFORMER_MAX_PERIOD);
    b.add_f32(
        KEY_MIMI_TRANSFORMER_LAYER_SCALE,
        MIMI_TRANSFORMER_LAYER_SCALE,
    );
    b.add_u32(KEY_MIMI_QUANTIZER_DIMENSION, q_dim as u32);
    b.add_u32(KEY_MIMI_QUANTIZER_N_Q, n_codebooks as u32);
    b.add_u32(KEY_MIMI_QUANTIZER_BINS, bins as u32);
    b.add_u32(KEY_MIMI_QUANTIZER_INPUT_DIMENSION, dimension as u32);
    b.add_u32(KEY_MIMI_QUANTIZER_OUTPUT_DIMENSION, dimension as u32);

    Ok(written)
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
    #[allow(clippy::identity_op, clippy::erasing_op)] // keep the (cb * rows + i) * d_model formula shape visible
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

    /// A miniature but geometry-complete moshi-native checkpoint: real
    /// ratio structure [8, 6, 5, 4] (so the 24 kHz / 12.5 Hz rate
    /// arithmetic closes at resample stride 2), nf = 2, dimension = 16,
    /// n_res = 1, transformer d = 16 / ff = 32 / 1 layer, quantizer
    /// q_dim = 4 / bins = 4 / n_q = 3 (1 semantic + 2 acoustic).
    fn synthetic_mimi_full() -> Vec<(String, Vec<usize>, Vec<f32>)> {
        // Deterministic fill: value at flat index i is (i as f32) * salt.
        fn pushv(
            e: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
            name: &str,
            shape: Vec<usize>,
            salt: f32,
        ) {
            let n: usize = shape.iter().product();
            e.push((
                name.to_owned(),
                shape,
                (0..n).map(|i| i as f32 * salt).collect(),
            ));
        }
        let mut e: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        // Encoder SEANet: init 0; stages (block, down) at (1,3) (4,6)
        // (7,9) (10,12); final 14. Encoder ratio order = reversed dec.
        pushv(
            &mut e,
            "encoder.model.0.conv.conv.weight",
            vec![2, 1, 5],
            0.01,
        );
        pushv(&mut e, "encoder.model.0.conv.conv.bias", vec![2], 0.1);
        let enc = [
            (1usize, 3usize, 2usize, 4usize), // (block_i, down_i, ch, ratio)
            (4, 6, 4, 5),
            (7, 9, 8, 6),
            (10, 12, 16, 8),
        ];
        for &(bi, di, ch, r) in &enc {
            let hidden = ch / 2;
            pushv(
                &mut e,
                &format!("encoder.model.{bi}.block.1.conv.conv.weight"),
                vec![hidden, ch, 3],
                0.01,
            );
            pushv(
                &mut e,
                &format!("encoder.model.{bi}.block.1.conv.conv.bias"),
                vec![hidden],
                0.1,
            );
            pushv(
                &mut e,
                &format!("encoder.model.{bi}.block.3.conv.conv.weight"),
                vec![ch, hidden, 1],
                0.01,
            );
            pushv(
                &mut e,
                &format!("encoder.model.{bi}.block.3.conv.conv.bias"),
                vec![ch],
                0.1,
            );
            pushv(
                &mut e,
                &format!("encoder.model.{di}.conv.conv.weight"),
                vec![ch * 2, ch, 2 * r],
                0.001,
            );
            pushv(
                &mut e,
                &format!("encoder.model.{di}.conv.conv.bias"),
                vec![ch * 2],
                0.1,
            );
        }
        pushv(
            &mut e,
            "encoder.model.14.conv.conv.weight",
            vec![16, 32, 3],
            0.001,
        );
        pushv(&mut e, "encoder.model.14.conv.conv.bias", vec![16], 0.1);

        // Decoder SEANet: init 0; stages (up, block) at (2,3) (5,6) (8,9)
        // (11,12); final 14. Ratios as given (coarsest first).
        pushv(
            &mut e,
            "decoder.model.0.conv.conv.weight",
            vec![32, 16, 5],
            0.001,
        );
        pushv(&mut e, "decoder.model.0.conv.conv.bias", vec![32], 0.1);
        let dec = [
            (2usize, 3usize, 32usize, 8usize), // (up_i, block_i, in_ch, ratio)
            (5, 6, 16, 6),
            (8, 9, 8, 5),
            (11, 12, 4, 4),
        ];
        for &(ui, bi, ch_in, r) in &dec {
            let ch = ch_in / 2;
            let hidden = ch / 2;
            pushv(
                &mut e,
                &format!("decoder.model.{ui}.convtr.convtr.weight"),
                vec![ch_in, ch, 2 * r],
                0.001,
            );
            pushv(
                &mut e,
                &format!("decoder.model.{ui}.convtr.convtr.bias"),
                vec![ch],
                0.1,
            );
            pushv(
                &mut e,
                &format!("decoder.model.{bi}.block.1.conv.conv.weight"),
                vec![hidden, ch, 3],
                0.01,
            );
            pushv(
                &mut e,
                &format!("decoder.model.{bi}.block.1.conv.conv.bias"),
                vec![hidden],
                0.1,
            );
            pushv(
                &mut e,
                &format!("decoder.model.{bi}.block.3.conv.conv.weight"),
                vec![ch, hidden, 1],
                0.01,
            );
            pushv(
                &mut e,
                &format!("decoder.model.{bi}.block.3.conv.conv.bias"),
                vec![ch],
                0.1,
            );
        }
        pushv(
            &mut e,
            "decoder.model.14.conv.conv.weight",
            vec![1, 2, 3],
            0.01,
        );
        pushv(&mut e, "decoder.model.14.conv.conv.bias", vec![1], 0.1);

        // Frame resample: dense downsample, channel-wise upsample (the
        // upstream `upsample_channel_wise_bug=True` shape).
        pushv(
            &mut e,
            "downsample.conv.conv.conv.weight",
            vec![16, 16, 4],
            0.001,
        );
        pushv(
            &mut e,
            "upsample.convtr.convtr.convtr.weight",
            vec![16, 1, 4],
            0.01,
        );

        // Bottleneck transformers (bias-less linears, fused in_proj).
        for p in ["encoder_transformer", "decoder_transformer"] {
            let base = format!("{p}.transformer.layers.0");
            pushv(&mut e, &format!("{base}.norm1.weight"), vec![16], 0.1);
            pushv(&mut e, &format!("{base}.norm1.bias"), vec![16], 0.01);
            pushv(
                &mut e,
                &format!("{base}.self_attn.in_proj_weight"),
                vec![48, 16],
                0.001,
            );
            pushv(
                &mut e,
                &format!("{base}.self_attn.out_proj.weight"),
                vec![16, 16],
                0.001,
            );
            pushv(
                &mut e,
                &format!("{base}.linear1.weight"),
                vec![32, 16],
                0.001,
            );
            pushv(
                &mut e,
                &format!("{base}.linear2.weight"),
                vec![16, 32],
                0.001,
            );
            pushv(&mut e, &format!("{base}.norm2.weight"), vec![16], 0.1);
            pushv(&mut e, &format!("{base}.norm2.bias"), vec![16], 0.01);
            pushv(
                &mut e,
                &format!("{base}.layer_scale_1.scale"),
                vec![16],
                0.01,
            );
            pushv(
                &mut e,
                &format!("{base}.layer_scale_2.scale"),
                vec![16],
                0.01,
            );
        }

        // Quantizer: q_dim = 4, bins = 4, io = 16, 1 semantic + 2 acoustic.
        for split in ["rvq_first", "rvq_rest"] {
            pushv(
                &mut e,
                &format!("quantizer.{split}.input_proj.weight"),
                vec![4, 16, 1],
                0.01,
            );
            pushv(
                &mut e,
                &format!("quantizer.{split}.output_proj.weight"),
                vec![16, 4, 1],
                0.01,
            );
        }
        for (split, layer) in [("rvq_first", 0usize), ("rvq_rest", 0), ("rvq_rest", 1)] {
            let base = format!("quantizer.{split}.vq.layers.{layer}._codebook");
            pushv(&mut e, &format!("{base}.embedding_sum"), vec![4, 4], 0.5);
            e.push((
                format!("{base}.cluster_usage"),
                vec![4],
                vec![1.0, 2.0, 0.0, 4.0],
            ));
            pushv(&mut e, &format!("{base}._initialized"), vec![1], 1.0);
        }
        e
    }

    #[test]
    fn convert_maps_the_neural_chain_to_structural_names_and_config() {
        let entries = synthetic_mimi_full();
        let (b, report) = convert(build_safetensors(&entries)).expect("convert");
        assert_eq!(report.n_codebooks, 3);
        assert_eq!(report.codebook_size, 4);
        assert_eq!(report.d_model, 16);
        // enc: init 2 + 4·(2+2+2) + final 2 = 28; dec mirror = 28;
        // resample 2; transformers 2·(6 derived + 6 copies) = 24;
        // input projs 2; raw codebooks 3 → 87.
        assert_eq!(report.structural_written, 87);

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let u32_of = |key: &str| -> u32 {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => *v,
                other => panic!("`{key}` missing or not U32: {other:?}"),
            }
        };
        // Shape-derived seanet config.
        assert_eq!(u32_of("vokra.mimi.seanet.dimension"), 16);
        assert_eq!(u32_of("vokra.mimi.seanet.n_filters"), 2);
        assert_eq!(u32_of("vokra.mimi.seanet.n_residual_layers"), 1);
        assert_eq!(u32_of("vokra.mimi.seanet.kernel_size"), 5);
        assert_eq!(u32_of("vokra.mimi.seanet.residual_kernel_size"), 3);
        assert_eq!(u32_of("vokra.mimi.seanet.last_kernel_size"), 3);
        assert_eq!(u32_of("vokra.mimi.seanet.compress"), 2);
        assert_eq!(u32_of("vokra.mimi.seanet.n_ratios"), 4);
        for (i, r) in [8u32, 6, 5, 4].iter().enumerate() {
            assert_eq!(u32_of(&format!("vokra.mimi.seanet.ratio.{i}")), *r);
        }
        // Constants transcribed from loaders.py.
        assert_eq!(u32_of("vokra.mimi.sample_rate"), 24_000);
        assert_eq!(u32_of("vokra.mimi.frame_rate_mhz"), 12_500);
        assert_eq!(u32_of("vokra.mimi.transformer.n_head"), 8);
        assert_eq!(u32_of("vokra.mimi.transformer.context"), 250);
        // Shape-derived transformer + quantizer config.
        assert_eq!(u32_of("vokra.mimi.transformer.d_model"), 16);
        assert_eq!(u32_of("vokra.mimi.transformer.ff_dim"), 32);
        assert_eq!(u32_of("vokra.mimi.transformer.n_layer"), 1);
        assert_eq!(u32_of("vokra.mimi.quantizer.dimension"), 4);
        assert_eq!(u32_of("vokra.mimi.quantizer.n_q"), 3);
        assert_eq!(u32_of("vokra.mimi.quantizer.bins"), 4);
        assert_eq!(u32_of("vokra.mimi.quantizer.input_dimension"), 16);
        assert_eq!(u32_of("vokra.mimi.quantizer.output_dimension"), 16);

        let f32s = |name: &str| -> Vec<f32> {
            file.tensor_data(name)
                .unwrap_or_else(|| panic!("tensor `{name}` missing"))
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        // Structural conv copies exist with the exact element counts.
        assert_eq!(f32s("mimi.enc.init.weight").len(), 2 * 5);
        assert_eq!(f32s("mimi.enc.s3.down.weight").len(), 32 * 16 * 16);
        assert_eq!(f32s("mimi.enc.final.weight").len(), 16 * 32 * 3);
        assert_eq!(f32s("mimi.dec.init.weight").len(), 32 * 16 * 5);
        assert_eq!(f32s("mimi.dec.s0.up.weight").len(), 32 * 16 * 16);
        assert_eq!(f32s("mimi.dec.final.weight").len(), 2 * 3);
        assert_eq!(f32s("mimi.enc.frame_down.weight").len(), 16 * 16 * 4);

        // Transformer linears are transposed to the runtime w_t layout:
        // in_proj row o, col c (value = (o·16 + c)·0.001) lands at
        // w_t[c·16 + o] of the q part.
        let q = f32s("mimi.enc.tf0.q");
        assert_eq!(q.len(), 16 * 16);
        let (o, c) = (3usize, 7usize);
        let expect = (o * 16 + c) as f32 * 0.001;
        assert_eq!(q[c * 16 + o], expect, "q must be the [in, out] transpose");
        // v part starts at fused row 32: value = ((32 + o)·16 + c)·0.001.
        let v = f32s("mimi.enc.tf0.v");
        assert_eq!(v[c * 16 + o], ((32 + o) * 16 + c) as f32 * 0.001);
        // fc1 [d, ff] from linear1 [ff, d].
        let fc1 = f32s("mimi.dec.tf0.fc1");
        assert_eq!(fc1.len(), 16 * 32);
        let (ff_row, d_col) = (5usize, 2usize);
        assert_eq!(
            fc1[d_col * 32 + ff_row],
            (ff_row * 16 + d_col) as f32 * 0.001
        );

        // Channel-wise upsample zero-expanded to dense [in, out, k]:
        // diagonal carries the kernel, off-diagonal is zero.
        let up = f32s("mimi.dec.frame_up.weight");
        assert_eq!(up.len(), 16 * 16 * 4);
        for kk in 0..4 {
            assert_eq!(up[2 * 16 * 4 + 2 * 4 + kk], (2 * 4 + kk) as f32 * 0.01);
            assert_eq!(up[2 * 16 * 4 + 3 * 4 + kk], 0.0);
        }

        // Raw (un-projected) encode-side codebooks: sum / clamp(usage).
        let cb0 = f32s("mimi.enc.cb0");
        assert_eq!(cb0.len(), 4 * 4);
        // Row 1 (usage 2.0): sum row = [4·0.5, 5·0.5, 6·0.5, 7·0.5].
        assert_eq!(&cb0[4..8], &[1.0, 1.25, 1.5, 1.75]);
        // Row 2 exercises the clamp (usage 0 → 1e-5).
        assert_eq!(cb0[8], 8.0 * 0.5 / 1e-5);

        // Both split input projections land.
        assert_eq!(f32s("mimi.enc.input_proj").len(), 4 * 16);
        assert_eq!(f32s("mimi.enc.input_proj_rest").len(), 4 * 16);
    }

    #[test]
    fn convert_rejects_biased_transformer_linears_loudly() {
        // The runtime models bias-less transformer linears — a bias tensor
        // is an unknown variant and must fail the conversion (FR-EX-08),
        // never be silently dropped.
        let mut entries = synthetic_mimi_full();
        entries.push((
            "encoder_transformer.transformer.layers.0.linear1.bias".to_owned(),
            vec![32],
            vec![0.0; 32],
        ));
        let err = convert(build_safetensors(&entries)).expect_err("must reject");
        assert!(
            err.to_string().contains("linear1.bias"),
            "error names the offending tensor: {err}"
        );
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
