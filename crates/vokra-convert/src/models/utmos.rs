//! UTMOS22-strong (SaruLab neural MOS predictor): prepared safetensors +
//! config side-car → `vokra.utmos.*` GGUF (M5-15 T14).
//!
//! Input is what `tools/parity/utmos_prepare_checkpoint.py` writes: the
//! upstream Lightning `state_dict` flattened verbatim (224 tensors — the
//! unused `mask_emb` is dropped there) plus a JSON side-car whose fields the
//! script derived from the tensor shapes themselves. Output is the
//! `wav2vec2_regression.v1` schema the runtime scorer
//! (`crates/vokra-eval/src/metrics/utmos.rs`) binds.
//!
//! # This converter renames; it does not reshape
//!
//! Unlike the Whisper/Kokoro converters (which ship upstream names verbatim),
//! UTMOS is mapped onto the ADR `M4-18-utmos-arch` §(d) naming, because the
//! runtime scorer is written against that schema and the upstream names carry
//! three levels of Lightning/fairseq nesting. The mapping is total and
//! **exact-shape checked**: every declared tensor must exist with exactly the
//! dims the config implies, and any upstream tensor left over at the end is a
//! hard error (`ConvertError::Parse`) rather than a silent drop — a tensor we
//! do not understand is a tensor we cannot claim to have converted.
//!
//! # The one piece of arithmetic: the positional conv's weight-norm fold
//!
//! `encoder.pos_conv.0` is `torch.nn.utils.weight_norm(conv, name="weight",
//! dim=2)`, so the checkpoint stores `weight_g [1, 1, k]` + `weight_v
//! [d, d/groups, k]` instead of a dense kernel. `dim=2` means the norm is
//! taken over **all axes except the kernel axis**, one scalar per kernel tap:
//!
//! ```text
//! norm[k]      = ‖ v[:, :, k] ‖₂                (over d × d/groups entries)
//! weight[:,:,k] = v[:, :, k] * (g[0,0,k] / norm[k])
//! ```
//!
//! The `v * (g / norm)` association (rather than `(v * g) / norm`) mirrors
//! torch's own `_weight_norm`, so the fold is not a source of divergence.
//! This is the same offline-fold posture as the DAC converter's `out_proj`
//! (`models/dac.rs`), and it means the runtime binds a plain grouped conv.

use vokra_core::compliance::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_core::json::{self, JsonValue};
use vokra_core::safetensors::SafetensorsFile;

use crate::ConvertError;

/// `vokra.model.arch` value.
const ARCH: &str = "utmos";
/// The only variant this converter emits.
const ARCH_VARIANT_V1: &str = "wav2vec2_regression.v1";

// Upstream key prefixes (PyTorch-Lightning module tree).
const SSL: &str = "feature_extractors.0.ssl_model";
const DOMAIN_EMB: &str = "feature_extractors.1.embedding.weight";
const LD: &str = "output_layers.0";
const PROJ: &str = "output_layers.1.net";

/// Parsed UTMOS config side-car. Every field is required: the prepare script
/// derives them all from the checkpoint, so a missing one means the side-car
/// and the weights disagree — which must fail loudly, not default.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UtmosConvertConfig {
    pub(crate) sample_rate: u32,
    pub(crate) conv_channels: Vec<usize>,
    pub(crate) conv_kernels: Vec<usize>,
    pub(crate) conv_strides: Vec<usize>,
    pub(crate) conv_group_norm_layers: Vec<usize>,
    pub(crate) conv_group_norm_groups: Vec<usize>,
    pub(crate) group_norm_eps: f32,
    pub(crate) ln_eps: f32,
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) ffn_dim: usize,
    pub(crate) pos_conv_kernel: usize,
    pub(crate) pos_conv_groups: usize,
    pub(crate) domain_dim: usize,
    pub(crate) domain_id: usize,
    pub(crate) judge_dim: usize,
    pub(crate) judge_id: usize,
    pub(crate) blstm_hidden: usize,
    pub(crate) head_dims: Vec<usize>,
    pub(crate) head_scale: f32,
    pub(crate) head_offset: f32,
}

impl UtmosConvertConfig {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
        let miss = |key: &str, what: &str| {
            ConvertError::Parse(format!(
                "utmos config: required {what} field `{key}` is missing or mistyped (the \
                 utmos_prepare_checkpoint.py side-car emits it from the checkpoint's own tensor \
                 shapes)"
            ))
        };
        let uint = |key: &str| -> Result<usize, ConvertError> {
            root.get(key)
                .and_then(JsonValue::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| miss(key, "non-negative integer"))
        };
        let float = |key: &str| -> Result<f32, ConvertError> {
            root.get(key)
                .and_then(json_f64)
                .map(|v| v as f32)
                .ok_or_else(|| miss(key, "number"))
        };
        let uints = |key: &str| -> Result<Vec<usize>, ConvertError> {
            let arr = root
                .get(key)
                .and_then(JsonValue::as_array)
                .ok_or_else(|| miss(key, "array"))?;
            arr.iter()
                .map(|v| {
                    v.as_u64()
                        .map(|x| x as usize)
                        .ok_or_else(|| miss(key, "array of non-negative integers"))
                })
                .collect()
        };
        let text = |key: &str| -> Result<String, ConvertError> {
            root.get(key)
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
                .ok_or_else(|| miss(key, "string"))
        };

        // Reject a side-car describing something this converter does not
        // emit, rather than quietly relabelling it.
        for (key, want) in [
            ("arch_variant", ARCH_VARIANT_V1),
            ("conv_activation", "gelu"),
            ("norm", "post"),
            ("head_pool", "mean_after"),
            ("head_activation", "relu"),
        ] {
            let got = text(key)?;
            if got != want {
                return Err(ConvertError::Parse(format!(
                    "utmos config: `{key}` is {got:?}, but this converter emits only {want:?} \
                     (the upstream UTMOS22-strong stack)"
                )));
            }
        }

        let cfg = Self {
            sample_rate: uint("sample_rate")? as u32,
            conv_channels: uints("conv_channels")?,
            conv_kernels: uints("conv_kernels")?,
            conv_strides: uints("conv_strides")?,
            conv_group_norm_layers: uints("conv_group_norm_layers")?,
            conv_group_norm_groups: uints("conv_group_norm_groups")?,
            group_norm_eps: float("group_norm_eps")?,
            ln_eps: float("ln_eps")?,
            n_layer: uint("n_layer")?,
            n_head: uint("n_head")?,
            hidden_dim: uint("hidden_dim")?,
            ffn_dim: uint("ffn_dim")?,
            pos_conv_kernel: uint("pos_conv_kernel")?,
            pos_conv_groups: uint("pos_conv_groups")?,
            domain_dim: uint("domain_dim")?,
            domain_id: uint("domain_id")?,
            judge_dim: uint("judge_dim")?,
            judge_id: uint("judge_id")?,
            blstm_hidden: uint("blstm_hidden")?,
            head_dims: uints("head_dims")?,
            head_scale: float("head_scale")?,
            head_offset: float("head_offset")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), ConvertError> {
        let bad = |m: String| Err(ConvertError::Parse(format!("utmos config: {m}")));
        if self.conv_channels.is_empty()
            || self.conv_kernels.len() != self.conv_channels.len()
            || self.conv_strides.len() != self.conv_channels.len()
        {
            return bad("conv channels/kernels/strides must be non-empty and equal length".into());
        }
        if self.conv_group_norm_layers.len() != self.conv_group_norm_groups.len() {
            return bad("conv_group_norm_layers and _groups must have equal length".into());
        }
        for &l in &self.conv_group_norm_layers {
            if l >= self.conv_channels.len() {
                return bad(format!(
                    "conv_group_norm_layers references layer {l}, out of range"
                ));
            }
        }
        if self.hidden_dim == 0 || self.pos_conv_groups == 0 {
            return bad("hidden_dim and pos_conv_groups must be > 0".into());
        }
        if self.hidden_dim % self.pos_conv_groups != 0 {
            return bad(format!(
                "hidden_dim {} is not divisible by pos_conv_groups {}",
                self.hidden_dim, self.pos_conv_groups
            ));
        }
        if self.head_dims.len() != 2 || self.head_dims[1] != 1 {
            return bad(format!(
                "head_dims must be [hidden, 1] (upstream's Linear→ReLU→Linear), got {:?}",
                self.head_dims
            ));
        }
        if self.blstm_hidden == 0 || self.n_layer == 0 || self.n_head == 0 {
            return bad("blstm_hidden / n_layer / n_head must be > 0".into());
        }
        Ok(())
    }
}

/// `JsonValue` → `f64` accepting both int and float literals (the side-car
/// writes `1e-05` as a float but `2.0` may serialize as `2`).
fn json_f64(v: &JsonValue) -> Option<f64> {
    match v {
        JsonValue::Int(i) => Some(*i as f64),
        JsonValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// Conversion report.
#[derive(Debug, Default)]
pub(crate) struct UtmosReport {
    /// Tensors emitted into the GGUF.
    pub(crate) written: usize,
    /// Upstream tensors consumed (should equal the input count).
    pub(crate) consumed: usize,
}

/// Converts a prepared UTMOS safetensors buffer + config into a GGUF builder.
pub(crate) fn convert(
    bytes: Vec<u8>,
    cfg: &UtmosConvertConfig,
) -> Result<(GgufBuilder, UtmosReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let mut unconsumed: Vec<String> = st.tensors().iter().map(|t| t.name.clone()).collect();
    let mut report = UtmosReport::default();
    let mut b = GgufBuilder::new();

    // Reads an upstream tensor as f32, checking its dims exactly, and marks
    // it consumed.
    let take = |name: &str,
                dims: &[usize],
                unconsumed: &mut Vec<String>|
     -> Result<Vec<f32>, ConvertError> {
        let info = st.tensor_info(name).ok_or_else(|| {
            ConvertError::Parse(format!("utmos: checkpoint is missing tensor `{name}`"))
        })?;
        let want: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        if info.shape != want {
            return Err(ConvertError::Parse(format!(
                "utmos: tensor `{name}` has shape {:?}, expected {want:?}",
                info.shape
            )));
        }
        let v = st
            .tensor_f32(name)
            .map_err(|e| ConvertError::Parse(format!("utmos: reading `{name}`: {e}")))?;
        unconsumed.retain(|n| n != name);
        Ok(v)
    };

    // ---- metadata ----------------------------------------------------------
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    // Self-describing redistribution (publishing to a public model hub): the
    // artifact must carry its own licence, not rely on a consumer running
    // Vokra's registry resolver. Values transcribed from
    // docs/license-audit.md §3, which holds the primary-source citations.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some("utmos"),
        Some("sarulab-speech/UTMOS22 (MIT)"),
    );
    b.add_string("vokra.utmos.arch.variant", ARCH_VARIANT_V1);
    b.add_u32("vokra.utmos.sample_rate", cfg.sample_rate);
    b.add_metadata("vokra.utmos.conv.channels", u32_array(&cfg.conv_channels));
    b.add_metadata("vokra.utmos.conv.kernels", u32_array(&cfg.conv_kernels));
    b.add_metadata("vokra.utmos.conv.strides", u32_array(&cfg.conv_strides));
    b.add_string("vokra.utmos.conv.activation", "gelu");
    b.add_metadata(
        "vokra.utmos.conv.group_norm_layers",
        u32_array(&cfg.conv_group_norm_layers),
    );
    b.add_metadata(
        "vokra.utmos.conv.group_norm_groups",
        u32_array(&cfg.conv_group_norm_groups),
    );
    b.add_f32("vokra.utmos.conv.group_norm_eps", cfg.group_norm_eps);
    b.add_u32("vokra.utmos.transformer.n_layer", cfg.n_layer as u32);
    b.add_u32("vokra.utmos.transformer.n_head", cfg.n_head as u32);
    b.add_u32("vokra.utmos.transformer.hidden_dim", cfg.hidden_dim as u32);
    b.add_u32("vokra.utmos.transformer.ffn_dim", cfg.ffn_dim as u32);
    b.add_string("vokra.utmos.transformer.norm", "post");
    b.add_f32("vokra.utmos.transformer.ln_eps", cfg.ln_eps);
    b.add_u32("vokra.utmos.pos_conv.kernel", cfg.pos_conv_kernel as u32);
    b.add_u32("vokra.utmos.pos_conv.groups", cfg.pos_conv_groups as u32);
    b.add_u32("vokra.utmos.cond.domain_dim", cfg.domain_dim as u32);
    b.add_u32("vokra.utmos.cond.domain_id", cfg.domain_id as u32);
    b.add_u32("vokra.utmos.cond.judge_dim", cfg.judge_dim as u32);
    b.add_u32("vokra.utmos.cond.judge_id", cfg.judge_id as u32);
    b.add_u32("vokra.utmos.blstm.hidden", cfg.blstm_hidden as u32);
    b.add_metadata("vokra.utmos.head.dims", u32_array(&cfg.head_dims));
    b.add_string("vokra.utmos.head.pool", "mean_after");
    b.add_string("vokra.utmos.head.activation", "relu");
    b.add_f32("vokra.utmos.head.scale", cfg.head_scale);
    b.add_f32("vokra.utmos.head.offset", cfg.head_offset);

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
        report.written += 1;
        Ok(())
    };

    // ---- conv feature encoder ---------------------------------------------
    let mut c_in = 1usize;
    for (i, ((&c_out, &k), _)) in cfg
        .conv_channels
        .iter()
        .zip(&cfg.conv_kernels)
        .zip(&cfg.conv_strides)
        .enumerate()
    {
        let w = take(
            &format!("{SSL}.feature_extractor.conv_layers.{i}.0.weight"),
            &[c_out, c_in, k],
            &mut unconsumed,
        )?;
        emit(
            &mut b,
            &format!("utmos.conv.{i}.weight"),
            &[c_out, c_in, k],
            &w,
        )?;
        // Upstream sets conv_bias=False, so no `.0.bias` exists; the runtime
        // treats the bias as optional and finds none.
        if cfg.conv_group_norm_layers.contains(&i) {
            let gw = take(
                &format!("{SSL}.feature_extractor.conv_layers.{i}.2.weight"),
                &[c_out],
                &mut unconsumed,
            )?;
            let gb = take(
                &format!("{SSL}.feature_extractor.conv_layers.{i}.2.bias"),
                &[c_out],
                &mut unconsumed,
            )?;
            emit(
                &mut b,
                &format!("utmos.conv.{i}.group_norm.weight"),
                &[c_out],
                &gw,
            )?;
            emit(
                &mut b,
                &format!("utmos.conv.{i}.group_norm.bias"),
                &[c_out],
                &gb,
            )?;
        }
        c_in = c_out;
    }
    let c_last = c_in;
    let d = cfg.hidden_dim;

    // ---- feature LayerNorm + projection ------------------------------------
    for (src, dst) in [
        (format!("{SSL}.layer_norm"), "utmos.feature_ln"),
        (format!("{SSL}.encoder.layer_norm"), "utmos.enc_in_ln"),
    ] {
        let n = if dst == "utmos.feature_ln" { c_last } else { d };
        let w = take(&format!("{src}.weight"), &[n], &mut unconsumed)?;
        let bi = take(&format!("{src}.bias"), &[n], &mut unconsumed)?;
        emit(&mut b, &format!("{dst}.weight"), &[n], &w)?;
        emit(&mut b, &format!("{dst}.bias"), &[n], &bi)?;
    }
    let pw = take(
        &format!("{SSL}.post_extract_proj.weight"),
        &[d, c_last],
        &mut unconsumed,
    )?;
    let pb = take(
        &format!("{SSL}.post_extract_proj.bias"),
        &[d],
        &mut unconsumed,
    )?;
    emit(&mut b, "utmos.feat_proj.weight", &[d, c_last], &pw)?;
    emit(&mut b, "utmos.feat_proj.bias", &[d], &pb)?;

    // ---- positional conv (weight-norm folded, see module docs) -------------
    let in_per = d / cfg.pos_conv_groups;
    let k = cfg.pos_conv_kernel;
    let g = take(
        &format!("{SSL}.encoder.pos_conv.0.weight_g"),
        &[1, 1, k],
        &mut unconsumed,
    )?;
    let v = take(
        &format!("{SSL}.encoder.pos_conv.0.weight_v"),
        &[d, in_per, k],
        &mut unconsumed,
    )?;
    let pcb = take(
        &format!("{SSL}.encoder.pos_conv.0.bias"),
        &[d],
        &mut unconsumed,
    )?;
    let folded = fold_weight_norm_dim2(&g, &v, d, in_per, k)?;
    emit(&mut b, "utmos.pos_conv.weight", &[d, in_per, k], &folded)?;
    emit(&mut b, "utmos.pos_conv.bias", &[d], &pcb)?;

    // ---- transformer blocks ------------------------------------------------
    for i in 0..cfg.n_layer {
        let src = format!("{SSL}.encoder.layers.{i}");
        for (up, dn) in [
            ("self_attn.q_proj", "attn.q"),
            ("self_attn.k_proj", "attn.k"),
            ("self_attn.v_proj", "attn.v"),
            ("self_attn.out_proj", "attn.o"),
        ] {
            let w = take(&format!("{src}.{up}.weight"), &[d, d], &mut unconsumed)?;
            let bi = take(&format!("{src}.{up}.bias"), &[d], &mut unconsumed)?;
            emit(&mut b, &format!("utmos.enc.{i}.{dn}.weight"), &[d, d], &w)?;
            emit(&mut b, &format!("utmos.enc.{i}.{dn}.bias"), &[d], &bi)?;
        }
        // Post-norm placement: `self_attn_layer_norm` runs after the attention
        // residual (= the runtime's ln1) and `final_layer_norm` after the MLP
        // residual (= ln2). Swapping these would still load and still produce
        // finite scores, so the mapping is pinned by the stage-by-stage parity
        // fixtures, not by shape alone.
        for (up, dn) in [("self_attn_layer_norm", "ln1"), ("final_layer_norm", "ln2")] {
            let w = take(&format!("{src}.{up}.weight"), &[d], &mut unconsumed)?;
            let bi = take(&format!("{src}.{up}.bias"), &[d], &mut unconsumed)?;
            emit(&mut b, &format!("utmos.enc.{i}.{dn}.weight"), &[d], &w)?;
            emit(&mut b, &format!("utmos.enc.{i}.{dn}.bias"), &[d], &bi)?;
        }
        for (up, dn, o, ii) in [
            ("fc1", "mlp.fc1", cfg.ffn_dim, d),
            ("fc2", "mlp.fc2", d, cfg.ffn_dim),
        ] {
            let w = take(&format!("{src}.{up}.weight"), &[o, ii], &mut unconsumed)?;
            let bi = take(&format!("{src}.{up}.bias"), &[o], &mut unconsumed)?;
            emit(&mut b, &format!("utmos.enc.{i}.{dn}.weight"), &[o, ii], &w)?;
            emit(&mut b, &format!("utmos.enc.{i}.{dn}.bias"), &[o], &bi)?;
        }
    }

    // ---- conditioning embeddings ------------------------------------------
    let dom_rows = st
        .tensor_info(DOMAIN_EMB)
        .map(|t| t.shape[0] as usize)
        .ok_or_else(|| ConvertError::Parse(format!("utmos: missing `{DOMAIN_EMB}`")))?;
    let dom = take(DOMAIN_EMB, &[dom_rows, cfg.domain_dim], &mut unconsumed)?;
    emit(
        &mut b,
        "utmos.cond.domain_emb",
        &[dom_rows, cfg.domain_dim],
        &dom,
    )?;
    let judge_key = format!("{LD}.judge_embedding.weight");
    let judge_rows = st
        .tensor_info(&judge_key)
        .map(|t| t.shape[0] as usize)
        .ok_or_else(|| ConvertError::Parse(format!("utmos: missing `{judge_key}`")))?;
    let jud = take(&judge_key, &[judge_rows, cfg.judge_dim], &mut unconsumed)?;
    emit(
        &mut b,
        "utmos.cond.judge_emb",
        &[judge_rows, cfg.judge_dim],
        &jud,
    )?;

    // ---- BLSTM -------------------------------------------------------------
    let gates = 4 * cfg.blstm_hidden;
    let blstm_in = d + cfg.domain_dim + cfg.judge_dim;
    for (suffix, dir) in [("", "fwd"), ("_reverse", "bwd")] {
        for (up, dn, dims) in [
            (
                format!("weight_ih_l0{suffix}"),
                "w_ih",
                vec![gates, blstm_in],
            ),
            (
                format!("weight_hh_l0{suffix}"),
                "w_hh",
                vec![gates, cfg.blstm_hidden],
            ),
            (format!("bias_ih_l0{suffix}"), "b_ih", vec![gates]),
            (format!("bias_hh_l0{suffix}"), "b_hh", vec![gates]),
        ] {
            let t = take(&format!("{LD}.decoder_rnn.{up}"), &dims, &mut unconsumed)?;
            emit(&mut b, &format!("utmos.blstm.{dir}.{dn}"), &dims, &t)?;
        }
    }

    // ---- regression head ---------------------------------------------------
    // Upstream's Sequential indices 0 and 3 (1 = ReLU, 2 = Dropout, both
    // parameterless) become head linears 0 and 1.
    let head_in = 2 * cfg.blstm_hidden;
    for (idx, (up, o, ii)) in [
        ("0", cfg.head_dims[0], head_in),
        ("3", cfg.head_dims[1], cfg.head_dims[0]),
    ]
    .into_iter()
    .enumerate()
    {
        let w = take(&format!("{PROJ}.{up}.weight"), &[o, ii], &mut unconsumed)?;
        let bi = take(&format!("{PROJ}.{up}.bias"), &[o], &mut unconsumed)?;
        emit(&mut b, &format!("utmos.head.{idx}.weight"), &[o, ii], &w)?;
        emit(&mut b, &format!("utmos.head.{idx}.bias"), &[o], &bi)?;
    }

    // ---- completeness ------------------------------------------------------
    if !unconsumed.is_empty() {
        let mut names = unconsumed.clone();
        names.sort();
        names.truncate(8);
        return Err(ConvertError::Parse(format!(
            "utmos: {} upstream tensor(s) were not consumed by the mapping — refusing to emit a \
             GGUF that silently drops weights (FR-EX-08). First: {names:?}",
            unconsumed.len()
        )));
    }
    report.consumed = st.tensors().len();
    Ok((b, report))
}

/// Folds `torch.nn.utils.weight_norm(conv, name="weight", dim=2)` into a
/// dense `[out, in_per_group, k]` kernel (see the module docs for the
/// definition and the association order).
fn fold_weight_norm_dim2(
    g: &[f32],
    v: &[f32],
    out: usize,
    in_per: usize,
    k: usize,
) -> Result<Vec<f32>, ConvertError> {
    let mut w = vec![0.0f32; out * in_per * k];
    for kk in 0..k {
        // ‖v[:, :, kk]‖₂ over the out × in_per plane. Accumulated in f64:
        // the plane has d × d/groups ≈ 37k entries for the upstream shape.
        let mut sq = 0.0f64;
        for o in 0..out {
            for i in 0..in_per {
                let x = f64::from(v[(o * in_per + i) * k + kk]);
                sq += x * x;
            }
        }
        let norm = sq.sqrt();
        if norm == 0.0 {
            return Err(ConvertError::Parse(format!(
                "utmos: pos_conv weight_v kernel tap {kk} has zero norm — cannot fold weight_norm \
                 (corrupt checkpoint?)"
            )));
        }
        // `v * (g / norm)`, matching torch's own `_weight_norm` association.
        let scale = (f64::from(g[kk]) / norm) as f32;
        for o in 0..out {
            for i in 0..in_per {
                let idx = (o * in_per + i) * k + kk;
                w[idx] = v[idx] * scale;
            }
        }
    }
    Ok(w)
}

fn u32_array(values: &[usize]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::U32,
        values: values
            .iter()
            .map(|&v| GgufMetadataValue::U32(v as u32))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature but structurally complete config: same shape family as
    /// upstream, small enough to build a synthetic checkpoint by hand.
    fn cfg_json() -> String {
        r#"{
          "arch_variant": "wav2vec2_regression.v1",
          "sample_rate": 16000,
          "conv_channels": [4, 4],
          "conv_kernels": [5, 3],
          "conv_strides": [3, 2],
          "conv_activation": "gelu",
          "conv_group_norm_layers": [0],
          "conv_group_norm_groups": [4],
          "group_norm_eps": 1e-05,
          "ln_eps": 1e-05,
          "n_layer": 1,
          "n_head": 2,
          "hidden_dim": 6,
          "ffn_dim": 12,
          "norm": "post",
          "pos_conv_kernel": 4,
          "pos_conv_groups": 2,
          "domain_dim": 3,
          "domain_id": 0,
          "judge_dim": 5,
          "judge_id": 2,
          "blstm_hidden": 4,
          "head_dims": [7, 1],
          "head_pool": "mean_after",
          "head_activation": "relu",
          "head_scale": 2.0,
          "head_offset": 3.0
        }"#
        .to_owned()
    }

    fn ramp(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Builds a synthetic upstream-named safetensors checkpoint for `cfg`.
    fn synth_checkpoint(cfg: &UtmosConvertConfig, skip: Option<&str>, extra: bool) -> Vec<u8> {
        let d = cfg.hidden_dim;
        let mut entries: Vec<(String, Vec<usize>)> = Vec::new();
        let mut c_in = 1usize;
        for (i, (&c_out, &k)) in cfg.conv_channels.iter().zip(&cfg.conv_kernels).enumerate() {
            entries.push((
                format!("{SSL}.feature_extractor.conv_layers.{i}.0.weight"),
                vec![c_out, c_in, k],
            ));
            if cfg.conv_group_norm_layers.contains(&i) {
                entries.push((
                    format!("{SSL}.feature_extractor.conv_layers.{i}.2.weight"),
                    vec![c_out],
                ));
                entries.push((
                    format!("{SSL}.feature_extractor.conv_layers.{i}.2.bias"),
                    vec![c_out],
                ));
            }
            c_in = c_out;
        }
        let c_last = c_in;
        entries.push((format!("{SSL}.layer_norm.weight"), vec![c_last]));
        entries.push((format!("{SSL}.layer_norm.bias"), vec![c_last]));
        entries.push((format!("{SSL}.encoder.layer_norm.weight"), vec![d]));
        entries.push((format!("{SSL}.encoder.layer_norm.bias"), vec![d]));
        entries.push((format!("{SSL}.post_extract_proj.weight"), vec![d, c_last]));
        entries.push((format!("{SSL}.post_extract_proj.bias"), vec![d]));
        entries.push((
            format!("{SSL}.encoder.pos_conv.0.weight_g"),
            vec![1, 1, cfg.pos_conv_kernel],
        ));
        entries.push((
            format!("{SSL}.encoder.pos_conv.0.weight_v"),
            vec![d, d / cfg.pos_conv_groups, cfg.pos_conv_kernel],
        ));
        entries.push((format!("{SSL}.encoder.pos_conv.0.bias"), vec![d]));
        for i in 0..cfg.n_layer {
            let s = format!("{SSL}.encoder.layers.{i}");
            for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                entries.push((format!("{s}.self_attn.{p}.weight"), vec![d, d]));
                entries.push((format!("{s}.self_attn.{p}.bias"), vec![d]));
            }
            for p in ["self_attn_layer_norm", "final_layer_norm"] {
                entries.push((format!("{s}.{p}.weight"), vec![d]));
                entries.push((format!("{s}.{p}.bias"), vec![d]));
            }
            entries.push((format!("{s}.fc1.weight"), vec![cfg.ffn_dim, d]));
            entries.push((format!("{s}.fc1.bias"), vec![cfg.ffn_dim]));
            entries.push((format!("{s}.fc2.weight"), vec![d, cfg.ffn_dim]));
            entries.push((format!("{s}.fc2.bias"), vec![d]));
        }
        entries.push((DOMAIN_EMB.to_owned(), vec![3, cfg.domain_dim]));
        entries.push((
            format!("{LD}.judge_embedding.weight"),
            vec![10, cfg.judge_dim],
        ));
        let g = 4 * cfg.blstm_hidden;
        let bin = d + cfg.domain_dim + cfg.judge_dim;
        for suf in ["", "_reverse"] {
            entries.push((format!("{LD}.decoder_rnn.weight_ih_l0{suf}"), vec![g, bin]));
            entries.push((
                format!("{LD}.decoder_rnn.weight_hh_l0{suf}"),
                vec![g, cfg.blstm_hidden],
            ));
            entries.push((format!("{LD}.decoder_rnn.bias_ih_l0{suf}"), vec![g]));
            entries.push((format!("{LD}.decoder_rnn.bias_hh_l0{suf}"), vec![g]));
        }
        entries.push((
            format!("{PROJ}.0.weight"),
            vec![cfg.head_dims[0], 2 * cfg.blstm_hidden],
        ));
        entries.push((format!("{PROJ}.0.bias"), vec![cfg.head_dims[0]]));
        entries.push((format!("{PROJ}.3.weight"), vec![1, cfg.head_dims[0]]));
        entries.push((format!("{PROJ}.3.bias"), vec![1]));
        if extra {
            entries.push((format!("{SSL}.mystery_tensor"), vec![3]));
        }

        // Minimal safetensors writer (header JSON + F32 payload).
        let mut header = String::from("{");
        let mut payload: Vec<u8> = Vec::new();
        let mut first = true;
        let mut seed = 1u32;
        for (name, dims) in &entries {
            if Some(name.as_str()) == skip {
                continue;
            }
            let n: usize = dims.iter().product();
            let start = payload.len();
            seed = seed.wrapping_add(17);
            for x in ramp(n, seed) {
                payload.extend_from_slice(&x.to_le_bytes());
            }
            if !first {
                header.push(',');
            }
            first = false;
            let shape = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{start},{}]}}",
                payload.len()
            ));
        }
        header.push('}');
        let hb = header.into_bytes();
        let mut out = Vec::with_capacity(8 + hb.len() + payload.len());
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn config_parses_and_rejects_foreign_variants() {
        let cfg = UtmosConvertConfig::parse(cfg_json().as_bytes()).expect("parses");
        assert_eq!(cfg.hidden_dim, 6);
        assert_eq!(cfg.conv_group_norm_layers, vec![0]);
        assert_eq!(cfg.head_scale, 2.0);

        // A side-car describing a different stack must not be relabelled.
        let bad = cfg_json().replace("wav2vec2_regression.v1", "wav2vec2_regression.v0");
        let err = UtmosConvertConfig::parse(bad.as_bytes()).expect_err("v0 side-car");
        assert!(format!("{err}").contains("arch_variant"), "got: {err}");

        let bad = cfg_json().replace(
            "\"head_activation\": \"relu\"",
            "\"head_activation\": \"gelu\"",
        );
        assert!(UtmosConvertConfig::parse(bad.as_bytes()).is_err());

        // Missing field.
        let bad = cfg_json().replace("\"blstm_hidden\": 4,", "");
        let err = UtmosConvertConfig::parse(bad.as_bytes()).expect_err("missing field");
        assert!(format!("{err}").contains("blstm_hidden"), "got: {err}");
    }

    #[test]
    fn converts_a_complete_checkpoint_and_emits_the_v1_schema() {
        let cfg = UtmosConvertConfig::parse(cfg_json().as_bytes()).unwrap();
        let bytes = synth_checkpoint(&cfg, None, false);
        let (b, report) = convert(bytes, &cfg).expect("convert");
        assert!(report.written > 0);
        let gguf = vokra_core::gguf::GgufFile::parse(b.to_bytes().unwrap()).expect("parse gguf");
        assert_eq!(
            gguf.get("vokra.utmos.arch.variant").unwrap().as_str(),
            Some(ARCH_VARIANT_V1)
        );
        for name in [
            "utmos.conv.0.weight",
            "utmos.conv.0.group_norm.weight",
            "utmos.feature_ln.weight",
            "utmos.feat_proj.weight",
            "utmos.pos_conv.weight",
            "utmos.enc_in_ln.weight",
            "utmos.enc.0.attn.q.weight",
            "utmos.enc.0.ln1.weight",
            "utmos.enc.0.mlp.fc1.weight",
            "utmos.cond.domain_emb",
            "utmos.cond.judge_emb",
            "utmos.blstm.fwd.w_ih",
            "utmos.blstm.bwd.b_hh",
            "utmos.head.0.weight",
            "utmos.head.1.weight",
        ] {
            assert!(gguf.tensor_info(name).is_some(), "missing `{name}`");
        }
        // conv layer 1 has no GroupNorm (upstream group-norms layer 0 only).
        assert!(gguf.tensor_info("utmos.conv.1.group_norm.weight").is_none());
    }

    #[test]
    fn an_unconsumed_upstream_tensor_is_a_hard_error() {
        // A weight we do not understand must not be silently dropped.
        let cfg = UtmosConvertConfig::parse(cfg_json().as_bytes()).unwrap();
        let bytes = synth_checkpoint(&cfg, None, true);
        let err = convert(bytes, &cfg).expect_err("extra tensor");
        let msg = format!("{err}");
        assert!(msg.contains("not consumed"), "got: {msg}");
        assert!(msg.contains("mystery_tensor"), "must name it: {msg}");
    }

    #[test]
    fn a_missing_upstream_tensor_is_named() {
        let cfg = UtmosConvertConfig::parse(cfg_json().as_bytes()).unwrap();
        let bytes = synth_checkpoint(&cfg, Some(&format!("{LD}.decoder_rnn.bias_hh_l0")), false);
        let err = convert(bytes, &cfg).expect_err("missing tensor");
        assert!(
            format!("{err}").contains("bias_hh_l0"),
            "must name it: {err}"
        );
    }

    #[test]
    fn weight_norm_fold_matches_the_definition() {
        // dim=2 ⇒ one scalar norm per kernel tap, over the out × in plane.
        // out=2, in=1, k=2 with v[:, :, 0] = [3, 4] (norm 5) and
        // v[:, :, 1] = [0, 1] (norm 1); g = [10, 7].
        let v = vec![3.0f32, 0.0, /* o=0 */ 4.0, 1.0 /* o=1 */];
        let g = vec![10.0f32, 7.0];
        let w = fold_weight_norm_dim2(&g, &v, 2, 1, 2).unwrap();
        // tap 0: scale 10/5 = 2 → [3*2, 4*2] = [6, 8]
        assert!((w[0] - 6.0).abs() < 1e-6, "w[o=0,k=0] = {}", w[0]);
        assert!((w[2] - 8.0).abs() < 1e-6, "w[o=1,k=0] = {}", w[2]);
        // tap 1: scale 7/1 = 7 → [0*7, 1*7] = [0, 7]
        assert!((w[1] - 0.0).abs() < 1e-6, "w[o=0,k=1] = {}", w[1]);
        assert!((w[3] - 7.0).abs() < 1e-6, "w[o=1,k=1] = {}", w[3]);
    }

    #[test]
    fn weight_norm_fold_rejects_a_zero_norm_tap() {
        let v = vec![0.0f32, 0.0];
        let g = vec![1.0f32, 1.0];
        assert!(fold_weight_norm_dim2(&g, &v, 1, 1, 2).is_err());
    }
}
