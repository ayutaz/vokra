//! CAM++ (3D-Speaker) speaker encoder: ONNX to GGUF conversion (M0-08).
//!
//! Input: a `campplus.onnx` graph (INPUT `input[B, seq, 80]` 80-d fbank,
//! OUTPUT `output[B, 192]` speaker embedding). Output: a GGUF carrying every
//! weight tensor under clean, module-scoped names plus the `vokra.campplus.*`
//! metadata the native forward pass (`vokra-models`, a later stage) loads
//! against. No ONNX is ever touched at runtime (FR-LD-05); this offline tool is
//! the only place it is handled.
//!
//! # Weight naming: canonical names from the node scope path
//!
//! CAM++ is exported so that its FCM front-end convolutions (and every conv/BN
//! that had a BatchNorm folded into it) carry export-run-specific opaque
//! `onnx::Conv_4423`-style initializer names, while the untouched D-TDNN convs
//! and standalone BatchNorms keep their clean `xvector.block1.tdnnd1.…` names.
//! Loading against the opaque names would be fragile, so the converter derives
//! a *canonical* name for every weight from the producing node's scope path:
//! the `Conv` at `/head/conv1/Conv` names its weight `head.conv1.weight` and
//! its (BN-fold) bias `head.conv1.bias`; the `BatchNormalization` at
//! `/xvector/block1/tdnnd1/nonlinear1/batchnorm/…` names its four buffers
//! `<base>.{weight,bias,running_mean,running_var}`. The doubled PyTorch
//! `ModuleList` scope (`/head/layer1/layer1.0/conv1`) is collapsed to
//! `head.layer1.0.conv1`, matching the upstream `state_dict`. Every clean
//! initializer name derives back to itself; every opaque one gets its clean
//! module name. The runtime therefore loads self-describing tensor names.
//!
//! # Affine-free final BatchNorm
//!
//! The tail `xvector.dense.nonlinear.batchnorm` is affine-free: its scale/bias
//! are exported as folded `Constant` nodes (all-ones / all-zeros), not
//! initializers, so only `running_mean` / `running_var` are present as weights.
//! The converter *synthesizes* `…weight = ones` / `…bias = zeros` so the
//! runtime can fold all 56 BatchNorms uniformly to per-channel scale/shift.
//!
//! # dtype
//!
//! CAM++ weights are all FP32; the converter widens the (defensive) FP16 case
//! to FP32 too, so the runtime loads a single dtype and computes in FP32.

use std::collections::{BTreeMap, HashMap};
use vokra_core::compliance::LicenseClass;

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::onnx::{self, ONNX_DTYPE_FLOAT, ONNX_DTYPE_FLOAT16, OnnxGraph, OnnxNode};

/// `vokra.model.arch` value written for CAM++ speaker-encoder GGUFs.
pub(crate) const ARCH: &str = "campplus";

// --- vokra.campplus.* metadata keys (M0-08 chunk design) --------------------

/// `vokra.campplus.block_config` — D-TDNN dense layers per block (`ARRAY<U32>`).
const KEY_BLOCK_CONFIG: &str = "vokra.campplus.block_config";
/// `vokra.campplus.growth` — channel growth per dense layer (`UINT32`).
const KEY_GROWTH: &str = "vokra.campplus.growth";
/// `vokra.campplus.dilations` — `cam_layer.linear_local` dilation per block
/// (`ARRAY<U32>`).
const KEY_DILATIONS: &str = "vokra.campplus.dilations";
/// `vokra.campplus.cam_seg_len` — CAM `seg_pool` `AvgPool1d` kernel/stride
/// (`UINT32`).
const KEY_CAM_SEG_LEN: &str = "vokra.campplus.cam_seg_len";
/// `vokra.campplus.bn_eps` — BatchNorm epsilon used for the load-time fold
/// (`FLOAT32`).
const KEY_BN_EPS: &str = "vokra.campplus.bn_eps";
/// `vokra.campplus.feat_dim` — input fbank feature dimension (`UINT32`).
const KEY_FEAT_DIM: &str = "vokra.campplus.feat_dim";
/// `vokra.campplus.embed_dim` — output speaker-embedding dimension (`UINT32`).
const KEY_EMBED_DIM: &str = "vokra.campplus.embed_dim";

// --- Verified architectural constants (from a full graph walk of the
// reference `campplus.onnx`; the runtime hard-codes the same topology). -------

/// `cam_layer.linear_local` dilation per D-TDNN block (verified: block1 = 1,
/// block2/3 = 2 — graph dilation histogram {1: 12, 2: 40}).
const DILATIONS: [u32; 3] = [1, 2, 2];
/// CAM `seg_pool` `AvgPool1d` kernel = stride (verified `kernel_shape=100`,
/// `strides=100`, `ceil_mode=1`).
const CAM_SEG_LEN: u32 = 100;
/// BatchNorm epsilon (verified: every BN carries `epsilon=1e-5`).
const BN_EPS: f32 = 1e-5;
/// Input fbank feature dimension (graph input `input[B, seq, 80]`).
const FEAT_DIM: u32 = 80;
/// Fallbacks used only if a value cannot be derived from the graph.
const DEFAULT_BLOCK_CONFIG: [u32; 3] = [12, 24, 16];
const DEFAULT_GROWTH: u32 = 32;
const DEFAULT_EMBED_DIM: u32 = 192;

/// Outcome of a CAM++ conversion.
#[derive(Debug, Default)]
pub(crate) struct CamPlusReport {
    /// Weight tensors written from graph initializers.
    pub(crate) written: usize,
    /// Opaque `onnx::*` initializers whose clean module name was recovered.
    pub(crate) renamed: usize,
    /// Affine-free BatchNorm scale/bias tensors synthesized (ones / zeros).
    pub(crate) synthesized: usize,
    /// Float initializers with no producing conv/BN node (should be 0 — a guard
    /// against an unexpected export where a weight went unnamed).
    pub(crate) unmapped: usize,
    /// Non-float initializers skipped (int64 shape/index constants — CAM++ has
    /// none, but the guard is kept).
    pub(crate) skipped_non_float: usize,
    /// D-TDNN dense-layer count per block, derived from the initializer names.
    pub(crate) block_config: Vec<u32>,
}

/// Converts CAM++ `onnx_bytes` into a populated GGUF builder plus a report.
pub(crate) fn convert(onnx_bytes: &[u8]) -> Result<(GgufBuilder, CamPlusReport), ConvertError> {
    let graph = onnx::read_graph(onnx_bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
    let init_names: HashMap<&str, &onnx::OnnxInitializer> = graph
        .initializers
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    // Walk weight-bearing nodes, mapping each initializer input to its
    // canonical, scope-derived name; collect affine-free BNs to synthesize.
    let mut rename: HashMap<String, String> = HashMap::new();
    let mut synth: Vec<(String, u32)> = Vec::new();
    for node in &graph.nodes {
        let Some(base) = canonical_base(node) else {
            continue;
        };
        match node.op_type.as_str() {
            "Conv" | "ConvTranspose" => {
                map_input(&mut rename, &init_names, node, 1, &base, "weight");
                map_input(&mut rename, &init_names, node, 2, &base, "bias");
            }
            "BatchNormalization" => {
                let scale_is_init = node
                    .inputs
                    .get(1)
                    .is_some_and(|n| init_names.contains_key(n.as_str()));
                map_input(&mut rename, &init_names, node, 1, &base, "weight");
                map_input(&mut rename, &init_names, node, 2, &base, "bias");
                map_input(&mut rename, &init_names, node, 3, &base, "running_mean");
                map_input(&mut rename, &init_names, node, 4, &base, "running_var");
                if !scale_is_init {
                    // Affine-free BN: scale/bias were folded to Constants, so
                    // only mean/var are initializers. Channel count = |mean|.
                    let ch = node
                        .inputs
                        .get(3)
                        .and_then(|n| init_names.get(n.as_str()))
                        .and_then(|t| t.dims.first().copied())
                        .unwrap_or(0) as u32;
                    if ch > 0 {
                        synth.push((base.clone(), ch));
                    }
                }
            }
            _ => {}
        }
    }

    let block_config = derive_block_config(&graph);
    let growth = derive_growth(&graph);
    let embed_dim = derive_embed_dim(&graph);

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    // Self-describing redistribution (publishing to a public model hub): the
    // artifact must carry its own licence, not rely on a consumer running
    // Vokra's registry resolver. Values transcribed from
    // docs/license-audit.md §3, which holds the primary-source citations.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "Apache-2.0",
        Some("campplus"),
        Some("iic/speech_campplus via ayousanz/campplus-onnx (Apache-2.0)"),
    );
    b.add_string(chunks::KEY_MODEL_NAME, "campplus");
    add_u32_array(&mut b, KEY_BLOCK_CONFIG, &block_config);
    b.add_u32(KEY_GROWTH, growth);
    add_u32_array(&mut b, KEY_DILATIONS, &DILATIONS);
    b.add_u32(KEY_CAM_SEG_LEN, CAM_SEG_LEN);
    b.add_f32(KEY_BN_EPS, BN_EPS);
    b.add_u32(KEY_FEAT_DIM, FEAT_DIM);
    b.add_u32(KEY_EMBED_DIM, embed_dim);

    let mut report = CamPlusReport {
        block_config,
        ..CamPlusReport::default()
    };

    // Emit each initializer under its canonical name (F32; widen F16).
    for t in &graph.initializers {
        let dtype = match t.data_type {
            ONNX_DTYPE_FLOAT | ONNX_DTYPE_FLOAT16 => GgmlType::F32,
            _ => {
                report.skipped_non_float += 1;
                continue;
            }
        };
        let name = match rename.get(&t.name) {
            Some(clean) => {
                if t.name.starts_with("onnx::") {
                    report.renamed += 1;
                }
                clean.clone()
            }
            None => {
                report.unmapped += 1;
                t.name.clone()
            }
        };
        let data = match t.data_type {
            ONNX_DTYPE_FLOAT => t.raw_le_bytes.clone(),
            ONNX_DTYPE_FLOAT16 => widen_f16_to_f32(&t.raw_le_bytes),
            _ => unreachable!(),
        };
        b.add_tensor(&name, dtype, t.dims.clone(), data)?;
        report.written += 1;
    }

    // Synthesize the affine-free BN scale = ones / bias = zeros.
    for (base, ch) in &synth {
        let ones: Vec<u8> = (0..*ch).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let zeros: Vec<u8> = vec![0u8; *ch as usize * 4];
        b.add_tensor(
            &format!("{base}.weight"),
            GgmlType::F32,
            vec![u64::from(*ch)],
            ones,
        )?;
        b.add_tensor(
            &format!("{base}.bias"),
            GgmlType::F32,
            vec![u64::from(*ch)],
            zeros,
        )?;
        report.synthesized += 2;
    }

    Ok((b, report))
}

/// Derives the canonical module base from a node's scope path.
///
/// `/head/conv1/Conv` → `head.conv1`; the doubled `ModuleList` scope
/// `/head/layer1/layer1.0/conv1/Conv` collapses to `head.layer1.0.conv1`. Uses
/// `NodeProto.name` when present, else the (scope-carrying) first output edge.
fn canonical_base(node: &OnnxNode) -> Option<String> {
    let raw = if node.name.is_empty() {
        strip_output_suffix(node.outputs.first()?)
    } else {
        node.name.as_str()
    };
    let mut segs: Vec<&str> = raw.trim_start_matches('/').split('/').collect();
    segs.pop()?; // drop the trailing op-type segment (Conv / BatchNormalization)
    if segs.is_empty() {
        return None;
    }
    // Collapse `.../layer1/layer1.0/...` → `.../layer1.0/...`: a child scope that
    // repeats its parent as a `<parent>.<idx>` prefix subsumes the parent.
    let mut out: Vec<&str> = Vec::with_capacity(segs.len());
    for seg in segs {
        match out.last() {
            Some(last) if seg.starts_with(&format!("{last}.")) => *out.last_mut()? = seg,
            _ => out.push(seg),
        }
    }
    Some(out.join("."))
}

/// Strips a trailing `_output_<n>` from a scope-carrying output edge name.
fn strip_output_suffix(s: &str) -> &str {
    if let Some(idx) = s.rfind("_output_") {
        let tail = &s[idx + "_output_".len()..];
        if idx > 0 && !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            return &s[..idx];
        }
    }
    s
}

/// Records `input[idx]` → `<base>.<suffix>` if that input is an initializer.
fn map_input(
    rename: &mut HashMap<String, String>,
    inits: &HashMap<&str, &onnx::OnnxInitializer>,
    node: &OnnxNode,
    idx: usize,
    base: &str,
    suffix: &str,
) {
    if let Some(name) = node.inputs.get(idx) {
        if !name.is_empty() && inits.contains_key(name.as_str()) {
            rename
                .entry(name.clone())
                .or_insert_with(|| format!("{base}.{suffix}"));
        }
    }
}

/// Derives the D-TDNN dense-layer count per block from the initializer names
/// (`xvector.block<N>.tdnnd<M>.…`): the max `M` seen per block `N`, in block
/// order. Falls back to the verified medium config if none are present.
fn derive_block_config(graph: &OnnxGraph) -> Vec<u32> {
    let mut max_idx: BTreeMap<u32, u32> = BTreeMap::new();
    for t in &graph.initializers {
        let Some(rest) = t.name.strip_prefix("xvector.block") else {
            continue;
        };
        let mut it = rest.splitn(2, '.');
        let (Some(n), Some(after)) = (it.next(), it.next()) else {
            continue;
        };
        let (Some(n), Some(m)) = (
            n.parse::<u32>().ok(),
            after
                .strip_prefix("tdnnd")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<u32>().ok()),
        ) else {
            continue;
        };
        let e = max_idx.entry(n).or_insert(0);
        *e = (*e).max(m);
    }
    if max_idx.is_empty() {
        DEFAULT_BLOCK_CONFIG.to_vec()
    } else {
        max_idx.into_values().collect()
    }
}

/// Channel growth per dense layer = `cam_layer.linear_local` output channels.
fn derive_growth(graph: &OnnxGraph) -> u32 {
    graph
        .initializers
        .iter()
        .find(|t| t.name.ends_with(".cam_layer.linear_local.weight"))
        .and_then(|t| t.dims.first().copied())
        .map(|d| d as u32)
        .unwrap_or(DEFAULT_GROWTH)
}

/// Output embedding dimension = `xvector.dense.linear` output channels.
fn derive_embed_dim(graph: &OnnxGraph) -> u32 {
    graph
        .initializers
        .iter()
        .find(|t| t.name == "xvector.dense.linear.weight")
        .and_then(|t| t.dims.first().copied())
        .map(|d| d as u32)
        .unwrap_or(DEFAULT_EMBED_DIM)
}

/// Adds a homogeneous `ARRAY<U32>` metadata value.
fn add_u32_array(b: &mut GgufBuilder, key: &str, values: &[u32]) {
    let values = values.iter().map(|&v| GgufMetadataValue::U32(v)).collect();
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values,
        }),
    );
}

/// Widens a little-endian IEEE-754 half buffer to little-endian f32 bytes.
fn widen_f16_to_f32(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&half_to_f32(bits).to_le_bytes());
    }
    out
}

/// Converts an IEEE-754 half-precision bit pattern to `f32` (pure integer, so
/// no external crate is pulled into the offline tool).
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = u32::from(bits & 0x03FF);
    let out = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut k = 0u32;
                let mut m = mant;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    k += 1;
                }
                m &= 0x03FF;
                let exp32 = 113 - k;
                sign | (exp32 << 23) | (m << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (mant << 13),
        _ => {
            let exp32 = (i32::from(exp) - 15 + 127) as u32;
            sign | (exp32 << 23) | (mant << 13)
        }
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    // --- protobuf encoders (test-only) ---
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
    }
    fn len_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        varint(out, (u64::from(field) << 3) | 2);
        varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }
    fn varint_field(out: &mut Vec<u8>, field: u32, v: u64) {
        varint(out, u64::from(field) << 3);
        varint(out, v);
    }
    fn tensor(name: &str, dims: &[u64], data_type: i32, raw: &[u8]) -> Vec<u8> {
        let mut t = Vec::new();
        let mut packed = Vec::new();
        for &d in dims {
            varint(&mut packed, d);
        }
        len_field(&mut t, 1, &packed);
        varint_field(&mut t, 2, data_type as u64);
        if !name.is_empty() {
            len_field(&mut t, 8, name.as_bytes());
        }
        len_field(&mut t, 9, raw);
        t
    }
    /// A node with a scope `name`, inputs and outputs.
    fn node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            len_field(&mut n, 1, i.as_bytes());
        }
        for o in outputs {
            len_field(&mut n, 2, o.as_bytes());
        }
        len_field(&mut n, 3, name.as_bytes());
        len_field(&mut n, 4, op_type.as_bytes());
        n
    }
    fn model(nodes: &[Vec<u8>], inits: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        for n in nodes {
            len_field(&mut graph, 1, n);
        }
        for t in inits {
            len_field(&mut graph, 5, t);
        }
        let mut m = Vec::new();
        len_field(&mut m, 7, &graph);
        m
    }
    fn f32_raw(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn canonical_base_collapses_module_list_scope() {
        let mk = |name: &str| OnnxNode {
            op_type: "Conv".to_owned(),
            name: name.to_owned(),
            inputs: vec![],
            outputs: vec![],
        };
        assert_eq!(
            canonical_base(&mk("/head/conv1/Conv")).unwrap(),
            "head.conv1"
        );
        assert_eq!(
            canonical_base(&mk("/head/layer1/layer1.0/conv1/Conv")).unwrap(),
            "head.layer1.0.conv1"
        );
        assert_eq!(
            canonical_base(&mk("/head/layer1/layer1.0/shortcut/shortcut.0/Conv")).unwrap(),
            "head.layer1.0.shortcut.0"
        );
        assert_eq!(
            canonical_base(&mk(
                "/xvector/block1/tdnnd1/nonlinear1/batchnorm/BatchNormalization"
            ))
            .unwrap(),
            "xvector.block1.tdnnd1.nonlinear1.batchnorm"
        );
    }

    #[test]
    fn canonical_base_from_output_edge_when_name_empty() {
        let n = OnnxNode {
            op_type: "Conv".to_owned(),
            name: String::new(),
            inputs: vec![],
            outputs: vec!["/xvector/tdnn/linear/Conv_output_0".to_owned()],
        };
        assert_eq!(canonical_base(&n).unwrap(), "xvector.tdnn.linear");
    }

    #[test]
    fn recovers_opaque_conv_weight_and_bias_from_scope() {
        // An FCM head conv: opaque weight + opaque bias, named by node scope.
        let w = tensor("onnx::Conv_4423", &[1], ONNX_DTYPE_FLOAT, &f32_raw(&[2.0]));
        let bias = tensor("onnx::Conv_4424", &[1], ONNX_DTYPE_FLOAT, &f32_raw(&[3.0]));
        let conv = node(
            "Conv",
            "/head/conv1/Conv",
            &["input", "onnx::Conv_4423", "onnx::Conv_4424"],
            &["/head/conv1/Conv_output_0"],
        );
        let onnx = model(&[conv], &[w, bias]);

        let (builder, report) = convert(&onnx).unwrap();
        assert_eq!(report.written, 2);
        assert_eq!(report.renamed, 2);
        assert_eq!(report.unmapped, 0);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.tensor_data("head.conv1.weight").unwrap(),
            2.0f32.to_le_bytes()
        );
        assert_eq!(
            file.tensor_data("head.conv1.bias").unwrap(),
            3.0f32.to_le_bytes()
        );
    }

    #[test]
    fn clean_bn_names_derive_to_themselves() {
        // A standalone BN with clean initializer names: the derived canonical
        // name must equal the existing name (no spurious rename).
        let base = "xvector.block1.tdnnd1.nonlinear1.batchnorm";
        let inits = ["weight", "bias", "running_mean", "running_var"]
            .iter()
            .map(|s| {
                tensor(
                    &format!("{base}.{s}"),
                    &[2],
                    ONNX_DTYPE_FLOAT,
                    &f32_raw(&[1.0, 1.0]),
                )
            })
            .collect::<Vec<_>>();
        let bn = node(
            "BatchNormalization",
            "/xvector/block1/tdnnd1/nonlinear1/batchnorm/BatchNormalization",
            &[
                "x",
                &format!("{base}.weight"),
                &format!("{base}.bias"),
                &format!("{base}.running_mean"),
                &format!("{base}.running_var"),
            ],
            &["/xvector/block1/tdnnd1/nonlinear1/batchnorm/BatchNormalization_output_0"],
        );
        let onnx = model(&[bn], &inits);
        let (builder, report) = convert(&onnx).unwrap();
        assert_eq!(report.written, 4);
        assert_eq!(report.renamed, 0);
        assert_eq!(report.unmapped, 0);
        assert_eq!(report.synthesized, 0);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert!(file.tensor_info(&format!("{base}.running_var")).is_some());
    }

    #[test]
    fn synthesizes_affine_free_final_bn() {
        // The dense BN: scale/bias are Constant outputs (not initializers), only
        // mean/var are weights → converter must synthesize ones/zeros.
        let base = "xvector.dense.nonlinear.batchnorm";
        let mean = tensor(
            &format!("{base}.running_mean"),
            &[3],
            ONNX_DTYPE_FLOAT,
            &f32_raw(&[0.0, 0.0, 0.0]),
        );
        let var = tensor(
            &format!("{base}.running_var"),
            &[3],
            ONNX_DTYPE_FLOAT,
            &f32_raw(&[1.0, 1.0, 1.0]),
        );
        let bn = node(
            "BatchNormalization",
            "/xvector/dense/nonlinear/batchnorm/BatchNormalization",
            &[
                "x",
                "/xvector/dense/nonlinear/batchnorm/Constant_output_0",
                "/xvector/dense/nonlinear/batchnorm/Constant_1_output_0",
                &format!("{base}.running_mean"),
                &format!("{base}.running_var"),
            ],
            &["output"],
        );
        let onnx = model(&[bn], &[mean, var]);
        let (builder, report) = convert(&onnx).unwrap();
        assert_eq!(report.written, 2); // mean + var
        assert_eq!(report.synthesized, 2); // weight(ones) + bias(zeros)
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.tensor_data(&format!("{base}.weight")).unwrap(),
            f32_raw(&[1.0, 1.0, 1.0])
        );
        assert_eq!(
            file.tensor_data(&format!("{base}.bias")).unwrap(),
            f32_raw(&[0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn writes_arch_and_derived_metadata() {
        // A tiny graph carrying one dense layer in each of two blocks + the
        // growth/embed shape anchors, so derivation is exercised end to end.
        let ll = |b: u32, m: u32| {
            tensor(
                &format!("xvector.block{b}.tdnnd{m}.cam_layer.linear_local.weight"),
                &[32, 128, 3],
                ONNX_DTYPE_FLOAT,
                &f32_raw(&vec![0.0; 32 * 128 * 3]),
            )
        };
        let dense = tensor(
            "xvector.dense.linear.weight",
            &[192, 1024, 1],
            ONNX_DTYPE_FLOAT,
            &f32_raw(&vec![0.0; 192 * 1024]),
        );
        // block1 has 2 layers, block2 has 1 → block_config [2, 1].
        let onnx = model(&[], &[ll(1, 1), ll(1, 2), ll(2, 1), dense]);
        let (builder, report) = convert(&onnx).unwrap();
        assert_eq!(report.block_config, vec![2, 1]);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("campplus")
        );
        assert_eq!(file.get(KEY_GROWTH), Some(&GgufMetadataValue::U32(32)));
        assert_eq!(file.get(KEY_EMBED_DIM), Some(&GgufMetadataValue::U32(192)));
        assert_eq!(
            file.get(KEY_CAM_SEG_LEN),
            Some(&GgufMetadataValue::U32(100))
        );
        assert_eq!(file.get(KEY_BN_EPS), Some(&GgufMetadataValue::F32(1e-5)));
        assert_eq!(file.get(KEY_FEAT_DIM), Some(&GgufMetadataValue::U32(80)));
        let dils = file.get(KEY_DILATIONS).and_then(|v| v.as_array()).unwrap();
        assert_eq!(dils.values.len(), 3);
        assert_eq!(dils.values[0], GgufMetadataValue::U32(1));
        assert_eq!(dils.values[1], GgufMetadataValue::U32(2));
        let bc = file
            .get(KEY_BLOCK_CONFIG)
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(bc.values.len(), 2);
    }

    #[test]
    fn skips_non_float_initializer() {
        let int_const = tensor("some.shape", &[1], 7, &[3, 0, 0, 0, 0, 0, 0, 0]);
        let onnx = model(&[], &[int_const]);
        let (_b, report) = convert(&onnx).unwrap();
        assert_eq!(report.skipped_non_float, 1);
        assert_eq!(report.written, 0);
    }
}
