//! GGUF weight binding for the native CAM++ (3D-Speaker) speaker encoder
//! (M0-08). Loads only weight **values** under the canonical, module-scoped
//! names emitted by `vokra-convert` (`head.conv1.weight`,
//! `xvector.block1.tdnnd1.cam_layer.linear2.bias`, …) and folds every
//! BatchNorm to a per-channel scale/shift at load time, so the forward pass in
//! [`super::camplus`] never touches BN running statistics or ONNX at runtime.
//!
//! # Topology is hard-coded, values are loaded (whisper.cpp pattern)
//!
//! The runtime does not read the ONNX graph; it hard-codes the verified CAM++
//! topology (FCM 2-D residual front-end → `xvector.tdnn` → three D-TDNN dense
//! blocks with the CAM attention module → transitions → statistics pooling →
//! `dense` → affine-free BN) and binds each tensor by name. Channel counts that
//! grow per dense layer (`base + i·growth`) are derived from the metadata
//! `block_config` / `growth`; the transition output widths (512→256, 1024→512,
//! 1024→512) are read back from each transition weight's shape rather than
//! assumed.
//!
//! # BatchNorm fold
//!
//! Inference BN is `y = (x − mean)/sqrt(var + eps)·γ + β`, folded here to
//! `scale = γ/sqrt(var + eps)`, `shift = β − mean·scale` so the forward applies
//! one per-channel affine. The affine-free tail BN (`xvector.dense.nonlinear`)
//! carries `γ = 1`, `β = 0` (synthesized by the converter), so its fold reduces
//! to the standardization `scale = 1/sqrt(var + eps)`, `shift = −mean·scale`.

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

/// Bottleneck width of every D-TDNN dense layer (`bn_size·growth = 4·32`); the
/// `linear1` conv maps its (growing) input to this fixed width.
pub(super) const BN_CHANNELS: usize = 128;
/// `xvector.tdnn` output channels (D-TDNN block-1 input width).
pub(super) const TDNN_OUT: usize = 128;
/// CAM context bottleneck width (`linear1` output / `linear2` input).
pub(super) const CAM_CTX: usize = 64;

/// A 1-D convolution weight `[c_out, c_in, k]` (row-major) with an optional
/// per-output-channel bias.
pub(super) struct Conv1dW {
    /// Flattened `[c_out, c_in, k]` weight (row-major).
    pub(super) weight: Vec<f32>,
    /// Optional `[c_out]` bias (present iff the ONNX conv carried one — e.g.
    /// `linear1`, the CAM `linear1`/`linear2`, `transit3`; absent for
    /// `linear_local`, `transit1`/`transit2` and `dense`).
    pub(super) bias: Option<Vec<f32>>,
    /// Output channel count.
    pub(super) c_out: usize,
    /// Input channel count.
    pub(super) c_in: usize,
    /// Kernel width.
    pub(super) k: usize,
}

/// A 2-D convolution weight `[c_out, c_in, kh, kw]` (row-major) with a
/// per-output-channel bias (every FCM conv carries one — the FCM BatchNorms are
/// folded into the convs at export).
pub(super) struct Conv2dW {
    /// Flattened `[c_out, c_in, kh, kw]` weight (row-major).
    pub(super) weight: Vec<f32>,
    /// `[c_out]` bias.
    pub(super) bias: Vec<f32>,
    /// Output channel count.
    pub(super) c_out: usize,
    /// Input channel count.
    pub(super) c_in: usize,
    /// Kernel height (frequency axis).
    pub(super) kh: usize,
    /// Kernel width (time axis).
    pub(super) kw: usize,
}

/// A BatchNorm folded to a per-channel affine `y = x·scale + shift`.
pub(super) struct Bn {
    /// Per-channel multiplicative term `γ/sqrt(var + eps)`.
    pub(super) scale: Vec<f32>,
    /// Per-channel additive term `β − mean·scale`.
    pub(super) shift: Vec<f32>,
}

/// One FCM `BasicResBlock`: `conv1` (optionally frequency-downsampling) → ReLU →
/// `conv2`, plus either a 1×1 projection shortcut (downsampling blocks) or an
/// identity shortcut, summed and ReLU'd.
pub(super) struct ResBlockW {
    /// First 3×3 conv (stride `(2,1)` when `shortcut` is present, else `(1,1)`).
    pub(super) conv1: Conv2dW,
    /// Second 3×3 conv (stride `(1,1)`).
    pub(super) conv2: Conv2dW,
    /// 1×1 projection shortcut (`Some` for the frequency-downsampling blocks).
    pub(super) shortcut: Option<Conv2dW>,
}

/// The FCM 2-D residual front-end: `conv1` → `layer1` (2 res-blocks) → `layer2`
/// (2 res-blocks) → `conv2`. Frequency is halved at `layer1.0`, `layer2.0` and
/// `conv2` (80→40→20→10); the 32 channels × 10 frequencies are then reshaped to
/// 320 channels for the D-TDNN stack.
pub(super) struct FcmW {
    /// Stem 3×3 conv `1→32` (stride 1).
    pub(super) conv1: Conv2dW,
    /// `layer1`: `[downsampling, identity]` res-blocks (freq 80→40).
    pub(super) layer1: [ResBlockW; 2],
    /// `layer2`: `[downsampling, identity]` res-blocks (freq 40→20).
    pub(super) layer2: [ResBlockW; 2],
    /// Tail 3×3 conv `32→32`, stride `(2,1)` (freq 20→10).
    pub(super) conv2: Conv2dW,
}

/// The CAM (Context-Aware Masking) attention module of one D-TDNN layer.
pub(super) struct CamW {
    /// Dilated local conv `128→32`, k=3 (no bias); the value branch `y`.
    pub(super) linear_local: Conv1dW,
    /// Context bottleneck `128→64`, k=1 (with bias); ReLU follows.
    pub(super) linear1: Conv1dW,
    /// Context gate `64→32`, k=1 (with bias); Sigmoid follows, then `y·m`.
    pub(super) linear2: Conv1dW,
}

/// One `CAMDenseTDNNLayer`: BN → ReLU → `linear1` (→128) → ReLU → CAM module,
/// whose 32-channel output is dense-concatenated onto the block state.
pub(super) struct DtdnnLayerW {
    /// `nonlinear1` BatchNorm on the (growing) layer input.
    pub(super) bn1: Bn,
    /// `linear1` conv `c_in→128`, k=1 (with the folded `nonlinear2` bias).
    pub(super) linear1: Conv1dW,
    /// The CAM attention module.
    pub(super) cam: CamW,
}

/// One dense block: its per-layer weights plus the CAM dilation used by all its
/// layers (block1=1, block2=2, block3=2).
pub(super) struct BlockW {
    /// Dense layers, in order (`tdnnd1..tdnndN`).
    pub(super) layers: Vec<DtdnnLayerW>,
    /// CAM `linear_local` dilation (= padding) for this block.
    pub(super) dilation: usize,
}

/// One transition: BN → ReLU → 1×1 conv (channel reduction between blocks).
pub(super) struct TransitionW {
    /// `nonlinear` BatchNorm on the block output.
    pub(super) bn: Bn,
    /// `linear` 1×1 conv (bias only on `transit3`, into which the tail
    /// `out_nonlinear` BN is folded).
    pub(super) linear: Conv1dW,
}

/// Static, verified CAM++ hyper-parameters (mirrors the converter's metadata;
/// used as a fallback when a key is absent).
pub(super) struct CamPlusConfig {
    /// Dense layers per block (`[12, 24, 16]`).
    pub(super) block_config: Vec<usize>,
    /// Channel growth per dense layer (`32`).
    pub(super) growth: usize,
    /// CAM `linear_local` dilation per block (`[1, 2, 2]`).
    pub(super) dilations: Vec<usize>,
    /// CAM `seg_pool` `AvgPool1d` kernel = stride (`100`).
    pub(super) cam_seg_len: usize,
    /// Output speaker-embedding dimension (`192`).
    pub(super) embed_dim: usize,
    /// Input fbank feature dimension (`80`).
    pub(super) feat_dim: usize,
}

/// Every bound weight of the CAM++ network plus its config.
pub(super) struct CamPlusWeights {
    /// Resolved hyper-parameters.
    pub(super) cfg: CamPlusConfig,
    /// FCM 2-D residual front-end.
    pub(super) fcm: FcmW,
    /// `xvector.tdnn` conv `320→128`, k=5, stride 2 (with bias).
    pub(super) tdnn: Conv1dW,
    /// The three D-TDNN dense blocks.
    pub(super) blocks: Vec<BlockW>,
    /// The three transitions (one after each block).
    pub(super) transitions: Vec<TransitionW>,
    /// `dense` conv `1024→192`, k=1 (no bias).
    pub(super) dense: Conv1dW,
    /// Affine-free tail BatchNorm on the 192-d embedding.
    pub(super) final_bn: Bn,
}

/// Bias-presence requirement for a loaded conv.
#[derive(Clone, Copy)]
enum Bias {
    /// The bias tensor must be present.
    Yes,
    /// No bias tensor is expected.
    No,
    /// Load the bias tensor iff it is present (transitions differ).
    IfPresent,
}

impl CamPlusWeights {
    /// Binds and folds the CAM++ weights from a parsed GGUF (FR-LD-01).
    ///
    /// Missing tensors, wrong shapes or non-`F32` dtypes are reported as
    /// [`VokraError::ModelLoad`]. The BatchNorm epsilon is read from
    /// `vokra.campplus.bn_eps` (default `1e-5`).
    pub(super) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let cfg = CamPlusConfig::from_gguf(gguf);
        let eps = bn_eps(gguf);

        let fcm = load_fcm(gguf)?;
        let tdnn = conv1d(gguf, "xvector.tdnn.linear", TDNN_OUT, 320, 5, Bias::Yes)?;

        let mut blocks = Vec::with_capacity(cfg.block_config.len());
        let mut transitions = Vec::with_capacity(cfg.block_config.len());
        let mut base = TDNN_OUT;
        for (bi, &n_layers) in cfg.block_config.iter().enumerate() {
            let dilation = cfg.dilations.get(bi).copied().unwrap_or(1);
            let mut layers = Vec::with_capacity(n_layers);
            for li in 0..n_layers {
                let c_in = base + li * cfg.growth;
                let p = format!("xvector.block{}.tdnnd{}", bi + 1, li + 1);
                layers.push(load_dtdnn_layer(gguf, &p, c_in, eps)?);
            }
            blocks.push(BlockW { layers, dilation });

            let block_out = base + n_layers * cfg.growth;
            let tp = format!("xvector.transit{}", bi + 1);
            let bn = fold_bn(gguf, &format!("{tp}.nonlinear.batchnorm"), block_out, eps)?;
            let linear =
                conv1d_infer_cout(gguf, &format!("{tp}.linear"), block_out, 1, Bias::IfPresent)?;
            base = linear.c_out;
            transitions.push(TransitionW { bn, linear });
        }

        let stats_dim = 2 * base; // [mean; std] over the last transition width.
        let dense = conv1d(
            gguf,
            "xvector.dense.linear",
            cfg.embed_dim,
            stats_dim,
            1,
            Bias::No,
        )?;
        let final_bn = fold_bn(
            gguf,
            "xvector.dense.nonlinear.batchnorm",
            cfg.embed_dim,
            eps,
        )?;

        Ok(Self {
            cfg,
            fcm,
            tdnn,
            blocks,
            transitions,
            dense,
            final_bn,
        })
    }
}

impl CamPlusConfig {
    /// Reads the `vokra.campplus.*` metadata, falling back to the verified
    /// constants when a key is absent (so a minimal test GGUF still loads).
    fn from_gguf(gguf: &GgufFile) -> Self {
        let block_config =
            u32_array(gguf, "vokra.campplus.block_config").unwrap_or_else(|| vec![12, 24, 16]);
        let dilations =
            u32_array(gguf, "vokra.campplus.dilations").unwrap_or_else(|| vec![1, 2, 2]);
        let growth = u32_scalar(gguf, "vokra.campplus.growth").unwrap_or(32) as usize;
        let cam_seg_len = u32_scalar(gguf, "vokra.campplus.cam_seg_len").unwrap_or(100) as usize;
        let embed_dim = u32_scalar(gguf, "vokra.campplus.embed_dim").unwrap_or(192) as usize;
        let feat_dim = u32_scalar(gguf, "vokra.campplus.feat_dim").unwrap_or(80) as usize;
        Self {
            block_config: block_config.into_iter().map(|v| v as usize).collect(),
            growth,
            dilations: dilations.into_iter().map(|v| v as usize).collect(),
            cam_seg_len,
            embed_dim,
            feat_dim,
        }
    }
}

/// Loads the FCM front-end (stem conv, two res-block layers, tail conv).
fn load_fcm(gguf: &GgufFile) -> Result<FcmW> {
    let conv1 = conv2d(gguf, "head.conv1", 32, 1, 3, 3)?;
    let layer1 = [
        load_resblock(gguf, "head.layer1.0", 32, true)?,
        load_resblock(gguf, "head.layer1.1", 32, false)?,
    ];
    let layer2 = [
        load_resblock(gguf, "head.layer2.0", 32, true)?,
        load_resblock(gguf, "head.layer2.1", 32, false)?,
    ];
    let conv2 = conv2d(gguf, "head.conv2", 32, 32, 3, 3)?;
    Ok(FcmW {
        conv1,
        layer1,
        layer2,
        conv2,
    })
}

/// Loads one FCM res-block. `channels` is both in and out (32 throughout);
/// `downsample` selects whether the 1×1 projection shortcut is present.
fn load_resblock(
    gguf: &GgufFile,
    base: &str,
    channels: usize,
    downsample: bool,
) -> Result<ResBlockW> {
    let conv1 = conv2d(gguf, &format!("{base}.conv1"), channels, channels, 3, 3)?;
    let conv2 = conv2d(gguf, &format!("{base}.conv2"), channels, channels, 3, 3)?;
    let shortcut = if downsample {
        Some(conv2d(
            gguf,
            &format!("{base}.shortcut.0"),
            channels,
            channels,
            1,
            1,
        )?)
    } else {
        None
    };
    Ok(ResBlockW {
        conv1,
        conv2,
        shortcut,
    })
}

/// Loads one D-TDNN dense layer (BN → linear1 → CAM module).
fn load_dtdnn_layer(gguf: &GgufFile, base: &str, c_in: usize, eps: f32) -> Result<DtdnnLayerW> {
    let bn1 = fold_bn(gguf, &format!("{base}.nonlinear1.batchnorm"), c_in, eps)?;
    let linear1 = conv1d(
        gguf,
        &format!("{base}.linear1"),
        BN_CHANNELS,
        c_in,
        1,
        Bias::Yes,
    )?;
    let cam = CamW {
        linear_local: conv1d(
            gguf,
            &format!("{base}.cam_layer.linear_local"),
            32,
            BN_CHANNELS,
            3,
            Bias::No,
        )?,
        linear1: conv1d(
            gguf,
            &format!("{base}.cam_layer.linear1"),
            CAM_CTX,
            BN_CHANNELS,
            1,
            Bias::Yes,
        )?,
        linear2: conv1d(
            gguf,
            &format!("{base}.cam_layer.linear2"),
            32,
            CAM_CTX,
            1,
            Bias::Yes,
        )?,
    };
    Ok(DtdnnLayerW { bn1, linear1, cam })
}

/// Loads a `Conv1d` `[c_out, c_in, k]` with the given bias requirement.
fn conv1d(
    gguf: &GgufFile,
    base: &str,
    c_out: usize,
    c_in: usize,
    k: usize,
    bias: Bias,
) -> Result<Conv1dW> {
    let weight = tensor_vec(gguf, &format!("{base}.weight"), &[c_out, c_in, k])?;
    let bias = load_bias(gguf, base, c_out, bias)?;
    Ok(Conv1dW {
        weight,
        bias,
        c_out,
        c_in,
        k,
    })
}

/// Loads a `Conv1d` whose output width is read back from the weight shape
/// (transitions: 512→256, 1024→512, 1024→512).
fn conv1d_infer_cout(
    gguf: &GgufFile,
    base: &str,
    c_in: usize,
    k: usize,
    bias: Bias,
) -> Result<Conv1dW> {
    let name = format!("{base}.weight");
    let info = gguf
        .tensor_info(&name)
        .ok_or_else(|| VokraError::ModelLoad(format!("CAM++: missing tensor `{name}`")))?;
    let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if dims.len() != 3 || dims[1] != c_in || dims[2] != k {
        return Err(VokraError::ModelLoad(format!(
            "CAM++: tensor `{name}` shape {dims:?}, expected [c_out, {c_in}, {k}]"
        )));
    }
    let c_out = dims[0];
    let weight = tensor_vec(gguf, &name, &[c_out, c_in, k])?;
    let bias = load_bias(gguf, base, c_out, bias)?;
    Ok(Conv1dW {
        weight,
        bias,
        c_out,
        c_in,
        k,
    })
}

/// Resolves the optional bias tensor `<base>.bias` per the [`Bias`] mode.
fn load_bias(gguf: &GgufFile, base: &str, c_out: usize, bias: Bias) -> Result<Option<Vec<f32>>> {
    let name = format!("{base}.bias");
    match bias {
        Bias::Yes => Ok(Some(tensor_vec(gguf, &name, &[c_out])?)),
        Bias::No => Ok(None),
        Bias::IfPresent => {
            if gguf.tensor_info(&name).is_some() {
                Ok(Some(tensor_vec(gguf, &name, &[c_out])?))
            } else {
                Ok(None)
            }
        }
    }
}

/// Loads a `Conv2d` `[c_out, c_in, kh, kw]` with its mandatory bias.
fn conv2d(
    gguf: &GgufFile,
    base: &str,
    c_out: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
) -> Result<Conv2dW> {
    let weight = tensor_vec(gguf, &format!("{base}.weight"), &[c_out, c_in, kh, kw])?;
    let bias = tensor_vec(gguf, &format!("{base}.bias"), &[c_out])?;
    Ok(Conv2dW {
        weight,
        bias,
        c_out,
        c_in,
        kh,
        kw,
    })
}

/// Loads a BatchNorm's four `[c]` buffers and folds them to per-channel
/// `scale = γ/sqrt(var + eps)`, `shift = β − mean·scale`.
fn fold_bn(gguf: &GgufFile, base: &str, c: usize, eps: f32) -> Result<Bn> {
    let gamma = tensor_vec(gguf, &format!("{base}.weight"), &[c])?;
    let beta = tensor_vec(gguf, &format!("{base}.bias"), &[c])?;
    let mean = tensor_vec(gguf, &format!("{base}.running_mean"), &[c])?;
    let var = tensor_vec(gguf, &format!("{base}.running_var"), &[c])?;
    let mut scale = Vec::with_capacity(c);
    let mut shift = Vec::with_capacity(c);
    for i in 0..c {
        let s = gamma[i] / (var[i] + eps).sqrt();
        scale.push(s);
        shift.push(beta[i] - mean[i] * s);
    }
    Ok(Bn { scale, shift })
}

/// Reads a tensor as `Vec<f32>`, checking presence, `F32` dtype and exact shape.
fn tensor_vec(gguf: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("CAM++: missing tensor `{name}`")))?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "CAM++: tensor `{name}` dtype {:?}, expected F32",
            info.dtype
        )));
    }
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != expected {
        return Err(VokraError::ModelLoad(format!(
            "CAM++: tensor `{name}` shape {got:?}, expected {expected:?}"
        )));
    }
    Ok(gguf.tensor_f32(name)?)
}

/// Reads a scalar `UINT32`-family metadata value.
fn u32_scalar(gguf: &GgufFile, key: &str) -> Option<u32> {
    match gguf.get(key)? {
        GgufMetadataValue::U32(v) => Some(*v),
        other => other.as_u64().map(|v| v as u32),
    }
}

/// Reads a `FLOAT32` metadata value; defaults to `1e-5` when absent.
fn bn_eps(gguf: &GgufFile) -> f32 {
    match gguf.get("vokra.campplus.bn_eps") {
        Some(GgufMetadataValue::F32(v)) => *v,
        _ => 1e-5,
    }
}

/// Reads a homogeneous `ARRAY<U32>` metadata value.
fn u32_array(gguf: &GgufFile, key: &str) -> Option<Vec<u32>> {
    let arr = gguf.get(key)?.as_array()?;
    Some(
        arr.values
            .iter()
            .filter_map(|v| match v {
                GgufMetadataValue::U32(x) => Some(*x),
                other => other.as_u64().map(|x| x as u32),
            })
            .collect(),
    )
}
