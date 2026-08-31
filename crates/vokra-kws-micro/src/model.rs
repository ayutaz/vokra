//! microWakeWord GGUF model loader (M5-03b Phase 2).
//!
//! Reads a Vokra microWakeWord GGUF (emitted by the offline sidecar
//! [`tools/parity/microwakeword/prepare_checkpoint.py`]) and yields a typed
//! [`Model`] carrying the audio-frontend contract (from `vokra.kws.*`
//! metadata) plus every dense F32, I32, or Q8_0 weight tensor.
//!
//! # What a `Model` can and cannot reach today
//!
//! This module is the runtime *loader*: it parses and validates, and does not
//! itself run inference. The forward it would feed is real — the INT8 kernels
//! ([`crate::kernels`]), the chain executor ([`crate::interpreter`]) and
//! [`crate::KwsMicro::detect`] all execute for real on an attached
//! [`crate::interpreter::ChainConfig`].
//!
//! The generic join is [`Model::bind_untrusted_topology`]: it consumes an
//! explicitly untrusted/synthetic [`TopologyManifest`] and checks every graph
//! edge, shape, dtype, quantization vector, and GGUF constant before
//! constructing a [`crate::interpreter::ChainConfig`]. Production authority
//! is closed behind [`Model::bind_authenticated_chain`], which remains
//! blocked until the reviewed VAST topology and parity evidence are compiled
//! into this crate.
//!
//! # Design rationale (why a two-layer parser, not a monolithic one)
//!
//! The wire format is a stock **GGUF v3** file: the Python sidecar writes the
//! format directly so the runtime has no dependency on a Python writer. This
//! crate therefore reuses
//! [`vokra_core::gguf::GgufFile`] (which is `no_std`-clean under
//! `default-features = false`) for the outer layer — magic / version / OOB
//! bounds / UTF-8 metadata strings / tensor payload alignment are all
//! validated there once — and adds a `vokra.kws.*`-specific typing layer on
//! top ([`ModelHeader`] + [`Tensor`]). This mirrors the sister
//! [`vokra_vad_micro::SileroWeights::from_gguf`] pattern and keeps
//! the FlatBuffer parser (which the runtime does NOT need — the sidecar is
//! the only reader of `.tflite`) out of `vokra-kws-micro` entirely.
//!
//! # `vokra.kws.*` metadata contract
//!
//! Mirrors the sidecar's `KEY_*` constants byte-for-byte:
//!
//! | key                        | type   | meaning                                    |
//! |----------------------------|--------|--------------------------------------------|
//! | `vokra.kws.arch`           | string | Must equal [`EXPECTED_ARCH`] (fail-closed) |
//! | `vokra.kws.model`          | string | Human-readable model name                  |
//! | `vokra.kws.threshold`      | f32    | Wake-decision cutoff in `[0.0, 1.0]`       |
//! | `vokra.kws.sample_rate`    | u32    | Audio sample rate in Hz                    |
//! | `vokra.kws.hop_ms`         | u32    | Feature-extraction hop in ms               |
//! | `vokra.kws.window_ms`      | u32    | Feature-extraction window in ms            |
//! | `vokra.kws.n_mels`         | u32    | Number of mel bands                        |
//! | `vokra.kws.feature_dim`    | u32    | Per-frame feature vector length            |
//! | `vokra.kws.tflite_sha256`  | string | Upstream `.tflite` hex sha256              |
//! | `vokra.kws.upstream`       | string | Upstream release URL                       |
//!
//! Every required key is enforced (FR-EX-08): a missing key, a wrong value
//! type, or a mismatched `arch` string is an explicit
//! [`VokraError::ModelLoad`], never a silent bind. The `tests` module below
//! sweeps all ten keys of this table for both the missing case and the
//! wrong-type case, so the claim is pinned rather than merely asserted.
//!
//! # Tensors (shape-generic)
//!
//! Every tensor is bound generically as a [`Tensor`] (name + shape + typed
//! F32, I32, or Q8_0 payload). Per-layer typed bindings (Conv2d / DwConv2d / Dense weight
//! blocks, mirroring the `Conv1dW` pattern in
//! `crates/vokra-vad-micro/src/weights.rs` — that module is private and
//! `Conv1dW` is `pub(crate)`, so neither has a docs.rs page to link) are
//! assembled by [`Model::bind_untrusted_topology`] only from an explicitly
//! supplied typed topology. Graduation remains blocked on a real manifest and
//! end-to-end parity evidence; the production authenticated entry point cannot
//! be unlocked by caller data.
//!
//! Q8_0 and I32 tensors carry the source TFLite quantization vector in indexed metadata
//! keys `vokra.kws.tensor.<ordinal>.name` (string), `.quant.scales` (float
//! array), `.quant.zero_points` (signed-integer array), and
//! `.quant.quantized_dimension` (I32). The indexed form avoids inventing a
//! metadata-key escaping scheme for arbitrary source tensor names; the stamped
//! name prevents declaration reordering from binding the wrong parameters.
//! A missing, wrongly typed, or shape-inconsistent vector is rejected rather
//! than silently treating the GGUF block scale as the source model's affine
//! scale. The eventual chain binder must still reject per-axis vectors until
//! the checkpoint's operator axis contract is independently inspected.
//!
//! # Ops are represented out-of-band
//!
//! The sidecar emits weights and a separate authenticated topology manifest.
//! [`TopologyManifest`] is deliberately not inferred from GGUF names: the
//! binder rejects branches, custom operators, unsupported fusions, and any
//! edge/shape/quantization mismatch before [`crate::KwsMicro::detect`] can be
//! attached to the scalar INT8 kernels.
//!
//! [`vokra_vad_micro`]: https://docs.rs/vokra-vad-micro
//! [`vokra_vad_micro::SileroWeights::from_gguf`]: https://docs.rs/vokra-vad-micro/latest/vokra_vad_micro/struct.SileroWeights.html#method.from_gguf

// `alloc` items that are in the prelude under `std` need explicit imports
// under `#![no_std]`. Mirrors the sister `crates/vokra-vad-micro/src/weights.rs` gate
// exactly: `format!` / `String` (+ `ToString`) / `Vec` are the only alloc
// items this module touches.
#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufTensorInfo, GgufValueType};
use vokra_core::{Result, VokraError};

use crate::interpreter::{ChainConfig, LayerSpec};
use crate::kernels::ConvDims;

/// The `vokra.kws.arch` discriminator the sidecar emits for microWakeWord
/// artifacts. A GGUF whose `arch` string differs is rejected outright:
/// microWakeWord (MC-MobileNet on M55) and openWakeWord (speech-embed MLP
/// on RPi/Linux) are separate ecosystems and their weight layouts do not
/// interchange. Downstream binders switch on this key.
pub const EXPECTED_ARCH: &str = "microwakeword";

/// Compiled review authority for the canonical TFLite topology. This remains
/// unset until the official VAST artifact, GGUF bind, and independent parity
/// evidence are reviewed together; a caller cannot supply or override it.
pub const REVIEWED_TOPOLOGY_SHA256: Option<&str> = None;

// --- Metadata key names (mirror the Python sidecar's `KEY_*` constants
// byte-for-byte). Deliberately duplicated instead of imported from
// `vokra-core`: no cross-crate `vokra.kws.*` string constants live there
// (the sister vad-micro crate follows the same pattern for
// `vokra.silero.*` — one string per crate, one authority per key). -------

const KEY_ARCH: &str = "vokra.kws.arch";
const KEY_MODEL: &str = "vokra.kws.model";
const KEY_THRESHOLD: &str = "vokra.kws.threshold";
const KEY_SAMPLE_RATE: &str = "vokra.kws.sample_rate";
const KEY_HOP_MS: &str = "vokra.kws.hop_ms";
const KEY_WINDOW_MS: &str = "vokra.kws.window_ms";
const KEY_N_MELS: &str = "vokra.kws.n_mels";
const KEY_FEATURE_DIM: &str = "vokra.kws.feature_dim";
const KEY_TFLITE_SHA256: &str = "vokra.kws.tflite_sha256";
const KEY_UPSTREAM: &str = "vokra.kws.upstream";

/// Audio-frontend + provenance contract carried by every microWakeWord
/// GGUF's `vokra.kws.*` metadata group.
///
/// Every field is validated at bind time: strings must be UTF-8 (enforced
/// by [`GgufFile`]); numeric fields must be present, non-zero (for the
/// dimensioning fields), and in-range (for [`threshold`](Self::threshold)).
#[derive(Debug, Clone)]
pub struct ModelHeader {
    /// Human-readable model identifier (e.g. `"hey_jarvis"`).
    pub model: String,
    /// Wake-decision cutoff in `[0.0, 1.0]`, as declared by the checkpoint.
    /// A caller wiring this model up copies it into
    /// [`crate::KeywordDef::threshold`], which is what
    /// [`crate::KwsMicro::detect`] actually compares scores against.
    pub threshold: f32,
    /// Audio sample rate in Hz (typically 16000).
    pub sample_rate: u32,
    /// Feature-extraction hop in ms (typically 10).
    pub hop_ms: u32,
    /// Feature-extraction window in ms (typically 32).
    pub window_ms: u32,
    /// Number of mel bands (typically 40).
    pub n_mels: u32,
    /// Per-frame feature vector length. Kept independent of `n_mels`
    /// because a stacked-frame model may carry `feature_dim != n_mels`.
    pub feature_dim: u32,
    /// Hex `sha256` of the upstream `.tflite` file (provenance audit).
    pub tflite_sha256: String,
    /// Upstream release URL (provenance audit).
    pub upstream: String,
}

/// Quantization parameters carried by a source TFLite tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorQuantization {
    /// One or more affine dequantization scales from the source tensor.
    pub scales: Vec<f32>,
    /// One zero point for each source scale.
    pub zero_points: Vec<i64>,
    /// TFLite `quantized_dimension` (`-1` is the scalar/per-tensor sentinel).
    pub quantized_dimension: i32,
}

/// TFLite tensor type carried by an untrusted or owner-reviewed topology manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDtype {
    /// Signed 8-bit activation or weight.
    Int8,
    /// Signed 32-bit pre-scaled bias.
    Int32,
    /// Dense float tensor (not accepted by the INT8 ChainConfig binder).
    Float32,
}

/// A tensor reference from the TFLite subgraph.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyTensor {
    /// TFLite tensor ordinal.
    pub index: u32,
    /// Exact TFLite tensor name.
    pub name: String,
    /// Source (framework-order) dimensions, not GGUF wire-order dimensions.
    pub shape: Vec<u64>,
    /// Source tensor dtype.
    pub dtype: TopologyDtype,
    /// Whether the tensor owns persistent FlatBuffer bytes.
    pub constant: bool,
    /// TFLite affine parameters; production use requires the closed review
    /// authority in [`REVIEWED_TOPOLOGY_SHA256`].
    pub quantization: Option<TensorQuantization>,
}

/// TFLite padding mode admitted by the typed binder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyPadding {
    /// TFLite SAME padding; only symmetric derived padding is representable.
    Same,
    /// TFLite VALID padding.
    Valid,
}

/// One operator in a validated, linear TFLite chain.
#[derive(Debug, Clone, PartialEq)]
pub enum TopologyOperator {
    /// TFLite CONV_2D with activation, filter, and bias inputs.
    Conv2d {
        /// Operator ordinal in execution order.
        index: usize,
        /// Activation tensor index.
        input: u32,
        /// Filter tensor index.
        weight: u32,
        /// Bias tensor index.
        bias: u32,
        /// Output tensor index.
        output: u32,
        /// TFLite padding mode.
        padding: TopologyPadding,
        /// Horizontal stride.
        stride_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal dilation.
        dilation_w: usize,
        /// Vertical dilation.
        dilation_h: usize,
    },
    /// TFLite DEPTHWISE_CONV_2D with depth multiplier one.
    DepthwiseConv2d {
        /// Operator ordinal in execution order.
        index: usize,
        /// Activation tensor index.
        input: u32,
        /// Filter tensor index.
        weight: u32,
        /// Bias tensor index.
        bias: u32,
        /// Output tensor index.
        output: u32,
        /// TFLite padding mode.
        padding: TopologyPadding,
        /// Horizontal stride.
        stride_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal dilation.
        dilation_w: usize,
        /// Vertical dilation.
        dilation_h: usize,
        /// TFLite depth multiplier (must be one).
        depth_multiplier: usize,
    },
    /// TFLite FULLY_CONNECTED.
    FullyConnected {
        /// Operator ordinal in execution order.
        index: usize,
        /// Activation tensor index.
        input: u32,
        /// Matrix tensor index.
        weight: u32,
        /// Bias tensor index.
        bias: u32,
        /// Output tensor index.
        output: u32,
    },
    /// TFLite LOGISTIC.
    Logistic {
        /// Operator ordinal in execution order.
        index: usize,
        /// Input tensor index.
        input: u32,
        /// Output tensor index.
        output: u32,
    },
    /// TFLite SOFTMAX with beta one.
    Softmax {
        /// Operator ordinal in execution order.
        index: usize,
        /// Input tensor index.
        input: u32,
        /// Output tensor index.
        output: u32,
        /// TFLite beta option.
        beta: f32,
    },
}

/// Typed topology handoff from the VAST FlatBuffer producer to native Rust.
///
/// This is deliberately not populated by guessing from GGUF tensor names. A
/// caller may provide records for synthetic/untrusted validation;
/// [`Model::bind_untrusted_topology`] checks them against GGUF constants before
/// constructing a [`ChainConfig`]. This type cannot confer production
/// authentication authority.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyManifest {
    /// All source tensors, including activation tensors with no GGUF payload.
    pub tensors: Vec<TopologyTensor>,
    /// Single graph input boundary.
    pub inputs: Vec<u32>,
    /// Single graph output boundary.
    pub outputs: Vec<u32>,
    /// Operators in exact execution order.
    pub operators: Vec<TopologyOperator>,
}

/// Exact storage state for one GGUF tensor.
///
/// I32 is intentionally a separate variant: it cannot be accidentally
/// consumed through the F32/Q8 dequantization view, and values above `2^24`
/// remain exact for bias binding.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// Native dense F32 values.
    F32(Vec<f32>),
    /// Q8_0 source-byte carrier and its decoded carrier view.
    Q8 {
        /// GGUF identity-carrier values (the exact source bytes widened to F32).
        values: Vec<f32>,
        /// Exact signed source bytes carried after each Q8 block scale.
        raw: Vec<i8>,
    },
    /// Native dense signed I32 values, decoded without an F32 round-trip.
    I32(Vec<i32>),
}

/// One weight tensor decoded from the GGUF.
///
/// The `data` field owns a decoded copy of the payload — GGUF stores F32
/// as little-endian 4-byte quads on disk; this struct converts them to
/// host `f32` up front so the consumer sees the same values regardless of
/// endianness (thumbv8m is little-endian, so this is the same code path
/// on both host and target — the cost is one alloc per weight).
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name as emitted by the sidecar (equals the upstream `.tflite`
    /// tensor name).
    pub name: String,
    /// Dimensions exactly as stored on disk: innermost (fastest-varying)
    /// axis first, which is the GGUF wire order every Vokra binder sees.
    ///
    /// This is the **reverse** of numpy `.shape` on the source tensor.
    /// The sidecar writes the source NumPy shape back-to-front, as required
    /// by GGUF's innermost-first wire convention, so a NumPy `(out, in)`
    /// weight arrives here as `[in, out]`. Read this field as wire order and
    /// reverse it if you need the upstream framework's convention.
    pub shape: Vec<u64>,
    /// Decoded storage, with I32 kept exact and separate from float values.
    pub data: TensorData,
    /// Source TFLite affine quantization vector. `None` for F32 tensors.
    pub quantization: Option<TensorQuantization>,
}

/// A parsed microWakeWord GGUF: audio-frontend contract + every dense F32, I32, or Q8_0
/// weight tensor. Constructed via [`Model::from_bytes`] (owns a `Vec<u8>`
/// copy of the input) or [`Model::from_gguf`] (borrows a prebuilt
/// [`GgufFile`]).
#[derive(Debug, Clone)]
pub struct Model {
    /// Typed view of the `vokra.kws.*` metadata group.
    pub header: ModelHeader,
    /// Every dense F32, I32, or Q8_0 tensor in the file, in GGUF declaration order.
    /// Per-layer typed bindings on top of this are not written yet (see the
    /// module docs); the shape-generic list is what shape audits (via
    /// `tests`) and any future binder walk.
    pub tensors: Vec<Tensor>,
}

impl Model {
    /// Parses a microWakeWord GGUF from an in-memory byte slice.
    ///
    /// A convenience wrapper over [`Model::from_gguf`]: allocates a
    /// `Vec<u8>` copy of `bytes` because [`GgufFile::parse`] wants an
    /// owned buffer. Suited to test fixtures and short-lived loads;
    /// callers with a persistent flash / XIP mapping should build a
    /// [`GgufFile`] via [`vokra_core::gguf::GgufFile::from_external`]
    /// first and call [`Model::from_gguf`] directly (zero extra copy).
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::ModelLoad`] on:
    /// - invalid GGUF magic, unsupported version, or OOB read (the
    ///   [`GgufFile`] layer);
    /// - missing or wrong-type `vokra.kws.*` metadata (this layer);
    /// - `vokra.kws.arch` != [`EXPECTED_ARCH`] (this layer);
    /// - `vokra.kws.threshold` outside `[0.0, 1.0]` or non-finite;
    /// - any zero-valued dimensioning key (`sample_rate` / `hop_ms` /
    ///   `window_ms` / `n_mels` / `feature_dim`);
    /// - any tensor with a dtype other than [`GgmlType::F32`] or
    ///   [`GgmlType::Q8_0`]; quantized tensors additionally require their
    ///   indexed source-TFLite scale and zero-point metadata.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let gguf = GgufFile::parse(bytes.to_vec()).map_err(VokraError::from)?;
        Self::from_gguf(&gguf)
    }

    /// Binds a microWakeWord [`Model`] from a prebuilt [`GgufFile`].
    ///
    /// This is the primary entry point when the caller already holds a
    /// [`GgufFile`] (e.g. one built via
    /// [`vokra_core::gguf::GgufFile::from_external`] over a flash mapping).
    /// It avoids the `Vec<u8>` copy [`Model::from_bytes`] makes.
    ///
    /// # Errors
    ///
    /// See [`Model::from_bytes`] for the `vokra.kws.*`-layer and tensor
    /// validation errors this returns. (This entry point does not itself
    /// hit the GGUF outer-layer parser — the caller has already done that.)
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let header = ModelHeader::from_gguf(gguf)?;
        let mut tensors = Vec::with_capacity(gguf.tensors().len());
        for (tensor_index, info) in gguf.tensors().iter().enumerate() {
            if !matches!(info.dtype, GgmlType::F32 | GgmlType::I32 | GgmlType::Q8_0) {
                return Err(VokraError::ModelLoad(format!(
                    "tensor `{}` has dtype {:?}, expected F32, I32, or Q8_0",
                    info.name, info.dtype
                )));
            }
            let bytes = gguf.tensor_bytes(info);
            if info.dtype == GgmlType::I32 {
                let source_name = get_tensor_string(gguf, tensor_index, "name")?;
                if source_name != info.name {
                    return Err(VokraError::ModelLoad(format!(
                        "tensor ordinal {tensor_index} is named `{}` but source name metadata is `{source_name}`",
                        info.name
                    )));
                }
                let quantization = get_tensor_quantization(gguf, tensor_index, info)?;
                if quantization.zero_points.iter().any(|&value| value != 0) {
                    return Err(VokraError::ModelLoad(format!(
                        "tensor `{}` I32 bias requires zero point 0",
                        info.name
                    )));
                }
                tensors.push(Tensor {
                    name: info.name.clone(),
                    shape: info.dimensions.clone(),
                    data: TensorData::I32(gguf.tensor_i32(&info.name).map_err(VokraError::from)?),
                    quantization: Some(quantization),
                });
                continue;
            }
            if info.dtype == GgmlType::Q8_0 {
                if bytes
                    .chunks_exact(34)
                    .any(|block| block[..2] != [0x00, 0x3c])
                {
                    return Err(VokraError::ModelLoad(format!(
                        "tensor `{}` Q8_0 source-byte carrier requires every block scale to be exact FP16 1.0",
                        info.name
                    )));
                }
                let source_name = get_tensor_string(gguf, tensor_index, "name")?;
                if source_name != info.name {
                    return Err(VokraError::ModelLoad(format!(
                        "tensor ordinal {tensor_index} is named `{}` but source name metadata is `{source_name}`",
                        info.name
                    )));
                }
                let quantization = get_tensor_quantization(gguf, tensor_index, info)?;
                let data = gguf.tensor_f32(&info.name).map_err(VokraError::from)?;
                let data_i8 = bytes
                    .chunks_exact(34)
                    .flat_map(|block| block[2..].iter().copied().map(|value| value as i8))
                    .collect();
                tensors.push(Tensor {
                    name: info.name.clone(),
                    shape: info.dimensions.clone(),
                    data: TensorData::Q8 {
                        values: data,
                        raw: data_i8,
                    },
                    quantization: Some(quantization),
                });
                continue;
            }
            for field in [
                "name",
                "quant.scales",
                "quant.zero_points",
                "quant.quantized_dimension",
            ] {
                let key = tensor_metadata_key(tensor_index, field);
                if gguf.get(&key).is_some() {
                    return Err(VokraError::ModelLoad(format!(
                        "F32 tensor ordinal {tensor_index} carries Q8_0 metadata `{key}`"
                    )));
                }
            }
            // Every F32 tensor payload is a whole number of 4-byte quads;
            // the GGUF layer already validated the byte length against the
            // shape, so this is a defensive belt-and-braces check that also
            // guards against a hypothetical future GGUF variant with a
            // different F32 encoding (FR-EX-08).
            if !bytes.len().is_multiple_of(4) {
                return Err(VokraError::ModelLoad(format!(
                    "tensor `{}` byte length {} is not a multiple of 4 (F32)",
                    info.name,
                    bytes.len()
                )));
            }
            let data: Vec<f32> = bytes
                // The multiple-of-four gate above makes every chunk exactly
                // four bytes. Use `chunks` instead of the newer
                // `slice::as_chunks` API to preserve the workspace MSRV.
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.push(Tensor {
                name: info.name.clone(),
                shape: info.dimensions.clone(),
                data: TensorData::F32(data),
                quantization: None,
            });
        }
        Ok(Self { header, tensors })
    }

    /// Looks up a tensor by name. Returns `None` when no tensor has that
    /// name (Phase 3 typed bindings will use this to bind
    /// per-layer conv / dense weights by their upstream `.tflite` names).
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Number of tensors in this model (equivalent to
    /// `self.tensors.len()`).
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Validates an untrusted/synthetic topology handoff and builds a typed
    /// INT8 [`ChainConfig`] for tests and integration development. This method
    /// does not confer production or authenticated authority.
    pub fn bind_untrusted_topology(&self, topology: &TopologyManifest) -> Result<ChainConfig> {
        validate_topology(topology)?;
        let mut layers = Vec::with_capacity(topology.operators.len());
        for operator in &topology.operators {
            let layer = match operator {
                TopologyOperator::Conv2d {
                    input,
                    weight,
                    bias,
                    output,
                    padding,
                    stride_w,
                    stride_h,
                    dilation_w,
                    dilation_h,
                    ..
                } => {
                    let input_ref = topology_tensor(topology, *input)?;
                    let weight_ref = topology_tensor(topology, *weight)?;
                    let bias_ref = topology_tensor(topology, *bias)?;
                    let output_ref = topology_tensor(topology, *output)?;
                    require_conv_types(input_ref, weight_ref, bias_ref, output_ref, "Conv2d")?;
                    let (input_scale, input_zero_point) = scalar_quant(input_ref, "Conv2d input")?;
                    let (weight_scale, _) = scalar_quant(weight_ref, "Conv2d weight")?;
                    let (bias_scale, _) = scalar_quant(bias_ref, "Conv2d bias")?;
                    let (output_scale, output_zero_point) =
                        scalar_quant(output_ref, "Conv2d output")?;
                    require_bias_scale(bias_scale, input_scale, weight_scale, "Conv2d")?;
                    let (in_h, in_w, in_c, out_c, kh, kw, out_h, out_w) = conv_shape(
                        input_ref,
                        weight_ref,
                        output_ref,
                        *stride_h,
                        *stride_w,
                        *dilation_h,
                        *dilation_w,
                        *padding,
                        false,
                        "Conv2d",
                    )?;
                    let dims = ConvDims {
                        in_h,
                        in_w,
                        in_c,
                        out_c,
                        kh,
                        kw,
                        stride_h: *stride_h,
                        stride_w: *stride_w,
                        pad_h: derived_padding(
                            input_ref.shape[1],
                            kh,
                            *stride_h,
                            *dilation_h,
                            output_ref.shape[1],
                            *padding,
                            "Conv2d height",
                        )?,
                        pad_w: derived_padding(
                            input_ref.shape[2],
                            kw,
                            *stride_w,
                            *dilation_w,
                            output_ref.shape[2],
                            *padding,
                            "Conv2d width",
                        )?,
                    };
                    let _ = (out_h, out_w);
                    LayerSpec::Conv2d {
                        weight_i8: bound_weight(self, weight_ref)?,
                        bias_i32: bound_bias(self, bias_ref)?,
                        dims,
                        input_zero_point,
                        output_zero_point,
                        output_scale: requant_scale(
                            input_scale,
                            weight_scale,
                            output_scale,
                            "Conv2d",
                        )?,
                    }
                }
                TopologyOperator::DepthwiseConv2d {
                    input,
                    weight,
                    bias,
                    output,
                    padding,
                    stride_w,
                    stride_h,
                    dilation_w,
                    dilation_h,
                    depth_multiplier,
                    ..
                } => {
                    if *depth_multiplier != 1 {
                        return Err(model_error(
                            "DepthwiseConv2d depth_multiplier other than one is unsupported",
                        ));
                    }
                    let input_ref = topology_tensor(topology, *input)?;
                    let weight_ref = topology_tensor(topology, *weight)?;
                    let bias_ref = topology_tensor(topology, *bias)?;
                    let output_ref = topology_tensor(topology, *output)?;
                    require_conv_types(
                        input_ref,
                        weight_ref,
                        bias_ref,
                        output_ref,
                        "DepthwiseConv2d",
                    )?;
                    let (input_scale, input_zero_point) =
                        scalar_quant(input_ref, "DepthwiseConv2d input")?;
                    let (weight_scale, _) = scalar_quant(weight_ref, "DepthwiseConv2d weight")?;
                    let (bias_scale, _) = scalar_quant(bias_ref, "DepthwiseConv2d bias")?;
                    let (output_scale, output_zero_point) =
                        scalar_quant(output_ref, "DepthwiseConv2d output")?;
                    require_bias_scale(bias_scale, input_scale, weight_scale, "DepthwiseConv2d")?;
                    let (in_h, in_w, in_c, _, kh, kw, out_h, out_w) = conv_shape(
                        input_ref,
                        weight_ref,
                        output_ref,
                        *stride_h,
                        *stride_w,
                        *dilation_h,
                        *dilation_w,
                        *padding,
                        true,
                        "DepthwiseConv2d",
                    )?;
                    let dims = ConvDims {
                        in_h,
                        in_w,
                        in_c,
                        out_c: in_c,
                        kh,
                        kw,
                        stride_h: *stride_h,
                        stride_w: *stride_w,
                        pad_h: derived_padding(
                            input_ref.shape[1],
                            kh,
                            *stride_h,
                            *dilation_h,
                            output_ref.shape[1],
                            *padding,
                            "DepthwiseConv2d height",
                        )?,
                        pad_w: derived_padding(
                            input_ref.shape[2],
                            kw,
                            *stride_w,
                            *dilation_w,
                            output_ref.shape[2],
                            *padding,
                            "DepthwiseConv2d width",
                        )?,
                    };
                    let _ = (out_h, out_w);
                    LayerSpec::DepthwiseConv2d {
                        weight_i8: bound_weight(self, weight_ref)?,
                        bias_i32: bound_bias(self, bias_ref)?,
                        dims,
                        input_zero_point,
                        output_zero_point,
                        output_scale: requant_scale(
                            input_scale,
                            weight_scale,
                            output_scale,
                            "DepthwiseConv2d",
                        )?,
                    }
                }
                TopologyOperator::FullyConnected {
                    input,
                    weight,
                    bias,
                    output,
                    ..
                } => {
                    let input_ref = topology_tensor(topology, *input)?;
                    let weight_ref = topology_tensor(topology, *weight)?;
                    let bias_ref = topology_tensor(topology, *bias)?;
                    let output_ref = topology_tensor(topology, *output)?;
                    require_conv_types(
                        input_ref,
                        weight_ref,
                        bias_ref,
                        output_ref,
                        "FullyConnected",
                    )?;
                    if weight_ref.shape.len() != 2 || bias_ref.shape.len() != 1 {
                        return Err(model_error(
                            "FullyConnected requires rank-2 weight and rank-1 bias",
                        ));
                    }
                    let in_dim = shape_size(&input_ref.shape, "FullyConnected input")?;
                    let out_dim = shape_size(&output_ref.shape, "FullyConnected output")?;
                    if weight_ref.shape != [out_dim as u64, in_dim as u64]
                        || bias_ref.shape != [out_dim as u64]
                    {
                        return Err(model_error("FullyConnected tensor shapes do not agree"));
                    }
                    let (input_scale, input_zero_point) =
                        scalar_quant(input_ref, "FullyConnected input")?;
                    let (weight_scale, _) = scalar_quant(weight_ref, "FullyConnected weight")?;
                    let (bias_scale, _) = scalar_quant(bias_ref, "FullyConnected bias")?;
                    let (output_scale, output_zero_point) =
                        scalar_quant(output_ref, "FullyConnected output")?;
                    require_bias_scale(bias_scale, input_scale, weight_scale, "FullyConnected")?;
                    LayerSpec::FullyConnected {
                        weight_i8: bound_weight(self, weight_ref)?,
                        bias_i32: bound_bias(self, bias_ref)?,
                        in_dim,
                        out_dim,
                        input_zero_point,
                        output_zero_point,
                        output_scale: requant_scale(
                            input_scale,
                            weight_scale,
                            output_scale,
                            "FullyConnected",
                        )?,
                    }
                }
                TopologyOperator::Logistic { input, output, .. } => {
                    let input_ref = topology_tensor(topology, *input)?;
                    let output_ref = topology_tensor(topology, *output)?;
                    require_activation_types(input_ref, output_ref, "LOGISTIC")?;
                    if input_ref.shape != output_ref.shape {
                        return Err(model_error("LOGISTIC input/output shapes do not agree"));
                    }
                    let (input_scale, input_zero_point) =
                        scalar_quant(input_ref, "LOGISTIC input")?;
                    let (output_scale, output_zero_point) =
                        scalar_quant(output_ref, "LOGISTIC output")?;
                    LayerSpec::Sigmoid {
                        size: shape_size(&input_ref.shape, "LOGISTIC")?,
                        input_scale,
                        input_zero_point,
                        output_scale,
                        output_zero_point,
                    }
                }
                TopologyOperator::Softmax {
                    input,
                    output,
                    beta,
                    ..
                } => {
                    if *beta != 1.0 {
                        return Err(model_error("SOFTMAX beta other than one is unsupported"));
                    }
                    let input_ref = topology_tensor(topology, *input)?;
                    let output_ref = topology_tensor(topology, *output)?;
                    require_activation_types(input_ref, output_ref, "SOFTMAX")?;
                    if input_ref.shape != output_ref.shape {
                        return Err(model_error("SOFTMAX input/output shapes do not agree"));
                    }
                    let (input_scale, input_zero_point) = scalar_quant(input_ref, "SOFTMAX input")?;
                    let (output_scale, output_zero_point) =
                        scalar_quant(output_ref, "SOFTMAX output")?;
                    if output_scale != 1.0 / 256.0 || output_zero_point != -128 {
                        return Err(model_error(
                            "SOFTMAX output quantization is not TFLite's 1/256,-128 contract",
                        ));
                    }
                    LayerSpec::Softmax {
                        size: shape_size(&input_ref.shape, "SOFTMAX")?,
                        input_scale,
                        input_zero_point,
                        output_scale,
                        output_zero_point,
                    }
                }
            };
            layers.push(layer);
        }
        ChainConfig::new(layers)
    }

    /// Production entry point for the reviewed canonical topology.
    ///
    /// No caller-supplied digest or topology can unlock this path. It remains
    /// fail-closed until the repository compiles the reviewed topology digest
    /// after VAST artifact binding and independent parity sign-off.
    pub fn bind_authenticated_chain(&self) -> Result<ChainConfig> {
        let _ = self;
        if REVIEWED_TOPOLOGY_SHA256.is_none() {
            return Err(model_error("AUTHENTICATED_TOPOLOGY_REQUIRED"));
        }
        Err(model_error("AUTHENTICATED_TOPOLOGY_REQUIRED"))
    }
}

fn model_error(message: &str) -> VokraError {
    VokraError::ModelLoad(message.into())
}

fn topology_tensor<'a>(topology: &'a TopologyManifest, index: u32) -> Result<&'a TopologyTensor> {
    topology
        .tensors
        .iter()
        .find(|tensor| tensor.index == index)
        .ok_or_else(|| model_error("topology references an unknown tensor"))
}

fn op_edges(operator: &TopologyOperator) -> (Vec<u32>, u32, usize) {
    match operator {
        TopologyOperator::Conv2d {
            input,
            weight,
            bias,
            output,
            index,
            ..
        }
        | TopologyOperator::DepthwiseConv2d {
            input,
            weight,
            bias,
            output,
            index,
            ..
        }
        | TopologyOperator::FullyConnected {
            input,
            weight,
            bias,
            output,
            index,
            ..
        } => (vec![*input, *weight, *bias], *output, *index),
        TopologyOperator::Logistic {
            input,
            output,
            index,
        }
        | TopologyOperator::Softmax {
            input,
            output,
            index,
            ..
        } => (vec![*input], *output, *index),
    }
}

fn validate_topology(topology: &TopologyManifest) -> Result<()> {
    if topology.inputs.len() != 1 || topology.outputs.len() != 1 || topology.operators.is_empty() {
        return Err(model_error(
            "topology must have one input, one output, and at least one operator",
        ));
    }
    if topology.tensors.is_empty() {
        return Err(model_error("topology has no tensor records"));
    }
    let mut tensor_indices = Vec::with_capacity(topology.tensors.len());
    for tensor in &topology.tensors {
        if tensor_indices.contains(&tensor.index) {
            return Err(model_error("topology contains duplicate tensor indices"));
        }
        tensor_indices.push(tensor.index);
    }
    for (position, operator) in topology.operators.iter().enumerate() {
        let (inputs, output, index) = op_edges(operator);
        if index != position
            || inputs.is_empty()
            || inputs
                .iter()
                .any(|tensor| topology_tensor(topology, *tensor).is_err())
            || topology_tensor(topology, output).is_err()
        {
            return Err(model_error(
                "topology operator order or tensor references are invalid",
            ));
        }
        if position == 0 && inputs[0] != topology.inputs[0] {
            return Err(model_error(
                "first operator does not consume the graph input",
            ));
        }
        if position > 0 {
            let (_, previous_output, _) = op_edges(&topology.operators[position - 1]);
            if inputs[0] != previous_output {
                return Err(model_error(
                    "topology contains a branch, skip, or reordered activation",
                ));
            }
        }
        if output == topology.inputs[0]
            || inputs.iter().any(|tensor| *tensor == topology.outputs[0])
        {
            return Err(model_error(
                "topology graph boundary is used in the wrong direction",
            ));
        }
    }
    if op_edges(topology.operators.last().unwrap()).1 != topology.outputs[0] {
        return Err(model_error(
            "last operator does not produce the graph output",
        ));
    }
    let mut produced: Vec<(u32, usize)> = Vec::new();
    let mut consumed: Vec<(u32, usize)> = Vec::new();
    for operator in &topology.operators {
        let (inputs, output, _) = op_edges(operator);
        for input in inputs {
            if let Some(item) = consumed.iter_mut().find(|item| item.0 == input) {
                item.1 += 1;
            } else {
                consumed.push((input, 1));
            }
        }
        if let Some(item) = produced.iter_mut().find(|item| item.0 == output) {
            item.1 += 1;
        } else {
            produced.push((output, 1));
        }
    }
    for tensor in &topology.tensors {
        let producer_count = produced
            .iter()
            .find(|item| item.0 == tensor.index)
            .map_or(0, |item| item.1);
        let consumer_count = consumed
            .iter()
            .find(|item| item.0 == tensor.index)
            .map_or(0, |item| item.1);
        if tensor.constant {
            if producer_count != 0 || consumer_count != 1 {
                return Err(model_error(
                    "constant tensor must have exactly one consumer and no producer",
                ));
            }
        } else if tensor.index == topology.inputs[0] {
            if producer_count != 0 || consumer_count != 1 {
                return Err(model_error(
                    "graph input tensor has an invalid producer/consumer boundary",
                ));
            }
        } else if tensor.index == topology.outputs[0] {
            if producer_count != 1 || consumer_count != 0 {
                return Err(model_error(
                    "graph output tensor has an invalid producer/consumer boundary",
                ));
            }
        } else if producer_count != 1 || consumer_count != 1 {
            return Err(model_error(
                "activation tensor must have one producer and one consumer",
            ));
        }
    }
    Ok(())
}

fn require_activation_types(
    input: &TopologyTensor,
    output: &TopologyTensor,
    label: &str,
) -> Result<()> {
    if input.constant
        || output.constant
        || input.dtype != TopologyDtype::Int8
        || output.dtype != TopologyDtype::Int8
    {
        return Err(model_error(label));
    }
    Ok(())
}

fn require_conv_types(
    input: &TopologyTensor,
    weight: &TopologyTensor,
    bias: &TopologyTensor,
    output: &TopologyTensor,
    label: &str,
) -> Result<()> {
    if input.constant
        || !weight.constant
        || !bias.constant
        || output.constant
        || input.dtype != TopologyDtype::Int8
        || weight.dtype != TopologyDtype::Int8
        || bias.dtype != TopologyDtype::Int32
        || output.dtype != TopologyDtype::Int8
    {
        return Err(model_error(label));
    }
    Ok(())
}

fn shape_size(shape: &[u64], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |size, dimension| {
        let dimension = usize::try_from(*dimension).map_err(|_| model_error(label))?;
        if dimension == 0 {
            return Err(model_error(label));
        }
        size.checked_mul(dimension)
            .ok_or_else(|| model_error(label))
    })
}

fn scalar_quant(tensor: &TopologyTensor, label: &str) -> Result<(f32, i8)> {
    let quantization = tensor
        .quantization
        .as_ref()
        .ok_or_else(|| model_error(label))?;
    let axis = usize::try_from(quantization.quantized_dimension).ok();
    let scalar_sentinel = quantization.quantized_dimension == -1;
    if quantization.scales.len() != 1
        || quantization.zero_points.len() != 1
        || !quantization.scales[0].is_finite()
        || quantization.scales[0] <= 0.0
        || (!scalar_sentinel && axis.is_none())
        || axis.is_some_and(|value| value >= tensor.shape.len())
    {
        return Err(model_error(label));
    }
    let zero_point = i8::try_from(quantization.zero_points[0]).map_err(|_| model_error(label))?;
    Ok((quantization.scales[0], zero_point))
}

fn require_bias_scale(
    bias_scale: f32,
    input_scale: f32,
    weight_scale: f32,
    label: &str,
) -> Result<()> {
    if bias_scale != input_scale * weight_scale {
        return Err(model_error(label));
    }
    Ok(())
}

fn requant_scale(
    input_scale: f32,
    weight_scale: f32,
    output_scale: f32,
    label: &str,
) -> Result<f32> {
    let value = input_scale * weight_scale / output_scale;
    if !value.is_finite() || value <= 0.0 {
        return Err(model_error(label));
    }
    Ok(value)
}

fn bound_weight(model: &Model, tensor: &TopologyTensor) -> Result<Vec<i8>> {
    if !tensor.constant || tensor.dtype != TopologyDtype::Int8 {
        return Err(model_error("topology weight is not an INT8 constant"));
    }
    let stored = model
        .tensor(&tensor.name)
        .ok_or_else(|| model_error("topology weight is absent from GGUF"))?;
    if stored.name != tensor.name
        || stored.shape != tensor.shape.iter().rev().copied().collect::<Vec<_>>()
        || stored.quantization.as_ref() != tensor.quantization.as_ref()
    {
        return Err(model_error(
            "topology weight identity or quantization does not match GGUF",
        ));
    }
    match &stored.data {
        TensorData::Q8 { raw, .. } if raw.len() == shape_size(&tensor.shape, "weight")? => {
            Ok(raw.clone())
        }
        _ => Err(model_error("topology weight must be a Q8_0 tensor")),
    }
}

fn bound_bias(model: &Model, tensor: &TopologyTensor) -> Result<Vec<i32>> {
    if !tensor.constant || tensor.dtype != TopologyDtype::Int32 {
        return Err(model_error("topology bias is not an INT32 constant"));
    }
    let stored = model
        .tensor(&tensor.name)
        .ok_or_else(|| model_error("topology bias is absent from GGUF"))?;
    if stored.name != tensor.name
        || stored.shape != tensor.shape.iter().rev().copied().collect::<Vec<_>>()
        || stored.quantization.as_ref() != tensor.quantization.as_ref()
    {
        return Err(model_error(
            "topology bias identity or quantization does not match GGUF",
        ));
    }
    match &stored.data {
        TensorData::I32(values) if values.len() == shape_size(&tensor.shape, "bias")? => {
            Ok(values.clone())
        }
        _ => Err(model_error("topology bias must be an I32 tensor")),
    }
}

fn conv_shape(
    input: &TopologyTensor,
    weight: &TopologyTensor,
    output: &TopologyTensor,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    padding: TopologyPadding,
    depthwise: bool,
    label: &str,
) -> Result<(usize, usize, usize, usize, usize, usize, usize, usize)> {
    if input.shape.len() != 4
        || weight.shape.len() != 4
        || output.shape.len() != 4
        || input.shape[0] != 1
        || output.shape[0] != 1
        || stride_h == 0
        || stride_w == 0
        || dilation_h == 0
        || dilation_w == 0
    {
        return Err(model_error(label));
    }
    let in_h = usize::try_from(input.shape[1]).map_err(|_| model_error(label))?;
    let in_w = usize::try_from(input.shape[2]).map_err(|_| model_error(label))?;
    let in_c = usize::try_from(input.shape[3]).map_err(|_| model_error(label))?;
    let out_c = if depthwise {
        in_c
    } else {
        usize::try_from(weight.shape[0]).map_err(|_| model_error(label))?
    };
    let kh = usize::try_from(weight.shape[1]).map_err(|_| model_error(label))?;
    let kw = usize::try_from(weight.shape[2]).map_err(|_| model_error(label))?;
    let output_c = usize::try_from(output.shape[3]).map_err(|_| model_error(label))?;
    if (depthwise && weight.shape[0] != 1)
        || weight.shape[3] != input.shape[3]
        || output_c != out_c
        || kh == 0
        || kw == 0
    {
        return Err(model_error(label));
    }
    let effective_h = kh
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation_h))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| model_error(label))?;
    let effective_w = kw
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation_w))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| model_error(label))?;
    let (out_h, out_w) = match padding {
        TopologyPadding::Valid => (
            (in_h
                .checked_sub(effective_h)
                .ok_or_else(|| model_error(label))?
                / stride_h)
                + 1,
            (in_w
                .checked_sub(effective_w)
                .ok_or_else(|| model_error(label))?
                / stride_w)
                + 1,
        ),
        TopologyPadding::Same => (
            (in_h + stride_h - 1) / stride_h,
            (in_w + stride_w - 1) / stride_w,
        ),
    };
    if output.shape[1] != out_h as u64 || output.shape[2] != out_w as u64 {
        return Err(model_error(label));
    }
    Ok((in_h, in_w, in_c, out_c, kh, kw, out_h, out_w))
}

fn derived_padding(
    input: u64,
    kernel: usize,
    stride: usize,
    dilation: usize,
    output: u64,
    padding: TopologyPadding,
    label: &str,
) -> Result<usize> {
    if padding == TopologyPadding::Valid {
        return Ok(0);
    }
    let input = usize::try_from(input).map_err(|_| model_error(label))?;
    let output = usize::try_from(output).map_err(|_| model_error(label))?;
    let effective = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| model_error(label))?;
    let total = output
        .checked_sub(1)
        .and_then(|value| value.checked_mul(stride))
        .and_then(|value| value.checked_add(effective))
        .and_then(|value| value.checked_sub(input))
        .unwrap_or(0);
    if total % 2 != 0 {
        return Err(model_error(label));
    }
    Ok(total / 2)
}

impl ModelHeader {
    /// Extracts and validates the `vokra.kws.*` metadata group from a
    /// prebuilt [`GgufFile`].
    fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = get_str(gguf, KEY_ARCH)?;
        if arch != EXPECTED_ARCH {
            return Err(VokraError::ModelLoad(format!(
                "GGUF `{KEY_ARCH}` is `{arch}`, expected `{EXPECTED_ARCH}` \
                 (openWakeWord and microWakeWord are separate ecosystems \
                 and their weights do not interchange)"
            )));
        }
        let model = get_str(gguf, KEY_MODEL)?.to_string();
        let threshold = get_f32(gguf, KEY_THRESHOLD)?;
        if !(0.0..=1.0).contains(&threshold) {
            return Err(VokraError::ModelLoad(format!(
                "`{KEY_THRESHOLD}` = {threshold} outside [0.0, 1.0]"
            )));
        }
        let sample_rate = require_nonzero_u32(gguf, KEY_SAMPLE_RATE)?;
        let hop_ms = require_nonzero_u32(gguf, KEY_HOP_MS)?;
        let window_ms = require_nonzero_u32(gguf, KEY_WINDOW_MS)?;
        let n_mels = require_nonzero_u32(gguf, KEY_N_MELS)?;
        let feature_dim = require_nonzero_u32(gguf, KEY_FEATURE_DIM)?;
        let tflite_sha256 = get_str(gguf, KEY_TFLITE_SHA256)?.to_string();
        let upstream = get_str(gguf, KEY_UPSTREAM)?.to_string();
        Ok(Self {
            model,
            threshold,
            sample_rate,
            hop_ms,
            window_ms,
            n_mels,
            feature_dim,
            tflite_sha256,
            upstream,
        })
    }
}

/// Reads a required string metadata key. Missing key or non-string value
/// is a fail-closed [`VokraError::ModelLoad`].
fn get_str<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str> {
    let v = gguf
        .get(key)
        .ok_or_else(|| VokraError::ModelLoad(format!("missing required metadata key `{key}`")))?;
    v.as_str()
        .ok_or_else(|| VokraError::ModelLoad(format!("metadata key `{key}` is not a string")))
}

/// Reads a required unsigned-integer metadata key (accepts U8 / U16 / U32
/// / U64 via [`vokra_core::gguf::GgufMetadataValue::as_u64`]) and narrows
/// to `u32`. Values that do not fit are a fail-closed
/// [`VokraError::ModelLoad`] — a real microWakeWord model never emits a
/// dimensioning field larger than `u32::MAX`, so an oversized value is a
/// corrupt or misidentified artifact.
fn get_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    let v = gguf
        .get(key)
        .ok_or_else(|| VokraError::ModelLoad(format!("missing required metadata key `{key}`")))?;
    let u = v.as_u64().ok_or_else(|| {
        VokraError::ModelLoad(format!("metadata key `{key}` is not an unsigned integer"))
    })?;
    u32::try_from(u).map_err(|_| {
        VokraError::ModelLoad(format!(
            "metadata key `{key}` value {u} does not fit in u32"
        ))
    })
}

/// Reads a required `u32` metadata key and additionally rejects `0`.
/// Every dimensioning key (`sample_rate` / `hop_ms` / `window_ms` /
/// `n_mels` / `feature_dim`) has zero as a meaningless value that would
/// crash the downstream forward with a divide-by-zero — catch it here
/// (FR-EX-08).
fn require_nonzero_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    let v = get_u32(gguf, key)?;
    if v == 0 {
        return Err(VokraError::ModelLoad(format!("`{key}` is 0")));
    }
    Ok(v)
}

/// Reads a required floating-point metadata key (accepts F32 / F64 via
/// [`vokra_core::gguf::GgufMetadataValue::as_f64`]) and narrows to `f32`.
/// Non-finite values (NaN / ±inf after narrowing) are a fail-closed
/// [`VokraError::ModelLoad`] — the sidecar emits `threshold` as a
/// well-formed f32, so a non-finite value is a corrupt artifact
/// (FR-EX-08).
fn get_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    let v = gguf
        .get(key)
        .ok_or_else(|| VokraError::ModelLoad(format!("missing required metadata key `{key}`")))?;
    let f = v
        .as_f64()
        .ok_or_else(|| VokraError::ModelLoad(format!("metadata key `{key}` is not a float")))?;
    // Out-of-range f64 becomes ±inf when narrowed; NaN silently
    // propagates. Either is a corrupt artifact.
    let n = f as f32;
    if !n.is_finite() {
        return Err(VokraError::ModelLoad(format!(
            "metadata key `{key}` value {f} is non-finite as f32"
        )));
    }
    Ok(n)
}

/// Returns the indexed source-TFLite metadata key used for quantized tensors.
fn tensor_metadata_key(index: usize, field: &str) -> String {
    format!("vokra.kws.tensor.{index}.{field}")
}

fn get_tensor_string<'a>(gguf: &'a GgufFile, index: usize, field: &str) -> Result<&'a str> {
    let key = tensor_metadata_key(index, field);
    get_str(gguf, &key)
}

fn get_tensor_array<'a>(
    gguf: &'a GgufFile,
    index: usize,
    field: &str,
) -> Result<&'a vokra_core::gguf::GgufArray> {
    let key = tensor_metadata_key(index, field);
    let value = gguf
        .get(&key)
        .ok_or_else(|| VokraError::ModelLoad(format!("missing required metadata key `{key}`")))?;
    value
        .as_array()
        .ok_or_else(|| VokraError::ModelLoad(format!("metadata key `{key}` is not an array")))
}

fn get_tensor_quantization(
    gguf: &GgufFile,
    index: usize,
    info: &GgufTensorInfo,
) -> Result<TensorQuantization> {
    let scales_key = tensor_metadata_key(index, "quant.scales");
    let zero_points_key = tensor_metadata_key(index, "quant.zero_points");
    let dimension_key = tensor_metadata_key(index, "quant.quantized_dimension");
    let scales_array = get_tensor_array(gguf, index, "quant.scales")?;
    if scales_array.element_type != GgufValueType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "metadata key `{scales_key}` must be an F32 array"
        )));
    }
    let scales = scales_array
        .values
        .iter()
        .map(|value| {
            let scale = match value {
                GgufMetadataValue::F32(value) => *value,
                _ => {
                    return Err(VokraError::ModelLoad(format!(
                        "metadata key `{scales_key}` contains a non-F32 element"
                    )));
                }
            };
            if !(scale > 0.0 && scale.is_finite()) {
                return Err(VokraError::ModelLoad(format!(
                    "metadata key `{scales_key}` contains invalid scale {scale}"
                )));
            }
            Ok(scale)
        })
        .collect::<Result<Vec<_>>>()?;
    let zero_points_array = get_tensor_array(gguf, index, "quant.zero_points")?;
    if zero_points_array.element_type != GgufValueType::I64 {
        return Err(VokraError::ModelLoad(format!(
            "metadata key `{zero_points_key}` must be an I64 array"
        )));
    }
    let zero_points = zero_points_array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::I64(value) => Ok(*value),
            _ => Err(VokraError::ModelLoad(format!(
                "metadata key `{zero_points_key}` contains a non-I64 element"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let dimension = gguf
        .get(&dimension_key)
        .and_then(|value| match value {
            GgufMetadataValue::I32(value) => Some(*value),
            _ => None,
        })
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("metadata key `{dimension_key}` is not an I32"))
        })?;
    if scales.is_empty() || scales.len() != zero_points.len() {
        return Err(VokraError::ModelLoad(format!(
            "tensor ordinal {index} has {} scales but {} zero points",
            scales.len(),
            zero_points.len()
        )));
    }
    if zero_points
        .iter()
        .any(|&value| !(-128..=127).contains(&value))
    {
        return Err(VokraError::ModelLoad(format!(
            "metadata key `{zero_points_key}` contains a value outside signed INT8 range [-128, 127]"
        )));
    }
    let rank = info.dimensions.len();
    if scales.len() == 1 {
        // TFLite reports quantized_dimension=0 for some per-tensor tensors,
        // while other APIs use -1 as the scalar sentinel. Both are legal for
        // a non-scalar source tensor; any non-negative value must still name
        // an existing source axis. The core GGUF contract already rejects
        // rank-0 quantized tensors.
        let invalid_scalar_dimension = if dimension == -1 {
            false
        } else {
            match usize::try_from(dimension) {
                Ok(axis) => axis >= rank,
                Err(_) => true,
            }
        };
        if invalid_scalar_dimension {
            return Err(VokraError::ModelLoad(format!(
                "metadata key `{dimension_key}` scalar quantization dimension {dimension} is invalid for source rank {rank}"
            )));
        }
    } else {
        let Some(source_axis) = usize::try_from(dimension).ok().filter(|&axis| axis < rank) else {
            return Err(VokraError::ModelLoad(format!(
                "metadata key `{dimension_key}` per-axis dimension {dimension} is invalid for source rank {rank}"
            )));
        };
        let wire_axis = rank - 1 - source_axis;
        let axis_len = info.dimensions[wire_axis];
        if axis_len != scales.len() as u64 {
            return Err(VokraError::ModelLoad(format!(
                "tensor ordinal {index} per-axis metadata has {} scales for source axis {source_axis}, but wire axis {wire_axis} has length {axis_len}",
                scales.len()
            )));
        }
    }
    Ok(TensorQuantization {
        scales,
        zero_points,
        quantized_dimension: dimension,
    })
}

// The tests below use the std-only `GgufBuilder` writer (feature-gated in
// `vokra-core::gguf`) to synthesize valid + malformed GGUFs in memory, so
// they gate on `feature = "std"` — the same posture the sister
// `crates/vokra-vad-micro/src/weights.rs` tests use. The no_std load path itself is
// implicitly exercised: `Model::from_gguf` accepts any `&GgufFile`, and
// `GgufFile::parse` (called by `Model::from_bytes`) is the same code path
// under `#![no_std]` as under std.
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufBuilder, GgufMetadataValue};

    /// Every key the module-doc contract table declares required, in the
    /// order `ModelHeader::from_gguf` reads them.
    ///
    /// Kept as its own list so the sweeps below iterate the contract rather
    /// than a hand-copied subset;
    /// `required_key_list_covers_exactly_the_header_pairs` pins it against
    /// `valid_header_pairs`.
    const ALL_REQUIRED_KEYS: [&str; 10] = [
        KEY_ARCH,
        KEY_MODEL,
        KEY_THRESHOLD,
        KEY_SAMPLE_RATE,
        KEY_HOP_MS,
        KEY_WINDOW_MS,
        KEY_N_MELS,
        KEY_FEATURE_DIM,
        KEY_TFLITE_SHA256,
        KEY_UPSTREAM,
    ];

    /// The string-typed keys, read through `get_str`.
    const STRING_KEYS: [&str; 4] = [KEY_ARCH, KEY_MODEL, KEY_TFLITE_SHA256, KEY_UPSTREAM];

    /// The unsigned-integer dimensioning keys, read through
    /// `require_nonzero_u32` and so through `get_u32`.
    const U32_KEYS: [&str; 5] = [
        KEY_SAMPLE_RATE,
        KEY_HOP_MS,
        KEY_WINDOW_MS,
        KEY_N_MELS,
        KEY_FEATURE_DIM,
    ];

    /// The canonical ten-key header as typed `(key, value)` pairs, in the
    /// order the sidecar
    /// (`tools/parity/microwakeword/prepare_checkpoint.py`) emits them.
    ///
    /// Single source for both the happy-path builder and the per-key
    /// omission sweep, so a value used in one can never drift from the
    /// other.
    fn valid_header_pairs() -> Vec<(&'static str, GgufMetadataValue)> {
        vec![
            (
                KEY_ARCH,
                GgufMetadataValue::String(EXPECTED_ARCH.to_string()),
            ),
            (
                KEY_MODEL,
                GgufMetadataValue::String("hey_jarvis".to_string()),
            ),
            (KEY_THRESHOLD, GgufMetadataValue::F32(0.5)),
            (KEY_SAMPLE_RATE, GgufMetadataValue::U32(16_000)),
            (KEY_HOP_MS, GgufMetadataValue::U32(10)),
            (KEY_WINDOW_MS, GgufMetadataValue::U32(32)),
            (KEY_N_MELS, GgufMetadataValue::U32(40)),
            (KEY_FEATURE_DIM, GgufMetadataValue::U32(40)),
            (
                KEY_TFLITE_SHA256,
                GgufMetadataValue::String("0123456789abcdef".to_string()),
            ),
            (
                KEY_UPSTREAM,
                GgufMetadataValue::String("https://example.com/hey_jarvis.tflite".to_string()),
            ),
        ]
    }

    /// The full set of `vokra.kws.*` keys the sidecar emits for the
    /// canonical `hey_jarvis` v2 release. Every test that wants a
    /// well-formed header starts from this and mutates one field.
    fn add_valid_header(b: &mut GgufBuilder) {
        for (key, value) in valid_header_pairs() {
            b.add_metadata(key, value);
        }
    }

    /// Serializes an otherwise-valid header with one key replaced by
    /// `value`. [`GgufBuilder::add_metadata`] replaces an existing key in
    /// place, so the override wins over what `add_valid_header` stamped.
    fn build_with_override(key: &str, value: GgufMetadataValue) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_metadata(key, value);
        b.to_bytes().expect("serialize gguf")
    }

    /// Loads `bytes` requiring a [`VokraError::ModelLoad`], returning its
    /// message for substring assertions. `context` names the case so a
    /// sweep failure identifies which iteration broke.
    fn expect_model_load(bytes: &[u8], context: &str) -> String {
        match Model::from_bytes(bytes) {
            Err(VokraError::ModelLoad(m)) => m,
            other => panic!("expected VokraError::ModelLoad for {context}, got {other:?}"),
        }
    }

    fn topology_quant(scale: f32, zero_point: i64) -> TensorQuantization {
        TensorQuantization {
            scales: vec![scale],
            zero_points: vec![zero_point],
            quantized_dimension: -1,
        }
    }

    #[test]
    fn binds_untrusted_synthetic_linear_fully_connected_topology() {
        let model = Model {
            header: ModelHeader {
                model: "synthetic".into(),
                threshold: 0.5,
                sample_rate: 16_000,
                hop_ms: 10,
                window_ms: 32,
                n_mels: 1,
                feature_dim: 1,
                tflite_sha256: "0".repeat(64),
                upstream: "synthetic".into(),
            },
            tensors: vec![
                Tensor {
                    name: "input".into(),
                    shape: vec![1],
                    data: TensorData::Q8 {
                        values: vec![0.0],
                        raw: vec![0],
                    },
                    quantization: Some(topology_quant(1.0, 0)),
                },
                Tensor {
                    name: "weight".into(),
                    shape: vec![1, 1],
                    data: TensorData::Q8 {
                        values: vec![1.0],
                        raw: vec![1],
                    },
                    quantization: Some(topology_quant(1.0, 0)),
                },
                Tensor {
                    name: "bias".into(),
                    shape: vec![1],
                    data: TensorData::I32(vec![0]),
                    quantization: Some(topology_quant(1.0, 0)),
                },
            ],
        };
        let topology = TopologyManifest {
            tensors: vec![
                TopologyTensor {
                    index: 0,
                    name: "input".into(),
                    shape: vec![1],
                    dtype: TopologyDtype::Int8,
                    constant: false,
                    quantization: Some(topology_quant(1.0, 0)),
                },
                TopologyTensor {
                    index: 1,
                    name: "weight".into(),
                    shape: vec![1, 1],
                    dtype: TopologyDtype::Int8,
                    constant: true,
                    quantization: Some(topology_quant(1.0, 0)),
                },
                TopologyTensor {
                    index: 2,
                    name: "bias".into(),
                    shape: vec![1],
                    dtype: TopologyDtype::Int32,
                    constant: true,
                    quantization: Some(topology_quant(1.0, 0)),
                },
                TopologyTensor {
                    index: 3,
                    name: "output".into(),
                    shape: vec![1],
                    dtype: TopologyDtype::Int8,
                    constant: false,
                    quantization: Some(topology_quant(1.0, 0)),
                },
            ],
            inputs: vec![0],
            outputs: vec![3],
            operators: vec![TopologyOperator::FullyConnected {
                index: 0,
                input: 0,
                weight: 1,
                bias: 2,
                output: 3,
            }],
        };
        let chain = model
            .bind_untrusted_topology(&topology)
            .expect("synthetic topology binds");
        assert_eq!(chain.input_size(), 1);
        assert_eq!(chain.output_size(), 1);
        assert_eq!(chain.layer_count(), 1);
    }

    #[test]
    fn rejects_topology_quantization_tampering_against_gguf_identity() {
        let model = Model {
            header: ModelHeader {
                model: "synthetic".into(),
                threshold: 0.5,
                sample_rate: 16_000,
                hop_ms: 10,
                window_ms: 32,
                n_mels: 1,
                feature_dim: 1,
                tflite_sha256: "0".repeat(64),
                upstream: "synthetic".into(),
            },
            tensors: vec![
                Tensor {
                    name: "input".into(),
                    shape: vec![1],
                    data: TensorData::Q8 {
                        values: vec![0.0],
                        raw: vec![0],
                    },
                    quantization: Some(topology_quant(1.0, 0)),
                },
                Tensor {
                    name: "weight".into(),
                    shape: vec![1, 1],
                    data: TensorData::Q8 {
                        values: vec![1.0],
                        raw: vec![1],
                    },
                    quantization: Some(topology_quant(1.0, 0)),
                },
                Tensor {
                    name: "bias".into(),
                    shape: vec![1],
                    data: TensorData::I32(vec![0]),
                    quantization: Some(topology_quant(1.0, 0)),
                },
            ],
        };
        let topology = TopologyManifest {
            tensors: vec![
                TopologyTensor {
                    index: 0,
                    name: "input".into(),
                    shape: vec![1],
                    dtype: TopologyDtype::Int8,
                    constant: false,
                    quantization: Some(topology_quant(1.0, 0)),
                },
                TopologyTensor {
                    index: 1,
                    name: "weight".into(),
                    shape: vec![1, 1],
                    dtype: TopologyDtype::Int8,
                    constant: true,
                    quantization: Some(topology_quant(1.0, 0)),
                },
                TopologyTensor {
                    index: 2,
                    name: "bias".into(),
                    shape: vec![1],
                    dtype: TopologyDtype::Int32,
                    constant: true,
                    quantization: Some(topology_quant(1.0, 0)),
                },
                TopologyTensor {
                    index: 3,
                    name: "output".into(),
                    shape: vec![1],
                    dtype: TopologyDtype::Int8,
                    constant: false,
                    quantization: Some(topology_quant(1.0, 0)),
                },
            ],
            inputs: vec![0],
            outputs: vec![3],
            operators: vec![TopologyOperator::FullyConnected {
                index: 0,
                input: 0,
                weight: 1,
                bias: 2,
                output: 3,
            }],
        };
        for quantization in [
            TensorQuantization {
                scales: vec![2.0],
                zero_points: vec![0],
                quantized_dimension: -1,
            },
            TensorQuantization {
                scales: vec![1.0],
                zero_points: vec![1],
                quantized_dimension: -1,
            },
            TensorQuantization {
                scales: vec![1.0],
                zero_points: vec![0],
                quantized_dimension: 0,
            },
        ] {
            let mut tampered = topology.clone();
            tampered.tensors[1].quantization = Some(quantization);
            let error = model
                .bind_untrusted_topology(&tampered)
                .expect_err("caller quantization cannot override GGUF identity");
            assert!(matches!(error, VokraError::ModelLoad(_)));
        }
    }

    #[test]
    fn rejects_topology_with_invalid_graph_boundary() {
        let topology = TopologyManifest {
            tensors: vec![TopologyTensor {
                index: 0,
                name: "input".into(),
                shape: vec![1],
                dtype: TopologyDtype::Int8,
                constant: false,
                quantization: Some(topology_quant(1.0, 0)),
            }],
            inputs: vec![0],
            outputs: vec![0],
            operators: vec![TopologyOperator::Logistic {
                index: 0,
                input: 0,
                output: 0,
            }],
        };
        let error = validate_topology(&topology).expect_err("graph boundary cycle rejected");
        assert!(matches!(error, VokraError::ModelLoad(_)));
    }

    #[test]
    fn authenticated_binding_is_closed_until_reviewed_topology_exists() {
        let model = Model {
            header: ModelHeader {
                model: "synthetic".into(),
                threshold: 0.5,
                sample_rate: 16_000,
                hop_ms: 10,
                window_ms: 32,
                n_mels: 1,
                feature_dim: 1,
                tflite_sha256: "0".repeat(64),
                upstream: "synthetic".into(),
            },
            tensors: Vec::new(),
        };
        let error = model
            .bind_authenticated_chain()
            .expect_err("caller data cannot unlock production binding");
        assert!(
            matches!(error, VokraError::ModelLoad(message) if message == "AUTHENTICATED_TOPOLOGY_REQUIRED")
        );
    }

    /// Queues an F32 tensor whose element values are the arithmetic
    /// sequence `[0, 1, 2, ...]` (round-trip identity check).
    fn add_f32_tensor(b: &mut GgufBuilder, name: &str, dims: &[u64]) {
        let n: u64 = dims.iter().product();
        let mut data = Vec::with_capacity((n as usize) * 4);
        for i in 0..n {
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), data)
            .expect("add f32 tensor");
    }

    fn build_valid_bytes(tensors: &[(&str, &[u64])]) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        for (name, dims) in tensors {
            add_f32_tensor(&mut b, name, dims);
        }
        b.to_bytes().expect("serialize gguf")
    }

    // ---- round-trip -----------------------------------------------------

    #[test]
    fn from_bytes_round_trips_full_header() {
        let bytes = build_valid_bytes(&[]);
        let m = Model::from_bytes(&bytes).expect("valid gguf loads");
        assert_eq!(m.header.model, "hey_jarvis");
        assert_eq!(m.header.threshold, 0.5);
        assert_eq!(m.header.sample_rate, 16_000);
        assert_eq!(m.header.hop_ms, 10);
        assert_eq!(m.header.window_ms, 32);
        assert_eq!(m.header.n_mels, 40);
        assert_eq!(m.header.feature_dim, 40);
        assert_eq!(m.header.tflite_sha256, "0123456789abcdef");
        assert_eq!(m.header.upstream, "https://example.com/hey_jarvis.tflite");
        assert_eq!(m.tensor_count(), 0);
    }

    #[test]
    fn from_bytes_round_trips_multiple_tensors_in_declaration_order() {
        // Three tensors with distinct shapes; sizes chosen small so the
        // whole GGUF stays under ~1 KiB (this test does not go anywhere
        // near the real microWakeWord model's ~200 KiB weight footprint).
        let bytes =
            build_valid_bytes(&[("first", &[2, 3]), ("second", &[5]), ("third", &[1, 4, 2])]);
        let m = Model::from_bytes(&bytes).expect("valid gguf loads");
        assert_eq!(m.tensor_count(), 3);

        assert_eq!(m.tensors[0].name, "first");
        assert_eq!(m.tensors[0].shape, vec![2, 3]);
        assert_eq!(
            m.tensors[0].data,
            TensorData::F32((0..6).map(|i| i as f32).collect::<Vec<_>>())
        );

        assert_eq!(m.tensors[1].name, "second");
        assert_eq!(m.tensors[1].shape, vec![5]);
        assert_eq!(
            m.tensors[1].data,
            TensorData::F32((0..5).map(|i| i as f32).collect::<Vec<_>>())
        );

        assert_eq!(m.tensors[2].name, "third");
        assert_eq!(m.tensors[2].shape, vec![1, 4, 2]);
        assert_eq!(
            m.tensors[2].data,
            TensorData::F32((0..8).map(|i| i as f32).collect::<Vec<_>>())
        );
    }

    #[test]
    fn from_gguf_and_from_bytes_produce_identical_models() {
        let bytes = build_valid_bytes(&[("weights", &[4, 2])]);
        let via_bytes = Model::from_bytes(&bytes).expect("from_bytes loads");
        let gguf = GgufFile::parse(bytes).expect("outer parse");
        let via_gguf = Model::from_gguf(&gguf).expect("from_gguf loads");
        assert_eq!(via_bytes.header.model, via_gguf.header.model);
        assert_eq!(via_bytes.header.threshold, via_gguf.header.threshold);
        assert_eq!(via_bytes.header.feature_dim, via_gguf.header.feature_dim);
        assert_eq!(via_bytes.tensor_count(), via_gguf.tensor_count());
        assert_eq!(via_bytes.tensors[0].name, via_gguf.tensors[0].name);
        assert_eq!(via_bytes.tensors[0].shape, via_gguf.tensors[0].shape);
        assert_eq!(via_bytes.tensors[0].data, via_gguf.tensors[0].data);
    }

    #[test]
    fn tensor_lookup_returns_some_for_known_and_none_for_unknown() {
        let bytes = build_valid_bytes(&[("known", &[3])]);
        let m = Model::from_bytes(&bytes).unwrap();
        let t = m.tensor("known").expect("known tensor is present");
        assert_eq!(t.shape, vec![3]);
        assert!(
            m.tensor("unknown").is_none(),
            "unknown tensor lookup must be None (not a fake empty tensor)"
        );
    }

    #[test]
    fn feature_dim_can_differ_from_n_mels() {
        // A stacked-frame model may carry `feature_dim != n_mels`; the
        // loader must accept it (the sidecar emits both keys precisely so
        // this case is expressible in the wire format).
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_u32(KEY_FEATURE_DIM, 80); // stacked = 2 frames of 40 mels
        let bytes = b.to_bytes().expect("serialize");
        let m = Model::from_bytes(&bytes).expect("stacked-frame header loads");
        assert_eq!(m.header.n_mels, 40);
        assert_eq!(m.header.feature_dim, 80);
    }

    #[test]
    fn header_accepts_threshold_at_range_endpoints() {
        // Both endpoints of [0.0, 1.0] are legal (`contains` is inclusive
        // on both ends). Emitting `threshold = 1.0` is silly but not a
        // format violation, and `0.0` is a legitimate "wake on any
        // frame" degenerate mode useful for smoke tests.
        for t in [0.0_f32, 1.0_f32] {
            let mut b = GgufBuilder::new();
            add_valid_header(&mut b);
            b.add_f32(KEY_THRESHOLD, t);
            let bytes = b.to_bytes().expect("serialize");
            let m = Model::from_bytes(&bytes).expect("endpoint threshold loads");
            assert_eq!(m.header.threshold, t);
        }
    }

    // ---- malformed rejection --------------------------------------------

    #[test]
    fn rejects_missing_arch_key() {
        let mut b = GgufBuilder::new();
        // Omit only `KEY_ARCH`; every other required key is present.
        b.add_string(KEY_MODEL, "hey_jarvis");
        b.add_f32(KEY_THRESHOLD, 0.5);
        b.add_u32(KEY_SAMPLE_RATE, 16_000);
        b.add_u32(KEY_HOP_MS, 10);
        b.add_u32(KEY_WINDOW_MS, 32);
        b.add_u32(KEY_N_MELS, 40);
        b.add_u32(KEY_FEATURE_DIM, 40);
        b.add_string(KEY_TFLITE_SHA256, "abc");
        b.add_string(KEY_UPSTREAM, "u");
        let bytes = b.to_bytes().unwrap();
        match Model::from_bytes(&bytes) {
            Err(VokraError::ModelLoad(m)) => {
                assert!(m.contains(KEY_ARCH), "message names the missing key: {m}");
            }
            other => panic!("expected ModelLoad for missing arch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_arch_value() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string(KEY_ARCH, "openwakeword"); // wrong ecosystem
        let bytes = b.to_bytes().unwrap();
        match Model::from_bytes(&bytes) {
            Err(VokraError::ModelLoad(m)) => {
                assert!(
                    m.contains("openwakeword"),
                    "message names the wrong arch: {m}"
                );
                assert!(
                    m.contains(EXPECTED_ARCH),
                    "message names the expected arch: {m}"
                );
            }
            other => panic!("expected ModelLoad for wrong arch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_f32_tensor_dtype() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        // Add an F16 tensor (dtype tag 1) — the loader rejects any
        // non-F32 dtype. The GGUF outer layer happily accepts F16 (it is
        // in the `UnsupportedDtype`-not-listed set), so the rejection
        // happens in `Model::from_gguf`.
        b.add_tensor("f16_weight", GgmlType::F16, vec![4], vec![0u8; 4 * 2])
            .expect("queue f16 tensor");
        let bytes = b.to_bytes().unwrap();
        match Model::from_bytes(&bytes) {
            Err(VokraError::ModelLoad(m)) => {
                assert!(m.contains("f16_weight"), "message names the offender: {m}");
                assert!(m.contains("F32"), "message names the required dtype: {m}");
            }
            other => panic!("expected ModelLoad for F16 dtype, got {other:?}"),
        }
    }

    #[test]
    fn loads_q8_0_tensor_with_source_tflite_quantization() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "q8_weight");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(-3)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; GgmlType::Q8_0.type_size()];
        payload[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        payload[2] = 7;
        payload[3] = 0x80;
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32], payload)
            .expect("queue q8 tensor");
        let bytes = b.to_bytes().expect("serialize q8 gguf");
        let model = Model::from_bytes(&bytes).expect("q8 gguf loads");
        let tensor = model.tensor("q8_weight").expect("q8 tensor present");
        assert_eq!(
            tensor.data,
            TensorData::Q8 {
                values: vec![7.0, -128.0]
                    .into_iter()
                    .chain(std::iter::repeat(0.0).take(30))
                    .collect(),
                raw: vec![7, -128]
                    .into_iter()
                    .chain(std::iter::repeat(0).take(30))
                    .collect(),
            }
        );
        assert_eq!(
            tensor.quantization,
            Some(TensorQuantization {
                scales: vec![0.125],
                zero_points: vec![-3],
                quantized_dimension: 0,
            })
        );
    }

    #[test]
    fn loads_i32_bias_exactly_without_an_f32_mirror() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "bias");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(0)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(-1),
        );
        let values = [i32::MIN, (1 << 24) + 1, i32::MAX];
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        b.add_tensor("bias", GgmlType::I32, vec![values.len() as u64], payload)
            .unwrap();
        let model = Model::from_bytes(&b.to_bytes().unwrap()).expect("I32 bias loads");
        assert!(matches!(
            model.tensor("bias").unwrap().data,
            TensorData::I32(ref actual) if actual == &values
        ));
    }

    fn build_q8_metadata_type_case(
        scales: vokra_core::gguf::GgufArray,
        zero_points: vokra_core::gguf::GgufArray,
    ) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "q8_weight");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(scales),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(zero_points),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; GgmlType::Q8_0.type_size()];
        payload[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32], payload)
            .expect("queue q8 tensor");
        b.to_bytes().expect("serialize q8 metadata type case")
    }

    #[test]
    fn rejects_q8_metadata_arrays_with_nonproducer_element_types() {
        let valid_zero_points = vokra_core::gguf::GgufArray {
            element_type: vokra_core::gguf::GgufValueType::I64,
            values: vec![GgufMetadataValue::I64(0)],
        };
        let scales_wrong_type = build_q8_metadata_type_case(
            vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F64,
                values: vec![GgufMetadataValue::F64(0.125)],
            },
            valid_zero_points.clone(),
        );
        let message = expect_model_load(&scales_wrong_type, "F64 scales array");
        assert!(message.contains("scales") && message.contains("F32 array"));

        let zero_points_wrong_type = build_q8_metadata_type_case(
            vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125)],
            },
            vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I32,
                values: vec![GgufMetadataValue::I32(0)],
            },
        );
        let message = expect_model_load(&zero_points_wrong_type, "I32 zero_points array");
        assert!(message.contains("zero_points") && message.contains("I64 array"));
    }

    #[test]
    fn loads_q8_0_per_axis_quantization_using_source_to_wire_axis_mapping() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "q8_weight");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125), GgufMetadataValue::F32(0.25)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(-3), GgufMetadataValue::I64(7)],
            }),
        );
        // Source shape [2, 32] is written in GGUF wire order as [32, 2].
        // quantized_dimension=0 therefore maps to wire axis 1.
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; 2 * GgmlType::Q8_0.type_size()];
        for block in payload.chunks_exact_mut(GgmlType::Q8_0.type_size()) {
            block[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32, 2], payload)
            .expect("queue per-axis q8 tensor");
        let model = Model::from_bytes(&b.to_bytes().unwrap()).expect("per-axis q8 loads");
        assert_eq!(
            model.tensor("q8_weight").unwrap().quantization,
            Some(TensorQuantization {
                scales: vec![0.125, 0.25],
                zero_points: vec![-3, 7],
                quantized_dimension: 0,
            })
        );
    }

    #[test]
    fn rejects_q8_0_per_axis_metadata_when_wire_axis_length_differs() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "q8_weight");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125), GgufMetadataValue::F32(0.25)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(-3), GgufMetadataValue::I64(7)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; 3 * GgmlType::Q8_0.type_size()];
        for block in payload.chunks_exact_mut(GgmlType::Q8_0.type_size()) {
            block[0] = 0x00;
            block[1] = 0x3c;
        }
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32, 3], payload)
            .expect("queue malformed-axis q8 tensor");
        let error = Model::from_bytes(&b.to_bytes().unwrap())
            .expect_err("per-axis shape mismatch must fail");
        assert!(matches!(error, VokraError::ModelLoad(message) if message.contains("wire axis")));
    }

    #[test]
    fn rejects_q8_0_zero_point_outside_int8_range() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "q8_weight");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(128)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; GgmlType::Q8_0.type_size()];
        payload[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32], payload)
            .expect("queue overflow-zero-point q8 tensor");
        let error =
            Model::from_bytes(&b.to_bytes().unwrap()).expect_err("zero point overflow must fail");
        assert!(matches!(error, VokraError::ModelLoad(message) if message.contains("INT8 range")));
    }

    #[test]
    fn rejects_q8_metadata_attached_to_f32_ordinal() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "f32_weight");
        b.add_tensor("f32_weight", GgmlType::F32, vec![1], vec![0u8; 4])
            .expect("queue f32 tensor");
        let error = Model::from_bytes(&b.to_bytes().unwrap())
            .expect_err("Q8 metadata on F32 must fail closed");
        assert!(
            matches!(error, VokraError::ModelLoad(message) if message.contains("carries Q8_0 metadata"))
        );
    }

    #[test]
    fn rejects_q8_0_block_scale_that_is_not_exactly_one() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "q8_weight");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(-3)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; GgmlType::Q8_0.type_size()];
        payload[..2].copy_from_slice(&0x4000u16.to_le_bytes()); // 2.0, not 1.0
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32], payload)
            .expect("queue q8 tensor");
        let bytes = b.to_bytes().expect("serialize q8 gguf");
        let error = Model::from_bytes(&bytes).expect_err("non-unit carrier scale must fail");
        assert!(
            matches!(error, VokraError::ModelLoad(message) if message.contains("exact FP16 1.0"))
        );
    }

    #[test]
    fn rejects_q8_0_source_name_mismatch() {
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.kws.tensor.0.name", "different_name");
        b.add_metadata(
            "vokra.kws.tensor.0.quant.scales",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::F32,
                values: vec![GgufMetadataValue::F32(0.125)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.zero_points",
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::I64,
                values: vec![GgufMetadataValue::I64(-3)],
            }),
        );
        b.add_metadata(
            "vokra.kws.tensor.0.quant.quantized_dimension",
            GgufMetadataValue::I32(0),
        );
        let mut payload = vec![0u8; GgmlType::Q8_0.type_size()];
        payload[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b.add_tensor("q8_weight", GgmlType::Q8_0, vec![32], payload)
            .expect("queue q8 tensor");
        let bytes = b.to_bytes().expect("serialize q8 gguf");
        let error = Model::from_bytes(&bytes).expect_err("source name mismatch must fail");
        assert!(
            matches!(error, VokraError::ModelLoad(ref message) if message.contains("source name metadata")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn rejects_threshold_out_of_range() {
        // Both above 1.0 and below 0.0 are rejected.
        for t in [-0.1_f32, 1.5_f32] {
            let mut b = GgufBuilder::new();
            add_valid_header(&mut b);
            b.add_f32(KEY_THRESHOLD, t);
            let bytes = b.to_bytes().unwrap();
            match Model::from_bytes(&bytes) {
                Err(VokraError::ModelLoad(m)) => {
                    assert!(m.contains(KEY_THRESHOLD), "message names the key: {m}");
                }
                other => panic!("expected ModelLoad for threshold {t}, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_zero_dimensioning_key() {
        // Every dimensioning key rejects 0. Iterate so a new dimensioning
        // key added later must extend this list (fail-closed by omission).
        let zeroable_keys = [
            KEY_SAMPLE_RATE,
            KEY_HOP_MS,
            KEY_WINDOW_MS,
            KEY_N_MELS,
            KEY_FEATURE_DIM,
        ];
        for key in zeroable_keys {
            let mut b = GgufBuilder::new();
            add_valid_header(&mut b);
            b.add_u32(key, 0); // override the valid value with 0
            let bytes = b.to_bytes().unwrap();
            match Model::from_bytes(&bytes) {
                Err(VokraError::ModelLoad(m)) => {
                    assert!(m.contains(key), "message names the zero key `{key}`: {m}");
                }
                other => panic!("expected ModelLoad for zero {key}, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_bad_magic_bytes() {
        // Not "GGUF" at all — the GGUF outer layer rejects at parse time,
        // which surfaces as `VokraError::ModelLoad` through the
        // `GgufError` → `VokraError` conversion.
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0];
        match Model::from_bytes(&bytes) {
            Err(VokraError::ModelLoad(m)) => {
                // The GgufError::BadMagic Display carries "bad GGUF magic".
                assert!(
                    m.contains("magic") || m.contains("GGUF"),
                    "message mentions magic / GGUF: {m}"
                );
            }
            other => panic!("expected ModelLoad for bad magic, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_input() {
        // Fewer bytes than even the GGUF header (magic 4 + version 4 +
        // tensor_count 8 + kv_count 8 = 24 bytes minimum).
        let bytes = vec![b'G', b'G', b'U', b'F']; // magic only, then truncated
        assert!(
            matches!(Model::from_bytes(&bytes), Err(VokraError::ModelLoad(_))),
            "truncated input must surface as ModelLoad (FR-EX-08)"
        );
    }

    #[test]
    fn rejects_wrong_type_for_threshold() {
        // `KEY_THRESHOLD` is contractually f32; sending a string must fail
        // fast, not silently coerce to zero (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ARCH, EXPECTED_ARCH);
        b.add_string(KEY_MODEL, "hey_jarvis");
        b.add_string(KEY_THRESHOLD, "not-a-float"); // wrong type
        b.add_u32(KEY_SAMPLE_RATE, 16_000);
        b.add_u32(KEY_HOP_MS, 10);
        b.add_u32(KEY_WINDOW_MS, 32);
        b.add_u32(KEY_N_MELS, 40);
        b.add_u32(KEY_FEATURE_DIM, 40);
        b.add_string(KEY_TFLITE_SHA256, "abc");
        b.add_string(KEY_UPSTREAM, "u");
        let bytes = b.to_bytes().unwrap();
        match Model::from_bytes(&bytes) {
            Err(VokraError::ModelLoad(m)) => {
                assert!(
                    m.contains(KEY_THRESHOLD),
                    "message names the wrong-type key: {m}"
                );
                assert!(m.contains("float"), "message mentions expected type: {m}");
            }
            other => panic!("expected ModelLoad for wrong threshold type, got {other:?}"),
        }
    }

    // ---- contract sweeps: every required key, every key kind ------------
    //
    // The focused tests above cover one representative of each failure
    // mode. These sweeps cover the whole ten-key contract, so a key that
    // gains an accessor without gaining enforcement cannot slip through.

    #[test]
    fn required_key_list_covers_exactly_the_header_pairs() {
        // Fail-closed structural pin: a key added to the contract (and so
        // to `valid_header_pairs`) without being added to
        // `ALL_REQUIRED_KEYS` would silently drop out of every sweep below,
        // leaving it unenforced but apparently covered.
        let pair_keys: Vec<&str> = valid_header_pairs().into_iter().map(|(k, _)| k).collect();
        assert_eq!(pair_keys, ALL_REQUIRED_KEYS.to_vec());
        assert_eq!(
            ALL_REQUIRED_KEYS.len(),
            10,
            "the module doc declares a ten-key contract"
        );
    }

    #[test]
    fn key_kind_groups_partition_the_required_keys() {
        // Every required key is read by exactly one of the three typed
        // accessors. `KEY_THRESHOLD` is the sole float key, so the two
        // group constants plus it must reconstruct the full set — a new
        // key left out of both groups fails here.
        let mut covered: Vec<&str> = STRING_KEYS.to_vec();
        covered.extend_from_slice(&U32_KEYS);
        covered.push(KEY_THRESHOLD);
        covered.sort_unstable();
        let mut all = ALL_REQUIRED_KEYS.to_vec();
        all.sort_unstable();
        assert_eq!(covered, all, "every required key needs a typed-kind group");
    }

    #[test]
    fn rejects_every_missing_required_key_by_name() {
        // Omit exactly one key per iteration; every other key stays valid,
        // so the error raised is unambiguously the one under test.
        for skip in ALL_REQUIRED_KEYS {
            let mut b = GgufBuilder::new();
            for (key, value) in valid_header_pairs() {
                if key != skip {
                    b.add_metadata(key, value);
                }
            }
            let bytes = b.to_bytes().expect("serialize gguf");
            let m = expect_model_load(&bytes, skip);
            assert!(
                m.contains(skip),
                "message must name the missing key `{skip}`: {m}"
            );
            assert!(
                m.contains("missing"),
                "message must say the key is missing `{skip}`: {m}"
            );
        }
    }

    #[test]
    fn rejects_non_string_value_for_every_string_key() {
        // A U32 where a string is contractually required must fail, not
        // bind a stringified fallback (FR-EX-08).
        for key in STRING_KEYS {
            let bytes = build_with_override(key, GgufMetadataValue::U32(7));
            let m = expect_model_load(&bytes, key);
            assert!(m.contains(key), "message must name `{key}`: {m}");
            assert!(
                m.contains("not a string"),
                "message must state the expected kind for `{key}`: {m}"
            );
        }
    }

    #[test]
    fn rejects_non_integer_value_for_every_u32_key() {
        // A numeric-looking STRING must not be parsed into a number: GGUF
        // values are typed, and text parsing here would also drag in the
        // locale-dependent `strtod` trap the format deliberately avoids.
        for key in U32_KEYS {
            let bytes = build_with_override(key, GgufMetadataValue::String("40".to_string()));
            let m = expect_model_load(&bytes, key);
            assert!(m.contains(key), "message must name `{key}`: {m}");
            assert!(
                m.contains("not an unsigned integer"),
                "a numeric-looking string must not be parsed for `{key}`: {m}"
            );
        }
    }

    #[test]
    fn rejects_signed_integer_for_every_u32_key() {
        // `as_u64` refuses signed variants outright, so -1 surfaces as a
        // wrong-type refusal instead of widening to u64::MAX and failing
        // later with a confusing out-of-range message.
        for key in U32_KEYS {
            let bytes = build_with_override(key, GgufMetadataValue::I32(-1));
            let m = expect_model_load(&bytes, key);
            assert!(m.contains(key), "message must name `{key}`: {m}");
            assert!(
                m.contains("not an unsigned integer"),
                "signed -1 must not widen for `{key}`: {m}"
            );
        }
    }

    #[test]
    fn rejects_u32_key_whose_value_exceeds_u32() {
        // `get_u32` accepts any unsigned width and then narrows; a value
        // past u32::MAX is a corrupt or misidentified artifact and must be
        // refused rather than truncated.
        let too_big = u64::from(u32::MAX) + 1;
        for key in U32_KEYS {
            let bytes = build_with_override(key, GgufMetadataValue::U64(too_big));
            let m = expect_model_load(&bytes, key);
            assert!(m.contains(key), "message must name `{key}`: {m}");
            assert!(
                m.contains("u32"),
                "message must state the narrowing failure for `{key}`: {m}"
            );
        }
    }

    #[test]
    fn accepts_narrower_unsigned_widths_for_u32_keys() {
        // The documented widening (U8 / U16 / U32 / U64 all accepted) is
        // deliberate: GGUF writers pick the narrowest type that fits, so a
        // U8-tagged `n_mels` is a legitimate file, not a corrupt one. This
        // is the positive half of the narrowing contract above.
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_metadata(KEY_SAMPLE_RATE, GgufMetadataValue::U64(16_000));
        b.add_metadata(KEY_HOP_MS, GgufMetadataValue::U16(10));
        b.add_metadata(KEY_N_MELS, GgufMetadataValue::U8(40));
        let bytes = b.to_bytes().expect("serialize gguf");
        let m = Model::from_bytes(&bytes).expect("widened unsigned header loads");
        assert_eq!(m.header.sample_rate, 16_000);
        assert_eq!(m.header.hop_ms, 10);
        assert_eq!(m.header.n_mels, 40);
    }

    #[test]
    fn rejects_integer_typed_threshold() {
        // `as_f64` accepts F32 / F64 only, so an integer-tagged threshold
        // is a wrong-type refusal — it must NOT coerce to 1.0, which would
        // silently install a never-fires wake cutoff.
        let bytes = build_with_override(KEY_THRESHOLD, GgufMetadataValue::U32(1));
        let m = expect_model_load(&bytes, KEY_THRESHOLD);
        assert!(m.contains(KEY_THRESHOLD), "message must name the key: {m}");
        assert!(
            m.contains("not a float"),
            "message must state the expected kind: {m}"
        );
    }

    #[test]
    fn rejects_non_finite_threshold() {
        // NaN and ±inf are corrupt artifacts. The F64 case additionally
        // pins the "non-finite *after narrowing*" path: 1e300 is a finite
        // f64 that becomes +inf as f32, so a reader that checked
        // finiteness before narrowing would bind an infinite threshold —
        // and `(0.0..=1.0).contains(&inf)` being false would then produce
        // a misleading out-of-range message instead.
        let cases = [
            GgufMetadataValue::F32(f32::NAN),
            GgufMetadataValue::F32(f32::INFINITY),
            GgufMetadataValue::F32(f32::NEG_INFINITY),
            GgufMetadataValue::F64(1e300),
        ];
        for value in cases {
            let bytes = build_with_override(KEY_THRESHOLD, value);
            let m = expect_model_load(&bytes, KEY_THRESHOLD);
            assert!(m.contains(KEY_THRESHOLD), "message must name the key: {m}");
            assert!(
                m.contains("non-finite"),
                "message must state non-finiteness: {m}"
            );
        }
    }

    #[test]
    fn rejects_threshold_just_outside_the_inclusive_bounds() {
        // Pairs with `header_accepts_threshold_at_range_endpoints`:
        // together they pin the bound as inclusive at exactly 0.0 and 1.0,
        // so a `<` / `<=` slip in either direction is caught. Both values
        // are the adjacent representable f32 to the endpoint.
        for t in [-f32::EPSILON, 1.0_f32 + f32::EPSILON] {
            let bytes = build_with_override(KEY_THRESHOLD, GgufMetadataValue::F32(t));
            let m = expect_model_load(&bytes, KEY_THRESHOLD);
            assert!(
                m.contains(KEY_THRESHOLD),
                "message must name the key for {t}: {m}"
            );
        }
    }

    #[test]
    fn rejects_every_non_f32_tensor_dtype() {
        // The loader accepts F32 and the source-byte-carrier Q8_0 only. F16
        // has its own focused test above; this sweep adds the other dtypes
        // `GgmlType` can carry so the accepted set cannot silently widen.
        let cases: [(&str, GgmlType, Vec<u64>, usize); 3] = [
            ("bf16_weight", GgmlType::BF16, vec![4], 4 * 2),
            ("f16_weight", GgmlType::F16, vec![4], 4 * 2),
            // K-quants are block-addressed: one Q6_K super-block is 256
            // elements stored in 210 bytes (`GgmlType::type_size`).
            ("q6k_weight", GgmlType::Q6K, vec![256], 210),
        ];
        for (name, dtype, dims, payload) in cases {
            let mut b = GgufBuilder::new();
            add_valid_header(&mut b);
            b.add_tensor(name, dtype, dims, vec![0u8; payload])
                .expect("queue tensor");
            let bytes = b.to_bytes().expect("serialize gguf");
            let m = expect_model_load(&bytes, name);
            assert!(
                m.contains(name),
                "message must name the offending tensor: {m}"
            );
            assert!(
                m.contains("F32"),
                "message must name the required dtype: {m}"
            );
        }
    }

    #[test]
    fn ignores_unknown_metadata_keys() {
        // The sidecar also stamps `vokra.provenance.*`, and the writer
        // injects `general.*`. The loader must read its own group and
        // leave the rest alone — otherwise every provenance-stamped real
        // file (i.e. every file the sidecar actually emits) would fail.
        let mut b = GgufBuilder::new();
        add_valid_header(&mut b);
        b.add_string("vokra.provenance.license", "apache-2.0");
        b.add_string("vokra.provenance.upstream_hf", "kahrendt/microWakeWord");
        b.add_u32("some.unrelated.key", 1);
        let bytes = b.to_bytes().expect("serialize gguf");
        let m = Model::from_bytes(&bytes).expect("extra metadata must not break the bind");
        assert_eq!(m.header.model, "hey_jarvis");
    }

    #[test]
    fn tensor_shape_is_preserved_in_on_disk_order() {
        // `Tensor::shape` is GGUF wire order (innermost axis first), copied
        // verbatim from the tensor info. A sidecar-produced file has the
        // reverse of NumPy `.shape`, while a Rust builder fixture already
        // supplies wire-order dimensions directly.
        let bytes = build_valid_bytes(&[("w", &[2, 3, 4])]);
        let m = Model::from_bytes(&bytes).expect("valid gguf loads");
        assert_eq!(m.tensor("w").expect("tensor present").shape, vec![2, 3, 4]);
        assert!(matches!(m.tensors[0].data, TensorData::F32(ref values) if values.len() == 24));
    }
}
