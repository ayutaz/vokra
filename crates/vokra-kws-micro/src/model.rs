//! microWakeWord GGUF model loader (M5-03b Phase 2).
//!
//! Reads a Vokra microWakeWord GGUF (emitted by the offline sidecar
//! [`tools/parity/microwakeword/prepare_checkpoint.py`]) and yields a typed
//! [`Model`] carrying the audio-frontend contract (from `vokra.kws.*`
//! metadata) plus every dense F32 weight tensor.
//!
//! # What a `Model` can and cannot reach today
//!
//! This module is the runtime *loader*: it parses and validates, and does not
//! itself run inference. The forward it would feed is real — the INT8 kernels
//! ([`crate::kernels`]), the chain executor ([`crate::interpreter`]) and
//! [`crate::KwsMicro::detect`] all execute for real on an attached
//! [`crate::interpreter::ChainConfig`].
//!
//! What is missing is the join between the two: **no code path converts a
//! [`Model`] into a [`crate::interpreter::ChainConfig`]**, and it cannot be
//! written from what a [`Model`] currently holds. The sidecar dequantises the
//! upstream INT8 weights to F32 at export time (see the note under *Tensors*
//! below), so the per-tensor `(scale, zero_point)` pairs that every
//! [`crate::interpreter::LayerSpec`] needs are simply not in the file. A
//! [`Model`] is therefore usable for shape audits and metadata inspection,
//! but a chain built from an upstream checkpoint waits on the sidecar
//! re-emitting those params.
//!
//! # Design rationale (why a two-layer parser, not a monolithic one)
//!
//! The wire format is a stock **GGUF v3** file: the Python sidecar uses
//! `gguf.GGUFWriter` (Apache-2.0) so the metadata + tensor layout matches
//! every other GGUF the Vokra ecosystem emits. This crate therefore reuses
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
//! Every tensor is bound generically as a [`Tensor`] (name + shape + F32
//! payload). Per-layer typed bindings (Conv2d / DwConv2d / Dense weight
//! blocks, mirroring the `Conv1dW` pattern in
//! `crates/vokra-vad-micro/src/weights.rs` — that module is private and
//! `Conv1dW` is `pub(crate)`, so neither has a docs.rs page to link) are not
//! written yet, and are blocked on the same missing quantisation params as
//! the chain builder.
//!
//! Quantization params are absent from the file: the sidecar
//! (`tools/parity/microwakeword/prepare_checkpoint.py`) dequantizes
//! INT8 → F32 before emit (the arithmetic is
//! `f32 = scale · (int8 - zero_point)` for a fixed per-tensor
//! `(scale, zero_point)` pair, so the values round-trip losslessly — but the
//! pair itself is discarded). Re-emitting them alongside Q8_0 storage, once
//! [`vokra_core::gguf::GgmlType`] gains that variant, is the follow-up that
//! unblocks binding an upstream checkpoint to a chain.
//!
//! # Ops are NOT represented
//!
//! The sidecar emits weights only — the microWakeWord architecture is a
//! fixed MC-MobileNet whose op chain is hard-coded on the consumer side.
//! Adding a `Op { kind, inputs, outputs, attributes }` struct here would be
//! fake-complete (it would carry no data). [`crate::KwsMicro::detect`]
//! instead drives scalar INT8 kernels through a hand-written topology (see
//! [`crate::interpreter`]), matching the sister [`vokra_vad_micro`] and
//! whisper.cpp `whisper_encoder` patterns.
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
    vec::Vec,
};

use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

/// The `vokra.kws.arch` discriminator the sidecar emits for microWakeWord
/// artifacts. A GGUF whose `arch` string differs is rejected outright:
/// microWakeWord (MC-MobileNet on M55) and openWakeWord (speech-embed MLP
/// on RPi/Linux) are separate ecosystems and their weight layouts do not
/// interchange. Downstream binders switch on this key.
pub const EXPECTED_ARCH: &str = "microwakeword";

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

/// One dense F32 weight tensor decoded from the GGUF.
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
    /// `gguf.GGUFWriter`, which the sidecar uses, packs dims back-to-front
    /// at write time (the `ti.shape[n_dims - 1 - j]` index in
    /// `gguf/gguf_writer.py`), so a numpy `(out, in)` weight arrives here
    /// as `[in, out]`. Read this field as wire order and reverse it if you
    /// need the upstream framework's convention.
    pub shape: Vec<u64>,
    /// Decoded F32 payload (little-endian off-disk → host `f32`).
    pub data: Vec<f32>,
}

/// A parsed microWakeWord GGUF: audio-frontend contract + every dense F32
/// weight tensor. Constructed via [`Model::from_bytes`] (owns a `Vec<u8>`
/// copy of the input) or [`Model::from_gguf`] (borrows a prebuilt
/// [`GgufFile`]).
#[derive(Debug, Clone)]
pub struct Model {
    /// Typed view of the `vokra.kws.*` metadata group.
    pub header: ModelHeader,
    /// Every dense F32 tensor in the file, in GGUF declaration order.
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
    /// - any tensor with a dtype other than
    ///   [`GgmlType::F32`] (Phase 2 = F32 only; Q8_0 lands in Phase 3).
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
        for info in gguf.tensors() {
            if info.dtype != GgmlType::F32 {
                return Err(VokraError::ModelLoad(format!(
                    "tensor `{}` has dtype {:?}, expected F32 \
                     (Phase 2 = F32 only; Q8_0 lands in Phase 3)",
                    info.name, info.dtype
                )));
            }
            let bytes = gguf.tensor_bytes(info);
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
                data,
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
            (0..6).map(|i| i as f32).collect::<Vec<_>>()
        );

        assert_eq!(m.tensors[1].name, "second");
        assert_eq!(m.tensors[1].shape, vec![5]);
        assert_eq!(
            m.tensors[1].data,
            (0..5).map(|i| i as f32).collect::<Vec<_>>()
        );

        assert_eq!(m.tensors[2].name, "third");
        assert_eq!(m.tensors[2].shape, vec![1, 4, 2]);
        assert_eq!(
            m.tensors[2].data,
            (0..8).map(|i| i as f32).collect::<Vec<_>>()
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
        // Add an F16 tensor (dtype tag 1) — Phase 2 loader rejects any
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
        // Phase 2 is F32-only. F16 has its own focused test above; this
        // sweep adds the other dtypes `GgmlType` can carry so "F32 only"
        // cannot decay into "anything but F16".
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
        // verbatim from the tensor info: the Rust writer and reader both
        // round-trip dims without reordering. This is the REVERSE of numpy
        // `.shape` on a sidecar-produced file, because `gguf.GGUFWriter`
        // packs dims back-to-front at write time — a Rust round-trip
        // cannot exhibit that, so the field doc carries the citation.
        let bytes = build_valid_bytes(&[("w", &[2, 3, 4])]);
        let m = Model::from_bytes(&bytes).expect("valid gguf loads");
        assert_eq!(m.tensor("w").expect("tensor present").shape, vec![2, 3, 4]);
        assert_eq!(m.tensors[0].data.len(), 24);
    }
}
