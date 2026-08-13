//! microWakeWord GGUF model loader (M5-03b Phase 2).
//!
//! Reads a Vokra microWakeWord GGUF (emitted by the offline sidecar
//! [`tools/parity/microwakeword/prepare_checkpoint.py`]) and yields a typed
//! [`Model`] carrying the audio-frontend contract (from `vokra.kws.*`
//! metadata) plus every dense F32 weight tensor. This is the bridge that
//! lets a future [`crate::KwsMicro::detect`] (Phase 3) consume the weights
//! the sidecar emitted.
//!
//! # SCAFFOLD posture (Phase 2 of 3)
//!
//! Phase 1 (WF1) landed the offline TFLite → GGUF sidecar and the log-mel
//! feature extractor ([`crate::features`]). This module (Phase 2) lands the
//! runtime *loader* — a shape-generic view of the GGUF contents that Phase 3
//! (INT8 kernel chain + real [`crate::KwsMicro::detect`]) will consume.
//! It does NOT run inference; a [`Model`] carrying weights but no forward
//! is inert on purpose (matches the crate-level SCAFFOLD contract).
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
//! [`vokra_vad_micro::weights::SileroWeights::from_gguf`] pattern and keeps
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
//! [`VokraError::ModelLoad`], never a silent bind.
//!
//! # Tensors (Phase 2 shape-generic)
//!
//! Every tensor is bound generically as a [`Tensor`] (name + shape + F32
//! payload). Phase 3 will introduce per-layer typed bindings (Conv2d /
//! DwConv2d / Dense weight blocks, mirroring the [`Conv1dW`] pattern from
//! [`vokra_vad_micro::weights`]) once the MC-MobileNet forward is wired.
//!
//! Quantization params are deliberately absent: the Phase 1 sidecar
//! dequantizes INT8 → F32 losslessly before emit (the arithmetic is
//! `f32 = scale · (int8 - zero_point)` for a fixed per-tensor
//! `(scale, zero_point)` pair). Phase 3 will re-introduce them alongside
//! Q8_0 support once [`vokra_core::gguf::GgmlType`] gains that variant.
//!
//! # Ops are NOT represented
//!
//! The sidecar emits weights only — the microWakeWord architecture is a
//! fixed MC-MobileNet whose op chain is hard-coded on the consumer side.
//! Adding a `Op { kind, inputs, outputs, attributes }` struct here would be
//! fake-complete (it would carry no data). Phase 3's `KwsMicro::detect`
//! will call scalar INT8 kernels directly against these [`Tensor`] weights
//! in a hand-written topology, matching the sister [`vokra_vad_micro`] and
//! whisper.cpp `whisper_encoder` patterns.
//!
//! [`Conv1dW`]: https://docs.rs/vokra-vad-micro/latest/vokra_vad_micro/weights/struct.Conv1dW.html
//! [`vokra_vad_micro`]: https://docs.rs/vokra-vad-micro
//! [`vokra_vad_micro::weights`]: https://docs.rs/vokra-vad-micro/latest/vokra_vad_micro/weights/index.html
//! [`vokra_vad_micro::weights::SileroWeights::from_gguf`]: https://docs.rs/vokra-vad-micro/latest/vokra_vad_micro/weights/struct.SileroWeights.html#method.from_gguf

// `alloc` items that are in the prelude under `std` need explicit imports
// under `#![no_std]`. Mirrors the sister `vokra-vad-micro::weights` gate
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
    /// Wake-decision cutoff in `[0.0, 1.0]`. The Phase 3 forward will emit
    /// a wake event only when the model's softmax score exceeds this.
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
    /// Row-major dimensions, innermost first, as stored on disk. Matches
    /// numpy `.shape` on the source tensor.
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
    /// Phase 3 will introduce per-layer typed bindings on top of this;
    /// Phase 2 keeps the shape-generic list so shape audits (via
    /// `tests`) and the future forward can walk it independently.
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
                .chunks_exact(4)
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
// `vokra-vad-micro::weights` tests use. The no_std load path itself is
// implicitly exercised: `Model::from_gguf` accepts any `&GgufFile`, and
// `GgufFile::parse` (called by `Model::from_bytes`) is the same code path
// under `#![no_std]` as under std.
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    /// The full set of `vokra.kws.*` keys the sidecar emits for the
    /// canonical `hey_jarvis` v2 release. Every test that wants a
    /// well-formed header starts from this and mutates one field.
    fn add_valid_header(b: &mut GgufBuilder) {
        b.add_string(KEY_ARCH, EXPECTED_ARCH);
        b.add_string(KEY_MODEL, "hey_jarvis");
        b.add_f32(KEY_THRESHOLD, 0.5);
        b.add_u32(KEY_SAMPLE_RATE, 16_000);
        b.add_u32(KEY_HOP_MS, 10);
        b.add_u32(KEY_WINDOW_MS, 32);
        b.add_u32(KEY_N_MELS, 40);
        b.add_u32(KEY_FEATURE_DIM, 40);
        b.add_string(KEY_TFLITE_SHA256, "0123456789abcdef");
        b.add_string(KEY_UPSTREAM, "https://example.com/hey_jarvis.tflite");
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
}
