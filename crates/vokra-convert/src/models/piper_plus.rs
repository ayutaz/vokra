//! piper-plus (MB-iSTFT-VITS2) voice: ONNX + config.json to GGUF conversion
//! (M0-07-T06/T07).
//!
//! Input: a distributed piper-plus voice — the FP16 ONNX graph plus its
//! `config.json` (phoneme table, sample rate, inference defaults). Output: a
//! GGUF carrying every weight tensor (widened FP16 → FP32) plus the
//! `vokra.piper.*` metadata the native MB-iSTFT-VITS2 implementation
//! (`vokra-models`, M0-07-T11..T20) loads against. No ONNX is ever handled at
//! runtime (FR-LD-05); this offline tool is the only place it is touched.
//!
//! # Weight naming: undoing the weight-norm export obfuscation
//!
//! piper-plus voices are exported after `remove_weight_norm()`, and the ONNX
//! tracer renames the folded convolution weights of the flow WN blocks and the
//! PQMF / prosody / speaker buffers to generic `onnx::Conv_9261`-style names
//! (the biases keep their module names). Loading against those opaque names
//! would be fragile, so the converter rebuilds the clean module names by
//! walking the graph: for every `Conv` / `ConvTranspose` node whose bias input
//! is a named `*.bias`, the paired weight input is renamed `*.weight`. The
//! handful of remaining buffers (PQMF filters, `prosody_proj`, `spk_proj`, the
//! `dp.flows.0` affine `logs`) are mapped by their unique shapes. The GGUF the
//! runtime loads therefore has stable, self-describing tensor names, exactly as
//! the Whisper converter gives M0-06 the upstream names verbatim.
//!
//! Graph inputs cast every FP16 initializer to FP32 (`*_fp32` Cast outputs)
//! before each op, so the reference runs in FP32 with FP16-rounded weights —
//! which is why widening to FP32 here and computing in FP32 natively can meet
//! the FP32 parity bound (NFR-QL-01 atol = 0.01).
//!
//! # FP16 → FP32
//!
//! Every float tensor is widened to FP32 in the GGUF so the native runtime
//! loads a single dtype and computes in FP32. This roughly doubles the file
//! size versus the FP16 ONNX, which is acceptable for M0 (no size gate).

use std::collections::HashMap;

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::json::{self, JsonValue};

/// `vokra.model.arch` value written for piper-plus voice GGUFs.
pub(crate) const ARCH: &str = "piper-plus-mb-istft-vits2";

// --- vokra.piper.* metadata keys (M0-07-T06 chunk design) -------------------

/// `vokra.model.name` — human-readable voice name (`STRING`).
const KEY_MODEL_NAME: &str = "vokra.model.name";
/// `vokra.piper.sample_rate` — output PCM sample rate, Hz (`UINT32`).
const KEY_SAMPLE_RATE: &str = "vokra.piper.sample_rate";
/// `vokra.piper.num_symbols` — phoneme embedding table size (`UINT32`).
const KEY_NUM_SYMBOLS: &str = "vokra.piper.num_symbols";
/// `vokra.piper.num_languages` — language embedding table size (`UINT32`).
const KEY_NUM_LANGUAGES: &str = "vokra.piper.num_languages";
/// `vokra.piper.noise_scale` — default z_p noise scale (`FLOAT32`).
const KEY_NOISE_SCALE: &str = "vokra.piper.noise_scale";
/// `vokra.piper.length_scale` — default duration length scale (`FLOAT32`).
const KEY_LENGTH_SCALE: &str = "vokra.piper.length_scale";
/// `vokra.piper.noise_w` — default stochastic-duration noise scale (`FLOAT32`).
const KEY_NOISE_W: &str = "vokra.piper.noise_w";
/// `vokra.piper.istft.n_fft` — decoder iSTFT FFT size (`UINT32`).
const KEY_ISTFT_N_FFT: &str = "vokra.piper.istft.n_fft";
/// `vokra.piper.istft.hop` — decoder iSTFT hop length (`UINT32`).
const KEY_ISTFT_HOP: &str = "vokra.piper.istft.hop";
/// `vokra.piper.pqmf.subbands` — PQMF sub-band count (`UINT32`).
const KEY_PQMF_SUBBANDS: &str = "vokra.piper.pqmf.subbands";
/// `vokra.piper.phoneme_symbols` — phoneme string per id (`ARRAY<STRING>`).
const KEY_PHONEME_SYMBOLS: &str = "vokra.piper.phoneme_symbols";
/// `vokra.piper.language_codes` — language code per id (`ARRAY<STRING>`).
const KEY_LANGUAGE_CODES: &str = "vokra.piper.language_codes";

/// iSTFT / PQMF hyper-parameters, fixed by the MB-iSTFT-VITS2 medium
/// architecture (piper-plus `mb_istft.py`: `n_fft=16`, `hop_length=4`,
/// `subbands=4`). Stored so the runtime never hard-codes them.
const ISTFT_N_FFT: u32 = 16;
const ISTFT_HOP: u32 = 4;
const PQMF_SUBBANDS: u32 = 4;

/// Outcome of a piper-plus voice conversion.
#[derive(Debug, Default)]
pub(crate) struct PiperPlusReport {
    /// Float weight tensors written to the GGUF.
    pub(crate) written: usize,
    /// `onnx::*` weights whose clean module name was recovered by graph trace.
    pub(crate) renamed: usize,
    /// Non-float initializers skipped (int64 shape/index constants).
    pub(crate) skipped_non_float: usize,
    /// Phoneme-map ids that exceed `num_symbols` (out of embedding range —
    /// M0-07 §8 A-4: kept in the symbol table but flagged; the runtime rejects
    /// them at tokenise time).
    pub(crate) phoneme_ids_over_range: usize,
}

/// Converts a piper-plus voice (`onnx_bytes` + `config_bytes`) into a populated
/// GGUF builder plus a report of what was written and recovered.
pub(crate) fn convert(
    onnx_bytes: &[u8],
    config_bytes: &[u8],
) -> Result<(GgufBuilder, PiperPlusReport), ConvertError> {
    let graph = Graph::parse(onnx_bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
    let rename = build_rename_map(&graph);
    let config = Config::parse(config_bytes)?;

    let mut b = GgufBuilder::new();
    let mut report = PiperPlusReport::default();

    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(KEY_MODEL_NAME, &config.name);
    b.add_u32(KEY_SAMPLE_RATE, config.sample_rate);
    b.add_u32(KEY_NUM_SYMBOLS, config.num_symbols);
    b.add_u32(KEY_NUM_LANGUAGES, config.num_languages);
    b.add_f32(KEY_NOISE_SCALE, config.noise_scale);
    b.add_f32(KEY_LENGTH_SCALE, config.length_scale);
    b.add_f32(KEY_NOISE_W, config.noise_w);
    b.add_u32(KEY_ISTFT_N_FFT, ISTFT_N_FFT);
    b.add_u32(KEY_ISTFT_HOP, ISTFT_HOP);
    b.add_u32(KEY_PQMF_SUBBANDS, PQMF_SUBBANDS);
    add_string_array(&mut b, KEY_PHONEME_SYMBOLS, &config.phoneme_symbols);
    add_string_array(&mut b, KEY_LANGUAGE_CODES, &config.language_codes);
    report.phoneme_ids_over_range = config.phoneme_ids_over_range;

    for t in &graph.initializers {
        // Skip non-float initializers (int64 shape/index constants).
        let dtype = match t.data_type {
            ONNX_FLOAT => GgmlType::F32,
            ONNX_FLOAT16 => GgmlType::F32, // widened below
            _ => {
                report.skipped_non_float += 1;
                continue;
            }
        };
        let name = match rename.get(&t.name) {
            Some(clean) => {
                report.renamed += 1;
                clean.clone()
            }
            None => t.name.clone(),
        };
        let data = match t.data_type {
            ONNX_FLOAT => t.raw.clone(),
            ONNX_FLOAT16 => widen_f16_to_f32(&t.raw),
            _ => unreachable!(),
        };
        b.add_tensor(&name, dtype, t.dims.clone(), data)?;
        report.written += 1;
    }

    Ok((b, report))
}

/// Adds a homogeneous `ARRAY<STRING>` metadata value.
fn add_string_array(b: &mut GgufBuilder, key: &str, values: &[String]) {
    let values = values
        .iter()
        .map(|s| GgufMetadataValue::String(s.clone()))
        .collect();
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
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

/// Converts an IEEE-754 half-precision bit pattern to `f32`.
///
/// Handles subnormals, infinities and NaN; a pure-integer implementation so no
/// external crate is pulled into the offline tool.
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = u32::from(bits & 0x03FF);
    let out = match exp {
        0 => {
            if mant == 0 {
                sign // +/- zero
            } else {
                // Subnormal half = mant · 2^-24. Shift the leading 1 up to the
                // hidden-bit position (bit 10), counting `k` shifts; the f32
                // biased exponent is then `113 - k` (= -14 - k + 127).
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
        0x1F => sign | 0x7F80_0000 | (mant << 13), // Inf / NaN
        _ => {
            let exp32 = (i32::from(exp) - 15 + 127) as u32;
            sign | (exp32 << 23) | (mant << 13)
        }
    };
    f32::from_bits(out)
}

// ===========================================================================
// config.json
// ===========================================================================

/// The parts of a piper-plus `config.json` the converter needs.
struct Config {
    name: String,
    sample_rate: u32,
    num_symbols: u32,
    num_languages: u32,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    /// Phoneme string indexed by id (`phoneme_symbols[id]`), length `max_id+1`.
    phoneme_symbols: Vec<String>,
    /// Language code indexed by id, length `num_languages`.
    language_codes: Vec<String>,
    phoneme_ids_over_range: usize,
}

impl Config {
    fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
        let get_f32 = |v: &JsonValue| -> Option<f32> {
            match v {
                JsonValue::Int(i) => Some(*i as f32),
                JsonValue::Float(f) => Some(*f as f32),
                _ => None,
            }
        };

        let num_symbols = root
            .get("num_symbols")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| ConvertError::Parse("config: missing num_symbols".to_owned()))?
            as u32;
        let sample_rate = root
            .get("audio")
            .and_then(|a| a.get("sample_rate"))
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| ConvertError::Parse("config: missing audio.sample_rate".to_owned()))?
            as u32;
        let num_languages = root
            .get("num_languages")
            .and_then(JsonValue::as_u64)
            .unwrap_or(1) as u32;

        let inference = root.get("inference");
        let noise_scale = inference
            .and_then(|i| i.get("noise_scale"))
            .and_then(&get_f32)
            .unwrap_or(0.667);
        let length_scale = inference
            .and_then(|i| i.get("length_scale"))
            .and_then(&get_f32)
            .unwrap_or(1.0);
        let noise_w = inference
            .and_then(|i| i.get("noise_w").or_else(|| i.get("noise_scale_w")))
            .and_then(&get_f32)
            .unwrap_or(0.8);

        let name = root
            .get("dataset")
            .and_then(JsonValue::as_str)
            .map(|d| format!("piper-plus-{d}"))
            .unwrap_or_else(|| "piper-plus".to_owned());

        // Build the id → symbol table from `phoneme_id_map` (symbol → [id]).
        let mut phoneme_symbols: Vec<String> = Vec::new();
        let mut over_range = 0usize;
        if let Some(map) = root.get("phoneme_id_map").and_then(JsonValue::as_object) {
            for (symbol, ids) in map {
                if let Some(arr) = ids.as_array() {
                    for id in arr {
                        if let Some(id) = id.as_u64() {
                            let id = id as usize;
                            if id >= num_symbols as usize {
                                over_range += 1;
                            }
                            if id >= phoneme_symbols.len() {
                                phoneme_symbols.resize(id + 1, String::new());
                            }
                            phoneme_symbols[id] = symbol.clone();
                        }
                    }
                }
            }
        }

        // Language codes indexed by id (from `language_id_map`: code → id).
        let mut language_codes: Vec<String> = vec![String::new(); num_languages as usize];
        if let Some(map) = root.get("language_id_map").and_then(JsonValue::as_object) {
            for (code, id) in map {
                if let Some(id) = id.as_u64() {
                    let id = id as usize;
                    if id < language_codes.len() {
                        language_codes[id] = code.clone();
                    }
                }
            }
        }

        Ok(Self {
            name,
            sample_rate,
            num_symbols,
            num_languages,
            noise_scale,
            length_scale,
            noise_w,
            phoneme_symbols,
            language_codes,
            phoneme_ids_over_range: over_range,
        })
    }
}

// ===========================================================================
// onnx:: weight-name recovery (graph trace)
// ===========================================================================

/// Builds the `onnx::*` → clean-module-name rename map (see module docs).
fn build_rename_map(graph: &Graph) -> HashMap<String, String> {
    let mut rename = HashMap::new();

    // 1. Conv / ConvTranspose weight recovered via the paired named bias.
    for node in &graph.nodes {
        if node.op_type != "Conv" && node.op_type != "ConvTranspose" {
            continue;
        }
        let (Some(w), Some(bias)) = (node.inputs.get(1), node.inputs.get(2)) else {
            continue;
        };
        let w = strip_fp32(w);
        let bias = strip_fp32(bias);
        if w.starts_with("onnx::") && bias.ends_with(".bias") {
            let clean = format!("{}.weight", &bias[..bias.len() - ".bias".len()]);
            rename.insert(w.to_owned(), clean);
        }
    }

    // 2. Remaining `onnx::*` buffers mapped by their unique shapes: PQMF
    //    filters, prosody / speaker projections and the `dp.flows.0` affine
    //    `logs` (its `m` counterpart keeps a clean name).
    for t in &graph.initializers {
        if !t.name.starts_with("onnx::") || rename.contains_key(&t.name) {
            continue;
        }
        let clean = match t.dims.as_slice() {
            [3, 16] => "prosody_proj.weight",
            [4, 1, 4] => "dec.pqmf.updown_filter",
            [1, 4, 63] => "dec.pqmf.synthesis_filter",
            [2, 1] => "dp.flows.0.logs",
            [256, 512] => "spk_proj.weight",
            _ => continue,
        };
        rename.insert(t.name.clone(), clean.to_owned());
    }

    rename
}

/// Strips the `_fp32` Cast-output suffix the ONNX exporter appends.
fn strip_fp32(name: &str) -> &str {
    name.strip_suffix("_fp32").unwrap_or(name)
}

// ===========================================================================
// Minimal ONNX protobuf reader (initializers + node topology only)
// ===========================================================================
//
// A self-contained decoder that pulls exactly what the converter needs — the
// `graph.initializer` tensors and the `Conv`/`ConvTranspose` node wiring — in a
// single pass. It is deliberately kept here (not in `crate::onnx`, which
// collects `Constant` values for Silero and does not expose node topology): the
// piper-plus weights live in `graph.initializer`, so folding in the 2694
// `Constant` shape nodes would be pure noise. No protobuf crate is used
// (FR-LD-05 / NFR-DS-02). Field numbers from onnx/onnx `onnx.proto`.

/// ONNX `TensorProto.DataType`: 32-bit float.
const ONNX_FLOAT: i32 = 1;
/// ONNX `TensorProto.DataType`: 16-bit float.
const ONNX_FLOAT16: i32 = 10;

const WIRE_VARINT: u8 = 0;
const WIRE_I64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_I32: u8 = 5;

/// An initializer tensor pulled from `graph.initializer`.
struct RawTensor {
    name: String,
    dims: Vec<u64>,
    data_type: i32,
    raw: Vec<u8>,
}

/// A `Conv` / `ConvTranspose` node's op type and input names (for weight-name
/// recovery). Other node kinds are recorded with empty inputs.
struct NodeInfo {
    op_type: String,
    inputs: Vec<String>,
}

/// The decoded parts of a piper-plus ONNX `ModelProto`.
struct Graph {
    initializers: Vec<RawTensor>,
    nodes: Vec<NodeInfo>,
}

/// A protobuf decoding error.
#[derive(Debug)]
enum PbError {
    Truncated,
    VarintOverflow,
    BadWireType(u8),
}

impl std::fmt::Display for PbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "piper-plus ONNX buffer truncated"),
            Self::VarintOverflow => write!(f, "piper-plus ONNX varint overflow"),
            Self::BadWireType(w) => write!(f, "piper-plus ONNX unsupported wire type {w}"),
        }
    }
}

impl Graph {
    /// Decodes a `ModelProto`, keeping its `graph.initializer` tensors and node
    /// wiring.
    fn parse(buf: &[u8]) -> Result<Self, PbError> {
        let mut initializers = Vec::new();
        let mut nodes = Vec::new();
        let mut model = Reader::new(buf);
        while let Some((field, wire)) = model.read_tag()? {
            // ModelProto.graph = 7
            if field == 7 && wire == WIRE_LEN {
                let graph = model.read_len()?;
                Self::parse_graph(graph, &mut initializers, &mut nodes)?;
            } else {
                model.skip(wire)?;
            }
        }
        Ok(Self {
            initializers,
            nodes,
        })
    }

    fn parse_graph(
        buf: &[u8],
        inits: &mut Vec<RawTensor>,
        nodes: &mut Vec<NodeInfo>,
    ) -> Result<(), PbError> {
        let mut r = Reader::new(buf);
        while let Some((field, wire)) = r.read_tag()? {
            match (field, wire) {
                // GraphProto.node = 1
                (1, WIRE_LEN) => nodes.push(parse_node(r.read_len()?)?),
                // GraphProto.initializer = 5
                (5, WIRE_LEN) => inits.push(parse_tensor(r.read_len()?)?),
                _ => r.skip(wire)?,
            }
        }
        Ok(())
    }
}

/// Decodes a `NodeProto`, keeping the op type and input names.
fn parse_node(buf: &[u8]) -> Result<NodeInfo, PbError> {
    let mut r = Reader::new(buf);
    let mut op_type = String::new();
    let mut inputs = Vec::new();
    while let Some((field, wire)) = r.read_tag()? {
        match (field, wire) {
            // NodeProto.input = 1 (repeated string)
            (1, WIRE_LEN) => inputs.push(String::from_utf8_lossy(r.read_len()?).into_owned()),
            // NodeProto.op_type = 4
            (4, WIRE_LEN) => op_type = String::from_utf8_lossy(r.read_len()?).into_owned(),
            _ => r.skip(wire)?,
        }
    }
    // Only conv nodes drive weight recovery; drop the inputs of others to keep
    // the trace small.
    if op_type != "Conv" && op_type != "ConvTranspose" {
        inputs.clear();
    }
    Ok(NodeInfo { op_type, inputs })
}

/// Decodes a `TensorProto` (name, dims, data_type, raw_data / float_data).
fn parse_tensor(buf: &[u8]) -> Result<RawTensor, PbError> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0i32;
    let mut raw_data: Option<Vec<u8>> = None;
    let mut float_bytes: Vec<u8> = Vec::new();
    while let Some((field, wire)) = r.read_tag()? {
        match (field, wire) {
            (1, WIRE_LEN) => {
                let packed = r.read_len()?;
                let mut pr = Reader::new(packed);
                while pr.remaining() > 0 {
                    dims.push(pr.read_varint()?);
                }
            }
            (1, WIRE_VARINT) => dims.push(r.read_varint()?),
            (2, WIRE_VARINT) => data_type = r.read_varint()? as i32,
            (4, WIRE_LEN) => float_bytes.extend_from_slice(r.read_len()?),
            (4, WIRE_I32) => float_bytes.extend_from_slice(&r.read_fixed32()?),
            (8, WIRE_LEN) => name = String::from_utf8_lossy(r.read_len()?).into_owned(),
            (9, WIRE_LEN) => raw_data = Some(r.read_len()?.to_vec()),
            _ => r.skip(wire)?,
        }
    }
    Ok(RawTensor {
        name,
        dims,
        data_type,
        raw: raw_data.unwrap_or(float_bytes),
    })
}

/// A bounds-checked protobuf cursor.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn read_varint(&mut self) -> Result<u64, PbError> {
        let mut result = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self.buf.get(self.pos).ok_or(PbError::Truncated)?;
            self.pos += 1;
            if shift >= 64 {
                return Err(PbError::VarintOverflow);
            }
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    fn read_tag(&mut self) -> Result<Option<(u32, u8)>, PbError> {
        if self.remaining() == 0 {
            return Ok(None);
        }
        let tag = self.read_varint()?;
        Ok(Some(((tag >> 3) as u32, (tag & 0x7) as u8)))
    }

    fn read_len(&mut self) -> Result<&'a [u8], PbError> {
        let len = self.read_varint()? as usize;
        if self.remaining() < len {
            return Err(PbError::Truncated);
        }
        let slice = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn read_fixed32(&mut self) -> Result<[u8; 4], PbError> {
        if self.remaining() < 4 {
            return Err(PbError::Truncated);
        }
        let mut out = [0u8; 4];
        out.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(out)
    }

    fn skip(&mut self, wire: u8) -> Result<(), PbError> {
        match wire {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_I64 => {
                if self.remaining() < 8 {
                    return Err(PbError::Truncated);
                }
                self.pos += 8;
            }
            WIRE_LEN => {
                self.read_len()?;
            }
            WIRE_I32 => {
                if self.remaining() < 4 {
                    return Err(PbError::Truncated);
                }
                self.pos += 4;
            }
            other => return Err(PbError::BadWireType(other)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    #[test]
    fn half_to_f32_matches_known_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        // Largest normal half = 65504.
        assert_eq!(half_to_f32(0x7BFF), 65504.0);
        // A subnormal half (smallest positive) = 2^-24.
        assert!((half_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-30);
    }

    #[test]
    fn half_to_f32_handles_inf_nan_and_signed_zero() {
        // exp==0x1F, mant==0 -> +/-infinity.
        assert_eq!(half_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(half_to_f32(0xFC00), f32::NEG_INFINITY);
        // exp==0x1F, mant!=0 -> NaN (both signs).
        assert!(half_to_f32(0x7E00).is_nan());
        assert!(half_to_f32(0xFE00).is_nan());
        // 0x8000 is negative zero: == 0.0 compares equal, so pin the sign bit.
        let neg_zero = half_to_f32(0x8000);
        assert_eq!(neg_zero, 0.0);
        assert!(neg_zero.is_sign_negative());
    }

    #[test]
    fn config_missing_required_fields_errors() {
        let onnx = model(&[], &[]);
        // Missing num_symbols.
        assert!(matches!(
            convert(&onnx, br#"{"audio": {"sample_rate": 22050}}"#),
            Err(ConvertError::Parse(_))
        ));
        // Missing audio.sample_rate.
        assert!(matches!(
            convert(&onnx, br#"{"num_symbols": 3}"#),
            Err(ConvertError::Parse(_))
        ));
    }

    #[test]
    fn config_noise_scale_w_alias_populates_noise_w() {
        // piper-plus configs may spell the stochastic-duration noise scale
        // `noise_scale_w`; the converter must read it into vokra.piper.noise_w.
        let onnx = model(&[], &[]);
        let cfg = br#"{"audio": {"sample_rate": 22050}, "num_symbols": 3, "inference": {"noise_scale_w": 0.5}}"#;
        let (builder, _report) = convert(&onnx, cfg).unwrap();
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(file.get(KEY_NOISE_W), Some(&GgufMetadataValue::F32(0.5)));
    }

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
    fn node(op_type: &str, inputs: &[&str]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            len_field(&mut n, 1, i.as_bytes());
        }
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

    /// A minimal config.json with two phonemes and one language.
    const CONFIG: &str = r#"{
        "dataset": "unit",
        "audio": {"sample_rate": 22050},
        "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
        "num_symbols": 3,
        "num_languages": 2,
        "phoneme_id_map": {"_": [0], "a": [1], "b": [2], "over": [5]},
        "language_id_map": {"ja": 0, "en": 1}
    }"#;

    #[test]
    fn traces_onnx_conv_weight_via_bias_and_widens_fp16() {
        // A Conv node whose weight is an opaque `onnx::Conv_1` and whose bias is
        // the named `flow.flows.0.enc.in_layers.0.bias` (both via `_fp32` Casts).
        let f16_one = 0x3C00u16.to_le_bytes(); // 1.0
        let w = tensor("onnx::Conv_1", &[1], ONNX_FLOAT16, &f16_one);
        let bias = tensor(
            "flow.flows.0.enc.in_layers.0.bias",
            &[1],
            ONNX_FLOAT16,
            &f16_one,
        );
        let conv = node(
            "Conv",
            &[
                "x_fp32",
                "onnx::Conv_1_fp32",
                "flow.flows.0.enc.in_layers.0.bias_fp32",
            ],
        );
        let onnx = model(&[conv], &[w, bias]);

        let (builder, report) = convert(&onnx, CONFIG.as_bytes()).unwrap();
        assert_eq!(report.written, 2);
        assert_eq!(report.renamed, 1);

        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        // The opaque weight got its clean module name; both are F32 now.
        let w = file
            .tensor_info("flow.flows.0.enc.in_layers.0.weight")
            .expect("renamed weight present");
        assert_eq!(w.dtype, GgmlType::F32);
        assert_eq!(
            file.tensor_data("flow.flows.0.enc.in_layers.0.weight")
                .unwrap(),
            1.0f32.to_le_bytes()
        );
        assert!(
            file.tensor_info("flow.flows.0.enc.in_layers.0.bias")
                .is_some()
        );
    }

    #[test]
    fn maps_special_buffers_by_shape() {
        let f16 = 0x3C00u16.to_le_bytes();
        let raw: Vec<u8> = std::iter::repeat_n(f16, 48).flatten().collect(); // 3*16
        let prosody = tensor("onnx::MatMul_9102", &[3, 16], ONNX_FLOAT16, &raw);
        let onnx = model(&[], &[prosody]);
        let (builder, report) = convert(&onnx, CONFIG.as_bytes()).unwrap();
        assert_eq!(report.renamed, 1);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert!(file.tensor_info("prosody_proj.weight").is_some());
    }

    #[test]
    fn writes_metadata_and_phoneme_table() {
        let onnx = model(&[], &[]);
        let (builder, report) = convert(&onnx, CONFIG.as_bytes()).unwrap();
        // id 5 exceeds num_symbols=3.
        assert_eq!(report.phoneme_ids_over_range, 1);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE),
            Some(&GgufMetadataValue::U32(22050))
        );
        assert_eq!(file.get(KEY_NOISE_W), Some(&GgufMetadataValue::F32(0.8)));
        assert_eq!(file.get(KEY_ISTFT_N_FFT), Some(&GgufMetadataValue::U32(16)));
        let syms = file
            .get(KEY_PHONEME_SYMBOLS)
            .and_then(|v| v.as_array())
            .unwrap();
        // Table spans id 0..=5 (max id present).
        assert_eq!(syms.values.len(), 6);
        assert_eq!(syms.values[1].as_str(), Some("a"));
        let langs = file
            .get(KEY_LANGUAGE_CODES)
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(langs.values[0].as_str(), Some("ja"));
        assert_eq!(langs.values[1].as_str(), Some("en"));
    }

    #[test]
    fn skips_non_float_initializers() {
        let int_const = tensor(
            "some.shape",
            &[1],
            7, /* int64 */
            &[3, 0, 0, 0, 0, 0, 0, 0],
        );
        let onnx = model(&[], &[int_const]);
        let (_b, report) = convert(&onnx, CONFIG.as_bytes()).unwrap();
        assert_eq!(report.skipped_non_float, 1);
        assert_eq!(report.written, 0);
    }
}
