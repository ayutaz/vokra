//! A minimal, dependency-free ONNX (protobuf) weight reader.
//!
//! This decodes just enough of the ONNX wire format to pull weight tensors out
//! of a model — no graph execution, no code generation, and (per the red line
//! in FR-LD-05 / NFR-DS-02) **no protobuf or ONNX crate**. Only the fields
//! Vokra needs are interpreted; everything else is skipped by wire type.
//!
//! # Where weights live
//!
//! Weights may be stored either as `graph.initializer` tensors or — as in
//! Silero VAD v5 — as the `value` attribute of `Constant` nodes buried inside
//! subgraphs (e.g. the `then_branch` / `else_branch` of a top-level `If`).
//! [`read_weight_tensors`] therefore walks the graph recursively, collecting
//! both. Graph *structure* (node topology, control flow) is deliberately not
//! reconstructed here — that is M0-05's job (the 1:1 subgraph, FR-LD-06); this
//! tool only extracts the named weight payloads.
//!
//! Field numbers (source: onnx/onnx `onnx.proto`):
//! - `ModelProto.graph = 7`
//! - `GraphProto`: `node = 1`, `initializer = 5`
//! - `NodeProto`: `output = 2`, `op_type = 4`, `attribute = 5`
//! - `AttributeProto`: `name = 1`, `t = 5` (tensor), `g = 6` (graph),
//!   `graphs = 11` (repeated graph)
//! - `TensorProto`: `dims = 1`, `data_type = 2`, `float_data = 4`,
//!   `name = 8`, `raw_data = 9`, `data_location = 14`
//! - `TensorProto.DataType`: `FLOAT = 1`, `FLOAT16 = 10`
//! - `TensorProto.DataLocation`: `DEFAULT = 0`, `EXTERNAL = 1`

use std::fmt;

/// ONNX `TensorProto.DataType` value for 32-bit float.
pub(crate) const ONNX_DTYPE_FLOAT: i32 = 1;
/// ONNX `TensorProto.DataType` value for 16-bit float.
pub(crate) const ONNX_DTYPE_FLOAT16: i32 = 10;

const WIRE_VARINT: u8 = 0;
const WIRE_I64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_I32: u8 = 5;

/// Guard against pathological subgraph nesting (NFR-RL-07). Real models nest a
/// couple of levels (If / Loop / Scan branches).
const MAX_GRAPH_DEPTH: usize = 32;

/// Error while decoding an ONNX protobuf buffer.
#[derive(Debug)]
pub(crate) enum OnnxError {
    /// The buffer ended mid-field.
    Truncated,
    /// A varint ran longer than 10 bytes (invalid).
    VarintOverflow,
    /// An unknown/unsupported wire type (groups) was encountered.
    BadWireType(u8),
    /// A tensor used external data, which the offline tool does not resolve.
    ExternalData(String),
    /// Subgraphs nested beyond [`MAX_GRAPH_DEPTH`].
    GraphTooDeep,
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "ONNX buffer truncated"),
            Self::VarintOverflow => write!(f, "ONNX varint overflow"),
            Self::BadWireType(w) => write!(f, "ONNX unsupported wire type {w}"),
            Self::ExternalData(name) => {
                write!(f, "ONNX tensor `{name}` uses external data (unsupported)")
            }
            Self::GraphTooDeep => write!(f, "ONNX subgraphs nested too deep"),
        }
    }
}

impl std::error::Error for OnnxError {}

/// A weight tensor extracted from a graph (initializer or `Constant` value).
#[derive(Debug, Clone)]
pub(crate) struct OnnxInitializer {
    /// Tensor name (initializer name, or the `Constant` node's output name).
    pub(crate) name: String,
    /// Shape (from `dims`).
    pub(crate) dims: Vec<u64>,
    /// ONNX `data_type` (see [`ONNX_DTYPE_FLOAT`] / [`ONNX_DTYPE_FLOAT16`]).
    pub(crate) data_type: i32,
    /// Payload bytes in little-endian order (from `raw_data`, or `float_data`
    /// serialized to little-endian floats).
    pub(crate) raw_le_bytes: Vec<u8>,
}

/// Decodes a `ModelProto` and returns every weight tensor it can find,
/// recursing into subgraphs and `Constant` nodes.
pub(crate) fn read_weight_tensors(buf: &[u8]) -> Result<Vec<OnnxInitializer>, OnnxError> {
    let mut out = Vec::new();
    let mut model = Reader::new(buf);
    while let Some((field, wire)) = model.read_tag()? {
        if field == 7 && wire == WIRE_LEN {
            let graph = model.read_len_delimited()?;
            collect_graph(graph, &mut out, 0)?;
        } else {
            model.skip(wire)?;
        }
    }
    Ok(out)
}

/// Recursively collects initializers and `Constant` values from a `GraphProto`.
fn collect_graph(
    buf: &[u8],
    out: &mut Vec<OnnxInitializer>,
    depth: usize,
) -> Result<(), OnnxError> {
    if depth > MAX_GRAPH_DEPTH {
        return Err(OnnxError::GraphTooDeep);
    }
    let mut r = Reader::new(buf);
    while let Some((field, wire)) = r.read_tag()? {
        match (field, wire) {
            (5, WIRE_LEN) => out.push(read_tensor(r.read_len_delimited()?)?),
            (1, WIRE_LEN) => collect_node(r.read_len_delimited()?, out, depth)?,
            _ => r.skip(wire)?,
        }
    }
    Ok(())
}

/// Extracts a `Constant` node's `value` tensor and recurses into any subgraph
/// attributes (e.g. an `If`'s branches).
fn collect_node(buf: &[u8], out: &mut Vec<OnnxInitializer>, depth: usize) -> Result<(), OnnxError> {
    let mut r = Reader::new(buf);
    let mut op_type: Vec<u8> = Vec::new();
    let mut first_output: Option<String> = None;
    let mut value_tensor: Option<Vec<u8>> = None;
    let mut subgraphs: Vec<Vec<u8>> = Vec::new();

    while let Some((field, wire)) = r.read_tag()? {
        match (field, wire) {
            (2, WIRE_LEN) => {
                let o = r.read_len_delimited()?;
                if first_output.is_none() {
                    first_output = Some(String::from_utf8_lossy(o).into_owned());
                }
            }
            (4, WIRE_LEN) => op_type = r.read_len_delimited()?.to_vec(),
            (5, WIRE_LEN) => {
                let attr = parse_attribute(r.read_len_delimited()?)?;
                if let Some(t) = attr.tensor {
                    value_tensor = Some(t);
                }
                subgraphs.extend(attr.subgraphs);
            }
            _ => r.skip(wire)?,
        }
    }

    for g in subgraphs {
        collect_graph(&g, out, depth + 1)?;
    }

    // Only `Constant` nodes carry a weight in a tensor `value` attribute.
    if op_type == b"Constant" {
        if let Some(bytes) = value_tensor {
            let mut init = read_tensor(&bytes)?;
            if init.name.is_empty() {
                if let Some(name) = first_output {
                    init.name = name;
                }
            }
            out.push(init);
        }
    }
    Ok(())
}

/// The parts of an `AttributeProto` this tool cares about.
struct Attribute {
    tensor: Option<Vec<u8>>,
    subgraphs: Vec<Vec<u8>>,
}

/// Decodes an `AttributeProto`, keeping any tensor value and subgraph(s).
fn parse_attribute(buf: &[u8]) -> Result<Attribute, OnnxError> {
    let mut r = Reader::new(buf);
    let mut tensor = None;
    let mut subgraphs = Vec::new();
    while let Some((field, wire)) = r.read_tag()? {
        match (field, wire) {
            (5, WIRE_LEN) => tensor = Some(r.read_len_delimited()?.to_vec()),
            (6, WIRE_LEN) => subgraphs.push(r.read_len_delimited()?.to_vec()),
            (11, WIRE_LEN) => subgraphs.push(r.read_len_delimited()?.to_vec()),
            _ => r.skip(wire)?,
        }
    }
    Ok(Attribute { tensor, subgraphs })
}

/// Decodes a single `TensorProto`.
fn read_tensor(buf: &[u8]) -> Result<OnnxInitializer, OnnxError> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type: i32 = 0;
    let mut raw_data: Option<Vec<u8>> = None;
    let mut float_bytes: Vec<u8> = Vec::new();
    let mut external = false;

    while let Some((field, wire)) = r.read_tag()? {
        match (field, wire) {
            // dims: packed (LEN) or unpacked (VARINT) repeated int64.
            (1, WIRE_LEN) => {
                let packed = r.read_len_delimited()?;
                let mut pr = Reader::new(packed);
                while pr.remaining() > 0 {
                    dims.push(pr.read_varint()?);
                }
            }
            (1, WIRE_VARINT) => dims.push(r.read_varint()?),
            // data_type: int32.
            (2, WIRE_VARINT) => data_type = r.read_varint()? as i32,
            // float_data: packed (LEN, raw LE f32 bytes) or unpacked (I32).
            (4, WIRE_LEN) => float_bytes.extend_from_slice(r.read_len_delimited()?),
            (4, WIRE_I32) => float_bytes.extend_from_slice(&r.read_fixed32()?),
            // name: string.
            (8, WIRE_LEN) => {
                name = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            // raw_data: bytes (already little-endian per the ONNX spec).
            (9, WIRE_LEN) => raw_data = Some(r.read_len_delimited()?.to_vec()),
            // data_location: 1 == EXTERNAL.
            (14, WIRE_VARINT) => external = r.read_varint()? == 1,
            _ => r.skip(wire)?,
        }
    }

    if external {
        return Err(OnnxError::ExternalData(name));
    }

    let raw_le_bytes = raw_data.unwrap_or(float_bytes);
    Ok(OnnxInitializer {
        name,
        dims,
        data_type,
        raw_le_bytes,
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

    fn read_varint(&mut self) -> Result<u64, OnnxError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let byte = *self.buf.get(self.pos).ok_or(OnnxError::Truncated)?;
            self.pos += 1;
            if shift >= 64 {
                return Err(OnnxError::VarintOverflow);
            }
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    /// Reads a field tag, returning `None` at clean end of buffer.
    fn read_tag(&mut self) -> Result<Option<(u32, u8)>, OnnxError> {
        if self.remaining() == 0 {
            return Ok(None);
        }
        let tag = self.read_varint()?;
        let field = (tag >> 3) as u32;
        let wire = (tag & 0x7) as u8;
        Ok(Some((field, wire)))
    }

    fn read_len_delimited(&mut self) -> Result<&'a [u8], OnnxError> {
        let len = self.read_varint()? as usize;
        if self.remaining() < len {
            return Err(OnnxError::Truncated);
        }
        let slice = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn read_fixed32(&mut self) -> Result<[u8; 4], OnnxError> {
        if self.remaining() < 4 {
            return Err(OnnxError::Truncated);
        }
        let mut out = [0u8; 4];
        out.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(out)
    }

    /// Skips a field of the given wire type.
    fn skip(&mut self, wire: u8) -> Result<(), OnnxError> {
        match wire {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_I64 => {
                if self.remaining() < 8 {
                    return Err(OnnxError::Truncated);
                }
                self.pos += 8;
            }
            WIRE_LEN => {
                self.read_len_delimited()?;
            }
            WIRE_I32 => {
                if self.remaining() < 4 {
                    return Err(OnnxError::Truncated);
                }
                self.pos += 4;
            }
            other => return Err(OnnxError::BadWireType(other)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_varint(out: &mut Vec<u8>, mut v: u64) {
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

    fn write_tag(out: &mut Vec<u8>, field: u32, wire: u8) {
        write_varint(out, (u64::from(field) << 3) | u64::from(wire));
    }

    fn write_len_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        write_tag(out, field, WIRE_LEN);
        write_varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }

    fn write_varint_field(out: &mut Vec<u8>, field: u32, v: u64) {
        write_tag(out, field, WIRE_VARINT);
        write_varint(out, v);
    }

    /// Encodes a TensorProto (optional name, dims packed, data_type, raw_data).
    fn tensor_proto(name: &str, dims: &[u64], data_type: i32, raw: &[u8]) -> Vec<u8> {
        let mut t = Vec::new();
        let mut packed = Vec::new();
        for &d in dims {
            write_varint(&mut packed, d);
        }
        write_len_field(&mut t, 1, &packed);
        write_varint_field(&mut t, 2, data_type as u64);
        if !name.is_empty() {
            write_len_field(&mut t, 8, name.as_bytes());
        }
        write_len_field(&mut t, 9, raw);
        t
    }

    fn graph_proto(nodes: &[Vec<u8>], initializers: &[Vec<u8>]) -> Vec<u8> {
        let mut g = Vec::new();
        for n in nodes {
            write_len_field(&mut g, 1, n);
        }
        for t in initializers {
            write_len_field(&mut g, 5, t);
        }
        g
    }

    fn model_proto(graph: &[u8]) -> Vec<u8> {
        let mut m = Vec::new();
        write_len_field(&mut m, 7, graph);
        m
    }

    #[test]
    fn extracts_float_raw_data_initializer() {
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let t = tensor_proto("encoder.weight", &[2, 2], ONNX_DTYPE_FLOAT, &payload);
        let model = model_proto(&graph_proto(&[], &[t]));

        let inits = read_weight_tensors(&model).expect("decode");
        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].name, "encoder.weight");
        assert_eq!(inits[0].dims, vec![2, 2]);
        assert_eq!(inits[0].data_type, ONNX_DTYPE_FLOAT);
        assert_eq!(inits[0].raw_le_bytes, payload);
    }

    #[test]
    fn extracts_float_data_packed() {
        let mut t = Vec::new();
        let mut packed_dims = Vec::new();
        write_varint(&mut packed_dims, 3);
        write_len_field(&mut t, 1, &packed_dims);
        write_varint_field(&mut t, 2, ONNX_DTYPE_FLOAT as u64);
        write_len_field(&mut t, 8, b"w");
        let floats: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        write_len_field(&mut t, 4, &floats);
        let model = model_proto(&graph_proto(&[], &[t]));

        let inits = read_weight_tensors(&model).expect("decode");
        assert_eq!(inits[0].raw_le_bytes, floats);
    }

    /// Encodes an AttributeProto with a tensor `value`.
    fn attr_tensor(name: &str, tensor: &[u8]) -> Vec<u8> {
        let mut a = Vec::new();
        write_len_field(&mut a, 1, name.as_bytes());
        write_len_field(&mut a, 5, tensor); // t
        a
    }

    /// Encodes an AttributeProto holding a subgraph.
    fn attr_graph(name: &str, graph: &[u8]) -> Vec<u8> {
        let mut a = Vec::new();
        write_len_field(&mut a, 1, name.as_bytes());
        write_len_field(&mut a, 6, graph); // g
        a
    }

    fn node_proto(op_type: &str, output: &str, attrs: &[Vec<u8>]) -> Vec<u8> {
        let mut n = Vec::new();
        write_len_field(&mut n, 2, output.as_bytes()); // output[0]
        write_len_field(&mut n, 4, op_type.as_bytes()); // op_type
        for a in attrs {
            write_len_field(&mut n, 5, a);
        }
        n
    }

    #[test]
    fn extracts_constant_node_value_named_by_output() {
        // A Constant node with an anonymous tensor value; the extracted name
        // must come from the node's output.
        let payload: Vec<u8> = [0.5f32, -0.5]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let value = tensor_proto("", &[2], ONNX_DTYPE_FLOAT, &payload);
        let node = node_proto("Constant", "conv.weight", &[attr_tensor("value", &value)]);
        let model = model_proto(&graph_proto(&[node], &[]));

        let inits = read_weight_tensors(&model).expect("decode");
        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].name, "conv.weight");
        assert_eq!(inits[0].raw_le_bytes, payload);
    }

    #[test]
    fn recurses_into_if_subgraph() {
        // An If node whose then_branch subgraph holds a Constant weight.
        let payload: Vec<u8> = [7.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        let value = tensor_proto("", &[1], ONNX_DTYPE_FLOAT, &payload);
        let inner = node_proto("Constant", "inner.weight", &[attr_tensor("value", &value)]);
        let branch = graph_proto(&[inner], &[]);
        let if_node = node_proto("If", "out", &[attr_graph("then_branch", &branch)]);
        let model = model_proto(&graph_proto(&[if_node], &[]));

        let inits = read_weight_tensors(&model).expect("decode");
        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].name, "inner.weight");
        assert_eq!(inits[0].raw_le_bytes, payload);
    }

    #[test]
    fn skips_unknown_fields() {
        let t = tensor_proto("w", &[1], ONNX_DTYPE_FLOAT16, &[0x00, 0x3C]);
        let mut graph = Vec::new();
        write_len_field(&mut graph, 2, b"graph-name"); // name (skipped)
        write_len_field(&mut graph, 5, &t);
        let mut model = Vec::new();
        write_varint_field(&mut model, 1, 7); // ir_version (skipped)
        write_len_field(&mut model, 7, &graph);

        let inits = read_weight_tensors(&model).expect("decode");
        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].data_type, ONNX_DTYPE_FLOAT16);
        assert_eq!(inits[0].raw_le_bytes, vec![0x00, 0x3C]);
    }

    #[test]
    fn rejects_external_data() {
        let mut t = Vec::new();
        write_len_field(&mut t, 8, b"ext");
        write_varint_field(&mut t, 2, ONNX_DTYPE_FLOAT as u64);
        write_varint_field(&mut t, 14, 1); // data_location = EXTERNAL
        let model = model_proto(&graph_proto(&[], &[t]));
        assert!(matches!(
            read_weight_tensors(&model),
            Err(OnnxError::ExternalData(_))
        ));
    }
}
