//! Silero VAD v5: ONNX checkpoint to GGUF conversion.
//!
//! Input: the upstream `snakers4/silero-vad` `silero_vad.onnx`. Output: a
//! **both-rate** GGUF (`sr8k.*` / `sr16k.*` tensors, the corrected scheme of
//! the silero_vad SPEC) plus `vokra.model.*`.
//!
//! # Where Silero's weights live — and why the branch matters
//!
//! Silero VAD v5 stores no top-level `graph.initializer`s. Its weights are the
//! `value` attributes of `Constant` nodes inside the `then_branch` /
//! `else_branch` subgraphs of a top-level `If(sr == 16000)` — the model is
//! really **two independently-trained networks**, then = 16 kHz, else = 8 kHz,
//! and *all 15* parameters differ in value between them (SPEC "known
//! conversion gap"). The embedded tensor names are the bare PyTorch parameter
//! names, identical in both branches; only each `Constant`'s scope-qualified
//! *output* name (`If_0_then_branch__Inline_0__…` / `If_0_else_branch__…`)
//! says which network a weight belongs to. This converter therefore keys on
//! [`crate::onnx::OnnxInitializer::output_name`] and emits every parameter
//! under its rate namespace. (The original M0 version de-duped the colliding
//! embedded names instead, which silently dropped the entire 16 kHz model —
//! confirmed empirically in the 2026-07-16 real-weight eval.)
//!
//! Non-float constants (the many int64 shape/slice/index constants that drive
//! control flow) fall outside the M0 dtype range (FP32/FP16) and are counted
//! in [`SileroReport::skipped_non_float`], not written. Float constants whose
//! stripped name is an op-scope path (e.g. `…/stft/Constant_22_output_0`) are
//! graph-internal scalars, not parameters of the SPEC weight map; they are
//! counted in [`SileroReport::skipped_stray`]. The GGUF tensor order is
//! name-sorted, matching `tests/parity/silero_vad/gen_reference.py`, so the
//! same ONNX produces a byte-comparable GGUF.
//!
//! # No `vokra.frontend.*` chunk (M0-03-T08 decision)
//!
//! Silero's pseudo-STFT front-end is an implementation detail hidden inside the
//! 1:1 subgraph (FR-LD-06, M0-05), not a Vokra-controlled feature extractor, so
//! no `frontend_spec` is written.
//!
//! # Scope note
//!
//! This extracts named weight *payloads* only. Reconstructing Silero's graph
//! (control flow, which weight feeds which op) is M0-05's 1:1-subgraph job; the
//! tensor names here are the contract M0-05 loads against.

use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::onnx::{self, ONNX_DTYPE_FLOAT, ONNX_DTYPE_FLOAT16};

/// `vokra.model.arch` value written for Silero VAD GGUFs.
pub(crate) const ARCH: &str = "silero-vad";
/// `vokra.model.name` value written for the Silero VAD v5 GGUF.
pub(crate) const NAME: &str = "silero-vad-v5";

/// The 16 kHz branch prefix on `Constant` output names: the `If` selector is
/// `sr == 16000`, so `then` = 16 kHz (verified against the graph's compare
/// constant; silero_vad SPEC "Exact details").
const THEN_PREFIX: &str = "If_0_then_branch__Inline_0__";
/// The 8 kHz (`else`) branch prefix.
const ELSE_PREFIX: &str = "If_0_else_branch__Inline_0__";

/// Outcome of a Silero conversion.
#[derive(Debug, Default)]
pub(crate) struct SileroReport {
    /// Number of float weight tensors written to the GGUF (15 per rate for
    /// upstream v5).
    pub(crate) written: usize,
    /// Constants skipped because their dtype is outside M0's FP32/FP16 range
    /// (mostly int64 shape/index constants).
    pub(crate) skipped_non_float: usize,
    /// Float constants skipped because they are graph-internal op outputs
    /// (op-scope names like `…/stft/Constant_22_output_0`), not parameters.
    pub(crate) skipped_stray: usize,
}

/// Converts a Silero VAD ONNX buffer into a populated GGUF builder plus a
/// report of what was written vs. skipped.
///
/// Errors with [`ConvertError::Parse`] if no `If`-branch weights are found
/// (not the documented upstream layout — refusing beats emitting a GGUF that
/// cannot serve either rate, FR-EX-08), and propagates the GGUF writer's
/// duplicate-name error rather than de-duping.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, SileroReport), ConvertError> {
    let tensors = onnx::read_weight_tensors(&bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);

    let mut report = SileroReport::default();
    let mut named: Vec<(String, GgmlType, Vec<u64>, Vec<u8>)> = Vec::new();
    for init in tensors {
        let dtype = match init.data_type {
            ONNX_DTYPE_FLOAT => GgmlType::F32,
            ONNX_DTYPE_FLOAT16 => GgmlType::F16,
            _ => {
                report.skipped_non_float += 1;
                continue;
            }
        };
        // The branch (= sample rate) is only visible on the scope-qualified
        // output name; the embedded name is the bare, branch-ambiguous
        // parameter name.
        let scoped = init.output_name.as_deref().unwrap_or(&init.name);
        let (tag, param) = if let Some(p) = scoped.strip_prefix(THEN_PREFIX) {
            ("sr16k", p)
        } else if let Some(p) = scoped.strip_prefix(ELSE_PREFIX) {
            ("sr8k", p)
        } else {
            // Upstream v5 keeps every weight inside the two If branches.
            report.skipped_stray += 1;
            continue;
        };
        if param.starts_with('/') {
            // Op-scope float scalar (graph constant), not a model parameter.
            report.skipped_stray += 1;
            continue;
        }
        named.push((
            format!("{tag}.{param}"),
            dtype,
            init.dims,
            init.raw_le_bytes,
        ));
    }

    if named.is_empty() {
        return Err(ConvertError::Parse(
            "no Silero `If`-branch weights found: expected the upstream v5 layout \
             (all parameters as Constants inside the If(sr == 16000) then/else \
             subgraphs); refusing to emit an empty GGUF"
                .to_owned(),
        ));
    }

    // Deterministic name-sorted tensor order (as gen_reference.py writes the
    // parity fixture), so identical inputs yield byte-comparable GGUFs.
    named.sort_by(|a, c| a.0.cmp(&c.0));
    for (name, dtype, dims, data) in named {
        b.add_tensor(&name, dtype, dims, data)?;
        report.written += 1;
    }

    Ok((b, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    // Minimal protobuf encoders (test-only).
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
    fn write_len_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        write_varint(out, (u64::from(field) << 3) | 2);
        write_varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }
    fn write_varint_field(out: &mut Vec<u8>, field: u32, v: u64) {
        write_varint(out, u64::from(field) << 3);
        write_varint(out, v);
    }
    fn tensor(name: &str, dims: &[u64], data_type: i32, raw: &[u8]) -> Vec<u8> {
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
    fn model_with_initializers(tensors: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        for t in tensors {
            write_len_field(&mut graph, 5, t);
        }
        let mut m = Vec::new();
        write_len_field(&mut m, 7, &graph);
        m
    }

    fn attribute_with_tensor(t: &[u8]) -> Vec<u8> {
        let mut a = Vec::new();
        write_len_field(&mut a, 1, b"value"); // AttributeProto.name
        write_len_field(&mut a, 5, t); // AttributeProto.t
        a
    }
    fn attribute_with_graph(name: &[u8], g: &[u8]) -> Vec<u8> {
        let mut a = Vec::new();
        write_len_field(&mut a, 1, name); // AttributeProto.name
        write_len_field(&mut a, 6, g); // AttributeProto.g
        a
    }
    /// A `Constant` node: `output` names the edge, the weight rides in the
    /// `value` tensor attribute (whose embedded name may be empty or the bare
    /// parameter name — exactly the upstream silero layout).
    fn constant_node(output: &str, value: &[u8]) -> Vec<u8> {
        let mut n = Vec::new();
        write_len_field(&mut n, 2, output.as_bytes()); // NodeProto.output
        write_len_field(&mut n, 4, b"Constant"); // NodeProto.op_type
        write_len_field(&mut n, 5, &attribute_with_tensor(value)); // NodeProto.attribute
        n
    }
    fn graph_with_nodes(nodes: &[Vec<u8>]) -> Vec<u8> {
        let mut g = Vec::new();
        for n in nodes {
            write_len_field(&mut g, 1, n); // GraphProto.node
        }
        g
    }
    /// A top-level `If` node carrying `then_branch` / `else_branch` subgraphs
    /// (the silero v5 layout: all weights live inside the two branches).
    fn model_with_if_branches(then_nodes: &[Vec<u8>], else_nodes: &[Vec<u8>]) -> Vec<u8> {
        let mut if_node = Vec::new();
        write_len_field(&mut if_node, 2, b"If_0_output"); // NodeProto.output
        write_len_field(&mut if_node, 4, b"If"); // NodeProto.op_type
        write_len_field(
            &mut if_node,
            5,
            &attribute_with_graph(b"then_branch", &graph_with_nodes(then_nodes)),
        );
        write_len_field(
            &mut if_node,
            5,
            &attribute_with_graph(b"else_branch", &graph_with_nodes(else_nodes)),
        );
        let mut m = Vec::new();
        write_len_field(&mut m, 7, &graph_with_nodes(&[if_node]));
        m
    }

    const THEN: &str = "If_0_then_branch__Inline_0__";
    const ELSE: &str = "If_0_else_branch__Inline_0__";

    #[test]
    fn writes_both_branches_under_rate_prefixes() {
        // Same embedded (bare) parameter name in both branches — the upstream
        // silero layout that the old de-dup collapsed to 8 kHz only — with
        // *different* payloads so cross-branch mix-ups are detectable.
        let w16: Vec<u8> = [0.5f32, 1.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        let w8: Vec<u8> = [2.5f32, 3.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        let then_c = constant_node(
            &format!("{THEN}stft.forward_basis_buffer"),
            &tensor("stft.forward_basis_buffer", &[2], ONNX_DTYPE_FLOAT, &w16),
        );
        let else_c = constant_node(
            &format!("{ELSE}stft.forward_basis_buffer"),
            &tensor("stft.forward_basis_buffer", &[2], ONNX_DTYPE_FLOAT, &w8),
        );
        let onnx_bytes = model_with_if_branches(&[then_c], &[else_c]);

        let (builder, report) = convert(onnx_bytes).unwrap();
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.skipped_stray, 0);

        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("silero-vad")
        );
        assert!(file.get(chunks::KEY_FRONTEND_N_FFT).is_none());
        assert_eq!(file.tensors().len(), 2);
        // then-branch = 16 kHz, else-branch = 8 kHz (If(sr == 16000); SPEC).
        assert_eq!(
            file.tensor_data("sr16k.stft.forward_basis_buffer").unwrap(),
            w16.as_slice()
        );
        assert_eq!(
            file.tensor_data("sr8k.stft.forward_basis_buffer").unwrap(),
            w8.as_slice()
        );
        // Deterministic name-sorted (byte-lexicographic) tensor order, exactly
        // as gen_reference.py's `sorted(tensors)`: "sr16k…" < "sr8k…".
        let names: Vec<&str> = file.tensors().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "sr16k.stft.forward_basis_buffer",
                "sr8k.stft.forward_basis_buffer"
            ]
        );
    }

    #[test]
    fn skips_int_constants_and_op_scope_float_strays() {
        let w: Vec<u8> = [1.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        let param = constant_node(
            &format!("{THEN}decoder.decoder.2.bias"),
            &tensor("decoder.decoder.2.bias", &[1], ONNX_DTYPE_FLOAT, &w),
        );
        // Op-scope float scalar with an empty embedded name (upstream:
        // `.../stft/Constant_22_output_0`) — a graph constant, not a parameter.
        let stray = constant_node(
            &format!("{THEN}/stft/Constant_22_output_0"),
            &tensor("", &[], ONNX_DTYPE_FLOAT, &w),
        );
        // An INT64 (data_type 7) constant that must be skipped, not written.
        let int_c = constant_node(
            &format!("{THEN}/Slice_output_0"),
            &tensor("", &[1], 7, &[3, 0, 0, 0, 0, 0, 0, 0]),
        );
        let onnx_bytes = model_with_if_branches(&[param, stray, int_c], &[]);

        let (builder, report) = convert(onnx_bytes).unwrap();
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 1);
        assert_eq!(report.skipped_stray, 1);
        assert_eq!(builder.tensor_count(), 1);
    }

    #[test]
    fn rejects_model_without_branch_weights() {
        // Weights outside the two `If` branches are not the documented silero
        // v5 layout: converting must fail loudly, not emit an empty/partial
        // GGUF (FR-EX-08).
        let w: Vec<u8> = [0.5f32, 1.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        let float_t = tensor("stft.weight", &[2], ONNX_DTYPE_FLOAT, &w);
        let err = convert(model_with_initializers(&[float_t])).unwrap_err();
        assert!(
            matches!(err, crate::ConvertError::Parse(ref m) if m.contains("If")),
            "want Parse error naming the If-branch layout, got: {err:?}"
        );
    }
}
