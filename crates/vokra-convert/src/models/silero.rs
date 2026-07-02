//! Silero VAD v5: ONNX checkpoint to GGUF conversion.
//!
//! Input: the upstream `snakers4/silero-vad` `silero_vad.onnx`. Output: a GGUF
//! carrying every float weight plus `vokra.model.*`.
//!
//! # Where Silero's weights live
//!
//! Silero VAD v5 stores no top-level `graph.initializer`s. Its weights are the
//! `value` attributes of `Constant` nodes inside the `then_branch` /
//! `else_branch` subgraphs of a top-level `If`. [`crate::onnx::read_weight_tensors`]
//! walks those subgraphs and returns the weight tensors, named by each
//! `Constant` node's output. Because the two `If` branches recompute the same
//! network, a weight name can appear twice; the second occurrence is de-duped
//! (kept once) rather than triggering a duplicate-tensor error.
//!
//! Non-float constants (the many int64 shape/slice/index constants that drive
//! control flow) fall outside the M0 dtype range (FP32/FP16) and are counted in
//! [`SileroReport::skipped_non_float`], not written.
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

/// Outcome of a Silero conversion.
#[derive(Debug, Default)]
pub(crate) struct SileroReport {
    /// Number of float weight tensors written to the GGUF.
    pub(crate) written: usize,
    /// Constants skipped because their dtype is outside M0's FP32/FP16 range
    /// (mostly int64 shape/index constants).
    pub(crate) skipped_non_float: usize,
    /// Float tensors dropped as duplicate names (same weight in both `If`
    /// branches).
    pub(crate) deduped: usize,
}

/// Converts a Silero VAD ONNX buffer into a populated GGUF builder plus a
/// report of what was written vs. skipped.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, SileroReport), ConvertError> {
    let tensors = onnx::read_weight_tensors(&bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);

    let mut report = SileroReport::default();
    let mut seen = std::collections::HashSet::new();
    for init in tensors {
        let dtype = match init.data_type {
            ONNX_DTYPE_FLOAT => GgmlType::F32,
            ONNX_DTYPE_FLOAT16 => GgmlType::F16,
            _ => {
                report.skipped_non_float += 1;
                continue;
            }
        };
        if !seen.insert(init.name.clone()) {
            report.deduped += 1;
            continue;
        }
        b.add_tensor(&init.name, dtype, init.dims.clone(), init.raw_le_bytes)?;
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

    #[test]
    fn converts_float_weights_and_skips_int_constants() {
        let w: Vec<u8> = [0.5f32, 1.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        let float_t = tensor("stft.weight", &[2], ONNX_DTYPE_FLOAT, &w);
        // An INT64 (data_type 7) constant that must be skipped, not written.
        let int_t = tensor("const_shape", &[1], 7, &[3, 0, 0, 0, 0, 0, 0, 0]);
        let onnx_bytes = model_with_initializers(&[float_t, int_t]);

        let (builder, report) = convert(onnx_bytes).unwrap();
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 1);

        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        // Model metadata present, but NO frontend chunk.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("silero-vad")
        );
        assert!(file.get(chunks::KEY_FRONTEND_N_FFT).is_none());
        // Only the float weight was written, bytes intact.
        assert_eq!(file.tensors().len(), 1);
        assert_eq!(file.tensor_data("stft.weight").unwrap(), w.as_slice());
    }

    #[test]
    fn dedupes_repeated_names() {
        let w: Vec<u8> = [1.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        let a = tensor("shared.weight", &[1], ONNX_DTYPE_FLOAT, &w);
        let b = tensor("shared.weight", &[1], ONNX_DTYPE_FLOAT, &w);
        let (builder, report) = convert(model_with_initializers(&[a, b])).unwrap();
        assert_eq!(report.written, 1);
        assert_eq!(report.deduped, 1);
        assert_eq!(builder.tensor_count(), 1);
    }
}
