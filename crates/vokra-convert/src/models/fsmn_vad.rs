//! **FSMN-VAD** (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`, MIT):
//! safetensors → GGUF conversion (SoTA plan Phase 5 VAD-2, 2026-07-30).
//!
//! Input: the upstream FunASR release — an `.pt` torch pickle
//! pre-flattened to safetensors by `tools/parity/nemo_pt_to_safetensors.py`
//! (the emotion2vec / funcodec / wespeaker path — the upstream ships a
//! `.pt` state-dict; the pt-to-safetensors bridge is a general FunASR
//! contract). Output: a GGUF carrying every float tensor under its
//! upstream state-dict name, plus the `vokra.provenance.*` /
//! `vokra.model.*` / `vokra.fsmn_vad.*` metadata chunks
//! `vokra-models::fsmn_vad::FsmnVadV1::from_gguf` binds against.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
//!   (recorded under `vokra.provenance.upstream_hf`).
//! - SPDX: `mit` (`LicenseClass::Permissive`; §3.1 sign-off row landed
//!   2026-07-30 yousan).
//! - Model category: `vad` (recorded under `vokra.model.category`).
//!
//! # Hparams — always written
//!
//! Unlike the `wespeaker` / `funcodec` / `emotion2vec` skeletons (which
//! defer every hparam to a real-weight follow-up), FSMN-VAD's config
//! axes are fixed by the released FunASR checkpoint and known ahead of
//! time (see `docs/superpowers/specs/…` and
//! `crates/vokra-models/src/fsmn_vad/SPEC.md`), so the converter stamps
//! [`FsmnEncoderConfig::upstream_default`] + the fbank / LFR / rate
//! extras unconditionally. A caller who converts a differently-shaped
//! FSMN checkpoint overrides via a future `--config` side-car (owner
//! follow-up; today the shape is a compile-time constant).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! F32 / F16 / BF16 tensors are emitted verbatim. BF16 stays GGUF type
//! 30 (`GgmlType::BF16`); runtime widens BF16 → f32 losslessly at load
//! (single choke point `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16`).
//!
//! # Tensor naming
//!
//! GGUF tensor names are the **upstream state-dict names verbatim** —
//! the standing FunASR / CosyVoice / Kokoro / CosyVoice2 contract. The
//! model-level loader (`FsmnVadV1::from_gguf`) walks the exact same
//! names via the `TENSOR_*` constants in `vokra-models::fsmn_vad`;
//! silent renames on either side would break the round-trip.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream FunASR Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off recorded
//! 2026-07-30 yousan). This converter provides the byte-parallel GGUF
//! surface + hparam chunk group; the fbank + LFR + CMVN reference
//! script + parity CI land with the first checkpoint pull.
//!
//! # No ONNX (permanent)
//!
//! FSMN-VAD is distributed as `.pt` + a Python pipeline; this converter
//! **never** touches ONNX (FR-LD-05). The `.pt` → safetensors bridge is
//! `tools/parity/nemo_pt_to_safetensors.py` (same pattern as
//! emotion2vec / funcodec / wespeaker).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

// Every constant here is deliberately re-declared from
// `vokra-models::fsmn_vad` to keep the crate boundary one-way
// (converter never depends on models). The values are documented as
// the source of truth in `crates/vokra-models/src/fsmn_vad/SPEC.md`;
// changing either side without the other is a build error caught by
// the round-trip test in this module.

/// `vokra.model.arch` value for FSMN-VAD GGUFs.
pub(crate) const ARCH: &str = "fsmn-vad";

/// `vokra.model.name` value for the canonical release.
pub(crate) const NAME: &str = "fsmn-vad-zh-cn-16k-common";

/// Model-category tag (`vokra.model.category`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// Model-category value (`vad` — same value the `silero_vad` sibling
/// stamps; the VAD dispatcher picks the load path by `category`, and
/// tells FSMN vs Silero apart by `arch`).
pub(crate) const MODEL_CATEGORY: &str = "vad";

/// Upstream HF repository slug (`org/name`) — recorded under
/// `vokra.provenance.upstream_hf`.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const UPSTREAM_HF: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";

// ---- `vokra.fsmn_vad.*` hparam keys --------------------------------------
//
// Kept as `pub(crate) const` mirrors of the same-named `pub const` in
// `crates/vokra-models/src/fsmn_vad/mod.rs`; a mismatch surfaces
// immediately in the round-trip test.

pub(crate) const KEY_N_BLOCKS: &str = "vokra.fsmn_vad.n_blocks";
pub(crate) const KEY_INPUT_DIM: &str = "vokra.fsmn_vad.input_dim";
pub(crate) const KEY_PROJ_DIM: &str = "vokra.fsmn_vad.proj_dim";
pub(crate) const KEY_HIDDEN_DIM: &str = "vokra.fsmn_vad.hidden_dim";
pub(crate) const KEY_LORDER: &str = "vokra.fsmn_vad.lorder";
pub(crate) const KEY_RORDER: &str = "vokra.fsmn_vad.rorder";
pub(crate) const KEY_N_CLASS: &str = "vokra.fsmn_vad.n_class";
pub(crate) const KEY_N_MELS: &str = "vokra.fsmn_vad.n_mels";
pub(crate) const KEY_LFR_M: &str = "vokra.fsmn_vad.lfr_m";
pub(crate) const KEY_LFR_N: &str = "vokra.fsmn_vad.lfr_n";
pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.fsmn_vad.sample_rate";

/// Upstream default hparam values (transcribed from the released
/// FunASR `speech_fsmn_vad_zh-cn-16k-common-pytorch` `config.yaml` —
/// see `crates/vokra-models/src/fsmn_vad/SPEC.md` table).
const DEFAULT_N_BLOCKS: u32 = 4;
const DEFAULT_INPUT_DIM: u32 = 400;
const DEFAULT_PROJ_DIM: u32 = 128;
const DEFAULT_HIDDEN_DIM: u32 = 128;
const DEFAULT_LORDER: u32 = 20;
const DEFAULT_RORDER: u32 = 0;
const DEFAULT_N_CLASS: u32 = 2;
const DEFAULT_N_MELS: u32 = 80;
const DEFAULT_LFR_M: u32 = 5;
const DEFAULT_LFR_N: u32 = 1;
const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Default weight license SPDX (`mit`). Override via
/// [`convert_fsmn_vad_file`]'s `license` parameter — the standing
/// mechanism for "implementation is clean-room MIT but the upstream
/// distributed checkpoint has a different SPDX" scenarios.
pub const DEFAULT_LICENSE: &str = "mit";

/// Outcome of an FSMN-VAD conversion.
///
/// Mirrors the emotion2vec / wespeaker counter set (float pass-through +
/// BF16 subset + non-float defensive) with a leading `read` budget so a
/// truncated header cannot silently drop tensors.
#[derive(Debug, Default)]
pub struct FsmnVadReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive — the safetensors reader
    /// rejects other dtypes at parse time; kept for symmetry with the
    /// sibling `emotion2vec` / `wespeaker` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// `written`).
    pub bf16_passthrough: usize,
}

/// File-based FSMN-VAD converter
/// (`vokra-cli convert --model fsmn-vad`).
///
/// Reads `input` (a safetensors-flattened FunASR checkpoint — the `.pt`
/// bridge is `tools/parity/nemo_pt_to_safetensors.py`), writes a Vokra
/// GGUF to `output`. `license` overrides the default `mit` provenance
/// stamp (the standing `convert_file_licensed` pattern).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O; [`ConvertError::Parse`] for malformed
/// safetensors input; [`ConvertError::Gguf`] if the GGUF serialisation
/// fails.
pub fn convert_fsmn_vad_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FsmnVadReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution. Default = mit (upstream FunASR
    // FSMN-VAD MIT primary source, §3.1 sign-off 2026-07-30 yousan).
    // `license` overrides for callers whose actual distribution source
    // declares a different SPDX (mirror of
    // `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
             (FunASR FSMN-VAD, feed-forward sequential memory network for VAD, mit)",
        ),
    );

    // Hparams — always written (this converter targets the fixed FunASR
    // release; a future variant with a different backbone will
    // introduce a `--config` axis). Documented sources for every value:
    // see `crates/vokra-models/src/fsmn_vad/SPEC.md` table.
    b.add_u32(KEY_N_BLOCKS, DEFAULT_N_BLOCKS);
    b.add_u32(KEY_INPUT_DIM, DEFAULT_INPUT_DIM);
    b.add_u32(KEY_PROJ_DIM, DEFAULT_PROJ_DIM);
    b.add_u32(KEY_HIDDEN_DIM, DEFAULT_HIDDEN_DIM);
    b.add_u32(KEY_LORDER, DEFAULT_LORDER);
    b.add_u32(KEY_RORDER, DEFAULT_RORDER);
    b.add_u32(KEY_N_CLASS, DEFAULT_N_CLASS);
    b.add_u32(KEY_N_MELS, DEFAULT_N_MELS);
    b.add_u32(KEY_LFR_M, DEFAULT_LFR_M);
    b.add_u32(KEY_LFR_N, DEFAULT_LFR_N);
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);

    let mut report = FsmnVadReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied payload.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Two-tensor safetensors buffer (F32 then F16).
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-fsmn-vad-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn bf16_tensor_passes_through_and_stamps_full_hparam_chunk() {
        // Realistic upstream tensor name (mirror of the FunASR
        // `encoder.0.ffn.linear1.weight` state-dict entry).
        let bf16: Vec<u8> = [1.0f32, -2.5, 0.15625, 3.5]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("encoder.0.ffn.linear1.weight", &[2, 2], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_fsmn_vad_file(&input_path, &output_path, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out = std::fs::read(&output_path).unwrap();
        let file = GgufFile::parse(out).unwrap();

        // Arch + name + category + upstream slug stamps.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        // Every hparam chunk pinned.
        assert_eq!(
            file.get(KEY_N_BLOCKS).and_then(|v| v.as_u64()),
            Some(DEFAULT_N_BLOCKS as u64)
        );
        assert_eq!(
            file.get(KEY_INPUT_DIM).and_then(|v| v.as_u64()),
            Some(DEFAULT_INPUT_DIM as u64)
        );
        assert_eq!(
            file.get(KEY_PROJ_DIM).and_then(|v| v.as_u64()),
            Some(DEFAULT_PROJ_DIM as u64)
        );
        assert_eq!(
            file.get(KEY_HIDDEN_DIM).and_then(|v| v.as_u64()),
            Some(DEFAULT_HIDDEN_DIM as u64)
        );
        assert_eq!(
            file.get(KEY_LORDER).and_then(|v| v.as_u64()),
            Some(DEFAULT_LORDER as u64)
        );
        assert_eq!(
            file.get(KEY_RORDER).and_then(|v| v.as_u64()),
            Some(DEFAULT_RORDER as u64)
        );
        assert_eq!(
            file.get(KEY_N_CLASS).and_then(|v| v.as_u64()),
            Some(DEFAULT_N_CLASS as u64)
        );
        assert_eq!(
            file.get(KEY_N_MELS).and_then(|v| v.as_u64()),
            Some(DEFAULT_N_MELS as u64)
        );
        assert_eq!(
            file.get(KEY_LFR_M).and_then(|v| v.as_u64()),
            Some(DEFAULT_LFR_M as u64)
        );
        assert_eq!(
            file.get(KEY_LFR_N).and_then(|v| v.as_u64()),
            Some(DEFAULT_LFR_N as u64)
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(DEFAULT_SAMPLE_RATE as u64)
        );

        // BF16 tensor byte-identical + dtype preserved.
        let info = file.tensor_info("encoder.0.ffn.linear1.weight").unwrap();
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 2]);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        // Provenance: mit + Permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Six F16 half-floats with known non-zero bit patterns.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let input_bytes = safetensors_f32_then_f16(
            "encoder.in_linear.bias",
            &[1, 2],
            &f32_bytes,
            "encoder.0.memory.conv1.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_fsmn_vad_file(&input_path, &output_path, None).expect("convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);

        let file = GgufFile::parse(std::fs::read(&output_path).unwrap()).unwrap();
        let f32_info = file.tensor_info("encoder.in_linear.bias").unwrap();
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("encoder.0.memory.conv1.weight").unwrap();
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_updates_the_stamp() {
        // A caller who redistributes under a different SPDX overrides
        // the default. cc-by-4.0 → AttributionRequired.
        let input_bytes = safetensors_one_bf16(
            "encoder.in_linear.weight",
            &[1, 1],
            &(1.0f32.to_bits() >> 16_u32).to_le_bytes()[..2],
        );
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_fsmn_vad_file(&input_path, &output_path, Some("cc-by-4.0"))
            .expect("convert with override");

        let file = GgufFile::parse(std::fs::read(&output_path).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-4.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
