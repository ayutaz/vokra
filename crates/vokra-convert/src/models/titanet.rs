//! **NVIDIA TitaNet-Large** (`nvidia/speakerverification_en_titanet_large`,
//! **CC-BY-4.0**): safetensors → GGUF conversion (SoTA follow-on, 2026-07-30).
//!
//! Input: the upstream `nvidia/speakerverification_en_titanet_large` release
//! — a ~23 M-parameter depth-wise-separable Conv1D speaker-verification
//! network trained on VoxCeleb-1 / VoxCeleb-2 / Fisher / Switchboard /
//! LibriSpeech / SRE (16 kHz mono, 192-d embedding). Output: a GGUF
//! carrying every float tensor verbatim under its upstream safetensors
//! name, plus the `vokra.provenance.*` / `vokra.model.*` metadata chunks
//! a future native TitaNet loader will read.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `nvidia/speakerverification_en_titanet_large`
//!   (recorded under `vokra.provenance.upstream_hf`).
//! - SPDX: `cc-by-4.0` (`LicenseClass::AttributionRequired`, primary source
//!   = HF model card `cardData.license` YAML frontmatter + card body
//!   "License to use this model is covered by the CC-BY-4.0", fetched
//!   2026-07-30). Attribution to NVIDIA is required (both NOTICE §11
//!   for the code-level credit and the runtime-side
//!   [`vokra_core::stamp_attribution`] chunk for the FR-MD-09 display
//!   surface).
//! - Model category: `speaker` (recorded under `vokra.model.category`).
//!
//! # Upstream format bridge (`.nemo` → safetensors)
//!
//! NVIDIA releases TitaNet as a `.nemo` tarball (a tar-of-yaml + torch
//! `.ckpt` pickle). This converter accepts **only** safetensors input;
//! `.nemo` callers must first extract the checkpoint through
//! `tools/parity/nemo_pt_to_safetensors.py` (which unwraps the tar,
//! `torch.load`s the pickle, and strips training-only int counters like
//! BatchNorm `num_batches_tracked` — exactly the Canary / Parakeet-CTC
//! precedent). Int-tensor strip is done at the bridge script; this
//! converter accepts safetensors and its reader admits only F32 / F16 /
//! BF16 (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
//! remaining non-float would fail parse before reaching us — the
//! defensive `skipped_non_float` counter in [`TitaNetReport`] preserves
//! the sibling converter shape.
//!
//! # BF16 pass-through (mirror of `wespeaker` / `ecapa_tdnn` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling skeleton
//! converters. No convert-time widening; runtime widens BF16 → f32
//! losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec / WeSpeaker / ECAPA-TDNN contract). Real-weight
//! binding is a **follow-up wave** gated on the M5-residual op landing
//! (`TITANET_SPEAKER_ENCODE_OP` in
//! `crates/vokra-core/src/m5_residual_ops.rs`, FR-OP-80 variant).
//!
//! # Runtime port is out-of-scope
//!
//! This converter provides the byte-parallel GGUF surface only; a
//! consumer needing a speaker embedding today should use CAM++
//! (`vokra-models::speaker_encode`) which already covers fbank-80 →
//! 192-d embedding under Apache-2.0 (no attribution overhead). TitaNet
//! runtime binding is M5-residual (`docs/adr/M5-ORPHAN-SCOPE-residual-ops-amx-sme.md`).
//!
//! # No ONNX (permanent)
//!
//! NVIDIA distributes TitaNet as a `.nemo` (torch pickle inside a tar);
//! this converter **never** touches ONNX (FR-LD-05); a native
//! re-implementation lives in a future `crates/vokra-models/src/titanet/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream NeMo `EncDecSpeakerLabelModel`
//! inference path is deferred to the runtime landing wave — this
//! converter only guarantees byte-identical tensor pass-through +
//! metadata stamps. `docs/license-audit.md` §3.1 sign-off = ☑ Commercial
//! 2026-07-30 yousan (weight license verified via HF primary source;
//! runtime parity harness is a follow-up when the M5-residual op lands).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for TitaNet-Large GGUFs. Intentionally **distinct**
/// from `ecapa_tdnn` / `wespeaker` / `campplus` / `speaker_3d` because
/// every speaker-encoder family shares the functional surface
/// (`waveform 16 kHz → 192-d embedding`) but NOT its tensor topology
/// (TitaNet = depth-wise separable Conv1D with Squeeze-Excitation blocks;
/// ECAPA-TDNN = SE-Res2Blocks + attentive stat pooling; CAM++ = D-TDNN
/// with context-aware masking; ERes2Net = enhanced Res2Net stack).
/// Silently sharing an arch tag would mis-route runtime dispatch.
pub const ARCH: &str = "titanet-large";

/// `vokra.model.name` value written for the canonical
/// `nvidia/speakerverification_en_titanet_large` GGUF.
pub const NAME: &str = "titanet-large";

/// Model-category tag written under `vokra.model.category`. `"speaker"`
/// distinguishes speaker-embedding / speaker-verification networks from
/// TTS / ASR / codec / vocoder siblings so downstream consumers can
/// pick a load path without inspecting the arch (same key + value the
/// `wespeaker` / `ecapa_tdnn` / `speaker_3d` siblings use).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub const MODEL_CATEGORY: &str = "speaker";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`. Preserves upstream casing.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const UPSTREAM_HF: &str = "nvidia/speakerverification_en_titanet_large";

/// Default upstream weight licence (SPDX). Primary source: HF model
/// card `cardData.license` YAML frontmatter (`license: cc-by-4.0`) +
/// card body "License to use this model is covered by the CC-BY-4.0"
/// (fetched 2026-07-30). Callers who obtained the weight under a
/// different SPDX may override at the outer
/// `convert_file --license <spdx>` boundary; the class is re-derived
/// via [`LicenseClass::from_license_str`].
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-4.0";

/// FR-MD-09 attribution text stamped into `vokra.provenance.attribution`
/// (the runtime-side companion of NOTICE §11) — wording aligned with
/// the sibling `mimi` / `moshi` / `parakeet` / `kyutai_stt` CC-BY 4.0
/// converters. A deployer surfaces this string in their UI / About
/// screen to satisfy the CC-BY 4.0 display obligation (plus the
/// NFR-LG-03 store checklists) via [`vokra_core::resolve_attribution`].
pub const TITANET_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA TitaNet-Large \
    (speaker verification, depth-wise-separable Conv1D + Squeeze-Excitation, \
    16 kHz mono → 192-d embedding). Model weights are licensed under CC-BY 4.0 \
    (attribution required; commercial use permitted). Copyright (c) NVIDIA. \
    Source: https://huggingface.co/nvidia/speakerverification_en_titanet_large";

/// Outcome of a TitaNet conversion.
///
/// Mirrors `crate::models::wespeaker::WespeakerReport`'s counter set
/// (float pass-through + BF16 subset counter + non-float defensive
/// counter), plus a leading `read` count of every tensor observed in
/// the input safetensors header. `read == written + skipped_non_float`
/// is an invariant preserved by [`convert_titanet_file`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TitaNetReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `wespeaker` /
    /// `ecapa_tdnn` / `speaker_3d` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Mirrors the sibling `bf16_passthrough`
    /// observability counters — a silent widen / downcast cannot slip
    /// in undetected without this counter also drifting.
    pub bf16_passthrough: usize,
}

/// File-based TitaNet converter (`vokra-cli convert --model titanet-large`).
///
/// Reads `input` (a safetensors extracted from the upstream
/// `speakerverification_en_titanet_large.nemo` via
/// `tools/parity/nemo_pt_to_safetensors.py`), writes a Vokra GGUF to
/// `output`. `license` overrides the default `cc-by-4.0` provenance
/// stamp (Whisper / kokoro-family override pattern — see
/// `convert_file_licensed` in `lib.rs`); pass `None` to keep the
/// built-in `cc-by-4.0` stamp + attribution.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_titanet_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<TitaNetReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a load path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-4.0 (upstream
    // `nvidia/speakerverification_en_titanet_large` HF cardData).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (
            DEFAULT_LICENSE_SPDX.to_owned(),
            LicenseClass::AttributionRequired,
        ),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "nvidia/speakerverification_en_titanet_large \
             (TitaNet-Large speaker verification, VoxCeleb + Fisher + Switchboard \
              + LibriSpeech + SRE, 16 kHz mono → 192-d embedding, CC-BY-4.0)",
        ),
    );
    // FR-MD-09: CC-BY-4.0 obliges whoever redistributes these weights
    // to carry the credit. Burning the string into the artifact is
    // what lets a downstream consumer (and `scripts/publish/make_model_card.py`)
    // discharge that without having to know NVIDIA's terms
    // independently. Only stamped when the effective class actually
    // requires attribution — a caller who overrides to a permissive
    // SPDX (e.g. their own permissive retrain) does not carry the
    // attribution obligation.
    if class.requires_attribution() {
        vokra_core::stamp_attribution(&mut b, TITANET_ATTRIBUTION_TEXT);
    }

    let mut report = TitaNetReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as wespeaker /
    // ecapa_tdnn / voxcpm2; runtime widens BF16 → f32 exactly at load
    // via `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
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
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Mirrors the wespeaker helper.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
        );
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

    /// Builds a mixed F32 + F16 safetensors buffer with synthetic
    /// TitaNet-esque tensor names (`encoder.encoder.0.mconv.conv.weight`,
    /// `decoder.pool_dense.dense.weight`) so the round-trip exercises
    /// realistic strings, not synthetic ones.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            f32_elems as usize * 4,
            "F32 payload len must match shape × 4"
        );
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(
            f16_bytes.len(),
            f16_elems as usize * 2,
            "F16 payload len must match shape × 2"
        );
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
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
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

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-titanet-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's file → file round-trip with its dtype preserved
    /// (`GgmlType::BF16`, GGUF type 30) and its payload byte-identical.
    /// A silent widen at convert time would still round-trip _values_
    /// (BF16 → f32 widen is exact), so this test asserts on the dtype
    /// AND the raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror an actual upstream TitaNet tensor name (per NeMo
        // `EncDecSpeakerLabelModel` — ContextNet-style encoder + attentive
        // stats pool decoder) so the round-trip exercises a realistic
        // string, not a synthetic one.
        let input_bytes =
            safetensors_one_bf16("encoder.encoder.0.mconv.0.conv.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_titanet_file(&input_path, &output_path, None)
            .expect("convert_titanet_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror wespeaker / ecapa_tdnn / voxcpm2)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("encoder.encoder.0.mconv.0.conv.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "2 rows × 3 cols × 2 B BF16 verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Round-trips a synthetic ECAPA/TitaNet-shape mixed-dtype (F32 +
    /// F16) buffer and verifies:
    ///   1. every tensor rides the pass-through arm with its dtype
    ///      preserved,
    ///   2. the BF16 counter stays at Default 0 (additive-field
    ///      regression guard — a stray increment would signal that a
    ///      non-BF16 tensor was misclassified),
    ///   3. the arch / provenance / attribution / category stamps land.
    #[test]
    fn f32_and_f16_tensors_pass_through_with_metadata() {
        // Non-zero payloads so a silent-widen regression can't hide
        // behind trivial round-trips (avoid `3.14` / `2.71` etc. that
        // trip `clippy::approx_constant`; use arbitrary non-round
        // finite floats instead).
        let f32_vals: [f32; 6] = [7.0, -8.25, 3.125, -2.5, 0.75, 1.5];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). 1.0 = 0x3C00, -2.0 = 0xC000,
        // -0.5 = 0xB800, 3.0 = 0x4200, 0.15625 = 0x3100, 42.0 = 0x5140.
        // Six values for a [2,3] tensor = 12 bytes.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12, "6 elements × 2 bytes F16 payload");

        let input_bytes = safetensors_f32_then_f16(
            // TitaNet ContextNet-style depthwise-separable conv block.
            "encoder.encoder.0.mconv.0.conv.weight",
            &[2, 3],
            &f32_bytes,
            // Attentive stats pool + final linear decoder.
            "decoder.pool_dense.dense.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_titanet_file(&input_path, &output_path, None)
            .expect("convert_titanet_file must accept a mixed F32/F16 checkpoint");

        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Round-trip carries both tensors with their dtypes preserved
        // AND the arch / provenance / category / attribution stamps
        // land (the AttributionRequired path is exercised by the
        // default-license branch).
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("encoder.encoder.0.mconv.0.conv.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("decoder.pool_dense.dense.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance / category / attribution chunks landed (task-spec
        // pins — every one is asserted to catch a silent metadata drop).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );
        // FR-MD-09 attribution chunk must be present + non-empty
        // (silent skip = CC-BY-4.0 obligation breach).
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("FR-MD-09 attribution chunk present");
        assert!(
            attr.contains("NVIDIA") && attr.contains("CC-BY 4.0"),
            "attribution text names NVIDIA + CC-BY 4.0 (was: {attr})"
        );
        assert_eq!(
            attr, TITANET_ATTRIBUTION_TEXT,
            "attribution text is the constant, verbatim"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Licence override at the outer `convert_file --license <spdx>`
    /// boundary re-derives the class through
    /// [`LicenseClass::from_license_str`]. When the caller overrides
    /// to a permissive SPDX (e.g. their own permissive retrain of the
    /// TitaNet architecture), the AttributionRequired chunk **must
    /// not** be written (attribution obligation applies only to the
    /// upstream CC-BY-4.0 weight).
    #[test]
    fn license_override_to_permissive_drops_attribution_chunk() {
        let values: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.encoder.0.mconv.0.conv.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&f32_bytes);

        let input_path = write_temp("override-in", &input);
        let output_path = write_temp("override-out", &[]);

        convert_titanet_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("convert_titanet_file with permissive override");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX is stamped verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override class is re-derived (apache-2.0 → Permissive)"
        );
        // FR-MD-09 attribution chunk **must not** be present when the
        // caller overrides away from CC-BY (silent inheritance of the
        // NVIDIA attribution string would misattribute their own work).
        assert!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION).is_none(),
            "attribution chunk must NOT be stamped when class is not AttributionRequired"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Attribution constant is a public API surface — it must name
    /// NVIDIA (the copyright holder) and CC-BY 4.0 (the licence
    /// citation) so a downstream consumer scanning the constant knows
    /// what obligation it discharges without inspecting the runtime
    /// `resolve_attribution` flow.
    #[test]
    fn attribution_text_names_copyright_holder_and_license() {
        assert!(
            TITANET_ATTRIBUTION_TEXT.contains("NVIDIA"),
            "attribution names the copyright holder"
        );
        assert!(
            TITANET_ATTRIBUTION_TEXT.contains("CC-BY 4.0"),
            "attribution cites CC-BY 4.0"
        );
        assert!(
            TITANET_ATTRIBUTION_TEXT
                .contains("https://huggingface.co/nvidia/speakerverification_en_titanet_large"),
            "attribution carries the source URL"
        );
    }

    /// `AttributionRequired` is the correct classification for CC-BY 4.0:
    /// the [`LicenseClass::from_license_str`] parser + the default arm
    /// must agree, otherwise a caller wiring their own SPDX string
    /// would silently land in a different class than the built-in
    /// default.
    #[test]
    fn cc_by_4_0_maps_to_attribution_required() {
        assert_eq!(
            LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
            LicenseClass::AttributionRequired,
            "cc-by-4.0 must classify as AttributionRequired"
        );
        assert!(
            LicenseClass::AttributionRequired.requires_attribution(),
            "AttributionRequired implies requires_attribution() = true"
        );
        assert!(
            LicenseClass::AttributionRequired.commercial_ok(),
            "CC-BY 4.0 permits commercial redistribution"
        );
        assert!(
            LicenseClass::AttributionRequired.redistributable(),
            "CC-BY 4.0 permits redistribution"
        );
    }
}
