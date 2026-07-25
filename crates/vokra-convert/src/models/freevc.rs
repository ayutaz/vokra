//! **FreeVC**: safetensors checkpoint → GGUF conversion.
//!
//! Upstream: `huggingface.co/OlaWod/FreeVC` (SPDX: `mit`).
//! Category: `vc` — one-shot voice conversion built on a VITS backbone
//! (posterior encoder + flow + HiFi-GAN decoder), conditioned by a
//! WavLM content encoder + external speaker encoder.
//!
//! # Contract (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! Every F32 / F16 / BF16 tensor in the input safetensors is emitted
//! **verbatim** into the output GGUF under its upstream name — no
//! convert-time widening (BF16 → GGUF type 30 `GgmlType::BF16`; runtime
//! widens to f32 losslessly via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`, which is exact
//! `bits << 16`). Every other dtype is counted in
//! `FreevcReport::skipped_non_float` (defensive: the safetensors reader
//! only accepts F32/F16/BF16 at parse time today, so this arm is a
//! reader-change tripwire).
//!
//! # Provenance
//!
//! The output GGUF carries:
//! - `vokra.model.arch = "freevc"` (runtime dispatch tag).
//! - `vokra.model.name = "freevc"`.
//! - `vokra.provenance.weight_license` = the canonical
//!   [`LicenseClass`] for the *effective* SPDX (MIT by default;
//!   overridable per call via `license`).
//! - `vokra.provenance.license` = the raw SPDX string
//!   (default `"mit"`; overridden if `license` is `Some`).
//! - `vokra.provenance.model_id = "freevc"`.
//! - `vokra.provenance.source = "OlaWod/FreeVC (mit)"` — advisory
//!   upstream note, mirroring the sibling converters.
//! - `vokra.schema.version` / `vokra.schema.producer` — written
//!   automatically by `GgufBuilder::to_bytes()`
//!   (`crates/vokra-core/src/gguf/writer.rs effective_metadata`), one
//!   choke point for the whole crate. No per-converter stamp is needed.
//!
//! # License override
//!
//! The upstream OlaWod/FreeVC repo is MIT end-to-end, but a
//! redistributor whose training corpus or fine-tuned artifact carries a
//! different SPDX passes the actual redistribution license as
//! `license = Some("<spdx>")`. The override wins over the built-in
//! default and is re-classified via
//! [`LicenseClass::from_license_str`] so the compliance gate reflects
//! the artifact the caller is actually publishing (single source of
//! truth = the GGUF, no card/artifact drift — see
//! `crates/vokra-convert/src/lib.rs convert_file_licensed` rustdoc).
//!
//! # No side-car config
//!
//! FreeVC ships hparams as PyTorch defaults in
//! `github.com/OlaWod/FreeVC/blob/main/models.py` (there is no upstream
//! `config.json`). The runtime side is a follow-up wave; this converter
//! is intentionally shape-driven — every float tensor passes through
//! under its upstream name so a future `FreevcWeights::from_gguf` can
//! walk the same names without a converter round-trip.
//!
//! # No ONNX (permanent)
//!
//! FreeVC is distributed as PyTorch checkpoints + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05). The pipeline is
//! re-implemented natively in `crates/vokra-models/src/freevc/` when the
//! runtime wave lands (whisper.cpp 型 self re-implementation, CLAUDE.md
//! 設計判断 4).
//!
//! # Dead-code allowance (TDD skeleton)
//!
//! Every item in this module is `pub` so the outer boundary
//! (`crates/vokra-convert/src/lib.rs` — `convert_file_licensed` +
//! `pub use models::freevc::…`) can wire the converter into the CLI
//! and the `ModelKind::Freevc` dispatch. That integration is a **later
//! commit** (kept out of this commit's diff so the TDD skeleton lands
//! atomically without dragging `lib.rs` / `main.rs` / `ModelKind` +
//! `convert_file_licensed`); until it lands the module's pub API is
//! reachable only from the tests under `#[cfg(test)]`, which the
//! lib-target dead-code pass does not see. The module-level
//! `#![allow(dead_code)]` below is the honest interim state — remove
//! it in the follow-up that adds the `pub use` re-export and the
//! `ModelKind::Freevc` arm.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FreeVC GGUFs. Kept distinct from every other
/// arch tag so silently sharing one cannot mis-route the runtime
/// dispatch (voice conversion pipeline vs. TTS / ASR).
pub const ARCH: &str = "freevc";

/// `vokra.model.name` value written for the canonical FreeVC GGUF.
pub const NAME: &str = "freevc";

/// `vokra.model.category` value. FreeVC is a one-shot voice conversion
/// model — the `vc` category.
pub const CATEGORY: &str = "vc";

/// Upstream HuggingFace path — recorded in the provenance chunk as the
/// advisory source note.
pub const UPSTREAM_HF: &str = "OlaWod/FreeVC";

/// Default SPDX identifier of the upstream OlaWod/FreeVC release. The
/// per-call `license` parameter overrides this when the caller
/// redistributes an artifact under a different license.
pub const DEFAULT_LICENSE: &str = "mit";

/// Outcome of a FreeVC conversion.
///
/// Field layout mirrors `Qwen3TtsReport` / `VibeVoiceReport` /
/// `VoxCpm2Report`, plus the explicit `read` counter so the invariant
/// `read == written + skipped_non_float` is machine-checkable at the
/// call site (guards a silent drop / double-count regression the
/// sibling in-memory converters catch only through the per-arm counter
/// sum).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FreevcReport {
    /// Total tensors observed in the input safetensors (regardless of
    /// dtype). Invariant: `read == written + skipped_non_float`.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time today, so
    /// anything reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling converters and to
    /// make the invariant above machine-checkable).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — the same
    /// symmetric-rewrite red-line the M4-06 posture pin was rewritten
    /// under (see `qwen3_tts::Qwen3TtsReport::bf16_passthrough`).
    pub bf16_passthrough: usize,
}

/// Converts a FreeVC safetensors checkpoint at `input` into a GGUF at
/// `output`, returning the per-arm [`FreevcReport`].
///
/// `license` is the SPDX string to stamp into
/// `vokra.provenance.license` (and the class re-derived into
/// `vokra.provenance.weight_license`). `None` keeps the built-in
/// default (`"mit"` for upstream OlaWod/FreeVC); passing `Some("<spdx>")`
/// overrides both metadata entries — the mechanism
/// `convert_file_licensed` uses at the outer boundary
/// (`crates/vokra-convert/src/lib.rs`).
///
/// # Errors
///
/// - [`ConvertError::Io`] if reading `input` or writing `output` fails.
/// - [`ConvertError::Parse`] if the safetensors header is malformed
///   (delegated to [`SafetensorsFile::parse`]).
/// - [`ConvertError::Gguf`] if the GGUF builder rejects a tensor
///   declaration.
pub fn convert_freevc_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FreevcReport, ConvertError> {
    // Read the whole checkpoint. FreeVC is small enough (VITS backbone +
    // WavLM references — under 200 MB even at F32) that the in-memory
    // parse is fine; a streaming path (Moshi-style) is not required and
    // would trade complexity for no observable savings.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // FreeVC has no side-car config today (upstream ships hparams as
    // PyTorch defaults in `models.py`), so no `write_hparams` call — the
    // metadata surface is provenance + runtime dispatch tags only.
    //
    // `vokra.model.category` — advisory taxonomy tag ("vc" for
    // voice-conversion) so tooling can filter by model class without
    // parsing the arch string. The key is not (yet) in
    // `vokra-core::gguf::chunks`; it lands here first because FreeVC is
    // the first `vc` model in the tree and a chunks entry can be
    // promoted in a later commit without breaking round-trip. Kept as a
    // literal (not a `chunks::` constant) to make that boundary visible
    // — no cross-crate change is required to land the converter.
    b.add_string("vokra.model.category", CATEGORY);

    // License override: caller-supplied SPDX wins over the built-in
    // default; both metadata entries (`weight_license` + `license`) are
    // re-derived from the effective SPDX so the compliance gate reflects
    // the artifact the caller is actually publishing.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    let effective_class = LicenseClass::from_license_str(effective_license);
    let source_note = format!("{UPSTREAM_HF} ({effective_license})");
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some(&source_note),
    );

    let mut report = FreevcReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30); runtime widens BF16 → f32 exactly
    // at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
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
    debug_assert_eq!(
        report.read,
        report.written + report.skipped_non_float,
        "per-arm counter invariant"
    );

    let gguf_bytes = b.to_bytes()?;
    std::fs::write(output, gguf_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-test scratch path under the OS temp dir, tagged with pid so
    /// concurrent test runs never clobber each other.
    fn scratch_path(basename: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vokra-freevc-{}-{basename}", std::process::id()));
        p
    }

    /// Emits a single-tensor safetensors buffer with dtype `dtype_str`,
    /// shape `shape`, and raw `payload` under `tensor_name`. Panics if
    /// `payload.len()` disagrees with `shape × sizeof(dtype)`.
    fn one_tensor_safetensors(
        tensor_name: &str,
        dtype_str: &str,
        shape: &[u64],
        payload: &[u8],
    ) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{tensor_name}":{{"dtype":"{dtype_str}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Two-tensor safetensors buffer with F32 first then F16 (byte
    /// order: `f32_bytes` then `f16_bytes`). The header declares the
    /// two `data_offsets` spans matching those byte ranges, with shapes
    /// derived from the payload lengths so the safetensors parser's
    /// `shape × sizeof(dtype)` check never rejects a synthetic input.
    fn two_tensor_f32_then_f16(f32_bytes: &[u8], f16_bytes: &[u8]) -> Vec<u8> {
        assert_eq!(
            f32_bytes.len() % 4,
            0,
            "F32 payload must be a whole f32 count"
        );
        assert_eq!(
            f16_bytes.len() % 2,
            0,
            "F16 payload must be a whole f16 count"
        );
        let f32_elems = (f32_bytes.len() / 4) as u64;
        let f16_elems = (f16_bytes.len() / 2) as u64;
        let f32_end = f32_bytes.len();
        let f16_end = f32_end + f16_bytes.len();
        let header = format!(
            r#"{{"enc_p.weight":{{"dtype":"F32","shape":[{f32_elems}],"data_offsets":[0,{f32_end}]}},"dec.weight":{{"dtype":"F16","shape":[{f16_elems}],"data_offsets":[{f32_end},{f16_end}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    /// The pinned BF16 pass-through contract: a synthetic BF16
    /// safetensors goes in, the output GGUF must carry the tensor with
    /// `GgmlType::BF16` and the payload byte-identical to the input
    /// (no convert-time widen). Regression fence for the FR-EX-08
    /// silent-widen prohibition.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 payload so a silent widen to F32/F16 cannot
        // hide behind a trivially-round-tripping zero blob.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        let blob = one_tensor_safetensors("dec.weight", "BF16", &[2, 3], &bf16);
        let input = scratch_path("bf16-in.safetensors");
        let output = scratch_path("bf16-out.gguf");
        std::fs::write(&input, &blob).expect("write scratch safetensors");

        let result = convert_freevc_file(&input, &output, None);
        // Clean up before asserting so a panic-in-assert does not leak
        // the scratch files across test invocations.
        let _ = std::fs::remove_file(&input);

        let report = result.expect("convert_freevc_file");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the GGUF: dtype preserved, payload
        // byte-identical (no convert-time widen).
        let gguf_bytes = std::fs::read(&output).expect("read output GGUF");
        let _ = std::fs::remove_file(&output);
        let file = GgufFile::parse(gguf_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("dec.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        // Provenance / model-arch handshake (the runtime dispatch key).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE),
            "no license override → default `mit`"
        );
    }

    /// F32 and F16 tensors in the same input both pass through with
    /// their dtypes preserved; the BF16 counter stays at 0. Pins the
    /// "mixed-dtype loops don't collapse to one arm" contract.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 payload: three known bit patterns (1.0, 2.0, 3.0 as
        // IEEE-754 binary16) so any silent widen / drop trips loudly.
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let blob = two_tensor_f32_then_f16(&f32_bytes, &f16_bytes);
        let input = scratch_path("f32-f16-in.safetensors");
        let output = scratch_path("f32-f16-out.gguf");
        std::fs::write(&input, &blob).expect("write scratch safetensors");

        let result = convert_freevc_file(&input, &output, None);
        let _ = std::fs::remove_file(&input);

        let report = result.expect("convert_freevc_file");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(
            report.bf16_passthrough, 0,
            "no BF16 tensor in this input; counter must be 0"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Round-trip: both tensors survive with their dtypes preserved.
        let gguf_bytes = std::fs::read(&output).expect("read output GGUF");
        let _ = std::fs::remove_file(&output);
        let file = GgufFile::parse(gguf_bytes).expect("parse output GGUF");

        let f32_info = file.tensor_info("enc_p.weight").expect("F32 present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file.tensor_info("dec.weight").expect("F16 present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());
    }
}
