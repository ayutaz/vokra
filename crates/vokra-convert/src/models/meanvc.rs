//! **MeanVC**: `ASLP-lab/MeanVC` safetensors checkpoint → GGUF conversion
//! (SoTA plan follow-up, 2026-07-25).
//!
//! Category: `vc` (voice conversion). Upstream release notes describe MeanVC
//! as a 2025 streaming zero-shot voice-conversion model built on the
//! **Mean Flow** family (a rectified-flow style continuous decoder).
//!
//! # Primary sources
//!
//! - Weights + license: `huggingface.co/ASLP-lab/MeanVC` (apache-2.0,
//!   fetched 2026-07-25 — CLAUDE.md「ハルシネーション厳禁」).
//! - Category: `vc` — voice-conversion; drives the model-zoo browse tab
//!   and the runtime dispatch group.
//!
//! # BF16 posture — pass-through (Moshi / Voxtral / qwen3-tts pattern)
//!
//! Every float tensor (`F32` / `F16` / `BF16`) passes through under its
//! upstream safetensors name. BF16 is emitted as GGUF type 30
//! (`GgmlType::BF16`) verbatim — no convert-time widening; the runtime
//! widens BF16 → f32 losslessly on load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
//! bits of an f32, so `bits << 16` is exact). This mirrors
//! `qwen3_tts::convert` / `moshi::convert` / `vibevoice::convert` /
//! `voxcpm2::convert` and preserves NFR-DS-02 (zero-dep) end-to-end.
//!
//! # Provenance
//!
//! Provenance is stamped through [`vokra_core::stamp_provenance`] plus
//! two ad-hoc namespace keys the prompt requested verbatim:
//!
//! - `vokra.provenance.upstream_hf` = `"ASLP-lab/MeanVC"`
//! - `vokra.provenance.license` = `"apache-2.0"` (may be overridden by
//!   the `license` argument for redistribution scenarios where the
//!   caller's own audit chose a different SPDX id — cf. the `--license`
//!   documentation on [`crate::convert_file_licensed`]).
//! - `vokra.model.category` = `"vc"`
//!
//! The GGUF writer's two unconditional `vokra.schema.*` stamps
//! (`vokra.schema.version` + `vokra.schema.producer`) are appended
//! automatically by [`vokra_core::gguf::GgufBuilder`] on serialise, so
//! this module does not touch them explicitly (see
//! `crates/vokra-core/src/gguf/writer.rs:737` for the pin).
//!
//! # No side-car config
//!
//! MeanVC ships a real upstream `config.json`, but this converter takes
//! **no** `--config` path today because the file is a shape-driven
//! pass-through — the runtime side (a future
//! `crates/vokra-models/src/meanvc/`) will bind weights by name and
//! surface a loud shape-gate error at forward time (FR-EX-08) if any
//! tensor disagrees with a transcribed axis. A future release that
//! reshapes the backbone would demand `--config`; the plain
//! [`convert_meanvc_file`] path stays additive.
//!
//! # No ONNX (permanent)
//!
//! MeanVC is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/meanvc/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

// The MeanVC converter is registered via `pub mod meanvc;` in
// `crates/vokra-convert/src/models/mod.rs`, but `mod models;` in
// `crates/vokra-convert/src/lib.rs` stays private for now (the
// crate-level `pub use models::…::convert_*` re-export is a follow-up,
// mirroring the `pub use models::denoise::{…}` and
// `pub use models::voxtral::VoxtralConfig` lines that promote sibling
// modules to the top-level API surface). Until that re-export lands
// the `pub fn convert_meanvc_file` here is reachable only from the
// in-file `#[cfg(test)]` module — which counts for correctness but not
// for dead-code analysis in release builds. The module-scope
// `#![allow(dead_code)]` below keeps `cargo clippy -D warnings` green
// without needing a lib.rs edit in this TDD land.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MeanVC GGUFs — kept in sync with the future
/// runtime constant `vokra-models::meanvc::EXPECTED_ARCH`. Intentionally
/// **distinct** from every sibling arch tag: MeanVC is a *voice
/// conversion* Mean Flow model, not a TTS vocoder or diffusion sampler,
/// so silently sharing an arch tag would misroute the runtime dispatch.
pub(crate) const ARCH: &str = "meanvc";

/// `vokra.model.name` value written for the canonical MeanVC GGUF.
pub(crate) const NAME: &str = "meanvc";

/// `vokra.model.category` value — voice conversion tab.
pub(crate) const CATEGORY: &str = "vc";

/// Upstream Hugging Face path — recorded verbatim in
/// `vokra.provenance.upstream_hf` so model cards / audit tooling can
/// walk back to the source without heuristic parsing.
pub(crate) const UPSTREAM_HF: &str = "ASLP-lab/MeanVC";

/// Canonical upstream SPDX id (apache-2.0). Callers pass `license =
/// Some("…")` to [`convert_meanvc_file`] to override this at
/// redistribution time.
pub(crate) const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Ad-hoc metadata keys the prompt requested verbatim (there is no
// existing `KEY_PROVENANCE_UPSTREAM_HF` / `KEY_MODEL_CATEGORY` constant
// in `vokra-core/src/gguf/chunks.rs` today — writing them as literal
// strings mirrors how sibling converters embed model-specific keys).
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Outcome of a [`convert_meanvc_file`] run.
///
/// Mirrors the counter layout the `qwen3_tts` / `vibevoice` / `voxcpm2`
/// pattern uses (float tensors written verbatim, non-float skipped,
/// BF16 subset counter) — plus a `read` counter that surfaces how many
/// tensor declarations the safetensors reader saw (so a zero-`written`
/// / zero-`read` divergence flags a schema drift loudly).
#[derive(Debug, Default)]
pub struct MeanvcReport {
    /// Tensor declarations the safetensors reader parsed.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader today accepts only F32 / F16 / BF16 at parse time, so any
    /// tensor reaching this arm would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `vokra-core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a MeanVC safetensors checkpoint at `input` into a GGUF
/// written to `output`, returning a [`MeanvcReport`].
///
/// `license` is an optional SPDX override for the raw
/// `vokra.provenance.license` stamp — pass `None` to keep the built-in
/// upstream default (`apache-2.0`), or `Some("mit")` (etc.) when the
/// caller's own audit is redistributing under a different SPDX id.
pub fn convert_meanvc_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MeanvcReport, ConvertError> {
    // 1. Read the upstream safetensors checkpoint into memory (the
    //    qwen3_tts / vibevoice / voxcpm2 pattern — MeanVC is 3-order-
    //    of-magnitude smaller than the Moshi 7B streaming target so the
    //    Moshi `SafetensorsFileReader` bounded-memory path is not
    //    required here; a future 3B+ MeanVC sibling can promote to
    //    streaming without changing this API).
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    // 2. Populate the metadata builder — arch + name + provenance + the
    //    two ad-hoc namespace keys the prompt requested verbatim.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // The `vokra.model.category` key is not yet promoted to a
    // `vokra-core::gguf::chunks` constant — write the literal here (the
    // sibling converters use the same ad-hoc-string pattern for
    // model-specific keys — no `chunks` constant is required).
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // 3. Provenance stamp. `license` overrides the built-in
    //    `apache-2.0` default when the caller's own audit is
    //    redistributing under a different SPDX id (the same posture
    //    `crate::convert_file_licensed` documents at the crate boundary).
    //    The `LicenseClass::Permissive` stamp is preserved on override
    //    — the two current sibling SPDX ids we accept (`apache-2.0` /
    //    `mit`) are both Permissive; a caller that hands in a
    //    non-Permissive SPDX id is out of scope for this shape-only
    //    stub and would surface at the `vokra-core` compliance gate at
    //    load time (FR-CP-03).
    let effective_license = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );
    // Additive ad-hoc `vokra.provenance.upstream_hf` key — the
    // canonical `vokra.provenance.source` written by `stamp_provenance`
    // is a free-form URL / note field; the prompt asks for a distinct
    // machine-parseable `upstream_hf` slot that always contains just the
    // HF `owner/name` slug (never a URL, never an attribution blob).
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // 4. Walk float tensors — F32 / F16 / BF16 all pass through
    //    verbatim (byte-copy under upstream name, no widening / no
    //    downcast, ADR A_passthrough — mirrors
    //    `qwen3_tts::convert` / `moshi::convert` / `vibevoice::convert`
    //    / `voxcpm2::convert`).
    let mut report = MeanvcReport::default();
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

    // 5. Serialise + write out (`to_bytes` appends the two unconditional
    //    `vokra.schema.*` stamps — see `vokra-core/src/gguf/writer.rs:737`).
    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Writes `bytes` to a scratch file under `std::env::temp_dir()` with
    /// a suffix that keeps parallel tests from colliding. The returned
    /// path outlives the caller — the tests explicitly `remove_file` on
    /// success and rely on the tmpfs cleaner on failure (the moshi test
    /// module uses the same pattern).
    fn tmp_path(suffix: &str) -> std::path::PathBuf {
        // The `suffix` argument is unique per test-callsite (the RED /
        // GREEN tests pass distinct human-readable names — `bf16-in`,
        // `mixed-out`, `license-in`, …) so a per-process id is enough
        // to avoid concurrent-test collisions inside `cargo test`; the
        // moshi test module (`crates/vokra-convert/src/models/moshi.rs`)
        // uses the same pattern. Deliberately avoids
        // `std::thread::ThreadId::as_u64` (nightly-only, `thread_id_value`).
        let mut p = std::env::temp_dir();
        p.push(format!("vokra-meanvc-{}-{}", std::process::id(), suffix));
        p
    }

    /// Serialises a single BF16 tensor named `"encoder.weight"` (`shape
    /// = [2, 3]`) at file offset 0 with a caller-supplied payload. The
    /// values `[1.0, -2.5, 0.15625, 3.5, -0.5, 42.0]` come out as the
    /// non-zero BF16 bit patterns qwen3_tts uses so a silent widen /
    /// downcast can't sneak past a byte-identity assert.
    fn one_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);
        let header = r#"{"encoder.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&bf16);
        (out, bf16)
    }

    /// Serialises an F32 + F16 pair. Shapes are `[2, 3]` (24 B F32) and
    /// `[2, 3]` (12 B F16); both use non-zero content so a silent-widen
    /// bug would be caught by a payload compare (added defensively even
    /// though the second test asserts on counters, not bytes).
    fn one_f32_and_one_f16_safetensors() -> Vec<u8> {
        // F32 payload = 6 × 4 = 24 B. Non-zero to avoid trivial round-trip.
        let f32_vals: [f32; 6] = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 payload = 6 × 2 = 12 B. Use raw non-zero patterns so a silent
        // downcast is loud on any byte-compare regression pin.
        let f16_bytes: Vec<u8> = [
            0x00, 0x3C, 0x00, 0xBC, 0x00, 0x40, 0x00, 0xC0, 0x00, 0x38, 0x00, 0xB8,
        ]
        .to_vec();
        // Safetensors sorts entries by data_offsets, so declare in ascending
        // offset order: F32 first ([0..24)), then F16 ([24..36)).
        let header = r#"{"encoder.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"decoder.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[24,36]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    /// STEP 1 RED anchor: a synthetic BF16 tensor round-trips through
    /// `convert_meanvc_file` as a byte-identical BF16 payload on the
    /// output GGUF (Moshi's `assert_eq!(info.dtype, GgmlType::BF16, "no
    /// convert-time widening")` fence — the safetensors.rs:728-738 pin
    /// pattern, verbatim). Fails with `unimplemented!()` in the RED
    /// phase; passes end-to-end once STEP 2 lands the pass-through body.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (blob, bf16_payload) = one_bf16_safetensors();
        let input = tmp_path("bf16-in.safetensors");
        let output = tmp_path("bf16-out.gguf");
        std::fs::write(&input, &blob).expect("write synthetic input");

        let report = convert_meanvc_file(&input, &output, None).expect("convert_meanvc_file");
        assert_eq!(report.read, 1, "safetensors reader saw 1 declaration");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must NOT land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Fence 1: dtype survives the round trip as BF16 (no convert-time
        // widening — the ADR red-line qwen3_tts pins).
        let gguf = std::fs::read(&output).expect("read output gguf");
        let file = GgufFile::parse(gguf).expect("parse gguf");
        let info = file
            .tensor_info("encoder.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        // Fence 2: raw bytes byte-identical to input payload.
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// STEP 1 RED anchor: F32 + F16 pair both pass through and increment
    /// `written` (2, 2 not 1 — guard against the "match arm collapses to
    /// one dtype" regression), while `bf16_passthrough` stays at the
    /// `Default 0` (guard against additive-counter contamination).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let blob = one_f32_and_one_f16_safetensors();
        let input = tmp_path("mixed-in.safetensors");
        let output = tmp_path("mixed-out.gguf");
        std::fs::write(&input, &blob).expect("write synthetic input");

        let report = convert_meanvc_file(&input, &output, None).expect("convert_meanvc_file");
        assert_eq!(report.read, 2, "safetensors reader saw 2 declarations");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only + F16-only input must leave the BF16 counter at Default 0"
        );

        let gguf = std::fs::read(&output).expect("read output gguf");
        let file = GgufFile::parse(gguf).expect("parse gguf");
        let f32_info = file.tensor_info("encoder.weight").expect("F32 present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        let f16_info = file.tensor_info("decoder.weight").expect("F16 present");
        assert_eq!(f16_info.dtype, GgmlType::F16);

        // Provenance survives too — the `apache-2.0` upstream default
        // wins when the caller passes `license = None`, and the ad-hoc
        // upstream_hf / model.category keys land verbatim.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Guards the `license: Option<&str>` override — when the caller
    /// passes `Some("mit")`, the raw `vokra.provenance.license` stamp
    /// switches to that SPDX id instead of the built-in `apache-2.0`
    /// default. Added defensively so the STEP 2 override wiring doesn't
    /// silently regress (there is no separate integration test).
    #[test]
    fn license_override_wins_over_upstream_default() {
        let blob = one_f32_and_one_f16_safetensors();
        let input = tmp_path("license-in.safetensors");
        let output = tmp_path("license-out.gguf");
        std::fs::write(&input, &blob).expect("write synthetic input");

        let report =
            convert_meanvc_file(&input, &output, Some("mit")).expect("convert_meanvc_file");
        assert_eq!(report.written, 2);

        let gguf = std::fs::read(&output).expect("read output gguf");
        let file = GgufFile::parse(gguf).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "license override must beat the built-in apache-2.0 default"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
