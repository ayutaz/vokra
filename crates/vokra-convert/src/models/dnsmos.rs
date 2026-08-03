//! **DNSMOS P.808 / P.835** (Microsoft DNS-Challenge MOS predictors):
//! prepared safetensors → GGUF conversion (coverage-audit Wave A ticket
//! `dnsmos-p808-p835`, 2026-08-03).
//!
//! # Model class
//!
//! DNSMOS is Microsoft's neural MOS (Mean Opinion Score) predictor released
//! under MIT as part of the **DNS-Challenge** repository
//! (`github.com/microsoft/DNS-Challenge`). It ships two ONNX checkpoints:
//!
//! * `model_v8.onnx` — the **P.808** predictor (single overall quality
//!   scalar, ITU-T P.808 scale).
//! * `sig_bak_ovr.onnx` — the **P.835** predictor (three scalars: signal
//!   quality / background noise / overall, ITU-T P.835 scale).
//!
//! This is Vokra's first `category = "eval"` model (the sibling UTMOS
//! predictor in `models::utmos` has been the informal reference for the
//! `vokra-eval` NFR-QL-02 5 % quality gate; DNSMOS joins it as an
//! independent MOS oracle). The runtime side is `vokra_eval::dnsmos::{
//! p808_score, p835_score}` (deferred CC-implementation follow-up — this
//! ticket lands only the offline **converter** contract).
//!
//! # ONNX bridge — offline only, permanent
//!
//! DNSMOS is distributed as ONNX only. Per **FR-LD-05** the runtime never
//! loads ONNX, so the checkpoint enters Vokra through the offline
//! `tools/parity/dnsmos_prepare_checkpoint.py` sidecar which flattens both
//! ONNX graphs into a **single merged safetensors** with the tensor names
//! prefixed by the bundle variant (`p808.<upstream_name>` /
//! `p835.<upstream_name>`) so the runtime binder can walk both models from
//! the same GGUF without a graph load. The prefixing scheme mirrors the
//! `mimi` / `csm` internal-namespace convention (`mimi.enc.*` /
//! `mimi.dec.*`) — a single conversion step, a single artifact, two
//! runtime consumers.
//!
//! # BF16 pass-through — n/a (DNSMOS ships F32)
//!
//! DNSMOS weights are F32 in both the ONNX release and the prep script's
//! safetensors output (there is no BF16 variant published). The pass-
//! through arm still accepts F32 / F16 / BF16 for symmetry with the
//! sibling BF16-fleet converters (openvoice_v2 / ecapa_tdnn / etc.),
//! and the `bf16_passthrough` counter stays at its Default 0 in normal
//! operation — a nonzero value would signal an upstream reformat that
//! the caller must audit.
//!
//! # License
//!
//! End-to-end **MIT** (`github.com/microsoft/DNS-Challenge/blob/master/LICENSE`
//! = MIT with `Copyright (c) Microsoft Corporation`, fetched 2026-08-03
//! — CLAUDE.md「ハルシネーション厳禁」). MIT is a `Permissive` class —
//! no runtime-side attribution obligation. The `license` override lets a
//! downstream repackager stamp a different SPDX if they redistribute
//! under stricter terms (the same knob `convert_file_licensed` exposes
//! in `lib.rs`).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **prepared safetensors names verbatim**
//! (prefixed `p808.<upstream>` / `p835.<upstream>` by the sidecar). The
//! future `vokra_eval::dnsmos::from_gguf` walks the two prefixes as
//! independent sub-models; a single GGUF publishes both scores without a
//! second artifact upload (bundle option (a) in the coverage-audit ticket
//! §Converter — "2 ONNX を単一 GGUF に merge、bundle metadata で variant
//! tag").

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value written for DNSMOS GGUFs. Distinct from
/// every sibling arch tag (in particular from `utmos`, which shares the
/// `"eval"` category but has a wav2vec 2.0 backbone rather than a
/// standalone CNN + LSTM MOS predictor) — silently sharing an arch tag
/// would misroute the runtime dispatch.
pub const ARCH: &str = "dnsmos";

/// `vokra.model.name` value written for the canonical DNSMOS bundle GGUF
/// (both P.808 and P.835 in a single file).
pub const NAME: &str = "dnsmos-p808-p835";

/// `vokra.model.category` value — `"eval"` (evaluation / MOS predictor
/// tier, the first entry in the converter tree). Consumed by the model-
/// card generator and zoo manifest tier gate so an eval oracle is not
/// accidentally advertised as an ASR / TTS release.
pub const CATEGORY: &str = "eval";

/// `vokra.provenance.upstream_url` value — the primary redistribution
/// source. DNSMOS is not on Hugging Face; it ships from the Microsoft
/// DNS-Challenge GitHub repository under `DNSMOS/`.
pub const UPSTREAM_URL: &str = "https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS";

/// Canonical weight license SPDX. Overridable via the [`convert_dnsmos_file`]
/// `license` parameter (the same knob `convert_file_licensed` in `lib.rs`
/// exposes for the "implementation is clean-room MIT but the redistributed
/// checkpoint is another license" scenario — irrelevant for DNSMOS itself
/// but preserved for consistency with the file-based sibling converters).
pub const DEFAULT_LICENSE: &str = "mit";

/// Metadata key: model category tag (`"tts"` / `"asr"` / `"vad"` /
/// `"s2s"` / `"vc"` / `"speaker"` / `"codec"` / `"emotion"` / `"bert"` /
/// `"eval"`). Ad-hoc converter-side constant — not yet a first-class
/// `chunks::KEY_*` alias in `vokra-core`. Same convention every recent
/// file-based converter (`openvoice_v2`, `ecapa_tdnn`, `emotion2vec`,
/// `funcodec`, …) uses.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key: raw upstream source URL. Distinct from
/// `chunks::KEY_PROVENANCE_SOURCE` (a longer human-readable label) —
/// this key is the machine-parseable URL a downstream tool can re-fetch
/// without guessing. Sibling of `vokra.provenance.upstream_hf` (used by
/// every HF-mirrored converter) — this variant fits GitHub-native
/// releases like DNSMOS that never had an HF mirror.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Metadata key: bundle inventory. A U32-string array naming the
/// sub-models present in the GGUF (in canonical order: `["p808",
/// "p835"]`). A runtime consumer walks this list to know which
/// tensor-name prefixes are bindable; a partial bundle (only P.808 or
/// only P.835 flattened) advertises the truthful subset here so the
/// binder fails loudly rather than silently on the missing half
/// (FR-EX-08).
const KEY_DNSMOS_BUNDLE: &str = "vokra.dnsmos.bundle";

/// Metadata key: sample rate of the model's audio front-end (16 000 Hz
/// for both DNSMOS variants — same as UTMOS). Written unconditionally so
/// a downstream binder can validate its resampler before the tensor
/// walk.
const KEY_DNSMOS_SAMPLE_RATE: &str = "vokra.dnsmos.sample_rate";

/// Metadata key: the P.808 upstream checkpoint filename recorded on the
/// GGUF for auditability. Written only when the bundle contains P.808.
const KEY_DNSMOS_P808_CKPT: &str = "vokra.dnsmos.p808.checkpoint";

/// Metadata key: the P.835 upstream checkpoint filename recorded on the
/// GGUF for auditability. Written only when the bundle contains P.835.
const KEY_DNSMOS_P835_CKPT: &str = "vokra.dnsmos.p835.checkpoint";

/// Outcome of a DNSMOS conversion.
///
/// Mirrors the recent file-based sibling report shape
/// ([`super::openvoice_v2::OpenvoiceV2Report`] /
/// [`super::ecapa_tdnn::EcapaTdnnReport`] /
/// [`super::emotion2vec::Emotion2vecReport`]) with an added
/// `bundle_variants` counter so the caller can distinguish "both models
/// present" from "single-variant partial bundle" without walking the
/// GGUF metadata.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DnsmosReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a non-zero
    /// value here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). DNSMOS ships F32 upstream so this counter
    /// stays at `0` in normal operation; a non-zero value flags an
    /// upstream reformat.
    pub bf16_passthrough: usize,
    /// Bundle variants detected in the prepared safetensors: 1 = single-
    /// model partial bundle (only P.808 or only P.835), 2 = full bundle
    /// (both). A run that produces 0 (no `p808.` or `p835.` prefix
    /// present) is a hard error — the caller pointed at a safetensors
    /// file that is not a DNSMOS bundle.
    pub bundle_variants: usize,
}

/// Reads a prepared DNSMOS safetensors bundle at `input` and writes a
/// DNSMOS GGUF to `output`.
///
/// The input is expected to be the output of
/// `tools/parity/dnsmos_prepare_checkpoint.py`: a single safetensors
/// file whose tensor names carry the `p808.` and/or `p835.` prefix
/// identifying the sub-model each tensor belongs to. Every F32 / F16 /
/// BF16 tensor is emitted verbatim under its prefixed name; the
/// `vokra.model.*` / `vokra.provenance.*` / `vokra.dnsmos.*` chunk
/// groups pin the artifact's identity for the runtime compliance gate
/// (FR-CP-03) and the future `vokra_eval::dnsmos::from_gguf` binder.
///
/// `license` overrides [`DEFAULT_LICENSE`] (`"mit"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when a redistributed
/// checkpoint carries a different SPDX.
///
/// # Errors
///
/// * [`ConvertError::Io`] on read/write failure.
/// * [`ConvertError::Parse`] on a malformed safetensors input or on a
///   bundle that carries neither `p808.` nor `p835.` prefixed tensors
///   (the caller supplied a non-DNSMOS safetensors file).
/// * A GGUF writer failure surfaces as [`ConvertError::Gguf`] via the
///   `From<gguf::WriterError>` impl.
pub fn convert_dnsmos_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<DnsmosReport, ConvertError> {
    // Whole-file read: DNSMOS is ~10 MiB combined (both ONNX flattened
    // to safetensors) — orders of magnitude smaller than the streaming-
    // mandated Moshi 14 GiB / Voxtral 8.7 GiB tier, so the simple
    // `std::fs::read` posture the sibling non-streaming converters
    // (openvoice_v2 / ecapa_tdnn / emotion2vec / …) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    // Detect which bundle variants the safetensors carries: at least
    // one `p808.` or `p835.` prefixed tensor must exist. A bundle with
    // neither prefix is a hard error rather than an empty conversion —
    // the caller almost certainly pointed at the wrong file.
    let mut has_p808 = false;
    let mut has_p835 = false;
    for t in st.tensors() {
        if t.name.starts_with("p808.") {
            has_p808 = true;
        }
        if t.name.starts_with("p835.") {
            has_p835 = true;
        }
    }
    if !has_p808 && !has_p835 {
        return Err(ConvertError::Parse(
            "dnsmos: prepared safetensors carries neither `p808.` nor `p835.` prefixed \
             tensors — the input does not look like a DNSMOS bundle produced by \
             `tools/parity/dnsmos_prepare_checkpoint.py`. Refusing to emit an empty \
             GGUF (FR-EX-08)"
                .to_owned(),
        ));
    }

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Default provenance stamp — Permissive MIT end-to-end (upstream
    // `microsoft/DNS-Challenge/LICENSE` verified 2026-08-03). The
    // optional `license` argument overrides below.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE,
        Some(NAME),
        Some("microsoft/DNS-Challenge/DNSMOS (MIT)"),
    );

    // Bundle inventory — the runtime binder walks this to know which
    // sub-model prefixes are advertised. Written in canonical order
    // (`p808` before `p835`) so an equality check across two conversions
    // of the same bundle is stable.
    let mut bundle: Vec<GgufMetadataValue> = Vec::new();
    if has_p808 {
        bundle.push(GgufMetadataValue::String("p808".to_owned()));
    }
    if has_p835 {
        bundle.push(GgufMetadataValue::String("p835".to_owned()));
    }
    b.add_metadata(
        KEY_DNSMOS_BUNDLE,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: bundle,
        }),
    );
    // Both DNSMOS variants operate on 16 kHz PCM (the same rate UTMOS
    // and every current Vokra MOS oracle uses). Written unconditionally
    // so a downstream binder can validate its resampler before the
    // tensor walk. If a future DNSMOS revision changes the audio front-
    // end the prep script should stamp the value — until then a
    // constant is honest.
    b.add_u32(KEY_DNSMOS_SAMPLE_RATE, 16_000);
    // Per-variant provenance: record the upstream checkpoint filenames
    // so a downstream auditor can trace an emitted score back to the
    // Microsoft release's exact `.onnx` file without a separate
    // manifest lookup.
    if has_p808 {
        b.add_string(KEY_DNSMOS_P808_CKPT, "model_v8.onnx");
    }
    if has_p835 {
        b.add_string(KEY_DNSMOS_P835_CKPT, "sig_bak_ovr.onnx");
    }

    let mut report = DnsmosReport {
        bundle_variants: usize::from(has_p808) + usize::from(has_p835),
        ..DnsmosReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (mirror of qwen3_tts / vibevoice / voxcpm2 / moshi); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
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

    // Optional weight-license override — mirrors the outer
    // `convert_file_licensed` (lib.rs) branch so both a Vokra-CLI caller
    // and a direct `convert_dnsmos_file` caller land the same
    // provenance surface for the same SPDX string. Restates the source
    // neutrally so it does not contradict the stamped default's
    // parenthetical.
    if let Some(lic) = license {
        let class = LicenseClass::from_license_str(lic);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, lic);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("{UPSTREAM_URL} (licence {lic} per source)"),
        );
    }

    // Serialize and land the emitted GGUF at `output`. `to_bytes()`
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its
    // own via the writer's built-in schema stamper — no per-converter
    // duplication needed.
    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + tag + nanosecond suffix) so
    /// two parallel `cargo test` runs never collide on the same file.
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-dnsmos-{}-{}-{}.{}",
            std::process::id(),
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            ext,
        ));
        p
    }

    /// Builds a minimal DNSMOS-like safetensors buffer with one tensor
    /// per bundle variant (both `p808.` and `p835.` prefixed) so the
    /// converter's `has_p808 && has_p835` bundle detection path is
    /// exercised. F32 tensors match the upstream DNSMOS release's
    /// native dtype.
    fn safetensors_full_bundle() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // Two non-trivial payloads so a silent byte-level corruption
        // would flip a fence rather than round-trip a zeroed buffer.
        let p808_vals: [f32; 4] = [1.0, -2.5, 3.5, -0.25];
        let p808_bytes: Vec<u8> = p808_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(p808_bytes.len(), 16);
        let p835_vals: [f32; 6] = [0.5, 1.5, 2.5, -1.0, -3.0, 42.0];
        let p835_bytes: Vec<u8> = p835_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(p835_bytes.len(), 24);
        // Header layout: `p808.model_v8.conv1.weight` @ [0..16),
        // `p835.sig_bak_ovr.conv1.weight` @ [16..40).
        let header = r#"{"p808.model_v8.conv1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"p835.sig_bak_ovr.conv1.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[16,40]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&p808_bytes);
        buf.extend_from_slice(&p835_bytes);
        (buf, p808_bytes, p835_bytes)
    }

    /// Builds a partial bundle (only `p808.` tensors) so the converter's
    /// single-variant branch is exercised. The bundle inventory
    /// metadata must reflect the truthful subset (only `"p808"`).
    fn safetensors_p808_only() -> Vec<u8> {
        let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let payload: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"p808.model_v8.dense.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);
        buf
    }

    /// Builds a safetensors buffer whose tensors carry neither `p808.`
    /// nor `p835.` prefix so the converter's "not a DNSMOS bundle" hard
    /// error path is exercised (a naive "just pass everything through"
    /// implementation would silently accept a non-DNSMOS file).
    fn safetensors_no_prefix() -> Vec<u8> {
        let vals: [f32; 2] = [1.0, 2.0];
        let payload: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"random.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);
        buf
    }

    /// Full bundle round-trip: both `p808.` and `p835.` prefixed
    /// tensors survive the converter's file → file pipeline with their
    /// dtypes and payloads byte-identical, and the emitted GGUF carries
    /// the identifying `vokra.model.*` / `vokra.dnsmos.*` metadata.
    #[test]
    fn full_bundle_round_trips_verbatim() {
        let (input_bytes, p808_payload, p835_payload) = safetensors_full_bundle();
        let input = scratch_path("full", "safetensors");
        let output = scratch_path("full", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_dnsmos_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visited");
        assert_eq!(report.written, 2, "both tensors pass through");
        assert_eq!(report.skipped_non_float, 0, "no non-float in input");
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only input must leave the BF16 counter at Default 0"
        );
        assert_eq!(report.bundle_variants, 2, "both variants detected");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");

        // Provenance / identity chunks pinned on the artifact itself.
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
            Some(CATEGORY),
            "category chunk pins the first `eval` model in the converter tree"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
        );
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
        // Schema stamps are written unconditionally by the GGUF writer.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        // Bundle inventory carries both variants in canonical order.
        let bundle = file
            .get(KEY_DNSMOS_BUNDLE)
            .and_then(|v| v.as_array())
            .expect("bundle inventory present");
        assert_eq!(bundle.values.len(), 2, "both variants advertised");
        assert_eq!(bundle.values[0].as_str(), Some("p808"));
        assert_eq!(bundle.values[1].as_str(), Some("p835"));

        // Sample rate + per-variant checkpoint filenames.
        assert_eq!(
            file.get(KEY_DNSMOS_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(16_000)
        );
        assert_eq!(
            file.get(KEY_DNSMOS_P808_CKPT).and_then(|v| v.as_str()),
            Some("model_v8.onnx")
        );
        assert_eq!(
            file.get(KEY_DNSMOS_P835_CKPT).and_then(|v| v.as_str()),
            Some("sig_bak_ovr.onnx")
        );

        // Both tensors survive the round-trip byte-identical.
        let p808 = file
            .tensor_info("p808.model_v8.conv1.weight")
            .expect("p808 tensor present");
        assert_eq!(p808.dtype, GgmlType::F32);
        assert_eq!(p808.dimensions, vec![2, 2]);
        assert_eq!(file.tensor_bytes(p808), p808_payload.as_slice());
        let p835 = file
            .tensor_info("p835.sig_bak_ovr.conv1.weight")
            .expect("p835 tensor present");
        assert_eq!(p835.dtype, GgmlType::F32);
        assert_eq!(p835.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(p835), p835_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// A partial bundle (only P.808 present) advertises only `"p808"`
    /// in the bundle inventory — the P.835 checkpoint filename metadata
    /// key is absent so a downstream binder cannot silently fall back
    /// to a stale value from a previous conversion.
    #[test]
    fn partial_bundle_reports_only_present_variant() {
        let input_bytes = safetensors_p808_only();
        let input = scratch_path("partial", "safetensors");
        let output = scratch_path("partial", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_dnsmos_file(&input, &output, None).expect("convert");
        assert_eq!(report.bundle_variants, 1, "single variant detected");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");

        let bundle = file
            .get(KEY_DNSMOS_BUNDLE)
            .and_then(|v| v.as_array())
            .expect("bundle inventory present");
        assert_eq!(bundle.values.len(), 1);
        assert_eq!(bundle.values[0].as_str(), Some("p808"));

        assert_eq!(
            file.get(KEY_DNSMOS_P808_CKPT).and_then(|v| v.as_str()),
            Some("model_v8.onnx")
        );
        assert!(
            file.get(KEY_DNSMOS_P835_CKPT).is_none(),
            "P.835 checkpoint key must be absent when bundle has no P.835 tensors"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// A safetensors that carries neither `p808.` nor `p835.` prefixed
    /// tensors is a hard error (FR-EX-08) — silently emitting an empty
    /// GGUF would let a caller who pointed at the wrong file ship a
    /// meaningless artifact.
    #[test]
    fn non_dnsmos_bundle_is_a_hard_error() {
        let input_bytes = safetensors_no_prefix();
        let input = scratch_path("noprefix", "safetensors");
        let output = scratch_path("noprefix", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input");

        let err = convert_dnsmos_file(&input, &output, None)
            .expect_err("must reject a non-DNSMOS bundle");
        let msg = format!("{err}");
        assert!(msg.contains("dnsmos"), "error must name the model: {msg}");
        assert!(
            msg.contains("p808") && msg.contains("p835"),
            "error must name the expected prefixes: {msg}"
        );

        std::fs::remove_file(&input).ok();
        // The output file should NOT have been created — the hard error
        // fires before any write.
        assert!(
            !output.exists(),
            "hard error must not leak a partially-written GGUF"
        );
    }

    /// The `license` override propagates to the emitted GGUF and re-
    /// derives the `LicenseClass` from the caller-supplied SPDX — the
    /// same behaviour `convert_file_licensed` provides at the outer
    /// dispatch surface.
    #[test]
    fn license_override_reaches_the_emitted_gguf() {
        let (input_bytes, _, _) = safetensors_full_bundle();
        let input = scratch_path("licover", "safetensors");
        let output = scratch_path("licover", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input");

        convert_dnsmos_file(&input, &output, Some("apache-2.0"))
            .expect("convert with license override");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override propagates to the raw SPDX key"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "class is re-derived from the override (apache-2.0 → Permissive)"
        );
        // Restated source string names the license neutrally.
        let source = file
            .get(chunks::KEY_PROVENANCE_SOURCE)
            .and_then(|v| v.as_str())
            .expect("source restated on override");
        assert!(
            source.contains("apache-2.0"),
            "source names the override: {source}"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
