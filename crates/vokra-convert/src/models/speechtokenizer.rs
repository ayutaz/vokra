//! **SpeechTokenizer** — 16 kHz RVQ-8 codec with HuBERT semantic distillation
//! (SoTA plan follow-on, 2026-07-25).
//!
//! Input: the upstream `fnlp/SpeechTokenizer` release safetensors
//! checkpoint. Output: a Vokra GGUF carrying every float tensor plus
//! the minimal `vokra.model.*` / `vokra.provenance.*` metadata chunks
//! the compliance gate (FR-CP-03) and the model-zoo index consume.
//!
//! # License
//!
//! `apache-2.0` end-to-end (`huggingface.co/fnlp/SpeechTokenizer` model
//! card `license: apache-2.0`, fetched 2026-07-25 — CLAUDE.md
//! 「ハルシネーション厳禁」). The M2-13 gate passes commercially with no
//! runtime-side attribution obligation. A caller who publishes from a
//! different distribution source may override the licence at the
//! function boundary (`license: Option<&str>`, mirror of the
//! `convert_file_licensed` posture in `lib.rs`).
//!
//! # BF16 posture
//!
//! Every `F32` / `F16` / `BF16` tensor passes through **verbatim** — no
//! convert-time widening (the qwen3-tts / vibevoice / voxcpm2 / moshi /
//! voxtral pattern). BF16 stays GGUF type 30 (`GgmlType::BF16`); the
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). The
//! observability counter [`SpeechtokenizerReport::bf16_passthrough`]
//! records how many BF16 tensors landed on this arm.
//!
//! # Category
//!
//! Recorded under `vokra.model.category = "codec"` — a 16 kHz
//! Residual-Vector-Quantized codec with an 8-codebook RVQ head whose
//! first quantizer is distilled against HuBERT features so it carries
//! semantic content (the release's namesake trick).
//!
//! # Real-weight parity
//!
//! Deferred to owner (`docs/license-audit.md` §3.1 sign-off). This
//! converter guarantees byte-preserving pass-through only; a
//! `SpeechTokenizerWeights::from_gguf` native runtime binding is a
//! follow-up wave gated on the upstream tensor-name manifest fetch.
//!
//! # Dead-code suppression
//!
//! `pub fn convert_speechtokenizer_file` and its companion
//! [`SpeechtokenizerReport`] are wired only by this module's own
//! `#[cfg(test)]` block today — the outer CLI / `convert_file_licensed`
//! plumbing lands in a follow-up wave. In the non-test lib target the
//! Rust dead-code lint therefore fires on the module-private
//! constants + `pub` API items even though they are exercised end-to-end
//! by two passing tests, so we opt them out at the module level. The
//! test coverage is authoritative — the lint here would be a
//! false-positive gate.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for SpeechTokenizer GGUFs. Intentionally distinct
/// from every codec sibling (`dac` / `mimi` / `encodec_rvq`) so silently
/// sharing an arch tag cannot misroute the runtime dispatch — the
/// SpeechTokenizer first quantizer carries HuBERT-distilled semantic
/// content, which downstream ASR / TTS pipelines depend on.
pub(crate) const ARCH: &str = "speechtokenizer";

/// `vokra.model.name` — canonical release name.
pub(crate) const NAME: &str = "speechtokenizer";

/// `vokra.model.category` — categorical tag for the model-zoo index.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` — programmatic HuggingFace path of
/// the upstream release. Distinct from the existing
/// `vokra.provenance.source` chunk (advisory free-form string) so
/// tooling can programmatically resolve the upstream without regex.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "fnlp/SpeechTokenizer";

/// Default weight-license SPDX for the SpeechTokenizer release.
const DEFAULT_LICENSE: &str = "apache-2.0";

/// Outcome of a SpeechTokenizer conversion.
///
/// Mirrors the counter fields established by
/// `crates/vokra-convert/src/models/qwen3_tts.rs` for the pass-through
/// arm (`written`, `skipped_non_float`, `bf16_passthrough`) and adds a
/// [`Self::read`] counter (the number of tensors the safetensors reader
/// exposed to the loop — invariant `read == written + skipped_non_float`
/// on every well-formed input, so a regression that double-counts or
/// drops a tensor trips loudly at the test surface).
#[derive(Debug, Default)]
pub struct SpeechtokenizerReport {
    /// Tensors observed by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader currently accepts only F32 / F16 / BF16 at
    /// parse time; anything reaching this arm signals a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emitted as GGUF type 30 verbatim.
    pub bf16_passthrough: usize,
}

/// Converts a SpeechTokenizer safetensors file at `input` into a Vokra
/// GGUF written to `output`, returning a [`SpeechtokenizerReport`].
///
/// If `license` is `Some`, the raw `vokra.provenance.license` and the
/// canonical `vokra.provenance.weight_license` are overridden with the
/// caller-supplied SPDX id (right thing to do when the distribution
/// source's licence differs from the model's canonical release — same
/// posture as `convert_file_licensed` in `crates/vokra-convert/src/lib.rs`).
pub fn convert_speechtokenizer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SpeechtokenizerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);

    // Provenance: SpeechTokenizer ships apache-2.0 end-to-end (Permissive,
    // no attribution obligation). `stamp_provenance` writes the canonical
    // `vokra.provenance.weight_license` alongside the raw license string,
    // model id, and source — the licence-override branch below overwrites
    // in place (`GgufBuilder::add_string` replaces same-key entries).
    // `vokra.schema.version` / `vokra.schema.producer` stamps are injected
    // automatically at serialisation time by `GgufBuilder::to_bytes` (see
    // `crates/vokra-core/src/gguf/writer.rs::effective_metadata`), so we
    // do not — and must not — add them here (the writer strips duplicates
    // so its self-stamp always describes the actual writer build).
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE,
        Some(NAME),
        Some(UPSTREAM_HF),
    );
    // Programmatic upstream pointer (advisory `source` above is free-form).
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Licence override — mirror of `convert_file_licensed` posture in
    // `crates/vokra-convert/src/lib.rs`: re-derive the canonical class from
    // the raw SPDX string, overwrite `weight_license` + `license`, and
    // restate `source` neutrally so the built-in parenthetical (which names
    // the default licence) does not silently contradict the override.
    if let Some(lic) = license {
        let class = LicenseClass::from_license_str(lic);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, lic);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("{UPSTREAM_HF} (licence {lic} per source)"),
        );
    }

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30); the runtime widens BF16 → f32
    // exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
    // bits of an f32 — `bits << 16` is exact). Mirror of the qwen3-tts /
    // vibevoice / voxcpm2 / moshi / voxtral pass-through arm.
    let mut report = SpeechtokenizerReport::default();
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

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::{GgufFile, chunks};

    /// Returns a unique per-test tempfile path. Uses `std::env::temp_dir`
    /// + `std::process::id` + a monotonic nanosecond suffix so parallel
    /// `cargo test` invocations do not clash (the moshi.rs pattern).
    fn tempfile_path(prefix: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-speechtokenizer-{prefix}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        p
    }

    fn write_tempfile(prefix: &str, ext: &str, bytes: &[u8]) -> PathBuf {
        let p = tempfile_path(prefix, ext);
        std::fs::write(&p, bytes).expect("write tempfile");
        p
    }

    /// Builds a single-tensor BF16 safetensors buffer. Mirror of
    /// qwen3_tts's `safetensors_one_bf16` helper: keeps the tensor
    /// scaffolding hand-written (no external crate) per the crate's
    /// zero-dep contract (NFR-DS-02).
    fn safetensors_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
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

    /// Builds a two-tensor safetensors buffer: one F32 (`[1,2]`, 8 B) and
    /// one F16 (`[3]`, 6 B). Payload layout matches the JSON
    /// `data_offsets` (F32 first, F16 second).
    fn safetensors_f32_and_f16() -> Vec<u8> {
        // F32 payload: two deterministic non-zero values.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        // F16 payload: 1.0, 2.0, 3.0 as F16 bit patterns.
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f16_bytes.len(), 6);
        let header = format!(
            r#"{{"a_f32":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"b_f16":{{"dtype":"F16","shape":[3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    /// Pins the BF16 pass-through arm: the upstream tensor lands in the
    /// output GGUF with `GgmlType::BF16` dtype and byte-identical
    /// payload (no silent widen / downcast). Mirror of qwen3-tts's
    /// `bf16_tensor_passes_through_verbatim`.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 payload so any silent widen / downcast fails the
        // byte-identity assert (a zero-payload fixture would trivially
        // survive a F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);
        let input_bytes = safetensors_bf16("codec.embed.weight", &[2, 3], &bf16);
        let input = write_tempfile("bf16-in", "safetensors", &input_bytes);
        let output = tempfile_path("bf16-out", "gguf");

        let report = convert_speechtokenizer_file(&input, &output, None)
            .expect("convert_speechtokenizer_file must succeed for a BF16 tensor");
        let _ = std::fs::remove_file(&input);

        assert_eq!(report.read, 1, "safetensors exposed 1 tensor");
        assert_eq!(report.written, 1, "BF16 lands on the pass-through arm");
        assert_eq!(report.skipped_non_float, 0, "BF16 is a float");
        assert_eq!(
            report.bf16_passthrough, 1,
            "the one BF16 tensor increments the observability counter"
        );

        let gguf_bytes = std::fs::read(&output).expect("read output GGUF");
        let _ = std::fs::remove_file(&output);
        let file = GgufFile::parse(gguf_bytes).expect("parse output GGUF");

        let info = file
            .tensor_info("codec.embed.weight")
            .expect("BF16 tensor is present in the output GGUF");
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
            "BF16 payload byte-identical to input (no silent widen)"
        );
    }

    /// Pins the F32 + F16 pass-through legs of the match arm: both
    /// dtypes reach the pass-through arm, counters are correct (written
    /// = 2, bf16_passthrough = 0), and the required provenance /
    /// category metadata are present.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let input_bytes = safetensors_f32_and_f16();
        let input = write_tempfile("f32f16-in", "safetensors", &input_bytes);
        let output = tempfile_path("f32f16-out", "gguf");

        let report = convert_speechtokenizer_file(&input, &output, None)
            .expect("convert_speechtokenizer_file must succeed for F32 + F16");
        let _ = std::fs::remove_file(&input);

        assert_eq!(report.read, 2, "safetensors exposed 2 tensors");
        assert_eq!(report.written, 2, "both F32 and F16 pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0, "no BF16 in this input");

        let gguf_bytes = std::fs::read(&output).expect("read output GGUF");
        let _ = std::fs::remove_file(&output);
        let file = GgufFile::parse(gguf_bytes).expect("parse output GGUF");

        // Provenance & category metadata are present with the defaults.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "default weight license",
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY),
        );

        // Dtypes preserved verbatim.
        let a = file.tensor_info("a_f32").expect("F32 tensor present");
        assert_eq!(a.dtype, GgmlType::F32);
        let b = file.tensor_info("b_f16").expect("F16 tensor present");
        assert_eq!(b.dtype, GgmlType::F16);
    }
}
