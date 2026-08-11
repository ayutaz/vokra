//! **aiola Whisper-Medusa-v1**: safetensors checkpoint → GGUF conversion
//! (coverage-audit wave-b, 2026-08-03).
//!
//! Input: the upstream `aiola/whisper-medusa-v1` release from
//! `huggingface.co/aiola/whisper-medusa-v1` — an OpenAI Whisper backbone
//! (base or small) augmented with **Medusa speculative-decoding heads**
//! (Cai et al. 2024, `arXiv:2401.10774`). Output: a GGUF carrying every
//! float tensor plus the `vokra.provenance.*` / `vokra.model.*` /
//! `vokra.schema.*` metadata chunks a future native
//! `vokra-models::whisper_medusa_v1::*` implementation will read.
//!
//! # Model class
//!
//! Whisper-Medusa is the **speculative-decoding** family: a frozen
//! Whisper encoder + Whisper decoder + N Medusa prediction heads that
//! generate N speculative future tokens per step, verified by the base
//! decoder. The runtime gain is a per-step throughput multiplier
//! (published upstream at 1.5-2×) without model-quality regression.
//! Distinct arch tag from vanilla [`models::whisper`] because the
//! **Medusa heads are an extra tensor family** the base
//! `WhisperWeights::from_gguf` walk does not know about; silently
//! aliasing `"whisper"` would either (a) drop the Medusa heads on the
//! floor at load or (b) fail a tensor-count sanity check the runtime
//! adds later — either way a wrong-shape / misrouted GGUF.
//!
//! The **runtime speculative-decoding op** (`vokra_ops::speculative_
//! decode` or equivalent) is a separate WP; this converter provides the
//! byte-parallel GGUF surface only. Ticket
//! (`docs/tickets/coverage-audit-2026-08-03/wave-b/whisper-medusa-v1.md`
//! §Converter) explicitly scopes runtime binding to a follow-up.
//!
//! # License
//!
//! The upstream ticket header names `Apache-2.0` (the aiola precedent),
//! but the primary source (`huggingface.co/aiola/whisper-medusa-v1`
//! model-card front-matter) requires owner sign-off in
//! `docs/license-audit.md` §3.1 before publish. Fail-closed default per
//! [[feedback-license-signoff-primary-source]] applies until then —
//! this converter stamps the ticket-header-derived `apache-2.0` /
//! `Permissive` default and callers who legitimately hold the weight
//! under a distinct SPDX id override at the outer
//! `convert_file --license <spdx>` boundary (the standing mechanism the
//! whisper / kokoro / wespeaker / hibiki paths all expose).
//!
//! # BF16 pass-through
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of `neucodec` / `emotion2vec` /
//! `qwen3_tts` / `vibevoice` / `voxcpm2` — the landed sibling posture
//! that keeps the CI cache footprint at the smallest tensor payload
//! while preserving the exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec / hibiki contract). Real-weight
//! parity binding — including the Medusa-head naming walk the future
//! runtime needs to route past the base Whisper decoder — is a
//! follow-up wave gated on the license §3.1 sign-off
//! (`docs/license-audit.md`); this converter passes every float tensor
//! through unchanged so a future `WhisperMedusaV1Weights::from_gguf`
//! can walk the same names.
//!
//! # No prep script needed
//!
//! Upstream `aiola/whisper-medusa-v1` ships `model.safetensors` (the
//! ticket's download command excludes `.bin` / `.pt` — the safetensors
//! is the authoritative artifact). Mirror of the hibiki / kyutai-stt
//! posture; no `.pth` bridge required. A future v2 release that ships
//! only `.pt` would grow a
//! `tools/parity/whisper_medusa_v1_prepare_checkpoint.py` bridge
//! (mirror of `dfn3_prepare_checkpoint.py` / `dac_prepare_checkpoint.py`).
//!
//! # No ONNX (permanent)
//!
//! Whisper-Medusa is distributed as safetensors + a Python / Transformers
//! pipeline; this converter **never** touches ONNX (FR-LD-05); the
//! pipeline (base Whisper decode + Medusa head speculative sampling +
//! base-decoder verification) will be re-implemented natively when a
//! `crates/vokra-models/src/whisper_medusa_v1/` module lands
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Whisper-Medusa-v1 GGUFs. Distinct from
/// vanilla [`models::whisper`]'s `"whisper"` arch tag because the
/// Medusa-head tensor family is not present in the base Whisper
/// checkpoint; silently aliasing `"whisper"` would either drop the
/// Medusa heads on the floor at load or fail a tensor-count sanity
/// check the runtime adds later (either way a wrong-shape / misrouted
/// GGUF). A hypothetical `aiola/whisper-medusa-multilingual` sibling
/// would carry a distinct arch tag (`"whisper-medusa-multilingual"`)
/// when it lands as its own `ModelKind` — the ticket header names it
/// as a candidate; this v1 (English) landing does not preempt that
/// decision.
pub const ARCH: &str = "whisper-medusa-v1";

/// `vokra.model.name` value written for the canonical Whisper-Medusa-v1
/// GGUF (mirror of the Wave B siblings — hibiki-2b, canary-1b-flash,
/// sortformer-diar-4spk-v1, etc — carrying the release id verbatim).
pub const NAME: &str = "whisper-medusa-v1";

/// `vokra.model.category` value — same `"asr"` tier as vanilla
/// Whisper / distil-whisper / kotoba-whisper / canary / parakeet /
/// parakeet-ctc / kyutai-stt / omniasr-ctc / canary-1b-flash. The
/// speculative-decoding subcategory (per the ticket's
/// `asr/speculative-decoding` slash-separated label) is a runtime
/// dispatch axis the arch tag distinguishes, not a top-level category
/// (mirror of hibiki's `"s2s"` category with the simultaneous-
/// translation subcategory resolved by arch tag).
pub const CATEGORY: &str = "asr";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of the neucodec / hibiki /
/// canary_1b_flash local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream HuggingFace repository slug (`org/name`) recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace
/// the artifact back to its serving location.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const UPSTREAM_HF: &str = "aiola/whisper-medusa-v1";

/// Canonical weight license SPDX (`apache-2.0` — the ticket-header
/// default per the aiola-lab precedent; the primary-source
/// verification is deferred to owner sign-off in
/// `docs/license-audit.md` §3.1). Overrides via the
/// [`convert_whisper_medusa_v1_file`] `license` parameter — the
/// standing mechanism for "implementation is clean-room MIT but the
/// upstream distributed checkpoint is another license" scenarios
/// (mirror of `convert_file_licensed` in `lib.rs` and the `license`
/// arg on the hibiki / canary_1b_flash / wespeaker / frcrn siblings).
pub const DEFAULT_LICENSE: &str = "apache-2.0";

/// Outcome of a Whisper-Medusa-v1 conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `WhisperMedusaV1Report::default()` and the
/// caller remains responsible for surfacing the "no float tensors"
/// loud note (mirror of the qwen3_tts / vibevoice / voxcpm2 /
/// wespeaker / emotion2vec / hibiki / canary_1b_flash `Report`
/// pattern). `read == written + skipped_non_float` is an invariant
/// preserved by [`convert_whisper_medusa_v1_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WhisperMedusaV1Report {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm signals a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is
    /// exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a
/// Whisper-Medusa-v1 GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin
/// the upstream slug, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by
/// the GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"apache-2.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed checkpoint
/// carries a different SPDX (e.g. an owner who confirms the primary-
/// source declares a stricter class).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_whisper_medusa_v1_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<WhisperMedusaV1Report, ConvertError> {
    // Whole-file read: Whisper-Medusa-v1 is ~500 MB - 2 GB safetensors
    // (Whisper base or small + N Medusa heads), well within the
    // whole-file range on any development host — no need for the
    // streaming path the Moshi 15 GB / Voxtral 8.7 GB converters run.
    // Any future >8 GB whisper-medusa sibling (e.g. medusa-large-v3)
    // would swap this call for `SafetensorsFileReader::open` +
    // `GgufStreamWriter::begin` per the moshi.rs / qwen3_tts.rs ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough).
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (ticket header, aiola-lab precedent;
    // primary-source verification is deferred to owner sign-off in
    // docs/license-audit.md §3.1). `license` overrides for callers who
    // obtained the weight under a different SPDX (see
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
            "aiola/whisper-medusa-v1 (Whisper backbone + Medusa speculative-\
             decoding heads, apache-2.0 per ticket header; primary-source \
             sign-off pending in docs/license-audit.md §3.1)",
        ),
    );

    let mut report = WhisperMedusaV1Report::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `neucodec::convert` / `emotion2vec::convert` /
    // `qwen3_tts::convert` / `vibevoice::convert` /
    // `hibiki::convert_hibiki_file` / `canary_1b_flash::convert_canary_
    // 1b_flash_file`.
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
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir
    /// (moshi / emotion2vec / wespeaker / hibiki / canary_1b_flash
    /// test pattern — no external `tempfile` dep, preserving zero-dep
    /// NFR-DS-02). The nanosecond suffix separates parallel `cargo
    /// test` runs so they cannot clobber each other's files.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-whisper-medusa-v1-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor.
    ///
    /// The payload is chosen from a known set of non-zero BF16 bit
    /// patterns so a byte-identity assert catches any silent widen /
    /// downcast attempt — the raw zeroed payload would round-trip
    /// trivially through F32 / F16 widen and defeat the pin (mirror of
    /// emotion2vec / hibiki / canary_1b_flash fixture).
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        // Use a tensor name from the Medusa-head family so the fixture
        // documents the tensor topology this converter has to survive
        // (the base Whisper walk plus the Medusa heads).
        let header = r#"{"medusa_head.0.linear.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor
    /// (`shape=[1,4]`, 8 B). The offsets are chosen so the tensors are
    /// contiguous in the data region — mirror of emotion2vec / hibiki
    /// / canary_1b_flash fixture. The tensor names span both the base
    /// Whisper encoder / decoder namespace and the Medusa-head
    /// namespace so the pass-through walk is exercised across both
    /// tensor families.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        let header = r#"{"model.encoder.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"medusa_head.1.linear.weight":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f32_bytes);
        buf.extend_from_slice(&f16_bytes);
        (buf, f32_bytes, f16_bytes)
    }

    /// BF16 pass-through: the upstream BF16 checkpoint (the primary
    /// upstream posture for a modern HF Whisper release) must survive
    /// the file-based converter round-trip with its dtype preserved
    /// (GGUF type 30 = `GgmlType::BF16`) and its payload byte-
    /// identical to the input. Mirror of the emotion2vec / hibiki /
    /// canary_1b_flash equivalent.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_whisper_medusa_v1_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of emotion2vec / hibiki)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        // Round-trip: dtype preserved, payload byte-identical (no silent widen).
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("medusa_head.0.linear.weight")
            .expect("Medusa-head BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Provenance + category chunks pinned on the artifact itself.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "arch tag must be `whisper-medusa-v1` — silently aliasing \
             `whisper` would drop the Medusa heads at load"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 must resolve to Permissive"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins the `asr` tier"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "upstream slug pins traceability back to aiola/whisper-medusa-v1"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 pass-through: two float tensors of distinct dtypes in
    /// the same input must both reach the pass-through arm without
    /// collapsing into a single dtype branch, and the BF16 counter must
    /// remain 0. Guards against a naive `if bf16 { … } else` refactor.
    /// The fixture spans both the base Whisper (encoder q_proj) and
    /// Medusa-head namespaces so both tensor families are exercised.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_whisper_medusa_v1_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved. Cross-family walk (encoder + medusa head).
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("model.encoder.layers.0.self_attn.q_proj.weight")
            .expect("base Whisper encoder F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("medusa_head.1.linear.weight")
            .expect("Medusa-head F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override: the caller-supplied SPDX must replace the
    /// default `apache-2.0` stamp on the artifact, and the class must
    /// re-derive from the override. Mirror of the wespeaker / frcrn /
    /// canary_1b_flash license-override test posture. This is the seam
    /// through which an owner who confirms a stricter class in the
    /// primary source (e.g. `cc-by-4.0` or `mit`) can restamp the
    /// artifact without editing the converter's baked-in default.
    #[test]
    fn license_override_replaces_default_stamp() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        // Override with `mit` — a distinct Permissive SPDX. The
        // override must land in `vokra.provenance.license`, the class
        // must re-derive to Permissive, and the built-in
        // `apache-2.0` default must not survive.
        convert_whisper_medusa_v1_file(&input, &output, Some("mit"))
            .expect("convert with license override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit must resolve to Permissive (same class as the default apache-2.0)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
