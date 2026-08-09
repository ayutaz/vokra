//! NVIDIA **Nemotron-Speech-Streaming-v2603**: safetensors checkpoint →
//! GGUF conversion (coverage-audit 2026-08-03 Wave B ticket
//! `nemotron-speech-streaming-v2603`).
//!
//! Input: an upstream `nvidia/nemo/nemotron_speech_streaming_v2603` release
//! (the release ships `.nemo`, which the operator flattens to
//! safetensors offline via `tools/parity/nemo_pt_to_safetensors.py`
//! before invoking this converter — same posture as `canary` /
//! `parakeet` / `parakeet_ctc`). Output: a GGUF carrying every F32 /
//! F16 / BF16 tensor verbatim plus the `vokra.provenance.*` /
//! `vokra.model.*` / `vokra.schema.*` metadata chunks a future native
//! `vokra-models::nemotron_speech_streaming_v2603::*` implementation
//! will read.
//!
//! # Model class
//!
//! Streaming ASR across 40 languages, a **2026-03 (v2603) release** of
//! the NVIDIA NeMo Nemotron speech family. Topology is FastConformer
//! encoder + streaming cache-aware output — a member of the sibling
//! Parakeet / Canary FastConformer family (`vokra_ops::conformer` +
//! `Stacking { factor: N }` will cover the encoder), with a streaming
//! output head that consumes the same encoder features. This converter
//! is the byte-parallel GGUF surface for that pipeline; the real-weight
//! numeric axes (encoder layer count, `hidden_size`,
//! `attention_bias`, `subsampling_factor`, `num_mel_bins`,
//! streaming chunk hparams, blank-token id) are transcribed on the
//! owner-facing follow-up wave once the first primary-source
//! `config.json` (embedded in the `.nemo` yaml) is fetched — the
//! landing skeleton deliberately does not invent axes.
//!
//! # Why a distinct `ModelKind` variant
//!
//! A distinct arch tag + `ModelKind` for the v2603 release, rather than
//! a shared branch inside a future sibling `nemotron_asr` module, is
//! chosen because:
//!
//! * v2603 is a **release-id pinned** SKU — the same pattern
//!   [`crate::models::chatterbox`] / [`crate::models::chatterbox_turbo`] /
//!   [`crate::models::chatterbox_nano`] use for release-tagged
//!   fine-tunes. The v2603 axis reshape (40-language corpus + streaming
//!   cache) is training-side and will not be interchangeable with a
//!   future v2612 or a non-streaming sibling.
//! * The `docs/license-audit.md §3.1` sign-off row for
//!   `nemotron-speech-streaming-v2603` is separate from any future
//!   sibling `nemotron_asr` row; the owner-facing publish pipeline
//!   gates on the exact SKU id the CLI accepted (`--model
//!   nemotron-speech-streaming-v2603`).
//! * A silent alias with a future `nemotron_asr` arm that hard-codes
//!   axes from another release would misroute this v2603 checkpoint at
//!   `from_gguf` load time and produce wrong-shape reads before the
//!   runtime's FR-EX-08 gate fires.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec / parakeet / parakeet_ctc /
//! canary contract). Real-weight parity binding is a follow-up wave
//! gated on the upstream tensor-name manifest fetch + license §3.1
//! sign-off (`docs/license-audit.md`); this converter passes every
//! float tensor through unchanged so a future
//! `NemotronSpeechStreamingV2603Weights::from_gguf` can walk the same
//! names.
//!
//! # BF16 posture
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Per the qwen3-tts ADR
//! (`docs/adr/qwen3-tts-bf16.md`, strategy `A_passthrough`, Accepted
//! 2026-07-25) the release BF16 checkpoint stays bit-identical through
//! the offline pipe. Mirror of `parakeet_ctc` (the sibling 1.1B CTC
//! variant) which lands the same BF16 pass-through skeleton.
//!
//! # License
//!
//! Both code and weights ship **Apache-2.0** (NVIDIA NeMo standard,
//! primary-source verified in the `.nemo` tarball's LICENSE — see the
//! ticket's "License SPDX" entry
//! `docs/tickets/coverage-audit-2026-08-03/wave-b/nemotron-speech-streaming-v2603.md`).
//! Apache-2.0 is a `Permissive` license class in
//! `crates/vokra-core/src/compliance/license_class.rs` — no
//! runtime-side attribution obligation (unlike the sibling CC-BY 4.0
//! Parakeet / Canary rows).
//!
//! # Distribution: NGC (no HF mirror at land time)
//!
//! Upstream ships from **NGC** (NVIDIA GPU Cloud catalog) —
//! `catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/
//! nemotron_speech_streaming_v2603`. There is **no HuggingFace mirror
//! confirmed at land time**, so provenance is recorded on
//! `vokra.provenance.upstream_url` (per the orchestrator wire-up
//! contract "HF mirror あり → upstream_hf / GitHub / NGC / ModelScope
//! only → upstream_url"). If NVIDIA later publishes a
//! `nvidia/nemotron-speech-streaming-v2603` HF mirror, the mirror slug
//! is added under `vokra.provenance.upstream_hf` in a follow-up commit
//! and the shared verify arm picks it up automatically — the
//! `upstream_url` stamp stays alongside so the NGC canonical serving
//! location remains traceable.
//!
//! # Prep script bridge
//!
//! Upstream ships `.nemo` (tar.gz of yaml + ckpt); the operator runs
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed, Python 3.12,
//! part of the existing tools/parity venv) to flatten it to safetensors
//! before this converter runs — the same posture the sibling
//! [`crate::models::canary`] and [`crate::models::parakeet_ctc`] NeMo
//! converters use. **No new prep script is added** for this variant:
//! `nemo_pt_to_safetensors.py` is family-generic. The runtime never
//! sees Python / torch (FR-LD-05).
//!
//! # No ONNX (permanent)
//!
//! Nemotron ships as `.nemo` (tar.gz of yaml + PyTorch pickle) — this
//! converter **never** touches ONNX (FR-LD-05); the ASR pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/nemotron_speech_streaming_v2603/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Wiring status
//!
//! Fully wired: `ModelKind::NemotronSpeechStreamingV2603` +
//! `from_arg("nemotron-speech-streaming-v2603" | …)` + `as_arg() ==
//! "nemotron-speech-streaming-v2603"` + `convert_file_licensed`
//! dispatch arm all land with this module (mirror of the parakeet-tdt-
//! 1.1b / frcrn / rnnoise / nsnet2 / dnsmos wiring pattern).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Nemotron-Speech-Streaming-v2603 GGUFs.
/// Distinct from every sibling arch tag — silently aliasing with a
/// future generic `"nemotron-asr"` (release-agnostic) would misroute
/// the runtime dispatch because a downstream `nemotron_asr` arm may
/// hard-code non-v2603 axes. The `_v2603` release suffix pins the SKU
/// on the arch tag itself so a downstream reader can dispatch without
/// a second hparam lookup (mirror of the `_1_1b` size suffix
/// `parakeet-tdt-1_1b` uses).
pub const ARCH: &str = "nemotron-speech-streaming-v2603";

/// `vokra.model.name` value written for the canonical
/// Nemotron-Speech-Streaming-v2603 GGUF. Matches the future
/// `huggingface.co/vokra/nemotron-speech-streaming-v2603` publish slug
/// and the `as_arg` return value in `lib.rs` so the CLI /
/// model-card / publish pipe all agree on a single identifier.
pub const NAME: &str = "nemotron-speech-streaming-v2603";

/// `vokra.model.category` value — an `asr` model (40-language
/// streaming ASR). Consumed by the model-card generator + zoo manifest
/// tier gate so a downstream picks the correct decode path.
pub const CATEGORY: &str = "asr";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of the wespeaker /
/// speaker_3d / emotion2vec / frcrn / rnnoise local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream canonical URL (NGC catalog).
/// Distinct from `vokra.provenance.upstream_hf` — Nemotron ships from
/// NGC only (no HuggingFace mirror confirmed at land time), so the URL
/// key records the NGC catalog path directly. Per the orchestrator
/// wire-up contract: "HF mirror あり → upstream_hf / GitHub / NGC /
/// ModelScope only → upstream_url".
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Value written to [`KEY_PROVENANCE_UPSTREAM_URL`] — the canonical
/// NVIDIA NGC catalog URL. Fetched from the ticket
/// (`docs/tickets/coverage-audit-2026-08-03/wave-b/
/// nemotron-speech-streaming-v2603.md`).
pub const UPSTREAM_URL: &str =
    "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/nemotron_speech_streaming_v2603";

/// Canonical weight license SPDX (`apache-2.0`). Overrides via the
/// [`convert_nemotron_speech_streaming_v2603_file`] `license` parameter
/// — the standing mechanism for "implementation is clean-room but the
/// upstream distributed checkpoint is another SPDX" scenarios (mirror
/// of `convert_file_licensed` in `lib.rs` and the `license` arg on
/// `convert_parakeet_tdt_1_1b_file` / `convert_frcrn_file` /
/// `convert_rnnoise_file`).
pub const DEFAULT_LICENSE: &str = "apache-2.0";

/// Outcome of a Nemotron-Speech-Streaming-v2603 conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns
/// `NemotronSpeechStreamingV2603Report::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the parakeet_tdt_1_1b / frcrn / rnnoise / qwen3_tts /
/// vibevoice `Report` pattern). `read == written + skipped_non_float`
/// is an invariant preserved by
/// [`convert_nemotron_speech_streaming_v2603_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NemotronSpeechStreamingV2603Report {
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
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a
/// Nemotron-Speech-Streaming-v2603 GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin the
/// upstream URL, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by the
/// GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"apache-2.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX. The class is re-derived from the override string via
/// [`LicenseClass::from_license_str`] so an override to `mit` /
/// `cc-by-4.0` correctly re-tags to `Permissive` / `AttributionRequired`
/// rather than staying on the default `Permissive` class.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_nemotron_speech_streaming_v2603_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NemotronSpeechStreamingV2603Report, ConvertError> {
    // Whole-file read: Nemotron-Speech-Streaming-v2603 ships as
    // ~1.2-2 GB safetensors — comfortably below the 8 GB threshold at
    // which the moshi / voxtral streaming path becomes mandatory (per
    // memory [[feedback-large-models-on-vast-ai]] which flags >8 GB as
    // vast.ai-preferred; 1.2-2 GB is Mac-tight but safe). Any future
    // Nemotron 3B+ variant that grows past 8 GB would swap this call
    // for `SafetensorsFileReader::open` + `GgufStreamWriter::begin`
    // per the moshi.rs / voxtral.rs ADR.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (NVIDIA NeMo standard — the .nemo
    // tarball's LICENSE, primary-source verified per the wave-b
    // ticket). `license` overrides for callers who obtained the weight
    // under a different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_URL));

    let mut report = NemotronSpeechStreamingV2603Report::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `parakeet_ctc::convert` / `frcrn::convert` / `rnnoise::convert`.
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

    /// Per-process, per-test scratch path in the system temp dir (frcrn
    /// / rnnoise / emotion2vec / parakeet_tdt_1_1b test pattern — no
    /// external `tempfile` dep, preserving zero-dep NFR-DS-02). The
    /// nanosecond suffix separates parallel `cargo test` runs so they
    /// cannot clobber each other's files.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-nemotron-speech-streaming-v2603-{}-{}-{}.bin",
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
    /// downcast attempt — a zeroed payload would round-trip trivially
    /// through F32 / F16 widen and defeat the pin (mirror of frcrn's
    /// fixture).
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        // Tensor name modelled on the Nemotron FastConformer encoder
        // topology (`encoder.blocks.0.attn.qkv_proj.weight`, per the
        // parakeet.rs / parakeet_ctc.rs / canary.rs test fixtures) —
        // the shape here is a stand-in `[2, 3]` for the synthetic
        // pass-through pin; the real prep-script tensor names are the
        // follow-up.
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor
    /// (`shape=[1,4]`, 8 B). The offsets are chosen so the tensors are
    /// contiguous in the data region — mirror of frcrn / rnnoise's
    /// fixtures.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        // Tensor names track the Nemotron FastConformer encoder +
        // streaming head topology (`encoder.blocks.0.mlp.fc1.weight` /
        // `decoder.output_layer.weight`); shapes are synthetic stand-
        // ins for the pass-through pin.
        let header = r#"{"encoder.blocks.0.mlp.fc1.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"decoder.output_layer.weight":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f32_bytes);
        buf.extend_from_slice(&f16_bytes);
        (buf, f32_bytes, f16_bytes)
    }

    /// BF16 pass-through: the upstream BF16 checkpoint must survive the
    /// file-based converter round-trip with its dtype preserved (GGUF
    /// type 30 = `GgmlType::BF16`) and its payload byte-identical to the
    /// input. Mirror of the frcrn / rnnoise / neucodec / emotion2vec /
    /// parakeet_tdt_1_1b equivalent.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report =
            convert_nemotron_speech_streaming_v2603_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of parakeet_ctc / parakeet_tdt_1_1b)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        assert_eq!(
            report.read,
            report.written + report.skipped_non_float,
            "read = written + skipped invariant (mirror of qwen3_tts pattern)"
        );

        // Round-trip: dtype preserved, payload byte-identical (no silent widen).
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("BF16 tensor present after pass-through");
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
            "arch stamp distinct from any generic nemotron / sibling parakeet / canary tag — \
             silent alias would misroute"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category groups Nemotron-Speech-Streaming-v2603 with the ASR family for the zoo manifest"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL),
            "upstream URL pins traceability back to the NGC catalog (no HF mirror at land time)"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE),
            "default license is apache-2.0 (Permissive)"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// F32 + F16 pass-through: two float tensors of distinct dtypes in
    /// the same input must both reach the pass-through arm without
    /// collapsing into a single dtype branch, and the BF16 counter must
    /// remain 0 (default). Guards against a naive `if bf16 { ... } else`
    /// refactor.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report =
            convert_nemotron_speech_streaming_v2603_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at the Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("encoder.blocks.0.mlp.fc1.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("decoder.output_layer.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// License override: a caller with an SPDX id distinct from the
    /// default (`apache-2.0`) must land on the artifact's license stamp;
    /// the license class is re-derived from the override string (mirror
    /// of the `convert_file_licensed` pattern in `lib.rs`). Uses
    /// `cc-by-4.0` (AttributionRequired) so the class ALSO changes — a
    /// permissive-only override would flip the SPDX but keep the class,
    /// missing the class-derivation regression window.
    #[test]
    fn license_override_lands_on_the_artifact_and_reshapes_the_class() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let _ = convert_nemotron_speech_streaming_v2603_file(&input, &output, Some("cc-by-4.0"))
            .expect("convert with license override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-4.0"),
            "override MUST land on the raw licence slot"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str()),
            "override to cc-by-4.0 MUST re-derive the class to AttributionRequired \
             (a stale Permissive stamp would tag the artifact as no-attribution-required \
             in the publish gate)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
