//! **FSMN-VAD** (FunASR): safetensors → GGUF conversion (TIER 1 F wave,
//! 2026-07-30).
//!
//! Input: the upstream `funasr/fsmn-vad` release (safetensors form).
//! FSMN = Feedforward Sequential Memory Network (Zhang et al. 2015 —
//! `arXiv:1512.08301`), a compact filter-window VAD that FunASR
//! re-published for streaming voice-activity detection at 200 ms hops.
//! Output: a GGUF carrying every F32 / F16 / BF16 tensor verbatim under
//! its upstream name plus the `vokra.provenance.*` / `vokra.model.*`
//! metadata chunks a future `vokra-models::fsmn_vad::*` loader will
//! read.
//!
//! # Sibling / alias (F2)
//!
//! `FunAudioLLM/fsmn-vad-GGUF` is a re-hosted GGUF sibling carrying the
//! same weight — no separate `ModelKind`. Callers who hand the CLI
//! either `funasr/fsmn-vad` (safetensors, this converter) or
//! `FunAudioLLM/fsmn-vad-GGUF` (already GGUF, no conversion needed) get
//! the same runtime binding — the alias lives in
//! [`crate::ModelKind::from_arg`] under `ModelKind::FsmnVad`.
//!
//! # Provenance
//!
//! - **HF path (default)**: `funasr/fsmn-vad`.
//! - **HF path (alias)**: `FunAudioLLM/fsmn-vad-GGUF` (same weight,
//!   re-hosted; caller-side `--input` distinguishes).
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`) — per the FunASR
//!   family license (`github.com/modelscope/FunASR/blob/main/LICENSE`
//!   ships MIT, but the ModelScope FSMN-VAD model card + HF mirror both
//!   pin `apache-2.0` for the released weight). Verified 2026-07-30.
//! - **Category**: `vad` (recorded under `vokra.model.category`) —
//!   Vokra's second first-party `"vad"` category weight (Silero VAD
//!   being the first, though Silero uses a bespoke `silero_vad` arch tag
//!   not shared with generic VADs).
//!
//! # BF16 pass-through (mirror of `wespeaker` / `neucodec` / `ecapa_tdnn`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30 (`GgmlType::BF16`).
//! Runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 =
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream safetensors name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec / WeSpeaker contract). Real-weight binding is a
//! follow-up wave gated on the upstream tensor-name manifest fetch +
//! §3.1 sign-off; this converter passes every float tensor through
//! unchanged so a future `FsmnVadWeights::from_gguf` can walk the same
//! names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream FunASR Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off) — this
//! converter provides the byte-parallel GGUF surface only. The internal
//! FSMN forward is a future
//! `vokra-models::fsmn_vad::FsmnVad::forward` that will land under the
//! loud-partial `VokraError::UnsupportedOp` precedent (see RMVPE /
//! Charsiu) until a real inference topology transcription lands.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FSMN-VAD GGUFs. Distinct from `silero_vad`
/// (Vokra's first `category = "vad"` arch) because FSMN uses a
/// filter-window feed-forward memory topology unlike Silero's
/// TF-Lite-derived STFT + Conv1d + BiLSTM chain — silently sharing an
/// arch tag would misroute the runtime dispatch.
pub const ARCH: &str = "fsmn_vad";

/// `vokra.model.name` value written for the canonical FSMN-VAD GGUF.
pub const NAME: &str = "fsmn-vad";

/// `vokra.model.category` value — `"vad"`. Consumed by the model-card
/// generator + zoo manifest tier gate so a VAD is not accidentally
/// advertised as an ASR / TTS release.
pub const CATEGORY: &str = "vad";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator. The `FunAudioLLM/fsmn-vad-GGUF`
/// alias is a caller-side `--input` choice, not a separate GGUF stamp.
pub const UPSTREAM_HF: &str = "funasr/fsmn-vad";

/// Default upstream weight license (SPDX). Overrides via the `license`
/// parameter of [`convert_fsmn_vad_file`].
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Raw metadata key for the model category — kept as a converter-side
/// constant (the cross-crate constant duplication rule the sibling
/// converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Raw metadata key for the upstream HF path.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an FSMN-VAD conversion.
///
/// Mirrors the sibling converters' counter shape
/// ([`super::wespeaker::WespeakerReport`],
/// [`super::neucodec::NeucodecReport`],
/// [`super::ecapa_tdnn::EcapaTdnnReport`]) — a leading `read` counter
/// pinning the total tensor budget the safetensors reader surfaced, so
/// `read == written + skipped_non_float` is an auditable invariant.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FsmnVadReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a silent
    /// widen / downcast cannot slip in undetected.
    pub bf16_passthrough: usize,
}

/// File-based FSMN-VAD converter (`vokra-cli convert --model fsmn-vad`).
///
/// Reads `input` (upstream `funasr/fsmn-vad` `model.safetensors` — or
/// any `FunAudioLLM/fsmn-vad-GGUF`-provenanced sibling safetensors),
/// writes a Vokra GGUF to `output`. `license` overrides the default
/// `apache-2.0` provenance stamp (Whisper / kokoro-family override
/// pattern — see `convert_file_licensed` in `lib.rs`); pass `None` to
/// keep the built-in `apache-2.0` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
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
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "funasr/fsmn-vad (FSMN feedforward sequential memory VAD, apache-2.0; \
             alias FunAudioLLM/fsmn-vad-GGUF re-hosts same weight)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

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

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-fsmn-vad-{tag}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2, "shape × 2 B BF16");
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

    /// BF16 tensor round-trips through the file-based converter with its
    /// dtype preserved and payload byte-identical — the standing
    /// pass-through pin across the sibling BF16-capable converters.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero bit patterns so a silent widen cannot round-trip
        // trivially.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        // FSMN-VAD-flavour upstream tensor name (feedforward memory
        // block filter). Realistic string so the round-trip exercises
        // the actual on-disk shape, not a synthetic one.
        let input_bytes = safetensors_one_bf16("encoder.fsmn.filter.weight", &[2, 3], &bf16);
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_fsmn_vad_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out = std::fs::read(&output).expect("read output");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        let file = GgufFile::parse(out).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.fsmn.filter.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        // Provenance + category stamps landed.
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
            Some(CATEGORY)
        );
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
    }

    /// The `license` override lands on the artifact — `--license
    /// mit` from the CLI must stamp `mit` + `Permissive` (both remain
    /// permissive, but the SPDX text changes). Mirror of the
    /// `convert_file_licensed` outer contract.
    #[test]
    fn license_override_lands_on_artifact() {
        let bf16 = [0u8; 12]; // shape [2,3] BF16 zeros — content irrelevant here.
        let input_bytes = safetensors_one_bf16("dummy.weight", &[2, 3], &bf16);
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_fsmn_vad_file(&input, &output, Some("mit")).expect("convert");
        assert_eq!(report.written, 1);

        let out = std::fs::read(&output).expect("read output");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        let file = GgufFile::parse(out).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override SPDX must land verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT still resolves to Permissive"
        );
    }
}
