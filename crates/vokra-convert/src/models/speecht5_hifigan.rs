#![allow(clippy::doc_lazy_continuation)]
//! **Microsoft SpeechT5 HiFi-GAN vocoder** (`microsoft/speecht5_hifigan`,
//! MIT): pytorch-pickle → safetensors → GGUF conversion (2026-07-31 wave).
//!
//! Input: the upstream `microsoft/speecht5_hifigan` release — the HiFi-GAN
//! vocoder companion to `microsoft/speecht5_tts`, trained on LibriTTS at
//! 16 kHz. The upstream repo ships a **torch pickle** `pytorch_model.bin`
//! + `config.json` **only** (no `model.safetensors` mirror as of
//! 2026-07-31 — verified via `https://huggingface.co/api/models/
//! microsoft/speecht5_hifigan`); callers pre-flatten to safetensors
//! offline via `tools/parity/speecht5_hifigan_prepare_checkpoint.py`
//! (a thin wrapper over `tools/parity/bin_to_safetensors.py` — the same
//! pattern DeBERTa v3 large / VoxCPM-0.5B / Fun-CosyVoice3 use, and the
//! reason is the same: Vokra's Rust converter is safetensors-only by
//! design so the runtime never grows a pickle parser, keeping the
//! NFR-DS-02 zero-dep posture).
//!
//! Output: a GGUF carrying every float tensor verbatim under its upstream
//! HF-transformers `SpeechT5HifiGan` name (`conv_pre.*`,
//! `upsampler.{i}.*`, `resblocks.{i}.convs1.{j}.*`,
//! `resblocks.{i}.convs2.{j}.*`, `conv_post.*`, plus learned `mean` /
//! `scale` for `normalize_before=true`), plus the `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future native SpeechT5-HifiGan
//! vocoder loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `microsoft/speecht5_hifigan`.
//! - **License (SPDX)**: `mit` — end-to-end (Microsoft SpeechT5 code +
//!   trained weight; primary source = HF cardData `license: mit`,
//!   fetched 2026-07-31 via `https://huggingface.co/api/models/
//!   microsoft/speecht5_hifigan` — CLAUDE.md 「ハルシネーション厳禁」).
//! - **Category**: `vocoder` — mel spectrogram → PCM waveform generator.
//!   Category tag is written under the raw `vokra.model.category` key
//!   so the model-card tooling can classify without reaching into
//!   per-converter constants.
//!
//! # Distinct arch from the SpeechBrain `hifigan_vocoder` sibling
//!
//! `ARCH = "speecht5_hifigan"` is intentionally distinct from the
//! sibling `crates/vokra-convert/src/models/hifigan_vocoder.rs`
//! (`speechbrain/tts-hifigan-libritts-22050Hz`). The two share the
//! HiFi-GAN family lineage (Kong et al. 2020, arXiv:2010.05646 — MRF
//! + leaky_relu + transposed-conv upsample) but differ in every
//! runtime-relevant respect:
//!
//! - **Sampling rate**: 16 kHz (SpeechT5) vs 22 050 Hz (SpeechBrain).
//! - **Mel band count**: 80 (both); `model_in_dim = 80` per
//!   SpeechT5's `config.json` — same as SpeechBrain here, but the
//!   frontend that feeds it lives inside a different TTS system
//!   (SpeechT5's transformer decoder vs SpeechBrain's Tacotron 2).
//! - **Upsample topology**: `upsample_rates = [4, 4, 4, 4]` with
//!   `upsample_kernel_sizes = [8, 8, 8, 8]` (SpeechT5) vs the
//!   SpeechBrain LibriTTS variant's own upsample recipe.
//! - **Normalization**: SpeechT5 sets `normalize_before = true` and
//!   carries learned scalar-per-mel-bin `mean` / `scale` tensors
//!   applied before the network — SpeechBrain LibriTTS does not.
//! - **Tensor naming convention**: HF-transformers
//!   `SpeechT5HifiGan` class emits `upsampler.{i}` / `resblocks.{i}`
//!   / `conv_pre` / `conv_post` names, versus SpeechBrain's
//!   `generator.` prefix. Sharing an arch tag would silently
//!   mis-route runtime dispatch.
//!
//! Also distinct from `bigvgan` (snake / snakebeta activation + alias-
//! free wrapper) and `piper-plus` (full TTS, not a standalone vocoder).
//!
//! # BF16 pass-through (mirror of wespeaker / ecapa_tdnn / hifigan_vocoder)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single
//! choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 is the top 16 bits of an f32 — `bits << 16` is exact). The
//! observability counter [`Speecht5HifiganReport::bf16_passthrough`]
//! records how many BF16 tensors landed on this arm so a silent widen
//! / downcast cannot slip in undetected. Upstream `config.json` pins
//! `torch_dtype: float32` so BF16 is not expected on the primary
//! release, but the counter is kept for the SKU-parity contract and
//! for third-party BF16 re-quantizations.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice /
//! VoxCPM / WeSpeaker / ECAPA-TDNN / hifigan_vocoder contract).
//! Real-weight parity vs the upstream `transformers.SpeechT5HifiGan`
//! Python forward is deferred to owner (`docs/license-audit.md` §3.1
//! sign-off queue).
//!
//! # No ONNX (permanent)
//!
//! SpeechT5 ships PyTorch pickle checkpoints; this converter **never**
//! touches ONNX (FR-LD-05); the pipeline is re-implemented natively in
//! `crates/vokra-models/src/speecht5_hifigan/` (or folded into the
//! shared HiFi-GAN family loader) when the vocoder lands
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Loud-partial precedent
//!
//! Real-weight forward binding is deferred: the runtime consumer will
//! walk the emitted tensor names and either succeed or fail loudly per
//! FR-EX-08. Today's converter surface is byte-exact provenance +
//! tensor-name preservation only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for the SpeechT5-HifiGan vocoder GGUFs.
///
/// Intentionally distinct from `hifigan_vocoder` (SpeechBrain
/// LibriTTS 22050Hz variant) — see the module-level docstring.
pub const ARCH: &str = "speecht5_hifigan";

/// `vokra.model.name` value written for the canonical
/// `microsoft/speecht5_hifigan` GGUF.
pub const NAME: &str = "speecht5-hifigan";

/// `vokra.model.category` value written for every SpeechT5-HifiGan
/// vocoder GGUF. Same tag as the sibling hifigan_vocoder / bigvgan /
/// focalcodec (all `vocoder`), used by the model-card generator
/// classifier.
pub const CATEGORY: &str = "vocoder";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator.
pub const UPSTREAM_HF: &str = "microsoft/speecht5_hifigan";

/// Default upstream weight licence (SPDX). Microsoft SpeechT5 family
/// (both `speecht5_tts` and `speecht5_hifigan`) ships MIT end-to-end;
/// verified 2026-07-31 via HF cardData API.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of the sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a SpeechT5-HifiGan vocoder conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::hifigan_vocoder::HifiganVocoderReport`,
/// `super::wespeaker::WespeakerReport`,
/// `super::ecapa_tdnn::EcapaTdnnReport`) adapted to the
/// file-oriented `convert_speecht5_hifigan_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Speecht5HifiganReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
}

/// Converts a `microsoft/speecht5_hifigan` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`, returning a
/// [`Speecht5HifiganReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// HF-transformers `SpeechT5HifiGan` name (see the module-level
/// docstring for the naming convention); the `vokra.model.*`
/// (arch / name / category) and `vokra.provenance.*` (weight_license
/// / license / model_id / source / upstream_hf) chunks are stamped
/// for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`) — the upstream HF
/// release ships MIT end-to-end.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_speecht5_hifigan_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Speecht5HifiganReport, ConvertError> {
    // SpeechT5 HifiGan vocoder is ~51 MiB per upstream pytorch_model.bin
    // (verified 2026-07-31 via HF file listing) — 3 orders of magnitude
    // smaller than the streaming-mandated Moshi 14 GiB tier, so the
    // simple `std::fs::read` posture the sibling non-streaming BF16
    // pass-through converters use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Default provenance stamp — Permissive MIT end-to-end
    // (upstream `microsoft/speecht5_hifigan` model card `license: mit`
    // + Microsoft SpeechT5 code MIT). The optional `license` argument
    // overrides below.
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
            "microsoft/speecht5_hifigan \
             (SpeechT5 HiFi-GAN vocoder, 16 kHz, 80-band mel, MIT)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = Speecht5HifiganReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (mirror of wespeaker / ecapa_tdnn / hifigan_vocoder / qwen3_tts /
    // vibevoice / voxcpm2 / moshi); the runtime widens BF16 → f32
    // exactly at load via the single choke point
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
    /// caller-supplied raw payload.
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

    /// Builds a mixed F32 + F16 safetensors buffer using realistic
    /// upstream `SpeechT5HifiGan` tensor names:
    ///   `upsampler.0.weight`  — F32, `[1,2]`  →  8 bytes @ [0..8)
    ///   `mean`                — F16, `[2,3]`  → 12 bytes @ [8..20)
    fn safetensors_f32_and_f16() -> Vec<u8> {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). 1.0 = 0x3C00, -2.0 = 0xC000, -0.5 = 0xB800,
        // 3.0 = 0x4200, 0.15625 = 0x3100, 42.0 = 0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);
        let header = format!(
            r#"{{"upsampler.0.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"mean":{{"dtype":"F16","shape":[2,3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// PID + nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-speecht5-hifigan-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic upstream tensor name from
        // microsoft/speecht5_hifigan's HF-transformers `SpeechT5HifiGan`
        // class (`resblocks.0.convs1.0.weight` is one of the MRF branches
        // at stage 0 — the HF names have no `generator.` prefix, unlike
        // the SpeechBrain sibling).
        let input_bytes = safetensors_one_bf16("resblocks.0.convs1.0.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_speecht5_hifigan_file(&input_path, &output_path, None)
            .expect("convert_speecht5_hifigan_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror wespeaker / ecapa_tdnn)"
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
            .tensor_info("resblocks.0.convs1.0.weight")
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

    #[test]
    fn f32_and_f16_tensors_pass_through_and_stamps_land() {
        let input_bytes = safetensors_f32_and_f16();
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_speecht5_hifigan_file(&input_path, &output_path, None)
            .expect("convert_speecht5_hifigan_file must accept a mixed F32/F16 checkpoint");

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
        // AND the arch / provenance / category stamps land.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("upsampler.0.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");

        // `mean` is one of the two learned normalize_before scalars
        // (`normalize_before: true` in the SpeechT5 HifiGan config).
        let f16_info = file.tensor_info("mean").expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");

        // Provenance / category chunks landed (task-spec pins).
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
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `vocoder`",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_flows_through() {
        // A user who re-trained on a permissive corpus supplies a different
        // SPDX id at conversion time — the override must land on
        // KEY_PROVENANCE_LICENSE + KEY_PROVENANCE_WEIGHT_LICENSE and the
        // LicenseClass must be re-derived by from_license_str.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("resblocks.0.convs1.0.weight", &[2, 3], &bf16);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        let _report = convert_speecht5_hifigan_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 is Permissive class (same as MIT)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
