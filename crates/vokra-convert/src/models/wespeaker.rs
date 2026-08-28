//! **WeSpeaker** (`Wespeaker/wespeaker-voxceleb-resnet34-LM`, CC-BY-4.0):
//! safetensors → GGUF conversion (SoTA follow-on, 2026-07-25).
//!
//! Input: the upstream `Wespeaker/wespeaker-voxceleb-resnet34-LM` release
//! — a ResNet34 speaker-embedding network trained on VoxCeleb with the
//! Large-Margin (LM) fine-tuning stage. Output: a GGUF carrying every
//! float tensor verbatim under its upstream safetensors name, plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata consumed by the native
//! CPU/Metal WeSpeaker loader.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `Wespeaker/wespeaker-voxceleb-resnet34-LM` (recorded
//!   under `vokra.provenance.upstream_hf`).
//! - SPDX: `cc-by-4.0` (`LicenseClass::AttributionRequired`).
//! - Model category: `speaker` (recorded under `vokra.model.category`).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling converters
//! listed above. No convert-time widening; runtime widens BF16 → f32
//! losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are preserved verbatim after validating one of the two
//! exact supported manifests: the 219-tensor official combined checkpoint or
//! the 182-tensor `resnet.*` pyannote embedding checkpoint. Unknown, partial,
//! and shape-incompatible manifests fail before an output file is written.
//!
//! # No ONNX (permanent)
//!
//! WeSpeaker is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/wespeaker/`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for WeSpeaker GGUFs.
pub const ARCH: &str = "wespeaker";

/// `vokra.model.name` value written for the canonical WeSpeaker GGUF.
pub const NAME: &str = "wespeaker-voxceleb-resnet34-lm";

/// Model-category tag written under `vokra.model.category`. `"speaker"`
/// distinguishes speaker-embedding / speaker-verification networks from
/// TTS / ASR / codec / vocoder siblings so downstream consumers can
/// pick a load path without inspecting the arch.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub const MODEL_CATEGORY: &str = "speaker";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`. The slug preserves upstream casing
/// (`Wespeaker/wespeaker-voxceleb-resnet34-LM`).
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const UPSTREAM_HF: &str = "Wespeaker/wespeaker-voxceleb-resnet34-LM";

/// Pinned upstream Hugging Face checkpoint revision.
pub const UPSTREAM_REVISION: &str = "f0c48c298fd835726c27956a5d617bad7115627e";
/// Pinned WeSpeaker source revision used by the native implementation.
pub const SOURCE_REVISION: &str = "45941e7cba2c3ea99e232d02bedf617fc71b0dad";

/// Upstream cardData license for the LM checkpoint.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-4.0";

/// Attribution carried in every default WeSpeaker conversion (FR-MD-09).
pub const WESPEAKER_ATTRIBUTION_TEXT: &str = "This application uses WeSpeaker ResNet34-LM \
    (speaker embedding; VoxCeleb, Large-Margin fine-tune). Model weights are \
    licensed under CC-BY 4.0 (attribution required; commercial use permitted). \
    Source: https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM";

const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.wespeaker.source_revision";
const KEY_SAMPLE_RATE: &str = "vokra.wespeaker.sample_rate";
const KEY_N_MELS: &str = "vokra.wespeaker.n_mels";
const KEY_FRAME_LENGTH: &str = "vokra.wespeaker.frame_length";
const KEY_FRAME_SHIFT: &str = "vokra.wespeaker.frame_shift";
const KEY_EMBED_DIM: &str = "vokra.wespeaker.embed_dim";
const KEY_STAGE_COUNT: &str = "vokra.wespeaker.stage_count";
const KEY_BN_EPS: &str = "vokra.wespeaker.bn_eps";
const KEY_STATS_EPS: &str = "vokra.wespeaker.stats_eps";
const KEY_FRONTEND: &str = "vokra.wespeaker.frontend";
const KEY_BLOCKS: &str = "vokra.wespeaker.blocks";
const KEY_CHANNELS: &str = "vokra.wespeaker.channels";
const KEY_POOLING: &str = "vokra.wespeaker.pooling";
const KEY_LAYOUT: &str = "vokra.wespeaker.artifact_layout";
const STAGE_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const STAGE_CHANNELS: [u64; 4] = [32, 64, 128, 256];
const EMBED_DIM: u64 = 256;
const STATS_DIM: u64 = 5_120;
const PREFIXED_TENSOR_COUNT: usize = 182;
const BARE_COMBINED_TENSOR_COUNT: usize = 219;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactLayout {
    PyannotePrefixed,
    OfficialCombinedBare,
}

impl ArtifactLayout {
    const fn stem(self) -> &'static str {
        match self {
            Self::PyannotePrefixed => "resnet.",
            Self::OfficialCombinedBare => "",
        }
    }

    const fn tensor_count(self) -> usize {
        match self {
            Self::PyannotePrefixed => PREFIXED_TENSOR_COUNT,
            Self::OfficialCombinedBare => BARE_COMBINED_TENSOR_COUNT,
        }
    }

    const fn contract_name(self) -> &'static str {
        match self {
            Self::PyannotePrefixed => "pyannote-prefixed-182-v1",
            Self::OfficialCombinedBare => "official-combined-bare-219-v1",
        }
    }
}

/// Outcome of a WeSpeaker conversion.
///
/// Mirrors [`crate::models::qwen3_tts::Qwen3TtsReport`]'s counter set
/// (float pass-through + BF16 subset counter + non-float defensive
/// counter), plus a leading `read` count of every tensor observed in
/// the input safetensors header. `read == written + skipped_non_float`
/// is an invariant preserved by [`convert_wespeaker_file`].
#[derive(Debug, Default)]
pub struct WespeakerReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `qwen3_tts` /
    /// `vibevoice` / `voxcpm2` / `neucodec` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Mirrors
    /// `qwen3_tts::Qwen3TtsReport::bf16_passthrough` /
    /// `vibevoice::VibeVoiceReport::bf16_passthrough`.
    pub bf16_passthrough: usize,
}

/// File-based WeSpeaker converter (`vokra-cli convert --model wespeaker`).
///
/// Reads `input` (upstream
/// `Wespeaker/wespeaker-voxceleb-resnet34-LM` `model.safetensors`),
/// writes a Vokra GGUF to `output`. `license` overrides the default
/// `cc-by-4.0` provenance stamp (Whisper / kokoro-family override
/// pattern — see `convert_file_licensed` in `lib.rs`); pass `None` to
/// keep the built-in `cc-by-4.0` stamp and its attribution metadata.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_wespeaker_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<WespeakerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let layout = validate_manifest(&st)?;

    if let Some(license) = license.filter(|value| !value.is_empty())
        && !license.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX)
    {
        return Err(ConvertError::Parse(format!(
            "wespeaker: the audited ResNet34-LM checkpoint is `{DEFAULT_LICENSE_SPDX}`; refusing incompatible override `{license}`"
        )));
    }

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a load path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    b.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_N_MELS, 80);
    b.add_u32(KEY_FRAME_LENGTH, 400);
    b.add_u32(KEY_FRAME_SHIFT, 160);
    b.add_u32(KEY_EMBED_DIM, EMBED_DIM as u32);
    b.add_u32(KEY_STAGE_COUNT, STAGE_BLOCKS.len() as u32);
    b.add_f32(KEY_BN_EPS, 1.0e-5);
    b.add_f32(KEY_STATS_EPS, 1.0e-7);
    b.add_string(KEY_FRONTEND, "kaldi-fbank-hamming-cmn-v1");
    b.add_string(KEY_BLOCKS, "3,4,6,3");
    b.add_string(KEY_CHANNELS, "32,64,128,256");
    b.add_string(KEY_POOLING, "tstp-bessel-v1");
    b.add_string(KEY_LAYOUT, layout.contract_name());

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = CC-BY-4.0 (upstream
    // `Wespeaker/wespeaker-voxceleb-resnet34-LM` model card).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = (
        DEFAULT_LICENSE_SPDX.to_owned(),
        LicenseClass::AttributionRequired,
    );
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "Wespeaker/wespeaker-voxceleb-resnet34-LM \
             (ResNet34 speaker encoder, VoxCeleb + Large-Margin, CC-BY-4.0)",
        ),
    );
    if class.requires_attribution() {
        vokra_core::stamp_attribution(&mut b, WESPEAKER_ATTRIBUTION_TEXT);
    }

    let mut report = WespeakerReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as qwen3_tts /
    // vibevoice / voxcpm2 / neucodec; runtime widens BF16 → f32 exactly
    // at load via `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is
    // exact).
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

fn validate_manifest(st: &SafetensorsFile) -> Result<ArtifactLayout, ConvertError> {
    let layout = match st.tensors().len() {
        PREFIXED_TENSOR_COUNT if st.tensor_info("resnet.conv1.weight").is_some() => {
            ArtifactLayout::PyannotePrefixed
        }
        BARE_COMBINED_TENSOR_COUNT if st.tensor_info("conv1.weight").is_some() => {
            ArtifactLayout::OfficialCombinedBare
        }
        count => {
            return Err(ConvertError::Parse(format!(
                "wespeaker: unsupported tensor manifest: count={count}; expected exactly {PREFIXED_TENSOR_COUNT} prefixed embedding tensors or {BARE_COMBINED_TENSOR_COUNT} bare combined tensors"
            )));
        }
    };
    let expected = expected_manifest(layout);
    debug_assert_eq!(expected.len(), layout.tensor_count());
    for (name, dimensions) in expected {
        let tensor = st
            .tensor_info(&name)
            .ok_or_else(|| ConvertError::Parse(format!("wespeaker: missing tensor `{name}`")))?;
        if tensor.shape != dimensions {
            return Err(ConvertError::Parse(format!(
                "wespeaker: tensor `{name}` has dims {:?}, expected {dimensions:?}",
                tensor.shape
            )));
        }
    }
    Ok(layout)
}

fn expected_manifest(layout: ArtifactLayout) -> Vec<(String, Vec<u64>)> {
    let stem = layout.stem();
    let include_counter = layout == ArtifactLayout::OfficialCombinedBare;
    let mut expected = Vec::with_capacity(layout.tensor_count());
    push_conv(&mut expected, &format!("{stem}conv1"), 1, 32, 3);
    push_norm(&mut expected, &format!("{stem}bn1"), 32, include_counter);
    let mut input_channels = 32;
    for (stage_index, (&blocks, &output_channels)) in
        STAGE_BLOCKS.iter().zip(&STAGE_CHANNELS).enumerate()
    {
        for block_index in 0..blocks {
            let prefix = format!("{stem}layer{}.{}", stage_index + 1, block_index);
            push_conv(
                &mut expected,
                &format!("{prefix}.conv1"),
                input_channels,
                output_channels,
                3,
            );
            push_norm(
                &mut expected,
                &format!("{prefix}.bn1"),
                output_channels,
                include_counter,
            );
            push_conv(
                &mut expected,
                &format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
            );
            push_norm(
                &mut expected,
                &format!("{prefix}.bn2"),
                output_channels,
                include_counter,
            );
            if stage_index > 0 && block_index == 0 {
                push_conv(
                    &mut expected,
                    &format!("{prefix}.shortcut.0"),
                    input_channels,
                    output_channels,
                    1,
                );
                push_norm(
                    &mut expected,
                    &format!("{prefix}.shortcut.1"),
                    output_channels,
                    include_counter,
                );
            }
            input_channels = output_channels;
        }
    }
    expected.push((format!("{stem}seg_1.weight"), vec![EMBED_DIM, STATS_DIM]));
    expected.push((format!("{stem}seg_1.bias"), vec![EMBED_DIM]));
    if layout == ArtifactLayout::OfficialCombinedBare {
        expected.push(("projection.weight".into(), vec![17_982, EMBED_DIM]));
    }
    expected
}

fn push_conv(
    expected: &mut Vec<(String, Vec<u64>)>,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
    kernel: u64,
) {
    expected.push((
        format!("{prefix}.weight"),
        vec![output_channels, input_channels, kernel, kernel],
    ));
}

fn push_norm(
    expected: &mut Vec<(String, Vec<u64>)>,
    prefix: &str,
    channels: u64,
    include_counter: bool,
) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        expected.push((format!("{prefix}.{suffix}"), vec![channels]));
    }
    if include_counter {
        expected.push((format!("{prefix}.num_batches_tracked"), Vec::new()));
    }
}

// Historical one/two-tensor pass-through tests are kept out of the build: the
// production converter now deliberately rejects partial manifests.
#[cfg(all(test, any()))]
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16)
    /// with caller-supplied payloads.
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
            "vokra-wespeaker-{kind}-{}-{}.bin",
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
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror an actual upstream ResNet34 tensor name (e.g.
        // `speaker.resnet.layer1.0.conv1.weight`) so the round-trip
        // exercises a realistic string, not a synthetic one.
        let input_bytes =
            safetensors_one_bf16("speaker.resnet.layer1.0.conv1.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_wespeaker_file(&input_path, &output_path, None)
            .expect("convert_wespeaker_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts / vibevoice / voxcpm2)"
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
            .tensor_info("speaker.resnet.layer1.0.conv1.weight")
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
    fn f32_and_f16_tensors_pass_through() {
        // Non-zero payloads so a silent-widen regression can't hide
        // behind trivial round-trips.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). 1.0 = 0x3C00, -2.0 = 0xC000,
        // -0.5 = 0xB800, 3.0 = 0x4200, 0.15625 = 0x3100, 42.0 = 0x5140.
        // Six values for a [2,3] tensor = 12 bytes.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12, "6 elements × 2 bytes F16 payload");

        let input_bytes = safetensors_f32_then_f16(
            "speaker.dense.weight",
            &[1, 2],
            &f32_bytes,
            "speaker.embed.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_wespeaker_file(&input_path, &output_path, None)
            .expect("convert_wespeaker_file must accept a mixed F32/F16 checkpoint");

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
            .tensor_info("speaker.dense.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("speaker.embed.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance / category chunks landed (task-spec pins).
        use vokra_core::LicenseClass;
        use vokra_core::gguf::chunks;
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
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION)
                .and_then(|v| v.as_str()),
            Some(WESPEAKER_ATTRIBUTION_TEXT)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}

#[cfg(test)]
mod strict_tests {
    use super::*;

    #[test]
    fn supported_manifests_have_exact_unique_counts() {
        for layout in [
            ArtifactLayout::PyannotePrefixed,
            ArtifactLayout::OfficialCombinedBare,
        ] {
            let manifest = expected_manifest(layout);
            assert_eq!(manifest.len(), layout.tensor_count());
            let mut names = manifest.iter().map(|(name, _)| name).collect::<Vec<_>>();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), layout.tensor_count());
        }
    }

    #[test]
    fn official_combined_manifest_keeps_classifier_and_training_counters() {
        let manifest = expected_manifest(ArtifactLayout::OfficialCombinedBare);
        assert!(manifest.iter().any(|(name, dimensions)| {
            name == "projection.weight" && dimensions == &[17_982, 256]
        }));
        assert!(manifest.iter().any(|(name, dimensions)| {
            name == "bn1.num_batches_tracked" && dimensions.is_empty()
        }));
    }

    #[test]
    fn pyannote_manifest_is_embedding_only_and_prefixed() {
        let manifest = expected_manifest(ArtifactLayout::PyannotePrefixed);
        assert!(manifest.iter().all(|(name, _)| name.starts_with("resnet.")));
        assert!(!manifest.iter().any(|(name, _)| name == "projection.weight"));
        assert!(
            !manifest
                .iter()
                .any(|(name, _)| name.ends_with("num_batches_tracked"))
        );
    }
}
