//! **StyleTTS 2**: safetensors checkpoint → GGUF conversion
//! (Config-only scaffold — the yl4579 pre-trained weights are
//! **voice-consent gated** and NOT redistributable by Vokra).
//!
//! Input: a StyleTTS 2 (Li et al. 2023, arXiv:2306.07691) safetensors
//! checkpoint from `github.com/yl4579/StyleTTS2` — LJSpeech
//! (`Models/LJSpeech/`), LibriTTS (`Models/LibriTTS/`), or a downstream
//! re-training on a permissive corpus. Output: a GGUF carrying every
//! float tensor plus the `vokra.styletts2.*` + `vokra.provenance.*` +
//! `vokra.model.*` metadata chunks the native StyleTTS 2 scaffold
//! (`crates/vokra-models/src/styletts2/`) reads.
//!
//! # ⚠️  Weight distribution — **fail-closed by default**
//!
//! The upstream README (`github.com/yl4579/StyleTTS2/blob/main/README.md`
//! §Pre-trained Models) conditions weight use on **voice consent +
//! disclosure**:
//!
//! > "Before using these pre-trained models, you agree to inform the
//! > listeners that the speech samples are synthesized by the pre-trained
//! > models, unless you have the permission to use the voice you
//! > synthesize."
//! >
//! > "only use voices whose speakers grant the permission to have their
//! > voice cloned..."
//!
//! This is a **usage agreement**, not a standard SPDX permissive license.
//! The Vokra registry (`vokra-core::LicenseClass::from_id`) resolves
//! `styletts2` / `styletts-2` to [`LicenseClass::Unknown`] (fail-closed
//! under M2-13), matching the sign-off in
//! `docs/license-audit.md` §3.1 StyleTTS 2 row = `☑ Rejected 2026-07-23
//! yousan` (weight redistribution declined).
//!
//! The converter therefore stamps the provenance as
//! [`LicenseClass::Unknown`] with SPDX id `unknown`. A user who trained
//! their own StyleTTS 2 on a permissive corpus overrides at the outer
//! `vokra-convert --license <spdx>` boundary; the runtime `from_gguf` is
//! **still** unwired (see `styletts2::StyleTts2Tts::from_gguf` — the
//! architecture surface exists so a future wave can bind real weights
//! under the caller-provided license, but the default posture is
//! unimplemented, not silently loaded).
//!
//! Architecture rides MIT code (`github.com/yl4579/StyleTTS2/LICENSE`)
//! and is *always* independently implementable (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.styletts2.*` chunk group is transcribed **verbatim** from
//!   the primary sources
//!   `github.com/yl4579/StyleTTS2/blob/main/Models/LJSpeech/config.yml`
//!   and `Models/LibriTTS/config.yml` (fetched 2026-07-30 — CLAUDE.md
//!   「ハルシネーション厳禁」). Default recipe = LJSpeech single-speaker
//!   (24 kHz, no style diffusion sampler). A LibriTTS or downstream
//!   multi-speaker variant would override at bind time.
//! - **Sample rate** — 24 000 Hz on both released variants
//!   (`config.yml`: `sample_rate: 24000`).
//! - **Style diffusion** — this scaffold defaults to the LJSpeech
//!   single-speaker recipe (`uses_style_diffusion=false`); a caller who
//!   trained on LibriTTS or a downstream multi-speaker corpus would use
//!   a `--config` side-car (future wave — not wired today because no
//!   permissive-license StyleTTS 2 checkpoint has arrived yet to gate
//!   the design).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / VoxCPM / VibeVoice / VITS-JA
//! contract). Real-weight binding is a follow-up wave gated on
//! `--license <spdx>` at conversion time (see the license section
//! above); this converter passes every F32 / F16 / BF16 tensor through
//! unchanged so a future `StyleTts2Weights::from_gguf` can walk the
//! same names.
//!
//! # BF16 posture
//!
//! Mirror of `vits_ja` / `qwen3-tts` / `vibevoice` / `voxcpm2` / `moshi`
//! / `voxtral`: BF16 bytes emit as GGUF type 30 verbatim and the runtime
//! widens on load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
//! bits of an f32 — `bits << 16` is exact, no precision loss).
//!
//! # No ONNX (permanent)
//!
//! StyleTTS 2 upstream ships PyTorch `.pth` checkpoints; this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in `crates/vokra-models/src/styletts2/` (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for StyleTTS 2 GGUFs — kept in sync with any
/// future runtime `EXPECTED_ARCH` constant.
pub(crate) const ARCH: &str = "styletts2";

/// `vokra.model.name` value written for the canonical yl4579 StyleTTS 2
/// (LJSpeech single-speaker) GGUF. A LibriTTS or downstream re-training
/// would override at bind time (future `--config` side-car).
pub(crate) const NAME: &str = "styletts2-ljspeech-24khz";

// --- vokra.styletts2.* metadata keys ------------------------------------

/// Model family marker.
const KEY_MODEL_FAMILY: &str = "vokra.styletts2.model_family";
/// PCM sample rate (Hz).
const KEY_SAMPLE_RATE_HZ: &str = "vokra.styletts2.sample_rate_hz";
/// Style vector dimension.
const KEY_STYLE_DIM: &str = "vokra.styletts2.style_dim";
/// Residual / hidden width.
const KEY_HIDDEN_DIM: &str = "vokra.styletts2.hidden_dim";
/// Mel bin count fed to iSTFTNet decoder.
const KEY_N_MELS: &str = "vokra.styletts2.n_mels";
/// Text encoder — number of residual 1D-conv blocks.
const KEY_TEXT_ENCODER_N_LAYER: &str = "vokra.styletts2.text_encoder.n_layer";
/// Duration / style predictor hidden width.
const KEY_PREDICTOR_HIDDEN_DIM: &str = "vokra.styletts2.predictor.hidden_dim";
/// Style diffusion sampler — number of Heun steps (0 if disabled).
const KEY_DIFFUSION_STEPS: &str = "vokra.styletts2.diffusion.steps";
/// Whether the checkpoint carries a trained style diffusion sampler.
const KEY_USES_STYLE_DIFFUSION: &str = "vokra.styletts2.diffusion.uses_style_diffusion";
/// iSTFTNet decoder — channels at entry.
const KEY_DECODER_DIM_IN: &str = "vokra.styletts2.decoder.dim_in";
/// iSTFTNet decoder — post-net iSTFT n_fft.
const KEY_DECODER_GEN_ISTFT_N_FFT: &str = "vokra.styletts2.decoder.gen_istft_n_fft";
/// iSTFTNet decoder — post-net iSTFT hop size.
const KEY_DECODER_GEN_ISTFT_HOP_SIZE: &str = "vokra.styletts2.decoder.gen_istft_hop_size";

// --- Transcribed constants ------------------------------------------------
// Primary sources: `Models/LJSpeech/config.yml` + `Models/LibriTTS/config.yml`
// (fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).

/// Model family marker.
const MODEL_FAMILY: &str = "styletts2";
/// PCM sample rate the LJSpeech / LibriTTS release emits.
const SAMPLE_RATE_HZ: u32 = 24_000;
/// Style vector dimension shared across LJSpeech / LibriTTS.
const STYLE_DIM: u32 = 128;
/// Residual / hidden width shared across LJSpeech / LibriTTS.
const HIDDEN_DIM: u32 = 512;
/// Mel bin count shared across LJSpeech / LibriTTS.
const N_MELS: u32 = 80;
/// Text encoder — number of residual 1D-conv blocks (LJSpeech / LibriTTS).
const TEXT_ENCODER_N_LAYER: u32 = 3;
/// Duration / style predictor hidden width (LJSpeech / LibriTTS).
const PREDICTOR_HIDDEN_DIM: u32 = 512;
/// LJSpeech single-speaker default = 0 (diffusion sampler disabled).
/// The LibriTTS variant overrides to 5 via a future `--config` side-car.
const DIFFUSION_STEPS: u32 = 0;
/// LJSpeech single-speaker default = false. LibriTTS = true.
const USES_STYLE_DIFFUSION: bool = false;
/// iSTFTNet decoder — channels at entry.
const DECODER_DIM_IN: u32 = 512;
/// iSTFTNet decoder — post-net iSTFT n_fft.
const DECODER_GEN_ISTFT_N_FFT: u32 = 20;
/// iSTFTNet decoder — post-net iSTFT hop size.
const DECODER_GEN_ISTFT_HOP_SIZE: u32 = 5;

/// Outcome of a StyleTTS 2 conversion.
#[derive(Debug, Default)]
pub(crate) struct StyleTts2Report {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a StyleTTS 2 safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream name;
/// the `vokra.styletts2.*` chunk group is written from the transcribed
/// constants above (LJSpeech 24 kHz single-speaker recipe defaults); the
/// provenance stamps mark the weight as [`LicenseClass::Unknown`] by
/// default — the yl4579 release carries a voice-consent / disclosure
/// usage agreement instead of a standard SPDX permissive license, so
/// the M2-13 runtime gate refuses to load in commercial mode. A user
/// who trained their own StyleTTS 2 on a permissive corpus overrides
/// via `vokra-convert --license <spdx>` at the outer boundary
/// (`crates/vokra-convert/src/lib.rs`).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, StyleTts2Report), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    // Self-describing licence stamp: the yl4579 pre-trained models ship
    // under a voice-consent / disclosure usage agreement (README §Pre-
    // trained Models) — NOT a standard SPDX permissive license. Stamp
    // Unknown so the M2-13 runtime gate refuses to load the resulting
    // GGUF outside `--i-understand-risks --research-only` mode. A user
    // who trained on a permissive corpus overrides at the
    // `convert_file --license <spdx>` boundary.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Unknown,
        "unknown",
        Some(NAME),
        Some(
            "StyleTTS 2 (yl4579) — architecture MIT (code); pretrained weight bound by a \
             voice-consent / disclosure usage agreement (README §Pre-trained Models) that is \
             NOT a standard SPDX permissive license. Provenance defaults to Unknown \
             (fail-closed under M2-13). Override with --license <spdx> at conversion time if \
             you trained this StyleTTS 2 on a permissive corpus. docs/license-audit.md §3.1 \
             StyleTTS 2 sign-off = ☑ Rejected 2026-07-23 yousan (weight redistribution \
             declined).",
        ),
    );

    let mut report = StyleTts2Report::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through mirror of vits_ja / qwen3-tts / vibevoice
            // / voxcpm2 / moshi / voxtral.
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
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and the runtime will \
             refuse to bind any weights (FR-EX-08). StyleTTS 2 upstream ships PyTorch `.pth`; \
             callers must first flatten with `tools/parity/pytorch_to_safetensors.py` (or the \
             CSM / DAC pattern) before invoking this converter."
                .into(),
        );
    }
    // Always emit the licence note so an operator reading the conversion
    // output cannot miss the fail-closed default's meaning.
    report.notes.push(
        "provenance defaults to `Unknown` — yl4579 StyleTTS 2 pre-trained models ride a \
         voice-consent / disclosure usage agreement (NOT a standard SPDX permissive license). \
         Override with `vokra-convert --license <spdx>` if you trained this StyleTTS 2 on a \
         permissive corpus. See docs/license-audit.md §3.1 StyleTTS 2 sign-off = ☑ Rejected."
            .into(),
    );
    Ok((b, report))
}

/// Writes the `vokra.styletts2.*` chunk group from the transcribed
/// constants above (primary sources: `Models/LJSpeech/config.yml` +
/// `Models/LibriTTS/config.yml`).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);
    b.add_u32(KEY_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ);
    b.add_u32(KEY_STYLE_DIM, STYLE_DIM);
    b.add_u32(KEY_HIDDEN_DIM, HIDDEN_DIM);
    b.add_u32(KEY_N_MELS, N_MELS);
    b.add_u32(KEY_TEXT_ENCODER_N_LAYER, TEXT_ENCODER_N_LAYER);
    b.add_u32(KEY_PREDICTOR_HIDDEN_DIM, PREDICTOR_HIDDEN_DIM);
    b.add_u32(KEY_DIFFUSION_STEPS, DIFFUSION_STEPS);
    b.add_bool(KEY_USES_STYLE_DIFFUSION, USES_STYLE_DIFFUSION);
    b.add_u32(KEY_DECODER_DIM_IN, DECODER_DIM_IN);
    b.add_u32(KEY_DECODER_GEN_ISTFT_N_FFT, DECODER_GEN_ISTFT_N_FFT);
    b.add_u32(KEY_DECODER_GEN_ISTFT_HOP_SIZE, DECODER_GEN_ISTFT_HOP_SIZE);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    /// A minimal StyleTTS 2 fixture with one F32 tensor named to mirror
    /// an upstream StyleTTS 2 module (`decoder.generator.conv_pre.weight`).
    /// The point is that the pass-through path emits the float tensor
    /// byte-identically — same contract as vits_ja / qwen3-tts /
    /// vibevoice / voxcpm2.
    fn minimal_safetensors_one_f32() -> Vec<u8> {
        let header = r#"{"decoder.generator.conv_pre.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn get_string(file: &GgufFile, key: &str) -> String {
        file.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{key}: missing"))
            .to_owned()
    }

    fn get_u32(file: &GgufFile, key: &str) -> u32 {
        match file.get(key) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_bool(file: &GgufFile, key: &str) -> bool {
        match file.get(key) {
            Some(GgufMetadataValue::Bool(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    #[test]
    fn convert_stamps_unknown_provenance_by_default() {
        let bytes = minimal_safetensors_one_f32();
        let (b, report) = convert(bytes).expect("convert must succeed on a valid safetensors");
        // Every float tensor passes through.
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        // The licence note is always present.
        assert!(
            report.notes.iter().any(|n| n.contains("Unknown")),
            "notes must contain the fail-closed default warning: {:?}",
            report.notes,
        );
        // Provenance is stamped Unknown so the runtime gate refuses to
        // load in commercial mode (M2-13). Round-trip the built GGUF
        // through the reader to confirm.
        let bytes = b.to_bytes().expect("serialize built GGUF");
        let gguf = GgufFile::parse(bytes).expect("built GGUF must round-trip through the reader");
        assert_eq!(
            get_string(&gguf, "vokra.provenance.weight_license"),
            "unknown",
        );
        assert_eq!(get_string(&gguf, chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(get_string(&gguf, chunks::KEY_MODEL_NAME), NAME);
    }

    #[test]
    fn convert_emits_hparam_chunk_group() {
        let bytes = minimal_safetensors_one_f32();
        let (b, _report) = convert(bytes).unwrap();
        let bytes = b.to_bytes().expect("serialize built GGUF");
        let gguf = GgufFile::parse(bytes).unwrap();
        // Every transcribed axis rides its `vokra.styletts2.*` key —
        // primary source `Models/LJSpeech/config.yml` +
        // `Models/LibriTTS/config.yml`.
        assert_eq!(get_u32(&gguf, KEY_SAMPLE_RATE_HZ), 24_000);
        assert_eq!(get_u32(&gguf, KEY_STYLE_DIM), 128);
        assert_eq!(get_u32(&gguf, KEY_HIDDEN_DIM), 512);
        assert_eq!(get_u32(&gguf, KEY_N_MELS), 80);
        assert_eq!(get_u32(&gguf, KEY_TEXT_ENCODER_N_LAYER), 3);
        assert_eq!(get_u32(&gguf, KEY_PREDICTOR_HIDDEN_DIM), 512);
        // LJSpeech single-speaker default → diffusion sampler off.
        assert_eq!(get_u32(&gguf, KEY_DIFFUSION_STEPS), 0);
        assert!(!get_bool(&gguf, KEY_USES_STYLE_DIFFUSION));
        // iSTFTNet decoder axes.
        assert_eq!(get_u32(&gguf, KEY_DECODER_DIM_IN), 512);
        assert_eq!(get_u32(&gguf, KEY_DECODER_GEN_ISTFT_N_FFT), 20);
        assert_eq!(get_u32(&gguf, KEY_DECODER_GEN_ISTFT_HOP_SIZE), 5);
        // Family marker.
        assert_eq!(get_string(&gguf, KEY_MODEL_FAMILY), MODEL_FAMILY);
    }

    #[test]
    fn convert_metadata_only_is_never_silently_permissive() {
        // Even a checkpoint that ships zero float tensors must not
        // silently succeed under a permissive class — the runtime gate
        // still refuses to load. This locks the FR-EX-08 posture.
        let empty = minimal_safetensors_no_tensors();
        let (b, report) = convert(empty).expect("empty is a valid safetensors payload");
        assert_eq!(report.written, 0);
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("no float tensors passed through")),
            "notes must warn on empty payload: {:?}",
            report.notes,
        );
        let bytes = b.to_bytes().expect("serialize built GGUF");
        let gguf = GgufFile::parse(bytes).unwrap();
        assert_eq!(
            get_string(&gguf, "vokra.provenance.weight_license"),
            "unknown",
        );
    }
}
