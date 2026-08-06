#![allow(clippy::doc_lazy_continuation)]
//! **WavTokenizer-large-speech-75token** (`novateur/WavTokenizer-large-speech-75token`,
//! MIT): pytorch-pickle `.ckpt` → safetensors → GGUF conversion
//! (Wave 3 codec add, 2026-08-01).
//!
//! Input: the upstream `novateur/WavTokenizer-large-speech-75token`
//! release — a single-codebook FSQ neural audio codec (Ji et al. 2024,
//! arXiv:2408.16532) at **24 kHz** with `hop_length = 320` → **75
//! tokens/sec** (24 000 / 320 = 75, hence the repo slug's "75token"
//! suffix). Single upstream sibling `wavtokenizer_large_speech_320_v2.ckpt`
//! (~1.75 GB, verified 2026-08-01 via `https://huggingface.co/api/models/
//! novateur/WavTokenizer-large-speech-75token`) ships as a **torch pickle
//! Lightning-style `.ckpt`** — there is no `model.safetensors` mirror.
//! Callers must pre-flatten to safetensors offline via a dedicated
//! `tools/parity/wavtokenizer_prepare_checkpoint.py` bridge (the DFN3 /
//! DAC / CSM / SpeechT5-HiFi-GAN / FCPE pattern), keeping Vokra's Rust
//! converter safetensors-only by design (NFR-DS-02 zero-dep + FR-LD-05
//! no pickle in the runtime).
//!
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream state-dict name, plus the `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future native WavTokenizer loader
//! will read.
//!
//! # Provenance
//!
//! - **HF path**: `novateur/WavTokenizer-large-speech-75token`.
//! - **License (SPDX)**: `mit` — end-to-end (Novateur WavTokenizer code
//!   under `jishengpeng/WavTokenizer` MIT, primary source = HF cardData
//!   `license: mit`, fetched 2026-08-01 via
//!   `https://huggingface.co/api/models/novateur/
//!   WavTokenizer-large-speech-75token` — CLAUDE.md 「ハルシネーション
//!   厳禁」).
//! - **Category**: `codec` — audio codec (waveform → discrete tokens →
//!   waveform). The category tag is written under the raw
//!   `vokra.model.category` key so the model-card tooling can classify
//!   without reaching into per-converter constants (mirror of neucodec /
//!   xcodec2 / focalcodec).
//!
//! # WavTokenizer vs sibling codecs
//!
//! Distinct arch tag from every sibling codec (Mimi / DAC / neucodec /
//! xcodec2 / focalcodec / speechtokenizer / bicodec / xy_tokenizer /
//! funcodec / step_audio2_mini):
//!
//! - **X-Codec 2** (HKUSTAudio/xcodec2): FSQ codec but **cc-by-nc-4.0**
//!   (NonCommercial); topology + frame rate differ.
//! - **Neucodec** (neuphonic/neucodec): FSQ codec at 24 kHz, but 50 Hz
//!   frame rate (0.8 kbps @ 50 Hz), not 75 Hz.
//! - **FocalCodec** (lucadellalib/focalcodec_*): focal-modulation
//!   single-codebook, not FSQ; different family entirely.
//! - **DAC / Mimi**: RVQ (residual vector quantization), multi-codebook
//!   residual chain — decode paths differ (RVQ residual sum vs FSQ
//!   single-lookup).
//!
//! The `ARCH = "wavtokenizer"` tag matches the existing runtime
//! `vokra_ops::fsq_codec::wavtokenizer_vq` op naming (M4-16 landing,
//! `crates/vokra-ops/src/fsq_codec.rs`) and the
//! `vokra_core::compliance::license_class` registry exact-match arm
//! (`"dac" | "wavtokenizer" => Permissive`). Silently sharing an arch
//! with a sibling FSQ codec would mis-route the runtime dispatch.
//!
//! **Registry note**: `registry_lookup` resolves the bare `"wavtokenizer"`
//! model_id via the exact-match arm; the variant-specific model_id
//! (`"wavtokenizer-large-speech-75token"`) that this converter stamps
//! falls through in the current registry (no prefix walker landed for
//! this family). This is **not** a load-time fault because the
//! resolver's Path 1 (`vokra.provenance.weight_license`) fires first
//! from the `stamp_provenance` output below and returns `Permissive`
//! directly — the registry_lookup Path 3 is only consulted when the
//! explicit class stamp is missing (e.g. hand-hacked GGUF without a
//! license triple). A defensive family prefix walker (mirror of the
//! focalcodec pattern at `license_class.rs:963`) can be added when a
//! second WavTokenizer variant lands — deferred until that variant
//! actually exists, to keep this landing minimal.
//!
//! # BF16 pass-through (mirror of neucodec / xcodec2 / speecht5_hifigan)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single
//! choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 is the top 16 bits of an f32 — `bits << 16` is exact). The
//! observability counter [`WavtokenizerReport::bf16_passthrough`]
//! records how many BF16 tensors landed on this arm so a silent widen
//! / downcast cannot slip in undetected. Upstream `.ckpt` is F32 at
//! rest (verified 2026-08-01), so the BF16 arm is defensive today; the
//! counter is kept for future BF16-quantized derivative releases.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict names verbatim**
//! (the CSM / Kokoro / neucodec / xcodec2 / focalcodec / speecht5_hifigan
//! contract). The offline `wavtokenizer_prepare_checkpoint.py` bridge
//! is responsible for extracting `state_dict` from the Lightning
//! `.ckpt` wrapper (strip the `optimizer_states` / `hparams` /
//! `pytorch-lightning_version` shells) and re-emitting the tensors
//! under their bare module names. Real-weight parity vs the upstream
//! `WavTokenizer` Python reference (encode → codes → decode) is deferred
//! to owner (`docs/license-audit.md` §3.1 sign-off queue); the M4-16
//! op-side (`wavtokenizer_vq`, `crates/vokra-ops/src/fsq_codec.rs`) already
//! runs the FSQ decode on a synthetic projection.
//!
//! # No ONNX / no pickle in runtime (permanent)
//!
//! WavTokenizer ships PyTorch Lightning pickle `.ckpt`; this converter
//! **never** touches ONNX (FR-LD-05) and **never** touches pickle
//! (NFR-DS-02 zero-dep). The pipeline is re-implemented natively in a
//! future `crates/vokra-models/src/wavtokenizer/` module (whisper.cpp
//! 型 self re-implementation, CLAUDE.md 設計判断 4). Between now and
//! that landing, the runtime consumer walks the emitted tensor names
//! and either succeeds or fails loudly per FR-EX-08 — today's converter
//! surface is byte-exact provenance + tensor-name preservation only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for WavTokenizer GGUFs.
///
/// Matches the existing runtime `vokra_ops::fsq_codec::wavtokenizer_vq`
/// op naming (M4-16 landing) and the `vokra_core::compliance::
/// license_class` registry exact-match arm (`"dac" | "wavtokenizer" =>
/// Permissive`). Intentionally distinct from every sibling codec
/// (`xcodec2`, `neucodec`, `focalcodec`, `mimi`, `dac`, `funcodec`,
/// `speechtokenizer`, `bicodec`, `xy_tokenizer`, `step_audio2_mini`) —
/// see the module-level docstring for the differentiation matrix.
pub const ARCH: &str = "wavtokenizer";

/// `vokra.model.name` value written for the canonical WavTokenizer
/// large-speech-75token GGUF (the variant-specific display name a
/// consumer sees when inspecting a converted artifact).
///
/// The variant-specific spelling mirrors the focalcodec pattern
/// (`focalcodec-25hz` / `focalcodec-12-5hz`) so a future
/// `WavTokenizer-large-unify-40token` or `WavTokenizer-base-40token`
/// variant lands as a distinct `NAME` under the shared `ARCH` tag —
/// the same shared-arch / distinct-name split focalcodec's variant
/// enum uses. Today's single-variant landing does not need the enum;
/// it can be added later when a second variant lands.
pub const NAME: &str = "wavtokenizer-large-speech-75token";

/// `vokra.model.category` value written for every WavTokenizer GGUF.
/// Same tag as the sibling neucodec / xcodec2 / focalcodec / mimi / dac
/// (all `codec`), used by the model-card generator classifier.
pub const CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator.
pub const UPSTREAM_HF: &str = "novateur/WavTokenizer-large-speech-75token";

/// Default upstream weight licence (SPDX). Novateur WavTokenizer family
/// ships MIT end-to-end (HF cardData `license: mit`, verified
/// 2026-08-01 via HF API). Callers can override at the outer
/// `convert_wavtokenizer_file(_, _, license=Some(_))` boundary when the
/// source distribution declares a different SPDX id — the Whisper /
/// kokoro / xcodec2 override pattern.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of the sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a WavTokenizer conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::neucodec::NeucodecReport`],
/// [`super::xcodec2::XCodec2Report`],
/// [`super::speecht5_hifigan::Speecht5HifiganReport`]) adapted to the
/// file-oriented `convert_wavtokenizer_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WavtokenizerReport {
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

/// Converts a `novateur/WavTokenizer-large-speech-75token` safetensors
/// checkpoint at `input` into a Vokra-native GGUF at `output`, returning
/// a [`WavtokenizerReport`].
///
/// The upstream distribution is a **torch pickle Lightning `.ckpt`** —
/// callers must first flatten it to safetensors via the offline
/// `tools/parity/wavtokenizer_prepare_checkpoint.py` bridge (the DFN3
/// / DAC / CSM / SpeechT5-HiFi-GAN pattern). This function accepts
/// safetensors only.
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"mit"`, `Permissive`) — the upstream HF
/// release ships MIT end-to-end.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_wavtokenizer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<WavtokenizerReport, ConvertError> {
    // WavTokenizer-large-speech-75token is ~1.75 GB in the upstream
    // pickle .ckpt (verified 2026-08-01 via HF file listing); after the
    // offline `wavtokenizer_prepare_checkpoint.py` bridge strips the
    // Lightning wrapper (optimizer_states / hparams / pytorch-lightning_
    // version) the safetensors is ~1.5-1.7 GB of pure F32 tensor
    // payload. Still 1 order of magnitude smaller than the
    // streaming-mandated Moshi 14 GiB tier, so the simple
    // `std::fs::read` posture the sibling non-streaming BF16
    // pass-through converters use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Default provenance stamp — Permissive MIT end-to-end (upstream
    // `novateur/WavTokenizer-large-speech-75token` model card
    // `license: mit`, Novateur code lineage from `jishengpeng/
    // WavTokenizer` MIT). The optional `license` argument overrides
    // below.
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
            "novateur/WavTokenizer-large-speech-75token \
             (single-codebook FSQ audio codec, 24 kHz, hop 320 → 75 tok/s, \
             arXiv:2408.16532, MIT)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = WavtokenizerReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR (mirror of
    // neucodec / xcodec2 / speecht5_hifigan / wespeaker / ecapa_tdnn /
    // focalcodec); the runtime widens BF16 → f32 exactly at load via the
    // single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    // decode_bf16`.
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

    /// Builds an F32 tensor safetensors buffer — matches upstream
    /// WavTokenizer dtype (F32 verified via HF API 2026-08-01).
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 4;
        assert_eq!(f32_bytes.len(), expected);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out
    }

    /// Builds a mixed F32 + F16 safetensors buffer using realistic
    /// upstream `WavTokenizer` state-dict tensor names:
    ///   `feature_extractor.encoder.conv_pre.weight` — F32, `[1,2]` →  8 bytes @ [0..8)
    ///   `quantizer.vq.codebook`                     — F16, `[2,3]` → 12 bytes @ [8..20)
    fn safetensors_f32_and_f16() -> Vec<u8> {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling:
        //   1.0 = 0x3C00, -2.0 = 0xC000, -0.5 = 0xB800,
        //   3.0 = 0x4200, 0.15625 = 0x3100, 42.0 = 0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);
        let header = format!(
            r#"{{"feature_extractor.encoder.conv_pre.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"quantizer.vq.codebook":{{"dtype":"F16","shape":[2,3],"data_offsets":[{},{}]}}}}"#,
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
            "vokra-wavtokenizer-{kind}-{}-{}.bin",
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
    fn f32_tensor_passes_through_and_stamps_land() {
        // Upstream WavTokenizer-large-speech-75token is F32
        // (verified 2026-08-01 via HF file listing on the single
        // `wavtokenizer_large_speech_320_v2.ckpt` sibling) — this test
        // pins the primary code path.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from WavTokenizer's
        // encoder module tree (`feature_extractor.encoder.*` in the
        // Novateur state-dict).
        let input_bytes = safetensors_one_f32(
            "feature_extractor.encoder.conv_pre.weight",
            &[2, 3],
            &f32_bytes,
        );
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_wavtokenizer_file(&input_path, &output_path, None)
            .expect("convert_wavtokenizer_file must accept a well-formed F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("feature_extractor.encoder.conv_pre.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance / category chunks landed.
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
            "vokra.model.category must be `codec`",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16-quantized derivatives should
        // ride the same arm as the sibling BF16-pass-through
        // converters (neucodec / xcodec2 / speecht5_hifigan).
        // Non-zero BF16 bit patterns so a subsequent byte-identity
        // assert catches any silent widen / downcast attempt (zeroed
        // payloads would round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic upstream FSQ codebook tensor name.
        let input_bytes = safetensors_one_bf16("quantizer.vq.codebook", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_wavtokenizer_file(&input_path, &output_path, None)
            .expect("convert_wavtokenizer_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror neucodec / xcodec2)"
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
            .tensor_info("quantizer.vq.codebook")
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
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let input_bytes = safetensors_f32_and_f16();
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_wavtokenizer_file(&input_path, &output_path, None)
            .expect("convert_wavtokenizer_file must accept a mixed F32/F16 checkpoint");

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
            .tensor_info("feature_extractor.encoder.conv_pre.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");

        let f16_info = file
            .tensor_info("quantizer.vq.codebook")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");

        // Arch / name land here too so a copy-paste regression that
        // swapped ARCH / NAME cannot slip through even if the F32 test
        // above is skipped in isolation.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        // The default license path must stamp mit / Permissive (the
        // whole point of not being cc-by-nc-4.0 like the sibling
        // xcodec2 — WavTokenizer is MIT end-to-end).
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

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_flows_through() {
        // A user who re-trained on a permissive corpus supplies a
        // different SPDX id at conversion time — the override must land
        // on KEY_PROVENANCE_LICENSE + KEY_PROVENANCE_WEIGHT_LICENSE and
        // the LicenseClass must be re-derived by from_license_str.
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32(
            "feature_extractor.encoder.conv_pre.weight",
            &[1, 2],
            &f32_bytes,
        );
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_wavtokenizer_file(&input_path, &output_path, Some("apache-2.0"))
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

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// stamp — that would be a silent research-flag downgrade. The
    /// `Some(s) if !s.is_empty()` guard in `convert_wavtokenizer_file`
    /// keeps the default MIT / Permissive stamp (mirror of xcodec2's
    /// empty-string guard test).
    #[test]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_bytes: Vec<u8> = [0.5_f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32(
            "feature_extractor.encoder.conv_pre.weight",
            &[1, 2],
            &f32_bytes,
        );
        let input_path = write_temp("empty-in", &input_bytes);
        let output_path = write_temp("empty-out", &[]);

        convert_wavtokenizer_file(&input_path, &output_path, Some(""))
            .expect("empty override must succeed and be ignored");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "empty string must NOT downgrade the license stamp"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "empty string must NOT downgrade the class"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
