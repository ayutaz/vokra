//! **ReDimNet** (`Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`,
//! apache-2.0): safetensors → GGUF conversion (Wave 4, speaker-fleet
//! extension, 2026-08-14).
//!
//! ReDimNet = **Reshape Dimensions Network** — a speaker-embedding
//! network that alternates 2D CNN dim-reduction blocks (`basic_resnet`)
//! with 1D `conv+att` transformer-lite blocks and pools through
//! **Attentive Statistics Pooling (ASTP)** to a 192-d speaker
//! embedding. Paper: arXiv:2402.01049 ("Reshape Dimensions Network for
//! Speaker Recognition"). Upstream: the `Wespeaker` HF org
//! (`wespeaker-voxceleb-redimnet2-B6-LM`) — the WeSpeaker/VoxCeleb
//! release fine-tuned via the Large-Margin (LM) stage sibling to
//! `Wespeaker/wespeaker-voxceleb-resnet34-LM` (converted separately
//! by `crates/vokra-convert/src/models/wespeaker.rs`).
//!
//! # Vokra scope — speaker fleet extension over wespeaker/titanet/ecapa_tdnn
//!
//! Complements the sibling speaker converters:
//! - `wespeaker` (`Wespeaker/wespeaker-voxceleb-resnet34-LM`) —
//!   ResNet-34 backbone, ~25 MB.
//! - `ecapa_tdnn` (`speechbrain/spkrec-ecapa-voxceleb`) — ECAPA-TDNN
//!   backbone.
//! - `titanet` (`nvidia/speakerverification_en_titanet_large`) —
//!   depth-wise separable Conv1D backbone.
//! - `speaker_3d` (`iic/speech_eres2net_sv_zh-cn_16k-common`) —
//!   ERes2Net backbone.
//!
//! ReDimNet2 (the "B6-LM" release) uses a distinct **2D
//! dim-reduction + 1D conv+att** hybrid backbone that no sibling
//! covers — the whole speaker fleet ships as loud-partial (from_gguf
//! real, encode = UnsupportedOp) today, and this converter is one
//! more strand for the future WeSpeaker Python source transcription
//! wave. Distinct arch tag `redimnet` (never `wespeaker`,
//! `ecapa_tdnn`, `titanet`, `speaker_3d`, or `campplus` — silently
//! sharing an arch would misroute runtime dispatch, FR-EX-08).
//! Category `speaker` (mirror of the sibling speaker fleet — the
//! converter fleet groups speaker-embedding / verification networks
//! under one category so downstream consumers pick a load path
//! without inspecting the arch).
//!
//! # License posture — apache-2.0 (HF cardData primary source)
//!
//! HF `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM` cardData
//! declares `license: apache-2.0` (scout-time WebFetch, 2026-08-14).
//! Registered explicitly in
//! `crates/vokra-core/src/compliance/license_class.rs` under the
//! WeSpeaker family bucket — the existing `wespeaker` prefix walk
//! already resolves this id to `Permissive`, but an explicit
//! id-lookup registration matches the sibling
//! `whisper-large-v3-turbo-german` / `speechbrain-spkrec-ecapa-voxceleb`
//! pattern for callers who don't want to depend on prefix-walk
//! semantics.
//!
//! §3.1 sign-off column is **BLANK** in `docs/license-audit.md`
//! (fail-closed default — CC MUST NOT sign a license row, that is
//! owner-only per memory `[[feedback-license-signoff-primary-source]]`).
//!
//! # Scale — local convert OK (~55.5 MB `avg_model.pt`)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! `[[feedback-large-models-on-vast-ai]]`: <2 GB safe). No vast.ai
//! handoff needed. The upstream release ships `avg_model.pt` (torch
//! pickle averaged across the LM fine-tune ensemble), bridged
//! offline to safetensors through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` flow (uv-managed Python
//! 3.12 sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`); this converter accepts safetensors
//! only (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # No ONNX / no pickle (permanent)
//!
//! ReDimNet ships as PyTorch pickle upstream (`avg_model.pt`); this
//! converter **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02). Runtime tree carries neither `torch` nor `onnxruntime`.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16 is
//! emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (the same
//! choke point every sibling BF16 pass-through converter binds
//! against — never fabricated fp32 conversions elsewhere in the
//! tree).
//!
//! # Wiring status — WIRED (2026-08-15)
//!
//! BF16 / F16 / F32 pass-through + provenance / category / topology
//! chunk stamps, reachable end to end:
//!
//! - `ModelKind::Redimnet` with `from_arg("redimnet")` /
//!   `as_arg() == "redimnet"` — the exact spelling the runtime binder
//!   `crates/vokra-models/src/redimnet/mod.rs` names in its four
//!   recovery messages;
//! - a `convert_file_licensed` dispatch arm and a `verify()` arm in
//!   `vokra-convert/src/main.rs`;
//! - `pub use models::redimnet::{RedimnetReport, convert_redimnet_file}`
//!   in `lib.rs`, without which the whole module is unreachable through
//!   the private `mod models`.
//!
//! This module previously carried a module-level `#![allow(dead_code)]`
//! while none of the above existed. That attribute was load-bearing
//! camouflage: it silenced the warnings that were correctly reporting
//! an unreachable module, so nothing surfaced the fact that the binder
//! demanded a GGUF no entry point could produce. It is deleted rather
//! than kept "just in case" — if a `pub` item here ever goes dead
//! again, that is a fact worth learning from the compiler.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for ReDimNet GGUFs. Distinct from every sibling
/// speaker converter arch — never `wespeaker` (ResNet-34), never
/// `ecapa_tdnn` (TDNN stack), never `titanet` (depth-wise separable
/// Conv1D), never `speaker_3d` (ERes2Net), never `campplus`
/// (CAM++). Silently sharing an arch would misroute runtime dispatch
/// (FR-EX-08).
pub const ARCH: &str = "redimnet";

/// `vokra.model.name` — canonical `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`
/// release, lowercase (HF org uses capital `LM` but our arch-tag /
/// slug space is lowercase-only per the whole speaker fleet
/// convention — see wespeaker's `NAME` = `"wespeaker-voxceleb-resnet34-lm"`).
pub const NAME: &str = "wespeaker-voxceleb-redimnet2-b6-lm";

/// `vokra.model.category` — speaker (mirror of the sibling speaker
/// fleet — wespeaker / ecapa_tdnn / titanet / speaker_3d all stamp
/// this same category so downstream consumers can dispatch to a
/// shared speaker-embedding load path).
pub const CATEGORY: &str = "speaker";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`. The slug preserves upstream casing
/// (`Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`).
pub const UPSTREAM_HF: &str = "Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM";

/// Default SPDX. Upstream HF `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`
/// cardData declares `license: apache-2.0` (scout-time WebFetch,
/// 2026-08-14). Overridable through the `license` argument.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// ---------------------------------------------------------------------------
// ReDimNet2 "B6-LM" hyperparameters — transcribed verbatim from the
// upstream `config.yaml` of `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`
// (scout-time transcription, 2026-08-14). Stamped on the GGUF so the
// runtime binder can validate topology + surface embed_dim without
// re-inspecting tensor shapes.
//
// Sources:
// - Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM/config.yaml (HF)
// - github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/redimnet2.py
// - arXiv:2402.01049 (Yakovlev et al. 2024 "Reshape Dimensions Network
//   for Speaker Recognition")
// ---------------------------------------------------------------------------

/// Speaker embedding dimension (`embed_dim`) — 192.
pub const EMBED_DIM: u32 = 192;
/// Output channel count of the last 1D `conv+att` block — 224.
pub const OUT_CHANNELS: u32 = 224;
/// ReDimNet2 channel expansion base (`C`) — 64.
pub const C: u32 = 64;
/// ReDimNet2 mel-frequency dim after the 2D stem (`F`) — 72. Matches
/// [`N_MELS`] (the 2D dim-reduction stem preserves the frequency
/// axis of the input mel-spec).
pub const F: u32 = 72;
/// Log-mel filterbank count fed into the 2D stem — 72.
pub const N_MELS: u32 = 72;
/// STFT window size (`n_fft`) — 512 samples.
pub const N_FFT: u32 = 512;
/// STFT hop size (`hop_length`) — 160 samples (10 ms at 16 kHz).
pub const HOP_LENGTH: u32 = 160;
/// STFT window length (`win_length`) — 400 samples (25 ms at 16 kHz).
pub const WIN_LENGTH: u32 = 400;
/// Audio sample rate — 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16000;
/// Log-mel lower frequency (Hz) — 20.
pub const F_MIN: u32 = 20;
/// Log-mel upper frequency (Hz) — 7600.
pub const F_MAX: u32 = 7600;
/// Pre-emphasis flag (`do_preemph`) — 1 (upstream `config.yaml`
/// enables 0.97 pre-emphasis on the raw waveform).
pub const DO_PREEMPH: u32 = 1;

// ---------------------------------------------------------------------------
// GGUF chunk keys — mirror of
// `crates/vokra-models/src/redimnet/mod.rs` `GGUF_KEY_*` (see runtime
// binder module doc for the cross-crate duplication rationale —
// `vokra-models` must not gain a dep edge onto `vokra-convert`).
// ---------------------------------------------------------------------------

/// `vokra.model.category` — auxiliary category stamp (not covered by
/// `vokra_core::stamp_provenance`, which handles SPDX + class +
/// model_id + source only).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// `vokra.provenance.upstream_hf` — auxiliary provenance stamp.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const KEY_REDIMNET_EMBED_DIM: &str = "vokra.redimnet.embed_dim";
const KEY_REDIMNET_OUT_CHANNELS: &str = "vokra.redimnet.out_channels";
const KEY_REDIMNET_C: &str = "vokra.redimnet.c";
const KEY_REDIMNET_F: &str = "vokra.redimnet.f";
const KEY_REDIMNET_N_MELS: &str = "vokra.redimnet.n_mels";
const KEY_REDIMNET_N_FFT: &str = "vokra.redimnet.n_fft";
const KEY_REDIMNET_HOP_LENGTH: &str = "vokra.redimnet.hop_length";
const KEY_REDIMNET_WIN_LENGTH: &str = "vokra.redimnet.win_length";
const KEY_REDIMNET_SAMPLE_RATE: &str = "vokra.redimnet.sample_rate";
const KEY_REDIMNET_F_MIN: &str = "vokra.redimnet.f_min";
const KEY_REDIMNET_F_MAX: &str = "vokra.redimnet.f_max";
const KEY_REDIMNET_DO_PREEMPH: &str = "vokra.redimnet.do_preemph";

const UPSTREAM_SOURCE: &str = "Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM \
     (ReDimNet2 speaker encoder, VoxCeleb + Large-Margin, 2D basic_resnet + 1D conv+att + ASTP \
     pooling → 192-d embedding, ~55.5 MB avg_model.pt, arXiv:2402.01049, apache-2.0)";

/// Outcome of a ReDimNet conversion. Mirrors the counter shape of
/// [`crate::models::wespeaker::WespeakerReport`] — the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RedimnetReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling wespeaker /
    /// ecapa_tdnn / titanet / speaker_3d reports).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (subset counter). Emits GGUF type 30 verbatim; the runtime
    /// widens BF16 → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`
/// safetensors checkpoint at `input` into a Vokra-native GGUF at
/// `output`, returning a [`RedimnetReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` /
/// `vokra.redimnet.*` chunks are stamped for the runtime compliance
/// gate (FR-CP-03) and the runtime binder topology validation.
///
/// # License override
///
/// `license` overrides the default apache-2.0 SPDX string stamped on
/// `vokra.provenance.license` (whisper / kokoro-family override
/// pattern — see `convert_file_licensed` in `lib.rs`). `None` keeps
/// the built-in apache-2.0 stamp. Unlike MT3 (where the weight
/// license is UNCLEAR and the class is hard-mapped to Unknown), the
/// ReDimNet weight license is confirmed apache-2.0 through HF
/// cardData primary source, so the class resolves via
/// `LicenseClass::from_license_str` for whatever SPDX the caller
/// passes.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_redimnet_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<RedimnetReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // ReDimNet2 "B6-LM" topology axes (transcribed verbatim from the
    // upstream config.yaml and `wespeaker/models/redimnet2.py`). The
    // runtime binder is a strict loader: every axis is required
    // (FR-EX-08 — no primary-source constant fallback since a partial
    // stamp would fabricate axes without primary-source backing).
    b.add_u32(KEY_REDIMNET_EMBED_DIM, EMBED_DIM);
    b.add_u32(KEY_REDIMNET_OUT_CHANNELS, OUT_CHANNELS);
    b.add_u32(KEY_REDIMNET_C, C);
    b.add_u32(KEY_REDIMNET_F, F);
    b.add_u32(KEY_REDIMNET_N_MELS, N_MELS);
    b.add_u32(KEY_REDIMNET_N_FFT, N_FFT);
    b.add_u32(KEY_REDIMNET_HOP_LENGTH, HOP_LENGTH);
    b.add_u32(KEY_REDIMNET_WIN_LENGTH, WIN_LENGTH);
    b.add_u32(KEY_REDIMNET_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(KEY_REDIMNET_F_MIN, F_MIN);
    b.add_u32(KEY_REDIMNET_F_MAX, F_MAX);
    b.add_u32(KEY_REDIMNET_DO_PREEMPH, DO_PREEMPH);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (HF cardData primary source
    // 2026-08-14). `license` overrides for callers who obtained the
    // weight under a different SPDX.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = RedimnetReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as wespeaker /
    // ecapa_tdnn / titanet / speaker_3d; runtime widens BF16 → f32
    // exactly at load via `vokra_core::gguf::quant::decode_bf16`
    // (`bits << 16` is exact).
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
    /// caller-supplied raw payload. Mirror of the wespeaker test
    /// harness — same JSON header shape.
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
    /// with caller-supplied payloads. Mirror of the wespeaker test
    /// harness.
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
            "vokra-redimnet-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    // -----------------------------------------------------------------------
    // 1. BF16 round-trip (byte-identical, counter surfaces, provenance
    //    stamps landed)
    // -----------------------------------------------------------------------

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

        // Mirror a plausible upstream ReDimNet2 tensor name. The
        // `speaker.` prefix + a stem block weight is a realistic
        // string shape, not a synthetic one.
        let input_bytes =
            safetensors_one_bf16("speaker.stem.dim_reduction.0.conv.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_redimnet_file(&input_path, &output_path, None)
            .expect("convert_redimnet_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror wespeaker)"
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
            .tensor_info("speaker.stem.dim_reduction.0.conv.weight")
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

        // Provenance stamps landed on the arch / name / category /
        // upstream-hf axes.
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
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    // -----------------------------------------------------------------------
    // 2. Mixed F32 / F16 round-trip with metadata assertions
    // -----------------------------------------------------------------------

    #[test]
    fn f32_and_f16_tensors_pass_through_with_full_metadata() {
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
            "speaker.pooling.astp.linear.weight",
            &[1, 2],
            &f32_bytes,
            "speaker.head.projection.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_redimnet_file(&input_path, &output_path, None)
            .expect("convert_redimnet_file must accept a mixed F32/F16 checkpoint");

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
        // AND the arch / provenance / category / topology stamps land.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("speaker.pooling.astp.linear.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("speaker.head.projection.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Task-spec pins: `KEY_MODEL_ARCH` / `KEY_MODEL_CATEGORY` /
        // `KEY_PROVENANCE_UPSTREAM_HF` all land.
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
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    // -----------------------------------------------------------------------
    // 3. Full topology chunk group is stamped and readable
    // -----------------------------------------------------------------------

    #[test]
    fn topology_chunks_round_trip() {
        // A single F32 tensor is enough to trigger the write path;
        // all 12 topology axes must round-trip through the GGUF.
        let f32_bytes: Vec<u8> = [0.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!(
            r#"{{"dummy":{{"dtype":"F32","shape":[1],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&f32_bytes);
        let input_path = write_temp("topology-in", &input);
        let output_path = write_temp("topology-out", &[]);

        convert_redimnet_file(&input_path, &output_path, None).expect("convert");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // Every stamped topology axis must round-trip; a rename would
        // land here in the same commit or fail this test.
        assert_eq!(
            file.get(KEY_REDIMNET_EMBED_DIM).and_then(|v| v.as_u64()),
            Some(u64::from(EMBED_DIM))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_OUT_CHANNELS).and_then(|v| v.as_u64()),
            Some(u64::from(OUT_CHANNELS))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_C).and_then(|v| v.as_u64()),
            Some(u64::from(C))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_F).and_then(|v| v.as_u64()),
            Some(u64::from(F))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_N_MELS).and_then(|v| v.as_u64()),
            Some(u64::from(N_MELS))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_N_FFT).and_then(|v| v.as_u64()),
            Some(u64::from(N_FFT))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_HOP_LENGTH).and_then(|v| v.as_u64()),
            Some(u64::from(HOP_LENGTH))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_WIN_LENGTH).and_then(|v| v.as_u64()),
            Some(u64::from(WIN_LENGTH))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(SAMPLE_RATE))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_F_MIN).and_then(|v| v.as_u64()),
            Some(u64::from(F_MIN))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_F_MAX).and_then(|v| v.as_u64()),
            Some(u64::from(F_MAX))
        );
        assert_eq!(
            file.get(KEY_REDIMNET_DO_PREEMPH).and_then(|v| v.as_u64()),
            Some(u64::from(DO_PREEMPH))
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
