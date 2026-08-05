//! **NSNet2** (Microsoft DNS Challenge NR baseline): safetensors checkpoint
//! → GGUF conversion (Coverage-audit 2026-08-03 Wave A ticket).
//!
//! Input: the upstream Microsoft DNS-Challenge NR baseline —
//! `NSNet2-baseline/nsnet2-20ms-baseline.onnx` (~2 MB). Because the upstream
//! release is ONNX-only and Vokra's runtime never links ONNX / protobuf
//! (FR-LD-05, NFR-DS-02), the offline sidecar
//! `tools/parity/nsnet2_prepare_checkpoint.py` first bridges ONNX → safetensors;
//! this converter then consumes that safetensors input and stamps the
//! `vokra.model.*` / `vokra.provenance.*` chunk groups a future native
//! `vokra-models::nsnet2::*` implementation will read.
//!
//! # Model class
//!
//! NSNet2 is a 20 ms-frame single-channel noise-suppression baseline (ICASSP
//! 2020, `arXiv:2005.07551`): a 2-layer GRU + 3-Linear mask predictor operating
//! over the 257-bin log-power spectrum of the 16 kHz input (STFT `n_fft=512`,
//! hop 10 ms, 20 ms Hann window). Its role in the Vokra catalogue is the
//! quantization-CI / industry-baseline reference for the `denoise` op family;
//! it is deliberately **weaker** than DeepFilterNet3 (M4-20 T17) but
//! architecturally distinct enough that silently sharing the `denoise` arch tag
//! would misroute the runtime dispatch.
//!
//! # License
//!
//! Both code and weights ship **MIT** end-to-end
//! (`github.com/microsoft/DNS-Challenge/blob/master/LICENSE`, fetched
//! 2026-08-03 — CLAUDE.md「ハルシネーション厳禁」). MIT is a `Permissive`
//! license class — same commercial verdict as apache-2.0 (no runtime-side
//! attribution obligation).
//!
//! # BF16 posture
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the matching
//! GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no convert-time widening
//! — the runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`). Mirror of
//! `emotion2vec` / `ecapa_tdnn` / `qwen3_tts` / `vibevoice` / `voxcpm2` /
//! `moshi` / `voxtral` — the landed sibling posture. NSNet2 itself ships F32
//! (the ONNX release stores every initializer as `FLOAT`), but the pass-through
//! arm keeps the door open for any downstream half-precision variant without a
//! new converter arm.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream initializer names verbatim** as the
//! prep script exports them (e.g. `fc1.weight` / `gru_1.W` / `mask.bias`) —
//! the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM / VibeVoice
//! / emotion2vec contract. Real-weight parity binding is a follow-up wave
//! gated on the upstream tensor-name manifest fetch + license §3.1 sign-off
//! (`docs/license-audit.md`); this converter passes every float tensor through
//! unchanged so a future `Nsnet2Weights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! NSNet2 is distributed as ONNX; this converter **never** touches ONNX
//! directly (FR-LD-05). The offline
//! `tools/parity/nsnet2_prepare_checkpoint.py` sidecar performs the ONNX →
//! safetensors bridge with `onnx` + `numpy` + `safetensors` in a Python venv
//! that is not part of the runtime shipping surface (mirror of
//! `bin_to_safetensors.py`'s posture for pytorch `.bin` inputs). The pipeline
//! will be re-implemented natively when a `crates/vokra-models/src/nsnet2/`
//! lands (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for NSNet2 GGUFs. Distinct from every sibling arch tag
/// (in particular from `denoise` = DeepFilterNet3, which is a completely
/// different topology — DFN3 uses an ERB analysis / synthesis pair around a
/// convolutional recurrent network, whereas NSNet2 is a 2-layer GRU + 3-Linear
/// mask over 257-bin STFT log-magnitude). Silently sharing an arch tag would
/// misroute the runtime dispatch.
pub const ARCH: &str = "nsnet2";

/// `vokra.model.name` value written for the canonical NSNet2 20 ms baseline
/// GGUF. Matches the upstream ONNX filename stem (dashes preserved) so a
/// downstream reader can reconstruct the source from the artifact alone.
pub const NAME: &str = "nsnet2-20ms-baseline";

/// `vokra.model.category` value — `enhancement` (the noise-suppression /
/// speech-enhancement family). Consumed by the model-card generator + zoo
/// manifest tier gate so a NR baseline is not accidentally advertised as an
/// ASR / TTS release.
pub const CATEGORY: &str = "enhancement";

/// `vokra.provenance.upstream_url` value — the GitHub tree the release ships
/// from. NSNet2 is not hosted on HuggingFace (the upstream is Microsoft's
/// public DNS Challenge repository), so this uses `upstream_url` rather than
/// `upstream_hf`; the model-card generator picks up either.
pub const UPSTREAM_URL: &str = "github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline";

/// Canonical weight license SPDX (`mit`). Overrides via the
/// [`convert_nsnet2_file`] `license` parameter — the standing mechanism for
/// "implementation is clean-room MIT but the upstream distributed checkpoint
/// is another license" scenarios (mirror of `convert_file_licensed` in
/// `lib.rs`).
pub const DEFAULT_LICENSE: &str = "mit";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) matching the sibling
/// `emotion2vec` / `ecapa_tdnn` posture until a first-class `category`
/// consumer lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream URL (used for non-HF sources such as
/// GitHub / Zenodo / ModelScope). Sibling to
/// `emotion2vec::KEY_PROVENANCE_UPSTREAM_HF` — kept as a converter-side
/// constant to avoid premature promotion until a second non-HF converter
/// lands.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---- `vokra.nsnet2.*` hparam chunk group ---------------------------------
//
// Mirror of `fsmn_vad::KEY_*` posture: every runtime hparam the future
// `vokra-models::nsnet2::Nsnet2V1::from_gguf` needs is stamped here so a
// downstream reader is fully self-describing (no external config side-car
// needed). Values are FunASR-style `u32` chunks; a `0`-sentinel on any of
// them makes the runtime binder refuse to load (FR-EX-08 — no silent
// default).

/// GGUF metadata key: STFT bin count (u32; upstream = 257 = `n_fft/2 + 1`).
pub const KEY_N_BINS: &str = "vokra.nsnet2.n_bins";
/// GGUF metadata key: GRU / fc_in hidden width (u32; upstream = 400).
pub const KEY_HIDDEN_DIM: &str = "vokra.nsnet2.hidden_dim";
/// GGUF metadata key: `fc_1` output width (u32; upstream = 600).
pub const KEY_FC1_DIM: &str = "vokra.nsnet2.fc1_dim";
/// GGUF metadata key: `fc_2` output width (u32; upstream = 600).
pub const KEY_FC2_DIM: &str = "vokra.nsnet2.fc2_dim";
/// GGUF metadata key: STFT FFT length (u32; upstream = 512).
pub const KEY_N_FFT: &str = "vokra.nsnet2.n_fft";
/// GGUF metadata key: STFT hop (u32 samples; upstream = 160 = 10 ms @ 16 kHz).
pub const KEY_HOP: &str = "vokra.nsnet2.hop";
/// GGUF metadata key: STFT window length (u32 samples; upstream = 320 = 20 ms
/// @ 16 kHz). A window shorter than `n_fft` is centred and zero-padded to
/// `n_fft` by the analysis op.
pub const KEY_WIN_LENGTH: &str = "vokra.nsnet2.win_length";
/// GGUF metadata key: PCM sample rate (u32 Hz; upstream = 16 000).
pub const KEY_SAMPLE_RATE: &str = "vokra.nsnet2.sample_rate";

/// Upstream STFT bin count (`n_fft/2 + 1` for `n_fft = 512`).
pub const DEFAULT_N_BINS: u32 = 257;
/// Upstream GRU / fc_in hidden width.
pub const DEFAULT_HIDDEN_DIM: u32 = 400;
/// Upstream `fc_1` output width.
pub const DEFAULT_FC1_DIM: u32 = 600;
/// Upstream `fc_2` output width.
pub const DEFAULT_FC2_DIM: u32 = 600;
/// Upstream FFT length (samples).
pub const DEFAULT_N_FFT: u32 = 512;
/// Upstream STFT hop (samples).
pub const DEFAULT_HOP: u32 = 160;
/// Upstream STFT window length (samples).
pub const DEFAULT_WIN_LENGTH: u32 = 320;
/// Upstream PCM sample rate (Hz).
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// Outcome of an NSNet2 conversion.
///
/// All counters are additive and default to zero — a zero-tensor checkpoint
/// returns `Nsnet2Report::default()` and the caller remains responsible for
/// surfacing the "no float tensors" loud note (mirror of the
/// `emotion2vec` / `ecapa_tdnn` `Report` pattern).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Nsnet2Report {
    /// Total tensors surfaced by the safetensors reader (the sum of
    /// `written + skipped_non_float`). Pins the budget so a truncated header
    /// cannot silently drop tensors without the caller noticing.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through the
    /// same byte-copy path since the BF16 pass-through landed 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an NSNet2 GGUF to
/// `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url) chunk groups are stamped for the runtime compliance gate
/// (FR-CP-03). `vokra.schema.*` is written unconditionally by the GGUF
/// writer.
///
/// `license` overrides [`DEFAULT_LICENSE`] (`"mit"`) — the same mechanism
/// `lib.rs::convert_file_licensed` uses when the implementation is
/// clean-room but the redistributed checkpoint carries a different SPDX
/// (e.g. `cc-by-4.0`).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_nsnet2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Nsnet2Report, ConvertError> {
    // Whole-file read: NSNet2 ships as a ~2 MB ONNX which the prep script
    // flattens into a similarly tiny safetensors — no need for the
    // streaming path the Moshi 15 GB / Voxtral 8.7 GB converters run.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own licence.
    // NSNet2 ships MIT end-to-end (github.com/microsoft/DNS-Challenge/blob/
    // master/LICENSE, fetched 2026-08-03 — CLAUDE.md「ハルシネーション厳禁」).
    // The `license` override lets a downstream repackager stamp a different
    // SPDX if they redistribute under stricter terms (the same knob
    // `convert_file_licensed` exposes in `lib.rs`).
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some("Microsoft DNS-Challenge NSNet2-baseline (MIT end-to-end)"),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // NSNet2 has one canonical topology — the 20 ms baseline
    // (`nsnet2-20ms-baseline.onnx`) — and every hparam is fixed at that
    // release. Stamping them here (mirror of `fsmn_vad::stamp_hparams`
    // posture) makes the artifact self-describing so the future
    // `vokra-models::nsnet2::Nsnet2V1::from_gguf` binder can validate
    // against these values loudly (FR-EX-08 — a checkpoint that came from a
    // different topology cannot silently misload).
    b.add_u32(KEY_N_BINS, DEFAULT_N_BINS);
    b.add_u32(KEY_HIDDEN_DIM, DEFAULT_HIDDEN_DIM);
    b.add_u32(KEY_FC1_DIM, DEFAULT_FC1_DIM);
    b.add_u32(KEY_FC2_DIM, DEFAULT_FC2_DIM);
    b.add_u32(KEY_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_HOP, DEFAULT_HOP);
    b.add_u32(KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH);
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);

    let mut report = Nsnet2Report::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (`docs/adr/qwen3-tts-bf16.md`, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `emotion2vec::convert_emotion2vec_file` / `ecapa_tdnn::convert_ecapa_tdnn_file`.
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanosecond timestamp — the
    /// emotion2vec / ecapa_tdnn test pattern; no external `tempfile` dep,
    /// preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-nsnet2-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer carrying a single F32 tensor
    /// shaped like NSNet2's first GRU input-weight matrix
    /// (`3*hidden_dim x n_bins = 3*257 x 257 = 771 x 257`, F32) — small
    /// enough to keep the buffer under 1 MB but non-trivial so a
    /// silent-widen bug would surface as a byte diff.
    fn synthetic_f32_safetensors() -> (Vec<u8>, Vec<u8>) {
        // 6 non-zero F32 values reused as a 2x3 shape (so the assertions can
        // pin an exact 24-byte payload). NSNet2's real tensors are larger
        // but the pass-through semantics do not depend on shape; the
        // fixture keeps the CI cache footprint minimal.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(payload.len(), 24, "6 elements x 4 bytes F32 payload");
        // Upstream NSNet2 initializer names are unqualified (`fc1.weight`,
        // `gru_1.W`, `mask.bias` — no module-prefix tree). The fixture uses
        // the recurrent-input `gru_1.W` shape as a representative anchor.
        let header = r#"{"gru_1.W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);
        (buf, payload)
    }

    /// Builds a synthetic safetensors buffer carrying one BF16 tensor
    /// (`shape=[2,3]`) so the pass-through arm and the BF16 subset counter
    /// are exercised even though upstream NSNet2 ships F32 today. Any
    /// future half-precision distillation would land on this arm without a
    /// converter change.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements x 2 bytes BF16 payload");
        let header = r#"{"gru_2.W":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Baseline pass-through pin: an F32 tensor round-trips through the
    /// file-based converter with its dtype preserved and bytes intact, and
    /// the provenance / category / schema chunks land on the artifact.
    #[test]
    fn f32_tensor_passes_through_verbatim() {
        let (input_bytes, payload) = synthetic_f32_safetensors();
        let input = scratch_path("f32-in");
        let output = scratch_path("f32-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_nsnet2_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 1, "one tensor visible in header");
        assert_eq!(report.written, 1, "F32 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "F32 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only input must leave the BF16 subset counter at Default 0"
        );

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("gru_1.W")
            .expect("F32 tensor present after pass-through");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "F32 payload must be byte-identical to input"
        );

        // Provenance + category chunks pinned on the artifact itself.
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
            "category chunk pins NSNet2 as `enhancement`"
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
            "MIT weight license normalises to LicenseClass::Permissive"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL),
            "upstream_url chunk pins the GitHub tree the release ships from"
        );
        // Every `vokra.nsnet2.*` hparam must be stamped verbatim so a
        // downstream `Nsnet2V1::from_gguf` binder validates the topology.
        for (k, want) in [
            (KEY_N_BINS, DEFAULT_N_BINS),
            (KEY_HIDDEN_DIM, DEFAULT_HIDDEN_DIM),
            (KEY_FC1_DIM, DEFAULT_FC1_DIM),
            (KEY_FC2_DIM, DEFAULT_FC2_DIM),
            (KEY_N_FFT, DEFAULT_N_FFT),
            (KEY_HOP, DEFAULT_HOP),
            (KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH),
            (KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE),
        ] {
            let got = file.get(k).and_then(|v| v.as_u64());
            assert_eq!(
                got,
                Some(u64::from(want)),
                "hparam `{k}` must be stamped as {want}"
            );
        }
        // Schema stamp is written unconditionally by the GGUF writer.
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

    /// BF16 pass-through pin: even though upstream NSNet2 is F32, any
    /// future half-precision distillation must ride the same arm without a
    /// converter change. The dtype must stay BF16 (GGUF type 30) and the
    /// payload must be byte-identical (a silent widen would still round-trip
    /// values but would break the byte pin).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_nsnet2_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 1, "one tensor visible in header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of ecapa_tdnn)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("gru_2.W")
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

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Licence override pin: passing `Some("cc-by-4.0")` re-derives the
    /// class through `LicenseClass::from_license_str` and stamps the new
    /// SPDX + class on the artifact. Guards against a hard-coded
    /// `Permissive` regression when a downstream repackager needs to
    /// override the stamped default.
    #[test]
    fn license_override_re_derives_class() {
        let (input_bytes, _payload) = synthetic_f32_safetensors();
        let input = scratch_path("override-in");
        let output = scratch_path("override-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let _report =
            convert_nsnet2_file(&input, &output, Some("cc-by-4.0")).expect("convert with override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-4.0"),
            "override SPDX lands verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str()),
            "cc-by-4.0 normalises to LicenseClass::AttributionRequired (not Permissive)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
