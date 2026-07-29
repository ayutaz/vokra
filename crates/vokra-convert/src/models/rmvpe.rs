//! **RMVPE** (Robust Model for Vocal Pitch Estimation): safetensors →
//! GGUF conversion (F0 pitch-extractor tier, 2026-07-30).
//!
//! Input: an offline `.pt` → safetensors flattening (via
//! `tools/parity/nemo_pt_to_safetensors.py`) of the upstream
//! `yxlllc/RMVPE` (fork of `Dream-High/RMVPE`) release. Output: a GGUF
//! carrying every float tensor plus the `vokra.rmvpe.*` metadata chunk
//! group a native `crates/vokra-models/src/f0/rmvpe.rs` binder will
//! read.
//!
//! # Model class
//!
//! RMVPE (Wei et al. 2023) is a CNN + GRU U-Net polyphonic vocal pitch
//! extractor:
//!
//! ```text
//! PCM (16 kHz mono)
//!   -> mel spectrogram (n_mels=128, hop=160, win=1024, n_fft=2048)
//!   -> U-Net encoder (5 down blocks: Conv2d + BN + LReLU * N, then
//!      MaxPool2d)
//!   -> intermediate GRU (bidirectional, hidden ~256)
//!   -> U-Net decoder (5 up blocks: ConvTranspose2d + skip + Conv2d + BN
//!      + LReLU * N)
//!   -> 360-pitch-class head (Conv1d → Sigmoid → per-class probability
//!      over a log-Hz grid from ~30 Hz to ~1000 Hz, 20 cents per class)
//! ```
//!
//! This is the pitch front-end **required by RVC v2** and is commonly
//! reused by other singing-voice / voice-conversion (GPT-SoVITS,
//! Retrieval-based VC) pipelines. It shares the "per-hop F0 track"
//! output contract with the CREPE / FCPE / PyIN / Harvest siblings in
//! `vokra-models::f0`.
//!
//! # License
//!
//! **MIT** end-to-end — code + weights ship under the standard MIT
//! license (upstream `github.com/Dream-High/RMVPE/blob/main/LICENSE` +
//! `github.com/yxlllc/RMVPE/blob/main/LICENSE`, both fetched
//! 2026-07-30 — CLAUDE.md 「ハルシネーション厳禁」). MIT classifies as
//! [`LicenseClass::Permissive`] — same commercial verdict as
//! apache-2.0 (no runtime-side attribution obligation, unlike the CC-BY
//! codec weights).
//!
//! # BF16 posture
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of the emotion2vec / qwen3_tts /
//! vibevoice / voxcpm2 / moshi / voxtral posture that keeps the CI
//! cache footprint at the smallest tensor payload while preserving the
//! exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream flattened `state_dict` keys
//! verbatim** (the emotion2vec / kokoro / deberta_v2 contract). The
//! runtime `RmvpeWeights::from_gguf` walks the same names — a missing
//! or mis-shaped tensor is a loud [`vokra_core::VokraError::ModelLoad`]
//! (FR-EX-08). Real-weight parity binding is a follow-up wave gated on
//! the owner-side upstream checkpoint fetch + `docs/license-audit.md`
//! §3.1 sign-off (fail-closed).
//!
//! # No ONNX (permanent)
//!
//! RMVPE upstream is distributed as a torch `.pt` pickle; this
//! converter **never** touches ONNX (FR-LD-05). The `.pt` → safetensors
//! bridge lives in `tools/parity/nemo_pt_to_safetensors.py` (an offline
//! side-car tool, not part of the runtime).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for RMVPE GGUFs. Distinct from every sibling arch
/// tag because RMVPE is the first `category = "f0"` binder in the
/// converter tree — silently sharing an arch tag would misroute the
/// runtime dispatch (an ASR / TTS backbone would try to interpret the
/// 360-class pitch head).
pub const ARCH: &str = "rmvpe";

/// `vokra.model.name` value written for the canonical RMVPE GGUF.
pub const NAME: &str = "rmvpe";

/// `vokra.model.category` value — the first `"f0"` in the converter
/// tree. Consumed by the model-card generator + zoo manifest tier gate
/// so an F0 extractor is not accidentally advertised as an ASR / TTS
/// release.
pub const CATEGORY: &str = "f0";

/// `vokra.provenance.upstream_hf` value — the source repository the
/// weights come from. Recorded so a downstream consumer can re-fetch /
/// re-verify without a separate manifest lookup. RMVPE is distributed
/// via GitHub (no HF Hub mirror at time of writing); this is the
/// GitHub coordinate rather than an `<org>/<repo>` HF path.
pub const UPSTREAM_HF: &str = "yxlllc/RMVPE";

/// Canonical weight license SPDX (`mit`). Overrides via the
/// [`convert_rmvpe_file`] `license` parameter — the standing mechanism
/// for "implementation is clean-room MIT but the upstream distributed
/// checkpoint is another license" scenarios (mirror of
/// `convert_file_licensed` in `lib.rs`).
pub const DEFAULT_LICENSE: &str = "mit";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core`. Same key emotion2vec uses (they
/// share the same `vokra.model.category` chunk namespace).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

// GGUF metadata keys for the RMVPE hparam chunk group. Kept in sync
// with the runtime consumer `vokra_models::f0::rmvpe::{GGUF_KEY_HOP,
// GGUF_KEY_FMIN, GGUF_KEY_FMAX, GGUF_KEY_N_MELS, GGUF_KEY_N_FFT,
// GGUF_KEY_WIN_LENGTH, GGUF_KEY_SAMPLE_RATE, GGUF_KEY_N_CLASS,
// GGUF_KEY_CENTS_PER_CLASS, GGUF_KEY_BASE_HZ}`.
pub(crate) const KEY_HOP: &str = "vokra.rmvpe.hop";
pub(crate) const KEY_FMIN: &str = "vokra.rmvpe.fmin";
pub(crate) const KEY_FMAX: &str = "vokra.rmvpe.fmax";
pub(crate) const KEY_N_MELS: &str = "vokra.rmvpe.n_mels";
pub(crate) const KEY_N_FFT: &str = "vokra.rmvpe.n_fft";
pub(crate) const KEY_WIN_LENGTH: &str = "vokra.rmvpe.win_length";
pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.rmvpe.sample_rate";
pub(crate) const KEY_N_CLASS: &str = "vokra.rmvpe.n_class";
pub(crate) const KEY_CENTS_PER_CLASS: &str = "vokra.rmvpe.cents_per_class";
pub(crate) const KEY_BASE_HZ: &str = "vokra.rmvpe.base_hz";

// Canonical hparam values transcribed from the upstream RMVPE README
// (github.com/yxlllc/RMVPE — fetched 2026-07-30). Kept here as
// converter-side compile-time constants so a GGUF that never had a
// `vokra.rmvpe.*` chunk written (e.g. an emergency hand-crafted
// checkpoint) still round-trips through the runtime binder's default
// fallback.
pub const DEFAULT_HOP: u32 = 160;
pub const DEFAULT_FMIN: f32 = 30.0;
pub const DEFAULT_FMAX: f32 = 1000.0;
pub const DEFAULT_N_MELS: u32 = 128;
pub const DEFAULT_N_FFT: u32 = 2048;
pub const DEFAULT_WIN_LENGTH: u32 = 1024;
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const DEFAULT_N_CLASS: u32 = 360;
/// Upstream RMVPE pitch-class grid spacing (20 cents / class = 12
/// classes per semitone). The 360-class head therefore spans
/// `360 * 20 = 7200` cents ≈ 6 octaves starting at `base_hz`.
pub const DEFAULT_CENTS_PER_CLASS: f32 = 20.0;
/// Log-Hz grid anchor: class 0 corresponds to this Hz (10 * 2^(class *
/// cents/1200)). Upstream RMVPE anchors class 0 at ~32.703 Hz (C1)
/// which yields ~1975 Hz at class 360 — well above the fmax cutoff so
/// the head simply saturates unused classes at the upper tail.
pub const DEFAULT_BASE_HZ: f32 = 32.703_197;

/// Outcome of an RMVPE conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `RmvpeReport::default()` and the caller remains
/// responsible for surfacing the "no float tensors" loud note (mirror
/// of the emotion2vec / qwen3_tts / vibevoice / voxcpm2 `Report`
/// pattern with a `read` counter that pins the total tensor budget the
/// safetensors reader surfaced).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RmvpeReport {
    /// Total tensors seen in the upstream safetensors header (the sum
    /// of `written + skipped_non_float`). Pins the budget so a
    /// truncated header cannot silently drop tensors without the caller
    /// noticing.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 →
    /// f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 =
    /// top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an RMVPE GGUF
/// to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` + `vokra.rmvpe.*`
/// chunk groups pin the upstream repo, weight license, model category
/// and RMVPE hparams so the runtime binder can bring the graph up
/// without a side-car config lookup.
///
/// `license` overrides [`DEFAULT_LICENSE`] (`"mit"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed checkpoint
/// carries a different SPDX (e.g. `cc-by-4.0`).
pub fn convert_rmvpe_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<RmvpeReport, ConvertError> {
    // Whole-file read: an RMVPE checkpoint is ~40 MB — no need for the
    // streaming path the Moshi / Voxtral GB-scale converters run.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // RMVPE hparam chunk group — every value is a primary-source
    // constant transcribed from the upstream RMVPE README (fetched
    // 2026-07-30, CLAUDE.md「ハルシネーション厳禁」). The runtime
    // binder's `from_gguf` falls back to the same constants when a key
    // is absent, so a checkpoint that never carried a `vokra.rmvpe.*`
    // chunk still loads.
    b.add_u32(KEY_HOP, DEFAULT_HOP);
    b.add_f32(KEY_FMIN, DEFAULT_FMIN);
    b.add_f32(KEY_FMAX, DEFAULT_FMAX);
    b.add_u32(KEY_N_MELS, DEFAULT_N_MELS);
    b.add_u32(KEY_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH);
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_N_CLASS, DEFAULT_N_CLASS);
    b.add_f32(KEY_CENTS_PER_CLASS, DEFAULT_CENTS_PER_CLASS);
    b.add_f32(KEY_BASE_HZ, DEFAULT_BASE_HZ);

    // Self-describing redistribution: the artifact carries its own
    // licence. RMVPE ships MIT end-to-end (upstream `Dream-High/RMVPE`
    // + `yxlllc/RMVPE` LICENSE, fetched 2026-07-30). The `license`
    // override lets a downstream repackager stamp a different SPDX if
    // they redistribute under stricter terms.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = RmvpeReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `emotion2vec::convert_emotion2vec_file` / `qwen3_tts::convert`.
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
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir
    /// (emotion2vec pattern — no external `tempfile` dep, preserving
    /// zero-dep NFR-DS-02). The nanosecond suffix separates the tests
    /// in this module so a parallel `cargo test` cannot clobber files
    /// across them.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-rmvpe-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor
    /// (mirror of `emotion2vec::tests::synthetic_bf16_safetensors`) so
    /// a byte-identity assert catches any silent widen / downcast
    /// attempt — the raw zeroed payload would round-trip trivially
    /// through F32 / F16 widen and defeat the pin.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"unet.encoder.layer0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// STEP 1 RED (BF16 pass-through): the upstream BF16 checkpoint
    /// must survive the file-based converter round-trip with its dtype
    /// preserved (GGUF type 30 = `GgmlType::BF16`) and its payload
    /// byte-identical to the input. Mirror of emotion2vec /
    /// qwen3_tts / vibevoice / voxcpm2 / moshi / voxtral.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_rmvpe_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of emotion2vec)"
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
            .tensor_info("unet.encoder.layer0.weight")
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
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins the first `f0` model in the converter tree"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        // Hparam chunk pins the primary-source RMVPE constants.
        assert_eq!(
            file.get(KEY_HOP).and_then(|v| v.as_u64()),
            Some(DEFAULT_HOP as u64)
        );
        assert_eq!(
            file.get(KEY_N_MELS).and_then(|v| v.as_u64()),
            Some(DEFAULT_N_MELS as u64)
        );
        assert_eq!(
            file.get(KEY_N_CLASS).and_then(|v| v.as_u64()),
            Some(DEFAULT_N_CLASS as u64)
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(DEFAULT_SAMPLE_RATE as u64)
        );
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

    /// Empty-input pin: a safetensors buffer with no tensors must round-
    /// trip through the converter cleanly (report all-zero counters,
    /// GGUF header + metadata still stamped). Guards against a naive
    /// "at least one tensor required" gate that would refuse a valid
    /// empty-checkpoint round-trip.
    #[test]
    fn empty_input_produces_metadata_only_gguf() {
        // Minimal safetensors: 8-byte header size (=2), empty JSON `{}`,
        // and no tensor data.
        let header = b"{}";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header);

        let input = scratch_path("empty-in");
        let output = scratch_path("empty-out");
        std::fs::write(&input, &buf).expect("write empty safetensors");

        let report = convert_rmvpe_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
        );
        assert_eq!(
            file.get(KEY_N_CLASS).and_then(|v| v.as_u64()),
            Some(DEFAULT_N_CLASS as u64),
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override: a caller who redistributes under a different
    /// SPDX (e.g. via `vokra-cli convert --license`) must land the
    /// override on the artifact's `vokra.provenance.license` chunk.
    #[test]
    fn license_override_lands_on_provenance_chunk() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        convert_rmvpe_file(&input, &output, Some("apache-2.0")).expect("convert");
        let file = GgufFile::parse(std::fs::read(&output).expect("read out")).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
