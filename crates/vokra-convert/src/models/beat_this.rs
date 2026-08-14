#![allow(clippy::doc_lazy_continuation)]
//! **beat_this** (`CPJKU/beat_this`, mit): safetensors → GGUF conversion
//! (music-understanding wave, 2026-08-14, post-Wave-1 CC-gap).
//!
//! Input: the upstream `CPJKU/beat_this` release — Foscarin et al.
//! ISMIR 2024 arXiv:2407.21658 "Beat this! Accurate beat tracking without
//! DBN postprocessing". A **Transformer-based beat and downbeat tracker**
//! that dispenses with the traditional Dynamic Bayesian Network post-
//! processing step every prior beat tracker relied on — the model itself
//! emits per-frame beat + downbeat activation posteriorgrams the caller
//! peak-picks directly, so the entire tracker is a single forward pass
//! (log-mel front-end → stacked Transformer blocks → 2-head classifier
//! producing beat + downbeat activations). Weights ship as a PyTorch
//! `.pt` checkpoint (~20 MB) from the upstream `github.com/CPJKU/beat_this`
//! release page — CC Music Processing (Vienna University of Music and
//! Performing Arts).
//!
//! # Vokra scope — music understanding (2026-07-30 scope expansion)
//!
//! Complements the sibling music-understanding family (`basic_pitch` =
//! polyphonic pitch/MIDI transcription, `crepe` / `fcpe` / `rmvpe` =
//! monophonic F0 extractors, `beats` = Microsoft SSL audio encoder,
//! `mert` / `muq` / `dasheng` = music-embedding SSL backbones) with a
//! **beat + downbeat tracker** — a distinct axis of music understanding
//! (rhythm / tempo / metrical structure) none of the sibling models
//! target. Distinct arch tag `beat-this` because the output surface (per-
//! frame beat + downbeat activation posteriorgrams, no DBN post-processing)
//! and the input topology (log-mel Transformer) are both distinct from
//! (a) the sibling monophonic F0 grid (single-head 360-class or 384-class),
//! (b) the sibling polyphonic 3-head posteriorgram of `basic_pitch`, and
//! (c) the audio-embedding backbones which have no beat / downbeat head at
//! all. New category `beat-tracking` — distinct from `pitch-transcription`
//! (owned by `basic_pitch`), from `f0` (owned by `crepe` / `fcpe` /
//! `rmvpe`), and from `audio-embedding` / `music-embedding` (owned by the
//! SSL family) because the output is a **metrical-structure stream** (beat
//! + downbeat frame indices) rather than a pitch grid or a general audio
//! feature.
//!
//! # Distinct arch tag rationale — silently sharing would misroute FR-EX-08
//!
//! The sibling `beats` (`microsoft/unilm/tree/master/beats`, ARCH tag
//! `beats`) is Microsoft's foundational **self-supervised audio encoder**
//! trained with iterative acoustic tokenizer + mask acoustic modeling
//! (Chen et al. 2023 ICML arXiv:2212.09058); it emits general audio
//! embeddings, not beat / downbeat activations. Sharing arch tag `beats`
//! between "Microsoft SSL audio encoder" and "CPJKU Transformer beat
//! tracker" would let runtime dispatch bind e.g. a beat-tracking classifier
//! head over an SSL-encoder checkpoint (or vice versa), a silent-wrong
//! shape mismatch that FR-EX-08 forbids. The hyphenated tag `beat-this`
//! keeps the two families distinct in the string-namespace runtime
//! dispatch walks.
//!
//! # License posture — mit (**Permissive**)
//!
//! Upstream `CPJKU/beat_this` LICENSE = MIT (CC Music Processing standard
//! for their public releases). §3.1 sign-off stays blank fail-closed until
//! owner completes primary-source confirmation (memory
//! `[[feedback-license-signoff-primary-source]]` — no CC pre-fill; the
//! judgement here is docs-only, not a Vokra distribution decision).
//!
//! # Scale — local convert OK (~0.02 GB / ~20 MB `.pt`)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! `[[feedback-large-models-on-vast-ai]]`: <2 GB safe = ~1/100 of the
//! threshold). No vast.ai handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! beat_this ships as PyTorch `.pt` pickle from the upstream release page;
//! this converter **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02).
//! Callers pre-flatten `beat_this.pt` → `.safetensors` offline via a future
//! `tools/parity/beat_this_prepare_checkpoint.py` uv-managed Python 3.12
//! sidecar (memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`)
//! mirroring the DAC / Kokoro / UTMOSv2 / beats bridge pattern.
//!
//! # `vokra.beat_this.*` chunk group (written here)
//!
//! Read by `vokra-models::beat_this::BeatThis::from_gguf`:
//!
//! - `vokra.beat_this.sample_rate` (`u32`): input PCM sample rate the
//!   log-mel front-end was tuned for.
//! - `vokra.beat_this.n_frames` (`u32`): number of log-mel frames per
//!   analysis window (the Transformer's temporal input length).
//! - `vokra.beat_this.d_model` (`u32`): Transformer hidden dimension.
//! - `vokra.beat_this.n_layers` (`u32`): stacked Transformer encoder
//!   layer count.
//! - `vokra.beat_this.n_head` (`u32`): multi-head attention head count.
//! - `vokra.beat_this.n_classes` (`u32`): output class count of the
//!   terminal 2-head classifier (canonical `2` = beat + downbeat, though
//!   held as a u32 axis to future-proof for a downstream variant that
//!   adds a downbeat sub-class).
//!
//! Every axis is **caller-supplied** on the converter side via
//! [`BeatThisHparams`] — the upstream checkpoint carries the axes implicitly
//! in tensor shapes rather than a first-class `config.yaml`, so
//! primary-source verification of the true axes is a step the caller
//! (owner + `tools/parity/beat_this_prepare_checkpoint.py`) performs
//! before invoking this converter. No hard-coded default constants are
//! fabricated here (CLAUDE.md 「ハルシネーション厳禁」).
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16 is
//! emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for beat_this GGUFs.
///
/// Hyphenated to stay distinct from the sibling `beats` arch (Microsoft
/// SSL audio encoder), so string-namespace runtime dispatch cannot bind
/// a beat-tracker classifier head over an SSL-encoder checkpoint — that
/// silent-wrong shape mismatch is what FR-EX-08 forbids.
pub const ARCH: &str = "beat-this";

/// `vokra.model.name` — canonical `CPJKU/beat_this` release.
pub const NAME: &str = "beat-this";

/// `vokra.model.category` — beat-tracking, distinct from `pitch-transcription`
/// (owned by `basic_pitch`), `f0` (owned by `crepe` / `fcpe` / `rmvpe`), and
/// `audio-embedding` / `music-embedding` (owned by the SSL backbones).
pub const CATEGORY: &str = "beat-tracking";

/// `vokra.provenance.upstream_url` value — the GitHub tree the release
/// ships from. beat_this is not hosted on HuggingFace as a first-party
/// CPJKU release as of 2026-08-14, so we record `upstream_url` rather than
/// `upstream_hf` (sibling posture: `beats`, `nsnet2`, `emotion2vec`).
pub const UPSTREAM_URL: &str = "github.com/CPJKU/beat_this";

/// Default SPDX. Upstream `CPJKU/beat_this` LICENSE = MIT (CC Music
/// Processing standard for their public releases). A caller with a
/// different attestation may override at the outer boundary
/// (`--license <spdx>`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str = "CPJKU/beat_this (Transformer-based beat + downbeat tracker without DBN \
     postprocessing, ~20 MB PyTorch .pt, log-mel Transformer emitting per-frame beat + downbeat \
     activation posteriorgrams, Foscarin et al. ISMIR 2024 arXiv:2407.21658, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---------------------------------------------------------------------------
// `vokra.beat_this.*` metadata keys — mirrored on the runtime side at
// `crates/vokra-models/src/beat_this/mod.rs::GGUF_KEY_*`. Two copies of
// each string constant is deliberate (same pattern as `pyannote` / `snac` /
// `hifigan` — see the runtime binder's module doc). Cross-crate sharing
// would add a `vokra-convert` dep on `vokra-models` (or vice-versa) that
// reverses the layer stack.
// ---------------------------------------------------------------------------

/// `vokra.beat_this.sample_rate` — input PCM sample rate the log-mel
/// front-end was tuned for.
pub const KEY_SAMPLE_RATE: &str = "vokra.beat_this.sample_rate";
/// `vokra.beat_this.n_frames` — number of log-mel frames per analysis
/// window (the Transformer's temporal input length).
pub const KEY_N_FRAMES: &str = "vokra.beat_this.n_frames";
/// `vokra.beat_this.d_model` — Transformer hidden dimension.
pub const KEY_D_MODEL: &str = "vokra.beat_this.d_model";
/// `vokra.beat_this.n_layers` — stacked Transformer encoder layer count.
pub const KEY_N_LAYERS: &str = "vokra.beat_this.n_layers";
/// `vokra.beat_this.n_head` — multi-head attention head count.
pub const KEY_N_HEAD: &str = "vokra.beat_this.n_head";
/// `vokra.beat_this.n_classes` — terminal classifier output class count.
pub const KEY_N_CLASSES: &str = "vokra.beat_this.n_classes";

// ---------------------------------------------------------------------------
// BeatThisHparams — caller-supplied axes for the `vokra.beat_this.*` chunk
// group. The upstream `.pt` release does not ship a first-class
// `config.yaml`, so primary-source verification of these axes is a
// caller-side step (owner + `tools/parity/beat_this_prepare_checkpoint.py`).
// No hard-coded defaults are fabricated here (CLAUDE.md 「ハルシネーション
// 厳禁」).
// ---------------------------------------------------------------------------

/// Caller-supplied beat_this hyperparameters for the `vokra.beat_this.*`
/// chunk group. Every axis is `u32` in the emitted GGUF.
///
/// Callers source these from the upstream config transcription step
/// (`tools/parity/beat_this_prepare_checkpoint.py`, uv-managed Python 3.12
/// sidecar). This converter does **not** fabricate any of them — the
/// upstream `.pt` release stores axes implicitly in tensor shapes rather
/// than a first-class `config.yaml`, so a primary-source-verified caller
/// value is the honest input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeatThisHparams {
    /// Input PCM sample rate the log-mel front-end was tuned for.
    pub sample_rate: u32,
    /// Number of log-mel frames per analysis window (the Transformer's
    /// temporal input length).
    pub n_frames: u32,
    /// Transformer hidden dimension.
    pub d_model: u32,
    /// Stacked Transformer encoder layer count.
    pub n_layers: u32,
    /// Multi-head attention head count.
    pub n_head: u32,
    /// Terminal classifier output class count (canonical `2` = beat +
    /// downbeat, held as a `u32` axis to future-proof for downstream
    /// variants).
    pub n_classes: u32,
}

// ---------------------------------------------------------------------------
// BeatThisReport — outcome-counter mirror of the sibling BF16 pass-through
// converters (`beats` / `basic_pitch` / `dasheng` / ...). Invariant
// `read == written + skipped_non_float` is auditable at the report level.
// ---------------------------------------------------------------------------

/// Outcome of a beat_this conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BeatThisReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16, so any tensor reaching this
    /// counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16 → f32
    /// losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a `CPJKU/beat_this` safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning a [`BeatThisReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` /
/// `vokra.beat_this.*` chunks are stamped for the runtime compliance gate
/// (FR-CP-03) and the runtime binder (`vokra-models::beat_this`).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is [`DEFAULT_LICENSE_SPDX`] (`"mit"`,
/// `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_beat_this_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    hparams: BeatThisHparams,
) -> Result<BeatThisReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // `vokra.beat_this.*` chunk group — caller-supplied axes (see the
    // `BeatThisHparams` docstring for the primary-source rationale).
    b.add_u32(KEY_SAMPLE_RATE, hparams.sample_rate);
    b.add_u32(KEY_N_FRAMES, hparams.n_frames);
    b.add_u32(KEY_D_MODEL, hparams.d_model);
    b.add_u32(KEY_N_LAYERS, hparams.n_layers);
    b.add_u32(KEY_N_HEAD, hparams.n_head);
    b.add_u32(KEY_N_CLASSES, hparams.n_classes);

    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = BeatThisReport::default();
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

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-beat-this-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn sample_hparams() -> BeatThisHparams {
        // These values are illustrative fixtures — they are structurally
        // consistent (n_head divides d_model, n_classes >= 2 for beat +
        // downbeat) but they are NOT transcribed from an upstream config.
        // The converter has no responsibility to primary-source them —
        // the caller (owner + `tools/parity/beat_this_prepare_checkpoint.py`)
        // sources them from the upstream `.pt` tensor-shape walk.
        BeatThisHparams {
            sample_rate: 22_050,
            n_frames: 128,
            d_model: 128,
            n_layers: 6,
            n_head: 8,
            n_classes: 2,
        }
    }

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // A representative beat_this tensor name from a Transformer encoder
        // block (a stacked `nn.TransformerEncoderLayer`'s attention proj).
        let st = safetensors_one(
            "encoder.layers.0.self_attn.q_proj.weight",
            "F32",
            &[1, 3],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let hp = sample_hparams();
        let r = convert_beat_this_file(&inp, &outp, None, hp).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.skipped_non_float, 0);
        assert_eq!(r.bf16_passthrough, 0);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |k: &str| -> String {
            g.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{k}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_URL), UPSTREAM_URL);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Permissive.as_str(),
            "mit must resolve to Permissive"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("encoder.layers.0.linear1.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let hp = sample_hparams();
        let r = convert_beat_this_file(&inp, &outp, None, hp).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.layers.0.linear1.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        // A caller with a stricter attestation may downgrade off the
        // default MIT (e.g. a per-checkpoint NC posture surfacing during
        // primary-source verification).
        convert_beat_this_file(&inp, &outp, Some("cc-by-nc-4.0"), sample_hparams())
            .expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-nc-4.0"),
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn arch_category_and_upstream_url_stamp_are_distinct_and_stable() {
        // Pin the string constants so a rename would land here in the
        // same commit or fail this test. The sibling `beats` arch
        // (Microsoft SSL audio encoder) MUST NOT collide with our
        // `beat-this` arch — silently sharing would misroute runtime
        // dispatch (FR-EX-08).
        assert_eq!(ARCH, "beat-this");
        assert_ne!(
            ARCH, "beats",
            "must not collide with sibling Microsoft BEATs arch"
        );
        assert_eq!(NAME, "beat-this");
        assert_eq!(CATEGORY, "beat-tracking");
        assert_ne!(
            CATEGORY, "pitch-transcription",
            "must not collide with sibling basic_pitch category"
        );
        assert_ne!(
            CATEGORY, "f0",
            "must not collide with sibling crepe/fcpe/rmvpe category"
        );
        assert_ne!(
            CATEGORY, "audio-embedding",
            "must not collide with SSL audio-encoder family category"
        );
        assert_eq!(UPSTREAM_URL, "github.com/CPJKU/beat_this");
        assert_eq!(DEFAULT_LICENSE_SPDX, "mit");
    }

    #[test]
    fn hparams_chunk_group_round_trip() {
        let inp = tmp_path("hp-in");
        let outp = tmp_path("hp-out");
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("y", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        // Non-default hparams values so a silent zero-init would break the
        // round-trip.
        let hp = BeatThisHparams {
            sample_rate: 44_100,
            n_frames: 256,
            d_model: 512,
            n_layers: 12,
            n_head: 16,
            n_classes: 3,
        };
        convert_beat_this_file(&inp, &outp, None, hp).expect("convert with custom hparams");
        let g = GgufFile::open(&outp).unwrap();
        let read_u32 = |k: &str| -> u32 {
            g.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or_else(|| panic!("{k}: missing or non-u64"))
        };
        assert_eq!(read_u32(KEY_SAMPLE_RATE), 44_100);
        assert_eq!(read_u32(KEY_N_FRAMES), 256);
        assert_eq!(read_u32(KEY_D_MODEL), 512);
        assert_eq!(read_u32(KEY_N_LAYERS), 12);
        assert_eq!(read_u32(KEY_N_HEAD), 16);
        assert_eq!(read_u32(KEY_N_CLASSES), 3);

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
