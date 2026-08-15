#![allow(clippy::doc_lazy_continuation)]
//! **M2D** (`nttcslab/m2d`, **license unknown**): safetensors → GGUF
//! conversion (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `nttcslab/m2d` release — M2D
//! ("Masked Modeling Duo") is a self-supervised audio encoder
//! from NTT Communication Science Laboratories that jointly
//! predicts masked patches from a **target** online branch AND
//! its **predictive representation** via a dual-branch objective
//! (Niizumi et al. 2023, ICASSP arXiv:2210.14648, "Masked Modeling
//! Duo: Learning Representations by Encouraging Both Networks to
//! Model the Input"; TASLP 2024 extension for sound event detection
//! and speech). Positioned as an efficient audio-embedding backbone
//! for downstream sound-event detection / audio-tagging / speaker
//! tasks. ~86M parameter class base variant (~200 MB).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `beats` (iterative-tokenizer SSL), `eat`
//! (utterance-level Transformer + inverse block masking),
//! `atst` (teacher-student patchout), `dasheng` (universal MAE).
//! Distinct arch tag `m2d` because the masked-modeling-**duo**
//! (dual online + target branch, joint prediction of masked
//! patches AND their online-branch representation) topology is a
//! distinct axis from every sibling SSL encoder (single-branch
//! MAE = Dasheng / EAT, teacher-student patchout = ATST,
//! iterative tokenizer = BEATs). Silently sharing would misroute
//! the runtime dispatch and try to bind e.g. a single-branch MAE
//! decoder over a dual-branch checkpoint (FR-EX-08). Category
//! `audio-embedding`.
//!
//! # License posture — **Unknown** (fail-closed)
//!
//! Upstream `github.com/nttcslab/m2d` LICENSE is a **PDF file**
//! (`LICENSE.pdf`) that GitHub's classifier cannot machine-read —
//! GitHub API `/repos/nttcslab/m2d/license` returns
//! `spdx_id: NOASSERTION` with body decoding to:
//!
//! > "Please find the LICENSE at
//! > `https://github.com/nttcslab/m2d/blob/master/LICENSE.pdf`"
//!
//! (verified via GitHub API primary source task input 2026-08-13).
//! **No HuggingFace mirror exists as of 2026-08-13** (search of
//! `nttcslab/m2d` and `m2d` audio-tagged returned no matches).
//! Provenance stamp defaults to [`LicenseClass::Unknown`]
//! (fail-closed under M2-13). Owner must:
//!
//! 1. Download `LICENSE.pdf` and read it,
//! 2. Complete primary-source confirmation on the SPDX tier,
//! 3. Override via `--license <spdx>` at the outer boundary
//!    (`convert_m2d_file`'s `license` parameter → the CLI
//!    `--license` flag propagates through `convert_file_licensed`).
//!
//! §3.1 sign-off stays blank fail-closed until this owner ADR
//! completes (memory `[[feedback-license-signoff-primary-source]]`
//! — no CC pre-fill).
//!
//! # Scale — local convert OK (~0.2 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! M2D ships as PyTorch `.pth` pickle from the upstream release
//! (linked from README, hosted externally); this converter
//! **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02).
//! Callers pre-flatten via a future
//! `tools/parity/m2d_prepare_checkpoint.py` uv-managed Python
//! 3.12 sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) mirroring the DAC / Kokoro /
//! UTMOSv2 bridge pattern.
//!
//! # Topology axes — `vokra.m2d.*` (primary-source transcribed)
//!
//! The runtime binder `crates/vokra-models/src/m2d/mod.rs` declares an
//! eight-key `vokra.m2d.*` axis group and refuses its encoder forward
//! while any key is unstamped (`M2dConfig::validate_for_forward`).
//! This converter stamps **all eight**, under exactly the binder's key
//! spellings. Every value is transcribed from a primary source that was
//! actually read — never a sibling SSL encoder's ViT-Base numbers
//! borrowed across a different release (CLAUDE.md「ハルシネーション厳禁」):
//!
//! - `hidden_size` = 768, `num_hidden_layers` = 12,
//!   `num_attention_heads` = 12 — upstream `examples/portable_m2d.py`
//!   `get_backbone()` constructs the encoder as `LocalViT(in_chans=1,
//!   ..., embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
//!   norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))`. These three
//!   are hard-coded there (not parsed out of the weight name), so they
//!   hold for every `m2d_vit_base` weight the upstream wrapper loads.
//!   Corroborated by paper §4.1: "We used vanilla ViT-Base with a 768-d
//!   output feature as our encoders (f_θ and f_ξ)".
//! - `patch_height` = 16, `patch_width` = 16 — `portable_m2d.py`
//!   `Config.patch_size = [16, 16]` (ordered `[freq bins, time
//!   frames]`, matching `input_size = [80, 208]`); README pre-training
//!   command `--patch_size 16x16`; paper §4.1: "fixed the patch size to
//!   16×16 for all experiments".
//! - `n_mels` = 80, `sample_rate` = 16000 — `portable_m2d.py`
//!   `get_to_melspec()` 16 kHz arm sets `cfg.sample_rate, cfg.n_fft,
//!   cfg.window_size, cfg.hop_size = 16000, 400, 400, 160` and
//!   `cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 8000`; paper §4.1: "We
//!   preprocessed audio samples to a log-scaled mel spectrogram with a
//!   sampling frequency of 16,000 Hz ... and mel-spaced frequency bins
//!   F=80 in the range of 50 to 8,000 Hz".
//! - `inference_branch` = `"online"` — paper §3 defines the duo as "the
//!   online encoder f_θ" versus "The target network ... consists only
//!   of momentum encoder f_ξ", then states outright: "After the
//!   training, we transfer only the f_θ as a pre-trained model."
//!   `util/to_encoder_only_weight.py` corroborates operationally: it
//!   saves `PortableM2D(src).backbone.state_dict()`, i.e. the single
//!   encoder `get_backbone()` bound, discarding the rest.
//!
//! Primary sources read for the above: `github.com/nttcslab/m2d`
//! (`examples/portable_m2d.py`, `util/to_encoder_only_weight.py`,
//! `README.md`, tree at default branch `master`) and the ICASSP 2023
//! paper `arxiv.org/abs/2210.14648` §3 + §4.1.
//!
//! ## Axes deliberately NOT stamped
//!
//! The binder reads exactly the eight keys above and silently ignores
//! any other `vokra.m2d.*` key, so stamping extras would add artifact
//! surface with no consumer. Primary source does supply more (`mlp_ratio`
//! = 4, LayerNorm eps = 1e-6, `f_min` = 50, `f_max` = 8000, pre-training
//! `input_size` = 80×608 time frames while the portable `Config` default
//! is 80×208); those land only if and when the binder grows fields for
//! them, in the same commit.
//!
//! ## The 32 kHz release is a different identity
//!
//! `portable_m2d.py` also carries a 32 kHz arm (`sample_rate = 32000`,
//! `f_max = 16000`; `n_mels` stays 80), selected from the weight
//! directory name's third `p`-separated field (`...p32k`), which
//! `parse_sizes_by_name` defaults to `'16k'` when absent. [`SAMPLE_RATE`]
//! below is therefore the canonical 16 kHz [`NAME`] release **only**. A
//! 32 kHz weight is a separate release identity that must arrive as its
//! own `ModelKind` + `NAME` + stamp (the `snac_24khz` / `snac_44khz`
//! precedent), never converted through this arm: a wrong sample-rate
//! stamp resamples silently and has no loud failure mode downstream
//! (FR-EX-08).
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16
//! is emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime
//! widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for M2D GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`beats` / `eat` / `atst` / `dasheng` /
/// `mert` / `muq`) — M2D's dual-branch (online + target) masked-
/// modeling-duo training target is a distinct topology axis from
/// every sibling.
pub const ARCH: &str = "m2d";

/// `vokra.model.name` — canonical `m2d-base` size point. Sibling
/// variants (`m2d-eat` sound-event-detection specialization,
/// speech-specific fine-tunes etc.) are distinct release
/// identities published as their own future `NAME` following the
/// snac_24khz / snac_44khz pattern (added via separate future
/// ModelKind).
pub const NAME: &str = "m2d-base";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `beats` / `eat` / `atst`; downstream sound-event
/// detection / audio-tagging / speaker heads feed from the
/// encoder's hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. M2D is not hosted on HuggingFace, so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either. Sibling of `beats::UPSTREAM_URL` /
/// `eat::UPSTREAM_URL` / `atst::UPSTREAM_URL` /
/// `nsnet2::UPSTREAM_URL` posture.
pub const UPSTREAM_URL: &str = "github.com/nttcslab/m2d";

/// Default SPDX. Upstream `nttcslab/m2d` LICENSE is a **PDF file**
/// (`LICENSE.pdf`); GitHub API `/repos/nttcslab/m2d/license`
/// returns `spdx_id: NOASSERTION` (task input 2026-08-13). The
/// classifier `from_license_str("unknown")` correctly resolves to
/// [`LicenseClass::Unknown`] (fail-closed under M2-13, runtime
/// gate refuses to load without a research flag). Owner must
/// download `LICENSE.pdf`, read it, and override via
/// `--license <spdx>` at the outer boundary once the SPDX tier
/// is confirmed.
pub const DEFAULT_LICENSE_SPDX: &str = "unknown";

// ---------------------------------------------------------------------------
// M2D topology axes — transcribed verbatim from upstream primary sources
// (see the module docstring section "Topology axes" for the per-value
// citation and the exact quoted sentences).
//
// Sources actually read, 2026-08-15:
// - github.com/nttcslab/m2d `examples/portable_m2d.py` (`Config`,
//   `get_backbone`, `get_to_melspec`, `parse_sizes_by_name`)
// - github.com/nttcslab/m2d `util/to_encoder_only_weight.py`
// - github.com/nttcslab/m2d `README.md` (pre-training commands)
// - arxiv.org/abs/2210.14648 (Niizumi et al., ICASSP 2023) §3 + §4.1
//
// Stamped so the runtime binder `crates/vokra-models/src/m2d/mod.rs`
// can shape the encoder without re-deriving topology from tensor
// shapes. The binder treats a present-but-unreadable key as a loud
// error, so a half-landed group here cannot masquerade as a silent
// converter (FR-EX-08).
// ---------------------------------------------------------------------------

/// Transformer hidden width (ViT-Base embed dim) — **768**.
///
/// `portable_m2d.py` `get_backbone()`: `LocalViT(..., embed_dim=768,
/// depth=12, num_heads=12, ...)`; paper §4.1 "vanilla ViT-Base with a
/// 768-d output feature as our encoders".
pub const HIDDEN_SIZE: u32 = 768;

/// Transformer encoder block count — **12**.
///
/// `portable_m2d.py` `get_backbone()`: `depth=12`.
pub const NUM_HIDDEN_LAYERS: u32 = 12;

/// Self-attention head count — **12**.
///
/// `portable_m2d.py` `get_backbone()`: `num_heads=12`.
pub const NUM_ATTENTION_HEADS: u32 = 12;

/// Spectrogram patch height in **mel bins** — **16**.
///
/// `portable_m2d.py` `Config.patch_size = [16, 16]`, ordered
/// `[freq bins, time frames]` to match `input_size = [80, 208]`; paper
/// §4.1 "fixed the patch size to 16×16 for all experiments".
pub const PATCH_HEIGHT: u32 = 16;

/// Spectrogram patch width in **time frames** — **16**.
///
/// Same primary sources as [`PATCH_HEIGHT`]; M2D's patch is square, so
/// the two axes coincide numerically but are kept distinct because the
/// binder's grid arithmetic reads them separately.
pub const PATCH_WIDTH: u32 = 16;

/// Mel-filterbank bin count of the log-mel front-end — **80**.
///
/// `portable_m2d.py` `get_to_melspec()`: `cfg.n_mels, cfg.f_min,
/// cfg.f_max = 80, 50, 8000`; paper §4.1 "mel-spaced frequency bins
/// F=80 in the range of 50 to 8,000 Hz". Note 80 holds on the 32 kHz
/// arm too — `sample_rate` is the axis that differs there.
pub const N_MELS: u32 = 80;

/// Input sample rate in Hz — **16000**, the canonical [`NAME`] release.
///
/// `portable_m2d.py` `get_to_melspec()` 16 kHz arm: `cfg.sample_rate,
/// cfg.n_fft, cfg.window_size, cfg.hop_size = 16000, 400, 400, 160`;
/// paper §4.1 "a sampling frequency of 16,000 Hz".
///
/// **A 32 kHz M2D weight must not be converted through this arm** — see
/// the module docstring section "The 32 kHz release is a different
/// identity". A wrong sample-rate stamp has no loud failure mode
/// downstream (FR-EX-08).
pub const SAMPLE_RATE: u32 = 16_000;

/// Which network of the Masked Modeling **Duo** the inference path
/// reads — **`"online"`**.
///
/// This is the one axis the binder singles out as unguessable: both
/// branches are shape-compatible, so a wrong pick returns a
/// plausible-but-wrong embedding with no loud failure mode. It is
/// stamped here only because a primary source states it outright.
/// Paper §3 defines "the online encoder f_θ" against "The target
/// network ... consists only of momentum encoder f_ξ", then concludes:
/// "After the training, we transfer only the f_θ as a pre-trained
/// model." Corroborated by `util/to_encoder_only_weight.py`, which
/// persists `PortableM2D(src).backbone.state_dict()` — the single
/// encoder `get_backbone()` bound — and discards the rest.
///
/// Wire form must match the binder's `M2dBranch::from_wire` exactly
/// (`"online"` / `"target"`, case-sensitive).
pub const INFERENCE_BRANCH: &str = "online";

const UPSTREAM_SOURCE: &str = "nttcslab/m2d (Masked Modeling Duo — dual-branch SSL audio encoder joint-predicting \
     masked patches AND online-branch representation, ~86M params base, Niizumi et al. \
     arXiv:2210.14648 ICASSP 2023 + TASLP 2024 extension, LICENSE.pdf non-machine-readable \
     — fail-closed unknown)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// Scalar topology chunks (`vokra.m2d.*`). These spellings are a mirror
// of the runtime binder's `GGUF_KEY_*` constants in
// `crates/vokra-models/src/m2d/mod.rs` — the two halves only meet if
// they match byte for byte, and `topology_axis_keys_mirror_the_runtime_binder`
// pins every one of them. Duplicated rather than shared because
// `vokra-models` must not gain a dependency edge onto `vokra-convert`
// (layering: `vokra-core` = GGUF reader, `vokra-convert` = GGUF writer).
const KEY_M2D_HIDDEN_SIZE: &str = "vokra.m2d.hidden_size";
const KEY_M2D_NUM_HIDDEN_LAYERS: &str = "vokra.m2d.num_hidden_layers";
const KEY_M2D_NUM_ATTENTION_HEADS: &str = "vokra.m2d.num_attention_heads";
const KEY_M2D_PATCH_HEIGHT: &str = "vokra.m2d.patch_height";
const KEY_M2D_PATCH_WIDTH: &str = "vokra.m2d.patch_width";
const KEY_M2D_N_MELS: &str = "vokra.m2d.n_mels";
const KEY_M2D_SAMPLE_RATE: &str = "vokra.m2d.sample_rate";
/// String-valued, unlike every sibling axis: the binder parses it
/// through `M2dBranch::from_wire` and rejects anything outside
/// `{"online", "target"}`.
const KEY_M2D_INFERENCE_BRANCH: &str = "vokra.m2d.inference_branch";

/// Outcome of an M2D conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`beats` / `eat` / `atst`
/// / `dasheng` / `mert` / `muq` / `yamnet`) — the invariant
/// `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct M2dReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16, so any tensor reaching
    /// this counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts an M2D safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`M2dReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"unknown"`,
/// `Unknown`) — fail-closed under M2-13, publish refused until a
/// caller supplies a real SPDX + §3.1 sign-off completes after owner
/// reads upstream `LICENSE.pdf`.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_m2d_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<M2dReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // M2D topology axes. Every value is transcribed from a primary
    // source that was read (upstream `examples/portable_m2d.py` +
    // arXiv:2210.14648 §3/§4.1 — see the constants' rustdoc for the
    // per-axis citation), never inferred from a sibling SSL encoder.
    // The whole group is stamped together: the binder distinguishes
    // "converter silent" from "half a group landed", and a partial
    // stamp would read as the latter.
    b.add_u32(KEY_M2D_HIDDEN_SIZE, HIDDEN_SIZE);
    b.add_u32(KEY_M2D_NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS);
    b.add_u32(KEY_M2D_NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS);
    b.add_u32(KEY_M2D_PATCH_HEIGHT, PATCH_HEIGHT);
    b.add_u32(KEY_M2D_PATCH_WIDTH, PATCH_WIDTH);
    b.add_u32(KEY_M2D_N_MELS, N_MELS);
    b.add_u32(KEY_M2D_SAMPLE_RATE, SAMPLE_RATE);
    // String-valued: the binder parses it via `M2dBranch::from_wire`.
    b.add_string(KEY_M2D_INFERENCE_BRANCH, INFERENCE_BRANCH);

    // Unknown fail-closed default: if the caller passes no license
    // string, the classifier resolves `"unknown"` to
    // `LicenseClass::Unknown` and the M2-13 runtime gate refuses to
    // load without a research flag. Any caller who has resolved the
    // license out-of-band (by downloading LICENSE.pdf and reading
    // it) supplies `--license <spdx>` at the outer boundary.
    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = M2dReport::default();
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
            "vokra-convert-m2d-{tag}-{}-{n}",
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

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_unknown_fail_closed() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // M2D uses an online + target duo — realistic upstream
        // state-dict name from the dual-branch objective.
        let st = safetensors_one("online.blocks.0.attn.qkv.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_m2d_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);

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
            LicenseClass::Unknown.as_str(),
            "unknown must resolve to Unknown (fail-closed under M2-13)"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -2.5, 0.15625, 3.5];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("target.blocks.0.attn.qkv.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_m2d_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("target.blocks.0.attn.qkv.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_apache_flips_to_permissive() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_m2d_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");

        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    // -----------------------------------------------------------------------
    // Topology axes — `vokra.m2d.*`
    //
    // The runtime binder (`crates/vokra-models/src/m2d/mod.rs`) refuses its
    // encoder forward while any of its eight `vokra.m2d.*` keys is unstamped.
    // These rows prove the two halves meet: identical key spellings, values
    // equal to the transcribed upstream constants, and a strictly ADDITIVE
    // change to the artifact (arch / provenance / tensor bytes untouched).
    // -----------------------------------------------------------------------

    #[test]
    fn topology_axis_keys_mirror_the_runtime_binder() {
        // Byte-for-byte pin against `vokra-models`' `GGUF_KEY_*` constants.
        // The crates cannot share these (no `vokra-models` → `vokra-convert`
        // dependency edge), so a rename on either side has to land here or
        // fail this row — otherwise a stamped axis would simply never be read.
        assert_eq!(KEY_M2D_HIDDEN_SIZE, "vokra.m2d.hidden_size");
        assert_eq!(KEY_M2D_NUM_HIDDEN_LAYERS, "vokra.m2d.num_hidden_layers");
        assert_eq!(KEY_M2D_NUM_ATTENTION_HEADS, "vokra.m2d.num_attention_heads");
        assert_eq!(KEY_M2D_PATCH_HEIGHT, "vokra.m2d.patch_height");
        assert_eq!(KEY_M2D_PATCH_WIDTH, "vokra.m2d.patch_width");
        assert_eq!(KEY_M2D_N_MELS, "vokra.m2d.n_mels");
        assert_eq!(KEY_M2D_SAMPLE_RATE, "vokra.m2d.sample_rate");
        assert_eq!(KEY_M2D_INFERENCE_BRANCH, "vokra.m2d.inference_branch");

        // Exactly eight, all distinct — the binder's `TOTAL_OPTIONAL_AXES`
        // is 8 and it reports "partially stamped" for anything short of the
        // full group, so a duplicated or dropped key would silently degrade
        // a real artifact to `PartiallyStamped`.
        let mut keys = vec![
            KEY_M2D_HIDDEN_SIZE,
            KEY_M2D_NUM_HIDDEN_LAYERS,
            KEY_M2D_NUM_ATTENTION_HEADS,
            KEY_M2D_PATCH_HEIGHT,
            KEY_M2D_PATCH_WIDTH,
            KEY_M2D_N_MELS,
            KEY_M2D_SAMPLE_RATE,
            KEY_M2D_INFERENCE_BRANCH,
        ];
        assert_eq!(keys.len(), 8, "the binder reads exactly 8 axes");
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 8, "axis keys must be pairwise distinct");
        for k in keys {
            assert!(
                k.starts_with("vokra.m2d."),
                "`{k}` must live under the arch's own metadata namespace"
            );
        }
    }

    #[test]
    fn transcribed_constants_match_their_primary_sources() {
        // Each literal below is quoted in the constant's rustdoc against the
        // exact upstream line or paper sentence it came from. Pinning them
        // here means a "tidy-up" edit to a constant fails a test instead of
        // silently shipping a fabricated axis.
        //
        // `examples/portable_m2d.py` `get_backbone()`:
        //   LocalViT(..., embed_dim=768, depth=12, num_heads=12, ...)
        assert_eq!(HIDDEN_SIZE, 768);
        assert_eq!(NUM_HIDDEN_LAYERS, 12);
        assert_eq!(NUM_ATTENTION_HEADS, 12);
        // `portable_m2d.py` `Config.patch_size = [16, 16]` ([freq, time]);
        // paper §4.1 "fixed the patch size to 16×16 for all experiments".
        assert_eq!(PATCH_HEIGHT, 16);
        assert_eq!(PATCH_WIDTH, 16);
        // `portable_m2d.py` `get_to_melspec()` 16k arm:
        //   sample_rate, n_fft, window_size, hop_size = 16000, 400, 400, 160
        //   n_mels, f_min, f_max = 80, 50, 8000
        assert_eq!(N_MELS, 80);
        assert_eq!(SAMPLE_RATE, 16_000);
        // Paper §3: "After the training, we transfer only the f_θ as a
        // pre-trained model", where f_θ is "the online encoder". The
        // binder's `M2dBranch::from_wire` is case-sensitive and accepts only
        // `online` / `target`, so the casing here is load-bearing: `"Online"`
        // would be a loud ModelLoad on the read side.
        assert_eq!(INFERENCE_BRANCH, "online");
    }

    #[test]
    fn topology_axes_round_trip_and_stay_purely_additive() {
        let inp = tmp_path("topo-in");
        let outp = tmp_path("topo-out");
        let payload: Vec<u8> = [1.5_f32, -0.25, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st = safetensors_one("online.blocks.0.mlp.fc1.weight", "F32", &[1, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_m2d_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_u64 = |k: &str| -> u64 {
            g.get(k)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("{k}: missing or not an unsigned integer"))
        };

        // Every scalar axis round-trips as the transcribed constant. The
        // literal on the right restates the upstream value, so this row
        // fails if either the stamp or the constant drifts.
        assert_eq!(read_u64(KEY_M2D_HIDDEN_SIZE), u64::from(HIDDEN_SIZE));
        assert_eq!(read_u64(KEY_M2D_HIDDEN_SIZE), 768);
        assert_eq!(
            read_u64(KEY_M2D_NUM_HIDDEN_LAYERS),
            u64::from(NUM_HIDDEN_LAYERS)
        );
        assert_eq!(read_u64(KEY_M2D_NUM_HIDDEN_LAYERS), 12);
        assert_eq!(
            read_u64(KEY_M2D_NUM_ATTENTION_HEADS),
            u64::from(NUM_ATTENTION_HEADS)
        );
        assert_eq!(read_u64(KEY_M2D_NUM_ATTENTION_HEADS), 12);
        assert_eq!(read_u64(KEY_M2D_PATCH_HEIGHT), u64::from(PATCH_HEIGHT));
        assert_eq!(read_u64(KEY_M2D_PATCH_HEIGHT), 16);
        assert_eq!(read_u64(KEY_M2D_PATCH_WIDTH), u64::from(PATCH_WIDTH));
        assert_eq!(read_u64(KEY_M2D_PATCH_WIDTH), 16);
        assert_eq!(read_u64(KEY_M2D_N_MELS), u64::from(N_MELS));
        assert_eq!(read_u64(KEY_M2D_N_MELS), 80);
        assert_eq!(read_u64(KEY_M2D_SAMPLE_RATE), u64::from(SAMPLE_RATE));
        assert_eq!(
            read_u64(KEY_M2D_SAMPLE_RATE),
            16_000,
            "canonical 16 kHz release — a 32 kHz weight is a separate identity"
        );

        // The branch selector is a STRING, not a u32: the binder reads it
        // with `as_str()` and fails loud on any other GGUF value type.
        assert_eq!(
            g.get(KEY_M2D_INFERENCE_BRANCH).and_then(|v| v.as_str()),
            Some(INFERENCE_BRANCH),
        );
        assert_eq!(
            g.get(KEY_M2D_INFERENCE_BRANCH).and_then(|v| v.as_str()),
            Some("online"),
            "paper §3: after training only f_θ (the online encoder) is transferred"
        );
        assert_eq!(
            g.get(KEY_M2D_INFERENCE_BRANCH).and_then(|v| v.as_u64()),
            None,
            "branch must NOT be stamped as an integer — the binder parses it \
             with `M2dBranch::from_wire`"
        );

        // ---- purely additive: nothing that was already stamped moved ----
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
            DEFAULT_LICENSE_SPDX,
            "stamping topology must not disturb the fail-closed license default"
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Unknown.as_str(),
            "M2D stays Unknown (LICENSE.pdf unreadable) — topology is not a \
             license signal"
        );

        // …and the weights are still a verbatim pass-through.
        let info = g
            .tensor_info("online.blocks.0.mlp.fc1.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![1, 3]);
        assert_eq!(
            g.tensor_bytes(info),
            payload.as_slice(),
            "topology stamps must not perturb tensor payloads"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
