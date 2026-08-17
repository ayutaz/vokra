#![allow(clippy::doc_lazy_continuation)]
//! **StoRM** (`sp-uhh/storm`, **MIT**) — Stochastic Regeneration Model
//! for Speech Enhancement and Dereverberation: safetensors → GGUF
//! conversion (Wave 7 2026-08-14 audit follow-up RETRY of a Wave 6 lost
//! item — workflow silently swallowed the previous result; see WAVE 6
//! LESSON in the directive).
//!
//! # Model class — diffusion score-based two-stage speech enhancement
//!
//! StoRM (Lay et al. 2023, arXiv:2312.09386
//! *"StoRM: A Diffusion-based Stochastic Regeneration Model for Speech
//! Enhancement and Dereverberation"*) — a two-stage STFT-domain speech
//! enhancement and dereverberation model that combines a deterministic
//! predictive first stage with a diffusion score-model refinement
//! second stage:
//!
//! - an **initial deterministic predictive estimator** (an NCSN++ v2
//!   U-Net variant per arXiv:2312.09386 §III trained under an MSE
//!   objective rather than the score-matching objective — same
//!   topology family as the score network but a *distinct forward*
//!   from the second stage);
//! - an **NCSN++ v2 U-Net score-network** (Noise Conditional Score
//!   Network++ v2 backbone — U-Net with attention blocks and
//!   feature-wise linear modulation (FiLM) over noise-conditioning σ,
//!   per Song et al. arXiv:2011.13456 §3.3 as extended in StoRM §III)
//!   applied iteratively over a schedule of noise levels;
//! - an **OUVE-SDE (Ornstein-Uhlenbeck Variance-Exploding stochastic
//!   differential equation) predictor-corrector sampler** (drift-
//!   diffusion iterative refinement per arXiv:2312.09386 §III + Welker
//!   et al. 2022 SGMSE+ Interspeech precedent) that composes the two
//!   stages into a single conditional generative process.
//!
//! Its role in the Vokra catalogue is a *first diffusion-based entry
//! on the enhancement arm* — a topologically distinct sibling of the
//! any-to-any mask-based enhancement family (`denoise` (DFN3),
//! `nsnet2`, `rnnoise`, `gtcrn`). StoRM's diffusion score-model
//! refinement is FUNDAMENTALLY different from grouped Conv2D +
//! SB-TF-LSTM mask predictors, ERB analysis / synthesis + CRN mask
//! generators, or 2-layer GRU + 3-Linear log-magnitude mask
//! regressors — silently sharing an arch tag would misroute the
//! runtime dispatch (FR-EX-08).
//!
//! # Distinct arch tag from every sibling enhancement / separator family
//!
//! [`ARCH`] = `"storm"` is **deliberately distinct** from every sibling
//! enhancement / separation arch tag:
//!
//! - `denoise` — DeepFilterNet3 (ERB analysis/synthesis + CRN
//!   convolutional recurrent network — a completely different topology
//!   axis from StoRM's diffusion score-model refinement);
//! - `rnnoise` — Xiph RNNoise (GRU + BSD BFCC/Bark features);
//! - `nsnet2` — Microsoft DNS baseline (2-layer GRU + 3-Linear mask
//!   over 257-bin STFT log-magnitude);
//! - `dnsmos` — Microsoft P.808/P.835 DNSMOS objective quality
//!   estimator (a metric, not a denoiser);
//! - `gtcrn` — GTCRN (grouped Conv2D + SB-TF-LSTM + ERB grouping — a
//!   ~23K parameter mask predictor, different topology axis);
//! - `metricgan_plus`, `mp_senet_dns`, `frcrn`, `facebook_denoiser`,
//!   `mossformer2_ss_16k` — other enhancement variants with distinct
//!   topologies;
//! - `sepformer`, `conv_tasnet`, `demucs`, `tiger_separator`,
//!   `bs_roformer`, `mp_senet` — separator families with fundamentally
//!   different masker topologies.
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route
//! a StoRM checkpoint onto a wrong-topology loader — the diffusion
//! score-based two-stage stack has no near-neighbor in the catalogue
//! (StoRM is the FIRST diffusion-based entry on the enhancement arm).
//! FR-EX-08 forbids the silent shape misroute across enhancement
//! families.
//!
//! # License — MIT (primary source: upstream repo LICENSE)
//!
//! Both code and weights ship **MIT** end-to-end per the upstream
//! GitHub repo LICENSE
//! (`github.com/sp-uhh/storm/blob/main/LICENSE`, per task scout input
//! 2026-08-14 — CLAUDE.md「ハルシネーション厳禁」; the owner MUST
//! primary-source confirm the LICENSE at sign-off time. HF cardData
//! primary source is not applicable — the release is GitHub-only with
//! model checkpoints distributed via Google Drive, no HF mirror as of
//! 2026-08-14). MIT is a [`LicenseClass::Permissive`] license class —
//! same commercial verdict as apache-2.0 (no runtime-side attribution
//! obligation).
//!
//! §3.1 sign-off column in `docs/license-audit.md` is **BLANK**
//! (fail-closed default — CC MUST NOT sign a license row, that is
//! owner-only per memory `[[feedback-license-signoff-primary-source]]`).
//! Runtime binder land is unblocked (converter output can be *produced*
//! and *runtime-loaded* for structural testing under
//! [`LicenseClass::Unknown`] fail-closed at the M2-13 compliance gate);
//! *publish* is blocked until §3.1 is signed. **Publish path is also
//! gated on an owner ADR** — no HF mirror exists (Google Drive
//! distribution only) so the existing publish pipe (`upstream_hf` →
//! HF org mirror) is not directly applicable; owner must decide
//! whether to (a) treat StoRM as a T4 Research-only precedent similar
//! to X-Codec-2 or (b) establish a new T1 Permissive GitHub-source
//! precedent.
//!
//! # `vokra.storm.*` topology chunk group (6 axes)
//!
//! Every runtime hparam the future
//! `vokra-models::storm::Storm::from_gguf` needs is stamped here so a
//! downstream reader is fully self-describing (no external config
//! side-car needed). Values are FunASR-style `u32` chunks:
//!
//! - `vokra.storm.sample_rate` = 16000 (16 kHz mono speech per
//!   SGMSE+/StoRM standard config);
//! - `vokra.storm.n_fft` = 510 (STFT window size — NCSN++ speech-
//!   enhancement standard config per StoRM/SGMSE+);
//! - `vokra.storm.hop` = 128 (STFT hop, samples — typical for 16 kHz
//!   diffusion speech enhancement);
//! - `vokra.storm.d_model` = 128 (NCSN++ v2 U-Net base channel width);
//! - `vokra.storm.n_stages` = 4 (NCSN++ U-Net down/up-sampling stage
//!   count);
//! - `vokra.storm.score_channels` = 128 (score network base output
//!   width).
//!
//! Values above are the **typical StoRM config per arXiv:2312.09386
//! §III and the SGMSE+ Interspeech 2022 precedent Welker et al. cite**;
//! the implementer **MUST** re-confirm against the upstream
//! `github.com/sp-uhh/storm` `configs/*.yaml` at land time rather than
//! encoding from memory (CLAUDE.md「ハルシネーション厳禁」). The
//! runtime binder [`crates/vokra-models::storm::StormConfig`] holds
//! the same values as a `#[must_use]` `const` for the loud-partial
//! error message — the converter and the binder mirror the same
//! primary-source-transcribed defaults so a rename or axis-value change
//! must land in both crates in the same commit.
//!
//! **StoRM-specific note on structural invariants**: unlike sibling
//! GTCRN (n_bands = n_fft/2 + 1 real-input FFT), StoRM's score network
//! operates on the **full complex STFT** (real + imaginary channels),
//! not the half-real spectrum — so n_fft = 510 is even but the score
//! network does not decompose to a 256-bin band-count invariant. This
//! is a StoRM-specific difference from GTCRN's grouped Conv2D + SB-TF-
//! LSTM + ERB grouping stack; the runtime binder documents this in
//! [`StormConfig`] rustdoc rather than asserting a false invariant.
//!
//! # BF16 pass-through (mirror of the Wave 5/6 sepformer / conv_tasnet /
//! demucs / gtcrn skeletons)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 ([`GgmlType::BF16`]); runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. StoRM's
//! upstream distribution ships F32 checkpoints (typical <100 MB per
//! sub-model per sp-uhh/storm README), but the defensive BF16 path is
//! exercised for parity with the sibling pass-through skeletons.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `state_dict` names verbatim**.
//! Real-weight parity + a native `Storm::enhance` forward path are
//! **loud-partial** in the runtime binder pending the NCSN++ v2 U-Net
//! score-network + OUVE-SDE predictor-corrector sampler primitive
//! composition (`crates/vokra-models/src/storm/mod.rs` — no fabricated
//! denoised waveform ever emitted, FR-EX-08).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through + provenance
//! / category / topology chunk stamps). CLI + `ModelKind` + `pub use`
//! re-export in `lib.rs` land in the same commit. The module-level
//! `#[allow(dead_code)]` is temporary and removed as soon as callers
//! exercise the API — the same sibling wespeaker / redimnet /
//! conv_tasnet / sepformer / demucs / gtcrn pattern.
//!
//! # No ONNX / no pickle (permanent)
//!
//! StoRM ships as PyTorch state dict upstream (via Google Drive per
//! sp-uhh/storm README convention shared with SGMSE+); this converter
//! **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The `.pt`
//! → safetensors bridge lives offline through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), not part of the runtime — pickle
//! deserialization inside the Rust runtime would violate the FR-LD-05
//! "no arbitrary code execution at load" rule.

// Skeleton-only allowance: the public API is exercised by the
// in-module tests + wired to the CLI + `ModelKind` + `pub use`
// re-export in `lib.rs` in the same commit. Removed once callers
// exercise the API outside tests.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` = `storm` — distinct from every sibling denoise /
/// separator arch tag (`denoise` (DFN3), `rnnoise`, `nsnet2`, `dnsmos`,
/// `metricgan_plus`, `mp_senet_dns`, `sepformer`, `conv_tasnet`,
/// `demucs`, `gtcrn`). FR-EX-08 forbids silent shape misroute across
/// enhancement families. StoRM is the FIRST diffusion score-based entry
/// on the enhancement arm — no near-neighbor exists in the catalogue.
pub const ARCH: &str = "storm";

/// `vokra.model.name` — canonical `storm` release (single-config
/// StoRM checkpoint from sp-uhh/storm; no sibling variants distributed
/// by upstream as of 2026-08-14 — the whole StoRM release is one
/// topology at 16 kHz, so a `-16k` / `-se` suffix would be redundant).
pub const NAME: &str = "storm";

/// `vokra.model.category` = `enhancement` — single-mask enhancement +
/// dereverberation output. Even though StoRM has a two-stage
/// predict+refine pipeline, its ultimate output is single-channel
/// enhanced PCM (denoised + dereverberated), matching the
/// `enhancement` category posture of sibling GTCRN / DFN3 / NSNet2 /
/// RNNoise. Consumed by the model-card generator + zoo manifest tier
/// gate so a diffusion-based enhancement release is not accidentally
/// advertised as a generative TTS / music model.
pub const CATEGORY: &str = "enhancement";

/// Upstream GitHub tree the release ships from. StoRM is not hosted on
/// HuggingFace — the release lives at `github.com/sp-uhh/storm` with
/// checkpoints distributed via Google Drive (mirror of NSNet2 /
/// RNNoise / facebook_denoiser / NKF-AEC / GTCRN posture), so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either.
pub const UPSTREAM_URL: &str = "github.com/sp-uhh/storm";

/// Default weight license SPDX (`mit`) per the upstream repo LICENSE
/// (`github.com/sp-uhh/storm/blob/main/LICENSE`, per task scout input
/// 2026-08-14 — owner must primary-source confirm at sign-off time).
/// Overrides via the [`convert_storm_file`] `license` parameter — the
/// standing mechanism for "implementation is clean-room MIT but the
/// upstream distributed checkpoint is another license" scenarios
/// (mirror of `convert_file_licensed` in `lib.rs`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) matching the sibling
/// `nsnet2` / `gtcrn` / `emotion2vec` / `ecapa_tdnn` / `redimnet`
/// posture until a first-class `category` consumer lands in
/// `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream URL (used for non-HF sources
/// such as GitHub / Zenodo / ModelScope / Google Drive). Sibling to
/// `nsnet2::KEY_PROVENANCE_UPSTREAM_URL` and `gtcrn::
/// KEY_PROVENANCE_UPSTREAM_URL` — kept as a converter-side constant
/// to avoid premature promotion until a first-class consumer lands.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---- `vokra.storm.*` hparam chunk group (6 axes) ------------------------
//
// Mirror of `nsnet2::KEY_*` / `gtcrn::KEY_*` / `redimnet::GGUF_KEY_*`
// posture: every runtime hparam the future
// `vokra-models::storm::Storm::from_gguf` needs is stamped here so a
// downstream reader is fully self-describing (no external config
// side-car needed). Values are FunASR-style `u32` chunks; a
// `0`-sentinel on any of them makes the runtime binder refuse to load
// (FR-EX-08 — no silent default).

/// GGUF metadata key: PCM sample rate (u32 Hz; typical StoRM per
/// SGMSE+/StoRM standard config = 16 000).
pub const KEY_STORM_SAMPLE_RATE: &str = "vokra.storm.sample_rate";
/// GGUF metadata key: STFT FFT length (u32; typical StoRM = 510 per
/// NCSN++ speech-enhancement standard config).
pub const KEY_STORM_N_FFT: &str = "vokra.storm.n_fft";
/// GGUF metadata key: STFT hop (u32 samples; typical StoRM = 128 for
/// 16 kHz diffusion speech enhancement).
pub const KEY_STORM_HOP: &str = "vokra.storm.hop";
/// GGUF metadata key: NCSN++ v2 U-Net base channel width (u32; typical
/// StoRM = 128).
pub const KEY_STORM_D_MODEL: &str = "vokra.storm.d_model";
/// GGUF metadata key: NCSN++ U-Net down/up-sampling stage count (u32;
/// typical StoRM = 4).
pub const KEY_STORM_N_STAGES: &str = "vokra.storm.n_stages";
/// GGUF metadata key: score network base output width (u32; typical
/// StoRM = 128, mirrors `d_model` by construction in NCSN++ v2 but
/// stamped separately so a downstream reader can validate the shape).
pub const KEY_STORM_SCORE_CHANNELS: &str = "vokra.storm.score_channels";

/// Upstream PCM sample rate (Hz), typical StoRM per SGMSE+/StoRM
/// standard config — implementer MUST re-confirm against
/// `github.com/sp-uhh/storm/configs/*.yaml` at land time
/// (CLAUDE.md「ハルシネーション厳禁」).
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// Upstream STFT window size (samples), typical NCSN++ speech-
/// enhancement config per StoRM/SGMSE+ — implementer MUST re-confirm
/// against `github.com/sp-uhh/storm/configs/*.yaml` at land time.
pub const DEFAULT_N_FFT: u32 = 510;
/// Upstream STFT hop (samples), typical for 16 kHz diffusion speech
/// enhancement per StoRM — implementer MUST re-confirm against
/// `github.com/sp-uhh/storm/configs/*.yaml` at land time.
pub const DEFAULT_HOP: u32 = 128;
/// Upstream NCSN++ v2 U-Net base channel width, typical StoRM —
/// implementer MUST re-confirm against `github.com/sp-uhh/storm/
/// configs/*.yaml` at land time.
pub const DEFAULT_D_MODEL: u32 = 128;
/// Upstream NCSN++ U-Net down/up-sampling stage count, typical StoRM —
/// implementer MUST re-confirm against `github.com/sp-uhh/storm/
/// configs/*.yaml` at land time.
pub const DEFAULT_N_STAGES: u32 = 4;
/// Upstream score network base output width, typical StoRM —
/// implementer MUST re-confirm against `github.com/sp-uhh/storm/
/// configs/*.yaml` at land time.
pub const DEFAULT_SCORE_CHANNELS: u32 = 128;

const UPSTREAM_SOURCE: &str = "sp-uhh/storm (StoRM: A Diffusion-based Stochastic Regeneration Model \
     for Speech Enhancement and Dereverberation, NCSN++ v2 score network + OUVE-SDE predictor-corrector, \
     16 kHz speech, arXiv:2312.09386, MIT)";

/// Outcome of a StoRM conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `StormReport::default()` and the caller remains
/// responsible for surfacing the "no float tensors" loud note (mirror
/// of the `gtcrn` / `nsnet2` / `emotion2vec` / `ecapa_tdnn` `Report`
/// pattern).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StormReport {
    /// Total tensors surfaced by the safetensors reader (the sum of
    /// `written + skipped_non_float`). Pins the budget so a truncated
    /// header cannot silently drop tensors without the caller noticing.
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
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a StoRM GGUF
/// to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url) chunk groups are stamped for the runtime compliance
/// gate (FR-CP-03) alongside the `vokra.storm.*` 6-axis topology chunk
/// group. `vokra.schema.*` is written unconditionally by the GGUF
/// writer.
///
/// `license` overrides `DEFAULT_LICENSE_SPDX` (`"mit"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed checkpoint
/// carries a different SPDX.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_storm_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<StormReport, ConvertError> {
    // Whole-file read: StoRM ships checkpoints <100 MB per sub-model
    // per sp-uhh/storm README (both the initial deterministic
    // predictive estimator and the NCSN++ v2 score-network fit
    // comfortably) — no need for the streaming path the Moshi 15 GB /
    // Voxtral 8.7 GB converters run. M1 iMac (16 GB) safe per memory
    // `[[feedback-large-models-on-vast-ai]]`.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own
    // licence. StoRM ships MIT end-to-end per the upstream GitHub repo
    // LICENSE (`github.com/sp-uhh/storm/blob/main/LICENSE`).
    // The `license` override lets a downstream repackager stamp a
    // different SPDX if they redistribute under stricter terms (the
    // same knob `convert_file_licensed` exposes in `lib.rs`).
    let effective_license = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // StoRM has one canonical single-config release (SGMSE+/StoRM
    // 16 kHz baseline) and every hparam is fixed at that release.
    // Stamping them here (mirror of `nsnet2::stamp_hparams` /
    // `gtcrn::convert_gtcrn_file` posture) makes the artifact
    // self-describing so the future `vokra-models::storm::Storm::
    // from_gguf` binder can validate against these values loudly
    // (FR-EX-08 — a checkpoint that came from a different topology
    // cannot silently misload). CLAUDE.md「ハルシネーション厳禁」:
    // owner MUST re-confirm these axes against the upstream repo
    // `configs/*.yaml` at land time rather than trusting the
    // transcribed constants alone.
    b.add_u32(KEY_STORM_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_STORM_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_STORM_HOP, DEFAULT_HOP);
    b.add_u32(KEY_STORM_D_MODEL, DEFAULT_D_MODEL);
    b.add_u32(KEY_STORM_N_STAGES, DEFAULT_N_STAGES);
    b.add_u32(KEY_STORM_SCORE_CHANNELS, DEFAULT_SCORE_CHANNELS);

    let mut report = StormReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (`docs/adr/qwen3-tts-bf16.md`, strategy A_passthrough); the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    // Mirrors `nsnet2::convert_nsnet2_file` / `sepformer::
    // convert_sepformer_file` / `gtcrn::convert_gtcrn_file`.
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
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + monotonically increasing
    /// sequence — the sepformer / conv_tasnet / gtcrn test pattern; no
    /// external `tempfile` dep, preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-storm-{tag}-{}-{n}",
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

    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(f32_bytes.len(), f32_elems as usize * 4);
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(f16_bytes.len(), f16_elems as usize * 2);
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

    // -----------------------------------------------------------------
    // Test 1 — BF16 round-trip + full topology + provenance stamps
    // -----------------------------------------------------------------

    /// BF16 pass-through pin: even though upstream StoRM is F32, any
    /// future half-precision distillation must ride the same arm
    /// without a converter change. The dtype must stay BF16 (GGUF type
    /// 30) and the payload must be byte-identical (a silent widen
    /// would still round-trip values but would break the byte pin).
    /// The `vokra.model.category = "enhancement"` +
    /// `vokra.provenance.upstream_url = github.com/sp-uhh/storm` +
    /// `vokra.model.arch = "storm"` stamps + every 6-axis
    /// `vokra.storm.*` chunk MUST land on the artifact.
    #[test]
    fn bf16_tensor_passes_through_and_full_metadata_lands() {
        // Non-zero BF16 bit patterns so any silent widen / downcast
        // attempt is caught by the subsequent byte-identity assert
        // (zeroed payloads would round-trip trivially through F32 /
        // F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements x 2 bytes BF16 payload");

        // Mirror a plausible upstream StoRM state_dict tensor name
        // (NCSN++ v2 U-Net down-sampling stage 0 initial convolution).
        // Actual state_dict prefixes are speculative here — owner will
        // pin the real names once the manifest lands via the sidecar
        // `.pt` → safetensors bridge. This fixture only exercises the
        // byte-copy path.
        let input_bytes =
            safetensors_one("score_network.down.0.conv.weight", "BF16", &[2, 3], &bf16);
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_storm_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor visible in header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror sepformer / conv_tasnet / gtcrn)"
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
            .tensor_info("score_network.down.0.conv.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
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
            "category chunk pins StoRM as `enhancement`"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
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
        // Every `vokra.storm.*` axis must be stamped verbatim so a
        // downstream `Storm::from_gguf` binder can validate the
        // topology.
        for (k, want) in [
            (KEY_STORM_SAMPLE_RATE, DEFAULT_SAMPLE_RATE),
            (KEY_STORM_N_FFT, DEFAULT_N_FFT),
            (KEY_STORM_HOP, DEFAULT_HOP),
            (KEY_STORM_D_MODEL, DEFAULT_D_MODEL),
            (KEY_STORM_N_STAGES, DEFAULT_N_STAGES),
            (KEY_STORM_SCORE_CHANNELS, DEFAULT_SCORE_CHANNELS),
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

    // -----------------------------------------------------------------
    // Test 2 — F32 + F16 mixed pass-through (BF16 counter stays at 0)
    // -----------------------------------------------------------------

    /// Mixed F32/F16 round-trip pin: both dtypes must ride the same
    /// pass-through arm; the BF16 subset counter MUST stay at zero
    /// (defence against a hypothetical regression where a widen-to-BF16
    /// path silently upcasts F32 / F16 into BF16 to inflate the
    /// pass-through counter).
    #[test]
    fn f32_and_f16_tensors_pass_through_no_bf16_upcast() {
        // Non-zero payloads so a silent-widen regression can't hide
        // behind trivial round-trips.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). Values chosen to be exact F16 patterns.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);

        let input_bytes = safetensors_f32_then_f16(
            "predictor.encoder.0.weight",
            &[1, 2],
            &f32_bytes,
            "score_network.up.0.conv.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = scratch_path("mixed-in");
        let output = scratch_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_storm_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter (no silent upcast)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let f32_info = file
            .tensor_info("predictor.encoder.0.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("score_network.up.0.conv.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 3 — License override swaps the stamped SPDX + class
    // -----------------------------------------------------------------

    /// License override pin: passing `Some("apache-2.0")` re-derives
    /// the class through `LicenseClass::from_license_str` and stamps
    /// the new SPDX + class on the artifact. Both MIT and apache-2.0
    /// map to `Permissive`, so the class stays permissive but the raw
    /// SPDX string flips — this pin guards against a hard-coded
    /// `"mit"` regression at the provenance stamp site.
    #[test]
    fn license_override_swaps_spdx_and_class_stays_permissive() {
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("x", "F32", &[1], &payload);
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_storm_file(&input, &output, Some("apache-2.0")).expect("convert with override");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX lands verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 normalises to LicenseClass::Permissive (same class as the MIT default)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 4 — All 6 `vokra.storm.*` axes emit the transcribed typical
    //          values (rename / axis-value regression pin)
    // -----------------------------------------------------------------

    /// Pin the primary-source-transcribed axes (arXiv:2312.09386 §III
    /// + SGMSE+ Interspeech 2022 precedent typical config) as u32
    /// chunks. A rename or default-value change would land here in the
    /// same commit or fail this test.
    ///
    /// **StoRM-specific note**: unlike sibling GTCRN
    /// (n_bands = n_fft/2 + 1 real-input FFT), StoRM's score network
    /// operates on the **full complex STFT** (real + imaginary
    /// channels), not the half-real spectrum. So n_fft = 510 is even
    /// but the score network does not decompose to a 256-bin band-
    /// count invariant. We deliberately DO NOT assert
    /// `n_bands == n_fft/2 + 1` here (there is no n_bands axis in
    /// StoRM's config — it operates on the full complex STFT directly).
    /// This is a StoRM-specific difference from GTCRN documented in
    /// the module doc.
    #[test]
    fn all_six_storm_axes_emit_expected_typical_values() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("dummy.weight", "F32", &[1], &payload);
        let input = scratch_path("axes-in");
        let output = scratch_path("axes-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_storm_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // Every axis default MUST match the transcribed
        // arXiv:2312.09386 §III typical config. Owner MUST re-confirm
        // against `github.com/sp-uhh/storm/configs/*.yaml` at land time.
        assert_eq!(
            file.get(KEY_STORM_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(16_000u32)),
            "sample_rate default = 16 kHz (StoRM typical per paper §III)"
        );
        assert_eq!(
            file.get(KEY_STORM_N_FFT).and_then(|v| v.as_u64()),
            Some(u64::from(510u32)),
            "n_fft default = 510 (StoRM typical per NCSN++ speech-enhancement config)"
        );
        assert_eq!(
            file.get(KEY_STORM_HOP).and_then(|v| v.as_u64()),
            Some(u64::from(128u32)),
            "hop default = 128 samples (StoRM typical per NCSN++ speech-enhancement config)"
        );
        assert_eq!(
            file.get(KEY_STORM_D_MODEL).and_then(|v| v.as_u64()),
            Some(u64::from(128u32)),
            "d_model default = 128 (NCSN++ v2 U-Net base channel width, StoRM typical)"
        );
        assert_eq!(
            file.get(KEY_STORM_N_STAGES).and_then(|v| v.as_u64()),
            Some(u64::from(4u32)),
            "n_stages default = 4 (NCSN++ U-Net down/up-sampling stage count, StoRM typical)"
        );
        assert_eq!(
            file.get(KEY_STORM_SCORE_CHANNELS).and_then(|v| v.as_u64()),
            Some(u64::from(128u32)),
            "score_channels default = 128 (score network base output width, StoRM typical)"
        );

        // NOTE: unlike sibling GTCRN, we do NOT assert
        // `n_bands == n_fft/2 + 1` — StoRM's score network operates on
        // the full complex STFT (real + imaginary channels), not the
        // half-real spectrum. There is no n_bands axis in StoRM's
        // config by design. This is a StoRM-specific difference from
        // GTCRN's grouped Conv2D + SB-TF-LSTM + ERB grouping stack.

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 5 — `vokra.model.name = "storm"` (name-tag stability pin)
    // -----------------------------------------------------------------

    /// Pin the stamped `vokra.model.name` to `"storm"` so a rename
    /// would land here in the same commit or fail this test. StoRM
    /// ships a single 16 kHz release; sibling variants would each
    /// carry their own [`crate::ModelKind`] arm + distinct
    /// [`NAME`] stamp per the sibling naming convention.
    #[test]
    fn model_name_pin_is_storm() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("x", "F32", &[1], &payload);
        let input = scratch_path("name-in");
        let output = scratch_path("name-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");
        convert_storm_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("storm"),
            "vokra.model.name must be stamped as `storm`"
        );
        assert_eq!(NAME, "storm", "NAME constant pinned");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 6 — arch tag distinct from every sibling enhancement /
    //          separator family (FR-EX-08 pin)
    // -----------------------------------------------------------------

    /// Pin `ARCH = "storm"` and assert distinctness against every
    /// sibling enhancement / separator arch string. A future rename
    /// of any sibling arch tag would land here in the same commit or
    /// fail this test (mirror of the sepformer / conv_tasnet / gtcrn
    /// distinctness pins).
    ///
    /// Note: `gtcrn` goes into the sibling list because StoRM is a
    /// distinct arch from GTCRN too — StoRM's diffusion score-based
    /// two-stage stack (NCSN++ v2 + OUVE-SDE) and GTCRN's grouped
    /// Conv2D + SB-TF-LSTM + ERB grouping stack are FUNDAMENTALLY
    /// different topology axes.
    #[test]
    fn arch_tag_distinct_from_sibling_enhancement_and_separator_arches() {
        assert_eq!(ARCH, "storm");
        assert_eq!(CATEGORY, "enhancement");
        // Direct string comparisons against every sibling arch tag to
        // document the "which sibling should NOT be aliased" contract
        // at test time. `gtcrn` included: StoRM ≠ GTCRN (different
        // topology).
        for sibling in [
            "denoise",            // DeepFilterNet3
            "rnnoise",            // Xiph RNNoise (BSD)
            "nsnet2",             // Microsoft DNS baseline
            "dnsmos",             // Microsoft DNSMOS metric
            "metricgan_plus",     // MetricGAN+
            "mp_senet_dns",       // MP-SENet DNS variant
            "sepformer",          // SpeechBrain SepFormer
            "conv_tasnet",        // Asteroid ConvTasNet
            "demucs",             // Facebook Demucs / HT-Demucs
            "frcrn",              // FRCRN
            "mossformer2_ss_16k", // MossFormer2
            "facebook_denoiser",  // Meta Denoiser
            "gtcrn",              // GTCRN (Wave 6 sibling)
        ] {
            assert_ne!(
                ARCH, sibling,
                "storm (NCSN++ v2 score-network + OUVE-SDE predictor-corrector) and \
                 `{sibling}` are distinct enhancement / separator arches — sharing arch \
                 tag would misroute the runtime dispatch (FR-EX-08)"
            );
        }
    }
}
