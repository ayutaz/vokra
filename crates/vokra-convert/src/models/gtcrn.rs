#![allow(clippy::doc_lazy_continuation)]
//! **GTCRN** (`Xiaobin-Rong/gtcrn`, **MIT**) — Groupwise Temporal
//! Convolutional Recurrent Network speech enhancement: safetensors →
//! GGUF conversion (Wave 6 2026-08-14 audit follow-up, denoise
//! alternative sibling to DFN3 / NSNet2 / RNNoise).
//!
//! # Model class — ultra-lightweight streaming speech enhancement
//!
//! GTCRN (Rong et al. arXiv:2211.02063 "GTCRN: A Speech Enhancement
//! Model Requiring Ultralow Computational Resources") — a ~23K
//! parameter STFT-domain enhancement model designed for embedded /
//! streaming applications. The architecture combines:
//!
//! - a **grouped 2D Conv encoder** (channel-grouped depthwise-style
//!   convolutions over log-magnitude STFT to compress the feature
//!   frame count) with **PReLU** activations;
//! - an **SB-TF-LSTM (sub-band time-frequency LSTM)** bottleneck that
//!   models both temporal and frequency dependencies with a per-band
//!   grouped RNN (per upstream terminology, GTCRN uses **LSTM** cells
//!   rather than GRU in the sub-band branch);
//! - an **ERB (equivalent rectangular bandwidth) frequency-band
//!   grouping** applied over the 257-bin STFT (linear STFT → perceptual
//!   ERB band aggregation) as an efficiency-preserving frequency-axis
//!   pooler; and
//! - a **grouped 2D Conv decoder** (mirror of the encoder) emitting a
//!   per-bin gain mask that is applied to the complex STFT (phase
//!   preserved) before the streaming iSTFT.
//!
//! Its role in the Vokra catalogue is a *third* denoise alternative
//! (sibling of `denoise` / DeepFilterNet3 and `nsnet2` / Microsoft DNS
//! baseline and `rnnoise` / Xiph BSD baseline) — deliberately weaker /
//! smaller than DFN3 but structurally distinct enough that silently
//! sharing the `denoise` arch tag would misroute the runtime dispatch
//! (FR-EX-08).
//!
//! # Distinct arch tag from every sibling enhancement / separator family
//!
//! [`ARCH`] = `"gtcrn"` is **deliberately distinct** from every sibling
//! enhancement / separation arch tag:
//!
//! - `denoise` — DeepFilterNet3 (ERB analysis/synthesis + convolutional
//!   recurrent network — a different ERB posture: DFN3 uses a real
//!   ERB analysis/synthesis pair around a CRN; GTCRN uses ERB grouping
//!   only for feature aggregation over a grouped Conv2D backbone);
//! - `rnnoise` — Xiph RNNoise (GRU + BSD BFCC/Bark features);
//! - `nsnet2` — Microsoft DNS baseline (2-layer GRU + 3-Linear mask
//!   over 257-bin STFT log-magnitude);
//! - `dnsmos` — Microsoft P.808/P.835 DNSMOS objective quality
//!   estimator (a metric, not a denoiser);
//! - `metricgan_plus`, `mp_senet_dns`, `frcrn`, `facebook_denoiser`,
//!   `mossformer2_ss_16k` — other enhancement / separator variants
//!   with distinct topologies;
//! - `sepformer`, `conv_tasnet`, `demucs`, `tiger_separator`,
//!   `bs_roformer`, `mp_senet` — separator families with fundamentally
//!   different masker topologies.
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route
//! a GTCRN checkpoint onto a wrong-topology loader — the grouped Conv2D
//! + SB-TF-LSTM + ERB-grouping stack has no near-neighbor in the
//! catalogue. FR-EX-08 forbids the silent shape misroute across
//! enhancement families.
//!
//! # License — MIT (primary source: upstream repo LICENSE)
//!
//! Both code and weights ship **MIT** end-to-end per the upstream
//! GitHub repo LICENSE
//! (`github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE`, per task
//! scout input 2026-08-14 — CLAUDE.md「ハルシネーション厳禁」;
//! the owner must primary-source confirm the LICENSE at sign-off
//! time. HF cardData primary source is not applicable — the release
//! is GitHub-only, no HF mirror as of 2026-08-14). MIT is a
//! [`LicenseClass::Permissive`] license class — same commercial verdict
//! as apache-2.0 (no runtime-side attribution obligation).
//!
//! §3.1 sign-off column in `docs/license-audit.md` is **BLANK**
//! (fail-closed default — CC MUST NOT sign a license row, that is
//! owner-only per memory `[[feedback-license-signoff-primary-source]]`).
//! Runtime binder land is unblocked (converter output can be *produced*
//! and *runtime-loaded* for structural testing under
//! `LicenseClass::Unknown` fail-closed at the M2-13 compliance gate);
//! *publish* is blocked until §3.1 is signed.
//!
//! # `vokra.gtcrn.*` topology chunk group (5 axes)
//!
//! Every runtime hparam the future
//! `vokra-models::gtcrn::Gtcrn::from_gguf` needs is stamped here so a
//! downstream reader is fully self-describing (no external config
//! side-car needed). Values are FunASR-style `u32` chunks:
//!
//! - `vokra.gtcrn.sample_rate` = 16000 (16 kHz mono, per arXiv:2211.02063 §III);
//! - `vokra.gtcrn.n_fft` = 512 (STFT window size);
//! - `vokra.gtcrn.hop` = 256 (STFT hop, samples);
//! - `vokra.gtcrn.n_bands` = 257 (STFT bin count, `n_fft/2 + 1`);
//! - `vokra.gtcrn.gru_hidden` = 64 (sub-band recurrent hidden width —
//!   the metadata key uses the generic `gru_hidden` label per the task
//!   scout to avoid promoting an implementation-detail RNN cell kind
//!   into the metadata surface; the module doc calls out that the
//!   upstream sub-band branch is an **LSTM** per arXiv:2211.02063 §III).
//!
//! Values above are the **typical GTCRN config per arXiv:2211.02063
//! §III**; the implementer **MUST** re-confirm against the upstream
//! `github.com/Xiaobin-Rong/gtcrn` config at land time rather than
//! encoding from memory (CLAUDE.md「ハルシネーション厳禁」). The
//! runtime binder [`crates/vokra-models::gtcrn::GtcrnConfig`] holds
//! the same values as a `#[must_use]` `const` for the loud-partial
//! error message — the converter and the binder mirror the same
//! primary-source-transcribed defaults so a rename or axis-value change
//! must land in both crates in the same commit.
//!
//! # BF16 pass-through (mirror of the Wave 5 sepformer / conv_tasnet /
//! demucs skeletons)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 ([`GgmlType::BF16`]); runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. GTCRN itself
//! ships F32 in the wild (~23K parameters, well under 200 KB total —
//! the smallest converter footprint in the catalogue), but the
//! defensive BF16 path is exercised for parity with the sibling
//! pass-through skeletons.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `state_dict` names verbatim**.
//! Real-weight parity + a native `Gtcrn::denoise` forward path are
//! **loud-partial** in the runtime binder pending the WeSpeaker /
//! Asteroid style tensor-name walk + grouped Conv2D / PReLU / SB-TF-LSTM
//! / ERB-grouping primitive composition (`crates/vokra-models/src/
//! gtcrn/mod.rs` — no fabricated denoised waveform ever emitted,
//! FR-EX-08).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through + provenance
//! / category / topology chunk stamps). CLI + `ModelKind` + `pub use`
//! re-export in `lib.rs` land in the same commit. The module-level
//! `#[allow(dead_code)]` is temporary and removed as soon as callers
//! exercise the API — the same sibling wespeaker / redimnet /
//! conv_tasnet / sepformer / demucs pattern.
//!
//! # No ONNX / no pickle (permanent)
//!
//! GTCRN ships as PyTorch state dict upstream; this converter **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The `.pt` →
//! safetensors bridge lives offline through the sibling
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

/// `vokra.model.arch` = `gtcrn` — distinct from every sibling denoise /
/// separator arch tag (`denoise` (DFN3), `rnnoise`, `nsnet2`, `dnsmos`,
/// `metricgan_plus`, `mp_senet_dns`, `sepformer`, `conv_tasnet`,
/// `demucs`). FR-EX-08 forbids silent shape misroute across enhancement
/// families.
pub const ARCH: &str = "gtcrn";

/// `vokra.model.name` — canonical `gtcrn` release (~23K parameter
/// single-config release; no sibling variants distributed by upstream
/// as of 2026-08-14 — the whole GTCRN release is one topology at
/// 16 kHz, so a `-16k` suffix is redundant).
pub const NAME: &str = "gtcrn";

/// `vokra.model.category` = `enhancement` — single-mask denoise head
/// (mirror of the sibling `denoise` (DFN3) / `nsnet2` / `rnnoise`
/// enhancement family posture). Consumed by the model-card generator +
/// zoo manifest tier gate so a denoise baseline is not accidentally
/// advertised as an ASR / TTS release.
pub const CATEGORY: &str = "enhancement";

/// Upstream GitHub tree the release ships from. GTCRN is not hosted on
/// HuggingFace (upstream is a GitHub-only public repository — mirror of
/// NSNet2 / RNNoise / facebook_denoiser / NKF-AEC posture), so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either.
pub const UPSTREAM_URL: &str = "github.com/Xiaobin-Rong/gtcrn";

/// Default weight license SPDX (`mit`) per the upstream repo LICENSE
/// (`github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE`, per task scout
/// input 2026-08-14 — owner must primary-source confirm at sign-off
/// time). Overrides via the [`convert_gtcrn_file`] `license` parameter
/// — the standing mechanism for "implementation is clean-room MIT but
/// the upstream distributed checkpoint is another license" scenarios
/// (mirror of `convert_file_licensed` in `lib.rs`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) matching the sibling
/// `nsnet2` / `emotion2vec` / `ecapa_tdnn` / `redimnet` posture until
/// a first-class `category` consumer lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream URL (used for non-HF sources
/// such as GitHub / Zenodo / ModelScope). Sibling to
/// `nsnet2::KEY_PROVENANCE_UPSTREAM_URL` — kept as a converter-side
/// constant to avoid premature promotion until a first-class consumer
/// lands.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---- `vokra.gtcrn.*` hparam chunk group ---------------------------------
//
// Mirror of `nsnet2::KEY_*` / `redimnet::GGUF_KEY_*` posture: every
// runtime hparam the future `vokra-models::gtcrn::Gtcrn::from_gguf`
// needs is stamped here so a downstream reader is fully self-describing
// (no external config side-car needed). Values are FunASR-style `u32`
// chunks; a `0`-sentinel on any of them makes the runtime binder
// refuse to load (FR-EX-08 — no silent default).

/// GGUF metadata key: PCM sample rate (u32 Hz; typical GTCRN per
/// arXiv:2211.02063 §III = 16 000).
pub const KEY_GTCRN_SAMPLE_RATE: &str = "vokra.gtcrn.sample_rate";
/// GGUF metadata key: STFT FFT length (u32; typical GTCRN = 512).
pub const KEY_GTCRN_N_FFT: &str = "vokra.gtcrn.n_fft";
/// GGUF metadata key: STFT hop (u32 samples; typical GTCRN = 256 =
/// 16 ms at 16 kHz — a longer hop than NSNet2's 10 ms per GTCRN's
/// low-latency streaming budget).
pub const KEY_GTCRN_HOP: &str = "vokra.gtcrn.hop";
/// GGUF metadata key: STFT bin count / ERB analysis width (u32; typical
/// GTCRN = 257 = `n_fft/2 + 1`).
pub const KEY_GTCRN_N_BANDS: &str = "vokra.gtcrn.n_bands";
/// GGUF metadata key: sub-band recurrent hidden width (u32; typical
/// GTCRN = 64). The metadata key uses the generic `gru_hidden` label
/// per the task scout to avoid promoting an implementation-detail RNN
/// cell kind (LSTM vs GRU) into the on-disk metadata surface — the
/// runtime binder's rustdoc calls out that the upstream sub-band
/// branch is **LSTM** per arXiv:2211.02063 §III, not GRU.
pub const KEY_GTCRN_GRU_HIDDEN: &str = "vokra.gtcrn.gru_hidden";

/// Upstream PCM sample rate (Hz), typical GTCRN per arXiv:2211.02063 §III.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// Upstream STFT window size (samples), typical GTCRN per arXiv:2211.02063 §III.
pub const DEFAULT_N_FFT: u32 = 512;
/// Upstream STFT hop (samples = 16 ms at 16 kHz), typical GTCRN per
/// arXiv:2211.02063 §III.
pub const DEFAULT_HOP: u32 = 256;
/// Upstream STFT bin count (= `n_fft/2 + 1`), typical GTCRN per
/// arXiv:2211.02063 §III.
pub const DEFAULT_N_BANDS: u32 = 257;
/// Upstream sub-band recurrent hidden width, typical GTCRN per
/// arXiv:2211.02063 §III.
pub const DEFAULT_GRU_HIDDEN: u32 = 64;

const UPSTREAM_SOURCE: &str = "Xiaobin-Rong/gtcrn (GTCRN: A Speech Enhancement Model Requiring Ultralow \
     Computational Resources, ~23K params, 16 kHz mono streaming denoise, \
     arXiv:2211.02063, MIT)";

/// Outcome of a GTCRN conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `GtcrnReport::default()` and the caller remains
/// responsible for surfacing the "no float tensors" loud note (mirror
/// of the `nsnet2` / `emotion2vec` / `ecapa_tdnn` `Report` pattern).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GtcrnReport {
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

/// Reads a safetensors checkpoint at `input` and writes a GTCRN GGUF
/// to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url) chunk groups are stamped for the runtime compliance
/// gate (FR-CP-03) alongside the `vokra.gtcrn.*` 5-axis topology chunk
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
pub fn convert_gtcrn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<GtcrnReport, ConvertError> {
    // Whole-file read: GTCRN ships ~23K parameters (~90 KB F32
    // safetensors) — no need for the streaming path the Moshi 15 GB /
    // Voxtral 8.7 GB converters run.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own
    // licence. GTCRN ships MIT end-to-end per the upstream GitHub repo
    // LICENSE (`github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE`).
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

    // GTCRN has one canonical single-config release — the ~23K
    // parameter 16 kHz streaming baseline — and every hparam is fixed
    // at that release. Stamping them here (mirror of `nsnet2::
    // stamp_hparams` / `redimnet::convert_redimnet_file` posture) makes
    // the artifact self-describing so the future `vokra-models::gtcrn::
    // Gtcrn::from_gguf` binder can validate against these values loudly
    // (FR-EX-08 — a checkpoint that came from a different topology
    // cannot silently misload). CLAUDE.md「ハルシネーション厳禁」:
    // owner MUST re-confirm these axes against the upstream repo at
    // land time rather than trusting the transcribed constants alone.
    b.add_u32(KEY_GTCRN_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_GTCRN_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_GTCRN_HOP, DEFAULT_HOP);
    b.add_u32(KEY_GTCRN_N_BANDS, DEFAULT_N_BANDS);
    b.add_u32(KEY_GTCRN_GRU_HIDDEN, DEFAULT_GRU_HIDDEN);

    let mut report = GtcrnReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (`docs/adr/qwen3-tts-bf16.md`, strategy A_passthrough); the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    // Mirrors `nsnet2::convert_nsnet2_file` / `sepformer::
    // convert_sepformer_file` / `conv_tasnet_libri1mix::
    // convert_conv_tasnet_libri1mix_file`.
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
    /// sequence — the sepformer / conv_tasnet test pattern; no external
    /// `tempfile` dep, preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-gtcrn-{tag}-{}-{n}",
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

    /// BF16 pass-through pin: even though upstream GTCRN is F32, any
    /// future half-precision distillation must ride the same arm
    /// without a converter change. The dtype must stay BF16 (GGUF type
    /// 30) and the payload must be byte-identical (a silent widen
    /// would still round-trip values but would break the byte pin).
    /// The `vokra.model.category = "enhancement"` +
    /// `vokra.provenance.upstream_url = github.com/Xiaobin-Rong/gtcrn`
    /// + `vokra.model.arch = "gtcrn"` stamps + every 5-axis
    /// `vokra.gtcrn.*` chunk MUST land on the artifact.
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

        // Mirror a plausible upstream GTCRN state_dict tensor name.
        // Encoder / decoder grouped-Conv2D block layout is speculative
        // here (owner will pin the real name once the manifest lands);
        // this fixture only exercises the byte-copy path.
        let input_bytes = safetensors_one("en_conv.0.conv.weight", "BF16", &[2, 3], &bf16);
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_gtcrn_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor visible in header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror sepformer / conv_tasnet)"
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
            .tensor_info("en_conv.0.conv.weight")
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
            "category chunk pins GTCRN as `enhancement`"
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
        // Every `vokra.gtcrn.*` axis must be stamped verbatim so a
        // downstream `Gtcrn::from_gguf` binder can validate the
        // topology.
        for (k, want) in [
            (KEY_GTCRN_SAMPLE_RATE, DEFAULT_SAMPLE_RATE),
            (KEY_GTCRN_N_FFT, DEFAULT_N_FFT),
            (KEY_GTCRN_HOP, DEFAULT_HOP),
            (KEY_GTCRN_N_BANDS, DEFAULT_N_BANDS),
            (KEY_GTCRN_GRU_HIDDEN, DEFAULT_GRU_HIDDEN),
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
            "sb_lstm.weight_ih_l0",
            &[1, 2],
            &f32_bytes,
            "de_conv.0.conv.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = scratch_path("mixed-in");
        let output = scratch_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_gtcrn_file(&input, &output, None).expect("convert");
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
            .tensor_info("sb_lstm.weight_ih_l0")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("de_conv.0.conv.weight")
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

        convert_gtcrn_file(&input, &output, Some("apache-2.0")).expect("convert with override");
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
    // Test 4 — All 5 `vokra.gtcrn.*` axes emit the transcribed typical
    //          values (rename / axis-value regression pin)
    // -----------------------------------------------------------------

    /// Pin the primary-source-transcribed axes (arXiv:2211.02063 §III
    /// typical config) as u32 chunks. A rename or default-value change
    /// would land here in the same commit or fail this test.
    #[test]
    fn all_five_gtcrn_axes_emit_expected_typical_values() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("dummy.weight", "F32", &[1], &payload);
        let input = scratch_path("axes-in");
        let output = scratch_path("axes-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_gtcrn_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // Every axis default MUST match the transcribed
        // arXiv:2211.02063 §III typical config. Owner MUST re-confirm
        // against `github.com/Xiaobin-Rong/gtcrn` at land time.
        assert_eq!(
            file.get(KEY_GTCRN_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(16_000u32)),
            "sample_rate default = 16 kHz (GTCRN typical per paper §III)"
        );
        assert_eq!(
            file.get(KEY_GTCRN_N_FFT).and_then(|v| v.as_u64()),
            Some(u64::from(512u32)),
            "n_fft default = 512 (GTCRN typical per paper §III)"
        );
        assert_eq!(
            file.get(KEY_GTCRN_HOP).and_then(|v| v.as_u64()),
            Some(u64::from(256u32)),
            "hop default = 256 samples (16 ms at 16 kHz — GTCRN typical per paper §III)"
        );
        assert_eq!(
            file.get(KEY_GTCRN_N_BANDS).and_then(|v| v.as_u64()),
            Some(u64::from(257u32)),
            "n_bands default = 257 (= n_fft/2 + 1)"
        );
        assert_eq!(
            file.get(KEY_GTCRN_GRU_HIDDEN).and_then(|v| v.as_u64()),
            Some(u64::from(64u32)),
            "gru_hidden default = 64 (sub-band recurrent hidden width)"
        );

        // Structural invariant: n_bands = n_fft/2 + 1 (real-input FFT).
        // The converter defaults respect this; encoding a mismatched
        // GGUF is a converter bug that would be caught here.
        assert_eq!(
            DEFAULT_N_BANDS,
            DEFAULT_N_FFT / 2 + 1,
            "structural invariant: n_bands must equal n_fft/2 + 1"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 5 — `vokra.model.name = "gtcrn"` (name-tag stability pin)
    // -----------------------------------------------------------------

    /// Pin the stamped `vokra.model.name` to `"gtcrn"` so a rename
    /// would land here in the same commit or fail this test. GTCRN
    /// ships a single 16 kHz release; sibling variants would each
    /// carry their own [`crate::ModelKind`] arm + distinct
    /// [`NAME`] stamp per the sibling naming convention.
    #[test]
    fn model_name_pin_is_gtcrn() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("x", "F32", &[1], &payload);
        let input = scratch_path("name-in");
        let output = scratch_path("name-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");
        convert_gtcrn_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("gtcrn"),
            "vokra.model.name must be stamped as `gtcrn`"
        );
        assert_eq!(NAME, "gtcrn", "NAME constant pinned");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 6 — arch tag distinct from every sibling denoise / separator
    //          family (FR-EX-08 pin)
    // -----------------------------------------------------------------

    /// Pin `ARCH = "gtcrn"` and assert distinctness against every
    /// sibling enhancement / separator arch string. A future rename
    /// of any sibling arch tag would land here in the same commit or
    /// fail this test (mirror of the sepformer / conv_tasnet distinctness
    /// pins).
    #[test]
    fn arch_tag_distinct_from_sibling_enhancement_and_separator_arches() {
        assert_eq!(ARCH, "gtcrn");
        assert_eq!(CATEGORY, "enhancement");
        // Direct string comparisons against every sibling arch tag to
        // document the "which sibling should NOT be aliased" contract
        // at test time.
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
        ] {
            assert_ne!(
                ARCH, sibling,
                "gtcrn (grouped Conv2D + SB-TF-LSTM + ERB grouping) and `{sibling}` are \
                 distinct enhancement / separator arches — sharing arch tag would \
                 misroute the runtime dispatch (FR-EX-08)"
            );
        }
    }
}
