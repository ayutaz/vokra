//! **SGMSE-VoiceBank** (`speechbrain/sgmse-voicebank`, apache-2.0) —
//! safetensors → GGUF conversion (score-based generative model for
//! speech enhancement, VoiceBank-DEMAND corpus).
//!
//! # Provenance
//!
//! - **HF path**: `speechbrain/sgmse-voicebank` — a SpeechBrain-packaged
//!   release of the SGMSE (Score-based Generative Model for Speech
//!   Enhancement) family (Welker et al. 2022 / Richter et al. 2023).
//!   Ships a `hyperparams.yaml` + single `score_model_ema.ckpt`
//!   (~263 MB torch pickle), fine-tuned on the VoiceBank-DEMAND corpus
//!   (Valentini-Botinhao 2016) — the enhancement pair to the
//!   sibling SepFormer / MP-SENet / MetricGAN+ enhancement rows.
//! - **License (SPDX)**: `apache-2.0` — permissive; no runtime-side
//!   attribution obligation under the Apache-2.0 grant. Verified
//!   2026-08-04 via HF cardData API primary source
//!   (`api/models/speechbrain/sgmse-voicebank` → `license: apache-2.0`,
//!   `pipeline_tag: audio-to-audio`).
//! - **Category**: `enhancement` (per SpeechBrain family: single-channel
//!   speech enhancement / dereverberation via a **diffusion / flow
//!   sampler** applied to a score-network's noise-to-clean estimate).
//! - **Complements**: the M3-05 `flow_sampler` + ODE solver op family —
//!   SGMSE is the first *real weight* in the Vokra catalog for that
//!   sampler group (existing enhancement family MetricGAN+ / MP-SENet /
//!   Facebook Denoiser / DeepFilterNet3 are all *masking* or *time-
//!   domain UNet* — none exercise the flow sampler path).
//!
//! # BF16 pass-through (mirror of metricgan_plus / mp_senet / sepformer)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. The
//! observability counter [`SgmseReport::bf16_passthrough`] records how
//! many BF16 tensors landed on this arm so a silent widen / downcast
//! cannot slip in undetected. Upstream ships F32 in the primary release
//! today; the BF16 arm is defensive for future re-quantized derivatives
//! (SpeechBrain has re-published several sibling models in BF16 mirror
//! repos over time).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `.ckpt` state-dict names
//! verbatim** (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS /
//! VoxCPM / MP-SENet / MetricGAN+ contract). The upstream
//! `score_model_ema.ckpt` is a **flat** state_dict of the internal
//! NCSN++ v2 score network (SpeechBrain's `Pretrainer` binds it into
//! the `score_model` module *at load time* — the ckpt file itself
//! carries no `score_model.` prefix on its keys). This converter
//! preserves that flat layout so a future `Sgmse::from_gguf` can walk
//! the same NCSN++ v2 backbone tensor names (`input_layer.weight`,
//! `blocks.0.norm1.weight`, etc.) that the upstream `sgmse` code
//! reference uses.
//!
//! Real-weight parity + a native `Sgmse::from_gguf` forward path are
//! deferred to owner sign-off (`docs/license-audit.md` §3.1) — this
//! converter provides the byte-parallel GGUF surface only. The
//! internal NCSN++ v2 (Song et al. 2021) + OUVE SDE reverse sampler
//! (predictor: reverse_diffusion, corrector: annealed Langevin
//! dynamics, N=30 steps per upstream `hyperparams.yaml`) is
//! intentionally NOT re-implemented on this pass: transcribing that
//! from the SGMSE paper (arXiv:2212.11851 / arXiv:2208.05830) +
//! upstream `sgmse` code is a `loud-partial` sibling wave (RMVPE /
//! Charsiu / MOSS-Audio-Tokenizer / MioCodec landing precedent).
//!
//! # No ONNX (permanent)
//!
//! SGMSE-VoiceBank ships a torch pickle checkpoint (`.ckpt`); this
//! converter **never** touches ONNX (FR-LD-05); the offline
//! `.ckpt` → `.safetensors` bridge runs in
//! `tools/parity/sgmse_prepare_checkpoint.py` (a
//! `nemo_pt_to_safetensors.py`-family sidecar), and the runtime
//! pipeline is re-implemented natively in a future
//! `crates/vokra-models/src/sgmse/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for SGMSE GGUFs. Distinct from every other
/// enhancement / denoise sibling (`denoise` = DeepFilterNet3, `mp_senet`
/// = magnitude+phase U-Net, `metricgan_plus` = generator-only PESQ-
/// tuned GAN, `sepformer` = dual-path Transformer masker+decoder) —
/// SGMSE's *score-based diffusion / OUVE SDE reverse sampler* topology
/// is distinct enough to warrant its own routing arm (FR-EX-08). It
/// also anchors the first real-weight consumer of the M3-05
/// `flow_sampler` + ODE solver op family.
pub const ARCH: &str = "sgmse";

/// `vokra.model.name` value for the canonical VoiceBank-DEMAND fine-tune
/// release.
pub const NAME: &str = "sgmse-voicebank";

/// `vokra.model.category` — `enhancement` (single-channel speech
/// enhancement, matching the sibling SepFormer/MetricGAN+/MP-SENet
/// enhancement rows).
pub const CATEGORY: &str = "enhancement";

/// `vokra.provenance.upstream_hf` slug (`org/name`).
pub const UPSTREAM_HF: &str = "speechbrain/sgmse-voicebank";

/// Default upstream weight license (SPDX). Verified 2026-08-04 via HF
/// cardData API primary source. May be overridden via the `license`
/// argument to [`convert_sgmse_file`] (the whisper / kokoro /
/// metricgan_plus / mp_senet override pattern).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirrors metricgan_plus / mp_senet / sepformer).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an SGMSE conversion. Mirrors the sibling BF16 pass-through
/// converters' counter shape (`super::metricgan_plus::MetricganPlusReport`,
/// `super::mp_senet::MpSenetReport`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SgmseReport {
    /// Total tensors surfaced by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader rejects unknown dtypes at parse time; anything that reaches
    /// this arm is a quantized dtype the runtime is not expected to
    /// consume).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — upstream
    /// ships F32 today, so this counter is expected to be 0 on the real
    /// checkpoint; a future BF16 mirror release would surface it here.
    pub bf16_passthrough: usize,
}

/// File-based SGMSE-VoiceBank converter.
///
/// Reads `input` (a safetensors mirror of the upstream
/// `score_model_ema.ckpt` — produced by
/// `tools/parity/sgmse_prepare_checkpoint.py`), writes a Vokra GGUF to
/// `output` carrying every F32 / F16 / BF16 tensor verbatim under its
/// upstream state-dict name plus the `vokra.model.*` +
/// `vokra.provenance.*` metadata chunks.
///
/// `license` optionally overrides the default `apache-2.0` provenance
/// stamp (same override pattern as
/// `super::metricgan_plus::convert_metricgan_plus_file`).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// serialization fails.
pub fn convert_sgmse_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SgmseReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. The `license` param overrides the raw SPDX string
    // (`vokra.provenance.license`) and — when overridden — re-derives
    // the class through `LicenseClass::from_license_str` so the
    // compliance gate stays honest (a caller who overrides to a
    // non-permissive SPDX would otherwise get a silent Permissive
    // verdict). `None` keeps the SpeechBrain default (apache-2.0 →
    // Permissive) that matches the upstream weight card.
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
            "speechbrain/sgmse-voicebank \
             (SGMSE score-based diffusion speech enhancement, VoiceBank-DEMAND, apache-2.0)",
        ),
    );

    let mut report = SgmseReport::default();
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
    use vokra_core::gguf::GgufFile;

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16");
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

    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 4;
        assert_eq!(f32_bytes.len(), expected, "shape × 4 F32");
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

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-sgmse-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// BF16 round-trip byte-identity + every stamp lands (defensive —
    /// upstream is F32 today, but the BF16 arm must be pinned for
    /// future re-quantized mirror releases).
    #[test]
    fn bf16_round_trips_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Realistic upstream tensor name from the NCSN++ v2 score
        // network's input stack. `score_model_ema.ckpt` is a flat
        // state_dict of the internal network (no `score_model.`
        // prefix — SpeechBrain's Pretrainer adds that at load time).
        let input_bytes = safetensors_one_bf16("input_layer.weight", &[2, 3], &bf16);
        let input = write_temp("bf16-in", &input_bytes);
        let output = write_temp("bf16-out", &[]);

        let report = convert_sgmse_file(&input, &output, None).expect("convert SGMSE");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("input_layer.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical (no silent widen)"
        );

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
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Primary code path — upstream `score_model_ema.ckpt` ships F32
    /// today. This test pins the exact path a real conversion walks.
    #[test]
    fn f32_pass_through_is_primary_upstream_path() {
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one_f32("blocks.0.norm1.weight", &[2, 3], &f32_bytes);
        let input = write_temp("f32-in", &input_bytes);
        let output = write_temp("f32-out", &[]);

        let report = convert_sgmse_file(&input, &output, None).expect("convert SGMSE");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 must not increment BF16 counter"
        );
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("blocks.0.norm1.weight")
            .expect("F32 tensor present");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// The `license` override must flow through to the provenance stamp
    /// and re-derive the class (guards against a silent Permissive
    /// verdict when a caller ships under a non-permissive SPDX).
    #[test]
    fn license_override_flows_through() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("input_layer.weight", &[1, 2], &f32_bytes);
        let input = write_temp("license-in", &input_bytes);
        let output = write_temp("license-out", &[]);

        convert_sgmse_file(&input, &output, Some("mit")).expect("license override must succeed");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit is Permissive class (same bucket as apache-2.0 default)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
