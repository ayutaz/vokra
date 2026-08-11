//! **MossFormer2-SS-16K** (`alibabasglab/MossFormer2_SS_16K`,
//! Apache-2.0): safetensors → GGUF conversion (coverage-audit-2026-08-03
//! Wave A permissive continuation, 2026-08-04).
//!
//! Input: the upstream Alibaba SGLab MossFormer2 speech-separation
//! release from the ClearerVoice-Studio project (Zhao et al. 2024
//! Interspeech, "MossFormer2: Combining Transformer and RNN-Free
//! Recurrent Network for Enhanced Time-Domain Monaural Speech
//! Separation"). Cocktail-party / multi-speaker speech separator at
//! 16 kHz — WSJ-2mix / Libri2Mix SoTA-tier — targeting a distinctly
//! **speech** (not music) separation topology that composes FSMN
//! (Feed-forward Sequential Memory Network) blocks with gated
//! attention. The upstream release is distributed on HF; callers
//! pre-flatten the torch checkpoint (Lightning `.ckpt` or bare
//! `state_dict.pt`) to safetensors offline via
//! `tools/parity/mossformer2_ss_prepare_checkpoint.py` (the DFN3 /
//! DAC / CSM pickle-bridge pattern — no pickle enters the runtime,
//! FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime source-
//! separation path binds against.
//!
//! # License
//!
//! - SPDX: **Apache-2.0** ([`vokra_core::LicenseClass::Permissive`]).
//!   Verified against the ClearerVoice-Studio upstream LICENSE at
//!   `github.com/modelscope/ClearerVoice-Studio`.
//! - Category: **source-separation** (speech source separation —
//!   sibling of `htdemucs_multi` (music) and `sepformer` under the
//!   shared source-separation umbrella covering both speech and music
//!   separation families).
//! - Notes: **Not related to** the sibling `MOSS-Audio-Tokenizer` /
//!   `MOSS-TTS` models in the tree — the "MOSS" naming collision
//!   is a distinct project trees; FR-EX-08 forbids silent shape
//!   misroute across them, and this converter carries the
//!   `mossformer2_ss_16k` arch tag to make that boundary explicit.
//!
//! # BF16 pass-through (mirror of sensevoicesmall / neucodec /
//! # ecapa_tdnn / speaker_3d)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream ClearerVoice-Studio state-dict
//! keys verbatim** (`separator.*` / `encoder.*` / `decoder.*` per the
//! MossFormer2 class layout). Real-weight parity binding to a future
//! `vokra-models::mossformer2_ss` runtime module (native FSMN block +
//! gated attention forward) is deferred to owner sign-off per
//! `docs/license-audit.md §3.1`. The FSMN block kernel may be shareable
//! with the existing `FsmnVad` (2026-07-30 land) runtime binder —
//! op consolidation vs new dedicated `fsmn_block` op is a follow-up
//! ADR question.
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "mossformer2_ss_16k"` is intentionally distinct
//! from every sibling source-separation arch tag (`sepformer` /
//! `demucs_htdemucs` / `htdemucs_multi` / `bs_roformer` /
//! `tiger_separator` / `mp_senet` / `conv_tasnet`) and from the
//! sibling `moss_tts` / `moss_audio_tokenizer` (Tsinghua MOSS
//! project — unrelated naming collision) and `fsmn_vad` (FunASR VAD
//! — related FSMN block but different task head). Silently sharing
//! an arch tag with any of these would mis-route the runtime
//! dispatch.
//!
//! # No ONNX (permanent)
//!
//! The upstream MossFormer2 release ships PyTorch pickle files
//! (Lightning `.ckpt` / `state_dict.pt`); this converter **never**
//! touches ONNX (FR-LD-05).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native FSMN block +
//! gated attention forward is a follow-up wave, deferred to owner
//! sign-off (see `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for MossFormer2-SS-16K GGUFs. Intentionally
/// distinct from every sibling source-separation arch tag
/// (`sepformer` / `demucs_htdemucs` / `htdemucs_multi` /
/// `bs_roformer` / `tiger_separator` / `mp_senet` / `conv_tasnet`)
/// and from the unrelated Tsinghua-MOSS-project `moss_tts` /
/// `moss_audio_tokenizer` naming collision and from the related
/// FSMN-block `fsmn_vad` (different task head). Silently sharing an
/// arch tag with any of these would mis-route the runtime dispatch.
pub const ARCH: &str = "mossformer2_ss_16k";

/// `vokra.model.name` value written for the canonical
/// `alibabasglab/MossFormer2_SS_16K` release.
pub const NAME: &str = "mossformer2_ss_16k";

/// `vokra.model.category` value written for every MossFormer2-SS-16K
/// GGUF. Sibling of `htdemucs_multi` / `sepformer` / `bs_roformer` /
/// `mp_senet` / `conv_tasnet` (`source-separation` umbrella).
pub const CATEGORY: &str = "source-separation";

/// Upstream HF repository slug (`org/name`).
pub const UPSTREAM_HF: &str = "alibabasglab/MossFormer2_SS_16K";

/// Default upstream weight licence (SPDX). Verified against the
/// upstream ClearerVoice-Studio project LICENSE (Apache-2.0).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key. Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MossFormer2-SS-16K conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Mossformer2Ss16kReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16.
    pub bf16_passthrough: usize,
}

/// Converts a MossFormer2-SS-16K safetensors checkpoint at `input`
/// (pre-flattened from the upstream ClearerVoice-Studio Lightning
/// `.ckpt` / `state_dict.pt` by
/// `tools/parity/mossformer2_ss_prepare_checkpoint.py`) into a
/// Vokra-native GGUF at `output`, returning a
/// [`Mossformer2Ss16kReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` chunks are stamped for the runtime compliance
/// gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license. The
/// default is `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_mossformer2_ss_16k_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Mossformer2Ss16kReport, ConvertError> {
    // Load the whole checkpoint into memory — ~200 MB (below the
    // streaming-mandated Moshi 14 GiB tier).
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "alibabasglab/MossFormer2_SS_16K (Alibaba SGLab cocktail-party / multi-speaker \
             speech separator at 16 kHz, FSMN + gated-attention topology, ClearerVoice-Studio \
             project, Zhao et al. 2024 Interspeech, Apache-2.0)",
        ),
    );

    let mut report = Mossformer2Ss16kReport::default();
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
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-mossformer2-ss-16k-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Pins the BF16 pass-through end-to-end.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12);
        // MossFormer2 separator gated-attention linear weight — the
        // upstream ClearerVoice-Studio state-dict key convention
        // preserved verbatim through the
        // `mossformer2_ss_prepare_checkpoint.py` bridge.
        let header = r#"{"separator.layers.0.gating.linear.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_mossformer2_ss_16k_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("separator.layers.0.gating.linear.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    /// Pins F32 and F16 pass-through. Apache-2.0 default → Permissive.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = format!(
            r#"{{"encoder.conv.0.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"decoder.output.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);

        let input_path = scratch_path("mixed-in", "safetensors");
        let output_path = scratch_path("mixed-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_mossformer2_ss_16k_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("encoder.conv.0.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("decoder.output.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

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
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 must resolve to Permissive (T1 tier)"
        );
    }

    /// Pins the license override boundary.
    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_mossformer2_ss_16k_file(&input_path, &output_path, Some("mit"))
            .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
    }
}
