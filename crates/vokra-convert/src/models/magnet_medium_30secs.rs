//! **Meta MAGNeT Medium 30secs** (`facebook/magnet-medium-30secs`,
//! **cc-by-nc-4.0**): safetensors → GGUF conversion
//! (coverage-audit-2026-08-03 Wave D T4, post-audit CC-gap
//! 2026-08-13, Wave D remaining WF7).
//!
//! Meta AudioCraft's Masked Audio Generation using a Single
//! Non-Autoregressive Transformer — **1.5B parameter medium**
//! variant that generates **30-second music clips** via parallel
//! masked-LM decoding (Ziv et al. 2024 arXiv:2401.04577 "Masked
//! Audio Generation using a Single Non-Autoregressive Transformer"),
//! distributed on HuggingFace at
//! `huggingface.co/facebook/magnet-medium-30secs`. Sibling of the
//! smaller `magnet-small-10secs` (500 M params, 10 sec clips) — same
//! non-autoregressive topology + confidence-based span masking
//! schedule, with expanded hidden width / layer count / max span
//! (30 sec generation slot matching MusicGen family's max, ~7x
//! faster than AR baselines at that horizon).
//!
//! Weight licence is **CC-BY-NC-4.0** (research-only, T4 tier —
//! MusicGen family / X-Codec-2 / jasco_400m_chords_drums / sibling
//! `magnet_small_10secs` precedent), so publish requires
//! `--allow-noncommercial` and the runtime M2-13 gate refuses
//! commercial-mode load.
//!
//! # Distinct arch tag (FR-EX-08 dispatch boundary)
//!
//! MAGNeT is **non-autoregressive** (parallel masked-LM decoding
//! with a confidence-based span masking schedule, ~7x faster than
//! AR baselines). This variant is **also distinct from
//! `magnet_small_10secs`**: silently sharing an arch tag with the
//! sibling small (`magnet_small_10secs`) would mis-route runtime
//! dispatch — the small variant fits a 10-second generation slot
//! with a narrower hidden width, while medium widens to fit the
//! 30-second horizon (span_len / max codebook stream position span
//! differ, so the masked-decode + span-masking schedule primitive
//! parameters change even though the op path is the same). And
//! silently sharing with the AR sibling family
//! (`musicgen_small` / `musicgen_medium` / `musicgen_large` /
//! `audiogen_medium`) or the joint-symbolic sibling
//! (`jasco_400m_chords_drums`) or the latent-diffusion
//! (`audioldm2` / `stable_audio_open_small`) or DiT
//! (`ace_step`) or music-source-separation (`bs_roformer`)
//! releases mis-routes at a coarser level — different decoder
//! topology entirely. This converter therefore stamps a distinct
//! `vokra.model.arch = "magnet_medium_30secs"` so a future runtime
//! binder cannot accidentally silently share **either** the
//! sibling small MAGNeT loader **or** the MusicGen AR loader.
//!
//! # BF16 pass-through (mirror of magnet_small_10secs /
//! # jasco_400m_chords_drums / musicgen_medium / audiogen_medium /
//! # xcodec2 / stable_audio_open_small)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30
//! ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream HF safetensors keys verbatim**
//! (sibling to magnet_small_10secs / musicgen_medium / audiogen_medium /
//! audioldm2 / jasco_400m_chords_drums — the runtime `vokra-models::magnet`
//! future binder can rely on the upstream key set without a rename
//! layer). Runtime `magnet_masked_decode` + `span_masking_scheduler`
//! ops are the FR-OP-85 anchor for a follow-up wave (owner ADR
//! judgement; runtime binder deferred per RMVPE / Charsiu /
//! MOSS-Audio-Tokenizer / MioCodec / sibling `magnet_small_10secs`
//! loud-partial precedent).
//!
//! # No ONNX (permanent)
//!
//! The upstream release ships safetensors + torch pickle; this
//! converter accepts safetensors only (never touches ONNX,
//! FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for MAGNeT Medium 30secs GGUFs. Distinct
/// from every sibling music-generation family — silently sharing an
/// arch tag with `magnet_small_10secs` (Meta AudioCraft MAGNeT
/// **10-second** sibling with narrower hidden width — same
/// non-autoregressive masked-LM decoding op path but different span
/// / hidden / layer count hyperparameters), `musicgen_medium` /
/// `musicgen_large` / `musicgen_small` (Meta AudioCraft **AR** over
/// EnCodec — token-by-token autoregressive generation, entirely
/// different decoder loop), `audiogen_medium` (AR SFX /
/// environmental sound sibling of MusicGen), `audioldm2` (latent
/// diffusion), `stable_audio_open_small` (DiT + VAE),
/// `jasco_400m_chords_drums` (joint audio-symbolic conditioning),
/// `ace_step`, `yue_bundle`, or `bs_roformer` (music-source
/// separation) would mis-route the runtime dispatch. MAGNeT's
/// non-autoregressive masked-LM parallel decoding stack is a
/// distinct topology, and the medium variant needs its own hparam
/// set (30 sec span, wider hidden, more layers) that a shared arch
/// tag would obscure.
pub const ARCH: &str = "magnet_medium_30secs";

/// `vokra.model.name` value written for the canonical
/// `facebook/magnet-medium-30secs` release.
pub const NAME: &str = "magnet_medium_30secs";

/// `vokra.model.category` value written for every MAGNeT Medium 30secs
/// GGUF. Shared with the sibling MusicGen / AudioGen / JASCO /
/// magnet-small family (`music` — 2026-07-30 scope expansion
/// `[[project-scope-expansion-2026-07-30]]`).
pub const CATEGORY: &str = "music";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`. Verified against
/// `huggingface.co/facebook/magnet-medium-30secs`.
pub const UPSTREAM_HF: &str = "facebook/magnet-medium-30secs";

/// Default upstream weight licence (SPDX). Verified against the
/// upstream HF card — CC-BY-NC-4.0 (research-only, non-commercial,
/// T4 tier per MusicGen family / X-Codec-2 / sibling
/// `magnet_small_10secs` precedent).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / musicgen_medium / funcodec / jasco_400m_chords_drums
/// / magnet_small_10secs convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MAGNeT Medium 30secs conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MagnetMedium30secsReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter).
    pub bf16_passthrough: usize,
}

/// Converts a MAGNeT Medium 30secs safetensors checkpoint at `input`
/// into a Vokra-native GGUF at `output`, returning a
/// [`MagnetMedium30secsReport`].
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-4.0"`)
/// which resolves to [`LicenseClass::NonCommercial`] (T4 fail-closed).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_magnet_medium_30secs_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MagnetMedium30secsReport, ConvertError> {
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
            "facebook/magnet-medium-30secs (Meta AudioCraft Masked Audio Generation, \
             non-autoregressive masked-LM parallel decoding for 30-second music \
             generation — 1.5B medium variant, sibling of magnet-small-10secs, \
             Ziv et al. 2024 arXiv:2401.04577, CC-BY-NC-4.0 — owner §3.1 sign-off \
             required, publish requires --allow-noncommercial per MusicGen family / \
             sibling magnet_small_10secs T4 precedent)",
        ),
    );

    let mut report = MagnetMedium30secsReport::default();
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
            "vokra-magnet-medium-30secs-{tag}-{}-{}.{ext}",
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

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        let header = r#"{"lm.emb.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_magnet_medium_30secs_file(&input_path, &output_path, None)
            .expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("lm.emb.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_fail_closed() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"lm.transformer.layers.0.self_attn.in_proj.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"lm.linears.0.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_magnet_medium_30secs_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("lm.transformer.layers.0.self_attn.in_proj.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        let f16_info = file.tensor_info("lm.linears.0.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);

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
            Some(LicenseClass::NonCommercial.as_str()),
            "cc-by-nc-4.0 must resolve to NonCommercial (T4 fail-closed)"
        );
    }

    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"lm.emb.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_magnet_medium_30secs_file(&input_path, &output_path, Some("apache-2.0"))
                .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 reclassifies away from the NonCommercial default"
        );
    }

    #[test]
    fn arch_and_name_are_stable_constants_distinct_from_magnet_small_and_musicgen_family() {
        // Guards the FR-EX-08 dispatch boundary along two axes:
        //
        //   1. medium vs sibling small MAGNeT — same non-autoregressive
        //      masked-LM decoding op path but different hyperparameters
        //      (30 sec span vs 10 sec, wider hidden, more layers), so a
        //      shared arch tag would obscure the hparam difference and
        //      let the runtime binder silently load small hparams into
        //      medium weights.
        //   2. MAGNeT (non-AR masked-LM) vs sibling MusicGen (AR over
        //      EnCodec) — entirely different decoder loop, silently
        //      sharing arch would mis-route to token-by-token AR path.
        //
        // A future rename of this constant would break runtime dispatch
        // downstream — this pin makes such a change require an explicit
        // ADR / test update.
        assert_eq!(ARCH, "magnet_medium_30secs");
        assert_eq!(NAME, "magnet_medium_30secs");
        assert_ne!(
            ARCH, "magnet_small_10secs",
            "silently sharing arch tag with sibling MAGNeT small (different span/hidden/layers) \
             is a FR-EX-08 violation — hparams differ even though op path is shared"
        );
        assert_ne!(
            ARCH, "musicgen",
            "silently sharing arch tag with MusicGen AR family is FR-EX-08 violation"
        );
        assert_ne!(
            ARCH, "jasco_400m_chords_drums",
            "sibling JASCO joint conditioning is a distinct topology"
        );
        assert_ne!(
            ARCH, "audioldm2",
            "sibling latent-diffusion audio family is a distinct topology"
        );
    }

    #[test]
    fn upstream_and_category_pin_expected_values() {
        // Pin CATEGORY = "music" (shared sibling class) and UPSTREAM_HF =
        // canonical HF slug so a scanner rename cannot silently drift the
        // catalog-reality gate away from the license-audit §3.1 row.
        assert_eq!(CATEGORY, "music");
        assert_eq!(UPSTREAM_HF, "facebook/magnet-medium-30secs");
        assert_eq!(DEFAULT_LICENSE_SPDX, "cc-by-nc-4.0");
    }

    #[test]
    fn medium_variant_stays_distinct_from_small_upstream_slug() {
        // Guard the catalog-reality / signoff_match / license-audit
        // §3.1 boundary: a future refactor that unifies the small and
        // medium constants under a common `MAGNET_UPSTREAM_HF` would
        // silently collapse two distinct HF repos into one and let
        // publish-one.sh reject or mis-target the second. Pin the
        // canonical slugs to prevent that collision.
        assert_ne!(
            UPSTREAM_HF, "facebook/magnet-small-10secs",
            "medium (30 sec) and small (10 sec) are distinct HF repositories"
        );
        assert!(
            UPSTREAM_HF.starts_with("facebook/magnet-"),
            "upstream slug must remain under facebook/magnet-* to preserve \
             the family provenance stamp shape"
        );
        assert!(
            UPSTREAM_HF.ends_with("-30secs"),
            "upstream slug must carry the -30secs horizon suffix (30 sec \
             generation slot, MusicGen-family-max, ~7x faster than AR)"
        );
    }
}
