//! **Meta MelodyFlow T24 30secs** (`facebook/melodyflow-t24-30secs`,
//! **cc-by-nc-4.0**): safetensors → GGUF conversion
//! (coverage-audit-2026-08-03 Wave D T4, post-audit CC-gap
//! 2026-08-13, Wave D remaining WF8).
//!
//! Meta AudioCraft's **flow-matching music editing** model —
//! DiT-style backbone with **24 timesteps** and a **30 second**
//! max horizon at **48 kHz**, distributed on HuggingFace at
//! `huggingface.co/facebook/melodyflow-t24-30secs`. Primary
//! use-case = **text-conditioned music editing** (an existing audio
//! clip is inverted through the ODE, then regenerated under a new
//! text prompt), which is a distinct code path from text-to-music
//! sibling releases (MusicGen AR-over-EnCodec / MAGNeT non-AR
//! masked-LM / JASCO joint audio-symbolic conditioning). Le Lan et
//! al. 2024 arXiv:2407.03648 "MelodyFlow: High-Fidelity Music
//! Generation and Editing with Rectified Flow Matching".
//!
//! Weight licence is **CC-BY-NC-4.0** (research-only, T4 tier —
//! MusicGen family / X-Codec-2 / jasco_400m_chords_drums / sibling
//! `magnet_small_10secs` / `magnet_medium_30secs` precedent), so
//! publish requires `--allow-noncommercial` and the runtime M2-13
//! gate refuses commercial-mode load.
//!
//! # Distinct arch tag (FR-EX-08 dispatch boundary)
//!
//! MelodyFlow is a **flow-matching / DiT** model — the runtime
//! forward path is an ODE integrator (`vokra_ops::flow_sampler`
//! from M3-05, Euler / Sway schedule) over a Diffusion Transformer
//! backbone, with dual text + audio prefix conditioning for the
//! editing use-case. This is fundamentally different from every
//! sibling music-generation family:
//!
//! - **MAGNeT** (`magnet_small_10secs` / `magnet_medium_30secs`) —
//!   non-autoregressive parallel masked-LM decoding with a
//!   confidence-based span masking schedule (Ziv et al. 2024).
//!   Different decoder loop, different sampler.
//! - **MusicGen** family (`musicgen_small` / `musicgen_medium` /
//!   `musicgen_large` / `audiogen_medium`) — autoregressive
//!   token-by-token generation over EnCodec residual-VQ codes.
//!   Different decoder loop entirely.
//! - **JASCO** (`jasco_400m_chords_drums`) — flow-matching with
//!   **temporal symbolic** conditioning (chord progression + drum
//!   tracks) rather than the dual text + audio prefix that
//!   MelodyFlow uses for editing. Same op family (flow-matching /
//!   ODE), different conditioning stack.
//! - **AudioLDM2** — latent-diffusion U-Net (score-based rather
//!   than rectified flow), different sampler.
//! - **Stable Audio Open** — DiT + audio VAE, different
//!   conditioning.
//! - **ACE-Step** — separate music-gen family.
//! - **BS-RoFormer** — music-source separation, entirely different
//!   task.
//!
//! Silently sharing an arch tag with any of these would mis-route
//! runtime dispatch. This converter therefore stamps a distinct
//! `vokra.model.arch = "melodyflow_t24_30secs"` so a future runtime
//! binder cannot accidentally silently share any sibling loader.
//!
//! # BF16 pass-through (mirror of magnet_medium_30secs /
//! # magnet_small_10secs / jasco_400m_chords_drums / musicgen_medium /
//! # audiogen_medium / xcodec2 / stable_audio_open_small)
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
//! (sibling to magnet_medium_30secs / magnet_small_10secs /
//! musicgen_medium / audiogen_medium / audioldm2 /
//! jasco_400m_chords_drums — the runtime `vokra-models::melodyflow`
//! future binder can rely on the upstream key set without a rename
//! layer). Runtime `flow_editing_inversion` +
//! `t24_transformer` ops are the **FR-OP-86 anchor** for a follow-up
//! wave (owner ADR judgement; runtime binder deferred per RMVPE /
//! Charsiu / MOSS-Audio-Tokenizer / MioCodec / sibling MAGNeT
//! loud-partial precedent). The core DiT forward can reuse
//! `vokra_ops::flow_sampler` from M3-05, but the editing-specific
//! ODE inversion path and the 48 kHz RVQ codec bundle need explicit
//! binder decisions.
//!
//! # No hparam bake (deferred to runtime binder, sibling MAGNeT
//! # precedent)
//!
//! The published `n_timesteps=24` / `max_duration=30s` /
//! `sample_rate=48000` hparams are transcribed in `NAME` /
//! `docs/license-audit.md` §3.1 / this docstring for provenance,
//! but **not baked into GGUF metadata** in this land. Rationale
//! mirrors the sibling MAGNeT posture (magnet_small_10secs /
//! magnet_medium_30secs): hparams are runtime-forward concerns
//! that a future binder wave will consume, and premature
//! `vokra.melodyflow.*` chunk keys would force a rename cycle when
//! the binder ADR pins the final schema. The converter deliberately
//! stays pass-through-only.
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

/// `vokra.model.arch` value for MelodyFlow T24 30secs GGUFs. Distinct
/// from every sibling music-generation family — silently sharing an
/// arch tag with `magnet_small_10secs` / `magnet_medium_30secs`
/// (Meta AudioCraft MAGNeT non-autoregressive masked-LM parallel
/// decoding, different sampler), `musicgen_small` / `musicgen_medium`
/// / `musicgen_large` (Meta AudioCraft **AR** over EnCodec —
/// token-by-token autoregressive generation, entirely different
/// decoder loop), `audiogen_medium` (AR SFX / environmental sound
/// sibling of MusicGen), `audioldm2` (latent diffusion U-Net,
/// score-based rather than rectified flow),
/// `stable_audio_open_small` (DiT + VAE with different
/// conditioning stack), `jasco_400m_chords_drums` (flow-matching
/// with **temporal symbolic** conditioning, same op family but
/// different conditioning stack from MelodyFlow's dual text +
/// audio prefix for editing), `ace_step`, `yue_bundle`, or
/// `bs_roformer` (music-source separation) would mis-route the
/// runtime dispatch. MelodyFlow's flow-matching / DiT editing
/// stack (with the ODE inversion path for existing-audio editing)
/// is a distinct topology.
pub const ARCH: &str = "melodyflow_t24_30secs";

/// `vokra.model.name` value written for the canonical
/// `facebook/melodyflow-t24-30secs` release.
pub const NAME: &str = "melodyflow_t24_30secs";

/// `vokra.model.category` value written for every MelodyFlow T24
/// 30secs GGUF. Shared with the sibling MusicGen / AudioGen / JASCO
/// / MAGNeT family (`music` — 2026-07-30 scope expansion
/// `[[project-scope-expansion-2026-07-30]]`).
pub const CATEGORY: &str = "music";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`. Verified against
/// `huggingface.co/facebook/melodyflow-t24-30secs`.
pub const UPSTREAM_HF: &str = "facebook/melodyflow-t24-30secs";

/// Default upstream weight licence (SPDX). Verified against the
/// upstream HF card — CC-BY-NC-4.0 (research-only, non-commercial,
/// T4 tier per MusicGen family / X-Codec-2 / sibling
/// `magnet_small_10secs` / `magnet_medium_30secs` /
/// `jasco_400m_chords_drums` precedent).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / musicgen_medium / funcodec /
/// jasco_400m_chords_drums / magnet_small_10secs /
/// magnet_medium_30secs convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MelodyFlow T24 30secs conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MelodyflowT2430secsReport {
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

/// Converts a MelodyFlow T24 30secs safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`, returning a
/// [`MelodyflowT2430secsReport`].
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
pub fn convert_melodyflow_t24_30secs_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MelodyflowT2430secsReport, ConvertError> {
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
            "facebook/melodyflow-t24-30secs (Meta AudioCraft flow-matching \
             music editing — DiT backbone with 24 timesteps, 30 sec max horizon \
             at 48 kHz, dual text + audio prefix conditioning for the editing \
             use-case, Le Lan et al. 2024 arXiv:2407.03648, CC-BY-NC-4.0 — \
             owner §3.1 sign-off required, publish requires --allow-noncommercial \
             per MusicGen family / sibling magnet_small_10secs / \
             magnet_medium_30secs / jasco_400m_chords_drums T4 precedent)",
        ),
    );

    let mut report = MelodyflowT2430secsReport::default();
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
            "vokra-melodyflow-t24-30secs-{tag}-{}-{}.{ext}",
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
        let header = r#"{"transformer.blocks.0.attn.qkv.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_melodyflow_t24_30secs_file(&input_path, &output_path, None)
            .expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("transformer.blocks.0.attn.qkv.weight")
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
            r#"{{"transformer.blocks.0.attn.qkv.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"transformer.norm.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_melodyflow_t24_30secs_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("transformer.blocks.0.attn.qkv.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        let f16_info = file
            .tensor_info("transformer.norm.bias")
            .expect("F16 tensor");
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
        let header = r#"{"transformer.blocks.0.attn.qkv.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
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
            convert_melodyflow_t24_30secs_file(&input_path, &output_path, Some("apache-2.0"))
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
    fn arch_and_name_are_stable_constants_distinct_from_music_gen_family() {
        // Guards the FR-EX-08 dispatch boundary along three axes:
        //
        //   1. MelodyFlow (flow-matching / DiT with editing ODE
        //      inversion path) vs MAGNeT family (magnet_small_10secs /
        //      magnet_medium_30secs, non-autoregressive masked-LM
        //      parallel decoding) — entirely different decoder /
        //      sampler stack, silently sharing arch would mis-route
        //      dispatch (Euler / Sway ODE integrator vs
        //      confidence-based span masking scheduler).
        //   2. MelodyFlow (flow-matching) vs MusicGen family
        //      (musicgen_small / musicgen_medium / musicgen_large,
        //      AR over EnCodec) — entirely different decoder loop,
        //      silently sharing arch would mis-route to token-by-token
        //      AR path.
        //   3. MelodyFlow (dual text + audio prefix conditioning for
        //      editing) vs JASCO (jasco_400m_chords_drums, joint
        //      audio-symbolic conditioning with chord progression +
        //      drum tracks) — same op family (flow-matching) but
        //      different conditioning stack, silently sharing arch
        //      would let the runtime binder load JASCO conditioning
        //      into MelodyFlow's editing path.
        //
        // A future rename of this constant would break runtime dispatch
        // downstream — this pin makes such a change require an explicit
        // ADR / test update.
        assert_eq!(ARCH, "melodyflow_t24_30secs");
        assert_eq!(NAME, "melodyflow_t24_30secs");
        assert_ne!(
            ARCH, "magnet_small_10secs",
            "silently sharing arch tag with MAGNeT small sibling is a FR-EX-08 violation \
             — MAGNeT is non-autoregressive masked-LM, MelodyFlow is flow-matching / DiT"
        );
        assert_ne!(
            ARCH, "magnet_medium_30secs",
            "silently sharing arch tag with MAGNeT medium sibling is a FR-EX-08 violation \
             — same family split as sibling small"
        );
        assert_ne!(
            ARCH, "musicgen",
            "silently sharing arch tag with MusicGen AR family is a FR-EX-08 violation \
             — MusicGen is AR over EnCodec, MelodyFlow is flow-matching / DiT"
        );
        assert_ne!(
            ARCH, "jasco_400m_chords_drums",
            "sibling JASCO joint audio-symbolic conditioning is a distinct conditioning \
             stack even though op family (flow-matching) is shared"
        );
        assert_ne!(
            ARCH, "audioldm2",
            "sibling latent-diffusion audio family is a distinct topology \
             (score-based U-Net vs rectified flow DiT)"
        );
        assert_ne!(
            ARCH, "stable_audio_open_small",
            "sibling Stable Audio Open DiT is a distinct conditioning stack"
        );
        assert_ne!(
            ARCH, "ace_step",
            "sibling ACE-Step is a separate music-gen family"
        );
    }

    #[test]
    fn upstream_and_category_pin_expected_values() {
        // Pin CATEGORY = "music" (shared sibling class) and UPSTREAM_HF =
        // canonical HF slug so a scanner rename cannot silently drift the
        // catalog-reality gate away from the license-audit §3.1 row.
        assert_eq!(CATEGORY, "music");
        assert_eq!(UPSTREAM_HF, "facebook/melodyflow-t24-30secs");
        assert_eq!(DEFAULT_LICENSE_SPDX, "cc-by-nc-4.0");
    }

    #[test]
    fn upstream_slug_stays_under_facebook_melodyflow_family() {
        // Guard the catalog-reality / signoff_match / license-audit
        // §3.1 boundary: a future refactor that unifies MelodyFlow
        // variants under a common `MELODYFLOW_UPSTREAM_HF` (if a
        // sibling `melodyflow-t12-30secs` or `melodyflow-t48-30secs`
        // ever lands) must NOT collapse the T24 variant into a shared
        // slug — the HF repos are distinct and publish-one.sh would
        // reject or mis-target a merged constant.
        assert_ne!(
            UPSTREAM_HF, "facebook/magnet-small-10secs",
            "MelodyFlow (flow-matching editing) and MAGNeT (masked-LM) are \
             distinct HF repositories in different families"
        );
        assert_ne!(
            UPSTREAM_HF, "facebook/magnet-medium-30secs",
            "MelodyFlow (flow-matching editing) and MAGNeT medium (masked-LM) \
             are distinct HF repositories in different families"
        );
        assert_ne!(
            UPSTREAM_HF, "facebook/jasco-chords-drums-400M",
            "MelodyFlow (dual text + audio editing conditioning) and JASCO \
             (joint audio-symbolic chord/drum conditioning) are distinct \
             HF repositories even within the flow-matching op family"
        );
        assert!(
            UPSTREAM_HF.starts_with("facebook/melodyflow-"),
            "upstream slug must remain under facebook/melodyflow-* to preserve \
             the family provenance stamp shape"
        );
        assert!(
            UPSTREAM_HF.contains("-t24-"),
            "T24 variant marker must survive rename cycles so a future \
             melodyflow-t12-30secs / melodyflow-t48-30secs cannot silently \
             collide with the 24-timestep variant"
        );
    }

    #[test]
    fn default_license_class_matches_x_codec_2_precedent() {
        // Pin the LicenseClass hard-map: cc-by-nc-4.0 → NonCommercial.
        // This mirrors the X-Codec-2 T4 first-precedent (2026-07-28 land)
        // and the sibling MAGNeT / JASCO T4 rows. A regression in
        // `LicenseClass::from_license_str` that reclassifies NC-4.0
        // as Permissive or Unknown would silently unlock publish for
        // T4 tier weights — this pin makes such a regression fail HERE
        // rather than at publish-time.
        assert_eq!(
            LicenseClass::from_license_str("cc-by-nc-4.0"),
            LicenseClass::NonCommercial,
            "cc-by-nc-4.0 must resolve to NonCommercial (T4 fail-closed) — \
             any drift here silently unlocks publish for T4 tier weights"
        );
        assert_eq!(
            LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
            LicenseClass::NonCommercial,
            "DEFAULT_LICENSE_SPDX must resolve to NonCommercial via \
             from_license_str — pin the round-trip through the resolver \
             so a future const rename cannot bypass the hard-map"
        );
    }
}
