//! **Audiobox Aesthetics** (`facebook/audiobox-aesthetics`, cc-by-4.0):
//! safetensors checkpoint → GGUF conversion (TIER 2 land, 2026-07-30).
//!
//! Input: the upstream `facebook/audiobox-aesthetics` release —
//! `model.safetensors` (~104M F32 params, arXiv:2502.05139 — "Meta
//! Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech,
//! Music, and Sound"). Output: a GGUF carrying every float tensor
//! verbatim under its upstream safetensors name, plus the
//! `vokra.audiobox_aesthetics.*` / `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks the native
//! `vokra-models::audiobox_aesthetics::*` implementation reads.
//!
//! # What the model produces (four audio-quality ratings)
//!
//! Audiobox Aesthetics is an **audio-classification** head: given an
//! audio clip, it emits four real-valued ratings: CONTENT_ENJOYMENT
//! (CE), CONTENT_USEFULNESS (CU), PRODUCTION_COMPLEXITY (PC), and
//! PRODUCTION_QUALITY (PQ).  The upstream `AesMultiOutput.AXES_NAME`
//! list at source revision [`SOURCE_REVISION`] is exactly those four
//! values; there is no learned `BALANCED` output. Per the immutable HF
//! [`CHECKPOINT_REVISION`] `config.json`:
//!
//! - Backbone: WavLM SSL encoder (weighted-layer-sum over the
//!   `nth_layer` = 13 encoder outputs, `use_weighted_layer_sum: true`).
//! - Head: 5-layer projection MLP (`proj_num_layer: 5`, `proj_act_fn:
//!   gelu`, `proj_dropout: 0.0`, `proj_ln: true`) producing an
//!   `output_dim: 1` scalar per axis (per-axis heads share the backbone).
//! - Precision: `"32"` (F32 tensors upstream — the safetensors payload
//!   is F32 verbatim, ~104M × 4 B ≈ 415 MB on disk).
//! - Target normalisation: per-axis `{mean, std}` transform recorded in
//!   `config.json.target_transform` (CE: μ 5.06865 σ 1.93029, CU:
//!   μ 5.73633 σ 1.75669, PC: μ 3.18591 σ 1.86637, PQ: μ 6.57505
//!   σ 1.51466).
//! - Embed normalisation: `normalize_embed: true`.
//!
//! # HF / licence / category (primary-source verified 2026-07-30)
//!
//! - Upstream HF: `facebook/audiobox-aesthetics` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - HF cardData `license: cc-by-4.0` — `LicenseClass::AttributionRequired`
//!   (`docs/license-audit.md` §3.1 Facebook / Audiobox row).
//!   The M2-13 gate passes commercially *and* the FR-MD-09 attribution
//!   surface activates (Meta / Facebook AI Research attribution, mirror
//!   of the Kyutai `AttributionRequired` templates).
//! - Model category: `classification` (**first Vokra converter with this
//!   category** — audiobox-aesthetics is an audio-quality regression
//!   head, distinct from ASR / TTS / codec / speaker / emotion / s2s /
//!   tts / bert; silently sharing an existing category would misroute a
//!   downstream catalog consumer that ranks-by-category).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! Conversion is variant-closed: the immutable public checkpoint has
//! exactly [`TENSOR_COUNT`] F32 tensors, and every name and shape is
//! checked before a byte is written. A missing, renamed, additional, or
//! differently-shaped tensor is a loud parse error rather than a
//! best-effort pass-through.
//!
//! # Real-weight parity
//!
//! Real-weight parity uses the upstream package at [`SOURCE_REVISION`]
//! as an independent oracle; see
//! `tools/parity/audiobox_aesthetics_dump_reference.py` and
//! `docs/handoff/parity-audiobox-aesthetics-real.md`.
//!
//! # No ONNX (permanent)
//!
//! Audiobox-Aesthetics is distributed as safetensors + a Python
//! pipeline; this converter **never** touches ONNX (FR-LD-05).
//! The pipeline is re-implemented natively in the
//! `crates/vokra-models/src/audiobox_aesthetics/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Audiobox-Aesthetics GGUFs. Distinct from
/// every sibling arch tag — audiobox-aesthetics is a WavLM-backboned
/// four-axis quality regression head, and silently sharing an arch would
/// mis-route the runtime dispatch (an ASR / speaker / emotion path
/// would try to interpret the projection head as its own).
pub(crate) const ARCH: &str = "audiobox-aesthetics";

/// `vokra.model.name` for the canonical Audiobox-Aesthetics GGUF.
pub(crate) const NAME: &str = "audiobox-aesthetics";

/// `vokra.model.category` value — `"classification"`. This is the
/// **first Vokra converter with this category tag**; sibling categories
/// today are `asr` / `tts` / `codec` / `speaker` / `emotion` / `s2s` /
/// `bert` / `vad`. Category is a taxonomy tag orthogonal to `arch`
/// (the runtime dispatches on arch, not category); zoo / catalog
/// surfaces group by category so a per-axis quality regressor is
/// visibly distinct from a per-label emotion classifier.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "classification";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const UPSTREAM_HF: &str = "facebook/audiobox-aesthetics";

/// Immutable Hugging Face checkpoint revision whose 324-tensor manifest is
/// accepted by this converter and the native runtime.
pub const CHECKPOINT_REVISION: &str = "9b1dd8e5df9af7216e836a98974fe3b82c56ded6";
/// Immutable upstream source revision used to transcribe the WavLM and
/// four-axis predictor forward.
pub const SOURCE_REVISION: &str = "2618e9d451b456e9328b39495b5e6234678aa550";

const SAMPLE_RATE: u32 = 16_000;
const WINDOW_SAMPLES: u32 = 160_000;
const HOP_SAMPLES: u32 = 160_000;
const FEATURE_DIM: u32 = 512;
const HIDDEN_SIZE: u32 = 768;
const FFN_DIM: u32 = 3072;
const N_LAYER: u32 = 12;
const N_HEAD: u32 = 12;
const POS_CONV_KERNEL: u32 = 128;
const POS_CONV_GROUPS: u32 = 16;
const NUM_BUCKETS: u32 = 320;
const MAX_DISTANCE: u32 = 800;
const NTH_LAYER: u32 = 13;
const PROJ_NUM_LAYER: u32 = 5;
const OUTPUT_DIM: u32 = 1;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const TENSOR_COUNT: usize = 324;
const AXES: [&str; 4] = ["CE", "CU", "PC", "PQ"];
const TARGET_MEANS: [f32; 4] = [5.06865, 5.73633, 3.18591, 6.57505];
const TARGET_STDS: [f32; 4] = [1.93029, 1.75669, 1.86637, 1.51466];

const PREFIX: &str = "vokra.audiobox_aesthetics";
const KEY_CHECKPOINT_REVISION: &str = "vokra.audiobox_aesthetics.checkpoint_revision";
const KEY_SOURCE_REVISION: &str = "vokra.audiobox_aesthetics.source_revision";
const KEY_AXES: &str = "vokra.audiobox_aesthetics.axes";
const KEY_TARGET_MEANS: &str = "vokra.audiobox_aesthetics.target_means";
const KEY_TARGET_STDS: &str = "vokra.audiobox_aesthetics.target_stds";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` Meta / Audiobox row. CC-BY 4.0 requires
/// attribution on display / distribution; this text is what the runtime
/// + catalog generator surface verbatim.
pub(crate) const AUDIOBOX_AESTHETICS_ATTRIBUTION_TEXT: &str = "This application uses the Audiobox \
     Aesthetics model (WavLM SSL backbone + 5-layer projection MLP heads predicting \
     CONTENT_ENJOYMENT / CONTENT_USEFULNESS / PRODUCTION_COMPLEXITY / PRODUCTION_QUALITY \
     audio-quality axes; arXiv:2502.05139). Model weights are licensed \
     under CC-BY 4.0 (attribution required; commercial use permitted). Copyright (c) Meta / \
     Facebook AI Research. Source: \
     https://github.com/facebookresearch/audiobox-aesthetics / \
     https://huggingface.co/facebook/audiobox-aesthetics";

/// Outcome of an Audiobox-Aesthetics conversion.
///
/// Mirrors [`crate::models::wespeaker::WespeakerReport`]'s counter
/// contract (leading `read` count + `written`/`skipped_non_float` split
/// plus a retained BF16 diagnostic). `read == written + skipped_non_float`
/// is an invariant preserved by [`convert_audiobox_aesthetics_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AudioboxAestheticsReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// F32 tensors written verbatim under their upstream names. The exact
    /// pinned checkpoint is F32-only; another dtype is rejected before this
    /// counter is populated.
    pub written: usize,
    /// Retained for report-shape compatibility. Always zero after the strict
    /// F32 manifest gate.
    pub skipped_non_float: usize,
    /// Retained for report-shape compatibility. Always zero for the pinned
    /// F32 checkpoint; BF16 fine-tunes require their own audited variant.
    pub bf16_passthrough: usize,
}

fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::new();
    for axis in AXES {
        expected.insert(format!("layer_weights.{axis}"), vec![NTH_LAYER as u64]);
        for index in [0, 3, 6, 9] {
            expected.insert(
                format!("proj_layer.{axis}.{index}.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            expected.insert(
                format!("proj_layer.{axis}.{index}.bias"),
                vec![HIDDEN_SIZE as u64],
            );
        }
        for index in [1, 4, 7, 10] {
            expected.insert(
                format!("proj_layer.{axis}.{index}.weight"),
                vec![HIDDEN_SIZE as u64],
            );
            expected.insert(
                format!("proj_layer.{axis}.{index}.bias"),
                vec![HIDDEN_SIZE as u64],
            );
        }
        expected.insert(
            format!("proj_layer.{axis}.12.weight"),
            vec![OUTPUT_DIM as u64, HIDDEN_SIZE as u64],
        );
        expected.insert(
            format!("proj_layer.{axis}.12.bias"),
            vec![OUTPUT_DIM as u64],
        );
    }

    expected.insert(
        "wavlm_model.encoder.layer_norm.weight".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    expected.insert(
        "wavlm_model.encoder.layer_norm.bias".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    for layer in 0..N_LAYER as usize {
        let p = format!("wavlm_model.encoder.layers.{layer}");
        for (suffix, shape) in [
            ("fc1.weight", vec![FFN_DIM as u64, HIDDEN_SIZE as u64]),
            ("fc1.bias", vec![FFN_DIM as u64]),
            ("fc2.weight", vec![HIDDEN_SIZE as u64, FFN_DIM as u64]),
            ("fc2.bias", vec![HIDDEN_SIZE as u64]),
            ("final_layer_norm.weight", vec![HIDDEN_SIZE as u64]),
            ("final_layer_norm.bias", vec![HIDDEN_SIZE as u64]),
            ("self_attn.grep_a", vec![1, N_HEAD as u64, 1, 1]),
            ("self_attn.grep_linear.weight", vec![8, 64]),
            ("self_attn.grep_linear.bias", vec![8]),
            (
                "self_attn.k_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.k_proj.bias", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.out_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.out_proj.bias", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.q_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.q_proj.bias", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.v_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.v_proj.bias", vec![HIDDEN_SIZE as u64]),
            ("self_attn_layer_norm.weight", vec![HIDDEN_SIZE as u64]),
            ("self_attn_layer_norm.bias", vec![HIDDEN_SIZE as u64]),
        ] {
            expected.insert(format!("{p}.{suffix}"), shape);
        }
    }
    expected.insert(
        "wavlm_model.encoder.layers.0.self_attn.relative_attention_bias.weight".to_owned(),
        vec![NUM_BUCKETS as u64, N_HEAD as u64],
    );
    expected.insert(
        "wavlm_model.encoder.pos_conv.0.weight_g".to_owned(),
        vec![1, 1, POS_CONV_KERNEL as u64],
    );
    expected.insert(
        "wavlm_model.encoder.pos_conv.0.weight_v".to_owned(),
        vec![
            HIDDEN_SIZE as u64,
            (HIDDEN_SIZE / POS_CONV_GROUPS) as u64,
            POS_CONV_KERNEL as u64,
        ],
    );
    expected.insert(
        "wavlm_model.encoder.pos_conv.0.bias".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );

    let kernels = [10_u64, 3, 3, 3, 3, 2, 2];
    for (layer, kernel) in kernels.into_iter().enumerate() {
        let input = if layer == 0 { 1 } else { FEATURE_DIM as u64 };
        expected.insert(
            format!("wavlm_model.feature_extractor.conv_layers.{layer}.0.weight"),
            vec![FEATURE_DIM as u64, input, kernel],
        );
    }
    for suffix in ["weight", "bias"] {
        expected.insert(
            format!("wavlm_model.feature_extractor.conv_layers.0.2.{suffix}"),
            vec![FEATURE_DIM as u64],
        );
        expected.insert(
            format!("wavlm_model.layer_norm.{suffix}"),
            vec![FEATURE_DIM as u64],
        );
    }
    expected.insert("wavlm_model.mask_emb".to_owned(), vec![HIDDEN_SIZE as u64]);
    expected.insert(
        "wavlm_model.post_extract_proj.weight".to_owned(),
        vec![HIDDEN_SIZE as u64, FEATURE_DIM as u64],
    );
    expected.insert(
        "wavlm_model.post_extract_proj.bias".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    debug_assert_eq!(expected.len(), TENSOR_COUNT);
    expected
}

fn validate_manifest_entries<'a>(
    entries: impl IntoIterator<Item = (&'a str, GgmlType, &'a [u64])>,
) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    let mut seen = BTreeSet::new();
    let mut count = 0_usize;
    for (name, dtype, actual_shape) in entries {
        count += 1;
        let Some(shape) = expected.get(name) else {
            return Err(ConvertError::Parse(format!(
                "audiobox-aesthetics: unexpected tensor `{name}` at revision {CHECKPOINT_REVISION}"
            )));
        };
        if actual_shape != shape.as_slice() {
            return Err(ConvertError::Parse(format!(
                "audiobox-aesthetics: tensor `{}` has shape {:?}, expected {:?} at revision {CHECKPOINT_REVISION}",
                name, actual_shape, shape
            )));
        }
        if dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "audiobox-aesthetics: tensor `{}` is {:?}, expected F32 at revision {CHECKPOINT_REVISION}",
                name, dtype
            )));
        }
        seen.insert(name);
    }
    if count != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "audiobox-aesthetics: checkpoint has {count} tensors, expected exactly {TENSOR_COUNT} at revision {CHECKPOINT_REVISION}"
        )));
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(ConvertError::Parse(format!(
            "audiobox-aesthetics: missing tensor `{missing}` at revision {CHECKPOINT_REVISION}"
        )));
    }
    Ok(())
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    validate_manifest_entries(
        st.tensors()
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.dtype, tensor.shape.as_slice())),
    )
}

fn add_string_array(builder: &mut GgufBuilder, key: &str, values: &[&str]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: values
                .iter()
                .map(|value| GgufMetadataValue::String((*value).to_owned()))
                .collect(),
        }),
    );
}

fn add_f32_array(builder: &mut GgufBuilder, key: &str, values: &[f32]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::F32,
            values: values.iter().copied().map(GgufMetadataValue::F32).collect(),
        }),
    );
}

fn stamp_contract(builder: &mut GgufBuilder) {
    builder.add_string(KEY_CHECKPOINT_REVISION, CHECKPOINT_REVISION);
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    for (name, value) in [
        ("sample_rate", SAMPLE_RATE),
        ("window_samples", WINDOW_SAMPLES),
        ("hop_samples", HOP_SAMPLES),
        ("feature_dim", FEATURE_DIM),
        ("hidden_size", HIDDEN_SIZE),
        ("ffn_dim", FFN_DIM),
        ("n_layer", N_LAYER),
        ("n_head", N_HEAD),
        ("pos_conv_kernel", POS_CONV_KERNEL),
        ("pos_conv_groups", POS_CONV_GROUPS),
        ("num_buckets", NUM_BUCKETS),
        ("max_distance", MAX_DISTANCE),
        ("nth_layer", NTH_LAYER),
        ("proj_num_layer", PROJ_NUM_LAYER),
        ("output_dim", OUTPUT_DIM),
    ] {
        builder.add_u32(&format!("{PREFIX}.{name}"), value);
    }
    builder.add_bool(&format!("{PREFIX}.normalize_embed"), true);
    builder.add_bool(&format!("{PREFIX}.use_weighted_layer_sum"), true);
    builder.add_f32(&format!("{PREFIX}.layer_norm_eps"), LAYER_NORM_EPS);
    add_string_array(builder, KEY_AXES, &AXES);
    add_f32_array(builder, KEY_TARGET_MEANS, &TARGET_MEANS);
    add_f32_array(builder, KEY_TARGET_STDS, &TARGET_STDS);
}

/// File-based Audiobox-Aesthetics converter
/// (`vokra-cli convert --model audiobox-aesthetics`).
///
/// Reads `input` (upstream `facebook/audiobox-aesthetics`
/// `model.safetensors`), writes a Vokra GGUF to `output`. `license`
/// overrides the default `cc-by-4.0` provenance stamp (the same
/// `convert_file_licensed` override mechanism the Whisper / kokoro
/// family paths use); pass `None` to keep the built-in `cc-by-4.0`
/// stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_audiobox_aesthetics_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioboxAestheticsReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_manifest(&st)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    stamp_contract(&mut b);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-4.0 (upstream
    // `facebook/audiobox-aesthetics` cardData `license: cc-by-4.0`,
    // primary-source verified 2026-07-30 via
    // `https://huggingface.co/api/models/facebook/audiobox-aesthetics`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("cc-by-4.0".to_owned(), LicenseClass::AttributionRequired),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "facebook/audiobox-aesthetics (WavLM backbone + four-axis quality regression, cc-by-4.0)",
        ),
    );
    // FR-MD-09 attribution surface — CC-BY 4.0 requires attribution on
    // *display / distribution*; we stamp the text so the runtime + the
    // catalog generator surface it verbatim.
    vokra_core::stamp_attribution(&mut b, AUDIOBOX_AESTHETICS_ATTRIBUTION_TEXT);

    let mut report = AudioboxAestheticsReport::default();
    // The pinned public revision is all F32 and has already passed the exact
    // 324-name/shape/dtype manifest above. Bytes now pass through verbatim.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
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
mod contract_tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn validate_owned(entries: &[(String, Vec<u64>, GgmlType)]) -> Result<(), ConvertError> {
        validate_manifest_entries(
            entries
                .iter()
                .map(|(name, shape, dtype)| (name.as_str(), *dtype, shape.as_slice())),
        )
    }

    #[test]
    fn pinned_manifest_has_exactly_324_tensors_and_four_axes() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(AXES, ["CE", "CU", "PC", "PQ"]);
        assert!(!manifest.keys().any(|name| name.contains("BALANCED")));
        assert_eq!(
            manifest["wavlm_model.encoder.layers.0.self_attn.relative_attention_bias.weight"],
            vec![320, 12]
        );
        assert_eq!(
            manifest["wavlm_model.encoder.pos_conv.0.weight_v"],
            vec![768, 48, 128]
        );
        assert_eq!(manifest["proj_layer.PQ.12.weight"], vec![1, 768]);
    }

    #[test]
    fn manifest_validation_accepts_only_the_pinned_names_shapes_and_dtype() {
        let mut entries = expected_manifest()
            .into_iter()
            .map(|(name, shape)| (name, shape, GgmlType::F32))
            .collect::<Vec<_>>();
        validate_owned(&entries).expect("exact pinned manifest");

        let target = entries
            .iter()
            .position(|(name, _, _)| name == "proj_layer.CE.12.weight")
            .expect("target tensor");
        entries[target].1 = vec![2, 768];
        let error = validate_owned(&entries).unwrap_err().to_string();
        assert!(error.contains("proj_layer.CE.12.weight"));
        assert!(error.contains("expected [1, 768]"));

        entries[target].1 = vec![1, 768];
        entries[target].2 = GgmlType::BF16;
        let error = validate_owned(&entries).unwrap_err().to_string();
        assert!(error.contains("expected F32"));
    }

    #[test]
    fn manifest_validation_rejects_missing_tensor() {
        let mut entries = expected_manifest()
            .into_iter()
            .map(|(name, shape)| (name, shape, GgmlType::F32))
            .collect::<Vec<_>>();
        entries.pop();
        let error = validate_owned(&entries).unwrap_err().to_string();
        assert!(error.contains("323 tensors"));
        assert!(error.contains("expected exactly 324"));
    }

    #[test]
    fn canonical_metadata_pins_revisions_topology_and_target_transform() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        stamp_contract(&mut builder);
        builder
            .add_tensor(
                "fixture",
                GgmlType::F32,
                vec![1],
                1.0_f32.to_le_bytes().to_vec(),
            )
            .expect("fixture tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");

        assert_eq!(
            file.get(KEY_CHECKPOINT_REVISION)
                .and_then(|value| value.as_str()),
            Some(CHECKPOINT_REVISION)
        );
        assert_eq!(
            file.get(KEY_SOURCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(SOURCE_REVISION)
        );
        assert_eq!(
            file.get(&format!("{PREFIX}.n_layer"))
                .and_then(|value| value.as_u64()),
            Some(12)
        );
        let axes = file
            .get(KEY_AXES)
            .and_then(|value| value.as_array())
            .expect("axes array");
        assert_eq!(
            axes.values
                .iter()
                .map(|value| value.as_str().expect("axis string"))
                .collect::<Vec<_>>(),
            AXES
        );
        let means = file
            .get(KEY_TARGET_MEANS)
            .and_then(|value| value.as_array())
            .expect("mean array");
        assert_eq!(means.values.len(), 4);
        assert!(AUDIOBOX_AESTHETICS_ATTRIBUTION_TEXT.contains("WavLM"));
        assert!(!AUDIOBOX_AESTHETICS_ATTRIBUTION_TEXT.contains("BALANCED"));
    }
}
