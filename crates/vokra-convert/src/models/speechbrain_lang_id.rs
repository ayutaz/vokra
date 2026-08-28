//! **SpeechBrain lang-ID family**: safetensors → GGUF conversion
//! (TIER 1 F wave, 2026-07-30).
//!
//! Covers two upstream SpeechBrain ECAPA language-ID releases:
//!
//! - **F7** = `speechbrain/lang-id-voxlingua107-ecapa` — 107-language
//!   ID trained on VoxLingua107 (Valk & Alumäe 2021 —
//!   `arXiv:2011.12998`).
//! - **F9** = `speechbrain/lang-id-commonlanguage_ecapa` — variant
//!   trained on the CommonLanguage dataset (~45 languages).
//!
//! They share the ECAPA family, but **not** one complete topology.  The
//! official VoxLingua107 release uses 60-bin features, a 256-d embedding and
//! an XVector MLP classifier.  CommonLanguage uses 80-bin features, a 192-d
//! embedding and the ECAPA cosine classifier.  The converter therefore reads
//! an explicit prepared-checkpoint contract and never derives one variant
//! from the other.
//!
//! The public Vokra artifact predating this contract contains only the
//! embedding module.  It has neither the official classifier nor the label
//! encoder, and cannot identify a language.  This converter now requires the
//! output of `tools/parity/speechbrain_lang_id_prepare_checkpoint.py`, which
//! loads the real pinned SpeechBrain release and stores both modules plus the
//! ordered label inventory in safetensors `__metadata__`.  An embedding-only
//! input fails before a GGUF is written (FR-EX-08).
//!
//! # Provenance
//!
//! - **HF paths**:
//!   - `speechbrain/lang-id-voxlingua107-ecapa` (F7, canonical)
//!   - `speechbrain/lang-id-commonlanguage_ecapa` (F9, sibling)
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`) for both
//!   (per SpeechBrain family license `github.com/speechbrain/speechbrain/blob/develop/LICENSE`).
//! - **Category**: `classification` (language identification is a fixed
//!   N-way classifier — recorded under `vokra.model.category`).
//!
//! # BF16 pass-through
//!
//! Mirror of `wespeaker` / `ecapa_tdnn` / `clap`.

use std::collections::HashSet;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for the SpeechBrain ECAPA language-ID family.
/// Variant-specific topology is carried by the required `vokra.lang_id.*`
/// contract rather than inferred from this family dispatch tag.
pub const ARCH: &str = "lang_id_ecapa";

/// Model name for the F7 (VoxLingua107) variant.
pub const NAME_VOXLINGUA107: &str = "lang-id-voxlingua107-ecapa";

/// Model name for the F9 (CommonLanguage) variant.
pub const NAME_COMMONLANGUAGE: &str = "lang-id-commonlanguage-ecapa";

pub const CATEGORY: &str = "classification";

/// Upstream HF path for F7.
pub const UPSTREAM_HF_VOXLINGUA107: &str = "speechbrain/lang-id-voxlingua107-ecapa";

/// Upstream HF path for F9.
pub const UPSTREAM_HF_COMMONLANGUAGE: &str = "speechbrain/lang-id-commonlanguage_ecapa";

pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const PREPARED_CONTRACT_KEY: &str = "vokra.lang_id.contract";
const PREPARED_FORMAT: &str = "vokra-speechbrain-lang-id-prepared-v2";

pub const KEY_UPSTREAM_REVISION: &str = "vokra.lang_id.upstream_revision";
pub const KEY_SAMPLE_RATE: &str = "vokra.lang_id.sample_rate";
pub const KEY_N_MELS: &str = "vokra.lang_id.n_mels";
pub const KEY_TDNN_CHANNELS: &str = "vokra.lang_id.tdnn_channels";
pub const KEY_MFA_CHANNELS: &str = "vokra.lang_id.mfa_channels";
pub const KEY_ATTENTION_CHANNELS: &str = "vokra.lang_id.attention_channels";
pub const KEY_RES2NET_SCALE: &str = "vokra.lang_id.res2net_scale";
pub const KEY_BLOCK_KERNELS: &str = "vokra.lang_id.block_kernels";
pub const KEY_BLOCK_DILATIONS: &str = "vokra.lang_id.block_dilations";
pub const KEY_EMBEDDING_DIM: &str = "vokra.lang_id.embedding_dim";
pub const KEY_CLASSIFIER_KIND: &str = "vokra.lang_id.classifier_kind";
pub const KEY_CLASSIFIER_HIDDEN_DIM: &str = "vokra.lang_id.classifier_hidden_dim";
pub const KEY_CLASS_COUNT: &str = "vokra.lang_id.class_count";
pub const KEY_LABELS: &str = "vokra.lang_id.labels";
pub const KEY_BN_EPS: &str = "vokra.lang_id.bn_eps";
pub const KEY_STATS_EPS: &str = "vokra.lang_id.stats_eps";
pub const KEY_LEAKY_RELU_SLOPE: &str = "vokra.lang_id.leaky_relu_slope";
pub const KEY_ARTIFACT_LAYOUT: &str = "vokra.lang_id.artifact_layout";
pub const ARTIFACT_LAYOUT: &str = "speechbrain-lang-id-prepared-v2";

/// Which upstream variant to stamp on the GGUF.
///
/// The variants intentionally retain one family arch tag, while the prepared
/// contract pins their different frontend, embedding and classifier layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// F7: `speechbrain/lang-id-voxlingua107-ecapa` (default).
    VoxLingua107,
    /// F9: `speechbrain/lang-id-commonlanguage_ecapa`.
    CommonLanguage,
}

impl Variant {
    /// The `vokra.model.name` value stamped for this variant.
    pub const fn name(self) -> &'static str {
        match self {
            Self::VoxLingua107 => NAME_VOXLINGUA107,
            Self::CommonLanguage => NAME_COMMONLANGUAGE,
        }
    }

    /// The upstream HF slug for this variant.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::VoxLingua107 => UPSTREAM_HF_VOXLINGUA107,
            Self::CommonLanguage => UPSTREAM_HF_COMMONLANGUAGE,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PreparedContract {
    upstream_revision: String,
    sample_rate: u32,
    n_mels: u32,
    tdnn_channels: u32,
    mfa_channels: u32,
    attention_channels: u32,
    res2net_scale: u32,
    block_kernels: Vec<u32>,
    block_dilations: Vec<u32>,
    embedding_dim: u32,
    classifier_kind: String,
    classifier_hidden_dim: Option<u32>,
    class_count: u32,
    labels: Vec<String>,
    bn_eps: f32,
    stats_eps: f32,
    leaky_relu_slope: Option<f32>,
}

impl PreparedContract {
    fn parse(bytes: &[u8], variant: Variant) -> Result<Self, ConvertError> {
        if bytes.len() < 8 {
            return Err(ConvertError::Parse(
                "lang_id_ecapa: prepared safetensors is shorter than its header prefix".into(),
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| {
            ConvertError::Parse("lang_id_ecapa: invalid safetensors header prefix".into())
        })?);
        let header_end = 8_u64.checked_add(header_len).ok_or_else(|| {
            ConvertError::Parse("lang_id_ecapa: safetensors header length overflow".into())
        })?;
        if header_end > bytes.len() as u64 {
            return Err(ConvertError::Parse(
                "lang_id_ecapa: truncated safetensors header".into(),
            ));
        }
        let root = json::parse(&bytes[8..header_end as usize])
            .map_err(|error| ConvertError::Parse(format!("lang_id_ecapa header: {error}")))?;
        let metadata = root
            .get("__metadata__")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                ConvertError::Parse(
                    "lang_id_ecapa: missing safetensors `__metadata__`; run ".to_owned()
                        + "tools/parity/speechbrain_lang_id_prepare_checkpoint.py on VAST "
                        + "instead of converting the embedding checkpoint directly",
                )
            })?;
        let contract_text = metadata
            .iter()
            .find(|(key, _)| key == PREPARED_CONTRACT_KEY)
            .and_then(|(_, value)| value.as_str())
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "lang_id_ecapa: prepared metadata missing string `{PREPARED_CONTRACT_KEY}`"
                ))
            })?;
        let contract = json::parse(contract_text.as_bytes()).map_err(|error| {
            ConvertError::Parse(format!(
                "lang_id_ecapa `{PREPARED_CONTRACT_KEY}` JSON: {error}"
            ))
        })?;
        require_json_string(&contract, "format", PREPARED_FORMAT)?;
        require_json_string(&contract, "model_name", variant.name())?;
        require_json_string(&contract, "source", variant.upstream_hf())?;

        let upstream_revision = json_string(&contract, "revision")?.to_owned();
        if upstream_revision.len() != 40
            || !upstream_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: revision must be a full 40-hex commit, got `{upstream_revision}`"
            )));
        }
        let classifier_kind = json_string(&contract, "classifier_kind")?.to_owned();
        let expected_classifier = match variant {
            Variant::VoxLingua107 => "xvector-mlp-log-softmax-v1",
            Variant::CommonLanguage => "ecapa-cosine-v1",
        };
        if classifier_kind != expected_classifier {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: classifier_kind `{classifier_kind}` does not match {} `{expected_classifier}`",
                variant.name()
            )));
        }
        let labels = contract
            .get("labels")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                ConvertError::Parse("lang_id_ecapa: contract `labels` is not an array".into())
            })?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .filter(|label| !label.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        ConvertError::Parse(format!(
                            "lang_id_ecapa: label index {index} is not a non-empty string"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unique = labels.iter().collect::<HashSet<_>>();
        if unique.len() != labels.len() {
            return Err(ConvertError::Parse(
                "lang_id_ecapa: label inventory contains duplicates".into(),
            ));
        }
        let class_count = json_u32(&contract, "class_count")?;
        if class_count as usize != labels.len() {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: class_count={class_count} but labels.len()={}",
                labels.len()
            )));
        }
        let classifier_hidden_dim = match contract.get("classifier_hidden_dim") {
            Some(JsonValue::Null) | None => None,
            Some(value) => Some(json_value_u32(value, "classifier_hidden_dim")?),
        };
        match variant {
            Variant::VoxLingua107 if classifier_hidden_dim.is_none() => {
                return Err(ConvertError::Parse(
                    "lang_id_ecapa: VoxLingua107 XVector classifier is missing hidden width".into(),
                ));
            }
            Variant::CommonLanguage if classifier_hidden_dim.is_some() => {
                return Err(ConvertError::Parse(
                    "lang_id_ecapa: CommonLanguage cosine classifier must not declare an MLP hidden width"
                        .into(),
                ));
            }
            _ => {}
        }
        let leaky_relu_slope = match contract.get("leaky_relu_slope") {
            Some(JsonValue::Null) | None => None,
            Some(_) => Some(json_f32(&contract, "leaky_relu_slope")?),
        };
        match variant {
            Variant::VoxLingua107 if leaky_relu_slope.is_none() => {
                return Err(ConvertError::Parse(
                    "lang_id_ecapa: VoxLingua107 XVector classifier is missing LeakyReLU slope"
                        .into(),
                ));
            }
            Variant::CommonLanguage if leaky_relu_slope.is_some() => {
                return Err(ConvertError::Parse(
                    "lang_id_ecapa: CommonLanguage cosine classifier must not declare a LeakyReLU slope"
                        .into(),
                ));
            }
            _ => {}
        }

        let parsed = Self {
            upstream_revision,
            sample_rate: json_u32(&contract, "sample_rate")?,
            n_mels: json_u32(&contract, "n_mels")?,
            tdnn_channels: json_u32(&contract, "tdnn_channels")?,
            mfa_channels: json_u32(&contract, "mfa_channels")?,
            attention_channels: json_u32(&contract, "attention_channels")?,
            res2net_scale: json_u32(&contract, "res2net_scale")?,
            block_kernels: json_u32_array(&contract, "block_kernels", 3)?,
            block_dilations: json_u32_array(&contract, "block_dilations", 3)?,
            embedding_dim: json_u32(&contract, "embedding_dim")?,
            classifier_kind,
            classifier_hidden_dim,
            class_count,
            labels,
            bn_eps: json_f32(&contract, "bn_eps")?,
            stats_eps: json_f32(&contract, "stats_eps")?,
            leaky_relu_slope,
        };
        parsed.validate(variant)?;
        Ok(parsed)
    }

    fn validate(&self, variant: Variant) -> Result<(), ConvertError> {
        for (name, value) in [
            ("sample_rate", self.sample_rate),
            ("n_mels", self.n_mels),
            ("tdnn_channels", self.tdnn_channels),
            ("mfa_channels", self.mfa_channels),
            ("attention_channels", self.attention_channels),
            ("res2net_scale", self.res2net_scale),
            ("embedding_dim", self.embedding_dim),
            ("class_count", self.class_count),
        ] {
            if value == 0 {
                return Err(ConvertError::Parse(format!(
                    "lang_id_ecapa: contract `{name}` must be non-zero"
                )));
            }
        }
        if self.tdnn_channels % self.res2net_scale != 0 {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: tdnn_channels={} is not divisible by res2net_scale={}",
                self.tdnn_channels, self.res2net_scale
            )));
        }
        if self.mfa_channels != self.tdnn_channels * 3 {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: mfa_channels={} must equal three ECAPA block outputs ({})",
                self.mfa_channels,
                self.tdnn_channels * 3
            )));
        }
        for (name, values) in [
            ("block_kernels", &self.block_kernels),
            ("block_dilations", &self.block_dilations),
        ] {
            if values.len() != 3 || values.contains(&0) {
                return Err(ConvertError::Parse(format!(
                    "lang_id_ecapa: contract `{name}` must contain three positive values"
                )));
            }
        }
        if self.block_kernels.iter().any(|value| value % 2 == 0) {
            return Err(ConvertError::Parse(
                "lang_id_ecapa: ECAPA block kernels must be odd for SpeechBrain same padding"
                    .into(),
            ));
        }
        for (name, value) in [("bn_eps", self.bn_eps), ("stats_eps", self.stats_eps)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConvertError::Parse(format!(
                    "lang_id_ecapa: contract `{name}` must be finite and positive, got {value}"
                )));
            }
        }
        if let Some(value) = self.leaky_relu_slope
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: contract `leaky_relu_slope` must be finite and positive, got {value}"
            )));
        }
        let (n_mels, embedding_dim, block_kernels, class_count, hidden, slope) = match variant {
            Variant::VoxLingua107 => (60, 256, &[3, 3, 3][..], 107, Some(512), Some(0.01)),
            Variant::CommonLanguage => (80, 192, &[3, 3, 1][..], 45, None, None),
        };
        for (name, actual, expected) in [
            ("sample_rate", self.sample_rate, 16_000),
            ("n_mels", self.n_mels, n_mels),
            ("tdnn_channels", self.tdnn_channels, 1_024),
            ("mfa_channels", self.mfa_channels, 3_072),
            ("attention_channels", self.attention_channels, 128),
            ("res2net_scale", self.res2net_scale, 8),
            ("embedding_dim", self.embedding_dim, embedding_dim),
            ("class_count", self.class_count, class_count),
        ] {
            if actual != expected {
                return Err(ConvertError::Parse(format!(
                    "lang_id_ecapa: {} contract `{name}` is {actual}, expected official value {expected}",
                    variant.name()
                )));
            }
        }
        if self.block_kernels.as_slice() != block_kernels
            || self.block_dilations.as_slice() != [2, 3, 4]
        {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: {} block contract is kernels={:?} dilations={:?}, expected kernels={block_kernels:?} dilations=[2, 3, 4]",
                variant.name(),
                self.block_kernels,
                self.block_dilations
            )));
        }
        if self.classifier_hidden_dim != hidden {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: {} classifier hidden width {:?}, expected {hidden:?}",
                variant.name(),
                self.classifier_hidden_dim
            )));
        }
        if self.bn_eps.to_bits() != 1.0e-5_f32.to_bits()
            || self.stats_eps.to_bits() != 1.0e-12_f32.to_bits()
            || self.leaky_relu_slope.map(f32::to_bits) != slope.map(f32::to_bits)
        {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: {} normalization contract does not match the official release",
                variant.name()
            )));
        }
        Ok(())
    }

    fn stamp(&self, builder: &mut GgufBuilder) {
        builder.add_string(KEY_UPSTREAM_REVISION, &self.upstream_revision);
        builder.add_u32(KEY_SAMPLE_RATE, self.sample_rate);
        builder.add_u32(KEY_N_MELS, self.n_mels);
        builder.add_u32(KEY_TDNN_CHANNELS, self.tdnn_channels);
        builder.add_u32(KEY_MFA_CHANNELS, self.mfa_channels);
        builder.add_u32(KEY_ATTENTION_CHANNELS, self.attention_channels);
        builder.add_u32(KEY_RES2NET_SCALE, self.res2net_scale);
        builder.add_metadata(KEY_BLOCK_KERNELS, u32_array(&self.block_kernels));
        builder.add_metadata(KEY_BLOCK_DILATIONS, u32_array(&self.block_dilations));
        builder.add_u32(KEY_EMBEDDING_DIM, self.embedding_dim);
        builder.add_string(KEY_CLASSIFIER_KIND, &self.classifier_kind);
        if let Some(hidden) = self.classifier_hidden_dim {
            builder.add_u32(KEY_CLASSIFIER_HIDDEN_DIM, hidden);
        }
        builder.add_u32(KEY_CLASS_COUNT, self.class_count);
        builder.add_metadata(KEY_LABELS, string_array(&self.labels));
        builder.add_f32(KEY_BN_EPS, self.bn_eps);
        builder.add_f32(KEY_STATS_EPS, self.stats_eps);
        if let Some(slope) = self.leaky_relu_slope {
            builder.add_f32(KEY_LEAKY_RELU_SLOPE, slope);
        }
        builder.add_string(KEY_ARTIFACT_LAYOUT, ARTIFACT_LAYOUT);
    }
}

fn json_string<'a>(root: &'a JsonValue, key: &str) -> Result<&'a str, ConvertError> {
    root.get(key).and_then(JsonValue::as_str).ok_or_else(|| {
        ConvertError::Parse(format!(
            "lang_id_ecapa: contract `{key}` is missing or not a string"
        ))
    })
}

fn require_json_string(root: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    let actual = json_string(root, key)?;
    if actual != expected {
        return Err(ConvertError::Parse(format!(
            "lang_id_ecapa: contract `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn json_value_u32(value: &JsonValue, key: &str) -> Result<u32, ConvertError> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .filter(|number| *number > 0)
        .ok_or_else(|| {
            ConvertError::Parse(format!(
                "lang_id_ecapa: contract `{key}` is not a positive u32"
            ))
        })
}

fn json_u32(root: &JsonValue, key: &str) -> Result<u32, ConvertError> {
    let value = root
        .get(key)
        .ok_or_else(|| ConvertError::Parse(format!("lang_id_ecapa: contract missing `{key}`")))?;
    json_value_u32(value, key)
}

fn json_u32_array(
    root: &JsonValue,
    key: &str,
    expected_len: usize,
) -> Result<Vec<u32>, ConvertError> {
    let values = root.get(key).and_then(JsonValue::as_array).ok_or_else(|| {
        ConvertError::Parse(format!(
            "lang_id_ecapa: contract `{key}` is missing or not an array"
        ))
    })?;
    if values.len() != expected_len {
        return Err(ConvertError::Parse(format!(
            "lang_id_ecapa: contract `{key}` has {} values, expected {expected_len}",
            values.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| json_value_u32(value, &format!("{key}[{index}]")))
        .collect()
}

fn json_f32(root: &JsonValue, key: &str) -> Result<f32, ConvertError> {
    let value = match root.get(key) {
        Some(JsonValue::Int(value)) => *value as f32,
        Some(JsonValue::Float(value)) => *value as f32,
        _ => {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: contract `{key}` is missing or not numeric"
            )));
        }
    };
    if !value.is_finite() {
        return Err(ConvertError::Parse(format!(
            "lang_id_ecapa: contract `{key}` is not finite"
        )));
    }
    Ok(value)
}

fn string_array(values: &[String]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: values
            .iter()
            .cloned()
            .map(GgufMetadataValue::String)
            .collect(),
    })
}

fn u32_array(values: &[u32]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::U32,
        values: values.iter().copied().map(GgufMetadataValue::U32).collect(),
    })
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeechbrainLangIdReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

/// Variant-aware converter. Callers who want the F7 default use
/// [`convert_speechbrain_lang_id_file`]; the CLI dispatch routes F9
/// (`--model lang-id-commonlanguage`) here with
/// [`Variant::CommonLanguage`].
pub fn convert_speechbrain_lang_id_variant(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: Variant,
) -> Result<SpeechbrainLangIdReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let contract = PreparedContract::parse(&bytes, variant)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_prepared_tensors(&st, &contract)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    let source_note = match variant {
        Variant::VoxLingua107 => {
            "speechbrain/lang-id-voxlingua107-ecapa (ECAPA-TDNN + 107-class lang-id, apache-2.0)"
        }
        Variant::CommonLanguage => {
            "speechbrain/lang-id-commonlanguage_ecapa (ECAPA-TDNN + CommonLanguage lang-id, apache-2.0)"
        }
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(source_note),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    contract.stamp(&mut b);

    let mut report = SpeechbrainLangIdReport::default();
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

fn validate_prepared_tensors(
    st: &SafetensorsFile,
    contract: &PreparedContract,
) -> Result<(), ConvertError> {
    let mut embedding_count = 0_usize;
    let mut classifier_count = 0_usize;
    for tensor in st.tensors() {
        if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: prepared tensor `{}` has non-floating dtype {:?}",
                tensor.name, tensor.dtype
            )));
        }
        if tensor.name.starts_with("embedding_model.") {
            embedding_count += 1;
        } else if tensor.name.starts_with("classifier.") {
            classifier_count += 1;
        } else {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: prepared checkpoint contains unexpected tensor `{}`; only the official embedding_model/classifier modules are allowed",
                tensor.name
            )));
        }
    }
    if embedding_count != 200 {
        return Err(ConvertError::Parse(format!(
            "lang_id_ecapa: prepared ECAPA backbone has {embedding_count} inference tensors, expected exactly 200"
        )));
    }
    if classifier_count == 0 {
        return Err(ConvertError::Parse(
            "lang_id_ecapa: prepared checkpoint has no `classifier.` tensors; refusing the historical embedding-only artifact"
                .into(),
        ));
    }

    require_tensor_shape(
        st,
        "embedding_model.blocks.0.conv.conv.weight",
        &[contract.tdnn_channels as u64, contract.n_mels as u64, 5],
    )?;
    let res2net_channels = contract.tdnn_channels / contract.res2net_scale;
    for (index, kernel) in contract.block_kernels.iter().enumerate() {
        require_tensor_shape(
            st,
            &format!(
                "embedding_model.blocks.{}.res2net_block.blocks.0.conv.conv.weight",
                index + 1
            ),
            &[
                res2net_channels as u64,
                res2net_channels as u64,
                *kernel as u64,
            ],
        )?;
    }
    require_tensor_shape(
        st,
        "embedding_model.mfa.conv.conv.weight",
        &[
            contract.mfa_channels as u64,
            contract.mfa_channels as u64,
            1,
        ],
    )?;
    require_tensor_shape(
        st,
        "embedding_model.asp.tdnn.conv.conv.weight",
        &[
            contract.attention_channels as u64,
            (contract.mfa_channels * 3) as u64,
            1,
        ],
    )?;
    require_tensor_shape(
        st,
        "embedding_model.fc.conv.weight",
        &[
            contract.embedding_dim as u64,
            (contract.mfa_channels * 2) as u64,
            1,
        ],
    )?;

    validate_classifier_tensors(st, contract, classifier_count)?;
    Ok(())
}

fn validate_classifier_tensors(
    st: &SafetensorsFile,
    contract: &PreparedContract,
    classifier_count: usize,
) -> Result<(), ConvertError> {
    let embedding = contract.embedding_dim as u64;
    let classes = contract.class_count as u64;
    match contract.classifier_kind.as_str() {
        "ecapa-cosine-v1" => {
            if classifier_count != 1 {
                return Err(ConvertError::Parse(format!(
                    "lang_id_ecapa: cosine classifier has {classifier_count} tensors, expected exactly 1"
                )));
            }
            require_tensor_shape(st, "classifier.cosine.weight", &[classes, embedding])?;
        }
        "xvector-mlp-log-softmax-v1" => {
            if classifier_count != 12 {
                return Err(ConvertError::Parse(format!(
                    "lang_id_ecapa: XVector classifier has {classifier_count} tensors, expected exactly 12"
                )));
            }
            let hidden = contract.classifier_hidden_dim.ok_or_else(|| {
                ConvertError::Parse(
                    "lang_id_ecapa: XVector classifier is missing hidden width".into(),
                )
            })? as u64;
            for (prefix, width) in [
                ("classifier.input_norm", embedding),
                ("classifier.hidden_norm", hidden),
            ] {
                for suffix in ["weight", "bias", "running_mean", "running_var"] {
                    require_tensor_shape(st, &format!("{prefix}.{suffix}"), &[width])?;
                }
            }
            require_tensor_shape(st, "classifier.hidden.weight", &[hidden, embedding])?;
            require_tensor_shape(st, "classifier.hidden.bias", &[hidden])?;
            require_tensor_shape(st, "classifier.output.weight", &[classes, hidden])?;
            require_tensor_shape(st, "classifier.output.bias", &[classes])?;
        }
        other => {
            return Err(ConvertError::Parse(format!(
                "lang_id_ecapa: unsupported classifier kind `{other}`"
            )));
        }
    }
    Ok(())
}

fn require_tensor_shape(
    st: &SafetensorsFile,
    name: &str,
    expected: &[u64],
) -> Result<(), ConvertError> {
    let tensor = st.tensor_info(name).ok_or_else(|| {
        ConvertError::Parse(format!(
            "lang_id_ecapa: prepared checkpoint missing `{name}`"
        ))
    })?;
    if tensor.shape != expected {
        return Err(ConvertError::Parse(format!(
            "lang_id_ecapa: `{name}` shape {:?}, expected {expected:?}",
            tensor.shape
        )));
    }
    Ok(())
}

/// Default variant convenience (F7 = VoxLingua107).
pub fn convert_speechbrain_lang_id_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SpeechbrainLangIdReport, ConvertError> {
    convert_speechbrain_lang_id_variant(input, output, license, Variant::VoxLingua107)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn escape_json_string(value: &str) -> String {
        value.replace('\\', "\\\\").replace('"', "\\\"")
    }

    fn contract_json(variant: Variant) -> String {
        let (
            model_name,
            source,
            n_mels,
            embedding_dim,
            block_kernels,
            class_count,
            classifier_kind,
            hidden,
            slope,
        ) = match variant {
            Variant::VoxLingua107 => (
                NAME_VOXLINGUA107,
                UPSTREAM_HF_VOXLINGUA107,
                60,
                256,
                "[3,3,3]",
                107,
                "xvector-mlp-log-softmax-v1",
                "512",
                "0.01",
            ),
            Variant::CommonLanguage => (
                NAME_COMMONLANGUAGE,
                UPSTREAM_HF_COMMONLANGUAGE,
                80,
                192,
                "[3,3,1]",
                45,
                "ecapa-cosine-v1",
                "null",
                "null",
            ),
        };
        let labels = (0..class_count)
            .map(|index| format!(r#""label-{index:03}""#))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"format":"{PREPARED_FORMAT}","model_name":"{model_name}","source":"{source}","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sample_rate":16000,"n_mels":{n_mels},"tdnn_channels":1024,"mfa_channels":3072,"attention_channels":128,"res2net_scale":8,"block_kernels":{block_kernels},"block_dilations":[2,3,4],"embedding_dim":{embedding_dim},"classifier_kind":"{classifier_kind}","classifier_hidden_dim":{hidden},"class_count":{class_count},"labels":[{labels}],"bn_eps":0.00001,"stats_eps":0.000000000001,"leaky_relu_slope":{slope}}}"#
        )
    }

    fn prepared_header(variant: Variant, include_metadata: bool) -> Vec<u8> {
        let metadata = if include_metadata {
            format!(
                r#""__metadata__":{{"{PREPARED_CONTRACT_KEY}":"{}"}},"#,
                escape_json_string(&contract_json(variant))
            )
        } else {
            String::new()
        };
        let header = format!(
            r#"{{{metadata}"embedding_model.test":{{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    #[test]
    fn prepared_contract_pins_both_real_variant_topologies() {
        let vox = PreparedContract::parse(
            &prepared_header(Variant::VoxLingua107, true),
            Variant::VoxLingua107,
        )
        .unwrap();
        assert_eq!(vox.n_mels, 60);
        assert_eq!(vox.embedding_dim, 256);
        assert_eq!(vox.block_kernels, [3, 3, 3]);
        assert_eq!(vox.block_dilations, [2, 3, 4]);
        assert_eq!(vox.classifier_hidden_dim, Some(512));
        assert_eq!(vox.labels.len(), 107);
        assert_eq!(vox.labels[0], "label-000");

        let common = PreparedContract::parse(
            &prepared_header(Variant::CommonLanguage, true),
            Variant::CommonLanguage,
        )
        .unwrap();
        assert_eq!(common.n_mels, 80);
        assert_eq!(common.embedding_dim, 192);
        assert_eq!(common.block_kernels, [3, 3, 1]);
        assert_eq!(common.labels.len(), 45);
        assert_eq!(common.classifier_hidden_dim, None);
        assert_eq!(common.classifier_kind, "ecapa-cosine-v1");
    }

    #[test]
    fn missing_or_cross_variant_contract_fails_closed() {
        let missing = PreparedContract::parse(
            &prepared_header(Variant::VoxLingua107, false),
            Variant::VoxLingua107,
        )
        .unwrap_err();
        assert!(missing.to_string().contains("prepare_checkpoint.py"));

        let mismatch = PreparedContract::parse(
            &prepared_header(Variant::VoxLingua107, true),
            Variant::CommonLanguage,
        )
        .unwrap_err();
        assert!(mismatch.to_string().contains("model_name"));
    }

    #[test]
    fn contract_stamps_runtime_axes_and_ordered_labels() {
        let contract = PreparedContract::parse(
            &prepared_header(Variant::VoxLingua107, true),
            Variant::VoxLingua107,
        )
        .unwrap();
        let mut builder = GgufBuilder::new();
        contract.stamp(&mut builder);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_N_MELS).and_then(|value| value.as_u64()),
            Some(60)
        );
        assert_eq!(
            file.get(KEY_EMBEDDING_DIM).and_then(|value| value.as_u64()),
            Some(256)
        );
        let kernels = file
            .get(KEY_BLOCK_KERNELS)
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert_eq!(kernels.element_type, GgufValueType::U32);
        assert_eq!(
            kernels
                .values
                .iter()
                .filter_map(GgufMetadataValue::as_u64)
                .collect::<Vec<_>>(),
            [3, 3, 3]
        );
        let labels = file
            .get(KEY_LABELS)
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert_eq!(labels.element_type, GgufValueType::String);
        let expected_labels = (0..107)
            .map(|index| format!("label-{index:03}"))
            .collect::<Vec<_>>();
        assert_eq!(
            labels
                .values
                .iter()
                .filter_map(GgufMetadataValue::as_str)
                .collect::<Vec<_>>(),
            expected_labels
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
    }
}
