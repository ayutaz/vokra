//! Native SGMSE-VoiceBank source graph and sampler contracts.
//!
//! This module records the checkpoint-independent part of the pinned
//! `sp-uhh/sgmse` implementation. The score network consumes complex STFT
//! state as four real channels and emits a two-real-channel score, applies the
//! source NCSN++ v2 graph plan, and
//! is sampled with the OUVE predictor/corrector seam. Checkpoint-specific
//! tensor names and shapes are deliberately not guessed: the public binder
//! remains closed until VAST produces the complete safe-loaded manifest.
//!
//! Primary source pin:
//! <https://github.com/sp-uhh/sgmse/tree/1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e>
//! (`sgmse/backbones/ncsnpp.py`, `sgmse/data_module.py`, and
//! `sgmse/model.py`).

use std::collections::BTreeSet;
use std::path::Path;

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::ir::graph::{IstftAttrs, PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::{Result, VokraError};
use vokra_ops::{Spectrogram, istft, stft};

use crate::compute::{Compute, HotOp};

/// Fixed source revision for all SGMSE native source logic.
pub const SOURCE_REVISION: &str = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e";
/// The SGMSE VoiceBank model identity used by the converter contract.
pub const MODEL_NAME: &str = "sgmse-voicebank";
/// Native artifact arch tag. No upstream artifact is accepted without this
/// Vokra-owned tag and the separate authenticated tensor contract.
pub const ARCH: &str = "sgmse_voicebank";
/// Metadata key written only by a future authenticated converter.
pub const KEY_MANIFEST_STATUS: &str = "vokra.sgmse.manifest_status";
/// Digest of the complete role/name/dtype/shape manifest.  A converter must
/// stamp this only after the VAST inspection has reviewed the safe-loaded
/// checkpoint; a status string alone is never sufficient authentication.
pub const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.sgmse.tensor_manifest_sha256";
/// Array<String> rows encoded as `role|exact-name|dtype-tag|dim,dim,...`.
pub const KEY_TENSOR_MANIFEST: &str = "vokra.sgmse.tensor_manifest";
/// SHA-256 identity of the fixed VoiceBank checkpoint inspected by the worker.
pub const CHECKPOINT_SHA256: &str =
    "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3";
/// The release digest is deliberately uninstantiated until VAST reviews the
/// real safe-loaded checkpoint and role assignment.  This prevents a caller
/// from self-authorizing an arbitrary GGUF by supplying its own digest.
pub const REVIEWED_TENSOR_MANIFEST_SHA256: Option<[u8; 32]> = None;
/// The reviewed role set is held out with the digest so neither names nor
/// required assignment targets can be supplied by an artifact caller.
pub const REVIEWED_TENSOR_ROLES: Option<&'static [SgmseTensorRole]> = None;
/// Status required before a native checkpoint can be opened.
pub const AUTHENTICATED_MANIFEST: &str = "AUTHENTICATED";
/// The typed manifest/binder boundary is implemented, but the native graph is
/// not runnable until checkpoint-specific weights, GroupNorm and OUVE seams
/// are wired through the selected Compute backend; full residual/FIR/
/// resampling score graph, real parity, and device-resident execution remain
/// incomplete.
pub const SGMSE_STATUS: &str = "SOURCE_PLAN_ONLY";

/// The only learned parameter roles accepted by the SGMSE binder.  Tensor
/// names remain checkpoint-specific data from the authenticated manifest; no
/// spelling or prefix is inferred here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SgmseTensorSlot {
    Weight,
    Bias,
    NormGamma,
    NormBeta,
    TimeEmbedding,
    Query,
    Key,
    Value,
    Output,
    FirKernel,
}

/// Typed assignment target for one authenticated checkpoint tensor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SgmseTensorRole {
    FourierFrequencies,
    SigmaFirstProjection,
    SigmaFirstBias,
    SigmaSecondProjection,
    SigmaSecondBias,
    NcsnppStage {
        stage_index: usize,
        kind: NcsnppStageKind,
        block: usize,
        slot: SgmseTensorSlot,
    },
}

impl SgmseTensorRole {
    fn canonical_name(&self) -> String {
        match self {
            Self::FourierFrequencies => "fourier_frequencies".to_owned(),
            Self::SigmaFirstProjection => "sigma_first_projection".to_owned(),
            Self::SigmaFirstBias => "sigma_first_bias".to_owned(),
            Self::SigmaSecondProjection => "sigma_second_projection".to_owned(),
            Self::SigmaSecondBias => "sigma_second_bias".to_owned(),
            Self::NcsnppStage {
                stage_index,
                kind,
                block,
                slot,
            } => format!(
                "stage:{stage_index}:{}:{block}:{}",
                stage_kind_name(*kind),
                slot_name(*slot)
            ),
        }
    }
}

fn stage_kind_name(kind: NcsnppStageKind) -> &'static str {
    match kind {
        NcsnppStageKind::Input => "input",
        NcsnppStageKind::Residual => "residual",
        NcsnppStageKind::Attention => "attention",
        NcsnppStageKind::Downsample => "downsample",
        NcsnppStageKind::Upsample => "upsample",
        NcsnppStageKind::ProgressiveOutput => "progressive_output",
        NcsnppStageKind::ProgressiveInput => "progressive_input",
        NcsnppStageKind::Middle => "middle",
        NcsnppStageKind::Output => "output",
    }
}

fn slot_name(slot: SgmseTensorSlot) -> &'static str {
    match slot {
        SgmseTensorSlot::Weight => "weight",
        SgmseTensorSlot::Bias => "bias",
        SgmseTensorSlot::NormGamma => "norm_gamma",
        SgmseTensorSlot::NormBeta => "norm_beta",
        SgmseTensorSlot::TimeEmbedding => "time_embedding",
        SgmseTensorSlot::Query => "query",
        SgmseTensorSlot::Key => "key",
        SgmseTensorSlot::Value => "value",
        SgmseTensorSlot::Output => "output",
        SgmseTensorSlot::FirKernel => "fir_kernel",
    }
}

fn parse_stage_kind(name: &str) -> Option<NcsnppStageKind> {
    Some(match name {
        "input" => NcsnppStageKind::Input,
        "residual" => NcsnppStageKind::Residual,
        "attention" => NcsnppStageKind::Attention,
        "downsample" => NcsnppStageKind::Downsample,
        "upsample" => NcsnppStageKind::Upsample,
        "progressive_output" => NcsnppStageKind::ProgressiveOutput,
        "progressive_input" => NcsnppStageKind::ProgressiveInput,
        "middle" => NcsnppStageKind::Middle,
        "output" => NcsnppStageKind::Output,
        _ => return None,
    })
}

fn parse_slot(name: &str) -> Option<SgmseTensorSlot> {
    Some(match name {
        "weight" => SgmseTensorSlot::Weight,
        "bias" => SgmseTensorSlot::Bias,
        "norm_gamma" => SgmseTensorSlot::NormGamma,
        "norm_beta" => SgmseTensorSlot::NormBeta,
        "time_embedding" => SgmseTensorSlot::TimeEmbedding,
        "query" => SgmseTensorSlot::Query,
        "key" => SgmseTensorSlot::Key,
        "value" => SgmseTensorSlot::Value,
        "output" => SgmseTensorSlot::Output,
        "fir_kernel" => SgmseTensorSlot::FirKernel,
        _ => return None,
    })
}

fn parse_role(name: &str) -> Option<SgmseTensorRole> {
    Some(match name {
        "fourier_frequencies" => SgmseTensorRole::FourierFrequencies,
        "sigma_first_projection" => SgmseTensorRole::SigmaFirstProjection,
        "sigma_first_bias" => SgmseTensorRole::SigmaFirstBias,
        "sigma_second_projection" => SgmseTensorRole::SigmaSecondProjection,
        "sigma_second_bias" => SgmseTensorRole::SigmaSecondBias,
        _ => {
            let mut fields = name.split(':');
            if fields.next()? != "stage" {
                return None;
            }
            let stage_index = fields.next()?.parse().ok()?;
            let kind = parse_stage_kind(fields.next()?)?;
            let block = fields.next()?.parse().ok()?;
            let slot = parse_slot(fields.next()?)?;
            if fields.next().is_some() {
                return None;
            }
            SgmseTensorRole::NcsnppStage {
                stage_index,
                kind,
                block,
                slot,
            }
        }
    })
}

/// One row from the VAST-authenticated checkpoint contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgmseTensorManifestEntry {
    /// Exact source/checkpoint tensor name, preserved without normalization.
    pub name: String,
    /// Exact GGUF storage type declared by the converter.
    pub dtype: GgmlType,
    /// Exact GGUF dimensions (innermost dimension first).
    pub dimensions: Vec<u64>,
    /// Typed native destination; arbitrary pass-through roles are impossible.
    pub role: SgmseTensorRole,
}

/// Checkpoint-specific contract assembled only by the production binder after
/// the repository's reviewed digest and role set are compiled in.
#[derive(Debug, Clone, PartialEq)]
pub struct SgmseTensorManifest {
    source_revision: String,
    checkpoint_sha256: String,
    graph_config: NcsnppV2Config,
    sampler_config: SgmseConfig,
    required_roles: Vec<SgmseTensorRole>,
    entries: Vec<SgmseTensorManifestEntry>,
}

impl SgmseTensorManifest {
    /// Reads the producer's exact typed rows without assigning roles by
    /// ordinal or name prefix. This parser has no authority to claim source
    /// identity, configuration, or role completeness.
    pub fn from_gguf_metadata(file: &GgufFile) -> Result<Vec<SgmseTensorManifestEntry>> {
        let value = file.get(KEY_TENSOR_MANIFEST).ok_or_else(|| {
            VokraError::ModelLoad("sgmse: typed tensor manifest metadata is missing".to_owned())
        })?;
        let array = value.as_array().ok_or_else(|| {
            VokraError::ModelLoad("sgmse: typed tensor manifest is not an array".to_owned())
        })?;
        if array.element_type != GgufValueType::String || array.values.is_empty() {
            return Err(VokraError::ModelLoad(
                "sgmse: typed tensor manifest must be a non-empty Array<String>".to_owned(),
            ));
        }
        let mut entries = Vec::with_capacity(array.values.len());
        for (index, value) in array.values.iter().enumerate() {
            let encoded = match value {
                GgufMetadataValue::String(value) => value,
                _ => {
                    return Err(VokraError::ModelLoad(format!(
                        "sgmse: typed tensor manifest row {index} is not a string"
                    )));
                }
            };
            let mut fields = encoded.splitn(4, '|');
            let role = fields.next().and_then(parse_role).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "sgmse: typed tensor manifest row {index} has an unknown role"
                ))
            })?;
            let name = fields
                .next()
                .filter(|value| {
                    !value.is_empty()
                        && !value
                            .chars()
                            .any(|character| character == '|' || character.is_control())
                })
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "sgmse: typed tensor manifest row {index} has an empty tensor name"
                    ))
                })?;
            let dtype = fields
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .and_then(|value| GgmlType::from_tag(value).ok())
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "sgmse: typed tensor manifest row {index} has an invalid dtype"
                    ))
                })?;
            let dimensions = fields
                .next()
                .filter(|value| !value.is_empty())
                .map(|value| {
                    value
                        .split(',')
                        .map(|dimension| dimension.parse::<u64>())
                        .collect::<std::result::Result<Vec<_>, _>>()
                })
                .transpose()
                .map_err(|_| {
                    VokraError::ModelLoad(format!(
                        "sgmse: typed tensor manifest row {index} has an invalid shape"
                    ))
                })?
                .filter(|dimensions| !dimensions.is_empty())
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "sgmse: typed tensor manifest row {index} has an empty shape"
                    ))
                })?;
            if dimensions.contains(&0) {
                return Err(VokraError::ModelLoad(format!(
                    "sgmse: typed tensor manifest row {index} has a zero dimension"
                )));
            }
            entries.push(SgmseTensorManifestEntry {
                name: name.to_owned(),
                dtype,
                dimensions,
                role,
            });
        }
        if entries.iter().any(|entry| entry.name.contains('|')) {
            return Err(VokraError::ModelLoad(
                "sgmse: typed tensor name contains the manifest separator".to_owned(),
            ));
        }
        Ok(entries)
    }

    /// Validates identity, exact role completeness, and the canonical digest
    /// supplied by the release's VAST review.  The expected digest is kept
    /// outside the mutable GGUF metadata so a self-stamped file cannot close
    /// the gate by itself.
    fn validate(&self, plan: &NcsnppV2GraphPlan, expected_digest: [u8; 32]) -> Result<()> {
        if expected_digest == [0; 32] {
            return Err(VokraError::ModelLoad(
                "sgmse: reviewed tensor manifest digest is missing".to_owned(),
            ));
        }
        if self.source_revision != SOURCE_REVISION
            || self.checkpoint_sha256 != CHECKPOINT_SHA256
            || self.graph_config != plan.config
            || self.sampler_config != SgmseConfig::voicebank()
        {
            return Err(VokraError::ModelLoad(
                "sgmse: authenticated manifest source identity or configuration mismatch"
                    .to_owned(),
            ));
        }
        if self.entries.is_empty() || self.required_roles.is_empty() {
            return Err(VokraError::ModelLoad(
                "sgmse: authenticated tensor manifest is empty".to_owned(),
            ));
        }
        let required: BTreeSet<_> = self.required_roles.iter().cloned().collect();
        if required.len() != self.required_roles.len() || self.entries.len() != required.len() {
            return Err(VokraError::ModelLoad(
                "sgmse: tensor manifest role set is duplicate or incomplete".to_owned(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut roles = BTreeSet::new();
        for entry in &self.entries {
            if entry.name.is_empty()
                || !matches!(entry.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
                || entry.dimensions.is_empty()
                || !names.insert(entry.name.as_str())
                || !roles.insert(entry.role.clone())
                || !required.contains(&entry.role)
            {
                return Err(VokraError::ModelLoad(
                    "sgmse: tensor manifest has duplicate, unknown, or unsupported entry"
                        .to_owned(),
                ));
            }
            if let SgmseTensorRole::NcsnppStage {
                stage_index,
                kind,
                block,
                ..
            } = &entry.role
            {
                let Some(stage) = plan.stages.get(*stage_index) else {
                    return Err(VokraError::ModelLoad(
                        "sgmse: tensor role references a missing graph stage".to_owned(),
                    ));
                };
                if stage.kind != *kind || stage.block != *block {
                    return Err(VokraError::ModelLoad(
                        "sgmse: tensor role graph stage metadata mismatches source plan".to_owned(),
                    ));
                }
            }
        }
        if roles != required {
            return Err(VokraError::ModelLoad(
                "sgmse: tensor manifest is missing a required typed role".to_owned(),
            ));
        }
        if self.canonical_sha256() != expected_digest {
            return Err(VokraError::ModelLoad(
                "sgmse: tensor manifest digest does not match the reviewed release digest"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Computes the stable digest over role, exact name, dtype, and shape.
    #[must_use]
    pub fn canonical_sha256(&self) -> [u8; 32] {
        let mut rows: Vec<_> = self.entries.iter().collect();
        rows.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        let mut bytes = Vec::new();
        for row in rows {
            bytes.extend_from_slice(row.role.canonical_name().as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(row.name.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(&row.dtype.tag().to_le_bytes());
            bytes.extend_from_slice(&(row.dimensions.len() as u64).to_le_bytes());
            for dimension in &row.dimensions {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        crate::strict_checkpoint::sha256_bytes(&bytes)
    }
}

/// Bound, finite graph operands produced only after the complete manifest is
/// checked.  This is a typed assignment boundary, not an executable graph;
/// unsupported residual/FIR/resampling kernels still keep SGMSE closed.
#[derive(Debug, Clone)]
pub struct SgmseGraphWeights {
    plan: NcsnppV2GraphPlan,
    tensors: Vec<(SgmseTensorRole, String, Vec<u64>, Vec<f32>)>,
}

impl SgmseGraphWeights {
    /// Binds every declared tensor and rejects any missing/extra descriptor.
    pub fn bind_authenticated(file: &GgufFile) -> Result<Self> {
        let (Some(expected_digest), Some(reviewed_roles)) =
            (REVIEWED_TENSOR_MANIFEST_SHA256, REVIEWED_TENSOR_ROLES)
        else {
            return Err(VokraError::ModelLoad(
                "sgmse: AUTHENTICATED_MANIFEST_REQUIRED until VAST-reviewed tensor digest is compiled in"
                    .to_owned(),
            ));
        };
        let plan = NcsnppV2GraphPlan::from_config(NcsnppV2Config::source_default())?;
        let manifest = SgmseTensorManifest {
            source_revision: SOURCE_REVISION.to_owned(),
            checkpoint_sha256: CHECKPOINT_SHA256.to_owned(),
            graph_config: plan.config.clone(),
            sampler_config: SgmseConfig::voicebank(),
            required_roles: reviewed_roles.to_vec(),
            entries: Self::from_gguf_metadata(file)?,
        };
        manifest.validate(&plan, expected_digest)?;
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str());
        if arch != Some(ARCH) {
            return Err(VokraError::ModelLoad(
                "sgmse: GGUF model arch is not the authenticated SGMSE arch".to_owned(),
            ));
        }
        if file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|value| value.as_str())
            != Some(MODEL_NAME)
        {
            return Err(VokraError::ModelLoad(
                "sgmse: GGUF model name is not the authenticated VoiceBank model".to_owned(),
            ));
        }
        let status = file
            .get(KEY_MANIFEST_STATUS)
            .and_then(|value| value.as_str());
        if status != Some(AUTHENTICATED_MANIFEST) {
            return Err(VokraError::ModelLoad(
                "sgmse: AUTHENTICATED_MANIFEST_REQUIRED".to_owned(),
            ));
        }
        let stamped = file
            .get(KEY_TENSOR_MANIFEST_SHA256)
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "sgmse: tensor manifest digest metadata is missing".to_owned(),
                )
            })?;
        if stamped != hex_digest(&expected_digest) {
            return Err(VokraError::ModelLoad(
                "sgmse: tensor manifest digest metadata mismatch".to_owned(),
            ));
        }
        if file.tensors().len() != manifest.entries.len() {
            return Err(VokraError::ModelLoad(
                "sgmse: GGUF tensor count differs from authenticated manifest".to_owned(),
            ));
        }
        let mut tensors = Vec::with_capacity(manifest.entries.len());
        for entry in &manifest.entries {
            let info = file.tensor_info(&entry.name).ok_or_else(|| {
                VokraError::ModelLoad(format!("sgmse: missing tensor {:?}", entry.name))
            })?;
            if info.dtype != entry.dtype || info.dimensions != entry.dimensions {
                return Err(VokraError::ModelLoad(format!(
                    "sgmse: tensor {:?} dtype/shape differs from authenticated manifest",
                    entry.name
                )));
            }
            let values = file.tensor_f32(&entry.name).map_err(|error| {
                VokraError::ModelLoad(format!(
                    "sgmse: tensor {:?} decode failed: {error}",
                    entry.name
                ))
            })?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::ModelLoad(format!(
                    "sgmse: tensor {:?} contains a non-finite value",
                    entry.name
                )));
            }
            tensors.push((
                entry.role.clone(),
                entry.name.clone(),
                entry.dimensions.clone(),
                values,
            ));
        }
        let bound = Self { plan, tensors };
        bound.validate_before_dispatch()?;
        Ok(bound)
    }

    /// Rechecks all operands immediately before a future backend dispatch.
    /// This catches corruption introduced after bind (including late-loaded
    /// or device-transfer operands) rather than relying on constructor-time
    /// validation alone.
    pub fn validate_before_dispatch(&self) -> Result<()> {
        self.plan.config.validate()?;
        if self.tensors.is_empty()
            || self.tensors.iter().any(|(_, _, dimensions, values)| {
                let shape_count = dimensions.iter().try_fold(1usize, |count, &dimension| {
                    usize::try_from(dimension).ok()?.checked_mul(count)
                });
                dimensions.is_empty()
                    || values.is_empty()
                    || shape_count != Some(values.len())
                    || values.iter().any(|v| !v.is_finite())
            })
        {
            return Err(VokraError::ModelLoad(
                "sgmse: graph operands are incomplete or non-finite".to_owned(),
            ));
        }
        Ok(())
    }

    /// Returns the bound operand for a typed role, if present.
    #[must_use]
    pub fn tensor_for_role(&self, role: &SgmseTensorRole) -> Option<&[f32]> {
        self.tensors
            .iter()
            .find(|(bound_role, _, _, _)| bound_role == role)
            .map(|(_, _, _, values)| values.as_slice())
    }
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from(DIGITS[(byte >> 4) as usize]));
        output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    output
}

/// Source-authenticated frontend and sampler configuration for SGMSE+.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgmseConfig {
    /// Audio sample rate from the pinned VoiceBank hyperparameters.
    pub sample_rate: u32,
    /// FFT size; 510 yields 256 one-sided bins in the source.
    pub n_fft: usize,
    /// Frame hop in samples.
    pub hop_length: usize,
    /// Complex magnitude exponent in the source exponent transform.
    pub spec_abs_exponent: f32,
    /// Complex coefficient scale in the source exponent transform.
    pub spec_factor: f32,
    /// OUVE predictor/corrector integration steps.
    pub steps: usize,
    /// Minimum reverse-time endpoint (`t_eps`).
    pub t_eps: f32,
    /// Annealed Langevin corrector SNR.
    pub snr: f32,
}

impl SgmseConfig {
    /// Values from the pinned `SpecsDataModule` / VoiceBank hyperparameters.
    #[must_use]
    pub const fn voicebank() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 510,
            hop_length: 128,
            spec_abs_exponent: 0.5,
            spec_factor: 0.15,
            steps: 30,
            t_eps: 0.03,
            snr: 0.5,
        }
    }

    /// Validates source configuration before any allocation or dispatch.
    pub fn validate(self) -> Result<()> {
        if self.sample_rate == 0 || self.n_fft == 0 || self.hop_length == 0 {
            return Err(VokraError::InvalidArgument(
                "sgmse: sample rate, n_fft, and hop_length must be non-zero".to_owned(),
            ));
        }
        if self.hop_length >= self.n_fft
            || self.steps == 0
            || !self.spec_abs_exponent.is_finite()
            || self.spec_abs_exponent <= 0.0
            || !self.spec_factor.is_finite()
            || self.spec_factor <= 0.0
            || !self.t_eps.is_finite()
            || !(0.0..1.0).contains(&self.t_eps)
            || !self.snr.is_finite()
            || self.snr < 0.0
        {
            return Err(VokraError::InvalidArgument(
                "sgmse: invalid STFT, transform, or sampler configuration".to_owned(),
            ));
        }
        Ok(())
    }

    /// Source STFT attributes: centered periodic Hann, reflected padding, and
    /// one-sided real input, matching `SpecsDataModule.istft_kwargs`.
    pub fn stft_attrs(self) -> Result<StftAttrs> {
        self.validate()?;
        Ok(StftAttrs {
            n_fft: self.n_fft,
            hop_length: self.hop_length,
            win_length: self.n_fft,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            pad_mode: PadMode::Reflect,
            normalization: vokra_core::ir::graph::Normalization::Backward,
            causal: false,
            real_input: true,
        })
    }

    /// Matching iSTFT attributes for the source transform.
    pub fn istft_attrs(self, length: Option<usize>) -> Result<IstftAttrs> {
        self.validate()?;
        Ok(IstftAttrs {
            n_fft: self.n_fft,
            hop_length: self.hop_length,
            win_length: self.n_fft,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            normalization: vokra_core::ir::graph::Normalization::Backward,
            real_input: true,
            length,
            normalize_window: true,
        })
    }
}

/// The source NCSN++ v2 graph configuration. Dimensions and tensor bindings
/// are supplied by the authenticated checkpoint manifest; these fields only
/// describe the source architecture defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct NcsnppV2Config {
    /// Real channels entering the conditional score network (x/y complex
    /// parts concatenated).
    pub input_channels: usize,
    /// Real channels emitted by the complex score network (real/imaginary).
    pub output_channels: usize,
    /// Spectrogram frame resolution (`num_frames` in VoiceBank source).
    pub input_resolution: usize,
    /// Base feature width (`nf` in the source).
    pub nf: usize,
    /// Per-resolution channel multipliers.
    pub ch_mult: Vec<usize>,
    /// Residual blocks per resolution.
    pub num_res_blocks: usize,
    /// Spatial resolutions receiving self-attention.
    pub attention_resolutions: Vec<usize>,
    /// Source graph uses Gaussian Fourier noise embedding.
    pub fourier_embedding: bool,
    /// Source graph uses BigGAN++ residual blocks.
    pub biggan_resblocks: bool,
    /// GroupNorm maximum groups; source uses `min(channels / 4, 32)`.
    pub group_norm_max_groups: usize,
    pub group_norm_eps: f32,
    /// Source graph uses output-skip pyramid and input-skip pyramid paths.
    pub progressive_output_skip: bool,
    pub progressive_input_skip: bool,
    /// Source FIR resampling taps, retained as architecture metadata.
    pub fir_kernel: [usize; 4],
    /// Source residual/attention output rescaling.
    pub skip_rescale: bool,
}

/// A source-level NCSN++ stage. This is a topology description, not a weight
/// manifest: the latter must be supplied by the safe-loader before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NcsnppStageKind {
    /// Initial four-real-channel projection.
    Input,
    /// BigGAN-style residual block with a sigma embedding.
    Residual,
    /// Self-attention at the source-selected resolution.
    Attention,
    /// Strided downsampling between resolutions.
    Downsample,
    /// Strided upsampling between resolutions.
    Upsample,
    /// Progressive output-skip projection/accumulation at a resolution.
    ProgressiveOutput,
    /// Progressive input-skip downsample/concatenation.
    ProgressiveInput,
    /// The middle residual/attention/residual sequence.
    Middle,
    /// Final projection back to the two real score channels.
    Output,
}

/// One stage in the source NCSN++ v2 execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcsnppStage {
    /// Source resolution in spectrogram pixels (the caller supplies actual
    /// dimensions; this is only the architecture's nominal resolution).
    pub resolution: usize,
    /// Source channel multiplier for this stage.
    pub channel_multiplier: usize,
    /// Stage role.
    pub kind: NcsnppStageKind,
    /// One-based residual-block index, or zero for non-residual stages.
    pub block: usize,
}

/// Deterministic topology generated from the pinned NCSN++ v2 defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct NcsnppV2GraphPlan {
    /// Validated source options used to produce the plan.
    pub config: NcsnppV2Config,
    /// Ordered down/middle/up stages, with no checkpoint-specific bindings.
    pub stages: Vec<NcsnppStage>,
}

/// Upper-bounds the stage vector without allowing any source-loop arithmetic
/// to wrap. The bound includes both residual paths, attention/pyramid stages,
/// and input/middle/output sentinels; reserving it is only an allocation hint.
fn checked_stage_capacity(config: &NcsnppV2Config) -> Result<usize> {
    let up_blocks = config.num_res_blocks.checked_add(1).ok_or_else(|| {
        VokraError::InvalidArgument("sgmse ncsnpp: up block count overflow".to_owned())
    })?;
    let levels = config.ch_mult.len();
    let down_residuals = levels.checked_mul(config.num_res_blocks).ok_or_else(|| {
        VokraError::InvalidArgument("sgmse ncsnpp: down stage count overflow".to_owned())
    })?;
    let up_residuals = levels.checked_mul(up_blocks).ok_or_else(|| {
        VokraError::InvalidArgument("sgmse ncsnpp: up stage count overflow".to_owned())
    })?;
    let pyramid = levels.checked_mul(6).ok_or_else(|| {
        VokraError::InvalidArgument("sgmse ncsnpp: pyramid stage count overflow".to_owned())
    })?;
    down_residuals
        .checked_add(up_residuals)
        .and_then(|count| count.checked_add(pyramid))
        .and_then(|count| count.checked_add(4))
        .ok_or_else(|| VokraError::InvalidArgument("sgmse ncsnpp: stage count overflow".to_owned()))
}

impl NcsnppV2GraphPlan {
    /// Builds the source order from `NCSNpp`'s down, middle, and up loops.
    pub fn from_config(config: NcsnppV2Config) -> Result<Self> {
        config.validate()?;
        let stage_capacity = checked_stage_capacity(&config)?;
        let mut stages = Vec::with_capacity(stage_capacity);
        stages.push(NcsnppStage {
            resolution: config.input_resolution,
            channel_multiplier: 1,
            kind: NcsnppStageKind::Input,
            block: 0,
        });
        for (level, &multiplier) in config.ch_mult.iter().enumerate() {
            let resolution = config
                .input_resolution
                .checked_shr(level as u32)
                .ok_or_else(|| {
                    VokraError::InvalidArgument("sgmse ncsnpp: resolution overflow".to_owned())
                })?;
            for block in 1..=config.num_res_blocks {
                stages.push(NcsnppStage {
                    resolution,
                    channel_multiplier: multiplier,
                    kind: NcsnppStageKind::Residual,
                    block,
                });
                if config.attention_resolutions.contains(&resolution) {
                    stages.push(NcsnppStage {
                        resolution,
                        channel_multiplier: multiplier,
                        kind: NcsnppStageKind::Attention,
                        block,
                    });
                }
            }
            if level + 1 < config.ch_mult.len() {
                stages.push(NcsnppStage {
                    resolution,
                    channel_multiplier: multiplier,
                    kind: NcsnppStageKind::Downsample,
                    block: 0,
                });
                stages.push(NcsnppStage {
                    resolution: resolution / 2,
                    channel_multiplier: config.ch_mult[level + 1],
                    kind: NcsnppStageKind::ProgressiveInput,
                    block: 0,
                });
            }
        }
        stages.push(NcsnppStage {
            resolution: config
                .input_resolution
                .checked_shr((config.ch_mult.len() - 1) as u32)
                .ok_or_else(|| {
                    VokraError::InvalidArgument("sgmse ncsnpp: resolution overflow".to_owned())
                })?,
            channel_multiplier: *config.ch_mult.last().ok_or_else(|| {
                VokraError::InvalidArgument("sgmse ncsnpp: missing channel levels".to_owned())
            })?,
            kind: NcsnppStageKind::Middle,
            block: 1,
        });
        stages.push(NcsnppStage {
            resolution: config
                .input_resolution
                .checked_shr((config.ch_mult.len() - 1) as u32)
                .ok_or_else(|| {
                    VokraError::InvalidArgument("sgmse ncsnpp: resolution overflow".to_owned())
                })?,
            channel_multiplier: *config.ch_mult.last().ok_or_else(|| {
                VokraError::InvalidArgument("sgmse ncsnpp: missing channel levels".to_owned())
            })?,
            kind: NcsnppStageKind::Attention,
            block: 0,
        });
        stages.push(NcsnppStage {
            resolution: config
                .input_resolution
                .checked_shr((config.ch_mult.len() - 1) as u32)
                .ok_or_else(|| {
                    VokraError::InvalidArgument("sgmse ncsnpp: resolution overflow".to_owned())
                })?,
            channel_multiplier: *config.ch_mult.last().ok_or_else(|| {
                VokraError::InvalidArgument("sgmse ncsnpp: missing channel levels".to_owned())
            })?,
            kind: NcsnppStageKind::Middle,
            block: 2,
        });
        for (level, &multiplier) in config.ch_mult.iter().enumerate().rev() {
            let resolution = config
                .input_resolution
                .checked_shr(level as u32)
                .ok_or_else(|| {
                    VokraError::InvalidArgument("sgmse ncsnpp: resolution overflow".to_owned())
                })?;
            // NCSN++ uses one additional block on the up path because each
            // level consumes one skip tensor from the down path.
            let up_blocks = config.num_res_blocks.checked_add(1).ok_or_else(|| {
                VokraError::InvalidArgument("sgmse ncsnpp: up block count overflow".to_owned())
            })?;
            for block in 1..=up_blocks {
                stages.push(NcsnppStage {
                    resolution,
                    channel_multiplier: multiplier,
                    kind: NcsnppStageKind::Residual,
                    block,
                });
            }
            if config.attention_resolutions.contains(&resolution) {
                stages.push(NcsnppStage {
                    resolution,
                    channel_multiplier: multiplier,
                    kind: NcsnppStageKind::Attention,
                    block: 0,
                });
            }
            stages.push(NcsnppStage {
                resolution,
                channel_multiplier: multiplier,
                kind: NcsnppStageKind::ProgressiveOutput,
                block: 0,
            });
            if level > 0 {
                stages.push(NcsnppStage {
                    resolution,
                    channel_multiplier: multiplier,
                    kind: NcsnppStageKind::Upsample,
                    block: 0,
                });
            }
        }
        stages.push(NcsnppStage {
            resolution: config.input_resolution,
            channel_multiplier: 1,
            kind: NcsnppStageKind::Output,
            block: 0,
        });
        Ok(Self { config, stages })
    }
}

/// Fixed Fourier embedding helper used by the source's sigma conditioning.
/// Frequencies are caller-supplied because the upstream projection is a
/// fixed module buffer and must never be invented or regenerated by a binder.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierSigmaEmbedding {
    frequencies: Vec<f32>,
}

impl FourierSigmaEmbedding {
    /// Creates an embedding from the exact projection buffer recovered by the
    /// safe-loader or an independent synthetic oracle.
    pub fn new(frequencies: Vec<f32>) -> Result<Self> {
        if frequencies.is_empty() || frequencies.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "sgmse Fourier embedding requires finite non-empty frequencies".to_owned(),
            ));
        }
        Ok(Self { frequencies })
    }

    /// Computes `[sin(2π log_sigma W), cos(2π log_sigma W)]`, matching the
    /// source `GaussianFourierProjection` convention.
    pub fn forward(&self, log_sigma: f32, out: &mut [f32]) -> Result<()> {
        if !log_sigma.is_finite() || out.len() != self.frequencies.len() * 2 {
            return Err(VokraError::InvalidArgument(
                "sgmse Fourier embedding input/output shape or value is invalid".to_owned(),
            ));
        }
        let two_pi = 2.0 * core::f32::consts::PI;
        for (index, &frequency) in self.frequencies.iter().enumerate() {
            let phase = two_pi * log_sigma * frequency;
            if !phase.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "sgmse Fourier embedding phase is not finite".to_owned(),
                ));
            }
            out[index] = phase.sin();
            out[index + self.frequencies.len()] = phase.cos();
        }
        Ok(())
    }
}

/// Sigma conditioning dense projection. The projection matrix is supplied by
/// the authenticated tensor binder; no checkpoint dimensions are assumed.
/// The dense and activation calls go through [`Compute`] so a Metal-bound
/// score graph cannot silently execute learned work on the host.
pub struct SigmaConditioner {
    embedding: FourierSigmaEmbedding,
    first_projection: Vec<f32>,
    first_bias: Vec<f32>,
    second_projection: Vec<f32>,
    second_bias: Vec<f32>,
    nf: usize,
}

impl SigmaConditioner {
    /// Builds the source two-layer conditioning MLP: biased
    /// `Linear(2*nf,4*nf)`, SiLU, then biased `Linear(4*nf,4*nf)`.
    /// Matrices are row-major and all tensors are supplied by the caller so
    /// the checkpoint binder remains shape-authenticated.
    pub fn new(
        nf: usize,
        embedding: FourierSigmaEmbedding,
        first_projection: Vec<f32>,
        first_bias: Vec<f32>,
        second_projection: Vec<f32>,
        second_bias: Vec<f32>,
    ) -> Result<Self> {
        let embedding_width = nf.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument(
                "sgmse sigma conditioner embedding width overflow".to_owned(),
            )
        })?;
        let output_width = nf.checked_mul(4).ok_or_else(|| {
            VokraError::InvalidArgument("sgmse sigma conditioner output width overflow".to_owned())
        })?;
        let first_len = embedding_width.checked_mul(output_width).ok_or_else(|| {
            VokraError::InvalidArgument(
                "sgmse sigma conditioner projection size overflow".to_owned(),
            )
        })?;
        let second_len = output_width.checked_mul(output_width).ok_or_else(|| {
            VokraError::InvalidArgument(
                "sgmse sigma conditioner projection size overflow".to_owned(),
            )
        })?;
        if nf == 0
            || embedding.frequencies.len() != nf
            || first_projection.len() != first_len
            || first_bias.len() != output_width
            || second_projection.len() != second_len
            || second_bias.len() != output_width
            || first_projection
                .iter()
                .chain(&first_bias)
                .chain(&second_projection)
                .chain(&second_bias)
                .any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "sgmse sigma conditioner projection shape or values are invalid".to_owned(),
            ));
        }
        Ok(Self {
            embedding,
            first_projection,
            first_bias,
            second_projection,
            second_bias,
            nf,
        })
    }

    /// Projects log-sigma through the source conditioning MLP.
    pub fn forward(&self, compute: &Compute, log_sigma: f32, out: &mut [f32]) -> Result<()> {
        let output_width = self.nf * 4;
        if out.len() != output_width {
            return Err(VokraError::InvalidArgument(
                "sgmse sigma conditioner output width mismatch".to_owned(),
            ));
        }
        let mut embedding = vec![0.0; self.nf * 2];
        self.embedding.forward(log_sigma, &mut embedding)?;
        let mut hidden = vec![0.0; output_width];
        compute.gemm_f32(
            1,
            output_width,
            embedding.len(),
            &embedding,
            &self.first_projection,
            Some(&self.first_bias),
            &mut hidden,
        )?;
        let activated = hidden.clone();
        compute.silu_f32(&activated, &mut hidden)?;
        compute.gemm_f32(
            1,
            output_width,
            output_width,
            &hidden,
            &self.second_projection,
            Some(&self.second_bias),
            out,
        )
    }
}

/// 1×1 self-attention projections used by the source attention block. The four
/// NIN matrices and their biases are mapped by the authenticated binder and
/// are deliberately represented without checkpoint names or fixed dimensions.
pub struct NcsnppAttentionWeights {
    channels: usize,
    norm_groups: usize,
    norm_eps: f32,
    norm_gamma: Vec<f32>,
    norm_beta: Vec<f32>,
    q: Vec<f32>,
    q_bias: Vec<f32>,
    k: Vec<f32>,
    k_bias: Vec<f32>,
    v: Vec<f32>,
    v_bias: Vec<f32>,
    out: Vec<f32>,
    out_bias: Vec<f32>,
    skip_rescale: bool,
}

impl NcsnppAttentionWeights {
    /// Creates weights with row-major `[channels, channels]` matrices.
    pub fn new(
        channels: usize,
        norm_groups: usize,
        norm_eps: f32,
        norm_gamma: Vec<f32>,
        norm_beta: Vec<f32>,
        q: Vec<f32>,
        q_bias: Vec<f32>,
        k: Vec<f32>,
        k_bias: Vec<f32>,
        v: Vec<f32>,
        v_bias: Vec<f32>,
        out: Vec<f32>,
        out_bias: Vec<f32>,
        skip_rescale: bool,
    ) -> Result<Self> {
        let expected = channels.checked_mul(channels).ok_or_else(|| {
            VokraError::InvalidArgument("sgmse attention channel size overflow".to_owned())
        })?;
        if channels == 0
            || norm_groups == 0
            || channels % norm_groups != 0
            || !norm_eps.is_finite()
            || norm_eps <= 0.0
            || norm_gamma.len() != channels
            || norm_beta.len() != channels
            || q_bias.len() != channels
            || k_bias.len() != channels
            || v_bias.len() != channels
            || out_bias.len() != channels
            || [q.len(), k.len(), v.len(), out.len()]
                .into_iter()
                .any(|length| length != expected)
            || q.iter()
                .chain(&norm_gamma)
                .chain(&norm_beta)
                .chain(&q_bias)
                .chain(&k)
                .chain(&k_bias)
                .chain(&v)
                .chain(&v_bias)
                .chain(&out)
                .chain(&out_bias)
                .any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "sgmse attention weight shape or values are invalid".to_owned(),
            ));
        }
        Ok(Self {
            channels,
            norm_groups,
            norm_eps,
            norm_gamma,
            norm_beta,
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            out,
            out_bias,
            skip_rescale,
        })
    }

    /// Runs flattened spatial self-attention through the selected Compute
    /// backend. `input`/`output` use the source channel-major
    /// `[channels, positions]` layout; an explicit transpose surrounds the
    /// row-major GEMM attention core.
    pub fn forward(&self, compute: &Compute, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.is_empty() || input.len() % self.channels != 0 || output.len() != input.len() {
            return Err(VokraError::InvalidArgument(
                "sgmse attention input/output shape is invalid".to_owned(),
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "sgmse attention input contains a non-finite value".to_owned(),
            ));
        }
        let positions = input.len() / self.channels;
        let mut normalized = vec![0.0; input.len()];
        compute.group_norm_groups_f32(
            input,
            &mut normalized,
            self.channels,
            positions,
            self.norm_groups,
            &self.norm_gamma,
            &self.norm_beta,
            self.norm_eps,
        )?;
        let mut normalized_rows = vec![0.0; input.len()];
        for channel in 0..self.channels {
            for position in 0..positions {
                normalized_rows[position * self.channels + channel] =
                    normalized[channel * positions + position];
            }
        }
        let mut q = vec![0.0; input.len()];
        let mut k = vec![0.0; input.len()];
        let mut v = vec![0.0; input.len()];
        compute.gemm_f32(
            positions,
            self.channels,
            self.channels,
            &normalized_rows,
            &self.q,
            Some(&self.q_bias),
            &mut q,
        )?;
        compute.gemm_f32(
            positions,
            self.channels,
            self.channels,
            &normalized_rows,
            &self.k,
            Some(&self.k_bias),
            &mut k,
        )?;
        compute.gemm_f32(
            positions,
            self.channels,
            self.channels,
            &normalized_rows,
            &self.v,
            Some(&self.v_bias),
            &mut v,
        )?;
        let mut k_transposed = vec![0.0; input.len()];
        for row in 0..positions {
            for channel in 0..self.channels {
                k_transposed[channel * positions + row] = k[row * self.channels + channel];
            }
        }
        let mut scores = vec![0.0; positions * positions];
        compute.gemm_f32(
            positions,
            positions,
            self.channels,
            &q,
            &k_transposed,
            None,
            &mut scores,
        )?;
        let scale = (self.channels as f32).sqrt().recip();
        if !scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse attention scale is not finite".to_owned(),
            ));
        }
        for score in &mut scores {
            *score *= scale;
        }
        if scores.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "sgmse attention scores contain a non-finite value".to_owned(),
            ));
        }
        let mut probabilities = vec![0.0; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, positions, positions)?;
        let mut context = vec![0.0; input.len()];
        compute.gemm_f32(
            positions,
            self.channels,
            positions,
            &probabilities,
            &v,
            None,
            &mut context,
        )?;
        compute.gemm_f32(
            positions,
            self.channels,
            self.channels,
            &context,
            &self.out,
            Some(&self.out_bias),
            output,
        )?;
        let output_rows = output.to_vec();
        for channel in 0..self.channels {
            for position in 0..positions {
                output[channel * positions + position] =
                    output_rows[position * self.channels + channel];
            }
        }
        if self.skip_rescale {
            let scale = 2.0f32.sqrt().recip();
            for (value, &residual) in output.iter_mut().zip(input) {
                *value = (*value + residual) * scale;
            }
        } else {
            for (value, &residual) in output.iter_mut().zip(input) {
                *value += residual;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "sgmse attention output contains a non-finite value".to_owned(),
            ));
        }
        Ok(())
    }
}

impl NcsnppV2Config {
    /// Defaults transcribed from `ncsnpp.py` at the pinned revision.
    #[must_use]
    pub fn source_default() -> Self {
        Self {
            input_channels: 4,
            output_channels: 2,
            input_resolution: 256,
            nf: 128,
            ch_mult: vec![1, 1, 2, 2, 2, 2, 2],
            num_res_blocks: 2,
            attention_resolutions: vec![16],
            fourier_embedding: true,
            biggan_resblocks: true,
            group_norm_max_groups: 32,
            group_norm_eps: 1.0e-6,
            progressive_output_skip: true,
            progressive_input_skip: true,
            fir_kernel: [1, 3, 3, 1],
            skip_rescale: true,
        }
    }

    /// Rejects malformed source graph options before a manifest can bind.
    pub fn validate(&self) -> Result<()> {
        // `checked_shr` below is the source-level resolution ladder. Reject a
        // level count that would truncate to zero, and reject the up-path
        // skip-block addition before constructing a plan.
        let levels_have_resolution = self.ch_mult.len() <= usize::BITS as usize
            && self.ch_mult.iter().enumerate().all(|(level, _)| {
                self.input_resolution
                    .checked_shr(level as u32)
                    .is_some_and(|resolution| resolution != 0)
            });
        let stage_capacity_fits = checked_stage_capacity(self).is_ok();
        if self.input_channels != 4
            || self.output_channels != 2
            || self.input_resolution == 0
            || !self.input_resolution.is_power_of_two()
            || self.nf == 0
            || self.ch_mult.is_empty()
            || !levels_have_resolution
            || !stage_capacity_fits
            || self.ch_mult.iter().any(|&value| value == 0)
            || self.num_res_blocks == 0
            || self.num_res_blocks.checked_add(1).is_none()
            || !self.fourier_embedding
            || !self.biggan_resblocks
            || self.group_norm_max_groups == 0
            || !self.group_norm_eps.is_finite()
            || self.group_norm_eps <= 0.0
            || !self.progressive_output_skip
            || !self.progressive_input_skip
            || self.fir_kernel != [1, 3, 3, 1]
            || !self.skip_rescale
        {
            return Err(VokraError::InvalidArgument(
                "sgmse ncsnpp_v2: unsupported or incomplete source graph configuration".to_owned(),
            ));
        }
        Ok(())
    }

    /// Computes the source `AttnBlockpp` GroupNorm count: `min(C / 4, 32)`.
    pub fn group_norm_groups(&self, channels: usize) -> Result<usize> {
        self.validate()?;
        if channels < 4 {
            return Err(VokraError::InvalidArgument(
                "sgmse GroupNorm requires at least four channels".to_owned(),
            ));
        }
        Ok((channels / 4).min(self.group_norm_max_groups))
    }
}

/// Converts one real PCM signal into the source's transformed complex state.
#[derive(Debug, Clone)]
pub struct SgmseFrontend {
    config: SgmseConfig,
}

impl SgmseFrontend {
    /// Builds the fixed VoiceBank frontend.
    pub fn new(config: SgmseConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Computes centered STFT and applies the source exponent transform.
    pub fn forward(&self, pcm: &[f32]) -> Result<Spectrogram> {
        let raw = stft(pcm, &self.config.stft_attrs()?)?;
        transform_forward(raw, self.config.spec_factor, self.config.spec_abs_exponent)
    }

    /// Inverts a transformed spectrogram with the source matching iSTFT.
    pub fn inverse(&self, transformed: &Spectrogram, length: Option<usize>) -> Result<Vec<f32>> {
        let raw = transform_inverse(
            transformed,
            self.config.spec_factor,
            self.config.spec_abs_exponent,
        )?;
        istft(&raw, &self.config.istft_attrs(length)?)
    }
}

fn transform_forward(mut spec: Spectrogram, factor: f32, exponent: f32) -> Result<Spectrogram> {
    if !factor.is_finite() || factor <= 0.0 || !exponent.is_finite() || exponent <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "sgmse transform forward requires finite positive parameters".to_owned(),
        ));
    }
    for (re, im) in spec.re.iter_mut().zip(&mut spec.im) {
        let magnitude = (*re * *re + *im * *im).sqrt();
        if !magnitude.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse transform forward received a non-finite spectrogram".to_owned(),
            ));
        }
        let scale = if magnitude == 0.0 {
            0.0
        } else {
            factor * magnitude.powf(exponent - 1.0)
        };
        if !scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse transform forward produced a non-finite scale".to_owned(),
            ));
        }
        *re *= scale;
        *im *= scale;
        if !re.is_finite() || !im.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse transform forward produced a non-finite value".to_owned(),
            ));
        }
    }
    Ok(spec)
}

fn transform_inverse(spec: &Spectrogram, factor: f32, exponent: f32) -> Result<Spectrogram> {
    if !factor.is_finite() || factor <= 0.0 || !exponent.is_finite() || exponent <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "sgmse transform inverse requires finite positive parameters".to_owned(),
        ));
    }
    let mut raw = spec.clone();
    let inverse_exponent = 1.0 / exponent;
    for (re, im) in raw.re.iter_mut().zip(&mut raw.im) {
        let magnitude = (*re * *re + *im * *im).sqrt();
        if !magnitude.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse transform inverse received a non-finite spectrogram".to_owned(),
            ));
        }
        let scale = if magnitude == 0.0 {
            0.0
        } else {
            (magnitude / factor).powf(inverse_exponent - 1.0) / factor
        };
        if !scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse transform inverse produced a non-finite scale".to_owned(),
            ));
        }
        *re *= scale;
        *im *= scale;
        if !re.is_finite() || !im.is_finite() {
            return Err(VokraError::InvalidArgument(
                "sgmse transform inverse produced a non-finite value".to_owned(),
            ));
        }
    }
    Ok(raw)
}

/// Score callback implemented by the authenticated NCSN++ binder.
pub trait NcsnppScore {
    /// Writes a flat real score for `state` conditioned on `condition` at time
    /// `t`. The callback must preserve the supplied length.
    fn score(&mut self, state: &[f32], condition: &[f32], t: f32, out: &mut [f32]) -> Result<()>;
}

/// Deterministic noise provider used by the sampler and independent fixtures.
pub trait SgmseNoise {
    /// Fills the initial prior noise. The default preserves legacy callers by
    /// delegating to `fill` with a reserved step marker.
    fn fill_prior(&mut self, out: &mut [f32]) -> Result<()> {
        self.fill(usize::MAX, false, out)
    }

    /// Fills one finite noise vector for a predictor/corrector step.
    fn fill(&mut self, step: usize, corrector: bool, out: &mut [f32]) -> Result<()>;
}

/// Source predictor-corrector orchestration over a selected Compute backend.
pub struct SgmseSampler {
    config: SgmseConfig,
    graph_plan: NcsnppV2GraphPlan,
}

impl SgmseSampler {
    /// Creates the fixed 30-step OUVE sampler configuration.
    pub fn new(config: SgmseConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            graph_plan: NcsnppV2GraphPlan::from_config(NcsnppV2Config::source_default())?,
        })
    }

    /// Runs the pinned `get_pc_sampler` schedule with denoising enabled.
    pub fn sample<S: NcsnppScore, R: SgmseNoise>(
        &self,
        compute: &Compute,
        score_model: &mut S,
        noise: &mut R,
        condition: &[f32],
    ) -> Result<Vec<f32>> {
        self.sample_with_options(compute, score_model, noise, condition, true)
    }

    /// Runs the pinned `get_pc_sampler` schedule. `prior_sampling(y)` is
    /// `y + std(1) * noise`, timesteps are inclusive `linspace(1,t_eps,N)`,
    /// and every timestep performs corrector then predictor. The final
    /// predictor mean is returned only when `denoise` is true.
    pub fn sample_with_options<S: NcsnppScore, R: SgmseNoise>(
        &self,
        compute: &Compute,
        score_model: &mut S,
        noise: &mut R,
        condition: &[f32],
        denoise: bool,
    ) -> Result<Vec<f32>> {
        if condition.is_empty() {
            return Err(VokraError::InvalidArgument(
                "sgmse sampler condition must be non-empty".to_owned(),
            ));
        }
        // Keep the source graph validation on the execution boundary. The
        // callback owns actual tensor bindings and must prove them separately;
        // this prevents a sampler from accidentally running a non-source
        // topology while the manifest gate is still closed.
        self.graph_plan.config.validate()?;
        let config = vokra_ops::OuvEConfig::new(1.5, 0.05, 0.5)?;
        let prior_std = config.std(1.0)?;
        let mut prior_noise = vec![0.0; condition.len()];
        noise.fill_prior(&mut prior_noise)?;
        let mut state = condition
            .iter()
            .zip(&prior_noise)
            .map(|(&value, &random)| value + prior_std * random)
            .collect::<Vec<_>>();
        if state.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "sgmse prior sampling produced a non-finite value".to_owned(),
            ));
        }
        let mut score = vec![0.0; state.len()];
        let mut step_noise = vec![0.0; state.len()];
        let mut next = vec![0.0; state.len()];
        let mut next_mean = vec![0.0; state.len()];
        for step_index in 0..self.config.steps {
            let denominator = self.config.steps.saturating_sub(1).max(1) as f32;
            let t = 1.0 - (step_index as f32 / denominator) * (1.0 - self.config.t_eps);
            let next_t = if step_index + 1 < self.config.steps {
                1.0 - ((step_index + 1) as f32 / denominator) * (1.0 - self.config.t_eps)
            } else {
                self.config.t_eps
            };
            let dt = if step_index + 1 < self.config.steps {
                (t - next_t).max(0.0)
            } else {
                // `get_pc_sampler` uses the terminal epsilon interval for
                // the final predictor instead of silently issuing a zero-
                // length update.
                self.config.t_eps
            };
            // Upstream `get_pc_sampler` calls the corrector first, using its
            // score evaluated at the current state/time.
            score_model.score(&state, condition, t, &mut score)?;
            noise.fill(step_index, true, &mut step_noise)?;
            compute.ouve_annealed_langevin_step(
                config,
                &state,
                &score,
                t,
                self.config.snr,
                &step_noise,
                &mut next,
                &mut next_mean,
            )?;
            state.copy_from_slice(&next);

            // Predictor consumes a fresh score/noise pair after the corrector.
            score_model.score(&state, condition, t, &mut score)?;
            noise.fill(step_index, false, &mut step_noise)?;
            compute.ouve_reverse_diffusion_step(
                config,
                &state,
                condition,
                &score,
                t,
                dt,
                &step_noise,
                false,
                &mut next,
                &mut next_mean,
            )?;
            state.copy_from_slice(&next);
            if denoise && step_index + 1 == self.config.steps {
                state.copy_from_slice(&next_mean);
            }
        }
        Ok(state)
    }
}

/// Strict public checkpoint gate. It intentionally cannot produce a runnable
/// model until the VAST safe-load manifest and native tensor binder are closed.
pub struct SgmseModel;

impl SgmseModel {
    /// Rejects arbitrary GGUF files and unreviewed manifests fail-closed.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str())
            .ok_or_else(|| VokraError::ModelLoad("sgmse: missing model arch".to_owned()))?;
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "sgmse: unsupported model arch {arch:?}; expected {ARCH:?}"
            )));
        }
        let status = file
            .get(KEY_MANIFEST_STATUS)
            .and_then(|value| value.as_str());
        if status != Some(AUTHENTICATED_MANIFEST) {
            return Err(VokraError::ModelLoad(
                "sgmse: AUTHENTICATED_MANIFEST_REQUIRED; VAST must bind the complete source tensor manifest"
                    .to_owned(),
            ));
        }
        Err(VokraError::ModelLoad(
            "sgmse: AUTHENTICATED_MANIFEST_REQUIRED until the reviewed tensor digest and semantic bindings are compiled in"
                .to_owned(),
        ))
    }

    /// Management-only helper used by static checks to document the failure
    /// boundary without opening or downloading model weights.
    pub fn require_manifest(path: &Path) -> Result<()> {
        if path.as_os_str().is_empty() {
            return Err(VokraError::InvalidArgument(
                "sgmse: empty checkpoint path".to_owned(),
            ));
        }
        Err(VokraError::ModelLoad(
            "sgmse: AUTHENTICATED_MANIFEST_REQUIRED".to_owned(),
        ))
    }
}

/// Hot ops required by the source graph. This inventory is intentionally not
/// a completion claim: source-exact FIR/resampling paths and full learned
/// graph dispatch remain unavailable even after typed weights are bound.
pub const SGMSE_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Conv2d,
    HotOp::GroupNorm,
    HotOp::Softmax,
    HotOp::Silu,
    HotOp::OuveSde,
];

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroScore;
    impl NcsnppScore for ZeroScore {
        fn score(
            &mut self,
            state: &[f32],
            condition: &[f32],
            _t: f32,
            out: &mut [f32],
        ) -> Result<()> {
            if state.len() != condition.len() || out.len() != state.len() {
                return Err(VokraError::InvalidArgument("test score shape".to_owned()));
            }
            out.fill(0.0);
            Ok(())
        }
    }

    struct ZeroNoise;
    impl SgmseNoise for ZeroNoise {
        fn fill(&mut self, _step: usize, _corrector: bool, out: &mut [f32]) -> Result<()> {
            out.fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn source_configs_are_pinned_and_validated() {
        assert_eq!(SgmseConfig::voicebank().n_fft, 510);
        assert_eq!(SgmseConfig::voicebank().hop_length, 128);
        assert_eq!(
            NcsnppV2Config::source_default().ch_mult,
            vec![1, 1, 2, 2, 2, 2, 2]
        );
        let config = NcsnppV2Config::source_default();
        assert_eq!(config.input_channels, 4);
        assert_eq!(config.output_channels, 2);
        assert_eq!(config.input_resolution, 256);
        assert_eq!(config.group_norm_max_groups, 32);
        assert_eq!(config.group_norm_groups(128).unwrap(), 32);
        assert_eq!(config.group_norm_eps, 1.0e-6);
        assert_eq!(config.fir_kernel, [1, 3, 3, 1]);
        assert!(config.skip_rescale);
        assert!(config.validate().is_ok());
        let plan = NcsnppV2GraphPlan::from_config(config).unwrap();
        assert!(
            plan.stages.iter().any(|stage| {
                stage.kind == NcsnppStageKind::Attention && stage.resolution == 16
            })
        );
        assert_eq!(
            plan.stages
                .iter()
                .filter(|stage| {
                    stage.kind == NcsnppStageKind::Attention && stage.resolution == 16
                })
                .count(),
            3
        );
        assert_eq!(
            plan.stages
                .iter()
                .filter(|stage| stage.kind == NcsnppStageKind::Attention)
                .count(),
            4
        );
        assert!(plan.stages.iter().any(|stage| {
            stage.kind == NcsnppStageKind::Attention && stage.resolution == 4 && stage.block == 0
        }));
        assert_eq!(
            plan.stages
                .iter()
                .filter(|stage| stage.kind == NcsnppStageKind::Downsample)
                .count(),
            6
        );
        assert_eq!(
            plan.stages
                .iter()
                .filter(|stage| stage.kind == NcsnppStageKind::ProgressiveInput)
                .count(),
            6
        );
        assert_eq!(
            plan.stages
                .iter()
                .filter(|stage| stage.kind == NcsnppStageKind::Upsample)
                .count(),
            6
        );
        let first_upsample = plan
            .stages
            .iter()
            .position(|stage| stage.kind == NcsnppStageKind::Upsample)
            .unwrap();
        assert_eq!(
            plan.stages[first_upsample - 1].kind,
            NcsnppStageKind::ProgressiveOutput
        );
        assert!(
            plan.stages[..first_upsample]
                .iter()
                .rev()
                .skip(1)
                .take(3)
                .all(|stage| stage.kind == NcsnppStageKind::Attention
                    || stage.kind == NcsnppStageKind::Residual)
        );
        assert_eq!(
            plan.stages
                .iter()
                .filter(|stage| stage.kind == NcsnppStageKind::Residual)
                .count(),
            7 * 2 + 7 * 3 + 2
        );
        let mut bad = NcsnppV2Config::source_default();
        bad.fourier_embedding = false;
        assert!(bad.validate().is_err());
        let mut zero_resolution = NcsnppV2Config::source_default();
        zero_resolution.input_resolution = 2;
        zero_resolution.ch_mult = vec![1, 1, 1];
        assert!(zero_resolution.validate().is_err());
        let mut too_many_levels = NcsnppV2Config::source_default();
        too_many_levels.ch_mult = vec![1; usize::BITS as usize + 1];
        assert!(too_many_levels.validate().is_err());
        let mut overflowing_up_blocks = NcsnppV2Config::source_default();
        overflowing_up_blocks.num_res_blocks = usize::MAX;
        assert!(overflowing_up_blocks.validate().is_err());
        let mut overflowing_stage_capacity = NcsnppV2Config::source_default();
        overflowing_stage_capacity.num_res_blocks = usize::MAX / 2;
        assert!(overflowing_stage_capacity.validate().is_err());
    }

    #[test]
    fn transform_round_trip_preserves_finite_phase_values() {
        let input = Spectrogram {
            frames: 1,
            bins: 2,
            re: vec![1.0, -0.25],
            im: vec![0.5, 0.75],
        };
        let transformed = transform_forward(input.clone(), 0.15, 0.5).unwrap();
        let restored = transform_inverse(&transformed, 0.15, 0.5).unwrap();
        for (left, right) in input.re.iter().zip(restored.re.iter()) {
            assert!((left - right).abs() < 1e-5);
        }
        for (left, right) in input.im.iter().zip(restored.im.iter()) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn fourier_embedding_uses_supplied_projection_and_rejects_tampering() {
        let embedding = FourierSigmaEmbedding::new(vec![0.25, -0.5]).unwrap();
        let mut output = [0.0; 4];
        embedding.forward(-1.0, &mut output).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(embedding.forward(f32::NAN, &mut output).is_err());
        assert!(embedding.forward(0.0, &mut output[..3]).is_err());
        assert!(FourierSigmaEmbedding::new(vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn sigma_conditioner_routes_projection_and_activation_through_compute() {
        let embedding = FourierSigmaEmbedding::new(vec![0.25]).unwrap();
        let conditioner = SigmaConditioner::new(
            1,
            embedding,
            vec![1.0; 8],
            vec![0.0; 4],
            vec![1.0; 16],
            vec![0.0; 4],
        )
        .unwrap();
        let mut out = [0.0; 4];
        conditioner
            .forward(&Compute::cpu(), -1.0, &mut out)
            .unwrap();
        assert!(out.iter().all(|value| value.is_finite()));
        assert!(
            SigmaConditioner::new(
                1,
                FourierSigmaEmbedding::new(vec![0.25]).unwrap(),
                vec![1.0; 8],
                vec![0.0; 4],
                vec![1.0; 15],
                vec![0.0; 4],
            )
            .is_err()
        );
    }

    #[test]
    fn attention_shape_contract_is_backend_dispatched() {
        let weights = NcsnppAttentionWeights::new(
            2,
            1,
            1.0e-6,
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.1, -0.2],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.3, -0.4],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.5, -0.6],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.7, -0.8],
            true,
        )
        .unwrap();
        // Source layout is channel-major: rows are [1, 2] and [3, 4].
        let input = [1.0, 3.0, 2.0, 4.0];
        let mut output = [0.0; 4];
        weights
            .forward(&Compute::cpu(), &input, &mut output)
            .unwrap();
        let inv_std = 1.25f32.sqrt().recip();
        let normalized = [
            (1.0 - 2.5) * inv_std * 2.0 + 0.3,
            (3.0 - 2.5) * inv_std * 0.5 - 0.7,
            (2.0 - 2.5) * inv_std * 2.0 + 0.3,
            (4.0 - 2.5) * inv_std * 0.5 - 0.7,
        ];
        let q = [normalized[0] + 0.1, normalized[1] - 0.2];
        let q_next = [normalized[2] + 0.1, normalized[3] - 0.2];
        let k = [normalized[0] + 0.3, normalized[1] - 0.4];
        let k_next = [normalized[2] + 0.3, normalized[3] - 0.4];
        let v = [normalized[0] + 0.5, normalized[1] - 0.6];
        let v_next = [normalized[2] + 0.5, normalized[3] - 0.6];
        let scale = 2.0f32.sqrt().recip();
        let score =
            |left: [f32; 2], right: [f32; 2]| (left[0] * right[0] + left[1] * right[1]) * scale;
        let softmax = |a: f32, b: f32| {
            let ea = a.exp();
            let eb = b.exp();
            [ea / (ea + eb), eb / (ea + eb)]
        };
        let p0 = softmax(score(q, k), score(q, k_next));
        let p1 = softmax(score(q_next, k), score(q_next, k_next));
        let context0 = [
            p0[0] * v[0] + p0[1] * v_next[0],
            p0[0] * v[1] + p0[1] * v_next[1],
        ];
        let context1 = [
            p1[0] * v[0] + p1[1] * v_next[0],
            p1[0] * v[1] + p1[1] * v_next[1],
        ];
        let expected_rows = [
            (context0[0] + 0.7 + 1.0) * scale,
            (context0[1] - 0.8 + 2.0) * scale,
            (context1[0] + 0.7 + 3.0) * scale,
            (context1[1] - 0.8 + 4.0) * scale,
        ];
        let expected = [
            expected_rows[0],
            expected_rows[2],
            expected_rows[1],
            expected_rows[3],
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "{actual} != {expected}"
            );
        }
        let no_rescale = NcsnppAttentionWeights::new(
            2,
            1,
            1.0e-6,
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            false,
        )
        .unwrap();
        no_rescale
            .forward(&Compute::cpu(), &input, &mut output)
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(
            weights
                .forward(&Compute::cpu(), &input[..3], &mut output)
                .is_err()
        );
        let unsupported_groups = NcsnppAttentionWeights::new(
            2,
            2,
            1.0e-6,
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            true,
        )
        .unwrap();
        unsupported_groups
            .forward(&Compute::cpu(), &input, &mut output)
            .unwrap();
        assert!(
            NcsnppAttentionWeights::new(
                2,
                1,
                1.0e-6,
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0; 3],
                vec![0.0; 2],
                vec![0.0; 4],
                vec![0.0; 2],
                vec![0.0; 4],
                vec![0.0; 2],
                vec![0.0; 4],
                vec![0.0; 2],
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn strict_binder_stays_closed_without_authenticated_manifest() {
        assert!(SgmseModel::require_manifest(Path::new("checkpoint.gguf")).is_err());
    }

    #[test]
    fn typed_manifest_requires_exact_roles_and_source_identity() {
        let plan = NcsnppV2GraphPlan::from_config(NcsnppV2Config::source_default()).unwrap();
        let role = SgmseTensorRole::NcsnppStage {
            stage_index: 0,
            kind: NcsnppStageKind::Input,
            block: 0,
            slot: SgmseTensorSlot::Weight,
        };
        let mut manifest = SgmseTensorManifest {
            source_revision: SOURCE_REVISION.to_owned(),
            checkpoint_sha256: CHECKPOINT_SHA256.to_owned(),
            graph_config: plan.config.clone(),
            sampler_config: SgmseConfig::voicebank(),
            required_roles: vec![SgmseTensorRole::FourierFrequencies, role.clone()],
            entries: vec![
                SgmseTensorManifestEntry {
                    name: "source.frequency".to_owned(),
                    dtype: GgmlType::F32,
                    dimensions: vec![128],
                    role: SgmseTensorRole::FourierFrequencies,
                },
                SgmseTensorManifestEntry {
                    name: "source.input.weight".to_owned(),
                    dtype: GgmlType::F32,
                    dimensions: vec![4, 4],
                    role: role.clone(),
                },
            ],
        };
        let digest = manifest.canonical_sha256();
        assert_eq!(
            hex_digest(&digest),
            "7c77bb306c39ac21e019c069e329bc47751d5ded02519c3b09755db49658a4a2"
        );
        manifest.validate(&plan, digest).unwrap();

        manifest.required_roles.push(role);
        assert!(manifest.validate(&plan, digest).is_err());
    }

    #[test]
    fn graph_operands_are_revalidated_at_dispatch_boundary() {
        let plan = NcsnppV2GraphPlan::from_config(NcsnppV2Config::source_default()).unwrap();
        let role = SgmseTensorRole::FourierFrequencies;
        let mut weights = SgmseGraphWeights {
            plan,
            tensors: vec![(
                role.clone(),
                "source.frequencies".to_owned(),
                vec![1],
                vec![1.0],
            )],
        };
        weights.validate_before_dispatch().unwrap();
        weights.tensors[0].2 = vec![2];
        assert!(weights.validate_before_dispatch().is_err());
        weights.tensors[0].2 = vec![1];
        weights.tensors[0].3[0] = f32::NAN;
        assert!(weights.validate_before_dispatch().is_err());
    }

    #[test]
    fn sampler_rejects_empty_state_before_backend_dispatch() {
        let sampler = SgmseSampler::new(SgmseConfig::voicebank()).unwrap();
        let mut score = ZeroScore;
        let mut noise = ZeroNoise;
        assert!(
            sampler
                .sample(&Compute::cpu(), &mut score, &mut noise, &[])
                .is_err()
        );
    }

    struct TraceScore {
        times: Vec<f32>,
    }

    impl NcsnppScore for TraceScore {
        fn score(
            &mut self,
            state: &[f32],
            condition: &[f32],
            t: f32,
            out: &mut [f32],
        ) -> Result<()> {
            if state.len() != condition.len() || out.len() != state.len() {
                return Err(VokraError::InvalidArgument("trace score shape".to_owned()));
            }
            self.times.push(t);
            out.fill(0.0);
            Ok(())
        }
    }

    struct TraceNoise {
        events: Vec<(usize, bool)>,
    }

    impl SgmseNoise for TraceNoise {
        fn fill_prior(&mut self, out: &mut [f32]) -> Result<()> {
            self.events.push((usize::MAX, false));
            out.fill(0.0);
            Ok(())
        }

        fn fill(&mut self, step: usize, corrector: bool, out: &mut [f32]) -> Result<()> {
            self.events.push((step, corrector));
            out.fill(if corrector { 0.0 } else { 1.0 });
            Ok(())
        }
    }

    #[test]
    fn sampler_trace_matches_upstream_prior_order_and_inclusive_schedule() {
        let mut config = SgmseConfig::voicebank();
        config.steps = 3;
        config.snr = 0.0;
        let sampler = SgmseSampler::new(config).unwrap();
        let mut score = TraceScore { times: Vec::new() };
        let mut noise = TraceNoise { events: Vec::new() };
        sampler
            .sample_with_options(&Compute::cpu(), &mut score, &mut noise, &[0.0], true)
            .unwrap();
        assert_eq!(
            noise.events,
            vec![
                (usize::MAX, false),
                (0, true),
                (0, false),
                (1, true),
                (1, false),
                (2, true),
                (2, false),
            ]
        );
        assert_eq!(score.times.len(), 6);
        for (actual, expected) in score.times.iter().zip([1.0, 1.0, 0.515, 0.515, 0.03, 0.03]) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        let mut score_no_denoise = TraceScore { times: Vec::new() };
        let mut noise_no_denoise = TraceNoise { events: Vec::new() };
        let stochastic = sampler
            .sample_with_options(
                &Compute::cpu(),
                &mut score_no_denoise,
                &mut noise_no_denoise,
                &[0.0],
                false,
            )
            .unwrap();
        let mut score_denoise = TraceScore { times: Vec::new() };
        let mut noise_denoise = TraceNoise { events: Vec::new() };
        let deterministic = sampler
            .sample_with_options(
                &Compute::cpu(),
                &mut score_denoise,
                &mut noise_denoise,
                &[0.0],
                true,
            )
            .unwrap();
        assert!(
            (stochastic[0] - deterministic[0]).abs() > 1.0e-6,
            "final predictor must use a positive terminal epsilon interval"
        );
    }
}
