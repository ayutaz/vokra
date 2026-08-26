//! Native bounded-memory NeuTTS Air language-model and NeuCodec route.
//!
//! The public `vokra/neutts-air` GGUF is the exact 291-tensor Qwen2-family
//! causal LM from `neuphonic/neutts-air`.  It emits one 65,536-way NeuCodec
//! token per 20 ms frame.  Waveform synthesis therefore composes two explicit
//! public artifacts: this LM and either official NeuCodec decoder variant.
//! Both stages use the same selected backend; there is no silent CPU fallback.
//!
//! The public GGUF does not contain its Qwen tokenizer or an eSpeak
//! phonemizer.  Callers must supply the exact already-tokenized official
//! prompt, including reference NeuCodec tokens.  Raw text and raw reference
//! audio are deliberately not guessed or substituted.

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::compliance::{CompliancePolicy, check_weight_license};
use vokra_core::decode::SamplerConfig;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::neucodec::{NeuCodec, NeuCodecVariant};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod decoder;
mod weights;

use decoder::NeuTtsAirDecoderRuntime;
use weights::NeuTtsAirMappedDescriptors;

/// Exact architecture tag of the public Vokra artifact.
pub const ARCH: &str = "neutts-air";
/// Exact public Vokra repository revision audited for this binder.
pub const PUBLIC_VOKRA_REVISION: &str = "df2b47ec81862f0e3a19eb2638a6a2bcd2f13b8c";
/// Exact public filename admitted by the external validation contract.
pub const PUBLIC_FILENAME: &str = "neutts-air.gguf";
/// Exact public file length in bytes.
pub const PUBLIC_FILE_BYTES: u64 = 1_495_883_328;
/// Exact public file SHA-256.
pub const PUBLIC_FILE_SHA256: &str =
    "f6caf559e919b16d77ac28177e59ee5427a5de92bdeedd719ecab00b4afbb754";
/// Pinned upstream revision whose dense topology matches the public GGUF.
pub const UPSTREAM_REVISION: &str = "3b58b776406b62fdc137e31ea53d728f5c22a4ed";
/// Released Python implementation used as the independent behavior source.
pub const UPSTREAM_SOURCE_REVISION: &str = "3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e";
/// NeuCodec waveform sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// Maximum Qwen2 context recorded by the official configuration.
pub const MAX_CONTEXT: usize = 32_768;
/// Total-sequence ceiling chosen by the released Python wrapper.
pub const RELEASE_MAX_SEQUENCE: usize = 2_048;

/// Official tokenizer control IDs.  These are independently visible in the
/// ungated `neuphonic/neutts-air-onnx` tokenizer distribution.
pub const TEXT_REPLACE_TOKEN_ID: u32 = 151_665;
pub const TEXT_PROMPT_START_TOKEN_ID: u32 = 151_666;
pub const TEXT_PROMPT_END_TOKEN_ID: u32 = 151_667;
pub const SPEECH_REPLACE_TOKEN_ID: u32 = 151_668;
pub const SPEECH_GENERATION_START_TOKEN_ID: u32 = 151_669;
pub const SPEECH_GENERATION_END_TOKEN_ID: u32 = 151_670;
/// Vocabulary ID representing NeuCodec code zero.
pub const SPEECH_TOKEN_BASE: u32 = 151_671;
/// First non-code IPA token after the contiguous NeuCodec interval.
pub const FIRST_IPA_TOKEN_ID: u32 = 217_207;

const LABEL: &str = "neutts_air";
const CATEGORY: &str = "tts";
const UPSTREAM_HF: &str = "neuphonic/neutts-air";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: "neutts-air",
    model_name_alias: None,
    tensor_count: 291,
    manifest_sha256: [
        0x1d, 0xc9, 0xeb, 0xd7, 0x7a, 0x88, 0x3c, 0x74, 0xbb, 0xb7, 0x2a, 0xd8, 0xa9, 0x7f, 0x08,
        0x8e, 0xf6, 0x66, 0x2e, 0xe8, 0x72, 0xb9, 0x8e, 0x3c, 0x12, 0x36, 0xd7, 0xa5, 0xea, 0xc9,
        0x38, 0x42,
    ],
};

/// Every learned operation in the NeuTTS Air Qwen2 LM.
pub const NEUTTS_AIR_LM_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Exact immutable Qwen2 axes of the public checkpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuTtsAirConfig {
    pub hidden_size: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    pub max_position_embeddings: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: u32,
}

impl NeuTtsAirConfig {
    /// Official release configuration cross-checked against the complete
    /// public tensor manifest and the official ONNX generation descriptor.
    pub const OFFICIAL: Self = Self {
        hidden_size: 896,
        n_layer: 24,
        n_head: 14,
        n_kv_head: 2,
        head_dim: 64,
        ffn_dim: 4_864,
        max_position_embeddings: 32_768,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1.0e-6,
        vocab_size: 217_652,
    };
}

/// Official generation controls over the explicit token-id prompt.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct NeuTtsAirGenerationOptions {
    /// Maximum tokens generated after the prompt, including a terminal token.
    /// The released 2,048-token total-sequence ceiling clamps this cap.
    pub max_new_tokens: usize,
    /// Suppress the speech-end token until this many tokens were generated.
    pub min_new_tokens: usize,
    /// Sampling temperature. Zero selects deterministic greedy generation.
    pub temperature: f32,
    /// Highest-logit candidate count.
    pub top_k: Option<usize>,
    /// Nucleus probability threshold. The released Python wrapper leaves this
    /// disabled; callers may explicitly select the ONNX-oriented policy.
    pub top_p: Option<f32>,
    /// CTRL-style penalty applied to repeated generated tokens. The released
    /// Python wrapper leaves this disabled.
    pub repetition_penalty: Option<f32>,
    /// Deterministic Vokra sampler seed.
    pub seed: u64,
}

impl Default for NeuTtsAirGenerationOptions {
    fn default() -> Self {
        // Mirror `_infer_torch` at source commit
        // 3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e. The separate ONNX GenAI
        // export intentionally carries a different search policy
        // (temperature=.7/top_k=1/top_p=.8/repetition_penalty=1.1); mixing the
        // two would reproduce neither official execution surface.
        Self {
            max_new_tokens: 2_048,
            min_new_tokens: 50,
            temperature: 1.0,
            top_k: Some(50),
            top_p: None,
            repetition_penalty: None,
            seed: 0,
        }
    }
}

impl NeuTtsAirGenerationOptions {
    /// Deterministic route used by numerical and backend parity tests.
    #[must_use]
    pub const fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            min_new_tokens: 0,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 0,
        }
    }

    fn validate(&self, prompt_len: usize) -> Result<()> {
        if self.max_new_tokens == 0 {
            return Err(VokraError::InvalidArgument(
                "neutts_air: max_new_tokens must be positive".to_owned(),
            ));
        }
        if self.min_new_tokens > self.max_new_tokens {
            return Err(VokraError::InvalidArgument(format!(
                "neutts_air: min_new_tokens {} exceeds max_new_tokens {}",
                self.min_new_tokens, self.max_new_tokens
            )));
        }
        if prompt_len >= RELEASE_MAX_SEQUENCE || prompt_len >= MAX_CONTEXT {
            return Err(VokraError::InvalidArgument(format!(
                "neutts_air: prompt length {prompt_len} leaves no generation slot under the released total-sequence ceiling {RELEASE_MAX_SEQUENCE}"
            )));
        }
        let effective = self.effective_max_new_tokens(prompt_len);
        if self.min_new_tokens > effective {
            return Err(VokraError::InvalidArgument(format!(
                "neutts_air: min_new_tokens {} exceeds the effective generated-token cap {effective} after a {prompt_len}-token prompt under the released total-sequence ceiling {RELEASE_MAX_SEQUENCE}",
                self.min_new_tokens
            )));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "neutts_air: temperature must be finite and non-negative, got {}",
                self.temperature
            )));
        }
        if self.top_k.is_some_and(|top_k| {
            top_k == 0 || top_k > NeuTtsAirConfig::OFFICIAL.vocab_size as usize
        }) {
            return Err(VokraError::InvalidArgument(
                "neutts_air: top_k must be in 1..=217652 when present".to_owned(),
            ));
        }
        if self
            .top_p
            .is_some_and(|top_p| !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0))
        {
            return Err(VokraError::InvalidArgument(
                "neutts_air: top_p must be finite and in (0,1] when present".to_owned(),
            ));
        }
        if self
            .repetition_penalty
            .is_some_and(|penalty| !penalty.is_finite() || penalty <= 0.0)
        {
            return Err(VokraError::InvalidArgument(
                "neutts_air: repetition_penalty must be finite and positive when present"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn effective_max_new_tokens(&self, prompt_len: usize) -> usize {
        self.max_new_tokens
            .min(RELEASE_MAX_SEQUENCE.saturating_sub(prompt_len))
            .min(MAX_CONTEXT.saturating_sub(prompt_len))
    }

    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
}

/// Generated vocabulary tokens plus the contiguous NeuCodec code projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuTtsAirGeneration {
    /// Every generated vocabulary token, including speech end when emitted.
    pub token_ids: Vec<u32>,
    /// Valid NeuCodec codes extracted exactly like the official wrapper.
    pub codes: Vec<u32>,
    /// Non-speech, non-terminal tokens ignored by the official decode step.
    pub ignored_token_ids: Vec<u32>,
    /// Whether generation observed the official speech-end token.
    pub ended: bool,
}

/// Complete explicit-companion synthesis result.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuTtsAirSynthesis {
    pub generation: NeuTtsAirGeneration,
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

/// Strict mapped NeuTTS Air language model.
pub struct NeuTtsAir {
    checkpoint: StrictCheckpoint,
    mapped: Arc<NeuTtsAirMappedDescriptors>,
    runtime: NeuTtsAirDecoderRuntime,
    backend: BackendKind,
}

impl std::fmt::Debug for NeuTtsAir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuTtsAir")
            .field("tensor_count", &self.checkpoint.tensor_count())
            .field("weight_license", &self.checkpoint.weight_license())
            .field("backend", &self.backend)
            .finish()
    }
}

impl NeuTtsAir {
    /// Opens the exact public dense GGUF by mmap under the strict policy.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_mapped_with_policy_and_backend(
            path,
            &CompliancePolicy::strict(),
            BackendKind::Cpu,
        )
    }

    /// Opens, license-gates and preflights one backend for the complete LM.
    pub fn open_mapped_with_policy_and_backend(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped_with_policy_and_backend(Arc::new(file), policy, backend)
    }

    /// Strictly binds an already mmap-backed public artifact.
    pub fn from_gguf_mapped_with_policy_and_backend(
        file: Arc<GgufFile>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        check_weight_license(&file, policy)?;
        let checkpoint = StrictCheckpoint::bind(&file, SPEC)?;
        require_string(&file, KEY_CATEGORY, CATEGORY)?;
        require_string(&file, chunks::KEY_PROVENANCE_MODEL_ID, "neutts-air")?;
        require_string(&file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "neutts_air: exact public manifest must carry permissive weights, got {:?}",
                checkpoint.weight_license()
            )));
        }
        Compute::for_backend(backend, NEUTTS_AIR_LM_HOT_OPS)?;
        let mapped = Arc::new(NeuTtsAirMappedDescriptors::bind(
            file,
            NeuTtsAirConfig::OFFICIAL,
        )?);
        Ok(Self {
            checkpoint,
            mapped,
            runtime: NeuTtsAirDecoderRuntime::default(),
            backend,
        })
    }

    #[must_use]
    pub const fn config(&self) -> NeuTtsAirConfig {
        NeuTtsAirConfig::OFFICIAL
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Generates NeuCodec codes from an exact pre-tokenized official prompt.
    pub fn generate_codes(
        &self,
        prompt_token_ids: &[u32],
        options: &NeuTtsAirGenerationOptions,
    ) -> Result<NeuTtsAirGeneration> {
        validate_prompt(prompt_token_ids, options)?;
        decoder::generate(
            &self.mapped,
            self.backend,
            &self.runtime,
            prompt_token_ids,
            options,
        )
    }

    /// Returns the full official-vocabulary logits for the first generated
    /// position. This is the deterministic numerical-parity tap; it executes
    /// the same mapped prefill and selected backend as [`Self::generate_codes`]
    /// without sampling or silently changing the prompt.
    pub fn next_token_logits(&self, prompt_token_ids: &[u32]) -> Result<Vec<f32>> {
        let options = NeuTtsAirGenerationOptions::greedy(1);
        validate_prompt(prompt_token_ids, &options)?;
        decoder::next_token_logits(&self.mapped, self.backend, &self.runtime, prompt_token_ids)
    }

    /// Generates codes and decodes them with an explicit official companion.
    pub fn synthesize_with_companion(
        &self,
        companion: &NeuTtsAirCompanion,
        prompt_token_ids: &[u32],
        options: &NeuTtsAirGenerationOptions,
    ) -> Result<NeuTtsAirSynthesis> {
        if companion.backend() != self.backend {
            return Err(VokraError::InvalidArgument(format!(
                "neutts_air: LM backend {:?} and NeuCodec companion backend {:?} differ; every learned stage must use one backend",
                self.backend,
                companion.backend()
            )));
        }
        let generation = self.generate_codes(prompt_token_ids, options)?;
        if generation.codes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "neutts_air: generation produced no NeuCodec speech tokens".to_owned(),
            ));
        }
        let pcm = companion.codec.decode_codes(&generation.codes)?;
        Ok(NeuTtsAirSynthesis {
            generation,
            pcm,
            sample_rate: SAMPLE_RATE,
        })
    }
}

/// Explicit separately licensed NeuCodec waveform decoder.
#[derive(Debug, Clone)]
pub struct NeuTtsAirCompanion {
    codec: NeuCodec,
}

impl NeuTtsAirCompanion {
    /// Opens either exact official Base or Distill public NeuCodec checkpoint.
    pub fn from_path_with_policy_and_backend(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        let file = GgufFile::open(path)?;
        check_weight_license(&file, policy)?;
        let codec = NeuCodec::from_gguf(&file)?.with_backend(backend);
        if codec.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "neutts_air: official NeuCodec companion must carry permissive weights, got {:?}",
                codec.weight_license()
            )));
        }
        Ok(Self { codec })
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.codec.backend()
    }

    #[must_use]
    pub const fn variant(&self) -> NeuCodecVariant {
        self.codec.variant()
    }
}

/// Maps one official NeuCodec code to the checkpoint vocabulary.
pub fn speech_token_id(code: u32) -> Result<u32> {
    if code >= crate::neucodec::CODEBOOK_SIZE as u32 {
        return Err(VokraError::InvalidArgument(format!(
            "neutts_air: NeuCodec code {code} is outside 0..{}",
            crate::neucodec::CODEBOOK_SIZE
        )));
    }
    Ok(SPEECH_TOKEN_BASE + code)
}

/// Returns the NeuCodec code represented by `token_id`, if any.
#[must_use]
pub fn speech_code(token_id: u32) -> Option<u32> {
    let code = token_id.checked_sub(SPEECH_TOKEN_BASE)?;
    (code < crate::neucodec::CODEBOOK_SIZE as u32).then_some(code)
}

fn validate_prompt(prompt: &[u32], options: &NeuTtsAirGenerationOptions) -> Result<()> {
    if prompt.is_empty() {
        return Err(VokraError::InvalidArgument(
            "neutts_air: explicit prompt_token_ids must not be empty".to_owned(),
        ));
    }
    options.validate(prompt.len())?;
    if let Some((index, token)) = prompt
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= NeuTtsAirConfig::OFFICIAL.vocab_size)
    {
        return Err(VokraError::InvalidArgument(format!(
            "neutts_air: prompt token {token} at index {index} is outside vocabulary 0..{}",
            NeuTtsAirConfig::OFFICIAL.vocab_size
        )));
    }
    let starts: Vec<usize> = prompt
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == SPEECH_GENERATION_START_TOKEN_ID).then_some(index))
        .collect();
    if starts.len() != 1 {
        return Err(VokraError::InvalidArgument(format!(
            "neutts_air: prompt must contain exactly one speech-generation-start token {}, found {}",
            SPEECH_GENERATION_START_TOKEN_ID,
            starts.len()
        )));
    }
    let reference = &prompt[starts[0] + 1..];
    if reference.is_empty() {
        return Err(VokraError::InvalidArgument(
            "neutts_air: prompt must append at least one pre-encoded reference NeuCodec token after speech-generation-start"
                .to_owned(),
        ));
    }
    for (offset, &token) in reference.iter().enumerate() {
        if speech_code(token).is_none() {
            return Err(VokraError::InvalidArgument(format!(
                "neutts_air: prompt token {token} at index {} follows speech-generation-start but is not a NeuCodec speech token",
                starts[0] + 1 + offset
            )));
        }
    }
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key).and_then(GgufMetadataValue::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(VokraError::ModelLoad(format!(
            "neutts_air: `{key}`={actual:?}, expected {expected:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "neutts_air: missing/non-string `{key}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_token_intervals_are_contiguous_and_disjoint() {
        assert_eq!(speech_token_id(0).unwrap(), 151_671);
        assert_eq!(speech_token_id(65_535).unwrap(), 217_206);
        assert_eq!(speech_code(151_671), Some(0));
        assert_eq!(speech_code(217_206), Some(65_535));
        assert_eq!(speech_code(SPEECH_GENERATION_END_TOKEN_ID), None);
        assert_eq!(speech_code(FIRST_IPA_TOKEN_ID), None);
        assert!(speech_token_id(65_536).is_err());
        assert_eq!(FIRST_IPA_TOKEN_ID, SPEECH_TOKEN_BASE + 65_536);
        assert_eq!(NeuTtsAirConfig::OFFICIAL.vocab_size, 217_652);
    }

    #[test]
    fn prompt_contract_requires_exact_reference_code_suffix() {
        let options = NeuTtsAirGenerationOptions::greedy(1);
        let prompt = [
            TEXT_PROMPT_START_TOKEN_ID,
            42,
            TEXT_PROMPT_END_TOKEN_ID,
            SPEECH_GENERATION_START_TOKEN_ID,
            SPEECH_TOKEN_BASE + 7,
        ];
        validate_prompt(&prompt, &options).unwrap();

        let mut invalid = prompt;
        invalid[4] = FIRST_IPA_TOKEN_ID;
        let error = validate_prompt(&invalid, &options).unwrap_err();
        assert!(format!("{error}").contains("not a NeuCodec speech token"));
    }

    #[test]
    fn official_sampling_defaults_match_release_python_wrapper() {
        let options = NeuTtsAirGenerationOptions::default();
        assert_eq!(options.max_new_tokens, 2_048);
        assert_eq!(options.min_new_tokens, 50);
        assert_eq!(options.temperature, 1.0);
        assert_eq!(options.top_k, Some(50));
        assert_eq!(options.top_p, None);
        assert_eq!(options.repetition_penalty, None);
        assert_eq!(RELEASE_MAX_SEQUENCE, 2_048);
        assert_eq!(options.effective_max_new_tokens(48), 2_000);
    }
}
