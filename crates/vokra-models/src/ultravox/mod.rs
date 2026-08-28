//! Native Ultravox v0.5 audio tower and multimodal projector.
//!
//! The public `vokra/ultravox-v0-5-llama-3-2-1b` GGUF is not a standalone
//! audio language model.  It contains the exact 32-layer Whisper
//! large-v3-turbo encoder and the Ultravox frame-stacking projector, but the
//! `meta-llama/Llama-3.2-1B-Instruct` language model is intentionally not
//! bundled.  This module therefore exposes the complete learned audio path and
//! keeps text generation separate from the user-acquired Llama companion.
//! [`UltravoxLlamaCompanion`] strictly binds that exact gated checkpoint and
//! replaces the official consecutive prompt span with projected audio before
//! bounded greedy generation. Tokenization/chat templating remains an explicit
//! caller boundary because neither artifact embeds those sidecars.
//!
//! Weight binding is mmap-backed and layer-at-a-time.  Selecting Metal sends
//! every learned operation through the same preflighted [`crate::compute::Compute`]
//! seam as CPU; no operation silently falls back to the host CPU.

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::compliance::{CompliancePolicy, check_weight_license};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod companion;
mod companion_decoder;
mod companion_weights;
mod encoder;
mod projector;
mod weights;

pub use companion::{
    COMPANION_ARCH, COMPANION_LICENSE, COMPANION_MANIFEST_SHA256, COMPANION_MODEL_NAME,
    COMPANION_SOURCE_REVISION, COMPANION_UPSTREAM_HF, ULTRAVOX_LLAMA_HOT_OPS, UltravoxGeneration,
    UltravoxGenerationOptions, UltravoxLlamaCompanion, UltravoxLlamaConfig,
};
use encoder::UltravoxAudioRuntime;
pub use projector::UltravoxAudioEmbeddings;
use weights::UltravoxMappedDescriptors;

/// Exact architecture tag stamped by the public Vokra artifact.
pub const ARCH: &str = "ultravox";
/// Exact public Vokra repository revision audited by this binder.
pub const PUBLIC_VOKRA_REVISION: &str = "ddbbeec5bfcb09c71a1f88971b794e3e5da811f9";
/// Exact public GGUF filename.
pub const PUBLIC_FILENAME: &str = "ultravox-v0-5-llama-3-2-1b.gguf";
/// Exact public GGUF length in bytes.
pub const PUBLIC_FILE_BYTES: u64 = 1_366_275_264;
/// SHA-256 of the complete public GGUF file.
pub const PUBLIC_FILE_SHA256: &str =
    "376c79a7219bb38fc6a857b0bd9ccf57daff878e7bb4723c4801000c0d7b8c9c";
/// Immutable upstream snapshot whose audio weights define the public GGUF.
pub const UPSTREAM_REVISION: &str = "b95bec8ab291eeb04b5cd600dd473377f6b79026";
/// The separately acquired text companion required by the released model.
pub const TEXT_COMPANION: &str = "meta-llama/Llama-3.2-1B-Instruct";
/// Input waveform sample rate used by the official Whisper feature extractor.
pub const SAMPLE_RATE: u32 = 16_000;

const LABEL: &str = "ultravox";
const MODEL_NAME: &str = "ultravox-v0-5-llama-3-2-1b";
const CATEGORY: &str = "audio-llm";
const UPSTREAM_HF: &str = "fixie-ai/ultravox-v0_5-llama-3_2-1b";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: MODEL_NAME,
    model_name_alias: None,
    tensor_count: 491,
    manifest_sha256: [
        0xac, 0x07, 0x97, 0xd0, 0x35, 0x5a, 0x2e, 0x7d, 0x73, 0x82, 0xcc, 0x31, 0xd0, 0xcb, 0x0f,
        0x80, 0x0a, 0x4c, 0xf2, 0xed, 0xc7, 0x31, 0x3f, 0xbd, 0x5d, 0x5b, 0x57, 0x80, 0xd5, 0x75,
        0x4e, 0xbc,
    ],
};

/// All learned operations in the released audio encoder and projector.
pub const ULTRAVOX_AUDIO_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::RmsNorm,
    HotOp::Gelu,
    HotOp::Silu,
    HotOp::Conv1d,
];

/// Union of every learned operation in the complete audio-to-text route.
///
/// The audio tower and Llama companion each preflight their own subset when
/// they bind.  The composed route also checks this union before doing any
/// work, making the all-or-error CPU/Metal contract explicit at the model
/// boundary rather than relying on the order in which the two artifacts were
/// opened.
pub const ULTRAVOX_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::RmsNorm,
    HotOp::Gelu,
    HotOp::Silu,
    HotOp::Conv1d,
];

/// Immutable dimensions of the public Ultravox v0.5 audio checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UltravoxAudioConfig {
    /// Whisper log-mel channel count.
    pub n_mels: usize,
    /// Maximum log-mel frame count accepted by the encoder.
    pub max_mel_frames: usize,
    /// Whisper encoder width.
    pub hidden_size: usize,
    /// Number of Whisper encoder blocks.
    pub n_layer: usize,
    /// Number of bidirectional attention heads.
    pub n_head: usize,
    /// Whisper MLP width.
    pub ffn_dim: usize,
    /// Number of adjacent encoder frames stacked by the projector.
    pub stack_factor: usize,
    /// Width of a packed stack before projection.
    pub stacked_size: usize,
    /// Packed SwiGLU linear output width.
    pub projector_packed_size: usize,
    /// Width consumed by Llama-3.2-1B token embeddings.
    pub text_hidden_size: usize,
}

impl UltravoxAudioConfig {
    /// Exact axes authenticated from the released config and GGUF manifest.
    pub const OFFICIAL: Self = Self {
        n_mels: 128,
        max_mel_frames: 3_000,
        hidden_size: 1_280,
        n_layer: 32,
        n_head: 20,
        ffn_dim: 5_120,
        stack_factor: 8,
        stacked_size: 10_240,
        projector_packed_size: 4_096,
        text_hidden_size: 2_048,
    };
}

/// Strict mmap-backed Ultravox audio encoder and projector.
pub struct UltravoxAudioTower {
    checkpoint: StrictCheckpoint,
    mapped: Arc<UltravoxMappedDescriptors>,
    runtime: UltravoxAudioRuntime,
    backend: BackendKind,
}

impl std::fmt::Debug for UltravoxAudioTower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UltravoxAudioTower")
            .field("tensor_count", &self.checkpoint.tensor_count())
            .field("weight_license", &self.checkpoint.weight_license())
            .field("backend", &self.backend)
            .finish()
    }
}

impl UltravoxAudioTower {
    /// Opens the exact public dense GGUF under the strict compliance policy on CPU.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_mapped_with_policy_and_backend(
            path,
            &CompliancePolicy::strict(),
            BackendKind::Cpu,
        )
    }

    /// Opens, license-gates and preflights one backend for the full audio path.
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
        require_string(&file, chunks::KEY_PROVENANCE_MODEL_ID, MODEL_NAME)?;
        require_string(&file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "ultravox: exact public audio artifact must carry permissive MIT weights, got {:?}",
                checkpoint.weight_license()
            )));
        }
        Compute::for_backend(backend, ULTRAVOX_AUDIO_HOT_OPS)?;
        let mapped = Arc::new(UltravoxMappedDescriptors::bind(
            file,
            UltravoxAudioConfig::OFFICIAL,
        )?);
        Ok(Self {
            checkpoint,
            mapped,
            runtime: UltravoxAudioRuntime::default(),
            backend,
        })
    }

    /// Exact audio-tower topology.
    #[must_use]
    pub const fn config(&self) -> UltravoxAudioConfig {
        UltravoxAudioConfig::OFFICIAL
    }

    /// Backend selected for every learned audio operation.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// License class authenticated from the GGUF provenance chunk.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Encodes an exact variable-length `[128, n_frames]` Whisper log-mel
    /// tensor and applies the released frame-stacking projector.
    ///
    /// `n_frames` must be in `1..=3000`.  The method never pads an unknown
    /// duration on the caller's behalf: the official feature extractor's
    /// attention-mask length must be preserved by passing only valid frames.
    pub fn encode_log_mel(
        &self,
        log_mel: &[f32],
        n_frames: usize,
    ) -> Result<UltravoxAudioEmbeddings> {
        encoder::encode(self, log_mel, n_frames)
    }

    /// Runs the complete learned audio tower and separately licensed Llama
    /// companion over an exact pre-tokenized prompt.
    ///
    /// Both artifacts must have been opened on the same backend. The method
    /// checks that before any encoder work, so a Metal request can never run
    /// one stage on CPU or leave a partially executed mixed-backend result.
    pub fn generate_from_log_mel_with_companion(
        &self,
        companion: &UltravoxLlamaCompanion,
        log_mel: &[f32],
        n_frames: usize,
        prompt_token_ids: &[u32],
        audio_token_start_idx: usize,
        options: &UltravoxGenerationOptions,
    ) -> Result<UltravoxGeneration> {
        if companion.backend() != self.backend {
            return Err(VokraError::InvalidArgument(format!(
                "ultravox: audio tower backend {:?} and Llama companion backend {:?} differ; every learned stage must use one backend",
                self.backend,
                companion.backend()
            )));
        }
        Compute::for_backend(self.backend, ULTRAVOX_HOT_OPS)?;
        let audio = self.encode_log_mel(log_mel, n_frames)?;
        companion.generate_with_audio_embeddings(
            prompt_token_ids,
            audio_token_start_idx,
            &audio,
            options,
        )
    }

    /// Reports the deliberate standalone-generation boundary of this public
    /// artifact.
    ///
    /// The base Llama weights are governed and distributed separately.  A
    /// caller must not interpret a successful audio encode as a complete
    /// audio-to-text engine.
    pub fn require_text_companion(&self) -> Result<()> {
        Err(VokraError::UnsupportedOp(format!(
            "ultravox: `{PUBLIC_FILENAME}` contains only the MIT Whisper audio tower and projector; text generation requires the separately licensed `{TEXT_COMPANION}` companion plus an exact pre-tokenized prompt passed to `generate_from_log_mel_with_companion`. No Llama weights or tokenizer/chat sidecars are bundled, and Vokra will not substitute another decoder or silently run a partial model."
        )))
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "ultravox: `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strict_checkpoint::sha256_bytes;

    fn hex(bytes: &[u8; 32]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in bytes {
            output.push(char::from(DIGITS[(byte >> 4) as usize]));
            output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
        }
        output
    }

    #[test]
    fn public_tensor_contract_reproduces_authenticated_manifest() {
        let mut contract = weights::tensor_contract(UltravoxAudioConfig::OFFICIAL);
        assert_eq!(contract.len(), SPEC.tensor_count);
        contract.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut canonical = Vec::new();
        for (name, dimensions) in contract {
            canonical.extend_from_slice(name.as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(&(dimensions.len() as u64).to_le_bytes());
            for dimension in dimensions {
                canonical.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        assert_eq!(sha256_bytes(&canonical), SPEC.manifest_sha256);
        assert_eq!(
            hex(&SPEC.manifest_sha256),
            "ac0797d0355a2e7d7382cc31d0cb0f800a4cf2edc7313fbd5d5b5780d5754ebc"
        );
    }

    #[test]
    fn public_file_identity_and_companion_boundary_are_pinned() {
        assert_eq!(PUBLIC_FILE_BYTES, 1_366_275_264);
        assert_eq!(PUBLIC_FILE_SHA256.len(), 64);
        assert_eq!(UltravoxAudioConfig::OFFICIAL.stacked_size, 8 * 1_280);
        assert_eq!(
            UltravoxAudioConfig::OFFICIAL.projector_packed_size,
            2 * UltravoxAudioConfig::OFFICIAL.text_hidden_size
        );
    }

    #[test]
    fn complete_route_preflights_the_union_on_cpu() {
        let compute = Compute::for_backend(BackendKind::Cpu, ULTRAVOX_HOT_OPS)
            .expect("CPU implements the complete Ultravox learned-op union");
        assert_eq!(compute.backend_name(), "cpu");
        assert!(ULTRAVOX_HOT_OPS.contains(&HotOp::Gemv));
        assert!(ULTRAVOX_HOT_OPS.contains(&HotOp::LayerNorm));
        assert!(ULTRAVOX_HOT_OPS.contains(&HotOp::Conv1d));
    }
}
