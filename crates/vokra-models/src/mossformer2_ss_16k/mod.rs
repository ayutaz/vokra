//! Native ClearerVoice-Studio MossFormer2 speech separation on CPU and Metal.
//!
//! The binder accepts only the exact 1,076-tensor public
//! `vokra/mossformer2-ss-16k` checkpoint.  Its immutable manifest is pinned
//! before tensor decoding, and every learned dense/convolution/reduction op is
//! dispatched through one selected [`Compute`] backend.  Layout transforms,
//! rotary/sinusoidal coordinates, scalar gates, PReLU and the small affine
//! following InstanceNorm remain deterministic host glue; selecting Metal
//! never invokes a CPU kernel as a fallback.
//!
//! Primary source:
//! `modelscope/ClearerVoice-Studio@6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`,
//! `clearvoice/clearvoice/models/mossformer2_ss/{mossformer2.py,
//! mossformer2_block.py,fsmn.py,conv_module.py}`.

mod nn;
mod weights;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::weights::Mossformer2Weights;

pub const ARCH: &str = "mossformer2_ss_16k";
pub const NAME: &str = "mossformer2_ss_16k";
pub const UPSTREAM_HF: &str = "alibabasglab/MossFormer2_SS_16K";
pub const PUBLIC_HF: &str = "vokra/mossformer2-ss-16k";
pub const UPSTREAM_REVISION: &str = "407cb030cd66340918ebb6c8cc63b18f8592cdbe";
pub const SOURCE_REVISION: &str = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61";
pub const PUBLIC_REVISION: &str = "0e9ba9258cead4252f8e5279598af296ada08bf7";
pub const PUBLIC_MODEL_SHA256: &str =
    "822516b75873dbeb814dac72f7ca0b5fb75254dd051dfdfdda54987347330f0c";
pub const MANIFEST_SHA256: &str =
    "eb4b366872789b95228a172846259f6aa205a75c678f90941d5e8a3e9a47fb8b";

pub const SAMPLE_RATE: u32 = 16_000;
pub const OUTPUT_STREAMS: usize = 2;
pub const ENCODER_CHANNELS: usize = 512;
pub const ENCODER_KERNEL: usize = 16;
pub const ENCODER_STRIDE: usize = 8;
pub const BLOCKS: usize = 24;
pub const GROUP_SIZE: usize = 256;
pub const QUERY_KEY_DIM: usize = 128;
pub const ATTENTION_HIDDEN: usize = 2_048;
pub const FSMN_CHANNELS: usize = 256;

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const STRICT_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "mossformer2-ss-16k",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: Some("mossformer2-ss-16k"),
    tensor_count: 1_076,
    manifest_sha256: [
        0xeb, 0x4b, 0x36, 0x68, 0x72, 0x78, 0x9b, 0x95, 0x22, 0x8a, 0x17, 0x28, 0x46, 0x25, 0x9f,
        0x6a, 0xa2, 0x05, 0xa7, 0x5c, 0x67, 0x8f, 0x90, 0x94, 0x1d, 0x5e, 0x8a, 0x3e, 0x9a, 0x47,
        0xfb, 0x8b,
    ],
};

/// Complete learned-op registry for the released MossFormer2 separator.
pub const MOSSFORMER2_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::LayerNorm,
    HotOp::ScaleNorm,
    HotOp::GroupNorm,
    HotOp::Relu,
    HotOp::Silu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

/// Strict native MossFormer2-SS-16K separator.
#[derive(Debug, Clone)]
pub struct Mossformer2Ss16k {
    weights: Box<Mossformer2Weights>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl Mossformer2Ss16k {
    /// Binds the exact public checkpoint and defaults execution to Mac CPU.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, STRICT_SPEC)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, KEY_CATEGORY, "source-separation")?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "mossformer2-ss-16k: public Apache-2.0 checkpoint resolved to {}",
                checkpoint.weight_license()
            )));
        }
        Ok(Self {
            weights: Box::new(Mossformer2Weights::bind(file)?),
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Preflights the whole learned-op set for one explicit backend.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, MOSSFORMER2_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    #[must_use]
    pub const fn output_streams(&self) -> usize {
        OUTPUT_STREAMS
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the official two-speaker decode policy and returns two PCM lanes.
    pub fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        validate_pcm(pcm)?;
        let compute = Compute::for_backend(self.backend, MOSSFORMER2_HOT_OPS)?;
        nn::separate(&compute, &self.weights, pcm)
    }

    #[cfg(test)]
    pub(crate) fn separate_core_for_test(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        validate_pcm(pcm)?;
        let compute = Compute::for_backend(self.backend, MOSSFORMER2_HOT_OPS)?;
        nn::separate_core(&compute, &self.weights, pcm)
    }
}

impl vokra_core::engines::SeparationEngine for Mossformer2Ss16k {
    fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        Mossformer2Ss16k::separate(self, pcm)
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn output_streams(&self) -> usize {
        OUTPUT_STREAMS
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.len() < ENCODER_KERNEL {
        return Err(VokraError::InvalidArgument(format!(
            "mossformer2-ss-16k: input needs at least {ENCODER_KERNEL} samples, got {}",
            pcm.len()
        )));
    }
    if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "mossformer2-ss-16k: PCM sample {index} is not finite"
        )));
    }
    let energy = pcm.iter().map(|value| value * value).sum::<f32>();
    if !energy.is_finite() || energy <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: input PCM must have finite non-zero energy".to_owned(),
        ));
    }
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("mossformer2-ss-16k: missing/non-string `{key}`"))
        })?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "mossformer2-ss-16k: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn f32_file(path: &Path) -> Vec<f32> {
        let bytes =
            std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "{} f32 alignment", path.display());
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk")))
            .collect()
    }

    fn numeric_error(actual: &[f32], expected: &[f32]) -> (usize, f32, f64, f64) {
        assert_eq!(actual.len(), expected.len());
        assert!(!actual.is_empty());
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(expected.iter().all(|value| value.is_finite()));
        let mut max_index = 0usize;
        let mut max_abs = 0.0f32;
        let mut squared = 0.0f64;
        let mut absolute_l1 = 0.0f64;
        let mut reference_l1 = 0.0f64;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            if delta.total_cmp(&max_abs).is_gt() {
                max_index = index;
                max_abs = delta;
            }
            squared += f64::from(delta) * f64::from(delta);
            absolute_l1 += f64::from(delta);
            reference_l1 += f64::from(expected.abs());
        }
        let rms = (squared / actual.len() as f64).sqrt();
        let relative_l1 = absolute_l1 / reference_l1.max(1.0e-30);
        (max_index, max_abs, rms, relative_l1)
    }

    fn flatten(streams: Vec<Vec<f32>>) -> Vec<f32> {
        assert_eq!(streams.len(), OUTPUT_STREAMS);
        streams.into_iter().flatten().collect()
    }

    #[test]
    fn immutable_public_contract_is_pinned() {
        assert_eq!(STRICT_SPEC.tensor_count, 1_076);
        assert_eq!(MANIFEST_SHA256.len(), 64);
        assert_eq!(PUBLIC_MODEL_SHA256.len(), 64);
        assert_eq!(BLOCKS, 24);
        assert_eq!(OUTPUT_STREAMS, 2);
    }

    #[test]
    fn pcm_validation_fails_loudly() {
        assert!(validate_pcm(&[0.0; ENCODER_KERNEL - 1]).is_err());
        assert!(validate_pcm(&[0.0; ENCODER_KERNEL]).is_err());
        let mut pcm = [0.1; ENCODER_KERNEL];
        pcm[3] = f32::NAN;
        assert!(validate_pcm(&pcm).is_err());
    }

    #[test]
    #[ignore = "requires the authenticated public GGUF and independent official VAST reference"]
    fn measure_real_cpu_and_optional_metal_against_official() {
        let gguf_path = std::path::PathBuf::from(
            std::env::var_os("VOKRA_MOSSFORMER2_GGUF")
                .expect("set VOKRA_MOSSFORMER2_GGUF for ignored real measurement"),
        );
        let reference_dir = std::path::PathBuf::from(
            std::env::var_os("VOKRA_MOSSFORMER2_REFERENCE_DIR")
                .expect("set VOKRA_MOSSFORMER2_REFERENCE_DIR for ignored real measurement"),
        );
        let pcm = f32_file(&reference_dir.join("pcm.f32.bin"));
        let reference = f32_file(&reference_dir.join("separated.f32.bin"));
        assert_eq!(pcm.len(), 4_096, "official PCM extent");
        assert_eq!(reference.len(), OUTPUT_STREAMS * pcm.len());

        let file = GgufFile::open(&gguf_path).expect("open strict MossFormer2 GGUF");
        let cpu = Mossformer2Ss16k::from_gguf(&file).expect("strict MossFormer2 CPU bind");
        let cpu = flatten(
            cpu.separate_core_for_test(&pcm)
                .expect("MossFormer2 CPU core forward"),
        );
        let (index, max_abs, rms, relative_l1) = numeric_error(&cpu, &reference);
        eprintln!(
            "MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs={max_abs:.9e} rms={rms:.9e} relative_l1={relative_l1:.9e} index={index} actual={:.9e} reference={:.9e}",
            cpu[index], reference[index]
        );

        #[cfg(all(feature = "metal", target_os = "macos"))]
        if std::env::var_os("VOKRA_MOSSFORMER2_METAL_MEASUREMENT").is_some() {
            let metal = Mossformer2Ss16k::from_gguf_with_backend(&file, BackendKind::Metal)
                .expect("strict MossFormer2 Metal bind");
            let metal = flatten(
                metal
                    .separate_core_for_test(&pcm)
                    .expect("MossFormer2 Metal core forward"),
            );
            let (index, max_abs, rms, relative_l1) = numeric_error(&metal, &cpu);
            eprintln!(
                "MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs={max_abs:.9e} rms={rms:.9e} relative_l1={relative_l1:.9e} index={index} metal={:.9e} cpu={:.9e}",
                metal[index], cpu[index]
            );
        }
    }
}
