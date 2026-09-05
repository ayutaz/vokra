//! UTMOS22-strong adapter from `vokra-eval`'s topology to the runtime
//! [`vokra_models::Compute`] seam.
//!
//! CPU deliberately uses `Utmos::score`, the independent-upstream-parity
//! oracle. Every non-CPU selection is coverage-gated once at construction and
//! then dispatches all learned primitives through one `Compute`; an unavailable
//! operation is returned to the caller and is never retried on CPU.

use vokra_core::gguf::GgufFile;
use vokra_core::{BackendKind, Result, VokraError};
use vokra_eval::metrics::utmos::{Utmos, UtmosBackendOps};
use vokra_eval::nn::BiLstmBackendOps;
use vokra_models::{Compute, HotOp};

/// Complete learned-op set used by UTMOS22-strong v1.
const UTMOS_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

/// One bound UTMOS scorer and its selected backend.
pub(crate) struct UtmosRuntime {
    model: Utmos,
    device_ops: Option<ComputeUtmosOps>,
}

impl UtmosRuntime {
    /// Binds the real UTMOS topology and coverage-gates `backend`.
    pub(crate) fn from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let model = Utmos::from_gguf(file)?;
        let device_ops = if backend == BackendKind::Cpu {
            None
        } else {
            Some(ComputeUtmosOps {
                compute: Compute::for_backend(backend, UTMOS_HOT_OPS)?,
            })
        };
        Ok(Self { model, device_ops })
    }

    /// Scores one mono clip without any per-op backend fallback.
    pub(crate) fn score(&self, audio: &[f32], sample_rate: u32) -> Result<f64> {
        match &self.device_ops {
            None => self.model.score(audio, sample_rate),
            Some(ops) => self.model.score_with_backend_ops(audio, sample_rate, ops),
        }
    }
}

struct ComputeUtmosOps {
    compute: Compute,
}

impl BiLstmBackendOps for ComputeUtmosOps {
    fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        self.compute.gemm_f32(m, n, k, a, b, bias, out)
    }

    fn gemv_f32(
        &self,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        self.compute.gemv_f32(m, k, a, x, bias, out)
    }
}

impl UtmosBackendOps for ComputeUtmosOps {
    fn conv1d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        self.compute.conv1d_f32(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
        )
    }

    fn grouped_conv1d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        groups: usize,
        out: &mut [f32],
    ) -> Result<()> {
        self.compute.grouped_conv1d_f32(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, groups, out,
        )
    }

    fn group_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        channels: usize,
        len: usize,
        groups: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        if channels == 0 || len == 0 || groups == 0 || channels % groups != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "utmos group_norm: channels/len/groups must be non-zero and channels must be divisible by groups (got {channels}/{len}/{groups})"
            )));
        }
        let expected = channels.checked_mul(len).ok_or_else(|| {
            VokraError::InvalidArgument("utmos group_norm: channels*len overflow".to_owned())
        })?;
        if input.len() != expected
            || out.len() != expected
            || gamma.len() != channels
            || beta.len() != channels
        {
            return Err(VokraError::InvalidArgument(format!(
                "utmos group_norm: expected input/out {expected} and affine {channels}, got input {}, out {}, gamma {}, beta {}",
                input.len(),
                out.len(),
                gamma.len(),
                beta.len()
            )));
        }

        // Compute's GroupNorm primitive is exactly one group. A general
        // GroupNorm is therefore a sequence of independent contiguous group
        // dispatches. UTMOS22 uses groups == channels (512 one-channel
        // groups), while this composition also preserves every valid config.
        let channels_per_group = channels / groups;
        for group in 0..groups {
            let channel_start = group * channels_per_group;
            let channel_end = channel_start + channels_per_group;
            let value_start = channel_start * len;
            let value_end = channel_end * len;
            self.compute.group_norm_f32(
                &input[value_start..value_end],
                &mut out[value_start..value_end],
                channels_per_group,
                len,
                &gamma[channel_start..channel_end],
                &beta[channel_start..channel_end],
                eps,
            )?;
        }
        Ok(())
    }

    fn gelu_f32(&self, input: &[f32], out: &mut [f32]) -> Result<()> {
        self.compute.gelu_f32(input, out)
    }

    fn softmax_f32(&self, input: &[f32], out: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        self.compute.softmax_f32(input, out, rows, cols)
    }

    fn layer_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        self.compute
            .layer_norm_f32(input, out, rows, cols, gamma, beta, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_group_norm_composition_matches_scalar_reference() {
        let channels = 4;
        let len = 5;
        let groups = 2;
        let input = (0..channels * len)
            .map(|index| ((index * 17 + 3) % 29) as f32 / 13.0 - 1.0)
            .collect::<Vec<_>>();
        let gamma = vec![0.5, 1.0, 1.5, 2.0];
        let beta = vec![-0.25, 0.0, 0.25, 0.5];
        let mut expected = vec![0.0; input.len()];
        vokra_eval::nn::group_norm_f32(
            &input,
            &mut expected,
            channels,
            len,
            groups,
            &gamma,
            &beta,
            1e-5,
        )
        .unwrap();

        let ops = ComputeUtmosOps {
            compute: Compute::cpu(),
        };
        let mut actual = vec![0.0; input.len()];
        ops.group_norm_f32(
            &input,
            &mut actual,
            channels,
            len,
            groups,
            &gamma,
            &beta,
            1e-5,
        )
        .unwrap();

        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "max_abs={max_abs:e}");
    }

    #[test]
    fn uncovered_backend_is_rejected_before_utmos_inference() {
        let error = match Compute::for_backend(BackendKind::Vulkan, UTMOS_HOT_OPS) {
            Ok(_) => panic!("Vulkan must not claim the complete UTMOS learned-op set"),
            Err(error) => error,
        };
        match error {
            VokraError::UnsupportedOp(message) | VokraError::BackendUnavailable(message) => {
                assert!(message.to_lowercase().contains("vulkan"), "{message}");
            }
            other => panic!(
                "uncovered Vulkan backend must fail with UnsupportedOp or BackendUnavailable, got {other:?}"
            ),
        }
    }

    // Pre-registered before the first real-device run. The bound is wider
    // than the existing FP32 stage-parity deltas but still strict enough to
    // catch a wrong transpose, missing residual, or LSTM direction swap.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const UTMOS_METAL_SCORE_ATOL: f64 = 1e-3;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const UTMOS_METAL_TAP_ATOL: f32 = 1e-3;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn tiny_v1_cpu_metal_parity_with_preregistered_bound() {
        use vokra_eval::metrics::utmos::{
            ArchVariant, ConvActivation, HeadActivation, HeadPool, TransformerNorm, UtmosConfig,
            V1Spec,
        };

        let config = UtmosConfig {
            variant: ArchVariant::V1,
            v1: Some(V1Spec {
                conv_group_norms: vec![(0, 2)],
                group_norm_eps: 1e-5,
                pos_conv_kernel: 4,
                pos_conv_groups: 1,
                domain_dim: 1,
                domain_id: 0,
                judge_dim: 1,
                judge_id: 0,
                blstm_hidden: 2,
                head_activation: HeadActivation::Relu,
            }),
            sample_rate: 16_000,
            conv_channels: vec![2],
            conv_kernels: vec![5],
            conv_strides: vec![2],
            conv_activation: ConvActivation::Gelu,
            n_layer: 1,
            n_head: 1,
            hidden_dim: 2,
            ffn_dim: 4,
            norm: TransformerNorm::Post,
            ln_eps: 1e-5,
            head_dims: vec![3, 1],
            head_pool: HeadPool::MeanAfter,
            head_scale: 2.0,
            head_offset: 3.0,
        };
        let model = Utmos::synthesized(config, 0x5554_4D4F_535F_4D31).unwrap();
        let audio = (0..256)
            .map(|index| (2.0 * std::f32::consts::PI * 220.0 * index as f32 / 16_000.0).sin())
            .collect::<Vec<_>>();
        let (cpu_score, cpu_taps) = model.score_with_taps(&audio, 16_000).unwrap();
        let ops = ComputeUtmosOps {
            compute: Compute::for_backend(BackendKind::Metal, UTMOS_HOT_OPS).unwrap(),
        };
        let (metal_score, metal_taps) = model
            .score_with_backend_ops_and_taps(&audio, 16_000, &ops)
            .unwrap();
        let score_delta = (metal_score - cpu_score).abs();
        assert!(
            score_delta <= UTMOS_METAL_SCORE_ATOL,
            "score |delta|={score_delta:e} > {UTMOS_METAL_SCORE_ATOL:e}"
        );

        let mut cpu_stages = vec![
            cpu_taps.conv_out,
            cpu_taps.feature_ln,
            cpu_taps.feat_proj,
            cpu_taps.pos_conv,
            cpu_taps.enc_in_ln,
            cpu_taps.blstm_out,
            cpu_taps.head_out,
        ];
        cpu_stages.extend(cpu_taps.enc_blocks);
        let mut metal_stages = vec![
            metal_taps.conv_out,
            metal_taps.feature_ln,
            metal_taps.feat_proj,
            metal_taps.pos_conv,
            metal_taps.enc_in_ln,
            metal_taps.blstm_out,
            metal_taps.head_out,
        ];
        metal_stages.extend(metal_taps.enc_blocks);
        let max_abs = cpu_stages
            .iter()
            .zip(&metal_stages)
            .flat_map(|(cpu, metal)| cpu.iter().zip(metal))
            .map(|(cpu, metal)| (cpu - metal).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= UTMOS_METAL_TAP_ATOL,
            "stage max_abs={max_abs:e} > {UTMOS_METAL_TAP_ATOL:e}"
        );
    }
}
