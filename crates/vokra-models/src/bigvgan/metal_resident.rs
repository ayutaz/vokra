//! BigVGAN's Metal-resident execution adapter.
//!
//! This adapter is intentionally local to the BigVGAN binder. It owns no
//! model state and only translates the backend-independent resident seam into
//! `MetalContext` device primitives. No activation is downloaded until the
//! terminal waveform.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use vokra_backend_metal::{MetalContext, MetalDeviceTensor};
use vokra_core::{Result, VokraError};
use vokra_ops::bigvgan_generator::BigVganResidentOps;

pub(crate) struct MetalBigVganResidentOps<'ctx> {
    context: &'ctx MetalContext,
}

impl<'ctx> MetalBigVganResidentOps<'ctx> {
    pub(crate) const fn new(context: &'ctx MetalContext) -> Self {
        Self { context }
    }

    fn conv1d_len(in_len: usize, kernel: usize, dilation: usize, padding: usize) -> Result<usize> {
        if in_len == 0 || kernel == 0 || dilation == 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGAN Metal conv1d dimensions must be > 0".to_owned(),
            ));
        }
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal conv1d overflow".to_owned())
            })?;
        let padded = padding
            .checked_mul(2)
            .and_then(|v| v.checked_add(in_len))
            .ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal conv1d padding overflow".to_owned())
            })?;
        if padded < effective {
            return Err(VokraError::InvalidArgument(
                "BigVGAN Metal conv1d padded input is smaller than effective kernel".to_owned(),
            ));
        }
        Ok(padded - effective + 1)
    }

    fn conv_transpose_len(
        in_len: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<usize> {
        if in_len == 0 || kernel == 0 || stride == 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGAN Metal conv_transpose dimensions must be > 0".to_owned(),
            ));
        }
        let base = (in_len - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal conv_transpose overflow".to_owned())
            })?;
        let trim = padding.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument("BigVGAN Metal conv_transpose padding overflow".to_owned())
        })?;
        base.checked_sub(trim).ok_or_else(|| {
            VokraError::InvalidArgument(
                "BigVGAN Metal conv_transpose padding exceeds output".to_owned(),
            )
        })
    }

    fn alias_down_len(time: usize, ratio: usize, taps: usize) -> Result<usize> {
        if time == 0 || ratio == 0 || taps == 0 || taps % 2 != 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGAN Metal alias-free dimensions require non-zero even taps".to_owned(),
            ));
        }
        let padded = time
            .checked_add(taps / 2 - 1)
            .and_then(|v| v.checked_add(taps / 2))
            .ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal downsample overflow".to_owned())
            })?;
        if padded < taps {
            return Err(VokraError::InvalidArgument(
                "BigVGAN Metal downsample filter exceeds padded input".to_owned(),
            ));
        }
        Ok((padded - taps) / ratio + 1)
    }
}

impl<'ctx> BigVganResidentOps for MetalBigVganResidentOps<'ctx> {
    type Tensor = MetalDeviceTensor<'ctx>;

    fn upload(&mut self, data: &[f32]) -> Result<Self::Tensor> {
        self.context.upload(data)
    }

    fn conv1d(
        &mut self,
        input: &Self::Tensor,
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        dilation: usize,
        padding: usize,
    ) -> Result<Self::Tensor> {
        let out_len = Self::conv1d_len(in_len, kernel, dilation, padding)?;
        let weight = self.context.upload(weight)?;
        let bias = bias.map(|b| self.context.upload(b)).transpose()?;
        let mut output = self
            .context
            .alloc_dev(out_ch.checked_mul(out_len).ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal conv1d output overflow".to_owned())
            })?)?;
        self.context.conv1d_dev(
            &mut output,
            input,
            &weight,
            bias.as_ref(),
            in_ch,
            in_len,
            out_ch,
            kernel,
            1,
            dilation,
            padding,
        )?;
        Ok(output)
    }

    fn conv_transpose1d(
        &mut self,
        input: &Self::Tensor,
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
    ) -> Result<Self::Tensor> {
        let out_len = Self::conv_transpose_len(in_len, kernel, stride, padding)?;
        let weight = self.context.upload(weight)?;
        let bias = bias.map(|b| self.context.upload(b)).transpose()?;
        let mut output = self
            .context
            .alloc_dev(out_ch.checked_mul(out_len).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "BigVGAN Metal conv_transpose output overflow".to_owned(),
                )
            })?)?;
        self.context.conv_transpose1d_dev(
            &mut output,
            input,
            &weight,
            bias.as_ref(),
            in_ch,
            in_len,
            out_ch,
            kernel,
            stride,
            padding,
            0,
        )?;
        Ok(output)
    }

    fn snake(
        &mut self,
        input: &Self::Tensor,
        alpha: &[f32],
        channels: usize,
        time: usize,
    ) -> Result<Self::Tensor> {
        let alpha = self.context.upload(alpha)?;
        let mut output = self
            .context
            .alloc_dev(channels.checked_mul(time).ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal Snake output overflow".to_owned())
            })?)?;
        self.context
            .snake_activation_dev(&mut output, input, &alpha, channels, time)?;
        Ok(output)
    }

    fn snake_beta(
        &mut self,
        input: &Self::Tensor,
        alpha: &[f32],
        beta: &[f32],
        channels: usize,
        time: usize,
    ) -> Result<Self::Tensor> {
        let alpha = self.context.upload(alpha)?;
        let beta = self.context.upload(beta)?;
        let mut output = self
            .context
            .alloc_dev(channels.checked_mul(time).ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal SnakeBeta output overflow".to_owned())
            })?)?;
        self.context
            .snake_beta_dev(&mut output, input, &alpha, &beta, channels, time)?;
        Ok(output)
    }

    fn anti_aliased_upsample(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        time: usize,
        ratio: usize,
        filter: &[f32],
    ) -> Result<Self::Tensor> {
        if filter.len() < ratio || filter.len() % 2 != 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGAN Metal upsample filter must be even and cover ratio".to_owned(),
            ));
        }
        let time_out = time.checked_mul(ratio).ok_or_else(|| {
            VokraError::InvalidArgument("BigVGAN Metal upsample overflow".to_owned())
        })?;
        let kernel = self.context.upload(filter)?;
        let mut output = self
            .context
            .alloc_dev(channels.checked_mul(time_out).ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal upsample output overflow".to_owned())
            })?)?;
        self.context.anti_aliased_upsample_dev(
            &mut output,
            input,
            &kernel,
            ratio,
            channels,
            time,
            filter.len(),
        )?;
        Ok(output)
    }

    fn anti_aliased_downsample(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        time: usize,
        ratio: usize,
        filter: &[f32],
    ) -> Result<Self::Tensor> {
        let time_out = Self::alias_down_len(time, ratio, filter.len())?;
        let kernel = self.context.upload(filter)?;
        let mut output = self
            .context
            .alloc_dev(channels.checked_mul(time_out).ok_or_else(|| {
                VokraError::InvalidArgument("BigVGAN Metal downsample output overflow".to_owned())
            })?)?;
        self.context.anti_aliased_downsample_dev(
            &mut output,
            input,
            &kernel,
            ratio,
            channels,
            time,
            filter.len(),
        )?;
        Ok(output)
    }

    fn residual_add(&mut self, dst: &mut Self::Tensor, src: &Self::Tensor) -> Result<()> {
        self.context.residual_add_dev(dst, src)
    }

    fn scale(&mut self, input: &Self::Tensor, scale: f32) -> Result<Self::Tensor> {
        let mut output = self.context.alloc_dev(input.len())?;
        self.context.scale_dev(&mut output, input, scale)?;
        Ok(output)
    }

    fn tanh(&mut self, input: &Self::Tensor) -> Result<Self::Tensor> {
        let mut output = self.context.alloc_dev(input.len())?;
        self.context.tanh_dev(&mut output, input)?;
        Ok(output)
    }

    fn clamp(&mut self, input: &Self::Tensor, lower: f32, upper: f32) -> Result<Self::Tensor> {
        let mut output = self.context.alloc_dev(input.len())?;
        self.context.clamp_dev(&mut output, input, lower, upper)?;
        Ok(output)
    }

    fn download(&mut self, input: &Self::Tensor, output: &mut [f32]) -> Result<()> {
        self.context.download(input, output)
    }
}
