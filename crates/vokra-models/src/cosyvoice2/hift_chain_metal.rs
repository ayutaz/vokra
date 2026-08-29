//! Metal adapter for the complete CosyVoice2 HiFTNet resident graph.
//!
//! This module is compiled only for the optional Metal feature on Apple
//! targets.  It translates the backend-independent [`HiFTResidentOps`] seam
//! into context-owned `MetalDeviceTensor`s; no intermediate tensor is ever
//! downloaded.  The chain owner performs one explicit final download.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use vokra_backend_metal::{MetalContext, MetalDeviceTensor};
use vokra_core::{Result, VokraError};
use vokra_ops::hiftnet::HiFTResidentOps;
use vokra_ops::nsf::SineGenConfig;

pub(crate) struct MetalHiFTResidentOps<'ctx> {
    context: &'ctx MetalContext,
}

impl<'ctx> MetalHiFTResidentOps<'ctx> {
    pub(crate) const fn new(context: &'ctx MetalContext) -> Self {
        Self { context }
    }

    fn conv1d_time(
        input_time: usize,
        stride: usize,
        dilation: usize,
        kernel: usize,
        padding: usize,
    ) -> Result<usize> {
        if input_time == 0 || stride == 0 || dilation == 0 || kernel == 0 {
            return Err(VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT Conv1d dimensions must be > 0".to_owned(),
            ));
        }
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "CosyVoice2 Metal HiFT Conv1d effective kernel overflow".to_owned(),
                )
            })?;
        let padded = input_time
            .checked_add(padding.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "CosyVoice2 Metal HiFT Conv1d padding overflow".to_owned(),
                )
            })?)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "CosyVoice2 Metal HiFT Conv1d length overflow".to_owned(),
                )
            })?;
        if padded < effective {
            return Err(VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT Conv1d padded input is smaller than effective kernel"
                    .to_owned(),
            ));
        }
        Ok((padded - effective) / stride + 1)
    }

    fn conv_transpose_time(
        input_time: usize,
        stride: usize,
        kernel: usize,
        padding: usize,
    ) -> Result<usize> {
        if input_time == 0 || stride == 0 || kernel == 0 {
            return Err(VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT ConvTranspose dimensions must be > 0".to_owned(),
            ));
        }
        let base = (input_time - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "CosyVoice2 Metal HiFT ConvTranspose length overflow".to_owned(),
                )
            })?;
        base.checked_sub(padding.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT ConvTranspose padding overflow".to_owned(),
            )
        })?)
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT ConvTranspose padding exceeds output".to_owned(),
            )
        })
    }

    fn alloc(&self, channels: usize, time: usize, label: &str) -> Result<MetalDeviceTensor<'ctx>> {
        let len = channels.checked_mul(time).ok_or_else(|| {
            VokraError::InvalidArgument(format!("CosyVoice2 Metal HiFT {label} shape overflow"))
        })?;
        self.context.alloc_dev(len)
    }
}

impl<'ctx> HiFTResidentOps for MetalHiFTResidentOps<'ctx> {
    type Tensor = MetalDeviceTensor<'ctx>;

    fn upload(&mut self, data: &[f32], channels: usize, time: usize) -> Result<Self::Tensor> {
        if data.len()
            != channels.checked_mul(time).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "CosyVoice2 Metal HiFT upload shape overflow".to_owned(),
                )
            })?
        {
            return Err(VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT upload shape mismatch".to_owned(),
            ));
        }
        self.context.upload(data)
    }

    fn download(
        &mut self,
        tensor: &Self::Tensor,
        channels: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        let len = channels.checked_mul(time).ok_or_else(|| {
            VokraError::InvalidArgument("CosyVoice2 Metal HiFT download shape overflow".to_owned())
        })?;
        let mut out = vec![0.0; len];
        self.context.download(tensor, &mut out)?;
        Ok(out)
    }

    fn conv1d(
        &mut self,
        input: &Self::Tensor,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        input_time: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Self::Tensor> {
        let output_time = Self::conv1d_time(input_time, stride, dilation, kernel, padding)?;
        let weight = self.context.upload(weight)?;
        let bias = self.context.upload(bias)?;
        let mut output = self.alloc(out_channels, output_time, "Conv1d")?;
        self.context.conv1d_dev(
            &mut output,
            input,
            &weight,
            Some(&bias),
            in_channels,
            input_time,
            out_channels,
            kernel,
            stride,
            dilation,
            padding,
        )?;
        Ok(output)
    }

    fn conv_transpose1d(
        &mut self,
        input: &Self::Tensor,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        input_time: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Self::Tensor> {
        let output_time = Self::conv_transpose_time(input_time, stride, kernel, padding)?;
        let weight = self.context.upload(weight)?;
        let bias = self.context.upload(bias)?;
        let mut output = self.alloc(out_channels, output_time, "ConvTranspose1d")?;
        self.context.conv_transpose1d_dev(
            &mut output,
            input,
            &weight,
            Some(&bias),
            in_channels,
            input_time,
            out_channels,
            kernel,
            stride,
            padding,
            0,
        )?;
        Ok(output)
    }

    fn reflection_pad_left(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        input_time: usize,
        pad_left: usize,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(
            channels,
            input_time.checked_add(pad_left).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "CosyVoice2 Metal HiFT reflection shape overflow".to_owned(),
                )
            })?,
            "reflection pad",
        )?;
        self.context
            .pad1d_dev(&mut output, input, channels, input_time, pad_left, 0, true)?;
        Ok(output)
    }

    fn elu(&mut self, input: &Self::Tensor, channels: usize, time: usize) -> Result<Self::Tensor> {
        let mut output = self.alloc(channels, time, "ELU")?;
        self.context.elu_dev(&mut output, input)?;
        Ok(output)
    }

    fn leaky_relu(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        time: usize,
        negative_slope: f32,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(channels, time, "LeakyReLU")?;
        self.context
            .leaky_relu_dev(&mut output, input, negative_slope)?;
        Ok(output)
    }

    fn snake(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        time: usize,
        alpha: &[f32],
    ) -> Result<Self::Tensor> {
        let alpha = self.context.upload(alpha)?;
        let mut output = self.alloc(channels, time, "Snake")?;
        self.context
            .snake_activation_dev(&mut output, input, &alpha, channels, time)?;
        Ok(output)
    }

    fn add(
        &mut self,
        lhs: &Self::Tensor,
        rhs: &Self::Tensor,
        channels: usize,
        time: usize,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(channels, time, "add")?;
        self.context.copy_dev(&mut output, lhs)?;
        self.context.residual_add_dev(&mut output, rhs)?;
        Ok(output)
    }

    fn scale(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        time: usize,
        factor: f32,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(channels, time, "scale")?;
        self.context.scale_dev(&mut output, input, factor)?;
        Ok(output)
    }

    fn linear_abs(
        &mut self,
        input: &Self::Tensor,
        in_channels: usize,
        time: usize,
        weight: &[f32],
        bias: f32,
    ) -> Result<Self::Tensor> {
        let weight = self.context.upload(weight)?;
        let bias = self.context.upload(&[bias])?;
        let mut output = self.alloc(1, time, "F0 linear")?;
        self.context
            .linear_abs_dev(&mut output, input, &weight, &bias, in_channels, time)?;
        Ok(output)
    }

    fn linear_tanh(
        &mut self,
        input: &Self::Tensor,
        in_channels: usize,
        time: usize,
        weight: &[f32],
        bias: f32,
    ) -> Result<Self::Tensor> {
        let weight = self.context.upload(weight)?;
        let bias = self.context.upload(&[bias])?;
        let mut output = self.alloc(1, time, "NSF linear")?;
        self.context
            .linear_tanh_dev(&mut output, input, &weight, &bias, in_channels, time)?;
        Ok(output)
    }

    fn nearest_upsample(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        input_time: usize,
        factor: usize,
    ) -> Result<Self::Tensor> {
        let output_time = input_time.checked_mul(factor).ok_or_else(|| {
            VokraError::InvalidArgument("CosyVoice2 Metal HiFT nearest shape overflow".to_owned())
        })?;
        let mut output = self.alloc(channels, output_time, "nearest upsample")?;
        self.context
            .nearest_upsample_dev(&mut output, input, channels, input_time, factor)?;
        Ok(output)
    }

    fn sinegen_deterministic(
        &mut self,
        f0: &Self::Tensor,
        time: usize,
        config: &SineGenConfig,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(config.out_channels(), time, "SineGen")?;
        self.context.sinegen_deterministic_channel_major_dev(
            &mut output,
            f0,
            config.samp_rate,
            config.harmonic_num,
            config.sine_amp,
            config.voiced_threshold,
        )?;
        Ok(output)
    }

    fn stft_concat(
        &mut self,
        input: &Self::Tensor,
        time: usize,
        n_fft: usize,
        hop_len: usize,
    ) -> Result<(Self::Tensor, usize)> {
        let frames = time / hop_len + 1;
        let mut output = self.alloc(n_fft + 2, frames, "STFT")?;
        self.context
            .hift_stft_dev(&mut output, input, n_fft, hop_len)?;
        Ok((output, frames))
    }

    fn complex_from_logits(
        &mut self,
        logits: &Self::Tensor,
        frames: usize,
        n_fft: usize,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(n_fft + 2, frames, "complex postprocess")?;
        self.context
            .complex_from_logits_dev(&mut output, logits, n_fft, frames)?;
        Ok(output)
    }

    fn istft(
        &mut self,
        spectrum: &Self::Tensor,
        frames: usize,
        n_fft: usize,
        hop_len: usize,
    ) -> Result<(Self::Tensor, usize)> {
        let total = (frames - 1)
            .checked_mul(hop_len)
            .and_then(|v| v.checked_add(n_fft))
            .ok_or_else(|| {
                VokraError::InvalidArgument("CosyVoice2 Metal HiFT iSTFT shape overflow".to_owned())
            })?;
        let time = total.checked_sub(n_fft).ok_or_else(|| {
            VokraError::InvalidArgument(
                "CosyVoice2 Metal HiFT iSTFT center trim underflow".to_owned(),
            )
        })?;
        let mut output = self.alloc(1, time, "iSTFT")?;
        self.context
            .istft_dev(&mut output, spectrum, n_fft, hop_len, frames)?;
        Ok((output, time))
    }

    fn clamp(
        &mut self,
        input: &Self::Tensor,
        channels: usize,
        time: usize,
        limit: f32,
    ) -> Result<Self::Tensor> {
        let mut output = self.alloc(channels, time, "clamp")?;
        self.context.clamp_dev(&mut output, input, -limit, limit)?;
        Ok(output)
    }
}
