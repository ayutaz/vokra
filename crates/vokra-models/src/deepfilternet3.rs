//! Native DeepFilterNet3 model wrapper with CPU / Metal dispatch.
//!
//! The faithful topology, GGUF binder and scalar numerical oracle live in
//! [`vokra_ops::denoise`]. This module supplies the model-layer backend
//! contract: CPU preserves the established scalar path exactly; non-CPU
//! backends lower every learned Conv2D, grouped/dense projection and GRU
//! projection through [`Compute::gemm_f32`]. STFT/iSTFT, ERB state,
//! nonlinearities, residual/layout glue and complex deep-filter assembly stay
//! on the host, matching the backend posture of NSNet2 and RNNoise.
//!
//! A backend is validated against the complete hot-op set before inference.
//! Any unavailable backend or failed device operation is returned directly;
//! there is no scalar fallback (FR-EX-08).

use vokra_core::backend::BackendKind;
use vokra_core::engines::{DenoiseEngine, DenoiseStreamHandle};
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{Result, VokraError};
use vokra_ops::denoise::{DeepFilterNetConfig, DenoiseBackendOps, DenoiseModel};

use crate::compute::{Compute, HotOp};

/// Historical converter/runtime architecture tag.
pub const ARCH: &str = "denoise";

/// Every learned DeepFilterNet3 reduction is lowered to GEMM (Conv2D uses
/// host-side im2col/scatter; GRU keeps only gate nonlinearities on the host).
pub const DEEPFILTERNET3_HOT_OPS: &[HotOp] = &[HotOp::Gemm];

/// A strict bound DeepFilterNet3 model plus its selected execution backend.
#[derive(Debug, Clone)]
pub struct DeepFilterNet3 {
    inner: DenoiseModel,
    backend: BackendKind,
}

impl DeepFilterNet3 {
    /// Strictly binds the public `denoise` GGUF and defaults to CPU.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "deepfilternet3: missing string `vokra.model.arch`".to_owned(),
                )
            })?;
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "deepfilternet3: expected `vokra.model.arch = {ARCH}`, got `{arch}`"
            )));
        }
        Ok(Self {
            inner: DenoiseModel::from_gguf(gguf)?,
            backend: BackendKind::Cpu,
        })
    }

    /// Wraps an already validated operator model. Useful to converter and
    /// parity code that constructs the strict tensor map directly.
    pub fn from_model(inner: DenoiseModel) -> Self {
        Self {
            inner,
            backend: BackendKind::Cpu,
        }
    }

    /// Selects the backend used by [`Self::enhance`].
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected backend kind.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Bound architecture/frontend configuration.
    pub fn config(&self) -> &DeepFilterNetConfig {
        self.inner.config()
    }

    /// Enhances a complete mono utterance. CPU uses the original scalar
    /// oracle; every non-CPU learned reduction uses the selected `Compute`.
    pub fn enhance(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        if self.backend == BackendKind::Cpu {
            return self.inner.enhance(pcm);
        }
        let compute = Compute::for_backend(self.backend, DEEPFILTERNET3_HOT_OPS)?;
        self.inner
            .enhance_with_backend_ops(pcm, &ComputeDenoiseOps { compute: &compute })
    }

    #[cfg(test)]
    fn enhance_via_compute(&self, pcm: &[f32], backend: BackendKind) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(backend, DEEPFILTERNET3_HOT_OPS)?;
        self.inner
            .enhance_with_backend_ops(pcm, &ComputeDenoiseOps { compute: &compute })
    }
}

impl DenoiseEngine for DeepFilterNet3 {
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn DenoiseStreamHandle + Send>> {
        if sample_rate != self.config().sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "deepfilternet3: expected {} Hz PCM, got {sample_rate} Hz; resample explicitly",
                self.config().sample_rate
            )));
        }
        Ok(Box::new(DeepFilterNet3Stream {
            model: self.clone(),
            pending: Vec::new(),
            finished: false,
        }))
    }
}

/// DFN3's published path is an offline/lookahead utterance model. The shared
/// stream facade buffers pushes and emits the same one-shot result at
/// `finalize`; it never pretends the model is causal.
struct DeepFilterNet3Stream {
    model: DeepFilterNet3,
    pending: Vec<f32>,
    finished: bool,
}

impl DenoiseStreamHandle for DeepFilterNet3Stream {
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(VokraError::InvalidArgument(
                "deepfilternet3: push after finalize; reset the stream first".to_owned(),
            ));
        }
        if pcm.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "deepfilternet3: input contains a non-finite sample".to_owned(),
            ));
        }
        self.pending.extend_from_slice(pcm);
        Ok(Vec::new())
    }

    fn finalize(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        self.model.enhance(&self.pending)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.finished = false;
    }
}

struct ComputeDenoiseOps<'a> {
    compute: &'a Compute,
}

fn invalid(message: impl Into<String>) -> VokraError {
    VokraError::InvalidArgument(format!("deepfilternet3 backend: {}", message.into()))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "{name} has {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}

fn checked_product(name: &str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| invalid(format!("{name} shape overflows usize: {dims:?}")))
    })
}

fn transpose_out_in(weight: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; weight.len()];
    for out in 0..out_dim {
        for input in 0..in_dim {
            transposed[input * out_dim + out] = weight[out * in_dim + input];
        }
    }
    transposed
}

impl DenoiseBackendOps for ComputeDenoiseOps<'_> {
    fn conv2d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        t_len: usize,
        f_in: usize,
        weight: &[f32],
        out_ch: usize,
        groups: usize,
        kernel_t: usize,
        kernel_f: usize,
        stride_f: usize,
        padding_f: usize,
        output: &mut [f32],
    ) -> Result<()> {
        if groups == 0
            || stride_f == 0
            || kernel_t == 0
            || kernel_f == 0
            || in_ch % groups != 0
            || out_ch % groups != 0
            || f_in + 2 * padding_f < kernel_f
        {
            return Err(invalid("invalid grouped Conv2D geometry"));
        }
        require_len(
            "conv2d input",
            input.len(),
            checked_product("conv2d input", &[in_ch, t_len, f_in])?,
        )?;
        let in_g = in_ch / groups;
        let out_g = out_ch / groups;
        let kernel_size = checked_product("conv2d kernel", &[in_g, kernel_t, kernel_f])?;
        require_len(
            "conv2d weight",
            weight.len(),
            checked_product("conv2d weight", &[out_ch, kernel_size])?,
        )?;
        let f_out = (f_in + 2 * padding_f - kernel_f) / stride_f + 1;
        require_len(
            "conv2d output",
            output.len(),
            checked_product("conv2d output", &[out_ch, t_len, f_out])?,
        )?;
        let positions = checked_product("conv2d positions", &[t_len, f_out])?;
        for group in 0..groups {
            let mut columns = vec![0.0; positions * kernel_size];
            for t in 0..t_len {
                for fo in 0..f_out {
                    let row = t * f_out + fo;
                    for local_in in 0..in_g {
                        let channel = group * in_g + local_in;
                        for dt in 0..kernel_t {
                            let Some(source_t) = (t + dt).checked_sub(kernel_t - 1) else {
                                continue;
                            };
                            for df in 0..kernel_f {
                                let Some(source_f) = (fo * stride_f + df).checked_sub(padding_f)
                                else {
                                    continue;
                                };
                                if source_f >= f_in {
                                    continue;
                                }
                                let column = (local_in * kernel_t + dt) * kernel_f + df;
                                columns[row * kernel_size + column] =
                                    input[(channel * t_len + source_t) * f_in + source_f];
                            }
                        }
                    }
                }
            }
            let mut matrix = vec![0.0; kernel_size * out_g];
            for local_out in 0..out_g {
                let channel = group * out_g + local_out;
                for k in 0..kernel_size {
                    matrix[k * out_g + local_out] = weight[channel * kernel_size + k];
                }
            }
            let mut result = vec![0.0; positions * out_g];
            self.compute.gemm_f32(
                positions,
                out_g,
                kernel_size,
                &columns,
                &matrix,
                None,
                &mut result,
            )?;
            for t in 0..t_len {
                for fo in 0..f_out {
                    let row = t * f_out + fo;
                    for local_out in 0..out_g {
                        let channel = group * out_g + local_out;
                        output[(channel * t_len + t) * f_out + fo] =
                            result[row * out_g + local_out];
                    }
                }
            }
        }
        Ok(())
    }

    fn conv_transpose2d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        t_len: usize,
        f_in: usize,
        weight: &[f32],
        out_ch: usize,
        groups: usize,
        kernel_f: usize,
        stride_f: usize,
        padding_f: usize,
        output_padding_f: usize,
        output: &mut [f32],
    ) -> Result<()> {
        if groups == 0
            || stride_f == 0
            || kernel_f == 0
            || in_ch % groups != 0
            || out_ch % groups != 0
            || f_in == 0
        {
            return Err(invalid("invalid grouped ConvTranspose2D geometry"));
        }
        require_len(
            "conv_transpose2d input",
            input.len(),
            checked_product("conv_transpose2d input", &[in_ch, t_len, f_in])?,
        )?;
        let in_g = in_ch / groups;
        let out_g = out_ch / groups;
        require_len(
            "conv_transpose2d weight",
            weight.len(),
            checked_product("conv_transpose2d weight", &[in_ch, out_g, kernel_f])?,
        )?;
        let base = (f_in - 1)
            .checked_mul(stride_f)
            .and_then(|value| value.checked_add(kernel_f + output_padding_f))
            .ok_or_else(|| invalid("ConvTranspose2D output extent overflows"))?;
        let twice_padding = padding_f
            .checked_mul(2)
            .ok_or_else(|| invalid("ConvTranspose2D padding overflows"))?;
        let f_out = base
            .checked_sub(twice_padding)
            .ok_or_else(|| invalid("ConvTranspose2D padding exceeds output extent"))?;
        require_len(
            "conv_transpose2d output",
            output.len(),
            checked_product("conv_transpose2d output", &[out_ch, t_len, f_out])?,
        )?;
        output.fill(0.0);
        let positions = checked_product("conv_transpose2d positions", &[t_len, f_in])?;
        let expanded_dim = checked_product("conv_transpose2d expanded", &[out_g, kernel_f])?;
        for group in 0..groups {
            let mut rows = vec![0.0; positions * in_g];
            for t in 0..t_len {
                for fi in 0..f_in {
                    let row = t * f_in + fi;
                    for local_in in 0..in_g {
                        let channel = group * in_g + local_in;
                        rows[row * in_g + local_in] = input[(channel * t_len + t) * f_in + fi];
                    }
                }
            }
            let mut matrix = vec![0.0; in_g * expanded_dim];
            for local_in in 0..in_g {
                let channel = group * in_g + local_in;
                for local_out in 0..out_g {
                    for k in 0..kernel_f {
                        matrix[local_in * expanded_dim + local_out * kernel_f + k] =
                            weight[(channel * out_g + local_out) * kernel_f + k];
                    }
                }
            }
            let mut expanded = vec![0.0; positions * expanded_dim];
            self.compute.gemm_f32(
                positions,
                expanded_dim,
                in_g,
                &rows,
                &matrix,
                None,
                &mut expanded,
            )?;
            for t in 0..t_len {
                for fi in 0..f_in {
                    let row = t * f_in + fi;
                    for local_out in 0..out_g {
                        let channel = group * out_g + local_out;
                        for k in 0..kernel_f {
                            let Some(fo) = (fi * stride_f + k).checked_sub(padding_f) else {
                                continue;
                            };
                            if fo < f_out {
                                output[(channel * t_len + t) * f_out + fo] +=
                                    expanded[row * expanded_dim + local_out * kernel_f + k];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn pointwise_conv2d_f32(
        &self,
        input: &[f32],
        channels: usize,
        t_len: usize,
        f_len: usize,
        weight: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let elements = checked_product("pointwise activation", &[channels, t_len, f_len])?;
        require_len("pointwise input", input.len(), elements)?;
        require_len("pointwise weight", weight.len(), channels * channels)?;
        require_len("pointwise output", output.len(), elements)?;
        let positions = checked_product("pointwise positions", &[t_len, f_len])?;
        let mut rows = vec![0.0; positions * channels];
        for t in 0..t_len {
            for f in 0..f_len {
                let row = t * f_len + f;
                for channel in 0..channels {
                    rows[row * channels + channel] = input[(channel * t_len + t) * f_len + f];
                }
            }
        }
        let matrix = transpose_out_in(weight, channels, channels);
        let mut result = vec![0.0; positions * channels];
        self.compute.gemm_f32(
            positions,
            channels,
            channels,
            &rows,
            &matrix,
            None,
            &mut result,
        )?;
        for t in 0..t_len {
            for f in 0..f_len {
                let row = t * f_len + f;
                for channel in 0..channels {
                    output[(channel * t_len + t) * f_len + f] = result[row * channels + channel];
                }
            }
        }
        Ok(())
    }

    fn grouped_linear_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        groups: usize,
        output: &mut [f32],
    ) -> Result<()> {
        if groups == 0 || in_dim % groups != 0 || out_dim % groups != 0 {
            return Err(invalid("invalid grouped-linear geometry"));
        }
        require_len("grouped-linear input", input.len(), rows * in_dim)?;
        require_len(
            "grouped-linear weight",
            weight.len(),
            in_dim * (out_dim / groups),
        )?;
        require_len("grouped-linear output", output.len(), rows * out_dim)?;
        let in_g = in_dim / groups;
        let out_g = out_dim / groups;
        for group in 0..groups {
            let mut group_input = vec![0.0; rows * in_g];
            for row in 0..rows {
                group_input[row * in_g..(row + 1) * in_g].copy_from_slice(
                    &input[row * in_dim + group * in_g..row * in_dim + (group + 1) * in_g],
                );
            }
            let group_weight = &weight[group * in_g * out_g..(group + 1) * in_g * out_g];
            let mut group_output = vec![0.0; rows * out_g];
            self.compute.gemm_f32(
                rows,
                out_g,
                in_g,
                &group_input,
                group_weight,
                None,
                &mut group_output,
            )?;
            for row in 0..rows {
                output[row * out_dim + group * out_g..row * out_dim + (group + 1) * out_g]
                    .copy_from_slice(&group_output[row * out_g..(row + 1) * out_g]);
            }
        }
        Ok(())
    }

    fn linear_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        output: &mut [f32],
    ) -> Result<()> {
        require_len("linear input", input.len(), rows * in_dim)?;
        require_len("linear weight", weight.len(), out_dim * in_dim)?;
        require_len("linear bias", bias.len(), out_dim)?;
        require_len("linear output", output.len(), rows * out_dim)?;
        let matrix = transpose_out_in(weight, out_dim, in_dim);
        self.compute
            .gemm_f32(rows, out_dim, in_dim, input, &matrix, Some(bias), output)
    }

    fn gru_f32(
        &self,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        hidden_dim: usize,
        weight_ih_t: &[f32],
        weight_hh_t: &[f32],
        bias_ih: &[f32],
        bias_hh: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let gates = hidden_dim
            .checked_mul(3)
            .ok_or_else(|| invalid("GRU gate width overflows"))?;
        require_len("gru input", input.len(), rows * input_dim)?;
        require_len("gru weight_ih_t", weight_ih_t.len(), input_dim * gates)?;
        require_len("gru weight_hh_t", weight_hh_t.len(), hidden_dim * gates)?;
        require_len("gru bias_ih", bias_ih.len(), gates)?;
        require_len("gru bias_hh", bias_hh.len(), gates)?;
        require_len("gru output", output.len(), rows * hidden_dim)?;
        let mut input_gates = vec![0.0; rows * gates];
        self.compute.gemm_f32(
            rows,
            gates,
            input_dim,
            input,
            weight_ih_t,
            Some(bias_ih),
            &mut input_gates,
        )?;
        let mut hidden = vec![0.0; hidden_dim];
        let mut recurrent_gates = vec![0.0; gates];
        for row in 0..rows {
            self.compute.gemm_f32(
                1,
                gates,
                hidden_dim,
                &hidden,
                weight_hh_t,
                Some(bias_hh),
                &mut recurrent_gates,
            )?;
            let xg = &input_gates[row * gates..(row + 1) * gates];
            let out = &mut output[row * hidden_dim..(row + 1) * hidden_dim];
            for index in 0..hidden_dim {
                let reset = 1.0 / (1.0 + (-(xg[index] + recurrent_gates[index])).exp());
                let update = 1.0
                    / (1.0
                        + (-(xg[hidden_dim + index] + recurrent_gates[hidden_dim + index])).exp());
                let candidate = (xg[2 * hidden_dim + index]
                    + reset * recurrent_gates[2 * hidden_dim + index])
                    .tanh();
                out[index] = (1.0 - update) * candidate + update * hidden[index];
            }
            hidden.copy_from_slice(out);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use vokra_ops::denoise::denoise_synthesized_tensors;

    fn small_config() -> DeepFilterNetConfig {
        DeepFilterNetConfig {
            n_fft: 64,
            hop: 32,
            sample_rate: 16_000,
            n_erb: 8,
            df_bins: 12,
            df_order: 3,
            min_nb_erb_freqs: 1,
            conv_lookahead: 1,
            df_lookahead: 1,
            conv_ch: 8,
            emb_hidden: 16,
            df_hidden: 16,
            enc_linear_groups: 4,
            linear_groups: 4,
            df_gru_linear_groups: 2,
            emb_num_layers: 3,
            df_num_layers: 2,
            lsnr_min: -15.0,
            lsnr_max: 35.0,
            norm_alpha: 0.99,
        }
    }

    fn small_model(seed: u64) -> DeepFilterNet3 {
        let config = small_config();
        let tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&config, seed)
            .into_iter()
            .map(|(spec, data)| (spec.name, data))
            .collect();
        DeepFilterNet3::from_model(
            DenoiseModel::from_tensors(config, tensors).expect("bind synthetic DFN3"),
        )
    }

    fn pcm() -> Vec<f32> {
        (0..192)
            .map(|index| {
                let phase = index as f32;
                (phase * 0.071).sin() * 0.2 + (phase * 0.193).cos() * 0.03
            })
            .collect()
    }

    fn metrics(expected: &[f32], actual: &[f32]) -> (f32, f32) {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (&want, &got) in expected.iter().zip(actual) {
            let delta = (want - got).abs();
            max_abs = max_abs.max(delta);
            sum_sq += f64::from(delta) * f64::from(delta);
        }
        (max_abs, (sum_sq / expected.len() as f64).sqrt() as f32)
    }

    #[test]
    fn cpu_wrapper_preserves_scalar_bits() {
        let model = small_model(41);
        let input = pcm();
        let direct = model.inner.enhance(&input).unwrap();
        let wrapped = model.enhance(&input).unwrap();
        assert_eq!(direct.len(), wrapped.len());
        for (index, (&want, &got)) in direct.iter().zip(&wrapped).enumerate() {
            assert_eq!(
                want.to_bits(),
                got.to_bits(),
                "CPU wrapper changed scalar output at sample {index}"
            );
        }
    }

    /// Self-consistency lowering check, not an independent upstream parity
    /// oracle. The independent real-checkpoint gate remains
    /// `parity_denoise_dfn3`; this test isolates im2col/layout/GRU dispatch.
    /// Bounds were pre-registered before the first run: max 5e-4, RMSE 1e-4.
    #[test]
    fn compute_cpu_lowering_matches_scalar_oracle() {
        let model = small_model(43);
        let input = pcm();
        let scalar = model.inner.enhance(&input).unwrap();
        let dispatched = model.enhance_via_compute(&input, BackendKind::Cpu).unwrap();
        let (max_abs, rmse) = metrics(&scalar, &dispatched);
        eprintln!(
            "deepfilternet3 lowering parity: samples={} max_abs={max_abs:.9e} rmse={rmse:.9e}",
            scalar.len()
        );
        assert!(max_abs <= 5e-4, "max_abs {max_abs:.9e} exceeds 5e-4");
        assert!(rmse <= 1e-4, "RMSE {rmse:.9e} exceeds 1e-4");
    }

    #[test]
    fn unavailable_backend_fails_without_scalar_fallback() {
        let err = small_model(47)
            .with_backend(BackendKind::Vulkan)
            .enhance(&pcm())
            .unwrap_err();
        let message = err.to_string().to_lowercase();
        assert!(message.contains("vulkan"), "{message}");
        assert!(
            message.contains("not built") || message.contains("no arm"),
            "{message}"
        );
    }

    #[test]
    fn stream_buffers_until_finalize_and_resets() {
        let model = small_model(53);
        let input = pcm();
        let expected = model.enhance(&input).unwrap();
        let mut stream = model.open_stream(16_000).unwrap();
        assert!(stream.push_pcm(&input[..80]).unwrap().is_empty());
        assert!(stream.push_pcm(&input[80..]).unwrap().is_empty());
        assert_eq!(stream.finalize().unwrap(), expected);
        assert!(stream.finalize().unwrap().is_empty());
        stream.reset();
        assert!(stream.push_pcm(&input).unwrap().is_empty());
        assert_eq!(stream.finalize().unwrap(), expected);
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn synthetic_cpu_metal_waveform_parity() {
        let model = small_model(59);
        let input = pcm();
        let cpu = model.enhance(&input).unwrap();
        let metal = match model.with_backend(BackendKind::Metal).enhance(&input) {
            Ok(output) => output,
            Err(VokraError::BackendUnavailable(message)) => {
                eprintln!("skipping DeepFilterNet3 Metal parity: {message}");
                return;
            }
            Err(error) => panic!("DeepFilterNet3 Metal forward failed: {error}"),
        };
        let (max_abs, rmse) = metrics(&cpu, &metal);
        eprintln!(
            "deepfilternet3 CPU/Metal: samples={} max_abs={max_abs:.9e} rmse={rmse:.9e}",
            cpu.len()
        );
        assert!(max_abs <= 5e-4, "max_abs {max_abs:.9e} exceeds 5e-4");
        assert!(rmse <= 1e-4, "RMSE {rmse:.9e} exceeds 1e-4");
    }
}
