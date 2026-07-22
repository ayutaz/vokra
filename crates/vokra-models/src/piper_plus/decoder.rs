//! MB-iSTFT decoder (M0-07-T18/T19): latent → fullband PCM.
//!
//! Follows piper-plus `vits/mb_istft.py::MBiSTFTGenerator`, supporting both
//! conditioning modes (`docs/piper-plus-integration.md` §2.4): the distributed
//! single-speaker voices (`piper_version` 1.11.0) use a single additive
//! `x + cond(g)` after `conv_pre` (`dec.cond` is `[256, 512, 1]`); the zero-shot
//! v7 voice uses multi-stage gated FiLM — `dec.cond` `[2·256, 512, 1]` after
//! `conv_pre`, plus a `dec.cond_layers.{i}` after each upsample+MRF stage.
//! `n_ups` transposed-conv upsample stages (**per-stage shape-driven**: the
//! kernel comes from each `dec.ups.{i}.weight` shape, the stride is uniform
//! `DEC_UP_STRIDE`, the pad is `(kernel − stride)/2`; the ResBlock MRF table and
//! the sub-band post-conv width follow `n_ups` too), an MRF of three ResBlock2
//! branches per stage, a sub-band iSTFT, and PQMF synthesis. The shipping css10
//! / v7 voices are the canonical **two** stride-4 / kernel-16 stages: total
//! upsample = 4·4 (ups) · 4 (iSTFT hop) · 4 (PQMF) = 256 samples/frame.
//!
//! The sub-band iSTFT is the **first real consumer of the M0-04 `istft` op**
//! (`vokra-ops`), which is the point of doing piper-plus natively (ADR-0002
//! reason b). One caveat is recorded at [`Decoder::subband_istft`].

use vokra_core::ir::graph::IstftAttrs;
use vokra_ops::{Spectrogram, istft};

use super::config::{
    DEC_INITIAL, DEC_UP_STRIDE, Dims, GIN, HIDDEN, LRELU_SLOPE, PQMF_TAPS, RESBLOCK_DILATIONS,
    RESBLOCK_KERNELS,
};
use super::nn;
use super::weights::TensorStore;
use crate::compute::Compute;
use vokra_core::Result;

/// A HiFi-GAN ResBlock2: two dilated convs, each `x += conv(leaky_relu(x))`.
struct ResBlock {
    convs: [(Vec<f32>, Vec<f32>); 2],
    kernel: usize,
    dilations: [usize; 2],
    channels: usize,
}

impl ResBlock {
    fn forward(&self, compute: &Compute, mut x: Vec<f32>, t: usize) -> Vec<f32> {
        for (i, (w, b)) in self.convs.iter().enumerate() {
            let mut xt = x.clone();
            nn::leaky_relu(&mut xt, LRELU_SLOPE);
            let d = self.dilations[i];
            let pad = d * (self.kernel - 1) / 2; // same padding
            let (conv, _) = nn::conv1d(
                compute,
                &xt,
                self.channels,
                t,
                w,
                self.channels,
                self.kernel,
                Some(b),
                1,
                pad,
                d,
                1,
            );
            for (xv, cv) in x.iter_mut().zip(&conv) {
                *xv += cv;
            }
        }
        x
    }
}

/// One gated-FiLM conditioning stage (zero-shot v7). `cond(g) =
/// Conv1d(gin → 2·C, 1)` on the global conditioning splits into `gamma | beta`,
/// applied as `y[c,t] = x[c,t]·(sigmoid(gamma[c]) + 0.5) + beta[c]`. The `+0.5`
/// centres the multiplicative gate near 1 (`sigmoid(0)+0.5`), per piper-plus
/// HEAD's multi-scale FiLM. Mul and add stay unfused (separate f32 ops) to
/// match the onnxruntime reference's `Mul`/`Add`.
struct FilmStage {
    /// `{name}.weight` `[2·channels, gin, 1]`.
    w: Vec<f32>,
    /// `{name}.bias` `[2·channels]`.
    b: Vec<f32>,
    /// Conditioned-signal channel count `C` (`gamma` / `beta` each `C` wide).
    channels: usize,
}

impl FilmStage {
    /// Loads a stage from `{name}.weight` `[2·channels, gin, 1]` and
    /// `{name}.bias` `[2·channels]` — the shape assertion is the FiLM
    /// split-size cross-check (a mismatched voice fails loudly at load).
    fn load(store: &TensorStore, name: &str, channels: usize, gin: usize) -> Result<Self> {
        Ok(Self {
            w: store.tensor_shaped(&format!("{name}.weight"), &[2 * channels, gin, 1])?,
            b: store.tensor_shaped(&format!("{name}.bias"), &[2 * channels])?,
            channels,
        })
    }

    /// Applies gated FiLM in place to `sig` `[channels, t]` under `g` `[gin]`.
    fn apply(&self, sig: &mut [f32], t: usize, g: &[f32]) {
        // cond(g) → [2C]: first C = gamma (gate), last C = beta (shift).
        let cond = cond_vector(&self.w, &self.b, g);
        let c = self.channels;
        for ch in 0..c {
            let scale = nn::sigmoid(cond[ch]) + 0.5;
            let shift = cond[c + ch];
            for v in &mut sig[ch * t..ch * t + t] {
                *v = *v * scale + shift;
            }
        }
    }
}

/// Decoder speaker/language conditioning mode.
///
/// The distributed single-speaker voices (`piper_version` 1.11.0) use a single
/// additive `x + cond(g)` after `conv_pre`. The zero-shot v7 voice uses
/// multi-stage gated FiLM: `dec.cond` after `conv_pre` and one
/// `dec.cond_layers.{i}` after each upsample+MRF stage.
enum Cond {
    /// `dec.cond` `[dec_initial, gin, 1]` added after `conv_pre`.
    Additive { w: Vec<f32>, b: Vec<f32> },
    /// Multi-stage gated FiLM: `pre` (`dec.cond`) after `conv_pre`, then
    /// `stages[i]` (`dec.cond_layers.{i}`) after upsample+MRF stage `i`.
    Film {
        pre: FilmStage,
        stages: Vec<FilmStage>,
    },
}

/// One transposed-conv upsample stage: weights + the per-stage geometry the
/// generalized loader derives (kernel from the tensor shape, stride uniform,
/// pad = (kernel − stride)/2). The shipping css10 / v7 voices have two uniform
/// kernel-16 / stride-4 / pad-6 stages; a voice with per-stage-varying kernels
/// (the general MB-iSTFT geometry) loads here without a hard-coded kernel.
struct UpStage {
    /// `dec.ups.{i}.weight` `[in_ch, out_ch, kernel]`.
    w: Vec<f32>,
    /// `dec.ups.{i}.bias` `[out_ch]`.
    b: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
}

/// The MB-iSTFT decoder.
pub(super) struct Decoder {
    conv_pre: (Vec<f32>, Vec<f32>), // [256, 192, 7]
    cond: Cond,                     // additive (1.11.0) or FiLM (v7)
    ups: Vec<UpStage>,
    resblocks: Vec<ResBlock>,
    /// Input-channel count the sub-band post-conv expects = the last upsample
    /// stage's output width (`dec_up_out[n_ups−1]`; 64 for css10 / v7, where it
    /// equals the former `DEC_INITIAL/4` constant).
    subband_in_ch: usize,
    subband_conv_post: (Vec<f32>, Vec<f32>), // [subbands·(n_fft+2), subband_in_ch, 7]
    pqmf_updown: Vec<f32>,                   // [4, 1, 4]
    pqmf_synthesis: Vec<f32>,                // [1, 4, 63]
    n_fft: usize,
    hop: usize,
    subbands: usize,
    /// Periodic-Hann synthesis window (length `n_fft`) and its constant WSS —
    /// used to re-normalise the op output to piper's convention (see
    /// [`Decoder::subband_istft`]).
    window: Vec<f32>,
    const_wss: f32,
}

/// Below this window energy the M0-04 op leaves a sample un-normalised; the
/// renormalisation branches on the same threshold (matches `istft.rs` NOLA_EPS).
const NOLA_EPS: f32 = 1e-8;

impl Decoder {
    pub(super) fn load(
        store: &TensorStore,
        dims: &Dims,
        n_fft: usize,
        hop: usize,
        subbands: usize,
    ) -> Result<Self> {
        // Upsample stages, shape-driven from Dims: `ups[i]` maps
        // `dec_channels[i] → dec_up_out[i]` (`== dec_channels[i+1]`). The kernel
        // is per-stage (`dec_up_kernel[i]`, shape-derived — the former
        // `DEC_UP_KERNEL = 16` hard-assert is gone), the stride is uniform
        // (`DEC_UP_STRIDE`, not shape-derivable), and the pad follows the
        // `(kernel − stride)/2` same-padding convention. For css10 / v7 (kernel
        // 16, stride 4) that is pad 6 — the former `DEC_UP_PAD` constant.
        let mut ups = Vec::with_capacity(dims.n_ups);
        for i in 0..dims.n_ups {
            let in_ch = dims.dec_channels[i];
            let out_ch = dims.dec_up_out[i];
            let kernel = dims.dec_up_kernel[i];
            let stride = DEC_UP_STRIDE;
            let pad = kernel.saturating_sub(stride) / 2;
            ups.push(UpStage {
                w: store.tensor_shaped(&format!("dec.ups.{i}.weight"), &[in_ch, out_ch, kernel])?,
                b: store.tensor_shaped(&format!("dec.ups.{i}.bias"), &[out_ch])?,
                in_ch,
                out_ch,
                kernel,
                stride,
                pad,
            });
        }

        // MRF ResBlock table, one MRF (three branches) per upsample stage — so
        // `n_ups · RESBLOCK_KERNELS.len()` blocks, NOT a fixed 6. Each stage's
        // channel width is that stage's upsample output `dec_up_out[stage]`
        // (128, 64 for the shipping 2-stage voice = the former `DEC_INITIAL/2`,
        // `/4`). This keeps `forward`'s `resblocks[i·num_kernels + branch]`
        // in-range for every stage (a 3-stage voice used to index 6..8 into a
        // 6-element table = OOB panic, hidden behind the old kernel assert).
        let num_kernels = RESBLOCK_KERNELS.len();
        let mut resblocks = Vec::with_capacity(dims.n_ups * num_kernels);
        for stage in 0..dims.n_ups {
            let ch = dims.dec_up_out[stage];
            for (branch, (&k, &dil)) in RESBLOCK_KERNELS
                .iter()
                .zip(RESBLOCK_DILATIONS.iter())
                .enumerate()
            {
                let idx = stage * num_kernels + branch;
                let p = format!("dec.resblocks.{idx}");
                resblocks.push(ResBlock {
                    convs: [
                        (
                            store.tensor_shaped(&format!("{p}.convs.0.weight"), &[ch, ch, k])?,
                            store.tensor_shaped(&format!("{p}.convs.0.bias"), &[ch])?,
                        ),
                        (
                            store.tensor_shaped(&format!("{p}.convs.1.weight"), &[ch, ch, k])?,
                            store.tensor_shaped(&format!("{p}.convs.1.bias"), &[ch])?,
                        ),
                    ],
                    kernel: k,
                    dilations: dil,
                    channels: ch,
                });
            }
        }

        // Multi-stage gated FiLM (v7) vs single additive cond (1.11.0). FiLM
        // applies `dec.cond` after `conv_pre` (C = dec_channels[0]) and
        // `dec.cond_layers.{i}` after each upsample+MRF stage (C =
        // dec_channels[i+1]); `FilmStage::load` shape-checks each `[2·C, gin, 1]`
        // split. The additive path loads the single projection it uses.
        let cond = if dims.film {
            // Stage 0 conditions the conv_pre output (`dec_initial` channels).
            let pre = FilmStage::load(store, "dec.cond", dims.dec_initial, dims.gin)?;
            let mut stages = Vec::with_capacity(dims.n_ups);
            for i in 0..dims.n_ups {
                stages.push(FilmStage::load(
                    store,
                    &format!("dec.cond_layers.{i}"),
                    dims.dec_channels[i + 1],
                    dims.gin,
                )?);
            }
            Cond::Film { pre, stages }
        } else {
            Cond::Additive {
                w: store.tensor_shaped("dec.cond.weight", &[DEC_INITIAL, GIN, 1])?,
                b: store.tensor_shaped("dec.cond.bias", &[DEC_INITIAL])?,
            }
        };

        let sub_out = subbands * (n_fft + 2);
        // The sub-band post-conv consumes the last upsample stage's output
        // (64 for css10 / v7 = the former `DEC_INITIAL/4` constant); shape-driven
        // so a voice whose final stage is not 64 wide still loads.
        let subband_in_ch = *dims.dec_channels.last().expect("n_ups >= 1 (Dims::derive)");
        Ok(Self {
            conv_pre: (
                store.tensor_shaped("dec.conv_pre.weight", &[DEC_INITIAL, HIDDEN, 7])?,
                store.tensor_shaped("dec.conv_pre.bias", &[DEC_INITIAL])?,
            ),
            cond,
            ups,
            resblocks,
            subband_in_ch,
            subband_conv_post: (
                store
                    .tensor_shaped("dec.subband_conv_post.weight", &[sub_out, subband_in_ch, 7])?,
                store.tensor_shaped("dec.subband_conv_post.bias", &[sub_out])?,
            ),
            pqmf_updown: store.tensor_shaped("dec.pqmf.updown_filter", &[subbands, 1, subbands])?,
            pqmf_synthesis: store
                .tensor_shaped("dec.pqmf.synthesis_filter", &[1, subbands, PQMF_TAPS + 1])?,
            n_fft,
            hop,
            subbands,
            window: periodic_hann(n_fft),
            const_wss: {
                let win = periodic_hann(n_fft);
                win.iter().map(|w| w * w).sum::<f32>() * hop as f32 / n_fft as f32
            },
        })
    }

    /// Generates fullband PCM from the decoder-input latent `z` `[HIDDEN, T]`
    /// under global conditioning `g` `[GIN]`.
    ///
    /// # Errors
    ///
    /// Propagates a [`VokraError`](vokra_core::VokraError) from the sub-band
    /// `istft` op (M0-04) rather than panicking, so a malformed spectrogram is a
    /// clean error at the API boundary (M1-01-C).
    pub(super) fn forward(
        &self,
        compute: &Compute,
        z: &[f32],
        t_frames: usize,
        g: &[f32],
    ) -> Result<Vec<f32>> {
        // conv_pre, then the first conditioning stage.
        let (cw, cb) = &self.conv_pre;
        let (mut x, _) = nn::conv1d(
            compute,
            z,
            HIDDEN,
            t_frames,
            cw,
            DEC_INITIAL,
            7,
            Some(cb),
            1,
            3,
            1,
            1,
        );
        match &self.cond {
            Cond::Additive { w, b } => {
                let cg = cond_vector(w, b, g);
                for c in 0..DEC_INITIAL {
                    for t in 0..t_frames {
                        x[c * t_frames + t] += cg[c];
                    }
                }
            }
            // FiLM stage 0 (`dec.cond`) on the conv_pre output.
            Cond::Film { pre, .. } => pre.apply(&mut x, t_frames, g),
        }

        // Upsample stages (`n_ups`, not a fixed 2), each followed by the MRF
        // average. Per-stage kernel / stride / pad (uniform for css10 / v7 =
        // the former constants; shape-driven so a per-stage-varying voice runs).
        let mut t = t_frames;
        let num_kernels = RESBLOCK_KERNELS.len();
        for (i, up_stage) in self.ups.iter().enumerate() {
            nn::leaky_relu(&mut x, LRELU_SLOPE);
            let (up, tout) = nn::conv_transpose1d(
                &x,
                up_stage.in_ch,
                t,
                &up_stage.w,
                up_stage.out_ch,
                up_stage.kernel,
                Some(&up_stage.b),
                up_stage.stride,
                up_stage.pad,
                1,
            );
            t = tout;
            let mut xs = vec![0.0f32; up_stage.out_ch * t];
            for branch in 0..num_kernels {
                let rb = &self.resblocks[i * num_kernels + branch];
                let out = rb.forward(compute, up.clone(), t);
                for (a, b) in xs.iter_mut().zip(&out) {
                    *a += b;
                }
            }
            let inv = 1.0 / num_kernels as f32;
            for v in &mut xs {
                *v *= inv;
            }
            x = xs;
            // FiLM stage i+1 (`dec.cond_layers.{i}`) after the MRF average; the
            // additive single-speaker voice conditions only after conv_pre.
            if let Cond::Film { stages, .. } = &self.cond {
                stages[i].apply(&mut x, t, g);
            }
        }

        // subband_conv_post → [subbands*(n_fft+2), T]. Its input width is the
        // last upsample stage's output (`subband_in_ch`, shape-driven; 64 for
        // css10 / v7 = the former `DEC_INITIAL/4`).
        nn::leaky_relu(&mut x, LRELU_SLOPE);
        let sub_out = self.subbands * (self.n_fft + 2);
        let (sw, sb) = &self.subband_conv_post;
        let (spec_raw, _) = nn::conv1d(
            compute,
            &x,
            self.subband_in_ch,
            t,
            sw,
            sub_out,
            7,
            Some(sb),
            1,
            3,
            1,
            1,
        );

        // Per-subband iSTFT → sub-band waveforms, trimmed to T·hop.
        let sub_len = t * self.hop;
        let mut subbands_sig = vec![0.0f32; self.subbands * sub_len];
        for s in 0..self.subbands {
            let wav = self.subband_istft(&spec_raw, s, t)?;
            subbands_sig[s * sub_len..(s + 1) * sub_len].copy_from_slice(&wav[..sub_len]);
        }

        // PQMF synthesis → fullband [1, T·256].
        Ok(self.pqmf_synthesis(compute, &subbands_sig, sub_len))
    }

    /// iSTFT of sub-band `s` via the M0-04 `istft` op.
    ///
    /// `mag = exp(x[:n_half])`, `phase = sin(x[n_half:])·π`, then `real =
    /// mag·cos(phase)`, `imag = mag·sin(phase)` (piper `stft_onnx.py`). Fed to
    /// the op as a `[frames, bins]` spectrogram with piper's iSTFT settings
    /// (n_fft, hop, periodic Hann, backward norm, `center = false`).
    ///
    /// **T19 finding (`docs/piper-plus-integration.md` §9.2):** piper's
    /// `OnnxISTFT` bakes a *constant* window-sum-of-squares into its inverse
    /// basis, whereas the M0-04 op divides by the *running* per-sample WSS. The
    /// two agree in the steady-state interior (parity ~1e-4) but the op
    /// over-normalises the first/last ~`n_fft` samples of each sub-band, where
    /// fewer frames overlap (up to ~0.2 error). Until the op grows a
    /// `constant_wss` attribute (followup for M0-04), this re-normalises the op
    /// output back to piper's constant-WSS convention exactly:
    /// `numerator[i] = op[i]·running_wss[i]` (or `op[i]` where the op left it
    /// un-normalised), then `/ const_wss`.
    fn subband_istft(&self, spec_raw: &[f32], s: usize, t: usize) -> Result<Vec<f32>> {
        let n_half = self.n_fft / 2 + 1;
        let per_sub = self.n_fft + 2;
        let base = s * per_sub;
        let mut re = vec![0.0f32; t * n_half];
        let mut im = vec![0.0f32; t * n_half];
        for frame in 0..t {
            for fc in 0..n_half {
                let mag = spec_raw[(base + fc) * t + frame].exp();
                let phase =
                    (spec_raw[(base + n_half + fc) * t + frame]).sin() * std::f32::consts::PI;
                re[frame * n_half + fc] = mag * phase.cos();
                im[frame * n_half + fc] = mag * phase.sin();
            }
        }
        let spec = Spectrogram {
            frames: t,
            bins: n_half,
            re,
            im,
        };
        let mut attrs = IstftAttrs::new(self.n_fft, self.hop);
        attrs.center = false;
        attrs.length = Some(t * self.hop);
        let mut wav = istft(&spec, &attrs)?;

        // Re-normalise the op's running-WSS output to piper's constant WSS.
        let total = if t > 0 {
            (t - 1) * self.hop + self.n_fft
        } else {
            0
        };
        let mut running = vec![0.0f32; total];
        for f in 0..t {
            for n in 0..self.n_fft {
                running[f * self.hop + n] += self.window[n] * self.window[n];
            }
        }
        for (i, v) in wav.iter_mut().enumerate() {
            // Recover the OLA numerator (op divided by running only where it
            // exceeded NOLA_EPS), then apply piper's constant WSS.
            let numerator = if running[i] > NOLA_EPS {
                *v * running[i]
            } else {
                *v
            };
            *v = numerator / self.const_wss;
        }
        Ok(wav)
    }

    /// PQMF synthesis: upsample each sub-band (transposed conv, groups =
    /// sub-bands), pad, then the shared synthesis filter → fullband `[T·256]`.
    fn pqmf_synthesis(&self, compute: &Compute, subbands_sig: &[f32], sub_len: usize) -> Vec<f32> {
        let m = self.subbands;
        // Upsample: ConvTranspose1d(updown·M, stride = M, groups = M).
        let (up, up_len) = nn::conv_transpose1d(
            subbands_sig,
            m,
            sub_len,
            &self.pqmf_updown,
            m,
            m,
            None,
            m,
            0,
            m,
        );
        // ConstantPad1d(taps/2) on the time axis of every sub-band channel.
        let pad = PQMF_TAPS / 2;
        let padded_len = up_len + 2 * pad;
        let mut padded = vec![0.0f32; m * padded_len];
        for c in 0..m {
            padded[c * padded_len + pad..c * padded_len + pad + up_len]
                .copy_from_slice(&up[c * up_len..(c + 1) * up_len]);
        }
        // synthesis_filter [1, M, taps+1]: conv1d M→1.
        let (out, _) = nn::conv1d(
            compute,
            &padded,
            m,
            padded_len,
            &self.pqmf_synthesis,
            1,
            PQMF_TAPS + 1,
            None,
            1,
            0,
            1,
            1,
        );
        out
    }
}

/// `cond(g)` = `Conv1d(gin, out, 1)` applied to `g` → `[out]` (additive
/// conditioning: `out = b.len()`, `gin = g.len()`).
#[allow(clippy::needless_range_loop)] // channel-major matrix indexing
fn cond_vector(w: &[f32], b: &[f32], g: &[f32]) -> Vec<f32> {
    let out_ch = b.len();
    let gin = g.len();
    let mut out = b.to_vec();
    for c in 0..out_ch {
        let wrow = c * gin;
        let mut acc = out[c];
        for i in 0..gin {
            acc += w[wrow + i] * g[i];
        }
        out[c] = acc;
    }
    out
}

/// Periodic Hann window of length `n` (`np.hanning(n+1)[:n]`,
/// `torch.hann_window(n, periodic=True)`).
fn periodic_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    //! Per-stage shape-driven geometry tests (M4-RESIDUAL-B (A), T04/T05/T07).
    //!
    //! A synthetic degenerate-dims `TensorStore` is assembled directly through
    //! `GgufBuilder`, then `Decoder::load` + `forward` are exercised for a
    //! **3-stage** voice (the shipping voices are 2-stage). This is a
    //! **self-consistency structural smoke** (no reference oracle): it proves
    //! (a) the ResBlock table is built `n_ups·num_kernels` deep so `forward`'s
    //! `resblocks[i·num_kernels + branch]` never runs off the end (the latent
    //! OOB the old fixed `for stage in 0..2` hid behind the kernel hard-assert),
    //! (b) per-stage kernels differing from 16 load without the former
    //! `DEC_UP_KERNEL` assert, and (c) `forward` completes with the output length
    //! the fixture's known geometry (∏ strides · hop · sub-bands) predicts —
    //! computed independently, NOT from the metadata-only `samples_per_frame`.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

    /// Little-endian f32 bytes.
    fn f32le(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Deterministic small `[-0.1, 0.1)` weights (no external RNG — NFR-DS-02),
    /// so the conv / GEMM / iSTFT / PQMF math runs on non-degenerate data.
    fn pat(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
                bits as f32 / (1u32 << 24) as f32 * 0.2 - 0.1
            })
            .collect()
    }

    fn add(b: &mut GgufBuilder, name: &str, dims: &[usize], seed: u64) {
        let n: usize = dims.iter().product();
        b.add_tensor(
            name,
            GgmlType::F32,
            dims.iter().map(|&d| d as u64).collect(),
            f32le(&pat(n, seed)),
        )
        .expect("add synthetic tensor");
    }

    /// Assembles a `TensorStore` + matching `Dims` for an `n_ups`-stage additive
    /// (single-speaker) MB-iSTFT decoder with the given per-stage upsample
    /// kernels and output widths. Only the tensors `Decoder::load` reads are
    /// written; the ResBlock channel of stage `s` is `dec_up_out[s]` and the
    /// sub-band post-conv input width is the last stage output.
    fn synth_store(
        kernels: &[usize],
        dec_up_out: &[usize],
        n_fft: usize,
        subbands: usize,
    ) -> (TensorStore, Dims) {
        let n_ups = kernels.len();
        assert_eq!(dec_up_out.len(), n_ups);
        let mut dec_channels = vec![DEC_INITIAL];
        dec_channels.extend_from_slice(dec_up_out);
        let subband_in = *dec_channels.last().unwrap();
        let num_kernels = RESBLOCK_KERNELS.len();
        let sub_out = subbands * (n_fft + 2);

        let mut b = GgufBuilder::new();
        // conv_pre + additive cond (fixed medium widths).
        add(&mut b, "dec.conv_pre.weight", &[DEC_INITIAL, HIDDEN, 7], 1);
        add(&mut b, "dec.conv_pre.bias", &[DEC_INITIAL], 2);
        add(&mut b, "dec.cond.weight", &[DEC_INITIAL, GIN, 1], 3);
        add(&mut b, "dec.cond.bias", &[DEC_INITIAL], 4);
        // Upsample stages (per-stage kernel).
        for i in 0..n_ups {
            add(
                &mut b,
                &format!("dec.ups.{i}.weight"),
                &[dec_channels[i], dec_up_out[i], kernels[i]],
                100 + i as u64,
            );
            add(
                &mut b,
                &format!("dec.ups.{i}.bias"),
                &[dec_up_out[i]],
                150 + i as u64,
            );
        }
        // ResBlock MRF table, n_ups · num_kernels deep.
        for (stage, &ch) in dec_up_out.iter().enumerate() {
            for (branch, &k) in RESBLOCK_KERNELS.iter().enumerate() {
                let idx = stage * num_kernels + branch;
                let p = format!("dec.resblocks.{idx}");
                add(
                    &mut b,
                    &format!("{p}.convs.0.weight"),
                    &[ch, ch, k],
                    200 + idx as u64,
                );
                add(
                    &mut b,
                    &format!("{p}.convs.0.bias"),
                    &[ch],
                    250 + idx as u64,
                );
                add(
                    &mut b,
                    &format!("{p}.convs.1.weight"),
                    &[ch, ch, k],
                    300 + idx as u64,
                );
                add(
                    &mut b,
                    &format!("{p}.convs.1.bias"),
                    &[ch],
                    350 + idx as u64,
                );
            }
        }
        add(
            &mut b,
            "dec.subband_conv_post.weight",
            &[sub_out, subband_in, 7],
            400,
        );
        add(&mut b, "dec.subband_conv_post.bias", &[sub_out], 401);
        add(
            &mut b,
            "dec.pqmf.updown_filter",
            &[subbands, 1, subbands],
            402,
        );
        add(
            &mut b,
            "dec.pqmf.synthesis_filter",
            &[1, subbands, PQMF_TAPS + 1],
            403,
        );

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let store = TensorStore::new(file);
        let dims = Dims {
            gin: GIN,
            spk_emb_dim: 192,
            hidden: HIDDEN,
            n_enc_layers: 1,
            ffn: 768,
            dp_filter: 208,
            prosody_in: 3,
            prosody_out: 16,
            dec_initial: DEC_INITIAL,
            dec_up_out: dec_up_out.to_vec(),
            dec_up_kernel: kernels.to_vec(),
            n_ups,
            dec_channels,
            flow_n_flows: 1,
            flow_wn_layers: 1,
            flow_wn_dilation_rate: 1,
            film: false,
        };
        (store, dims)
    }

    /// A 3-stage voice with a **non-uniform** last kernel (16, 16, 8 — the mera
    /// HiFi-GAN geometry's upsample kernels) loads: per-stage kernels honoured,
    /// pad = (kernel − stride)/2, and the ResBlock table is 3·3 = 9 deep. The old
    /// `for stage in 0..2` built only 6, so `forward` would index 6..8 (OOB).
    #[test]
    fn synth_3stage_loads_with_nups_deep_resblock_table() {
        let (store, dims) = synth_store(&[16, 16, 8], &[8, 8, 8], 4, 2);
        let dec = Decoder::load(&store, &dims, 4, 2, 2).expect("load 3-stage decoder");
        assert_eq!(dec.ups.len(), 3, "3 upsample stages");
        assert_eq!(
            dec.resblocks.len(),
            3 * RESBLOCK_KERNELS.len(),
            "ResBlock table must be n_ups·num_kernels deep (not the fixed 6)"
        );
        // Per-stage kernel / stride / pad honoured (pad = (kernel − stride)/2).
        assert_eq!(dec.ups[0].kernel, 16);
        assert_eq!(dec.ups[2].kernel, 8);
        assert_eq!(dec.ups[0].pad, (16 - DEC_UP_STRIDE) / 2);
        assert_eq!(dec.ups[2].pad, (8 - DEC_UP_STRIDE) / 2);
        assert_eq!(dec.subband_in_ch, 8);
    }

    /// `forward` completes for the 3-stage voice (no `resblocks` OOB) and emits
    /// exactly `t_frames · ∏strides · hop · sub-bands` samples — derived from the
    /// fixture geometry, not the metadata `samples_per_frame`. With pad =
    /// (kernel − stride)/2 each transposed conv is an exact `×stride` upsample.
    #[test]
    fn synth_3stage_forward_output_length_matches_geometry() {
        let (n_fft, hop, subbands) = (4, 2, 2);
        let (store, dims) = synth_store(&[16, 16, 8], &[8, 8, 8], n_fft, subbands);
        let dec = Decoder::load(&store, &dims, n_fft, hop, subbands).expect("load");
        let t_frames = 4usize;
        let z = pat(HIDDEN * t_frames, 9);
        let g = pat(GIN, 10);
        let pcm = dec
            .forward(&Compute::cpu(), &z, t_frames, &g)
            .expect("3-stage forward");
        // Each stage upsamples ×DEC_UP_STRIDE (pad = (k−s)/2 ⇒ out = in·stride).
        let strides_product = DEC_UP_STRIDE.pow(dims.n_ups as u32);
        let expected = t_frames * strides_product * hop * subbands;
        assert_eq!(
            pcm.len(),
            expected,
            "output length must follow the geometry"
        );
        assert!(pcm.iter().all(|s| s.is_finite()), "PCM must be finite");
    }

    /// The 2-stage geometry (the shipping css10 / v7 voices) still builds the
    /// canonical 6-deep table and forwards — the generalization reduces cleanly.
    #[test]
    fn synth_2stage_reduces_to_six_resblocks() {
        let (n_fft, hop, subbands) = (4, 4, 4);
        let (store, dims) = synth_store(&[16, 16], &[128, 64], n_fft, subbands);
        let dec = Decoder::load(&store, &dims, n_fft, hop, subbands).expect("load 2-stage");
        assert_eq!(dec.resblocks.len(), 6, "2-stage table is 6 deep");
        assert_eq!(dec.subband_in_ch, 64, "last stage width = DEC_INITIAL/4");
        let t_frames = 3usize;
        let pcm = dec
            .forward(
                &Compute::cpu(),
                &pat(HIDDEN * t_frames, 11),
                t_frames,
                &pat(GIN, 12),
            )
            .expect("2-stage forward");
        assert_eq!(pcm.len(), t_frames * DEC_UP_STRIDE.pow(2) * hop * subbands);
    }
}
