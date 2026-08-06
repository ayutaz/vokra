//! NSNet2 — Microsoft DNS-Challenge NR baseline runtime binder
//! (Coverage-audit 2026-08-03 Wave B follow-up, sibling of the converter
//! at `crates/vokra-convert/src/models/nsnet2.rs`).
//!
//! # Distinct posture from DeepFilterNet3 (`vokra_ops::denoise`) and RNNoise v0.2
//!
//! Both DFN3 and RNNoise v0.2 are already Vokra runtime members; NSNet2
//! is a third, deliberately-weaker baseline whose value is comparative
//! (industry-baseline quantization-CI reference — CLAUDE.md audio
//! dialect §"Speech Enhancement / AGC / AEC"). Its topology is a plain
//! 2-layer GRU + 3-Linear mask predictor over the 257-bin log-power
//! STFT of 16 kHz PCM — structurally distinct from either sibling so it
//! carries its own `vokra.model.arch = "nsnet2"` tag and its own
//! `vokra.nsnet2.*` hparam chunk group. Silently sharing an arch tag
//! would misroute the runtime dispatch.
//!
//! # Architecture (source of truth: arXiv:2005.07551 §3.2)
//!
//! ```text
//! PCM (16 kHz mono f32)
//!   -> `vokra_ops::stft` (n_fft=512, hop=160, win=320, Hann, center)
//!      -> 257 complex bins per 10 ms frame
//!   -> log(|X|^2 + eps)                                    [t, 257]
//!   -> `fc_in`  Linear(257 -> 400) + ReLU                  [t, 400]
//!   -> `gru_1`  GRU(400 -> 400, stateful)                  [t, 400]
//!   -> `gru_2`  GRU(400 -> 400, stateful)                  [t, 400]
//!   -> `fc_1`   Linear(400 -> 600) + ReLU                  [t, 600]
//!   -> `fc_2`   Linear(600 -> 600) + ReLU                  [t, 600]
//!   -> `mask`   Linear(600 -> 257) + sigmoid               [t, 257]
//!      -> per-bin gain G ∈ [0, 1]
//!   -> Y = G * X (phase preserved verbatim)
//!   -> `vokra_ops::istft_streaming` (tail buffer)
//!   -> denoised PCM
//! ```
//!
//! Per-stream mutable state (FR-LD-06 — hidden inside the stream handle):
//! two `[hidden_dim]` GRU hidden vectors + the streaming iSTFT tail
//! buffer + the not-yet-consumed PCM samples (frames that did not fill
//! the current hop). A fresh `open_stream` reproduces the first run
//! bit-for-bit.
//!
//! # GRU gate ordering (upstream convention)
//!
//! Upstream NSNet2 is trained in PyTorch and exported through
//! `torch.onnx.export`; both PyTorch's `nn.GRU` and its ONNX export use
//! the **`[Z; R; H]`** row block layout (update, reset, hidden — ONNX
//! GRU spec `Wz/Wr/Wh`), and PyTorch sets `linear_before_reset=1` on
//! export (the recurrent `W_hn · h` projection is computed first, then
//! gated by `r`, matching `rnnoise::gru_forward`).
//!
//! To reuse the tested [`vokra_ops::rnnoise_gru_forward`] primitive
//! (`[R; Z; H]` layout, `linear_before_reset=1`), the loader permutes
//! the row blocks once at bind time (`[Z; R; H]` → `[R; Z; H]`). The
//! forward path never touches ONNX-native layout after that — every
//! runtime call uses the primitive verbatim.
//!
//! # Real-weight parity posture
//!
//! Real-weight parity against the upstream ONNX Runtime pipeline is
//! deferred to the owner (env-gated harness
//! `crates/vokra-models/tests/parity_nsnet2.rs`, `VOKRA_NSNET2_REAL_GGUF`
//! + `VOKRA_NSNET2_REAL_WAV`). This module ships:
//!
//! - the exact tensor / hparam contract [`Nsnet2V1::from_gguf`] binds
//!   against;
//! - synthetic-weight structural tests pinning FR-EX-08 (loud errors on
//!   every shape / rate / tensor-name mismatch);
//! - identity-gain sanity: a synthetic mask that forces the sigmoid
//!   pre-activation to +∞ must reproduce the input STFT bit-for-bit up
//!   to the streaming iSTFT accumulator's steady-state region.

use std::sync::Arc;

use vokra_core::engines::{DenoiseEngine, DenoiseStreamHandle};
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::ir::graph::{IstftAttrs, IstftStreamingAttrs, StftAttrs};
use vokra_core::{Result, VokraError};
use vokra_ops::{IstftStreamingState, Spectrogram, rnnoise_gru_forward, stft};

#[cfg(test)]
mod tests;

// ---- arch / provenance constants ----------------------------------------
//
// Mirror of `vokra-convert::models::nsnet2::{ARCH, NAME, CATEGORY,
// UPSTREAM_URL}` — kept as duplicated `pub const` so the runtime binder
// does not add a cross-crate dependency edge onto the converter (the
// sibling `fsmn_vad` / `openwakeword` / `silero_vad` convention).

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model nsnet2`.
pub const ARCH: &str = "nsnet2";

/// Default `vokra.model.name` value written by the converter.
pub const DEFAULT_NAME: &str = "nsnet2-20ms-baseline";

/// `vokra.model.category` — enhancement (speech-enhancement /
/// noise-suppression family; shared with DFN3 / RNNoise v0.2).
pub const CATEGORY: &str = "enhancement";

/// PCM sample rate the upstream 20 ms baseline was trained at (Hz).
/// Real-weight parity harnesses assert against this so a fixture at a
/// different rate is refused loudly (FR-EX-08).
pub const SAMPLE_RATE_DEFAULT: u32 = 16_000;

// ---- vokra.nsnet2.* metadata keys ---------------------------------------
//
// Mirror of the converter-side constants (kept duplicated per the
// runtime-vs-converter no-cross-dep convention). Any change here must be
// mirrored in `vokra-convert::models::nsnet2::KEY_*`.

/// GGUF metadata key: STFT bin count (u32; upstream = 257).
pub const KEY_N_BINS: &str = "vokra.nsnet2.n_bins";
/// GGUF metadata key: GRU / fc_in hidden width (u32; upstream = 400).
pub const KEY_HIDDEN_DIM: &str = "vokra.nsnet2.hidden_dim";
/// GGUF metadata key: `fc_1` output width (u32; upstream = 600).
pub const KEY_FC1_DIM: &str = "vokra.nsnet2.fc1_dim";
/// GGUF metadata key: `fc_2` output width (u32; upstream = 600).
pub const KEY_FC2_DIM: &str = "vokra.nsnet2.fc2_dim";
/// GGUF metadata key: STFT FFT length (u32; upstream = 512).
pub const KEY_N_FFT: &str = "vokra.nsnet2.n_fft";
/// GGUF metadata key: STFT hop (u32 samples; upstream = 160).
pub const KEY_HOP: &str = "vokra.nsnet2.hop";
/// GGUF metadata key: STFT window length (u32 samples; upstream = 320).
pub const KEY_WIN_LENGTH: &str = "vokra.nsnet2.win_length";
/// GGUF metadata key: PCM sample rate (u32 Hz; upstream = 16 000).
pub const KEY_SAMPLE_RATE: &str = "vokra.nsnet2.sample_rate";

// ---- tensor-name convention ---------------------------------------------
//
// The prep sidecar (`tools/parity/nsnet2_prepare_checkpoint.py`) emits
// upstream ONNX initializer names verbatim (mirror of the CSM / Kokoro /
// emotion2vec contract). NSNet2's initializer names are unqualified —
// `fc_in.weight` / `gru_1.W` / `mask.bias` — so no module-prefix walk is
// needed. These `TENSOR_*` constants are the single-source-of-truth
// spelling; the parity sidecar's audit step pins these against the real
// ONNX before the real-weight wave lands.

/// Input Linear weight `[hidden_dim, n_bins]` (`fc_in.weight`).
pub const TENSOR_FC_IN_WEIGHT: &str = "fc_in.weight";
/// Input Linear bias `[hidden_dim]` (`fc_in.bias`).
pub const TENSOR_FC_IN_BIAS: &str = "fc_in.bias";
/// GRU-1 input-hidden weights `[3*hidden_dim, hidden_dim]` (ONNX `[Z;R;H]`).
pub const TENSOR_GRU_1_W: &str = "gru_1.W";
/// GRU-1 recurrent weights `[3*hidden_dim, hidden_dim]` (ONNX `[Z;R;H]`).
pub const TENSOR_GRU_1_R: &str = "gru_1.R";
/// GRU-1 bias `[6*hidden_dim]` (ONNX packs input + recurrent biases
/// contiguously: `[Wb_z; Wb_r; Wb_h; Rb_z; Rb_r; Rb_h]`).
pub const TENSOR_GRU_1_B: &str = "gru_1.B";
/// GRU-2 input-hidden weights `[3*hidden_dim, hidden_dim]` (ONNX `[Z;R;H]`).
pub const TENSOR_GRU_2_W: &str = "gru_2.W";
/// GRU-2 recurrent weights `[3*hidden_dim, hidden_dim]` (ONNX `[Z;R;H]`).
pub const TENSOR_GRU_2_R: &str = "gru_2.R";
/// GRU-2 bias `[6*hidden_dim]` (ONNX `[Wb;Rb]` packing).
pub const TENSOR_GRU_2_B: &str = "gru_2.B";
/// First-post-GRU Linear weight `[fc1_dim, hidden_dim]` (`fc_1.weight`).
pub const TENSOR_FC_1_WEIGHT: &str = "fc_1.weight";
/// First-post-GRU Linear bias `[fc1_dim]`.
pub const TENSOR_FC_1_BIAS: &str = "fc_1.bias";
/// Second-post-GRU Linear weight `[fc2_dim, fc1_dim]`.
pub const TENSOR_FC_2_WEIGHT: &str = "fc_2.weight";
/// Second-post-GRU Linear bias `[fc2_dim]`.
pub const TENSOR_FC_2_BIAS: &str = "fc_2.bias";
/// Mask head Linear weight `[n_bins, fc2_dim]` (`mask.weight`).
pub const TENSOR_MASK_WEIGHT: &str = "mask.weight";
/// Mask head Linear bias `[n_bins]` (`mask.bias`).
pub const TENSOR_MASK_BIAS: &str = "mask.bias";

/// Floor added under the log to guard against `log(0)`.
const LOG_EPS: f32 = 1e-7;

// -------------------------------------------------------------------------
// Config
// -------------------------------------------------------------------------

/// NSNet2 runtime config (transcribed verbatim from `vokra.nsnet2.*` at
/// load time). Every field is validated at [`Self::validate`]; a
/// `0`-sentinel on any of them is a load-time refusal (FR-EX-08).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nsnet2Config {
    /// STFT bin count (upstream = 257 = `n_fft/2 + 1`).
    pub n_bins: usize,
    /// GRU / fc_in hidden width (upstream = 400).
    pub hidden_dim: usize,
    /// `fc_1` output width (upstream = 600).
    pub fc1_dim: usize,
    /// `fc_2` output width (upstream = 600).
    pub fc2_dim: usize,
    /// STFT FFT length (upstream = 512).
    pub n_fft: usize,
    /// STFT hop in samples (upstream = 160 = 10 ms @ 16 kHz).
    pub hop: usize,
    /// STFT window length in samples (upstream = 320 = 20 ms @ 16 kHz).
    pub win_length: usize,
    /// PCM sample rate (Hz; upstream = 16 000).
    pub sample_rate: u32,
}

impl Nsnet2Config {
    /// The upstream default (fixed at the released 20 ms baseline —
    /// there is no other topology in circulation).
    pub fn upstream_default() -> Self {
        Self {
            n_bins: 257,
            hidden_dim: 400,
            fc1_dim: 600,
            fc2_dim: 600,
            n_fft: 512,
            hop: 160,
            win_length: 320,
            sample_rate: 16_000,
        }
    }

    /// Validates the config loudly (FR-EX-08).
    ///
    /// Every field is refused at `0`, and the cross-invariant
    /// `n_bins == n_fft/2 + 1` is enforced (a checkpoint that came from
    /// a different `n_fft` would silently misshape the log-power
    /// feature otherwise).
    pub fn validate(&self) -> Result<()> {
        for (label, v) in [
            ("n_bins", self.n_bins as u64),
            ("hidden_dim", self.hidden_dim as u64),
            ("fc1_dim", self.fc1_dim as u64),
            ("fc2_dim", self.fc2_dim as u64),
            ("n_fft", self.n_fft as u64),
            ("hop", self.hop as u64),
            ("win_length", self.win_length as u64),
            ("sample_rate", u64::from(self.sample_rate)),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "nsnet2 config: {label} must be > 0 (got 0 — the GGUF's \
                     vokra.nsnet2.* chunk is missing or malformed)"
                )));
            }
        }
        // `n_bins == n_fft/2 + 1` is a load-bearing invariant of the
        // real-input STFT pipeline; violating it means the fc_in Linear
        // silently connects the wrong number of columns.
        let expected_bins = self.n_fft / 2 + 1;
        if self.n_bins != expected_bins {
            return Err(VokraError::InvalidArgument(format!(
                "nsnet2 config: n_bins ({}) must equal n_fft/2 + 1 (= {}) for the \
                 real-input STFT pipeline",
                self.n_bins, expected_bins,
            )));
        }
        if self.win_length > self.n_fft {
            return Err(VokraError::InvalidArgument(format!(
                "nsnet2 config: win_length ({}) must be <= n_fft ({}); a longer \
                 window cannot be centred and zero-padded",
                self.win_length, self.n_fft,
            )));
        }
        Ok(())
    }

    /// Reads config from `vokra.nsnet2.*` metadata in a parsed GGUF.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on any missing / non-fitting chunk
    /// (FR-EX-08 — no silent zero-default).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| -> Result<u32> {
            let v = gguf.get(k).and_then(|v| v.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!("nsnet2 GGUF missing required u32 metadata `{k}`"))
            })?;
            u32::try_from(v).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "nsnet2 GGUF metadata `{k}` = {v} does not fit in u32"
                ))
            })
        };
        let cfg = Self {
            n_bins: get_u32(KEY_N_BINS)? as usize,
            hidden_dim: get_u32(KEY_HIDDEN_DIM)? as usize,
            fc1_dim: get_u32(KEY_FC1_DIM)? as usize,
            fc2_dim: get_u32(KEY_FC2_DIM)? as usize,
            n_fft: get_u32(KEY_N_FFT)? as usize,
            hop: get_u32(KEY_HOP)? as usize,
            win_length: get_u32(KEY_WIN_LENGTH)? as usize,
            sample_rate: get_u32(KEY_SAMPLE_RATE)?,
        };
        cfg.validate()
            .map_err(|e| VokraError::ModelLoad(e.to_string()))?;
        Ok(cfg)
    }
}

// -------------------------------------------------------------------------
// Weights
// -------------------------------------------------------------------------

/// NSNet2 weight bundle: three Linears + two GRU cells + the mask head.
///
/// GRU rows are stored in the **`[R; Z; H]`** layout (already permuted
/// from ONNX `[Z; R; H]` by the loader so [`rnnoise_gru_forward`] can
/// consume them verbatim). Bias vectors are `[3 * hidden_dim]` — ONNX's
/// input + recurrent bias halves are summed together at load time
/// (they contribute additively to each gate under ONNX's default).
#[derive(Debug, Clone)]
pub struct Nsnet2Weights {
    /// `fc_in.weight`, row-major `[hidden_dim, n_bins]`.
    pub fc_in_weight: Vec<f32>,
    /// `fc_in.bias`, `[hidden_dim]`.
    pub fc_in_bias: Vec<f32>,
    /// `gru_1.W` (input-hidden) after `[Z;R;H]` → `[R;Z;H]` permutation,
    /// row-major `[3 * hidden_dim, hidden_dim]`.
    pub gru_1_w_ih: Vec<f32>,
    /// `gru_1.R` (recurrent) after `[Z;R;H]` → `[R;Z;H]` permutation,
    /// row-major `[3 * hidden_dim, hidden_dim]`.
    pub gru_1_w_hh: Vec<f32>,
    /// `gru_1.B` combined bias `Wb + Rb`, permuted to `[R;Z;H]`, length
    /// `3 * hidden_dim`.
    pub gru_1_bias: Vec<f32>,
    /// `gru_2.W`, same layout as `gru_1_w_ih`.
    pub gru_2_w_ih: Vec<f32>,
    /// `gru_2.R`, same layout as `gru_1_w_hh`.
    pub gru_2_w_hh: Vec<f32>,
    /// `gru_2.B`, same layout as `gru_1_bias`.
    pub gru_2_bias: Vec<f32>,
    /// `fc_1.weight`, row-major `[fc1_dim, hidden_dim]`.
    pub fc_1_weight: Vec<f32>,
    /// `fc_1.bias`, `[fc1_dim]`.
    pub fc_1_bias: Vec<f32>,
    /// `fc_2.weight`, row-major `[fc2_dim, fc1_dim]`.
    pub fc_2_weight: Vec<f32>,
    /// `fc_2.bias`, `[fc2_dim]`.
    pub fc_2_bias: Vec<f32>,
    /// `mask.weight`, row-major `[n_bins, fc2_dim]`.
    pub mask_weight: Vec<f32>,
    /// `mask.bias`, `[n_bins]`.
    pub mask_bias: Vec<f32>,
}

// -------------------------------------------------------------------------
// Model
// -------------------------------------------------------------------------

/// NSNet2 model — an immutable shareable weight bundle plus the config
/// it was bound against.
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open), then
/// obtain a stateful stream through the [`DenoiseEngine`] trait
/// ([`open_stream`](DenoiseEngine::open_stream)). All mutable recurrent
/// state (GRU hidden vectors, iSTFT tail buffer, unconsumed PCM
/// samples) lives in the stream handle (FR-LD-06).
#[derive(Debug)]
pub struct Nsnet2V1 {
    cfg: Nsnet2Config,
    weights: Arc<Nsnet2Weights>,
}

impl Nsnet2V1 {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Returns [`VokraError::ModelLoad`] on any missing hparam / tensor
    /// or on any shape mismatch (FR-EX-08 — no silent reshape).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Verify the arch tag first so a wrong-family GGUF handed to us
        // by mistake fails with a clear message instead of a downstream
        // "missing tensor" surprise.
        match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "nsnet2: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "nsnet2: GGUF is missing `vokra.model.arch` (converter did not stamp it)"
                        .to_owned(),
                ));
            }
        }

        let cfg = Nsnet2Config::from_gguf(gguf)?;
        let weights = load_weights(gguf, &cfg)?;
        Ok(Self {
            cfg,
            weights: Arc::new(weights),
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns the checkpoint's config.
    pub fn config(&self) -> &Nsnet2Config {
        &self.cfg
    }

    /// Convenience one-shot: denoises `pcm` end-to-end starting from a
    /// fresh zero state and returns the enhanced PCM. Equivalent to
    /// opening a stream, pushing the whole buffer, and finalising.
    pub fn denoise_pcm(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mut stream = Nsnet2Stream::new(self.cfg.clone(), Arc::clone(&self.weights))?;
        let mut out = stream.push_pcm_internal(pcm)?;
        out.extend(stream.finalize()?);
        Ok(out)
    }
}

impl DenoiseEngine for Nsnet2V1 {
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn DenoiseStreamHandle + Send>> {
        if sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "nsnet2: sample rate mismatch — requested {sample_rate} Hz but the model \
                 was trained for {} Hz (resample upstream, or open a stream on the matching \
                 rate)",
                self.cfg.sample_rate
            )));
        }
        let stream = Nsnet2Stream::new(self.cfg.clone(), Arc::clone(&self.weights))?;
        Ok(Box::new(stream))
    }
}

// -------------------------------------------------------------------------
// Stream
// -------------------------------------------------------------------------

/// Stateful NSNet2 denoise stream — hides every recurrent tail (FR-LD-06).
pub struct Nsnet2Stream {
    cfg: Nsnet2Config,
    weights: Arc<Nsnet2Weights>,
    /// GRU-1 hidden state, `[hidden_dim]`.
    h1: Vec<f32>,
    /// GRU-2 hidden state, `[hidden_dim]`.
    h2: Vec<f32>,
    /// Rolling raw-PCM tail: samples not yet consumed into a complete
    /// STFT frame. Snip-edges framing keeps the last `n_fft - hop`
    /// samples of any pending window here until the next call closes a
    /// frame.
    pending_pcm: Vec<f32>,
    /// Streaming iSTFT state (tail buffer + overlap-add accumulator).
    /// Owns the inter-frame overlap so successive `push_pcm` calls
    /// reproduce a whole-utterance call bit-for-bit up to steady state.
    istft: IstftStreamingState,
    /// Cached STFT attrs so per-call `push_pcm` does not re-derive them.
    stft_attrs: StftAttrs,
    /// Cached synthesis attrs (used by the streaming iSTFT).
    #[allow(dead_code)]
    istft_attrs: IstftStreamingAttrs,
    /// Number of frames pushed through the analysis so far (used only
    /// for diagnostic assertions).
    frames_seen: usize,
}

impl Nsnet2Stream {
    fn new(cfg: Nsnet2Config, weights: Arc<Nsnet2Weights>) -> Result<Self> {
        let stft_attrs = analysis_stft_attrs(&cfg);
        let istft_attrs = synthesis_istft_attrs(&cfg);
        let istft = IstftStreamingState::new(&istft_attrs)?;
        let h1 = vec![0.0f32; cfg.hidden_dim];
        let h2 = vec![0.0f32; cfg.hidden_dim];
        Ok(Self {
            cfg,
            weights,
            h1,
            h2,
            pending_pcm: Vec::new(),
            istft,
            stft_attrs,
            istft_attrs,
            frames_seen: 0,
        })
    }

    /// One-shot push: runs analysis + core forward + synthesis on
    /// `pcm`, buffering the rolling PCM tail across calls. This is the
    /// inner entry-point `denoise_pcm` and the [`DenoiseStreamHandle`]
    /// impl share.
    fn push_pcm_internal(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.pending_pcm.extend_from_slice(pcm);

        // How many complete hop-strided frames can we emit? Analysis
        // needs at least `n_fft` samples for the first frame; every
        // additional frame needs `hop` more. Snip-edges keeps the last
        // `n_fft - hop` samples for the next call.
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.hop;
        if self.pending_pcm.len() < n_fft {
            return Ok(Vec::new());
        }
        let n_frames = (self.pending_pcm.len() - n_fft) / hop + 1;
        let consume = (n_frames - 1) * hop + n_fft;
        // We do NOT drain `consume` samples up front: `torch.stft` with
        // `center=True` needs `n_fft/2` reflect-padding on both ends,
        // which spans across the current-and-next window. For the
        // streaming NSNet2 pipeline we intentionally use `center=false`
        // (a causal one-sided window) so no future samples are needed.
        // See `analysis_stft_attrs` for the rationale pin.
        let analysis_slice = &self.pending_pcm[..consume];

        // Run one STFT chunk. `stft` allocates its own `[frames, bins]`
        // Spectrogram; we consume it in-place below (no clone).
        let spec = stft(analysis_slice, &self.stft_attrs)?;
        if spec.bins != self.cfg.n_bins {
            return Err(VokraError::InvalidArgument(format!(
                "nsnet2: STFT produced {} bins, expected {} (check n_fft / real_input agreement)",
                spec.bins, self.cfg.n_bins,
            )));
        }
        if spec.frames != n_frames {
            return Err(VokraError::InvalidArgument(format!(
                "nsnet2: STFT produced {} frames, expected {n_frames} (framing off-by-one)",
                spec.frames,
            )));
        }

        // Compute the per-frame gain and apply it to the complex STFT
        // in-place. We accumulate the gated (Y = G * X) result into new
        // vectors sized `frames * bins` (no allocation per frame).
        let mut y_re = vec![0.0f32; spec.frames * spec.bins];
        let mut y_im = vec![0.0f32; spec.frames * spec.bins];
        for f in 0..spec.frames {
            let base = f * spec.bins;
            let re_row = &spec.re[base..base + spec.bins];
            let im_row = &spec.im[base..base + spec.bins];

            // log(|X|^2 + eps) — the NSNet2 input feature (per arXiv:2005.07551 §3.2).
            let mut feature = vec![0.0f32; spec.bins];
            for k in 0..spec.bins {
                let p = re_row[k] * re_row[k] + im_row[k] * im_row[k];
                feature[k] = (p + LOG_EPS).ln();
            }

            // fc_in -> ReLU
            let x = linear_relu(
                &feature,
                &self.weights.fc_in_weight,
                &self.weights.fc_in_bias,
                self.cfg.hidden_dim,
                self.cfg.n_bins,
            )?;

            // gru_1 (stateful h1)
            rnnoise_gru_forward(
                &x,
                &mut self.h1,
                &self.weights.gru_1_w_ih,
                &self.weights.gru_1_w_hh,
                &self.weights.gru_1_bias,
            )?;

            // gru_2 (stateful h2, input = h1)
            rnnoise_gru_forward(
                &self.h1,
                &mut self.h2,
                &self.weights.gru_2_w_ih,
                &self.weights.gru_2_w_hh,
                &self.weights.gru_2_bias,
            )?;

            // fc_1 + ReLU -> fc_2 + ReLU -> mask + sigmoid
            let a = linear_relu(
                &self.h2,
                &self.weights.fc_1_weight,
                &self.weights.fc_1_bias,
                self.cfg.fc1_dim,
                self.cfg.hidden_dim,
            )?;
            let b = linear_relu(
                &a,
                &self.weights.fc_2_weight,
                &self.weights.fc_2_bias,
                self.cfg.fc2_dim,
                self.cfg.fc1_dim,
            )?;
            let gain_pre = linear(
                &b,
                &self.weights.mask_weight,
                &self.weights.mask_bias,
                self.cfg.n_bins,
                self.cfg.fc2_dim,
            )?;

            // Sigmoid — bounded ∈ [0, 1]. Apply to complex STFT
            // (phase preserved verbatim per arXiv:2005.07551 §3.2).
            for k in 0..spec.bins {
                let g = sigmoid_stable(gain_pre[k]);
                y_re[base + k] = g * re_row[k];
                y_im[base + k] = g * im_row[k];
            }
        }
        self.frames_seen += spec.frames;

        // Feed the gated spectrogram to the streaming iSTFT.
        let gated = Spectrogram {
            frames: spec.frames,
            bins: spec.bins,
            re: y_re,
            im: y_im,
        };
        let pcm_out = self.istft.push(&gated)?;

        // Drop the samples we have already consumed *up to but not
        // including* the overlap tail that the next window needs. For a
        // non-`center` analysis with `hop <= n_fft`, the next window
        // starts at absolute sample `n_frames * hop`.
        let drop = (n_frames * hop).min(self.pending_pcm.len());
        self.pending_pcm.drain(..drop);
        Ok(pcm_out)
    }

    /// Flushes the streaming iSTFT tail once no more PCM will arrive.
    /// Idempotent: a second call returns an empty vec.
    pub fn finalize(&mut self) -> Result<Vec<f32>> {
        Ok(self.istft.finish())
    }

    /// GRU-1 hidden state (read-only view, for parity harnesses).
    pub fn gru_1_hidden(&self) -> &[f32] {
        &self.h1
    }

    /// GRU-2 hidden state (read-only view, for parity harnesses).
    pub fn gru_2_hidden(&self) -> &[f32] {
        &self.h2
    }
}

impl std::fmt::Debug for Nsnet2Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nsnet2Stream")
            .field("cfg", &self.cfg)
            .field("h1_len", &self.h1.len())
            .field("h2_len", &self.h2.len())
            .field("pending_pcm_len", &self.pending_pcm.len())
            .field("frames_seen", &self.frames_seen)
            .finish()
    }
}

impl DenoiseStreamHandle for Nsnet2Stream {
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.push_pcm_internal(pcm)
    }

    fn reset(&mut self) {
        self.h1.fill(0.0);
        self.h2.fill(0.0);
        self.pending_pcm.clear();
        self.istft.reset();
        self.frames_seen = 0;
    }
}

// -------------------------------------------------------------------------
// Analysis / synthesis attributes
// -------------------------------------------------------------------------

/// Analysis STFT attributes matching the NSNet2 training pipeline (`n_fft=512`,
/// `hop=160`, `win=320`, Hann, backward-normalised).
///
/// `center = false` — the streaming NSNet2 pipeline is causal: a frame `t`
/// depends only on samples `<= t · hop + n_fft`. Enabling `center` would
/// require `n_fft/2` future samples of look-ahead, which is not available
/// in a real-time denoiser. The upstream ONNX model handles the framing
/// externally, so bit-exact reproduction against the upstream is the
/// concern of the parity harness (env-gated); the design here matches the
/// causal-streaming contract the runtime deploys.
fn analysis_stft_attrs(cfg: &Nsnet2Config) -> StftAttrs {
    let mut a = StftAttrs::new(cfg.n_fft, cfg.hop);
    a.win_length = cfg.win_length;
    // Neither `center` (n_fft/2 look-ahead / look-behind pad) nor
    // `causal` (n_fft-hop left pad) is used: the streaming pipeline
    // gets its "first-frame" left-context naturally by holding back
    // the trailing `n_fft-hop` samples of `pending_pcm` between pushes
    // (snip-edges framing). Under this pure raw-frames mode the STFT
    // frame count is `(len - n_fft)/hop + 1`, matching the pending-PCM
    // drain arithmetic in `push_pcm_internal`.
    a.center = false;
    a.causal = false;
    a
}

/// Synthesis iSTFT attributes: same window / hop as the analysis so the
/// COLA (constant-overlap-add) invariant holds. `center = false` mirrors
/// the analysis so the head / tail trims match (both zero).
fn synthesis_istft_attrs(cfg: &Nsnet2Config) -> IstftStreamingAttrs {
    let mut i = IstftAttrs::new(cfg.n_fft, cfg.hop);
    i.win_length = cfg.win_length;
    i.center = false;
    IstftStreamingAttrs::from_istft(i)
}

// -------------------------------------------------------------------------
// Loader helper
// -------------------------------------------------------------------------

fn load_weights(gguf: &GgufFile, cfg: &Nsnet2Config) -> Result<Nsnet2Weights> {
    let load_f32 = |name: &str, expect: usize| -> Result<Vec<f32>> {
        let v = gguf.tensor_f32(name).map_err(|e| {
            VokraError::ModelLoad(format!("nsnet2: tensor `{name}` load failed: {e}"))
        })?;
        if v.len() != expect {
            return Err(VokraError::ModelLoad(format!(
                "nsnet2: tensor `{name}` has {} elements, expected {expect}",
                v.len()
            )));
        }
        Ok(v)
    };
    let assert_dims = |name: &str, want: &[u64]| -> Result<()> {
        let info = gguf.tensor_info(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "nsnet2: tensor `{name}` info unavailable after successful load"
            ))
        })?;
        if info.dimensions.as_slice() != want {
            return Err(VokraError::ModelLoad(format!(
                "nsnet2: tensor `{name}` dims {:?} — expected {want:?} row-major \
                 (see docstring on Nsnet2Weights for the layout contract)",
                info.dimensions,
            )));
        }
        Ok(())
    };

    let h = cfg.hidden_dim;
    let n_bins = cfg.n_bins;
    let fc1 = cfg.fc1_dim;
    let fc2 = cfg.fc2_dim;

    // Input Linear.
    let fc_in_weight = load_f32(TENSOR_FC_IN_WEIGHT, h * n_bins)?;
    assert_dims(TENSOR_FC_IN_WEIGHT, &[h as u64, n_bins as u64])?;
    let fc_in_bias = load_f32(TENSOR_FC_IN_BIAS, h)?;
    assert_dims(TENSOR_FC_IN_BIAS, &[h as u64])?;

    // GRU-1: ONNX ships `[Z; R; H]` — permute to `[R; Z; H]` to match
    // `rnnoise_gru_forward`. Bias is `[Wb_ZRH; Rb_ZRH]` = 6*hidden
    // long; ONNX default sums the two halves for each gate.
    let gru_1_w_raw = load_f32(TENSOR_GRU_1_W, 3 * h * h)?;
    assert_dims(TENSOR_GRU_1_W, &[(3 * h) as u64, h as u64])?;
    let gru_1_r_raw = load_f32(TENSOR_GRU_1_R, 3 * h * h)?;
    assert_dims(TENSOR_GRU_1_R, &[(3 * h) as u64, h as u64])?;
    let gru_1_b_raw = load_f32(TENSOR_GRU_1_B, 6 * h)?;
    assert_dims(TENSOR_GRU_1_B, &[(6 * h) as u64])?;

    let gru_2_w_raw = load_f32(TENSOR_GRU_2_W, 3 * h * h)?;
    assert_dims(TENSOR_GRU_2_W, &[(3 * h) as u64, h as u64])?;
    let gru_2_r_raw = load_f32(TENSOR_GRU_2_R, 3 * h * h)?;
    assert_dims(TENSOR_GRU_2_R, &[(3 * h) as u64, h as u64])?;
    let gru_2_b_raw = load_f32(TENSOR_GRU_2_B, 6 * h)?;
    assert_dims(TENSOR_GRU_2_B, &[(6 * h) as u64])?;

    let (gru_1_w_ih, gru_1_w_hh, gru_1_bias) =
        permute_onnx_gru(&gru_1_w_raw, &gru_1_r_raw, &gru_1_b_raw, h, h);
    let (gru_2_w_ih, gru_2_w_hh, gru_2_bias) =
        permute_onnx_gru(&gru_2_w_raw, &gru_2_r_raw, &gru_2_b_raw, h, h);

    // Post-GRU Linears + mask head.
    let fc_1_weight = load_f32(TENSOR_FC_1_WEIGHT, fc1 * h)?;
    assert_dims(TENSOR_FC_1_WEIGHT, &[fc1 as u64, h as u64])?;
    let fc_1_bias = load_f32(TENSOR_FC_1_BIAS, fc1)?;
    assert_dims(TENSOR_FC_1_BIAS, &[fc1 as u64])?;

    let fc_2_weight = load_f32(TENSOR_FC_2_WEIGHT, fc2 * fc1)?;
    assert_dims(TENSOR_FC_2_WEIGHT, &[fc2 as u64, fc1 as u64])?;
    let fc_2_bias = load_f32(TENSOR_FC_2_BIAS, fc2)?;
    assert_dims(TENSOR_FC_2_BIAS, &[fc2 as u64])?;

    let mask_weight = load_f32(TENSOR_MASK_WEIGHT, n_bins * fc2)?;
    assert_dims(TENSOR_MASK_WEIGHT, &[n_bins as u64, fc2 as u64])?;
    let mask_bias = load_f32(TENSOR_MASK_BIAS, n_bins)?;
    assert_dims(TENSOR_MASK_BIAS, &[n_bins as u64])?;

    Ok(Nsnet2Weights {
        fc_in_weight,
        fc_in_bias,
        gru_1_w_ih,
        gru_1_w_hh,
        gru_1_bias,
        gru_2_w_ih,
        gru_2_w_hh,
        gru_2_bias,
        fc_1_weight,
        fc_1_bias,
        fc_2_weight,
        fc_2_bias,
        mask_weight,
        mask_bias,
    })
}

/// Permutes an ONNX GRU triplet (`W [Z;R;H]`, `R [Z;R;H]`, `B [Wb_ZRH; Rb_ZRH]`)
/// into the `[R; Z; H]` layout with fused biases that
/// [`rnnoise_gru_forward`] consumes verbatim.
///
/// The row block size in `W` is `[hidden, in_dim]`; for `R` it is
/// `[hidden, hidden]`. ONNX packs the bias as two halves of size
/// `3 * hidden` (`Wb` = input, `Rb` = recurrent); NSNet2 sums them
/// (ONNX `linear_before_reset=1` PyTorch export — see module docstring
/// for the audit trail).
///
/// Returns `(w_ih_permuted, w_hh_permuted, bias_fused_permuted)`.
fn permute_onnx_gru(
    w: &[f32],
    r: &[f32],
    b: &[f32],
    hidden: usize,
    in_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_ih = vec![0.0f32; 3 * hidden * in_dim];
    let mut w_hh = vec![0.0f32; 3 * hidden * hidden];
    let mut bias = vec![0.0f32; 3 * hidden];

    // ONNX order is Z (update), R (reset), H (hidden/new).
    // rnnoise order is R (reset), Z (update), N (new).
    // So we swap the first two blocks of `hidden` rows.
    let block_w = hidden * in_dim;
    let block_r = hidden * hidden;

    // Z (onnx) -> Z (rnnoise index 1)
    w_ih[block_w..2 * block_w].copy_from_slice(&w[0..block_w]);
    w_hh[block_r..2 * block_r].copy_from_slice(&r[0..block_r]);
    // R (onnx) -> R (rnnoise index 0)
    w_ih[0..block_w].copy_from_slice(&w[block_w..2 * block_w]);
    w_hh[0..block_r].copy_from_slice(&r[block_r..2 * block_r]);
    // H (onnx) -> N (rnnoise index 2)
    w_ih[2 * block_w..3 * block_w].copy_from_slice(&w[2 * block_w..3 * block_w]);
    w_hh[2 * block_r..3 * block_r].copy_from_slice(&r[2 * block_r..3 * block_r]);

    // Bias: `b` is `[Wb_z | Wb_r | Wb_h | Rb_z | Rb_r | Rb_h]`. Sum
    // input + recurrent halves per gate, then permute Z/R.
    let wb_z = &b[0..hidden];
    let wb_r = &b[hidden..2 * hidden];
    let wb_h = &b[2 * hidden..3 * hidden];
    let rb_z = &b[3 * hidden..4 * hidden];
    let rb_r = &b[4 * hidden..5 * hidden];
    let rb_h = &b[5 * hidden..6 * hidden];
    for i in 0..hidden {
        // R (index 0)
        bias[i] = wb_r[i] + rb_r[i];
        // Z (index 1)
        bias[hidden + i] = wb_z[i] + rb_z[i];
        // N (index 2)
        bias[2 * hidden + i] = wb_h[i] + rb_h[i];
    }
    (w_ih, w_hh, bias)
}

// -------------------------------------------------------------------------
// Small numeric helpers (kept local — 2-GRU-cell topology, no primitive
// extraction until a second consumer lands).
// -------------------------------------------------------------------------

/// `y = max(0, W · x + b)` where `W` is row-major `[out_dim, in_dim]`.
fn linear_relu(x: &[f32], w: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Result<Vec<f32>> {
    let mut y = linear(x, w, b, out_dim, in_dim)?;
    for v in &mut y {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    Ok(y)
}

/// `y = W · x + b` where `W` is row-major `[out_dim, in_dim]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize, in_dim: usize) -> Result<Vec<f32>> {
    if x.len() != in_dim {
        return Err(VokraError::InvalidArgument(format!(
            "nsnet2 linear: x has {} elements, expected in_dim={in_dim}",
            x.len(),
        )));
    }
    if w.len() != out_dim * in_dim {
        return Err(VokraError::InvalidArgument(format!(
            "nsnet2 linear: W has {} elements, expected out_dim*in_dim={}",
            w.len(),
            out_dim * in_dim,
        )));
    }
    if b.len() != out_dim {
        return Err(VokraError::InvalidArgument(format!(
            "nsnet2 linear: b has {} elements, expected out_dim={out_dim}",
            b.len(),
        )));
    }
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = b[i];
        let row = &w[i * in_dim..(i + 1) * in_dim];
        for (wij, xj) in row.iter().zip(x.iter()) {
            acc += wij * xj;
        }
        y[i] = acc;
    }
    Ok(y)
}

/// Numerically stable sigmoid (uses `tanh` — same trick as
/// `rnnoise::sigmoid`, no `exp` overflow at large magnitudes).
#[inline]
fn sigmoid_stable(x: f32) -> f32 {
    0.5 * (0.5 * x).tanh() + 0.5
}
