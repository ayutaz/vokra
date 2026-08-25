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
//! 2-layer GRU + 3-Linear mask predictor over the 161-bin log-power
//! STFT of 16 kHz PCM — structurally distinct from either sibling so it
//! carries its own `vokra.model.arch = "nsnet2"` tag and its own
//! `vokra.nsnet2.*` hparam chunk group. Silently sharing an arch tag
//! would misroute the runtime dispatch.
//!
//! # Architecture (source of truth: arXiv:2005.07551 §3.2)
//!
//! ```text
//! PCM (16 kHz mono f32)
//!   -> `vokra_ops::stft` (n_fft=320, hop=160, sqrt-Hann, no center)
//!      -> 161 complex bins per 10 ms frame
//!   -> log10(max(|X|^2, 1e-12))                            [t, 161]
//!   -> `fc_in`  Linear(161 -> 400) + ReLU                  [t, 400]
//!   -> `gru_1`  GRU(400 -> 400, stateful)                  [t, 400]
//!   -> `gru_2`  GRU(400 -> 400, stateful)                  [t, 400]
//!   -> `fc_1`   Linear(400 -> 600) + ReLU                  [t, 600]
//!   -> `fc_2`   Linear(600 -> 600) + ReLU                  [t, 600]
//!   -> `mask`   Linear(600 -> 161) + sigmoid               [t, 161]
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
//! export. Its `linear_before_reset=1` candidate is
//! `W_h x + Wb_h + r · (R_h h + Rb_h)`.
//!
//! The loader permutes row blocks once at bind time (`[Z; R; H]` →
//! `[R; Z; H]`) and keeps input/recurrent biases separate. The forward
//! path never touches ONNX-native layout after that.
//!
//! # Real-weight parity posture
//!
//! The canonical 14-tensor artifact passed the env-gated independent official
//! ONNX reference and CPU/Metal PCM legs on 2026-08-24 (fixed `5e-5` bound;
//! see `docs/handoff/mac-cpu-metal-coverage-2026-08-24.md`). The same harness
//! (`crates/vokra-models/tests/parity_nsnet2.rs`,
//! `VOKRA_NSNET2_REAL_GGUF` + `VOKRA_NSNET2_REAL_WAV`) accepts either the
//! canonical conversion or the exact historical public Hub contract. This
//! module ships:
//!
//! - the exact canonical tensor / hparam contract and the immutable historical
//!   public header contract [`Nsnet2V1::from_gguf`] binds against;
//! - synthetic-weight structural tests pinning FR-EX-08 (loud errors on
//!   every shape / rate / tensor-name mismatch);
//! - identity-gain sanity: a synthetic mask that forces the sigmoid
//!   pre-activation to +∞ must reproduce the input STFT bit-for-bit up
//!   to the streaming iSTFT accumulator's steady-state region.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::engines::{DenoiseEngine, DenoiseStreamHandle};
use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::ir::graph::{IstftAttrs, IstftStreamingAttrs, StftAttrs, Window, WindowSymmetry};
use vokra_core::{Result, VokraError};
use vokra_ops::{IstftStreamingState, Spectrogram, stft};

use crate::compute::{Compute, HotOp};

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

/// Backend-dispatched NSNet2 hot ops. The STFT/iSTFT and scalar nonlinearities
/// remain host-side DSP/glue; every learned projection and the complex mask
/// application go through these complete backend seams.
const NSNET2_HOT_OPS: &[HotOp] = &[HotOp::Gemv, HotOp::DenoiseApplyMask];

// ---- vokra.nsnet2.* metadata keys ---------------------------------------
//
// Mirror of the converter-side constants (kept duplicated per the
// runtime-vs-converter no-cross-dep convention). Any change here must be
// mirrored in `vokra-convert::models::nsnet2::KEY_*`.

/// GGUF metadata key: STFT bin count (u32; upstream = 161).
pub const KEY_N_BINS: &str = "vokra.nsnet2.n_bins";
/// GGUF metadata key: GRU / fc_in hidden width (u32; upstream = 400).
pub const KEY_HIDDEN_DIM: &str = "vokra.nsnet2.hidden_dim";
/// GGUF metadata key: `fc_1` output width (u32; upstream = 600).
pub const KEY_FC1_DIM: &str = "vokra.nsnet2.fc1_dim";
/// GGUF metadata key: `fc_2` output width (u32; upstream = 600).
pub const KEY_FC2_DIM: &str = "vokra.nsnet2.fc2_dim";
/// GGUF metadata key: STFT FFT length (u32; upstream = 320).
pub const KEY_N_FFT: &str = "vokra.nsnet2.n_fft";
/// GGUF metadata key: STFT hop (u32 samples; upstream = 160).
pub const KEY_HOP: &str = "vokra.nsnet2.hop";
/// GGUF metadata key: STFT window length (u32 samples; upstream = 320).
pub const KEY_WIN_LENGTH: &str = "vokra.nsnet2.win_length";
/// GGUF metadata key: PCM sample rate (u32 Hz; upstream = 16 000).
pub const KEY_SAMPLE_RATE: &str = "vokra.nsnet2.sample_rate";

const HPARAM_KEYS: &[&str] = &[
    KEY_N_BINS,
    KEY_HIDDEN_DIM,
    KEY_FC1_DIM,
    KEY_FC2_DIM,
    KEY_N_FFT,
    KEY_HOP,
    KEY_WIN_LENGTH,
    KEY_SAMPLE_RATE,
];

// ---- tensor-name convention ---------------------------------------------
//
// The prep sidecar (`tools/parity/nsnet2_prepare_checkpoint.py`) preserves the
// numeric ONNX initializer names. The canonical converter validates that fixed
// source manifest, then renames it to the semantic `fc_in.weight` / `gru_1.W`
// / `mask.bias` contract below and normalizes MatMul axes. These `TENSOR_*`
// constants are the runtime spelling; the exact old numeric layout is isolated
// in `LEGACY_PUBLIC_TENSORS`.

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

const CANONICAL_TENSOR_NAMES: &[&str] = &[
    TENSOR_FC_IN_WEIGHT,
    TENSOR_FC_IN_BIAS,
    TENSOR_GRU_1_W,
    TENSOR_GRU_1_R,
    TENSOR_GRU_1_B,
    TENSOR_GRU_2_W,
    TENSOR_GRU_2_R,
    TENSOR_GRU_2_B,
    TENSOR_FC_1_WEIGHT,
    TENSOR_FC_1_BIAS,
    TENSOR_FC_2_WEIGHT,
    TENSOR_FC_2_BIAS,
    TENSOR_MASK_WEIGHT,
    TENSOR_MASK_BIAS,
];

// The first public `vokra/nsnet2` GGUF predates the strict semantic tensor
// schema.  Its header was audited remotely (Range request: no tensor payload)
// at this immutable Hub revision. Keep its source revision and content digest
// beside the header contract as audit evidence. The runtime gates the complete
// header, not the whole-file digest; a bit-identical payload hash is rechecked
// by publication/verification tooling. Missing metadata never becomes an
// unbounded "use defaults" rule.
const LEGACY_PUBLIC_REVISION: &str = "983e1cc1397810201f93a121a9daf60cf247813b";
const LEGACY_PUBLIC_SHA256: &str =
    "abeca882165909fb0897b39b97882d0ebd9f95cf176a4d2e58482e52a8b19e13";
const LEGACY_PUBLIC_SOURCE: &str = "Microsoft DNS-Challenge NSNet2-baseline (MIT end-to-end)";
const LEGACY_PUBLIC_UPSTREAM_URL: &str =
    "github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline";
const LEGACY_PUBLIC_SCHEMA_PRODUCER: &str = "vokra-core 0.1.0-alpha.0";
const CURRENT_SCHEMA_PRODUCER: &str = concat!("vokra-core ", env!("CARGO_PKG_VERSION"));
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

#[derive(Debug, Clone, Copy)]
struct LegacyTensorSpec {
    name: &'static str,
    dimensions: &'static [u64],
    offset: u64,
}

// Exact tensor directory observed at the immutable Hub revision above.  The
// offset is relative to the tensor-data region, as GGUF v3 specifies.  Names,
// order, dtype, dimensions and offsets must all match before fixed topology is
// repaired.  This is intentionally stricter than count-only compatibility.
const LEGACY_PUBLIC_TENSORS: &[LegacyTensorSpec] = &[
    LegacyTensorSpec {
        name: "172",
        dimensions: &[161, 400],
        offset: 0,
    },
    LegacyTensorSpec {
        name: "192",
        dimensions: &[1, 1200, 400],
        offset: 257_600,
    },
    LegacyTensorSpec {
        name: "193",
        dimensions: &[1, 1200, 400],
        offset: 2_177_600,
    },
    LegacyTensorSpec {
        name: "194",
        dimensions: &[1, 2400],
        offset: 4_097_600,
    },
    LegacyTensorSpec {
        name: "212",
        dimensions: &[1, 1200, 400],
        offset: 4_107_200,
    },
    LegacyTensorSpec {
        name: "213",
        dimensions: &[1, 1200, 400],
        offset: 6_027_200,
    },
    LegacyTensorSpec {
        name: "214",
        dimensions: &[1, 2400],
        offset: 7_947_200,
    },
    LegacyTensorSpec {
        name: "215",
        dimensions: &[400, 600],
        offset: 7_956_800,
    },
    LegacyTensorSpec {
        name: "216",
        dimensions: &[600, 600],
        offset: 8_916_800,
    },
    LegacyTensorSpec {
        name: "217",
        dimensions: &[600, 161],
        offset: 10_356_800,
    },
    LegacyTensorSpec {
        name: "fc_in.0.bias",
        dimensions: &[400],
        offset: 10_743_200,
    },
    LegacyTensorSpec {
        name: "fc_out.0.bias",
        dimensions: &[600],
        offset: 10_744_800,
    },
    LegacyTensorSpec {
        name: "fc_out.2.bias",
        dimensions: &[600],
        offset: 10_747_200,
    },
    LegacyTensorSpec {
        name: "fc_out.4.bias",
        dimensions: &[161],
        offset: 10_749_600,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactLayout {
    Canonical,
    LegacyPublic,
}

/// Official floor inside `log10(max(power, floor))`.
const POWER_FLOOR: f32 = 1e-12;
/// Official `mingain = 10 ** (-80 / 20)` floor applied after the ONNX
/// sigmoid (`enhance_onnx.py`, pinned Microsoft DNS-Challenge revision).
const MIN_GAIN: f32 = 1e-4;

// -------------------------------------------------------------------------
// Config
// -------------------------------------------------------------------------

/// NSNet2 runtime config (transcribed verbatim from `vokra.nsnet2.*` at
/// load time). Every field is validated at [`Self::validate`]; a
/// `0`-sentinel on any of them is a load-time refusal (FR-EX-08).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nsnet2Config {
    /// STFT bin count (upstream = 161 = `n_fft/2 + 1`).
    pub n_bins: usize,
    /// GRU / fc_in hidden width (upstream = 400).
    pub hidden_dim: usize,
    /// `fc_1` output width (upstream = 600).
    pub fc1_dim: usize,
    /// `fc_2` output width (upstream = 600).
    pub fc2_dim: usize,
    /// STFT FFT length (upstream = 320).
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
            n_bins: 161,
            hidden_dim: 400,
            fc1_dim: 600,
            fc2_dim: 600,
            n_fft: 320,
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
/// GRU rows are stored in the **`[R; Z; H]`** layout (permuted from ONNX
/// `[Z; R; H]` by the loader). Input and recurrent bias halves remain
/// separate: ONNX `linear_before_reset=1` gates the recurrent candidate bias
/// by `r`, so fusing that half would change the released graph.
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
    /// `gru_1.B` input bias half, permuted to `[R;Z;H]`.
    pub gru_1_bias_ih: Vec<f32>,
    /// `gru_1.B` recurrent bias half, permuted to `[R;Z;H]`.
    pub gru_1_bias_hh: Vec<f32>,
    /// `gru_2.W`, same layout as `gru_1_w_ih`.
    pub gru_2_w_ih: Vec<f32>,
    /// `gru_2.R`, same layout as `gru_1_w_hh`.
    pub gru_2_w_hh: Vec<f32>,
    /// `gru_2.B` input bias half, same layout as `gru_1_bias_ih`.
    pub gru_2_bias_ih: Vec<f32>,
    /// `gru_2.B` recurrent bias half, same layout as `gru_1_bias_hh`.
    pub gru_2_bias_hh: Vec<f32>,
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
    backend: BackendKind,
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

        let layout = resolve_artifact_layout(gguf)?;
        let cfg = match layout {
            ArtifactLayout::Canonical => Nsnet2Config::from_gguf(gguf)?,
            ArtifactLayout::LegacyPublic => Nsnet2Config::upstream_default(),
        };
        let weights = match layout {
            ArtifactLayout::Canonical => load_canonical_weights(gguf, &cfg)?,
            ArtifactLayout::LegacyPublic => load_legacy_public_weights(gguf)?,
        };
        Ok(Self {
            cfg,
            weights: Arc::new(weights),
            backend: BackendKind::Cpu,
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

    /// Selects the backend used by every learned projection and mask apply.
    /// Unsupported or unavailable backends fail when the stream first runs;
    /// they are never replaced by CPU execution.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Convenience one-shot: denoises `pcm` end-to-end starting from a
    /// fresh zero state and returns the enhanced PCM. Equivalent to
    /// opening a stream, pushing the whole buffer, and finalising.
    pub fn denoise_pcm(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mut stream =
            Nsnet2Stream::new(self.cfg.clone(), Arc::clone(&self.weights), self.backend)?;
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
        let stream = Nsnet2Stream::new(self.cfg.clone(), Arc::clone(&self.weights), self.backend)?;
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
    backend: BackendKind,
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
    /// Set after the upstream-compatible right-padded tail is flushed.
    input_finished: bool,
}

impl Nsnet2Stream {
    fn new(cfg: Nsnet2Config, weights: Arc<Nsnet2Weights>, backend: BackendKind) -> Result<Self> {
        let stft_attrs = analysis_stft_attrs(&cfg);
        let istft_attrs = synthesis_istft_attrs(&cfg);
        let istft = IstftStreamingState::new(&istft_attrs)?;
        let h1 = vec![0.0f32; cfg.hidden_dim];
        let h2 = vec![0.0f32; cfg.hidden_dim];
        Ok(Self {
            cfg,
            weights,
            backend,
            h1,
            h2,
            pending_pcm: Vec::new(),
            istft,
            stft_attrs,
            istft_attrs,
            frames_seen: 0,
            input_finished: false,
        })
    }

    /// One-shot push: runs analysis + core forward + synthesis on
    /// `pcm`, buffering the rolling PCM tail across calls. This is the
    /// inner entry-point `denoise_pcm` and the [`DenoiseStreamHandle`]
    /// impl share.
    fn push_pcm_internal(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if self.input_finished {
            return Err(VokraError::InvalidArgument(
                "nsnet2: push after finalize; reset the stream first".to_owned(),
            ));
        }
        // `MetalContext` is thread-affine, so streams retain only the
        // sendable BackendKind and construct the dispatcher on the calling
        // thread for each push. The coverage gate runs before any frame can
        // execute, preventing a mixed Metal/CPU neural path.
        let compute = Compute::for_backend(self.backend, NSNET2_HOT_OPS)?;
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

            // Official featurelib.py: log10(max(|X|^2, 1e-12)).
            let mut feature = vec![0.0f32; spec.bins];
            for k in 0..spec.bins {
                let p = re_row[k] * re_row[k] + im_row[k] * im_row[k];
                feature[k] = p.max(POWER_FLOOR).log10();
            }

            // fc_in -> ReLU
            let x = linear_relu_dispatch(
                &compute,
                self.backend,
                &feature,
                &self.weights.fc_in_weight,
                &self.weights.fc_in_bias,
                self.cfg.hidden_dim,
                self.cfg.n_bins,
            )?;

            // gru_1 (stateful h1)
            gru_forward_dispatch(
                &compute,
                self.backend,
                &x,
                &mut self.h1,
                GruWeightsRef {
                    w_ih: &self.weights.gru_1_w_ih,
                    w_hh: &self.weights.gru_1_w_hh,
                    bias_ih: &self.weights.gru_1_bias_ih,
                    bias_hh: &self.weights.gru_1_bias_hh,
                },
            )?;

            // gru_2 (stateful h2, input = h1)
            gru_forward_dispatch(
                &compute,
                self.backend,
                &self.h1,
                &mut self.h2,
                GruWeightsRef {
                    w_ih: &self.weights.gru_2_w_ih,
                    w_hh: &self.weights.gru_2_w_hh,
                    bias_ih: &self.weights.gru_2_bias_ih,
                    bias_hh: &self.weights.gru_2_bias_hh,
                },
            )?;

            // fc_1 + ReLU -> fc_2 + ReLU -> mask + sigmoid
            let a = linear_relu_dispatch(
                &compute,
                self.backend,
                &self.h2,
                &self.weights.fc_1_weight,
                &self.weights.fc_1_bias,
                self.cfg.fc1_dim,
                self.cfg.hidden_dim,
            )?;
            let b = linear_relu_dispatch(
                &compute,
                self.backend,
                &a,
                &self.weights.fc_2_weight,
                &self.weights.fc_2_bias,
                self.cfg.fc2_dim,
                self.cfg.fc1_dim,
            )?;
            let gain_pre = linear_dispatch(
                &compute,
                self.backend,
                &b,
                &self.weights.mask_weight,
                &self.weights.mask_bias,
                self.cfg.n_bins,
                self.cfg.fc2_dim,
            )?;

            // Sigmoid — bounded ∈ [0, 1]. Apply to complex STFT
            // (phase preserved verbatim per arXiv:2005.07551 §3.2).
            let gain = gain_pre
                .iter()
                .map(|value| sigmoid_stable(*value).clamp(MIN_GAIN, 1.0))
                .collect::<Vec<_>>();
            compute.denoise_apply_mask_f32(
                re_row,
                im_row,
                &gain,
                1,
                spec.bins,
                &mut y_re[base..base + spec.bins],
                &mut y_im[base..base + spec.bins],
            )?;
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
        if self.input_finished {
            return Ok(Vec::new());
        }

        // Microsoft's featurelib pads the input to a hop boundary and emits
        // ceil(input_len / hop) no-delay frames. After normal streaming pushes
        // `pending_pcm` contains the overlap plus a possible partial hop. Pad
        // it so the ordinary raw-frame STFT emits exactly the remaining
        // ceil(pending/hop) frames, including the final half-zero window.
        let remaining_frames = self.pending_pcm.len().div_ceil(self.cfg.hop);
        let mut output = Vec::new();
        if remaining_frames > 0 {
            let target_len = self.cfg.n_fft + (remaining_frames - 1) * self.cfg.hop;
            self.pending_pcm.resize(target_len, 0.0);
            output.extend(self.push_pcm_internal(&[])?);
        }
        self.pending_pcm.clear();
        self.input_finished = true;
        output.extend(self.istft.finish());
        Ok(output)
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
            .field("backend", &self.backend)
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

    fn finalize(&mut self) -> Result<Vec<f32>> {
        Nsnet2Stream::finalize(self)
    }

    fn reset(&mut self) {
        self.h1.fill(0.0);
        self.h2.fill(0.0);
        self.pending_pcm.clear();
        self.istft.reset();
        self.frames_seen = 0;
        self.input_finished = false;
    }
}

// -------------------------------------------------------------------------
// Analysis / synthesis attributes
// -------------------------------------------------------------------------

/// Analysis STFT attributes matching Microsoft's pinned `featurelib.py`:
/// `n_fft=320`, `hop=160`, symmetric square-root Hann, backward-normalised.
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
    a.window = Window::SqrtHann;
    a.window_symmetry = WindowSymmetry::Symmetric;
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
    i.window = Window::SqrtHann;
    i.window_symmetry = WindowSymmetry::Symmetric;
    i.center = false;
    // Upstream `featurelib.istft` performs raw overlap-add without WSS
    // normalization. With sqrt-Hann on both sides the product is Hann.
    i.normalize_window = false;
    IstftStreamingAttrs::from_istft(i)
}

// -------------------------------------------------------------------------
// Loader helper
// -------------------------------------------------------------------------

fn resolve_artifact_layout(gguf: &GgufFile) -> Result<ArtifactLayout> {
    let hparam_count = HPARAM_KEYS
        .iter()
        .filter(|key| gguf.get(key).is_some())
        .count();
    let canonical_count = CANONICAL_TENSOR_NAMES
        .iter()
        .filter(|name| gguf.tensor_info(name).is_some())
        .count();
    let legacy_count = LEGACY_PUBLIC_TENSORS
        .iter()
        .filter(|spec| gguf.tensor_info(spec.name).is_some())
        .count();

    if hparam_count != 0 && hparam_count != HPARAM_KEYS.len() {
        let missing = HPARAM_KEYS
            .iter()
            .filter(|key| gguf.get(key).is_none())
            .copied()
            .collect::<Vec<_>>();
        return Err(VokraError::ModelLoad(format!(
            "nsnet2: partial `vokra.nsnet2.*` metadata ({hparam_count}/{} keys); \
             missing {missing:?}. Refusing topology repair",
            HPARAM_KEYS.len(),
        )));
    }
    if canonical_count > 0 && legacy_count > 0 {
        return Err(VokraError::ModelLoad(format!(
            "nsnet2: mixed canonical and historical public tensor schemas \
             ({canonical_count}/{} canonical, {legacy_count}/{} legacy); refusing ambiguous precedence",
            CANONICAL_TENSOR_NAMES.len(),
            LEGACY_PUBLIC_TENSORS.len(),
        )));
    }

    if hparam_count == HPARAM_KEYS.len() {
        if legacy_count > 0 {
            return Err(VokraError::ModelLoad(
                "nsnet2: historical numeric initializer names cannot be combined with the \
                 canonical `vokra.nsnet2.*` metadata group"
                    .to_owned(),
            ));
        }
        return Ok(ArtifactLayout::Canonical);
    }

    if legacy_count == LEGACY_PUBLIC_TENSORS.len() && canonical_count == 0 {
        validate_legacy_public_contract(gguf)?;
        return Ok(ArtifactLayout::LegacyPublic);
    }
    if legacy_count > 0 {
        return Err(VokraError::ModelLoad(format!(
            "nsnet2: incomplete historical public tensor schema ({legacy_count}/{} tensors); \
             only the complete header contract audited from Hub revision \
             {LEGACY_PUBLIC_REVISION} (source SHA-256 {LEGACY_PUBLIC_SHA256}) may repair \
             missing topology metadata",
            LEGACY_PUBLIC_TENSORS.len(),
        )));
    }
    if canonical_count > 0 {
        return Err(VokraError::ModelLoad(format!(
            "nsnet2: canonical tensor schema is present but all {} required \
             `vokra.nsnet2.*` metadata keys are absent; refusing implicit defaults",
            HPARAM_KEYS.len(),
        )));
    }

    Err(VokraError::ModelLoad(format!(
        "nsnet2: GGUF contains neither the canonical semantic tensor schema nor the exact \
         historical public schema from revision {LEGACY_PUBLIC_REVISION}"
    )))
}

fn validate_legacy_public_contract(gguf: &GgufFile) -> Result<()> {
    // These provenance values are the exact observed identity of the old Hub
    // object, not an endorsement of its MIT/permissive classification. The
    // fixed Microsoft source revision separates MIT code from CC-BY-4.0
    // released content; the live-repository audit therefore remains partial
    // until an authorized gated replacement corrects the model provenance and
    // attribution. Runtime topology repair must still pin the mis-stamped
    // historical header exactly so unrelated files cannot enter this branch.
    if gguf.metadata().len() != 10 {
        return Err(legacy_public_error(format!(
            "metadata count is {}, expected exactly 10",
            gguf.metadata().len(),
        )));
    }

    for (key, expected) in [
        (chunks::KEY_MODEL_ARCH, ARCH),
        (chunks::KEY_MODEL_NAME, DEFAULT_NAME),
        (KEY_MODEL_CATEGORY, CATEGORY),
        (chunks::KEY_PROVENANCE_LICENSE, "mit"),
        (chunks::KEY_PROVENANCE_MODEL_ID, DEFAULT_NAME),
        (chunks::KEY_PROVENANCE_SOURCE, LEGACY_PUBLIC_SOURCE),
        (chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive"),
        ("vokra.provenance.upstream_url", LEGACY_PUBLIC_UPSTREAM_URL),
    ] {
        let actual = gguf.get(key).and_then(|value| value.as_str());
        if actual != Some(expected) {
            return Err(legacy_public_error(format!(
                "metadata `{key}` is {actual:?}, expected {expected:?}",
            )));
        }
    }
    if gguf
        .get(chunks::KEY_SCHEMA_VERSION)
        .and_then(|value| value.as_u64())
        != Some(1)
    {
        return Err(legacy_public_error(
            "`vokra.schema.version` must be unsigned integer 1".to_owned(),
        ));
    }
    let producer = gguf
        .get(chunks::KEY_SCHEMA_PRODUCER)
        .and_then(|value| value.as_str());
    if producer != Some(LEGACY_PUBLIC_SCHEMA_PRODUCER) && producer != Some(CURRENT_SCHEMA_PRODUCER)
    {
        return Err(legacy_public_error(format!(
            "`vokra.schema.producer` is {producer:?}, expected the audited original \
             {LEGACY_PUBLIC_SCHEMA_PRODUCER:?} or current first-party writer {CURRENT_SCHEMA_PRODUCER:?}",
        )));
    }

    if gguf.tensors().len() != LEGACY_PUBLIC_TENSORS.len() {
        return Err(legacy_public_error(format!(
            "tensor count is {}, expected exactly {}",
            gguf.tensors().len(),
            LEGACY_PUBLIC_TENSORS.len(),
        )));
    }
    for (index, (actual, expected)) in gguf.tensors().iter().zip(LEGACY_PUBLIC_TENSORS).enumerate()
    {
        if actual.name != expected.name
            || actual.dtype != GgmlType::F32
            || actual.dimensions.as_slice() != expected.dimensions
            || actual.offset != expected.offset
        {
            return Err(legacy_public_error(format!(
                "tensor directory row {index} is name={:?} dtype={:?} dims={:?} offset={}, \
                 expected name={:?} dtype=F32 dims={:?} offset={}",
                actual.name,
                actual.dtype,
                actual.dimensions,
                actual.offset,
                expected.name,
                expected.dimensions,
                expected.offset,
            )));
        }
    }
    Ok(())
}

fn legacy_public_error(detail: String) -> VokraError {
    VokraError::ModelLoad(format!(
        "nsnet2: historical public GGUF contract mismatch: {detail}. Only the \
         header contract audited from vokra/nsnet2 revision {LEGACY_PUBLIC_REVISION} \
         (source SHA-256 {LEGACY_PUBLIC_SHA256}) is eligible for metadata repair"
    ))
}

fn load_canonical_weights(gguf: &GgufFile, cfg: &Nsnet2Config) -> Result<Nsnet2Weights> {
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

    // ONNX ships `[Z; R; H]`; permute to `[R; Z; H]`. Keep Wb and Rb
    // separate because `linear_before_reset=1` gates the recurrent candidate
    // contribution, including Rb_h.
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

    let (gru_1_w_ih, gru_1_w_hh, gru_1_bias_ih, gru_1_bias_hh) =
        permute_onnx_gru(&gru_1_w_raw, &gru_1_r_raw, &gru_1_b_raw, h, h);
    let (gru_2_w_ih, gru_2_w_hh, gru_2_bias_ih, gru_2_bias_hh) =
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
        gru_1_bias_ih,
        gru_1_bias_hh,
        gru_2_w_ih,
        gru_2_w_hh,
        gru_2_bias_ih,
        gru_2_bias_hh,
        fc_1_weight,
        fc_1_bias,
        fc_2_weight,
        fc_2_bias,
        mask_weight,
        mask_bias,
    })
}

/// Binds the exact 2026-08-03 public GGUF layout into the same canonical
/// in-memory weights used by [`load_canonical_weights`].  The public file kept
/// ONNX initializer names and axes: MatMul matrices are `[in, out]`, while GRU
/// tensors retain their singleton direction axis.  No arithmetic changes here;
/// matrices are transposed once and singleton axes disappear by reading the
/// identical flat payload under the already-validated shape contract.
fn load_legacy_public_weights(gguf: &GgufFile) -> Result<Nsnet2Weights> {
    let load = |name: &str, expected: usize| -> Result<Vec<f32>> {
        let values = gguf.tensor_f32(name).map_err(|error| {
            VokraError::ModelLoad(format!(
                "nsnet2: historical public tensor `{name}` load failed: {error}"
            ))
        })?;
        if values.len() != expected {
            return Err(VokraError::ModelLoad(format!(
                "nsnet2: historical public tensor `{name}` has {} elements, expected {expected}",
                values.len(),
            )));
        }
        Ok(values)
    };

    let fc_in_weight = transpose_row_major(&load("172", 161 * 400)?, 161, 400);
    let fc_in_bias = load("fc_in.0.bias", 400)?;

    let gru_1_w_raw = load("192", 3 * 400 * 400)?;
    let gru_1_r_raw = load("193", 3 * 400 * 400)?;
    let gru_1_b_raw = load("194", 6 * 400)?;
    let gru_2_w_raw = load("212", 3 * 400 * 400)?;
    let gru_2_r_raw = load("213", 3 * 400 * 400)?;
    let gru_2_b_raw = load("214", 6 * 400)?;
    let (gru_1_w_ih, gru_1_w_hh, gru_1_bias_ih, gru_1_bias_hh) =
        permute_onnx_gru(&gru_1_w_raw, &gru_1_r_raw, &gru_1_b_raw, 400, 400);
    let (gru_2_w_ih, gru_2_w_hh, gru_2_bias_ih, gru_2_bias_hh) =
        permute_onnx_gru(&gru_2_w_raw, &gru_2_r_raw, &gru_2_b_raw, 400, 400);

    let fc_1_weight = transpose_row_major(&load("215", 400 * 600)?, 400, 600);
    let fc_1_bias = load("fc_out.0.bias", 600)?;
    let fc_2_weight = transpose_row_major(&load("216", 600 * 600)?, 600, 600);
    let fc_2_bias = load("fc_out.2.bias", 600)?;
    let mask_weight = transpose_row_major(&load("217", 600 * 161)?, 600, 161);
    let mask_bias = load("fc_out.4.bias", 161)?;

    Ok(Nsnet2Weights {
        fc_in_weight,
        fc_in_bias,
        gru_1_w_ih,
        gru_1_w_hh,
        gru_1_bias_ih,
        gru_1_bias_hh,
        gru_2_w_ih,
        gru_2_w_hh,
        gru_2_bias_ih,
        gru_2_bias_hh,
        fc_1_weight,
        fc_1_bias,
        fc_2_weight,
        fc_2_bias,
        mask_weight,
        mask_bias,
    })
}

fn transpose_row_major(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), rows * cols);
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = values[row * cols + col];
        }
    }
    transposed
}

/// Permutes an ONNX GRU triplet (`W [Z;R;H]`, `R [Z;R;H]`, `B [Wb_ZRH; Rb_ZRH]`)
/// into the `[R; Z; H]` layout with separate input/recurrent biases.
///
/// The row block size in `W` is `[hidden, in_dim]`; for `R` it is
/// `[hidden, hidden]`. ONNX packs the bias as two halves of size
/// `3 * hidden` (`Wb` = input, `Rb` = recurrent). They must not be fused:
/// ONNX `linear_before_reset=1` computes `r * (R_h h + Rb_h)`.
///
/// Returns `(w_ih, w_hh, bias_ih, bias_hh)` in `[R; Z; H]` order.
fn permute_onnx_gru(
    w: &[f32],
    r: &[f32],
    b: &[f32],
    hidden: usize,
    in_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_ih = vec![0.0f32; 3 * hidden * in_dim];
    let mut w_hh = vec![0.0f32; 3 * hidden * hidden];
    let mut bias_ih = vec![0.0f32; 3 * hidden];
    let mut bias_hh = vec![0.0f32; 3 * hidden];

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

    // Bias: `b` is `[Wb_z | Wb_r | Wb_h | Rb_z | Rb_r | Rb_h]`.
    let wb_z = &b[0..hidden];
    let wb_r = &b[hidden..2 * hidden];
    let wb_h = &b[2 * hidden..3 * hidden];
    let rb_z = &b[3 * hidden..4 * hidden];
    let rb_r = &b[4 * hidden..5 * hidden];
    let rb_h = &b[5 * hidden..6 * hidden];
    for i in 0..hidden {
        // R (index 0)
        bias_ih[i] = wb_r[i];
        bias_hh[i] = rb_r[i];
        // Z (index 1)
        bias_ih[hidden + i] = wb_z[i];
        bias_hh[hidden + i] = rb_z[i];
        // N (index 2)
        bias_ih[2 * hidden + i] = wb_h[i];
        bias_hh[2 * hidden + i] = rb_h[i];
    }
    (w_ih, w_hh, bias_ih, bias_hh)
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

/// Backend-dispatched sibling of [`linear_relu`]. CPU deliberately retains
/// the established scalar oracle; GPU backends execute the complete learned
/// projection through GEMV and keep only the element-wise ReLU on the host.
fn linear_relu_dispatch(
    compute: &Compute,
    backend: BackendKind,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if backend == BackendKind::Cpu {
        return linear_relu(x, w, b, out_dim, in_dim);
    }
    let mut y = linear_dispatch(compute, backend, x, w, b, out_dim, in_dim)?;
    for value in &mut y {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    Ok(y)
}

/// Routes a learned dense projection through the selected backend while
/// preserving the original scalar implementation as the CPU oracle.
fn linear_dispatch(
    compute: &Compute,
    backend: BackendKind,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if backend == BackendKind::Cpu {
        return linear(x, w, b, out_dim, in_dim);
    }
    let mut y = vec![0.0; out_dim];
    compute.gemv_f32(out_dim, in_dim, w, x, Some(b), &mut y)?;
    Ok(y)
}

/// Borrowed learned parameters for one NSNet2 GRU layer.
struct GruWeightsRef<'a> {
    w_ih: &'a [f32],
    w_hh: &'a [f32],
    bias_ih: &'a [f32],
    bias_hh: &'a [f32],
}

/// NSNet2's `[R; Z; N]`, `linear_before_reset=1` GRU cell with both learned
/// projections dispatched as one GEMV each. The recurrent candidate bias is
/// kept inside the reset gate exactly as ONNX specifies.
fn gru_forward_dispatch(
    compute: &Compute,
    backend: BackendKind,
    input: &[f32],
    state: &mut [f32],
    weights: GruWeightsRef<'_>,
) -> Result<()> {
    if backend == BackendKind::Cpu {
        return onnx_gru_forward_cpu(
            input,
            state,
            weights.w_ih,
            weights.w_hh,
            weights.bias_ih,
            weights.bias_hh,
        );
    }

    let hidden = state.len();
    let in_dim = input.len();
    if hidden == 0 {
        return Err(VokraError::InvalidArgument(
            "nsnet2 GRU state must be non-empty".to_owned(),
        ));
    }
    let rows = 3 * hidden;
    if weights.w_ih.len() != rows * in_dim
        || weights.w_hh.len() != rows * hidden
        || weights.bias_ih.len() != rows
        || weights.bias_hh.len() != rows
    {
        return Err(VokraError::InvalidArgument(format!(
            "nsnet2 GRU shape mismatch: input={} state={} w_ih={} w_hh={} bias_ih={} bias_hh={}",
            in_dim,
            hidden,
            weights.w_ih.len(),
            weights.w_hh.len(),
            weights.bias_ih.len(),
            weights.bias_hh.len(),
        )));
    }

    let mut input_gates = vec![0.0; rows];
    let mut recurrent_gates = vec![0.0; rows];
    compute.gemv_f32(
        rows,
        in_dim,
        weights.w_ih,
        input,
        Some(weights.bias_ih),
        &mut input_gates,
    )?;
    compute.gemv_f32(
        rows,
        hidden,
        weights.w_hh,
        state,
        Some(weights.bias_hh),
        &mut recurrent_gates,
    )?;

    for index in 0..hidden {
        let reset = sigmoid_stable(input_gates[index] + recurrent_gates[index]);
        let update = sigmoid_stable(input_gates[hidden + index] + recurrent_gates[hidden + index]);
        let candidate =
            (input_gates[2 * hidden + index] + reset * recurrent_gates[2 * hidden + index]).tanh();
        state[index] = (1.0 - update) * candidate + update * state[index];
    }
    Ok(())
}

fn onnx_gru_forward_cpu(
    input: &[f32],
    state: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    bias_ih: &[f32],
    bias_hh: &[f32],
) -> Result<()> {
    let hidden = state.len();
    let in_dim = input.len();
    let rows = 3 * hidden;
    if hidden == 0
        || w_ih.len() != rows * in_dim
        || w_hh.len() != rows * hidden
        || bias_ih.len() != rows
        || bias_hh.len() != rows
    {
        return Err(VokraError::InvalidArgument(
            "nsnet2 CPU GRU shape mismatch".to_owned(),
        ));
    }

    let project = |weights: &[f32], vector: &[f32], bias: &[f32], width: usize| {
        let mut output = bias.to_vec();
        for row in 0..rows {
            let weights = &weights[row * width..(row + 1) * width];
            for (weight, value) in weights.iter().zip(vector) {
                output[row] += weight * value;
            }
        }
        output
    };
    let input_gates = project(w_ih, input, bias_ih, in_dim);
    let recurrent_gates = project(w_hh, state, bias_hh, hidden);
    for index in 0..hidden {
        let reset = sigmoid_stable(input_gates[index] + recurrent_gates[index]);
        let update = sigmoid_stable(input_gates[hidden + index] + recurrent_gates[hidden + index]);
        let candidate =
            (input_gates[2 * hidden + index] + reset * recurrent_gates[2 * hidden + index]).tanh();
        state[index] = (1.0 - update) * candidate + update * state[index];
    }
    Ok(())
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
