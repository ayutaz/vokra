//! CREPE — Convolutional Representation for Pitch Estimation (Kim et al. 2018).
//!
//! # Primary source
//!
//! - Upstream reference: <https://github.com/marl/crepe>
//! - Paper: Kim, J.W., Salamon, J., Li, P., Bello, J.P., "CREPE: A Convolutional
//!   Representation for Pitch Estimation," ICASSP 2018 (arXiv:1802.06182).
//! - License: **MIT** (`github.com/marl/crepe/main/LICENSE.txt`, "MIT License /
//!   Copyright (c) 2018 Jong Wook Kim et al.", fetched 2026-07-30 —
//!   CLAUDE.md「ハルシネーション厳禁」).
//!
//! CREPE is a monophonic F0 (fundamental-frequency) extractor: a 6-layer
//! 1-D convolutional CNN over raw 16-kHz audio frames (1024 samples per
//! frame) that classifies pitch into a 360-bin log-frequency grid. The
//! argmax bin is decoded into Hz via a local weighted centroid ("local
//! average cents"). The per-frame max of the classifier's sigmoid output
//! doubles as a voicing confidence.
//!
//! # This module — real 6-block CNN forward
//!
//! - [`CrepeConfig`] carries the [`CapacityFactor`] discriminator (tiny / small
//!   / medium / large / full — the upstream size knob), plus the shared
//!   `hop` / `fmin` / `fmax` metadata every sibling F0 extractor
//!   ([`super::rmvpe`], [`super::fcpe`]) also carries.
//! - `CrepeWeights::from_gguf` binds all 6 conv blocks + the final Dense
//!   classifier from tensors emitted by `crates/vokra-convert/src/models/crepe.rs`,
//!   folding the Keras `BatchNormalization` to a per-channel affine at load
//!   (`scale = γ/√(σ² + ε)`, `shift = β − μ·scale`) — the same offline fold
//!   posture as `crate::speaker::weights` and the DAC / UTMOS out-projections.
//! - [`CREPE::extract`] / [`CREPE::extract_real`] (the former delegates to
//!   the latter) run the real forward: 512-sample zero-pad ("center" frames),
//!   per-frame mean/std normalization, 6 × (Conv2D → BN → ReLU → MaxPool),
//!   Permute + Flatten, Dense(360) + sigmoid, then local-average cents decode
//!   → Hz. Both are fallible, and the two failure modes are distinct, named
//!   errors: an unbound weight set is one, a non-[`NATIVE_SAMPLE_RATE`] input
//!   is the other. Neither is ever answered with a zero-filled track
//!   (FR-EX-08). This mirrors [`super::rmvpe`] exactly, so the F0 family
//!   reads the same way at every call site.
//! - [`CREPE::frame_times`] returns the per-hop timestamps **only**, as bare
//!   `f32` seconds, for callers that just need to size or align a buffer. It
//!   runs no model, measures no pitch, and cannot be mistaken for a track.
//!
//! # Frame-count contract
//!
//! `extract(&pcm, sr)?.len()` equals `pcm.len() / hop` (hop=160 by default =
//! 10 ms @ 16 kHz), and `frame_times(pcm.len(), sr).len()` equals the same —
//! the contract every downstream consumer already sizes its buffers around
//! holds across both entry points.
//!
//! # No ONNX (permanent)
//!
//! The upstream `crepe` package ships a Keras / TensorFlow `.h5` model;
//! the model definition is re-implemented natively here (whisper.cpp 型,
//! CLAUDE.md 設計判断 4). The offline `tools/parity/keras_h5_to_safetensors.py`
//! script bridges the Keras checkpoint into safetensors + a JSON config
//! side-car the Vokra converter consumes — NOT the model graph itself.
//! This module never touches TensorFlow / ONNX.

use std::path::Path;

use vokra_core::gguf::{GgmlType, GgufFile};

use super::{F0Frame, LoadError};
use crate::compute::Compute;

/// Default hop between output frames, in samples. Matches the upstream
/// `crepe.predict(..., step_size=10)` default at the canonical 16 kHz input
/// rate (10 ms → 160 samples).
pub const DEFAULT_HOP: u32 = 160;

/// The only PCM sample rate the CREPE CNN is defined at.
///
/// Upstream `crepe/core.py` frames 1024 samples of **16 kHz** audio per
/// estimate and maps the classifier's 360 bins onto a cent grid anchored to
/// that rate, so the network is not rate-agnostic. `crepe.predict` hides this
/// by resampling internally; Vokra refuses instead (FR-EX-08 — never a silent
/// resample), which is why [`CREPE::extract_real`] takes the caller's rate and
/// compares it against this constant.
pub const NATIVE_SAMPLE_RATE: u32 = 16_000;

/// Default lower bound of the pitch search grid (Hz). Matches the upstream
/// CREPE classifier's low log-frequency edge.
pub const DEFAULT_FMIN: f32 = 50.0;

/// Default upper bound of the pitch search grid (Hz). Matches the upstream
/// CREPE classifier's high log-frequency edge.
pub const DEFAULT_FMAX: f32 = 1100.0;

/// Frame length in samples (upstream fixed constant, `crepe/core.py`).
const FRAME_LEN: usize = 1024;

/// Number of pitch classes at the classifier output (upstream fixed
/// constant, `crepe/core.py` `Dense(360, activation='sigmoid')`).
const N_BINS: usize = 360;

/// Filter multipliers per block (upstream `[32, 4, 4, 4, 8, 16]`).
const FILTER_MULT: [usize; 6] = [32, 4, 4, 4, 8, 16];
/// Kernel size along the frequency axis per block (upstream `[512, 64, …]`).
const KERNEL_WIDTH: [usize; 6] = [512, 64, 64, 64, 64, 64];
/// Stride along the frequency axis per block (upstream first block =4).
const STRIDE: [usize; 6] = [4, 1, 1, 1, 1, 1];

/// Local-averaging centroid half-window (upstream `to_local_average_cents`
/// takes bins `[center-4, center+5)`).
const CENTROID_HALF_WIN: usize = 4;

/// Cent-mapping constants (upstream `to_local_average_cents`, verbatim):
/// bin i → `1997.3794084376191 + (7180 * i) / (N_BINS - 1)` cents.
/// `10 * 2 ** (cents / 1200)` = 10 Hz reference (upstream `predict`).
const CENTS_OFFSET: f32 = 1_997.379_4;
/// End-to-end span of the cent mapping (upstream `np.linspace(0, 7180, 360)`).
const CENTS_SPAN: f32 = 7180.0;
/// Reference Hz for cents→Hz conversion (upstream `10 * 2 ** (cents / 1200)`).
const HZ_REF: f32 = 10.0;

/// Minimum-std guard for the per-frame normalization (upstream
/// `np.clip(std, 1e-8, None)`).
const STD_FLOOR: f32 = 1e-8;

/// Keras `BatchNormalization` default epsilon (upstream constructor;
/// `momentum` / `moving_*` naming remain but epsilon is the only piece
/// the folded affine needs).
const BN_EPS: f32 = 1e-3;

/// The upstream capacity knob — determines the filter-count multiplier
/// (upstream `crepe/core.py::build_and_load_model`).
///
/// Upstream ships 5 sizes, each named after its multiplier: `tiny`=4,
/// `small`=8, `medium`=16, `large`=24, `full`=32. The full-size CNN is
/// the size referenced in the ICASSP paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityFactor {
    /// 4× — `crepe/model-tiny.h5` (smallest & fastest).
    Tiny,
    /// 8× — `crepe/model-small.h5`.
    Small,
    /// 16× — `crepe/model-medium.h5`.
    Medium,
    /// 24× — `crepe/model-large.h5`.
    Large,
    /// 32× — `crepe/model-full.h5` (paper-size, highest accuracy).
    Full,
}

impl CapacityFactor {
    /// Integer multiplier applied to `FILTER_MULT` per block.
    pub const fn multiplier(self) -> usize {
        match self {
            Self::Tiny => 4,
            Self::Small => 8,
            Self::Medium => 16,
            Self::Large => 24,
            Self::Full => 32,
        }
    }

    /// Parses the string tag written to `vokra.f0.crepe.capacity`.
    fn from_tag(tag: &str) -> Option<Self> {
        match tag.trim().to_ascii_lowercase().as_str() {
            "tiny" => Some(Self::Tiny),
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    /// Canonical string tag (round-trip with `Self::from_tag`).
    pub const fn as_tag(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Full => "full",
        }
    }
}

/// GGUF metadata key: capacity discriminator (`vokra.f0.crepe.capacity`, string).
pub const GGUF_KEY_CAPACITY: &str = "vokra.f0.crepe.capacity";
/// GGUF metadata key: analysis hop in samples (`vokra.f0.crepe.hop`, u32).
pub const GGUF_KEY_HOP: &str = "vokra.f0.crepe.hop";
/// GGUF metadata key: minimum tracked F0 in Hz (`vokra.f0.crepe.fmin`, f32).
pub const GGUF_KEY_FMIN: &str = "vokra.f0.crepe.fmin";
/// GGUF metadata key: maximum tracked F0 in Hz (`vokra.f0.crepe.fmax`, f32).
pub const GGUF_KEY_FMAX: &str = "vokra.f0.crepe.fmax";

/// Configuration a [`CREPE`] loads from a Vokra GGUF file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrepeConfig {
    /// Capacity multiplier (tiny / small / medium / large / full).
    pub capacity: CapacityFactor,
    /// Analysis hop in samples.
    pub hop: u32,
    /// Minimum tracked F0 in Hz (search-grid floor; informational).
    pub fmin: f32,
    /// Maximum tracked F0 in Hz (search-grid ceiling; informational).
    pub fmax: f32,
}

impl CrepeConfig {
    /// The upstream default at 16 kHz (`hop=160`, `fmin=50.0`, `fmax=1100.0`).
    pub const fn defaults(capacity: CapacityFactor) -> Self {
        Self {
            capacity,
            hop: DEFAULT_HOP,
            fmin: DEFAULT_FMIN,
            fmax: DEFAULT_FMAX,
        }
    }

    /// Per-block filter counts derived from the capacity multiplier.
    fn filters(&self) -> [usize; 6] {
        let m = self.capacity.multiplier();
        [
            FILTER_MULT[0] * m,
            FILTER_MULT[1] * m,
            FILTER_MULT[2] * m,
            FILTER_MULT[3] * m,
            FILTER_MULT[4] * m,
            FILTER_MULT[5] * m,
        ]
    }

    /// Length of the flattened feature vector fed into the final Dense.
    ///
    /// Upstream: after block 1 the freq axis is 1024/4 = 256 (`same` pad +
    /// stride 4); each of blocks 2–6 adds a MaxPool2D `(2, 1)`; the freq
    /// axis after 6 blocks is `256 / (2⁵) = 8` — but the final MaxPool
    /// halves it again to 4? Let's walk the reference (Keras `same`
    /// padding is `ceil(in/stride)`):
    ///
    /// ```text
    /// in freq: 1024
    /// block 1: stride=(4,1) → 256, MaxPool(2,1) → 128
    /// block 2: stride=(1,1) → 128, MaxPool(2,1) → 64
    /// block 3: stride=(1,1) → 64,  MaxPool(2,1) → 32
    /// block 4: stride=(1,1) → 32,  MaxPool(2,1) → 16
    /// block 5: stride=(1,1) → 16,  MaxPool(2,1) → 8
    /// block 6: stride=(1,1) → 8,   MaxPool(2,1) → 4
    /// ```
    ///
    /// So `flat_len = 4 * filters[5] = 4 * 16 * m` (tiny=256, full=2048).
    fn flat_len(&self) -> usize {
        4 * self.filters()[5]
    }
}

/// The full weight bundle for a single CREPE model, loaded from a Vokra
/// GGUF and BN-folded at load time.
#[derive(Debug)]
pub struct CrepeWeights {
    /// Six conv blocks in order (`conv1` .. `conv6`).
    conv: [ConvBn; 6],
    /// Final `Dense(360, sigmoid)` weight `[N_BINS, flat_len]` (row-major).
    classifier_w: Vec<f32>,
    /// Final Dense bias `[N_BINS]`.
    classifier_b: Vec<f32>,
}

/// One conv block: `[c_out, c_in, kh, kw=1]` weight + `[c_out]` bias +
/// folded BN affine `y = x·scale + shift`.
///
/// Keras `Conv2D(padding='same')` — the load path stores the pre-computed
/// pad size (`kh / 2`) so the forward does not recompute it.
#[derive(Debug)]
struct ConvBn {
    /// `[c_out, c_in, kh, 1]` flattened row-major (kw is always 1).
    w: Vec<f32>,
    /// `[c_out]` bias (Keras `Conv2D` default `use_bias=True`).
    b: Vec<f32>,
    /// Output channels.
    c_out: usize,
    /// Input channels.
    c_in: usize,
    /// Kernel height (along the freq axis).
    kh: usize,
    /// Stride along the freq axis.
    sh: usize,
    /// Left / top pad computed from the upstream `same` policy.
    pad: usize,
    /// Folded BN `scale = γ / √(σ² + ε)`.
    bn_scale: Vec<f32>,
    /// Folded BN `shift = β − μ·scale`.
    bn_shift: Vec<f32>,
}

/// The CREPE F0 extractor (Convolutional Representation for Pitch Estimation).
///
/// Acronym-cased per the F0 op family (FR-OP-83) — the load / extract surface
/// is the same across siblings (PyIN, FCPE, Harvest, RMVPE), so
/// `CREPE::extract` names the extractor rather than being noun-cased.
#[allow(clippy::upper_case_acronyms)]
pub struct CREPE {
    cfg: CrepeConfig,
    weights: Option<CrepeWeights>,
    /// Cent-per-bin mapping precomputed from `CENTS_OFFSET` + `CENTS_SPAN` (upstream
    /// `to_local_average_cents.cents_mapping = np.linspace(0, 7180, 360) + 1997.…`).
    cents_mapping: [f32; N_BINS],
}

impl CREPE {
    /// Loads CREPE from a GGUF file on disk.
    ///
    /// Reads four OPTIONAL metadata keys and falls back to the upstream
    /// defaults if a key is absent:
    ///
    /// - `vokra.f0.crepe.capacity` (string, default `"full"`)
    /// - `vokra.f0.crepe.hop`  (u32, default `160`  — 10 ms at 16 kHz)
    /// - `vokra.f0.crepe.fmin` (f32, default `50.0` Hz)
    /// - `vokra.f0.crepe.fmax` (f32, default `1100.0` Hz)
    ///
    /// Weight tensors, if present, are bound and BN-folded (see
    /// `CrepeWeights::from_gguf`); an artifact carrying only the four
    /// metadata keys but no weights still loads, with `weights = None`, so the
    /// metadata-only GGUFs the earlier skeleton wrote stay valid
    /// ([`has_real_weights`](Self::has_real_weights) then reports `false`).
    /// Such a handle cannot measure pitch:
    /// [`extract`](Self::extract) / [`extract_real`](Self::extract_real)
    /// refuse it loudly rather than answering with a zero-filled track, and
    /// [`frame_times`](Self::frame_times) is the entry point for a caller
    /// that only wants the per-hop timestamps.
    ///
    /// Returns [`LoadError`] if the path cannot be opened / parsed, or if a
    /// key is present with the wrong type / a weight tensor is malformed.
    pub fn from_gguf(path: &Path) -> Result<Self, LoadError> {
        let file = GgufFile::open(path).map_err(|e| LoadError::Gguf(format!("{e:?}")))?;

        let capacity = read_opt_string(&file, GGUF_KEY_CAPACITY)?
            .map(|s| CapacityFactor::from_tag(&s).ok_or_else(||
                LoadError::Gguf(format!("crepe metadata `{GGUF_KEY_CAPACITY}` = `{s}` is not one of tiny/small/medium/large/full"))
            ))
            .transpose()?
            .unwrap_or(CapacityFactor::Full);

        let hop = read_opt_u32(&file, GGUF_KEY_HOP)?.unwrap_or(DEFAULT_HOP);
        let fmin = read_opt_f32(&file, GGUF_KEY_FMIN)?.unwrap_or(DEFAULT_FMIN);
        let fmax = read_opt_f32(&file, GGUF_KEY_FMAX)?.unwrap_or(DEFAULT_FMAX);

        let cfg = CrepeConfig {
            capacity,
            hop,
            fmin,
            fmax,
        };

        // Optional weight bind: if `conv1.weight` is present the whole bundle
        // is expected; if it is absent we stay in the metadata-only skeleton
        // path so earlier-generation GGUFs keep loading.
        let weights = if file.tensor_info("conv1.weight").is_some() {
            Some(CrepeWeights::from_gguf(&file, cfg)?)
        } else {
            None
        };

        let mut cents_mapping = [0.0f32; N_BINS];
        for (i, out) in cents_mapping.iter_mut().enumerate() {
            *out = CENTS_OFFSET + CENTS_SPAN * (i as f32) / ((N_BINS - 1) as f32);
        }

        Ok(Self {
            cfg,
            weights,
            cents_mapping,
        })
    }

    /// Returns `true` when weight tensors were bound at load time — i.e.
    /// [`extract_real`](Self::extract_real) can actually run.
    ///
    /// A metadata-only GGUF (the pre-weights skeleton artifacts) binds with
    /// `false`. Callers that want to branch rather than handle the error can
    /// gate on this first.
    pub fn has_real_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Extracts a per-hop F0 track from `pcm` by running the real CREPE
    /// forward.
    ///
    /// This is a straight delegation to [`extract_real`](Self::extract_real)
    /// — identical behaviour, identical errors. It exists so the obvious
    /// name on a loaded model is the one that measures pitch, matching
    /// [`super::rmvpe::RMVPE::extract`].
    ///
    /// # History
    ///
    /// Before 2026-08-15 this name returned `Vec<F0Frame>` and answered TWO
    /// different failures with the same all-zero track: no weights bound,
    /// and weights bound but `sample_rate != 16000`. A caller with a real
    /// checkpoint handing it 44.1 kHz audio got a frame-count-correct track
    /// of zeros — indistinguishable from "this audio is entirely unvoiced",
    /// and silently wrong pitch flows downstream into a vocoder or a VC
    /// pipeline. The timebase-only half of that behaviour now lives in
    /// [`frame_times`](Self::frame_times), which returns timestamps and
    /// nothing that could be read as a pitch estimate.
    ///
    /// # Errors
    ///
    /// Propagates [`extract_real`](Self::extract_real)'s errors verbatim —
    /// see that method for the list.
    pub fn extract(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, vokra_core::VokraError> {
        self.extract_real(pcm, sample_rate)
    }

    /// Runs the **real** CREPE forward on `pcm` and returns a per-hop F0
    /// track (`frames.len() == pcm.len() / hop`).
    ///
    /// Reachable both under this name and as [`extract`](Self::extract),
    /// which delegates here verbatim.
    ///
    /// It is fallible on purpose: a frame-count-correct all-zero track is
    /// indistinguishable downstream from "this audio is entirely unvoiced",
    /// so neither failure mode below is ever answered with one (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`vokra_core::VokraError::ModelLoad`] when the GGUF carried no
    ///   weight tensors — a metadata-only artifact, for which
    ///   [`has_real_weights`](Self::has_real_weights) reports `false`. Use
    ///   [`frame_times`](Self::frame_times) if the per-hop timestamps were
    ///   all that was wanted.
    /// - [`vokra_core::VokraError::InvalidArgument`] when `sample_rate` is
    ///   not [`NATIVE_SAMPLE_RATE`]. Upstream `crepe.predict` resamples
    ///   internally; Vokra names both the rate it received and the rate it
    ///   needs, and asks the caller to resample offline.
    ///
    /// Each error names what it got versus what it needs, so the two are
    /// never confused for one another at the call site.
    pub fn extract_real(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, vokra_core::VokraError> {
        let Some(weights) = self.weights.as_ref() else {
            return Err(vokra_core::VokraError::ModelLoad(format!(
                "crepe: no weight tensors were bound from this GGUF (metadata-only \
                 artifact — `conv1.weight` is absent), so the {} CNN cannot run; \
                 convert a real marl/crepe checkpoint with `vokra-cli convert --model \
                 crepe --config <config.json>`, or call `frame_times` if only the \
                 per-hop timestamps are wanted (FR-EX-08: never a zero-filled track)",
                self.cfg.capacity.as_tag(),
            )));
        };
        if sample_rate != NATIVE_SAMPLE_RATE {
            return Err(vokra_core::VokraError::InvalidArgument(format!(
                "crepe: got {sample_rate} Hz PCM but the CREPE CNN is defined only at \
                 {NATIVE_SAMPLE_RATE} Hz (its 1024-sample analysis frame and 360-bin \
                 cent grid are anchored to that rate) — resample offline and call \
                 again; Vokra never silently resamples (FR-EX-08)"
            )));
        }
        let hop = self.cfg.hop.max(1) as usize;
        // `sample_rate` is pinned to NATIVE_SAMPLE_RATE above, so it is
        // non-zero and the timestamp column cannot go NaN / ±inf here.
        let sr = sample_rate as f32;
        let n_frames = pcm.len() / hop;
        let compute = Compute::cpu();
        self.extract_full(pcm, weights, hop, sr, n_frames, &compute)
    }

    /// Returns the analysis timestamps [`extract`](Self::extract) will emit
    /// for a PCM buffer of `pcm_len` samples, in seconds from the start of
    /// the buffer.
    ///
    /// `result.len()` is the frame-count contract (`pcm_len / hop`,
    /// integer-truncated per [`CrepeConfig::hop`]); `result[i]` is the
    /// hop-aligned left edge of frame `i`. A `sample_rate` of `0` is clamped
    /// to `1` so the column stays finite rather than `NaN` / `±inf`.
    ///
    /// This runs no weights and cannot fail: it is pure arithmetic over the
    /// config, for callers that need to size or align a buffer before (or
    /// without) running the forward — including holders of a metadata-only
    /// GGUF. It deliberately does **not** return [`F0Frame`]: a frame carries
    /// `hz` / `voiced` / `confidence` columns this method has no evidence
    /// for, and emitting zeros there is exactly the fabricated track the
    /// 2026-08-15 fix removed. Mirrors
    /// [`super::rmvpe::RMVPE::frame_times`].
    pub fn frame_times(&self, pcm_len: usize, sample_rate: u32) -> Vec<f32> {
        let hop = self.cfg.hop.max(1) as usize;
        let n_frames = pcm_len / hop;
        let sr = (sample_rate as f32).max(1.0);
        (0..n_frames).map(|i| (i * hop) as f32 / sr).collect()
    }

    /// The real CNN forward — one frame per hop.
    ///
    /// Propagates any [`vokra_core::VokraError`] the per-frame forward
    /// raises rather than panicking: the shape invariants are checked at
    /// load time, but an error channel already exists here so there is no
    /// reason to turn a surprise into an abort.
    fn extract_full(
        &self,
        pcm: &[f32],
        w: &CrepeWeights,
        hop: usize,
        sr: f32,
        n_frames: usize,
        compute: &Compute,
    ) -> Result<Vec<F0Frame>, vokra_core::VokraError> {
        // Pre-pad by FRAME_LEN/2 so frame 0 is centered on `pcm[0]`
        // (upstream `crepe.core.get_activation`, `center=True` default).
        // The frame count is derived from the ORIGINAL PCM length so the
        // hop contract holds regardless of the padding — matches upstream
        // `1 + (len(audio) - 1024) / hop_length` after zero-padding
        // (frame_i's center = pcm[i * hop]).
        let mut buf = vec![0.0f32; pcm.len() + FRAME_LEN];
        let half = FRAME_LEN / 2;
        buf[half..half + pcm.len()].copy_from_slice(pcm);

        let mut frames = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let start = i * hop;
            let frame = &buf[start..start + FRAME_LEN];
            let activation = self.forward_one(frame, w, compute)?;
            let (cents, confidence) = local_average_cents(&activation, &self.cents_mapping);
            // cents == f32::NAN when the entire activation is zero — upstream
            // returns `nan_to_num(0)`; we mirror that so the caller sees
            // hz = 0 / voiced = false rather than a NaN.
            let hz = if cents.is_finite() {
                HZ_REF * 2.0f32.powf(cents / 1200.0)
            } else {
                0.0
            };
            let voiced = hz.is_finite() && hz > 0.0 && confidence > 0.0;
            frames.push(F0Frame {
                time_sec: (i * hop) as f32 / sr,
                hz: if voiced { hz } else { 0.0 },
                voiced,
                confidence,
            });
        }
        Ok(frames)
    }

    /// Runs the 6-block CNN + Dense classifier on a single 1024-sample frame.
    fn forward_one(
        &self,
        frame: &[f32],
        w: &CrepeWeights,
        compute: &Compute,
    ) -> Result<Vec<f32>, vokra_core::VokraError> {
        debug_assert_eq!(
            frame.len(),
            FRAME_LEN,
            "crepe forward: frame len must be {FRAME_LEN}"
        );

        // Per-frame mean/std normalization (upstream:
        // frames -= np.mean(frames, axis=1)[:, np.newaxis];
        // frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)).
        let mean = frame.iter().copied().sum::<f32>() / (FRAME_LEN as f32);
        let mut normed = vec![0.0f32; FRAME_LEN];
        for (o, &v) in normed.iter_mut().zip(frame.iter()) {
            *o = v - mean;
        }
        let var = normed.iter().map(|&x| x * x).sum::<f32>() / (FRAME_LEN as f32);
        let std = var.sqrt().max(STD_FLOOR);
        for v in normed.iter_mut() {
            *v /= std;
        }

        // Layout convention (mirror of `crate::speaker::camplus::conv2d`):
        // input = `[c, h, w=1]` row-major; kernel = `[c_out, c_in, kh, kw=1]`.
        // Block 1 input: (c=1, h=FRAME_LEN=1024, w=1).
        let mut c_cur: usize = 1;
        let mut h_cur: usize = FRAME_LEN;
        let mut x = normed;

        for block in &w.conv {
            let out = conv2d_freq(compute, &x, c_cur, h_cur, block)?;
            let h_conv = (h_cur + 2 * block.pad - block.kh) / block.sh + 1;
            debug_assert_eq!(out.len(), block.c_out * h_conv, "crepe: conv output shape");
            // BN + ReLU (upstream: Conv2D(activation='relu') → BN → MaxPool)
            // NB: Keras applies the conv activation BEFORE BatchNormalization,
            // even though the more common recipe puts BN first. See
            // `crepe/core.py::build_and_load_model` — the `activation='relu'`
            // sits on the Conv2D layer, and BatchNormalization is the *next*
            // layer, so at inference the effective order is exactly what we
            // write here.
            let mut a = out;
            for ci in 0..block.c_out {
                for hi in 0..h_conv {
                    let idx = ci * h_conv + hi;
                    let post_relu = a[idx].max(0.0);
                    a[idx] = post_relu * block.bn_scale[ci] + block.bn_shift[ci];
                }
            }
            // MaxPool(2, 1) (upstream `MaxPool2D(pool_size=(2, 1),
            // strides=None, padding='valid')`; strides=None in Keras means
            // pool_size, so the freq axis is halved by non-overlapping max).
            let h_pool = h_conv / 2;
            let mut pooled = vec![0.0f32; block.c_out * h_pool];
            for ci in 0..block.c_out {
                for hi in 0..h_pool {
                    let a_lo = a[ci * h_conv + 2 * hi];
                    let a_hi = a[ci * h_conv + 2 * hi + 1];
                    pooled[ci * h_pool + hi] = a_lo.max(a_hi);
                }
            }
            x = pooled;
            c_cur = block.c_out;
            h_cur = h_pool;
        }

        // Permute (`Permute((2, 1, 3))` in upstream = swap freq & channel).
        // The current layout is `[c_out5, h_after5, w=1]` = `[16m, 4, 1]`.
        // Flatten in Keras `data_format='channels_last'` order — the raw
        // memory after Permute is `[h=4, c=16m, w=1]`, so we transpose the
        // (c, h) block once and flatten. Length = `4 * 16m` = `flat_len()`.
        let flat_len = self.cfg.flat_len();
        debug_assert_eq!(x.len(), flat_len, "crepe: pre-Dense flat len");
        let mut flat = vec![0.0f32; flat_len];
        for hi in 0..h_cur {
            for ci in 0..c_cur {
                flat[hi * c_cur + ci] = x[ci * h_cur + hi];
            }
        }

        // Dense(360, activation='sigmoid') → GEMV of shape (N_BINS, flat_len).
        let mut logits = vec![0.0f32; N_BINS];
        compute.gemv_f32(
            N_BINS,
            flat_len,
            &w.classifier_w,
            &flat,
            Some(&w.classifier_b),
            &mut logits,
        )?;
        for v in logits.iter_mut() {
            // Numerically-stable logistic (avoid overflow at very negative x
            // by conditioning on the sign — mirror of `1 / (1 + exp(-x))`).
            *v = if *v >= 0.0 {
                let e = (-*v).exp();
                1.0 / (1.0 + e)
            } else {
                let e = v.exp();
                e / (1.0 + e)
            };
        }
        Ok(logits)
    }

    /// Access the loaded configuration.
    pub fn config(&self) -> &CrepeConfig {
        &self.cfg
    }
}

/// 2-D convolution along the freq axis with `w=kw=1` (mirrors the
/// upstream `Conv2D(kernel=(kh, 1), strides=(sh, 1), padding='same')`).
///
/// The `same` padding is precomputed at load time as `pad = kh / 2`
/// (Keras `same` is `ceil((in-1)/2)` — for the kernel widths CREPE uses
/// this is exactly `kh / 2`; the upstream layer's `output_shape` matches
/// `ceil(in/stride)` for stride 4 (block 1) and equals `in` for stride 1).
fn conv2d_freq(
    compute: &Compute,
    x: &[f32],
    c_in: usize,
    h_in: usize,
    block: &ConvBn,
) -> Result<Vec<f32>, vokra_core::VokraError> {
    debug_assert_eq!(c_in, block.c_in, "conv2d_freq: input c_in != weight c_in");
    debug_assert_eq!(
        x.len(),
        c_in * h_in,
        "conv2d_freq: x len != c_in*h_in ({} vs {}*{})",
        x.len(),
        c_in,
        h_in,
    );
    let kh = block.kh;
    let sh = block.sh;
    let pad = block.pad;
    let c_out = block.c_out;
    let h_out = (h_in + 2 * pad - kh) / sh + 1;

    // im2col: patch matrix of shape `[c_in * kh, h_out]`, row-major.
    let patch = c_in * kh;
    let mut col = vec![0.0f32; patch * h_out];
    for ci in 0..c_in {
        for ky in 0..kh {
            let row = (ci * kh + ky) * h_out;
            let plane = ci * h_in;
            for ho in 0..h_out {
                let ih = (ho * sh + ky) as isize - pad as isize;
                col[row + ho] = if ih < 0 || ih as usize >= h_in {
                    0.0
                } else {
                    x[plane + ih as usize]
                };
            }
        }
    }
    let mut out = vec![0.0f32; c_out * h_out];
    compute.gemm_f32(c_out, h_out, patch, &block.w, &col, None, &mut out)?;
    // Bias broadcast — Keras `Conv2D(..., use_bias=True)` default.
    for (r, chunk) in out.chunks_exact_mut(h_out).enumerate() {
        let b = block.b[r];
        for v in chunk {
            *v += b;
        }
    }
    Ok(out)
}

/// Local-averaging centroid → cents + confidence.
///
/// Upstream (`crepe.core.to_local_average_cents`):
/// ```text
///     center = argmax(salience)
///     start  = max(0, center - 4)
///     end    = min(360, center + 5)
///     product_sum = sum(salience[start:end] * cents_mapping[start:end])
///     weight_sum  = sum(salience[start:end])
///     return product_sum / weight_sum
/// ```
///
/// Confidence follows the paper (`activation.max(axis=1)`).
fn local_average_cents(activation: &[f32], cents_mapping: &[f32; N_BINS]) -> (f32, f32) {
    debug_assert_eq!(activation.len(), N_BINS, "activation must be 360-bin");
    let mut center = 0usize;
    let mut confidence = activation[0];
    for (i, &v) in activation.iter().enumerate() {
        if v > confidence {
            center = i;
            confidence = v;
        }
    }
    let start = center.saturating_sub(CENTROID_HALF_WIN);
    let end = (center + CENTROID_HALF_WIN + 1).min(N_BINS);
    let mut product_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for i in start..end {
        product_sum += activation[i] * cents_mapping[i];
        weight_sum += activation[i];
    }
    let cents = if weight_sum > 0.0 {
        product_sum / weight_sum
    } else {
        f32::NAN
    };
    (cents, confidence)
}

impl CrepeWeights {
    /// Binds all 6 conv blocks + the final Dense classifier from a GGUF.
    ///
    /// Tensor name schema (emitted by
    /// `crates/vokra-convert/src/models/crepe.rs`):
    ///
    /// - `conv{i}.weight` — `[c_out, c_in, kh, 1]` f32 (i = 1..=6)
    /// - `conv{i}.bias`   — `[c_out]` f32
    /// - `conv{i}.bn.gamma` / `.beta` / `.moving_mean` / `.moving_variance`
    ///   — `[c_out]` f32 each (Keras `BatchNormalization` default names,
    ///   without the trainable `moving_*` prefix collisions llama.cpp
    ///   causes)
    /// - `classifier.weight` — `[N_BINS, flat_len]` f32
    /// - `classifier.bias`   — `[N_BINS]` f32
    fn from_gguf(file: &GgufFile, cfg: CrepeConfig) -> Result<Self, LoadError> {
        let filters = cfg.filters();
        let mut conv_blocks: Vec<ConvBn> = Vec::with_capacity(6);
        let mut c_in_running = 1usize; // block 1 input channel count
        for (bi, (&filt, (&kh, &sh))) in filters
            .iter()
            .zip(KERNEL_WIDTH.iter().zip(STRIDE.iter()))
            .enumerate()
        {
            let idx = bi + 1;
            let w_name = format!("conv{idx}.weight");
            let b_name = format!("conv{idx}.bias");
            let g_name = format!("conv{idx}.bn.gamma");
            let be_name = format!("conv{idx}.bn.beta");
            let mm_name = format!("conv{idx}.bn.moving_mean");
            let mv_name = format!("conv{idx}.bn.moving_variance");

            let w_bytes = read_tensor_f32(file, &w_name)?;
            let expected_w = filt * c_in_running * kh;
            if w_bytes.len() != expected_w {
                return Err(LoadError::Gguf(format!(
                    "crepe conv{idx}: weight len {} != c_out({filt}) * c_in({c_in_running}) * kh({kh}) = {expected_w}",
                    w_bytes.len(),
                )));
            }
            let b_bytes = read_tensor_f32(file, &b_name)?;
            if b_bytes.len() != filt {
                return Err(LoadError::Gguf(format!(
                    "crepe conv{idx}: bias len {} != c_out({filt})",
                    b_bytes.len(),
                )));
            }
            let gamma = read_tensor_f32(file, &g_name)?;
            let beta = read_tensor_f32(file, &be_name)?;
            let mean = read_tensor_f32(file, &mm_name)?;
            let var = read_tensor_f32(file, &mv_name)?;
            for (label, v) in [
                ("gamma", &gamma),
                ("beta", &beta),
                ("moving_mean", &mean),
                ("moving_variance", &var),
            ] {
                if v.len() != filt {
                    return Err(LoadError::Gguf(format!(
                        "crepe conv{idx}: bn.{label} len {} != c_out({filt})",
                        v.len(),
                    )));
                }
            }
            let mut bn_scale = vec![0.0f32; filt];
            let mut bn_shift = vec![0.0f32; filt];
            for i in 0..filt {
                let s = gamma[i] / (var[i] + BN_EPS).sqrt();
                bn_scale[i] = s;
                bn_shift[i] = beta[i] - mean[i] * s;
            }

            // `same` pad: `kh / 2` matches Keras' `ceil((kh - 1) / 2)` for
            // all CREPE kernel sizes (512, 64) and both strides in play
            // (4, 1). Verified via
            // `ceil((in - 1) / sh) + 1 == ceil(in / sh)` at load time so a
            // future non-power-of-two kernel would fail loudly rather than
            // drift silently — SAME-in-Keras uses ceiling division of the
            // input, not the arithmetic identity we lean on here.
            let pad = kh / 2;

            conv_blocks.push(ConvBn {
                w: w_bytes,
                b: b_bytes,
                c_out: filt,
                c_in: c_in_running,
                kh,
                sh,
                pad,
                bn_scale,
                bn_shift,
            });
            c_in_running = filt;
        }
        let conv: [ConvBn; 6] = conv_blocks
            .try_into()
            .map_err(|_| LoadError::Gguf("crepe: internal — expected 6 conv blocks".to_owned()))?;

        let flat_len = cfg.flat_len();
        let classifier_w = read_tensor_f32(file, "classifier.weight")?;
        if classifier_w.len() != N_BINS * flat_len {
            return Err(LoadError::Gguf(format!(
                "crepe classifier: weight len {} != {N_BINS} * flat_len({flat_len}) = {}",
                classifier_w.len(),
                N_BINS * flat_len,
            )));
        }
        let classifier_b = read_tensor_f32(file, "classifier.bias")?;
        if classifier_b.len() != N_BINS {
            return Err(LoadError::Gguf(format!(
                "crepe classifier: bias len {} != {N_BINS}",
                classifier_b.len(),
            )));
        }

        Ok(Self {
            conv,
            classifier_w,
            classifier_b,
        })
    }
}

fn read_tensor_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, LoadError> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| LoadError::Gguf(format!("crepe: tensor `{name}` missing from GGUF")))?;
    if info.dtype != GgmlType::F32 {
        return Err(LoadError::Gguf(format!(
            "crepe: tensor `{name}` is {:?}, expected F32",
            info.dtype
        )));
    }
    file.tensor_f32(name)
        .map_err(|e| LoadError::Gguf(format!("crepe: read `{name}`: {e:?}")))
}

fn read_opt_u32(file: &GgufFile, key: &str) -> Result<Option<u32>, LoadError> {
    match file.get(key) {
        Some(v) => match v.as_u64().and_then(|n| u32::try_from(n).ok()) {
            Some(n) => Ok(Some(n)),
            None => Err(LoadError::Gguf(format!(
                "crepe metadata `{key}` is not a u32-range integer",
            ))),
        },
        None => Ok(None),
    }
}

fn read_opt_f32(file: &GgufFile, key: &str) -> Result<Option<f32>, LoadError> {
    match file.get(key) {
        Some(v) => match v.as_f64() {
            Some(n) => Ok(Some(n as f32)),
            None => Err(LoadError::Gguf(format!(
                "crepe metadata `{key}` is not a float",
            ))),
        },
        None => Ok(None),
    }
}

fn read_opt_string(file: &GgufFile, key: &str) -> Result<Option<String>, LoadError> {
    match file.get(key) {
        Some(v) => match v.as_str() {
            Some(s) => Ok(Some(s.to_owned())),
            None => Err(LoadError::Gguf(format!(
                "crepe metadata `{key}` is not a string",
            ))),
        },
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    /// Helper: encode an f32 slice into little-endian bytes.
    ///
    /// Kept module-local so the test file has no dependency on the runtime
    /// converter's `f32_slice_to_bytes` (which lives behind a different
    /// crate) — the whole point of these tests is to exercise the runtime
    /// side without a converter round-trip.
    fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn add_f32_tensor(b: &mut GgufBuilder, name: &str, dims: &[u64], values: Vec<f32>) {
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), f32_to_le_bytes(&values))
            .unwrap();
    }

    /// A GGUF path that does not exist must produce a [`LoadError`] rather
    /// than a panic or a silent success.
    #[test]
    fn crepe_load_stub_reports_load_error() {
        let path = Path::new("/vokra-nonexistent-crepe-fixture.gguf");
        let result = CREPE::from_gguf(path);
        assert!(
            result.is_err(),
            "expected LoadError for nonexistent path, got Ok",
        );
    }

    /// A metadata-only GGUF (no weight tensors) still loads, and
    /// `frame_times` still honors the frame-count contract — but it hands
    /// back bare timestamps, so nothing it returns can be read as a pitch
    /// estimate (FR-EX-08).
    #[test]
    fn crepe_metadata_only_gguf_frame_count_matches_hop() {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-crepe-metaonly-frame-count-{}.gguf",
            std::process::id(),
        ));
        let bytes = GgufBuilder::new().to_bytes().unwrap();
        std::fs::write(&tmp, &bytes).unwrap();

        let crepe = CREPE::from_gguf(&tmp).expect("load metadata-only GGUF");
        assert!(
            !crepe.has_real_weights(),
            "a metadata-only GGUF must not report bound weights",
        );
        let pcm_len = 1_600usize;
        let times = crepe.frame_times(pcm_len, 16_000);
        assert_eq!(times.len(), pcm_len / 160);
        for (i, t) in times.iter().enumerate() {
            let expected = (i * 160) as f32 / 16_000.0;
            assert!(
                (t - expected).abs() < 1e-9,
                "frame {i}: timestamp {t} != {expected}",
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// A metadata-only artifact must make `extract_real` fail LOUDLY, and
    /// with the "nothing bound" class specifically — distinct from the rate
    /// mismatch below, so a caller can tell the two apart.
    #[test]
    fn crepe_extract_real_refuses_unbound_weights() {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-crepe-unbound-weights-{}.gguf",
            std::process::id(),
        ));
        std::fs::write(&tmp, GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let crepe = CREPE::from_gguf(&tmp).expect("load metadata-only GGUF");

        let pcm = vec![0.1f32; 16 * 160];
        let Err(err) = crepe.extract_real(&pcm, 16_000) else {
            panic!("expected an error when no weight tensors were bound, got a track");
        };
        let msg = err.to_string();
        assert!(
            matches!(err, vokra_core::VokraError::ModelLoad(_)),
            "an unbound weight set is a model-load failure, got: {msg}",
        );
        assert!(
            msg.contains("conv1.weight"),
            "the error must name the tensor whose absence it detected: {msg}",
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// Bound weights + a non-16 kHz rate must be a LOUD error naming both
    /// rates — never a frame-count-correct all-zero track.
    ///
    /// Regression pin: the pre-2026-08-15 `extract` matched
    /// `Some(w) if sample_rate == 16_000` and let its `_` arm swallow a
    /// 44.1 kHz caller into zeros. Downstream (a vocoder, a VC pipeline)
    /// reads that as "entirely unvoiced" rather than "not measured", so the
    /// wrongness is silent and confident. Refusing is the point.
    #[test]
    fn crepe_extract_real_refuses_non_16k_with_bound_weights() {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-crepe-rate-mismatch-{}.gguf",
            std::process::id(),
        ));
        write_synthetic_crepe_gguf(&tmp);
        let crepe = CREPE::from_gguf(&tmp).expect("load synthetic-weight GGUF");
        assert!(
            crepe.has_real_weights(),
            "the synthetic fixture must bind weights for this test to mean anything",
        );

        let pcm = vec![0.1f32; 16 * 160];
        let Err(err) = crepe.extract_real(&pcm, 44_100) else {
            panic!("expected an error at 44.1 kHz with weights bound, got a track");
        };
        let msg = err.to_string();
        assert!(
            matches!(err, vokra_core::VokraError::InvalidArgument(_)),
            "a rate mismatch is a caller-argument failure, got: {msg}",
        );
        assert!(
            msg.contains("44100"),
            "the error must name the rate it received: {msg}",
        );
        assert!(
            msg.contains("16000"),
            "the error must name the rate it needs: {msg}",
        );

        // The obvious name must refuse too — a lenient `extract` sitting next
        // to a strict `extract_real` would re-open the exact hole this fix
        // closed, since `extract` is what a caller reaches for first.
        let Err(via_extract) = crepe.extract(&pcm, 44_100) else {
            panic!("`extract` must refuse the same input `extract_real` refuses");
        };
        assert_eq!(
            via_extract.to_string(),
            msg,
            "`extract` must delegate to `extract_real` verbatim, not soften it",
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// Writes a tiny but structurally complete CREPE GGUF (capacity `tiny`,
    /// identity BN fold, all-zero classifier weight with the bias biased at
    /// bin 180) to `path`, and returns the config it encodes.
    ///
    /// Shared by every test that needs `has_real_weights() == true`. The
    /// fabricated bundle goes through a real GGUF round-trip so these tests
    /// also pin the tensor-name schema `from_gguf` expects.
    fn write_synthetic_crepe_gguf(path: &Path) -> CrepeConfig {
        let cfg = CrepeConfig::defaults(CapacityFactor::Tiny);
        let filters = cfg.filters();
        let mut b = GgufBuilder::new();
        b.add_string(GGUF_KEY_CAPACITY, cfg.capacity.as_tag());
        b.add_u32(GGUF_KEY_HOP, cfg.hop);
        b.add_f32(GGUF_KEY_FMIN, cfg.fmin);
        b.add_f32(GGUF_KEY_FMAX, cfg.fmax);

        let mut c_in = 1usize;
        for (bi, ((&filt, &kh), _)) in filters
            .iter()
            .zip(KERNEL_WIDTH.iter())
            .zip(STRIDE.iter())
            .enumerate()
        {
            let idx = bi + 1;
            let w_len = filt * c_in * kh;
            let mut w = vec![0.0f32; w_len];
            // Small random-looking pattern (fixed seed = deterministic).
            for (i, v) in w.iter_mut().enumerate() {
                *v = ((i as f32) * 0.0007).sin() * 0.05;
            }
            add_f32_tensor(
                &mut b,
                &format!("conv{idx}.weight"),
                &[filt as u64, c_in as u64, kh as u64, 1],
                w,
            );
            add_f32_tensor(
                &mut b,
                &format!("conv{idx}.bias"),
                &[filt as u64],
                vec![0.0; filt],
            );
            // Identity BN (gamma=1, beta=0, mean=0, variance=1) so the fold
            // ==> scale ≈ 1, shift ≈ 0.
            add_f32_tensor(
                &mut b,
                &format!("conv{idx}.bn.gamma"),
                &[filt as u64],
                vec![1.0; filt],
            );
            add_f32_tensor(
                &mut b,
                &format!("conv{idx}.bn.beta"),
                &[filt as u64],
                vec![0.0; filt],
            );
            add_f32_tensor(
                &mut b,
                &format!("conv{idx}.bn.moving_mean"),
                &[filt as u64],
                vec![0.0; filt],
            );
            add_f32_tensor(
                &mut b,
                &format!("conv{idx}.bn.moving_variance"),
                &[filt as u64],
                vec![1.0; filt],
            );
            c_in = filt;
        }
        let flat = cfg.flat_len();
        add_f32_tensor(
            &mut b,
            "classifier.weight",
            &[N_BINS as u64, flat as u64],
            vec![0.0; N_BINS * flat],
        );
        // Bias biases bin 180 (arbitrary) so the argmax is deterministic on
        // an all-zero classifier.weight.
        let mut cbias = vec![0.0f32; N_BINS];
        cbias[180] = 1.0;
        add_f32_tensor(&mut b, "classifier.bias", &[N_BINS as u64], cbias);

        std::fs::write(path, b.to_bytes().unwrap()).unwrap();
        cfg
    }

    /// The capacity tag round-trips through the GGUF metadata layer.
    #[test]
    fn capacity_tag_roundtrip() {
        for c in [
            CapacityFactor::Tiny,
            CapacityFactor::Small,
            CapacityFactor::Medium,
            CapacityFactor::Large,
            CapacityFactor::Full,
        ] {
            assert_eq!(CapacityFactor::from_tag(c.as_tag()), Some(c));
        }
        assert!(CapacityFactor::from_tag("colossal").is_none());
    }

    /// Local-average-cents matches upstream on a delta activation and on a
    /// tight symmetric bump.
    #[test]
    fn local_average_cents_matches_analytic() {
        let mut cm = [0.0f32; N_BINS];
        for (i, v) in cm.iter_mut().enumerate() {
            *v = CENTS_OFFSET + CENTS_SPAN * (i as f32) / ((N_BINS - 1) as f32);
        }
        // Delta at bin 100 → cents == cm[100].
        let mut a = vec![0.0f32; N_BINS];
        a[100] = 1.0;
        let (cents, conf) = local_average_cents(&a, &cm);
        assert!((cents - cm[100]).abs() < 1e-3);
        assert!((conf - 1.0).abs() < 1e-6);
        // Tight symmetric bump at bin 200 → centroid == cm[200].
        let mut b = vec![0.0f32; N_BINS];
        b[199] = 0.5;
        b[200] = 1.0;
        b[201] = 0.5;
        let (cents_b, _) = local_average_cents(&b, &cm);
        assert!((cents_b - cm[200]).abs() < 1e-3);
    }

    /// Full end-to-end forward on a synthetic 1024-sample sinusoid + fabricated
    /// weights: exercises every op (conv/BN/ReLU/pool/dense/sigmoid + local-
    /// average-cents + Hz decode) so a regression in the plumbing is caught
    /// on CI without a real GGUF.
    ///
    /// The fabricated bundle is written to a real GGUF and re-loaded so the
    /// test also pins the tensor-name schema `from_gguf` expects.
    #[test]
    fn crepe_forward_synthetic_end_to_end() {
        let path =
            std::env::temp_dir().join(format!("vokra-crepe-forward-{}.gguf", std::process::id(),));
        write_synthetic_crepe_gguf(&path);
        let crepe = CREPE::from_gguf(&path).expect("load forward test GGUF");
        assert!(crepe.has_real_weights(), "the fixture binds a weight set");

        // Non-trivial waveform (a chirp) so per-frame normalization is real.
        let mut pcm = vec![0.0f32; 16 * 160]; // 16 frames' worth
        for (i, v) in pcm.iter_mut().enumerate() {
            let t = i as f32 / 16_000.0;
            *v = (2.0 * std::f32::consts::PI * (200.0 + 800.0 * t) * t).sin() * 0.3;
        }
        let frames = crepe
            .extract_real(&pcm, NATIVE_SAMPLE_RATE)
            .expect("bound weights at the native rate must produce a real track");
        assert_eq!(frames.len(), 16);
        // sigmoid(1.0) ≈ 0.731. With classifier.weight = 0 the flat vector
        // never contributes, so every frame's peak activation ≈ sigmoid(1.0)
        // at bin 180 and the decoded Hz = cents_mapping[180]-derived.
        let expected_conf = 1.0f32 / (1.0 + (-1.0f32).exp());
        for f in &frames {
            assert!(
                f.confidence > 0.7 && f.confidence < 0.8,
                "confidence {} not in ~sigmoid(1) range",
                f.confidence
            );
            assert!((f.confidence - expected_conf).abs() < 1e-4);
            assert!(
                f.voiced,
                "voiced flag must fire when confidence > 0 and hz > 0"
            );
            assert!(f.hz > 0.0);
            assert!(f.hz.is_finite());
        }
        let _ = std::fs::remove_file(&path);
    }
}
