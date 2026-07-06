//! Flat op enum and the audio graph descriptor (M0-02-T08/T09).
//!
//! FR-EX-01: "MVP は ggml 型 flat op enum" — the MVP IR is a ggml-style flat
//! op enum; an MLIR audio dialect + StableHLO is re-evaluated in v1.5+.
//!
//! # Design red line (permanent)
//!
//! **ONNX graphs are never loaded at runtime** (FR-LD-05, SRS §5-(2)): ONNX
//! models are handled exclusively by the offline conversion tool. This IR is
//! Vokra's own definition and depends on none of protobuf / abseil / onnx
//! (NFR-DS-02); the `deny.toml` bans list enforces this at the dependency
//! level.

use std::collections::{HashSet, VecDeque};

use crate::error::{Result, VokraError};

use super::fusion::{FusedOp, FusionRewrite};
use super::tensor::{Dim, TensorDesc, TensorId};

// ===========================================================================
// Audio front-end operator attributes (M0-04-T01; FR-OP-01 / FR-OP-03)
// ===========================================================================
//
// These attribute types are embedded in the speech front-end [`OpKind`]
// variants (`Stft` / `Istft` / `MelFilterbank` / `Mfcc` / `Dct`) so the audio
// graph descriptor can *represent* a log-mel / MFCC front-end (the Whisper
// front-end of M0-06 is assembled from these). The operator implementations
// live in `vokra-ops`, which depends on `vokra-core` and consumes these exact
// types; for ergonomics they are re-exported there as `vokra_ops::attrs::*`.
//
// Placement note — deviation from M0-04-T01, which proposed
// `crates/vokra-ops/src/attrs.rs`: because [`OpKind`] lives in `vokra-core`
// and `vokra-core` must not depend on `vokra-ops` (the crate dependency edge
// runs the other way), any type an `OpKind` variant embeds has to be defined
// in `vokra-core`. The types therefore live here in the IR module.
//
// STFT ≠ FFT (CLAUDE.md pitfall): an STFT is *window + framing + normalization
// + causal mode* wrapped around an FFT, not a bare FFT. Every one of those
// knobs is an explicit attribute below rather than an implicit default.
//
// Out of scope for M0-04 (kept representable so this maps 1:1 onto the future
// `vokra.frontend.*` GGUF chunk): the chunk bit-exact check (FR-LD-03, M1), the
// GPU side of the FFT lowering (FR-OP-05) and the public `complex64` IR dtype
// (FR-EX-09, v0.1 MVP — FFT complex values stay a `vokra-ops`-internal type for
// now). Streaming iSTFT (`istft_streaming`, FR-OP-02) has since **landed in
// M2-05** — see [`IstftStreamingAttrs`] and [`OpKind::IstftStreaming`] below.

/// Analysis / synthesis window function
/// (FR-OP-01: "window (Hann/Hamming/Blackman-Harris/Kaiser)").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Window {
    /// Hann (raised-cosine) window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Four-term Blackman-Harris window.
    BlackmanHarris,
    /// Kaiser window with shape parameter `beta`.
    Kaiser {
        /// Kaiser shape parameter β: larger β lowers side-lobes and widens the
        /// main lobe.
        beta: f32,
    },
}

/// Whether a length-`N` window samples its periodic (DFT-even) or symmetric
/// form.
///
/// STFT front-ends use the **periodic** form — it matches
/// `torch.*_window(..., periodic=True)` and librosa's `sym=False`; filter
/// design and one-shot analysis use the symmetric form. Kept explicit because
/// reference implementations disagree on the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSymmetry {
    /// Periodic (DFT-even) window of period `N`: `w[n] = f(2πn / N)`.
    Periodic,
    /// Symmetric window: `w[n] = f(2πn / (N - 1))`.
    Symmetric,
}

/// Signal-extension mode used by `center` padding (FR-OP-01: "pad_mode").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Pad with zeros.
    Constant,
    /// Mirror the signal about the boundary sample without repeating the edge,
    /// matching `torch` / librosa `pad_mode="reflect"`.
    Reflect,
    /// Repeat the edge sample.
    Edge,
}

/// FFT normalization convention
/// (FR-OP-01: "normalization ('forward'/'backward'/'ortho')").
///
/// For a length-`N` transform: `Forward` scales the forward transform by
/// `1/N`, `Backward` scales the inverse by `1/N` (the engineering default),
/// and `Ortho` scales both directions by `1/√N` (unitary / energy-preserving).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// `1/N` applied on the forward transform only.
    Forward,
    /// `1/N` applied on the inverse transform only (default).
    Backward,
    /// `1/√N` applied on both directions (unitary).
    Ortho,
}

/// Hz→mel warping convention (FR-OP-03: "Slaney/HTK 両対応").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// HTK formula: `mel = 2595 · log10(1 + f / 700)`.
    Htk,
    /// Slaney (auditory-toolbox) scale — linear below 1 kHz, logarithmic above;
    /// the librosa default.
    Slaney,
}

/// Mel filter-bank normalization (FR-OP-03).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    /// No normalization; each triangular filter peaks at 1.0.
    None,
    /// Slaney normalization: each filter scaled to unit area (librosa
    /// `norm="slaney"`).
    Slaney,
}

/// Domain in which the triangular filter slopes are interpolated (FR-OP-03).
///
/// The band *edges* are always uniform on the mel scale; this selects the
/// domain of the rising/falling ramps between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelInterp {
    /// librosa/torchaudio convention: ramps are linear in **Hz** (the weight of
    /// an FFT bin is a triangle over frequency). The Vokra default.
    Hz,
    /// Kaldi convention: ramps are linear in the **mel** domain (the weight of
    /// an FFT bin is a triangle over `mel(f)`). Needed for the CosyVoice /
    /// 3D-Speaker CAM++ Kaldi fbank front-end.
    Mel,
}

/// Attributes of the `stft` operator (FR-OP-01).
///
/// Covers all eight FR-OP-01 knobs explicitly — `window`, `hop_length`,
/// `n_fft`, `center` (padding), `pad_mode`, `normalization`, `causal` and
/// `real_input` — plus `win_length` (a `vokra.frontend.*` key; a window
/// shorter than `n_fft` is centered and zero-padded to `n_fft`).
#[derive(Debug, Clone, PartialEq)]
pub struct StftAttrs {
    /// FFT size: the frame length in samples handed to the transform.
    pub n_fft: usize,
    /// Hop between successive frames, in samples.
    pub hop_length: usize,
    /// Window length in samples (`<= n_fft`; centered and zero-padded to
    /// `n_fft`).
    pub win_length: usize,
    /// Window function.
    pub window: Window,
    /// Periodic vs symmetric sampling of the window.
    pub window_symmetry: WindowSymmetry,
    /// When `true`, the signal is padded by `n_fft / 2` on both ends so frame
    /// `t` is centered on sample `t · hop_length`.
    pub center: bool,
    /// Padding mode applied when `center` is `true`.
    pub pad_mode: PadMode,
    /// FFT normalization convention.
    pub normalization: Normalization,
    /// Causal mode: no look-ahead. When `true`, framing pads only the left
    /// (history) side, so frame `t` depends only on samples `<= t · hop_length`.
    pub causal: bool,
    /// Emit a real-input (RFFT) spectrum of `n_fft / 2 + 1` bins. When `false`,
    /// the full `n_fft`-bin complex spectrum is produced.
    pub real_input: bool,
}

impl StftAttrs {
    /// Builds attributes with librosa/`torch`-like defaults: `win_length =
    /// n_fft`, Hann window sampled periodically, `center = true` with reflect
    /// padding, backward normalization, non-causal, real (RFFT) input.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            win_length: n_fft,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            pad_mode: PadMode::Reflect,
            normalization: Normalization::Backward,
            causal: false,
            real_input: true,
        }
    }
}

/// Attributes of the `istft` operator (FR-OP-01).
///
/// The inverse of [`StftAttrs`]; it mirrors the analysis parameters that must
/// agree for a faithful overlap-add reconstruction. `istft_streaming`
/// (FR-OP-02, tail buffering / per-layer state carry-over) is v0.5 and out of
/// scope here.
#[derive(Debug, Clone, PartialEq)]
pub struct IstftAttrs {
    /// FFT size used by the forward analysis.
    pub n_fft: usize,
    /// Hop between successive frames, in samples.
    pub hop_length: usize,
    /// Window length in samples (`<= n_fft`).
    pub win_length: usize,
    /// Synthesis window (must match the analysis window for exact inversion).
    pub window: Window,
    /// Periodic vs symmetric sampling of the window.
    pub window_symmetry: WindowSymmetry,
    /// Whether the forward transform used `center` padding (both ends are
    /// trimmed on reconstruction when `true`).
    pub center: bool,
    /// FFT normalization convention used by the forward analysis.
    pub normalization: Normalization,
    /// Whether the input spectrum is a real (RFFT) half-spectrum.
    pub real_input: bool,
    /// Target output length in samples; `None` infers it from the frame count.
    pub length: Option<usize>,
}

impl IstftAttrs {
    /// Builds inverse attributes matching [`StftAttrs::new`] defaults.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            win_length: n_fft,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            normalization: Normalization::Backward,
            real_input: true,
            length: None,
        }
    }
}

/// Attributes of the `istft_streaming` operator (FR-OP-02; M2-05).
///
/// The chunked, tail-buffering variant of [`OpKind::Istft`]. It wraps the batch
/// [`IstftAttrs`] verbatim — the streaming op reconstructs, chunk-by-chunk,
/// *bit-for-bit* what the one-shot `istft` produces for the same parameters —
/// and adds the FR-OP-02 knob the batch op has no need for: **`tail_len`**, the
/// length of the overlap tail carried across chunk boundaries (the per-layer
/// state the streaming reconstruction retains).
///
/// The overlap between two successive frames is `n_fft − hop_length` samples;
/// that is the minimum `tail_len` a faithful carry-over needs, and the value
/// [`new`](Self::new) / [`from_istft`](Self::from_istft) default to. A smaller
/// `tail_len` cannot reconstruct the inter-frame overlap and the streaming op
/// rejects it (`InvalidArgument`); a larger one is accepted (it merely reserves
/// a deeper tail buffer). The `center` head/tail trim is handled internally and
/// does *not* consume `tail_len`.
///
/// Per-layer carry-over: a vocoder chain may hold several iSTFT stages, each an
/// independent streaming instance with its own tail state; this attribute set
/// describes one such stage. The tail state itself is never a graph tensor — it
/// lives in the stream handle (FR-ST-05).
#[derive(Debug, Clone, PartialEq)]
pub struct IstftStreamingAttrs {
    /// The batch inverse-STFT parameters (the eight FR-OP-01 knobs + `length`).
    /// The streaming variant reproduces this exact reconstruction.
    pub istft: IstftAttrs,
    /// Overlap tail length in samples carried across chunk boundaries
    /// (FR-OP-02). Must be `>= n_fft − hop_length` (the inter-frame overlap).
    pub tail_len: usize,
}

impl IstftStreamingAttrs {
    /// Builds streaming attributes from [`IstftAttrs::new`] defaults, with
    /// `tail_len` set to the inter-frame overlap `n_fft − hop_length`.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self::from_istft(IstftAttrs::new(n_fft, hop_length))
    }

    /// Wraps an existing [`IstftAttrs`], defaulting `tail_len` to the
    /// inter-frame overlap `n_fft − hop_length` (saturating at 0 when
    /// `hop_length >= n_fft`, i.e. no overlap).
    pub fn from_istft(istft: IstftAttrs) -> Self {
        let tail_len = istft.n_fft.saturating_sub(istft.hop_length);
        Self { istft, tail_len }
    }
}

/// Attributes of the `mel_filterbank` operator (FR-OP-03).
///
/// Describes the triangular mel filter bank projecting an `n_fft / 2 + 1`-bin
/// power (or magnitude) spectrum onto `n_mels` mel bands.
#[derive(Debug, Clone, PartialEq)]
pub struct MelAttrs {
    /// Sample rate of the analysed audio, in Hz.
    pub sample_rate: u32,
    /// FFT size the spectrum was produced with (bins = `n_fft / 2 + 1`).
    pub n_fft: usize,
    /// Number of mel bands.
    pub n_mels: usize,
    /// Lowest band edge, in Hz.
    pub fmin: f32,
    /// Highest band edge, in Hz; `None` uses the Nyquist frequency
    /// (`sample_rate / 2`).
    pub fmax: Option<f32>,
    /// Hz→mel warping convention.
    pub scale: MelScale,
    /// Filter-bank normalization.
    pub norm: MelNorm,
    /// Domain of the triangular filter ramps ([`MelInterp::Hz`] = librosa,
    /// [`MelInterp::Mel`] = Kaldi).
    pub interp: MelInterp,
}

impl MelAttrs {
    /// Builds attributes with librosa-like defaults: `fmin = 0`, `fmax =`
    /// Nyquist, Slaney scale, Slaney normalization, Hz-domain interpolation.
    pub fn new(sample_rate: u32, n_fft: usize, n_mels: usize) -> Self {
        Self {
            sample_rate,
            n_fft,
            n_mels,
            fmin: 0.0,
            fmax: None,
            scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
            interp: MelInterp::Hz,
        }
    }
}

/// Attributes of the `dct` operator (FR-OP-03) — DCT-II.
///
/// `normalization` selects the scaling of the DCT-II: [`Normalization::Ortho`]
/// is the orthonormal DCT-II (`scipy` `norm="ortho"`, torchaudio's MFCC DCT),
/// while [`Normalization::Backward`] is the unnormalized DCT-II
/// (`scipy` default). [`Normalization::Forward`] additionally scales the
/// unnormalized transform by `1/N`.
#[derive(Debug, Clone, PartialEq)]
pub struct DctAttrs {
    /// Number of leading coefficients to keep; `None` keeps all `N`.
    pub n_out: Option<usize>,
    /// Scaling convention of the DCT-II.
    pub normalization: Normalization,
}

impl DctAttrs {
    /// Builds an orthonormal DCT-II keeping all coefficients.
    pub fn new() -> Self {
        Self {
            n_out: None,
            normalization: Normalization::Ortho,
        }
    }
}

impl Default for DctAttrs {
    fn default() -> Self {
        Self::new()
    }
}

/// Attributes of the `mfcc` operator (FR-OP-03).
///
/// MFCCs are computed as `dct(ln(max(mel, log_floor)))`: a [`MelAttrs`] mel
/// projection, a natural-log compression floored at `log_floor`, then a
/// DCT-II keeping `n_mfcc` coefficients. Reference tools differ in the log
/// stage (librosa's `power_to_db` uses `10·log10`); matching a specific one is
/// a parity-fixture concern (M0-04-T16) rather than a structural one.
#[derive(Debug, Clone, PartialEq)]
pub struct MfccAttrs {
    /// Mel filter-bank attributes feeding the cepstral transform.
    pub mel: MelAttrs,
    /// Number of cepstral coefficients (DCT-II outputs) to keep.
    pub n_mfcc: usize,
    /// Scaling convention of the DCT-II (typically [`Normalization::Ortho`]).
    pub dct_norm: Normalization,
    /// Lower floor applied before the natural log to avoid `ln(0)`.
    pub log_floor: f32,
}

impl MfccAttrs {
    /// Builds MFCC attributes: orthonormal DCT-II, `log_floor = 1e-10`.
    pub fn new(mel: MelAttrs, n_mfcc: usize) -> Self {
        Self {
            mel,
            n_mfcc,
            dct_norm: Normalization::Ortho,
            log_floor: 1e-10,
        }
    }
}

/// Attributes of the `resample` operator (FR-OP-04, M1-06).
///
/// Polyphase Kaiser-windowed-sinc sample-rate conversion. Both rates are graph
/// attributes — the audio graph fixes the capture rate at build time — so a
/// node fully describes the conversion; the implementation lives in `vokra-ops`
/// (`resample`). This is the [`OpKind::Resample`] wrapping of the standalone
/// M1-06 op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleAttrs {
    /// Input sample rate in Hz.
    pub in_rate: u32,
    /// Output sample rate in Hz.
    pub out_rate: u32,
    /// Filter quality (higher = sharper transition band, more taps).
    pub quality: u8,
}

/// Attributes of the `pre_emphasis` operator (FR-OP-64, M1-06).
///
/// First-order high-pass applied ahead of framing; the [`OpKind::PreEmphasis`]
/// wrapping of the standalone M1-06 op. DC-offset removal, its companion, takes
/// no parameters and is the attribute-less [`OpKind::DcOffsetRemove`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreEmphasisAttrs {
    /// Pre-emphasis coefficient `a` in `y[n] = x[n] − a·x[n−1]`; `0.0` is the
    /// identity filter.
    pub coeff: f32,
}

/// Operation kind — the ggml-style *flat op enum* of the Vokra IR (FR-EX-01).
///
/// M0-02 carried only minimal **placeholder** variants so the graph plumbing
/// could be exercised. Op families are added by their owning work packages —
/// do not add them here ahead of schedule:
///
/// - speech front-end ops (`stft` / `istft` / `mel_filterbank` / `mfcc` /
///   `dct`, FR-OP-01/03) and their attribute definitions: **M0-04** (landed —
///   see [`StftAttrs`], [`MelAttrs`], … above);
/// - amplitude preprocessing ops (`resample` / `dc_offset_remove` /
///   `pre_emphasis`, FR-OP-04/64): **M1-06** (landed — see [`ResampleAttrs`],
///   [`PreEmphasisAttrs`]);
/// - LSTM family for the Silero VAD subgraph: **M0-05**;
/// - attention / decoder family for Whisper: **M0-06**.
///
/// The enum is `#[non_exhaustive]` so those additions do not break
/// downstream matches; backends must treat unknown ops as unsupported
/// (explicit error, FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OpKind {
    /// Matrix multiplication (placeholder).
    MatMul,
    /// Element-wise addition (placeholder).
    Add,
    /// Element-wise multiplication (placeholder).
    Mul,
    /// Softmax over the innermost dimension (placeholder).
    Softmax,
    /// Short-time Fourier transform (FR-OP-01). Real signal `[samples]` →
    /// complex spectrogram `[frames, bins]`; implemented in `vokra-ops`.
    Stft(StftAttrs),
    /// Inverse STFT / overlap-add resynthesis (FR-OP-01). Complex spectrogram
    /// `[frames, bins]` → real signal `[samples]`.
    Istft(IstftAttrs),
    /// Streaming inverse STFT (FR-OP-02, M2-05): the chunked, tail-buffering
    /// variant of [`OpKind::Istft`]. A one-shot node (all frames in a single
    /// evaluation) reconstructs bit-for-bit what `Istft` produces; the per-chunk
    /// overlap tail is carried by the stream handle (FR-ST-05), never exposed as
    /// a graph tensor.
    IstftStreaming(IstftStreamingAttrs),
    /// Mel filter-bank projection of a power/magnitude spectrum
    /// `[frames, bins]` → `[frames, n_mels]` (FR-OP-03).
    MelFilterbank(MelAttrs),
    /// Mel-frequency cepstral coefficients `[frames, bins]` → `[frames, n_mfcc]`
    /// (FR-OP-03).
    Mfcc(MfccAttrs),
    /// Discrete cosine transform, type II, over the innermost axis (FR-OP-03).
    Dct(DctAttrs),
    /// Sample-rate conversion (FR-OP-04). Real `[samples]` → real `[samples']`;
    /// implemented in `vokra-ops`.
    Resample(ResampleAttrs),
    /// DC-offset removal — subtract the per-utterance mean (FR-OP-64). Real
    /// `[samples]` → real `[samples]`.
    DcOffsetRemove,
    /// First-order pre-emphasis high-pass (FR-OP-64). Real `[samples]` → real
    /// `[samples]`.
    PreEmphasis(PreEmphasisAttrs),
    /// Fused op emitted by the M2-04 graph-fusion pass (see [`crate::ir::fusion`]).
    ///
    /// The pass rewrites recognized subgraphs (currently only `Stft → Power →
    /// MelFilterbank → Log` for the log-mel front-end) into a single fused node.
    /// Backends opt in by returning `true` from
    /// [`Backend::supports`](crate::Backend::supports) for the corresponding
    /// [`FusedOp`] variant and providing an `eval_op` kernel; the fusion pass
    /// checks `supports` and, when the target backend does not cover a fused
    /// variant, leaves the base ops in place (de-fusion — FR-EX-08 rules out
    /// silent fallback). The wrapped `FusedOp` is itself `#[non_exhaustive]` so
    /// new fused patterns (Snake / BigVGAN AMP …) can be added without breaking
    /// downstream matches.
    Fused(FusedOp),
}

/// One node of an [`AudioGraph`]: an op together with its tensor
/// inputs / outputs (referenced by [`TensorId`]).
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) op: OpKind,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
}

impl Node {
    /// The operation this node performs.
    pub fn op(&self) -> &OpKind {
        &self.op
    }

    /// Tensors read by this node.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Tensors written by this node.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }
}

/// The *audio graph descriptor* — Vokra's own IR container (FR-EX-01).
///
/// A graph owns a tensor table ([`TensorDesc`]) plus a flat list of
/// [`Node`]s, and declares which tensors are graph inputs / outputs.
/// Construct it with [`GraphBuilder`]; [`AudioGraph::validate`] checks
/// structural consistency.
#[derive(Debug, Clone)]
pub struct AudioGraph {
    pub(crate) tensors: Vec<TensorDesc>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
}

impl AudioGraph {
    /// Tensor table of the graph.
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    /// Descriptor for `id`, or `None` if `id` is out of range.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors.get(id.0)
    }

    /// Nodes in insertion order (M0: no scheduling / topological pass yet).
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Tensors declared as graph inputs.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Tensors declared as graph outputs.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    /// Validates structural consistency of the graph (M0-02-T09).
    ///
    /// Checks performed:
    ///
    /// - every [`TensorId`] referenced by nodes and by the graph
    ///   input / output lists is in range (no dangling ids),
    /// - tensor names are unique,
    /// - a fully-static tensor's element count does not overflow `usize`
    ///   (element-count checks are **skipped on [`Dim::Dynamic`] axes**, whose
    ///   extent is variable-length — M1-04 sub-part 2),
    /// - every tensor is produced by at most one node (single-producer
    ///   consistency of node outputs).
    ///
    /// Op-kind specific *shape* checks (e.g. that a
    /// [`FusedOp::LogMel`](super::fusion::FusedOp::LogMel) input is a
    /// `[samples]` real signal and its output a `[n_mels, n_frames]` mel
    /// spectrogram) are **deferred to the backend**: `validate` only enforces
    /// structural consistency and accepts every [`OpKind`] variant — including
    /// [`OpKind::Fused`] added by the M2-04 fusion pass — uniformly by
    /// walking `inputs` / `outputs`.
    ///
    /// Violations are reported as [`VokraError::GraphValidation`].
    pub fn validate(&self) -> Result<()> {
        let len = self.tensors.len();

        let mut names: HashSet<&str> = HashSet::with_capacity(len);
        for desc in &self.tensors {
            if !names.insert(desc.name.as_str()) {
                return Err(VokraError::GraphValidation(format!(
                    "duplicate tensor name `{}`",
                    desc.name
                )));
            }
        }

        // Element-count sanity, skipping symbolic axes. A tensor with any
        // `Dim::Dynamic` axis has a variable-length extent (M1-04 sub-part 2):
        // its element count is intentionally unknown, so element-count checks do
        // not apply. A *fully static* tensor whose fixed extents overflow
        // `usize`, by contrast, can never be allocated and is rejected here.
        for desc in &self.tensors {
            let fully_static = desc.shape.iter().all(|d| matches!(d, Dim::Fixed(_)));
            if fully_static && desc.num_elements().is_none() {
                return Err(VokraError::GraphValidation(format!(
                    "tensor `{}` has a static shape whose element count overflows usize",
                    desc.name
                )));
            }
        }

        for (i, node) in self.nodes.iter().enumerate() {
            for id in &node.inputs {
                check_id(*id, len, &format!("node #{i} ({:?}) input", node.op))?;
            }
            for id in &node.outputs {
                check_id(*id, len, &format!("node #{i} ({:?}) output", node.op))?;
            }
        }

        let mut producer: Vec<Option<usize>> = vec![None; len];
        for (i, node) in self.nodes.iter().enumerate() {
            for id in &node.outputs {
                if let Some(prev) = producer[id.0] {
                    return Err(VokraError::GraphValidation(format!(
                        "tensor `{}` is produced by both node #{prev} and node #{i}",
                        self.tensors[id.0].name
                    )));
                }
                producer[id.0] = Some(i);
            }
        }

        for id in &self.inputs {
            check_id(*id, len, "graph input")?;
        }
        for id in &self.outputs {
            check_id(*id, len, "graph output")?;
        }

        Ok(())
    }

    /// Returns node indices in a topological (dataflow) order: every node
    /// appears after all nodes producing tensors it reads.
    ///
    /// The dependency edges are `producer(t) → consumer(t)` for each tensor `t`
    /// a node reads that another node writes; leaf tensors (graph inputs,
    /// constants / weights) contribute no edge. Ordering uses Kahn's algorithm
    /// seeded and processed in ascending node index, so independent nodes keep
    /// their insertion order and the schedule is deterministic run-to-run — a
    /// property the graph evaluator ([`run_graph`](crate::run_graph)) and the
    /// M2-04 fusion pass both rely on.
    ///
    /// This is the scheduling pass `nodes()` deliberately omits. It reuses the
    /// single-producer table [`validate`](Self::validate) guarantees, so the
    /// producer of each tensor is unambiguous.
    ///
    /// # Errors
    ///
    /// [`VokraError::GraphValidation`] if the graph contains a cycle (a node
    /// that transitively depends on its own output), for which no topological
    /// order exists.
    pub fn topo_order(&self) -> Result<Vec<usize>> {
        let n_nodes = self.nodes.len();
        let n_tensors = self.tensors.len();

        // Single-producer table (post-`validate`, at most one writer per tensor).
        let mut producer: Vec<Option<usize>> = vec![None; n_tensors];
        for (i, node) in self.nodes.iter().enumerate() {
            for id in &node.outputs {
                producer[id.0] = Some(i);
            }
        }

        // Dataflow edges producer → consumer, and each node's in-degree. A node
        // consuming its own output creates a self-edge (in-degree never hits 0)
        // and is reported as a cycle below. Duplicate edges (two inputs from the
        // same producer) are kept: in-degree and adjacency stay consistent.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        let mut in_degree: Vec<usize> = vec![0; n_nodes];
        for (consumer, node) in self.nodes.iter().enumerate() {
            for id in &node.inputs {
                if let Some(producer_node) = producer[id.0] {
                    successors[producer_node].push(consumer);
                    in_degree[consumer] += 1;
                }
            }
        }

        let mut ready: VecDeque<usize> = (0..n_nodes).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n_nodes);
        while let Some(node) = ready.pop_front() {
            order.push(node);
            for &consumer in &successors[node] {
                in_degree[consumer] -= 1;
                if in_degree[consumer] == 0 {
                    ready.push_back(consumer);
                }
            }
        }

        if order.len() != n_nodes {
            return Err(VokraError::GraphValidation(
                "audio graph contains a cycle: no topological node order exists".to_owned(),
            ));
        }
        Ok(order)
    }
}

impl AudioGraph {
    // M2-04: mutations restricted to ir::fusion pass
    /// Applies fusion rewrites in-place (M2-04-T02). Restricted to the fusion
    /// pass in [`crate::ir::fusion`] — no code path outside the IR module may
    /// mutate an `AudioGraph` after construction (§R2 of the M2-04 plan; the
    /// public builder-only invariant is preserved).
    ///
    /// This is the *only* mutation entry point on [`AudioGraph`] outside
    /// [`GraphBuilder`], and its visibility is narrowed to
    /// `pub(in crate::ir)`; the `#[doc(hidden)]` attribute keeps it off the
    /// public rustdoc surface (see risk register R2).
    ///
    /// Each [`FusionRewrite`](super::fusion::FusionRewrite) is applied in
    /// order: nodes whose indices appear in `removed_nodes` are dropped, and
    /// each rewrite's `inserted_node` is appended. Callers must guarantee the
    /// resulting node set is still structurally valid — `run_graph` calls
    /// [`topo_order`](Self::topo_order) which will detect any residual cycles.
    #[doc(hidden)]
    pub(in crate::ir) fn rewrite_with(&mut self, rewrites: Vec<FusionRewrite>) {
        if rewrites.is_empty() {
            return;
        }
        // Collect every removed index across all rewrites, then drop them in a
        // single pass (avoids O(n·r) shifts and preserves relative order of the
        // survivors).
        let mut removed: HashSet<usize> = HashSet::new();
        let mut inserted: Vec<Node> = Vec::with_capacity(rewrites.len());
        for r in rewrites {
            removed.extend(r.removed_nodes.iter().copied());
            inserted.push(r.inserted_node);
        }
        let mut kept: Vec<Node> = Vec::with_capacity(self.nodes.len() - removed.len());
        for (i, node) in self.nodes.drain(..).enumerate() {
            if !removed.contains(&i) {
                kept.push(node);
            }
        }
        kept.extend(inserted);
        self.nodes = kept;
    }
}

fn check_id(id: TensorId, len: usize, what: &str) -> Result<()> {
    if id.0 >= len {
        return Err(VokraError::GraphValidation(format!(
            "{what} references tensor id {} but the graph has only {len} tensors",
            id.0
        )));
    }
    Ok(())
}

/// Incremental builder for [`AudioGraph`] (M0-02-T09).
///
/// [`GraphBuilder::finish`] runs [`AudioGraph::validate`] so an invalid
/// graph is rejected at construction time.
///
/// # Examples
///
/// ```
/// use vokra_core::{DType, GraphBuilder, OpKind, TensorDesc};
///
/// let mut builder = GraphBuilder::new();
/// let x = builder.add_tensor(TensorDesc::new("x", DType::F32, [2, 4]));
/// let w = builder.add_tensor(TensorDesc::new("w", DType::F32, [4, 8]));
/// let y = builder.add_tensor(TensorDesc::new("y", DType::F32, [2, 8]));
/// builder.add_node(OpKind::MatMul, &[x, w], &[y]);
/// builder.mark_input(x);
/// builder.mark_output(y);
///
/// let graph = builder.finish().expect("graph is structurally valid");
/// assert_eq!(graph.nodes().len(), 1);
/// assert_eq!(graph.tensor(y).unwrap().name, "y");
/// ```
#[derive(Debug, Default)]
pub struct GraphBuilder {
    tensors: Vec<TensorDesc>,
    nodes: Vec<Node>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
}

impl GraphBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tensor and returns its id within the graph being built.
    pub fn add_tensor(&mut self, desc: TensorDesc) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(desc);
        id
    }

    /// Appends a node executing `op` over the given tensors.
    pub fn add_node(&mut self, op: OpKind, inputs: &[TensorId], outputs: &[TensorId]) {
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        });
    }

    /// Declares `id` as a graph input.
    pub fn mark_input(&mut self, id: TensorId) {
        self.inputs.push(id);
    }

    /// Declares `id` as a graph output.
    pub fn mark_output(&mut self, id: TensorId) {
        self.outputs.push(id);
    }

    /// Finalizes the graph, running [`AudioGraph::validate`].
    pub fn finish(self) -> Result<AudioGraph> {
        let graph = AudioGraph {
            tensors: self.tensors,
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
        };
        graph.validate()?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::{DType, Dim};

    fn desc(name: &str) -> TensorDesc {
        TensorDesc::new(name, DType::F32, [2, 2])
    }

    fn assert_graph_validation_err(result: Result<AudioGraph>, needle: &str) {
        match result {
            Err(VokraError::GraphValidation(msg)) => {
                assert!(
                    msg.contains(needle),
                    "message `{msg}` should contain `{needle}`"
                );
            }
            other => panic!("expected GraphValidation error, got {other:?}"),
        }
    }

    #[test]
    fn small_graph_builds_and_validates() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let w = b.add_tensor(desc("w"));
        let h = b.add_tensor(desc("h"));
        let bias = b.add_tensor(desc("bias"));
        let y = b.add_tensor(desc("y"));
        b.add_node(OpKind::MatMul, &[x, w], &[h]);
        b.add_node(OpKind::Add, &[h, bias], &[y]);
        b.mark_input(x);
        b.mark_output(y);

        let graph = b.finish().expect("valid graph");
        assert_eq!(graph.tensors().len(), 5);
        assert_eq!(graph.nodes().len(), 2);
        assert_eq!(graph.inputs(), &[x]);
        assert_eq!(graph.outputs(), &[y]);
        assert_eq!(graph.nodes()[1].op(), &OpKind::Add);
        assert_eq!(graph.nodes()[0].inputs(), &[x, w]);
        assert_eq!(graph.nodes()[0].outputs(), &[h]);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn dangling_node_input_is_rejected() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let y = b.add_tensor(desc("y"));
        // TensorId(42) does not exist in this graph (ids are crate-internal,
        // so a dangling id can only be fabricated here in unit tests or by
        // mixing ids across builders).
        b.add_node(OpKind::Add, &[x, TensorId(42)], &[y]);
        b.mark_output(y);
        assert_graph_validation_err(b.finish(), "tensor id 42");
    }

    #[test]
    fn duplicate_tensor_name_is_rejected() {
        let mut b = GraphBuilder::new();
        let a = b.add_tensor(desc("same"));
        let _dup = b.add_tensor(desc("same"));
        b.mark_output(a);
        assert_graph_validation_err(b.finish(), "duplicate tensor name `same`");
    }

    #[test]
    fn undefined_graph_output_is_rejected() {
        let mut b = GraphBuilder::new();
        let _x = b.add_tensor(desc("x"));
        b.mark_output(TensorId(9));
        assert_graph_validation_err(b.finish(), "graph output");
    }

    #[test]
    fn double_producer_is_rejected() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let y = b.add_tensor(desc("y"));
        b.add_node(OpKind::Softmax, &[x], &[y]);
        b.add_node(OpKind::Mul, &[x, x], &[y]);
        b.mark_output(y);
        assert_graph_validation_err(b.finish(), "produced by both node #0 and node #1");
    }

    #[test]
    fn dangling_node_output_is_rejected() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        // An out-of-range node OUTPUT id (only a dangling node input and a
        // dangling graph output are otherwise tested). This check_id call is
        // what keeps the later single-producer `producer[id.0]` indexing from
        // panicking out-of-bounds on an output id >= tensors.len().
        b.add_node(OpKind::Softmax, &[x], &[TensorId(99)]);
        match b.finish() {
            Err(VokraError::GraphValidation(msg)) => {
                assert!(
                    msg.contains("output"),
                    "message `{msg}` should mention the node output"
                );
                assert!(
                    msg.contains("tensor id 99"),
                    "message `{msg}` should name the out-of-range id"
                );
            }
            other => panic!("expected GraphValidation error, got {other:?}"),
        }
    }

    #[test]
    fn dangling_graph_input_is_rejected() {
        let mut b = GraphBuilder::new();
        let _x = b.add_tensor(desc("x"));
        // The graph-INPUT check_id site has its own distinct message; only the
        // graph-OUTPUT list was covered before.
        b.mark_input(TensorId(5));
        assert_graph_validation_err(b.finish(), "graph input");
    }

    #[test]
    fn dynamic_dim_tensor_validates() {
        // A graph whose I/O tensors carry a symbolic (variable-length) axis must
        // pass structural validation: element-count checks skip Dynamic axes
        // (M1-04 sub-part 2), so a graph with `[Dynamic, 80]` inputs is valid.
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(TensorDesc::from_dims(
            "x",
            DType::F32,
            [Dim::Dynamic, Dim::Fixed(80)],
        ));
        let y = b.add_tensor(TensorDesc::from_dims(
            "y",
            DType::F32,
            [Dim::Dynamic, Dim::Fixed(80)],
        ));
        b.add_node(OpKind::Softmax, &[x], &[y]);
        b.mark_input(x);
        b.mark_output(y);
        assert!(b.finish().is_ok());
    }

    #[test]
    fn static_overflowing_tensor_is_rejected() {
        // A fully-static shape whose element count overflows `usize` can never
        // be allocated → rejected. (A `Dim::Dynamic` axis would instead be
        // skipped, as `dynamic_dim_tensor_validates` shows.)
        let mut b = GraphBuilder::new();
        let _x = b.add_tensor(TensorDesc::new("x", DType::F32, [usize::MAX, 2]));
        assert_graph_validation_err(b.finish(), "overflows usize");
    }

    #[test]
    fn tensor_getter_is_none_when_out_of_range() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let y = b.add_tensor(desc("y"));
        b.mark_output(y);
        let graph = b.finish().expect("valid graph");
        // In-range ids resolve; id 2 is one past the last tensor (len == 2) and
        // returns None rather than panicking.
        assert!(graph.tensor(x).is_some());
        assert!(graph.tensor(y).is_some());
        assert!(graph.tensor(TensorId(2)).is_none());
    }

    #[test]
    fn topo_order_of_empty_and_linear_graphs() {
        // Empty graph → empty order.
        let empty = GraphBuilder::new().finish().expect("empty graph is valid");
        assert_eq!(empty.topo_order().unwrap(), Vec::<usize>::new());

        // Linear chain inserted in order: identity schedule.
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let h = b.add_tensor(desc("h"));
        let y = b.add_tensor(desc("y"));
        b.add_node(OpKind::Softmax, &[x], &[h]); // node 0
        b.add_node(OpKind::Softmax, &[h], &[y]); // node 1
        b.mark_output(y);
        let graph = b.finish().expect("valid graph");
        assert_eq!(graph.topo_order().unwrap(), vec![0, 1]);
    }

    #[test]
    fn topo_order_reorders_when_insertion_violates_dataflow() {
        // Insert the CONSUMER (node 0) before the PRODUCER (node 1): the only
        // valid schedule is [1, 0], so this proves the pass is not just echoing
        // insertion order.
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let t1 = b.add_tensor(desc("t1"));
        let t2 = b.add_tensor(desc("t2"));
        b.add_node(OpKind::Softmax, &[t1], &[t2]); // node 0: needs t1
        b.add_node(OpKind::Softmax, &[x], &[t1]); // node 1: produces t1
        b.mark_input(x);
        b.mark_output(t2);
        let graph = b.finish().expect("valid graph");
        assert_eq!(graph.topo_order().unwrap(), vec![1, 0]);
    }

    #[test]
    fn topo_order_handles_a_diamond() {
        // root → {left, right} → join. root is node 0; join must come last.
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let r = b.add_tensor(desc("r"));
        let l = b.add_tensor(desc("l"));
        let rt = b.add_tensor(desc("rt"));
        let j = b.add_tensor(desc("j"));
        b.add_node(OpKind::Softmax, &[x], &[r]); // 0: root
        b.add_node(OpKind::Softmax, &[r], &[l]); // 1: left
        b.add_node(OpKind::Softmax, &[r], &[rt]); // 2: right
        b.add_node(OpKind::Add, &[l, rt], &[j]); // 3: join
        b.mark_input(x);
        b.mark_output(j);
        let graph = b.finish().expect("valid graph");

        let order = graph.topo_order().unwrap();
        let pos = |n: usize| order.iter().position(|&x| x == n).unwrap();
        assert_eq!(order.len(), 4);
        assert!(pos(0) < pos(1) && pos(0) < pos(2));
        assert!(pos(1) < pos(3) && pos(2) < pos(3));
    }

    #[test]
    fn topo_order_detects_a_cycle() {
        // node0: ta → tb, node1: tb → ta. Structurally valid (each tensor has a
        // single producer, ids in range) but has no topological order.
        let mut b = GraphBuilder::new();
        let ta = b.add_tensor(desc("ta"));
        let tb = b.add_tensor(desc("tb"));
        b.add_node(OpKind::Softmax, &[ta], &[tb]);
        b.add_node(OpKind::Softmax, &[tb], &[ta]);
        let graph = b
            .finish()
            .expect("structurally valid (validate() ignores cycles)");

        match graph.topo_order() {
            Err(VokraError::GraphValidation(msg)) => assert!(msg.contains("cycle")),
            other => panic!("expected a cycle GraphValidation error, got {other:?}"),
        }
    }

    #[test]
    fn topo_order_detects_a_self_cycle() {
        // A node that reads its own output is a one-node cycle.
        let mut b = GraphBuilder::new();
        let t = b.add_tensor(desc("t"));
        b.add_node(OpKind::Softmax, &[t], &[t]);
        let graph = b.finish().expect("structurally valid");
        assert!(matches!(
            graph.topo_order(),
            Err(VokraError::GraphValidation(_))
        ));
    }

    // ---- M2-04-T02: OpKind::Fused(FusedOp) + AudioGraph::rewrite_with ----
    //
    // These tests exercise the two additions this ticket makes:
    //  (a) `OpKind::Fused(FusedOp)` is a valid, structurally-checked variant
    //      of the `#[non_exhaustive]` op enum — `validate()` walks its
    //      inputs / outputs uniformly, no per-variant shape check;
    //  (b) `AudioGraph::rewrite_with` swaps a set of source nodes for one
    //      fused node in-place, and is the *only* mutation entry point
    //      outside `GraphBuilder`.

    fn logmel_fused_op() -> FusedOp {
        FusedOp::LogMel {
            stft: StftAttrs::new(400, 160),
            mel: MelAttrs::new(16_000, 400, 80),
        }
    }

    #[test]
    fn validate_accepts_fused_logmel_variant() {
        // Structural check: a graph containing `OpKind::Fused(FusedOp::LogMel)`
        // — the M2-04-T02 variant added to the `#[non_exhaustive]` op enum —
        // must pass `validate()` just like any other op. Shape checks are
        // backend-deferred; this only proves the tensor-id walk handles the
        // new variant.
        let mut b = GraphBuilder::new();
        let pcm = b.add_tensor(desc("pcm"));
        let mel = b.add_tensor(desc("mel"));
        b.add_node(OpKind::Fused(logmel_fused_op()), &[pcm], &[mel]);
        b.mark_input(pcm);
        b.mark_output(mel);
        let graph = b.finish().expect("Fused(LogMel) is structurally valid");
        assert_eq!(graph.nodes().len(), 1);
        assert!(matches!(
            graph.nodes()[0].op(),
            OpKind::Fused(FusedOp::LogMel { .. })
        ));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn validate_rejects_fused_variant_with_dangling_input() {
        // Structural checks apply uniformly. A `Fused` node with an
        // out-of-range input id must still be rejected by `validate()` — the
        // new variant is *not* a bypass of the tensor-id walk.
        let mut b = GraphBuilder::new();
        let out = b.add_tensor(desc("out"));
        b.add_node(OpKind::Fused(logmel_fused_op()), &[TensorId(99)], &[out]);
        b.mark_output(out);
        assert_graph_validation_err(b.finish(), "tensor id 99");
    }

    #[test]
    fn rewrite_with_swaps_base_ops_for_fused_node() {
        // The `pub(in crate::ir)` mutator drops the source nodes and appends
        // the fused replacement. This test lives in `crate::ir::graph`, i.e.
        // *inside* the `ir` module, so calling `rewrite_with` compiles — an
        // external caller outside `ir::` would fail to compile, which is
        // exactly the R2 mutation-restriction invariant.
        let mut b = GraphBuilder::new();
        let pcm = b.add_tensor(desc("pcm"));
        let spec = b.add_tensor(desc("spec"));
        let mel = b.add_tensor(desc("mel"));
        let fused_out = b.add_tensor(desc("fused_out"));
        b.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[pcm], &[spec]); // 0
        b.add_node(
            OpKind::MelFilterbank(MelAttrs::new(16_000, 400, 80)),
            &[spec],
            &[mel],
        ); // 1
        b.mark_input(pcm);
        b.mark_output(fused_out);
        let mut graph = b.finish().expect("valid base graph");
        assert_eq!(graph.nodes().len(), 2);

        // Apply one rewrite: drop nodes {0,1} (Stft + MelFilterbank), insert
        // one fused LogMel replacement `pcm → fused_out`. `tensor_remap` is
        // empty here — the rewrite produces the exact same graph-output
        // tensor id, so no downstream consumer needs redirecting.
        let rewrite = FusionRewrite {
            removed_nodes: vec![0, 1],
            inserted_node: Node {
                op: OpKind::Fused(logmel_fused_op()),
                inputs: vec![pcm],
                outputs: vec![fused_out],
            },
            tensor_remap: std::collections::HashMap::new(),
        };
        graph.rewrite_with(vec![rewrite]);

        assert_eq!(graph.nodes().len(), 1);
        assert!(matches!(
            graph.nodes()[0].op(),
            OpKind::Fused(FusedOp::LogMel { .. })
        ));
        assert_eq!(graph.nodes()[0].inputs(), &[pcm]);
        assert_eq!(graph.nodes()[0].outputs(), &[fused_out]);
    }

    #[test]
    fn rewrite_with_is_a_no_op_when_given_no_rewrites() {
        let mut b = GraphBuilder::new();
        let x = b.add_tensor(desc("x"));
        let y = b.add_tensor(desc("y"));
        b.add_node(OpKind::Softmax, &[x], &[y]);
        b.mark_output(y);
        let mut graph = b.finish().expect("valid");
        let before = graph.nodes().len();
        graph.rewrite_with(Vec::new());
        assert_eq!(graph.nodes().len(), before);
    }
}
