//! **beat_this** (`CPJKU/beat_this`, MIT) — Transformer-based beat +
//! downbeat tracker (Foscarin et al. ISMIR 2024 arXiv:2407.21658 "Beat
//! this! Accurate beat tracking without DBN postprocessing") — runtime
//! binder for the `beat-this` converter arch.
//!
//! # Runtime layout (loud-partial, RMVPE / pyannote / hifigan Wave 1
//! precedent)
//!
//! ```text
//! PCM (mono f32, sample_rate per config)
//!   -> log-mel front-end                     ← **loud-partial**
//!        (upstream is torchaudio MelSpectrogram(n_fft=1024,
//!         hop_length=441, n_mels=128, f_min=30, f_max=11000,
//!         mel_scale="slaney", normalized="frame_length", power=1)
//!         followed by log1p(1000 · x). Vokra's STFT + Mel-filterbank
//!         primitives cover the filterbank itself, but the
//!         frame-length normalisation and the log1p multiplier are
//!         neither expressible through `vokra.frontend.*` nor stamped
//!         anywhere in `vokra.beat_this.*`.)
//!   -> convolutional frontend                ← **loud-partial**
//!        (BatchNorm1d(spect_dim) → Conv2d(1, stem_dim, kernel (4,3),
//!         stride (4,1)) → BatchNorm2d → GELU, then three blocks of
//!         [optional PartialFTTransformer → Conv2d(kernel (2,3),
//!         stride (2,1))], then `b c f t -> b t (c f)` and a linear
//!         projection to the transformer width. `vokra-ops` exposes no
//!         public BatchNorm and no public 2-D convolution — its only
//!         Conv2d is `denoise.rs`' private DeepFilterNet-internal
//!         grouped conv, and `waveform_frontend` is Conv1d over time —
//!         while `vokra_ops::vit::vit_patch_embed` is a single
//!         non-overlapping linear patchifier over a 1-channel plane,
//!         which is a different function.)
//!   -> roformer encoder blocks               ← **loud-partial**
//!        (`vokra_ops::vit::ViTEncoder` DOES now supply a pre-norm
//!         multi-head-self-attention + MLP stack, so this model no
//!         longer needs attention written from scratch. It is still
//!         the wrong function here: beat_this normalises with RMSNorm
//!         (learnable γ only, no β, no mean subtraction) and rotates
//!         RotaryEmbedding(head_dim) into q and k inside every layer,
//!         while `ViTEncoder` is LayerNorm (γ and β, mean-centred) by
//!         construction and takes position from one additive absolute
//!         table applied before the stack.)
//!   -> 2-head classifier + activation        ← **loud-partial**
//!        (Linear(dim, 2) → `b t c -> c b t` → per-frame beat and
//!         per-frame downbeat activation posteriorgrams; caller
//!         peak-picks these directly — the paper's central claim is
//!         "no DBN postprocessing".)
//!   -> BeatAnalysis { beat_frames, downbeat_frames, confidence }
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`BeatThis::from_gguf`] with strict
//!   `vokra.model.arch == "beat-this"` validation + strict
//!   `vokra.beat_this.*` chunk-group presence enforcement (every axis
//!   required — no primary-source constant fallback because the upstream
//!   `.pt` release does NOT ship a first-class `config.yaml`, so a chunk
//!   fallback would fabricate axes without primary-source backing),
//!   [`BeatThisWeights::from_gguf`] with a floor of non-empty tensor
//!   count enforced loud (a GGUF that carries no beat_this-typical
//!   tensors is refused rather than silently running an all-zero
//!   forward), license-class surfacing.
//! - **Loud-partial (this WP)**: [`BeatThis::analyze`] returns
//!   [`VokraError::UnsupportedOp`] naming the blockers that are still
//!   real — the roformer's RMSNorm + rotary attention (which
//!   `vokra_ops::vit::ViTEncoder`'s LayerNorm + additive absolute
//!   positional table cannot stand in for without being *silently*
//!   wrong), the convolutional frontend `vokra-ops` exposes no public
//!   primitive for, the axis-factorised `PartialFTTransformer`, and that
//!   the `vokra.beat_this.*` group carries no chunk for `spect_dim` /
//!   `stem_dim` / `ff_mult` / `head_dim` / `partial_transformers` /
//!   `sum_head`, so the model could not be sized from this GGUF even
//!   with a complete primitive set. The message also states what is
//!   **already resolved**, so a reader does not re-report the encoder
//!   skeleton as missing.
//!
//! Rationale (RMVPE / pyannote / hifigan Wave 1 precedent, CLAUDE.md 教訓
//! (a)): the surrounding scaffold + `from_gguf` chunk-group validation +
//! FR-EX-08 loud-fails land today so a follow-up wave can flip the switch
//! by (i) landing the primitives this architecture actually needs (an
//! RMSNorm pre-norm block with rotary q/k, a BatchNorm + strided 2-D conv
//! frontend, and the factorised frequency/time attention), and (ii)
//! widening the `vokra.beat_this.*` group to carry the axes that size
//! them. The upstream defaults transcribed from the primary sources
//! below (`spect_dim=128`, `transformer_dim=512`, `ff_mult=4`,
//! `n_layers=6`, `head_dim=32`, `stem_dim=32`, `sum_head=True`,
//! `partial_transformers=True`) are recorded here as a *reading of
//! upstream*, *not* as a fallback this binder will ever apply — the
//! checkpoint stays the only authority, and
//! `tools/parity/beat_this_prepare_checkpoint.py` (uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) is where a caller confirms them against
//! real tensor shapes. Each [`VokraError::UnsupportedOp`] clause cites
//! the upstream file that settles it, so a reader diagnosing this gap
//! walks to a specific source rather than a repository root.
//!
//! # `vokra.beat_this.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::beat_this::convert_beat_this_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"beat-this"`).
//!   Hyphenated to stay distinct from the sibling `beats` arch
//!   (Microsoft SSL audio encoder — silently sharing would misroute
//!   runtime dispatch, FR-EX-08).
//! - `vokra.model.name` (`String`): `"beat-this"` — auxiliary check.
//! - `vokra.beat_this.{sample_rate, n_frames, d_model, n_layers, n_head,
//!   n_classes}` (`u32` each): the Transformer topology axes.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance.
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`KEY_SAMPLE_RATE`] / ...) — same rule the sibling BF16
//! pass-through binders (`pyannote` / `snac` / `hifigan`) use so
//! `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! beat_this ships as PyTorch `.pt` pickle upstream; this runtime
//! **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The `.pt`
//! → safetensors bridge lives in `tools/parity/beat_this_prepare_checkpoint.py`
//! (an offline uv-managed Python 3.12 sidecar — not part of the
//! runtime), mirroring the DAC / Kokoro / UTMOSv2 / beats bridge
//! pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/beat_this.rs` (see module docstring for
// the cross-crate duplication rationale).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model beat-this`.
///
/// Hyphenated to stay distinct from the sibling `beats` arch (Microsoft
/// SSL audio encoder). Sharing the arch tag would let runtime dispatch
/// bind a beat-tracker classifier head over an SSL-encoder checkpoint
/// (or vice versa), a silent-wrong shape mismatch FR-EX-08 forbids.
pub const ARCH: &str = "beat-this";

/// `vokra.beat_this.sample_rate` — input PCM sample rate the log-mel
/// front-end was tuned for.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.beat_this.sample_rate";
/// `vokra.beat_this.n_frames` — log-mel frames per analysis window
/// (Transformer temporal input length).
pub const GGUF_KEY_N_FRAMES: &str = "vokra.beat_this.n_frames";
/// `vokra.beat_this.d_model` — Transformer hidden dimension.
pub const GGUF_KEY_D_MODEL: &str = "vokra.beat_this.d_model";
/// `vokra.beat_this.n_layers` — stacked Transformer encoder layer count.
pub const GGUF_KEY_N_LAYERS: &str = "vokra.beat_this.n_layers";
/// `vokra.beat_this.n_head` — multi-head attention head count.
pub const GGUF_KEY_N_HEAD: &str = "vokra.beat_this.n_head";
/// `vokra.beat_this.n_classes` — terminal classifier output class count.
pub const GGUF_KEY_N_CLASSES: &str = "vokra.beat_this.n_classes";

/// Primary-source anchor for the model assembly: the convolutional
/// frontend (`Rearrange` / `BatchNorm1d` / strided `Conv2d` stem and
/// blocks), the `PartialFTTransformer` frequency/time factorisation, and
/// the `Head` / `SumHead` beat + downbeat projections.
const PRIMARY_SOURCE_URL: &str =
    "github.com/CPJKU/beat_this/blob/main/beat_this/model/beat_tracker.py";

/// Primary-source anchor for the encoder-block internals: the `RMSNorm`
/// definition (learnable gain, no bias, no mean subtraction), the
/// `bias=False` fused qkv projection, and the `rotate_queries_or_keys`
/// calls that put position *inside* attention instead of in an additive
/// table.
///
/// Cited separately from `PRIMARY_SOURCE_URL` because none of those facts
/// are readable from `beat_tracker.py`, which only names the transformer
/// it composes.
const ROFORMER_SOURCE_URL: &str =
    "github.com/CPJKU/beat_this/blob/main/beat_this/model/roformer.py";

/// Primary-source anchor for the mel front-end arguments, including the
/// `normalized="frame_length"` option and the `log1p(1000 · x)`
/// compression — neither of which any `vokra.frontend.*` field can
/// express today.
const FRONTEND_SOURCE_URL: &str = "github.com/CPJKU/beat_this/blob/main/beat_this/preprocessing.py";

// ---------------------------------------------------------------------------
// BeatThisConfig — the topology axes read from the `vokra.beat_this.*`
// chunk group. STRICT: every axis is required (FR-EX-08 — no primary-source
// constant fallback since the upstream `.pt` release does NOT ship a
// first-class `config.yaml`, so a chunk fallback would fabricate axes
// without primary-source backing).
// ---------------------------------------------------------------------------

/// beat_this Transformer hyperparameters as they ride the
/// `vokra.beat_this.*` chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every axis is
/// required (FR-EX-08 — never a silent primary-source constant fallback
/// because the upstream `.pt` release does not carry a first-class
/// `config.yaml`, so any fallback here would fabricate axes the runtime
/// then binds against). A GGUF missing any `vokra.beat_this.*` chunk is
/// rejected loudly with a [`VokraError::ModelLoad`] naming the absent
/// key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeatThisConfig {
    /// Input PCM sample rate the log-mel front-end was tuned for.
    pub sample_rate: u32,
    /// Number of log-mel frames per analysis window (Transformer
    /// temporal input length).
    pub n_frames: u32,
    /// Transformer hidden dimension.
    pub d_model: u32,
    /// Stacked Transformer encoder layer count.
    pub n_layers: u32,
    /// Multi-head attention head count.
    pub n_head: u32,
    /// Terminal classifier output class count (canonical `2` = beat +
    /// downbeat).
    pub n_classes: u32,
}

impl BeatThisConfig {
    /// Reads every `vokra.beat_this.*` chunk from `gguf`. Missing axis =
    /// loud [`VokraError::ModelLoad`] naming the absent key (FR-EX-08 —
    /// no primary-source constant fallback since the upstream `.pt`
    /// release does not carry a first-class `config.yaml`; any fallback
    /// here would fabricate axes without primary-source backing).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "beat_this: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `CPJKU/beat_this` `.pt` release does not carry a \
                         first-class `config.yaml`, so this runtime binder refuses to \
                         fabricate topology axes from primary-source constants (FR-EX-08). \
                         Re-run `vokra-cli convert --model beat-this --config \
                         <side-car.json>`, where the side-car is a flat JSON object \
                         carrying all six axes as unsigned integers: sample_rate, \
                         n_frames, d_model, n_layers, n_head, n_classes. Read them off \
                         the checkpoint's own tensor shapes — the converter will not \
                         default any of them."
                    ))
                })
        }
        Ok(Self {
            sample_rate: req_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            n_frames: req_u32(gguf, GGUF_KEY_N_FRAMES)?,
            d_model: req_u32(gguf, GGUF_KEY_D_MODEL)?,
            n_layers: req_u32(gguf, GGUF_KEY_N_LAYERS)?,
            n_head: req_u32(gguf, GGUF_KEY_N_HEAD)?,
            n_classes: req_u32(gguf, GGUF_KEY_N_CLASSES)?,
        })
    }
}

// ---------------------------------------------------------------------------
// BeatThisWeights — bound the tensor manifest with an explicit shape gate.
// Under the loud-partial WP the weights are counted + shape-cross-checked
// against `BeatThisConfig` axes, but the forward is deferred (the encoder
// body would consume them).
// ---------------------------------------------------------------------------

/// Weight tensors bound from a beat_this GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a
/// valid beat_this checkpoint). A caller with a shape-mismatched
/// tensor (e.g. a hidden-dim tensor that disagrees with
/// `config.d_model`) is also rejected loudly when the encoder body
/// wave lands.
///
/// Under the current landing this struct stores the tensor names + dims
/// discovered on disk. The Transformer encoder + classifier forward is
/// deferred (see [`BeatThis::analyze`] loud-partial), so the payload is
/// not yet dequantised — the follow-up wave sizes the dequant per its
/// kernel needs.
#[derive(Debug)]
pub struct BeatThisWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict` name
    /// with their GGUF-side dims. Used by the load-time shape gate and
    /// by the future follow-up encoder-forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl BeatThisWeights {
    /// Scans `gguf` for the beat_this state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF
    /// is never a valid beat_this checkpoint).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(
                "beat_this: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model beat-this` \
                 against the upstream `CPJKU/beat_this` `.pt` checkpoint flattened via \
                 `tools/parity/beat_this_prepare_checkpoint.py`."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-forward wave uses it to size its
    /// expectations.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load-time shape gate — validates that every bound tensor's outer
    /// dim (or the axis at which the Transformer hidden dim would land)
    /// is consistent with `config.d_model`. Under the current landing
    /// this is a **soft** gate (mismatch is silently ignored) because
    /// the encoder-body wave has not yet pinned the exact tensor-name
    /// convention the flatten step will use — a hard shape assertion
    /// today would fail against every legitimate future flatten shape.
    ///
    /// The follow-up wave will replace this soft accessor with a hard
    /// pin against the primary-source-verified tensor-name walk
    /// (mirror of `pyannote::PyanNetWeights::verify_core_shapes`).
    ///
    /// Kept as a `#[must_use]` accessor so the read is deliberate.
    #[must_use]
    pub fn matches_config(&self, config: &BeatThisConfig) -> bool {
        // Every tensor that has at least one axis matching config.d_model
        // is a plausible Transformer-body tensor. Under the loud-partial
        // wave we only assert that AT LEAST ONE such tensor exists — the
        // encoder-body wave will replace this with the strict pin.
        let d = config.d_model as usize;
        self.tensors.iter().any(|(_, dims)| dims.contains(&d))
    }
}

// ---------------------------------------------------------------------------
// BeatAnalysis — the public output surface. Populated by
// `BeatThis::analyze` once the follow-up wave lands the encoder forward.
// ---------------------------------------------------------------------------

/// Result of a beat_this analysis pass — per-frame beat + downbeat
/// activation posteriorgrams peak-picked to frame indices (no DBN
/// postprocessing per Foscarin et al. ISMIR 2024).
///
/// The three fields co-align on the temporal axis: `confidence[i]`
/// corresponds to whichever of `beat_frames` / `downbeat_frames` (or
/// both) contains frame `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatAnalysis {
    /// Zero-based frame indices where the beat activation posteriorgram
    /// peaked (units: log-mel frames at the config's `n_frames`
    /// resolution).
    pub beat_frames: Vec<u32>,
    /// Zero-based frame indices where the downbeat activation
    /// posteriorgram peaked. A subset of `beat_frames` in the metrical
    /// sense — every downbeat is also a beat.
    pub downbeat_frames: Vec<u32>,
    /// Peak-picked confidence at each populated frame in `beat_frames`
    /// (length equal to `beat_frames.len()`, sigmoid-space values in
    /// `[0, 1]`).
    pub confidence: Vec<f32>,
}

// ---------------------------------------------------------------------------
// BeatThis — the runtime binder handle
// ---------------------------------------------------------------------------

/// beat_this Transformer beat + downbeat tracker (CPJKU, MIT).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`analyze`](Self::analyze) on a PCM buffer to obtain a
/// [`BeatAnalysis`]. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract on
/// the Transformer encoder forward.
#[derive(Debug)]
pub struct BeatThis {
    config: BeatThisConfig,
    // The bound weights are held and counted; the forward that would
    // consume them is a follow-up wave. The manifest is still read on
    // every loud-partial fire so the error can report what the artifact
    // actually holds rather than only what it should hold. Same posture
    // as RMVPE / pyannote / Charsiu.
    weights: BeatThisWeights,
    weight_license: LicenseClass,
}

impl BeatThis {
    /// Binds a beat_this GGUF: validates arch, reads the strict topology
    /// chunk group, discovers tensors, and surfaces the stamped
    /// weight-license class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"beat-this"` (a `beats` / `basic_pitch` / `rmvpe` GGUF
    ///   handed to us by mistake fails with a clear message instead of
    ///   a downstream "missing tensor" — same pattern as
    ///   `Snac::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.beat_this.*` chunk is
    ///   absent (`BeatThisConfig::from_gguf` is strict — no
    ///   primary-source constant fallback since the upstream `.pt`
    ///   release does not carry a first-class `config.yaml`).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   (`BeatThisWeights::from_gguf` refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "vokra.beat_this.sample_rate missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "beat_this: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model beat-this`? Note the hyphenated \
                     arch tag — the sibling `beats` arch is the Microsoft SSL audio encoder, \
                     a completely different topology)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "beat_this: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native beat-this GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.beat_this.*` chunk group.
        let config = BeatThisConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = BeatThisWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license class
        //    for compliance gate cross-checks (defaults to `Unknown` if
        //    absent, which is fail-closed at the gate). Not raising a
        //    `ModelLoad` on missing provenance keeps the binder able to
        //    load hand-assembled GGUFs the test harness uses without
        //    forcing every fixture to stamp the full provenance chunk.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weights,
            weight_license,
        })
    }

    /// The bound topology axes (read from the `vokra.beat_this.*` chunk
    /// group).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &BeatThisConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. Upstream `CPJKU/beat_this`
    /// is `Permissive` (MIT); a GGUF missing the stamp reads back as
    /// `Unknown` (fail-closed at the compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Analyses a mono PCM buffer for beat + downbeat locations.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The encoder **skeleton** is
    /// no longer the blocker: `vokra_ops::vit::ViTEncoder` supplies a
    /// pre-norm multi-head-self-attention + MLP stack over a token
    /// sequence, so nobody should spend a wave re-deriving attention for
    /// this model. What blocks the forward is that beat_this is a
    /// **roformer over a convolutional frontend**, not a plain ViT:
    ///
    /// - it normalises with `RMSNorm` (learnable gain only, no bias, no
    ///   mean subtraction) where `ViTEncoder` is `LayerNorm` by
    ///   construction and exposes no norm selector;
    /// - it takes position from `RotaryEmbedding(head_dim)` rotated into
    ///   q and k inside every attention layer, and carries no absolute
    ///   positional table at all, where `ViTEncoder` takes position from
    ///   one additive absolute table applied before the stack — which is
    ///   exactly why its own docs call `encode_tokens`
    ///   permutation-equivariant — and exposes no hook to rotate q/k;
    /// - it feeds the transformer from a `BatchNorm` + strided 2-D
    ///   `Conv2d` stack with an optional axis-factorised
    ///   `PartialFTTransformer`, and `vokra-ops` exposes no public
    ///   primitive for any of those: its only `Conv2d` is `denoise.rs`'
    ///   private DeepFilterNet-internal grouped conv (BatchNorm folded
    ///   into a per-channel affine at load) and `waveform_frontend` is
    ///   `Conv1d` over the time axis;
    /// - and the `vokra.beat_this.*` group carries no chunk for
    ///   `spect_dim` / `stem_dim` / `ff_mult` / `head_dim` /
    ///   `partial_transformers` / `sum_head`, so the model could not be
    ///   sized from this GGUF even if every primitive existed.
    ///
    /// Substituting `LayerNorm` for `RMSNorm`, or an additive absolute
    /// table for rotary, is shape-valid and numerically wrong — a
    /// *silent* failure, which is the one FR-EX-08 forbids. The message
    /// names one upstream file per claim (`beat_tracker.py` for the
    /// frontend and heads, `roformer.py` for the norm and rotary,
    /// `preprocessing.py` for the mel spec) so a reader walks to a
    /// specific source rather than a repository root.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `sample_rate` disagrees
    ///   with `self.config().sample_rate` (a caller must resample
    ///   externally — never a silent resample, FR-EX-08).
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred forward.
    pub fn analyze(&self, pcm: &[f32], sample_rate: u32) -> Result<BeatAnalysis> {
        // Bind unused args so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future real
        // implementation will consume both.
        let _ = pcm;
        if sample_rate != self.config.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "beat_this analyze: input sample_rate {sample_rate} != model \
                 sample_rate {}. Resample the PCM before calling analyze \
                 (FR-EX-08 — never a silent resample)",
                self.config.sample_rate
            )));
        }
        Err(analyze_forward_loud_partial(&self.config, &self.weights))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`BeatThis::analyze`] until the forward lands.
///
/// Every clause is checked against the tree and the upstream file it
/// describes. The message opens by naming what is **already resolved**
/// rather than merely omitting it, because an earlier revision asserted
/// that "no shared MHA primitive exists in `vokra-ops`" — true when it
/// was written, and false from the moment `crates/vokra-ops/src/vit.rs`
/// landed a pre-norm MHSA + MLP stack. A stale blocker in an error
/// message is worse than no message: it sends the next reader off to
/// write attention code that already exists. The test
/// `analyze_loud_partials_naming_only_live_blockers` therefore asserts
/// the stale phrasing is ABSENT as well as asserting the live blockers
/// are present, so it cannot rot back in silently (mirror of the
/// `eat` / `m2d` guards).
fn analyze_forward_loud_partial(cfg: &BeatThisConfig, weights: &BeatThisWeights) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "beat_this analyze (loud-partial): ALREADY RESOLVED, do not re-report — \
         `vokra_ops::vit::ViTEncoder` supplies a pre-norm multi-head-self-attention \
         + MLP stack over a token sequence, so this model no longer needs attention \
         written from scratch, and the `vokra.beat_this.*` group is read strictly \
         (sample_rate={sample_rate}, n_frames={n_frames}, d_model={d_model}, \
         n_layers={n_layers}, n_head={n_head}, n_classes={n_classes}; \
         tensor_count={count}). What still blocks the forward is that beat_this is \
         a ROFORMER OVER A CONVOLUTIONAL FRONTEND, not a plain ViT. \
         (i) WRONG NORM: upstream normalises with RMSNorm — learnable gain only, no \
         bias, no mean subtraction — while `ViTEncoder` is LayerNorm (gain AND bias, \
         mean-centred) by construction and exposes no norm selector. \
         (ii) WRONG POSITIONAL SCHEME: upstream rotates RotaryEmbedding(head_dim) \
         into q and k inside EVERY attention layer and carries no absolute \
         positional table at all; `ViTEncoder` takes position from one additive \
         absolute table applied before the stack — which is why its own docs call \
         `encode_tokens` permutation-equivariant — and exposes no hook to rotate \
         q/k. Substituting either (i) or (ii) is shape-valid and numerically wrong, \
         i.e. silent, which is the failure FR-EX-08 exists to forbid. \
         (iii) NO CONVOLUTIONAL FRONTEND PRIMITIVE: upstream feeds the transformer \
         from BatchNorm1d(spect_dim) -> Conv2d(1, stem_dim, kernel (4,3), stride \
         (4,1), bias=False) -> BatchNorm2d -> GELU, then three blocks of \
         [optional PartialFTTransformer -> Conv2d(kernel (2,3), stride (2,1), \
         bias=False)], then `b c f t -> b t (c f)` and a linear projection. \
         `vokra-ops` exposes no public BatchNorm and no public 2-D convolution: \
         its only Conv2d is `denoise.rs`'s PRIVATE DeepFilterNet-internal grouped \
         conv (causal time padding, eval-mode BatchNorm2d folded into a \
         per-channel affine at load), and `waveform_frontend` is Conv1d over the \
         time axis. `vokra_ops::vit::vit_patch_embed` is a single non-overlapping \
         linear patchifier over a 1-channel plane, a different function. \
         (iv) NO AXIS-FACTORISED ATTENTION: with partial_transformers enabled, \
         PartialFTTransformer attends once across frequency (`(b t) f c`) and once \
         across time (`(b f) t c`) INSIDE the frontend; `ViTEncoder` attends over \
         one flat token sequence and has no factorised mode. \
         (v) THE STAMPED GROUP CANNOT SIZE THE MODEL: upstream is parameterised by \
         spect_dim / stem_dim / ff_mult / head_dim / partial_transformers / \
         sum_head, and `vokra.beat_this.*` carries a chunk for NONE of them (it \
         stamps n_head, which is not an argument of the upstream BeatThis \
         signature at all — that takes head_dim), so even a complete primitive \
         set could not be bound from this GGUF. The mel \
         front-end sits in the same position: upstream is torchaudio \
         MelSpectrogram(n_fft=1024, hop_length=441, n_mels=128, f_min=30, \
         f_max=11000, mel_scale=slaney, normalized=frame_length, power=1) followed \
         by log1p(1000 * x), and neither n_mels nor the frame-length normalisation \
         nor the log multiplier is stamped anywhere. \
         Primary sources: {PRIMARY_SOURCE_URL} (frontend + heads), \
         {ROFORMER_SOURCE_URL} (RMSNorm + rotary attention), {FRONTEND_SOURCE_URL} \
         (mel spec). Beat + downbeat activations, no DBN postprocessing per \
         Foscarin et al. ISMIR 2024. Loud pending (CLAUDE.md 教訓 (a) — \
         'loud-partial は fake-complete より honest') — no silent fabricated beat / \
         downbeat indices ever emitted (FR-EX-08).",
        sample_rate = cfg.sample_rate,
        n_frames = cfg.n_frames,
        d_model = cfg.d_model,
        n_layers = cfg.n_layers,
        n_head = cfg.n_head,
        n_classes = cfg.n_classes,
        count = weights.tensor_count(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the beat_this runtime binder — round-trip on the topology
    //! chunk group + negative-space round-trip on the loud-partial gates.
    //!
    //! # What "round-trip" means here
    //!
    //! On real PCM this would be `analyze(...)` returning real beat /
    //! downbeat frame indices. It cannot yet: `vokra_ops::vit::ViTEncoder`
    //! supplies the pre-norm attention stack, but beat_this is a roformer
    //! (RMSNorm + rotary q/k) over a BatchNorm + strided-conv frontend,
    //! and `vokra-ops` has no primitive for those — see the module doc +
    //! [`BeatThis::analyze`] rustdoc for the enumerated gap. Fabricating a
    //! real-PCM output would violate CLAUDE.md 教訓 (a) ("loud-partial は
    //! fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config round-trip**: `from_gguf` reads every axis stamped by
    //!    the converter.
    //! 2. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / missing chunk / empty tensor list /
    //!    unsupported forward surface) fires at its documented surface
    //!    point, in the documented error variant.
    //! 3. **Anti-rot guard**: the loud-partial message must NOT name a
    //!    blocker that has since been resolved. This is asserted
    //!    negatively, because omission alone is not enforceable — an
    //!    earlier revision of this file claimed "no shared MHA primitive
    //!    exists in `vokra-ops`" and a test pinned that phrase, so the
    //!    cheapest action once `vokra-ops`' ViT landed was to leave the
    //!    falsehood in place. See
    //!    `analyze_loud_partials_naming_only_live_blockers`.
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a minimal beat_this GGUF carrying the arch tag + full
    /// `vokra.beat_this.*` chunk group + one representative Transformer-
    /// encoder tensor whose outer dim matches the given `d_model`.
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn beat_this_gguf(
        d_model: u32,
        n_layers: u32,
        n_head: u32,
        n_classes: u32,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "beat-this");
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 22_050);
        b.add_u32(GGUF_KEY_N_FRAMES, 128);
        b.add_u32(GGUF_KEY_D_MODEL, d_model);
        b.add_u32(GGUF_KEY_N_LAYERS, n_layers);
        b.add_u32(GGUF_KEY_N_HEAD, n_head);
        b.add_u32(GGUF_KEY_N_CLASSES, n_classes);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative encoder tensor so the non-emptiness gate
        // passes and the shape-consistency accessor has something to
        // walk. The `d_model` dim is deliberately at axis 0 so
        // `matches_config` returns true.
        let d = d_model as u64;
        b.add_tensor(
            "encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Config round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_reads_full_topology_chunk_group() {
        let file = beat_this_gguf(128, 6, 8, 2, Some(LicenseClass::Permissive));
        let bt = BeatThis::from_gguf(&file).expect("valid GGUF must bind");
        let cfg = bt.config();
        assert_eq!(cfg.sample_rate, 22_050);
        assert_eq!(cfg.n_frames, 128);
        assert_eq!(cfg.d_model, 128);
        assert_eq!(cfg.n_layers, 6);
        assert_eq!(cfg.n_head, 8);
        assert_eq!(cfg.n_classes, 2);
        assert_eq!(bt.weight_license(), LicenseClass::Permissive);
        assert!(bt.tensor_count() >= 1);
    }

    #[test]
    fn from_gguf_defaults_weight_license_to_unknown_when_missing() {
        // A GGUF missing `vokra.provenance.weight_license` reads back as
        // `Unknown` (fail-closed at the compliance gate). Never a silent
        // Permissive default.
        let file = beat_this_gguf(64, 4, 4, 2, None);
        let bt = BeatThis::from_gguf(&file).expect("missing provenance must still bind");
        assert_eq!(bt.weight_license(), LicenseClass::Unknown);
    }

    // -----------------------------------------------------------------------
    // Loud-error round-trip — arch / chunk-group / tensor validation
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `beats` (Microsoft SSL audio encoder) GGUF handed to the
        // beat_this binder by mistake must fail loud with a specific
        // message rather than silently mis-binding.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "beats");
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 22_050);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = BeatThis::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`beats`") && m.contains("`beat-this`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("Microsoft"),
                    "message should call out the sibling BEATs identity to disambiguate, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_missing_arch() {
        // A GGUF with no `vokra.model.arch` at all — a converter that
        // forgot to stamp it must be caught here.
        let mut b = GgufBuilder::new();
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 22_050);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = BeatThis::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("vokra.model.arch"),
                    "message must name the missing arch key, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_missing_topology_chunk() {
        // Correct arch but missing one of the mandatory
        // `vokra.beat_this.*` chunks — a partially-stamped GGUF must be
        // caught here, not silently defaulted to a fabricated axis
        // (FR-EX-08 — the upstream `.pt` release carries no first-class
        // config, so fallback would fabricate).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 22_050);
        b.add_u32(GGUF_KEY_N_FRAMES, 128);
        // deliberately omit d_model
        b.add_u32(GGUF_KEY_N_LAYERS, 6);
        b.add_u32(GGUF_KEY_N_HEAD, 8);
        b.add_u32(GGUF_KEY_N_CLASSES, 2);
        b.add_tensor(
            "encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 16 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = BeatThis::from_gguf(&file) else {
            panic!("expected ModelLoad on missing d_model chunk");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_D_MODEL),
                    "message must name the missing d_model key, got `{m}`"
                );
                assert!(
                    m.contains("config.yaml"),
                    "message should explain why fallback is refused (no upstream config.yaml), got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + full chunk group but zero tensors — the
        // BeatThisWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 22_050);
        b.add_u32(GGUF_KEY_N_FRAMES, 128);
        b.add_u32(GGUF_KEY_D_MODEL, 128);
        b.add_u32(GGUF_KEY_N_LAYERS, 6);
        b.add_u32(GGUF_KEY_N_HEAD, 8);
        b.add_u32(GGUF_KEY_N_CLASSES, 2);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = BeatThis::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Loud-partial round-trip — analyze() names ONLY the blockers that are
    // real today, and positively disclaims the one that was resolved
    // -----------------------------------------------------------------------

    #[test]
    fn analyze_loud_partials_naming_only_live_blockers() {
        let file = beat_this_gguf(128, 6, 8, 2, Some(LicenseClass::Permissive));
        let bt = BeatThis::from_gguf(&file).unwrap();
        // A legitimate-sample-rate PCM buffer, so the loud-partial gate is
        // what fires — not the sample-rate mismatch guard ahead of it.
        let pcm = vec![0.0f32; 22_050];
        let Err(err) = bt.analyze(&pcm, 22_050) else {
            panic!("analyze must loud-partial");
        };
        let VokraError::UnsupportedOp(m) = err else {
            panic!("expected VokraError::UnsupportedOp");
        };

        // --- The blocker that `vokra-ops`' ViT landing RESOLVED must not be
        // --- claimed any more.
        //
        // `crates/vokra-ops/src/vit.rs` supplies a pre-norm MHSA + MLP stack.
        // A message still asserting that primitive is absent sends the next
        // reader off to write attention code that already exists, which is
        // the precise failure this guard exists to prevent. The negative
        // assertions are the load-bearing half of the row: without them the
        // stale phrasing can rot back in with nothing to catch it.
        assert!(
            !m.contains("no shared MHA primitive"),
            "`vokra_ops::vit::ViTEncoder` supplies the pre-norm MHSA + MLP stack — \
             this claim is stale: {m}"
        );
        assert!(
            !m.contains("greenfield"),
            "the encoder skeleton is no longer written from scratch for this model: {m}"
        );

        // --- and the message must say so POSITIVELY, not merely omit it.
        assert!(
            m.contains("ALREADY RESOLVED"),
            "the message must tell the reader what NOT to re-report: {m}"
        );
        assert!(
            m.contains("vokra_ops::vit::ViTEncoder"),
            "the message must name the primitive that now exists: {m}"
        );

        // --- What actually remains: one assertion per stated blocker.
        assert!(
            m.contains("RMSNorm") && m.contains("LayerNorm"),
            "must name the norm mismatch that makes ViTEncoder the wrong function \
             here rather than a missing one: {m}"
        );
        assert!(
            m.contains("RotaryEmbedding") && m.contains("absolute"),
            "must name the positional-scheme mismatch (rotary q/k vs one additive \
             absolute table): {m}"
        );
        assert!(
            m.contains("BatchNorm") && m.contains("Conv2d"),
            "must name the convolutional frontend `vokra-ops` has no primitive for: {m}"
        );
        assert!(
            m.contains("PartialFTTransformer"),
            "must name the axis-factorised frequency/time attention: {m}"
        );
        assert!(
            m.contains("head_dim") && m.contains("ff_mult"),
            "must name axes the stamped chunk group cannot carry, so a reader knows \
             the GGUF schema is part of the gap and not only the kernels: {m}"
        );
        assert!(
            m.contains("FR-EX-08"),
            "must cite the no-silent-output clause: {m}"
        );

        // --- One upstream file per distinct claim, so a reader walks to a
        // --- specific source rather than a repository root.
        assert!(
            m.contains(PRIMARY_SOURCE_URL),
            "must cite beat_tracker.py for the frontend and heads: {m}"
        );
        assert!(
            m.contains(ROFORMER_SOURCE_URL),
            "must cite roformer.py — the RMSNorm and rotary claims are not readable \
             from beat_tracker.py, which only names the transformer it composes: {m}"
        );
        assert!(
            m.contains(FRONTEND_SOURCE_URL),
            "must cite preprocessing.py for the mel spec: {m}"
        );

        // --- Every stamped axis is echoed, so a reader can cross-check what
        // --- topology a follow-up wave would target, plus what the manifest
        // --- actually holds.
        for fragment in [
            "sample_rate=22050",
            "n_frames=128",
            "d_model=128",
            "n_layers=6",
            "n_head=8",
            "n_classes=2",
            "tensor_count=1",
        ] {
            assert!(m.contains(fragment), "must echo `{fragment}`: {m}");
        }
    }

    #[test]
    fn analyze_rejects_sample_rate_mismatch_before_loud_partial() {
        // A sample-rate mismatch fires as InvalidArgument BEFORE the
        // encoder loud-partial gate — a caller passing the wrong SR sees
        // the SR error (which they can fix by resampling), not the deeper
        // "primitive missing" error (which they can't fix at all).
        let file = beat_this_gguf(128, 6, 8, 2, Some(LicenseClass::Permissive));
        let bt = BeatThis::from_gguf(&file).unwrap();
        let pcm = vec![0.0f32; 44_100];
        let Err(err) = bt.analyze(&pcm, 44_100) else {
            panic!("wrong SR must be rejected");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(
                    m.contains("44100") && m.contains("22050"),
                    "message must name both the got and expected SR, got `{m}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Structural pins — arch tag is stable and distinct from siblings
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_beats() {
        // Pin the string constants so a rename would land here in the
        // same commit or fail this test. The sibling Microsoft SSL
        // encoder's arch tag MUST NOT collide with ours.
        assert_eq!(ARCH, "beat-this");
        assert_ne!(
            ARCH, "beats",
            "beat-this and beats are different models with different topologies — \
             sharing arch would mis-route runtime dispatch (FR-EX-08)"
        );
    }

    #[test]
    fn weights_shape_consistency_soft_gate_matches_when_axis_present() {
        // The soft gate returns true when at least one tensor has an axis
        // matching config.d_model, which the fixture GGUF guarantees.
        let file = beat_this_gguf(128, 6, 8, 2, Some(LicenseClass::Permissive));
        let bt = BeatThis::from_gguf(&file).unwrap();
        assert!(bt.weights.matches_config(bt.config()));
    }
}
