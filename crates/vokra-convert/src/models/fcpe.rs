//! **FCPE** — Fast Context-based Pitch Estimator: safetensors → GGUF
//! conversion (M5-16 / FR-OP-83).
//!
//! Upstream: `CNChTu/FCPE` (MIT). This converter is the offline
//! `.safetensors → Vokra GGUF` half; the upstream release is a
//! torch-pickle `.pt`, so callers pre-flatten it to safetensors via
//! `tools/parity/fcpe_prepare_checkpoint.py` (the DFN3 / DAC / CSM
//! bridge pattern — no pickle ever enters the runtime, FR-LD-05).
//!
//! # Category / arch / provenance
//!
//! - `vokra.model.arch = "fcpe"`
//! - `vokra.model.name = "fcpe"`
//! - `vokra.model.category = "f0"` — pitch / F0 extractor family
//!   (distinct taxonomy from `codec` / `tts` / `asr` / `s2s`; the
//!   runtime dispatches on `arch`, this is a catalog tag).
//! - `vokra.provenance.upstream_hf = "CNChTu/FCPE"` — GitHub-only
//!   release; the string preserves the CC-verified upstream anchor
//!   even though there is no HF mirror.
//!
//! # License posture — MIT (Permissive)
//!
//! Default `LicenseClass::Permissive` (SPDX `mit`), CC-verified on
//! 2026-07-30 against GitHub `CNChTu/FCPE` `LICENSE` (`docs/license-
//! audit.md` §3.1). Callers who ship the weights under a distinct
//! SPDX id (e.g. a fine-tune redistribution) can override at the outer
//! `convert_file --license <spdx>` boundary (the Whisper / kokoro /
//! xcodec2 pattern).
//!
//! # BF16 pass-through (mirror of `neucodec` / `xcodec2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling codec / TTS
//! converters. No convert-time widening; the runtime widens BF16 →
//! f32 losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Every F32 /
//! F16 tensor passes through under its upstream safetensors name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **prep-script's canonical output names**
//! (Vokra-defined for FCPE — the offline
//! `fcpe_prepare_checkpoint.py` remaps upstream `torchfcpe.model.
//! CFNaiveMelPE` state-dict keys to the flat layout the runtime binds).
//! The runtime layout is documented in
//! `crates/vokra-models/src/f0/fcpe.rs` module docs.
//!
//! # `vokra.f0.fcpe.*` config chunk (added 2026-08-15)
//!
//! Until 2026-08-15 this converter stamped **none** of the fourteen
//! `vokra.f0.fcpe.*` axes the runtime reads, and the runtime answered
//! every absent key with a built-in default. An FCPE GGUF therefore
//! described no topology at all: the runtime simply assumed the released
//! `fcpe_c_v001` shape and ran. For any checkpoint that differed on an
//! axis with no tensor-shape cross-check — `n_layers` (an artifact with
//! *more* layers was silently truncated), `stem_groups`, `hop`, `n_fft`,
//! `sample_rate`, `fmin`, `fmax`, `confidence_threshold` — the forward
//! completed and the numbers were quietly wrong. `main.rs`'s verify arm
//! even claimed the chunk was "written by the model, not the converter";
//! nothing wrote it.
//!
//! Two halves now close that:
//!
//! 1. **Seven topology axes are derived from the checkpoint's own tensor
//!    shapes** and stamped unconditionally — `d_model`, `n_mels`,
//!    `stem_kernel` (from `input_stack.0.weight`), `ffn_dim`,
//!    `conv_kernel` (from layer 0), `n_layers` (by walking
//!    `net.encoder_layers.{i}.…`), `n_pitch_bins` (from
//!    `output_proj.weight`). These are read off the artifact, not
//!    assumed, so they are correct for any variant. Every cross-check the
//!    shapes permit is enforced here (see [`derive_topology`]) rather
//!    than deferred.
//! 2. **Seven front-end / decode axes are not in the checkpoint at all**
//!    — `hop`, `n_fft`, `sample_rate`, `fmin`, `fmax`, `stem_groups`,
//!    `confidence_threshold` live in upstream's Python config, and the
//!    prep script drops the one buffer (`cent_table`) that would have
//!    pinned `fmin` / `fmax`. They are stamped from the documented
//!    [`FCPE_V001_TOPOLOGY`] reference constants **only when the derived
//!    topology matches that reference exactly** — i.e. only when the
//!    artifact proves from its own shapes that it is the released
//!    `fcpe_c_v001` architecture. On any other topology they are
//!    withheld, because asserting a 16 kHz front-end onto a variant we
//!    cannot identify is exactly the invented axis this change removes.
//!
//! A withheld front-end means the emitted GGUF carries seven of fourteen
//! axes, and the runtime refuses to bind it (naming the first absent
//! key). That is the intended outcome: the weights are preserved and
//! correct, and the axes nobody can derive have to be supplied by whoever
//! knows the variant. The conversion note says so out loud.
//!
//! # Wiring
//!
//! CLI dispatch (`vokra-cli convert --model fcpe`) resolves to this
//! module through [`crate::ModelKind::Fcpe`]; the file-based public
//! entry point is [`convert_fcpe_file`].

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FCPE GGUFs. Distinct from every other F0
/// extractor family (`rmvpe` / `crepe`) — silently sharing an arch
/// would mis-route the runtime binder (they load different tensor
/// name sets + different topologies).
pub(crate) const ARCH: &str = "fcpe";
/// `vokra.model.name` value for the canonical FCPE GGUF.
pub(crate) const NAME: &str = "fcpe";

/// `vokra.model.category` value — FCPE is an **F0 / pitch extractor**
/// (FR-OP-83). Orthogonal to `arch` (the runtime dispatches on arch);
/// the category tag is a machine-readable catalog surface (see
/// `docs/license-audit.md` §3.1 for the tier-and-tag registry).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const MODEL_CATEGORY: &str = "f0";

/// Upstream release slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source` string. FCPE ships on GitHub only — the
/// slug preserves the CC-verified upstream anchor even though there is
/// no HF mirror.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const UPSTREAM_HF: &str = "CNChTu/FCPE";

/// Default upstream weight license — `mit`, per the CNChTu/FCPE LICENSE
/// file (CC-verified 2026-07-30; sign-off 2026-07-30 yousan =
/// ☑ Commercial, `docs/license-audit.md` §3.1).
const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source`. Short — the license machine class is
/// carried separately in the `vokra.provenance.weight_license` chunk.
const UPSTREAM_SOURCE: &str = "CNChTu/FCPE (Fast Context-based Pitch Estimator, Conformer + 360-bin log-freq classifier, MIT)";

// -- `vokra.f0.fcpe.*` schema keys -------------------------------------------
//
// Kept in lock-step with the runtime consumer
// `vokra_models::f0::fcpe::FcpeConfig::from_gguf`, which requires every one
// of these on a weight-carrying artifact. Adding a key here without adding
// it there (or vice versa) is caught by the round-trip test at the bottom of
// this file, which asserts the stamped set against a literal list.

/// GGUF key: encoder hidden width (derived from `input_stack.0.weight`).
pub(crate) const KEY_D_MODEL: &str = "vokra.f0.fcpe.d_model";
/// GGUF key: mel channel count (derived from `input_stack.0.weight`).
pub(crate) const KEY_N_MELS: &str = "vokra.f0.fcpe.n_mels";
/// GGUF key: input-stack Conv1d kernel (derived from `input_stack.0.weight`).
pub(crate) const KEY_STEM_KERNEL: &str = "vokra.f0.fcpe.stem_kernel";
/// GGUF key: pre-GLU pointwise expansion width (derived from layer 0's
/// `conformer.net.2.weight`).
pub(crate) const KEY_FFN_DIM: &str = "vokra.f0.fcpe.ffn_dim";
/// GGUF key: depthwise conv kernel (derived from layer 0's
/// `conformer.net.4.conv.weight`).
pub(crate) const KEY_CONV_KERNEL: &str = "vokra.f0.fcpe.conv_kernel";
/// GGUF key: encoder block count (derived by walking the
/// `net.encoder_layers.{i}.` tensor-name prefix).
pub(crate) const KEY_N_LAYERS: &str = "vokra.f0.fcpe.n_layers";
/// GGUF key: output class count (derived from `output_proj.weight`).
pub(crate) const KEY_N_PITCH_BINS: &str = "vokra.f0.fcpe.n_pitch_bins";

/// GGUF key: mel-frame hop in samples. Not derivable from any tensor.
pub(crate) const KEY_HOP: &str = "vokra.f0.fcpe.hop";
/// GGUF key: STFT FFT / window size. Not derivable from any tensor.
pub(crate) const KEY_N_FFT: &str = "vokra.f0.fcpe.n_fft";
/// GGUF key: expected PCM sample rate. Not derivable from any tensor.
pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.f0.fcpe.sample_rate";
/// GGUF key: lowest tracked pitch in Hz. Not derivable — the upstream
/// `cent_table` buffer that would pin it is dropped by the prep script.
pub(crate) const KEY_FMIN: &str = "vokra.f0.fcpe.fmin";
/// GGUF key: highest tracked pitch in Hz. Not derivable (see [`KEY_FMIN`]).
pub(crate) const KEY_FMAX: &str = "vokra.f0.fcpe.fmax";
/// GGUF key: input-stack GroupNorm group count. Not derivable — GroupNorm
/// gamma / beta are per-channel `[d_model]` whatever the group count is.
pub(crate) const KEY_STEM_GROUPS: &str = "vokra.f0.fcpe.stem_groups";
/// GGUF key: V/UV threshold on `max(sigmoid(logits))`. A decode-time knob,
/// not a checkpoint property.
pub(crate) const KEY_CONFIDENCE_THRESHOLD: &str = "vokra.f0.fcpe.confidence_threshold";

// -- Canonical tensor names the topology is read from ------------------------

/// First input-stack Conv1d weight — `[d_model, n_mels, stem_kernel]`.
const TENSOR_STEM_W: &str = "input_stack.0.weight";
/// Output projection weight — `[n_pitch_bins, d_model]` (weight-norm folded
/// at prep time).
const TENSOR_HEAD_W: &str = "output_proj.weight";

/// Upper bound on the encoder-layer walk. Purely a runaway guard: the walk
/// stops at the first absent index, so this only bounds a pathological
/// artifact that names ten thousand layers.
const MAX_ENCODER_LAYERS: u32 = 1024;

// -- Released `fcpe_c_v001` reference ----------------------------------------

/// The topology of the released `torchfcpe/assets/fcpe_c_v001.pt`, mirroring
/// the `V001_*` constants in `vokra_models::f0::fcpe`.
///
/// This is the *gate* on the seven front-end / decode axes below: a
/// checkpoint whose derived topology equals this one is the released
/// architecture, so its front-end is the released front-end. A checkpoint
/// that differs on any axis is a variant this converter cannot identify, and
/// stamping a front-end onto it would be a fabricated claim.
pub const FCPE_V001_TOPOLOGY: FcpeTopology = FcpeTopology {
    d_model: 512,
    n_mels: 128,
    stem_kernel: 3,
    ffn_dim: 2048,
    conv_kernel: 31,
    n_layers: 6,
    n_pitch_bins: 360,
};

/// `fcpe_c_v001` mel-frame hop, samples (10 ms at 16 kHz).
const V001_HOP: u32 = 160;
/// `fcpe_c_v001` STFT FFT / window size.
const V001_N_FFT: u32 = 1024;
/// `fcpe_c_v001` expected PCM sample rate.
const V001_SAMPLE_RATE: u32 = 16_000;
/// `fcpe_c_v001` lowest tracked pitch, Hz (C1 — the cent-grid anchor).
const V001_FMIN: f32 = 32.7;
/// `fcpe_c_v001` highest tracked pitch, Hz (~ B6).
const V001_FMAX: f32 = 1975.5;
/// `fcpe_c_v001` input-stack GroupNorm group count. Upstream writes the
/// literal `nn.GroupNorm(4, hidden_dims)` — a constant of the architecture,
/// not a constructor parameter, which is why it is asserted rather than
/// derived.
const V001_STEM_GROUPS: u32 = 4;
/// `fcpe_c_v001` V/UV threshold — the upstream public waveform-to-F0 wrapper
/// (`InferCFNaiveMelPE.forward` / `infer`) uses
/// `threshold=0.006`. The lower-level local decoder's standalone default is
/// 0.05, but it is overridden by the official public inference path.
const V001_CONFIDENCE_THRESHOLD: f32 = 0.006;

/// The count of `vokra.f0.fcpe.*` axes derived from tensor shapes.
const DERIVED_AXIS_COUNT: usize = 7;
/// The count of `vokra.f0.fcpe.*` axes asserted from the v001 reference.
const FRONT_END_AXIS_COUNT: usize = 7;

/// The seven FCPE topology axes this converter reads directly off a
/// checkpoint's tensor shapes.
///
/// Every field is measured, never assumed — which is what makes it safe to
/// compare against [`FCPE_V001_TOPOLOGY`] and decide whether the released
/// front-end constants may be asserted onto this artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FcpeTopology {
    /// Encoder hidden width — `input_stack.0.weight` dim 0.
    pub d_model: u32,
    /// Mel channel count — `input_stack.0.weight` dim 1.
    pub n_mels: u32,
    /// Input-stack Conv1d kernel — `input_stack.0.weight` dim 2.
    pub stem_kernel: u32,
    /// Pre-GLU pointwise expansion width — layer 0's
    /// `conformer.net.2.weight` dim 0. Halves to `ffn_dim / 2` after the GLU.
    pub ffn_dim: u32,
    /// Depthwise conv kernel — layer 0's `conformer.net.4.conv.weight` dim 2.
    pub conv_kernel: u32,
    /// Encoder block count — the number of consecutive
    /// `net.encoder_layers.{i}.conformer.net.0.weight` tensors from 0.
    pub n_layers: u32,
    /// Output class count — `output_proj.weight` dim 0.
    pub n_pitch_bins: u32,
}

impl FcpeTopology {
    /// Whether the seven front-end / decode axes may be asserted onto an
    /// artifact with this topology.
    ///
    /// True exactly when the topology is the released
    /// [`FCPE_V001_TOPOLOGY`]. Nothing weaker would do: the front-end
    /// constants are a property of that one release, and a variant that
    /// happens to share (say) `n_pitch_bins` tells us nothing about its
    /// sample rate.
    pub fn front_end_is_assertable(&self) -> bool {
        *self == FCPE_V001_TOPOLOGY
    }
}

/// Looks up a tensor's shape by name, or `None` when it is absent.
fn shape_of<'a>(st: &'a SafetensorsFile, name: &str) -> Option<&'a [u64]> {
    st.tensors()
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.shape.as_slice())
}

/// Reads one dimension as a non-zero `u32`, refusing a zero or oversized
/// extent rather than letting it collapse a downstream product.
fn axis(value: u64, tensor: &str, which: &str) -> Result<u32, ConvertError> {
    if value == 0 {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{tensor}` has a zero-length {which} axis — a checkpoint with an \
             empty dimension cannot describe a runnable topology (FR-EX-08)"
        )));
    }
    u32::try_from(value).map_err(|_| {
        ConvertError::Parse(format!(
            "fcpe: `{tensor}` {which} axis = {value} does not fit in u32, so it cannot \
             be stamped into a `vokra.f0.fcpe.*` chunk"
        ))
    })
}

/// Asserts a tensor's rank, naming the shape it actually had.
fn expect_rank(shape: &[u64], want: usize, tensor: &str, layout: &str) -> Result<(), ConvertError> {
    if shape.len() == want {
        return Ok(());
    }
    Err(ConvertError::Parse(format!(
        "fcpe: `{tensor}` is rank {} (shape {shape:?}), expected rank {want} {layout} — \
         this is not the FCPE tensor layout the topology is derived from",
        shape.len(),
    )))
}

/// Derives the seven topology axes from a parsed checkpoint's tensor shapes.
///
/// Returns `Ok(None)` when neither canonical trigger tensor
/// (`input_stack.0.weight`, `output_proj.weight`) is present — the input is
/// not an FCPE checkpoint, and the caller stamps no `vokra.f0.fcpe.*` chunk.
///
/// Every cross-check the shapes make available is enforced here rather than
/// left to the runtime binder: the binder only sees flat element counts, so
/// it cannot tell `[d_model=256, n_mels=256]` from `[d_model=512,
/// n_mels=128]`. Here the axes are separable, so a checkpoint that is
/// internally inconsistent is caught at conversion time, once, instead of
/// producing an artifact that binds and computes the wrong thing.
///
/// # Errors
///
/// [`ConvertError::Parse`] when exactly one trigger tensor is present (half
/// an FCPE), when a canonical tensor has the wrong rank, when the shapes
/// disagree with each other (e.g. layer 0's pointwise conv does not project
/// from `d_model`), when the encoder layers are not mutually uniform, or
/// when an axis is zero / oversized / violates a shape invariant the forward
/// depends on (`ffn_dim` even, both kernels odd).
fn derive_topology(st: &SafetensorsFile) -> Result<Option<FcpeTopology>, ConvertError> {
    let (stem, head) = match (shape_of(st, TENSOR_STEM_W), shape_of(st, TENSOR_HEAD_W)) {
        (None, None) => return Ok(None),
        (Some(stem), Some(head)) => (stem, head),
        (stem, head) => {
            let stem_present = stem.is_some();
            let head_present = head.is_some();
            return Err(ConvertError::Parse(format!(
                "fcpe: partially populated checkpoint (`{TENSOR_STEM_W}` present={stem_present}, \
                 `{TENSOR_HEAD_W}` present={head_present}) — refusing to derive a topology from \
                 half an FCPE (FR-EX-08)"
            )));
        }
    };

    // Stem: [d_model, n_mels, stem_kernel].
    expect_rank(stem, 3, TENSOR_STEM_W, "[d_model, n_mels, stem_kernel]")?;
    let d_model = axis(stem[0], TENSOR_STEM_W, "d_model")?;
    let n_mels = axis(stem[1], TENSOR_STEM_W, "n_mels")?;
    let stem_kernel = axis(stem[2], TENSOR_STEM_W, "stem_kernel")?;
    if stem_kernel % 2 == 0 {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{TENSOR_STEM_W}` kernel {stem_kernel} is even; the forward's same-padding \
             (`pad = k / 2`) is only symmetric for an odd kernel"
        )));
    }

    // Head: [n_pitch_bins, d_model].
    expect_rank(head, 2, TENSOR_HEAD_W, "[n_pitch_bins, d_model]")?;
    let n_pitch_bins = axis(head[0], TENSOR_HEAD_W, "n_pitch_bins")?;
    if head[1] != u64::from(d_model) {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{TENSOR_HEAD_W}` projects from {} channels but `{TENSOR_STEM_W}` produces \
             d_model={d_model} — the checkpoint is internally inconsistent",
            head[1],
        )));
    }

    // Encoder depth: walk consecutive layer indices from 0.
    let mut n_layers = 0u32;
    while shape_of(st, &layer_tensor(n_layers, "conformer.net.0.weight")).is_some() {
        n_layers += 1;
        if n_layers > MAX_ENCODER_LAYERS {
            return Err(ConvertError::Parse(format!(
                "fcpe: more than {MAX_ENCODER_LAYERS} encoder layers found; refusing to walk \
                 further (a checkpoint this deep is not an FCPE)"
            )));
        }
    }
    if n_layers == 0 {
        return Err(ConvertError::Parse(format!(
            "fcpe: the checkpoint carries `{TENSOR_STEM_W}` and `{TENSOR_HEAD_W}` but no \
             `net.encoder_layers.0.conformer.net.0.weight`, so it has zero encoder blocks — \
             the mel would go straight from the input stack to the output head. Refusing to \
             stamp a topology for a forward that cannot be the model (FR-EX-08)"
        )));
    }

    // Layer 0 pointwise expansion: [ffn_dim, d_model, 1].
    let pw1_name = layer_tensor(0, "conformer.net.2.weight");
    let pw1 = require_layer_tensor(st, &pw1_name)?;
    expect_rank(pw1, 3, &pw1_name, "[ffn_dim, d_model, 1]")?;
    let ffn_dim = axis(pw1[0], &pw1_name, "ffn_dim")?;
    if pw1[1] != u64::from(d_model) {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{pw1_name}` projects from {} channels but d_model={d_model}",
            pw1[1],
        )));
    }
    if ffn_dim % 2 != 0 {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{pw1_name}` ffn_dim={ffn_dim} is odd, but the forward's `GLU(dim=1)` \
             splits that axis in half"
        )));
    }
    let inner_dim = u64::from(ffn_dim / 2);

    // Layer 0 depthwise conv: [ffn_dim / 2, 1, conv_kernel].
    let dw_name = layer_tensor(0, "conformer.net.4.conv.weight");
    let dw = require_layer_tensor(st, &dw_name)?;
    expect_rank(dw, 3, &dw_name, "[ffn_dim / 2, 1, conv_kernel]")?;
    if dw[0] != inner_dim {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{dw_name}` has {} channels but the post-GLU stream is {inner_dim} wide",
            dw[0],
        )));
    }
    let conv_kernel = axis(dw[2], &dw_name, "conv_kernel")?;
    if conv_kernel % 2 == 0 {
        return Err(ConvertError::Parse(format!(
            "fcpe: `{dw_name}` kernel {conv_kernel} is even; the forward's same-padding \
             (`pad = k / 2`) is only symmetric for an odd kernel"
        )));
    }

    // Uniformity: the runtime binds every layer with layer 0's shapes, so a
    // heterogeneous stack would be mis-bound (or silently truncated) rather
    // than refused. Check it once here, where both shapes are visible.
    for i in 1..n_layers {
        for (tail, want) in [
            ("conformer.net.2.weight", pw1),
            ("conformer.net.4.conv.weight", dw),
        ] {
            let name = layer_tensor(i, tail);
            let got = require_layer_tensor(st, &name)?;
            if got != want {
                return Err(ConvertError::Parse(format!(
                    "fcpe: layer {i} tensor `{name}` has shape {got:?} but layer 0's is \
                     {want:?} — the runtime binds every layer with layer 0's shapes, so a \
                     non-uniform encoder cannot be represented (FR-EX-08)"
                )));
            }
        }
    }

    Ok(Some(FcpeTopology {
        d_model,
        n_mels,
        stem_kernel,
        ffn_dim,
        conv_kernel,
        n_layers,
        n_pitch_bins,
    }))
}

/// Builds a per-layer tensor name (`net.encoder_layers.{i}.{tail}`).
fn layer_tensor(i: u32, tail: &str) -> String {
    format!("net.encoder_layers.{i}.{tail}")
}

/// Looks up a per-layer tensor that the layer walk proved should exist,
/// failing loudly on the ragged case (layer `i`'s LayerNorm present but its
/// conv weights absent).
fn require_layer_tensor<'a>(
    st: &'a SafetensorsFile,
    name: &str,
) -> Result<&'a [u64], ConvertError> {
    shape_of(st, name).ok_or_else(|| {
        ConvertError::Parse(format!(
            "fcpe: `{name}` is missing, but the encoder-layer walk found this layer's \
             LayerNorm — the checkpoint is a ragged FCPE (FR-EX-08)"
        ))
    })
}

/// Outcome of an FCPE conversion (additive counters — a non-zero value
/// on any field is a positive report; a zero `written` value means the
/// input safetensors carried no float tensors and the runtime will
/// refuse to bind any weights, FR-EX-08).
#[derive(Debug, Default)]
pub struct FcpeReport {
    /// Total tensors observed in the input safetensors.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (observability counter — the ADR pattern shared with neucodec /
    /// xcodec2 so a latent silent-widen cannot slip in undetected).
    pub bf16_passthrough: usize,
    /// The topology read off the checkpoint's tensor shapes, or `None`
    /// when neither canonical trigger tensor was present (the input is not
    /// an FCPE checkpoint, so no `vokra.f0.fcpe.*` chunk was stamped).
    pub topology: Option<FcpeTopology>,
    /// How many `vokra.f0.fcpe.*` keys were written: `0` with no topology,
    /// `7` when the topology is a variant (front-end withheld), `14` when
    /// it matches [`FCPE_V001_TOPOLOGY`].
    pub axes_stamped: usize,
    /// `true` when a topology was derived but the seven front-end / decode
    /// axes were withheld because it is not the released v001 architecture.
    ///
    /// An artifact in this state carries correct weights and a correct
    /// topology, and the runtime will refuse to bind it until someone who
    /// knows the variant supplies its front-end axes. That refusal is the
    /// point — see the module doc.
    pub front_end_withheld: bool,
}

/// Internal shared conversion: reads a parsed safetensors buffer,
/// writes every F32 / F16 / BF16 tensor verbatim under its upstream
/// name, and stamps the `vokra.model.*` + `vokra.provenance.*`
/// metadata chunks.
///
/// The caller handles the `license` override at the outer boundary —
/// this function always stamps the built-in default (`mit`,
/// [`LicenseClass::Permissive`]). The [`crate::convert_file_licensed`]
/// outer wrapper re-stamps the `vokra.provenance.{license,
/// weight_license,source}` chunks when the caller supplied a non-
/// default SPDX id.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, FcpeReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a decode path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Built-in stamp = mit Permissive. The outer `convert_file_licensed`
    // layer overrides these three chunks if the caller passed a distinct
    // `--license <spdx>`.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );

    // `vokra.f0.fcpe.*` — the topology is measured off the tensor shapes; the
    // front-end is asserted only when those shapes prove this is v001. See
    // the module doc for why the two halves are treated differently.
    //
    // Built through struct-update rather than default-then-assign so the
    // `?` on `derive_topology` short-circuits before any report exists —
    // clippy::field_reassign_with_default flags the other order.
    let mut report = FcpeReport {
        topology: derive_topology(&st)?,
        ..Default::default()
    };
    if let Some(topo) = report.topology {
        b.add_u32(KEY_D_MODEL, topo.d_model);
        b.add_u32(KEY_N_MELS, topo.n_mels);
        b.add_u32(KEY_STEM_KERNEL, topo.stem_kernel);
        b.add_u32(KEY_FFN_DIM, topo.ffn_dim);
        b.add_u32(KEY_CONV_KERNEL, topo.conv_kernel);
        b.add_u32(KEY_N_LAYERS, topo.n_layers);
        b.add_u32(KEY_N_PITCH_BINS, topo.n_pitch_bins);
        report.axes_stamped += DERIVED_AXIS_COUNT;

        if topo.front_end_is_assertable() {
            b.add_u32(KEY_HOP, V001_HOP);
            b.add_u32(KEY_N_FFT, V001_N_FFT);
            b.add_u32(KEY_SAMPLE_RATE, V001_SAMPLE_RATE);
            b.add_f32(KEY_FMIN, V001_FMIN);
            b.add_f32(KEY_FMAX, V001_FMAX);
            b.add_u32(KEY_STEM_GROUPS, V001_STEM_GROUPS);
            b.add_f32(KEY_CONFIDENCE_THRESHOLD, V001_CONFIDENCE_THRESHOLD);
            report.axes_stamped += FRONT_END_AXIS_COUNT;
        } else {
            report.front_end_withheld = true;
        }
    }
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }
    Ok((b, report))
}

/// File-based FCPE converter (standalone entry — mirror of
/// `convert_neucodec_file` / `convert_xcodec2_file`).
///
/// Reads `input` (prep-script-flattened FCPE safetensors), writes a
/// Vokra GGUF to `output`. `license` overrides the default `mit`
/// provenance stamp (the Whisper / kokoro override pattern); pass
/// `None` to keep the built-in `mit` stamp.
///
/// # Errors
///
/// - [`ConvertError::Io`] if the input cannot be read or the output
///   cannot be written.
/// - [`ConvertError::Parse`] if the safetensors header is malformed.
/// - [`ConvertError::Gguf`] if the GGUF cannot be assembled.
pub fn convert_fcpe_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FcpeReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let (mut b, report) = convert(bytes)?;

    // Standalone-entry license override: mirror the outer
    // `convert_file_licensed` logic so a caller invoking this function
    // directly (bypassing `ModelKind` dispatch) still gets the same
    // license-override semantics.
    if let Some(spdx) = license.filter(|s| !s.is_empty()) {
        let class = LicenseClass::from_license_str(spdx);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, spdx);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("upstream distribution source (licence {spdx} per source)"),
        );
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// A unique temp path — per-process id **plus** a monotonic counter so
    /// two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-fcpe-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// Encodes an f32 array as little-endian BF16 bytes (top 16 bits of
    /// the f32 pattern — the exact inverse of the runtime's
    /// `decode_bf16 : bits << 16`).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-declared dtype and raw payload.
    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// A BF16 safetensors payload converts to a BF16 GGUF tensor with the
    /// canonical arch / name / category / provenance chunks stamped.
    #[test]
    fn convert_bf16_pass_through_stamps_metadata() {
        let values = vec![1.0f32, -2.0, 3.5, 0.25];
        let payload = bf16_bytes(&values);
        let st = safetensors_one("stem.weight", "BF16", &[2, 2], &payload);

        let in_path = tmp_path("bf16-in");
        let out_path = tmp_path("bf16-out");
        std::fs::write(&in_path, &st).unwrap();

        let report = convert_fcpe_file(&in_path, &out_path, None)
            .expect("well-formed BF16 checkpoint must convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let bytes = std::fs::read(&out_path).unwrap();
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );

        // BF16 tensor was preserved verbatim (top-16 f32 bits == our payload).
        let info = file.tensor_info("stem.weight").expect("tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions.iter().product::<u64>(), 4);
        let out_bytes = file.tensor_bytes(info);
        assert_eq!(out_bytes, &payload[..]);

        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }

    /// Builds a synthetic multi-tensor F32 safetensors buffer from a
    /// `(name, shape)` list, zero-filled. Only the shapes matter to the
    /// topology derivation, so the payload is left at zero.
    fn safetensors_f32(specs: &[(&str, &[u64])]) -> Vec<u8> {
        let mut entries: Vec<String> = Vec::with_capacity(specs.len());
        let mut payload: Vec<u8> = Vec::new();
        for (name, shape) in specs {
            let elems: u64 = shape.iter().product();
            let start = payload.len();
            payload.resize(start + elems as usize * 4, 0u8);
            let end = payload.len();
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                r#""{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#
            ));
        }
        let body = entries.join(",");
        let header = format!("{{{body}}}");
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// The tensor set of a structurally complete but deliberately NON-v001
    /// FCPE: `d_model=8, n_mels=4, stem_kernel=3, ffn_dim=8, conv_kernel=3,
    /// n_layers=2, n_pitch_bins=16`. Small enough for a unit test, and every
    /// axis differs from [`FCPE_V001_TOPOLOGY`], so it also exercises the
    /// front-end withholding gate.
    fn tiny_variant_safetensors() -> Vec<u8> {
        safetensors_f32(&[
            ("input_stack.0.weight", &[8, 4, 3]),
            ("net.encoder_layers.0.conformer.net.0.weight", &[8]),
            ("net.encoder_layers.0.conformer.net.2.weight", &[8, 8, 1]),
            (
                "net.encoder_layers.0.conformer.net.4.conv.weight",
                &[4, 1, 3],
            ),
            ("net.encoder_layers.1.conformer.net.0.weight", &[8]),
            ("net.encoder_layers.1.conformer.net.2.weight", &[8, 8, 1]),
            (
                "net.encoder_layers.1.conformer.net.4.conv.weight",
                &[4, 1, 3],
            ),
            ("output_proj.weight", &[16, 8]),
        ])
    }

    /// The topology is READ OFF the tensor shapes, not assumed: a checkpoint
    /// whose every axis differs from v001 gets its own seven values stamped.
    ///
    /// Regression pin for the pre-2026-08-15 converter, which stamped none of
    /// the fourteen `vokra.f0.fcpe.*` axes at all — so the runtime silently
    /// read this artifact as `d_model=512, n_mels=128, n_layers=6, …`.
    #[test]
    fn convert_derives_topology_axes_from_tensor_shapes() {
        let (b, report) =
            convert(tiny_variant_safetensors()).expect("a complete FCPE variant must convert");
        let topo = report
            .topology
            .expect("a complete FCPE must yield a topology");
        assert_eq!(
            topo,
            FcpeTopology {
                d_model: 8,
                n_mels: 4,
                stem_kernel: 3,
                ffn_dim: 8,
                conv_kernel: 3,
                n_layers: 2,
                n_pitch_bins: 16,
            },
        );

        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse GGUF");
        for (key, want) in [
            (KEY_D_MODEL, 8u64),
            (KEY_N_MELS, 4),
            (KEY_STEM_KERNEL, 3),
            (KEY_FFN_DIM, 8),
            (KEY_CONV_KERNEL, 3),
            (KEY_N_LAYERS, 2),
            (KEY_N_PITCH_BINS, 16),
        ] {
            assert_eq!(
                file.get(key).and_then(|v| v.as_u64()),
                Some(want),
                "`{key}` must carry the value derived from the tensor shapes",
            );
        }
    }

    /// A variant topology means the released front-end constants describe
    /// nothing we can vouch for, so all seven are withheld — an artifact the
    /// runtime then refuses rather than running at an assumed 16 kHz.
    #[test]
    fn convert_withholds_front_end_axes_on_a_variant_topology() {
        let (b, report) = convert(tiny_variant_safetensors()).expect("variant must convert");
        assert!(
            report.front_end_withheld,
            "a non-v001 topology must withhold the front-end axes",
        );
        assert_eq!(
            report.axes_stamped, DERIVED_AXIS_COUNT,
            "exactly the derived axes, and none of the asserted ones",
        );

        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse GGUF");
        for key in [
            KEY_HOP,
            KEY_N_FFT,
            KEY_SAMPLE_RATE,
            KEY_FMIN,
            KEY_FMAX,
            KEY_STEM_GROUPS,
            KEY_CONFIDENCE_THRESHOLD,
        ] {
            assert!(
                file.get(key).is_none(),
                "`{key}` is not derivable from this checkpoint and must not be invented",
            );
        }
    }

    /// The front-end gate is exact equality with the released topology —
    /// sharing six axes out of seven is not evidence about a sample rate.
    #[test]
    fn front_end_is_assertable_only_for_the_exact_v001_topology() {
        assert!(FCPE_V001_TOPOLOGY.front_end_is_assertable());
        let near_miss = FcpeTopology {
            n_layers: FCPE_V001_TOPOLOGY.n_layers + 2,
            ..FCPE_V001_TOPOLOGY
        };
        assert!(
            !near_miss.front_end_is_assertable(),
            "a deeper fine-tune shares v001's front-end only by coincidence, if at all",
        );
    }

    /// A safetensors carrying neither trigger tensor is not an FCPE: no
    /// topology, no `vokra.f0.fcpe.*` chunk, and no error either (the
    /// pass-through path still writes the weights).
    #[test]
    fn convert_stamps_no_config_chunk_without_the_trigger_tensors() {
        let st = safetensors_f32(&[("some.other.weight", &[2, 2])]);
        let (b, report) = convert(st).expect("a non-FCPE buffer still passes tensors through");
        assert!(report.topology.is_none());
        assert_eq!(report.axes_stamped, 0);
        assert!(!report.front_end_withheld);
        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse GGUF");
        assert!(file.get(KEY_D_MODEL).is_none());
        assert!(file.get(KEY_HOP).is_none());
    }

    /// Half an FCPE (stem present, head absent) is refused rather than
    /// half-derived.
    #[test]
    fn derive_topology_refuses_a_partial_checkpoint() {
        let st = SafetensorsFile::parse(safetensors_f32(&[("input_stack.0.weight", &[8, 4, 3])]))
            .expect("parse safetensors");
        let Err(err) = derive_topology(&st) else {
            panic!("a checkpoint with a stem but no head must not yield a topology");
        };
        let msg = err.to_string();
        assert!(
            msg.contains(TENSOR_HEAD_W),
            "the error must name the tensor it could not find: {msg}",
        );
    }

    /// Zero discoverable encoder layers is refused — the round-7 RMVPE shape
    /// (mel fed straight past the encoder), caught here at conversion time
    /// instead of producing an artifact whose stamped `n_layers` is a guess.
    #[test]
    fn derive_topology_refuses_zero_encoder_layers() {
        let st = SafetensorsFile::parse(safetensors_f32(&[
            ("input_stack.0.weight", &[8, 4, 3]),
            ("output_proj.weight", &[16, 8]),
        ]))
        .expect("parse safetensors");
        let Err(err) = derive_topology(&st) else {
            panic!("a checkpoint with no encoder layer must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("zero encoder blocks"),
            "the error must say what is missing: {msg}",
        );
    }

    /// The runtime binds every layer with layer 0's shapes, so a
    /// heterogeneous stack is refused here rather than mis-bound there.
    #[test]
    fn derive_topology_refuses_non_uniform_encoder_layers() {
        let st = SafetensorsFile::parse(safetensors_f32(&[
            ("input_stack.0.weight", &[8, 4, 3]),
            ("net.encoder_layers.0.conformer.net.0.weight", &[8]),
            ("net.encoder_layers.0.conformer.net.2.weight", &[8, 8, 1]),
            (
                "net.encoder_layers.0.conformer.net.4.conv.weight",
                &[4, 1, 3],
            ),
            ("net.encoder_layers.1.conformer.net.0.weight", &[8]),
            // Layer 1 expands to 16 instead of 8.
            ("net.encoder_layers.1.conformer.net.2.weight", &[16, 8, 1]),
            (
                "net.encoder_layers.1.conformer.net.4.conv.weight",
                &[8, 1, 3],
            ),
            ("output_proj.weight", &[16, 8]),
        ]))
        .expect("parse safetensors");
        let Err(err) = derive_topology(&st) else {
            panic!("a non-uniform encoder must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("layer 1"),
            "the error must name the layer that diverged: {msg}",
        );
    }

    /// A head that projects from a different width than the stem produces is
    /// internally inconsistent — caught here because the axes are separable
    /// in the shapes, where the runtime binder only ever sees flat counts.
    #[test]
    fn derive_topology_refuses_inconsistent_head_width() {
        let st = SafetensorsFile::parse(safetensors_f32(&[
            ("input_stack.0.weight", &[8, 4, 3]),
            ("net.encoder_layers.0.conformer.net.0.weight", &[8]),
            ("net.encoder_layers.0.conformer.net.2.weight", &[8, 8, 1]),
            (
                "net.encoder_layers.0.conformer.net.4.conv.weight",
                &[4, 1, 3],
            ),
            // Head projects from 16 channels, stem produces 8.
            ("output_proj.weight", &[16, 16]),
        ]))
        .expect("parse safetensors");
        let Err(err) = derive_topology(&st) else {
            panic!("an inconsistent head width must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("internally inconsistent"),
            "the error must say the checkpoint disagrees with itself: {msg}",
        );
    }

    /// A caller-supplied `--license` override rewrites the provenance
    /// chunks (the Whisper / kokoro / xcodec2 pattern) — the callable
    /// path preserves it without touching the tensor bytes.
    #[test]
    fn convert_honors_license_override() {
        let value = 1.0f32;
        let st = safetensors_one("head.weight", "F32", &[1], &value.to_le_bytes());

        let in_path = tmp_path("license-in");
        let out_path = tmp_path("license-out");
        std::fs::write(&in_path, &st).unwrap();
        let _ = convert_fcpe_file(&in_path, &out_path, Some("apache-2.0"))
            .expect("licence override must be accepted");

        let file = GgufFile::parse(std::fs::read(&out_path).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }
}
