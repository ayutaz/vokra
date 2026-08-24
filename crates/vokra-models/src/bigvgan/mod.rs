//! **BigVGAN** (`nvidia/bigvgan_*` family, MIT) — standalone runtime binder
//! for the `bigvgan` converter arch.
//!
//! Upstream: <https://github.com/NVIDIA/BigVGAN>. The sibling converter at
//! `crates/vokra-convert/src/models/bigvgan.rs` emits the four released
//! variants:
//!
//! - `nvidia/bigvgan_v2_22khz_80band_256x` (D2)
//! - `nvidia/bigvgan_v2_44khz_128band_512x` (D3)
//! - `nvidia/bigvgan_v2_24khz_100band_256x` (D4)
//! - `nvidia/bigvgan_base_24khz_100band` (D5, v1 base)
//!
//! # Distinct from `HiFiGan`
//!
//! `crates/vokra-models/src/hifigan/mod.rs::HiFiGan` is the sibling
//! **HiFi-GAN family** vocoder binder (leaky_relu activation, no
//! alias-free activation wrapping). BigVGAN uses **Snake / SnakeBeta**
//! periodic activations with optional alias-free upsample wrapping and
//! ships an AMPBlock1 MRF (2 conv pairs per dilation), so they are
//! intentionally distinct arch tags — silently sharing an arch would
//! mis-route runtime dispatch (mirror of HiFiGan's
//! `arch_tags_are_distinct_and_match_converters` regression pin).
//!
//! # Design: strict real-weight dispatch and tensor walk
//!
//! [`BigVGan::from_gguf`] dispatches on `vokra.model.arch` (must equal
//! [`ARCH`]) + `vokra.bigvgan.variant` (must match one of the four
//! [`BigVGanVariant`] tags), strictly binds the complete folded-convolution,
//! activation, and alias-free filter manifest, then constructs the native
//! generator. Missing, extra, renamed, or wrong-shaped tensors fail with
//! [`VokraError::ModelLoad`] (FR-EX-08).
//!
//! Hand-built [`BigVGan::new`] and [`BigVGan::synthesized`] work today —
//! they never touch the `from_gguf` weight-load path. The
//! [`BigVGan::decode`] method delegates verbatim to
//! [`BigVGanGenerator::forward`] with **no re-wiring**; every internal
//! forward primitive (Snake, SnakeBeta, AMPBlock1, transposed
//! upsample, MRF averaging, terminal tanh / clamp) already lives in
//! `crates/vokra-ops/src/bigvgan_generator.rs`, so this binder is a
//! thin shape-checked wrapper.
//!
//! # Anti-aliased activation
//!
//! Upstream wraps every `Snake` / `SnakeBeta` call with an `Activation1d`
//! module that inserts a polyphase `UpSample1d → activation → DownSample1d`
//! chain (upstream `alias_free_activation/torch/act.py`, cited from
//! `bigvgan.py:87` + `bigvgan.py:277`). That wrapper is what makes
//! BigVGAN "anti-aliased". The runtime binds every checkpoint-stored
//! `upsample.filter` and `downsample.lowpass.filter` buffer and applies the
//! reference-equivalent wrapper around every periodic activation.
//!
//! # `vokra.bigvgan.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::bigvgan::convert_bigvgan_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"bigvgan"`).
//! - `vokra.model.name` (`String`): `"bigvgan-{variant-slug}"` per
//!   variant — auxiliary check.
//! - `vokra.bigvgan.variant` (`String`): one of the four
//!   `VARIANT_TAG_*` constants below — the discriminator the runtime
//!   dispatches on (mirror of `vokra.focalcodec.variant` + the sibling
//!   `vokra.snac.variant`).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] and the variant tag string constants below are intentionally
//! duplicated between this binder and
//! `crates/vokra-convert/src/models/bigvgan.rs` so `vokra-models` does
//! not gain a dependency edge onto `vokra-convert` (mirror of the SNAC
//! + FSMN-VAD + openwakeword + dnsmos + FocalCodec + WeSpeaker
//!   binders — same rule keeps the layered convention `vokra-ops →
//! nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models →
//! GGUF binder`, `vokra-convert → GGUF writer`). Drift is caught by
//!   the [`arch_and_variant_tags_match_converter`] regression pin below.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{BackendKind, Result, VokraError};
use vokra_ops::bigvgan_generator::{
    AliasFreeActivationWeights, AmpBlock1Weights, BigVGanConfig, BigVGanGenerator, BigVGanWeights,
    BigVganBackendOps, SnakeKind,
};

use crate::compute::{Compute, HotOp};
use crate::hifigan::HifiGanComputeOps;

/// Complete learned-op registry for every released BigVGAN variant.
pub const BIGVGAN_HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::SnakeActivation, HotOp::SnakeBeta];

impl BigVganBackendOps for HifiGanComputeOps<'_> {
    fn snake(
        &self,
        input: &[f32],
        alpha: &[f32],
        channels: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; input.len()];
        self.compute
            .snake_activation_f32(input, alpha, channels, time, &mut output)?;
        Ok(output)
    }

    fn snake_beta(
        &self,
        input: &[f32],
        alpha: &[f32],
        beta: &[f32],
        channels: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; input.len()];
        self.compute
            .snake_beta_f32(input, alpha, beta, channels, time, &mut output)?;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Arch / variant / metadata-key constants — mirror of
// crates/vokra-convert/src/models/bigvgan.rs (see the module docstring).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model bigvgan-*`.
///
/// Intentionally distinct from every sibling vocoder arch tag
/// (`hifigan_vocoder`, `speecht5_hifigan`, `snac`, `dac`, `mimi`,
/// `wavtokenizer`, …) because BigVGAN's forward chain (Snake /
/// SnakeBeta + AMPBlock1 + optional anti-aliased activation wrapping)
/// is topologically distinct — silently sharing an arch tag would
/// mis-route runtime dispatch. Pinned verbatim against the converter
/// by [`arch_and_variant_tags_match_converter`].
pub const ARCH: &str = "bigvgan";

/// `vokra.bigvgan.variant` metadata key — the discriminator the
/// runtime dispatches on to pick the correct per-variant
/// [`BigVGanConfig`]. Consumers dispatch on this key rather than
/// parsing free-text `vokra.model.name`.
pub const KEY_BIGVGAN_VARIANT: &str = "vokra.bigvgan.variant";

/// Variant tag written for `nvidia/bigvgan_v2_22khz_80band_256x` (D2).
pub const VARIANT_TAG_V2_22KHZ_80BAND_256X: &str = "v2_22khz_80band_256x";

/// Variant tag written for `nvidia/bigvgan_v2_44khz_128band_512x` (D3).
pub const VARIANT_TAG_V2_44KHZ_128BAND_512X: &str = "v2_44khz_128band_512x";

/// Variant tag written for `nvidia/bigvgan_v2_24khz_100band_256x` (D4).
pub const VARIANT_TAG_V2_24KHZ_100BAND_256X: &str = "v2_24khz_100band_256x";

/// Variant tag written for `nvidia/bigvgan_base_24khz_100band` (D5,
/// v1 base).
pub const VARIANT_TAG_BASE_V1_24KHZ_100BAND: &str = "base_v1_24khz_100band";

/// Canonical ratio-2, 12-tap Kaiser-sinc buffer generated by upstream
/// `Activation1d`. Used only by synthesized fixtures; real GGUFs bind every
/// stored filter buffer independently.
const SYNTHETIC_ALIAS_FREE_FILTER: [f32; 12] = [
    0.002_028_964_7,
    0.009_389_466,
    -0.025_543_459,
    -0.057_657_383,
    0.128_572_58,
    0.443_209_8,
    0.443_209_8,
    0.128_572_58,
    -0.057_657_383,
    -0.025_543_459,
    0.009_389_466,
    0.002_028_964_7,
];

fn synthesized_alias_free_filter() -> AliasFreeActivationWeights {
    AliasFreeActivationWeights {
        upsample_filter: SYNTHETIC_ALIAS_FREE_FILTER.to_vec(),
        downsample_filter: SYNTHETIC_ALIAS_FREE_FILTER.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// BigVGanVariant — mirror of crates/vokra-convert/src/models/bigvgan.rs
// ---------------------------------------------------------------------------

/// Which BigVGAN release the loaded GGUF carries. Selected via the
/// `vokra.bigvgan.variant` chunk written by the converter.
///
/// Mirror of `BigVGanVariant` in
/// `crates/vokra-convert/src/models/bigvgan.rs` — the
/// two enums are kept structurally identical (same order, same
/// `#[derive]`s, same variant docstrings) so a reader that inspects
/// one side has no drift risk on the other. The cross-crate constant
/// duplication rule (see module doc) applies: adding a dependency
/// edge `vokra-models → vokra-convert` would reverse the layer stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BigVGanVariant {
    /// `nvidia/bigvgan_v2_22khz_80band_256x` (D2): 22 050 Hz output,
    /// 80-band mel input, 256× total upsample.
    V2_22khz80Band256x,
    /// `nvidia/bigvgan_v2_44khz_128band_512x` (D3): 44 100 Hz output,
    /// 128-band mel input, 512× total upsample.
    V2_44khz128Band512x,
    /// `nvidia/bigvgan_v2_24khz_100band_256x` (D4): 24 000 Hz output,
    /// 100-band mel input, 256× total upsample.
    V2_24khz100Band256x,
    /// `nvidia/bigvgan_base_24khz_100band` (D5): v1 base 24 000 Hz
    /// output, 100-band mel input, 256× total upsample. Topologically
    /// distinct from D4 — `upsample_initial_channel = 512` vs D4's
    /// 1536, `upsample_rates = [8, 8, 2, 2]` vs D4's [4, 4, 2, 2, 2, 2]
    /// (only 4 upsample stages vs 6). Both `use_bias_at_final` and
    /// `use_tanh_at_final` are absent from base_v1's config.json and
    /// pick up the upstream Python `.get(_, True)` default (see
    /// `bigvgan.py:313` + `bigvgan.py:322`).
    BaseV1_24khz100Band,
}

impl BigVGanVariant {
    /// Wire tag written into `vokra.bigvgan.variant`.
    ///
    /// Kept `const fn` so the constant [`ARCH`] + variant-tag
    /// [`arch_and_variant_tags_match_converter`] regression pin can
    /// assert every mapping at compile time.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::V2_22khz80Band256x => VARIANT_TAG_V2_22KHZ_80BAND_256X,
            Self::V2_44khz128Band512x => VARIANT_TAG_V2_44KHZ_128BAND_512X,
            Self::V2_24khz100Band256x => VARIANT_TAG_V2_24KHZ_100BAND_256X,
            Self::BaseV1_24khz100Band => VARIANT_TAG_BASE_V1_24KHZ_100BAND,
        }
    }

    /// Parses a `vokra.bigvgan.variant` chunk value into a variant, or
    /// returns `None` for an unrecognized string.
    ///
    /// Kept as a free function (not a `TryFrom` impl) so the caller
    /// keeps the ability to attach a per-key context prefix to the
    /// loud error message — [`BigVGan::from_gguf`] uses that below.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            VARIANT_TAG_V2_22KHZ_80BAND_256X => Some(Self::V2_22khz80Band256x),
            VARIANT_TAG_V2_44KHZ_128BAND_512X => Some(Self::V2_44khz128Band512x),
            VARIANT_TAG_V2_24KHZ_100BAND_256X => Some(Self::V2_24khz100Band256x),
            VARIANT_TAG_BASE_V1_24KHZ_100BAND => Some(Self::BaseV1_24khz100Band),
            _ => None,
        }
    }

    /// Output sample rate from the corresponding upstream `config.json`.
    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        match self {
            Self::V2_22khz80Band256x => 22_050,
            Self::V2_44khz128Band512x => 44_100,
            Self::V2_24khz100Band256x | Self::BaseV1_24khz100Band => 24_000,
        }
    }
}

// ---------------------------------------------------------------------------
// config_for_variant — per-variant hardcoded table transcribed verbatim
// from each variant's upstream `config.json`.
// ---------------------------------------------------------------------------

/// Resolves the [`BigVGanConfig`] for a given variant, transcribed
/// verbatim from the upstream HF `config.json` files.
///
/// # Primary sources (verified 2026-08-14)
///
/// Each variant's `config.json` was fetched from Hugging Face; the
/// axes below match those files field-for-field (CLAUDE.md
/// 「ハルシネーション厳禁」).
///
/// - `V2_22khz80Band256x`: <https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/raw/main/config.json>
/// - `V2_44khz128Band512x`: <https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/raw/main/config.json>
/// - `V2_24khz100Band256x`: <https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/raw/main/config.json>
/// - `BaseV1_24khz100Band`: <https://huggingface.co/nvidia/bigvgan_base_24khz_100band/raw/main/config.json>
///
/// # base_v1 default fallbacks
///
/// The `use_bias_at_final` and `use_tanh_at_final` keys are **absent**
/// from base_v1's `config.json`. Upstream `bigvgan.py:313` reads them
/// as `h.get("use_bias_at_final", True)` and `bigvgan.py:322` as
/// `h.get("use_tanh_at_final", True)` — both default to `True` when
/// absent. Every other variant explicitly sets both to `false`, so
/// this asymmetry is deliberate on upstream's part and mirrored here.
///
#[must_use]
pub fn config_for_variant(variant: BigVGanVariant) -> BigVGanConfig {
    match variant {
        BigVGanVariant::V2_22khz80Band256x => BigVGanConfig {
            in_channels: 80,
            upsample_initial_channel: 1536,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            use_bias_at_final: false,
            use_tanh_at_final: false,
        },
        BigVGanVariant::V2_44khz128Band512x => BigVGanConfig {
            in_channels: 128,
            upsample_initial_channel: 1536,
            upsample_rates: vec![8, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            use_bias_at_final: false,
            use_tanh_at_final: false,
        },
        BigVGanVariant::V2_24khz100Band256x => BigVGanConfig {
            in_channels: 100,
            upsample_initial_channel: 1536,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            use_bias_at_final: false,
            use_tanh_at_final: false,
        },
        BigVGanVariant::BaseV1_24khz100Band => BigVGanConfig {
            in_channels: 100,
            upsample_initial_channel: 512,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            // Both keys absent from base_v1's config.json → upstream
            // Python defaults to True (see `bigvgan.py:313` +
            // `bigvgan.py:322`).
            use_bias_at_final: true,
            use_tanh_at_final: true,
        },
    }
}

// ---------------------------------------------------------------------------
// BigVGan — the runtime binder
// ---------------------------------------------------------------------------

/// Standalone BigVGAN vocoder handle: owns a [`BigVGanGenerator`] built
/// from a variant-driven [`BigVGanConfig`] plus a caller-supplied
/// [`BigVGanWeights`] bundle. Exposes [`decode`](Self::decode) as the
/// primary mel → PCM entry point.
///
/// # Sibling wrappers
///
/// - `crates/vokra-models/src/hifigan/mod.rs::HiFiGan` — sibling
///   HiFi-GAN vocoder binder (leaky_relu, no alias-free activation).
/// - `crates/vokra-models/src/snac/mod.rs::Snac` — SNAC codec binder
///   (multi-scale hierarchical RVQ; loud-partial encode / decode).
#[derive(Debug, Clone)]
pub struct BigVGan {
    generator: BigVGanGenerator,
    variant: BigVGanVariant,
    backend: BackendKind,
}

impl BigVGan {
    /// Assembles a BigVGAN handle from a variant tag + pre-built weight
    /// bundle. Runs [`BigVGanGenerator::new`]'s full shape validation
    /// on the variant-driven config + weights pair (SbV2Decoder /
    /// HiFiGan::new precedent) so a mismatched pair fails loudly at
    /// construction, never deep inside a forward (FR-EX-08).
    ///
    /// # Errors
    ///
    /// Any [`VokraError::InvalidArgument`] raised by
    /// [`BigVGanGenerator::new`] on a weight/config shape mismatch
    /// (upsample-stage channel schedule, MRF branch count / kernel /
    /// dilation, activation-post alpha/beta pairing, terminal conv
    /// bias presence, …).
    pub fn new(variant: BigVGanVariant, weights: BigVGanWeights) -> Result<Self> {
        let cfg = config_for_variant(variant);
        let generator = BigVGanGenerator::new(cfg, weights)?;
        Ok(Self {
            generator,
            variant,
            backend: BackendKind::Cpu,
        })
    }

    /// Deterministic zero-initialised fixture — **construction /
    /// shape-flow scaffold**.
    ///
    /// Materialises a valid [`BigVGanWeights`] bundle whose shape
    /// matches the variant-driven [`BigVGanConfig`], then delegates
    /// to [`Self::new`]. Every conv / MRF-branch weight and every
    /// per-channel activation `alpha` (and `beta` for SnakeBeta) is
    /// zero-initialised, so any downstream [`decode`](Self::decode)
    /// call emits near-zero audio bounded by the terminal `tanh` /
    /// clamp (upstream `bigvgan.py:334-337`).
    ///
    /// # Memory budget
    ///
    /// The weight bundle scales with `upsample_initial_channel²` × MRF
    /// branch count. Approximate zero-init allocation sizes (both the
    /// input `weights` and the cloned copies stored inside each
    /// [`AmpBlock1`]):
    ///
    /// - `BaseV1_24khz100Band` (ich=512, 4 stages) — ~140 MB
    /// - `V2_22khz80Band256x` / `V2_24khz100Band256x` (ich=1536,
    ///   6 stages) — ~370 MB
    /// - `V2_44khz128Band512x` (ich=1536, 6 stages) — ~370 MB
    ///
    /// Rust `vec![0.0f32; N]` for a zero fill maps to `alloc_zeroed`
    /// (anonymous mmap + zero-page overcommit on Linux) so physical
    /// footprint stays low until forward writes actually touch the
    /// pages — but CI-friendly unit tests should prefer the tests-only
    /// tiny synthetic construction path (see this module's tests) over
    /// synthesizing a full v2 variant.
    ///
    /// # Errors
    ///
    /// Any [`VokraError::InvalidArgument`] raised by
    /// [`BigVGanGenerator::new`] — this constructor's zero-init
    /// weights bundle matches the variant-driven config by
    /// construction, so an error here would indicate an internal bug
    /// in the shape-derivation logic below.
    pub fn synthesized(variant: BigVGanVariant) -> Result<Self> {
        let cfg = config_for_variant(variant);
        let weights = synthesized_weights_for_config(&cfg);
        Self::new(variant, weights)
    }

    /// Which BigVGAN release this binder was built for.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> BigVGanVariant {
        self.variant
    }

    /// Immutable view of the shape-metadata bundle the underlying
    /// generator was built with (transcribed from primary source by
    /// [`config_for_variant`]).
    ///
    /// Not marked `const fn` because [`BigVGanGenerator::config`] is
    /// not itself `const fn`; a follow-up wave that lifts that
    /// constraint can promote this accessor too.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &BigVGanConfig {
        self.generator.config()
    }

    /// Selects the backend used by every learned convolution and periodic
    /// activation. CPU remains the constructor default.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Runs the BigVGAN forward on `mel` (`[in_channels, t_mel]`
    /// row-major, `mel.len() == config().in_channels * t_mel`) and
    /// returns the raw PCM waveform bounded to `[-1, 1]` by the op's
    /// terminal `tanh` (or `clamp` when `use_tanh_at_final` is false).
    ///
    /// Delegates verbatim to [`BigVGanGenerator::forward`] — this
    /// binder adds no extra pre / post processing.
    ///
    /// # Errors
    ///
    /// See [`BigVGanGenerator::forward`]. In practice, once `self` has
    /// passed [`Self::new`], the only reachable errors are
    /// [`VokraError::InvalidArgument`] on a `mel.len()` mismatch or a
    /// `t_mel == 0`.
    pub fn decode(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        if self.backend == BackendKind::Cpu {
            self.generator.forward(mel, t_mel)
        } else {
            let compute = Compute::for_backend(self.backend, BIGVGAN_HOT_OPS)?;
            let ops = HifiGanComputeOps { compute: &compute };
            self.generator.forward_with_backend_ops(mel, t_mel, &ops)
        }
    }

    /// Dispatches on `vokra.model.arch` + `vokra.bigvgan.variant`, walks the
    /// complete upstream tensor tree into [`BigVGanWeights`], rejects manifest
    /// drift, and constructs the native generator.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is missing,
    ///   not a UTF-8 string, or does not match [`ARCH`].
    /// - [`VokraError::ModelLoad`] when `vokra.bigvgan.variant` is
    ///   missing.
    /// - [`VokraError::ModelLoad`] when `vokra.bigvgan.variant`
    ///   carries an unrecognised tag (a rogue converter, or a future
    ///   5th variant this runtime does not know how to dispatch on —
    ///   refuse loud rather than silently defaulting to any known
    ///   variant, FR-EX-08).
    /// - [`VokraError::ModelLoad`] for a missing, extra, or wrong-shaped
    ///   tensor, including any alias-free filter buffer.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.bigvgan.variant missing".
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "BigVGan::from_gguf: missing or non-string GGUF metadata key `{}` — the \
                     bigvgan converter (`vokra-cli convert --model bigvgan-*`) stamps this key; \
                     a GGUF without it is either not a BigVGAN vocoder or was produced by a \
                     converter that predates the arch-dispatch discipline.",
                    chunks::KEY_MODEL_ARCH
                ))
            })?;
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "BigVGan::from_gguf: unsupported `vokra.model.arch` = {arch:?}. This binder \
                 accepts only {ARCH:?} (`nvidia/bigvgan_*` family). Other vocoder-family GGUFs \
                 route through their own binder modules (`hifigan_vocoder` / `speecht5_hifigan` \
                 → `HiFiGan`, `snac` → `Snac`, etc)."
            )));
        }

        // 2. Variant discrimination — `vokra.bigvgan.variant` is
        //    required (no silent default: a v2_44khz GGUF loaded as
        //    v2_22khz would corrupt every downstream mel-shape check).
        let variant_tag = file
            .get(KEY_BIGVGAN_VARIANT)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "BigVGan::from_gguf: GGUF is missing `{KEY_BIGVGAN_VARIANT}` — every \
                     bigvgan GGUF must declare its variant so the runtime can pick the \
                     correct per-variant config bundle. Expected one of {:?}, {:?}, {:?}, \
                     {:?}.",
                    VARIANT_TAG_V2_22KHZ_80BAND_256X,
                    VARIANT_TAG_V2_44KHZ_128BAND_512X,
                    VARIANT_TAG_V2_24KHZ_100BAND_256X,
                    VARIANT_TAG_BASE_V1_24KHZ_100BAND,
                ))
            })?;
        let variant = BigVGanVariant::from_tag(variant_tag).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "BigVGan::from_gguf: `{KEY_BIGVGAN_VARIANT}` = {variant_tag:?} is not a \
                 recognised variant tag. Expected one of {:?}, {:?}, {:?}, {:?} — a rogue \
                 converter or a future 5th variant this runtime does not dispatch on; \
                 refusing loud rather than silently defaulting to any known variant \
                 (FR-EX-08).",
                VARIANT_TAG_V2_22KHZ_80BAND_256X,
                VARIANT_TAG_V2_44KHZ_128BAND_512X,
                VARIANT_TAG_V2_24KHZ_100BAND_256X,
                VARIANT_TAG_BASE_V1_24KHZ_100BAND,
            ))
        })?;

        let cfg = config_for_variant(variant);
        let weights = load_weights(file, &cfg)?;
        Self::new(variant, weights).map_err(|error| {
            VokraError::ModelLoad(format!(
                "BigVGan::from_gguf({variant:?}): loaded tensor tree failed generator validation: {error}"
            ))
        })
    }
}

fn load_tensor(
    file: &GgufFile,
    name: &str,
    expected_shape: &[usize],
    expected_names: &mut std::collections::BTreeSet<String>,
) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("BigVGan: required tensor `{name}` is missing"))
    })?;
    let actual_shape: Vec<usize> = info.dimensions.iter().map(|&dim| dim as usize).collect();
    if actual_shape != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "BigVGan: tensor `{name}` shape {actual_shape:?}, expected {expected_shape:?}"
        )));
    }
    expected_names.insert(name.to_owned());
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("BigVGan: tensor `{name}` decode failed: {error}"))
    })
}

fn load_alias_free_filter(
    file: &GgufFile,
    prefix: &str,
    expected_names: &mut std::collections::BTreeSet<String>,
) -> Result<AliasFreeActivationWeights> {
    let upsample_name = format!("{prefix}.upsample.filter");
    let downsample_name = format!("{prefix}.downsample.lowpass.filter");
    Ok(AliasFreeActivationWeights {
        upsample_filter: load_tensor(file, &upsample_name, &[1, 1, 12], expected_names)?,
        downsample_filter: load_tensor(file, &downsample_name, &[1, 1, 12], expected_names)?,
    })
}

fn load_weights(file: &GgufFile, cfg: &BigVGanConfig) -> Result<BigVGanWeights> {
    use std::collections::BTreeSet;

    let mut expected_names = BTreeSet::new();
    let n_ups = cfg.num_upsamples();
    let n_kernels = cfg.num_kernels();
    let initial = cfg.upsample_initial_channel as usize;
    let input = cfg.in_channels as usize;

    let conv_pre_w = load_tensor(
        file,
        "conv_pre.weight",
        &[initial, input, 7],
        &mut expected_names,
    )?;
    let conv_pre_b = load_tensor(file, "conv_pre.bias", &[initial], &mut expected_names)?;

    let mut ups_w = Vec::with_capacity(n_ups);
    let mut ups_b = Vec::with_capacity(n_ups);
    for stage in 0..n_ups {
        let in_channels = (cfg.upsample_initial_channel >> stage as u32) as usize;
        let out_channels = cfg.output_channels_at(stage) as usize;
        let kernel = cfg.upsample_kernel_sizes[stage] as usize;
        ups_w.push(load_tensor(
            file,
            &format!("ups.{stage}.0.weight"),
            &[in_channels, out_channels, kernel],
            &mut expected_names,
        )?);
        ups_b.push(load_tensor(
            file,
            &format!("ups.{stage}.0.bias"),
            &[out_channels],
            &mut expected_names,
        )?);
    }

    let mut amp_blocks = Vec::with_capacity(n_ups * n_kernels);
    for stage in 0..n_ups {
        let channels = cfg.output_channels_at(stage) as usize;
        for branch in 0..n_kernels {
            let block = stage * n_kernels + branch;
            let kernel = cfg.resblock_kernel_sizes[branch] as usize;
            let layers = cfg.resblock_dilation_sizes[branch].len();
            let mut convs1_w = Vec::with_capacity(layers);
            let mut convs1_b = Vec::with_capacity(layers);
            let mut convs2_w = Vec::with_capacity(layers);
            let mut convs2_b = Vec::with_capacity(layers);
            let mut activations1_alpha = Vec::with_capacity(layers);
            let mut activations2_alpha = Vec::with_capacity(layers);
            let mut activations1_beta =
                matches!(cfg.activation, SnakeKind::SnakeBeta).then(|| Vec::with_capacity(layers));
            let mut activations2_beta =
                matches!(cfg.activation, SnakeKind::SnakeBeta).then(|| Vec::with_capacity(layers));
            let mut activations1_filters = Vec::with_capacity(layers);
            let mut activations2_filters = Vec::with_capacity(layers);

            for layer in 0..layers {
                for (destination, conv_group) in [
                    ((&mut convs1_w, &mut convs1_b), "convs1"),
                    ((&mut convs2_w, &mut convs2_b), "convs2"),
                ] {
                    let prefix = format!("resblocks.{block}.{conv_group}.{layer}");
                    destination.0.push(load_tensor(
                        file,
                        &format!("{prefix}.weight"),
                        &[channels, channels, kernel],
                        &mut expected_names,
                    )?);
                    destination.1.push(load_tensor(
                        file,
                        &format!("{prefix}.bias"),
                        &[channels],
                        &mut expected_names,
                    )?);
                }

                for (activation_index, alpha_destination, beta_destination, filter_destination) in [
                    (
                        layer * 2,
                        &mut activations1_alpha,
                        activations1_beta.as_mut(),
                        &mut activations1_filters,
                    ),
                    (
                        layer * 2 + 1,
                        &mut activations2_alpha,
                        activations2_beta.as_mut(),
                        &mut activations2_filters,
                    ),
                ] {
                    let prefix = format!("resblocks.{block}.activations.{activation_index}");
                    alpha_destination.push(load_tensor(
                        file,
                        &format!("{prefix}.act.alpha"),
                        &[channels],
                        &mut expected_names,
                    )?);
                    if let Some(beta_destination) = beta_destination {
                        beta_destination.push(load_tensor(
                            file,
                            &format!("{prefix}.act.beta"),
                            &[channels],
                            &mut expected_names,
                        )?);
                    }
                    filter_destination.push(load_alias_free_filter(
                        file,
                        &prefix,
                        &mut expected_names,
                    )?);
                }
            }
            amp_blocks.push(AmpBlock1Weights {
                convs1_w,
                convs1_b,
                convs2_w,
                convs2_b,
                activations1_alpha,
                activations2_alpha,
                activations1_beta,
                activations2_beta,
                activations1_filters,
                activations2_filters,
            });
        }
    }

    let last_channels = cfg.output_channels_at(n_ups - 1) as usize;
    let activation_post_alpha = load_tensor(
        file,
        "activation_post.act.alpha",
        &[last_channels],
        &mut expected_names,
    )?;
    let activation_post_beta = if matches!(cfg.activation, SnakeKind::SnakeBeta) {
        Some(load_tensor(
            file,
            "activation_post.act.beta",
            &[last_channels],
            &mut expected_names,
        )?)
    } else {
        None
    };
    let activation_post_filter =
        load_alias_free_filter(file, "activation_post", &mut expected_names)?;
    let conv_post_w = load_tensor(
        file,
        "conv_post.weight",
        &[1, last_channels, 7],
        &mut expected_names,
    )?;
    let conv_post_b = if cfg.use_bias_at_final {
        Some(load_tensor(
            file,
            "conv_post.bias",
            &[1],
            &mut expected_names,
        )?)
    } else {
        None
    };

    let actual_names: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|info| info.name.clone())
        .collect();
    if actual_names != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual_names).take(4).collect();
        let extra: Vec<&String> = actual_names.difference(&expected_names).take(4).collect();
        return Err(VokraError::ModelLoad(format!(
            "BigVGan: tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual_names.len()
        )));
    }

    Ok(BigVGanWeights {
        conv_pre_w,
        conv_pre_b,
        ups_w,
        ups_b,
        amp_blocks,
        activation_post_alpha,
        activation_post_beta,
        activation_post_filter,
        conv_post_w,
        conv_post_b,
    })
}

// ---------------------------------------------------------------------------
// synthesized_weights_for_config — internal shape-matched zero-init helper
// ---------------------------------------------------------------------------

/// Builds a valid [`BigVGanWeights`] bundle whose shape matches `cfg`
/// exactly, with every conv / MRF-branch weight and every activation
/// `alpha` (+ `beta` for SnakeBeta) zero-initialised.
///
/// Every axis is derived from `cfg` (no fabricated dimensions), so
/// [`BigVGanGenerator::new`]'s upfront shape validation always
/// succeeds for a `cfg` that itself validates.
fn synthesized_weights_for_config(cfg: &BigVGanConfig) -> BigVGanWeights {
    let n_ups = cfg.num_upsamples();
    let n_kernels = cfg.num_kernels();
    let bc = cfg.upsample_initial_channel as usize;
    let inc = cfg.in_channels as usize;

    // conv_pre: [upsample_initial_channel, in_channels, 7] (upstream
    // L212 — kernel size is fixed at 7 in `BigVGanGenerator::forward`).
    let conv_pre_w = vec![0.0f32; bc * inc * 7];
    let conv_pre_b = vec![0.0f32; bc];

    // Per-stage upsample ConvTranspose1d: [in_ch_i, out_ch_i, k_i]
    // (upstream L235-245, in-channels-leading PyTorch layout).
    let mut ups_w = Vec::with_capacity(n_ups);
    let mut ups_b = Vec::with_capacity(n_ups);
    for i in 0..n_ups {
        let in_ch = (cfg.upsample_initial_channel >> (i as u32)) as usize;
        let out_ch = (cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
        let k = cfg.upsample_kernel_sizes[i] as usize;
        ups_w.push(vec![0.0f32; in_ch * out_ch * k]);
        ups_b.push(vec![0.0f32; out_ch]);
    }

    // AMPBlock1 per (stage, kernel) slot — row-major
    // `amp_blocks[i * num_kernels + j]` (upstream L254-260). Each block
    // uses `channels = out_ch_i`, `kernel = resblock_kernel_sizes[j]`,
    // `dilations = resblock_dilation_sizes[j]`.
    let mut amp_blocks = Vec::with_capacity(n_ups * n_kernels);
    for i in 0..n_ups {
        let ch = cfg.output_channels_at(i) as usize;
        for j in 0..n_kernels {
            let k = cfg.resblock_kernel_sizes[j] as usize;
            let dils = &cfg.resblock_dilation_sizes[j];
            let n_layers = dils.len();
            let (activations1_beta, activations2_beta) = match cfg.activation {
                SnakeKind::Snake => (None, None),
                SnakeKind::SnakeBeta => (
                    Some(vec![vec![0.0f32; ch]; n_layers]),
                    Some(vec![vec![0.0f32; ch]; n_layers]),
                ),
            };
            amp_blocks.push(AmpBlock1Weights {
                convs1_w: vec![vec![0.0f32; ch * ch * k]; n_layers],
                convs1_b: vec![vec![0.0f32; ch]; n_layers],
                convs2_w: vec![vec![0.0f32; ch * ch * k]; n_layers],
                convs2_b: vec![vec![0.0f32; ch]; n_layers],
                activations1_alpha: vec![vec![0.0f32; ch]; n_layers],
                activations2_alpha: vec![vec![0.0f32; ch]; n_layers],
                activations1_beta,
                activations2_beta,
                activations1_filters: vec![synthesized_alias_free_filter(); n_layers],
                activations2_filters: vec![synthesized_alias_free_filter(); n_layers],
            });
        }
    }

    // activation_post + conv_post: [1, last_ch, 7] (upstream L263-283).
    let last_ch = cfg.output_channels_at(n_ups - 1) as usize;
    let activation_post_alpha = vec![0.0f32; last_ch];
    let activation_post_beta = match cfg.activation {
        SnakeKind::Snake => None,
        SnakeKind::SnakeBeta => Some(vec![0.0f32; last_ch]),
    };
    let conv_post_w = vec![0.0f32; last_ch * 7];
    let conv_post_b = if cfg.use_bias_at_final {
        Some(vec![0.0f32; 1])
    } else {
        None
    };

    BigVGanWeights {
        conv_pre_w,
        conv_pre_b,
        ups_w,
        ups_b,
        amp_blocks,
        activation_post_alpha,
        activation_post_beta,
        activation_post_filter: synthesized_alias_free_filter(),
        conv_post_w,
        conv_post_b,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    // ---- T1: constants pinned against the converter ------------------

    /// Task-spec pin: [`ARCH`] + [`KEY_BIGVGAN_VARIANT`] + every
    /// variant tag MUST match verbatim the constants the sibling
    /// converter emits (`crates/vokra-convert/src/models/bigvgan.rs`).
    /// A converter rename that skipped this module would silently
    /// route to the unknown-arch / unknown-variant error path instead
    /// of the deferred-loader loud path — this test catches that
    /// drift.
    #[test]
    fn arch_and_variant_tags_match_converter() {
        assert_eq!(ARCH, "bigvgan");
        assert_eq!(KEY_BIGVGAN_VARIANT, "vokra.bigvgan.variant");
        assert_eq!(
            BigVGanVariant::V2_22khz80Band256x.tag(),
            "v2_22khz_80band_256x"
        );
        assert_eq!(
            BigVGanVariant::V2_44khz128Band512x.tag(),
            "v2_44khz_128band_512x"
        );
        assert_eq!(
            BigVGanVariant::V2_24khz100Band256x.tag(),
            "v2_24khz_100band_256x"
        );
        assert_eq!(
            BigVGanVariant::BaseV1_24khz100Band.tag(),
            "base_v1_24khz_100band"
        );
    }

    // ---- T2: variant tag round-trip ----------------------------------

    /// Every variant's `tag()` must round-trip through `from_tag()` to
    /// itself, and unknown / empty tags MUST return `None` (never a
    /// silent default variant).
    #[test]
    fn all_four_variants_round_trip_tag_encoding() {
        for variant in [
            BigVGanVariant::V2_22khz80Band256x,
            BigVGanVariant::V2_44khz128Band512x,
            BigVGanVariant::V2_24khz100Band256x,
            BigVGanVariant::BaseV1_24khz100Band,
        ] {
            let tag = variant.tag();
            let round_trip = BigVGanVariant::from_tag(tag)
                .expect("every variant's tag() must be recognised by from_tag()");
            assert_eq!(round_trip, variant, "round-trip failed for {variant:?}");
        }
        assert_eq!(BigVGanVariant::from_tag(""), None);
        assert_eq!(BigVGanVariant::from_tag("v3_not_a_variant"), None);
        assert_eq!(BigVGanVariant::from_tag("bigvgan"), None);
    }

    // ---- T3: config axes match primary source ------------------------

    /// Every axis in [`config_for_variant`] must match the verbatim
    /// upstream `config.json` value for that variant (CLAUDE.md
    /// 「ハルシネーション厳禁」). Primary sources are cited in
    /// [`config_for_variant`]'s rustdoc. A pinning change here MUST
    /// come with a corresponding update to the upstream config fetch
    /// note in that rustdoc.
    #[test]
    fn config_axes_match_primary_source_all_four_variants() {
        // V2_22khz80Band256x — nvidia/bigvgan_v2_22khz_80band_256x
        let cfg = config_for_variant(BigVGanVariant::V2_22khz80Band256x);
        assert_eq!(cfg.in_channels, 80);
        assert_eq!(cfg.upsample_initial_channel, 1536);
        assert_eq!(cfg.upsample_rates, vec![4, 4, 2, 2, 2, 2]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![8, 8, 4, 4, 4, 4]);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            cfg.resblock_dilation_sizes,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        assert_eq!(cfg.activation, SnakeKind::SnakeBeta);
        assert!(cfg.snake_logscale);
        assert!(!cfg.use_bias_at_final);
        assert!(!cfg.use_tanh_at_final);
        assert_eq!(cfg.total_upsample_factor(), 256);

        // V2_44khz128Band512x — nvidia/bigvgan_v2_44khz_128band_512x
        let cfg = config_for_variant(BigVGanVariant::V2_44khz128Band512x);
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.upsample_initial_channel, 1536);
        assert_eq!(cfg.upsample_rates, vec![8, 4, 2, 2, 2, 2]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![16, 8, 4, 4, 4, 4]);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            cfg.resblock_dilation_sizes,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        assert_eq!(cfg.activation, SnakeKind::SnakeBeta);
        assert!(cfg.snake_logscale);
        assert!(!cfg.use_bias_at_final);
        assert!(!cfg.use_tanh_at_final);
        assert_eq!(cfg.total_upsample_factor(), 512);

        // V2_24khz100Band256x — nvidia/bigvgan_v2_24khz_100band_256x
        let cfg = config_for_variant(BigVGanVariant::V2_24khz100Band256x);
        assert_eq!(cfg.in_channels, 100);
        assert_eq!(cfg.upsample_initial_channel, 1536);
        assert_eq!(cfg.upsample_rates, vec![4, 4, 2, 2, 2, 2]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![8, 8, 4, 4, 4, 4]);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            cfg.resblock_dilation_sizes,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        assert_eq!(cfg.activation, SnakeKind::SnakeBeta);
        assert!(cfg.snake_logscale);
        // Primary source says `false` for both — this deliberately
        // disagrees with `BigVGanConfig::default()` in ops (see the
        // `config_for_variant` rustdoc "Divergence" section).
        assert!(!cfg.use_bias_at_final);
        assert!(!cfg.use_tanh_at_final);
        assert_eq!(cfg.total_upsample_factor(), 256);

        // BaseV1_24khz100Band — nvidia/bigvgan_base_24khz_100band
        let cfg = config_for_variant(BigVGanVariant::BaseV1_24khz100Band);
        assert_eq!(cfg.in_channels, 100);
        assert_eq!(cfg.upsample_initial_channel, 512);
        assert_eq!(cfg.upsample_rates, vec![8, 8, 2, 2]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![16, 16, 4, 4]);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(
            cfg.resblock_dilation_sizes,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
        );
        // Primary source confirms base_v1 uses "snakebeta", not
        // "snake" — a common misconception because v1 base predates
        // the v2 anti-aliased activation wrapper.
        assert_eq!(cfg.activation, SnakeKind::SnakeBeta);
        assert!(cfg.snake_logscale);
        // Both keys are absent from base_v1's `config.json`;
        // upstream defaults them to `True` (see rustdoc "base_v1
        // default fallbacks" section).
        assert!(cfg.use_bias_at_final);
        assert!(cfg.use_tanh_at_final);
        assert_eq!(cfg.total_upsample_factor(), 8 * 8 * 2 * 2);
    }

    // ---- T4-T7: from_gguf loud error paths ---------------------------

    /// A GGUF that does not carry `vokra.model.arch` at all must fail
    /// with [`VokraError::ModelLoad`] naming the missing key — never
    /// a silent success on a zero-tensor fixture.
    #[test]
    fn from_gguf_missing_arch_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.name", "no-arch-here");
        let bytes = b.to_bytes().expect("build minimal GGUF");
        let file = GgufFile::parse(bytes).expect("parse minimal GGUF");
        let Err(err) = BigVGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming the missing arch key");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(chunks::KEY_MODEL_ARCH),
                    "error must name the missing key: {msg}"
                );
            }
            other => panic!("expected ModelLoad naming the missing arch key, got: {other}"),
        }
    }

    /// A GGUF carrying an arch tag that is not `"bigvgan"` must fail
    /// with [`VokraError::ModelLoad`] naming both the accepted arch
    /// and the offending value.
    #[test]
    fn from_gguf_wrong_arch_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "hifigan_vocoder");
        let bytes = b.to_bytes().expect("build GGUF with wrong arch");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = BigVGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming supported arch on wrong arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(ARCH), "error must name accepted arch: {msg}");
                assert!(
                    msg.contains("hifigan_vocoder"),
                    "error must name the offending value: {msg}"
                );
            }
            other => panic!("expected ModelLoad naming supported arch, got: {other}"),
        }
    }

    /// A well-formed BigVGAN GGUF missing `vokra.bigvgan.variant` must
    /// fail with [`VokraError::ModelLoad`] naming the missing key.
    #[test]
    fn from_gguf_missing_variant_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        let bytes = b.to_bytes().expect("build minimal bigvgan GGUF");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = BigVGan::from_gguf(&file) else {
            panic!("expected ModelLoad on missing variant key");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(KEY_BIGVGAN_VARIANT),
                    "error must name the missing variant key: {msg}"
                );
            }
            other => panic!("expected ModelLoad naming variant key, got: {other}"),
        }
    }

    /// A GGUF whose `vokra.bigvgan.variant` value is not one of the
    /// four accepted tags must fail with [`VokraError::ModelLoad`]
    /// naming every accepted tag so a downstream caller can pick the
    /// right converter.
    #[test]
    fn from_gguf_unknown_variant_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(KEY_BIGVGAN_VARIANT, "v3_fake_variant");
        let bytes = b.to_bytes().expect("build GGUF with unknown variant");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = BigVGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming supported variants");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(VARIANT_TAG_V2_22KHZ_80BAND_256X),
                    "error must name v2_22khz tag: {msg}"
                );
                assert!(
                    msg.contains(VARIANT_TAG_V2_44KHZ_128BAND_512X),
                    "error must name v2_44khz tag: {msg}"
                );
                assert!(
                    msg.contains(VARIANT_TAG_V2_24KHZ_100BAND_256X),
                    "error must name v2_24khz tag: {msg}"
                );
                assert!(
                    msg.contains(VARIANT_TAG_BASE_V1_24KHZ_100BAND),
                    "error must name base_v1 tag: {msg}"
                );
                assert!(
                    msg.contains("v3_fake_variant"),
                    "error must name the offending value: {msg}"
                );
            }
            other => panic!("expected ModelLoad naming variants, got: {other}"),
        }
    }

    // ---- T8: valid metadata reaches the strict tensor loader --------

    /// Metadata-only files for all four variants must pass dispatch and fail
    /// at the first required tensor. This pins that no variant regresses to a
    /// placeholder branch while keeping the unit fixture small; real-weight
    /// verification covers the complete manifest.
    #[test]
    fn from_gguf_all_four_variants_reach_strict_tensor_loader() {
        for variant in [
            BigVGanVariant::V2_22khz80Band256x,
            BigVGanVariant::V2_44khz128Band512x,
            BigVGanVariant::V2_24khz100Band256x,
            BigVGanVariant::BaseV1_24khz100Band,
        ] {
            let mut b = GgufBuilder::new();
            b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
            b.add_string(KEY_BIGVGAN_VARIANT, variant.tag());
            let bytes = b.to_bytes().expect("build valid bigvgan GGUF");
            let file = GgufFile::parse(bytes).expect("parse GGUF");
            let Err(err) = BigVGan::from_gguf(&file) else {
                panic!("expected missing-tensor failure on metadata-only {variant:?}");
            };
            match err {
                VokraError::ModelLoad(msg) => {
                    assert!(
                        msg.contains("conv_pre.weight") && msg.contains("missing"),
                        "strict loader must name the first missing tensor for {variant:?}: {msg}"
                    );
                }
                other => panic!("expected ModelLoad on {variant:?}, got: {other}"),
            }
        }
    }

    // ---- T9: synthesized construction smoke on base_v1 ---------------

    /// `BigVGan::synthesized(BaseV1_24khz100Band)` must construct
    /// successfully: config_for_variant + shape-matched zero-init
    /// weights pass every upfront shape check in
    /// [`BigVGanGenerator::new`] (upsample-stage channel schedule, MRF
    /// branch count / kernel / dilation, activation-post alpha/beta
    /// pairing, terminal conv bias presence).
    ///
    /// Scoped to base_v1 (ich=512, 4 stages) because the v2 variants
    /// (ich=1536, 6 stages) allocate ~370 MB of zero weights — see
    /// [`BigVGan::synthesized`]'s "Memory budget" rustdoc for the
    /// per-variant breakdown. Construction-only (no decode) — the
    /// actual mel → PCM path is exercised on a much smaller synthetic
    /// config below in `decode_delegates_to_generator_forward`.
    #[test]
    fn synthesized_construction_smoke_base_v1() {
        let vg = BigVGan::synthesized(BigVGanVariant::BaseV1_24khz100Band)
            .expect("synthesized base_v1 must construct");
        assert_eq!(vg.variant(), BigVGanVariant::BaseV1_24khz100Band);
        assert_eq!(vg.config().in_channels, 100);
        assert_eq!(vg.config().upsample_initial_channel, 512);
        assert_eq!(vg.config().num_upsamples(), 4);
        assert_eq!(vg.config().activation, SnakeKind::SnakeBeta);
        // Both fallback-to-True defaults (see `config_for_variant`
        // "base_v1 default fallbacks" section).
        assert!(vg.config().use_bias_at_final);
        assert!(vg.config().use_tanh_at_final);
    }

    // ---- T10: decode delegates to generator.forward -----------------

    /// Builds a tiny synthetic bundle (2 upsample stages, 1 MRF
    /// branch, initial_channel = 8, in_channels = 4) directly and
    /// wraps it in a `BigVGan` — bypassing `config_for_variant` for
    /// a CI-cheap decode smoke.
    ///
    /// This exercises the `decode → generator.forward` delegation
    /// path; every real variant's config produces the same forward
    /// math, so a tiny synthetic proves the plumbing. The full v2
    /// variants (~370 MB weights + billions of ops per t_mel=1 call)
    /// are not CI-friendly and are covered by the shape assertion in
    /// [`synthesized_construction_smoke_base_v1`] above.
    #[test]
    fn decode_delegates_to_generator_forward() {
        let cfg = BigVGanConfig {
            in_channels: 4,
            upsample_initial_channel: 8,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
            activation: SnakeKind::Snake,
            snake_logscale: false,
            use_bias_at_final: true,
            use_tanh_at_final: true,
        };
        let weights = synthesized_weights_for_config(&cfg);
        let generator = BigVGanGenerator::new(cfg, weights).expect("tiny generator");
        // Private-field construction — this test lives in the same
        // module as `BigVGan`, so we can bypass the variant-driven
        // config path for a CI-cheap smoke without polluting the
        // public API with a test-only variant-override constructor.
        let vg = BigVGan {
            generator,
            variant: BigVGanVariant::V2_24khz100Band256x,
            backend: BackendKind::Cpu,
        };
        assert_eq!(vg.variant(), BigVGanVariant::V2_24khz100Band256x);

        let t_mel = 3;
        let mel = vec![0.0f32; 4 * t_mel];
        let pcm = vg.decode(&mel, t_mel).expect("decode zero mel");
        // Every stage doubles the time base (upstream transposed-conv
        // formula for these upsample_rates + kernel choices).
        let expected_len = t_mel * (2 * 2);
        assert_eq!(
            pcm.len(),
            expected_len,
            "decode output length matches product of upsample_rates"
        );
        for (i, &v) in pcm.iter().enumerate() {
            assert!(v.is_finite(), "PCM[{i}] = {v} must be finite (tanh output)");
            assert!(
                (-1.0..=1.0).contains(&v),
                "PCM[{i}] = {v} must lie in [-1, 1] after terminal tanh"
            );
        }
    }

    #[test]
    fn compute_backend_matches_scalar_full_bigvgan_stack() {
        let cfg = BigVGanConfig {
            in_channels: 4,
            upsample_initial_channel: 8,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            use_bias_at_final: true,
            use_tanh_at_final: true,
        };
        let mut weights = synthesized_weights_for_config(&cfg);
        fn fill(values: &mut [f32], scale: f32) {
            for (index, value) in values.iter_mut().enumerate() {
                *value = ((index % 13) as f32 - 6.0) * scale;
            }
        }
        fill(&mut weights.conv_pre_w, 0.002);
        fill(&mut weights.conv_pre_b, 0.001);
        for (weight, bias) in weights.ups_w.iter_mut().zip(&mut weights.ups_b) {
            fill(weight, 0.002);
            fill(bias, 0.001);
        }
        for block in &mut weights.amp_blocks {
            for weight in block.convs1_w.iter_mut().chain(block.convs2_w.iter_mut()) {
                fill(weight, 0.001);
            }
            for bias in block.convs1_b.iter_mut().chain(block.convs2_b.iter_mut()) {
                fill(bias, 0.0005);
            }
        }
        fill(&mut weights.conv_post_w, 0.002);
        fill(weights.conv_post_b.as_mut().expect("terminal bias"), 0.001);
        let generator = BigVGanGenerator::new(cfg, weights).expect("tiny non-zero generator");
        let t_mel = 4;
        let mel: Vec<f32> = (0..4 * t_mel)
            .map(|index| ((index % 9) as f32 - 4.0) * 0.03)
            .collect();
        let scalar = generator.forward(&mel, t_mel).expect("scalar forward");
        let compute = Compute::cpu();
        let ops = HifiGanComputeOps { compute: &compute };
        let dispatched = generator
            .forward_with_backend_ops(&mel, t_mel, &ops)
            .expect("Compute backend forward");
        assert_eq!(scalar.len(), dispatched.len());
        let max_abs = scalar
            .iter()
            .zip(&dispatched)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "BigVGAN composed backend drift {max_abs} exceeds the registered synthetic FP32 gate"
        );
    }

    /// Regression pin: `decode` must forward the underlying generator's
    /// bounded-output guarantee even when the terminal activation is
    /// `clamp` instead of `tanh` (`use_tanh_at_final = false`).
    /// Distinct from the smoke above because base_v1 uses the tanh
    /// path via config default, while every v2 variant uses clamp; a
    /// regression that broke clamp routing would surface here.
    #[test]
    fn decode_bounded_when_terminal_clamp_used() {
        let cfg = BigVGanConfig {
            in_channels: 4,
            upsample_initial_channel: 8,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            // clamp path (mirrors every v2 variant's config.json).
            use_bias_at_final: false,
            use_tanh_at_final: false,
        };
        let mut weights = synthesized_weights_for_config(&cfg);
        // Force a huge pre-clamp value to actually exercise the clamp
        // path (all-zero weights + clamp would leave the assertion
        // trivially true).
        weights.conv_post_w = vec![1000.0f32; weights.conv_post_w.len()];
        weights.conv_pre_w = vec![1.0f32; weights.conv_pre_w.len()];
        let generator = BigVGanGenerator::new(cfg, weights).expect("tiny clamp generator");
        let vg = BigVGan {
            generator,
            variant: BigVGanVariant::V2_22khz80Band256x,
            backend: BackendKind::Cpu,
        };
        let t_mel = 2;
        let mel = vec![1.0f32; 4 * t_mel];
        let pcm = vg.decode(&mel, t_mel).expect("decode all-ones mel");
        for (i, &v) in pcm.iter().enumerate() {
            assert!(v.is_finite(), "PCM[{i}] = {v} must be finite");
            assert!(
                (-1.0..=1.0).contains(&v),
                "PCM[{i}] = {v} must lie in [-1, 1] after clamp"
            );
        }
    }
}
