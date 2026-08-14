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
//! # Design: real from_gguf dispatch + loud-partial tensor walk
//!
//! [`BigVGan::from_gguf`] dispatches on `vokra.model.arch` (must equal
//! [`ARCH`]) + `vokra.bigvgan.variant` (must match one of the four
//! [`BigVGanVariant`] tags), then returns
//! [`VokraError::UnsupportedOp`] naming the exact tensor tree walk the
//! follow-up wave needs to implement — this is the **loud-partial**
//! pattern (RMVPE + DFN3 Phase B + HiFiGan + SNAC precedent, CLAUDE.md
//! 「loud-partial は fake-complete より honest」). Missing / wrong arch
//! and missing / unknown variant fail with [`VokraError::ModelLoad`]
//! so a mis-produced GGUF surfaces the specific gap (FR-EX-08).
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
//! # Anti-aliased activation (deferred — mirror of ops-side note)
//!
//! Upstream wraps every `Snake` / `SnakeBeta` call with an `Activation1d`
//! module that inserts a polyphase `UpSample1d → activation → DownSample1d`
//! chain (upstream `alias_free_activation/torch/act.py`, cited from
//! `bigvgan.py:87` + `bigvgan.py:277`). That wrapper is what makes
//! BigVGAN "anti-aliased". The current `vokra_ops::bigvgan_generator`
//! op skeleton lands the *unwrapped* activation (see the module
//! docstring section "Anti-aliased activation (deferred)" in
//! `crates/vokra-ops/src/bigvgan_generator.rs` L66-84) — this binder
//! mirrors that honest omission. When the shared polyphase
//! Kaiser-window filter primitive lands in `vokra-ops`, the
//! `BigVGanGenerator` op body picks it up transparently and this
//! binder's forward semantics upgrade with zero surface change.
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
//! binders — same rule keeps the layered convention `vokra-ops →
//! nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models →
//! GGUF binder`, `vokra-convert → GGUF writer`). Drift is caught by
//! the [`arch_and_variant_tags_match_converter`] regression pin below.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{Result, VokraError};
use vokra_ops::bigvgan_generator::{
    AmpBlock1Weights, BigVGanConfig, BigVGanGenerator, BigVGanWeights, SnakeKind,
};

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

// ---------------------------------------------------------------------------
// BigVGanVariant — mirror of crates/vokra-convert/src/models/bigvgan.rs
// ---------------------------------------------------------------------------

/// Which BigVGAN release the loaded GGUF carries. Selected via the
/// `vokra.bigvgan.variant` chunk written by the converter.
///
/// Mirror of `vokra_convert::models::bigvgan::BigVGanVariant` — the
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
/// # Divergence from `BigVGanConfig::default()`
///
/// `vokra_ops::bigvgan_generator::BigVGanConfig::default()` claims to
/// mirror the `V2_24khz100Band256x` release but sets
/// `use_bias_at_final = true` and `use_tanh_at_final = true`. That
/// disagrees with the actual `nvidia/bigvgan_v2_24khz_100band_256x`
/// `config.json` (both are `false`), so [`config_for_variant`] uses
/// the primary-source values here (`false` for both) rather than the
/// ops-side default. The ops-side default is a pre-existing test
/// fixture (`bigvgan_generator::tests::bigvgan_config_defaults_match_v2_24k_100band_256x`)
/// that this binder does **not** modify; a follow-up wave can align
/// it with primary source separately.
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
        Ok(Self { generator, variant })
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
        self.generator.forward(mel, t_mel)
    }

    /// Dispatches on `vokra.model.arch` + `vokra.bigvgan.variant` and
    /// (in a future wave) walks the tensor tree into a
    /// [`BigVGanWeights`] bundle.
    ///
    /// # Current status — loud-partial
    ///
    /// This entry point is intentionally loud-partial today (RMVPE +
    /// DFN3 Phase B + HiFiGan + SNAC precedent, CLAUDE.md
    /// 「loud-partial は fake-complete より honest」): the arch +
    /// variant dispatch works (missing / wrong arch →
    /// [`VokraError::ModelLoad`], missing / unknown variant →
    /// [`VokraError::ModelLoad`]), and a valid GGUF hits an
    /// [`VokraError::UnsupportedOp`] naming the precise blocker (the
    /// upstream `bigvgan.py:212-322` tensor tree walk into
    /// [`BigVGanWeights`]).
    ///
    /// Hand-built [`Self::new`] and [`Self::synthesized`] work today —
    /// they never touch this path. When the tensor-walk wave lands,
    /// the switch flips inline here without any surface change to the
    /// binder's downstream [`Self::decode`] path.
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
    /// - [`VokraError::UnsupportedOp`] on a well-formed BigVGAN GGUF
    ///   until the tensor-walk wave lands, naming the four
    ///   accepted variant tags + the specific upstream module tree
    ///   the loader needs to walk.
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

        // 3. Tensor-walk defer — the derived-tensor loader that turns
        //    the GGUF's upstream-verbatim tensor tree into a
        //    `BigVGanWeights` bundle is a follow-up wave. Fail loud
        //    with an [`UnsupportedOp`] naming the exact upstream
        //    module tree + tensor prefix walk so the follow-up wave
        //    has a single place to look.
        Err(loud_partial_tensor_walk(variant))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`BigVGan::from_gguf`] until the real-weight loader wave lands.
///
/// Names the specific upstream module tree the loader will walk
/// (`bigvgan.py:212-322`) plus the tensor-prefix pattern the sibling
/// converter emits verbatim (`conv_pre.*` / `ups.{i}.0.*` /
/// `resblocks.{i*num_kernels+j}.convs{1,2}.{k}.*` +
/// `activations.{2k}.alpha[+beta]` / `activation_post.alpha[+beta]` /
/// `conv_post.*`). The `docs/handoff/` cross-reference points a reader
/// diagnosing this gap at exactly one primary source (upstream
/// `NVIDIA/BigVGAN` `bigvgan.py`).
fn loud_partial_tensor_walk(variant: BigVGanVariant) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "BigVGan::from_gguf({variant:?}): real-weight loader is deferred — the sibling \
         converter `crates/vokra-convert/src/models/bigvgan.rs` emits every float tensor \
         verbatim under its upstream safetensors name (`conv_pre.{{weight,bias}}` / \
         `ups.{{i}}.0.{{weight,bias}}` / \
         `resblocks.{{i*num_kernels+j}}.convs{{1,2}}.{{k}}.{{weight,bias}}` / \
         `resblocks.{{i*num_kernels+j}}.activations.{{2k}}.alpha` (+ `beta` for \
         SnakeBeta) / `activation_post.alpha` (+ `beta`) / `conv_post.{{weight,bias}}` — \
         upstream `NVIDIA/BigVGAN/bigvgan.py:212-322` defines the module tree, this \
         binder + `config_for_variant({variant:?})` supply the shape metadata). Follow-up \
         wave walks those tensors into a `BigVGanWeights` bundle + routes through \
         `BigVGan::new`. Hand-built `BigVGan::new` + `BigVGan::synthesized` fixtures \
         work today. Loud pending (CLAUDE.md 「loud-partial は fake-complete より \
         honest」) — no silent fabricated forward ever emitted (FR-EX-08)."
    ))
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

    // ---- T8: valid GGUF hits loud-partial tensor walk ---------------

    /// A well-formed bigvgan GGUF (correct arch + valid variant tag)
    /// must reach the loud-partial arm and fail with
    /// [`VokraError::UnsupportedOp`] naming the upstream module tree
    /// the follow-up wave will walk (`bigvgan.py:212-322`). Runs the
    /// same assertion for every accepted variant tag so a regression
    /// that drops one variant from dispatch surfaces here.
    #[test]
    fn from_gguf_valid_gguf_all_four_variants_returns_loud_partial() {
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
                panic!("expected UnsupportedOp for deferred tensor walk on {variant:?}");
            };
            match err {
                VokraError::UnsupportedOp(msg) => {
                    // The message must name the upstream module tree
                    // so a reader diagnosing this gap has exactly one
                    // place to walk (CLAUDE.md 教訓 (a) — loud
                    // messages cite primary source).
                    assert!(
                        msg.contains("bigvgan.py:212-322"),
                        "loud-partial must cite upstream module tree for {variant:?}: {msg}"
                    );
                    assert!(
                        msg.contains("conv_pre"),
                        "loud-partial must name conv_pre tensor prefix for {variant:?}: {msg}"
                    );
                    assert!(
                        msg.contains("ups"),
                        "loud-partial must name ups.* prefix for {variant:?}: {msg}"
                    );
                    assert!(
                        msg.contains("resblocks"),
                        "loud-partial must name resblocks.* prefix for {variant:?}: {msg}"
                    );
                    assert!(
                        msg.contains("conv_post"),
                        "loud-partial must name conv_post tensor prefix for {variant:?}: {msg}"
                    );
                }
                other => panic!(
                    "expected UnsupportedOp for deferred tensor walk on {variant:?}, got: {other}"
                ),
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
