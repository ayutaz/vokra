//! **Conv-TasNet** (`JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`,
//! CC-BY-SA-4.0 — T3 Copyleft) — Convolutional Time-domain Speech
//! Separation runtime binder for the `conv_tasnet` converter arch
//! (Wave 5 2026-08-14 audit follow-up, first entry on the
//! separation-runtime arm).
//!
//! # Primary source
//!
//! - HF model card:
//!   <https://huggingface.co/JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k>
//! - Asteroid reference implementation:
//!   <https://github.com/asteroid-team/asteroid>
//!   (`asteroid/models/conv_tasnet.py` — encoder + TCN masker + decoder).
//! - Paper: Luo & Mesgarani, *"Conv-TasNet: Surpassing Ideal
//!   Time-Frequency Magnitude Masking for Speech Separation"*, IEEE/ACM
//!   Trans. Audio, Speech & Language Processing 2019 (arXiv:1809.07454).
//! - Weight license: **CC-BY-SA-4.0** per HF cardData primary source
//!   (`license: cc-by-sa-4.0`) → [`LicenseClass::Copyleft`] (SA cascade;
//!   `docs/license-audit.md` §3.1 row 470 `conv-tasnet-libri1mix` = ☑
//!   Commercial 2026-08-02 yousan, T3 tier Copyleft — the same
//!   Copyleft posture the converter's `DEFAULT_LICENSE_SPDX =
//!   "cc-by-sa-4.0"` stamps).
//!
//! # Architecture (transcribed from primary sources)
//!
//! ```text
//! Mixed PCM (mono f32, 16 kHz — see [`ConvTasnetConfig::sample_rate`])
//!   -> 1D Conv encoder                              ← **loud-partial**
//!        (n_filters=512 filters, kernel=16 samples, stride=8 —
//!         `asteroid/models/conv_tasnet.py` `Encoder` block;
//!         composable from Vokra's shared Conv1D primitive but
//!         the tensor-name walk from the upstream state_dict
//!         prefixes has NOT been pinned pending the manifest
//!         fetch.)
//!   -> TCN masker stack                             ← **loud-partial**
//!        (n_repeats=3 repeats × n_blocks=8 dilated 1D Conv
//!         blocks with bottleneck (bn_chan=128) / hidden
//!         (hid_chan=512) / skip (skip_chan=128) channels,
//!         Global LayerNorm or cLN + PReLU + sigmoid or ReLU
//!         mask activation — `asteroid/models/conv_tasnet.py`
//!         `TDConvNet` block. The dilated-conv stack is
//!         composable from Vokra's shared Conv1D + LayerNorm +
//!         PReLU primitives; what is missing is the block
//!         composition + the tensor-name walk from the upstream
//!         state_dict prefixes.)
//!   -> 1D ConvTranspose decoder                     ← **loud-partial**
//!        (mirror of the encoder — n_filters=512 basis, kernel=16
//!         samples, stride=8 — `asteroid/models/conv_tasnet.py`
//!         `Decoder` block; composable from Vokra's shared
//!         ConvTranspose1D primitive.)
//!   -> Vec<Vec<f32>>  (one PCM stream per output source; n_src=1 for
//!                      the `enhsingle` enhancement head, n_src=2+
//!                      for the sibling `Libri2Mix_sep_clean` /
//!                      Libri3Mix multi-source separation variants
//!                      when they land under a distinct ModelKind arm.)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`ConvTasnet::from_gguf`] with strict
//!   `vokra.model.arch == "conv_tasnet"` validation +
//!   [`ConvTasnetConfig::asteroid_libri1mix_default`] primary-source
//!   constant hold (the Conv-TasNet converter currently does NOT stamp
//!   a `vokra.conv_tasnet.*` chunk group — only arch / name / category /
//!   upstream_hf / provenance — so the binder holds the primary-source
//!   constants transcribed from `asteroid/models/conv_tasnet.py`
//!   `ConvTasNet.__init__` + the Asteroid recipe. A future converter
//!   sub-wave that extends the converter to stamp the chunk group will
//!   land alongside a switch to strict axis read; the current
//!   const-only hold is a **documented deviation** from the RMVPE /
//!   ReDimNet strict-read precedent and matches the Sortformer /
//!   pyannote fallback precedent for post-launch chunk-group additions),
//!   [`ConvTasnetWeights::from_gguf`] with a floor of non-empty tensor
//!   count enforced loud (a GGUF that carries zero tensors is refused
//!   rather than silently running an all-zero forward — FR-EX-08),
//!   and weight-license class surfacing.
//! - **Loud-partial (this WP)**: [`ConvTasnet::separate`] returns
//!   [`VokraError::UnsupportedOp`] naming **all three** deferred
//!   pieces of Conv-TasNet's encoder-masker-decoder decomposition:
//!   1. **1D Conv encoder** (n_filters filters, n_kernel=16 sample
//!      kernel, stride=8) — composable from Vokra's Conv1D primitive
//!      but the tensor-name walk from the upstream state_dict prefixes
//!      to Conv1D inputs has NOT been pinned pending the manifest
//!      fetch;
//!   2. **TCN masker stack** (n_repeats × n_blocks dilated 1D Conv
//!      blocks with bn_chan / hid_chan / skip_chan wiring + Global
//!      LayerNorm/cLN + PReLU + mask activation) — composable from
//!      Vokra's Conv1D + LayerNorm + PReLU primitives but the block
//!      composition + tensor-name walk are pending;
//!   3. **1D ConvTranspose decoder** (mirror of the encoder) —
//!      composable from Vokra's ConvTranspose1D primitive.
//!
//! The error names **all three** primary-source anchors (the Asteroid
//! reference implementation, the paper, and the JorisCos HF release)
//! so a reader diagnosing this gap has exactly three places to walk —
//! mirror of the `Sortformer::diarize` / `Mt3::transcribe` / `BeatThis::analyze`
//! / `RMVPE` Wave 3-4 loud-partial-message precedent.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / sortformer loud-partial precedent, CLAUDE.md 教訓
//! (a) "loud-partial は fake-complete より honest"): the surrounding
//! scaffold + `from_gguf` arch validation + `ConvTasnetConfig`
//! primary-source hold + FR-EX-08 loud-fails land today so a follow-up
//! wave can flip the switch by (i) landing the tensor-name walk against
//! a real Conv-TasNet state_dict via the standard Asteroid
//! `bin_to_safetensors.py` prep, (ii) wiring the Conv1D encoder + TCN
//! masker composition + ConvTranspose1D decoder, and (iii) extending
//! the converter to stamp the `vokra.conv_tasnet.*` chunk group so
//! this reader can move to strict axis read. **No fabricated separator
//! output is ever emitted** (FR-EX-08).
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`NAME`] / [`CATEGORY`]) — same rule the sibling BF16
//! pass-through binders (`pyannote` / `snac` / `hifigan` / `beat_this` /
//! `mt3` / `sortformer_diar_4spk_v1`) use so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # Family posture — distinct from SepFormer / Demucs / TIGER / BS-Roformer / MP-SENet
//!
//! [`ARCH`] = `"conv_tasnet"` is **deliberately distinct** from every
//! sibling separator arch tag:
//!
//! - `sepformer` — SpeechBrain SepFormer (dual-path Transformer
//!   masker; encoder-decoder shape is similar but the masker is
//!   attention-based, not a dilated TCN);
//! - `demucs` — Facebook Demucs (hybrid U-Net + spectrogram + cross-
//!   domain attention; fundamentally different family);
//! - `tiger_separator` — TIGER dual-path family (dual-path
//!   RNN / Transformer variants);
//! - `bs_roformer` — Band-Split Roformer (frequency-domain band-
//!   split attention);
//! - `mp_senet` — MP-SENet (magnitude+phase speech enhancement).
//!
//! Silently sharing arch would let runtime dispatch mis-route a
//! Conv-TasNet checkpoint onto a SepFormer / Demucs / TIGER /
//! BS-Roformer / MP-SENet loader — the masker topologies are
//! completely different and the tensor-name walks would fail with a
//! downstream missing-tensor error instead of a specific arch-mismatch
//! message. FR-EX-08 forbids the silent shape misroute across
//! separator families.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Asteroid ships PyTorch `pytorch_model.bin` (raw `torch.save`) or
//! safetensors checkpoints; this runtime **never** touches ONNX or
//! pickle (FR-LD-05 / NFR-DS-02). The `pytorch_model.bin` → safetensors
//! bridge lives in the sibling `bin_to_safetensors.py` prep step (an
//! offline uv-managed Python 3.12 sidecar per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`), not
//! part of the runtime — pickle deserialization inside the Rust
//! runtime would violate the FR-LD-05 "no arbitrary code execution at
//! load" rule.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/conv_tasnet_libri1mix.rs`. See
// module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model conv-tasnet-libri1mix`.
///
/// Deliberately distinct from every sibling separator arch tag
/// (`sepformer` / `demucs` / `tiger_separator` / `bs_roformer` /
/// `mp_senet`) — see the module docstring "Family posture" section
/// for the FR-EX-08 rationale. Version-neutral (a future ConvTasNet-
/// Libri2Mix or ConvTasNet-Libri3Mix multi-source separation variant
/// keeps the tag; only [`NAME`] is versioned).
pub const ARCH: &str = "conv_tasnet";

/// Expected `vokra.model.name` value — matches the
/// `vokra/conv-tasnet-libri1mix` publish slug (when it lands under
/// the T3 Copyleft distribution gate).
pub const NAME: &str = "conv-tasnet-libri1mix";

/// Expected `vokra.model.category` value — single-speaker enhancement
/// head (`Libri1Mix enhsingle` = one clean speaker + additive noise,
/// one output stream). Distinct from the sibling multi-source
/// separation variants (Libri2Mix / Libri3Mix) which would carry
/// `category = "separation"` under a distinct [`crate::compute`]-side
/// arm.
pub const CATEGORY: &str = "enhancement";

/// Primary-source anchor for the Asteroid reference implementation.
/// Cited in the loud-partial error so a reader diagnosing the gap
/// knows the tensor-name walk anchor. Points at the definitive
/// `ConvTasNet` PyTorch reference in the Asteroid tree.
const PRIMARY_SOURCE_ASTEROID: &str =
    "github.com/asteroid-team/asteroid/blob/master/asteroid/models/conv_tasnet.py";
/// Primary-source anchor for the Luo & Mesgarani 2019 paper. Cited
/// alongside the Asteroid + HF anchors so a reader has the theoretical
/// context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/1809.07454";
/// Primary-source anchor for the JorisCos HF release. Cited in the
/// loud-partial error so a reader diagnosing the gap knows the
/// definitive checkpoint source (the enhsingle 16 kHz single-source
/// enhancement Asteroid recipe).
const PRIMARY_SOURCE_HF: &str = "huggingface.co/JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k";

const KEY_N_FILTERS: &str = "vokra.conv_tasnet.n_filters";
const KEY_N_KERNEL: &str = "vokra.conv_tasnet.n_kernel";
const KEY_STRIDE: &str = "vokra.conv_tasnet.stride";
const KEY_N_BLOCKS: &str = "vokra.conv_tasnet.n_blocks";
const KEY_N_REPEATS: &str = "vokra.conv_tasnet.n_repeats";
const KEY_BN_CHAN: &str = "vokra.conv_tasnet.bn_chan";
const KEY_HID_CHAN: &str = "vokra.conv_tasnet.hid_chan";
const KEY_SKIP_CHAN: &str = "vokra.conv_tasnet.skip_chan";
const KEY_CONV_KERNEL_SIZE: &str = "vokra.conv_tasnet.conv_kernel_size";
const KEY_SAMPLE_RATE: &str = "vokra.conv_tasnet.sample_rate";
const KEY_N_SRC: &str = "vokra.conv_tasnet.n_src";
const KEY_CAUSAL: &str = "vokra.conv_tasnet.causal";

// ---------------------------------------------------------------------------
// ConvTasnetConfig — primary-source-transcribed Asteroid ConvTasNet
// Libri1Mix `enhsingle_16k` hparams.
//
// The converter stamps all twelve `vokra.conv_tasnet.*` axes transcribed
// from `asteroid/models/conv_tasnet.py` + the Libri1Mix `enhsingle_16k`
// recipe. The binder reads every key strictly; an older metadata-free GGUF
// is rejected instead of silently assuming a topology.
// ---------------------------------------------------------------------------

/// Asteroid ConvTasNet `enhsingle_16k` hyperparameters, held as the
/// primary-source-transcribed defaults from `asteroid/models/
/// conv_tasnet.py` `ConvTasNet.__init__` + the Libri1Mix `enhsingle_16k`
/// recipe.
///
/// See the module docstring for the "no GGUF chunk read" deviation
/// rationale. All axes are `u32` for uniform serialization when the
/// follow-up wave extends the converter to stamp the chunk group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTasnetConfig {
    /// Encoder filter count — the number of 1D convolution filters
    /// applied to the raw waveform. Primary-source default: 512.
    pub n_filters: u32,
    /// Encoder/decoder kernel size in samples — the 1D Conv encoder
    /// kernel width. Primary-source default: 16 samples (~1 ms at
    /// 16 kHz). Also the ConvTranspose1D decoder kernel width by the
    /// encoder/decoder mirror rule.
    pub n_kernel: u32,
    /// Encoder/decoder stride in samples. Primary-source default: 8
    /// (`n_kernel / 2` — 50 % overlap between successive encoder
    /// frames). Enforced as a structural invariant: [`Self::v1_default`]
    /// asserts `stride == n_kernel / 2`.
    pub stride: u32,
    /// TCN blocks per repeat — the number of stacked dilated 1D Conv
    /// blocks in each repeat of the temporal convolutional network
    /// masker. Primary-source default: 8.
    pub n_blocks: u32,
    /// TCN repeat count — the number of times the `n_blocks` dilated
    /// 1D Conv block sequence is repeated in the masker stack.
    /// Primary-source default: 3.
    pub n_repeats: u32,
    /// Bottleneck channels — the reduced channel count between the
    /// encoder output and each TCN block's dilated Conv input.
    /// Primary-source default: 128.
    pub bn_chan: u32,
    /// Hidden channels — the expanded channel count inside each TCN
    /// block's dilated Conv. Primary-source default: 512.
    pub hid_chan: u32,
    /// Skip connection channels — the channel count of the skip-
    /// connection branch each TCN block contributes to the mask output.
    /// Primary-source default: 128.
    pub skip_chan: u32,
    /// TCN dilated Conv kernel size (not the encoder/decoder kernel;
    /// that is [`Self::n_kernel`]). Primary-source default: 3.
    pub conv_kernel_size: u32,
    /// Sample rate in Hz. Primary-source default: 16000 (matches
    /// `Libri1Mix_enhsingle_16k`).
    pub sample_rate: u32,
    /// Output source count — the number of PCM streams the decoder
    /// produces. Primary-source default: 1 (`Libri1Mix enhsingle` is
    /// single-source enhancement). Sibling multi-source separation
    /// variants (`Libri2Mix_sep_clean_16k`, `Libri3Mix_sep_clean_16k`)
    /// would land under a distinct `ModelKind` arm with `n_src = 2` /
    /// `3` respectively.
    pub n_src: u32,
    /// Causal flag (0 = non-causal / bidirectional cLN, 1 = causal /
    /// gLN + causal-cumulative dilated Conv). Primary-source default:
    /// 0. The Asteroid recipe uses non-causal LayerNorm for
    /// enhancement; a future streaming variant would set 1.
    pub causal: u32,
}

impl Default for ConvTasnetConfig {
    /// The primary-source-transcribed Asteroid Libri1Mix `enhsingle_16k`
    /// defaults. Used by [`ConvTasnet::from_gguf`] as the hold-only
    /// path (no GGUF chunk read — see the module docstring).
    fn default() -> Self {
        Self::asteroid_libri1mix_default()
    }
}

impl ConvTasnetConfig {
    /// The Asteroid Libri1Mix `enhsingle_16k` axes as a `const` —
    /// primary-source-transcribed from `asteroid/models/conv_tasnet.py`
    /// `ConvTasNet.__init__` + the Libri1Mix `enhsingle_16k` recipe.
    ///
    /// Structural invariant: `stride == n_kernel / 2` (50 % overlap
    /// between successive encoder frames — the Asteroid recipe's
    /// `stride=8, n_kernel=16` pin).
    #[must_use]
    pub const fn asteroid_libri1mix_default() -> Self {
        Self {
            n_filters: 512,
            n_kernel: 16,
            stride: 8,
            n_blocks: 8,
            n_repeats: 3,
            bn_chan: 128,
            hid_chan: 512,
            skip_chan: 128,
            conv_kernel_size: 3,
            sample_rate: 16000,
            n_src: 1,
            causal: 0,
        }
    }

    fn from_gguf(file: &GgufFile) -> Result<Self> {
        fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
            match file.get(key) {
                Some(vokra_core::gguf::GgufMetadataValue::U32(value)) => Ok(*value),
                Some(other) => Err(VokraError::ModelLoad(format!(
                    "conv_tasnet: GGUF metadata `{key}` must be U32, got {other:?}"
                ))),
                None => Err(VokraError::ModelLoad(format!(
                    "conv_tasnet: GGUF is missing required topology metadata `{key}`; \
                     reconvert with the current conv-tasnet-libri1mix converter"
                ))),
            }
        }

        let config = Self {
            n_filters: required_u32(file, KEY_N_FILTERS)?,
            n_kernel: required_u32(file, KEY_N_KERNEL)?,
            stride: required_u32(file, KEY_STRIDE)?,
            n_blocks: required_u32(file, KEY_N_BLOCKS)?,
            n_repeats: required_u32(file, KEY_N_REPEATS)?,
            bn_chan: required_u32(file, KEY_BN_CHAN)?,
            hid_chan: required_u32(file, KEY_HID_CHAN)?,
            skip_chan: required_u32(file, KEY_SKIP_CHAN)?,
            conv_kernel_size: required_u32(file, KEY_CONV_KERNEL_SIZE)?,
            sample_rate: required_u32(file, KEY_SAMPLE_RATE)?,
            n_src: required_u32(file, KEY_N_SRC)?,
            causal: required_u32(file, KEY_CAUSAL)?,
        };
        if config.n_filters == 0
            || config.n_kernel == 0
            || config.stride == 0
            || config.n_blocks == 0
            || config.n_repeats == 0
            || config.bn_chan == 0
            || config.hid_chan == 0
            || config.skip_chan == 0
            || config.conv_kernel_size == 0
            || config.sample_rate == 0
            || config.n_src == 0
        {
            return Err(VokraError::ModelLoad(format!(
                "conv_tasnet: topology axes must be non-zero except causal; got {config:?}"
            )));
        }
        if config.stride * 2 != config.n_kernel {
            return Err(VokraError::ModelLoad(format!(
                "conv_tasnet: stride={} must equal n_kernel/2={} for the 50% overlap contract",
                config.stride,
                config.n_kernel / 2
            )));
        }
        if config.causal > 1 {
            return Err(VokraError::ModelLoad(format!(
                "conv_tasnet: causal must be 0 or 1, got {}",
                config.causal
            )));
        }
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// ConvTasnetWeights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// encoder + TCN masker + decoder forward is deferred. Mirror of
// `Mt3Weights` / `BeatThisWeights` / `SortformerWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a Conv-TasNet GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid Conv-TasNet checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave that lands the
/// encoder + TCN masker + decoder forward sizes its dequant per its
/// kernel needs — today only the count + names are consumed so a future
/// `ConvTasnetWeights::bind_encoder_masker_decoder_weights` tensor walk
/// can find its inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct ConvTasnetWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// encoder-masker-decoder-forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl ConvTasnetWeights {
    /// Scans `gguf` for the Conv-TasNet state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid Conv-TasNet checkpoint).
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
                "conv_tasnet: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model conv-tasnet-libri1mix` \
                 against an upstream safetensors checkpoint (Asteroid ships \
                 `pytorch_model.bin`; run the standard `bin_to_safetensors.py` prep step \
                 first — same workflow as the SepFormer `.ckpt` families)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-masker-decoder-forward wave uses it to
    /// size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// ConvTasnet — the runtime binder handle
// ---------------------------------------------------------------------------

/// Conv-TasNet runtime binder (`JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`,
/// CC-BY-SA-4.0 T3 Copyleft tier).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`separate`](Self::separate) on a mixed PCM buffer to obtain a
/// `Vec<Vec<f32>>` (one PCM stream per output source; `n_src=1` for
/// the enhsingle enhancement head). See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the encoder + TCN masker + decoder composition.
#[derive(Debug)]
pub struct ConvTasnet {
    config: ConvTasnetConfig,
    // The bound weights are held (real, counted) but the encoder + TCN
    // masker + decoder composition is a follow-up wave; the field is
    // deliberately `#[allow(dead_code)]` until the composition lands so
    // a reader is not misled by an unused field. Same posture as
    // RMVPE / pyannote / mt3 / beat_this / sortformer.
    #[allow(dead_code)]
    weights: ConvTasnetWeights,
    weight_license: LicenseClass,
}

impl ConvTasnet {
    /// Binds a Conv-TasNet GGUF: validates arch, strictly reads the
    /// source-transcribed [`ConvTasnetConfig`], discovers tensors, and surfaces the
    /// stamped weight-license class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"conv_tasnet"` (a `sepformer` / `demucs` /
    ///   `tiger_separator` / `bs_roformer` / `mp_senet` GGUF handed to
    ///   us by mistake fails with a clear message instead of a
    ///   downstream missing-tensor — all these separator families have
    ///   completely different masker topologies, so the runtime
    ///   dispatch discipline forbids silent aliasing).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero
    ///   tensors ([`ConvTasnetWeights::from_gguf`] refuses to bind an
    ///   all-zero forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "conv_tasnet: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model conv-tasnet-libri1mix`? Note \
                     that the sibling separator arches — `sepformer` (SpeechBrain SepFormer, \
                     dual-path Transformer masker), `demucs` (Facebook Demucs, hybrid U-Net + \
                     spectrogram + cross-domain attention), `tiger_separator` (TIGER dual-path \
                     family), `bs_roformer` (Band-Split Roformer, frequency-domain band-split \
                     attention), `mp_senet` (magnitude+phase speech enhancement) — all have \
                     completely different masker topologies from Conv-TasNet's stacked dilated \
                     TCN. Silently aliasing arch would misroute the runtime dispatch, FR-EX-08.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "conv_tasnet: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native conv_tasnet GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology read. No fallback to constants: old GGUFs that
        //    predate these chunks must be reconverted.
        let config = ConvTasnetConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = ConvTasnetWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The Conv-TasNet
        //    converter defaults to `Copyleft` per the HF card's
        //    `license: cc-by-sa-4.0` (SA cascade — publish is
        //    redistributable with the LICENSE preserved, T3 tier).
        //    Missing provenance falls back to `Unknown` which is
        //    fail-closed at the M2-13 compliance gate — same posture
        //    as MT3 / Sortformer.
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

    /// The bound topology axes read from the GGUF chunk group.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &ConvTasnetConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The Conv-TasNet
    /// converter stamps `Copyleft` by default per the HF card's
    /// `license: cc-by-sa-4.0` (T3 tier — publish redistributable with
    /// LICENSE preserved, SA cascade must carry forward on every
    /// derivative). A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] which is also fail-closed at the
    /// M2-13 compliance gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-masker-decoder-forward wave uses it to
    /// size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Separates a mixed mono PCM buffer (16 kHz per
    /// [`ConvTasnetConfig::sample_rate`]) into a `Vec<Vec<f32>>` (one
    /// PCM stream per output source).
    ///
    /// The return shape carries [`ConvTasnetConfig::n_src`] entries;
    /// for the Libri1Mix `enhsingle` enhancement head this is `1`, but
    /// the API accommodates future multi-source separation variants
    /// (`Libri2Mix_sep_clean_16k` = 2, `Libri3Mix_sep_clean_16k` = 3)
    /// under distinct `ModelKind` arms.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — Conv-TasNet's inference
    /// path requires **three** deferred pieces (the encoder-masker-
    /// decoder decomposition):
    ///
    /// 1. **1D Conv encoder** (n_filters filters, n_kernel=16 sample
    ///    kernel, stride=8) — composable from Vokra's Conv1D primitive
    ///    but the tensor-name walk from the upstream state_dict
    ///    prefixes has NOT been pinned pending the manifest fetch.
    /// 2. **TCN masker stack** (n_repeats × n_blocks dilated 1D Conv
    ///    blocks with bn_chan / hid_chan / skip_chan wiring + Global
    ///    LayerNorm / cLN + PReLU + mask activation) — composable
    ///    from Vokra's Conv1D + LayerNorm + PReLU primitives but the
    ///    block composition + tensor-name walk are pending.
    /// 3. **1D ConvTranspose decoder** (mirror of the encoder) —
    ///    composable from Vokra's ConvTranspose1D primitive.
    ///
    /// The error names **all three** primary-source anchors (the
    /// Asteroid reference implementation, the paper, and the JorisCos
    /// HF release) so a reader diagnosing this gap has exactly three
    /// places to walk. **No fabricated separator output is ever
    /// emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred encoder-masker-decoder composition.
    pub fn separate(&self, mixed_pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = mixed_pcm;
        Err(separate_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`ConvTasnet::separate`] until the tensor-name walk +
/// encoder-masker-decoder composition land.
///
/// Names **all three** primary source URLs (Asteroid reference impl +
/// paper + JorisCos HF release) so a reader diagnosing the gap has
/// exactly three places to walk. Mirrors the Sortformer / MT3 /
/// beat_this / RMVPE / pyannote / snac / hifigan Wave 3-4
/// loud-partial-message precedent — CLAUDE.md 教訓 (a).
///
/// Echoes every [`ConvTasnetConfig`] axis so the reader can
/// cross-check what topology the follow-up wave targets.
fn separate_forward_loud_partial(cfg: &ConvTasnetConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "conv_tasnet separate: 1D Conv encoder + TCN (temporal convolutional network) \
         masker + 1D ConvTranspose decoder composition pending. Conv-TasNet's Asteroid \
         reference decomposes as (a) a 1D Conv encoder ({n_filters} filters, kernel={n_kernel} \
         samples, stride={stride}) that maps the raw waveform to a filterbank representation, \
         (b) a TCN masker stack ({n_repeats} repeats × {n_blocks} dilated 1D Conv blocks with \
         bottleneck bn_chan={bn_chan} / hidden hid_chan={hid_chan} / skip skip_chan={skip_chan} \
         channels, dilated Conv kernel={conv_kernel_size}, Global LayerNorm/cLN + PReLU + \
         sigmoid/ReLU mask activation) that estimates a soft mask over the encoder frames, \
         and (c) a 1D ConvTranspose decoder (mirror of the encoder) that reconstructs the \
         masked PCM. Every piece is composable from Vokra's existing Conv1D + ConvTranspose1D + \
         LayerNorm + PReLU primitives — what is missing is (i) the tensor-name walk from the \
         upstream `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k` state_dict prefixes to those \
         primitives' inputs (pending the manifest fetch — same posture as pyannote / Charsiu \
         real-weight bind), and (ii) the encoder-masker-decoder block composition itself. \
         Config: n_filters={n_filters}, n_kernel={n_kernel}, stride={stride}, \
         n_blocks={n_blocks}, n_repeats={n_repeats}, bn_chan={bn_chan}, hid_chan={hid_chan}, \
         skip_chan={skip_chan}, conv_kernel_size={conv_kernel_size}, \
         sample_rate={sample_rate}, n_src={n_src}, causal={causal}. Primary sources: \
         {asteroid} + {paper} + {hf}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は \
         fake-complete より honest') — no silent fabricated separator output ever emitted \
         (FR-EX-08).",
        n_filters = cfg.n_filters,
        n_kernel = cfg.n_kernel,
        stride = cfg.stride,
        n_blocks = cfg.n_blocks,
        n_repeats = cfg.n_repeats,
        bn_chan = cfg.bn_chan,
        hid_chan = cfg.hid_chan,
        skip_chan = cfg.skip_chan,
        conv_kernel_size = cfg.conv_kernel_size,
        sample_rate = cfg.sample_rate,
        n_src = cfg.n_src,
        causal = cfg.causal,
        asteroid = PRIMARY_SOURCE_ASTEROID,
        paper = PRIMARY_SOURCE_PAPER,
        hf = PRIMARY_SOURCE_HF,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Conv-TasNet runtime binder — primary-source axis
    //! pin + `from_gguf` metadata round-trip + negative-space round-
    //! trip on the loud-partial gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would be
    //! `separate(...)` returning real per-source PCM streams, but the
    //! encoder + TCN masker + decoder composition + tensor-name walk
    //! are all deferred (see the module doc + [`ConvTasnet::separate`]
    //! rustdoc). Fabricating a real-PCM output would violate CLAUDE.md
    //! 教訓 (a) ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config default pin**: `ConvTasnetConfig::asteroid_libri1mix_default`
    //!    matches the primary-source axes byte-for-byte (all 12 axes).
    //! 2. **Metadata round-trip**: `from_gguf` binds a legitimate GGUF
    //!    (arch + name + category + provenance license class + one
    //!    representative tensor), reads back the primary-source config
    //!    hold + license class + tensor count.
    //! 3. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / empty tensor list /
    //!    unsupported forward surface) fires at its documented
    //!    surface point, in the documented error variant.
    //! 4. **Arch-tag distinctness pin**: [`ARCH`] is deliberately
    //!    distinct from every sibling separator arch.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn stamp_test_topology(builder: &mut GgufBuilder, config: ConvTasnetConfig) {
        for (key, value) in [
            (KEY_N_FILTERS, config.n_filters),
            (KEY_N_KERNEL, config.n_kernel),
            (KEY_STRIDE, config.stride),
            (KEY_N_BLOCKS, config.n_blocks),
            (KEY_N_REPEATS, config.n_repeats),
            (KEY_BN_CHAN, config.bn_chan),
            (KEY_HID_CHAN, config.hid_chan),
            (KEY_SKIP_CHAN, config.skip_chan),
            (KEY_CONV_KERNEL_SIZE, config.conv_kernel_size),
            (KEY_SAMPLE_RATE, config.sample_rate),
            (KEY_N_SRC, config.n_src),
            (KEY_CAUSAL, config.causal),
        ] {
            builder.add_u32(key, value);
        }
    }

    /// Builds a Conv-TasNet GGUF carrying the arch tag + name +
    /// category + one representative encoder tensor. Optional
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn conv_tasnet_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        stamp_test_topology(&mut b, ConvTasnetConfig::asteroid_libri1mix_default());
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative encoder tensor so the non-emptiness gate
        // passes. Uses an upstream state-dict-like name so the naming
        // contract (verbatim key pass-through by the converter) is
        // exercised alongside.
        b.add_tensor(
            "encoder.filterbank.weight",
            GgmlType::F32,
            vec![512, 1, 16],
            vec![0u8; 512 * 16 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — ConvTasnetConfig default matches Asteroid Libri1Mix axes
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_matches_asteroid_libri1mix_axes() {
        // Pin the primary-source axes transcribed from
        // `asteroid/models/conv_tasnet.py` `ConvTasNet.__init__` + the
        // Libri1Mix `enhsingle_16k` recipe. A rename or axis-value
        // change would land here in the same commit or fail this test.
        let cfg = ConvTasnetConfig::asteroid_libri1mix_default();
        assert_eq!(cfg.n_filters, 512, "n_filters primary-source pin");
        assert_eq!(cfg.n_kernel, 16, "n_kernel primary-source pin");
        assert_eq!(cfg.stride, 8, "stride primary-source pin");
        assert_eq!(cfg.n_blocks, 8, "n_blocks primary-source pin");
        assert_eq!(cfg.n_repeats, 3, "n_repeats primary-source pin");
        assert_eq!(cfg.bn_chan, 128, "bn_chan primary-source pin");
        assert_eq!(cfg.hid_chan, 512, "hid_chan primary-source pin");
        assert_eq!(cfg.skip_chan, 128, "skip_chan primary-source pin");
        assert_eq!(
            cfg.conv_kernel_size, 3,
            "conv_kernel_size primary-source pin"
        );
        assert_eq!(cfg.sample_rate, 16000, "sample_rate primary-source pin");
        assert_eq!(
            cfg.n_src, 1,
            "n_src=1 for Libri1Mix `enhsingle` (single-source enhancement)"
        );
        assert_eq!(
            cfg.causal, 0,
            "causal=0 for the Asteroid non-causal (bidirectional cLN) enhancement recipe"
        );
        // Structural invariant: stride == n_kernel / 2 (50 % overlap
        // between successive encoder frames — Asteroid recipe pin).
        assert_eq!(
            cfg.stride,
            cfg.n_kernel / 2,
            "structural invariant: stride must equal n_kernel / 2 (50 % encoder-frame overlap)"
        );
        // Sanity: `Default` matches `asteroid_libri1mix_default` (both
        // must be primary-source-transcribed constants; no silent
        // divergence).
        assert_eq!(ConvTasnetConfig::default(), cfg);
    }

    // -----------------------------------------------------------------------
    // Test 2 — from_gguf metadata round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        // Build a legitimate GGUF (arch + name + category + provenance
        // license class + one representative tensor). The binder must
        // bind, hold the primary-source config, surface the Copyleft
        // license class, and report at least one tensor bound.
        let file = conv_tasnet_gguf(Some(LicenseClass::Copyleft));
        let ct = ConvTasnet::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip: the primary-source hold reads back with
        // every axis matching `asteroid_libri1mix_default`.
        assert_eq!(*ct.config(), ConvTasnetConfig::asteroid_libri1mix_default());
        // License-class surface: the Conv-TasNet converter defaults to
        // Copyleft per the HF card's `license: cc-by-sa-4.0` (SA cascade
        // — publish redistributable with LICENSE preserved, T3 tier).
        assert_eq!(
            ct.weight_license(),
            LicenseClass::Copyleft,
            "conv_tasnet converter defaults to Copyleft per HF card cc-by-sa-4.0 — the \
             runtime binder must surface it so the M2-13 compliance gate can enforce the \
             SA cascade on every derivative"
        );
        assert!(
            ct.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    #[test]
    fn from_gguf_rejects_missing_topology_key() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_tensor("x", GgmlType::F32, vec![1], vec![0; 4])
            .expect("add tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = ConvTasnet::from_gguf(&file).expect_err("metadata-free GGUF must fail");
        match err {
            VokraError::ModelLoad(message) => assert!(message.contains(KEY_N_FILTERS)),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `sepformer` GGUF handed to the Conv-TasNet binder by
        // mistake must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08). Conv-TasNet's stacked
        // dilated TCN and SepFormer's dual-path Transformer masker are
        // completely different topologies, so silent aliasing would
        // misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "sepformer");
        b.add_tensor("some.tensor", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ConvTasnet::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`sepformer`") && m.contains("`conv_tasnet`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("dual-path Transformer"),
                    "message should disambiguate SepFormer's dual-path Transformer masker \
                     from Conv-TasNet's fully-convolutional TCN masker, got `{m}`"
                );
                assert!(
                    m.contains("dilated TCN") || m.contains("stacked dilated"),
                    "message should call out Conv-TasNet's dilated-TCN topology so the \
                     reader knows why the arches are distinct, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — separate returns UnsupportedOp with three primary sources
    //          + names all three encoder-masker-decoder pieces + every
    //          config axis
    // -----------------------------------------------------------------------

    #[test]
    fn separate_loud_partial_returns_unsupported_op() {
        let file = conv_tasnet_gguf(Some(LicenseClass::Copyleft));
        let ct = ConvTasnet::from_gguf(&file).unwrap();
        // 1 s of 16 kHz mono silence — legitimate input shape, so the
        // loud-partial gate fires (not some pre-separate validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = ct.separate(&pcm) else {
            panic!("separate must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("conv_tasnet separate"),
                    "message must call out the conv_tasnet separate surface, got `{m}`"
                );
                // All three encoder-masker-decoder pieces must be named
                // so the follow-up wave knows the composition anchors.
                assert!(
                    m.contains("1D Conv encoder"),
                    "message must name the 1D Conv encoder piece, got `{m}`"
                );
                assert!(
                    m.contains("TCN") && m.contains("temporal convolutional network"),
                    "message must name the TCN (temporal convolutional network) masker piece, \
                     got `{m}`"
                );
                assert!(
                    m.contains("1D ConvTranspose decoder"),
                    "message must name the 1D ConvTranspose decoder piece, got `{m}`"
                );
                // All three primary-source anchors must be cited.
                assert!(
                    m.contains("asteroid-team/asteroid"),
                    "message must contain the Asteroid GitHub org anchor, got `{m}`"
                );
                assert!(
                    m.contains("1809.07454"),
                    "message must contain the arXiv paper id, got `{m}`"
                );
                assert!(
                    m.contains("JorisCos"),
                    "message must contain the JorisCos HF org anchor, got `{m}`"
                );
                // Every config axis must be echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(m.contains("n_filters=512"), "n_filters axis missing: {m}");
                assert!(m.contains("n_kernel=16"), "n_kernel axis missing: {m}");
                assert!(m.contains("stride=8"), "stride axis missing: {m}");
                assert!(m.contains("n_blocks=8"), "n_blocks axis missing: {m}");
                assert!(m.contains("n_repeats=3"), "n_repeats axis missing: {m}");
                assert!(m.contains("bn_chan=128"), "bn_chan axis missing: {m}");
                assert!(m.contains("hid_chan=512"), "hid_chan axis missing: {m}");
                assert!(m.contains("skip_chan=128"), "skip_chan axis missing: {m}");
                assert!(
                    m.contains("conv_kernel_size=3"),
                    "conv_kernel_size axis missing: {m}"
                );
                assert!(
                    m.contains("sample_rate=16000"),
                    "sample_rate axis missing: {m}"
                );
                assert!(m.contains("n_src=1"), "n_src axis missing: {m}");
                assert!(m.contains("causal=0"), "causal axis missing: {m}");
                // FR-EX-08 clause citation (the "no fabricated
                // separator output" leg).
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                // CLAUDE.md 教訓 (a) reference.
                assert!(
                    m.contains("教訓 (a)") || m.contains("loud-partial は fake-complete"),
                    "message must cite CLAUDE.md 教訓 (a), got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — arch tag distinct from sibling separator arches
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_separator_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. Every sibling separator arch tag
        // MUST NOT collide with ours — the masker topologies are
        // fundamentally different across families.
        assert_eq!(ARCH, "conv_tasnet");
        assert_eq!(NAME, "conv-tasnet-libri1mix");
        assert_eq!(CATEGORY, "enhancement");
        // Direct string comparisons against the sibling arch tags to
        // document the "which sibling should NOT be aliased" contract
        // at test time (a future rename of any sibling arch would
        // land here in the same commit or fail this test).
        assert_ne!(
            ARCH, "sepformer",
            "conv_tasnet (fully-convolutional TCN masker) and sepformer (dual-path Transformer \
             masker) are distinct separator arches — sharing arch tag would misroute the \
             runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "demucs",
            "conv_tasnet (time-domain TCN) and demucs (hybrid U-Net + spectrogram + \
             cross-domain attention) are distinct separator arches — sharing arch tag \
             would misroute the runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "tiger_separator",
            "conv_tasnet and tiger_separator (TIGER dual-path family) are distinct separator \
             arches — sharing arch tag would misroute the runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "bs_roformer",
            "conv_tasnet (time-domain TCN) and bs_roformer (frequency-domain band-split \
             Roformer) are distinct separator arches — sharing arch tag would misroute the \
             runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "mp_senet",
            "conv_tasnet (single-mask enhancement) and mp_senet (magnitude+phase speech \
             enhancement) are distinct separator arches — sharing arch tag would misroute \
             the runtime dispatch (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6 — from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch_chunk() {
        // A GGUF that carries no `vokra.model.arch` at all (e.g. a
        // hand-assembled fixture from an unrelated pipeline) must
        // fail loud rather than mis-bind.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "not-conv-tasnet");
        // No `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ConvTasnet::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("conv_tasnet"),
                    "message must name the conv_tasnet binder so a reader knows which \
                     loader complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — Empty tensor manifest fails loud (never binds all-zero
    //          forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch tag but zero tensors — the ConvTasnetWeights
        // non-emptiness gate must fire with an FR-EX-08 clause.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        stamp_test_topology(&mut b, ConvTasnetConfig::asteroid_libri1mix_default());
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ConvTasnet::from_gguf(&file) else {
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
                assert!(
                    m.contains("conv-tasnet-libri1mix"),
                    "message should name the converter --model slug so the reader knows how \
                     to re-produce the GGUF, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 (bonus) — Missing provenance stamp falls back to Unknown
    // -----------------------------------------------------------------------

    #[test]
    fn missing_provenance_stamp_falls_back_to_unknown_license_class() {
        // A GGUF that carries the arch tag + one tensor but NO
        // `vokra.provenance.weight_license` chunk must fall back to
        // `LicenseClass::Unknown` — the same fail-closed posture the
        // MT3 / Sortformer binders take. `Unknown` is refused at the
        // M2-13 compliance gate so a mis-stamped artifact cannot slip
        // past commercial-mode dispatch.
        let file = conv_tasnet_gguf(None);
        let ct = ConvTasnet::from_gguf(&file).expect("bind without provenance");
        assert_eq!(
            ct.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed at M2-13)"
        );
    }
}
