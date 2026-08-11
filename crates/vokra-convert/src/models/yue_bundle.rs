#![allow(clippy::doc_lazy_continuation)]
//! **YuE bundle** (`m-a-p/YuE-upsampler` + `m-a-p/xcodec_mini_infer`,
//! apache-2.0) safetensors → GGUF conversion (2026-08-01 Wave 3
//! sibling-pair codec + vocoder add).
//!
//! Input: the codec / vocoder half of the YuE full-song music-generation
//! system (Yuan et al. 2025, arXiv:2503.08638
//! `github.com/multimodal-art-projection/YuE`) — the six 7B stage-1 LLMs
//! and one 1B stage-2 LLM are OUT OF SCOPE for this converter (they are
//! text-conditioned song generators; this bundle carries only the codec
//! + vocoder that convert their token stream to PCM). Two sibling HF
//! repos share this converter, each mapped to a distinct
//! [`YueBundleVariant`]:
//!
//! ============================================   ======   =============================================
//! Variant                                        Size     Payload
//! ============================================   ======   =============================================
//! [`YueBundleVariant::Upsampler`]                145 MB   Vocos backbone + iSTFT head — 44.1 kHz
//!                                                          vocoder decoding YuE codec latents to PCM
//! [`YueBundleVariant::XcodecMini`]                2.2 GB   SoundStream RVQ codec (16 kHz, 25 Hz
//!                                                          frame rate, 640x downsample, up to 6 kbps)
//!                                                          + HuBERT-base semantic encoder + a byte-
//!                                                          identical copy of the same Vocos decoder
//! ============================================   ======   =============================================
//!
//! Both upstream repos ship **only** torch pickle `.pth` / `.bin` — no
//! `model.safetensors` mirror on either side (verified 2026-08-01 via
//! `https://huggingface.co/api/models/m-a-p/YuE-upsampler` and
//! `https://huggingface.co/api/models/m-a-p/xcodec_mini_infer`).
//! Callers pre-flatten to safetensors offline via
//! `tools/parity/yue_bundle_prepare_checkpoint.py` (a dedicated
//! uv-managed Python 3.12 bridge mirror of the multi-file
//! `naturalspeech3_facodec_prepare_checkpoint.py` +
//! `sepformer_prepare_checkpoint.py` + `bin_to_safetensors.py`
//! precedents). Vokra's Rust converter is safetensors-only by design so
//! the runtime never grows a pickle parser, keeping the NFR-DS-02
//! zero-dep posture (FR-LD-05 permanent, whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).
//!
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream state-dict name (for [`YueBundleVariant::Upsampler`] =
//! `backbone.*` / `head.*` from the Vocos decoder; for
//! [`YueBundleVariant::XcodecMini`] = the prep-script's role-prefixed
//! `codec.*` / `semantic.*` / `decoder.*` names — the prep script picks
//! these prefixes so a future `YueXcodecMini::from_gguf` can locate the
//! three sub-modules that share one repo). Plus the
//! `vokra.provenance.*` / `vokra.model.*` / `vokra.yue_bundle.variant`
//! metadata chunks a future native YuE codec / vocoder loader will read.
//!
//! # Provenance
//!
//! - **HF paths** (two distinct variants, each with its own ModelKind
//!   entry — mirror of the `Snac` collapse-into-one pattern is
//!   INTENTIONALLY NOT used here because these are two independent
//!   HF repos with two independent publish targets, not a per-frontend
//!   variant of one release):
//!   - `m-a-p/YuE-upsampler`      (145 MB, apache-2.0, last modified
//!     2025-03-12).
//!   - `m-a-p/xcodec_mini_infer`  (2.26 GB usedStorage, apache-2.0,
//!     last modified 2025-01-27).
//! - **License (SPDX)**: `apache-2.0` for both variants — verified
//!   2026-08-01 via HF cardData API `license: apache-2.0` on both
//!   repos (CLAUDE.md「ハルシネーション厳禁」). Upstream YuE code at
//!   `github.com/multimodal-art-projection/YuE` also ships apache-2.0.
//! - **Category**:
//!   - `vocoder` for [`YueBundleVariant::Upsampler`] — VocosBackbone +
//!     ISTFTHead maps codec-space latents → 44.1 kHz waveform. Same
//!     category tag as sibling `vocos` / `bigvgan` / `hifigan_vocoder`
//!     / `speecht5_hifigan`.
//!   - `codec` for [`YueBundleVariant::XcodecMini`] — waveform ↔
//!     discrete tokens (SoundStream RVQ family). Same category tag as
//!     sibling `mimi` / `dac` / `snac` / `wavtokenizer` / `neucodec`
//!     / `focalcodec` / `funcodec` / `moss_audio_tokenizer`.
//!
//! # YuE bundle vs sibling codecs / vocoders
//!
//! Distinct arch tags per variant so a downstream dispatcher never
//! silently mis-routes:
//!
//! - `yue_upsampler` — even though the Vocos-family topology
//!   (VocosBackbone + ISTFTHead) is technically identical to the
//!   sibling `vocos` (Charactr AI mel-24khz / encodec-24khz), the
//!   YuE upsampler is trained on YuE-specific codec latents at 44.1
//!   kHz output with a 3528-point iSTFT (n_fft=3528, hop_length=882
//!   → 50 Hz frame rate) — the config.yaml axes differ from every
//!   Charactr AI variant. Sharing `vocos` tag would trip a runtime
//!   binder that reads `n_fft` from an axes chunk and reshapes
//!   accordingly. The two upstream repos are distinct HF publish
//!   targets → distinct `vokra.model.name` → distinct arch tag.
//! - `yue_xcodec_mini` — SoundStream RVQ (n_filters=32, D=256,
//!   ratios=[8,5,4,2] → 640x downsample, sample_rate=16000, bins=1024,
//!   6 RVQ target bandwidths). Sibling **RVQ** codecs (Mimi / DAC /
//!   SNAC) share the same quantizer family but wrap different
//!   encoder/decoder backbones; sibling **FSQ** codecs (WavTokenizer /
//!   X-Codec 2 / neucodec / focalcodec) are a different quantizer
//!   family entirely. YuE xcodec-mini is a multi-part bundle
//!   (SoundStream RVQ + HuBERT semantic encoder + Vocos decoder head
//!   in one repo) — the semantic-encoder fusion is what distinguishes
//!   YuE from every plain sibling RVQ codec. Sharing an arch tag with
//!   `mimi` / `dac` / `snac` would mis-route to a codec-only decode
//!   path that has no semantic fusion input.
//!
//! # Byte-identical decoder between the two variants
//!
//! The two decoder files (`decoders/decoder_{131000,151000}.pth` in
//! `xcodec_mini_infer` and the top-level `decoder_{131000,151000}.pth`
//! in `YuE-upsampler`) are **byte-identical** — same xet
//! content-addressable hash (`c030b262…` for 131k / `70e4fbd9…` for
//! 151k, verified 2026-08-01 via HF API). YuE-upsampler is
//! essentially the standalone re-package of the same Vocos decoder
//! head, published separately for callers who need only the vocoder
//! (145 MB) not the full 2.2 GB bundle. This converter honors the
//! upstream release model — two distinct ModelKind entries + two
//! distinct publish repos + two distinct §3.1 sign-off rows — even
//! though the underlying decoder weights are shared. Silently
//! collapsing them would misalign the model-card story with what
//! the upstream org publishes.
//!
//! # Snapshot selection (131k vs 151k)
//!
//! Both upstream repos ship two training snapshots: `_131000.pth` and
//! `_151000.pth`. The 151k snapshot is the later training step and is
//! typically the "final" one. The prep bridge accepts
//! `--snapshot 131000|151000` and defaults to 151000. This converter
//! never sees the difference — the prep bridge picks one and re-emits
//! it under the bare `backbone.*` / `head.*` module names (for the
//! upsampler) or under the `decoder.*` role prefix (for the xcodec-mini
//! bundle, since the same file must coexist with `codec.*` and
//! `semantic.*` prefixed tensors in one merged safetensors).
//!
//! # Semantic encoder is HuBERT-base
//!
//! The `semantic_ckpts/hf_1_325000/pytorch_model.bin` sub-part of
//! `xcodec_mini_infer` is a standard HF-transformers `HubertModel`
//! (hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
//! conv_stride=[5,2,2,2,2,2,2], intermediate_size=3072, do_normalize=true,
//! sampling_rate=16000). YuE's xcodec-mini fuses semantic tokens
//! (from this encoder) with acoustic RVQ codes (from the SoundStream
//! codec) for its codec token stream — this is what distinguishes it
//! from a plain SoundStream codec. The Rust runtime can either
//! re-implement the HuBERT forward natively (whisper.cpp 型) or
//! delegate to the sibling wav2vec2/HuBERT-family binder when that
//! lands; today's converter surface is byte-exact tensor-name
//! preservation only.
//!
//! # Upstream source-tree attribution (RepCodec + descript-audio-codec)
//!
//! `xcodec_mini_infer` ships full source-tree copies of RepCodec
//! (ByteDance/Chutong Meng, MIT) and Descript-Audio-Codec (MIT) at
//! `RepCodec/` and `descriptaudiocodec/dac/` — these are inference-tree
//! artefacts of the upstream release process, **not** loaded at
//! runtime and **not** touched by this converter. The prep bridge
//! must NOT recurse into these subtrees. NOTICE credit is preserved
//! because their code informed the YuE codec design, but no weights
//! are lifted from them.
//!
//! # BF16 pass-through (mirror of vocos / snac / focalcodec)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`YueBundleReport::bf16_passthrough`] records how many
//! BF16 tensors landed on this arm so a silent widen / downcast
//! cannot slip in undetected. Both upstream releases are F32 at rest
//! (verified 2026-08-01), so the BF16 arm is defensive today; the
//! counter is kept for future BF16-quantized derivative releases.
//!
//! # Vocos quantization warning (CLAUDE.md 設計判断 §Vocos)
//!
//! The upsampler variant's Vocos-family topology inherits the
//! CLAUDE.md pin **Vocos INT8-fragile**: 「Vocos は量子化耐性弱 (INT8
//! 崩壊) → fp16 必須」. This converter never emits INT8 (the K-quant
//! path is Whisper-only per `main.rs --quantize` guard); BF16 is
//! expected to be safe (BF16 loss is mantissa-only, not
//! activation-crushing INT8 saturation), but no parity data yet — an
//! owner-side follow-up when the runtime binder lands. The xcodec-mini
//! variant's SoundStream RVQ codec is not subject to the Vocos
//! INT8 warning per se, but its embedded Vocos decoder head is; the
//! K-quant refusal is a global converter guard so this is moot.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict names verbatim**
//! (the vocos / snac / focalcodec / speecht5_hifigan / neucodec
//! contract). Real-weight parity vs the upstream YuE Python
//! inference pipeline
//! (`github.com/multimodal-art-projection/YuE/inference/xcodec_mini_infer/`)
//! is deferred to owner (`docs/license-audit.md` §3.1 sign-off queue).
//!
//! # No ONNX / no pickle in runtime (permanent)
//!
//! Both upstream repos ship PyTorch pickle checkpoints only; this
//! converter **never** touches ONNX (FR-LD-05) and **never** touches
//! pickle (NFR-DS-02 zero-dep). The pipeline is re-implemented
//! natively when the runtime binders land (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Loud-partial precedent
//!
//! Real-weight forward binding is deferred: the runtime consumer will
//! walk the emitted tensor names and either succeed or fail loudly per
//! FR-EX-08. Today's converter surface is byte-exact provenance +
//! tensor-name preservation only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for the YuE-upsampler GGUF (Vocos backbone +
/// iSTFT head trained on YuE codec latents, 44.1 kHz output).
///
/// Intentionally distinct from Charactr AI `vocos`
/// (`crates/vokra-convert/src/models/vocos.rs`) even though the
/// backbone family is Vocos: the YuE-upsampler ships different
/// config axes (n_fft=3528, hop_length=882, 44.1 kHz output, 8-layer
/// backbone) and a different training corpus (YuE codec-latent inputs
/// vs mel or EnCodec inputs). Silently sharing arch would mis-route
/// the runtime dispatch.
pub const ARCH_UPSAMPLER: &str = "yue_upsampler";

/// `vokra.model.arch` for the YuE xcodec-mini GGUF (SoundStream RVQ
/// codec + HuBERT semantic encoder + Vocos decoder head bundle at
/// 16 kHz / 25 Hz frame rate).
///
/// Intentionally distinct from every sibling codec (`mimi` / `dac` /
/// `snac` / `wavtokenizer` / `neucodec` / `xcodec2` / `focalcodec` /
/// `funcodec` / `speechtokenizer` / `bicodec` / `xy_tokenizer` /
/// `step_audio2_mini` / `moss_audio_tokenizer` / `facodec`) — YuE
/// xcodec-mini is a multi-part bundle (codec + semantic encoder +
/// vocoder decoder head), the semantic-encoder fusion is what
/// distinguishes it from plain RVQ / FSQ codecs.
pub const ARCH_XCODEC_MINI: &str = "yue_xcodec_mini";

/// `vokra.model.category` value for the [`YueBundleVariant::Upsampler`]
/// GGUF.
pub const CATEGORY_VOCODER: &str = "vocoder";

/// `vokra.model.category` value for the [`YueBundleVariant::XcodecMini`]
/// GGUF.
pub const CATEGORY_CODEC: &str = "codec";

/// Default upstream weight license (SPDX). Both `m-a-p/YuE-upsampler`
/// and `m-a-p/xcodec_mini_infer` ship apache-2.0 end-to-end (HF
/// cardData API verified 2026-08-01; upstream YuE GitHub also
/// apache-2.0).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// `vokra.model.name` value written by the enum-arm-default dispatch
/// (used from `lib.rs`'s `ModelKind::YueUpsampler` arm). Callers that
/// need to override use [`convert_yue_bundle_variant_file`] directly.
#[allow(dead_code)]
pub const NAME_UPSAMPLER: &str = "yue-upsampler";

/// `vokra.model.name` value written by the enum-arm-default dispatch
/// (used from `lib.rs`'s `ModelKind::YueXcodecMini` arm).
#[allow(dead_code)]
pub const NAME_XCODEC_MINI: &str = "yue-xcodec-mini";

/// `vokra.provenance.upstream_hf` for the [`YueBundleVariant::Upsampler`]
/// variant (backward-compat / discovery alias).
#[allow(dead_code)]
pub const UPSTREAM_HF_UPSAMPLER: &str = "m-a-p/YuE-upsampler";

/// `vokra.provenance.upstream_hf` for the [`YueBundleVariant::XcodecMini`]
/// variant.
#[allow(dead_code)]
pub const UPSTREAM_HF_XCODEC_MINI: &str = "m-a-p/xcodec_mini_infer";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.yue_bundle.variant`: `"upsampler"` / `"xcodec_mini"`.
/// Consumers pick a specific bundle member without parsing free-text
/// `vokra.model.name` (mirrors `vokra.snac.variant` /
/// `vokra.focalcodec.variant`).
pub const KEY_YUE_BUNDLE_VARIANT: &str = "vokra.yue_bundle.variant";

/// Which YuE bundle repo the caller is converting. Selects the model
/// name / upstream HF slug / category / arch tag / variant tag
/// written into the GGUF.
///
/// The two variants ship in **separate** HF repos (`m-a-p/YuE-upsampler`
/// and `m-a-p/xcodec_mini_infer`) with distinct scopes:
/// [`Self::Upsampler`] carries only the Vocos vocoder head (145 MB);
/// [`Self::XcodecMini`] carries the full SoundStream RVQ codec +
/// HuBERT semantic encoder + a byte-identical copy of the same
/// Vocos decoder (2.2 GB). Each variant becomes a distinct
/// [`crate::ModelKind`] entry + distinct publish target — mirror of
/// the Charactr AI `Vocos` mel-24khz / encodec-24khz posture is
/// INTENTIONALLY NOT used because these are two independent HF org
/// releases, not two frontends of one release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YueBundleVariant {
    /// `m-a-p/YuE-upsampler`: Vocos vocoder head (backbone + iSTFT
    /// head, input_channels=1024, dim=512, intermediate_dim=1536,
    /// num_layers=8; n_fft=3528, hop_length=882, 44.1 kHz output).
    /// Category = `vocoder`. `vokra.yue_bundle.variant = "upsampler"`.
    Upsampler,
    /// `m-a-p/xcodec_mini_infer`: SoundStream RVQ codec
    /// (n_filters=32, D=256, ratios=[8,5,4,2] → 640x downsample,
    /// sample_rate=16000, bins=1024, target_bandwidths=[0.5, 1, 1.5,
    /// 2, 4, 6] kbps) + HuBERT-base semantic encoder (hidden_size=768,
    /// 12 layers, sampling_rate=16000, do_normalize=true) + Vocos
    /// decoder head (byte-identical to the Upsampler variant). All
    /// three sub-parts share this one repo, prep-script role-prefixes
    /// tensors under `codec.*` / `semantic.*` / `decoder.*`.
    /// Category = `codec`. `vokra.yue_bundle.variant = "xcodec_mini"`.
    XcodecMini,
}

impl YueBundleVariant {
    /// The `vokra.model.name` string for this variant.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Upsampler => NAME_UPSAMPLER,
            Self::XcodecMini => NAME_XCODEC_MINI,
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for this
    /// variant — the primary redistribution source the model-card
    /// generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Upsampler => UPSTREAM_HF_UPSAMPLER,
            Self::XcodecMini => UPSTREAM_HF_XCODEC_MINI,
        }
    }

    /// The `vokra.yue_bundle.variant` short tag written under
    /// [`KEY_YUE_BUNDLE_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Upsampler => "upsampler",
            Self::XcodecMini => "xcodec_mini",
        }
    }

    /// The `vokra.model.arch` tag for this variant.
    pub const fn arch(self) -> &'static str {
        match self {
            Self::Upsampler => ARCH_UPSAMPLER,
            Self::XcodecMini => ARCH_XCODEC_MINI,
        }
    }

    /// The `vokra.model.category` tag for this variant (vocoder vs
    /// codec — the model-card generator classifier fans out on this).
    pub const fn category(self) -> &'static str {
        match self {
            Self::Upsampler => CATEGORY_VOCODER,
            Self::XcodecMini => CATEGORY_CODEC,
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s `source`
    /// argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Upsampler => {
                "m-a-p/YuE-upsampler (YuE Vocos vocoder head, VocosBackbone \
                 8-layer + ISTFTHead n_fft=3528 hop=882 @ 44.1 kHz, apache-2.0)"
            }
            Self::XcodecMini => {
                "m-a-p/xcodec_mini_infer (YuE xcodec-mini bundle = SoundStream \
                 RVQ codec 25 Hz + HuBERT-base semantic encoder + Vocos decoder \
                 head, apache-2.0)"
            }
        }
    }
}

/// Outcome of a YuE bundle conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::vocos::VocosReport`, `super::snac::SnacReport`,
/// `super::focalcodec::FocalcodecReport`) adapted to the
/// variant-taking [`convert_yue_bundle_variant_file`] surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct YueBundleReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
    /// Which YuE bundle variant was written.
    pub variant: Option<YueBundleVariant>,
}

/// Backward-compat entry: convert a YuE bundle safetensors checkpoint,
/// defaulting to the [`YueBundleVariant::Upsampler`] variant (the
/// smaller and simpler of the pair — matches the sibling default-
/// canonical dispatch convention where "the canonical short name
/// picks the default variant").
///
/// New callers should prefer [`convert_yue_bundle_variant_file`] with
/// an explicit [`YueBundleVariant`] argument.
///
/// # Errors
///
/// As [`convert_yue_bundle_variant_file`].
#[allow(dead_code)]
pub fn convert_yue_bundle_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<YueBundleReport, ConvertError> {
    convert_yue_bundle_variant_file(input, output, YueBundleVariant::Upsampler, license)
}

/// Converts a YuE bundle safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, tagging the emitted GGUF as the
/// supplied [`YueBundleVariant`] (mirror of `convert_vocos_file` /
/// `convert_snac_file` / `convert_focalcodec_file`).
///
/// **Prerequisite**: both upstream repos ship torch pickle only
/// (`.pth` / `.bin`, no `model.safetensors` mirror). Callers pre-flatten
/// to safetensors via `tools/parity/yue_bundle_prepare_checkpoint.py`
/// (a dedicated uv-managed Python 3.12 bridge mirror of the
/// `naturalspeech3_facodec_prepare_checkpoint.py` multi-file precedent)
/// before invoking this converter — no pickle parser enters the Vokra
/// runtime (NFR-DS-02 / FR-LD-05).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` (arch / name / category),
/// `vokra.provenance.*` (weight_license / license / model_id / source
/// / upstream_hf), and `vokra.yue_bundle.variant` chunks are stamped
/// for the runtime compliance gate (FR-CP-03) and shape-checked config
/// dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`) — both
/// upstream YuE HF releases ship apache-2.0.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input.
pub fn convert_yue_bundle_variant_file(
    input: &Path,
    output: &Path,
    variant: YueBundleVariant,
    license: Option<&str>,
) -> Result<YueBundleReport, ConvertError> {
    // Upsampler (~145 MB after prep flatten) and XcodecMini merged
    // safetensors (~1.88 GB of unique weights = codec 1.36 GB +
    // HuBERT 377 MB + Vocos decoder 145 MB; the prep script may or
    // may not include the byte-identical Vocos decoder depending on
    // caller choice) are both below the memory
    // [[feedback-large-models-on-vast-ai]] ≥8 GB vast.ai threshold,
    // so the simple `std::fs::read` posture the sibling non-streaming
    // BF16 pass-through converters use applies (both fit comfortably
    // on the M1 iMac 16 GB local converter host).
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, variant.arch());
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, variant.category());
    b.add_string(KEY_YUE_BUNDLE_VARIANT, variant.tag());

    // Default provenance stamp — Permissive apache-2.0 end-to-end
    // (both upstream YuE model cards verified via HF API 2026-08-01,
    // upstream `github.com/multimodal-art-projection/YuE` also
    // apache-2.0). The optional `license` argument overrides below.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.source_description()),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());

    let mut report = YueBundleReport {
        variant: Some(variant),
        ..YueBundleReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // ADR (mirror of vocos / snac / focalcodec / speecht5_hifigan);
    // the runtime widens BF16 → f32 exactly at load via the single
    // choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a single-BF16-tensor safetensors buffer with a caller-supplied
    /// raw payload.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Builds an F32 tensor safetensors buffer — matches upstream YuE dtype
    /// (torch-native F32 pickles for both variants).
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 4;
        assert_eq!(f32_bytes.len(), expected);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-yue-bundle-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn arch_strings_are_distinct_and_never_alias_siblings() {
        // A defensive pin: silently sharing an arch tag with a
        // sibling Vocos vocoder (Charactr AI mel-24khz / encodec-24khz)
        // or a sibling RVQ / FSQ codec would mis-route runtime
        // dispatch to a wrong-shape forward.
        assert_eq!(ARCH_UPSAMPLER, "yue_upsampler");
        assert_eq!(ARCH_XCODEC_MINI, "yue_xcodec_mini");
        assert_ne!(ARCH_UPSAMPLER, ARCH_XCODEC_MINI);
        for sibling in [
            "vocos",
            "bigvgan",
            "hifigan_vocoder",
            "speecht5_hifigan",
            "mimi",
            "dac",
            "snac",
            "wavtokenizer",
            "neucodec",
            "xcodec2",
            "focalcodec",
            "funcodec",
            "speechtokenizer",
            "bicodec",
            "xy_tokenizer",
            "step_audio2_mini",
            "moss_audio_tokenizer",
            "facodec",
        ] {
            assert_ne!(
                ARCH_UPSAMPLER, sibling,
                "upsampler must not alias {sibling}"
            );
            assert_ne!(
                ARCH_XCODEC_MINI, sibling,
                "xcodec_mini must not alias {sibling}"
            );
        }
    }

    #[test]
    fn f32_upsampler_tensor_passes_through_and_stamps_land() {
        // Upstream YuE-upsampler is F32 (torch-native pickle) — this
        // test pins the primary code path for the vocoder variant.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from Vocos's
        // backbone body (upstream `backbone.norm.weight` is the final
        // LayerNorm of the ConvNeXt V2 stack).
        let input_bytes = safetensors_one_f32("backbone.norm.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("upsampler-f32-in", &input_bytes);
        let output_path = write_temp("upsampler-f32-out", &[]);

        let report = convert_yue_bundle_variant_file(
            &input_path,
            &output_path,
            YueBundleVariant::Upsampler,
            None,
        )
        .expect("convert_yue_bundle_variant_file must accept an F32 upsampler checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );
        assert_eq!(report.variant, Some(YueBundleVariant::Upsampler));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("backbone.norm.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance / category / arch / variant chunks landed.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH_UPSAMPLER)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_UPSAMPLER)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_UPSAMPLER)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY_VOCODER),
            "vokra.model.category must be `vocoder` for the Upsampler variant"
        );
        assert_eq!(
            file.get(KEY_YUE_BUNDLE_VARIANT).and_then(|v| v.as_str()),
            Some("upsampler")
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn xcodec_mini_variant_emits_distinct_stamps() {
        // The XcodecMini variant reuses the same converter body but
        // the arch / name / variant / upstream / category stamps all
        // differ. Silently sharing stamps would misroute a downstream
        // loader that dispatches on `vokra.model.arch` OR
        // `vokra.model.category` (the vocoder-vs-codec split matters).
        let f32_bytes: Vec<u8> = [7.0_f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Realistic upstream tensor name from the prep-script's
        // role-prefixed `codec.*` namespace (SoundStream encoder body).
        let input_bytes = safetensors_one_f32(
            "codec.encoder.model.0.block.0.conv.weight",
            &[1, 2],
            &f32_bytes,
        );
        let input_path = write_temp("xcodec-mini-in", &input_bytes);
        let output_path = write_temp("xcodec-mini-out", &[]);

        let report = convert_yue_bundle_variant_file(
            &input_path,
            &output_path,
            YueBundleVariant::XcodecMini,
            None,
        )
        .expect("convert XcodecMini variant");
        assert_eq!(report.variant, Some(YueBundleVariant::XcodecMini));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH_XCODEC_MINI),
            "XcodecMini must emit its own arch tag, not the Upsampler arch"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_XCODEC_MINI),
            "XcodecMini must emit its own model.name, not fall back to Upsampler"
        );
        assert_eq!(
            file.get(KEY_YUE_BUNDLE_VARIANT).and_then(|v| v.as_str()),
            Some("xcodec_mini")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_XCODEC_MINI)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY_CODEC),
            "vokra.model.category must be `codec` for the XcodecMini variant"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16-quantized derivatives should
        // ride the same arm as the sibling BF16-pass-through
        // converters (vocos / snac / focalcodec). Non-zero BF16 bit
        // patterns so a subsequent byte-identity assert catches any
        // silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic upstream Vocos iSTFT head tensor name.
        let input_bytes = safetensors_one_bf16("head.out.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_yue_bundle_variant_file(
            &input_path,
            &output_path,
            YueBundleVariant::Upsampler,
            None,
        )
        .expect("convert BF16 upsampler");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("head.out.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_flows_through() {
        // A user who re-trained on a permissive corpus supplies a
        // different SPDX id at conversion time — the override must
        // land on KEY_PROVENANCE_LICENSE + KEY_PROVENANCE_WEIGHT_LICENSE
        // and the LicenseClass must be re-derived by from_license_str.
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("backbone.pos_embed", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_yue_bundle_variant_file(
            &input_path,
            &output_path,
            YueBundleVariant::Upsampler,
            Some("mit"),
        )
        .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT is Permissive class (same as apache-2.0)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Backward-compat entry defaults to the Upsampler variant — the
    /// smaller and simpler of the pair matches the sibling
    /// default-canonical dispatch convention.
    #[test]
    fn backward_compat_entry_defaults_to_upsampler() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("backbone.norm.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("compat-in", &input_bytes);
        let output_path = write_temp("compat-out", &[]);

        let report = convert_yue_bundle_file(&input_path, &output_path, None)
            .expect("backward-compat entry must succeed");
        assert_eq!(report.variant, Some(YueBundleVariant::Upsampler));

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Every enum variant maps to a distinct `(name, tag, arch,
    /// category, upstream_hf, source_description)` tuple — a defensive
    /// pin against a copy-paste that would silently re-use one
    /// variant's strings for the other (matches the focalcodec /
    /// snac / vocos `every_variant_has_distinct_stamps` precedent).
    #[test]
    fn every_variant_has_distinct_stamps() {
        let variants = [YueBundleVariant::Upsampler, YueBundleVariant::XcodecMini];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let a = variants[i];
                let b = variants[j];
                assert_ne!(a.name(), b.name(), "names must differ ({a:?} vs {b:?})");
                assert_ne!(a.tag(), b.tag(), "tags must differ ({a:?} vs {b:?})");
                assert_ne!(a.arch(), b.arch(), "arch must differ ({a:?} vs {b:?})");
                assert_ne!(
                    a.category(),
                    b.category(),
                    "category must differ ({a:?} vs {b:?}) — vocoder vs codec is the whole point"
                );
                assert_ne!(
                    a.upstream_hf(),
                    b.upstream_hf(),
                    "upstream_hf must differ ({a:?} vs {b:?})"
                );
                assert_ne!(
                    a.source_description(),
                    b.source_description(),
                    "source_description must differ ({a:?} vs {b:?})"
                );
            }
        }
    }
}
