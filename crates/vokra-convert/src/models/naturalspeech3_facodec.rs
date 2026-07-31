#![allow(clippy::doc_lazy_continuation)]
//! **NaturalSpeech 3 FACodec** (Amphion, `amphion/naturalspeech3_facodec`,
//! apache-2.0): pytorch-pickle `.bin` → safetensors → GGUF conversion
//! (Wave 3 codec add, 2026-08-01).
//!
//! Input: the upstream `amphion/naturalspeech3_facodec` release — a
//! **factorized VQ** neural audio codec at 16 kHz with `hop_size = 200`
//! (80 tokens/sec) from Amphion/NaturalSpeech 3 (Ju et al. 2024,
//! arXiv:2403.03100). Unlike sibling RVQ codecs (Mimi / DAC / SNAC) or
//! FSQ codecs (WavTokenizer / X-Codec 2 / FocalCodec), FACodec runs
//! **3 factorized quantizer heads in PARALLEL over disentangled
//! subspaces** — prosody (1 codebook), content (2 codebooks), detail
//! (3 codebooks) — for a total of 6 codebooks fed into the decoder.
//! This is the fundamental topology difference vs every prior codec
//! in the tree: silently sharing an arch tag with any sibling codec
//! would misroute the runtime dispatch to a wrong-shape forward.
//!
//! The upstream release ships **5 separate `torch.save()` pickle `.bin`
//! files** at repo root (no `model.safetensors` mirror, no `config.json`
//! — hparams live in the paper + Amphion source at
//! `github.com/open-mmlab/Amphion/tree/main/models/codec/ns3_codec`):
//! `ns3_facodec_encoder.bin` (~16.9 MB) / `ns3_facodec_encoder_v2.bin`
//! (~17.1 MB) / `ns3_facodec_decoder.bin` (~398 MB) /
//! `ns3_facodec_decoder_v2.bin` (~432 MB) / `ns3_facodec_redecoder.bin`
//! (~151 MB, zero-shot voice conversion). Callers pre-flatten the
//! caller-selected subset to a single safetensors offline via
//! `tools/parity/naturalspeech3_facodec_prepare_checkpoint.py` (a
//! dedicated uv-managed Python 3.12 bridge mirror of
//! `sepformer_prepare_checkpoint.py` — same multi-file merge shape,
//! same INT-dtype filter, same `.stripped-manifest.json` sidecar,
//! same fail-loud posture). Vokra's Rust converter is safetensors-only
//! by design so the runtime never grows a pickle parser, keeping the
//! NFR-DS-02 zero-dep posture (FR-LD-05 permanent).
//!
//! Output: a GGUF carrying every float tensor verbatim under a
//! role-prefixed upstream state-dict name (`encoder.*` / `decoder.*` /
//! `redecoder.*` — the prep script decides these prefixes since the
//! upstream `.bin` files ship un-prefixed flat state dicts), plus the
//! `vokra.provenance.*` / `vokra.model.*` / `vokra.facodec.*` metadata
//! chunks a future native FACodec loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `amphion/naturalspeech3_facodec`.
//! - **License (SPDX)**: `apache-2.0` — end-to-end (Amphion GitHub
//!   `open-mmlab/Amphion/LICENSE` and HF `amphion/naturalspeech3_facodec`
//!   cardData both declare apache-2.0, verified 2026-08-01 via HF
//!   cardData API — CLAUDE.md「ハルシネーション厳禁」).
//! - **Category**: `codec` — audio codec (waveform → 6-parallel-codebook
//!   FVQ tokens → waveform). Same category tag as sibling Mimi / DAC /
//!   SNAC / WavTokenizer / X-Codec 2 / FunCodec / FocalCodec (all
//!   `codec`), used by the model-card generator classifier.
//! - **Variant tag** (`vokra.facodec.variant`): `"v1"` / `"v2"` /
//!   `"redecoder-v1"` / `"redecoder-v2"` so a consumer that needs to
//!   pick a specific pair can inspect this without parsing free-text
//!   `vokra.model.name` (mirrors `vokra.snac.variant` +
//!   `vokra.focalcodec.variant`).
//!
//! # FACodec vs sibling codecs
//!
//! Distinct arch tag from every sibling codec (Mimi / DAC /
//! WavTokenizer / SNAC / X-Codec 2 / FunCodec / FocalCodec /
//! SpeechTokenizer / neucodec / BiCodec / XyTokenizer /
//! Step-Audio2-Mini / MOSS-Audio-Tokenizer):
//!
//! - **RVQ codecs** (Mimi / DAC / SNAC): residual vector quantization,
//!   sequential residual chain across N codebooks. FACodec runs 6
//!   codebooks in PARALLEL over 3 disentangled subspaces —
//!   fundamentally different decode structure.
//! - **FSQ codecs** (WavTokenizer / X-Codec 2): single-codebook finite
//!   scalar quantization. FACodec is neither RVQ nor FSQ — it is
//!   factorized VQ (FVQ), a third quantizer family.
//! - **FocalCodec**: focal-modulation single-codebook, distinct family
//!   entirely.
//!
//! The `ARCH = "facodec"` tag is intentionally distinct — silently
//! sharing an arch with a sibling codec would mis-route the runtime
//! dispatch to a decode structure that expects sequential residual
//! sum (RVQ) or single-lookup (FSQ) instead of factorized parallel
//! subspaces.
//!
//! # BF16 pass-through (mirror of snac / neucodec / moss_audio_tokenizer)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at load
//! via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`FacodecReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm so a silent widen / downcast cannot
//! slip in undetected. Upstream `.bin` files are F32 at rest
//! (verified 2026-08-01 — Amphion training pipeline defaults to F32),
//! so the BF16 arm is defensive today; the counter is kept for future
//! BF16-quantized derivative releases.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **role-prefixed upstream state-dict names
//! verbatim** — the prep script adds the `encoder.` / `decoder.` /
//! `redecoder.` prefix since the upstream `.bin` files ship un-prefixed
//! flat state dicts (contrast the CSM / Kokoro / SNAC / neucodec /
//! MOSS-Audio-Tokenizer contract where upstream tensor names already
//! carry sub-module paths). Real-weight parity vs the upstream Amphion
//! Python reference is deferred to owner (`docs/license-audit.md` §3.1
//! sign-off queue); the runtime consumer will walk the emitted tensor
//! names and either succeed or fail loudly per FR-EX-08.
//!
//! # Voice-conversion policy note (redecoder variants)
//!
//! FACodec itself is a codec (encoder + decoder + FVQ), NOT a
//! voice-clone trigger model like RVC v2 / GPT-SoVITS which live in
//! `vokra-voiceclone-experimental` (CLAUDE.md 設計判断 8, ELVIS Act /
//! NO FAKES Act policy). However the **redecoder variants**
//! (`RedecoderV1` / `RedecoderV2`) specifically enable zero-shot voice
//! conversion by swapping the timbre subspace while preserving prosody
//! + content codes. Whether the redecoder variants should be published
//! in the main `ayutaz/vokra` org or gated to
//! `vokra-voiceclone-experimental` per the same policy that pushed
//! openvoice_v2 / knn_vc / freevc / meanvc into the separate repo is
//! an **owner routing decision**; this converter emits the artifact but
//! does not decide where it lands. The base V1/V2 variants (encoder +
//! decoder only, no redecoder) are unambiguously codec-class and
//! belong in the main zoo.
//!
//! # No ONNX / no pickle in runtime (permanent)
//!
//! FACodec ships PyTorch pickle `.bin` files; this converter **never**
//! touches ONNX (FR-LD-05) and **never** touches pickle (NFR-DS-02
//! zero-dep). The pipeline will be re-implemented natively in a future
//! `crates/vokra-models/src/facodec/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4). Between now and that
//! landing, the runtime consumer walks the emitted tensor names and
//! either succeeds or fails loudly per FR-EX-08 — today's converter
//! surface is byte-exact provenance + tensor-name preservation only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for NaturalSpeech 3 FACodec GGUFs. Shared across
/// every [`FacodecVariant`] — upstream's Amphion FACodec class routes
/// every (encoder,decoder,[redecoder]) tuple to the same forward; the
/// topology (encoder / decoder body + 3-way factorized VQ head
/// {prosody 1cb, content 2cb, detail 3cb} + optional redecoder for
/// zero-shot VC) is structurally identical, only which upstream
/// checkpoint pair the tensors come from differs.
///
/// Intentionally distinct from every sibling codec (`mimi`, `dac`,
/// `wavtokenizer`, `snac`, `xcodec2`, `focalcodec`, `speechtokenizer`,
/// `neucodec`, `bicodec`, `xy_tokenizer`, `funcodec`,
/// `step_audio2_mini`, `moss_audio_tokenizer`) — FACodec is the first
/// factorized-VQ (FVQ) codec in the tree; RVQ / FSQ / focal-modulation
/// siblings expect a fundamentally different decode structure.
pub const ARCH: &str = "facodec";

/// `vokra.model.name` value written for the canonical (default `V2`)
/// variant (backward-compat alias — new callers should use
/// [`FacodecVariant::name`]).
#[allow(dead_code)]
pub const NAME: &str = "naturalspeech3-facodec-v2";

/// `vokra.model.category` value written for every FACodec GGUF. Same
/// tag as sibling `mimi` / `dac` / `wavtokenizer` / `snac` /
/// `focalcodec` / `neucodec` / `moss_audio_tokenizer` (all `codec`),
/// used by the model-card generator classifier.
pub const CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator. All four variants share
/// this single HF repo (the variant selects which of the 5 `.bin`
/// files in the repo the prep script merges).
#[allow(dead_code)]
pub const UPSTREAM_HF: &str = "amphion/naturalspeech3_facodec";

/// Default upstream weight licence (SPDX). Amphion NaturalSpeech 3
/// FACodec ships apache-2.0 end-to-end — HF cardData API
/// `license: apache-2.0` on `amphion/naturalspeech3_facodec` and
/// GitHub `open-mmlab/Amphion/LICENSE` = Apache-2.0 both verified
/// 2026-08-01 (CLAUDE.md「ハルシネーション厳禁」).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of the sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// `vokra.facodec.variant`: `"v1"` / `"v2"` / `"redecoder-v1"` /
/// `"redecoder-v2"`. Consumers pick a specific encoder+decoder pair
/// (+ optional redecoder) without parsing free-text
/// `vokra.model.name` (mirrors [`super::snac`] +
/// [`super::focalcodec`] discriminators).
pub const KEY_FACODEC_VARIANT: &str = "vokra.facodec.variant";

// Frontend / factorization chunk keys — the Amphion FACodec release
// carries no `config.json`; these values are transcribed from the paper
// (arXiv:2403.03100) + Amphion source
// (`open-mmlab/Amphion/models/codec/ns3_codec`) 2026-08-01.
const KEY_SAMPLE_RATE: &str = "vokra.facodec.sample_rate";
const KEY_HOP_SIZE: &str = "vokra.facodec.hop_size";
const KEY_N_Q_PROSODY: &str = "vokra.facodec.n_quantizers_prosody";
const KEY_N_Q_CONTENT: &str = "vokra.facodec.n_quantizers_content";
const KEY_N_Q_DETAIL: &str = "vokra.facodec.n_quantizers_detail";

// Transcribed constants (from paper arXiv:2403.03100 + Amphion source
// tree at github.com/open-mmlab/Amphion/tree/main/models/codec/ns3_codec,
// fetched 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」).
const SAMPLE_RATE: u32 = 16_000; // 16 kHz per paper + Amphion default
const HOP_SIZE: u32 = 200; // 16000/200 = 80 tok/s per subspace
const N_Q_PROSODY: u32 = 1; // 1 codebook for the prosody subspace
const N_Q_CONTENT: u32 = 2; // 2 codebooks for the content subspace
const N_Q_DETAIL: u32 = 3; // 3 codebooks for the detail subspace
// → 6 total codebooks over 3 factorized subspaces

/// Compile-time sanity: FACodec is defined by exactly 6 codebooks split
/// 1 (prosody) + 2 (content) + 3 (detail) across 3 factorized
/// subspaces (paper §3.1, table 1). A copy-paste that reshapes any of
/// the 3 sub-counts must also update this assertion — otherwise the
/// FVQ decode topology silently changes.
const _: () = assert!(N_Q_PROSODY + N_Q_CONTENT + N_Q_DETAIL == 6);

/// Which NaturalSpeech 3 FACodec (encoder,decoder,[redecoder]) pair
/// the caller is converting. Selects the model name / variant tag
/// written into the GGUF.
///
/// All four variants share [`ARCH`] `facodec` — the topology
/// (encoder / decoder body + 3-way FVQ head + optional redecoder) is
/// structurally identical, only which upstream checkpoint pair the
/// tensors came from differs. Only [`Self::V2`] is written by the
/// enum-arm default dispatch (the highest-quality pair); the other
/// three variants land through the `convert_file_with_slug` path.
///
/// # Per-variant primary-source axes
///
/// | axis | v1 | v2 (default) | redecoder-v1 | redecoder-v2 |
/// |---|---|---|---|---|
/// | encoder .bin | encoder | encoder_v2 | encoder | encoder_v2 |
/// | decoder .bin | decoder | decoder_v2 | decoder | decoder_v2 |
/// | redecoder .bin | (absent) | (absent) | redecoder | redecoder |
/// | zero-shot VC capable | no | no | yes | yes |
/// | approx. peak resident | ~415 MB | ~450 MB | ~566 MB | ~601 MB |
///
/// All four variants comfortably fit on the M1 iMac 16 GB dev
/// machine — vast.ai is NOT required even for `RedecoderV2`
/// (~601 MB peak resident, well under the memory
/// [[feedback-large-models-on-vast-ai]] ≥8 GB threshold).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacodecVariant {
    /// v1 = `encoder` + `decoder` (base pair, no zero-shot VC).
    /// `vokra.facodec.variant = "v1"`.
    #[allow(dead_code)]
    V1,
    /// v2 = `encoder_v2` + `decoder_v2` (improved pair, no zero-shot
    /// VC). Canonical / default — the best-quality codec-only pair.
    /// `vokra.facodec.variant = "v2"`.
    V2,
    /// redecoder-v1 = `encoder` + `decoder` + `redecoder` (base pair
    /// with zero-shot voice conversion enabled — swap timbre subspace
    /// while preserving prosody + content codes). See module-doc
    /// "Voice-conversion policy note" for the
    /// `vokra-voiceclone-experimental` routing question this variant
    /// raises. `vokra.facodec.variant = "redecoder-v1"`.
    #[allow(dead_code)]
    RedecoderV1,
    /// redecoder-v2 = `encoder_v2` + `decoder_v2` + `redecoder`
    /// (improved pair with zero-shot voice conversion). See module-doc
    /// "Voice-conversion policy note" for the
    /// `vokra-voiceclone-experimental` routing question this variant
    /// raises. `vokra.facodec.variant = "redecoder-v2"`.
    #[allow(dead_code)]
    RedecoderV2,
}

impl FacodecVariant {
    /// The `vokra.model.name` string for this variant.
    pub const fn name(self) -> &'static str {
        match self {
            Self::V1 => "naturalspeech3-facodec-v1",
            Self::V2 => "naturalspeech3-facodec-v2",
            Self::RedecoderV1 => "naturalspeech3-facodec-redecoder-v1",
            Self::RedecoderV2 => "naturalspeech3-facodec-redecoder-v2",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug — all four variants
    /// share the same HF repo (the variant selects which of the 5
    /// `.bin` files the prep script merges).
    pub const fn upstream_hf(self) -> &'static str {
        "amphion/naturalspeech3_facodec"
    }

    /// The `vokra.facodec.variant` tag written under
    /// [`KEY_FACODEC_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
            Self::RedecoderV1 => "redecoder-v1",
            Self::RedecoderV2 => "redecoder-v2",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s `source`
    /// argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::V1 => {
                "amphion/naturalspeech3_facodec v1 (Amphion NaturalSpeech 3 FACodec — \
                 encoder + decoder, factorized VQ codec, prosody 1cb + content 2cb + \
                 detail 3cb, 16 kHz, hop 200 → 80 tok/s, arXiv:2403.03100, apache-2.0)"
            }
            Self::V2 => {
                "amphion/naturalspeech3_facodec v2 (Amphion NaturalSpeech 3 FACodec — \
                 encoder_v2 + decoder_v2, factorized VQ codec, prosody 1cb + content 2cb + \
                 detail 3cb, 16 kHz, hop 200 → 80 tok/s, arXiv:2403.03100, apache-2.0)"
            }
            Self::RedecoderV1 => {
                "amphion/naturalspeech3_facodec redecoder-v1 (Amphion NaturalSpeech 3 \
                 FACodec — encoder + decoder + redecoder for zero-shot voice conversion, \
                 factorized VQ codec, 16 kHz, arXiv:2403.03100, apache-2.0)"
            }
            Self::RedecoderV2 => {
                "amphion/naturalspeech3_facodec redecoder-v2 (Amphion NaturalSpeech 3 \
                 FACodec — encoder_v2 + decoder_v2 + redecoder for zero-shot voice \
                 conversion, factorized VQ codec, 16 kHz, arXiv:2403.03100, apache-2.0)"
            }
        }
    }
}

/// Outcome of a NaturalSpeech 3 FACodec conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::snac::SnacReport`],
/// [`super::moss_audio_tokenizer::MossAudioTokenizerReport`],
/// [`super::neucodec::NeucodecReport`]) adapted to the
/// variant-taking `convert_naturalspeech3_facodec_variant_file`
/// surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FacodecReport {
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
    /// silent widen / downcast cannot slip in undetected without
    /// this counter also drifting.
    pub bf16_passthrough: usize,
    /// Which FACodec variant was written.
    pub variant: Option<FacodecVariant>,
}

/// Converts an Amphion NaturalSpeech 3 FACodec safetensors checkpoint
/// (pre-flattened from the upstream `.bin` bundle via
/// `tools/parity/naturalspeech3_facodec_prepare_checkpoint.py`) at
/// `input` into a Vokra-native GGUF at `output`, defaulting the
/// variant tag to [`FacodecVariant::V2`] (backward-compat entry — the
/// canonical highest-quality pair).
///
/// **Prerequisite**: the upstream release ships 5 separate
/// `torch.save()` pickle `.bin` files (no `model.safetensors`
/// mirror). Callers pre-flatten the selected variant subset to a
/// single safetensors via
/// `tools/parity/naturalspeech3_facodec_prepare_checkpoint.py`
/// (multi-file merger mirror of `sepformer_prepare_checkpoint.py`)
/// before invoking this converter — no pickle parser enters the
/// Vokra runtime (NFR-DS-02 / FR-LD-05).
///
/// # Note
///
/// The `ModelKind::Facodec` CLI dispatch arm calls
/// [`convert_naturalspeech3_facodec_variant_file`] directly with an
/// explicit [`FacodecVariant::V2`] default (mirror of snac /
/// moss_audio_tokenizer). This backward-compat wrapper stays for the
/// library API — external callers of
/// `crate::models::naturalspeech3_facodec` that do not need variant
/// selection can call this shorter form.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input.
#[allow(dead_code)]
pub fn convert_naturalspeech3_facodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FacodecReport, ConvertError> {
    convert_naturalspeech3_facodec_variant_file(input, output, FacodecVariant::V2, license)
}

/// Variant-taking entry — the explicit form used by
/// `convert_file_with_slug` for per-slug dispatch (mirror of
/// [`super::snac::convert_snac_file`] +
/// [`super::moss_audio_tokenizer::convert_moss_audio_tokenizer_variant_file`]).
///
/// See [`convert_naturalspeech3_facodec_file`] for the prerequisite
/// (offline `.bin` → safetensors bridge via
/// `tools/parity/naturalspeech3_facodec_prepare_checkpoint.py`).
///
/// Every F32 / F16 / BF16 tensor passes through under its
/// role-prefixed upstream state-dict name (`encoder.*` / `decoder.*`
/// / `redecoder.*` — the prep script decides these prefixes); the
/// `vokra.model.*` (arch / name / category), `vokra.provenance.*`
/// (weight_license / license / model_id / source / upstream_hf),
/// `vokra.facodec.*` (sample_rate / hop_size / n_quantizers_* per
/// subspace), and `vokra.facodec.variant` chunks are stamped for the
/// runtime compliance gate (FR-CP-03) and shape-checked config
/// dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`, `Permissive`) — the
/// upstream Amphion release ships apache-2.0 end-to-end.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input.
pub fn convert_naturalspeech3_facodec_variant_file(
    input: &Path,
    output: &Path,
    variant: FacodecVariant,
    license: Option<&str>,
) -> Result<FacodecReport, ConvertError> {
    // Largest variant (redecoder-v2) is ~601 MB peak resident after
    // the prep-script merge — 1 order of magnitude smaller than the
    // streaming-mandated Moshi 14 GiB tier, so the simple
    // `std::fs::read` posture the sibling non-streaming BF16
    // pass-through converters use applies. Every variant fits
    // comfortably on M1 iMac 16 GB per the memory
    // [[feedback-large-models-on-vast-ai]] threshold matrix (vast.ai
    // is required only for >8 GB checkpoints).
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_FACODEC_VARIANT, variant.tag());
    write_hparams(&mut b);

    // Default provenance stamp — Permissive apache-2.0 end-to-end
    // (Amphion GitHub LICENSE + HF cardData verified via authenticated
    // API 2026-08-01). The optional `license` argument overrides below.
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

    let mut report = FacodecReport {
        variant: Some(variant),
        ..FacodecReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // ADR (mirror of snac / neucodec / moss_audio_tokenizer /
    // focalcodec); the runtime widens BF16 → f32 exactly at load via
    // the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    // decode_bf16`.
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

/// Writes the `vokra.facodec.*` hparam chunk group from the
/// transcribed primary-source constants above (paper arXiv:2403.03100
/// + Amphion source, fetched 2026-08-01).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(KEY_HOP_SIZE, HOP_SIZE);
    b.add_u32(KEY_N_Q_PROSODY, N_Q_PROSODY);
    b.add_u32(KEY_N_Q_CONTENT, N_Q_CONTENT);
    b.add_u32(KEY_N_Q_DETAIL, N_Q_DETAIL);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue};

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
        );
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

    /// Builds an F32 tensor safetensors buffer — matches upstream
    /// Amphion FACodec dtype (F32 verified via Amphion training
    /// pipeline defaults, 2026-08-01).
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
            "vokra-facodec-{kind}-{}-{}.bin",
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
    fn arch_string_is_distinct_from_every_sibling_codec() {
        // ARCH is the sole cross-crate handshake with the future
        // `vokra-models::facodec::EXPECTED_ARCH` — pinning it here
        // catches an accidental rename that would silently mis-route
        // runtime dispatch. Also assert that ARCH does NOT collide
        // with any sibling codec (FACodec is FVQ, not RVQ / FSQ /
        // focal-modulation — the topology difference is what makes
        // the distinct arch load-bearing).
        assert_eq!(ARCH, "facodec");
        for sibling in [
            "mimi",
            "dac",
            "wavtokenizer",
            "snac",
            "xcodec2",
            "focalcodec",
            "speechtokenizer",
            "neucodec",
            "bicodec",
            "xy_tokenizer",
            "funcodec",
            "step_audio2_mini",
            "moss_audio_tokenizer",
        ] {
            assert_ne!(
                ARCH, sibling,
                "FACodec must not silently alias sibling codec `{sibling}` — different quantizer family"
            );
        }
    }

    #[test]
    fn f32_tensor_passes_through_and_stamps_land_v2() {
        // Upstream Amphion FACodec `.bin` files are F32 (Amphion
        // training pipeline default) — this test pins the primary
        // code path.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic role-prefixed tensor name — the prep
        // script namespaces every upstream key under its role
        // (`encoder.` / `decoder.` / `redecoder.`).
        let input_bytes = safetensors_one_f32("encoder.conv_pre.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("v2-in", &input_bytes);
        let output_path = write_temp("v2-out", &[]);

        let report = convert_naturalspeech3_facodec_variant_file(
            &input_path,
            &output_path,
            FacodecVariant::V2,
            None,
        )
        .expect("convert_naturalspeech3_facodec_variant_file must accept F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );
        assert_eq!(report.variant, Some(FacodecVariant::V2));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("encoder.conv_pre.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance / category / arch / variant chunks landed.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("naturalspeech3-facodec-v2")
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
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `codec`",
        );
        assert_eq!(
            file.get(KEY_FACODEC_VARIANT).and_then(|v| v.as_str()),
            Some("v2"),
            "vokra.facodec.variant must be `v2` for default variant",
        );
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16-quantized derivatives should
        // ride the same arm as the sibling BF16-pass-through
        // converters (snac / neucodec / moss_audio_tokenizer).
        // Non-zero BF16 bit patterns so a subsequent byte-identity
        // assert catches any silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic FVQ codebook tensor name — decoder-side
        // hosts the 3 quantizer heads (prosody / content / detail).
        let input_bytes =
            safetensors_one_bf16("decoder.quantizer_prosody.codebook.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_naturalspeech3_facodec_variant_file(
            &input_path,
            &output_path,
            FacodecVariant::V2,
            None,
        )
        .expect("convert must accept BF16 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("decoder.quantizer_prosody.codebook.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    /// Every enum variant maps to a distinct `(name, tag)` pair —
    /// a defensive pin against a copy-paste that would silently
    /// re-use the V2 strings for a new variant. `upstream_hf` is
    /// intentionally shared across all four (they all live in the
    /// same HF repo — the variant selects which of the 5 `.bin`
    /// files the prep script merges).
    #[test]
    fn every_variant_has_distinct_name_and_tag() {
        let variants = [
            FacodecVariant::V1,
            FacodecVariant::V2,
            FacodecVariant::RedecoderV1,
            FacodecVariant::RedecoderV2,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let a = variants[i];
                let b = variants[j];
                assert_ne!(a.name(), b.name(), "names must differ ({a:?} vs {b:?})");
                assert_ne!(a.tag(), b.tag(), "tags must differ ({a:?} vs {b:?})");
                assert_ne!(
                    a.source_description(),
                    b.source_description(),
                    "source_description must differ ({a:?} vs {b:?})"
                );
                // The HF slug IS shared — one repo, four variants.
                assert_eq!(
                    a.upstream_hf(),
                    b.upstream_hf(),
                    "all variants share the same HF repo"
                );
            }
        }
    }

    #[test]
    fn redecoder_v2_variant_emits_distinct_stamps() {
        let f32_bytes: Vec<u8> = [7.0_f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Realistic redecoder-side tensor name — the redecoder module
        // is responsible for zero-shot voice conversion (swap timbre
        // subspace while preserving prosody + content).
        let input_bytes = safetensors_one_f32("redecoder.timbre_swap.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("redecoder-v2-in", &input_bytes);
        let output_path = write_temp("redecoder-v2-out", &[]);

        let report = convert_naturalspeech3_facodec_variant_file(
            &input_path,
            &output_path,
            FacodecVariant::RedecoderV2,
            None,
        )
        .expect("convert RedecoderV2 variant");
        assert_eq!(report.variant, Some(FacodecVariant::RedecoderV2));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("naturalspeech3-facodec-redecoder-v2"),
            "RedecoderV2 must emit its own model.name, not fall back to V2"
        );
        assert_eq!(
            file.get(KEY_FACODEC_VARIANT).and_then(|v| v.as_str()),
            Some("redecoder-v2")
        );
        // Arch + category are shared with V2 (same downstream
        // dispatch — all four variants route to the same FACodec
        // class upstream).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
    }

    #[test]
    fn hparam_chunk_group_lands_with_transcribed_axes() {
        // Round-trip every transcribed `vokra.facodec.*` U32 hparam
        // and cross-check the emitted value against the primary
        // source (paper arXiv:2403.03100 + Amphion source at
        // `github.com/open-mmlab/Amphion/tree/main/models/codec/ns3_codec`,
        // fetched 2026-08-01).
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("encoder.conv_pre.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("hparam-in", &input_bytes);
        let output_path = write_temp("hparam-out", &[]);

        convert_naturalspeech3_facodec_variant_file(
            &input_path,
            &output_path,
            FacodecVariant::V2,
            None,
        )
        .expect("convert must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        for (key, want) in [
            (KEY_SAMPLE_RATE, SAMPLE_RATE),
            (KEY_HOP_SIZE, HOP_SIZE),
            (KEY_N_Q_PROSODY, N_Q_PROSODY),
            (KEY_N_Q_CONTENT, N_Q_CONTENT),
            (KEY_N_Q_DETAIL, N_Q_DETAIL),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: expected U32 {want}, got {other:?}"),
            }
        }
    }

    #[test]
    fn primary_source_axes_agree_with_paper() {
        // Pins every transcribed constant to the primary source. A
        // future contributor who edits a constant without also
        // updating the docstring + license-audit row + this test
        // fails loudly. Sourced from paper §3.1 table 1 +
        // Amphion source tree
        // `open-mmlab/Amphion/models/codec/ns3_codec` fetched
        // 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」.
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(HOP_SIZE, 200);
        assert_eq!(N_Q_PROSODY, 1);
        assert_eq!(N_Q_CONTENT, 2);
        assert_eq!(N_Q_DETAIL, 3);
        // The 6-codebook total is the FVQ family fingerprint —
        // silently changing any sub-count would misroute the runtime
        // FVQ decode topology (paper §3.1).
        assert_eq!(N_Q_PROSODY + N_Q_CONTENT + N_Q_DETAIL, 6);
        // Sample rate / hop_size ratio pins the token rate — 16000 /
        // 200 = 80 tok/s per subspace. A future variant that
        // reshapes this ratio would need a new ModelKind.
        assert_eq!(SAMPLE_RATE / HOP_SIZE, 80);
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
        let input_bytes = safetensors_one_f32("encoder.conv_pre.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_naturalspeech3_facodec_variant_file(
            &input_path,
            &output_path,
            FacodecVariant::V2,
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
    }

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// stamp — that would be a silent research-flag downgrade. The
    /// `Some(s) if !s.is_empty()` guard keeps the default
    /// apache-2.0 / Permissive stamp (mirror of the sibling empty-string
    /// guard tests).
    #[test]
    fn empty_string_license_override_keeps_default_stamp() {
        let f32_bytes: Vec<u8> = [0.5_f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("encoder.conv_pre.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("empty-in", &input_bytes);
        let output_path = write_temp("empty-out", &[]);

        convert_naturalspeech3_facodec_variant_file(
            &input_path,
            &output_path,
            FacodecVariant::V2,
            Some(""),
        )
        .expect("empty override must succeed and be ignored");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "empty string must NOT downgrade the license stamp"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "empty string must NOT downgrade the class"
        );
    }

    #[test]
    fn convert_file_wrapper_defaults_to_v2() {
        // The backward-compat `convert_naturalspeech3_facodec_file`
        // entry (no variant argument) must default to V2 — the
        // highest-quality codec-only pair. A drift here would silently
        // publish V1 tensors under the canonical `--model
        // naturalspeech3-facodec` slug.
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("encoder.conv_pre.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("default-in", &input_bytes);
        let output_path = write_temp("default-out", &[]);

        let report = convert_naturalspeech3_facodec_file(&input_path, &output_path, None)
            .expect("backward-compat convert must succeed");
        assert_eq!(report.variant, Some(FacodecVariant::V2));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("naturalspeech3-facodec-v2"),
            "backward-compat entry must default to V2 (canonical highest-quality pair)"
        );
    }
}
