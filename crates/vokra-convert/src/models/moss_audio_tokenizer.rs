#![allow(clippy::doc_lazy_continuation)]
//! **MOSS-Audio-Tokenizer** (`OpenMOSS-Team/MOSS-Audio-Tokenizer`,
//! `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`, apache-2.0): safetensors →
//! GGUF conversion (2026-08-01 Wave 3).
//!
//! Input: one of the two OpenMOSS MOSS-Audio-Tokenizer release
//! safetensors — the codec half of the MOSS-TTS pipeline (waveform →
//! discrete tokens fed into the sibling MOSS-TTS LLM at
//! `crates/vokra-convert/src/models/moss_tts.rs`). Publishing this
//! converter unblocks MOSS-TTS end-to-end audio synthesis. Output: a
//! GGUF carrying every float tensor verbatim under its upstream
//! `MossAudioTokenizerModel` state-dict name, plus the
//! `vokra.provenance.*` / `vokra.model.*` / `vokra.moss_audio_tokenizer.*`
//! metadata chunks a future native MOSS-Audio-Tokenizer loader will
//! read.
//!
//! # Family coverage — variant selectors
//!
//! Both variants share `model_type = "moss-audio-tokenizer"` and a
//! single `MossAudioTokenizerModel` class per upstream
//! `config.json.architectures`; they are distinct codec topologies, not
//! merely two parameter scales
//! ([`MossAudioTokenizerVariant`] discriminator stamped under
//! `vokra.moss_audio_tokenizer.variant`). Values transcribed 2026-08-01
//! via the HF `api/models/<id>` (CLAUDE.md「ハルシネーション厳禁」):
//!
//! - [`MossAudioTokenizerVariant::Full`] —
//!   `OpenMOSS-Team/MOSS-Audio-Tokenizer` (1,774,566,400 F32 params →
//!   7.10 GB `usedStorage` across 2 sharded safetensors, sibling
//!   `[model-00001-of-00002.safetensors, model-00002-of-00002.safetensors,
//!   model.safetensors.index.json, config.json,
//!   configuration_moss_audio_tokenizer.py,
//!   modeling_moss_audio_tokenizer.py]`).
//! - [`MossAudioTokenizerVariant::Nano`] —
//!   `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (21,969,664 F32 params
//!   → 87.9 MB `usedStorage`, single-shard
//!   `[model-00001-of-00001.safetensors, model.safetensors.index.json,
//!   config.json, configuration_moss_audio_tokenizer.py,
//!   modeling_moss_audio_tokenizer.py]`). Additionally distilled per
//!   arXiv:2603.18090 encoder distillation reference.
//!
//! # Sharded safetensors → single-file bridge
//!
//! Both variants ship as sharded safetensors with a
//! `model.safetensors.index.json` weight-map. Vokra's Rust converter is
//! **single-file safetensors-in** by design (NFR-DS-02 zero-dep — the
//! runtime never grows a shard-index reader), so callers pre-merge the
//! shards to a single safetensors offline via
//! `tools/parity/moss_audio_tokenizer_prepare_checkpoint.py` (a
//! dedicated uv-managed Python 3.12 bridge that walks the weight-map,
//! merges the shards, and re-serializes). Same posture as
//! `granite_speech.rs` (3-shard release) — Vokra converters keep the
//! runtime tree free of the shard-index-json reader.
//!
//! Both variants are **safetensors-native** (no pickle bridge required)
//! — contrast the sibling `MOSS-TTS-Nano-100M` which requires
//! `bin_to_safetensors.py`. **No ONNX** either (FR-LD-05 permanent);
//! mirror ONNX repos (e.g. `OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX`)
//! exist upstream but Vokra never touches them.
//!
//! # HF / licence / category
//!
//! - Upstream HF (recorded under `vokra.provenance.upstream_hf` +
//!   `vokra.model.name`): `OpenMOSS-Team/MOSS-Audio-Tokenizer[-Nano]`.
//! - SPDX: `apache-2.0` for both variants (`cardData.license =
//!   "apache-2.0"` on both HF model cards, verified 2026-08-01 via
//!   authenticated `https://huggingface.co/api/models/<id>` — both
//!   `private: false`, `gated: false`).
//! - No `LICENSE` file in the repos — apache-2.0 declared via HF
//!   cardData tag only, fetched at publish time by
//!   `scripts/fetch_license.sh --spdx apache-2.0`.
//! - Model category: `codec` (recorded under `vokra.model.category`).
//!   The codec half of the MOSS-TTS pipeline; same category tag as
//!   sibling `snac` / `mimi` / `dac` / `wavtokenizer` / `neucodec` /
//!   `focalcodec` codecs (all `codec`).
//!
//! # BF16 pass-through (mirror of `snac` / `neucodec` / `focalcodec`)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at load
//! via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`MossAudioTokenizerReport::bf16_passthrough`] records how
//! many BF16 tensors landed on this arm so a silent widen / downcast
//! cannot slip in undetected. Both upstream releases ship **F32
//! exclusively** (no BF16 override in HF metadata verified 2026-08-01)
//! so the BF16 arm is defensive today; the counter is kept for the
//! sibling BF16 pass-through contract and for third-party BF16
//! re-quantizations.
//!
//! # Distinct arch from every sibling codec
//!
//! `ARCH = "moss_audio_tokenizer"` (with an underscore separator per
//! GGUF metadata convention — sibling `moss_tts.rs` uses `moss_tts`).
//! Intentionally distinct from every sibling codec (`mimi` / `dac` /
//! `wavtokenizer` / `neucodec` / `xcodec2` / `focalcodec` /
//! `speechtokenizer` / `bicodec` / `xy_tokenizer` / `snac` /
//! `step_audio2_mini` / `funcodec`) — silently sharing an arch tag with
//! any of these would mis-route the runtime dispatch. MOSS-Audio-Tokenizer
//! is the OpenMOSS-specific codec designed to pair with the MOSS-TTS
//! LLM (`moss_tts` arch); no other Vokra converter carries this
//! `MossAudioTokenizerModel` class.
//!
//! # Custom code / trust_remote_code
//!
//! Both variants ship `modeling_moss_audio_tokenizer.py` +
//! `configuration_moss_audio_tokenizer.py` requiring
//! `trust_remote_code=True` for the reference Python forward. Vokra
//! never touches Python at runtime, so this only affects the owner-side
//! parity dumper (mirror `tools/parity/kokoro_prepare_checkpoint.py`).
//! This converter reads tensor bytes verbatim; the modeling code is
//! external to the runtime.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `MossAudioTokenizerModel`
//! state-dict names verbatim** (the CSM / Kokoro / Neucodec /
//! MOSS-TTS / SNAC contract). Real-weight parity vs the upstream
//! Python reference is deferred to owner (`docs/license-audit.md` §3.1
//! sign-off queue); this converter provides the byte-parallel GGUF
//! surface only — a "loud-partial" landing per the RMVPE / Charsiu /
//! MOSS-TTS precedent.
//!
//! # No ONNX / no pickle in runtime (permanent)
//!
//! MOSS-Audio-Tokenizer is distributed as safetensors + a Python
//! pipeline; this converter **never** touches ONNX (FR-LD-05) and
//! **never** touches pickle (NFR-DS-02 zero-dep). The pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/moss_audio_tokenizer/` module (whisper.cpp
//! 型 self re-implementation, CLAUDE.md 設計判断 4). Between now and
//! that landing, the runtime consumer walks the emitted tensor names
//! and either succeeds or fails loudly per FR-EX-08 — today's
//! converter surface is byte-exact provenance + tensor-name
//! preservation only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MOSS-Audio-Tokenizer GGUFs. Shared across
/// both [`MossAudioTokenizerVariant`] entries — upstream's
/// `MossAudioTokenizerModel` class (per `config.json.architectures`)
/// is shared by `Full` and `Nano`, but the topology is not. Full is a
/// 24 kHz mono, 32-quantizer codec; Nano is a 48 kHz
/// stereo/interleaved, 16-quantizer codec with a different staged
/// Transformer stack. Runtime dispatch therefore uses this arch together
/// with a strict complete tensor manifest and the variant tag.
///
/// Intentionally distinct from every sibling codec (`mimi`, `dac`,
/// `wavtokenizer`, `neucodec`, `xcodec2`, `focalcodec`,
/// `speechtokenizer`, `bicodec`, `xy_tokenizer`, `snac`,
/// `step_audio2_mini`, `funcodec`) — see the module-level docstring.
/// Underscore separator per GGUF metadata convention (sibling
/// `moss_tts.rs` uses `moss_tts`).
pub const ARCH: &str = "moss_audio_tokenizer";

/// `vokra.model.category` value written for every MOSS-Audio-Tokenizer
/// GGUF. Same tag as the sibling `snac` / `mimi` / `dac` /
/// `wavtokenizer` / `neucodec` / `focalcodec` codecs (all `codec`),
/// used by the model-card generator classifier.
pub const CATEGORY: &str = "codec";

/// `vokra.model.name` value written for the canonical `Full` variant
/// (backward-compat alias — new callers should use
/// [`MossAudioTokenizerVariant::name`]).
#[allow(dead_code)]
pub const NAME: &str = "moss-audio-tokenizer";

/// `vokra.provenance.upstream_hf` value for the canonical `Full`
/// variant (backward-compat alias — new callers should use
/// [`MossAudioTokenizerVariant::upstream_hf`]).
#[allow(dead_code)]
pub const UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer";

/// Default upstream weight licence (SPDX). Both
/// `OpenMOSS-Team/MOSS-Audio-Tokenizer` and
/// `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` ship apache-2.0
/// end-to-end (HF `cardData.license = "apache-2.0"` verified 2026-08-01
/// via authenticated API; no `LICENSE` file in the repos — declared
/// via HF cardData tag only).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of the sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.moss_audio_tokenizer.variant`: `"full"` / `"nano"`.
/// Consumers pick a specific scale without parsing free-text
/// `vokra.model.name` (mirrors `super::snac` +
/// `super::focalcodec` discriminators).
pub const KEY_VARIANT: &str = "vokra.moss_audio_tokenizer.variant";

/// Which MOSS-Audio-Tokenizer release the caller is converting.
/// Selects the model name / upstream HF slug / variant tag written
/// into the GGUF.
///
/// Both variants share [`ARCH`] `moss_audio_tokenizer` because upstream uses
/// the same `MossAudioTokenizerModel` class, while their runtime topology is
/// selected independently from the variant tag and complete tensor manifest.
///
/// # Per-variant primary-source axes
///
/// Values from HF `api/models/<id>` fetched 2026-08-01:
///
/// | axis | Full | Nano |
/// |---|---|---|
/// | params (F32) | 1,774,566,400 | 21,969,664 |
/// | `usedStorage` | 7.10 GB | 87.9 MB |
/// | shards | 2 | 1 |
/// | vast.ai required | yes (>=2 GB artifact) | no (trivial local) |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MossAudioTokenizerVariant {
    /// `OpenMOSS-Team/MOSS-Audio-Tokenizer`: the full-scale (~1.77B
    /// params, F32) release — the codec half of the MOSS-TTS pipeline
    /// (waveform → discrete tokens for the sibling `moss_tts` LLM).
    /// Ships as 2 sharded safetensors + `model.safetensors.index.json`
    /// weight-map (~6.6 GB effective weights → 7.10 GB `usedStorage`).
    /// Canonical / default — the higher-fidelity release the MOSS-TTS
    /// consumer pairs with.
    /// `vokra.moss_audio_tokenizer.variant = "full"`.
    Full,
    /// `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`: the compact (~22M
    /// params, F32) distilled variant per arXiv:2603.18090 encoder
    /// distillation reference. This is a distinct 48 kHz
    /// stereo/interleaved topology, not a width-reduced Full checkpoint.
    /// Ships as 1 sharded safetensors +
    /// `model.safetensors.index.json` weight-map (~88 MB — trivial to
    /// convert locally on the M1 iMac dev machine).
    /// `vokra.moss_audio_tokenizer.variant = "nano"`.
    Nano,
}

impl MossAudioTokenizerVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Full => "moss-audio-tokenizer",
            Self::Nano => "moss-audio-tokenizer-nano",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for this
    /// release — the primary redistribution source the model-card
    /// generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Full => "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            Self::Nano => "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        }
    }

    /// The `vokra.moss_audio_tokenizer.variant` tag written under
    /// [`KEY_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Nano => "nano",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s
    /// `source` argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Full => {
                "OpenMOSS-Team/MOSS-Audio-Tokenizer (MOSS-Audio-Tokenizer \
                 codec Full ~1.77B params F32, arXiv:2602.10934, apache-2.0)"
            }
            Self::Nano => {
                "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano (MOSS-Audio-Tokenizer \
                 codec Nano ~22M params F32 distilled per arXiv:2603.18090, \
                 apache-2.0)"
            }
        }
    }
}

/// Outcome of a MOSS-Audio-Tokenizer conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::snac::SnacReport`,
/// `super::neucodec::NeucodecReport`,
/// `super::focalcodec::FocalcodecReport`) adapted to the
/// variant-taking `convert_moss_audio_tokenizer_variant_file`
/// surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MossAudioTokenizerReport {
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
    /// counter also drifting. Both upstream releases ship F32
    /// exclusively (verified 2026-08-01) so this counter is defensive
    /// on the primary release; it stays for the sibling BF16
    /// pass-through contract and for third-party BF16
    /// re-quantizations.
    pub bf16_passthrough: usize,
    /// Which MOSS-Audio-Tokenizer variant was written.
    pub variant: Option<MossAudioTokenizerVariant>,
}

/// Converts a `OpenMOSS-Team/MOSS-Audio-Tokenizer[-Nano]` safetensors
/// checkpoint at `input` into a Vokra-native GGUF at `output`,
/// defaulting the variant tag to
/// [`MossAudioTokenizerVariant::Full`] (backward-compat entry — the
/// canonical release).
///
/// **Prerequisite**: both variants ship as sharded safetensors +
/// `model.safetensors.index.json` weight-map. Callers pre-flatten to
/// a single safetensors via
/// `tools/parity/moss_audio_tokenizer_prepare_checkpoint.py` (the
/// `granite_speech.rs` posture) before invoking this converter — no
/// shard-index-json reader enters the Vokra runtime (NFR-DS-02 /
/// FR-LD-05).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// `MossAudioTokenizerModel` state-dict name; the `vokra.model.*`
/// (arch / name / category), `vokra.provenance.*` (weight_license /
/// license / model_id / source / upstream_hf), and
/// `vokra.moss_audio_tokenizer.variant` chunks are stamped for the
/// runtime compliance gate (FR-CP-03) and shape-checked config
/// dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`) — both
/// upstream MOSS-Audio-Tokenizer releases ship apache-2.0 end-to-end.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input.
///
/// # Note
///
/// The `ModelKind::MossAudioTokenizer` CLI dispatch arm calls
/// [`convert_moss_audio_tokenizer_variant_file`] directly with an
/// explicit [`MossAudioTokenizerVariant::Full`] default (mirror of
/// snac / focalcodec). This backward-compat wrapper stays for the
/// library API — external callers of `crate::models::moss_audio_tokenizer`
/// that do not need variant selection can call this shorter form.
#[allow(dead_code)]
pub fn convert_moss_audio_tokenizer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MossAudioTokenizerReport, ConvertError> {
    convert_moss_audio_tokenizer_variant_file(
        input,
        output,
        MossAudioTokenizerVariant::Full,
        license,
    )
}

/// Variant-taking entry — the explicit form used by
/// `convert_file_with_slug` for per-slug dispatch (mirror of
/// `super::snac::convert_snac_file` +
/// `super::focalcodec::convert_focalcodec_file`).
///
/// See [`convert_moss_audio_tokenizer_file`] for the semantics.
pub fn convert_moss_audio_tokenizer_variant_file(
    input: &Path,
    output: &Path,
    variant: MossAudioTokenizerVariant,
    license: Option<&str>,
) -> Result<MossAudioTokenizerReport, ConvertError> {
    // Full's merged safetensors is ~6.6 GB (F32), so repository policy
    // requires this converter to run on vast.ai (all model artifacts >=2 GB
    // are remote work). This non-streaming reader remains valid there. Nano
    // is ~88 MB and is safe for a focused local conversion, although parity
    // generation still follows the model-family verification runbook.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_VARIANT, variant.tag());

    // Default provenance stamp — Permissive apache-2.0 end-to-end
    // (both upstream MOSS-Audio-Tokenizer model cards verified via
    // authenticated HF API 2026-08-01). The optional `license`
    // argument overrides below.
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

    let mut report = MossAudioTokenizerReport {
        variant: Some(variant),
        ..MossAudioTokenizerReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // ADR; the runtime widens BF16 → f32 exactly at load via the
    // single choke point `crates/vokra-core/src/gguf/quant/mod.rs
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

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile};

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
    /// MOSS-Audio-Tokenizer dtype (both variants ship F32 exclusively,
    /// verified 2026-08-01 via HF API).
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

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// PID + nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-moss-audio-tokenizer-{kind}-{}-{}.bin",
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
    fn arch_is_distinct_from_every_sibling_codec_and_moss_tts() {
        // ARCH is the sole cross-crate handshake with the future
        // `vokra-models::moss_audio_tokenizer::EXPECTED_ARCH` —
        // pinning it here catches an accidental rename that would
        // silently mis-route runtime dispatch. Sibling arch tags come
        // from the sibling module `pub const ARCH: &str = ...`
        // constants.
        assert_eq!(ARCH, "moss_audio_tokenizer");
        // MOSS-TTS is the LM half of the pipeline; if silently
        // shared, the runtime would try to feed a codec input into an
        // LM forward and crash somewhere deep. Distinct arch tags
        // make the mismatch loud at load time.
        assert_ne!(ARCH, "moss_tts");
        // Sibling codecs — the module's whole reason for a dedicated
        // arch tag is to not silently share dispatch with these.
        for other in [
            "snac",
            "mimi",
            "dac",
            "wavtokenizer",
            "neucodec",
            "xcodec2",
            "focalcodec",
            "speechtokenizer",
            "bicodec",
            "xy_tokenizer",
            "step_audio2_mini",
            "funcodec",
        ] {
            assert_ne!(
                ARCH, other,
                "MOSS-Audio-Tokenizer arch must not silently alias sibling codec {other}"
            );
        }
    }

    #[test]
    fn variant_name_and_upstream_hf_lock_the_full_release() {
        assert_eq!(
            MossAudioTokenizerVariant::Full.name(),
            "moss-audio-tokenizer"
        );
        assert_eq!(
            MossAudioTokenizerVariant::Full.upstream_hf(),
            "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        );
        // Enum-arm-default constants agree with the Full variant (a
        // drift would break the ModelKind::MossAudioTokenizer
        // dispatch in lib.rs).
        assert_eq!(NAME, MossAudioTokenizerVariant::Full.name());
        assert_eq!(UPSTREAM_HF, MossAudioTokenizerVariant::Full.upstream_hf());
    }

    #[test]
    fn f32_tensor_passes_through_and_stamps_land_full() {
        // Upstream MOSS-Audio-Tokenizer is F32 (both variants,
        // verified 2026-08-01 via HF `safetensors.parameters` API) —
        // this test pins the primary code path.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from
        // MOSS-Audio-Tokenizer's encoder body — the actual state-dict
        // walk (from the custom-code Python) uses names of the form
        // `encoder.conv.<idx>.weight` etc. Any name works for the
        // pass-through test; a canonical-shape name catches
        // copy-paste-from-wrong-model regressions in code review.
        let input_bytes = safetensors_one_f32("encoder.conv.0.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("full-f32-in", &input_bytes);
        let output_path = write_temp("full-f32-out", &[]);

        let report = convert_moss_audio_tokenizer_variant_file(
            &input_path,
            &output_path,
            MossAudioTokenizerVariant::Full,
            None,
        )
        .expect("convert_moss_audio_tokenizer_variant_file must accept F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );
        assert_eq!(report.variant, Some(MossAudioTokenizerVariant::Full));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("encoder.conv.0.weight")
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
            Some("moss-audio-tokenizer"),
            "Full must emit the canonical `moss-audio-tokenizer` model name"
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
            Some("OpenMOSS-Team/MOSS-Audio-Tokenizer"),
            "Full must emit the canonical upstream slug"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `codec`",
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("full"),
            "vokra.moss_audio_tokenizer.variant must be `full` for Full",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16 re-quantized derivatives
        // should ride the same arm as the sibling BF16 pass-through
        // converters (snac / neucodec / focalcodec). Non-zero BF16
        // bit patterns so a subsequent byte-identity assert catches
        // any silent widen / downcast attempt (zeroed payloads would
        // round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic upstream codec quantizer tensor name.
        let input_bytes = safetensors_one_bf16("quantizer.codebook.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_moss_audio_tokenizer_variant_file(
            &input_path,
            &output_path,
            MossAudioTokenizerVariant::Full,
            None,
        )
        .expect("convert must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror snac / neucodec / focalcodec)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("quantizer.codebook.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// The Nano variant reuses the same converter body but the name /
    /// variant / upstream stamps differ. Silently sharing stamps
    /// would misroute a downstream loader that dispatches on
    /// `vokra.model.name` or the variant discriminator — this test
    /// guards the variant switch (the
    /// `super::snac::tests::hz44_variant_emits_distinct_stamps`
    /// precedent).
    #[test]
    fn nano_variant_emits_distinct_stamps() {
        let f32_bytes: Vec<u8> = [7.0_f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("decoder.conv.0.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("nano-in", &input_bytes);
        let output_path = write_temp("nano-out", &[]);

        let report = convert_moss_audio_tokenizer_variant_file(
            &input_path,
            &output_path,
            MossAudioTokenizerVariant::Nano,
            None,
        )
        .expect("convert Nano variant");
        assert_eq!(report.variant, Some(MossAudioTokenizerVariant::Nano));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("moss-audio-tokenizer-nano"),
            "Nano must emit its own model.name, not fall back to Full"
        );
        assert_eq!(file.get(KEY_VARIANT).and_then(|v| v.as_str()), Some("nano"));
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
        );
        // Arch + category are shared with Full (same downstream
        // dispatch — both variants route to the same
        // MossAudioTokenizerModel class upstream).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_flows_through() {
        // A user who re-trained on a permissive corpus supplies a
        // different SPDX id at conversion time — the override must
        // land on KEY_PROVENANCE_LICENSE + KEY_PROVENANCE_WEIGHT_LICENSE
        // and the LicenseClass must be re-derived by
        // from_license_str.
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("encoder.conv.0.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_moss_audio_tokenizer_variant_file(
            &input_path,
            &output_path,
            MossAudioTokenizerVariant::Full,
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

    /// An empty `Some("")` license override must NOT wipe the
    /// built-in stamp — that would be a silent research-flag
    /// downgrade. The `Some(s) if !s.is_empty()` guard in
    /// `convert_moss_audio_tokenizer_variant_file` keeps the default
    /// apache-2.0 / Permissive stamp (mirror of xcodec2's empty-string
    /// guard test).
    #[test]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_bytes: Vec<u8> = [0.5_f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("encoder.conv.0.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("empty-in", &input_bytes);
        let output_path = write_temp("empty-out", &[]);

        convert_moss_audio_tokenizer_variant_file(
            &input_path,
            &output_path,
            MossAudioTokenizerVariant::Full,
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

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Every enum variant maps to a distinct `(name, tag,
    /// upstream_hf, source_description)` tuple — a defensive pin
    /// against a copy-paste that would silently re-use the Full
    /// strings for a new variant (matches the snac
    /// `every_variant_has_distinct_stamps` precedent).
    #[test]
    fn every_variant_has_distinct_stamps() {
        let variants = [
            MossAudioTokenizerVariant::Full,
            MossAudioTokenizerVariant::Nano,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let a = variants[i];
                let b = variants[j];
                assert_ne!(a.name(), b.name(), "names must differ ({a:?} vs {b:?})");
                assert_ne!(a.tag(), b.tag(), "tags must differ ({a:?} vs {b:?})");
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

    #[test]
    fn malformed_input_returns_parse_error() {
        let input_path = write_temp("malformed-in", &[]);
        let output_path = write_temp("malformed-out", &[]);
        let err = convert_moss_audio_tokenizer_file(&input_path, &output_path, None)
            .expect_err("empty input must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
