//! **Neucodec** (`neuphonic/neucodec` + `neuphonic/distill-neucodec`,
//! apache-2.0): safetensors → GGUF conversion (SoTA follow-on,
//! 2026-07-25; `distill-neucodec` variant added 2026-08-01).
//!
//! Input: an upstream `neuphonic/neucodec` release (2.35 GB
//! safetensors, 24 kHz, 0.8 kbps @ 50 Hz FSQ codec, X-Codec 2 lineage)
//! OR `neuphonic/distill-neucodec` (978 MB `pytorch_model.bin` — same
//! NeuCodec architecture, distilled to ~10× fewer parameters and
//! ~7.5× fewer inference MACs by swapping BigCodec's acoustic encoder
//! with SQCodec (70M → 36M) and w2v-bert-2.0's semantic encoder with
//! DistilHuBERT (600M → 21M); API + `fsq_codes` shape identical to
//! base per the upstream README primary source, so a single Rust
//! converter drives both). The distill release ships torch pickle
//! only — pre-flatten to safetensors offline via
//! `tools/parity/nemo_pt_to_safetensors.py` (the funcodec / wespeaker
//! / emotion2vec pattern — the converter refuses to touch pickle
//! because that would require embedding a Python interpreter and
//! re-breaking the NFR-DS-02 zero-dep posture). Output: a GGUF
//! carrying every float tensor verbatim under its upstream
//! safetensors name, plus the `vokra.provenance.*` / `vokra.model.*`
//! / `vokra.neucodec.variant` metadata chunks the native
//! `vokra_models::neucodec::NeuCodec` loader reads.
//!
//! # Variant identity
//!
//! Both upstream releases share the [`ARCH`] tag `neucodec` — the
//! decoder topology is identical, while the decoder parameter values and
//! encoder-side parameter counts may differ. The
//! [`NeucodecVariant`] discriminator
//! tags the emitted GGUF under [`KEY_NEUCODEC_VARIANT`]
//! (`"base"` / `"distill"`) so the runtime + model-card generator can
//! pick the right upstream-anchored provenance without parsing the
//! free-text `vokra.model.name`. Distinct arch tag from every sibling
//! codec (Mimi / DAC / WavTokenizer / xcodec2 / speechtokenizer /
//! funcodec / bicodec / focalcodec) — silently sharing the arch tag
//! would mis-route the runtime dispatch.
//!
//! # HF / licence / category
//!
//! - Upstream HF variants (recorded under
//!   `vokra.provenance.upstream_hf`):
//!   - `neuphonic/neucodec` → [`NeucodecVariant::Base`] (2.35 GB
//!     safetensors).
//!   - `neuphonic/distill-neucodec` → [`NeucodecVariant::Distill`]
//!     (978 MB `pytorch_model.bin` → safetensors offline bridge —
//!     same API + `fsq_codes` shape, ~10× fewer parameters).
//! - SPDX: `apache-2.0` for both variants ([`LicenseClass::Permissive`])
//!   — HF cardData primary source verified 2026-07-28 (base) +
//!   2026-08-01 (distill).
//! - Model category: `codec` (recorded under `vokra.model.category`).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling converters
//! listed above. No convert-time widening; runtime widens BF16 → f32
//! losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice contract). The native runtime accepts this pass-through layout
//! for Distill. The first public base GGUF predates it and uses a normalized
//! decoder namespace; both complete public manifests are pinned separately.
//!
//! # Real-weight parity
//!
//! Real-weight token-to-waveform parity is pinned for both public variants by
//! `tools/parity/neucodec/dump_reference.py`, which restores the GGUF weights
//! into the official `CodecDecoderVocos` module. Waveform-to-code encoding is
//! still explicitly unsupported by the native runtime.
//!
//! # No ONNX (permanent)
//!
//! Neucodec is distributed as safetensors (base) or torch pickle
//! (distill) + a Python pipeline; this converter **never** touches
//! ONNX (FR-LD-05); token-to-waveform decode is re-implemented natively in
//! `crates/vokra-models/src/neucodec/` (whisper.cpp 型 self
//! re-implementation, project design decision 4).
//!
//! # Wiring status
//!
//! [`convert_neucodec_file`] is the M0-era backward-compat entry
//! that always writes [`NeucodecVariant::Base`] (thin wrapper over
//! [`convert_neucodec_variant_file`] — byte-identical output). It is
//! reached from `convert_file_licensed`. [`convert_neucodec_variant_file`]
//! takes an explicit variant and is reached from `convert_file_with_slug`,
//! which picks the variant slug-driven (BigVGan / Focalcodec pattern).
//!
//! Because both entry points are live, this module carries no
//! module-wide dead-code allowance: a blanket one on a wired module
//! would swallow a genuine regression instead of surfacing it at the
//! workspace `-D warnings` gate. Two items are unreachable from
//! non-test code **today** and carry an item-level allowance instead —
//! the `NAME` and `UPSTREAM_HF` backward-compat aliases, superseded by
//! [`NeucodecVariant::name`] and [`NeucodecVariant::upstream_hf`],
//! which every live stamp site now goes through. (`super::focalcodec`
//! holds the same posture for its own two aliases.) Drop each
//! allowance when a caller starts reading that alias — or drop the
//! alias.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Neucodec GGUFs. Shared across every
/// [`NeucodecVariant`] — the base and distill releases have
/// byte-identical arch topology, only the encoder-side parameter
/// counts differ (upstream `neuphonic/distill-neucodec` README
/// primary source: same NeuCodec API + `fsq_codes` shape).
///
/// Intentionally distinct from every sibling codec (`mimi`, `dac`,
/// `wavtokenizer`, `xcodec2`, `funcodec`, `speechtokenizer`,
/// `bicodec`, `xy_tokenizer`, `step_audio2_mini`, `focalcodec`) —
/// NeuCodec is an FSQ codec in the X-Codec 2 lineage with the
/// Neuphonic-specific encoder stack.
pub const ARCH: &str = "neucodec";

/// `vokra.model.name` value written for the canonical
/// `neuphonic/neucodec` GGUF (backward-compat alias — new callers
/// should use [`NeucodecVariant::name`]).
///
/// Unreachable from non-test code today: every stamp site goes through
/// [`NeucodecVariant::name`]. Kept as the documented alias for the
/// canonical release name (mirror of `super::focalcodec::NAME`).
#[allow(dead_code)]
pub(crate) const NAME: &str = "neucodec";

/// `vokra.model.category` value written for every Neucodec GGUF.
/// Distinguishes codec-only models (RVQ / FSQ audio codecs) from
/// vocoder-LM (HiFTChain) or codec-LM (multi-codebook AR) siblings —
/// consumers use it to pick a decode path without inspecting the arch.
pub const CATEGORY: &str = "codec";

/// Default upstream weight licence (SPDX). Verified 2026-07-28 (base)
/// and 2026-08-01 (distill) via HF API cardData `license: apache-2.0`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// `vokra.neucodec.variant`: `"base"` / `"distill"`. Consumers pick a
/// specific NeuCodec release without parsing free-text
/// `vokra.model.name` (mirrors `super::bigvgan` +
/// `super::focalcodec` discriminators).
pub const KEY_NEUCODEC_VARIANT: &str = "vokra.neucodec.variant";

/// Upstream HF repository slug (`org/name`) for the canonical
/// `NeucodecVariant::Base` release (backward-compat alias — new
/// callers should use [`NeucodecVariant::upstream_hf`]).
///
/// Unreachable from non-test code today: the `vokra.provenance.upstream_hf`
/// stamp goes through [`NeucodecVariant::upstream_hf`]. Kept as the
/// documented alias for the canonical release slug (mirror of
/// `super::focalcodec::UPSTREAM_HF`).
#[allow(dead_code)]
pub(crate) const UPSTREAM_HF: &str = "neuphonic/neucodec";

/// Which Neucodec release the caller is converting. Selects the
/// model name / upstream HF slug / variant tag written into the GGUF.
///
/// Both variants share [`ARCH`] `neucodec` — the topology is
/// identical (same NeuCodec architecture + `fsq_codes` output shape
/// per the upstream `neuphonic/distill-neucodec` README primary
/// source), only the encoder-side parameter counts differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeucodecVariant {
    /// `neuphonic/neucodec` (2026-07-28 canonical publish): the
    /// 2.35 GB safetensors release, 24 kHz / 0.8 kbps @ 50 Hz FSQ
    /// codec, X-Codec 2 lineage. `vokra.neucodec.variant = "base"`.
    Base,
    /// `neuphonic/distill-neucodec` (2026-08-01 add): the 978 MB
    /// `pytorch_model.bin` distilled release — same NeuCodec arch,
    /// ~10× fewer parameters (BigCodec acoustic encoder 70M → SQCodec
    /// 36M; w2v-bert-2.0 semantic encoder 600M → DistilHuBERT 21M),
    /// ~7.5× fewer inference MACs. API + `fsq_codes` output shape
    /// identical to base per the upstream README primary source, so
    /// the same converter + tensor-name contract drives it.
    /// `vokra.neucodec.variant = "distill"`.
    Distill,
}

impl NeucodecVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Base => "neucodec",
            Self::Distill => "distill-neucodec",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for this
    /// release — the primary redistribution source the model-card
    /// generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Base => "neuphonic/neucodec",
            Self::Distill => "neuphonic/distill-neucodec",
        }
    }

    /// The `vokra.neucodec.variant` tag written under
    /// [`KEY_NEUCODEC_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Distill => "distill",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s `source`
    /// argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Base => "neuphonic/neucodec (24 kHz, 0.8 kbps @ 50 Hz FSQ codec, apache-2.0)",
            Self::Distill => {
                "neuphonic/distill-neucodec (distilled NeuCodec, ~10x fewer params, \
                 ~7.5x fewer MACs vs base; same NeuCodec arch + `fsq_codes` shape, \
                 apache-2.0)"
            }
        }
    }
}

/// Outcome of a Neucodec conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::focalcodec::FocalcodecReport`,
/// `super::wespeaker::WespeakerReport`) adapted to the
/// file-oriented `convert_neucodec_variant_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NeucodecReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `qwen3_tts` /
    /// `vibevoice` / `voxcpm2` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Mirrors
    /// `qwen3_tts::Qwen3TtsReport::bf16_passthrough` /
    /// `vibevoice::VibeVoiceReport::bf16_passthrough`.
    pub bf16_passthrough: usize,
    /// Which Neucodec variant was written (`None` only in the
    /// pre-variant `Default::default()` slot; every path through
    /// [`convert_neucodec_variant_file`] sets it — mirror of
    /// `super::focalcodec::FocalcodecReport::variant`).
    pub variant: Option<NeucodecVariant>,
}

/// File-based Neucodec converter (backward-compat wrapper — writes
/// [`NeucodecVariant::Base`]).
///
/// Reads `input` (upstream `neuphonic/neucodec` `model.safetensors`),
/// writes a Vokra GGUF to `output`. `license` overrides the default
/// `apache-2.0` provenance stamp (Whisper / kokoro-family override
/// pattern — see `convert_file_licensed` in `lib.rs`); pass `None` to
/// keep the built-in `apache-2.0` stamp.
///
/// This is a thin wrapper over [`convert_neucodec_variant_file`] with
/// [`NeucodecVariant::Base`] — kept for backward compatibility with
/// the 2026-07-28 lib.rs `convert_file` dispatch. New callers that
/// need the 2026-08-01 distill variant should call
/// [`convert_neucodec_variant_file`] directly with
/// [`NeucodecVariant::Distill`] (or route the raw `--model` slug
/// through `convert_file_with_slug`).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_neucodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NeucodecReport, ConvertError> {
    convert_neucodec_variant_file(input, output, license, NeucodecVariant::Base)
}

/// File-based Neucodec converter with explicit variant selection
/// (`vokra-cli convert --model {neucodec|distill-neucodec}` via
/// `convert_file_with_slug`).
///
/// Reads `input` (upstream safetensors — for the distill variant,
/// bridged from `pytorch_model.bin` via
/// `tools/parity/nemo_pt_to_safetensors.py` offline), writes a Vokra
/// GGUF to `output` tagged as the supplied [`NeucodecVariant`].
/// `license` overrides the default `apache-2.0` provenance stamp;
/// pass `None` to keep the built-in stamp.
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// name; the `vokra.model.*` (arch / name / category),
/// `vokra.provenance.*` (weight_license / license / model_id / source
/// / upstream_hf), and `vokra.neucodec.variant` chunks are stamped
/// for the runtime compliance gate (FR-CP-03) and shape-checked
/// config dispatch.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_neucodec_variant_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: NeucodecVariant,
) -> Result<NeucodecReport, ConvertError> {
    // NeuCodec base is 2.35 GB safetensors; distill is 978 MB pickle →
    // ~1 GB safetensors after bridge. Both fit the sibling non-streaming
    // BF16 pass-through posture (still 1-2 orders of magnitude smaller
    // than the streaming-mandated Moshi 14 GiB tier that requires the
    // `MappedTextBlocks` / `restamp_provenance` mmap path).
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    // Category / upstream-HF / variant stamps — not covered by
    // `stamp_provenance` (which handles the SPDX + class + model_id +
    // source group only), so written directly. Consumers pick a decode
    // path by category, trace the artifact back to its serving location
    // by upstream_hf, and pick the shape-checked config bundle by
    // variant tag (`"base"` / `"distill"`).
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_NEUCODEC_VARIANT, variant.tag());

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (both `neuphonic/neucodec` and
    // `neuphonic/distill-neucodec` HF cardData primary source verified
    // 2026-07-28 and 2026-08-01 respectively). `license` overrides for
    // callers who obtained the weight under a different SPDX (see
    // `convert_file_licensed`).
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

    let mut report = NeucodecReport {
        variant: Some(variant),
        ..NeucodecReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as qwen3_tts /
    // vibevoice / voxcpm2; runtime widens BF16 → f32 exactly at load via
    // `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
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
    use vokra_core::gguf::GgufFile;

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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16)
    /// with caller-supplied payloads.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            f32_elems as usize * 4,
            "F32 payload len must match shape × 4"
        );
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(
            f16_bytes.len(),
            f16_elems as usize * 2,
            "F16 payload len must match shape × 2"
        );
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        // Safetensors offsets are relative to the data-region start (after
        // the JSON header) — F32 lives at [0, f32_len), F16 at [f32_len,
        // f32_len + f16_len). The parser tolerates any JSON key order.
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-neucodec-{kind}-{}-{}.bin",
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
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        let input_bytes = safetensors_one_bf16("codec.embed.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_neucodec_file(&input_path, &output_path, None)
            .expect("convert_neucodec_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts / vibevoice / voxcpm2)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );
        // Backward-compat wrapper must tag the report with the default
        // (Base) variant — regression against a silent variant-drop.
        assert_eq!(
            report.variant,
            Some(NeucodecVariant::Base),
            "convert_neucodec_file wrapper must default to Base"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("codec.embed.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "2 rows × 3 cols × 2 B BF16 verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
        // The new variant discriminator chunk must land for Base too —
        // a Base-side silent-drop would be invisible to any consumer
        // that inspects `vokra.neucodec.variant` to pick a decode path.
        assert_eq!(
            file.get(KEY_NEUCODEC_VARIANT).and_then(|v| v.as_str()),
            Some("base"),
            "Base variant must stamp vokra.neucodec.variant = \"base\""
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Non-zero payloads so a silent-widen regression can't hide
        // behind trivial round-trips.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16: pack a couple of exact half-representable values via
        // manual half bit-fiddling (no external crate). 1.0 = 0x3C00,
        // -2.0 = 0xC000, -0.5 = 0xB800, 3.0 = 0x4200, 0.15625 = 0x3100,
        // 42.0 = 0x5140. Six values for a [2,3] tensor = 12 bytes.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12, "6 elements × 2 bytes F16 payload");

        let input_bytes = safetensors_f32_then_f16(
            "codec.dense.weight",
            &[1, 2],
            &f32_bytes,
            "codec.embed.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_neucodec_file(&input_path, &output_path, None)
            .expect("convert_neucodec_file must accept a mixed F32/F16 checkpoint");

        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.variant,
            Some(NeucodecVariant::Base),
            "convert_neucodec_file wrapper must default to Base"
        );

        // Round-trip carries both tensors with their dtypes preserved
        // AND the arch / provenance / category / variant stamps land.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("codec.dense.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("codec.embed.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance / category / variant chunks landed (task-spec pins).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
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
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_NEUCODEC_VARIANT).and_then(|v| v.as_str()),
            Some("base"),
            "Base variant must stamp vokra.neucodec.variant = \"base\""
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn distill_variant_writes_correct_provenance_and_variant_tag() {
        // The Distill variant path must stamp `vokra.model.name` /
        // `vokra.neucodec.variant` / `vokra.provenance.upstream_hf` /
        // `vokra.provenance.model_id` with the distill release's values
        // — a silent-drop back to Base would misroute the runtime
        // dispatch AND publish the artifact under the wrong upstream
        // credit in its model card. Arch tag stays shared per the
        // upstream README primary source (same NeuCodec topology).
        let values: [f32; 4] = [1.0, -1.0, 0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("codec.encoder.weight", &[2, 2], &bf16);
        let input_path = write_temp("distill-in", &input_bytes);
        let output_path = write_temp("distill-out", &[]);

        let report = convert_neucodec_variant_file(
            &input_path,
            &output_path,
            None,
            NeucodecVariant::Distill,
        )
        .expect("convert_neucodec_variant_file must accept a distill BF16 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(
            report.variant,
            Some(NeucodecVariant::Distill),
            "Distill variant must be tagged in the report — regression against silent Base drop"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // Arch tag is shared with the base variant (same NeuCodec
        // topology, ~10x fewer params) — a distinct arch would
        // mis-route the runtime dispatch away from the shared
        // NeuCodec loader.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "Distill must share arch tag `neucodec` with Base (same NeuCodec topology)"
        );
        // But name / upstream / variant tag must distinguish the two so
        // consumers can pick the right shape-checked config bundle and
        // the model-card generator credits the distill release.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("distill-neucodec"),
            "Distill variant must stamp vokra.model.name = \"distill-neucodec\""
        );
        assert_eq!(
            file.get(KEY_NEUCODEC_VARIANT).and_then(|v| v.as_str()),
            Some("distill"),
            "Distill variant must stamp vokra.neucodec.variant = \"distill\""
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("neuphonic/distill-neucodec"),
            "Distill variant must stamp upstream_hf = \"neuphonic/distill-neucodec\""
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "Category stays `codec` (shared across variants)"
        );
        // Licence stamp — Distill is apache-2.0 too (HF cardData primary
        // source verified 2026-08-01).
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

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn base_variant_via_wrapper_matches_variant_call() {
        // `convert_neucodec_file` (backward-compat wrapper) must produce
        // a byte-identical GGUF vs the explicit
        // `convert_neucodec_variant_file(..., Base)` call. Regression
        // against a silent divergence between the two entry points that
        // would drift consumers who depend on either path.
        let values: [f32; 2] = [1.0, -1.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();

        let in_a = safetensors_one_bf16("codec.a.weight", &[1, 2], &bf16);
        // Identical input bytes for both calls; separate paths keep the
        // parallel-`cargo test` name-collision guard in `write_temp`
        // effective.
        let in_b = safetensors_one_bf16("codec.a.weight", &[1, 2], &bf16);
        let path_a_in = write_temp("wrapper-a-in", &in_a);
        let path_b_in = write_temp("wrapper-b-in", &in_b);
        let path_a_out = write_temp("wrapper-a-out", &[]);
        let path_b_out = write_temp("wrapper-b-out", &[]);

        convert_neucodec_file(&path_a_in, &path_a_out, None)
            .expect("backward-compat wrapper must succeed");
        convert_neucodec_variant_file(&path_b_in, &path_b_out, None, NeucodecVariant::Base)
            .expect("explicit-Base call must succeed");

        let bytes_a = std::fs::read(&path_a_out).expect("read wrapper output");
        let bytes_b = std::fs::read(&path_b_out).expect("read explicit-Base output");
        assert_eq!(
            bytes_a, bytes_b,
            "wrapper and explicit-Base entry points must be byte-identical"
        );

        for p in [&path_a_in, &path_b_in, &path_a_out, &path_b_out] {
            std::fs::remove_file(p).ok();
        }
    }
}
